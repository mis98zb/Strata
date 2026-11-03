#include "prefill.hpp"
#include "cpu.hpp"
#include "expert_compute.hpp"
#include "state_transfer.hpp"
#include "weights.hpp"
#include "batch_gemm.hpp"

#include <algorithm>
#include <map>

namespace strata::glm5 {
namespace {

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "glm5 full-layer stream"); }
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
};

struct Workspace {
    MemoryBudget& budget;
    std::vector<std::unique_ptr<Buffer>> storage;
    GlmBuffers single, group;
    GlmChunkBuffers chunk;
    MoEBuffers moe;
    BlockBuffers block;
    float *hidden = nullptr, *logits = nullptr;
    ExpertComputer experts;
    std::unique_ptr<BatchGemm> gemm;
    void* alloc(uint64_t bytes) {
        auto b = std::make_unique<Buffer>(bytes, &budget);
        auto* p = b->get(); storage.push_back(std::move(b)); return p;
    }
    Workspace(const Geometry& g, int batch, int64_t cells, MemoryBudget& limit, void* stream)
        : budget(limit), experts(g, &limit, batch) {
        const auto one = glm_buffers_bytes(g, 1, cells);
        glm_buffers_init(g, 1, cells, alloc(one), single);
        const auto many = glm_buffers_bytes(g, batch, cells);
        glm_buffers_init(g, batch, cells, alloc(many), group);
        glm_chunk_init(g, g.n_expert_used, batch, alloc(glm_chunk_bytes(g, g.n_expert_used, batch)), chunk);
        moe.logits = chunk.logits; moe.ids = chunk.ids; moe.weights = chunk.weights; moe.shared = chunk.shared;
        block.R = chunk.R; block.block_out = chunk.block_out;
        hidden = static_cast<float*>(alloc(batch * g.n_embd * sizeof(float)));
        logits = static_cast<float*>(alloc(g.n_vocab * sizeof(float)));
        if (batch > 8) gemm = std::make_unique<BatchGemm>(g, batch, stream, &budget);
    }
    void route(const Geometry& g, const WeightBank& bank, int layer, const float* input, const int32_t* ids,
               int count, float* output, cudaStream_t stream) {
        std::vector<int32_t> host_ids(static_cast<size_t>(count) * g.n_expert_used);
        check(cudaMemcpyAsync(host_ids.data(), ids, host_ids.size() * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream), "glm5 full-layer routing");
        check(cudaStreamSynchronize(stream), "glm5 full-layer routing sync");
        std::map<int, std::vector<ExpertMiss>> groups;
        for (size_t i = 0; i < host_ids.size(); ++i)
            groups[host_ids[i]].push_back({static_cast<int>(i / g.n_expert_used), static_cast<int>(i), host_ids[i]});
        for (const auto& [expert, entries] : groups) experts.run(bank.expert(layer, expert), input, entries, output, stream);
    }
};

uint64_t host_bytes(const ModelFile& file, const Options& o, size_t count,
                    int64_t previous_position, uint64_t previous_bytes) {
    const auto& g = file.geometry();
    const auto total = checked_add(previous_position, count);
    uint64_t bytes = checked_add(previous_bytes, checked_mul(count, g.hc_dim() * sizeof(float)));
    bytes = checked_add(bytes, host_state_bytes(g, total, o.mtp > 0));
    if (o.mtp) {
        bytes = checked_add(bytes, checked_mul(count, g.n_embd * sizeof(float)));
        bytes = checked_add(bytes, checked_mul(2 * o.prefill, g.n_embd * sizeof(float)));
    }
    bytes = checked_add(bytes, g.n_embd * sizeof(float));
    bytes = checked_add(bytes, checked_mul(o.prefill, g.n_expert_used * (sizeof(int32_t) + 2 * sizeof(ExpertMiss))));
    uint64_t conversion = 0;
    for (size_t s = 0; s < file.gguf().size(); ++s)
        for (const auto& t : file.gguf().shard(s).tensors())
            if ((t.type == GGML_TYPE_F16 || t.type == GGML_TYPE_BF16) &&
                decode_tensor(t, g.n_layers + (o.mtp ? 1 : 0)))
                conversion = std::max(conversion, checked_mul(t.elements(), sizeof(float)));
    return checked_add(bytes, conversion);
}

void embed(const ModelFile& file, int32_t id, bool mtp, std::vector<float>& row) {
    const auto& g = file.geometry();
    if (id < 0 || id >= g.n_vocab) throw std::invalid_argument("glm5 full-layer: token outside vocabulary");
    const auto optional = "blk." + std::to_string(g.n_layers) + ".nextn.embed_tokens.weight";
    const auto& t = file.tensor(mtp && file.gguf().find(optional) ? optional : "token_embd.weight");
    const auto bytes = tensor_payload_bytes(t) / g.n_vocab;
    dequantize(t, file.data(t.name) + static_cast<uint64_t>(id) * bytes, row.data(), g.n_embd);
}

}  // namespace

uint64_t full_layer_host_bytes(const ModelFile& file, const Options& o, size_t count,
                               int64_t position, uint64_t previous_bytes) {
    return host_bytes(file, o, count, position, previous_bytes);
}

PrefillResult full_layer_prefill(const ModelFile& file, const Options& o,
                                 const std::vector<int32_t>& tokens, const HostState& initial) {
    init_cpu();
    const auto& g = file.geometry();
    initial.validate(g, o.max_context, o.mtp > 0);
    if (tokens.empty() || tokens.size() > static_cast<uint64_t>(o.max_context - initial.position))
        throw std::invalid_argument("glm5 full-layer: empty prompt or context exhausted");
    const int64_t cells = initial.position + tokens.size();
    PrefillResult result;
    result.stats.host_tensor_bytes = full_layer_host_bytes(file, o, tokens.size(), initial.position, initial.bytes());
    if (result.stats.host_tensor_bytes > checked_mul(o.prefill_host_mib, 1048576))
        throw std::runtime_error("glm5 full-layer host tensor budget exceeded");
    std::vector<float> activations(checked_mul(tokens.size(), g.hc_dim())), row(g.n_embd);
    std::vector<float> target_hidden(o.mtp ? checked_mul(tokens.size(), g.n_embd) : 0);
    for (size_t t = 0; t < tokens.size(); ++t) {
        embed(file, tokens[t], false, row);
        for (int64_t h = 0; h < g.hc; ++h)
            std::copy(row.begin(), row.end(), activations.begin() + t * g.hc_dim() + h * g.n_embd);
    }
    check(cudaSetDevice(o.gpu), "glm5 full-layer device");
    Stream stream;
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total), "glm5 full-layer available VRAM");
    const auto reserve = checked_mul(o.prefill_reserve_mib, 1048576);
    const auto available = free > reserve ? free - reserve : 0;
    const auto cap = o.prefill_gpu_mib ? checked_mul(o.prefill_gpu_mib, 1048576) : available;
    if (!cap || cap > available) throw std::runtime_error("glm5 full-layer GPU budget leaves insufficient reserve");
    MemoryBudget budget(cap);
    Workspace work(g, o.prefill, cells, budget, stream.value);
    BatchScope batch_scope(work.gemm.get());
    const auto workspace_bytes = budget.used();
    result.state.position = cells;
    result.state.layers.resize(g.n_layers);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        {
            const auto prefix = "blk." + std::to_string(l) + ".";
            WeightBank bank(file, [&](const TensorInfo& t) {
                return t.name.starts_with(prefix) && t.name.find("indexer") == std::string::npos;
            }, &budget);
            ++result.stats.layer_loads;
            if (!g.is_dense_ffn_layer(l)) result.stats.experts_loaded += g.n_expert;
            const auto bytes = glm_layer_state_bytes(g, cells, l);
            Buffer device_state(bytes, &budget);
            check(cudaMemsetAsync(device_state.get(), 0, bytes, stream.value), "glm5 full-layer initial state");
            GlmLayerState state;
            if (glm_state_init(g, cells, l, device_state.get(), state) != bytes)
                throw std::runtime_error("glm5 full-layer state layout mismatch");
            if (!initial.layers.empty()) restore_layer(g, initial.layers[l], state, stream.value);
            for (size_t first = 0; first < tokens.size(); first += o.prefill) {
                const int n = static_cast<int>(std::min<size_t>(o.prefill, tokens.size() - first));
                check(cudaMemcpyAsync(work.chunk.R, activations.data() + first * g.hc_dim(),
                                       n * g.hc_dim() * sizeof(float), cudaMemcpyHostToDevice, stream.value),
                      "glm5 full-layer activation upload");
                std::string err;
                {
                    GlmBuffers b; MoEBuffers m; BlockBuffers bb;
                    glm_chunk_group_view(work.chunk, work.group, work.moe, work.block, 0, n, b, m, bb);
                    if (!glm_block_layer_pre(bank.table(), g, l, initial.position + first, 0,
                                             b, state, m, g.n_expert_used, bb, stream.value, err, nullptr))
                        throw std::runtime_error(err);
                }
                if (!g.is_dense_ffn_layer(l))
                    work.route(g, bank, l, work.chunk.cur, work.chunk.ids, n, work.chunk.parts, stream.value);
                {
                    GlmBuffers b; MoEBuffers m; BlockBuffers bb;
                    glm_chunk_group_view(work.chunk, work.group, work.moe, work.block, 0, n, b, m, bb);
                    if (!glm_block_layer_post(bank.table(), g, l, g.n_expert_used, b, m, bb,
                                              work.chunk.parts, stream.value, err))
                        throw std::runtime_error(err);
                }
                check(cudaMemcpyAsync(activations.data() + first * g.hc_dim(), work.chunk.R,
                                       n * g.hc_dim() * sizeof(float), cudaMemcpyDeviceToHost, stream.value),
                      "glm5 full-layer activation download");
                check(cudaStreamSynchronize(stream.value), "glm5 full-layer microbatch sync");
            }
            result.state.layers[l] = capture_layer(g, state, cells, glm_is_kda_layer(g, l), stream.value);
        }
        if (budget.used() != workspace_bytes) throw std::runtime_error("glm5 full-layer: bank was not released");
    }
    {
        WeightBank head(file, [](const TensorInfo& t) { return t.name == "output_norm.weight" || t.name == "output.weight"; },
                        &budget);
        for (size_t first = 0; first < tokens.size(); first += o.prefill) {
            const auto count = std::min<size_t>(o.prefill, tokens.size() - first);
            if (!o.mtp && first + count != tokens.size()) continue;
            const size_t begin = o.mtp ? first : tokens.size() - 1;
            const size_t n = o.mtp ? count : 1;
            check(cudaMemcpyAsync(work.chunk.R, activations.data() + begin * g.hc_dim(), n * g.hc_dim() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream.value), "glm5 full-layer head input");
            std::string err;
            if (!glm_head_mix(head.table(), g, work.block, work.hidden, stream.value, err, n)) throw std::runtime_error(err);
            if (o.mtp)
                check(cudaMemcpyAsync(target_hidden.data() + first * g.n_embd, work.hidden, n * g.n_embd * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream.value), "glm5 full-layer hidden output");
            if (first + count == tokens.size()) {
                result.state.logits.resize(g.n_vocab);
                project_head(head.table(), g, work.hidden + (n - 1) * g.n_embd, work.logits, false, stream.value);
                check(cudaMemcpyAsync(result.state.logits.data(), work.logits, g.n_vocab * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream.value), "glm5 full-layer logits");
            }
            check(cudaStreamSynchronize(stream.value), "glm5 full-layer head sync");
        }
    }
    if (o.mtp) {
        const auto prefix = "blk." + std::to_string(g.n_layers) + ".";
        WeightBank bank(file, [&](const TensorInfo& t) {
            return t.name == "output_norm.weight" || (t.name.starts_with(prefix) &&
                !t.name.ends_with(".nextn.embed_tokens.weight") && t.name.find("indexer") == std::string::npos);
        }, &budget);
        ++result.stats.layer_loads;
        result.stats.experts_loaded += g.n_expert;
        const auto bytes = glm_mla_state_bytes(g, cells);
        Buffer device_state(bytes, &budget);
        check(cudaMemsetAsync(device_state.get(), 0, bytes, stream.value), "glm5 full-layer MTP state");
        MtpBuffers mtp;
        glm_mla_state_init(g, cells, device_state.get(), mtp.attention);
        if (!initial.layers.empty()) restore_layer(g, initial.mtp, mtp.attention, stream.value);
        const auto batch = static_cast<int64_t>(o.prefill);
        mtp.capacity = batch;
        Buffer scratch(6 * batch * g.n_embd * sizeof(float), &budget);
        auto* p = static_cast<float*>(scratch.get());
        mtp.embedding = p; mtp.hidden = p + batch * g.n_embd; mtp.concat = p + 2 * batch * g.n_embd;
        mtp.residual = p + 4 * batch * g.n_embd; mtp.output = p + 5 * batch * g.n_embd;
        std::vector<float> previous(batch * g.n_embd), embeddings(batch * g.n_embd);
        for (size_t first = 0; first < tokens.size(); first += batch) {
            const auto n = std::min<size_t>(batch, tokens.size() - first);
            for (size_t t = 0; t < n; ++t) {
                embed(file, tokens[first + t], true, row);
                std::copy(row.begin(), row.end(), embeddings.begin() + t * g.n_embd);
                const auto position = first + t;
                const float* prior = position ? target_hidden.data() + (position - 1) * g.n_embd :
                    (initial.carry_hidden.empty() ? nullptr : initial.carry_hidden.data());
                if (prior) std::copy_n(prior, g.n_embd, previous.data() + t * g.n_embd);
                else std::fill_n(previous.data() + t * g.n_embd, g.n_embd, 0.0f);
            }
            check(cudaMemcpyAsync(mtp.embedding, embeddings.data(), n * g.n_embd * sizeof(float), cudaMemcpyHostToDevice,
                                   stream.value), "glm5 full-layer MTP embedding");
            check(cudaMemcpyAsync(work.hidden, previous.data(), n * g.n_embd * sizeof(float), cudaMemcpyHostToDevice,
                                   stream.value), "glm5 full-layer MTP previous hidden");
            auto b = work.group;
            b.ntok = n;
            std::string err;
            if (!glm_mtp_pre(bank.table(), g, initial.position + first, work.hidden, b, work.moe, mtp,
                             stream.value, err)) throw std::runtime_error(err);
            work.route(g, bank, g.n_layers, b.cur, work.chunk.ids, n, work.chunk.parts, stream.value);
            if (!glm_mtp_post(bank.table(), g, b, work.moe, work.chunk.parts, mtp, stream.value, err))
                throw std::runtime_error(err);
            check(cudaStreamSynchronize(stream.value), "glm5 full-layer MTP catch-up sync");
        }
        result.state.mtp = capture_layer(g, mtp.attention, cells, false, stream.value);
        result.state.carry_hidden.assign(target_hidden.end() - g.n_embd, target_hidden.end());
    }
    result.stats.gpu_peak_bytes = budget.peak();
    if (work.gemm) {
        const auto& stats = work.gemm->stats();
        result.stats.max_matrix_columns = stats.max_columns;
        result.stats.mmq_calls = stats.mmq_calls;
        result.stats.f16_gemm_calls = stats.f16_calls;
        result.stats.f32_gemm_calls = stats.f32_calls;
    }
    result.state.validate(g, o.max_context, o.mtp > 0);
    return result;
}

}  // namespace strata::glm5
