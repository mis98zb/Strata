#include "prefill.hpp"
#include "cpu.hpp"
#include "expert_compute.hpp"
#include "state_transfer.hpp"
#include "weights.hpp"
#include "batch_gemm.hpp"
#include "attention_scope.hpp"
#include "prefill_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <span>

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
    std::vector<uint32_t>* heat = nullptr;
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
        for (size_t i = 0; i < host_ids.size(); ++i) {
            if (host_ids[i] < 0 || host_ids[i] >= g.n_expert)
                throw std::runtime_error("glm5 full-layer: invalid routed expert");
            if (heat) {
                auto& value = (*heat)[layer * g.n_expert + host_ids[i]];
                if (value != UINT32_MAX) ++value;
            }
            groups[host_ids[i]].push_back({static_cast<int>(i / g.n_expert_used), static_cast<int>(i), host_ids[i]});
        }
        for (const auto& [expert, entries] : groups) experts.run(bank.expert(layer, expert), input, entries, output, stream);
    }
};

uint64_t host_bytes(const ModelFile& file, const Options& o, size_t count,
                    int64_t previous_position, uint64_t previous_bytes) {
    const auto& g = file.geometry();
    if (o.dsa_tail >= 0 && o.dsa_tail != g.idx_select_tail)
        throw std::invalid_argument("glm5: ModelFile and options have different DSA tail policies");
    const auto total = checked_add(previous_position, count);
    uint64_t bytes = checked_add(previous_bytes, checked_mul(count, g.hc_dim() * sizeof(float)));
    bytes = checked_add(bytes, host_state_bytes(g, total, o.mtp > 0, o.gpu_warm, o.dsa));
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
                decode_tensor(t, g.n_layers + (o.mtp ? 1 : 0), o.dsa))
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

WeightBank::Select layer_select(int64_t layer, bool dsa) {
    return [prefix = "blk." + std::to_string(layer) + ".", dsa](const TensorInfo& t) {
        return t.name.starts_with(prefix) && (dsa || t.name.find("indexer") == std::string::npos);
    };
}

bool head_select(const TensorInfo& t) {
    return t.name == "output_norm.weight" || t.name == "output.weight";
}

WeightBank::Select mtp_select(const Geometry& g, bool dsa) {
    return [prefix = "blk." + std::to_string(g.n_layers) + ".", dsa](const TensorInfo& t) {
        return t.name == "output_norm.weight" || (t.name.starts_with(prefix) &&
            !t.name.ends_with(".nextn.embed_tokens.weight") &&
            (dsa || t.name.find("indexer") == std::string::npos));
    };
}

struct LayerGroup {
    int first = 0, count = 0, prefetch = 0;
    uint64_t bytes = 0;
    bool fused_head = false;
};

constexpr uint64_t weight_staging_bytes = 1048576;

std::vector<LayerGroup> plan_groups(const ModelFile& file, const Options& o, int64_t cells,
                                    uint64_t workspace, uint64_t cap, uint64_t host,
                                    const std::vector<uint64_t>& weights,
                                    const std::vector<uint64_t>& conversions) {
    const auto& g = file.geometry();
    const uint64_t boundary = checked_mul(o.prefill, g.hc_dim() * sizeof(float));
    const uint64_t device_transfer = o.prefill_pipeline ? checked_mul(2, boundary) : 0;
    const uint64_t transfer = device_transfer;
    const uint64_t transfer_cap = checked_mul(o.prefill_transfer_mib, 1048576);
    const uint64_t host_cap = checked_mul(o.prefill_host_mib, 1048576);
    if (transfer > transfer_cap || host > host_cap)
        throw BudgetExceeded("glm5 full-layer host/transfer budget cannot fit activation pipeline");
    const auto fixed = checked_add(workspace, device_transfer);
    const auto fits = [&](uint64_t bytes) { return fixed <= cap && bytes <= cap - fixed; };
    if (!fits(WeightBank::allocation_bytes(file, head_select)))
        throw BudgetExceeded("glm5 full-layer GPU budget cannot fit output head");
    if (o.mtp) {
        const auto mtp = checked_add(WeightBank::allocation_bytes(file, mtp_select(g, o.dsa)),
            checked_add(glm_mla_state_bytes(g, cells), checked_mul(6 * o.prefill, g.n_embd * sizeof(float))));
        if (!fits(mtp)) throw BudgetExceeded("glm5 full-layer GPU budget cannot fit NextN bank");
    }
    std::vector<LayerGroup> groups;
    const int group_limit = o.prefill_group_layers ? o.prefill_group_layers : (o.prefill_auto ? 4 : 1);
    const auto prefetch_count = [&](int first, uint64_t group_bytes) {
        int count = 0;
        uint64_t extra = 0, staging = 0, conversion = 0;
        while (count < o.prefill_prefetch_layers && first + count < g.n_layers) {
            const auto l = first + count;
            const auto w = checked_add(extra, weights[l]);
            const auto s = checked_add(staging, weight_staging_bytes);
            const auto c = checked_add(conversion, conversions[l]);
            if (!fits(checked_add(group_bytes, w)) || checked_add(transfer, s) > transfer_cap ||
                checked_add(host, checked_add(s, c)) > host_cap) break;
            extra = w; staging = s; conversion = c;
            ++count;
        }
        return count;
    };
    for (int first = 0; first < g.n_layers;) {
        LayerGroup group;
        group.first = first;
        const auto single_bytes = checked_add(weights[first], glm_layer_state_bytes(g, cells, first));
        if (!fits(single_bytes))
            throw BudgetExceeded("glm5 full-layer GPU budget cannot fit a complete layer");
        const int preferred_prefetch = prefetch_count(first + 1, single_bytes);
        while (group.count < group_limit && first + group.count < g.n_layers) {
            const auto l = first + group.count;
            const auto next = checked_add(group.bytes, checked_add(weights[l], glm_layer_state_bytes(g, cells, l)));
            if (!fits(next)) {
                if (!o.prefill_auto || !group.count)
                    throw BudgetExceeded("glm5 full-layer GPU budget cannot fit requested layer group");
                break;
            }
            if (o.prefill_auto && group.count) {
                const int next_first = l + 1;
                const int required = std::min<int64_t>(preferred_prefetch, g.n_layers - next_first);
                if (prefetch_count(next_first, next) < required) break;
            }
            group.bytes = next;
            ++group.count;
        }
        first += group.count;
        groups.push_back(group);
    }
    for (size_t i = 0; i + 1 < groups.size(); ++i) {
        auto& current = groups[i];
        const auto& next = groups[i + 1];
        current.prefetch = std::min(prefetch_count(next.first, current.bytes), next.count);
    }
    if (o.prefill_auto || o.prefill_group_layers || o.prefill_pipeline || o.prefill_prefetch_layers)
        groups.back().fused_head = fits(checked_add(groups.back().bytes, WeightBank::allocation_bytes(file, head_select)));
    return groups;
}

}  // namespace

uint64_t full_layer_host_bytes(const ModelFile& file, const Options& o, size_t count,
                               int64_t position, uint64_t previous_bytes) {
    return host_bytes(file, o, count, position, previous_bytes);
}

PrefillResult full_layer_prefill(const ModelFile& file, const Options& requested,
                                 const std::vector<int32_t>& tokens, const HostState& initial) {
    Options o = resolve_options(requested);
    AttentionScope attention(o);
    if (o.flash_attn && o.mtp && !o.dsa)
        throw std::invalid_argument("glm5: Flash MTP currently requires --dsa");
    if (o.dsa) file.validate_dsa(o.mtp > 0);
    init_cpu();
    const auto& g = file.geometry();
    initial.validate(g, o.max_context, o.mtp > 0, o.dsa);
    if (tokens.empty() || tokens.size() > static_cast<uint64_t>(o.max_context - initial.position))
        throw std::invalid_argument("glm5 full-layer: empty prompt or context exhausted");
    const int64_t cells = initial.position + tokens.size();
    PrefillResult result;
    check(cudaSetDevice(o.gpu), "glm5 full-layer device");
    if (o.flash_attn && !kernels::glm_flash_supported())
        throw std::runtime_error("glm5 Flash Attention: unsupported CUDA device");
    Stream stream;
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total), "glm5 full-layer available VRAM");
    const auto reserve = checked_mul(o.prefill_reserve_mib, 1048576);
    const auto available = free > reserve ? free - reserve : 0;
    const auto cap = o.prefill_gpu_mib ? checked_mul(o.prefill_gpu_mib, 1048576) : available;
    if (!cap || cap > available) throw std::runtime_error("glm5 full-layer GPU budget leaves insufficient reserve");
    MemoryBudget budget(cap);
    const auto planning_start = std::chrono::steady_clock::now();
    std::vector<uint64_t> weights(g.n_layers), conversions(g.n_layers);
    for (int l = 0; l < g.n_layers; ++l) {
        const auto select = layer_select(l, o.dsa);
        weights[l] = WeightBank::allocation_bytes(file, select);
        for (size_t s = 0; s < file.gguf().size(); ++s)
            for (const auto& t : file.gguf().shard(s).tensors())
                if (select(t) && (t.type == GGML_TYPE_F16 || t.type == GGML_TYPE_BF16))
                    conversions[l] = std::max(conversions[l], checked_mul(t.elements(), sizeof(float)));
    }
    if (o.prefill_auto) o.prefill = std::min<size_t>(o.prefill, tokens.size());
    std::unique_ptr<Workspace> workspace;
    std::vector<LayerGroup> groups;
    while (true) {
        ++result.stats.planning_attempts;
        try {
            const auto host = full_layer_host_bytes(file, o, tokens.size(), initial.position, initial.bytes());
            if (host > checked_mul(o.prefill_host_mib, 1048576))
                throw BudgetExceeded("glm5 full-layer host tensor budget exceeded");
            workspace = std::make_unique<Workspace>(g, o.prefill, cells, budget, stream.value);
            groups = plan_groups(file, o, cells, budget.used(), cap, host, weights, conversions);
            result.stats.host_tensor_bytes = host;
            break;
        } catch (const BudgetExceeded&) {
            workspace.reset();
            if (!o.prefill_auto || o.prefill == 1) throw;
            o.prefill = std::max(1, o.prefill / 2);
        }
    }
    auto& work = *workspace;
    result.stats.selected_batch = o.prefill;
    result.stats.groups = groups.size();
    for (const auto& group : groups)
        result.stats.plan.push_back({group.first, group.count, group.prefetch});
    result.stats.planning_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - planning_start).count();
    BatchScope batch_scope(work.gemm.get());
    const auto boundary = checked_mul(o.prefill, g.hc_dim() * sizeof(float));
    std::unique_ptr<PinnedBuffer> pinned_activations;
    std::unique_ptr<ActivationPipeline> pipeline;
    if (o.prefill_pipeline) {
        pipeline = std::make_unique<ActivationPipeline>(boundary, budget);
        result.stats.transfer_peak_bytes = 2 * boundary;
    }
    const auto activation_elements = checked_mul(tokens.size(), g.hc_dim());
    std::vector<float> pageable_activations;
    if (pipeline) pinned_activations = std::make_unique<PinnedBuffer>(checked_mul(activation_elements, sizeof(float)));
    else pageable_activations.resize(activation_elements);
    std::span<float> activations(pipeline ? static_cast<float*>(pinned_activations->get()) : pageable_activations.data(),
                                 activation_elements);
    std::vector<float> row(g.n_embd);
    std::vector<float> target_hidden(o.mtp ? checked_mul(tokens.size(), g.n_embd) : 0);
    for (size_t t = 0; t < tokens.size(); ++t) {
        embed(file, tokens[t], false, row);
        for (int64_t h = 0; h < g.hc; ++h)
            std::copy(row.begin(), row.end(), activations.begin() + t * g.hc_dim() + h * g.n_embd);
    }
    const auto workspace_bytes = budget.used();
    result.state.position = cells;
    result.state.dsa = o.dsa;
    result.state.dsa_tail = o.dsa ? g.idx_select_tail : -1;
    result.state.layers.resize(g.n_layers);
    if (o.gpu_warm) {
        result.state.expert_heat = initial.expert_heat;
        result.state.expert_heat.resize((g.n_layers + (o.mtp ? 1 : 0)) * g.n_expert, 0);
        work.heat = &result.state.expert_heat;
    }
    std::vector<std::unique_ptr<WeightBank>> prefetched;
    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        {
            const auto& group = groups[group_index];
            result.stats.max_group_layers = std::max<uint64_t>(result.stats.max_group_layers, group.count);
            auto banks = std::move(prefetched);
            prefetched.clear();
            for (auto& bank : banks) bank->wait_ready();
            for (int i = banks.size(); i < group.count; ++i)
                banks.push_back(std::make_unique<WeightBank>(file, layer_select(group.first + i, o.dsa), &budget));
            std::unique_ptr<WeightBank> final_head;
            if (group.fused_head) {
                final_head = std::make_unique<WeightBank>(file, head_select, &budget);
                result.stats.weight_upload_bytes += final_head->uploaded_bytes();
                result.stats.fused_head = true;
            }
            std::vector<std::unique_ptr<Buffer>> state_storage;
            std::vector<GlmLayerState> states(group.count);
            for (int i = 0; i < group.count; ++i) {
                const int l = group.first + i;
                ++result.stats.layer_loads;
                result.stats.weight_upload_bytes += banks[i]->uploaded_bytes();
                if (!g.is_dense_ffn_layer(l)) result.stats.experts_loaded += g.n_expert;
                const auto bytes = glm_layer_state_bytes(g, cells, l);
                state_storage.push_back(std::make_unique<Buffer>(bytes, &budget));
                check(cudaMemsetAsync(state_storage.back()->get(), 0, bytes, stream.value), "glm5 full-layer initial state");
                if (glm_state_init(g, cells, l, state_storage.back()->get(), states[i]) != bytes)
                    throw std::runtime_error("glm5 full-layer state layout mismatch");
                if (!initial.layers.empty()) restore_layer(g, initial.layers[l], states[i], stream.value);
            }
            if (group.prefetch) {
                uint64_t host_extra = group.prefetch * weight_staging_bytes;
                for (int i = 0; i < group.prefetch; ++i) {
                    const auto l = groups[group_index + 1].first + i;
                    prefetched.push_back(std::make_unique<WeightBank>(
                        file, layer_select(l, o.dsa), &budget, weight_staging_bytes, true));
                    host_extra = checked_add(host_extra, conversions[l]);
                    ++result.stats.prefetched_layers;
                }
                result.stats.host_tensor_bytes = std::max(result.stats.host_tensor_bytes,
                    checked_add(full_layer_host_bytes(file, o, tokens.size(), initial.position, initial.bytes()),
                                host_extra));
                result.stats.transfer_peak_bytes = std::max(result.stats.transfer_peak_bytes,
                    (pipeline ? 2 * boundary : 0) + group.prefetch * weight_staging_bytes);
                for (auto& bank : prefetched) bank->start_upload();
            }
            if (pipeline)
                pipeline->prepare(0, activations.data(), std::min<size_t>(o.prefill, tokens.size()) *
                                                           g.hc_dim() * sizeof(float));
            size_t batch_index = 0;
            for (size_t first = 0; first < tokens.size(); first += o.prefill, ++batch_index) {
                const int n = static_cast<int>(std::min<size_t>(o.prefill, tokens.size() - first));
                const auto bytes = n * g.hc_dim() * sizeof(float);
                const int slot = batch_index % 2;
                if (pipeline) {
                    pipeline->input(slot, work.chunk.R, stream.value);
                    if (first + n < tokens.size()) {
                        const auto next = std::min<size_t>(o.prefill, tokens.size() - first - n);
                        pipeline->prepare(1 - slot, activations.data() + (first + n) * g.hc_dim(),
                                          next * g.hc_dim() * sizeof(float));
                    }
                } else {
                    check(cudaMemcpyAsync(work.chunk.R, activations.data() + first * g.hc_dim(),
                                           bytes, cudaMemcpyHostToDevice, stream.value),
                          "glm5 full-layer activation upload");
                }
                result.stats.activation_upload_bytes += bytes;
                for (int i = 0; i < group.count; ++i) {
                    const int l = group.first + i;
                    const auto& bank = *banks[i];
                    std::string err;
                    {
                        GlmBuffers b; MoEBuffers m; BlockBuffers bb;
                        glm_chunk_group_view(work.chunk, work.group, work.moe, work.block, 0, n, b, m, bb);
                        if (!glm_block_layer_pre(bank.table(), g, l, initial.position + first, 0,
                                                 b, states[i], m, g.n_expert_used, bb, stream.value, err, nullptr))
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
                }
                if (final_head) {
                    if (o.mtp || first + n == tokens.size()) {
                        auto head_block = work.block;
                        if (!o.mtp) head_block.R += (n - 1) * g.hc_dim();
                        const int count = o.mtp ? n : 1;
                        std::string err;
                        if (!glm_head_mix(final_head->table(), g, head_block, work.hidden, stream.value, err, count))
                            throw std::runtime_error(err);
                        if (o.mtp)
                            check(cudaMemcpyAsync(target_hidden.data() + first * g.n_embd, work.hidden,
                                                   count * g.n_embd * sizeof(float), cudaMemcpyDeviceToHost, stream.value),
                                  "glm5 fused head hidden output");
                        if (first + n == tokens.size()) {
                            result.state.logits.resize(g.n_vocab);
                            project_head(final_head->table(), g, work.hidden + (count - 1) * g.n_embd,
                                         work.logits, false, stream.value);
                            check(cudaMemcpyAsync(result.state.logits.data(), work.logits,
                                                   g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, stream.value),
                                  "glm5 fused head logits");
                        }
                    }
                    check(cudaStreamSynchronize(stream.value), "glm5 fused head microbatch sync");
                } else if (pipeline) {
                    pipeline->output(slot, work.chunk.R, activations.data() + first * g.hc_dim(), stream.value);
                    result.stats.activation_download_bytes += bytes;
                } else {
                    check(cudaMemcpyAsync(activations.data() + first * g.hc_dim(), work.chunk.R,
                                           bytes, cudaMemcpyDeviceToHost, stream.value),
                          "glm5 full-layer activation download");
                    check(cudaStreamSynchronize(stream.value), "glm5 full-layer microbatch sync");
                    result.stats.activation_download_bytes += bytes;
                }
            }
            if (pipeline) pipeline->finish();
            for (int i = 0; i < group.count; ++i) {
                const auto l = group.first + i;
                result.state.layers[l] = capture_layer(g, states[i], cells, glm_is_kda_layer(g, l), stream.value);
            }
        }
        uint64_t retained = workspace_bytes;
        for (const auto& bank : prefetched) retained += bank->bytes();
        if (budget.used() != retained) throw std::runtime_error("glm5 full-layer: bank was not released");
    }
    if (!result.stats.fused_head) {
        WeightBank head(file, head_select, &budget);
        result.stats.weight_upload_bytes += head.uploaded_bytes();
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
        WeightBank bank(file, mtp_select(g, o.dsa), &budget);
        result.stats.weight_upload_bytes += bank.uploaded_bytes();
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
    result.state.validate(g, o.max_context, o.mtp > 0, o.dsa);
    return result;
}

}  // namespace strata::glm5
