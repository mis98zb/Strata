#include "strata/glm5/model.hpp"
#include "strata/glm5/layer.hpp"
#include "strata/glm5/model_file.hpp"
#include "gemv_util.hpp"
#include "gpu_support.hpp"
#include "strata/glm5/speculative.hpp"

#include "ggml.h"
#include "ggml-cpu.h"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace strata::glm5 {
namespace {

void init_cpu() {
    // CPU dots and dequantizers need ggml's tables even though this engine does not build a ggml graph.
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

const ggml_type_traits_cpu& cpu_traits(uint32_t type) {
    if (type >= GGML_TYPE_COUNT) throw std::runtime_error("glm5: unsupported CPU GGUF type " + std::to_string(type));
    const auto* traits = ggml_get_type_traits_cpu(static_cast<ggml_type>(type));
    if (!traits || !traits->vec_dot)
        throw std::runtime_error("glm5: no CPU dot product for GGUF type " + std::to_string(type));
    return *traits;
}

std::vector<uint8_t> quantize(const float* x, int64_t width, uint32_t type) {
    const auto act = cpu_traits(type).vec_dot_type;
    const auto* tr = ggml_get_type_traits_cpu(act);
    if (!tr || !tr->from_float || width % ggml_blck_size(act))
        throw std::runtime_error("glm5: unsupported expert activation quantization");
    std::vector<uint8_t> out(ggml_row_size(act, width));
    tr->from_float(x, out.data(), width);
    return out;
}

void dequantize(const TensorInfo& t, const uint8_t* src, float* dst, int64_t count) {
    if (t.type == GGML_TYPE_F32) {
        std::memcpy(dst, src, count * sizeof(float));
        return;
    }
    if (t.type >= GGML_TYPE_COUNT)
        throw std::runtime_error(t.name + ": unsupported GGUF type");
    const auto* tr = ggml_get_type_traits(static_cast<ggml_type>(t.type));
    if (!tr || !tr->to_float || count % ggml_blck_size(static_cast<ggml_type>(t.type)))
        throw std::runtime_error(t.name + ": cannot dequantize complete rows");
    tr->to_float(src, dst, count);
}

bool scalar_tensor(const std::string& name) {
    return name.find("norm") != std::string::npos || name.find("ssm_conv1d") != std::string::npos ||
           name.ends_with("ssm_a") || name.ends_with(".bias") ||
           name.find("_base.") != std::string::npos || name.find("_scale.") != std::string::npos;
}

}  // namespace

struct Model::Impl {
    ModelFile file;
    TensorTable tensors;
    std::vector<std::unique_ptr<Buffer>> storage;
    std::vector<std::pair<void*, size_t>> state;
    std::vector<GlmLayerState> layers;
    GlmBuffers buffers;
    GlmBuffers group_buffers;
    GlmChunkBuffers chunk;
    MtpBuffers mtp;
    Options options;
    MtpStats draft_stats;
    std::unique_ptr<GpuExperts> gpu_experts;
    struct Backup { void* source; void* copy; size_t bytes; };
    std::vector<Backup> backups;
    MoEBuffers moe;
    BlockBuffers block;
    cudaStream_t stream = nullptr;
    int gpu = 0;
    int64_t context = 0, position = 0;
    int64_t mtp_position = 0;
    float* parts = nullptr;
    float* logits = nullptr;
    float* hidden_rows = nullptr;
    float* carry_hidden = nullptr;
    float* mtp_logits = nullptr;
    std::vector<float> embedding, residual, input, outputs, host_logits;
    std::vector<float> draft_logits;
    std::vector<std::vector<float>> verify_logits;
    std::vector<int32_t> ids;

    void* alloc(size_t bytes) {
        auto buffer = std::make_unique<Buffer>(bytes);
        auto* p = buffer->get();
        storage.push_back(std::move(buffer));
        return p;
    }

    Impl(const std::string& path, const Options& opts)
        : file(path), options(opts), gpu(opts.gpu), context(opts.max_context) {
        init_cpu();
        if (context < 1 || context > 131072) throw std::runtime_error("glm5: max context must be in 1..131072");
        if (options.prefill < 1 || options.prefill > 4096 || options.mtp < 0 || options.mtp > 7 ||
            options.gpu_experts_mib < -1 || options.gpu_reserve_mib > 131072)
            throw std::invalid_argument("glm5: invalid prefill, MTP or GPU expert options");
        if (options.gpu_check && options.gpu_experts_mib < 0)
            throw std::invalid_argument("glm5: --glm-gpu-check requires --glm-gpu-experts");
        file.validate();
        if (options.mtp) file.validate_mtp();
        check(cudaSetDevice(gpu), "glm5 device");
        const auto& g = file.geometry();
        int64_t widest = 0;
        const auto& gguf = file.gguf();
        for (size_t s = 0; s < gguf.size(); ++s) {
            for (const auto& t : gguf.shard(s).tensors()) {
                if (t.shape.empty() || t.shape.front() > INT_MAX)
                    throw std::runtime_error(t.name + ": invalid tensor width");
                widest = std::max(widest, static_cast<int64_t>(t.shape.front()));
            }
        }
        if (widest <= 0 || widest > INT_MAX) throw std::runtime_error("glm5: invalid projection width");
        void* scratch = alloc(kernels::native_q8_1_bytes(static_cast<int>(widest), 8));
        for (size_t s = 0; s < gguf.size(); ++s) {
            for (const auto& t : gguf.shard(s).tensors()) {
                if (t.name == "token_embd.weight" || t.name.ends_with(".nextn.embed_tokens.weight") ||
                    t.name.find("_exps.weight") != std::string::npos ||
                    t.name.starts_with("nextn.") || t.name.find("indexer") != std::string::npos) continue;
                if (t.name.starts_with("blk.")) {
                    size_t end = 0;
                    const auto layer = std::stoll(t.name.substr(4), &end);
                    if (end == 0 || 4 + end >= t.name.size() || t.name[4 + end] != '.' || layer < 0)
                        throw std::runtime_error("glm5: invalid layer name " + t.name);
                    if (layer >= g.n_layers + (options.mtp ? 1 : 0)) continue;
                }
                if (!gguf.in_bounds(t, s)) throw std::runtime_error(t.name + ": invalid tensor payload");
                WeightRef ref;
                ref.ne0 = static_cast<int64_t>(t.shape.front());
                ref.elements = static_cast<int64_t>(t.elements());
                ref.ne1 = ref.elements / ref.ne0;
                const bool native = !scalar_tensor(t.name) && kernels::native_mmvq_supported(t.type);
                if (native) {
                    const auto bytes = tensor_payload_bytes(t);
                    auto* dst = alloc(bytes);
                    check(cudaMemcpy(dst, gguf.shard(s).tensor_data(t), bytes, cudaMemcpyHostToDevice), "glm5 weight upload");
                    ref.native_data = dst;
                    ref.native_type = static_cast<int>(t.type);
                    ref.native_q8_1 = scratch;
                    ref.resident = false;
                } else {
                    if (t.type != GGML_TYPE_F32 && t.type != GGML_TYPE_F16 && t.type != GGML_TYPE_BF16 &&
                        !scalar_tensor(t.name))
                        throw std::runtime_error(t.name + ": no native GPU kernel for " + t.type_name());
                    std::vector<float> values(static_cast<size_t>(ref.elements));
                    dequantize(t, gguf.shard(s).tensor_data(t), values.data(), ref.elements);
                    ref.bytes = values.size() * sizeof(float);
                    auto* dst = alloc(ref.bytes);
                    check(cudaMemcpy(dst, values.data(), ref.bytes, cudaMemcpyHostToDevice), "glm5 scalar upload");
                    ref.data = dst;
                    ref.kind = core::WeightKind::F32;
                }
                if (!tensors.insert(t.name, ref)) throw std::runtime_error("glm5: duplicate tensor " + t.name);
            }
        }
        const auto scratch_bytes = glm_buffers_bytes(g, 1, context);
        if (glm_buffers_init(g, 1, context, alloc(scratch_bytes), buffers) != scratch_bytes)
            throw std::runtime_error("glm5: layer scratch layout mismatch");
        const auto k = g.n_expert_used;
        const auto capacity = std::max(options.prefill, options.mtp + 1);
        const auto chunk_bytes = glm_chunk_bytes(g, k, capacity);
        if (glm_chunk_init(g, k, capacity, alloc(chunk_bytes), chunk) != chunk_bytes)
            throw std::runtime_error("glm5: chunk layout mismatch");
        group_buffers = buffers;
        if (capacity > 1) {
            const auto group_bytes = glm_buffers_bytes(g, GLM_MAX_NTOK, context);
            if (glm_buffers_init(g, GLM_MAX_NTOK, context, alloc(group_bytes), group_buffers) != group_bytes)
                throw std::runtime_error("glm5: grouped scratch layout mismatch");
        }
        moe.logits = chunk.logits; moe.ids = chunk.ids; moe.weights = chunk.weights; moe.shared = chunk.shared;
        block.R = chunk.R; block.block_out = chunk.block_out;
        block.mixed = static_cast<float*>(alloc(g.n_embd * sizeof(float)));
        parts = chunk.parts;
        logits = static_cast<float*>(alloc(g.n_vocab * sizeof(float)));
        hidden_rows = static_cast<float*>(alloc(capacity * g.n_embd * sizeof(float)));
        layers.resize(g.n_layers);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const auto bytes = glm_layer_state_bytes(g, context, l);
            auto* p = alloc(bytes);
            if (glm_state_init(g, context, l, p, layers[l]) != bytes)
                throw std::runtime_error("glm5: layer state layout mismatch");
            state.emplace_back(p, bytes);
            if (options.mtp && glm_is_kda_layer(g, l)) backups.push_back({p, alloc(bytes), bytes});
        }
        if (options.mtp) {
            const auto bytes = glm_mla_state_bytes(g, context);
            auto* p = alloc(bytes);
            if (glm_mla_state_init(g, context, p, mtp.attention) != bytes)
                throw std::runtime_error("glm5: MTP attention layout mismatch");
            state.emplace_back(p, bytes);
            mtp.embedding = static_cast<float*>(alloc(g.n_embd * sizeof(float)));
            mtp.hidden = static_cast<float*>(alloc(g.n_embd * sizeof(float)));
            mtp.concat = static_cast<float*>(alloc(2 * g.n_embd * sizeof(float)));
            mtp.residual = static_cast<float*>(alloc(g.n_embd * sizeof(float)));
            mtp.output = static_cast<float*>(alloc(g.n_embd * sizeof(float)));
            carry_hidden = static_cast<float*>(alloc(g.n_embd * sizeof(float)));
            mtp_logits = static_cast<float*>(alloc(g.n_vocab * sizeof(float)));
            draft_logits.resize(g.n_vocab);
        }
        embedding.resize(g.n_embd);
        residual.resize(g.hc_dim());
        input.resize(g.n_embd);
        outputs.resize(k * g.n_embd);
        ids.resize(k);
        host_logits.resize(g.n_vocab);
        for (int64_t l = g.n_layer_dense_lead; l < g.n_layers + (options.mtp ? 1 : 0); ++l)
            for (const char* role : {"gate", "up", "down"})
                cpu_traits(file.tensor("blk." + std::to_string(l) + ".ffn_" + role + "_exps.weight").type);
        if (options.gpu_experts_mib >= 0)
            gpu_experts = std::make_unique<GpuExperts>(file, g.n_layers + (options.mtp ? 1 : 0),
                                                      options.gpu_experts_mib, options.gpu_reserve_mib);
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "glm5 stream");
    }

    ~Impl() {
        if (stream) {
            cudaSetDevice(gpu);
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }

    void experts(int64_t layer, int nt, const std::vector<ExpertMiss>& requested) {
        const auto& g = file.geometry();
        const auto p = "blk." + std::to_string(layer) + ".ffn_";
        const auto& gate = file.tensor(p + "gate_exps.weight");
        const auto& up = file.tensor(p + "up_exps.weight");
        const auto& down = file.tensor(p + "down_exps.weight");
        std::vector<std::vector<uint8_t>> gate_acts(nt), up_acts(nt);
        const auto gr = tensor_payload_bytes(gate) / (g.n_expert * g.n_ff);
        const auto ur = tensor_payload_bytes(up) / (g.n_expert * g.n_ff);
        const auto dr = tensor_payload_bytes(down) / (g.n_expert * g.n_embd);
        std::vector<float> hidden(g.n_ff);
        for (const auto& entry : requested) {
            const int32_t e = entry.expert;
            if (e < 0 || e >= g.n_expert) throw std::runtime_error("glm5: invalid routed expert id");
            auto& ga = gate_acts.at(entry.token);
            auto& ua = up_acts.at(entry.token);
            if (ga.empty()) {
                ga = quantize(input.data() + entry.token * g.n_embd, g.n_embd, gate.type);
                ua = quantize(input.data() + entry.token * g.n_embd, g.n_embd, up.type);
            }
            const auto* gw = file.data(gate.name) + e * g.n_ff * gr;
            const auto* uw = file.data(up.name) + e * g.n_ff * ur;
            const auto* dw = file.data(down.name) + e * g.n_embd * dr;
            for (int64_t r = 0; r < g.n_ff; ++r) {
                float gv = 0, uv = 0;
                cpu_traits(gate.type).vec_dot(g.n_embd, &gv, 0, gw + r * gr, 0, ga.data(), 0, 1);
                cpu_traits(up.type).vec_dot(g.n_embd, &uv, 0, uw + r * ur, 0, ua.data(), 0, 1);
                hidden[r] = swiglu(gv, uv, static_cast<float>(g.swiglu_clamp));
            }
            const auto ha = quantize(hidden.data(), g.n_ff, down.type);
            for (int64_t r = 0; r < g.n_embd; ++r)
                cpu_traits(down.type).vec_dot(g.n_ff, &outputs[entry.row * g.n_embd + r], 0,
                                             dw + r * dr, 0, ha.data(), 0, 1);
        }
    }

    void route(int layer, const float* device_input, const int32_t* device_ids, int nt,
               float* device_output, bool stream_all) {
        const auto& g = file.geometry();
        input.resize(nt * g.n_embd);
        ids.resize(nt * g.n_expert_used);
        outputs.resize(nt * g.n_expert_used * g.n_embd);
        check(cudaMemcpyAsync(input.data(), device_input, input.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream), "glm5 expert input");
        check(cudaMemcpyAsync(ids.data(), device_ids, ids.size() * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream), "glm5 expert ids");
        check(cudaStreamSynchronize(stream), "glm5 expert routing sync");
        std::vector<ExpertMiss> all;
        all.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i)
            all.push_back({static_cast<int>(i / g.n_expert_used), static_cast<int>(i), ids[i]});
        if (!gpu_experts) {
            experts(layer, nt, all);
            check(cudaMemcpyAsync(device_output, outputs.data(), outputs.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream), "glm5 expert output");
            return;
        }
        const auto& misses = gpu_experts->run(layer, device_input, ids.data(), nt, device_output, stream_all, stream);
        experts(layer, nt, misses);
        for (const auto& entry : misses)
            check(cudaMemcpyAsync(device_output + entry.row * g.n_embd, outputs.data() + entry.row * g.n_embd,
                                   g.n_embd * sizeof(float), cudaMemcpyHostToDevice, stream), "glm5 CPU miss output");
        gpu_experts->admit(layer, ids.data(), misses, stream);
        if (options.gpu_check) {
            std::vector<float> actual(outputs.size());
            check(cudaMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream), "glm5 GPU expert diagnostic");
            check(cudaStreamSynchronize(stream), "glm5 GPU expert diagnostic sync");
            experts(layer, nt, all);
            for (size_t row = 0; row < ids.size(); ++row) {
                double error = 0, reference = 0;
                for (int64_t c = 0; c < g.n_embd; ++c) {
                    const auto i = row * g.n_embd + c;
                    if (!std::isfinite(actual[i]) || !std::isfinite(outputs[i]))
                        throw std::runtime_error("glm5 GPU expert check: non-finite result");
                    const double delta = actual[i] - outputs[i];
                    error += delta * delta;
                    reference += static_cast<double>(outputs[i]) * outputs[i];
                }
                if (error > std::max(reference, 1e-12))
                    throw std::runtime_error("glm5 GPU expert check: layer " + std::to_string(layer) +
                                             " row " + std::to_string(row) + " at position " + std::to_string(position) +
                                             " error/CPU RMS " + std::to_string(std::sqrt(error / std::max(reference, 1e-12))) +
                                             ", first GPU/CPU " + std::to_string(actual[row * g.n_embd]) + "/" +
                                             std::to_string(outputs[row * g.n_embd]));
            }
        }
    }

    void reset() {
        check(cudaSetDevice(gpu), "glm5 device");
        for (const auto& [ptr, bytes] : state)
            check(cudaMemsetAsync(ptr, 0, bytes, stream), "glm5 reset");
        if (carry_hidden) check(cudaMemsetAsync(carry_hidden, 0, file.geometry().n_embd * sizeof(float), stream),
                                 "glm5 MTP initial hidden");
        check(cudaStreamSynchronize(stream), "glm5 reset sync");
        position = mtp_position = 0;
        host_logits.clear();
        draft_stats = {};
        if (gpu_experts) gpu_experts->reset();
    }

    void embed(int32_t id, bool draft) {
        const auto& g = file.geometry();
        if (id < 0 || id >= g.n_vocab) throw std::runtime_error("glm5: token outside vocabulary");
        const auto optional = "blk." + std::to_string(g.n_layers) + ".nextn.embed_tokens.weight";
        const auto& emb = file.tensor(draft && file.gguf().find(optional) ? optional : "token_embd.weight");
        const auto row = tensor_payload_bytes(emb) / g.n_vocab;
        dequantize(emb, file.data(emb.name) + id * row, embedding.data(), g.n_embd);
    }

    void head(const float* hidden, float* destination, bool draft) {
        const auto& g = file.geometry();
        const auto optional = "blk." + std::to_string(g.n_layers) + ".nextn.shared_head_head.weight";
        const auto* weight = draft ? tensors.find(optional) : nullptr;
        if (!weight) weight = tensors.find("output.weight");
        if (!weight) throw std::runtime_error("glm5: missing output projection");
        std::string err;
        if (weight->native_data) {
            if (!gemv::gemv_quantized(*weight, {}, {}, nullptr, nullptr, destination, g.n_embd, g.n_vocab,
                                      "output.weight", stream, err, hidden))
                throw std::runtime_error(err);
        } else {
            kernels::glm_f32_gemv(hidden, static_cast<const float*>(weight->data), destination,
                                  g.n_embd, g.n_vocab, stream);
        }
    }

    int32_t mtp_step(int32_t id, const float* previous_hidden, int64_t pos, bool project) {
        const auto& g = file.geometry();
        if (pos != mtp_position || pos >= context) throw std::runtime_error("glm5 MTP: non-contiguous draft position");
        embed(id, true);
        check(cudaMemcpyAsync(mtp.embedding, embedding.data(), g.n_embd * sizeof(float),
                               cudaMemcpyHostToDevice, stream), "glm5 MTP embedding");
        std::string err;
        if (!glm_mtp_pre(tensors, g, pos, previous_hidden, buffers, moe, mtp, stream, err))
            throw std::runtime_error(err);
        route(g.n_layers, buffers.cur, reinterpret_cast<const int32_t*>(moe.ids), 1, parts, false);
        if (!glm_mtp_post(tensors, g, buffers, moe, parts, mtp, stream, err)) throw std::runtime_error(err);
        ++mtp_position;
        if (!project) {
            check(cudaStreamSynchronize(stream), "glm5 MTP catch-up sync");
            return -1;
        }
        head(mtp.output, mtp_logits, true);
        check(cudaMemcpyAsync(draft_logits.data(), mtp_logits, draft_logits.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream), "glm5 draft logits");
        check(cudaStreamSynchronize(stream), "glm5 draft sync");
        return greedy(draft_logits);
    }

    void catch_up(const std::vector<int32_t>& tokens, int64_t start) {
        const auto n = file.geometry().n_embd;
        if (!options.mtp) return;
        for (size_t t = 0; t < tokens.size(); ++t) {
            mtp_step(tokens[t], carry_hidden, start + t, false);
            check(cudaMemcpyAsync(carry_hidden, hidden_rows + t * n, n * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream), "glm5 target hidden carry");
        }
        check(cudaStreamSynchronize(stream), "glm5 MTP carry sync");
    }

    void run_batch(const std::vector<int32_t>& tokens, bool all_logits, bool catchup) {
        const auto& g = file.geometry();
        const auto nt = static_cast<int>(tokens.size());
        if (tokens.empty() || tokens.size() > static_cast<size_t>(chunk.T) ||
            tokens.size() > static_cast<uint64_t>(context - position))
            throw std::invalid_argument("glm5: empty batch, batch exceeds allocation or context exhausted");
        check(cudaSetDevice(gpu), "glm5 device");
        const auto start = position;
        residual.resize(tokens.size() * g.hc_dim());
        for (int t = 0; t < nt; ++t) {
            embed(tokens[t], false);
            for (int64_t h = 0; h < g.hc; ++h)
                std::copy(embedding.begin(), embedding.end(), residual.begin() + t * g.hc_dim() + h * g.n_embd);
        }
        check(cudaMemcpyAsync(chunk.R, residual.data(), residual.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream), "glm5 batch embedding");
        std::string err;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            for (int first = 0; first < nt; first += GLM_MAX_NTOK) {
                GlmBuffers b; MoEBuffers m; BlockBuffers bb;
                glm_chunk_group_view(chunk, group_buffers, moe, block, first,
                                      std::min<int64_t>(GLM_MAX_NTOK, nt - first), b, m, bb);
                if (!glm_block_layer_pre(tensors, g, l, start + first, 0, b, layers[l], m, g.n_expert_used,
                                         bb, stream, err, nullptr)) throw std::runtime_error(err);
            }
            if (!g.is_dense_ffn_layer(l))
                route(l, chunk.cur, chunk.ids, nt, chunk.parts, nt > 1);
            for (int t = 0; t < nt; ++t) {
                GlmBuffers b; MoEBuffers m; BlockBuffers bb;
                glm_chunk_view(chunk, buffers, moe, block, t, b, m, bb);
                if (!glm_block_layer_post(tensors, g, l, g.n_expert_used, b, m, bb,
                                          chunk.parts + t * g.n_expert_used * g.n_embd, stream, err))
                    throw std::runtime_error(err);
            }
        }
        host_logits.resize(g.n_vocab);
        verify_logits.clear();
        if (all_logits) verify_logits.assign(tokens.size(), std::vector<float>(g.n_vocab));
        for (int t = 0; t < nt; ++t) {
            if (t + 1 != nt && !all_logits && !options.mtp) continue;
            BlockBuffers bb = block;
            bb.R = chunk.R + t * g.hc_dim();
            if (!glm_head_mix(tensors, g, bb, hidden_rows + t * g.n_embd, stream, err))
                throw std::runtime_error(err);
            if (all_logits || t + 1 == nt) {
                head(hidden_rows + t * g.n_embd, logits, false);
                auto& destination = all_logits ? verify_logits[t] : host_logits;
                check(cudaMemcpyAsync(destination.data(), logits, destination.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream), "glm5 target logits");
            }
        }
        check(cudaGetLastError(), "glm5 kernel launch");
        check(cudaStreamSynchronize(stream), "glm5 batch sync");
        if (all_logits) host_logits = verify_logits.back();
        position += nt;
        if (catchup) catch_up(tokens, start);
    }

    const std::vector<float>& token(int32_t id) {
        run_batch({id}, false, true);
        return host_logits;
    }

    const std::vector<float>& prefill(const std::vector<int32_t>& tokens) {
        if (tokens.empty() || tokens.size() > static_cast<uint64_t>(context - position))
            throw std::invalid_argument("glm5: empty prompt or prompt exceeds context");
        for (size_t first = 0; first < tokens.size(); first += options.prefill) {
            const auto last = std::min(tokens.size(), first + options.prefill);
            run_batch(std::vector<int32_t>(tokens.begin() + first, tokens.begin() + last), false, true);
        }
        return host_logits;
    }

    void checkpoint(bool restore) {
        for (const auto& backup : backups)
            check(cudaMemcpyAsync(restore ? backup.source : backup.copy, restore ? backup.copy : backup.source,
                                   backup.bytes, cudaMemcpyDeviceToDevice, stream), "glm5 KDA checkpoint");
    }

    std::vector<int32_t> generate(int count) {
        if (host_logits.empty() || count < 0 || count > context - position)
            throw std::invalid_argument("glm5: prefill first and keep generation within the context");
        std::vector<int32_t> output;
        while (output.size() < static_cast<size_t>(count)) {
            const auto next = greedy(host_logits);
            if (file.stop_ids().count(next)) break;
            const int drafts = std::min<int64_t>(options.mtp, count - output.size() - 1);
            if (!drafts) { output.push_back(next); token(next); continue; }
            const auto base = position;
            std::vector<int32_t> proposed{next};
            const float* hidden = carry_hidden;
            for (int i = 0; i < drafts; ++i) {
                const auto draft = mtp_step(proposed.back(), hidden, base + i, true);
                proposed.push_back(draft);
                hidden = mtp.output;
                if (file.stop_ids().count(draft)) break;
            }
            draft_stats.offered += proposed.size() - 1;
            checkpoint(false);
            run_batch(proposed, true, false);
            const auto accepted = verify_greedy(proposed, verify_logits, file.stop_ids());
            ++draft_stats.verify_batches;
            draft_stats.accepted += accepted.committed ? accepted.committed - 1 : 0;
            std::vector<int32_t> committed(proposed.begin(), proposed.begin() + accepted.committed);
            mtp_position = base;
            if (accepted.committed != proposed.size()) {
                checkpoint(true);
                position = base;
                ++draft_stats.replays;
                if (!committed.empty()) run_batch(committed, false, true);
            } else catch_up(committed, base);
            output.insert(output.end(), committed.begin(), committed.end());
            if (accepted.stopped) break;
        }
        return output;
    }
};

Model::Model(const std::string& path, int gpu, int64_t context)
    : Model(path, Options{.gpu = gpu, .max_context = context}) {}
Model::Model(const std::string& path, const Options& options) : impl_(std::make_unique<Impl>(path, options)) {
    impl_->reset();
}
Model::~Model() = default;
const Geometry& Model::geometry() const { return impl_->file.geometry(); }
const std::set<int32_t>& Model::stop_ids() const { return impl_->file.stop_ids(); }
const std::vector<float>& Model::token(int32_t id) { return impl_->token(id); }
const std::vector<float>& Model::prefill(const std::vector<int32_t>& ids) { return impl_->prefill(ids); }
std::vector<int32_t> Model::generate(int count) { return impl_->generate(count); }
ExpertCacheStats Model::cache_stats() const { return impl_->gpu_experts ? impl_->gpu_experts->stats() : ExpertCacheStats{}; }
MtpStats Model::mtp_stats() const { return impl_->draft_stats; }
void Model::reset() { impl_->reset(); }

}  // namespace strata::glm5
