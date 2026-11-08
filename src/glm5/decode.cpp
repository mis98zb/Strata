#include "decode.hpp"
#include "cpu.hpp"
#include "weights.hpp"
#include "state_transfer.hpp"
#include "strata/glm5/layer.hpp"
#include "strata/glm5/model_file.hpp"
#include "gemv_util.hpp"
#include "gpu_support.hpp"
#include "strata/glm5/speculative.hpp"
#include "compute_scope.hpp"
#include "phase_probe.hpp"

#include "ggml.h"
#include "ggml-cpu.h"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace strata::glm5 {
struct DecodeExecutor::Impl {
    const ModelFile& file;
    std::unique_ptr<WeightBank> weights;
    TensorTable tensors;
    std::vector<std::unique_ptr<Buffer>> storage;
    std::vector<std::pair<void*, size_t>> state;
    std::vector<GlmLayerState> layers;
    std::vector<KdaReplay> replay;
    std::vector<DsaReplay> dsa_replay;
    GlmBuffers buffers;
    GlmBuffers group_buffers;
    GlmChunkBuffers chunk;
    MtpBuffers mtp;
    Options options;
    PhaseProbe probe;
    MtpStats draft_stats;
    std::unique_ptr<GpuExperts> gpu_experts;
    std::unique_ptr<CpuRowPool> cpu_row_pool;
    struct Backup { void* source; void* copy; size_t bytes; };
    std::vector<Backup> backups;
    std::vector<Backup> draft_indexer_backups;
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

    Impl(const ModelFile& model, const Options& opts, const HostState* initial)
        : file(model), options(resolve_options(opts)), probe(opts.probe), gpu(opts.gpu), context(opts.max_context) {
        init_cpu();
        if (options.kda_register_state && file.geometry().kda_head_dim != 128)
            throw std::invalid_argument("glm5: register KDA requires head dimension 128");
        if (context < 1 || context > max_context_limit)
            throw std::runtime_error("glm5: max context must be in 1.." + std::to_string(max_context_limit));
        if (options.prefill < 1 || options.prefill > 4096 || options.mtp < 0 || options.mtp > 7 ||
            options.cpu_threads < 1 || options.cpu_threads > 256 ||
            options.gpu_experts_mib < default_gpu_expert_budget || options.gpu_reserve_mib > 131072 ||
            options.gpu_stage_slots < 0 || options.gpu_stage_slots > 8 ||
            ((options.gpu_warm || options.gpu_stage_slots) && options.gpu_experts_mib == -1) ||
            (options.gpu_warm && !options.full_layer_prefill) || (options.gpu_heat_quotas && !options.gpu_warm))
            throw std::invalid_argument("glm5: invalid prefill, MTP or GPU expert options");
        if (options.dsa_tail < -1 || options.dsa_tail > 1 || (options.dsa_tail >= 0 && !options.dsa) ||
            (options.dsa_parallel_topk && !options.dsa) ||
            (options.dsa_tail >= 0 && options.dsa_tail != file.geometry().idx_select_tail))
            throw std::invalid_argument("glm5: invalid or mismatched DSA tail policy");
        if (options.flash_attn && options.mtp && !options.dsa)
            throw std::invalid_argument("glm5: Flash MTP currently requires --dsa");
        if (initial) initial->validate(file.geometry(), context, options.mtp > 0, options.dsa);
        if (options.gpu_warm && (!initial || initial->expert_heat.empty()))
            throw std::invalid_argument("glm5: expert warmup requires full-layer prompt heat");
        if (options.cpu_threads > 1 && options.cpu_expert_mode != CpuExpertMode::Rows && !cpu_parallel_available())
            throw std::invalid_argument("glm5: multiple CPU expert threads require an OpenMP-enabled build");
        if (options.gpu_check && options.gpu_experts_mib == -1)
            throw std::invalid_argument("glm5: --glm-gpu-check requires --glm-gpu-experts");
        file.validate();
        if (options.dsa) file.validate_dsa(options.mtp > 0);
        if (options.mtp) file.validate_mtp();
        check(cudaSetDevice(gpu), "glm5 device");
        options = resolve_execution_options(options, file.geometry());
        if (options.cpu_expert_mode == CpuExpertMode::Rows)
            cpu_row_pool = std::make_unique<CpuRowPool>(
                file, file.geometry().n_layers + (options.mtp ? 1 : 0), options.cpu_threads,
                options.cpu_atomic_completion);
        if (options.cpu_threads > 1 || cpu_row_pool)
            std::fprintf(stderr, "glm5 CPU experts: up to %d threads mode=%s\n",
                         options.cpu_threads, cpu_row_pool ? "rows" : "reference");
        if (options.cpu_atomic_completion)
            std::fprintf(stderr, "glm5 CPU row wait: atomic completion, caller poll up to 100 us; idle workers sleep\n");
        ComputeScope compute(options);
        if (options.flash_attn && !kernels::glm_flash_supported())
            throw std::runtime_error("glm5 Flash Attention: unsupported CUDA device");
        const auto& g = file.geometry();
        weights = std::make_unique<WeightBank>(file, [&](const TensorInfo& t) {
            return decode_tensor(t, g.n_layers + (options.mtp ? 1 : 0), options.dsa);
        });
        tensors = weights->table();
        const auto scratch_bytes = glm_buffers_bytes(g, 1, context);
        if (glm_buffers_init(g, 1, context, alloc(scratch_bytes), buffers) != scratch_bytes)
            throw std::runtime_error("glm5: layer scratch layout mismatch");
        const auto k = g.n_expert_used;
        const auto capacity = std::max(options.full_layer_prefill ? 1 : options.prefill, options.mtp + 1);
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
        replay.resize(g.n_layers);
        dsa_replay.resize(g.n_layers);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const auto bytes = glm_layer_state_bytes(g, context, l);
            auto* p = alloc(bytes);
            if (glm_state_init(g, context, l, p, layers[l]) != bytes)
                throw std::runtime_error("glm5: layer state layout mismatch");
            state.emplace_back(p, bytes);
            if (options.mtp && glm_is_kda_layer(g, l)) {
                backups.push_back({p, alloc(bytes), bytes});
                auto& capture = replay[l];
                capture.capacity = options.mtp + 1;
                const auto nv = g.n_head * g.kda_head_dim;
                const auto span = capture.capacity * nv;
                auto* data = static_cast<float*>(alloc((4 * span + capture.capacity * g.n_head) * sizeof(float)));
                capture.q = data; capture.k = data + span; capture.v = data + 2 * span;
                capture.gate = data + 3 * span; capture.beta = data + 4 * span;
            }
            if (options.mtp && options.dsa && !glm_is_kda_layer(g, l)) {
                const auto partial_bytes = g.idx_kpool * g.idx_key_dim * sizeof(float);
                for (float* source : {layers[l].idx_partial_k, layers[l].idx_partial_g})
                    backups.push_back({source, alloc(partial_bytes), partial_bytes});
                auto& capture = dsa_replay[l];
                capture.capacity = options.mtp + 1;
                const auto span = capture.capacity * g.idx_key_dim;
                capture.key = static_cast<float*>(alloc(2 * span * sizeof(float)));
                capture.gate = capture.key + span;
            }
        }
        if (options.mtp) {
            const auto bytes = glm_mla_state_bytes(g, context);
            auto* p = alloc(bytes);
            if (glm_mla_state_init(g, context, p, mtp.attention) != bytes)
                throw std::runtime_error("glm5: MTP attention layout mismatch");
            state.emplace_back(p, bytes);
            if (options.dsa) {
                const auto partial_bytes = g.idx_kpool * g.idx_key_dim * sizeof(float);
                for (float* source : {mtp.attention.idx_partial_k, mtp.attention.idx_partial_g})
                    draft_indexer_backups.push_back({source, alloc(partial_bytes), partial_bytes});
            }
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
        const std::vector<uint32_t> empty_heat;
        if (options.gpu_experts_mib >= 0)
            gpu_experts = std::make_unique<GpuExperts>(file, g.n_layers + (options.mtp ? 1 : 0),
                options.gpu_experts_mib, options.gpu_reserve_mib, initial ? initial->expert_heat : empty_heat,
                options.gpu_heat_quotas, options.gpu_stage_slots, options.gpu_expert_mode == GpuExpertMode::Single);
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
        if (requested.empty()) return;
        auto timing = probe.host(ProbePhase::CpuExperts);
        if (cpu_row_pool) {
            probe.bytes(ProbePhase::CpuExperts, checked_mul(requested.size(), cpu_row_pool->expert_bytes(layer)));
            probe.cpu_experts(requested.size());
            cpu_row_pool->run(static_cast<int>(layer), nt, input, requested, outputs);
            return;
        }
        const auto& g = file.geometry();
        const auto p = "blk." + std::to_string(layer) + ".ffn_";
        const auto& gate = file.tensor(p + "gate_exps.weight");
        const auto& up = file.tensor(p + "up_exps.weight");
        const auto& down = file.tensor(p + "down_exps.weight");
        std::vector<std::vector<uint8_t>> gate_acts(nt), up_acts(nt);
        const auto gr = tensor_payload_bytes(gate) / (g.n_expert * g.n_ff);
        const auto ur = tensor_payload_bytes(up) / (g.n_expert * g.n_ff);
        const auto dr = tensor_payload_bytes(down) / (g.n_expert * g.n_embd);
        probe.bytes(ProbePhase::CpuExperts, checked_mul(requested.size(),
            g.n_ff * (gr + ur) + g.n_embd * dr));
        probe.cpu_experts(requested.size());
        const auto* gate_data = file.data(gate.name);
        const auto* up_data = file.data(up.name);
        const auto* down_data = file.data(down.name);
        for (const auto& entry : requested) {
            const int32_t e = entry.expert;
            if (e < 0 || e >= g.n_expert || entry.token < 0 || entry.token >= nt ||
                entry.row < 0 || entry.row >= nt * g.n_expert_used)
                throw std::runtime_error("glm5: invalid routed CPU expert");
            auto& ga = gate_acts.at(entry.token);
            auto& ua = up_acts.at(entry.token);
            if (ga.empty()) {
                ga = quantize(input.data() + entry.token * g.n_embd, g.n_embd, gate.type);
                ua = quantize(input.data() + entry.token * g.n_embd, g.n_embd, up.type);
            }
        }
        const auto gate_dot = cpu_traits(gate.type).vec_dot;
        const auto up_dot = cpu_traits(up.type).vec_dot;
        const auto& down_traits = cpu_traits(down.type);
        const auto* act = ggml_get_type_traits_cpu(down_traits.vec_dot_type);
        if (!act || !act->from_float || g.n_ff % ggml_blck_size(down_traits.vec_dot_type))
            throw std::runtime_error("glm5: unsupported expert activation quantization");
        const auto workers = std::min<size_t>(options.cpu_threads, requested.size());
        std::vector<std::vector<float>> hidden_storage(workers, std::vector<float>(g.n_ff));
        std::vector<std::vector<uint8_t>> activation_storage(
            workers, std::vector<uint8_t>(ggml_row_size(down_traits.vec_dot_type, g.n_ff)));
#ifdef _OPENMP
#pragma omp parallel for num_threads(workers) schedule(static) if(workers > 1)
#endif
        for (size_t i = 0; i < requested.size(); ++i) {
            size_t worker = 0;
#ifdef _OPENMP
            worker = omp_get_thread_num();
#endif
            auto& hidden = hidden_storage[worker];
            auto& ha = activation_storage[worker];
            const auto& entry = requested[i];
            const auto e = entry.expert;
            const auto& ga = gate_acts[entry.token];
            const auto& ua = up_acts[entry.token];
            const auto* gw = gate_data + e * g.n_ff * gr;
            const auto* uw = up_data + e * g.n_ff * ur;
            const auto* dw = down_data + e * g.n_embd * dr;
            for (int64_t r = 0; r < g.n_ff; ++r) {
                float gv = 0, uv = 0;
                gate_dot(g.n_embd, &gv, 0, gw + r * gr, 0, ga.data(), 0, 1);
                up_dot(g.n_embd, &uv, 0, uw + r * ur, 0, ua.data(), 0, 1);
                hidden[r] = swiglu(gv, uv, static_cast<float>(g.swiglu_clamp));
            }
            act->from_float(hidden.data(), ha.data(), g.n_ff);
            for (int64_t r = 0; r < g.n_embd; ++r)
                down_traits.vec_dot(g.n_ff, &outputs[entry.row * g.n_embd + r], 0,
                                             dw + r * dr, 0, ha.data(), 0, 1);
        }
    }

    void route(int layer, const float* device_input, const int32_t* device_ids, int nt,
               float* device_output, bool stream_all) {
        const auto& g = file.geometry();
        input.resize(nt * g.n_embd);
        ids.resize(nt * g.n_expert_used);
        outputs.resize(nt * g.n_expert_used * g.n_embd);
        probe.gpu(ProbePhase::RouteD2H, stream, [&] {
            check(cudaMemcpyAsync(input.data(), device_input, input.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream), "glm5 expert input");
            check(cudaMemcpyAsync(ids.data(), device_ids, ids.size() * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream), "glm5 expert ids");
        });
        probe.bytes(ProbePhase::RouteD2H, input.size() * sizeof(float) + ids.size() * sizeof(int32_t));
        {
            auto timing = probe.host(ProbePhase::Synchronize);
            check(cudaStreamSynchronize(stream), "glm5 expert routing sync");
        }
        probe.collect();
        std::vector<ExpertMiss> all;
        all.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i)
            all.push_back({static_cast<int>(i / g.n_expert_used), static_cast<int>(i), ids[i]});
        if (!gpu_experts) {
            experts(layer, nt, all);
            probe.gpu(ProbePhase::CpuResultH2D, stream, [&] {
                check(cudaMemcpyAsync(device_output, outputs.data(), outputs.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream), "glm5 expert output");
                probe.cpu_result_copy();
            });
            probe.bytes(ProbePhase::CpuResultH2D, outputs.size() * sizeof(float));
            return;
        }
        const std::vector<ExpertMiss>* missing = nullptr;
        auto uploaded = gpu_experts->stats().upload_bytes;
        probe.gpu(ProbePhase::GpuExperts, stream, [&] {
            missing = &gpu_experts->run(layer, device_input, ids.data(), nt, device_output, stream_all, stream);
        });
        probe.bytes(ProbePhase::GpuExperts, gpu_experts->stats().upload_bytes - uploaded);
        const auto& misses = *missing;
        experts(layer, nt, misses);
        probe.gpu(ProbePhase::CpuResultH2D, stream, [&] {
            for (const auto& entry : misses) {
                check(cudaMemcpyAsync(device_output + entry.row * g.n_embd, outputs.data() + entry.row * g.n_embd,
                                       g.n_embd * sizeof(float), cudaMemcpyHostToDevice, stream), "glm5 CPU miss output");
                probe.cpu_result_copy();
            }
        });
        probe.bytes(ProbePhase::CpuResultH2D, misses.size() * g.n_embd * sizeof(float));
        uploaded = gpu_experts->stats().upload_bytes;
        probe.gpu(ProbePhase::CacheAdmit, stream, [&] { gpu_experts->admit(layer, ids.data(), misses, stream); });
        probe.bytes(ProbePhase::CacheAdmit, gpu_experts->stats().upload_bytes - uploaded);
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
        probe.reset();
        for (auto& capture : replay) capture.tokens = 0;
        for (auto& capture : dsa_replay) capture.tokens = 0;
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
        project_head(tensors, file.geometry(), hidden, destination, draft, stream);
    }

    int32_t mtp_step(int32_t id, const float* previous_hidden, int64_t pos, bool project) {
        ProbeScope probe_scope(probe);
        const auto& g = file.geometry();
        if (pos != mtp_position || pos >= context) throw std::runtime_error("glm5 MTP: non-contiguous draft position");
        embed(id, true);
        probe.rows(1);
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
        ProbeScope probe_scope(probe);
        const auto& g = file.geometry();
        const auto nt = static_cast<int>(tokens.size());
        if (tokens.empty() || tokens.size() > static_cast<size_t>(chunk.T) ||
            tokens.size() > static_cast<uint64_t>(context - position))
            throw std::invalid_argument("glm5: empty batch, batch exceeds allocation or context exhausted");
        if (all_logits && options.mtp && nt > options.mtp + 1)
            throw std::invalid_argument("glm5 MTP: verification exceeds replay capacity");
        if (all_logits) for (auto& capture : replay) capture.tokens = 0;
        if (all_logits) for (auto& capture : dsa_replay) capture.tokens = 0;
        check(cudaSetDevice(gpu), "glm5 device");
        const auto start = position;
        residual.resize(tokens.size() * g.hc_dim());
        for (int t = 0; t < nt; ++t) {
            embed(tokens[t], false);
            for (int64_t h = 0; h < g.hc; ++h)
                std::copy(embedding.begin(), embedding.end(), residual.begin() + t * g.hc_dim() + h * g.n_embd);
        }
        probe.gpu(ProbePhase::EmbeddingH2D, stream, [&] {
            check(cudaMemcpyAsync(chunk.R, residual.data(), residual.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream), "glm5 batch embedding");
        });
        probe.bytes(ProbePhase::EmbeddingH2D, residual.size() * sizeof(float));
        probe.rows(nt);
        std::string err;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            for (int first = 0; first < nt; first += GLM_MAX_NTOK) {
                GlmBuffers b; MoEBuffers m; BlockBuffers bb;
                glm_chunk_group_view(chunk, group_buffers, moe, block, first,
                                      std::min<int64_t>(GLM_MAX_NTOK, nt - first), b, m, bb);
                if (all_logits && options.mtp && glm_is_kda_layer(g, l)) b.replay = &replay[l];
                if (all_logits && options.mtp && options.dsa && !glm_is_kda_layer(g, l))
                    b.dsa_replay = &dsa_replay[l];
                probe.gpu(ProbePhase::LayerPre, stream, [&] {
                    if (!glm_block_layer_pre(tensors, g, l, start + first, 0, b, layers[l], m, g.n_expert_used,
                                             bb, stream, err, nullptr)) throw std::runtime_error(err);
                });
            }
            if (!g.is_dense_ffn_layer(l))
                route(l, chunk.cur, chunk.ids, nt, chunk.parts, nt > 1);
            for (int t = 0; t < nt; ++t) {
                GlmBuffers b; MoEBuffers m; BlockBuffers bb;
                glm_chunk_view(chunk, buffers, moe, block, t, b, m, bb);
                probe.gpu(ProbePhase::LayerPost, stream, [&] {
                    if (!glm_block_layer_post(tensors, g, l, g.n_expert_used, b, m, bb,
                                              chunk.parts + t * g.n_expert_used * g.n_embd, stream, err))
                        throw std::runtime_error(err);
                });
            }
        }
        host_logits.resize(g.n_vocab);
        verify_logits.clear();
        if (all_logits) verify_logits.assign(tokens.size(), std::vector<float>(g.n_vocab));
        for (int t = 0; t < nt; ++t) {
            if (t + 1 != nt && !all_logits && !options.mtp) continue;
            probe.gpu(ProbePhase::Head, stream, [&] {
            BlockBuffers bb = block;
            bb.R = chunk.R + t * g.hc_dim();
            if (!glm_head_mix(tensors, g, bb, hidden_rows + t * g.n_embd, stream, err))
                throw std::runtime_error(err);
            if (all_logits || t + 1 == nt) {
                head(hidden_rows + t * g.n_embd, logits, false);
                auto& destination = all_logits ? verify_logits[t] : host_logits;
                check(cudaMemcpyAsync(destination.data(), logits, destination.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream), "glm5 target logits");
                probe.bytes(ProbePhase::Head, destination.size() * sizeof(float));
            }
            });
        }
        check(cudaGetLastError(), "glm5 kernel launch");
        {
            auto timing = probe.host(ProbePhase::Synchronize);
            check(cudaStreamSynchronize(stream), "glm5 batch sync");
        }
        probe.collect();
        if (all_logits) host_logits = verify_logits.back();
        position += nt;
        if (options.mtp) ++draft_stats.target_batches;
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
                                   backup.bytes, cudaMemcpyDeviceToDevice, stream), "glm5 recurrent/indexer checkpoint");
    }

    void checkpoint_draft_indexer(bool restore) {
        for (const auto& backup : draft_indexer_backups)
            check(cudaMemcpyAsync(restore ? backup.source : backup.copy, restore ? backup.copy : backup.source,
                                   backup.bytes, cudaMemcpyDeviceToDevice, stream), "glm5 NextN indexer checkpoint");
    }

    void replay_indexer(int64_t layer, int64_t base, int64_t kept) {
        const auto& g = file.geometry();
        const auto& capture = dsa_replay[layer];
        if (kept < 1 || kept > capture.tokens || capture.tokens > capture.capacity)
            throw std::logic_error("glm5 DSA replay: incomplete verification capture");
        const auto name = "blk." + std::to_string(layer) + ".indexer_compressor_ape.weight";
        const auto ape = tensors.find(name);
        if (!ape || ape->bytes < g.idx_kpool * g.idx_key_dim * sizeof(float))
            throw std::runtime_error(name + ": missing DSA replay pool embedding");
        auto& state = layers[layer];
        kernels::glm_dsa_append(capture.key, capture.gate, static_cast<const float*>(ape->data),
            g.idx_key_dim, g.idx_kpool, kept, base, state.idx_partial_k, state.idx_partial_g,
            state.idx_pooled, stream);
        ++draft_stats.dsa_replayed_layers;
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
            checkpoint(false);
            checkpoint_draft_indexer(false);
            const float* hidden = carry_hidden;
            for (int i = 0; i < drafts; ++i) {
                const auto draft = mtp_step(proposed.back(), hidden, base + i, true);
                proposed.push_back(draft);
                hidden = mtp.output;
                if (file.stop_ids().count(draft)) break;
            }
            draft_stats.offered += proposed.size() - 1;
            run_batch(proposed, true, false);
            const auto accepted = verify_greedy(proposed, verify_logits, file.stop_ids());
            ++draft_stats.verify_batches;
            draft_stats.accepted += accepted.committed ? accepted.committed - 1 : 0;
            std::vector<int32_t> committed(proposed.begin(), proposed.begin() + accepted.committed);
            mtp_position = base;
            checkpoint_draft_indexer(true);
            if (accepted.committed != proposed.size()) {
                checkpoint(true);
                ++draft_stats.replays;
                if (committed.empty()) throw std::logic_error("glm5 MTP: sampled base token was not committed");
                std::string err;
                for (int64_t l = 0; l < file.geometry().n_layers; ++l) {
                    if (!glm_is_kda_layer(file.geometry(), l)) {
                        if (options.dsa) replay_indexer(l, base, committed.size());
                        continue;
                    }
                    if (!glm_kda_replay(tensors, file.geometry(), l, group_buffers, layers[l], replay[l],
                                         committed.size(), stream, err)) throw std::runtime_error(err);
                    ++draft_stats.replayed_layers;
                }
                position = base + committed.size();
                host_logits = verify_logits[committed.size() - 1];
                catch_up(committed, base);
            } else catch_up(committed, base);
            output.insert(output.end(), committed.begin(), committed.end());
            if (accepted.stopped) break;
        }
        return output;
    }
};

DecodeExecutor::DecodeExecutor(const ModelFile& file, const Options& options, const HostState* initial)
    : impl_(std::make_unique<Impl>(file, options, initial)) {
    impl_->reset();
    if (initial) {
        const auto& g = file.geometry();
        initial->validate(g, options.max_context, options.mtp > 0, options.dsa);
        if (!initial->layers.empty()) {
            for (int64_t l = 0; l < g.n_layers; ++l)
                restore_layer(g, initial->layers[l], impl_->layers[l], impl_->stream);
            if (options.mtp) {
                restore_layer(g, initial->mtp, impl_->mtp.attention, impl_->stream);
                check(cudaMemcpyAsync(impl_->carry_hidden, initial->carry_hidden.data(), g.n_embd * sizeof(float),
                                       cudaMemcpyHostToDevice, impl_->stream), "glm5 restore MTP hidden");
            }
        }
        impl_->position = impl_->mtp_position = initial->position;
        impl_->host_logits = initial->logits;
        check(cudaStreamSynchronize(impl_->stream), "glm5 decode handoff sync");
    }
    if (options.gpu_warm) impl_->gpu_experts->warm(initial->expert_heat, impl_->stream);
}
DecodeExecutor::~DecodeExecutor() = default;
const std::vector<float>& DecodeExecutor::token(int32_t id) {
    ComputeScope compute(impl_->options);
    return impl_->token(id);
}
const std::vector<float>& DecodeExecutor::prefill(const std::vector<int32_t>& ids) {
    ComputeScope compute(impl_->options);
    return impl_->prefill(ids);
}
std::vector<int32_t> DecodeExecutor::generate(int count) {
    ComputeScope compute(impl_->options);
    return impl_->generate(count);
}
ExpertCacheStats DecodeExecutor::cache_stats() const { return impl_->gpu_experts ? impl_->gpu_experts->stats() : ExpertCacheStats{}; }
MtpStats DecodeExecutor::mtp_stats() const { return impl_->draft_stats; }
ProbeStats DecodeExecutor::probe_stats() { return impl_->probe.snapshot(); }
int64_t DecodeExecutor::position() const { return impl_->position; }
HostState DecodeExecutor::export_state() {
    ComputeScope compute(impl_->options);
    const auto& g = impl_->file.geometry();
    check(cudaSetDevice(impl_->gpu), "glm5 handoff device");
    if (impl_->options.mtp && impl_->mtp_position != impl_->position)
        throw std::runtime_error("glm5 handoff: draft context has uncommitted positions");
    HostState image;
    image.position = impl_->position;
    image.dsa = impl_->options.dsa;
    image.dsa_tail = impl_->options.dsa ? g.idx_select_tail : -1;
    image.logits = impl_->host_logits;
    if (impl_->options.gpu_warm) image.expert_heat = impl_->gpu_experts->heat();
    for (int64_t l = 0; l < g.n_layers; ++l)
        image.layers.push_back(capture_layer(g, impl_->layers[l], image.position, glm_is_kda_layer(g, l), impl_->stream));
    if (impl_->options.mtp) {
        image.mtp = capture_layer(g, impl_->mtp.attention, image.position, false, impl_->stream);
        image.carry_hidden.resize(g.n_embd);
        check(cudaMemcpyAsync(image.carry_hidden.data(), impl_->carry_hidden, g.n_embd * sizeof(float),
                               cudaMemcpyDeviceToHost, impl_->stream), "glm5 export MTP hidden");
        check(cudaStreamSynchronize(impl_->stream), "glm5 export hidden sync");
    }
    image.validate(g, impl_->context, impl_->options.mtp > 0, impl_->options.dsa);
    return image;
}
void DecodeExecutor::reset() { impl_->reset(); }

}  // namespace strata::glm5
