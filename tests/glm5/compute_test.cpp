#include "compute_scope.hpp"
#include "gpu_support.hpp"
#include "strata/glm5/layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

namespace {
using namespace strata::glm5;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class T> void upload(Buffer& device, const std::vector<T>& values) {
    check(cudaMemcpy(device.get(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice), "compute upload");
}

float measure(cudaStream_t stream, const std::function<void()>& operation) {
    cudaEvent_t start = nullptr, end = nullptr;
    check(cudaEventCreate(&start), "compute timing start");
    check(cudaEventCreate(&end), "compute timing end");
    try {
        operation();
        check(cudaStreamSynchronize(stream), "compute warmup");
        check(cudaEventRecord(start, stream), "compute begin timing");
        for (int i = 0; i < 3; ++i) operation();
        check(cudaEventRecord(end, stream), "compute end timing");
        check(cudaEventSynchronize(end), "compute timing sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start, end), "compute elapsed");
        cudaEventDestroy(start); cudaEventDestroy(end);
        return ms / 3;
    } catch (...) {
        cudaEventDestroy(start); cudaEventDestroy(end);
        throw;
    }
}

void kda(int heads, int tokens, bool stress, cudaStream_t stream, bool benchmark = false) {
    constexpr int width = 128, guard = 16;
    const size_t elements = static_cast<size_t>(heads) * width * tokens;
    std::vector<float> q(elements), k(elements), v(elements), gate(elements), beta(heads * tokens);
    std::vector<float> initial(static_cast<size_t>(heads) * width * width);
    for (size_t i = 0; i < elements; ++i) {
        q[i] = std::sin(i * 0.173) * 0.08f;
        k[i] = std::cos(i * 0.137) * 0.08f;
        v[i] = std::sin(i * 0.119) * (stress ? 1e6f : 0.3f);
        gate[i] = stress && i % 7 == 0 ? 50.0f : -2.5f + std::cos(i * 0.071) * 2.5f;
    }
    for (size_t i = 0; i < beta.size(); ++i) beta[i] = std::sin(i * 0.31) * 3;
    for (size_t i = 0; i < initial.size(); ++i)
        initial[i] = std::cos(i * 0.113) * (stress ? 1e7f : 0.1f);
    std::vector<float> a(elements + 2 * guard, 12345), b(a);
    Buffer dq(elements * 4), dk(elements * 4), dv(elements * 4), dg(elements * 4), db(beta.size() * 4);
    Buffer sa(initial.size() * 4), sb(initial.size() * 4), da(a.size() * 4), dout(b.size() * 4);
    upload(dq, q); upload(dk, k); upload(dv, v); upload(dg, gate); upload(db, beta);
    upload(sa, initial); upload(sb, initial); upload(da, a); upload(dout, b);
    check(cudaStreamSynchronize(nullptr), "compute uploads before nonblocking execution");
    const auto run = [&](Buffer& state, Buffer& output) {
        strata::kernels::glm_kda_delta(static_cast<float*>(dq.get()), static_cast<float*>(dk.get()),
            static_cast<float*>(dv.get()), static_cast<float*>(dg.get()), static_cast<float*>(db.get()),
            static_cast<float*>(state.get()), static_cast<float*>(output.get()) + guard, width, heads, tokens, stream);
    };
    float old_ms = 0, new_ms = 0;
    {
        ComputeScope scope(Options{});
        if (benchmark) old_ms = measure(stream, [&] { run(sa, da); });
        else run(sa, da);
    }
    {
        Options options; options.kda_register_state = true;
        ComputeScope scope(options);
        if (benchmark) new_ms = measure(stream, [&] { run(sb, dout); });
        else run(sb, dout);
    }
    check(cudaStreamSynchronize(stream), "KDA register test sync");
    check(cudaMemcpy(a.data(), da.get(), a.size() * 4, cudaMemcpyDeviceToHost), "KDA reference output");
    check(cudaMemcpy(b.data(), dout.get(), b.size() * 4, cudaMemcpyDeviceToHost), "KDA register output");
    for (float value : b) require(std::isfinite(value), "KDA produced non-finite output");
    require(std::memcmp(a.data(), b.data(), a.size() * 4) == 0, "register KDA changed output bytes or guards");
    std::vector<float> state_a(initial.size()), state_b(initial.size());
    check(cudaMemcpy(state_a.data(), sa.get(), state_a.size() * 4, cudaMemcpyDeviceToHost), "KDA reference state");
    check(cudaMemcpy(state_b.data(), sb.get(), state_b.size() * 4, cudaMemcpyDeviceToHost), "KDA register state");
    for (float value : state_b) require(std::isfinite(value), "KDA produced non-finite state");
    require(std::memcmp(state_a.data(), state_b.data(), state_a.size() * 4) == 0, "register KDA changed final state bytes");
    if (heads == 1 && tokens == 7 && !stress) {
        std::vector<double> state(initial.begin(), initial.end()), reference(elements);
        const double scale = 1.0 / std::sqrt(static_cast<double>(width));
        for (int t = 0; t < tokens; ++t) {
            double attention = 0;
            for (int col = 0; col < width; ++col)
                attention += k[t * width + col] * (q[t * width + col] * scale);
            const double weight = 1 / (1 + std::exp(-static_cast<double>(beta[t])));
            for (int row = 0; row < width; ++row) {
                double prediction = 0, result = 0;
                for (int col = 0; col < width; ++col) {
                    const double decay = std::exp(std::min<double>(gate[t * width + col], 50));
                    const double s = state[row + col * width];
                    prediction += s * (k[t * width + col] * decay);
                    result += s * (q[t * width + col] * decay);
                }
                const double change = weight * (v[t * width + row] - prediction);
                reference[t * width + row] = result * scale + change * attention;
                for (int col = 0; col < width; ++col) {
                    const double decay = std::exp(std::min<double>(gate[t * width + col], 50));
                    state[row + col * width] = std::clamp(decay * state[row + col * width] +
                        change * k[t * width + col], -1e6, 1e6);
                }
            }
        }
        double error = 0, power = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            error = std::max(error, std::abs(b[guard + i] - reference[i]));
            power = std::max(power, std::abs(reference[i]));
        }
        require(error <= 1e-5 * power, "KDA register output disagrees with independent double reference");
        error = 0; power = 0;
        for (size_t i = 0; i < state.size(); ++i) {
            error = std::max(error, std::abs(state_b[i] - state[i]));
            power = std::max(power, std::abs(state[i]));
        }
        require(error <= 1e-5 * power, "KDA register state disagrees with independent double reference");
    }
    if (benchmark)
        std::printf("KDA compute: heads=%d tokens=%d global_ms=%.6f register_ms=%.6f ratio=%.3f exact=1\n",
                    heads, tokens, old_ms, new_ms, old_ms / new_ms);
}

void scopes() {
    Options options; options.kda_register_state = true; options.skip_unused_quant = true;
    {
        ComputeScope outer(options);
        require(gemv::skip_unused_quant() && strata::kernels::glm_kda_register_state(), "compute options not applied");
        {
            ComputeScope inner(Options{});
            require(!gemv::skip_unused_quant() && !strata::kernels::glm_kda_register_state(), "nested compute options leaked");
        }
        require(gemv::skip_unused_quant() && strata::kernels::glm_kda_register_state(), "compute scope not restored");
    }
    require(!gemv::skip_unused_quant() && !strata::kernels::glm_kda_register_state(), "compute options leaked");
    try {
        ComputeScope scope(options);
        throw std::runtime_error("scope unwind test");
    } catch (const std::runtime_error&) {}
    require(!gemv::skip_unused_quant() && !strata::kernels::glm_kda_register_state(), "compute options leaked on error");
}

void scratch_layout() {
    Geometry g;
    for (int tokens : {1, 17}) {
        const auto reference_bytes = glm_buffers_bytes(g, tokens, 96);
        Options optimized; optimized.skip_unused_quant = true;
        ComputeScope scope(optimized);
        const auto compact_bytes = glm_buffers_bytes(g, tokens, 96);
        require(compact_bytes < reference_bytes, "unused images must release scratch memory");
        Buffer storage(compact_bytes);
        GlmBuffers buffers;
        require(glm_buffers_init(g, tokens, 96, storage.get(), buffers) == compact_bytes,
                "compact scratch carving differs from its admitted byte count");
        require(!buffers.normed_q8k && !buffers.normed_q8_0 && !buffers.cur_q8k &&
                !buffers.cur_q8_0 && !buffers.wide_q8k && !buffers.wide_q8_0 &&
                !buffers.tail_q8_0 && !buffers.heads_q8k && !buffers.heads_q8_0,
                "compact scratch retained unconsumed quantization images");
        require(buffers.normed_bf16 && buffers.cur_bf16 && buffers.wide_bf16 &&
                buffers.heads_in && buffers.band_out, "compact scratch removed live projection inputs");
    }
    for (int tokens : {1, 17, 2048}) {
        Options token; token.skip_unused_quant = true; token.flash_attn = true;
        token.mla_attention_layout = MlaAttentionLayout::TokenMajor;
        uint64_t token_bytes = 0;
        { ComputeScope scope(token); token_bytes = glm_buffers_bytes(g, tokens, 96); }
        auto head = token; head.mla_attention_layout = MlaAttentionLayout::HeadMajor;
        ComputeScope scope(head);
        const auto head_bytes = glm_buffers_bytes(g, tokens, 96);
        require(token_bytes - head_bytes == 2ULL * tokens * g.n_head * g.kv_lora_rank * sizeof(float),
                "direct attention must release exactly its two token-major images");
        if (tokens <= 17) {
            Buffer storage(head_bytes);
            GlmBuffers buffers;
            require(glm_buffers_init(g, tokens, 96, storage.get(), buffers) == head_bytes &&
                    !buffers.qabs && !buffers.kqv && buffers.heads_in && buffers.band_out,
                    "direct attention scratch sizing/carving or live buffers disagree");
        }
        head.flash_attn = false;
        ComputeScope no_flash(head);
        require(glm_buffers_bytes(g, tokens, 96) == token_bytes,
                "head-major selection without Flash must retain scalar attention scratch");
    }
}

void policy() {
    Geometry g;
    g.kda_head_dim = 128; g.idx_q_heads = 32; g.idx_key_dim = 128; g.idx_top_k = 2048; g.idx_kpool = 4;
    Options base; base.full_layer_prefill = true; base.dsa = true;
    require(base.gpu_experts_mib == default_gpu_expert_budget && base.gpu_reserve_mib == 1024,
            "default cache policy must remain distinguishable from an explicit off override");
    auto measured = g;
    measured.n_embd = 4096; measured.n_ff = 2048; measured.n_expert = 288; measured.n_expert_used = 8;
    require(select_compute_options(resolve_options(base), measured, 86).gpu_experts_mib == 0,
            "measured SM86 geometry must default to automatic expert capacity");
    for (int arch : {0, 80, 89, 90})
        require(select_compute_options(resolve_options(base), measured, arch).gpu_experts_mib == -1,
                "unvalidated backend must retain disabled default cache");
    auto unsupported_cache = measured; unsupported_cache.n_ff = 768;
    require(select_compute_options(resolve_options(base), unsupported_cache, 86).gpu_experts_mib == -1,
            "unvalidated model geometry must retain disabled default cache");
    for (int64_t cap : {-1, 0, 4096}) {
        auto explicit_cache = base; explicit_cache.gpu_experts_mib = cap;
        require(select_compute_options(resolve_options(explicit_cache), measured, 86).gpu_experts_mib == cap &&
                select_compute_options(resolve_options(explicit_cache), measured, 89).gpu_experts_mib == cap,
                "explicit cache budget must take precedence on every backend");
    }
    auto reference_cache = base; reference_cache.reference = true;
    require(resolve_execution_options(reference_cache, measured).gpu_experts_mib == -1,
            "reference default must not silently enable GPU cache");
    reference_cache.gpu_experts_mib = 0;
    require(resolve_execution_options(reference_cache, measured).gpu_experts_mib == 0,
            "reference must preserve an explicit automatic cache request");
    auto disabled_features = base; disabled_features.reference = true; disabled_features.gpu_check = true;
    bool rejected_features = false;
    try { resolve_execution_options(disabled_features, measured); }
    catch (const std::invalid_argument&) { rejected_features = true; }
    require(rejected_features, "cache diagnostics must reject a disabled resolved default");
    const auto automatic = select_compute_options(resolve_options(base), g, 86);
    require(automatic.skip_unused_quant && automatic.kda_register_state && automatic.prefill_expert_rows &&
            automatic.prefill_transfer_cost &&
            automatic.dsa_score_load == DsaScoreLoad::Prefetch &&
            automatic.dsa_parallel_topk && automatic.dsa_score == DsaScore::Tiled &&
            automatic.gpu_expert_mode == GpuExpertMode::Single,
            "validated SM86 geometry must automatically select all successful compute paths");
    auto flash = base; flash.flash_attn = true;
    require(select_compute_options(resolve_options(flash), g, 86).mla_attention_layout == MlaAttentionLayout::HeadMajor,
            "validated Flash geometry must select direct head-major attention");
    require(select_compute_options(resolve_options(flash), g, 86).flash_kv_load == FlashKvLoad::Prefetch,
            "validated Flash geometry must select asynchronous KV prefetch");
    auto direct_kv = flash; direct_kv.flash_kv_load = FlashKvLoad::Direct;
    require(select_compute_options(resolve_options(direct_kv), g, 86).flash_kv_load == FlashKvLoad::Direct,
            "automatic KV load mode must honor an isolated direct-load comparison");
    auto direct_score = flash; direct_score.dsa_score_load = DsaScoreLoad::Direct;
    require(select_compute_options(resolve_options(direct_score), g, 86).dsa_score_load == DsaScoreLoad::Direct,
            "automatic score load mode must honor an isolated direct-load comparison");
    auto explicit_token = flash; explicit_token.mla_attention_layout = MlaAttentionLayout::TokenMajor;
    require(select_compute_options(resolve_options(explicit_token), g, 86).mla_attention_layout ==
                MlaAttentionLayout::TokenMajor, "automatic layout must honor a private token-major comparison");
    auto reference_layout = flash; reference_layout.reference = true;
    reference_layout.mla_attention_layout = MlaAttentionLayout::HeadMajor;
    reference_layout.flash_kv_load = FlashKvLoad::Prefetch;
    reference_layout.dsa_score_load = DsaScoreLoad::Prefetch;
    const auto resolved_reference = resolve_execution_options(reference_layout, g);
    require(resolved_reference.mla_attention_layout == MlaAttentionLayout::TokenMajor &&
            resolved_reference.flash_kv_load == FlashKvLoad::Direct,
            "reference mode must retain token-major attention storage and direct KV loads");
    require(resolved_reference.dsa_score_load == DsaScoreLoad::Direct, "reference mode must retain direct score loads");
    auto invalid_kv = flash; invalid_kv.flash_kv_load = static_cast<FlashKvLoad>(99);
    bool refused = false;
    try { resolve_execution_options(invalid_kv, g); } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "invalid KV load mode was accepted");
    auto invalid_score = flash; invalid_score.dsa_score_load = static_cast<DsaScoreLoad>(99);
    refused = false;
    try { resolve_execution_options(invalid_score, g); } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "invalid score load mode was accepted");
    for (int arch : {0, 70, 75, 80, 89, 90, 120}) {
        const auto fallback = select_compute_options(resolve_options(base), g, arch);
        require(!fallback.skip_unused_quant && !fallback.kda_register_state && !fallback.prefill_expert_rows &&
                !fallback.prefill_transfer_cost &&
                fallback.dsa_score_load == DsaScoreLoad::Automatic &&
                !fallback.dsa_parallel_topk && fallback.dsa_score == DsaScore::Automatic &&
                fallback.gpu_expert_mode == GpuExpertMode::Automatic,
                "unvalidated devices must retain the previous compute path");
        require(select_compute_options(resolve_options(flash), g, arch).mla_attention_layout ==
                    MlaAttentionLayout::Automatic, "unvalidated devices must not select head-major Flash");
        require(select_compute_options(resolve_options(flash), g, arch).flash_kv_load == FlashKvLoad::Automatic,
                "unvalidated devices must not select KV prefetch");
    }
    auto reference = base; reference.reference = true;
    const auto plain = select_compute_options(resolve_options(reference), g, 86);
    require(!plain.skip_unused_quant && !plain.kda_register_state && !plain.prefill_expert_rows &&
            !plain.prefill_transfer_cost &&
            !plain.dsa_parallel_topk && plain.dsa_score == DsaScore::Automatic &&
            !plain.prefill_auto && !plain.prefill_pipeline && plain.prefill == 1,
            "reference mode must disable automatic compute and scheduling choices");
    auto private_reference = base; private_reference.gpu_expert_mode = GpuExpertMode::Reference;
    require(select_compute_options(resolve_options(private_reference), g, 86).gpu_expert_mode == GpuExpertMode::Reference,
            "automatic policy must preserve an isolated GPU expert reference override");
    reference.gpu_expert_mode = GpuExpertMode::Single;
    reference.prefill_transfer_cost = true;
    require(resolve_execution_options(reference, g).gpu_expert_mode == GpuExpertMode::Reference,
            "global reference mode must disable direct single-token experts");
    require(!resolve_execution_options(reference, g).prefill_transfer_cost,
            "global reference mode must retain capacity-only grouping");
    reference.prefill = 2048;
    require(resolve_options(reference).prefill == 2048, "reference mode must preserve an explicit batch size");
    auto smaller = g; smaller.kda_head_dim = 64; smaller.idx_q_heads = 4;
    const auto fallback = select_compute_options(resolve_options(base), smaller, 86);
    require(!fallback.kda_register_state && fallback.dsa_score == DsaScore::Automatic &&
            fallback.dsa_parallel_topk, "automatic kernels must respect geometry restrictions independently");
    smaller = g; smaller.idx_top_k = 4096;
    require(!select_compute_options(resolve_options(base), smaller, 86).dsa_parallel_topk,
            "oversized top-k must retain the reference selector");
    auto scalar = base; scalar.dsa_score = DsaScore::Scalar;
    require(select_compute_options(resolve_options(scalar), g, 86).dsa_score == DsaScore::Scalar,
            "automatic selection must honor a diagnostic scorer override");
    base.dsa = false;
    const auto dense = select_compute_options(resolve_options(base), g, 86);
    require(dense.dsa_score == DsaScore::Automatic && !dense.dsa_parallel_topk,
            "automatic compute selection must not enable sparse attention");
    require(dense.mla_attention_layout == MlaAttentionLayout::Automatic,
            "layout selection must not enable Flash when it was not requested");
    require(dense.flash_kv_load == FlashKvLoad::Automatic, "KV prefetch must not enable Flash");
    require(dense.dsa_score_load == DsaScoreLoad::Automatic, "score prefetch must not enable DSA");
    auto other_score = g; other_score.idx_q_heads = 16;
    require(select_compute_options(resolve_options(flash), other_score, 86).dsa_score_load == DsaScoreLoad::Automatic,
            "score prefetch must respect indexer geometry");
    auto different_flash = g; different_flash.kv_lora_rank = 256;
    require(select_compute_options(resolve_options(flash), different_flash, 86).flash_kv_load == FlashKvLoad::Automatic,
            "KV prefetch must respect its geometry restriction");
}
}

int main(int argc, char** argv) {
    cudaStream_t stream = nullptr;
    try {
        const bool benchmark = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
        const bool kernel_check = argc == 2 && std::strcmp(argv[1], "--kernel-check") == 0;
        if (argc != 1 && !benchmark && !kernel_check)
            throw std::invalid_argument("Usage: glm5_compute_test [--bench | --kernel-check]");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "compute test stream");
        scopes();
        scratch_layout();
        policy();
        if (kernel_check) {
            kda(2, 7, false, stream);
            kda(2, 32, false, stream);
            kda(2, 7, true, stream);
        } else if (benchmark) {
            kda(64, 2048, false, stream, true);
        } else {
            for (int heads : {1, 2, 64})
                for (int tokens : {1, 3, 7, 32, 129}) kda(heads, tokens, false, stream);
            kda(2, 7, true, stream);
            kda(2, 2048, false, stream);
            Buffer dummy(4);
            Options options; options.kda_register_state = true;
            ComputeScope scope(options);
            auto* p = static_cast<float*>(dummy.get());
            for (int width : {64, 256, 512}) {
                bool refused = false;
                try { strata::kernels::glm_kda_delta(p, p, p, p, p, p, p, width, 1, 1, stream); }
                catch (const std::invalid_argument&) { refused = true; }
                require(refused, "register KDA must reject unsupported head dimensions");
            }
        }
        check(cudaStreamDestroy(stream), "compute test stream destroy");
        std::puts("glm5 compute: exact KDA states/outputs, clamp stress and scoped policies passed");
        return 0;
    } catch (const std::exception& e) {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        std::fprintf(stderr, "glm5 compute: %s\n", e.what());
        return 1;
    }
}
