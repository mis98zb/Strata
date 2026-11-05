#include "decode.hpp"
#include "prefill.hpp"
#include "runtime_fixture.hpp"
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>

namespace {
using glm_fixture::require;
using glm_fixture::near;

void compare_cache(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const char* message) {
    glm_fixture::near_fp16(a, b, 0.05f, message);
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    const auto path = std::filesystem::temp_directory_path() /
        ("glm5-batch-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    try {
        using namespace strata::glm5;
        glm_fixture::write_model(path, false);
        ModelFile file(path.string());
        std::vector<int32_t> tokens(2049);
        for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = 1 + i % 17;
        Options small;
        small.full_layer_prefill = true;
        small.prefill_reference = true;
        small.prefill = 8; small.max_context = 2080; small.mtp = 3;
        small.prefill_gpu_mib = 256; small.prefill_host_mib = 64;
        small.gpu_reserve_mib = 128; small.prefill_reserve_mib = 128;
        const auto reference = full_layer_prefill(file, small, tokens, {});
        auto large = small;
        large.prefill = 2048;
        const auto actual = full_layer_prefill(file, large, tokens, {});
        auto defaults = large;
        defaults.prefill_reference = false;
        defaults.prefill = 0;
        const auto automatic = full_layer_prefill(file, defaults, tokens, {});
        require(automatic.stats.selected_batch == 2048 && automatic.stats.max_matrix_columns == 2048 &&
                automatic.stats.transfer_peak_bytes > 0,
                "full-layer alone must admit actual 2048-column products when the budget fits");
        near(automatic.state.logits, actual.state.logits, 1e-5f, "default scheduling at batch 2048");
        require(actual.stats.max_matrix_columns == 2048 && actual.stats.f32_gemm_calls > 0 &&
                actual.stats.f16_gemm_calls > 0, "prefill did not run real 2048-column products");
#ifdef STRATA_PREFILL_MMQ
        require(actual.stats.mmq_calls > 0, "large prefill bypassed the MMQ backend");
#endif
        near(actual.state.logits, reference.state.logits, 0.05f, "2048+tail target logits");
        near(actual.state.carry_hidden, reference.state.carry_hidden, 0.05f, "2048-token hidden carry");
        for (size_t l = 0; l < actual.state.layers.size(); ++l) {
            const auto& a = actual.state.layers[l];
            const auto& b = reference.state.layers[l];
            if (!a.recurrent.empty()) near(a.recurrent, b.recurrent, 0.05f, "2048-token KDA/conv state");
            else compare_cache(a.kv, b.kv, "2048-token MLA state");
        }
        compare_cache(actual.state.mtp.kv, reference.state.mtp.kv, "2048-token NextN warmup");
        {
            DecodeExecutor decode(file, large, &actual.state);
            DecodeExecutor baseline(file, small, &reference.state);
            require(decode.generate(4) == baseline.generate(4), "large prefill handoff changed greedy output");
            require(decode.mtp_stats().verify_batches > 0, "MTP verification was not exercised after large prefill");
            near(decode.token(7), baseline.token(7), 0.05f, "large prefill/decode continuation");
        }
        std::printf("glm5 large prefill: 2048+1 tokens, max_columns=%llu MMQ=%llu FP16=%llu F32=%llu; state handoff passed\n",
                    static_cast<unsigned long long>(actual.stats.max_matrix_columns),
                    static_cast<unsigned long long>(actual.stats.mmq_calls),
                    static_cast<unsigned long long>(actual.stats.f16_gemm_calls),
                    static_cast<unsigned long long>(actual.stats.f32_gemm_calls));
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(path);
        std::fprintf(stderr, "glm5 large prefill test: %s\n", e.what());
        return 1;
    }
}
