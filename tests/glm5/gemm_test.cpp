#include "batch_gemm.hpp"
#include "strata/glm5/glm.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
using namespace strata::glm5;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

void matrix(BatchGemm& gemm, int tokens, int rows, int width, int type, void* stream) {
    std::vector<float> x(static_cast<size_t>(tokens) * width), weights(static_cast<size_t>(rows) * width);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<int>(i * 17 % 101) - 50) * 0.01f;
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = (static_cast<int>(i * 7 % 29) - 14) * 0.005f;
    std::vector<uint8_t> quantized;
    if (type >= 0) {
        const auto* cpu = ggml_get_type_traits_cpu(static_cast<ggml_type>(type));
        const auto* trait = ggml_get_type_traits(static_cast<ggml_type>(type));
        require(cpu && cpu->from_float && trait && trait->to_float, "missing reference quantizer");
        quantized.resize(ggml_row_size(static_cast<ggml_type>(type), weights.size()));
        cpu->from_float(weights.data(), quantized.data(), weights.size());
        trait->to_float(quantized.data(), weights.data(), weights.size());
    }
    Buffer input(x.size() * sizeof(float)), w(type < 0 ? weights.size() * sizeof(float) : quantized.size());
    const size_t count = static_cast<size_t>(tokens) * rows;
    std::vector<float> result(count + 32, 123.5f);
    Buffer output(result.size() * sizeof(float));
    check(cudaMemcpy(input.get(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "GEMM test input");
    check(cudaMemcpy(w.get(), type < 0 ? static_cast<const void*>(weights.data()) : quantized.data(),
                      type < 0 ? weights.size() * sizeof(float) : quantized.size(), cudaMemcpyHostToDevice), "GEMM test weight");
    check(cudaMemcpy(output.get(), result.data(), result.size() * sizeof(float), cudaMemcpyHostToDevice), "GEMM test guard");
    auto* y = static_cast<float*>(output.get()) + 16;
    if (type < 0) gemm.f32(static_cast<const float*>(w.get()), static_cast<const float*>(input.get()), y, tokens, rows, width);
    else gemm.native(type, w.get(), static_cast<const float*>(input.get()), y, tokens, rows, width);
    check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "GEMM test sync");
    check(cudaMemcpy(result.data(), output.get(), result.size() * sizeof(float), cudaMemcpyDeviceToHost), "GEMM test result");
    for (size_t i = 0; i < 16; ++i)
        require(result[i] == 123.5f && result[count + 16 + i] == 123.5f, "GEMM wrote outside its output matrix");
    double error = 0, scale = 1e-9;
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < rows; ++r) {
            double expected = 0;
            for (int k = 0; k < width; ++k)
                expected += static_cast<double>(x[static_cast<size_t>(t) * width + k]) * weights[r * width + k];
            const auto actual = result[16 + static_cast<size_t>(t) * rows + r];
            require(std::isfinite(actual), "GEMM produced non-finite output");
            error = std::max(error, std::abs(actual - expected));
            scale = std::max(scale, std::abs(expected));
        }
    const double tolerance = type < 0 ? 1e-5 : (width % 256 ? 0.002 : 0.02);
    if (error > tolerance * scale)
        throw std::runtime_error("GEMM reference mismatch: T=" + std::to_string(tokens) +
                                 " type=" + std::to_string(type) + " relative max=" + std::to_string(error / scale));
}

void transpose_and_router(void* opaque) {
    const auto stream = static_cast<cudaStream_t>(opaque);
    constexpr int heads = 64, tokens = 2048, width = 8, experts = 8, top = 3;
    const size_t count = static_cast<size_t>(heads) * tokens * width;
    std::vector<float> input(count), actual(count);
    for (size_t i = 0; i < count; ++i) input[i] = static_cast<float>(i);
    Buffer a(count * sizeof(float)), b(count * sizeof(float));
    check(cudaMemcpy(a.get(), input.data(), count * sizeof(float), cudaMemcpyHostToDevice), "transpose input");
    strata::kernels::glm_heads_major(static_cast<const float*>(a.get()), static_cast<float*>(b.get()),
                                     heads, tokens, width, stream);
    check(cudaMemcpyAsync(actual.data(), b.get(), count * sizeof(float), cudaMemcpyDeviceToHost, stream), "transpose result");
    check(cudaStreamSynchronize(stream), "transpose sync");
    for (int h = 0; h < heads; ++h)
        for (int t = 0; t < tokens; ++t)
            for (int k = 0; k < width; ++k)
                require(actual[(static_cast<size_t>(h) * tokens + t) * width + k] ==
                        input[(static_cast<size_t>(t) * heads + h) * width + k], "2048-token head transpose");
    strata::kernels::glm_heads_major(static_cast<const float*>(b.get()), static_cast<float*>(a.get()),
                                     tokens, heads, width, stream);
    check(cudaMemcpyAsync(actual.data(), a.get(), count * sizeof(float), cudaMemcpyDeviceToHost, stream), "transpose inverse");
    check(cudaStreamSynchronize(stream), "transpose inverse sync");
    require(actual == input, "head transpose round trip");
    std::vector<float> logits(tokens * experts), weights(tokens * top);
    std::vector<int32_t> ids(tokens * top);
    for (int t = 0; t < tokens; ++t)
        for (int e = 0; e < experts; ++e) logits[t * experts + e] = ((e + t) % experts) * 0.1f;
    Buffer l(logits.size() * sizeof(float)), w(weights.size() * sizeof(float)), i(ids.size() * sizeof(int32_t));
    check(cudaMemcpy(l.get(), logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "router input");
    strata::kernels::glm_router_sigmoid_topk(static_cast<const float*>(l.get()), nullptr, static_cast<int32_t*>(i.get()),
                                             static_cast<float*>(w.get()), experts, top, 2.5f, stream, tokens);
    check(cudaMemcpyAsync(ids.data(), i.get(), ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream), "router ids");
    check(cudaMemcpyAsync(weights.data(), w.get(), weights.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "router weights");
    check(cudaStreamSynchronize(stream), "router sync");
    for (int t = 0; t < tokens; ++t) {
        double sum = 0;
        for (int z = 0; z < top; ++z) sum += 1.0 / (1 + std::exp(-(7 - z) * 0.1));
        for (int z = 0; z < top; ++z) {
            require(ids[t * top + z] == (7 - z - t % experts + experts) % experts, "batched router token identity");
            const double expected = 2.5 / ((1 + std::exp(-(7 - z) * 0.1)) * sum);
            require(std::abs(weights[t * top + z] - expected) < 1e-6, "batched router normalized weight");
        }
    }
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaStream_t stream = nullptr;
    try {
        ggml_cpu_init();
        check(cudaStreamCreate(&stream), "GEMM test stream");
        {
            Geometry g; g.n_embd = 256; g.n_head = 2; g.n_ff = 768; g.n_ff_dense = 256; g.q_lora_rank = 256;
            MemoryBudget budget(64 * 1048576ULL);
            {
                BatchGemm ragged(g, 17, stream, &budget);
                matrix(ragged, 17, 128, 256, GGML_TYPE_Q8_0, stream);
                require(ragged.stats().max_columns == 17, "ragged-capacity GEMM dispatch");
            }
            BatchGemm gemm(g, 4096, stream, &budget);
            matrix(gemm, 2048, 24, 256, GGML_TYPE_Q8_0, stream);
            matrix(gemm, 4096, 24, 256, GGML_TYPE_Q8_0, stream);
            matrix(gemm, 2048, 128, 256, GGML_TYPE_Q4_K, stream);
            matrix(gemm, 2048, 17, 128, GGML_TYPE_Q8_0, stream);
            matrix(gemm, 2048, 8193, 128, GGML_TYPE_Q8_0, stream);
            matrix(gemm, 2048, 17, 256, -1, stream);
            matrix(gemm, 4096, 17, 256, -1, stream);
            require(gemm.stats().max_columns == 4096 && gemm.stats().f32_calls >= 2 &&
                    gemm.stats().f16_calls > 0, "large GEMM dispatch did not execute");
#ifdef STRATA_PREFILL_MMQ
            const char* kq = std::getenv("STRATA_MMQ_KQ");
            const int expected_mmq = kq && kq[0] == '1' ? 3 : 2;
            if (gemm.stats().mmq_calls < static_cast<uint64_t>(expected_mmq))
                throw std::runtime_error("2048/4096 columns did not reach actual MMQ products: MMQ=" +
                                         std::to_string(gemm.stats().mmq_calls) + " FP16=" +
                                         std::to_string(gemm.stats().f16_calls));
#endif
            transpose_and_router(stream);
            std::printf("glm5 GEMM: max columns=%llu, MMQ=%llu FP16=%llu F32=%llu; transpose/router passed\n",
                        static_cast<unsigned long long>(gemm.stats().max_columns),
                        static_cast<unsigned long long>(gemm.stats().mmq_calls),
                        static_cast<unsigned long long>(gemm.stats().f16_calls),
                        static_cast<unsigned long long>(gemm.stats().f32_calls));
        }
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception& e) {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        std::fprintf(stderr, "glm5 GEMM test: %s\n", e.what());
        return 1;
    }
}
