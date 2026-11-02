#include "strata/glm5/glm.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::puts("glm5 head: skipped (no GPU)");
        return 77;
    }
    constexpr int width = 32, rows = 154880;
    float *weights = nullptr, *input = nullptr, *output = nullptr;
    cudaStream_t stream = nullptr;
    int result = 1;
    try {
        std::vector<float> host_weights(static_cast<size_t>(width) * rows), x(width, 1), y(rows);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < width; ++c) host_weights[r * width + c] = (1 + r % 3) * 0.125f;
        check(cudaMalloc(reinterpret_cast<void**>(&weights), host_weights.size() * sizeof(float)));
        check(cudaMalloc(reinterpret_cast<void**>(&input), x.size() * sizeof(float)));
        check(cudaMalloc(reinterpret_cast<void**>(&output), y.size() * sizeof(float)));
        check(cudaStreamCreate(&stream));
        check(cudaMemcpy(weights, host_weights.data(), host_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        strata::kernels::glm_f32_gemv(input, weights, output, width, rows, stream);
        check(cudaMemcpyAsync(y.data(), output, y.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        for (int r = 0; r < rows; ++r)
            if (y[r] != 4.0f * (1 + r % 3))
                throw std::runtime_error("wrong output at vocabulary row " + std::to_string(r));
        std::puts("glm5 head: all 154880 vocabulary rows passed");
        result = 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm5 head: %s\n", e.what());
    }
    if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    if (weights) cudaFree(weights);
    if (input) cudaFree(input);
    if (output) cudaFree(output);
    return result;
}
