#include "batch_gemm.hpp"
#include "expert_compute.hpp"
#include "runtime_fixture.hpp"
#include "strata/glm5/glm.hpp"
#include "weights.hpp"

#include <chrono>
#include <cstdio>
#include <numeric>

namespace {
using namespace strata::glm5;
using glm_fixture::require;

void movement(cudaStream_t stream) {
    for (const int width : {1, 17, 256, 4096}) {
        constexpr int rows = 19, tokens = 7, destinations = 31, guard = 16;
        std::vector<float> input(tokens * width), packed(rows * width + 2 * guard, -12345);
        std::vector<float> output(destinations * width + 2 * guard, -12345);
        std::vector<int32_t> routes(rows * 2);
        for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i) * 0.125f;
        for (int i = 0; i < rows; ++i) { routes[2 * i] = i % tokens; routes[2 * i + 1] = (i * 13) % destinations; }
        Buffer di(input.size() * 4), dp(packed.size() * 4), dout(output.size() * 4), dr(routes.size() * 4);
        check(cudaMemcpy(di.get(), input.data(), input.size() * 4, cudaMemcpyHostToDevice), "row test input");
        check(cudaMemcpy(dp.get(), packed.data(), packed.size() * 4, cudaMemcpyHostToDevice), "row test packed");
        check(cudaMemcpy(dout.get(), output.data(), output.size() * 4, cudaMemcpyHostToDevice), "row test output");
        check(cudaMemcpy(dr.get(), routes.data(), routes.size() * 4, cudaMemcpyHostToDevice), "row test routes");
        strata::kernels::glm_expert_gather(static_cast<float*>(di.get()), static_cast<int32_t*>(dr.get()),
                                          static_cast<float*>(dp.get()) + guard, rows, width, stream);
        strata::kernels::glm_expert_scatter(static_cast<float*>(dp.get()) + guard, static_cast<int32_t*>(dr.get()),
                                           static_cast<float*>(dout.get()) + guard, rows, width, stream);
        check(cudaStreamSynchronize(stream), "row test sync");
        check(cudaMemcpy(packed.data(), dp.get(), packed.size() * 4, cudaMemcpyDeviceToHost), "row test gathered");
        std::vector<float> actual(output.size());
        check(cudaMemcpy(actual.data(), dout.get(), actual.size() * 4, cudaMemcpyDeviceToHost), "row test scattered");
        for (int r = 0; r < rows; ++r)
            for (int d = 0; d < width; ++d) {
                const float expected = input[routes[2 * r] * width + d];
                require(packed[guard + r * width + d] == expected, "gather row or column mismatch");
                output[guard + routes[2 * r + 1] * width + d] = expected;
            }
        require(actual == output, "scatter changed unselected rows, guards or output bytes");
        for (int i = 0; i < guard; ++i)
            require(packed[i] == -12345 && packed[packed.size() - 1 - i] == -12345, "gather guard overwritten");
    }
}

void experts(const ModelFile& file, cudaStream_t stream) {
    const auto& g = file.geometry();
    WeightBank bank(file, [](const auto& t) { return t.name.starts_with("blk.3."); });
    for (int capacity : {8, 17, 64}) {
        MemoryBudget budget(128 * 1048576ULL);
        ExpertComputer computer(g, &budget, capacity);
        std::unique_ptr<BatchGemm> gemm;
        if (capacity > 8) gemm = std::make_unique<BatchGemm>(g, capacity, stream, &budget);
        BatchScope scope(gemm.get());
        for (int count : {0, 1, 7, 8, 17, 65, 129}) {
            std::vector<ExpertMiss> entries;
            std::vector<int32_t> routes(2 * std::max(1, count));
            const int output_rows = std::max(1, count * 2);
            std::vector<float> input(7 * g.n_embd), reference(output_rows * g.n_embd, -12345), actual(reference);
            for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(i * 0.17f);
            for (int r = 0; r < count; ++r) {
                entries.push_back({r % 7, 2 * r, 1});
                routes[2 * r] = r % 7; routes[2 * r + 1] = 2 * r;
            }
            Buffer di(input.size() * 4), dr(reference.size() * 4), da(actual.size() * 4), dm(routes.size() * 4);
            check(cudaMemcpy(di.get(), input.data(), input.size() * 4, cudaMemcpyHostToDevice), "expert row input");
            check(cudaMemcpy(dr.get(), reference.data(), reference.size() * 4, cudaMemcpyHostToDevice), "expert row reference");
            check(cudaMemcpy(da.get(), actual.data(), actual.size() * 4, cudaMemcpyHostToDevice), "expert row actual");
            check(cudaMemcpy(dm.get(), routes.data(), routes.size() * 4, cudaMemcpyHostToDevice), "expert row map");
            computer.run(bank.expert(3, 1), static_cast<float*>(di.get()), entries, static_cast<float*>(dr.get()), stream);
            computer.run(bank.expert(3, 1), static_cast<float*>(di.get()), entries, static_cast<float*>(da.get()), stream,
                          static_cast<int32_t*>(dm.get()));
            check(cudaStreamSynchronize(stream), "expert row arithmetic sync");
            check(cudaMemcpy(reference.data(), dr.get(), reference.size() * 4, cudaMemcpyDeviceToHost), "expert row reference result");
            check(cudaMemcpy(actual.data(), da.get(), actual.size() * 4, cudaMemcpyDeviceToHost), "expert row actual result");
            require(reference == actual, "batched expert rows changed arithmetic or unselected output rows");
        }
    }
}
}

int main() {
    cudaStream_t stream = nullptr;
    const auto path = std::filesystem::temp_directory_path() /
        ("glm5-expert-rows-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "expert rows stream");
        movement(stream);
        glm_fixture::write_model(path, false);
        ModelFile file(path.string());
        experts(file, stream);
        check(cudaStreamDestroy(stream), "expert rows destroy");
        std::filesystem::remove(path);
        std::puts("glm5 expert rows: guarded movement, repeated inputs, sparse outputs and exact expert arithmetic passed");
        return 0;
    } catch (const std::exception& e) {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        std::filesystem::remove(path);
        std::fprintf(stderr, "glm5 expert rows: %s\n", e.what());
        return 1;
    }
}
