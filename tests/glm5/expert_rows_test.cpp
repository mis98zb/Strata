#include "batch_gemm.hpp"
#include "compute_scope.hpp"
#include "expert_compute.hpp"
#include "runtime_fixture.hpp"
#include "strata/glm5/glm.hpp"
#include "weights.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
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
            {
                ComputeScope reference_scope(Options{});
                computer.run(bank.expert(3, 1), static_cast<float*>(di.get()), entries, static_cast<float*>(dr.get()), stream);
            }
            const auto reused_before = gemm ? gemm->stats().reused_quant_calls : 0;
            {
                Options optimized; optimized.skip_unused_quant = true;
                ComputeScope optimized_scope(optimized);
                computer.run(bank.expert(3, 1), static_cast<float*>(di.get()), entries, static_cast<float*>(da.get()), stream,
                              static_cast<int32_t*>(dm.get()));
            }
            check(cudaStreamSynchronize(stream), "expert row arithmetic sync");
            check(cudaMemcpy(reference.data(), dr.get(), reference.size() * 4, cudaMemcpyDeviceToHost), "expert row reference result");
            check(cudaMemcpy(actual.data(), da.get(), actual.size() * 4, cudaMemcpyDeviceToHost), "expert row actual result");
            require(reference == actual, "batched expert rows changed arithmetic or unselected output rows");
#ifdef STRATA_PREFILL_MMQ
            if (gemm) {
                uint64_t expected_reuses = 0;
                for (int first = 0; first < count; first += capacity)
                    expected_reuses += std::min(capacity, count - first) > 8;
                require(gemm->stats().reused_quant_calls == reused_before + expected_reuses,
                        "same-format gate/up must reuse one native quantization per large expert batch");
            }
#else
            (void) reused_before;
#endif
        }
    }
}

void single_experts(const ModelFile& file, cudaStream_t stream) {
    const auto& g = file.geometry();
    WeightBank bank(file, [](const auto& t) { return t.name.starts_with("blk.3."); });
    ExpertComputer computer(g);
    constexpr int guard = 16, rows = 19;
    std::vector<float> input(g.n_embd), reference(rows * g.n_embd + 2 * guard, 12345), actual(reference);
    Buffer di(input.size() * sizeof(float)), dr(reference.size() * sizeof(float)), da(actual.size() * sizeof(float));
    for (int pass = 0; pass < 3; ++pass) {
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = std::sin(i * 0.173f + pass) * (pass == 2 ? 10000.0f : 1.0f);
        check(cudaMemcpyAsync(di.get(), input.data(), input.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream), "single expert input");
        check(cudaMemcpyAsync(dr.get(), reference.data(), reference.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream), "single expert reference guard");
        check(cudaMemcpyAsync(da.get(), reference.data(), reference.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream), "single expert output guard");
        for (int row : {0, 7, 18}) {
            const std::vector<ExpertMiss> entries{{0, row, row % static_cast<int>(g.n_expert)}};
            computer.run(bank.expert(3, entries.front().expert), static_cast<float*>(di.get()), entries,
                         static_cast<float*>(dr.get()) + guard, stream);
        }
        const auto* prepared = computer.prepare_single_input(static_cast<float*>(di.get()), stream);
        for (int row : {0, 7, 18})
            computer.run_single(bank.expert(3, row % static_cast<int>(g.n_expert)), prepared,
                                static_cast<float*>(da.get()) + guard + row * g.n_embd, stream);
        check(cudaMemcpyAsync(reference.data(), dr.get(), reference.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream), "single expert reference");
        check(cudaMemcpyAsync(actual.data(), da.get(), actual.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream), "single expert result");
        check(cudaStreamSynchronize(stream), "single expert comparison sync");
        for (const auto value : actual) require(std::isfinite(value), "single expert produced non-finite output");
        require(std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) == 0,
                "direct expert changed output bytes, guards or unselected slots");
    }
}

void single_cache(const ModelFile& file, cudaStream_t stream) {
    const auto& g = file.geometry();
    for (int stage : {0, 1}) {
        GpuExperts reference(file, g.n_layers + 1, 2, 128, {}, false, stage, false);
        GpuExperts direct(file, g.n_layers + 1, 2, 128, {}, false, stage, true);
        uint64_t expected_direct = 0, expected_preparations = 0, expected_reuse = 0;
        for (int tokens : {1, 3, 1}) {
            std::vector<float> input(tokens * g.n_embd), a(tokens * g.n_expert_used * g.n_embd, 12345), b(a);
            std::vector<int32_t> ids(tokens * g.n_expert_used);
            Buffer di(input.size() * sizeof(float)), da(a.size() * sizeof(float)), db(b.size() * sizeof(float));
            if (tokens == 1 && direct.stats().hits == 0) {
                for (size_t i = 0; i < ids.size(); ++i) ids[i] = i % g.n_expert;
                const auto& left = reference.run(g.n_layer_dense_lead, static_cast<float*>(di.get()), ids.data(),
                                                  1, static_cast<float*>(da.get()), false, stream);
                const auto& right = direct.run(g.n_layer_dense_lead, static_cast<float*>(di.get()), ids.data(),
                                               1, static_cast<float*>(db.get()), false, stream);
                require(left.size() == ids.size() && right.size() == ids.size() &&
                        direct.stats().single_input_quantizations == 0 && direct.stats().direct_rows == 0,
                        "an all-miss dispatch must not prepare unused GPU activations");
            }
            for (int pass = 0; pass < 8; ++pass) {
                for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(i * 0.173f + pass);
                for (size_t i = 0; i < ids.size(); ++i) ids[i] = pass % 3 == 2 ? 0 : (i + pass) % g.n_expert;
                check(cudaMemcpyAsync(di.get(), input.data(), input.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream), "single cache input");
                check(cudaMemcpyAsync(da.get(), a.data(), a.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream), "single cache reference");
                check(cudaMemcpyAsync(db.get(), a.data(), a.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream), "single cache initial");
                const int layer = g.n_layer_dense_lead + pass % 2;
                const auto before = direct.stats();
                const auto& left = reference.run(layer, static_cast<float*>(di.get()), ids.data(), tokens,
                                                  static_cast<float*>(da.get()), pass % 2 == 0, stream);
                const auto& right = direct.run(layer, static_cast<float*>(di.get()), ids.data(), tokens,
                                               static_cast<float*>(db.get()), pass % 2 == 0, stream);
                require(left.size() == right.size(), "direct dispatch changed CPU miss count");
                for (size_t i = 0; i < left.size(); ++i)
                    require(left[i].row == right[i].row && left[i].token == right[i].token &&
                            left[i].expert == right[i].expert, "direct dispatch changed CPU miss ordering");
                const auto& after = direct.stats();
                const auto computed = after.hits + after.streamed - before.hits - before.streamed;
                if (tokens == 1 && pass % 3 != 2 && computed) {
                    expected_direct += computed; ++expected_preparations;
                    expected_reuse += computed - 1;
                }
                reference.admit(layer, ids.data(), left, stream);
                direct.admit(layer, ids.data(), right, stream);
                check(cudaMemcpyAsync(a.data(), da.get(), a.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream), "single cache reference result");
                check(cudaMemcpyAsync(b.data(), db.get(), b.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream), "single cache direct result");
                check(cudaStreamSynchronize(stream), "single cache result sync");
                require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
                        "single cache changed products or miss slots");
                require(reference.stats().hits == after.hits && reference.stats().misses == after.misses &&
                        reference.stats().streamed == after.streamed &&
                        reference.stats().swaps == after.swaps &&
                        reference.stats().upload_bytes == after.upload_bytes,
                        "single dispatch changed cache placement or transfer counts");
            }
        }
        const auto& stats = direct.stats();
        require(stats.direct_rows == expected_direct && stats.single_input_quantizations == expected_preparations &&
                stats.avoided_input_quantizations == expected_reuse &&
                stats.avoided_d2d_bytes == expected_direct * 2 * g.n_embd * sizeof(float) && expected_reuse > 0,
                "single dispatch counters or multicolumn/duplicate fallback failed");
        direct.reset();
        require(direct.stats().direct_rows == 0 && direct.stats().single_input_quantizations == 0,
                "reset retained direct dispatch counters");
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
        {
            ModelFile file(path.string());
            experts(file, stream);
        }
        const std::array<std::array<uint32_t, 3>, 3> formats = {{
            {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
            {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K},
            {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}
        }};
        for (const auto& types : formats) {
            glm_fixture::write_model(path, false, false, 4, 768, types);
            ModelFile file(path.string());
            single_experts(file, stream);
            single_cache(file, stream);
        }
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
