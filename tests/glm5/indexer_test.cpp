#include "attention_scope.hpp"
#include "gpu_support.hpp"
#include "prefill.hpp"
#include "decode.hpp"
#include "runtime_fixture.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace {
using namespace strata::glm5;
namespace kernels = strata::kernels;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class T> void upload(Buffer& device, const std::vector<T>& values) {
    check(cudaMemcpy(device.get(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice), "indexer upload");
}

float measure(cudaStream_t stream, const std::function<void()>& operation, int repeats = 3) {
    operation();
    check(cudaStreamSynchronize(stream), "indexer warmup");
    // Timing needs events without cudaEventDisableTiming.
    cudaEvent_t start = nullptr, finish = nullptr;
    check(cudaEventCreate(&start), "indexer timing start");
    check(cudaEventCreate(&finish), "indexer timing finish");
    try {
        check(cudaEventRecord(start, stream), "indexer begin timing");
        for (int i = 0; i < repeats; ++i) operation();
        check(cudaEventRecord(finish, stream), "indexer end timing");
        check(cudaEventSynchronize(finish), "indexer timing sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start, finish), "indexer elapsed");
        cudaEventDestroy(start); cudaEventDestroy(finish);
        return ms / repeats;
    } catch (...) {
        cudaEventDestroy(start); cudaEventDestroy(finish);
        throw;
    }
}

void scores(int queries, int pools, cudaStream_t stream, bool benchmark = false) {
    constexpr int heads = 32, width = 128;
    std::vector<float> q(queries * heads * width), k(pools * width), weights(queries * heads);
    uint32_t seed = 1234567;
    auto next = [&] {
        seed = 1664525u * seed + 1013904223u;
        return (static_cast<int>(seed >> 8) - 8388608) * (1.0f / 8388608);
    };
    for (auto& x : q) x = next();
    for (auto& x : k) x = next();
    for (auto& x : weights) x = next();
    Buffer dq(q.size() * 4), dk(k.size() * 4), dw(weights.size() * 4);
    std::vector<float> reference(queries * pools + 32, 12345), actual(reference);
    Buffer dr(reference.size() * 4), da(actual.size() * 4);
    upload(dq, q); upload(dk, k); upload(dw, weights); upload(dr, reference); upload(da, actual);
    Options base; base.dsa = true; base.dsa_score = DsaScore::Parallel;
    auto run = [&](Buffer& output) {
        kernels::glm_dsa_score_batch(static_cast<float*>(dq.get()), static_cast<float*>(dk.get()),
            static_cast<float*>(dw.get()), width, heads, queries, pools, static_cast<float*>(output.get()) + 16, stream);
    };
    float old_ms = 0, new_ms = 0;
    {
        AttentionScope scope(base);
        if (benchmark) old_ms = measure(stream, [&] { run(dr); });
        else run(dr);
    }
    base.dsa_score = DsaScore::Tiled;
    {
        AttentionScope scope(base);
        if (benchmark) new_ms = measure(stream, [&] { run(da); });
        else run(da);
    }
    check(cudaStreamSynchronize(stream), "indexer score sync");
    check(cudaMemcpy(reference.data(), dr.get(), reference.size() * 4, cudaMemcpyDeviceToHost), "indexer reference scores");
    check(cudaMemcpy(actual.data(), da.get(), actual.size() * 4, cudaMemcpyDeviceToHost), "indexer tiled scores");
    require(std::memcmp(reference.data(), actual.data(), reference.size() * 4) == 0,
            "F32 tiled scorer must preserve complete score bytes and guards");
    if (benchmark)
        std::printf("indexer score: queries=%d pools=%d parallel_ms=%.6f tiled_ms=%.6f speedup=%.3f exact=1\n",
                    queries, pools, old_ms, new_ms, old_ms / new_ms);
}

void selection(int pools, int queries, int top, int tail, int pattern, cudaStream_t stream, bool benchmark = false) {
    constexpr int kpool = 4;
    const int selected = top * kpool + (tail ? kpool - 1 : 0) + 7;
    std::vector<float> score(static_cast<size_t>(queries) * std::max(1, pools));
    std::vector<int32_t> positions(queries);
    for (int t = 0; t < queries; ++t) {
        positions[t] = benchmark ? pools * kpool - 1 :
            (t == 0 ? 0 : t % 3 == 0 ? std::min(pools * kpool, 17) : pools * kpool - t % 4 - 1);
        positions[t] = std::max(0, positions[t]);
        for (int p = 0; p < pools; ++p) {
            float value = std::sin((p + 13 * t) * 0.173f);
            if (pattern == 1) value = static_cast<float>((p * 7 + t) % 19 - 9);
            if (pattern == 2) value = p % 2 ? -0.0f : 0.0f;
            if (pattern == 3 && p % 7 == 0)
                value = p % 2 ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
            score[static_cast<size_t>(t) * pools + p] = value;
        }
    }
    std::vector<int32_t> reference(queries * selected + 32, 12345), actual(reference);
    Buffer ds(score.size() * 4), dp(positions.size() * 4), dr(reference.size() * 4), da(actual.size() * 4);
    upload(ds, score); upload(dp, positions); upload(dr, reference); upload(da, actual);
    auto run = [&](Buffer& output) {
        kernels::glm_dsa_select(static_cast<float*>(ds.get()), pools, kpool, top, tail, queries, selected,
                                static_cast<int32_t*>(dp.get()), static_cast<int32_t*>(output.get()) + 16, stream);
    };
    Options base; base.dsa = true;
    float old_ms = 0, new_ms = 0;
    {
        AttentionScope scope(base);
        if (benchmark) old_ms = measure(stream, [&] { run(dr); });
        else run(dr);
    }
    base.dsa_parallel_topk = true;
    {
        AttentionScope scope(base);
        if (benchmark) new_ms = measure(stream, [&] { run(da); });
        else run(da);
    }
    check(cudaStreamSynchronize(stream), "indexer selection sync");
    check(cudaMemcpy(reference.data(), dr.get(), reference.size() * 4, cudaMemcpyDeviceToHost), "indexer reference selection");
    check(cudaMemcpy(actual.data(), da.get(), actual.size() * 4, cudaMemcpyDeviceToHost), "indexer parallel selection");
    require(reference == actual, "parallel top-k changed sorted cells, tie order, tail or padding");
    for (int t = 0; t < queries; ++t) {
        const int visible = (positions[t] + 1) / kpool;
        std::vector<int> order(visible);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return score[static_cast<size_t>(t) * pools + a] > score[static_cast<size_t>(t) * pools + b];
        });
        for (int i = 0; i < std::min(visible, top); ++i)
            for (int m = 0; m < kpool; ++m)
                require(actual[16 + t * selected + i * kpool + m] == order[i] * kpool + m,
                        "top-k disagrees with CPU stable sort");
    }
    if (benchmark)
        std::printf("indexer topk: queries=%d pools=%d scalar_ms=%.6f parallel_ms=%.6f speedup=%.3f exact=1\n",
                    queries, pools, old_ms, new_ms, old_ms / new_ms);
}

void runtime(bool reject) {
    const auto path = std::filesystem::temp_directory_path() /
        ("glm5-indexer-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    struct Cleanup {
        const std::filesystem::path& path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
            if (error) std::fprintf(stderr, "indexer fixture cleanup: %s\n", error.message().c_str());
        }
    } cleanup{path};
    glm_fixture::write_model(path, reject, true, 32);
    for (const int tail : {0, 1}) {
        ModelFile file(path.string(), tail);
        for (const int mtp : {0, 3}) {
            Options base;
            base.reference = true;
            base.full_layer_prefill = true; base.dsa = true; base.flash_attn = true;
            base.dsa_tail = tail; base.mtp = mtp; base.max_context = 2400; base.prefill = 64;
            base.prefill_gpu_mib = 64; base.prefill_reserve_mib = 128; base.gpu_reserve_mib = 128;
            std::vector<int32_t> tokens(2305);
            for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = 1 + i % 17;
            const auto reference = full_layer_prefill(file, base, tokens, {});
            auto fast = base;
            fast.dsa_score = DsaScore::Tiled; fast.dsa_parallel_topk = true;
            fast.prefill_expert_rows = true;
            fast.skip_unused_quant = true; fast.kda_register_state = true;
            const auto actual = full_layer_prefill(file, fast, tokens, {});
            auto equal = [&](const HostState& a, const HostState& b) {
                require(a.position == b.position && a.logits == b.logits && a.carry_hidden == b.carry_hidden,
                        "indexer policy changed logits or hidden bytes");
                for (size_t l = 0; l < a.layers.size(); ++l)
                    require(a.layers[l].kv == b.layers[l].kv && a.layers[l].indexer == b.layers[l].indexer &&
                            a.layers[l].recurrent == b.layers[l].recurrent, "indexer policy changed trunk state bytes");
                require(a.mtp.kv == b.mtp.kv && a.mtp.indexer == b.mtp.indexer,
                        "indexer policy changed NextN state bytes");
            };
            equal(actual.state, reference.state);
            auto automatic = base; automatic.reference = false;
            equal(full_layer_prefill(file, automatic, tokens, {}).state, actual.state);
            DecodeExecutor left(file, fast, &actual.state), right(file, base, &reference.state);
            require(left.generate(8) == right.generate(8), "optimized indexer changed greedy/MTP continuation");
            equal(left.export_state(), right.export_state());
            const std::vector<int32_t> suffix{3, 4, 5};
            const auto appended = full_layer_prefill(file, fast, suffix, left.export_state());
            const auto expected = full_layer_prefill(file, base, suffix, right.export_state());
            equal(appended.state, expected.state);
            if (mtp && reject)
                require(left.mtp_stats().replays > 0, "optimized indexer MTP rejection path was not tested");
            if (mtp && !reject)
                require(left.mtp_stats().accepted > 0, "optimized indexer MTP acceptance path was not tested");
            Model model(path.string(), fast);
            require(model.prefill({1, 2, 3}).size() == static_cast<size_t>(file.geometry().n_vocab),
                    "Model dispatcher rejected optimized indexer options");
            model.prepare_decode();
            model.token(4);
            model.reset();
            model.prefill({5, 6, 7});
        }
    }
}
}

int main(int argc, char** argv) {
    cudaStream_t stream = nullptr;
    try {
        const bool benchmark = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
        const bool kernel_check = argc == 2 && std::strcmp(argv[1], "--kernel-check") == 0;
        if (argc != 1 && !benchmark && !kernel_check)
            throw std::invalid_argument("Usage: glm5_indexer_test [--bench | --kernel-check]");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "indexer test stream");
        if (kernel_check) {
            scores(7, 513, stream);
            for (int tail : {0, 1})
                for (int pattern : {0, 1, 2, 3}) selection(50000, 3, 512, tail, pattern, stream);
            selection(19, 7, 19, 1, 1, stream);
        } else if (benchmark) {
            scores(64, 50000, stream, true);
            selection(50000, 64, 512, 0, 0, stream, true);
        } else {
            for (int queries : {1, 3, 4, 7, 64, 65})
                for (int pools : {1, 19, 64, 65, 513}) scores(queries, pools, stream);
            scores(7, 50000, stream);
            for (int pools : {1, 19, 512, 513, 50000})
                for (int tail : {0, 1})
                    for (int pattern : {0, 1, 2, 3}) selection(pools, 7, std::min(512, pools), tail, pattern, stream);
            selection(513, 7, 1, 1, 1, stream);
            selection(513, 7, 511, 1, 1, stream);
            selection(0, 7, 0, 0, 2, stream);
            selection(0, 7, 0, 1, 2, stream);
            {
                Options tiled;
                tiled.dsa = true; tiled.dsa_score = DsaScore::Tiled; tiled.dsa_parallel_topk = true;
                AttentionScope outer(tiled);
                require(kernels::glm_dsa_tiled_score() && kernels::glm_dsa_parallel_select(), "indexer options not applied");
                {
                    AttentionScope inner(Options{});
                    require(!kernels::glm_dsa_tiled_score() && !kernels::glm_dsa_parallel_select(), "nested indexer options leaked");
                }
                require(kernels::glm_dsa_tiled_score() && kernels::glm_dsa_parallel_select(), "nested indexer scope not restored");
            }
            runtime(false);
            runtime(true);
        }
        require(!kernels::glm_dsa_tiled_score() && !kernels::glm_dsa_parallel_select(), "indexer launch scope leaked");
        check(cudaStreamDestroy(stream), "indexer test stream destroy");
        std::puts("glm5 indexer: exact tiled scores, cooperative stable top-k and guarded outputs passed");
        return 0;
    } catch (const std::exception& e) {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        std::fprintf(stderr, "glm5 indexer: %s\n", e.what());
        return 1;
    }
}
