#include "strata/glm5/model.hpp"
#include "runtime_fixture.hpp"
#include "cpu.hpp"
#include "../../src/glm5/weights.hpp"
#include "strata/glm5/layer.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

using glm_fixture::require;
using glm_fixture::write_model;
using glm_fixture::near;

void replay_prefixes(const std::filesystem::path& path) {
    using namespace strata::glm5;
    constexpr int window = 4;
    ModelFile file(path.string());
    const auto& g = file.geometry();
    WeightBank bank(file, [](const strata::TensorInfo& t) { return t.name.starts_with("blk.0."); });
    std::vector<std::unique_ptr<Buffer>> storage;
    const auto alloc = [&](uint64_t bytes) {
        auto b = std::make_unique<Buffer>(bytes);
        auto* p = b->get(); storage.push_back(std::move(b)); return p;
    };
    const auto scratch = [&](int tokens, GlmBuffers& b, MoEBuffers& m, BlockBuffers& bb) {
        GlmBuffers base;
        glm_buffers_init(g, tokens, 16, alloc(glm_buffers_bytes(g, tokens, 16)), base);
        GlmChunkBuffers chunk;
        glm_chunk_init(g, g.n_expert_used, tokens, alloc(glm_chunk_bytes(g, g.n_expert_used, tokens)), chunk);
        glm_chunk_group_view(chunk, base, m, bb, 0, tokens, b, m, bb);
    };
    GlmBuffers group, single;
    MoEBuffers gm, sm;
    BlockBuffers gb, sb;
    scratch(window, group, gm, gb);
    scratch(1, single, sm, sb);
    const auto bytes = glm_layer_state_bytes(g, 16, 0);
    auto* forward_data = alloc(bytes);
    auto* expected_data = alloc(bytes);
    auto* replay_data = alloc(bytes);
    GlmLayerState forward, expected, replayed;
    glm_state_init(g, 16, 0, forward_data, forward);
    glm_state_init(g, 16, 0, expected_data, expected);
    glm_state_init(g, 16, 0, replay_data, replayed);
    KdaReplay capture;
    capture.capacity = window;
    const auto span = window * g.n_head * g.kda_head_dim;
    auto* raw = static_cast<float*>(alloc((4 * span + window * g.n_head) * sizeof(float)));
    capture.q = raw; capture.k = raw + span; capture.v = raw + 2 * span;
    capture.gate = raw + 3 * span; capture.beta = raw + 4 * span;
    group.replay = &capture;
    std::vector<float> checkpoint(bytes / sizeof(float)), input(window * g.hc_dim()), a(checkpoint.size()), b(checkpoint.size());
    for (size_t i = 0; i < checkpoint.size(); ++i) checkpoint[i] = (static_cast<int>(i % 11) - 5) * 1e-4f;
    for (size_t i = 0; i < input.size(); ++i) input[i] = 0.1f + (i % 53) * 0.003f;
    struct Stream {
        cudaStream_t value = nullptr;
        Stream() { check(cudaStreamCreate(&value), "KDA replay stream"); }
        ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
    } stream;
    check(cudaMemcpyAsync(forward_data, checkpoint.data(), bytes, cudaMemcpyHostToDevice, stream.value), "KDA seed");
    check(cudaMemcpyAsync(gb.R, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream.value), "KDA input");
    std::string err;
    if (!glm_block_layer_pre(bank.table(), g, 0, 0, 0, group, forward, gm, g.n_expert_used, gb,
                             stream.value, err, nullptr)) throw std::runtime_error(err);
    check(cudaStreamSynchronize(stream.value), "KDA capture sync");
    require(capture.tokens == window, "verification inputs were not captured");
    for (int keep = 0; keep <= window; ++keep) {
        check(cudaMemcpyAsync(expected_data, checkpoint.data(), bytes, cudaMemcpyHostToDevice, stream.value), "KDA expected reset");
        check(cudaMemcpyAsync(replay_data, checkpoint.data(), bytes, cudaMemcpyHostToDevice, stream.value), "KDA replay reset");
        for (int t = 0; t < keep; ++t) {
            check(cudaMemcpyAsync(sb.R, input.data() + t * g.hc_dim(), g.hc_dim() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream.value), "KDA prefix input");
            if (!glm_block_layer_pre(bank.table(), g, 0, t, 0, single, expected, sm, g.n_expert_used, sb,
                                     stream.value, err, nullptr)) throw std::runtime_error(err);
        }
        if (!glm_kda_replay(bank.table(), g, 0, group, replayed, capture, keep, stream.value, err))
            throw std::runtime_error(err);
        check(cudaMemcpyAsync(a.data(), expected_data, bytes, cudaMemcpyDeviceToHost, stream.value), "KDA reference state");
        check(cudaMemcpyAsync(b.data(), replay_data, bytes, cudaMemcpyDeviceToHost, stream.value), "KDA committed state");
        check(cudaStreamSynchronize(stream.value), "KDA replay sync");
        near(b, a, 1e-5f, "captured-prefix KDA/conv state");
    }
    require(!glm_kda_replay(bank.table(), g, 0, group, replayed, capture, window + 1, stream.value, err) &&
            !err.empty(), "out-of-range replay prefix must fail explicitly");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--write-fixture") {
        try {
            write_model(argv[2], false);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "glm5 fixture: %s\n", e.what());
            return 1;
        }
    }
    if (argc != 1) {
        std::fprintf(stderr, "usage: glm5_runtime_test [--write-fixture PATH]\n");
        return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::puts("glm5 runtime: skipped (no GPU)");
        return 77;
    }
    const auto directory = std::filesystem::temp_directory_path();
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto accept_path = directory / ("strata-glm5-accept-" + stamp + ".gguf");
    const auto reject_path = directory / ("strata-glm5-reject-" + stamp + ".gguf");
    try {
        write_model(accept_path, false);
        write_model(reject_path, true);
        replay_prefixes(accept_path);
        std::vector<int32_t> prompt;
        for (int i = 0; i < 19; ++i) prompt.push_back(1 + i % 17);
        strata::glm5::Options base;
        base.max_context = 96;
        base.gpu_reserve_mib = 128;
        std::vector<float> reference;
        {
            strata::glm5::Model model(accept_path.string(), base);
            reference = model.prefill(prompt);
            for (const int width : {2, 7, 8, 17, 32}) {
                auto options = base; options.prefill = width;
                strata::glm5::Model chunked(accept_path.string(), options);
                near(chunked.prefill(prompt), reference, 1e-5f, "chunked prefill");
                near(chunked.token(5), model.token(5), 1e-5f, "prefill continuation");
                model.reset(); model.prefill(prompt);
            }
        }
        {
            auto options = base;
            options.cpu_threads = 2;
            if (strata::glm5::cpu_parallel_available()) {
                strata::glm5::Model serial(accept_path.string(), base);
                strata::glm5::Model parallel(accept_path.string(), options);
                require(parallel.prefill(prompt) == serial.prefill(prompt), "parallel CPU experts changed prompt bits");
                require(parallel.generate(12) == serial.generate(12), "parallel CPU experts changed greedy output");
                require(parallel.token(5) == serial.token(5), "parallel CPU experts changed continuation bits");
            } else {
                bool refused = false;
                try { strata::glm5::Model unsupported(accept_path.string(), options); }
                catch (const std::invalid_argument&) { refused = true; }
                require(refused, "CPU threading must fail explicitly without OpenMP");
            }
        }
        {
            auto options = base;
            options.gpu_experts_mib = 1;
            options.gpu_check = true;
            strata::glm5::Model cached(accept_path.string(), options);
            near(cached.prefill(prompt), reference, 0.05f, "cached expert decode");
            const auto stats = cached.cache_stats();
            require(stats.hits && stats.misses, "cache test needs both hits and CPU misses");
            options.prefill = 17;
            strata::glm5::Model streaming(accept_path.string(), options);
            near(streaming.prefill(prompt), reference, 0.05f, "streamed expert prefill");
            const auto streamed = streaming.cache_stats();
            require(streamed.streamed && streamed.swaps, "prefill test must stream and replace expert slots");
            streaming.token(5);
            require(streaming.cache_stats().hits, "decode must reuse the prefill slot table");
        }
        for (const bool reject : {false, true}) {
            const auto& path = reject ? reject_path : accept_path;
            auto plain = base; plain.prefill = 8;
            auto speculative = plain; speculative.mtp = 3;
            if (strata::glm5::cpu_parallel_available()) speculative.cpu_threads = 2;
            strata::glm5::Model target(path.string(), plain);
            strata::glm5::Model draft(path.string(), speculative);
            near(draft.prefill(prompt), target.prefill(prompt), 1e-5f, "MTP prompt target state");
            const auto before = draft.mtp_stats();
            const auto expected = target.generate(12);
            const auto actual = draft.generate(12);
            require(actual == expected && actual.size() == 12, "MTP changed greedy output");
            const auto stats = draft.mtp_stats();
            require(stats.offered && stats.verify_batches, "MTP must run a real verification batch");
            require(reject ? stats.replays > 0 : stats.accepted > 0, "missing rejection replay or draft acceptance");
            require(stats.target_batches - before.target_batches == stats.verify_batches + (reject ? 1 : 0),
                    "rejection repeated full target forwards instead of state-only commit");
            require(reject ? stats.replayed_layers == stats.replays * 3 : stats.replayed_layers == 0,
                    "only the three recurrent layers should be replayed");
            const auto continuation = target.token(17);
            near(draft.token(17), continuation, 1e-5f, "MTP rejection/acceptance state continuation");
            draft.reset();
            draft.prefill(prompt);
            require(draft.generate(12) == expected, "MTP reset leaked draft or target state");
            auto mixed = speculative;
            mixed.prefill = 17;
            mixed.gpu_experts_mib = 2;
            mixed.gpu_stage_slots = 1;
            mixed.gpu_check = true;
            strata::glm5::Model accelerated(path.string(), mixed);
            near(accelerated.prefill(prompt), reference, 0.05f, "combined prefill/cache/MTP prompt");
            require(accelerated.generate(12) == expected, "combined accelerators changed greedy output");
            near(accelerated.token(17), continuation, 0.05f, "combined acceleration continuation");
            require(accelerated.cache_stats().streamed && accelerated.mtp_stats().verify_batches,
                    "combined test must execute both GPU streaming and MTP verification");
            require(accelerated.cache_stats().pinned_bytes && accelerated.cache_stats().upload_bytes,
                    "combined test must use the bounded asynchronous upload ring");
        }
        std::filesystem::remove(accept_path);
        std::filesystem::remove(reject_path);
        std::puts("glm5 runtime: ragged prefill, GPU cache, streamed eviction, MTP acceptance and rejection passed");
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(accept_path);
        std::filesystem::remove(reject_path);
        std::fprintf(stderr, "glm5 runtime test: %s\n", e.what());
        return 1;
    }
}
