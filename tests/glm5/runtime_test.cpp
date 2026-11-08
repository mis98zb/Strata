#include "strata/glm5/model.hpp"
#include "runtime_fixture.hpp"
#include "cpu.hpp"
#include "phase_probe.hpp"
#include "../../src/glm5/weights.hpp"
#include "strata/glm5/layer.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

using glm_fixture::require;
using glm_fixture::write_model;
using glm_fixture::near;

void cpu_row_reference(const strata::glm5::ModelFile& file, int layer, int tokens,
                        const std::vector<float>& input, std::span<const strata::glm5::ExpertMiss> entries,
                        std::span<float> output) {
    using namespace strata::glm5;
    const auto& g = file.geometry();
    const auto prefix = "blk." + std::to_string(layer) + ".ffn_";
    const auto& gate = file.tensor(prefix + "gate_exps.weight");
    const auto& up = file.tensor(prefix + "up_exps.weight");
    const auto& down = file.tensor(prefix + "down_exps.weight");
    const auto gr = tensor_payload_bytes(gate) / (g.n_expert * g.n_ff);
    const auto ur = tensor_payload_bytes(up) / (g.n_expert * g.n_ff);
    const auto dr = tensor_payload_bytes(down) / (g.n_expert * g.n_embd);
    require(input.size() == static_cast<size_t>(tokens * g.n_embd), "CPU reference shape");
    for (const auto& entry : entries) {
        const auto ga = quantize(input.data() + entry.token * g.n_embd, g.n_embd, gate.type);
        const auto ua = quantize(input.data() + entry.token * g.n_embd, g.n_embd, up.type);
        std::vector<float> hidden(g.n_ff);
        for (int64_t r = 0; r < g.n_ff; ++r) {
            float gv = 0, uv = 0;
            cpu_traits(gate.type).vec_dot(g.n_embd, &gv, 0,
                file.data(gate.name) + (entry.expert * g.n_ff + r) * gr, 0, ga.data(), 0, 1);
            cpu_traits(up.type).vec_dot(g.n_embd, &uv, 0,
                file.data(up.name) + (entry.expert * g.n_ff + r) * ur, 0, ua.data(), 0, 1);
            hidden[r] = swiglu(gv, uv, static_cast<float>(g.swiglu_clamp));
        }
        const auto ha = quantize(hidden.data(), g.n_ff, down.type);
        for (int64_t r = 0; r < g.n_embd; ++r)
            cpu_traits(down.type).vec_dot(g.n_ff, &output[entry.row * g.n_embd + r], 0,
                file.data(down.name) + (entry.expert * g.n_embd + r) * dr, 0, ha.data(), 0, 1);
    }
}

void cpu_rows(const std::filesystem::path& path) {
    using namespace strata::glm5;
    init_cpu();
    Geometry shape; shape.n_embd = 4096; shape.n_ff = 2048; shape.n_expert = 288; shape.n_expert_used = 8;
    Options reference; reference.reference = true; reference.cpu_expert_mode = CpuExpertMode::Rows;
    require(resolve_execution_options(reference, shape).cpu_expert_mode == CpuExpertMode::Reference,
            "global reference mode must keep the original CPU executor");
    for (int threads : {1, 2, 4, 5, 7, 8, 13})
        require(!cpu_row_pool_default(shape, threads), "CPU default selected an unmeasured thread count");
    require(cpu_row_pool_default(shape, 12) == cpu_row_pool_default(shape, 6),
            "validated six-core SMT pool must accept an explicit twelve-thread request");
    Options twelve;
    twelve.cpu_threads = 12; twelve.gpu_experts_mib = -1;
    const auto explicit_count = resolve_execution_options(twelve, shape);
    require(explicit_count.cpu_threads == 12, "CPU execution selection must preserve the requested thread count");
    twelve.cpu_expert_mode = CpuExpertMode::Reference;
    require(resolve_execution_options(twelve, shape).cpu_expert_mode == CpuExpertMode::Reference,
            "explicit reference CPU executor must remain available with twelve threads");
    require(!twelve.cpu_atomic_completion, "atomic CPU completion must remain opt-in");
    twelve.cpu_atomic_completion = true;
    bool rejected_completion = false;
    try { resolve_execution_options(twelve, shape); }
    catch (const std::invalid_argument&) { rejected_completion = true; }
    require(rejected_completion, "atomic completion must reject the reference executor");
    twelve.cpu_expert_mode = CpuExpertMode::Rows;
    require(resolve_execution_options(twelve, shape).cpu_atomic_completion,
            "explicit row executor must accept atomic completion");
    twelve.reference = true;
    rejected_completion = false;
    try { resolve_execution_options(twelve, shape); }
    catch (const std::invalid_argument&) { rejected_completion = true; }
    require(rejected_completion, "atomic completion must reject global reference mode");
    shape.n_ff = 768;
    require(!cpu_row_pool_default(shape, 6), "CPU default selected an unmeasured expert geometry");
    const std::array<std::array<uint32_t, 3>, 3> formats = {{
        {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
        {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K},
        {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}
    }};
    for (const auto& types : formats) {
        write_model(path, false, false, 4, 768, types);
        ModelFile file(path.string());
        const auto& g = file.geometry();
        for (const bool atomic_completion : {false, true}) for (int threads : {1, 2, 3, 6, 12}) {
            CpuRowPool pool(file, g.n_layers + 1, threads, atomic_completion);
            for (int tokens : {1, 3, 19, 3}) {
                std::vector<float> input(tokens * g.n_embd);
                for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(i * 0.173f) * 0.1f;
                const int rows = tokens * g.n_expert_used;
                for (int count : {0, 1, rows - 1, rows}) {
                    if (count == rows) for (auto& value : input) value *= 10000.0f;
                    std::vector<ExpertMiss> entries;
                    for (int i = 0; i < count; ++i)
                        entries.push_back({i % tokens, (i * (rows - 1)) % rows, i % static_cast<int>(g.n_expert)});
                    std::vector<float> expected(rows * g.n_embd + 32, 123.5f), actual(expected);
                    for (int layer = static_cast<int>(g.n_layer_dense_lead); layer <= g.n_layers; ++layer) {
                        auto e = std::span(expected).subspan(16, rows * g.n_embd);
                        auto a = std::span(actual).subspan(16, rows * g.n_embd);
                        cpu_row_reference(file, layer, tokens, input, entries, e);
                        pool.run(layer, tokens, input, entries, a);
                        require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                                "CPU row pool changed dot/SwiGLU bytes, output slots or guards");
                    }
                    for (auto& value : input) value = value * -0.75f + 0.01f;
                }
            }
            std::vector<float> input(g.n_embd, 0.1f), output(g.n_expert_used * g.n_embd);
            const std::vector<ExpertMiss> duplicate{{0, 0, 0}, {0, 0, 1}};
            bool failed = false;
            try { pool.run(g.n_layer_dense_lead, 1, input, duplicate, output); }
            catch (const std::invalid_argument&) { failed = true; }
            require(failed, "CPU row pool must reject duplicate output slots");
            const std::vector<ExpertMiss> valid{{0, 0, 0}};
            pool.run(g.n_layer_dense_lead, 1, input, valid, output);
            std::vector<float> expected(output);
            for (int step = 0; atomic_completion && step < 128; ++step) {
                input[0] = step * 0.001f;
                cpu_row_reference(file, g.n_layer_dense_lead, 1, input, valid, expected);
                pool.run(g.n_layer_dense_lead, 1, input, valid, output);
                require(std::memcmp(expected.data(), output.data(), output.size() * sizeof(float)) == 0,
                        "atomic completion returned before CPU rows were visible");
            }
        }
    }
}

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
    if (argc == 3 && (std::string(argv[1]) == "--write-fixture" ||
                      std::string(argv[1]) == "--write-dsa-fixture")) {
        try {
            write_model(argv[2], false, std::string(argv[1]) == "--write-dsa-fixture");
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "glm5 fixture: %s\n", e.what());
            return 1;
        }
    }
    const bool cpu_only = argc == 2 && std::string(argv[1]) == "--cpu-rows";
    if (argc != 1 && !cpu_only) {
        std::fprintf(stderr, "usage: glm5_runtime_test [--write-fixture PATH | --write-dsa-fixture PATH | --cpu-rows]\n");
        return 2;
    }
    int devices = 0;
    if (!cpu_only && (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)) {
        std::puts("glm5 runtime: skipped (no GPU)");
        return 77;
    }
    const auto directory = std::filesystem::temp_directory_path();
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto accept_path = directory / ("strata-glm5-accept-" + stamp + ".gguf");
    const auto reject_path = directory / ("strata-glm5-reject-" + stamp + ".gguf");
    try {
        cpu_rows(accept_path);
        if (cpu_only) {
            std::filesystem::remove(accept_path);
            std::puts("glm5 CPU rows: exact mixed-format rows, guards, tails, reuse and worker shutdown passed");
            return 0;
        }
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
            for (int mtp : {0, 3}) {
                auto plain = base;
                plain.prefill = 1; plain.mtp = mtp; plain.gpu_experts_mib = 2; plain.gpu_stage_slots = 1;
                plain.cpu_expert_mode = strata::glm5::CpuExpertMode::Rows; plain.cpu_threads = 3;
                auto instrumented = plain; instrumented.probe = true;
                strata::glm5::Model left(accept_path.string(), plain), right(accept_path.string(), instrumented);
                require(left.prefill(prompt) == right.prefill(prompt), "CPU upload probes changed prompt logits");
                require(left.generate(8) == right.generate(8), "CPU upload probes changed generated IDs");
                require(left.token(5) == right.token(5), "CPU upload probes changed continuation");
                const auto before = right.decode_probe_stats();
                const auto phase = static_cast<size_t>(strata::glm5::ProbePhase::CpuResultH2D);
                require(before.cpu_result_copies > 0 && before.cpu_result_copies == right.cache_stats().misses &&
                        before.cpu_experts == before.cpu_result_copies &&
                        before.phases[phase].bytes == before.cpu_result_copies * right.geometry().n_embd * sizeof(float) &&
                        left.decode_probe_stats().cpu_result_copies == 0 &&
                        left.cache_stats().hits == right.cache_stats().hits &&
                        left.cache_stats().misses == right.cache_stats().misses,
                        "CPU upload probes must count executed row copies without changing routing");
                require(left.token(6) == right.token(6), "CPU upload interval probes changed continuation");
                const auto delta = strata::glm5::probe_delta(right.decode_probe_stats(), before);
                require(delta.cpu_result_copies == right.cache_stats().misses - before.cpu_experts,
                        "CPU upload copy interval must exclude prior requests");
                right.reset(); left.reset();
                require(right.decode_probe_stats().cpu_result_copies == 0, "reset retained old CPU upload counters");
                require(left.prefill(prompt) == right.prefill(prompt), "CPU upload probes changed reset prompt");
            }
            {
                auto all_cpu = base; all_cpu.probe = true;
                strata::glm5::Model measured(accept_path.string(), all_cpu);
                require(measured.prefill(prompt) == reference, "all-CPU copy probes changed prompt logits");
                const auto stats = measured.decode_probe_stats();
                const auto& copy = stats.phases[static_cast<size_t>(strata::glm5::ProbePhase::CpuResultH2D)];
                require(stats.cpu_result_copies == prompt.size() &&
                        stats.cpu_result_copies == copy.stream_calls &&
                        stats.cpu_result_copies * measured.geometry().n_expert_used == stats.cpu_experts &&
                        copy.bytes == stats.cpu_result_copies * measured.geometry().n_expert_used *
                            measured.geometry().n_embd * sizeof(float),
                        "all-CPU results must count one complete-array upload per layer batch");
            }
            for (const bool atomic_completion : {false, true}) for (bool reject : {false, true}) {
                const auto& path = reject ? reject_path : accept_path;
                auto plain = base; plain.mtp = 3; plain.prefill = 7;
                auto pooled = plain; pooled.cpu_threads = 3; pooled.cpu_expert_mode = strata::glm5::CpuExpertMode::Rows;
                pooled.cpu_atomic_completion = atomic_completion;
                strata::glm5::Model expected(path.string(), plain), actual(path.string(), pooled);
                require(expected.prefill(prompt) == actual.prefill(prompt), "row pool changed prompt bits");
                require(expected.generate(12) == actual.generate(12), "row pool changed MTP output IDs");
                require(expected.token(5) == actual.token(5), "row pool changed continuation bits");
                actual.reset(); actual.prefill(prompt);
                expected.reset(); expected.prefill(prompt);
                require(expected.generate(12) == actual.generate(12), "row pool retained stale scratch after reset");
                auto cached = pooled;
                cached.gpu_expert_mode = strata::glm5::GpuExpertMode::Single;
                cached.gpu_experts_mib = 2; cached.gpu_stage_slots = 1; cached.gpu_check = true;
                auto cached_reference = cached; cached_reference.cpu_expert_mode = strata::glm5::CpuExpertMode::Reference;
                cached_reference.cpu_atomic_completion = false;
                cached_reference.gpu_expert_mode = strata::glm5::GpuExpertMode::Reference;
                cached_reference.cpu_threads = 1;
                strata::glm5::Model left(path.string(), cached_reference), right(path.string(), cached);
                require(left.prefill(prompt) == right.prefill(prompt), "row pool changed staged/cache-check prefill");
                require(left.generate(8) == right.generate(8), "row pool changed staged/cache-check MTP output");
                require(right.cache_stats().direct_rows > 0, "MTP single-token phases must exercise direct expert dispatch");
            }
            {
                auto plain = base;
                plain.full_layer_prefill = true; plain.prefill = 7; plain.mtp = 3;
                plain.prefill_gpu_mib = 32; plain.prefill_reserve_mib = 128;
                auto pooled = plain; pooled.cpu_threads = 3;
                pooled.cpu_expert_mode = strata::glm5::CpuExpertMode::Rows;
                pooled.cpu_atomic_completion = true;
                strata::glm5::Model left(accept_path.string(), plain), right(accept_path.string(), pooled);
                require(left.prefill(prompt) == right.prefill(prompt), "row pool changed independent prefill");
                require(left.generate(4) == right.generate(4), "row pool changed state handoff output");
                const std::vector<int32_t> append{3, 4, 5};
                require(left.prefill(append) == right.prefill(append), "row pool changed append after decode teardown");
                require(left.generate(4) == right.generate(4), "row pool changed recreated decoder output");
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
