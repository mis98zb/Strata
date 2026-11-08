#include "strata/glm5/model.hpp"
#include "decode.hpp"
#include "prefill.hpp"
#include "runtime_fixture.hpp"
#include "phase_probe.hpp"
#include "weights.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

namespace {
using glm_fixture::require;
using glm_fixture::near;

void compare_state(const strata::glm5::HostState& a, const strata::glm5::HostState& b) {
    require(a.position == b.position && a.layers.size() == b.layers.size(), "handoff state shape changed");
    for (size_t l = 0; l < a.layers.size(); ++l) {
        if (!a.layers[l].kv.empty() || !b.layers[l].kv.empty())
            glm_fixture::near_fp16(a.layers[l].kv, b.layers[l].kv, 0.002f, "MLA cache across scheduling");
        if (!a.layers[l].recurrent.empty())
            near(a.layers[l].recurrent, b.layers[l].recurrent, 1e-5f, "KDA/conv state handoff");
        if (!a.layers[l].indexer.empty() || !b.layers[l].indexer.empty())
            near(a.layers[l].indexer, b.layers[l].indexer, 1e-5f, "DSA indexer state handoff");
    }
    if (!a.mtp.kv.empty() || !b.mtp.kv.empty())
        glm_fixture::near_fp16(a.mtp.kv, b.mtp.kv, 0.002f, "MTP prompt cache across scheduling");
    if (!a.mtp.indexer.empty() || !b.mtp.indexer.empty())
        near(a.mtp.indexer, b.mtp.indexer, 1e-5f, "NextN indexer state handoff");
    if (!a.carry_hidden.empty() || !b.carry_hidden.empty())
        near(a.carry_hidden, b.carry_hidden, 1e-5f, "MTP hidden carry");
    near(a.logits, b.logits, 1e-5f, "full-layer prompt logits");
}

void expert_grouping(const std::filesystem::path& path) {
    using namespace strata::glm5;
    glm_fixture::write_model(path, false, true);
    ModelFile file(path.string(), 0);
    Options options;
    options.full_layer_prefill = true; options.prefill = 64; options.max_context = 128;
    options.prefill_gpu_mib = 32; options.prefill_host_mib = 64; options.prefill_reserve_mib = 128;
    options.dsa = true; options.dsa_tail = 0; options.gpu_warm = true; options.gpu_experts_mib = 2;
    std::vector<int32_t> prompt(83);
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = 1 + i % 17;
    auto grouped = options; grouped.prefill_expert_groups = true;
    const auto reference = full_layer_prefill(file, options, prompt, {});
    const auto candidate = full_layer_prefill(file, grouped, prompt, {});
    require(reference.stats.expert_groups == 0 && candidate.stats.expert_groups > 0 &&
            candidate.stats.grouped_experts >= 2 && candidate.stats.grouped_rows > 0,
            "expert group flag must execute grouped work only when requested");
    const auto same_state = [](const HostState& a, const HostState& b) {
        require(a.position == b.position && a.logits == b.logits && a.carry_hidden == b.carry_hidden &&
                a.expert_heat == b.expert_heat && a.layers.size() == b.layers.size(),
                "expert groups changed exact handoff or heat");
        for (size_t l = 0; l < a.layers.size(); ++l)
            require(a.layers[l].kv == b.layers[l].kv && a.layers[l].recurrent == b.layers[l].recurrent &&
                    a.layers[l].indexer == b.layers[l].indexer,
                    "expert groups changed exact layer state");
        require(a.mtp.kv == b.mtp.kv && a.mtp.recurrent == b.mtp.recurrent && a.mtp.indexer == b.mtp.indexer,
                "expert groups changed exact NextN state");
    };
    same_state(reference.state, candidate.state);
    require(reference.stats.groups == candidate.stats.groups &&
            reference.stats.weight_upload_bytes == candidate.stats.weight_upload_bytes &&
            reference.stats.activation_upload_bytes == candidate.stats.activation_upload_bytes &&
            reference.stats.activation_download_bytes == candidate.stats.activation_download_bytes &&
            reference.stats.mmq_calls == candidate.stats.mmq_calls &&
            reference.stats.f16_gemm_calls == candidate.stats.f16_gemm_calls &&
            reference.stats.native_quant_calls == candidate.stats.native_quant_calls,
            "expert groups changed resource grouping, matrix work or quantization count");
    const std::vector<int32_t> suffix(19, 1);
    const auto appended_reference = full_layer_prefill(file, options, suffix, reference.state);
    const auto appended_candidate = full_layer_prefill(file, grouped, suffix, candidate.state);
    same_state(appended_reference.state, appended_candidate.state);
    auto mtp_reference = options; mtp_reference.mtp = 3;
    auto mtp_grouped = mtp_reference; mtp_grouped.prefill_expert_groups = true;
    const auto reference_mtp = full_layer_prefill(file, mtp_reference, prompt, {});
    const auto candidate_mtp = full_layer_prefill(file, mtp_grouped, prompt, {});
    same_state(reference_mtp.state, candidate_mtp.state);
    std::vector<int32_t> expected_ids;
    {
        DecodeExecutor decoder(file, mtp_reference, &reference_mtp.state);
        expected_ids = decoder.generate(4);
    }
    {
        DecodeExecutor decoder(file, mtp_grouped, &candidate_mtp.state);
        require(decoder.generate(4) == expected_ids, "expert groups changed MTP output IDs");
    }
}

void transfer_grouping(const std::filesystem::path& path) {
    using namespace strata::glm5;
    glm_fixture::write_model(path, false);
    ModelFile file(path.string());
    const auto& g = file.geometry();
    uint64_t largest = 0, total = 0;
    for (int l = 0; l < g.n_layers; ++l) {
        const auto prefix = "blk." + std::to_string(l) + ".";
        const auto bytes = WeightBank::upload_bytes(file, [&](const auto& t) {
            return t.name.starts_with(prefix) && t.name.find("indexer") == std::string::npos;
        });
        largest = std::max(largest, bytes);
        total += bytes;
    }
    const auto count = largest / (g.hc_dim() * sizeof(float)) + 1;
    require(count > 5 && count < 8192, "transfer fixture must cross the actual weight/activation boundary");
    Options options;
    options.full_layer_prefill = true; options.prefill = 64;
    options.prefill_transfer_cost = true;
    options.max_context = count + 16; options.prefill_gpu_mib = 32;
    options.prefill_host_mib = 64; options.prefill_reserve_mib = 128;
    const std::vector<int32_t> short_prompt(5, 1), long_prompt(count, 1);
    auto explicit_group = options; explicit_group.prefill_group_layers = 4;
    const auto short_auto = full_layer_prefill(file, options, short_prompt, {});
    const auto short_capacity = full_layer_prefill(file, explicit_group, short_prompt, {});
    require(short_auto.stats.groups == static_cast<uint64_t>(g.n_layers) &&
            short_auto.stats.prefetched_layers == static_cast<uint64_t>(g.n_layers - 1) &&
            short_capacity.stats.groups == 1, "short input must prefer covered prefetch over grouping");
    require(short_auto.stats.selected_batch == short_capacity.stats.selected_batch &&
            short_auto.stats.unprefetched_weight_bytes < short_capacity.stats.unprefetched_weight_bytes &&
            short_capacity.stats.unprefetched_weight_bytes == total,
            "transfer grouping must count uncovered tensor uploads without reducing batch");
    require(short_auto.state.logits == short_capacity.state.logits,
            "short transfer grouping changed exact logits");
    compare_state(short_auto.state, short_capacity.state);
    const auto long_auto = full_layer_prefill(file, options, long_prompt, {});
    const auto long_capacity = full_layer_prefill(file, explicit_group, long_prompt, {});
    require(long_auto.stats.groups == 1 && long_auto.stats.selected_batch == long_capacity.stats.selected_batch &&
            long_auto.state.logits == long_capacity.state.logits,
            "large activation boundaries must retain beneficial grouping and exact logits");
    compare_state(long_auto.state, long_capacity.state);
    const auto appended = full_layer_prefill(file, options, short_prompt, long_auto.state);
    const auto appended_capacity = full_layer_prefill(file, explicit_group, short_prompt, long_auto.state);
    require(appended.stats.groups == static_cast<uint64_t>(g.n_layers) &&
            appended.state.logits == appended_capacity.state.logits,
            "append grouping must use new-token traffic, not accumulated context length");
    compare_state(appended.state, appended_capacity.state);
    auto no_prefetch = options; no_prefetch.prefill_prefetch_layers = 0;
    const auto unhidden = full_layer_prefill(file, no_prefetch, short_prompt, {});
    require(unhidden.stats.groups == 1 && unhidden.stats.prefetched_layers == 0 &&
            unhidden.state.logits == short_auto.state.logits,
            "without prefetch, grouping must not be charged for already exposed weights");
    auto no_pipeline = options; no_pipeline.prefill_no_pipeline = true;
    const auto synchronous = full_layer_prefill(file, no_pipeline, short_prompt, {});
    require(synchronous.stats.groups == 1 && synchronous.state.logits == short_capacity.state.logits,
            "non-pipelined execution must retain its prior grouping policy");
    for (const auto* pair : {&short_auto, &long_auto, &appended}) {
        const auto& expected = pair == &short_auto ? short_capacity : pair == &long_auto ? long_capacity : appended_capacity;
        require(pair->state.carry_hidden == expected.state.carry_hidden,
                "transfer grouping changed exact hidden carry");
        for (size_t layer = 0; layer < pair->state.layers.size(); ++layer)
            require(pair->state.layers[layer].kv == expected.state.layers[layer].kv &&
                    pair->state.layers[layer].indexer == expected.state.layers[layer].indexer &&
                    pair->state.layers[layer].recurrent == expected.state.layers[layer].recurrent,
                    "transfer grouping changed exact persistent state bytes");
    }
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::puts("glm5 full-layer: skipped (no GPU)");
        return 77;
    }
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto path = std::filesystem::temp_directory_path() / ("strata-glm5-full-" + stamp + ".gguf");
    try {
        using namespace strata::glm5;
        std::vector<int32_t> prompt;
        for (int i = 0; i < 19; ++i) prompt.push_back(1 + i % 17);
        Options options;
        options.max_context = 96; options.prefill = 7; options.mtp = 3;
        options.gpu_experts_mib = 2; options.gpu_check = true;
        options.gpu_reserve_mib = 128; options.prefill_reserve_mib = 128;
        for (const bool reject : {false, true}) {
            glm_fixture::write_model(path, reject);
            ModelFile file(path.string());
            HostState expected;
            {
                DecodeExecutor resident(file, options);
                resident.prefill(prompt);
                expected = resident.export_state();
            }
            auto full = options;
            full.full_layer_prefill = true;
            full.prefill_reference = true;
            full.prefill_gpu_mib = 16;
            const auto result = full_layer_prefill(file, full, prompt, {});
            compare_state(result.state, expected);
            {
                DecodeExecutor restored(file, options, &result.state);
                const auto copied = restored.export_state();
                require(copied.position == result.state.position && copied.layers.size() == result.state.layers.size(),
                        "restored handoff shape changed");
                for (size_t l = 0; l < copied.layers.size(); ++l)
                    require(copied.layers[l].kv == result.state.layers[l].kv &&
                            copied.layers[l].recurrent == result.state.layers[l].recurrent,
                            "raw KDA/MLA handoff must be bit-exact");
                require(copied.mtp.kv == result.state.mtp.kv && copied.carry_hidden == result.state.carry_hidden &&
                        copied.logits == result.state.logits, "raw MTP/logit handoff must be bit-exact");
            }
            require(result.stats.layer_loads == 5 && result.stats.experts_loaded == 4,
                    "prefill must load every complete trunk/NextN layer and all its experts");
            require(result.stats.gpu_peak_bytes > 0 && result.stats.gpu_peak_bytes <= 16 * 1048576ULL,
                    "full-layer tracked GPU budget");
            auto exact_budget = full;
            exact_budget.prefill_gpu_mib = (result.stats.gpu_peak_bytes + 1048575) / 1048576;
            compare_state(full_layer_prefill(file, exact_budget, prompt, {}).state, expected);
            bool failed = false;
            auto too_small = full; too_small.prefill_gpu_mib = 1;
            try { full_layer_prefill(file, too_small, prompt, {}); }
            catch (const std::runtime_error& e) { failed = std::string(e.what()).find("GPU budget") != std::string::npos; }
            require(failed, "GPU budget must fail rather than use partial layers or a CPU fallback");
            too_small = full; too_small.max_context = 512; too_small.prefill_host_mib = 1;
            failed = false;
            try { full_layer_prefill(file, too_small, std::vector<int32_t>(257, 1), {}); }
            catch (const std::runtime_error& e) { failed = std::string(e.what()).find("host tensor budget") != std::string::npos; }
            require(failed, "host tensor budget must be enforced before GPU uploads");
            Model baseline(path.string(), options);
            Model independent(path.string(), full);
            require(!independent.decode_loaded(), "full-layer construction must not load decode weights");
            near(independent.prefill(prompt), baseline.prefill(prompt), 1e-5f, "independent prefill");
            require(!independent.decode_loaded() && independent.cache_stats().bytes == 0,
                    "prefill must not create decode residency or expert slots");
            require(independent.generate(0).empty() && !independent.decode_loaded(),
                    "prefill-only requests must not initialize decode");
            const auto loads = independent.prefill_stats().layer_loads;
            require(independent.generate(12) == baseline.generate(12), "handoff changed greedy output");
            require(independent.decode_loaded() && independent.mtp_stats().verify_batches > 0,
                    "first generation must initialize decode and run MTP there");
            require(independent.prefill_stats().layer_loads == loads, "MTP verification reloaded prefill banks");
            near(independent.token(17), baseline.token(17), 0.05f, "decode continuation after handoff");
            const std::vector<int32_t> append{3, 4, 5, 6, 7};
            const auto appended = independent.prefill(append);
            std::vector<float> appended_reference;
            for (const auto id : append) appended_reference = baseline.token(id);
            require(!independent.decode_loaded(), "follow-up prefill must release resident decode first");
            near(appended, appended_reference, 0.05f, "follow-up prefill");
            near(independent.token(8), baseline.token(8), 0.05f, "follow-up handoff");
            independent.reset();
            require(!independent.decode_loaded(), "reset must release decode in full-layer mode");
            independent.prefill(prompt);
            require(!independent.decode_loaded(), "reset must return to the independent prefill path");
            auto constrained = full;
            constrained.prefill_host_mib = 1;
            Model preserved(path.string(), constrained);
            Model control(path.string(), full);
            preserved.prefill(prompt); control.prefill(prompt);
            preserved.generate(1); control.generate(1);
            failed = false;
            try { preserved.prefill(std::vector<int32_t>(70, 1)); }
            catch (const std::runtime_error& e) { failed = std::string(e.what()).find("host tensor budget") != std::string::npos; }
            require(failed && preserved.decode_loaded(), "host budget rejection must precede phase teardown");
            near(preserved.token(5), control.token(5), 0.05f, "state preserved after budget rejection");
        }
        {
            auto resident = options;
            resident.mtp = 0;
            resident.prefill = 17;
            auto full = resident;
            full.full_layer_prefill = true;
            full.prefill_reference = true;
            full.prefill_gpu_mib = 16;
            Model baseline(path.string(), resident);
            Model independent(path.string(), full);
            near(independent.prefill(prompt), baseline.prefill(prompt), 1e-5f, "multi-group prefill without MTP");
            require(!independent.decode_loaded() && independent.prefill_stats().layer_loads == 4 &&
                    independent.prefill_stats().experts_loaded == 2, "MTP-off full-layer residency");
            independent.prepare_decode();
            require(independent.decode_loaded(), "explicit decode preparation must load without consuming a token");
            require(independent.generate(8) == baseline.generate(8), "MTP-off handoff changed output");
            near(independent.token(17), baseline.token(17), 0.05f, "MTP-off continuation");
        }
        for (const int mtp : {0, 3}) {
            auto warm = options;
            warm.full_layer_prefill = true;
            warm.prefill_reference = true;
            warm.prefill_gpu_mib = 16;
            warm.gpu_warm = true;
            warm.gpu_heat_quotas = true;
            warm.mtp = mtp;
            ModelFile file(path.string());
            const auto result = full_layer_prefill(file, warm, prompt, {});
            require(result.state.expert_heat.size() == (4 + (mtp ? 1 : 0)) * 2, "prompt heat shape");
            for (int l = 3; l < 4 + (mtp ? 1 : 0); ++l)
                require(result.state.expert_heat[l * 2] + result.state.expert_heat[l * 2 + 1] == prompt.size() * 2,
                        "every routed prompt expert must count exactly once");
            require(result.state.bytes() == host_state_bytes(file.geometry(), prompt.size(), mtp > 0, true),
                    "heat included in state image size");
            for (const int stages : {1, 3}) {
                DecodeExecutor baseline(file, warm, &result.state);
                auto staged = warm;
                staged.gpu_stage_slots = stages;
                DecodeExecutor asynchronous(file, staged, &result.state);
                const auto stats = asynchronous.cache_stats();
                require(stats.warmed > 0 && stats.warmed == stats.slots && stats.pinned_bytes > 0 &&
                        stats.pinned_bytes <= 256 * 1048576ULL, "bounded pinned warmup");
                require(asynchronous.generate(12) == baseline.generate(12), "asynchronous warm cache changed output");
                require(asynchronous.token(17) == baseline.token(17), "asynchronous cache changed continuation bits");
                const auto handoff = asynchronous.export_state();
                require(!handoff.expert_heat.empty(), "decode heat must survive another phase handoff");
                DecodeExecutor restored(file, staged, &handoff);
                require(restored.token(5) == asynchronous.token(5), "warm cache phase handoff changed output bits");
                restored.reset();
                require(restored.cache_stats().warmed == 0, "reset must discard warmup statistics");
            }
            auto staged = warm;
            staged.gpu_stage_slots = 1;
            Model model(path.string(), staged);
            model.prefill(prompt);
            require(!model.decode_loaded(), "collecting heat must not construct decode");
            model.prepare_decode();
            require(model.cache_stats().warmed > 0, "dispatcher must preload hot experts");
            model.generate(2);
            model.prefill({3, 4});
            model.prepare_decode();
            require(model.cache_stats().warmed > 0, "appended prompt must warm the new decode cache");
        }
        for (const bool sparse : {false, true}) {
            glm_fixture::write_model(path, true, sparse);
            ModelFile file(path.string());
            for (const int mtp : {0, 3}) {
                for (const bool flash : {false, true}) {
                    if (flash && !sparse) continue;
                    auto reference = options;
                    reference.full_layer_prefill = true;
                    reference.prefill_reference = true;
                    reference.prefill_gpu_mib = 32;
                    reference.mtp = mtp;
                    reference.dsa = sparse;
                    reference.flash_attn = flash;
                    reference.gpu_warm = true;
                    const auto serial = full_layer_prefill(file, reference, prompt, {});
                    for (int mode = 0; mode < 8; ++mode) {
                        auto optimized = reference;
                        optimized.prefill_group_layers = mode == 0 ? 2 : (mode == 1 ? 4 : 1);
                        optimized.prefill_pipeline = mode != 0;
                        optimized.prefill_prefetch_layers = 2;
                        optimized.prefill_auto = mode == 3;
                        optimized.prefill_expert_rows = mode == 4;
                        optimized.skip_unused_quant = mode == 5 || mode == 7;
                        optimized.kda_register_state = mode == 6 || mode == 7;
                        const auto grouped = full_layer_prefill(file, optimized, prompt, {});
                        compare_state(grouped.state, serial.state);
                        if (mode >= 5) {
                            require(grouped.state.logits == serial.state.logits &&
                                    grouped.state.carry_hidden == serial.state.carry_hidden &&
                                    grouped.state.mtp.kv == serial.state.mtp.kv &&
                                    grouped.state.mtp.indexer == serial.state.mtp.indexer,
                                    "compute options changed exact head/NextN state");
                            for (size_t l = 0; l < grouped.state.layers.size(); ++l)
                                require(grouped.state.layers[l].kv == serial.state.layers[l].kv &&
                                        grouped.state.layers[l].indexer == serial.state.layers[l].indexer &&
                                        grouped.state.layers[l].recurrent == serial.state.layers[l].recurrent,
                                        "compute options changed exact trunk state");
                        }
                        require(grouped.state.expert_heat == serial.state.expert_heat, "grouping changed prompt heat");
                        require(grouped.stats.selected_batch == static_cast<uint64_t>(reference.prefill) &&
                                grouped.stats.max_group_layers == static_cast<uint64_t>(optimized.prefill_group_layers),
                                "planner did not execute the selected group/batch shape");
                        require(grouped.stats.layer_loads == serial.stats.layer_loads &&
                                grouped.stats.weight_upload_bytes == serial.stats.weight_upload_bytes,
                                "prefetch must upload every weight exactly once");
                        require(grouped.stats.gpu_peak_bytes <= 32 * 1048576ULL &&
                                grouped.stats.host_tensor_bytes <= optimized.prefill_host_mib * 1048576ULL &&
                                grouped.stats.transfer_peak_bytes <= optimized.prefill_transfer_mib * 1048576ULL,
                                "optimized prefill exceeded a tracked budget");
                        require(grouped.stats.fused_head &&
                                grouped.stats.activation_download_bytes < serial.stats.activation_download_bytes,
                                "last group must omit the unused residual download when head fits");
                        if (grouped.stats.groups > 1)
                            require(grouped.stats.prefetched_layers > 0, "weight prefetch path was not exercised");
                        const std::vector<int32_t> suffix{8, 7, 6, 5, 4};
                        const auto continued = full_layer_prefill(file, optimized, suffix, grouped.state);
                        const auto expected = full_layer_prefill(file, reference, suffix, serial.state);
                        compare_state(continued.state, expected.state);
                        DecodeExecutor left(file, optimized, &continued.state), right(file, reference, &expected.state);
                        require(left.generate(8) == right.generate(8), "optimized handoff changed MTP/decode output");
                    }
                }
            }
        }
        {
            ModelFile file(path.string());
            auto automatic = options;
            automatic.max_context = 320;
            automatic.full_layer_prefill = true;
            automatic.prefill_reference = true;
            automatic.mtp = 0;
            automatic.prefill = 256;
            automatic.prefill_gpu_mib = 32;
            automatic.prefill_auto = true;
            automatic.prefill_pipeline = true;
            automatic.prefill_transfer_mib = 1;
            std::vector<int32_t> long_prompt(257, 1);
            const auto planned = full_layer_prefill(file, automatic, long_prompt, {});
            require(planned.stats.selected_batch == 128 && planned.stats.planning_attempts == 2,
                    "automatic planning must halve the batch to fit actual transfer buffers");
            require(planned.stats.max_group_layers == 4 && planned.stats.groups == 1,
                    "automatic planning must pack multiple layers");
            auto control = automatic;
            control.prefill_auto = false; control.prefill_pipeline = false; control.prefill = 128;
            compare_state(planned.state, full_layer_prefill(file, control, long_prompt, {}).state);
            automatic.prefill_auto = false;
            bool refused = false;
            try { full_layer_prefill(file, automatic, long_prompt, {}); }
            catch (const BudgetExceeded&) { refused = true; }
            require(refused, "manual scheduling must not silently reduce the batch");
            automatic.prefill_auto = true;
            automatic.prefill_gpu_mib = 1;
            refused = false;
            try { full_layer_prefill(file, automatic, long_prompt, {}); }
            catch (const BudgetExceeded&) { refused = true; }
            require(refused, "automatic planning must fail when even a complete single layer cannot fit");
        }
        {
            const auto resident = resolve_options({});
            require(resident.prefill == 1 && !resident.prefill_auto && !resident.prefill_pipeline &&
                    resident.prefill_prefetch_layers == 0, "resident defaults must remain unchanged");
            Options defaults;
            defaults.full_layer_prefill = true;
            defaults.max_context = 96;
            defaults.prefill_gpu_mib = 32;
            defaults.prefill_reserve_mib = 128;
            const auto resolved = resolve_options(defaults);
            require(resolved.prefill == 2048 && resolved.prefill_auto && resolved.prefill_pipeline &&
                    resolved.prefill_prefetch_layers == 1, "full-layer alone must enable automatic scheduling");
            ModelFile file(path.string());
            const auto expected = full_layer_prefill(file, defaults, prompt, {});
            require(expected.stats.selected_batch == prompt.size() && expected.stats.transfer_peak_bytes > 0,
                    "default full-layer must use actual prompt length and activation pipeline");
            Model model(path.string(), defaults);
            near(model.prefill(prompt), expected.state.logits, 1e-5f, "dispatcher full-layer defaults");
            require(!model.decode_loaded(), "automatic default must preserve lazy decode");

            defaults.prefill = 64;
            defaults.prefill_group_layers = 1;
            defaults.prefill_prefetch_layers = 0;
            defaults.reference = true;
            defaults.prefill_pipeline = true;
            std::vector<int32_t> input(65, 1);
            const auto pipelined = full_layer_prefill(file, defaults, input, {});
            auto pageable = defaults;
            pageable.prefill_no_pipeline = true;
            const auto synchronous = full_layer_prefill(file, pageable, input, {});
            compare_state(pipelined.state, synchronous.state);
            const auto boundary_bytes = 2 * 64 * file.geometry().hc_dim() * sizeof(float);
            require(pipelined.stats.gpu_peak_bytes == synchronous.stats.gpu_peak_bytes + boundary_bytes,
                    "pipeline GPU budget must include exactly two device boundary buffers");
            require(pipelined.stats.host_tensor_bytes == synchronous.stats.host_tensor_bytes,
                    "pipeline pins the existing host array without adding a copy");
            auto batched_rows = pageable;
            batched_rows.prefill_expert_rows = true;
            const auto mapped = full_layer_prefill(file, batched_rows, input, {});
            compare_state(mapped.state, synchronous.state);
            require(mapped.state.logits == synchronous.state.logits,
                    "stable counting routes must preserve exact expert and token order");
            const auto map_bytes = 64 * file.geometry().n_expert_used * 2 * sizeof(int32_t);
            require(mapped.stats.host_tensor_bytes == synchronous.stats.host_tensor_bytes + map_bytes +
                        (3 * file.geometry().n_expert + 1) * sizeof(int32_t) &&
                    mapped.stats.gpu_peak_bytes == synchronous.stats.gpu_peak_bytes + map_bytes &&
                    mapped.stats.transfer_peak_bytes == synchronous.stats.transfer_peak_bytes + map_bytes,
                    "expert row maps must count once against each host, GPU and transfer budget");

            defaults.prefill_group_layers = 0;
            defaults.prefill_prefetch_layers = -1;
            defaults.reference = false;
            bool exercised_priority = false;
            for (int cap = 8; cap <= 24; ++cap) {
                defaults.prefill_gpu_mib = cap;
                PrefillResult planned;
                try { planned = full_layer_prefill(file, defaults, input, {}); }
                catch (const BudgetExceeded&) { continue; }
                auto single = defaults; single.prefill_group_layers = 1;
                const auto lookahead = full_layer_prefill(file, single, input, {});
                auto no_lookahead = defaults; no_lookahead.prefill_prefetch_layers = 0;
                const auto grouped = full_layer_prefill(file, no_lookahead, input, {});
                require(planned.stats.selected_batch == lookahead.stats.selected_batch &&
                        planned.stats.selected_batch == grouped.stats.selected_batch,
                        "prefetch and grouping must not reduce the admitted single-layer batch");
                for (const auto& group : planned.stats.plan) {
                    if (group.first_layer + group.layers < file.geometry().n_layers &&
                        lookahead.stats.plan[group.first_layer].prefetch_layers)
                        require(group.prefetch_layers > 0, "grouping must not consume next-layer prefetch capacity");
                }
                if (planned.stats.groups > grouped.stats.groups && planned.stats.prefetched_layers > 0)
                    exercised_priority = true;
            }
            require(exercised_priority, "tight-budget test must demonstrate prefetch taking priority over grouping");
        }
        for (int invalid = 0; invalid < 11; ++invalid) {
            auto bad = options;
            if (invalid == 0) bad.gpu_warm = true;
            if (invalid == 1) bad.gpu_heat_quotas = true;
            if (invalid == 2) bad.gpu_stage_slots = 9;
            if (invalid == 3) { bad.gpu_stage_slots = 1; bad.gpu_experts_mib = -1; }
            if (invalid == 4) {
                bad.full_layer_prefill = true; bad.gpu_warm = true; bad.gpu_experts_mib = -1;
            }
            if (invalid == 5) bad.prefill_auto = true;
            if (invalid == 6) bad.prefill_pipeline = true;
            if (invalid == 7) { bad.full_layer_prefill = true; bad.prefill_group_layers = 65; }
            if (invalid == 8) { bad.full_layer_prefill = true; bad.prefill_prefetch_layers = -2; }
            if (invalid == 9) { bad.full_layer_prefill = true; bad.prefill_transfer_mib = 0; }
            if (invalid == 10) bad.prefill_expert_rows = true;
            bool refused = false;
            try { Model invalid_model(path.string(), bad); }
            catch (const std::invalid_argument&) { refused = true; }
            require(refused, "invalid expert-cache feature combinations must fail explicitly");
        }
        {
            ModelFile file(path.string());
            const auto& g = file.geometry();
            auto high = options;
            high.full_layer_prefill = true;
            high.dsa = true;
            high.mtp = 0;
            high.max_context = max_context_limit;
            high.prefill = 8;
            high.prefill_gpu_mib = 512;
            high.prefill_host_mib = 1024;
            for (const int64_t position : {131071, 199999}) {
                HostState initial;
                initial.position = position;
                initial.dsa = true;
                initial.dsa_tail = g.idx_select_tail;
                initial.logits.resize(g.n_vocab);
                initial.layers.resize(g.n_layers);
                for (int l = 0; l < g.n_layers; ++l) {
                    auto& image = initial.layers[l];
                    if (!strata::core::is_qsa_layer(g, l)) image.recurrent.resize(g.kda_state_floats());
                    else {
                        image.kv.resize(position * g.kv_lora_rank);
                        image.indexer.resize((2 * g.idx_kpool + position / g.idx_kpool) * g.idx_key_dim);
                        const auto last_pool = position / g.idx_kpool - 1;
                        for (int d = 0; d < g.idx_key_dim; ++d)
                            image.indexer[(2 * g.idx_kpool + last_pool) * g.idx_key_dim + d] = 1.0f;
                    }
                }
                initial.validate(g, high.max_context, false, true);
                const std::vector<int32_t> suffix{1, 2, 3, 4, 5, 6, 7};
                const auto scheduled = full_layer_prefill(file, high, suffix, initial);
                auto reference = high;
                reference.prefill_reference = true;
                const auto serial = full_layer_prefill(file, reference, suffix, initial);
                require(scheduled.state.position == position + 7, "high-position prefill truncated its context");
                compare_state(scheduled.state, serial.state);
                DecodeExecutor left(file, high, &scheduled.state), right(file, reference, &serial.state);
                near(left.token(8), right.token(8), 1e-5f, "high-position sparse continuation");
            }
            auto invalid = high;
            invalid.max_context = max_context_limit + 1;
            bool refused = false;
            try { Model model(path.string(), invalid); }
            catch (const std::invalid_argument&) { refused = true; }
            require(refused, "context above the validated implementation limit must fail");
        }
        {
            glm_fixture::write_model(path, false);
            Options plain;
            plain.full_layer_prefill = true; plain.prefill = 7; plain.max_context = 96;
            plain.prefill_gpu_mib = 32; plain.prefill_reserve_mib = 128;
            plain.gpu_experts_mib = 2; plain.gpu_reserve_mib = 128;
            for (int mtp : {0, 3}) {
                plain.mtp = mtp;
                auto instrumented = plain; instrumented.probe = true;
                Model reference(path.string(), plain), measured(path.string(), instrumented);
                const std::vector<int32_t> input(19, 1);
                require(reference.prefill(input) == measured.prefill(input), "probes changed prefill logits");
                const auto stats = measured.prefill_stats();
                const auto sample = [&](ProbePhase phase) -> const ProbeSample& {
                    return stats.probe.phases[static_cast<size_t>(phase)];
                };
                require(stats.probe.rows == input.size() && stats.probe.event_pairs > 0 &&
                        stats.probe.event_pairs <= 256 && stats.probe.forced_waits == 0,
                        "probe events must stay bounded without forced waits");
                require(sample(ProbePhase::ActivationH2D).bytes == stats.activation_upload_bytes &&
                        sample(ProbePhase::ActivationD2H).bytes == stats.activation_download_bytes &&
                        sample(ProbePhase::WeightWait).bytes == stats.weight_upload_bytes,
                        "probed transfer bytes differ from executed full-layer traffic");
                require(sample(ProbePhase::LayerPre).stream_calls > 0 &&
                        sample(ProbePhase::GpuExperts).stream_calls > 0, "prefill GPU probes did not run");
                for (auto phase : {ProbePhase::MlaQuery, ProbePhase::MlaAbsorb, ProbePhase::MlaKv,
                                   ProbePhase::MlaAttention, ProbePhase::MlaDeabsorb,
                                   ProbePhase::MlaOutput, ProbePhase::MlaLayout})
                    require(sample(phase).stream_calls > 0, "fine MLA probe did not run");
                require(reference.generate(4) == measured.generate(4), "probes changed generated IDs");
                const auto decode = measured.decode_probe_stats();
                require(decode.rows >= 4 && decode.cpu_experts > 0 &&
                        decode.phases[static_cast<size_t>(ProbePhase::CpuExperts)].host_calls > 0 &&
                        decode.phases[static_cast<size_t>(ProbePhase::RouteD2H)].bytes > 0,
                        "decode CPU/route probes did not run");
                require(reference.prefill_stats().probe.event_pairs == 0 &&
                        reference.decode_probe_stats().rows == 0, "disabled probes must stay inactive");
                measured.reset();
                require(measured.decode_probe_stats().rows == 0, "reset retained old probe counters");
            }
        }
        {
            PhaseProbe probe(true);
            bool failed = false;
            try {
                ProbeScope scope(probe);
                probe.gpu(ProbePhase::Head, nullptr, [] { throw std::invalid_argument("probe unwind test"); });
            } catch (const std::invalid_argument&) { failed = true; }
            require(failed && current_probe == nullptr, "probe scope leaked after an operation failed");
            Buffer output(sizeof(float));
            probe.gpu(ProbePhase::Head, nullptr, [&] {
                check(cudaMemsetAsync(output.get(), 0, sizeof(float)), "probe retry");
            });
            require(probe.snapshot().phases[static_cast<size_t>(ProbePhase::Head)].stream_calls == 1,
                    "failed probe operation retained an unfinished event pair");
        }
        {
            Options plain; plain.prefill = 7; plain.max_context = 96; plain.gpu_experts_mib = 2;
            auto instrumented = plain; instrumented.probe = true;
            Model reference(path.string(), plain), measured(path.string(), instrumented);
            const std::vector<int32_t> input(19, 1);
            require(reference.prefill(input) == measured.prefill(input), "resident probes changed prefill");
            const auto before = measured.decode_probe_stats();
            require(before.rows == input.size(), "resident prefill rows missing from probes");
            require(reference.generate(4) == measured.generate(4), "resident probes changed output IDs");
            const auto delta = probe_delta(measured.decode_probe_stats(), before);
            require(delta.rows == 4 && delta.phases[static_cast<size_t>(ProbePhase::Head)].stream_calls == 4,
                    "decode interval must exclude resident prefill");
            bool failed = false;
            try { probe_delta(ProbeStats{}, before); } catch (const std::logic_error&) { failed = true; }
            require(failed, "probe intervals must reject counter resets");
        }
        expert_grouping(path);
        transfer_grouping(path);
        std::filesystem::remove(path);
        std::puts("glm5 full-layer: complete banks, budgets, KDA/MLA/MTP handoff and lazy decode passed");
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(path);
        std::fprintf(stderr, "glm5 full-layer test: %s\n", e.what());
        return 1;
    }
}
