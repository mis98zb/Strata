#include "strata/glm5/model.hpp"
#include "decode.hpp"
#include "prefill.hpp"
#include "runtime_fixture.hpp"

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
                    for (int mode = 0; mode < 4; ++mode) {
                        auto optimized = reference;
                        optimized.prefill_group_layers = mode == 0 ? 2 : (mode == 1 ? 4 : 1);
                        optimized.prefill_pipeline = mode != 0;
                        optimized.prefill_prefetch_layers = 2;
                        optimized.prefill_auto = mode == 3;
                        const auto grouped = full_layer_prefill(file, optimized, prompt, {});
                        compare_state(grouped.state, serial.state);
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

            defaults.prefill_group_layers = 0;
            defaults.prefill_prefetch_layers = -1;
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
        for (int invalid = 0; invalid < 10; ++invalid) {
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
            bool refused = false;
            try { Model invalid_model(path.string(), bad); }
            catch (const std::invalid_argument&) { refused = true; }
            require(refused, "invalid expert-cache feature combinations must fail explicitly");
        }
        std::filesystem::remove(path);
        std::puts("glm5 full-layer: complete banks, budgets, KDA/MLA/MTP handoff and lazy decode passed");
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(path);
        std::fprintf(stderr, "glm5 full-layer test: %s\n", e.what());
        return 1;
    }
}
