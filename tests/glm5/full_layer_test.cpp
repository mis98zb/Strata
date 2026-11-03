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
    }
    glm_fixture::near_fp16(a.mtp.kv, b.mtp.kv, 0.002f, "MTP prompt cache across scheduling");
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
        std::filesystem::remove(path);
        std::puts("glm5 full-layer: complete banks, budgets, KDA/MLA/MTP handoff and lazy decode passed");
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(path);
        std::fprintf(stderr, "glm5 full-layer test: %s\n", e.what());
        return 1;
    }
}
