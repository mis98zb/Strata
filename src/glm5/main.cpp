#include "strata/glm5/model.hpp"
#include "cpu.hpp"
#include "phase_probe.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int64_t integer(const std::string& value, const std::string& option) {
    int64_t number = 0;
    const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), number);
    if (ec != std::errc{} || end != value.data() + value.size())
        throw std::runtime_error(option + ": expected an integer");
    return number;
}

void usage() {
    std::puts("Experimental strata-glm5 (based on Strata PR #1315)\n"
              "Usage: strata-glm5 --model MODEL.gguf --ids ID,ID,... [--n 32]\n"
              "                  [--ids-file CSV_FILE instead of --ids]\n"
              "                  [--gpu 0] [--max-context 1..262144 (default 4096)] [--dump-logits FILE]\n"
              "                  [--prefill 1..4096] (default: full-layer 2048, resident 1) [--mtp 0..7]\n"
              "                  [--cpu-threads 1..256]\n"
              "                  [--dsa] [--flash-attn] (Flash: CUDA SM80+; Flash MTP requires --dsa)\n"
              "                  [--dsa-tail 0|1] (default: model/Strata policy)\n"
              "                  [--prefill-full-layer] [--prefill-gpu-mib MiB]\n"
              "                  [--prefill-host-mib MiB] [--prefill-reserve-mib MiB]\n"
              "                  [--prefill-group-layers 1..64] [--prefill-prefetch-layers 0..64]\n"
              "                  [--prefill-transfer-mib MiB]\n"
              "                  [--prefill-no-pipeline]\n"
              "                  [--reference] (developer comparison; no automatic optimization)\n"
              "                  [--dsa-score auto|scalar|parallel] (diagnostic override; requires --dsa)\n"
              "                  [--glm-gpu-experts MiB] [--glm-gpu-reserve MiB] [--glm-gpu-check]\n"
              "                  [--glm-gpu-warm] [--glm-gpu-heat-quotas] [--glm-gpu-stage 0..8]\n"
              "                  [--timings] [--probe] (opt-in host/stream phase diagnostics)\n"
              "GPU expert budget: default auto on validated SM86 GLM geometry, off elsewhere.\n"
              "Override: -1 off, 0 free VRAM minus reserve, positive MiB cap; reserve default 1024 MiB.\n"
              "Full-layer defaults: automatic batch, pipeline, next-layer prefetch before grouping.\n"
              "Validated GLM compute paths are selected automatically on CUDA SM86; other devices keep the prior path.\n"
              "Reference: single-layer synchronous execution, default batch 1; --prefill can override it.\n"
              "Single GPU, greedy text tokens. All GGUF shards must be present.\n"
              "Prints generated token IDs as CSV. No chat template is applied.\n"
              "No multi-GPU, vision or API server in this port.");
}

void print_probe(const char* label, const strata::glm5::ProbeStats& stats) {
    using namespace strata::glm5;
    std::fprintf(stderr, "glm5 probe %s: rows=%llu cpu_experts=%llu event_pairs=%llu forced_waits=%llu cpu_result_copies=%llu\n", label,
                 static_cast<unsigned long long>(stats.rows), static_cast<unsigned long long>(stats.cpu_experts),
                 static_cast<unsigned long long>(stats.event_pairs), static_cast<unsigned long long>(stats.forced_waits),
                 static_cast<unsigned long long>(stats.cpu_result_copies));
    for (size_t i = 0; i < stats.phases.size(); ++i) {
        const auto& p = stats.phases[i];
        if (!p.host_calls && !p.stream_calls && !p.bytes) continue;
        std::fprintf(stderr, "glm5 probe %s %s: host_ms=%.3f process_cpu_ms=%.3f stream_ms=%.3f "
                     "host_calls=%llu stream_calls=%llu bytes=%llu\n", label, probe_name(static_cast<ProbePhase>(i)),
                     p.host_ms, p.process_cpu_ms, p.stream_ms, static_cast<unsigned long long>(p.host_calls),
                     static_cast<unsigned long long>(p.stream_calls), static_cast<unsigned long long>(p.bytes));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string path, prompt, dump, ids_file;
        bool ids_argument = false;
        int64_t gpu = 0, context = 4096, count = 32;
        strata::glm5::Options options;
        bool full_budget_flags = false;
        bool timings = false;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") { usage(); return 0; }
            if (key == "--timings") { timings = true; continue; }
            if (key == "--probe") { options.probe = true; timings = true; continue; }
            if (key == "--reference") { options.reference = true; continue; }
            if (key == "--dsa") { options.dsa = true; continue; }
            if (key == "--flash-attn") { options.flash_attn = true; continue; }
            if (key == "--glm-gpu-check") { options.gpu_check = true; continue; }
            if (key == "--glm-gpu-warm") { options.gpu_warm = true; continue; }
            if (key == "--glm-gpu-heat-quotas") { options.gpu_heat_quotas = true; continue; }
            if (key == "--prefill-full-layer") { options.full_layer_prefill = true; continue; }
            if (key == "--prefill-auto") { options.prefill_auto = true; full_budget_flags = true; continue; }
            if (key == "--prefill-pipeline") { options.prefill_pipeline = true; full_budget_flags = true; continue; }
            if (key == "--prefill-no-pipeline") { options.prefill_no_pipeline = true; full_budget_flags = true; continue; }
            if (key == "--prefill-reference") {
                options.reference = true; options.prefill_reference = true; full_budget_flags = true; continue;
            }
            if (key != "--model" && key != "--ids" && key != "--ids-file" && key != "--n" && key != "--gpu" &&
                key != "--max-context" && key != "--dump-logits" && key != "--prefill" &&
                key != "--mtp" && key != "--cpu-threads" && key != "--glm-gpu-experts" && key != "--glm-gpu-reserve" &&
                key != "--glm-gpu-stage" &&
                key != "--dsa-tail" && key != "--dsa-score" &&
                key != "--prefill-group-layers" && key != "--prefill-prefetch-layers" &&
                key != "--prefill-transfer-mib" &&
                key != "--prefill-gpu-mib" && key != "--prefill-host-mib" && key != "--prefill-reserve-mib")
                throw std::runtime_error("unknown option: " + key);
            if (++i == argc) throw std::runtime_error(key + " needs a value");
            const std::string value = argv[i];
            if (key == "--model") path = value;
            else if (key == "--ids") { prompt = value; ids_argument = true; }
            else if (key == "--ids-file") {
                if (value.empty()) throw std::runtime_error("--ids-file needs a nonempty path");
                ids_file = value;
            }
            else if (key == "--dump-logits") dump = value;
            else if (key == "--dsa-score") {
                if (value == "auto") options.dsa_score = strata::glm5::DsaScore::Automatic;
                else if (value == "scalar") options.dsa_score = strata::glm5::DsaScore::Scalar;
                else if (value == "parallel") options.dsa_score = strata::glm5::DsaScore::Parallel;
                else throw std::runtime_error(key + ": expected auto, scalar or parallel");
            }
            else if (key == "--n") count = integer(value, key);
            else if (key == "--gpu") gpu = integer(value, key);
            else if (key == "--dsa-tail") {
                const auto n = integer(value, key);
                if (n < 0 || n > 1) throw std::runtime_error(key + ": expected 0 or 1");
                options.dsa_tail = static_cast<int>(n);
            }
            else if (key == "--glm-gpu-stage") {
                const auto n = integer(value, key);
                if (n < 0 || n > 8) throw std::runtime_error(key + ": value outside supported range");
                options.gpu_stage_slots = static_cast<int>(n);
            }
            else if (key == "--cpu-threads") {
                const auto n = integer(value, key);
                if (n < 1 || n > 256) throw std::runtime_error(key + ": value outside supported range");
                options.cpu_threads = static_cast<int>(n);
            }
            else if (key == "--prefill" || key == "--mtp") {
                const auto n = integer(value, key);
                if (n < (key == "--prefill" ? 1 : 0) || n > (key == "--prefill" ? 4096 : 7))
                    throw std::runtime_error(key + ": value outside supported range");
                if (key == "--prefill") options.prefill = static_cast<int>(n);
                else options.mtp = static_cast<int>(n);
            }
            else if (key == "--glm-gpu-experts") {
                options.gpu_experts_mib = integer(value, key);
                if (options.gpu_experts_mib < -1)
                    throw std::runtime_error("--glm-gpu-experts: expected -1, 0 or a positive MiB cap");
            }
            else if (key == "--prefill-group-layers" || key == "--prefill-prefetch-layers") {
                const auto n = integer(value, key);
                if (n < (key == "--prefill-group-layers" ? 1 : 0) || n > 64)
                    throw std::runtime_error(key + ": value outside supported range");
                if (key == "--prefill-group-layers") options.prefill_group_layers = static_cast<int>(n);
                else options.prefill_prefetch_layers = static_cast<int>(n);
                full_budget_flags = true;
            }
            else if (key == "--prefill-transfer-mib") {
                const auto n = integer(value, key);
                if (n < 1 || n > 131072) throw std::runtime_error(key + ": invalid MiB budget");
                options.prefill_transfer_mib = static_cast<uint64_t>(n);
                full_budget_flags = true;
            }
            else if (key == "--prefill-gpu-mib" || key == "--prefill-host-mib" || key == "--prefill-reserve-mib") {
                const auto n = integer(value, key);
                const auto maximum = key == "--prefill-host-mib" ? 1048576 : 131072;
                if (n < (key == "--prefill-host-mib" ? 1 : 0) || n > maximum)
                    throw std::runtime_error(key + ": invalid MiB budget");
                if (key == "--prefill-gpu-mib") options.prefill_gpu_mib = n;
                else if (key == "--prefill-host-mib") options.prefill_host_mib = n;
                else options.prefill_reserve_mib = n;
                full_budget_flags = true;
            }
            else if (key == "--glm-gpu-reserve") {
                const auto n = integer(value, key);
                if (n < 0 || n > 131072) throw std::runtime_error(key + ": invalid MiB reserve");
                options.gpu_reserve_mib = static_cast<uint64_t>(n);
            }
            else context = integer(value, key);
        }
        if (full_budget_flags && !options.full_layer_prefill)
            throw std::runtime_error("prefill budgets require --prefill-full-layer");
        if (options.reference && (options.prefill_auto || options.prefill_pipeline))
            throw std::runtime_error("--reference cannot be combined with --prefill-auto or --prefill-pipeline");
        if (!ids_file.empty()) {
            if (ids_argument) throw std::runtime_error("--ids and --ids-file are mutually exclusive");
            std::ifstream file(ids_file, std::ios::binary | std::ios::ate);
            if (!file) throw std::runtime_error("cannot open token IDs: " + ids_file);
            const auto bytes = file.tellg();
            if (bytes <= 0 || bytes > 4 * 1024 * 1024)
                throw std::runtime_error("--ids-file must contain 1 byte to 4 MiB of CSV IDs");
            prompt.resize(static_cast<size_t>(bytes));
            file.seekg(0);
            file.read(prompt.data(), prompt.size());
            if (!file) throw std::runtime_error("cannot read token IDs: " + ids_file);
        }
        if (path.empty() || prompt.empty()) throw std::runtime_error("--model and token IDs are required (see --help)");
        if (gpu < 0 || gpu > INT32_MAX || count < 0 || context < 1 ||
            context > strata::glm5::max_context_limit || count > context)
            throw std::runtime_error("invalid GPU index, generation length or context");
        std::vector<int32_t> ids;
        size_t start = 0;
        while (true) {
            const auto end = prompt.find(',', start);
            auto text = prompt.substr(start, end - start);
            if (!ids_file.empty()) {
                const auto first = text.find_first_not_of(" \t\r\n");
                if (first == std::string::npos) throw std::runtime_error("--ids-file: empty CSV entry");
                text = text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
            }
            const auto value = integer(text, "--ids");
            if (value < 0 || value > INT32_MAX) throw std::runtime_error("--ids: token outside int32 range");
            ids.push_back(static_cast<int32_t>(value));
            if (end == std::string::npos) break;
            start = end + 1;
        }
        if (ids.size() > static_cast<uint64_t>(context) ||
            static_cast<uint64_t>(count) > context - ids.size())
            throw std::runtime_error("prompt plus generation exceeds --max-context");
        options.gpu = static_cast<int>(gpu);
        options.max_context = context;
        if (options.prefill_pipeline && options.prefill_no_pipeline)
            throw std::runtime_error("--prefill-pipeline and --prefill-no-pipeline are mutually exclusive");
        options = strata::glm5::resolve_options(options);
        using Clock = std::chrono::steady_clock;
        const auto init_start = Clock::now();
        strata::glm5::Model model(path, options);
        options = model.execution_options();
        std::fprintf(stderr, "glm5 compute: policy=%s skip_unused_quant=%d kda_register_state=%d expert_rows=%d gpu_single=%d mla_layout=%s kv_load=%s\n",
                     options.reference || options.prefill_reference ? "reference" : "auto",
                     options.skip_unused_quant, options.kda_register_state, options.prefill_expert_rows,
                     options.gpu_expert_mode == strata::glm5::GpuExpertMode::Single,
                     options.flash_attn && options.mla_attention_layout == strata::glm5::MlaAttentionLayout::HeadMajor ?
                         "head" : "token",
                     options.flash_attn && options.flash_kv_load == strata::glm5::FlashKvLoad::Prefetch ?
                         "prefetch" : "direct");
        if (options.dsa || options.flash_attn)
            std::fprintf(stderr, "glm5 attention: dsa=%d flash=%d tail=%lld query_window=64 score=%s topk=%s score_load=%s\n",
                         options.dsa, options.flash_attn, static_cast<long long>(model.geometry().idx_select_tail),
                         options.dsa_score == strata::glm5::DsaScore::Scalar ? "scalar" :
                         options.dsa_score == strata::glm5::DsaScore::Parallel ? "parallel" :
                         options.dsa_score == strata::glm5::DsaScore::Tiled ? "tiled" :
                         options.flash_attn ? "parallel(auto)" : "scalar(auto)",
                         options.dsa_parallel_topk ? "parallel" : "scalar",
                         options.dsa && options.dsa_score == strata::glm5::DsaScore::Tiled &&
                             options.dsa_score_load == strata::glm5::DsaScoreLoad::Prefetch ? "prefetch" : "direct");
        const auto init_end = Clock::now();
        for (const auto id : ids)
            if (id >= model.geometry().n_vocab) throw std::runtime_error("--ids: token outside model vocabulary");
        const auto prefill_start = Clock::now();
        const auto& logits = model.prefill(ids);
        const auto prefill_end = Clock::now();
        const auto resident_prefill_probe = options.probe ? model.decode_probe_stats() : strata::glm5::ProbeStats{};
        if (options.full_layer_prefill) {
            const auto stats = model.prefill_stats();
            std::fprintf(stderr, "glm5 uploads: calls=%llu staging_peak_mib=%.2f\n",
                         static_cast<unsigned long long>(stats.weight_upload_calls),
                         stats.weight_staging_peak_bytes / 1048576.0);
            std::fprintf(stderr, "glm5 full-layer prefill: layers=%llu experts=%llu tracked GPU peak=%.2f MiB host tensors=%.2f MiB\n",
                         static_cast<unsigned long long>(stats.layer_loads),
                         static_cast<unsigned long long>(stats.experts_loaded),
                         stats.gpu_peak_bytes / 1048576.0, stats.host_tensor_bytes / 1048576.0);
            std::fprintf(stderr, "glm5 prefill matrices: max_columns=%llu MMQ=%llu FP16_GEMM=%llu F32_GEMM=%llu "
                         "native_quant=%llu reused_quant=%llu bounds_launches=%llu reused_bounds=%llu\n",
                         static_cast<unsigned long long>(stats.max_matrix_columns),
                         static_cast<unsigned long long>(stats.mmq_calls),
                         static_cast<unsigned long long>(stats.f16_gemm_calls),
                         static_cast<unsigned long long>(stats.f32_gemm_calls),
                         static_cast<unsigned long long>(stats.native_quant_calls),
                         static_cast<unsigned long long>(stats.reused_quant_calls),
                         static_cast<unsigned long long>(stats.bounds_launches),
                         static_cast<unsigned long long>(stats.reused_bounds));
            if (options.prefill_auto || options.prefill_group_layers || options.prefill_prefetch_layers ||
                options.prefill_pipeline) {
                std::fprintf(stderr,
                    "glm5 prefill plan: batch=%llu groups=%llu max_layers=%llu attempts=%llu planning_ms=%.3f "
                    "prefetched=%llu fused_head=%d transfer_peak_mib=%.2f weights_mib=%.2f activation_H2D_mib=%.2f activation_D2H_mib=%.2f "
                    "group_policy=%s unprefetched_weights_mib=%.2f\n",
                    static_cast<unsigned long long>(stats.selected_batch),
                    static_cast<unsigned long long>(stats.groups),
                    static_cast<unsigned long long>(stats.max_group_layers),
                    static_cast<unsigned long long>(stats.planning_attempts), stats.planning_ms,
                    static_cast<unsigned long long>(stats.prefetched_layers),
                    stats.fused_head,
                    stats.transfer_peak_bytes / 1048576.0, stats.weight_upload_bytes / 1048576.0,
                    stats.activation_upload_bytes / 1048576.0, stats.activation_download_bytes / 1048576.0,
                    options.prefill_auto && options.prefill_pipeline && options.prefill_transfer_cost &&
                        !options.prefill_group_layers ?
                        "transfer" : "capacity",
                    stats.unprefetched_weight_bytes / 1048576.0);
                for (const auto& group : stats.plan)
                    std::fprintf(stderr, "glm5 prefill group: first=%d layers=%d prefetch_next=%d\n",
                                 group.first_layer, group.layers, group.prefetch_layers);
            }
        }
        if (!dump.empty()) {
            std::ofstream out(dump, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
            if (!out) throw std::runtime_error("cannot write logits to " + dump);
        }
        const auto setup_start = Clock::now();
        if (timings && count > 0) model.prepare_decode();
        const auto setup_end = Clock::now();
        const auto decode_start = Clock::now();
        const auto generated = model.generate(static_cast<int>(count));
        const auto decode_end = Clock::now();
        if (options.probe) {
            std::fprintf(stderr, "glm5 probe CPU ISA: %s\n", strata::glm5::cpu_isa_summary().c_str());
            std::fprintf(stderr, "glm5 probe: stream spans include launch starvation; phases overlap; "
                         "cpu_experts bytes are logical weight reads, not measured DRAM traffic\n");
            print_probe("prefill", options.full_layer_prefill ? model.prefill_stats().probe : resident_prefill_probe);
            print_probe("decode", strata::glm5::probe_delta(model.decode_probe_stats(), resident_prefill_probe));
        }
        if (timings) {
            const auto ms = [](auto from, auto to) {
                return std::chrono::duration<double, std::milli>(to - from).count();
            };
            const double pp_ms = ms(prefill_start, prefill_end);
            const double tg_ms = ms(decode_start, decode_end);
            std::fprintf(stderr,
                         "glm5 timings: init_ms=%.3f prefill_ms=%.3f decode_setup_ms=%.3f decode_ms=%.3f "
                         "prompt_tokens=%zu generated_tokens=%zu prefill_tps=%.6f decode_tps=%.6f\n",
                         ms(init_start, init_end), pp_ms, ms(setup_start, setup_end), tg_ms,
                         ids.size(), generated.size(), pp_ms > 0 ? ids.size() * 1000.0 / pp_ms : 0.0,
                         !generated.empty() && tg_ms > 0 ? generated.size() * 1000.0 / tg_ms : 0.0);
        }
        for (size_t i = 0; i < generated.size(); ++i) std::printf("%s%d", i ? "," : "", generated[i]);
        std::puts("");
        if (options.gpu_experts_mib >= 0) {
            const auto stats = model.cache_stats();
            std::fprintf(stderr, "glm5 cache: hits=%llu misses=%llu streamed=%llu swaps=%llu\n",
                         static_cast<unsigned long long>(stats.hits), static_cast<unsigned long long>(stats.misses),
                         static_cast<unsigned long long>(stats.streamed), static_cast<unsigned long long>(stats.swaps));
            if (stats.direct_rows)
                std::fprintf(stderr, "glm5 single experts: rows=%llu input_quant=%llu avoided_input_quant=%llu avoided_d2d_bytes=%llu\n",
                             static_cast<unsigned long long>(stats.direct_rows),
                             static_cast<unsigned long long>(stats.single_input_quantizations),
                             static_cast<unsigned long long>(stats.avoided_input_quantizations),
                             static_cast<unsigned long long>(stats.avoided_d2d_bytes));
            if (options.gpu_warm || options.gpu_heat_quotas || options.gpu_stage_slots)
                std::fprintf(stderr, "glm5 cache setup: slots=%llu warmed=%llu pinned_mib=%.2f uploads_mib=%.2f\n",
                             static_cast<unsigned long long>(stats.slots),
                             static_cast<unsigned long long>(stats.warmed),
                             stats.pinned_bytes / 1048576.0, stats.upload_bytes / 1048576.0);
        }
        if (options.mtp) {
            const auto stats = model.mtp_stats();
            std::fprintf(stderr, "glm5 MTP: offered=%llu accepted=%llu verify_batches=%llu replays=%llu target_batches=%llu KDA_replay_layers=%llu DSA_replay_layers=%llu\n",
                         static_cast<unsigned long long>(stats.offered), static_cast<unsigned long long>(stats.accepted),
                         static_cast<unsigned long long>(stats.verify_batches), static_cast<unsigned long long>(stats.replays),
                         static_cast<unsigned long long>(stats.target_batches),
                         static_cast<unsigned long long>(stats.replayed_layers),
                         static_cast<unsigned long long>(stats.dsa_replayed_layers));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm5: %s\n", e.what());
        return 1;
    }
}
