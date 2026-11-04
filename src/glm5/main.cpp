#include "strata/glm5/model.hpp"

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
              "                  [--gpu 0] [--max-context 4096] [--dump-logits FILE]\n"
              "                  [--prefill 1..4096] [--mtp 0..7]\n"
              "                  [--cpu-threads 1..256]\n"
              "                  [--dsa] [--flash-attn] (Flash: CUDA SM80+; Flash MTP requires --dsa)\n"
              "                  [--dsa-tail 0|1] (default: model/Strata policy)\n"
              "                  [--dsa-score auto|scalar|parallel] (overrides require --dsa)\n"
              "                  [--prefill-full-layer] [--prefill-gpu-mib MiB]\n"
              "                  [--prefill-host-mib MiB] [--prefill-reserve-mib MiB]\n"
              "                  [--glm-gpu-experts MiB] [--glm-gpu-reserve MiB] [--glm-gpu-check]\n"
              "                  [--glm-gpu-warm] [--glm-gpu-heat-quotas] [--glm-gpu-stage 0..8]\n"
              "                  [--timings]\n"
              "GPU expert budget: -1 off (default), 0 free VRAM minus reserve, positive MiB cap.\n"
              "Single GPU, greedy text tokens. All GGUF shards must be present.\n"
              "Prints generated token IDs as CSV. No chat template is applied.\n"
              "No multi-GPU, vision or API server in this port.");
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
            if (key == "--dsa") { options.dsa = true; continue; }
            if (key == "--flash-attn") { options.flash_attn = true; continue; }
            if (key == "--glm-gpu-check") { options.gpu_check = true; continue; }
            if (key == "--glm-gpu-warm") { options.gpu_warm = true; continue; }
            if (key == "--glm-gpu-heat-quotas") { options.gpu_heat_quotas = true; continue; }
            if (key == "--prefill-full-layer") { options.full_layer_prefill = true; continue; }
            if (key != "--model" && key != "--ids" && key != "--ids-file" && key != "--n" && key != "--gpu" &&
                key != "--max-context" && key != "--dump-logits" && key != "--prefill" &&
                key != "--mtp" && key != "--cpu-threads" && key != "--glm-gpu-experts" && key != "--glm-gpu-reserve" &&
                key != "--glm-gpu-stage" &&
                key != "--dsa-tail" && key != "--dsa-score" &&
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
            else if (key == "--glm-gpu-experts") options.gpu_experts_mib = integer(value, key);
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
        if (gpu < 0 || gpu > INT32_MAX || count < 0 || context < 1 || context > 131072 || count > context)
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
        using Clock = std::chrono::steady_clock;
        const auto init_start = Clock::now();
        strata::glm5::Model model(path, options);
        if (options.dsa || options.flash_attn)
            std::fprintf(stderr, "glm5 attention: dsa=%d flash=%d tail=%lld query_window=64 score=%s\n",
                         options.dsa, options.flash_attn, static_cast<long long>(model.geometry().idx_select_tail),
                         options.dsa_score == strata::glm5::DsaScore::Scalar ? "scalar" :
                         options.dsa_score == strata::glm5::DsaScore::Parallel ? "parallel" :
                         options.flash_attn ? "parallel(auto)" : "scalar(auto)");
        const auto init_end = Clock::now();
        for (const auto id : ids)
            if (id >= model.geometry().n_vocab) throw std::runtime_error("--ids: token outside model vocabulary");
        const auto prefill_start = Clock::now();
        const auto& logits = model.prefill(ids);
        const auto prefill_end = Clock::now();
        if (options.full_layer_prefill) {
            const auto stats = model.prefill_stats();
            std::fprintf(stderr, "glm5 full-layer prefill: layers=%llu experts=%llu tracked GPU peak=%.2f MiB host tensors=%.2f MiB\n",
                         static_cast<unsigned long long>(stats.layer_loads),
                         static_cast<unsigned long long>(stats.experts_loaded),
                         stats.gpu_peak_bytes / 1048576.0, stats.host_tensor_bytes / 1048576.0);
            std::fprintf(stderr, "glm5 prefill matrices: max_columns=%llu MMQ=%llu FP16_GEMM=%llu F32_GEMM=%llu\n",
                         static_cast<unsigned long long>(stats.max_matrix_columns),
                         static_cast<unsigned long long>(stats.mmq_calls),
                         static_cast<unsigned long long>(stats.f16_gemm_calls),
                         static_cast<unsigned long long>(stats.f32_gemm_calls));
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
