#include "strata/glm5/model.hpp"

#include <algorithm>
#include <charconv>
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
              "                  [--gpu 0] [--max-context 4096] [--dump-logits FILE]\n"
              "                  [--prefill 1..4096] [--mtp 0..7]\n"
              "                  [--glm-gpu-experts MiB] [--glm-gpu-reserve MiB] [--glm-gpu-check]\n"
              "GPU expert budget: -1 off (default), 0 free VRAM minus reserve, positive MiB cap.\n"
              "Single GPU, greedy text tokens. All GGUF shards must be present.\n"
              "Prints generated token IDs as CSV. No chat template is applied.\n"
              "No multi-GPU, vision or API server in this port.");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string path, prompt, dump;
        int64_t gpu = 0, context = 4096, count = 32;
        strata::glm5::Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") { usage(); return 0; }
            if (key == "--glm-gpu-check") { options.gpu_check = true; continue; }
            if (key != "--model" && key != "--ids" && key != "--n" && key != "--gpu" &&
                key != "--max-context" && key != "--dump-logits" && key != "--prefill" &&
                key != "--mtp" && key != "--glm-gpu-experts" && key != "--glm-gpu-reserve")
                throw std::runtime_error("unknown option: " + key);
            if (++i == argc) throw std::runtime_error(key + " needs a value");
            const std::string value = argv[i];
            if (key == "--model") path = value;
            else if (key == "--ids") prompt = value;
            else if (key == "--dump-logits") dump = value;
            else if (key == "--n") count = integer(value, key);
            else if (key == "--gpu") gpu = integer(value, key);
            else if (key == "--prefill" || key == "--mtp") {
                const auto n = integer(value, key);
                if (n < (key == "--prefill" ? 1 : 0) || n > (key == "--prefill" ? 4096 : 7))
                    throw std::runtime_error(key + ": value outside supported range");
                if (key == "--prefill") options.prefill = static_cast<int>(n);
                else options.mtp = static_cast<int>(n);
            }
            else if (key == "--glm-gpu-experts") options.gpu_experts_mib = integer(value, key);
            else if (key == "--glm-gpu-reserve") {
                const auto n = integer(value, key);
                if (n < 0 || n > 131072) throw std::runtime_error(key + ": invalid MiB reserve");
                options.gpu_reserve_mib = static_cast<uint64_t>(n);
            }
            else context = integer(value, key);
        }
        if (path.empty() || prompt.empty()) throw std::runtime_error("--model and --ids are required (see --help)");
        if (gpu < 0 || gpu > INT32_MAX || count < 0 || context < 1 || context > 131072 || count > context)
            throw std::runtime_error("invalid GPU index, generation length or context");
        std::vector<int32_t> ids;
        size_t start = 0;
        while (true) {
            const auto end = prompt.find(',', start);
            const auto value = integer(prompt.substr(start, end - start), "--ids");
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
        strata::glm5::Model model(path, options);
        for (const auto id : ids)
            if (id >= model.geometry().n_vocab) throw std::runtime_error("--ids: token outside model vocabulary");
        const auto& logits = model.prefill(ids);
        if (!dump.empty()) {
            std::ofstream out(dump, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
            if (!out) throw std::runtime_error("cannot write logits to " + dump);
        }
        const auto generated = model.generate(static_cast<int>(count));
        for (size_t i = 0; i < generated.size(); ++i) std::printf("%s%d", i ? "," : "", generated[i]);
        std::puts("");
        if (options.gpu_experts_mib >= 0) {
            const auto stats = model.cache_stats();
            std::fprintf(stderr, "glm5 cache: hits=%llu misses=%llu streamed=%llu swaps=%llu\n",
                         static_cast<unsigned long long>(stats.hits), static_cast<unsigned long long>(stats.misses),
                         static_cast<unsigned long long>(stats.streamed), static_cast<unsigned long long>(stats.swaps));
        }
        if (options.mtp) {
            const auto stats = model.mtp_stats();
            std::fprintf(stderr, "glm5 MTP: offered=%llu accepted=%llu verify_batches=%llu replays=%llu\n",
                         static_cast<unsigned long long>(stats.offered), static_cast<unsigned long long>(stats.accepted),
                         static_cast<unsigned long long>(stats.verify_batches), static_cast<unsigned long long>(stats.replays));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm5: %s\n", e.what());
        return 1;
    }
}
