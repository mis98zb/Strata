#include "cpu.hpp"
#include "prefill.hpp"
#include "runtime_fixture.hpp"
#include "weights.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace {
using namespace strata::glm5;
using glm_fixture::require;

void bench(const char* model, const char* ids, const char* stage, const char* output) {
    std::ifstream input(ids);
    if (!input) throw std::runtime_error("upload benchmark: cannot open IDs");
    std::vector<int32_t> tokens;
    std::string item;
    while (std::getline(input, item, ',')) tokens.push_back(std::stoi(item));
    if (tokens.empty()) throw std::runtime_error("upload benchmark: empty IDs");
    Options options;
    options.full_layer_prefill = true; options.dsa = true; options.flash_attn = true; options.dsa_tail = 0;
    options.max_context = tokens.size() + 128;
    options.prefill_gpu_mib = 14336; options.prefill_host_mib = 32768; options.prefill_reserve_mib = 1024;
    options.prefill_weight_stage_mib = std::stoul(stage);
    Model runtime(model, options);
    const auto start = std::chrono::steady_clock::now();
    const auto& logits = runtime.prefill(tokens);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::ofstream scores(output, std::ios::binary);
    scores.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
    if (!scores) throw std::runtime_error("upload benchmark: cannot save scores");
    const auto stats = runtime.prefill_stats();
    std::printf("upload benchmark: tokens=%zu stage=%s seconds=%.6f tps=%.6f upload_calls=%llu "
                "staging_mib=%.6f batch=%llu groups=%llu prefetch=%llu gpu_peak=%llu host=%llu transfer=%llu\n",
                tokens.size(), stage, seconds, tokens.size() / seconds,
                static_cast<unsigned long long>(stats.weight_upload_calls), stats.weight_staging_peak_bytes / 1048576.0,
                static_cast<unsigned long long>(stats.selected_batch), static_cast<unsigned long long>(stats.groups),
                static_cast<unsigned long long>(stats.prefetched_layers),
                static_cast<unsigned long long>(stats.gpu_peak_bytes),
                static_cast<unsigned long long>(stats.host_tensor_bytes),
                static_cast<unsigned long long>(stats.transfer_peak_bytes));
}

void banks(const ModelFile& file) {
    const auto select = [](const auto& t) { return t.name.starts_with("blk.3."); };
    uint64_t previous_calls = 0;
    for (int mib : {1, 8}) {
        MemoryBudget budget(64 * 1048576ULL);
        WeightBank bank(file, select, &budget, mib * 1048576ULL, true);
        bank.start_upload();
        bank.wait_ready();
        require(bank.uploaded_bytes() == WeightBank::upload_bytes(file, select),
                "planned transfer bytes must equal actual tensor uploads, excluding allocation scratch");
        require(bank.uploaded_bytes() > 8 * 1048576ULL && bank.upload_calls() > 0, "upload fixture must span many chunks");
        if (previous_calls) require(bank.upload_calls() < previous_calls, "larger ring must reduce actual upload calls");
        previous_calls = bank.upload_calls();
        for (size_t s = 0; s < file.gguf().size(); ++s) {
            for (const auto& tensor : file.gguf().shard(s).tensors()) {
                if (!select(tensor)) continue;
                const auto* ref = bank.table().find(tensor.name);
                require(ref != nullptr, "uploaded tensor missing");
                const size_t bytes = ref->native_data ? tensor_payload_bytes(tensor) : ref->bytes;
                std::vector<uint8_t> actual(bytes);
                check(cudaMemcpy(actual.data(), ref->native_data ? ref->native_data : ref->data,
                                   bytes, cudaMemcpyDeviceToHost), "upload tensor verification");
                require(std::memcmp(actual.data(), file.data(tensor.name), bytes) == 0,
                        "staged upload changed tensor payload bytes");
            }
        }
    }
}

void budget(const ModelFile& file) {
    Options base;
    base.reference = true; base.full_layer_prefill = true; base.prefill_auto = true;
    base.prefill = 7; base.prefill_group_layers = 1; base.prefill_prefetch_layers = 1;
    base.prefill_gpu_mib = 32; base.prefill_host_mib = 64; base.prefill_reserve_mib = 128;
    base.max_context = 96;
    std::vector<int32_t> tokens(19, 1);
    base.prefill_weight_stage_mib = 1;
    const auto reference = full_layer_prefill(file, base, tokens, {});
    auto larger = base; larger.prefill_weight_stage_mib = 8;
    const auto actual = full_layer_prefill(file, larger, tokens, {});
    require(actual.state.logits == reference.state.logits && actual.stats.plan.size() == reference.stats.plan.size() &&
            actual.stats.prefetched_layers == reference.stats.prefetched_layers,
            "ring granularity must not change arithmetic or prefetch admission");
    require(actual.stats.weight_staging_peak_bytes == 8 * 1048576ULL, "larger ring was not executed");
    require(actual.stats.host_tensor_bytes == reference.stats.host_tensor_bytes + 7 * 1048576ULL,
            "staging increase must be included in the host tensor bound");
    larger.prefill_transfer_mib = 1;
    const auto limited = full_layer_prefill(file, larger, tokens, {});
    require(limited.stats.weight_staging_peak_bytes == 1048576 &&
            limited.stats.selected_batch == actual.stats.selected_batch &&
            limited.stats.prefetched_layers == actual.stats.prefetched_layers &&
            limited.state.logits == actual.state.logits, "staging must shrink before batch or prefetch is sacrificed");
}
}

int main(int argc, char** argv) {
    const auto path = std::filesystem::temp_directory_path() /
        ("glm5-upload-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        init_cpu();
        if (argc == 6 && std::strcmp(argv[1], "--bench") == 0) {
            bench(argv[2], argv[3], argv[4], argv[5]);
            return 0;
        }
        if (argc != 1) throw std::invalid_argument("Usage: glm5_upload_test [--bench MODEL IDS STAGE_MIB SCORES]");
        glm_fixture::write_model(path, false, false, 4, 8192);
        banks(ModelFile(path.string()));
        glm_fixture::write_model(path, false);
        budget(ModelFile(path.string()));
        std::filesystem::remove(path);
        std::puts("glm5 upload: exact tensor payloads, observed call reduction and adaptive staging budgets passed");
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(path);
        std::fprintf(stderr, "glm5 upload: %s\n", e.what());
        return 1;
    }
}
