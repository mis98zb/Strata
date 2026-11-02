#include "strata/glm5/model.hpp"
#include "fixture.hpp"
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

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void write_model(const std::filesystem::path& path, bool reject) {
    auto md = glm_fixture::metadata();
    md[7] = fixture::u32("glm5-next.expert_feed_forward_length", 768);
    md[11] = fixture::u32s("glm5-next.swiglu_clamp_exp", {1, 1, 1, 1, 1});
    auto rows = glm_fixture::tensors(true, 768);
    rows.push_back({"blk.4.nextn.shared_head_head.weight", {256, 32}, 8, 1});
    for (auto& row : rows) {
        const auto& name = row.name;
        if (row.shape.size() >= 2 && name != "token_embd.weight" &&
            name.find("ffn_gate_inp") == std::string::npos && name.find("ssm_conv") == std::string::npos)
            row.type = GGML_TYPE_Q8_0;
    }
    const auto written = fixture::write(path, md, rows);
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    for (size_t t = 0; t < rows.size(); ++t) {
        const auto& row = rows[t];
        uint64_t elements = 1;
        for (auto d : row.shape) elements *= d;
        std::vector<float> values(elements);
        uint32_t seed = 2166136261u;
        for (const char c : row.name) seed = (seed ^ static_cast<uint8_t>(c)) * 16777619u;
        for (size_t i = 0; i < values.size(); ++i) {
            seed = seed * 1664525u + 1013904223u;
            values[i] = 0.002f + (seed >> 8) * (0.008f / 16777216.0f);
            if (row.name.find("norm") != std::string::npos) values[i] = 1.0f;
            else if (row.name.find("_base.") != std::string::npos || row.name.ends_with(".bias")) values[i] = 0;
            else if (row.name.find("_scale.") != std::string::npos) values[i] = 1.0f;
            else if (row.name.ends_with("ssm_a")) values[i] = -1.0f;
            else if (row.name.find("ssm_conv") != std::string::npos) values[i] = i % 4 == 3 ? 1.0f : 0.0f;
            else if (row.name == "token_embd.weight") values[i] += 0.01f * (1 + i / 256);
            if (row.name.ends_with("ffn_down_exps.weight")) {
                // Opposite expert directions make a wrong slot fail the row-RMS diagnostic.
                const auto expert = i / (row.shape[0] * row.shape[1]);
                const auto channel = (i / row.shape[0]) % row.shape[1];
                values[i] *= ((channel < 128) == (expert == 0)) ? 0.25f : 4.0f;
            }
            if (row.name == "output.weight" || row.name.ends_with("nextn.shared_head_head.weight")) {
                const int chosen = row.name == "output.weight" || !reject ? 0 : 1;
                values[i] = i / 256 == static_cast<size_t>(chosen) ? 0.02f * (1.0f + (i % 17) / 17.0f) : 0;
            }
        }
        file.seekp(static_cast<std::streamoff>(written.data_start + written.offsets[t]));
        if (row.type == GGML_TYPE_Q8_0) {
            const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
            require(traits && traits->from_float, "missing Q8_0 quantizer");
            std::vector<uint8_t> bytes(ggml_row_size(GGML_TYPE_Q8_0, elements));
            traits->from_float(values.data(), bytes.data(), elements);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        } else file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
        require(static_cast<bool>(file), "cannot write synthetic GLM weights");
    }
}

void near(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance, const char* what) {
    require(actual.size() == expected.size() && !actual.empty(), "invalid comparison shape");
    float error = 0, scale = 1e-6f;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "non-finite model logits");
        error = std::max(error, std::abs(actual[i] - expected[i]));
        scale = std::max(scale, std::abs(expected[i]));
    }
    if (error > tolerance * scale)
        throw std::runtime_error(std::string(what) + ": relative max error " + std::to_string(error / scale));
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
        std::vector<int32_t> prompt;
        for (int i = 0; i < 19; ++i) prompt.push_back(1 + i % 17);
        strata::glm5::Options base;
        base.max_context = 96;
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
            strata::glm5::Model target(path.string(), plain);
            strata::glm5::Model draft(path.string(), speculative);
            near(draft.prefill(prompt), target.prefill(prompt), 1e-5f, "MTP prompt target state");
            const auto expected = target.generate(12);
            const auto actual = draft.generate(12);
            require(actual == expected && actual.size() == 12, "MTP changed greedy output");
            const auto stats = draft.mtp_stats();
            require(stats.offered && stats.verify_batches, "MTP must run a real verification batch");
            require(reject ? stats.replays > 0 : stats.accepted > 0, "missing rejection replay or draft acceptance");
            const auto continuation = target.token(17);
            near(draft.token(17), continuation, 1e-5f, "MTP rejection/acceptance state continuation");
            draft.reset();
            draft.prefill(prompt);
            require(draft.generate(12) == expected, "MTP reset leaked draft or target state");
            auto mixed = speculative;
            mixed.prefill = 17;
            mixed.gpu_experts_mib = 2;
            mixed.gpu_check = true;
            strata::glm5::Model accelerated(path.string(), mixed);
            near(accelerated.prefill(prompt), reference, 0.05f, "combined prefill/cache/MTP prompt");
            require(accelerated.generate(12) == expected, "combined accelerators changed greedy output");
            near(accelerated.token(17), continuation, 0.05f, "combined acceleration continuation");
            require(accelerated.cache_stats().streamed && accelerated.mtp_stats().verify_batches,
                    "combined test must execute both GPU streaming and MTP verification");
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
