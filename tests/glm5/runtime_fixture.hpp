#pragma once

#include "fixture.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>

namespace glm_fixture {

inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

inline void write_model(const std::filesystem::path& path, bool reject, bool dsa = false,
                         uint32_t indexer_heads = 4, uint32_t expert_width = 768,
                         std::array<uint32_t, 3> expert_types = {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0}) {
    auto md = glm_fixture::metadata();
    md[7] = fixture::u32("glm5-next.expert_feed_forward_length", expert_width);
    md[11] = fixture::u32s("glm5-next.swiglu_clamp_exp", {1, 1, 1, 1, 1});
    auto rows = glm_fixture::tensors(true, expert_width, dsa);
    if (dsa && indexer_heads != 4) {
        md.push_back(fixture::u32("glm5-next.attention.indexer.head_count", indexer_heads));
        for (auto& row : rows) {
            if (row.name.ends_with("indexer.attn_q_b.weight")) row.shape[1] = 128 * indexer_heads;
            if (row.name.ends_with("indexer.proj.weight")) row.shape[1] = indexer_heads;
        }
    }
    rows.push_back({"blk.4.nextn.shared_head_head.weight", {256, 32}, 8, 1});
    for (auto& row : rows) {
        const auto& name = row.name;
        if (row.shape.size() >= 2 && name != "token_embd.weight" &&
            name.find("ffn_gate_inp") == std::string::npos && name.find("ssm_conv") == std::string::npos)
            row.type = GGML_TYPE_Q8_0;
        if (name.ends_with("ffn_gate_exps.weight")) row.type = expert_types[0];
        if (name.ends_with("ffn_up_exps.weight")) row.type = expert_types[1];
        if (name.ends_with("ffn_down_exps.weight")) row.type = expert_types[2];
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
        if (row.type != GGML_TYPE_F32) {
            const auto* traits = ggml_get_type_traits_cpu(static_cast<ggml_type>(row.type));
            require(traits && traits->from_float, "missing fixture quantizer");
            std::vector<uint8_t> bytes(ggml_row_size(static_cast<ggml_type>(row.type), elements));
            traits->from_float(values.data(), bytes.data(), elements);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        } else file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
        require(static_cast<bool>(file), "cannot write synthetic GLM weights");
    }
}

inline void near(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance, const char* what) {
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

inline void near_fp16(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& expected,
                      float tolerance, const char* what) {
    require(actual.size() == expected.size() && !actual.empty(), "invalid cache comparison shape");
    std::vector<float> x(actual.size()), y(expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        x[i] = ggml_fp16_to_fp32(actual[i]);
        y[i] = ggml_fp16_to_fp32(expected[i]);
    }
    near(x, y, tolerance, what);
}

}  // namespace glm_fixture
