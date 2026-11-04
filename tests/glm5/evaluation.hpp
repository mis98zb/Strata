#pragma once

#include "cpu.hpp"
#include "strata/glm5/model.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace glm_evaluation {

inline void projection_delta(const char* model_path, const char* input_path) {
    std::ifstream input(input_path, std::ios::binary);
    std::array<int64_t, 8> h{};
    input.read(reinterpret_cast<char*>(h.data()), sizeof(h));
    if (!input || h[0] != 0x474c4d44 || h[1] < 1 || h[1] > 256 || h[2] < 0 ||
        h[3] < 1 || h[3] > 65536 || h[4] < 1 || h[4] > 65536 || h[3] * h[4] > 128 * 1024 * 1024)
        throw std::runtime_error("invalid projection-delta input header");
    std::string name(h[1], '\0');
    input.read(name.data(), name.size());
    std::vector<float> a(h[3]), b(h[3]), ya(h[4]), yb(h[4]), weights(h[3] * h[4]);
    auto read = [&](auto& data) {
        input.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(data[0]));
        if (!input) throw std::runtime_error("truncated projection-delta payload");
        for (float value : data)
            if (!std::isfinite(value)) throw std::runtime_error("non-finite projection-delta input");
    };
    read(a); read(b); read(ya); read(yb);
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("trailing projection-delta input");
    strata::glm5::init_cpu();
    strata::glm5::ModelFile file(model_path);
    const auto& tensor = file.tensor(name);
    const auto rows = tensor.elements() / tensor.shape.front();
    if (tensor.shape.front() != static_cast<uint64_t>(h[3]) || h[2] > static_cast<int64_t>(rows) ||
        h[4] > static_cast<int64_t>(rows) - h[2] || tensor.type != h[5])
        throw std::runtime_error("projection-delta weight geometry or type mismatch");
    const auto stride = tensor_payload_bytes(tensor) / rows;
    strata::glm5::dequantize(tensor, file.data(name) + h[2] * stride, weights.data(), weights.size());
    double input_power = 0, expected_power = 0, actual_power = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double delta = static_cast<double>(a[i]) - b[i];
        input_power += delta * delta;
    }
    for (size_t r = 0; r < ya.size(); ++r) {
        double delta = 0;
        for (size_t c = 0; c < a.size(); ++c)
            delta += weights[r * a.size() + c] * (static_cast<double>(a[c]) - b[c]);
        expected_power += delta * delta;
        const double observed = static_cast<double>(ya[r]) - yb[r];
        actual_power += observed * observed;
    }
    const double input_rms = std::sqrt(input_power / a.size());
    const double expected = std::sqrt(expected_power / ya.size()), actual = std::sqrt(actual_power / ya.size());
    if (!std::isfinite(expected) || !std::isfinite(actual))
        throw std::runtime_error("non-finite projection-delta result");
    std::printf("{\"row0\":%lld,\"input_rmse\":%.12g,\"linear_double_rmse\":%.12g,"
                "\"actual_quantized_output_rmse\":%.12g}\n",
                static_cast<long long>(h[2]), input_rms, expected, actual);
}

inline void teacher_forced(const char* model_path, const char* input_path, const char* output_path) {
    std::ifstream input(input_path, std::ios::binary);
    std::array<int32_t, 5> header{};
    input.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (!input || header[0] != 0x474c4d45 || header[1] != 1 ||
        header[2] < 1 || header[3] < 1 || header[3] > 256 || header[2] > 131072 - header[3] ||
        header[4] < 0 || header[4] > 1)
        throw std::runtime_error("invalid teacher-forced input header");
    const int prefix = header[2], steps = header[3];
    std::vector<int32_t> tokens(prefix + steps);
    input.read(reinterpret_cast<char*>(tokens.data()), tokens.size() * sizeof(int32_t));
    if (!input || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("invalid teacher-forced token payload");
    const auto output_absolute = std::filesystem::absolute(output_path).lexically_normal();
    if (output_absolute == std::filesystem::absolute(model_path).lexically_normal() ||
        output_absolute == std::filesystem::absolute(input_path).lexically_normal())
        throw std::runtime_error("teacher-forced output must not overwrite an input");
    if (std::filesystem::exists(std::filesystem::symlink_status(output_absolute)))
        throw std::runtime_error("teacher-forced output already exists");
    strata::glm5::Options options;
    options.dsa = true;
    options.dsa_tail = 0;
    options.dsa_score = strata::glm5::DsaScore::Parallel;
    options.flash_attn = (header[4] & 1) != 0;
    options.full_layer_prefill = true;
    options.prefill = 2048;
    options.max_context = tokens.size();
    options.prefill_gpu_mib = 0;
    options.prefill_host_mib = 32768;
    options.prefill_reserve_mib = 1024;
    options.cpu_threads = strata::glm5::cpu_parallel_available() ? 6 : 1;
    strata::glm5::Model model(model_path, options);
    const auto vocabulary = model.geometry().n_vocab;
    if (vocabulary < 2 || vocabulary > 1000000)
        throw std::runtime_error("unsupported teacher-forced vocabulary size");
    const auto vocab = static_cast<int32_t>(vocabulary);
    for (int32_t id : tokens)
        if (id < 0 || id >= vocab) throw std::runtime_error("teacher-forced token outside vocabulary");
    std::ofstream output(output_path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open teacher-forced output");
    const std::array<int32_t, 6> out_header{0x474c4d50, 1, vocab, steps, prefix, header[4]};
    output.write(reinterpret_cast<const char*>(out_header.data()), sizeof(out_header));
    output.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int32_t));
    if (!output) throw std::runtime_error("cannot write teacher-forced output header");
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    std::vector<float> logits = model.prefill(std::vector<int32_t>(tokens.begin(), tokens.begin() + prefix));
    const auto prefilled = Clock::now();
    model.prepare_decode();
    const auto prepared = Clock::now();
    for (int i = 0; i < steps; ++i) {
        if (logits.size() != static_cast<size_t>(vocab))
            throw std::runtime_error("teacher-forced logit shape mismatch");
        for (float value : logits)
            if (!std::isfinite(value)) throw std::runtime_error("non-finite teacher-forced logits");
        const int32_t target = tokens[prefix + i];
        output.write(reinterpret_cast<const char*>(&target), sizeof(target));
        output.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
        if (!output) throw std::runtime_error("cannot write teacher-forced logits");
        if (i + 1 < steps) logits = model.token(target);
    }
    output.flush();
    if (!output) throw std::runtime_error("cannot flush teacher-forced logits");
    const auto scored = Clock::now();
    const auto seconds = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    std::printf("{\"prefix\":%d,\"positions\":%d,\"vocab\":%d,\"flash\":%s,\"cpu_threads\":%d,"
                "\"prefill_seconds\":%.6f,\"setup_seconds\":%.6f,\"score_seconds\":%.6f}\n",
                prefix, steps, vocab, options.flash_attn ? "true" : "false", options.cpu_threads,
                seconds(start, prefilled), seconds(prefilled, prepared), seconds(prepared, scored));
}

}  // namespace glm_evaluation
