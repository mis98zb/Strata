#include "expert_compute.hpp"
#include "gemv_util.hpp"
#include "batch_gemm.hpp"
#include "strata/glm5/glm.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>

namespace strata::glm5 {

void* ExpertComputer::alloc(size_t bytes) {
    auto buffer = std::make_unique<Buffer>(bytes, budget_);
    auto* p = buffer->get();
    storage_.push_back(std::move(buffer));
    return p;
}

ExpertComputer::ExpertComputer(const Geometry& g, MemoryBudget* budget, int columns)
    : geometry_(g), budget_(budget), capacity_(columns) {
    if (columns < 1 || columns > 4096) throw std::invalid_argument("glm5 expert GEMM: invalid batch");
    x_ = static_cast<float*>(alloc(columns * g.n_embd * sizeof(float)));
    result_ = static_cast<float*>(alloc(columns * g.n_embd * sizeof(float)));
    gate_ = static_cast<float*>(alloc(columns * g.n_ff * sizeof(float)));
    up_ = static_cast<float*>(alloc(columns * g.n_ff * sizeof(float)));
    xq_ = alloc(kernels::native_q8_1_bytes(g.n_embd, 8));
    hq_ = alloc(kernels::native_q8_1_bytes(g.n_ff, 8));
}

const void* ExpertComputer::prepare_single_input(const float* input, void* opaque) {
    if (!input || !opaque) throw std::invalid_argument("glm5 single expert: missing input or stream");
    kernels::native_quantize_q8_1(input, xq_, geometry_.n_embd, 1, static_cast<cudaStream_t>(opaque));
    return xq_;
}

void ExpertComputer::run_single(const ExpertWeights& w, const void* prepared_input, float* output, void* opaque) {
    if (!w.gate || !w.up || !w.down || !prepared_input || !output || !opaque ||
        !kernels::native_mmvq_supported(w.gate_type) || !kernels::native_mmvq_supported(w.up_type) ||
        !kernels::native_mmvq_supported(w.down_type))
        throw std::invalid_argument("glm5 single expert: invalid weights, activation or output");
    const auto& g = geometry_;
    const auto stream = static_cast<cudaStream_t>(opaque);
    kernels::native_mmvq(w.gate_type, w.gate, prepared_input, gate_, g.n_embd, g.n_ff, 1, stream);
    kernels::native_mmvq(w.up_type, w.up, prepared_input, up_, g.n_embd, g.n_ff, 1, stream);
    kernels::glm_swiglu(gate_, up_, g.swiglu_limit_or_off(), g.n_ff, stream);
    kernels::native_quantize_q8_1(gate_, hq_, g.n_ff, 1, stream);
    kernels::native_mmvq(w.down_type, w.down, hq_, output, g.n_ff, g.n_embd, 1, stream);
}

void ExpertComputer::run(const ExpertWeights& w, const float* input, std::span<const ExpertMiss> entries,
                         float* output, void* opaque, const int32_t* device_routes) {
    const auto& g = geometry_;
    const auto stream = static_cast<cudaStream_t>(opaque);
    if (!w.gate || !w.up || !w.down || !stream) throw std::runtime_error("glm5: invalid resident expert");
    for (size_t first = 0; first < entries.size(); first += capacity_) {
        const auto n = static_cast<int>(std::min<size_t>(capacity_, entries.size() - first));
        if (device_routes)
            kernels::glm_expert_gather(input, device_routes + 2 * first, x_, n, g.n_embd, stream);
        else for (int i = 0; i < n; ++i)
            check(cudaMemcpyAsync(x_ + i * g.n_embd, input + entries[first + i].token * g.n_embd,
                                   g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, stream), "glm5 expert gather");
        const auto project = [&](const void* weights, int type, const float* x, float* y, int64_t k, int64_t rows) {
            core::WeightRef ref;
            ref.native_data = weights; ref.native_type = type; ref.ne0 = k; ref.ne1 = rows;
            std::string err;
            if (!gemv::gemv_quantized(ref, {}, {}, nullptr, nullptr, y, k, rows, "routed expert",
                                      stream, err, x, false, n)) throw std::runtime_error(err);
        };
        if (n > 8) {
            QuantInputScope immutable(gemv::skip_unused_quant() && w.gate_type == w.up_type ?
                active_batch_gemm() : nullptr, x_, w.gate_type, n, g.n_embd);
            project(w.gate, w.gate_type, x_, gate_, g.n_embd, g.n_ff);
            project(w.up, w.up_type, x_, up_, g.n_embd, g.n_ff);
        } else {
            kernels::native_quantize_q8_1(x_, xq_, g.n_embd, n, stream);
            kernels::native_mmvq(w.gate_type, w.gate, xq_, gate_, g.n_embd, g.n_ff, n, stream);
            kernels::native_mmvq(w.up_type, w.up, xq_, up_, g.n_embd, g.n_ff, n, stream);
        }
        kernels::glm_swiglu(gate_, up_, g.swiglu_limit_or_off(), n * g.n_ff, stream);
        if (n > 8) project(w.down, w.down_type, gate_, result_, g.n_ff, g.n_embd);
        else {
            kernels::native_quantize_q8_1(gate_, hq_, g.n_ff, n, stream);
            kernels::native_mmvq(w.down_type, w.down, hq_, result_, g.n_ff, g.n_embd, n, stream);
        }
        if (device_routes)
            kernels::glm_expert_scatter(result_, device_routes + 2 * first, output, n, g.n_embd, stream);
        else for (int i = 0; i < n; ++i)
            check(cudaMemcpyAsync(output + entries[first + i].row * g.n_embd, result_ + i * g.n_embd,
                                   g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, stream), "glm5 expert scatter");
    }
}

void ExpertComputer::run_group(std::span<const ExpertWeights> weights, std::span<const int32_t> counts,
                               const float* input, float* output, void* opaque, const int32_t* routes) {
    auto* batch = active_batch_gemm();
    const auto stream = static_cast<cudaStream_t>(opaque);
    if (!batch || !stream || !input || !output || !routes || weights.empty() || weights.size() != counts.size())
        throw std::invalid_argument("glm5 expert group: invalid workspace, weights or routes");
    int total = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        const auto& w = weights[i];
        if (counts[i] <= 8 || counts[i] > capacity_ - total || !w.gate || !w.up || !w.down ||
            !kernels::native_mmvq_supported(w.gate_type) || !kernels::native_mmvq_supported(w.up_type) ||
            !kernels::native_mmvq_supported(w.down_type))
            throw std::invalid_argument("glm5 expert group: unsupported matrix or group exceeds capacity");
        total += counts[i];
    }
    const auto& g = geometry_;
    kernels::glm_expert_gather(input, routes, x_, total, g.n_embd, stream);
    const auto project = [&](const void* weight, int type, const float* x, float* y,
                             int columns, int64_t width, int64_t rows) {
        core::WeightRef ref;
        ref.native_data = weight; ref.native_type = type; ref.ne0 = width; ref.ne1 = rows;
        std::string error;
        if (!gemv::gemv_quantized(ref, {}, {}, nullptr, nullptr, y, width, rows, "routed expert",
                                  stream, error, x, false, columns))
            throw std::runtime_error(error);
    };
    int first = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        const auto& w = weights[i];
        const auto* x = x_ + first * g.n_embd;
        QuantInputScope immutable(gemv::skip_unused_quant() && w.gate_type == w.up_type ? batch : nullptr,
                                   x, w.gate_type, counts[i], g.n_embd);
        project(w.gate, w.gate_type, x, gate_ + first * g.n_ff, counts[i], g.n_embd, g.n_ff);
        project(w.up, w.up_type, x, up_ + first * g.n_ff, counts[i], g.n_embd, g.n_ff);
        first += counts[i];
    }
    kernels::glm_swiglu(gate_, up_, g.swiglu_limit_or_off(), total * g.n_ff, stream);
    first = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        project(weights[i].down, weights[i].down_type, gate_ + first * g.n_ff,
                result_ + first * g.n_embd, counts[i], g.n_ff, g.n_embd);
        first += counts[i];
    }
    kernels::glm_expert_scatter(result_, routes, output, total, g.n_embd, stream);
}

}  // namespace strata::glm5
