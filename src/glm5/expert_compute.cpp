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

}  // namespace strata::glm5
