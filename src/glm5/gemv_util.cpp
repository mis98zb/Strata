#include "gemv_util.hpp"
#include "batch_gemm.hpp"

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/glm5/glm.hpp"

#include <exception>
#include <limits>

namespace strata::glm5::gemv {
namespace {
thread_local bool skip_images = false;
}

void set_skip_unused_quant(bool enabled) { skip_images = enabled; }
bool skip_unused_quant() { return skip_images; }

bool native_bf16_projections = true;
bool native_flash_attn_short = false;

uint64_t q8k_bytes(int64_t n) { return static_cast<uint64_t>(n / 256) * 292; }

ProjStats& proj_stats() {
    static thread_local ProjStats stats;
    return stats;
}
void proj_stats_reset() { proj_stats() = {}; }

bool sform_of(const WeightRef&, kernels::SForm&, const std::string& name, std::string& err) {
    err = name + ": strata-glm5 requires native GGUF projections, not a canonical Qwen pack";
    return false;
}
bool plane_ptrs(const WeightRef&, const std::string& name, Planes&, std::string& err) {
    err = name + ": strata-glm5 does not load canonical weight planes";
    return false;
}

void project_bf16(const float* x, const uint16_t*, const uint16_t* weights, float* out,
                  int64_t n_in, int64_t n_out, bool, void* stream) {
    kernels::bf16_gemv_fp32_mmvf(x, weights, out, n_in, n_out, stream);
}

void project_f32(const float* x, const float* weights, float* out, int64_t n_in, int64_t n_out,
                  int64_t columns, void* stream) {
    if (columns > 8) {
        auto* batch = active_batch_gemm();
        if (!batch) throw std::runtime_error("glm5: large F32 projection requires full-layer GEMM context");
        batch->f32(weights, x, out, columns, n_out, n_in);
        return;
    }
    for (int64_t c = 0; c < columns; ++c)
        kernels::glm_f32_gemv(x + c * n_in, weights, out + c * n_out, n_in, n_out, stream);
}

bool gemv_quantized(const WeightRef& w, const Planes&, const kernels::SForm&, const uint8_t*,
                    const uint8_t*, float* y, int64_t n_in, int64_t n_out, const std::string& name,
                    void* stream, std::string& err, const float* x, bool ready, int ncols) {
    if (!w.native_data || (ncols <= 8 && !w.native_q8_1) || !x || !y || !stream ||
        n_in != w.ne0 || n_out != w.ne1 || n_in <= 0 || n_in > INT32_MAX ||
        n_out <= 0 || n_out > INT32_MAX || ncols < 1 || ncols > 4096) {
        err = name + ": invalid native projection shape, input or scratch";
        return false;
    }
    try {
        if (ncols > 8) {
            auto* batch = active_batch_gemm();
            if (!batch) throw std::runtime_error("large native projection requires full-layer GEMM context");
            batch->native(w.native_type, w.native_data, x, y, ncols, n_out, n_in);
        } else {
            if (!ready || ncols != 1)
                kernels::native_quantize_q8_1(x, w.native_q8_1, static_cast<int>(n_in), ncols, stream);
            kernels::native_mmvq(w.native_type, w.native_data, w.native_q8_1, y,
                                static_cast<int>(n_in), static_cast<int>(n_out), ncols, stream);
        }
        auto& stats = proj_stats();
        const auto bytes = kernels::native_mmvq_weight_bytes(w.native_type, n_in, n_out);
        ++stats.calls;
        stats.cols += ncols;
        ++stats.native_calls;
        stats.bytes += bytes;
        auto& by = stats.by_name[name];
        ++by.calls;
        by.cols += ncols;
        by.bytes += bytes;
        by.native = true;
        return true;
    } catch (const std::exception& e) {
        err = name + ": " + e.what();
        return false;
    }
}

bool project_rows(const WeightRef& w, const std::string& name, const float* x, const uint8_t* x80,
                  const uint8_t* xk, float* y, int64_t n_in, int64_t row0, int64_t rows, void* stream,
                  std::string& err, int64_t ncols) {
    if (row0 < 0 || rows <= 0 || row0 > w.ne1 || rows > w.ne1 - row0 || n_in != w.ne0 ||
        ncols < 1 || ncols > 4096 || !x || !y || !stream) {
        err = name + ": invalid native weight row band";
        return false;
    }
    try {
        if (!w.native_data) {
            if (!w.data || w.bytes != static_cast<uint64_t>(w.elements) * sizeof(float)) {
                err = name + ": row band requires native GGUF or F32 weights";
                return false;
            }
            const auto* weights = static_cast<const float*>(w.data) + row0 * n_in;
            project_f32(x, weights, y, n_in, rows, ncols, stream);
            return true;
        }
        WeightRef band = w;
        const auto stride = kernels::native_mmvq_weight_bytes(w.native_type, n_in, 1);
        band.native_data = static_cast<const uint8_t*>(w.native_data) + row0 * stride;
        band.ne1 = rows;
        return gemv_quantized(band, {}, {}, x80, xk, y, n_in, rows, name, stream, err,
                              x, false, static_cast<int>(ncols));
    } catch (const std::exception& e) {
        err = name + ": " + e.what();
        return false;
    }
}

}  // namespace strata::glm5::gemv
