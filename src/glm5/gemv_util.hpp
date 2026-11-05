#pragma once

#include "strata/core/weights.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace strata::glm5::gemv {
void set_skip_unused_quant(bool enabled);
bool skip_unused_quant();

using core::WeightRef;
constexpr int Q8K_BYTES_PER_BLOCK = 292;
constexpr int Q8K_ELEMS_PER_BLOCK = 256;
constexpr float RMS_EPS = 1e-6f;
uint64_t q8k_bytes(int64_t n);
bool sform_of(const WeightRef&, kernels::SForm&, const std::string& name, std::string& err);
struct Planes {
    const uint8_t* codes = nullptr;
    const float* scales = nullptr;
    const float* offset = nullptr;
};
bool plane_ptrs(const WeightRef&, const std::string& name, Planes&, std::string& err);
bool gemv_quantized(const WeightRef& w, const Planes&, const kernels::SForm&, const uint8_t*,
                    const uint8_t*, float* y, int64_t n_in, int64_t n_out, const std::string& name,
                    void* stream, std::string& err, const float* x = nullptr, bool ready = false, int ncols = 1);
bool project_rows(const WeightRef& w, const std::string& name, const float* x, const uint8_t* x80,
                  const uint8_t* xk, float* y, int64_t n_in, int64_t row0, int64_t rows, void* stream,
                  std::string& err, int64_t ncols = 1);
void project_bf16(const float* x, const uint16_t* x_bf16, const uint16_t* weights, float* out,
                  int64_t n_in, int64_t n_out, bool split, void* stream);
void project_f32(const float* x, const float* weights, float* out, int64_t n_in, int64_t n_out,
                  int64_t columns, void* stream);
extern bool native_bf16_projections;
extern bool native_flash_attn_short;

struct ProjStats {
    long long calls = 0, native_calls = 0, multi_calls = 0, single_calls = 0, multi_cols = 0, cols = 0;
    unsigned long long bytes = 0, multi_bytes = 0;
    struct ByName {
        long long calls = 0, cols = 0;
        unsigned long long bytes = 0;
        bool native = false;
    };
    std::unordered_map<std::string, ByName> by_name;
};
ProjStats& proj_stats();
void proj_stats_reset();

}  // namespace strata::glm5::gemv
