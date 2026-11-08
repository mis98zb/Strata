#pragma once

#include "gpu_support.hpp"
#include <memory>

namespace strata::glm5 {

struct BatchStats {
    uint64_t mmq_calls = 0, f16_calls = 0, f32_calls = 0, max_columns = 0;
    uint64_t native_quant_calls = 0, reused_quant_calls = 0;
    uint64_t bounds_launches = 0, reused_bounds = 0;
};

class BatchGemm {
public:
    BatchGemm(const Geometry& g, int columns, void* stream, MemoryBudget* budget, bool cache_group_bounds = false);
    ~BatchGemm();
    void native(int type, const void* weights, const float* x, float* y, int64_t columns, int64_t rows, int64_t width);
    void f32(const float* weights, const float* x, float* y, int64_t columns, int64_t rows, int64_t width);
    const BatchStats& stats() const;
private:
    friend class QuantInputScope;
    void begin_input(const float* input, int type, int64_t columns, int64_t width, int64_t heads);
    void end_input();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The caller keeps input immutable until scope exit; no cache survives that lifetime.
class QuantInputScope {
public:
    QuantInputScope(BatchGemm* backend, const float* input, int type, int64_t columns, int64_t width, int64_t heads = 1)
        : backend_(backend) { if (backend_) backend_->begin_input(input, type, columns, width, heads); }
    ~QuantInputScope() { if (backend_) backend_->end_input(); }
    QuantInputScope(const QuantInputScope&) = delete;
    QuantInputScope& operator=(const QuantInputScope&) = delete;
private:
    BatchGemm* backend_;
};

BatchGemm* active_batch_gemm();
class BatchScope {
public:
    explicit BatchScope(BatchGemm* backend);
    ~BatchScope();
    BatchScope(const BatchScope&) = delete;
    BatchScope& operator=(const BatchScope&) = delete;
private:
    BatchGemm* previous_;
};

}  // namespace strata::glm5
