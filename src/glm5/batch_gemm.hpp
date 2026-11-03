#pragma once

#include "gpu_support.hpp"
#include <memory>

namespace strata::glm5 {

struct BatchStats {
    uint64_t mmq_calls = 0, f16_calls = 0, f32_calls = 0, max_columns = 0;
};

class BatchGemm {
public:
    BatchGemm(const Geometry& g, int columns, void* stream, MemoryBudget* budget);
    ~BatchGemm();
    void native(int type, const void* weights, const float* x, float* y, int64_t columns, int64_t rows, int64_t width);
    void f32(const float* weights, const float* x, float* y, int64_t columns, int64_t rows, int64_t width);
    const BatchStats& stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
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
