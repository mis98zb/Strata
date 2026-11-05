#pragma once

#include "gpu_support.hpp"
#include "strata/glm5/gpu_experts.hpp"

#include <memory>

namespace strata::glm5 {

struct ExpertWeights {
    const void* gate = nullptr;
    const void* up = nullptr;
    const void* down = nullptr;
    int gate_type = -1, up_type = -1, down_type = -1;
};

class ExpertComputer {
public:
    ExpertComputer(const Geometry& g, MemoryBudget* budget = nullptr, int columns = 8);
    void run(const ExpertWeights& weights, const float* input, const std::vector<ExpertMiss>& entries,
             float* output, void* stream, const int32_t* device_routes = nullptr);
private:
    void* alloc(size_t bytes);
    const Geometry& geometry_;
    MemoryBudget* budget_;
    int capacity_;
    std::vector<std::unique_ptr<Buffer>> storage_;
    float *x_, *gate_, *up_, *result_;
    void *xq_, *hq_;
};

}  // namespace strata::glm5
