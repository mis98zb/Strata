#pragma once

#include "gpu_support.hpp"
#include "strata/glm5/gpu_experts.hpp"

#include <memory>
#include <span>

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
    void run(const ExpertWeights& weights, const float* input, std::span<const ExpertMiss> entries,
             float* output, void* stream, const int32_t* device_routes = nullptr);
    void run_group(std::span<const ExpertWeights> weights, std::span<const int32_t> counts,
                   const float* input, float* output, void* stream, const int32_t* device_routes);
    // The returned image is valid on this stream until this computer next prepares or runs another input.
    const void* prepare_single_input(const float* input, void* stream);
    void run_single(const ExpertWeights& weights, const void* prepared_input, float* output, void* stream);
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
