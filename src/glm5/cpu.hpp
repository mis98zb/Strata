#pragma once

#include "strata/glm5/model_file.hpp"
#include "ggml-cpu.h"
#include "strata/glm5/gpu_experts.hpp"

#include <memory>
#include <span>

namespace strata::glm5 {
void init_cpu();
bool cpu_parallel_available();
std::string cpu_isa_summary();
bool cpu_row_pool_default(const Geometry& geometry, int threads);
const ggml_type_traits_cpu& cpu_traits(uint32_t type);
std::vector<uint8_t> quantize(const float* x, int64_t width, uint32_t type);
void dequantize(const TensorInfo& t, const uint8_t* src, float* dst, int64_t count);

// One synchronous caller; scratch and borrowed views are reused only after all row tasks finish.
class CpuRowPool {
public:
    CpuRowPool(const ModelFile& file, int layers, int threads, bool atomic_completion = false);
    ~CpuRowPool();
    void run(int layer, int tokens, std::span<const float> input, std::span<const ExpertMiss> entries,
             std::span<float> output);
    uint64_t expert_bytes(int layer) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::glm5
