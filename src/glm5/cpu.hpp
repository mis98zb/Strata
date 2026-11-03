#pragma once

#include "strata/glm5/model_file.hpp"
#include "ggml-cpu.h"

namespace strata::glm5 {
void init_cpu();
bool cpu_parallel_available();
const ggml_type_traits_cpu& cpu_traits(uint32_t type);
std::vector<uint8_t> quantize(const float* x, int64_t width, uint32_t type);
void dequantize(const TensorInfo& t, const uint8_t* src, float* dst, int64_t count);
}  // namespace strata::glm5
