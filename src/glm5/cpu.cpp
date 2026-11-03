#include "cpu.hpp"

#include <cstring>
#include <mutex>
#include <stdexcept>

namespace strata::glm5 {

bool cpu_parallel_available() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

void init_cpu() {
    // CPU dots and dequantizers need ggml's tables even though this engine does not build a ggml graph.
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

const ggml_type_traits_cpu& cpu_traits(uint32_t type) {
    if (type >= GGML_TYPE_COUNT) throw std::runtime_error("glm5: unsupported CPU GGUF type " + std::to_string(type));
    const auto* traits = ggml_get_type_traits_cpu(static_cast<ggml_type>(type));
    if (!traits || !traits->vec_dot)
        throw std::runtime_error("glm5: no CPU dot product for GGUF type " + std::to_string(type));
    return *traits;
}

std::vector<uint8_t> quantize(const float* x, int64_t width, uint32_t type) {
    const auto act = cpu_traits(type).vec_dot_type;
    const auto* tr = ggml_get_type_traits_cpu(act);
    if (!tr || !tr->from_float || width % ggml_blck_size(act))
        throw std::runtime_error("glm5: unsupported expert activation quantization");
    std::vector<uint8_t> out(ggml_row_size(act, width));
    tr->from_float(x, out.data(), width);
    return out;
}

void dequantize(const TensorInfo& t, const uint8_t* src, float* dst, int64_t count) {
    if (t.type == GGML_TYPE_F32) {
        std::memcpy(dst, src, count * sizeof(float));
        return;
    }
    if (t.type >= GGML_TYPE_COUNT)
        throw std::runtime_error(t.name + ": unsupported GGUF type");
    const auto* tr = ggml_get_type_traits(static_cast<ggml_type>(t.type));
    if (!tr || !tr->to_float || count % ggml_blck_size(static_cast<ggml_type>(t.type)))
        throw std::runtime_error(t.name + ": cannot dequantize complete rows");
    tr->to_float(src, dst, count);
}

}  // namespace strata::glm5
