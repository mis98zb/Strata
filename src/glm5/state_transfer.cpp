#include "state_transfer.hpp"
#include "gpu_support.hpp"
#include "strata/glm5/glm_dsa.hpp"

namespace strata::glm5 {

LayerImage capture_layer(const Geometry& g, const GlmLayerState& state, int64_t position, bool kda, void* opaque) {
    if (kernels::glm_dsa_enabled()) throw std::runtime_error("glm5 handoff: DSA state transfer is not implemented");
    const auto stream = static_cast<cudaStream_t>(opaque);
    LayerImage image;
    if (kda) {
        image.recurrent.resize(g.kda_state_floats());
        check(cudaMemcpyAsync(image.recurrent.data(), state.kda_state, image.recurrent.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream), "glm5 capture KDA state");
    } else {
        if (position > state.max_cells) throw std::runtime_error("glm5 handoff: cache position exceeds allocation");
        image.kv.resize(static_cast<size_t>(position) * g.kv_lora_rank);
        if (!image.kv.empty())
            check(cudaMemcpyAsync(image.kv.data(), state.mla_cache, image.kv.size() * sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost, stream), "glm5 capture MLA cache");
    }
    check(cudaStreamSynchronize(stream), "glm5 state capture sync");
    return image;
}

void restore_layer(const Geometry&, const LayerImage& image, const GlmLayerState& state, void* opaque) {
    const auto stream = static_cast<cudaStream_t>(opaque);
    if (!image.recurrent.empty())
        check(cudaMemcpyAsync(state.kda_state, image.recurrent.data(), image.recurrent.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream), "glm5 restore KDA state");
    if (!image.kv.empty())
        check(cudaMemcpyAsync(state.mla_cache, image.kv.data(), image.kv.size() * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, stream), "glm5 restore MLA cache");
}

}  // namespace strata::glm5
