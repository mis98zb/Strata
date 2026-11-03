#pragma once

#include "strata/glm5/layer.hpp"
#include "strata/glm5/state.hpp"

namespace strata::glm5 {
LayerImage capture_layer(const Geometry& g, const GlmLayerState& state, int64_t position, bool kda, void* stream);
void restore_layer(const Geometry& g, const LayerImage& image, const GlmLayerState& state, void* stream);
}  // namespace strata::glm5
