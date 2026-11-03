#pragma once

#include "strata/glm5/model.hpp"
#include "strata/glm5/model_file.hpp"
#include "strata/glm5/state.hpp"

namespace strata::glm5 {

struct PrefillResult {
    HostState state;
    PrefillStats stats;
};

uint64_t full_layer_host_bytes(const ModelFile& file, const Options& options, size_t count,
                               int64_t previous_position, uint64_t previous_state_bytes);

PrefillResult full_layer_prefill(const ModelFile& file, const Options& options,
                                 const std::vector<int32_t>& tokens, const HostState& initial);

}  // namespace strata::glm5
