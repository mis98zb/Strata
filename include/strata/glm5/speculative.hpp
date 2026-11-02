#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

namespace strata::glm5 {

inline int32_t greedy(const std::vector<float>& logits) {
    if (logits.empty() || !std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("glm5: empty or non-finite logits");
    return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

struct Verification {
    size_t committed = 0;
    bool stopped = false;
};

inline Verification verify_greedy(const std::vector<int32_t>& inputs,
                                   const std::vector<std::vector<float>>& logits,
                                   const std::set<int32_t>& stop_ids) {
    if (inputs.empty() || inputs.size() != logits.size())
        throw std::invalid_argument("glm5 verification: one target logit row is required per input token");
    Verification result;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i && inputs[i] != greedy(logits[i - 1])) break;
        if (stop_ids.count(inputs[i])) { result.stopped = true; break; }
        ++result.committed;
    }
    return result;
}

}  // namespace strata::glm5
