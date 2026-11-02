#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace strata::glm5 {

class ExpertCachePolicy {
public:
    ExpertCachePolicy(std::vector<int> quotas, int experts);
    static std::vector<int> quotas(const std::vector<uint64_t>& expert_bytes, uint64_t budget, int experts);
    int resident(int layer, int expert) const;
    int candidate(int layer, int expert, bool force, bool allow_swap) const;
    void bind(int layer, int slot, int expert);
    void observe(int layer, const int32_t* ids, int count);
    void clear();
    bool consistent() const;
    int slots() const { return static_cast<int>(owners_.size()); }
    int layer_begin(int layer) const;
    int layer_end(int layer) const;
    int owner(int slot) const;
private:
    size_t index(int layer, int expert) const;
    int experts_ = 0;
    std::vector<int> begin_, end_, resident_, owners_;
    std::vector<uint32_t> counts_, epochs_;
};

}  // namespace strata::glm5
