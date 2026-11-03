#pragma once

#include "strata/glm5/geometry.hpp"

#include <stdexcept>
#include <vector>

namespace strata::glm5 {

inline uint64_t checked_add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("glm5 memory size overflows");
    return a + b;
}
inline uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error("glm5 memory size overflows");
    return a * b;
}

class MemoryBudget {
public:
    explicit MemoryBudget(uint64_t limit) : limit_(limit) {}
    void acquire(uint64_t bytes) {
        if (bytes > limit_ - used_) throw std::runtime_error("glm5 full-layer GPU budget exceeded");
        used_ += bytes;
        if (used_ > peak_) peak_ = used_;
    }
    void release(uint64_t bytes) { used_ -= bytes; }
    uint64_t peak() const { return peak_; }
    uint64_t used() const { return used_; }
private:
    uint64_t limit_, used_ = 0, peak_ = 0;
};

struct LayerImage {
    std::vector<float> recurrent;
    std::vector<uint16_t> kv;
    uint64_t bytes() const { return recurrent.size() * sizeof(float) + kv.size() * sizeof(uint16_t); }
};

struct HostState {
    int64_t position = 0;
    std::vector<LayerImage> layers;
    LayerImage mtp;
    std::vector<float> carry_hidden, logits;
    uint64_t bytes() const;
    void validate(const Geometry& g, int64_t context, bool use_mtp) const;
};

uint64_t host_state_bytes(const Geometry& g, int64_t position, bool use_mtp);

}  // namespace strata::glm5
