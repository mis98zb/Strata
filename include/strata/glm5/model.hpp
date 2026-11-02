#pragma once

#include "strata/glm5/geometry.hpp"
#include "strata/glm5/gpu_experts.hpp"

#include <memory>
#include <set>
#include <vector>

namespace strata::glm5 {

struct Options {
    int gpu = 0;
    int64_t max_context = 4096;
    int prefill = 1;
    int mtp = 0;
    int64_t gpu_experts_mib = -1;
    uint64_t gpu_reserve_mib = 512;
    bool gpu_check = false;
};

struct MtpStats {
    uint64_t offered = 0, accepted = 0, verify_batches = 0, replays = 0;
};

class Model {
public:
    Model(const std::string& path, int gpu, int64_t max_context);
    Model(const std::string& path, const Options& options);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    const Geometry& geometry() const;
    const std::set<int32_t>& stop_ids() const;
    const std::vector<float>& token(int32_t id);
    const std::vector<float>& prefill(const std::vector<int32_t>& ids);
    std::vector<int32_t> generate(int count);
    ExpertCacheStats cache_stats() const;
    MtpStats mtp_stats() const;
    void reset();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::glm5
