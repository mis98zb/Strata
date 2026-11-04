#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace strata::glm5 {

class ModelFile;
struct ExpertMiss {
    int token = 0;
    int row = 0;
    int expert = 0;
};
struct ExpertCacheStats {
    uint64_t hits = 0, misses = 0, swaps = 0, streamed = 0, bytes = 0;
    uint64_t slots = 0, warmed = 0, upload_bytes = 0, pinned_bytes = 0;
};

class GpuExperts {
public:
    GpuExperts(const ModelFile& file, int layers, int64_t budget_mib, uint64_t reserve_mib,
               const std::vector<uint32_t>& heat = {}, bool heat_quotas = false, int stage_slots = 0);
    ~GpuExperts();
    const std::vector<ExpertMiss>& run(int layer, const float* input, const int32_t* ids, int tokens,
                                      float* output, bool stream_all, void* stream);
    void admit(int layer, const int32_t* ids, const std::vector<ExpertMiss>& misses, void* stream);
    const ExpertCacheStats& stats() const;
    const std::vector<uint32_t>& heat() const;
    void warm(const std::vector<uint32_t>& heat, void* stream);
    void reset();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::glm5
