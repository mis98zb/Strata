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
    bool full_layer_prefill = false;
    uint64_t prefill_gpu_mib = 0;
    uint64_t prefill_host_mib = 4096;
    uint64_t prefill_reserve_mib = 512;
};

struct PrefillStats {
    uint64_t gpu_peak_bytes = 0, host_tensor_bytes = 0, layer_loads = 0, experts_loaded = 0;
    uint64_t max_matrix_columns = 0, mmq_calls = 0, f16_gemm_calls = 0, f32_gemm_calls = 0;
};
struct MtpStats {
    uint64_t offered = 0, accepted = 0, verify_batches = 0, replays = 0;
    uint64_t target_batches = 0, replayed_layers = 0;
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
    PrefillStats prefill_stats() const;
    bool decode_loaded() const;
    void prepare_decode();
    void reset();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::glm5
