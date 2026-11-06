#pragma once

#include <array>

#include "strata/glm5/geometry.hpp"
#include "strata/glm5/gpu_experts.hpp"

#include <memory>
#include <set>
#include <vector>

namespace strata::glm5 {

inline constexpr int64_t max_context_limit = 262144;

enum class DsaScore { Automatic, Scalar, Parallel, Tiled };
enum class CpuExpertMode { Automatic, Reference, Rows };
enum class GpuExpertMode { Automatic, Reference, Single };
enum class MlaAttentionLayout { Automatic, TokenMajor, HeadMajor };
enum class FlashKvLoad { Automatic, Direct, Prefetch };

struct Options {
    bool reference = false;
    int gpu = 0;
    int64_t max_context = 4096;
    int prefill = 0;
    int mtp = 0;
    int cpu_threads = 1;
    bool dsa = false;
    bool flash_attn = false;
    DsaScore dsa_score = DsaScore::Automatic;
    bool dsa_parallel_topk = false;
    int dsa_tail = -1;
    int64_t gpu_experts_mib = -1;
    uint64_t gpu_reserve_mib = 512;
    bool gpu_check = false;
    bool gpu_warm = false;
    bool gpu_heat_quotas = false;
    int gpu_stage_slots = 0;
    bool full_layer_prefill = false;
    uint64_t prefill_gpu_mib = 0;
    uint64_t prefill_host_mib = 4096;
    uint64_t prefill_reserve_mib = 512;
    bool prefill_auto = false;
    int prefill_group_layers = 0;
    int prefill_prefetch_layers = -1;
    bool prefill_pipeline = false;
    bool prefill_reference = false;
    bool prefill_no_pipeline = false;
    bool prefill_expert_rows = false;
    bool skip_unused_quant = false;
    bool kda_register_state = false;
    uint64_t prefill_transfer_mib = 512;
    uint64_t prefill_weight_stage_mib = 0;
    bool probe = false;
    CpuExpertMode cpu_expert_mode = CpuExpertMode::Automatic;
    GpuExpertMode gpu_expert_mode = GpuExpertMode::Automatic;
    MlaAttentionLayout mla_attention_layout = MlaAttentionLayout::Automatic;
    FlashKvLoad flash_kv_load = FlashKvLoad::Automatic;
};

Options resolve_options(Options options);
Options select_compute_options(Options options, const Geometry& geometry, int cuda_arch);
Options resolve_execution_options(const Options& options, const Geometry& geometry);

struct PrefillGroupStats {
    int first_layer = 0, layers = 0, prefetch_layers = 0;
};

enum class ProbePhase {
    WeightWait, ActivationH2D, ActivationD2H, LayerPre, RouteD2H, GpuExperts,
    CpuExperts, CpuResultH2D, CacheAdmit, LayerPost, Head, EmbeddingH2D,
    StateTransfer, Synchronize, KdaMixer, MlaMixer, Ffn, Router,
    MlaQuery, MlaAbsorb, MlaKv, MlaIndexProject, MlaIndexPool, MlaIndexScore,
    MlaIndexSelect, MlaAttention, MlaDeabsorb, MlaOutput, MlaLayout, Count
};
struct ProbeSample {
    double host_ms = 0, process_cpu_ms = 0, stream_ms = 0;
    uint64_t host_calls = 0, stream_calls = 0, bytes = 0;
};
struct ProbeStats {
    std::array<ProbeSample, static_cast<size_t>(ProbePhase::Count)> phases{};
    uint64_t rows = 0, cpu_experts = 0, forced_waits = 0, event_pairs = 0;
    uint64_t cpu_result_copies = 0;
};

struct PrefillStats {
    uint64_t gpu_peak_bytes = 0, host_tensor_bytes = 0, layer_loads = 0, experts_loaded = 0;
    uint64_t max_matrix_columns = 0, mmq_calls = 0, f16_gemm_calls = 0, f32_gemm_calls = 0;
    uint64_t selected_batch = 0, groups = 0, max_group_layers = 0, planning_attempts = 0;
    uint64_t prefetched_layers = 0, weight_upload_bytes = 0, activation_upload_bytes = 0;
    uint64_t activation_download_bytes = 0, transfer_peak_bytes = 0;
    uint64_t weight_staging_peak_bytes = 0, weight_upload_calls = 0;
    uint64_t native_quant_calls = 0, reused_quant_calls = 0;
    uint64_t bounds_launches = 0, reused_bounds = 0;
    double planning_ms = 0;
    bool fused_head = false;
    std::vector<PrefillGroupStats> plan;
    ProbeStats probe;
};
struct MtpStats {
    uint64_t offered = 0, accepted = 0, verify_batches = 0, replays = 0;
    uint64_t target_batches = 0, replayed_layers = 0;
    uint64_t dsa_replayed_layers = 0;
};

class Model {
public:
    Model(const std::string& path, int gpu, int64_t max_context);
    Model(const std::string& path, const Options& options);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    const Geometry& geometry() const;
    const Options& execution_options() const;
    const std::set<int32_t>& stop_ids() const;
    const std::vector<float>& token(int32_t id);
    const std::vector<float>& prefill(const std::vector<int32_t>& ids);
    std::vector<int32_t> generate(int count);
    ExpertCacheStats cache_stats() const;
    MtpStats mtp_stats() const;
    PrefillStats prefill_stats() const;
    ProbeStats decode_probe_stats() const;
    bool decode_loaded() const;
    void prepare_decode();
    void reset();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::glm5
