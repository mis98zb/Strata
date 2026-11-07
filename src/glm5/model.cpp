#include "strata/glm5/model.hpp"
#include "cpu.hpp"
#include "decode.hpp"
#include "prefill.hpp"
#include "gpu_support.hpp"

namespace strata::glm5 {

Options resolve_options(Options o) {
    if (o.reference && o.full_layer_prefill) o.prefill_reference = true;
    const bool automatic = o.full_layer_prefill && !o.prefill_reference;
    if (!o.prefill) o.prefill = automatic ? 2048 : 1;
    if (o.prefill_prefetch_layers == -1) o.prefill_prefetch_layers = automatic ? 1 : 0;
    if (automatic) {
        o.prefill_auto = true;
        o.prefill_pipeline = true;
    }
    if (o.prefill_no_pipeline) o.prefill_pipeline = false;
    return o;
}

Options select_compute_options(Options o, const Geometry& g, int cuda_arch) {
    if (o.gpu_experts_mib == default_gpu_expert_budget)
        o.gpu_experts_mib = !o.reference && !o.prefill_reference && cuda_arch == 86 &&
            g.n_embd == 4096 && g.n_ff == 2048 && g.n_expert == 288 && g.n_expert_used == 8 ? 0 : -1;
    if (o.reference || o.prefill_reference || cuda_arch != 86) return o;
    if (o.gpu_expert_mode == GpuExpertMode::Automatic) o.gpu_expert_mode = GpuExpertMode::Single;
    if (o.flash_attn && o.mla_attention_layout == MlaAttentionLayout::Automatic &&
        g.kv_lora_rank == 512 && g.mla_head_dim == 256 && g.n_head > 0)
        o.mla_attention_layout = MlaAttentionLayout::HeadMajor;
    if (o.flash_attn && o.flash_kv_load == FlashKvLoad::Automatic &&
        g.kv_lora_rank == 512 && g.mla_head_dim == 256 && g.n_head > 0)
        o.flash_kv_load = FlashKvLoad::Prefetch;
    o.skip_unused_quant = true;
    if (g.kda_head_dim == 128) o.kda_register_state = true;
    if (o.full_layer_prefill) {
        o.prefill_expert_rows = true;
        o.prefill_transfer_cost = true;
    }
    if (o.dsa) {
        if (o.dsa_score == DsaScore::Automatic && g.idx_q_heads == 32 && g.idx_key_dim == 128)
            o.dsa_score = DsaScore::Tiled;
        if (o.dsa_score == DsaScore::Tiled && g.idx_q_heads == 32 && g.idx_key_dim == 128 &&
            o.dsa_score_load == DsaScoreLoad::Automatic)
            o.dsa_score_load = DsaScoreLoad::Prefetch;
        if (g.idx_kpool > 0 && g.idx_top_k / g.idx_kpool <= 512) o.dsa_parallel_topk = true;
    }
    return o;
}

Options resolve_execution_options(const Options& requested, const Geometry& g) {
    auto options = resolve_options(requested);
    if (options.dsa_score_load != DsaScoreLoad::Automatic && options.dsa_score_load != DsaScoreLoad::Direct &&
        options.dsa_score_load != DsaScoreLoad::Prefetch)
        throw std::invalid_argument("glm5: invalid DSA score load mode");
    if (options.flash_kv_load != FlashKvLoad::Automatic && options.flash_kv_load != FlashKvLoad::Direct &&
        options.flash_kv_load != FlashKvLoad::Prefetch)
        throw std::invalid_argument("glm5: invalid Flash KV load mode");
    if (options.mla_attention_layout != MlaAttentionLayout::Automatic &&
        options.mla_attention_layout != MlaAttentionLayout::TokenMajor &&
        options.mla_attention_layout != MlaAttentionLayout::HeadMajor)
        throw std::invalid_argument("glm5: invalid MLA attention layout");
    if (options.cpu_expert_mode != CpuExpertMode::Automatic && options.cpu_expert_mode != CpuExpertMode::Reference &&
        options.cpu_expert_mode != CpuExpertMode::Rows)
        throw std::invalid_argument("glm5: invalid CPU expert execution mode");
    if (options.gpu_expert_mode != GpuExpertMode::Automatic && options.gpu_expert_mode != GpuExpertMode::Reference &&
        options.gpu_expert_mode != GpuExpertMode::Single)
        throw std::invalid_argument("glm5: invalid GPU expert execution mode");
    if (options.reference || options.prefill_reference) {
        if (options.gpu_experts_mib == default_gpu_expert_budget) options.gpu_experts_mib = -1;
        if (options.gpu_experts_mib < 0 &&
            (options.gpu_warm || options.gpu_heat_quotas || options.gpu_stage_slots || options.gpu_check))
            throw std::invalid_argument("glm5: cache features require an enabled GPU expert budget");
        options.cpu_expert_mode = CpuExpertMode::Reference;
        options.gpu_expert_mode = GpuExpertMode::Reference;
        options.mla_attention_layout = MlaAttentionLayout::TokenMajor;
        options.flash_kv_load = FlashKvLoad::Direct;
        options.prefill_transfer_cost = false;
        options.dsa_score_load = DsaScoreLoad::Direct;
        if (!options.prefill_weight_stage_mib) options.prefill_weight_stage_mib = 1;
        return options;
    }
    int arch = 0;
#ifndef STRATA_USE_HIP
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, options.gpu), "glm5 automatic compute device");
    arch = properties.major * 10 + properties.minor;
#endif
    options = select_compute_options(options, g, arch);
    if (options.gpu_experts_mib < 0 &&
        (options.gpu_warm || options.gpu_heat_quotas || options.gpu_stage_slots || options.gpu_check))
        throw std::invalid_argument("glm5: cache features require an enabled GPU expert budget");
    if (options.dsa_score_load == DsaScoreLoad::Automatic) options.dsa_score_load = DsaScoreLoad::Direct;
    if (options.mla_attention_layout == MlaAttentionLayout::Automatic)
        options.mla_attention_layout = MlaAttentionLayout::TokenMajor;
    if (options.flash_kv_load == FlashKvLoad::Automatic) options.flash_kv_load = FlashKvLoad::Direct;
    if (options.gpu_expert_mode == GpuExpertMode::Automatic) options.gpu_expert_mode = GpuExpertMode::Reference;
    if (options.cpu_expert_mode == CpuExpertMode::Automatic)
        options.cpu_expert_mode = arch == 86 && cpu_row_pool_default(g, options.cpu_threads) ?
            CpuExpertMode::Rows : CpuExpertMode::Reference;
    if (!options.prefill_weight_stage_mib) options.prefill_weight_stage_mib = 1;
    return options;
}

struct Model::Impl {
    ModelFile file;
    Options options;
    std::unique_ptr<HostState> pending;
    std::unique_ptr<DecodeExecutor> decode;
    PrefillStats stats;

    Impl(const std::string& path, const Options& requested)
        : file(path, requested.dsa_tail), options(resolve_options(requested)) {
        const auto& o = options;
        init_cpu();
        if (o.gpu < 0 || o.max_context < 1 || o.max_context > max_context_limit || o.prefill < 1 || o.prefill > 4096 ||
            o.mtp < 0 || o.mtp > 7 || o.cpu_threads < 1 || o.cpu_threads > 256 ||
            o.gpu_experts_mib < default_gpu_expert_budget || o.gpu_reserve_mib > 131072 ||
            o.prefill_gpu_mib > 131072 || !o.prefill_host_mib || o.prefill_host_mib > 1048576 ||
            o.prefill_reserve_mib > 131072 || (o.gpu_check && o.gpu_experts_mib == -1) ||
            o.prefill_group_layers < 0 || o.prefill_group_layers > 64 ||
            o.prefill_prefetch_layers < 0 || o.prefill_prefetch_layers > 64 ||
            !o.prefill_transfer_mib || o.prefill_transfer_mib > 131072 ||
            o.prefill_weight_stage_mib > 64 ||
            (!o.full_layer_prefill && (o.prefill_auto || o.prefill_group_layers ||
                                      o.prefill_prefetch_layers || o.prefill_pipeline ||
                                      o.prefill_reference || o.prefill_no_pipeline || o.prefill_expert_rows)) ||
            o.gpu_stage_slots < 0 || o.gpu_stage_slots > 8 || o.dsa_tail < -1 || o.dsa_tail > 1 ||
            (o.dsa_tail >= 0 && !o.dsa) ||
            (o.dsa_score != DsaScore::Automatic && !o.dsa) ||
            (o.dsa_parallel_topk && !o.dsa) ||
            (o.dsa_score != DsaScore::Automatic && o.dsa_score != DsaScore::Scalar &&
             o.dsa_score != DsaScore::Parallel && o.dsa_score != DsaScore::Tiled) ||
            ((o.gpu_warm || o.gpu_stage_slots) && o.gpu_experts_mib == -1) ||
            (o.gpu_warm && !o.full_layer_prefill) || (o.gpu_heat_quotas && !o.gpu_warm))
            throw std::invalid_argument("glm5: invalid execution or memory budget options");
        if (o.flash_attn && o.mtp && !o.dsa)
            throw std::invalid_argument("glm5: Flash MTP currently requires --dsa");
        if (o.cpu_threads > 1 && o.cpu_expert_mode != CpuExpertMode::Rows && !cpu_parallel_available())
            throw std::invalid_argument("glm5: multiple CPU expert threads require an OpenMP-enabled build");
        file.validate();
        options = resolve_execution_options(options, file.geometry());
        if (o.kda_register_state && file.geometry().kda_head_dim != 128)
            throw std::invalid_argument("glm5: register KDA requires head dimension 128");
        if (o.dsa) file.validate_dsa(o.mtp > 0);
        if (o.mtp) file.validate_mtp();
        if (!o.full_layer_prefill) decode = std::make_unique<DecodeExecutor>(file, options);
    }

    DecodeExecutor& decoder() {
        if (!decode) {
            decode = std::make_unique<DecodeExecutor>(file, options, pending.get());
            pending.reset();
        }
        return *decode;
    }

    const std::vector<float>& prefill(const std::vector<int32_t>& ids) {
        if (!options.full_layer_prefill) return decoder().prefill(ids);
        const auto position = decode ? decode->position() : (pending ? pending->position : 0);
        if (ids.empty() || ids.size() > static_cast<uint64_t>(options.max_context - position))
            throw std::invalid_argument("glm5 full-layer: empty prompt or context exhausted");
        for (const auto id : ids)
            if (id < 0 || id >= file.geometry().n_vocab) throw std::invalid_argument("glm5: token outside vocabulary");
        const auto previous_bytes = decode ? host_state_bytes(file.geometry(), position, options.mtp > 0,
                                                               options.gpu_warm, options.dsa) :
                                             (pending ? pending->bytes() : 0);
        auto admission = options;
        if (admission.prefill_auto) admission.prefill = 1;
        if (full_layer_host_bytes(file, admission, ids.size(), position, previous_bytes) >
            checked_mul(options.prefill_host_mib, 1048576))
            throw std::runtime_error("glm5 full-layer host tensor budget exceeded");
        if (decode) {
            pending = std::make_unique<HostState>(decode->export_state());
            decode.reset();
        }
        const HostState empty;
        const auto& initial = pending ? *pending : empty;
        // Keep the previous host checkpoint intact if a bank cannot fit or an upload fails.
        auto result = full_layer_prefill(file, options, ids, initial);
        stats = result.stats;
        pending = std::make_unique<HostState>(std::move(result.state));
        return pending->logits;
    }
};

Model::Model(const std::string& path, int gpu, int64_t context)
    : Model(path, Options{.gpu = gpu, .max_context = context}) {}
Model::Model(const std::string& path, const Options& options) : impl_(std::make_unique<Impl>(path, options)) {}
Model::~Model() = default;
const Geometry& Model::geometry() const { return impl_->file.geometry(); }
const Options& Model::execution_options() const { return impl_->options; }
const std::set<int32_t>& Model::stop_ids() const { return impl_->file.stop_ids(); }
const std::vector<float>& Model::token(int32_t id) { return impl_->decoder().token(id); }
const std::vector<float>& Model::prefill(const std::vector<int32_t>& ids) { return impl_->prefill(ids); }
std::vector<int32_t> Model::generate(int count) {
    if (impl_->pending) {
        if (count < 0 || count > impl_->options.max_context - impl_->pending->position)
            throw std::invalid_argument("glm5: generation exceeds the remaining context");
        if (!count) return {};
    }
    return impl_->decoder().generate(count);
}
ExpertCacheStats Model::cache_stats() const {
    return impl_->decode ? impl_->decode->cache_stats() : ExpertCacheStats{};
}
MtpStats Model::mtp_stats() const { return impl_->decode ? impl_->decode->mtp_stats() : MtpStats{}; }
PrefillStats Model::prefill_stats() const { return impl_->stats; }
ProbeStats Model::decode_probe_stats() const {
    return impl_->decode ? impl_->decode->probe_stats() : ProbeStats{};
}
bool Model::decode_loaded() const { return static_cast<bool>(impl_->decode); }
void Model::prepare_decode() { impl_->decoder(); }
void Model::reset() {
    impl_->stats = {};
    if (impl_->options.full_layer_prefill) {
        impl_->decode.reset();
        impl_->pending.reset();
    } else impl_->decoder().reset();
}

}  // namespace strata::glm5
