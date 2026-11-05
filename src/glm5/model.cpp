#include "strata/glm5/model.hpp"
#include "cpu.hpp"
#include "decode.hpp"
#include "prefill.hpp"

namespace strata::glm5 {

Options resolve_options(Options o) {
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
        if (o.gpu < 0 || o.max_context < 1 || o.max_context > 131072 || o.prefill < 1 || o.prefill > 4096 ||
            o.mtp < 0 || o.mtp > 7 || o.cpu_threads < 1 || o.cpu_threads > 256 ||
            o.gpu_experts_mib < -1 || o.gpu_reserve_mib > 131072 ||
            o.prefill_gpu_mib > 131072 || !o.prefill_host_mib || o.prefill_host_mib > 1048576 ||
            o.prefill_reserve_mib > 131072 || (o.gpu_check && o.gpu_experts_mib < 0) ||
            o.prefill_group_layers < 0 || o.prefill_group_layers > 64 ||
            o.prefill_prefetch_layers < 0 || o.prefill_prefetch_layers > 64 ||
            !o.prefill_transfer_mib || o.prefill_transfer_mib > 131072 ||
            (!o.full_layer_prefill && (o.prefill_auto || o.prefill_group_layers ||
                                      o.prefill_prefetch_layers || o.prefill_pipeline ||
                                      o.prefill_reference || o.prefill_no_pipeline)) ||
            o.gpu_stage_slots < 0 || o.gpu_stage_slots > 8 || o.dsa_tail < -1 || o.dsa_tail > 1 ||
            (o.dsa_tail >= 0 && !o.dsa) ||
            (o.dsa_score != DsaScore::Automatic && !o.dsa) ||
            (o.dsa_score != DsaScore::Automatic && o.dsa_score != DsaScore::Scalar &&
             o.dsa_score != DsaScore::Parallel) ||
            ((o.gpu_warm || o.gpu_stage_slots) && o.gpu_experts_mib < 0) ||
            (o.gpu_warm && !o.full_layer_prefill) || (o.gpu_heat_quotas && !o.gpu_warm))
            throw std::invalid_argument("glm5: invalid execution or memory budget options");
        if (o.flash_attn && o.mtp && !o.dsa)
            throw std::invalid_argument("glm5: Flash MTP currently requires --dsa");
        if (o.cpu_threads > 1 && !cpu_parallel_available())
            throw std::invalid_argument("glm5: multiple CPU expert threads require an OpenMP-enabled build");
        file.validate();
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
