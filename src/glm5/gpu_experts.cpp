#include "strata/glm5/gpu_experts.hpp"
#include "strata/glm5/cache_policy.hpp"
#include "strata/glm5/model_file.hpp"
#include "strata/glm5/glm.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "gpu_support.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace strata::glm5 {

struct GpuExperts::Impl {
    struct Layout {
        int gate = -1, up = -1, down = -1;
        uint64_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
        std::string prefix;
        uint64_t bytes() const { return gate_bytes + up_bytes + down_bytes; }
    };
    const ModelFile& file;
    std::vector<Layout> layout;
    std::unique_ptr<ExpertCachePolicy> policy;
    std::vector<std::unique_ptr<Buffer>> allocations;
    std::vector<uint8_t*> slots;
    std::vector<ExpertMiss> missing;
    ExpertCacheStats counters;
    float *x = nullptr, *gate = nullptr, *up = nullptr, *result = nullptr;
    void *xq = nullptr, *hq = nullptr;

    void* alloc(size_t bytes) {
        auto buffer = std::make_unique<Buffer>(bytes);
        void* p = buffer->get();
        allocations.push_back(std::move(buffer));
        return p;
    }
    Impl(const ModelFile& model, int layers, int64_t budget_mib, uint64_t reserve_mib) : file(model) {
        if (budget_mib < 0) throw std::invalid_argument("glm5 GPU experts: budget must be nonnegative");
        const auto& g = file.geometry();
        if (layers < g.n_layers || layers > g.n_layers + 1 || reserve_mib > UINT64_MAX / 1048576ULL)
            throw std::invalid_argument("glm5 GPU experts: invalid layer range or reserve");
        layout.resize(layers);
        std::vector<uint64_t> bytes(layers);
        for (int l = static_cast<int>(g.n_layer_dense_lead); l < layers; ++l) {
            auto& shape = layout[l];
            shape.prefix = "blk." + std::to_string(l) + ".ffn_";
            const auto& a = file.tensor(shape.prefix + "gate_exps.weight");
            const auto& b = file.tensor(shape.prefix + "up_exps.weight");
            const auto& c = file.tensor(shape.prefix + "down_exps.weight");
            shape.gate = a.type; shape.up = b.type; shape.down = c.type;
            if (!kernels::native_mmvq_supported(a.type) || !kernels::native_mmvq_supported(b.type) ||
                !kernels::native_mmvq_supported(c.type))
                throw std::runtime_error(shape.prefix + ": GPU expert cache needs supported native quantized matrices");
            shape.gate_bytes = tensor_payload_bytes(a) / g.n_expert;
            shape.up_bytes = tensor_payload_bytes(b) / g.n_expert;
            shape.down_bytes = tensor_payload_bytes(c) / g.n_expert;
            bytes[l] = shape.bytes();
        }
        x = static_cast<float*>(alloc(8 * g.n_embd * sizeof(float)));
        result = static_cast<float*>(alloc(8 * g.n_embd * sizeof(float)));
        gate = static_cast<float*>(alloc(8 * g.n_ff * sizeof(float)));
        up = static_cast<float*>(alloc(8 * g.n_ff * sizeof(float)));
        xq = alloc(kernels::native_q8_1_bytes(g.n_embd, 8));
        hq = alloc(kernels::native_q8_1_bytes(g.n_ff, 8));
        size_t free = 0, total = 0;
        check(cudaMemGetInfo(&free, &total), "glm5 GPU expert budget");
        const uint64_t reserve = reserve_mib * 1048576ULL;
        const uint64_t available = free > reserve ? free - reserve : 0;
        if (static_cast<uint64_t>(budget_mib) > UINT64_MAX / 1048576ULL)
            throw std::invalid_argument("glm5 GPU expert budget overflows");
        const uint64_t budget = budget_mib ? static_cast<uint64_t>(budget_mib) * 1048576ULL : available;
        if (budget > available) throw std::runtime_error("glm5 GPU expert budget leaves insufficient VRAM reserve");
        const auto quotas = ExpertCachePolicy::quotas(bytes, budget, static_cast<int>(g.n_expert));
        policy = std::make_unique<ExpertCachePolicy>(quotas, static_cast<int>(g.n_expert));
        if (!policy->slots()) throw std::runtime_error("glm5 GPU expert budget cannot hold one slot per MoE layer");
        uint64_t allocated = 0;
        for (int l = 0; l < layers; ++l) allocated += static_cast<uint64_t>(quotas[l]) * bytes[l];
        auto* arena = static_cast<uint8_t*>(alloc(allocated));
        for (int l = 0; l < layers; ++l)
            for (int i = 0; i < quotas[l]; ++i) { slots.push_back(arena); arena += bytes[l]; }
        counters.bytes = allocated;
        std::fprintf(stderr, "glm5 GPU experts: %d slots, %.1f MiB, %llu MiB reserve\n",
                     policy->slots(), allocated / 1048576.0, static_cast<unsigned long long>(reserve_mib));
    }

    void fill(int layer, int slot, int expert, cudaStream_t stream) {
        const auto& shape = layout.at(layer);
        auto* destination = slots.at(slot);
        for (const auto& [role, size] : std::vector<std::pair<const char*, uint64_t>>{
                 {"gate", shape.gate_bytes}, {"up", shape.up_bytes}, {"down", shape.down_bytes}}) {
            check(cudaMemcpyAsync(destination, file.data(shape.prefix + role + "_exps.weight") +
                                  static_cast<uint64_t>(expert) * size, size, cudaMemcpyHostToDevice, stream),
                  "glm5 expert upload");
            destination += size;
        }
        if (policy->owner(slot) >= 0 && policy->owner(slot) != expert) ++counters.swaps;
        policy->bind(layer, slot, expert);
        if (policy->resident(layer, expert) != slot || policy->owner(slot) != expert)
            throw std::runtime_error("glm5 GPU expert slot table is inconsistent after upload");
    }

    void compute(int layer, int slot, const float* input, const std::vector<ExpertMiss>& entries,
                 float* output, cudaStream_t stream) {
        const auto& g = file.geometry();
        const auto& shape = layout[layer];
        const auto* weights = slots.at(slot);
        for (size_t first = 0; first < entries.size(); first += 8) {
            const auto n = static_cast<int>(std::min<size_t>(8, entries.size() - first));
            for (int i = 0; i < n; ++i)
                check(cudaMemcpyAsync(x + i * g.n_embd, input + entries[first + i].token * g.n_embd,
                                       g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, stream), "glm5 expert gather");
            kernels::native_quantize_q8_1(x, xq, g.n_embd, n, stream);
            kernels::native_mmvq(shape.gate, weights, xq, gate, g.n_embd, g.n_ff, n, stream);
            kernels::native_mmvq(shape.up, weights + shape.gate_bytes, xq, up, g.n_embd, g.n_ff, n, stream);
            kernels::glm_swiglu(gate, up, g.swiglu_limit_or_off(), n * g.n_ff, stream);
            kernels::native_quantize_q8_1(gate, hq, g.n_ff, n, stream);
            kernels::native_mmvq(shape.down, weights + shape.gate_bytes + shape.up_bytes, hq, result,
                                 g.n_ff, g.n_embd, n, stream);
            for (int i = 0; i < n; ++i)
                check(cudaMemcpyAsync(output + entries[first + i].row * g.n_embd, result + i * g.n_embd,
                                       g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, stream), "glm5 expert scatter");
        }
    }
};

GpuExperts::GpuExperts(const ModelFile& file, int layers, int64_t budget, uint64_t reserve)
    : impl_(std::make_unique<Impl>(file, layers, budget, reserve)) {}
GpuExperts::~GpuExperts() = default;
const ExpertCacheStats& GpuExperts::stats() const { return impl_->counters; }
void GpuExperts::reset() {
    impl_->policy->clear();
    const auto bytes = impl_->counters.bytes;
    impl_->counters = {};
    impl_->counters.bytes = bytes;
}

const std::vector<ExpertMiss>& GpuExperts::run(int layer, const float* input, const int32_t* ids, int tokens,
                                            float* output, bool stream_all, void* opaque) {
    auto& p = *impl_;
    const auto stream = static_cast<cudaStream_t>(opaque);
    const int k = static_cast<int>(p.file.geometry().n_expert_used);
    if (!stream || !input || !output || !ids || tokens < 1)
        throw std::invalid_argument("glm5 GPU experts: invalid routing buffers");
    p.missing.clear();
    std::map<int, std::vector<ExpertMiss>> groups;
    if (!stream_all) p.policy->observe(layer, ids, tokens * k);
    for (int t = 0; t < tokens; ++t) {
        for (int z = 0; z < k; ++z) {
            const int expert = ids[t * k + z];
            const int resident = p.policy->resident(layer, expert);
            const ExpertMiss entry{t, t * k + z, expert};
            if (resident >= 0 || stream_all) groups[expert].push_back(entry);
            else p.missing.push_back(entry);
        }
    }
    p.counters.misses += p.missing.size();
    for (const auto& [expert, entries] : groups) {
        int slot = p.policy->resident(layer, expert);
        if (slot < 0) {
            slot = p.policy->candidate(layer, expert, true, true);
            if (slot < 0) throw std::runtime_error("glm5 GPU experts: no slot for streamed expert");
            p.fill(layer, slot, expert, stream);
            p.counters.streamed += entries.size();
        } else p.counters.hits += entries.size();
        p.compute(layer, slot, input, entries, output, stream);
    }
    if (!p.policy->consistent()) throw std::runtime_error("glm5 GPU expert slot table is inconsistent after dispatch");
    return p.missing;
}

void GpuExperts::admit(int layer, const int32_t*, const std::vector<ExpertMiss>& misses, void* opaque) {
    auto& p = *impl_;
    bool swapped = false;
    for (const auto& entry : misses) {
        if (p.policy->resident(layer, entry.expert) >= 0) continue;
        const int slot = p.policy->candidate(layer, entry.expert, false, !swapped);
        if (slot < 0) continue;
        swapped = swapped || p.policy->owner(slot) >= 0;
        p.fill(layer, slot, entry.expert, static_cast<cudaStream_t>(opaque));
    }
    if (!p.policy->consistent()) throw std::runtime_error("glm5 GPU expert slot table is inconsistent after admission");
}

}  // namespace strata::glm5
