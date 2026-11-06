#include "strata/glm5/gpu_experts.hpp"
#include "strata/glm5/cache_policy.hpp"
#include "strata/glm5/model_file.hpp"
#include "strata/glm5/glm.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "gpu_support.hpp"
#include "expert_compute.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>

namespace strata::glm5 {
namespace {

struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreateWithFlags(&value, cudaEventDisableTiming), "glm5 upload event"); }
    ~Event() { if (value) cudaEventDestroy(value); }
};

struct CopyStream {
    cudaStream_t value = nullptr;
    CopyStream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "glm5 copy stream"); }
    ~CopyStream() { if (value) cudaStreamDestroy(value); }
};

struct Stage {
    uint8_t* data = nullptr;
    Event done;
    bool pending = false;
    explicit Stage(size_t bytes) {
        check(cudaHostAlloc(reinterpret_cast<void**>(&data), bytes, cudaHostAllocDefault), "glm5 pinned staging");
    }
    ~Stage() { if (data) cudaFreeHost(data); }
};

struct Uploads {
    CopyStream copy;
    std::vector<std::unique_ptr<Stage>> ring;
    std::vector<std::unique_ptr<Event>> ready, used;
    std::vector<bool> used_recorded;
    size_t next = 0;
    Uploads(int count, size_t bytes) {
        for (int i = 0; i < count; ++i) ring.push_back(std::make_unique<Stage>(bytes));
    }
    ~Uploads() {
        const auto status = cudaStreamSynchronize(copy.value);
        if (status != cudaSuccess)
            std::fprintf(stderr, "glm5 upload cleanup: %s\n", cudaGetErrorString(status));
        for (size_t s = 0; s < used.size(); ++s) {
            if (!used_recorded[s]) continue;
            const auto used_status = cudaEventSynchronize(used[s]->value);
            if (used_status != cudaSuccess)
                std::fprintf(stderr, "glm5 expert read cleanup: %s\n", cudaGetErrorString(used_status));
        }
    }
    void slots(size_t count) {
        used_recorded.assign(count, false);
        for (size_t i = 0; i < count; ++i) {
            ready.push_back(std::make_unique<Event>());
            used.push_back(std::make_unique<Event>());
        }
    }
    void drain() {
        check(cudaStreamSynchronize(copy.value), "glm5 upload drain");
        for (size_t s = 0; s < used.size(); ++s)
            if (used_recorded[s]) check(cudaEventSynchronize(used[s]->value), "glm5 expert read drain");
    }
};

}  // namespace

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
    bool track_stream_heat = false;
    bool direct_single = false;
    std::unique_ptr<Uploads> uploads;
    std::unique_ptr<ExpertComputer> computer;

    void* alloc(size_t bytes) {
        auto buffer = std::make_unique<Buffer>(bytes);
        void* p = buffer->get();
        allocations.push_back(std::move(buffer));
        return p;
    }
    Impl(const ModelFile& model, int layers, int64_t budget_mib, uint64_t reserve_mib,
         const std::vector<uint32_t>& heat, bool heat_quotas, int stage_slots, bool single)
        : file(model), direct_single(single) {
        if (budget_mib < 0 || stage_slots < 0 || stage_slots > 8)
            throw std::invalid_argument("glm5 GPU experts: invalid budget or stage count");
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
        computer = std::make_unique<ExpertComputer>(g);
        if (stage_slots) {
            const auto largest = *std::max_element(bytes.begin(), bytes.end());
            const auto pinned = checked_mul(largest, stage_slots);
            if (!largest || pinned > 256 * 1048576ULL)
                throw std::invalid_argument("glm5 expert staging exceeds the 256 MiB pinned budget");
            uploads = std::make_unique<Uploads>(stage_slots, largest);
            counters.pinned_bytes = pinned;
        }
        size_t free = 0, total = 0;
        check(cudaMemGetInfo(&free, &total), "glm5 GPU expert budget");
        const uint64_t reserve = reserve_mib * 1048576ULL;
        const uint64_t available = free > reserve ? free - reserve : 0;
        if (static_cast<uint64_t>(budget_mib) > UINT64_MAX / 1048576ULL)
            throw std::invalid_argument("glm5 GPU expert budget overflows");
        const uint64_t budget = budget_mib ? static_cast<uint64_t>(budget_mib) * 1048576ULL : available;
        if (budget > available) throw std::runtime_error("glm5 GPU expert budget leaves insufficient VRAM reserve");
        const auto quotas = heat_quotas ?
            ExpertCachePolicy::heat_quotas(bytes, budget, static_cast<int>(g.n_expert), heat) :
            ExpertCachePolicy::quotas(bytes, budget, static_cast<int>(g.n_expert));
        policy = std::make_unique<ExpertCachePolicy>(quotas, static_cast<int>(g.n_expert));
        if (!policy->slots()) throw std::runtime_error("glm5 GPU expert budget cannot hold one slot per MoE layer");
        uint64_t allocated = 0;
        for (int l = 0; l < layers; ++l) allocated += static_cast<uint64_t>(quotas[l]) * bytes[l];
        auto* arena = static_cast<uint8_t*>(alloc(allocated));
        for (int l = 0; l < layers; ++l)
            for (int i = 0; i < quotas[l]; ++i) { slots.push_back(arena); arena += bytes[l]; }
        counters.bytes = allocated;
        counters.slots = policy->slots();
        if (uploads) uploads->slots(policy->slots());
        std::fprintf(stderr, "glm5 GPU experts: %d slots, %.1f MiB, %llu MiB reserve\n",
                     policy->slots(), allocated / 1048576.0, static_cast<unsigned long long>(reserve_mib));
        if (heat_quotas) {
            int minimum = static_cast<int>(g.n_expert), maximum = 0;
            for (size_t l = 0; l < bytes.size(); ++l) {
                if (!bytes[l]) continue;
                minimum = std::min(minimum, quotas[l]);
                maximum = std::max(maximum, quotas[l]);
            }
            std::fprintf(stderr, "glm5 heat quotas: min=%d max=%d slots/layer\n", minimum, maximum);
        }
    }

    void fill(int layer, int slot, int expert, cudaStream_t stream) {
        const auto& shape = layout.at(layer);
        auto* destination = slots.at(slot);
        Stage* stage = nullptr;
        uint8_t* packed = nullptr;
        if (uploads) {
            stage = uploads->ring[uploads->next++ % uploads->ring.size()].get();
            if (stage->pending) check(cudaEventSynchronize(stage->done.value), "glm5 staging reuse");
            packed = stage->data;
            if (uploads->used_recorded[slot])
                check(cudaStreamWaitEvent(uploads->copy.value, uploads->used[slot]->value, 0), "glm5 slot reuse");
        }
        for (const auto& [role, size] : std::vector<std::pair<const char*, uint64_t>>{
                 {"gate", shape.gate_bytes}, {"up", shape.up_bytes}, {"down", shape.down_bytes}}) {
            const auto* source = file.data(shape.prefix + role + "_exps.weight") + static_cast<uint64_t>(expert) * size;
            if (stage) {
                std::memcpy(packed, source, size);
                packed += size;
            } else {
                check(cudaMemcpyAsync(destination, source, size, cudaMemcpyHostToDevice, stream), "glm5 expert upload");
                destination += size;
            }
        }
        if (stage) {
            check(cudaMemcpyAsync(destination, stage->data, shape.bytes(), cudaMemcpyHostToDevice, uploads->copy.value),
                  "glm5 staged expert upload");
            check(cudaEventRecord(stage->done.value, uploads->copy.value), "glm5 staging completion");
            check(cudaEventRecord(uploads->ready[slot]->value, uploads->copy.value), "glm5 expert readiness");
            stage->pending = true;
        }
        counters.upload_bytes += shape.bytes();
        if (policy->owner(slot) >= 0 && policy->owner(slot) != expert) ++counters.swaps;
        policy->bind(layer, slot, expert);
        if (policy->resident(layer, expert) != slot || policy->owner(slot) != expert)
            throw std::runtime_error("glm5 GPU expert slot table is inconsistent after upload");
    }

    void compute(int layer, int slot, const float* input, const std::vector<ExpertMiss>& entries,
                 float* output, cudaStream_t stream, const void* prepared_single) {
        const auto& shape = layout[layer];
        const auto* weights = slots.at(slot);
        if (uploads)
            check(cudaStreamWaitEvent(stream, uploads->ready[slot]->value, 0), "glm5 expert wait upload");
        const ExpertWeights w{weights, weights + shape.gate_bytes, weights + shape.gate_bytes + shape.up_bytes,
                              shape.gate, shape.up, shape.down};
        if (prepared_single) {
            computer->run_single(w, prepared_single, output + entries.front().row * file.geometry().n_embd, stream);
            ++counters.direct_rows;
            counters.avoided_d2d_bytes += 2 * file.geometry().n_embd * sizeof(float);
        } else computer->run(w, input, entries, output, stream);
        if (uploads) {
            check(cudaEventRecord(uploads->used[slot]->value, stream), "glm5 expert last read");
            uploads->used_recorded[slot] = true;
        }
    }
};

GpuExperts::GpuExperts(const ModelFile& file, int layers, int64_t budget, uint64_t reserve,
                       const std::vector<uint32_t>& heat, bool heat_quotas, int stage_slots, bool direct_single)
    : impl_(std::make_unique<Impl>(file, layers, budget, reserve, heat, heat_quotas, stage_slots, direct_single)) {}
GpuExperts::~GpuExperts() = default;
const ExpertCacheStats& GpuExperts::stats() const { return impl_->counters; }
const std::vector<uint32_t>& GpuExperts::heat() const { return impl_->policy->heat(); }
void GpuExperts::warm(const std::vector<uint32_t>& heat, void* opaque) {
    auto& p = *impl_;
    const auto stream = static_cast<cudaStream_t>(opaque);
    if (!stream) throw std::invalid_argument("glm5 cache warmup requires a stream");
    p.policy->seed(heat);
    p.track_stream_heat = true;
    const auto ne = static_cast<int>(p.file.geometry().n_expert);
    std::vector<int> order(ne);
    for (size_t l = 0; l < p.layout.size(); ++l) {
        if (!p.layout[l].bytes()) continue;
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return heat[l * ne + a] > heat[l * ne + b]; });
        for (const auto expert : order) {
            if (!heat[l * ne + expert]) break;
            const auto slot = p.policy->candidate(l, expert, false, false);
            if (slot < 0) break;
            p.fill(l, slot, expert, stream);
            ++p.counters.warmed;
        }
    }
    if (p.uploads) p.uploads->drain();
    else check(cudaStreamSynchronize(stream), "glm5 expert warmup sync");
}
void GpuExperts::reset() {
    if (impl_->uploads) impl_->uploads->drain();
    impl_->policy->clear();
    impl_->track_stream_heat = false;
    const auto saved = impl_->counters;
    impl_->counters = {};
    impl_->counters.bytes = saved.bytes;
    impl_->counters.slots = saved.slots;
    impl_->counters.pinned_bytes = saved.pinned_bytes;
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
    if (!stream_all || p.track_stream_heat) p.policy->observe(layer, ids, tokens * k);
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
    const bool single = p.direct_single && tokens == 1 && !groups.empty() &&
        std::all_of(groups.begin(), groups.end(), [](const auto& group) { return group.second.size() == 1; });
    const void* prepared = nullptr;
    if (single) {
        prepared = p.computer->prepare_single_input(input, stream);
        ++p.counters.single_input_quantizations;
        p.counters.avoided_input_quantizations += groups.size() - 1;
    }
    for (const auto& [expert, entries] : groups) {
        int slot = p.policy->resident(layer, expert);
        if (slot < 0) {
            slot = p.policy->candidate(layer, expert, true, true);
            if (slot < 0) throw std::runtime_error("glm5 GPU experts: no slot for streamed expert");
            p.fill(layer, slot, expert, stream);
            p.counters.streamed += entries.size();
        } else p.counters.hits += entries.size();
        p.compute(layer, slot, input, entries, output, stream, prepared);
    }
    if (!p.policy->consistent(layer)) throw std::runtime_error("glm5 GPU expert slot table is inconsistent after dispatch");
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
    if (!p.policy->consistent(layer)) throw std::runtime_error("glm5 GPU expert slot table is inconsistent after admission");
}

}  // namespace strata::glm5
