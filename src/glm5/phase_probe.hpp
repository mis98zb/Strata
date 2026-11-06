#pragma once

#include "gpu_support.hpp"
#include "strata/glm5/model.hpp"

#include <chrono>
#include <ctime>
#include <memory>
#include <utility>
#include <vector>

namespace strata::glm5 {

class PhaseProbe {
    struct Pair {
        cudaEvent_t first = nullptr, last = nullptr;
        ProbePhase phase = ProbePhase::LayerPre;
        Pair() {
            check(cudaEventCreate(&first), "glm5 probe start event");
            try { check(cudaEventCreate(&last), "glm5 probe end event"); }
            catch (...) { cudaEventDestroy(first); throw; }
        }
        ~Pair() { cudaEventDestroy(first); cudaEventDestroy(last); }
    };
    bool enabled_;
    ProbeStats stats_;
    std::vector<std::unique_ptr<Pair>> pool_;
    std::vector<Pair*> free_, pending_;
    ProbeSample& sample(ProbePhase phase) { return stats_.phases.at(static_cast<size_t>(phase)); }
public:
    static constexpr size_t max_pairs = 256;
    explicit PhaseProbe(bool enabled) : enabled_(enabled) {}
    bool enabled() const { return enabled_; }

    class Host {
        PhaseProbe& probe_;
        ProbePhase phase_;
        std::chrono::steady_clock::time_point start_;
        std::clock_t cpu_ = 0;
    public:
        Host(PhaseProbe& probe, ProbePhase phase) : probe_(probe), phase_(phase) {
            if (probe_.enabled_) {
                start_ = std::chrono::steady_clock::now();
                if (phase == ProbePhase::CpuExperts) cpu_ = std::clock();
            }
        }
        ~Host() {
            if (!probe_.enabled_) return;
            auto& out = probe_.sample(phase_);
            out.host_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start_).count();
            ++out.host_calls;
            if (phase_ == ProbePhase::CpuExperts)
                out.process_cpu_ms += (std::clock() - cpu_) * (1000.0 / CLOCKS_PER_SEC);
        }
        Host(const Host&) = delete;
        Host& operator=(const Host&) = delete;
    };
    Host host(ProbePhase phase) { return Host(*this, phase); }

    void collect(bool wait = false) {
        if (!enabled_) return;
        auto out = pending_.begin();
        for (auto* pair : pending_) {
            const auto status = wait ? cudaEventSynchronize(pair->last) : cudaEventQuery(pair->last);
            if (!wait && status == cudaErrorNotReady) { *out++ = pair; continue; }
            check(status, "glm5 probe event completion");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, pair->first, pair->last), "glm5 probe elapsed time");
            auto& value = sample(pair->phase);
            value.stream_ms += ms;
            ++value.stream_calls;
            free_.push_back(pair);
        }
        pending_.erase(out, pending_.end());
    }

    // Stream spans can include host launch starvation; they are not kernel-busy times.
    template<class Operation> void gpu(ProbePhase phase, cudaStream_t stream, Operation&& operation) {
        if (!enabled_) { std::forward<Operation>(operation)(); return; }
        auto timing = host(phase);
        if (free_.empty()) collect();
        if (free_.empty() && pool_.size() < max_pairs) {
            pool_.push_back(std::make_unique<Pair>());
            free_.push_back(pool_.back().get());
            stats_.event_pairs = pool_.size();
        }
        if (free_.empty()) { ++stats_.forced_waits; collect(true); }
        auto* pair = free_.back();
        free_.pop_back();
        pair->phase = phase;
        try {
            check(cudaEventRecord(pair->first, stream), "glm5 probe start");
            std::forward<Operation>(operation)();
            check(cudaEventRecord(pair->last, stream), "glm5 probe end");
        } catch (...) { free_.push_back(pair); throw; }
        pending_.push_back(pair);
    }
    void bytes(ProbePhase phase, uint64_t bytes) { if (enabled_) sample(phase).bytes += bytes; }
    void cpu_experts(uint64_t count) { if (enabled_) stats_.cpu_experts += count; }
    void cpu_result_copy() { if (enabled_) ++stats_.cpu_result_copies; }
    void rows(uint64_t count) { if (enabled_) stats_.rows += count; }
    ProbeStats snapshot() { collect(true); return stats_; }
    void reset() { collect(true); stats_ = {}; stats_.event_pairs = pool_.size(); }
};

inline thread_local PhaseProbe* current_probe = nullptr;
class ProbeScope {
    PhaseProbe* previous_;
public:
    explicit ProbeScope(PhaseProbe& probe) : previous_(current_probe) {
        current_probe = probe.enabled() ? &probe : nullptr;
    }
    ~ProbeScope() { current_probe = previous_; }
    ProbeScope(const ProbeScope&) = delete;
    ProbeScope& operator=(const ProbeScope&) = delete;
};
template<class Operation> void probe_gpu(ProbePhase phase, void* stream, Operation&& operation) {
    if (current_probe) current_probe->gpu(phase, static_cast<cudaStream_t>(stream), std::forward<Operation>(operation));
    else std::forward<Operation>(operation)();
}
template<class Operation> bool probe_gpu_result(ProbePhase phase, void* stream, Operation&& operation) {
    bool result = false;
    probe_gpu(phase, stream, [&] { result = std::forward<Operation>(operation)(); });
    return result;
}

inline const char* probe_name(ProbePhase phase) {
    constexpr const char* names[] = {
        "weight_wait", "activation_h2d", "activation_d2h", "layer_pre", "route_d2h", "gpu_experts",
        "cpu_experts", "cpu_result_h2d", "cache_admit", "layer_post", "head", "embedding_h2d",
        "state_transfer", "synchronize", "kda_mixer", "mla_mixer", "ffn", "router",
        "mla_query", "mla_absorb", "mla_kv", "mla_index_project", "mla_index_pool", "mla_index_score",
        "mla_index_select", "mla_attention", "mla_deabsorb", "mla_output", "mla_layout"
    };
    static_assert(sizeof(names) / sizeof(*names) == static_cast<size_t>(ProbePhase::Count));
    return names[static_cast<size_t>(phase)];
}

inline ProbeStats probe_delta(ProbeStats current, const ProbeStats& previous) {
    if (current.rows < previous.rows || current.cpu_experts < previous.cpu_experts ||
        current.forced_waits < previous.forced_waits || current.cpu_result_copies < previous.cpu_result_copies)
        throw std::logic_error("glm5 probe counters reset during an interval");
    current.rows -= previous.rows;
    current.cpu_experts -= previous.cpu_experts;
    current.forced_waits -= previous.forced_waits;
    current.cpu_result_copies -= previous.cpu_result_copies;
    for (size_t i = 0; i < current.phases.size(); ++i) {
        auto& value = current.phases[i];
        const auto& before = previous.phases[i];
        if (value.host_calls < before.host_calls || value.stream_calls < before.stream_calls || value.bytes < before.bytes)
            throw std::logic_error("glm5 probe phase counters reset during an interval");
        value.host_ms -= before.host_ms;
        value.process_cpu_ms -= before.process_cpu_ms;
        value.stream_ms -= before.stream_ms;
        value.host_calls -= before.host_calls;
        value.stream_calls -= before.stream_calls;
        value.bytes -= before.bytes;
    }
    return current;
}

}  // namespace strata::glm5
