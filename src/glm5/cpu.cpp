#include "cpu.hpp"

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>

namespace strata::glm5 {

bool cpu_parallel_available() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

std::string cpu_isa_summary() {
    std::ostringstream out;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    out << "host_avx512f=" << !!__builtin_cpu_supports("avx512f")
        << " host_avx512bw=" << !!__builtin_cpu_supports("avx512bw")
        << " host_avx512_vnni=" << !!__builtin_cpu_supports("avx512vnni") << ' ';
#endif
    out << "ggml_avx2=" << ggml_cpu_has_avx2() << " ggml_avx512=" << ggml_cpu_has_avx512()
        << " ggml_avx512_vnni=" << ggml_cpu_has_avx512_vnni()
        << " ggml_avx512_bf16=" << ggml_cpu_has_avx512_bf16() << " ggml_fma=" << ggml_cpu_has_fma();
    return out.str();
}

bool cpu_row_pool_default(const Geometry& g, int threads) {
#if defined(__linux__) && defined(__GNUC__) && defined(__x86_64__)
    __builtin_cpu_init();
    return cpu_parallel_available() && __builtin_cpu_is("skylake-avx512") &&
        std::thread::hardware_concurrency() == 12 && ggml_cpu_has_avx2() &&
        (threads == 3 || threads == 6 || threads == 12) && g.n_embd == 4096 && g.n_ff == 2048 &&
        g.n_expert == 288 && g.n_expert_used == 8;
#else
    (void) g; (void) threads;
    return false;
#endif
}

void init_cpu() {
    // CPU dots and dequantizers need ggml's tables even though this engine does not build a ggml graph.
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

const ggml_type_traits_cpu& cpu_traits(uint32_t type) {
    if (type >= GGML_TYPE_COUNT) throw std::runtime_error("glm5: unsupported CPU GGUF type " + std::to_string(type));
    const auto* traits = ggml_get_type_traits_cpu(static_cast<ggml_type>(type));
    if (!traits || !traits->vec_dot)
        throw std::runtime_error("glm5: no CPU dot product for GGUF type " + std::to_string(type));
    return *traits;
}

std::vector<uint8_t> quantize(const float* x, int64_t width, uint32_t type) {
    const auto act = cpu_traits(type).vec_dot_type;
    const auto* tr = ggml_get_type_traits_cpu(act);
    if (!tr || !tr->from_float || width % ggml_blck_size(act))
        throw std::runtime_error("glm5: unsupported expert activation quantization");
    std::vector<uint8_t> out(ggml_row_size(act, width));
    tr->from_float(x, out.data(), width);
    return out;
}

void dequantize(const TensorInfo& t, const uint8_t* src, float* dst, int64_t count) {
    if (t.type == GGML_TYPE_F32) {
        std::memcpy(dst, src, count * sizeof(float));
        return;
    }
    if (t.type >= GGML_TYPE_COUNT)
        throw std::runtime_error(t.name + ": unsupported GGUF type");
    const auto* tr = ggml_get_type_traits(static_cast<ggml_type>(t.type));
    if (!tr || !tr->to_float || count % ggml_blck_size(static_cast<ggml_type>(t.type)))
        throw std::runtime_error(t.name + ": cannot dequantize complete rows");
    tr->to_float(src, dst, count);
}

struct CpuRowPool::Impl {
    static constexpr int tile_rows = 64, batch_entries = 32;
    struct Layer {
        const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
        const ggml_type_traits_cpu *gt = nullptr, *ut = nullptr, *dt = nullptr;
        const ggml_type_traits_cpu *gact = nullptr, *uact = nullptr, *hact = nullptr;
        uint64_t gr = 0, ur = 0, dr = 0;
        size_t ga_bytes = 0, ua_bytes = 0, ha_bytes = 0;
        bool same_input = false;
    };
    const Geometry& g;
    std::vector<Layer> layers;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable start, finished;
    bool quit = false;
    uint64_t epoch = 0;
    size_t done = 0, tasks = 0;
    std::atomic<size_t> next{0};
    std::atomic<size_t> completed{0};
    bool atomic_completion = false;
    bool down_phase = false;
    const Layer* current = nullptr;
    std::span<const ExpertMiss> entries;
    std::span<float> output;
    std::vector<std::vector<uint8_t>> gate_acts, up_acts;
    std::vector<uint8_t> token_ready, row_seen, hidden_q;
    std::vector<float> hidden;
    size_t ga_capacity = 0, ua_capacity = 0, ha_capacity = 0;

    Impl(const ModelFile& file, int count, int threads, bool atomic_wait)
        : g(file.geometry()), atomic_completion(atomic_wait) {
        init_cpu();
        if (count < g.n_layers || count > g.n_layers + 1 || threads < 1 || threads > 256)
            throw std::invalid_argument("glm5 CPU row pool: invalid layers or threads");
        layers.resize(count);
        for (int l = static_cast<int>(g.n_layer_dense_lead); l < count; ++l) {
            const auto prefix = "blk." + std::to_string(l) + ".ffn_";
            const auto& gate = file.tensor(prefix + "gate_exps.weight");
            const auto& up = file.tensor(prefix + "up_exps.weight");
            const auto& down = file.tensor(prefix + "down_exps.weight");
            auto& v = layers[l];
            v.gate = file.data(gate.name); v.up = file.data(up.name); v.down = file.data(down.name);
            v.gt = &cpu_traits(gate.type); v.ut = &cpu_traits(up.type); v.dt = &cpu_traits(down.type);
            v.gact = ggml_get_type_traits_cpu(v.gt->vec_dot_type);
            v.uact = ggml_get_type_traits_cpu(v.ut->vec_dot_type);
            v.hact = ggml_get_type_traits_cpu(v.dt->vec_dot_type);
            if (!v.gact || !v.uact || !v.hact || !v.gact->from_float || !v.uact->from_float ||
                !v.hact->from_float || g.n_embd % ggml_blck_size(v.gt->vec_dot_type) ||
                g.n_embd % ggml_blck_size(v.ut->vec_dot_type) || g.n_ff % ggml_blck_size(v.dt->vec_dot_type))
                throw std::runtime_error(prefix + ": unsupported CPU row-pool activation format");
            v.gr = tensor_payload_bytes(gate) / (g.n_expert * g.n_ff);
            v.ur = tensor_payload_bytes(up) / (g.n_expert * g.n_ff);
            v.dr = tensor_payload_bytes(down) / (g.n_expert * g.n_embd);
            v.ga_bytes = ggml_row_size(v.gt->vec_dot_type, g.n_embd);
            v.ua_bytes = ggml_row_size(v.ut->vec_dot_type, g.n_embd);
            v.ha_bytes = ggml_row_size(v.dt->vec_dot_type, g.n_ff);
            v.same_input = v.gt->vec_dot_type == v.ut->vec_dot_type;
            ga_capacity = std::max(ga_capacity, v.ga_bytes);
            ua_capacity = std::max(ua_capacity, v.ua_bytes);
            ha_capacity = std::max(ha_capacity, v.ha_bytes);
        }
        hidden.resize(batch_entries * g.n_ff);
        hidden_q.resize(batch_entries * ha_capacity);
        try {
            for (int t = 1; t < threads; ++t) workers.emplace_back([this] {
                uint64_t seen = 0;
                for (;;) {
                    std::unique_lock lock(mutex);
                    start.wait(lock, [&] { return quit || epoch != seen; });
                    if (quit) return;
                    seen = epoch;
                    lock.unlock();
                    drain();
                    if (atomic_completion) {
                        if (completed.fetch_add(1, std::memory_order_release) + 1 == workers.size())
                            completed.notify_one();
                    } else {
                        lock.lock();
                        ++done;
                        finished.notify_one();
                    }
                }
            });
        } catch (...) { stop(); throw; }
    }
    ~Impl() { stop(); }
    void stop() {
        { std::lock_guard lock(mutex); quit = true; }
        start.notify_all();
        for (auto& worker : workers) if (worker.joinable()) worker.join();
    }
    void drain() noexcept {
        const auto& v = *current;
        const auto width = down_phase ? g.n_embd : g.n_ff;
        const size_t tiles = (width + tile_rows - 1) / tile_rows;
        for (;;) {
            const auto task = next.fetch_add(1, std::memory_order_relaxed);
            if (task >= tasks) return;
            const auto index = task / tiles;
            const auto& entry = entries[index];
            const auto first = static_cast<int64_t>(task % tiles) * tile_rows;
            const auto last = std::min(first + tile_rows, width);
            if (!down_phase) {
                const auto* gw = v.gate + entry.expert * g.n_ff * v.gr;
                const auto* uw = v.up + entry.expert * g.n_ff * v.ur;
                const auto& ga = gate_acts[entry.token];
                const auto& ua = v.same_input ? ga : up_acts[entry.token];
                for (auto r = first; r < last; ++r) {
                    float gv = 0, uv = 0;
                    v.gt->vec_dot(g.n_embd, &gv, 0, gw + r * v.gr, 0, ga.data(), 0, 1);
                    v.ut->vec_dot(g.n_embd, &uv, 0, uw + r * v.ur, 0, ua.data(), 0, 1);
                    hidden[index * g.n_ff + r] = swiglu(gv, uv, static_cast<float>(g.swiglu_clamp));
                }
            } else {
                const auto* dw = v.down + entry.expert * g.n_embd * v.dr;
                const auto* ha = hidden_q.data() + index * ha_capacity;
                for (auto r = first; r < last; ++r)
                    v.dt->vec_dot(g.n_ff, &output[entry.row * g.n_embd + r], 0,
                                  dw + r * v.dr, 0, ha, 0, 1);
            }
        }
    }
    void dispatch(bool down) {
        // A new phase is published only after every worker has returned from the previous drain.
        {
            std::lock_guard lock(mutex);
            down_phase = down;
            tasks = entries.size() * (((down ? g.n_embd : g.n_ff) + tile_rows - 1) / tile_rows);
            next.store(0, std::memory_order_relaxed);
            done = 0;
            if (atomic_completion) completed.store(0, std::memory_order_relaxed);
            ++epoch;
        }
        start.notify_all();
        drain();
        if (atomic_completion) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(100);
            unsigned polls = 0;
            auto observed = completed.load(std::memory_order_acquire);
            while (observed != workers.size()) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
                __builtin_ia32_pause();
#else
                std::this_thread::yield();
#endif
                if ((++polls & 63) == 0 && std::chrono::steady_clock::now() >= deadline)
                    completed.wait(observed, std::memory_order_acquire);
                observed = completed.load(std::memory_order_acquire);
            }
        } else {
            std::unique_lock lock(mutex);
            finished.wait(lock, [&] { return done == workers.size(); });
        }
    }
    void run(int layer, int tokens, std::span<const float> input, std::span<const ExpertMiss> requested,
             std::span<float> destination) {
        if (layer < 0 || static_cast<size_t>(layer) >= layers.size() || !layers[layer].gate || tokens < 1 ||
            tokens > 4096 || input.size() != static_cast<size_t>(tokens * g.n_embd) ||
            destination.size() != static_cast<size_t>(tokens * g.n_expert_used * g.n_embd))
            throw std::invalid_argument("glm5 CPU row pool: invalid input shape or layer");
        auto& v = layers[layer];
        token_ready.assign(tokens, 0);
        row_seen.assign(tokens * g.n_expert_used, 0);
        if (gate_acts.size() < static_cast<size_t>(tokens)) gate_acts.resize(tokens);
        if (up_acts.size() < static_cast<size_t>(tokens)) up_acts.resize(tokens);
        for (const auto& entry : requested) {
            if (entry.expert < 0 || entry.expert >= g.n_expert || entry.token < 0 || entry.token >= tokens ||
                entry.row < 0 || entry.row >= tokens * g.n_expert_used || row_seen[entry.row]++)
                throw std::invalid_argument("glm5 CPU row pool: invalid or duplicate output row");
        }
        for (const auto& entry : requested) {
            if (token_ready[entry.token]) continue;
            auto& ga = gate_acts[entry.token]; ga.resize(ga_capacity);
            v.gact->from_float(input.data() + entry.token * g.n_embd, ga.data(), g.n_embd);
            if (!v.same_input) {
                auto& ua = up_acts[entry.token]; ua.resize(ua_capacity);
                v.uact->from_float(input.data() + entry.token * g.n_embd, ua.data(), g.n_embd);
            }
            token_ready[entry.token] = 1;
        }
        current = &v; output = destination;
        for (size_t first = 0; first < requested.size(); first += batch_entries) {
            entries = requested.subspan(first, std::min<size_t>(batch_entries, requested.size() - first));
            dispatch(false);
            for (size_t i = 0; i < entries.size(); ++i)
                v.hact->from_float(hidden.data() + i * g.n_ff, hidden_q.data() + i * ha_capacity, g.n_ff);
            dispatch(true);
        }
    }
};

CpuRowPool::CpuRowPool(const ModelFile& file, int layers, int threads, bool atomic_completion)
    : impl_(std::make_unique<Impl>(file, layers, threads, atomic_completion)) {}
CpuRowPool::~CpuRowPool() = default;
void CpuRowPool::run(int layer, int tokens, std::span<const float> input, std::span<const ExpertMiss> entries,
                     std::span<float> output) { impl_->run(layer, tokens, input, entries, output); }
uint64_t CpuRowPool::expert_bytes(int layer) const {
    if (layer < 0 || static_cast<size_t>(layer) >= impl_->layers.size() || !impl_->layers[layer].gate)
        throw std::invalid_argument("glm5 CPU row pool: invalid expert layer");
    const auto& v = impl_->layers[layer];
    return impl_->g.n_ff * (v.gr + v.ur) + impl_->g.n_embd * v.dr;
}

}  // namespace strata::glm5
