#include "strata/glm5/cache_policy.hpp"
#include "strata/glm5/speculative.hpp"
#include "strata/glm5/state.hpp"

#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}

int main() {
    try {
        using strata::glm5::ExpertCachePolicy;
        require(ExpertCachePolicy::quotas({0, 100, 200}, 900, 8) == std::vector<int>({0, 3, 3}),
                "heterogeneous budgets must give an equal per-layer quota");
        require(ExpertCachePolicy::quotas({0, 100, 200}, 299, 8) == std::vector<int>({0, 0, 0}),
                "budget smaller than one slot per layer");
        const std::vector<uint32_t> heat{0, 0, 0, 0, 100, 0, 0, 0, 25, 25, 25, 25};
        require(ExpertCachePolicy::heat_quotas({0, 100, 200}, 900, 4, heat) == std::vector<int>({0, 1, 4}),
                "marginal routing coverage must guide layer quotas");
        require(ExpertCachePolicy::heat_quotas({0, 100, 200}, 299, 4, heat) == std::vector<int>({0, 0, 0}),
                "heat quotas must preserve the minimum layer budget");
        require(ExpertCachePolicy::heat_quotas({0, 100, 200}, 900, 4, std::vector<uint32_t>(12)) ==
                    std::vector<int>({0, 3, 3}), "zero-heat ties must balance quotas deterministically");
        bool bad_heat = false;
        try { ExpertCachePolicy::heat_quotas({100}, 900, 4, {}); }
        catch (const std::invalid_argument&) { bad_heat = true; }
        require(bad_heat, "heat quotas reject invalid shapes");
        ExpertCachePolicy prior({1}, 4);
        prior.seed({4000, 1000, 0, 0});
        require(prior.heat() == std::vector<uint32_t>({32, 8, 0, 0}), "prefill prior must be scaled to at most 32");
        prior.bind(0, 0, 0);
        const int32_t new_hot[] = {2};
        for (int i = 0; i < 34; ++i) prior.observe(0, new_hot, 1);
        require(prior.candidate(0, 2, false, true) == 0, "decode must be able to replace a prompt-prior expert");
        prior.clear();
        prior.seed({0, 0, 0, 0});
        require(prior.heat() == std::vector<uint32_t>(4), "zero prior must not divide by zero");
        ExpertCachePolicy policy({0, 2, 1}, 4);
        const int32_t first[] = {0, 1};
        policy.observe(1, first, 2);
        policy.bind(1, policy.candidate(1, 0, false, true), 0);
        policy.bind(1, policy.candidate(1, 1, false, true), 1);
        require(policy.candidate(1, 2, false, true) < 0, "cold expert must not evict a hotter resident");
        const int32_t hot[] = {2};
        for (int i = 0; i < 3; ++i) policy.observe(1, hot, 1);
        const int victim = policy.candidate(1, 2, false, true);
        require(victim >= 0, "LFU requires a two-observation margin");
        const int old = policy.owner(victim);
        policy.bind(1, victim, 2);
        require(policy.resident(1, old) < 0 && policy.consistent(), "eviction must clear inverse mapping");
        require(policy.candidate(1, 3, false, false) < 0, "at most one swap per decode layer");
        for (int e = 0; e < 4; ++e) policy.bind(1, policy.candidate(1, e, true, true), e);
        require(policy.consistent(), "streamed prefill must leave a valid decode slot table");
        const int moved = policy.layer_begin(1);
        policy.bind(1, moved, policy.owner(policy.layer_end(1) - 1));
        require(policy.consistent(), "re-homing an expert must leave the previous slot vacant");
        require(policy.candidate(1, 0, false, false) >= 0, "vacant slots are reusable even after a swap");
        for (int i = 0; i < 8200; ++i) policy.observe(1, hot, 1);
        policy.bind(2, policy.candidate(2, 3, true, true), 3);
        require(policy.consistent() && policy.resident(2, 3) >= 0, "aging and layer isolation");
        bool rejected = false;
        try { policy.bind(1, policy.layer_begin(2), 0); }
        catch (const std::out_of_range&) { rejected = true; }
        require(rejected, "cross-layer slot assignment must fail");
        policy.clear();
        require(policy.consistent() && policy.resident(1, 2) < 0, "reset invalidates slot ownership");
        {
            std::vector<int> quotas(45, 6);
            quotas[0] = quotas[1] = quotas[2] = 0;
            ExpertCachePolicy routed(quotas, 288);
            uint32_t random = 17;
            for (int step = 0; step < 4096; ++step) {
                random = random * 1664525u + 1013904223u;
                const int layer = 3 + random % 42;
                random = random * 1664525u + 1013904223u;
                const int expert = random % 288;
                const int32_t ids[] = {expert};
                routed.observe(layer, ids, 1);
                const int slot = routed.candidate(layer, expert, true, true);
                routed.bind(layer, slot, expert);
                require(routed.consistent(layer) && routed.consistent(),
                        "layer validator disagreed with full-cache validation after replacement");
                if (step % 137 == 0) {
                    routed.clear();
                    require(routed.consistent(), "clear left inconsistent layer state");
                }
            }
            for (int layer = 0; layer < 45; ++layer)
                require(routed.consistent(layer), "layer validator must handle dense layers with no slots");
            for (int layer : {-1, 45}) {
                bool refused = false;
                try { routed.consistent(layer); } catch (const std::out_of_range&) { refused = true; }
                require(refused, "layer validator must reject invalid layer indices");
            }
        }

        const std::vector<int32_t> proposed{0, 1, 2};
        std::vector<std::vector<float>> logits{{0, 3, 0}, {0, 0, 3}, {3, 0, 0}};
        require(strata::glm5::verify_greedy(proposed, logits, {}).committed == 3, "all drafts accepted");
        logits[1] = {3, 0, 0};
        require(strata::glm5::verify_greedy(proposed, logits, {}).committed == 2, "rejection in the middle");
        logits[0] = {3, 0, 0};
        require(strata::glm5::verify_greedy(proposed, logits, {}).committed == 1, "first draft rejected");
        logits[0] = {0, 3, 0}; logits[1] = {0, 0, 3};
        const auto stopped = strata::glm5::verify_greedy(proposed, logits, {2});
        require(stopped.stopped && stopped.committed == 2, "EOS must not be emitted or committed");
        logits[1] = {3, 0, 0};
        const auto rejected_stop = strata::glm5::verify_greedy(proposed, logits, {2});
        require(!rejected_stop.stopped && rejected_stop.committed == 2, "rejected draft EOS must not end generation");
        require(strata::glm5::greedy({1, 1, 0}) == 0, "greedy ties use the smaller token id");
        rejected = false;
        try { strata::glm5::greedy({0, std::numeric_limits<float>::quiet_NaN()}); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "NaN logits must fail explicitly");
        strata::glm5::MemoryBudget budget(32);
        budget.acquire(12); budget.acquire(20); budget.release(20);
        rejected = false;
        try { budget.acquire(21); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected && budget.used() == 12 && budget.peak() == 32, "GPU budget accounting on failure");
        budget.release(12);
        require(budget.used() == 0, "bank release must return the budget");
        strata::glm5::Geometry geometry;
        geometry.n_layers = 4; geometry.n_head = 2; geometry.n_embd = 256; geometry.n_vocab = 32;
        geometry.n_expert = 4;
        strata::glm5::HostState state;
        state.position = 3; state.layers.resize(4);
        for (int l = 0; l < 3; ++l) state.layers[l].recurrent.resize(geometry.kda_state_floats());
        state.layers[3].kv.resize(3 * geometry.kv_lora_rank);
        state.mtp.kv.resize(3 * geometry.kv_lora_rank);
        state.carry_hidden.resize(geometry.n_embd); state.logits.resize(geometry.n_vocab);
        state.validate(geometry, 96, true);
        require(state.bytes() == strata::glm5::host_state_bytes(geometry, 3, true), "planned handoff byte count");
        state.expert_heat.assign(5 * geometry.n_expert, 0);
        state.expert_heat[3 * geometry.n_expert] = 100;
        state.validate(geometry, 96, true);
        require(state.bytes() == strata::glm5::host_state_bytes(geometry, 3, true, true), "heat handoff byte budget");
        state.expert_heat[0] = 1;
        rejected = false;
        try { state.validate(geometry, 96, true); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "dense layers must not have expert heat");
        state.expert_heat[0] = 0;
        state.expert_heat.pop_back();
        rejected = false;
        try { state.validate(geometry, 96, true); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "handoff must reject an incomplete heat table");
        state.expert_heat.push_back(0);
        auto sparse = state;
        sparse.dsa = true;
        sparse.dsa_tail = geometry.idx_select_tail;
        sparse.mtp = {};
        sparse.carry_hidden.clear();
        sparse.expert_heat.resize(4 * geometry.n_expert);
        sparse.layers[3].indexer.resize((2 * geometry.idx_kpool + sparse.position / geometry.idx_kpool) *
                                         geometry.idx_key_dim);
        sparse.validate(geometry, 96, false, true);
        require(sparse.bytes() == strata::glm5::host_state_bytes(geometry, 3, false, true, true),
                "DSA state budget");
        auto sparse_mtp = state;
        sparse_mtp.dsa = true;
        sparse_mtp.dsa_tail = geometry.idx_select_tail;
        sparse_mtp.layers[3].indexer = sparse.layers[3].indexer;
        sparse_mtp.mtp.indexer = sparse.layers[3].indexer;
        sparse_mtp.validate(geometry, 96, true, true);
        require(sparse_mtp.bytes() == strata::glm5::host_state_bytes(geometry, 3, true, true, true),
                "DSA MTP state budget");
        sparse_mtp.mtp.indexer.pop_back();
        rejected = false;
        try { sparse_mtp.validate(geometry, 96, true, true); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "DSA MTP must reject incomplete draft indexer state");
        sparse.dsa_tail = 1 - geometry.idx_select_tail;
        rejected = false;
        try { sparse.validate(geometry, 96, false, true); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "DSA tail policy cannot change across a handoff");
        sparse.dsa_tail = geometry.idx_select_tail;
        sparse.layers[3].indexer.pop_back();
        rejected = false;
        try { sparse.validate(geometry, 96, false, true); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "incomplete indexer handoff must fail");
        state.layers[3].kv.pop_back();
        rejected = false;
        try { state.validate(geometry, 96, true); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "handoff must reject an incomplete MLA cache");
        std::puts("glm5 cache and greedy verification policy passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm5 cache test: %s\n", e.what());
        return 1;
    }
}
