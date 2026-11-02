#include "strata/glm5/cache_policy.hpp"
#include "strata/glm5/speculative.hpp"

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
        std::puts("glm5 cache and greedy verification policy passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm5 cache test: %s\n", e.what());
        return 1;
    }
}
