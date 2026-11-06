#include "strata/glm5/cache_policy.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace strata::glm5 {

std::vector<int> ExpertCachePolicy::quotas(const std::vector<uint64_t>& bytes, uint64_t budget, int experts) {
    if (experts < 1) throw std::invalid_argument("glm5 cache: expert count must be positive");
    uint64_t one_each = 0;
    for (const auto size : bytes) {
        if (size > UINT64_MAX - one_each) throw std::overflow_error("glm5 cache: layer sizes overflow");
        one_each += size;
    }
    const auto each = one_each ? std::min<uint64_t>(experts, budget / one_each) : 0;
    std::vector<int> result;
    for (const auto size : bytes) result.push_back(size ? static_cast<int>(each) : 0);
    return result;
}

ExpertCachePolicy::ExpertCachePolicy(std::vector<int> quotas, int experts) : experts_(experts) {
    if (experts < 1 || quotas.empty()) throw std::invalid_argument("glm5 cache: invalid geometry");
    int total = 0;
    for (const int q : quotas) {
        if (q < 0 || q > experts || total > INT32_MAX - q)
            throw std::invalid_argument("glm5 cache: invalid layer quota");
        begin_.push_back(total);
        total += q;
        end_.push_back(total);
    }
    owners_.assign(total, -1);
    resident_.assign(quotas.size() * static_cast<size_t>(experts), -1);
    counts_.assign(resident_.size(), 0);
    epochs_.assign(quotas.size(), 0);
}

std::vector<int> ExpertCachePolicy::heat_quotas(const std::vector<uint64_t>& bytes, uint64_t budget, int experts,
                                               const std::vector<uint32_t>& heat) {
    if (experts < 1 || bytes.size() > SIZE_MAX / static_cast<size_t>(experts) ||
        heat.size() != bytes.size() * static_cast<size_t>(experts))
        throw std::invalid_argument("glm5 cache: invalid heat geometry");
    const auto minimum = quotas(bytes, budget, 1);
    if (std::none_of(minimum.begin(), minimum.end(), [](int n) { return n != 0; })) return minimum;
    auto result = minimum;
    std::vector<std::vector<uint32_t>> ranked(bytes.size());
    std::vector<uint64_t> totals(bytes.size(), 0);
    uint64_t remaining = budget;
    for (size_t l = 0; l < bytes.size(); ++l) {
        if (!bytes[l]) continue;
        remaining -= bytes[l];
        ranked[l].assign(heat.begin() + l * experts, heat.begin() + (l + 1) * experts);
        std::sort(ranked[l].rbegin(), ranked[l].rend());
        totals[l] = std::accumulate(ranked[l].begin(), ranked[l].end(), uint64_t(0));
    }
    for (;;) {
        size_t best = bytes.size();
        long double best_gain = -1;
        for (size_t l = 0; l < bytes.size(); ++l) {
            if (!bytes[l] || result[l] >= experts || bytes[l] > remaining) continue;
            const long double gain = totals[l] ? static_cast<long double>(ranked[l][result[l]]) / totals[l] : 0;
            if (gain > best_gain || (gain == best_gain && best < bytes.size() && result[l] < result[best])) {
                best = l;
                best_gain = gain;
            }
        }
        if (best == bytes.size()) break;
        ++result[best];
        remaining -= bytes[best];
    }
    return result;
}

void ExpertCachePolicy::seed(const std::vector<uint32_t>& heat) {
    if (heat.size() != counts_.size()) throw std::invalid_argument("glm5 cache: invalid prior shape");
    for (size_t l = 0; l < begin_.size(); ++l) {
        const auto first = heat.begin() + l * experts_;
        const auto maximum = *std::max_element(first, first + experts_);
        for (int e = 0; e < experts_; ++e) {
            const auto value = heat[l * experts_ + e];
            counts_[l * experts_ + e] = value ? static_cast<uint32_t>(
                std::max<uint64_t>(1, static_cast<uint64_t>(value) * 32 / maximum)) : 0;
        }
    }
    std::fill(epochs_.begin(), epochs_.end(), 0);
}

size_t ExpertCachePolicy::index(int layer, int expert) const {
    if (layer < 0 || static_cast<size_t>(layer) >= begin_.size() || expert < 0 || expert >= experts_)
        throw std::out_of_range("glm5 cache: layer or expert out of range");
    return static_cast<size_t>(layer) * experts_ + expert;
}
int ExpertCachePolicy::resident(int layer, int expert) const { return resident_[index(layer, expert)]; }
int ExpertCachePolicy::layer_begin(int layer) const { index(layer, 0); return begin_[layer]; }
int ExpertCachePolicy::layer_end(int layer) const { index(layer, 0); return end_[layer]; }
int ExpertCachePolicy::owner(int slot) const { return owners_.at(slot); }

void ExpertCachePolicy::observe(int layer, const int32_t* ids, int count) {
    index(layer, 0);
    if (!ids || count < 1) throw std::invalid_argument("glm5 cache: empty routing observation");
    for (int i = 0; i < count; ++i) index(layer, ids[i]);
    if (++epochs_[layer] % 4096 == 0) {
        for (int e = 0; e < experts_; ++e) counts_[index(layer, e)] /= 2;
    }
    for (int i = 0; i < count; ++i) {
        auto& value = counts_[index(layer, ids[i])];
        if (value != UINT32_MAX) ++value;
    }
}

int ExpertCachePolicy::candidate(int layer, int expert, bool force, bool allow_swap) const {
    const size_t at = index(layer, expert);
    if (resident_[at] >= 0) return resident_[at];
    int victim = -1;
    for (int s = begin_[layer]; s < end_[layer]; ++s) {
        if (owners_[s] < 0) return s;
        if (victim < 0 || counts_[index(layer, owners_[s])] < counts_[index(layer, owners_[victim])])
            victim = s;
    }
    if (victim < 0 || (!force && !allow_swap)) return -1;
    if (!force && static_cast<uint64_t>(counts_[at]) <
                      static_cast<uint64_t>(counts_[index(layer, owners_[victim])]) + 2) return -1;
    return victim;
}

void ExpertCachePolicy::bind(int layer, int slot, int expert) {
    const auto at = index(layer, expert);
    if (slot < begin_[layer] || slot >= end_[layer]) throw std::out_of_range("glm5 cache: slot in wrong layer");
    const int previous = resident_[at];
    if (previous >= 0 && previous != slot) owners_[previous] = -1;
    if (owners_[slot] >= 0) resident_[index(layer, owners_[slot])] = -1;
    owners_[slot] = expert;
    resident_[at] = slot;
}

void ExpertCachePolicy::clear() {
    std::fill(resident_.begin(), resident_.end(), -1);
    std::fill(owners_.begin(), owners_.end(), -1);
    std::fill(counts_.begin(), counts_.end(), 0);
    std::fill(epochs_.begin(), epochs_.end(), 0);
}

bool ExpertCachePolicy::consistent() const {
    for (size_t l = 0; l < begin_.size(); ++l)
        if (!consistent(static_cast<int>(l))) return false;
    return true;
}

bool ExpertCachePolicy::consistent(int layer) const {
    const auto first = index(layer, 0);
    for (int s = begin_[layer]; s < end_[layer]; ++s) {
        const int e = owners_[s];
        if (e >= 0 && (e >= experts_ || resident_[first + e] != s)) return false;
    }
    for (int e = 0; e < experts_; ++e) {
        const int s = resident_[first + e];
        if (s >= 0 && (s < begin_[layer] || s >= end_[layer] || owners_[s] != e)) return false;
    }
    return true;
}

}  // namespace strata::glm5
