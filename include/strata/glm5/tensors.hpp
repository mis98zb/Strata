#pragma once

#include "strata/core/weights.hpp"

#include <map>
#include <string>
#include <utility>

namespace strata::glm5 {

using core::WeightRef;

class TensorTable {
public:
    bool insert(std::string name, WeightRef ref) { return rows_.emplace(std::move(name), ref).second; }
    const WeightRef* find(const std::string& name) const {
        const auto it = rows_.find(name);
        return it == rows_.end() ? nullptr : &it->second;
    }
private:
    std::map<std::string, WeightRef> rows_;
};

class LayerView {
public:
    LayerView(const TensorTable& table, int64_t layer) : table_(table), layer_(layer) {}
    std::string name(const char* suffix) const {
        return "blk." + std::to_string(layer_) + "." + suffix;
    }
    const WeightRef* get(const char* suffix) const { return table_.find(name(suffix)); }
private:
    const TensorTable& table_;
    int64_t layer_;
};

}  // namespace strata::glm5
