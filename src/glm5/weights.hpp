#pragma once

#include "expert_compute.hpp"
#include "strata/glm5/model_file.hpp"
#include "strata/glm5/tensors.hpp"

#include <functional>

namespace strata::glm5 {

class WeightBank {
public:
    using Select = std::function<bool(const TensorInfo&)>;
    WeightBank(const ModelFile& file, const Select& select, MemoryBudget* budget = nullptr);
    const TensorTable& table() const { return table_; }
    ExpertWeights expert(int layer, int expert) const;
    uint64_t bytes() const { return bytes_; }
private:
    const Geometry& geometry_;
    std::unique_ptr<Buffer> arena_;
    TensorTable table_;
    uint64_t bytes_ = 0;
};

bool decode_tensor(const TensorInfo& tensor, int64_t layers, bool dsa = false);
void project_head(const TensorTable& tensors, const Geometry& g, const float* hidden, float* destination,
                   bool draft, void* stream);

}  // namespace strata::glm5
