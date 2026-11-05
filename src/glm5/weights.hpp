#pragma once

#include "expert_compute.hpp"
#include "strata/glm5/model_file.hpp"
#include "strata/glm5/tensors.hpp"

#include <functional>
#include <future>

namespace strata::glm5 {

class WeightBank {
public:
    using Select = std::function<bool(const TensorInfo&)>;
    WeightBank(const ModelFile& file, const Select& select, MemoryBudget* budget = nullptr,
               uint64_t staging_bytes = 0, bool defer_upload = false);
    ~WeightBank();
    void wait_ready();
    void start_upload();
    static uint64_t allocation_bytes(const ModelFile& file, const Select& select);
    const TensorTable& table() const { return table_; }
    ExpertWeights expert(int layer, int expert) const;
    uint64_t bytes() const { return bytes_; }
    uint64_t uploaded_bytes() const { return uploaded_bytes_; }
    uint64_t upload_calls() const { return upload_calls_; }
private:
    const Geometry& geometry_;
    std::unique_ptr<Buffer> arena_;
    TensorTable table_;
    uint64_t bytes_ = 0;
    uint64_t uploaded_bytes_ = 0;
    uint64_t upload_calls_ = 0;
    std::future<void> upload_;
    std::function<void()> pending_upload_;
};

bool decode_tensor(const TensorInfo& tensor, int64_t layers, bool dsa = false);
void project_head(const TensorTable& tensors, const Geometry& g, const float* hidden, float* destination,
                   bool draft, void* stream);

}  // namespace strata::glm5
