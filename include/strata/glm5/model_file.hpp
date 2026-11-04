#pragma once

#include "strata/glm5/geometry.hpp"

#include <memory>
#include <set>

namespace strata::glm5 {

class ModelFile {
public:
    explicit ModelFile(const std::string& path, int dsa_tail = -1);
    const Geometry& geometry() const { return geometry_; }
    const GgufModel& gguf() const { return *gguf_; }
    const std::set<int32_t>& stop_ids() const { return stop_ids_; }
    const TensorInfo& tensor(const std::string& name, size_t* shard = nullptr) const;
    const uint8_t* data(const std::string& name) const;
    void validate() const;
    void validate_mtp() const;
    void validate_dsa(bool use_mtp = false) const;
private:
    std::unique_ptr<GgufModel> gguf_;
    Geometry geometry_;
    std::set<int32_t> stop_ids_;
};

}  // namespace strata::glm5
