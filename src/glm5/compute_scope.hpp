#pragma once

#include "attention_scope.hpp"
#include "gemv_util.hpp"
#include "strata/glm5/glm.hpp"

namespace strata::glm5 {

class ComputeScope {
public:
    explicit ComputeScope(const Options& options)
        : attention_(options), skip_(gemv::skip_unused_quant()), registers_(kernels::glm_kda_register_state()) {
        gemv::set_skip_unused_quant(options.skip_unused_quant);
        kernels::glm_kda_set_register_state(options.kda_register_state);
    }
    ~ComputeScope() {
        gemv::set_skip_unused_quant(skip_);
        kernels::glm_kda_set_register_state(registers_);
    }
    ComputeScope(const ComputeScope&) = delete;
    ComputeScope& operator=(const ComputeScope&) = delete;
private:
    AttentionScope attention_;
    bool skip_, registers_;
};

}  // namespace strata::glm5
