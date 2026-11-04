#pragma once

#include "strata/glm5/glm_dsa.hpp"
#include "strata/glm5/model.hpp"

namespace strata::glm5 {

class AttentionScope {
public:
    explicit AttentionScope(const Options& options)
        : dsa_(kernels::glm_dsa_enabled()), flash_(kernels::glm_flash_enabled()),
          parallel_(kernels::glm_dsa_parallel_score()) {
        kernels::glm_dsa_set_enabled(options.dsa);
        kernels::glm_flash_set_enabled(options.flash_attn);
        kernels::glm_dsa_set_parallel_score(options.dsa_score == DsaScore::Parallel ||
            (options.dsa_score == DsaScore::Automatic && options.flash_attn));
    }
    ~AttentionScope() {
        kernels::glm_dsa_set_enabled(dsa_);
        kernels::glm_flash_set_enabled(flash_);
        kernels::glm_dsa_set_parallel_score(parallel_);
    }
    AttentionScope(const AttentionScope&) = delete;
    AttentionScope& operator=(const AttentionScope&) = delete;
private:
    bool dsa_, flash_, parallel_;
};

}  // namespace strata::glm5
