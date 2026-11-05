#pragma once

#include "strata/glm5/glm_dsa.hpp"
#include "strata/glm5/model.hpp"

namespace strata::glm5 {

class AttentionScope {
public:
    explicit AttentionScope(const Options& options)
        : dsa_(kernels::glm_dsa_enabled()), flash_(kernels::glm_flash_enabled()),
          parallel_(kernels::glm_dsa_parallel_score()), tiled_(kernels::glm_dsa_tiled_score()),
          topk_(kernels::glm_dsa_parallel_select()) {
        kernels::glm_dsa_set_enabled(options.dsa);
        kernels::glm_flash_set_enabled(options.flash_attn);
        kernels::glm_dsa_set_parallel_score(options.dsa_score == DsaScore::Parallel || options.dsa_score == DsaScore::Tiled ||
            (options.dsa_score == DsaScore::Automatic && options.flash_attn));
        kernels::glm_dsa_set_tiled_score(options.dsa_score == DsaScore::Tiled);
        kernels::glm_dsa_set_parallel_select(options.dsa_parallel_topk);
    }
    ~AttentionScope() {
        kernels::glm_dsa_set_enabled(dsa_);
        kernels::glm_flash_set_enabled(flash_);
        kernels::glm_dsa_set_parallel_score(parallel_);
        kernels::glm_dsa_set_tiled_score(tiled_);
        kernels::glm_dsa_set_parallel_select(topk_);
    }
    AttentionScope(const AttentionScope&) = delete;
    AttentionScope& operator=(const AttentionScope&) = delete;
private:
    bool dsa_, flash_, parallel_, tiled_, topk_;
};

}  // namespace strata::glm5
