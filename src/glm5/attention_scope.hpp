#pragma once

#include "strata/glm5/glm_dsa.hpp"
#include "strata/glm5/model.hpp"
#include <stdexcept>

namespace strata::glm5 {

class AttentionScope {
public:
    explicit AttentionScope(const Options& options)
        : dsa_(kernels::glm_dsa_enabled()), flash_(kernels::glm_flash_enabled()),
          parallel_(kernels::glm_dsa_parallel_score()), tiled_(kernels::glm_dsa_tiled_score()),
          topk_(kernels::glm_dsa_parallel_select()), head_major_(kernels::glm_flash_head_major()),
          prefetch_(kernels::glm_flash_prefetch()) {
        if (options.mla_attention_layout != MlaAttentionLayout::Automatic &&
            options.mla_attention_layout != MlaAttentionLayout::TokenMajor &&
            options.mla_attention_layout != MlaAttentionLayout::HeadMajor)
            throw std::invalid_argument("glm5: invalid scoped MLA attention layout");
        if (options.flash_kv_load != FlashKvLoad::Automatic && options.flash_kv_load != FlashKvLoad::Direct &&
            options.flash_kv_load != FlashKvLoad::Prefetch)
            throw std::invalid_argument("glm5: invalid scoped Flash KV load mode");
        kernels::glm_dsa_set_enabled(options.dsa);
        kernels::glm_flash_set_enabled(options.flash_attn);
        kernels::glm_flash_set_head_major(options.mla_attention_layout == MlaAttentionLayout::HeadMajor);
        kernels::glm_flash_set_prefetch(options.flash_attn && options.flash_kv_load == FlashKvLoad::Prefetch);
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
        kernels::glm_flash_set_head_major(head_major_);
        kernels::glm_flash_set_prefetch(prefetch_);
    }
    AttentionScope(const AttentionScope&) = delete;
    AttentionScope& operator=(const AttentionScope&) = delete;
private:
    bool dsa_, flash_, parallel_, tiled_, topk_, head_major_, prefetch_;
};

}  // namespace strata::glm5
