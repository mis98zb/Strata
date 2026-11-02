#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <string>

namespace strata::glm5 {

struct Geometry : core::ModelGeometry {
    int64_t q_lora_rank = 1536;
    int64_t kv_lora_rank = 512;
    int64_t mla_head_dim = 256;
    int64_t kda_head_dim = 128;
    int64_t kda_conv_kernel = 4;
    double kda_gate_floor = -5.0;
    int64_t n_layer_dense_lead = 3;
    int64_t n_ff_dense = 12288;
    int64_t n_expert_shared = 1;
    int64_t n_nextn = 1;
    int64_t hc_mix = 24;
    int64_t idx_kpool = 4;
    int64_t idx_select_tail = 1;
    double expert_weights_scale = 2.5;
    double swiglu_clamp = 0.0;
    double swiglu_clamp_shexp = 0.0;

    Geometry();
    bool is_dense_ffn_layer(int64_t layer) const { return layer < n_layer_dense_lead; }
    float swiglu_limit_or_off() const { return static_cast<float>(swiglu_clamp); }
    float swiglu_limit_shexp_or_off() const { return static_cast<float>(swiglu_clamp_shexp); }
};

// Unlike the Qwen geometry reader, this never changes the shared engine's architecture guard.
bool read_geometry(const GgufFile& file, Geometry& out, std::string& err);
float swiglu(float gate, float up, float limit);

}  // namespace strata::glm5
