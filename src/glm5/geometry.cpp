#include "strata/glm5/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::glm5 {

Geometry::Geometry() {
    n_embd = 4096;
    n_layers = 45;
    n_vocab = 154880;
    n_head = 64;
    n_head_kv = 1;
    head_dim = 512;
    n_rot = 0;
    n_expert = 288;
    n_expert_used = 8;
    n_ff = 2048;
    n_ff_shexp = 2048;
    rms_eps = 1e-5;
}

bool read_geometry(const GgufFile& file, Geometry& out, std::string& err) {
    try {
        const auto* arch = file.get("general.architecture");
        if (!arch || arch->type != MetaType::STRING || arch->s != "glm5-next")
            throw std::runtime_error("expected general.architecture = glm5-next");
        Geometry g;
        const std::string p = "glm5-next.";
        auto integer = [&](const char* name, int64_t& value) {
            const auto* v = file.get(p + name);
            if (!v) return;
            if (!v->is_num() || v->type == MetaType::F32 || v->type == MetaType::F64 ||
                v->type == MetaType::BOOL || v->u > static_cast<uint64_t>(INT64_MAX))
                throw std::runtime_error(p + name + " must be a nonnegative integer");
            value = static_cast<int64_t>(v->u);
        };
        auto scalar = [&](const char* name, double& value) {
            const auto* v = file.get(p + name);
            if (!v) return;
            if (!v->is_num() || !std::isfinite(v->num()))
                throw std::runtime_error(p + name + " must be a finite number");
            value = v->num();
        };
        integer("embedding_length", g.n_embd);
        integer("vocab_size", g.n_vocab);
        if (const auto* v = file.get("tokenizer.ggml.tokens"); v && v->type == MetaType::ARRAY)
            g.n_vocab = static_cast<int64_t>(v->count);
        integer("attention.head_count", g.n_head);
        integer("q_lora_rank", g.q_lora_rank);
        integer("attention.q_lora_rank", g.q_lora_rank);
        integer("attention.kv_lora_rank", g.kv_lora_rank);
        integer("attention.key_length_mla", g.mla_head_dim);
        integer("attention.full_attention_interval", g.qsa_interval);
        integer("full_attention_interval", g.qsa_interval);
        integer("kda.head_dim", g.kda_head_dim);
        integer("ssm.conv_kernel", g.kda_conv_kernel);
        integer("leading_dense_block_count", g.n_layer_dense_lead);
        integer("feed_forward_length", g.n_ff_dense);
        integer("expert_count", g.n_expert);
        integer("expert_used_count", g.n_expert_used);
        integer("expert_feed_forward_length", g.n_ff);
        integer("expert_shared_count", g.n_expert_shared);
        integer("hyper_connection.count", g.hc);
        integer("rope.dimension_count", g.n_rot);
        integer("attention.indexer.head_count", g.idx_q_heads);
        integer("attention.indexer.key_length", g.idx_key_dim);
        integer("attention.indexer.top_k", g.idx_top_k);
        integer("attention.indexer.kpool", g.idx_kpool);
        scalar("attention.layer_norm_rms_epsilon", g.rms_eps);
        scalar("kda.gate_lower_bound", g.kda_gate_floor);
        scalar("expert_weights_scale", g.expert_weights_scale);
        integer("nextn_predict_layers", g.n_nextn);
        int64_t blocks = g.n_layers + g.n_nextn;
        integer("block_count", blocks);
        g.n_layers = blocks - g.n_nextn;
        g.head_dim = g.kv_lora_rank;
        g.n_ff_shexp = g.n_ff;
        auto clamp = [&](const char* name, double& dst) {
            const auto* v = file.get(p + name);
            if (!v) return;
            if (v->type != MetaType::ARRAY || v->count != static_cast<uint64_t>(blocks))
                throw std::runtime_error(p + name + " must have one value per block");
            for (const auto& item : v->items) {
                if (!item.is_num() || !std::isfinite(item.num()) || item.num() < 0 ||
                    item.num() != v->items.front().num())
                    throw std::runtime_error(p + name + " must be finite, nonnegative and uniform");
            }
            dst = v->items.front().num();
        };
        if (g.n_layers <= 0 || g.n_layers > 4096 || g.qsa_interval != 4 ||
            g.n_layer_dense_lead < 0 || g.n_layer_dense_lead > g.n_layers ||
            g.n_embd <= 0 || g.n_embd > 65536 || g.n_embd % 256 ||
            g.n_vocab <= 0 || g.n_vocab > INT32_MAX || g.n_expert <= 0 || g.n_expert > 4096 ||
            g.n_expert_used <= 0 || g.n_expert_used > std::min<int64_t>(15, g.n_expert) ||
            g.n_head <= 0 || g.n_head > 256 || g.n_head % 2 || g.hc != 4 || g.n_expert_shared != 1 ||
            g.n_rot != 0 || g.kda_head_dim != 128 || g.kda_conv_kernel != 4 ||
            g.q_lora_rank <= 0 || g.q_lora_rank > 65536 || g.q_lora_rank % 256 ||
            g.kv_lora_rank != 512 || g.mla_head_dim != 256 ||
            g.n_ff <= 0 || g.n_ff > 65536 || g.n_ff % 256 ||
            g.n_ff_dense <= 0 || g.n_ff_dense > 65536 || g.n_ff_dense % 256 ||
            g.idx_q_heads <= 0 || g.idx_q_heads > 256 || g.idx_key_dim != 128 ||
            g.idx_top_k != 2048 || g.idx_kpool != 4 ||
            g.rms_eps <= 0 || g.kda_gate_floor >= 0 || g.expert_weights_scale <= 0)
            throw std::runtime_error("unsupported glm5-next geometry (requires NoPE MLA, four mHC streams, "
                                     "128-wide KDA heads, one shared expert and whole quantization blocks)");
        if (const auto* v = file.get(p + "attention.head_count_kv")) {
            if (v->type != MetaType::ARRAY || v->count != static_cast<uint64_t>(blocks))
                throw std::runtime_error(p + "attention.head_count_kv must have one entry per block");
            for (int64_t l = 0; l < g.n_layers; ++l) {
                if (v->items[l].type == MetaType::ARRAY || !v->items[l].is_num() ||
                    v->items[l].num() != (core::is_qsa_layer(g, l) ? 1 : 0))
                    throw std::runtime_error(p + "attention.head_count_kv disagrees with KDA/MLA layer order");
            }
        }
        if (const auto* v = file.get(p + "attention.indexer.kpool_select_tail")) {
            if (!v->is_num() || (v->num() != 0 && v->num() != 1))
                throw std::runtime_error(p + "attention.indexer.kpool_select_tail must be boolean");
            g.idx_select_tail = static_cast<int64_t>(v->num());
        }
        clamp("swiglu_clamp_exp", g.swiglu_clamp);
        clamp("swiglu_clamp_shexp", g.swiglu_clamp_shexp);
        out = g;
        err.clear();
        return true;
    } catch (const std::exception& e) {
        err = std::string("glm5-next geometry: ") + e.what();
        return false;
    }
}

float swiglu(float gate, float up, float limit) {
    float activated = gate / (1.0f + std::exp(-gate));
    if (limit > 1e-6f) {
        activated = std::min(activated, limit);
        up = std::clamp(up, -limit, limit);
    }
    return activated * up;
}

}  // namespace strata::glm5
