#include "strata/glm5/model_file.hpp"

#include <stdexcept>

namespace strata::glm5 {

ModelFile::ModelFile(const std::string& path)
    : gguf_(std::make_unique<GgufModel>(gguf_split_paths(path))) {
    std::string err;
    if (!read_geometry(gguf_->meta(), geometry_, err)) throw std::runtime_error(err);
    for (const char* name : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                             "tokenizer.ggml.eom_token_id"}) {
        if (const auto* v = gguf_->meta().get(name)) {
            if (!v->is_num() || v->num() < 0 || v->num() >= geometry_.n_vocab ||
                v->num() != static_cast<int64_t>(v->num()))
                throw std::runtime_error(std::string(name) + ": invalid stop token");
            stop_ids_.insert(static_cast<int32_t>(v->num()));
        }
    }
}

const TensorInfo& ModelFile::tensor(const std::string& name, size_t* shard) const {
    size_t at = 0;
    const auto* t = gguf_->find(name, &at);
    if (!t) throw std::runtime_error("glm5-next: missing tensor " + name);
    if (!gguf_->in_bounds(*t, at)) throw std::runtime_error("glm5-next: invalid or truncated tensor " + name);
    if (shard) *shard = at;
    return *t;
}

const uint8_t* ModelFile::data(const std::string& name) const {
    size_t shard = 0;
    const auto& t = tensor(name, &shard);
    return gguf_->shard(shard).tensor_data(t);
}

void ModelFile::validate() const {
    const auto& g = geometry_;
    auto shape = [&](const std::string& name, uint64_t width, uint64_t rows) {
        const auto& t = tensor(name);
        if (!tensor_payload_bytes(t) || t.shape.front() != width || t.elements() / width != rows)
            throw std::runtime_error(name + ": expected " + std::to_string(width) + " x " +
                                     std::to_string(rows) + " elements in GGUF row order");
    };
    auto values = [&](const std::string& name, uint64_t count) {
        const auto& t = tensor(name);
        if (!tensor_payload_bytes(t) || t.elements() != count)
            throw std::runtime_error(name + ": expected " + std::to_string(count) + " values");
    };
    shape("token_embd.weight", g.n_embd, g.n_vocab);
    shape("output.weight", g.n_embd, g.n_vocab);
    values("output_norm.weight", g.n_embd);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const auto p = "blk." + std::to_string(l) + ".";
        for (const char* half : {"attn", "ffn"}) {
            const auto hc = p + "hc_" + half;
            shape(hc + "_fn.weight", g.hc * g.n_embd, g.hc_mix);
            values(hc + "_base.weight", g.hc_mix);
            values(hc + "_scale.weight", 3);
            values(p + half + "_norm.weight", g.n_embd);
        }
        if (core::is_qsa_layer(g, l)) {
            shape(p + "attn_q_a.weight", g.n_embd, g.q_lora_rank);
            values(p + "attn_q_a_norm.weight", g.q_lora_rank);
            shape(p + "attn_q_b.weight", g.q_lora_rank, g.n_head * g.mla_head_dim);
            shape(p + "attn_k_b.weight", g.mla_head_dim, g.kv_lora_rank * g.n_head);
            shape(p + "attn_kv_a_mqa.weight", g.n_embd, g.kv_lora_rank);
            values(p + "attn_kv_a_norm.weight", g.kv_lora_rank);
            shape(p + "attn_v_b.weight", g.kv_lora_rank, g.mla_head_dim * g.n_head);
            shape(p + "attn_output.weight", g.n_head * g.mla_head_dim, g.n_embd);
        } else {
            const int64_t nv = g.n_head * g.kda_head_dim;
            for (const char* role : {"q", "k", "v"}) {
                shape(p + "attn_" + role + ".weight", g.n_embd, nv);
                values(p + "ssm_conv1d_" + role + ".weight", g.kda_conv_kernel * nv);
            }
            shape(p + "attn_output.weight", nv, g.n_embd);
            shape(p + "ssm_beta.weight", g.n_embd, g.n_head);
            for (const char* role : {"f", "g"}) {
                shape(p + "ssm_" + role + "_a.weight", g.n_embd, 128);
                shape(p + "ssm_" + role + "_b.weight", 128, nv);
            }
            values(p + "ssm_a", g.n_head);
            values(p + "ssm_dt.bias", nv);
            values(p + "ssm_norm.weight", g.kda_head_dim);
        }
        if (g.is_dense_ffn_layer(l)) {
            shape(p + "ffn_gate.weight", g.n_embd, g.n_ff_dense);
            shape(p + "ffn_up.weight", g.n_embd, g.n_ff_dense);
            shape(p + "ffn_down.weight", g.n_ff_dense, g.n_embd);
        } else {
            shape(p + "ffn_gate_inp.weight", g.n_embd, g.n_expert);
            if (gguf_->find(p + "exp_probs_b.bias")) values(p + "exp_probs_b.bias", g.n_expert);
            for (const char* role : {"gate", "up"}) {
                const auto& t = tensor(p + "ffn_" + role + "_exps.weight");
                if (t.shape != std::vector<uint64_t>{static_cast<uint64_t>(g.n_embd),
                        static_cast<uint64_t>(g.n_ff), static_cast<uint64_t>(g.n_expert)})
                    throw std::runtime_error(t.name + ": invalid expert tensor shape");
                shape(p + "ffn_" + role + "_shexp.weight", g.n_embd, g.n_ff);
            }
            const auto& down = tensor(p + "ffn_down_exps.weight");
            if (down.shape != std::vector<uint64_t>{static_cast<uint64_t>(g.n_ff),
                    static_cast<uint64_t>(g.n_embd), static_cast<uint64_t>(g.n_expert)})
                throw std::runtime_error(down.name + ": invalid expert tensor shape");
            shape(p + "ffn_down_shexp.weight", g.n_ff, g.n_embd);
        }
    }
}

void ModelFile::validate_mtp() const {
    const auto& g = geometry_;
    if (g.n_nextn != 1) throw std::runtime_error("glm5 MTP requires exactly one NextN block");
    const auto p = "blk." + std::to_string(g.n_layers) + ".";
    auto shape = [&](const std::string& name, uint64_t width, uint64_t rows) {
        const auto& t = tensor(name);
        if (t.shape.front() != width || t.elements() / width != rows)
            throw std::runtime_error(name + ": invalid MTP tensor shape");
    };
    auto values = [&](const std::string& name, uint64_t count) {
        if (tensor(name).elements() != count) throw std::runtime_error(name + ": invalid MTP norm shape");
    };
    shape(p + "nextn.eh_proj.weight", 2 * g.n_embd, g.n_embd);
    for (const char* name : {"nextn.enorm.weight", "nextn.hnorm.weight", "attn_norm.weight", "ffn_norm.weight"})
        values(p + name, g.n_embd);
    if (gguf_->find(p + "nextn.shared_head_norm.weight")) values(p + "nextn.shared_head_norm.weight", g.n_embd);
    for (const char* name : {"nextn.embed_tokens.weight", "nextn.shared_head_head.weight"})
        if (gguf_->find(p + name)) shape(p + name, g.n_embd, g.n_vocab);
    shape(p + "attn_q_a.weight", g.n_embd, g.q_lora_rank);
    values(p + "attn_q_a_norm.weight", g.q_lora_rank);
    shape(p + "attn_q_b.weight", g.q_lora_rank, g.n_head * g.mla_head_dim);
    shape(p + "attn_k_b.weight", g.mla_head_dim, g.kv_lora_rank * g.n_head);
    shape(p + "attn_kv_a_mqa.weight", g.n_embd, g.kv_lora_rank);
    values(p + "attn_kv_a_norm.weight", g.kv_lora_rank);
    shape(p + "attn_v_b.weight", g.kv_lora_rank, g.mla_head_dim * g.n_head);
    shape(p + "attn_output.weight", g.n_head * g.mla_head_dim, g.n_embd);
    shape(p + "ffn_gate_inp.weight", g.n_embd, g.n_expert);
    if (gguf_->find(p + "exp_probs_b.bias")) values(p + "exp_probs_b.bias", g.n_expert);
    for (const char* role : {"gate", "up"}) {
        const auto& t = tensor(p + "ffn_" + role + "_exps.weight");
        if (t.shape != std::vector<uint64_t>{static_cast<uint64_t>(g.n_embd),
                static_cast<uint64_t>(g.n_ff), static_cast<uint64_t>(g.n_expert)})
            throw std::runtime_error(t.name + ": invalid MTP expert shape");
        shape(p + "ffn_" + role + "_shexp.weight", g.n_embd, g.n_ff);
    }
    const auto& down = tensor(p + "ffn_down_exps.weight");
    if (down.shape != std::vector<uint64_t>{static_cast<uint64_t>(g.n_ff),
            static_cast<uint64_t>(g.n_embd), static_cast<uint64_t>(g.n_expert)})
        throw std::runtime_error(down.name + ": invalid MTP expert shape");
    shape(p + "ffn_down_shexp.weight", g.n_ff, g.n_embd);
}

}  // namespace strata::glm5
