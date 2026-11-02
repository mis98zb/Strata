#pragma once

#include "gguf_fixture.hpp"

namespace glm_fixture {

inline std::vector<fixture::Kv> metadata() {
    return {fixture::str("general.architecture", "glm5-next"),
            fixture::u32("glm5-next.embedding_length", 256),
            fixture::u32("glm5-next.vocab_size", 32),
            fixture::u32("glm5-next.block_count", 5),
            fixture::u32("glm5-next.attention.head_count", 2),
            fixture::u32("glm5-next.attention.q_lora_rank", 256),
            fixture::u32("glm5-next.feed_forward_length", 256),
            fixture::u32("glm5-next.expert_feed_forward_length", 256),
            fixture::u32("glm5-next.expert_count", 2),
            fixture::u32("glm5-next.expert_used_count", 2),
            fixture::u32s("glm5-next.attention.head_count_kv", {0, 0, 0, 1, 1}),
            fixture::u32s("glm5-next.swiglu_clamp_exp", {10, 10, 10, 10, 10}),
            fixture::u32s("glm5-next.swiglu_clamp_shexp", {1, 1, 1, 1, 1}),
            fixture::u32("tokenizer.ggml.eos_token_id", 30),
            fixture::u32("tokenizer.ggml.eot_token_id", 31)};
}

inline std::vector<fixture::Tensor> tensors(bool mtp = false, uint64_t expert_width = 256) {
    std::vector<fixture::Tensor> rows;
    auto add = [&](std::string name, std::vector<uint64_t> shape) {
        rows.push_back({std::move(name), std::move(shape), 0, 1});
    };
    add("token_embd.weight", {256, 32});
    add("output.weight", {256, 32});
    add("output_norm.weight", {256});
    for (int l = 0; l < (mtp ? 5 : 4); ++l) {
        const auto p = "blk." + std::to_string(l) + ".";
        for (const char* half : {"attn", "ffn"}) {
            if (l < 4) {
                add(p + "hc_" + half + "_fn.weight", {1024, 24});
                add(p + "hc_" + half + "_base.weight", {24});
                add(p + "hc_" + half + "_scale.weight", {3});
            }
            add(p + half + "_norm.weight", {256});
        }
        if (l >= 3) {
            add(p + "attn_q_a.weight", {256, 256});
            add(p + "attn_q_a_norm.weight", {256});
            add(p + "attn_q_b.weight", {256, 512});
            add(p + "attn_k_b.weight", {256, 512, 2});
            add(p + "attn_kv_a_mqa.weight", {256, 512});
            add(p + "attn_kv_a_norm.weight", {512});
            add(p + "attn_v_b.weight", {512, 256, 2});
            add(p + "attn_output.weight", {512, 256});
        } else {
            for (const char* role : {"q", "k", "v"}) {
                add(p + "attn_" + role + ".weight", {256, 256});
                add(p + "ssm_conv1d_" + role + ".weight", {4, 256});
            }
            add(p + "attn_output.weight", {256, 256});
            add(p + "ssm_beta.weight", {256, 2});
            for (const char* role : {"f", "g"}) {
                add(p + "ssm_" + role + "_a.weight", {256, 128});
                add(p + "ssm_" + role + "_b.weight", {128, 256});
            }
            add(p + "ssm_a", {2});
            add(p + "ssm_dt.bias", {256});
            add(p + "ssm_norm.weight", {128});
        }
        for (const char* role : {"gate", "up", "down"}) {
            if (l < 3) add(p + "ffn_" + role + ".weight", {256, 256});
            else {
                const bool down = std::string(role) == "down";
                const uint64_t width = down ? expert_width : 256, rows = down ? 256 : expert_width;
                add(p + "ffn_" + role + "_exps.weight", {width, rows, 2});
                add(p + "ffn_" + role + "_shexp.weight", {width, rows});
            }
        }
        if (l >= 3) {
            add(p + "ffn_gate_inp.weight", {256, 2});
            add(p + "exp_probs_b.bias", {2});
        }
        if (mtp && l == 4) {
            add("blk.4.nextn.enorm.weight", {256});
            add("blk.4.nextn.hnorm.weight", {256});
            add("blk.4.nextn.eh_proj.weight", {512, 256});
        }
    }
    return rows;
}

}  // namespace glm_fixture
