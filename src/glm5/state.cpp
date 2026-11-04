#include "strata/glm5/state.hpp"

namespace strata::glm5 {

uint64_t host_state_bytes(const Geometry& g, int64_t position, bool use_mtp, bool expert_heat, bool dsa) {
    if (position < 0) throw std::invalid_argument("glm5 handoff: negative position");
    uint64_t bytes = position ? checked_mul(g.n_vocab, sizeof(float)) : 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        bytes = checked_add(bytes, core::is_qsa_layer(g, l) ?
            checked_mul(position, g.kv_lora_rank * sizeof(uint16_t)) : g.kda_state_floats() * sizeof(float));
        if (dsa && core::is_qsa_layer(g, l))
            bytes = checked_add(bytes, checked_mul(2 * g.idx_kpool + position / g.idx_kpool,
                                                    g.idx_key_dim * sizeof(float)));
    }
    if (use_mtp) {
        bytes = checked_add(bytes, checked_mul(position, g.kv_lora_rank * sizeof(uint16_t)));
        if (dsa)
            bytes = checked_add(bytes, checked_mul(2 * g.idx_kpool + position / g.idx_kpool,
                                                    g.idx_key_dim * sizeof(float)));
        bytes = checked_add(bytes, checked_mul(g.n_embd, sizeof(float)));
    }
    if (expert_heat)
        bytes = checked_add(bytes, checked_mul(g.n_layers + (use_mtp ? 1 : 0), g.n_expert * sizeof(uint32_t)));
    return bytes;
}

uint64_t HostState::bytes() const {
    uint64_t bytes = mtp.bytes() + (carry_hidden.size() + logits.size()) * sizeof(float) +
                     expert_heat.size() * sizeof(uint32_t);
    for (const auto& layer : layers) bytes = checked_add(bytes, layer.bytes());
    return bytes;
}

void HostState::validate(const Geometry& g, int64_t context, bool use_mtp, bool use_dsa) const {
    if (position < 0 || position > context) throw std::runtime_error("glm5 handoff: invalid position");
    if (!expert_heat.empty()) {
        if (expert_heat.size() != static_cast<uint64_t>(g.n_layers + (use_mtp ? 1 : 0)) * g.n_expert)
            throw std::runtime_error("glm5 handoff: invalid expert heat shape");
        for (int64_t l = 0; l < g.n_layer_dense_lead; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e)
                if (expert_heat[l * g.n_expert + e])
                    throw std::runtime_error("glm5 handoff: expert heat on a dense layer");
    }
    if (!position && layers.empty()) return;
    if (dsa != use_dsa) throw std::runtime_error("glm5 handoff: attention mode mismatch");
    if (use_dsa && dsa_tail != g.idx_select_tail)
        throw std::runtime_error("glm5 handoff: DSA tail policy mismatch");
    if (layers.size() != static_cast<size_t>(g.n_layers))
        throw std::runtime_error("glm5 handoff: wrong layer count");
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const auto& image = layers[l];
        const auto indexer_size = use_dsa && core::is_qsa_layer(g, l) ?
            (2 * g.idx_kpool + position / g.idx_kpool) * g.idx_key_dim : 0;
        if (image.indexer.size() != static_cast<uint64_t>(indexer_size))
            throw std::runtime_error("glm5 handoff: incomplete indexer state");
        if (!core::is_qsa_layer(g, l)) {
            if (!image.kv.empty() || image.recurrent.size() != g.kda_state_floats())
                throw std::runtime_error("glm5 handoff: invalid KDA state");
        } else if (!image.recurrent.empty() || image.kv.size() != static_cast<uint64_t>(position) * g.kv_lora_rank)
            throw std::runtime_error("glm5 handoff: invalid MLA cache");
    }
    if (position && logits.size() != static_cast<size_t>(g.n_vocab))
        throw std::runtime_error("glm5 handoff: missing target logits");
    if (use_mtp && (!mtp.recurrent.empty() ||
        mtp.kv.size() != static_cast<uint64_t>(position) * g.kv_lora_rank ||
        mtp.indexer.size() != static_cast<uint64_t>(use_dsa ?
            (2 * g.idx_kpool + position / g.idx_kpool) * g.idx_key_dim : 0) ||
        carry_hidden.size() != static_cast<size_t>(g.n_embd)))
        throw std::runtime_error("glm5 handoff: incomplete MTP context");
}

}  // namespace strata::glm5
