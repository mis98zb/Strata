#include "strata/glm5/state.hpp"

namespace strata::glm5 {

uint64_t host_state_bytes(const Geometry& g, int64_t position, bool use_mtp) {
    if (position < 0) throw std::invalid_argument("glm5 handoff: negative position");
    uint64_t bytes = position ? checked_mul(g.n_vocab, sizeof(float)) : 0;
    for (int64_t l = 0; l < g.n_layers; ++l)
        bytes = checked_add(bytes, core::is_qsa_layer(g, l) ?
            checked_mul(position, g.kv_lora_rank * sizeof(uint16_t)) : g.kda_state_floats() * sizeof(float));
    if (use_mtp) {
        bytes = checked_add(bytes, checked_mul(position, g.kv_lora_rank * sizeof(uint16_t)));
        bytes = checked_add(bytes, checked_mul(g.n_embd, sizeof(float)));
    }
    return bytes;
}

uint64_t HostState::bytes() const {
    uint64_t bytes = mtp.bytes() + (carry_hidden.size() + logits.size()) * sizeof(float);
    for (const auto& layer : layers) bytes = checked_add(bytes, layer.bytes());
    return bytes;
}

void HostState::validate(const Geometry& g, int64_t context, bool use_mtp) const {
    if (position < 0 || position > context) throw std::runtime_error("glm5 handoff: invalid position");
    if (!position && layers.empty()) return;
    if (layers.size() != static_cast<size_t>(g.n_layers))
        throw std::runtime_error("glm5 handoff: wrong layer count");
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const auto& image = layers[l];
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
        carry_hidden.size() != static_cast<size_t>(g.n_embd)))
        throw std::runtime_error("glm5 handoff: incomplete MTP context");
}

}  // namespace strata::glm5
