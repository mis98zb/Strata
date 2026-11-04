#include "weights.hpp"
#include "cpu.hpp"
#include "gemv_util.hpp"
#include "strata/glm5/glm.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <climits>

namespace strata::glm5 {
namespace {

bool scalar_tensor(const std::string& name) {
    return name.find("norm") != std::string::npos || name.find("ssm_conv1d") != std::string::npos ||
           name.ends_with("ssm_a") || name.ends_with(".bias") ||
           name.find("_base.") != std::string::npos || name.find("_scale.") != std::string::npos ||
           name.find("indexer_compressor_ape") != std::string::npos;
}
uint64_t align256(uint64_t bytes) { return checked_add(bytes, 255) & ~uint64_t(255); }

}

bool decode_tensor(const TensorInfo& t, int64_t layers, bool dsa) {
    if (t.name == "token_embd.weight" || t.name.ends_with(".nextn.embed_tokens.weight") ||
        t.name.find("_exps.weight") != std::string::npos || t.name.starts_with("nextn.") ||
        (!dsa && t.name.find("indexer") != std::string::npos)) return false;
    if (t.name.starts_with("blk.")) {
        size_t end = 0;
        const auto layer = std::stoll(t.name.substr(4), &end);
        if (!end || 4 + end >= t.name.size() || t.name[4 + end] != '.' || layer < 0)
            throw std::runtime_error("glm5: invalid layer name " + t.name);
        if (layer >= layers) return false;
    }
    return true;
}

WeightBank::WeightBank(const ModelFile& file, const Select& select, MemoryBudget* budget) : geometry_(file.geometry()) {
    init_cpu();
    struct Row { const TensorInfo* tensor; size_t shard; uint64_t bytes; bool native; };
    std::vector<Row> rows;
    int64_t widest = 256;
    const auto& model = file.gguf();
    for (size_t s = 0; s < model.size(); ++s) {
        for (const auto& t : model.shard(s).tensors()) {
            if (!select(t)) continue;
            if (!model.in_bounds(t, s) || t.shape.front() > INT_MAX || t.elements() > INT64_MAX)
                throw std::runtime_error(t.name + ": invalid tensor payload or dimensions");
            const bool native = !scalar_tensor(t.name) && kernels::native_mmvq_supported(t.type);
            if (!native && !scalar_tensor(t.name) && t.type != GGML_TYPE_F32 &&
                t.type != GGML_TYPE_F16 && t.type != GGML_TYPE_BF16)
                throw std::runtime_error(t.name + ": unsupported native GPU matrix format");
            if (t.name.find("_exps.weight") != std::string::npos && !native)
                throw std::runtime_error(t.name + ": full-layer experts require native quantized matrices");
            const auto bytes = native ? tensor_payload_bytes(t) : checked_mul(t.elements(), sizeof(float));
            bytes_ = checked_add(bytes_, align256(bytes));
            widest = std::max(widest, static_cast<int64_t>(t.shape.front()));
            rows.push_back({&t, s, bytes, native});
        }
    }
    const auto scratch_bytes = kernels::native_q8_1_bytes(widest, 8);
    bytes_ = checked_add(bytes_, align256(scratch_bytes));
    arena_ = std::make_unique<Buffer>(bytes_, budget);
    auto* base = static_cast<uint8_t*>(arena_->get());
    void* scratch = base + bytes_ - align256(scratch_bytes);
    for (const auto& row : rows) {
        const auto& t = *row.tensor;
        WeightRef ref;
        ref.ne0 = t.shape.front(); ref.elements = t.elements(); ref.ne1 = ref.elements / ref.ne0;
        if (row.native) {
            check(cudaMemcpy(base, model.shard(row.shard).tensor_data(t), row.bytes, cudaMemcpyHostToDevice),
                  "glm5 native weight upload");
            ref.native_data = base; ref.native_type = t.type; ref.native_q8_1 = scratch; ref.resident = false;
        } else {
            if (t.type == GGML_TYPE_F32) {
                check(cudaMemcpy(base, model.shard(row.shard).tensor_data(t), row.bytes, cudaMemcpyHostToDevice),
                      "glm5 F32 weight upload");
            } else {
                std::vector<float> values(ref.elements);
                dequantize(t, model.shard(row.shard).tensor_data(t), values.data(), ref.elements);
                check(cudaMemcpy(base, values.data(), row.bytes, cudaMemcpyHostToDevice), "glm5 scalar weight upload");
            }
            ref.bytes = row.bytes; ref.data = base; ref.kind = core::WeightKind::F32;
        }
        if (!table_.insert(t.name, ref)) throw std::runtime_error("glm5: duplicate tensor " + t.name);
        base += align256(row.bytes);
    }
}

ExpertWeights WeightBank::expert(int layer, int expert) const {
    if (expert < 0 || expert >= geometry_.n_expert) throw std::runtime_error("glm5: invalid resident expert id");
    const auto prefix = "blk." + std::to_string(layer) + ".ffn_";
    ExpertWeights w;
    auto set = [&](const char* role, const void*& pointer, int& type) {
        const auto* ref = table_.find(prefix + role + "_exps.weight");
        if (!ref || !ref->native_data) throw std::runtime_error(prefix + role + ": missing complete expert bank");
        const auto bytes = kernels::native_mmvq_weight_bytes(ref->native_type, ref->ne0, ref->ne1);
        pointer = static_cast<const uint8_t*>(ref->native_data) + expert * (bytes / geometry_.n_expert);
        type = ref->native_type;
    };
    set("gate", w.gate, w.gate_type); set("up", w.up, w.up_type); set("down", w.down, w.down_type);
    return w;
}

void project_head(const TensorTable& tensors, const Geometry& g, const float* hidden, float* destination,
                   bool draft, void* stream) {
    const auto optional = "blk." + std::to_string(g.n_layers) + ".nextn.shared_head_head.weight";
    const auto* weight = draft ? tensors.find(optional) : nullptr;
    if (!weight) weight = tensors.find("output.weight");
    if (!weight) throw std::runtime_error("glm5: missing output projection");
    std::string err;
    if (weight->native_data) {
        if (!gemv::gemv_quantized(*weight, {}, {}, nullptr, nullptr, destination, g.n_embd, g.n_vocab,
                                  "output.weight", stream, err, hidden)) throw std::runtime_error(err);
    } else kernels::glm_f32_gemv(hidden, static_cast<const float*>(weight->data), destination, g.n_embd, g.n_vocab, stream);
}

}  // namespace strata::glm5
