#include "strata/glm5/model_file.hpp"
#include "fixture.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

using glm_fixture::metadata;
using glm_fixture::tensors;

}  // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("strata-glm5-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    try {
        auto md = metadata();
        const auto rows = tensors();
        fixture::write(path, md, rows);
        {
            strata::glm5::ModelFile file(path.string());
            file.validate();
            const auto& g = file.geometry();
            require(g.n_layers == 4 && g.n_nextn == 1, "MTP must not be counted in trunk");
            require(g.n_qsa_layers() == 1 && g.n_gdn_layers() == 3, "KDA/MLA layer split");
            require(g.swiglu_clamp == 10 && g.swiglu_clamp_shexp == 1, "separate SwiGLU limits");
            require(file.stop_ids() == std::set<int32_t>{30, 31}, "model-owned EOS and EOT");
        }
        auto rejected = [&](const std::vector<fixture::Kv>& kv, const std::vector<fixture::Tensor>& ts,
                            const std::string& reason, uint64_t cut = 0) {
            fixture::write(path, kv, ts, cut);
            try {
                strata::glm5::ModelFile file(path.string());
                file.validate();
            } catch (const std::exception& e) {
                require(std::string(e.what()).find(reason) != std::string::npos, e.what());
                return;
            }
            throw std::runtime_error("accepted invalid fixture: " + reason);
        };
        auto changed = md;
        changed[0] = fixture::str("general.architecture", "qwen4exp");
        rejected(changed, rows, "general.architecture");
        changed = md;
        changed.push_back(fixture::u32("glm5-next.rope.dimension_count", 64));
        rejected(changed, rows, "unsupported glm5-next geometry");
        changed = md;
        changed[10] = fixture::u32s("glm5-next.attention.head_count_kv", {0, 0, 1, 0, 1});
        rejected(changed, rows, "layer order");
        changed = md;
        changed[11] = fixture::u32s("glm5-next.swiglu_clamp_exp", {10, 10, 1, 10, 10});
        rejected(changed, rows, "uniform");
        changed = md;
        changed[5] = fixture::str("glm5-next.attention.q_lora_rank", "wrong");
        rejected(changed, rows, "nonnegative integer");
        changed = md;
        changed.push_back(fixture::u32("glm5-next.hyper_connection.count", 8));
        rejected(changed, rows, "unsupported glm5-next geometry");
        auto bad_rows = rows;
        bad_rows.front().shape = {512, 32};
        rejected(md, bad_rows, "token_embd.weight");
        bad_rows = rows;
        bad_rows.erase(bad_rows.begin() + 1);
        rejected(md, bad_rows, "missing tensor");
        rejected(md, rows, "truncated tensor", 8);
        require(strata::glm5::swiglu(100, 100, 1) == 1, "clamp after SiLU, clamp up too");
        const float negative = strata::glm5::swiglu(-2, -3, 1);
        require(std::abs(negative - 2.0f / (1 + std::exp(2.0f))) < 1e-6f, "negative gate is not lower-clamped");
        require(std::abs(strata::glm5::swiglu(2, 3, 0) - 6.0f / (1 + std::exp(-2.0f))) < 1e-6f,
                "zero disables clamp");
        require(strata::core::ModelGeometry{}.n_embd == 2560 &&
                strata::core::ModelGeometry{}.n_expert == 512, "Qwen defaults unchanged");
        std::filesystem::remove(path);
        std::puts("glm5 format: geometry, layer map, tensor shapes, stops, bounds and clamp checks passed");
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove(path);
        std::fprintf(stderr, "glm5 format test: %s\n", e.what());
        return 1;
    }
}
