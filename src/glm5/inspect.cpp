#include "strata/glm5/model_file.hpp"

#include <cstdio>
#include <exception>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) == "--help") {
        std::fprintf(argc == 2 ? stdout : stderr, "Usage: glm5-inspect MODEL.gguf\n"
                     "Checks all shards, GLM geometry and required trunk tensor shapes. No GPU or upload.\n");
        return argc == 2 ? 0 : 2;
    }
    try {
        strata::glm5::ModelFile file(argv[1]);
        file.validate();
        const auto& g = file.geometry();
        std::printf("glm5-next: %lld trunk layers, %lld embedding, %lld experts (%lld active), %lld vocabulary\n",
                    static_cast<long long>(g.n_layers), static_cast<long long>(g.n_embd),
                    static_cast<long long>(g.n_expert), static_cast<long long>(g.n_expert_used),
                    static_cast<long long>(g.n_vocab));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm5-inspect: %s\n", e.what());
        return 1;
    }
}
