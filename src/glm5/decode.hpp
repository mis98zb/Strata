#pragma once

#include "strata/glm5/model.hpp"
#include "strata/glm5/model_file.hpp"
#include "strata/glm5/state.hpp"

namespace strata::glm5 {

class DecodeExecutor {
public:
    DecodeExecutor(const ModelFile& file, const Options& options, const HostState* initial = nullptr);
    ~DecodeExecutor();
    const std::vector<float>& token(int32_t id);
    const std::vector<float>& prefill(const std::vector<int32_t>& ids);
    std::vector<int32_t> generate(int count);
    ExpertCacheStats cache_stats() const;
    MtpStats mtp_stats() const;
    ProbeStats probe_stats();
    HostState export_state();
    int64_t position() const;
    void reset();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::glm5
