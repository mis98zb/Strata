#pragma once

#include "gpu_support.hpp"
#include "phase_probe.hpp"

#include <array>

namespace strata::glm5 {

class ActivationPipeline {
    struct Slot {
        std::unique_ptr<Buffer> device;
        cudaEvent_t uploaded = nullptr, computed = nullptr, downloaded = nullptr;
        float* destination = nullptr;
        size_t bytes = 0;
        ~Slot() {
            if (uploaded) cudaEventDestroy(uploaded);
            if (computed) cudaEventDestroy(computed);
            if (downloaded) cudaEventDestroy(downloaded);
        }
    };
    std::array<Slot, 2> slots_;
    cudaStream_t upload_ = nullptr, download_ = nullptr;
    PhaseProbe* probe_ = nullptr;
public:
    ActivationPipeline(uint64_t row_batch_bytes, MemoryBudget& budget, PhaseProbe* probe = nullptr) : probe_(probe) {
        try {
            check(cudaStreamCreateWithFlags(&upload_, cudaStreamNonBlocking), "glm5 activation upload stream");
            check(cudaStreamCreateWithFlags(&download_, cudaStreamNonBlocking), "glm5 activation download stream");
            for (auto& s : slots_) {
                s.device = std::make_unique<Buffer>(row_batch_bytes, &budget);
                check(cudaEventCreateWithFlags(&s.uploaded, cudaEventDisableTiming), "glm5 activation upload event");
                check(cudaEventCreateWithFlags(&s.computed, cudaEventDisableTiming), "glm5 activation compute event");
                check(cudaEventCreateWithFlags(&s.downloaded, cudaEventDisableTiming), "glm5 activation download event");
            }
        } catch (...) {
            if (upload_) cudaStreamDestroy(upload_);
            if (download_) cudaStreamDestroy(download_);
            throw;
        }
    }
    ~ActivationPipeline() {
        if (upload_) { cudaStreamSynchronize(upload_); cudaStreamDestroy(upload_); }
        if (download_) { cudaStreamSynchronize(download_); cudaStreamDestroy(download_); }
    }
    void collect(int index) {
        auto& s = slots_.at(index);
        if (!s.destination) return;
        if (probe_) {
            auto timing = probe_->host(ProbePhase::Synchronize);
            check(cudaEventSynchronize(s.downloaded), "glm5 activation slot reuse");
        } else check(cudaEventSynchronize(s.downloaded), "glm5 activation slot reuse");
        s.destination = nullptr;
    }
    void prepare(int index, float* rows, size_t bytes) {
        collect(index);
        auto& s = slots_.at(index);
        s.bytes = bytes;
        const auto copy = [&] {
            check(cudaMemcpyAsync(s.device->get(), rows, bytes, cudaMemcpyHostToDevice, upload_),
                  "glm5 pipelined activation upload");
        };
        if (probe_) {
            probe_->gpu(ProbePhase::ActivationH2D, upload_, copy);
            probe_->bytes(ProbePhase::ActivationH2D, bytes);
        } else copy();
        check(cudaEventRecord(s.uploaded, upload_), "glm5 activation upload ready");
    }
    void input(int index, float* device, cudaStream_t compute) {
        auto& s = slots_.at(index);
        check(cudaStreamWaitEvent(compute, s.uploaded, 0), "glm5 activation input dependency");
        check(cudaMemcpyAsync(device, s.device->get(), s.bytes, cudaMemcpyDeviceToDevice, compute),
              "glm5 activation input");
    }
    void output(int index, const float* device, float* destination, cudaStream_t compute) {
        auto& s = slots_.at(index);
        s.destination = destination;
        check(cudaMemcpyAsync(s.device->get(), device, s.bytes, cudaMemcpyDeviceToDevice, compute),
              "glm5 activation output");
        check(cudaEventRecord(s.computed, compute), "glm5 activation computed");
        check(cudaStreamWaitEvent(download_, s.computed, 0), "glm5 activation output dependency");
        const auto copy = [&] {
            check(cudaMemcpyAsync(destination, s.device->get(), s.bytes, cudaMemcpyDeviceToHost, download_),
                  "glm5 pipelined activation download");
        };
        if (probe_) {
            probe_->gpu(ProbePhase::ActivationD2H, download_, copy);
            probe_->bytes(ProbePhase::ActivationD2H, s.bytes);
        } else copy();
        check(cudaEventRecord(s.downloaded, download_), "glm5 activation download ready");
    }
    void finish() { collect(0); collect(1); }
};

}  // namespace strata::glm5
