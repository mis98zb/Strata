#pragma once

#include <cuda_runtime.h>
#include "strata/glm5/state.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace strata::glm5 {

inline void check(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

class Buffer {
public:
    explicit Buffer(size_t bytes, MemoryBudget* budget = nullptr) : bytes_(bytes), budget_(budget) {
        if (budget_) budget_->acquire(bytes);
        const auto status = cudaMalloc(&ptr_, bytes);
        if (status != cudaSuccess) {
            if (budget_) budget_->release(bytes);
            check(status, "glm5 allocation");
        }
    }
    ~Buffer() { if (ptr_) cudaFree(ptr_); if (budget_) budget_->release(bytes_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void* get() const { return ptr_; }
private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
    MemoryBudget* budget_ = nullptr;
};

}  // namespace strata::glm5
