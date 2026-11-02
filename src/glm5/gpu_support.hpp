#pragma once

#include <cuda_runtime.h>

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
    explicit Buffer(size_t bytes) { check(cudaMalloc(&ptr_, bytes), "glm5 allocation"); }
    ~Buffer() { if (ptr_) cudaFree(ptr_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void* get() const { return ptr_; }
private:
    void* ptr_ = nullptr;
};

}  // namespace strata::glm5
