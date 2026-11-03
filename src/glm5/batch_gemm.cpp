#include "batch_gemm.hpp"
#include "strata/glm5/glm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cublas_v2.h>
#include <algorithm>
#include <memory>

namespace strata::glm5 {
namespace {
thread_local BatchGemm* current = nullptr;
void blas_check(cublasStatus_t status, const char* action) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(action) + ": cuBLAS status " + std::to_string(status));
}
struct Blas {
    cublasHandle_t handle = nullptr;
    Blas() { blas_check(cublasCreate(&handle), "glm5 GEMM create"); }
    ~Blas() { if (handle) cublasDestroy(handle); }
};
}

struct BatchGemm::Impl {
    cudaStream_t stream;
    MemoryBudget* budget;
    int capacity;
    int64_t max_width, weight_elements;
    std::vector<std::unique_ptr<Buffer>> buffers;
    Blas blas;
    uint16_t *x16 = nullptr, *w16 = nullptr;
    void* xq = nullptr;
    int32_t *identity = nullptr, *bounds = nullptr;
#ifdef STRATA_PREFILL_MMQ
    std::unique_ptr<strata::prefill::mmq::Context> mmq;
#endif
    BatchStats counts;

    void* alloc(uint64_t bytes) {
        auto b = std::make_unique<Buffer>(bytes, budget);
        auto* p = b->get(); buffers.push_back(std::move(b)); return p;
    }
    Impl(const Geometry& g, int cols, void* opaque, MemoryBudget* limit)
        : stream(static_cast<cudaStream_t>(opaque)), budget(limit), capacity(cols) {
        if (!stream || cols < 1 || cols > 4096) throw std::invalid_argument("glm5 GEMM: invalid stream or batch");
        max_width = std::max({g.hc_dim(), 2 * g.n_embd, g.n_head * g.mla_head_dim, g.q_lora_rank,
                              g.n_ff, g.n_ff_dense, g.n_head * g.kda_head_dim});
        const auto max_rows = std::max({g.n_embd, g.n_ff, g.n_ff_dense, g.n_head * g.kv_lora_rank,
                                        g.n_head * g.kda_head_dim, g.n_head * g.mla_head_dim});
        weight_elements = std::min<int64_t>(32 * 1024 * 1024, max_width * max_rows);
        x16 = static_cast<uint16_t*>(alloc(checked_mul(cols, max_width * sizeof(uint16_t))));
        w16 = static_cast<uint16_t*>(alloc(weight_elements * sizeof(uint16_t)));
        blas_check(cublasSetStream(blas.handle, stream), "glm5 GEMM stream");
        constexpr size_t workspace_bytes = 4 * 1024 * 1024;
        blas_check(cublasSetWorkspace(blas.handle, alloc(workspace_bytes), workspace_bytes), "glm5 GEMM workspace");
#ifdef STRATA_PREFILL_MMQ
        if (strata::prefill::mmq::built()) {
            mmq = std::make_unique<strata::prefill::mmq::Context>();
            xq = alloc(strata::prefill::mmq::q8_bytes(cols, max_width));
            identity = static_cast<int32_t*>(alloc(cols * sizeof(int32_t)));
            bounds = static_cast<int32_t*>(alloc(2 * sizeof(int32_t)));
            strata::prefill::mmq::iota(identity, cols, stream);
        }
#endif
    }
    ~Impl() { cudaStreamSynchronize(stream); }
    void validate(int64_t columns, int64_t rows, int64_t width) {
        if (columns < 1 || columns > capacity || rows < 1 || rows > INT32_MAX ||
            width < 1 || width > max_width)
            throw std::invalid_argument("glm5 GEMM: matrix exceeds the planned workspace");
        counts.max_columns = std::max<uint64_t>(counts.max_columns, columns);
    }
};

BatchGemm::BatchGemm(const Geometry& g, int cols, void* stream, MemoryBudget* budget)
    : impl_(std::make_unique<Impl>(g, cols, stream, budget)) {}
BatchGemm::~BatchGemm() = default;
const BatchStats& BatchGemm::stats() const { return impl_->counts; }

void BatchGemm::native(int type, const void* weights, const float* x, float* y,
                       int64_t columns, int64_t rows, int64_t width) {
    auto& p = *impl_;
    p.validate(columns, rows, width);
    if (!weights || !x || !y || !kernels::native_mmvq_supported(type))
        throw std::invalid_argument("glm5 GEMM: unsupported native matrix");
#ifdef STRATA_PREFILL_MMQ
    namespace mmq = strata::prefill::mmq;
    if (p.mmq && width % 256 == 0 && mmq::fits(type, rows)) {
        mmq::quantize(x, nullptr, p.xq, type, width, width, columns, p.stream);
        kernels::glm_matrix_bounds(p.bounds, columns, p.stream);
        mmq::Product product;
        product.w = weights; product.type = type; product.w_rows = rows; product.w_cols = width;
        product.expert_bytes = mmq::matrix_bytes(type, rows, width); product.n = 1;
        product.xq = p.xq; product.bounds = p.bounds; product.ids = p.identity;
        product.total_rows = columns; product.max_rows = columns; product.dst = y; product.ld_dst = rows;
        p.mmq->run(product, p.stream);
        ++p.counts.mmq_calls;
        return;
    }
#endif
    kernels::f32_to_f16_bulk(x, p.x16, columns * width, p.stream);
    const int64_t tile_rows = p.weight_elements / width;
    if (!tile_rows) throw std::runtime_error("glm5 GEMM: dequantization scratch is too small");
    blas_check(cublasSetMathMode(p.blas.handle, CUBLAS_DEFAULT_MATH), "glm5 FP16 GEMM math mode");
    const float alpha = 1, beta = 0;
    for (int64_t first = 0; first < rows; first += tile_rows) {
        const auto n = std::min(tile_rows, rows - first);
        kernels::dequant_f16(type, weights, first, n, width, p.w16, p.stream);
        blas_check(cublasGemmEx(p.blas.handle, CUBLAS_OP_T, CUBLAS_OP_N, n, columns, width, &alpha,
                               p.w16, CUDA_R_16F, width, p.x16, CUDA_R_16F, width, &beta, y + first,
                               CUDA_R_32F, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                   "glm5 native FP16 GEMM");
        ++p.counts.f16_calls;
    }
}

void BatchGemm::f32(const float* weights, const float* x, float* y,
                    int64_t columns, int64_t rows, int64_t width) {
    auto& p = *impl_;
    p.validate(columns, rows, width);
    if (!weights || !x || !y) throw std::invalid_argument("glm5 SGEMM: null matrix");
#ifndef STRATA_USE_HIP
    blas_check(cublasSetMathMode(p.blas.handle, CUBLAS_PEDANTIC_MATH), "glm5 SGEMM math mode");
#endif
    const float alpha = 1, beta = 0;
    blas_check(cublasSgemm(p.blas.handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, columns, width,
                          &alpha, weights, width, x, width, &beta, y, rows), "glm5 F32 GEMM");
    ++p.counts.f32_calls;
}

BatchGemm* active_batch_gemm() { return current; }
BatchScope::BatchScope(BatchGemm* backend) : previous_(current) { current = backend; }
BatchScope::~BatchScope() { current = previous_; }

}  // namespace strata::glm5
