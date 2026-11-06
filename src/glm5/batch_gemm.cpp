#include "batch_gemm.hpp"
#include "gemv_util.hpp"
#include "strata/glm5/glm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cublas_v2.h>
#include <algorithm>
#include <memory>

#ifdef STRATA_PREFILL_MMQ
#include "ggml.h"
// Pinned ggml host entry point; its header also imports device-only CUDA definitions.
void quantize_mmq_q8_1_cuda(const float* x, const int32_t* ids, void* output,
                           ggml_type type, int64_t width, int64_t row_stride, int64_t head_stride,
                           int64_t batch_stride, int64_t padded_width, int64_t rows, int64_t heads,
                           int64_t batches, cudaStream_t stream);
#endif

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
    const float* immutable_input = nullptr;
    int input_type = -1;
    int64_t input_columns = 0, input_width = 0;
    bool input_ready = false;
    int64_t head_capacity = 1, head_width = 0, input_heads = 1;
    size_t input_stride = 0;
    bool reuse_bounds = false;
    int64_t bounds_columns = 0;

    void* alloc(uint64_t bytes) {
        auto b = std::make_unique<Buffer>(bytes, budget);
        auto* p = b->get(); buffers.push_back(std::move(b)); return p;
    }
    Impl(const Geometry& g, int cols, void* opaque, MemoryBudget* limit)
        : stream(static_cast<cudaStream_t>(opaque)), budget(limit), capacity(cols),
          reuse_bounds(gemv::skip_unused_quant()) {
        if (!stream || cols < 1 || cols > 4096) throw std::invalid_argument("glm5 GEMM: invalid stream or batch");
        head_capacity = g.n_head;
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
            auto image_bytes = strata::prefill::mmq::q8_bytes(cols, max_width);
            if (gemv::skip_unused_quant()) {
                head_width = std::max(g.kv_lora_rank, g.mla_head_dim);
                image_bytes = std::max(image_bytes,
                    strata::prefill::mmq::q8_bytes(checked_mul(cols, head_capacity), head_width));
            }
            xq = alloc(image_bytes);
            // MMQ loads a complete ID tile, including the masked columns of a ragged tail.
            constexpr int id_padding = 128;
            identity = static_cast<int32_t*>(alloc((cols + id_padding) * sizeof(int32_t)));
            bounds = static_cast<int32_t*>(alloc(2 * sizeof(int32_t)));
            strata::prefill::mmq::iota(identity, cols + id_padding, stream);
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

void BatchGemm::begin_input(const float* input, int type, int64_t columns, int64_t width, int64_t heads) {
    auto& p = *impl_;
    if (!input || p.immutable_input || columns < 1 || columns > p.capacity || width < 1 ||
        width > p.max_width || heads < 1 || heads > p.head_capacity)
        throw std::invalid_argument("glm5 quantized input scope: invalid or nested immutable input");
    p.input_type = type; p.input_columns = columns; p.input_width = width;
    p.input_ready = false;
    p.input_heads = 1;
#ifdef STRATA_PREFILL_MMQ
    namespace mmq = strata::prefill::mmq;
    if (heads > 1 && p.mmq && columns > 8 && columns % 4 == 0 && width % 256 == 0 &&
        width <= p.head_width && mmq::supported(type) && !mmq::fp4_activations(type)) {
        const auto padded_width = (width + 511) / 512 * 512;
        quantize_mmq_q8_1_cuda(input, nullptr, p.xq, static_cast<ggml_type>(type), width, width,
                              columns * width, columns * heads * width, padded_width, columns, heads, 1, p.stream);
        check(cudaGetLastError(), "glm5 native head quantization");
        ++p.counts.native_quant_calls;
        p.input_heads = heads;
        // Separate ggml channels keep each head's K-block stride unchanged. The guard is once per image.
        p.input_stride = mmq::q8_bytes(columns, width) - mmq::q8_bytes(0, width);
        p.input_ready = true;
    }
#endif
    p.immutable_input = input;
}

void BatchGemm::end_input() {
    impl_->immutable_input = nullptr;
    impl_->input_ready = false;
}

void BatchGemm::native(int type, const void* weights, const float* x, float* y,
                       int64_t columns, int64_t rows, int64_t width) {
    auto& p = *impl_;
    p.validate(columns, rows, width);
    if (!weights || !x || !y || !kernels::native_mmvq_supported(type))
        throw std::invalid_argument("glm5 GEMM: unsupported native matrix");
#ifdef STRATA_PREFILL_MMQ
    namespace mmq = strata::prefill::mmq;
    if (p.mmq && width % 256 == 0 && mmq::fits(type, rows)) {
        const auto address = reinterpret_cast<uintptr_t>(x);
        const auto first = reinterpret_cast<uintptr_t>(p.immutable_input);
        const auto stride = static_cast<uint64_t>(columns * width) * sizeof(float);
        const bool matching_shape = p.immutable_input && p.input_type == type &&
            p.input_columns == columns && p.input_width == width;
        const bool shared_input = matching_shape && address >= first &&
            (address - first) % stride == 0 && (address - first) / stride < static_cast<uint64_t>(p.input_heads);
        const auto head = shared_input ? (address - first) / stride : 0;
        const void* image = p.xq;
        if (!shared_input || !p.input_ready) {
            mmq::quantize(x, nullptr, p.xq, type, width, width, columns, p.stream);
            ++p.counts.native_quant_calls;
            p.input_ready = shared_input && p.input_heads == 1;
        } else {
            ++p.counts.reused_quant_calls;
            image = static_cast<const uint8_t*>(p.xq) + head * p.input_stride;
        }
        // Only this method writes these integer bounds, ordered on the backend's one stream.
        if (!p.reuse_bounds || p.bounds_columns != columns) {
            kernels::glm_matrix_bounds(p.bounds, columns, p.stream);
            p.bounds_columns = columns;
            ++p.counts.bounds_launches;
        } else ++p.counts.reused_bounds;
        mmq::Product product;
        product.w = weights; product.type = type; product.w_rows = rows; product.w_cols = width;
        product.expert_bytes = mmq::matrix_bytes(type, rows, width); product.n = 1;
        product.xq = image; product.bounds = p.bounds; product.ids = p.identity;
        product.total_rows = columns; product.max_rows = columns; product.dst = y; product.ld_dst = rows;
        p.mmq->run(product, p.stream);
        ++p.counts.mmq_calls;
        return;
    }
#endif
    p.input_ready = false;
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
