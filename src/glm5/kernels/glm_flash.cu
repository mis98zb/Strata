// GLM latent attention adapted from Strata's qsa_prompt_attn: online softmax,
// F16 KV and two-part F16 Q/probabilities with F32 MMA accumulation.
#include "strata/glm5/glm_dsa.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
thread_local bool enabled = false;

#if !defined(__HIPCC__)
constexpr int D = 512, H = 16, C = 32, STRIDE = D + 8, THREADS = 128;
constexpr float LOG2E = 1.4426950408889634f;
struct Shared {
    __half qh[H][STRIDE], ql[H][STRIDE], kv[C][STRIDE];
    __half ph[H][C], pl[H][C];
    float score[H][C + 1], maximum[H], sum[H], sum_error[H], alpha[H], qmax[4];
    int rows[C];
};

__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

// Preserve the compensation steps under the GLM target's fast-math compiler flags.
__device__ __forceinline__ void compensated_add(float value, float& sum, float& correction) {
    const float delta = __fsub_rn(value, correction);
    const float next = __fadd_rn(sum, delta);
    correction = __fsub_rn(__fsub_rn(next, sum), delta);
    sum = next;
}

__device__ __forceinline__ void compensated_scale(float scale, float& sum, float& correction) {
    const float next = __fmul_rn(sum, scale);
    correction = __fmaf_rn(correction, scale, -__fmaf_rn(sum, scale, -next));
    sum = next;
}

__global__ void flash(const float* q, const __half* latent, const int* ids, int heads, int selected,
                       int cached, int pos_base, float scale, float* out) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& s = *reinterpret_cast<Shared*>(storage);
    const int query = blockIdx.x, head0 = blockIdx.y * H;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    q += (static_cast<size_t>(query) * heads + head0) * D;
    out += (static_cast<size_t>(query) * heads + head0) * D;
    const int live_heads = min(H, heads - head0), position = pos_base + query;
    const int count = ids ? selected : min(cached, position + 1);
    if (ids) ids += static_cast<size_t>(query) * selected;
    float qm = 0;
    for (int i = t; i < live_heads * D; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
    for (int shift = 16; shift; shift /= 2) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, shift));
    if (lane == 0) s.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(s.qmax[0], s.qmax[1]), fmaxf(s.qmax[2], s.qmax[3]));
    int exponent = 0;
    if (qm > 0) frexpf(qm, &exponent);
    const float qup = ldexpf(1.0f, 14 - exponent), qdown = ldexpf(scale, exponent - 14);
    for (int i = t; i < H * D; i += THREADS) {
        const int h = i / D, d = i % D;
        const float x = h < live_heads ? q[i] * qup : 0;
        s.qh[h][d] = __float2half_rn(x);
        s.ql[h][d] = __float2half_rn(x - __half2float(s.qh[h][d]));
    }
    if (t < H) { s.maximum[t] = -INFINITY; s.sum[t] = 0; s.sum_error[t] = 0; }
    float acc[16][4] = {}, acc_error[16][4] = {};
    for (int base = 0; base < count; base += C) {
        __syncthreads();
        if (t < C) {
            const int cell = base + t < count ? (ids ? ids[base + t] : base + t) : -1;
            s.rows[t] = cell >= 0 && cell < cached && cell <= position ? cell : -1;
        }
        __syncthreads();
        for (int i = t; i < C * (D / 8); i += THREADS) {
            const int row = i / (D / 8), part = i % (D / 8), cell = s.rows[row];
            const uint4 v = cell >= 0 ? __ldg(reinterpret_cast<const uint4*>(
                latent + static_cast<size_t>(cell) * D) + part) : make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(&s.kv[row][0])[part] = v;
        }
        __syncthreads();
        float score[4] = {}, score_error[4] = {};
        const int cb = warp * 8;
#pragma unroll
        for (int k = 0; k < D; k += 16) {
            uint32_t hi[4], lo[4], b[2];
            hi[0] = *reinterpret_cast<const uint32_t*>(&s.qh[gid][k + 2 * tig]);
            hi[1] = *reinterpret_cast<const uint32_t*>(&s.qh[gid + 8][k + 2 * tig]);
            hi[2] = *reinterpret_cast<const uint32_t*>(&s.qh[gid][k + 2 * tig + 8]);
            hi[3] = *reinterpret_cast<const uint32_t*>(&s.qh[gid + 8][k + 2 * tig + 8]);
            lo[0] = *reinterpret_cast<const uint32_t*>(&s.ql[gid][k + 2 * tig]);
            lo[1] = *reinterpret_cast<const uint32_t*>(&s.ql[gid + 8][k + 2 * tig]);
            lo[2] = *reinterpret_cast<const uint32_t*>(&s.ql[gid][k + 2 * tig + 8]);
            lo[3] = *reinterpret_cast<const uint32_t*>(&s.ql[gid + 8][k + 2 * tig + 8]);
            b[0] = *reinterpret_cast<const uint32_t*>(&s.kv[cb + gid][k + 2 * tig]);
            b[1] = *reinterpret_cast<const uint32_t*>(&s.kv[cb + gid][k + 2 * tig + 8]);
            // Do not repeatedly round low Q components into a large running MMA accumulator.
            float high[4] = {}, low[4] = {};
            mma(high, hi, b);
            mma(low, lo, b);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                compensated_add(high[i], score[i], score_error[i]);
                compensated_add(low[i], score[i], score_error[i]);
            }
        }
        const int col = cb + 2 * tig;
        s.score[gid][col] = s.rows[col] >= 0 ? score[0] * qdown : -INFINITY;
        s.score[gid][col + 1] = s.rows[col + 1] >= 0 ? score[1] * qdown : -INFINITY;
        s.score[gid + 8][col] = s.rows[col] >= 0 ? score[2] * qdown : -INFINITY;
        s.score[gid + 8][col + 1] = s.rows[col + 1] >= 0 ? score[3] * qdown : -INFINITY;
        __syncthreads();
        {
            const int row = t >> 3, sub = t & 7;
            float x[4], maximum = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                x[j] = s.score[row][sub * 4 + j];
                maximum = fmaxf(maximum, x[j]);
            }
            for (int shift = 1; shift < 8; shift *= 2)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, shift));
            const float previous = s.maximum[row], next = fmaxf(previous, maximum);
            float sum = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p = x[j] == -INFINITY ? 0 : exp2f(__fmul_rn(__fsub_rn(x[j], next), LOG2E));
                const float scaled = p * 16384;
                const __half high = __float2half_rn(scaled);
                s.ph[row][sub * 4 + j] = high;
                s.pl[row][sub * 4 + j] = __float2half_rn(scaled - __half2float(high));
                sum += p;
            }
            for (int shift = 1; shift < 8; shift *= 2)
                sum += __shfl_xor_sync(0xffffffffu, sum, shift);
            __syncwarp();
            if (sub == 0) {
                const float alpha = previous == -INFINITY ? 0 :
                    exp2f(__fmul_rn(__fsub_rn(previous, next), LOG2E));
                s.alpha[row] = alpha;
                compensated_scale(alpha, s.sum[row], s.sum_error[row]);
                compensated_add(sum, s.sum[row], s.sum_error[row]);
                s.maximum[row] = next;
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            float tmp[4] = {}, tmp_error[4] = {};
#pragma unroll
            for (int k = 0; k < C; k += 16) {
                const int ca = k + 2 * tig, cb2 = ca + 8;
                const uint32_t hi[4] = {
                    *reinterpret_cast<const uint32_t*>(&s.ph[gid][ca]),
                    *reinterpret_cast<const uint32_t*>(&s.ph[gid + 8][ca]),
                    *reinterpret_cast<const uint32_t*>(&s.ph[gid][cb2]),
                    *reinterpret_cast<const uint32_t*>(&s.ph[gid + 8][cb2])};
                const uint32_t lo[4] = {
                    *reinterpret_cast<const uint32_t*>(&s.pl[gid][ca]),
                    *reinterpret_cast<const uint32_t*>(&s.pl[gid + 8][ca]),
                    *reinterpret_cast<const uint32_t*>(&s.pl[gid][cb2]),
                    *reinterpret_cast<const uint32_t*>(&s.pl[gid + 8][cb2])};
                const int dim = warp * 128 + j * 8 + gid;
                const __half2 v0 = __halves2half2(s.kv[ca][dim], s.kv[ca + 1][dim]);
                const __half2 v1 = __halves2half2(s.kv[cb2][dim], s.kv[cb2 + 1][dim]);
                const uint32_t values[2] = {*reinterpret_cast<const uint32_t*>(&v0),
                                             *reinterpret_cast<const uint32_t*>(&v1)};
                float high[4] = {}, low[4] = {};
                mma(high, hi, values);
                mma(low, lo, values);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    compensated_add(high[i], tmp[i], tmp_error[i]);
                    compensated_add(low[i], tmp[i], tmp_error[i]);
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                compensated_scale(s.alpha[gid + (i >= 2 ? 8 : 0)], acc[j][i], acc_error[j][i]);
                compensated_add(tmp[i] / 16384, acc[j][i], acc_error[j][i]);
                compensated_add(-tmp_error[i] / 16384, acc[j][i], acc_error[j][i]);
            }
        }
    }
    __syncthreads();
    const float sum0 = __fsub_rn(s.sum[gid], s.sum_error[gid]);
    const float sum1 = __fsub_rn(s.sum[gid + 8], s.sum_error[gid + 8]);
    const float inv0 = sum0 > 0 ? __fdiv_rn(1.0f, sum0) : 0;
    const float inv1 = sum1 > 0 ? __fdiv_rn(1.0f, sum1) : 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int dim = warp * 128 + j * 8 + 2 * tig;
        if (gid < live_heads)
            *reinterpret_cast<float2*>(out + gid * D + dim) =
                make_float2(__fmul_rn(__fsub_rn(acc[j][0], acc_error[j][0]), inv0),
                            __fmul_rn(__fsub_rn(acc[j][1], acc_error[j][1]), inv0));
        if (gid + 8 < live_heads)
            *reinterpret_cast<float2*>(out + (gid + 8) * D + dim) =
                make_float2(__fmul_rn(__fsub_rn(acc[j][2], acc_error[j][2]), inv1),
                            __fmul_rn(__fsub_rn(acc[j][3], acc_error[j][3]), inv1));
    }
}
#endif
}  // namespace

void glm_flash_set_enabled(bool value) { enabled = value; }
bool glm_flash_enabled() { return enabled; }
bool glm_flash_supported() {
#if defined(__HIPCC__)
    return false;
#else
    int device = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&p, device) != cudaSuccess)
        throw std::runtime_error("glm5 Flash Attention: cannot query CUDA device");
    return p.major >= 8 && p.sharedMemPerBlockOptin >= sizeof(Shared);
#endif
}

void glm_flash_attn(const float* q, const uint16_t* latent, const int32_t* cells, int kv, int heads,
                    int qk_width, int tokens, int selected, int cached, int pos_base, float* out, void* opaque) {
    if (!q || !latent || !out || !opaque || kv != 512 || heads < 1 || qk_width < 1 ||
        tokens < 1 || selected < 1 || cached < 1 || pos_base < 0)
        throw std::invalid_argument("glm5 Flash Attention: invalid geometry or buffers");
#if defined(__HIPCC__)
    throw std::runtime_error("glm5 Flash Attention currently requires CUDA SM80+");
#else
    int device = 0;
    const auto query_status = cudaGetDevice(&device);
    if (query_status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(query_status));
    static thread_local int configured = -1;
    if (configured != device) {
        if (!glm_flash_supported()) throw std::runtime_error("glm5 Flash Attention requires SM80+ and 70 KiB shared memory");
        const auto status = cudaFuncSetAttribute(flash, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared));
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        configured = device;
    }
    flash<<<dim3(tokens, (heads + H - 1) / H), THREADS, sizeof(Shared), static_cast<cudaStream_t>(opaque)>>>(
        q, reinterpret_cast<const __half*>(latent), cells, heads, selected, cached, pos_base,
        1.0f / sqrtf(static_cast<float>(qk_width)), out);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("glm5 Flash Attention launch: ") + cudaGetErrorString(status));
#endif
}
}  // namespace strata::kernels
