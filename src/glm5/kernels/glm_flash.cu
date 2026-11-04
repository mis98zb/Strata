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
struct Shared {
    __half qh[H][STRIDE], ql[H][STRIDE], kv[C][STRIDE];
    float score[H][C + 1], maximum[H], sum[H], alpha[H], qmax[4];
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

__device__ __forceinline__ uint32_t halves(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__global__ void flash(const float* q, const __half* latent, const int* ids, int heads, int selected,
                       int cached, int pos_base, float scale_log2, float* out) {
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
    const float qup = ldexpf(1.0f, 14 - exponent), qdown = ldexpf(scale_log2, exponent - 14);
    for (int i = t; i < H * D; i += THREADS) {
        const int h = i / D, d = i % D;
        const float x = h < live_heads ? q[i] * qup : 0;
        s.qh[h][d] = __float2half_rn(x);
        s.ql[h][d] = __float2half_rn(x - __half2float(s.qh[h][d]));
    }
    if (t < H) { s.maximum[t] = -INFINITY; s.sum[t] = 0; }
    float acc[16][4] = {};
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
        float score[4] = {};
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
            mma(score, hi, b);
            mma(score, lo, b);
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
                const float p = x[j] == -INFINITY ? 0 : exp2f(x[j] - next);
                s.score[row][sub * 4 + j] = p;
                sum += p;
            }
            for (int shift = 1; shift < 8; shift *= 2)
                sum += __shfl_xor_sync(0xffffffffu, sum, shift);
            __syncwarp();
            if (sub == 0) {
                const float alpha = previous == -INFINITY ? 0 : exp2f(previous - next);
                s.alpha[row] = alpha;
                s.sum[row] = fmaf(s.sum[row], alpha, sum);
                s.maximum[row] = next;
            }
        }
        __syncthreads();
        float tmp[16][4] = {};
#pragma unroll
        for (int k = 0; k < C; k += 16) {
            const int ca = k + 2 * tig, cb2 = ca + 8;
            const float a = s.score[gid][ca] * 16384, b = s.score[gid][ca + 1] * 16384;
            const float c = s.score[gid + 8][ca] * 16384, d = s.score[gid + 8][ca + 1] * 16384;
            const float e = s.score[gid][cb2] * 16384, f = s.score[gid][cb2 + 1] * 16384;
            const float g = s.score[gid + 8][cb2] * 16384, h = s.score[gid + 8][cb2 + 1] * 16384;
            uint32_t hi[4] = {halves(a, b), halves(c, d), halves(e, f), halves(g, h)}, lo[4];
            const auto* hp = reinterpret_cast<const __half2*>(hi);
            float2 v = __half22float2(hp[0]); lo[0] = halves(a - v.x, b - v.y);
            v = __half22float2(hp[1]); lo[1] = halves(c - v.x, d - v.y);
            v = __half22float2(hp[2]); lo[2] = halves(e - v.x, f - v.y);
            v = __half22float2(hp[3]); lo[3] = halves(g - v.x, h - v.y);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int dim = warp * 128 + j * 8 + gid;
                const __half2 v0 = __halves2half2(s.kv[ca][dim], s.kv[ca + 1][dim]);
                const __half2 v1 = __halves2half2(s.kv[cb2][dim], s.kv[cb2 + 1][dim]);
                const uint32_t values[2] = {*reinterpret_cast<const uint32_t*>(&v0),
                                             *reinterpret_cast<const uint32_t*>(&v1)};
                mma(tmp[j], hi, values);
                mma(tmp[j], lo, values);
            }
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[j][0] = fmaf(acc[j][0], s.alpha[gid], tmp[j][0] / 16384);
            acc[j][1] = fmaf(acc[j][1], s.alpha[gid], tmp[j][1] / 16384);
            acc[j][2] = fmaf(acc[j][2], s.alpha[gid + 8], tmp[j][2] / 16384);
            acc[j][3] = fmaf(acc[j][3], s.alpha[gid + 8], tmp[j][3] / 16384);
        }
    }
    __syncthreads();
    const float inv0 = s.sum[gid] > 0 ? 1 / s.sum[gid] : 0;
    const float inv1 = s.sum[gid + 8] > 0 ? 1 / s.sum[gid + 8] : 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int dim = warp * 128 + j * 8 + 2 * tig;
        if (gid < live_heads)
            *reinterpret_cast<float2*>(out + gid * D + dim) = make_float2(acc[j][0] * inv0, acc[j][1] * inv0);
        if (gid + 8 < live_heads)
            *reinterpret_cast<float2*>(out + (gid + 8) * D + dim) = make_float2(acc[j][2] * inv1, acc[j][3] * inv1);
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
        if (!glm_flash_supported()) throw std::runtime_error("glm5 Flash Attention requires SM80+ and 68 KiB shared memory");
        const auto status = cudaFuncSetAttribute(flash, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared));
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        configured = device;
    }
    flash<<<dim3(tokens, (heads + H - 1) / H), THREADS, sizeof(Shared), static_cast<cudaStream_t>(opaque)>>>(
        q, reinterpret_cast<const __half*>(latent), cells, heads, selected, cached, pos_base,
        1.4426950408889634f / sqrtf(static_cast<float>(qk_width)), out);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("glm5 Flash Attention launch: ") + cudaGetErrorString(status));
#endif
}
}  // namespace strata::kernels
