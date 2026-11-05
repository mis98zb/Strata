// src/glm5/kernels/glm_delta.cu - the KDA delta rule: the only part of GLM-5.3-Flash that is sequential over
// tokens.
//
// Ported from `ggml_compute_forward_kda_f32` (ik_llama.cpp ggml/src/ggml.c), which is the CPU reference the
// CUDA path is checked against.  The whole recurrence, per head and per token:
//
//     attn      = sum_i k[i] * (q[i] * scale)
//     v'        = sum_col S[row,col] * k[col] * decay[col]
//     out[row]  = (sum_col S[row,col] * q[col] * decay[col]) * scale + v_new * attn   where
//     v_new     = beta * (v[row] - v')
//     S[row,col] = clamp(decay[col] * S[row,col] + v_new * k[col], -1e6, 1e6)
//
// Note where `scale` sits and where it does not: it is inside `attn` on q, and separately on the q-side state
// read, but NOT on `k[col]` in the state update and not on the decayed k in `v'`.  The decay is applied to the
// state and to both state reads; the raw `k` is what writes the state.  Every one of those choices changes the
// result and none of them changes whether the model runs.
//
// LAYOUT.  q, k, v, g and out are all [head_dim, head_count, T] with head_dim fastest - which is the engine's
// [width, tokens] convention with `width = head_dim * head_count`, since `head*head_dim + col` for row-major
// (head, col) is the same offset either way.  `beta` is [head_count, T].  `state` is per head, [row, col] with
// ROW FASTEST: `state[h*hd*hd + row + col*hd]`.
#include "strata/glm5/glm.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
thread_local bool register_state = false;
}

void glm_kda_set_register_state(bool enabled) { register_state = enabled; }
bool glm_kda_register_state() { return register_state; }

namespace {

/// The largest head_dim the shared window supports.  KDA's is 128; the bound is here so a geometry mistake is a
/// refusal at launch rather than a shared-memory overrun.
constexpr int MAX_HEAD_DIM = 256;

void sync_if_needed(void* stream, const char* what) {
    if (stream != nullptr) return;
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + " launch: " + cudaGetErrorString(e));
}

/// ONE BLOCK PER HEAD, ONE THREAD PER STATE ROW.
///
/// The state is `head_dim` squared per head and a thread's row is the only thing it touches, so the whole
/// recurrence needs no communication: thread `row` reads and writes row `row` and nothing else.  What does cross
/// threads is the token's `k`, `q` and decay, which are read 128 times each per token by every row - so they go
/// to shared memory once, and the two barriers per token are the entire synchronisation cost.
//
// The two barriers per token are not an accident: the first is the WAR guard for the shared arrays (the previous
// token's readers), the second makes this token's values visible.  Removing either produces a kernel that is
// correct on the first token and intermitently wrong afterwards.
__global__ void kda_delta_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                 const float* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta, float* __restrict__ state,
                                 float* __restrict__ out, int HD, int64_t T) {
    __shared__ float ks[MAX_HEAD_DIM];
    __shared__ float qs[MAX_HEAD_DIM];
    __shared__ float dec[MAX_HEAD_DIM];

    const int h = blockIdx.x;
    const int row = threadIdx.x;
    const int64_t n_head = gridDim.x;
    const int64_t n_v = (int64_t) HD * n_head;
    float* S = state + (int64_t) h * HD * HD;
    // `1.0f/sqrtf`, not `rsqrtf`: the reference divides and the approximation is 2 ulp away from it.
    const float scale = 1.0f / sqrtf((float) HD);

    for (int64_t t = 0; t < T; ++t) {
        const int64_t base = t * n_v + (int64_t) h * HD;
        __syncthreads();
        ks[row] = k[base + row];
        qs[row] = q[base + row];
        // The gate is already bounded to (-5, 0) upstream, so the `min(g, 50)` is unreachable here - it is kept
        // because it is in the reference and because it is the difference between a large decay and an inf if the
        // gate is ever fed from somewhere else.
        dec[row] = expf(fminf(g[base + row], 50.0f));
        __syncthreads();

        // Computed redundantly by every row: a block reduction would be 2 more barriers per token to save 128
        // shared broadcasts, and shared broadcast is the cheapest thing this kernel does.
        float attn = 0.0f;
        for (int i = 0; i < HD; ++i) attn += ks[i] * (qs[i] * scale);

        const float beta_val = 1.0f / (1.0f + expf(-beta[t * n_head + h]));

        float v_prime = 0.0f;
        float out_val = 0.0f;
        const float* Sr = S + row;
        for (int col = 0; col < HD; ++col) {
            const int64_t at = (int64_t) col * HD;
            const float d = dec[col];
            const float s = Sr[at];
            v_prime += s * (ks[col] * d);
            out_val += s * (qs[col] * d);
        }
        const float v_new = beta_val * (v[base + row] - v_prime);
        out[base + row] = out_val * scale + v_new * attn;

        // `ks[col]` bare - the decay is on the state, not on the key that writes it.
        float* Sw = S + row;
        for (int col = 0; col < HD; ++col) {
            const int64_t at = (int64_t) col * HD;
            const float s = dec[col] * Sw[at] + v_new * ks[col];
            Sw[at] = fminf(fmaxf(s, -1e6f), 1e6f);
        }
    }
}

__global__ void kda_register_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                    const float* __restrict__ v, const float* __restrict__ gate,
                                    const float* __restrict__ beta, float* __restrict__ state,
                                    float* __restrict__ out, int head_dim, int64_t tokens) {
    constexpr int width = 128, rows = 32;
    // Avoid hoisting all shared operands into registers alongside the persistent state row.
    __shared__ volatile float keys[width], queries[width], decay[width];
    const int lane = threadIdx.x, row = blockIdx.y * rows + lane, head = blockIdx.x;
    const int64_t heads = gridDim.x, stride = heads * width;
    float* base_state = state + static_cast<int64_t>(head) * width * width + row;
    // Fixed-index unrolling keeps each row in registers across all tokens.
    float values[width];
#pragma unroll
    for (int col = 0; col < width; ++col) values[col] = base_state[col * width];
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    for (int64_t t = 0; t < tokens; ++t) {
        const int64_t base = t * stride + head * width;
        __syncthreads();
        for (int d = lane; d < width; d += rows) {
            keys[d] = k[base + d];
            queries[d] = q[base + d];
            decay[d] = expf(fminf(gate[base + d], 50.0f));
        }
        __syncthreads();
        float attention = 0;
        for (int d = 0; d < width; ++d) attention += keys[d] * (queries[d] * scale);
        const float weight = 1.0f / (1.0f + expf(-beta[t * heads + head]));
        float prediction = 0, result = 0;
#pragma unroll
        for (int col = 0; col < width; ++col) {
            const float s = values[col], d = decay[col];
            prediction += s * (keys[col] * d);
            result += s * (queries[col] * d);
        }
        const float change = weight * (v[base + row] - prediction);
        out[base + row] = result * scale + change * attention;
#pragma unroll
        for (int col = 0; col < width; ++col) {
            const float s = decay[col] * values[col] + change * keys[col];
            values[col] = fminf(fmaxf(s, -1e6f), 1e6f);
        }
    }
#pragma unroll
    for (int col = 0; col < width; ++col) base_state[col * width] = values[col];
}

}  // namespace

void glm_kda_delta(const float* q, const float* k, const float* v, const float* gate, const float* beta,
                   float* state, float* out, int64_t head_dim, int64_t n_head, int64_t T, void* stream) {
    if (T <= 0 || n_head <= 0 || head_dim <= 0) return;
    if (register_state && head_dim != 128)
        throw std::invalid_argument("glm5 register KDA requires head dimension 128");
    if (head_dim > MAX_HEAD_DIM) {
        std::fprintf(stderr, "glm_kda_delta: head_dim %lld exceeds %d\n", (long long) head_dim, MAX_HEAD_DIM);
        return;
    }
    if (register_state) {
        kda_register_kernel<<<dim3(static_cast<unsigned>(n_head), 4), 32, 0, static_cast<cudaStream_t>(stream)>>>(
            q, k, v, gate, beta, state, out, head_dim, T);
        check_launch("glm_kda_register");
        sync_if_needed(stream, "glm_kda_register");
        return;
    }
    kda_delta_kernel<<<(unsigned) n_head, (unsigned) head_dim, 0, (cudaStream_t) stream>>>(
        q, k, v, gate, beta, state, out, (int) head_dim, T);
    check_launch("glm_kda_delta");
    sync_if_needed(stream, "glm_kda_delta");
}

}  // namespace strata::kernels
