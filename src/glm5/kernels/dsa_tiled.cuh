// Adapted from Project Maya v1.1.49 (26aacb6), glm_batch.cu and dsa_topk.cuh.
//
// MIT License
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

// Keep both scalar accumulation orders. Maya reduces four head quarters;
// this epilogue instead shares individual dots and sums all heads in order.
struct ScoreTile {
    union {
        struct {
            float q[32][4 * 32 + 4];
            float k[32][64 + 4];
        } inputs;
        float dots[4 * 32][64];
    };
};

template<bool Prefetch>
__global__ void score_tiled(const float* __restrict__ iq, const float* __restrict__ pooled,
                            const float* __restrict__ weights, int queries, int pools, float* out) {
    constexpr int heads = 32, width = 128, query_tile = 4, pool_tile = 64, reduction_tile = 32;
    __shared__ ScoreTile shared;
    const int tid = threadIdx.x, row = tid >> 4, column = tid & 15;
    const int first_query = blockIdx.y * query_tile, first_pool = blockIdx.x * pool_tile;
    float accumulators[8][4] = {};
    float4 next_q[4], next_k[2];
    for (int first = 0; first < width; first += reduction_tile) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int item = tid + 256 * i, r = item >> 3, d = (item & 7) * 4;
            const int query = first_query + r / heads;
            float4 value = make_float4(0, 0, 0, 0);
            if constexpr (Prefetch) {
                if (first)
                    value = next_q[i];
                else if (query < queries)
                    value = *reinterpret_cast<const float4*>(iq + (static_cast<size_t>(query) * heads + r % heads) * width + d);
            } else if (query < queries)
                value = *reinterpret_cast<const float4*>(iq + (static_cast<size_t>(query) * heads + r % heads) * width + first + d);
            shared.inputs.q[d][r] = value.x;
            shared.inputs.q[d + 1][r] = value.y;
            shared.inputs.q[d + 2][r] = value.z;
            shared.inputs.q[d + 3][r] = value.w;
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int item = tid + 256 * i, p = item >> 3, d = (item & 7) * 4;
            float4 value = make_float4(0, 0, 0, 0);
            if constexpr (Prefetch) {
                if (first)
                    value = next_k[i];
                else if (first_pool + p < pools)
                    value = *reinterpret_cast<const float4*>(pooled + static_cast<size_t>(first_pool + p) * width + d);
            } else if (first_pool + p < pools)
                value = *reinterpret_cast<const float4*>(pooled + static_cast<size_t>(first_pool + p) * width + first + d);
            shared.inputs.k[d][p] = value.x;
            shared.inputs.k[d + 1][p] = value.y;
            shared.inputs.k[d + 2][p] = value.z;
            shared.inputs.k[d + 3][p] = value.w;
        }
        __syncthreads();
        if constexpr (Prefetch) {
            if (first + reduction_tile < width) {
                // Keep the next reduction tile in registers while consuming the current shared tile.
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int item = tid + 256 * i, r = item >> 3, d = (item & 7) * 4;
                    const int query = first_query + r / heads;
                    next_q[i] = make_float4(0, 0, 0, 0);
                    if (query < queries)
                        next_q[i] = *reinterpret_cast<const float4*>(iq +
                            (static_cast<size_t>(query) * heads + r % heads) * width + first + reduction_tile + d);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int item = tid + 256 * i, p = item >> 3, d = (item & 7) * 4;
                    next_k[i] = make_float4(0, 0, 0, 0);
                    if (first_pool + p < pools)
                        next_k[i] = *reinterpret_cast<const float4*>(pooled +
                            static_cast<size_t>(first_pool + p) * width + first + reduction_tile + d);
                }
            }
        }
#pragma unroll 8
        for (int d = 0; d < reduction_tile; ++d) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    accumulators[i][j] = fmaf(shared.inputs.q[d][row * 8 + i],
                                              shared.inputs.k[d][column * 4 + j], accumulators[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            shared.dots[row * 8 + i][column * 4 + j] = accumulators[i][j];
    __syncthreads();
    const int query = first_query + tid / pool_tile, pool = first_pool + tid % pool_tile;
    if (query < queries && pool < pools) {
        float sum = 0;
        for (int h = 0; h < heads; ++h)
            sum += fmaxf(shared.dots[(tid / pool_tile) * heads + h][tid % pool_tile], 0.0f) *
                   weights[query * heads + h];
        out[static_cast<size_t>(query) * pools + pool] = sum;
    }
}

__device__ void select_top_pools(const float* scores, int visible, int top, int kpool, int* cells) {
    constexpr int threads = 1024;
    __shared__ unsigned histogram[256], prefix, mask;
    __shared__ int need, greater[threads], equal[threads];
    __shared__ unsigned long long winners[threads];
    const int tid = threadIdx.x;
    if (!tid) { prefix = 0; mask = 0; need = top; }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) histogram[tid] = 0;
        __syncthreads();
        const unsigned p = prefix, m = mask;
        for (int i = tid; i < visible; i += threads) {
            const unsigned key = glm_dsa_key(scores[i]);
            if ((key & m) == p) atomicAdd(histogram + ((key >> shift) & 255), 1u);
        }
        __syncthreads();
        if (!tid) {
            int remaining = need;
            for (int b = 255; b >= 0; --b) {
                if (histogram[b] >= static_cast<unsigned>(remaining)) {
                    prefix = p | (static_cast<unsigned>(b) << shift);
                    mask = m | (255u << shift);
                    need = remaining;
                    break;
                }
                remaining -= histogram[b];
            }
        }
        __syncthreads();
    }
    const unsigned cut = prefix;
    const int needed_equal = need, chunk = (visible + threads - 1) / threads;
    const int lo = min(visible, tid * chunk), hi = min(visible, lo + chunk);
    int gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const auto key = glm_dsa_key(scores[i]);
        gt += key > cut; eq += key == cut;
    }
    greater[tid] = gt; equal[tid] = eq;
    __syncthreads();
    for (int offset = 1; offset < threads; offset <<= 1) {
        const int a = tid >= offset ? greater[tid - offset] : 0;
        const int b = tid >= offset ? equal[tid - offset] : 0;
        __syncthreads();
        greater[tid] += a; equal[tid] += b;
        __syncthreads();
    }
    const int total_greater = greater[threads - 1];
    int greater_offset = greater[tid] - gt, equal_offset = equal[tid] - eq;
    for (int i = lo; i < hi; ++i) {
        const auto key = glm_dsa_key(scores[i]);
        const auto packed = (static_cast<unsigned long long>(key) << 32) | ~static_cast<unsigned>(i);
        if (key > cut) winners[greater_offset++] = packed;
        else if (key == cut) {
            if (equal_offset < needed_equal) winners[total_greater + equal_offset] = packed;
            ++equal_offset;
        }
    }
    int padded = 1;
    while (padded < top) padded <<= 1;
    if (tid >= top && tid < padded) winners[tid] = 0;
    __syncthreads();
    for (int size = 2; size <= padded; size <<= 1) {
        for (int stride = size >> 1; stride; stride >>= 1) {
            if (tid < padded) {
                const int other = tid ^ stride;
                if (other > tid) {
                    const auto a = winners[tid], b = winners[other];
                    if ((tid & size) == 0 ? a < b : a > b) { winners[tid] = b; winners[other] = a; }
                }
            }
            __syncthreads();
        }
    }
    if (tid < top) {
        const int pool = static_cast<int>(~static_cast<unsigned>(winners[tid]));
        for (int m = 0; m < kpool; ++m) cells[tid * kpool + m] = pool * kpool + m;
    }
}

__global__ void select_parallel(const float* score, int pools, int kpool, int top_pools,
                                int tail, int selected, const int* positions, int* cells) {
    const int query = blockIdx.x, tid = threadIdx.x, position = positions[query];
    const int visible = (position + 1) / kpool, top = min(visible, top_pools);
    int* row = cells + static_cast<size_t>(query) * selected;
    for (int s = tid; s < selected; s += blockDim.x) row[s] = -1;
    __syncthreads();
    if (top) select_top_pools(score + static_cast<size_t>(query) * pools, visible, top, kpool, row);
    __syncthreads();
    if (!tid && tail)
        for (int m = 0; m < kpool - 1; ++m) {
            const int cell = visible * kpool + m;
            if (cell <= position && top_pools * kpool + m < selected) row[top_pools * kpool + m] = cell;
        }
}
