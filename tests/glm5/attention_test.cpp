#include "runtime_fixture.hpp"
#include "gpu_support.hpp"
#include "attention_scope.hpp"
#include "prefill.hpp"
#include "decode.hpp"
#include "strata/glm5/layer.hpp"
#include "evaluation.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>

namespace {
using namespace strata::glm5;
namespace kernels = strata::kernels;
using glm_fixture::require;
using glm_fixture::near;

template<class T> void upload(Buffer& b, const std::vector<T>& v) {
    check(cudaMemcpy(b.get(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "attention test upload");
}

void score_scope() {
    Options outer;
    outer.dsa = true; outer.flash_attn = true;
    outer.mla_attention_layout = MlaAttentionLayout::HeadMajor;
    outer.flash_kv_load = FlashKvLoad::Prefetch;
    {
        AttentionScope automatic(outer);
        require(strata::kernels::glm_dsa_parallel_score(), "Flash automatic scorer is not parallel");
        require(kernels::glm_flash_head_major(), "head-major layout was not scoped");
        require(kernels::glm_flash_prefetch(), "KV prefetch was not scoped");
        auto inner = outer; inner.dsa_score = DsaScore::Scalar;
        inner.mla_attention_layout = MlaAttentionLayout::TokenMajor;
        inner.flash_kv_load = FlashKvLoad::Direct;
        {
            AttentionScope scalar(inner);
            require(!strata::kernels::glm_dsa_parallel_score(), "scalar score override ignored");
            require(!kernels::glm_flash_head_major(), "token-major layout override ignored");
            require(!kernels::glm_flash_prefetch(), "direct KV load override ignored");
        }
        require(strata::kernels::glm_dsa_parallel_score(), "nested score scope was not restored");
        require(kernels::glm_flash_head_major(), "nested MLA layout was not restored");
        require(kernels::glm_flash_prefetch(), "nested KV prefetch was not restored");
        auto invalid = outer; invalid.mla_attention_layout = static_cast<MlaAttentionLayout>(99);
        bool refused = false;
        try { AttentionScope bad(invalid); } catch (const std::invalid_argument&) { refused = true; }
        require(refused && kernels::glm_flash_head_major() && kernels::glm_flash_enabled(),
                "invalid MLA layout leaked scoped state");
        invalid = outer; invalid.flash_kv_load = static_cast<FlashKvLoad>(99);
        refused = false;
        try { AttentionScope bad(invalid); } catch (const std::invalid_argument&) { refused = true; }
        require(refused && kernels::glm_flash_prefetch() && kernels::glm_flash_head_major(),
                "invalid KV load mode leaked scoped state");
    }
    require(!strata::kernels::glm_dsa_parallel_score(), "score scope leaked");
    require(!kernels::glm_flash_head_major(), "MLA layout scope leaked");
    require(!kernels::glm_flash_prefetch(), "KV prefetch scope leaked");
    outer.flash_attn = false;
    {
        AttentionScope automatic(outer);
        require(!strata::kernels::glm_dsa_parallel_score(), "non-Flash automatic scorer changed");
        require(!kernels::glm_flash_prefetch(), "KV prefetch must not enable Flash");
    }
    outer.dsa_score = DsaScore::Parallel;
    {
        AttentionScope parallel(outer);
        require(strata::kernels::glm_dsa_parallel_score(), "non-Flash parallel override ignored");
    }
}

std::vector<double> attention_reference(const std::vector<float>& q, const std::vector<uint16_t>& kv,
        const std::vector<int32_t>& ids, int heads, int tokens, int cached, int selected, int pos,
        bool sparse, int qk_width, double* score_max = nullptr) {
    constexpr int width = 512;
    std::vector<double> result(q.size());
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> valid;
        const int n = sparse ? selected : cached;
        for (int s = 0; s < n; ++s) {
            const int cell = sparse ? ids[t * selected + s] : s;
            if (cell >= 0 && cell < cached && cell <= pos + t) valid.push_back(cell);
        }
        for (int h = 0; h < heads; ++h) {
            std::vector<double> scores(valid.size());
            double maximum = -INFINITY, sum = 0;
            for (size_t s = 0; s < valid.size(); ++s) {
                double dot = 0;
                for (int d = 0; d < width; ++d)
                    dot += q[(t * heads + h) * width + d] *
                           static_cast<double>(ggml_fp16_to_fp32(kv[valid[s] * width + d]));
                scores[s] = dot / std::sqrt(static_cast<double>(qk_width));
                maximum = std::max(maximum, scores[s]);
            }
            if (score_max) *score_max = std::max(*score_max, maximum);
            for (auto& s : scores) { s = std::exp(s - maximum); sum += s; }
            for (int d = 0; d < width; ++d) {
                double expected = 0;
                for (size_t s = 0; s < valid.size(); ++s)
                    expected += scores[s] * ggml_fp16_to_fp32(kv[valid[s] * width + d]);
                result[(t * heads + h) * width + d] = sum ? expected / sum : 0;
            }
        }
    }
    return result;
}

struct AttentionError {
    double rmse, nmse, maximum;
};

AttentionError attention_error(const std::vector<float>& actual, const std::vector<double>& expected) {
    require(actual.size() == expected.size() && !actual.empty(), "invalid attention comparison shape");
    double error = 0, power = 0, maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "non-finite attention output");
        const double delta = actual[i] - expected[i];
        error += delta * delta; power += expected[i] * expected[i];
        maximum = std::max(maximum, std::abs(delta));
    }
    return {std::sqrt(error / actual.size()), error / std::max(power, 1e-12), maximum};
}

void flash_case(int heads, int tokens, int cached, int selected, bool sparse, cudaStream_t stream,
                bool precision_stress = false) {
    constexpr int width = 512;
    const int pos = cached - tokens;
    std::vector<float> q(tokens * heads * width), result(q.size() + 32, 12345);
    std::vector<uint16_t> kv(cached * width);
    std::vector<int32_t> ids(tokens * selected);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = std::sin(i * 0.173) *
            (precision_stress ? 4.0f : i < static_cast<size_t>(heads * width) ? 1e-6f : 0.3f);
    for (size_t i = 0; i < kv.size(); ++i)
        kv[i] = ggml_fp32_to_fp16(std::cos(i * 0.137) * (precision_stress ? 1.0f : 0.25f));
    for (int t = 0; t < tokens; ++t)
        for (int s = 0; s < selected; ++s)
            ids[t * selected + s] = (t == 0 && tokens > 1) || s % 11 == 0 ? -1 :
                (s % 13 == 0 ? cached + 7 : (s * 17 + t * 3) % cached);
    Buffer dq(q.size() * 4), dk(kv.size() * 2), di(ids.size() * 4), output(result.size() * 4);
    upload(dq, q); upload(dk, kv); upload(di, ids); upload(output, result);
    strata::kernels::glm_flash_attn(static_cast<float*>(dq.get()), static_cast<uint16_t*>(dk.get()),
        sparse ? static_cast<int32_t*>(di.get()) : nullptr, width, heads, 256, tokens,
        sparse ? selected : cached, cached, pos, static_cast<float*>(output.get()) + 16, stream);
    check(cudaStreamSynchronize(stream), "Flash Attention test sync");
    check(cudaMemcpy(result.data(), output.get(), result.size() * 4, cudaMemcpyDeviceToHost), "attention output");
    for (int i = 0; i < 16; ++i)
        require(result[i] == 12345 && result[result.size() - 1 - i] == 12345, "attention output guard changed");
    const auto reference = attention_reference(q, kv, ids, heads, tokens, cached, selected, pos, sparse, 256);
    const auto error = attention_error(std::vector<float>(result.begin() + 16, result.end() - 16), reference);
    require(error.nmse < 1e-7 && error.maximum < 2e-4,
            "native Flash Attention differs from double CPU reference");
    if (precision_stress)
        require(error.nmse < 1e-12 && error.maximum < 2e-5,
                "Flash QK precision regression against double CPU reference");
}

void head_major_case(int heads, int tokens, int cached, int selected, bool sparse, int stress,
                       cudaStream_t stream, bool benchmark = false, bool tight_batch = false,
                       bool prefetch_check = false) {
    constexpr int width = 512, guard = 16;
    const int first = benchmark || tight_batch ? 0 : 2;
    const int total = benchmark || tight_batch ? tokens : tokens + 5, pos = cached - tokens;
    std::vector<float> q(tokens * heads * width), packed(heads * total * width + 2 * guard, 12345);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = stress == 2 ? (i % 2 ? -0.0f : 0.0f) :
            std::sin(i * 0.173) * (stress == 1 ? ((i / width) % 3 == 0 ? 1e-6f : 1000.0f) : 0.3f);
    for (int h = 0; h < heads; ++h)
        for (int t = 0; t < tokens; ++t)
            std::copy_n(q.data() + (t * heads + h) * width, width,
                        packed.data() + guard + (h * total + first + t) * width);
    std::vector<uint16_t> kv(cached * width);
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = ggml_fp32_to_fp16(std::cos(i * 0.137) * 0.25f);
    std::vector<int32_t> ids(tokens * selected);
    for (int t = 0; t < tokens; ++t)
        for (int s = 0; s < selected; ++s)
            ids[t * selected + s] = (t == 0 && tokens > 1) || s % 11 == 0 ? -1 :
                (s % 13 == 0 ? cached + 7 : (s * 17 + t * 3) % cached);
    std::vector<float> reference(q.size() + 2 * guard, 12345);
    std::vector<float> actual(packed.size(), 12345), reference_heads(actual);
    Buffer dq(q.size() * sizeof(float)), dh(packed.size() * sizeof(float)), dk(kv.size() * sizeof(uint16_t));
    Buffer di(ids.size() * sizeof(int32_t)), dr(reference.size() * sizeof(float));
    Buffer da(actual.size() * sizeof(float)), drh(reference_heads.size() * sizeof(float));
    upload(dq, q); upload(dh, packed); upload(dk, kv); upload(di, ids);
    upload(dr, reference); upload(da, actual); upload(drh, reference_heads);
    check(cudaStreamSynchronize(nullptr), "head-major uploads complete");
    const auto run = [&](bool direct, bool transpose, bool prefetch = false) {
        struct PrefetchScope {
            bool old = kernels::glm_flash_prefetch();
            explicit PrefetchScope(bool value) { kernels::glm_flash_set_prefetch(value); }
            ~PrefetchScope() { kernels::glm_flash_set_prefetch(old); }
        } scope(prefetch);
        const auto* latent = static_cast<const uint16_t*>(dk.get());
        const auto* cells = sparse ? static_cast<const int32_t*>(di.get()) : nullptr;
        if (direct) {
            kernels::glm_flash_attn_head_major(static_cast<float*>(dh.get()) + guard, latent, cells,
                width, heads, 256, total, first, tokens, sparse ? selected : cached,
                cached, pos, static_cast<float*>(da.get()) + guard, stream);
        } else {
            if (transpose)
                kernels::glm_heads_major(static_cast<float*>(dh.get()) + guard, static_cast<float*>(dq.get()),
                                         tokens, heads, width, stream);
            kernels::glm_flash_attn(static_cast<float*>(dq.get()), latent, cells, width, heads, 256, tokens,
                sparse ? selected : cached, cached, pos, static_cast<float*>(dr.get()) + guard, stream);
            if (transpose)
                kernels::glm_heads_major(static_cast<float*>(dr.get()) + guard, static_cast<float*>(drh.get()) + guard,
                                         heads, tokens, width, stream);
        }
    };
    if (benchmark) {
        if (prefetch_check) run(false, false);
        struct Timer {
            cudaEvent_t begin = nullptr, end = nullptr;
            ~Timer() { if (begin) cudaEventDestroy(begin); if (end) cudaEventDestroy(end); }
        } timer;
        check(cudaEventCreate(&timer.begin), "layout timing begin");
        check(cudaEventCreate(&timer.end), "layout timing end");
        for (bool direct : {false, true, true, false}) {
            run(prefetch_check || direct, !prefetch_check, prefetch_check && direct);
            check(cudaStreamSynchronize(stream), "layout warmup");
            check(cudaEventRecord(timer.begin, stream), "layout begin");
            for (int repeat = 0; repeat < (prefetch_check ? 30 : 10); ++repeat)
                run(prefetch_check || direct, !prefetch_check, prefetch_check && direct);
            check(cudaEventRecord(timer.end, stream), "layout end");
            check(cudaEventSynchronize(timer.end), "layout timing sync");
            float ms = 0; check(cudaEventElapsedTime(&ms, timer.begin, timer.end), "layout elapsed");
            std::printf("Flash layout: heads=%d tokens=%d selected=%d head_major=%d transposes=%d prefetch=%d ms=%.6f\n",
                        heads, tokens, selected, prefetch_check || direct, prefetch_check || direct ? 0 : 2,
                        prefetch_check && direct, ms / (prefetch_check ? 30 : 10));
        }
        if (prefetch_check) run(true, false, true);
    } else { run(false, false); run(true, false, prefetch_check); }
    check(cudaStreamSynchronize(stream), "head-major result sync");
    check(cudaMemcpy(reference.data(), dr.get(), reference.size() * sizeof(float), cudaMemcpyDeviceToHost), "layout reference");
    if (prefetch_check) {
        run(false, false, true);
        check(cudaStreamSynchronize(stream), "token-major prefetch result sync");
        std::vector<float> token_prefetched(reference.size());
        check(cudaMemcpy(token_prefetched.data(), dr.get(), token_prefetched.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "token-major prefetch result");
        require(std::memcmp(token_prefetched.data(), reference.data(), reference.size() * sizeof(float)) == 0,
                "token-major KV prefetch changed attention bytes or guards");
    }
    check(cudaMemcpy(actual.data(), da.get(), actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "layout direct");
    for (int h = 0; h < heads; ++h)
        for (int t = 0; t < total; ++t) {
            auto* row = actual.data() + guard + (h * total + t) * width;
            if (t >= first && t < first + tokens)
                require(std::memcmp(row, reference.data() + guard + ((t - first) * heads + h) * width,
                                    width * sizeof(float)) == 0, "head-major addressing changed attention bytes");
            else for (int d = 0; d < width; ++d)
                require(row[d] == 12345, "head-major attention wrote outside its query window");
        }
    for (auto v : actual) require(std::isfinite(v), "head-major attention produced non-finite output");
    for (int i = 0; i < guard; ++i)
        require(actual[i] == 12345 && actual[actual.size() - 1 - i] == 12345, "head-major output guard changed");
    std::vector<float> unchanged(packed.size());
    check(cudaMemcpy(unchanged.data(), dh.get(), unchanged.size() * sizeof(float), cudaMemcpyDeviceToHost), "layout input");
    require(std::memcmp(unchanged.data(), packed.data(), packed.size() * sizeof(float)) == 0,
            "head-major attention changed its input");
}

void replay_attention(const char* path, cudaStream_t stream) {
    std::ifstream file(path, std::ios::binary);
    std::array<int32_t, 8> header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    require(file && header[0] == 0x474c4d41, "invalid attention replay header");
    const int layer = header[1], heads = header[2], width = header[3], qk = header[4];
    const int selected = header[5], cached = header[6], pos = header[7];
    require(heads > 0 && heads <= 128 && width == 512 && qk == 256 && selected > 0 && selected <= 4096 &&
            cached > 0 && cached <= 131072 && pos >= 0 && pos < cached, "unsupported attention replay shape");
    std::vector<float> q(heads * width), recorded(q.size()), scalar(q.size()), flashed(q.size());
    std::vector<uint16_t> kv(static_cast<size_t>(cached) * width);
    std::vector<int32_t> ids(selected);
    auto read = [&](auto& data) {
        file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(data[0]));
        require(static_cast<bool>(file), "incomplete attention replay payload");
    };
    read(q); read(kv); read(ids); read(recorded);
    require(file.peek() == std::char_traits<char>::eof(), "trailing attention replay payload");
    for (float x : q) require(std::isfinite(x), "non-finite replay query");
    for (uint16_t x : kv) require(std::isfinite(ggml_fp16_to_fp32(x)), "non-finite replay latent");
    for (int id : ids) require(id == -1 || (id >= 0 && id <= pos), "invalid replay selection");
    Buffer dq(q.size() * 4), dk(kv.size() * 2), di(ids.size() * 4), ds(q.size() * 4), df(q.size() * 4);
    upload(dq, q); upload(dk, kv); upload(di, ids);
    strata::kernels::glm_dsa_attn(static_cast<float*>(dq.get()), static_cast<uint16_t*>(dk.get()),
        static_cast<int32_t*>(di.get()), width, heads, qk, 1, selected, static_cast<float*>(ds.get()), stream);
    strata::kernels::glm_flash_attn(static_cast<float*>(dq.get()), static_cast<uint16_t*>(dk.get()),
        static_cast<int32_t*>(di.get()), width, heads, qk, 1, selected, cached, pos,
        static_cast<float*>(df.get()), stream);
    check(cudaStreamSynchronize(stream), "attention replay sync");
    check(cudaMemcpy(scalar.data(), ds.get(), scalar.size() * 4, cudaMemcpyDeviceToHost), "replay scalar output");
    check(cudaMemcpy(flashed.data(), df.get(), flashed.size() * 4, cudaMemcpyDeviceToHost), "replay Flash output");
    double score_max = -INFINITY, qmax = 0;
    for (float x : q) qmax = std::max(qmax, std::abs(static_cast<double>(x)));
    const auto reference = attention_reference(q, kv, ids, heads, 1, cached, selected, pos, true, qk, &score_max);
    const auto se = attention_error(scalar, reference), fe = attention_error(flashed, reference);
    const auto re = attention_error(recorded, reference);
    const auto delta = attention_error(flashed, std::vector<double>(scalar.begin(), scalar.end()));
    char score_text[64];
    if (std::isfinite(score_max)) std::snprintf(score_text, sizeof(score_text), "%.12g", score_max);
    else std::strcpy(score_text, "null");
    std::printf("{\"layer\":%d,\"heads\":%d,\"selected\":%d,\"qmax\":%.12g,\"score_max\":%s,"
                "\"scalar\":{\"rmse\":%.12g,\"nmse\":%.12g,\"max_abs\":%.12g},"
                "\"flash\":{\"rmse\":%.12g,\"nmse\":%.12g,\"max_abs\":%.12g},"
                "\"recorded_rmse\":%.12g,\"flash_scalar_rmse\":%.12g,"
                "\"recorded_matches_scalar\":%s,\"recorded_matches_flash\":%s}\n",
                layer, heads, selected, qmax, score_text, se.rmse, se.nmse, se.maximum,
                fe.rmse, fe.nmse, fe.maximum, re.rmse, delta.rmse,
                std::memcmp(recorded.data(), scalar.data(), recorded.size() * sizeof(float)) == 0 ? "true" : "false",
                std::memcmp(recorded.data(), flashed.data(), recorded.size() * sizeof(float)) == 0 ? "true" : "false");
    require(fe.nmse < 1e-12 && fe.maximum < 2e-5,
            "real-input Flash precision regression against double reference");
}

void pools_and_scores(cudaStream_t stream, int heads) {
    constexpr int kd = 128, kp = 4, count = 78, pools = count / kp, queries = 65;
    std::vector<float> k(count * kd), g(k.size()), ape(kp * kd), q(queries * heads * kd), w(queries * heads);
    for (size_t i = 0; i < k.size(); ++i) { k[i] = std::sin(i * 0.19); g[i] = std::cos(i * 0.07); }
    for (size_t i = 0; i < ape.size(); ++i) ape[i] = std::sin(i * 0.11) * 0.1f;
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::cos(i * 0.17) * 0.13f;
    for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin(i * 0.31) * 0.1f;
    Buffer dk(k.size() * 4), dg(g.size() * 4), da(ape.size() * 4), dq(q.size() * 4), dw(w.size() * 4);
    Buffer partial(2 * kp * kd * 4), reference(pools * kd * 4), actual(pools * kd * 4);
    upload(dk, k); upload(dg, g); upload(da, ape); upload(dq, q); upload(dw, w);
    check(cudaMemset(partial.get(), 0, 2 * kp * kd * 4), "partial init");
    int start = 0;
    for (const int chunk : {3, 1, 7, 2, 65}) {
        strata::kernels::glm_dsa_append(static_cast<float*>(dk.get()) + start * kd,
            static_cast<float*>(dg.get()) + start * kd, static_cast<float*>(da.get()), kd, kp, chunk, start,
            static_cast<float*>(partial.get()), static_cast<float*>(partial.get()) + kp * kd,
            static_cast<float*>(actual.get()), stream);
        start += chunk;
    }
    strata::kernels::glm_dsa_pool(static_cast<float*>(dk.get()), static_cast<float*>(dg.get()),
        static_cast<float*>(da.get()), kd, kp, pools, static_cast<float*>(reference.get()), stream);
    check(cudaStreamSynchronize(stream), "pool sync");
    std::vector<float> a(pools * kd), b(a.size()), tail(2 * kp * kd);
    check(cudaMemcpy(a.data(), actual.get(), a.size() * 4, cudaMemcpyDeviceToHost), "pooled actual");
    check(cudaMemcpy(b.data(), reference.get(), b.size() * 4, cudaMemcpyDeviceToHost), "pooled reference");
    near(a, b, 1e-6f, "batched pool append");
    check(cudaMemcpy(tail.data(), partial.get(), tail.size() * 4, cudaMemcpyDeviceToHost), "partial output");
    for (int slot = 0; slot < kp; ++slot) {
        const int last = count - 1 - ((count - 1 - slot + kp) % kp);
        for (int d = 0; d < kd; ++d)
            require(tail[slot * kd + d] == k[last * kd + d] &&
                    tail[kp * kd + slot * kd + d] == g[last * kd + d], "partial pool lost a boundary row");
    }
    Buffer sr(queries * pools * 4), sa(queries * pools * 4);
    strata::kernels::glm_dsa_score(static_cast<float*>(dq.get()), static_cast<float*>(reference.get()),
        static_cast<float*>(dw.get()), kd, heads, queries, pools, static_cast<float*>(sr.get()), stream);
    strata::kernels::glm_dsa_score_batch(static_cast<float*>(dq.get()), static_cast<float*>(reference.get()),
        static_cast<float*>(dw.get()), kd, heads, queries, pools, static_cast<float*>(sa.get()), stream);
    check(cudaStreamSynchronize(stream), "score sync");
    a.resize(queries * pools); b.resize(a.size());
    check(cudaMemcpy(a.data(), sa.get(), a.size() * 4, cudaMemcpyDeviceToHost), "score actual");
    check(cudaMemcpy(b.data(), sr.get(), b.size() * 4, cudaMemcpyDeviceToHost), "score reference");
    require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
            "parallel indexer scores changed scalar arithmetic");
}

void pool_commit_prefixes(cudaStream_t stream) {
    constexpr int kd = 128, kp = 4, partial = kd * kp, floats = 2 * partial + 4 * kd;
    std::vector<float> key(16 * kd), gate(key.size()), ape(partial);
    for (size_t i = 0; i < key.size(); ++i) { key[i] = std::sin(i * 0.17); gate[i] = std::cos(i * 0.13); }
    for (size_t i = 0; i < ape.size(); ++i) ape[i] = std::sin(i * 0.11) * 0.1f;
    Buffer dk(key.size() * 4), dg(gate.size() * 4), da(ape.size() * 4);
    Buffer saved(floats * 4), reference(floats * 4), actual(floats * 4);
    upload(dk, key); upload(dg, gate); upload(da, ape);
    auto append = [&](Buffer& state, int base, int count) {
        auto* p = static_cast<float*>(state.get());
        strata::kernels::glm_dsa_append(static_cast<float*>(dk.get()) + base * kd,
            static_cast<float*>(dg.get()) + base * kd, static_cast<float*>(da.get()), kd, kp, count, base,
            p, p + partial, p + 2 * partial, stream);
    };
    for (int base : {1, 2, 3, 4, 5, 7}) {
        check(cudaMemsetAsync(saved.get(), 0, floats * 4, stream), "pool commit initial state");
        append(saved, 0, base);
        for (int kept = 1; kept <= 8; ++kept) {
            check(cudaMemcpyAsync(reference.get(), saved.get(), floats * 4, cudaMemcpyDeviceToDevice, stream),
                  "pool commit reference checkpoint");
            check(cudaMemcpyAsync(actual.get(), saved.get(), floats * 4, cudaMemcpyDeviceToDevice, stream),
                  "pool commit speculative checkpoint");
            append(reference, base, kept);
            append(actual, base, 8);
            check(cudaMemcpyAsync(actual.get(), saved.get(), 2 * partial * 4, cudaMemcpyDeviceToDevice, stream),
                  "pool commit partial restore");
            append(actual, base, kept);
            const int valid = 2 * partial + (base + kept) / kp * kd;
            std::vector<float> expected(valid), result(valid);
            check(cudaMemcpyAsync(expected.data(), reference.get(), valid * 4, cudaMemcpyDeviceToHost, stream),
                  "pool commit reference output");
            check(cudaMemcpyAsync(result.data(), actual.get(), valid * 4, cudaMemcpyDeviceToHost, stream),
                  "pool commit actual output");
            check(cudaStreamSynchronize(stream), "pool commit sync");
            require(std::memcmp(expected.data(), result.data(), valid * 4) == 0,
                    "accepted pool prefix differs after speculative rollback");
        }
    }
}

void sparse_mtp_case(const ModelFile& file, bool reject, int tail, int prefix, int depth, bool flash,
                     bool head_major = false) {
    Options plain;
    plain.dsa = true; plain.dsa_tail = tail; plain.flash_attn = flash;
    plain.dsa_score = DsaScore::Parallel;
    plain.full_layer_prefill = true; plain.prefill = 64;
    plain.prefill_reference = true;
    if (head_major) {
        plain.prefill_reference = false;
        plain.prefill_no_pipeline = true; plain.prefill_group_layers = 1; plain.prefill_prefetch_layers = 0;
        plain.mla_attention_layout = MlaAttentionLayout::HeadMajor;
    }
    plain.max_context = prefix + 32; plain.prefill_gpu_mib = 64;
    plain.prefill_reserve_mib = 128;
    auto options = plain; options.mtp = depth;
    std::vector<int32_t> prompt(prefix);
    for (int i = 0; i < prefix; ++i) prompt[i] = 1 + i % 17;
    const auto target_image = full_layer_prefill(file, plain, prompt, {});
    const auto draft_image = full_layer_prefill(file, options, prompt, {});
    near(draft_image.state.logits, target_image.state.logits, 1e-5f, "DSA MTP prefill target logits");
    require(draft_image.state.bytes() == host_state_bytes(file.geometry(), prefix, true, false, true),
            "DSA MTP full-layer handoff budget");
    DecodeExecutor target(file, plain, &target_image.state);
    DecodeExecutor draft(file, options, &draft_image.state);
    DecodeExecutor serial(file, options, &draft_image.state);
    const auto before = draft.mtp_stats();
    const auto expected = target.generate(10);
    const auto actual = draft.generate(10);
    require(actual == expected && actual.size() == 10, "DSA MTP changed greedy output");
    for (int token : expected) serial.token(token);
    const auto stats = draft.mtp_stats();
    require(stats.offered && stats.verify_batches, "DSA MTP did not verify drafts");
    require(reject ? stats.replays > 0 : stats.accepted > 0, "DSA MTP missing forced acceptance/rejection");
    require(stats.dsa_replayed_layers == stats.replays, "DSA MTP did not replay the sparse trunk layer");
    require(stats.replayed_layers == stats.replays * 3, "DSA MTP recurrent replay count");
    require(stats.target_batches - before.target_batches == stats.verify_batches + (reject ? 1 : 0),
            "DSA MTP replay reran target projections");
    auto committed = draft.export_state(), reference = serial.export_state();
    near(committed.mtp.indexer, reference.mtp.indexer, 1e-5f, "NextN partial/pooled state after MTP");
    near(committed.carry_hidden, reference.carry_hidden, 1e-5f, "DSA MTP target hidden carry");
    for (size_t l = 0; l < committed.layers.size(); ++l) {
        if (glm_is_kda_layer(file.geometry(), l))
            near(committed.layers[l].recurrent, reference.layers[l].recurrent, 1e-5f, "DSA MTP recurrent commit");
        else {
            near(committed.layers[l].indexer, reference.layers[l].indexer, 1e-5f, "DSA accepted pool state");
            glm_fixture::near_fp16(committed.layers[l].kv, reference.layers[l].kv, 1e-5f, "DSA MTP latent commit");
        }
    }
    near(draft.token(17), target.token(17), 1e-5f, "DSA MTP continuation");
    serial.token(17);
    if (depth == 3 && prefix == 7) {
        auto cached = options;
        cached.gpu_experts_mib = 2; cached.gpu_warm = true;
        cached.gpu_heat_quotas = true; cached.gpu_stage_slots = 1;
        const auto warmed = full_layer_prefill(file, cached, prompt, {});
        DecodeExecutor combined(file, cached, &warmed.state);
        require(combined.generate(10) == expected, "cached sparse MTP changed fixture greedy output");
        const auto stats = combined.cache_stats();
        require(stats.warmed && stats.pinned_bytes, "sparse MTP lost prompt heat or pinned staging");
        const auto appended = full_layer_prefill(file, cached, {6, 7, 8}, combined.export_state());
        DecodeExecutor resumed(file, cached, &appended.state);
        require(resumed.generate(5).size() == 5, "cached sparse MTP append failed");
        resumed.export_state();
    }
    if (depth == 3 && (prefix == 3 || prefix == 2305)) {
        const std::vector<int32_t> suffix{6, 7, 8};
        const auto appended = full_layer_prefill(file, options, suffix, draft.export_state());
        const auto serial_appended = full_layer_prefill(file, options, suffix, serial.export_state());
        near(appended.state.mtp.indexer, serial_appended.state.mtp.indexer, 1e-5f,
             "DSA MTP appended full-layer draft pools");
        DecodeExecutor resumed(file, options, &appended.state);
        DecodeExecutor restored_serial(file, options, &serial_appended.state);
        const auto tokens = resumed.generate(5);
        for (int token : tokens) restored_serial.token(token);
        near(resumed.export_state().mtp.indexer, restored_serial.export_state().mtp.indexer, 1e-5f,
             "restored DSA MTP partial pools");
        resumed.reset();
        serial.reset();
        resumed.prefill({1, 2, 3});
        serial.prefill({1, 2, 3});
        const auto reset_tokens = resumed.generate(10);
        for (int token : reset_tokens) serial.token(token);
        near(resumed.export_state().mtp.indexer, serial.export_state().mtp.indexer, 1e-5f,
             "DSA MTP reset leaked draft pools");
    }
}

void runtime(bool mtp_check) {
    const auto path = std::filesystem::temp_directory_path() /
        ("glm5-dsa-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    struct Cleanup {
        const std::filesystem::path& path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
            if (error) std::fprintf(stderr, "attention fixture cleanup: %s\n", error.message().c_str());
        }
    } cleanup{path};
    glm_fixture::write_model(path, false, true);
    if (!mtp_check) {
        ModelFile file(path.string());
        file.validate_dsa();
        Options options;
        options.full_layer_prefill = true; options.dsa = true; options.prefill = 64;
        options.prefill_reference = true;
        options.max_context = 2400; options.prefill_gpu_mib = 64; options.gpu_reserve_mib = 128;
        options.prefill_reserve_mib = 128;
        std::vector<int32_t> prompt(2305);
        for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = 1 + i % 17;
        auto fast = options; fast.flash_attn = true;
        const auto reference = full_layer_prefill(file, options, prompt, {});
        const auto flashed = full_layer_prefill(file, fast, prompt, {});
        auto direct = fast;
        direct.prefill_reference = false; direct.reference = false;
        direct.prefill_group_layers = 1; direct.prefill_prefetch_layers = 0; direct.prefill_no_pipeline = true;
        direct.mla_attention_layout = MlaAttentionLayout::HeadMajor;
        direct.flash_kv_load = FlashKvLoad::Direct;
        auto token_major = direct; token_major.mla_attention_layout = MlaAttentionLayout::TokenMajor;
        const auto token_image = full_layer_prefill(file, token_major, prompt, {});
        const auto head_image = full_layer_prefill(file, direct, prompt, {});
        auto prefetch = direct; prefetch.flash_kv_load = FlashKvLoad::Prefetch;
        const auto prefetched_image = full_layer_prefill(file, prefetch, prompt, {});
        require(token_image.state.logits == head_image.state.logits &&
                token_image.state.carry_hidden == head_image.state.carry_hidden &&
                token_image.state.position == head_image.state.position, "head-major MLA changed complete logits");
        require(prefetched_image.state.logits == head_image.state.logits &&
                prefetched_image.state.carry_hidden == head_image.state.carry_hidden &&
                prefetched_image.state.position == head_image.state.position, "KV prefetch changed complete logits");
        for (size_t layer = 0; layer < token_image.state.layers.size(); ++layer)
            require(token_image.state.layers[layer].kv == head_image.state.layers[layer].kv &&
                    token_image.state.layers[layer].indexer == head_image.state.layers[layer].indexer &&
                    token_image.state.layers[layer].recurrent == head_image.state.layers[layer].recurrent &&
                    prefetched_image.state.layers[layer].kv == head_image.state.layers[layer].kv &&
                    prefetched_image.state.layers[layer].indexer == head_image.state.layers[layer].indexer &&
                    prefetched_image.state.layers[layer].recurrent == head_image.state.layers[layer].recurrent,
                    "head-major MLA or KV prefetch changed persistent state bytes");
        auto dense_token = token_major; dense_token.dsa = false; dense_token.dsa_tail = -1;
        auto dense_head = dense_token; dense_head.mla_attention_layout = MlaAttentionLayout::HeadMajor;
        const std::vector<int32_t> short_prompt{1, 2, 3, 4, 5, 6, 7, 8, 9};
        const auto dense_reference = full_layer_prefill(file, dense_token, short_prompt, {});
        const auto dense_direct = full_layer_prefill(file, dense_head, short_prompt, {});
        dense_head.flash_kv_load = FlashKvLoad::Prefetch;
        const auto dense_prefetched = full_layer_prefill(file, dense_head, short_prompt, {});
        require(dense_reference.state.logits == dense_direct.state.logits,
                "head-major dense Flash changed complete logits");
        require(dense_prefetched.state.logits == dense_direct.state.logits &&
                dense_prefetched.state.carry_hidden == dense_direct.state.carry_hidden,
                "dense KV prefetch changed complete logits");
        for (size_t layer = 0; layer < dense_reference.state.layers.size(); ++layer)
            require(dense_reference.state.layers[layer].kv == dense_direct.state.layers[layer].kv &&
                    dense_reference.state.layers[layer].recurrent == dense_direct.state.layers[layer].recurrent &&
                    dense_prefetched.state.layers[layer].kv == dense_direct.state.layers[layer].kv &&
                    dense_prefetched.state.layers[layer].recurrent == dense_direct.state.layers[layer].recurrent,
                    "head-major dense Flash or KV prefetch changed persistent state bytes");
        auto instrumented = fast; instrumented.probe = true;
        const auto measured = full_layer_prefill(file, instrumented, prompt, {});
        require(measured.state.logits == flashed.state.logits && measured.stats.probe.forced_waits == 0 &&
                measured.stats.probe.event_pairs <= 256, "fine MLA probes changed logits or forced unbounded waits");
        for (auto phase : {ProbePhase::MlaIndexProject, ProbePhase::MlaIndexPool,
                           ProbePhase::MlaIndexScore, ProbePhase::MlaIndexSelect, ProbePhase::MlaAttention})
            require(measured.stats.probe.phases[static_cast<size_t>(phase)].stream_calls > 0,
                    "sparse MLA probe did not run");
        auto scheduled = fast;
        scheduled.prefill_auto = true;
        scheduled.prefill_group_layers = 2;
        scheduled.prefill_prefetch_layers = 1;
        scheduled.prefill_pipeline = true;
        const auto optimized = full_layer_prefill(file, scheduled, prompt, {});
        near(optimized.state.logits, flashed.state.logits, 1e-5f, "scheduled sparse Flash logits");
        require(optimized.stats.groups == 2 && optimized.stats.max_group_layers == 2 &&
                optimized.stats.prefetched_layers == 1 && optimized.stats.fused_head,
                "long sparse test did not exercise grouped/prefetched execution");
        for (size_t l = 0; l < optimized.state.layers.size(); ++l) {
            if (!flashed.state.layers[l].indexer.empty())
                near(optimized.state.layers[l].indexer, flashed.state.layers[l].indexer, 1e-5f,
                     "scheduled sparse pooled state");
            if (!flashed.state.layers[l].recurrent.empty())
                near(optimized.state.layers[l].recurrent, flashed.state.layers[l].recurrent, 1e-5f,
                     "scheduled KDA state");
        }
        DecodeExecutor optimized_decode(file, scheduled, &optimized.state);
        DecodeExecutor flash_decode(file, fast, &flashed.state);
        near(optimized_decode.token(3), flash_decode.token(3), 1e-5f, "scheduled sparse continuation");
        auto scalar_flash = fast; scalar_flash.dsa_score = DsaScore::Scalar;
        auto parallel_scalar = options; parallel_scalar.dsa_score = DsaScore::Parallel;
        const auto mixed_flash = full_layer_prefill(file, scalar_flash, prompt, {});
        const auto mixed_scalar = full_layer_prefill(file, parallel_scalar, prompt, {});
        near(mixed_flash.state.logits, reference.state.logits, 1e-5f, "scalar-score Flash logits");
        near(mixed_scalar.state.logits, reference.state.logits, 1e-5f, "parallel-score scalar logits");
        require(!strata::kernels::glm_dsa_parallel_score(), "indexer score mode leaked");
        require(!strata::kernels::glm_dsa_enabled() && !strata::kernels::glm_flash_enabled(), "attention mode leaked");
        near(flashed.state.logits, reference.state.logits, 1e-5f, "sparse Flash full-layer logits");
        require(flashed.state.layers[3].indexer == reference.state.layers[3].indexer, "Flash changed pooled state");
        require(flashed.state.bytes() == host_state_bytes(file.geometry(), prompt.size(), false, false, true),
                "DSA handoff budget mismatch");
        DecodeExecutor left(file, fast, &flashed.state), right(file, options, &reference.state);
        near(left.token(3), right.token(3), 1e-5f, "DSA decode continuation");
        const auto image = left.export_state();
        DecodeExecutor restored(file, fast, &image);
        const auto copied = restored.export_state();
        require(copied.layers[3].indexer == image.layers[3].indexer, "indexer raw transfer changed bits");
        require(restored.token(4) == left.token(4), "indexer partial-pool continuation changed bits");
        Model model(path.string(), fast);
        model.prefill({1, 2, 3});
        model.prepare_decode();
        model.token(4);
        model.prefill({5, 6, 7});
        model.prepare_decode();
        model.generate(2);
        model.reset();
        model.prefill({1, 2});
        auto without_tail = fast; without_tail.dsa_tail = 0;
        auto scalar = without_tail; scalar.flash_attn = false;
        Model tail_flash(path.string(), without_tail), tail_scalar(path.string(), scalar);
        near(tail_flash.prefill({1, 2, 3}), tail_scalar.prefill({1, 2, 3}), 1e-5f, "empty pooled prefix");
        for (const int token : {4, 5, 6, 7})
            near(tail_flash.token(token), tail_scalar.token(token), 1e-5f, "tail-off pool continuation");
        auto cached = fast;
        cached.gpu_experts_mib = 2;
        cached.gpu_warm = true;
        cached.gpu_heat_quotas = true;
        cached.gpu_stage_slots = 1;
        Model combined(path.string(), cached);
        combined.prefill({1, 2, 3, 4, 5});
        combined.prepare_decode();
        require(combined.cache_stats().warmed > 0, "DSA handoff lost expert heat");
        combined.generate(2);
        combined.prefill({6, 7, 8});
        combined.prepare_decode();
        require(combined.cache_stats().warmed > 0 && combined.cache_stats().pinned_bytes > 0,
                "DSA append lost asynchronous expert cache");
        auto invalid = fast; invalid.mtp = 8;
        bool refused = false;
        try { Model unsupported(path.string(), invalid); }
        catch (const std::invalid_argument&) { refused = true; }
        require(refused, "out-of-range MTP must be refused");
    }
    if (!mtp_check) {
        const std::filesystem::path input_path = path.string() + ".eval-input";
        const std::filesystem::path output_path = path.string() + ".eval-output";
        const auto model_name = path.string(), input_name = input_path.string(), output_name = output_path.string();
        Cleanup input_cleanup{input_path}, output_cleanup{output_path};
        const std::array<int32_t, 5> header{0x474c4d45, 1, 3, 4, 0};
        const std::array<int32_t, 7> tokens{1, 2, 3, 4, 5, 6, 7};
        {
            std::ofstream input(input_path, std::ios::binary);
            input.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
            input.write(reinterpret_cast<const char*>(tokens.data()), sizeof(tokens));
            require(static_cast<bool>(input), "cannot write teacher-forced fixture");
        }
        glm_evaluation::teacher_forced(model_name.c_str(), input_name.c_str(), output_name.c_str());
        std::ifstream output(output_path, std::ios::binary);
        std::array<int32_t, 6> written{};
        std::array<int32_t, 7> conditioning{};
        output.read(reinterpret_cast<char*>(written.data()), sizeof(written));
        output.read(reinterpret_cast<char*>(conditioning.data()), sizeof(conditioning));
        require(written == std::array<int32_t, 6>{0x474c4d50, 1, 32, 4, 3, 0} && conditioning == tokens,
                "teacher-forced header or conditioning changed");
        for (int i = 0; i < 4; ++i) {
            int32_t target = -1;
            std::array<float, 32> logits{};
            output.read(reinterpret_cast<char*>(&target), sizeof(target));
            output.read(reinterpret_cast<char*>(logits.data()), sizeof(logits));
            require(output && target == tokens[3 + i], "teacher-forced position or target mismatch");
            for (float value : logits) require(std::isfinite(value), "non-finite scored fixture logit");
        }
        require(output.peek() == std::char_traits<char>::eof(), "teacher-forced output has extra rows");
        bool refused = false;
        try { glm_evaluation::teacher_forced(model_name.c_str(), input_name.c_str(), output_name.c_str()); }
        catch (const std::runtime_error& e) {
            refused = std::string(e.what()).find("already exists") != std::string::npos;
        }
        require(refused, "teacher-forced output must not overwrite an existing file");
    }
    for (bool reject : {false, true}) {
        glm_fixture::write_model(path, reject, true);
        for (int tail : {0, 1}) {
            ModelFile file(path.string(), tail);
            file.validate_mtp();
            file.validate_dsa(true);
            for (int prefix : {1, 2, 3, 4, 7, 2305}) {
                if (mtp_check && prefix == 2305) continue;
                for (int depth : {1, 3, 7})
                    for (bool flash : {false, true}) {
                        try {
                            sparse_mtp_case(file, reject, tail, prefix, depth, flash);
                        }
                        catch (const std::exception& e) {
                            throw std::runtime_error("MTP case reject=" + std::to_string(reject) +
                                " tail=" + std::to_string(tail) + " prefix=" + std::to_string(prefix) +
                                " depth=" + std::to_string(depth) + " flash=" + std::to_string(flash) +
                                ": " + e.what());
                        }
                    }
                for (int depth : {1, 3, 7})
                    sparse_mtp_case(file, reject, tail, prefix, depth, true, true);
            }
        }
    }
    std::filesystem::remove(path);
}
}  // namespace

int main(int argc, char** argv) {
    const bool layout_check = argc == 2 && std::strcmp(argv[1], "--layout-check") == 0;
    const bool layout_bench = argc == 2 && std::strcmp(argv[1], "--layout-bench") == 0;
    const bool prefetch_check = argc == 2 && std::strcmp(argv[1], "--prefetch-check") == 0;
    const bool prefetch_bench = argc == 2 && std::strcmp(argv[1], "--prefetch-bench") == 0;
    const bool mtp_check = argc == 2 && std::strcmp(argv[1], "--mtp-check") == 0;
    const bool precision_check = argc == 2 && std::strcmp(argv[1], "--precision-check") == 0;
    const bool replay = argc == 3 && std::strcmp(argv[1], "--replay-attention") == 0;
    const bool evaluate = argc == 5 && std::strcmp(argv[1], "--evaluate") == 0;
    const bool projection = argc == 4 && std::strcmp(argv[1], "--projection-reference") == 0;
    if (argc != 1 && !mtp_check && !precision_check && !replay && !evaluate && !projection &&
        !layout_check && !layout_bench && !prefetch_check && !prefetch_bench) {
        std::fputs("Usage: glm5_attention_test [--mtp-check | --precision-check | --replay-attention FILE | "
                   "--evaluate MODEL INPUT OUTPUT | --projection-reference MODEL PAIR | --layout-check | --layout-bench | "
                   "--prefetch-check | --prefetch-bench]\n", stderr);
        return 1;
    }
    cudaStream_t stream = nullptr;
    try {
        ggml_cpu_init();
        if (projection) {
            glm_evaluation::projection_delta(argv[2], argv[3]);
            return 0;
        }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        if (!strata::kernels::glm_flash_supported()) return 77;
        if (evaluate) {
            glm_evaluation::teacher_forced(argv[2], argv[3], argv[4]);
            return 0;
        }
        check(cudaStreamCreate(&stream), "attention stream");
        if (replay) {
            replay_attention(argv[2], stream);
            check(cudaStreamDestroy(stream), "attention replay stream destroy");
            return 0;
        }
        if (layout_check || layout_bench || prefetch_check || prefetch_bench) {
            if (prefetch_bench) {
                for (int tokens : {1, 17, 64})
                    head_major_case(64, tokens, 50000, 2048, true, 0, stream, true, false, true);
            } else if (layout_bench) head_major_case(64, 64, 50000, 2048, true, 0, stream, true);
            else {
                for (int heads : {1, 2, 16, 17, 32, 64})
                    for (int tokens : {1, 3, 17})
                        for (bool sparse : {false, true})
                            for (int stress : {0, 1, 2})
                                head_major_case(heads, tokens, 257, 67, sparse, stress, stream, false, false, prefetch_check);
                head_major_case(64, 1, 4096, 2048, true, 1, stream, false, true, prefetch_check);
                if (prefetch_check) {
                    for (int selected : {1, 15, 16, 17, 31, 32, 33, 63, 64, 65})
                        head_major_case(17, 3, 257, selected, true, 1, stream, false, false, true);
                    head_major_case(17, 1, 1, 1, false, 0, stream, false, false, true);
                }
                Buffer dummy(4);
                bool refused = false;
                try {
                    kernels::glm_flash_attn_head_major(static_cast<float*>(dummy.get()),
                        static_cast<uint16_t*>(dummy.get()), nullptr, 512, 16, 256, 3, 3, 1, 1, 1, 0,
                        static_cast<float*>(dummy.get()), stream);
                } catch (const std::invalid_argument&) { refused = true; }
                require(refused, "head-major Flash must reject a window outside its batch");
            }
            check(cudaStreamDestroy(stream), "layout stream destroy");
            std::puts("glm5 Flash layout: exact bytes, query windows, tails, masks and guards passed");
            return 0;
        }
        score_scope();
        pool_commit_prefixes(stream);
        if (!mtp_check) {
            for (int heads : {1, 4, 32, 33, 64}) pools_and_scores(stream, heads);
            flash_case(2, 7, 131, 35, true, stream);
            flash_case(17, 3, 2051, 2051, true, stream);
            flash_case(64, 1, 2048, 67, true, stream);
            flash_case(16, 65, 257, 67, false, stream);
            flash_case(64, 1, 4096, 2048, true, stream, true);
            head_major_case(17, 3, 257, 67, true, 1, stream);
            head_major_case(16, 17, 257, 33, false, 0, stream);
        }
        if (!precision_check) runtime(mtp_check);
        check(cudaStreamDestroy(stream), "attention stream destroy");
        if (precision_check)
            std::puts("glm5 attention precision: CPU reference, causal masks and pool commit checks passed");
        else if (mtp_check)
            std::puts("glm5 sparse MTP: pool commits, acceptance/rejection and state handoff passed");
        else
            std::puts("glm5 native attention: CPU reference, causal sparse masks, pool boundaries and state handoff passed");
        return 0;
    } catch (const std::exception& e) {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        std::fprintf(stderr, "glm5 attention: %s\n", e.what());
        return 1;
    }
}
