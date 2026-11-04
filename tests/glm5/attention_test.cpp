#include "runtime_fixture.hpp"
#include "gpu_support.hpp"
#include "attention_scope.hpp"
#include "prefill.hpp"
#include "decode.hpp"
#include "strata/glm5/layer.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>

namespace {
using namespace strata::glm5;
using glm_fixture::require;
using glm_fixture::near;

template<class T> void upload(Buffer& b, const std::vector<T>& v) {
    check(cudaMemcpy(b.get(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "attention test upload");
}

void score_scope() {
    Options outer;
    outer.dsa = true; outer.flash_attn = true;
    {
        AttentionScope automatic(outer);
        require(strata::kernels::glm_dsa_parallel_score(), "Flash automatic scorer is not parallel");
        auto inner = outer; inner.dsa_score = DsaScore::Scalar;
        {
            AttentionScope scalar(inner);
            require(!strata::kernels::glm_dsa_parallel_score(), "scalar score override ignored");
        }
        require(strata::kernels::glm_dsa_parallel_score(), "nested score scope was not restored");
    }
    require(!strata::kernels::glm_dsa_parallel_score(), "score scope leaked");
    outer.flash_attn = false;
    {
        AttentionScope automatic(outer);
        require(!strata::kernels::glm_dsa_parallel_score(), "non-Flash automatic scorer changed");
    }
    outer.dsa_score = DsaScore::Parallel;
    {
        AttentionScope parallel(outer);
        require(strata::kernels::glm_dsa_parallel_score(), "non-Flash parallel override ignored");
    }
}

void flash_case(int heads, int tokens, int cached, int selected, bool sparse, cudaStream_t stream) {
    constexpr int width = 512;
    const int pos = cached - tokens;
    std::vector<float> q(tokens * heads * width), result(q.size() + 32, 12345);
    std::vector<uint16_t> kv(cached * width);
    std::vector<int32_t> ids(tokens * selected);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = std::sin(i * 0.173) * (i < static_cast<size_t>(heads * width) ? 1e-6f : 0.3f);
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = ggml_fp32_to_fp16(std::cos(i * 0.137) * 0.25f);
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
    double error = 0, power = 0, maximum_error = 0;
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
                scores[s] = dot / 16;
                maximum = std::max(maximum, scores[s]);
            }
            for (auto& s : scores) { s = std::exp(s - maximum); sum += s; }
            for (int d = 0; d < width; ++d) {
                double expected = 0;
                for (size_t s = 0; s < valid.size(); ++s)
                    expected += scores[s] * ggml_fp16_to_fp32(kv[valid[s] * width + d]);
                if (sum) expected /= sum;
                const double actual = result[16 + (t * heads + h) * width + d];
                require(std::isfinite(actual), "non-finite attention output");
                error += (actual - expected) * (actual - expected);
                power += expected * expected;
                maximum_error = std::max(maximum_error, std::abs(actual - expected));
            }
        }
    }
    require(error / std::max(power, 1e-12) < 1e-7 && maximum_error < 2e-4,
            "native Flash Attention differs from double CPU reference");
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

void sparse_mtp_case(const ModelFile& file, bool reject, int tail, int prefix, int depth, bool flash) {
    Options plain;
    plain.dsa = true; plain.dsa_tail = tail; plain.flash_attn = flash;
    plain.dsa_score = DsaScore::Parallel;
    plain.full_layer_prefill = true; plain.prefill = 64;
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
        options.max_context = 2400; options.prefill_gpu_mib = 64; options.gpu_reserve_mib = 128;
        options.prefill_reserve_mib = 128;
        std::vector<int32_t> prompt(2305);
        for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = 1 + i % 17;
        auto fast = options; fast.flash_attn = true;
        const auto reference = full_layer_prefill(file, options, prompt, {});
        const auto flashed = full_layer_prefill(file, fast, prompt, {});
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
    for (bool reject : {false, true}) {
        glm_fixture::write_model(path, reject, true);
        for (int tail : {0, 1}) {
            ModelFile file(path.string(), tail);
            file.validate_mtp();
            file.validate_dsa(true);
            for (int prefix : {1, 2, 3, 4, 7, 2305}) {
                if (mtp_check && prefix == 2305) continue;
                for (int depth : {1, 3, 7})
                    for (bool flash : {false, true})
                        sparse_mtp_case(file, reject, tail, prefix, depth, flash);
            }
        }
    }
    std::filesystem::remove(path);
}
}  // namespace

int main(int argc, char** argv) {
    const bool mtp_check = argc == 2 && std::strcmp(argv[1], "--mtp-check") == 0;
    if (argc != 1 && !mtp_check) {
        std::fputs("Usage: glm5_attention_test [--mtp-check]\n", stderr);
        return 1;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaStream_t stream = nullptr;
    try {
        ggml_cpu_init();
        if (!strata::kernels::glm_flash_supported()) return 77;
        check(cudaStreamCreate(&stream), "attention stream");
        score_scope();
        pool_commit_prefixes(stream);
        if (!mtp_check) {
            for (int heads : {1, 4, 32, 33, 64}) pools_and_scores(stream, heads);
            flash_case(2, 7, 131, 35, true, stream);
            flash_case(17, 3, 2051, 2051, true, stream);
            flash_case(64, 1, 2048, 67, true, stream);
            flash_case(16, 65, 257, 67, false, stream);
        }
        runtime(mtp_check);
        check(cudaStreamDestroy(stream), "attention stream destroy");
        std::puts("glm5 native attention: CPU reference, causal sparse masks, pool boundaries and state handoff passed");
        return 0;
    } catch (const std::exception& e) {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        std::fprintf(stderr, "glm5 attention: %s\n", e.what());
        return 1;
    }
}
