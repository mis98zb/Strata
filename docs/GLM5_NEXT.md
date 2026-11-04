# Experimental GLM-5.3-Flash engine

This branch starts a standalone `glm5-next` engine, following the directory and
build separation used by DeepSeek. It is not a released model family or a desktop
installer option. No performance or hardware compatibility is claimed by this
port beyond the specific checks and measurements below.

The GLM block, six GPU kernel files and GPU parity fixtures come from
[Strata PR #1315](https://github.com/Niko1221/Strata/pull/1315), at commit
`7dbde61cb46a936b68f06e6053d682cc992e9cb2`. That PR is closed and was not merged.
Its reported measurements are not measurements of this branch.

The prefill and MTP semantics were also checked against the local llama.cpp
checkout at `e129d48751534b1510619d2d719bfd6771827d9c`, especially
`src/models/glm5-next.cpp` and `common/speculative.cpp`. The reference checkout
is not modified or required at runtime.

The local Project Maya reference is v1.1.49,
`26aacb6bcea9333e4f408a68ee60c4c1c316bd68`. Its `src/core/glm_mtp.cu`,
`glm_mtp_rows.cu` and `glm_window.cu` provide the draft/verify/commit reference.
This port keeps llama.cpp's cache coordinates: row `p` pairs `h[p-1]` with
`x[p]`. Maya documents the same logical pair as entry `p-1`; its indexing and
prompt-cache convention are not copied interchangeably.

## Code layout

- [src/glm5](../src/glm5): standalone entry point, model runtime, GGUF checks,
  layer composition and projection helpers.
- [src/glm5/kernels](../src/glm5/kernels): the imported mHC, KDA, delta-rule,
  MLA, DSA and elementwise kernels.
- [include/strata/glm5](../include/strata/glm5): the family-specific interfaces.
- [tools/glm5](../tools/glm5): encode a formatted prompt and decode token IDs.
- [tests/glm5](../tests/glm5): host format checks and the imported GPU parity test.

The host code uses `strata::glm5`. The imported math entry points keep their
`strata::kernels::glm_*` names, in a separate `strata_glm5_kernels` library.
The existing Qwen and DeepSeek execution paths are not changed. The new build
option is off by default.

The model dispatcher is [model.cpp](../src/glm5/model.cpp).
[decode.cpp](../src/glm5/decode.cpp) owns resident decode and MTP verification.
[prefill.cpp](../src/glm5/prefill.cpp) owns the independent full-layer path.
They share [weights.cpp](../src/glm5/weights.cpp) and the GPU expert arithmetic,
but exchange only host-side KDA, MLA and NextN state through
[state_transfer.cpp](../src/glm5/state_transfer.cpp).

The resident decoder reads the original GGUF shards directly. There is no GLM-specific
canonical pack: dense projections stay in their supported native GPU format,
small tensors are converted to FP32, and routed experts stay mapped on the CPU
unless the optional GPU tier holds them.
CPU experts use ggml's initialized quantizers and dot products, with the GLM clamp
applied after SiLU. CPU misses run serially by default. Optional CPU parallelism
is described below; it does not reproduce the PR's optimized expert pool.

## Optional acceleration

Acceleration paths are off by default. They can be enabled separately or together.

### Independent full-layer prefill

`--prefill-full-layer` selects the independent executor. It does not construct
the resident decode engine before prefill:

1. Keep prompt activations and completed layer states in host memory.
2. Load one complete layer's projections and all its routed expert matrices
   into a temporary GPU bank.
3. Process every prompt microbatch through that layer, preserving sequential
   KDA updates and causal MLA positions.
4. Copy its final state to the host and release the entire bank before loading
   the next layer.
5. Run the output head as a separate bank. With MTP enabled, run NextN prompt
   catch-up in its own complete-layer bank, using every target hidden row.
6. Release the prefill workspace. Load resident weights, decode state and expert
   slots only on the first subsequent decode call.

There is no partial-layer, expert-slot or CPU-expert fallback in this mode.
Embedding rows are still gathered from the host GGUF; small metadata and
activation management also remain on the host. A layer that does not fit the
budget fails explicitly.

The outer loop is the layer, not the prompt chunk. `--prefill N` controls its
microbatch size. This first implementation uses one layer at a time; it does
not yet overlap loading of the next layer or form multi-layer banks.

Its budgets are independent of decode's expert-cache budget:

- `--prefill-gpu-mib N`: cap tracked GPU allocations for banks, layer state and
  scratch. `0` (default) uses available memory less the prefill reserve.
- `--prefill-reserve-mib N`: leave this much GPU memory outside that budget
  (512 MiB by default). CUDA context/module memory is not part of the tracked
  allocation total; the reserve is separate.
- `--prefill-host-mib N`: major host tensor budget (4096 MiB by default).
  It includes activations, target hidden rows, old and new handoff states and
  conversion workspace. It is not a process RAM limit and does not include
  the operating system's file page cache or all container overhead.

Budget flags require `--prefill-full-layer`. Host-budget rejection happens
before decode is torn down. A later bank/upload failure keeps the previous
host checkpoint available. A later prompt can also export the current decode
state, release decode resources and use the full-layer executor again.
`--n 0` can therefore inspect prefill output without ever loading resident
decode weights.

The handoff copies KDA delta/conv state, the valid MLA latent rows, NextN cache,
target logits and the final post-norm hidden state. Allocation capacities may
differ between phases; raw arenas are not copied as if their layouts matched.
With `--dsa`, it also copies the partial indexer key/gate pool and completed
F32 pooled keys. Dense/sparse mode and incomplete-tail policy must match
across the handoff; an incomplete or mismatched image fails explicitly.

Without this flag, the committed chunk-major resident path remains available
for numerical comparison. Its projections still use groups of at most eight
columns. Independent full-layer prefill instead accepts complete microbatches
of up to 4096 columns:

```text
--prefill-full-layer --prefill 2048
```

Quantized matrices use llama.cpp's MMQ products when the format, reduction
width and device tile are supported. Other supported native matrices use
GPU dequantization plus FP16 cuBLAS GEMM with FP32 accumulation. F32 projections,
including the router, use SGEMM without TF32 rounding. The fallback keeps the
whole token batch; it does not split it into eight-token matrix-vector calls.
Dequantization scratch may tile output rows while retaining all input columns.

The same backend serves resident full-layer experts and batched NextN prompt
catch-up. Router selection, mHC mixing and head/token transposes also accept the
complete batch. KDA recurrence and MLA attention retain their existing causal
GPU kernels; this is not a port of every fused-attention or recurrent-scan
optimization in llama.cpp.

The matrix scratch and cuBLAS workspace are included in the tracked allocation
budget. Library-owned MMQ pool/context allocations are outside that accounting
and need the separate reserve. Larger batches require more activation memory.
Short prompts and the last microbatch use their actual token count.

The CLI prints actual matrix dispatch counters: `max_columns`, `MMQ`,
`FP16_GEMM` and `F32_GEMM`. These are observed product shapes and backend calls,
not the configured batch capacity.

### CPU expert misses

`--cpu-threads N` lets independent CPU expert misses run on up to `N` OpenMP
threads (1 to 256). The default is 1. CMake detects OpenMP for this standalone
runtime without changing the shared ggml CPU backend's configuration. A build
without OpenMP rejects values above 1 explicitly.

Each worker owns its hidden and quantized down-activation buffers. Gate/up
activations are prepared once per input row and shared read-only. Experts keep
the same ggml dot products, row order and GLM post-SiLU clamp; router reduction
order is unchanged. The full-layer GPU prefill path is not changed.
This option also covers CPU misses in resident prefill and NextN.

For example, on the six-core machine measured below:

```sh
OMP_WAIT_POLICY=PASSIVE build-glm5/src/glm5/strata-glm5 \
  --model /path/to/first-shard.gguf --ids 123,456 --n 64 \
  --cpu-threads 6 --glm-gpu-experts 4096 --timings
```

The IDs are placeholders. `OMP_WAIT_POLICY=PASSIVE` asks idle OpenMP workers
to sleep rather than spin between expert batches. The engine does not change
the process's OpenMP environment or pin other threads.

### GPU expert cache

`--glm-gpu-experts N` keeps routed experts in VRAM:

- `-1` (default): off.
- `0`: use available VRAM after the other model allocations, less the reserve.
- Positive `N`: a cap in MiB. A budget that cannot hold one expert per MoE layer
  fails with an explanation.

`--glm-gpu-reserve N` leaves that many MiB free (512 by default). This is separate
from the slot budget. The tier's own activation scratch is allocated before the
budget is calculated.

The default cache policy is adapted to the direct-GGUF loader: equal per-layer slot
quotas, LFU counts, a two-observation admission margin, at most one replacement
per layer/decode call, and count aging every 4096 calls. Both directions of the
slot/owner table are checked when a slot is reassigned.

Resident decode experts run on the GPU while CPU misses are computed. A prefill
chunk groups all occurrences of each expert, streams its original GGUF matrices
through the slots and computes up to eight rows together. The final slot table
is reused by decode in the resident chunk path. Independent full-layer prefill
does not use these slots; decode starts its cache after the phase handoff.
Without the opt-in warmup below, that cache starts empty.
The gate/up/down kernels apply the same GLM clamp as the CPU.
This port does not include the PR's complete pinned RAM arena or double-wave
planner. A bounded asynchronous upload ring is available separately below.

`--glm-gpu-check` recomputes GPU results on the CPU and fails on non-finite values
or a row error larger than that CPU row's RMS, following the PR's diagnostic
criterion. It is an expensive wrong-slot check, not an accuracy guarantee:
GPU Q8_1 and CPU weight-specific activation quantizers are different.

### Prompt heat, guided quotas and asynchronous uploads

The following options are independent experiments, off by default:

- `--glm-gpu-warm`: requires full-layer prefill and an enabled GPU expert cache.
  Record each selected expert's frequency while the existing host routing
  groups are built; there is no second model pass or extra ID download.
  The host handoff includes trunk and optional NextN heat (about 50 KiB for
  the measured model), accounted for in the host tensor budget. Decode
  preloads the hottest positive-frequency experts into its slots. Startup
  loading is included in `decode_setup_ms`, not hidden in the prefill timer.
- `--glm-gpu-heat-quotas`: requires warmup. Keep at least one slot per MoE
  layer, then allocate each further affordable slot to the layer whose next
  ranked expert covers the greatest fraction of that layer's prompt routes.
  Equal gains prefer fewer assigned slots, then the lower layer index. This
  is an initialization plan, not live cross-layer reallocation.
- `--glm-gpu-stage N`: use `N` pinned host buffers (0 disables it, 1 to 8
  enables it). Requires an enabled cache, but not warmup. Buffers hold the
  largest expert in this model; the complete ring is capped at 256 MiB and
  fails explicitly if it would exceed the cap. This is separate from the
  prefill host tensor budget and does not pin the model mapping.

Prompt frequencies seed LFU counts scaled to at most 32 per expert, so a new
decode topic can replace the prior. Ties in warmup use ascending expert ID.
Later prompts carry aged decode heat forward; reset discards the conversation.
No cached GPU weight bank is handed from prefill to decode.

The staging ring packs gate/up/down into one contiguous host buffer and issues
one H2D copy on a nonblocking copy stream. A buffer cannot be reused until its
copy completes. Each GPU slot has readiness and last-read events: compute
waits for its upload, and an overwrite waits for the previous computation.
This also applies to forced streaming during resident prefill and MTP.
Reset and teardown drain outstanding work. Full-layer bank uploads themselves
are not pipelined by this option.

For all three, append these flags to the existing full-layer/cache command:

```text
--glm-gpu-warm --glm-gpu-heat-quotas --glm-gpu-stage 3
```

The CLI reports loaded experts, assigned slots, pinned bytes and total uploaded
bytes separately from hits/misses. Warmup loads are not counted as hits.
For MTP, include `streamed` along with `hits` and CPU `misses` when calculating
the fraction of accesses already resident in VRAM.

### Chunked prefill

`--prefill N` reads up to `N` prompt tokens per layer (1 by default, maximum
4096). Without `--prefill-full-layer`, each chunk passes through the entire model
before the next chunk starts. Dense projections share a weight read over groups of at most eight
tokens. KDA convolution and recurrence still update in token order; MLA cache
writes and causal attention use absolute positions. Router outputs and mHC
post-mix values remain separate for every token.
KDA's L2 norm uses the reference's `1e-6` floor, separately from the model's RMS
normalization epsilon.

The eight-token groups above describe the resident chunk path only. Full-layer
prefill uses the large matrix backend described above. Neither path exposes
llama.cpp's fused indexer. The native DSA indexer remains F32; it is not
silently replaced by a half-query WMMA implementation.

### Greedy MTP verification

`--mtp N` enables up to `N` draft tokens per round (0 by default, maximum 7).
The target verifies the sampled base token and the draft prefix in one batch
of at most eight inputs. Only the matching greedy prefix is committed.

The NextN layer uses the reference's pair `(h[p-1], x[p])`: the target's
post-output-norm hidden state and the next input token. The first hidden state
is zero; the first token embedding is not masked. Every prompt position warms
the draft cache. Optional NextN embeddings/head weights are used when present;
otherwise the main model's embedding, output norm and output head are reused.

On rejection, the KDA states and convolution histories are restored. Following
Maya's window-commit strategy, the verification pass captures each recurrent
layer's pre-convolution q/k/v, bounded decay gates and raw beta values. Only
convolution, L2 normalization and delta-state updates are replayed for the
accepted prefix. The target's dense projections, MLA attention, experts and
output head are not run a second time.

With DSA enabled, verification also captures bounded raw indexer key/gate
rows. Before drafting, the decoder snapshots the main and NextN partial
pools. A rejected prefix restores the main partial pools and rebuilds only
the accepted completed pools from captured rows. Later speculative pools
and MLA rows are hidden by the committed position. NextN's partial pool
is restored before every catch-up, including fully accepted rounds.
Its catch-up uses verified target hidden states, not speculative draft hidden
states. Sparse NextN state is included in full-layer handoff and host budgets.

The committed logits and target hidden rows are taken from the verification
pass itself. MLA rows are append-only and become invisible beyond the restored
position. The draft's speculative rows are replaced by catch-up steps using
verified target hidden states. This is a verification/acceptance loop, not the
old PR's measurement-only draft probe.

Replay captures are allocated only with MTP, before sizing the GPU expert tier.
The CLI reports resident `target_batches` and `KDA_replay_layers` alongside
verification/acceptance counts. These expose whether rejection adds a full
target pass; they are not a throughput measurement.

Full-layer prompt catch-up processes NextN in microbatches, pairing every token
with its preceding target hidden row. The resident chunk path's catch-up still
runs token by token. KDA checkpoints and rejection replay also cost time.
Acceptance and speed must be measured on the actual model; enabling MTP is not
a promise of a speedup.
Verification and replay always run in the resident decode executor, never in the
full-layer loader.

Maya's device-side draft chains, adaptive draft length, reduced draft vocabulary
and multi-conversation speculative scheduler are not ported here. Drafting still
uses the full model head and a fixed user-selected length; sampling remains
greedy. Maya's default hardware gates are not treated as measured recommendations
for this engine.
NextN cache catch-up still runs its forward pass; Maya's cache-writing-only
shortcut is also not ported.

## Build without a GPU

This builds the inspector and format tests, not a CPU inference engine:

```sh
cmake -S . -B build-glm5-host -G Ninja \
  -DSTRATA_BUILD_GLM5=ON -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON
cmake --build build-glm5-host --target glm5-inspect glm5_format_test glm5_cache_test
ctest --test-dir build-glm5-host -R '^glm5_(format|cache)$' --output-on-failure
build-glm5-host/src/glm5/glm5-inspect /path/to/model-00001-of-00005.gguf
```

The inspector checks shard membership, architecture, geometry, required trunk
tensor shapes, payload bounds and model-owned stop IDs. It does not check GPU
kernel availability or prove that generated text is correct.

## Build for a GPU

Use the repository's normal CUDA or HIP prerequisites. For CUDA:

Use the ggml revision pinned in `third_party/ggml/VERSION.txt`
(`3cf03257f219afbe7334045ff7c6a06ac68c627d` for this tree). The newer local
llama.cpp checkout is a semantic reference, not a compatible replacement for
the CUDA MMQ internal API. The default CMake dependency selection uses the
pinned version.

```sh
cmake -S . -B build-glm5 -G Ninja \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_GLM5=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-glm5 --target strata-glm5 glm5-inspect glm5_format_test \
  glm5_cache_test glm5_parity glm5_head_test glm5_runtime_test
cmake --build build-glm5 --target glm5_full_layer_test glm5_gemm_test glm5_batch_runtime_test
ctest --test-dir build-glm5 -R '^glm5_(format|cache|parity|head|runtime|full_layer|gemm|batch_runtime)$' --output-on-failure
```

For HIP, replace the CUDA option with `-DSTRATA_ENABLE_HIP=ON` and use the architecture and toolchain
settings in [AMD_HIP.md](AMD_HIP.md). This port does not enable gfx906, SYCL or
Metal support.

CUDA K-quant MMQ keeps the repository's opt-in `STRATA_MMQ_KQ=1` switch.
Its `strata_kq` library is copied beside the standalone executable by the
build; keep them together. Without that switch, K-quants still use large FP16
GEMMs rather than the MMVQ path. Supported Q8/IQ MMQ formats do not require that switch.

## Generate token IDs

Keep all GGUF shards in the same folder. The engine uses one GPU and greedy
sampling. By default it reads the prompt one token at a time:

```sh
build-glm5/src/glm5/strata-glm5 \
  --model /path/to/model-00001-of-00005.gguf \
  --ids 123,456,789 --n 32 --gpu 0 --max-context 4096
```

For an explicitly enabled acceleration experiment, add:

```text
--glm-gpu-experts 0 --glm-gpu-reserve 512 --prefill 512 --mtp 3
```

For independent full-layer prefill followed by the same decode path, also add:

```text
--prefill-full-layer --prefill 2048 --prefill-gpu-mib 0 --prefill-host-mib 4096 --prefill-reserve-mib 512
```

Reserve context space for the prompt plus the requested output. Generated
non-stop tokens are committed to the model state, including the last one.
Cache and MTP counters are printed to stderr, separately from the token CSV.

Add `--timings` to report model initialization, prefill, resident decode setup
and generation separately on stderr. In full-layer mode, initialization only
opens the model metadata; prefill includes layer loading and first-use CUDA
setup. The optional `Model::prepare_decode()` loads resident resources without
consuming a token. The CLI calls it before timing generation when `--timings`
is set and `--n` is positive. `--n 0` still leaves decode unloaded.
Generation throughput includes expert-cache warm-up and commits every emitted
non-stop token, including the last. It is not a steady-state-only measurement.

The IDs above are placeholders, not a model prompt or a test vector.
The program prints generated IDs as CSV. Use `--dump-logits out.f32` to save the
last prompt position's FP32 logits for a reference comparison. End-of-generation
and end-of-turn IDs come from the model's tokenizer metadata.

For real text, first render the prompt with the model's own chat template. This
helper deliberately does not invent a template or insert role markers:

```sh
python tools/glm5/tokens.py --model /path/to/first-shard.gguf \
  --text-file formatted-prompt.txt
python tools/glm5/tokens.py --model /path/to/first-shard.gguf \
  --decode '123,456'
```

The tokenizer uses the GGUF's `glm4` or `chatglm-bpe` pre-tokenizer, including its
three-digit groups. A successful text round trip alone does not establish token
ID parity.

## Native DSA and Flash Attention

All new real-model tests now use the local six-shard Huihui
`GLM-5.3-Flash-UD-Q4_K_XL`, matching the quantization in the local historical
llama.cpp measurements. Older IQ4_XS measurements below are historical,
not directly comparable speed claims.

The attention paths are explicit opt-ins:

- `--dsa`: load and validate independent indexer tensors, build k-pools,
  score/select causal sparse positions, and preserve the complete indexer
  state through full-layer/decode transitions.
- `--flash-attn`: use GLM latent attention adapted from Strata's own
  `qsa_prompt_attn` design. A block handles 16 heads and 32 selected cells per
  tile, with online softmax and F32 accumulation. Q and probabilities are
  split into F16 hi/lo parts rather than silently cast to F16 alone. This
  does not invoke llama.cpp's executor or its pinned short-context adapter.
  The flag also works without DSA for a dense causal attention control.
- `--dsa-tail 0|1`: explicitly choose incomplete-tail visibility. With no
  override, model metadata applies; if absent, Strata retains its existing
  tail-on geometry default. The Q4 comparisons use `0`, matching the saved
  local reference configuration. Neither choice is claimed lossless against
  the other.
- `--dsa-score auto|scalar|parallel`: select indexer scoring independently of
  attention. `auto` preserves the launch policy: parallel with Flash, scalar
  without it. Explicit overrides require DSA. The parallel scorer now assigns
  heads to lanes while preserving the scalar dimension and head accumulation
  orders, rather than reducing dimensions in a different order.

The first supported scope is one CUDA SM80+ GPU with enough opt-in shared
memory (about 70 KiB per block), one text sequence, F16 latent K=V and F32
activation staging. Shared trunk indexers are rejected rather than ignored.
DSA supports greedy MTP with one complete, independently validated NextN
indexer. Flash can also be combined with this DSA MTP path; dense Flash MTP
without DSA remains explicitly unsupported. This does not implement shared
trunk indexers or reuse the trunk selection for NextN. The current full
NextN construction follows the local llama.cpp reference.

Indexer projections stay batched. Completed pools are appended in one GPU
pass using the previous partial pool where necessary; absolute positions
are generated on the GPU. Selection/attention execute in windows of at most
64 queries. The score reservation is linear in cached pools and independent
of total prompt length times context length; no dense whole-prompt sparse
mask is materialized. Persistent indexer state is bounded by the current
partial pool plus completed compressed keys.

Launch choices are scoped per Model call and host thread. Two Models can use
different attention paths without a process-wide mode leak. The old dense
path remains the default. DSA changes attention selection, and tensor-core
reduction changes rounding: neither is described as bitwise equivalent to
the dense path.

Long prompts can use `--ids-file prompt.ids` instead of an oversized `--ids`
shell argument. It reads comma-separated IDs (up to 4 MiB), permits surrounding
whitespace, rejects empty entries, and is mutually exclusive with `--ids`.

For native sparse prompt-only evaluation, add to the model/input command:

```text
--prefill-full-layer --prefill 2048 --dsa --flash-attn --dsa-tail 0 --mtp 0
--prefill-host-mib 32768 --n 0 --timings
```

Set `--max-context` for the total input, not just the microbatch size.
The current maximum remains 131072; 200k support is not claimed by changing
a limit or by citing the historical reference's 200k result. Use
`glm5-inspect MODEL.gguf --dsa` for host-only tensor/variant validation.

### Q4_K_XL prompt measurements

These measurements precede the score-order and Flash precision corrections below; their engine
hash is `f8a59219868532eea858e4ee08aad3b7c759af9830889839dc14e9df04380fa8`.
On the RTX 3080 reporting 20 GiB and Xeon W-2133, that native build used
2048-token microbatches, F32 activations, F16 latent cache, tail policy `0`,
and no generation or MTP. The repeated-text input is the same token prefix
as the saved local llama.cpp workload. Model pages were warm in these runs.
Prefill time includes loading each complete layer from host memory; it
excludes initial model inspection and subsequent decode setup.

| Total prompt tokens | Attention path | Prefill seconds | Tokens/s |
|---|---|---:|---:|
| 4096 | Existing dense | 49.532 | 82.69 |
| 4096 | Scalar DSA | 70.268 | 58.29 |
| 4096 | Native DSA + Flash | 48.642 | 84.21 |
| 16384 | Native DSA + Flash | 106.475 | 153.88 |
| 65536 | Native DSA + Flash | 389.702 | 168.17 |

The native path is 1.445 times the scalar DSA throughput at 4096 tokens.
This includes both parallel F32 indexer scoring and Flash Attention, not an
isolated Flash kernel speedup. It is only about 1.8% faster than the dense
4096-token control in these single runs. Sampled process GPU memory peaked
at 10292 MiB in the 65536-token run; tracked allocations peaked at 8822 MiB.

A fresh local llama.cpp run on the same Q4 model and 4096-token prefix measured
44.043 seconds, or 93.00 tokens/s. Its saved 200k-token result near 525 tokens/s
has a different total input length and is not a matched comparison. This
port does not yet reproduce that long-input throughput.

The arithmetic smoke generated `2`. All real-model logits dumps were finite.
However, scalar DSA versus native DSA/Flash at 4096 tokens had logit RMSE
0.1767, maximum absolute difference 0.9957 and centered cosine 0.9978.
Their next-token argmax matched, but this is not numerical parity or a
quality evaluation. Indexer reduction and Flash both change arithmetic;
the real-model difference has not been isolated to one of them.

### Subsequent score-order correction

Four 4096-token Q4 controls separated scorer and attention choices. Before
the correction, changing only the scorer had logit RMSE 0.2205 against the
scalar/scalar control; changing only attention had RMSE 0.2107. Enabling
both had RMSE 0.1767. These differences are not additive.

After preserving scalar accumulation order in the parallel scorer, parallel
scoring with scalar attention produced a byte-identical 154880-value logits
dump to scalar scoring with scalar attention: RMSE and maximum error were
zero. With Flash, scalar and corrected parallel scoring also produced
byte-identical logits to each other. Tests require bitwise score equality
for 1, 4, 32, 33 and 64 indexer heads. The default dense real-model dump
remained byte-identical.

This fixes a scorer-dependent source of drift, not full Flash parity.
Flash still had RMSE 0.2107 against scalar attention in the corrected
full-model run. The previous lower combined RMSE was not proof of more
accurate arithmetic. The remaining value includes propagation through later
layers and discrete routing/selection, not an isolated attention-kernel error.
For a strict reference comparison, use `--dsa --dsa-score parallel` without
`--flash-attn`. Do not treat that slower reference path as Flash acceleration.

### Subsequent Flash precision correction

Real Q4 inputs were captured at the last query of all 11 MLA layers.
At the first MLA layer, scalar and Flash runs had identical Q, latent KV
and selected positions. The old Flash output differed from a double CPU
reference by RMSE `4.42e-7`; later sampled layers reached `3.22e-6`.
By the next MLA layer the two full-model runs already had different inputs
and selected positions. Final logit RMSE was therefore not an isolated
attention-kernel error.

The native kernel now compensates QK and PV accumulation, online rescaling
and the softmax denominator. It subtracts the score maximum before converting
to base 2. Explicit rounded operations preserve compensation under the
target's fast-math flags. It still uses Strata's Tensor Core path, F16 hi/lo
components and F32 accumulation, not a scalar or CPU fallback.

Across 22 matched real-input rows, Flash output RMSE against the double
reference fell by 16.4 to 42.6 times. All 33 collected rows, including an
intermediate corrected trajectory, passed NMSE below `1e-12` and maximum
absolute error below `2e-5`. Worst RMSE was `9.81e-8`; every sampled row was
closer to the double reference than the scalar GPU implementation.
These are rows from one prompt, not 33 independent quality tests.

A separate regression exposed incomplete weight-upload ordering: pageable
H2D copies can return before DMA completes, while the bank's consumers use
non-blocking streams. WeightBank now waits for its default-stream uploads
before returning. The old upload path reproduced MTP target-state drift
before the first MLA layer; three focused runs passed with the completion
wait. The existing MTP state tolerance was not relaxed.

The final Q4 4096-token comparison reduced full-model logit RMSE from
0.2107 to 0.1880. Maximum absolute difference increased from 0.9884 to 1.0851,
so this is still not numerical parity or a demonstrated quality improvement.
The next-token argmax matched. Matched dense and scalar controls stayed
byte-identical to their saved outputs.

On the same RTX 3080 reporting 20 GiB and Xeon W-2133, final warm-file
4096-token prefill measured 71.51 tokens/s with DSA/Flash and 48.07 with
scalar DSA, at microbatch 2048. These are single runs in a shared environment.
Flash MTP 0/3 also retained identical 16-token output and prefill logits
on the repeated-text check. No new long-context or broad quality result is
claimed.

Run the numerical checks without model weights:

```text
glm5_attention_test --precision-check
```

`glm5_attention_test --replay-attention FILE` checks a captured fixed-input
row against the same double reference. Captures are local diagnostic data,
not bundled model weights. Temporary runtime capture code was removed.

### Natural-text teacher-forced checks

The test executable also accepts:

```text
glm5_attention_test --evaluate MODEL INPUT.bin OUTPUT.scores
python tools/glm5/compare_logits.py REFERENCE.scores CANDIDATE.scores --output comparison.json
```

This diagnostic mode scores a supplied continuation. It never feeds a
generated token. The first prediction follows the supplied prefix; every
later prediction receives the recorded next token. Output files include
all conditioning IDs, which the comparison tool checks for exact equality.
Existing output files are rejected rather than overwritten.

The input is five little-endian int32 values: magic `0x474c4d45`, version `1`,
prefix length, prediction count (1 to 256), and attention choice (`0` scalar,
`1` Flash). They are followed by `prefix + count` token IDs.
The output starts with six int32 values: magic `0x474c4d50`, version,
vocabulary size, count, prefix and attention choice. All input IDs follow.
Each prediction then stores its target ID and a complete F32 logits row.
This is a test interface, not a released engine-service protocol.

The same Q4 model was checked on four original, non-repetitive text segments:
English, Chinese, code/math, and a mixed segment with a 2305-token prefix.
There were 48 scored positions per segment. The low-margin threshold was
fixed beforehand at a reference top-two logit gap of at most `0.5`.
These self-authored texts are not a standard or representative quality set.

Across 192 positions, scalar/Flash top-1 differed at 7 positions. All 7 were
among the 48 low-margin positions; none of the other 144 positions changed
top-1. Mean KL was `0.01314`, mean total variation `0.04721`, and maximum
total variation `0.24764`. Both paths predicted the supplied next token
exactly at 105 positions. Mean supplied-token NLL was 2.31769 for scalar and
2.29363 for Flash. Agreement is not itself proof of better text quality.
The earlier almost-certain repeated-text prediction did not reveal this
range of distribution changes.

Temporary projection tracing reproduced the uninstrumented outputs
byte-for-byte. At the first MLA layer, Q inputs and selected positions were
identical. One V-expansion example had input RMSE `3.32e-8`, but one changed
Q8 activation code produced output RMSE `4.83e-5`. A double-precision linear
counterfactual with the same dequantized weights had output difference
`1.15e-8`. Thus this case directly identifies activation-quantization
threshold amplification, not merely a large attention-kernel error.
Later sparse choices and expert sets also changed.

Two opt-in experiments kept the GGUF weights unchanged but used F32
dequantized weights and F32 input projections: V expansion alone, or V
expansion plus the attention output projection. Both attention paths used
the same policy. Neither gave a robust overall improvement:

| Policy | Mean KL | Top-1 changes / 192 | Flash supplied-token NLL | Extra sampled GPU memory |
|---|---:|---:|---:|---:|
| Current native | 0.01314 | 7 | 2.29363 | None |
| V expansion + output F32 | 0.01529 | 12 | 2.31018 | 2326 MiB |
| V expansion F32 only | 0.01308 | 8 | 2.31184 | 258 MiB |

These are single short runs; per-case results differed. The experimental
precision flags were removed, not enabled by default or retained as an
unqualified improvement. The current native scalar and Flash score files
remained byte-identical after the experiments. Temporary tracing was removed.

The [comparison tool](../tools/glm5/compare_logits.py) reports KL, total
variation, supplied-token NLL, candidate overlap and low-margin changes.
Its tests run with `python -m unittest tools.test_glm5_logits -v`.
`glm5_attention_test --projection-reference MODEL PAIR.bin` reproduces the
same-weight double linear-delta diagnostic without allocating GPU tensors.

### DSA MTP integration and measurements

Start with `--dsa --dsa-score parallel --dsa-tail 0 --mtp 1`, without Flash.
Use `glm5-inspect MODEL.gguf --dsa --mtp` to validate the complete NextN
weights without GPU uploads. Missing NextN indexer tensors fail explicitly.
After checking the reference path, `--flash-attn` can be added. Matching MTP
to its own attention path does not establish Flash/scalar numerical parity.

On the same RTX 3080 reporting 20 GiB and Xeon W-2133, the six-shard Q4 model
ran a 2048-token historical repeated-text prefix and generated 16 tokens.
CPU expert threads were 6 and the GPU expert cache was disabled to avoid
mixing expert placement changes into this comparison:

| Draft depth | DSA decode tokens/s | DSA + Flash decode tokens/s |
|---|---:|---:|
| 0 | 3.138 | 3.043 |
| 1 | 3.777 | 3.940 |
| 2 | 4.093 | 4.304 |
| 3 | 4.190 | 4.453 |

Each path's MTP 0/1/2/3 runs produced identical 16-token sequences and
byte-identical prefill logits. All offered drafts were accepted on this
repetitive input. These are single short runs, not general acceptance or
throughput estimates. Prefill and decode setup are excluded from decode
speed. The first DSA MTP run additionally read cold NextN pages.
Sampled process GPU memory peaked at 10714 MiB with depth 3.

An arithmetic prompt of 27 tokens generated `2` with MTP off and depth 3,
with and without Flash. EOS truncation triggered one state-only rollback:
34 KDA and 11 trunk indexer replays, without another full target forward.
This one-output-token smoke is not a throughput benchmark.

These MTP timings precede the subsequent Flash precision correction above.
Enabling MTP does not itself fix Flash/scalar differences. General quality,
longer generation, real GPU-cache performance
and matched long-context MTP measurements remain unvalidated.

## Current limits and validation needed

- Text only, one GPU, one sequence, greedy sampling.
- No HTTP server or stdin engine-service protocol, desktop setup, Terminal
  model catalog, automatic engine download or release package.
- No multi-GPU scheduling or persistent conversation checkpoints.
- MLA defaults to the full causal cache. Native DSA/Flash are opt-in within
  the limited scope above. Sparse MTP requires a full NextN indexer; shared
  trunk indexers and dense Flash MTP are not supported.
- Dense projections must fit the selected GPU. Unsupported native formats fail
  explicitly. A successful host inspection does not establish that they fit.
- Synthetic Q8_0 models test ragged prefill widths, cache hits/misses, streamed
  eviction followed by decode, and MTP acceptance and forced rejection with
  continuation-state comparisons. They also combine all three accelerators.
- Full-layer tests compare raw KDA/conv states, MLA/NextN cache rows, hidden
  carry and logits, then test delayed decode, repeated phase switches, budgets
  and MTP staying on the decode path.
- Full-model logits, token IDs, memory use and generated text still need
  comparison with a reference using the same real GGUF, prompt, context and
  sampling settings. CPU/GPU quantization and grouped arithmetic can move
  near-tie greedy choices; synthetic tests do not establish full-model quality
  or performance.

The host checks use synthetic files and no model weights. Run tokenizer
regressions with:

```sh
python -m unittest tools.test_glm5_tokenizer tools.test_strata_tokenizer \
  tools.test_tokenizer_plain_spans -v
```

Tests requiring a real Qwen vocabulary may skip unless `STRATA_TOKENIZER` or
`STRATA_GGUF_PY` points to one. Do not count those skips as vocabulary parity.

## Checks run on this port

On 2 November 2026, the CUDA targets were built on Linux with GNU C++ 15.2,
CUDA 13.3 and `CMAKE_CUDA_ARCHITECTURES=86`. The large-batch build uses the
repository's pinned ggml revision above. GPU tests ran on an NVIDIA GeForce
RTX 3080 with 20 GiB reported VRAM.

Passed:

- The standalone engine, format/cache tests, GPU math parity, head and runtime
  tests built. The eight GLM CTest entries, including large GEMM and full-layer
  handoff, passed.
- The host-only format/cache and existing geometry, split-shard and architecture
  default tests passed (five entries).
- The runtime tests exercised four synthetic trunk layers and one NextN layer,
  Q8_0 experts, one-slot eviction, ragged prefill widths 2/7/8/17/32, accepted and
  rejected drafts, reset, continuation-state comparisons and combined modes.
- State-only commit was checked for every kept prefix of a four-token window,
  starting from nonzero KDA/conv state. Forced rejection also checks the actual
  target-forward count: no full target pass is added for prefix replay.
- The CLI produced exactly 12 verified tokens and a 32-row FP32 logits file
  from a synthetic model with all three options enabled.
- The full-layer path matched the resident reference's KDA/conv state,
  MLA/NextN cache rows, hidden carry and logits. Tests also covered no-MTP
  multi-group batches, delayed decode, appended prompts and budget failure.
- Direct matrix tests ran 2048 and 4096 columns against CPU dequantized
  references, including FP32 SGEMM, native MMQ, FP16 fallback and output-row
  tiling. They also checked the 64-head/2048-token transpose and batched router.
- A 2049-token full-layer run used an actual 2048-column product plus a tail.
  Its observed counters were `max_columns=2048 MMQ=73 FP16_GEMM=6 F32_GEMM=2`.
  State and decode/MTP continuation were compared with eight-column processing.
  These are synthetic-model dispatch checks, not a full-model throughput result.
- CUDA Compute Sanitizer memcheck reported zero errors and zero leaked bytes
  for the runtime, full-layer, large-matrix and 2049-token tests. The K-quant
  shared library now releases its cached CUDA contexts on normal process exit;
  that cleanup was checked with `STRATA_MMQ_KQ=1`.
- Strict C++ syntax/type checks passed. Python tokenizer tests ran 23 entries
  with three real-vocabulary skips.

The runtime checks also caught two porting issues that are fixed here: the
ggml CPU tables must be initialized before direct dot products, and chunk
arenas must align whole column groups rather than pad every row separately.

The subsequent native-attention build passed all nine GLM CTest entries,
including the new attention test. Attention and full-layer handoff each
passed five consecutive runs. Direct attention checks use a double-precision
CPU reference, require NMSE below `1e-7` and maximum absolute error below
`2e-4`, and cover causal masks and empty/invalid selections. A 2305-token
synthetic model exercises actual sparse pruning, both tail policies and
prefill/decode state transitions. Combined tests preserve expert heat and
bounded pinned staging through sparse state handoffs.

Native attention memcheck reported zero errors and leaked bytes. Flash-filtered
racecheck reported zero hazards; synccheck reported zero errors. Host-only
format/cache tests and 11 CLI parsing/rejection checks also passed.

The subsequent DSA MTP suite passed 144 synthetic combinations across
attention modes, tail policies, draft depths 1/3/7, pool boundaries,
forced acceptance/rejection and 2305-token pruning. It compares target and
NextN states to serial stepping, and tests append/restore/reset plus prompt
heat and bounded asynchronous expert staging. A separate 48-case pool commit
test checks every accepted prefix from 1 to 8 for bitwise state equality.
The focused 120-short-prefix MTP suite passed memcheck without errors or
leaks and synccheck without errors. Append-filtered racecheck reported
zero hazards. The full long-prefill memcheck attempt was stopped because
of instrumentation cost; it is not counted as a completed check.

Not run: HIP builds, other GPU architectures, broad quality
evaluations or a matched long-context reference comparison. Real Q4 sparse
prefill ran through 65536 tokens as recorded above. Passing these checks
does not establish full-model numerical equivalence.

## Real-model measurements

On 3 November 2026, the local
`huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF` model,
`GLM-5.3-Flash-UD-IQ4_XS`, ran on the RTX 3080 above, with an Intel Xeon
W-2133 (6 cores, 12 threads) and 247 GiB RAM. This is an abliterated fine-tune,
not the official base model. The run used all 45 trunk layers and, with MTP,
the separate NextN layer.

An arithmetic smoke prompt returned `2` for `1 + 1`. A second prompt asked why
the daytime sky is blue. Its exact 2048 IDs were checked against native
`llama-tokenize`, including role markers and a closed thinking block.
Each generation run produced 64 readable English tokens without an error.

The options were full-layer prefill, batch 2048, context 4096, expert-cache cap
4096 MiB, tracked prefill GPU cap 12288 MiB, host tensor budget 4096 MiB and
1024 MiB GPU reserves. K-quant MMQ was enabled, CUDA modules loaded eagerly
and TF32 disabled. Runs were sequential, without dropping system file caches.
The second run of each mode reported zero input blocks.

| Mode | Prefill, two runs | Generation, two runs | Sampled peak VRAM | Peak process RSS |
|---|---|---|---|---|
| MTP off | 48.31 / 47.32 tokens/s | 1.010 / 1.040 tokens/s | 12.64 GiB | 144.08 GiB |
| MTP 3 | 46.52 / 46.13 tokens/s | 1.237 / 1.301 tokens/s | 13.14 GiB | 147.55 GiB |

VRAM was sampled for the measured process at nominal 0.5-second intervals;
these are sampled peaks, not exact allocation maxima. RSS includes mapped
weight pages, not just heap allocations. Both prefill modes observed actual
2048-column products. MTP-off dispatch counted 27109 MMQ, 68 FP16 GEMM and
42 F32 GEMM calls.

Across the two runs, MTP generation was 1.238 times as fast by pooled elapsed
time. It accepted 39 of 74 proposed drafts (52.70%) and used 25 target
verification batches for 64 output tokens. However, MTP-on and MTP-off output
first differed at output token 17. Each mode reproduced its own output across
both runs. These results do **not** establish lossless MTP equivalence.
The ordinary decoder recorded 6084 cache hits and 15420 CPU misses; those
baseline measurements used the default serial CPU expert path.

A separate same-ID comparison called the local llama.cpp library's full-layer
API directly, with F32 activation staging, no flash attention and one layer per
bank. Its prefill timer reported 59.65 tokens/s; Strata's separate prefill-only
run reported 49.67 tokens/s. Both whole processes took about 42.7 seconds.
The timers place first-use CUDA/model setup in different phases, so their
prefill numbers alone are not an end-to-end speed comparison.

Both returned first-token ID 785 and finite logits for all 154880 vocabulary
entries. The full-vector comparison measured RMSE 0.47315, maximum absolute
difference 2.18527, centered cosine 0.97172 and KL(llama.cpp || Strata) 0.03648;
their top-ten sets shared eight entries. This is not numerical parity.
The cause of the differences has not been isolated. Real-weight numerical
alignment remains necessary before claiming reference-equivalent inference.
An earlier reference invocation tokenized role markers as ordinary text and
evaluated 2063 tokens; it was excluded from these comparisons.

An initial regression passed seven entries and intermittently failed the
cross-executor bitwise MLA-cache comparison. A diagnostic repeat measured a
maximum two-FP16-bit distance and absolute difference 0.000977 in layer 3.
Different expert/reduction paths are now compared numerically: FP16 caches
use a relative maximum tolerance of 0.002, while KDA/conv state, hidden carry
and logits retain 1e-5. A separate export/restore/export check requires
bit-exact KDA, MLA, NextN, hidden and logits copies. No inference arithmetic
was changed by this test correction.

Another `strata-host` process then used about 16 GiB VRAM. To finish the
small-fixture tests without stopping it, the tests explicitly use 128 MiB
reserves and a 256 MiB large-batch prefill cap, with
`CUDA_MODULE_LOADING=LAZY`. Production defaults and the real benchmark's
1024 MiB reserves are unchanged. The corrected full-layer test passed 15
consecutive runs, and the final eight-entry GLM suite passed all eight.
These synthetic checks do not resolve the real-model numerical differences
reported above.

### CPU miss optimization follow-up

After checkpoint `22c107fb`, an opt-in CPU expert dispatch was measured on the
same model, hardware, 2048 IDs, 64 generated tokens and GPU budgets. MTP was off.
Both modes used the same updated executable and `OMP_WAIT_POLICY=PASSIVE`.

| CPU expert threads | Decode, two runs | Generation time, two runs |
|---|---|---|
| 1 (default) | 1.025 / 1.052 tokens/s | 62.411 / 60.838 s |
| 6 | 1.818 / 1.859 tokens/s | 35.199 / 34.426 s |

This is about 1.77 times the pooled decode throughput. All four 64-token
outputs were byte-identical to each other and to the preceding checkpoint's
ordinary decode output. Expert-cache counters and sampled peak process VRAM
(12940 MiB) were unchanged. More CPU work runs concurrently; this is not a GPU
kernel speedup or a claim that six threads fit every system.

The first six-thread run read about 40 GiB through filesystem input blocks
and took 52.0 seconds in prefill; the warm repeat took 40.7 seconds. Prefill
does not use this new CPU expert dispatcher, and no prefill speedup is claimed.
A single six-thread MTP3 combination run reached 1.378 tokens/s and reproduced
the previous MTP3 output, still slower than six-thread ordinary decode here.
MTP's existing non-equivalence to MTP-off output is not fixed by this change.
Neither CPU parallelism nor MTP is enabled by default.

### Automatic expert-cache measurement

The same committed engine (`0a84ee59`) was then tested with
`--glm-gpu-experts 0 --glm-gpu-reserve 1024`, six CPU expert threads and MTP off.
The model, exact 2048 input IDs, 64 requested output tokens and other settings
were unchanged. Two automatic-cache runs and a fresh fixed-cache control
exited successfully; all reported zero filesystem input blocks.

| Cache mode | Actual cache | Slots per MoE layer | Hit rate | Decode | Sampled process VRAM peak |
|---|---|---|---|---|---|
| 4 GiB cap, fresh control | 3806.5 MiB | 8 | 28.29% | 1.889 tokens/s | 12.64 GiB |
| Automatic, two runs | 6661.4 MiB | 14 | 36.26% | 1.817 / 1.812 tokens/s | 15.42 GiB |

Automatic sizing worked and used more memory, but no decode speedup was
observed in this short test. Cache replacements increased from 456 to 594,
and 63.74% of expert accesses still missed. The measurements include cold
expert-cache initialization: full-layer prefill does not seed its contents
or frequency history. They are not steady-state-only hit rates.

The automatic runs reproduced each other's 64-token output. They first
differed from the fixed-cache output at token 17; the fixed-cache control
reproduced its preceding output exactly. Changing residency changes which
experts use the CPU versus GPU path, whose real-model numerical equivalence
is not established. Thus these are same-prompt generation measurements, not
a strict identical-output workload comparison. More VRAM alone has not
resolved the current decode bottleneck; phase profiling is still needed.
No implementation or default was changed for this measurement.

### Three-step cache optimization checks

On the same RTX 3080 / Xeon W-2133 and IQ4_XS fine-tune, new controls used a
6144 MiB cache cap, six CPU expert threads, 2048 input IDs and 64 output tokens:

| Mode | Decode | Decode setup | Runtime replacements |
|---|---|---|---|
| Cold, uniform quotas, two runs | 1.980 / 1.998 tokens/s | 1.26 s | 547 |
| Warm, uniform quotas, two runs | 2.168 / 2.173 tokens/s | 2.10 / 2.12 s | 197 |
| Warm, heat quotas | 2.169 tokens/s | 2.17 s | 181 |
| Warm, heat quotas, 3 staging buffers | 2.159 tokens/s | 2.07 s | 181 |

Cold and warm uniform runs produced the same 64 IDs. Pooled decode throughput
improved 9.12%, while whole-process throughput improved 2.93%, including the
added warmup cost. Prompt heat did not improve the hit rate on this prompt:
it fell from 34.04% to 29.78%, despite less runtime replacement. This prompt's
repetitive input notes and generated science prose need not route alike.
Do not treat prompt heat as a guaranteed decode-frequency prediction.

Heat quotas used 540 slots (8 to 22 per MoE layer), compared with 504 uniform
slots under the same cap, but did not show an additional throughput benefit.
The three-buffer ring used 45.19 MiB pinned host memory, not the whole model.
With automatic capacity, the three-step combination allocated 6993.7 MiB,
used 615 slots (8 to 23 per layer), and reached 2.134 tokens/s in one run.

A separate MTP3 pair, with identical quotas and output IDs, measured
1.835 tokens/s for serial uploads and 1.946 tokens/s with the ring (6.03%
decode improvement in that one pair). It still did not beat ordinary warm
decode. Staging on/off matched 64 IDs in both MTP modes; changing quota policy
can change residency and output, so not every row is an identical-output A/B.
These checks do not resolve CPU/GPU numerical or lossless-MTP equivalence.

Nsight Systems 2026.1.3 captured 15 seconds of ordinary decode, starting after
prefill/warmup. All 102 large H2D transfers were pinned-host to device on the
copy stream. Their measured GPU activity intervals totaled 154.46 ms;
98.36 ms (63.68%) intersected actual kernel intervals on the compute stream.
This establishes overlap, not a 63.68% throughput gain. Kernel activity occupied
2.139 seconds of the window; CPU work and host scheduling were not separately
profiled, so the remaining time is not a measured CPU-math percentage.

The final eight GLM CTests passed, runtime/full-layer tests each passed five
repeats, and the host-only format/cache tests passed. Memcheck reported zero
errors and zero leaks for full-layer, streamed runtime/MTP and GEMM tests.
The first memcheck exposed an MMQ ragged-column ID-tile overread; the GLM
identity buffer now pads the 128-column tile, with a 17-column-capacity
regression. HIP and other GPU/CPU systems were not tested.

To repeat the memory check after building:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 --leak-check full \
  build-glm5/src/glm5/glm5_runtime_test
compute-sanitizer --tool memcheck --error-exitcode 1 --leak-check full \
  build-glm5/src/glm5/glm5_full_layer_test
STRATA_MMQ_KQ=1 compute-sanitizer --tool memcheck --error-exitcode 1 --leak-check full \
  build-glm5/src/glm5/glm5_gemm_test
compute-sanitizer --tool memcheck --error-exitcode 1 --leak-check full \
  build-glm5/src/glm5/glm5_batch_runtime_test
```
