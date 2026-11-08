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

Measured bottlenecks and a hierarchical list of optimization ideas:
[GLM optimization plan (Chinese)](GLM5_OPTIMIZATION_PLAN.md).

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

## Normal use and automatic optimizations

On the validated CUDA SM86 backend, GLM selects the successful compute
paths internally. There is no need to specify separate speed switches:

```text
--prefill-full-layer --dsa --flash-attn
```

DSA, Flash, MTP and CPU threads remain explicit choices.
On the validated SM86 GLM geometry, expert-cache capacity defaults to
automatic VRAM budgeting; explicit capacity/resource limits remain
available. Full-layer mode retains its batch/prefetch/group capacity
planner and activation pipeline. `--prefill-no-pipeline` remains available
for its real host/VRAM tradeoff.

Automatic compute selection uses:

- tiled F32 indexer scores for 32 heads of dimension 128;
- register-prefetched reduction tiles for that scorer, retaining the
  four-query tile and exact FMA/head accumulation order;
- cooperative top-k for at most 512 selected pools;
- register-state KDA for head dimension 128;
- omission of unused legacy Q8_K/Q8_0 activation images;
- omission of their scratch allocations, with required native/BF16 inputs retained;
- scoped native input-quantization reuse for compatible expert gate/up products;
- batched MLA head-input quantization with unchanged individual head products;
- direct head-major MLA Flash attention for KV rank 512 and head width 256,
  omitting two attention-boundary transposes and their temporary images;
- asynchronous half-tile KV prefetch for that Flash geometry, preserving
  the original 32-cell arithmetic tiles and accumulation order;
- batched expert-row gather/scatter during full-layer prefill.

On SM86 the GPU expert cache also selects direct single-token execution:
one native Q8_1 input preparation per participating dispatch, no
per-expert input gather or output scatter copy. Individual gate/up/down
products retain their original dimensions and accumulation.
Multi-token or duplicate-expert routes use the original path.
`gpu_single` in the compute log and `glm5 single experts` counters show
selection and actual use. Other architectures keep the original
dispatch. No extra CLI speed switch is required.

The expert-row path also reuses bounded host routing buffers and uses a
stable counting pass instead of allocating per-expert vectors each time.
It still synchronizes routing IDs with the host.

CPU expert row scheduling has a separate, narrow automatic policy:
CUDA SM86, Linux/x86_64, a Skylake AVX-512 host reporting 12 logical CPUs,
an OpenMP/ggml AVX2 build, 3, 6 or 12 requested CPU threads, and the measured
4096-hidden/2048-expert-FFN/288-expert/top-8 geometry.
Other configurations and the default single CPU thread keep the original
CPU path. `--reference` disables this selection too. The startup log
reports `mode=rows` or `mode=reference`; no new CLI speed switch is needed.
This changes scheduling, not dot arithmetic or the CPU instruction set.
See the [row-pool measurements](#cpu-expert-row-pool-follow-up).
On the measured RTX 3080 / Xeon W-2133, a twelve-thread row pool improved
decode over six threads by 4.61% at 8k, 5.56% at 32k and 5.69% at 200k,
using the same 4096 MiB expert-cache cap, prompt warming and heat quotas.
Use `--cpu-threads 12` to request it on that configuration. This does not
raise the default thread count or change the instruction set.
These are same-run thread-count comparisons, not gains over the earlier
automatic-VRAM-cache measurements. Details and limits are in
[the decode-only experiments](GLM5_OPTIMIZATION_PLAN.md#二十六只优化-decodecpu-工作池与-gpu-未命中分流).

`--cpu-atomic-wait` is a separate, off-by-default row-pool experiment.
It uses an atomic completion count instead of taking the completion mutex
for every worker. The caller polls for at most 100 microseconds, then
sleeps with atomic wait; the last worker wakes it. Idle workers still
sleep, and no CPU dot, activation quantizer or GLM clamp changes.
It requires the row executor; unsupported automatic selections and
`--reference` report an error rather than silently ignoring the flag.
The caller still waits for every worker before reusing phase data.
For the measured configuration, append it to `--cpu-threads 12`.
See [the Maya-inspired CPU experiments](GLM5_OPTIMIZATION_PLAN.md#二十七参考-maya-优化-cpu-等待).

Each geometry condition is independent. CUDA devices outside SM86 and HIP
keep their prior compute dispatch until validated. Automatic selection
does not enable DSA or Flash by itself. Diagnostic `--dsa-score scalar`
or `parallel` still overrides the automatic scorer.

Use `--reference` for developer comparisons. It disables automatic compute
selection and, with full-layer mode, uses the single-layer synchronous
reference configuration. Supply the same explicit `--prefill N` for a
matched-batch comparison. The old `--prefill-reference` spelling remains
an alias requiring full-layer mode; `--prefill-auto` and
`--prefill-pipeline` remain accepted compatibility spellings, not required
speed switches.

The former experimental switches `--dsa-score tiled`,
`--dsa-parallel-topk`, `--prefill-expert-rows`, `--skip-unused-quant` and
`--kda-register-state` are no longer CLI inputs. Historical measurements
below name those switches as they existed at measurement time; use normal
automatic selection to reproduce the combined path now. The private
kernel/executor tests retain explicit policies for exact A/B checks.

Multi-expert MMQ was not retained: its option, executor and dedicated test
were removed, as was the added shared-backend query. All retained source
and test changes are under the GLM-specific directories. Existing shared
MMQ execution and other model families are unchanged.

## Optional acceleration

For bottleneck diagnostics, add `--probe`. It implies `--timings`, reports
prefill and decode separately, and prints host/ggml CPU ISA capabilities.
It is off by default. See [phase-probe measurements](#phase-probes-transfer-and-cpu-checks)
for the meaning of its counters and measured limits.

Full-layer prefill, CPU expert parallelism and MTP are opt-in. Full-layer
prefill includes automatic scheduling and transfer overlap by default.

### Independent full-layer prefill

`--prefill-full-layer` selects the independent executor. It does not construct
the resident decode engine before prefill:

1. Keep prompt activations and completed layer states in host memory.
2. Plan the batch, next-layer prefetch and consecutive layer groups from the
   available budgets. Load each group's complete weights into GPU banks.
3. Process every prompt microbatch through that group, preserving sequential
   KDA updates and causal MLA positions.
4. Copy its final state to the host and release the group's banks. Reuse
   already-prefetched weights when advancing to the next group.
5. Run the output head with the final group if it fits, or in a separate
   bank. With MTP enabled, run NextN prompt
   catch-up in its own complete-layer bank, using every target hidden row.
6. Release the prefill workspace. Load resident weights, decode state and expert
   slots only on the first subsequent decode call.

There is no partial-layer, expert-slot or CPU-expert fallback in this mode.
Embedding rows are still gathered from the host GGUF; small metadata and
activation management also remain on the host. A layer that does not fit the
budget fails explicitly.

The outer loop is a layer group, not the prompt chunk. `--prefill N`
bounds the microbatch size (2048 by default in full-layer mode, 1 in the
unchanged resident path). Scheduling follows the priority order below.

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

#### Capacity planning, grouping and transfer overlap

The standalone scheduler follows the local llama.cpp reference at
`e129d48751534b1510619d2d719bfd6771827d9c`, especially
`src/llama-context-prefill.cpp` and `src/llama-prefill-plan.cpp`.
It uses Strata's own kernels, state images and allocation accounting. It
does not call llama.cpp's executor.
The reference's complete-layer mode disables expert-only prefetch. Here the
next-group staging mechanism is adapted to complete layer banks instead;
an expert-only upload is never counted as a complete layer.

`--prefill-full-layer` alone enables automatic scheduling:

```text
--prefill-full-layer
```

The priority order is:

1. A complete layer, its state and required workspace must fit. There is no
   partial-layer or CPU fallback. If even batch 1 cannot fit, report an error.
2. Keep the largest admitted candidate microbatch. Start at 2048 unless
   `--prefill N` overrides it, bounded by the actual suffix length. Halve
   only when the single-layer execution requirements cannot fit.
3. Reserve space to prefetch the next complete layer when possible. Never
   shrink an admitted batch merely to enable prefetch.
4. Grow consecutive layer groups only within the remaining capacity, without
   sacrificing an available next-layer prefetch. The default group cap is 4.
   On CUDA SM86 with the activation pipeline and no explicit group maximum,
   also reject an extension when weights losing prefetch coverage are at
   least as large as the full suffix activation H2D boundary it would save.
   This is a conservative byte-based heuristic, not a measured latency model.

Pipeline is enabled as a fixed execution cost and included in admission
before these choices. It is not switched off silently to obtain a larger
batch. Use `--prefill-no-pipeline` to explicitly choose that tradeoff.

- `--prefill-auto`: explicitly select the automatic planner. This is already
  enabled by full-layer mode.
- `--prefill-group-layers N`: set the group maximum (1 to 64). Without
  automatic planning in reference mode, groups have this size except at the end; a group that
  does not fit is an error, not a smaller-group fallback. Each layer retains
  its own causal attention, indexer and recurrent state. Intermediate
  activations stay on the GPU with the same F32 precision.
  An explicit maximum retains capacity-based grouping and bypasses the
  transfer heuristic, allowing controlled comparisons without disabling
  other compute optimizations.
- `--prefill-prefetch-layers N`: prefetch up to this many complete layers
  from the next group (0 to 64, default 1; 0 disables it). The planner reduces the count
  to fit current-group state, both weight banks and host/transfer budgets.
  Prefetched weights replace, rather than duplicate, that group's normal
  uploads. There is no expert-only or CPU fallback.
- `--prefill-pipeline`: explicitly enable the default pipeline. Pin the single host activation array and use two
  bounded device activation slots with
  separate upload/download streams. Events protect uploads, computation,
  downloads and slot reuse. There is no extra host activation copy.
- `--prefill-no-pipeline`: use pageable host activations and synchronous
  group-boundary transfers instead.
- `--reference`: retain the single-layer synchronous reference
  configuration. Its default batch is 1, without automatic batch reduction,
  prefetch or pipeline. Explicit batch/group sizes and resource limits
  remain available for controlled comparisons. For a matched batch-2048 reference,
  use `--prefill-full-layer --reference --prefill 2048`.
- `--prefill-transfer-mib N`: bound device boundary buffers plus pinned
  weight staging (512 MiB by default). These allocations also count against
  their respective host and GPU budgets. The pinned host activation array
  counts once against the host tensor budget, not the transfer limit. A large
  microbatch may need a larger limit or automatic batch reduction.

Pipeline needs two additional device buffers:
`2 * batch * hc_dim * sizeof(float)`. For this GLM's 4096 channels and four
streams, that is 64 MiB at batch 512 or 256 MiB at batch 2048. Disabling it
saves those buffers, not the host activation array. The array still exists
in ordinary pageable RAM. With pipeline it is pinned: about 256 MiB at
4096 input tokens or 4 GiB at 65536 tokens, counted once in the host budget.
These are activation sizes, not total process memory or weight-prefetch costs.

Automatic expert-row kernels replace the complete-layer executor's
per-row expert gather/scatter copies on CUDA SM86, except in reference
mode. This includes NextN prompt
catch-up, but does not change resident decode or its expert cache.
The host still groups routes in the same expert and token order; each
layer batch uploads one packed `{input token, output row}` map, then
launches one gather and one scatter kernel per expert batch. Repeated
input tokens are allowed; output rows remain distinct. Expert arithmetic,
matrix dispatch, clamp and router-reduction order are unchanged.

The map uses `batch * active_experts * 2 * sizeof(int32_t)` bytes in both
pinned host memory and device memory. For batch 2048 and eight active
experts, each is 128 KiB. Host/device allocations count against their
respective budgets; pinned map staging also counts against the transfer
limit. Upload completion protects the pinned buffer during cleanup.
This path does not combine experts into a single MMQ product or remove
the host route/count synchronization.

Planning checks native weight layouts without uploading them and measures
the executor's real scratch allocations for the candidate batch. Unlike
llama.cpp's graph scheduler, this executor has explicit workspace arenas;
there is no graph-workspace estimate to calibrate. It also checks the output
head and NextN bank. Backend-private allocations still need the GPU reserve.
Only tracked capacity rejections trigger batch reduction. Invalid formats,
CUDA failures and upload errors propagate explicitly.

This is **capacity planning**, not a latency search. It stops at the first
admitted batch/group plan; it does not benchmark candidates or claim that
the largest group is fastest. llama.cpp's separate expert-only calibrated
latency planner is not substituted for complete-layer capacity planning.

Weight prefetch uses host workers, nonblocking copy streams and a two-slot
pinned staging ring of 1 MiB per prefetched layer. Scalar conversion storage
is included in host admission. Upload completion is checked before use, and
cleanup waits for outstanding uploads. Group state is returned to the host
before its bank is released. NextN prompt catch-up remains a separate bank
and uses every verified target hidden row.

When scheduling is enabled and the output head fits alongside the final
group, that group computes hidden rows/logits directly and omits the unused
final residual download and subsequent head-input upload. Otherwise the
separate-head reference path remains available. No precision change is
made to obtain a plan.

The CLI reports selected batch, group count, largest group, planning
attempts/time, prefetched layers, fused-head selection and transfer peak.
Weight bytes count tensor payload uploads; activation bytes count trunk
group-boundary residual transfers, not state, hidden-row or logits copies.
The tracked GPU peak includes planning allocations. The reported host
tensor size is a conservative bound, not process RSS.

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

`--cpu-threads N` supplies up to `N` CPU workers including the caller
(1 to 256); the default is 1. The reference path divides whole experts
between OpenMP workers. The measured configuration described above
automatically selects the persistent row pool instead. CMake detects
OpenMP without changing the shared ggml backend's configuration; the
normal CLI still rejects values above 1 in a build without OpenMP.

The reference path uses per-worker hidden/activation buffers. The row
pool reuses bounded shared scratch, with each output row written by one
task. Input activations are shared read-only. Both paths keep the same
ggml dot products and GLM post-SiLU clamp, and preserve each row's
accumulation and router reduction order. Full-layer GPU prefill is
unchanged. CPU misses in resident prefill and NextN use the selected path.

For example, on the six-core machine measured below:

```sh
OMP_WAIT_POLICY=PASSIVE build-glm5/src/glm5/strata-glm5 \
  --model /path/to/first-shard.gguf --ids 123,456 --n 64 \
  --cpu-threads 6 --glm-gpu-experts 4096 --timings
```

The IDs are placeholders. `OMP_WAIT_POLICY=PASSIVE` asks idle OpenMP workers
to sleep rather than spin between expert batches. Row-pool workers wait on
condition variables. The engine does not change the process's OpenMP
environment or pin other threads.

### GPU expert cache

`--glm-gpu-experts N` keeps routed experts in VRAM:

- Omitted: automatic on CUDA SM86 with 4096 hidden width, 2048 expert FFN,
  288 experts and top-8 routing; off for other geometry/backends or reference.
- `-1`: explicitly off, including on the validated configuration.
- `0`: use available VRAM after the other model allocations, less the reserve.
- Positive `N`: a cap in MiB. A budget that cannot hold one expert per MoE layer
  fails with an explanation.

`--glm-gpu-reserve N` leaves that many MiB free (1024 by default). This is separate
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
cmake --build build-glm5 --target glm5_attention_test glm5_indexer_test glm5_expert_rows_test
cmake --build build-glm5 --target glm5_compute_test
ctest --test-dir build-glm5 -R '^glm5_(format|cache|parity|head|compute|attention|indexer|expert_rows|runtime|full_layer|gemm|batch_runtime)$' --output-on-failure
```

For HIP, replace the CUDA option with `-DSTRATA_ENABLE_HIP=ON` and use the architecture and toolchain
settings in [AMD_HIP.md](AMD_HIP.md). This port does not enable gfx906, SYCL or
Metal support.

CUDA K-quant MMQ keeps the repository's opt-in `STRATA_MMQ_KQ=1` switch.
Its `strata_kq` library is copied beside the standalone executable by the
build; keep them together. Without that switch, K-quants still use large FP16
GEMMs rather than the MMVQ path. Supported Q8/IQ MMQ formats do not require that switch.

### Opt-in expert work groups

`--prefill-expert-groups` groups consecutive routed experts into the existing
expert workspace. It gathers their rows once, keeps each expert's original
gate/up/down GEMM and quantization, applies the same GLM SwiGLU clamp and scatters
the results once. It does not merge the MMQ products or change the expert combine
order. Experts with eight or fewer rows, and experts larger than the workspace,
keep the original path.

This requires non-reference full-layer prefill and a requested batch above eight.
It is off by default and does not change decode or MTP settings. With a batch of
2048, its per-column bounds table adds 16 KiB of tracked GPU memory. Its small
host work tables are included in the host budget. Explicit GPU/host limits still
apply, and the existing planner may reduce batch or report insufficient capacity.

The CLI prints the actual group, expert and row counts. Matrix and quantization
counts remain comparable to the original path; cached bounds reduce bounds
launches. This is a modest prompt optimization, not a decode speed-up.
Measurements and rejected alternatives are in
[the optimization plan](GLM5_OPTIMIZATION_PLAN.md#二十五按顺序验证四个优化方向).

For a separate benchmark executable, put `strata_kq` beside **that executable**,
not just beside the normal engine. `LD_LIBRARY_PATH` does not replace this rule.
Reject a run whose log says that the library is missing and experts use the
FP16 prompt path; it is a different arithmetic/performance baseline.

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
  attention. `auto` selects tiled F32 on validated SM86/32-head/128-dimension
  geometry. Otherwise it keeps the prior policy: parallel with Flash,
  scalar without it. Explicit overrides require DSA. The parallel scorer assigns
  heads to lanes while preserving the scalar dimension and head accumulation
  orders, rather than reducing dimensions in a different order.
- Automatic tiled scores: the F32 scorer supports 32 heads of width
  128. Four queries and 64 pooled keys share each tile. Unlike Maya's
  four-quarter reduction, the epilogue sums individual heads in the original
  order; dot-product dimensions also retain their original order. There is
  no FP16 query conversion or TF32 rounding. Other head counts retain the
  previous scorer.
- Automatic parallel top-k: block-cooperative radix selection and
  bitonic sorting, adapted from Maya. Each query uses 1024 threads. Scores
  descend, equal scores prefer the lower pool index, and signed zeros tie.
  Even when all visible pools fit, they are sorted rather than emitted in
  cell order. Tail offsets and unused `-1` cells retain the existing contract.
  At most 512 selected pools are supported; larger selections retain the prior selector.

Both paths require `--dsa` and are independent of Flash and each other.
They apply to main-model prefill/decode, NextN and MTP verification/replay
through the same scoped launch policy. Reference mode preserves the old compute path. The
F32 tile and cooperative selector are adapted from Project Maya v1.1.49
(`26aacb6`), with its MIT notice retained in the kernel header.
They need no extra global score arena; the existing bounded 64-query
scratch and host/GPU budgets remain in use. Per-block shared memory is
part of GPU execution, not a new persistent model allocation.

Use automatic selection with:

```text
--prefill-full-layer --dsa --flash-attn
```

The focused regression and synthetic timing commands are:

```sh
build-glm5/src/glm5/glm5_indexer_test
build-glm5/src/glm5/glm5_indexer_test --bench
compute-sanitizer --tool memcheck --error-exitcode 1 --leak-check full \
  build-glm5/src/glm5/glm5_indexer_test
compute-sanitizer --tool racecheck --error-exitcode 1 \
  build-glm5/src/glm5/glm5_indexer_test --kernel-check
compute-sanitizer --tool synccheck --error-exitcode 1 \
  build-glm5/src/glm5/glm5_indexer_test --kernel-check
```

`--kernel-check` bounds sanitizer cost while retaining ragged tiles, 50000
pool selections, tied scores, signed zeros, infinities, tails and padding.
The normal test also checks 32-head synthetic models with actual sparse
pruning, both tail policies, MTP off/on with acceptance and rejection, state handoff,
appended prompts and reset. Score and selection comparisons require exact
bytes; CPU stable sorting independently checks selected positions.

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

Set `--max-context` for input plus requested output, not just the microbatch
size. The standalone implementation limit is 262144. The default remains
4096. An implementation limit is not a model-quality or memory-fit guarantee;
the historical reference's 200k result is not a measurement of this engine. Use
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

### Complete-layer scheduler checks, 4 November 2026

The measurements in this subsection describe the initial opt-in scheduler,
before the simplified full-layer defaults and priority order below. Use
`--prefill-reference` to reproduce its synchronous reference configuration.

The CUDA SM86 build and all nine GLM CTests passed with the scheduler changes.
The expanded full-layer test checks grouped and prefetched banks, pipelined
transfers, dense and sparse attention, Flash, MTP off/on, appended prefixes,
prompt heat and decode continuation. It checks actual group/batch sizes,
one upload per weight, skipped final residual downloads and tracked budgets.
Automatic planning reduces a 256-column request to 128 under a 1 MiB
transfer limit; manual mode rejects the same oversized request.
The 2305-token sparse attention test also checks grouped execution, actual
pruning, pooled indexer state and decode continuation.

Passed commands, using the CUDA build instructions above:

```sh
ctest --test-dir build-glm5 \
  -R '^glm5_(format|cache|parity|head|attention|runtime|full_layer|gemm|batch_runtime)$' \
  --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 --leak-check full \
  build-glm5/src/glm5/glm5_full_layer_test
compute-sanitizer --tool synccheck --error-exitcode 1 \
  build-glm5/src/glm5/glm5_full_layer_test
```

Memcheck reported zero errors and zero leaked bytes; synccheck reported zero
errors. The host-only format/cache build passed both tests. Strict C++20
syntax checks with `-Wall -Wextra -Werror` passed for the changed runtime and
test sources. Eight CLI rejection cases and all five new help entries passed.
HIP builds and other GPU architectures were not run.

#### Matched real-model prefill measurements

The six-shard Huihui Q4_K_XL model above ran on the same RTX 3080 reporting
20 GiB and Xeon W-2133. Every run used the same 2048 IDs, microbatch 512,
context capacity 4096, DSA/Flash, tail policy 0 and MTP off. The input was
an original repeated English paragraph about telescope optics and scheduling,
not a representative quality set. Its CSV SHA256 was
`0114993c50dab473402a4692310d5761068c9f05ecbdad5e0ccafae50d1d80fe`.
No generation was requested.

Tracked GPU capacity was 14336 MiB, host capacity 4096 MiB, GPU reserve
1024 MiB and transfer capacity 512 MiB. K-quant MMQ was enabled, CUDA modules
loaded lazily, TF32 disabled and OpenMP wait policy passive. Runs were
sequential in a shared environment, in the table's order twice. No file
cache was dropped. The first serial run reported 2776 filesystem input
blocks; the other seven reported zero.

| Scheduling | Prefill tokens/s, two runs | Tracked GPU peak, MiB | Prefetched layers |
|---|---:|---:|---:|
| Single-layer synchronous reference | 51.139 / 51.018 | 6151.56 | 0 |
| Two-layer groups | 51.579 / 51.237 | 10502.67 | 0 |
| Single-layer groups, one-layer lookahead | 64.430 / 64.446 | 10804.39 | 44 |
| Automatic, two-layer groups, lookahead and pipeline | 63.165 / 63.890 | 14046.44 | 19 |

These are tracked allocations, not sampled total process VRAM. All eight
complete 154880-entry logits files were finite and byte-identical.
The optimized rows each reported 185436.82 MiB of tensor payload uploads;
the regression checks require one upload per weight, including prefetch.
The optimized rows also used the final-group head shortcut, so the lookahead
comparison is not an isolated copy-kernel benchmark.

By pooled prefill elapsed time, lookahead improved throughput by 26.16%;
the combined path improved it by 24.37%. Two-layer grouping alone improved
it by only 0.65%, which does not establish a meaningful general speedup.
Grouping reduced trunk boundary residual uploads from 5760 to 2944 MiB
and downloads from 5760 to 2816 MiB with the fused head. The combined
planner selected 23 groups at batch 512 in one attempt. Capacity planning
does not choose the fastest row in this table.

A separate real-model check used a ragged 257-token prefix, batch 256,
sparse Flash, MTP depth 3 and six CPU expert threads. Serial and combined
scheduling produced byte-identical prefill logits and the same eight
generated IDs. This compares scheduling policies with MTP enabled in both;
it does not establish lossless MTP versus non-MTP or reference-model parity.

Nsight Systems 2026.1.3 profiled a separate combined-path run. Its 157383
pinned weight H2D copies transferred 82348972552 bytes. Copy activity totaled
10771.23 ms, of which 4298.74 ms (39.91%) overlapped actual GPU kernel
intervals. Activation uploads overlapped kernels for 280.34 of 386.70 ms
(72.50%); downloads overlapped for 85.20 of 353.13 ms (24.13%).
These measurements establish real transfer/compute overlap, not equivalent
percentage throughput gains. The profiled run is excluded from the timing
table.

### Full-layer defaults and pipeline memory checks

The simplified full-layer defaults were checked on the same hardware,
Q4_K_XL weights and 2048 input IDs. This time the reference used an explicit
batch 2048, and the automatic modes used only `--prefill-full-layer` to
select scheduling. Context capacity, budgets and environment matched the
preceding measurements. Runs were sequential in reference/default/no-pipeline
order twice. The first reference run reported 264 filesystem input blocks;
the remaining five reported zero.

| Scheduling | Prefill tokens/s, two runs | Tracked GPU peak, MiB | Host tensor bound, MiB |
|---|---:|---:|---:|
| Reference, explicit batch 2048 | 47.402 / 47.200 | 8737.23 | 299.40 |
| Full-layer defaults | 58.339 / 58.547 | 13646.06 | 300.40 |
| Full-layer defaults, `--prefill-no-pipeline` | 57.715 / 57.741 | 13390.06 | 300.40 |

Both automatic modes kept actual 2048-column products, selected 41 groups
(at most 4 layers) and prefetched 40 layers. Disabling pipeline saved
exactly 256 MiB of tracked GPU allocations, while the host tensor bound
stayed unchanged. Process peak RSS was about 182.05 GiB in all six runs,
including mapped model pages; it did not drop by the host activation size.
Pipeline changes the activation array's pinned/pageable status, not its size.
At this input length and batch there is only one microbatch per group,
so this is not a long-prompt activation-overlap benchmark.

All six complete logits files were finite and byte-identical. The automatic
default was 23.56% faster than the matched batch-2048 reference by pooled
prefill time. Pipeline on/off differed by about 1.24% in these short checks;
this is not evidence that pipeline is faster on every system or workload.
The earlier batch-512 lookahead result was faster than this batch-2048
result. The requested policy prioritizes batch capacity, not measured
latency, and does not claim the largest batch is always fastest.

The four affected CUDA CTests (`attention`, `runtime`, `full_layer` and
`batch_runtime`) passed. The full-layer test sweeps 8 through 24 MiB GPU
caps and requires prefetch to take priority over grouping without reducing
the single-layer batch. It also checks that resident defaults remain at
batch 1, full-layer defaults enable automatic scheduling, disabling pipeline
saves exactly its two device buffers without reducing the host tensor bound,
and insufficient complete-layer capacity remains an explicit error.
The large-batch test checks actual 2048-column products through the default
full-layer configuration. Six CLI rejection checks and the new help entries
passed. Strict C++20 syntax/warning checks passed for the changed sources.
The updated full-layer test also passed memcheck with zero errors and zero
leaked bytes. HIP and other GPU architectures remain untested.

### 200000 input tokens and 128 generated tokens, 5 November 2026

The same six-shard Huihui Q4_K_XL model completed one 200000-token input
and generated exactly 128 tokens on the RTX 3080 reporting 20 GiB,
Xeon W-2133 and 247 GiB RAM. Exit status was zero. This is a single
performance run in a shared environment, not a quality evaluation.

The input repeats the earlier 2048-ID telescope/scheduling prefix and
truncates the repeated ID sequence to exactly 200000 entries. Repeating
token IDs does not re-tokenize or add role markers at each boundary.
Its CSV SHA256 is
`dd9eba14fbe5397106a039a5ec6b22525706e7a29c584277fb015a3c81031cb8`.
The prompt is synthetic and repetitive, not a representative natural
200k-token conversation.

The standalone context guard was raised from 131072 to 262144 for this
check, using one shared bound across CLI, model and decoder. Arithmetic,
attention selection and cache formats did not change. Synthetic
checkpoint tests append across positions 131072 and 200000, compare
scheduled and synchronous state/logits, and continue sparse decoding.
Those tests check addressing and handoff, not the content of a real
200k-token prefix. The real-model run below evaluated all 200000 input
rows; it did not substitute a synthetic checkpoint.

Use the first shard as `MODEL` and the saved 200000 CSV IDs as `INPUT`:

```sh
STRATA_MMQ_KQ=1 CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0 \
OMP_WAIT_POLICY=PASSIVE /usr/bin/time -v \
  build-glm5/src/glm5/strata-glm5 \
  --model "$MODEL" --ids-file "$INPUT" --max-context 200128 \
  --prefill-full-layer --prefill-host-mib 32768 --prefill-reserve-mib 1024 \
  --dsa --flash-attn --dsa-tail 0 --cpu-threads 6 \
  --glm-gpu-experts 4096 --glm-gpu-reserve 1024 --glm-gpu-warm \
  --mtp 0 --n 128 --timings --dump-logits prefill-200k.f32
```

The prefill GPU cap was automatic: current free memory less 1024 MiB.
The host tensor cap was explicitly increased to 32 GiB; the default
4 GiB cannot hold this prompt's F32 activations. Pipeline, prefetch and
grouping used the full-layer defaults. MTP was off; this is ordinary
decode with six CPU expert threads and a 4 GiB expert-cache cap,
prewarmed from prompt routes.

| Phase | Elapsed | Throughput |
|---|---:|---:|
| Model initialization | 0.014 s | Not applicable |
| Prefill, including planning and layer loads | 3412.681 s (56 min 52.7 s) | 58.605 tokens/s |
| Resident decode setup and cache warmup | 2.393 s | Not applicable |
| Generation, 128 tokens | 64.757 s | 1.977 tokens/s |
| Whole process, including teardown (`time -v`) | 58 min 1.11 s | Not a decode-only rate |

Observed matrix products reached 2048 columns. Capacity planning admitted
batch 2048 in one attempt, selected 22 groups of up to four layers and
prefetched 21 complete layers. The final head did not fit alongside the
last group and ran separately. Trunk weights/head uploaded 185436.82 MiB
of payload. Trunk group-boundary residual traffic was 275000 MiB in each
direction. These counters do not include state and logits transfers.

| Resource | Observed value |
|---|---:|
| Tracked prefill GPU peak | 17385.26 MiB (16.98 GiB) |
| Sampled process GPU peak, nominal 2-second interval | 18590 MiB (18.15 GiB) |
| Conservative prefill host tensor bound | 15064.69 MiB (14.71 GiB) |
| Peak process RSS, including mapped weight pages | 196.47 GiB |
| Pipeline device buffers | 256 MiB |

The two-second samples are not exact allocation maxima and exclude other
GPU processes. Peak RSS is not the pinned activation size: the activation
array itself occupies about 12.21 GiB at this input length.
`time -v` reported 264 filesystem input blocks, two major page faults and
zero swaps. File caches were not dropped before the run.

Decode reported 16111 resident expert hits and 26897 CPU misses
(37.46% hits), 132 replacements, and 252 prewarmed slots. All 154880
prefill logits were finite, and the saved output contains exactly
128 token IDs. This run does not establish broad output quality,
reference-model parity, or a controlled speed comparison against the
historical llama.cpp 200k workload.

The CUDA build and the four affected GLM regressions passed, including
the new high-position cases. Strict C++20 syntax/warning checks passed.
Memcheck of the full-layer regression, including high-position cases,
reported zero errors and zero leaked bytes. This was not a sanitizer
run of the full real-model 200k workload.
HIP, other GPU architectures, MTP at 200k, and a repeated real-model
200k timing were not run.

### Exact-order F32 indexer optimizations, 5 November 2026

This follow-up implements the first compute-optimization batch: cooperative
top-k and tiled F32 scoring. Complete-layer scheduling, expert execution,
KDA, Flash arithmetic and activation precision are unchanged. Tensor-core
indexer scoring, grouped multi-expert MMQ and larger expert-aggregation
batches are not part of this change.

CUDA SM86 builds passed. All ten GLM CTests passed, including the new
`glm5_indexer` entry. The focused test compares complete score bytes and
sorted cell arrays, with independent CPU stable sorting. It covers ragged
tile dimensions, 50000 pools, short prefixes, ties, signed zeros,
infinities, tails, empty selections and guard regions. A 32-head miniature
model checks actual pruning, both tail policies, MTP acceptance/rejection,
prefill/decode state, appended prompts and reset. Defaults and nested
launch-policy restoration are checked separately.

Memcheck reported zero errors and zero leaked bytes. Bounded kernel
racecheck reported zero hazards; synccheck reported zero errors. An initial
racecheck of the full long-shape timing loop was stopped because of
instrumentation cost and is not counted as passed. Three CLI rejection
checks and strict C++20 syntax/warning checks passed. HIP and other GPU
architectures were not built or tested.

#### Kernel timings

On the RTX 3080 reporting 20 GiB, `glm5_indexer_test --bench` measured
64 queries against 50000 pooled keys, with 32 F32 query heads of width
128 and 512 selected pools. CUDA events measured the mean of three
calls after warmup:

| Kernel | Existing path | Optional path | Ratio |
|---|---:|---:|---:|
| Score | 105.928 ms | 1.894 ms | 55.93x |
| Top-k | 28.120 ms | 0.289 ms | 97.15x |

Both comparisons required exact output bytes. These synthetic kernel
ratios are not full-model throughput gains. GPU clocks and other users
were not isolated.

#### Real-model comparisons

The same Huihui Q4_K_XL six-shard model ran on the RTX 3080 20 GiB,
Xeon W-2133 and 247 GiB RAM. Inputs are exact prefixes of the saved
200000-ID repeated-text workload above. Batch capacity was 2048,
activation staging F32, latent cache F16, DSA/Flash on and tail policy 0.
K-quant MMQ was enabled, CUDA modules loaded lazily, TF32 disabled,
OpenMP wait policy passive, and host capacity 32768 MiB. Prefill capacity
was current free VRAM minus a 1024 MiB reserve.

| Input | Existing path, tokens/s | Both optimizations, tokens/s |
|---|---:|---:|
| 4096, one run per policy | 113.37 | 119.72 |
| 32768, two runs per policy | 155.07 / 154.95 | 227.94 / 228.37 |
| 200000, one run per policy | 58.60 (preceding run) | 243.77 |

At 4096 tokens, top-k alone measured 116.63 tokens/s and tiled scoring
alone 116.57. All four policy runs returned byte-identical complete
logits and reported zero filesystem input blocks. These single short
measurements do not establish small-prompt speedups on other systems.

The unprofiled 32768-token runs used baseline/combined/combined/baseline
order. Both policies selected batch 2048, 21 groups, 20 prefetched layers,
and a fused final head. Matrix counts and residual transfer bytes matched;
tracked GPU peak was 18038.43 MiB for both. All four complete logits
files were byte-identical. Pooled throughput improved by 47.19%.

The 200000-input/128-output run repeated the preceding full command,
adding only `--dsa-score tiled --dsa-parallel-topk` and a new logits filename.
It completed with status zero and exactly 128 output IDs:

| Phase/resource | Existing run | Optimized run |
|---|---:|---:|
| Prefill | 3412.681 s | 820.452 s (13 min 40.5 s) |
| Decode setup | 2.393 s | 2.201 s |
| 128-token decode | 64.757 s, 1.977 tokens/s | 31.370 s, 4.080 tokens/s |
| Whole process (`time -v`) | 58 min 1.11 s | 14 min 15.23 s |
| Tracked prefill GPU peak | 17385.26 MiB | 18028.13 MiB |
| Sampled process GPU peak, 2-second interval | 18590 MiB | 18314 MiB |
| Peak RSS, including mapped weights | 196.47 GiB | 196.47 GiB |

The complete 154880-entry prefill logits and all 128 generated token IDs
were byte-identical to the preceding run. Cache hits, misses, replacements
and warmed-slot counts also matched. MTP remained off; the new options
also accelerate the indexer during ordinary decode.

The observed prefill ratio was 4.16x and decode ratio 2.06x. This is not
a strict isolated-kernel A/B: runs were separated in time, available VRAM
was not fixed, and the optimized run admitted a fused final head where
the previous run used a separate head. Both used 22 groups, 21 prefetched
layers and identical matrix call counts, but the fused head reduced
residual downloads from 275000 to 262500 MiB. The optimized run reported
zero filesystem input blocks; the earlier run reported 264.
The controlled 32k comparison above provides the same-plan evidence.
None of these repeated-text checks establishes broad language quality.

#### Complete CUDA profile

Nsight Systems 2026.1.3 captured separate complete 16384-token prefill
runs with a fixed 14336 MiB GPU cap and 2048 MiB reserve. Profiling used
`--trace=cuda --sample=none --cpuctxsw=none --cuda-event-trace=false
--cuda-flush-interval=1000 --cuda-graph-trace=node`, without a delayed
capture window. Both full traces had no warning diagnostics and retained
2816 score calls and 2816 top-k calls, exactly
`16384 / 64 * 11`. Complete logits matched byte-for-byte.
Profiled wall times are excluded from the throughput table.

| GPU kernel family | Existing summed duration | Optimized summed duration |
|---|---:|---:|
| Indexer score | 12.686 s | 0.252 s |
| Top-k | 6.308 s | 0.093 s |
| KDA recurrence | 10.577 s | 10.567 s |
| Attention | 5.091 s | 5.064 s |
| Activation quantization | 8.691 s | 8.719 s |
| Quantized matrix products | 8.495 s | 8.414 s |
| Other kernels | 4.236 s | 4.198 s |
| All kernels | 56.084 s | 37.308 s |

Indexer scoring plus selection fell from 33.87% to 0.93% of summed
kernel duration in this 16k workload. These are sums of device intervals,
not percentages of process wall time or measurements of the full 200k run.
The remaining measured GPU targets include KDA recurrence, activation
quantization and matrix products.

Both traces also recorded 11011416 device-to-device copies and 11362985
`cudaMemcpyAsync` API calls. Their counts did not change: per-row expert
gather/scatter is still present. Host API spans include queue backpressure
and are not pure CPU-math time. This supports investigating batched
expert row movement next, without claiming its speedup in advance.

Earlier delayed 32k captures were not usable for a complete kernel
breakdown: an automatic-cap run failed under profiler memory overhead;
bounded delayed captures lost CUDA events, including all baseline kernel
records. Those captures are excluded. The complete fixed-budget profiles
above replaced them; no missing trace records were treated as zero work.

### Batched expert row movement, 5 November 2026

The next opt-in step is `--prefill-expert-rows`. It replaces only row
movement inside complete-layer expert execution. Every expert still uses
the same input row order, matrix products, SwiGLU clamp and output slots.
Gate/up remain separate products; there is no grouped multi-expert MMQ
or fused gate/up arithmetic in this change. Resident decode is unchanged.

For each layer batch, the host builds the same ordered expert groups and
uploads one pinned map of `{input token, output row}` pairs. Each expert
batch uses one GPU gather and one scatter kernel instead of one
`cudaMemcpyAsync` call per row in each direction. The map is reused by
NextN prompt catch-up. Its pinned and device allocations are bounded;
the full-layer regression checks their exact host, GPU and transfer
budget increments. Upload completion protects the pinned map on teardown.

#### Validation

The CUDA SM86 build and all eleven GLM CTests passed. The new
`glm5_expert_rows` test checks widths 1/17/256/4096, repeated input rows,
permuted output rows, untouched output slots and guards. Complete expert
arithmetic is compared byte-for-byte at capacities 8/17/64 and request
sizes 0/1/7/8/17/65/129, covering MMVQ, large GEMM and capacity splitting.
The existing full-layer suite adds batched-row modes across dense/sparse
attention, Flash and MTP. The 32-head indexer test now combines both
indexer optimizations with batched expert movement and checks exact
state/logit bytes, MTP acceptance/rejection and appended prompts.

Passed after building:

```sh
ctest --test-dir build-glm5 \
  -R '^glm5_(format|cache|parity|head|indexer|expert_rows|attention|full_layer|runtime|gemm|batch_runtime)$' \
  --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 --leak-check full \
  build-glm5/src/glm5/glm5_expert_rows_test
compute-sanitizer --tool racecheck --error-exitcode 1 \
  build-glm5/src/glm5/glm5_expert_rows_test
compute-sanitizer --tool synccheck --error-exitcode 1 \
  build-glm5/src/glm5/glm5_expert_rows_test
compute-sanitizer --tool memcheck --error-exitcode 1 --leak-check full \
  build-glm5/src/glm5/glm5_full_layer_test
```

Memcheck reported zero errors and zero leaked bytes for both executables.
Racecheck reported zero hazards; synccheck reported zero errors.
Strict C++20 syntax/warning checks, CLI rejection without full-layer mode,
and the help entry passed. HIP and other GPU architectures were not built
or tested.

#### Fixed-budget real-model measurements

The same six-shard Huihui Q4_K_XL model, RTX 3080 reporting 20 GiB,
Xeon W-2133 and 247 GiB RAM were used. All modes enabled the preceding
F32 tiled scorer and parallel top-k, with DSA/Flash, tail policy 0,
microbatch 2048 and a fixed 14336 MiB prefill GPU cap. Host capacity was
32768 MiB, GPU reserve 1024 MiB. Environment remained
`STRATA_MMQ_KQ=1 CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0
OMP_WAIT_POLICY=PASSIVE`. Inputs are exact prefixes of the saved 200k
repeated-text ID workload. No file caches were dropped.

Each 4k/32k comparison ran reference/batched/batched/reference, without
overlapping GPU tests or profiling. Both policies used the same batch,
41 groups, 40 prefetched layers, fused final head, matrix call counts,
weight payload bytes and residual transfer bytes. All eight complete
logits files were byte-identical to their same-length reference, and all
runs reported zero filesystem input blocks.

| Input tokens | Per-row copies, tokens/s | Batched row kernels, tokens/s | Pooled ratio |
|---|---:|---:|---:|
| 4096 | 125.32 / 124.96 | 126.69 / 126.94 | 1.013x |
| 32768 | 243.56 / 243.18 | 338.14 / 338.19 | 1.390x |

These are shared-environment measurements, not universal speedups.
The 4k difference is small and does not establish a general short-prompt
improvement. The extra route map adds 128 KiB each to tracked GPU and
host storage at batch 2048; it does not duplicate expert activations.

#### Same-budget 200k/128 comparison

Both runs used the preceding 200k/128 command with
`--dsa-score tiled --dsa-parallel-topk --prefill-gpu-mib 14336`.
The second added only `--prefill-expert-rows` and changed the output
filenames. CPU expert threads were 6, decode expert-cache cap 4096 MiB,
prompt cache warmup enabled, MTP off, and context capacity 200128.
Each run exited zero and generated exactly 128 IDs.

| Measurement | Per-row copies | Batched row kernels |
|---|---:|---:|
| Prefill | 834.099 s, 239.780 tokens/s | 702.289 s, 284.783 tokens/s |
| Decode setup | 2.535 s | 2.371 s |
| 128-token generation | 35.882 s, 3.567 tokens/s | 35.224 s, 3.634 tokens/s |
| Whole process (`time -v`) | 14 min 33.78 s | 12 min 21.12 s |
| Tracked prefill GPU peak | 13911.87 MiB | 13911.99 MiB |
| Host tensor bound | 15064.69 MiB | 15064.82 MiB |
| Sampled process GPU peak, 2-second interval | 15152 MiB | 15152 MiB |
| Peak RSS, including mapped weight pages | 196.47 GiB | 196.47 GiB |

Both retained batch 2048, 41 groups, 40 prefetched layers and a fused
final head. Their complete 154880-entry prefill logits and all 128 output
IDs were byte-identical, including comparison against the original slow
200k run. Decode cache hits/misses, replacements and warmup counts matched.
The fixed-cap pair removes the earlier final-head admission difference.
Prefill throughput improved by 18.77% in this single pair; no repeated
200k A/B was run. The option does not change ordinary decode, so its
small decode-time difference is not attributed to this optimization.
Both runs reported zero filesystem input blocks and zero swaps.

The optimized prefill took 11 min 42.3 s. The lower GPU cap selects more
groups than the earlier automatic-cap test; do not treat different-budget
rows as an isolated scheduling comparison. These repetitive prompts do
not establish broad language quality or reference-model equivalence.

#### Complete 16k CUDA profiles

Two separate 16384-token runs used the same fixed 14336 MiB GPU cap,
2048 MiB reserve, batch 2048 and tiled/parallel indexer policies.
Nsight Systems 2026.1.3 captured complete traces with the no-delay,
periodically flushed settings above. Both had zero warning diagnostics
and byte-identical logits. Profiled wall times are excluded from the
unprofiled timing tables.

| Trace measurement | Per-row copies | Batched row kernels |
|---|---:|---:|
| Device-to-device copy calls | 11011416 | 1368 |
| Device-to-device copy activity | 15.013 s | 0.544 s |
| `cudaMemcpyAsync` API calls | 11362985 | 353273 |
| Gather kernels | None | 65593 calls, 0.597 s |
| Scatter kernels | None | 65593 calls, 0.436 s |

H2D copy count increased by 336, one packed map per MoE layer batch
(`42 * 8`). Other device-to-device copies remain, including attention
projection staging; no claim is made that every remaining copy is
necessary. Device row movement is now counted as kernel work rather
than DMA-copy work, so summed kernel time alone is not a suitable
before/after speed metric. Host API spans include queue backpressure,
not just CPU submission cost.

The trace verifies that the per-row copy-call problem was removed.
Further work on multi-expert MMQ and fused products remains separate,
as does reducing the existing CPU routing synchronization.

### Multi-expert MMQ experiment was removed

The discarded prototype reduced 4096-token MMQ calls from 43610 to 8076,
but final logits differed by maximum absolute error 1.30845 and RMSE
0.23377. A stricter variant preserved exact logits but admitted few
products: at 32768 tokens total MMQ calls fell only from 264949 to
264592. Its reference/grouped/grouped/reference throughput samples were
307.26 / 274.00 / 274.33 / 278.64 tokens/s, a pooled ratio of 0.938x.
The reference itself varied substantially in the shared environment.

Neither result justified retaining the experiment. Its CLI option,
private executor, test and shared accumulation-order query were removed.
The retained expert-row optimization still computes individual experts
and does not depend on that query.

### KDA and activation preparation implementation

The automatic policy selects these paths on validated CUDA SM86:

- Native activation preparation omits the layer composer's legacy Q8_K/Q8_0
  activation images. The standalone native projection helpers do not read
  these images: MMVQ/MMQ create their own q8_1 inputs, FP16 GEMM converts
  from F32, and SGEMM reads F32. The option does not remove those required
  native conversions, CPU-expert quantization or BF16 preparation. It
  neither changes model weights nor caches an activation solely by pointer.
  Existing activation arenas are retained, so no persistent VRAM saving
  is claimed.
- Register-state KDA uses a recurrence specialized for head dimension
  128. Each block owns 32 state rows; four blocks cover one head. State
  rows remain in registers across the token loop and are written back once
  at the end. Tokens remain sequential. The per-column accumulation order,
  scale placement, beta sigmoid, bounded decay and per-token state clamp
  are unchanged. Other head dimensions retain the original recurrence.

Both paths apply to standalone full-layer prefill and resident decode,
including target verification and KDA rejection replay with MTP. NextN
uses the same activation-preparation policy. Their host-thread execution
scope restores previous settings on nested calls and errors; another
Model does not inherit these policies.

No additional speed switches are needed. For a developer comparison:

```text
--reference --prefill 2048
```

The register kernel uses no new persistent arena. Its block-local shared
memory and registers are GPU execution resources, not allocations that
can be handed to the expert cache. Fixed-index unrolling preserves the
state in registers; shared operands remain volatile to prevent the
compiler from also keeping all query/key/decay values live there.
CUDA SM86 resource inspection of the final kernel reported 168 registers,
1536 shared-memory bytes and zero stack/local bytes. An earlier build
spilled to a 680-byte stack and was replaced before final measurements.

The focused checks and synthetic timing commands are:

```sh
cmake --build build-glm5 --target glm5_compute_test
build-glm5/src/glm5/glm5_compute_test
build-glm5/src/glm5/glm5_compute_test --bench
compute-sanitizer --tool memcheck --error-exitcode 1 --leak-check full \
  build-glm5/src/glm5/glm5_compute_test
compute-sanitizer --tool racecheck --error-exitcode 1 \
  build-glm5/src/glm5/glm5_compute_test --kernel-check
compute-sanitizer --tool synccheck --error-exitcode 1 \
  build-glm5/src/glm5/glm5_compute_test --kernel-check
```

The focused test requires exact output/state bytes and guard regions
against the original recurrence, checks a small independent double
reference, and covers nonzero initial states, clamp stress, ragged token
counts, 1/2/64 heads, long token loops, unsupported dimensions and nested
execution scopes. Full-layer and 32-head sparse/MTP tests also combine
both controls with the preceding indexer and expert-row options.

### KDA and unused-quantization measurements, 5 November 2026

The final CUDA SM86 register kernel passed exact output and state checks
against the original recurrence. On the RTX 3080 reporting 20 GiB,
`glm5_compute_test --bench` measured 64 heads, dimension 128 and
2048 tokens, using CUDA events and three calls after warmup:

| Recurrence | Mean device time |
|---|---:|
| Original global-state kernel | 40.827 ms |
| Register-state kernel, no spills | 8.581 ms |

This 4.76x ratio is a synthetic recurrence result, not a full-model
speedup. An earlier spilled register build measured 22.180 ms and was
replaced after inspecting compiled resource usage. The final implementation
retains the scalar per-column order rather than parallelizing reductions.

#### Fixed-budget full-model tests

The same six-shard Huihui Q4_K_XL model ran on the RTX 3080 20 GiB,
Xeon W-2133 and 247 GiB RAM. Every policy enabled tiled F32 indexer
scoring, parallel top-k and expert-row kernels; guarded grouped MMQ was
off. Inputs are exact prefixes of the saved 200000-ID repeated-text
workload. Microbatch was 2048, GPU cap 14336 MiB, host cap 32768 MiB,
GPU reserve 1024 MiB, DSA/Flash on and tail policy 0. Environment was
`STRATA_MMQ_KQ=1 CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0
OMP_WAIT_POLICY=PASSIVE`. File caches were not dropped.

| Input tokens | Reference | Skip unused quant only | Register KDA only | Both |
|---|---:|---:|---:|---:|
| 4096, tokens/s | 126.85 | 127.40 | 127.33 | 130.33 |
| 32768, tokens/s | 341.99 / 338.13 | 401.72 | 409.98 | 497.53 / 498.15 |

At 4096 tokens the policies ran in table order once. At 32768 tokens,
the order was reference/quant/KDA/both/both/reference. Both versus the
two reference runs improved pooled prefill throughput by 46.40%.
The isolated quant/KDA rows are single samples. The small 4k differences
do not establish general short-prompt improvements.

All ten complete logits files were finite and byte-identical to their
same-length reference. Matrix dispatch counts, batch, group plan and
weight/activation transfer counts matched. Both policies used 41 groups,
40 prefetched layers and a fused final head. Tracked GPU peak stayed
13648.94 MiB at 4k and 13687.44 MiB at 32k. All runs reported zero
filesystem input blocks. GPU clocks and other users were not isolated.

#### 200000 input tokens, 128 output tokens

One additional run repeated the prior fixed-budget batched-row 200k/128
command, adding `--skip-unused-quant --kda-register-state` and changing
only output filenames. It used context capacity 200128, six CPU expert
threads, a 4096 MiB decode expert-cache cap, prompt expert warmup, and
MTP off.

| Measurement | Previous batched-row run | Both compute controls |
|---|---:|---:|
| Prefill | 702.289 s, 284.783 tokens/s | 414.737 s, 482.234 tokens/s |
| Decode setup | 2.371 s | 2.199 s |
| 128-token generation | 35.224 s, 3.634 tokens/s | 27.754 s, 4.612 tokens/s |
| Whole process (`time -v`) | 12 min 21.12 s | 7 min 25.90 s |
| Tracked prefill GPU peak | 13911.99 MiB | 13911.99 MiB |
| Sampled process GPU peak, 2-second interval | 15152 MiB | 15152 MiB |
| Peak RSS, including mapped weight pages | 196.47 GiB | 196.47 GiB |

Prefill took 6 min 54.7 s, about 1.693x the preceding throughput.
This comparison uses an earlier baseline rather than a new 200k ABBA
sequence. The current 32k repeated comparison provides the tighter
same-session evidence. This is not a matched comparison with llama.cpp.

The run exited zero, evaluated exactly 200000 input rows and generated
exactly 128 IDs. All 154880 prefill logits and all generated IDs were
byte-identical to the original slow 200k run. The same 2048-column
products, 41 groups, 40 prefetched layers, fused final head, matrix
counts, weight payload and activation boundary traffic were observed.
Decode cache hits, misses, replacements and warmed-slot counts matched.
The run reported zero filesystem input blocks, major faults and swaps.
Both controls also affect resident decode, but the single decode timing
does not establish which control accounts for its speed change.

#### Full 16k CUDA traces

Separate complete 16384-token profiles used the fixed 14336 MiB GPU cap
and 2048 MiB reserve, with the no-delay Nsight Systems settings described
above. Both captures had no warning diagnostics and byte-identical
complete logits:

| Kernel family | Reference calls / summed time | Both controls calls / summed time |
|---|---:|---:|
| KDA recurrence | 272 / 10.754 s | 272 / 2.276 s |
| Legacy Q8_K/Q8_0 image quantization | 5392 / 7.555 s | 0 / 0 s |
| Required native activation quantization | 188433 / 1.162 s | 188433 / 1.163 s |
| Attention | 2816 / 5.181 s | 2816 / 5.156 s |
| Quantized matrix products | 187739 / 8.606 s | 187739 / 8.597 s |
| Other kernels | 501668 / 5.654 s | 501668 / 5.648 s |
| All kernels | 38.912 s | 22.839 s |

The trace verifies that the unused image kernels disappear while required
native quantization and matrix products remain. These are summed device
durations, not wall-time percentages or a profile of the complete 200k
run. Profiled wall times are excluded from the throughput table.
Further reuse of native quantization across compatible projections is
not implemented by the legacy-image omission.

#### Validation and limits

The CUDA SM86 build and all thirteen GLM CTests passed. The full-layer
suite checks each compute control separately and together with exact
head, NextN, KDA, MLA and indexer state comparisons. The 32-head sparse
suite combines them with actual pruning and MTP acceptance/rejection,
append/reset and decode continuation. The focused KDA test covers
clamp stress and compares a small case's output and final state to an
independent double reference, in addition to exact original-kernel checks.

Memcheck reported zero errors and zero leaked bytes for the focused
compute test and combined full-layer suite. Bounded register-kernel
racecheck reported zero hazards; synccheck reported zero errors.
A separate real Q4 4096-token MTP3 smoke produced byte-identical complete
prefill logits and the same 16 output IDs as the earlier reference.
All 12 proposed drafts were accepted in that smoke; rejection is covered
by the synthetic tests, not by this real prompt. These checks do not
establish equivalence between MTP and non-MTP.

These measurements used explicit private switches before consolidation.
The published normal policy now selects the same paths automatically on
validated CUDA SM86 geometry. The 200k workload is repetitive
and only one new timing was run; general text quality, other CPU/GPU
systems, HIP builds and other CUDA architectures were not evaluated.
No FP16/TF32 shortcut, state approximation or changed reduction order
was used to obtain these results.

### Automatic-policy cleanup and publication checks

The retained optimizations are now selected by the GLM-private runtime,
not individual CLI speed switches. CUDA SM86 and supported geometry
select the previously measured kernels; other device architectures keep
the prior compute dispatch. Policy tests simulate SM70/75/80/89/90/120
and HIP's zero CUDA-architecture value to verify fallback selection.
These are policy checks, not hardware execution on those devices.

`--reference` is the single normal developer comparison entry. The
pre-existing `--prefill-reference` alias remains available for full-layer
comparisons. Redundant `--prefill-auto` and `--prefill-pipeline` spellings
are accepted for compatibility but cannot be combined with `--reference`.
Resource controls, DSA/Flash semantics, CPU threads, MTP and expert-cache
choices are unchanged. The CLI logs the actual resolved compute policy.

All multi-expert MMQ experiment code was removed. The shared
`include/strata/prefill/moe_mmq.hpp` and `src/prefill/moe_mmq.cu` files
match the pre-experiment committed versions exactly. No retained change
modifies shared MMQ execution or another model family.

After cleanup, all twelve current GLM CUDA CTests passed. The indexer
suite compares normal automatic execution to explicit private optimized
policies and the reference, including exact state/logit checks. The
compute test covers device/geometry policy selection, diagnostic
overrides and nested/error scope restoration. Strict C++20 warning/type
checks passed.

A normal 32768-token command, without any individual speed flags, selected
batch 2048, tiled scoring, cooperative top-k, register KDA, native-only
activation preparation and expert-row kernels on the RTX 3080. Its
complete logits matched the earlier fully optimized dump byte-for-byte.
A 4096-token `--reference --prefill 2048` command selected the original
compute paths and likewise matched the saved logits exactly. These were
selection/equality checks, not new isolated throughput comparisons.
The removed CLI switches are rejected, and help exposes the consolidated
reference/resource controls.
Memcheck of the cleaned full-layer suite reported zero errors and zero
leaked bytes. Reference/automatic-planner conflict checks passed. HIP
and other device architectures were not built or hardware-tested.

### Ordered upload, quantization, scratch and routing follow-up

These changes remain inside the GLM source, interfaces and tests.
There are no new CLI speed switches and no shared MMQ changes.
The existing validated SM86 policy selects the compute changes.
`--reference` keeps the original quantization arithmetic.

#### Changes in order

1. **Upload staging.** Weight banks count actual upload calls. A private
   staging target is admitted only after batch, prefetch and group selection.
   The ring shrinks to the remaining host/transfer budget before losing
   prefetch or reducing the batch. The normal target remains 1 MiB.
   Larger rings are available to executor tests, not as a new CLI switch.
2. **Native gate/up input reuse.** An explicit immutable-input scope shares
   quantized activations only for the same input, format, width and column
   count. The scope ends before SwiGLU or another expert changes the input.
   No cache survives scope exit. Both products still use their original
   weights, dimensions and individual `n=1` MMQ launches.
3. **Scratch reduction.** Full-layer execution no longer allocates its unused
   single-token workspace. The optimized layout also omits unconsumed legacy
   Q8_K/Q8_0 images. Reference images and live native/BF16 inputs remain.
   Persistent KDA, MLA and NextN state layouts are unchanged.
4. **MLA preparation.** Compatible head-major inputs use one native
   quantization launch with separate ggml channels for each head. Flattening
   heads into one row dimension is not equivalent: it changes the K-block
   stride. Each subsequent head product keeps its original column count,
   weight band and accumulation schedule. The larger temporary image is
   charged to the GPU workspace budget, with one final guard region.
   Ragged or unsupported shapes retain individual preparation.
5. **Routing buffers.** Full-layer expert-row execution reuses host IDs,
   counts, prefix offsets, cursors and entries. Stable counting preserves
   ascending expert order and original token/slot order within each expert.
   The three expert arrays and final prefix element add
   `(3 * expert_count + 1) * sizeof(int32_t)` to the admitted host bound.
   Expert submission and the host routing synchronization remain; this is
   not GPU-only routing or CUDA graph capture.

#### Upload experiment

The same six-shard Huihui Q4_K_XL model ran on the RTX 3080 reporting
20 GiB, Xeon W-2133 and 247 GiB RAM. Input was the first 4096 saved IDs,
batch 2048, GPU cap 14336 MiB, host cap 32768 MiB, reserve 1024 MiB,
DSA/Flash on and tail 0. Ring sizes ran in order 1/8/16/16/8/1 MiB.

| Ring per worker | Upload calls | Prefill seconds, first / reverse |
|---|---:|---:|
| 1 MiB | 350648 | 35.595 / 37.107 |
| 8 MiB | 44840 | 36.144 / 36.263 |
| 16 MiB | 23022 | 35.475 / 35.427 |

All six complete logits files matched. Batch, 41 groups and 40 prefetched
layers stayed the same. A larger ring greatly reduced call count, but
these wall times do not establish a stable speed gain. The normal target
therefore stays at 1 MiB rather than spending more pinned memory by default.

#### Matched full-model measurements

The preserved `c57840c0` executable and the new build ran the same saved
IDs on the same hardware and with the same environment:
`STRATA_MMQ_KQ=1 CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0
OMP_WAIT_POLICY=PASSIVE`. GPU cap was 14336 MiB, host cap 32768 MiB and
reserve 1024 MiB. Both used normal automatic compute selection,
`--prefill-full-layer --dsa --flash-attn --dsa-tail 0`, six CPU expert
threads, MTP off and context capacity equal to input length plus 128.

The 32k sequence was baseline/new/new/baseline. Both builds used batch
2048, 41 groups, 40 prefetched layers and a fused final head.

| 32768 input tokens | Baseline | New |
|---|---:|---:|
| Prefill seconds | 77.545 / 78.452 | 77.047 / 77.308 |
| Prefill tokens/s | 422.57 / 417.68 | 425.30 / 423.86 |
| Tracked GPU peak | 13687.44 MiB | 13417.71 MiB |
| MMQ activation quantizations | 264949 | 164846 |
| Individual MMQ products | 264949 | 264949 |

Pooled 32k throughput improved by 1.06%; the wall-time gain is small.
Tracked GPU peak fell by 269.73 MiB. The baseline quantization count
follows its one-quantization-per-MMQ-product implementation; the new
counter records actual preparation launches, not summed GPU duration.
Small-column MMVQ preparation is not included in this counter.

One subsequent baseline/new pair evaluated exactly 200000 input tokens
and generated exactly 128 IDs. It used `--max-context 200128 --n 128`,
`--glm-gpu-experts 4096 --glm-gpu-reserve 1024 --glm-gpu-warm`, and the
same budgets and compute settings as above.

| 200000 input / 128 output | Baseline | New |
|---|---:|---:|
| Prefill | 461.764 s | 411.006 s |
| Prefill tokens/s | 433.121 | 486.611 |
| Decode setup | 2.195 s | 2.192 s |
| Generation | 27.876 s, 4.592 tokens/s | 27.918 s, 4.585 tokens/s |
| Whole process (`time -v`) | 8 min 13.07 s | 7 min 22.36 s |
| Tracked prefill GPU peak | 13911.99 MiB | 13621.85 MiB |
| MMQ activation quantizations | 1588127 | 986558 |
| Individual MMQ products | 1588127 | 1588127 |

This pair measured 12.35% higher prefill throughput, with 37.88% fewer
MMQ activation quantizations and 290.14 MiB less tracked GPU memory.
The larger wall-time difference is a single sequential pair, not a
general 12% speedup claim. The repeated 32k comparison above showed
only a small gain. GPU clocks and other users were not isolated.

Every full-model logits file was byte-identical to the saved same-length
reference. Both 200k runs also matched all 128 original output IDs.
MMQ, FP16 and F32 product counts, selected batch, group count, weight
payload, activation boundary traffic and decode-cache counters matched.
All six runs reported zero filesystem input blocks, major faults and
swaps. These are repeated-text performance checks, not quality tests
or a fresh comparison against llama.cpp or Maya.

#### Validation

- The CUDA SM86 build passed.
- `ctest --test-dir BUILD -R '^glm5_' --output-on-failure` passed all
  13 GLM tests.
- `ctest --test-dir BUILD -R '^glm5_compute$' --repeat until-fail:10
  --output-on-failure` passed all ten repetitions. Scratch checks exposed
  a fixture upload-order issue: default-stream uploads are now completed
  before nonblocking KDA execution. The KDA kernels were not changed.
- Native preparation tests compare exact product bytes for Q8_0 and
  Q4_K, aligned and ragged columns, 128/256/512-wide inputs where the
  format supports them, changed contents at reused input addresses,
  mismatched formats and rejected nested scopes.
- Upload tests check actual tensor payload bytes, reduced call counts,
  adaptive host/transfer bounds and retained prefetch under a tight cap.
- Compact scratch sizing equals its carved size. Full-layer tests check
  exact routing logits, map/metadata budgets, state handoff and MTP.
- Compute Sanitizer memcheck with full leak checking reported zero errors
  and zero leaked bytes for both `glm5_gemm_test` and
  `glm5_full_layer_test`.
- Strict C++20 syntax/warning checks passed for all 12 changed C++ sources.
  The four affected compute/executor sources also passed syntax checks
  with `STRATA_PREFILL_MMQ` undefined. This is not a fallback runtime test.

HIP, SYCL and other CUDA architectures were not built or hardware-tested.
No new whole-model CUDA trace was captured for this follow-up.

### Phase probes, transfer and CPU checks

`--probe` is a GLM-only diagnostic, not a speed option. For example, add it
to the existing full-layer command:

```text
--prefill-full-layer --dsa --flash-attn --dsa-tail 0
--prefill-gpu-mib 14336 --prefill-host-mib 32768 --prefill-reserve-mib 1024
--cpu-threads 6 --glm-gpu-experts 4096 --glm-gpu-reserve 1024
--glm-gpu-warm --mtp 0 --probe
```

The normal path creates no probe events and takes no probe timestamps.
An enabled probe reuses at most 256 event pairs. It collects completed
events at existing synchronization points; if the pool is exhausted it
waits and increments `forced_waits`. The real runs below used 13-20 pairs
and reported zero forced waits.

The CLI separates resident prefill from subsequent generation as well as
supporting independent full-layer prefill. The model's decode probe API
returns cumulative counters since decoder creation/reset; the CLI takes
an interval difference. Event-pair count is a high-water mark, not a
subtractable operation count.

#### Read the counters carefully

- `host_ms` is wall time inside the host operation. For CPU experts this
  includes activation preparation, scratch allocation, OpenMP scheduling
  and dot products. It is not just arithmetic time.
- `process_cpu_ms` is process CPU time during CPU expert work. It includes
  worker/runtime threads and can exceed wall time. It is not a hardware
  stall counter. It uses `std::clock`, validated here on Linux; clock
  semantics on other operating systems have not been validated.
- `stream_ms` is elapsed time between CUDA events. A span can include
  host launch starvation; it is not summed kernel-busy time.
- GPU phases also report host API/submission time. This matters for
  pageable `cudaMemcpyAsync`: the call can block while earlier GPU work
  completes, before an explicit stream synchronization is reached.
- KDA/MLA/FFN/router spans are children of `layer_pre`. Upload/download
  streams can overlap compute. Do not add all phase times or treat them
  as disjoint wall-time percentages.
- `bytes` records selected logical payloads: weight-bank uploads,
  activation boundaries, route readback, CPU result uploads, cache
  admission, and decode embedding/logit transfers. CPU-expert bytes are
  logical weight reads, not measured DRAM traffic. State handoff is
  timed on the host; its payload and scratch D2D copies are not included
  in these byte counters. These are not PCIe bus-transaction counters.
- `cpu_result_copies` counts actual CPU-result H2D API calls, not parent
  phase events. With a GPU cache this is currently one call per CPU-miss
  row; the all-CPU path uploads its complete result array in one call.
  The counter supports interval subtraction and resets, and stays zero
  with probes disabled.
- Fine-grained mixer spans cover normal trunk blocks. MTP adds draft and
  verification work that is not fully partitioned into these categories.
  The measurements below use MTP off.

#### Hardware bandwidth and ISA

Checks ran on the same RTX 3080 reporting 20 GiB, Xeon W-2133
(6 cores/12 threads) and 247 GiB RAM. The current link reported PCIe
3.0 x16. A separate bounded calibration used a 1 GiB integer array,
four checksum-read passes, and five 256 MiB pinned CUDA copies in each
direction. Read checksums and the PCIe round-trip payload matched.
Rates use decimal GB/s.

| Calibration | Measured rate |
|---|---:|
| Streaming host reads, 1 thread | 12.711 GB/s |
| Streaming host reads, 3 threads | 35.245 GB/s |
| Streaming host reads, 6 threads | 38.678 GB/s |
| Pinned H2D | 7.907 GB/s |
| Pinned D2H | 8.395 GB/s |

These are single-session calibration results, not guaranteed peak rates
or the same benchmark used by another project's user report.

CPU detection reported AVX-512F/BW, AVX2 and FMA, but no AVX-512 VNNI or
BF16. ggml reported `ggml_avx512=1`, `ggml_avx2=1`,
`ggml_avx512_vnni=0`, `ggml_avx512_bf16=0` and `ggml_fma=1`.
The actual CPU quantization compile commands use `-march=native`;
the cached `GGML_AVX*=OFF` options do not mean this native build lacks SIMD.

Disassembly of the actual Q4_K/Q5_K/Q6_K dot functions showed YMM
operations, no ZMM operations and no VNNI instructions. The pinned
ggml Q4_K implementation explicitly selects 256-bit AVX2 intrinsics;
its VNNI helper requires a feature this CPU does not have. An AVX-512
build capability is therefore not proof that every hot dot uses 512-bit
vectors. A dedicated wider/multi-row kernel would require a separate
numerical and speed comparison. No shared CPU kernels or ISA defaults
were changed for these probes.

#### CPU threads and cache capacity

The initial sweep used the same saved 8192-ID prefix and 64 greedy
outputs, context capacity 8320, the budgets above, cache warmup and
`STRATA_MMQ_KQ=1 CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0
OMP_WAIT_POLICY=PASSIVE`. All profiled runs preserved the complete
prefill logits and all output IDs, including the larger-cache sample.

| CPU threads / cache choice | Actual cache | Decode | CPU-expert host time |
|---|---:|---:|---:|
| 1 / 4096 MiB cap | 3685.1 MiB, 252 slots | 2.216 tokens/s | 25.693 s |
| 3 / 4096 MiB cap | 3685.1 MiB, 252 slots | 3.529 tokens/s | 14.546 s |
| 6 / 4096 MiB cap | 3685.1 MiB, 252 slots | 3.840 tokens/s | 13.075 s |
| 6 / auto, same 1024 MiB reserve | 7984.4 MiB, 546 slots | 4.332 tokens/s | 10.936 s |

The 4 GiB-cap samples had 7000 hits and 14504 misses, a 32.55% hit rate.
The auto-cache sample had 10758 hits and 10746 misses, a 50.03% hit rate.
It reduced logical CPU weight reads from 207.43 to 153.81 GiB over the
64 outputs. The six-thread CPU-expert phase used 48.651 process-CPU
seconds over 13.075 wall seconds, about 3.72 active process cores on
average in that interval.

This confirms CPU-expert work and cache capacity matter on this workload.
It does not prove that the remaining CPU time is all DRAM stalls:
unpacking, dot arithmetic, task imbalance, allocation and scheduling
are included. Three-to-six-thread scaling was small; the streaming-read
calibration also flattened in that range. Row-split scheduling and ISA
changes still need their own A/B tests.

#### Final host/stream breakdown

A follow-up recorded host API time alongside stream time. At 8192
input/64 output, prefill took 40.064 s and decode 17.248 s:

| Stage | Measurement |
|---|---:|
| Exposed weight-loader host wait | 19.553 s |
| Activation H2D | 20.50 GiB, 3.083 stream seconds |
| Activation D2H | 20.00 GiB, 2.813 stream seconds |
| CPU experts during decode | 13.504 host seconds |
| Decode route readback | 1.603 host seconds, 0.070 stream seconds |

Short-prompt prefill spent about half its wall time waiting for weight
loading. Decode spent about 78% in the CPU-expert phase. The large
host/stream difference for route readback is not slow copying of a
large tensor: the host API also waits for GPU dependencies.

The same 200000-ID input then generated 32 tokens, with context 200128,
six CPU threads and the 4 GiB expert-cache cap. Prefill took 483.322 s;
generation took 8.508 s. Complete logits matched the original 200k dump,
and all 32 IDs matched the original output prefix.

| Long-context prefill stage | Measurement |
|---|---:|
| Weight uploads | 181.09 GiB; exposed loader wait 1.751 s |
| Activation H2D | 500.49 GiB, 74.329 stream seconds |
| Activation D2H | 488.28 GiB, 64.566 stream seconds |
| Layer-pre parent | 261.393 stream seconds |
| KDA mixer child | 70.610 stream seconds |
| MLA mixer child | 156.778 stream seconds |
| Routed GPU experts | 133.164 stream seconds, 63.117 host submission seconds |
| Layer post | 7.019 stream seconds |

The long prompt largely hid weight loading. Activation traffic is still
substantial, but its stream times overlap compute and cannot be summed
into an exposed-transfer penalty. MLA and routed experts are the largest
measured compute-stream regions; the expert region can also contain
host submission gaps.

| Long-context decode stage, 32 outputs | Measurement |
|---|---:|
| CPU experts, including scheduling/preparation | 6.536 host seconds, 76.8% of generation wall time |
| Layer-pre parent | 1.404 stream seconds |
| Cached GPU experts | 0.303 stream seconds |
| Route D2H | 0.0205 GiB, 0.919 host seconds, 0.035 stream seconds |
| CPU-result H2D | 0.1096 GiB, 0.056 stream seconds |
| Cache admission | 0.7615 GiB, 0.140 host seconds, 0.133 stream seconds |

CPU experts logically read 102.74 GiB and evaluated 7181 misses over
these 32 outputs. The measured critical cost is the CPU-expert path,
not the small route/result payloads or cache-admission transfer time.
Route host time includes GPU dependency waits; the 0.007 s spent in
explicit decode synchronizations alone must not be reported as all
synchronization overhead.

#### Validation and measurement limits

The final CUDA build and all 13 GLM CTests passed. Probe tests compare
exact full-layer and resident logits/IDs, MTP off/on, transfer counts,
bounded event use, resets, interval subtraction, and operation-error
unwinding. Strict changed-source warning checks passed. Memcheck of
the probe-enabled full-layer suite reported zero errors and zero leaked bytes.

Unprofiled 8k controls measured 227.53 then 205.71 prefill tokens/s,
and 3.902 then 3.794 decode tokens/s. This drift is why these sequential
samples are not fine-grained speedup claims. A matching unprofiled
200k/32 control took 485.760 s for prefill and 8.477 s for decode,
close to the diagnostic run above. The probe is still opt-in and its
overhead is not guaranteed by this one comparison.

These are repeated-text diagnostics, not the earlier 128-output timing
window or a matched Maya benchmark. CPU clocks and other users were
not isolated. No PMU DRAM/stall counters, HIP build, other GPU backend,
or dedicated new AVX-512 kernel was tested.

### CPU expert row-pool follow-up

The GLM-private CPU executor now has a persistent worker pool. The caller
is one of its workers. Gate/up and down are separate row-tiled phases,
with complete hidden activation quantization between them. The original
ggml dot functions, SwiGLU clamp and per-row accumulation remain unchanged.
Only independent output rows are scheduled differently.

Scratch holds at most 32 entries at a time; input activation buffers grow
only up to the supported 4096-token batch bound and are reused. Gate/up
share a preparation only when their `vec_dot_type` matches. Every call
invalidates input-ready flags, so a reused input address cannot reuse old
activation contents. Workers finish a phase before the next epoch resets
the task counter. Shutdown joins all workers.

Matched tests retained the same model, saved IDs, GPU compute choices,
4096 MiB expert-cache cap, warmup, DSA/Flash/tail 0, MTP off and probe
settings. Environment and hardware were the same as the phase checks.
The private `CpuExpertMode` isolated CPU selection without disabling
the preceding GPU optimizations.

| 8192 input / 64 output | Original CPU path | Row pool |
|---|---:|---:|
| 6 threads, decode tokens/s | 4.488 / 4.483 | 6.466 / 6.576 |
| 6 threads, CPU-expert seconds | 10.974 / 10.986 | 6.658 / 6.507 |
| 3 threads, decode tokens/s | 4.140 / 4.119 | 5.138 / 5.187 |

Each thread-count sequence was reference/rows/rows/reference. Pooled
throughput rose 45.37% with six threads and 24.99% with three.
All eight complete logits files and 64 output IDs were byte-identical.
Cache hits, misses and logical CPU weight reads matched. A normal CLI
automatic-selection smoke selected rows and measured 6.572 decode
tokens/s, with the same complete logits and IDs.

One subsequent 200000-input/128-output reference/rows pair used six
threads, context 200128 and the same cache and prefill budgets:

| Measurement | Original CPU path | Row pool |
|---|---:|---:|
| Prefill | 405.471 s | 408.677 s |
| Decode setup | 2.179 s | 2.190 s |
| Generation | 28.117 s, 4.552 tokens/s | 19.598 s, 6.531 tokens/s |
| CPU-expert host time | 21.333 s | 12.903 s |
| CPU logical weight reads | 384.90 GiB | 384.90 GiB |
| Cache hits / misses | 16111 / 26897 | 16111 / 26897 |

This pair measured 43.47% higher decode throughput and 39.52% less
CPU-expert host time. All 154880 prefill logits and all 128 output IDs
matched the original saved 200k result byte-for-byte. The pool is
created for decode, not independent full-layer prefill; the prefill
timing difference is not a claimed pool effect. The long pair was not
repeated ABBA. CPU/GPU clocks and other users were not isolated.

One more normal-CLI 200k/128 sample used the selected row pool and the
existing `--glm-gpu-experts 0`, retaining the same 1024 MiB reserve.
It allocated 6141.9 MiB/420 slots instead of 3685.1 MiB/252 slots.
Cache hits rose from 16111 to 22272, misses fell from 26897 to 20736,
and CPU logical weight reads fell from 384.90 to 297.21 GiB.
CPU-expert host time was 10.478 s; generation took 17.057 s,
7.504 tokens/s. This separate cache-size sample measured 14.89% higher
decode throughput than the fixed-cache row-pool sample. Setup increased
from 2.190 to 2.508 s. More cache costs VRAM and warmup time.

Complete prefill logits and all 128 output IDs matched the original
saved results in that sample too. Different cache sizes change CPU/GPU
placement, so this is not proof of universal numerical equivalence
between cache configurations. No default cache value or explicit cap
was changed. This was a single cache comparison, not another ABBA.

Mixed-format CPU fixtures cover Q8_0, Q4_K, Q5_K and Q6_K, different
activation formats, 1/2/3/6 threads, changing token counts and input
contents, batches crossing the 32-entry scratch bound, clamp stress,
output guards and invalid duplicate rows. Model comparisons cover MTP
acceptance/rejection, GPU cache diagnostics, asynchronous staging,
continuation, reset and append across decoder teardown/recreation.
The CPU pool translation
unit passed ThreadSanitizer and AddressSanitizer/UndefinedBehaviorSanitizer
checks; the reference libraries were not rebuilt under those sanitizers.
No shared CPU kernels or GPU matrix products were modified.

### Direct single-token GPU expert dispatch

This follow-up removes repeated preparation and D2D copies, not matrix
products. With one token and one route per participating expert, the
cache executor prepares its existing native Q8_1 image once, then runs
the original single-column gate/up/down products for each expert.
The down product writes directly to the expert's original output slot.
Every dispatch prepares its current input; no pointer-keyed cache
survives the call.

Multi-token and duplicate-expert routes keep the preceding implementation.
An all-miss route prepares nothing on the GPU. Upload readiness and
last-read events, cache policy, CPU miss order and expert reduction order
are unchanged. Existing buffers are reused without new workspace.
Normal SM86 selection enables the path; global `--reference` disables
it, and private `GpuExpertMode` permits isolated tests.

Tests used the same six-shard Q4_K_XL model, RTX 3080 reporting 20 GiB,
W-2133/247 GiB RAM, six-thread CPU row pool, 4096 MiB expert-cache cap,
warmup, MTP off, DSA/Flash/tail 0 and probes. Prefill budgets were
14336 MiB GPU, 32768 MiB host and 1024 MiB reserve, with context
capacity input length plus 128. Environment remained
`STRATA_MMQ_KQ=1 CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0
OMP_WAIT_POLICY=PASSIVE`.

| 8192 input / 64 output, control/single/single/control | Control | Direct |
|---|---:|---:|
| Decode tokens/s | 5.871 / 6.104 | 5.932 / 6.164 |
| GPU expert stream interval | 310.316 / 303.900 ms | 265.486 / 262.458 ms |
| Expert input quantizations per run | 7000 | 2245 |

Pooled GPU expert intervals fell 14.05%; whole-decode throughput
improved only 1.02%. The CPU-expert phase varied between runs too.
The first prefill took 97.218 s and later ones about 35-36 s; its cause
was not measured. No prefill speedup is attributed to this decode change.

| 200000 input / 128 output, one control/direct pair | Control | Direct |
|---|---:|---:|
| Prefill | 411.120 s | 414.444 s |
| Generation | 21.612 s, 5.923 tokens/s | 21.379 s, 5.987 tokens/s |
| GPU expert stream interval | 719.652 ms | 621.890 ms |
| CPU-expert host time | 14.445 s | 14.354 s |
| Expert input quantizations | 16111 | 4749 |
| Cache hits / misses | 16111 / 26897 | 16111 / 26897 |
| CPU logical reads | 384.90 GiB | 384.90 GiB |

The direct long run omitted 11362 input quantizations and 32222 D2D
copies carrying 503.47 MiB. This is GPU D2D payload, not PCIe traffic.
Its GPU expert interval fell 13.58%, but decode throughput improved
only 1.09%. Event spans include launch starvation. Clocks and other
users were not isolated; the small overall gain is not a general
speedup guarantee. Absolute rates from earlier CPU-pool sessions are
not matched controls for these later runs.

All six complete logits files and generated ID sequences matched the
saved same-length originals byte-for-byte. The normal CLI also selected
`gpu_single=1` and produced the same 8k logits and 64 IDs. That
selection/equality smoke measured 5.793 decode tokens/s and is not
another throughput A/B.

Mixed-format tests cover Q8_0/Q4_K/Q5_K/Q6_K, direct products, sparse
slots and guards, input-address reuse with changed contents, clamp stress,
all-miss routes, duplicate-route/multi-token fallback, staged uploads,
slot replacement, counters and resets. Cached MTP tests cover single
steps and multi-token verification fallback. All 13 GLM tests passed.
Memcheck of the expert-row and runtime tests reported zero errors and
zero leaked bytes. Strict changed-source checks and no-MMQ fallback
syntax checks passed. HIP and other CUDA architectures were not built
or hardware-tested; their automatic dispatch was not changed.

### Cache-control validation and existing heat quotas

The per-layer GPU expert dispatch and admission paths previously checked
the entire cache's inverse mappings after each operation. They now call
`consistent(layer)`, which validates all owners and resident entries in
the layer they changed. Full `consistent()` remains available and checks
every layer. Invalid layer numbers are rejected. This does not disable
checks or change cache placement, quotas, budgets, or expert arithmetic.

A host microbenchmark used 45 layers, 288 experts and six slots in each
of 42 MoE layers. For 200000 calls in full/local/local/full order, the
retained full validator took 1575.90/1586.45 ms and local validation
35.12/35.19 ms. The roughly 45x ratio applies to validation alone.
Earlier versus updated 8k equal-quota samples differed by 1.84% in
pooled decode throughput, but CPU-expert timings also changed; no
isolated whole-decode speedup is attributed to that difference.
Outputs and all placement/traffic counters matched across the check change.

Separate comparisons evaluated the existing opt-in
`--glm-gpu-warm --glm-gpu-heat-quotas`, not a new allocation algorithm.
The same six-shard Q4_K_XL model ran on the RTX 3080 reporting 20 GiB,
W-2133/247 GiB RAM, with the six-thread CPU row pool, direct GPU single
experts, MTP off, DSA/Flash/tail 0 and probes. Prefill budgets remained
14336/32768/1024 MiB GPU/host/reserve, decode reserve 1024 MiB,
and context capacity input length plus 128. Environment remained
`STRATA_MMQ_KQ=1 CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0
OMP_WAIT_POLICY=PASSIVE`.

Each input was evaluated once by full-layer prefill; all comparisons
restored the exact same host prefill checkpoint into a new decoder and
warmed an independent cache. Equal/heat/heat/equal ran with a 4096 MiB
cap, followed by two heat-quota samples capped at the equal allocation
rounded up to a whole MiB. The latter separates placement from filling
otherwise unused memory.

| 200000 input / 128 output | Equal | Heat, same cap | Heat, similar actual bytes |
|---|---:|---:|---:|
| Actual cache | 3685.13 MiB / 252 slots | 4089.94 MiB / 281 | 3682.88 MiB / 253 |
| Hit rate | 37.46% | 44.24% | 41.39% |
| CPU logical reads | 384.90 GiB | 344.03 GiB | 361.42 GiB |
| Decode tokens/s | 6.253 / 5.974 | 6.636 / 6.659 | 6.255 / 6.311 |
| Pooled tokens/s | 6.111 | 6.647 | 6.283 |

Heat quotas used more of the same cap than equal per-layer slot counts,
which left about 411 MiB unused. The same-cap comparison measured 8.78%
higher throughput; with similar actual allocation, it was 2.82%.
The similar-allocation samples followed ABBA rather than being another
interleaved ABBA. Clock/environment drift is not excluded.

At 8192 input/128 output, pooled throughput was 5.956 equal, 6.096 heat
under the same cap, and 6.012 heat with similar actual allocation.
The heat allocations were 4090.19/3683.13 MiB. Those measured changes
were 2.34%/0.94%. This is evidence for an existing optional configuration,
not a general new-algorithm speedup. More cache costs VRAM and warmup.
Explicit caps and default quota selection were not changed.

All six 200k results matched original complete prefill logits and all
128 generated IDs. The 8k variants also matched complete logits and
all 128 IDs. This sample equivalence does not prove universal numerical
equivalence across different CPU/GPU cache placements.

Cache tests cover 4096 cross-layer replacements with both full and local
validation, resets, dense layers with no slots and invalid indices.
All 13 GLM tests passed. The cache-policy translation unit passed
AddressSanitizer/UndefinedBehaviorSanitizer checks; remaining format
objects were not rebuilt under those sanitizers. Strict C++ and
no-MMQ GPU-cache syntax checks passed. No shared backend, kernel,
new workspace or new CLI flag was introduced.

### Rejected CPU row-scheduling experiments

Two exact-arithmetic scheduling experiments were measured and removed.
Neither changes the retained dynamic 64-row worker pool.

The static-contiguous candidate assigned a fixed interval of 64-row
tasks to each worker, avoiding per-tile atomic claims. At 8192 input and
128 output, six-thread dynamic/static decode rates were
6.317/6.284 versus 4.996/5.033 tokens/s in baseline/candidate/candidate/baseline
order: pooled throughput fell 20.41%. With three threads it fell 2.00%
(4.912/4.944 versus 4.824/4.834). Six-thread CPU expert time increased
from about 13.85/13.94 to 19.10/18.89 seconds.

A separate candidate retained dynamic atomic claims but doubled the
tile from 64 to 128 rows. Six-thread baseline/candidate rates were
6.265/6.273 versus 6.203/6.195 tokens/s; three-thread rates were
4.945/4.921 versus 4.908/4.897. Pooled throughput changed
-1.12%/-0.61%. Those small differences do not establish a general
regression, but provide no reason to switch the default.

Both experiments used the same saved 8k prefix, context 8320, six-shard
Q4_K_XL, RTX 3080 reporting 20 GiB, W-2133/247 GiB RAM, DSA/Flash/tail 0,
MTP off, probes and GPU single-token direct execution. Cache stayed
at a 4096 MiB cap/3685.13 MiB actual allocation with equal quotas.
Prefill budgets and environment variables were unchanged from the
preceding comparisons. Each thread-count sequence reused one identical
host prefill checkpoint in four newly warmed decoders.

All sixteen output sequences and complete prompt logits matched their
references; cache and logical CPU/transfer counters matched. Static
assignment also passed mixed-format row tests, MTP and CPU-translation-unit
TSAN/ASAN/UBSAN checks. Its rejection is a performance result, not a
numerical failure. No PMU or affinity study established the cause of the
static slowdown. Both experimental production modes were removed.

### Rejected CPU-result upload coalescing

After the scheduling tests, another experiment merged adjacent CPU-miss
output rows into one H2D copy. Gaps containing GPU results were never
copied. Payload bytes, expert arithmetic and cache routing were unchanged;
no extra packing allocation was introduced.

The same model/hardware, six-thread dynamic 64-row pool, direct GPU single
experts, DSA/Flash/tail 0, MTP off and 4096 MiB equal-quota cache were used.
The cap allocated 3685.13 MiB. Budgets/environment remained as above,
context was input length plus 128, and each sequence reused one identical
host prefill checkpoint for row/coalesced/coalesced/row decoders.

| Measurement | 8k row | 8k coalesced | 200k row | 200k coalesced |
|---|---:|---:|---:|---:|
| H2D result calls per 128 outputs | 28838 | 10305 | 26897 | 10249 |
| Result payload | 450.59 MiB | unchanged | 420.27 MiB | unchanged |
| Mean result-copy host API time | 230.84 ms | 215.28 ms | 225.05 ms | 211.86 ms |
| Mean result-copy stream interval | 184.26 ms | 201.52 ms | 179.29 ms | 194.06 ms |
| Pooled decode tokens/s | 6.265 | 6.309 | 6.278 | 6.271 |

Calls fell 64.27%/61.90%, but the overall throughput changes were only
+0.69%/-0.11%. Stream intervals grew while host submission time fell
slightly. Those spans are not pure DMA-busy times. Copy-call reduction
did not establish an end-to-end benefit on these workloads.

All output IDs, complete prompt logits, cache counters and payload bytes
matched. Sparse-run/guard tests, MTP/reset checks and memcheck of the
experimental upload path passed. Coalescing and its private execution
option were removed rather than enabled by default.

Only the diagnostic `cpu_result_copies` counter was retained. Tests verify
real miss-call/payload counts, disabled probes, interval differences,
MTP, continuation and reset. The final execution paths remain the dynamic
64-row pool and original result uploads; all 13 GLM tests passed.
This iteration establishes no new stable speedup.

### Fine MLA probes and immutable MMQ bounds

`--probe` now separates query preparation/projections, absorption, KV
projection/storage, indexer projection/pooling/scoring/selection,
attention, deabsorption, output projection and layout conversions.
These are children of `mla_mixer`, not extra disjoint wall-time regions.
Native preparation remains included in the head-product regions.

On the current matched baseline, the MLA parent/attention/index-score
intervals were 14.530/11.028/0.991 s at 32768 input, and
132.472/77.817/36.913 s at 200000 input. Attention dominated short
prefixes; index scoring became a substantial additional long-context
cost. The fine probes kept complete real-model logits unchanged.

Two exact-output GPU candidates were tried and removed:

- Preparing high/low Flash queries into caller scratch reduced the
  launch's shared storage from 71120 to 37824 bytes, but raised compiled
  registers from 211 to 250, with no stack/local spills. Repeated
  64-query/64-head/50000-cache/2048-selected microbenchmarks remained
  about 2 ms including preparation. Exact dense/sparse output, masks,
  extreme scales, immutable inputs and scratch guards passed. It was
  not integrated into model execution.
- A two-query F32 indexer tile preserved dimension/head accumulation
  order and exact complete scores, but measured about 2.65-2.69 ms
  versus 1.91-1.95 ms for four queries at 64 queries/50000 pools.
  Compiled registers/shared bytes fell from 93/32768 to 53/17408;
  logical block count and pooled-key tile reloads doubled. Sparse/MTP
  state and append tests passed. Four-query execution was retained.

Flash SASS contains `HMMA.16816.F32`. Nsight Compute
LaunchStats/Occupancy/SpeedOfLight collection was attempted but denied
with `ERR_NVGPUCTRPERM`; actual achieved occupancy and Tensor Core
utilization were not measured. No system permissions were changed.
Smaller resource footprints were not treated as speedup evidence.

The retained simplification reuses the MMQ backend's integer
`{0, columns}` bounds while the column count remains unchanged. Only
that backend writes the array, on its one compute stream. Shape changes
rewrite it; FP16/F32 fallback does not modify it. No float-input cache,
extra tensor allocation, transfer or CLI flag was added. It follows
the preceding validated SM86 modern-compute policy; reference/fallback
execution keeps its previous per-product initialization.

Tests retained the same model/hardware, DSA/Flash/tail 0, batch 2048,
14336/32768/1024 MiB prefill GPU/host/reserve budgets, six CPU threads,
4096 MiB decode-cache cap, warmup and MTP off. Environment remained
`STRATA_MMQ_KQ=1 CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0
OMP_WAIT_POLICY=PASSIVE`. Context capacity was input length plus 128.
Both executables used the same fine probes. The preserved baseline
preceded only the bounds reuse change.

| Measurement | Baseline | Bounds reuse |
|---|---:|---:|
| 32k prefill tokens/s, baseline/new/new/baseline | 508.780 / 497.504 | 504.751 / 502.353 |
| 32k bound initializations | 264949 | 77280 |
| 200k prefill, one pair | 415.676 s / 481.144 tokens/s | 413.923 s / 483.182 tokens/s |
| 200k bound initializations | 1588127 | 461128 |

New `bounds_launches`/`reused_bounds` counters sum to the unchanged MMQ
product count. Baseline initialization counts follow its one-launch-per-
product implementation, not a separate profiler census. The long run
omitted 1126999 initialization calls, about 70.96% of that kernel
category, not 70.96% of all kernels or model work.

Pooled 32k throughput changed +0.09%; the single 200k pair changed
+0.42%. There is no substantial stable end-to-end speedup claim.
The simplification removes deterministic no-op metadata writes without
changing floating-point computation, buffers or scheduling policy.

All six complete logits files matched the saved originals byte-for-byte;
both long runs matched all 128 original generated IDs. Products,
quantization counts, group plans, memory peaks, weight/activation traffic
and decode-cache counters matched. Integer-bound tests cover column
changes, ragged shapes, varied widths and FP16/F32 fallback. Existing
full-layer and sparse tests cover fine probe counters and exact
probe-on/off results. CUDA SM86, strict C++/no-MMQ syntax checks and
focused GEMM/full-layer memcheck checks passed with zero errors and
zero leaked bytes. A final normal-CLI 32k run with probes disabled
preserved complete original logits and the same bounds counts.
Other GPU backends were not built or
hardware-tested. The backend is not currently used inside CUDA Graphs;
graph replay would need a separate lifetime/invalidation design.

### Direct head-major MLA Flash layout, 6 November 2026

Flash can now read the head-major absorb output and write directly into
the head-major deabsorb input. The original MMA, online softmax and
compensated accumulation are unchanged. Only addressing differs.
The query split and final output merge transposes remain.
Sparse windows use the complete batch's head stride and their own query
offset; empty windows clear the corresponding rows across heads.

Automatic selection is limited to CUDA SM86 with explicitly enabled Flash,
KV rank 512 and head width 256. Reference and unvalidated devices retain
token-major addressing. Flash-off paths retain their required scratch.
There is no new regular CLI switch; `mla_layout=head|token` reports the
resolved choice. Private test policies retain an isolated layout A/B.

On the same RTX 3080 / W-2133, Q4_K_XL model, budgets and saved inputs as
the preceding measurements, with six CPU threads and MTP off:

| Measurement | Token-major | Direct head-major |
|---|---:|---:|
| 32768 input prefill tokens/s, token/head/head/token | 515.805 / 505.684 | 513.588 / 510.040 |
| 32768 tracked GPU peak, MiB | 13417.713 | 12905.713 |
| 200000 input prefill, one pair | 413.668 s / 483.479 tokens/s | 412.152 s / 485.258 tokens/s |
| 200000 tracked GPU peak, MiB | 13621.853 | 13109.853 |
| 200000 layout calls | 4312 | 2156 |
| 200000 layout stream interval | 2612.257 ms | 857.689 ms |
| 200000 attention stream interval | 77848.594 ms | 78687.641 ms |

The verified benefit is **512 MiB less prefill scratch** at batch 2048:
two omitted FP32 images of `2048 * 64 * 512` elements.
Pooled 32k throughput changed about +0.22%; the single long pair changed
about +0.37%. These are not substantial stable speedups. Attention's
interval increased slightly, offsetting much of the transpose saving.
Logical removed transpose reads/writes total 176 GiB at 32k and
1074.219 GiB at 200k; these are calculated payloads, not measured DRAM
transactions or PCIe savings. Host activation transfers are unchanged.

Batch 2048, 41 groups, at most four layers per group, matrix/quantization
counts, weights, host tensors and cache hits/misses were unchanged.
All six complete logits files matched the saved originals byte-for-byte,
and both long runs matched all 128 original output IDs.
Low-level layout memcheck reported zero errors and zero leaked bytes.
GLM regression checks cover exact dense/sparse state, scratch sizing,
fallback policy, window offsets, guards, append/reset and MTP.
All 13 GLM CTests passed, followed by the expanded direct-layout
MTP-depth 1/3/7 acceptance/rejection regression. Sparse MTP-depth 3
memcheck also passed with zero errors and zero leaked bytes.
A normal-CLI 32k smoke with probes disabled automatically selected
head-major, reproduced the original logits byte-for-byte, and reported
12905.71 MiB tracked peak with unchanged batch, groups and matrix counts.
This smoke is not a new matched timing comparison.

This is not 512 MiB of extra decode expert cache: at batch one the two
images total only 256 KiB. A singleton identity-layout shortcut reuses the
original kernel; it was added after the model timing runs above, so no
decode performance improvement is claimed from those runs.
Other GPU backends were not built or hardware-tested.

### Half-tile MLA KV prefetch, 6 November 2026

The original Flash kernel loads 32 latent KV rows before computing QK,
softmax and PV. The new path uses three rotating 16-row shared-memory
buffers. After QK consumes the current cell IDs, it stages the next
tile's first half with `cp.async` while softmax/PV consume the current
KV tile. After PV releases the oldest half-buffer, it stages the second
half. The next iteration waits for copies and synchronizes the block
before reading the new tile.

Invalid, future and out-of-range cells are zero-filled without reading
invalid global addresses. F16 KV, query/probability decomposition, MMA,
compensated sums, tile size and arithmetic order are unchanged.
Token-major and head-major addressing both support the load mode.
Automatic selection is restricted to CUDA SM86, explicit Flash,
KV rank 512 and head width 256. Reference and unvalidated devices retain
direct loads. Private `FlashKvLoad` policies isolate tests; no ordinary
CLI switch was added. The compute log reports `kv_load=prefetch|direct`.

Shared memory is on-chip per-block storage, not the card's 20 GiB VRAM.
The original kernel uses 71120 B; complete KV double buffering would need
104400 B, exceeding SM86's 99 KiB per-block limit. Three half-buffers need
87760 B. Head-major registers increased from 211 to 234, token-major
from 215 to 248, with no stack/local spills. These compiler resources
do not measure achieved occupancy. There is no new global GPU/host/pinned
tensor allocation, and the preceding 512 MiB prefill saving is retained.

On the same RTX 3080 / W-2133 and Q4_K_XL model, with the same saved
inputs, batch 2048, budgets, six CPU threads and MTP off:

| Measurement | Direct loads | Half-tile prefetch |
|---|---:|---:|
| 32k prefill tokens/s, direct/prefetch/prefetch/direct | 508.700 / 503.687 | 536.289 / 535.008 |
| 32k attention stream interval, seconds | 11.241 / 11.695 | 7.878 / 7.924 |
| 200k prefill, one pair | 412.402 s / 484.963 tokens/s | 385.374 s / 518.976 tokens/s |
| 200k attention stream interval, seconds | 78.588 | 51.214 |

Pooled 32k throughput improved 5.82%; the long pair improved 7.01%.
Attention intervals decreased 31.11% and 34.83%, respectively. These
are event spans, not hardware-counter kernel-busy measurements.
The shared environment's clocks were not locked. The microbenchmark
with 64 heads, 64 queries, 50000 cached cells and 2048 selected cells
likewise changed about 2.17-2.23 ms to 1.487-1.489 ms.
No counter evidence isolates overlap from the changed copy instructions
and removal of register-mediated KV transfers.

All six complete logits files matched the saved originals byte-for-byte;
both long runs matched all 128 original output IDs. GPU/host tensor
peaks, matrix/quantization/bounds counts, plans, weight/activation bytes
and cache hits/misses were unchanged. Decode measured 6.028 versus
6.938 tokens/s in the long pair, but without a repeated decode-specific
comparison this is not a stable decode speedup claim.

All 13 GLM CTests and strict C++ syntax checks passed. Exact low-level
tests cover both layouts, window offsets, singleton batches, ragged
16/32-cell boundaries, all-masked queries, invalid/causal cells,
different head counts, scale stress, signed zeros, unchanged inputs
and output guards. Full-model fixtures compare direct/prefetched dense
and sparse state/logits, and exercise MTP depths 1/3/7, acceptance,
rejection, append/reset, staging and state handoff.
`--prefetch-check` passed memcheck with zero errors/leaks, racecheck
with zero errors/warnings, and synccheck with zero errors.
Other GPU backends were not built or hardware-tested.

### Rejected CPU Q4_K multirow and AVX-512 prototypes

On the W-2133, isolated prototypes compared two/four output rows with
AVX2, paired output rows in AVX-512, and a single-row AVX-512 integer
expansion retaining the original eight-lane floating-point accumulation.
The existing ggml Q4_K dot accepts only `nrc == 1`; the existing shared
K-quant batch kernels reuse weights across tokens, not activations across
output rows. Neither interface is a drop-in multi-output-row replacement.

The real Q4_K_XL expert matrices are Q4_K (82), Q5_K (41) and Q6_K (3).
Only Q4_K was prototyped, using real 4096-column/2048-row gate weights.
Tests used 1/16/64 experts (4.5/72/288 MiB), ordinary/mixed-scale/zero
activations, and 1/3/6 threads with dynamic 64-row work distribution.
All candidate outputs matched the original ggml dot byte-for-byte.
Fixed-loop unrolling was tested after inspecting generated array/register
traffic; it did not reverse the large-working-set result.

For the final 288 MiB, six-thread run, the two samples per mode were:

| Mode | Milliseconds |
|---|---:|
| Original ggml single-row | 7.091 / 7.437 |
| Two-row AVX2 | 9.940 / 9.765 |
| Four-row AVX2 | 9.783 / 9.506 |
| Paired AVX-512 | 9.352 / 8.976 |
| Single-row integer widening | 7.366 / 7.390 |

There was no reliable benefit supporting integration. Smaller working
sets showed some faster combinations and substantial reference timing
variation; they do not establish end-to-end decode gains.
ZMM instructions were confirmed, but no PMU/frequency measurements
identify downclocking, bandwidth or stalls as the cause.
The prototypes remain isolated session experiments. No production source,
dispatch, scratch or shared ggml implementation changed.
The original `glm5_runtime` regression passed. No prototype model-decode,
MTP or sanitizer run was performed, and no model-speed improvement is
claimed. Other formats, layouts and CPUs remain separate research.

### Prefill VRAM budget and incomplete group prefetch

The earlier 32k available-VRAM slowdown was reproduced with phase probes
on the same RTX 3080 / W-2133 and Q4_K_XL model. Batch 2048, compute
policies, host budget, 1024 MiB reserve, input IDs and MTP-off settings
were unchanged. Fixed/available/available/fixed budget runs were followed
by two available-budget runs with the existing
`--prefill-group-layers 1` control:

| 32k measurement | 14336 MiB, auto groups | Available VRAM, auto groups | Available VRAM, single-layer groups |
|---|---:|---:|---:|
| Prefill tokens/s | 544.962 / 532.271 | 467.073 / 463.949 | 532.324 / 531.973 |
| Exposed weight wait, seconds | 1.615 / 1.639 | 18.311 / 18.467 | 0.382 / 0.374 |
| Groups / prefetched layers | 41 / 40 | 21 / 20 | 45 / 44 |
| Activation H2D / D2H, MiB | 83968 / 81920 | 43008 / 40960 | 92160 / 90112 |

The capacity planner retains the configured one-layer prefetch before
expanding groups. Most intermediate MoE groups grow from one to two
layers at the larger budget, but only one member of each following
group is prefetched. At the group boundary,
[prefill.cpp](../src/glm5/prefill.cpp) waits for prefetched banks and
constructs uncovered banks synchronously.
[WeightBank](../src/glm5/weights.cpp) uses synchronous `cudaMemcpy`
without a staging ring for those banks. Twenty additional layers lose
prefetch coverage. The weight-wait probe includes bank construction,
allocation, conversion and readiness, not just DMA.
The reduced activation intervals do not compensate for this on 32k.
The same-large-budget single-layer control restores throughput, rather
than implicating VRAM capacity itself or a changed GPU arithmetic path.

A subsequent same-round 200000-input pair measured:

| Measurement | 14336 MiB, auto groups | Available VRAM, auto groups |
|---|---:|---:|
| Prefill | 384.232 s / 520.519 tokens/s | 370.467 s / 539.858 tokens/s |
| Exposed weight wait | 1.531 s | 17.663 s |
| Activation H2D / D2H, MiB | 512500 / 500000 | 262500 / 250000 |
| Tracked GPU peak, MiB | 13109.85 | 17460.97 |

The larger budget improved this long pair's throughput by 3.72%, despite
the same lost prefetch coverage: avoiding many more activation bytes
outweighed exposed loading. Overlapping event spans cannot be summed
to reconstruct the wall-time difference. Long results are one pair,
not a repeated speed guarantee.

All eight complete logits files matched their saved originals byte-for-
byte; matrix, quantization and bounds counts, weight bytes and batch
were unchanged within each input size. Probes forced no waits, and
`glm5_full_layer` passed. No generation was requested and no decode
cache was constructed. `cpu_experts=0`: CPU expert thread count does
not control the group-boundary uploads in this full-layer prefill.

For the measured 32k workload, `--prefill-gpu-mib 0
--prefill-group-layers 1` is an existing control, at the cost of more
activation traffic. It is not a recommendation to force single-layer
groups on long inputs. No production source, automatic group policy
or budget default changed. A future latency-aware planner would need
request-length and loading/activation cost evidence, not an unvalidated
fixed token threshold.

### Transfer-aware automatic grouping

The SM86 automatic full-layer planner now checks the prefetch opportunity
cost before adding another layer to a pipelined group. Complete-layer,
batch, GPU/host/transfer limits and the existing next-layer prefetch
priority remain unchanged.

When a prospective member is not already covered by the preceding group's
prefetch, but could be prefetched if it remained in a later group, the
planner compares its actual upload payload with
`new_input_tokens * hc_dim * sizeof(float)`. It grows the group only when
that payload is smaller than the activation H2D boundary removed.
Equal byte counts favor prefetch. D2H is not added as another guaranteed
critical-path saving because it can overlap on a separate copy engine.
Weights already covered by incoming prefetch do not incur the penalty;
neither do weights that could not be prefetched anyway.

`WeightBank::upload_bytes` uses the actual native or converted tensor
payload, excluding arena padding and quantization scratch. Append uses
the new suffix length, not accumulated KV context. There is no fixed
token threshold, GPU calibration workload, extra weight upload or new
regular CLI speed switch. This remains a greedy byte heuristic: it
does not estimate bandwidth, conversion time, launch overhead or how
much of a prefetched layer finishes before it is needed.

Explicit `--prefill-group-layers N`, `--prefill-no-pipeline`, reference
mode and unvalidated backends retain capacity-based grouping.
`group_policy=transfer|capacity` and `unprefetched_weights_mib` report the
resolved policy and trunk payload outside preceding-group prefetch.
The byte count includes initial trunk loading, but excludes separate
output-head/NextN banks; it is not a measured wait or DRAM counter.

Validation completed before the GPU became unavailable:

- CUDA SM86 build and all 13 GLM CTests passed.
- New fixtures cover short/long input choices, exact raw state/logits,
  append length, explicit grouping, no-prefetch and no-pipeline behavior.
- Reference/fallback policy and actual-versus-estimated upload bytes pass.
- Strict standard/no-MMQ C++ syntax checks pass.
- Full-layer memcheck reports zero errors and zero leaked bytes.

A separate CPU-only planning check used the real Q4_K_XL model and a
mirrored workspace-size calculation at explicit 14336/18432/19456 MiB
caps. At 32768 input tokens, the new policy chooses 43 groups/42
prefetched layers at all three caps, versus 41/40 at the smaller cap
and 21/20 at the larger caps for capacity-only planning.
At 200000 tokens the complete group/prefetch plans are unchanged:
41/40 at 14336 MiB and 21/20 at the larger caps. Batch stays 2048.
This is a planning check, not real GPU execution or timing.

The initial matched real-model timing attempt could not start: the driver
reported the CUDA device busy/unavailable, `nvidia-smi` displayed `ERR!`
and unavailable utilization, and the small CUDA check also skipped.
No GPU reset or system change was performed by the benchmark runner.
The GPU subsequently recovered, and the measurements below completed.

#### Real-model checks after GPU recovery

The same RTX 3080 / W-2133, Q4_K_XL weights, saved inputs, batch 2048,
six CPU threads, Flash/DSA/tail 0, MTP off and 1024 MiB reserves were
used. Both paths use the same executable and compute choices. The old
capacity policy is selected with the existing explicit group maximum
`--prefill-group-layers 4`; the new path leaves that maximum automatic.
Both use available VRAM (`--prefill-gpu-mib 0`) unless noted.

The first capacity run after recovery took 104.962 s with 53.807 s of
weight wait, versus 70.181 s / 17.838 s on its closing repeat. The cold
start was not isolated from filesystem and initialization effects.
To avoid crediting that difference to the planner, a second warm
capacity/new/new/capacity comparison was run:

| 32768 input measurement | Capacity policy | Transfer-aware policy |
|---|---:|---:|
| Prefill tokens/s | 469.917 / 468.856 | 536.944 / 537.029 |
| Pooled tokens/s | 469.386 | 536.986 |
| Exposed weight wait, seconds | 17.372 / 17.273 | 0.311 / 0.312 |
| Groups / prefetched layers | 21 / 20 | 43 / 42 |
| Unprefetched trunk payload, MiB | 97554.20 | 881.07 |
| Activation H2D / D2H, MiB | 43008 / 40960 | 88064 / 86016 |

Warm pooled prefill throughput improved **14.40%**. The first cold
sample is excluded from that percentage. Activation traffic increases,
but much less weight loading is exposed on this input. With an explicit
14336 MiB cap, a separate pair measured 532.247 versus 537.314 tokens/s
(+0.95%, a single pair, not a stable small-budget speedup claim).

For 200000 input tokens and 128 requested output tokens, capacity/new
measured prefill 370.561 / 370.958 s, or 539.723 / 539.145 tokens/s.
The -0.11% difference is noise-sized, not evidence of a stable regression
or gain. Both executed the identical 21-group/20-prefetched-layer plan,
with identical activation traffic and uncovered trunk bytes.
Decode was 6.451 / 6.267 tokens/s; this prefill-only policy does not
change decode execution, so no decode performance effect is claimed.

All twelve complete logits files (six initial short checks, four warm
short checks and two long checks) matched the saved originals byte-for-
byte. Both long runs matched all 128 original output IDs. Within each
input size, batch, matrix/quantization/bounds counts and total uploaded
weight bytes were unchanged; probes forced no waits.
No Xid entries were found in the current-boot kernel log after the runs.
The shared system's clocks were not locked and other GPUs, models and
link speeds remain unvalidated. These measurements validate the bounded
SM86 policy, not a universal optimality claim for its byte heuristic.

### Exact indexer register-prefetch pipeline

The F32 tiled scorer still processes four queries, 64 pooled keys,
32 heads and 128 dimensions. It keeps the same 32-dimension reduction
tiles, per-output FMA order and ordered head sum. After staging the
current inputs into the existing transposed shared-memory tile, each
thread prepares the next four query `float4`s and two key `float4`s in
registers. The next iteration reuses those values.

There is no extra global/shared tensor, precision change, top-k change
or changed route accumulation. SM86 automatically selects this for
the 32-head/128-width tiled DSA geometry. Scalar/parallel scoring,
reference and unvalidated architectures retain the old loads.
Private `DsaScoreLoad` permits isolated comparisons; no ordinary CLI
speed switch was added. `score_load=prefetch|direct` reports selection.
Nested scopes and invalid policy values are tested.

An earlier `cp.async` prototype added continuous staging before
transposition. Its shared memory increased from 32768 to 57344 B;
the dynamic-shared direct/prefetched variants used 80/115 registers.
At 64 queries and 50000 pools it measured about 1.60-1.73 versus
2.34-2.40 ms despite exact outputs. It was removed from production.
That dynamic-shared experimental control is not the final baseline.
The retained static-shared kernels use 93/122 registers, both with
32768 B shared and no stack/local spills. These are compiler resources,
not achieved occupancy measurements.

On the same RTX 3080 / W-2133, Q4_K_XL model, saved IDs, batch 2048,
14336 MiB prefill GPU cap, 32768 MiB host cap, 1024 MiB reserves,
six CPU threads, 4096 MiB decode expert cap and MTP off, only the
score-load policy changed:

| Measurement | Direct | Register prefetch |
|---|---:|---:|
| 32768 prefill tokens/s, direct/new/new/direct | 551.368 / 544.476 | 549.089 / 548.169 |
| 32768 score stream interval, ms | 1007.310 / 1055.803 | 914.927 / 911.050 |
| 200000 prefill, one pair | 382.421 s / 522.984 tokens/s | 378.672 s / 528.162 tokens/s |
| 200000 score stream interval | 36.963 s | 32.063 s |

Pooled short-input throughput changed +0.13%, essentially unchanged;
score intervals decreased 11.49%. The long pair changed overall
throughput +0.99% and score interval -13.25%. Retention is supported
by repeatable scorer microbenchmarks and isolated phase improvement,
not a claim of a large or universally stable whole-model speedup.
The 1/7-query, 50000-pool microbenchmarks measured about 0.11991/0.23833
ms direct versus 0.10317/0.20685 ms prefetched. The 64-query control
varied from 1.948 to 2.249 ms, versus 1.835-1.869 ms prefetched.
Clocks were not locked and hardware counters did not isolate
memory-latency hiding from other scheduling effects.

All six complete logits dumps matched saved originals byte-for-byte;
both long runs matched all 128 original output IDs. Matrix,
quantization/bounds and score-call counts, weight/activation bytes,
global GPU/host tensor peaks, cache hits/misses and group plans were
unchanged within each input size. No forced probe waits occurred.
The 32k plan remains 43 groups/max three layers, the 200k plan
41 groups/max four layers, with batch 2048 throughout.
No decode speedup is claimed.

CUDA SM86 build, all 13 GLM CTests and strict standard/no-MMQ syntax
checks passed. Exact tests cover query/pool tails, varied shapes,
scale stress, signed zeros, output guards and unchanged inputs.
Fixtures compare exact state/logits and MTP acceptance/rejection,
append, reset and dispatcher paths. Low-level prefetch memcheck,
racecheck and synccheck reported zero errors; racecheck also zero
warnings. Full `glm5_indexer_test`, including state/MTP and append
fixtures, also passed memcheck with zero errors and zero leaked bytes.
A normal-CLI 32k run with probes disabled selected prefetch,
preserved the original logits and retained resource/matrix counts.
Other GPU backends and hardware were not built or tested.

### Wider weight lookahead and exposed bank upload checks

On the same RTX 3080 / W-2133 and Q4_K_XL model, with the latest
register-prefetched indexer and transfer-aware grouping, only the existing
`--prefill-prefetch-layers 1|2` changed. Prefill used available VRAM,
batch 2048, 32768 MiB host budget, 1024 MiB reserves, six CPU threads,
4096 MiB decode expert cap, DSA/Flash/tail 0 and MTP off.

| Measurement | One-layer limit | Two-layer limit |
|---|---:|---:|
| 32k tokens/s, 1/2/2/1 | 554.759 / 541.648 | 545.081 / 540.318 |
| 32k groups / prefetched layers | 43 / 42 | 42 / 42 |
| 32k uncovered trunk payload, MiB | 881.07 | 881.07 |
| 32k tracked GPU peak, MiB | 12905.71 | 16374.82 |
| 200k prefill, one pair | 361.609 s / 553.084 tokens/s | 378.461 s / 528.456 tokens/s |
| 200k groups / prefetched layers | 21 / 20 | 40 / 40 |
| 200k exposed weight wait | 17.702 s | 1.534 s |
| 200k activation H2D / D2H, MiB | 262500 / 250000 | 500000 / 487500 |

Pooled 32k throughput changed -0.99%, with no reliable benefit.
The long pair changed -4.45%: less exposed weight loading did not
compensate for nearly doubled activation traffic.
The planner reserves capacity for the configured lookahead before
growing groups, but actual prefetch is clipped to the immediately
following group's size. Most intermediate groups under the two-layer
limit become single-layer groups and still prefetch only one layer.
The added total prefetch coverage comes largely from extra group
boundaries, not full prefetch of two-layer groups.
These are workload-specific comparisons, not a general proof against
larger prefetch on other cards. No default was changed.

A separate complete `blk.4` bank microbenchmark compared existing
direct uploads and 1/8 MiB staging. After a direct warmup, the
direct/1/8/8/1/direct samples measured:

| Upload mode | Two samples, ms | API calls |
|---|---:|---:|
| Direct | 668.585 / 641.217 | 31 |
| 1 MiB staging | 703.990 / 643.375 | 8710 |
| 8 MiB staging | 673.343 / 672.582 | 1111 |

Each transfers 4557834328 B into a 4557982720 B arena. Timing includes
bank allocation, conversion, packing and readiness, not just DMA.
Sampled tensor prefixes/suffixes matched; this is not a full real-weight
byte comparison. The existing complete-fixture upload test passed.
Staging did not reliably reduce readiness time, so no deferred
uncovered-bank executor path was added based on this microbenchmark.
It did not test compute overlap and does not disprove other async designs.

All six model logits dumps matched saved originals byte-for-byte;
both long runs matched all 128 original output IDs. Batch,
matrix/quantization/bounds counts and total uploaded weight bytes
were unchanged within each input size; probes forced no waits.
`glm5_upload` and `glm5_full_layer` passed and the current-boot
kernel journal contained no NVIDIA Xid entries.
Production code and defaults are unchanged. Cross-group lookahead,
safe weight-slot reuse and earlier partial uploads remain separate
experiments requiring lifetime, capacity and performance validation.

### Rejected Flash shared-memory matrix-load experiments

Three isolated `ldmatrix` variants preserved the original Flash tile,
MMA, query/probability decomposition, softmax and compensated accumulation:
all QK/PV fragments; only PV's transposed KV fragment; and only QK.
Inputs/outputs retained their original layouts and KV half-tile prefetch.
The full variant was also tested without unnecessary whole-memory
clobbers on read-only fragment loads.

On the RTX 3080, 64 heads, 50000 cached cells and 2048 selected cells,
the 64-query direct/new/new/direct samples (30 repeats per sample) were:

| Variant | Original manual loads, ms | Matrix loads, ms |
|---|---:|---:|
| All QK/PV fragments | 1.323179 / 1.319181 | 2.148796 / 2.147661 |
| PV KV fragment only | 1.311573 / 1.329693 | 1.418445 / 1.423155 |
| QK fragments only | 1.315567 / 1.328606 | 1.438508 / 1.434385 |

Elapsed intervals increased about 62.60%, 7.58% and 8.65%.
One/17-query cases were also slower. Removing excess clobbers did not
reverse the full variant's result. The initial full variant reduced
head-major/prefetch registers from its experimental control's 242 to
198, without stack/local spills or more shared memory, but did not
improve speed. The control's fragment restructuring also affected
compiler allocation; after removal the original kernel returns to
234 registers. No hardware counters identify bank conflicts,
stalls or achieved occupancy as the cause.

All candidates matched their control outputs byte-for-byte across both
layouts, varied head/query counts, dense/sparse masks, KV prefetch
on/off, query windows, singleton storage, scale stress, signed zeros,
unchanged query inputs and output guards.
The candidate kernels, TLS flags, API and dedicated test modes were
removed completely. Original `glm5_attention` and `--prefetch-check`
passed after restoration and the current-boot journal had no Xid entries.
No candidate real-model throughput or sanitizer run was performed.
Production code/defaults are unchanged; these results reject these
specific fragment-loading layouts, not all attention optimizations.

### Rejected static CPU-cost cache quotas

An isolated benchmark calibrated six-thread `CpuRowPool` full-expert
latency on the real model's 42 MoE layers: one token and one expert,
four alternating forward/reverse passes, first pass warmup. Timing
includes activation preparation, gate/up, SwiGLU, hidden quantization,
down and row-pool dispatch, but not pool creation or GPU work.
Measured layer averages were 0.432-0.546 ms in the 8k experiment and
0.502-0.636 ms in the separately calibrated 200k experiment.
Clocks were not locked and the calibration activation is synthetic.

The session-only candidate replaces each marginal heat gain with
`expert_heat / layer_heat * cpu_ms / slot_bytes`. It retains one slot
per layer, capacity, ties, warming and runtime replacement rules.
This estimates CPU time saved per cache byte, not total CPU/GPU
execution cost: GPU hit and cache promotion costs are omitted.
The production cache policy was not modified.

Both baseline and candidate use existing prompt warming and heat
quotas, the same saved prefill HostState, 4096 MiB cap, six CPU threads,
DSA/Flash/tail 0 and MTP off on the RTX 3080 / W-2133 with Q4_K_XL.
Prefill budgets remain 14336 MiB GPU / 32768 MiB host with 1024 MiB
reserves. Decoder instances are recreated for baseline/new/new/baseline
comparisons, followed by two candidate runs at a cap rounded up from
baseline's actual allocation.

| Decode measurement | Existing heat quotas | CPU-cost quotas |
|---|---:|---:|
| 8k/64 tokens/s, ABBA | 6.635 / 6.644 | 6.673 / 6.644 |
| 8k actual cache, MiB / slots | 4090.188 / 281 | 4088.062 / 281 |
| 8k hits / misses | 7908 / 13596 | 7910 / 13594 |
| 8k logical CPU reads, GiB | 194.824 | 194.827 |
| 200k/128 tokens/s, ABBA | 6.905 / 6.894 | 6.914 / 6.896 |
| 200k actual cache, MiB / slots | 4089.938 / 281 | 4089.938 / 281 |
| 200k hits / misses | 19025 / 23983 | 19025 / 23983 |
| 200k logical CPU reads, GiB | 344.030 | 344.030 |

The 8k candidate changes six layers by one slot each; actual cache
is 2.125 MiB smaller (about 0.052%), so this is approximate, not exact,
equal allocation. Pooled throughput changes +0.28%, hit rate barely
changes and CPU reads do not decrease. The long-input quota vectors
are identical; pooled +0.08% is timing variation with identical cache
placement/counters, not a strategy gain. The later matched-cap runs
are not additional ABBA controls, and calibration time is excluded
from decode rates. No reliable benefit supports production integration.

Both complete prefill logits dumps and all twelve decode ID sequences
matched their saved originals, including each full 128-token long output.
Existing cache/runtime regressions passed and no current-boot Xid entries
were found. No candidate sanitizer or full MTP run was performed.
Defaults, production source and replacement/admission logic are unchanged.
Dynamic decode-heat redistribution and other hardware/cost models remain
separate research; these results do not establish universal CPU/GPU parity.

### Rejected prompt-prior weakening and faster decode heat aging

A session-only cache-policy experiment kept quotas, initial warm
resident experts, capacity and admission rules unchanged, comparing
the original maximum prior 32 / decay every 4096 per-layer observations
with prior 8, decay every 64 observations, and their combination.
The interval counts `observe` calls, not elapsed time or necessarily
tokens in a batched/MTP dispatch. Nonzero prior counts still have a
minimum of one. The original two-count admission margin and
one-existing-slot replacement limit remain.

Each input size prefills once and recreates decoders from the same
HostState. Both use existing heat quotas/prompt warming, a 4096 MiB
expert cap, six CPU threads, DSA/Flash/tail 0, MTP off and the latest
retained compute choices on RTX 3080 / W-2133 with Q4_K_XL.
Prefill limits remain 14336 MiB GPU / 32768 MiB host and 1024 MiB
reserves. The run order was original/weak/fast/combined followed by
its reverse. Actual allocations are identical across modes:
4090.188 MiB at 8k and 4089.938 MiB at 200k, 281 slots/warm experts.

| Pooled decode tokens/s | Original | Weak prior | Faster aging | Combined |
|---|---:|---:|---:|---:|
| 8k input / 64 output | 6.554 | 6.377 | 6.597 | 6.385 |
| 200k input / 128 output | 7.012 | 6.953 | 6.962 | 6.931 |

At 200k/128, weak prior increased hits 19025 to 19376 and reduced
logical CPU reads 344.030 to 339.049 GiB, but replacements increased
97 to 165 and cache uploads 5.370 to 6.332 GiB. It did not improve
whole decode. Faster aging reduced hits to 18803, with 347.169 GiB
CPU reads, 121 replacements and 5.713 GiB cache uploads. Combined
had 19170 hits, 173 replacements and 6.448 GiB uploads.
These are correlated counters, not isolated proof of PCIe causation.
The 8k faster-aging run had identical cache counters to baseline,
so its +0.66% timing change is not evidence of a cache improvement.
Clocks were not locked; sub-percent differences are not stable claims.

Both complete prefill logits dumps and all sixteen decode sequences
matched saved originals, including all eight full 128-token outputs.
Probes forced no waits; original cache/runtime tests passed.
No candidate has a reliable overall benefit, so all three remain
rejected experiments, with no production/default changes.
Candidate sanitizers and full MTP validation were not run.
Other horizons and cost-aware admission remain separate research;
hit rate alone does not account for cache promotion traffic or time.

### Latest automatic decode cache budget check

The earlier 4096 MiB cap was an explicit experimental control, not
the engine's maximum or automatic policy. Existing
`--glm-gpu-experts 0 --glm-gpu-reserve 1024` allocates expert capacity
from free VRAM after major decoder allocations, minus reserve.
Independent prefill releases its GPU banks/workspace before decoder
creation. Decoder state is allocated for the configured maximum context;
this is startup sizing, not continual runtime cache resizing.
The original off-by-default expert-cache setting was not changed.

The latest retained code was tested on the same RTX 3080 / W-2133
and Q4_K_XL model with 200000 input / 128 output, six CPU threads,
DSA/Flash/tail 0, MTP off and existing prompt warming/heat quotas.
Prefill limits were 14336 MiB GPU / 32768 MiB host with 1024 MiB
reserves. One prefill HostState was reused in newly created decoders
with fixed/automatic/automatic/fixed expert budgets:

| Measurement | 4096 MiB cap | Automatic cache |
|---|---:|---:|
| Decode tokens/s | 6.828 / 6.843 | 7.830 / 7.876 |
| Pooled tokens/s | 6.836 | 7.853 |
| Setup seconds | 2.089 / 2.416 | 2.816 / 2.896 |
| Actual cache, GiB / slots | 3.994 / 281 | 7.367 / 517 |
| Hits / misses | 19025 / 23983 | 24641 / 18367 |
| Hit rate | 44.236% | 57.294% |
| Logical CPU reads, GiB | 344.030 | 263.730 |

Pooled decode improved 14.88%, spending more VRAM and setup time.
This is a resource-budget comparison, not an equal-memory algorithm
gain. Setup is excluded from decode rates. The latest best sample
is 7.876 tokens/s, with paired automatic pooled throughput 7.853.
Capacity depends on free VRAM at creation; clocks were not locked.

The complete logits and all four 128-token sequences matched saved
originals byte-for-byte. Each mode's allocation/hit/read/swap/upload
counts repeated identically, with no forced probe waits.
Other contexts, MTP and concurrent workloads remain untested here.
No production source, defaults or cache algorithm changed.
Future reports should distinguish fixed-resource optimization controls
from automatic-budget user performance, rather than presenting the
4 GiB controlled configuration as the best available speed.

### Default automatic expert-cache capacity

Following the latest automatic-budget measurement, omitted expert
capacity now resolves to zero (automatic) on the validated CUDA SM86
4096-hidden/2048-expert-FFN/288-expert/top-8 geometry. Other geometry,
backends and reference mode retain a disabled omitted cache.
Explicit `--glm-gpu-experts -1|0|N` always wins; reference can still use
an explicitly requested cache. Reserve defaults to 1024 MiB and remains
overridable. Insufficient capacity still reports the existing allocation
error rather than silently shrinking context, disabling the cache or
running past reserve.

The internal `default_gpu_expert_budget` sentinel distinguishes omission
from an explicit off choice; it is resolved before cache construction,
and is not accepted as a CLI numeric input.
Prefill-only calls retain lazy decoder creation and allocate no expert
cache. Prompt warming, heat quotas, staging, CPU thread count, DSA,
Flash and MTP defaults were not changed.
Thus the earlier 7.853 pooled tokens/s with explicit warming and heat
quotas is not a claim about all default settings.

Validation on the real model used 8192 saved input IDs and 64 outputs,
six CPU threads, DSA/Flash/tail 0, identical 14336/32768 MiB prefill
budgets and no warm/heat flags. Omitted cache and explicit
`--glm-gpu-experts 0` both allocated 672 slots / 9827.0 MiB, reported
11164 hits / 10340 misses / 548 swaps, and produced byte-identical
complete logits and output IDs. These are selection/parity checks,
not a matched default-speedup benchmark. Policy tests cover off,
automatic, positive caps, reference and unsupported geometry/backends.
All 13 GLM CTests and strict C++ syntax checks passed after the change.
