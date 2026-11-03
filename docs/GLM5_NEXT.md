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
DSA state transfer is not implemented and is refused.

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

The PR's cache policy is adapted to the direct-GGUF loader: equal per-layer slot
quotas, LFU counts, a two-observation admission margin, at most one replacement
per layer/decode call, and count aging every 4096 calls. Both directions of the
slot/owner table are checked when a slot is reassigned.

Resident decode experts run on the GPU while CPU misses are computed. A prefill
chunk groups all occurrences of each expert, streams its original GGUF matrices
through the slots and computes up to eight rows together. The final slot table
is reused by decode in the resident chunk path. Independent full-layer prefill
does not use these slots; decode starts its cache after the phase handoff.
The gate/up/down kernels apply the same GLM clamp as the CPU.
This port does not yet include the PR's double-wave DMA overlap or pinned RAM
arena.

`--glm-gpu-check` recomputes GPU results on the CPU and fails on non-finite values
or a row error larger than that CPU row's RMS, following the PR's diagnostic
criterion. It is an expensive wrong-slot check, not an accuracy guarantee:
GPU Q8_1 and CPU weight-specific activation quantizers are different.

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
the fused indexer yet.

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

## Current limits and validation needed

- Text only, one GPU, one sequence, greedy sampling.
- No HTTP server or stdin engine-service protocol, desktop setup, Terminal
  model catalog, automatic engine download or release package.
- No multi-GPU scheduling or persistent conversation checkpoints.
- MLA uses the full cache. DSA math and parity fixtures are retained from the PR,
  but this runtime does not expose a DSA switch.
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

Not run: HIP builds, other GPU architectures, broad quality evaluations and
long-context DSA validation. Passing the synthetic tests does not establish
full-model numerical equivalence.

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
