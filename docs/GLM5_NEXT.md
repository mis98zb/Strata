# Experimental GLM-5.3-Flash engine

This branch starts a standalone `glm5-next` engine, following the directory and
build separation used by DeepSeek. It is not a released model family or a desktop
installer option. No performance or hardware compatibility is claimed by this
port until it has been built and checked on that hardware.

The GLM block, six GPU kernel files and GPU parity fixtures come from
[Strata PR #1315](https://github.com/Niko1221/Strata/pull/1315), at commit
`7dbde61cb46a936b68f06e6053d682cc992e9cb2`. That PR is closed and was not merged.
Its reported measurements are not measurements of this branch.

The prefill and MTP semantics were also checked against the local llama.cpp
checkout at `e129d48751534b1510619d2d719bfd6771827d9c`, especially
`src/models/glm5-next.cpp` and `common/speculative.cpp`. The reference checkout
is not modified or required at runtime.

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

The runtime reads the original GGUF shards directly. There is no GLM-specific
canonical pack: dense projections stay in their supported native GPU format,
small tensors are converted to FP32, and routed experts stay mapped on the CPU
unless the optional GPU tier holds them.
CPU experts use ggml's initialized quantizers and dot products, with the GLM clamp
applied after SiLU. CPU misses still run serially; this does not reproduce the
PR's optimized CPU expert pool or its throughput.

## Optional acceleration

All three paths are off by default. They can be enabled separately or together.

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
is reused by decode. The gate/up/down kernels apply the same GLM clamp as the CPU.
This port does not yet include the PR's double-wave DMA overlap or pinned RAM
arena.

`--glm-gpu-check` recomputes GPU results on the CPU and fails on non-finite values
or a row error larger than that CPU row's RMS, following the PR's diagnostic
criterion. It is an expensive wrong-slot check, not an accuracy guarantee:
GPU Q8_1 and CPU weight-specific activation quantizers are different.

### Chunked prefill

`--prefill N` reads up to `N` prompt tokens per layer (1 by default, maximum
4096). Dense projections share a weight read over groups of at most eight
tokens. KDA convolution and recurrence still update in token order; MLA cache
writes and causal attention use absolute positions. Router outputs and mHC
post-mix values remain separate for every token.
KDA's L2 norm uses the reference's `1e-6` floor, separately from the model's RMS
normalization epsilon.

This follows the reference's layer-major graph semantics, but does not yet use
llama.cpp's large-chunk MMQ/GEMM planner or fused indexer. A chunk can exceed
eight tokens; its dense projections are then processed in eight-token groups.

### Greedy MTP verification

`--mtp N` enables up to `N` draft tokens per round (0 by default, maximum 7).
The target verifies the sampled base token and the draft prefix in one batch
of at most eight inputs. Only the matching greedy prefix is committed.

The NextN layer uses the reference's pair `(h[p-1], x[p])`: the target's
post-output-norm hidden state and the next input token. The first hidden state
is zero; the first token embedding is not masked. Every prompt position warms
the draft cache. Optional NextN embeddings/head weights are used when present;
otherwise the main model's embedding, output norm and output head are reused.

On rejection, the KDA states and convolution histories are restored, and the
accepted target prefix is replayed. MLA rows are append-only and become
invisible beyond the restored position. The draft's speculative rows are
replaced by catch-up steps using verified target hidden states. This is a
verification/acceptance loop, not the old PR's measurement-only draft probe.

Prompt catch-up currently runs the NextN layer token by token. KDA checkpoints
and rejection replay also cost time. Acceptance and speed must be measured on
the actual model; enabling MTP is not a promise of a speedup.

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

```sh
cmake -S . -B build-glm5 -G Ninja \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_GLM5=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-glm5 --target strata-glm5 glm5-inspect glm5_format_test \
  glm5_cache_test glm5_parity glm5_head_test glm5_runtime_test
ctest --test-dir build-glm5 -R '^glm5_(format|cache|parity|head|runtime)$' --output-on-failure
```

For HIP, replace the CUDA option with `-DSTRATA_ENABLE_HIP=ON` and use the architecture and toolchain
settings in [AMD_HIP.md](AMD_HIP.md). This port does not enable gfx906, SYCL or
Metal support.

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

Reserve context space for the prompt plus the requested output. Generated
non-stop tokens are committed to the model state, including the last one.
Cache and MTP counters are printed to stderr, separately from the token CSV.

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
CUDA 13.3 and `CMAKE_CUDA_ARCHITECTURES=86`, using the local llama.cpp checkout
above for ggml-cpu. GPU tests ran on an NVIDIA GeForce RTX 3080 with 20 GiB
reported VRAM.

Passed:

- The standalone engine, format/cache tests, GPU math parity, head and runtime
  tests built. The five GLM CTest entries passed.
- The host-only format/cache and existing geometry, split-shard and architecture
  default tests passed (five entries).
- The runtime tests exercised four synthetic trunk layers and one NextN layer,
  Q8_0 experts, one-slot eviction, ragged prefill widths 2/7/8/17/32, accepted and
  rejected drafts, reset, continuation-state comparisons and combined modes.
- The CLI produced exactly 12 verified tokens and a 32-row FP32 logits file
  from a synthetic model with all three options enabled.
- CUDA Compute Sanitizer memcheck reported zero errors and zero leaked bytes
  for the runtime test.
- Strict C++ syntax/type checks passed. Python tokenizer tests ran 23 entries
  with three real-vocabulary skips.

The runtime checks also caught two porting issues that are fixed here: the
ggml CPU tables must be initialized before direct dot products, and chunk
arenas must align whole column groups rather than pad every row separately.

Not run: HIP builds, other GPU architectures, and inference/quality/performance
tests on real GLM-5.3-Flash weights. Passing the synthetic tests is not a
full-model speed or quality result.

To repeat the memory check after building:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 --leak-check full \
  build-glm5/src/glm5/glm5_runtime_test
```
