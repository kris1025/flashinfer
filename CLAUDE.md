# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashInfer is a GPU kernel library for LLM serving that uses **JIT (Just-In-Time) compilation by default**. This means kernel code changes are automatically picked up without reinstalling the package - extremely convenient for development.

## Quick Reference

| Task | Command |
|------|---------|
| Install for development | `pip install --no-build-isolation -e . -v` |
| Initialize submodules | `git submodule update --init --recursive` |
| Install CUPTI for benchmarking | `pip install -U cupti-python` |
| Run regular tests | `pytest tests/` |
| Run full parameter matrices | `pytest tests/ --full` |
| Run specific test | `pytest tests/path/test_file.py::test_function` |
| Run multi-GPU test | `mpirun -np 4 pytest tests/comm/test_allreduce_unified_api.py` |
| Run benchmark | `python benchmarks/flashinfer_benchmark.py --routine <name> <flags>` |
| Run linting | `pre-commit run -a` |
| Dump environment report (bug reports) | `python -m flashinfer.collect_env` (or `flashinfer collect-env [--json]`) |
| Install pre-commit hooks | `pre-commit install` |
| Clear JIT cache | `rm -rf ~/.cache/flashinfer/` |
| Enable API logging (basic) | `export FLASHINFER_LOGLEVEL=1` |
| Enable API logging (detailed) | `export FLASHINFER_LOGLEVEL=3` |
| Enable API logging (with stats) | `export FLASHINFER_LOGLEVEL=5` |
| Set API log destination | `export FLASHINFER_LOGDEST=mylog.txt` |
| Enable verbose JIT logging | `export FLASHINFER_JIT_VERBOSE=1` |
| Enable debug build | `export FLASHINFER_JIT_DEBUG=1` |
| Set target architectures | `export FLASHINFER_CUDA_ARCH_LIST="8.0 9.0a"` |
| Set parallel compilation | `export FLASHINFER_NVCC_THREADS=4` |
| Limit parallel ninja jobs | `export MAX_JOBS=4` |
| Enable GDN native short-T path | `export FLASHINFER_GDN_WY_NATIVE_T=1` |
| Enable GDN strided QKV path | `export FLASHINFER_GDN_WY_STRIDED_QKV=1` |
| Enable GDN native A/B tensors | `export FLASHINFER_GDN_WY_NATIVE_AB=1` |
| Let `backend="auto"` pick experimental backends | `export FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1` |
| Override ragged-prefill `auto` backend order (Blackwell) | `export FLASHINFER_RAGGED_AUTO_BACKEND_ORDER=cutlass,cudnn` |
| Override CuTe-DSL prefill scheduling | `export FLASHINFER_CUTE_PREFILL_PERSISTENT=0` (non-persistent) or `1` (persistent) |
| Skip MoE EP CuTe-DSL import/version guard | `export FLASHINFER_MOE_EP_SKIP_DSL_CHECK=1` |
| Override MoE EP knob-cache path | `export FLASHINFER_MOE_EP_KNOB_CACHE=/path/to/knobs.json` |
| Disable MoE EP fused staging kernel | `export FLASHINFER_MEGA_FUSED_STAGE=0` |
| Enable distribution-aware MoE autotune and kernel dispatch (experimental; TRT-LLM MoE only) | `export FLASHINFER_DIST_AWARE_AUTOTUNE=1` |
| Prims-TS debug checks / task-manager verify | `export FLASHINFER_PRIMS_TS_DEBUG_CHECKS=1` |
| Override Prims-TS CuTe-DSL compile options | `export FLASHINFER_PRIMS_TS_COMPILE_OPTIONS="--opt-level 2"` |

The minimum Python version used by CI and build tooling is defined in
`.python-version`. CI Docker images use a stable Conda environment name and
derive their interpreter and site-packages paths from that file. When raising
the minimum, also update `project.requires-python` and `tool.mypy.python_version`
in `pyproject.toml`.

## Quick Start for Development

### Installation

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
pip install --no-build-isolation -e . -v
```

**Important**: The `--recursive` flag is required to initialize submodules in `3rdparty/` (cutlass, spdlog).

If you forgot `--recursive` when cloning:
```bash
git submodule update --init --recursive
```

That's it! You can now:

- Run all benchmarks and unit tests
- Modify kernel source code in `include/` without reinstalling
- Changes are JIT-compiled on next use

The `--no-build-isolation` flag prevents pip from pulling incompatible PyTorch/CUDA versions from PyPI.

### How JIT Compilation Works

When you call a FlashInfer API:

1. **First call**: Generates specialized CUDA code based on parameters (dtype, head_dim, etc.), compiles it with ninja, caches the .so file
2. **Subsequent calls**: Uses cached compiled module
3. **After kernel changes**: Automatically detects changes and recompiles

**No manual rebuild step needed** - just edit `.cuh` files and run your code again.

### Pre-compiled Packages (Optional)

FlashInfer provides optional pre-compiled packages for users who want faster initialization:

- `flashinfer-jit-cache`: Pre-built kernel cache
- `flashinfer-cubin`: Pre-compiled kernel binaries

**For development, you typically DON'T need these.** JIT compilation is fast enough and gives you live code reload.

## Testing

Run the regular test suite:

```bash
pytest tests/
```

Run full parameter matrices (used by nightly testing):

```bash
pytest tests/ --full
```

Run specific test file:

```bash
pytest tests/attention/test_hopper.py
```

Run specific test function:

```bash
pytest tests/attention/test_hopper.py::test_single_prefill
```

### Skipping Tests Based on CUDA Architecture

Use `flashinfer.utils` functions to skip tests on unsupported GPU architectures:

**Available check functions:**
- `get_compute_capability(device)` - Returns `(major, minor)` tuple
- `is_sm90a_supported()` - Hopper (requires CUDA 12.3+)
- `is_sm100a_supported()` - Blackwell (requires CUDA 12.8+)
- `is_sm100f_supported()` - Blackwell feature-set (requires CUDA 12.9+)
- `is_sm110a_supported()`, `is_sm120a_supported()`, `is_sm120f_supported()`, `is_sm121a_supported()`

**APIs decorated with `@backend_requirement`** also provide:
- `api_name.is_compute_capability_supported(cc)` - e.g., `mm_fp4.is_compute_capability_supported(100)`
- `api_name.is_backend_supported("backend")` - e.g., `mm_fp4.is_backend_supported("cudnn")`
- `api_name.is_backend_supported("backend", cc)` - e.g., `mm_fp4.is_backend_supported("cudnn", 80)`

**Example:**
```python
from flashinfer.utils import is_sm90a_supported

def test_hopper_attention():
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("Requires SM90a")
    # Test code...
```

**Common requirements:**

| Feature | Min SM | Check Function |
|---------|--------|----------------|
| FlashAttention-3 | SM90a | `is_sm90a_supported()` |
| MLA Attention | SM100a | `is_sm100a_supported()` |
| FP8 GEMM | SM89+ | `get_compute_capability()[0] >= 9` |

**Note:** `tests/conftest.py` auto-skips tests that trigger OOM, but tests should be written to avoid OOM by using appropriate problem sizes.

## Benchmarking

FlashInfer provides a unified benchmarking framework in `benchmarks/flashinfer_benchmark.py`.

**Key features:**
- Supports attention, GEMM, and MOE kernels
- Multiple backends: FlashAttention2/3, cuDNN, CUTLASS, TensorRT-LLM, cuBLAS
- **CUPTI timing (recommended)**: Hardware-level profiling for accurate GPU kernel time
  - Automatically falls back to CUDA events if CUPTI unavailable
  - Install: `pip install -U cupti-python` (requires CUDA 13+)
- Batch testing, reference checking, CSV output

**Quick example:**
```bash
python benchmarks/flashinfer_benchmark.py \
    --routine BatchDecodeWithPagedKVCacheWrapper \
    --backends fa2 cudnn \
    --batch_size 32 --s_kv 2048 \
    --num_qo_heads 32 --num_kv_heads 8 \
    --head_dim_qk 128 --head_dim_vo 128 \
    --page_size 16 --refcheck -vv
```

**Python API:**
```python
from flashinfer.testing import bench_gpu_time

# CUPTI preferred, auto-fallback to CUDA events.
# Returns per-iteration times in milliseconds (a list).
times = bench_gpu_time(
    my_kernel, input_args=(x, y), enable_cupti=True,
    dry_run_iters=10, repeat_iters=30,
)
median_time_ms = statistics.median(times)
```

→ **For complete benchmarking guide, see [`.claude/skills/benchmark-kernel/skill.md`](.claude/skills/benchmark-kernel/skill.md)**

`flashinfer.gemm.group_gemm_fp8_nt_groupwise_contiguous` is the standalone
CuTe-DSL grouped FP8 API; `group_deepgemm_fp8_nt_groupwise` keeps its original
DeepGEMM-only signature. Both use per-row expert indices, unlike the indptr-based
`group_gemm_fp8_nt_groupwise` API.

For contiguous grouped FP8 GEMM, `python benchmarks/bench_grouped_fp8.py
--production-shapes` compares DeepGEMM and CuTe-DSL with preallocated outputs;
`--cache-probe` measures first-call and new-token-count compilation separately.
Compiled grouped CuTe-DSL kernels are reused across token counts with the same
128-row alignment class, with at most two variants per device and weight shape.
Concurrent misses share one compilation; warm hits do not acquire the compile lock.
The standalone API's optional `validate_indices=True` checks expert-index values
and synchronizes with the CPU. Validate new routing data before CUDA graph
capture; the default path assumes valid indices and performs metadata checks only.

## Code Linting

Run all pre-commit hooks:

```bash
pre-commit run -a
```

Install hooks to run on every commit:

```bash
pre-commit install
```

## Opening a Pull Request

Follow [`CONTRIBUTING.md`](CONTRIBUTING.md) when opening a PR. The two requirements most often
missed:

- **Keep the default PR template.** Fill in `.github/pull_request_template.md` (Description,
  Related Issues, Checklist, Tests, Reviewer Notes). Do **not** overwrite or replace it with a
  custom or tool-generated description format. The title and description normally become the
  commit title and message on squash-merge, and are what a `git bisect` surfaces months later
  when someone is hunting the owner of a regression.
- **Report before/after numbers for performance work.** A perf PR must include measurements from
  a reproducible benchmark (e.g. `benchmarks/flashinfer_benchmark.py`), naming the GPU and the
  problem sizes. Speedup ratios without absolute numbers, or numbers without a named GPU, are
  not enough.

→ **For the complete contribution rules, see [`CONTRIBUTING.md`](CONTRIBUTING.md)**

## Code Review

When reviewing a diff (as an agent or a human), follow the shared focus areas, kernel-review
policy, and effort calibration in [`docs/code_review_guidance.md`](docs/code_review_guidance.md).
Note: unlike human review, agents keep **kernel implementation details in scope** — read the
kernel logic and report bugs, labeling findings by confidence.

→ **For the complete review rule set, see [`docs/code_review_guidance.md`](docs/code_review_guidance.md)**

## Architecture: JIT Compilation System

FlashInfer's JIT system has three layers:

### Layer 1: JitSpec (flashinfer/jit/core.py)

`JitSpec` is an abstract base class defining the kernel-module lifecycle
(`try_load()` / `build()` / `load()`, with the shared `build_and_load()`
template method handling caching, locking, and `FLASHINFER_DISABLE_JIT`).
One subclass per compilation toolchain:

- `JitSpecNvcc` (nvcc/ninja modules, returned by `gen_jit_spec()`) defines:
  - `name`: Unique identifier (URI hash from parameters)
  - `sources`: List of .cu/.cpp files to compile
  - `extra_cuda_cflags`, `extra_cflags`, `extra_ldflags`: Compiler flags
- `JitSpecCuteDsl` (flashinfer/jit/cute_dsl_core.py) caches CuTe-DSL kernels
  (see "CuTe-DSL kernels" under Module Caching below)

### JIT Directory Rules

**NEVER write to package directories** - they may be read-only after installation.

| Directory | Writable | Use for |
|-----------|----------|---------|
| `FLASHINFER_GEN_SRC_DIR` | ✓ Yes | Generated source files (Jinja output, copied .cu files) |
| `FLASHINFER_JIT_DIR` | ✓ Yes | Compiled `.so` outputs |
| `FLASHINFER_CSRC_DIR` | ✗ No | Read-only source templates |
| `FLASHINFER_AOT_DIR` | ✗ No | Read-only pre-compiled binaries |

### Compilation Context: Architecture-Specific Compilation

FlashInfer uses `CompilationContext` to manage CUDA architecture targets. Some kernels only work on specific GPU architectures (e.g., Hopper SM90, Blackwell SM100/SM12x).

**How it works:**
- Auto-detects GPUs in system or reads `FLASHINFER_CUDA_ARCH_LIST` environment variable
- CuTe-DSL FP4 MM compilation parallelism can be controlled with `FLASHINFER_MM_FP4_CUTE_DSL_COMPILE_WORKERS`
- JIT modules specify `supported_major_versions=[9, 10, 11, 12]` to limit compilation to specific SM versions
- If GPU not supported → `RuntimeError: No supported CUDA architectures found`

→ **See [`.claude/skills/add-cuda-kernel/skill.md`](.claude/skills/add-cuda-kernel/skill.md) for usage examples**

### Layer 2: Code Generation

Every `gen_*_module()` function in `flashinfer/jit/` follows this pattern:

```python
def gen_some_module(dtype_in, dtype_out, ...):
    # 1. Compute unique identifier from parameters
    uri = get_some_uri(dtype_in, dtype_out, ...)

    # 2. Create generation directory
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri

    # 3. (Optional) Render Jinja template to generate type-specialized config
    # Skip this step if you don't need type specialization
    with open(jit_env.FLASHINFER_CSRC_DIR / "some_customize_config.jinja") as f:
        template = jinja2.Template(f.read())
    config_content = template.render(
        dtype_in=dtype_map[dtype_in],
        dtype_out=dtype_map[dtype_out],
        # ... more parameters
    )
    write_if_different(gen_directory / "some_config.inc", config_content)

    # 4. Copy source files to gen directory
    sources = []
    for fname in ["some_kernel.cu", "some_jit_binding.cu"]:
        shutil.copy(jit_env.FLASHINFER_CSRC_DIR / fname, gen_directory / fname)
        sources.append(gen_directory / fname)

    # 5. Return JitSpec
    return gen_jit_spec(uri, sources, extra_cuda_cflags=[...])
```

**Note**: If your operation doesn't need type specialization, you can skip step 3 entirely and just copy the source files directly.

### Layer 3: Compilation and Loading

`JitSpecNvcc` methods:

- `write_ninja()` - Generates `build.ninja` file
- `build()` - Executes `ninja` to compile sources
- `build_and_load()` - Compiles and loads via TVM-FFI

The generated `build.ninja` file uses nvcc to compile .cu → .cuda.o → .so, then loads via TVM-FFI.

### Jinja Templates (Optional)

**Note: Jinja templates are NOT required.** You can write C++ code directly without templating.

For operations that need type specialization, templates in `csrc/*.jinja` can generate C++ code:

```jinja
// Input template
using DTypeIn = {{ dtype_in }};
using DTypeOut = {{ dtype_out }};
constexpr int PARAM = {{ param_value }};

// After render
using DTypeIn = float16;
using DTypeOut = float16;
constexpr int PARAM = 128;
```

This allows the same CUDA template code to be compiled with different concrete types. However, if your operation doesn't need this, you can skip Jinja and write the `.cu` files directly.

## Directory Structure

```
flashinfer/
├── include/flashinfer/           # Header-only CUDA kernel templates
│   ├── attention/                # Attention kernels
│   ├── gemm/                     # GEMM kernels
│   ├── comm/                     # Communication kernels
│   ├── mma.cuh                   # Matrix multiply utilities
│   ├── utils.cuh                 # Common utilities
│   └── [...]
│
├── csrc/                          # Framework bindings (via TVM-FFI)
│   ├── *.cu                       # Kernel launcher implementations
│   ├── *_jit_binding.cu           # TVM-FFI exports
│   ├── *_customize_config.jinja   # Type config templates (optional)
│   └── [...]
│
├── flashinfer/                    # Python package
│   ├── jit/
│   │   ├── core.py                # JitSpec, compilation infrastructure
│   │   ├── cpp_ext.py             # Ninja build generation
│   │   ├── env.py                 # Workspace paths
│   │   ├── attention/             # Attention module generators
│   │   ├── gemm/                  # GEMM module generators
│   │   ├── fused_moe/             # MOE module generators
│   │   └── [...]
│   ├── gemm/                      # GEMM Python APIs
│   ├── fused_moe/                 # MOE Python APIs
│   ├── comm/                      # Communication Python APIs
│   ├── *.py                       # Other high-level Python APIs
│   ├── aot.py                     # AOT compilation for pre-built packages
│   └── [...]
│
├── tests/                         # Test suite
│   ├── attention/                 # Attention kernel tests
│   ├── gemm/                      # GEMM kernel tests
│   ├── moe/                       # MOE kernel tests
│   ├── comm/                      # Communication tests
│   ├── utils/                     # Utility tests
│   └── conftest.py                # Pytest configuration
│
└── build_backend.py               # PEP 517 build backend
```

### Critical Rule: Framework Separation

**Torch headers MUST NOT be included in `include/` directory files.**

- `include/`: Framework-agnostic CUDA kernels (accept raw pointers)
- `csrc/`: Framework bindings via TVM-FFI (currently PyTorch, but can support other frameworks)

## Adding a New Operation

→ **For complete step-by-step tutorial, see [`.claude/skills/add-cuda-kernel/skill.md`](.claude/skills/add-cuda-kernel/skill.md)**

**Quick overview of the process:**
1. Write kernel in `include/flashinfer/new_op.cuh` (framework-agnostic, raw pointers)
2. Write launcher in `csrc/new_op.cu` (PyTorch tensor handling)
3. Create TVM-FFI bindings in `csrc/new_op_jit_binding.cu`
4. (Optional) Create Jinja template for type specialization
5. Write JIT module generator in `flashinfer/jit/new_op.py`
6. Write Python API in `flashinfer/new_op.py` with `@functools.cache`
7. Write tests in `tests/`
8. Register in `flashinfer/aot.py` for AOT compilation
9. Export in `flashinfer/__init__.py`
10. Add a `TraceTemplate` in `flashinfer/trace/templates/` and wire it via `@flashinfer_api(trace=...)` (see below)
11. Add an example call in `tests/trace/example.py`, re-run to regenerate `fi_trace_out/`, and commit the new JSON files

### Trace Template Checklist (for new or updated APIs)

Every public API decorated with `@flashinfer_api` should also carry a `trace=` argument so that `fi_trace()` works and auto-dump produces a benchmark definition JSON.

1. **Create or update a `TraceTemplate`** in `flashinfer/trace/templates/<category>.py` (e.g., `norm.py`, `activation.py`, `cascade.py`, `gdn.py`). Define `axes`, `inputs`, `outputs`, and optionally a `reference` function.
2. **Wire the template** to the API: `@flashinfer_api(trace=my_trace)` on the Python function (or class method's `run()`).
3. **Add an example call** in `tests/trace/example.py` that exercises the new trace with realistic shapes.
4. **Regenerate examples**: `rm -rf tests/trace/fi_trace_out && python tests/trace/example.py` — verify the expected JSON appears.
5. **Update the docstring** in `tests/trace/example.py` to list the new file(s).
6. **Run tests**: `pytest tests/trace/ -v` — all template-consistency and end-to-end tests must pass.
7. **Commit the new JSON files** under `tests/trace/fi_trace_out/` alongside the code changes.

**Example implementations:**
- **Simple**: `flashinfer/norm/__init__.py` (RMSNorm) - no Jinja, good starting point
- **Moderate**: `flashinfer/sampling.py` - with Jinja templating
- **Complex**: `flashinfer/decode.py` - plan-run pattern, advanced workspace

## Experimental Code

Experimental backends live under `flashinfer/experimental/`; experimental
public APIs live in core marked with `@flashinfer_experimental_api` (defined in
`flashinfer/api_logging.py`). Using them is an explicit opt-in: calling an
experimental API, or naming an experimental backend with `backend="<name>"`,
needs no environment variable (both warn once). Only automatic selection is
gated: `backend="auto"` skips backends marked `@experimental_backend` unless
`FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`. Experimental code is JIT-only
(never registered in `flashinfer/aot.py`) and tested under
`tests/experimental/`.

→ **When working under `flashinfer/experimental/`, follow
[`flashinfer/experimental/CLAUDE.md`](flashinfer/experimental/CLAUDE.md); the
full policy is [`flashinfer/experimental/README.md`](flashinfer/experimental/README.md).**

### Experimental DeepSeek V4.1 decode

`flashinfer/deepseek_v41.py` exposes mixed-cache and window-only decode with
explicit `backend="cute-dsl"` (existing Frost default) or `backend="prims-ts"`.
There is no `"auto"` selector. Imports remain function-local; both backends share
`experimental/deepseek_v41/decode.py::_validate_inputs` and require
backend- and mode-specific plans. Scope is SM100 / H64 / Sq1 / D512. The primary
Prims-TS mode uses explicit `torch.float8_e4m3fn` Q `[B,1,64,512]` with typed,
contiguous, unit-scale E4M3 SWA
`[pages,64,1,512]`, without a scale plane. The optional writer uses
`format="swa_fp8"` (finite clamp to [-448,448], round-to-nearest-even). This is not
MXFP8 with ignored E8M0 scales; decode never implicitly quantizes BF16 Q.
Frost retains BF16 Q and uint8 MXFP8 SWA (width 528). That explicit pairing also
selects the optional Prims-TS compatibility adapter in `prims_ts_mxfp8.py`, added
in a separate follow-up commit and imported only for BF16 Q. Native E4M3 remains
the primary implementation; no public decode-format argument is added. Both
backends accept optional uint8 FP4 global (width 288), using the existing
page-plane ABI.

The separate Prims-TS adapter requires **CuTe-DSL >= 4.8 with its bundled CUDA 13
compiler**. Fixed **Keep-AB / M64 / retained-KV8 / split 1 / nonpersistent**
execution uses BK128 and E4M3 Q/K/V/P for both native source passes, not measured
crossover heuristics. Each call refreshes live selections, execution lengths and
valid counts on the GPU. Ordinary FP8 SWA uses native TMA/QK/P/PV; it is not
dequantized to BF16. Main/global storage remains packed E2M1 with group16 E4M3
scales, width 288 and the existing page-plane ABI. Each request computes the exact
amax of its valid selected main values after applying those group scales, then
`s = div.rn.f32(amax, 448)` (1 for zero/empty selections). Only selected main tiles
are normalized by `div.rn.f32(x, s)` and rounded/saturated to E4M3, preserving
signed zero/subnormals. Scales below 1 are allowed; unused/invalid slots never
contribute. The same request descale restores K in the softmax factor and V in
normalized output, instead of clipping represented main KV to ±448. FP8
shared-memory tiles feed both QK and PV with the original FP8 Q, with no BF16
query expansion/scratch or full converted cache. Mixed mode owns only FP32 `[B]`
main-scale scratch; every call/replay refreshes it, including zero/empty
transitions. Computing amax adds a selected-main read; performance is unmeasured.
Sink-free normalized FP32 source O and log2 LSE are jointly merged; the sink is
applied once before public BF16 O / FP32 natural-log LSE publication. Window-only
omits global source states, main-scale storage and main amax work. The internal
`source_state_store` hook is restricted to this nonfinalizing 1CTA direct-index
topology; it does not add a public FP32 native output mode and cannot be combined
with output quantization. The concrete `Fp4Fp8KvProducer` /
`Fp4Fp8SmemKvResource` in `prims_ts_resources.py` publish selected main tiles
through retained FP8 stages, with all-writer AsyncThread-to-UMMA signaling and
per-writer fences. The same file's BF16 `Fp4KvProducer` / `Fp4SmemKvResource`
remain unchanged for compatibility; neither producer contains MXFP8 SWA staging.
`prims_ts_mxfp8_resources.py` adds only the E8M0 SWA conversion, tagged mixed-source
routing and packed launcher, reusing the BF16 FP4 resource's stores and pipeline
through a narrow subclass. The compatibility mode retains BF16 Q/K/V/P, BK64 and
fused split-1 finalization, with its own plan and JIT namespace. Neither
example nor benchmark selects it unless `--prims-ts-mode mxfp8` is supplied
alongside the explicit Prims-TS backend; the mode default is `fp8`.
It does not broaden `BatchSparseMLADecodePagedTSWrapper`: that native wrapper
still accepts matching native BF16 or E4M3 Q/KV, not packed quantized
cache inputs. Its existing toolchain contract is unchanged. See
[`SPARSE.md`](flashinfer/attention/prims_ts/kernels/mla_decode/SPARSE.md).

Current native references gather/mask selected main rows before decoding and
computing FP32 amax/448 and FP32 x/s, then reconstruct `double(E4M3(x/s))*double(s)`
before both logits and V in independent FP64 attention. All native modes,
including main-only, use the unchanged componentwise FP8 probability-rounding
bound, strict sink-exclusive LSE and exact empty gates, not BF16 main-only gates.
The joint KV maximum bounds both FP8 source passes' local maxima, so the existing
summed underflow bound remains conservative. Frost and optional BF16-Q/MXFP8
references, BF16 values and gates are unchanged. Request-scaled conversion and
restoration passed the tested B200 matrix below. The unit-scale FP8-main results
remain historical and do not qualify request scaling. Examples and benchmarks
have not run; no performance or checkpoint/full-model qualification is claimed.

Request-scaled FP8-main GPU validation on 2026-10-18 passed on one allocated B200 /
148 SMs in `umb-b200-237` (job `5103699`, DSL 4.8.0 / bundled compiler 13.4):
isolated capture **1/1**, smoke **32/32**, full **271/271** (157 decode / 48 cache /
66 native sparse MLA) and independent selected dense **57/57**, with no failures,
errors or skips. Both real-stage conversion branches, distinct per-request scales,
FP32-division/rounding and K/V restoration witnesses, zero/empty transitions,
all-width changed-data graph replay and writer/decode integration passed.
Full reported 1641 warnings in 3324.13 seconds; dense reported 980 in 1213.53
seconds. All phase exits were 0; Slurm root/batch/extern completed with `0:0`.
The independent report has `default_gpu_qualification=true`, with no evidence
errors. Exact ordered JUnit identities, complete setup/call/teardown reports,
all 6568 frozen fingerprints, Git state and scheduler visibility matched through
independent terminal corroboration before validation-note edits; the exact tested
patch is preserved. Numerical gates and default GC/capture settings were unchanged.
The root allocation used **1h36m07s** (5767 GPU-seconds), within the fresh
assistant-imposed three-GPU-hour cap. This qualifies only the tested request-scaled
B200/4.8/13.4 snapshot and selected dense subset, not the entire dense suite, other
hardware/toolchains or checkpoint/full-model behavior. Test durations are not
kernel-performance measurements. See the example README for evidence and commands.

Request-scaled FP8-main CPU/static acceptance on 2026-10-18 passed **413/413**
cases (219 source-state/resource/adapter/oracle, 41 optional-MXFP8, 7 conversion
lifetime, 100 benchmark/example-default, 40 native host and 6 unknown-backend
contracts). The fresh `-I -B -S` Python 3.12.3 / Torch 2.11.0+cu128 process used
the existing DSL 4.8.0 overlay / bundled compiler metadata 13.4, static `sm_100`,
thread caps and disabled plugin autoload/JIT. Pytest and strict exits were 0,
with no failures/errors/skips/intercepted CUDA/JIT calls and uninitialized Torch
CUDA. Independent postflight matched exact ordered JUnit identities, complete
setup/call/teardown reports, all 158 fingerprints, Git state and CUDA visibility
before these validation-note edits; the exact tested patch is preserved. The
28-check static compatibility/arithmetic-witness pass and focused Ruff/syntax/
whitespace checks also passed. This qualifies host configuration/plumbing/
reference contracts only, not CuTe lowering, GPU execution or performance.
See the example README for CPU evidence and the separate GPU qualification above.

Historical unit-scale FP8-main CPU/static acceptance on 2026-10-16 passed **321 cases**
(146 source-state/resource/adapter/oracle, 37 optional-MXFP8, 7 conversion-lifetime,
85 benchmark/example-default, 40 native host and 6 unknown-backend contracts).
The fresh `-I -B -S` process used the existing Python 3.12.3 / Torch 2.11.0+cu128 /
DSL 4.8.0 overlay / bundled compiler metadata 13.4, static `sm_100`, thread caps
and disabled plugin autoload/JIT. Pytest/strict exits were 0, with no failures,
errors, skips or intercepted CUDA/JIT calls and uninitialized Torch CUDA.
The latest rerun includes the probe's flat rank-one tuple-layout regression after
the GPU failure below. Exact ordered JUnit identities, all 154 fingerprints, Git
state and CUDA visibility matched through independent postflight, before these
validation-note edits. This guarded Torch/CuTe matrix is host-only, not GPU/JIT
qualification. See the example README for records and runnable GPU commands.

Historical unit-scale FP8-main GPU validation on 2026-10-16 passed on one allocated B200 /
148 SMs in `umb-b200-263` (job `5049128`, DSL 4.8.0 / bundled compiler 13.4):
isolated capture **1/1**, smoke **27/27**, full **271/271** (157 decode / 48 cache /
66 native sparse MLA) and independent selected dense **57/57**, with no failures,
errors or skips. Both real-stage conversion branches and analytic rounding/
saturation/underflow graph replay passed. Separate changed-data replay covered
all global widths `0/64/.../512`. Full reported 1651 warnings in 3502.50 seconds;
dense reported 980 in 1243.21 seconds. Every phase exited 0; Slurm root/batch/extern completed
with `0:0`. Independent corroboration matched exact ordered JUnit identities,
complete setup/call/teardown reports, all 5871 frozen fingerprints, Git state and
scheduler visibility before these validation-note edits. Numerical gates and
default GC/capture settings were unchanged. The independent report has
`default_gpu_qualification=true`, with no evidence errors. The allocation used
1h36m04s of the three-GPU-hour budget. This qualifies only the tested B200/4.8/13.4
unit-scale snapshot and selected dense subset, not request-scaled conversion/
restoration, the entire dense suite, other hardware/toolchains or checkpoint/
full-model behavior. Examples and benchmarks have not
run; no performance claim is made. See the example README for records.

The earlier allocated B200 attempt on `umb-b200-263` (job `5048418`, DSL 4.8.0 / bundled
compiler 13.4) passed the exact 271-case collection and isolated legacy capture
1/1, then failed the first new FP4-to-FP8 byte probe during JIT: a scalar CuTe
layout shape reached `cutlass.make_array_view`, raising `TypeError: object of
type 'int' has no len()`. No conversion kernel launched or accuracy check ran;
smoke reported 1 failed, pytest exited 1 and Slurm recorded `FAILED` / `1:0`.
Full and dense phases were skipped by the smoke prerequisite gate. All 5772
frozen fingerprints and Git state matched through independent terminal
corroboration, with recorded default GC/capture settings and visibility unchanged.
The probe now uses a flat one-element shape tuple; arithmetic, byte expectations
and gates are unchanged. The corrected probe and FP8-main attention subsequently
passed job `5049128` above; the earlier failed evidence is unchanged. Earlier
jobs `5047536` and `5047995` executed no cases because
of wrapper-only maxfail/forecast defects. See the example README for all records.

**Historical validation (pre-FP8-main):** the 257-case GPU and 299-case CPU results
below do not qualify current main conversion/staging/attention. The native FP8
snapshots used BF16 main attention and exact BF16 query expansion.
Native-first reorganization acceptance on 2026-10-15 passed **213 CPU cases**
(127 source-state/adapter/oracle, 6 conversion-lifetime, 34 stdlib-only
benchmark/example, 40 native host and 6 unknown-backend contracts) with the
existing DSL 4.8 overlay and strict guards. No CUDA/JIT calls were intercepted;
CUDA visibility and all 114 fingerprints stayed unchanged during the run, before
these validation-note edits. The optional-MXFP8 follow-up passed **299 CPU cases**,
including 37 new mode-isolation/resource contracts and 84 benchmark/example
contracts, with strict aggregate exit 0, no intercepted calls, uninitialized
Torch CUDA, unchanged visibility and all 119 fingerprints matching postflight.
See the example README for the exact case breakdown and records. No GPU
execution or benchmark ran for this split. The historical validation records
below describe their original pre-reorganization snapshots; they do not qualify
the rebuilt commits.

Decode tests are in `tests/experimental/test_deepseek_v41_decode.py`; cache-writer
integration tests are in `tests/experimental/test_deepseek_v41_cache.py`, with guarded
source-store/adapter/oracle host contracts in
`tests/experimental/test_deepseek_v41_source_state_cpu.py` (`--confcutdir=tests/experimental`).
Fresh guarded CPU processes use a static `CUTE_DSL_ARCH=sm_100` declaration,
thread caps and disabled plugin autoload/JIT without changing CUDA visibility.
The new 125-case suite passed with DSL 4.8.0 / bundled compiler metadata 13.4,
with no intercepted CUDA/JIT calls and strict aggregate exit 0. The tested
4.7.0+flashinfer.5 distribution / compiler metadata 13.3 also passed all 125
pytest cases, but eager environment-manager default evaluation attempted two
import-time `cuInit` calls despite the static target. Guards blocked both before
execution; strict aggregate exit was 1. The existing host40/API6 matrix has the
same 4.8-clean / 4.7-intercepted distinction. The fixture now guards raw driver/
runtime calls before CuTe imports and rejects caught attempts; its fresh 4.7
control reports a setup error. Earlier Torch/CuTe-only guard records did not
prove raw-driver isolation. Exact cases, source fingerprints and visibility
matched within each matrix; packages were not modified. See the example README
for warnings, exits and provenance. These checks do not qualify GPU execution
or change the native wrapper's DSL 4.7 contract.
The post-repair default run on `umb-b200-260` (job `5008381`, B200 / DSL 4.8.0 /
bundled CUDA 13.4) passed the complete expanded **257/257** matrix, including all
50 new ordinary-FP8 cases, and the independent selected dense **57/57** subset.
The first allocated B200 smoke (`umb-b200-248`, job `4943022`, DSL 4.8.0 /
bundled CUDA 13.4) passed preflight and collected the exact expanded 257-case
manifest, including all 50 new cases and the unchanged legacy 207. Its first smoke
case failed during JIT lowering because `cutlass.Array` does not support `len()`:
1 failed, pytest exit 1, Slurm `FAILED` / `1:0`; no accuracy or later matrices ran.
All 134 fingerprints and Slurm CUDA visibility were preserved. The source store
now reads its static width from `values.shape[0]`.
The corrected retry on one allocated B200 / 148 SMs (`umb-b200-263`, job `4943256`,
DSL 4.8.0 / bundled CUDA 13.4) passed the exact 16-case smoke. Its exact 257-case
full run completed with decode **142 passed / 1 failed**, cache **48 passed**,
and native sparse MLA **66 passed**: **256 passed / 1 failed**, with 0 errors or
skips and 1721 warnings in 3686.71 seconds. All 50 new cases passed (31 decode,
19 cache). Pytest exited 1 and Slurm recorded `FAILED` / `1:0`; exact JUnit
identities, all 134 fingerprints, Git state and Slurm CUDA visibility matched
through full and final postflight, before these validation-note updates.
The only failure was the unchanged legacy Frost
`test_e8m0_scale_boundary_changed_graph[None-512-0-cute-dsl]`, during prepared
graph capture with `cudaErrorStreamCaptureInvalidated` (901), before numerical
checks. That wrapper stopped before the selected dense 57-case phase.
A second default run (`umb-b200-263`, job `5000428`, the same toolchain) again
completed **256 passed / 1 failed**, with all 50 new cases passing; isolated
capture and smoke passed, but full-suite capture failed at the same case.
A diagnostic follow-up on `umb-b200-236` (job `5005300`) passed default isolated
capture **1/1**, smoke **16/16** and the independent default dense subset **57/57**.
Its instrumented full-order prefix stopped at **93 passed / 1 failed** in case 94;
this was not a default full257 run. The driver trace records `cuLibraryUnload`
returning 900 inside a GC interval during capture, before launch/end-capture 901.
The same library handle was loaded in the preceding standalone
`test_native_fp4_all_finite_codes` conversion probe. Runtime unload return type,
complete caller ownership and exhaustive producer coverage remain unproven;
unknown returns and truncated/short records limit the trace.
The two standalone conversion probes now explicitly unload their CuTe modules
at test teardown, instead of leaving cyclic executors for GC in a later capture.
This is a test-resource lifetime repair, not a decoder/JIT-core arithmetic change.
Six new CPU-only ownership regressions passed with strict exit 0, no accelerator
imports and no intercepted calls. Numerical references/gates and default GC/capture
settings are unchanged. The 5005300 dense phase had 0 failures/errors/skips;
all 185 fingerprints, Git state and Slurm visibility matched through independent
terminal corroboration before this repair. The job itself remains `FAILED` / `1:0`
from its diagnostic failure.
The post-repair run on `umb-b200-260` (job `5008381`, DSL 4.8.0 / bundled CUDA
13.4) passed default isolated capture **1/1**, smoke **16/16**, full **257/257**
(decode **143**, cache **48**, native sparse MLA **66**) and independent dense
**57/57**, with 0 failures, errors or skips. All 50 new cases passed; the former
case-94 Frost capture failure passed setup/call/teardown in full-suite order.
Full reported 1707 warnings in 3552.03 seconds; dense reported 980 in 1105.53
seconds. All phase exits were 0; Slurm root/batch/extern completed with `0:0`.
Exact ordered JUnit identities and all 5047 frozen fingerprints, Git state and
Slurm CUDA visibility matched through final postflight and independent terminal
corroboration, before these validation-note edits. No matrix sources, references,
tolerances or default GC/capture settings changed during execution.
The original frozen auditor's false verdict is retained: its four command-record
errors compared observer pre-parse argv with diagnostic argv after pytest's
in-place ini-addopts prepend. A separate post-run checker accepts only the exact
canonical phase argv, literal `--import-mode=importlib` ini prepend and complete
interpreter/module command; every other original check reran unchanged. Its
166 guarded CPU positive/negative checks passed without accelerator imports or
intercepted calls. The new independent report has `default_qualification=true`,
no evidence errors and no nonpassing phases; neither frozen inputs nor the
original report was rewritten. See the README for report paths and coverage
limits. This successful regression does not establish exhaustive JIT ownership
or historical runtime-unload return-status attribution.
Those pre-FP8-main tests included all global widths 0/64/.../512, source precision,
exact BF16 query expansion and changed-data graph replay. Their independent FP64
oracle used the native per-component FP8 rounding bound, strict sink-exclusive
LSE and exact empty gates; global-only then retained strict BF16 checks. Current
native main-only uses the E4M3 reference/bound instead. Legacy helpers/tolerances
are unchanged. The following 207/57-case B200 results additionally predate ordinary
FP8 SWA and do not qualify current FP8-main staging/attention.
Historical writer/decode integration covers both backends, window-only and global
widths 128/192/512, with CUDA Graph replay and exact encoded-byte references;
all eight expanded legacy integration cases passed in the full B200 run below.
The multi-backend benchmark is `benchmarks/bench_deepseek_v41_decode.py`.
Production-source B200 validation on `umbriel-b200-040` with CuTe-DSL 4.8.0 /
bundled CUDA 13.4 and a fresh scratch workspace passed all 10 targeted tests,
with no failures or skips:
selection refresh and zero-query tile counts at global widths 0/64/192/512,
plus two checks that Prims-TS does not compile Frost. Existing independent FP64
output/LSE gates are unchanged, including poisoned-selection CUDA Graph replay.
The earlier global192 failure was traced to a zero task-cache length despite a
live span of 384; the getter now snapshots current Python state and the
production regression passes. Five native automatic-dispatch regressions also
passed on the same node/toolchain, with no failures, errors or skips. The separate
`tests/attention/test_prims_ts_sparse_mla.py::test_keep_ab_runtime_task_cache`
now pins native BF16 Keep-AB for static and CLC persistence, asserts actual compile
configurations and a 768-slot runtime domain, and reuses the existing FP64 oracle
and changed-metadata graph replay. Both cases ran in the earlier full matrix: CLC
passed, while static failed during plan-time FrontendNext compilation.
That `--full` run on `umb-b200-250` (Slurm job `4907814`) with CuTe-DSL 4.8.0 /
bundled CUDA 13.4 completed all 207 cases: decode **112 passed**, cache **29 passed**,
and native sparse MLA **35 passed / 31 failed**, with no pytest errors or skips.
All 31 native failures occurred during plan-time FrontendNext/task-scheduling
compilation, before eager accuracy or graph replay checks.
Narrow Task-interface, work-tile type, G1 temporary-scope and task-body
activity-guard repairs are implemented. All 40 CPU-only acceptance cases in
`tests/attention/test_attention_ts_mla_decode.py` historically passed separately
with DSL 4.7.0 and 4.8.0, with no failures, errors or skips; the recorded Torch/CuTe
guards were not triggered. That scope did not cover raw CUDA bindings; the current
stricter matrix above rejects the intercepted 4.7 imports despite pytest passes.
These host checks do not qualify DSL 4.7 GPU execution; the separate dense
D576 GPU suite was not part of the 207-case matrix.
A post-repair smoke run on `umb-b200-236` (job `4909337`)
with the same toolchain completed 40 cases: **39 passed / 1 failed**, with no errors
or skips and unchanged source fingerprints. Both pinned static/CLC regressions
and all eight selected packed selection/cache replay cases passed. The remaining
native FP8 `small-grid-wide-kv` case failed at plan time with `SCOPE_READ_NEVER_SET`
in the fused 2CTA parallel reducer. Its warp-only statistics/output locals are now
separated without arithmetic changes.
The follow-on three-case GPU smoke and complete 207-case rerun on `umb-b200-041`
(job `4910237`), with CuTe-DSL 4.8.0 / bundled CUDA 13.4, **passed**: decode **112**,
cache **29**, native sparse MLA **66**, including both pinned static/CLC regressions.
There were **0 failures, errors or skips**, with 1679 warnings in 3427.86 seconds;
pytest exited 0 and Slurm completed with exit `0:0`. The exact 207-case node set
and all 111 source fingerprints matched through final postflight, before these
validation-note updates; kernels and accuracy gates were unchanged during testing.
A separate selected dense D576 QK / D512 V subset in
`tests/attention/test_attention_ts_mla_decode.py` passed **57/57** on one allocated
B200 in `umb-b200-250` (job `4936630`), with DSL 4.8.0 / bundled CUDA 13.4.
It covers BF16/FP8 1CTA/2CTA, runtime-K pruning, fully masked splits,
compact/empty/causal queries, CLC progress and live-metadata CUDA Graph replay.
There were no failures, errors or skips, with 980 warnings in 1063.65 seconds;
pytest exited 0 and Slurm completed with exit `0:0`. All three preceding smoke
cases passed and belong to the same 57-case manifest. Exact case identities and
all 111 source fingerprints matched through final postflight, before validation
notes changed; GPU visibility and independent accuracy gates were unchanged.
These runs qualify only their tested pre-FP8-main B200/4.8/13.4 snapshots and
selected dense subset, not current native main conversion/attention, the entire
dense GPU suite, DSL 4.7 GPU execution, other GPU/toolchain combinations or
checkpoint/full-model behavior.
A follow-up static implementation review found no confirmed correctness defects
in the reviewed adapter/shared-kernel/API/test-harness scope; this is not new GPU
qualification. Six unknown-backend CPU cases passed separately with each of DSL
4.7.0 and 4.8.0 without CUDA/JIT work. The README distinguishes the tested packed
widths from supported intermediate widths without end-to-end coverage, exhaustive
helper tests from direct packed staging, and forced fallback on CUDA 13.4 from
actual older-compiler execution. Review edits changed only docs/docstrings;
kernels, tests, references and numerical gates are unchanged.
Benchmarks have not run; no measurements or measured crossovers are claimed. See
[`examples/deepseek_v41/README.md`](examples/deepseek_v41/README.md) for the current
validation status, runnable example and commands. Both backends remain JIT-only
with no AOT or trace registration.

## Key Architectural Patterns

### Module Caching

FlashInfer uses two-level caching to avoid recompilation:

1. **Python-level** (`@functools.cache`): In-memory cache of loaded modules
2. **File-level** (`~/.cache/flashinfer/`): Compiled `.so` files on disk

**Cache invalidation** (automatic):
- Source file changes (SHA256 hash)
- Compilation flags change
- CUDA architecture change
- FlashInfer version change

URI computed as: `hash(operation_type + parameters + source_hashes + flags + cuda_arch)`

**Cache management:**
- Clear cache: `rm -rf ~/.cache/flashinfer/`
- Override location: `export FLASHINFER_WORKSPACE_BASE="/scratch"`

**CuTe-DSL kernels** (`flashinfer/jit/cute_dsl_core.py`): `cute.compile()` has no
persistent cache, so `JitSpecCuteDsl` (a `JitSpec` subclass, wrapped by the
`build_and_load_cute_dsl_kernel()` helper) exports compiled kernels as
object files (`export_to_c()`) and reloads them with `cute.runtime.load_module(...,
enable_tvm_ffi=True)`. Artifacts live in `cached_ops/` next to the nvcc modules, one
directory per op family — `cached_ops/<module>_<arch>_cute_dsl/` (e.g.
`nvfp4_quantize_sm100a_cute_dsl/`) holding one `meta.json` plus one `.o` per
specialization. The arch comes from the DSL's compile target (`CUTE_DSL_ARCH` or the
current device) since the artifacts are single-arch, unlike nvcc fatbins.
Invalidation is module-granular: a changed nvidia-cutlass-dsl version or
kernel-source SHA256 wipes and lazily rebuilds the module. Reference usage:
`flashinfer/quantization/kernels/nvfp4_quantize.py`. Disable with
`FLASHINFER_CUTE_DSL_DISABLE_CACHE=1`.

### Dispatch Macros

Handle combinatorial parameter spaces:

```cpp
DISPATCH_DTYPE(input_dtype, DTypeIn, {
  DISPATCH_DTYPE(output_dtype, DTypeOut, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      LaunchKernel<DTypeIn, DTypeOut, BLOCK_SIZE>(...);
    });
  });
});
```

Defined in `.jinja` files and expanded after rendering.

## API Logging with @flashinfer_api

FlashInfer provides the `@flashinfer_api` decorator for debugging API calls.

**Key features:**
- **Crash-safe**: Logs inputs BEFORE execution (preserves info even if kernel crashes)
- **Zero overhead when disabled**: `FLASHINFER_LOGLEVEL=0` (default)
- **Multiple verbosity levels**: 0 (off), 1 (names), 3 (inputs/outputs), 5 (+ statistics)
- **CUDA graph compatible**: Auto-skips stats during graph capture

**Quick usage:**
```bash
# Enable detailed logging
export FLASHINFER_LOGLEVEL=3              # 0, 1, 3, or 5
export FLASHINFER_LOGDEST=debug.log       # stdout, stderr, or file path

python my_script.py
```

**Why use this?**
- Debug CUDA crashes (see inputs that caused crash)
- Track tensor shapes/dtypes through pipeline
- Detect NaN/Inf issues (level 5)

→ **For complete debugging guide, see [`.claude/skills/debug-cuda-crash/skill.md`](.claude/skills/debug-cuda-crash/skill.md)**

## Debugging

### Enable Logging

```bash
export FLASHINFER_JIT_VERBOSE=1      # Verbose JIT output
export FLASHINFER_JIT_DEBUG=1        # Debug symbols, -O0
export FLASHINFER_LOGLEVEL=3         # API logging (0=off, 1=basic, 3=detailed)
export FLASHINFER_LOGDEST=stdout
```

### Inspect Generated Code

```bash
# Generated sources
ls -la ~/.cache/flashinfer/0.6.0/*/generated/

# Compiled modules
ls -la ~/.cache/flashinfer/0.6.0/*/cached_ops/

# Build files
cat ~/.cache/flashinfer/0.6.0/*/cached_ops/*/build.ninja
```

### Environment Variables

```bash
# Compilation
export FLASHINFER_NVCC_THREADS=4              # Threads per nvcc process (--threads=N)
export MAX_JOBS=4                             # Parallel ninja jobs (nvcc processes)
export FLASHINFER_CUDA_ARCH_LIST="8.0 9.0a"  # Target architectures
# Memory note: total compilation memory ≈ MAX_JOBS × FLASHINFER_NVCC_THREADS × per-thread mem.

# Behavior
export FLASHINFER_WORKSPACE_BASE="/scratch"   # Custom cache directory
```

#### Full Environment Variable Reference

Already covered above (Quick Reference / Debugging): `FLASHINFER_LOGLEVEL`,
`FLASHINFER_LOGDEST`, `FLASHINFER_JIT_VERBOSE`, `FLASHINFER_JIT_DEBUG`,
`FLASHINFER_CUDA_ARCH_LIST`, `FLASHINFER_NVCC_THREADS`,
`FLASHINFER_WORKSPACE_BASE`, `MAX_JOBS`.

The remaining `FLASHINFER_*` knobs read by the code are listed below. Defaults
match what the code uses today; values are strings unless noted.

##### JIT / Build Toolchain

| Variable | Default | Read in | Effect |
|----------|---------|---------|--------|
| `FLASHINFER_DISABLE_JIT` | unset | `flashinfer/jit/core.py` | If set (any non-empty value), JIT compilation is refused and modules must already exist in the cache or be provided via AOT packages. |
| `FLASHINFER_CUTE_DSL_DISABLE_CACHE` | `0` | `flashinfer/jit/cute_dsl_core.py` | `1` disables the on-disk cache for JIT-compiled CuTe-DSL kernels (every process recompiles via `cute.compile`). |
| `FLASHINFER_DISABLE_VERSION_CHECK` | unset | `flashinfer/jit/env.py` | Skip the AOT/JIT-cache version check that pins flashinfer-jit-cache to the installed flashinfer-python. Bypass only when you intentionally mix versions. |
| `FLASHINFER_JIT_LINEINFO` | `0` | `flashinfer/jit/core.py` | `1` adds `-lineinfo` to nvcc so profiler / `cuda-gdb` can map PTX back to CUDA source. |
| `FLASHINFER_JIT_PREBUILD_MAX_JOBS` | unset (uses `MAX_JOBS`) | `flashinfer/jit/core.py` | Maximum parallel Ninja jobs for a bulk `build_jit_specs()` prebuild. This does not control ordinary on-demand module builds. The sharded test runner sets it to the host-wide automatic build budget before reducing each worker's `MAX_JOBS`; an explicit value is preserved. |
| `FLASHINFER_NVCC` | `$cuda_home/bin/nvcc` | `flashinfer/jit/cpp_ext.py` | Override the nvcc binary used by the JIT (useful for sccache wrappers or non-default CUDA installs). |
| `FLASHINFER_NVCC_LAUNCHER` | `""` | `flashinfer/jit/cpp_ext.py` | Optional launcher prefix for nvcc (e.g. `ccache`, `sccache`). Combined with `FLASHINFER_NVCC`. |
| `FLASHINFER_CXX_LAUNCHER` | `""` | `flashinfer/jit/cpp_ext.py` | Same idea as `FLASHINFER_NVCC_LAUNCHER` but for the host C++ compiler. |
| `FLASHINFER_FMHA_V2_VERBOSE` | unset | `flashinfer/jit/attention/fmha_v2/fmha_library.py` (FMHA v2 codegen, C++ side) | When set, the FMHA-v2 codegen / runtime prints verbose dispatcher diagnostics. Leave unset for normal runs. |
| `FLASHINFER_EXTRA_CFLAGS` | unset | `flashinfer/jit/cpp_ext.py` | Extra compiler flags passed to the host C++ compiler. |
| `FLASHINFER_EXTRA_CUDAFLAGS` | unset | `flashinfer/jit/cpp_ext.py` | Extra compiler flags passed to `nvcc`. |
| `FLASHINFER_EXTRA_LDFLAGS` | unset | `flashinfer/jit/cpp_ext.py` | Extra linker flags passed to the linker. |
| `FLASHINFER_JIT_CACHE_PROVIDER_ARCHS` | required | `flashinfer-jit-cache/build_backend.py` | Space-separated provider architectures added to a shim wheel's exact `Requires-Dist` metadata. |
| `FLASHINFER_JIT_CACHE_PROVIDER_ARCH` | required | `flashinfer-jit-cache-provider/package_config.py` | Select exactly one architecture, such as `9.0a` or `sm120f`, when building a binary provider wheel. |

##### Cubin / Artifact Loader

| Variable | Default | Read in | Effect |
|----------|---------|---------|--------|
| `FLASHINFER_CUBIN_DIR` | `<FLASHINFER_WORKSPACE_BASE>/.cache/flashinfer/cubins` | `flashinfer/jit/env.py` (exposed via `flashinfer/__main__.py show-config`) | Local directory used to cache downloaded cubins. Override to share a cache between users. |
| `FLASHINFER_CUBINS_REPOSITORY` | `https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local` | `flashinfer/jit/cubin_loader.py` | Base URL the loader downloads cubins from. Point to a mirror for offline or air-gapped setups. |
| `FLASHINFER_CUBIN_CHECKSUM_DISABLED` | unset | `flashinfer/jit/cubin_loader.py` | If set, skip SHA checksum verification of downloaded cubins. Debug aid only. |
| `FLASHINFER_CUBIN_DOWNLOAD_THREADS` | `4` | `flashinfer/artifacts.py` | Thread-pool size used by `flashinfer artifacts download`. |
| `FLASHINFER_NO_DOWNLOAD` | unset | `flashinfer/jit/cubin_loader.py` | Hard-fail if a cubin is missing locally instead of attempting to download. Useful in CI / locked-down environments. |
| `FLASHINFER_DSL_FMHA_LOCAL_DIR` | unset | `flashinfer/attention/cute_dsl/fmha.py` | Path to a local checkout of the CuTe-DSL FMHA kernel sources. The loader checks here before downloading. |
| `FLASHINFER_LOGGING_LEVEL` | `INFO` | `flashinfer/artifacts.py`, `flashinfer/jit/core.py` | Python logging level for the artifacts/cubin loader and the JIT compiler (`DEBUG`/`INFO`/`WARNING`/`ERROR`). Distinct from `FLASHINFER_LOGLEVEL`. |
| `FLASHINFER_DISABLE_TINYGEMM2_SM100` | `0` | `flashinfer/gemm/routergemm.py` | Set to `1` to disable the generated SM100/SM103 `tinygemm2` backend and force the dispatcher to fall back to the legacy implementation. Useful as an escape hatch when debugging backend-selection or kernel issues on Blackwell systems. |

##### API Dump / Logging Extensions

These complement `FLASHINFER_LOGLEVEL`/`FLASHINFER_LOGDEST` and are all read in `flashinfer/api_logging.py`.

| Variable | Default | Effect |
|----------|---------|--------|
| `FLASHINFER_DUMP_DIR` | `flashinfer_dumps` | Directory where `@flashinfer_api` writes per-call tensor dumps when logging is enabled. |
| `FLASHINFER_DUMP_INCLUDE` | `""` | Comma-separated allow-list of API names; only matching calls are dumped. Wildcards (`*`) supported. |
| `FLASHINFER_DUMP_EXCLUDE` | `""` | Comma-separated deny-list of API names; matching calls skip the (expensive) stats / dump path. |
| `FLASHINFER_DUMP_MAX_COUNT` | `1000` | Hard cap on number of dumped events; once reached, further dumps are dropped with a warning. |
| `FLASHINFER_DUMP_MAX_SIZE_GB` | `20` | Hard cap on total dump size in **gigabytes** (GB) written to `FLASHINFER_DUMP_DIR` (parsed as `float`; see `_DUMP_MAX_SIZE_GB` in `flashinfer/api_logging.py`). |
| `FLASHINFER_DUMP_SAFETENSORS` | `0` | `1` writes dumps as `.safetensors` (portable / Hugging Face-style); otherwise plain `.pt`. |

##### Trace Capture

Used by `flashinfer.trace` / `fi_trace`.

| Variable | Default | Read in | Effect |
|----------|---------|---------|--------|
| `FLASHINFER_TRACE_DUMP` | unset | `flashinfer/fi_trace.py` | If set, decorated APIs auto-dump benchmark-definition JSON for each call (the "fi_trace" feature). |
| `FLASHINFER_TRACE_DUMP_DIR` | cwd | `flashinfer/fi_trace.py` | Directory where the trace JSON files are written. |
| `FLASHINFER_TRACE_APPLY` | `0` | `flashinfer/__init__.py`, `flashinfer/trace_apply/config.py` | Set to `1` to enable Trace Apply (runtime kernel substitution) when FlashInfer is imported. |
| `FLASHINFER_TRACE_APPLY_PATH` | unset | `flashinfer/trace_apply/config.py` | Directory from which deployment-configured solutions are loaded for Trace Apply. |

##### Validation / Autotuning / Routing / Kernel Selection

| Variable | Default | Read in | Effect |
|----------|---------|---------|--------|
| `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` | unset | `flashinfer/utils.py` (`backend_requirement`), `flashinfer/api_logging.py` | `1` lets `backend="auto"` (dispatch heuristics and autotuning) select backends marked `@experimental_backend` (see the "Experimental Code" section). Explicit use never needs it: calling an `@flashinfer_experimental_api` function or passing `backend="<experimental>"` always works and emits an `ExperimentalWarning` once. Without it, automatic selection considers only stable backends. |
| `FLASHINFER_VALIDATE_INPUTS` | `0` | `flashinfer/mla/_core.py` (MLA wrapper) | Non-zero / non-empty value enables defensive input validation inside the MLA wrapper. Device-synchronizing checks are skipped during CUDA Graph capture. Adds host-side overhead; intended for debugging. |
| `FLASHINFER_AUTOTUNER_LOAD_FROM_FILE` | `0` | `flashinfer/autotuner/autotuner.py` | `1` loads previously serialized autotune results from disk instead of re-running the search. |
| `FLASHINFER_DIST_AWARE_AUTOTUNE` | `0` | `flashinfer/fused_moe/da_config.py` | `1` enables experimental distribution-aware autotune and kernel dispatch (TRT-LLM MoE only). |
| `FLASHINFER_DA_DISTRIBUTIONS` | built-in distribution catalog | `flashinfer/fused_moe/da_config.py` | Comma-separated training distributions used by the experimental TRT-LLM distribution-aware MoE autotuner. |
| `FLASHINFER_PRIMS_TS_DEBUG_CHECKS` | `0` | `flashinfer/prims_ts/batched_gemm/batched_gemm_kernel.py`, `batched_gemm_run.py` | `1`/`true`/`yes`/`on` enables exhaustive deadlock/race checks and TaskManager verification in Prims-TS batched GEMM. |
| `FLASHINFER_PRIMS_TS_COMPILE_OPTIONS` | `--opt-level 2` | `flashinfer/prims_ts/utils.py`, `batched_gemm_run.py` | Extra CuTe-DSL / nvc compile options passed when building Prims-TS kernels. |
| `FLASHINFER_PRIMS_TS_BF16_CONSTANT_WEIGHTS` | unset | `flashinfer/prims_ts/batched_gemm/batched_gemm_run.py` | `1` fills BF16 autotune/cast-A weight tensors with constant `1.0` (debug / deterministic fill). |
| `FLASHINFER_PRIMS_TS_CASTA_CONSTANT_ACT` | unset | `flashinfer/prims_ts/batched_gemm/batched_gemm_run.py` | `1` fills cast-A / BF16 activation tensors with constant `1.0` during autotune buffer setup. |
| `FLASHINFER_PRIMS_TS_CASTA_PATTERN_ACT` | unset | `flashinfer/prims_ts/batched_gemm/batched_gemm_run.py` | `1` fills activations with a quarter-wise pattern `(1,2,4,6)` instead of random data. |
| `FLASHINFER_PRIMS_TS_CASTA_PATTERN_WEIGHTS` | unset | `flashinfer/prims_ts/batched_gemm/batched_gemm_run.py` | `1` fills cast-A weights with a deterministic pattern (takes precedence over constant fill). |
| `FLASHINFER_PRIMS_TS_CASTA_CONSTANT_WEIGHTS` | unset | `flashinfer/prims_ts/batched_gemm/batched_gemm_run.py` | `1` fills cast-A weights with constant values when pattern fill is not set. |
| `FLASHINFER_AUTOTUNE_DIR` | unset | `flashinfer/mla/_sparse_mla_sm120/_calibration.py`, `flashinfer/comm/pcie_ipc_tuning.py` | Override the disk path for tuning cache files (sparse-MLA SM120 cpb calibration constants, and the PCIe IPC all-reduce). Falls back to `FLASHINFER_WORKSPACE_DIR` when unset. |
| `FLASHINFER_AUTOTUNE_TIMER` | unset (auto) | `flashinfer/autotuner/autotuner.py` | Selects the autotuner's per-tactic timer: `globaltimer` forces the GPU `%globaltimer` register, `cuda_event` forces `cudaEvent`, unset/anything-else auto-detects (uses `%globaltimer` only when Confidential Computing is detected). Under CC `cudaEventElapsedTime` is unreliable (can go negative), so the globaltimer path keeps tactic ranking stable. |
| `FLASHINFER_CUTILE_AUTOTUNE_DISABLED` | `0` | `flashinfer/quantization/kernels/cutile/rope_quantize_fp8_cutile.py` | Non-zero skips exhaustive cuTile RoPE-FP8 tuning and uses the built-in token-count heuristic. |
| `FLASHINFER_CONFIDENTIAL_COMPUTE` | unset | `flashinfer/utils.py` | Override NVIDIA Confidential Computing (CC) auto-detection used by `is_confidential_compute()` (which drives the autotuner timer above): `1` forces CC, `0` forces non-CC. Useful for CI or hosts without `pynvml`. |
| `FLASHINFER_MSA_PREFILL_SCHEDULE` | unset | `flashinfer/msa_ops/_blackwell_sm100.py` | Set to `m64` to force the eligible M64 Blackwell MSA prefill schedule; any other non-empty value is rejected. Leave unset for automatic routing. |
| `FLASHINFER_MSA_FP8_Q1_SCHEDULE` | unset | `flashinfer/msa_ops/_blackwell_sm100.py` | Force an eligible FP8 Q1 MSA decode route: `batch_attention`, `q1_exact`, `q1_flat_xform2`, `q1_paged_xform2`, or `paged_uniform_fp8`. Leave unset for automatic routing. |
| `FLASHINFER_AUTOTUNE_CACHE_DIR` | `FLASHINFER_CACHE_DIR/autotune` | `flashinfer/autotune_cache.py` | Root **directory** of the managed v2 autotune store used by `autotune_v2()` (placement only; distinct from the MLA-specific `FLASHINFER_AUTOTUNE_DIR` above). |
| `FLASHINFER_RAGGED_AUTO_BACKEND_ORDER` | unset | `flashinfer/prefill.py` | Comma-separated backend order for `BatchPrefillWithRaggedKVCacheWrapper` under `backend="auto"` on Blackwell (e.g. `cutlass,cudnn`, or a single name to pin one). Overrides both the per-shape table and the global default; eligibility is still enforced, so a backend that cannot serve the problem is skipped rather than forced. For benchmarking and regression bisection on hardware whose ranking differs from the measured B200 defaults. |
| `FLASHINFER_NVFP4_QUANTIZE_USE_TMA` | unset | `flashinfer/quantization/kernels/nvfp4_quantize.py` | Overrides the NVFP4 CuTe-DSL TMA-kernel heuristic: `0` forces the vectorized-load kernel, `1` enables the shape heuristic on every architecture and input dtype. Unset, TMA is used only where it was measured to win (SM107 with FP16/BF16 input, layout-specific M/K crossovers). For benchmarking and re-tuning on new hardware. |
| `FLASHINFER_TOPK_ALGO` | unset | `flashinfer/topk.py` | Force a specific top-k backend (otherwise the dispatcher chooses based on shape/dtype/mode via benefit gates): `default` (radix), `clusters` (SM100), `cub` (cub::DeviceBatchedTopK; bypasses the benefit gates). Used for benchmarking / regression bisection. |
| `FLASHINFER_USE_CUDA_NORM` | `0` | `flashinfer/norm/__init__.py` | `1` switches the norm path from the default backend to the legacy CUDA-only kernels. Diagnostic toggle. |
| `FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN` | unset | `csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cuh` | Forces the TRT-LLM MoE custom-routing kernel into "one-block-per-token" mode regardless of the active routing policy. Mainly used to reproduce specific perf points. |
| `FLASHINFER_B12X_MICRO_SHARE_INPUT` | `1` | `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` | `0` disables the B12x MoE micro-batch input-sharing optimization. Internal/experimental — leave at the default unless investigating an SM12x MoE regression. |
| `FLASHINFER_B12X_FORCE_MOE_W4A16` | unset | `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` | When set (any non-empty value), forces the SM12x MoE dispatcher onto the W4A16 kernel path regardless of weight dtype. Internal/experimental — used to reproduce W4A16-specific issues. |
| `FLASHINFER_TACTICS_BLOCKLIST` | unset | `flashinfer/autotuner/autotuner.py` | Path to a JSON tactics-blocklist file generated by `flashinfer tactics-blocklist generate` (or `python -m flashinfer tactics-blocklist generate`). When set, the autotuner loads the file at startup and skips any kernel tactics listed as invalid for the current GPU/driver environment, preventing hang or crash on known-bad tactics. |
| `FLASHINFER_CUDNN_PREFILL_SHAPE_OVERRIDE` | `1` | `flashinfer/cudnn/prefill.py` (ragged token-indptr path) | `0` makes the cuDNN ragged prefill graph declare the exact `(batch, max_seq_len)` per call instead of building one override-enabled graph per length class (declared max_len 128 or 65536+, batch 4096+) and passing the real shapes at execute time via cuDNN's shape override. Override exists because cuDNN finalizes one execution plan per declared shape (55-70 ms, ~1 s with a kernel compile) and serving changes the shape every step; needs cuDNN 9.22+ and cudnn-frontend 1.29+, otherwise the exact declaration is used. Disable only to bisect a suspected override regression. |
| `FLASHINFER_ALLOW_CUDNN_PLAN_BUILD_IN_CAPTURE` | `0` | `flashinfer/gemm/gemm_base.py` (read once at import) | `1` downgrades the refusal to build a cuDNN execution plan during CUDA graph capture from a `CudnnPlanBuildInCaptureError` to a warning. Set it only if you have verified your cuDNN version tolerates plan build under capture; the supported alternative is to run each shape once eagerly outside the capture region, which populates the per-shape plan cache the captured call then reuses. |

##### Experimental KDA Output-Only Decode Tuning

Low-level overrides for the output-only / RecoverSSM-verify KDA decode
kernels (``flashinfer/kda_kernels/kda_decode_wy_output_only.py``); intended
for benchmarking and bring-up.

| Variable | Default | Effect |
|----------|---------|--------|
| `FLASHINFER_KDA_OO_REC_MAX_BH` | `192` | WY-vs-recurrent auto-dispatch threshold on `B * HV` at `T <= 2` (a quarter of it at `T` in (2, 4]). |
| `FLASHINFER_KDA_OO_REC_KS` | batch policy | Override the grouped recurrent backend's K-slices per state column. |
| `FLASHINFER_KDA_OO_REC_VSPLIT` | batch policy | Override the grouped recurrent backend's value-column split per CTA. |
| `FLASHINFER_KDA_OO_MBP` | grid policy | Override the WY kernel's minimum-blocks-per-SM launch-bound hint. |

##### Experimental Packed-KDA Decode Tuning

These low-level overrides are for benchmarking and kernel bring-up. Production
users should normally leave the batch-size policy at its defaults.

| Variable | Default | Effect |
|----------|---------|--------|
| `FLASHINFER_KDA_T1_FAST_PATH` | `1` | `0` disables routing eligible recurrent KDA T=1 calls to the packed CuTe-DSL decode kernel. |
| `FLASHINFER_PACKED_KDA_TILE_V` | batch policy | Override the V-dimension tile width with a supported packed-KDA tile size. |
| `FLASHINFER_PACKED_KDA_ILP` | batch policy | Override the number of rows processed per instruction-level-parallel group. |
| `FLASHINFER_PACKED_KDA_THREADS` | batch policy | Override CTA threads; the value divided by 16 must select a valid group count. |
| `FLASHINFER_PACKED_KDA_EVICT_FIRST` | batch policy | Set to `1` or `0` to override the policy's first-use cache-eviction hint. |
| `FLASHINFER_PACKED_KDA_CHUNKR` | policy-derived | Override rows per staged state-pool chunk. |
| `FLASHINFER_PACKED_KDA_STAGES` | batch policy | Override cp.async pipeline depth; use `0` for register prefetch or at least `2` for pipelining. |
| `FLASHINFER_PACKED_KDA_NO_CPASYNC` | `0` | `1` forces the register-prefetch kernel instead of cp.async staging. |
| `FLASHINFER_PACKED_KDA_PERSIST` | `0` | `1` enables the persistent-CTA schedule when alignment and pipeline constraints permit it. |
| `FLASHINFER_PACKED_KDA_TMA` | `0` | `1` uses TMA state-pool reads when the staged kernel is active; mutually exclusive with bulk stores. |
| `FLASHINFER_PACKED_KDA_BULK` | `0` | `1` uses bulk stores for an eligible staged kernel; mutually exclusive with TMA reads. |
| `FLASHINFER_PACKED_KDA_PRIVRING` | `1` | `1` selects private-ring synchronization when neither TMA nor bulk-store mode is active. |
| `FLASHINFER_PACKED_KDA_MAXRREG` | `0` | Override the generated kernel's maximum-register tuning hint (`0` leaves it unspecified). |
| `FLASHINFER_PACKED_KDA_L2HINT` | `256` | Override the generated kernel's L2-prefetch-size hint. |
| `FLASHINFER_PACKED_KDA_HPC` | `1` | Override heads combined per CTA for eligible staged whole-head tiles. |
| `FLASHINFER_PACKED_KDA_L2POL` | `0` | Override the experimental L2 cache-policy selector. |
| `FLASHINFER_PACKED_KDA_MINBLOCKS` | `0` | Override the generated kernel's minimum-blocks-per-SM launch-bound hint. |

## Development Workflow

### Typical Development Loop

1. Edit kernel code in `include/flashinfer/some_kernel.cuh`
2. Run test: `pytest tests/test_some_kernel.py::test_specific_case`
3. FlashInfer detects changes and recompiles automatically
4. No `pip install` needed!

### Modifying Existing Kernels

- **Kernel templates**: `include/flashinfer/**/*.cuh` - Changes picked up on next JIT compile
- **Launcher code**: `csrc/*.cu` - May need changes if adding new template parameters
- **Jinja templates**: `csrc/*.jinja` - Update if adding new config parameters
- **Python API**: `flashinfer/*.py` - Update if changing function signatures

### Creating Pre-compiled Packages

When ready to distribute:

```bash
# Build one binary provider for the target GPU.
export FLASHINFER_JIT_CACHE_PROVIDER_ARCH=9.0a
python -m build --no-isolation --wheel flashinfer-jit-cache-provider

# Build the matching shim with an exact dependency on that provider.
export FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="9.0a"
python -m build --no-isolation --wheel flashinfer-jit-cache
```

The provider build runs `flashinfer/aot.py`, which calls the registered
`gen_*_module()` functions and pre-compiles them for exactly one architecture.
The shim contains metadata and exact dependencies, not compiled kernels. Use
the same `FLASHINFER_LOCAL_VERSION` and `FLASHINFER_DEV_RELEASE_SUFFIX` values
for every provider and its shim when producing CUDA-specific or nightly wheels.

## Build System Details

- **Build backend**: Custom PEP 517 backend in `build_backend.py`
- **Data directories**: Build creates symlinks for editable installs:
  - `3rdparty/cutlass` → `flashinfer/data/cutlass`
  - `csrc` → `flashinfer/data/csrc`
  - `include` → `flashinfer/data/include`
- **Version**: Generated in `flashinfer/_build_meta.py` from `version.txt`

## External Integrations

### TVM-FFI: Cross-Language Unified ABI

FlashInfer uses **TVM-FFI** (Apache TVM's Foreign Function Interface) for bindings, which provides a **cross-language unified ABI**. This means:

- **Not limited to PyTorch**: The same compiled kernels can be used from multiple frameworks
- **Language agnostic**: Bindings can be created for Python, C++, Rust, etc.
- **Type-safe marshaling**: Automatic tensor/array conversion between languages
- **Export syntax**: Use `TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, func)` to expose C++ functions

While FlashInfer currently provides PyTorch bindings, the underlying kernels are framework-agnostic thanks to TVM-FFI.

### Other Integrations

- **PyTorch Custom Ops**: `torch.library` for `torch.compile()` and CUDA graph support
- **Ninja Build**: Direct ninja generation, no CMake complexity

## Supported GPU Architectures

FlashInfer supports NVIDIA SM75, SM80, SM86, SM89, SM90, SM100, SM103, SM110, SM120, and SM121.

## Release Versioning

FlashInfer follows a "right-shifted" versioning scheme (`major.minor.patch[.post1]`):

- **major**: Architectural milestone and/or incompatible API changes (similar to PyTorch 2.0)
- **minor**: Significant backwards-compatible new features
- **patch**: Small backwards-compatible features (new kernels, new SM support) and backwards-compatible bug fixes
- **post1**: Optional suffix for quick follow-up release with just backwards-compatible bug fixes

## External Documentation Resources

When working with FlashInfer's dependencies and tools, refer to these official documentation sources:

### Core Dependencies

- **TVM-FFI**: Apache TVM's Foreign Function Interface
  - Documentation: <https://tvm.apache.org/ffi/>
  - Package: `apache-tvm-ffi` (<https://pypi.org/project/apache-tvm-ffi/>)
  - Use for: Understanding FFI export syntax, cross-language bindings

- **CUTLASS**: NVIDIA's CUDA Templates for Linear Algebra Subroutines
  - **Recommended**: Read source code directly in `3rdparty/cutlass/` (documentation is often outdated)
  - Repository: <https://github.com/NVIDIA/cutlass>
  - Use for: GEMM kernel implementations, tensor core operations

- **CuTe (CUTE DSL)**: CUTLASS's Cute Layout and Tensor DSL
  - Documentation: <https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html>
  - **Tip**: Add `.md` to get Markdown format: <https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html.md>
  - The Cute DSL kernels rely on Python modules from the `nvidia-cutlass-dsl` pip package, not to be confused with Python modules in the `3rdparty/cutlass` submodule
  - Tutorial: <https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL>

- **PTX ISA (Parallel Thread Execution)**: NVIDIA's PTX instruction set documentation
  - Documentation: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
  - **Index/Table of Contents**: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.md>
  - **Tip**: Add `.md` to any page URL to get Markdown format
  - Use for: Low-level instruction details, new GPU architecture features, inline PTX assembly

### When to Consult These Docs

- **Understanding new GPU architecture features** → Check PTX ISA documentation for latest instruction details
- **Working on FFI bindings** → Check TVM-FFI docs for export patterns and type marshaling
- **Implementing Tensor-Core kernels using CUTLASS** → Read source code in `3rdparty/cutlass/`
- **Using tensor layouts or warp-level operations** → Refer to CuTe documentation
- **Writing inline PTX assembly** → Consult PTX ISA for instruction syntax and semantics

These dependencies are included in FlashInfer's `3rdparty/` directory or `requirements.txt`.

### Some final suggestions for all AI agents

> Because practical engineering involves the accumulated experience of trial and error, match the coding style, efficiency, complexity, verbosity, and defensiveness by learning from existing code as much as possible—this document contains many pointers on where to find examples. Document intentional departures with rationale. Mentioning "AI-assisted" in the git commit message is good transparency. For performance-critical hot paths, leave justification for the special algorithmic choices and other potential alternatives in a comment for review.

**Keep documentation in sync with code changes:** When modifying code that is referenced in this document or in `.claude/skills/`, update the corresponding documentation immediately. This includes:
- Important infrastructure changes (e.g., `@flashinfer_api`, `@backend_requirement`, TVM-FFI macros) → Update examples in `CLAUDE.md` and relevant skill files
- New patterns or conventions → Document them for future reference
- Deprecated approaches → Remove or mark as deprecated in docs
- New error handling patterns, macros, or utilities → Add to relevant skill tutorials
