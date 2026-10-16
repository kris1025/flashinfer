# Experimental DeepSeek V4.1 decode

The mixed-cache and window-only decode APIs offer two explicitly selected,
experimental backends on SM100:

| `backend` | Implementation |
|---|---|
| `"cute-dsl"` (default) | Existing **Frost** CuTe-DSL decode |
| `"prims-ts"` | Native D512 adapter: explicit FP8 Q + ordinary FP8 SWA |

Calling these experimental APIs is the opt-in; no environment variable is
required. There is no `"auto"` selector. Both backends are JIT-only, and plans
cannot be shared between backends. The existing `"cute-dsl"` default is unchanged.

The native-FP8 implementation is the first commit in the source reorganization;
optional Prims-TS MXFP8 compatibility is a separate follow-up. Native E4M3 Q/SWA
remains the primary adapter and the Prims-TS example/benchmark default. Explicit
BF16 Q with uint8 MXFP8 SWA selects the compatibility extension through a lazy
import, without a public decode-format argument. Historical validation records
below describe their original pre-reorganization snapshots, not the rebuilt
commits. No new GPU or performance qualification is claimed by this split.

## Native FP8 SWA

| Mode | Q `[B,1,64,512]` | SWA `[pages,64,1,width]` | Arithmetic |
|---|---|---|---|
| Prims-TS | `torch.float8_e4m3fn` | Same typed E4M3, width 512, unit scale, no side plane | Native FP8 TMA/QK/P/PV |
| Frost (public default), or optional Prims-TS MXFP8 | BF16 | uint8 MXFP8, width 528, group32 E8M0 scale plane | Selected tiles staged as BF16 |

Native FP8 mode requires **explicit FP8 Q from the caller**; decode does not
implicitly quantize BF16 Q. MXFP8 cannot be reinterpreted by discarding its scales,
and mixed BF16-Q/ordinary-FP8-SWA or FP8-Q/MXFP8-SWA declarations are rejected.
The optional writer uses `format="swa_fp8"`: finite BF16/FP32 rows are clamped
to `[-448,448]` then rounded to nearest-even E4M3 in plain row-major storage,
retaining signed zero and subnormals. The existing `swa_mxfp8` and `main_kv_fp4`
writers/encoding remain separate and unchanged.

SWA uses fixed Keep-AB/M64/BK128/D128/retained8/V1/split1/nonpersistent execution,
with no custom packed producer or BF16 SWA dequantization. Optional global FP4
keeps the width-288 page-plane ABI and selected-tile BK64/BF16 arithmetic; only Q
is exactly expanded to BF16 for that pass. Both passes export sink-free normalized
FP32 source O and log2 LSE, then a joint merge applies the sink once and rounds
only final O to BF16. Returned FP32 LSE remains natural-log and excludes sinks.
Window-only omits global attention, global source states and BF16 query scratch.
Each call/replay refreshes selections, fixed execution lengths and true valid
counts. Plans cannot cross modes or backends.

Optional BF16-Q/MXFP8-SWA compatibility lives in `prims_ts_mxfp8.py`, with its
own compiler, plan class and `deepseek_v41_prims_ts_mxfp8` JIT namespace. Its
BK64/BF16 Keep-AB graph uses fused split-1 finalization, not the native FP8
source merge. `prims_ts_mxfp8_resources.py` adds E8M0-scaled SWA conversion,
mixed-source tag routing and the packed launcher; its narrow subclass inherits
FP4 conversion, retained BF16 stores and all-writer AsyncThread publication from
the primary FP4-only resource. Native FP8 calls never import this extension.

**Post-repair default full257 and independent dense57 passed on B200 / DSL
4.8.0 / bundled CUDA 13.4** (job `5008381`). All 50 new ordinary-FP8 cases passed
within the full matrix, including writer/decode graph replay. The previously
failing legacy Frost capture case also passed in full-suite order. This follows
a test-local conversion-module lifetime repair, without changing numerical gates
or default GC/capture settings; see the clean post-repair validation below.
The added tests cover all
global widths `0/64/.../512`, source-only/empty/poisoned selections, changed-data
graph replay, exact query expansion, actual native FP8 configuration and FP32
source precision (`19/65`, not BF16 partials cast back to FP32). Its independent
FP64 oracle uses actual stored E4M3 Q/SWA and independently decoded FP4 global.
Each output component must satisfy the native FP8 rounding bound:

```text
(2**-4 + 3*2**-8 + 1e-5) * sensitivity
  + (2**-10 / 448) * underflow + 1e-6
```

Here `sensitivity = sum(weights * abs(V))` and
`underflow = exp(max(logits) - normalizer) * sum(abs(V))`, with the sink included
in `normalizer`. Empty bounds/output are exactly zero; KV LSE retains the strict
`atol=1e-5, rtol=0` gate and exact `-inf` mask. Global-only still uses the strict
BF16 relative-L2/max-scaled gates; legacy helpers/tolerances are not changed.
The historical 207-case BF16-Q/MXFP8 matrix does **not** qualify this new path;
the post-repair full257 result below does for its tested setup. No benchmark or
performance claim is made for either mode.

The guarded host-only source-store/adapter/oracle suite excludes the root CUDA
collection probes. Use the thread caps before importing PyTorch:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
FLASHINFER_DISABLE_JIT=1 CUTE_DSL_ARCH=sm_100 \
python -m pytest --confcutdir=tests/experimental \
  tests/experimental/test_deepseek_v41_source_state_cpu.py -v
```

These are CPU configuration/plumbing/reference checks, not GPU numerical or
CuTe compilation qualification. Preserve the existing CUDA visibility and use a
fresh process. The fixture installs raw CUDA driver/runtime guards before CuTe
imports, declares a static target, and blocks Torch initialization/queries and
CuTe compile/load. Intercepted calls fail acceptance even if a dependency catches
the exception. DSL 4.7 eagerly attempts `cuInit` during import despite the static
target; those calls are blocked, not allowed through. Use DSL 4.8 for clean strict
host acceptance. An uninitialized Torch flag alone does not prove raw-driver
isolation.

### Native-first split acceptance (2026-10-15)

The native-FP8-only checkpoint passed **213 CPU cases**: 127 guarded
source-state/adapter/oracle contracts, 6 accelerator-free conversion-lifetime
regressions, 34 stdlib-only benchmark/example contracts, 40 existing native host
contracts and 6 unknown-backend rejections. The run used the existing Python
3.12.3 / Torch 2.11.0+cu128 environment and DSL 4.8 / bundled compiler metadata
13.4 overlay, thread caps of one, disabled plugin autoload/JIT and a static
`sm_100` target. It reported 304 warnings in 5.29 s; pytest and strict aggregate
exited 0, with no intercepted CUDA/JIT calls. Torch stayed uninitialized, CUDA
visibility was unchanged and all 114 fingerprints matched through postflight,
before these validation-note edits. The native entry does not import the optional
MXFP8 adapter; the native global pass uses a concrete FP4-only producer.
Records are in `/home/scratch.linquanh_gpu/tmp/dsv41-fp8-first.iU3DYW/native-checkpoint.{json,xml}`.
No GPU, JIT compilation, example or benchmark execution was performed for this
reorganization. The older records below do not qualify the rewritten sources.

### Optional MXFP8 follow-up acceptance (2026-10-15)

The combined follow-up passed **299 CPU cases**: 126 guarded native
source-state/adapter/oracle contracts, 37 optional-MXFP8 routing/plan/resource
contracts, 6 conversion-lifetime regressions, 84 stdlib-only benchmark/example
contracts, 40 existing native host contracts and 6 unknown-backend rejections.
The native-only checkpoint's BF16-rejection case is replaced by the optional
mode's positive routing and both-direction incompatible-plan checks.

The fresh run used the same Python/Torch/DSL 4.8 overlay and strict static-target
settings above. It reported 376 warnings in 5.03 s, with pytest and strict
aggregate exit 0, no failures/errors/skips, no intercepted CUDA/JIT calls,
uninitialized Torch CUDA and unchanged CUDA visibility. All 119 frozen
source/runner/manifest fingerprints matched through postflight, before these
validation-note edits. Records are in
`/home/scratch.linquanh_gpu/tmp/dsv41-fp8-first.iU3DYW/mxfp8-checkpoint.{json,xml}`.

Static comparisons also preserved the original shared kernels, writer/validator
and conversion-lifetime tests byte-for-byte; native and legacy decode/cache test
fixture, oracle and gate helpers are AST-identical. The native compiler/source/plan/launch
bodies match after the explicit naming/FP4-only-class substitutions. The packed
launcher, all-writer pipeline and conversion/store bodies retain their original
ASTs. Ruff, formatting, in-memory Python parse/compile and whitespace checks
passed. These are host/static checks only: no GPU accuracy/graph replay,
CuTe compilation, example execution or benchmarking was performed.

### Historical CPU/static acceptance (2026-10-13)

Fresh processes used the existing Python 3.12.3 / Torch 2.11.0+cu128 environment,
thread caps of one, disabled plugin autoload/JIT, and `CUTE_DSL_ARCH=sm_100`.
The base distribution was `nvidia-cutlass-dsl==4.7.0+flashinfer.5` (module 4.7.0,
bundled compiler metadata 13.3); the existing overlay was package/module 4.8.0
(bundled compiler metadata 13.4). No packages were installed or modified, and no
CUDA compiler, GPU diagnostics, allocation or benchmark was executed.

| Selection | DSL 4.8 pytest | DSL 4.8 strict aggregate | DSL 4.7 pytest | DSL 4.7 strict aggregate |
|---|---|---|---|---|
| New source-state/adapter/oracle suite | 125 passed, exit 0 | Pass, exit 0; no intercepted calls | 125 passed, exit 0 | Fail, exit 1; two intercepted import-time `cuInit` calls |
| Existing native host contracts | 40 passed, exit 0 | Pass, exit 0; no intercepted calls | 40 passed, exit 0 | Fail, exit 1; two intercepted import-time `cuInit` calls |
| Existing unknown-backend rejection | 6 passed, exit 0 | Pass, exit 0; no intercepted calls | 6 passed, exit 0 | Fail, exit 1; two intercepted import-time `cuInit` calls |

The 125-case runs reported 268 warnings in 5.20 s (4.8) and 252 in 3.94 s (4.7).
The 40/6 runs reported 21/20 warnings in 1.84/0.51 s (4.8) and 5/4 warnings in
2.05/0.44 s (4.7). There were no pytest failures, errors or skips in these six
selections. **Pytest success does not erase the DSL 4.7 strict-guard failures.**
The tested 4.7 distribution eagerly evaluates `detect_gpu_arch(prefix)` while
constructing its environment manager, even with an explicit architecture. Both
attempts in each process were intercepted before the original driver function
could execute; caught guard exceptions still make aggregate acceptance fail.

A separate fresh-fixture control did not preload Cutlass: the first source-state
case passed on 4.8 (20 warnings, 3.96 s), while 4.7 produced one setup error
(`intercepted CUDA/JIT calls during import`, both `cuInit`; 4 warnings, 3.55 s,
pytest/aggregate exit 1). This verifies that the fixture itself exposes caught
import attempts rather than silently accepting them; it is not a numerical
failure or additional unique test coverage. Installed packages were not patched.
Earlier 125-case passes used narrower Torch/CuTe guards, without raw-binding
coverage, and do not establish strict raw-driver isolation.

Exact JUnit identities matched the prior 125/40/6 manifests. Each matrix retained
all 129 source/environment/runner/manifest fingerprints, including 115 repository
files and both newly untracked files; before/after Git state matched. CUDA
visibility remained initially unset, Torch remained uninitialized, and compile/
load guards were restored intact. Only the fixture and documentation changed
between the host40/API6 matrix and final125 matrix; no sources changed during
an active matrix. Records are in the local scratch directories
`/home/scratch.linquanh_gpu/tmp/dsv41-host-api-regression.JGtZuw` (static host/API)
and `/home/scratch.linquanh_gpu/tmp/dsv41-fp8-strict-host.uE7xg0sg` (final125 and
fresh-fixture controls). Subsequent validation-note edits do not alter those
recorded test-time fingerprints.

These results qualify CPU configuration/plumbing/reference checks only; they
predate the source-store Array-interface correction described below.

### Earlier ordinary-FP8 GPU attempts

One allocated B200 on `umb-b200-248` (Slurm job `4943022`) used the existing
DSL package/module 4.8.0 overlay and bundled CUDA 13.4. Preflight passed, and
allocated collection produced exactly 257 cases: the unchanged legacy 207 plus
50 new definitions (31 decode, 19 cache). The requested smoke manifest had 16
cases; the selected dense manifest retains the historical 57 identities.

The smoke stopped on its first case,
`test_fp8_swa_fp64_and_selection_counts[4-0]`, during plan-time JIT lowering:
`TypeError: object of type 'Array' has no len()`. It reported **1 failed**, 32
warnings in 18.94 s, pytest exit 1 and Slurm `FAILED` / exit `1:0`. No eager
accuracy or graph checks ran; the other 15 smoke cases, complete 257-case matrix
and selected dense regression phase did not run. Final postflight preserved all
134 source/environment/runner/manifest fingerprints (115 repository files),
Git state and Slurm's `CUDA_VISIBLE_DEVICES="0"`. No packages were modified.
Records remain in `/home/scratch.linquanh_gpu/tmp/dsv41-fp8-b200.O4zokZHT`.

The FP32 source store now obtains its static fragment width from
`values.shape[0]`, matching the bundled DSL Array interface rather than Python
`len()`. This changes neither arithmetic nor references/tolerances.
A fresh strict DSL 4.8 CPU replay after that correction passed the same 125 cases
(268 warnings, 5.41 s), with pytest and aggregate exit 0, zero intercepted calls,
and all 129 fingerprints, Git state and CUDA visibility preserved. Records are in
`/home/scratch.linquanh_gpu/tmp/dsv41-fp8-array-shape-host.PDUf2BWC`.

The corrected retry used one allocated B200 / 148 SMs on `umb-b200-263`
(Slurm job `4943256`), with the same DSL package/module 4.8.0 overlay and bundled
CUDA 13.4. Preflight and exact 257-case collection passed. The exact 16-case smoke
passed, including both SWA/global FP32 source-precision checks and changed-data
graph replay. The complete expanded matrix then finished:

| Selection | Result |
|---|---|
| DS4.1 decode | 142 passed / 1 failed |
| Cache writer/integration | 48 passed |
| Native sparse MLA | 66 passed |
| Total | 256 passed / 1 failed; 0 errors or skips |
| New ordinary-FP8 cases (within the full matrix) | 50 passed: 31 decode / 19 cache |

The full run reported 1721 warnings in 3686.71 s, pytest exit 1 and Slurm `FAILED`
/ exit `1:0`. Exact JUnit identities independently matched all 257 requested cases;
all 134 source/environment/runner/manifest fingerprints, Git state and Slurm GPU
visibility matched through full and final postflight, before these validation-note
updates. Records are in
`/home/scratch.linquanh_gpu/tmp/dsv41-fp8-array-shape.yTfsOLtQ`.

The sole failure was the unchanged legacy Frost case
`tests/experimental/test_deepseek_v41_decode.py::test_e8m0_scale_boundary_changed_graph[None-512-0-cute-dsl]`.
With effective batch 149 on 148 SMs, its prepared graph capture reported
`cudaErrorStreamCaptureInvalidated` (901), followed by the capture-end error
`operation failed due to a previous error during capture`. Numerical checks were
not reached. At that point, neither the originating CUDA API nor a source defect
had been identified. This failure must not be hidden by changing references,
tolerances, GC settings or capture error modes. The wrapper stopped after the
failed full phase, so the selected dense 57-case regression **did not run** in
this retry.

A fresh default retry on `umb-b200-263` (job `5000428`, the same toolchain) passed
isolated capture and smoke16, then completed the exact full257 with the same
**256 passed / 1 failed**, including all 50 new cases passing. There were 0 errors
or skips and 1707 warnings in 3738.65 s; pytest/runner exited 1 and Slurm recorded
`FAILED` / `1:0`. All 144 fingerprints, Git state and Slurm GPU visibility matched
through independent terminal corroboration. Dense57 again did not run.
Records: `/home/scratch.linquanh_gpu/tmp/dsv41-fp8-capture-default.osF1cWGG`.

### Capture diagnosis and test-resource lifetime repair

On one allocated B200 in `umb-b200-236` (job `5005300`, DSL 4.8.0 / bundled CUDA
13.4), default isolated capture **1/1**, smoke **16/16** and independent default
dense **57/57** passed. Dense took 1245.92 s including its wrapper; each default
phase had pytest/runner exit 0 and no failed, errored or skipped cases. The API
tracer was absent from those processes. All 185 fingerprints (115 repository
files), Git state and Slurm visibility matched through independent terminal
corroboration, before the following repair and validation-note edits.

A separate **instrumented diagnostic** preserved the full257 collection/order
but stopped at case 94 with **93 passed / 1 failed** in the same legacy Frost
case. This was not a completed or default full257 run. The trace records:

- A successful `cuLibraryLoadData` in case 81,
  `test_native_fp4_all_finite_codes`, returning library handle `0x67a484b0`;
  its next CuTe launch was the standalone `convert` probe.
- Capture begin/query success in case 94, followed by a GC marker interval
  containing `cuLibraryUnload` of that handle with typed driver result **900**.
- Subsequent launch/end-capture result **901**. The runtime
  `cudaLibraryUnload` return type was UNKNOWN and its status was not read.

CUPTI setup/finalization and record accounting succeeded with no dropped records,
write errors or visible sequence gaps. Same-node offline identity checks provide
provisional Torch/DSL stack evidence, not exhaustive producer coverage or complete
owning-object identification. The 341 unknown returns, 36 unknown parameter
records, 55,444 detail truncations and short PC chains limit the diagnostic.
Zero drops and GC overlap alone do not prove a complete causal chain.
The overall job remains `FAILED` / `1:0` from this diagnostic failure.
Records: `/home/scratch.linquanh_gpu/tmp/dsv41-capture-api-trace.6gAoXzEQ`.

The two standalone finite-code conversion tests now register their compiled
helpers with a test-scoped fixture that explicitly unloads loaded CuTe modules
at teardown. The installed standalone executor contains a bound-method cycle;
leaving its local owner to disappear can defer library cleanup into a later
capture. The repair releases those test resources before subsequent tests,
without changing decoder kernels, global JIT ownership, numerical gates, GC
policy or capture settings. Unexecuted modules and test-body failures are covered
by six CPU-only ownership regressions. A fresh `-S` process passed all six with
pytest/strict aggregate exit 0, no Torch/CUDA/CuTe imports and no intercepted calls.
An AST comparison also confirmed that every pre-existing test statement other
than the two owner registrations is unchanged.
Records: `/home/scratch.linquanh_gpu/tmp/dsv41-conversion-lifetime.AqXa1foM`.

### Clean post-repair GPU validation (2026-10-15)

One allocated B200 on `umb-b200-260` (Slurm job `5008381`) used CuTe-DSL
package/module 4.8.0 and its bundled CUDA 13.4 compiler, with fresh scratch/JIT,
CuTe, Triton and pytest caches and distinct Python processes per phase. Default
GC/capture settings and all independent references/tolerances were unchanged.
No tracer was loaded. The complete post-repair run passed:

| Phase | Result | Pytest elapsed | Warnings |
|---|---|---:|---:|
| Isolated legacy capture | 1 passed | 10.67 s | 35 |
| Ordinary-FP8 smoke | 16 passed | 370.02 s | 90 |
| Complete expanded matrix | 257 passed | 3552.03 s | 1707 |
| Independent selected dense subset | 57 passed | 1105.53 s | 980 |

The full matrix contains **143 decode, 48 cache and 66 native sparse MLA cases**.
Its unchanged legacy 207 and new ordinary-FP8 50 (31 decode / 19 cache) are a
disjoint exact partition, not additional tests beyond 257. Capture and smoke
repeat subsets of the full matrix. There were **0 failures, errors or skips** in
all four phases. The previously failing
`test_e8m0_scale_boundary_changed_graph[None-512-0-cute-dsl]` passed setup, call
and teardown at position 94 in full-suite order; all 771 full-matrix reports and
exact ordered JUnit identities passed. The new cases include FP32 source
precision, all global widths `0/64/.../512`, exact query expansion and changed-data
CUDA Graph replay, including `swa_fp8` writer/decode integration.

Raw/session/base pytest exits, Python wrappers and shell/timeout/tee/record
statuses were all 0. Actual Slurm root, `.batch` and `.extern` records were
`COMPLETED` / `0:0`; root elapsed time was 5307 s (88:27), within the three-hour
single-GPU allocation. Phase wrapper times are distinct from the pytest times
above (full 3570.71 s; dense 1120.68 s); none are kernel performance measurements.
All **5047 frozen source/environment/harness/manifest fingerprints**, Git state
and Slurm CUDA visibility matched through final postflight and independent
terminal corroboration, **before these documentation edits**. No installed
packages or tested sources were modified during the matrices.

The original frozen auditor exited 1 with `default_qualification=false` and
four command-record errors, but no nonpassing phase. It incorrectly compared
the observer's pre-parse argv snapshot with base_round's post-parse diagnostic
command: pinned pytest prepends the ini `--import-mode=importlib` token to the
caller's list in place. This did not represent another pytest execution or
unrecorded CLI options. A separate, versioned post-run checker requires the exact
canonical raw phase argv and ordered manifest, **only that literal ini prepend**,
and the complete interpreter/`-m pytest` command. It does not strip arbitrary
flags, compare suffixes only, filter errors or change the frozen runtime.
All other source/Git, JUnit, exit, GC/isolation and terminal checks reran unchanged.
The new report has **`default_qualification=true`, no evidence errors and no
nonpassing phases**, with an exact predicate diff and source/report hashes.
Its **166 guarded stdlib CPU positive/negative checks** passed with strict exit 0,
no accelerator/pytest imports and no intercepted calls; these are metadata checks,
not additional GPU qualification. The import-mode option itself was not directly
recorded; the reconciliation uses pinned parser/ini sources and both argv records.

Records are in
`/home/scratch.linquanh_gpu/tmp/dsv41-capture-fix-options.feoxy4_t`:
`terminal-accounting-independent.json` preserves main's actual scheduler query;
`final-corroboration.json` preserves the original false verdict;
`corroborate_terminal_postrun.py`, `postrun-command-selfchecks.json` and
`final-corroboration-postrun.json` preserve the strict correction and its evidence.
Neither the frozen inputs nor the original report was overwritten.

This qualifies the tested post-repair B200/4.8/13.4 full257 matrix and selected
dense57 subset, not all dense configurations, DSL 4.7 GPU execution, other
hardware/toolchains or checkpoint/full-model behavior. It is a successful default
regression of a test-resource lifetime repair, not exhaustive global JIT ownership
or historical unload-status proof. Earlier failed and diagnostic attempts remain
as recorded above. Benchmarks have not run; no speedup or measured crossover is
claimed.

## Validation status (historical BF16 Q / MXFP8 adapter)

**The historical DS4.1 decode/cache and native sparse MLA matrices passed on B200.**
The complete `--full` rerun used one allocated B200 in `umb-b200-041`
(Slurm job `4910237`) with CuTe-DSL package/module 4.8.0 and bundled CUDA 13.4:

| Suite | Passed | Failed |
|---|---:|---:|
| `tests/experimental/test_deepseek_v41_decode.py` | 112 | 0 |
| `tests/experimental/test_deepseek_v41_cache.py` | 29 | 0 |
| `tests/attention/test_prims_ts_sparse_mla.py` | 66 | 0 |
| **Total** | **207** | **0** |

All **207 collected cases completed**, with **0 failures, errors or skips**,
1679 warnings in 3427.86 seconds (57:07), and pytest exit code 0. Slurm recorded
`COMPLETED`, exit `0:0`, and elapsed time 1:00:09 for preflight, focused smoke and
full tests. The allocation ended normally. This used the production checkout and
a fresh writable scratch workspace. The exact 207-case node set matched the
original full matrix, and all 111 source fingerprints matched submission,
allocated preflight and final postflight before these validation-note updates.
Kernels and independent accuracy gates were unchanged during testing.

The native suite passed both BF16 and E4M3 coverage, including ordinary attention,
fused RoPE/quantized output, changed-metadata CUDA Graph replay, and both pinned
static/CLC Keep-AB runtime-domain regressions. All eight writer/decode integration
variants passed with exact encoded-byte references and unchanged FP64 gates.
A three-case focused smoke passed before the full run: the previously failing
FP8 `small-grid-wide-kv` graph, its BF16 counterpart, and the neighboring FP8
`two-cta` graph.

This qualifies only the tested B200/DSL 4.8.0/bundled CUDA 13.4 matrix. The separate
CPU-only acceptance subset in `tests/attention/test_attention_ts_mla_decode.py`
historically passed all **40 cases with each of DSL 4.7.0 and 4.8.0**, with no
failures, errors or skips. Its recorded Torch/CuTe guards were not triggered and
GPU visibility was unchanged; those guards did not cover raw CUDA bindings.
The current stricter matrix above distinguishes pytest results from intercepted
4.7 import attempts. These host checks do not establish DSL 4.7 GPU qualification.
The separate dense D576 GPU tests in that file were not part of the 207-case matrix; the selected
GPU subset below has its own qualification. Benchmarks have not run, and no
performance measurements, measured crossovers, checkpoint parity or full-model
qualification are claimed.

### Historical implementation review (2026-10-13, before ordinary FP8)

Static review found **no confirmed correctness defects** in the reviewed packed
adapter/resources, shared Prims-TS Task/kernel/reducer changes, public selectors,
example/benchmark harness and regression/integration tests. Inspection included
the actual DSL 4.7/4.8 Task and pipeline implementations, selection/source masks,
int64 page-plane addressing, D128 staging, all-writer publication, retained-KV
release, initialization/neutral-split duties and compile/cache identity. This is
source review, not additional GPU qualification.

The recorded 207/57-case manifests, JUnit identities and exits were independently
rechecked, along with unchanged reference/accuracy-gate ASTs. The six existing
unknown-backend rejection cases also passed in CPU-only runs with **each of DSL
4.7.0 and 4.8.0**, with no failures, errors or skips; the recorded Torch/CuTe
guards were not triggered and GPU visibility was preserved. That historical guard
scope did not include raw CUDA bindings; see the current stricter results above.
All 15 changed Python files passed
syntax, Ruff and formatting checks, and `git diff --check` was clean.

This review changed only validation documentation and Python docstrings. Kernel,
test and reference fingerprints were unchanged, and the executable ASTs of the
API, example and benchmark were preserved. No new GPU allocation/tests ran; the
standalone example CLI and benchmarks were not executed.

Remaining coverage limits, **not confirmed defects**:

- Packed end-to-end tests cover global selection widths `0/64/128/192/512`, not
  the supported intermediate widths `256/320/384/448`.
- The exhaustive MXFP8 helper test exercises Frost's resource; the all-finite
  FP4 helper test exercises shared conversion primitives. Neither is exhaustive
  direct coverage of the separate packed D128 staging resource.
- The forced FP4 fallback passed with bundled CUDA 13.4; actual execution with
  bundled CUDA 13.0/13.1 remains unqualified.
- Unselected dense configurations, native DSL 4.7 GPU execution, other
  GPU/toolchain combinations and checkpoint/full-model behavior remain
  unqualified. Performance and backend crossovers remain unmeasured.

### Selected dense MLA regressions

A separate run on one allocated B200 in `umb-b200-250` (job `4936630`), with
CuTe-DSL 4.8.0 / bundled CUDA 13.4 and fresh scratch/cache paths, passed all
**57 selected dense D576 QK / D512 V GPU cases** in
`tests/attention/test_attention_ts_mla_decode.py`. There were **0 failures,
errors or skips**, with 980 warnings in 1063.65 seconds (17:43), and pytest exit
0. All three preceding smoke cases also passed, with 89 warnings in 85.18 seconds.
The smoke cases are members of the 57-case manifest, not additional unique
coverage. Slurm completed with exit `0:0` and elapsed time 19:57; the allocation
ended normally.

Coverage includes BF16/FP8 1CTA/2CTA paths, dense FP8 2CTA scheduling, runtime-K
pruning and fully masked split partials, compact/empty/causal queries,
non-power-of-two heads, CLC empty-tile progress, row-strided page tables and
CUDA Graph replay changing live Q/K/page metadata. Exact JUnit node identities
matched both manifests, all 111 source fingerprints matched submission,
allocated preflight and final postflight, and GPU visibility was preserved.
Kernels, tests and independent accuracy gates were unchanged during this run.
This is a **selected subset**, not qualification of the entire dense GPU suite,
DSL 4.7 GPU execution, or other GPU/toolchain combinations.

The first allocation attempt (job `4936466`) failed in the scratch preflight
`nvidia-smi` driver diagnostic because PyTorch's bare UUID lacked the required
`GPU-` prefix. No pytest cases ran in that attempt. The fresh retry normalized
the prefix without changing GPU visibility or the selected physical device;
its actual preflight and all tests passed.

Reproduce this selected subset with the same toolchain on an allocated B200:

```bash
test_file=tests/attention/test_attention_ts_mla_decode.py
python -m pytest --full -vv -rs --tb=short \
  "$test_file"::test_attention_ts_mla_decode_2cta_page_size_dtype_product \
  "$test_file"::test_attention_ts_mla_decode_runtime_k_pruning_product \
  "$test_file"::test_attention_ts_mla_fp8_fully_masked_split_partials \
  "$test_file"::test_attention_ts_mla_decode_packed_q_public_parity \
  "$test_file"::test_attention_ts_mla_decode_packed_q_clc_empty_tile_progress \
  "$test_file"::test_attention_ts_mla_decode_graph_reloads_all_live_metadata \
  "$test_file"::test_attention_ts_mla_decode_sq2_head_dtype_mask_product \
  "$test_file"::test_attention_ts_mla_decode_packed_dtype_mask_page_product \
  "$test_file"::test_attention_ts_mla_decode_all_empty_packed_query_noop \
  "$test_file"::test_attention_ts_mla_decode_packed_non_power_of_two_heads \
  "$test_file"::test_attention_ts_mla_decode_page_size_dtype_product \
  "$test_file"::test_attention_ts_mla_row_strided_page_table_metadata_mutation \
  "$test_file"::test_attention_ts_mla_2cta_graph_reloads_remapped_page_window
```

### Compilation-failure repair history

The earlier full run on `umb-b200-250` (job `4907814`) completed all 207 cases:
decode 112 passed, cache 29 passed, native sparse MLA 35 passed / 31 failed, with
no errors or skips, 1540 warnings in 1959.36 seconds, and pytest exit 1. All 31
native failures occurred during plan-time FrontendNext/task-scheduling compilation,
before eager accuracy or graph replay checks:

| Earlier native compilation failure | Cases |
|---|---:|
| 2CTA `_run_pre_work_loop_entries` argument mismatch (3 arguments given, 2 accepted) | 10 |
| 1CTA `SCOPE_READ_NEVER_SET` at `parallel_reduction.py::flat_query_row_state` | 9 |
| 1CTA `CONTAINER_OBJECT_REPLACED` for `consumer_vars` during task initialization | 11 |
| Static Keep-AB `work_tile` TaskLocalVariable type mismatch | 1 |

The original pinned CLC regression passed; static failed during compilation,
not a configuration or numerical assertion. No baseline comparison was performed,
so these failures are not labeled pre-existing or attributed to the task-cache
snapshot correction or optional producer hooks.

Four narrow Task-interface/lowering repairs address method-specific context
forwarding, the canonical task-scheduling work-tile type, separate G1
statistics/output temporaries, and unconditional task initialization with guarded
data-work bodies. The plain-Python task-cache getter and numerical gates are
unchanged. A post-repair smoke run on `umb-b200-236` (job `4909337`), with the same
DSL 4.8.0 / bundled CUDA 13.4 toolchain and fresh scratch, completed all 40 selected
cases: **39 passed / 1 failed**, with no errors or skips and unchanged source
fingerprints. Both pinned static/CLC regressions and all eight selected packed
selection/cache replay cases passed. The remaining FP8 `small-grid-wide-kv` case
failed during plan-time compilation of the fused 2CTA parallel reducer with
`SCOPE_READ_NEVER_SET`. Separating its warp-only LSE/split locals from the all-thread
output locals, without changing arithmetic or gates, resolved the follow-on
failure. The three focused GPU cases and complete rerun in job `4910237` passed.

Run the current full suites with the same DSL/compiler on an allocated SM100 B200.
They now include new ordinary-FP8 cases in addition to the historical 207-case set;
the old pass count is not a result for this expanded command:

```bash
python -m pytest --full -vv -rs --tb=short \
  tests/experimental/test_deepseek_v41_decode.py \
  tests/experimental/test_deepseek_v41_cache.py \
  tests/attention/test_prims_ts_sparse_mla.py
```

### Earlier targeted runs

A production-source run on one B200 in `umbriel-b200-040` (Slurm job `4904378`)
reported **10 passed, 0 failed, 0 skipped**, with 30 warnings in 123.12 seconds:

| Targeted test | Global selection widths | Result |
|---|---|---|
| `test_prims_ts_selection_refresh` | 0, 64, 192, 512 | 4 passed, including poisoned selections and changed-index CUDA Graph replay |
| `test_prims_ts_zero_query_tile_counts` | 0, 64, 192, 512 | 4 passed, including mixed, global-only, window-only and empty requests |
| `test_prims_ts_does_not_compile_cute_dsl` | 0, 64 | 2 passed; Prims-TS does not invoke the Frost compiler |

The run used a fresh writable scratch workspace and the existing getter
correction, with all independent FP64 output/LSE gates unchanged. The verified
toolchain was CuTe-DSL package/module 4.8.0 with bundled CUDA 13.4, Torch
2.11.0+cu128 (Torch CUDA 12.8), and driver 615.71.09 (driver API 13.4), on SM100
with 148 SMs.

Earlier testing on `umbriel-b200-027` exposed the global192 failure: tracing
found a zero task-cache length despite a live execution span of 384, truncating
runtime domains to one BK64 tile. The task-cache getter now snapshots current
Python state without a separate `@cute.jit` boundary. The production
`test_prims_ts_selection_refresh[192]` regression now passes both eager and
changed-index graph checks, and the zero-query regressions verify sink-exclusive
LSE against the exact valid-selection count. CUDA initialization failures on the
old node had blocked subsequent testing; no GPU reset or service restart was
performed.

A separate native automatic-dispatch run on the same node, allocation and
toolchain reported **5 passed, 0 failures, 0 errors, 0 skips**, with 109 warnings
in 268.56 seconds. The five cases were:

- `test_native_graph[single-source-dtype0]`
- `test_native_graph[single-stream-short-kv-dtype0]`
- `test_native_graph[small-head-long-kv-reuse-dtype0]`
- `test_native_graph[multiwave-prefill-dtype0]`
- `test_native_graph[multiwave-decode-dtype0]`

Under the current native policy, the recorded BF16 multiwave-prefill case maps
to 1CTA Keep-AB/CLC with a 768-slot span; its generic accuracy test does not assert
the selected configuration. A separate `test_keep_ab_runtime_task_cache` now pins
a native BF16 Keep-AB/M64/BK64/retained-KV8/split-1 profile for both static and CLC
persistence. It checks the actual configurations handed to compilation, requires
runtime task domains (`fixed_sparse_kv_tiles == 0`) beyond 256 slots, and reuses
the existing FP64 oracle and changed-metadata graph replay with `SM-count + 1`
logical work tiles. Both cases ran in job `4907814`: CLC passed, while static
failed during plan-time FrontendNext compilation at `task_manager.run()` with a
`TaskLocalVariable 'work_tile'` type mismatch. Neither is an instrumentation
check of individual CTA execution counts. Both static and CLC passed after the
repairs in the 40-case smoke run (job `4909337`) and again in the complete 207-case
rerun (job `4910237`).

The earlier automatic-dispatch passes alone did not establish complete native or
persistent-family qualification; the final full matrix above now passes its
native and persistence coverage. This does not qualify other GPU/toolchain
combinations. Benchmarks have not run, and no benchmark measurements or measured
backend crossovers are claimed.

The **Frost** backend is written in CuTe DSL with
`cutlass.experimental.primitives` and adapted from Mengyu Guo's CuTe DSL HCA
([#3943](https://github.com/flashinfer-ai/flashinfer/pull/3943),
[#4368](https://github.com/flashinfer-ai/flashinfer/pull/4368)). Original attribution
and licenses are retained. MMA, TMEM and cache memory operations use
`cutlass.experimental.primitives`. FlashMLA informed the single-CTA resource
budget, deferred softmax rescaling, WS QK/PV layouts with partial-score
exchange, fused softmax/correction and shared-memory/TMA output. The original
HCA foundation and these FlashMLA design contributions are acknowledged in
the source.

Run the self-contained example with an explicit selector:

```bash
python examples/deepseek_v41/decode.py --backend cute-dsl
python examples/deepseek_v41/decode.py --backend prims-ts
# Optional legacy BF16-Q/MXFP8-SWA compatibility mode:
python examples/deepseek_v41/decode.py --backend prims-ts --prims-ts-mode mxfp8
```

The Prims-TS adapter requires **CuTe-DSL >= 4.8 with its bundled CUDA 13
compiler** on SM100. Installing a system CUDA 13 toolkit alone is insufficient:
CuTe uses its bundled compiler independently of `CUDA_HOME`. This adapter
requirement does not change the existing Frost default or the native sparse
wrapper's toolchain contract. Decode does not require the external `flash_mla`
package. The included optional cache writer makes the example self-contained.
The benchmark records CuTe-DSL, bundled compiler, PyTorch CUDA and Triton
versions. In Frost, bundled CUDA >= 13.2 selects native FP4-to-BF16 conversion
and packed BF16 multiplication. Older bundled compilers retain the existing
conversion sequence inside the same kernel. Packed conversion and WS MMA
retain small public `prims.inline_ptx` helpers; other MMA/TMEM operations use
the typed primitives.

The optional Triton cache writer uses its own bundled Blackwell `ptxas`,
separate from the CuTe compiler and system toolkit. Its FP4 path emits
`cvt.rn.satfinite.e2m1x2.f32` and is limited to SM100. This is not a package-wide
CUDA version requirement.

Frost keeps its existing split schedule for small batches and selects
WS QK/PV with fused softmax/correction for K512/B128+. When that batch exceeds
the device's SM count, CTAs process multiple requests through the same pipeline.
Dequant warps prefetch the next request's indices and compute cache
offsets in registers while the current request finishes. An existing boundary
barrier protects publication of this metadata into shared memory, where it
avoids repeated address calculation during gathers.
When both complete cache pools fit below 32 GiB, metadata uses 32-bit offsets
in 16-byte units; larger pools retain 64-bit byte offsets. This preserves support
for cache addresses above 2 GiB. This is internal scheduling within
`backend="cute-dsl"`, unchanged by the new selector.

Native Prims-TS uses fixed **Keep-AB / M64 / retained-KV8 / split 1 / nonpersistent**
execution: BK128 native FP8 SWA plus an optional BK64/BF16 global FP4 pass and
FP32 source merge. It does not use Frost's heuristics or the native wrapper's
automatic policy. There are no measured crossovers or tuned batch thresholds.
A compiled callable launches GPU sanitation, attention and merge on the same
stream. Only selected global FP4 tiles expand directly into retained BF16 shared
memory; `Fp4KvProducer` is a concrete FP4-only resource, not an MXFP8 SWA loader.

Both backends use queries `[B,1,64,512]`, 64-token pages, 128 window positions
and up to 512 FP4 compressed positions, padded to multiples of 64. The Prims-TS
example defaults to `swa_fp8` and caller-converted FP8 Q; explicit
`--prims-ts-mode mxfp8` instead selects BF16 Q/`swa_mxfp8`, as used by Frost.
The mode option does not affect the Frost backend. Window-only uses the same
selector and input mode, with no compressed stream.

This model-specific quantized-cache adapter is separate from
`BatchSparseMLADecodePagedTSWrapper`: that native wrapper still accepts matching
native BF16 or E4M3 Q/KV and **does not accept packed quantized cache inputs**.
The adapter does not broaden its input contract.
Physical slot `-1` and out-of-capacity slots are masked without reading cache
storage. Every valid cache slot must be initialized and causally visible.

Frost and the optional Prims-TS MXFP8 mode use BF16 Q/K/V/P and FP32
accumulation/softmax, with independent FP64
relative-L2 < 0.005, max-scaled error < 0.01 and absolute LSE error < 1e-5.
Those legacy gates are unchanged. Native FP8 SWA uses the per-component
rounding bound above, the same strict LSE/empty gates and additional strict BF16
checks when only global is live. Checkpoint accuracy has not been established.

`deepseek_v41_decode` returns `(out, lse, plan)`. Output is BF16 with Q's shape;
LSE is FP32 `[B,64,1]`, natural-log and **sink-exclusive**. A finite per-head
sink logit or `-inf` is accepted. The sink contributes only to output's
normalization; an empty row returns zero output and `-inf` LSE. Applying the
sink at final normalization preserves LSE for very large sink logits.

Prepare once eagerly and reuse the plan with the **same explicit backend and
input mode** for allocation-free CUDA Graph replay. Passing a plan to the other
backend or Prims-TS input mode is an error. Plan-owned output/LSE/workspace are overwritten on every call. Ordinary
plan calls may use new input addresses if shape, strides, dtype and device match.
A captured graph retains the pointers recorded during capture: keep input
addresses stable and update values in place during replay. Changing addresses
requires graph recapture or an appropriate graph update; keep captured buffers
alive for the graph's lifetime. A plan must not run concurrently on multiple
streams, and its buffers must not alias
inputs. Training, speculative multi-token queries, SM103 and full-model E2E
validation are outside this initial scope.

`deepseek_v41_quantize_cache` is an optional Triton helper for producing these
three cache formats from contiguous BF16/FP32 `[N,512]` rows. It fuses quantization
and writing into page64 storage; `out` plus int32 `slots[N]` supports incremental
updates without allocation. Negative and out-of-capacity slots skip publication.
Valid slots must be unique: this is a caller precondition, not checked by the
helper. Unwritten slots remain unchanged. Apply
model-required RoPE before quantization. This helper is provided for integration
and reproducible examples, with no competitive-performance claim. Decode does
not call it and accepts compatible cache bytes from any producer.

Within each packed MXFP8/FP4 64-token page, all data rows precede all scale rows.
The opaque uint8 `[pages,64,1,width]` view does not interleave scales per token.
Ordinary typed FP8 is different: just 64 contiguous payload rows, no side plane.

| Cache | Data bytes/token | Scale bytes/token | Scale format |
|---|---:|---:|---|
| Main FP4, uint8 width 288 | 256 E2M1 | 32 | E4M3, group16, no global scale |
| Window MXFP8, uint8 width 528 | 512 E4M3 | 16 | E8M0, group32 |
| Window ordinary FP8, typed E4M3 width 512 | 512 E4M3 | 0 | Unit scale |

The caller supplies physical slot IDs. For a logical token ID `t` in request
`b`, the usual mapping is `block_table[b, t // 64] * 64 + t % 64`, with invalid
entries mapped to `-1`. Slot mapping, RoPE, indexing and GEMM quantization remain
outside this decode API.

The reproducible GPU benchmark defaults to **explicit E4M3 Q / ordinary FP8 SWA**
when Prims-TS is selected. `--prims-ts-mode mxfp8` explicitly selects its optional
**BF16 Q / MXFP8 SWA** extension and unchanged strict legacy FP64 gates. Frost
and optional FlashMLA retain that legacy representation and their original gates. All arms derive from common seeded inputs
and share the global FP4 pool, selections and sink, but each is checked against
its own represented-value FP64 oracle. The native arm uses the componentwise
rounding bound and exact empty/strict sink-exclusive LSE gates above. Only selected
global rows are decoded for that reference, not the entire production-sized pool.
JSON labels each arm's dtype, SWA encoding, shapes, arithmetic and accuracy
contract, and distinguishes different-arithmetic comparisons. Native FP8 versus
MXFP8 timings are **not** a like-for-like backend comparison:

```bash
# The existing default remains cute-dsl only:
python benchmarks/bench_deepseek_v41_decode.py \
  --batches 1,4,16,32,64,128,256 --context 32768 --output frost.json
# Compare the two in-tree backends without external FlashMLA:
python benchmarks/bench_deepseek_v41_decode.py \
  --backends cute-dsl prims-ts --batches 1,4,16,32,64,128,256 \
  --context 32768 --output backends.json
# Explicit packed-mode comparison shares the same BF16 Q/MXFP8 SWA tuple:
python benchmarks/bench_deepseek_v41_decode.py \
  --backends cute-dsl prims-ts --prims-ts-mode mxfp8 \
  --batches 1,4,16,32,64,128,256 --context 32768 --output packed-backends.json
# Optional comparison requires an externally built FlashMLA installation:
python benchmarks/bench_deepseek_v41_decode.py \
  --backends cute-dsl prims-ts --batches 1,4,16,32,64,128,256 \
  --context 32768 --flashmla --output paired.json
```

`--flashmla` requires a FlashMLA build with DS4.1 MXFP8/FP4 mixed-cache support
and the `FlashMLASchedMeta` API. `get_mla_metadata()` returns an empty scheduler
object and a `None` placeholder; the first eager call initializes it from the
fixture before graph capture. The benchmark creates fresh metadata per shape.

Each backend gets an independent plan. Timing uses warm CUDA Graph/events,
32 complete public calls per replay and symmetric backend order, so all per-call
metadata preparation, decode and merge GPU work is included. JSON records
correctness metrics, per-backend split counts and latency samples, toolchain
versions and source SHA256 hashes. Quantization, JIT, allocation, graph capture
and CPU overhead are excluded. These component measurements are not model
token latency or throughput.

The prepared semantic and changed-input CUDA Graph tests require
CuTe-DSL >= 4.8 with its bundled CUDA 13 compiler on SM100 when including the
Prims-TS adapter. Current cache-writer integration covers ordinary-FP8 Prims-TS,
optional MXFP8 Prims-TS and legacy Frost, including window-only decode and selected
global widths. It
captures cache writing and decode together, changes values and selections at fixed
addresses, and checks poisoned-buffer replay and the transition to empty
selections. Exact encoded-byte references run before the independent FP64 decode
gates. The original legacy integration covered both backends at global widths
128/192/512; all eight expanded legacy cases passed in the historical full
matrix on the allocated B200 in Slurm job `4910237`, with CuTe-DSL 4.8.0 /
bundled CUDA 13.4. The new `swa_fp8` writer/decode integrations use explicit FP8 Q
at widths 0/128/192/512; all passed within the clean post-repair full257 run on
`umb-b200-260` (job `5008381`, B200 / DSL 4.8.0 / bundled CUDA 13.4), including
changed-data CUDA Graph replay and exact encoded-byte references.

```bash
pytest tests/experimental/test_deepseek_v41_decode.py \
  tests/experimental/test_deepseek_v41_cache.py --full -v
```

The APIs and both backends remain experimental and may change or disappear
without compatibility guarantees. They are not registered for AOT or trace.
