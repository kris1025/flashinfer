# Native sparse MLA

`BatchSparseMLADecodePagedTSWrapper` implements Dqk = Dv = 512 sparse
attention on SM100/SM103 through the existing Prims-TS MLA families. Q and KV
use matching native BF16 or E4M3; output is BF16. Calling the API opts into the
existing experimental Prims-TS backend. CuTe-DSL >= 4.7 is required.

## Code organization

- `attention/prims_ts/mla_decode.py`: the public `plan()` / `run()`
  wrapper, input validation, workspace ownership and kernel launch.
- `sparse_policy.py`: datatype-specific schedule selection, buffer sizing and
  split/V partition sizing from planned workload bounds and SM count.
- `sparse_views.py`: live per-request storage-row metadata used by the loaders.
- The existing 1CTA/2CTA reducers finalize split-KV outputs, sinks and LSE.
  `kernels/separate_reduction.py` shares their arithmetic and merges
  independently scaled source results.
- `helpers/gather.py` and the existing 1-CTA/2-CTA families: TMA gather4,
  cached offsets, retained KV and softmax/correction pipelines.

## Interface

```python
from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper

wrapper = BatchSparseMLADecodePagedTSWrapper(workspace_buffer)
wrapper.plan(q.device, B, H, max_topk=K, max_extra_topk=K_extra,
             max_seq_len_q=SQ, q_data_type=q.dtype,
             has_sinks=True, return_lse=True)
# Applications prepare SparseMLAPreparedMetadata before attention.
out, lse = wrapper.run(q, kv_cache, metadata, extra_kv_cache=extra_cache,
                       sinks=sinks, softmax_scale=model_softmax_scale)
```

Omit the extra pool and use `max_extra_topk=0` for one selected source.
Neither pool is required to be SWA. Both lists contribute to one softmax;
duplicate entries retain their multiplicity. Native pools may be paged
`[pages,page_size,512]` or singleton-head NHD/HND tensors, including page
padding. Queries are `[B,SQ,H,512]`, or `[total_q,H,512]` with `packed_query=True`
and `qo_indptr`. Projection, RoPE, compression and index selection belong to
the caller; packed FP8 cache formats are not accepted.

FP8 accepts positive per-tensor Q and per-source KV descales. Independent
source descales use two attention passes and a joint merge; shared scales
use one pass. BF16 descales must be one. Sinks add denominator mass only;
returned LSE is the natural logarithm and excludes sinks. Empty rows produce
zero output and negative-infinity LSE. Selected values and effective scale
products must be finite.

## DeepSeek V4.1 quantized-cache adapter (experimental)

The model-specific `deepseek_v41_decode(..., backend="prims-ts")` and
`deepseek_v41_window_decode(..., backend="prims-ts")` APIs provide a separate
adapter for the native D512 task graph. The default remains `"cute-dsl"` (Frost);
selection is explicit, with no `"auto"` route or required opt-in environment
variable. Plans belong to one backend and cannot cross backends.

The adapter's scope is SM100 / H64 / Sq1 / D512. The primary Prims-TS mode uses
explicit `torch.float8_e4m3fn` Q `[B,1,64,512]` and the same dtype of contiguous,
unit-scale SWA `[pages,64,1,512]`. This is plain row-major E4M3 without a scale
plane. `deepseek_v41_quantize_cache(..., format="swa_fp8")` is the optional
writer. Decode does not silently quantize BF16 Q or accept cross-paired formats.

The separate follow-up adds optional BF16-Q/uint8-MXFP8-SWA compatibility in
`experimental/deepseek_v41/prims_ts_mxfp8.py`. That explicit input pairing is
routed by a function-local import; native FP8 calls do not import or compile the
extension. Its compiler, plan and `deepseek_v41_prims_ts_mxfp8` JIT namespace are
separate from the primary native `deepseek_v41_prims_ts_fp8` path. Its BK64/BF16
Keep-AB graph uses fused split-1 finalization. `prims_ts_mxfp8_resources.py` adds
E8M0 SWA conversion and mixed-source routing, inheriting FP4 stores and all-writer
publication from the primary FP4-only resource. Plans cannot cross modes, even
with otherwise matching declarations.

Frost retains BF16 Q and uint8 page64 MXFP8 SWA of width 528, with group32 E8M0
scales. Both backends may use optional uint8 FP4 compressed caches of width 288.
Packed MXFP8/FP4 use a page-plane ABI: all data rows precede all scale rows within
each page. Window indices have capacity 128; compressed capacity is a multiple
of 64 up to 512. Shared validation opts into the FP8 contract only for Prims-TS;
the public default and Frost contract are unchanged.

**The native wrapper's packed-input restriction remains unchanged.**
`BatchSparseMLADecodePagedTSWrapper` still takes matching native BF16 or E4M3
Q/KV, not these uint8 MXFP8/FP4 pools. The cache/resource adapter is not a new
input mode of the native wrapper and does not extend its existing CuTe DSL 4.7
contract.

The V4.1 adapter requires **CuTe-DSL >= 4.8 with its bundled CUDA 13 compiler**;
a system CUDA 13 toolkit alone is insufficient. Fixed **Keep-AB / M64 /
retained-KV8 / split 1 / nonpersistent** execution uses BK128 for native FP8 SWA
and BK64 for optional global FP4, separate from the native scheduling policy
below. There are no measured backend crossovers or tuned batch thresholds.

The compiled callable first launches GPU index sanitation, then the native
sparse attention graph on the same stream. Sanitation maps negative,
out-of-capacity and reserved-sentinel indices to holes without compacting the
lists or removing duplicates. Plan-owned selection buffers are refreshed on
every call, including CUDA Graph replay. Fixed source lengths preserve list
positions; true valid counts and real zero extra lengths are refreshed for the
independent source passes.

In ordinary-FP8 mode, SWA uses the native typed E4M3 TMA/QK/P/PV path; there
are no BF16 SWA stages or dequantized KV buffers. Optional global FP4 uses a
separate BK64/BF16 selected-tile pass with an exact E4M3-to-BF16 expansion of
current Q. A narrow internal `source_state_store` hook publishes normalized
FP32 O before native BF16 packing, with sink-free log2 LSE. It permits only
nonfinalizing, direct-index, sink-free 1CTA Keep/M64/D512/no-RoPE/Sq1/V1,
split-1/nonpersistent configurations and is mutually exclusive with output
quantization. It does not add a public `out_dtype="fp32"` mode. The existing
P448 compensation is retained; source O must not receive another 448 factor.
`MergeMlaSourceStates` merges the FP32 states, applies the sink once and rounds
only final O to BF16, returning FP32 natural-log KV LSE. Window-only has no
global states or BF16 query scratch; unused compiled slots alias the live source.

The validation records below describe the original pre-reorganization source
snapshots. They do not qualify the rebuilt native-FP8 commit or its later MXFP8
extension; no new GPU or performance qualification is claimed by the split.

Post-conversion-lifetime-repair default validation passed on one allocated B200
in `umb-b200-260` (job `5008381`, DSL 4.8.0 / bundled CUDA 13.4): **257/257** full
cases, including all 50 new ordinary-FP8 cases, and the independent **57/57**
selected dense subset. The earlier attempts below remain historical failures.
The first allocated B200 smoke (`umb-b200-248`, job `4943022`) passed preflight
and collected the exact expanded 257-case manifest, but its first case failed
at JIT lowering: `cutlass.Array` does not support `len()`. Pytest exited 1 and
Slurm recorded `FAILED` / `1:0`; numerical checks and later matrices did not run.
All 134 fingerprints and CUDA visibility were preserved. The source store now
uses the array's static `shape[0]`.
The corrected retry on one B200 / 148 SMs (`umb-b200-263`, job `4943256`, the same
DSL 4.8.0 / bundled CUDA 13.4) passed the exact 16-case smoke and completed the
exact 257-case full matrix: decode **142 passed / 1 failed**, cache **48 passed**,
native sparse MLA **66 passed**, for **256 passed / 1 failed**, with 0 errors or
skips and 1721 warnings in 3686.71 seconds. All 50 new cases passed (31 decode,
19 cache). Pytest exited 1; Slurm recorded `FAILED` / `1:0`. Exact JUnit identities,
all 134 fingerprints, Git state and Slurm CUDA visibility matched through full
and final postflight, before these validation-note updates.
The only failure was the unchanged legacy Frost
`test_e8m0_scale_boundary_changed_graph[None-512-0-cute-dsl]`, at prepared graph
capture with `cudaErrorStreamCaptureInvalidated` (901), before numerical checks.
That wrapper stopped before the selected dense 57-case phase. A second default
full run (job `5000428`, the same node/toolchain) repeated **256 passed / 1 failed**,
including all 50 new cases passing, with exact identities and 144 fingerprints
preserved through terminal corroboration; dense57 was not run.
A follow-up on `umb-b200-236` (job `5005300`, the same toolchain) passed default
isolated capture **1/1**, smoke **16/16** and independent dense **57/57**, with
0 failures/errors/skips in those phases. All 185 fingerprints, Git state and Slurm
visibility matched through independent terminal corroboration before repair.
Its separate instrumented full-order prefix failed at case 94: **93 passed / 1
failed**. It records driver `cuLibraryUnload` result 900 in a GC interval during
capture, before launch/end-capture 901; the handle was loaded in the earlier
standalone finite-code FP4 conversion probe. Unknown runtime unload status,
unknown/truncated records and incomplete owner/producer attribution limit this
trace; it is not default full257 qualification. Slurm recorded `FAILED` / `1:0`.
The two standalone conversion probes now unload their loaded CuTe modules at
test teardown instead of deferring cyclic-executor cleanup to later GC. This
is test-resource management only: decoder/shared-kernel arithmetic, global JIT
ownership, numerical gates and default GC/capture policy are unchanged. Six
strict CPU-only ownership regressions passed without accelerator imports or
intercepted calls.
The post-repair default run (job `5008381`, `umb-b200-260`, B200 / DSL 4.8.0 /
bundled CUDA 13.4) passed isolated capture **1/1**, smoke **16/16**, full
**257/257** (decode **143**, cache **48**, native sparse MLA **66**) and independent
dense **57/57**, with 0 failures, errors or skips. The original case-94 capture
failure passed in full-suite order, including setup/call/teardown; all 50 new
ordinary-FP8 cases passed within the full matrix. Full reported 1707 warnings in
3552.03 seconds; dense reported 980 in 1105.53 seconds. All phase exits were 0,
and actual Slurm root/batch/extern records were `COMPLETED` / `0:0`.
Exact ordered identities, all 5047 frozen fingerprints, Git state and Slurm CUDA
visibility matched through final postflight and independent terminal
corroboration, before these validation-note edits. The first frozen audit's four
pre-/post-parse command-record errors and false verdict are preserved. A separate
post-run correction requires exact canonical argv, the literal ini importlib
prepend and full interpreter/module command, retaining every other original
check. Its 166 guarded CPU checks and complete independent corroboration passed;
no frozen input, original report, numerical gate or default GC/capture setting
was changed. See the example README for evidence paths and scope. This qualifies
the tested B200/4.8/13.4 matrix and dense subset, not other hardware/toolchains,
full dense coverage, checkpoint/full-model behavior or exhaustive JIT ownership.
The historical 207/57-case results below belong to the legacy adapter/shared
infrastructure and do not qualify this new mode. Its independent FP64 oracle uses actual stored E4M3
values, the native per-component FP8 probability/output bound, unchanged
`atol=1e-5, rtol=0` LSE gates and exact empty masks/outputs. Global-only retains
strict BF16 gates. Both SWA/global source-precision regressions require FP32
`19/65` before final rounding, rejecting BF16-source-to-FP32 casting.

The new host source-store/adapter/oracle suite passed 125 cases with DSL 4.8.0
and a declared CPU-only `CUTE_DSL_ARCH=sm_100` target, with zero intercepted
CUDA/JIT calls and strict aggregate exit 0. The tested 4.7.0+flashinfer.5 package
also passed all 125 pytest cases, but its eager environment-manager default
attempted `cuInit` twice during import despite the static target. Both calls were
blocked before execution; strict aggregate exit was 1, not a clean guard pass.
The existing host40/API6 matrix has the same distinction. The fixture now guards
raw driver/runtime calls before CuTe imports and fails on caught attempts; its
fresh 4.7 control reports a setup error. Historical Torch/CuTe-only guards did
not prove raw-driver isolation. All case identities, source fingerprints and
CUDA visibility were preserved within each matrix; packages were not patched.
See the example README for exact results/provenance. This is CPU/static checking,
not GPU/JIT qualification or a change to the native wrapper's DSL 4.7 contract.

The concrete `Fp4KvProducer` in `experimental/deepseek_v41/prims_ts_resources.py`
loads only selected global FP4 tiles into the native BF16 D128 shared-memory
stages. It contains no MXFP8 SWA staging and does not unpack the whole cache or
materialize selected dequantized KV in global memory. All 128 loader threads fence their
stores and arrive at an AsyncThread-to-UMMA pipeline; native TMA byte-completion
accounting is not used for these thread-written stages. K stays retained through
QK and delayed PV, using the existing native consumer-release protocol. The
ordinary native TMA producer is unchanged when no custom producer is selected.

Production-source B200 validation on `umbriel-b200-040` with CuTe-DSL 4.8.0 /
bundled CUDA 13.4 and a fresh scratch workspace passed all 10 targeted tests,
with no failures or skips. These cover selection refresh and zero-query tile
counts at global widths 0/64/192/512, including poisoned-selection CUDA Graph
replay, plus two checks that Prims-TS does not compile Frost. Existing independent
FP64 output/LSE gates are unchanged. The earlier global192 failure was traced to
a zero task-cache length despite a live span of 384, truncating runtime task
domains to one BK64 tile; the getter now snapshots current Python state and the
production regression passes. Five native automatic-dispatch regressions also
passed on the same node/toolchain, with no failures, errors or skips. These do
not establish complete native or persistent-family qualification. A separate
`tests/attention/test_prims_ts_sparse_mla.py::test_keep_ab_runtime_task_cache`
pins native BF16 Keep-AB/M64/BK64/retained-KV8/split-1 for static and CLC persistence,
asserts the actual compile configurations and a 768-slot runtime domain, and
reuses the existing FP64 oracle and changed-metadata graph checks. Both GPU cases
ran in the earlier full matrix: CLC passed, while static failed during plan-time
FrontendNext compilation. They do not instrument individual CTA execution counts.
That `--full` run on `umb-b200-250` (Slurm job `4907814`) with CuTe-DSL 4.8.0 /
bundled CUDA 13.4 completed all 207 cases: decode 112 passed, cache 29 passed,
and native sparse MLA 35 passed / 31 failed, with no pytest errors or skips.
All 31 native failures occurred during plan-time FrontendNext/task-scheduling
compilation, before eager accuracy or graph replay checks.
After the Task-interface/lowering repairs below, the complete rerun on
`umb-b200-041` (job `4910237`) with CuTe-DSL 4.8.0 / bundled CUDA 13.4 passed
**207/207**: decode 112, cache 29, native sparse MLA 66. There were no failures,
errors or skips, with 1679 warnings in 3427.86 seconds; pytest exited 0 and Slurm
completed with exit `0:0`. Both pinned static/CLC cases and all eight expanded
writer/decode integration cases passed, preserving exact encoded-byte references
and independent accuracy gates. The exact 207-case node set and all 111 source
fingerprints matched through final postflight, before validation-note updates.
This qualifies only the tested B200/4.8/13.4 matrix, not other GPU/toolchain
combinations or checkpoint/full-model behavior. Benchmarks have not run; no
performance measurements or measured crossovers are claimed.
Validation entry points are
`tests/experimental/test_deepseek_v41_decode.py`,
`tests/experimental/test_deepseek_v41_cache.py` and
`benchmarks/bench_deepseek_v41_decode.py`; see `examples/deepseek_v41/README.md`
for test outcomes, the runnable example, cache ABI and full-public-call timing
scope. The adapter is JIT-only, with no AOT or automatic trace registration.

## CuTe-DSL Task compatibility

Native MLA keeps its CuTe-DSL 4.7 minimum. The 2CTA overrides probe individual
base `Task` method signatures at import time and use constexpr dispatch: DSL 4.7
receives explicit resource context, while 4.8 uses the context stored by `Task`.
The static work queue returns `task_scheduling.resources.WorkTileInfo`, the
canonical exported type in both versions; the legacy scheduler class alone is
not the TS-local type required by 4.8.

For guarded nonpersistent routes, `TaskManager.run()` executes once outside
runtime activity control flow, so task/resource initialization and SMEM/TMEM
synchronization remain unconditional. `MlaDecodeTask.run_body()` guards only data
work. Query-active pruned splits still publish neutral partials and perform the
existing reduction duties; inactive packed-Q tiles publish nothing. Persistent
routes keep the unguarded sentinel and their existing queue skip/progress logic.
G1 reduction uses separate statistics/output coordinates for FrontendNext SSA
lowering. The 2CTA parallel reducer likewise separates warp-only LSE/split locals
from the all-thread output locals. Arithmetic, sink normalization, masked partial
loads and the scale-publication barrier are unchanged. The task-cache getter
remains a plain-Python tracing-time snapshot.

All 40 CPU-only acceptance cases in `tests/attention/test_attention_ts_mla_decode.py`
historically passed separately with DSL 4.7.0 and 4.8.0, with no failures, errors
or skips. They include method capabilities, canonical work-tile identity, graph
task classes/default activity predicates, CLC skip progress, public/plan contracts,
packed-Q offsets, int32 bounds and an independent CPU mask oracle. The recorded
Torch/CuTe guards were not triggered and GPU visibility was unchanged; that
historical scope did not include raw CUDA bindings. The current stricter results
above retain pytest passes but reject the intercepted 4.7 imports. These are not GPU
execution tests; the separate dense D576 GPU tests in that file were not part
of the 207-case matrix above. The allocated-B200 smoke run on
`umb-b200-236` (job `4909337`) completed all 40 cases with 39 passes and one
plan-time `SCOPE_READ_NEVER_SET` failure in the fused 2CTA parallel reducer.
Both pinned persistence cases and all eight selected packed selection/cache
replay cases passed; source fingerprints were unchanged. The follow-on 2CTA
local-name repair passed all three focused GPU regressions on `umb-b200-041`
(job `4910237`), followed by the complete 207-case matrix: all 66 native cases
passed, including both pinned static/CLC runtime-domain regressions, with no
failures, errors or skips and unchanged source fingerprints.

A separate selected dense D576 QK / D512 V subset in
`tests/attention/test_attention_ts_mla_decode.py` passed **57/57** on one allocated
B200 in `umb-b200-250` (job `4936630`), with DSL 4.8.0 / bundled CUDA 13.4.
It covers BF16/FP8 1CTA/2CTA paths, dense FP8 2CTA scheduling, runtime-K pruning,
fully masked split partials, compact/empty/causal queries, non-power-of-two heads,
CLC progress, row-strided page tables and live-metadata CUDA Graph replay.
There were no failures, errors or skips, with 980 warnings in 1063.65 seconds;
pytest exited 0 and Slurm completed with exit `0:0`. All three preceding smoke
cases passed and belong to the same 57-case manifest. Exact case identities and
all 111 source fingerprints matched through final postflight, before validation
notes changed; GPU visibility and independent accuracy gates were unchanged.
GPU qualification is limited to the 207-case matrix and this selected 57-case
subset on B200/DSL 4.8.0/bundled CUDA 13.4, not the entire dense GPU suite or
DSL 4.7 GPU execution. See `examples/deepseek_v41/README.md` for the exact subset
command and the scratch-preflight diagnostic failure before the successful retry.

The 2026-10-13 static adapter/shared-kernel review found no confirmed correctness
defects, including Task context forwarding, initialization/lifecycle duties,
neutral publication and retained packed-stage ownership. It adds no GPU
qualification. Six DS4.1 unknown-backend CPU cases also passed with each of DSL
4.7.0 and 4.8.0 without CUDA/JIT work. The README records the remaining packed
width/conversion coverage and actual older-compiler fallback gaps.

## Native prepared metadata

`SparseMLAPreparedMetadata` contains caller-owned CUDA tensors. Indices already
include physical page strides; attention consumes storage rows directly.

| Field | Shape/type | Meaning |
|---|---|---|
| `indices`, `extra_indices` | int32 `[rows, source_capacity]` | Storage rows; `-1` invalid |
| `lengths`, `extra_lengths` | int32 `[rows]` | Active source prefixes |
| `routes` | int32 `[passes, rows, route_capacity]` | Bit 31 selects the extra source; low 31 bits are its row; `0x7fffffff` invalid |
| `execution_lengths`, `valid_counts` | int32 `[passes, rows]` | Padded execution span and valid entry count |
| `scale_params` | FP32 `[passes, 2 + H + rows]` | QK scale, PV scale, sinks, counts |

Each source span is rounded to 128; route capacity is at least 256. Empty rows
execute one masked slot. Independent scales use two passes. Supply all
representations so metadata remains compatible with automatic dispatch;
direct-index schedules can omit the combined fields. The producer must keep
all representations consistent when indices, lengths, scales or sinks change.

Warm up before CUDA Graph capture. Use stable buffers, preallocated outputs
and `validate=False` during capture; keep all inputs, metadata and workspace
alive through replay. One wrapper/workspace supports one in-flight call.
Buffer overlap is unsupported. The workspace-size helper accepts the same
arguments as `plan`; the eager helper combines planning and execution.

## Native scheduling and validation

Split-KV and V partitions fill small grids. Larger query tiles share gather
work across heads; KV reuse retains K through PV when one CTA owns V and the
buffer budget permits it. BF16 BK64 permits two retained tiles where BK128
would not fit. Selection uses host bounds; live lengths remain on the GPU.
Selected BF16 1CTA schedules use FlashMLA's six-log2 bound; FP8 uses exact
maxima and P scale 448. `assume_valid_prefix=True` promises no holes before each length
and enables analytic masking in the direct FP8 2-CTA path.

DSL 4.7's credit simulator has a bounded iteration count that can report
unconsumed commits before BF16 retained-KV schedules drain. Those schedules
use the stock task manager's warning mode: checks still run, but failures
warn rather than abort compilation. Other schedules retain strict checking.

Accuracy, graph replay, output/workspace ownership and important validation
checks are in one file. Its small Triton preparer adapts vLLM's page-stride
mapping and compaction; its FP64 reference preserves the probability/output
rounding bound used for FP8.

```bash
pytest tests/attention/test_prims_ts_sparse_mla.py
```

Benchmarks are maintained locally, outside this change. One static two-source
BF16 definition remains under `tests/trace/fi_trace_out` as an example;
automatic `fi_trace()` integration is not included.
