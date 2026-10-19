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
`deepseek_v41_window_decode(..., backend="prims-ts")` APIs are separate adapters
for the native D512 task graph. The default remains `"cute-dsl"` (Frost), with
explicit selection and no `"auto"` route or required opt-in environment variable.
Their V4-style interface now takes caller workspace, combined metadata, optional
out and runtime BMM factors, returning only BF16 output. Public plans and tuple
returns are removed. `deepseek_v41_decode_workspace_size(B,K,backend=...)` is a
pure byte-sizing helper, without CUDA or executable-plan creation.

Scope is SM100/H64/Sq1/D512. Prims-TS requires explicit E4M3 Q `[B,1,64,512]`
and typed E4M3 SWA width512, without a side plane. Optional BF16-Q/MXFP8-SWA
compatibility remains removed; non-E4M3 Q rejects before device queries/shared
validation/compilation, without conversion or Frost fallback. Frost retains
BF16 Q and uint8 MXFP8 SWA width528/group32 E8M0 scales. Both accept optional
uint8 FP4 main width288/group16 E4M3. Packed page64 payload rows precede scale
rows. Default HND is `[pages,1,64,width]`, explicit NHD `[pages,64,1,width]`;
singleton-head views change no bytes. All three writer formats retain their
arithmetic, with the same new HND default. Main is not V4's384-byte NVFP4 ABI.

Required combined int32 `[B,128+K]` indices contain128 SWA columns then K main
columns, K=0/64/.../512. Required int32 `[B]` topk lengths include128 fixed SWA
positions; main prefix is `topk-128`. Required seq lengths keep SWA prefix
`[0,clamp(seq_lens,0,128))` as explicit V4.1 policy. The actual V4 H64/Sq1
CAKE routes do not receive seq_lens; default TRTLLM cubin short-sequence behavior
has not been independently qualified. Malformed lengths are clamped and negative,
sentinel and out-of-capacity IDs masked, preserving holes and duplicates. Native
checks raw combined metadata inside attention; Frost writes compact source tables
in prepare. Fixed execution spans and live metadata refresh every call/replay,
without CPU `.item()`.

Both sources must already be in common KV units. Runtime Python-scalar BMM1 is
positive finite FP32; BMM2 is finite FP32. For Q descale q_s, common KV descale
s>0 and attention factor alpha, pass `alpha*q_s*s` and `s`. Native main directly
rounds/saturates exact `E2M1*original_group16_E4M3_scale` to E4M3; it does not
calculate amax/request descales, divide by BMM2, or restore factors twice.
Signed zero/subnormals remain; +/-448 cache-unit saturation is intentional.
Normalize both writer inputs upstream for nonunit s; inconsistent encoded
units are not repaired. Public defaults are1.0/1.0, matching V4: pass
512**-0.5/1.0 explicitly for the former V4.1 attention factor. Runtime scalar
changes need no value-specific compile; captured values require recapture.
This local simplification does not establish official token/block64 quantization
or checkpoint equivalence.

The adapter opts into private `sparse_split_softmax_scale`, default-false in
shared native configuration and restricted to its fixed direct/static E4M3
Keep-AB/M64/BK128/D512/split1/nonpersistent topology. Maxima remain raw scores;
explicit non-FTZ RN operations scale differences by runtime BMM1 before log2
conversion. Each masked probability is zeroed before both FP32 summation and
FP8 packing, preventing tiny positive BMM1 from reviving the finite mask
sentinel. Natural-domain sink FMA and direct natural LSE avoid precomputing an
overflowing `BMM1*log2(e)`. Generic native arithmetic is unchanged. Frost uses
analogous difference-first arithmetic and raw-max/sum split statistics instead
of potentially overflowed scaled partial LSE; its policy/schedule is unchanged.

**The native wrapper's packed-input restriction is unchanged.**
`BatchSparseMLADecodePagedTSWrapper` still accepts matching native BF16 or E4M3
Q/KV, not the adapter's uint8 MXFP8/FP4 pools. Its existing DSL4.7 requirement
and public metadata/scale behavior remain unchanged.

The V4.1 adapter requires **CuTe-DSL >=4.8 with its bundled CUDA13 compiler**;
system CUDA13 alone is insufficient. Fixed Keep-AB/M64/BK128/retained-KV8/split1/
nonpersistent execution uses one joint E4M3 Q/K/V/P attention with inline raw
combined metadata checks and original optional sinks, without native prepare.
The producer-owned binder returns the same bounded physical-ID predicates to
KV gather and softmax, before task-manager construction; generic prepared-list
binding remains unchanged. A fixed SWA HEAD/main LOOP domain includes a masked
K0 main task, without reading a nonexistent combined-table column. One online
softmax/PV state applies P448 compensation and sink mass once before final BF16
output. Internal FP32 natural-log LSE excludes sinks; empty O=0/LSE=-inf. There
are no per-source O/LSE buffers, main scale tensor or merge launch. Native scratch
contains only LSE and the K0 dummy. Mixed/window target one launch, requiring
actual GPU-node validation; Frost retains prepare and its attention scheduling
(two unsplit/three split launches). No performance benefit is claimed.

`MixedFp8KvProducer` / `MixedFp8SmemKvResource` in `prims_ts_resources.py` use
physical `(main,SWA)` pools but logical primary SWA/extra main routes. Fixed128
SWA positions give one SWA HEAD even when empty; subsequent LOOP tiles load main.
Aligned16-byte SWA stores use native FP8 SW128, and exact selected FP4 converts
directly to the same E4M3 stages. Both use per-writer async-proxy fences and all128
writer arrivals before AsyncThread-to-UMMA publication. Adapter SWA intentionally
uses thread loads, not TMA, avoiding mixed completion protocols. Q TMA and the
generic native TMA producer are unchanged. Int64 widening precedes byte strides;
invalid rows read no payload and stage zeros. K remains retained through delayed
PV. There is no BF16 Q expansion or whole converted cache. BF16-named resource
bases retain exact FP4 register conversion/selected loads, not BF16-Q support.

Generic internal source-store/request-scale hooks remain available and tested
independently. The nonfinalizing 1CTA source store publishes normalized FP32 O
and sink-free log2 LSE without another448 factor; it is not a public FP32 output
mode or compatible with output quantization. Generic request factors and
`MergeMlaSourceStates` behavior are unchanged; this adapter no longer uses them.
The JIT namespace remains `deepseek_v41_prims_ts_fp8` with a versioned
`shared_pertensor_joint` signature and the new source closure.

Workspace is caller-owned contiguous int8/uint8,32-byte aligned and sufficiently
large; undersizing never allocates a fallback. Frost split scratch is
`B*64*split*514*4` bytes for FP32 partial O/raw maximum/sum. Out may allocate eagerly, but
capture requires supplied out and warm compiled declarations. Keep captured
pointers alive/fixed and update tensors in place. Concurrent calls need separate
workspace/out or synchronization; workspace/out cannot overlap inputs or each
other. Read-only inputs may alias. Caches hold no caller GPU buffers.

Independent FP64 references mask/gather, decode exact scaled E2M1 and independently
clamp/round to common-unit E4M3, then apply shared BMM factors once. The native
componentwise FP8 bound, strict sink-exclusive LSE `atol=1e-5,rtol=0` and exact
empty gates remain unchanged, including main-only. Frost values/references/gates
are unchanged. Historical records below do not qualify this joint/shared-scale/
API/HND-default revision; examples and benchmarks have not run.

The lowering retry (2026-10-19, job5123805 on `umbriel-b200-091`, B200/148SMs,
DSL4.8/compiler13.4) passed fresh strict CPU263 and isolated Frost capture1/1.
Its24 passing smoke cases include four production-stage byte probes, both
conversion-branch rounding replays, both >2GiB address cases, all eight actual
single-kernel graphs and all eight live-metadata replay cases. The graph cases
cover K0/64/192/512 with/without sinks; live metadata covers both backends at
those widths. Eight preserved DOTs each show one finalizing attention/no prepare/
zero edges; these cases passed unchanged output/LSE checks after real lowering.
The next Frost K0 shared-BMM/layout case failed strict LSE:3/128 mismatches,
max absolute1.0912699504928014e-5 versus1e-5/rtol0. Smoke ended24 passed/1 failed,
10 unexecuted; allocated host/full/dense were prerequisite-skipped. Root FAILED/
1:0 and all original GPU qualification fields remain false. Separate provenance-
only corroboration matched10168 frozen inputs/Git, strict CPU263, exact interrupted
reports/JUnit, eight graph DOTs and root-only accounting with0 errors; it does not
make the failed run pass or qualify later Frost repairs. Root750 GPU-seconds
brings cumulative use to7318,3482 remaining of10800. See the example README for
sealed evidence. Examples/benchmarks have not run; no performance claim is made.

The first inline-metadata GPU attempt (2026-10-19, job5122562 on B200/148SMs,
DSL4.8/compiler13.4) passed isolated Frost capture1/1 and all four real producer-
stage byte probes. The next smoke case failed during attention compilation with
`BOUNDARY_META_LOOP_CARRY` for `scale_softmax_log2`, before numerical checks:
4 passed/1 failed,30 smoke cases unexecuted; allocated host/full/dense were
prerequisite-skipped. Root FAILED/1:0 is preserved. Independent source/terminal
corroboration matched all10167 frozen inputs/Git and the strict CPU259 prerequisite;
it does not turn the failed GPU acceptance audit into a pass. Native single-kernel
graphs, live metadata replay and attention numerics remain unqualified, including
the follow-up lowering repair. That repair materializes split-mode `LOG2_E` as
FP32 SSA before constructing tasks: `Float32(Float32(LOG2_E).ir_value())` avoids
the Python-double/rounded-FP32 loop-carry comparison without changing arithmetic
or raw runtime scales. Host mock regressions are not CuTe lowering acceptance.
See the example README for exact evidence and scope.

The sealed pre-fusion removal baseline passed full221 in job5116654, but the
job remains FAILED/1:0 after dense13 passed/44 plan-time `/tmp` ENOSPC failures.
Fresh dense-only5118181 (B200/148SMs,DSL4.8/compiler13.4) passed57/57 with980
warnings and COMPLETED/0:0. Its independent dense qualification is true, whole-
regression qualification false;6617 frozen fingerprints/identities/reports/Git/
visibility matched, and terminal corroboration passed. It does not relabel the
original job or qualify fusion. See the example README for exact evidence.

The following dated records preserve their original checkpoint-time status,
not current fusion qualification. The later removal GPU baseline is above.

Historical post-removal CPU/static acceptance (2026-10-18) passed **382/382** cases: 262
source-state/resource/adapter/oracle, 7 conversion-lifetime, 67 benchmark/example
defaults, 40 native host and 6 unknown-backend contracts. The fresh `-I -B -S`
Python 3.12.3 / Torch 2.11.0+cu128 process used DSL 4.8.0 / bundled compiler
metadata 13.4, static `sm_100`, thread caps and disabled plugin autoload/JIT.
Pytest/strict exits were 0, with no failures/errors/skips/intercepted calls and
uninitialized Torch CUDA. Independent postflight matched exact ordered JUnit
identities, all 1146 passing setup/call/teardown reports, all 156 fingerprints,
Git state, inherited visibility and default GC before validation-note edits.
The tested patch is preserved; focused Ruff/syntax/whitespace checks passed.
This is host/static qualification only, not CuTe lowering, GPU numerics or
performance. GPU/example/benchmark execution has not run for the removal; see
the example README for evidence.

**Pre-removal validation boundary:** all dated records below, including the
request-scaled full271/dense57 and 413-case CPU snapshot preserved in
`d27624a74`, predate optional MXFP8 Prims-TS removal. They do not qualify the
removal or its revised test manifest. Native kernels/ABI and Frost arithmetic
are unchanged; GPU/example/benchmark execution has not been repeated for it.

Request-scaled GPU validation (2026-10-18) passed on one allocated B200 / 148 SMs
in `umb-b200-237` (job `5103699`, DSL 4.8.0 / bundled compiler 13.4): isolated
capture **1/1**, smoke **32/32**, full **271/271** (157 decode / 48 cache / 66 native
sparse MLA) and independent selected dense **57/57**, with no failures, errors or
skips. Both real-stage conversion branches, distinct request scales, FP32 division
and K/V restoration witnesses, zero/empty transitions, all-width changed-data
graph replay and writer/decode integration passed. All phase exits were 0 and
Slurm root/batch/extern completed with `0:0`. The independent report has
`default_gpu_qualification=true`, with exact ordered JUnit identities, complete
setup/call/teardown and all 6568 fingerprints/Git/visibility matching before
validation-note edits. Numerical gates and default GC/capture settings were
unchanged. The root allocation used 1h36m07s (5767 GPU-seconds) of the fresh
assistant-imposed three-GPU-hour cap. This qualifies only the tested request-scaled
B200/4.8/13.4 snapshot and selected dense subset, not the entire dense suite, other
hardware/toolchains or checkpoint/full-model behavior. Examples and benchmarks
have not run; no performance claim is made. See the example README for evidence.

Separate CPU/static acceptance (2026-10-18) passed **413/413** cases: 219 native source-state/
resource/adapter/oracle, 41 optional-MXFP8, 7 conversion-lifetime, 100 benchmark/
example-default, 40 native host and 6 unknown-backend contracts. The fresh
`-I -B -S` Python 3.12.3 / Torch 2.11.0+cu128 process used the existing DSL 4.8.0
overlay / bundled compiler metadata 13.4, static `sm_100`, thread caps and disabled
plugin autoload/JIT. Pytest/strict exits were 0, with no failures/errors/skips or
intercepted calls and uninitialized Torch CUDA. Independent postflight matched
exact ordered JUnit identities, complete setup/call/teardown reports, all 158
fingerprints, Git state and visibility before validation-note edits. The exact
tested patch is preserved; 28 static compatibility/arithmetic-witness checks and
focused Ruff/syntax/whitespace checks also passed. These qualify host/static
contracts, not CuTe lowering or GPU numerics; see the example README for evidence.

The following historical CPU/GPU records qualify the earlier unit-scale FP8-main
snapshot only, not the new amax, normalization or scale restoration.

Historical unit-scale FP8-main CPU/static acceptance (2026-10-16) passed **321/321** cases:
146 native source-state/resource/adapter/oracle, 37 optional-MXFP8, 7 conversion
lifetime, 85 benchmark/example defaults, 40 native host and 6 unknown-backend
contracts. The fresh `-I -B -S` Python 3.12.3 / Torch 2.11.0+cu128 process used the
existing DSL 4.8.0 overlay / bundled compiler metadata 13.4, static `sm_100`,
thread caps and disabled plugin autoload/JIT. Pytest/strict exits were 0, with
zero failures/errors/skips/intercepted calls and uninitialized Torch CUDA.
The latest rerun includes the real-stage probe's flat rank-one tuple-layout
regression. Exact ordered JUnit identities, all 154 fingerprints, Git state and
CUDA visibility matched independent postflight, before validation-note edits.
These are guarded host/static contracts, not GPU/JIT qualification; see the example
README for records and runnable allocated-GPU commands.

Historical unit-scale FP8-main GPU validation (2026-10-16) passed on one allocated B200 / 148 SMs
in `umb-b200-263` (job `5049128`, DSL 4.8.0 / bundled compiler 13.4): isolated
capture **1/1**, smoke **27/27**, full **271/271** (157 decode / 48 cache / 66 native
sparse MLA) and independent selected dense **57/57**, with no failures, errors or
skips. Both real-stage conversion branches and analytic rounding/saturation/
underflow graph replay passed, as did all global widths `0/64/.../512`.
Independent corroboration matched exact ordered JUnit identities, setup/call/
teardown reports, all 5871 frozen fingerprints, Git state and scheduler visibility
before validation-note edits; numerical gates and default GC/capture settings
were unchanged. Every phase exited 0 and Slurm root/batch/extern completed `0:0`.
The allocation used 1h36m04s of the three-GPU-hour budget. This qualifies only the
tested B200/4.8/13.4 unit-scale snapshot and selected dense subset, not request
scaling, other hardware/toolchains, the entire dense suite or checkpoint/full-model behavior. Examples
and benchmarks have not run; no performance claim is made. See the example README
for the independent report and terminal evidence.

The remaining validation records below describe pre-FP8-main source snapshots,
including the original pre-reorganization GPU runs. The historical 257-case GPU
and later 299-case CPU acceptances do not qualify current main conversion/staging/
attention; the unit-scale GPU run above also predates request scaling.

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
only the tested pre-FP8-main B200/4.8/13.4 snapshot and dense subset, not current
FP8-main staging/attention, other hardware/toolchains, full dense coverage,
checkpoint/full-model behavior or exhaustive JIT ownership. That snapshot used
FP8 SWA but BF16 main attention and exact BF16 Q expansion.
The historical 207/57-case results below belong to the legacy adapter/shared
infrastructure and additionally predate ordinary FP8 SWA. The pre-FP8-main FP64
oracle used actual stored E4M3 values, the native per-component FP8
probability/output bound, unchanged `atol=1e-5, rtol=0` LSE gates and exact empty
masks/outputs; global-only then retained strict BF16 gates. Current native
main-only uses the E4M3 reference/bound instead. Both SWA/global source-precision
regressions require FP32 `19/65` before final rounding, rejecting
BF16-source-to-FP32 casting.

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

The current `MixedFp8KvProducer` / `MixedFp8SmemKvResource` specialization in
`experimental/deepseek_v41/prims_ts_resources.py` reuses the `Fp4Fp8` common-unit
conversion bases and copies typed SWA bytes into the same D128 FP8 SW128 stages.
Selected main applies original group16 E4M3 scales and direct RTNE/saturation,
without request normalization. BF16-named bases provide exact register conversion
and selected loads, not BF16 KV staging or Q expansion. No producer restores
MXFP8-SWA support, unpacks the whole cache or materializes converted KV globally.
All128 loaders fence their stores and arrive at the AsyncThread-to-UMMA pipeline;
thread stages do not use native TMA byte completion. K stays retained through
QK/delayed PV via the existing consumer-release protocol. The ordinary native
TMA producer is unchanged when no custom producer is selected.

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
