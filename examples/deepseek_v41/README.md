# Experimental DeepSeek V4.1 decode

The mixed-cache and window-only decode APIs offer two explicitly selected,
experimental backends on SM100:

| `backend` | Implementation |
|---|---|
| `"cute-dsl"` (default) | Existing **Frost** CuTe-DSL decode |
| `"prims-ts"` | Native D512 adapter: explicit FP8 Q + ordinary FP8 SWA |

Calling these experimental APIs is the opt-in; no environment variable is
required. There is no `"auto"` selector. Both backends are JIT-only. The existing
`"cute-dsl"` default is unchanged, but the public interface now follows the local
V4 caller-workspace/output-only convention. No public plan is constructed.

Prims-TS accepts only native E4M3 Q/SWA. The optional BF16-Q/MXFP8-SWA
compatibility implementation and `--prims-ts-mode` CLI option remain removed;
non-E4M3 Q is rejected before shared validation, device queries or compilation,
without implicit conversion or fallback to Frost. The generic sparse MLA
wrapper's native BF16/E4M3 contract is not broadened.

**Current validation boundary:** the joint shared-scale attention, new public
API, caller workspace and HND-default writer are a new revision. Historical
CPU/GPU results below qualify only their frozen snapshots, not this revision.
The revised example and benchmark have not run; no performance or
checkpoint/full-model qualification is claimed. The older request-scaled
snapshot in `d27624a74` and the post-removal baseline use different arithmetic
and an older public API.

## V4-style caller interface

```python
workspace = torch.empty(
    ds.deepseek_v41_decode_workspace_size(B, K, backend=backend),
    device=q.device, dtype=torch.uint8,
)
out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
result = ds.deepseek_v41_decode(
    q, swa, workspace,
    sparse_indices=indices, compressed_kv_cache=main,
    sparse_topk_lens=topk_lens, seq_lens=seq_lens, out=out,
    bmm1_scale=512**-0.5, bmm2_scale=1.0, sinks=sinks,
    kv_layout="HND", backend=backend,
)
assert result is out
```

- `query` is `[B,1,64,512]`. Return only BF16 output with Q's shape; supplied
  `out` is returned by identity. Public `(out,lse,plan)` returns and `plan=`
  replay are removed. Internal FP32 LSE remains sink-exclusive natural log and
  is checked through private workspace views, not exposed as a public result.
- `sparse_indices` is required contiguous CUDA int32 `[B,128+K]`, with
  K=0/64/.../512. The first128 columns are SWA physical slots; the remaining K
  are main slots. K>0 requires `compressed_kv_cache`.
- `sparse_topk_lens` and `seq_lens` are required CUDA int32 `[B]`. Valid topk
  lengths are `[128,128+K]`; the active main prefix is `topk-128`, not
  `topk-valid_swa_count`. SWA uses the prefix `[0,clamp(seq_lens,0,128))`.
  This is explicit V4.1 policy: actual V4 H64/Sq1 CAKE routes do not receive
  seq_lens, and default TRTLLM cubin short-sequence behavior has not been
  independently qualified.
  Malformed live lengths are clamped safely on device, without CPU `.item()`.
- Negative, reserved-sentinel and out-of-capacity IDs become holes. Duplicates
  retain multiplicity; holes are not compacted. Every valid slot must be
  initialized and causally visible. Native consumes the original combined table
  inline; Frost refreshes compact sanitized tables on every call/replay. Neither
  needs hidden copies of noncompact combined-table slices.
- Default `kv_layout="HND"` is `[pages,1,64,width]`; explicit `"NHD"` is
  `[pages,64,1,width]`. Singleton-head normalization is zero-copy, with identical
  packed storage bytes. The optional writer has the same layout argument and
  HND default. Main remains the width288 page-plane format, not V4's width384
  NVFP4 cache.
- `deepseek_v41_window_decode` uses the same workspace/out/scales/layout rules,
  required `[B,128]` indices and topk lengths128, without a main cache.
- The sizing helper is pure arithmetic. Workspace must be contiguous int8 or
  uint8, 32-byte aligned and sufficiently large; there is no allocation fallback.
  Native contains only internal LSE and a K0 unread dummy. Frost additionally
  retains sanitized metadata, sink staging and any split scratch. Frost uses
  `B*64*split*514*4` scratch bytes:
  FP32 partial O plus raw-dot maximum and denominator sum, not a scaled partial
  LSE. The split-selection policy is unchanged. Cached factories retain kernels/
  host facts, not caller buffers or workspace-address registries.
- `out=None` may allocate eagerly. Capture requires supplied out, sufficient
  workspace and eager warmup for the same backend/declaration. Keep all captured
  addresses alive and fixed and update values in place. Concurrent calls require
  separate workspace/out or caller synchronization. Workspace/out cannot overlap
  inputs or each other; read-only inputs may alias.

## Common-unit E4M3 attention

| Mode | Q `[B,1,64,512]` | SWA (HND default) | Arithmetic |
|---|---|---|---|
| Prims-TS | `torch.float8_e4m3fn` | Same typed E4M3 `[pages,1,64,512]`, no side plane | Joint E4M3 Q/K/V/P for SWA/main |
| Frost (public default) | BF16 | uint8 `[pages,1,64,528]`, group32 E8M0 scale plane | Selected tiles staged as BF16 |

Native FP8 mode requires **explicit FP8 Q from the caller**. MXFP8 cannot be
reinterpreted by discarding its scales. Both caches must already be encoded in
common KV units. With Q descale `q_s`, shared KV descale `s>0` and attention
factor `alpha`, pass `bmm1_scale=alpha*q_s*s` and `bmm2_scale=s`:

```text
u_s = stored E4M3 SWA
u_c = E4M3_RN_SATFINITE(exact_E2M1(code) * original_group16_E4M3_scale)
z_j = bmm1_scale * dot(stored_Q, u_j)
O = bmm2_scale * sum_j(exp(z_j) * u_j) / (sum_j(exp(z_j)) + exp(sink))
```

BMM1 must be positive finite FP32; BMM2 finite FP32. Both are Python scalar
values, not device tensors. They remain runtime arguments, so eager scalar
changes do not require another compilation. CUDA Graphs capture their values;
changing factors requires recapture. V4's defaults are **1.0/1.0**, not the former
implicit `512**-0.5/1.0`; pass the latter pair explicitly to retain that math.

Maxima stay in raw-dot units. Both backends multiply score differences by BMM1
before log2 conversion with explicit non-FTZ RN operations, rather than forming
`BMM1*log2(e)` first. Native zeros each masked probability before its FP32 sum
and FP8 pack, so even tiny positive BMM1 does not revive finite mask sentinels.
This uses a private, default-false `sparse_split_softmax_scale` mode restricted
to this adapter's fixed topology; generic native arithmetic is unchanged.
Sink subtraction uses natural-domain FMA; internal LSE is formed directly in
natural units. Frost merges raw maxima and sums instead of scaled partial LSE.
A true LSE outside FP32 range may legitimately publish infinity internally,
without making the output normalization depend on that overflowed value.

There is **no decode-time KV amax, request-scale tensor, normalization division
or repeated restoration**. Main staging does not divide by BMM2. Finite negative
original group scales, signed zeros and subnormals are retained. Main values
outside +/-448 **cache units** intentionally saturate, corresponding to +/-448s
physical units for positive common descale s. The old request-selected dynamic
range protection is removed. Valid zeros remain attention positions.

For nonunit s, normalize **both** writer inputs upstream before quantization;
neither backend repairs inconsistent externally encoded units. The three writer
formats and group-scale arithmetic are unchanged. `swa_fp8` clamps to
[-448,448] then rounds to nearest-even E4M3 without a side plane; `swa_mxfp8`
retains group32 E8M0 and `main_kv_fp4` group16 E4M3. This shared-per-tensor local
simplification aligns the local V4 interface, not official model quantization:
the inspected DeepSeek-V4 reference separately quantizes token/block64 sources.
It is not a checkpoint-equivalence claim.

### One joint native attention, inline metadata

Prims-TS requires **CuTe-DSL >= 4.8 with its bundled CUDA 13 compiler** and fixes
Keep-AB/M64/BK128/D128/retained8/V1/split1/nonpersistent execution. It passes the
original combined indices, seq/topk lengths and optional sinks into attention,
without a standalone native prepare. A producer-owned private view checks live
prefixes and physical IDs identically for KV gather and probability masking.
The table reads themselves are bounded, including K64 tails and the masked K0
main LOOP, while the fixed SWA HEAD/main LOOP scheduling is unchanged. Native
workspace no longer materializes selections, lengths, counts or effective sinks.

This follows the default SM100 DS v4 host execution model: original metadata is
passed to attention; its optional strided-cache index remap is not sanitation.
It does not establish equivalence to unpublished cubin checks for malformed inputs,
or to H64/Sq1 CAKE short sequences. V4.1 retains its explicit defensive policy.

A single finalizing native attention shares online softmax, FP32 PV accumulation,
P448 compensation and sink normalization, then rounds final O once to BF16.
Internal natural-log LSE excludes sinks; empty O is exactly zero and LSE `-inf`.
There are no per-source O/LSE buffers or merge launch. Both mixed and window-only
target **one kernel launch**; actual graph-node topology still requires GPU
validation. Frost retains its preparation, attention scheduling and represented
BF16 values: target two launches unsplit, three split. No speedup is inferred.

`MixedFp8KvProducer` / `MixedFp8SmemKvResource` stage SWA bytes in the single
SWA HEAD and exact FP4-to-E4M3 main in subsequent LOOP tiles. Fixed128 SWA span
preserves this even when SWA is entirely invalid. The adapter deliberately uses
aligned16-byte thread loads/stores for SWA, not TMA, to keep one all-128-writer
AsyncThread-to-UMMA protocol with per-writer async-proxy fences. Native Q TMA and
the generic native TMA producer are unchanged. SW128 byte offsets are
`(row*128+dim)^((row&7)*16)`; Int64 address arithmetic precedes slot-byte strides.
K stages remain retained until delayed PV releases them. Only selected tiles
are converted; there is no full converted cache or BF16 Q scratch.

The BF16-named `Fp4KvProducer` / `Fp4SmemKvResource` bases still provide exact
FP4 register conversion and selected loads; they do not enable BF16 Q/MXFP8
Prims-TS. Generic source-state/request-scale hooks and tests remain independent
of this adapter; the adapter no longer uses them.

The independent native oracle masks/gathers packed main bytes, reconstructs
exact scaled E2M1, then independently clamps/rounds to E4M3 before FP64 joint
attention. It does not call production conversion. All native modes keep the
existing componentwise probability-rounding bound:

```text
(2**-4 + 3*2**-8 + 1e-5) * sensitivity
  + (2**-10 / 448) * underflow + 1e-6
```

`sensitivity = abs(bmm2_scale)*sum(weights*abs(u))` and
`underflow = abs(bmm2_scale)*exp(max(logits)-normalizer)*sum(abs(u))`, including
sink mass in the normalizer. Bound constants, exact empty checks and strict
sink-exclusive LSE `atol=1e-5, rtol=0` are unchanged. Frost keeps decoded BF16
values, independent FP64 references and strict relative-L2/max-scaled gates.

**Historical, pre-FP8-main:** post-repair default full257 and independent dense57
passed on B200 / DSL 4.8.0 / bundled CUDA 13.4 (job `5008381`). All 50 then-new
ordinary-FP8 cases passed within the full matrix, including writer/decode graph
replay; the previously failing legacy Frost capture case passed in full-suite
order after a test-local conversion-module lifetime repair, without numerical-gate
or default GC/capture changes. That snapshot used FP8 SWA but BF16 main attention.
Its tests covered all global widths `0/64/.../512`, source-only/empty/poisoned
selections, changed-data graph replay, exact BF16 query expansion and FP32 source
precision (`19/65`, not BF16 partials cast back to FP32). These historical results,
and the later 299-case CPU acceptance, do **not** qualify FP8 main attention. No
benchmark or performance claim is made for either mode.

The guarded host-only source-store/adapter/oracle suite excludes the root CUDA
collection probes. Use the thread caps before importing PyTorch:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
FLASHINFER_DISABLE_JIT=1 CUTE_DSL_ARCH=sm_100 \
python -m pytest --confcutdir=tests/experimental \
  tests/experimental/test_deepseek_v41_source_state_cpu.py \
  tests/experimental/test_deepseek_v41_workspace_cpu.py -v
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

### Inline-metadata lowering retry (2026-10-19)

This retry tests the FP32-SSA softmax-lowering repair below, before any later
Frost precision repair. It provides partial passing evidence, **not successful
whole-GPU acceptance**.

- Fresh strict CPU acceptance passed **263/263**: workspace99, defaults74,
  conversion-lifetime7 and bounded sparse-view83. All789 ordered setup/call/
  teardown reports passed, including four new split-scale materialization
  variants. The unchanged guard ran before Torch/pytest; there was no CuTe import
  or CUDA/JIT attempt, and Torch CUDA remained uninitialized. Independent CPU and
  preallocation full-byte audits passed with zero evidence errors.
- Job `5123805` ran on `umbriel-b200-091`, one B200/148SMs, DSL4.8.0 and bundled
  compiler13.4. Fresh collection independently confirmed253 GPU cases,57 selected
  dense cases,267 allocated host/static cases and35 mandatory smoke cases.
  Isolated legacy Frost capture passed **1/1**. Smoke passed **24 cases**:
  all four production-stage byte probes, both native conversion-branch rounding/
  saturation/underflow replays, both actual >2GiB cache-addressing cases, all
  **eight actual single-kernel CUDA Graph checks** and all **eight live combined
  metadata replay cases**. The latter two groups cover K0/64/192/512; graph checks
  cover sink/no-sink, while metadata replay covers both native and Frost. The
  eight preserved DOTs each have one finalizing `dense_kernel`, no prepare and
  zero edges. These cases include their unchanged independent output/LSE checks;
  native attention lowered and executed past the previous loop-carry error.
- The next case, `test_shared_bmm_scales_and_layout_parity[0-cute-dsl]`, failed
  Frost's strict sink-exclusive LSE gate:3/128 values differed, greatest absolute
  difference `1.0912699504928014e-5` against `atol=1e-5,rtol=0`. Smoke therefore
  ended **24 passed / 1 failed / 10 unexecuted** (188 warnings, pytest235.10s).
  Allocated host/full/dense were prerequisite-skipped, not passing regressions.
- Postflight passed; the root remains **FAILED / 1:0**. The original
  `gpu-audit.json` remains false with seven smoke nonpassing/short-prefix errors;
  all mandatory-smoke/full/dense/whole-GPU qualification fields remain false.
  Separate provenance-only `source-terminal-corroboration.json` passed with zero
  errors: all10168 frozen source/native/runtime inputs and Git, the identical
  strict CPU263 prerequisite, the exact interrupted JUnit/report prefix, all eight
  graph DOTs and root-only terminal accounting matched. It does not relabel the
  failed acceptance audit. Numerical gates, visibility and default GC/capture
  were unchanged, and the exact tested tracked patch/untracked sources are sealed.

Evidence is under
`/home/scratch.linquanh_gpu/tmp/dsv41-inline-softmax-retry.sm8oelma/`, with
`snapshots/pure-cpu-v1/` and `snapshots/gpu-final-v1/`. The GPU freeze SHA256 is
`f73dfc2c80b0b1c02e461eb3bfdd8a6bb73e9ca158b64a15d4e119ac7e4e0b33`.
Root750 GPU-seconds brings cumulative usage to7318 (4807+1399+362+750),3482
remaining of10800; batch/extern are not added. Allocation/pytest durations are
not kernel-performance measurements. Examples/benchmarks have not run; broader
regression and checkpoint/full-model qualification are not established.

### Inline-metadata validation attempt (2026-10-19)

These records describe the frozen snapshot **before the softmax-lowering repair**,
not acceptance of later edits.

- Strict CPU acceptance passed **259/259**: workspace99, defaults70,
  conversion-lifetime7 and bounded sparse-view83. The fresh `-I -B -S` process
  installed the unchanged guard before Torch/pytest; no CuTe import, CUDA/JIT
  attempt, failure, error or skip occurred, and Torch CUDA stayed uninitialized.
  Independent checking matched all777 passing setup/call/teardown reports,
  ordered identities, all10167 frozen fingerprints and Git state. This four-file
  scope excludes the CuTe-dependent source-state/native-host/API matrix.
- Job `5122562` ran on `umbriel-b200-034`, one B200/148SMs with DSL4.8.0 and bundled
  compiler13.4. Fresh collection found253 GPU cases,57 selected dense cases and267
  allocated host/static cases; the priority smoke subset contained35 cases.
  The isolated legacy Frost capture regression passed **1/1**. All **four** real
  producer-stage byte probes passed, covering main/SWA and both conversion
  branches. The fifth smoke case,
  `test_fp8_main_common_units_rounding_saturation_underflow_changed_graph[False]`,
  failed during attention compilation with `BOUNDARY_META_LOOP_CARRY` for the
  Python-valued `scale_softmax_log2`, before attention execution or numerical
  checks. Smoke ended **4 passed / 1 failed**; the other30 smoke cases and the
  allocated host/full/dense phases were not executed after the prerequisite
  failure. In particular, native single-kernel topology, live metadata replay,
  >2GiB accesses and attention output/LSE remain unqualified.
- Postflight passed, but the root job remains **FAILED / 1:0**. The original
  `gpu-audit.json` remains false with its seven smoke nonpassing/short-prefix
  errors. A separate `source-terminal-corroboration.json` independently rehashed
  all10167 source/native/runtime inputs, Git and the strict CPU prerequisite,
  and corroborated terminal accounting without changing that acceptance verdict.
  Numerical gates, CUDA visibility and default GC/capture settings were unchanged.
  The exact tested patch and untracked sources are preserved.

Evidence is under
`/home/scratch.linquanh_gpu/tmp/dsv41-inline-metadata-validation.2SbZIKNp/`:
`snapshots/pure-cpu-v1/` and `snapshots/gpu-final-v2/` contain manifests, complete
reports/JUnit, frozen helpers, preserved sources and audit/accounting records.
The earlier held job5122317 was cancelled without starting after a login-host
parser rejected Slurm's exact-one-node `NumNodes=1-1`; its original
`snapshots/gpu-final-v1/` is unchanged and it used0 GPU-seconds.
Job5122562 used362 root GPU-seconds: cumulative6568 (4807+1399+362),4232 remaining
of10800, without adding batch/extern time. These are allocation durations, not
kernel-performance measurements. Examples/benchmarks have not run.

The follow-up lowering repair materializes the private split-mode `LOG2_E`
constant as FP32 SSA before task construction, using
`Float32(Float32(LOG2_E).ir_value())`. This prevents a Python-double literal from
being compared with its rounded FP32 initializer by the loop-carry audit; it
changes no softmax operation or runtime BMM1/output scale. Four added host mock
cases cover split on/off and supplied/default scales, not actual CuTe lowering.
The repair needs fresh validation; neither the failed attempt nor the earlier
CPU snapshot qualifies it.

### Pre-fusion removal GPU baseline (2026-10-19)

These records were sealed **before** shared-scale/fusion/API source edits. They
qualify the removed-compatibility/request-scaled baseline only, not the code
currently described above.

- Job `5116654` on `umb-b200-247` passed isolated capture1/1, smoke44/44 and
  full221/221 (111 decode / 44 cache / 66 native sparse MLA). Its dense phase
  passed13 and failed44 at plan-time temporary-file creation with
  `OSError: [Errno 28] No space left on device`; those failures did not reach
  numerical checks. The root/batch job remains **FAILED / 1:0**, and its original
  `default_gpu_qualification=false` conclusion is unchanged. The evidence does
  not establish whether shared `/tmp` exhausted bytes or inodes, or its owner.
- A fresh selected-dense-only rerun, job `5118181` on `umb-b200-239`, passed
  **57/57**, with no failures/errors/skips and980 warnings (pytest1244.25s).
  Root/batch/extern completed with `0:0`; root elapsed1399s. The independent audit
  matched exact ordered JUnit identities, complete reports and all6617 frozen
  fingerprints, Git state and inherited visibility. It reports
  `default_dense_qualification=true`, `default_gpu_qualification=false`,
  `errors=[]`; terminal corroboration also passed. The false whole-regression
  field is intentional for this dense-only scope, not a dense failure.

Both used one B200/148SMs, DSL4.8.0 and bundled compiler13.4. The dense rerun
never relabels the original job successful or creates a single clean full-job
result. Numerical gates and default GC/capture settings were unchanged. Evidence:
`/home/scratch.linquanh_gpu/tmp/dsv41-drop-mxfp8-b200.1JqL6FtS/` and
`/home/scratch.linquanh_gpu/tmp/dsv41-drop-mxfp8-dense-b200.xNnJqDQq/`
(`independent-audit.json`, `terminal-receipt.json`). The cumulative ledger records
6206 root GPU-seconds used (4807+1399),4594 remaining of10800; batch/extern are
not double-counted. These durations are not kernel-performance measurements.

### Historical post-removal CPU/static acceptance (2026-10-18, pre-fusion)

After removing optional BF16-Q/MXFP8-SWA Prims-TS, a fresh exact manifest passed
**382/382 CPU cases**:

| Suite | Passed |
|---|---:|
| Native source-state/resource/adapter/independent-oracle contracts | 262 |
| Conversion-module lifetime contracts | 7 |
| Benchmark/example defaults and extracted reference contracts | 67 |
| Native dense wrapper host contracts | 40 |
| Unknown-backend public API contracts | 6 |

Coverage includes early non-E4M3 rejection through both public decode entry
points, native plan/signature forwarding and foreign-plan rejection, retained
FP4 factory/pipeline/fence contracts, removed CLI-flag rejection, native-only
source provenance, and unchanged request-scale references and numerical gates.
The deleted compatibility suite is not part of this manifest; historical
413-case CPU and 271-case GPU manifests are not reused.

The fresh Python 3.12.3 / Torch 2.11.0+cu128 process used `-I -B -S`, the existing
CuTe-DSL 4.8.0 overlay / bundled compiler metadata 13.4, static `sm_100`, thread
caps and disabled plugin autoload/JIT. Pre-import raw CUDA driver/runtime,
Torch CUDA and CuTe compile/load guards remained active. Pytest and strict
aggregate exits were **0**, with **zero failures, errors, skips or intercepted
CUDA/JIT calls** and uninitialized Torch CUDA. Pytest reported 570 warnings.
Inherited CUDA visibility was unchanged; GC stayed enabled with default
thresholds `[700,10,10]`. Installed packages were not modified.

Independent stdlib-only postflight matched the exact ordered collected/JUnit
identities, all **1146 passing setup/call/teardown reports**, all **156 source
fingerprints**, Git state, inherited visibility and default GC before these
validation-note edits. The independent report has
`cpu_static_qualification=true`, `gpu_qualification=false` and no evidence errors.
Focused Ruff, isolated in-memory syntax and whitespace checks also passed.

Evidence is preserved in
`/home/scratch.linquanh_gpu/tmp/dsv41-drop-mxfp8-cpu.8VrIRuac/`:
`validate_cpu.py`, `manifest.json`, `collect-01.json`, `collect-01.log`,
`run-01.json`, `run-01.xml`, `run-01.log`, `independent_postflight.py` and
`independent-postflight.json`. `tested-source.patch` preserves the exact tested
patch against `d27624a74`; `documentation-baseline.json` records the state before
subsequent validation-note edits. `documentation_postflight.py` and
`documentation-postflight.json` corroborate that those edits change only three
Markdown files, not executable code, tests, references or numerical gates.

This qualifies host configuration/plumbing/reference contracts only, **not
CuTe lowering, GPU execution or performance**. At this CPU checkpoint no GPU
allocation, example or benchmark had run for the removal; the later pre-fusion
GPU baseline is recorded separately above. All historical records retain their
original snapshot boundaries and do not qualify the joint/API revision.

### Request-scaled FP8-main GPU validation (2026-10-18)

The request-selected amax, FP8 normalization and K/V scale restoration passed on
one allocated **NVIDIA B200 / 148 SMs** in `umb-b200-237` (job `5103699`), using
Python 3.12.3 / Torch 2.11.0+cu128 / CuTe-DSL 4.8.0 / bundled compiler 13.4.
This is a fresh request-scaled run, not reuse of the historical unit-scale results.

| Phase | Passed | Pytest duration | Warnings |
|---|---:|---:|---:|
| Isolated legacy capture | 1/1 | 4.31 s | 35 |
| Smoke | 32/32 | 982.97 s | 187 |
| Full `--full` matrix | 271/271 | 3324.13 s | 1641 |
| Independent selected dense subset | 57/57 | 1213.53 s | 980 |

There were **zero failures, errors or skips**. Full coverage was **157 decode /
48 cache / 66 native sparse MLA** cases. Actual allocated collection matched the
exact ordered candidate manifests: full271, smoke32 and dense57, preserving all
207 original logical cases as 211 mode-expanded legacy identities. The smoke
and isolated capture cases overlap the full matrix; these phase counts are not
counts of distinct tests to add together. Full and dense ran without fail-fast;
only smoke used `-x`.

Both real-stage FP4-to-FP8 conversion branches passed all finite E2M1 codes and
finite E4M3 scales, including distinct per-request descales, signed zero,
page-plane addressing, retained stages, SW128 stores and guards. Both analytic
changed-data graph cases passed payload-dependent amax, late maxima, FP32 division
ties, below-one scales, independent K/V restoration and zero/empty transitions.
Changed-data replay covered every global width `0/64/.../512`. Writer/decode
integration passed, including native widths 0/128/192/512 and changed scale bytes.
Existing source-state precision, sink-once, empty/poison, large-offset, legacy
Frost and optional BF16-Q/MXFP8 regressions also passed under unchanged references
and numerical gates.

Every Python/timeout, tee, effective phase and job-wrapper exit was 0; Slurm
root/batch/extern completed with `0:0`. The independent report has
`default_gpu_qualification=true`, `fresh_arithmetic_results_only=true` and no
evidence errors. Exact ordered collected and JUnit identities and complete passing
setup/call/teardown triplets matched each phase manifest. All **6568 frozen
fingerprints**, Git state and scheduler-assigned visibility matched through final
postflight and independent terminal corroboration, before validation-note edits.
Default Python GC remained enabled with thresholds `[700,10,10]` and
`force_cudagraph_gc=False`; no GC/capture override manufactured these passes.
The allocation used fresh per-run JIT/cache roots and left installed packages
unchanged. The root allocation used **1h36m07s** (5767 GPU-seconds), within the
fresh assistant-imposed three-GPU-hour cap; batch/extern time is not added again.

Evidence is preserved in
`/home/scratch.linquanh_gpu/tmp/dsv41-request-scaled-fp8-b200.PDBIjWZb/`:
`independent-audit.json`, `terminal-receipt.json`, `freeze.json`,
`frozen-source.patch`, `collection-summary.json`, phase `*-results.json`,
`*.xml`, `*.log` and shell-exit records. The new ledger is `budget/budget.json`
in that directory; historical evidence and budget ledgers were not reused.
The exact tested patch is preserved against baseline `b41c42724`. Subsequent
validation-note edits touch only three Markdown files and the public API validation
docstring, not function bodies, tests, references or numerical gates; see
`documentation-baseline.json` and `documentation-postflight.json` for the separate
documentation-only drift record.

These results qualify only this frozen request-scaled B200/4.8/13.4 snapshot and
selected dense subset, **not the entire dense suite, other hardware/toolchains or
checkpoint/full-model behavior**. Examples and benchmarks have not run; no
performance or measured crossover is claimed. Pytest durations above are
validation runtime, not GPU kernel-latency measurements.

### Request-scaled FP8-main CPU/static acceptance (2026-10-18)

The request-selected scale implementation passed **413/413 CPU cases**:

| Suite | Passed |
|---|---:|
| Source-state/resource/adapter/independent-oracle contracts | 219 |
| Optional BF16-Q/MXFP8 isolation and legacy-factory contracts | 41 |
| Conversion-module lifetime contracts | 7 |
| Benchmark/example defaults and extracted reference contracts | 100 |
| Existing native host contracts | 40 |
| Unknown-backend rejections | 6 |

The fresh process used Python 3.12.3 / Torch 2.11.0+cu128 with the existing DSL
4.8.0 overlay and bundled compiler metadata 13.4, `-I -B -S`, explicit package
paths, thread caps of one, static `sm_100`, and disabled plugin autoload/JIT.
Pytest and strict aggregate exited 0, with no failures, errors, skips or
intercepted CUDA/JIT calls. Torch CUDA remained uninitialized; CUDA visibility
and default GC state were unchanged. This matrix imports Torch/CuTe under guards,
not in an accelerator-import-free process.

Independent stdlib-only postflight matched the exact collected 413-case manifest,
ordered JUnit identities, all passing setup/call/teardown reports, all **158**
source/runner/selector/manifest fingerprints, Git state and visibility, before
these validation-note edits. It also checked the complete interpreter/pytest
commands and the literal ini `--import-mode=importlib` prepend. The exact tested
patch is preserved against baseline `b41c42724`; later validation-note edits do
not change function bodies, references or numerical gates. Records are in
`/home/scratch.linquanh_gpu/tmp/dsv41-request-scaled-fp8-cpu.lOlVJCk7/`:
`collect48.{json,log}`, `acceptance48.{json,xml,log}`, `manifest.json`,
`postflight.json`, `static-checks.json` and `frozen-source.patch`.

All 13 changed Python files passed focused Ruff formatting/lint without an F811
exception, in-memory syntax compilation and whitespace checks. A separate
**28-check** static pass preserved public API bodies, BF16 resources, legacy
writers/adapters, source-store/merge arithmetic and numerical gates, and checked
FP32-division/FP8-rounding witnesses without accelerator imports. The new host
contracts cover shared per-request scale bindings, topology/ownership guards,
legacy scalar paths, selected-only amax, signed zero, zero/empty transitions and
independent represented-value references. These results qualify CPU/static
configuration/plumbing/reference contracts only, **not CuTe lowering, GPU numerical
accuracy/graph replay, examples or performance**. The separate allocated GPU
qualification is recorded above. Examples and benchmarks remain unrun.

### Historical unit-scale FP8-main CPU/static acceptance (2026-10-16)

The earlier unit-scale FP8-main snapshot passed **321 CPU cases**: 146 guarded
source-state/resource/adapter/oracle contracts, 37 optional-MXFP8 contracts,
7 conversion-lifetime checks, 85 benchmark/example-default contracts, 40 existing
native host contracts and 6 unknown-backend rejections. This includes independent
analytic main conversion before both logits and values, FP8 SW128/vector-store
and real-stage probe-view contracts, the reduced original-Q ABI, and independent
BF16 compatibility resource configurations.

The fresh process used Python 3.12.3 / Torch 2.11.0+cu128 with the existing DSL
4.8.0 overlay and bundled compiler metadata 13.4, `-I -B -S` with explicit package
paths, thread caps of one, static `sm_100`, and disabled plugin autoload/JIT.
Pytest and strict aggregate exited 0, with no failures/errors/skips or intercepted
CUDA/JIT calls. Torch CUDA remained uninitialized and visibility unchanged.
Exact ordered JUnit identities matched the frozen 321-case manifest, and all
154 source/runner/manifest fingerprints and Git state matched through independent
postflight, before these validation-note edits. The combined matrix imports
Torch/CuTe under guards; it is not an accelerator-import-free run.
The latest rerun includes the probe's flat rank-one tuple-layout regression after
the GPU failure below. Records are in
`/home/scratch.linquanh_gpu/tmp/dsv41-fp4-fp8-tuple-cpu.zg51v9kk/`:
`tuple-acceptance.{json,xml}`, `manifest.json` and `independent-postflight.json`.
The earlier 321-case records, including failed v1/v2 fixture-setup attempts, remain
preserved in `/home/scratch.linquanh_gpu/tmp/dsv41-fp4-fp8.MtsdGo/`.

Focused Ruff formatting/lint (retaining the existing MXFP8 fixture-import `F811`
exception), in-memory Python syntax compilation and whitespace checks passed.
These qualify host configuration/plumbing/reference and static contracts only,
not CuTe lowering, GPU numerical accuracy/graph replay, examples or performance.

### Historical unit-scale FP8-main GPU validation (2026-10-16)

These results predate request-selected scale computation and restoration and do
not qualify either. The corrected unit-scale snapshot passed on one allocated **NVIDIA B200 / 148 SMs**
in `umb-b200-263` (job `5049128`), using Python 3.12.3 / Torch 2.11.0+cu128 /
CuTe-DSL 4.8.0 / bundled compiler 13.4:

| Phase | Passed | Pytest duration | Warnings |
|---|---:|---:|---:|
| Isolated legacy capture | 1/1 | 4.10 s | 35 |
| Smoke | 27/27 | 817.02 s | 163 |
| Full `--full` matrix | 271/271 | 3502.50 s | 1651 |
| Independent selected dense subset | 57/57 | 1243.21 s | 980 |

There were **zero failures, errors or skips**. Full coverage was **157 decode /
48 cache / 66 native sparse MLA** cases. Both actual-stage FP4-to-FP8 byte probes
compiled, launched and passed for native and fallback conversion, including all
finite codes/scales, signed zero, page-plane addressing, SW128 stores and guards.
Both analytic rounding/saturation/underflow changed-data graph cases passed,
as did all global widths `0/64/.../512`, native source-state/sink/empty gates,
writer/decode replay, legacy Frost and optional MXFP8 cases. The four historical
plan-isolation IDs expand explicitly to both cache modes, preserving all 207
original logical cases as 211 mode-expanded identities.

Every Python/timeout, tee, effective phase and job-wrapper exit was 0; Slurm
root/batch/extern completed with `0:0`. Exact ordered JUnit identities and complete
passing setup/call/teardown triplets matched the phase manifests. Independent
corroboration matched all **5871 frozen fingerprints**, Git state and scheduler
visibility through terminal completion, before these validation-note edits.
Numerical references, byte expectations, tolerances and actual default
GC/capture settings were unchanged; `force_cudagraph_gc=False` was retained.
The independent report has `default_gpu_qualification=true` with no evidence
errors. The root allocation used **1h36m04s** (5764 GPU-seconds), within the
three-GPU-hour budget; batch/extern accounting is not added again.

Evidence is preserved in
`/home/scratch.linquanh_gpu/tmp/dsv41-fp4-fp8-b200-tuple.l7c2aj9y/`:
`independent-audit.json`, `terminal-corroboration.json`, `freeze.json`, phase
`*-results.json`, `*.xml`, `*.log` and shell-exit records. The budget ledger is
`/home/scratch.linquanh_gpu/tmp/dsv41-fp4-fp8-gpu-budget.3mr2xwkh/budget.json`.
These results qualify only that frozen B200/4.8/13.4 unit-scale FP8-main snapshot
and the selected dense subset, not request-scaled conversion/restoration, the
entire dense suite, other GPU/toolchain combinations or checkpoint/full-model
behavior. Examples and benchmarks have not
run; pytest durations above are validation runtime, not kernel-latency or
performance measurements.

#### Earlier FP8-main attempts

The allocated B200 attempt on `umb-b200-263` (job `5048418`, DSL 4.8.0 /
bundled compiler 13.4) passed preflight, the exact **271-case** `--full` collection
(157 decode / 48 cache / 66 native sparse MLA), and the isolated legacy capture
regression **1/1**. Its 27-case smoke stopped at the first new
`test_fp4_fp8_all_finite_codes[False]` during JIT: `cutlass.make_array_view`
received a scalar CuTe layout shape and failed with
`TypeError: object of type 'int' has no len()`. No conversion kernel launched or
byte/numerical check ran. Smoke reported **1 failed**, pytest exited 1 and Slurm
recorded `FAILED` / `1:0`; full and independent dense phases were skipped by the
smoke prerequisite gate. All 5772 frozen fingerprints and Git state matched
through independent terminal corroboration, with the recorded default GC/capture
settings and scheduler visibility unchanged. Evidence is in
`/home/scratch.linquanh_gpu/tmp/dsv41-fp4-fp8-b200-collected.75ea7jn5/`, including
`smoke.log`, `smoke.xml` and `terminal-corroboration.json`.

The probe now supplies a flat one-element shape tuple, with a host regression
assertion. This changes only test setup, not conversion arithmetic, byte
expectations or tolerances; all 321 guarded CPU cases passed again as recorded
above. The corrected probe and FP8-main attention subsequently passed the
separate job `5049128` recorded above; the earlier failed evidence is unchanged.
Two earlier allocations (`5047536`, `5047995`) executed no test cases: their
wrappers rejected pytest's default `maxfail=None` and an obsolete 266-case
forecast, respectively. Their records remain in
`/home/scratch.linquanh_gpu/tmp/dsv41-fp4-fp8-b200.CeS4DyWZ/` and
`/home/scratch.linquanh_gpu/tmp/dsv41-fp4-fp8-b200-retry.sqz58o3d/`.
That historical manifest came from real allocated collection; four original
plan-isolation IDs explicitly expanded to both `fp8` and `mxfp8`, preserving all
207 original logical cases as 211 mode-expanded identities.

### Historical request-scaled GPU probes (pre-fusion)

The commands below reproduce the **historical request-scaled snapshot**, not the
current common-unit staging assertions. Current tests migrate those probes to
shared units and joint staging; do not infer new validation from these records:

```bash
python -m pytest --full -v \
  tests/experimental/test_deepseek_v41_decode.py::test_fp4_fp8_all_finite_codes \
  tests/experimental/test_deepseek_v41_decode.py::test_fp8_main_scale_rounding_underflow_changed_graph \
  tests/experimental/test_deepseek_v41_decode.py::test_fp8_swa_changed_graph
```

The byte probe covers all finite E2M1 codes × finite E4M3 scales, including signed
zero, with neighboring group16 scales, permuted page selections, all eight
retained D128 stages, invalid IDs and shared-memory guards. It uses actual
production preparation, two requests with distinct descales, the inherited
page-plane loader and both conversion branches. Analytic replay checks K and V
restoration, FP32 division ties, below-one scales, payload-dependent maxima,
changed payload/scale/index data, poisoned scratch and zero/empty transitions.
These request-scaled probes/replays passed on B200 / DSL 4.8.0 / bundled compiler
13.4 in job `5103699`, as recorded above. Historical job `5049128` passed the
earlier unit-scale versions, not these assertions. The example and
reference-checked benchmark remain unrun.

### Historical native-first split acceptance (2026-10-15, pre-FP8-main)

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

### Historical optional MXFP8 acceptance (2026-10-15, pre-FP8-main)

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

### Earlier ordinary-FP8 GPU attempts (pre-FP8-main)

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

### Capture diagnosis and test-resource lifetime repair (pre-FP8-main)

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

### Historical post-repair GPU validation (2026-10-15, pre-FP8-main)

This snapshot used native FP8 SWA with BF16 main attention and exact BF16 Q
expansion; it predates FP8-main staging/attention. One allocated B200 on
`umb-b200-260` (Slurm job `5008381`) used CuTe-DSL package/module 4.8.0 and its
bundled CUDA 13.4 compiler, with fresh scratch/JIT,
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

This qualifies only the tested pre-FP8-main, post-repair B200/4.8/13.4 full257
snapshot and selected dense57 subset, not current FP8-main staging/attention,
all dense configurations, DSL 4.7 GPU execution, other hardware/toolchains or
checkpoint/full-model behavior. It is a successful default
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

Native Prims-TS uses fixed **Keep-AB / M64 / BK128 / retained-KV8 / split 1 /
nonpersistent** joint E4M3 Q/K/V/P attention with the original FP8 Q. It does not
use Frost's heuristics or the native wrapper's automatic policy. There are no
measured crossovers or tuned batch thresholds. Its callable checks raw metadata
inside one finalizing attention, with no separate preparation, source merge or KV amax.
`MixedFp8KvProducer` / `MixedFp8SmemKvResource` in `prims_ts_resources.py`
copy SWA bytes and directly round scaled FP4 to E4M3 in common cache units.
Thread-staged SWA/main share all-writer AsyncThread-to-UMMA publication; Q TMA is
unchanged. BF16-named bases provide exact register conversion and selected loads,
not BF16-Q support, BF16 query scratch or a full converted cache.

Both backends use queries `[B,1,64,512]`, 64-token pages, 128 window positions
and up to 512 FP4 compressed positions, padded to multiples of 64. The Prims-TS
example uses `swa_fp8` and caller-converted FP8 Q; Frost uses BF16 Q/`swa_mxfp8`.
There is no Prims-TS mode option. Window-only uses the same backend-specific
representation, with no compressed stream.

This model-specific quantized-cache adapter is separate from
`BatchSparseMLADecodePagedTSWrapper`: that native wrapper still accepts matching
native BF16 or E4M3 Q/KV and **does not accept packed quantized cache inputs**.
The adapter does not broaden its input contract.
Physical slot `-1` and out-of-capacity slots are masked without reading cache
storage. Every valid cache slot must be initialized and causally visible.

Frost uses BF16 Q/K/V/P and FP32 accumulation/softmax, with independent FP64
relative-L2 < 0.005, max-scaled error < 0.01 and absolute LSE error < 1e-5.
Those legacy gates and represented BF16 values are unchanged. Native E4M3
SWA/main attention, including main-only requests, uses the per-component rounding
bound above and the same strict LSE/exact-empty gates, not BF16 main-only gates.
Checkpoint accuracy has not been established.

`deepseek_v41_decode` returns **only BF16 output** with Q's shape; supplied out
is returned by identity. Internal FP32 `[B,64,1]` LSE is natural-log and
**sink-exclusive**. A finite per-head sink or `-inf` adds denominator mass once.
Empty rows produce exact zero O and `-inf` internal LSE, even for large sinks.

Allocate caller workspace with the pure sizing helper and warm the same
backend/declaration eagerly before capture. Capture requires preallocated out.
Workspace/output/input ownership and lifetime follow the interface section
above: stable live captured pointers, in-place data updates and independent
workspace/out for concurrent calls. Python BMM-factor values are captured;
changing them requires recapture, not recompilation. Training, speculative
multi-token queries, SM103 and full-model E2E validation remain outside scope.

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
The opaque uint8 HND `[pages,1,64,width]` or NHD `[pages,64,1,width]` view does
not interleave scales per token. Ordinary typed FP8 is different: just64
contiguous payload rows, no side plane. Normalize writer input rows upstream
for nonunit common KV descales; writing never selects that tensor descale.

| Cache | Data bytes/token | Scale bytes/token | Scale format |
|---|---:|---:|---|
| Main FP4, uint8 width 288 | 256 E2M1 | 32 | E4M3, group16, in caller's common units |
| Window MXFP8, uint8 width 528 | 512 E4M3 | 16 | E8M0, group32, in caller's common units |
| Window ordinary FP8, typed E4M3 width 512 | 512 E4M3 | 0 | Caller-normalized common units |

The caller supplies physical slot IDs. For a logical token ID `t` in request
`b`, the usual mapping is `block_table[b, t // 64] * 64 + t % 64`, with invalid
entries mapped to `-1`. Slot mapping, RoPE, indexing and GEMM quantization remain
outside this decode API.

The reproducible GPU benchmark uses **explicit E4M3 Q / ordinary FP8 SWA**
when Prims-TS is selected. Frost and optional FlashMLA retain their legacy
**BF16 Q / MXFP8 SWA** representation and original gates. All arms derive from
common seeded inputs and share the global FP4 pool, selections and sink, but
each is checked against its own represented-value FP64 oracle. The native arm uses the componentwise
rounding bound and exact empty/strict sink-exclusive LSE gates above. Its reference
gathers/masks only selected global bytes, exactly decodes scaled E2M1, then
independently clamps/rounds to E4M3 in common cache units before FP64 QK and V;
there is no request amax, division or whole-cache conversion. The legacy
references retain decoded BF16 main values. JSON labels each arm's dtype,
SWA encoding, packed main storage, common-unit conversion, BMM factors,
shapes, actual arithmetic and accuracy contract. Native main precision is
`shared_pertensor_e4m3`, with no request-scale scratch and the
`dsv41_shared_pertensor_joint_e4m3_swa512_fp4_inline_metadata_v2` compile marker.
Source provenance includes public API, workspace, inline metadata views, native
producer and FP8 packing helpers, without importing native adapters merely to hash
them.
Native FP8 versus Frost/FlashMLA MXFP8 timings are **not** a like-for-like backend
comparison:

```bash
# The existing default remains cute-dsl only:
python benchmarks/bench_deepseek_v41_decode.py \
  --batches 1,4,16,32,64,128,256 --context 32768 --output frost.json
# Compare the two in-tree backends without external FlashMLA:
python benchmarks/bench_deepseek_v41_decode.py \
  --backends cute-dsl prims-ts --batches 1,4,16,32,64,128,256 \
  --context 32768 --output backends.json
# Optional comparison requires an externally built FlashMLA installation:
python benchmarks/bench_deepseek_v41_decode.py \
  --backends cute-dsl prims-ts --batches 1,4,16,32,64,128,256 \
  --context 32768 --flashmla --output paired.json
```

`--flashmla` requires a FlashMLA build with DS4.1 MXFP8/FP4 mixed-cache support
and the `FlashMLASchedMeta` API. `get_mla_metadata()` returns an empty scheduler
object and a `None` placeholder; the first eager call initializes it from the
fixture before graph capture. The benchmark creates fresh metadata per shape.

Each backend gets independent caller workspace/out and combined live metadata.
Timing uses warm CUDA Graph/events,32 complete public calls per replay and
symmetric backend order. Per-call sanitation, selected-tile staging and attention
(including native joint finalization or Frost split reduction) remain inside
timing. No KV amax or request descale exists. JSON records correctness metrics,
workspace bytes, split counts, latency samples, toolchain versions and source
SHA256 hashes. Quantization, metadata allocation, JIT, allocation, graph capture
and CPU overhead are excluded. The revised benchmark has not run; future
component measurements are not model token latency or throughput.

The prepared semantic and changed-input CUDA Graph tests require
CuTe-DSL >= 4.8 with its bundled CUDA 13 compiler on SM100 when including the
Prims-TS adapter. Current cache-writer integration covers ordinary-FP8 Prims-TS
and legacy Frost, including window-only decode and selected global widths. It
captures cache writing and decode together, changes values and selections at fixed
addresses, and checks poisoned-buffer replay and the transition to empty
selections. Exact encoded-byte references run before the independent FP64 decode
gates. The original legacy integration covered both backends at global widths
128/192/512; all eight expanded legacy cases passed in the historical full
matrix on the allocated B200 in Slurm job `4910237`, with CuTe-DSL 4.8.0 /
bundled CUDA 13.4. The historical `swa_fp8` writer/decode integrations used
explicit FP8 Q and BF16 main attention at widths 0/128/192/512; all passed within
the pre-FP8-main clean post-repair full257 run on `umb-b200-260` (job `5008381`,
B200 / DSL 4.8.0 / bundled CUDA 13.4), including changed-data CUDA Graph replay
and exact encoded-byte references. They do not qualify current E4M3 main staging
or attention. The later historical unit-scale FP8-main full271 run on
`umb-b200-263` (job `5049128`, B200 / DSL 4.8.0 / bundled compiler 13.4) passed all
**48 cache cases**, including native writer/decode graph replay, as recorded above.
Neither historical result qualifies request-selected amax, FP8 normalization or
scale restoration. The separate request-scaled run on `umb-b200-237` (job
`5103699`, B200 / DSL 4.8.0 / bundled compiler 13.4) passed all **48 cache cases**,
including the new writer/replay assertions at native global widths 0/128/192/512.
The captured writer/decode path verified changed scale bytes, refreshed request
descales and zero/empty transitions under the unchanged byte and FP64 decode gates.

```bash
pytest tests/experimental/test_deepseek_v41_decode.py \
  tests/experimental/test_deepseek_v41_cache.py --full -v
```

The APIs and both backends remain experimental and may change or disappear
without compatibility guarantees. They are not registered for AOT or trace.
