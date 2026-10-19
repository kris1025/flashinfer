# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental DS4.1 decode backends and optional cache preparation on SM100."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def deepseek_v41_decode(
    query,
    swa_kv_cache,
    workspace_buffer,
    sparse_indices=None,
    compressed_kv_cache=None,
    sparse_topk_lens=None,
    seq_lens=None,
    out=None,
    bmm1_scale=1.0,
    bmm2_scale=1.0,
    sinks=None,
    kv_layout="HND",
    *,
    backend="cute-dsl",
):
    """Single-token SWA/main decode with the V4 caller-workspace convention.

    The supported V4-style prefix accepts Q ``[B,1,64,512]``, a caller-owned
    contiguous int8/uint8 workspace, optional BF16 ``out`` of Q's shape, and
    shared Python-scalar ``bmm1_scale`` / ``bmm2_scale``. Only the BF16 output
    is returned; supplied ``out`` is returned by identity. Use
    :func:`deepseek_v41_decode_workspace_size` to size the workspace. Its base
    must be 32-byte aligned. No public executable plan or LSE is returned.

    ``sparse_indices`` is required CUDA int32 ``[B,128+K]``: its first128
    columns address SWA physical slots and the remaining K columns address
    main slots, where K=0/64/.../512. K>0 requires ``compressed_kv_cache``.
    Also required are CUDA int32 ``sparse_topk_lens[B]`` and ``seq_lens[B]``.
    Topk lengths include the fixed128 SWA positions, so the active main
    prefix is ``topk-128``. The SWA prefix is ``[0,clamp(seq_lens,0,128))``
    as explicit V4.1 policy, not a claim about V4 H64/Sq1 CAKE or cubin
    behavior. Negative, reserved-sentinel and out-of-capacity slots are masked; holes and
    duplicate multiplicity are preserved. Valid topk values lie in
    ``[128,128+K]``. Lengths and selections refresh on every call/replay.
    Callers own causal visibility and initialization of every valid slot.

    ``kv_layout="HND"`` (default) expects contiguous
    ``[pages,1,64,width]`` caches; explicit ``"NHD"`` expects
    ``[pages,64,1,width]``. Layout normalization is zero-copy. Packed pages
    retain separate payload and scale planes, not interleaved token records.

    ``backend="cute-dsl"`` retains Frost (the default): BF16 Q, uint8
    MXFP8 SWA width528, and optional uint8 FP4 main width288. Its attention
    schedule is unchanged; standalone sanitation adds one preparation launch.
    ``backend="prims-ts"`` requires explicit torch.float8_e4m3fn Q and
    typed E4M3 SWA width512, with the same optional packed FP4 main. BF16 Q
    is rejected before device validation or compilation, without conversion
    or fallback. Prims-TS requires CuTe-DSL>=4.8 and its bundled CUDA13
    compiler. Fixed Keep-AB/M64/BK128/retained-KV8/split1/nonpersistent
    execution jointly attends SWA and main in one finalizing kernel, consuming
    raw metadata and optional sinks inline without a preparation launch. Its
    workspace contains internal LSE and a K0 dummy, not sanitized metadata.
    SWA uses thread-written FP8 stages; Q retains native TMA.

    Both caches must already use common cache units. ``bmm1_scale`` is the
    positive finite FP32 QK/softmax factor; ``bmm2_scale`` is a finite FP32
    output factor. For Q descale q_s, shared KV descale s>0 and attention
    factor alpha, supply ``alpha*q_s*s`` and ``s``. Native main staging
    directly rounds/saturates ``E2M1*group16_E4M3_scale`` to E4M3 in those
    units, retaining signed zeros/subnormals. It does not calculate amax or
    a request descale, divide by bmm2, or restore scales twice. Saturation
    at +/-448 cache units is intentional. Neither backend repairs caches
    encoded with inconsistent units. The original packed group scales and
    cache-writer ABI remain unchanged.

    The defaults are V4's ``bmm1_scale=1.0,bmm2_scale=1.0``. Explicitly pass
    ``bmm1_scale=512**-0.5`` to retain the former V4.1 attention factor.
    ``sinks`` is optional FP32[64], finite or -inf; omitted means disabled.
    Sink mass is applied once to output normalization. Score maxima stay in
    raw-dot units; differences are multiplied by BMM1 before log2 conversion,
    avoiding a prematurely overflowing BMM1*log2(e) coefficient. Frost split
    scratch retains raw maxima and sums rather than scaled partial LSE.
    Empty rows produce exact zero output. Effective Q/K/V must be finite;
    this is inference-only.

    Capture requires supplied out, sufficient workspace, and eager warmup
    for the same backend/declaration. Captured addresses must remain alive
    and fixed; update tensor values in place. Python scalar factors are
    captured values: changing them requires recapture, not recompilation.
    Eager calls may change addresses. Concurrent calls need separate output
    and workspace, or caller synchronization. Workspace and out must not overlap
    inputs or each other; read-only inputs may alias one another.

    This experimental API supports only SM100/H64/Sq1/D512; it does not add
    V4 varlen, PDL, RopeQuant, SM120, MTP or automatic-backend support.
    Historical tests do not qualify this shared-scale/API/inline-metadata revision.
    No performance or checkpoint/full-model equivalence is claimed. See
    ``examples/deepseek_v41/README.md`` for current validation boundaries.
    """
    if backend == "cute-dsl":
        from .experimental.deepseek_v41.decode import decode_mixed
    elif backend == "prims-ts":
        from .experimental.deepseek_v41.prims_ts import decode_mixed
    else:
        raise ValueError(f"backend must be 'cute-dsl' or 'prims-ts', got {backend!r}")
    return decode_mixed(
        query,
        swa_kv_cache,
        workspace_buffer,
        sparse_indices,
        compressed_kv_cache,
        sparse_topk_lens,
        seq_lens,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
        kv_layout,
    )


@flashinfer_experimental_api
def deepseek_v41_window_decode(
    query,
    swa_kv_cache,
    workspace_buffer,
    sparse_indices=None,
    sparse_topk_lens=None,
    seq_lens=None,
    out=None,
    bmm1_scale=1.0,
    bmm2_scale=1.0,
    sinks=None,
    kv_layout="HND",
    *,
    backend="cute-dsl",
):
    """K=0 helper with the same ownership/scaling contract as mixed decode.

    ``sparse_indices`` is required int32[B,128], ``sparse_topk_lens`` is
    required int32[B] with values128, and ``seq_lens`` is required int32[B].
    Both backends use caller workspace and optional BF16 out, and return
    only output. See :func:`deepseek_v41_decode` for cache units/layouts,
    the V4 scale defaults, capture lifetime and supported declarations.
    """
    return deepseek_v41_decode(
        query,
        swa_kv_cache,
        workspace_buffer,
        sparse_indices,
        None,
        sparse_topk_lens,
        seq_lens,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
        kv_layout,
        backend=backend,
    )


@flashinfer_experimental_api
def deepseek_v41_decode_workspace_size(
    batch_size, compressed_topk=0, *, backend="cute-dsl"
):
    """Return workspace bytes for a 32-byte-aligned allocation, without CUDA.

    ``batch_size`` is in [1,65535]; ``compressed_topk`` is0 or a multiple of64
    through512. This pure sizing helper neither compiles nor creates a plan.
    The returned size includes backend-specific scratch and internal LSE.
    """
    from .experimental.deepseek_v41._workspace import workspace_size

    return workspace_size(batch_size, compressed_topk, backend=backend)


@flashinfer_experimental_api
def deepseek_v41_quantize_cache(
    x, *, format, page_size=64, out=None, slots=None, kv_layout="HND"
):
    """Quantize/write DS4.1 cache bytes in caller-chosen cache units.

    Optional Triton preparation helper; decode accepts compatible caches from
    any producer. Contiguous finite BF16/FP32 x[N,512] must already include
    model-required RoPE and any normalization to shared per-tensor cache
    units. This writer does not calculate a shared tensor descale. SM100 and
    page_size=64 only. Decode factors restore the caller's common units.

    ``main_kv_fp4`` uses E2M1 data and group16/E4M3 scales, width288;
    ``swa_mxfp8`` uses E4M3 data and group32/E8M0 scales, width528. Opaque
    uint8 pages retain all payload rows followed by all scale rows.
    ``swa_fp8`` clamps to [-448,448] and rounds to nearest-even E4M3,
    retaining signed zero/subnormals, and returns typed E4M3 width512
    without a scale plane. It requires explicit E4M3 Q and Prims-TS decode.
    Main-group scales must be representable in E4M3.

    Default ``kv_layout="HND"`` output is contiguous [pages,1,64,width];
    explicit ``"NHD"`` gives [pages,64,1,width], with identical storage
    bytes. Supplied ``out`` is returned by identity. Without slots, input
    row i writes slot i. Incremental updates require caller-owned out and
    CUDA int32 slots[N]. Invalid slots skip publication; valid slots must
    be unique (unchecked caller precondition). Unwritten slots remain
    untouched and must not be attended to before initialization. Inputs
    and out must not overlap. Reusing out permits allocation-free replay.
    """
    from .experimental.deepseek_v41.cache import quantize_cache

    return quantize_cache(
        x,
        format=format,
        page_size=page_size,
        out=out,
        slots=slots,
        kv_layout=kv_layout,
    )
