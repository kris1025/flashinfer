# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental DS4.1 decode backends and optional cache preparation on SM100."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def deepseek_v41_decode(
    q,
    swa_cache,
    global_cache,
    swa_indices,
    global_indices,
    sink,
    *,
    plan=None,
    backend="cute-dsl",
):
    """Mixed SWA/FP4 decode using CuTe DSL primitives on SM100.

    ``backend="prims-ts"`` uses explicit torch.float8_e4m3fn Q[B,1,64,512]
    with the same dtype of contiguous, unit-scale SWA[pages,64,1,512], without
    a scale plane. This is ordinary E4M3, not MXFP8 with ignored scales.
    Decode never implicitly quantizes BF16 Q. SWA uses native FP8 Q/K/V/P
    and TMA, not BF16 dequantization. When optional uint8 FP4
    global[pages,64,1,288] is present, Q is exactly expanded to BF16 for that
    separate selected-tile pass. Sink-free normalized FP32 source outputs
    and log2 LSE are merged before the sole final BF16 rounding; the sink
    contributes denominator mass once. Window-only omits the global pass
    and BF16 query scratch.

    ``backend="cute-dsl"`` preserves the Frost implementation (default):
    contiguous BF16 Q[B,1,64,512], uint8 MXFP8 SWA[pages,64,1,528] and optional
    uint8 FP4 global[pages,64,1,288]. Both packed caches use page-plane storage:
    all payload rows precede all scale rows. With ``backend="prims-ts"``,
    explicit BF16 Q with this MXFP8 SWA selects a separate, lazily imported
    compatibility adapter. E4M3 Q/SWA remains the primary Prims-TS
    implementation; there is no implicit conversion or public decode-format
    selector.

    Prims-TS requires CuTe-DSL >= 4.8 with its bundled CUDA 13 compiler.
    Fixed Keep-AB/M64/retained-KV8/split-1/nonpersistent schedules use BK128
    for native FP8 SWA and BK64 for the optional global FP4 pass. MXFP8
    compatibility instead uses BK64 BF16 arithmetic and fused finalization.
    Live selections are refreshed on the GPU on every call/replay; native
    FP8 also refreshes true valid counts for its independent source passes.
    Plans cannot cross modes or backends. The native sparse wrapper's
    packed-input restriction is unchanged.

    SWA indices are int32[B,1,128]; global indices int32[B,1,K], K a positive
    multiple of64 up to512. Negative, reserved-sentinel and out-of-capacity
    slots are masked; duplicate positions retain their multiplicity. The
    caller owns causal visibility and initialization of every valid slot.
    Sink is FP32[64], finite or -inf (disabled). Effective Q/K/V must be
    finite. Accumulation and softmax use FP32; no backward parity is claimed.

    Returns (out,lse,plan): BF16 output shaped like Q, even for FP8 Q, and
    FP32 natural-log LSE [B,64,1] excluding the sink. Empty rows give exact
    zero output and -inf LSE. Prepare before capture and reuse a plan with
    matching shape, strides, dtype and device. Replay overwrites owned
    outputs/scratch and retains captured input addresses; update values in
    place or recapture when addresses change. Eager calls may change input
    addresses. One plan must not execute concurrently on different streams.

    The original pre-reorganization snapshot passed the full 257-case matrix
    (including native FP8) and selected 57-case dense subset on
    B200/DSL 4.8.0/bundled CUDA 13.4. Those historical results do not qualify
    this reorganized source tree. Performance, crossovers and
    checkpoint/full-model behavior are unmeasured.
    See examples/deepseek_v41/README.md for scope and commands. Single-token
    decode only; MTP and SM103 are outside this adapter's scope.
    """
    if backend == "cute-dsl":
        from .experimental.deepseek_v41.decode import decode
    elif backend == "prims-ts":
        from .experimental.deepseek_v41.prims_ts import decode
    else:
        raise ValueError(f"backend must be 'cute-dsl' or 'prims-ts', got {backend!r}")

    return decode(
        q, swa_cache, global_cache, swa_indices, global_indices, sink, plan=plan
    )


@flashinfer_experimental_api
def deepseek_v41_window_decode(
    q, cache, indices, sink, *, plan=None, backend="cute-dsl"
):
    """Window-only variant of deepseek_v41_decode, with explicit backend selection.

    ``backend="prims-ts"`` uses explicit E4M3 Q with typed, unit-scale E4M3
    cache of width 512, without BF16 SWA dequantization or BF16 query scratch.
    The Frost default retains BF16 Q with uint8 MXFP8 cache of width 528;
    that explicit pairing also selects the optional Prims-TS MXFP8 adapter.
    Q is [B,1,64,512], cache is [pages,64,1,width] and
    physical slot IDs are int32[B,1,128]. See deepseek_v41_decode for the
    mode/backend-specific plan, arithmetic, output and sink-exclusive LSE
    contracts. The default Frost backend remains unchanged.
    """
    if backend == "cute-dsl":
        from .experimental.deepseek_v41.decode import decode
    elif backend == "prims-ts":
        from .experimental.deepseek_v41.prims_ts import decode
    else:
        raise ValueError(f"backend must be 'cute-dsl' or 'prims-ts', got {backend!r}")

    return decode(q, cache, None, indices, None, sink, plan=plan)


@flashinfer_experimental_api
def deepseek_v41_quantize_cache(x, *, format, page_size=64, out=None, slots=None):
    """Quantize and write the cache formats consumed by DS4.1 decode backends.

    Optional Triton preparation helper; decode accepts compatible caches from
    any producer. Contiguous finite BF16/FP32 x[N,512] must already include
    any model-required RoPE. SM100 and page_size=64 only.

    ``main_kv_fp4`` uses E2M1 data, group16/E4M3 scales, no global scale;
    ``swa_mxfp8`` uses E4M3 data and group32/E8M0 scales. Main-cache scales
    must be representable in E4M3. Opaque uint8 output has shape
    [pages,64,1,288] or [pages,64,1,528]. Each page holds all data rows,
    followed by all scale rows; these are not interleaved token records.
    ``swa_fp8`` instead clamps finite input to [-448,448] and rounds to
    nearest even E4M3 with unit scale, retaining signed zero/subnormals.
    It returns contiguous typed torch.float8_e4m3fn[pages,64,1,512], ordinary
    row-major payload with no scale plane. This format requires explicit
    FP8 Q and ``backend="prims-ts"`` at decode; it is not interchangeable
    with ``swa_mxfp8``.

    Without slots, input row i writes physical slot i. For incremental
    updates, provide caller-owned out and CUDA int32 slots[N]. Negative or
    out-of-capacity slots skip publication. Valid slots must be unique;
    uniqueness is a caller precondition and is not checked. Unwritten slots
    remain untouched and must not be attended to before initialization.
    Inputs and output must not overlap. Reusing out permits allocation-free
    CUDA Graph replay. No implicit QAT derivative.
    """
    from .experimental.deepseek_v41.cache import quantize_cache

    return quantize_cache(x, format=format, page_size=page_size, out=out, slots=slots)
