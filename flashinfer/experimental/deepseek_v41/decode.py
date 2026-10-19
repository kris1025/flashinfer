# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Caller-workspace/JIT dispatch for the Frost DS4.1 decode backend."""

import functools

import torch

from . import _workspace
from ._workspace import bind_inputs, device_sm_count, split_kv


@functools.cache
def _compile_prepare(device_index, compressed_k, has_sink):
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("warm DS4.1 metadata preparation before CUDA Graph capture")
    import cutlass
    import cutlass.cute as cute
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import _metadata

    device_sm_count(device_index)
    batch = cute.sym_int()
    f = cute.runtime.make_fake_compact_tensor
    indices = f(
        cutlass.Int32,
        (batch, 128 + compressed_k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    selected_swa = f(cutlass.Int32, (batch, 128), stride_order=(1, 0), assumed_align=16)
    selected_main = f(
        cutlass.Int32,
        (batch, max(64, compressed_k)),
        stride_order=(1, 0),
        assumed_align=16,
    )
    lengths = f(cutlass.Int32, (batch,), assumed_align=16)
    sinks = f(cutlass.Float32, (64,), assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = _metadata.PrepareMetadata(compressed_k, has_sink)
    return build_and_load_cute_dsl_kernel(
        "deepseek_v41_metadata",
        f"v4_combined_c{compressed_k}_sink{int(has_sink)}_v1",
        lambda: cute.compile(
            kernel,
            indices,
            lengths,
            lengths,
            sinks,
            selected_swa,
            selected_main,
            lengths,
            lengths,
            lengths,
            lengths,
            lengths,
            sinks,
            cutlass.Int64(1),
            cutlass.Int64(1),
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        ),
        extra_key_files=(__file__, _metadata.__file__, _workspace.__file__),
    )


@functools.cache
def _compile(
    device_index,
    compressed_k,
    split,
    dequant_warps=8,
    phase_chunks=8,
    narrow_offsets=False,
    native_fp4=None,
    persistent_ctas=0,
    compact_metadata=False,
):
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("warm DS4.1 Frost decode before CUDA Graph capture")
    import cutlass
    import cutlass.cute as cute
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from ...cute_dsl.attention.dsa import hca_helpers
    from . import _metadata, hca_v41, hca_v41_primitives

    if native_fp4 is None:
        # cute.compile uses CUTLASS's bundled compiler, not CUDA_HOME.
        native_fp4 = (cutlass.CUDA_VERSION.major, cutlass.CUDA_VERSION.minor) >= (13, 2)

    sb, sq, sh, spw, spc, sws = (cute.sym_int() for _ in range(6))
    sd = cute.sym_int(divisibility=16)
    f = cute.runtime.make_fake_compact_tensor
    q = f(
        cutlass.BFloat16, (sb, sq, sh, sd), stride_order=(3, 2, 1, 0), assumed_align=16
    )
    w = f(cutlass.Uint8, (spw, 64 * 528), stride_order=(1, 0), assumed_align=16)
    c = f(cutlass.Uint8, (spc, 64 * 288), stride_order=(1, 0), assumed_align=16)
    wi = f(cutlass.Int32, (sb, 128), stride_order=(1, 0), assumed_align=16)
    ci = f(
        cutlass.Int32,
        (sb, max(64, compressed_k)),
        stride_order=(1, 0),
        assumed_align=16,
    )
    o = f(
        cutlass.BFloat16, (sb, sq, sh, sd), stride_order=(3, 2, 1, 0), assumed_align=16
    )
    lse = f(cutlass.Float32, (sb, sq, sh), stride_order=(2, 1, 0), assumed_align=16)
    ws = f(cutlass.Int8, (sws,), assumed_align=32) if split else None
    lengths = f(cutlass.Int32, (sb,), assumed_align=16)
    valid = f(cutlass.Int32, (sb,), assumed_align=16)
    sink = f(cutlass.Float32, (64,), assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    sm_count = device_sm_count(device_index)
    # WS is selected only for the measured K512 large-batch configuration.
    use_ws = dequant_warps == 16 and not split
    assert not persistent_ctas or use_ws
    kernel = hca_v41.BlackwellV41MixedCacheDecode(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(64, 64),
        mma_pv_tiler_mn=(64, 128),
        max_active_clusters=persistent_ctas or sm_count,
        page_size_cmp=64,
        skip_correction_threshold=6.0,
        is_persistent=bool(persistent_ctas),
        is_var_seq=True,
        is_var_split_kv=False,
        is_causal=True,
        seq_len_q=1,
        hca_compress_ratio=128,
        window_page_size=64,
        compressed_page_size=64,
        compressed_is_fp4=True,
        dequant_warps=dequant_warps,
        dequant_phase_chunks=phase_chunks,
        narrow_offsets=narrow_offsets,
        native_fp4=native_fp4,
        use_ws=use_ws,
        compact_metadata=compact_metadata,
    )
    return build_and_load_cute_dsl_kernel(
        "deepseek_v41_decode",
        f"v4_workspace_h64_s1_c{compressed_k}_split{int(split)}"
        f"_dq{dequant_warps}_phase{phase_chunks}_i{32 if narrow_offsets else 64}"
        f"_nativefp4{int(native_fp4)}_ws{int(use_ws)}_ctas{persistent_ctas}_sm{sm_count}"
        f"_meta16{int(compact_metadata)}_raw_bmm1_split_stats_v2",
        lambda: cute.compile(
            kernel,
            q,
            w,
            c,
            wi,
            ci,
            o,
            lse,
            ws,
            cutlass.Int32(1),
            lengths,
            None,
            lengths,
            valid,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            sink,
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        ),
        extra_key_files=(
            __file__,
            _workspace.__file__,
            _metadata.__file__,
            hca_v41.__file__,
            hca_v41_primitives.__file__,
            hca_helpers.__file__,
        ),
    )


def decode_mixed(
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
):
    bound = bind_inputs(
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
        backend="cute-dsl",
    )
    batch, ck, views = bound.batch, bound.compressed_k, bound.views
    split = split_kv(batch, ck)
    dequant_warps, phase_chunks, narrow_offsets = 8, 8, False
    largest_pool = max(bound.swa_cache.numel(), bound.main_cache.numel())
    if ck == 512 and batch >= 128:
        dequant_warps, phase_chunks = 16, 2
    elif ck == 512 and batch >= 64 and largest_pool < 2**31:
        phase_chunks, narrow_offsets = 2, True
    with torch.cuda.device(query.device):
        persistent_ctas = 0
        if dequant_warps == 16:
            sm_count = device_sm_count(query.device.index)
            if batch > sm_count:
                waves = (batch + sm_count - 1) // sm_count
                persistent_ctas = (batch + waves - 1) // waves
        kernel = _compile(
            query.device.index,
            ck,
            split > 1,
            dequant_warps,
            phase_chunks,
            narrow_offsets,
            persistent_ctas=persistent_ctas,
            compact_metadata=bool(persistent_ctas) and largest_pool < 2**35,
        )
        prepare = _compile_prepare(query.device.index, ck, sinks is not None)
        from cutlass import Int32, Int64, Float32

        # Compile both specializations before the first metadata write. Warm
        # cache hits need neither JIT nor device-property queries in capture.
        prepare(
            bound.indices,
            bound.topk_lens,
            bound.seq_lens,
            views.sink if sinks is None else sinks,
            views.selected_swa,
            views.selected_main,
            views.swa_lengths,
            views.main_lengths,
            views.total_lengths,
            views.swa_counts,
            views.main_counts,
            views.sink,
            Int64(bound.swa_cache.shape[0] * 64),
            Int64(bound.main_cache.shape[0] * 64 if ck else 0),
        )
        kernel(
            bound.query,
            bound.swa_cache.view(bound.swa_cache.shape[0], -1),
            bound.main_cache.view(bound.main_cache.shape[0], -1),
            views.selected_swa,
            views.selected_main,
            bound.out,
            views.lse.view(batch, 1, 64),
            views.frost_scratch,
            Int32(split),
            views.total_lengths,
            None,
            views.total_lengths,
            views.swa_lengths,
            Float32(bound.bmm1_scale),
            Float32(bound.bmm2_scale),
            views.sink,
        )
    return bound.out
