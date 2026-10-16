# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Optional BF16-Q/MXFP8-SWA compatibility adapter for D512 Prims-TS."""

from dataclasses import dataclass
import functools
import hashlib
from pathlib import Path

import torch

from ._utils import _overlaps


@functools.cache
def _compile(device_index, compressed_k, *, native_fp4=None):
    import cutlass
    import cutlass.cute as cute
    from packaging.version import Version

    if Version(cutlass.__version__) < Version("4.8.0"):
        raise RuntimeError("V4.1 prims-ts requires CuTe-DSL >= 4.8.0")
    compiler = (cutlass.CUDA_VERSION.major, cutlass.CUDA_VERSION.minor)
    if compiler[0] < 13:
        raise RuntimeError("V4.1 prims-ts requires the CuTe-DSL CUDA 13 compiler")

    from ...attention.prims_ts.kernels.mla_decode.sparse_policy import SparseMlaProfile
    from ...attention.prims_ts.kernels.mla_decode.throughput_latency_1cta import kernel
    from ...attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
        MlaProfile,
    )
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import hca_v41_primitives, prims_ts_mxfp8_resources, prims_ts_resources

    # This is a packed-cache bring-up schedule, not the native-cache GB300
    # crossover policy. Two complete K tiles stay resident for delayed PV.
    profile = SparseMlaProfile(
        family="keep",
        tile_size_q=64,
        split_kv=1,
        scheduler="nonpersistent",
        head_dim_ctas=1,
        gather_issue_warps=4,
        direct_inputs=True,
        reuse_kv=True,
        reuse_kv_stages=8,
        kv_tile_size=64,
        page_pipeline_stages=1,
    )
    if native_fp4 is None:
        # cute.compile uses the bundled compiler, not CUDA_HOME.
        native_fp4 = compiler >= (13, 2)
    if native_fp4 and compiler < (13, 2):
        raise RuntimeError("native FP4 conversion requires bundled CUDA >= 13.2")
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    capacity = max(256, 128 + ((compressed_k + 127) // 128) * 128)
    attention = kernel.ThroughputLatencyMlaDecodeTs(
        batch_size=1,
        num_heads=64,
        seq_len_q=1,
        seq_len_k=capacity,
        latent_dim=512,
        rope_dim=0,
        page_size=1,
        max_active_clusters=sm_count,
        qkv_dtype="bf16",
        out_dtype="bf16",
        profile=MlaProfile(
            name="dsv41_packed",
            kernel_variant="keeps_mma_ab",
            tile_size_q=64,
            use_persistent_scheduler=0,
        ),
        logical_num_heads=64,
        logical_seq_len_q=1,
        tile_size_q=64,
        explicit_split_kv=1,
        explicit_persistent=False,
        reduction_mode="gmem_separate",
        mask_type="dense",
        sparse_profile=profile,
        finalize_output=True,
        kv_producer=prims_ts_mxfp8_resources.PackedKvProducer(native_fp4=native_fp4),
    )
    attention.direct_sparse = True
    attention.direct_static_scales = True
    attention.direct_sparse_capacities = (128, compressed_k)
    packed = prims_ts_mxfp8_resources.PackedDecodeKernel(attention, compressed_k)

    batch = cute.sym_int()
    q = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (64, 512, 1, batch),
        (512, 1, 64 * 512, 64 * 512),
        assumed_align=16,
    )
    out = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (64, 512, 1, batch),
        (512, 1, 64 * 512, 64 * 512),
        assumed_align=16,
    )
    lse = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (64, 1, batch),
        (1, 64, 64),
        assumed_align=16,
    )
    f = cute.runtime.make_fake_compact_tensor
    # Storage extents, including pools larger than 2 GiB, remain 64-bit.
    swa = f(
        cutlass.Uint8,
        (cute.sym_int64(divisibility=64 * 528),),
        assumed_align=16,
    )
    global_cache = f(
        cutlass.Uint8,
        (cute.sym_int64(divisibility=64 * 288),),
        assumed_align=16,
    )
    wi = f(cutlass.Int32, (batch, 128), stride_order=(1, 0), assumed_align=16)
    ci = f(
        cutlass.Int32,
        (batch, max(64, compressed_k)),
        stride_order=(1, 0),
        assumed_align=16,
    )
    selected_swa = f(cutlass.Int32, (batch, 128), stride_order=(1, 0), assumed_align=16)
    selected_global = f(
        cutlass.Int32,
        (batch, max(64, compressed_k)),
        stride_order=(1, 0),
        assumed_align=16,
    )
    swa_lengths = f(cutlass.Int32, (batch,), assumed_align=16)
    global_lengths = f(cutlass.Int32, (batch,), assumed_align=16)
    sink = f(cutlass.Float32, (64,), assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    signature = (attention.compile_signature(), cutlass.__version__, compiler)
    key = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    # Include transitive native task/math/config sources, not just the packed
    # adapter. A native helper edit must invalidate this separate JIT namespace.
    native_root = Path(kernel.__file__).resolve().parents[3]
    sources = (
        __file__,
        prims_ts_resources.__file__,
        prims_ts_mxfp8_resources.__file__,
        hca_v41_primitives.__file__,
        *sorted(native_root.rglob("*.py")),
    )
    return build_and_load_cute_dsl_kernel(
        "deepseek_v41_prims_ts_mxfp8",
        f"h64_d512_swa128_c{compressed_k}_keep_bk64_kv8_split1_i64"
        f"_nativefp4{int(native_fp4)}_sm{sm_count}_{key}",
        lambda: cute.compile[cute.FrontendNext](
            packed,
            q,
            swa,
            global_cache,
            wi,
            ci,
            selected_swa,
            selected_global,
            swa_lengths,
            global_lengths,
            sink,
            out,
            lse,
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        ),
        extra_key_files=sources,
    )


@dataclass(frozen=True)
class _Plan:
    signature: tuple
    kernel: object
    split: int
    out: torch.Tensor
    lse: torch.Tensor
    selected_swa: torch.Tensor
    selected_global: torch.Tensor
    swa_lengths: torch.Tensor
    global_lengths: torch.Tensor
    empty_cache: torch.Tensor
    empty_indices: torch.Tensor


def decode(
    q,
    swa_cache,
    global_cache,
    swa_indices,
    global_indices,
    sink,
    *,
    signature,
    plan=None,
):
    batch = q.shape[0]
    inputs = tuple(
        x
        for x in (q, swa_cache, global_cache, swa_indices, global_indices, sink)
        if x is not None
    )
    if plan is not None and (
        not isinstance(plan, _Plan) or plan.signature != signature
    ):
        raise ValueError("decode plan declaration mismatch")
    with torch.cuda.device(q.device):
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("prepare a decode plan before CUDA Graph capture")
            ck = 0 if global_indices is None else global_indices.shape[2]
            plan = _Plan(
                signature=signature,
                kernel=_compile(q.device.index, ck),
                split=1,
                out=torch.empty_like(q),
                lse=torch.empty((batch, 64, 1), device=q.device, dtype=torch.float32),
                selected_swa=torch.empty(
                    (batch, 128), device=q.device, dtype=torch.int32
                ),
                selected_global=torch.empty(
                    (batch, max(64, ck)), device=q.device, dtype=torch.int32
                ),
                swa_lengths=torch.full(
                    (batch,), 128, device=q.device, dtype=torch.int32
                ),
                global_lengths=torch.full(
                    (batch,), ck, device=q.device, dtype=torch.int32
                ),
                empty_cache=torch.empty(
                    (1, 64, 1, 288), device=q.device, dtype=torch.uint8
                ),
                empty_indices=torch.full(
                    (batch, 64), -1, device=q.device, dtype=torch.int32
                ),
            )
        owned = (
            plan.out,
            plan.lse,
            plan.selected_swa,
            plan.selected_global,
            plan.swa_lengths,
            plan.global_lengths,
            plan.empty_cache,
            plan.empty_indices,
        )
        if any(_overlaps(dst, src) for dst in owned for src in inputs):
            raise ValueError("decode inputs must not overlap plan-owned buffers")
        if torch.is_grad_enabled() and any(x.requires_grad for x in owned):
            raise ValueError("DS4.1 decode is inference-only")
        main = plan.empty_cache if global_cache is None else global_cache
        ids = plan.empty_indices if global_indices is None else global_indices
        plan.kernel(
            q.permute(2, 3, 1, 0),
            swa_cache.view(-1),
            main.view(-1),
            swa_indices.view(batch, 128),
            ids.view(batch, -1),
            plan.selected_swa,
            plan.selected_global,
            plan.swa_lengths,
            plan.global_lengths,
            sink,
            plan.out.permute(2, 3, 1, 0),
            plan.lse.view(batch, 1, 64).permute(2, 1, 0),
        )
    return plan.out, plan.lse, plan
