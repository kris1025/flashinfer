# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""V4.1 native FP8 SWA adapter for D512 Prims-TS."""

from dataclasses import dataclass
import functools
import hashlib
from pathlib import Path

import torch

from ._utils import _overlaps
from .decode import _validate_inputs


def _make_fp8_sources(compressed_k, sm_count, native_fp4):
    from ...attention.prims_ts.kernels.mla_decode.sparse_policy import SparseMlaProfile
    from ...attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )
    from ...attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
        MlaProfile,
    )
    from .prims_ts_fp8 import Fp32SourceStore
    from .prims_ts_resources import Fp4KvProducer

    def source(dtype, capacity, tile_k, producer=None):
        attention = ThroughputLatencyMlaDecodeTs(
            batch_size=1,
            num_heads=64,
            seq_len_q=1,
            seq_len_k=max(256, ((capacity + 127) // 128) * 128),
            latent_dim=512,
            rope_dim=0,
            page_size=1,
            max_active_clusters=sm_count,
            qkv_dtype=dtype,
            out_dtype="bf16",
            profile=MlaProfile(
                name="dsv41_fp8_swa_source" if dtype == "e4m3" else "dsv41_fp4_source",
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
            sparse_profile=SparseMlaProfile(
                family="keep",
                tile_size_q=64,
                split_kv=1,
                scheduler="nonpersistent",
                head_dim_ctas=1,
                gather_issue_warps=4,
                direct_inputs=True,
                reuse_kv=True,
                reuse_kv_stages=8,
                kv_tile_size=tile_k,
                page_pipeline_stages=1,
            ),
            finalize_output=False,
            kv_producer=producer,
            source_state_store=Fp32SourceStore(),
        )
        attention.direct_static_scales = True
        attention.direct_sparse_capacities = (capacity, 0)
        return attention

    # FP8 requires BK128 and four D128 stages. Eight stages retain two full
    # tiles; the independent global FP4 producer keeps its BK64 topology.
    swa = source("e4m3", 128, 128)
    main = (
        source(
            "bf16",
            compressed_k,
            64,
            Fp4KvProducer(native_fp4=native_fp4),
        )
        if compressed_k
        else None
    )
    return swa, main


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
    if native_fp4 is None:
        native_fp4 = compiler >= (13, 2)
    if native_fp4 and compiler < (13, 2):
        raise RuntimeError("native FP4 conversion requires bundled CUDA >= 13.2")

    from ...attention.prims_ts.kernels.mla_decode.throughput_latency_1cta import kernel
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from . import hca_v41_primitives, prims_ts_fp8, prims_ts_resources

    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    swa_attention, global_attention = _make_fp8_sources(
        compressed_k, sm_count, native_fp4
    )
    decode_kernel = prims_ts_fp8.Fp8SwaDecodeKernel(
        swa_attention, global_attention, compressed_k
    )
    batch = cute.sym_int()
    f = cute.runtime.make_fake_compact_tensor

    def query(dtype):
        return cute.runtime.make_fake_tensor(
            dtype,
            (64, 512, 1, batch),
            (512, 1, 64 * 512, 64 * 512),
            assumed_align=16,
        )

    q = query(cutlass.Float8E4M3FN)
    expanded_q = query(cutlass.BFloat16)
    out = query(cutlass.BFloat16)
    swa = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (1, 512, cute.sym_int64(divisibility=64)),
        (512, 1, 512),
        assumed_align=16,
    )
    global_cache = f(
        cutlass.Uint8,
        (cute.sym_int64(divisibility=64 * 288),),
        assumed_align=16,
    )

    def indices(width):
        return f(cutlass.Int32, (batch, width), stride_order=(1, 0), assumed_align=16)

    def lengths():
        return f(cutlass.Int32, (batch,), assumed_align=16)

    def source_out():
        return f(
            cutlass.Float32,
            (batch, 64, 512),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )

    def source_lse():
        return f(cutlass.Float32, (batch, 64), stride_order=(1, 0), assumed_align=16)

    wi, selected_swa = indices(128), indices(128)
    ci, selected_global = indices(max(64, compressed_k)), indices(max(64, compressed_k))
    swa_lengths, global_lengths, empty_lengths = lengths(), lengths(), lengths()
    swa_counts, global_counts = lengths(), lengths()
    part_s, part_c = source_out(), source_out()
    lse_s, lse_c, lse = source_lse(), source_lse(), source_lse()
    sink = f(cutlass.Float32, (64,), assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    signature = (
        "dsv41_unit_e4m3_q_swa512_fp32_sources_v1",
        swa_attention.compile_signature(),
        global_attention.compile_signature() if global_attention is not None else None,
        "source_o_fp32",
        "source_lse_log2",
        "public_o_bf16_lse_ln",
        cutlass.__version__,
        compiler,
    )
    key = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    native_root = Path(kernel.__file__).resolve().parents[3]
    return build_and_load_cute_dsl_kernel(
        "deepseek_v41_prims_ts_fp8",
        f"h64_d512_swa128_c{compressed_k}_keep_bk128_globalbk64_kv8_split1_i64"
        f"_fp32src_nativefp4{int(native_fp4)}_sm{sm_count}_{key}",
        lambda: cute.compile[cute.FrontendNext](
            decode_kernel,
            q,
            expanded_q,
            swa,
            global_cache,
            wi,
            ci,
            selected_swa,
            selected_global,
            swa_lengths,
            global_lengths,
            empty_lengths,
            swa_counts,
            global_counts,
            part_s,
            part_c,
            lse_s,
            lse_c,
            sink,
            out,
            lse,
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        ),
        extra_key_files=(
            __file__,
            prims_ts_fp8.__file__,
            prims_ts_resources.__file__,
            hca_v41_primitives.__file__,
            *sorted(native_root.rglob("*.py")),
        ),
    )


def decode(q, swa_cache, global_cache, swa_indices, global_indices, sink, *, plan=None):
    signature = _validate_inputs(
        q,
        swa_cache,
        global_cache,
        swa_indices,
        global_indices,
        sink,
        allow_fp8_swa=True,
    )
    if q.dtype != torch.float8_e4m3fn:
        raise ValueError("prims-ts requires explicit E4M3 Q and unit-scale E4M3 SWA")
    return _decode_fp8(
        q,
        swa_cache,
        global_cache,
        swa_indices,
        global_indices,
        sink,
        signature=signature,
        plan=plan,
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
    empty_lengths: torch.Tensor
    swa_counts: torch.Tensor
    global_counts: torch.Tensor
    part_s: torch.Tensor
    part_c: object
    lse_s: torch.Tensor
    lse_c: object
    expanded_q: object
    empty_cache: torch.Tensor
    empty_indices: torch.Tensor


def _decode_fp8(
    q, swa_cache, global_cache, swa_indices, global_indices, sink, *, signature, plan
):
    if plan is not None and (
        not isinstance(plan, _Plan) or plan.signature != signature
    ):
        raise ValueError("decode plan declaration mismatch")
    batch = q.shape[0]
    ck = 0 if global_indices is None else global_indices.shape[2]
    inputs = tuple(
        x
        for x in (q, swa_cache, global_cache, swa_indices, global_indices, sink)
        if x is not None
    )
    with torch.cuda.device(q.device):
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("prepare a decode plan before CUDA Graph capture")

            def allocate(shape, dtype):
                return torch.empty(shape, device=q.device, dtype=dtype)

            plan = _Plan(
                signature=signature,
                kernel=_compile(q.device.index, ck),
                split=1,
                out=allocate(q.shape, torch.bfloat16),
                lse=allocate((batch, 64, 1), torch.float32),
                selected_swa=allocate((batch, 128), torch.int32),
                selected_global=allocate((batch, max(64, ck)), torch.int32),
                swa_lengths=allocate((batch,), torch.int32),
                global_lengths=allocate((batch,), torch.int32),
                empty_lengths=allocate((batch,), torch.int32),
                swa_counts=allocate((batch,), torch.int32),
                global_counts=allocate((batch,), torch.int32),
                part_s=allocate((batch, 64, 512), torch.float32),
                part_c=allocate((batch, 64, 512), torch.float32) if ck else None,
                lse_s=allocate((batch, 64), torch.float32),
                lse_c=allocate((batch, 64), torch.float32) if ck else None,
                expanded_q=allocate(q.shape, torch.bfloat16) if ck else None,
                empty_cache=allocate((1, 64, 1, 288), torch.uint8),
                empty_indices=torch.full(
                    (batch, 64), -1, device=q.device, dtype=torch.int32
                ),
            )
        owned = tuple(
            x
            for x in (
                plan.out,
                plan.lse,
                plan.selected_swa,
                plan.selected_global,
                plan.swa_lengths,
                plan.global_lengths,
                plan.empty_lengths,
                plan.swa_counts,
                plan.global_counts,
                plan.part_s,
                plan.part_c,
                plan.lse_s,
                plan.lse_c,
                plan.expanded_q,
                plan.empty_cache,
                plan.empty_indices,
            )
            if x is not None
        )
        if any(_overlaps(dst, src) for dst in owned for src in inputs):
            raise ValueError("decode inputs must not overlap plan-owned buffers")
        if torch.is_grad_enabled() and any(x.requires_grad for x in owned):
            raise ValueError("DS4.1 decode is inference-only")
        main = plan.empty_cache if global_cache is None else global_cache
        ids = plan.empty_indices if global_indices is None else global_indices
        # Window-only has no BF16 query scratch or global states. The unused
        # tensor ABI slots alias existing buffers; its compiled branch cannot
        # read them. Final O always has the public BF16 dtype, independent of Q.
        expanded_q = plan.out if plan.expanded_q is None else plan.expanded_q
        part_c = plan.part_s if plan.part_c is None else plan.part_c
        lse_c = plan.lse_s if plan.lse_c is None else plan.lse_c
        plan.kernel(
            q.permute(2, 3, 1, 0),
            expanded_q.permute(2, 3, 1, 0),
            swa_cache.view(-1, 1, 512).permute(1, 2, 0),
            main.view(-1),
            swa_indices.view(batch, 128),
            ids.view(batch, -1),
            plan.selected_swa,
            plan.selected_global,
            plan.swa_lengths,
            plan.global_lengths,
            plan.empty_lengths,
            plan.swa_counts,
            plan.global_counts,
            plan.part_s,
            part_c,
            plan.lse_s,
            lse_c,
            sink,
            plan.out.permute(2, 3, 1, 0),
            plan.lse.view(batch, 64),
        )
    return plan.out, plan.lse, plan
