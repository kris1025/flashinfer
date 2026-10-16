# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: BSD-3-Clause
#
# Packed conversion/store sequences adapted from hca_v41.py and
# hca_v41_primitives.py.
"""Mixed MXFP8/FP4 tiles for the D512 Prims-TS Keep-AB graph.

Page64/group32 MXFP8 SWA and page64/group16 FP4 global selections share the
concrete FP4 resource's retained BF16 stages, stores and all-writer pipeline.
Source spans are 128-aligned, so a BK64 tile never mixes formats.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.resources import StageInfo

from ...attention.prims_ts.kernels.mla_decode.helpers.stage import MlaStage
from . import hca_v41_primitives as hw
from .prims_ts_resources import Fp4KvProducer, Fp4SmemKvResource


@dataclass(frozen=True)
class PackedKvProducer(Fp4KvProducer):
    def compile_signature(self):
        return ("dsv41_page64_mxfp8_group32_fp4_group16", self.native_fp4)

    def make_resource(self, pools, **kwargs):
        return PackedSmemKvResource(
            fp4_cache=pools[1],
            swa_cache=pools[0],
            native_fp4=self.native_fp4,
            **kwargs,
        )


@dataclass(kw_only=True)
class PackedSmemKvResource(Fp4SmemKvResource):
    swa_cache: object = None

    @cute.jit
    def _store_fp8_chunk(self, chunk, scale_byte, stage_base, row, dim):
        bits = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Int32)
        code = scale_byte & 255
        # E8M0 code zero is 2**-127, not a zero scale.
        bits[0] = cutlass.Int32(0x00400000) if code == 0 else code << 23
        scale = cute.make_tensor(
            cute.recast_ptr(bits.iterator, dtype=cutlass.Float32), bits.layout
        )[0]
        fp8 = cute.make_tensor(
            cute.recast_ptr(chunk.iterator, dtype=cutlass.Float8E4M3FN),
            cute.make_layout(16),
        )
        for half in cutlass.range_constexpr(2):
            values = (
                cute.make_tensor(fp8.iterator + half * 8, cute.make_layout(8))
                .load()
                .to(cutlass.Float32)
            )
            fragment = cute.make_rmem_tensor(cute.make_layout(8), cutlass.BFloat16)
            fragment.store((values * scale).to(cutlass.BFloat16))
            self._store_fragment(fragment, stage_base, row, dim + half * 8)

    @cute.jit
    def _load_mxfp8_stage(
        self,
        pool,
        stage_base,
        cached_page_ids,
        *,
        stage_idx: cutlass.Constexpr[int],
        page_id_slot: cutlass.Constexpr[int],
    ):
        lane = cute.arch.thread_idx()[0] % 32
        warp = cute.arch.thread_idx()[0] // 32 - self.cfg.load_warp_idx
        chunk_elements = 16
        chunks_per_row = 128 // chunk_elements
        rows_per_group = 32 // chunks_per_row
        rows_per_warp = self.cfg.tile_size_kv // self.cfg.load_num_warps
        row_bytes, scale_bytes = 512, 16
        capacity = cutlass.Int64(cute.size(pool)) // (row_bytes + scale_bytes)
        base = page_id_slot * self.cfg.sparse_cache_words
        for group in cutlass.range_constexpr(rows_per_warp // rows_per_group):
            local_row = group * rows_per_group + lane // chunks_per_row
            row = warp * rows_per_warp + local_row
            chunk_idx = lane % chunks_per_row
            raw = cutlass.Int32(0x7FFFFFFF)
            # Page staging caches one quad in each of lanes 0:4. All writing
            # lanes recover their row from that cache; no index reread per D.
            for entry in cutlass.range_constexpr(4):
                value = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=cached_page_ids[base + entry],
                    offset=local_row // 4,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                if local_row % 4 == entry:
                    raw = cutlass.Int32(value)
            slot = raw & cutlass.Int32(0x7FFFFFFF)
            chunk = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
            for word in cutlass.range_constexpr(4):
                chunk[word] = 0
            scale_word = cutlass.Int32(0)
            if slot != cutlass.Int32(0x7FFFFFFF) and cutlass.Int64(slot) < capacity:
                # Page data and scale planes are separate. Widen before every
                # byte-stride multiplication, including pools above 2 GiB.
                page_base = cutlass.Int64(slot // 64) * (64 * (row_bytes + scale_bytes))
                token = cutlass.Int64(slot & 63)
                column = stage_idx * 128 + chunk_idx * chunk_elements
                data_offset = page_base + token * row_bytes + cutlass.Int64(column)
                scale_offset = (
                    page_base
                    + 64 * row_bytes
                    + token * scale_bytes
                    + cutlass.Int64(column // 32)
                )
                data_ptr = cute.make_ptr(
                    cutlass.Int32,
                    pool.iterator.toint() + data_offset,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                chunk.store(hw.load_cache_vector(data_ptr))
                scale_word = hw.load_cache_scale(
                    pool.iterator + scale_offset, cutlass.Uint8
                )
            dim = chunk_idx * chunk_elements
            self._store_fp8_chunk(chunk, scale_word, stage_base, row, dim)

    @cute.jit
    def _producer_load_kv_stage(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        del inst_id, is_v, section
        first = prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=cached_page_ids[page_id_slot * self.cfg.sparse_cache_words],
            offset=0,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.IDX,
        )
        stage_base = self._stage_base(stage_info)
        # Source spans are 128-aligned; a BK64 tile never mixes formats.
        if cutlass.Int32(first) < 0:
            self._load_fp4_stage(
                self.fp4_cache,
                stage_base,
                cached_page_ids,
                stage_idx=stage_idx,
                page_id_slot=page_id_slot,
            )
        else:
            self._load_mxfp8_stage(
                self.swa_cache,
                stage_base,
                cached_page_ids,
                stage_idx=stage_idx,
                page_id_slot=page_id_slot,
            )
        # Publication is the task graph's AsyncThread producer commit. Each
        # writer fences its own stores; all 128 arrivals precede UMMA's wait.
        cute.arch.fence_view_async_shared()


class PackedDecodeKernel:
    """Refresh selection holes and launch the existing sparse arithmetic graph."""

    def __init__(self, attention, compressed_k):
        self.attention = attention
        self.compressed_k = compressed_k

    @cute.kernel
    def _sanitize_indices(
        self,
        swa_indices: cute.Tensor,
        global_indices: cute.Tensor,
        selected_swa: cute.Tensor,
        selected_global: cute.Tensor,
        swa_capacity: cutlass.Int64,
        global_capacity: cutlass.Int64,
    ):
        request = cute.arch.block_idx()[0]
        lane = cute.arch.thread_idx()[0]
        index = cutlass.Int32(swa_indices[request, lane])
        selected_swa[request, lane] = (
            index
            if index >= 0
            and index != cutlass.Int32(0x7FFFFFFF)
            and cutlass.Int64(index) < swa_capacity
            else cutlass.Int32(-1)
        )
        if cutlass.const_expr(self.compressed_k > 0):
            for group in cutlass.range_constexpr((self.compressed_k + 127) // 128):
                column = lane + group * 128
                if column < self.compressed_k:
                    index = cutlass.Int32(global_indices[request, column])
                    selected_global[request, column] = (
                        index
                        if index >= 0
                        and index != cutlass.Int32(0x7FFFFFFF)
                        and cutlass.Int64(index) < global_capacity
                        else cutlass.Int32(-1)
                    )
        else:
            if lane < 64:
                selected_global[request, lane] = cutlass.Int32(-1)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        swa_cache: cute.Tensor,
        global_cache: cute.Tensor,
        swa_indices: cute.Tensor,
        global_indices: cute.Tensor,
        selected_swa: cute.Tensor,
        selected_global: cute.Tensor,
        swa_lengths: cute.Tensor,
        global_lengths: cute.Tensor,
        sink: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor,
        stream: object,
    ):
        # Fixed list lengths preserve duplicate positions and negative holes.
        # Sanitizing positive out-of-capacity values makes the softmax mask
        # agree with the packed loader's defensive load suppression.
        self._sanitize_indices(
            swa_indices,
            global_indices,
            selected_swa,
            selected_global,
            cutlass.Int64(cute.size(swa_cache)) // 528,
            cutlass.Int64(cute.size(global_cache)) // 288,
        ).launch(
            grid=(swa_indices.shape[0], 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )
        direct_args = (
            selected_swa,
            selected_global,
            swa_lengths,
            global_lengths,
            None,
            None,
            None,
            None,
            sink,
        )
        self.attention(
            q,
            q,
            swa_cache,
            global_cache,
            None,
            out,
            lse,
            None,
            cutlass.Int32(1),
            swa_lengths,
            None,
            direct_args,
            cutlass.Float32(512**-0.5),
            cutlass.Float32(1.0),
            stream,
        )
