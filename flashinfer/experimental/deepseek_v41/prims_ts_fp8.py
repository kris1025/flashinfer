# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Native E4M3 attention and FP32 source states for the V4.1 adapter.

SWA stays native FP8 through TMA, QK, P and PV. The optional main pass rounds
selected scaled FP4 tiles to E4M3 and uses the original FP8 Q. Neither pass
rounds its normalized O before merging.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from ...attention.prims_ts.kernels.separate_reduction import MergeMlaSourceStates


@dataclass(frozen=True)
class Fp32SourceStore:
    block_size: int = 16

    def compile_signature(self):
        return ("dsv41_fp32_normalized_source_v1", self.block_size, "log2_lse")

    @cute.jit
    def store(self, args, values, row, column, *, lane_stride=16, rotary=False):
        del lane_stride, rotary
        out, counts = args
        width = values.shape[0]
        stored = cutlass.Array(cutlass.Float32, width, space=cutlass.AddressSpace.rmem)
        have_source = cutlass.Int32(counts[row // 64]) > cutlass.Int32(0)
        for i in cutlass.range_constexpr(width):
            stored[i] = values[i] if have_source else cutlass.Float32(0.0)
        # Native FP8's normalization has already undone P448 in O units. Only
        # the LSE producer compensates its probability frame separately.
        offset = cutlass.Int64(row) * cutlass.Int64(512) + cutlass.Int64(column)
        cutlass.Pointer(out.iterator + offset, dtype=cutlass.Float32).store(
            stored.load(0, width), alignment=width * 4
        )


class Fp8SwaDecodeKernel:
    def __init__(self, swa_attention, global_attention, compressed_k):
        self.swa_attention = swa_attention
        self.global_attention = global_attention
        self.compressed_k = compressed_k
        self.merge = MergeMlaSourceStates(compressed_k > 0)

    @cute.kernel
    def _prepare(
        self,
        swa_indices: cute.Tensor,
        global_indices: cute.Tensor,
        selected_swa: cute.Tensor,
        selected_global: cute.Tensor,
        swa_lengths: cute.Tensor,
        global_lengths: cute.Tensor,
        empty_lengths: cute.Tensor,
        swa_counts: cute.Tensor,
        global_counts: cute.Tensor,
        swa_capacity: cutlass.Int64,
        global_capacity: cutlass.Int64,
    ):
        request = cute.arch.block_idx()[0]
        lane = cute.arch.thread_idx()[0]
        if lane == 0:
            swa_lengths[request] = cutlass.Int32(128)
            global_lengths[request] = cutlass.Int32(self.compressed_k)
            empty_lengths[request] = cutlass.Int32(0)
            swa_counts[request] = cutlass.Int32(0)
            global_counts[request] = cutlass.Int32(0)
        cute.arch.sync_threads()

        index = cutlass.Int32(swa_indices[request, lane])
        valid_s = (
            (index >= cutlass.Int32(0))
            & (index != cutlass.Int32(0x7FFFFFFF))
            & (cutlass.Int64(index) < swa_capacity)
        )
        selected_swa[request, lane] = index if valid_s else cutlass.Int32(-1)
        valid_c = cutlass.Int32(0)
        if cutlass.const_expr(self.compressed_k > 0):
            for group in cutlass.range_constexpr((self.compressed_k + 127) // 128):
                column = lane + group * 128
                if column < self.compressed_k:
                    global_index = cutlass.Int32(global_indices[request, column])
                    keep = (
                        (global_index >= cutlass.Int32(0))
                        & (global_index != cutlass.Int32(0x7FFFFFFF))
                        & (cutlass.Int64(global_index) < global_capacity)
                    )
                    selected_global[request, column] = (
                        global_index if keep else cutlass.Int32(-1)
                    )
                    valid_c += cutlass.Int32(keep)
        elif lane < 64:
            selected_global[request, lane] = cutlass.Int32(-1)

        # Execution spans retain holes/duplicates; counts describe actual live
        # positions so empty sources never contribute poisoned partial states.
        count_s = cute.arch.warp_reduction_sum(cutlass.Int32(valid_s))
        count_c = cute.arch.warp_reduction_sum(valid_c)
        if lane % 32 == 0:
            cute.arch.atomic_add(swa_counts.iterator + request, count_s, scope="cta")
            cute.arch.atomic_add(global_counts.iterator + request, count_c, scope="cta")

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
        empty_lengths: cute.Tensor,
        swa_counts: cute.Tensor,
        global_counts: cute.Tensor,
        part_s: cute.Tensor,
        part_c: cute.Tensor,
        lse_s: cute.Tensor,
        lse_c: cute.Tensor,
        sink: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor,
        stream: object,
    ):
        self._prepare(
            swa_indices,
            global_indices,
            selected_swa,
            selected_global,
            swa_lengths,
            global_lengths,
            empty_lengths,
            swa_counts,
            global_counts,
            cutlass.Int64(swa_cache.shape[2]),
            cutlass.Int64(cute.size(global_cache)) // 288,
        ).launch(
            grid=(swa_indices.shape[0], 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )
        source_lse_layout = cute.make_layout(
            (64, 1, swa_indices.shape[0]), stride=(1, 64, 64)
        )
        swa_lse = cute.make_tensor(lse_s.iterator, source_lse_layout)
        # Each pass has only a primary source. Extra slots bind a real tensor
        # with length zero, matching the native compiled direct-index ABI.
        direct_swa = (
            selected_swa,
            selected_global,
            swa_lengths,
            empty_lengths,
            None,
            None,
            None,
            None,
            None,
        )
        self.swa_attention(
            q,
            q,
            swa_cache,
            swa_cache,
            None,
            out,
            swa_lse,
            None,
            cutlass.Int32(1),
            swa_lengths,
            None,
            direct_swa,
            cutlass.Float32(512**-0.5),
            cutlass.Float32(1.0),
            stream,
            source_state_store_args=(part_s, swa_counts),
        )
        if cutlass.const_expr(self.compressed_k > 0):
            global_lse = cute.make_tensor(lse_c.iterator, source_lse_layout)
            direct_global = (
                selected_global,
                selected_swa,
                global_lengths,
                empty_lengths,
                None,
                None,
                None,
                None,
                None,
            )
            self.global_attention(
                q,
                q,
                global_cache,
                global_cache,
                None,
                out,
                global_lse,
                None,
                cutlass.Int32(1),
                global_lengths,
                None,
                direct_global,
                cutlass.Float32(512**-0.5),
                cutlass.Float32(1.0),
                stream,
                source_state_store_args=(part_c, global_counts),
            )
        final_out = cute.make_tensor(
            out.iterator,
            cute.make_layout(
                (swa_indices.shape[0], 64, 512), stride=(64 * 512, 512, 1)
            ),
        )
        self.merge(
            part_s,
            part_c,
            lse_s,
            lse_c,
            swa_counts,
            global_counts if self.compressed_k > 0 else swa_counts,
            sink,
            final_out,
            lse,
            stream,
        )
