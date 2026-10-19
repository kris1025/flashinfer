# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Shared live sparse-table sanitation, without cache reads or Prims-TS imports."""

import cutlass
import cutlass.cute as cute


class PrepareMetadata:
    def __init__(self, compressed_k, has_sink=False):
        if not 0 <= compressed_k <= 512 or compressed_k % 64:
            raise ValueError("compressed_k must be 0 or a multiple of64 up to512")
        self.compressed_k = compressed_k
        self.has_sink = has_sink

    @cute.kernel
    def _prepare(
        self,
        indices: cute.Tensor,
        topk_lens: cute.Tensor,
        seq_lens: cute.Tensor,
        sinks: cute.Tensor,
        selected_swa: cute.Tensor,
        selected_main: cute.Tensor,
        swa_lengths: cute.Tensor,
        main_lengths: cute.Tensor,
        total_lengths: cute.Tensor,
        swa_counts: cute.Tensor,
        main_counts: cute.Tensor,
        effective_sink: cute.Tensor,
        swa_capacity: cutlass.Int64,
        main_capacity: cutlass.Int64,
    ):
        request = cute.arch.block_idx()[0]
        lane = cute.arch.thread_idx()[0]
        if lane == 0:
            # Execution spans retain holes and the SWA HEAD even when all
            # entries are invalid. Counts describe actual selected positions.
            swa_lengths[request] = cutlass.Int32(128)
            main_lengths[request] = cutlass.Int32(self.compressed_k)
            total_lengths[request] = cutlass.Int32(128 + self.compressed_k)
            swa_counts[request] = cutlass.Int32(0)
            main_counts[request] = cutlass.Int32(0)
        if request == 0:
            if lane < 64:
                sink = cutlass.Float32(float("-inf"))
                if cutlass.const_expr(self.has_sink):
                    sink = cutlass.Float32(sinks[lane])
                effective_sink[lane] = sink
        cute.arch.sync_threads()

        swa_visible = cutlass.min(
            cutlass.max(cutlass.Int32(seq_lens[request]), cutlass.Int32(0)),
            cutlass.Int32(128),
        )
        # Clamp before subtracting 128 so malformed INT_MIN lengths cannot
        # wrap into a positive main prefix. Values never cause an OOB read.
        total_visible = cutlass.min(
            cutlass.max(cutlass.Int32(topk_lens[request]), cutlass.Int32(128)),
            cutlass.Int32(128 + self.compressed_k),
        )
        main_visible = total_visible - cutlass.Int32(128)
        index = cutlass.Int32(indices[request, lane])
        keep_s = (
            (lane < swa_visible)
            & (index >= cutlass.Int32(0))
            & (index != cutlass.Int32(0x7FFFFFFF))
            & (cutlass.Int64(index) < swa_capacity)
        )
        selected_swa[request, lane] = index if keep_s else cutlass.Int32(-1)
        count_c = cutlass.Int32(0)
        if cutlass.const_expr(self.compressed_k > 0):
            for group in cutlass.range_constexpr((self.compressed_k + 127) // 128):
                column = lane + group * 128
                if column < self.compressed_k:
                    main_index = cutlass.Int32(indices[request, 128 + column])
                    keep_c = (
                        (column < main_visible)
                        & (main_index >= cutlass.Int32(0))
                        & (main_index != cutlass.Int32(0x7FFFFFFF))
                        & (cutlass.Int64(main_index) < main_capacity)
                    )
                    selected_main[request, column] = (
                        main_index if keep_c else cutlass.Int32(-1)
                    )
                    count_c += cutlass.Int32(keep_c)
        elif lane < 64:
            selected_main[request, lane] = cutlass.Int32(-1)
        count_s = cute.arch.warp_reduction_sum(cutlass.Int32(keep_s))
        count_c = cute.arch.warp_reduction_sum(count_c)
        if lane % 32 == 0:
            cute.arch.atomic_add(swa_counts.iterator + request, count_s, scope="cta")
            cute.arch.atomic_add(main_counts.iterator + request, count_c, scope="cta")

    @cute.jit
    def __call__(
        self,
        indices: cute.Tensor,
        topk_lens: cute.Tensor,
        seq_lens: cute.Tensor,
        sinks: cute.Tensor,
        selected_swa: cute.Tensor,
        selected_main: cute.Tensor,
        swa_lengths: cute.Tensor,
        main_lengths: cute.Tensor,
        total_lengths: cute.Tensor,
        swa_counts: cute.Tensor,
        main_counts: cute.Tensor,
        effective_sink: cute.Tensor,
        swa_capacity: cutlass.Int64,
        main_capacity: cutlass.Int64,
        stream: object,
    ):
        self._prepare(
            indices,
            topk_lens,
            seq_lens,
            sinks,
            selected_swa,
            selected_main,
            swa_lengths,
            main_lengths,
            total_lengths,
            swa_counts,
            main_counts,
            effective_sink,
            swa_capacity,
            main_capacity,
        ).launch(grid=(indices.shape[0], 1, 1), block=(128, 1, 1), stream=stream)
