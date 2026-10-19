# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Shared-unit E4M3 SWA/main attention for the V4.1 adapter.

Original metadata is checked inside one finalizing native attention. Both sources
share online softmax, FP32 PV, sink normalization and final BF16 publication.
The generic FP32 source-store hook below remains available independently.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute


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
    def __init__(self, attention):
        self.attention = attention

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        swa_cache: cute.Tensor,
        main_cache: cute.Tensor,
        indices: cute.Tensor,
        topk_lens: cute.Tensor,
        seq_lens: cute.Tensor,
        sinks: object,
        out: cute.Tensor,
        lse: cute.Tensor,
        bmm1_scale: cutlass.Float32,
        bmm2_scale: cutlass.Float32,
        stream: object,
    ):
        final_lse = cute.make_tensor(
            lse.iterator,
            cute.make_layout((64, 1, indices.shape[0]), stride=(1, 64, 64)),
        )
        # The producer-owned binder consumes this original combined table and
        # its live lengths, rather than generic caller-prepared source lists.
        direct = (
            indices,
            None,
            seq_lens,
            topk_lens,
            None,
            None,
            None,
            None,
            sinks,
        )
        self.attention(
            q,
            q,
            main_cache,
            swa_cache,
            None,
            out,
            final_lse,
            None,
            cutlass.Int32(1),
            seq_lens,
            None,
            direct,
            bmm1_scale,
            bmm2_scale,
            stream,
        )
