# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental DS4.1 mixed-cache and window-only decode on SM100.

The existing cute-dsl backend remains the default. The prims-ts backend
requires CuTe-DSL >= 4.8 with its bundled CUDA 13 compiler and uses explicit
E4M3 Q/ordinary FP8 SWA here. The source implementation historically passed the
expanded 257-case B200 matrix and selected 57-case dense subset with DSL 4.8.0 /
bundled CUDA 13.4. Those runs predate the native-only source split; they are not
fresh qualification of this revision, other GPUs/toolchains or full models.
Performance is unmeasured. See README.md for cache contracts and validation scope.
"""

import argparse

import torch
from flashinfer import deepseek_v41 as ds


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("cute-dsl", "prims-ts"),
        default="cute-dsl",
        help="explicit backend; prims-ts needs CuTe-DSL >= 4.8 with bundled CUDA 13",
    )
    return parser


def main():
    opts = build_parser().parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("DS4.1 decode requires SM100")

    q_source = torch.randn(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    # Explicit Q quantization belongs to the caller, not decode. Ordinary FP8
    # SWA has unit scale/no side plane; Frost keeps its legacy MXFP8 contract.
    native_fp8 = opts.backend == "prims-ts"
    q = q_source.to(torch.float8_e4m3fn) if native_fp8 else q_source
    swa = ds.deepseek_v41_quantize_cache(
        x, format="swa_fp8" if native_fp8 else "swa_mxfp8"
    )
    main_cache = ds.deepseek_v41_quantize_cache(x, format="main_kv_fp4")
    ids = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, 128)
    sink = torch.zeros(64, device="cuda")
    out, lse, plan = ds.deepseek_v41_decode(
        q, swa, main_cache, ids, ids, sink, backend=opts.backend
    )
    q_source.normal_()
    q.copy_(q_source.to(q.dtype))
    out, lse, _ = ds.deepseek_v41_decode(
        q, swa, main_cache, ids, ids, sink, plan=plan, backend=opts.backend
    )
    print(f"{opts.backend} mixed-cache decode:", out.shape, lse.shape)
    # Prepare a separate window-only plan; plans cannot cross declarations
    # or backends. The explicit selector must also be supplied on plan replay.
    window_out, window_lse, _ = ds.deepseek_v41_window_decode(
        q, swa, ids, sink, backend=opts.backend
    )
    print(f"{opts.backend} window-only decode:", window_out.shape, window_lse.shape)


if __name__ == "__main__":
    main()
