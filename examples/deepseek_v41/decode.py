# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental DS4.1 mixed-cache and window-only decode on SM100.

The existing cute-dsl backend remains the default. The prims-ts backend
requires CuTe-DSL >= 4.8 with its bundled CUDA 13 compiler and uses explicit
E4M3 Q and typed E4M3 SWA. Both caches are written in common cache units, with
one shared pair of BMM factors and no decode-time KV amax. This example uses
the V4-style caller workspace, supplied output and default HND cache layout.
Historical GPU tests do not qualify this joint/shared-scale/API revision.
The revised example has not run; performance is unmeasured. See README.md
for cache contracts and validation scope.
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
    # The caller owns Q quantization and common KV units. Normalize both cache
    # writers upstream; decode restores the same descale through its BMM pair.
    native_fp8 = opts.backend == "prims-ts"
    q = q_source.to(torch.float8_e4m3fn) if native_fp8 else q_source
    kv_descale = 2.0
    cache_units = (x.float() / kv_descale).contiguous()
    swa = ds.deepseek_v41_quantize_cache(
        cache_units, format="swa_fp8" if native_fp8 else "swa_mxfp8"
    )
    main_cache = ds.deepseek_v41_quantize_cache(cache_units, format="main_kv_fp4")
    ids = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 128)
    indices = torch.cat((ids, ids), dim=1)
    topk_lens = torch.full((1,), 256, device="cuda", dtype=torch.int32)
    seq_lens = torch.full((1,), 128, device="cuda", dtype=torch.int32)
    sink = torch.zeros(64, device="cuda")
    workspace = torch.empty(
        ds.deepseek_v41_decode_workspace_size(1, 128, backend=opts.backend),
        device="cuda",
        dtype=torch.uint8,
    )
    out = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)

    def mixed_decode():
        return ds.deepseek_v41_decode(
            q,
            swa,
            workspace,
            indices,
            main_cache,
            topk_lens,
            seq_lens,
            out,
            # V4's public BMM1 default is 1.0; explicitly retain 1/sqrt(D).
            bmm1_scale=512**-0.5 * kv_descale,
            bmm2_scale=kv_descale,
            sinks=sink,
            backend=opts.backend,
        )

    assert mixed_decode() is out
    q_source.normal_()
    q.copy_(q_source.to(q.dtype))
    assert mixed_decode() is out
    print(f"{opts.backend} mixed-cache decode:", out.shape)

    window_workspace = torch.empty(
        ds.deepseek_v41_decode_workspace_size(1, backend=opts.backend),
        device="cuda",
        dtype=torch.uint8,
    )
    window_topk_lens = torch.full((1,), 128, device="cuda", dtype=torch.int32)
    window_out = ds.deepseek_v41_window_decode(
        q,
        swa,
        window_workspace,
        ids,
        window_topk_lens,
        seq_lens,
        bmm1_scale=512**-0.5 * kv_descale,
        bmm2_scale=kv_descale,
        sinks=sink,
        backend=opts.backend,
    )
    print(f"{opts.backend} window-only decode:", window_out.shape)


if __name__ == "__main__":
    main()
