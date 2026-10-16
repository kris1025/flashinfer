# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Quantized-cache FP64 gates and paired CUDA Graph timing of DS4.1 backends.

python benchmarks/bench_deepseek_v41_decode.py --batches 1,4,16,32,64,128,256 \
    --backends cute-dsl prims-ts --context 32768 --output results.json

The default remains cute-dsl with BF16 Q/MXFP8 SWA. Selecting prims-ts defaults
to explicit E4M3 Q/ordinary FP8 SWA and E4M3 Q/K/V/P attention for both sources.
Selected packed FP4 main tiles apply group16 E4M3 scales, then round to unit
E4M3 with RTNE and finite saturation to [-448,448] before QK/PV. This is lossy:
finite scaled values can reach +/-2688. No BF16 query expansion is used.
--prims-ts-mode mxfp8 explicitly selects BF16 Q/packed MXFP8 SWA and legacy gates.
Native and legacy arms have different represented inputs and arithmetic, not
identical precision. All derive from common seeded BF16 sources and share one
packed global FP4 pool/indices, not a full converted cache.
FlashMLA is an optional external legacy BF16-Q/MXFP8 benchmark dependency,
enabled only by --flashmla, not a shipped decode backend. Cache-writer
quantization, allocation, JIT and host overhead are excluded from GPU timing.
Each timed invocation includes the full public replay, including GPU metadata,
selected-tile conversion, attention and merge.
Prims-TS requires CuTe-DSL >= 4.8 with its bundled CUDA 13 compiler, split 1 and
a nonpersistent schedule. The source implementation historically passed the
expanded 257-case B200 matrix and selected 57-case dense subset with DSL 4.8.0 /
bundled CUDA 13.4. Those pre-FP8-main runs used BF16 main attention and predate the
native-only source split; neither they nor historical 299-case CPU acceptance
qualify this change. No GPU/JIT or benchmark validation has run for FP8 main
attention; performance and backend crossovers are unmeasured.
See examples/deepseek_v41/README.md for scope and reproducer commands.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import traceback

import torch

from flashinfer.deepseek_v41 import deepseek_v41_decode, deepseek_v41_quantize_cache


PRIMS_TS_SOURCES = (
    "experimental/deepseek_v41/prims_ts.py",
    "experimental/deepseek_v41/prims_ts_fp8.py",
    "experimental/deepseek_v41/prims_ts_resources.py",
    "attention/prims_ts/kernels/mla_decode/helpers/ops.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/kernel.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/tasks.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/config.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/parallel_reduction.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/smem_resources.py",
    "attention/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_corr.py",
    "attention/prims_ts/kernels/mla_decode/sparse_views.py",
    "attention/prims_ts/kernels/separate_reduction.py",
)
PRIMS_TS_MXFP8_SOURCES = (
    "experimental/deepseek_v41/prims_ts_mxfp8.py",
    "experimental/deepseek_v41/prims_ts_mxfp8_resources.py",
)


def prims_ts_sources(prims_ts_mode="fp8"):
    return PRIMS_TS_SOURCES + (
        PRIMS_TS_MXFP8_SOURCES if prims_ts_mode == "mxfp8" else ()
    )


def selected(pool, indices, fp4, *, round_bf16=True):
    pages = pool.view(pool.shape[0], -1)
    ids = indices[:, 0].long()
    page, row = ids // 64, ids % 64
    column = torch.arange(512, device=ids.device)
    if fp4:
        data = pages[page[..., None], row[..., None] * 256 + column // 2]
        code = (data >> (4 * (column % 2))) & 15
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=ids.device,
            dtype=torch.float64,
        )
        data = lut[code.long()]
        scale = pages[page[..., None], 64 * 256 + row[..., None] * 32 + column // 16]
        scale = scale.contiguous().view(torch.float8_e4m3fn).double()
    else:
        data = pages[page[..., None], row[..., None] * 512 + column]
        data = data.contiguous().view(torch.float8_e4m3fn).double()
        scale = pages[page[..., None], 64 * 512 + row[..., None] * 16 + column // 32]
        scale = torch.exp2(scale.double() - 127)
    represented = data * scale
    return represented.bfloat16().double() if round_bf16 else represented


def fixture(batch, context, seed, backends, flashmla=False, *, prims_ts_mode="fp8"):
    torch.manual_seed(seed + batch)
    q_source = torch.randn(batch, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    w_source = torch.randn(batch * 128, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.empty(
        batch * context // 64, 64, 1, 288, device="cuda", dtype=torch.uint8
    )
    ci = torch.empty(batch, 1, 512, device="cuda", dtype=torch.int32)
    for b in range(batch):
        values = torch.randn(context, 512, device="cuda", dtype=torch.bfloat16)
        cache = deepseek_v41_quantize_cache(values, format="main_kv_fp4")
        main[b * (context // 64) : (b + 1) * (context // 64)].copy_(cache)
        ci[b, 0] = (
            torch.randperm(context, device="cuda")[:512].sort().values + b * context
        )
    wi = (
        (
            torch.arange(batch, device="cuda")[:, None, None] * 128
            + torch.arange(128, device="cuda")[None, None]
        )
        .int()
        .contiguous()
    )
    sink = torch.randn(64, device="cuda") * 0.1
    # Q/SWA differ at rest; main arithmetic differs despite shared packed storage.
    # Do not allocate a second production-sized global pool or converted cache.
    shared = main, wi, ci, sink
    arms = {}
    prims_ts_mxfp8 = "prims-ts" in backends and prims_ts_mode == "mxfp8"
    if "cute-dsl" in backends or flashmla or prims_ts_mxfp8:
        legacy = (
            q_source,
            deepseek_v41_quantize_cache(w_source, format="swa_mxfp8"),
            *shared,
        )
        if "cute-dsl" in backends:
            arms["cute-dsl"] = legacy
        if flashmla:
            arms["flashmla"] = legacy
        if prims_ts_mxfp8:
            arms["prims-ts"] = legacy
    if "prims-ts" in backends and not prims_ts_mxfp8:
        arms["prims-ts"] = (
            q_source.to(torch.float8_e4m3fn),
            deepseek_v41_quantize_cache(w_source, format="swa_fp8"),
            *shared,
        )
    return arms


def legacy_reference(q, swa, main, wi, ci, sink):
    kv = torch.cat((selected(swa, wi, False), selected(main, ci, True)), dim=1)
    scores = torch.einsum("bhd,bkd->bhk", q[:, 0].double(), kv) * 512**-0.5
    aug = torch.cat(
        (scores, sink.double()[None, :, None].expand(q.shape[0], -1, 1)), dim=-1
    )
    expected = torch.einsum("bhk,bkd->bhd", aug.softmax(-1)[..., :-1], kv)
    return expected, scores.logsumexp(-1)


def fp8_oracle(q, swa, global_cache, wi, ci, sink):
    """FP64 represented-input oracle and native componentwise FP8 bound.

    Decode only selected FP4 rows, then finite-clamp/RTNE to unit E4M3 for both
    QK and V, not the entire production-sized global pool. Scaled FP4 values can
    reach +/-2688, so the [-448,448] conversion is lossy. The attention/bound
    calculation matches the independent native GPU tests; neither production
    conversion helpers nor intermediate source states enter this oracle.
    """
    values, masks = [], []
    for pool, ids, fp4 in ((swa, wi, False), (global_cache, ci, True)):
        if pool is None:
            continue
        index = ids[:, 0].long()
        valid = (index >= 0) & (index < pool.shape[0] * 64) & (index != 0x7FFFFFFF)
        safe = index.clamp(0, pool.shape[0] * 64 - 1)
        if not fp4:
            gathered = pool.double().reshape(-1, 512)[safe]
        else:
            gathered = selected(pool, safe[:, None], True, round_bf16=False)
            gathered = gathered.clamp(-448, 448).to(torch.float8_e4m3fn).double()
        values.append(torch.where(valid[..., None], gathered, 0))
        masks.append(valid)
    kv, valid = torch.cat(values, 1), torch.cat(masks, 1)
    logits = torch.einsum("bhd,bkd->bhk", q[:, 0].double(), kv) * 512**-0.5
    logits.masked_fill_(~valid[:, None], -math.inf)
    kv_lse = logits.logsumexp(-1)
    normalizer = torch.logaddexp(kv_lse, sink.double()[None, :])
    weights = torch.exp(logits - normalizer[..., None])
    weights = torch.where(torch.isfinite(normalizer)[..., None], weights, 0)
    expected = torch.einsum("bhk,bkd->bhd", weights, kv)
    sensitivity = torch.einsum("bhk,bkd->bhd", weights, kv.abs())
    max_logit = logits.max(-1).values
    frame_mass = torch.exp(max_logit - normalizer)
    frame_mass = torch.where(torch.isfinite(kv_lse), frame_mass, 0)
    underflow = frame_mass[..., None] * kv.abs().sum(1)[:, None, :]
    # The joint maximum bounds each FP8 source's local maximum, so the summed
    # underflow term remains conservative with two independent FP8 source passes.
    bound = (2**-4 + 3 * 2**-8 + 1e-5) * sensitivity + (2**-10 / 448) * underflow + 1e-6
    bound = torch.where(torch.isfinite(kv_lse)[..., None], bound, 0)
    return expected, kv_lse, bound


def check_fp8(out, lse, reference):
    expected, expected_lse, bound = reference
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert ((out[:, 0].double() - expected).abs() <= bound).all()
    empty = torch.isneginf(expected_lse)
    assert torch.equal(torch.isneginf(lse[..., 0]), empty)
    torch.testing.assert_close(
        lse[..., 0].double()[~empty], expected_lse[~empty], rtol=0, atol=1e-5
    )
    assert torch.equal(out[:, 0][empty], torch.zeros_like(out[:, 0][empty]))
    assert torch.equal(bound[empty], torch.zeros_like(bound[empty]))


def check_legacy(out, stats):
    assert (
        torch.isfinite(out).all()
        and stats["rel_l2"] < 0.005
        and stats["max_scaled"] < 0.01
        and stats["lse_abs"] < 1e-5
    )


def check_arm(name, args, fn, batch, output, *, prims_ts_mode="fp8"):
    native_fp8 = name == "prims-ts" and prims_ts_mode == "fp8"
    reference = fp8_oracle(*args) if native_fp8 else legacy_reference(*args)
    expected, expected_lse = reference[:2]
    out, lse = fn()
    difference = out[:, 0].double() - expected
    if native_fp8:
        nonempty = ~torch.isneginf(expected_lse)
        stats = {
            # Ratios are diagnostic only; the native gate is componentwise.
            "rel_l2": float(difference.norm() / expected.norm().clamp_min(1e-30)),
            "max_scaled": float(
                difference.abs().max() / expected.abs().max().clamp_min(1e-30)
            ),
            "lse_abs": (
                float(
                    (lse[..., 0].double()[nonempty] - expected_lse[nonempty])
                    .abs()
                    .max()
                )
                if nonempty.any()
                else 0.0
            ),
            "max_bound_fraction": float(
                (difference.abs() / reference[2].clamp_min(1e-30)).max()
            ),
        }
    else:
        # Preserve the original legacy metrics as well as their strict limits.
        stats = {
            "rel_l2": float(difference.norm() / expected.norm()),
            "max_scaled": float(difference.abs().max() / expected.abs().max()),
            "lse_abs": float((lse[..., 0].double() - expected_lse).abs().max()),
        }
    try:
        if native_fp8:
            check_fp8(out, lse, reference)
        else:
            check_legacy(out, stats)
    except AssertionError as error:
        torch.save(
            {
                "out": out.cpu(),
                "lse": lse.cpu(),
                "expected": expected.cpu(),
                "expected_lse": expected_lse.cpu(),
                "bound": reference[2].cpu() if native_fp8 else None,
            },
            output.with_suffix(".failure.pt"),
        )
        raise AssertionError((batch, name, stats)) from error
    return stats


def arm_metadata(name, args, *, prims_ts_mode="fp8"):
    q, swa, main, _, _, _ = args
    native_fp8 = name == "prims-ts" and prims_ts_mode == "fp8"
    contract = {"reference": "independent FP64 over this arm's represented Q/KV"}
    if native_fp8:
        arithmetic = (
            "native E4M3 Q/K/V/P for SWA and main; "
            "BK128/Keep-AB/M64/retained8/split1/nonpersistent; "
            "original E4M3 Q, no BF16 query scratch"
        )
        global_conversion = (
            "selected scaled E2M1 -> unit E4M3 SMEM; RTNE + finite saturation "
            "[-448,448], preserving signed zero/subnormals; lossy (finite products "
            "reach +/-2688)"
        )
        contract.update(
            reference=(
                "independent FP64 over stored E4M3 Q/SWA and selected main KV "
                "after scaled E2M1 -> finite-clamped/RTNE unit E4M3"
            ),
            output="componentwise native FP8 P-rounding bound (not legacy relative gates)",
            componentwise_bound=(
                "(2^-4 + 3*2^-8 + 1e-5)*sensitivity + "
                "(2^-10/448)*underflow + 1e-6; exactly zero for empty rows"
            ),
            lse="sink-exclusive; rtol=0, atol=1e-5",
            empty="exact zero output, negative-infinity LSE and zero bound",
        )
    else:
        global_conversion = "scaled E2M1 -> BF16 (unchanged legacy represented values)"
        if name == "prims-ts":
            arithmetic = (
                "Prims-TS BF16 Q/dequantized MXFP8 SWA and FP4 global "
                "(explicit mxfp8 override)"
            )
        elif name == "cute-dsl":
            arithmetic = "Frost BF16 Q/dequantized MXFP8 SWA and FP4 global"
        else:
            arithmetic = (
                "external FlashMLA legacy BF16-Q/MXFP8-SWA arm, not Prims-TS native FP8"
            )
        contract.update(
            rel_l2_lt=0.005, max_scaled_lt=0.01, sink_exclusive_lse_abs_lt=1e-5
        )
    return {
        "prims_ts_mode": prims_ts_mode if name == "prims-ts" else None,
        "q_dtype": str(q.dtype),
        "swa_encoding": "swa_fp8" if native_fp8 else "swa_mxfp8",
        "global_encoding": "main_kv_fp4",
        "global_storage": "packed E2M1 + group16 E4M3 scales; width288 page-plane",
        "global_conversion": global_conversion,
        "cache_shape": {"swa": list(swa.shape), "global_fp4": list(main.shape)},
        "arithmetic": arithmetic,
        "accuracy_contract": contract,
    }


def graph(fn):
    for _ in range(4):
        fn()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        for _ in range(32):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    return g


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,4,16,32,64,128,256")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=7003)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("cute-dsl", "prims-ts"),
        default=["cute-dsl"],
        help="prims-ts defaults to E4M3 Q/ordinary FP8 SWA; cute-dsl retains BF16 Q/MXFP8 SWA",
    )
    parser.add_argument(
        "--prims-ts-mode",
        choices=("fp8", "mxfp8"),
        default="fp8",
        help="prims-ts only: mxfp8 explicitly selects BF16 Q/packed MXFP8 SWA",
    )
    parser.add_argument("--flashmla", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main():
    opts = build_parser().parse_args()
    if opts.output.exists():
        raise FileExistsError(opts.output)
    if opts.context < 512 or opts.context % 64:
        raise ValueError("context must be a multiple of64 >=512")
    import cutlass
    from flashinfer.experimental.deepseek_v41 import cache, decode, hca_v41_primitives

    source_paths = {
        m.__name__: Path(m.__file__) for m in (cache, decode, hca_v41_primitives)
    }
    if "cute-dsl" in opts.backends:
        from flashinfer.experimental.deepseek_v41 import hca_v41

        source_paths[hca_v41.__name__] = Path(hca_v41.__file__)
    if "prims-ts" in opts.backends:
        # Hash sources without importing either mode just for provenance.
        package_root = Path(decode.__file__).resolve().parents[2]
        for relative in prims_ts_sources(opts.prims_ts_mode):
            source_paths[f"flashinfer.{relative[:-3].replace('/', '.')}"] = (
                package_root / relative
            )
    report = {
        "status": "running",
        "implementation": "deepseek_v41",
        "backends": opts.backends,
        "prims_ts_mode": opts.prims_ts_mode,
        "device": str(torch.cuda.get_device_properties(0)),
        "torch": torch.__version__,
        "toolchain": {
            "cutlass_dsl": cutlass.__version__,
            "cutlass_dsl_package": importlib.metadata.version("nvidia-cutlass-dsl"),
            "cutlass_module": str(Path(cutlass.__file__).resolve()),
            "cute_bundled_cuda": (
                f"{cutlass.CUDA_VERSION.major}.{cutlass.CUDA_VERSION.minor}"
            ),
            "torch_cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
        },
        "context_per_request": opts.context,
        "shape": (
            "H64/Sq1/D512; SWA128 + selected512 packed FP4; encodings/shapes per arm"
        ),
        "comparison": (
            "common seeded BF16 source Q/SWA; one shared packed global FP4 pool "
            "and indices; native FP8 (when selected) converts main tiles to lossy "
            "finite-clamped/RTNE E4M3; legacy represented main values remain BF16; "
            "arm-specific arithmetic and accuracy contracts"
        ),
        "seed": opts.seed,
        "timing": "CUDA Graph/event, 32 calls/replay, symmetric order, 30 samples/arm",
        "scope": (
            "full public replay GPU latency, including metadata+decode+merge; "
            "excludes cache preparation/JIT/allocation/capture/host time"
        ),
        "sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_paths.items()
        },
        "rows": [],
    }
    try:
        with torch.no_grad():
            for batch in map(int, opts.batches.split(",")):
                arm_args = fixture(
                    batch,
                    opts.context,
                    opts.seed,
                    opts.backends,
                    opts.flashmla,
                    prims_ts_mode=opts.prims_ts_mode,
                )
                plans, fns = {}, {}
                for backend in opts.backends:
                    args = arm_args[backend]
                    plans[backend] = deepseek_v41_decode(*args, backend=backend)[2]
                    # Capture the complete public call, not just attention:
                    # Prims-TS also sanitizes live selections on the GPU.
                    fns[backend] = lambda args=args, plan=plans[
                        backend
                    ], backend=backend: (
                        deepseek_v41_decode(*args, plan=plan, backend=backend)[:2]
                    )
                del args
                if opts.flashmla:
                    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

                    # FlashMLASchedMeta is initialized from this fixture by
                    # the eager correctness call below, before graph capture.
                    # The second return value is a None compatibility placeholder.
                    meta, _ = get_mla_metadata()

                    def flashmla(args=arm_args["flashmla"], meta=meta):
                        q, w, c, wi, ci, sink = args
                        return flash_mla_with_kvcache(
                            q,
                            w,
                            None,
                            None,
                            512,
                            meta,
                            softmax_scale=512**-0.5,
                            is_fp8_kvcache=True,
                            indices=wi,
                            attn_sink=sink,
                            extra_k_cache=c,
                            extra_indices_in_kvcache=ci,
                        )

                    fns["flashmla"] = flashmla
                    del flashmla
                stats = {
                    name: check_arm(
                        name,
                        arm_args[name],
                        fn,
                        batch,
                        opts.output,
                        prims_ts_mode=opts.prims_ts_mode,
                    )
                    for name, fn in fns.items()
                }
                graphs = {name: graph(fn) for name, fn in fns.items()}
                samples = {name: [] for name in fns}
                for _ in range(15):
                    for name in list(fns) + list(reversed(fns)):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) * 1000 / 32)
                row = {
                    "batch": batch,
                    "split": {name: plan.split for name, plan in plans.items()},
                    "arms": {
                        name: arm_metadata(
                            name, arm_args[name], prims_ts_mode=opts.prims_ts_mode
                        )
                        for name in fns
                    },
                    "correctness": stats,
                    "median_us": {k: statistics.median(v) for k, v in samples.items()},
                    "samples_us": samples,
                }
                report["rows"].append(row)
                opts.output.write_text(json.dumps(report, indent=2))
                print({k: row[k] for k in ("batch", "split", "median_us")}, flush=True)
                del graphs, fns, plans, arm_args
                if opts.flashmla:
                    del meta
            report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["error"] = traceback.format_exc()
        raise
    finally:
        opts.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
