# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Independent FP64 oracle and changed-input replay for both DS4.1 backends."""

import builtins
import dataclasses
import math

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_decode, deepseek_v41_window_decode


def gate(backend="cute-dsl"):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("DS4.1 decode validation targets SM100")
    pytest.importorskip("cutlass.experimental.primitives")
    if backend == "prims-ts":
        import cutlass
        from packaging.version import Version

        if Version(cutlass.__version__) < Version("4.8.0"):
            pytest.skip("Prims-TS requires CuTe-DSL >= 4.8.0")
        if cutlass.CUDA_VERSION.major < 13:
            pytest.skip("Prims-TS requires the bundled CUDA 13 compiler")


@pytest.fixture
def conversion_kernels():
    # CuTe's standalone executor has a bound-method cycle. Unload these local
    # probe modules at teardown, not during a later test's CUDA Graph capture.
    kernels = []
    try:
        yield kernels
    finally:
        for compiled in kernels:
            if compiled.jit_module is not None:
                compiled.jit_module.unload()


def pack(data, scales, width):
    return torch.cat(
        (
            data.reshape(-1, 64 * data.shape[1]),
            scales.reshape(-1, 64 * scales.shape[1]),
        ),
        dim=1,
    ).view(-1, 64, 1, width)


def make_case(batch, compressed_k, seed=42323):
    torch.manual_seed(seed + batch + compressed_k)
    q = torch.randn(batch, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    # Encoded finite bytes, with independent scale decoding in the oracle.
    wdata = (
        torch.randn(512, 512, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    )
    wscale = torch.randint(124, 130, (512, 16), device="cuda", dtype=torch.uint8)
    swa = pack(wdata, wscale, 528)
    cdata = torch.randint(0, 256, (1024, 256), device="cuda", dtype=torch.uint8)
    cscale = (
        (torch.rand(1024, 32, device="cuda") + 0.125)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    main = pack(cdata, cscale, 288) if compressed_k else None
    wi = torch.randint(0, 512, (batch, 1, 128), device="cuda", dtype=torch.int32)
    ci = (
        torch.randint(
            0, 1024, (batch, 1, compressed_k), device="cuda", dtype=torch.int32
        )
        if compressed_k
        else None
    )
    sink = torch.randn(64, device="cuda")
    return q, swa, main, wi, ci, sink


def unpack(pool, fp4):
    pages = pool.view(pool.shape[0], -1)
    if not fp4:
        data = (
            pages[:, : 64 * 512]
            .contiguous()
            .view(torch.float8_e4m3fn)
            .reshape(-1, 512)
            .double()
        )
        exponent = pages[:, 64 * 512 :].reshape(-1, 16).double() - 127
        return (
            (data * torch.exp2(exponent).repeat_interleave(32, -1)).bfloat16().double()
        )
    data = pages[:, : 64 * 256].reshape(-1, 256)
    codes = torch.stack((data & 15, data >> 4), dim=-1).reshape(-1, 512)
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float64,
    )
    scales = (
        pages[:, 64 * 256 :]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(-1, 32)
        .double()
    )
    return (table[codes.long()] * scales.repeat_interleave(16, -1)).bfloat16().double()


def oracle(args):
    q, swa, main, wi, ci, sink = args
    values, masks = [], []
    for pool, ids, fp4 in ((swa, wi, False), (main, ci, True)):
        if pool is None:
            continue
        decoded = unpack(pool, fp4)
        ids = ids[:, 0].long()
        valid = (ids >= 0) & (ids < decoded.shape[0])
        gathered = decoded[ids.clamp(0, decoded.shape[0] - 1)]
        values.append(torch.where(valid[..., None], gathered, 0))
        masks.append(valid)
    kv, valid = torch.cat(values, dim=1), torch.cat(masks, dim=1)
    scores = torch.einsum("bhd,bkd->bhk", q[:, 0].double(), kv) * (512**-0.5)
    scores.masked_fill_(~valid[:, None], -float("inf"))
    lse = scores.logsumexp(-1)
    augmented = torch.cat(
        (scores, sink.double()[None, :, None].expand(q.shape[0], -1, 1)), dim=-1
    )
    probability = augmented.softmax(-1)[..., :-1].nan_to_num()
    out = torch.einsum("bhk,bkd->bhd", probability, kv)
    return out, lse


def check(args, out, lse):
    expected, expected_lse = oracle(args)
    diff = out[:, 0].double() - expected
    assert torch.isfinite(out).all()
    assert diff.norm() / expected.norm().clamp_min(1e-30) < 0.005
    assert diff.abs().max() / expected.abs().max().clamp_min(1e-30) < 0.01
    empty = torch.isneginf(expected_lse)
    assert torch.equal(torch.isneginf(lse[..., 0]), empty)
    torch.testing.assert_close(
        lse[..., 0].double()[~empty], expected_lse[~empty], rtol=0, atol=1e-5
    )
    assert torch.equal(out[:, 0][empty], torch.zeros_like(out[:, 0][empty]))


@pytest.mark.parametrize("backend", ["auto", "unknown", None])
@pytest.mark.parametrize("window_only", [False, True])
def test_unknown_backend(backend, window_only):
    # Reject selectors before importing a kernel or checking CUDA inputs.
    with pytest.raises(ValueError, match="backend must"):
        if window_only:
            deepseek_v41_window_decode(None, None, None, None, backend=backend)
        else:
            deepseek_v41_decode(None, None, None, None, None, None, backend=backend)


@pytest.mark.parametrize("compressed_k", [0, 64])
def test_prims_ts_does_not_compile_cute_dsl(compressed_k, monkeypatch):
    gate("prims-ts")
    from flashinfer.experimental.deepseek_v41 import decode as cute_decode

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("prims-ts must not invoke the Frost compiler")

    monkeypatch.setattr(cute_decode, "_compile", forbidden_compile)
    args = make_case(2, compressed_k)
    out, lse, plan = deepseek_v41_decode(*args, backend="prims-ts")
    assert dataclasses.is_dataclass(plan)
    assert out is plan.out and lse is plan.lse
    assert isinstance(plan.signature, tuple)
    assert plan.split == 1  # Fixed Prims-TS bring-up split.
    check(args, out, lse)
    if compressed_k:
        replay = deepseek_v41_decode(*args, backend="prims-ts", plan=plan)
    else:
        replay = deepseek_v41_window_decode(
            args[0], args[1], args[3], args[5], backend="prims-ts", plan=plan
        )
    assert replay[0] is out and replay[1] is lse and replay[2] is plan


@pytest.mark.parametrize("compressed_k", [0, 64, 192, 512])
def test_prims_ts_selection_refresh(compressed_k):
    gate("prims-ts")
    args = make_case(2, compressed_k)
    for pool, indices in ((args[1], args[3]), (args[2], args[4])):
        if indices is None:
            continue
        capacity = pool.shape[0] * 64
        indices[:, 0, :8] = torch.tensor(
            [0, 0, capacity - 1, capacity, -1, -(2**31), 2**31 - 1, capacity + 1],
            device="cuda",
            dtype=torch.int32,
        )
    out, lse, plan = deepseek_v41_decode(*args, backend="prims-ts")

    def check_metadata():
        for pool, indices, selected in (
            (args[1], args[3], plan.selected_swa),
            (args[2], args[4], plan.selected_global),
        ):
            if indices is None:
                assert torch.equal(selected, torch.full_like(selected, -1))
            else:
                indices = indices[:, 0]
                valid = (indices >= 0) & (indices < pool.shape[0] * 64)
                assert torch.equal(selected, torch.where(valid, indices, -1))

    check_metadata()
    check(args, out, lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_decode(*args, backend="prims-ts", plan=plan)
    # Poison prior metadata and change which positions are holes, without
    # changing graph-bound addresses. The global tile can start with a hole.
    plan.selected_swa.fill_(2**31 - 1)
    plan.selected_global.fill_(2**31 - 1)
    for indices in (args[3], args[4]):
        if indices is not None:
            indices[:, 0, :8].copy_(indices[:, 0, :8].flip(-1))
    graph.replay()
    check_metadata()
    check(args, out, lse)


@pytest.mark.parametrize("compressed_k", [0, 64, 192, 512])
def test_prims_ts_zero_query_tile_counts(compressed_k):
    gate("prims-ts")
    args = make_case(4, compressed_k)
    args[0].zero_()
    for indices in (args[3], args[4]):
        if indices is not None:
            indices[:, 0, ::64].fill_(-1)
            indices[3].fill_(-1)
    args[3][1].fill_(-1)
    if args[4] is not None:
        args[4][2].fill_(-1)
    # With Q=0 every valid selection contributes one to the sink-exclusive
    # denominator. Cover mixed, global-only, window-only and empty requests;
    # holes at BK64 boundaries must not truncate the remaining tiles.
    counts = torch.zeros(4, device="cuda", dtype=torch.float64)
    for pool, indices in ((args[1], args[3]), (args[2], args[4])):
        if indices is not None:
            valid = (indices[:, 0] >= 0) & (indices[:, 0] < pool.shape[0] * 64)
            counts += valid.sum(-1)
    out, lse, _ = deepseek_v41_decode(*args, backend="prims-ts")
    torch.testing.assert_close(
        lse[..., 0].double(), counts.log()[:, None].expand(-1, 64), rtol=0, atol=1e-5
    )
    check(args, out, lse)


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
def test_prepare_plan_before_capture(backend, monkeypatch):
    gate(backend)
    args = make_case(1, 64)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="prepare a decode plan before CUDA Graph"):
        deepseek_v41_decode(*args, backend=backend)


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
@pytest.mark.parametrize("window_only", [False, True])
@pytest.mark.parametrize("prims_ts_mode", ["fp8", "mxfp8"])
def test_cross_backend_plan_rejected(backend, window_only, prims_ts_mode):
    gate("prims-ts")
    compressed = 0 if window_only else 64
    prims_case = make_fp8_case if prims_ts_mode == "fp8" else make_case
    cases = {
        "cute-dsl": make_case(1, compressed),
        "prims-ts": prims_case(1, compressed),
    }

    def run(selected_backend, plan=None):
        args = cases[selected_backend]
        if window_only:
            return deepseek_v41_window_decode(
                args[0], args[1], args[3], args[5], backend=selected_backend, plan=plan
            )
        return deepseek_v41_decode(*args, backend=selected_backend, plan=plan)

    other_backend = "prims-ts" if backend == "cute-dsl" else "cute-dsl"
    _, _, plan = run(other_backend)
    with pytest.raises(ValueError, match="mismatch"):
        run(backend, plan=plan)


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
@pytest.mark.parametrize(
    "batch,compressed_k",
    [
        (1, 512),
        (16, 512),
        (64, 512),
        (128, 512),
        (129, 512),
        pytest.param(None, 512, id="sm-count-plus-one"),
        (256, 512),
        (512, 512),
        (4, 0),
        (128, 0),
        (2, 64),
        (8, 192),
    ],
)
def test_decode_changed_graph_and_fp64(batch, compressed_k, backend, monkeypatch):
    gate(backend)
    if batch is None:
        # Span more than one SM wave; Frost also covers uneven persistent work.
        batch = torch.cuda.get_device_properties("cuda").multi_processor_count + 1
    original_import = builtins.__import__

    def no_old_decode(name, *a, **kw):
        if name.startswith("flash_mla") or name.endswith("decode_fp32"):
            raise AssertionError(f"unexpected alternative decode route: {name}")
        return original_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_old_decode)
    args = make_case(batch, compressed_k)
    out, lse, plan = deepseek_v41_decode(*args, backend=backend)

    def run():
        if compressed_k:
            return deepseek_v41_decode(*args, plan=plan, backend=backend)
        return deepseek_v41_window_decode(
            args[0], args[1], args[3], args[5], plan=plan, backend=backend
        )

    check(args, out, lse)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    for step in range(3):
        args[0].normal_()
        args[3].random_(0, 512)
        args[3][..., step::4] = -1
        if compressed_k:
            args[4].random_(0, 1024)
            args[4][..., step::3] = -1
        args[5].normal_()
        args[5][0], args[5][1], args[5][2] = 1000, -1000, -float("inf")
        if step == 1:
            # New cache bytes at the same addresses must affect graph replay.
            replacement = make_case(batch, compressed_k, seed=8)
            args[1].copy_(replacement[1])
            if compressed_k:
                args[2].copy_(replacement[2])
        if step == 2:
            args[3][0] = -1
            if compressed_k:
                args[4][0] = -1
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        check(args, out, lse)
    assert run()[0].data_ptr() == out.data_ptr()
    with pytest.raises(AssertionError):
        check(args, torch.ones_like(out), lse)
    with pytest.raises(AssertionError):
        check(args, out, torch.zeros_like(lse))
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_decode(
            *args, plan=dataclasses.replace(plan, out=args[0]), backend=backend
        )
    with pytest.raises(ValueError, match="mismatch"):
        deepseek_v41_decode(*args, plan=object(), backend=backend)
    with pytest.raises(ValueError, match="inference"):
        deepseek_v41_decode(
            args[0].requires_grad_(), *args[1:], plan=plan, backend=backend
        )


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
def test_empty_uninitialized_cache_and_declaration_guards(backend):
    gate(backend)
    args = make_case(2, 512)
    args[1].fill_(255)
    args[2].fill_(255)
    args[3].fill_(-1)
    args[4].fill_(2**31 - 1)
    args[5].fill_(-float("inf"))
    out, lse, _ = deepseek_v41_decode(*args, backend=backend)
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.isneginf(lse).all()
    with pytest.raises(ValueError, match="Q must"):
        deepseek_v41_decode(
            args[0].expand(-1, 2, -1, -1).contiguous(), *args[1:], backend=backend
        )
    with pytest.raises(ValueError, match="page size"):
        deepseek_v41_decode(
            args[0], args[1].view(-1, 32, 1, 528), *args[2:], backend=backend
        )
    with pytest.raises(ValueError, match="both be present"):
        deepseek_v41_decode(
            args[0], args[1], None, args[3], args[4], args[5], backend=backend
        )
    with pytest.raises(ValueError, match="SWA indices"):
        deepseek_v41_decode(*args[:3], args[3].long(), *args[4:], backend=backend)
    with pytest.raises(ValueError, match="global indices"):
        deepseek_v41_decode(
            *args[:4], args[4][..., :63].contiguous(), args[5], backend=backend
        )
    with pytest.raises(ValueError, match="sink must"):
        deepseek_v41_decode(*args[:5], args[5].double(), backend=backend)
    misaligned = torch.empty(
        args[0].numel() + 1, device=args[0].device, dtype=args[0].dtype
    )[1:].view_as(args[0])
    with pytest.raises(ValueError, match="16-byte aligned"):
        deepseek_v41_decode(misaligned, *args[1:], backend=backend)
    with pytest.raises(TypeError):
        deepseek_v41_decode(*args, arithmetic="bf16x3", backend=backend)


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
@pytest.mark.parametrize("batch", [128, 256])
@pytest.mark.parametrize("pool_index,width,ids_index", [(1, 528, 3), (2, 288, 4)])
def test_cache_offsets_above_two_gib(batch, backend, pool_index, width, ids_index):
    gate(backend)
    args = make_case(batch, 512)
    small = args[pool_index]
    # Leave the prefix uninitialized: only the copied tail is a valid slot.
    # The independent oracle still reads the original small encoded pool.
    page_bytes = 64 * width
    first_page = (2**31 + page_bytes - 1) // page_bytes
    large = torch.empty(
        (first_page + small.shape[0], 64, 1, width),
        device="cuda",
        dtype=torch.uint8,
    )
    large[first_page:].copy_(small)
    actual = list(args)
    actual[pool_index] = large
    actual[ids_index] = args[ids_index] + first_page * 64
    out, lse, _ = deepseek_v41_decode(*actual, backend=backend)
    check(args, out, lse)


@pytest.mark.parametrize("pool_index,width,ids_index", [(1, 528, 3), (2, 288, 4)])
@pytest.mark.parametrize("below_limit", [True, False])
def test_persistent_metadata_capacity_boundary(
    pool_index, width, ids_index, below_limit
):
    gate()
    batch = torch.cuda.get_device_properties("cuda").multi_processor_count + 1
    args = list(make_case(batch, 512))
    small = args[pool_index]
    page_bytes = 64 * width
    limit = 2**35  # Signed Int32 offsets expressed in 16-byte units.
    pages = (
        (limit - 1) // page_bytes
        if below_limit
        else (limit + page_bytes - 1) // page_bytes + small.shape[0]
    )
    if torch.cuda.mem_get_info()[0] < pages * page_bytes + 2 * 1024**3:
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < pages * page_bytes + 2 * 1024**3:
            pytest.skip("Capacity-boundary validation needs a 32-GiB cache pool")
    # Only the tail is initialized; the oracle continues to use the small pool.
    large = torch.empty((pages, 64, 1, width), device="cuda", dtype=torch.uint8)
    first_page = pages - small.shape[0]
    large[first_page:].copy_(small)
    actual = list(args)
    actual[pool_index] = large
    actual[ids_index] = args[ids_index] + first_page * 64
    out, lse, plan = deepseek_v41_decode(*actual, backend="cute-dsl")
    check(args, out, lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_decode(*actual, plan=plan, backend="cute-dsl")
    args[0].mul_(0.75)
    args[ids_index].copy_(args[ids_index].flip(-1))
    actual[ids_index].copy_(actual[ids_index].flip(-1))
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    check(args, out, lse)


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
@pytest.mark.parametrize(
    "levels",
    [
        [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.75],
        [0.0625, 0.5, 0.125, 0.75, 0.125, 0.375, 0.0625, 1.0],
        [0.0625] * 7 + [0.75],
    ],
)
@pytest.mark.parametrize("batch", [16, 128, 256])
def test_normalization_anchor_crossings(levels, batch, backend):
    gate(backend)
    q, swa, main, wi, ci, sink = make_case(batch, 512)
    # Exact scalar products with head-dependent signs exercise differing
    # rescale decisions in the same warp. Last tile introduces a new maximum.
    head_scale = torch.linspace(-0.75, 1.25, 64, device=q.device).bfloat16()
    q.copy_(head_scale[None, None, :, None].expand_as(q))
    if batch > 128:
        # Reverse normalization-anchor patterns across requests. Frost's
        # persistent CTAs must reset row statistics and O between requests.
        q[batch // 2 :].mul_(-0.5)
    wd = (
        torch.zeros((128, 512), device=q.device)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    ws = torch.full((128, 16), 127, device=q.device, dtype=torch.uint8)
    swa = pack(wd, ws, 528)
    cd = torch.full((512, 256), 0x22, device=q.device, dtype=torch.uint8)
    cs = (
        torch.tensor(levels, device=q.device)
        .repeat_interleave(64)[:, None]
        .expand(512, 32)
        .contiguous()
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    main = pack(cd, cs, 288)
    wi = (
        torch.arange(128, device=q.device, dtype=torch.int32)[None, None]
        .expand(batch, 1, 128)
        .contiguous()
    )
    ci = (
        torch.arange(512, device=q.device, dtype=torch.int32)[None, None]
        .expand(batch, 1, 512)
        .contiguous()
    )
    sink.fill_(-float("inf"))
    args = (q, swa, main, wi, ci, sink)
    out, lse, _ = deepseek_v41_decode(*args, backend=backend)
    check(args, out, lse)


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
def test_fp4_conversion_fallback(backend):
    gate(backend)
    args = make_case(128, 512)
    _, _, plan = deepseek_v41_decode(*args, backend=backend)
    if backend == "cute-dsl":
        from flashinfer.experimental.deepseek_v41.decode import _compile

        compiled = _compile(args[0].device.index, 512, False, 16, 2, False, False)
    else:
        from flashinfer.experimental.deepseek_v41.prims_ts_mxfp8 import _compile

        compiled = _compile(args[0].device.index, 512, native_fp4=False)
    fallback = dataclasses.replace(plan, kernel=compiled)
    out, lse, _ = deepseek_v41_decode(*args, plan=fallback, backend=backend)
    check(args, out, lse)


def test_native_fp4_all_finite_codes(conversion_kernels):
    gate()
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.experimental import primitives as prims
    from flashinfer.experimental.deepseek_v41 import hca_v41_primitives as hw

    if (cutlass.CUDA_VERSION.major, cutlass.CUDA_VERSION.minor) < (13, 2):
        pytest.skip("native BF16 conversion requires bundled CUDA >= 13.2")

    @cute.kernel
    def convert(words: cute.Tensor, scales: cute.Tensor, out: cute.Tensor):
        i = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if i < words.shape[0]:
            scale = hw.fp8x2_to_bf16_word(scales[i] * 257)
            values = hw.fp4x8_to_bf16_words(words[i])
            for j in cutlass.range_constexpr(4):
                out[i, j] = prims.mul_bf16x2(values[j], scale)

    @cute.jit
    def launch(words, scales, out, stream: cuda.CUstream):
        convert(words, scales, out).launch(
            grid=(cute.ceil_div(words.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    # Cross every pair of FP4 codes with every finite E4M3 scale. Permute
    # other bytes to expose lane-order bugs; compare signed zeros bitwise.
    ids = torch.arange(65536, device="cuda", dtype=torch.int64)
    sc, byte = ids & 255, ids >> 8
    keep = (sc != 127) & (sc != 255)
    words = (
        byte | ((byte ^ 0x53) << 8) | ((byte ^ 0x97) << 16) | ((byte ^ 0xE1) << 24)
    )[keep].int()
    scales = sc[keep].int()
    out = torch.empty((words.numel(), 4), device="cuda", dtype=torch.int32)
    tensors = [from_dlpack(x) for x in (words, scales, out)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(launch, *tensors, stream)
    conversion_kernels.append(compiled)
    compiled(*tensors, stream)
    lut = torch.tensor(
        [0.0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float64,
    )
    codes = (words.long()[:, None] >> (4 * torch.arange(8, device="cuda"))) & 15
    scale_values = scales.to(torch.uint8).view(torch.float8_e4m3fn).double()
    expected = (lut[codes] * scale_values[:, None]).bfloat16()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("native_fp4", [False, True])
def test_fp4_fp8_all_finite_codes(native_fp4, conversion_kernels):
    gate("prims-ts")
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_sources
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
        Fp4Fp8SmemKvResource,
    )

    if native_fp4 and (cutlass.CUDA_VERSION.major, cutlass.CUDA_VERSION.minor) < (
        13,
        2,
    ):
        pytest.skip("native BF16-register conversion requires bundled CUDA >= 13.2")

    _, main = _make_fp8_sources(512, 1, native_fp4)
    cfg = main._make_config()
    assert cfg.qkv_dtype == "e4m3" and cfg.tile_size_kv == 128
    assert cfg.head_dim_per_stage_kv == 128 and cfg.kv_stages == 8
    # The standalone CTA contains just the production four-warp load group.
    # Change its warp origin, not the tile shape, dtype or stage/store layout.
    resource = main.kv_producer.make_resource(
        (None,), name="fp4_fp8_probe", cfg=dataclasses.replace(cfg, load_warp_idx=0)
    )
    assert type(resource) is Fp4Fp8SmemKvResource
    stage_bytes = 128 * 128

    @cute.kernel
    def convert(
        pool: cute.Tensor, selected: cute.Tensor, out: cute.Tensor, guards: cute.Tensor
    ):
        tid = cute.arch.thread_idx()[0]
        tile = cute.arch.block_idx()[0]
        lane, warp = tid % 32, tid // 32
        storage = utils.SmemAllocator().allocate(8 * stage_bytes + 256, 128)
        smem = cute.make_tensor(
            cute.recast_ptr(storage, dtype=cutlass.Uint8),
            cute.make_layout((8 * stage_bytes + 256,)),
        )
        # Production stages are cutlass.Array subviews, not bare CuTe pointers.
        staging = cutlass.make_array_view(smem)
        smem[tid] = cutlass.Uint8(0x5A)
        smem[128 + 8 * stage_bytes + tid] = cutlass.Uint8(0xA5)
        cute.arch.sync_threads()
        cached = cute.make_rmem_tensor(
            cute.make_layout(2 * cfg.sparse_cache_words), cutlass.Int32
        )
        for page_slot in cutlass.range_constexpr(2):
            for entry in cutlass.range_constexpr(4):
                cached[page_slot * cfg.sparse_cache_words + entry] = cutlass.Int32(
                    0x7FFFFFFF
                )
                if lane < 8:
                    cached[page_slot * cfg.sparse_cache_words + entry] = selected[
                        tile * 256 + page_slot * 128 + warp * 32 + lane * 4 + entry
                    ]
            for stage_idx in cutlass.range_constexpr(4):
                # Exercise the inherited page-plane loader and real FP8 stores,
                # including both cached-page slots and all eight retained stages.
                resource._load_fp4_stage(
                    pool,
                    staging.subview(128 + (page_slot * 4 + stage_idx) * stage_bytes),
                    cached,
                    stage_idx=stage_idx,
                    page_id_slot=page_slot,
                )
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        for page_slot in cutlass.range_constexpr(2):
            for stage_idx in cutlass.range_constexpr(4):
                for row in range(128):
                    # Independently read the native E4M3 SW128 byte layout,
                    # rather than reusing the producer's store/swizzle helper.
                    swizzled = (row * 128 + tid) ^ ((row & 7) * 16)
                    out[tile * 256 + page_slot * 128 + row, stage_idx * 128 + tid] = (
                        smem[128 + (page_slot * 4 + stage_idx) * stage_bytes + swizzled]
                    )
        guards[tile, tid] = smem[tid]
        guards[tile, 128 + tid] = smem[128 + 8 * stage_bytes + tid]

    @cute.jit
    def launch(pool, selected, out, guards, stream: cuda.CUstream):
        convert(pool, selected, out, guards).launch(
            grid=(selected.shape[0] // 256, 1, 1), block=(128, 1, 1), stream=stream
        )

    rows = torch.arange(512, device="cuda", dtype=torch.int64)[:, None]
    dims = torch.arange(512, device="cuda", dtype=torch.int64)[None, :]
    # Each group contains every E2M1 code, with changing adjacent-code order.
    codes = ((dims & 15) * (2 * (rows % 8) + 1) + rows * 7 + (dims // 16) * 3) & 15
    payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    finite_scales = torch.tensor(
        [code for code in range(256) if code not in (127, 255)],
        device="cuda",
        dtype=torch.uint8,
    )
    groups = torch.arange(32, device="cuda", dtype=torch.int64)[None, :]
    scales = finite_scales[(rows * 17 + groups * 29) % finite_scales.numel()]
    assert (scales[:, :-1] != scales[:, 1:]).all()
    coverage = torch.bincount(
        (scales.long().repeat_interleave(16, -1) * 16 + codes).flatten(),
        minlength=256 * 16,
    ).view(256, 16)
    assert (coverage[finite_scales.long()] > 0).all()
    pool = pack(payload, scales, 288)
    permutation = (torch.arange(512, device="cuda", dtype=torch.int32) * 197 + 63) % 512
    boundaries = torch.tensor(
        [0, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 511, -1, 0x7FFFFFFF, 512, -17],
        device="cuda",
        dtype=torch.int32,
    ).repeat(16)
    selected = torch.cat((permutation, boundaries))
    out = torch.full((selected.numel(), 512), 255, device="cuda", dtype=torch.uint8)
    guards = torch.empty(
        (selected.numel() // 256, 256), device="cuda", dtype=torch.uint8
    )
    tensors = [from_dlpack(x) for x in (pool.view(-1), selected, out, guards)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(launch, *tensors, stream)
    conversion_kernels.append(compiled)
    compiled(*tensors, stream)

    # Do not use unpack()/the attention LUT here: code 8 is negative zero.
    lut = torch.tensor(
        [0.0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float64,
    )
    products = lut[codes] * scales.view(torch.float8_e4m3fn).double().repeat_interleave(
        16, -1
    )
    encoded = products.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert (encoded == 0x80).any() and (products.abs() > 448).any()
    assert (products.abs() == 2**-10).any()
    valid = (selected >= 0) & (selected < 512)
    expected = torch.where(valid[:, None], encoded[selected.clamp(0, 511).long()], 0)
    assert torch.equal(out, expected)
    assert (guards[:, :128] == 0x5A).all() and (guards[:, 128:] == 0xA5).all()


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
@pytest.mark.parametrize("scale_code", [0, 1, 127])
@pytest.mark.parametrize("batch,compressed_k", [(1, 0), (128, 512), (None, 512)])
def test_e8m0_scale_boundary_changed_graph(batch, compressed_k, scale_code, backend):
    gate(backend)
    if batch is None:
        batch = torch.cuda.get_device_properties(0).multi_processor_count + 1
    # 256 * E8M0(code) is normal BF16 even for code 0. Choose Q so every
    # channel contributes exactly 2 to QK, making both output and LSE analytic.
    q = torch.full(
        (batch, 1, 64, 512),
        2.0 ** (120 - scale_code),
        device="cuda",
        dtype=torch.bfloat16,
    )
    data = torch.full((64, 512), 256.0, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full((64, 16), scale_code, device="cuda", dtype=torch.uint8)
    swa = pack(data.view(torch.uint8), scales, 528)
    wi = torch.zeros((batch, 1, 128), device="cuda", dtype=torch.int32)
    main = (
        torch.full((1, 64, 1, 288), 255, device="cuda", dtype=torch.uint8)
        if compressed_k
        else None
    )
    ci = (
        torch.full((batch, 1, compressed_k), -1, device="cuda", dtype=torch.int32)
        if compressed_k
        else None
    )
    sink = torch.full((64,), -math.inf, device="cuda")
    args = q, swa, main, wi, ci, sink
    out, lse, plan = deepseek_v41_decode(*args, backend=backend)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_decode(*args, plan=plan, backend=backend)
    expected_value = 256.0 * 2.0 ** (scale_code - 127)
    for replay in range(2):
        if replay:
            q.mul_(0.5)
        out.fill_(math.nan)
        lse.fill_(math.nan)
        if backend == "cute-dsl" and plan.workspace is not None:
            plan.workspace.fill_(-1)
        graph.replay()
        # Absolute tolerance would hide the original all-zero output bug.
        torch.testing.assert_close(
            out,
            torch.full_like(out, expected_value),
            rtol=0,
            atol=0,
        )
        expected_lse = math.log(128) + (1024 >> replay) * 512**-0.5
        torch.testing.assert_close(
            lse.double(),
            torch.full_like(lse, expected_lse, dtype=torch.float64),
            rtol=0,
            atol=1e-5,
        )


def test_mxfp8_all_finite_codes(conversion_kernels):
    gate()
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.experimental.deepseek_v41.hca_v41 import (
        BlackwellV41MixedCacheDecode,
    )

    decoder = BlackwellV41MixedCacheDecode(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(64, 64),
        mma_pv_tiler_mn=(64, 128),
        max_active_clusters=1,
        page_size_cmp=64,
        skip_correction_threshold=6.0,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=False,
    )

    @cute.kernel
    def convert(words: cute.Tensor, scales: cute.Tensor, out: cute.Tensor):
        tid = cute.arch.thread_idx()[0]
        i = cute.arch.block_idx()[0] * 128 + tid
        storage = utils.SmemAllocator().allocate(128 * 16 * 2, 128)
        smem = cute.make_tensor(
            cute.recast_ptr(storage, dtype=cutlass.BFloat16),
            cute.make_layout(128 * 16),
        )
        if i < words.shape[0]:
            chunk = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
            for j in cutlass.range_constexpr(4):
                chunk[j] = words[i, j]
            # Exercise the actual vector decode, FP32 multiply, BF16 cast and
            # swizzled shared store, including gradual underflow and signed zero.
            decoder._store_fp8_chunk(chunk, scales[i], smem.iterator, tid * 16)
            for j in cutlass.range_constexpr(16):
                out[i, j] = smem[decoder._swz(tid * 16 + j)]

    @cute.jit
    def launch(words, scales, out, stream: cuda.CUstream):
        convert(words, scales, out).launch(
            grid=(cute.ceil_div(words.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    # Every E4M3 data byte crossed with every finite E8M0 scale. Permuting
    # lanes also checks the two 8-element halves of the production conversion.
    codes = (torch.arange(256, device="cuda") ^ 0x53).to(torch.uint8)
    data = codes.repeat(255, 1).reshape(-1, 16)
    scales = torch.arange(255, device="cuda", dtype=torch.int32).repeat_interleave(16)
    words = data.view(torch.int32)
    out = torch.full(data.shape, math.nan, device="cuda", dtype=torch.bfloat16)
    tensors = [from_dlpack(x) for x in (words, scales, out)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(launch, *tensors, stream)
    conversion_kernels.append(compiled)
    compiled(*tensors, stream)
    expected = (
        data.view(torch.float8_e4m3fn).double()
        * torch.exp2(scales.double() - 127)[:, None]
    ).bfloat16()
    finite = torch.isfinite(expected)
    assert finite.any() and (~finite).any()
    # Bitwise comparison preserves the sign of zero and covers subnormal BF16.
    assert torch.equal(
        out.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
@pytest.mark.parametrize(
    "batch,compressed_k",
    [(1, 0), (1, 512), (16, 512), (128, 0), (128, 512), (None, 512)],
)
def test_finite_sink_overflow_changed_graph(batch, compressed_k, backend):
    gate(backend)
    if batch is None:
        batch = max(128, torch.cuda.get_device_properties(0).multi_processor_count + 1)
    q_values = (torch.arange(64, device="cuda") - 32).to(torch.bfloat16) / 64
    q = q_values[None, None, :, None].expand(batch, 1, 64, 512).contiguous()
    data = torch.ones((64, 512), device="cuda").to(torch.float8_e4m3fn)
    swa = pack(
        data.view(torch.uint8),
        torch.full((64, 16), 127, device="cuda", dtype=torch.uint8),
        528,
    )
    wi = torch.zeros((batch, 1, 128), device="cuda", dtype=torch.int32)
    main = (
        torch.full((1, 64, 1, 288), 255, device="cuda", dtype=torch.uint8)
        if compressed_k
        else None
    )
    ci = (
        torch.full((batch, 1, compressed_k), -1, device="cuda", dtype=torch.int32)
        if compressed_k
        else None
    )
    maximum = torch.finfo(torch.float32).max
    boundary = torch.tensor(maximum / math.log2(math.e), device="cuda")
    below = torch.nextafter(boundary, torch.full_like(boundary, -math.inf)).item()
    above = torch.nextafter(boundary, torch.full_like(boundary, math.inf)).item()
    sink = torch.tensor(
        [
            maximum,
            above,
            below,
            -maximum,
            -above,
            -below,
            1e30,
            -1e30,
            1000,
            -1000,
            -math.inf,
            0,
            math.log(128),
            math.log(128) + 1,
            1,
            -1,
        ],
        device="cuda",
        dtype=torch.float32,
    ).repeat(4)
    args = q, swa, main, wi, ci, sink
    out, lse, plan = deepseek_v41_decode(*args, backend=backend)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        deepseek_v41_decode(*args, plan=plan, backend=backend)
    torch.cuda.current_stream().wait_stream(stream)
    for replay in range(3):
        if replay:
            q.mul_(-0.5)
            sink.copy_(sink.roll(7))
        if replay == 2:
            wi[0].fill_(-1)
        out.fill_(math.nan)
        lse.fill_(math.nan)
        if backend == "cute-dsl" and plan.workspace is not None:
            plan.workspace.fill_(-1)
        graph.replay()
        # All real values are exactly one, so the output is their probability
        # mass. Evaluate the sigmoid in FP64 without either kernel's log2 path.
        expected_lse = q[:, 0, :, 0].double() * math.sqrt(512) + math.log(128)
        expected_out = torch.sigmoid(expected_lse - sink.double()[None, :])
        if replay == 2:
            expected_out[0] = 0
            expected_lse[0] = -math.inf
        torch.testing.assert_close(
            out[:, 0].double(),
            expected_out[:, :, None].expand(batch, 64, 512),
            rtol=0.005,
            atol=0,
        )
        torch.testing.assert_close(
            lse[..., 0].double(),
            expected_lse,
            rtol=0,
            atol=1e-5,
        )


def make_fp8_case(batch, compressed):
    q, _, main, wi, ci, sink = make_case(batch, compressed, seed=9175)
    q = (q.float() * 0.5).to(torch.float8_e4m3fn)
    swa = (torch.randn(8, 64, 1, 512, device=q.device) * 0.5).to(torch.float8_e4m3fn)
    return q, swa, main, wi, ci, sink


def fp8_oracle(q, swa, global_cache, wi, ci, sink):
    """FP64 attention and the native componentwise FP8 rounding bound.

    Use stored E4M3 Q/SWA and independently decode E2M1 times group16 E4M3
    scales, then RTNE/saturate main KV to unit E4M3 before both QK and PV.
    Neither the adapter nor its source states enter the oracle.
    """
    values, masks = [], []
    for pool, ids, fp4 in ((swa, wi, False), (global_cache, ci, True)):
        if pool is None:
            continue
        if not fp4:
            decoded = pool.double().reshape(-1, 512)
        else:
            pages = pool.view(pool.shape[0], -1)
            payload = pages[:, : 64 * 256].reshape(-1, 256)
            codes = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(-1, 512)
            lut = torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
                device=q.device,
                dtype=torch.float64,
            )
            scales = (
                pages[:, 64 * 256 :]
                .contiguous()
                .view(torch.float8_e4m3fn)
                .reshape(-1, 32)
                .double()
            )
            decoded = lut[codes.long()] * scales.repeat_interleave(16, -1)
            decoded = decoded.clamp(-448, 448).to(torch.float8_e4m3fn).double()
        index = ids[:, 0].long()
        valid = (index >= 0) & (index < decoded.shape[0]) & (index != 0x7FFFFFFF)
        gathered = decoded[index.clamp(0, decoded.shape[0] - 1)]
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


@pytest.mark.parametrize("compressed", list(range(0, 513, 64)))
@pytest.mark.parametrize("batch", [1, 4])
def test_fp8_swa_fp64_and_selection_counts(batch, compressed):
    gate("prims-ts")
    args = make_fp8_case(batch, compressed)
    q, swa, main, wi, ci, sink = args
    wi[..., ::11] = -1
    wi[..., 1::17] = 0x7FFFFFFF
    wi[..., 2::23] = swa.shape[0] * 64
    wi[..., 3::29] = -17
    wi[..., 4:8] = torch.tensor([63, 64, 64, 127], device=q.device, dtype=torch.int32)
    if compressed:
        ci[..., ::7] = -1
        ci[..., 1::19] = 0x7FFFFFFF
        ci[..., 2::31] = main.shape[0] * 64
        ci[..., 3:7] = torch.tensor(
            [63, 64, 64, 127], device=q.device, dtype=torch.int32
        )
    if batch > 1:
        wi[0].fill_(-1)
        wi[1].fill_(-1)
        if compressed:
            ci[0].fill_(-1)
            ci[2].fill_(-1)
    reference = fp8_oracle(*args)
    out, lse, plan = deepseek_v41_decode(*args, backend="prims-ts")
    check_fp8(out, lse, reference)
    assert plan.part_s.dtype == torch.float32 and plan.lse_s.dtype == torch.float32
    assert not hasattr(plan, "expanded_q")
    expected_counts = ((wi >= 0) & (wi < swa.shape[0] * 64)).sum((1, 2)).int()
    torch.testing.assert_close(plan.swa_counts, expected_counts, rtol=0, atol=0)
    assert (plan.swa_lengths == 128).all() and (plan.empty_lengths == 0).all()
    if compressed:
        expected_global = ((ci >= 0) & (ci < main.shape[0] * 64)).sum((1, 2)).int()
        torch.testing.assert_close(plan.global_counts, expected_global, rtol=0, atol=0)
        assert (plan.global_lengths == compressed).all()
        assert plan.part_c.dtype == torch.float32 and plan.lse_c.dtype == torch.float32
    else:
        assert plan.part_c is None and plan.lse_c is None
        assert (plan.global_counts == 0).all() and (plan.global_lengths == 0).all()
        window_out, window_lse, _ = deepseek_v41_window_decode(
            q, swa, wi, sink, plan=plan, backend="prims-ts"
        )
        check_fp8(window_out, window_lse, reference)


@pytest.mark.parametrize("compressed", list(range(0, 513, 64)))
def test_fp8_swa_changed_graph(compressed):
    gate("prims-ts")
    args = make_fp8_case(4, compressed)
    q, swa, main, wi, ci, sink = args
    out, lse, plan = deepseek_v41_decode(*args, backend="prims-ts")
    check_fp8(out, lse, fp8_oracle(*args))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        deepseek_v41_decode(*args, plan=plan, backend="prims-ts")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        deepseek_v41_decode(*args, plan=plan, backend="prims-ts")
    torch.cuda.current_stream().wait_stream(stream)
    for step in range(3):
        q.copy_((torch.randn(q.shape, device=q.device) * 0.5).to(q.dtype))
        wi.random_(0, swa.shape[0] * 64)
        wi[..., step::4] = -1
        wi[..., 1] = 0x7FFFFFFF
        wi[..., 2] = swa.shape[0] * 64
        if compressed:
            ci.random_(0, main.shape[0] * 64)
            ci[..., step::3] = -1
            ci[..., 1] = 0x7FFFFFFF
            ci[..., 2] = main.shape[0] * 64
        sink.normal_()
        sink[0], sink[1], sink[2] = 1000, -1000, -math.inf
        if step == 1:
            replacement = make_fp8_case(4, compressed)
            swa.copy_(replacement[1])
            if compressed:
                main.copy_(replacement[2])
        if step == 2:
            wi[0].fill_(-1)
            wi[1].fill_(-1)
            if compressed:
                ci[0].fill_(-1)
                ci[2].fill_(-1)
        for state in (plan.part_s, plan.part_c, plan.lse_s, plan.lse_c):
            if state is not None:
                state.fill_(math.nan)
        for metadata in (
            plan.selected_swa,
            plan.selected_global,
            plan.swa_counts,
            plan.global_counts,
            plan.swa_lengths,
            plan.global_lengths,
            plan.empty_lengths,
        ):
            metadata.fill_(0x7FFFFFFF)
        assert not hasattr(plan, "expanded_q")
        out.fill_(math.nan)
        lse.fill_(math.nan)
        graph.replay()
        reference = fp8_oracle(*args)
        check_fp8(out, lse, reference)
        expected_counts = ((wi >= 0) & (wi < swa.shape[0] * 64)).sum((1, 2)).int()
        torch.testing.assert_close(plan.swa_counts, expected_counts, rtol=0, atol=0)
        assert (plan.swa_lengths == 128).all() and (plan.empty_lengths == 0).all()
        if compressed:
            expected_global = ((ci >= 0) & (ci < main.shape[0] * 64)).sum((1, 2)).int()
            torch.testing.assert_close(
                plan.global_counts, expected_global, rtol=0, atol=0
            )
            assert (plan.global_lengths == compressed).all()
        else:
            assert (plan.global_counts == 0).all() and (plan.global_lengths == 0).all()
    # Eager calls may change addresses but retain the declaration and ownership.
    replacement_args = tuple(None if x is None else x.clone() for x in args)
    new_out, new_lse, _ = deepseek_v41_decode(
        *replacement_args, plan=plan, backend="prims-ts"
    )
    assert new_out is out and new_lse is lse
    check_fp8(out, lse, fp8_oracle(*replacement_args))
    with pytest.raises(AssertionError):
        check_fp8(torch.ones_like(out), lse, fp8_oracle(*replacement_args))
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_decode(
            *args, plan=dataclasses.replace(plan, out=q), backend="prims-ts"
        )
    aliased_q = plan.part_s.view(torch.float8_e4m3fn).flatten()[: q.numel()].view_as(q)
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_decode(aliased_q, *args[1:], plan=plan, backend="prims-ts")


@pytest.mark.parametrize("source", ["swa", "global"])
def test_fp8_source_state_retains_fp32_precision(source):
    gate("prims-ts")
    q, swa, main, wi, ci, sink = make_fp8_case(1, 128 if source == "global" else 0)
    q.view(torch.uint8).zero_()
    swa.view(torch.uint8).zero_()
    wi.fill_(-1)
    sink.fill_(-math.inf)
    if source == "swa":
        swa.view(torch.uint8).reshape(-1, 512)[:19].fill_(0x38)  # exact E4M3 one
        wi[..., :65] = torch.arange(65, device=q.device, dtype=torch.int32)
    else:
        # Do not assume the writer maps BF16 one to exact one. Construct
        # independent E2M1=1 nibbles and E4M3=1 scale bytes in page-plane ABI.
        data = torch.zeros(128, 256, device=q.device, dtype=torch.uint8)
        data[:19].fill_(0x22)
        scales = torch.full((128, 32), 0x38, device=q.device, dtype=torch.uint8)
        main = pack(data, scales, 288)
        ci.fill_(-1)
        ci[..., :65] = torch.arange(65, device=q.device, dtype=torch.int32)
    args = q, swa, main, wi, ci, sink
    out, lse, plan = deepseek_v41_decode(*args, backend="prims-ts")
    expected = torch.full((1, 64, 512), 19 / 65, device=q.device, dtype=torch.float64)
    state = plan.part_s if source == "swa" else plan.part_c
    torch.testing.assert_close(state.double(), expected, rtol=0, atol=1e-7)
    assert (expected.bfloat16().double() - expected).abs().max() > 1e-7
    source_lse = plan.lse_s if source == "swa" else plan.lse_c
    torch.testing.assert_close(
        source_lse.double(),
        torch.full_like(source_lse, math.log2(65), dtype=torch.float64),
        rtol=0,
        atol=1e-6,
    )
    check_fp8(out, lse, fp8_oracle(*args))
    torch.testing.assert_close(out[:, 0], expected.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("compressed", [0, 128])
def test_fp8_swa_native_compile_configuration(compressed, monkeypatch):
    gate("prims-ts")
    from flashinfer.experimental.deepseek_v41 import decode as frost
    from flashinfer.experimental.deepseek_v41 import prims_ts

    recorded = []
    factory = prims_ts._make_fp8_sources

    def record(*args):
        sources = factory(*args)
        recorded.append(sources)
        return sources

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary FP8 must not compile Frost")

    monkeypatch.setattr(prims_ts, "_make_fp8_sources", record)
    monkeypatch.setattr(frost, "_compile", forbidden)
    prims_ts._compile.cache_clear()
    args = make_fp8_case(1, compressed)
    out, lse, plan = deepseek_v41_decode(*args, backend="prims-ts")
    launches = []

    def record_launch(*tensors):
        launches.append(tensors)
        return plan.kernel(*tensors)

    deepseek_v41_decode(
        *args, plan=dataclasses.replace(plan, kernel=record_launch), backend="prims-ts"
    )
    check_fp8(out, lse, fp8_oracle(*args))
    assert not hasattr(plan, "expanded_q")
    assert len(launches) == 1 and len(launches[0]) == 19
    assert launches[0][0].data_ptr() == args[0].data_ptr()
    assert launches[0][0].dtype == launches[0][1].dtype == torch.float8_e4m3fn
    assert launches[0][1].data_ptr() == args[1].data_ptr()
    bf16_buffers = [tensor for tensor in launches[0] if tensor.dtype == torch.bfloat16]
    assert len(bf16_buffers) == 1 and bf16_buffers[0].data_ptr() == out.data_ptr()
    assert len(recorded) == 1
    swa, main = recorded[0]
    cfg = swa._make_config()
    assert cfg.qkv_dtype == "e4m3" and cfg.is_fp8_qkv()
    assert cfg.tile_size_q == 64 and cfg.tile_size_kv == 128
    assert cfg.head_dim_per_stage_kv == 128 and cfg.kv_stages == 8
    assert cfg.sparse_reuse_kv and cfg.sparse_direct and not cfg.fuse_sparse_epilogue
    assert cfg.use_persistent_scheduler == 0 and cfg.num_ctas_per_seq_kv == 1
    assert swa.kv_producer is None and swa.source_state_store is not None
    assert (
        swa.source_state_store.compile_signature()[0]
        == "dsv41_fp32_normalized_source_v1"
    )
    assert swa.direct_sparse_capacities == (128, 0)
    if compressed:
        cfg = main._make_config()
        assert cfg.qkv_dtype == "e4m3" and cfg.is_fp8_qkv()
        assert cfg.tile_size_q == 64 and cfg.tile_size_kv == 128
        assert cfg.head_dim_per_stage_kv == 128 and cfg.kv_stages == 8
        assert cfg.sparse_reuse_kv and cfg.sparse_direct
        assert cfg.use_persistent_scheduler == 0 and cfg.num_ctas_per_seq_kv == 1
        from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
            Fp4Fp8KvProducer,
        )

        assert type(main.kv_producer) is Fp4Fp8KvProducer
        assert main.kv_producer.compile_signature() == (
            "dsv41_page64_fp4_group16_to_e4m3",
            main.kv_producer.native_fp4,
        )
        assert main.source_state_store is not None
        assert not cfg.fuse_sparse_epilogue and main.direct_sparse_capacities == (
            compressed,
            0,
        )
    else:
        assert main is None


@pytest.mark.parametrize("native_fp4", [False, True])
def test_fp8_main_rounding_saturation_underflow_changed_graph(native_fp4):
    gate("prims-ts")
    import cutlass
    from flashinfer.experimental.deepseek_v41.prims_ts import _compile

    if native_fp4 and (cutlass.CUDA_VERSION.major, cutlass.CUDA_VERSION.minor) < (
        13,
        2,
    ):
        pytest.skip("native BF16-register conversion requires bundled CUDA >= 13.2")

    q, swa, _, wi, ci, sink = make_fp8_case(1, 192)
    wi.fill_(-1)
    swa.view(torch.uint8).fill_(127)  # An empty SWA must not read poisoned data.
    sink.fill_(-math.inf)
    data = torch.empty((128, 256), device=q.device, dtype=torch.uint8)
    scales = torch.empty((128, 32), device=q.device, dtype=torch.uint8)
    main = torch.empty((2, 64, 1, 288), device=q.device, dtype=torch.uint8)
    query = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    args = q, swa, main, wi, ci, sink

    def update(code, scale_code, query_level, count):
        data.fill_(code | (code << 4))
        negative = code ^ 8
        data[:, 8:16].fill_(negative | (negative << 4))
        data[:, 16:24].fill_(0x22)  # An exact-one V group remains nonempty.
        scales.fill_(scale_code)
        scales[:, 2].fill_(0x38)
        main.copy_(pack(data, scales, 288))
        query.zero_()
        query[..., 0] = query_level
        q.copy_(query.to(q.dtype))
        ci.fill_(-1)
        ci[..., :count] = (
            torch.arange(count, device=q.device, dtype=torch.int32) * 197 + 63
        ) % 128

    # (E2M1 code, E4M3 scale byte, nonzero Q, exact product, unit E4M3, count).
    # The first is a halfway-even round-up; the others saturate and underflow.
    cases = [
        (3, 0x39, 1.0, 1.6875, 1.75, 64),
        (7, 0x7E, 2**-8, 2688.0, 448.0, 127),
        (1, 0x01, 1.0, 2**-10, 0.0, 129),
    ]
    update(cases[0][0], cases[0][1], cases[0][2], cases[0][5])
    _, _, plan = deepseek_v41_decode(*args, backend="prims-ts")
    plan = dataclasses.replace(
        plan, kernel=_compile(q.device.index, 192, native_fp4=native_fp4)
    )
    out, lse, _ = deepseek_v41_decode(*args, plan=plan, backend="prims-ts")
    assert not hasattr(plan, "expanded_q")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_decode(*args, plan=plan, backend="prims-ts")
    for code, scale_code, query_level, raw, converted, count in cases:
        update(code, scale_code, query_level, count)
        for state in (plan.part_s, plan.part_c, plan.lse_s, plan.lse_c):
            state.fill_(math.nan)
        for metadata in (
            plan.selected_swa,
            plan.selected_global,
            plan.swa_counts,
            plan.global_counts,
            plan.swa_lengths,
            plan.global_lengths,
            plan.empty_lengths,
        ):
            metadata.fill_(0x7FFFFFFF)
        out.fill_(math.nan)
        lse.fill_(math.nan)
        graph.replay()
        check_fp8(out, lse, fp8_oracle(*args))
        # Identical live rows make every normalized probability exactly the
        # same. This exact V gate, unlike the general FP8 rounding bound, rejects
        # a BF16 V pass even if its K values alone were converted correctly.
        expected = torch.full_like(out, converted)
        expected[..., 16:32] = -converted
        expected[..., 32:48] = 1
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        expected_lse = math.log(count) + query_level * converted * 512**-0.5
        torch.testing.assert_close(
            lse.double(),
            torch.full_like(lse, expected_lse, dtype=torch.float64),
            rtol=0,
            atol=1e-5,
        )
        # Conversely, a BF16 K pass with only V converted must fail strict LSE.
        assert abs(query_level * (raw - converted) * 512**-0.5) > 1e-5
        assert raw != converted
        assert (plan.global_counts == count).all() and (plan.swa_counts == 0).all()
        assert (plan.global_lengths == 192).all() and (plan.swa_lengths == 128).all()
        assert (plan.empty_lengths == 0).all()


def test_fp8_swa_fp4_conversion_fallback():
    gate("prims-ts")
    from flashinfer.experimental.deepseek_v41.prims_ts import _compile

    args = make_fp8_case(4, 512)
    args[3][0].fill_(-1)
    _, _, plan = deepseek_v41_decode(*args, backend="prims-ts")
    fallback = dataclasses.replace(
        plan, kernel=_compile(args[0].device.index, 512, native_fp4=False)
    )
    out, lse, _ = deepseek_v41_decode(*args, plan=fallback, backend="prims-ts")
    reference = fp8_oracle(*args)
    check_fp8(out, lse, reference)


def test_fp8_swa_sink_applied_once():
    gate("prims-ts")
    q, swa, main, wi, ci, sink = make_fp8_case(1, 512)
    q.view(torch.uint8).zero_()
    swa.view(torch.uint8).fill_(0x38)
    main = pack(
        torch.full((512, 256), 0x22, device=q.device, dtype=torch.uint8),
        torch.full((512, 32), 0x38, device=q.device, dtype=torch.uint8),
        288,
    )
    wi.copy_(torch.arange(128, device=q.device, dtype=torch.int32).view(1, 1, 128))
    ci.copy_(torch.arange(512, device=q.device, dtype=torch.int32).view(1, 1, 512))
    sink.fill_(math.log(640))
    out, lse, plan = deepseek_v41_decode(q, swa, main, wi, ci, sink, backend="prims-ts")
    torch.testing.assert_close(
        plan.part_s, torch.ones_like(plan.part_s), rtol=0, atol=1e-7
    )
    torch.testing.assert_close(
        plan.part_c, torch.ones_like(plan.part_c), rtol=0, atol=1e-7
    )
    torch.testing.assert_close(out, torch.full_like(out, 0.5), rtol=0, atol=0)
    check_fp8(out, lse, fp8_oracle(q, swa, main, wi, ci, sink))


def test_fp8_swa_empty_poison_and_mode_guards(monkeypatch):
    gate("prims-ts")
    from flashinfer.experimental.deepseek_v41 import prims_ts

    args = make_fp8_case(2, 64)
    args[1].view(torch.uint8).fill_(127)
    args[2].fill_(255)
    args[3].fill_(-1)
    args[4].fill_(0x7FFFFFFF)
    args[5].fill_(-math.inf)
    out, lse, plan = deepseek_v41_decode(*args, backend="prims-ts")
    check_fp8(out, lse, fp8_oracle(*args))
    assert torch.equal(plan.part_s, torch.zeros_like(plan.part_s))
    assert torch.equal(plan.part_c, torch.zeros_like(plan.part_c))
    with pytest.raises(ValueError, match="Q must be BF16"):
        deepseek_v41_decode(*args)
    with pytest.raises(ValueError, match="page size"):
        deepseek_v41_decode(args[0].bfloat16(), *args[1:], backend="prims-ts")
    legacy = make_case(2, 64)
    with pytest.raises(ValueError, match="E4M3 SWA"):
        deepseek_v41_decode(args[0], legacy[1], *args[2:], backend="prims-ts")
    _, _, frost_plan = deepseek_v41_decode(*legacy)
    with pytest.raises(ValueError, match="mismatch"):
        deepseek_v41_decode(*args, plan=frost_plan, backend="prims-ts")
    with pytest.raises(ValueError, match="mismatch"):
        deepseek_v41_decode(*legacy, plan=plan)
    _, _, packed_plan = deepseek_v41_decode(*legacy, backend="prims-ts")
    with pytest.raises(ValueError, match="mismatch"):
        deepseek_v41_decode(*args, plan=packed_plan, backend="prims-ts")
    with pytest.raises(ValueError, match="mismatch"):
        deepseek_v41_decode(*legacy, plan=plan, backend="prims-ts")
    with pytest.raises(ValueError, match="mismatch"):
        deepseek_v41_decode(*args, plan=object(), backend="prims-ts")

    def no_compile(*args, **kwargs):
        raise AssertionError("unprepared capture must reject before compilation")

    monkeypatch.setattr(prims_ts, "_compile", no_compile)
    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(graph),
        pytest.raises(ValueError, match="before CUDA Graph capture"),
    ):
        deepseek_v41_decode(*args, backend="prims-ts")


@pytest.mark.parametrize("source", ["swa", "global"])
def test_fp8_swa_cache_offsets_above_two_gib(source):
    gate("prims-ts")
    args = make_fp8_case(1, 64)
    index, width, ids_index = (1, 512, 3) if source == "swa" else (2, 288, 4)
    small = args[index]
    page_bytes = 64 * width
    first_page = (2**31 + page_bytes - 1) // page_bytes
    large = torch.empty(
        (first_page + small.shape[0], 64, 1, width),
        device=small.device,
        dtype=small.dtype,
    )
    large[first_page:].copy_(small)
    actual = list(args)
    actual[index] = large
    actual[ids_index] = args[ids_index] + first_page * 64
    out, lse, _ = deepseek_v41_decode(*actual, backend="prims-ts")
    # Reference the small pool: uninitialized large prefixes must never be read.
    check_fp8(out, lse, fp8_oracle(*args))


def test_fp8_swa_source_anchor_crossings_and_extreme_finite_sinks():
    gate("prims-ts")
    q, swa, _, wi, ci, sink = make_fp8_case(4, 512)
    q.copy_(torch.full(q.shape, 0.5, device=q.device).to(q.dtype))
    window_levels = torch.tensor([0.0625, 0.5], device=q.device).repeat_interleave(64)
    swa.view(torch.uint8).zero_()
    swa.view(-1, 512)[:128].copy_(window_levels[:, None].expand(-1, 512).to(swa.dtype))
    levels = torch.tensor([0.0625, 0.25, 0.5, 1, 4, 0.125, 8, 0.5], device=q.device)
    scales = (
        levels.repeat_interleave(64)[:, None]
        .expand(-1, 32)
        .contiguous()
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    data = torch.full((512, 256), 0x22, device=q.device, dtype=torch.uint8)
    main = pack(data, scales, 288)
    wi.copy_(
        torch.arange(128, device=q.device, dtype=torch.int32)
        .view(1, 1, 128)
        .expand(4, -1, -1)
    )
    ci.copy_(
        torch.arange(512, device=q.device, dtype=torch.int32)
        .view(1, 1, 512)
        .expand(4, -1, -1)
    )
    wi[1].fill_(-1)
    ci[2].fill_(-1)
    wi[3].fill_(-1)
    ci[3].fill_(-1)
    sink[0], sink[1], sink[2], sink[3] = (
        -math.inf,
        torch.finfo(torch.float32).max,
        -torch.finfo(torch.float32).max,
        1000,
    )
    args = q, swa, main, wi, ci, sink
    out, lse, _ = deepseek_v41_decode(*args, backend="prims-ts")
    reference = fp8_oracle(*args)
    check_fp8(out, lse, reference)
