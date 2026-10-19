# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Independent FP64 oracle and changed-input replay for both DS4.1 backends."""

import builtins
import dataclasses
import math

import pytest
import torch

from flashinfer.deepseek_v41 import (
    deepseek_v41_decode,
    deepseek_v41_decode_workspace_size,
    deepseek_v41_window_decode,
)


@dataclasses.dataclass
class DecodeBuffers:
    """Test-owned caller storage, not a production execution plan.

    Six-tensor oracle fixtures keep separate wi/ci tables. Only decode_case
    copies them into this caller-owned combined table; these extra test copies
    are intentionally excluded from the canonical single-kernel graph test.
    """

    workspace: torch.Tensor
    indices: torch.Tensor
    topk_lens: torch.Tensor
    seq_lens: torch.Tensor
    out: torch.Tensor
    backend: str

    @classmethod
    def allocate(cls, q, compressed, backend):
        batch = q.shape[0]
        return cls(
            torch.empty(
                deepseek_v41_decode_workspace_size(batch, compressed, backend=backend),
                device=q.device,
                dtype=torch.uint8,
            ),
            torch.empty((batch, 128 + compressed), device=q.device, dtype=torch.int32),
            torch.full((batch,), 128 + compressed, device=q.device, dtype=torch.int32),
            torch.full((batch,), 128, device=q.device, dtype=torch.int32),
            torch.empty(q.shape, device=q.device, dtype=torch.bfloat16),
            backend,
        )

    @property
    def views(self):
        from flashinfer.experimental.deepseek_v41._workspace import workspace_views

        return workspace_views(
            self.workspace,
            self.indices.shape[0],
            self.indices.shape[1] - 128,
            backend=self.backend,
        )

    @property
    def lse(self):
        return self.views.lse

    def call(self, q, swa, main, sink, **kwargs):
        return deepseek_v41_decode(
            q,
            swa,
            self.workspace,
            self.indices,
            main,
            self.topk_lens,
            self.seq_lens,
            out=self.out,
            bmm1_scale=kwargs.pop("bmm1_scale", 512**-0.5),
            bmm2_scale=kwargs.pop("bmm2_scale", 1.0),
            sinks=sink,
            kv_layout=kwargs.pop("kv_layout", "NHD"),
            backend=self.backend,
            **kwargs,
        )


def decode_case(
    q, swa, main, wi, ci, sink, *, buffers=None, backend="cute-dsl", **kwargs
):
    """Adapt oracle inputs explicitly; the public API returns only output."""
    if buffers is None:
        buffers = DecodeBuffers.allocate(q, 0 if ci is None else ci.shape[-1], backend)
    if not isinstance(buffers, DecodeBuffers) or buffers.backend != backend:
        raise ValueError("test buffers declaration mismatch")
    buffers.indices[:, :128].copy_(wi[:, 0])
    if ci is not None:
        buffers.indices[:, 128:].copy_(ci[:, 0])
    out = buffers.call(q, swa, main, sink, **kwargs)
    assert out is buffers.out
    return out, buffers.lse, buffers


def window_case(q, swa, wi, sink, *, buffers=None, backend="cute-dsl", **kwargs):
    if buffers is None:
        buffers = DecodeBuffers.allocate(q, 0, backend)
    buffers.indices.copy_(wi[:, 0])
    out = deepseek_v41_window_decode(
        q,
        swa,
        buffers.workspace,
        buffers.indices,
        buffers.topk_lens,
        buffers.seq_lens,
        out=buffers.out,
        bmm1_scale=kwargs.pop("bmm1_scale", 512**-0.5),
        bmm2_scale=kwargs.pop("bmm2_scale", 1.0),
        sinks=sink,
        kv_layout=kwargs.pop("kv_layout", "NHD"),
        backend=backend,
        **kwargs,
    )
    assert out is buffers.out
    return out, buffers.lse, buffers


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


def oracle(args, *, bmm1_scale=512**-0.5, bmm2_scale=1.0):
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
    scores = torch.einsum("bhd,bkd->bhk", q[:, 0].double(), kv) * bmm1_scale
    scores.masked_fill_(~valid[:, None], -float("inf"))
    lse = scores.logsumexp(-1)
    augmented = torch.cat(
        (scores, sink.double()[None, :, None].expand(q.shape[0], -1, 1)), dim=-1
    )
    probability = augmented.softmax(-1)[..., :-1].nan_to_num()
    out = torch.einsum("bhk,bkd->bhd", probability, kv) * bmm2_scale
    return out, lse


def check(args, out, lse, *, bmm1_scale=512**-0.5, bmm2_scale=1.0):
    expected, expected_lse = oracle(args, bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale)
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
        api = deepseek_v41_window_decode if window_only else deepseek_v41_decode
        api(None, None, None, backend=backend)


@pytest.mark.parametrize("compressed_k", [0, 64])
def test_prims_ts_does_not_compile_cute_dsl(compressed_k, monkeypatch):
    gate("prims-ts")
    from flashinfer.experimental.deepseek_v41 import decode as cute_decode

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("prims-ts must not invoke the Frost compiler")

    monkeypatch.setattr(cute_decode, "_compile", forbidden_compile)
    args = make_fp8_case(2, compressed_k)
    out, lse, buffers = decode_case(*args, backend="prims-ts")
    assert out is buffers.out and lse.data_ptr() == buffers.lse.data_ptr()
    assert buffers.workspace.dtype == torch.uint8
    assert not hasattr(buffers.views, "main_scale")
    assert not hasattr(buffers.views, "part_s")
    check_fp8(out, lse, fp8_oracle(*args))
    if compressed_k:
        replay = decode_case(*args, backend="prims-ts", buffers=buffers)
    else:
        replay = window_case(
            args[0], args[1], args[3], args[5], backend="prims-ts", buffers=buffers
        )
    assert replay[0] is out and replay[1].data_ptr() == lse.data_ptr()
    assert replay[2] is buffers


@pytest.mark.parametrize("compressed_k", [0, 64, 192, 512])
def test_prims_ts_selection_refresh(compressed_k):
    gate("prims-ts")
    args = make_fp8_case(2, compressed_k)
    for pool, indices in ((args[1], args[3]), (args[2], args[4])):
        if indices is None:
            continue
        capacity = pool.shape[0] * 64
        indices[:, 0, :8] = torch.tensor(
            [0, 0, capacity - 1, capacity, -1, -(2**31), 2**31 - 1, capacity + 1],
            device="cuda",
            dtype=torch.int32,
        )
    out, lse, buffers = decode_case(*args, backend="prims-ts")

    def check_selection():
        selected = []
        for pool, indices in ((args[1], args[3]), (args[2], args[4])):
            if indices is None:
                selected.append(None)
            else:
                valid = (
                    (indices >= 0)
                    & (indices != 0x7FFFFFFF)
                    & (indices < pool.shape[0] * 64)
                )
                selected.append(torch.where(valid, indices, -1))
        # Native selection is fused: reference masks, not prepared scratch,
        # determine which physical positions enter the independent oracle.
        reference_args = (*args[:3], *selected, args[5])
        check_fp8(out, lse, fp8_oracle(*reference_args))

    check_selection()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_case(*args, backend="prims-ts", buffers=buffers)
    # Poison caller scratch and change which positions are holes, without
    # changing graph-bound addresses. The global tile can start with a hole.
    buffers.workspace.fill_(255)
    for indices in (args[3], args[4]):
        if indices is not None:
            indices[:, 0, :8].copy_(indices[:, 0, :8].flip(-1))
    graph.replay()
    check_selection()


@pytest.mark.parametrize("compressed_k", [0, 64, 192, 512])
def test_prims_ts_zero_query_tile_counts(compressed_k):
    gate("prims-ts")
    args = make_fp8_case(4, compressed_k)
    args[0].view(torch.uint8).zero_()
    for indices in (args[3], args[4]):
        if indices is not None:
            indices[:, 0, ::64].fill_(-1)
            indices[3].fill_(-1)
    args[3][1].fill_(-1)
    if args[4] is not None:
        args[4][2].fill_(-1)
    # With Q=0 every valid selection contributes one to the sink-exclusive
    # denominator. Cover mixed, global-only, window-only and empty requests;
    # holes at page/BK128 boundaries must not truncate the remaining tiles.
    counts = torch.zeros(4, device="cuda", dtype=torch.float64)
    for pool, indices in ((args[1], args[3]), (args[2], args[4])):
        if indices is not None:
            valid = (indices[:, 0] >= 0) & (indices[:, 0] < pool.shape[0] * 64)
            counts += valid.sum(-1)
    out, lse, _ = decode_case(*args, backend="prims-ts")
    torch.testing.assert_close(
        lse[..., 0].double(), counts.log()[:, None].expand(-1, 64), rtol=0, atol=1e-5
    )
    check_fp8(out, lse, fp8_oracle(*args))


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
def test_prepare_buffers_before_capture(backend, monkeypatch):
    gate(backend)
    case = make_fp8_case if backend == "prims-ts" else make_case
    args = case(1, 64)
    buffers = DecodeBuffers.allocate(args[0], 64, backend)
    from flashinfer.experimental.deepseek_v41 import _workspace

    _workspace.device_sm_count.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="warm.*before CUDA Graph capture"):
        buffers.call(args[0], args[1], args[2], args[5])
    with pytest.raises(ValueError, match="caller-supplied out"):
        deepseek_v41_decode(
            args[0],
            args[1],
            buffers.workspace,
            buffers.indices,
            args[2],
            buffers.topk_lens,
            buffers.seq_lens,
            kv_layout="NHD",
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
@pytest.mark.parametrize("window_only", [False, True])
def test_removed_public_plan_rejected(backend, window_only):
    # The migration intentionally provides no production plan/tuple shim.
    api = deepseek_v41_window_decode if window_only else deepseek_v41_decode
    with pytest.raises(TypeError, match="plan"):
        api(None, None, None, backend=backend, plan=object())


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
def test_decode_changed_graph_and_fp64(batch, compressed_k, monkeypatch):
    gate()
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
    out, lse, buffers = decode_case(*args, backend="cute-dsl")

    def run():
        if compressed_k:
            return decode_case(*args, buffers=buffers, backend="cute-dsl")
        return window_case(
            args[0], args[1], args[3], args[5], buffers=buffers, backend="cute-dsl"
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
        decode_case(
            *args, buffers=dataclasses.replace(buffers, out=args[0]), backend="cute-dsl"
        )
    with pytest.raises(ValueError, match="mismatch"):
        decode_case(*args, buffers=object(), backend="cute-dsl")
    with pytest.raises(ValueError, match="inference"):
        decode_case(
            args[0].requires_grad_(), *args[1:], buffers=buffers, backend="cute-dsl"
        )


def test_empty_uninitialized_cache_and_declaration_guards():
    gate()
    q, swa, main, wi, ci, sink = make_case(2, 512)
    swa.fill_(255)
    main.fill_(255)
    wi.fill_(-1)
    ci.fill_(2**31 - 1)
    sink.fill_(-math.inf)
    out, lse, buffers = decode_case(q, swa, main, wi, ci, sink)
    assert torch.equal(out, torch.zeros_like(out)) and torch.isneginf(lse).all()
    with pytest.raises(ValueError, match="Q must"):
        buffers.call(q.expand(-1, 2, -1, -1).contiguous(), swa, main, sink)
    with pytest.raises(ValueError, match="page64 pool"):
        buffers.call(q, swa.view(-1, 32, 1, 528), main, sink)
    with pytest.raises(ValueError, match="compressed_kv_cache"):
        buffers.call(q, swa, None, sink)
    with pytest.raises(ValueError, match="sparse_indices"):
        dataclasses.replace(buffers, indices=buffers.indices.long()).call(
            q, swa, main, sink
        )
    with pytest.raises(ValueError, match="compressed_topk"):
        dataclasses.replace(buffers, indices=buffers.indices[:, :-1].contiguous()).call(
            q, swa, main, sink
        )
    with pytest.raises(ValueError, match="sinks"):
        buffers.call(q, swa, main, sink.double())
    misaligned = torch.empty(q.numel() + 1, device=q.device, dtype=q.dtype)[1:].view_as(
        q
    )
    with pytest.raises(ValueError, match="16-byte aligned"):
        buffers.call(misaligned, swa, main, sink)
    with pytest.raises(TypeError):
        buffers.call(q, swa, main, sink, arithmetic="bf16x3")


@pytest.mark.parametrize("batch", [128, 256])
@pytest.mark.parametrize("pool_index,width,ids_index", [(1, 528, 3), (2, 288, 4)])
def test_cache_offsets_above_two_gib(batch, pool_index, width, ids_index):
    gate()
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
    out, lse, _ = decode_case(*actual, backend="cute-dsl")
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
    out, lse, buffers = decode_case(*actual, backend="cute-dsl")
    check(args, out, lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_case(*actual, buffers=buffers, backend="cute-dsl")
    args[0].mul_(0.75)
    args[ids_index].copy_(args[ids_index].flip(-1))
    actual[ids_index].copy_(actual[ids_index].flip(-1))
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    check(args, out, lse)


@pytest.mark.parametrize(
    "levels",
    [
        [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.75],
        [0.0625, 0.5, 0.125, 0.75, 0.125, 0.375, 0.0625, 1.0],
        [0.0625] * 7 + [0.75],
    ],
)
@pytest.mark.parametrize("batch", [16, 128, 256])
def test_normalization_anchor_crossings(levels, batch):
    gate()
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
    out, lse, _ = decode_case(*args, backend="cute-dsl")
    check(args, out, lse)


def test_fp4_conversion_fallback(monkeypatch):
    gate()
    from flashinfer.experimental.deepseek_v41 import decode as frost

    factory = frost._compile

    def fallback(*args, **kwargs):
        kwargs["native_fp4"] = False
        return factory(*args, **kwargs)

    monkeypatch.setattr(frost, "_compile", fallback)
    args = make_case(128, 512)
    out, lse, _ = decode_case(*args, backend="cute-dsl")
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
@pytest.mark.parametrize("source", ["main", "swa"])
def test_fp4_fp8_all_finite_codes(native_fp4, source, conversion_kernels):
    gate("prims-ts")
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_attention
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
        MixedFp8SmemKvResource,
    )

    if native_fp4 and (cutlass.CUDA_VERSION.major, cutlass.CUDA_VERSION.minor) < (
        13,
        2,
    ):
        pytest.skip("native BF16-register conversion requires bundled CUDA >= 13.2")

    attention = _make_fp8_attention(512, 1, native_fp4)
    cfg = attention._make_config()
    assert cfg.qkv_dtype == "e4m3" and cfg.tile_size_kv == 128
    assert cfg.head_dim_per_stage_kv == 128 and cfg.kv_stages == 8
    assert not cfg.sparse_request_kv_scale
    # The standalone CTA contains just the production four-warp load group.
    # Change its warp origin, not the tile shape, dtype or stage/store layout.
    probe_cfg = dataclasses.replace(cfg, load_warp_idx=0)
    stage_bytes = 128 * 128

    @cute.kernel
    def convert(
        pool: cute.Tensor,
        selected: cute.Tensor,
        swa_pool: cute.Tensor,
        out: cute.Tensor,
        guards: cute.Tensor,
    ):
        tid = cute.arch.thread_idx()[0]
        request, tile = cute.arch.block_idx()[0], cute.arch.block_idx()[1]
        lane, warp = tid % 32, tid // 32
        resource = attention.kv_producer.make_resource(
            (pool, swa_pool),
            name="fp4_fp8_probe",
            cfg=probe_cfg,
            batch_idx=request,
        )
        assert type(resource) is MixedFp8SmemKvResource
        storage = utils.SmemAllocator().allocate(8 * stage_bytes + 256, 128)
        smem = cute.make_tensor(
            cute.recast_ptr(storage, dtype=cutlass.Uint8),
            cute.make_layout((8 * stage_bytes + 256,)),
        )
        # Keep the flat rank-one tuple layout required by make_array_view.
        # Production stages are native Array subviews, not bare CuTe pointers.
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
                        request,
                        tile * 256 + page_slot * 128 + warp * 32 + lane * 4 + entry,
                    ]
            for stage_idx in cutlass.range_constexpr(4):
                # Both physical producer paths use the real mixed resource,
                # two cached-page slots and every retained D128 stage.
                stage = staging.subview(128 + (page_slot * 4 + stage_idx) * stage_bytes)
                if cutlass.const_expr(source == "main"):
                    resource._load_fp4_stage(
                        pool,
                        stage,
                        cached,
                        stage_idx=stage_idx,
                        page_id_slot=page_slot,
                    )
                else:
                    resource._load_swa_stage(
                        stage,
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
                    out[
                        request,
                        tile * 256 + page_slot * 128 + row,
                        stage_idx * 128 + tid,
                    ] = smem[128 + (page_slot * 4 + stage_idx) * stage_bytes + swizzled]
        guards[request, tile, tid] = smem[tid]
        guards[request, tile, 128 + tid] = smem[128 + 8 * stage_bytes + tid]

    @cute.jit
    def launch(pool, selected, swa_pool, out, guards, stream: cuda.CUstream):
        convert(pool, selected, swa_pool, out, guards).launch(
            grid=(selected.shape[0], selected.shape[1] // 256, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    rows = torch.arange(768, device="cuda", dtype=torch.int64)[:, None]
    dims = torch.arange(512, device="cuda", dtype=torch.int64)[None, :]
    # The first request selects all 16 * 254 finite code/scale pairs, with
    # changing adjacent-code order, finite signed scales and both signed zeros.
    codes = ((dims & 15) * (2 * (rows % 8) + 1) + rows * 7 + (dims // 16) * 3) & 15
    payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    finite_scales = torch.tensor(
        [code for code in range(256) if code not in (127, 255)],
        device="cuda",
        dtype=torch.uint8,
    )
    groups = torch.arange(32, device="cuda", dtype=torch.int64)[None, :]
    scales = finite_scales[(rows * 17 + groups * 29) % finite_scales.numel()]
    assert (scales[:512, :-1] != scales[:512, 1:]).all()
    coverage = torch.bincount(
        (scales[:512].long().repeat_interleave(16, -1) * 16 + codes[:512]).flatten(),
        minlength=256 * 16,
    ).view(256, 16)
    assert (coverage[finite_scales.long()] > 0).all()
    # Disjoint finite scales cover independent requests without any coupled
    # normalization: a large value in one request cannot alter another's bytes.
    scales[512:].fill_(0x6E)
    pool = pack(payload, scales, 288)
    permutation = (torch.arange(512, device="cuda", dtype=torch.int32) * 197 + 63) % 512
    second = (
        torch.arange(256, device="cuda", dtype=torch.int32) * 197 + 63
    ) % 256 + 512
    boundaries = torch.tensor(
        [
            512,
            519,
            520,
            527,
            528,
            543,
            544,
            575,
            576,
            639,
            640,
            767,
            -1,
            0x7FFFFFFF,
            768,
            -17,
            0,
            7,
            8,
            15,
            16,
            31,
            32,
            63,
            64,
            127,
            128,
            129,
            767,
            768,
            -1,
            0x7FFFFFFF,
        ],
        device="cuda",
        dtype=torch.int32,
    ).repeat(8)
    ci = torch.stack((permutation, torch.cat((second, boundaries))))
    # Feed boundary addresses to the real loaders, not a pre-masked reference.
    # Each source must use its own capacity in both unequal-capacity directions.
    selected = ci
    swa_bytes = finite_scales[
        (torch.arange(768 * 512, device="cuda").view(768, 512) * 17 + 29) % 254
    ]
    full_pool = pool
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    for main_capacity, swa_capacity in ((768, 128), (128, 768)):
        pool = full_pool[: main_capacity // 64]
        source_bytes = swa_bytes[:swa_capacity]
        swa_pool = source_bytes.view(torch.float8_e4m3fn).view(-1, 64, 1, 512)
        out = torch.full((2, 512, 512), 255, device="cuda", dtype=torch.uint8)
        guards = torch.empty((2, 2, 256), device="cuda", dtype=torch.uint8)
        tensors = [
            from_dlpack(x)
            for x in (pool.view(-1), selected, swa_pool.view(-1), out, guards)
        ]
        compiled = cute.compile(launch, *tensors, stream)
        conversion_kernels.append(compiled)
        compiled(*tensors, stream)
        capacity = main_capacity if source == "main" else swa_capacity
        valid = (ci >= 0) & (ci < capacity) & (ci != 0x7FFFFFFF)
        if source == "main":
            staged, expected_valid = fp8_main_reference(pool, ci[:, None])
            assert torch.equal(valid, expected_valid)
            expected = staged.view(torch.uint8)
        else:
            expected = torch.where(
                valid[..., None], source_bytes[ci.clamp(0, capacity - 1).long()], 0
            )
        assert (expected == 0x80).any()
        assert torch.equal(out, expected)
        assert (ci == capacity - 1).any() and (ci == capacity).any()
        assert (out[ci == capacity - 1] != 0).any()
        assert (out[ci == capacity] == 0).all()
        assert (guards[..., :128] == 0x5A).all() and (guards[..., 128:] == 0xA5).all()


@pytest.mark.parametrize("scale_code", [0, 1, 127])
@pytest.mark.parametrize("batch,compressed_k", [(1, 0), (128, 512), (None, 512)])
def test_e8m0_scale_boundary_changed_graph(batch, compressed_k, scale_code):
    gate()
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
    out, lse, buffers = decode_case(*args, backend="cute-dsl")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_case(*args, buffers=buffers, backend="cute-dsl")
    expected_value = 256.0 * 2.0 ** (scale_code - 127)
    for replay in range(2):
        if replay:
            q.mul_(0.5)
        out.fill_(math.nan)
        lse.fill_(math.nan)
        if buffers.workspace is not None:
            buffers.workspace.fill_(-1)
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


@pytest.mark.parametrize(
    "batch,compressed_k",
    [(1, 0), (1, 512), (16, 512), (128, 0), (128, 512), (None, 512)],
)
def test_finite_sink_overflow_changed_graph(batch, compressed_k):
    gate()
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
    out, lse, buffers = decode_case(*args, backend="cute-dsl")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        decode_case(*args, buffers=buffers, backend="cute-dsl")
    torch.cuda.current_stream().wait_stream(stream)
    for replay in range(3):
        if replay:
            q.mul_(-0.5)
            sink.copy_(sink.roll(7))
        if replay == 2:
            wi[0].fill_(-1)
        out.fill_(math.nan)
        lse.fill_(math.nan)
        if buffers.workspace is not None:
            buffers.workspace.fill_(-1)
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


def make_fp8_case(batch, compressed, seed=9175):
    q, _, main, wi, ci, sink = make_case(batch, compressed, seed=seed)
    q = (q.float() * 0.5).to(torch.float8_e4m3fn)
    swa = (torch.randn(8, 64, 1, 512, device=q.device) * 0.5).to(torch.float8_e4m3fn)
    return q, swa, main, wi, ci, sink


def fp8_main_reference(pool, ids):
    """Independent selected-byte decode and E4M3 RN/satfinite in cache units.

    Mask invalid payload/scale bytes before interpreting E4M3 scales, so empty
    requests never decode clamped uninitialized/NaN rows. Every finite E2M1 *
    E4M3 product is exact in FP32. There is no request amax, division, or descale.
    """
    pages = pool.view(pool.shape[0], -1)
    # Split each page plane without flattening across page gaps (which could
    # copy the whole pool). Only the following page/token gathers read bytes.
    payload = pages[:, : 64 * 256].view(pool.shape[0], 64, 256)
    scale_bytes = pages[:, 64 * 256 :].view(pool.shape[0], 64, 32)
    index = ids[:, 0].long()
    capacity = pool.shape[0] * 64
    valid = (index >= 0) & (index < capacity) & (index != 0x7FFFFFFF)
    selected = index.clamp(0, capacity - 1)
    page, token = selected // 64, selected % 64
    payload = torch.where(valid[..., None], payload[page, token], 0)
    scale_bytes = torch.where(
        valid[..., None], scale_bytes[page, token], 0
    ).contiguous()
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(*index.shape, 512)
    lut = torch.tensor(
        [0.0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=pool.device,
        dtype=torch.float32,
    )
    scales = scale_bytes.view(torch.float8_e4m3fn).float()
    decoded = lut[codes.long()] * scales.repeat_interleave(16, -1)
    staged = decoded.clamp(-448, 448).to(torch.float8_e4m3fn)
    return staged, valid


def fp8_oracle(
    q, swa, global_cache, wi, ci, sink, *, bmm1_scale=512**-0.5, bmm2_scale=1.0
):
    """FP64 attention and the unchanged componentwise FP8 rounding bound.

    Stored E4M3 Q/SWA remain unchanged. Main is independently clamped/rounded
    directly to E4M3 cache units before both logits and V. Only the explicit
    BMM pair restores caller-selected physical units; no production state enters.
    """
    values, masks = [], []
    for pool, ids, fp4 in ((swa, wi, False), (global_cache, ci, True)):
        if pool is None:
            continue
        if fp4:
            staged, valid = fp8_main_reference(pool, ids)
            gathered = staged.double()
        else:
            decoded = pool.double().reshape(-1, 512)
            index = ids[:, 0].long()
            valid = (index >= 0) & (index < decoded.shape[0]) & (index != 0x7FFFFFFF)
            gathered = decoded[index.clamp(0, decoded.shape[0] - 1)]
        values.append(torch.where(valid[..., None], gathered, 0))
        masks.append(valid)
    kv, valid = torch.cat(values, 1), torch.cat(masks, 1)
    logits = torch.einsum("bhd,bkd->bhk", q[:, 0].double(), kv) * bmm1_scale
    logits.masked_fill_(~valid[:, None], -math.inf)
    kv_lse = logits.logsumexp(-1)
    normalizer = torch.logaddexp(kv_lse, sink.double()[None, :])
    weights = torch.exp(logits - normalizer[..., None])
    weights = torch.where(torch.isfinite(normalizer)[..., None], weights, 0)
    expected = torch.einsum("bhk,bkd->bhd", weights, kv) * bmm2_scale
    sensitivity = torch.einsum("bhk,bkd->bhd", weights, kv.abs()) * abs(bmm2_scale)
    max_logit = logits.max(-1).values
    frame_mass = torch.exp(max_logit - normalizer)
    frame_mass = torch.where(torch.isfinite(kv_lse), frame_mass, 0)
    underflow = frame_mass[..., None] * kv.abs().sum(1)[:, None, :] * abs(bmm2_scale)
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
    out, lse, buffers = decode_case(*args, backend="prims-ts")
    check_fp8(out, lse, reference)
    assert lse.dtype == torch.float32 and out.dtype == torch.bfloat16
    assert not hasattr(buffers.views, "main_scale")
    assert not hasattr(buffers, "expanded_q")
    expected_counts = ((wi >= 0) & (wi != 0x7FFFFFFF) & (wi < swa.shape[0] * 64)).sum(
        (1, 2)
    )
    if compressed:
        expected_counts += (
            (ci >= 0) & (ci != 0x7FFFFFFF) & (ci < main.shape[0] * 64)
        ).sum((1, 2))
    assert torch.equal(
        torch.isneginf(lse[..., 0]), (expected_counts == 0)[:, None].expand(-1, 64)
    )
    if not compressed:
        window_out, window_lse, _ = window_case(
            q, swa, wi, sink, buffers=buffers, backend="prims-ts"
        )
        check_fp8(window_out, window_lse, reference)


@pytest.mark.parametrize("compressed", list(range(0, 513, 64)))
def test_fp8_swa_changed_graph(compressed):
    gate("prims-ts")
    args = make_fp8_case(4, compressed)
    q, swa, main, wi, ci, sink = args
    out, lse, buffers = decode_case(*args, backend="prims-ts")
    check_fp8(out, lse, fp8_oracle(*args))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode_case(*args, buffers=buffers, backend="prims-ts")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        decode_case(*args, buffers=buffers, backend="prims-ts")
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
            replacement = make_fp8_case(4, compressed, seed=9176)
            assert not torch.equal(
                swa.view(torch.uint8), replacement[1].view(torch.uint8)
            )
            swa.copy_(replacement[1])
            if compressed:
                assert not torch.equal(main, replacement[2])
                main.copy_(replacement[2])
        if step == 2:
            wi[0].fill_(-1)
            wi[1].fill_(-1)
            if compressed:
                ci[0].fill_(-1)
                ci[2].fill_(-1)
        buffers.workspace.fill_(255)
        assert not hasattr(buffers, "expanded_q")
        out.fill_(math.nan)
        lse.fill_(math.nan)
        graph.replay()
        reference = fp8_oracle(*args)
        check_fp8(out, lse, reference)
        expected_counts = (
            (wi >= 0) & (wi != 0x7FFFFFFF) & (wi < swa.shape[0] * 64)
        ).sum((1, 2))
        if compressed:
            expected_counts += (
                (ci >= 0) & (ci != 0x7FFFFFFF) & (ci < main.shape[0] * 64)
            ).sum((1, 2))
        assert torch.equal(
            torch.isneginf(lse[..., 0]), (expected_counts == 0)[:, None].expand(-1, 64)
        )
    # Eager calls may change addresses but retain the declaration and ownership.
    replacement_args = tuple(None if x is None else x.clone() for x in args)
    new_out, new_lse, _ = decode_case(
        *replacement_args, buffers=buffers, backend="prims-ts"
    )
    assert new_out is out and new_lse.data_ptr() == lse.data_ptr()
    check_fp8(out, lse, fp8_oracle(*replacement_args))
    with pytest.raises(AssertionError):
        check_fp8(torch.ones_like(out), lse, fp8_oracle(*replacement_args))
    aliased_q = out.view(torch.uint8).flatten()[: q.numel()].view(q.dtype).view_as(q)
    with pytest.raises(ValueError, match="overlap"):
        decode_case(aliased_q, *args[1:], buffers=buffers, backend="prims-ts")
    with pytest.raises(ValueError, match="overlap"):
        aliased_indices = (
            out.view(torch.uint8)
            .flatten()[: buffers.indices.numel() * 4]
            .view(torch.int32)
            .view_as(buffers.indices)
        )
        dataclasses.replace(buffers, indices=aliased_indices).call(q, swa, main, sink)


@pytest.mark.parametrize("source", ["swa", "global"])
def test_fp8_source_state_retains_fp32_precision(source, conversion_kernels):
    # Generic nonfinalizing source publication remains independently exercised;
    # the public joint adapter no longer allocates source states or merges them.
    gate("prims-ts")
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.attention.prims_ts.kernels.mla_decode.sparse_policy import (
        SparseMlaProfile,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
        MlaProfile,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
        ThroughputLatencyMlaDecodeTs,
    )
    from flashinfer.experimental.deepseek_v41.prims_ts_fp8 import Fp32SourceStore
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import Fp4Fp8KvProducer

    q, swa, main, wi, ci, sink = make_fp8_case(1, 128 if source == "global" else 0)
    q.view(torch.uint8).zero_()
    swa.view(torch.uint8).zero_()
    wi.fill_(-1)
    sink.fill_(-math.inf)
    if source == "swa":
        swa.view(torch.uint8).reshape(-1, 512)[:19].fill_(0x38)
        wi[..., :65] = torch.arange(65, device=q.device, dtype=torch.int32)
    else:
        data = torch.zeros(128, 256, device=q.device, dtype=torch.uint8)
        data[:19].fill_(0x22)
        main = pack(
            data, torch.full((128, 32), 0x38, device=q.device, dtype=torch.uint8), 288
        )
        ci.fill_(-1)
        ci[..., :65] = torch.arange(65, device=q.device, dtype=torch.int32)
    args = q, swa, main, wi, ci, sink
    out, lse, _ = decode_case(*args, backend="prims-ts")
    attention = ThroughputLatencyMlaDecodeTs(
        batch_size=1,
        num_heads=64,
        seq_len_q=1,
        seq_len_k=256,
        latent_dim=512,
        rope_dim=0,
        page_size=1,
        max_active_clusters=1,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        profile=MlaProfile(
            name="generic_source_probe",
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
            kv_tile_size=128,
            page_pipeline_stages=1,
        ),
        finalize_output=False,
        source_state_store=Fp32SourceStore(),
        kv_producer=Fp4Fp8KvProducer(native_fp4=False) if source == "global" else None,
    )
    attention.direct_static_scales = True
    attention.direct_sparse_capacities = (128, 0)
    assert not attention.direct_request_kv_scale and not attention.finalize_output
    selected = (wi if source == "swa" else ci)[:, 0].contiguous()
    empty = torch.full((1, 64), -1, device=q.device, dtype=torch.int32)
    lengths = torch.full((1,), 128, device=q.device, dtype=torch.int32)
    zeros = torch.zeros_like(lengths)
    counts = torch.full_like(lengths, 65)
    part = torch.full((1, 64, 512), math.nan, device=q.device, dtype=torch.float32)
    source_lse = torch.full((1, 64), math.nan, device=q.device, dtype=torch.float32)
    cache = swa.view(-1, 1, 512).permute(1, 2, 0) if source == "swa" else main.view(-1)

    @cute.jit
    def launch(
        q,
        cache,
        out,
        lse,
        selected,
        empty,
        lengths,
        zeros,
        counts,
        part,
        stream: cuda.CUstream,
    ):
        native_lse = cute.make_tensor(
            lse.iterator, cute.make_layout((64, 1, 1), stride=(1, 64, 64))
        )
        direct = (selected, empty, lengths, zeros, None, None, None, None, None)
        attention(
            q,
            q,
            cache,
            cache,
            None,
            out,
            native_lse,
            None,
            cutlass.Int32(1),
            lengths,
            None,
            direct,
            cutlass.Float32(512**-0.5),
            cutlass.Float32(1.0),
            stream,
            source_state_store_args=(part, counts),
        )

    packet = [
        from_dlpack(x)
        for x in (
            q.permute(2, 3, 1, 0),
            cache,
            out.permute(2, 3, 1, 0),
            source_lse,
            selected,
            empty,
            lengths,
            zeros,
            counts,
            part,
        )
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile[cute.FrontendNext](launch, *packet, stream)
    conversion_kernels.append(compiled)
    compiled(*packet, stream)
    expected = torch.full((1, 64, 512), 19 / 65, device=q.device, dtype=torch.float64)
    torch.testing.assert_close(part.double(), expected, rtol=0, atol=1e-7)
    assert (expected.bfloat16().double() - expected).abs().max() > 1e-7
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
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
        MixedFp8KvProducer,
    )

    recorded, launches = [], []
    factory, compile_kernel = prims_ts._make_fp8_attention, prims_ts._compile

    def record(*args):
        attention = factory(*args)
        recorded.append(attention)
        return attention

    def record_compile(*args, **kwargs):
        compiled = compile_kernel(*args, **kwargs)

        def launch(*packet):
            launches.append(packet)
            return compiled(*packet)

        return launch

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary FP8 must not compile Frost")

    monkeypatch.setattr(prims_ts, "_make_fp8_attention", record)
    monkeypatch.setattr(prims_ts, "_compile", record_compile)
    monkeypatch.setattr(frost, "_compile", forbidden)
    compile_kernel.cache_clear()
    args = make_fp8_case(1, compressed)
    out, lse, buffers = decode_case(*args, backend="prims-ts")
    check_fp8(out, lse, fp8_oracle(*args))
    assert len(recorded) == len(launches) == 1
    attention, packet = recorded[0], launches[0]
    assert len(packet) == 11 and packet[0].data_ptr() == args[0].data_ptr()
    assert packet[0].dtype == packet[1].dtype == torch.float8_e4m3fn
    assert packet[1].data_ptr() == args[1].data_ptr()
    assert packet[3] is buffers.indices
    assert packet[4] is buffers.topk_lens and packet[5] is buffers.seq_lens
    assert packet[6] is args[5]
    assert packet[8].data_ptr() == lse.data_ptr() and packet[8].shape == (1, 64)
    bf16_buffers = [
        x for x in packet if isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16
    ]
    assert len(bf16_buffers) == 1 and bf16_buffers[0].data_ptr() == out.data_ptr()
    assert packet[-2:] == (float(torch.tensor(512**-0.5)), 1.0)
    cfg = attention._make_config()
    assert cfg.qkv_dtype == "e4m3" and cfg.is_fp8_qkv()
    assert (
        cfg.tile_size_q,
        cfg.tile_size_kv,
        cfg.head_dim_per_stage_kv,
        cfg.kv_stages,
    ) == (64, 128, 128, 8)
    assert cfg.sparse_direct and cfg.sparse_static_scales and cfg.sparse_reuse_kv
    assert not cfg.sparse_request_kv_scale and cfg.fuse_sparse_epilogue
    assert cfg.use_persistent_scheduler == 0 and cfg.num_ctas_per_seq_kv == 1
    assert cfg.num_insts_kv == 1 and cfg.load_num_warps == 4
    assert attention.finalize_output and attention.source_state_store is None
    assert attention.direct_sparse_capacities == (128, compressed)
    assert type(attention.kv_producer) is MixedFp8KvProducer
    assert "shared_pertensor_joint" in attention.kv_producer.compile_signature()[0]
    assert not hasattr(buffers.views, "main_scale") and not hasattr(
        buffers.views, "part_s"
    )
    # Numeric BMM values are forwarded at runtime, not specialization keys.
    decode_case(
        *args, buffers=buffers, backend="prims-ts", bmm1_scale=0.125, bmm2_scale=-2.0
    )
    assert len(recorded) == 1 and launches[-1][-2:] == (0.125, -2.0)
    check_fp8(out, lse, fp8_oracle(*args, bmm1_scale=0.125, bmm2_scale=-2.0))


@pytest.mark.parametrize("native_fp4", [False, True])
def test_fp8_main_common_units_rounding_saturation_underflow_changed_graph(
    native_fp4, monkeypatch
):
    gate("prims-ts")
    import cutlass
    from flashinfer.experimental.deepseek_v41 import prims_ts

    if native_fp4 and (cutlass.CUDA_VERSION.major, cutlass.CUDA_VERSION.minor) < (
        13,
        2,
    ):
        pytest.skip("native BF16-register conversion requires bundled CUDA >= 13.2")
    compile_kernel = prims_ts._compile

    def compile_branch(*args, **kwargs):
        return compile_kernel(*args, **kwargs, native_fp4=native_fp4)

    monkeypatch.setattr(prims_ts, "_compile", compile_branch)
    q, swa, _, wi, ci, sink = make_fp8_case(2, 192)
    wi.fill_(-1)
    swa.view(torch.uint8).fill_(127)
    sink.fill_(-math.inf)
    data = torch.empty((576, 256), device=q.device, dtype=torch.uint8)
    scales = torch.empty((576, 32), device=q.device, dtype=torch.uint8)
    main = torch.empty((9, 64, 1, 288), device=q.device, dtype=torch.uint8)
    query = torch.zeros(q.shape, device=q.device)
    args = q, swa, main, wi, ci, sink
    # Exact direct staging witnesses: halfway tie, positive/negative saturation,
    # signed subnormal/zero, valid zero and empty after live requests. The last
    # selected row crosses BK128; unrelated poisoned pages cannot set a scale.
    cases = [
        (3, 0x41, 3.5, (129, 127)),  # 1.5 * 2.25 = 3.375 -> 3.5
        (7, 0x7E, 448.0, (127, 129)),  # 6 * 448 -> saturation
        (15, 0x7E, -448.0, (129, 64)),
        (1, 0x01, 0.0, (129, 129)),  # halfway to the first subnormal
        (9, 0x03, -(2**-8), (64, 127)),  # -0.5 * 3*2^-9 -> -2*2^-9
        (0, 0xFE, -0.0, (127, 0)),
        (8, 0xFE, 0.0, (0, 0)),
    ]

    def update(case, step):
        code, scale_code, _, counts = case
        data.fill_(0x77)
        scales.fill_(0x7F)
        scales[-1].fill_(0x7E)
        ci.copy_(
            torch.tensor(
                [-1, 0x7FFFFFFF, 576, -17], device=q.device, dtype=torch.int32
            ).repeat(2, 1, 48)
        )
        for request, count in enumerate(counts):
            if not count:
                continue
            rows = (
                torch.arange(count, device=q.device, dtype=torch.int32) * 197
                + 63
                + step * 11
            ) % 256 + request * 256
            rows[3] = rows[2]
            ci[request, 0, :count] = rows
            data[rows] = 0
            scales[rows] = 0x38
            data[rows, :8] = code | (code << 4)
            scales[rows, 0] = scale_code
            data[rows, 8:16] = 0x22  # A separate exact-one V group.
            scales[rows, 1] = 0x38
        main.copy_(pack(data, scales, 288))
        query.zero_()
        query[..., 0] = -1.0 if step % 2 else 1.0
        q.copy_(query.to(q.dtype))

    update(cases[0], 0)
    out, lse, buffers = decode_case(
        *args, backend="prims-ts", bmm1_scale=0.125, bmm2_scale=2.0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_case(
            *args, buffers=buffers, backend="prims-ts", bmm1_scale=0.125, bmm2_scale=2.0
        )
    for step, case in enumerate(cases):
        update(case, step)
        _, _, value, counts = case
        buffers.workspace.fill_(255)
        out.fill_(math.nan)
        graph.replay()
        check_fp8(out, lse, fp8_oracle(*args, bmm1_scale=0.125, bmm2_scale=2.0))
        staged, valid = fp8_main_reference(main, ci)
        expected_lse = torch.full(
            (2, 64), -math.inf, device=q.device, dtype=torch.float64
        )
        for request, count in enumerate(counts):
            if not count:
                continue
            torch.testing.assert_close(
                staged[request, 0, 0].float(),
                torch.tensor(value, device=q.device),
                rtol=0,
                atol=0,
            )
            # Keep the original componentwise probability-rounding gate; only
            # the represented K and V contract changes to common cache units.
            expected_lse[request] = (
                math.log(count) + query[request, 0, :, 0].double() * value * 0.125
            )
        torch.testing.assert_close(
            lse[..., 0].double(), expected_lse, rtol=0, atol=1e-5
        )
        torch.testing.assert_close(
            valid.sum(-1).int(),
            torch.tensor(counts, device=q.device, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        valid_swa = (wi >= 0) & (wi != 0x7FFFFFFF) & (wi < swa.shape[0] * 64)
        assert not valid_swa.any()
        assert not hasattr(buffers.views, "main_scale")


def test_fp8_swa_fp4_conversion_fallback(monkeypatch):
    gate("prims-ts")
    from flashinfer.experimental.deepseek_v41 import prims_ts

    compile_kernel = prims_ts._compile

    def fallback(*args, **kwargs):
        return compile_kernel(*args, **kwargs, native_fp4=False)

    monkeypatch.setattr(prims_ts, "_compile", fallback)
    args = make_fp8_case(4, 512)
    args[3][0].fill_(-1)
    out, lse, _ = decode_case(*args, backend="prims-ts")
    check_fp8(out, lse, fp8_oracle(*args))


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
    out, lse, buffers = decode_case(q, swa, main, wi, ci, sink, backend="prims-ts")
    assert not hasattr(buffers.views, "part_s") and not hasattr(buffers.views, "part_c")
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
    out, lse, buffers = decode_case(*args, backend="prims-ts")
    check_fp8(out, lse, fp8_oracle(*args))
    for pool, indices in ((args[1], args[3]), (args[2], args[4])):
        valid = (
            (indices >= 0) & (indices != 0x7FFFFFFF) & (indices < pool.shape[0] * 64)
        )
        assert not valid.any()
    with pytest.raises(ValueError, match="Q must"):
        decode_case(*args)
    with pytest.raises(ValueError, match="Prims-TS decode requires E4M3 Q"):
        decode_case(args[0].bfloat16(), *args[1:], backend="prims-ts")
    legacy = make_case(2, 64)
    with pytest.raises(ValueError, match="swa_kv_cache"):
        decode_case(args[0], legacy[1], *args[2:], backend="prims-ts")
    with pytest.raises(ValueError, match="Prims-TS decode requires E4M3 Q"):
        decode_case(*legacy, backend="prims-ts")

    prims_ts._compile.cache_clear()
    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(graph),
        pytest.raises(ValueError, match="before CUDA Graph capture"),
    ):
        buffers.call(args[0], args[1], args[2], args[5])


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
    out, lse, _ = decode_case(*actual, backend="prims-ts")
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
    out, lse, buffers = decode_case(*args, backend="prims-ts")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        buffers.call(q, swa, main, sink)
    for step, query_level in enumerate((0.5, -0.5, 0.125)):
        q.copy_(torch.full(q.shape, query_level, device=q.device).to(q.dtype))
        if step == 2:
            data.fill_(0x66)
            main.copy_(pack(data, scales, 288))
        buffers.workspace.fill_(255)
        out.fill_(math.nan)
        graph.replay()
        check_fp8(out, lse, fp8_oracle(*args))
        if step == 0:
            staged, valid = fp8_main_reference(main, ci)
            logits = (
                torch.einsum("bhd,bkd->bhk", q[:, 0].double(), staged.double())
                * 512**-0.5
            )
            logits.masked_fill_(~valid[:, None], -math.inf)
            relative = torch.exp(logits[0] - logits[0].max(-1).values[:, None])
            # Positive, nonuniform main probabilities really fall below E4M3's
            # scaled half-subnormal threshold; this is not only KV underflow.
            assert ((relative > 0) & (relative < 2**-10 / 448)).any()


@pytest.mark.parametrize("compressed", [0, 64, 192, 512])
@pytest.mark.parametrize("has_sink", [False, True])
def test_fp8_joint_canonical_single_kernel_graph(compressed, has_sink, tmp_path):
    """Inspect real captured launch nodes, not Python calls or FFI invocations."""
    import cuda.bindings.driver as cuda

    gate("prims-ts")
    q, swa, main, wi, ci, sink = make_fp8_case(2, compressed)
    if not has_sink:
        sink = None
    buffers = DecodeBuffers.allocate(q, compressed, "prims-ts")
    buffers.indices[:, :128].copy_(wi[:, 0])
    if ci is not None:
        buffers.indices[:, 128:].copy_(ci[:, 0])

    def canonical():
        return deepseek_v41_decode(
            q,
            swa,
            buffers.workspace,
            buffers.indices,
            main,
            buffers.topk_lens,
            buffers.seq_lens,
            buffers.out,
            bmm1_scale=0.125,
            bmm2_scale=2.0,
            sinks=sink,
            kv_layout="NHD",
            backend="prims-ts",
        )

    assert canonical() is buffers.out
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        assert canonical() is buffers.out
    graph.instantiate()
    dot = tmp_path / "joint-native.dot"
    graph.debug_dump(str(dot))
    topology = dot.read_text()

    def checked(result):
        assert result[0] == cuda.CUresult.CUDA_SUCCESS, (result, topology)
        return result[1:]

    # Query actual graph handles/types/edges. DOT identifiers are opaque and
    # have changed across CUDA versions; keep that file only as a diagnostic.
    raw = cuda.CUgraph(graph.raw_cuda_graph())
    _, count = checked(cuda.cuGraphGetNodes(raw))
    assert count == 1, topology
    nodes, count = checked(cuda.cuGraphGetNodes(raw, count))
    assert len(nodes) == count == 1, topology
    names = {}
    for node in nodes:
        (kind,) = checked(cuda.cuGraphNodeGetType(node))
        assert kind == cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL, topology
        (params,) = checked(cuda.cuGraphKernelNodeGetParams(node))
        (name,) = checked(cuda.cuFuncGetName(params.func))
        names[int(node)] = name.decode()
    prepare = [node for node, name in names.items() if "_prepare" in name]
    final = [node for node, name in names.items() if "dense_kernel" in name]
    assert not prepare and len(final) == 1, names
    result = checked(cuda.cuGraphGetEdges(raw))
    assert result[-1] == 0, topology
    assert (
        not {"part_s", "part_c", "lse_s", "lse_c", "main_scale"}
        & vars(buffers.views).keys()
    )
    assert all(
        value is None
        for name, value in vars(buffers.views).items()
        if name not in {"lse", "dummy_main"}
    )
    buffers.workspace.fill_(255)
    buffers.out.fill_(math.nan)
    graph.replay()
    disabled_sink = (
        torch.full((64,), -math.inf, device=q.device) if sink is None else sink
    )
    check_fp8(
        buffers.out,
        buffers.lse,
        fp8_oracle(
            q, swa, main, wi, ci, disabled_sink, bmm1_scale=0.125, bmm2_scale=2.0
        ),
    )


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
@pytest.mark.parametrize("compressed", [0, 64, 192, 512])
def test_live_combined_metadata_prefix_replay(backend, compressed):
    """Replay live B>1 tables/lengths, including left-prefix and clamp extremes."""
    gate(backend)
    case = make_fp8_case if backend == "prims-ts" else make_case
    q, swa, main, _, _, sink = case(4, compressed)
    if backend == "prims-ts":
        q.view(torch.uint8).zero_()
    else:
        q.zero_()
    buffers = DecodeBuffers.allocate(q, compressed, backend)
    if compressed:
        assert not buffers.indices[:, :128].is_contiguous()
        assert not buffers.indices[:, 128:].is_contiguous()
    column = torch.arange(128 + compressed, device=q.device, dtype=torch.int32)[None, :]
    request = torch.arange(4, device=q.device, dtype=torch.int32)[:, None]
    swa_columns = column[:, :128]
    main_columns = column[:, 128:] - 128
    limits = [
        (
            (0, 1, 127, 128),
            (128, 128 + compressed // 2, 128 + compressed, 128 + compressed + 1024),
        ),
        ((1, 129, -(2**31), 2**31 - 1), (-(2**31), 127, 128, 2**31 - 1)),
        (
            (2**31 - 1, 0, 128, -1),
            (129, 128 + compressed, 128 + compressed // 2, 128 + compressed),
        ),
        ((0, 0, 0, 0), (-(2**31), -(2**31), -(2**31), -(2**31))),
        ((128, 128, 128, 128), (128 + compressed,) * 4),
    ]

    def update(step):
        indices = (column * 197 + request * 61 + step * 29).contiguous()
        indices[:, :128] %= swa.shape[0] * 64
        if compressed:
            indices[:, 128:] %= main.shape[0] * 64
        for offset, capacity, width in (
            (0, swa.shape[0] * 64, 128),
            (128, 0 if main is None else main.shape[0] * 64, compressed),
        ):
            if not width:
                continue
            indices[:, offset + 2] = indices[:, offset + 3]
            for position, value in zip(
                (0, 1, 63, 64, 127, 128, width - 1),
                (-1, -(2**31), 2**31 - 1, capacity, capacity + 17, -1, 2**31 - 1),
                strict=True,
            ):
                if position < width:
                    indices[:, offset + position] = value
        buffers.indices.copy_(indices)
        seq, topk = limits[step]
        buffers.seq_lens.copy_(torch.tensor(seq, device=q.device, dtype=torch.int32))
        buffers.topk_lens.copy_(torch.tensor(topk, device=q.device, dtype=torch.int32))
        sink.fill_(0.125 + step * 0.25)

    def canonical():
        return deepseek_v41_decode(
            q,
            swa,
            buffers.workspace,
            buffers.indices,
            main,
            buffers.topk_lens,
            buffers.seq_lens,
            buffers.out,
            bmm1_scale=512**-0.5,
            bmm2_scale=1.0,
            sinks=sink,
            kv_layout="NHD",
            backend=backend,
        )

    update(0)
    assert canonical() is buffers.out
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert canonical() is buffers.out
    for step in range(len(limits)):
        update(step)
        buffers.workspace.fill_(255)
        buffers.out.fill_(math.nan)
        graph.replay()
        swa_ids = buffers.indices[:, :128]
        keep_s = (
            (swa_columns < buffers.seq_lens.clamp(0, 128)[:, None])
            & (swa_ids >= 0)
            & (swa_ids < swa.shape[0] * 64)
            & (swa_ids != 0x7FFFFFFF)
        )
        selected_swa = torch.where(keep_s, swa_ids, -1)
        counts = keep_s.sum(-1)
        if backend == "cute-dsl":
            torch.testing.assert_close(
                buffers.views.selected_swa, selected_swa, rtol=0, atol=0
            )
            torch.testing.assert_close(
                buffers.views.swa_counts, keep_s.sum(-1).int(), rtol=0, atol=0
            )
        wi = selected_swa[:, None].contiguous()
        if compressed:
            main_ids = buffers.indices[:, 128:]
            main_visible = buffers.topk_lens.clamp(128, 128 + compressed) - 128
            keep_c = (
                (main_columns < main_visible[:, None])
                & (main_ids >= 0)
                & (main_ids < main.shape[0] * 64)
                & (main_ids != 0x7FFFFFFF)
            )
            selected_main = torch.where(keep_c, main_ids, -1)
            counts += keep_c.sum(-1)
            if backend == "cute-dsl":
                torch.testing.assert_close(
                    buffers.views.selected_main, selected_main, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    buffers.views.main_counts, keep_c.sum(-1).int(), rtol=0, atol=0
                )
            ci = selected_main[:, None].contiguous()
        else:
            ci = None
            if backend == "cute-dsl":
                assert (buffers.views.selected_main == -1).all() and (
                    buffers.views.main_counts == 0
                ).all()
        if backend == "cute-dsl":
            assert (buffers.views.swa_lengths == 128).all() and (
                buffers.views.main_lengths == compressed
            ).all()
            assert (buffers.views.total_lengths == 128 + compressed).all()
        torch.testing.assert_close(
            buffers.lse[..., 0].double(),
            counts.double().log()[:, None].expand(-1, 64),
            rtol=0,
            atol=1e-5,
        )
        args = q, swa, main, wi, ci, sink
        if backend == "prims-ts":
            check_fp8(buffers.out, buffers.lse, fp8_oracle(*args))
        else:
            check(args, buffers.out, buffers.lse)


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
@pytest.mark.parametrize("compressed", [0, 192])
def test_shared_bmm_scales_and_layout_parity(backend, compressed):
    """Nonunit/negative/zero output factors apply once, including captured HND."""
    gate(backend)
    case = make_fp8_case if backend == "prims-ts" else make_case
    args = case(2, compressed)
    q, swa, main, wi, ci, sink = args
    nhd = DecodeBuffers.allocate(q, compressed, backend)
    hnd = DecodeBuffers.allocate(q, compressed, backend)
    for buffers in (nhd, hnd):
        buffers.indices[:, :128].copy_(wi[:, 0])
        if ci is not None:
            buffers.indices[:, 128:].copy_(ci[:, 0])
    hnd_swa = swa.view(swa.shape[0], 1, 64, swa.shape[-1])
    hnd_main = None if main is None else main.view(main.shape[0], 1, 64, 288)
    assert hnd_swa.data_ptr() == swa.data_ptr()
    for bmm1, bmm2 in ((0.125, 2.0), (0.25, -0.5), (0.0625, 0.0)):
        scales = dict(bmm1_scale=bmm1, bmm2_scale=bmm2)
        assert nhd.call(q, swa, main, sink, **scales) is nhd.out

        # Omit kv_layout: this actually exercises the default HND convention.
        def canonical():
            return deepseek_v41_decode(
                q,
                hnd_swa,
                hnd.workspace,
                hnd.indices,
                hnd_main,
                hnd.topk_lens,
                hnd.seq_lens,
                hnd.out,
                bmm1_scale=bmm1,
                bmm2_scale=bmm2,
                sinks=sink,
                backend=backend,
            )

        assert canonical() is hnd.out
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert canonical() is hnd.out
        hnd.workspace.fill_(255)
        hnd.out.fill_(math.nan)
        graph.replay()
        assert torch.equal(hnd.out, nhd.out) and torch.equal(hnd.lse, nhd.lse)
        if backend == "prims-ts":
            check_fp8(hnd.out, hnd.lse, fp8_oracle(*args, **scales))
        else:
            check(args, hnd.out, hnd.lse, **scales)


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
@pytest.mark.parametrize("bmm1_scale", [2e-38, 3e38])
def test_canonical_positive_finite_bmm1_extremes(backend, bmm1_scale):
    """Zero Q and one real unit row remain O=1/LSE=0 at FP32 scale edges.

    Tiny factors must not revive finite mask sentinels. Large factors must not
    overflow a precomputed scale*log2(e) before multiplying an exact zero dot.
    Both factors are accepted positive finite FP32, not out-of-contract inputs.
    """
    gate(backend)
    native = backend == "prims-ts"
    q = torch.zeros((1, 1, 64, 512), device="cuda", dtype=torch.bfloat16)
    if native:
        q = q.to(torch.float8_e4m3fn)
        swa = torch.ones((1, 1, 64, 512), device=q.device).to(q.dtype)
    else:
        swa = pack(
            torch.ones((64, 512), device=q.device)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8),
            torch.full((64, 16), 127, device=q.device, dtype=torch.uint8),
            528,
        ).view(1, 1, 64, 528)
    buffers = DecodeBuffers.allocate(q, 0, backend)
    buffers.indices.zero_()
    buffers.seq_lens.fill_(1)

    def canonical():
        return deepseek_v41_decode(
            q,
            swa,
            buffers.workspace,
            buffers.indices,
            None,
            buffers.topk_lens,
            buffers.seq_lens,
            buffers.out,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            backend=backend,
        )

    assert canonical() is buffers.out
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert canonical() is buffers.out
    for visible in (1, 0, 1):
        buffers.seq_lens.fill_(visible)
        buffers.workspace.fill_(255)
        buffers.out.fill_(math.nan)
        graph.replay()
        torch.testing.assert_close(
            buffers.out, torch.full_like(buffers.out, visible), rtol=0, atol=0
        )
        if visible:
            torch.testing.assert_close(
                buffers.lse, torch.zeros_like(buffers.lse), rtol=0, atol=1e-5
            )
        else:
            assert torch.isneginf(buffers.lse).all()
        keep = (
            (
                torch.arange(128, device=q.device)[None, :]
                < buffers.seq_lens.clamp(0, 128)[:, None]
            )
            & (buffers.indices >= 0)
            & (buffers.indices != 0x7FFFFFFF)
            & (buffers.indices < swa.shape[0] * 64)
        )
        assert (keep.sum(-1) == visible).all()
        if not native:
            assert (buffers.views.swa_counts == visible).all() and (
                buffers.views.main_counts == 0
            ).all()


def test_frost_tied_overflowing_logits_across_splits():
    """Tied raw scores merge finite O even when their FP32 natural LSE is +inf."""
    gate()
    from flashinfer.experimental.deepseek_v41._workspace import split_kv

    assert split_kv(1, 0) == 2
    q = torch.zeros((1, 1, 64, 512), device="cuda", dtype=torch.bfloat16)
    q[..., 0] = 1
    values = torch.ones((128, 512), device=q.device)
    values[:, 0] = 2
    values[64:, 1] = 3
    scales = torch.full((128, 16), 127, device=q.device, dtype=torch.uint8)
    swa = pack(values.to(torch.float8_e4m3fn).view(torch.uint8), scales, 528)
    buffers = DecodeBuffers.allocate(q, 0, "cute-dsl")
    buffers.indices.copy_(torch.arange(128, device=q.device, dtype=torch.int32)[None])

    def canonical():
        return deepseek_v41_decode(
            q,
            swa,
            buffers.workspace,
            buffers.indices,
            None,
            buffers.topk_lens,
            buffers.seq_lens,
            buffers.out,
            bmm1_scale=3e38,
            bmm2_scale=1.0,
            kv_layout="NHD",
            backend="cute-dsl",
        )

    assert canonical() is buffers.out
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert canonical() is buffers.out
    for step, (visible, mean) in enumerate(((128, 2), (64, 1), (0, 0), (128, 1))):
        if step == 3:
            values[:64, 1] = -3
            values[64:, 1] = 5
            swa.copy_(
                pack(values.to(torch.float8_e4m3fn).view(torch.uint8), scales, 528)
            )
        buffers.seq_lens.fill_(visible)
        buffers.workspace.fill_(255)
        buffers.out.fill_(math.nan)
        graph.replay()
        expected = torch.full_like(buffers.out, 1 if visible else 0)
        if visible:
            expected[..., 0] = 2
            expected[..., 1] = mean
            assert torch.isposinf(buffers.lse).all()
        else:
            assert torch.isneginf(buffers.lse).all()
        torch.testing.assert_close(buffers.out, expected, rtol=0, atol=0)
        assert (buffers.views.swa_counts == visible).all()
        assert (buffers.views.main_counts == 0).all()


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
def test_canonical_near_limit_sink_cancellation(backend):
    """Natural-unit sink differences survive a large but finite tied KV anchor.

    The exact binary factors make the FP32 LSE anchor representable. Only its
    log(128) correction rounds away; output still needs the full 128/(128+1)
    denominator. One FP32 ULP either side makes the sink irrelevant/dominant.
    An FP64 logsumexp-minus-normalizer reference would also lose log(count)
    here, so these expectations are analytic rather than a relaxed old gate.
    """
    gate(backend)
    q = torch.zeros((1, 1, 64, 512), device="cuda", dtype=torch.bfloat16)
    q[..., 0] = 1
    values = torch.ones((128, 512), device=q.device)
    values[:, 0] = 1.5
    values[64:, 1] = 3
    if backend == "prims-ts":
        q = q.to(torch.float8_e4m3fn)
        swa = values.to(q.dtype).view(2, 64, 1, 512)
    else:
        swa = pack(
            values.to(torch.float8_e4m3fn).view(torch.uint8),
            torch.full((128, 16), 127, device=q.device, dtype=torch.uint8),
            528,
        )
    bmm1 = 2.0**127
    anchor, ulp = 1.5 * bmm1, 2.0**104
    sinks = torch.full((64,), anchor, device=q.device)
    buffers = DecodeBuffers.allocate(q, 0, backend)
    buffers.indices.copy_(torch.arange(128, device=q.device, dtype=torch.int32)[None])
    mean = torch.ones(buffers.out.shape, device=q.device, dtype=torch.float64)
    mean[..., 0] = 1.5
    mean[..., 1] = 2

    def canonical():
        return deepseek_v41_decode(
            q,
            swa,
            buffers.workspace,
            buffers.indices,
            None,
            buffers.topk_lens,
            buffers.seq_lens,
            buffers.out,
            bmm1_scale=bmm1,
            bmm2_scale=1.0,
            sinks=sinks,
            kv_layout="NHD",
            backend=backend,
        )

    assert canonical() is buffers.out
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert canonical() is buffers.out
    for delta, mass in ((0, 128 / 129), (-ulp, 1), (ulp, 0), (0, 128 / 129)):
        sinks.fill_(anchor + delta)
        buffers.workspace.fill_(255)
        buffers.out.fill_(math.nan)
        graph.replay()
        torch.testing.assert_close(
            buffers.out, (mean * mass).bfloat16(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            buffers.lse, torch.full_like(buffers.lse, anchor), rtol=0, atol=1e-5
        )
        keep = (
            (
                torch.arange(128, device=q.device)[None, :]
                < buffers.seq_lens.clamp(0, 128)[:, None]
            )
            & (buffers.indices >= 0)
            & (buffers.indices != 0x7FFFFFFF)
            & (buffers.indices < swa.shape[0] * 64)
        )
        assert (keep.sum(-1) == 128).all()
        if backend == "cute-dsl":
            assert (buffers.views.swa_counts == 128).all()
            assert (buffers.views.main_counts == 0).all()
