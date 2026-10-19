# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Benchmark/example contracts without importing Torch, CUDA, CuTe or FlashInfer.

Run with --confcutdir=tests/experimental to exclude root CUDA collection probes.
AST extraction executes parser/fixture/gate helpers with stdlib mocks and checks
common-unit reference arithmetic with a small stdlib-only numerical tensor.
These host contracts are not GPU/JIT or full decode numerical qualification.
"""

import argparse
import ast
import builtins
import math
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK = _ROOT / "benchmarks/bench_deepseek_v41_decode.py"
_EXAMPLE = _ROOT / "examples/deepseek_v41/decode.py"
_GPU_TEST = Path(__file__).with_name("test_deepseek_v41_decode.py")


@pytest.fixture(autouse=True)
def no_accelerator_imports(monkeypatch):
    original_import = builtins.__import__
    attempts = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "cuda",
            "cutlass",
            "flashinfer",
            "triton",
            "nvidia",
        }:
            attempts.append(name)
            raise AssertionError(f"accelerator import attempted: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    yield
    assert not attempts


def definition(path, name):
    return next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def extract(path, name, **bindings):
    namespace = {"__doc__": ast.get_docstring(ast.parse(path.read_text())), **bindings}
    function = definition(path, name)
    function.decorator_list = []
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def e4m3_value(code):
    exponent, mantissa = (code >> 3) & 15, code & 7
    assert (code & 127) != 127
    magnitude = (
        mantissa * 2**-9 if exponent == 0 else (8 + mantissa) * 2 ** (exponent - 10)
    )
    return math.copysign(magnitude, -1 if code & 128 else 1)


def e4m3_rn_satfinite(value):
    # Independent finite code enumeration; the code's low bit implements RTNE.
    code = min(
        range(127),
        key=lambda code: (abs(e4m3_value(code) - abs(value)), code & 1),
    )
    return math.copysign(e4m3_value(code), value)


class CommonUnitTensor:
    """Only independent FP32/clamp/E4M3/FP64 staging; no reductions or divides.

    Samples have shape [B,K,1]. A larger value in another row or request must
    never change a sample's represented cache units.
    """

    def __init__(self, rows, dtype="float64", staged=None):
        self.rows = tuple(tuple(row) for row in rows)
        assert self.rows and all(len(row) == len(self.rows[0]) for row in self.rows)
        self.shape = (len(self.rows), len(self.rows[0]), 1)
        self.dtype = dtype
        self.staged = [] if staged is None else staged

    def map(self, operation, dtype=None):
        return CommonUnitTensor(
            [[operation(value) for value in row] for row in self.rows],
            dtype or self.dtype,
            self.staged,
        )

    def float(self):
        return self.map(float32, "float32")

    def double(self):
        return self.map(float, "float64")

    def clamp(self, low, high):
        assert (low, high) == (-448, 448)
        return self.map(lambda value: min(max(value, low), high))

    def to(self, dtype):
        assert dtype == "float8_e4m3fn" and self.dtype == "float32"
        self.staged.append((self.dtype, self.rows))
        return self.map(e4m3_rn_satfinite, dtype)


def numeric_common_unit_main(rows):
    decoded = CommonUnitTensor(rows)
    stage = extract(
        _BENCHMARK,
        "common_unit_main",
        torch=SimpleNamespace(float8_e4m3fn="float8_e4m3fn"),
    )
    represented = stage(decoded)
    assert len(decoded.staged) == 1
    return represented, decoded.staged[0]


@pytest.mark.parametrize("backend", [None, "cute-dsl", "prims-ts"])
def test_benchmark_parser_defaults_and_explicit_backend(backend):
    parser = extract(_BENCHMARK, "build_parser", argparse=argparse, Path=Path)()
    argv = ["--output", "results.json"]
    if backend:
        argv += ["--backends", backend]
    opts = parser.parse_args(argv)
    assert opts.backends == [backend or "cute-dsl"]
    assert opts.output == Path("results.json")
    assert opts.context == 32768 and opts.seed == 7003
    assert opts.batches == "1,4,16,32,64,128,256" and not opts.flashmla
    assert not hasattr(opts, "prims_ts_mode")


def test_benchmark_parser_multiple_arms_and_required_output():
    parser = extract(_BENCHMARK, "build_parser", argparse=argparse, Path=Path)()
    opts = parser.parse_args(
        ["--output", "results.json", "--backends", "cute-dsl", "prims-ts", "--flashmla"]
    )
    assert opts.backends == ["cute-dsl", "prims-ts"] and opts.flashmla
    with pytest.raises(SystemExit) as error:
        parser.parse_args([])
    assert error.value.code == 2


@pytest.mark.parametrize("path", [_BENCHMARK, _EXAMPLE])
@pytest.mark.parametrize("backend", [None, "cute-dsl", "prims-ts"])
def test_removed_prims_ts_mode_flag_is_rejected(path, backend):
    parser = extract(path, "build_parser", argparse=argparse, Path=Path)()
    argv = ["--output", "results.json"] if path == _BENCHMARK else []
    if backend:
        argv += ["--backends" if path == _BENCHMARK else "--backend", backend]
    for removed_args in (
        ["--prims-ts-mode", "fp8"],
        ["--prims-ts-mode", "mxfp8"],
        ["--prims-ts-mode=fp8"],
    ):
        with pytest.raises(SystemExit) as error:
            parser.parse_args(argv + removed_args)
        assert error.value.code == 2


@pytest.mark.parametrize("backend", [None, "cute-dsl", "prims-ts"])
def test_example_parser_defaults_and_explicit_backend(backend):
    parser = extract(_EXAMPLE, "build_parser", argparse=argparse)()
    opts = parser.parse_args(["--backend", backend] if backend else [])
    assert opts.backend == (backend or "cute-dsl")
    assert not hasattr(opts, "prims_ts_mode")


@pytest.mark.parametrize(
    "backends", [["cute-dsl"], ["prims-ts"], ["cute-dsl", "prims-ts"]]
)
@pytest.mark.parametrize("flashmla", [False, True])
def test_benchmark_fixture_common_source_and_shared_global_pool(backends, flashmla):
    torch = MagicMock(name="torch_stub")
    q_source, w_source, values, sink = (
        MagicMock(name=name) for name in ("q_source", "w_source", "values", "sink")
    )
    main, ci = MagicMock(name="shared_main"), MagicMock(name="shared_ci")
    torch.randn.side_effect = [q_source, w_source, values, sink]
    torch.empty.side_effect = [main, ci]
    caches = {format: object() for format in ("main_kv_fp4", "swa_mxfp8", "swa_fp8")}
    quantize = Mock(
        side_effect=lambda source, *, format, kv_layout: caches[format]
        if kv_layout == "NHD"
        else None
    )
    fixture = extract(
        _BENCHMARK,
        "fixture",
        torch=torch,
        deepseek_v41_quantize_cache=quantize,
    )
    native = "prims-ts" in backends
    arms = fixture(1, 512, 7003, backends, flashmla)
    assert set(arms) == set(backends) | ({"flashmla"} if flashmla else set())
    torch.manual_seed.assert_called_once_with(7004)
    assert torch.randn.call_args_list == [
        call(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16),
        call(128, 512, device="cuda", dtype=torch.bfloat16),
        call(512, 512, device="cuda", dtype=torch.bfloat16),
        call(64, device="cuda"),
    ]
    # A single global quantization/allocation serves every arm, including FlashMLA.
    quantize.assert_any_call(values, format="main_kv_fp4", kv_layout="NHD")
    assert torch.empty.call_count == 2
    global_calls = [
        c for c in quantize.call_args_list if c.kwargs["format"] == "main_kv_fp4"
    ]
    assert len(global_calls) == 1
    shared = next(iter(arms.values()))[2:]
    assert shared[0] is main and shared[2] is ci
    for args in arms.values():
        assert all(
            actual is expected
            for actual, expected in zip(args[2:], shared, strict=True)
        )
    if native:
        q_source.to.assert_called_once_with(torch.float8_e4m3fn)
        assert arms["prims-ts"][0] is q_source.to.return_value
        assert arms["prims-ts"][1] is caches["swa_fp8"]
        quantize.assert_any_call(w_source, format="swa_fp8", kv_layout="NHD")
    else:
        q_source.to.assert_not_called()
    legacy_names = [name for name in arms if name != "prims-ts"]
    for name in legacy_names:
        assert arms[name][0] is q_source
        assert arms[name][1] is caches["swa_mxfp8"]
        assert arms[name] is arms[legacy_names[0]]
        quantize.assert_any_call(w_source, format="swa_mxfp8", kv_layout="NHD")
    formats = [c.kwargs["format"] for c in quantize.call_args_list]
    assert set(formats) == (
        {"main_kv_fp4"}
        | ({"swa_fp8"} if native else set())
        | ({"swa_mxfp8"} if legacy_names else set())
    )
    assert len(formats) == len(set(formats))


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
def test_example_representation_is_explicit_and_backend_scoped(backend):
    main = definition(_EXAMPLE, "main")
    assignments = {
        node.targets[0].id: node.value
        for node in main.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    native = eval(
        compile(ast.Expression(assignments["native_fp8"]), str(_EXAMPLE), "eval"),
        {"opts": SimpleNamespace(backend=backend)},
    )
    assert native == (backend == "prims-ts")
    assert ast.unparse(assignments["q"]) == (
        "q_source.to(torch.float8_e4m3fn) if native_fp8 else q_source"
    )
    format = next(kw.value for kw in assignments["swa"].keywords if kw.arg == "format")
    assert ast.unparse(format) == "'swa_fp8' if native_fp8 else 'swa_mxfp8'"


def test_native_oracle_attention_and_bound_match_independent_gpu_reference():
    def core(function):
        start = next(
            i
            for i, node in enumerate(function.body)
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "(kv, valid)"
        )
        return [ast.dump(node) for node in function.body[start:]]

    assert core(definition(_BENCHMARK, "fp8_oracle")) == core(
        definition(_GPU_TEST, "fp8_oracle")
    )
    assert [ast.dump(node) for node in definition(_BENCHMARK, "check_fp8").body] == [
        ast.dump(node) for node in definition(_GPU_TEST, "check_fp8").body
    ]


def test_native_oracle_uses_stored_values_and_only_selected_global_rows():
    oracle = definition(_BENCHMARK, "fp8_oracle")
    assignments = [node for node in ast.walk(oracle) if isinstance(node, ast.Assign)]
    gathered = [
        ast.unparse(node.value)
        for node in assignments
        if ast.unparse(node.targets[0]) == "gathered"
    ]
    assert gathered == [
        "pool.double().reshape(-1, 512)[safe]",
        "selected_main_fp8(pool, safe, valid)",
    ]
    valid = next(
        node for node in assignments if ast.unparse(node.targets[0]) == "valid"
    )
    assert ast.unparse(valid.value) == (
        "(index >= 0) & (index < pool.shape[0] * 64) & (index != 2147483647)"
    )
    # Legacy decoding retains its original BF16 values. Native main uses its
    # separate masked-byte gather and direct common-unit reference below.
    decoder = definition(_BENCHMARK, "selected")
    assert [ast.literal_eval(value) for value in decoder.args.kw_defaults] == [True]
    result = next(node.value for node in decoder.body if isinstance(node, ast.Return))
    represented = MagicMock(name="represented")
    for rounded in (False, True):
        value = eval(
            compile(ast.Expression(result), str(_BENCHMARK), "eval"),
            {"represented": represented, "round_bf16": rounded},
        )
        assert value is (represented.bfloat16().double() if rounded else represented)


def test_native_main_conversion_precedes_shared_qk_and_v_inputs():
    oracle = definition(_BENCHMARK, "fp8_oracle")
    loop = next(node for node in oracle.body if isinstance(node, ast.For))
    branch = next(
        node
        for node in loop.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not fp4"
    )
    assert [ast.unparse(node) for node in branch.orelse] == [
        "gathered = selected_main_fp8(pool, safe, valid)",
    ]
    assert ast.unparse(loop.body[-2]) == (
        "values.append(torch.where(valid[..., None], gathered, 0))"
    )
    # Duplicate and all-zero valid rows keep their attention multiplicity.
    assert ast.unparse(loop.body[-1]) == "masks.append(valid)"
    inputs = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in oracle.body
        if isinstance(node, ast.Assign)
    }
    assert inputs["logits"] == (
        "torch.einsum('bhd,bkd->bhk', q[:, 0].double(), kv) * bmm1_scale"
    )
    assert (
        inputs["expected"] == "torch.einsum('bhk,bkd->bhd', weights, kv) * bmm2_scale"
    )
    assert (
        inputs["sensitivity"]
        == "torch.einsum('bhk,bkd->bhd', weights, kv.abs()) * abs(bmm2_scale)"
    )
    assert (
        inputs["underflow"]
        == "frame_mass[..., None] * kv.abs().sum(1)[:, None, :] * abs(bmm2_scale)"
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "bfloat16"
        for function in (
            oracle,
            definition(_BENCHMARK, "selected_main_fp8"),
            definition(_BENCHMARK, "common_unit_main"),
        )
        for node in ast.walk(function)
    )


def test_selected_main_masks_packed_payload_and_scales_before_decode():
    pool, safe, valid, payload, raw_scale, masked_payload, masked_scale, data = (
        MagicMock(name=name)
        for name in (
            "pool",
            "safe",
            "valid",
            "payload",
            "raw_scale",
            "masked_payload",
            "masked_scale",
            "data",
        )
    )
    pool.shape = (4, 64, 1, 288)
    pages = pool.view.return_value
    pages.__getitem__.side_effect = [payload, raw_scale]
    torch = MagicMock(name="torch_stub")
    torch.where.side_effect = [masked_payload, masked_scale]
    torch.tensor.return_value.__getitem__.return_value = data
    represented = object()
    normalize = Mock(return_value=represented)
    gather = extract(
        _BENCHMARK, "selected_main_fp8", torch=torch, common_unit_main=normalize
    )
    assert gather(pool, safe, valid) is represented
    assert torch.where.call_args_list == [
        call(valid[..., None], payload, 0),
        call(valid[..., None], raw_scale, 0),
    ]
    # Invalid safe-row poison is zeroed as bytes, before any floating decode.
    payload.__rshift__.assert_not_called()
    raw_scale.contiguous.assert_not_called()
    masked_payload.__rshift__.assert_called_once()
    masked_scale.contiguous.return_value.view.assert_called_once_with(
        torch.float8_e4m3fn
    )
    scale = masked_scale.contiguous.return_value.view.return_value.float.return_value
    normalize.assert_called_once_with(data * scale)
    pool.double.assert_not_called()
    pool.view.assert_called_once_with(4, -1)
    assert pages.__getitem__.call_count == 2
    lut = torch.tensor.call_args.args[0]
    assert lut == [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    assert math.copysign(1, lut[8]) == -1
    assert torch.tensor.call_args.kwargs["dtype"] is torch.float32
    assignments = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in definition(_BENCHMARK, "selected_main_fp8").body
        if isinstance(node, ast.Assign)
    }
    assert assignments["(page, row)"] == "(safe // 64, safe % 64)"
    gathers = [
        ast.unparse(node.value)
        for node in definition(_BENCHMARK, "selected_main_fp8").body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Subscript)
        and ast.unparse(node.value.value) == "pages"
    ]
    assert gathers == [
        "pages[page[..., None], row[..., None] * 256 + column // 2]",
        "pages[page[..., None], 64 * 256 + row[..., None] * 32 + column // 16]",
    ]
    assert assignments["decoded"] == "data * scale"
    result = next(
        node.value
        for node in definition(_BENCHMARK, "selected_main_fp8").body
        if isinstance(node, ast.Return)
    )
    assert ast.unparse(result) == "common_unit_main(decoded)"


def test_common_unit_main_uses_decoded_payload_not_group_maximum():
    payloads = (0, 1, 1)
    scales = tuple(e4m3_value(code) for code in (0x7E, 0x40, 0xC0))
    decoded = [payload * scale for payload, scale in zip(payloads, scales, strict=True)]
    assert decoded == [0.0, 2.0, -2.0]
    represented, staged = numeric_common_unit_main([decoded])
    assert represented.dtype == "float64" and represented.rows == ((0.0, 2.0, -2.0),)
    assert staged == ("float32", ((0.0, 2.0, -2.0),))


def test_common_unit_main_materializes_fp32_then_direct_rounding_and_saturation():
    function = definition(_BENCHMARK, "common_unit_main")
    result = next(node.value for node in function.body if isinstance(node, ast.Return))
    assert (
        ast.unparse(result)
        == "decoded.float().clamp(-448, 448).to(torch.float8_e4m3fn).double()"
    )
    represented, staged = numeric_common_unit_main([[2688.0, 1.0, -1.0, 3.375]])
    assert represented.rows == ((448.0, 1.0, -1.0, 3.5),)
    assert staged == ("float32", ((448.0, 1.0, -1.0, 3.375),))
    assert not any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
        for node in ast.walk(function)
    )


@pytest.mark.parametrize(
    "value,expected", [(1.0, 1.0), (2**-10, 0.0), (3 * 2**-10, 2**-8), (224.0, 224.0)]
)
def test_common_unit_main_preserves_tiny_values_without_renormalization(
    value, expected
):
    represented, staged = numeric_common_unit_main([[value, -value]])
    assert represented.rows == ((expected, -expected),)
    assert math.copysign(1, represented.rows[0][1]) == -1
    assert staged == ("float32", ((float32(value), float32(-value)),))


@pytest.mark.parametrize("rows", [[[]], [[0.0, -0.0]]])
def test_common_unit_main_keeps_empty_or_zero_rows_without_scale_state(rows):
    represented, staged = numeric_common_unit_main(rows)
    assert represented.rows == tuple(tuple(row) for row in rows)
    assert represented.shape == (1, len(rows[0]), 1) and staged[0] == "float32"
    if rows[0]:
        assert math.copysign(1, represented.rows[0][1]) == -1


def test_common_unit_main_keeps_duplicates_zero_rows_and_request_independence():
    represented, _ = numeric_common_unit_main(
        [[0.0, -0.0], [7.0, 7.0], [3.375, 9.0], [2688.0, -2688.0]]
    )
    assert represented.rows == ((0.0, -0.0), (7.0, 7.0), (3.5, 9.0), (448.0, -448.0))
    changed, _ = numeric_common_unit_main(
        [[0.0, -0.0], [7.0, 7.0], [3.375, 9.0], [0.0, 0.0]]
    )
    assert changed.rows[:3] == represented.rows[:3] and represented.shape == (4, 2, 1)


@pytest.mark.parametrize(
    "value,expected",
    [(1.0625, 1.0), (1.1875, 1.25), (-1.0625, -1.0), (2**-10, 0.0), (2688, 448)],
)
def test_stdlib_e4m3_rn_satfinite_ties_subnormal_and_saturation(value, expected):
    assert e4m3_rn_satfinite(value) == expected


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "out_dtype",
        "lse_dtype",
        "finite",
        "bound",
        "mask",
        "lse",
        "empty_out",
        "empty_bound",
    ],
)
def test_native_gate_rejects_each_contract_violation(failure):
    out, lse, expected, expected_lse, bound = (
        MagicMock(name=name)
        for name in ("out", "lse", "expected", "expected_lse", "bound")
    )
    torch = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        isfinite=Mock(return_value=SimpleNamespace(all=lambda: failure != "finite")),
        isneginf=Mock(return_value=MagicMock(name="empty")),
        equal=Mock(
            side_effect=[
                failure != "mask",
                failure != "empty_out",
                failure != "empty_bound",
            ]
        ),
        zeros_like=Mock(),
        testing=SimpleNamespace(assert_close=Mock()),
    )
    out.dtype = "wrong" if failure == "out_dtype" else "bf16"
    lse.dtype = "wrong" if failure == "lse_dtype" else "fp32"
    error = (out[:, 0].double() - expected).abs()
    error.__le__.return_value = SimpleNamespace(all=lambda: failure != "bound")
    if failure == "lse":
        torch.testing.assert_close.side_effect = AssertionError("LSE mismatch")
    check = extract(_BENCHMARK, "check_fp8", torch=torch)
    if failure:
        with pytest.raises(AssertionError):
            check(out, lse, (expected, expected_lse, bound))
    else:
        check(out, lse, (expected, expected_lse, bound))
        assert torch.testing.assert_close.call_args.kwargs == {"rtol": 0, "atol": 1e-5}
        assert torch.equal.call_count == 3


@pytest.mark.parametrize("name", ["cute-dsl", "prims-ts", "flashmla"])
def test_arm_selects_backend_specific_oracle_and_gate(name):
    args = tuple(object() for _ in range(6))
    out, lse, expected, expected_lse, bound = (
        MagicMock(name=value)
        for value in ("out", "lse", "expected", "expected_lse", "bound")
    )
    native_reference = (expected, expected_lse, bound)
    legacy_reference = (expected, expected_lse)
    native_oracle = Mock(return_value=native_reference)
    legacy_oracle = Mock(return_value=legacy_reference)
    native_gate, legacy_gate = Mock(), Mock()
    check = extract(
        _BENCHMARK,
        "check_arm",
        torch=MagicMock(name="torch_stub"),
        fp8_oracle=native_oracle,
        legacy_reference=legacy_oracle,
        check_fp8=native_gate,
        check_legacy=legacy_gate,
    )
    stats = check(name, args, Mock(return_value=(out, lse)), 1, Path("unused.json"))
    if name == "prims-ts":
        native_oracle.assert_called_once_with(*args)
        native_gate.assert_called_once_with(out, lse, native_reference)
        legacy_oracle.assert_not_called()
        legacy_gate.assert_not_called()
        assert "max_bound_fraction" in stats
    else:
        legacy_oracle.assert_called_once_with(*args)
        legacy_gate.assert_called_once_with(out, stats)
        native_oracle.assert_not_called()
        native_gate.assert_not_called()
        assert set(stats) == {"rel_l2", "max_scaled", "lse_abs"}


@pytest.mark.parametrize("path", [_BENCHMARK, _EXAMPLE])
def test_cli_sources_have_no_compatibility_selector(path):
    source = path.read_text()
    assert "--prims-ts-mode" not in source
    assert "prims_ts_mode" not in source


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
def test_benchmark_times_canonical_public_replay_with_caller_storage(backend):
    q, swa, main, wi, ci, sink = (
        MagicMock(name=name) for name in ("q", "swa", "main", "wi", "ci", "sink")
    )
    q.shape = (2, 1, 64, 512)
    ci.shape = (2, 1, 192)
    indices, topk, seq, workspace, out, lse = (
        MagicMock(name=name)
        for name in ("indices", "topk", "seq", "workspace", "out", "lse")
    )
    workspace.numel.return_value = 4096
    torch = MagicMock(name="torch_stub")
    torch.cat.return_value = indices
    torch.full.side_effect = [topk, seq]
    torch.empty.side_effect = [workspace, out]
    decode = Mock(return_value=out)
    size = Mock(return_value=4096)
    views = Mock(return_value=SimpleNamespace(lse=lse))
    split = Mock(return_value=7)
    function = definition(_BENCHMARK, "prepare_decode")
    function.body = [
        node for node in function.body if not isinstance(node, ast.ImportFrom)
    ]
    namespace = {
        "torch": torch,
        "deepseek_v41_decode": decode,
        "deepseek_v41_decode_workspace_size": size,
        "workspace_views": views,
        "split_kv": split,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(_BENCHMARK), "exec"),
        namespace,
    )
    fn, info = namespace["prepare_decode"]((q, swa, main, wi, ci, sink), backend)
    assert not decode.called
    torch.cat.assert_called_once_with((wi[:, 0], ci[:, 0]), dim=1)
    assert torch.full.call_args_list == [
        call((2,), 320, device=q.device, dtype=torch.int32),
        call((2,), 128, device=q.device, dtype=torch.int32),
    ]
    assert torch.empty.call_args_list == [
        call(4096, device=q.device, dtype=torch.uint8),
        call(q.shape, device=q.device, dtype=torch.bfloat16),
    ]
    size.assert_called_once_with(2, 192, backend=backend)
    views.assert_called_once_with(workspace, 2, 192, backend=backend)
    assert info == {"workspace_bytes": 4096, "split": 1 if backend == "prims-ts" else 7}
    for _ in range(2):
        result, private_lse = fn()
        assert result is out and private_lse is lse
    assert (
        decode.call_args_list
        == [
            call(
                q,
                swa,
                workspace,
                indices,
                main,
                topk,
                seq,
                out,
                bmm1_scale=512**-0.5,
                bmm2_scale=1.0,
                sinks=sink,
                kv_layout="NHD",
                backend=backend,
            )
        ]
        * 2
    )
    assert (
        torch.empty.call_count == torch.full.call_count == 2
        and torch.cat.call_count == 1
    )
    main_function = definition(_BENCHMARK, "main")
    prepared = next(
        node.value
        for node in ast.walk(main_function)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "(fns[backend], buffers[backend])"
    )
    assert ast.unparse(prepared) == "prepare_decode(arm_args[backend], backend)"
    graphs = next(
        node.value
        for node in ast.walk(main_function)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "graphs"
    )
    assert ast.unparse(graphs) == "{name: graph(fn) for name, fn in fns.items()}"
    report = next(
        node.value
        for node in main_function.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "report"
    )
    metadata = {
        ast.literal_eval(key): ast.literal_eval(value)
        for key, value in zip(report.keys, report.values, strict=True)
        if ast.literal_eval(key) in {"scope", "comparison"}
    }
    assert "metadata sanitation and attention" in metadata["scope"]
    assert "selected-tile staging and joint native finalization" in metadata["scope"]
    assert (
        "common-unit E4M3" in metadata["comparison"]
        and "without amax or division" in metadata["comparison"]
    )


def test_legacy_metric_formulas_are_unchanged():
    branch = next(
        node
        for node in definition(_BENCHMARK, "check_arm").body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "native_fp8"
    )
    metrics = next(
        node.value
        for node in branch.orelse
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "stats"
    )
    original = ast.parse(
        """{
            "rel_l2": float(difference.norm() / expected.norm()),
            "max_scaled": float(difference.abs().max() / expected.abs().max()),
            "lse_abs": float((lse[..., 0].double() - expected_lse).abs().max()),
        }""",
        mode="eval",
    ).body
    assert ast.dump(metrics) == ast.dump(original)


@pytest.mark.parametrize(
    "field,limit", [("rel_l2", 0.005), ("max_scaled", 0.01), ("lse_abs", 1e-5)]
)
def test_legacy_gates_keep_strict_original_limits(field, limit):
    torch = SimpleNamespace(isfinite=lambda out: SimpleNamespace(all=lambda: out))
    check = extract(_BENCHMARK, "check_legacy", torch=torch)
    stats = {"rel_l2": 0.001, "max_scaled": 0.001, "lse_abs": 1e-6}
    check(True, stats)
    with pytest.raises(AssertionError):
        check(False, stats)
    stats[field] = limit
    with pytest.raises(AssertionError):
        check(True, stats)
    stats[field] = limit * 0.99
    check(True, stats)


@pytest.mark.parametrize("name", ["cute-dsl", "prims-ts", "flashmla"])
def test_arm_json_labels_representation_shape_and_accuracy(name):
    native = name == "prims-ts"
    q = SimpleNamespace(dtype="torch.float8_e4m3fn" if native else "torch.bfloat16")
    swa = SimpleNamespace(shape=(2, 64, 1, 512 if native else 528))
    main = SimpleNamespace(shape=(8, 64, 1, 288))
    metadata = extract(_BENCHMARK, "arm_metadata")(
        name, (q, swa, main, None, None, None)
    )
    assert "prims_ts_mode" not in metadata
    assert metadata["q_dtype"] == q.dtype
    assert metadata["swa_encoding"] == ("swa_fp8" if native else "swa_mxfp8")
    assert metadata["global_encoding"] == "main_kv_fp4"
    assert metadata["global_storage"] == (
        "packed E2M1 + group16 E4M3 scales; width288 page-plane"
    )
    assert metadata["cache_shape"] == {
        "swa": [2, 64, 1, 512 if native else 528],
        "global_fp4": [8, 64, 1, 288],
    }
    assert metadata["kv_layout"] == "NHD"
    assert metadata["bmm1_scale"] == 512**-0.5 and metadata["bmm2_scale"] == 1.0
    contract = metadata["accuracy_contract"]
    if native:
        assert "native E4M3 Q/K/V/P for SWA and main" in metadata["arithmetic"]
        assert (
            "BK128/Keep-AB/M64/retained8/split1/nonpersistent" in metadata["arithmetic"]
        )
        assert "original E4M3 Q, no BF16 query scratch" in metadata["arithmetic"]
        assert metadata["global_conversion"] == (
            "common cache units: valid selected E2M1 * signed finite group16 E4M3 "
            "-> E4M3 RN/satfinite SMEM; intentional +/-448 cache-unit saturation; "
            "no request amax, descale or division; shared runtime BMM1/BMM2"
        )
        assert metadata["global_precision"] == "shared_pertensor_e4m3"
        assert metadata["main_scale"] == (
            "none; caller's common KV units, shared runtime BMM factors"
        )
        assert metadata["compile_marker"] == (
            "dsv41_shared_pertensor_joint_e4m3_swa512_fp4_inline_metadata_v2"
        )
        assert contract["reference"] == (
            "independent FP64 over stored E4M3 Q/SWA and common-unit E4M3 main; "
            "mask selected packed bytes before exact decode, then independent "
            "FP32 clamp -> E4M3 RN -> FP64; shared BMM factors applied once"
        )
        assert "componentwise" in contract["output"]
        assert contract["lse"] == "sink-exclusive; rtol=0, atol=1e-5"
        assert "exact zero" in contract["empty"]
        assert "rel_l2_lt" not in contract
    else:
        assert metadata["global_conversion"] == (
            "scaled E2M1 -> BF16 (unchanged legacy represented values)"
        )
        assert (
            contract["reference"] == "independent FP64 over this arm's represented Q/KV"
        )
        assert contract["rel_l2_lt"] == 0.005
        assert contract["max_scaled_lt"] == 0.01
        assert contract["sink_exclusive_lse_abs_lt"] == 1e-5
        label = {
            "cute-dsl": "Frost",
            "flashmla": "FlashMLA",
        }
        assert label[name] in metadata["arithmetic"]
        assert "componentwise_bound" not in contract
        assert (
            not {"global_precision", "main_scale", "compile_marker"} & metadata.keys()
        )


def source_bindings():
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(_BENCHMARK.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("PRIMS_TS_")
    }


def test_source_hashes_stay_native_only():
    bindings = source_bindings()
    assert set(bindings) == {"PRIMS_TS_SOURCES"}
    sources = bindings["PRIMS_TS_SOURCES"]
    assert "experimental/deepseek_v41/_sparse_views.py" in sources
    assert "experimental/deepseek_v41/_metadata.py" not in sources
    assert "experimental/deepseek_v41/_workspace.py" in sources
    assert "experimental/deepseek_v41/prims_ts.py" in sources
    assert "experimental/deepseek_v41/prims_ts_fp8.py" in sources
    assert "experimental/deepseek_v41/prims_ts_resources.py" in sources
    assert "cute_dsl/fp4_common.py" in sources
    assert "attention/prims_ts/kernels/mla_decode/helpers/ops.py" in sources
    assert not any("mxfp8" in source for source in sources)


@pytest.mark.parametrize(
    "backends", [["cute-dsl"], ["prims-ts"], ["cute-dsl", "prims-ts"]]
)
@pytest.mark.parametrize("flashmla", [False, True])
def test_backend_specific_source_hashes_without_importing_modules(backends, flashmla):
    parser = extract(_BENCHMARK, "build_parser", argparse=argparse, Path=Path)()
    argv = ["--output", "unused.json", "--backends", *backends]
    if flashmla:
        argv.append("--flashmla")
    opts = parser.parse_args(argv)
    bindings = source_bindings()
    block = next(
        node
        for node in definition(_BENCHMARK, "main").body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "'prims-ts' in opts.backends"
    )
    paths = {}
    namespace = {
        "opts": opts,
        "Path": Path,
        "decode": SimpleNamespace(
            __file__=str(_ROOT / "flashinfer/experimental/deepseek_v41/decode.py")
        ),
        "source_paths": paths,
        **bindings,
    }
    exec(
        compile(ast.Module(body=[block], type_ignores=[]), str(_BENCHMARK), "exec"),
        namespace,
    )
    sources = ()
    if "prims-ts" in backends:
        sources = bindings["PRIMS_TS_SOURCES"]
    assert paths == {
        f"flashinfer.{relative[:-3].replace('/', '.')}": _ROOT / "flashinfer" / relative
        for relative in sources
    }
    assert all(path.is_file() for path in paths.values())


def test_example_uses_common_descale_hnd_and_output_only_caller_buffers():
    main = definition(_EXAMPLE, "main")
    assignments = {
        ast.unparse(node.targets[0]): node.value
        for node in main.body
        if isinstance(node, ast.Assign)
    }
    assert ast.literal_eval(assignments["kv_descale"]) == 2.0
    assert (
        ast.unparse(assignments["cache_units"])
        == "(x.float() / kv_descale).contiguous()"
    )
    for name in ("swa", "main_cache"):
        writer = assignments[name]
        assert ast.unparse(writer.func) == "ds.deepseek_v41_quantize_cache"
        assert [ast.unparse(arg) for arg in writer.args] == ["cache_units"]
        assert not any(kw.arg == "kv_layout" for kw in writer.keywords)
    nested = next(
        node
        for node in main.body
        if isinstance(node, ast.FunctionDef) and node.name == "mixed_decode"
    )
    decode = next(node.value for node in nested.body if isinstance(node, ast.Return))
    assert ast.unparse(decode.func) == "ds.deepseek_v41_decode"
    assert [ast.unparse(arg) for arg in decode.args] == [
        "q",
        "swa",
        "workspace",
        "indices",
        "main_cache",
        "topk_lens",
        "seq_lens",
        "out",
    ]
    scales = {kw.arg: ast.unparse(kw.value) for kw in decode.keywords}
    assert (
        scales["bmm1_scale"] == "512 ** (-0.5) * kv_descale"
        and scales["bmm2_scale"] == "kv_descale"
    )
    assert "kv_layout" not in scales and "plan" not in scales
    checks = [
        ast.unparse(node.test) for node in main.body if isinstance(node, ast.Assert)
    ]
    assert checks == ["mixed_decode() is out"] * 2


@pytest.mark.parametrize("split_scale", [False, True])
@pytest.mark.parametrize("use_defaults", [False, True])
def test_split_softmax_scale_materializes_fp32_before_task_construction(
    split_scale, use_defaults
):
    """Check setup plumbing with host mocks, not actual CuTe loop lowering."""
    kernel_dir = _ROOT / "flashinfer/attention/prims_ts/kernels/mla_decode"
    kernel = kernel_dir / "throughput_latency_1cta/kernel.py"
    constants = ast.parse((kernel_dir / "helpers/constants.py").read_text())
    log2_e = next(
        ast.literal_eval(node.value)
        for node in constants.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "LOG2_E"
    )
    materialized = []

    class HostFloat32:
        # Like CuTe, retain a Python literal until explicitly materialized.
        def __init__(self, value):
            self.value = value

        def ir_value(self):
            assert isinstance(self.value, float)
            materialized.append(self.value)
            return SimpleNamespace(value=float32(self.value))

    function = definition(kernel, "_make_single_kv_pipe_task_graph")
    setup_end = next(
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "_check_persistent_scheduler_modes"
    )
    supplied_scale, supplied_output = object(), object()
    namespace = {
        "cfg": SimpleNamespace(sparse_split_softmax_scale=split_scale),
        "scale_softmax_log2": None if use_defaults else supplied_scale,
        "output_scale": None if use_defaults else supplied_output,
        "_default_scales": extract(kernel, "_default_scales", Float32=HostFloat32),
        "Float32": HostFloat32,
        "LOG2_E": log2_e,
    }
    exec(
        compile(
            ast.Module(body=function.body[:setup_end], type_ignores=[]),
            str(kernel),
            "exec",
        ),
        namespace,
    )
    scale = namespace["scale_softmax_log2"]
    raw_scale = namespace["raw_softmax_scale"]
    output = namespace["output_scale"]
    if split_scale:
        assert materialized == [log2_e]
        assert isinstance(scale, HostFloat32)
        assert isinstance(scale.value, SimpleNamespace)
        assert scale.value.value == float32(log2_e) != log2_e
        if use_defaults:
            assert isinstance(raw_scale, HostFloat32) and raw_scale.value == 1.0
        else:
            assert raw_scale is supplied_scale
    else:
        assert not materialized and raw_scale is None
        if use_defaults:
            assert isinstance(scale, HostFloat32) and scale.value == 1.0
        else:
            assert scale is supplied_scale
    if use_defaults:
        assert isinstance(output, HostFloat32) and output.value == 1.0
    else:
        assert output is supplied_output
