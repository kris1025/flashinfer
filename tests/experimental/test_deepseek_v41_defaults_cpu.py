# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Benchmark/example contracts without importing Torch, CUDA, CuTe or FlashInfer.

Run with --confcutdir=tests/experimental to exclude root CUDA collection probes.
AST extraction executes parser/fixture/gate helpers with stdlib mocks and checks
request-scaled reference arithmetic with a small stdlib-only numerical tensor.
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


class RequestTensor:
    """Only the benchmark normalization ops, with explicit FP32 materialization.

    Each request has scalar samples shaped [B, K, 1]; no accelerator or array
    library is needed to exercise amax, broadcast descales and staged rounding.
    """

    device = "host"

    def __init__(self, rows, dtype="float64", staged=None):
        self.rows = tuple(tuple(row) for row in rows)
        assert self.rows and all(len(row) == len(self.rows[0]) for row in self.rows)
        self.shape = (len(self.rows), len(self.rows[0]), 1)
        self.dtype = dtype
        self.staged = [] if staged is None else staged

    def map(self, operation, dtype=None):
        return RequestTensor(
            [[operation(value) for value in row] for row in self.rows],
            dtype or self.dtype,
            self.staged,
        )

    def binary(self, other, operation, dtype=None):
        if not isinstance(other, RequestTensor):
            other = RequestTensor([[other] for _ in self.rows])
        assert len(self.rows) == len(other.rows)
        result = []
        for left, right in zip(self.rows, other.rows, strict=True):
            width = max(len(left), len(right)) if left and right else 0
            assert not width or (len(left) in (1, width) and len(right) in (1, width))
            result.append(
                [
                    operation(
                        left[i if len(left) > 1 else 0],
                        right[i if len(right) > 1 else 0],
                    )
                    for i in range(width)
                ]
            )
        value = RequestTensor(result, dtype or self.dtype, self.staged)
        return value.float() if value.dtype == "float32" else value

    def float(self):
        return self.map(float32, "float32")

    def double(self):
        return self.map(float, "float64")

    def abs(self):
        return self.map(abs)

    def amax(self, *, dim, keepdim):
        assert dim == (1, 2) and keepdim
        return RequestTensor([[max(row)] for row in self.rows], self.dtype, self.staged)

    def clamp(self, low, high):
        return self.map(lambda value: min(max(value, low), high))

    def to(self, dtype):
        assert dtype == "float8_e4m3fn"
        self.staged.append((self.dtype, self.rows))
        return self.map(e4m3_rn_satfinite, dtype)

    def __truediv__(self, other):
        assert isinstance(other, RequestTensor)
        assert self.dtype == other.dtype == "float32"
        return self.binary(other, lambda left, right: left / right)

    def __mul__(self, other):
        return self.binary(other, lambda left, right: left * right)

    def __gt__(self, other):
        return self.binary(other, lambda left, right: left > right, "bool")


def numeric_request_scaled_main(rows):
    decoded = RequestTensor(rows)

    def where(condition, candidate, fallback):
        assert condition.shape == candidate.shape
        return RequestTensor(
            [
                [
                    value if keep else fallback
                    for keep, value in zip(mask, row, strict=True)
                ]
                for mask, row in zip(condition.rows, candidate.rows, strict=True)
            ],
            candidate.dtype,
            candidate.staged,
        )

    def zeros(shape, *, device, dtype):
        assert shape == (len(rows), 1, 1) and device == "host" and dtype == "float32"
        return RequestTensor([[0.0] for _ in rows], dtype, decoded.staged)

    def full_like(tensor, value):
        assert tensor.dtype == "float32" and tensor.shape == (len(rows), 1, 1)
        assert value == 448.0
        return tensor.map(lambda _: float32(value))

    torch = SimpleNamespace(
        float32="float32",
        float8_e4m3fn="float8_e4m3fn",
        where=where,
        zeros=zeros,
        full_like=full_like,
    )
    normalize = extract(_BENCHMARK, "request_scaled_main", torch=torch)
    represented, scale = normalize(decoded)
    assert len(decoded.staged) == 1
    return represented, scale, decoded.staged[0]


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
    assert opts.prims_ts_mode == "fp8"


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
@pytest.mark.parametrize("mode", [None, "fp8", "mxfp8"])
@pytest.mark.parametrize("select_prims_ts", [False, True])
def test_prims_ts_mode_parser_defaults_and_override(path, mode, select_prims_ts):
    parser = extract(path, "build_parser", argparse=argparse, Path=Path)()
    argv = ["--output", "results.json"] if path == _BENCHMARK else []
    if mode:
        argv += ["--prims-ts-mode", mode]
    if select_prims_ts:
        argv += ["--backends" if path == _BENCHMARK else "--backend", "prims-ts"]
    opts = parser.parse_args(argv)
    assert opts.prims_ts_mode == (mode or "fp8")
    backend = "prims-ts" if select_prims_ts else "cute-dsl"
    assert (opts.backends if path == _BENCHMARK else [opts.backend]) == [backend]


@pytest.mark.parametrize("path", [_BENCHMARK, _EXAMPLE])
def test_prims_ts_mode_parser_rejects_unknown_mode(path):
    parser = extract(path, "build_parser", argparse=argparse, Path=Path)()
    argv = ["--output", "results.json"] if path == _BENCHMARK else []
    with pytest.raises(SystemExit) as error:
        parser.parse_args(argv + ["--prims-ts-mode", "bf16"])
    assert error.value.code == 2


@pytest.mark.parametrize("backend", [None, "cute-dsl", "prims-ts"])
def test_example_parser_defaults_and_explicit_backend(backend):
    parser = extract(_EXAMPLE, "build_parser", argparse=argparse)()
    opts = parser.parse_args(["--backend", backend] if backend else [])
    assert opts.backend == (backend or "cute-dsl")
    assert opts.prims_ts_mode == "fp8"


@pytest.mark.parametrize("mode", [None, "fp8", "mxfp8"])
@pytest.mark.parametrize(
    "backends,flashmla",
    [
        (["cute-dsl"], False),
        (["prims-ts"], False),
        (["cute-dsl", "prims-ts"], False),
        (["prims-ts"], True),
        (["cute-dsl", "prims-ts"], True),
    ],
)
def test_benchmark_fixture_common_source_and_shared_global_pool(
    backends, flashmla, mode
):
    torch = MagicMock(name="torch_stub")
    q_source, w_source, values, sink = (
        MagicMock(name=name) for name in ("q_source", "w_source", "values", "sink")
    )
    main, ci = MagicMock(name="shared_main"), MagicMock(name="shared_ci")
    torch.randn.side_effect = [q_source, w_source, values, sink]
    torch.empty.side_effect = [main, ci]
    caches = {format: object() for format in ("main_kv_fp4", "swa_mxfp8", "swa_fp8")}
    quantize = Mock(side_effect=lambda source, *, format: caches[format])
    fixture = extract(
        _BENCHMARK,
        "fixture",
        torch=torch,
        deepseek_v41_quantize_cache=quantize,
    )
    mode_args = {"prims_ts_mode": mode} if mode else {}
    native = "prims-ts" in backends and mode != "mxfp8"
    arms = fixture(1, 512, 7003, backends, flashmla, **mode_args)
    assert set(arms) == set(backends) | ({"flashmla"} if flashmla else set())
    torch.manual_seed.assert_called_once_with(7004)
    assert torch.randn.call_args_list == [
        call(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16),
        call(128, 512, device="cuda", dtype=torch.bfloat16),
        call(512, 512, device="cuda", dtype=torch.bfloat16),
        call(64, device="cuda"),
    ]
    # A single global quantization/allocation serves every arm, including FlashMLA.
    quantize.assert_any_call(values, format="main_kv_fp4")
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
        quantize.assert_any_call(w_source, format="swa_fp8")
    else:
        q_source.to.assert_not_called()
    legacy_names = [name for name in arms if name != "prims-ts" or not native]
    for name in legacy_names:
        assert arms[name][0] is q_source
        assert arms[name][1] is caches["swa_mxfp8"]
        assert arms[name] is arms[legacy_names[0]]
        quantize.assert_any_call(w_source, format="swa_mxfp8")
    formats = [c.kwargs["format"] for c in quantize.call_args_list]
    assert set(formats) == (
        {"main_kv_fp4"}
        | ({"swa_fp8"} if native else set())
        | ({"swa_mxfp8"} if legacy_names else set())
    )
    assert len(formats) == len(set(formats))


@pytest.mark.parametrize("backend", ["cute-dsl", "prims-ts"])
@pytest.mark.parametrize("mode", ["fp8", "mxfp8"])
def test_example_representation_is_explicit_and_mode_scoped(backend, mode):
    main = definition(_EXAMPLE, "main")
    assignments = {
        node.targets[0].id: node.value
        for node in main.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    native = eval(
        compile(ast.Expression(assignments["native_fp8"]), str(_EXAMPLE), "eval"),
        {"opts": SimpleNamespace(backend=backend, prims_ts_mode=mode)},
    )
    assert native == (backend == "prims-ts" and mode == "fp8")
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
    # separate masked-byte gather and request-scaled reference below.
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
        "torch.einsum('bhd,bkd->bhk', q[:, 0].double(), kv) * 512 ** (-0.5)"
    )
    assert inputs["expected"] == "torch.einsum('bhk,bkd->bhd', weights, kv)"
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "bfloat16"
        for function in (
            oracle,
            definition(_BENCHMARK, "selected_main_fp8"),
            definition(_BENCHMARK, "request_scaled_main"),
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
    normalize = Mock(return_value=(represented, object()))
    gather = extract(
        _BENCHMARK, "selected_main_fp8", torch=torch, request_scaled_main=normalize
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
    assert assignments["(represented, _)"] == "request_scaled_main(decoded)"


def test_request_scaled_main_uses_exact_payload_amax_not_max_group_times_six():
    payloads = (0, 1, 1)
    scales = tuple(e4m3_value(code) for code in (0x7E, 0x40, 0xC0))
    decoded = [payload * scale for payload, scale in zip(payloads, scales, strict=True)]
    assert decoded == [0.0, 2.0, -2.0]
    represented, scale, staged = numeric_request_scaled_main([decoded])
    expected_scale = float32(2 / 448)
    assert scale.dtype == "float32" and scale.rows == ((expected_scale,),)
    assert expected_scale != float32(max(map(abs, scales)) * 6 / 448)
    assert represented.dtype == "float64"
    assert represented.rows == ((0.0, 448 * expected_scale, -448 * expected_scale),)
    assert staged == (
        "float32",
        ((0.0, float32(2 / expected_scale), float32(-2 / expected_scale)),),
    )


def test_request_scaled_main_materializes_float32_division_and_restores_range():
    assignments = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in definition(_BENCHMARK, "request_scaled_main").body
        if isinstance(node, ast.Assign)
    }
    assert assignments["divisor"] == "torch.full_like(amax, 448.0)"
    assert assignments["scale"] == (
        "torch.where(amax > 0, (amax / divisor).float(), 1)"
    )
    represented, scale, staged = numeric_request_scaled_main([[2688.0, 1.0, -1.0, 7.0]])
    assert scale.rows == ((6.0,),)
    assert represented.rows == ((2688.0, 1.03125, -1.03125, 6.75),)
    assert staged == (
        "float32",
        ((448.0, float32(1 / 6), float32(-1 / 6), float32(7 / 6)),),
    )
    # This valid decoded FP4 value distinguishes division from reciprocal multiply;
    # the recorded FP32 intermediate also rejects an FP64 quotient before FP8 cast.
    assert staged[1][0][3] != 7 / 6
    assert staged[1][0][3] != float32(7 * float32(1 / 6))


@pytest.mark.parametrize("amax", [1.0, 2**-10, 224.0])
def test_request_scaled_main_allows_descales_below_one(amax):
    represented, scale, staged = numeric_request_scaled_main([[amax, -amax]])
    expected_scale = float32(amax / 448)
    assert 0 < expected_scale < 1 and scale.rows == ((expected_scale,),)
    assert represented.rows == ((448 * expected_scale, -448 * expected_scale),)
    assert staged[0] == "float32"


@pytest.mark.parametrize("rows", [[[]], [[0.0, -0.0]]])
def test_request_scaled_main_uses_one_for_empty_or_zero_requests(rows):
    represented, scale, _ = numeric_request_scaled_main(rows)
    assert scale.dtype == "float32" and scale.rows == ((1.0,),)
    assert represented.rows == tuple(tuple(row) for row in rows)
    if rows[0]:
        assert math.copysign(1, represented.rows[0][1]) == -1


def test_request_scaled_main_keeps_duplicate_and_all_zero_valid_samples():
    represented, scale, _ = numeric_request_scaled_main([[0.0, 0.0], [7.0, 7.0]])
    assert scale.rows == ((1.0,), (float32(7 / 448),))
    assert represented.rows == ((0.0, 0.0), (7.0, 7.0))
    assert represented.shape == (2, 2, 1)


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
@pytest.mark.parametrize("mode", [None, "fp8", "mxfp8"])
def test_arm_selects_mode_specific_oracle_and_gate(name, mode):
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
    mode_args = {"prims_ts_mode": mode} if mode else {}
    stats = check(
        name, args, Mock(return_value=(out, lse)), 1, Path("unused.json"), **mode_args
    )
    if name == "prims-ts" and mode != "mxfp8":
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


def test_benchmark_threads_mode_to_fixture_gate_and_json():
    main = definition(_BENCHMARK, "main")
    for helper in ("fixture", "check_arm", "arm_metadata"):
        calls = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == helper
        ]
        assert calls
        assert all(
            any(
                keyword.arg == "prims_ts_mode"
                and ast.unparse(keyword.value) == "opts.prims_ts_mode"
                for keyword in node.keywords
            )
            for node in calls
        )
    assert any(
        isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "prims_ts_mode"
            and ast.unparse(value) == "opts.prims_ts_mode"
            for key, value in zip(node.keys, node.values, strict=True)
        )
        for node in ast.walk(main)
    )


def test_benchmark_times_full_public_replay_and_labels_request_scaled_preparation():
    main = definition(_BENCHMARK, "main")
    replay = next(
        node.value
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "fns[backend]"
    )
    plan, out, lse = (object() for _ in range(3))
    args = (object(),) * 6
    decode = Mock(return_value=(out, lse, plan))
    # The callable given to graph() invokes the whole API on every replay.
    fn = eval(
        compile(ast.Expression(replay), str(_BENCHMARK), "eval"),
        {
            "args": args,
            "plans": {"prims-ts": plan},
            "backend": "prims-ts",
            "deepseek_v41_decode": decode,
        },
    )
    assert fn() == (out, lse) and fn() == (out, lse)
    assert decode.call_args_list == [call(*args, plan=plan, backend="prims-ts")] * 2
    graphs = next(
        node.value
        for node in ast.walk(main)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "graphs"
    )
    assert ast.unparse(graphs) == "{name: graph(fn) for name, fn in fns.items()}"
    report = next(
        node.value
        for node in main.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "report"
    )
    metadata = {
        ast.literal_eval(key): ast.literal_eval(value)
        for key, value in zip(report.keys, report.values, strict=True)
        if ast.literal_eval(key) in {"scope", "comparison"}
    }
    assert "selection/count/amax preparation" in metadata["scope"]
    assert "selected-tile staging, both source passes and merge" in metadata["scope"]
    assert "request_scaled E4M3" in metadata["comparison"]


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
@pytest.mark.parametrize("mode", [None, "fp8", "mxfp8"])
def test_arm_json_labels_representation_shape_and_accuracy(name, mode):
    native = name == "prims-ts" and mode != "mxfp8"
    q = SimpleNamespace(dtype="torch.float8_e4m3fn" if native else "torch.bfloat16")
    swa = SimpleNamespace(shape=(2, 64, 1, 512 if native else 528))
    main = SimpleNamespace(shape=(8, 64, 1, 288))
    mode_args = {"prims_ts_mode": mode} if mode else {}
    metadata = extract(_BENCHMARK, "arm_metadata")(
        name, (q, swa, main, None, None, None), **mode_args
    )
    assert metadata["prims_ts_mode"] == (
        (mode or "fp8") if name == "prims-ts" else None
    )
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
    contract = metadata["accuracy_contract"]
    if native:
        assert "native E4M3 Q/K/V/P for SWA and main" in metadata["arithmetic"]
        assert (
            "BK128/Keep-AB/M64/retained8/split1/nonpersistent" in metadata["arithmetic"]
        )
        assert "original E4M3 Q, no BF16 query scratch" in metadata["arithmetic"]
        assert metadata["global_conversion"] == (
            "request_scaled: valid selected E2M1 * signed finite group16 E4M3; "
            "one FP32 s[b]=RN(selected payload amax/448), 1 only for zero/empty; "
            "RN_FP32(x/s) -> E4M3 RN/satfinite SMEM; represented double(staged)*double(s), "
            "not unit-clipped (decoded finite range +/-2688); Q/SWA remain unit E4M3"
        )
        assert metadata["global_precision"] == "request_scaled_e4m3"
        assert metadata["main_scale"] == (
            "FP32[B] scratch, refreshed per call; None in window-only plans"
        )
        assert metadata["compile_marker"] == (
            "dsv41_request_scaled_e4m3_q_swa512_fp4_to_e4m3_fp32_sources_v3"
        )
        assert contract["reference"] == (
            "independent FP64 over stored E4M3 Q/SWA and request_scaled main KV; "
            "mask selected packed bytes before exact decode/amax, materialize FP32 "
            "x/s before E4M3 RN/satfinite, then double(staged)*double(s)"
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
            "prims-ts": "Prims-TS BF16",
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
        and node.targets[0].id in {"PRIMS_TS_SOURCES", "PRIMS_TS_MXFP8_SOURCES"}
    }


def test_native_source_hash_defaults_stay_native_only():
    bindings = source_bindings()
    sources = bindings["PRIMS_TS_SOURCES"]
    extensions = bindings["PRIMS_TS_MXFP8_SOURCES"]
    assert "experimental/deepseek_v41/prims_ts.py" in sources
    assert "experimental/deepseek_v41/prims_ts_fp8.py" in sources
    assert "experimental/deepseek_v41/prims_ts_resources.py" in sources
    assert "cute_dsl/fp4_common.py" in sources
    assert "attention/prims_ts/kernels/mla_decode/helpers/ops.py" in sources
    assert not any("mxfp8" in source for source in sources)
    assert extensions == (
        "experimental/deepseek_v41/prims_ts_mxfp8.py",
        "experimental/deepseek_v41/prims_ts_mxfp8_resources.py",
    )
    select_sources = extract(_BENCHMARK, "prims_ts_sources", **bindings)
    assert select_sources() == sources
    assert select_sources("fp8") == sources
    assert select_sources("mxfp8") == sources + extensions


@pytest.mark.parametrize(
    "backends", [["cute-dsl"], ["prims-ts"], ["cute-dsl", "prims-ts"]]
)
@pytest.mark.parametrize("mode", [None, "fp8", "mxfp8"])
def test_mode_specific_source_hashes_without_importing_modules(backends, mode):
    parser = extract(_BENCHMARK, "build_parser", argparse=argparse, Path=Path)()
    argv = ["--output", "unused.json", "--backends", *backends]
    if mode:
        argv += ["--prims-ts-mode", mode]
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
        "prims_ts_sources": extract(_BENCHMARK, "prims_ts_sources", **bindings),
    }
    exec(
        compile(ast.Module(body=[block], type_ignores=[]), str(_BENCHMARK), "exec"),
        namespace,
    )
    sources = ()
    if "prims-ts" in backends:
        sources = bindings["PRIMS_TS_SOURCES"] + (
            bindings["PRIMS_TS_MXFP8_SOURCES"] if mode == "mxfp8" else ()
        )
    assert paths == {
        f"flashinfer.{relative[:-3].replace('/', '.')}": _ROOT / "flashinfer" / relative
        for relative in sources
    }
    assert all(path.is_file() for path in paths.values())
