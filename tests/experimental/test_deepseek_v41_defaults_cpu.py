# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Benchmark/example contracts without importing Torch, CUDA, CuTe or FlashInfer.

Run with --confcutdir=tests/experimental to exclude root CUDA collection probes.
AST extraction executes only parser/fixture/gate helpers with stdlib mocks;
these checks are not numerical or GPU qualification of the rewritten sources.
"""

import argparse
import ast
import builtins
from pathlib import Path
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
def test_native_first_parser_has_no_mxfp8_override(path):
    parser = extract(path, "build_parser", argparse=argparse, Path=Path)()
    argv = ["--output", "results.json"] if path == _BENCHMARK else []
    with pytest.raises(SystemExit) as error:
        parser.parse_args(argv + ["--prims-ts-mode", "mxfp8"])
    assert error.value.code == 2


@pytest.mark.parametrize("backend", [None, "cute-dsl", "prims-ts"])
def test_example_parser_defaults_and_explicit_backend(backend):
    parser = extract(_EXAMPLE, "build_parser", argparse=argparse)()
    opts = parser.parse_args(["--backend", backend] if backend else [])
    assert opts.backend == (backend or "cute-dsl")
    assert not hasattr(opts, "prims_ts_mode")


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
def test_benchmark_fixture_common_source_and_shared_global_pool(backends, flashmla):
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
    if "prims-ts" in backends:
        q_source.to.assert_called_once_with(torch.float8_e4m3fn)
        assert arms["prims-ts"][0] is q_source.to.return_value
        assert arms["prims-ts"][1] is caches["swa_fp8"]
        quantize.assert_any_call(w_source, format="swa_fp8")
    else:
        q_source.to.assert_not_called()
    for name in ("cute-dsl", "flashmla"):
        if name in arms:
            assert arms[name][0] is q_source
            assert arms[name][1] is caches["swa_mxfp8"]
            quantize.assert_any_call(w_source, format="swa_mxfp8")
    if "cute-dsl" in arms and "flashmla" in arms:
        assert arms["cute-dsl"] is arms["flashmla"]
    assert {c.kwargs["format"] for c in quantize.call_args_list} == (
        {"main_kv_fp4"}
        | ({"swa_fp8"} if "prims-ts" in backends else set())
        | ({"swa_mxfp8"} if "cute-dsl" in backends or flashmla else set())
    )


def test_example_native_representation_is_explicit():
    main = definition(_EXAMPLE, "main")
    assignments = {
        node.targets[0].id: node.value
        for node in main.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert ast.unparse(assignments["native_fp8"]) == "opts.backend == 'prims-ts'"
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
        "selected(pool, safe[:, None], True, round_bf16=False)",
    ]
    valid = next(
        node for node in assignments if ast.unparse(node.targets[0]) == "valid"
    )
    assert ast.unparse(valid.value) == (
        "(index >= 0) & (index < pool.shape[0] * 64) & (index != 2147483647)"
    )
    # The legacy decoder still rounds represented values to BF16 by default;
    # the native reference explicitly bypasses that rounding for FP4 values.
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
    assert metadata["q_dtype"] == q.dtype
    assert metadata["swa_encoding"] == ("swa_fp8" if native else "swa_mxfp8")
    assert metadata["cache_shape"] == {
        "swa": [2, 64, 1, 512 if native else 528],
        "global_fp4": [8, 64, 1, 288],
    }
    contract = metadata["accuracy_contract"]
    if native:
        assert "native E4M3" in metadata["arithmetic"]
        assert "componentwise" in contract["output"]
        assert contract["lse"] == "sink-exclusive; rtol=0, atol=1e-5"
        assert "exact zero" in contract["empty"]
        assert "rel_l2_lt" not in contract
    else:
        assert contract["rel_l2_lt"] == 0.005
        assert contract["max_scaled_lt"] == 0.01
        assert contract["sink_exclusive_lse_abs_lt"] == 1e-5
        assert ("Frost" if name == "cute-dsl" else "FlashMLA") in metadata["arithmetic"]


def test_native_source_hash_list_has_fp8_and_shared_resources_only():
    sources = next(
        ast.literal_eval(node.value)
        for node in ast.parse(_BENCHMARK.read_text()).body
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "PRIMS_TS_SOURCES"
    )
    assert "experimental/deepseek_v41/prims_ts.py" in sources
    assert "experimental/deepseek_v41/prims_ts_fp8.py" in sources
    assert "experimental/deepseek_v41/prims_ts_resources.py" in sources
    assert not any("mxfp8" in source for source in sources)
    assert all((_ROOT / "flashinfer" / source).is_file() for source in sources)
    assert any(
        isinstance(node, ast.For) and ast.unparse(node.iter) == "PRIMS_TS_SOURCES"
        for node in ast.walk(definition(_BENCHMARK, "main"))
    )
