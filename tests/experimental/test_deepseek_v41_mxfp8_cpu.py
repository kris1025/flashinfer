# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Guarded host contracts for the optional Prims-TS MXFP8 extension.

Run in a fresh process with --confcutdir=tests/experimental. Reuse the native
fixture's strict Torch/raw-CUDA/CuTe guards before importing either adapter or
resource module; kernel launches and compilation are replaced with host stubs.
These tests do not qualify GPU execution or packed-cache numerics.
"""

import ast
import builtins
from dataclasses import replace
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from .test_deepseek_v41_source_state_cpu import (
    _host_case,
    _metadata_case,
    adapter as adapter,
    native as native,
)


def _declaration(args):
    return tuple(
        None if x is None else (tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
        for x in args
    )


def _bf16_packed_config(runtime):
    attention = runtime.kernel.ThroughputLatencyMlaDecodeTs(
        batch_size=1,
        num_heads=64,
        seq_len_q=1,
        seq_len_k=256,
        latent_dim=512,
        rope_dim=0,
        page_size=1,
        max_active_clusters=148,
        qkv_dtype="bf16",
        out_dtype="bf16",
        profile=runtime.config.MlaProfile(
            name="mxfp8_resource_cpu",
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
        sparse_profile=runtime.profile(
            family="keep",
            tile_size_q=64,
            split_kv=1,
            scheduler="nonpersistent",
            head_dim_ctas=1,
            gather_issue_warps=4,
            direct_inputs=True,
            reuse_kv=True,
            reuse_kv_stages=8,
            kv_tile_size=64,
            page_pipeline_stages=1,
        ),
        finalize_output=True,
    )
    attention.direct_sparse = True
    attention.direct_static_scales = True
    attention.direct_sparse_capacities = (128, 64)
    cfg = attention._make_config()
    assert (cfg.qkv_dtype, cfg.tile_size_kv, cfg.kv_stages) == ("bf16", 64, 8)
    assert cfg.sparse_direct
    return cfg


def _mxfp8_case(batch=2, compressed=64):
    import torch

    args = list(_host_case(batch=batch, compressed=compressed))
    args[0] = args[0].bfloat16()
    args[1] = torch.zeros(2, 64, 1, 528, dtype=torch.uint8)
    return tuple(args)


@pytest.fixture
def mxfp8(adapter, monkeypatch):
    from flashinfer.experimental.deepseek_v41 import prims_ts_mxfp8

    launches, compilations = [], []

    def compile_stub(device, compressed):
        compilations.append((device, compressed))
        return lambda *args: launches.append(args)

    monkeypatch.setattr(prims_ts_mxfp8, "_compile", compile_stub)

    def run(args, plan=None):
        return prims_ts_mxfp8.decode(*args, signature=_declaration(args), plan=plan)

    return SimpleNamespace(
        torch=adapter.torch,
        native=adapter,
        module=prims_ts_mxfp8,
        launches=launches,
        compilations=compilations,
        run=run,
    )


@pytest.mark.parametrize("compressed", (0, 192))
@pytest.mark.parametrize("prepared", (False, True))
def test_bf16_entry_lazily_routes_validated_declaration_and_plan(
    native, monkeypatch, compressed, prepared
):
    import torch
    from flashinfer.experimental.deepseek_v41 import prims_ts

    args = _metadata_case(fp8=False, compressed=compressed)
    signature = _declaration(args)
    plan = object() if prepared else None
    result = (object(), object(), object())
    events, declarations = [], []
    original_import = builtins.__import__
    original_validate = prims_ts._validate_inputs

    def validate(*actual, **kwargs):
        assert kwargs == {"allow_fp8_swa": True}
        declaration = original_validate(*actual, **kwargs)
        assert declaration == signature
        declarations.append(declaration)
        events.append("validated")
        return declaration

    def run(*actual, **kwargs):
        assert all(a is b for a, b in zip(actual, args, strict=True))
        assert kwargs["signature"] is declarations[0]
        assert kwargs["plan"] is plan
        events.append("mxfp8")
        return result

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "prims_ts_mxfp8" and level == 1:
            assert events == ["validated"]
            assert fromlist == ("decode",)
            events.append("import")
            return SimpleNamespace(decode=run)
        return original_import(name, globals, locals, fromlist, level)

    def forbidden(*args, **kwargs):
        raise AssertionError("BF16 Q must not enter native FP8 decode")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    monkeypatch.setattr(prims_ts, "_validate_inputs", validate)
    monkeypatch.setattr(prims_ts, "_decode_fp8", forbidden)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert prims_ts.decode(*args, plan=plan) is result
    assert events == ["validated", "import", "mxfp8"]


@pytest.mark.parametrize("fp8", (False, True))
def test_entry_validates_before_native_or_mxfp8_routing(native, monkeypatch, fp8):
    import torch
    from flashinfer.experimental.deepseek_v41 import prims_ts

    args = _metadata_case(fp8=fp8)
    args[0].shape = (2, 1, 32, 512)
    imports, routes = [], []
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if "prims_ts_mxfp8" in name or any(
            "prims_ts_mxfp8" in item for item in fromlist or ()
        ):
            imports.append(name)
            raise AssertionError("invalid declarations must not import MXFP8")
        return original_import(name, globals, locals, fromlist, level)

    def forbidden(*args, **kwargs):
        routes.append(args)
        raise AssertionError("invalid declarations must not reach a router")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    monkeypatch.setattr(prims_ts, "_decode_fp8", forbidden)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="Q must be BF16 or E4M3"):
        prims_ts.decode(*args, plan=object())
    assert not imports and not routes


@pytest.mark.parametrize("compressed", (0, 64, 192, 512))
def test_mxfp8_owned_shapes_bindings_and_changed_input_replay(mxfp8, compressed):
    torch = mxfp8.torch
    args = _mxfp8_case(compressed=compressed)
    out, lse, plan = mxfp8.run(args)
    assert type(plan) is mxfp8.module._Plan
    assert plan.signature == _declaration(args) and plan.split == 1
    assert out is plan.out and lse is plan.lse
    assert out.dtype == torch.bfloat16 and out.shape == args[0].shape
    assert lse.dtype == torch.float32 and lse.shape == (2, 64, 1)
    assert plan.selected_swa.dtype == torch.int32
    assert plan.selected_swa.shape == (2, 128)
    assert plan.selected_global.dtype == torch.int32
    assert plan.selected_global.shape == (2, max(64, compressed))
    for lengths, value in ((plan.swa_lengths, 128), (plan.global_lengths, compressed)):
        assert lengths.dtype == torch.int32 and lengths.shape == (2,)
        assert (lengths == value).all()
    assert plan.empty_cache.dtype == torch.uint8
    assert plan.empty_cache.shape == (1, 64, 1, 288)
    assert plan.empty_indices.dtype == torch.int32
    assert plan.empty_indices.shape == (2, 64) and (plan.empty_indices == -1).all()
    assert all(
        tensor.device.type == "cpu"
        for tensor in (
            out,
            lse,
            plan.selected_swa,
            plan.selected_global,
            plan.swa_lengths,
            plan.global_lengths,
            plan.empty_cache,
            plan.empty_indices,
        )
    )
    assert mxfp8.compilations == [(None, compressed)]
    call = mxfp8.launches[-1]
    assert len(call) == 12
    assert call[0].shape == (64, 512, 1, 2)
    assert call[0].data_ptr() == args[0].data_ptr()
    assert call[1].shape == (2 * 64 * 528,)
    assert call[1].data_ptr() == args[1].data_ptr()
    main = args[2] if compressed else plan.empty_cache
    ids = args[4] if compressed else plan.empty_indices
    assert call[2].data_ptr() == main.data_ptr() and call[2].shape == (main.numel(),)
    assert call[3].data_ptr() == args[3].data_ptr() and call[3].shape == (2, 128)
    assert call[4].data_ptr() == ids.data_ptr()
    assert call[4].shape == (2, max(64, compressed))
    assert call[5] is plan.selected_swa and call[6] is plan.selected_global
    assert call[7] is plan.swa_lengths and call[8] is plan.global_lengths
    assert call[9] is args[5]
    assert call[10].data_ptr() == out.data_ptr() and call[10].shape == (64, 512, 1, 2)
    assert call[11].data_ptr() == lse.data_ptr() and call[11].shape == (64, 1, 2)

    replacement = tuple(None if x is None else x.clone() for x in args)
    replacement[0].fill_(1)
    replacement[3][..., 0] = 0
    replay = mxfp8.run(replacement, plan)
    assert replay[0] is out and replay[1] is lse and replay[2] is plan
    assert len(mxfp8.compilations) == 1 and len(mxfp8.launches) == 2
    for slot, argument in ((0, 0), (1, 1), (3, 3)):
        assert mxfp8.launches[-1][slot].data_ptr() == replacement[argument].data_ptr()
    if compressed:
        assert mxfp8.launches[-1][2].data_ptr() == replacement[2].data_ptr()
        assert mxfp8.launches[-1][4].data_ptr() == replacement[4].data_ptr()
    assert mxfp8.launches[-1][9] is replacement[5]


@pytest.mark.parametrize("compressed", (0, 64, 192, 512))
def test_mxfp8_capture_rejects_unprepared_and_replays_without_allocating(
    mxfp8, monkeypatch, compressed
):
    args = _mxfp8_case(compressed=compressed)

    def forbidden(*args, **kwargs):
        raise AssertionError("capture must not compile or allocate decode buffers")

    with monkeypatch.context() as capture:
        capture.setattr(mxfp8.torch.cuda, "is_current_stream_capturing", lambda: True)
        capture.setattr(mxfp8.module, "_compile", forbidden)
        for name in ("empty", "empty_like", "full"):
            capture.setattr(mxfp8.torch, name, forbidden)
        with pytest.raises(ValueError, match="before CUDA Graph capture"):
            mxfp8.run(args)
    assert not mxfp8.compilations and not mxfp8.launches

    out, lse, plan = mxfp8.run(args)
    replacement = tuple(None if x is None else x.clone() for x in args)
    with monkeypatch.context() as capture:
        capture.setattr(mxfp8.torch.cuda, "is_current_stream_capturing", lambda: True)
        capture.setattr(mxfp8.module, "_compile", forbidden)
        for name in ("empty", "empty_like", "full"):
            capture.setattr(mxfp8.torch, name, forbidden)
        replay = mxfp8.run(replacement, plan)
    assert replay[0] is out and replay[1] is lse and replay[2] is plan
    assert len(mxfp8.compilations) == 1 and len(mxfp8.launches) == 2


@pytest.mark.parametrize("target", ("native", "mxfp8"))
@pytest.mark.parametrize("spoof_signature", (False, True))
def test_native_and_mxfp8_plans_cannot_cross_modes(mxfp8, target, spoof_signature):
    fp8_args, mx_args = _host_case(), _mxfp8_case()
    _, _, fp8_plan = mxfp8.native.run(fp8_args)
    _, _, mx_plan = mxfp8.run(mx_args)
    assert type(fp8_plan) is not type(mx_plan)
    if target == "native":
        run, args, own, wrong = mxfp8.native.run, fp8_args, fp8_plan, mx_plan
    else:
        run, args, own, wrong = mxfp8.run, mx_args, mx_plan, fp8_plan
    if spoof_signature:
        wrong = replace(wrong, signature=own.signature)
        assert wrong.signature == own.signature
    with pytest.raises(ValueError, match="declaration mismatch"):
        run(args, wrong)
    assert len(mxfp8.compilations) == len(mxfp8.native.compilations) == 1
    assert len(mxfp8.launches) == len(mxfp8.native.launches) == 1


@pytest.mark.parametrize("bad", ("object", "signature"))
def test_mxfp8_rejects_foreign_or_changed_declarations(mxfp8, bad):
    args = _mxfp8_case()
    _, _, plan = mxfp8.run(args)
    wrong = object() if bad == "object" else replace(plan, signature=("changed",))
    with pytest.raises(ValueError, match="declaration mismatch"):
        mxfp8.run(args, wrong)
    assert len(mxfp8.compilations) == len(mxfp8.launches) == 1


@pytest.mark.parametrize(
    "field",
    (
        "out",
        "lse",
        "selected_swa",
        "selected_global",
        "swa_lengths",
        "global_lengths",
        "empty_cache",
        "empty_indices",
    ),
)
def test_mxfp8_checks_every_owned_buffer_for_input_overlap(mxfp8, field):
    args = _mxfp8_case()
    _, _, plan = mxfp8.run(args)
    with pytest.raises(ValueError, match="overlap"):
        mxfp8.run(args, replace(plan, **{field: args[0]}))
    assert len(mxfp8.launches) == 1


@pytest.mark.parametrize("field", ("out", "lse"))
def test_mxfp8_checks_floating_owned_buffers_for_autograd(mxfp8, field):
    args = _mxfp8_case()
    _, _, plan = mxfp8.run(args)
    getattr(plan, field).requires_grad_(True)
    with pytest.raises(ValueError, match="inference-only"):
        mxfp8.run(args, plan)
    assert len(mxfp8.launches) == 1


def test_packed_resource_inherits_fp4_conversion_stores_and_pipeline(native):
    from flashinfer.experimental.deepseek_v41 import (
        prims_ts_mxfp8_resources as packed,
        prims_ts_resources as fp4,
    )

    assert issubclass(packed.PackedKvProducer, fp4.Fp4KvProducer)
    assert issubclass(packed.PackedSmemKvResource, fp4.Fp4SmemKvResource)
    assert packed.PackedKvProducer.pipeline_config is fp4.Fp4KvProducer.pipeline_config
    for method in (
        "_store_fragment",
        "_store_fp4_chunk",
        "_load_fp4_stage",
        "_stage_base",
    ):
        assert inspect.getattr_static(packed.PackedSmemKvResource, method) is (
            inspect.getattr_static(fp4.Fp4SmemKvResource, method)
        )
    signatures = {
        producer(native_fp4=flag).compile_signature()
        for producer in (fp4.Fp4KvProducer, packed.PackedKvProducer)
        for flag in (False, True)
    }
    assert signatures == {
        ("dsv41_page64_fp4_group16_primary", False),
        ("dsv41_page64_fp4_group16_primary", True),
        ("dsv41_page64_mxfp8_group32_fp4_group16", False),
        ("dsv41_page64_mxfp8_group32_fp4_group16", True),
    }


@pytest.mark.parametrize("native_fp4", (False, True))
def test_native_primary_and_packed_extra_fp4_pool_bindings(native, native_fp4):
    from flashinfer.experimental.deepseek_v41 import (
        prims_ts_mxfp8_resources as packed,
        prims_ts_resources as fp4,
    )

    cfg = _bf16_packed_config(native)
    primary, extra = object(), object()
    plain = fp4.Fp4KvProducer(native_fp4=native_fp4).make_resource(
        (primary, extra), name="native_fp4", cfg=cfg
    )
    mixed = packed.PackedKvProducer(native_fp4=native_fp4).make_resource(
        (primary, extra), name="packed_mxfp8", cfg=cfg
    )
    assert type(plain) is fp4.Fp4SmemKvResource
    assert type(mixed) is packed.PackedSmemKvResource
    assert plain.fp4_cache is primary and not hasattr(plain, "swa_cache")
    assert mixed.swa_cache is primary and mixed.fp4_cache is extra
    assert plain.native_fp4 is mixed.native_fp4 is native_fp4
    assert plain.cfg is mixed.cfg is cfg
    assert plain.name == "native_fp4" and mixed.name == "packed_mxfp8"


def test_packed_pipeline_inherits_all_writer_async_signaling(native, monkeypatch):
    from flashinfer.experimental.deepseek_v41 import (
        prims_ts_mxfp8_resources as packed,
        prims_ts_resources as fp4,
    )

    cfg = _bf16_packed_config(native)
    calls, pipeline = [], object()
    load_group, umma_group, layout = object(), object(), object()

    def create(**kwargs):
        calls.append(kwargs)
        return pipeline

    monkeypatch.setattr(
        fp4.PipelineConfig, "create_async_umma_pipeline_cfg", staticmethod(create)
    )
    for producer in (fp4.Fp4KvProducer, packed.PackedKvProducer):
        assert (
            producer(native_fp4=True).pipeline_config(
                cfg, load_group, umma_group, layout
            )
            is pipeline
        )
    expected = dict(
        num_stages=8,
        producer_group=load_group,
        consumer_group=umma_group,
        cta_layout_vmnk=layout,
        producer_signaling_threads=fp4.SignalingThreads.All,
        consumer_signaling_threads=fp4.SignalingThreads.CtaLeader,
    )
    assert calls == [expected, expected]


@pytest.mark.parametrize("mixed", (False, True))
def test_packed_stage_fences_every_writer_after_format_routing(native, mixed):
    from flashinfer.experimental.deepseek_v41 import (
        prims_ts_mxfp8_resources as packed,
        prims_ts_resources as fp4,
    )

    cls = packed.PackedSmemKvResource if mixed else fp4.Fp4SmemKvResource
    function = ast.parse(
        textwrap.dedent(inspect.getsource(cls._producer_load_kv_stage))
    ).body[0]
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    fences = [
        node
        for node in calls
        if ast.unparse(node.func) == "cute.arch.fence_view_async_shared"
    ]
    assert len(fences) == 1 and function.body[-1].value is fences[0]
    assert not any(isinstance(node, ast.Return) for node in ast.walk(function))
    assert not any(ast.unparse(node.func).endswith("elect_sync") for node in calls)
    if mixed:
        branch = next(node for node in function.body if isinstance(node, ast.If))
        assert ast.unparse(branch.test) == "cutlass.Int32(first) < 0"
        assert ast.unparse(branch.body[0].value.func) == "self._load_fp4_stage"
        assert ast.unparse(branch.body[0].value.args[0]) == "self.fp4_cache"
        assert ast.unparse(branch.orelse[0].value.func) == "self._load_mxfp8_stage"
        assert ast.unparse(branch.orelse[0].value.args[0]) == "self.swa_cache"


def test_native_and_mxfp8_jit_namespaces_and_source_keys_are_isolated(native):
    from flashinfer.experimental.deepseek_v41 import prims_ts, prims_ts_mxfp8

    assert prims_ts._compile is not prims_ts_mxfp8._compile
    assert prims_ts._Plan is not prims_ts_mxfp8._Plan
    for module, namespace, resource in (
        (prims_ts, "deepseek_v41_prims_ts_fp8", "prims_ts_fp8.__file__"),
        (
            prims_ts_mxfp8,
            "deepseek_v41_prims_ts_mxfp8",
            "prims_ts_mxfp8_resources.__file__",
        ),
    ):
        function = ast.parse(
            textwrap.dedent(inspect.getsource(inspect.unwrap(module._compile)))
        ).body[0]
        builds = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_and_load_cute_dsl_kernel"
        ]
        assert len(builds) == 1 and ast.literal_eval(builds[0].args[0]) == namespace
        files = next(
            kw.value for kw in builds[0].keywords if kw.arg == "extra_key_files"
        )
        if isinstance(files, ast.Name):
            files = next(
                node.value
                for node in function.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == files.id
                    for target in node.targets
                )
            )
        assert {ast.unparse(node) for node in files.elts} == {
            "__file__",
            "prims_ts_resources.__file__",
            "hca_v41_primitives.__file__",
            resource,
            "*sorted(native_root.rglob('*.py'))",
        }
        if module is prims_ts:
            assert "mxfp8" not in ast.unparse(function)

    tree = ast.parse(inspect.getsource(prims_ts))
    assert not any(
        isinstance(node, ast.ImportFrom) and "mxfp8" in (node.module or "")
        for node in tree.body
    )
