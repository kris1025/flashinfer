# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-only contracts for internal DS4.1 FP32 source-state publication.

Run in a fresh process with --confcutdir=tests/experimental to exclude the root
CUDA collection probes. The fixture declares a static target and blocks Torch,
raw CUDA bindings and JIT. Even intercepted calls caught by dependencies fail
acceptance; DSL 4.7 can attempt cuInit during import despite the static target.
"""

import ast
from dataclasses import dataclass, replace
import importlib
import inspect
import os
from types import SimpleNamespace

import pytest


@dataclass(frozen=True)
class _SourceStateStore:
    variant: str = "fp32_source"
    block_size: int = 16

    def compile_signature(self):
        return (self.variant, self.block_size)

    def store(self, args, values, row, column, *, lane_stride=16, rotary=False):
        raise AssertionError("host tests must not execute the source store")


@pytest.fixture
def native(monkeypatch):
    import torch

    visibility = (
        "CUDA_VISIBLE_DEVICES" in os.environ,
        os.getenv("CUDA_VISIBLE_DEVICES"),
    )
    blocked = []

    def forbidden(label):
        def fail(*args, **kwargs):
            blocked.append(label)
            raise AssertionError("host acceptance must not use CUDA or JIT: " + label)

        return fail

    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.0a")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    for name in (
        "init",
        "_lazy_init",
        "current_device",
        "get_device_capability",
        "get_device_properties",
        "synchronize",
        "current_stream",
        "default_stream",
        "is_current_stream_capturing",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden("torch.cuda." + name))

    # Install raw-driver/runtime guards before CuTe can cache imported functions.
    for module_name in (
        "cuda.bindings.driver",
        "cuda.bindings.runtime",
        "cuda.cuda",
        "cuda.cudart",
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if module_name == error.name or module_name.startswith(error.name + "."):
                continue
            raise
        for name in dir(module):
            value = getattr(module, name)
            if (
                name.startswith("cu")
                and callable(value)
                and not isinstance(value, type)
            ):
                monkeypatch.setattr(module, name, forbidden(module_name + "." + name))

    cutlass = pytest.importorskip("cutlass", minversion="4.7.0")
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode

    class CompileGuard:
        def __getitem__(self, frontend):
            return self

        __call__ = staticmethod(forbidden("cute.compile"))

    monkeypatch.setattr(cute, "compile", CompileGuard())
    monkeypatch.setattr(
        cute.runtime, "load_module", forbidden("cute.runtime.load_module")
    )
    monkeypatch.setattr(
        utils.HardwareInfo,
        "get_device_multiprocessor_count",
        forbidden("HardwareInfo.get_device_multiprocessor_count"),
    )
    if not hasattr(cute.nvgpu, "OperandMajorMode"):
        monkeypatch.setattr(
            cute.nvgpu, "OperandMajorMode", OperandMajorMode, raising=False
        )

    from flashinfer.attention.prims_ts.kernels import separate_reduction
    from flashinfer.attention.prims_ts.kernels.mla_decode import sparse_policy
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta import (
        config,
        kernel,
    )

    assert not torch.cuda._initialized
    assert not blocked, f"intercepted CUDA/JIT calls during import: {blocked}"
    try:
        yield SimpleNamespace(
            cutlass=cutlass,
            config=config,
            kernel=kernel,
            profile=sparse_policy.SparseMlaProfile,
            reduction=separate_reduction,
        )
    finally:
        assert not torch.cuda._initialized
        assert not blocked, f"intercepted CUDA/JIT calls: {blocked}"
        assert visibility == (
            "CUDA_VISIBLE_DEVICES" in os.environ,
            os.getenv("CUDA_VISIBLE_DEVICES"),
        )


def _make_kernel(native, **overrides):
    kwargs = dict(
        batch_size=2,
        num_heads=64,
        seq_len_q=1,
        seq_len_k=512,
        latent_dim=512,
        rope_dim=0,
        page_size=1,
        max_active_clusters=148,
        qkv_dtype="e4m3",
        out_dtype="bf16",
        profile=native.config.MlaProfile(
            name="source_state_cpu",
            tile_size_q=64,
            kernel_variant="keeps_mma_ab",
            num_ctas_per_seq_kv=1,
            use_multi_ctas_kv=0,
            use_persistent_scheduler=0,
        ),
        reduction_mode="gmem_separate",
        logical_num_heads=64,
        logical_seq_len_q=1,
        tile_size_q=64,
        mask_type="dense",
        sparse_profile=native.profile(
            family="keep", tile_size_q=64, split_kv=1, reuse_kv=True, reuse_kv_stages=10
        ),
        finalize_output=False,
    )
    kwargs.update(overrides)
    return native.kernel.ThroughputLatencyMlaDecodeTs(**kwargs)


def test_source_state_interface_is_appended_and_optional(native):
    cls = native.kernel.ThroughputLatencyMlaDecodeTs
    for method, name in (
        (cls.__init__, "source_state_store"),
        (cls.__call__, "source_state_store_args"),
    ):
        parameters = inspect.signature(method).parameters
        assert list(parameters)[-1] == name
        assert parameters[name].default is None
    assert inspect.signature(cls.__init__).parameters["source_state_store"].kind == (
        inspect.Parameter.KEYWORD_ONLY
    )


@pytest.mark.parametrize("qkv_dtype", ("bf16", "e4m3"))
def test_source_state_constructor_keeps_log2_nonfinalizing_config(native, qkv_dtype):
    kernel = _make_kernel(
        native, qkv_dtype=qkv_dtype, source_state_store=_SourceStateStore()
    )
    cfg = kernel._make_config()
    assert kernel.direct_sparse
    assert not kernel.finalize_output
    assert kernel.output_quant is None
    assert cfg.sparse_direct and cfg.single_kv_pipe
    assert cfg.kernel_variant == "keeps_mma_ab" and cfg.tile_size_q == 64
    assert (cfg.head_dim_qk, cfg.head_dim_v, cfg.rope_dim) == (512, 512, 0)
    assert (cfg.logical_seq_len_q, cfg.logical_num_heads_q) == (1, 64)
    assert (cfg.num_ctas_per_head_dim, cfg.num_ctas_per_seq_kv) == (1, 1)
    assert not any(
        (
            cfg.fuse_sparse_epilogue,
            cfg.fuse_sparse_cluster_epilogue,
            cfg.fuse_sparse_reduction,
            cfg.use_attention_sinks,
            cfg.use_persistent_scheduler,
            cfg.use_clc_dynamic_persistent_scheduler,
        )
    )


def test_source_state_opt_in_is_required_for_nonfinalizing_direct_inputs(native):
    kernel = _make_kernel(native)
    kernel.direct_sparse = True
    with pytest.raises(ValueError, match="separate sparse indices require"):
        kernel._make_config()
    source = _make_kernel(native, source_state_store=_SourceStateStore())
    assert source._make_config().sparse_direct


@pytest.mark.parametrize(
    "overrides,match",
    (
        ({"sparse_profile": None}, "direct sparse profile"),
        ({"finalize_output": True}, "cannot finalize output"),
        ({"out_dtype": "fp32"}, "dtype"),
        ({"source_state_store": _SourceStateStore(block_size=3)}, "store block size"),
    ),
)
def test_source_state_constructor_rejects_unsupported_modes(native, overrides, match):
    kwargs = {"source_state_store": _SourceStateStore(), **overrides}
    with pytest.raises(ValueError, match=match):
        _make_kernel(native, **kwargs)


@pytest.mark.parametrize(
    "field,value",
    (
        ("kernel_variant", "swaps_mma_ab"),
        ("tile_size_q", 16),
        ("head_dim_qk", 576),
        ("head_dim_v", 256),
        ("rope_dim", 64),
        ("num_tokens_per_page", 32),
        ("logical_num_heads_q", 32),
        ("logical_seq_len_q", 2),
        ("num_ctas_per_head_dim", 2),
        ("num_ctas_per_seq_kv", 2),
        ("use_multi_ctas_kv", 1),
        ("use_cluster_reduction", 1),
        ("use_persistent_scheduler", 1),
        ("use_clc_dynamic_persistent_scheduler", 1),
        ("use_attention_sinks", 1),
        ("fuse_sparse_epilogue", True),
        ("fuse_sparse_cluster_epilogue", True),
        ("fuse_sparse_reduction", True),
        ("sparse_direct", False),
        ("num_insts_kv", 2),
        ("qkv_dtype", "fp16"),
    ),
)
def test_source_state_config_guard(native, field, value):
    cfg = _make_kernel(native, source_state_store=_SourceStateStore())._make_config()
    with pytest.raises(ValueError, match="FP32 source states require"):
        native.config.validate_source_state_store_config(replace(cfg, **{field: value}))


def test_source_state_config_requires_direct_inputs(native):
    kernel = _make_kernel(native, source_state_store=_SourceStateStore())
    kernel.direct_sparse = False
    with pytest.raises(ValueError, match="FP32 source states require"):
        kernel._make_config()


def test_source_state_and_output_quant_are_mutually_exclusive(native):
    kernel = _make_kernel(native, source_state_store=_SourceStateStore())
    cfg = kernel._make_config()
    kernel.output_quant = object()
    with pytest.raises(ValueError, match="mutually exclusive"):
        kernel.compile_signature()
    with pytest.raises(ValueError, match="mutually exclusive"):
        native.kernel.build_throughput_latency_mla_task_manager(
            cfg, source_state_store=_SourceStateStore(), output_quant=object()
        )


def test_source_state_runtime_args_are_separate_and_guarded(native):
    default = _make_kernel(native)
    default._validate_source_state_store_args(None)
    with pytest.raises(ValueError, match="require source_state_store"):
        default._validate_source_state_store_args((object(), object()))
    source = _make_kernel(native, source_state_store=_SourceStateStore())
    with pytest.raises(ValueError, match="requires source_state_store_args"):
        source._validate_source_state_store_args(None)
    args = (object(), object())
    signature = source.compile_signature()
    source._validate_source_state_store_args(args)
    assert source.compile_signature() == signature
    with pytest.raises(ValueError, match="mutually exclusive"):
        source._validate_source_state_store_args(args, output_quant_args=(object(),))


def test_source_state_supports_bf16_packed_producer_signature(native):
    producer = SimpleNamespace(compile_signature=lambda: ("packed_bf16_source",))
    profile = native.profile(
        family="keep",
        tile_size_q=64,
        split_kv=1,
        reuse_kv=True,
        reuse_kv_stages=8,
        kv_tile_size=64,
    )
    source = _make_kernel(
        native,
        qkv_dtype="bf16",
        sparse_profile=profile,
        kv_producer=producer,
        source_state_store=_SourceStateStore(),
    )
    cfg = source._make_config()
    assert cfg.tile_size_kv == 64 and cfg.kv_stages == 8
    signature = source.compile_signature()
    assert len(signature) == 10
    assert signature[-2] == producer.compile_signature()
    assert signature[-1] == (
        "source_state_store",
        _SourceStateStore().compile_signature(),
    )
    default = _make_kernel(native, qkv_dtype="bf16", kv_producer=producer)
    assert len(default.compile_signature()) == 9
    assert default.compile_signature()[-1] == producer.compile_signature()


def test_source_state_compile_signature_is_opt_in_and_batch_independent(native):
    default = _make_kernel(native)
    explicit_none = _make_kernel(native, source_state_store=None)
    assert not default.direct_sparse
    assert default.compile_signature() == explicit_none.compile_signature()
    assert len(default.compile_signature()) == 8
    assert default.compile_signature()[-1] is None
    assert default._make_config() == explicit_none._make_config()

    store = _SourceStateStore()
    source = _make_kernel(native, source_state_store=store)
    other_batch = _make_kernel(
        native, batch_size=3, source_state_store=_SourceStateStore()
    )
    assert source.compile_signature() == other_batch.compile_signature()
    assert len(source.compile_signature()) == 9
    assert source.compile_signature()[-1] == (
        "source_state_store",
        store.compile_signature(),
    )
    changed = _make_kernel(
        native, source_state_store=_SourceStateStore(variant="other")
    )
    assert changed.compile_signature() != source.compile_signature()

    default.output_quant = store
    assert len(default.compile_signature()) == 8
    assert default.compile_signature()[-1] is store


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_source_state_task_resource_plumbing_preserves_ownership(native):
    store = _SourceStateStore()
    kernel = _make_kernel(native, source_state_store=store)
    cfg = kernel._make_config()
    args = (object(), object())
    manager, corr = native.kernel.build_throughput_latency_mla_task_manager(
        cfg,
        total_kv_tiles=cfg.total_kv_tiles,
        use_page_offsets=True,
        source_state_store=store,
        source_state_store_args=args,
    )
    assert manager.tasks
    assert corr.source_state_store is store
    assert corr.source_state_store_args is args
    assert corr.output_quant is None
    assert corr.acc_o_tensor is None and corr.acc_lse_tensor is None
    assert corr.get_smem_requirements() == []
    assert all(task._runtime_body_active is None for task in manager.tasks)


@pytest.mark.parametrize("source", ("part_s", "part_c"))
@pytest.mark.parametrize("bits", (16, 32))
@pytest.mark.parametrize("width", (2, 4))
def test_source_merge_load_alignment_uses_element_size(native, source, bits, width):
    tree = ast.parse(inspect.getsource(native.reduction.MergeMlaSourceStates))
    loads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "load" or not isinstance(node.func.value, ast.Call):
            continue
        pointer = node.func.value
        dtype = next((kw.value for kw in pointer.keywords if kw.arg == "dtype"), None)
        expected_dtype = ast.parse(f"{source}.element_type", mode="eval").body
        if dtype is not None and ast.dump(dtype) == ast.dump(expected_dtype):
            loads.append(node)
    assert len(loads) == 1
    alignment = next(kw.value for kw in loads[0].keywords if kw.arg == "alignment")
    expression = ast.Expression(alignment)
    actual = eval(
        compile(expression, "<source-load-alignment>", "eval"),
        {
            "width": width,
            source: SimpleNamespace(element_type=SimpleNamespace(width=bits)),
        },
    )
    assert actual == width * (bits // 8)


@pytest.fixture
def adapter(native, monkeypatch):
    from contextlib import nullcontext
    import torch
    from flashinfer.experimental.deepseek_v41 import prims_ts

    launches, compilations = [], []

    def compile_stub(device, compressed):
        compilations.append((device, compressed))
        return lambda *args: launches.append(args)

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(prims_ts, "_compile", compile_stub)

    def run(args, plan=None):
        signature = tuple(
            None
            if x is None
            else (tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
            for x in args
        )
        return prims_ts._decode_fp8(*args, signature=signature, plan=plan)

    return SimpleNamespace(
        native=native,
        torch=torch,
        module=prims_ts,
        launches=launches,
        compilations=compilations,
        run=run,
    )


def _host_case(batch=2, compressed=64):
    import torch

    q = torch.zeros(batch, 1, 64, 512).to(torch.float8_e4m3fn)
    swa = torch.zeros(2, 64, 1, 512).to(torch.float8_e4m3fn)
    main = torch.zeros(2, 64, 1, 288, dtype=torch.uint8) if compressed else None
    wi = torch.full((batch, 1, 128), -1, dtype=torch.int32)
    ci = (
        torch.full((batch, 1, compressed), -1, dtype=torch.int32)
        if compressed
        else None
    )
    sink = torch.full((64,), -float("inf"))
    return q, swa, main, wi, ci, sink


def test_fp8_entry_does_not_import_mxfp8(native, monkeypatch):
    import builtins
    import torch
    from flashinfer.experimental.deepseek_v41 import prims_ts

    imports, calls = [], []
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if "prims_ts_mxfp8" in name or any(
            "prims_ts_mxfp8" in item for item in fromlist or ()
        ):
            imports.append(name)
            raise AssertionError("native FP8 must not import MXFP8 support")
        return original_import(name, globals, locals, fromlist, level)

    def validate(*args, **kwargs):
        assert kwargs == {"allow_fp8_swa": True}
        return ("native",)

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return ("out", "lse", "plan")

    monkeypatch.setattr(prims_ts, "_validate_inputs", validate)
    monkeypatch.setattr(prims_ts, "_decode_fp8", run)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    args = (SimpleNamespace(dtype=torch.float8_e4m3fn), None, None, None, None, None)
    assert prims_ts.decode(*args) == ("out", "lse", "plan")
    assert calls == [(args, {"signature": ("native",), "plan": None})]
    assert not imports


def test_prims_ts_requires_explicit_fp8_q(native, monkeypatch):
    import torch
    from flashinfer.experimental.deepseek_v41 import prims_ts

    monkeypatch.setattr(prims_ts, "_validate_inputs", lambda *args, **kwargs: ())
    with pytest.raises(ValueError, match="requires explicit E4M3 Q"):
        prims_ts.decode(
            SimpleNamespace(dtype=torch.bfloat16), None, None, None, None, None
        )


@pytest.mark.parametrize("compressed", list(range(0, 513, 64)))
@pytest.mark.parametrize("native_fp4", (False, True))
def test_fp8_adapter_builds_native_swa_and_fp4_only_global(
    native, compressed, native_fp4
):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_sources

    swa, main = _make_fp8_sources(compressed, 148, native_fp4)
    cfg = swa._make_config()
    assert cfg.qkv_dtype == "e4m3" and cfg.is_fp8_qkv()
    assert (
        cfg.tile_size_q,
        cfg.tile_size_kv,
        cfg.head_dim_per_stage_kv,
        cfg.kv_stages,
    ) == (64, 128, 128, 8)
    assert cfg.sparse_direct and cfg.sparse_reuse_kv and cfg.single_kv_pipe
    assert cfg.num_ctas_per_head_dim == cfg.num_ctas_per_seq_kv == 1
    assert not cfg.fuse_sparse_epilogue and not cfg.use_persistent_scheduler
    assert swa.kv_producer is None and swa.direct_static_scales
    assert swa.direct_sparse_capacities == (128, 0)
    signature = swa.compile_signature()
    assert signature[-1] == (
        "source_state_store",
        swa.source_state_store.compile_signature(),
    )
    if compressed:
        cfg = main._make_config()
        assert (cfg.qkv_dtype, cfg.tile_size_kv, cfg.kv_stages) == ("bf16", 64, 8)
        assert main.direct_sparse_capacities == (compressed, 0)
        from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
            Fp4KvProducer,
        )

        assert type(main.kv_producer) is Fp4KvProducer
        assert main.kv_producer.native_fp4 == native_fp4
        assert (
            main.kv_producer.compile_signature()[0]
            == "dsv41_page64_fp4_group16_primary"
        )
        assert (
            main.source_state_store.compile_signature()
            == swa.source_state_store.compile_signature()
        )
        assert main.compile_signature() != signature
    else:
        assert main is None
    swa.source_state_store = _SourceStateStore(variant="changed_source_store")
    assert swa.compile_signature() != signature


@pytest.mark.parametrize("compressed", (0, 64, 192, 512))
def test_fp8_adapter_owned_shapes_and_window_aliases(adapter, compressed):
    torch = adapter.torch
    args = _host_case(compressed=compressed)
    out, lse, plan = adapter.run(args)
    assert out.dtype == torch.bfloat16 and out.shape == args[0].shape
    assert lse.dtype == torch.float32 and lse.shape == (2, 64, 1)
    assert plan.part_s.dtype == torch.float32 and plan.part_s.shape == (2, 64, 512)
    assert plan.lse_s.dtype == torch.float32 and plan.lse_s.shape == (2, 64)
    assert plan.empty_lengths.dtype == torch.int32 and plan.empty_lengths.shape == (2,)
    assert adapter.compilations == [(None, compressed)]
    call = adapter.launches[-1]
    assert len(call) == 20
    assert call[0].shape == (64, 512, 1, 2)
    assert call[2].shape == (1, 512, 128) and call[2].stride() == (512, 1, 512)
    assert call[10] is plan.empty_lengths
    assert call[11] is plan.swa_counts and call[12] is plan.global_counts
    assert call[13] is plan.part_s and call[15] is plan.lse_s
    assert (
        call[18].data_ptr() == out.data_ptr() and call[19].data_ptr() == lse.data_ptr()
    )
    if compressed:
        assert (
            plan.expanded_q.dtype == torch.bfloat16
            and plan.expanded_q.shape == args[0].shape
        )
        assert plan.part_c.dtype == torch.float32 and plan.lse_c.dtype == torch.float32
        assert call[1].data_ptr() == plan.expanded_q.data_ptr()
        assert call[14] is plan.part_c and call[16] is plan.lse_c
    else:
        assert plan.expanded_q is None and plan.part_c is None and plan.lse_c is None
        assert call[1].data_ptr() == out.data_ptr()
        assert call[14] is plan.part_s and call[16] is plan.lse_s
        assert call[3].data_ptr() == plan.empty_cache.data_ptr()
        assert call[5].data_ptr() == plan.empty_indices.data_ptr()
    replacement = tuple(None if x is None else x.clone() for x in args)
    replay = adapter.run(replacement, plan)
    assert replay[0] is out and replay[1] is lse and replay[2] is plan
    assert len(adapter.compilations) == 1 and len(adapter.launches) == 2
    assert adapter.launches[-1][0].data_ptr() == replacement[0].data_ptr()


@pytest.mark.parametrize(
    "field",
    (
        "out",
        "lse",
        "selected_swa",
        "selected_global",
        "swa_lengths",
        "global_lengths",
        "empty_lengths",
        "swa_counts",
        "global_counts",
        "part_s",
        "part_c",
        "lse_s",
        "lse_c",
        "expanded_q",
        "empty_cache",
        "empty_indices",
    ),
)
def test_fp8_adapter_checks_every_owned_buffer_for_input_overlap(adapter, field):
    args = _host_case()
    _, _, plan = adapter.run(args)
    with pytest.raises(ValueError, match="overlap"):
        adapter.run(args, replace(plan, **{field: args[0]}))
    assert len(adapter.launches) == 1


@pytest.mark.parametrize(
    "field", ("out", "lse", "part_s", "part_c", "lse_s", "lse_c", "expanded_q")
)
def test_fp8_adapter_checks_floating_scratch_for_autograd(adapter, field):
    args = _host_case()
    _, _, plan = adapter.run(args)
    getattr(plan, field).requires_grad_(True)
    with pytest.raises(ValueError, match="inference-only"):
        adapter.run(args, plan)
    assert len(adapter.launches) == 1


def test_fp8_adapter_rejects_unprepared_capture_and_wrong_plans(adapter, monkeypatch):
    args = _host_case()
    _, _, plan = adapter.run(args)
    for bad in (object(), replace(plan, signature=("different_mode",))):
        with pytest.raises(ValueError, match="declaration mismatch"):
            adapter.run(args, bad)
    monkeypatch.setattr(adapter.torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="before CUDA Graph capture"):
        adapter.run(args)
    assert len(adapter.compilations) == 1 and len(adapter.launches) == 1
    adapter.run(args, plan)
    assert len(adapter.compilations) == 1 and len(adapter.launches) == 2


class _MetadataTensor:
    def __init__(
        self,
        shape,
        dtype,
        *,
        contiguous=True,
        pointer=16,
        requires_grad=False,
        device=None,
    ):
        import torch

        self.shape, self.dtype = tuple(shape), dtype
        self.ndim = len(shape)
        self.device = torch.device("cuda:0") if device is None else device
        self.requires_grad = requires_grad
        self.contiguous, self.pointer = contiguous, pointer

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.pointer

    def stride(self):
        result = [1] * self.ndim
        for dim in range(self.ndim - 2, -1, -1):
            result[dim] = result[dim + 1] * self.shape[dim + 1]
        return tuple(result)


def _metadata_case(fp8=True, compressed=64):
    import torch

    return [
        _MetadataTensor(
            (2, 1, 64, 512), torch.float8_e4m3fn if fp8 else torch.bfloat16
        ),
        _MetadataTensor(
            (2, 64, 1, 512 if fp8 else 528), torch.float8_e4m3fn if fp8 else torch.uint8
        ),
        _MetadataTensor((2, 64, 1, 288), torch.uint8) if compressed else None,
        _MetadataTensor((2, 1, 128), torch.int32),
        _MetadataTensor((2, 1, compressed), torch.int32) if compressed else None,
        _MetadataTensor((64,), torch.float32),
    ]


@pytest.mark.parametrize("compressed", list(range(0, 513, 64)))
def test_fp8_shared_validator_opt_in_preserves_frost(native, monkeypatch, compressed):
    import torch
    from flashinfer.experimental.deepseek_v41.decode import _validate_inputs

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    legacy, fp8 = _metadata_case(False, compressed), _metadata_case(True, compressed)
    assert _validate_inputs(*legacy) == _validate_inputs(*legacy, allow_fp8_swa=True)
    signature = _validate_inputs(*fp8, allow_fp8_swa=True)
    assert signature[0][2] == signature[1][2] == torch.float8_e4m3fn
    with pytest.raises(ValueError, match="Q must be BF16"):
        _validate_inputs(*fp8)
    mixed = fp8.copy()
    mixed[0] = legacy[0]
    with pytest.raises(ValueError, match="uint8 page pool"):
        _validate_inputs(*mixed, allow_fp8_swa=True)
    mixed[0], mixed[1] = fp8[0], legacy[1]
    with pytest.raises(ValueError, match="E4M3 SWA"):
        _validate_inputs(*mixed, allow_fp8_swa=True)


@pytest.mark.parametrize(
    "bad",
    (
        "q_fp16",
        "q_b0",
        "q_b65536",
        "q_sq2",
        "q_h32",
        "q_d256",
        "swa_none",
        "swa_width528",
        "swa_bf16",
        "global_only_cache",
        "global_only_ids",
        "global_width512",
        "global_k63",
        "global_k513",
        "global_ids_int64",
        "swa_ids_k127",
        "swa_ids_int64",
        "sink_h63",
        "sink_bf16",
        "alignment",
        "noncontiguous",
        "device",
        "autograd",
    ),
)
def test_fp8_shared_validator_rejects_bad_declarations(native, monkeypatch, bad):
    import torch
    from flashinfer.experimental.deepseek_v41.decode import _validate_inputs

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    args = _metadata_case()
    if bad == "q_fp16":
        args[0].dtype = torch.float16
    elif bad.startswith("q_"):
        shape = list(args[0].shape)
        axis, value = {
            "q_b0": (0, 0),
            "q_b65536": (0, 65536),
            "q_sq2": (1, 2),
            "q_h32": (2, 32),
            "q_d256": (3, 256),
        }[bad]
        shape[axis] = value
        args[0] = _MetadataTensor(shape, args[0].dtype)
    elif bad == "swa_none":
        args[1] = None
    elif bad == "swa_width528":
        args[1] = _MetadataTensor((2, 64, 1, 528), args[1].dtype)
    elif bad == "swa_bf16":
        args[1].dtype = torch.bfloat16
    elif bad == "global_only_cache":
        args[4] = None
    elif bad == "global_only_ids":
        args[2] = None
    elif bad == "global_width512":
        args[2] = _MetadataTensor((2, 64, 1, 512), torch.uint8)
    elif bad in ("global_k63", "global_k513"):
        args[4] = _MetadataTensor(
            (2, 1, 63 if bad == "global_k63" else 513), torch.int32
        )
    elif bad == "global_ids_int64":
        args[4].dtype = torch.int64
    elif bad == "swa_ids_k127":
        args[3] = _MetadataTensor((2, 1, 127), torch.int32)
    elif bad == "swa_ids_int64":
        args[3].dtype = torch.int64
    elif bad == "sink_h63":
        args[5] = _MetadataTensor((63,), torch.float32)
    elif bad == "sink_bf16":
        args[5].dtype = torch.bfloat16
    elif bad == "alignment":
        args[1].pointer = 17
    elif bad == "noncontiguous":
        args[0].contiguous = False
    elif bad == "device":
        args[1].device = torch.device("cuda:1")
    else:
        args[5].requires_grad = True
    with pytest.raises(ValueError):
        _validate_inputs(*args, allow_fp8_swa=True)


def test_fp8_oracle_independent_page_planes_duplicates_and_bound(native):
    import math
    import torch
    from .test_deepseek_v41_decode import check_fp8, fp8_oracle

    q, swa, main, wi, ci, sink = _host_case(batch=1)
    swa = torch.full(swa.shape, 0.3125).to(torch.float8_e4m3fn)
    swa.view(torch.uint8).reshape(-1, 512)[1:].fill_(127)
    pages = main.view(2, -1)
    pages.fill_(255)
    pages[0, :256].fill_(0x63)  # low=1.5, high=4
    pages[0, 64 * 256 : 64 * 256 + 32].fill_(0x38)  # scale=1
    pages[1, :256].fill_(0x8C)  # low=-2, high=-0
    pages[1, 64 * 256 : 64 * 256 + 32].fill_(0x40)  # scale=2
    wi[..., :5] = torch.tensor([0, 0, -1, 128, 0x7FFFFFFF], dtype=torch.int32)
    ci[..., :6] = torch.tensor([0, 64, 64, -1, 128, 0x7FFFFFFF], dtype=torch.int32)
    expected, lse, bound = fp8_oracle(q, swa, main, wi, ci, sink)
    answer = torch.empty(1, 64, 512, dtype=torch.float64)
    answer[..., ::2] = (2 * 0.3125 + 1.5 - 4 - 4) / 5
    answer[..., 1::2] = (2 * 0.3125 + 4) / 5
    sensitivity = torch.empty_like(answer)
    sensitivity[..., ::2] = (2 * 0.3125 + 1.5 + 4 + 4) / 5
    sensitivity[..., 1::2] = (2 * 0.3125 + 4) / 5
    analytic_bound = ((2**-4 + 3 * 2**-8 + 1e-5) + 2**-10 / 448) * sensitivity + 1e-6
    torch.testing.assert_close(expected, answer, rtol=0, atol=1e-14)
    torch.testing.assert_close(
        lse, torch.full_like(lse, math.log(5)), rtol=0, atol=1e-14
    )
    torch.testing.assert_close(bound, analytic_bound, rtol=0, atol=1e-14)
    out, public_lse = expected.bfloat16()[:, None], lse.float()[..., None]
    check_fp8(out, public_lse, (expected, lse, bound))
    wrong = out.clone()
    wrong[0, 0, 0, 0] += 1
    with pytest.raises(AssertionError):
        check_fp8(wrong, public_lse, (expected, lse, bound))
    with pytest.raises(AssertionError):
        check_fp8(out, public_lse + 1e-3, (expected, lse, bound))


def test_fp8_oracle_uses_stored_query_and_sink_in_bound(native):
    import math
    import torch
    from .test_deepseek_v41_decode import fp8_oracle

    q, swa, _, wi, _, sink = _host_case(batch=1, compressed=0)
    q.view(torch.uint8)[..., 0] = torch.tensor(0.3).to(q.dtype).view(torch.uint8)
    swa = torch.zeros(swa.shape)
    swa.view(-1, 512)[0].fill_(0.625)
    swa.view(-1, 512)[1].fill_(0.15625)
    swa = swa.to(q.dtype)
    wi[..., :2] = torch.tensor([0, 1], dtype=torch.int32)
    sink.fill_(1)
    expected, lse, bound = fp8_oracle(q, swa, None, wi, None, sink)
    query = q[0, 0, 0, 0].double().item()
    logits = torch.tensor(
        [query * 0.625 / math.sqrt(512), query * 0.15625 / math.sqrt(512), 1],
        dtype=torch.float64,
    )
    weights = logits.softmax(0)[:2]
    answer = weights[0] * 0.625 + weights[1] * 0.15625
    underflow = torch.exp(logits[:2].max() - logits.logsumexp(0)) * (0.625 + 0.15625)
    analytic_bound = (
        (2**-4 + 3 * 2**-8 + 1e-5) * answer + (2**-10 / 448) * underflow + 1e-6
    )
    torch.testing.assert_close(
        expected, torch.full_like(expected, answer.item()), rtol=0, atol=1e-14
    )
    torch.testing.assert_close(
        lse, torch.full_like(lse, logits[:2].logsumexp(0).item()), rtol=0, atol=1e-14
    )
    torch.testing.assert_close(
        bound, torch.full_like(bound, analytic_bound.item()), rtol=0, atol=1e-14
    )
    assert query != 0.3


def test_fp8_oracle_empty_poison_and_exact_empty_gate(native):
    import torch
    from .test_deepseek_v41_decode import check_fp8, fp8_oracle

    args = _host_case(batch=2)
    args[1].view(torch.uint8).fill_(127)
    args[2].fill_(255)
    args[5][::2] = torch.finfo(torch.float32).max
    expected, lse, bound = fp8_oracle(*args)
    assert torch.equal(expected, torch.zeros_like(expected))
    assert torch.equal(bound, torch.zeros_like(bound)) and torch.isneginf(lse).all()
    out, public_lse = expected.bfloat16()[:, None], lse.float()[..., None]
    check_fp8(out, public_lse, (expected, lse, bound))
    with pytest.raises(AssertionError):
        check_fp8(torch.ones_like(out), public_lse, (expected, lse, bound))
    with pytest.raises(AssertionError):
        check_fp8(out, torch.zeros_like(public_lse), (expected, lse, bound))


def test_fp8_oracle_fp32_source_precision_target(native):
    import math
    import torch
    from .test_deepseek_v41_decode import check_fp8, fp8_oracle

    q, swa, _, wi, _, sink = _host_case(batch=1, compressed=0)
    swa.view(torch.uint8).reshape(-1, 512)[:19].fill_(0x38)
    wi[..., :65] = torch.arange(65, dtype=torch.int32)
    reference = fp8_oracle(q, swa, None, wi, None, sink)
    torch.testing.assert_close(
        reference[0], torch.full_like(reference[0], 19 / 65), rtol=0, atol=1e-14
    )
    torch.testing.assert_close(
        reference[1], torch.full_like(reference[1], math.log(65)), rtol=0, atol=1e-14
    )
    assert (reference[0].bfloat16().double() - reference[0]).abs().min() > 1e-7
    check_fp8(
        reference[0].bfloat16()[:, None], reference[1].float()[..., None], reference
    )
