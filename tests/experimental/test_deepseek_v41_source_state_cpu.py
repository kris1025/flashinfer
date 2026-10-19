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
import math
import os
import struct
import textwrap
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
    parameters = inspect.signature(cls.__init__).parameters
    assert list(parameters)[-1] == "source_state_store"
    assert parameters["source_state_store"].default is None
    assert parameters["source_state_store"].kind == inspect.Parameter.KEYWORD_ONLY
    for method in (
        cls.__call__,
        cls.dense_kernel,
        native.kernel.build_throughput_latency_mla_task_manager,
        native.kernel._make_single_kv_pipe_task_graph,
    ):
        parameters = inspect.signature(method).parameters
        assert list(parameters)[-2:] == [
            "source_state_store_args",
            "kv_producer_runtime_args",
        ]
        assert parameters["source_state_store_args"].default is None
        assert parameters["kv_producer_runtime_args"].default is None


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


@pytest.mark.parametrize("qkv_dtype", ("bf16", "e4m3"))
@pytest.mark.parametrize("source_store", (False, True))
def test_request_kv_scale_defaults_off_without_changing_old_signatures(
    native, qkv_dtype, source_store
):
    kwargs = dict(
        qkv_dtype=qkv_dtype,
        source_state_store=_SourceStateStore() if source_store else None,
    )
    kernel = _make_kernel(native, **kwargs)
    explicit = _make_kernel(native, **kwargs)
    explicit.direct_request_kv_scale = False
    assert kernel.direct_request_kv_scale is False
    assert kernel._make_config().sparse_request_kv_scale is False
    assert kernel.compile_signature() == explicit.compile_signature()
    parameter = inspect.signature(native.config.configure_sparse_mla).parameters[
        "request_kv_scale"
    ]
    assert (
        parameter.default is False and parameter.kind == inspect.Parameter.KEYWORD_ONLY
    )


def test_request_kv_scale_is_part_of_the_batch_independent_compile_identity(native):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_sources

    _, main = _make_fp8_sources(192, 148, True)
    scaled = main.compile_signature()
    main.batch_size = 3
    assert main.compile_signature() == scaled
    main.direct_request_kv_scale = False
    assert main.compile_signature() != scaled
    assert main._make_config().sparse_static_scales


@pytest.mark.parametrize(
    "overrides",
    (
        {"static_scales": False},
        {"direct_inputs": False},
        {"source_state_store": False},
        {"qkv_dtype": "bf16"},
    ),
)
def test_request_kv_scale_config_requires_static_direct_e4m3_sources(native, overrides):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_sources

    _, main = _make_fp8_sources(64, 148, True)
    cfg = main._make_config()
    options = dict(
        static_scales=True,
        direct_inputs=True,
        source_state_store=True,
        request_kv_scale=True,
    )
    if "qkv_dtype" in overrides:
        cfg = replace(cfg, **overrides)
    else:
        options.update(overrides)
    with pytest.raises(ValueError, match="static-base-scale direct E4M3 source states"):
        native.config.configure_sparse_mla(cfg, main.sparse_profile, **options)


@pytest.mark.parametrize("missing", ("producer", "source", "both"))
@pytest.mark.parametrize("entry", ("config", "task_manager"))
def test_request_kv_scale_requires_custom_producer_and_source_store(
    native, missing, entry
):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_sources

    _, main = _make_fp8_sources(64, 148, True)
    cfg = main._make_config()
    producer = None if missing in ("producer", "both") else main.kv_producer
    store = None if missing in ("source", "both") else main.source_state_store
    with pytest.raises(ValueError, match="custom producer and source store"):
        if entry == "config":
            main.kv_producer, main.source_state_store = producer, store
            main._make_config()
        else:
            native.kernel.build_throughput_latency_mla_task_manager(
                cfg, kv_producer=producer, source_state_store=store
            )


@pytest.mark.parametrize(
    "field,value",
    (
        ("num_ctas_per_seq_kv", 2),
        ("num_ctas_per_head_dim", 2),
        ("use_multi_ctas_kv", 1),
        ("use_cluster_reduction", 1),
        ("use_persistent_scheduler", 1),
        ("use_clc_dynamic_persistent_scheduler", 1),
        ("use_attention_sinks", 1),
        ("fuse_sparse_epilogue", True),
        ("fuse_sparse_cluster_epilogue", True),
        ("fuse_sparse_reduction", True),
    ),
)
def test_request_kv_scale_retains_nonfinalizing_source_topology_guards(
    native, field, value
):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_sources

    _, main = _make_fp8_sources(64, 148, True)
    cfg = replace(main._make_config(), **{field: value})
    assert cfg.sparse_request_kv_scale
    with pytest.raises(ValueError, match="FP32 source states require"):
        native.kernel.build_throughput_latency_mla_task_manager(
            cfg,
            kv_producer=main.kv_producer,
            source_state_store=main.source_state_store,
        )


def _function_ast(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def _host_function(function, namespace):
    """Execute an inspected host prefix, never its decorated DSL entry point."""
    function.decorator_list = []
    function.returns = None
    for arg in (
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
        function.args.vararg,
        function.args.kwarg,
    ):
        if arg is not None:
            arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "<host-contract>", "exec"), namespace)
    return namespace[function.name]


def _scale_metadata(native, **overrides):
    values = dict(
        element_type=native.cutlass.Float32,
        shape=(2,),
        stride=(1,),
        iterator=SimpleNamespace(toint=lambda: 0x100),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_request_scale_guard(native, kernel, scale, runtime_args, *, compact_q=None):
    function = _function_ast(native.kernel.ThroughputLatencyMlaDecodeTs.__call__)
    first_tma = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "tma_box0"
    )
    function.body = function.body[:first_tma]
    checks = []

    def runtime_assert(condition, message):
        checks.append(message)
        assert condition, message

    guard = _host_function(
        function,
        dict(
            cutlass=SimpleNamespace(
                const_expr=lambda value: value, Float32=native.cutlass.Float32
            ),
            runtime_assert=runtime_assert,
        ),
    )
    args = dict.fromkeys(inspect.signature(guard).parameters)
    args.update(
        self=kernel,
        q_latent=SimpleNamespace(shape=(64, 512, 1, 2)),
        block_split_kvs=(None,) * 6 + (scale, None, None),
        source_state_store_args=(object(), object()),
        kv_producer_runtime_args=runtime_args,
        cu_seqlens_q=compact_q,
    )
    guard(**args)
    return checks


def test_request_kv_scale_runtime_guard_requires_one_shared_contiguous_fp32_batch(
    native,
):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_sources

    _, main = _make_fp8_sources(64, 148, True)
    scale = _scale_metadata(native)
    assert _run_request_scale_guard(native, main, scale, (scale,)) == [
        "request KV scales must be contiguous",
        "request KV scales must match the batch size",
        "attention and producer must share request KV scales",
    ]


@pytest.mark.parametrize(
    "bad,error,match",
    (
        ("compact_q", ValueError, "fixed Q and one producer scale"),
        ("missing", ValueError, "fixed Q and one producer scale"),
        ("empty", ValueError, "fixed Q and one producer scale"),
        ("extra", ValueError, "fixed Q and one producer scale"),
        ("none", ValueError, "contiguous FP32"),
        ("dtype", ValueError, "contiguous FP32"),
        ("rank0", ValueError, "contiguous FP32"),
        ("rank2", ValueError, "contiguous FP32"),
        ("stride", AssertionError, "must be contiguous"),
        ("batch", AssertionError, "match the batch size"),
        ("pointer", AssertionError, "must share request KV scales"),
    ),
)
def test_request_kv_scale_runtime_guard_rejects_invalid_bindings(
    native, bad, error, match
):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_sources

    _, main = _make_fp8_sources(64, 148, True)
    scale = _scale_metadata(native)
    compact_q, runtime_args = None, (scale,)
    if bad == "compact_q":
        compact_q = object()
    elif bad == "missing":
        runtime_args = None
    elif bad == "empty":
        runtime_args = ()
    elif bad == "extra":
        runtime_args = (scale, scale)
    elif bad == "none":
        scale = None
    elif bad == "dtype":
        scale.element_type = native.cutlass.BFloat16
    elif bad == "rank0":
        scale.shape = ()
    elif bad == "rank2":
        scale.shape = (2, 1)
    elif bad == "stride":
        scale.stride = (2,)
    elif bad == "batch":
        scale.shape = (3,)
    else:
        runtime_args = (
            _scale_metadata(native, iterator=SimpleNamespace(toint=lambda: 0x200)),
        )
    with pytest.raises(error, match=match):
        _run_request_scale_guard(native, main, scale, runtime_args, compact_q=compact_q)


def test_kv_producer_runtime_args_cannot_reach_default_native_resources(native):
    kernel = _make_kernel(native, source_state_store=_SourceStateStore())
    with pytest.raises(ValueError, match="runtime arguments require a custom producer"):
        _run_request_scale_guard(native, kernel, None, (object(),))
    with pytest.raises(ValueError, match="runtime arguments require a custom producer"):
        native.kernel.build_throughput_latency_mla_task_manager(
            kernel._make_config(), kv_producer_runtime_args=(object(),)
        )


@pytest.mark.parametrize("mode", ("request", "static", "dynamic", "device", "host"))
def test_request_scale_restores_both_factors_without_changing_old_scale_paths(
    native, mode
):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_sources

    _, main = _make_fp8_sources(64, 148, True)
    cfg = main._make_config()
    function = _function_ast(native.kernel.ThroughputLatencyMlaDecodeTs.dense_kernel)
    branch = next(
        node
        for node in function.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(cfg.sparse_direct)"
    )
    # Only this host branch is executed. Substitute the view binder so even
    # the function-local import cannot enter a native/DSL execution path.
    branch.body = [node for node in branch.body if not isinstance(node, ast.ImportFrom)]
    function.name = "scale_factors"
    function.args = (
        ast.parse(
            "def scale_factors(self, cfg, sparse_scale_params, softmax_scale_log2, output_scale): pass"
        )
        .body[0]
        .args
    )
    function.body = [
        branch,
        ast.Return(
            value=ast.Tuple(
                elts=[
                    ast.Name(id=name, ctx=ast.Load())
                    for name in (
                        "softmax_scale_log2",
                        "output_scale",
                        "sparse_epilogue_params",
                    )
                ],
                ctx=ast.Load(),
            )
        ),
    ]
    reads, bindings = [], []

    class Scale(list):
        def __init__(self, name, values):
            super().__init__(values)
            self.name = name

        def __getitem__(self, index):
            reads.append((self.name, index))
            return super().__getitem__(index)

    def bind(*args, **kwargs):
        bindings.append((args, kwargs))
        return object(), object()

    log2_e = math.log2(math.e)
    factors = _host_function(
        function,
        dict(
            cutlass=SimpleNamespace(
                const_expr=lambda value: value, Float32=float, Int64=int
            ),
            cute=SimpleNamespace(arch=SimpleNamespace(block_idx=lambda: (0, 0, 1))),
            bind_sparse_views=bind,
            LOG2_E=log2_e,
        ),
    )
    sinks = object()
    si, ci, sl, cl = object(), object(), object(), object()
    if mode == "request":
        params = (si, ci, sl, cl, None, None, Scale("ss", (3.0, 6.0)), None, sinks)
        expected, expected_reads = (1.5, 15.0, sinks), [("ss", 1)]
    elif mode == "static":
        cfg = replace(cfg, sparse_request_kv_scale=False)
        params = (si, ci, sl, cl, None, None, None, None, sinks)
        expected, expected_reads = (0.25, 2.5, sinks), []
    elif mode == "dynamic":
        cfg = replace(cfg, sparse_static_scales=False, sparse_request_kv_scale=False)
        params = (
            si,
            ci,
            sl,
            cl,
            Scale("sm", (0.125,)),
            Scale("qs", (2.0,)),
            Scale("ss", (3.0,)),
            Scale("os", (5.0,)),
            sinks,
        )
        expected = (0.125 * 2.0 * 3.0 * log2_e, 15.0, sinks)
        expected_reads = [("sm", 0), ("qs", 0), ("ss", 0), ("os", 0), ("ss", 0)]
    else:
        cfg = replace(cfg, sparse_direct=False, sparse_request_kv_scale=False)
        params = Scale("device", (0.75, 3.0)) if mode == "device" else None
        expected = (
            (0.75 * log2_e, 3.0, params) if mode == "device" else (0.25, 2.5, None)
        )
        expected_reads = [("device", 0), ("device", 1)] if mode == "device" else []
    actual = factors(
        SimpleNamespace(device_scales=mode == "device"), cfg, params, 0.25, 2.5
    )
    assert actual == expected and reads == expected_reads
    if cfg.sparse_direct:
        assert bindings == [
            ((si, ci, sl, cl, cfg.total_kv_tiles * cfg.tile_size_kv), {"request": 1})
        ]
    else:
        assert not bindings


def test_kv_producer_runtime_args_are_forwarded_through_every_kernel_layer(native):
    cls = native.kernel.ThroughputLatencyMlaDecodeTs
    for function, callee, positional in (
        (cls.__call__, "self.dense_kernel", True),
        (cls.dense_kernel, "build_throughput_latency_mla_task_manager", False),
        (
            native.kernel.build_throughput_latency_mla_task_manager,
            "_make_single_kv_pipe_task_graph",
            False,
        ),
    ):
        calls = [
            node
            for node in ast.walk(_function_ast(function))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == callee
        ]
        assert len(calls) == 1
        argument = (
            calls[0].args[-1]
            if positional
            else next(
                kw.value
                for kw in calls[0].keywords
                if kw.arg == "kv_producer_runtime_args"
            )
        )
        assert ast.unparse(argument) == "kv_producer_runtime_args"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_request_scale_runtime_args_reach_the_single_pipe_resource(native, monkeypatch):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_sources

    _, main = _make_fp8_sources(192, 148, True)
    cfg = main._make_config()
    producer = main.kv_producer
    factory = type(producer).make_resource
    calls = []

    def make_resource(self, pools, **kwargs):
        resource = factory(self, pools, **kwargs)
        calls.append((pools, kwargs, resource))
        return resource

    monkeypatch.setattr(type(producer), "make_resource", make_resource)
    pools, runtime_args, source_args = (
        (object(), object()),
        (object(),),
        (object(), object()),
    )
    signature = main.compile_signature()
    manager, corr = native.kernel.build_throughput_latency_mla_task_manager(
        cfg,
        total_kv_tiles=cfg.total_kv_tiles,
        use_page_offsets=True,
        batch_idx=1,
        kv_producer=producer,
        kv_producer_args=pools,
        kv_producer_runtime_args=runtime_args,
        source_state_store=main.source_state_store,
        source_state_store_args=source_args,
    )
    assert manager.tasks and len(calls) == 1
    actual_pools, kwargs, resource = calls[0]
    assert actual_pools is pools and kwargs["runtime_args"] is runtime_args
    assert resource.main_scale is runtime_args[0] and resource.batch_idx == 1
    assert resource.fp4_cache is pools[0] and resource.cfg is cfg
    assert corr.source_state_store_args is source_args
    assert main.compile_signature() == signature


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
    assert not swa.direct_request_kv_scale and not cfg.sparse_request_kv_scale
    assert cfg.sparse_static_scales
    assert swa.direct_sparse_capacities == (128, 0)
    signature = swa.compile_signature()
    assert signature[-1] == (
        "source_state_store",
        swa.source_state_store.compile_signature(),
    )
    if compressed:
        cfg = main._make_config()
        assert (cfg.qkv_dtype, cfg.tile_size_kv, cfg.kv_stages) == ("e4m3", 128, 8)
        assert cfg.is_fp8_qkv() and cfg.head_dim_per_stage_kv == 128
        assert main.direct_static_scales and main.direct_request_kv_scale
        assert cfg.sparse_static_scales and cfg.sparse_request_kv_scale
        assert main.direct_sparse_capacities == (compressed, 0)
        assert main.seq_len_k == max(256, ((compressed + 127) // 128) * 128)
        from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
            Fp4Fp8KvProducer,
        )

        assert type(main.kv_producer) is Fp4Fp8KvProducer
        assert main.kv_producer.native_fp4 == native_fp4
        assert (
            main.kv_producer.compile_signature()[0]
            == "dsv41_page64_fp4_group16_to_request_scaled_e4m3"
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


@pytest.mark.parametrize("native_fp4", (False, True))
def test_fp4_fp8_resource_reuses_exact_conversion_and_all_writer_publication(
    native, monkeypatch, native_fp4
):
    from flashinfer.experimental.deepseek_v41 import prims_ts, prims_ts_resources as fp4

    _, main = prims_ts._make_fp8_sources(192, 148, native_fp4)
    cfg = main._make_config()
    producer = main.kv_producer
    assert producer.compile_signature() == (
        "dsv41_page64_fp4_group16_to_request_scaled_e4m3",
        native_fp4,
    )
    assert (
        producer.compile_signature()
        != fp4.Fp4KvProducer(native_fp4).compile_signature()
    )
    primary, unused, scale, request = object(), object(), object(), object()
    runtime_args = (scale,)
    resource = producer.make_resource(
        (primary, unused),
        runtime_args=runtime_args,
        name="fp4_fp8",
        cfg=cfg,
        batch_idx=request,
    )
    assert type(resource) is fp4.Fp4Fp8SmemKvResource
    assert resource.fp4_cache is primary and resource.native_fp4 is native_fp4
    assert resource.main_scale is scale and resource.batch_idx is request
    assert resource.cfg is cfg and cfg.kv_smem_tile_bytes == 128 * 128
    assert cfg.kv_smem_tile_bytes * cfg.kv_stages == 128 * 1024
    for method in (
        "_store_fp4_chunk",
        "_load_fp4_stage",
        "_producer_load_kv_stage",
        "_stage_base",
        "_init_smem_state",
    ):
        assert inspect.getattr_static(fp4.Fp4Fp8SmemKvResource, method) is (
            inspect.getattr_static(fp4.Fp4SmemKvResource, method)
        )
    assert inspect.getattr_static(fp4.Fp4Fp8SmemKvResource, "_store_fragment") is not (
        inspect.getattr_static(fp4.Fp4SmemKvResource, "_store_fragment")
    )
    calls, pipeline = [], object()
    load_group, umma_group, layout = object(), object(), object()

    def create(**kwargs):
        calls.append(kwargs)
        return pipeline

    monkeypatch.setattr(
        fp4.PipelineConfig, "create_async_umma_pipeline_cfg", staticmethod(create)
    )
    assert producer.pipeline_config(cfg, load_group, umma_group, layout) is pipeline
    assert calls == [
        dict(
            num_stages=8,
            producer_group=load_group,
            consumer_group=umma_group,
            cta_layout_vmnk=layout,
            producer_signaling_threads=fp4.SignalingThreads.All,
            consumer_signaling_threads=fp4.SignalingThreads.CtaLeader,
        )
    ]


@pytest.mark.parametrize("runtime_args", (None, (), (None,), (object(), object())))
def test_fp4_fp8_producer_rejects_missing_or_ambiguous_request_scale(
    native, runtime_args
):
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import Fp4Fp8KvProducer

    with pytest.raises(ValueError, match="requires its request descale tensor"):
        Fp4Fp8KvProducer(native_fp4=True).make_resource(
            (object(),), runtime_args=runtime_args
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("kernel_variant", "swaps_mma_ab"),
        ("qkv_dtype", "bf16"),
        ("rope_dim", 64),
        ("head_dim_qk", 576),
        ("head_dim_per_stage_kv", 64),
        ("tile_size_q", 32),
        ("tile_size_kv", 64),
        ("kv_stages", 10),
        ("load_num_warps", 2),
        ("sparse_reuse_kv", False),
        ("sparse_direct", False),
        ("sparse_static_scales", False),
        ("sparse_request_kv_scale", False),
    ),
)
def test_fp4_fp8_producer_rejects_other_staging_topologies(native, field, value):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_sources

    _, main = _make_fp8_sources(64, 148, True)
    cfg = replace(main._make_config(), **{field: value})
    with pytest.raises(ValueError, match="FP4-to-E4M3 KV requires"):
        main.kv_producer.pipeline_config(cfg, None, None, None)


def test_fp4_fp8_stage_is_one_sw128_plane_with_eight_byte_stores(native):
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
        Fp4Fp8SmemKvResource,
        pack_float4_to_fp8_e4m3,
    )

    function = ast.parse(
        textwrap.dedent(inspect.getsource(Fp4Fp8SmemKvResource._store_fragment))
    ).body[0]
    assignments = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in function.body
        if isinstance(node, ast.Assign)
    }
    assert assignments["values"] == "fragment.load().to(cutlass.Float32)"
    assert assignments["descale"] == "cutlass.Float32(self.main_scale[self.batch_idx])"
    pack = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "pack_float4_to_fp8_e4m3"
    )
    assert [ast.unparse(arg) for arg in pack.args] == [
        "fdiv_rn(values[column], descale)",
        "fdiv_rn(values[column + 1], descale)",
        "fdiv_rn(values[column + 2], descale)",
        "fdiv_rn(values[column + 3], descale)",
    ]
    offset = next(
        node.value
        for node in function.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "offset"
    )
    assert ast.unparse(offset) == "row * 128 + dim ^ (row & 7) * 16"
    expression = compile(ast.Expression(offset), "<fp8-stage-offset>", "eval")
    offsets = [
        eval(expression, {"__builtins__": {}}, {"row": row, "dim": dim})
        for row in range(128)
        for dim in range(128)
    ]
    assert sorted(offsets) == list(range(128 * 128))
    for start in range(0, len(offsets), 8):
        assert offsets[start] % 8 == 0
        assert offsets[start : start + 8] == list(
            range(offsets[start], offsets[start] + 8)
        )
    pointer = next(
        node.value
        for node in function.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "dst"
    )
    # CuTe Pointer.store is scalar-only; use the DSL pointer's vector/alignment API.
    assert ast.unparse(pointer.func) == "cutlass.inttoptr"
    assert [(kw.arg, ast.unparse(kw.value)) for kw in pointer.keywords] == [
        ("mem_space", "3"),
        ("dtype", "cutlass.Int32"),
    ]
    store = function.body[-1].value
    assert ast.unparse(store.func) == "dst.store"
    assert ast.unparse(store.args[0]) == "packed.load(0, 2)"
    assert [(kw.arg, ast.literal_eval(kw.value)) for kw in store.keywords] == [
        ("alignment", 8)
    ]
    assert "cvt.rn.satfinite.e4m3x2.f32" in inspect.getsource(pack_float4_to_fp8_e4m3)

    from .test_deepseek_v41_decode import test_fp4_fp8_all_finite_codes

    probe = ast.parse(textwrap.dedent(inspect.getsource(test_fp4_fp8_all_finite_codes)))
    staging = next(
        node.value
        for node in ast.walk(probe)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "staging"
    )
    assert ast.unparse(staging) == "cutlass.make_array_view(smem)"
    smem = next(
        node.value
        for node in ast.walk(probe)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "smem"
    )
    # Array views require flat shape/stride tuples, including a rank-one layout.
    layout = smem.args[1]
    assert ast.unparse(layout.func) == "cute.make_layout"
    assert isinstance(layout.args[0], ast.Tuple) and len(layout.args[0].elts) == 1
    assert ast.unparse(layout.args[0].elts[0]) == "8 * stage_bytes + 256"
    stage_loads = [
        node
        for node in ast.walk(probe)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "resource._load_fp4_stage"
    ]
    assert len(stage_loads) == 1
    assert ast.unparse(stage_loads[0].args[1]) == (
        "staging.subview(128 + (page_slot * 4 + stage_idx) * stage_bytes)"
    )
    factory = next(
        node
        for node in ast.walk(probe)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "main.kv_producer.make_resource"
    )
    keywords = {kw.arg: ast.unparse(kw.value) for kw in factory.keywords}
    assert keywords["runtime_args"] == "(main_scale,)"
    assert keywords["batch_idx"] == "request"
    preparation = next(
        node
        for node in ast.walk(probe)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "prepare._prepare"
    )
    assert [ast.unparse(arg) for arg in preparation.args[-4:]] == [
        "pool",
        "main_scale",
        "cutlass.Int64(0)",
        "cutlass.Int64(cute.size(pool)) // 288",
    ]
    conversion = next(
        node
        for node in ast.walk(probe)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "convert"
    )
    assert preparation.lineno < conversion.lineno
    assert [ast.unparse(arg) for arg in conversion.args[:3]] == [
        "pool",
        "selected_global",
        "main_scale",
    ]


def test_selected_main_amax_decodes_all_group16_values_and_signed_scale_planes(native):
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import _fp4_group_amax

    function = _function_ast(_fp4_group_amax)
    assignments = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in function.body
        if isinstance(node, ast.Assign)
    }
    assert assignments["page_base"] == "cutlass.Int64(slot // 64) * (64 * 288)"
    assert assignments["token"] == "cutlass.Int64(slot & 63)"
    assert (
        assignments["payload_offset"]
        == "page_base + token * 256 + cutlass.Int64(group) * 8"
    )
    assert (
        assignments["scale_offset"]
        == "page_base + 64 * 256 + token * 32 + cutlass.Int64(group)"
    )
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    assert [(ast.unparse(node.target), ast.unparse(node.iter)) for node in loops] == [
        ("word", "cutlass.range_constexpr(2)"),
        ("element", "cutlass.range_constexpr(8)"),
    ]
    updates = [
        ast.unparse(node.value)
        for node in ast.walk(loops[0])
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "magnitude"
    ]
    assert updates == ["fmax_f32(magnitude, fabs_f32(values[element]))"]
    assert ast.unparse(function.body[-1].value) == "magnitude * fabs_f32(scale)"
    source = ast.unparse(function)
    assert "dtype=cutlass.Float4E2M1FN" in source
    assert "cutlass.Float8E4M3FN" in source
    assert "nvw.cvt_f4e2m1x8_to_f16x8(fp4)" in source
    assert "cute.TensorSSA(fp16, (8,), cutlass.Float16).to(cutlass.Float32)" in source


def test_selected_main_prepare_refreshes_scale_from_sanitized_live_rows_only(native):
    from flashinfer.experimental.deepseek_v41.prims_ts_fp8 import Fp8SwaDecodeKernel

    function = _function_ast(Fp8SwaDecodeKernel._prepare)
    branch = function.body[-1]
    assert isinstance(branch, ast.If)
    assert ast.unparse(branch.test) == "cutlass.const_expr(self.compressed_k > 0)"
    # Window-only compilation must have no main-cache/scale read or reduction.
    main_nodes = {id(node) for node in ast.walk(branch)}
    uses = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and node.id in ("global_cache", "main_scale", "_fp4_group_amax", "block_reduce")
    ]
    assert uses and all(id(node) in main_nodes for node in uses)
    loop = next(node for node in branch.body if isinstance(node, ast.For))
    assert ast.unparse(loop.iter) == "cutlass.range(lane // 32, self.compressed_k, 4)"
    assert (
        ast.unparse(loop.body[0].value)
        == "cutlass.Int32(selected_global[request, column])"
    )
    live = loop.body[1]
    assert ast.unparse(live.test) == "slot >= cutlass.Int32(0)"
    assert (
        ast.unparse(live.body[0].value)
        == "_fp4_group_amax(global_cache, slot, lane % 32)"
    )
    assert ast.unparse(live.body[1].value) == "fmax_f32(local_amax, group_amax)"
    assignments = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in branch.body
        if isinstance(node, ast.Assign)
    }
    assert assignments["warp_amax"] == "warp_reduce(local_amax, fmax_f32)"
    assert assignments["amax"] == (
        "block_reduce(warp_amax, fmax_f32, reduction_buffer, cutlass.Float32(0.0))"
    )
    publish = branch.body[-1]
    assert ast.unparse(publish.test) == "lane == 0"
    assert ast.unparse(publish.body[0].value) == "cutlass.Float32(1.0)"
    assert ast.unparse(publish.body[1].test) == "amax > cutlass.Float32(0.0)"
    assert ast.unparse(publish.body[1].body[0].value) == (
        "fdiv_rn(amax, cutlass.Float32(448.0))"
    )
    assert ast.unparse(publish.body[-1]) == "main_scale[request] = descale"
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    atomic_counts = [
        ast.unparse(node.args[0])
        for node in calls
        if ast.unparse(node.func) == "cute.arch.atomic_add"
    ]
    assert atomic_counts == [
        "swa_counts.iterator + request",
        "global_counts.iterator + request",
    ]


def test_fp8_native_compiled_abi_and_both_source_queries_are_fp8(native):
    from flashinfer.experimental.deepseek_v41 import prims_ts, prims_ts_fp8

    function = ast.parse(
        textwrap.dedent(inspect.getsource(prims_ts_fp8.Fp8SwaDecodeKernel.__call__))
    ).body[0]
    assert len(function.args.args[1:-1]) == 20
    assert [arg.arg for arg in function.args.args[11:15]] == [
        "swa_counts",
        "global_counts",
        "main_scale",
        "part_s",
    ]
    assert function.args.args[-1].arg == "stream"
    assert "expanded_q" not in prims_ts._Plan.__dataclass_fields__
    assert not hasattr(prims_ts_fp8.Fp8SwaDecodeKernel, "_expand_query")
    source_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in ("self.swa_attention", "self.global_attention")
    ]
    assert len(source_calls) == 2
    assert all(
        [ast.unparse(arg) for arg in node.args[:2]] == ["q", "q"]
        for node in source_calls
    )
    compiled = ast.parse(
        textwrap.dedent(inspect.getsource(inspect.unwrap(prims_ts._compile)))
    ).body[0]
    calls = [
        node
        for node in ast.walk(compiled)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.compile[cute.FrontendNext]"
    ]
    assert len(calls) == 1
    assert [ast.unparse(arg) for arg in calls[0].args] == [
        "decode_kernel",
        "q",
        "swa",
        "global_cache",
        "wi",
        "ci",
        "selected_swa",
        "selected_global",
        "swa_lengths",
        "global_lengths",
        "empty_lengths",
        "swa_counts",
        "global_counts",
        "main_scale",
        "part_s",
        "part_c",
        "lse_s",
        "lse_c",
        "sink",
        "out",
        "lse",
        "stream",
    ]
    main_scale = next(
        node.value
        for node in compiled.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "main_scale"
    )
    assert ast.unparse(main_scale) == "f(cutlass.Float32, (batch,), assumed_align=16)"
    source = inspect.getsource(inspect.unwrap(prims_ts._compile))
    assert "dsv41_request_scaled_e4m3_q_swa512_fp4_to_e4m3_fp32_sources_v3" in source
    assert "globalbk128" in source and "globalbk64" not in source
    assert "requestscale_fp32src" in source

    bindings = {
        ast.unparse(node.targets[0]): [ast.unparse(item) for item in node.value.elts]
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) in ("direct_swa", "direct_global")
    }
    assert bindings["direct_swa"] == [
        "selected_swa",
        "selected_global",
        "swa_lengths",
        "empty_lengths",
        "None",
        "None",
        "None",
        "None",
        "None",
    ]
    assert bindings["direct_global"] == [
        "selected_global",
        "selected_swa",
        "global_lengths",
        "empty_lengths",
        "None",
        "None",
        "main_scale",
        "None",
        "None",
    ]
    for call in source_calls:
        keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        if ast.unparse(call.func) == "self.global_attention":
            assert keywords == {
                "source_state_store_args": "(part_c, global_counts)",
                "kv_producer_runtime_args": "(main_scale,)",
            }
        else:
            assert keywords == {"source_state_store_args": "(part_s, swa_counts)"}


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
    assert call[0].shape == (64, 512, 1, 2) and call[0].dtype == args[0].dtype
    assert call[0].data_ptr() == args[0].data_ptr()
    assert call[1].shape == (1, 512, 128) and call[1].stride() == (512, 1, 512)
    assert call[9] is plan.empty_lengths
    assert call[10] is plan.swa_counts and call[11] is plan.global_counts
    assert call[13] is plan.part_s and call[15] is plan.lse_s
    assert (
        call[18].data_ptr() == out.data_ptr() and call[19].data_ptr() == lse.data_ptr()
    )
    assert call[17] is args[5]
    assert call[12].dtype == torch.float32 and call[12].shape == (2,)
    assert call[12].device == args[0].device and call[12].is_contiguous()
    assert not hasattr(plan, "expanded_q")
    assert [
        name
        for name, value in vars(plan).items()
        if getattr(value, "dtype", None) == torch.bfloat16
    ] == ["out"]
    assert plan.selected_global.shape == (2, max(64, compressed))
    if compressed:
        assert plan.part_c.dtype == torch.float32 and plan.lse_c.dtype == torch.float32
        assert plan.main_scale.dtype == torch.float32 and plan.main_scale.shape == (2,)
        assert call[12] is plan.main_scale
        assert call[14] is plan.part_c and call[16] is plan.lse_c
    else:
        assert plan.part_c is None and plan.lse_c is None and plan.main_scale is None
        assert call[14] is plan.part_s and call[16] is plan.lse_s
        assert call[12].data_ptr() == plan.lse_s.data_ptr()
        assert call[2].data_ptr() == plan.empty_cache.data_ptr()
        assert call[4].data_ptr() == plan.empty_indices.data_ptr()
    floating_owned = {
        name
        for name, value in vars(plan).items()
        if getattr(value, "dtype", None) == torch.float32
    }
    assert floating_owned == (
        {"lse", "main_scale", "part_s", "part_c", "lse_s", "lse_c"}
        if compressed
        else {"lse", "part_s", "lse_s"}
    )
    replacement = tuple(None if x is None else x.clone() for x in args)
    replay = adapter.run(replacement, plan)
    assert replay[0] is out and replay[1] is lse and replay[2] is plan
    assert len(adapter.compilations) == 1 and len(adapter.launches) == 2
    assert adapter.launches[-1][0].data_ptr() == replacement[0].data_ptr()
    if compressed:
        assert adapter.launches[-1][12] is plan.main_scale
    else:
        assert adapter.launches[-1][12].data_ptr() == plan.lse_s.data_ptr()


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
        "main_scale",
        "part_s",
        "part_c",
        "lse_s",
        "lse_c",
        "empty_cache",
        "empty_indices",
    ),
)
def test_fp8_adapter_checks_every_owned_buffer_for_input_overlap(adapter, field):
    args = _host_case()
    _, _, plan = adapter.run(args)
    # Preserve FP32[B] metadata so the scale case reaches the overlap guard,
    # rather than passing merely because the earlier dtype check rejected Q.
    overlap = args[5][: args[0].shape[0]] if field == "main_scale" else args[0]
    with pytest.raises(ValueError, match="overlap"):
        adapter.run(args, replace(plan, **{field: overlap}))
    assert len(adapter.launches) == 1


@pytest.mark.parametrize(
    "field", ("out", "lse", "main_scale", "part_s", "part_c", "lse_s", "lse_c")
)
def test_fp8_adapter_checks_floating_scratch_for_autograd(adapter, field):
    args = _host_case()
    _, _, plan = adapter.run(args)
    getattr(plan, field).requires_grad_(True)
    with pytest.raises(ValueError, match="inference-only"):
        adapter.run(args, plan)
    assert len(adapter.launches) == 1


@pytest.mark.parametrize("compressed", (0, 64, 192, 512))
def test_fp8_adapter_rejects_unprepared_capture_and_wrong_plans(
    adapter, monkeypatch, compressed
):
    args = _host_case(compressed=compressed)
    _, _, plan = adapter.run(args)
    for bad in (object(), replace(plan, signature=("different_mode",))):
        with pytest.raises(ValueError, match="declaration mismatch"):
            adapter.run(args, bad)
    monkeypatch.setattr(adapter.torch.cuda, "is_current_stream_capturing", lambda: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared capture must not allocate or compile")

    monkeypatch.setattr(adapter.module, "_compile", forbidden)
    for name in ("empty", "empty_like", "full"):
        monkeypatch.setattr(adapter.torch, name, forbidden)
    with pytest.raises(ValueError, match="before CUDA Graph capture"):
        adapter.run(args)
    assert len(adapter.compilations) == 1 and len(adapter.launches) == 1
    adapter.run(args, plan)
    assert len(adapter.compilations) == 1 and len(adapter.launches) == 2


@pytest.mark.parametrize(
    "bad", ("missing", "object", "dtype", "rank", "batch", "strided", "device")
)
def test_fp8_adapter_rejects_invalid_owned_request_scale_metadata(adapter, bad):
    torch = adapter.torch
    args = _host_case()
    _, _, plan = adapter.run(args)
    if bad == "missing":
        scale = None
    elif bad == "object":
        scale = object()
    elif bad == "dtype":
        scale = torch.empty(2, dtype=torch.bfloat16)
    elif bad == "rank":
        scale = torch.empty(2, 1, dtype=torch.float32)
    elif bad == "batch":
        scale = torch.empty(3, dtype=torch.float32)
    elif bad == "strided":
        scale = torch.empty(4, dtype=torch.float32)[::2]
        assert scale.shape == (2,) and not scale.is_contiguous()
    else:
        # A meta tensor supplies a genuinely different device without touching
        # CUDA; metadata must be rejected before overlap/data-pointer checks.
        scale = torch.empty(2, device="meta", dtype=torch.float32)
    with pytest.raises(ValueError, match="contiguous same-device FP32"):
        adapter.run(args, replace(plan, main_scale=scale))
    assert len(adapter.compilations) == len(adapter.launches) == 1


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


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _e4m3_value(byte):
    magnitude = byte & 0x7F
    if magnitude == 0x7F:
        return math.nan
    exponent, mantissa = magnitude >> 3, magnitude & 7
    value = (
        math.ldexp(float(mantissa), -9)
        if exponent == 0
        else math.ldexp(1.0 + mantissa / 8, exponent - 7)
    )
    return -value if byte & 0x80 else value


def _e4m3_byte(value):
    """Independent finite E4M3 RTNE/satfinite, including signed zero."""
    assert not math.isnan(value)
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    magnitude = min(abs(value), 448.0)
    if magnitude < 2**-6:
        code = round(math.ldexp(magnitude, 9))
    else:
        exponent = math.frexp(magnitude)[1] - 1
        code = round(math.ldexp(magnitude, 3 - exponent)) + (exponent + 6) * 8
    return sign | code


def _fp4_pages(rows, page_count=2):
    """Build page planes independently; untouched rows deliberately contain NaNs."""
    pages = [bytearray([0xFF]) * (64 * 288) for _ in range(page_count)]
    for slot, codes, scales in rows:
        assert len(codes) == 512 and len(scales) == 32
        page, token = pages[slot // 64], slot % 64
        page[token * 256 : (token + 1) * 256] = bytes(
            codes[dim] | (codes[dim + 1] << 4) for dim in range(0, 512, 2)
        )
        start = 64 * 256 + token * 32
        page[start : start + 32] = bytes(scales)
    return tuple(bytes(page) for page in pages)


def _selected_main_reference(pages, selections):
    """Stdlib-only selected-value reference, with explicit binary32 division.

    Decode only live selected rows, all 512 components and original signed
    group16 E4M3 scales. Invalid addresses never inspect poisoned page bytes.
    No production conversion, Torch cast, reciprocal or BF16 scratch is used.
    """
    fp4 = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    )
    result = []
    for indices in selections:
        valid = tuple(
            0 <= slot < len(pages) * 64 and slot != 0x7FFFFFFF for slot in indices
        )
        rows = []
        for slot, keep in zip(indices, valid, strict=True):
            if not keep:
                rows.append((0.0,) * 512)
                continue
            page, token = pages[slot // 64], slot % 64
            row = []
            for dim in range(512):
                byte = page[token * 256 + dim // 2]
                code = (byte >> (4 * (dim % 2))) & 15
                scale = _e4m3_value(page[64 * 256 + token * 32 + dim // 16])
                row.append(fp4[code] * scale)
            rows.append(tuple(row))
        amax = max((abs(value) for row in rows for value in row), default=0.0)
        scale = _f32(amax / 448.0) if amax > 0 else 1.0
        staged = tuple(
            bytes(_e4m3_byte(_f32(value / scale)) for value in row) for row in rows
        )
        represented = tuple(
            tuple(_e4m3_value(byte) * scale for byte in row) for row in staged
        )
        result.append(
            SimpleNamespace(
                amax=amax,
                scale=scale,
                valid=valid,
                staged=staged,
                represented=represented,
                selected=tuple(
                    slot if keep else -1
                    for slot, keep in zip(indices, valid, strict=True)
                ),
            )
        )
    return result


def _check_selected_main_reference(native, pages, selections):
    import torch
    from .test_deepseek_v41_decode import fp8_main_reference

    expected = _selected_main_reference(pages, selections)
    pool = torch.tensor([list(page) for page in pages], dtype=torch.uint8).view(
        len(pages), 64, 1, 288
    )
    ids = torch.tensor(selections, dtype=torch.int32)[:, None]
    staged, scale, valid = fp8_main_reference(pool, ids)
    assert staged.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32 and scale.shape == (len(selections),)
    assert scale.tolist() == [request.scale for request in expected]
    assert staged.view(torch.uint8).tolist() == [
        [list(row) for row in request.staged] for request in expected
    ]
    assert valid.tolist() == [list(request.valid) for request in expected]
    assert valid.sum(-1).tolist() == [sum(request.valid) for request in expected]
    represented = staged.double() * scale.double()[:, None, None]
    assert represented.tolist() == [
        [list(row) for row in request.represented] for request in expected
    ]
    return expected


@pytest.mark.parametrize("sign", (0, 0x80))
def test_stdlib_e4m3_reference_roundtrips_all_finite_codes_and_signed_zero(sign):
    for magnitude in range(0x7F):
        code = sign | magnitude
        assert _e4m3_byte(_e4m3_value(code)) == code
    assert math.copysign(1.0, _e4m3_value(sign)) == (-1.0 if sign else 1.0)
    assert _e4m3_byte(-math.inf if sign else math.inf) == sign | 0x7E


def test_fp8_main_reference_selected_amax_fp32_division_signed_scales_and_replay(
    native,
):
    codes = [0] * 512
    codes[0], codes[511] = 5, 5
    # Large-scale all-zero groups must contribute zero, not 6 * 448. Only the
    # final component sets amax=9; the first component is the division witness.
    scales = [0x7E] * 32
    scales[0], scales[-1] = _e4m3_byte(1.125), _e4m3_byte(3.0)
    pages = _fp4_pages(
        (
            (0, codes, scales),
            (1, [7] * 512, [0x7E] * 32),  # finite but unselected amax=2688
            (63, [2] * 512, [_e4m3_byte(-2.0)] * 32),
            (64, [0, 8] * 256, [0xFE] * 32),
        )
    )
    indices = [
        [0, 63, 63, -1, 128, 0x7FFFFFFF, -17],
        [64, 64, -1, 128, 0x7FFFFFFF, -17, -1],
        [-1, 128, 0x7FFFFFFF, -17, -1, -1, -1],
    ]
    first, zero, empty = _check_selected_main_reference(native, pages, indices)
    assert first.amax == 9.0 and first.scale == _f32(9.0 / 448.0)
    assert first.valid == (True, True, True, False, False, False, False)
    assert first.selected == (0, 63, 63, -1, -1, -1, -1)
    assert first.staged[1] == first.staged[2]
    assert first.staged[0][511] == 0x7E
    assert _f32(3.375 / first.scale) == 168.0
    assert first.staged[0][0] == 0x72  # RTNE(168) = 160, not 176
    assert _e4m3_byte(3.375 / first.scale) == 0x73  # wrong FP64 normalization
    assert first.represented[0][0] == 160.0 * first.scale
    assert zero.amax == empty.amax == 0.0 and zero.scale == empty.scale == 1.0
    assert sum(zero.valid) == 2 and sum(empty.valid) == 0
    assert zero.staged[0][:2] == bytes((0x80, 0x00))
    assert math.copysign(1.0, zero.represented[0][0]) == -1.0
    assert all(row == bytes(512) for row in empty.staged)

    changed = [list(row) for row in indices]
    changed[0][0] = 1
    replay = _check_selected_main_reference(native, pages, changed)
    assert replay[0].amax == 2688.0 and replay[0].scale == 6.0
    assert replay[0].valid == first.valid
    assert replay[1].scale == replay[2].scale == 1.0


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
    # Selected main amax is 4. The normalized 1.5 is exactly the RTNE tie
    # 168 -> 160 after FP32 division; +/-4 stage at +/-448. SWA stays unit FP8.
    descale = _f32(4.0 / 448.0)
    small, large = 160.0 * descale, 448.0 * descale
    answer = torch.empty(1, 64, 512, dtype=torch.float64)
    answer[..., ::2] = (2 * 0.3125 + small - large - large) / 5
    answer[..., 1::2] = (2 * 0.3125 + large) / 5
    sensitivity = torch.empty_like(answer)
    sensitivity[..., ::2] = (2 * 0.3125 + small + large + large) / 5
    sensitivity[..., 1::2] = (2 * 0.3125 + large) / 5
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


@pytest.mark.parametrize(
    "code,scale,anchor_code,anchor_scale,staged_code,anchor_staged",
    (
        (3, 1.125, 2, 1.0, 0x7E, 0x78),
        (11, 1.125, 2, 1.0, 0xFE, 0x78),
        (7, 448.0, 2, 1.0, 0x7E, 0x23),
        (15, 448.0, 2, 1.0, 0xFE, 0x23),
        (1, 2**-9, 7, 448.0, 0x00, 0x7E),
        (9, 2**-9, 7, 448.0, 0x80, 0x7E),
        (1, 3 * 2**-7, 7, 448.0, 0x01, 0x7E),
        (9, 3 * 2**-7, 7, 448.0, 0x81, 0x7E),
        (1, 3 * 2**-9, 2, 1.0, 0x3A, 0x7E),
        (0, -448.0, 7, 448.0, 0x80, 0x7E),
        (8, -448.0, 7, 448.0, 0x00, 0x7E),
        (1, 2**-9, 1, 2**-9, 0x7E, 0x7E),
    ),
)
def test_fp8_oracle_converts_main_before_both_logits_and_values(
    native, code, scale, anchor_code, anchor_scale, staged_code, anchor_staged
):
    import torch
    from .test_deepseek_v41_decode import fp8_oracle

    q, swa, main, wi, ci, sink = _host_case(batch=1)
    q.view(torch.uint8)[..., 0] = torch.tensor(0.5).to(q.dtype).view(torch.uint8)
    pages = main.view(2, -1)
    pages.fill_(255)
    pages[0, :256].fill_(code | (code << 4))
    pages[0, 64 * 256 : 64 * 256 + 32].fill_(_e4m3_byte(scale))
    pages[1, :256].fill_(anchor_code | (anchor_code << 4))
    pages[1, 64 * 256 : 64 * 256 + 32].fill_(_e4m3_byte(anchor_scale))
    ci[..., :2] = torch.tensor([0, 64], dtype=torch.int32)
    sink.fill_(0.125)
    reference = _check_selected_main_reference(
        native, tuple(bytes(row) for row in pages.tolist()), [[0, 64]]
    )[0]
    assert reference.staged[0] == bytes([staged_code]) * 512
    assert reference.staged[1] == bytes([anchor_staged]) * 512
    represented = _e4m3_value(staged_code) * reference.scale
    anchor = _e4m3_value(anchor_staged) * reference.scale
    if abs(code) in (7, 15) and scale == 448.0:
        assert reference.scale == 6.0 and abs(represented) == 2688.0
        assert anchor == 1.03125  # s=6, raw 1 -> FP8(1/6) * 6
    expected, lse, bound = fp8_oracle(q, swa, main, wi, ci, sink)
    logits = torch.tensor(
        [0.5 * represented / math.sqrt(512), 0.5 * anchor / math.sqrt(512), 0.125],
        dtype=torch.float64,
    )
    weights = logits.softmax(0)[:2]
    answer = weights[0] * represented + weights[1] * anchor
    sensitivity = weights[0] * abs(represented) + weights[1] * abs(anchor)
    underflow = torch.exp(logits[:2].max() - logits.logsumexp(0)) * (
        abs(represented) + abs(anchor)
    )
    analytic_bound = (
        (2**-4 + 3 * 2**-8 + 1e-5) * sensitivity + (2**-10 / 448) * underflow + 1e-6
    )
    torch.testing.assert_close(
        expected, torch.full_like(expected, answer.item()), rtol=0, atol=1e-12
    )
    torch.testing.assert_close(
        lse, torch.full_like(lse, logits[:2].logsumexp(0).item()), rtol=0, atol=1e-14
    )
    torch.testing.assert_close(
        bound, torch.full_like(bound, analytic_bound.item()), rtol=0, atol=1e-12
    )


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
