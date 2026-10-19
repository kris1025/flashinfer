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

"""Host-only joint-decode, cache-unit staging and generic source-state contracts.

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


def _make_request_scaled_kernel(native, compressed):
    """Independent generic host-only producer, never the V4.1 finalizer.

    Its resource carries the generic runtime scale to exercise task plumbing;
    host contracts never execute its inherited loader or a DSL entry point.
    """
    from cutlass.experimental.task_scheduling.enums import SignalingThreads
    from cutlass.experimental.task_scheduling.resources import PipelineConfig
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.resources.smem_resources import (
        SmemKvResource,
    )

    @dataclass(kw_only=True)
    class RequestResource(SmemKvResource):
        fp4_cache: object = None
        main_scale: object = None

    @dataclass(frozen=True)
    class RequestProducer:
        def compile_signature(self):
            return ("generic_request_scale_host_contract",)

        def pipeline_config(self, cfg, load_group, umma_group, cta_layout_vmnk):
            return PipelineConfig.create_async_umma_pipeline_cfg(
                num_stages=cfg.kv_stages,
                producer_group=load_group,
                consumer_group=umma_group,
                cta_layout_vmnk=cta_layout_vmnk,
                producer_signaling_threads=SignalingThreads.All,
                consumer_signaling_threads=SignalingThreads.CtaLeader,
            )

        def make_resource(self, pools, *, runtime_args=None, **kwargs):
            assert runtime_args is not None and len(runtime_args) == 1
            return RequestResource(
                fp4_cache=pools[0], main_scale=runtime_args[0], **kwargs
            )

    kernel = _make_kernel(
        native,
        seq_len_k=max(256, ((compressed + 127) // 128) * 128),
        source_state_store=_SourceStateStore(),
        kv_producer=RequestProducer(),
        sparse_profile=native.profile(
            family="keep",
            tile_size_q=64,
            split_kv=1,
            direct_inputs=True,
            reuse_kv=True,
            reuse_kv_stages=8,
            kv_tile_size=128,
            page_pipeline_stages=1,
        ),
    )
    kernel.direct_static_scales = True
    kernel.direct_request_kv_scale = True
    kernel.direct_sparse_capacities = (compressed, 0)
    return kernel


def test_request_kv_scale_is_part_of_the_batch_independent_compile_identity(native):
    main = _make_request_scaled_kernel(native, 192)
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
    main = _make_request_scaled_kernel(native, 64)
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
    main = _make_request_scaled_kernel(native, 64)
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
    main = _make_request_scaled_kernel(native, 64)
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
    main = _make_request_scaled_kernel(native, 64)
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
    main = _make_request_scaled_kernel(native, 64)
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


@pytest.mark.parametrize(
    "mode", ("request", "static", "custom", "dynamic", "device", "host")
)
def test_request_scale_restores_both_factors_without_changing_old_scale_paths(
    native, mode
):
    main = _make_request_scaled_kernel(native, 64)
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
    pools = object()
    factors = _host_function(
        function,
        dict(
            cutlass=SimpleNamespace(
                const_expr=lambda value: value, Float32=float, Int64=int
            ),
            cute=SimpleNamespace(arch=SimpleNamespace(block_idx=lambda: (0, 0, 1))),
            bind_sparse_views=bind,
            kv_producer_args=pools,
            LOG2_E=log2_e,
        ),
    )
    sinks = object()
    si, ci, sl, cl = object(), object(), object(), object()
    if mode == "request":
        params = (si, ci, sl, cl, None, None, Scale("ss", (3.0, 6.0)), None, sinks)
        expected, expected_reads = (1.5, 15.0, sinks), [("ss", 1)]
    elif mode in ("static", "custom"):
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
        SimpleNamespace(
            device_scales=mode == "device",
            kv_producer=SimpleNamespace(bind_sparse_views=bind)
            if mode == "custom"
            else None,
        ),
        cfg,
        params,
        0.25,
        2.5,
    )
    assert actual == expected and reads == expected_reads
    if cfg.sparse_direct:
        args = (si, sl, cl, pools) if mode == "custom" else (si, ci, sl, cl)
        assert bindings == [
            ((*args, cfg.total_kv_tiles * cfg.tile_size_kv), {"request": 1})
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
    main = _make_request_scaled_kernel(native, 192)
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


@pytest.mark.parametrize("has_sink", (False, True))
def test_joint_adapter_forwards_caller_storage_and_runtime_scales(
    native, monkeypatch, has_sink
):
    from contextlib import nullcontext
    import builtins
    import torch
    from flashinfer.experimental.deepseek_v41 import prims_ts

    calls, imports, binds = [], [], []
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if "prims_ts_mxfp8" in name or any(
            "prims_ts_mxfp8" in item for item in fromlist or ()
        ):
            imports.append(name)
            raise AssertionError("joint native FP8 must not import MXFP8")
        return original(name, globals, locals, fromlist, level)

    q_view, swa_view, main_view, out_view, lse_view = (object() for _ in range(5))
    query = SimpleNamespace(
        dtype=torch.float8_e4m3fn,
        device=SimpleNamespace(index=0),
        permute=lambda *args: q_view,
    )
    out = SimpleNamespace(permute=lambda *args: out_view)
    # No prepared metadata/sink buffers exist on the native path.
    views = SimpleNamespace(lse=SimpleNamespace(view=lambda *args: lse_view))
    bound = SimpleNamespace(
        swa_cache=SimpleNamespace(
            view=lambda *args: SimpleNamespace(permute=lambda *args: swa_view)
        ),
        main_cache=SimpleNamespace(view=lambda *args: main_view),
        indices=object(),
        topk_lens=object(),
        seq_lens=object(),
        out=out,
        bmm1_scale=0.125,
        bmm2_scale=-2.0,
        sinks=object() if has_sink else None,
        views=views,
        batch=2,
        compressed_k=192,
    )

    def bind(*args, **kwargs):
        binds.append((args, kwargs))
        return bound

    def compile_stub(*args, **kwargs):
        assert args == (0, 192) and kwargs == {"has_sink": has_sink}
        return lambda *packet: calls.append(packet)

    monkeypatch.setattr(prims_ts, "bind_inputs", bind)
    monkeypatch.setattr(prims_ts, "_compile", compile_stub)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert (
        prims_ts.decode_mixed(
            query, None, object(), out=out, bmm1_scale=0.125, bmm2_scale=-2.0
        )
        is out
    )
    assert len(binds) == len(calls) == 1 and not imports
    assert binds[0][1] == {"backend": "prims-ts"}
    assert calls[0] == (
        q_view,
        swa_view,
        main_view,
        bound.indices,
        bound.topk_lens,
        bound.seq_lens,
        bound.sinks,
        out_view,
        lse_view,
        0.125,
        -2.0,
    )


@pytest.mark.parametrize("window_only", (False, True))
@pytest.mark.parametrize("supplied_out", (False, True))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16", "float32", "float8_e5m2"))
def test_prims_ts_rejects_unsupported_q_before_validation_or_execution(
    native, monkeypatch, window_only, supplied_out, dtype
):
    import torch
    from flashinfer.deepseek_v41 import deepseek_v41_decode, deepseek_v41_window_decode
    from flashinfer.experimental.deepseek_v41 import prims_ts

    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unsupported Q must fail before validation or execution")

    for name in ("bind_inputs", "_compile"):
        monkeypatch.setattr(prims_ts, name, forbidden)
    for name in ("device", "get_device_capability", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name in ("empty", "empty_like", "full"):
        monkeypatch.setattr(torch, name, forbidden)
    api = deepseek_v41_window_decode if window_only else deepseek_v41_decode
    with pytest.raises(ValueError, match="Prims-TS decode requires E4M3 Q"):
        api(
            SimpleNamespace(dtype=getattr(torch, dtype)),
            None,
            None,
            out=object() if supplied_out else None,
            backend="prims-ts",
        )
    assert not calls


def test_fp8_entry_validates_declaration_before_execution(native, monkeypatch):
    import torch
    from flashinfer.experimental.deepseek_v41 import prims_ts

    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid Q declaration reached execution")

    monkeypatch.setattr(prims_ts, "_compile", forbidden)
    with pytest.raises(ValueError, match="Q must"):
        prims_ts.decode_mixed(
            torch.zeros(2, 1, 32, 512).to(torch.float8_e4m3fn), None, None
        )
    assert not calls


def test_native_fp8_jit_namespace_and_complete_source_keys(native):
    from flashinfer.experimental.deepseek_v41 import prims_ts

    function = _function_ast(inspect.unwrap(prims_ts._compile))
    builds = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "build_and_load_cute_dsl_kernel"
    ]
    assert (
        len(builds) == 1
        and ast.literal_eval(builds[0].args[0]) == "deepseek_v41_prims_ts_fp8"
    )
    files = next(kw.value for kw in builds[0].keywords if kw.arg == "extra_key_files")
    assert {ast.unparse(n) for n in files.elts} == {
        "__file__",
        "_workspace.__file__",
        "_sparse_views.__file__",
        "prims_ts_resources.__file__",
        "hca_v41_primitives.__file__",
        "prims_ts_fp8.__file__",
        "fp4_common.__file__",
        "*sorted(native_root.rglob('*.py'))",
    }
    source = ast.unparse(function)
    assert "dsv41_shared_pertensor_joint_e4m3_swa512_fp4_inline_metadata_v2" in source
    assert "runtime_bmm1_bmm2_f32" in source and "mxfp8" not in source
    assert not hasattr(prims_ts, "_Plan") and not hasattr(prims_ts, "_make_fp8_sources")


@pytest.mark.parametrize("compressed", list(range(0, 513, 64)))
@pytest.mark.parametrize("native_fp4", (False, True))
def test_fp8_adapter_builds_one_finalizing_joint_attention(
    native, compressed, native_fp4
):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_attention
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
        MixedFp8KvProducer,
    )

    attention = _make_fp8_attention(compressed, 148, native_fp4)
    cfg = attention._make_config()
    assert cfg.qkv_dtype == "e4m3" and cfg.is_fp8_qkv()
    assert (
        cfg.tile_size_q,
        cfg.tile_size_kv,
        cfg.head_dim_per_stage_kv,
        cfg.kv_stages,
    ) == (64, 128, 128, 8)
    assert cfg.sparse_direct and cfg.sparse_reuse_kv and cfg.single_kv_pipe
    assert cfg.num_insts_kv == cfg.num_ctas_per_head_dim == cfg.num_ctas_per_seq_kv == 1
    assert cfg.load_num_warps == 4 and cfg.fuse_sparse_epilogue
    assert (
        not cfg.use_persistent_scheduler
        and not cfg.use_clc_dynamic_persistent_scheduler
    )
    assert (
        attention.finalize_output
        and attention.source_state_store is None
        and attention.output_quant is None
    )
    assert (
        attention.direct_sparse
        and attention.direct_static_scales
        and not attention.direct_request_kv_scale
    )
    assert cfg.sparse_static_scales and not cfg.sparse_request_kv_scale
    assert (
        attention.direct_split_softmax_scale is cfg.sparse_split_softmax_scale is True
    )
    native.config.validate_split_softmax_scale_config(cfg)
    assert attention.direct_sparse_capacities == (128, compressed)
    assert attention.seq_len_k == max(256, 128 + ((compressed + 127) // 128) * 128)
    assert type(attention.kv_producer) is MixedFp8KvProducer
    assert attention.kv_producer.compile_signature() == (
        "dsv41_shared_pertensor_joint_swa_fp4_e4m3_inline_metadata_v2",
        native_fp4,
    )
    signature = attention.compile_signature()
    attention.batch_size = 3
    assert attention.compile_signature() == signature
    assert len(signature) == 9 and "source_state_store" not in repr(signature)
    attention.direct_split_softmax_scale = False
    assert attention.compile_signature() != signature


@pytest.mark.parametrize("native_fp4", (False, True))
def test_fp4_fp8_resource_reuses_exact_conversion_and_all_writer_publication(
    native, monkeypatch, native_fp4
):
    from flashinfer.experimental.deepseek_v41 import prims_ts, prims_ts_resources as fp4

    attention = prims_ts._make_fp8_attention(192, 148, native_fp4)
    cfg = attention._make_config()
    producer = attention.kv_producer
    main, swa, request = object(), object(), object()
    resource = producer.make_resource(
        (main, swa), name="mixed_probe", cfg=cfg, batch_idx=request
    )
    assert type(resource) is fp4.MixedFp8SmemKvResource
    assert resource.fp4_cache is main and resource.swa_cache is swa
    assert (
        resource.native_fp4 is native_fp4
        and resource.batch_idx is request
        and resource.cfg is cfg
    )
    assert not hasattr(resource, "main_scale")
    assert (
        cfg.kv_smem_tile_bytes == 128 * 128
        and cfg.kv_stages * cfg.kv_smem_tile_bytes == 128 * 1024
    )
    for method in (
        "_store_fp4_chunk",
        "_load_fp4_stage",
        "_stage_base",
        "_init_smem_state",
    ):
        assert inspect.getattr_static(
            fp4.MixedFp8SmemKvResource, method
        ) is inspect.getattr_static(fp4.Fp4SmemKvResource, method)
    assert inspect.getattr_static(
        fp4.MixedFp8SmemKvResource, "_store_fragment"
    ) is inspect.getattr_static(fp4.Fp4Fp8SmemKvResource, "_store_fragment")
    calls, pipeline = [], object()

    def create(**kwargs):
        calls.append(kwargs)
        return pipeline

    monkeypatch.setattr(
        fp4.PipelineConfig, "create_async_umma_pipeline_cfg", staticmethod(create)
    )
    load, umma, layout = object(), object(), object()
    assert producer.pipeline_config(cfg, load, umma, layout) is pipeline
    assert calls == [
        dict(
            num_stages=8,
            producer_group=load,
            consumer_group=umma,
            cta_layout_vmnk=layout,
            producer_signaling_threads=fp4.SignalingThreads.All,
            consumer_signaling_threads=fp4.SignalingThreads.CtaLeader,
        )
    ]


def _fp4_base_config(native):
    attention = _make_kernel(
        native,
        qkv_dtype="bf16",
        sparse_profile=native.profile(
            family="keep",
            tile_size_q=64,
            split_kv=1,
            direct_inputs=True,
            reuse_kv=True,
            reuse_kv_stages=8,
            kv_tile_size=64,
        ),
        finalize_output=True,
    )
    attention.direct_sparse = True
    attention.direct_static_scales = True
    attention.direct_sparse_capacities = (128, 64)
    return attention._make_config()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("native_fp4", (False, True))
@pytest.mark.parametrize("explicit_none", (False, True))
def test_fp4_base_factory_omits_absent_runtime_args(
    native, monkeypatch, native_fp4, explicit_none
):
    from flashinfer.experimental.deepseek_v41 import prims_ts_resources as fp4

    cfg = _fp4_base_config(native)
    factory = fp4.Fp4KvProducer.make_resource
    calls = []

    def make_resource(self, pools, **kwargs):
        # Legacy factories forward keywords to a resource with no runtime_args.
        assert "runtime_args" not in kwargs
        resource = factory(self, pools, **kwargs)
        calls.append((pools, resource))
        return resource

    monkeypatch.setattr(fp4.Fp4KvProducer, "make_resource", make_resource)
    pools = (object(), object())
    kwargs = {"kv_producer_runtime_args": None} if explicit_none else {}
    manager, _ = native.kernel.build_throughput_latency_mla_task_manager(
        cfg,
        total_kv_tiles=cfg.total_kv_tiles,
        use_page_offsets=True,
        kv_producer=fp4.Fp4KvProducer(native_fp4=native_fp4),
        kv_producer_args=pools,
        **kwargs,
    )
    assert manager.tasks and len(calls) == 1
    actual_pools, resource = calls[0]
    assert actual_pools is pools and resource.fp4_cache is pools[0]
    assert type(resource) is fp4.Fp4SmemKvResource
    assert resource.native_fp4 is native_fp4 and resource.cfg is cfg
    assert not hasattr(resource, "main_scale") and not hasattr(resource, "swa_cache")


@pytest.mark.parametrize("native_fp4", (False, True))
def test_fp4_base_pipeline_signals_all_writers(native, monkeypatch, native_fp4):
    from flashinfer.experimental.deepseek_v41 import prims_ts_resources as fp4

    calls, pipeline = [], object()
    load_group, umma_group, layout = object(), object(), object()

    def create(**kwargs):
        calls.append(kwargs)
        return pipeline

    monkeypatch.setattr(
        fp4.PipelineConfig, "create_async_umma_pipeline_cfg", staticmethod(create)
    )
    producer = fp4.Fp4KvProducer(native_fp4=native_fp4)
    assert (
        producer.pipeline_config(
            _fp4_base_config(native), load_group, umma_group, layout
        )
        is pipeline
    )
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


def test_fp4_base_stage_fences_every_writer(native):
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
        Fp4SmemKvResource,
    )

    function = _function_ast(Fp4SmemKvResource._producer_load_kv_stage)
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    fences = [
        node
        for node in calls
        if ast.unparse(node.func) == "cute.arch.fence_view_async_shared"
    ]
    assert len(fences) == 1 and function.body[-1].value is fences[0]
    assert not any(isinstance(node, ast.Return) for node in ast.walk(function))
    assert not any(ast.unparse(node.func).endswith("elect_sync") for node in calls)


@pytest.mark.parametrize(
    "runtime_args", ((), (None,), (object(),), (object(), object()))
)
@pytest.mark.parametrize("producer_name", ("Fp4Fp8KvProducer", "MixedFp8KvProducer"))
def test_common_unit_producer_rejects_any_request_scale(
    native, runtime_args, producer_name
):
    from flashinfer.experimental.deepseek_v41 import prims_ts_resources as fp4

    with pytest.raises(ValueError, match="request scales"):
        getattr(fp4, producer_name)(native_fp4=True).make_resource(
            (object(), object()), runtime_args=runtime_args
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
        ("sparse_request_kv_scale", True),
        ("num_insts_kv", 2),
        ("num_ctas_per_seq_kv", 2),
        ("num_ctas_per_head_dim", 2),
        ("num_tokens_per_page", 64),
        ("use_persistent_scheduler", 1),
        ("use_clc_dynamic_persistent_scheduler", 1),
        ("fuse_sparse_epilogue", False),
        ("use_bf16_output", False),
        ("logical_num_heads_q", 32),
        ("logical_seq_len_q", 2),
        ("sparse_direct_capacities", (64, 64)),
        ("sparse_direct_capacities", (128, 63)),
        ("sparse_split_softmax_scale", False),
        ("o_dtype", "fp32"),
        ("use_fp8_output", True),
        ("head_dim_v", 256),
        ("head_dim_per_stage_v", 64),
        ("head_dim_per_cta_v", 256),
        ("num_heads_q", 32),
        ("seq_len_q", 2),
        ("num_ctas_per_seq_q", 2),
        ("num_ctas_for_all_heads", 2),
        ("use_multi_ctas_kv", 1),
        ("use_cluster_reduction", 1),
        ("mask_type", "causal"),
        ("cache_uniform_sparse_quads", True),
        ("q_stages", 2),
        ("page_offsets_stages", 2),
        ("defer_sparse_max_update", True),
        ("paired_sparse_correction", True),
        ("fuse_sparse_cluster_epilogue", True),
        ("fuse_sparse_reduction", True),
        ("sparse_direct_capacities", ()),
        ("sparse_direct_capacities", (128,)),
        ("sparse_direct_capacities", (128, 64, 64)),
        ("sparse_direct_capacities", (128, -64)),
        ("sparse_direct_capacities", (128, 576)),
        ("seq_len_kv", 128),
        ("seq_len_kv", 384),
    ),
)
def test_mixed_producer_rejects_other_staging_topologies(native, field, value):
    from flashinfer.experimental.deepseek_v41.prims_ts import _make_fp8_attention

    attention = _make_fp8_attention(64, 148, True)
    cfg = replace(attention._make_config(), **{field: value})
    if field == "sparse_split_softmax_scale":
        native.config.validate_split_softmax_scale_config(cfg)
        match = "mixed V4.1 KV requires"
    else:
        match = "split softmax scales require"
        with pytest.raises(ValueError, match=match):
            native.config.validate_split_softmax_scale_config(cfg)
    with pytest.raises(ValueError, match=match):
        attention.kv_producer.pipeline_config(cfg, None, None, None)


def test_fp4_fp8_stage_is_one_sw128_plane_with_eight_byte_stores(native):
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
        Fp4Fp8SmemKvResource,
        pack_float4_to_fp8_e4m3,
    )

    function = _function_ast(Fp4Fp8SmemKvResource._store_fragment)
    assignments = {
        ast.unparse(n.targets[0]): ast.unparse(n.value)
        for n in function.body
        if isinstance(n, ast.Assign)
    }
    assert assignments["values"] == "fragment.load().to(cutlass.Float32)"
    pack = next(
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "pack_float4_to_fp8_e4m3"
    )
    assert [ast.unparse(arg) for arg in pack.args] == [
        "values[column]",
        "values[column + 1]",
        "values[column + 2]",
        "values[column + 3]",
    ]
    assert not {
        "descale",
        "main_scale",
    } & assignments.keys() and "fdiv" not in ast.unparse(function)
    offset = next(
        n.value
        for n in function.body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "offset"
    )
    expression = compile(ast.Expression(offset), "<fp8-stage-offset>", "eval")
    offsets = [
        eval(expression, {"__builtins__": {}}, {"row": r, "dim": d})
        for r in range(128)
        for d in range(128)
    ]
    assert sorted(offsets) == list(range(128 * 128))
    for start in range(0, len(offsets), 8):
        assert offsets[start] % 8 == 0 and offsets[start : start + 8] == list(
            range(offsets[start], offsets[start] + 8)
        )
    store = function.body[-1].value
    assert (
        ast.unparse(store.func) == "dst.store"
        and ast.unparse(store.args[0]) == "packed.load(0, 2)"
    )
    assert [(kw.arg, ast.literal_eval(kw.value)) for kw in store.keywords] == [
        ("alignment", 8)
    ]
    assert "cvt.rn.satfinite.e4m3x2.f32" in inspect.getsource(pack_float4_to_fp8_e4m3)
    from .test_deepseek_v41_decode import test_fp4_fp8_all_finite_codes

    probe = _function_ast(test_fp4_fp8_all_finite_codes)
    staging = next(
        n.value
        for n in ast.walk(probe)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "staging"
    )
    assert ast.unparse(staging) == "cutlass.make_array_view(smem)"
    smem = next(
        n.value
        for n in ast.walk(probe)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "smem"
    )
    layout = smem.args[1]
    assert ast.unparse(layout.func) == "cute.make_layout" and isinstance(
        layout.args[0], ast.Tuple
    )
    assert (
        len(layout.args[0].elts) == 1
        and ast.unparse(layout.args[0].elts[0]) == "8 * stage_bytes + 256"
    )
    calls = {ast.unparse(n.func) for n in ast.walk(probe) if isinstance(n, ast.Call)}
    assert {"resource._load_fp4_stage", "resource._load_swa_stage"} <= calls
    factory = next(
        n
        for n in ast.walk(probe)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "attention.kv_producer.make_resource"
    )
    assert not any(kw.arg == "runtime_args" for kw in factory.keywords)


def test_mixed_swa_stage_widens_address_masks_rows_and_fences_all_writers(native):
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
        MixedFp8SmemKvResource,
    )

    function = _function_ast(MixedFp8SmemKvResource._load_swa_stage)
    source = ast.unparse(function)
    assert "cutlass.Int64(slot) * 512 + cutlass.Int64(stage_idx * 128 + dim)" in source
    assert (
        "cutlass.Int64(slot) < capacity" in source and "0x7FFFFFFF" not in source
    )  # AST renders decimal.
    assert "words[word] = cutlass.Int32(0)" in source and "alignment=16" in source
    offset = next(
        child.value
        for node in function.body
        if isinstance(node, ast.For)
        for child in node.body
        if isinstance(child, ast.Assign) and ast.unparse(child.targets[0]) == "offset"
    )
    # The last offset expression is the SW128 store; 16-byte vectors remain
    # disjoint even when adjacent half-chunks exchange their physical order.
    assert ast.unparse(offset) == "row * 128 + dim ^ (row & 7) * 16"
    expression = compile(ast.Expression(offset), "<swa-sw128>", "eval")
    offsets = [
        eval(expression, {"__builtins__": {}}, {"row": row, "dim": dim})
        for row in range(128)
        for dim in range(0, 128, 16)
    ]
    assert len(set(offsets)) == 1024 and all(value % 16 == 0 for value in offsets)
    publish = _function_ast(MixedFp8SmemKvResource._producer_load_kv_stage)
    assert ast.unparse(publish.body[-1].value) == "cute.arch.fence_view_async_shared()"
    assert not any(isinstance(n, ast.Return) for n in ast.walk(publish))
    assert "MlaStage.Head" in ast.unparse(publish) and "MlaStage.Loop" in ast.unparse(
        publish
    )
    assert "inst_id != 0 or is_v != 0" in ast.unparse(publish)


def test_metadata_preparation_never_reads_payload_or_computes_amax(native):
    from flashinfer.experimental.deepseek_v41 import _metadata, prims_ts_resources

    function = _function_ast(_metadata.PrepareMetadata._prepare)
    names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
    assert (
        not {
            "main_cache",
            "swa_cache",
            "pool",
            "main_scale",
            "descale",
            "amax",
            "_fp4_group_amax",
        }
        & names
    )
    assert not hasattr(prims_ts_resources, "_fp4_group_amax")
    source = ast.unparse(function)
    assert "column < main_visible" in source and "lane < swa_visible" in source
    assert (
        "cutlass.Int64(index) < swa_capacity" in source
        and "cutlass.Int64(main_index) < main_capacity" in source
    )
    assert "swa_lengths[request] = cutlass.Int32(128)" in source
    assert "main_lengths[request] = cutlass.Int32(self.compressed_k)" in source
    assert "selected_main[request, lane] = cutlass.Int32(-1)" in source


def test_fp8_native_compiled_abi_is_one_finalizer_with_runtime_bmm_pair(native):
    from flashinfer.experimental.deepseek_v41 import prims_ts, prims_ts_fp8

    function = _function_ast(prims_ts_fp8.Fp8SwaDecodeKernel.__call__)
    assert len(function.args.args[1:-1]) == 11
    assert [arg.arg for arg in function.args.args[-3:]] == [
        "bmm1_scale",
        "bmm2_scale",
        "stream",
    ]
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
    preparation = [n for n in calls if ast.unparse(n.func) == "self.prepare"]
    attention = [n for n in calls if ast.unparse(n.func) == "self.attention"]
    assert not preparation and len(attention) == 1
    assert not hasattr(prims_ts_fp8.Fp8SwaDecodeKernel(object()), "prepare")
    assert ast.unparse(attention[0].args[9]) == "seq_lens"
    assert [ast.unparse(arg) for arg in attention[0].args[:4]] == [
        "q",
        "q",
        "main_cache",
        "swa_cache",
    ]
    assert [ast.unparse(arg) for arg in attention[0].args[-3:]] == [
        "bmm1_scale",
        "bmm2_scale",
        "stream",
    ]
    assert not attention[0].keywords
    names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
    assert (
        not {"part_s", "part_c", "lse_s", "lse_c", "main_scale", "expanded_q"} & names
    )
    direct = next(
        n.value
        for n in function.body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "direct"
    )
    assert [ast.unparse(n) for n in direct.elts] == [
        "indices",
        "None",
        "seq_lens",
        "topk_lens",
        "None",
        "None",
        "None",
        "None",
        "sinks",
    ]
    compiled = _function_ast(inspect.unwrap(prims_ts._compile))
    compile_call = next(
        n
        for n in ast.walk(compiled)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "cute.compile[cute.FrontendNext]"
    )
    assert [ast.unparse(arg) for arg in compile_call.args] == [
        "decode_kernel",
        "q",
        "swa",
        "main_cache",
        "combined",
        "topk_lens",
        "seq_lens",
        "sinks",
        "out",
        "lse",
        "cutlass.Float32(1.0)",
        "cutlass.Float32(1.0)",
        "stream",
    ]


# Caller-buffer shape/overlap/autograd, HND/NHD and capture prerequisites live in
# test_deepseek_v41_workspace_cpu.py. No deleted public plan compatibility is kept.


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
    """Stdlib-only direct E4M3 cache-unit reference.

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
        staged = tuple(bytes(_e4m3_byte(value) for value in row) for row in rows)
        represented = tuple(tuple(_e4m3_value(byte) for byte in row) for row in staged)
        result.append(
            SimpleNamespace(
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
    staged, valid = fp8_main_reference(pool, ids)
    assert staged.dtype == torch.float8_e4m3fn
    assert staged.view(torch.uint8).tolist() == [
        [list(row) for row in request.staged] for request in expected
    ]
    assert valid.tolist() == [list(request.valid) for request in expected]
    assert valid.sum(-1).tolist() == [sum(request.valid) for request in expected]
    represented = staged.double()
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


def test_fp8_main_reference_common_units_signed_scales_saturation_and_replay(native):
    codes = [0] * 512
    codes[0], codes[511] = 5, 5
    scales = [0x7E] * 32
    scales[0], scales[-1] = _e4m3_byte(1.125), _e4m3_byte(3.0)
    pages = _fp4_pages(
        (
            (0, codes, scales),
            (1, [7] * 512, [0x7E] * 32),
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
    assert first.valid == (True, True, True, False, False, False, False)
    assert (
        first.selected == (0, 63, 63, -1, -1, -1, -1)
        and first.staged[1] == first.staged[2]
    )
    assert first.staged[0][0] == _e4m3_byte(3.5) and first.represented[0][0] == 3.5
    assert first.represented[0][511] == 9.0
    assert not hasattr(first, "scale") and not hasattr(first, "amax")
    assert sum(zero.valid) == 2 and sum(empty.valid) == 0
    assert zero.staged[0][:2] == bytes((0x80, 0x00))
    assert math.copysign(1.0, zero.represented[0][0]) == -1.0
    assert all(row == bytes(512) for row in empty.staged)
    changed = [list(row) for row in indices]
    changed[0][0] = 1
    replay = _check_selected_main_reference(native, pages, changed)
    assert replay[0].staged[0] == bytes([0x7E]) * 512
    assert replay[0].represented[0] == (448.0,) * 512
    assert replay[0].staged[1:] == first.staged[1:]
    assert replay[1].staged == zero.staged and replay[2].staged == empty.staged


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
    # Main 1.5 and +/-4 are exactly represented in common E4M3 cache units.
    # Duplicate selected rows keep their multiplicity; SWA stays unchanged.
    small, large = 1.5, 4.0
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
        (3, 1.125, 2, 1.0, 0x3E, 0x38),
        (11, 1.125, 2, 1.0, 0xBE, 0x38),
        (7, 448.0, 2, 1.0, 0x7E, 0x38),
        (15, 448.0, 2, 1.0, 0xFE, 0x38),
        (1, 2**-9, 7, 448.0, 0x00, 0x7E),
        (9, 2**-9, 7, 448.0, 0x80, 0x7E),
        (1, 3 * 2**-7, 7, 448.0, 0x06, 0x7E),
        (9, 3 * 2**-7, 7, 448.0, 0x86, 0x7E),
        (1, 3 * 2**-9, 2, 1.0, 0x02, 0x38),
        (0, -448.0, 7, 448.0, 0x80, 0x7E),
        (8, -448.0, 7, 448.0, 0x00, 0x7E),
        (1, 2**-9, 1, 2**-9, 0x00, 0x00),
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
    represented = _e4m3_value(staged_code)
    anchor = _e4m3_value(anchor_staged)
    if abs(code) in (7, 15) and scale == 448.0:
        assert abs(represented) == 448.0 and anchor == 1.0
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


@pytest.mark.parametrize("section_name", ["Head", "Loop"])
def test_mixed_head_loop_dispatch_fences_each_writer(native, section_name):
    from flashinfer.experimental.deepseek_v41.prims_ts_resources import (
        MixedFp8SmemKvResource,
        MlaStage,
    )

    calls = []
    stage, cache, pages = object(), object(), object()
    resource = SimpleNamespace(
        fp4_cache=cache,
        _stage_base=lambda info: stage,
        _load_swa_stage=lambda *args, **kwargs: calls.append(("swa", args, kwargs)),
        _load_fp4_stage=lambda *args, **kwargs: calls.append(("main", args, kwargs)),
    )
    function = _host_function(
        _function_ast(MixedFp8SmemKvResource._producer_load_kv_stage),
        {
            "cutlass": SimpleNamespace(const_expr=lambda value: value),
            "MlaStage": MlaStage,
            "cute": SimpleNamespace(
                arch=SimpleNamespace(
                    fence_view_async_shared=lambda: calls.append("fence")
                )
            ),
        },
    )
    for writer in range(128):
        for stage_idx in range(4):
            slot = writer % 2
            function(
                resource,
                object(),
                0,
                0,
                stage_idx=stage_idx,
                section=getattr(MlaStage, section_name),
                cached_page_ids=pages,
                page_id_slot=slot,
            )
            source = "swa" if section_name == "Head" else "main"
            args = (stage, pages) if source == "swa" else (cache, stage, pages)
            assert calls[-2:] == [
                (source, args, {"stage_idx": stage_idx, "page_id_slot": slot}),
                "fence",
            ]
    assert len(calls) == 128 * 4 * 2
    for inst_id, is_v, section in (
        (1, 0, MlaStage.Head),
        (0, 1, MlaStage.Loop),
        (0, 0, object()),
    ):
        before = len(calls)
        with pytest.raises(ValueError, match="one stream|unsupported"):
            function(
                resource,
                object(),
                inst_id,
                is_v,
                stage_idx=0,
                section=section,
                cached_page_ids=pages,
            )
        assert len(calls) == before
