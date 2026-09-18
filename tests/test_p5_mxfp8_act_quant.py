# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bit-wise alignment tests for P5-1 ``mxfp8_act_quant`` (P5-1 spec).

Both backends (CUDA and Triton) must reproduce the start-kit oracle
(``rl_engine.moe.mx_format.mx_quantize(x, "e4m3")``) byte for byte — not
within a tolerance — because the P5 contract requires train/infer byte
equality on one numeric profile. The tests therefore compare raw ``uint8``
codes and E8M0 scale bytes, the committed golden manifest, and the two
backends against each other.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.moe import fixtures
from rl_engine.moe.contract import tensor_sha256
from rl_engine.moe.mx_format import MX_BLOCK, mx_quantize
from rl_engine.moe.oracle import mxfp8_act_quant_bwd, mxfp8_act_quant_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="mxfp8_act_quant needs an E4M3-capable GPU (SM89+)",
)

DEV = "cuda"


def _backends():
    backends = []
    try:
        from rl_engine.kernels.ops.triton.moe import (
            mxfp8_act_quant_bwd_triton,
            mxfp8_act_quant_fwd_triton,
        )

        backends.append(("triton", mxfp8_act_quant_fwd_triton, mxfp8_act_quant_bwd_triton))
    except ImportError:  # pragma: no cover - triton is an optional dependency
        pass
    from rl_engine.kernels.ops.cuda.moe import mxfp8_act_quant_bwd_cuda, mxfp8_act_quant_fwd_cuda

    # The CUDA backend needs either the AOT rl_engine._C symbols or a working
    # JIT toolchain (nvcc + ninja); without both it is skipped, not failed.
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9):
        try:
            mxfp8_act_quant_fwd_cuda(torch.zeros(1, MX_BLOCK, device=DEV, dtype=torch.bfloat16))
        except RuntimeError as exc:
            _CUDA_BACKEND_ERROR.append(str(exc).splitlines()[0])
        else:
            backends.append(("cuda", mxfp8_act_quant_fwd_cuda, mxfp8_act_quant_bwd_cuda))
    return backends


_CUDA_BACKEND_ERROR: list[str] = []
BACKENDS = _backends()
BACKEND_IDS = [name for name, _, _ in BACKENDS]
_HAS_CUDA_BACKEND = "cuda" in BACKEND_IDS
FWD = [(name, fwd) for name, fwd, _ in BACKENDS]
BWD = [(name, bwd) for name, _, bwd in BACKENDS]


def _assert_byte_equal(got, want, what: str) -> None:
    assert got.codes.dtype == torch.uint8 and got.scales.dtype == torch.uint8
    assert got.shape == want.shape, what
    assert torch.equal(got.scales, want.scales), f"{what}: E8M0 scale bytes differ"
    assert torch.equal(got.codes, want.codes), f"{what}: E4M3 element bytes differ"


def _wide_range(rows: int, cols: int) -> torch.Tensor:
    """Random values spanning ~2**+-60 so scale selection is exercised hard."""
    g = torch.Generator(device="cpu").manual_seed(7)
    mant = torch.randn(rows, cols, generator=g)
    exp = torch.randint(-60, 60, (rows, cols), generator=g).float()
    return (mant * torch.exp2(exp)).to(DEV)


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 32), torch.bfloat16),
        ((1, 128), torch.bfloat16),
        ((24, 128), torch.bfloat16),
        ((37, 64), torch.bfloat16),
        ((512, 4096), torch.bfloat16),
        ((24, 128), torch.float32),
        ((24, 128), torch.float16),
        ((4, 8, 64), torch.bfloat16),
    ],
)
def test_forward_bitwise_matches_oracle(name, fwd, shape, dtype):
    torch.manual_seed(0)
    x = (torch.randn(*shape, device=DEV) * 3.0).to(dtype)
    _assert_byte_equal(fwd(x), mx_quantize(x, "e4m3"), f"{name} {shape} {dtype}")


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_forward_bitwise_edge_fixtures(name, fwd):
    """Start-kit edge inputs: powers of two, RNE ties, zero rows, clamp bounds."""
    x = fixtures.make_act_quant_edge_inputs().to(DEV)
    _assert_byte_equal(fwd(x), mxfp8_act_quant_fwd(x), f"{name} edge fixtures")


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_forward_matches_golden_manifest(name, fwd):
    """The committed CPU-anchored golden hashes, recomputed by the kernels."""
    golden = fixtures.load_manifest()["cases"]["act_quant_edges"]
    got = fwd(fixtures.make_act_quant_edge_inputs().to(DEV))
    assert tensor_sha256(got.codes) == golden["act_quant.codes"]
    assert tensor_sha256(got.scales) == golden["act_quant.scales"]


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
@pytest.mark.parametrize(
    "case",
    ["zeros", "wide_range", "subnormal", "saturating", "all_negative"],
)
def test_forward_bitwise_special_values(name, fwd, case):
    if case == "zeros":
        x = torch.zeros(3, 64, device=DEV, dtype=torch.bfloat16)
    elif case == "wide_range":
        x = _wide_range(64, 128)
    elif case == "subnormal":
        # amax below FLT_MIN: the oracle clamps before floor(log2(.)).
        x = torch.full((4, 32), 1e-40, device=DEV, dtype=torch.float32)
        x[1] = 0.0
    elif case == "saturating":
        # mantissa > 1.75 makes x / scale exceed 448 -> satfinite clamp fires.
        x = torch.full((2, 32), 1.9, device=DEV, dtype=torch.float32)
        x[1, 0] = 1.99
    else:
        x = -(torch.rand(4, 64, device=DEV, dtype=torch.float32) + 0.5)
    _assert_byte_equal(fwd(x), mx_quantize(x, "e4m3"), f"{name} {case}")


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_forward_is_batch_invariant(name, fwd):
    """A row's bytes never depend on how many rows are quantized with it.

    The amax reduction is row-local by contract, so one row alone, the same row
    inside a large batch, and a non-contiguous slice of it must agree bitwise.
    """
    torch.manual_seed(1)
    batch = (torch.randn(257, 256, device=DEV) * 2.0).to(torch.bfloat16)
    full = fwd(batch)
    for row in (0, 1, 42, 256):
        single = fwd(batch[row : row + 1])
        assert torch.equal(single.codes[0], full.codes[row]), f"{name} row {row} codes"
        assert torch.equal(single.scales[0], full.scales[row]), f"{name} row {row} scales"
    strided = fwd(batch[::2])
    assert torch.equal(strided.codes, full.codes[::2]), f"{name} strided codes"


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_forward_is_deterministic(name, fwd):
    x = _wide_range(128, 512).to(torch.bfloat16)
    first = fwd(x)
    for _ in range(5):
        again = fwd(x)
        assert torch.equal(again.codes, first.codes)
        assert torch.equal(again.scales, first.scales)


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_matches_torchao_reference_cast(name, fwd):
    """Third-party cross-check: torchao's OCP-MX cast in FLOOR scaling mode.

    torchao derives the E8M0 scale independently of this repo, so agreeing with
    it byte for byte validates the P5 recipe itself, not just our own oracle.
    """
    to_mx = pytest.importorskip(
        "torchao.prototype.mx_formats.mx_tensor", reason="torchao is an optional baseline"
    ).to_mx
    x = _wide_range(64, 256).to(torch.bfloat16)
    scales, codes = to_mx(x, torch.float8_e4m3fn, MX_BLOCK)
    got = fwd(x)
    assert torch.equal(got.codes, codes.view(torch.uint8))
    assert torch.equal(got.scales, scales.view(torch.uint8).reshape(got.scales.shape))


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_matches_triton_kernels_reference_cast(name, fwd):
    """Cross-check against the MX quantizer vLLM runs for its MXFP4 path.

    ``triton_kernels`` ships with the Triton repo; ROUND_DOWN is the scale mode
    that matches the P5 floor(log2(amax)) recipe.
    """
    mxfp = pytest.importorskip(
        "triton_kernels.numerics_details.mxfp", reason="triton_kernels is an optional baseline"
    )
    x = _wide_range(64, 256).to(torch.bfloat16)
    codes, scales = mxfp.downcast_to_mxfp(
        x,
        torch.float8_e4m3fn,
        axis=-1,
        DEQUANT_SCALE_ROUNDING_MODE=mxfp.DequantScaleRoundingMode.ROUND_DOWN,
    )
    got = fwd(x)
    assert torch.equal(got.codes, codes.view(torch.uint8))
    assert torch.equal(got.scales, scales.view(torch.uint8).reshape(got.scales.shape))


def test_backends_agree_bitwise():
    if len(BACKENDS) < 2:
        pytest.skip("needs both the CUDA and the Triton backend")
    x = _wide_range(64, 256).to(torch.bfloat16)
    (_, fwd_a, _), (_, fwd_b, _) = BACKENDS[0], BACKENDS[1]
    _assert_byte_equal(fwd_a(x), fwd_b(x), "triton vs cuda")


@pytest.mark.parametrize("name,bwd", BWD, ids=BACKEND_IDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_backward_is_straight_through(name, bwd, dtype):
    """STE: dX = dY exactly for any floating dtype (P5-1 spec contract table)."""
    dy = (torch.randn(64, 128, device=DEV) * 4.0).to(dtype)
    dx = bwd(dy)
    assert dx.dtype == dy.dtype
    assert torch.equal(dx, dy)
    assert torch.equal(dx, mxfp8_act_quant_bwd(dy))


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_forward_is_fail_closed_on_non_finite(name, fwd, bad):
    x = torch.randn(2, 64, device=DEV, dtype=torch.float32)
    x[1, 5] = bad
    with pytest.raises(ValueError, match="non-finite"):
        fwd(x)


@pytest.mark.parametrize("name,fwd", FWD, ids=BACKEND_IDS)
def test_forward_rejects_bad_shapes_and_devices(name, fwd):
    with pytest.raises(ValueError, match="MX block"):
        fwd(torch.randn(2, MX_BLOCK - 1, device=DEV, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="CUDA"):
        fwd(torch.randn(2, MX_BLOCK, dtype=torch.bfloat16))
    with pytest.raises(TypeError):
        fwd(torch.randn(2, MX_BLOCK, device=DEV, dtype=torch.float64))


@pytest.mark.parametrize(
    "provider_cls",
    ["TritonMXFP8ActQuantProvider", "CudaMXFP8ActQuantProvider"],
)
def test_provider_pipeline_is_byte_equal_to_oracle(provider_cls):
    """Full routed forward/backward with the kernel provider, hashed per boundary.

    This is the start-kit acceptance path (``scripts/check_p5.py``) narrowed to
    one fixture so it stays cheap in unit-test CI.
    """
    from rl_engine.moe import oracle
    from rl_engine.moe.backends import mxfp8_act_quant as backends
    from rl_engine.moe.trace import ExpertTrace, first_divergence

    if provider_cls == "CudaMXFP8ActQuantProvider" and not _HAS_CUDA_BACKEND:
        pytest.skip(f"CUDA backend unavailable: {_CUDA_BACKEND_ERROR}")
    provider = getattr(backends, provider_cls)()
    batch = fixtures.make_expert_batch("base_only_packed").to(DEV)

    gold_trace = ExpertTrace(numeric_profile="oracle")
    y_gold, saved_gold = oracle.routed_expert_forward(batch, gold_trace)
    dy = fixtures.make_grad_output("base_only_packed", tuple(y_gold.shape)).to(DEV)
    grads_gold = oracle.routed_expert_backward(batch, saved_gold, dy, gold_trace)

    cand_trace = ExpertTrace(numeric_profile=provider.numeric_profile)
    _, saved_cand = oracle.routed_expert_forward(batch, cand_trace, ops=provider)
    grads_cand = oracle.routed_expert_backward(batch, saved_cand, dy, cand_trace, ops=provider)

    assert first_divergence(gold_trace, cand_trace) is None
    # P5-1 spec s6: both call sites are hashed and the backward is marked STE.
    names = [r.name for r in cand_trace.records]
    for boundary in (
        "act_quant1.codes",
        "act_quant1.scales",
        "act_quant2.codes",
        "act_quant2.scales",
    ):
        assert boundary in names
    assert cand_trace.notes["act_quant_bwd"] == "ste"
    for key, grad in grads_gold.items():
        if grad is None:
            assert grads_cand[key] is None
            continue
        assert tensor_sha256(grads_cand[key]) == tensor_sha256(grad), f"grad.{key}"


def test_cuda_kernel_is_fast_math_immune(tmp_path):
    """A --use_fast_math build must emit the same bytes as the oracle.

    nvcc turns fmaxf/fabsf/__fdiv_rn into .ftz forms under fast-math; the
    kernel keeps every subnormal-sensitive step on integer bit patterns or
    explicit non-.ftz PTX, so subnormal inputs and the 2**-127 scale survive.
    """
    if not _HAS_CUDA_BACKEND:
        pytest.skip(f"CUDA backend unavailable: {_CUDA_BACKEND_ERROR}")
    from torch.utils.cpp_extension import load

    from rl_engine.kernels.ops.cuda.moe import mxfp8_act_quant as wrapper

    module = load(
        name="rl_engine_p5_mxfp8_act_quant_fastmath",
        sources=[str(wrapper._CU_SOURCE)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-DRL_KERNEL_P5_STANDALONE"],
        build_directory=str(tmp_path),
        verbose=False,
    )
    cases = [
        torch.full((4, 32), 1e-40, device=DEV, dtype=torch.float32),  # subnormal amax
        _wide_range(64, 128),
        fixtures.make_act_quant_edge_inputs().to(DEV),
    ]
    for x in cases:
        ref = mx_quantize(x, "e4m3")
        codes, scales, flag = module.mxfp8_act_quant_forward(x.contiguous(), True)
        assert int(flag.item()) == 0
        assert torch.equal(scales, ref.scales), "fast-math build changed E8M0 scale bytes"
        assert torch.equal(codes, ref.codes), "fast-math build changed E4M3 element bytes"
