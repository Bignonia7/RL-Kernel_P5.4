# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA MXFP8 activation quantization (P5-1).

Thin wrapper around ``csrc/cuda/moe/mxfp8_act_quant.cu``. The kernel reproduces
``rl_engine.moe.mx_format.mx_quantize(x, "e4m3")`` byte for byte (numeric
profile ``oracle-fp32-serial-v1``); see the ``.cu`` header for the frozen
recipe and how it stays immune to ``--use_fast_math``.

The compiled ``rl_engine._C`` extension is used when it exports the symbols. A
source tree without a built extension falls back to a JIT build of that single
``.cu`` file (``torch.utils.cpp_extension.load``) so the alignment tests and
the benchmark run without rebuilding the whole extension. The fallback is
CUDA-only (ROCm must register its own numeric profile), a failed build stays
failed for the process, and ``RL_KERNEL_P5_DISABLE_JIT=1`` turns it off so the
AOT symbols are required, like every other op in the tree.
"""

from __future__ import annotations

import pathlib
import threading
from typing import Any

import torch
from torch import Tensor

import envs
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.moe.mx_format import MXTensor, validate_act_quant_input, validate_ste_grad
from rl_engine.utils.logger import logger

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
_CU_SOURCE = _REPO_ROOT / "csrc" / "cuda" / "moe" / "mxfp8_act_quant.cu"
_AOT_SYMBOLS = ("mxfp8_act_quant_forward", "mxfp8_act_quant_ste_backward")

_jit_lock = threading.Lock()
_jit_module: Any = None
_jit_error: BaseException | None = None


def _aot_available() -> bool:
    return _EXT_AVAILABLE and _C is not None and all(hasattr(_C, s) for s in _AOT_SYMBOLS)


def _jit_load() -> Any:
    """Build the single P5-1 translation unit on demand (once per process)."""
    global _jit_module, _jit_error
    with _jit_lock:
        if _jit_module is not None:
            return _jit_module
        if _jit_error is not None:
            # torch's JIT versioner would otherwise report a misleading
            # "cannot open shared object" on every later call.
            raise RuntimeError("P5-1 CUDA JIT build failed earlier in this process") from _jit_error
        try:
            if envs.env_flag(envs.RL_KERNEL_P5_DISABLE_JIT):
                raise RuntimeError(
                    "CUDA mxfp8_act_quant requires the compiled rl_engine._C extension "
                    "(rebuild with csrc/cuda/moe/mxfp8_act_quant.cu); the JIT fallback "
                    f"is disabled by {envs.RL_KERNEL_P5_DISABLE_JIT}=1."
                )
            if torch.version.hip is not None:
                raise RuntimeError(
                    "CUDA mxfp8_act_quant has no ROCm build: the source needs cuda_fp8.h. "
                    "ROCm must register its own P5 numeric profile."
                )
            if not _CU_SOURCE.exists():
                raise RuntimeError(f"P5-1 CUDA source not found at {_CU_SOURCE}")
            from torch.utils.cpp_extension import load

            logger.warning(
                "rl_engine._C has no mxfp8_act_quant symbols; JIT-building "
                f"{_CU_SOURCE.name} (first call only)."
            )
            _jit_module = load(
                name="rl_engine_p5_mxfp8_act_quant",
                sources=[str(_CU_SOURCE)],
                extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DRL_KERNEL_P5_STANDALONE"],
                verbose=False,
            )
        except BaseException as exc:
            _jit_error = exc
            raise
        return _jit_module


def _backend() -> Any:
    return _C if _aot_available() else _jit_load()


def backend_name() -> str:
    """``"aot"`` when the symbols come from ``rl_engine._C``, else ``"jit"``."""
    return "aot" if _aot_available() else "jit"


def mxfp8_act_quant_fwd_cuda(x: Tensor, check_finite: bool = True) -> MXTensor:
    """BF16/FP16/FP32 ``[..., K]`` -> MXFP8 (E4M3 codes + block-32 E8M0 scales).

    ``check_finite`` reads back the kernel's fail-closed flag and therefore
    costs one device sync per call; callers that batch the check (or measure
    throughput) turn it off, which also skips the flag memset.
    """
    x = validate_act_quant_input(x, "cuda")
    codes, scales, nonfinite = _backend().mxfp8_act_quant_forward(x, check_finite)
    if check_finite and bool(nonfinite.item()):
        raise ValueError("non-finite values in mx_quantize input; P5 quantization is fail-closed")
    return MXTensor(codes=codes, scales=scales, elem_format="e4m3", shape=tuple(x.shape))


def mxfp8_act_quant_bwd_cuda(dy: Tensor) -> Tensor:
    """Straight-through estimator: ``dX = dY`` for any floating dtype (contiguous, same shape)."""
    return _backend().mxfp8_act_quant_ste_backward(validate_ste_grad(dy, "cuda"))
