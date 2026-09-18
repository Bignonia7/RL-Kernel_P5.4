# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5-1 (P5-1) providers: MXFP8 activation quantization backends.

Each provider subclasses :class:`~rl_engine.moe.provider.ReferenceProvider` and
overrides only the operators its backend delivers, per the start-kit protocol
(``docs/design/dsv4_p5_expert_start_kit.md``); everything else stays on the
FP32 oracle so ``scripts/check_p5.py`` runs end to end from day one.

Both P5-1 backends are byte-equal with the oracle, so they inherit the oracle's
numeric profile instead of registering a relaxed one:

    python scripts/check_p5.py --provider \
        rl_engine.moe.backends.mxfp8_act_quant:TritonMXFP8ActQuantProvider --device cuda
    python scripts/check_p5.py --provider \
        rl_engine.moe.backends.mxfp8_act_quant:CudaMXFP8ActQuantProvider --device cuda

Fail-closed: a CPU tensor or a non-finite input raises instead of silently
falling back to the oracle. P5-1 spec s4 makes the non-finite raise part of the
contract, so the providers always run the kernels' read-back (one device sync
per call); the kernels' ``check_finite=False`` exists for throughput
measurement only and is deliberately not reachable from a provider.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.moe.mx_format import MXTensor
from rl_engine.moe.provider import ReferenceProvider


class _ActQuantProviderBase(ReferenceProvider):
    """Shared plumbing for the P5-1 kernel providers."""

    backend = "unset"

    def _ops(self) -> Any:
        raise NotImplementedError

    def capabilities(self) -> dict[str, Any]:
        return {
            **super().capabilities(),
            "backend": self.backend,
            "operators": ["mxfp8_act_quant"],
            "devices": ["cuda"],
            "dtypes": ["bfloat16", "float16", "float32"],
            "byte_equal_with_oracle": True,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            **super().provenance(),
            "operators_overridden": ["mxfp8_act_quant_fwd", "mxfp8_act_quant_bwd"],
            "linkage": self._linkage(),
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        }

    def _linkage(self) -> str:
        return "python"

    def mxfp8_act_quant_fwd(self, x: torch.Tensor) -> MXTensor:
        return self._ops().fwd(x)

    def mxfp8_act_quant_bwd(self, dy: torch.Tensor) -> torch.Tensor:
        return self._ops().bwd(dy)


class _Ops:
    def __init__(self, fwd, bwd):
        self.fwd = fwd
        self.bwd = bwd


class TritonMXFP8ActQuantProvider(_ActQuantProviderBase):
    """Triton MXFP8 activation quantization; every other operator is the oracle."""

    name = "mxfp8-act-quant-triton"
    backend = "triton"

    def _ops(self) -> _Ops:
        from rl_engine.kernels.ops.triton import moe

        return _Ops(moe.mxfp8_act_quant_fwd_triton, moe.mxfp8_act_quant_bwd_triton)

    def _linkage(self) -> str:
        return "triton-jit"


class CudaMXFP8ActQuantProvider(_ActQuantProviderBase):
    """CUDA MXFP8 activation quantization; every other operator is the oracle."""

    name = "mxfp8-act-quant-cuda"
    backend = "cuda"

    def _ops(self) -> _Ops:
        from rl_engine.kernels.ops.cuda import moe

        return _Ops(moe.mxfp8_act_quant_fwd_cuda, moe.mxfp8_act_quant_bwd_cuda)

    def _linkage(self) -> str:
        from rl_engine.kernels.ops.cuda.moe import backend_name

        return backend_name()  # "aot" (rl_engine._C) or "jit"
