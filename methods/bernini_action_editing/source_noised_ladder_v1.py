#!/usr/bin/env python3
"""Shared-noise source ladder for source-aligned Bernini diagnostics.

The construction is a forward flow-matching noising ladder,

    x_source(sigma) = (1 - sigma) * z_source + sigma * epsilon,

The caller is required to reuse the epsilon of a matched edit query.  This
standalone helper cannot verify that external binding; a runtime receipt must
close it separately.  It is intentionally not called inversion: no reverse
ODE, solver state, or round-trip guarantee is provided here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, NoReturn, Sequence

try:
    from . import inference_sigma_strata as exact40
except ImportError:
    import inference_sigma_strata as exact40


SCHEMA_VERSION = "bernini-shared-noise-source-ladder-v1"
DEFAULT_SCHEDULE_INDICES = (16, 29, 35, 38)
DEFAULT_SIGMAS = tuple(
    exact40.PINNED_POSITIVE_SIGMAS[index] for index in DEFAULT_SCHEDULE_INDICES
)


class SourceNoisedLadderError(RuntimeError):
    """Raised before an invalid source coordinate can be constructed."""


def fail(message: str) -> NoReturn:
    raise SourceNoisedLadderError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SourceNoisedLadderError("receipt is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact_sigma(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail("sigma must be a real scalar")
    sigma = float(value)
    if not math.isfinite(sigma) or not 0.0 <= sigma <= 1.0:
        fail("sigma must be finite in [0,1]")
    return sigma


def shared_noise_source_state(source: Any, epsilon: Any, sigma: Any) -> Any:
    """Construct one same-epsilon source state with strict tensor checks."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - remote tensor runtime path
        raise SourceNoisedLadderError("torch is required for tensor construction") from error
    if (
        type(source) is not torch.Tensor
        or type(epsilon) is not torch.Tensor
        or source.shape != epsilon.shape
        or source.device != epsilon.device
        or source.dtype != epsilon.dtype
        or source.layout != torch.strided
        or epsilon.layout != torch.strided
        or source.requires_grad
        or epsilon.requires_grad
        or source.grad_fn is not None
        or epsilon.grad_fn is not None
        or not torch.is_floating_point(source)
        or source.numel() == 0
        or not bool(torch.isfinite(source).all().item())
        or not bool(torch.isfinite(epsilon).all().item())
    ):
        fail("source and epsilon must be matching detached finite dense tensors")
    weight = _exact_sigma(sigma)
    result = ((1.0 - weight) * source + weight * epsilon).detach().contiguous()
    if (
        result.shape != source.shape
        or result.device != source.device
        or result.dtype != source.dtype
        or not bool(torch.isfinite(result).all().item())
    ):
        fail("shared-noise source state is non-finite")
    if weight == 0.0 and not torch.equal(result, source):
        fail("sigma=0 source endpoint differs")
    if weight == 1.0 and not torch.equal(result, epsilon):
        fail("sigma=1 epsilon endpoint differs")
    return result


@dataclass(frozen=True)
class SourceLadderContract:
    schedule_indices: tuple[int, ...] = DEFAULT_SCHEDULE_INDICES
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS
    same_epsilon_as_matched_edit_required: bool = True
    inversion_claimed: bool = False
    exact_roundtrip_claimed: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schedule_indices, tuple)
            or not self.schedule_indices
            or any(type(index) is not int or not 0 <= index < 40 for index in self.schedule_indices)
            or tuple(sorted(set(self.schedule_indices))) != self.schedule_indices
            or not isinstance(self.sigmas, tuple)
            or len(self.sigmas) != len(self.schedule_indices)
        ):
            fail("schedule/sigma registry differs")
        normalized = tuple(_exact_sigma(value) for value in self.sigmas)
        if (
            any(left <= right for left, right in zip(normalized, normalized[1:]))
            or self.schedule_indices != DEFAULT_SCHEDULE_INDICES
            or tuple(value.hex() for value in normalized)
            != tuple(value.hex() for value in DEFAULT_SIGMAS)
            or self.same_epsilon_as_matched_edit_required is not True
            or self.inversion_claimed is not False
            or self.exact_roundtrip_claimed is not False
        ):
            fail("v1 must be a decreasing shared-epsilon forward-noising ladder")
        object.__setattr__(self, "sigmas", normalized)

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "equation": "x_source_sigma=(1-sigma)*z_source+sigma*epsilon",
            "schedule_indices": list(self.schedule_indices),
            "sigmas": list(self.sigmas),
            "exact40_schedule_sha256": exact40.SCHEDULE_SHA256,
            "schedule_cells": [
                {
                    "schedule_index": index,
                    "timestep_int64": exact40.PINNED_TIMESTEPS[index],
                    "sigma_float32_be_hex": exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index],
                }
                for index in self.schedule_indices
            ],
            "same_epsilon_as_matched_edit_required": True,
            "same_epsilon_as_matched_edit_verified": False,
            "clean_source_route_required": True,
            "clean_source_route_verified": False,
            "matched_edit_query_sigma_binding_verified": False,
            "runtime_integration_verified": False,
            "inversion_claimed": self.inversion_claimed,
            "exact_roundtrip_claimed": self.exact_roundtrip_claimed,
            "solver_state_replayed": False,
            "optimizer_authorized": False,
            "method_success_claimed": False,
        }
        return {**value, "receipt_digest": object_sha256(value)}


__all__ = [
    "DEFAULT_SCHEDULE_INDICES",
    "DEFAULT_SIGMAS",
    "SCHEMA_VERSION",
    "SourceLadderContract",
    "SourceNoisedLadderError",
    "object_sha256",
    "shared_noise_source_state",
]
