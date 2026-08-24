#!/usr/bin/env python3
"""Math for SAIC late/low-sigma source-appearance anchor pretraining.

This objective is deliberately independent of an edited target.  The clean
endpoint is the real exact81 source latent.  Its *conditioning* copy is
temporally scrambled before Bernini sees it, while the four native reference
frames remain source-derived.  Consequently the late self-attention adapter
can learn source appearance/dependence without receiving the original motion
order as an easy conditioning shortcut.

The objective has two non-substitutable roles:

* correct-source flow matching must reconstruct the real source endpoint; and
* an independently selected wrong source must be worse by a fixed margin.

The second term is a ranking constraint, not a replacement target.  This file
does not authenticate media, construct a Bernini model call, or authorize an
optimizer; the runtime must bind those artifacts and is responsible for
serial output-leaf VJP replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import torch
from torch.nn import functional as F


SCHEMA_VERSION = "bernini-saic-source-anchor-objective-v1"
EXACT_LATENT_CHANNELS = 16
EXACT_LATENT_PHASES = 21
ACTIVE_SIGMA_INDICES = (35, 36, 37, 38, 39)
DEFAULT_WRONG_SOURCE_MARGIN = 0.01
DEFAULT_RANKING_WEIGHT = 1.0


class SAICSourceAnchorObjectiveError(RuntimeError):
    """Raised before an ambiguous source-anchor cell is accepted."""


def _exact81_latent(
    value: Any, *, label: str, detached: bool | None = None
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.device.type == "meta"
        or value.layout != torch.strided
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3])
        != (1, EXACT_LATENT_CHANNELS, EXACT_LATENT_PHASES)
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
        or not value.is_floating_point()
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise SAICSourceAnchorObjectiveError(
            f"{label} must be finite exact81 [1,16,21,H,W]"
        )
    if detached is True and (value.requires_grad or value.grad_fn is not None):
        raise SAICSourceAnchorObjectiveError(f"{label} must be detached")
    # Serial VJP training deliberately supplies measured output *leaves*.
    # Requiring grad_fn here would reject that memory-bounded exact replay
    # pattern; requires_grad is the correct objective-boundary contract.
    if detached is False and not value.requires_grad:
        raise SAICSourceAnchorObjectiveError(f"{label} must require an output cotangent")
    return value


def temporal_scramble_indices(*, seed: int) -> tuple[int, ...]:
    """Return a deterministic non-identity/non-reversal permutation of 21 phases."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise SAICSourceAnchorObjectiveError("scramble seed must lie in [0,2^63)")
    material = hashlib.sha256(
        f"saic-source-anchor-scramble-v1\0{seed}".encode("ascii")
    ).digest()
    # Every multiplier below is coprime to 21 and excludes +1/-1.
    multipliers = (2, 4, 5, 8, 10, 11, 13, 16, 17, 19)
    multiplier = multipliers[int.from_bytes(material[:2], "big") % len(multipliers)]
    offset = 1 + int.from_bytes(material[2:4], "big") % 20
    order = tuple((multiplier * index + offset) % EXACT_LATENT_PHASES for index in range(21))
    if (
        len(set(order)) != EXACT_LATENT_PHASES
        or order == tuple(range(EXACT_LATENT_PHASES))
        or order == tuple(reversed(range(EXACT_LATENT_PHASES)))
    ):
        raise SAICSourceAnchorObjectiveError("temporal scramble construction failed")
    return order


def scramble_source_condition(
    source_clean_latent: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Scramble only the source-condition timeline; never the clean endpoint."""

    source = _exact81_latent(
        source_clean_latent, label="source clean latent", detached=True
    )
    order = temporal_scramble_indices(seed=seed)
    index = torch.tensor(order, dtype=torch.long, device=source.device)
    scrambled = source.index_select(2, index).detach().contiguous()
    if torch.equal(scrambled, source):
        raise SAICSourceAnchorObjectiveError(
            "source happens to be invariant to the registered scramble"
        )
    return scrambled, order


def build_source_flow_state(
    source_clean_latent: torch.Tensor,
    gaussian_noise: torch.Tensor,
    *,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build detached rectified-flow state and target ``epsilon - source``."""

    source = _exact81_latent(
        source_clean_latent, label="source clean latent", detached=True
    ).float()
    noise = _exact81_latent(
        gaussian_noise, label="official Gaussian", detached=True
    ).float()
    if tuple(source.shape) != tuple(noise.shape) or source.device != noise.device:
        raise SAICSourceAnchorObjectiveError("source and Gaussian geometry differ")
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise SAICSourceAnchorObjectiveError("sigma must be a real scalar")
    weight = float(sigma)
    if not math.isfinite(weight) or not 0.0 < weight < 1.0:
        raise SAICSourceAnchorObjectiveError("sigma must lie strictly inside (0,1)")
    state = ((1.0 - weight) * source + weight * noise).detach().contiguous()
    target = (noise - source).detach().contiguous()
    return state, target


@dataclass(frozen=True)
class SourceAnchorObjective:
    loss: torch.Tensor
    correct_flow_loss: torch.Tensor
    wrong_source_flow_loss: torch.Tensor
    wrong_source_advantage: torch.Tensor
    ranking_hinge: torch.Tensor
    wrong_source_margin: float
    ranking_weight: float


def build_source_anchor_objective(
    *,
    correct_source_prediction: torch.Tensor,
    wrong_source_prediction: torch.Tensor,
    source_flow_target: torch.Tensor,
    wrong_source_margin: float = DEFAULT_WRONG_SOURCE_MARGIN,
    ranking_weight: float = DEFAULT_RANKING_WEIGHT,
) -> SourceAnchorObjective:
    """Require source reconstruction and nontrivial correct-source dependence."""

    correct = _exact81_latent(
        correct_source_prediction,
        label="correct-source prediction",
        detached=False,
    ).float()
    wrong = _exact81_latent(
        wrong_source_prediction,
        label="wrong-source prediction",
        detached=False,
    ).float()
    target = _exact81_latent(
        source_flow_target, label="source flow target", detached=True
    ).float()
    if (
        tuple(correct.shape) != tuple(wrong.shape)
        or tuple(correct.shape) != tuple(target.shape)
        or correct.device != wrong.device
        or correct.device != target.device
    ):
        raise SAICSourceAnchorObjectiveError("prediction/target geometry differs")
    for label, value in (
        ("wrong_source_margin", wrong_source_margin),
        ("ranking_weight", ranking_weight),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SAICSourceAnchorObjectiveError(f"{label} must be a real scalar")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise SAICSourceAnchorObjectiveError(f"{label} must be finite and positive")
    correct_per_sample = (correct - target).square().flatten(start_dim=1).mean(dim=1)
    wrong_per_sample = (wrong - target).square().flatten(start_dim=1).mean(dim=1)
    advantage = wrong_per_sample - correct_per_sample
    margin_tensor = advantage.new_tensor(float(wrong_source_margin))
    hinge = F.relu(margin_tensor - advantage)
    correct_loss = correct_per_sample.mean()
    wrong_loss = wrong_per_sample.mean()
    ranking_hinge = hinge.mean()
    loss = correct_loss + float(ranking_weight) * ranking_hinge
    if (
        loss.dtype != torch.float32
        or loss.ndim != 0
        or not loss.requires_grad
        or loss.grad_fn is None
        or not bool(torch.isfinite(loss.detach()).item())
    ):
        raise SAICSourceAnchorObjectiveError(
            "source-anchor loss must be finite graph-connected FP32"
        )
    return SourceAnchorObjective(
        loss=loss,
        correct_flow_loss=correct_loss,
        wrong_source_flow_loss=wrong_loss,
        wrong_source_advantage=advantage.mean(),
        ranking_hinge=ranking_hinge,
        wrong_source_margin=float(wrong_source_margin),
        ranking_weight=float(ranking_weight),
    )


def validate_active_sigma_index(index: Any) -> int:
    if isinstance(index, bool) or not isinstance(index, int) or index not in ACTIVE_SIGMA_INDICES:
        raise SAICSourceAnchorObjectiveError(
            "source anchor is restricted to exact40 indices 35..39"
        )
    return index


__all__ = [
    "ACTIVE_SIGMA_INDICES",
    "DEFAULT_RANKING_WEIGHT",
    "DEFAULT_WRONG_SOURCE_MARGIN",
    "SAICSourceAnchorObjectiveError",
    "SCHEMA_VERSION",
    "SourceAnchorObjective",
    "build_source_anchor_objective",
    "build_source_flow_state",
    "scramble_source_condition",
    "temporal_scramble_indices",
    "validate_active_sigma_index",
]
