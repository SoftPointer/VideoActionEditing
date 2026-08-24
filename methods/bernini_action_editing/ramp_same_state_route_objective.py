#!/usr/bin/env python3
"""Same-state sigma=1 causal objective for Bernini RAMP-Edit.

At the rectified-flow noise endpoint, two exact motion-program arms share the
same source, text, and Gaussian target state ``x_1 = epsilon``.  Their clean
targets differ only because each donor packet specifies a different registered
program.  The native flow targets therefore obey

``v_a* - v_b* = (epsilon-z_a) - (epsilon-z_b) = z_b-z_a``.

This module turns that identity into a differentiable route loss.  It does not
run Bernini, accept donor videos, or authorize natural-action training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F


SCHEMA_VERSION = "bernini-ramp-sigma-one-same-state-route-objective-v1"
NORMALIZATION_EPSILON = 1.0e-6


class RAMPSameStateObjectiveError(ValueError):
    """The paired donor intervention is not a true same-state comparison."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RAMPSameStateObjectiveError(f"receipt is not canonical JSON: {error}") from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RAMPSameStateObjectiveError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class SameStateInterventionIdentity:
    """Auditable metadata held equal across a two-program intervention."""

    source_sha256: str
    text_sha256: str
    epsilon_sha256: str
    noisy_target_sha256: str
    timestep_token: str
    program_a_sha256: str
    program_b_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_sha256",
            "text_sha256",
            "epsilon_sha256",
            "noisy_target_sha256",
            "program_a_sha256",
            "program_b_sha256",
        ):
            _digest(getattr(self, name), label=name)
        if self.program_a_sha256 == self.program_b_sha256:
            raise RAMPSameStateObjectiveError("program interventions must be distinct")
        if self.timestep_token not in {"sigma=1", "1.0", "0x1.0000000000000p+0"}:
            raise RAMPSameStateObjectiveError("same-state route loss requires exact sigma=1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "text_sha256": self.text_sha256,
            "epsilon_sha256": self.epsilon_sha256,
            "noisy_target_sha256": self.noisy_target_sha256,
            "timestep_token": self.timestep_token,
            "program_a_sha256": self.program_a_sha256,
            "program_b_sha256": self.program_b_sha256,
            "held_equal": ["source", "text", "epsilon", "x_sigma", "sigma"],
            "changed_only": "relative_motion_program",
        }


def _detached_fp32(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RAMPSameStateObjectiveError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided or value.device.type == "meta":
        raise RAMPSameStateObjectiveError(f"{name} must be dense and materialized")
    if value.dtype != torch.float32:
        raise RAMPSameStateObjectiveError(f"{name} must be FP32")
    if value.requires_grad or value.grad_fn is not None:
        raise RAMPSameStateObjectiveError(f"{name} must be detached")
    if value.ndim < 2 or any(int(size) <= 0 for size in value.shape):
        raise RAMPSameStateObjectiveError(f"{name} has invalid batch-first geometry")
    if not value.is_contiguous():
        raise RAMPSameStateObjectiveError(f"{name} must be contiguous")
    if not bool(torch.isfinite(value).all().item()):
        raise RAMPSameStateObjectiveError(f"{name} contains NaN or infinity")
    return value


def _prediction(name: str, value: Any, *, shape: torch.Size, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RAMPSameStateObjectiveError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided or value.device.type == "meta":
        raise RAMPSameStateObjectiveError(f"{name} must be dense and materialized")
    if value.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise RAMPSameStateObjectiveError(f"{name} must use a supported floating dtype")
    if value.shape != shape or value.device != device:
        raise RAMPSameStateObjectiveError(f"{name} shape/device differs from targets")
    if not value.requires_grad:
        raise RAMPSameStateObjectiveError(f"{name} must retain a training graph")
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise RAMPSameStateObjectiveError(f"{name} contains NaN or infinity")
    return value.float()


def _per_sample_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim < 2:
        raise RAMPSameStateObjectiveError("MSE tensors must share batch-first shape")
    dimensions = tuple(range(1, left.ndim))
    return (left - right).square().mean(dim=dimensions)


@dataclass(frozen=True)
class SameStateRouteResult:
    total_loss: torch.Tensor
    flow_matching_loss: torch.Tensor
    route_loss: torch.Tensor
    donor_identity_invariance_loss: torch.Tensor
    order_invariance_loss: torch.Tensor
    map_loss: torch.Tensor
    target_velocity_a: torch.Tensor
    target_velocity_b: torch.Tensor
    target_delta: torch.Tensor
    prediction_delta: torch.Tensor
    normalization_energy: torch.Tensor
    route_explained_fraction: torch.Tensor
    own_target_ranking: torch.Tensor
    receipt: Mapping[str, Any]


def _soft_transport_cross_entropy(
    transport_logits: Any,
    transport_target: Any,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    logits = transport_logits
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RAMPSameStateObjectiveError("transport_logits must be [B,T,T]")
    if int(logits.shape[0]) != batch_size or int(logits.shape[1]) != int(logits.shape[2]):
        raise RAMPSameStateObjectiveError("transport_logits must be square with matching batch")
    if logits.device != device or logits.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }:
        raise RAMPSameStateObjectiveError("transport_logits device/dtype differs")
    if not logits.requires_grad or not bool(torch.isfinite(logits.detach()).all().item()):
        raise RAMPSameStateObjectiveError("transport_logits must be finite and differentiable")

    target = _detached_fp32("transport_target", transport_target)
    if target.ndim == 2:
        target = target.unsqueeze(0).expand(batch_size, -1, -1).clone().contiguous()
    if target.shape != logits.shape or target.device != device:
        raise RAMPSameStateObjectiveError("transport_target must match [B,T,T]")
    if bool((target < 0.0).any().item()) or not torch.allclose(
        target.sum(dim=-1),
        torch.ones_like(target.sum(dim=-1)),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RAMPSameStateObjectiveError("transport_target rows must be probabilities")
    return -(target * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def sigma_one_same_state_route_objective(
    prediction_a: Any,
    prediction_b: Any,
    clean_target_a: Any,
    clean_target_b: Any,
    shared_epsilon: Any,
    *,
    identity: SameStateInterventionIdentity,
    donor_identity_prediction_a: Any,
    donor_identity_prediction_b: Any,
    order_prediction_a: Any,
    order_prediction_b: Any,
    transport_logits: Any,
    transport_target: Any,
    route_weight: float = 0.5,
    donor_identity_weight: float = 0.1,
    order_weight: float = 0.05,
    map_weight: float = 0.2,
) -> SameStateRouteResult:
    """Compute the preregistered RAMP C1 sigma=1 paired objective."""

    if not isinstance(identity, SameStateInterventionIdentity):
        raise RAMPSameStateObjectiveError("identity has the wrong type")
    identity.__post_init__()
    weights = {
        "route": route_weight,
        "donor_identity": donor_identity_weight,
        "order": order_weight,
        "map": map_weight,
    }
    expected = {"route": 0.5, "donor_identity": 0.1, "order": 0.05, "map": 0.2}
    if weights != expected:
        raise RAMPSameStateObjectiveError(f"C1 canary weights must equal {expected}")

    target_a = _detached_fp32("clean_target_a", clean_target_a)
    target_b = _detached_fp32("clean_target_b", clean_target_b)
    epsilon = _detached_fp32("shared_epsilon", shared_epsilon)
    if target_a.shape != target_b.shape or target_a.shape != epsilon.shape:
        raise RAMPSameStateObjectiveError("clean targets and shared epsilon must match exactly")
    if not (target_a.device == target_b.device == epsilon.device):
        raise RAMPSameStateObjectiveError("clean targets and shared epsilon devices differ")
    if torch.equal(target_a, target_b):
        raise RAMPSameStateObjectiveError("distinct programs produced byte-equal clean targets")

    pred_a = _prediction("prediction_a", prediction_a, shape=target_a.shape, device=target_a.device)
    pred_b = _prediction("prediction_b", prediction_b, shape=target_a.shape, device=target_a.device)
    donor_a = _prediction(
        "donor_identity_prediction_a",
        donor_identity_prediction_a,
        shape=target_a.shape,
        device=target_a.device,
    )
    donor_b = _prediction(
        "donor_identity_prediction_b",
        donor_identity_prediction_b,
        shape=target_a.shape,
        device=target_a.device,
    )
    order_a = _prediction(
        "order_prediction_a", order_prediction_a, shape=target_a.shape, device=target_a.device
    )
    order_b = _prediction(
        "order_prediction_b", order_prediction_b, shape=target_a.shape, device=target_a.device
    )

    velocity_a = (epsilon - target_a).detach().contiguous()
    velocity_b = (epsilon - target_b).detach().contiguous()
    target_delta = (target_b - target_a).detach().contiguous()
    prediction_delta = pred_a - pred_b
    energy = 0.5 * (
        _per_sample_mse(velocity_a, torch.zeros_like(velocity_a))
        + _per_sample_mse(velocity_b, torch.zeros_like(velocity_b))
    ) + NORMALIZATION_EPSILON

    fm_per = 0.5 * (
        _per_sample_mse(pred_a, velocity_a) + _per_sample_mse(pred_b, velocity_b)
    )
    delta_mse = _per_sample_mse(prediction_delta, target_delta)
    route_per = delta_mse / energy
    donor_per = _per_sample_mse(donor_a, donor_b) / energy
    order_per = _per_sample_mse(order_a, order_b) / energy
    map_loss = _soft_transport_cross_entropy(
        transport_logits,
        transport_target,
        batch_size=int(target_a.shape[0]),
        device=target_a.device,
    )

    fm_loss = fm_per.mean()
    route_loss = route_per.mean()
    donor_loss = donor_per.mean()
    order_loss = order_per.mean()
    total = (
        fm_loss
        + route_weight * route_loss
        + donor_identity_weight * donor_loss
        + order_weight * order_loss
        + map_weight * map_loss
    )
    if not bool(torch.isfinite(total.detach()).item()):
        raise RAMPSameStateObjectiveError("combined route objective is non-finite")

    baseline_delta_energy = _per_sample_mse(torch.zeros_like(target_delta), target_delta)
    explained = 1.0 - delta_mse / baseline_delta_energy.clamp_min(NORMALIZATION_EPSILON)
    own_a = _per_sample_mse(pred_a, velocity_a) < _per_sample_mse(pred_a, velocity_b)
    own_b = _per_sample_mse(pred_b, velocity_b) < _per_sample_mse(pred_b, velocity_a)
    own_ranking = own_a & own_b
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity.as_dict(),
        "sigma": 1.0,
        "noisy_target_equation": "x_1=shared_epsilon",
        "native_velocity_equation": "v*=epsilon-clean_target",
        "delta_identity": "v_a*-v_b*=z_b-z_a",
        "weights": expected,
        "wrong_donor_degradation_margin_trained": False,
        "natural_action_training_authorized": False,
    }
    receipt["receipt_digest"] = _sha256(_canonical_json(receipt))
    return SameStateRouteResult(
        total_loss=total,
        flow_matching_loss=fm_loss,
        route_loss=route_loss,
        donor_identity_invariance_loss=donor_loss,
        order_invariance_loss=order_loss,
        map_loss=map_loss,
        target_velocity_a=velocity_a,
        target_velocity_b=velocity_b,
        target_delta=target_delta,
        prediction_delta=prediction_delta,
        normalization_energy=energy,
        route_explained_fraction=explained,
        own_target_ranking=own_ranking,
        receipt=receipt,
    )


__all__ = [
    "NORMALIZATION_EPSILON",
    "RAMPSameStateObjectiveError",
    "SCHEMA_VERSION",
    "SameStateInterventionIdentity",
    "SameStateRouteResult",
    "sigma_one_same_state_route_objective",
]
