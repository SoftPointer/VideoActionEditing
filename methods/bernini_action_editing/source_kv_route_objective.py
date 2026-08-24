#!/usr/bin/env python3
"""Carrier-separated motion-route objective for Bernini action editing V9.

The source-only K/V carrier is an attention memory, not a decoded appearance
field.  Five equal-state ``[source, target]`` branches read that memory and the
two adapted branches learn the complete action/no-op quotient::

    q_theta = Q0(A_theta - N_theta)
    q_0     = stopgrad(Q0(A_0 - N_0))
    q_star  = stopgrad(Q0(T_executable - S))

Here ``Q0(x) = x - x[:, :1]`` fixes the temporal gauge without supplying an
image, mask, pose, track, flow, trajectory, or first-frame condition.  The
paired target is an offline training label and is never an inference input.

Unlike V8, V9 does not radially project the teacher or prediction.  Low and
high temporal bands are supervised separately, and a frozen-prior direction
loss is enabled only where the frozen Bernini action prior agrees with the
paired motion.  A raw phase-zero penalty blocks the observed failure in which
an edited object is painted into frame zero before the requested action has
occurred.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


METHOD_NAME = "bernini-carrier-separated-value-action-route-v9"
RECEIPT_SCHEMA = "bernini-csv-art-objective-receipt-v9"
EXPECTED_PHASES = 21
FORWARD_BRANCH_ORDER = (
    "frozen_noop_source_only_carrier",
    "frozen_negative_full_pair",
    "frozen_noop_full_pair",
    "frozen_action_full_pair",
    "adapted_noop_full_pair",
    "adapted_action_full_pair",
)
GRAPH_BRANCHES = (
    "adapted_noop_full_pair",
    "adapted_action_full_pair",
)
FORBIDDEN_INFERENCE_CONDITIONS = (
    "paired_target_video",
    "mask",
    "track",
    "swept_tube",
    "pose",
    "trajectory",
    "optical_flow",
    "first_frame_anchor",
)


class SourceKVRouteObjectiveError(RuntimeError):
    """Raised before an invalid V9 objective can update an adapter."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceKVRouteObjectiveError(
            f"objective contract is not canonical JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class SourceKVRouteLossConfig:
    pair_low_weight: float = 1.0
    pair_high_weight: float = 0.10
    executed_field_weight: float = 0.25
    prior_direction_weight: float = 0.25
    log_amplitude_weight: float = 0.10
    noop_preservation_weight: float = 0.50
    raw_phase0_weight: float = 0.10
    charbonnier_epsilon: float = 1.0e-3
    normalization_floor: float = 1.0e-4
    amplitude_floor: float = 1.0e-5
    prior_alignment_floor: float = 1.0e-6
    sigma_floor: float = 0.1176510528

    def validate(self) -> None:
        for name in (
            "pair_low_weight",
            "pair_high_weight",
            "executed_field_weight",
            "prior_direction_weight",
            "log_amplitude_weight",
            "noop_preservation_weight",
            "raw_phase0_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise SourceKVRouteObjectiveError(
                    f"{name} must be finite and nonnegative"
                )
        if (
            float(self.pair_low_weight) <= 0.0
            or float(self.noop_preservation_weight) <= 0.0
            or float(self.raw_phase0_weight) <= 0.0
        ):
            raise SourceKVRouteObjectiveError(
                "low-band, no-op, and raw-phase0 weights must be positive"
            )
        for name in (
            "charbonnier_epsilon",
            "normalization_floor",
            "amplitude_floor",
            "prior_alignment_floor",
            "sigma_floor",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise SourceKVRouteObjectiveError(
                    f"{name} must be finite and strictly positive"
                )


@dataclass(frozen=True)
class RouteCleanFields:
    """Post-guidance float32 fields on the same source-endpoint query.

    Only ``adapted_noop`` and ``adapted_action`` may retain autograd.  The
    source and executable target are labels in phase-grid form ``[B,21,S,D]``.
    """

    frozen_noop: Any
    frozen_action: Any
    adapted_noop: Any
    adapted_action: Any
    source_clean: Any
    target_clean: Any


@dataclass(frozen=True)
class SourceKVRouteDiagnostics:
    frozen_quotient: Any
    adapted_quotient: Any
    target_quotient: Any
    frozen_low_increments: Any
    predicted_low_increments: Any
    target_low_increments: Any
    frozen_high_increments: Any
    predicted_high_increments: Any
    target_high_increments: Any
    frozen_target_alignment: Any
    prior_direction_weight: Any
    predicted_execution: Any
    target_execution: Any
    raw_phase0: Any
    target_energy_retention: Any
    target_clipped_fraction: Any
    sigma: Any
    inverse_sigma_weight: Any


@dataclass(frozen=True)
class SourceKVRouteLossResult:
    total: Any
    pair_low: Any
    pair_high: Any
    executed_field: Any
    prior_direction: Any
    log_amplitude: Any
    noop_preservation: Any
    raw_phase0: Any
    diagnostics: SourceKVRouteDiagnostics


def objective_contract(
    config: SourceKVRouteLossConfig = SourceKVRouteLossConfig(),
) -> dict[str, Any]:
    config.validate()
    value: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "frames": 81,
        "latent_phases": EXPECTED_PHASES,
        "forward_branch_order": list(FORWARD_BRANCH_ORDER),
        "graph_branches": list(GRAPH_BRANCHES),
        "carrier_role": "frozen_source_only_post_rope_kv_attention_memory",
        "carrier_is_decoded_output": False,
        "query_endpoint": "source_beta_zero",
        "paired_target_role": "offline_training_label_only",
        "inference_conditions": ["source_video", "action_instruction"],
        "forbidden_inference_conditions": list(FORBIDDEN_INFERENCE_CONDITIONS),
        "quotient": "Q0(adapted_action-adapted_noop)",
        "teacher": "Q0(executable_target-source)",
        "hard_radius_projection": False,
        "target_clipping_fraction": 0.0,
        "target_energy_retention": 1.0,
        "temporal_split": "FIR3_low_plus_exact_residual_high",
        "high_band_normalization": "max(detached_target_rms,detached_frozen_rms)",
        "phase_zero_policy": (
            "penalize_raw_action_noop_phase0_without_external_frame_anchor"
        ),
        "sigma_weighting": "L_inner/max(sigma,sigma_floor)",
        "loss_config": asdict(config),
    }
    value["contract_digest"] = _object_sha256(value)
    return value


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise SourceKVRouteObjectiveError(
            "V9 objective tensor operations require PyTorch"
        ) from error
    return torch


def _validate_fields(fields: RouteCleanFields) -> None:
    torch = _require_torch()
    names = tuple(RouteCleanFields.__dataclass_fields__)
    tensors = tuple(getattr(fields, name) for name in names)
    reference = tensors[0]
    if (
        not isinstance(reference, torch.Tensor)
        or reference.ndim != 4
        or int(reference.shape[0]) <= 0
        or int(reference.shape[1]) != EXPECTED_PHASES
        or int(reference.shape[2]) <= 0
        or int(reference.shape[3]) <= 0
        or reference.dtype != torch.float32
    ):
        raise SourceKVRouteObjectiveError(
            "clean fields must be float32 [B,21,S,D] tensors"
        )
    for name, tensor in zip(names, tensors):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != tuple(reference.shape)
            or tensor.dtype != reference.dtype
            or tensor.device != reference.device
            or not bool(torch.isfinite(tensor).all())
        ):
            raise SourceKVRouteObjectiveError(
                f"{name} shape, dtype, device, or finiteness differs"
            )
    graph_names = {"adapted_noop", "adapted_action"}
    for name in names:
        requires_grad = bool(getattr(fields, name).requires_grad)
        if (name in graph_names) is not requires_grad:
            expectation = "retain" if name in graph_names else "exclude"
            raise SourceKVRouteObjectiveError(
                f"{name} must {expectation} the adapter graph"
            )


def _phase_rms(value: Any) -> Any:
    return value.square().mean(dim=(2, 3)).sqrt()


def _stable_phase_rms(value: Any, *, floor: float) -> Any:
    """RMS with a flat finite subgradient inside the numerical floor."""

    return value.square().mean(dim=(2, 3)).clamp_min(float(floor) ** 2).sqrt()


def _sample_rms(value: Any) -> Any:
    return value.square().mean(dim=(1, 2, 3)).sqrt()


def _causal_gauge(value: Any) -> Any:
    gauged = value - value[:, :1]
    # Subtracting a tensor from itself is exact for finite IEEE values.  Keep
    # the explicit assignment so later compiler reassociation cannot weaken
    # the phase-zero contract.
    gauged = gauged.clone()
    gauged[:, :1] = 0.0
    return gauged


def _phase_increments(value: Any) -> Any:
    torch = _require_torch()
    return torch.cat((torch.zeros_like(value[:, :1]), value[:, 1:] - value[:, :-1]), dim=1)


def _smooth_low(increments: Any) -> Any:
    torch = _require_torch()
    active = increments[:, 1:]
    previous = torch.cat((active[:, :1], active[:, :-1]), dim=1)
    following = torch.cat((active[:, 1:], active[:, -1:]), dim=1)
    low = 0.25 * previous + 0.5 * active + 0.25 * following
    return torch.cat((torch.zeros_like(increments[:, :1]), low), dim=1)


def _charbonnier_mean(error: Any, scale: Any, *, epsilon: float) -> Any:
    normalized = error / scale[..., None, None]
    return ((normalized.square() + float(epsilon) ** 2).sqrt() - float(epsilon)).mean()


def _cosine_per_phase(left: Any, right: Any, *, epsilon: float) -> Any:
    numerator = (left * right).sum(dim=(2, 3))
    left_squared_norm = left.square().sum(dim=(2, 3))
    right_squared_norm = right.square().sum(dim=(2, 3))
    # Clamping the squared product before sqrt avoids the undefined sqrt(0)
    # backward path.  It also retains the useful numerator gradient when the
    # predicted vector is exactly zero and the reference is nonzero.
    denominator = (
        left_squared_norm * right_squared_norm
    ).clamp_min(float(epsilon) ** 2).sqrt()
    return numerator / denominator


def _validated_sigma(sigma: Any, *, reference: Any) -> Any:
    """Return a detached scalar schedule sigma on ``reference``'s device."""

    torch = _require_torch()
    if isinstance(sigma, bool):
        raise SourceKVRouteObjectiveError("sigma must be a finite nonnegative scalar")
    if isinstance(sigma, torch.Tensor):
        if sigma.numel() != 1 or sigma.requires_grad:
            raise SourceKVRouteObjectiveError(
                "sigma must be one detached schedule scalar"
            )
        value = sigma.detach().to(device=reference.device, dtype=reference.dtype)
    else:
        try:
            scalar = float(sigma)
        except (TypeError, ValueError) as error:
            raise SourceKVRouteObjectiveError(
                "sigma must be a finite nonnegative scalar"
            ) from error
        value = torch.tensor(scalar, device=reference.device, dtype=reference.dtype)
    value = value.reshape(())
    if not bool(torch.isfinite(value)) or bool(value < 0.0):
        raise SourceKVRouteObjectiveError(
            "sigma must be a finite nonnegative scalar"
        )
    return value


def compute_source_kv_route_objective(
    fields: RouteCleanFields,
    *,
    sigma: Any,
    config: SourceKVRouteLossConfig = SourceKVRouteLossConfig(),
) -> SourceKVRouteLossResult:
    """Compute the unclipped V9 motion-route loss at one schedule sigma."""

    torch = _require_torch()
    config.validate()
    _validate_fields(fields)
    sigma_value = _validated_sigma(sigma, reference=fields.adapted_action)

    raw_frozen = fields.frozen_action - fields.frozen_noop
    raw_adapted = fields.adapted_action - fields.adapted_noop
    raw_target = fields.target_clean - fields.source_clean
    frozen_q = _causal_gauge(raw_frozen).detach()
    adapted_q = _causal_gauge(raw_adapted)
    target_q = _causal_gauge(raw_target).detach()

    frozen_z = _phase_increments(frozen_q)
    predicted_z = _phase_increments(adapted_q)
    target_z = _phase_increments(target_q)
    frozen_low = _smooth_low(frozen_z).detach()
    predicted_low = _smooth_low(predicted_z)
    target_low = _smooth_low(target_z).detach()
    frozen_high = (frozen_z - frozen_low).detach()
    predicted_high = predicted_z - predicted_low
    target_high = (target_z - target_low).detach()

    low_scale = torch.maximum(
        _phase_rms(target_low), _phase_rms(frozen_low)
    ).detach().clamp_min(float(config.normalization_floor))
    high_scale = torch.maximum(
        _phase_rms(target_high), _phase_rms(frozen_high)
    ).detach().clamp_min(float(config.normalization_floor))
    pair_low = _charbonnier_mean(
        predicted_low - target_low,
        low_scale,
        epsilon=config.charbonnier_epsilon,
    )
    pair_high = _charbonnier_mean(
        predicted_high - target_high,
        high_scale,
        epsilon=config.charbonnier_epsilon,
    )

    predicted_execution = fields.source_clean + adapted_q
    target_execution = (fields.source_clean + target_q).detach()
    execution_scale = torch.maximum(
        _phase_rms(target_q), _phase_rms(frozen_q)
    ).detach().clamp_min(float(config.normalization_floor))
    executed_field = _charbonnier_mean(
        predicted_execution - target_execution,
        execution_scale,
        epsilon=config.charbonnier_epsilon,
    )

    frozen_target_alignment = _cosine_per_phase(
        frozen_low, target_low, epsilon=config.prior_alignment_floor
    ).detach()
    prior_weight = frozen_target_alignment.clamp(min=0.0, max=1.0)
    predicted_prior_cosine = _cosine_per_phase(
        predicted_low, frozen_low, epsilon=config.prior_alignment_floor
    )
    active_weight = prior_weight[:, 1:]
    prior_direction = (
        (1.0 - predicted_prior_cosine[:, 1:]).clamp(min=0.0)
        * active_weight
    ).sum() / active_weight.sum().clamp_min(float(config.prior_alignment_floor))

    predicted_amplitude = _stable_phase_rms(
        predicted_low[:, 1:], floor=config.amplitude_floor
    )
    target_amplitude = _stable_phase_rms(
        target_low[:, 1:], floor=config.amplitude_floor
    ).detach()
    amplitude_error = (
        torch.log1p(predicted_amplitude / float(config.amplitude_floor))
        - torch.log1p(target_amplitude / float(config.amplitude_floor))
    )
    log_amplitude = (
        amplitude_error.square() + float(config.charbonnier_epsilon) ** 2
    ).sqrt().mean() - float(config.charbonnier_epsilon)

    noop_scale = _sample_rms(fields.frozen_noop).detach().clamp_min(
        float(config.normalization_floor)
    )
    noop_error = (
        fields.adapted_noop - fields.frozen_noop
    ) / noop_scale[:, None, None, None]
    noop_preservation = (
        noop_error.square() + float(config.charbonnier_epsilon) ** 2
    ).sqrt().mean() - float(config.charbonnier_epsilon)

    raw_phase0_tensor = raw_adapted[:, :1]
    phase0_scale = _sample_rms(target_q).detach().clamp_min(
        float(config.normalization_floor)
    )
    normalized_phase0 = raw_phase0_tensor / phase0_scale[:, None, None, None]
    raw_phase0_loss = (
        normalized_phase0.square() + float(config.charbonnier_epsilon) ** 2
    ).sqrt().mean() - float(config.charbonnier_epsilon)

    inner = (
        float(config.pair_low_weight) * pair_low
        + float(config.pair_high_weight) * pair_high
        + float(config.executed_field_weight) * executed_field
        + float(config.prior_direction_weight) * prior_direction
        + float(config.log_amplitude_weight) * log_amplitude
        + float(config.noop_preservation_weight) * noop_preservation
        + float(config.raw_phase0_weight) * raw_phase0_loss
    )
    sigma_denominator = sigma_value.clamp_min(float(config.sigma_floor))
    inverse_sigma_weight = sigma_denominator.reciprocal()
    total = inner * inverse_sigma_weight
    if not bool(torch.isfinite(total)) or not total.requires_grad:
        raise SourceKVRouteObjectiveError(
            "V9 objective must be finite and retain the adapted graph"
        )

    one = torch.ones((), dtype=total.dtype, device=total.device)
    zero = torch.zeros((), dtype=total.dtype, device=total.device)
    diagnostics = SourceKVRouteDiagnostics(
        frozen_quotient=frozen_q,
        adapted_quotient=adapted_q,
        target_quotient=target_q,
        frozen_low_increments=frozen_low,
        predicted_low_increments=predicted_low,
        target_low_increments=target_low,
        frozen_high_increments=frozen_high,
        predicted_high_increments=predicted_high,
        target_high_increments=target_high,
        frozen_target_alignment=frozen_target_alignment,
        prior_direction_weight=prior_weight,
        predicted_execution=predicted_execution,
        target_execution=target_execution,
        raw_phase0=raw_phase0_tensor,
        target_energy_retention=one,
        target_clipped_fraction=zero,
        sigma=sigma_value,
        inverse_sigma_weight=inverse_sigma_weight,
    )
    return SourceKVRouteLossResult(
        total=total,
        pair_low=pair_low,
        pair_high=pair_high,
        executed_field=executed_field,
        prior_direction=prior_direction,
        log_amplitude=log_amplitude,
        noop_preservation=noop_preservation,
        raw_phase0=raw_phase0_loss,
        diagnostics=diagnostics,
    )


def detached_objective_metrics(result: SourceKVRouteLossResult) -> dict[str, float]:
    """Return scalar receipt fields without retaining either adapted graph."""

    torch = _require_torch()

    def scalar(value: Any) -> float:
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise SourceKVRouteObjectiveError("receipt metric must be scalar")
        number = float(value.detach().float().cpu().item())
        if not math.isfinite(number):
            raise SourceKVRouteObjectiveError("receipt metric is non-finite")
        return number

    diagnostics = result.diagnostics
    return {
        "loss_total": scalar(result.total),
        "loss_pair_low": scalar(result.pair_low),
        "loss_pair_high": scalar(result.pair_high),
        "loss_executed_field": scalar(result.executed_field),
        "loss_prior_direction": scalar(result.prior_direction),
        "loss_log_amplitude": scalar(result.log_amplitude),
        "loss_noop_preservation": scalar(result.noop_preservation),
        "loss_raw_phase0": scalar(result.raw_phase0),
        "target_energy_retention": scalar(diagnostics.target_energy_retention),
        "target_clipped_fraction": scalar(diagnostics.target_clipped_fraction),
        "sigma": scalar(diagnostics.sigma),
        "inverse_sigma_weight": scalar(diagnostics.inverse_sigma_weight),
        "raw_phase0_rms": scalar(_sample_rms(diagnostics.raw_phase0).mean()),
        "frozen_target_alignment_mean_active": scalar(
            diagnostics.frozen_target_alignment[:, 1:].mean()
        ),
    }


__all__ = [
    "EXPECTED_PHASES",
    "FORBIDDEN_INFERENCE_CONDITIONS",
    "FORWARD_BRANCH_ORDER",
    "GRAPH_BRANCHES",
    "METHOD_NAME",
    "RECEIPT_SCHEMA",
    "RouteCleanFields",
    "SourceKVRouteDiagnostics",
    "SourceKVRouteLossConfig",
    "SourceKVRouteLossResult",
    "SourceKVRouteObjectiveError",
    "compute_source_kv_route_objective",
    "detached_objective_metrics",
    "objective_contract",
]
