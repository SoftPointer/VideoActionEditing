#!/usr/bin/env python3
"""Detached inverse-recoverability gate and inverse FM authorization.

Inverse recoverability is a ranking/evaluation axis, not a differentiable
shortcut into the forward rollout.  Only detached codec-roundtripped rollout
measurements are accepted.  Even a strong recoverability score cannot
authorize inverse flow matching until the *forward midpoint* independently
passes an absolute terminal-event gate.  The authorized inverse model call is
bound to the re-encoded midpoint as its unique visual source and to the real
source as the clean rectified-flow endpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

import torch


SCHEMA_VERSION = "bernini-saic-inverse-recoverability-authorization-v1"
CODEC_RECEIPT_SCHEMA_VERSION = "bernini-saic-codec-reencoded-endpoint-v1"
INVERSE_FM_SCHEMA_VERSION = "bernini-saic-authorized-inverse-flow-matching-v1"
LATENT_CHANNELS = 16
LATENT_PHASES = 21
EXACT40_UPDATE_INDICES = (4, 12, 20, 28, 33, 34, 35, 37)
EXACT40_FORBIDDEN_UPDATE_INDICES = (38, 39)

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_CODEC_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "input_output_media_sha256",
        "decoded_rgb24_sha256",
        "codec_name",
        "codec_bitstream_sha256",
        "codec_decoded_rgb24_sha256",
        "vae_id",
        "vae_weights_sha256",
        "reencoded_latent_sha256",
        "frame_count",
        "fps_numerator",
        "fps_denominator",
        "endpoint_detached",
        "receipt_digest",
    }
)
_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "midpoint_codec_receipt_digest",
        "midpoint_reencoded_latent_sha256",
        "inverse_conditioning_source_sha256",
        "inverse_conditioning_uses_midpoint_as_unique_visual_source",
        "pure_t2v_visual_condition_used",
        "terminal_event_verified",
        "forward_event_score",
        "absolute_event_floor",
        "absolute_event_pass",
        "chosen_recoverability_score",
        "baseline_recoverability_score",
        "recoverability_floor",
        "minimum_recoverability_gain",
        "recoverability_floor_pass",
        "recoverability_rank_pass",
        "inverse_flow_matching_authorized",
        "zero_update_reason",
        "authorization_digest",
    }
)


class SAICInverseRecoverabilityError(ValueError):
    """An inverse evaluation or authorization invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SAICInverseRecoverabilityError(
            "value is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SAICInverseRecoverabilityError(f"{label} must be a mapping")
    actual = set(value)
    if not all(type(key) is str for key in actual):
        raise SAICInverseRecoverabilityError(f"{label} keys must be strings")
    if actual != fields:
        raise SAICInverseRecoverabilityError(
            f"{label} schema differs; missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return value


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise SAICInverseRecoverabilityError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise SAICInverseRecoverabilityError(f"{label} must be an identifier")
    return value


def _unit_real(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SAICInverseRecoverabilityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SAICInverseRecoverabilityError(f"{label} must lie in [0,1]")
    return result


def _detached_scalar(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.float32
        or value.numel() != 1
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SAICInverseRecoverabilityError(
            f"{label} must be one detached finite FP32 scalar"
        )
    result = float(value.item())
    if nonnegative and result < 0.0:
        raise SAICInverseRecoverabilityError(f"{label} must be nonnegative")
    return result


def recoverability_score(reconstruction_error: torch.Tensor) -> torch.Tensor:
    """Map a detached nonnegative inverse error to a detached score in (0,1]."""

    error = _detached_scalar(
        reconstruction_error, label="reconstruction_error", nonnegative=True
    )
    result = torch.tensor(
        1.0 / (1.0 + error),
        dtype=torch.float32,
        device=reconstruction_error.device,
    )
    if result.requires_grad or result.grad_fn is not None:
        raise SAICInverseRecoverabilityError("recoverability score acquired a graph")
    return result


def _validate_codec_receipt(value: Any) -> Mapping[str, Any]:
    row = _closed(value, _CODEC_FIELDS, label="midpoint codec receipt")
    if row["schema_version"] != CODEC_RECEIPT_SCHEMA_VERSION:
        raise SAICInverseRecoverabilityError("midpoint codec receipt schema differs")
    body = {key: item for key, item in row.items() if key != "receipt_digest"}
    if _sha(row["receipt_digest"], label="codec receipt digest") != object_sha256(body):
        raise SAICInverseRecoverabilityError("midpoint codec receipt digest differs")
    _safe_id(row["candidate_id"], label="candidate_id")
    for key in (
        "input_output_media_sha256",
        "decoded_rgb24_sha256",
        "codec_bitstream_sha256",
        "codec_decoded_rgb24_sha256",
        "vae_weights_sha256",
        "reencoded_latent_sha256",
    ):
        _sha(row[key], label=key)
    _safe_id(row["codec_name"], label="codec_name")
    _safe_id(row["vae_id"], label="vae_id")
    if (
        type(row["frame_count"]) is not int
        or row["frame_count"] != 81
        or type(row["fps_numerator"]) is not int
        or row["fps_numerator"] != 25
        or type(row["fps_denominator"]) is not int
        or row["fps_denominator"] != 1
        or row["endpoint_detached"] is not True
    ):
        raise SAICInverseRecoverabilityError(
            "midpoint must be a detached exact81 codec roundtrip"
        )
    return row


@dataclass(frozen=True)
class InverseAuthorization:
    authorized: bool
    zero_update_reason: str | None
    receipt: Mapping[str, Any]


def authorize_inverse_flow_matching(
    *,
    midpoint_codec_receipt: Mapping[str, Any],
    inverse_conditioning_source_sha256: str,
    terminal_event_verified: bool,
    forward_event_score: torch.Tensor,
    absolute_event_floor: float,
    chosen_reconstruction_error: torch.Tensor,
    baseline_reconstruction_error: torch.Tensor,
    recoverability_floor: float,
    minimum_recoverability_gain: float,
) -> InverseAuthorization:
    """Authorize inverse FM only after absolute action and detached rank gates."""

    codec = _validate_codec_receipt(midpoint_codec_receipt)
    condition_sha = _sha(
        inverse_conditioning_source_sha256,
        label="inverse_conditioning_source_sha256",
    )
    if condition_sha != codec["reencoded_latent_sha256"]:
        raise SAICInverseRecoverabilityError(
            "inverse visual source is not the codec-reencoded midpoint"
        )
    if type(terminal_event_verified) is not bool:
        raise SAICInverseRecoverabilityError("terminal_event_verified must be boolean")
    event_score = _detached_scalar(forward_event_score, label="forward_event_score")
    if not 0.0 <= event_score <= 1.0:
        raise SAICInverseRecoverabilityError("forward_event_score must lie in [0,1]")
    event_floor = _unit_real(absolute_event_floor, label="absolute_event_floor")
    inverse_floor = _unit_real(recoverability_floor, label="recoverability_floor")
    inverse_gain = _unit_real(
        minimum_recoverability_gain, label="minimum_recoverability_gain"
    )
    chosen_score = float(recoverability_score(chosen_reconstruction_error).item())
    baseline_score = float(recoverability_score(baseline_reconstruction_error).item())
    event_pass = terminal_event_verified and event_score >= event_floor
    floor_pass = chosen_score >= inverse_floor
    rank_pass = chosen_score - baseline_score >= inverse_gain
    authorized = event_pass and floor_pass and rank_pass
    if not event_pass:
        reason = "absolute_forward_event_not_verified"
    elif not floor_pass:
        reason = "inverse_recoverability_floor_failed"
    elif not rank_pass:
        reason = "inverse_recoverability_rank_failed"
    else:
        reason = None
    body = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": codec["candidate_id"],
        "midpoint_codec_receipt_digest": codec["receipt_digest"],
        "midpoint_reencoded_latent_sha256": codec["reencoded_latent_sha256"],
        "inverse_conditioning_source_sha256": condition_sha,
        "inverse_conditioning_uses_midpoint_as_unique_visual_source": True,
        "pure_t2v_visual_condition_used": False,
        "terminal_event_verified": terminal_event_verified,
        "forward_event_score": event_score,
        "absolute_event_floor": event_floor,
        "absolute_event_pass": event_pass,
        "chosen_recoverability_score": chosen_score,
        "baseline_recoverability_score": baseline_score,
        "recoverability_floor": inverse_floor,
        "minimum_recoverability_gain": inverse_gain,
        "recoverability_floor_pass": floor_pass,
        "recoverability_rank_pass": rank_pass,
        "inverse_flow_matching_authorized": authorized,
        "zero_update_reason": reason,
    }
    receipt = {**body, "authorization_digest": object_sha256(body)}
    return InverseAuthorization(
        authorized=authorized, zero_update_reason=reason, receipt=receipt
    )


def validate_authorization(value: Any) -> Mapping[str, Any]:
    row = _closed(value, _AUTHORIZATION_FIELDS, label="inverse authorization")
    if row["schema_version"] != SCHEMA_VERSION:
        raise SAICInverseRecoverabilityError("authorization schema differs")
    body = {key: item for key, item in row.items() if key != "authorization_digest"}
    if _sha(row["authorization_digest"], label="authorization_digest") != object_sha256(body):
        raise SAICInverseRecoverabilityError("authorization digest differs")
    _safe_id(row["candidate_id"], label="authorization candidate_id")
    for key in (
        "midpoint_codec_receipt_digest",
        "midpoint_reencoded_latent_sha256",
        "inverse_conditioning_source_sha256",
    ):
        _sha(row[key], label=f"authorization {key}")
    for key in (
        "inverse_conditioning_uses_midpoint_as_unique_visual_source",
        "pure_t2v_visual_condition_used",
        "terminal_event_verified",
        "absolute_event_pass",
        "recoverability_floor_pass",
        "recoverability_rank_pass",
        "inverse_flow_matching_authorized",
    ):
        if type(row[key]) is not bool:
            raise SAICInverseRecoverabilityError(
                f"authorization {key} must be boolean"
            )
    event_score = _unit_real(row["forward_event_score"], label="forward_event_score")
    event_floor = _unit_real(row["absolute_event_floor"], label="absolute_event_floor")
    chosen_score = _unit_real(
        row["chosen_recoverability_score"], label="chosen_recoverability_score"
    )
    baseline_score = _unit_real(
        row["baseline_recoverability_score"], label="baseline_recoverability_score"
    )
    recovery_floor = _unit_real(
        row["recoverability_floor"], label="recoverability_floor"
    )
    recovery_gain = _unit_real(
        row["minimum_recoverability_gain"],
        label="minimum_recoverability_gain",
    )
    expected_event_pass = row["terminal_event_verified"] and event_score >= event_floor
    expected_floor_pass = chosen_score >= recovery_floor
    expected_rank_pass = chosen_score - baseline_score >= recovery_gain
    expected_authorized = (
        expected_event_pass and expected_floor_pass and expected_rank_pass
    )
    expected_reason = (
        "absolute_forward_event_not_verified"
        if not expected_event_pass
        else "inverse_recoverability_floor_failed"
        if not expected_floor_pass
        else "inverse_recoverability_rank_failed"
        if not expected_rank_pass
        else None
    )
    if (
        row["absolute_event_pass"] is not expected_event_pass
        or row["recoverability_floor_pass"] is not expected_floor_pass
        or row["recoverability_rank_pass"] is not expected_rank_pass
        or row["inverse_flow_matching_authorized"] is not expected_authorized
        or row["zero_update_reason"] != expected_reason
    ):
        raise SAICInverseRecoverabilityError("authorization gates are inconsistent")
    if row["midpoint_reencoded_latent_sha256"] != row["inverse_conditioning_source_sha256"]:
        raise SAICInverseRecoverabilityError("authorized midpoint binding differs")
    if (
        row["inverse_conditioning_uses_midpoint_as_unique_visual_source"] is not True
        or row["pure_t2v_visual_condition_used"] is not False
    ):
        raise SAICInverseRecoverabilityError("authorization visual route differs")
    if not expected_authorized:
        raise SAICInverseRecoverabilityError("inverse flow matching is not authorized")
    return row


def _exact_latent(
    value: Any, *, label: str, trainable: bool
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.float32
        or value.ndim != 5
        or int(value.shape[0]) < 1
        or int(value.shape[1]) != LATENT_CHANNELS
        or int(value.shape[2]) != LATENT_PHASES
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise SAICInverseRecoverabilityError(
            f"{label} must be finite FP32 exact81 [B,16,21,H,W]"
        )
    if trainable:
        if not value.requires_grad:
            raise SAICInverseRecoverabilityError(
                f"{label} must carry an inverse-branch output cotangent"
            )
    elif value.requires_grad or value.grad_fn is not None:
        raise SAICInverseRecoverabilityError(f"{label} must be detached")
    return value


def _validate_update_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SAICInverseRecoverabilityError("exact40_index must be an integer")
    if value in EXACT40_FORBIDDEN_UPDATE_INDICES:
        raise SAICInverseRecoverabilityError("exact40 indices 38/39 are forbidden")
    if value not in EXACT40_UPDATE_INDICES:
        raise SAICInverseRecoverabilityError("exact40_index is outside Stage-B J")
    return value


@dataclass(frozen=True)
class InverseFlowMatchingResult:
    state: torch.Tensor
    velocity_target: torch.Tensor
    per_sample_loss: torch.Tensor
    loss: torch.Tensor
    exact40_index: int
    authorization_digest: str


def authorized_inverse_flow_matching(
    source_clean: torch.Tensor,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    inverse_prediction: torch.Tensor,
    *,
    exact40_index: int,
    midpoint_condition_sha256: str,
    authorization_receipt: Mapping[str, Any],
) -> InverseFlowMatchingResult:
    """Flow-match the real source endpoint after all inverse gates pass."""

    authorization = validate_authorization(authorization_receipt)
    step_index = _validate_update_index(exact40_index)
    condition_sha = _sha(midpoint_condition_sha256, label="midpoint_condition_sha256")
    if condition_sha != authorization["inverse_conditioning_source_sha256"]:
        raise SAICInverseRecoverabilityError(
            "inverse prediction is not bound to the authorized midpoint"
        )
    source = _exact_latent(source_clean, label="source_clean", trainable=False)
    noise = _exact_latent(epsilon, label="epsilon", trainable=False)
    prediction = _exact_latent(
        inverse_prediction, label="inverse_prediction", trainable=True
    )
    if source.shape != noise.shape or source.shape != prediction.shape:
        raise SAICInverseRecoverabilityError("inverse FM geometry differs")
    if source.device != noise.device or source.device != prediction.device:
        raise SAICInverseRecoverabilityError("inverse FM devices differ")
    sigma_value = _detached_scalar(sigma, label="sigma")
    if not 0.0 < sigma_value < 1.0:
        raise SAICInverseRecoverabilityError("sigma must lie strictly in (0,1)")
    state = ((1.0 - sigma_value) * source + sigma_value * noise).detach()
    target = (noise - source).detach()
    per_sample = (prediction - target).square().flatten(start_dim=1).mean(dim=1)
    loss = per_sample.mean()
    if (
        state.dtype != torch.float32
        or target.dtype != torch.float32
        or per_sample.dtype != torch.float32
        or loss.dtype != torch.float32
        or not bool(torch.isfinite(loss.detach()).item())
        or not loss.requires_grad
        or loss.grad_fn is None
    ):
        raise SAICInverseRecoverabilityError(
            "inverse FM result must be finite graph-connected FP32"
        )
    return InverseFlowMatchingResult(
        state=state,
        velocity_target=target,
        per_sample_loss=per_sample,
        loss=loss,
        exact40_index=step_index,
        authorization_digest=authorization["authorization_digest"],
    )


__all__ = [
    "CODEC_RECEIPT_SCHEMA_VERSION",
    "EXACT40_FORBIDDEN_UPDATE_INDICES",
    "EXACT40_UPDATE_INDICES",
    "INVERSE_FM_SCHEMA_VERSION",
    "InverseAuthorization",
    "InverseFlowMatchingResult",
    "SAICInverseRecoverabilityError",
    "SCHEMA_VERSION",
    "authorize_inverse_flow_matching",
    "authorized_inverse_flow_matching",
    "canonical_json_bytes",
    "object_sha256",
    "recoverability_score",
    "validate_authorization",
]
