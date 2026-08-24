#!/usr/bin/env python3
"""Temporary, rollback-only PAIR-v7 stateless Phase-B mutation boundary.

This module does not train Bernini and does not publish an adapter.  It is the
small mutation primitive needed by a future exact81 engineering canary:

1. validate a sealed, geometry-GO Phase-A receipt and an FP32 candidate delta;
2. require WORLD8 consensus on the exact pre-state;
3. apply the candidate directly to the closed Action-LoRA-B tensors;
4. audit the realized ``theta_after - theta_before`` displacement;
5. run one caller-supplied, sealed shadow evaluation; and
6. restore every Action-LoRA tensor byte-exactly in ``finally``.

The authority created here is deliberately unable to retain, save, or publish
the temporary state.  A decoded candidate latent may be returned by the caller
only after rollback; it is evidence for a canary, not a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Optional, Protocol

import torch

import pair_v7_dual_coordinate_nullspace_transport as transport_core


AUTHORITY_SCHEMA = "bernini-pair-v7-temporary-phase-b-canary-authority-v1"
EXECUTION_SCHEMA = "bernini-pair-v7-temporary-phase-b-execution-v1"
PHASE_A_SCHEMA = "bernini-pair-v7-phase-a-geometry-audit-v3"
UNION_SCHEMA = "bernini-pair-v7-phase-a-dp2-union-projection-v2"
TRANSPORT_SCHEMA = "bernini-pair-v7-nullspace-transport-receipt-v2"
WORLD_SIZE = 8
MAXIMUM_AUTHORITY_DELTA_NORM = 0.25
MAXIMUM_EFFECTIVE_DELTA_W_RELATIVE_NORM = 0.01


class PairV7PhaseBError(RuntimeError):
    """A temporary Phase-B add, audit, evaluation, or rollback is ambiguous."""


class WorldDigestConsensus(Protocol):
    def __call__(self, label: str, digest: str) -> None: ...


class ShadowEvaluation(Protocol):
    def __call__(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class TemporaryCanaryAuthority:
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class TemporaryCanaryResult:
    evaluation: Mapping[str, Any]
    realized_displacement_receipt: Mapping[str, Any]
    execution_receipt: Mapping[str, Any]


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise PairV7PhaseBError("receipt is already sealed")
    value = dict(unsigned)
    return {
        **value,
        "receipt_digest": transport_core.object_sha256(value),
    }


def _validate_seal(
    value: Any, *, schema: str, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != schema:
        raise PairV7PhaseBError(f"{label} schema differs")
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(digest, str)
        or transport_core.object_sha256(unsigned) != digest
    ):
        raise PairV7PhaseBError(f"{label} seal differs")
    return value


def _finite_positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairV7PhaseBError(f"{label} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PairV7PhaseBError(f"{label} must be positive")
    return result


def build_temporary_canary_authority(
    *,
    phase_a_receipt: Mapping[str, Any],
    expected_phase_a_receipt_digest: str,
    maximum_delta_norm: float,
    maximum_effective_delta_w_relative_norm: float = (
        MAXIMUM_EFFECTIVE_DELTA_W_RELATIVE_NORM
    ),
) -> TemporaryCanaryAuthority:
    """Create a narrow in-memory token from a sealed single-cell geometry GO.

    This is an engineering-canary authority only.  It cannot authorize a
    retained parameter state or a scientific action-editing success claim.
    """

    receipt = _validate_seal(
        phase_a_receipt, schema=PHASE_A_SCHEMA, label="Phase-A receipt"
    )
    if receipt.get("receipt_digest") != expected_phase_a_receipt_digest:
        raise PairV7PhaseBError("Phase-A receipt digest differs")
    union = _validate_seal(
        receipt.get("union_projection_receipt"),
        schema=UNION_SCHEMA,
        label="Phase-A union receipt",
    )
    transport = _validate_seal(
        receipt.get("nullspace_transport_receipt"),
        schema=TRANSPORT_SCHEMA,
        label="Phase-A transport receipt",
    )
    false_claims = (
        "optimizer_authorized",
        "parameter_update_authorized",
        "parameter_mutation_performed",
        "scientific_action_editing_success_claim",
        "global_population_go",
    )
    if (
        receipt.get("audit_complete") is not True
        or receipt.get("geometry_audit_passed") is not True
        or union.get("geometry_audit_passed") is not True
        or union.get("transport_geometry_authorized") is not True
        or transport.get("geometry_authorized") is not True
        or union.get("failure_codes") != []
        or transport.get("failure_codes") != []
        or any(receipt.get(field) is not False for field in false_claims)
    ):
        raise PairV7PhaseBError("Phase-A receipt is not a closed geometry GO")
    radius = _finite_positive(maximum_delta_norm, label="maximum delta norm")
    relative = _finite_positive(
        maximum_effective_delta_w_relative_norm,
        label="maximum effective delta-W relative norm",
    )
    if (
        radius > MAXIMUM_AUTHORITY_DELTA_NORM
        or relative > MAXIMUM_EFFECTIVE_DELTA_W_RELATIVE_NORM
    ):
        raise PairV7PhaseBError("temporary canary trust bound is too large")
    unsigned = {
        "schema_version": AUTHORITY_SCHEMA,
        "authority_scope": "one_temporary_direct_add_then_unconditional_rollback",
        "world_size": WORLD_SIZE,
        "phase_a_receipt_digest": receipt["receipt_digest"],
        "phase_a_union_receipt_digest": union["receipt_digest"],
        "phase_a_transport_receipt_digest": transport["receipt_digest"],
        "phase_a_parameter_state_sha256": receipt["parameter_state_sha256"],
        "maximum_delta_norm": radius,
        "maximum_effective_delta_w_relative_norm": relative,
        "temporary_direct_add_count": 1,
        "temporary_direct_add_authorized": True,
        "unconditional_finally_rollback_required": True,
        "world_pre_add_and_post_rollback_digest_consensus_required": True,
        "optimizer_authorized": False,
        "retained_parameter_update_authorized": False,
        "adapter_checkpoint_publication_authorized": False,
        "scientific_action_editing_success_claim": False,
    }
    return TemporaryCanaryAuthority(receipt=_seal(unsigned))


def _snapshot(
    mapping: Mapping[str, torch.Tensor], *, cpu: bool = False
) -> dict[str, torch.Tensor]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise PairV7PhaseBError("Action-LoRA state mapping is empty")
    result: dict[str, torch.Tensor] = {}
    identities: set[int] = set()
    for name, value in mapping.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, torch.Tensor)
            or value.device.type == "meta"
            or not bool(torch.isfinite(value.detach()).all().item())
            or id(value) in identities
        ):
            raise PairV7PhaseBError("Action-LoRA state mapping differs")
        identities.add(id(value))
        detached = value.detach()
        if cpu:
            detached = detached.to(device="cpu")
        result[name] = detached.clone()
    return result


def _state_digest(mapping: Mapping[str, torch.Tensor]) -> str:
    try:
        return transport_core.named_parameter_state_sha256(mapping)
    except Exception as error:
        raise PairV7PhaseBError("cannot digest Action-LoRA state") from error


def _generic_state_digest(mapping: Mapping[str, torch.Tensor]) -> str:
    """Hash frozen tensors without requiring the Action-LoRA FP32 dtype."""

    rows = []
    for name in sorted(mapping):
        value = mapping[name]
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise PairV7PhaseBError("cannot digest frozen tensor state")
        rows.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "tensor_sha256": transport_core._tensor_sha256(value),
            }
        )
    return transport_core.object_sha256(rows)


def _a_name_for_b(name: str) -> str:
    marker = "action_lora_b.weight"
    if marker not in name:
        raise PairV7PhaseBError("candidate layout is not Action-LoRA-B-only")
    return name.replace(marker, "action_lora_a.weight")


def _effective_delta_w_audit(
    *,
    candidate: transport_core.StatelessTrustRegionDelta,
    full_state: Mapping[str, torch.Tensor],
    base_projection_by_b_name: Mapping[str, torch.Tensor],
    maximum_relative_norm: float,
) -> list[Mapping[str, Any]]:
    names = tuple(candidate.layout.names)
    if set(base_projection_by_b_name) != set(names):
        raise PairV7PhaseBError("base projection trust mapping closure differs")
    rows: list[Mapping[str, Any]] = []
    for name in names:
        a_name = _a_name_for_b(name)
        if a_name not in full_state:
            raise PairV7PhaseBError(f"frozen LoRA-A is absent for {name}")
        delta_b = candidate.delta_by_parameter[name].detach().float()
        a = full_state[a_name].detach().float()
        base = base_projection_by_b_name[name].detach().float()
        if (
            delta_b.ndim != 2
            or a.ndim != 2
            or base.ndim != 2
            or int(delta_b.shape[1]) != int(a.shape[0])
            or tuple(base.shape)
            != (int(delta_b.shape[0]), int(a.shape[1]))
            or not bool(torch.isfinite(base).all().item())
        ):
            raise PairV7PhaseBError(f"effective delta-W geometry differs: {name}")
        delta_w_norm = float(torch.linalg.vector_norm(delta_b @ a).item())
        base_norm = float(torch.linalg.vector_norm(base).item())
        relative = delta_w_norm / base_norm if base_norm > 0.0 else math.inf
        if not math.isfinite(relative) or relative > maximum_relative_norm:
            raise PairV7PhaseBError(
                f"effective delta-W trust bound exceeded: {name}"
            )
        rows.append(
            {
                "b_parameter_name": name,
                "a_parameter_name": a_name,
                "effective_delta_w_norm": delta_w_norm,
                "base_projection_norm": base_norm,
                "relative_norm": relative,
                "maximum_relative_norm": maximum_relative_norm,
            }
        )
    return rows


def _validate_evaluation(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairV7PhaseBError("shadow evaluation must return a mapping")
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    if (
        value.get("temporary_canary_evaluation_complete") is not True
        or not isinstance(value.get("canary_gate_passed"), bool)
        or not isinstance(digest, str)
        or transport_core.object_sha256(unsigned) != digest
    ):
        raise PairV7PhaseBError("shadow evaluation receipt differs")
    return value


def run_temporary_stateless_canary(
    *,
    authority: TemporaryCanaryAuthority,
    transport: transport_core.TransportResult,
    candidate: transport_core.StatelessTrustRegionDelta,
    full_action_lora_state: Mapping[str, torch.Tensor],
    live_b_parameters: Mapping[str, torch.Tensor],
    base_projection_by_b_name: Mapping[str, torch.Tensor],
    world_digest_consensus: WorldDigestConsensus,
    evaluate_candidate: ShadowEvaluation,
    _fault_after_add_index: Optional[int] = None,
) -> TemporaryCanaryResult:
    """Temporarily apply one candidate and restore in ``finally``.

    ``_fault_after_add_index`` is a private adversarial-test hook.  Production
    callers must leave it ``None``.
    """

    auth = _validate_seal(
        authority.receipt,
        schema=AUTHORITY_SCHEMA,
        label="temporary canary authority",
    )
    if not callable(world_digest_consensus) or not callable(evaluate_candidate):
        raise PairV7PhaseBError("canary callbacks differ")
    if (
        candidate.receipt.get("transport_receipt_digest")
        != auth.get("phase_a_transport_receipt_digest")
        or candidate.receipt.get("pre_step_parameter_state_sha256")
        != auth.get("phase_a_parameter_state_sha256")
        or float(candidate.receipt.get("actual_fp32_delta_norm", math.inf))
        > float(auth["maximum_delta_norm"]) * (1.0 + 1.0e-7)
        or candidate.receipt.get("runtime_apply_authorized") is not False
        or candidate.receipt.get("optimizer_step_allowed") is not False
    ):
        raise PairV7PhaseBError("candidate is outside temporary authority")
    if (
        transport.layout.device.type != "cpu"
        or candidate.layout.device.type != "cpu"
        or transport.layout.layout_digest != candidate.layout.layout_digest
    ):
        raise PairV7PhaseBError(
            "temporary canary requires the authenticated rank-zero CPU candidate"
        )
    names = tuple(candidate.layout.names)
    if set(live_b_parameters) != set(names):
        raise PairV7PhaseBError("live Action-LoRA-B closure differs")
    # The geometry solve and sealed candidate are intentionally owned by the
    # rank-zero CPU path.  Keep every numerical audit on that same device;
    # only the direct add below is staged to the live accelerator tensors.
    # This also makes a CPU-origin candidate valid for GPU/MI210 execution
    # without silently changing its sealed FP32 bytes.
    before = _snapshot(full_action_lora_state, cpu=True)
    before_digest = _state_digest(before)
    if before_digest != auth["phase_a_parameter_state_sha256"]:
        raise PairV7PhaseBError("live pre-state differs from Phase-A state")
    for name in names:
        live = live_b_parameters[name]
        if (
            full_action_lora_state.get(name) is not live
            or live.dtype != torch.float32
            or not live.is_contiguous()
            or tuple(live.shape) != tuple(candidate.delta_by_parameter[name].shape)
        ):
            raise PairV7PhaseBError(f"live candidate tensor differs: {name}")
    base_before = _snapshot(base_projection_by_b_name, cpu=True)
    base_before_digest = _generic_state_digest(base_before)
    effective_rows = _effective_delta_w_audit(
        candidate=candidate,
        full_state=before,
        base_projection_by_b_name=base_before,
        maximum_relative_norm=float(
            auth["maximum_effective_delta_w_relative_norm"]
        ),
    )
    world_digest_consensus("pair-v7 Phase-B pre-add state", before_digest)

    realized: Optional[transport_core.RealizedDisplacementAudit] = None
    evaluation: Optional[Mapping[str, Any]] = None
    rollback_digest: Optional[str] = None
    try:
        with torch.no_grad():
            for index, name in enumerate(names):
                delta = candidate.delta_by_parameter[name]
                if delta.dtype not in (torch.float32, torch.float64):
                    raise PairV7PhaseBError("candidate delta dtype differs")
                live_b_parameters[name].add_(
                    delta.to(
                        device=live_b_parameters[name].device,
                        dtype=torch.float32,
                    )
                )
                if _fault_after_add_index == index:
                    raise PairV7PhaseBError("injected mid-add failure")
        after = _snapshot(full_action_lora_state, cpu=True)
        realized = transport_core.audit_realized_parameter_displacement(
            transport=transport,
            candidate=candidate,
            full_parameter_state_before=before,
            full_parameter_state_after=after,
        )
        if not realized.realized_displacement_safe:
            raise PairV7PhaseBError("realized parameter displacement is unsafe")
        world_digest_consensus(
            "pair-v7 Phase-B realized state", _state_digest(after)
        )
        evaluation = _validate_evaluation(evaluate_candidate())
    finally:
        with torch.no_grad():
            for name, snapshot in before.items():
                full_action_lora_state[name].copy_(
                    snapshot.to(
                        device=full_action_lora_state[name].device,
                        dtype=full_action_lora_state[name].dtype,
                    )
                )
        rollback = _snapshot(full_action_lora_state, cpu=True)
        rollback_digest = _state_digest(rollback)
        if rollback_digest != before_digest or any(
            not torch.equal(rollback[name], before[name]) for name in before
        ):
            raise PairV7PhaseBError("Action-LoRA rollback was not byte-exact")
        base_after = _snapshot(base_projection_by_b_name, cpu=True)
        if (
            _generic_state_digest(base_after) != base_before_digest
            or any(
                not torch.equal(base_after[name], base_before[name])
                for name in base_before
            )
        ):
            raise PairV7PhaseBError("frozen base projection changed during canary")
        world_digest_consensus(
            "pair-v7 Phase-B post-rollback state", rollback_digest
        )

    assert realized is not None and evaluation is not None and rollback_digest is not None
    unsigned = {
        "schema_version": EXECUTION_SCHEMA,
        "authority_receipt_digest": auth["receipt_digest"],
        "transport_receipt_digest": transport.receipt["receipt_digest"],
        "candidate_delta_receipt_digest": candidate.receipt["receipt_digest"],
        "realized_displacement_receipt_digest": realized.receipt["receipt_digest"],
        "evaluation_receipt_digest": evaluation["receipt_digest"],
        "pre_add_parameter_state_sha256": before_digest,
        "post_rollback_parameter_state_sha256": rollback_digest,
        "byte_exact_rollback_verified": True,
        "temporary_direct_add_count": 1,
        "optimizer_constructed": False,
        "optimizer_step_called": False,
        "effective_delta_w_trust_audit": effective_rows,
        "canary_gate_passed": evaluation["canary_gate_passed"],
        "adapter_checkpoint_written": False,
        "retained_parameter_update": False,
        "scientific_action_editing_success_claim": False,
    }
    return TemporaryCanaryResult(
        evaluation=evaluation,
        realized_displacement_receipt=realized.receipt,
        execution_receipt=_seal(unsigned),
    )


__all__ = [
    "AUTHORITY_SCHEMA",
    "EXECUTION_SCHEMA",
    "PairV7PhaseBError",
    "TemporaryCanaryAuthority",
    "TemporaryCanaryResult",
    "build_temporary_canary_authority",
    "run_temporary_stateless_canary",
]
