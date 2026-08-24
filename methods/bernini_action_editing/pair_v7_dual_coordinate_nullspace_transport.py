#!/usr/bin/env python3
"""PAIR-v7 dual-coordinate identity-nullspace action transport.

This module is deliberately a mathematical core, not a trainer.  A qualified
pure-T2V action arm supplies an Action-LoRA gradient while exact81 native RV2V
feature-sketch VJPs supply identity tangent rows.  The only coupling between
the two coordinates is their shared, explicitly closed LoRA parameter layout.

The constrained object is the *actual proposed parameter displacement*.  The
module therefore emits a stateless trust-region delta which must be applied
directly.  Passing the projected gradient through Adam, momentum, weight
decay, or any other stateful/preconditioned optimizer invalidates the proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import torch


METHOD_NAME = "bernini-pair-v7-dual-coordinate-identity-nullspace-action-transport"
TRANSPORT_SCHEMA = "bernini-pair-v7-nullspace-transport-receipt-v2"
DELTA_SCHEMA = "bernini-pair-v7-stateless-trust-region-delta-v1"
ROLLBACK_SCHEMA = "bernini-pair-v7-native-post-step-decision-untrusted-v2"
REALIZED_DELTA_SCHEMA = "bernini-pair-v7-realized-parameter-displacement-audit-v1"
CONTRACT_SCHEMA = "bernini-pair-v7-information-flow-contract-v2"

REQUIRED_IDENTITY_FAMILIES = (
    "deploy_noop_identity",
    "deploy_camera_delta",
)
EXACT81_FRAME_COUNT = 81
EXACT40_STEP_COUNT = 40
EXACT40_ZERO_UPDATE_INDICES = (38, 39)

_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,255}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NO_UPDATE_CLAIMS = {
    "global_population_go": False,
    "optimizer_authorized": False,
    "parameter_update_authorized": False,
    "action_success_claimed": False,
}


class PairV7TransportError(RuntimeError):
    """A malformed or non-finite input cannot be interpreted safely."""


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
        raise PairV7TransportError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV7TransportError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairV7TransportError(f"{label} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise PairV7TransportError(f"{label} must be finite")
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
        }
    )
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


def named_parameter_state_sha256(
    parameters: Mapping[str, torch.Tensor],
) -> str:
    """Hash an exact FP32 parameter state, including its closed key/shape set."""

    if not isinstance(parameters, Mapping) or not parameters:
        raise PairV7TransportError("parameter state mapping must be non-empty")
    raw_names = tuple(parameters)
    if any(
        not isinstance(name, str) or _SAFE_NAME_RE.fullmatch(name) is None
        for name in raw_names
    ):
        raise PairV7TransportError("parameter state name is unsafe")
    rows: list[dict[str, Any]] = []
    for name in sorted(raw_names):
        tensor = parameters[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.device.type == "meta"
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise PairV7TransportError(
                f"parameter state {name} must be materialized finite FP32"
            )
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "torch.float32",
                "tensor_sha256": _tensor_sha256(tensor),
            }
        )
    return object_sha256(rows)


@dataclass(frozen=True)
class GradientLayout:
    """Closed, deterministic layout shared by both coordinate arms."""

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    offsets: tuple[tuple[int, int], ...]
    device: torch.device
    total_numel: int
    layout_digest: str

    @classmethod
    def from_named_gradients(
        cls, gradients: Mapping[str, torch.Tensor]
    ) -> "GradientLayout":
        if not isinstance(gradients, Mapping) or not gradients:
            raise PairV7TransportError("gradient mapping must be non-empty")
        raw_names = tuple(gradients)
        if any(
            not isinstance(name, str) or _SAFE_NAME_RE.fullmatch(name) is None
            for name in raw_names
        ):
            raise PairV7TransportError("gradient parameter name is unsafe")
        names = tuple(sorted(raw_names))
        shapes: list[tuple[int, ...]] = []
        offsets: list[tuple[int, int]] = []
        first_device: torch.device | None = None
        cursor = 0
        for name in names:
            tensor = gradients[name]
            if not isinstance(tensor, torch.Tensor):
                raise PairV7TransportError(f"gradient {name} is not a tensor")
            if tensor.dtype != torch.float32:
                raise PairV7TransportError(f"gradient {name} must be exact FP32")
            if tensor.requires_grad:
                raise PairV7TransportError(f"gradient {name} must be detached")
            if not bool(torch.isfinite(tensor).all().item()):
                raise PairV7TransportError(f"gradient {name} is non-finite")
            if first_device is None:
                first_device = tensor.device
            elif tensor.device != first_device:
                raise PairV7TransportError("all gradients must share one device")
            shape = tuple(int(item) for item in tensor.shape)
            count = int(tensor.numel())
            if count == 0:
                raise PairV7TransportError(f"gradient {name} must be non-empty")
            shapes.append(shape)
            offsets.append((cursor, cursor + count))
            cursor += count
        assert first_device is not None
        manifest = [
            {"name": name, "shape": list(shape), "dtype": "torch.float32"}
            for name, shape in zip(names, shapes)
        ]
        return cls(
            names=names,
            shapes=tuple(shapes),
            offsets=tuple(offsets),
            device=first_device,
            total_numel=cursor,
            layout_digest=object_sha256(manifest),
        )

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "shape": list(shape), "dtype": "torch.float32"}
            for name, shape in zip(self.names, self.shapes)
        ]

    def flatten(
        self, gradients: Mapping[str, torch.Tensor], *, label: str
    ) -> torch.Tensor:
        if not isinstance(gradients, Mapping) or set(gradients) != set(self.names):
            actual = set(gradients) if isinstance(gradients, Mapping) else set()
            raise PairV7TransportError(
                f"{label} layout closure differs: missing={sorted(set(self.names)-actual)} "
                f"extra={sorted(actual-set(self.names))}"
            )
        result = torch.empty(
            self.total_numel, dtype=torch.float64, device=self.device
        )
        for name, shape, (start, stop) in zip(
            self.names, self.shapes, self.offsets
        ):
            tensor = gradients[name]
            if not isinstance(tensor, torch.Tensor):
                raise PairV7TransportError(f"{label} {name} is not a tensor")
            if (
                tensor.dtype != torch.float32
                or tensor.device != self.device
                or tuple(tensor.shape) != shape
                or tensor.requires_grad
            ):
                raise PairV7TransportError(f"{label} {name} layout/dtype/device differs")
            if not bool(torch.isfinite(tensor).all().item()):
                raise PairV7TransportError(f"{label} {name} is non-finite")
            result[start:stop].copy_(tensor.reshape(-1).to(dtype=torch.float64))
        return result

    def unflatten(self, flat: torch.Tensor, *, label: str) -> dict[str, torch.Tensor]:
        if (
            not isinstance(flat, torch.Tensor)
            or flat.dtype != torch.float64
            or flat.device != self.device
            or flat.ndim != 1
            or flat.numel() != self.total_numel
            or not bool(torch.isfinite(flat).all().item())
        ):
            raise PairV7TransportError(f"{label} flat vector differs")
        return {
            name: flat[start:stop].reshape(shape).to(dtype=torch.float32).clone()
            for name, shape, (start, stop) in zip(
                self.names, self.shapes, self.offsets
            )
        }


@dataclass(frozen=True)
class IdentityGradientProbe:
    """One non-zero native-coordinate feature-sketch Jacobian/VJP row.

    This is intentionally not a gradient of ``||feature-current-target||^2``:
    that gradient can vanish exactly at the identity-preserving operating
    point and therefore supplies no tangent constraint.
    """

    probe_id: str
    family: str
    gradient_by_parameter: Mapping[str, torch.Tensor]
    feature_sketch_sha256: str
    source_coordinate_receipt_digest: str
    gradient_computation_receipt_digest: str
    checkpoint_content_receipt_digest: str
    parameter_state_sha256: str

    def validate_metadata(self) -> None:
        if not isinstance(self.probe_id, str) or _SAFE_NAME_RE.fullmatch(self.probe_id) is None:
            raise PairV7TransportError("identity probe ID is unsafe")
        if self.family not in REQUIRED_IDENTITY_FAMILIES:
            raise PairV7TransportError(f"identity probe {self.probe_id} family differs")
        _sha256(self.feature_sketch_sha256, label=f"{self.probe_id} feature sketch")
        _sha256(
            self.source_coordinate_receipt_digest,
            label=f"{self.probe_id} source-coordinate receipt",
        )
        _sha256(
            self.gradient_computation_receipt_digest,
            label=f"{self.probe_id} gradient-computation receipt",
        )
        _sha256(
            self.checkpoint_content_receipt_digest,
            label=f"{self.probe_id} checkpoint-content receipt",
        )
        _sha256(
            self.parameter_state_sha256,
            label=f"{self.probe_id} parameter state",
        )


@dataclass(frozen=True)
class ActionGradientProvenance:
    """Declared provenance for an already-computed action gradient.

    The math core validates closure and binds these declarations into its
    receipt.  It does not execute Bernini and therefore cannot independently
    prove that a caller actually computed the gradient from these events.
    """

    candidate_ids: tuple[str, ...]
    action_families: tuple[str, ...]
    event_digests: tuple[str, ...]
    component_gradient_sha256: tuple[str, ...]
    gradient_computation_receipt_digests: tuple[str, ...]
    fit_only_geometry_authority_digest: str
    aggregation: str

    def validate(self) -> None:
        count = len(self.candidate_ids)
        if (
            count == 0
            or len(self.action_families) != count
            or len(self.event_digests) != count
            or len(self.component_gradient_sha256) != count
            or len(self.gradient_computation_receipt_digests) != count
            or len(set(self.candidate_ids)) != count
            or len(set(self.action_families)) != count
        ):
            raise PairV7TransportError("action-gradient provenance closure differs")
        for label, values in (
            ("candidate ID", self.candidate_ids),
            ("action family", self.action_families),
        ):
            if any(
                not isinstance(value, str) or _SAFE_NAME_RE.fullmatch(value) is None
                for value in values
            ):
                raise PairV7TransportError(f"action-gradient {label} differs")
        for label, values in (
            ("event digest", self.event_digests),
            ("component gradient", self.component_gradient_sha256),
            ("gradient computation receipt", self.gradient_computation_receipt_digests),
        ):
            for ordinal, value in enumerate(values):
                _sha256(value, label=f"action-gradient {label}[{ordinal}]")
        _sha256(
            self.fit_only_geometry_authority_digest,
            label="fit-only geometry authority",
        )
        allowed = {
            "single_fit_only_geometry_event",
            "arithmetic_mean_dp2_after_sp4_fit_only_geometry_gradients",
        }
        if self.aggregation not in allowed:
            raise PairV7TransportError("action-gradient aggregation differs")
        if count == 1 and self.aggregation != "single_fit_only_geometry_event":
            raise PairV7TransportError("single action event aggregation differs")
        if (
            count == 2
            and self.aggregation
            != "arithmetic_mean_dp2_after_sp4_fit_only_geometry_gradients"
        ):
            raise PairV7TransportError("DP2 action event aggregation differs")
        if count not in {1, 2}:
            raise PairV7TransportError("only single-event or exact DP2 action provenance is supported")


@dataclass(frozen=True)
class TransportConfig:
    minimum_action_norm: float = 1.0e-12
    minimum_identity_probe_norm: float = 1.0e-12
    singular_value_relative_tolerance: float = 1.0e-10
    eigenvalue_absolute_tolerance: float = 1.0e-12
    negative_eigenvalue_roundoff_tolerance: float = 1.0e-10
    maximum_effective_condition_number: float = 1.0e5
    minimum_action_gradient_survival: float = 5.0e-2
    minimum_action_descent_cosine: float = 5.0e-2
    minimum_action_descent_gain: float = 1.0e-16
    identity_dot_absolute_tolerance: float = 1.0e-9
    identity_dot_relative_tolerance: float = 2.0e-5
    maximum_identity_cosine: float = 2.0e-5
    maximum_probe_reconstruction_residual: float = 1.0e-6
    maximum_projection_invariant_error: float = 5.0e-5
    fp32_projection_refinement_passes: int = 2

    def validate(self) -> None:
        positive = (
            "minimum_action_norm",
            "minimum_identity_probe_norm",
            "singular_value_relative_tolerance",
            "eigenvalue_absolute_tolerance",
            "negative_eigenvalue_roundoff_tolerance",
            "maximum_effective_condition_number",
            "minimum_action_gradient_survival",
            "minimum_action_descent_cosine",
            "minimum_action_descent_gain",
            "identity_dot_absolute_tolerance",
            "identity_dot_relative_tolerance",
            "maximum_identity_cosine",
            "maximum_probe_reconstruction_residual",
            "maximum_projection_invariant_error",
        )
        for name in positive:
            value = _finite_scalar(getattr(self, name), label=name)
            if value <= 0.0:
                raise PairV7TransportError(f"{name} must be positive")
        if self.minimum_action_gradient_survival > 1.0:
            raise PairV7TransportError("minimum survival must not exceed one")
        if self.minimum_action_descent_cosine > 1.0 or self.maximum_identity_cosine > 1.0:
            raise PairV7TransportError("cosine thresholds must not exceed one")
        if type(self.fp32_projection_refinement_passes) is not int or not 0 <= self.fp32_projection_refinement_passes <= 8:
            raise PairV7TransportError("FP32 refinement passes must be 0..8")
        # Configuration may be made stricter for an experiment, but not looser
        # than the provisional v1 policy without a new schema/preregistration.
        if self.maximum_effective_condition_number > 1.0e5:
            raise PairV7TransportError("condition-number policy may not be loosened")
        if self.minimum_action_gradient_survival < 5.0e-2:
            raise PairV7TransportError("action-survival policy may not be loosened")
        if self.minimum_action_descent_cosine < 5.0e-2:
            raise PairV7TransportError("action-descent policy may not be loosened")
        if self.minimum_action_descent_gain < 1.0e-16:
            raise PairV7TransportError("action-descent-gain policy may not be loosened")
        if self.identity_dot_relative_tolerance > 2.0e-5:
            raise PairV7TransportError("identity-dot policy may not be loosened")
        if self.maximum_identity_cosine > 2.0e-5:
            raise PairV7TransportError("identity-cosine policy may not be loosened")
        if self.maximum_probe_reconstruction_residual > 1.0e-6:
            raise PairV7TransportError("probe-reconstruction policy may not be loosened")
        if self.maximum_projection_invariant_error > 5.0e-5:
            raise PairV7TransportError("projection-invariant policy may not be loosened")


@dataclass(frozen=True)
class TransportResult:
    layout: GradientLayout
    checkpoint_content_receipt_digest: str
    parameter_state_sha256: str
    action_gradient_provenance: ActionGradientProvenance
    action_gradient_by_parameter: Mapping[str, torch.Tensor]
    identity_probes: tuple[IdentityGradientProbe, ...]
    safe_gradient_by_parameter: Mapping[str, torch.Tensor]
    geometry_authorized: bool
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class StatelessTrustRegionDelta:
    layout: GradientLayout
    delta_by_parameter: Mapping[str, torch.Tensor]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class RealizedDisplacementAudit:
    layout: GradientLayout
    realized_delta_by_parameter: Mapping[str, torch.Tensor]
    realized_displacement_safe: bool
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class RollbackConfig:
    armijo_c1: float = 1.0e-4
    maximum_identity_absolute_regression: float = 1.0e-4
    maximum_identity_relative_regression: float = 1.0e-2

    def validate(self) -> None:
        c1 = _finite_scalar(self.armijo_c1, label="armijo_c1")
        absolute = _finite_scalar(
            self.maximum_identity_absolute_regression,
            label="maximum identity absolute regression",
        )
        relative = _finite_scalar(
            self.maximum_identity_relative_regression,
            label="maximum identity relative regression",
        )
        if not 0.0 < c1 < 1.0 or absolute < 0.0 or relative < 0.0:
            raise PairV7TransportError("rollback thresholds differ")


def _compact_gram(unit_vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    count = len(unit_vectors)
    gram = torch.empty((count, count), dtype=torch.float64, device="cpu")
    for left in range(count):
        for right in range(left, count):
            value = float(torch.dot(unit_vectors[left], unit_vectors[right]).item())
            gram[left, right] = value
            gram[right, left] = value
    return 0.5 * (gram + gram.transpose(0, 1))


def _rowspace_projection(
    vector: torch.Tensor,
    unit_vectors: Sequence[torch.Tensor],
    retained_eigenvalues: torch.Tensor,
    retained_eigenvectors: torch.Tensor,
) -> torch.Tensor:
    if not unit_vectors or retained_eigenvalues.numel() == 0:
        return torch.zeros_like(vector)
    correlations = torch.tensor(
        [float(torch.dot(unit, vector).item()) for unit in unit_vectors],
        dtype=torch.float64,
        device="cpu",
    )
    spectral = retained_eigenvectors.transpose(0, 1) @ correlations
    coefficients = retained_eigenvectors @ (spectral / retained_eigenvalues)
    result = torch.zeros_like(vector)
    for coefficient, unit in zip(coefficients.tolist(), unit_vectors):
        result.add_(unit, alpha=float(coefficient))
    return result


def _seal(unsigned: Mapping[str, Any], *, field: str = "receipt_digest") -> dict[str, Any]:
    if field in unsigned:
        raise PairV7TransportError("cannot seal an already sealed receipt")
    value = dict(unsigned)
    for claim, expected in _NO_UPDATE_CLAIMS.items():
        if claim in value and value[claim] is not expected:
            raise PairV7TransportError(f"{claim} must remain false")
        value[claim] = expected
    value[field] = object_sha256(value)
    return value


def project_action_gradient_to_identity_nullspace(
    *,
    action_gradient_by_parameter: Mapping[str, torch.Tensor],
    action_gradient_provenance: ActionGradientProvenance,
    identity_probes: Sequence[IdentityGradientProbe],
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
    config: TransportConfig = TransportConfig(),
) -> TransportResult:
    """Return a fail-closed FP32 action direction in the identity nullspace.

    ``identity_probes`` must be source-native feature-sketch VJPs.  No RGB,
    latent, mask, flow, pose, trajectory, or donor object is accepted here.
    """

    config.validate()
    if not isinstance(action_gradient_provenance, ActionGradientProvenance):
        raise PairV7TransportError("action-gradient provenance type differs")
    action_gradient_provenance.validate()
    checkpoint_digest = _sha256(
        checkpoint_content_receipt_digest,
        label="checkpoint-content receipt",
    )
    state_digest = _sha256(parameter_state_sha256, label="parameter state")
    layout = GradientLayout.from_named_gradients(action_gradient_by_parameter)
    action = layout.flatten(action_gradient_by_parameter, label="action gradient")
    action_norm = float(torch.linalg.vector_norm(action).item())
    action_digest = _tensor_sha256(action.to(dtype=torch.float32))
    if (
        len(action_gradient_provenance.candidate_ids) == 1
        and action_gradient_provenance.component_gradient_sha256[0]
        != action_digest
    ):
        raise PairV7TransportError(
            "single-event component gradient digest differs from supplied action gradient"
        )
    failures: list[str] = []
    if action_norm <= config.minimum_action_norm:
        failures.append("ACTION_GRADIENT_ZERO_OR_TOO_SMALL")
    if not isinstance(identity_probes, Sequence) or not identity_probes:
        raise PairV7TransportError("identity probe sequence must be non-empty")
    if any(not isinstance(probe, IdentityGradientProbe) for probe in identity_probes):
        raise PairV7TransportError("identity probe sequence contains an invalid row")

    ordered = sorted(identity_probes, key=lambda row: (row.family, row.probe_id))
    if len({row.probe_id for row in ordered}) != len(ordered):
        raise PairV7TransportError("identity probe IDs repeat")
    family_counts = {family: 0 for family in REQUIRED_IDENTITY_FAMILIES}
    unit_vectors: list[torch.Tensor] = []
    live_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for probe in ordered:
        probe.validate_metadata()
        if (
            probe.checkpoint_content_receipt_digest != checkpoint_digest
            or probe.parameter_state_sha256 != state_digest
        ):
            raise PairV7TransportError(
                f"identity probe {probe.probe_id} checkpoint/parameter state differs from action gradient"
            )
        family_counts[probe.family] += 1
        flat = layout.flatten(
            probe.gradient_by_parameter, label=f"identity probe {probe.probe_id}"
        )
        norm = float(torch.linalg.vector_norm(flat).item())
        row = {
            "probe_id": probe.probe_id,
            "family": probe.family,
            "feature_sketch_sha256": probe.feature_sketch_sha256,
            "source_coordinate_receipt_digest": probe.source_coordinate_receipt_digest,
            "gradient_computation_receipt_digest": (
                probe.gradient_computation_receipt_digest
            ),
            "checkpoint_content_receipt_digest": probe.checkpoint_content_receipt_digest,
            "parameter_state_sha256": probe.parameter_state_sha256,
            "gradient_sha256": _tensor_sha256(flat.to(dtype=torch.float32)),
            "gradient_norm": norm,
            "nonzero": norm > config.minimum_identity_probe_norm,
        }
        all_rows.append(row)
        if norm <= config.minimum_identity_probe_norm:
            failures.append(f"IDENTITY_PROBE_ZERO_OR_TOO_SMALL:{probe.probe_id}")
            continue
        unit = flat / norm
        unit_vectors.append(unit)
        live_rows.append(row)
    for family, count in family_counts.items():
        if count == 0:
            failures.append(f"MISSING_IDENTITY_FAMILY:{family}")

    gram = _compact_gram(unit_vectors)
    if unit_vectors:
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        except RuntimeError as error:
            raise PairV7TransportError("identity compact Gram EVD failed") from error
        lambda_max = max(float(eigenvalues[-1].item()), 0.0)
        negative_floor = -max(
            config.eigenvalue_absolute_tolerance,
            config.negative_eigenvalue_roundoff_tolerance * max(lambda_max, 1.0),
        )
        if float(eigenvalues[0].item()) < negative_floor:
            failures.append("IDENTITY_GRAM_HAS_NON_ROUNDOFF_NEGATIVE_EIGENVALUE")
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        threshold = max(
            config.eigenvalue_absolute_tolerance,
            (config.singular_value_relative_tolerance**2) * lambda_max,
        )
        retained_mask = eigenvalues > threshold
        retained_values = eigenvalues[retained_mask]
        retained_vectors = eigenvectors[:, retained_mask]
    else:
        eigenvalues = torch.empty(0, dtype=torch.float64)
        retained_values = torch.empty(0, dtype=torch.float64)
        retained_vectors = torch.empty((0, 0), dtype=torch.float64)
        threshold = config.eigenvalue_absolute_tolerance

    rank = int(retained_values.numel())
    if rank == 0:
        failures.append("IDENTITY_BASIS_HAS_ZERO_EFFECTIVE_RANK")
        condition_number = None
    else:
        condition_number = math.sqrt(
            float(retained_values[-1].item() / retained_values[0].item())
        )
        if condition_number > config.maximum_effective_condition_number:
            failures.append("IDENTITY_BASIS_ILL_CONDITIONED")

    # Two removals improve numerical idempotence before the FP32 apply audit.
    removed_first = _rowspace_projection(
        action, unit_vectors, retained_values, retained_vectors
    )
    safe = action - removed_first
    safe = safe - _rowspace_projection(
        safe, unit_vectors, retained_values, retained_vectors
    )
    applied = safe.to(dtype=torch.float32).to(dtype=torch.float64)
    for _ in range(config.fp32_projection_refinement_passes):
        applied = applied - _rowspace_projection(
            applied, unit_vectors, retained_values, retained_vectors
        )
        applied = applied.to(dtype=torch.float32).to(dtype=torch.float64)

    removed = action - applied
    safe_norm = float(torch.linalg.vector_norm(applied).item())
    removed_norm = float(torch.linalg.vector_norm(removed).item())
    survival = safe_norm / action_norm if action_norm > 0.0 else 0.0
    action_dot = float(torch.dot(action, applied).item())
    descent_cosine = (
        action_dot / (action_norm * safe_norm)
        if action_norm > 0.0 and safe_norm > 0.0
        else None
    )
    if survival < config.minimum_action_gradient_survival:
        failures.append("ACTION_GRADIENT_SURVIVAL_TOO_LOW")
    if action_dot <= config.minimum_action_descent_gain:
        failures.append("ACTION_DESCENT_GAIN_NONPOSITIVE_OR_TOO_SMALL")
    if descent_cosine is None or descent_cosine < config.minimum_action_descent_cosine:
        failures.append("ACTION_DESCENT_COSINE_TOO_LOW")

    action_energy = action_norm * action_norm
    energy_error = abs(action_energy - safe_norm**2 - removed_norm**2) / max(
        action_energy, config.minimum_action_norm**2
    )
    orthogonality_error = abs(float(torch.dot(applied, removed).item())) / max(
        action_energy, config.minimum_action_norm**2
    )
    cosine_survival_error = (
        abs(descent_cosine - survival) if descent_cosine is not None else None
    )
    projected_twice = _rowspace_projection(
        removed_first, unit_vectors, retained_values, retained_vectors
    )
    idempotence_error = float(
        torch.linalg.vector_norm(projected_twice - removed_first).item()
    ) / max(action_norm, config.minimum_action_norm)
    invariant_values = [energy_error, orthogonality_error, idempotence_error]
    if cosine_survival_error is not None:
        invariant_values.append(cosine_survival_error)
    if max(invariant_values) > config.maximum_projection_invariant_error:
        failures.append("ORTHOGONAL_PROJECTION_INVARIANT_FAILED")

    # Reconstruction is audited from the compact Gram, avoiding a tall Q.
    reconstruction_by_live_id: dict[str, float] = {}
    if rank:
        for column, row in enumerate(live_rows):
            correlations = gram[:, column]
            spectral = retained_vectors.transpose(0, 1) @ correlations
            coefficients = retained_vectors @ (spectral / retained_values)
            captured = float(torch.dot(correlations, coefficients).item())
            residual = math.sqrt(max(0.0, 1.0 - min(captured, 1.0)))
            reconstruction_by_live_id[row["probe_id"]] = residual
            if residual > config.maximum_probe_reconstruction_residual:
                failures.append(
                    f"IDENTITY_PROBE_RECONSTRUCTION_RESIDUAL_HIGH:{row['probe_id']}"
                )

    post_by_id: dict[str, dict[str, Any]] = {}
    for row, unit in zip(live_rows, unit_vectors):
        signed_unit_dot = float(torch.dot(unit, applied).item())
        absolute_unit_dot = abs(signed_unit_dot)
        cosine = absolute_unit_dot / safe_norm if safe_norm > 0.0 else None
        dot_limit = (
            config.identity_dot_absolute_tolerance
            + config.identity_dot_relative_tolerance * safe_norm
        )
        if absolute_unit_dot > dot_limit:
            failures.append(f"IDENTITY_DOT_EXCEEDED:{row['probe_id']}")
        if cosine is None or cosine > config.maximum_identity_cosine:
            failures.append(f"IDENTITY_COSINE_EXCEEDED:{row['probe_id']}")
        post_by_id[row["probe_id"]] = {
            "signed_unit_dot_with_applied_gradient": signed_unit_dot,
            "absolute_unit_dot_with_applied_gradient": absolute_unit_dot,
            "relative_unit_dot_to_action_norm": (
                absolute_unit_dot / action_norm if action_norm > 0.0 else None
            ),
            "absolute_cosine_with_applied_gradient": cosine,
            "dot_limit": dot_limit,
            "rowspace_reconstruction_residual": reconstruction_by_live_id.get(
                row["probe_id"]
            ),
        }

    probe_receipts: list[dict[str, Any]] = []
    for row in all_rows:
        probe_receipts.append({**row, **post_by_id.get(row["probe_id"], {})})
    max_identity_cosine = max(
        (
            item["absolute_cosine_with_applied_gradient"]
            for item in probe_receipts
            if item.get("absolute_cosine_with_applied_gradient") is not None
        ),
        default=None,
    )
    max_identity_dot = max(
        (
            item["absolute_unit_dot_with_applied_gradient"]
            for item in probe_receipts
            if item.get("absolute_unit_dot_with_applied_gradient") is not None
        ),
        default=None,
    )
    failures = sorted(set(failures))
    geometry_authorized = not failures
    safe_named = layout.unflatten(applied, label="applied safe gradient")
    unsigned = {
        "schema_version": TRANSPORT_SCHEMA,
        "method_name": METHOD_NAME,
        "mathematical_transport_passed": geometry_authorized,
        "geometry_authorized": geometry_authorized,
        "optimizer_authorized": False,
        "runtime_apply_authorized": False,
        "failure_codes": failures,
        "algorithm": "fp64-normalized-compact-gram-evd-two-pass-rowspace-removal-fp32-reaudit",
        "constrained_object": "actual_stateless_lora_parameter_displacement",
        "stateful_optimizer_after_projection_allowed": False,
        "parameter_layout": layout.manifest(),
        "parameter_layout_digest": layout.layout_digest,
        "parameter_count": layout.total_numel,
        "checkpoint_content_receipt_digest": checkpoint_digest,
        "pre_step_parameter_state_sha256": state_digest,
        "action_gradient_sha256": action_digest,
        "action_gradient_provenance": {
            "candidate_ids": list(action_gradient_provenance.candidate_ids),
            "action_families": list(action_gradient_provenance.action_families),
            "event_digests": list(action_gradient_provenance.event_digests),
            "component_gradient_sha256": list(
                action_gradient_provenance.component_gradient_sha256
            ),
            "gradient_computation_receipt_digests": list(
                action_gradient_provenance.gradient_computation_receipt_digests
            ),
            "fit_only_geometry_authority_digest": (
                action_gradient_provenance.fit_only_geometry_authority_digest
            ),
            "aggregation": action_gradient_provenance.aggregation,
            "gradient_origin_independently_proven_by_math_core": False,
        },
        "applied_safe_gradient_sha256": _tensor_sha256(applied.to(dtype=torch.float32)),
        "action_gradient_norm": action_norm,
        "applied_safe_gradient_norm": safe_norm,
        "removed_gradient_norm": removed_norm,
        "action_gradient_norm_survival": survival,
        "action_gradient_energy_survival": survival * survival,
        "action_descent_gain": action_dot,
        "action_descent_cosine": descent_cosine,
        "identity_family_counts": family_counts,
        "identity_probe_count": len(ordered),
        "identity_nonzero_probe_count": len(unit_vectors),
        "identity_effective_rank": rank,
        "identity_redundant_direction_count": len(unit_vectors) - rank,
        "compact_gram_sha256": _tensor_sha256(gram),
        "compact_gram_eigenvalues": [float(item) for item in eigenvalues.tolist()],
        "eigenvalue_retention_threshold": threshold,
        "effective_condition_number": condition_number,
        "projection_energy_decomposition_error": energy_error,
        "projection_removed_safe_orthogonality_error": orthogonality_error,
        "projection_idempotence_error": idempotence_error,
        "projection_cosine_survival_error": cosine_survival_error,
        "maximum_identity_unit_dot": max_identity_dot,
        "maximum_identity_cosine": max_identity_cosine,
        "identity_probes": probe_receipts,
        "thresholds": {
            name: getattr(config, name)
            for name in config.__dataclass_fields__
        },
        "information_flow": {
            "coordinate_coupling": "shared_action_lora_parameter_space_only",
            "pure_t2v_rgb_or_latent_role": "action_arm_reward_trajectory_only",
            "pure_t2v_visual_used_as_rv2v_target_noise_source_or_donor": False,
            "mask_flow_pose_track_or_trajectory_input": False,
            "inference_requires_only_source_prompt_and_lora": True,
            "exact81": True,
            "exact40_zero_update_indices": list(EXACT40_ZERO_UPDATE_INDICES),
        },
        "sealed_fit_only_geometry_evidence_required": True,
        "global_population_authority_required": False,
        "actual_checkpoint_content_audit_required_before_apply": True,
        "native_post_step_gate_required": True,
    }
    receipt = _seal(unsigned)
    bound_probes = tuple(
        IdentityGradientProbe(
            probe_id=probe.probe_id,
            family=probe.family,
            gradient_by_parameter={
                name: probe.gradient_by_parameter[name].detach().clone()
                for name in layout.names
            },
            feature_sketch_sha256=probe.feature_sketch_sha256,
            source_coordinate_receipt_digest=probe.source_coordinate_receipt_digest,
            gradient_computation_receipt_digest=(
                probe.gradient_computation_receipt_digest
            ),
            checkpoint_content_receipt_digest=probe.checkpoint_content_receipt_digest,
            parameter_state_sha256=probe.parameter_state_sha256,
        )
        for probe in ordered
    )
    return TransportResult(
        layout=layout,
        checkpoint_content_receipt_digest=checkpoint_digest,
        parameter_state_sha256=state_digest,
        action_gradient_provenance=action_gradient_provenance,
        action_gradient_by_parameter={
            name: action_gradient_by_parameter[name].detach().clone()
            for name in layout.names
        },
        identity_probes=bound_probes,
        safe_gradient_by_parameter=safe_named,
        geometry_authorized=geometry_authorized,
        receipt=receipt,
    )


def _revalidate_transport_result(transport: TransportResult) -> TransportResult:
    if not isinstance(transport, TransportResult):
        raise PairV7TransportError("transport result type differs")
    receipt = transport.receipt
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != TRANSPORT_SCHEMA:
        raise PairV7TransportError("transport receipt schema differs")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if not isinstance(declared, str) or object_sha256(unsigned) != declared:
        raise PairV7TransportError("transport receipt digest differs")
    thresholds = receipt.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(
        TransportConfig.__dataclass_fields__
    ):
        raise PairV7TransportError("transport threshold closure differs")
    try:
        config = TransportConfig(**dict(thresholds))
    except TypeError as error:
        raise PairV7TransportError("transport thresholds cannot be reconstructed") from error
    recomputed = project_action_gradient_to_identity_nullspace(
        action_gradient_by_parameter=transport.action_gradient_by_parameter,
        action_gradient_provenance=transport.action_gradient_provenance,
        identity_probes=transport.identity_probes,
        checkpoint_content_receipt_digest=transport.checkpoint_content_receipt_digest,
        parameter_state_sha256=transport.parameter_state_sha256,
        config=config,
    )
    safe = transport.layout.flatten(
        transport.safe_gradient_by_parameter, label="bound safe gradient"
    )
    if (
        recomputed.receipt["receipt_digest"] != declared
        or transport.layout.layout_digest != recomputed.layout.layout_digest
        or transport.checkpoint_content_receipt_digest
        != receipt.get("checkpoint_content_receipt_digest")
        or transport.parameter_state_sha256
        != receipt.get("pre_step_parameter_state_sha256")
        or _tensor_sha256(safe.to(dtype=torch.float32))
        != receipt.get("applied_safe_gradient_sha256")
    ):
        raise PairV7TransportError("transport tensors no longer match their receipt")
    return recomputed


def build_stateless_trust_region_delta(
    *,
    transport: TransportResult,
    learning_rate: float,
    maximum_delta_norm: float,
    pre_step_parameter_state_sha256: str,
) -> StatelessTrustRegionDelta:
    """Turn an authorized geometry result into the exact delta to apply.

    The caller must use direct ``parameter.add_(delta)`` semantics.  Calling an
    optimizer on these tensors is outside this contract.
    """

    transport = _revalidate_transport_result(transport)
    if not transport.geometry_authorized:
        raise PairV7TransportError("NO-GO transport cannot produce a parameter delta")
    learning_rate = _finite_scalar(learning_rate, label="learning rate")
    maximum_delta_norm = _finite_scalar(
        maximum_delta_norm, label="maximum delta norm"
    )
    if learning_rate <= 0.0 or maximum_delta_norm <= 0.0:
        raise PairV7TransportError("trust-region controls must be positive")
    state_digest = _sha256(
        pre_step_parameter_state_sha256, label="pre-step parameter state"
    )
    if state_digest != transport.parameter_state_sha256:
        raise PairV7TransportError(
            "candidate delta pre-step state differs from gradient/probe state"
        )
    safe = transport.layout.flatten(
        transport.safe_gradient_by_parameter, label="safe gradient"
    )
    safe_norm = float(torch.linalg.vector_norm(safe).item())
    if safe_norm <= 0.0:
        raise PairV7TransportError("authorized safe gradient unexpectedly vanished")
    scale = min(learning_rate, maximum_delta_norm / safe_norm)
    delta = -scale * safe
    # Round to the FP32 values that will actually be applied and enforce the
    # trust radius on those values, not on an ideal FP64 precursor.
    delta = delta.to(dtype=torch.float32).to(dtype=torch.float64)
    for _ in range(3):
        norm = float(torch.linalg.vector_norm(delta).item())
        if norm <= maximum_delta_norm * (1.0 + 1.0e-7):
            break
        delta.mul_(maximum_delta_norm / norm)
        delta = delta.to(dtype=torch.float32).to(dtype=torch.float64)
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    if delta_norm > maximum_delta_norm * (1.0 + 1.0e-7):
        raise PairV7TransportError("FP32 delta escaped the trust radius")
    effective_scale = delta_norm / safe_norm
    named = transport.layout.unflatten(delta, label="stateless trust-region delta")
    action = transport.layout.flatten(
        transport.action_gradient_by_parameter, label="bound action gradient"
    )
    action_norm = float(torch.linalg.vector_norm(action).item())
    action_directional_derivative = float(torch.dot(action, delta).item())
    action_descent_cosine = (
        -action_directional_derivative / (action_norm * delta_norm)
        if action_norm > 0.0 and delta_norm > 0.0
        else None
    )
    thresholds = transport.receipt["thresholds"]
    if (
        action_directional_derivative >= -float(thresholds["minimum_action_descent_gain"])
        or action_descent_cosine is None
        or action_descent_cosine
        < float(thresholds["minimum_action_descent_cosine"])
    ):
        raise PairV7TransportError("actual FP32 trust-region delta lost action descent")
    identity_delta_rows: list[dict[str, Any]] = []
    for probe in transport.identity_probes:
        probe_flat = transport.layout.flatten(
            probe.gradient_by_parameter,
            label=f"bound identity probe {probe.probe_id}",
        )
        probe_norm = float(torch.linalg.vector_norm(probe_flat).item())
        unit_dot = float(torch.dot(probe_flat / probe_norm, delta).item())
        absolute_unit_dot = abs(unit_dot)
        cosine = absolute_unit_dot / delta_norm
        dot_limit = (
            effective_scale * float(thresholds["identity_dot_absolute_tolerance"])
            + float(thresholds["identity_dot_relative_tolerance"]) * delta_norm
        )
        if (
            absolute_unit_dot > dot_limit
            or cosine > float(thresholds["maximum_identity_cosine"])
        ):
            raise PairV7TransportError(
                f"actual FP32 trust-region delta left identity nullspace: {probe.probe_id}"
            )
        identity_delta_rows.append(
            {
                "probe_id": probe.probe_id,
                "family": probe.family,
                "signed_unit_dot": unit_dot,
                "absolute_unit_dot": absolute_unit_dot,
                "absolute_cosine": cosine,
                "dot_limit": dot_limit,
            }
        )
    unsigned = {
        "schema_version": DELTA_SCHEMA,
        "method_name": METHOD_NAME,
        "transport_receipt_digest": transport.receipt["receipt_digest"],
        "parameter_layout_digest": transport.layout.layout_digest,
        "checkpoint_content_receipt_digest": (
            transport.checkpoint_content_receipt_digest
        ),
        "pre_step_parameter_state_sha256": state_digest,
        "learning_rate": learning_rate,
        "maximum_delta_norm": maximum_delta_norm,
        "requested_scale": scale,
        "effective_fp32_scale": effective_scale,
        "actual_fp32_delta_norm": delta_norm,
        "actual_fp32_delta_sha256": _tensor_sha256(delta.to(dtype=torch.float32)),
        "action_directional_derivative": action_directional_derivative,
        "action_descent_cosine": action_descent_cosine,
        "action_descent": action_directional_derivative < 0.0,
        "actual_fp32_delta_identity_audit": identity_delta_rows,
        "identity_nullspace_reaudited_on_actual_fp32_delta": True,
        "application_contract": "direct_parameter_add_only",
        "candidate_only_not_a_realized_parameter_displacement": True,
        "optimizer_step_allowed": False,
        "adam_momentum_weight_decay_or_preconditioner_allowed": False,
        "actual_checkpoint_content_audit_required_before_apply": True,
        "native_post_step_gate_required": True,
        "realized_theta_after_minus_theta_before_audit_required": True,
        "runtime_apply_authorized": False,
    }
    return StatelessTrustRegionDelta(
        layout=transport.layout,
        delta_by_parameter=named,
        receipt=_seal(unsigned),
    )


def audit_realized_parameter_displacement(
    *,
    transport: TransportResult,
    candidate: StatelessTrustRegionDelta,
    full_parameter_state_before: Mapping[str, torch.Tensor],
    full_parameter_state_after: Mapping[str, torch.Tensor],
    minimum_candidate_cosine: float = 0.99999,
    maximum_candidate_relative_error: float = 2.0e-5,
) -> RealizedDisplacementAudit:
    """Audit ``theta_after - theta_before`` after an external direct add.

    This function does not apply or roll back a step.  In particular, a tiny
    requested FP32 delta that rounds to zero at the realized parameter values
    becomes a sealed NO-GO result rather than inheriting the candidate audit.
    """

    transport = _revalidate_transport_result(transport)
    if not transport.geometry_authorized:
        raise PairV7TransportError("NO-GO transport cannot audit a realized delta")
    if not isinstance(candidate, StatelessTrustRegionDelta):
        raise PairV7TransportError("candidate delta type differs")
    candidate_receipt = candidate.receipt
    if not isinstance(candidate_receipt, Mapping):
        raise PairV7TransportError("candidate delta receipt differs")
    unsigned_candidate = dict(candidate_receipt)
    candidate_digest = unsigned_candidate.pop("receipt_digest", None)
    if (
        candidate_receipt.get("schema_version") != DELTA_SCHEMA
        or not isinstance(candidate_digest, str)
        or object_sha256(unsigned_candidate) != candidate_digest
        or candidate_receipt.get("transport_receipt_digest")
        != transport.receipt["receipt_digest"]
        or candidate_receipt.get("pre_step_parameter_state_sha256")
        != transport.parameter_state_sha256
        or candidate_receipt.get("checkpoint_content_receipt_digest")
        != transport.checkpoint_content_receipt_digest
        or candidate_receipt.get("parameter_layout_digest")
        != transport.layout.layout_digest
        or candidate.layout.layout_digest != transport.layout.layout_digest
    ):
        raise PairV7TransportError("candidate delta binding differs")
    requested = transport.layout.flatten(
        candidate.delta_by_parameter, label="candidate displacement"
    )
    requested_norm = float(torch.linalg.vector_norm(requested).item())
    if (
        _tensor_sha256(requested.to(dtype=torch.float32))
        != candidate_receipt.get("actual_fp32_delta_sha256")
        or not math.isclose(
            requested_norm,
            float(candidate_receipt.get("actual_fp32_delta_norm", float("nan"))),
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
    ):
        raise PairV7TransportError(
            "candidate delta tensors no longer match their sealed receipt"
        )
    minimum_cosine = _finite_scalar(
        minimum_candidate_cosine, label="minimum candidate cosine"
    )
    maximum_error = _finite_scalar(
        maximum_candidate_relative_error,
        label="maximum candidate relative error",
    )
    if not 0.0 < minimum_cosine <= 1.0 or not 0.0 < maximum_error <= 1.0e-3:
        raise PairV7TransportError("realized displacement thresholds differ")
    if (
        not isinstance(full_parameter_state_before, Mapping)
        or not isinstance(full_parameter_state_after, Mapping)
        or set(full_parameter_state_before) != set(full_parameter_state_after)
        or not set(transport.layout.names).issubset(full_parameter_state_before)
    ):
        raise PairV7TransportError("realized full parameter-state closure differs")
    before_digest = named_parameter_state_sha256(full_parameter_state_before)
    if before_digest != transport.parameter_state_sha256:
        raise PairV7TransportError(
            "realized displacement before-state differs from gradient/probe state"
        )
    after_digest = named_parameter_state_sha256(full_parameter_state_after)
    updated = set(transport.layout.names)
    for name in sorted(set(full_parameter_state_before) - updated):
        left = full_parameter_state_before[name]
        right = full_parameter_state_after[name]
        if (
            left.dtype != right.dtype
            or left.device != right.device
            or tuple(left.shape) != tuple(right.shape)
            or not torch.equal(left.detach(), right.detach())
        ):
            raise PairV7TransportError(
                f"non-candidate parameter changed during realized add: {name}"
            )
    realized_named: dict[str, torch.Tensor] = {}
    for name in transport.layout.names:
        before = full_parameter_state_before[name]
        after = full_parameter_state_after[name]
        if (
            before.dtype != torch.float32
            or after.dtype != torch.float32
            or before.device != after.device
            or tuple(before.shape) != tuple(after.shape)
        ):
            raise PairV7TransportError(f"realized parameter {name} layout differs")
        realized_named[name] = (after.detach() - before.detach()).float().contiguous()
    realized = transport.layout.flatten(realized_named, label="realized displacement")
    realized_norm = float(torch.linalg.vector_norm(realized).item())
    failures: list[str] = []
    if realized_norm <= 0.0:
        failures.append("REALIZED_PARAMETER_DISPLACEMENT_ROUNDED_TO_ZERO")
    candidate_cosine = (
        float(torch.dot(realized, requested).item())
        / (realized_norm * requested_norm)
        if realized_norm > 0.0 and requested_norm > 0.0
        else None
    )
    relative_error = float(torch.linalg.vector_norm(realized - requested).item()) / max(
        requested_norm, 1.0e-30
    )
    if candidate_cosine is None or candidate_cosine < minimum_cosine:
        failures.append("REALIZED_DISPLACEMENT_CANDIDATE_COSINE_TOO_LOW")
    if relative_error > maximum_error:
        failures.append("REALIZED_DISPLACEMENT_DIFFERS_FROM_CANDIDATE")
    maximum_norm = float(candidate_receipt["maximum_delta_norm"])
    if realized_norm > maximum_norm * (1.0 + 1.0e-7):
        failures.append("REALIZED_DISPLACEMENT_ESCAPED_TRUST_REGION")

    action = transport.layout.flatten(
        transport.action_gradient_by_parameter, label="bound action gradient"
    )
    action_norm = float(torch.linalg.vector_norm(action).item())
    directional = float(torch.dot(action, realized).item())
    action_cosine = (
        -directional / (action_norm * realized_norm)
        if action_norm > 0.0 and realized_norm > 0.0
        else None
    )
    thresholds = transport.receipt["thresholds"]
    if directional >= -float(thresholds["minimum_action_descent_gain"]):
        failures.append("REALIZED_DISPLACEMENT_IS_NOT_ACTION_DESCENT")
    if (
        action_cosine is None
        or action_cosine < float(thresholds["minimum_action_descent_cosine"])
    ):
        failures.append("REALIZED_ACTION_DESCENT_COSINE_TOO_LOW")
    identity_rows: list[dict[str, Any]] = []
    for probe in transport.identity_probes:
        probe_flat = transport.layout.flatten(
            probe.gradient_by_parameter,
            label=f"realized identity probe {probe.probe_id}",
        )
        unit = probe_flat / torch.linalg.vector_norm(probe_flat)
        dot = float(torch.dot(unit, realized).item())
        absolute = abs(dot)
        cosine = absolute / realized_norm if realized_norm > 0.0 else None
        dot_limit = (
            float(thresholds["identity_dot_absolute_tolerance"])
            + float(thresholds["identity_dot_relative_tolerance"]) * realized_norm
        )
        if absolute > dot_limit:
            failures.append(f"REALIZED_IDENTITY_DOT_EXCEEDED:{probe.probe_id}")
        if cosine is None or cosine > float(thresholds["maximum_identity_cosine"]):
            failures.append(f"REALIZED_IDENTITY_COSINE_EXCEEDED:{probe.probe_id}")
        identity_rows.append(
            {
                "probe_id": probe.probe_id,
                "family": probe.family,
                "signed_unit_dot": dot,
                "absolute_unit_dot": absolute,
                "absolute_cosine": cosine,
                "dot_limit": dot_limit,
            }
        )
    failures = sorted(set(failures))
    safe = not failures
    receipt = _seal(
        {
            "schema_version": REALIZED_DELTA_SCHEMA,
            "method_name": METHOD_NAME,
            "realized_displacement_safe": safe,
            "runtime_apply_authorized": False,
            "failure_codes": failures,
            "transport_receipt_digest": transport.receipt["receipt_digest"],
            "candidate_delta_receipt_digest": candidate_digest,
            "checkpoint_content_receipt_digest": (
                transport.checkpoint_content_receipt_digest
            ),
            "full_parameter_state_before_sha256": before_digest,
            "full_parameter_state_after_sha256": after_digest,
            "requested_delta_sha256": _tensor_sha256(requested.to(dtype=torch.float32)),
            "realized_delta_sha256": _tensor_sha256(realized.to(dtype=torch.float32)),
            "requested_delta_norm": requested_norm,
            "realized_delta_norm": realized_norm,
            "realized_candidate_cosine": candidate_cosine,
            "realized_candidate_relative_error": relative_error,
            "action_directional_derivative": directional,
            "action_descent_cosine": action_cosine,
            "identity_audit": identity_rows,
            "audit_observed_theta_after_minus_theta_before": True,
            "audit_executed_parameter_add": False,
            "rollback_executed": False,
        }
    )
    return RealizedDisplacementAudit(
        layout=transport.layout,
        realized_delta_by_parameter=transport.layout.unflatten(
            realized, label="realized displacement"
        ),
        realized_displacement_safe=safe,
        receipt=receipt,
    )


def build_post_step_native_rollback_receipt(
    *,
    pre_action_loss: float,
    post_action_loss: float,
    action_directional_derivative: float,
    pre_identity_metric_by_family: Mapping[str, float],
    post_identity_metric_by_family: Mapping[str, float],
    pre_step_parameter_state_sha256: str,
    post_step_parameter_state_sha256: str,
    config: RollbackConfig = RollbackConfig(),
) -> dict[str, Any]:
    """Build an untrusted metric decision; never claim or execute rollback.

    The scalar metrics are caller supplied and this math-only helper cannot
    bind them to actual Bernini forwards.  A future runtime must replace this
    with hash-bound native measurements and perform/verify the restoration.
    """

    config.validate()
    pre_action = _finite_scalar(pre_action_loss, label="pre action loss")
    post_action = _finite_scalar(post_action_loss, label="post action loss")
    directional = _finite_scalar(
        action_directional_derivative, label="action directional derivative"
    )
    expected = set(REQUIRED_IDENTITY_FAMILIES)
    for label, values in (
        ("pre identity metrics", pre_identity_metric_by_family),
        ("post identity metrics", post_identity_metric_by_family),
    ):
        if not isinstance(values, Mapping) or set(values) != expected:
            raise PairV7TransportError(f"{label} family closure differs")
    pre_identity = {
        family: _finite_scalar(pre_identity_metric_by_family[family], label=f"pre {family}")
        for family in REQUIRED_IDENTITY_FAMILIES
    }
    post_identity = {
        family: _finite_scalar(post_identity_metric_by_family[family], label=f"post {family}")
        for family in REQUIRED_IDENTITY_FAMILIES
    }
    failures: list[str] = []
    if directional >= 0.0:
        failures.append("CANDIDATE_DIRECTION_IS_NOT_ACTION_DESCENT")
    armijo_bound = pre_action + config.armijo_c1 * directional
    if post_action > armijo_bound:
        failures.append("ACTION_ARMIJO_GATE_FAILED")
    identity_rows: dict[str, Any] = {}
    for family in REQUIRED_IDENTITY_FAMILIES:
        regression = post_identity[family] - pre_identity[family]
        limit = (
            config.maximum_identity_absolute_regression
            + config.maximum_identity_relative_regression * abs(pre_identity[family])
        )
        passed = regression <= limit
        if not passed:
            failures.append(f"IDENTITY_NATIVE_FIELD_REGRESSION:{family}")
        identity_rows[family] = {
            "pre": pre_identity[family],
            "post": post_identity[family],
            "regression": regression,
            "allowed_regression": limit,
            "passed": passed,
        }
    failures = sorted(failures)
    unsigned = {
        "schema_version": ROLLBACK_SCHEMA,
        "method_name": METHOD_NAME,
        "accepted": not failures,
        "rollback_required": bool(failures),
        "decision_only": True,
        "measurement_inputs_authoritatively_bound": False,
        "rollback_executed": False,
        "runtime_apply_authorized": False,
        "failure_codes": failures,
        "pre_step_parameter_state_sha256": _sha256(
            pre_step_parameter_state_sha256, label="pre-step parameter state"
        ),
        "post_step_parameter_state_sha256": _sha256(
            post_step_parameter_state_sha256, label="post-step parameter state"
        ),
        "pre_action_loss": pre_action,
        "post_action_loss": post_action,
        "action_directional_derivative": directional,
        "armijo_bound": armijo_bound,
        "identity_native_field_metrics": identity_rows,
        "rollback_state_closure": [
            "lora_parameters",
            "optimizer_moments_and_step_if_any",
            "grad_scaler_if_any",
            "scheduler_if_any",
            "rng_state_all_ranks",
        ],
        "all_rank_state_digest_recheck_required": True,
        "held_out_confirmation_may_control_optimizer_or_rollback": False,
    }
    return _seal(unsigned)


def build_method_contract_receipt() -> dict[str, Any]:
    """Return the closed information-flow and schedule contract."""

    unsigned = {
        "schema_version": CONTRACT_SCHEMA,
        "method_name": METHOD_NAME,
        "method_status": "math_core_and_unit_tests_only_no_runtime_claim",
        "coordinate_arms": {
            "action": "sealed_fit_only_pure_t2v_geometry_gradient_only",
            "identity": "full_exact81_deployed_V_only_post_APG_feature_sketch_vjps_only",
            "coupling": "shared_action_lora_parameter_names_and_parameter_space_only",
        },
        "optimization": {
            "simple_loss_sum": False,
            "pcgrad": False,
            "constraint": "identity_tangent_first_order_nullspace",
            "constrained_object": "actual_stateless_lora_parameter_displacement",
            "direct_parameter_add_only": True,
            "adam_momentum_weight_decay_or_preconditioner_after_projection": False,
            "candidate_delta_is_not_realized_displacement": True,
            "theta_after_minus_theta_before_must_be_reaudited": True,
        },
        "forbidden_cross_coordinate_carriers": [
            "rgb",
            "latent",
            "residual",
            "target",
            "noise",
            "motion_donor",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
        ],
        "pure_t2v_visual_role": "action_arm_reward_trajectory_only",
        "pure_t2v_visual_used_as_rv2v_target_noise_source_or_donor": False,
        "inference_contract": "source_video_only_V_plus_prompt_plus_action_lora",
        "inference_image_reference_count": 0,
        "inference_requires_training_identity_probes": False,
        "frame_count": EXACT81_FRAME_COUNT,
        "schedule_step_count": EXACT40_STEP_COUNT,
        "exact40_zero_update_indices": list(EXACT40_ZERO_UPDATE_INDICES),
        "fit_only_geometry_evidence_must_be_create_only_and_sealed": True,
        "population_confirmation_or_optimizer_go_consumed": False,
        "math_core_candidate_binder_proves_gradient_origin": False,
        "math_core_rollback_helper_executes_rollback": False,
        "native_post_step_gate_required": True,
        "scientific_action_editing_success_claim": False,
    }
    return _seal(unsigned)


__all__ = [
    "ActionGradientProvenance",
    "CONTRACT_SCHEMA",
    "DELTA_SCHEMA",
    "EXACT40_ZERO_UPDATE_INDICES",
    "EXACT81_FRAME_COUNT",
    "GradientLayout",
    "IdentityGradientProbe",
    "METHOD_NAME",
    "PairV7TransportError",
    "REALIZED_DELTA_SCHEMA",
    "RealizedDisplacementAudit",
    "REQUIRED_IDENTITY_FAMILIES",
    "ROLLBACK_SCHEMA",
    "RollbackConfig",
    "StatelessTrustRegionDelta",
    "TRANSPORT_SCHEMA",
    "TransportConfig",
    "TransportResult",
    "audit_realized_parameter_displacement",
    "build_method_contract_receipt",
    "build_post_step_native_rollback_receipt",
    "build_stateless_trust_region_delta",
    "canonical_json_bytes",
    "named_parameter_state_sha256",
    "object_sha256",
    "project_action_gradient_to_identity_nullspace",
]
