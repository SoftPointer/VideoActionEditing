#!/usr/bin/env python3
"""Pure-tensor factorized marginal composition for DMIQ action editing.

The composer combines two deliberately different objects:

* a same-current-state, frozen no-op target-tail co-state ``h0`` carrying the
  source/identity marginal; and
* a generic source-invariant frozen action subspace ``U_A`` carrying only
  action coordinates.

For hidden layout ``[B,L,S,T,P,D]``, coordinate-local bases
``[L,S,T,P,D,R]``, and predicted coefficients ``[B,L,S,T,P,R]`` it constructs

``h = h0 + U_A alpha``.

Consequently the update has zero component outside ``span(U_A)`` and
``P_perp(h) == P_perp(h0)``.  This does *not* require a correct-source-specific
raw action residual.  Proposal RGB, proposal/video latents, paired targets,
masks, flow, pose, and tracks are intentionally absent from the API.

All carrier/evidence tensors are detached and frozen.  Only predicted action
coefficients may carry gradients, and the returned training loss is defined
solely in coefficient space.  This geometry core never authorizes an optimizer
step; external evidence and causal gates remain mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from typing import Any, Mapping, Sequence


METHOD_NAME = "dmiq-factorized-marginal-composer"
SCHEMA_VERSION = "dmiq-factorized-marginal-composer-v1"
PROVENANCE_SCHEMA = "dmiq-factorized-marginal-provenance-v1"
COORDINATE_SCHEMA = "dmiq-factorized-coordinate-binding-v1"
RAW_DIGEST_SCHEMA = "dmiq-canonical-fp32-little-endian-v1"

ACTION_BASIS_SEMANTICS = "generic_source_invariant_action_subspace"
NOOP_HIDDEN_SEMANTICS = "same_current_state_frozen_noop_costate"


class DMIQFactorizedMarginalError(RuntimeError):
    """Raised before an ambiguous or contract-breaking composition is returned."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise DMIQFactorizedMarginalError(
            "DMIQ factorized marginal composition requires PyTorch"
        ) from error
    return torch


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DMIQFactorizedMarginalError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _composite_digest(parts: Sequence[str], *, domain: str) -> str:
    digest = hashlib.sha256()
    domain_bytes = domain.encode("utf-8")
    digest.update(len(domain_bytes).to_bytes(8, "big"))
    digest.update(domain_bytes)
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_tensor(
    value: Any,
    *,
    label: str,
    ndim: int,
    frozen: bool,
) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise DMIQFactorizedMarginalError(f"{label} must be a torch.Tensor")
    if value.layout != torch.strided or value.is_meta:
        raise DMIQFactorizedMarginalError(
            f"{label} must be a dense non-meta tensor"
        )
    if value.dtype != torch.float32:
        raise DMIQFactorizedMarginalError(f"{label} must be torch.float32")
    if value.ndim != ndim or any(int(size) <= 0 for size in value.shape):
        raise DMIQFactorizedMarginalError(
            f"{label} must be rank {ndim} with positive dimensions"
        )
    if not value.is_contiguous():
        raise DMIQFactorizedMarginalError(f"{label} must be contiguous")
    if frozen and (value.requires_grad or value.grad_fn is not None):
        raise DMIQFactorizedMarginalError(
            f"{label} must be detached and frozen"
        )
    if not frozen and not value.requires_grad:
        raise DMIQFactorizedMarginalError(
            f"{label} must be a trainable predicted tensor"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise DMIQFactorizedMarginalError(f"{label} must be finite")
    return value


def _storage_identity(value: Any) -> tuple[str, int]:
    return (str(value.device), int(value.untyped_storage().data_ptr()))


def _require_distinct_storages(named_values: Sequence[tuple[str, Any]]) -> None:
    owners: dict[tuple[str, int], str] = {}
    for label, value in named_values:
        identity = _storage_identity(value)
        previous = owners.get(identity)
        if previous is not None:
            raise DMIQFactorizedMarginalError(
                f"{label} aliases storage owned by {previous}"
            )
        owners[identity] = label


def tensor_raw_value_digest(value: Any, *, label: str = "tensor") -> str:
    """Hash exact detached FP32 shape and canonical little-endian values."""

    candidate = _validate_tensor(
        value,
        label=label,
        ndim=int(getattr(value, "ndim", -1)),
        frozen=True,
    )
    array = (
        candidate.detach()
        .to(device="cpu")
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    header = (
        f"{RAW_DIGEST_SCHEMA}|shape="
        + ",".join(str(int(size)) for size in candidate.shape)
        + "|"
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    raw = memoryview(array).cast("B")
    chunk = 8 * 1024 * 1024
    for offset in range(0, len(raw), chunk):
        digest.update(raw[offset : offset + chunk])
    return digest.hexdigest()


def _validate_unique_int_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or not value
        or any(type(item) is not int or item < 0 for item in value)
        or len(set(value)) != len(value)
    ):
        raise DMIQFactorizedMarginalError(
            f"{label} must be a nonempty tuple of unique nonnegative integers"
        )
    return value


def _validate_unique_site_tuple(value: Any) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or not value
        or any(
            type(item) is not str
            or not item
            or item != item.strip()
            or any(character.isspace() for character in item)
            for item in value
        )
        or len(set(value)) != len(value)
    ):
        raise DMIQFactorizedMarginalError(
            "site_ids must be unique nonempty whitespace-free strings"
        )
    return value


@dataclass(frozen=True)
class MarginalCoordinateBinding:
    """Exact layer/site/phase/spatial axes for one factorized marginal."""

    schema_version: str
    layer_ids: tuple[int, ...]
    site_ids: tuple[str, ...]
    phase_ids: tuple[int, ...]
    spatial_ids: tuple[int, ...]

    def validate(self) -> None:
        if self.schema_version != COORDINATE_SCHEMA:
            raise DMIQFactorizedMarginalError(
                "coordinate schema version is not registered"
            )
        _validate_unique_int_tuple(self.layer_ids, label="layer_ids")
        _validate_unique_site_tuple(self.site_ids)
        _validate_unique_int_tuple(self.phase_ids, label="phase_ids")
        _validate_unique_int_tuple(self.spatial_ids, label="spatial_ids")

    def digest(self) -> str:
        self.validate()
        return _composite_digest(
            (
                "layer_ids",
                str(len(self.layer_ids)),
                *tuple(str(item) for item in self.layer_ids),
                "site_ids",
                str(len(self.site_ids)),
                *self.site_ids,
                "phase_ids",
                str(len(self.phase_ids)),
                *tuple(str(item) for item in self.phase_ids),
                "spatial_ids",
                str(len(self.spatial_ids)),
                *tuple(str(item) for item in self.spatial_ids),
            ),
            domain=COORDINATE_SCHEMA,
        )


def _positive_real(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DMIQFactorizedMarginalError(f"{label} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise DMIQFactorizedMarginalError(
            f"{label} must be finite and strictly positive"
        )
    return normalized


@dataclass(frozen=True)
class MarginalComposerConfig:
    """Pre-registered hard caps and numerical audit tolerances."""

    max_coefficient_norm: float = 4.0
    max_update_norm: float = 4.0
    max_relative_update: float = 0.50
    base_norm_floor: float = 1.0e-3
    orthogonality_tolerance: float = 2.0e-5
    complement_tolerance: float = 5.0e-5
    coefficient_recovery_tolerance: float = 5.0e-5
    epsilon: float = 1.0e-8

    def validate(self) -> None:
        for field_name in (
            "max_coefficient_norm",
            "max_update_norm",
            "max_relative_update",
            "base_norm_floor",
            "orthogonality_tolerance",
            "complement_tolerance",
            "coefficient_recovery_tolerance",
            "epsilon",
        ):
            _positive_real(getattr(self, field_name), label=field_name)

    def digest(self) -> str:
        self.validate()
        return _composite_digest(
            tuple(
                float(getattr(self, field_name)).hex()
                for field_name in (
                    "max_coefficient_norm",
                    "max_update_norm",
                    "max_relative_update",
                    "base_norm_floor",
                    "orthogonality_tolerance",
                    "complement_tolerance",
                    "coefficient_recovery_tolerance",
                    "epsilon",
                )
            ),
            domain="dmiq-factorized-marginal-config-v1",
        )


@dataclass(frozen=True)
class MarginalComposerProvenance:
    """Closed upstream/tensor binding accepted by the geometry core."""

    schema_version: str
    checkpoint_tree_sha256: str
    noop_query_receipt_sha256: str
    action_basis_evidence_receipt_sha256: str
    identity_feasibility_receipt_sha256: str
    noop_hidden_digest: str
    action_basis_digest: str
    pass_a_coefficients_digest: str
    source_prefix_digest: str
    coordinates_digest: str
    composer_config_digest: str
    action_basis_semantics: str
    noop_hidden_semantics: str
    raw_correct_source_action_residual_required: bool
    same_current_state_verified: bool
    basis_evidence_valid: bool
    action_evidence_valid: bool
    identity_feasible: bool
    binding_digest: str

    def validate(self) -> None:
        if self.schema_version != PROVENANCE_SCHEMA:
            raise DMIQFactorizedMarginalError(
                "marginal provenance schema version is not registered"
            )
        for field_name in (
            "checkpoint_tree_sha256",
            "noop_query_receipt_sha256",
            "action_basis_evidence_receipt_sha256",
            "identity_feasibility_receipt_sha256",
            "noop_hidden_digest",
            "action_basis_digest",
            "pass_a_coefficients_digest",
            "source_prefix_digest",
            "coordinates_digest",
            "composer_config_digest",
            "binding_digest",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)
        if self.action_basis_semantics != ACTION_BASIS_SEMANTICS:
            raise DMIQFactorizedMarginalError(
                "action basis must be generic and source invariant"
            )
        if self.noop_hidden_semantics != NOOP_HIDDEN_SEMANTICS:
            raise DMIQFactorizedMarginalError(
                "no-op hidden must be a same-current-state frozen co-state"
            )
        if self.raw_correct_source_action_residual_required is not False:
            raise DMIQFactorizedMarginalError(
                "raw correct-source-specific action residuals are forbidden"
            )
        for field_name in (
            "same_current_state_verified",
            "basis_evidence_valid",
            "action_evidence_valid",
            "identity_feasible",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise DMIQFactorizedMarginalError(
                    f"{field_name} must be an explicit boolean"
                )
        if self.binding_digest != _provenance_payload_digest(self):
            raise DMIQFactorizedMarginalError(
                "marginal provenance binding digest is inconsistent"
            )


def _provenance_payload_digest(provenance: MarginalComposerProvenance) -> str:
    return _composite_digest(
        (
            provenance.schema_version,
            provenance.checkpoint_tree_sha256,
            provenance.noop_query_receipt_sha256,
            provenance.action_basis_evidence_receipt_sha256,
            provenance.identity_feasibility_receipt_sha256,
            provenance.noop_hidden_digest,
            provenance.action_basis_digest,
            provenance.pass_a_coefficients_digest,
            provenance.source_prefix_digest,
            provenance.coordinates_digest,
            provenance.composer_config_digest,
            provenance.action_basis_semantics,
            provenance.noop_hidden_semantics,
            str(provenance.raw_correct_source_action_residual_required),
            str(provenance.same_current_state_verified),
            str(provenance.basis_evidence_valid),
            str(provenance.action_evidence_valid),
            str(provenance.identity_feasible),
        ),
        domain=PROVENANCE_SCHEMA,
    )


def _validate_frozen_geometry(
    noop_tail_hidden: Any,
    action_basis: Any,
    pass_a_coefficients: Any,
    source_prefix_snapshot: Any,
    coordinates: MarginalCoordinateBinding,
) -> tuple[int, int, int, int, int, int, int]:
    noop = _validate_tensor(
        noop_tail_hidden,
        label="noop_tail_hidden",
        ndim=6,
        frozen=True,
    )
    basis = _validate_tensor(
        action_basis,
        label="action_basis",
        ndim=6,
        frozen=True,
    )
    coefficients = _validate_tensor(
        pass_a_coefficients,
        label="pass_a_coefficients",
        ndim=6,
        frozen=True,
    )
    prefix = _validate_tensor(
        source_prefix_snapshot,
        label="source_prefix_snapshot",
        ndim=3,
        frozen=True,
    )
    if not isinstance(coordinates, MarginalCoordinateBinding):
        raise DMIQFactorizedMarginalError(
            "coordinates must be a MarginalCoordinateBinding"
        )
    coordinates.validate()
    batch, layers, sites, phases, positions, hidden = (
        int(size) for size in noop.shape
    )
    expected_basis_prefix = (layers, sites, phases, positions, hidden)
    if tuple(int(size) for size in basis.shape[:-1]) != expected_basis_prefix:
        raise DMIQFactorizedMarginalError(
            "action_basis must be [L,S,T,P,D,R] aligned with no-op hidden"
        )
    rank = int(basis.shape[-1])
    if rank > hidden:
        raise DMIQFactorizedMarginalError(
            "action basis rank cannot exceed hidden dimension"
        )
    expected_coefficients = (batch, layers, sites, phases, positions, rank)
    if tuple(int(size) for size in coefficients.shape) != expected_coefficients:
        raise DMIQFactorizedMarginalError(
            "pass_a_coefficients must be [B,L,S,T,P,R]"
        )
    if tuple(int(size) for size in prefix.shape)[::2] != (batch, hidden):
        raise DMIQFactorizedMarginalError(
            "source prefix must be [B,N,D] with matching batch/hidden size"
        )
    if (
        len(coordinates.layer_ids) != layers
        or len(coordinates.site_ids) != sites
        or len(coordinates.phase_ids) != phases
        or len(coordinates.spatial_ids) != positions
    ):
        raise DMIQFactorizedMarginalError(
            "coordinate counts differ from layer/site/phase/spatial geometry"
        )
    if not (
        noop.device == basis.device == coefficients.device == prefix.device
    ):
        raise DMIQFactorizedMarginalError(
            "all frozen marginal tensors must share one device"
        )
    _require_distinct_storages(
        (
            ("noop_tail_hidden", noop),
            ("action_basis", basis),
            ("pass_a_coefficients", coefficients),
            ("source_prefix_snapshot", prefix),
        )
    )
    return batch, layers, sites, phases, positions, hidden, rank


def bind_marginal_composer_provenance(
    noop_tail_hidden: Any,
    action_basis: Any,
    pass_a_coefficients: Any,
    source_prefix_snapshot: Any,
    coordinates: MarginalCoordinateBinding,
    *,
    checkpoint_tree_sha256: str,
    noop_query_receipt_sha256: str,
    action_basis_evidence_receipt_sha256: str,
    identity_feasibility_receipt_sha256: str,
    same_current_state_verified: bool,
    basis_evidence_valid: bool,
    action_evidence_valid: bool,
    identity_feasible: bool,
    config: MarginalComposerConfig = MarginalComposerConfig(),
) -> MarginalComposerProvenance:
    """Create a closed binding from actual frozen tensor values."""

    _validate_frozen_geometry(
        noop_tail_hidden,
        action_basis,
        pass_a_coefficients,
        source_prefix_snapshot,
        coordinates,
    )
    if not isinstance(config, MarginalComposerConfig):
        raise DMIQFactorizedMarginalError(
            "config must be MarginalComposerConfig"
        )
    config.validate()
    for field_name, value in (
        ("checkpoint_tree_sha256", checkpoint_tree_sha256),
        ("noop_query_receipt_sha256", noop_query_receipt_sha256),
        (
            "action_basis_evidence_receipt_sha256",
            action_basis_evidence_receipt_sha256,
        ),
        (
            "identity_feasibility_receipt_sha256",
            identity_feasibility_receipt_sha256,
        ),
    ):
        _require_sha256(value, label=field_name)
    for field_name, value in (
        ("same_current_state_verified", same_current_state_verified),
        ("basis_evidence_valid", basis_evidence_valid),
        ("action_evidence_valid", action_evidence_valid),
        ("identity_feasible", identity_feasible),
    ):
        if type(value) is not bool:
            raise DMIQFactorizedMarginalError(
                f"{field_name} must be an explicit boolean"
            )
    values = {
        "schema_version": PROVENANCE_SCHEMA,
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "noop_query_receipt_sha256": noop_query_receipt_sha256,
        "action_basis_evidence_receipt_sha256": (
            action_basis_evidence_receipt_sha256
        ),
        "identity_feasibility_receipt_sha256": (
            identity_feasibility_receipt_sha256
        ),
        "noop_hidden_digest": tensor_raw_value_digest(
            noop_tail_hidden,
            label="noop_tail_hidden",
        ),
        "action_basis_digest": tensor_raw_value_digest(
            action_basis,
            label="action_basis",
        ),
        "pass_a_coefficients_digest": tensor_raw_value_digest(
            pass_a_coefficients,
            label="pass_a_coefficients",
        ),
        "source_prefix_digest": tensor_raw_value_digest(
            source_prefix_snapshot,
            label="source_prefix_snapshot",
        ),
        "coordinates_digest": coordinates.digest(),
        "composer_config_digest": config.digest(),
        "action_basis_semantics": ACTION_BASIS_SEMANTICS,
        "noop_hidden_semantics": NOOP_HIDDEN_SEMANTICS,
        "raw_correct_source_action_residual_required": False,
        "same_current_state_verified": same_current_state_verified,
        "basis_evidence_valid": basis_evidence_valid,
        "action_evidence_valid": action_evidence_valid,
        "identity_feasible": identity_feasible,
    }
    unbound = MarginalComposerProvenance(binding_digest="0" * 64, **values)
    provenance = MarginalComposerProvenance(
        binding_digest=_provenance_payload_digest(unbound),
        **values,
    )
    provenance.validate()
    return provenance


@dataclass(frozen=True)
class ActionMarginalDiagnostics:
    """Action-only diagnostics; contains no identity compensation score."""

    basis_orthogonality_max_abs_error: float
    basis_orthogonality_passed: bool
    basis_evidence_valid: bool
    action_evidence_valid: bool
    same_current_state_verified: bool
    pass_a_within_trust_cap: bool
    predicted_coefficient_norm_max: float
    pass_a_coefficient_norm_max: float
    effective_coefficient_norm_max: float
    trust_radius_min: float
    trust_radius_max: float
    coefficient_cap_applied: bool
    coefficient_mse: float
    coefficient_recovery_max_abs_error: float
    coefficient_recovery_passed: bool
    null_action_update: bool


@dataclass(frozen=True)
class IdentityMarginalDiagnostics:
    """Identity/source-only diagnostics; never offsets action evidence."""

    identity_feasible: bool
    source_prefix_exact: bool
    source_prefix_no_write_passed: bool
    noop_hidden_unchanged: bool
    orthogonal_complement_update_max_abs: float
    orthogonal_complement_preservation_max_abs: float
    orthogonal_complement_preserved: bool
    scalar_compensation_used: bool


@dataclass(frozen=True)
class FactorizedMarginalResult:
    """Composed target-tail hidden state and fail-closed evidence."""

    composed_tail_hidden: Any
    effective_coefficients: Any
    coefficient_loss: Any
    action_diagnostics: ActionMarginalDiagnostics
    identity_diagnostics: IdentityMarginalDiagnostics
    local_composition_valid: bool
    optimizer_updates_authorized: bool
    receipt: Mapping[str, Any]


def _basis_orthogonality_error(action_basis: Any) -> float:
    torch = _torch()
    rank = int(action_basis.shape[-1])
    work = action_basis.detach().to(device="cpu", dtype=torch.float64)
    gram = work.transpose(-1, -2) @ work
    identity = torch.eye(rank, dtype=torch.float64).expand_as(gram)
    return float((gram - identity).abs().max().item())


def _max_abs(value: Any) -> float:
    return float(value.detach().abs().max().to(device="cpu").item())


def compose_factorized_action_marginal(
    noop_tail_hidden: Any,
    action_basis: Any,
    predicted_coefficients: Any,
    pass_a_coefficients: Any,
    source_prefix_snapshot: Any,
    source_prefix_runtime: Any,
    coordinates: MarginalCoordinateBinding,
    provenance: MarginalComposerProvenance,
    *,
    config: MarginalComposerConfig = MarginalComposerConfig(),
) -> FactorizedMarginalResult:
    """Compose a source-invariant action marginal onto a no-op co-state.

    Structural/provenance mutations raise.  Scientifically invalid evidence,
    infeasible identity, nonorthogonal basis, or an out-of-trust Pass-A target
    returns an exact null update and a zero-gradient coefficient loss.
    """

    torch = _torch()
    if not isinstance(config, MarginalComposerConfig):
        raise DMIQFactorizedMarginalError(
            "config must be MarginalComposerConfig"
        )
    config.validate()
    geometry = _validate_frozen_geometry(
        noop_tail_hidden,
        action_basis,
        pass_a_coefficients,
        source_prefix_snapshot,
        coordinates,
    )
    batch, layers, sites, phases, positions, hidden, rank = geometry
    predicted = _validate_tensor(
        predicted_coefficients,
        label="predicted_coefficients",
        ndim=6,
        frozen=False,
    )
    prefix_runtime = _validate_tensor(
        source_prefix_runtime,
        label="source_prefix_runtime",
        ndim=3,
        frozen=True,
    )
    expected_coefficients = (batch, layers, sites, phases, positions, rank)
    if tuple(int(size) for size in predicted.shape) != expected_coefficients:
        raise DMIQFactorizedMarginalError(
            "predicted_coefficients must be [B,L,S,T,P,R]"
        )
    if tuple(prefix_runtime.shape) != tuple(source_prefix_snapshot.shape):
        raise DMIQFactorizedMarginalError(
            "source prefix runtime shape differs from its snapshot"
        )
    if not (
        predicted.device
        == noop_tail_hidden.device
        == source_prefix_runtime.device
    ):
        raise DMIQFactorizedMarginalError(
            "predicted coefficients and frozen tensors must share one device"
        )
    _require_distinct_storages(
        (
            ("noop_tail_hidden", noop_tail_hidden),
            ("action_basis", action_basis),
            ("predicted_coefficients", predicted),
            ("pass_a_coefficients", pass_a_coefficients),
            ("source_prefix_snapshot", source_prefix_snapshot),
            ("source_prefix_runtime", prefix_runtime),
        )
    )
    if not torch.equal(source_prefix_snapshot, prefix_runtime):
        raise DMIQFactorizedMarginalError(
            "source prefix changed; composer cannot repair a prefix write"
        )
    if not isinstance(provenance, MarginalComposerProvenance):
        raise DMIQFactorizedMarginalError(
            "provenance must be MarginalComposerProvenance"
        )
    provenance.validate()
    if provenance.composer_config_digest != config.digest():
        raise DMIQFactorizedMarginalError(
            "runtime trust/tolerance config differs from provenance"
        )
    actual_digests = {
        "noop_hidden_digest": tensor_raw_value_digest(
            noop_tail_hidden,
            label="noop_tail_hidden",
        ),
        "action_basis_digest": tensor_raw_value_digest(
            action_basis,
            label="action_basis",
        ),
        "pass_a_coefficients_digest": tensor_raw_value_digest(
            pass_a_coefficients,
            label="pass_a_coefficients",
        ),
        "source_prefix_digest": tensor_raw_value_digest(
            source_prefix_snapshot,
            label="source_prefix_snapshot",
        ),
        "coordinates_digest": coordinates.digest(),
    }
    for field_name, actual_digest in actual_digests.items():
        if actual_digest != getattr(provenance, field_name):
            raise DMIQFactorizedMarginalError(
                f"{field_name} differs from bound tensor provenance"
            )
    if tensor_raw_value_digest(
        prefix_runtime,
        label="source_prefix_runtime",
    ) != provenance.source_prefix_digest:
        raise DMIQFactorizedMarginalError(
            "source prefix runtime digest differs from its bound snapshot"
        )

    noop_digest_before = actual_digests["noop_hidden_digest"]
    orthogonality_error = _basis_orthogonality_error(action_basis)
    basis_orthogonal = (
        orthogonality_error <= float(config.orthogonality_tolerance)
    )
    base_norm = torch.linalg.vector_norm(noop_tail_hidden, dim=-1)
    relative_radius = float(config.max_relative_update) * torch.clamp(
        base_norm,
        min=float(config.base_norm_floor),
    )
    coefficient_cap = torch.full_like(
        relative_radius,
        float(config.max_coefficient_norm),
    )
    update_cap = torch.full_like(
        relative_radius,
        float(config.max_update_norm),
    )
    trust_radius = torch.minimum(
        torch.minimum(relative_radius, coefficient_cap),
        update_cap,
    )
    pass_a_norm = torch.linalg.vector_norm(pass_a_coefficients, dim=-1)
    predicted_norm = torch.linalg.vector_norm(predicted, dim=-1)
    pass_a_within_cap = bool(
        (
            pass_a_norm
            <= trust_radius + float(config.coefficient_recovery_tolerance)
        ).all().item()
    )
    scale = torch.clamp(
        trust_radius / torch.clamp(predicted_norm, min=float(config.epsilon)),
        max=1.0,
    )
    capped_coefficients = predicted * scale.unsqueeze(-1)
    cap_applied = bool((scale < 1.0).any().item())
    evidence_valid = all(
        (
            provenance.same_current_state_verified,
            provenance.basis_evidence_valid,
            provenance.action_evidence_valid,
            provenance.identity_feasible,
            basis_orthogonal,
            pass_a_within_cap,
        )
    )
    raw_coefficient_error = predicted - pass_a_coefficients
    raw_coefficient_mse = raw_coefficient_error.square().mean()
    if evidence_valid:
        effective_coefficients = capped_coefficients
        coefficient_loss = raw_coefficient_mse
        update = torch.einsum(
            "lstpdr,blstpr->blstpd",
            action_basis,
            effective_coefficients,
        )
        composed = noop_tail_hidden + update
    else:
        effective_coefficients = predicted * 0.0
        coefficient_loss = predicted.sum() * 0.0
        composed = noop_tail_hidden.clone()

    if _storage_identity(composed) in {
        _storage_identity(noop_tail_hidden),
        _storage_identity(action_basis),
        _storage_identity(predicted),
        _storage_identity(pass_a_coefficients),
    }:
        raise DMIQFactorizedMarginalError(
            "composed hidden unexpectedly aliases an input allocation"
        )
    actual_update = composed - noop_tail_hidden
    recovered_coefficients = torch.einsum(
        "lstpdr,blstpd->blstpr",
        action_basis,
        actual_update,
    )
    reconstructed_update = torch.einsum(
        "lstpdr,blstpr->blstpd",
        action_basis,
        recovered_coefficients,
    )
    complement_update = actual_update - reconstructed_update
    coefficient_recovery_error = recovered_coefficients - effective_coefficients

    noop_action_coefficients = torch.einsum(
        "lstpdr,blstpd->blstpr",
        action_basis,
        noop_tail_hidden,
    )
    composed_action_coefficients = torch.einsum(
        "lstpdr,blstpd->blstpr",
        action_basis,
        composed,
    )
    noop_complement = noop_tail_hidden - torch.einsum(
        "lstpdr,blstpr->blstpd",
        action_basis,
        noop_action_coefficients,
    )
    composed_complement = composed - torch.einsum(
        "lstpdr,blstpr->blstpd",
        action_basis,
        composed_action_coefficients,
    )
    complement_preservation = composed_complement - noop_complement
    complement_update_error = _max_abs(complement_update)
    complement_preservation_error = _max_abs(complement_preservation)
    coefficient_recovery_max_error = _max_abs(coefficient_recovery_error)
    complement_passed = (
        complement_update_error <= float(config.complement_tolerance)
        and complement_preservation_error <= float(config.complement_tolerance)
    )
    recovery_passed = coefficient_recovery_max_error <= float(
        config.coefficient_recovery_tolerance
    )
    if evidence_valid and not complement_passed:
        raise DMIQFactorizedMarginalError(
            "orthogonal complement preservation exceeded tolerance"
        )
    if evidence_valid and not recovery_passed:
        raise DMIQFactorizedMarginalError(
            "action coefficient recovery exceeded tolerance"
        )
    noop_unchanged = (
        tensor_raw_value_digest(noop_tail_hidden, label="noop_tail_hidden")
        == noop_digest_before
    )
    if not noop_unchanged:
        raise DMIQFactorizedMarginalError(
            "no-op co-state changed during composition"
        )
    prefix_snapshot_unchanged = (
        tensor_raw_value_digest(
            source_prefix_snapshot,
            label="source_prefix_snapshot",
        )
        == provenance.source_prefix_digest
    )
    prefix_runtime_unchanged = (
        tensor_raw_value_digest(
            prefix_runtime,
            label="source_prefix_runtime",
        )
        == provenance.source_prefix_digest
    )
    prefix_exact_after = bool(
        prefix_snapshot_unchanged
        and prefix_runtime_unchanged
        and torch.equal(source_prefix_snapshot, prefix_runtime)
    )
    if not prefix_exact_after:
        raise DMIQFactorizedMarginalError(
            "source prefix changed during composition"
        )

    action_diagnostics = ActionMarginalDiagnostics(
        basis_orthogonality_max_abs_error=orthogonality_error,
        basis_orthogonality_passed=basis_orthogonal,
        basis_evidence_valid=provenance.basis_evidence_valid,
        action_evidence_valid=provenance.action_evidence_valid,
        same_current_state_verified=provenance.same_current_state_verified,
        pass_a_within_trust_cap=pass_a_within_cap,
        predicted_coefficient_norm_max=float(
            predicted_norm.detach().max().to(device="cpu").item()
        ),
        pass_a_coefficient_norm_max=float(
            pass_a_norm.detach().max().to(device="cpu").item()
        ),
        effective_coefficient_norm_max=float(
            torch.linalg.vector_norm(effective_coefficients, dim=-1)
            .detach()
            .max()
            .to(device="cpu")
            .item()
        ),
        trust_radius_min=float(trust_radius.min().to(device="cpu").item()),
        trust_radius_max=float(trust_radius.max().to(device="cpu").item()),
        coefficient_cap_applied=cap_applied,
        coefficient_mse=float(raw_coefficient_mse.detach().to(device="cpu").item()),
        coefficient_recovery_max_abs_error=coefficient_recovery_max_error,
        coefficient_recovery_passed=recovery_passed,
        null_action_update=not evidence_valid,
    )
    identity_diagnostics = IdentityMarginalDiagnostics(
        identity_feasible=provenance.identity_feasible,
        source_prefix_exact=prefix_exact_after,
        source_prefix_no_write_passed=bool(
            prefix_snapshot_unchanged and prefix_runtime_unchanged
        ),
        noop_hidden_unchanged=noop_unchanged,
        orthogonal_complement_update_max_abs=complement_update_error,
        orthogonal_complement_preservation_max_abs=(
            complement_preservation_error
        ),
        orthogonal_complement_preserved=complement_passed,
        scalar_compensation_used=False,
    )
    local_valid = bool(evidence_valid and complement_passed and recovery_passed)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_NAME,
        "provenance_binding_digest": provenance.binding_digest,
        "coordinate_binding_digest": provenance.coordinates_digest,
        "composer_config_digest": provenance.composer_config_digest,
        "tensor_layout": {
            "noop_and_output": "B,L,S,T,P,D",
            "action_basis": "L,S,T,P,D,R",
            "coefficients": "B,L,S,T,P,R",
            "source_prefix": "B,N,D",
        },
        "factorization": {
            "formula": "h=noop_same_state_costate+generic_U_A@alpha",
            "action_basis_semantics": ACTION_BASIS_SEMANTICS,
            "noop_hidden_semantics": NOOP_HIDDEN_SEMANTICS,
            "raw_correct_source_action_residual_required": False,
            "orthogonal_complement_contract": "P_perp(h)==P_perp(h0)",
        },
        "training": {
            "trainable_tensor": "predicted_coefficients_only",
            "loss": "mean_squared_error_in_action_coefficient_space_only",
            "hidden_space_loss_present": False,
            "identity_action_scalar_compensation_present": False,
        },
        "forbidden_inputs": [
            "proposal_rgb",
            "proposal_latent",
            "paired_video",
            "segmentation_mask",
            "flow",
            "pose",
            "track",
            "trajectory",
        ],
        "source_prefix": {
            "exact": prefix_exact_after,
            "write_permitted": False,
            "runtime_digest": provenance.source_prefix_digest,
        },
        "hard_caps": {
            "max_coefficient_norm": float(config.max_coefficient_norm),
            "max_update_norm": float(config.max_update_norm),
            "max_relative_update": float(config.max_relative_update),
            "cap_applied": cap_applied,
        },
        "gates": {
            "local_composition_valid": local_valid,
            "null_update": not evidence_valid,
            "action_and_identity_diagnostics_separate": True,
            "optimizer_updates_authorized": False,
        },
        "upstream_authentication_scope": (
            "receipt_sha_values_are_bound_not_cryptographically_authenticated_"
            "by_this_pure_tensor_core"
        ),
    }
    return FactorizedMarginalResult(
        composed_tail_hidden=composed,
        effective_coefficients=effective_coefficients,
        coefficient_loss=coefficient_loss,
        action_diagnostics=action_diagnostics,
        identity_diagnostics=identity_diagnostics,
        local_composition_valid=local_valid,
        optimizer_updates_authorized=False,
        receipt=receipt,
    )


def factorized_marginal_composer_contract_receipt() -> dict[str, Any]:
    """Return the dependency-free, fail-closed public contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_NAME,
        "pure_tensor_only": True,
        "formula": "h=h0+U_A@alpha",
        "generic_source_invariant_action_basis": True,
        "same_current_state_frozen_noop_costate": True,
        "correct_source_specific_raw_action_residual_required": False,
        "coordinates": ["layer", "site", "phase", "spatial"],
        "orthogonal_complement": "P_perp(h)==P_perp(h0)",
        "loss_space": "action_coefficients_only",
        "source_prefix": "exact_snapshot_runtime_equality_and_no_write",
        "action_identity_diagnostics": "strictly_separate_no_scalar_compensation",
        "invalid_scientific_evidence": "exact_null_update",
        "forbidden_inputs": [
            "proposal_rgb",
            "proposal_latent",
            "paired_video",
            "segmentation_mask",
            "flow",
            "pose",
            "track",
            "trajectory",
        ],
        "optimizer_updates_authorized": False,
    }


__all__ = [
    "ACTION_BASIS_SEMANTICS",
    "ActionMarginalDiagnostics",
    "COORDINATE_SCHEMA",
    "DMIQFactorizedMarginalError",
    "FactorizedMarginalResult",
    "IdentityMarginalDiagnostics",
    "METHOD_NAME",
    "MarginalComposerConfig",
    "MarginalComposerProvenance",
    "MarginalCoordinateBinding",
    "NOOP_HIDDEN_SEMANTICS",
    "PROVENANCE_SCHEMA",
    "SCHEMA_VERSION",
    "bind_marginal_composer_provenance",
    "compose_factorized_action_marginal",
    "factorized_marginal_composer_contract_receipt",
    "tensor_raw_value_digest",
]
