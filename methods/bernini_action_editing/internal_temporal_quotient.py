#!/usr/bin/env python3
"""Frozen internal temporal-quotient geometry for Bernini action evidence.

This module is deliberately smaller than a trainer or an inference runtime.  It
accepts detached FP32 hidden states, constructs a fixed temporal direct-sum
representation, removes a *typed observable nuisance* subspace learned on a
discovery split, and freezes an action subspace before confirmation is scored.

The tensor contract is intentionally narrow:

* hidden states are ``[B,21,H,W,1536]`` or phase-pooled
  ``[B,21,P,1536]``;
* causal-boundary, lag-1/2/4, and four-phase terminal-hold evidence are kept;
* every hard negative remains a separately labelled residual;
* nuisance directions are column-normalized and factorized with a thin SVD;
* scientific scans require separate actor/scene/camera/appearance/seed-quality
  nuisance types and a bound signed-spatial-sketch coordinate basis;
* the action basis and null floor are frozen on disjoint discovery groups;
* confirmation never replaces the discovery action basis or its null floor.

Structurally invalid or incomplete inputs raise
:class:`InternalTemporalQuotientError`.  Scientifically valid inputs that lack
the required confirmation evidence return ``local_geometry_eligible=False``;
the core never authorizes FITQ GO.  PyTorch is a lazy dependency so
configuration and source contracts remain inspectable on a lightweight
orchestration host.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping


METHOD_NAME = "frozen-internal-temporal-quotient"
SCHEMA_VERSION = "bernini-fitq-v1"
EVIDENCE_BINDING_SCHEMA = "bernini-fitq-evidence-binding-v1"
NUISANCE_AUDIT_SCHEMA = "bernini-fitq-nuisance-audit-v1"
ABSENT_NUISANCE_AUDIT_DIGEST = hashlib.sha256(
    b"bernini-fitq-nuisance-audit-absent-v1"
).hexdigest()
PINNED_BERNINI_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
PINNED_VEOMNI_REVISION = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
PINNED_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
EXPECTED_PHASES = 21
EXPECTED_HIDDEN_SIZE = 1536
TEMPORAL_LAGS = (1, 2, 4)
TERMINAL_HOLD_WINDOW = 4
TERMINAL_HOLD_PHASES = (17, 18, 19, 20)
TEMPORAL_FEATURE_STEPS = EXPECTED_PHASES * (1 + len(TEMPORAL_LAGS)) + 1
MIN_HARD_NEGATIVES = 2
DISCOVERY_EVIDENCE_PROFILES = ("engineering_micro", "scientific")
SPATIAL_DESCRIPTOR_POLICIES = (
    "global_mean_engineering",
    "fixed_signed_sketches",
)
EVIDENCE_MODES = ("t2v", "mv2v")
EVIDENCE_CROSS_MODE_CONTRACTS = (
    "same_mode",
    "t2v_to_mv2v",
    "mv2v_to_t2v",
)
EVIDENCE_HOOK_SITES = (
    "block_input",
    "attn1_to_out_input",
    "attn2_to_out_input",
    "block_output",
    "proj_out_input",
)
SCIENTIFIC_MIN_DISCOVERY_EPISODES = 8
SCIENTIFIC_MIN_CONFIRMATION_EPISODES = 4
SCIENTIFIC_SPATIAL_SKETCH_COORDINATES = 16
SCIENTIFIC_PATCH_HEIGHT = 31
SCIENTIFIC_PATCH_WIDTH = 30
SCIENTIFIC_PATCH_TOKENS = SCIENTIFIC_PATCH_HEIGHT * SCIENTIFIC_PATCH_WIDTH
SCIENTIFIC_LATENT_GEOMETRY = (16, EXPECTED_PHASES, 62, 60)
SCIENTIFIC_PATCH_GEOMETRY = (
    EXPECTED_PHASES,
    SCIENTIFIC_PATCH_HEIGHT,
    SCIENTIFIC_PATCH_WIDTH,
    EXPECTED_HIDDEN_SIZE,
)
SCIENTIFIC_REQUIRED_NUISANCE_TYPES = (
    "actor",
    "scene",
    "camera",
    "appearance",
    "seed_quality",
)
SCIENTIFIC_REQUIRED_NEGATIVE_LABELS = (
    "noop",
    "incomplete_action",
    "reverse_action",
    "shuffled_action",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)
TEMPORAL_BLOCK_SPECS = (
    ("causal_boundary", 0, 21),
    ("lag1", 21, 42),
    ("lag2", 42, 63),
    ("lag4", 63, 84),
    ("terminal_hold", 84, 85),
)


class InternalTemporalQuotientError(RuntimeError):
    """Raised when FITQ cannot establish its frozen tensor contract."""


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - host dependent
        raise InternalTemporalQuotientError(
            "PyTorch is required for FITQ tensor operations"
        ) from error
    return torch


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise InternalTemporalQuotientError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise InternalTemporalQuotientError(
            f"{name} must be a finite number"
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise InternalTemporalQuotientError(
            f"{name} must be finite and nonnegative"
        )
    return result


def _finite_positive(name: str, value: Any) -> float:
    result = _finite_nonnegative(name, value)
    if result <= 0.0:
        raise InternalTemporalQuotientError(f"{name} must be positive")
    return result


def _validate_hex_digest(name: str, value: Any, *, length: int) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InternalTemporalQuotientError(
            f"{name} must be a lowercase {length * 4}-bit hexadecimal digest"
        )
    return value


def _validate_labels(
    labels: Any,
    *,
    name: str,
    expected_count: int | None = None,
) -> tuple[str, ...]:
    if type(labels) is not tuple:
        raise InternalTemporalQuotientError(f"{name} must be an explicit tuple")
    if expected_count is not None and len(labels) != expected_count:
        raise InternalTemporalQuotientError(
            f"{name} must contain exactly {expected_count} labels"
        )
    if not labels:
        raise InternalTemporalQuotientError(f"{name} must not be empty")
    if any(type(label) is not str or not label.strip() for label in labels):
        raise InternalTemporalQuotientError(
            f"{name} must contain nonempty strings"
        )
    if len(set(labels)) != len(labels):
        raise InternalTemporalQuotientError(f"{name} must contain unique labels")
    return labels


def _validate_group_ids(
    group_ids: Any,
    *,
    name: str,
    expected_count: int,
) -> tuple[str, ...]:
    """Validate explicit factorial-group bindings without accepting coercion."""

    return _validate_labels(
        group_ids,
        name=name,
        expected_count=expected_count,
    )


def _validate_spatial_descriptor_binding(
    *,
    policy: Any,
    sketch_id: Any,
    sketch_digest: Any,
    evidence_profile: str,
) -> None:
    if policy not in SPATIAL_DESCRIPTOR_POLICIES:
        raise InternalTemporalQuotientError(
            "spatial_descriptor_policy is not registered"
        )
    if evidence_profile == "scientific" and policy != "fixed_signed_sketches":
        raise InternalTemporalQuotientError(
            "scientific evidence requires fixed signed spatial sketches"
        )
    if policy == "global_mean_engineering":
        if sketch_id is not None or sketch_digest is not None:
            raise InternalTemporalQuotientError(
                "global-mean policy must not claim a signed-sketch binding"
            )
        return
    if (
        type(sketch_id) is not str
        or not sketch_id
        or sketch_id != sketch_id.strip()
        or any(character.isspace() for character in sketch_id)
    ):
        raise InternalTemporalQuotientError(
            "fixed signed sketches require a nonempty whitespace-free sketch ID"
        )
    _validate_hex_digest("spatial_sketch_digest", sketch_digest, length=64)


def _validate_numeric_grid(name: str, value: Any) -> tuple[float, ...]:
    if type(value) is not tuple or not value:
        raise InternalTemporalQuotientError(f"{name} must be a nonempty tuple")
    result = []
    for item in value:
        if isinstance(item, bool):
            raise InternalTemporalQuotientError(f"{name} must contain numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise InternalTemporalQuotientError(
                f"{name} must contain finite numbers"
            ) from error
        if not math.isfinite(number):
            raise InternalTemporalQuotientError(
                f"{name} must contain finite numbers"
            )
        result.append(number)
    if any(right <= left for left, right in zip(result, result[1:])):
        raise InternalTemporalQuotientError(
            f"{name} must be strictly increasing without duplicates"
        )
    return tuple(result)


def _validate_geometry(
    name: str,
    value: Any,
    *,
    length: int,
) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or len(value) != length
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise InternalTemporalQuotientError(
            f"{name} must be a length-{length} tuple of positive integers"
        )
    return value


@dataclass(frozen=True)
class EvidenceBinding:
    """Closed, exact binding for one discovery/confirmation FITQ coordinate."""

    schema_version: str
    checkpoint_tree_sha256: str
    bernini_revision: str
    veomni_revision: str
    discovery_mode: str
    confirmation_mode: str
    cross_mode_contract: str
    layer: int
    hook_site: str
    sigma_grid: tuple[float, ...]
    lambda_grid: tuple[float, ...]
    latent_geometry: tuple[int, int, int, int]
    patch_geometry: tuple[int, int, int, int]
    negative_label_set: tuple[str, ...]
    bank_digest: str
    upstream_query_receipt_digest: str
    nuisance_config_digest: str
    discovery_config_digest: str
    confirmation_config_digest: str
    discovery_digest: str
    confirmation_digest: str
    nuisance_audit_digest: str
    spatial_sketch_matrix_digest: str

    def validate(self) -> None:
        if self.schema_version != EVIDENCE_BINDING_SCHEMA:
            raise InternalTemporalQuotientError(
                "EvidenceBinding schema version is not registered"
            )
        _validate_hex_digest(
            "checkpoint_tree_sha256", self.checkpoint_tree_sha256, length=64
        )
        _validate_hex_digest("bernini_revision", self.bernini_revision, length=40)
        _validate_hex_digest("veomni_revision", self.veomni_revision, length=40)
        if (
            self.discovery_mode not in EVIDENCE_MODES
            or self.confirmation_mode not in EVIDENCE_MODES
        ):
            raise InternalTemporalQuotientError(
                "EvidenceBinding discovery/confirmation mode is invalid"
            )
        expected_cross_mode_contract = (
            "same_mode"
            if self.discovery_mode == self.confirmation_mode
            else f"{self.discovery_mode}_to_{self.confirmation_mode}"
        )
        if (
            self.cross_mode_contract not in EVIDENCE_CROSS_MODE_CONTRACTS
            or self.cross_mode_contract != expected_cross_mode_contract
        ):
            raise InternalTemporalQuotientError(
                "EvidenceBinding cross-mode contract is inconsistent"
            )
        if type(self.layer) is not int or not 0 <= self.layer < 30:
            raise InternalTemporalQuotientError(
                "EvidenceBinding layer must be an integer in [0,29]"
            )
        if self.hook_site not in EVIDENCE_HOOK_SITES:
            raise InternalTemporalQuotientError(
                "EvidenceBinding hook_site is not registered"
            )
        _validate_numeric_grid("sigma_grid", self.sigma_grid)
        _validate_numeric_grid("lambda_grid", self.lambda_grid)
        _validate_geometry("latent_geometry", self.latent_geometry, length=4)
        _validate_geometry("patch_geometry", self.patch_geometry, length=4)
        if self.latent_geometry[1] != EXPECTED_PHASES:
            raise InternalTemporalQuotientError(
                "EvidenceBinding latent geometry must contain 21 phases"
            )
        if (
            self.patch_geometry[0] != EXPECTED_PHASES
            or self.patch_geometry[-1] != EXPECTED_HIDDEN_SIZE
        ):
            raise InternalTemporalQuotientError(
                "EvidenceBinding patch geometry must bind 21 phases and dim 1536"
            )
        _validate_labels(self.negative_label_set, name="negative_label_set")
        _validate_hex_digest("bank_digest", self.bank_digest, length=64)
        _validate_hex_digest(
            "upstream_query_receipt_digest",
            self.upstream_query_receipt_digest,
            length=64,
        )
        _validate_hex_digest(
            "nuisance_config_digest", self.nuisance_config_digest, length=64
        )
        _validate_hex_digest(
            "discovery_config_digest", self.discovery_config_digest, length=64
        )
        _validate_hex_digest(
            "confirmation_config_digest",
            self.confirmation_config_digest,
            length=64,
        )
        _validate_hex_digest("discovery_digest", self.discovery_digest, length=64)
        _validate_hex_digest(
            "confirmation_digest", self.confirmation_digest, length=64
        )
        _validate_hex_digest(
            "nuisance_audit_digest", self.nuisance_audit_digest, length=64
        )
        _validate_hex_digest(
            "spatial_sketch_matrix_digest",
            self.spatial_sketch_matrix_digest,
            length=64,
        )


def _validate_nested_donor_ids(
    value: Any,
    *,
    expected_types: tuple[str, ...],
    minimum_per_type: int,
) -> tuple[tuple[str, ...], ...]:
    if type(value) is not tuple or len(value) != len(expected_types):
        raise InternalTemporalQuotientError(
            "donor_group_ids must align exactly with expected nuisance types"
        )
    result = []
    all_ids = []
    for type_name, type_ids in zip(expected_types, value):
        labels = _validate_labels(
            type_ids,
            name=f"donor_group_ids[{type_name!r}]",
        )
        if len(labels) < minimum_per_type:
            raise InternalTemporalQuotientError(
                f"nuisance type {type_name!r} requires at least "
                f"{minimum_per_type} donors"
            )
        result.append(labels)
        all_ids.extend(labels)
    if len(set(all_ids)) != len(all_ids):
        raise InternalTemporalQuotientError(
            "nuisance donor group IDs must be globally unique"
        )
    return tuple(result)


@dataclass(frozen=True)
class NuisanceAuditEvidence:
    """Externally signed leave-one-type/donor audit bound to exact nuisance data."""

    schema_version: str
    nuisance_observation_digest: str
    nuisance_basis_digest: str
    donor_group_ids: tuple[tuple[str, ...], ...]
    leave_one_donor_passed: bool
    leave_one_type_out_passed: bool
    audit_artifact_sha256: str
    signer_id: str
    signed_evidence_sha256: str
    signature_verified: bool

    def validate(self, *, expected_types: tuple[str, ...]) -> None:
        if self.schema_version != NUISANCE_AUDIT_SCHEMA:
            raise InternalTemporalQuotientError(
                "NuisanceAuditEvidence schema version is not registered"
            )
        _validate_hex_digest(
            "nuisance_observation_digest",
            self.nuisance_observation_digest,
            length=64,
        )
        _validate_hex_digest(
            "nuisance_basis_digest", self.nuisance_basis_digest, length=64
        )
        _validate_nested_donor_ids(
            self.donor_group_ids,
            expected_types=expected_types,
            minimum_per_type=2,
        )
        if type(self.leave_one_donor_passed) is not bool or type(
            self.leave_one_type_out_passed
        ) is not bool:
            raise InternalTemporalQuotientError(
                "nuisance leave-one audit decisions must be booleans"
            )
        _validate_hex_digest(
            "audit_artifact_sha256", self.audit_artifact_sha256, length=64
        )
        if (
            type(self.signer_id) is not str
            or not self.signer_id
            or self.signer_id != self.signer_id.strip()
        ):
            raise InternalTemporalQuotientError(
                "nuisance audit signer_id must be nonempty and trimmed"
            )
        _validate_hex_digest(
            "signed_evidence_sha256", self.signed_evidence_sha256, length=64
        )
        if type(self.signature_verified) is not bool:
            raise InternalTemporalQuotientError(
                "signature_verified must be an explicit boolean"
            )


def canonical_nuisance_audit_evidence_digest(
    evidence: NuisanceAuditEvidence | None,
    *,
    expected_types: tuple[str, ...],
) -> str:
    """Bind an exact external audit attestation, or its explicit absence."""

    if evidence is None:
        return ABSENT_NUISANCE_AUDIT_DIGEST
    if not isinstance(evidence, NuisanceAuditEvidence):
        raise InternalTemporalQuotientError(
            "nuisance audit digest requires NuisanceAuditEvidence"
        )
    evidence.validate(expected_types=expected_types)
    donor_digest_parts = ["ordered_nuisance_donor_ids_v1"]
    for type_name, donor_ids in zip(expected_types, evidence.donor_group_ids):
        donor_digest_parts.extend((type_name, *donor_ids))
    return _composite_digest(
        (
            "fitq-exact-nuisance-audit-attestation-v1",
            evidence.schema_version,
            evidence.nuisance_observation_digest,
            evidence.nuisance_basis_digest,
            _composite_digest(tuple(donor_digest_parts)),
            str(evidence.leave_one_donor_passed),
            str(evidence.leave_one_type_out_passed),
            evidence.audit_artifact_sha256,
            evidence.signer_id,
            evidence.signed_evidence_sha256,
            str(evidence.signature_verified),
        )
    )


def _validate_tensor(
    value: Any,
    *,
    name: str,
    ranks: tuple[int, ...],
    last_dimension: int | None = EXPECTED_HIDDEN_SIZE,
) -> Any:
    torch = _require_torch()
    if not isinstance(value, torch.Tensor):
        raise InternalTemporalQuotientError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise InternalTemporalQuotientError(f"{name} must use a dense layout")
    if value.is_meta:
        raise InternalTemporalQuotientError(f"{name} must not be a meta tensor")
    if value.dtype != torch.float32:
        raise InternalTemporalQuotientError(f"{name} must be float32")
    if value.ndim not in ranks:
        expected = " or ".join(str(rank) for rank in ranks)
        raise InternalTemporalQuotientError(
            f"{name} must have rank {expected}, got {value.ndim}"
        )
    if any(int(size) <= 0 for size in value.shape):
        raise InternalTemporalQuotientError(
            f"{name} must have strictly positive dimensions"
        )
    if last_dimension is not None and int(value.shape[-1]) != last_dimension:
        raise InternalTemporalQuotientError(
            f"{name} hidden dimension must be {last_dimension}"
        )
    if value.requires_grad or value.grad_fn is not None:
        raise InternalTemporalQuotientError(
            f"{name} must be detached before FITQ analysis"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise InternalTemporalQuotientError(f"{name} must be finite")
    return value


def canonical_tensor_raw_value_digest(value: Any, *, name: str = "tensor") -> str:
    """Hash detached finite FP32 values with canonical shape and little-endian bytes."""

    candidate = _validate_tensor(
        value,
        name=name,
        ranks=tuple(range(1, 9)),
        last_dimension=None,
    )
    cpu = candidate.to(device="cpu").contiguous()
    array = cpu.numpy().astype("<f4", copy=False)
    header = (
        "fitq-canonical-fp32-little-endian-v1|shape="
        + ",".join(str(int(size)) for size in candidate.shape)
        + "|"
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    raw_bytes = memoryview(array).cast("B")
    chunk_size = 8 * 1024 * 1024
    for offset in range(0, len(raw_bytes), chunk_size):
        digest.update(raw_bytes[offset : offset + chunk_size])
    return digest.hexdigest()


def _composite_digest(parts: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"fitq-composite-digest-v1|")
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _ordered_string_tuple_digest(domain: str, values: tuple[str, ...]) -> str:
    return _composite_digest((domain, *values))


def canonical_fitq_split_content_digest(
    positive_features: Any,
    semantic_negative_features: Any,
    null_features: Any,
    *,
    split: str,
    group_ids: tuple[str, ...],
    semantic_negative_labels: tuple[str, ...],
    nuisance_observation_digest: str,
    nuisance_basis_digest: str,
    spatial_sketch_matrix_digest: str | None,
    evidence_binding: EvidenceBinding,
) -> str:
    """Hash one exact split without trusting caller-supplied content metadata."""

    if split not in ("discovery", "confirmation"):
        raise InternalTemporalQuotientError(
            "FITQ content digest split must be discovery or confirmation"
        )
    if not isinstance(evidence_binding, EvidenceBinding):
        raise InternalTemporalQuotientError(
            "FITQ content digest requires an EvidenceBinding"
        )
    evidence_binding.validate()
    labels = _validate_labels(
        semantic_negative_labels,
        name=f"{split}_semantic_negative_labels",
    )
    groups = _validate_labels(group_ids, name=f"{split}_group_ids")
    _validate_hex_digest(
        "nuisance_observation_digest", nuisance_observation_digest, length=64
    )
    _validate_hex_digest("nuisance_basis_digest", nuisance_basis_digest, length=64)
    sketch_token = "none"
    if spatial_sketch_matrix_digest is not None:
        sketch_token = _validate_hex_digest(
            "spatial_sketch_matrix_digest",
            spatial_sketch_matrix_digest,
            length=64,
        )
    if labels != evidence_binding.negative_label_set:
        raise InternalTemporalQuotientError(
            "FITQ content digest labels differ from EvidenceBinding"
        )
    if sketch_token != evidence_binding.spatial_sketch_matrix_digest:
        raise InternalTemporalQuotientError(
            "FITQ content digest sketch differs from EvidenceBinding"
        )
    binding_parts = (
        EVIDENCE_BINDING_SCHEMA,
        evidence_binding.checkpoint_tree_sha256,
        evidence_binding.bernini_revision,
        evidence_binding.veomni_revision,
        evidence_binding.discovery_mode,
        evidence_binding.confirmation_mode,
        evidence_binding.cross_mode_contract,
        str(evidence_binding.layer),
        evidence_binding.hook_site,
        _ordered_string_tuple_digest(
            "sigma_grid_float_hex_v1",
            tuple(float(item).hex() for item in evidence_binding.sigma_grid),
        ),
        _ordered_string_tuple_digest(
            "lambda_grid_float_hex_v1",
            tuple(float(item).hex() for item in evidence_binding.lambda_grid),
        ),
        ",".join(str(item) for item in evidence_binding.latent_geometry),
        ",".join(str(item) for item in evidence_binding.patch_geometry),
        evidence_binding.bank_digest,
        evidence_binding.upstream_query_receipt_digest,
        evidence_binding.nuisance_config_digest,
        evidence_binding.discovery_config_digest,
        evidence_binding.confirmation_config_digest,
        evidence_binding.nuisance_audit_digest,
        sketch_token,
        nuisance_observation_digest,
        nuisance_basis_digest,
        _ordered_string_tuple_digest("ordered_negative_labels_v1", labels),
        _ordered_string_tuple_digest("ordered_factorial_group_ids_v1", groups),
    )
    if split == "confirmation":
        binding_parts = (*binding_parts, evidence_binding.discovery_digest)
    return _composite_digest(
        (
            "fitq-exact-split-content-v1",
            split,
            *binding_parts,
            canonical_tensor_raw_value_digest(
                positive_features,
                name=f"{split}_positive_features",
            ),
            canonical_tensor_raw_value_digest(
                semantic_negative_features,
                name=f"{split}_semantic_negative_features",
            ),
            canonical_tensor_raw_value_digest(
                null_features,
                name=f"{split}_null_features",
            ),
        )
    )


def _validate_spatial_sketch_matrix(
    matrix: Any,
    *,
    policy: str,
    expected_coordinates: int,
    expected_source_positions: int | None,
    expected_digest: str | None,
) -> tuple[str | None, Any | None]:
    torch = _require_torch()
    if policy == "global_mean_engineering":
        if matrix is not None:
            raise InternalTemporalQuotientError(
                "global-mean descriptors must not accept a sketch matrix"
            )
        return None, None
    candidate = _validate_tensor(
        matrix,
        name="spatial_sketch_matrix",
        ranks=(2,),
        last_dimension=None,
    )
    if int(candidate.shape[0]) != expected_coordinates:
        raise InternalTemporalQuotientError(
            "spatial sketch matrix row count differs from feature coordinates"
        )
    if (
        expected_source_positions is not None
        and int(candidate.shape[1]) != expected_source_positions
    ):
        raise InternalTemporalQuotientError(
            "spatial sketch matrix column count differs from the registered "
            "source patch grid"
        )
    if int(candidate.shape[1]) < expected_coordinates:
        raise InternalTemporalQuotientError(
            "spatial sketch matrix cannot support independent signed coordinates"
        )
    singular_values = torch.linalg.svdvals(candidate)
    cutoff = singular_values[0] * 1.0e-6
    if int((singular_values > cutoff).sum().item()) != expected_coordinates:
        raise InternalTemporalQuotientError(
            "spatial sketch matrix rows must be full rank"
        )
    digest = canonical_tensor_raw_value_digest(
        candidate,
        name="spatial_sketch_matrix",
    )
    if expected_digest is None or digest != expected_digest:
        raise InternalTemporalQuotientError(
            "spatial sketch matrix raw-value digest differs from registration"
        )
    return digest, candidate.clone()


def _same_storage_contract(reference: Any, candidate: Any, *, name: str) -> None:
    if candidate.dtype != reference.dtype:
        raise InternalTemporalQuotientError(f"{name} dtype does not match")
    if candidate.device != reference.device:
        raise InternalTemporalQuotientError(f"{name} device does not match")


def _canonical_hidden(hidden: Any, *, name: str) -> tuple[Any, tuple[int, ...]]:
    value = _validate_tensor(hidden, name=name, ranks=(4, 5))
    if int(value.shape[1]) != EXPECTED_PHASES:
        raise InternalTemporalQuotientError(
            f"{name} must contain exactly {EXPECTED_PHASES} phases"
        )
    if value.ndim == 5:
        batch, phases, height, width, channels = map(int, value.shape)
        canonical = value.reshape(batch, phases, height * width, channels)
        geometry = (height, width)
    else:
        batch, phases, positions, channels = map(int, value.shape)
        canonical = value
        geometry = (positions,)
    return canonical, geometry


@dataclass(frozen=True)
class TemporalBundle:
    """A fixed direct-sum temporal representation with exact causal zeros."""

    canonical_hidden: Any
    causal_boundary: Any
    lag1: Any
    lag2: Any
    lag4: Any
    terminal_hold: Any
    features: Any
    source_geometry: tuple[int, ...]


def _lag_difference(causal: Any, lag: int) -> Any:
    torch = _require_torch()
    leading = torch.zeros_like(causal[:, :lag])
    valid = causal[:, lag:] - causal[:, :-lag]
    return torch.cat((leading, valid), dim=1)


def _temporal_bundle_from_canonical(
    canonical: Any,
    *,
    source_geometry: tuple[int, ...],
) -> TemporalBundle:
    torch = _require_torch()
    causal = canonical - canonical[:, :1]
    # Assignment makes the causal boundary exact rather than merely subject to
    # x-x roundoff assumptions on unusual backends.
    causal = torch.cat((torch.zeros_like(causal[:, :1]), causal[:, 1:]), dim=1)
    lag1 = _lag_difference(causal, 1)
    lag2 = _lag_difference(causal, 2)
    lag4 = _lag_difference(causal, 4)
    if TERMINAL_HOLD_PHASES != tuple(
        range(EXPECTED_PHASES - TERMINAL_HOLD_WINDOW, EXPECTED_PHASES)
    ):
        raise InternalTemporalQuotientError(
            "terminal-hold phase signature is internally inconsistent"
        )
    terminal_hold = causal[:, TERMINAL_HOLD_PHASES[0] :].mean(dim=1)
    features = torch.cat(
        (causal, lag1, lag2, lag4, terminal_hold.unsqueeze(1)), dim=1
    )
    if int(features.shape[1]) != TEMPORAL_FEATURE_STEPS:
        raise InternalTemporalQuotientError(
            "internal temporal feature schema was not preserved"
        )
    if not bool(torch.isfinite(features).all().item()):
        raise InternalTemporalQuotientError(
            "temporal quotient produced a non-finite value"
        )
    return TemporalBundle(
        canonical_hidden=canonical,
        causal_boundary=causal,
        lag1=lag1,
        lag2=lag2,
        lag4=lag4,
        terminal_hold=terminal_hold,
        features=features,
        source_geometry=source_geometry,
    )


def build_temporal_bundle(hidden: Any) -> TemporalBundle:
    """Validate and transform one hidden-state batch into the FITQ bundle."""

    canonical, geometry = _canonical_hidden(hidden, name="hidden")
    return _temporal_bundle_from_canonical(canonical, source_geometry=geometry)


@dataclass(frozen=True)
class HardNegativeTemporalResiduals:
    """Independently labelled action-minus-negative temporal residuals."""

    negative_labels: tuple[str, ...]
    raw_residuals: Any
    causal_boundary: Any
    lag1: Any
    lag2: Any
    lag4: Any
    terminal_hold: Any
    features: Any
    source_geometry: tuple[int, ...]


def compute_hard_negative_temporal_residuals(
    action_hidden: Any,
    hard_negative_hidden: Any,
    *,
    negative_labels: tuple[str, ...],
) -> HardNegativeTemporalResiduals:
    """Return one temporal residual per negative without reducing ``K``."""

    action, geometry = _canonical_hidden(action_hidden, name="action_hidden")
    negatives = _validate_tensor(
        hard_negative_hidden,
        name="hard_negative_hidden",
        ranks=(5, 6),
    )
    if int(negatives.shape[2]) != EXPECTED_PHASES:
        raise InternalTemporalQuotientError(
            "hard_negative_hidden must contain exactly 21 phases"
        )
    batch, count = int(negatives.shape[0]), int(negatives.shape[1])
    if count < MIN_HARD_NEGATIVES:
        raise InternalTemporalQuotientError(
            f"at least {MIN_HARD_NEGATIVES} hard negatives are required"
        )
    labels = _validate_labels(
        negative_labels,
        name="negative_labels",
        expected_count=count,
    )
    if negatives.ndim == 6:
        _, _, _, height, width, channels = map(int, negatives.shape)
        negative_geometry = (height, width)
        canonical_negatives = negatives.reshape(
            batch, count, EXPECTED_PHASES, height * width, channels
        )
    else:
        _, _, _, positions, _ = map(int, negatives.shape)
        negative_geometry = (positions,)
        canonical_negatives = negatives
    if batch != int(action.shape[0]):
        raise InternalTemporalQuotientError(
            "hard-negative batch size does not match action batch size"
        )
    if negative_geometry != geometry:
        raise InternalTemporalQuotientError(
            "hard-negative geometry does not match action geometry"
        )
    _same_storage_contract(action, canonical_negatives, name="hard_negative_hidden")

    residual = action[:, None] - canonical_negatives
    positions = int(action.shape[2])
    flattened = residual.reshape(
        batch * count, EXPECTED_PHASES, positions, EXPECTED_HIDDEN_SIZE
    )
    bundle = _temporal_bundle_from_canonical(
        flattened,
        source_geometry=geometry,
    )

    def restore(value: Any, trailing_phase: bool = True) -> Any:
        if trailing_phase:
            return value.reshape(batch, count, *value.shape[1:])
        return value.reshape(batch, count, *value.shape[1:])

    return HardNegativeTemporalResiduals(
        negative_labels=labels,
        raw_residuals=residual,
        causal_boundary=restore(bundle.causal_boundary),
        lag1=restore(bundle.lag1),
        lag2=restore(bundle.lag2),
        lag4=restore(bundle.lag4),
        terminal_hold=restore(bundle.terminal_hold, trailing_phase=False),
        features=restore(bundle.features),
        source_geometry=geometry,
    )


@dataclass(frozen=True)
class NuisanceBasisConfig:
    """Pre-registered policy for a typed observable nuisance factorization."""

    expected_types: tuple[str, ...]
    evidence_profile: str = "engineering_micro"
    donor_group_ids: tuple[tuple[str, ...], ...] = ()
    min_leave_one_donor_cosine: float = 0.80
    rank_rtol: float = 1.0e-5
    max_condition_number: float = 1.0e6
    epsilon: float = 1.0e-8
    orthogonality_tolerance: float = 2.0e-4

    def validate(self) -> None:
        _validate_labels(self.expected_types, name="expected_types")
        if self.evidence_profile not in DISCOVERY_EVIDENCE_PROFILES:
            raise InternalTemporalQuotientError(
                "nuisance evidence_profile is not registered"
            )
        if self.evidence_profile == "scientific":
            if self.expected_types != SCIENTIFIC_REQUIRED_NUISANCE_TYPES:
                raise InternalTemporalQuotientError(
                    "scientific nuisance config requires the exact five types"
                )
            _validate_nested_donor_ids(
                self.donor_group_ids,
                expected_types=self.expected_types,
                minimum_per_type=2,
            )
        elif self.donor_group_ids:
            _validate_nested_donor_ids(
                self.donor_group_ids,
                expected_types=self.expected_types,
                minimum_per_type=1,
            )
        donor_cosine = _finite_nonnegative(
            "min_leave_one_donor_cosine", self.min_leave_one_donor_cosine
        )
        if donor_cosine > 1.0:
            raise InternalTemporalQuotientError(
                "min_leave_one_donor_cosine must not exceed one"
            )
        rank_rtol = _finite_positive("rank_rtol", self.rank_rtol)
        if rank_rtol >= 1.0:
            raise InternalTemporalQuotientError("rank_rtol must be below one")
        if _finite_positive(
            "max_condition_number", self.max_condition_number
        ) <= 1.0:
            raise InternalTemporalQuotientError(
                "max_condition_number must be greater than one"
            )
        _finite_positive("epsilon", self.epsilon)
        tolerance = _finite_positive(
            "orthogonality_tolerance", self.orthogonality_tolerance
        )
        if tolerance >= 1.0:
            raise InternalTemporalQuotientError(
                "orthogonality_tolerance must be below one"
            )


@dataclass(frozen=True)
class NuisanceLeaveOneOutDiagnostics:
    """Donor support plus an explicit marker for external type-out auditing."""

    per_type_min_leave_one_donor_cosine: Any
    per_type_leave_one_donor_passed: tuple[bool, ...]
    leave_one_donor_gate_passed: bool
    leave_one_type_out_rank: tuple[int, ...]
    leave_one_type_out_external_audit_required: bool


@dataclass(frozen=True)
class TypedObservableNuisanceBasis:
    """Joint orthobasis derived from explicitly typed observable residuals."""

    type_names: tuple[str, ...]
    raw_observation_snapshots: tuple[Any, ...]
    type_counts: tuple[int, ...]
    discarded_zero_counts: tuple[int, ...]
    normalized_columns: Any
    singular_values: Any
    basis: Any
    rank: int
    condition_number: Any
    donor_group_ids: tuple[tuple[str, ...], ...]
    observation_digest: str
    normalized_columns_digest: str
    basis_digest: str
    leave_one_out_diagnostics: NuisanceLeaveOneOutDiagnostics
    config: NuisanceBasisConfig


def build_typed_observable_nuisance_basis(
    typed_directions: Mapping[str, Any],
    *,
    config: NuisanceBasisConfig,
) -> TypedObservableNuisanceBasis:
    """Fit a joint nuisance basis from typed, column-normalized observations."""

    torch = _require_torch()
    if not isinstance(config, NuisanceBasisConfig):
        raise InternalTemporalQuotientError(
            "config must be a NuisanceBasisConfig"
        )
    config.validate()
    if not isinstance(typed_directions, Mapping):
        raise InternalTemporalQuotientError("typed_directions must be a mapping")
    keys = tuple(typed_directions.keys())
    if any(type(key) is not str for key in keys):
        raise InternalTemporalQuotientError(
            "typed_directions keys must be strings"
        )
    if set(keys) != set(config.expected_types) or len(keys) != len(
        config.expected_types
    ):
        raise InternalTemporalQuotientError(
            "typed_directions must contain exactly the pre-registered types"
        )

    retained_rows = []
    raw_observation_snapshots = []
    per_type_rows = []
    type_counts = []
    discarded_counts = []
    per_type_donor_cosines = []
    per_type_donor_passed = []
    observation_parts = []
    donor_group_ids = (
        config.donor_group_ids
        if config.donor_group_ids
        else tuple(() for _ in config.expected_types)
    )
    reference = None
    for type_index, type_name in enumerate(config.expected_types):
        value = _validate_tensor(
            typed_directions[type_name],
            name=f"typed_directions[{type_name!r}]",
            ranks=tuple(range(2, 8)),
        )
        if reference is None:
            reference = value
        else:
            _same_storage_contract(
                reference,
                value,
                name=f"typed_directions[{type_name!r}]",
            )
        observation_parts.extend(
            (
                type_name,
                ",".join(donor_group_ids[type_index]),
                canonical_tensor_raw_value_digest(
                    value,
                    name=f"typed_directions[{type_name!r}]",
                ),
            )
        )
        raw_observation_snapshots.append(value.clone())
        rows = value.reshape(-1, EXPECTED_HIDDEN_SIZE)
        norms = torch.linalg.vector_norm(rows, dim=1)
        keep = norms > float(config.epsilon)
        count = int(keep.sum().item())
        discarded = int((~keep).sum().item())
        if count == 0:
            raise InternalTemporalQuotientError(
                f"nuisance type {type_name!r} has no observable direction"
            )
        if config.evidence_profile == "scientific":
            if value.ndim != 2 or int(value.shape[0]) != len(
                donor_group_ids[type_index]
            ):
                raise InternalTemporalQuotientError(
                    f"scientific nuisance type {type_name!r} must provide one row "
                    "per registered donor"
                )
            if discarded != 0:
                raise InternalTemporalQuotientError(
                    f"scientific nuisance type {type_name!r} contains a zero donor"
                )
        normalized = rows[keep] / norms[keep, None]
        retained_rows.append(normalized)
        per_type_rows.append(normalized)
        type_counts.append(count)
        discarded_counts.append(discarded)
        if count >= 2:
            cosine = torch.abs(normalized @ normalized.transpose(0, 1))
            cosine.fill_diagonal_(-1.0)
            best_other = cosine.max(dim=1).values
            minimum_cosine = best_other.min()
            donor_passed = bool(
                (
                    best_other >= float(config.min_leave_one_donor_cosine)
                ).all().item()
            )
        else:
            minimum_cosine = torch.zeros(
                (), dtype=normalized.dtype, device=normalized.device
            )
            donor_passed = config.evidence_profile != "scientific"
        per_type_donor_cosines.append(minimum_cosine)
        per_type_donor_passed.append(donor_passed)

    normalized_columns = torch.cat(retained_rows, dim=0).transpose(0, 1).contiguous()
    left, singular_values, _ = torch.linalg.svd(
        normalized_columns,
        full_matrices=False,
    )
    if singular_values.numel() == 0 or float(singular_values[0].item()) <= 0.0:
        raise InternalTemporalQuotientError("nuisance SVD is rank deficient")
    cutoff = singular_values[0] * float(config.rank_rtol)
    active = singular_values > cutoff
    rank = int(active.sum().item())
    if rank <= 0 or rank >= EXPECTED_HIDDEN_SIZE:
        raise InternalTemporalQuotientError(
            "nuisance rank must leave a nonempty complementary space"
        )
    retained_singular = singular_values[:rank]
    condition_number = retained_singular[0] / retained_singular[-1]
    if not bool(torch.isfinite(condition_number).item()) or float(
        condition_number.item()
    ) > float(config.max_condition_number):
        raise InternalTemporalQuotientError(
            "nuisance observations exceed the registered condition limit"
        )
    basis = left[:, :rank].contiguous()
    gram = basis.transpose(0, 1) @ basis
    eye = torch.eye(rank, dtype=basis.dtype, device=basis.device)
    if float((gram - eye).abs().max().item()) > float(
        config.orthogonality_tolerance
    ):
        raise InternalTemporalQuotientError(
            "nuisance SVD did not produce an orthonormal basis"
        )
    leave_one_type_out_rank = []
    for omitted_index in range(len(per_type_rows)):
        remaining = [
            rows
            for index, rows in enumerate(per_type_rows)
            if index != omitted_index
        ]
        if not remaining:
            leave_one_type_out_rank.append(0)
            continue
        type_out_columns = torch.cat(remaining, dim=0).transpose(0, 1)
        type_out_singular = torch.linalg.svdvals(type_out_columns)
        type_out_cutoff = type_out_singular[0] * float(config.rank_rtol)
        leave_one_type_out_rank.append(
            int((type_out_singular > type_out_cutoff).sum().item())
        )
    observation_digest = _composite_digest(tuple(observation_parts))
    normalized_columns_digest = canonical_tensor_raw_value_digest(
        normalized_columns,
        name="nuisance_normalized_columns",
    )
    basis_digest = canonical_tensor_raw_value_digest(
        basis,
        name="nuisance_basis",
    )
    leave_one_out = NuisanceLeaveOneOutDiagnostics(
        per_type_min_leave_one_donor_cosine=torch.stack(per_type_donor_cosines),
        per_type_leave_one_donor_passed=tuple(per_type_donor_passed),
        leave_one_donor_gate_passed=all(per_type_donor_passed),
        leave_one_type_out_rank=tuple(leave_one_type_out_rank),
        leave_one_type_out_external_audit_required=True,
    )
    return TypedObservableNuisanceBasis(
        type_names=config.expected_types,
        raw_observation_snapshots=tuple(raw_observation_snapshots),
        type_counts=tuple(type_counts),
        discarded_zero_counts=tuple(discarded_counts),
        normalized_columns=normalized_columns,
        singular_values=singular_values,
        basis=basis,
        rank=rank,
        condition_number=condition_number,
        donor_group_ids=donor_group_ids,
        observation_digest=observation_digest,
        normalized_columns_digest=normalized_columns_digest,
        basis_digest=basis_digest,
        leave_one_out_diagnostics=leave_one_out,
        config=config,
    )


def _validate_nuisance_basis(basis: Any) -> TypedObservableNuisanceBasis:
    torch = _require_torch()
    if not isinstance(basis, TypedObservableNuisanceBasis):
        raise InternalTemporalQuotientError(
            "nuisance_basis must be a TypedObservableNuisanceBasis"
        )
    basis.config.validate()
    if basis.type_names != basis.config.expected_types:
        raise InternalTemporalQuotientError(
            "nuisance basis type binding does not match its registered config"
        )
    expected_donor_ids = (
        basis.config.donor_group_ids
        if basis.config.donor_group_ids
        else tuple(() for _ in basis.type_names)
    )
    if basis.donor_group_ids != expected_donor_ids:
        raise InternalTemporalQuotientError(
            "nuisance donor-group binding differs from its config"
        )
    if (
        type(basis.type_counts) is not tuple
        or len(basis.type_counts) != len(basis.type_names)
        or any(type(item) is not int or item <= 0 for item in basis.type_counts)
        or type(basis.discarded_zero_counts) is not tuple
        or len(basis.discarded_zero_counts) != len(basis.type_names)
        or any(
            type(item) is not int or item < 0
            for item in basis.discarded_zero_counts
        )
    ):
        raise InternalTemporalQuotientError(
            "nuisance type counts are malformed"
        )
    if basis.config.evidence_profile == "scientific" and any(
        count != len(donor_ids) or discarded != 0
        for count, discarded, donor_ids in zip(
            basis.type_counts,
            basis.discarded_zero_counts,
            basis.donor_group_ids,
        )
    ):
        raise InternalTemporalQuotientError(
            "scientific nuisance donors do not match retained observations"
        )
    if (
        type(basis.raw_observation_snapshots) is not tuple
        or len(basis.raw_observation_snapshots) != len(basis.type_names)
    ):
        raise InternalTemporalQuotientError(
            "nuisance basis lacks exact raw observation snapshots"
        )
    reconstructed_rows = []
    reconstructed_observation_parts = []
    snapshot_reference = None
    for type_index, (type_name, snapshot) in enumerate(
        zip(basis.type_names, basis.raw_observation_snapshots)
    ):
        snapshot = _validate_tensor(
            snapshot,
            name=f"nuisance_basis.raw_observations[{type_name!r}]",
            ranks=tuple(range(2, 8)),
        )
        if snapshot_reference is None:
            snapshot_reference = snapshot
        else:
            _same_storage_contract(
                snapshot_reference,
                snapshot,
                name=f"nuisance_basis.raw_observations[{type_name!r}]",
            )
        reconstructed_observation_parts.extend(
            (
                type_name,
                ",".join(basis.donor_group_ids[type_index]),
                canonical_tensor_raw_value_digest(
                    snapshot,
                    name=f"nuisance_basis.raw_observations[{type_name!r}]",
                ),
            )
        )
        rows = snapshot.reshape(-1, EXPECTED_HIDDEN_SIZE)
        norms = torch.linalg.vector_norm(rows, dim=1)
        keep = norms > float(basis.config.epsilon)
        if (
            int(keep.sum().item()) != basis.type_counts[type_index]
            or int((~keep).sum().item())
            != basis.discarded_zero_counts[type_index]
        ):
            raise InternalTemporalQuotientError(
                "nuisance raw observations differ from retained type counts"
            )
        reconstructed_rows.append(rows[keep] / norms[keep, None])
    reconstructed_observation_digest = _composite_digest(
        tuple(reconstructed_observation_parts)
    )
    reconstructed_normalized_columns = (
        torch.cat(reconstructed_rows, dim=0).transpose(0, 1).contiguous()
    )
    _validate_hex_digest(
        "nuisance observation_digest", basis.observation_digest, length=64
    )
    if reconstructed_observation_digest != basis.observation_digest:
        raise InternalTemporalQuotientError(
            "nuisance raw observation fingerprint changed after factorization"
        )
    value = _validate_tensor(
        basis.basis,
        name="nuisance_basis.basis",
        ranks=(2,),
        last_dimension=basis.rank,
    )
    if int(value.shape[0]) != EXPECTED_HIDDEN_SIZE:
        raise InternalTemporalQuotientError(
            "nuisance basis has the wrong ambient dimension"
        )
    if not 1 <= basis.rank < EXPECTED_HIDDEN_SIZE:
        raise InternalTemporalQuotientError("nuisance basis rank is invalid")
    if canonical_tensor_raw_value_digest(
        basis.normalized_columns,
        name="nuisance_basis.normalized_columns",
    ) != basis.normalized_columns_digest:
        raise InternalTemporalQuotientError(
            "nuisance normalized-column raw values changed after factorization"
        )
    if not torch.equal(
        reconstructed_normalized_columns,
        basis.normalized_columns,
    ):
        raise InternalTemporalQuotientError(
            "nuisance normalized columns differ from raw observation snapshots"
        )
    if tuple(basis.normalized_columns.shape) != (
        EXPECTED_HIDDEN_SIZE,
        sum(basis.type_counts),
    ):
        raise InternalTemporalQuotientError(
            "nuisance normalized columns do not match retained type counts"
        )
    if canonical_tensor_raw_value_digest(
        value,
        name="nuisance_basis.basis",
    ) != basis.basis_digest:
        raise InternalTemporalQuotientError(
            "nuisance basis raw values changed after factorization"
        )
    expected_left, expected_singular_values, _ = torch.linalg.svd(
        reconstructed_normalized_columns,
        full_matrices=False,
    )
    expected_cutoff = expected_singular_values[0] * float(basis.config.rank_rtol)
    expected_rank = int((expected_singular_values > expected_cutoff).sum().item())
    if expected_rank != basis.rank or not torch.allclose(
        basis.singular_values,
        expected_singular_values,
        atol=float(basis.config.orthogonality_tolerance),
        rtol=float(basis.config.orthogonality_tolerance),
    ):
        raise InternalTemporalQuotientError(
            "nuisance factorization differs from raw observation snapshots"
        )
    expected_basis = expected_left[:, :expected_rank]
    principal_cosines = torch.linalg.svdvals(value.transpose(0, 1) @ expected_basis)
    if bool(
        (
            torch.abs(principal_cosines - 1.0)
            > float(basis.config.orthogonality_tolerance)
        ).any().item()
    ):
        raise InternalTemporalQuotientError(
            "nuisance basis span differs from raw observation factorization"
        )
    if not isinstance(
        basis.leave_one_out_diagnostics,
        NuisanceLeaveOneOutDiagnostics,
    ):
        raise InternalTemporalQuotientError(
            "nuisance basis lacks leave-one-out diagnostics"
        )
    diagnostics = basis.leave_one_out_diagnostics
    per_type_cosine = _validate_tensor(
        diagnostics.per_type_min_leave_one_donor_cosine,
        name="nuisance_basis.leave_one_out.per_type_min_cosine",
        ranks=(1,),
        last_dimension=len(basis.type_names),
    )
    recomputed_donor_cosines = []
    recomputed_donor_passes = []
    for normalized in reconstructed_rows:
        count = int(normalized.shape[0])
        if count >= 2:
            cosine = torch.abs(normalized @ normalized.transpose(0, 1))
            cosine.fill_diagonal_(-1.0)
            best_other = cosine.max(dim=1).values
            minimum_cosine = best_other.min()
            donor_passed = bool(
                (
                    best_other
                    >= float(basis.config.min_leave_one_donor_cosine)
                ).all().item()
            )
        else:
            minimum_cosine = torch.zeros(
                (), dtype=normalized.dtype, device=normalized.device
            )
            donor_passed = basis.config.evidence_profile != "scientific"
        recomputed_donor_cosines.append(minimum_cosine)
        recomputed_donor_passes.append(donor_passed)
    recomputed_per_type_cosine = torch.stack(recomputed_donor_cosines)
    if not torch.equal(per_type_cosine, recomputed_per_type_cosine):
        raise InternalTemporalQuotientError(
            "nuisance leave-one-donor diagnostics differ from raw observations"
        )
    if (
        type(diagnostics.per_type_leave_one_donor_passed) is not tuple
        or len(diagnostics.per_type_leave_one_donor_passed)
        != len(basis.type_names)
        or any(
            type(item) is not bool
            for item in diagnostics.per_type_leave_one_donor_passed
        )
        or type(diagnostics.leave_one_donor_gate_passed) is not bool
        or diagnostics.leave_one_donor_gate_passed
        != all(diagnostics.per_type_leave_one_donor_passed)
    ):
        raise InternalTemporalQuotientError(
            "nuisance leave-one-donor diagnostics are inconsistent"
        )
    expected_donor_passes = tuple(
        float(value.item()) >= float(basis.config.min_leave_one_donor_cosine)
        if basis.type_counts[index] >= 2
        else basis.config.evidence_profile != "scientific"
        for index, value in enumerate(per_type_cosine)
    )
    if diagnostics.per_type_leave_one_donor_passed != expected_donor_passes:
        raise InternalTemporalQuotientError(
            "nuisance leave-one-donor decisions differ from their diagnostics"
        )
    if diagnostics.per_type_leave_one_donor_passed != tuple(
        recomputed_donor_passes
    ):
        raise InternalTemporalQuotientError(
            "nuisance leave-one-donor decisions differ from raw observations"
        )
    if (
        type(diagnostics.leave_one_type_out_rank) is not tuple
        or len(diagnostics.leave_one_type_out_rank) != len(basis.type_names)
        or any(
            type(item) is not int or item < 0
            for item in diagnostics.leave_one_type_out_rank
        )
        or diagnostics.leave_one_type_out_external_audit_required is not True
    ):
        raise InternalTemporalQuotientError(
            "nuisance leave-one-type-out diagnostics are incomplete"
        )
    recomputed_type_out_ranks = []
    for omitted_index in range(len(reconstructed_rows)):
        remaining = [
            rows
            for index, rows in enumerate(reconstructed_rows)
            if index != omitted_index
        ]
        if not remaining:
            recomputed_type_out_ranks.append(0)
            continue
        columns = torch.cat(remaining, dim=0).transpose(0, 1)
        singular_values = torch.linalg.svdvals(columns)
        cutoff = singular_values[0] * float(basis.config.rank_rtol)
        recomputed_type_out_ranks.append(
            int((singular_values > cutoff).sum().item())
        )
    if diagnostics.leave_one_type_out_rank != tuple(recomputed_type_out_ranks):
        raise InternalTemporalQuotientError(
            "nuisance leave-one-type-out ranks differ from raw observations"
        )
    gram = value.transpose(0, 1) @ value
    eye = torch.eye(basis.rank, dtype=value.dtype, device=value.device)
    if float((gram - eye).abs().max().item()) > float(
        basis.config.orthogonality_tolerance
    ):
        raise InternalTemporalQuotientError(
            "nuisance basis is not orthonormal"
        )
    return basis


@dataclass(frozen=True)
class ObservableNuisanceProjection:
    """A value and diagnostics after joint observable-nuisance projection."""

    projected: Any
    coefficients: Any
    input_rms: Any
    projected_rms: Any
    retention: Any
    max_abs_postprojection_basis_cosine: Any


def project_observable_nuisance(
    value: Any,
    nuisance_basis: TypedObservableNuisanceBasis,
) -> ObservableNuisanceProjection:
    """Remove the registered observable nuisance span from ``value``."""

    torch = _require_torch()
    candidate = _validate_tensor(
        value,
        name="value",
        ranks=tuple(range(2, 8)),
    )
    basis = _validate_nuisance_basis(nuisance_basis)
    _same_storage_contract(candidate, basis.basis, name="nuisance_basis.basis")
    flat = candidate.reshape(-1, EXPECTED_HIDDEN_SIZE)
    coefficients = flat @ basis.basis
    projected_flat = flat - coefficients @ basis.basis.transpose(0, 1)
    projected = projected_flat.reshape_as(candidate)
    coefficients = coefficients.reshape(*candidate.shape[:-1], basis.rank)
    input_rms = torch.sqrt(torch.mean(candidate.square()))
    projected_rms = torch.sqrt(torch.mean(projected.square()))
    if float(input_rms.item()) <= float(basis.config.epsilon):
        retention = torch.zeros_like(input_rms)
    else:
        retention = projected_rms / input_rms
    row_norm = torch.linalg.vector_norm(projected_flat, dim=1, keepdim=True)
    safe_norm = torch.clamp(row_norm, min=float(basis.config.epsilon))
    post_cosine = torch.abs(projected_flat @ basis.basis) / safe_norm
    max_post_cosine = (
        post_cosine.max()
        if post_cosine.numel()
        else torch.zeros((), dtype=candidate.dtype, device=candidate.device)
    )
    if not bool(torch.isfinite(projected).all().item()):
        raise InternalTemporalQuotientError(
            "nuisance projection produced a non-finite value"
        )
    return ObservableNuisanceProjection(
        projected=projected,
        coefficients=coefficients,
        input_rms=input_rms,
        projected_rms=projected_rms,
        retention=retention,
        max_abs_postprojection_basis_cosine=max_post_cosine,
    )


@dataclass(frozen=True)
class DiscoverySubspaceConfig:
    """Pre-registered profile, rank, and hard gates for discovery."""

    rank: int
    evidence_profile: str
    spatial_descriptor_policy: str
    spatial_sketch_id: str | None = None
    spatial_sketch_digest: str | None = None
    rank_rtol: float = 1.0e-5
    epsilon: float = 1.0e-8
    orthogonality_tolerance: float = 2.0e-4
    minimum_episodes: int = 2
    null_quantile: float = 0.99
    null_floor_multiplier: float = 2.0
    absolute_null_floor: float = 1.0e-6
    min_positive_consensus_cosine: float = 0.50
    min_contrast_cosine: float = 0.50
    min_semantic_margin: float = 0.10

    def validate(self) -> None:
        if type(self.rank) is not int or not 1 <= self.rank < EXPECTED_HIDDEN_SIZE:
            raise InternalTemporalQuotientError(
                "rank must be an integer in [1,1535]"
            )
        rank_rtol = _finite_positive("rank_rtol", self.rank_rtol)
        if rank_rtol >= 1.0:
            raise InternalTemporalQuotientError("rank_rtol must be below one")
        _finite_positive("epsilon", self.epsilon)
        tolerance = _finite_positive(
            "orthogonality_tolerance", self.orthogonality_tolerance
        )
        if tolerance >= 1.0:
            raise InternalTemporalQuotientError(
                "orthogonality_tolerance must be below one"
            )
        if type(self.minimum_episodes) is not int or self.minimum_episodes < 2:
            raise InternalTemporalQuotientError(
                "minimum_episodes must be an integer of at least two"
            )
        if self.evidence_profile not in DISCOVERY_EVIDENCE_PROFILES:
            raise InternalTemporalQuotientError(
                "evidence_profile must be engineering_micro or scientific"
            )
        _validate_spatial_descriptor_binding(
            policy=self.spatial_descriptor_policy,
            sketch_id=self.spatial_sketch_id,
            sketch_digest=self.spatial_sketch_digest,
            evidence_profile=self.evidence_profile,
        )
        null_quantile = _finite_positive("null_quantile", self.null_quantile)
        if null_quantile > 1.0:
            raise InternalTemporalQuotientError(
                "null_quantile must not exceed one"
            )
        _finite_positive("null_floor_multiplier", self.null_floor_multiplier)
        _finite_positive("absolute_null_floor", self.absolute_null_floor)
        for name in (
            "min_positive_consensus_cosine",
            "min_contrast_cosine",
        ):
            value = _finite_nonnegative(name, getattr(self, name))
            if value > 1.0:
                raise InternalTemporalQuotientError(f"{name} must not exceed one")
        semantic_margin = _finite_nonnegative(
            "min_semantic_margin", self.min_semantic_margin
        )
        if semantic_margin > 2.0:
            raise InternalTemporalQuotientError(
                "min_semantic_margin must not exceed two"
            )


@dataclass(frozen=True)
class DiscoveryContrastDiagnostics:
    """Unreduced discovery evidence for every episode and negative label."""

    semantic_negative_labels: tuple[str, ...]
    positive_cosine: Any
    leave_one_group_positive_cosine: Any
    semantic_negative_cosine: Any
    contrast_cosine: Any
    semantic_margins: Any
    positive_action_rms: Any
    semantic_negative_action_rms: Any
    contrast_action_rms: Any
    contrast_observable: Any
    per_negative_passed: Any
    per_label_passed: Any
    prototype_consensus_defined: bool
    positive_consensus_passed: bool
    positive_null_floor_passed: bool
    all_registered_negatives_passed: bool
    discovery_gate_passed: bool


@dataclass(frozen=True)
class FixedDiscoveryActionSubspace:
    """A discovery-frozen action basis, signed prototype, and spatial binding."""

    basis: Any
    prototype: Any
    singular_values: Any
    rank: int
    evidence_profile: str
    scientific_local_discovery_eligible: bool
    discovery_episode_count: int
    discovery_group_ids: tuple[str, ...]
    temporal_feature_steps: int
    temporal_weighting: str
    spatial_descriptor_policy: str
    spatial_descriptor_size: int
    spatial_sketch_id: str | None
    spatial_sketch_digest: str | None
    spatial_sketch_matrix_snapshot: Any | None
    nuisance_type_names: tuple[str, ...]
    nuisance_donor_group_ids: tuple[tuple[str, ...], ...]
    nuisance_config_digest: str
    nuisance_observation_digest: str
    nuisance_basis_digest: str
    nuisance_basis_snapshot: Any
    nuisance_leave_one_donor_passed: bool
    nuisance_audit_evidence: NuisanceAuditEvidence | None
    nuisance_audit_passed: bool
    evidence_binding: EvidenceBinding | None
    nuisance_projection_retention: Any
    discovery_null_action_rms: Any
    frozen_null_quantile_value: Any
    frozen_null_floor: Any
    discovery_diagnostics: DiscoveryContrastDiagnostics
    config: DiscoverySubspaceConfig


def _validate_feature_batch(
    features: Any,
    *,
    name: str,
    minimum_episodes: int = 1,
) -> Any:
    value = _validate_tensor(features, name=name, ranks=(4,))
    if int(value.shape[0]) < minimum_episodes:
        raise InternalTemporalQuotientError(
            f"{name} requires at least {minimum_episodes} episodes"
        )
    if int(value.shape[1]) != TEMPORAL_FEATURE_STEPS:
        raise InternalTemporalQuotientError(
            f"{name} must contain exactly {TEMPORAL_FEATURE_STEPS} temporal steps"
        )
    return value


def weight_temporal_direct_sum(features: Any) -> Any:
    """Apply fixed equal-block inner-product weights to an FITQ feature tensor.

    The temporal dimension must be the third dimension from the end, so both
    ``[E,F,P,D]`` and grouped ``[E,K,F,P,D]`` are accepted.  Every one of the
    five temporal blocks contributes its *mean* inner product divided by five;
    the 21-step blocks therefore cannot dominate the one-step terminal hold.
    """

    torch = _require_torch()
    value = _validate_tensor(features, name="features", ranks=(4, 5))
    if int(value.shape[-3]) != TEMPORAL_FEATURE_STEPS:
        raise InternalTemporalQuotientError(
            f"features must contain exactly {TEMPORAL_FEATURE_STEPS} temporal steps"
        )
    weights = torch.empty(
        TEMPORAL_FEATURE_STEPS,
        dtype=value.dtype,
        device=value.device,
    )
    block_count = float(len(TEMPORAL_BLOCK_SPECS))
    for _, start, stop in TEMPORAL_BLOCK_SPECS:
        weights[start:stop] = 1.0 / math.sqrt(block_count * float(stop - start))
    shape = (1,) * (value.ndim - 3) + (TEMPORAL_FEATURE_STEPS, 1, 1)
    weighted = value * weights.reshape(shape)
    if not bool(torch.isfinite(weighted).all().item()):
        raise InternalTemporalQuotientError(
            "temporal direct-sum weighting produced a non-finite value"
        )
    return weighted


def _signed_action_descriptors(
    coefficients: Any,
    *,
    epsilon: float,
    spatial_descriptor_policy: str,
) -> tuple[Any, Any, Any]:
    """Return signed-only temporal descriptors and their norms.

    Spatial RMS is intentionally absent: nonnegative energy may establish
    nondegeneracy elsewhere, but it is forbidden from rescuing an orientation
    that cancels across factorial groups.
    """

    torch = _require_torch()
    if spatial_descriptor_policy == "global_mean_engineering":
        raw = coefficients.mean(dim=-2).flatten(start_dim=-2)
    elif spatial_descriptor_policy == "fixed_signed_sketches":
        # P is already a pre-registered signed-sketch coordinate basis.  It is
        # part of the directional descriptor and must never be averaged away.
        raw = coefficients.flatten(start_dim=-3)
    else:
        raise InternalTemporalQuotientError(
            "signed descriptor received an unregistered spatial policy"
        )
    norms = torch.linalg.vector_norm(raw, dim=-1)
    unit = raw / torch.clamp(norms.unsqueeze(-1), min=float(epsilon))
    return raw, unit, norms


def build_fixed_discovery_action_subspace(
    discovery_positive_features: Any,
    discovery_semantic_negative_features: Any,
    nuisance_basis: TypedObservableNuisanceBasis,
    *,
    discovery_semantic_negative_labels: tuple[str, ...],
    expected_discovery_semantic_negative_labels: tuple[str, ...],
    discovery_null_features: Any,
    discovery_group_ids: tuple[str, ...],
    config: DiscoverySubspaceConfig,
    spatial_sketch_matrix: Any | None = None,
    evidence_binding: EvidenceBinding | None = None,
    nuisance_audit_evidence: NuisanceAuditEvidence | None = None,
) -> FixedDiscoveryActionSubspace:
    """Jointly fit positives and every labelled discovery contrast.

    The SVD input is an explicit concatenation of same-action positive cells
    and all ``positive-negative[k]`` cells.  Each cell is Frobenius-normalized;
    positives receive half of the joint covariance weight and the ``K``
    separately retained contrast groups share the other half.  No negative is
    averaged, mined, or selected before factorization.  The signed prototype is
    then frozen from positives only, and every registered negative is gated and
    retained in :class:`DiscoveryContrastDiagnostics`.
    """

    torch = _require_torch()
    if not isinstance(config, DiscoverySubspaceConfig):
        raise InternalTemporalQuotientError(
            "config must be a DiscoverySubspaceConfig"
        )
    config.validate()
    minimum_episodes = config.minimum_episodes
    if config.evidence_profile == "scientific":
        minimum_episodes = max(
            minimum_episodes,
            SCIENTIFIC_MIN_DISCOVERY_EPISODES,
        )
    positives = _validate_feature_batch(
        discovery_positive_features,
        name="discovery_positive_features",
        minimum_episodes=minimum_episodes,
    )
    episodes = int(positives.shape[0])
    spatial_descriptor_size = int(positives.shape[2])
    if (
        config.evidence_profile == "scientific"
        and spatial_descriptor_size != SCIENTIFIC_SPATIAL_SKETCH_COORDINATES
    ):
        raise InternalTemporalQuotientError(
            "scientific signed-sketch evidence requires exactly 16 coordinates"
        )
    sketch_matrix_digest, sketch_matrix_snapshot = _validate_spatial_sketch_matrix(
        spatial_sketch_matrix,
        policy=config.spatial_descriptor_policy,
        expected_coordinates=spatial_descriptor_size,
        expected_source_positions=(
            SCIENTIFIC_PATCH_TOKENS
            if config.evidence_profile == "scientific"
            else None
        ),
        expected_digest=config.spatial_sketch_digest,
    )
    group_ids = _validate_group_ids(
        discovery_group_ids,
        name="discovery_group_ids",
        expected_count=episodes,
    )
    expected_labels = _validate_labels(
        expected_discovery_semantic_negative_labels,
        name="expected_discovery_semantic_negative_labels",
    )
    if len(expected_labels) < MIN_HARD_NEGATIVES:
        raise InternalTemporalQuotientError(
            f"discovery requires at least {MIN_HARD_NEGATIVES} registered negatives"
        )
    labels = _validate_labels(
        discovery_semantic_negative_labels,
        name="discovery_semantic_negative_labels",
        expected_count=len(expected_labels),
    )
    if labels != expected_labels:
        raise InternalTemporalQuotientError(
            "discovery negative labels or their registered order do not match"
        )
    if (
        config.evidence_profile == "scientific"
        and expected_labels != SCIENTIFIC_REQUIRED_NEGATIVE_LABELS
    ):
        raise InternalTemporalQuotientError(
            "scientific discovery requires the complete registered negative label set"
        )
    if evidence_binding is not None:
        if not isinstance(evidence_binding, EvidenceBinding):
            raise InternalTemporalQuotientError(
                "evidence_binding must be an EvidenceBinding"
            )
        evidence_binding.validate()
        if evidence_binding.negative_label_set != labels:
            raise InternalTemporalQuotientError(
                "EvidenceBinding negative labels differ from discovery"
            )
        if evidence_binding.spatial_sketch_matrix_digest != sketch_matrix_digest:
            raise InternalTemporalQuotientError(
                "EvidenceBinding sketch digest differs from raw matrix"
            )
    if config.evidence_profile == "scientific":
        if evidence_binding is None:
            raise InternalTemporalQuotientError(
                "scientific discovery requires an exact EvidenceBinding"
            )
        if (
            evidence_binding.checkpoint_tree_sha256
            != PINNED_CHECKPOINT_TREE_SHA256
            or evidence_binding.bernini_revision != PINNED_BERNINI_REVISION
            or evidence_binding.veomni_revision != PINNED_VEOMNI_REVISION
        ):
            raise InternalTemporalQuotientError(
                "scientific EvidenceBinding differs from pinned model artifacts"
            )
        if (
            evidence_binding.latent_geometry != SCIENTIFIC_LATENT_GEOMETRY
            or evidence_binding.patch_geometry != SCIENTIFIC_PATCH_GEOMETRY
        ):
            raise InternalTemporalQuotientError(
                "scientific EvidenceBinding differs from the exact Bernini "
                "81-frame latent/patch geometry"
            )
    negatives = _validate_grouped_features(
        discovery_semantic_negative_features,
        name="discovery_semantic_negative_features",
        episodes=episodes,
    )
    if int(negatives.shape[1]) != len(labels):
        raise InternalTemporalQuotientError(
            "discovery negative group count does not match its labels"
        )
    discovery_nulls = _validate_grouped_features(
        discovery_null_features,
        name="discovery_null_features",
        episodes=episodes,
    )
    if int(negatives.shape[3]) != int(positives.shape[2]) or int(
        discovery_nulls.shape[3]
    ) != int(positives.shape[2]):
        raise InternalTemporalQuotientError(
            "all discovery branches must share the same spatial token count"
        )
    nuisance = _validate_nuisance_basis(nuisance_basis)
    if (
        config.evidence_profile == "scientific"
        and nuisance.type_names != SCIENTIFIC_REQUIRED_NUISANCE_TYPES
    ):
        raise InternalTemporalQuotientError(
            "scientific discovery requires the exact registered nuisance types"
        )
    if (
        config.evidence_profile == "scientific"
        and nuisance.config.evidence_profile != "scientific"
    ):
        raise InternalTemporalQuotientError(
            "scientific discovery requires a scientific nuisance config"
        )
    nuisance_audit_passed = False
    if nuisance_audit_evidence is not None:
        if not isinstance(nuisance_audit_evidence, NuisanceAuditEvidence):
            raise InternalTemporalQuotientError(
                "nuisance_audit_evidence must be NuisanceAuditEvidence"
            )
        nuisance_audit_evidence.validate(expected_types=nuisance.type_names)
        if (
            nuisance_audit_evidence.nuisance_observation_digest
            != nuisance.observation_digest
            or nuisance_audit_evidence.nuisance_basis_digest
            != nuisance.basis_digest
            or nuisance_audit_evidence.donor_group_ids
            != nuisance.donor_group_ids
        ):
            raise InternalTemporalQuotientError(
                "nuisance audit evidence differs from the exact nuisance basis"
            )
        nuisance_audit_passed = all(
            (
                nuisance_audit_evidence.leave_one_donor_passed,
                nuisance_audit_evidence.leave_one_type_out_passed,
                nuisance_audit_evidence.signature_verified,
            )
        )
    if evidence_binding is not None:
        if (
            canonical_nuisance_basis_config_digest(nuisance.config)
            != evidence_binding.nuisance_config_digest
            or canonical_discovery_subspace_config_digest(config)
            != evidence_binding.discovery_config_digest
        ):
            raise InternalTemporalQuotientError(
                "EvidenceBinding discovery/nuisance config digest differs from "
                "runtime policy"
            )
        actual_nuisance_audit_digest = canonical_nuisance_audit_evidence_digest(
            nuisance_audit_evidence,
            expected_types=nuisance.type_names,
        )
        if actual_nuisance_audit_digest != evidence_binding.nuisance_audit_digest:
            raise InternalTemporalQuotientError(
                "EvidenceBinding nuisance audit digest differs from exact "
                "attestation"
            )
        actual_discovery_digest = canonical_fitq_split_content_digest(
            positives,
            negatives,
            discovery_nulls,
            split="discovery",
            group_ids=group_ids,
            semantic_negative_labels=labels,
            nuisance_observation_digest=nuisance.observation_digest,
            nuisance_basis_digest=nuisance.basis_digest,
            spatial_sketch_matrix_digest=sketch_matrix_digest,
            evidence_binding=evidence_binding,
        )
        if actual_discovery_digest != evidence_binding.discovery_digest:
            raise InternalTemporalQuotientError(
                "EvidenceBinding discovery digest differs from exact split content"
            )
    for name, value in (
        ("nuisance_basis.basis", nuisance.basis),
        ("discovery_semantic_negative_features", negatives),
        ("discovery_null_features", discovery_nulls),
    ):
        _same_storage_contract(positives, value, name=name)

    positive_projection = project_observable_nuisance(positives, nuisance)
    negative_projection = project_observable_nuisance(negatives, nuisance)
    weighted_positives = weight_temporal_direct_sum(
        positive_projection.projected
    )
    weighted_negatives = weight_temporal_direct_sum(
        negative_projection.projected
    )
    weighted_contrasts = weighted_positives[:, None] - weighted_negatives

    positive_norms = torch.linalg.vector_norm(
        weighted_positives.reshape(episodes, -1),
        dim=1,
    )
    if bool((positive_norms <= float(config.epsilon)).any().item()):
        raise InternalTemporalQuotientError(
            "every discovery positive must contain observable action evidence"
        )
    contrast_norms = torch.linalg.vector_norm(
        weighted_contrasts.reshape(episodes, len(labels), -1),
        dim=2,
    )
    normalized_positives = weighted_positives / positive_norms[:, None, None, None]
    normalized_contrasts = weighted_contrasts / torch.clamp(
        contrast_norms[:, :, None, None, None],
        min=float(config.epsilon),
    )
    # The two explicit matrices are concatenated, never reduced across K.
    positive_scale = 1.0 / math.sqrt(2.0 * float(episodes))
    contrast_scale = 1.0 / math.sqrt(
        2.0 * float(episodes) * float(len(labels))
    )
    matrix = torch.cat(
        (
            (positive_scale * normalized_positives).reshape(
                -1, EXPECTED_HIDDEN_SIZE
            ),
            (contrast_scale * normalized_contrasts).reshape(
                -1, EXPECTED_HIDDEN_SIZE
            ),
        ),
        dim=0,
    )
    _, singular_values, right_h = torch.linalg.svd(matrix, full_matrices=False)
    if int(singular_values.numel()) < config.rank:
        raise InternalTemporalQuotientError(
            "discovery matrix cannot support the registered action rank"
        )
    cutoff = singular_values[0] * float(config.rank_rtol)
    if float(singular_values[config.rank - 1].item()) <= float(cutoff.item()):
        raise InternalTemporalQuotientError(
            "discovery action evidence is rank deficient"
        )
    action_basis = right_h[: config.rank].transpose(0, 1).contiguous()
    nuisance_leakage = torch.abs(
        nuisance.basis.transpose(0, 1) @ action_basis
    ).max()
    if float(nuisance_leakage.item()) > float(config.orthogonality_tolerance):
        raise InternalTemporalQuotientError(
            "discovery action basis leaks into the nuisance span"
        )
    gram = action_basis.transpose(0, 1) @ action_basis
    eye = torch.eye(
        config.rank,
        dtype=action_basis.dtype,
        device=action_basis.device,
    )
    if float((gram - eye).abs().max().item()) > float(
        config.orthogonality_tolerance
    ):
        raise InternalTemporalQuotientError(
            "discovery action basis is not orthonormal"
        )
    positive_coefficients = weighted_positives @ action_basis
    negative_coefficients = weighted_negatives @ action_basis
    contrast_coefficients = weighted_contrasts @ action_basis
    positive_raw, positive_unit, positive_descriptor_norms = _signed_action_descriptors(
        positive_coefficients,
        epsilon=config.epsilon,
        spatial_descriptor_policy=config.spatial_descriptor_policy,
    )
    if bool((positive_descriptor_norms <= float(config.epsilon)).any().item()):
        raise InternalTemporalQuotientError(
            "a discovery action descriptor is degenerate"
        )
    prototype_raw = positive_unit.sum(dim=0)
    prototype_norm = torch.linalg.vector_norm(prototype_raw)
    prototype_consensus_defined = bool(
        float(prototype_norm.item()) > float(config.epsilon)
    )
    if prototype_consensus_defined:
        prototype = prototype_raw / prototype_norm
    else:
        # Keep a deterministic signed diagnostic orientation so an N0 result
        # remains inspectable.  The explicit consensus flag below prevents this
        # fallback from authorizing scientific confirmation.
        prototype = positive_unit[0]

    null_projection = project_observable_nuisance(discovery_nulls, nuisance)
    weighted_nulls = weight_temporal_direct_sum(null_projection.projected)
    discovery_null_coefficients = weighted_nulls @ action_basis
    discovery_null_action_rms = torch.sqrt(
        discovery_null_coefficients.square().mean(dim=(2, 3, 4))
    )
    frozen_quantile = torch.quantile(
        discovery_null_action_rms.reshape(-1),
        float(config.null_quantile),
    )
    absolute_floor = torch.full_like(
        frozen_quantile,
        float(config.absolute_null_floor),
    )
    frozen_null_floor = torch.maximum(
        absolute_floor,
        float(config.null_floor_multiplier) * frozen_quantile,
    )
    if not bool(torch.isfinite(frozen_null_floor).item()):
        raise InternalTemporalQuotientError(
            "discovery null controls produced a non-finite frozen floor"
        )

    negative_raw, negative_unit, _ = _signed_action_descriptors(
        negative_coefficients,
        epsilon=config.epsilon,
        spatial_descriptor_policy=config.spatial_descriptor_policy,
    )
    _, contrast_unit, contrast_descriptor_norms = _signed_action_descriptors(
        contrast_coefficients,
        epsilon=config.epsilon,
        spatial_descriptor_policy=config.spatial_descriptor_policy,
    )
    positive_cosine = positive_unit @ prototype
    leave_one_sum = prototype_raw[None] - positive_unit
    leave_one_norm = torch.linalg.vector_norm(leave_one_sum, dim=1)
    leave_one_unit = leave_one_sum / torch.clamp(
        leave_one_norm[:, None],
        min=float(config.epsilon),
    )
    leave_one_group_positive_cosine = (positive_unit * leave_one_unit).sum(dim=1)
    leave_one_defined = leave_one_norm > float(config.epsilon)
    negative_cosine = negative_unit @ prototype
    contrast_cosine = contrast_unit @ prototype
    positive_signed_score = (positive_raw @ prototype) / torch.clamp(
        positive_descriptor_norms,
        min=float(config.epsilon),
    )
    negative_relative_score = (negative_raw @ prototype) / torch.clamp(
        positive_descriptor_norms[:, None],
        min=float(config.epsilon),
    )
    semantic_margins = positive_signed_score[:, None] - negative_relative_score
    positive_action_rms = torch.sqrt(
        positive_coefficients.square().mean(dim=(1, 2, 3))
    )
    negative_action_rms = torch.sqrt(
        negative_coefficients.square().mean(dim=(2, 3, 4))
    )
    contrast_action_rms = torch.sqrt(
        contrast_coefficients.square().mean(dim=(2, 3, 4))
    )
    contrast_observable = contrast_descriptor_norms > float(config.epsilon)
    per_negative_passed = (
        contrast_observable
        & (contrast_cosine >= float(config.min_contrast_cosine))
        & (semantic_margins >= float(config.min_semantic_margin))
        & (contrast_action_rms > frozen_null_floor)
    )
    per_label_passed = per_negative_passed.all(dim=0)
    positive_consensus_passed = bool(
        prototype_consensus_defined
        and leave_one_defined.all().item()
        and (
            positive_cosine >= float(config.min_positive_consensus_cosine)
        ).all().item()
        and (
            leave_one_group_positive_cosine
            >= float(config.min_positive_consensus_cosine)
        ).all().item()
    )
    positive_null_floor_passed = bool(
        (positive_action_rms > frozen_null_floor).all().item()
    )
    all_registered_negatives_passed = bool(per_negative_passed.all().item())
    discovery_gate_passed = all(
        (
            positive_consensus_passed,
            positive_null_floor_passed,
            all_registered_negatives_passed,
        )
    )
    diagnostics = DiscoveryContrastDiagnostics(
        semantic_negative_labels=labels,
        positive_cosine=positive_cosine,
        leave_one_group_positive_cosine=leave_one_group_positive_cosine,
        semantic_negative_cosine=negative_cosine,
        contrast_cosine=contrast_cosine,
        semantic_margins=semantic_margins,
        positive_action_rms=positive_action_rms,
        semantic_negative_action_rms=negative_action_rms,
        contrast_action_rms=contrast_action_rms,
        contrast_observable=contrast_observable,
        per_negative_passed=per_negative_passed,
        per_label_passed=per_label_passed,
        prototype_consensus_defined=prototype_consensus_defined,
        positive_consensus_passed=positive_consensus_passed,
        positive_null_floor_passed=positive_null_floor_passed,
        all_registered_negatives_passed=all_registered_negatives_passed,
        discovery_gate_passed=discovery_gate_passed,
    )
    scientific_local_discovery_eligible = bool(
        config.evidence_profile == "scientific"
        and discovery_gate_passed
        and nuisance.leave_one_out_diagnostics.leave_one_donor_gate_passed
        and nuisance_audit_passed
    )
    return FixedDiscoveryActionSubspace(
        basis=action_basis,
        prototype=prototype,
        singular_values=singular_values,
        rank=config.rank,
        evidence_profile=config.evidence_profile,
        scientific_local_discovery_eligible=scientific_local_discovery_eligible,
        discovery_episode_count=episodes,
        discovery_group_ids=group_ids,
        temporal_feature_steps=TEMPORAL_FEATURE_STEPS,
        temporal_weighting="five_block_equal_mean_inner_product",
        spatial_descriptor_policy=config.spatial_descriptor_policy,
        spatial_descriptor_size=spatial_descriptor_size,
        spatial_sketch_id=config.spatial_sketch_id,
        spatial_sketch_digest=sketch_matrix_digest,
        spatial_sketch_matrix_snapshot=sketch_matrix_snapshot,
        nuisance_type_names=nuisance.type_names,
        nuisance_donor_group_ids=nuisance.donor_group_ids,
        nuisance_config_digest=canonical_nuisance_basis_config_digest(
            nuisance.config
        ),
        nuisance_observation_digest=nuisance.observation_digest,
        nuisance_basis_digest=nuisance.basis_digest,
        nuisance_basis_snapshot=nuisance.basis.clone(),
        nuisance_leave_one_donor_passed=(
            nuisance.leave_one_out_diagnostics.leave_one_donor_gate_passed
        ),
        nuisance_audit_evidence=nuisance_audit_evidence,
        nuisance_audit_passed=nuisance_audit_passed,
        evidence_binding=evidence_binding,
        nuisance_projection_retention=positive_projection.retention,
        discovery_null_action_rms=discovery_null_action_rms,
        frozen_null_quantile_value=frozen_quantile,
        frozen_null_floor=frozen_null_floor,
        discovery_diagnostics=diagnostics,
        config=config,
    )


def _validate_action_subspace(
    action_subspace: Any,
) -> FixedDiscoveryActionSubspace:
    torch = _require_torch()
    if not isinstance(action_subspace, FixedDiscoveryActionSubspace):
        raise InternalTemporalQuotientError(
            "action_subspace must be a FixedDiscoveryActionSubspace"
        )
    action_subspace.config.validate()
    if action_subspace.evidence_profile != action_subspace.config.evidence_profile:
        raise InternalTemporalQuotientError(
            "action subspace evidence profile binding is inconsistent"
        )
    if (
        action_subspace.spatial_descriptor_policy
        != action_subspace.config.spatial_descriptor_policy
        or action_subspace.spatial_sketch_id
        != action_subspace.config.spatial_sketch_id
        or action_subspace.spatial_sketch_digest
        != action_subspace.config.spatial_sketch_digest
    ):
        raise InternalTemporalQuotientError(
            "action subspace spatial descriptor binding is inconsistent"
        )
    _validate_spatial_descriptor_binding(
        policy=action_subspace.spatial_descriptor_policy,
        sketch_id=action_subspace.spatial_sketch_id,
        sketch_digest=action_subspace.spatial_sketch_digest,
        evidence_profile=action_subspace.evidence_profile,
    )
    if (
        type(action_subspace.spatial_descriptor_size) is not int
        or action_subspace.spatial_descriptor_size <= 0
    ):
        raise InternalTemporalQuotientError(
            "action subspace spatial descriptor size is invalid"
        )
    if (
        action_subspace.evidence_profile == "scientific"
        and action_subspace.spatial_descriptor_size
        != SCIENTIFIC_SPATIAL_SKETCH_COORDINATES
    ):
        raise InternalTemporalQuotientError(
            "scientific action subspace requires exactly 16 signed-sketch "
            "coordinates"
        )
    _, validated_sketch_snapshot = _validate_spatial_sketch_matrix(
        action_subspace.spatial_sketch_matrix_snapshot,
        policy=action_subspace.spatial_descriptor_policy,
        expected_coordinates=action_subspace.spatial_descriptor_size,
        expected_source_positions=(
            SCIENTIFIC_PATCH_TOKENS
            if action_subspace.evidence_profile == "scientific"
            else None
        ),
        expected_digest=action_subspace.spatial_sketch_digest,
    )
    if (
        validated_sketch_snapshot is not None
        and not torch.equal(
            validated_sketch_snapshot,
            action_subspace.spatial_sketch_matrix_snapshot,
        )
    ):
        raise InternalTemporalQuotientError(
            "action subspace sketch snapshot changed"
        )
    if not isinstance(
        action_subspace.discovery_diagnostics,
        DiscoveryContrastDiagnostics,
    ):
        raise InternalTemporalQuotientError(
            "action subspace lacks discovery contrast diagnostics"
        )
    diagnostic_labels = _validate_labels(
        action_subspace.discovery_diagnostics.semantic_negative_labels,
        name="action_subspace.discovery_diagnostics.semantic_negative_labels",
    )
    if len(diagnostic_labels) < MIN_HARD_NEGATIVES:
        raise InternalTemporalQuotientError(
            "action subspace lacks the minimum discovery negatives"
        )
    if action_subspace.evidence_profile == "scientific":
        if action_subspace.discovery_episode_count < SCIENTIFIC_MIN_DISCOVERY_EPISODES:
            raise InternalTemporalQuotientError(
                "scientific action subspace has too few discovery episodes"
            )
        if diagnostic_labels != SCIENTIFIC_REQUIRED_NEGATIVE_LABELS:
            raise InternalTemporalQuotientError(
                "scientific action subspace negative binding is incomplete"
            )
        if action_subspace.nuisance_type_names != SCIENTIFIC_REQUIRED_NUISANCE_TYPES:
            raise InternalTemporalQuotientError(
                "scientific action subspace nuisance binding is incomplete"
            )
        if action_subspace.evidence_binding is None:
            raise InternalTemporalQuotientError(
                "scientific action subspace lacks EvidenceBinding"
            )
        if (
            action_subspace.evidence_binding.latent_geometry
            != SCIENTIFIC_LATENT_GEOMETRY
            or action_subspace.evidence_binding.patch_geometry
            != SCIENTIFIC_PATCH_GEOMETRY
        ):
            raise InternalTemporalQuotientError(
                "scientific action subspace EvidenceBinding geometry is "
                "incompatible"
            )
    if action_subspace.evidence_binding is not None:
        action_subspace.evidence_binding.validate()
        if (
            action_subspace.evidence_binding.negative_label_set
            != diagnostic_labels
            or action_subspace.evidence_binding.spatial_sketch_matrix_digest
            != action_subspace.spatial_sketch_digest
            or action_subspace.evidence_binding.discovery_config_digest
            != canonical_discovery_subspace_config_digest(action_subspace.config)
            or action_subspace.evidence_binding.nuisance_config_digest
            != action_subspace.nuisance_config_digest
        ):
            raise InternalTemporalQuotientError(
                "action subspace EvidenceBinding content is inconsistent"
            )
        if action_subspace.evidence_profile == "scientific" and (
            action_subspace.evidence_binding.checkpoint_tree_sha256
            != PINNED_CHECKPOINT_TREE_SHA256
            or action_subspace.evidence_binding.bernini_revision
            != PINNED_BERNINI_REVISION
            or action_subspace.evidence_binding.veomni_revision
            != PINNED_VEOMNI_REVISION
        ):
            raise InternalTemporalQuotientError(
                "scientific action subspace EvidenceBinding differs from pinned "
                "model artifacts"
            )
    nuisance_snapshot = _validate_tensor(
        action_subspace.nuisance_basis_snapshot,
        name="action_subspace.nuisance_basis_snapshot",
        ranks=(2,),
        last_dimension=None,
    )
    _validate_hex_digest(
        "action_subspace.nuisance_config_digest",
        action_subspace.nuisance_config_digest,
        length=64,
    )
    minimum_donors = (
        2 if action_subspace.evidence_profile == "scientific" else 0
    )
    if minimum_donors:
        _validate_nested_donor_ids(
            action_subspace.nuisance_donor_group_ids,
            expected_types=action_subspace.nuisance_type_names,
            minimum_per_type=minimum_donors,
        )
    elif (
        type(action_subspace.nuisance_donor_group_ids) is not tuple
        or len(action_subspace.nuisance_donor_group_ids)
        != len(action_subspace.nuisance_type_names)
    ):
        raise InternalTemporalQuotientError(
            "action subspace nuisance donor binding is malformed"
        )
    if int(nuisance_snapshot.shape[0]) != EXPECTED_HIDDEN_SIZE:
        raise InternalTemporalQuotientError(
            "action subspace nuisance snapshot ambient dimension is invalid"
        )
    _validate_hex_digest(
        "action_subspace.nuisance_observation_digest",
        action_subspace.nuisance_observation_digest,
        length=64,
    )
    if canonical_tensor_raw_value_digest(
        nuisance_snapshot,
        name="action_subspace.nuisance_basis_snapshot",
    ) != action_subspace.nuisance_basis_digest:
        raise InternalTemporalQuotientError(
            "action subspace nuisance snapshot raw values changed"
        )
    if type(action_subspace.nuisance_leave_one_donor_passed) is not bool:
        raise InternalTemporalQuotientError(
            "nuisance leave-one-donor gate must be boolean"
        )
    if action_subspace.nuisance_audit_evidence is not None:
        action_subspace.nuisance_audit_evidence.validate(
            expected_types=action_subspace.nuisance_type_names
        )
        audit = action_subspace.nuisance_audit_evidence
        if (
            audit.nuisance_observation_digest
            != action_subspace.nuisance_observation_digest
            or audit.nuisance_basis_digest != action_subspace.nuisance_basis_digest
            or audit.donor_group_ids
            != action_subspace.nuisance_donor_group_ids
        ):
            raise InternalTemporalQuotientError(
                "action subspace nuisance audit binding is inconsistent"
            )
    expected_audit_passed = bool(
        action_subspace.nuisance_audit_evidence is not None
        and action_subspace.nuisance_audit_evidence.leave_one_donor_passed
        and action_subspace.nuisance_audit_evidence.leave_one_type_out_passed
        and action_subspace.nuisance_audit_evidence.signature_verified
    )
    if action_subspace.nuisance_audit_passed != expected_audit_passed:
        raise InternalTemporalQuotientError(
            "action subspace nuisance audit decision is inconsistent"
        )
    if action_subspace.evidence_binding is not None and (
        canonical_nuisance_audit_evidence_digest(
            action_subspace.nuisance_audit_evidence,
            expected_types=action_subspace.nuisance_type_names,
        )
        != action_subspace.evidence_binding.nuisance_audit_digest
    ):
        raise InternalTemporalQuotientError(
            "action subspace nuisance audit differs from EvidenceBinding"
        )
    expected_scientific_eligibility = bool(
        action_subspace.evidence_profile == "scientific"
        and action_subspace.discovery_diagnostics.discovery_gate_passed
        and action_subspace.nuisance_leave_one_donor_passed
        and action_subspace.nuisance_audit_passed
    )
    if (
        action_subspace.scientific_local_discovery_eligible
        != expected_scientific_eligibility
    ):
        raise InternalTemporalQuotientError(
            "action subspace scientific eligibility binding is inconsistent"
        )
    if (
        action_subspace.temporal_feature_steps != TEMPORAL_FEATURE_STEPS
        or action_subspace.temporal_weighting
        != "five_block_equal_mean_inner_product"
    ):
        raise InternalTemporalQuotientError(
            "action subspace temporal feature binding is incompatible"
        )
    basis = _validate_tensor(
        action_subspace.basis,
        name="action_subspace.basis",
        ranks=(2,),
        last_dimension=action_subspace.rank,
    )
    if int(basis.shape[0]) != EXPECTED_HIDDEN_SIZE:
        raise InternalTemporalQuotientError(
            "action subspace has the wrong ambient dimension"
        )
    expected_prototype = TEMPORAL_FEATURE_STEPS * action_subspace.rank
    if action_subspace.spatial_descriptor_policy == "fixed_signed_sketches":
        expected_prototype *= action_subspace.spatial_descriptor_size
    prototype = _validate_tensor(
        action_subspace.prototype,
        name="action_subspace.prototype",
        ranks=(1,),
        last_dimension=expected_prototype,
    )
    _same_storage_contract(basis, prototype, name="action_subspace.prototype")
    gram = basis.transpose(0, 1) @ basis
    eye = torch.eye(
        action_subspace.rank,
        dtype=basis.dtype,
        device=basis.device,
    )
    if float((gram - eye).abs().max().item()) > float(
        action_subspace.config.orthogonality_tolerance
    ):
        raise InternalTemporalQuotientError(
            "action subspace basis is not orthonormal"
        )
    if abs(float(torch.linalg.vector_norm(prototype).item()) - 1.0) > float(
        action_subspace.config.orthogonality_tolerance
    ):
        raise InternalTemporalQuotientError(
            "action prototype must have unit norm"
        )
    _validate_group_ids(
        action_subspace.discovery_group_ids,
        name="action_subspace.discovery_group_ids",
        expected_count=action_subspace.discovery_episode_count,
    )
    floor = _validate_tensor(
        action_subspace.frozen_null_floor,
        name="action_subspace.frozen_null_floor",
        ranks=(0,),
        last_dimension=None,
    )
    if float(floor.item()) <= 0.0:
        raise InternalTemporalQuotientError(
            "action subspace frozen null floor must be positive"
        )
    _same_storage_contract(basis, floor, name="action_subspace.frozen_null_floor")
    return action_subspace


@dataclass(frozen=True)
class ConfirmationConfig:
    """Pre-registered thresholds for a frozen discovery/confirmation scan."""

    min_correct_cosine: float = 0.50
    min_grassmann_similarity: float = 0.80
    min_semantic_margin: float = 0.10
    min_null_cosine_margin: float = 0.10
    rank_rtol: float = 1.0e-5
    epsilon: float = 1.0e-8

    def validate(self) -> None:
        for name in ("min_correct_cosine", "min_grassmann_similarity"):
            value = _finite_nonnegative(name, getattr(self, name))
            if value > 1.0:
                raise InternalTemporalQuotientError(f"{name} must not exceed one")
        for name in ("min_semantic_margin", "min_null_cosine_margin"):
            value = _finite_nonnegative(name, getattr(self, name))
            if value > 2.0:
                raise InternalTemporalQuotientError(f"{name} must not exceed two")
        rank_rtol = _finite_positive("rank_rtol", self.rank_rtol)
        if rank_rtol >= 1.0:
            raise InternalTemporalQuotientError("rank_rtol must be below one")
        _finite_positive("epsilon", self.epsilon)


def canonical_nuisance_basis_config_digest(config: NuisanceBasisConfig) -> str:
    """Hash every nuisance policy field that can affect a scientific gate."""

    if not isinstance(config, NuisanceBasisConfig):
        raise InternalTemporalQuotientError(
            "nuisance config digest requires NuisanceBasisConfig"
        )
    config.validate()
    donor_parts = ["ordered_nuisance_config_donors_v1"]
    for type_name, donor_ids in zip(config.expected_types, config.donor_group_ids):
        donor_parts.extend((type_name, *donor_ids))
    return _composite_digest(
        (
            "fitq-nuisance-basis-config-v1",
            *config.expected_types,
            config.evidence_profile,
            _composite_digest(tuple(donor_parts)),
            float(config.min_leave_one_donor_cosine).hex(),
            float(config.rank_rtol).hex(),
            float(config.max_condition_number).hex(),
            float(config.epsilon).hex(),
            float(config.orthogonality_tolerance).hex(),
        )
    )


def canonical_discovery_subspace_config_digest(
    config: DiscoverySubspaceConfig,
) -> str:
    """Hash every discovery rank, policy, floor, and decision threshold."""

    if not isinstance(config, DiscoverySubspaceConfig):
        raise InternalTemporalQuotientError(
            "discovery config digest requires DiscoverySubspaceConfig"
        )
    config.validate()
    return _composite_digest(
        (
            "fitq-discovery-subspace-config-v1",
            str(config.rank),
            config.evidence_profile,
            config.spatial_descriptor_policy,
            config.spatial_sketch_id or "none",
            config.spatial_sketch_digest or "none",
            float(config.rank_rtol).hex(),
            float(config.epsilon).hex(),
            float(config.orthogonality_tolerance).hex(),
            str(config.minimum_episodes),
            float(config.null_quantile).hex(),
            float(config.null_floor_multiplier).hex(),
            float(config.absolute_null_floor).hex(),
            float(config.min_positive_consensus_cosine).hex(),
            float(config.min_contrast_cosine).hex(),
            float(config.min_semantic_margin).hex(),
        )
    )


def canonical_confirmation_config_digest(config: ConfirmationConfig) -> str:
    """Hash every confirmation threshold so post-hoc relaxation is detectable."""

    if not isinstance(config, ConfirmationConfig):
        raise InternalTemporalQuotientError(
            "confirmation config digest requires ConfirmationConfig"
        )
    config.validate()
    return _composite_digest(
        (
            "fitq-confirmation-config-v1",
            float(config.min_correct_cosine).hex(),
            float(config.min_grassmann_similarity).hex(),
            float(config.min_semantic_margin).hex(),
            float(config.min_null_cosine_margin).hex(),
            float(config.rank_rtol).hex(),
            float(config.epsilon).hex(),
        )
    )


@dataclass(frozen=True)
class ConfirmationMetrics:
    """Local geometry evidence; never an authorization for DMIQ/FITQ GO."""

    discovery_group_ids: tuple[str, ...]
    confirmation_group_ids: tuple[str, ...]
    discovery_evidence_profile: str
    semantic_negative_labels: tuple[str, ...]
    correct_cosine: Any
    semantic_negative_cosine: Any
    null_cosine: Any
    semantic_margins: Any
    null_cosine_margin: Any
    correct_action_rms: Any
    semantic_negative_action_rms: Any
    null_action_rms: Any
    null_floor: Any
    grassmann_similarity: Any
    correct_nonzero_passed: bool
    correct_cosine_passed: bool
    semantic_ordering_passed: bool
    null_cosine_margin_passed: bool
    null_floor_passed: bool
    confirmation_rank_passed: bool
    grassmann_passed: bool
    discovery_gate_passed: bool
    scientific_profile_passed: bool
    local_geometry_eligible: bool
    fitq_go_authorized: bool


def _validate_grouped_features(
    value: Any,
    *,
    name: str,
    episodes: int,
) -> Any:
    candidate = _validate_tensor(value, name=name, ranks=(5,))
    if int(candidate.shape[0]) != episodes:
        raise InternalTemporalQuotientError(
            f"{name} episode count does not match correct_features"
        )
    if int(candidate.shape[1]) <= 0:
        raise InternalTemporalQuotientError(f"{name} must contain a group")
    if int(candidate.shape[2]) != TEMPORAL_FEATURE_STEPS:
        raise InternalTemporalQuotientError(
            f"{name} must contain exactly {TEMPORAL_FEATURE_STEPS} temporal steps"
        )
    return candidate


def _diagnostic_confirmation_subspace(
    projected_correct: Any,
    *,
    rank: int,
    rank_rtol: float,
    epsilon: float,
) -> tuple[Any | None, bool]:
    torch = _require_torch()
    episode_norms = torch.linalg.vector_norm(
        projected_correct.reshape(int(projected_correct.shape[0]), -1), dim=1
    )
    if bool((episode_norms <= float(epsilon)).any().item()):
        return None, False
    normalized = projected_correct / episode_norms[:, None, None, None]
    matrix = normalized.reshape(-1, EXPECTED_HIDDEN_SIZE)
    _, singular_values, right_h = torch.linalg.svd(matrix, full_matrices=False)
    if int(singular_values.numel()) < rank:
        return None, False
    cutoff = singular_values[0] * float(rank_rtol)
    if float(singular_values[rank - 1].item()) <= float(cutoff.item()):
        return None, False
    return right_h[:rank].transpose(0, 1).contiguous(), True


def evaluate_fixed_action_subspace_confirmation(
    correct_features: Any,
    semantic_negative_features: Any,
    null_features: Any,
    *,
    confirmation_group_ids: tuple[str, ...],
    confirmation_spatial_sketch_id: str | None,
    confirmation_spatial_sketch_digest: str | None,
    confirmation_spatial_sketch_matrix: Any | None = None,
    confirmation_evidence_binding: EvidenceBinding | None = None,
    semantic_negative_labels: tuple[str, ...],
    expected_semantic_negative_labels: tuple[str, ...],
    action_subspace: FixedDiscoveryActionSubspace,
    nuisance_basis: TypedObservableNuisanceBasis,
    config: ConfirmationConfig = ConfirmationConfig(),
) -> ConfirmationMetrics:
    """Score confirmation against a fixed discovery basis and prototype.

    Label order is content binding: a permutation is a contract error rather
    than an opportunity to select the most favorable negative after the fact.
    """

    torch = _require_torch()
    if not isinstance(config, ConfirmationConfig):
        raise InternalTemporalQuotientError(
            "config must be a ConfirmationConfig"
        )
    config.validate()
    fixed = _validate_action_subspace(action_subspace)
    nuisance = _validate_nuisance_basis(nuisance_basis)
    if (
        fixed.nuisance_type_names != nuisance.type_names
        or fixed.nuisance_donor_group_ids != nuisance.donor_group_ids
        or fixed.nuisance_observation_digest != nuisance.observation_digest
        or fixed.nuisance_basis_digest != nuisance.basis_digest
        or fixed.nuisance_config_digest
        != canonical_nuisance_basis_config_digest(nuisance.config)
        or fixed.nuisance_leave_one_donor_passed
        != nuisance.leave_one_out_diagnostics.leave_one_donor_gate_passed
        or not torch.equal(fixed.nuisance_basis_snapshot, nuisance.basis)
    ):
        raise InternalTemporalQuotientError(
            "confirmation nuisance basis is not exact discovery-frozen evidence"
        )
    if confirmation_evidence_binding != fixed.evidence_binding:
        raise InternalTemporalQuotientError(
            "confirmation EvidenceBinding differs from discovery"
        )
    if confirmation_evidence_binding is not None:
        confirmation_evidence_binding.validate()
        if (
            canonical_confirmation_config_digest(config)
            != confirmation_evidence_binding.confirmation_config_digest
        ):
            raise InternalTemporalQuotientError(
                "EvidenceBinding confirmation config digest differs from runtime "
                "thresholds"
            )
    expected_labels = _validate_labels(
        expected_semantic_negative_labels,
        name="expected_semantic_negative_labels",
    )
    labels = _validate_labels(
        semantic_negative_labels,
        name="semantic_negative_labels",
        expected_count=len(expected_labels),
    )
    if labels != expected_labels:
        raise InternalTemporalQuotientError(
            "semantic-negative labels or their registered order do not match"
        )
    if expected_labels != fixed.discovery_diagnostics.semantic_negative_labels:
        raise InternalTemporalQuotientError(
            "confirmation negative binding differs from frozen discovery"
        )

    correct = _validate_feature_batch(
        correct_features,
        name="correct_features",
    )
    episodes = int(correct.shape[0])
    if int(correct.shape[2]) != fixed.spatial_descriptor_size:
        raise InternalTemporalQuotientError(
            "confirmation spatial descriptor size differs from discovery"
        )
    _validate_spatial_descriptor_binding(
        policy=fixed.spatial_descriptor_policy,
        sketch_id=confirmation_spatial_sketch_id,
        sketch_digest=confirmation_spatial_sketch_digest,
        evidence_profile=fixed.evidence_profile,
    )
    if (
        confirmation_spatial_sketch_id != fixed.spatial_sketch_id
        or confirmation_spatial_sketch_digest != fixed.spatial_sketch_digest
    ):
        raise InternalTemporalQuotientError(
            "confirmation signed-sketch binding differs from discovery"
        )
    _, confirmation_sketch_snapshot = _validate_spatial_sketch_matrix(
        confirmation_spatial_sketch_matrix,
        policy=fixed.spatial_descriptor_policy,
        expected_coordinates=fixed.spatial_descriptor_size,
        expected_source_positions=(
            SCIENTIFIC_PATCH_TOKENS
            if fixed.evidence_profile == "scientific"
            else None
        ),
        expected_digest=fixed.spatial_sketch_digest,
    )
    if (
        confirmation_sketch_snapshot is not None
        and not torch.equal(
            confirmation_sketch_snapshot,
            fixed.spatial_sketch_matrix_snapshot,
        )
    ):
        raise InternalTemporalQuotientError(
            "confirmation sketch matrix raw values differ from discovery"
        )
    if (
        fixed.evidence_profile == "scientific"
        and episodes < SCIENTIFIC_MIN_CONFIRMATION_EPISODES
    ):
        raise InternalTemporalQuotientError(
            "scientific confirmation requires at least four held-out episodes"
        )
    confirmation_ids = _validate_group_ids(
        confirmation_group_ids,
        name="confirmation_group_ids",
        expected_count=episodes,
    )
    overlap_ids = set(confirmation_ids).intersection(fixed.discovery_group_ids)
    if overlap_ids:
        raise InternalTemporalQuotientError(
            "discovery and confirmation factorial group IDs must be disjoint"
        )
    negatives = _validate_grouped_features(
        semantic_negative_features,
        name="semantic_negative_features",
        episodes=episodes,
    )
    nulls = _validate_grouped_features(
        null_features,
        name="null_features",
        episodes=episodes,
    )
    if int(negatives.shape[1]) != len(labels):
        raise InternalTemporalQuotientError(
            "semantic-negative group count does not match its labels"
        )
    for name, value in (
        ("semantic_negative_features", negatives),
        ("null_features", nulls),
        ("action_subspace.basis", fixed.basis),
    ):
        _same_storage_contract(correct, value, name=name)
    if int(negatives.shape[3]) != int(correct.shape[2]) or int(
        nulls.shape[3]
    ) != int(correct.shape[2]):
        raise InternalTemporalQuotientError(
            "confirmation groups must share the same spatial token count"
        )
    if confirmation_evidence_binding is not None:
        actual_confirmation_digest = canonical_fitq_split_content_digest(
            correct,
            negatives,
            nulls,
            split="confirmation",
            group_ids=confirmation_ids,
            semantic_negative_labels=labels,
            nuisance_observation_digest=nuisance.observation_digest,
            nuisance_basis_digest=nuisance.basis_digest,
            spatial_sketch_matrix_digest=fixed.spatial_sketch_digest,
            evidence_binding=confirmation_evidence_binding,
        )
        if (
            actual_confirmation_digest
            != confirmation_evidence_binding.confirmation_digest
        ):
            raise InternalTemporalQuotientError(
                "EvidenceBinding confirmation digest differs from exact split "
                "content"
            )

    correct_projection = project_observable_nuisance(correct, nuisance)
    negative_projection = project_observable_nuisance(negatives, nuisance)
    null_projection = project_observable_nuisance(nulls, nuisance)
    weighted_correct = weight_temporal_direct_sum(correct_projection.projected)
    weighted_negatives = weight_temporal_direct_sum(negative_projection.projected)
    weighted_nulls = weight_temporal_direct_sum(null_projection.projected)
    correct_coefficients = weighted_correct @ fixed.basis
    negative_coefficients = weighted_negatives @ fixed.basis
    null_coefficients = weighted_nulls @ fixed.basis

    correct_raw, correct_unit, correct_norm = _signed_action_descriptors(
        correct_coefficients,
        epsilon=config.epsilon,
        spatial_descriptor_policy=fixed.spatial_descriptor_policy,
    )
    negative_raw, negative_unit, _ = _signed_action_descriptors(
        negative_coefficients,
        epsilon=config.epsilon,
        spatial_descriptor_policy=fixed.spatial_descriptor_policy,
    )
    _, null_unit, _ = _signed_action_descriptors(
        null_coefficients,
        epsilon=config.epsilon,
        spatial_descriptor_policy=fixed.spatial_descriptor_policy,
    )
    correct_cosine = correct_unit @ fixed.prototype
    negative_cosine = negative_unit @ fixed.prototype
    null_cosine = null_unit @ fixed.prototype
    correct_signed_score = (correct_raw @ fixed.prototype) / torch.clamp(
        correct_norm,
        min=float(config.epsilon),
    )
    negative_relative_score = (negative_raw @ fixed.prototype) / torch.clamp(
        correct_norm[:, None],
        min=float(config.epsilon),
    )
    semantic_margins = correct_signed_score[:, None] - negative_relative_score
    null_cosine_margin = correct_cosine - null_cosine.max(dim=1).values

    correct_action_rms = torch.sqrt(
        correct_coefficients.square().mean(dim=(1, 2, 3))
    )
    negative_action_rms = torch.sqrt(
        negative_coefficients.square().mean(dim=(2, 3, 4))
    )
    null_action_rms = torch.sqrt(null_coefficients.square().mean(dim=(2, 3, 4)))
    # This is a frozen discovery statistic.  Confirmation nulls are reported
    # and used for cosine separation, but can never move the threshold.
    null_floor = fixed.frozen_null_floor.expand_as(correct_action_rms)

    diagnostic_basis, rank_passed = _diagnostic_confirmation_subspace(
        weighted_correct,
        rank=fixed.rank,
        rank_rtol=config.rank_rtol,
        epsilon=config.epsilon,
    )
    if rank_passed and diagnostic_basis is not None:
        overlap = fixed.basis.transpose(0, 1) @ diagnostic_basis
        grassmann_similarity = overlap.square().sum() / float(fixed.rank)
    else:
        grassmann_similarity = torch.zeros(
            (), dtype=correct.dtype, device=correct.device
        )

    correct_nonzero_passed = bool(
        (correct_norm > float(config.epsilon)).all().item()
    )
    correct_cosine_passed = bool(
        (correct_cosine >= float(config.min_correct_cosine)).all().item()
    )
    semantic_ordering_passed = bool(
        (semantic_margins >= float(config.min_semantic_margin)).all().item()
    )
    null_cosine_margin_passed = bool(
        (
            null_cosine_margin >= float(config.min_null_cosine_margin)
        ).all().item()
    )
    null_floor_passed = bool((correct_action_rms > null_floor).all().item())
    grassmann_passed = bool(
        rank_passed
        and float(grassmann_similarity.item())
        >= float(config.min_grassmann_similarity)
    )
    discovery_gate_passed = bool(
        fixed.discovery_diagnostics.discovery_gate_passed
    )
    scientific_profile_passed = bool(
        fixed.evidence_profile == "scientific"
        and fixed.scientific_local_discovery_eligible
        and episodes >= SCIENTIFIC_MIN_CONFIRMATION_EPISODES
    )
    local_geometry_eligible = all(
        (
            correct_nonzero_passed,
            correct_cosine_passed,
            semantic_ordering_passed,
            null_cosine_margin_passed,
            null_floor_passed,
            rank_passed,
            grassmann_passed,
            discovery_gate_passed,
            scientific_profile_passed,
        )
    )
    return ConfirmationMetrics(
        discovery_group_ids=fixed.discovery_group_ids,
        confirmation_group_ids=confirmation_ids,
        discovery_evidence_profile=fixed.evidence_profile,
        semantic_negative_labels=labels,
        correct_cosine=correct_cosine,
        semantic_negative_cosine=negative_cosine,
        null_cosine=null_cosine,
        semantic_margins=semantic_margins,
        null_cosine_margin=null_cosine_margin,
        correct_action_rms=correct_action_rms,
        semantic_negative_action_rms=negative_action_rms,
        null_action_rms=null_action_rms,
        null_floor=null_floor,
        grassmann_similarity=grassmann_similarity,
        correct_nonzero_passed=correct_nonzero_passed,
        correct_cosine_passed=correct_cosine_passed,
        semantic_ordering_passed=semantic_ordering_passed,
        null_cosine_margin_passed=null_cosine_margin_passed,
        null_floor_passed=null_floor_passed,
        confirmation_rank_passed=rank_passed,
        grassmann_passed=grassmann_passed,
        discovery_gate_passed=discovery_gate_passed,
        scientific_profile_passed=scientific_profile_passed,
        local_geometry_eligible=local_geometry_eligible,
        fitq_go_authorized=False,
    )


def internal_temporal_quotient_contract_receipt() -> dict[str, Any]:
    """Return the immutable, dependency-free FITQ tensor schema."""

    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_NAME,
        "hidden_layouts": ["B,21,H,W,1536", "B,21,P,1536"],
        "dtype": "float32",
        "input_state": "detached_and_finite",
        "temporal_bundle": {
            "causal_boundary": "H[t]-H[0]",
            "lags": list(TEMPORAL_LAGS),
            "lag_boundary": "exact_leading_zero_no_roll_no_wrap",
            "terminal_hold": "mean(causal_phases_17_through_20)",
            "terminal_hold_phase_indices": list(TERMINAL_HOLD_PHASES),
            "feature_steps": TEMPORAL_FEATURE_STEPS,
            "metric_weighting": "five_block_equal_mean_inner_product",
        },
        "hard_negatives": "label_bound_and_never_reduced_across_K",
        "nuisance": "typed_observable_column_normalized_thin_svd_projection",
        "nuisance_evidence_binding": {
            "identity": (
                "exact_raw_observation_fingerprint_plus_basis_raw_value_digest"
            ),
            "snapshot": "discovery_basis_values_frozen_and_torch_equal_at_confirmation",
            "minimum_scientific_donors_per_type": 2,
            "local_donor_gate": (
                "leave_one_donor_per_type_sign_invariant_subspace_axis_cosine"
            ),
            "type_out_gate": "explicit_verified_external_signed_evidence",
            "missing_external_audit": "local_geometry_ineligible",
        },
        "action_subspace": {
            "factorization": (
                "equal_weight_common_svd_of_positive_cells_and_every_"
                "labelled_positive_minus_negative_cell"
            ),
            "negative_reduction": "none_before_fit_or_gate",
            "prototype": "positive_signed_coordinates_only_then_frozen",
            "rms_role": "nondegeneracy_only_never_directional_rescue",
            "positive_consensus": "per_episode_and_leave_one_group_signed_cosine",
            "discovery_gate": "conjunction_over_every_episode_and_negative",
        },
        "evidence_profiles": {
            "engineering_micro": "diagnostics_only_never_scientific_eligible",
            "scientific": {
                "minimum_discovery_episodes": SCIENTIFIC_MIN_DISCOVERY_EPISODES,
                "minimum_confirmation_episodes": (
                    SCIENTIFIC_MIN_CONFIRMATION_EPISODES
                ),
                "spatial_sketch_coordinates": (
                    SCIENTIFIC_SPATIAL_SKETCH_COORDINATES
                ),
                "patch_grid": [SCIENTIFIC_PATCH_HEIGHT, SCIENTIFIC_PATCH_WIDTH],
                "patch_tokens": SCIENTIFIC_PATCH_TOKENS,
                "latent_geometry": list(SCIENTIFIC_LATENT_GEOMETRY),
                "patch_geometry": list(SCIENTIFIC_PATCH_GEOMETRY),
                "required_nuisance_types": list(
                    SCIENTIFIC_REQUIRED_NUISANCE_TYPES
                ),
                "required_negative_labels": list(
                    SCIENTIFIC_REQUIRED_NEGATIVE_LABELS
                ),
            },
        },
        "spatial_descriptors": {
            "global_mean_engineering": (
                "diagnostics_only_may_erase_localized_signed_action"
            ),
            "fixed_signed_sketches": {
                "directional_coordinates": "preserve_and_flatten_F_by_P_by_rank",
                "scientific_coordinates": SCIENTIFIC_SPATIAL_SKETCH_COORDINATES,
                "scientific_matrix_shape": [
                    SCIENTIFIC_SPATIAL_SKETCH_COORDINATES,
                    SCIENTIFIC_PATCH_TOKENS,
                ],
                "scientific_patch_grid": [
                    SCIENTIFIC_PATCH_HEIGHT,
                    SCIENTIFIC_PATCH_WIDTH,
                ],
                "construction": (
                    "preregistered_sha256_counter_rademacher_row_normalized_fp32"
                ),
                "raw_value_digest": (
                    "sha256(header='fitq-canonical-fp32-little-endian-v1|"
                    "shape=16,930|' + detached_contiguous_f32le_c_order_bytes)"
                ),
                "binding": (
                    "core_recomputes_exact_matrix_digest_and_freezes_values_"
                    "across_splits"
                ),
                "authentication_limit": (
                    "external_audit_must_verify_preregistration_and_generation_"
                    "algorithm"
                ),
            },
        },
        "evidence_binding": {
            "schema": EVIDENCE_BINDING_SCHEMA,
            "confirmation_match": "exact_dataclass_equality",
            "fields": [
                "checkpoint_tree_sha256",
                "bernini_revision",
                "veomni_revision",
                "discovery_mode",
                "confirmation_mode",
                "cross_mode_contract",
                "layer",
                "hook_site",
                "sigma_grid",
                "lambda_grid",
                "latent_geometry",
                "patch_geometry",
                "negative_label_set",
                "bank_digest",
                "upstream_query_receipt_digest",
                "nuisance_config_digest",
                "discovery_config_digest",
                "confirmation_config_digest",
                "discovery_digest",
                "confirmation_digest",
                "nuisance_audit_digest",
                "spatial_sketch_matrix_digest",
            ],
            "split_content_digest": {
                "scheme": "fitq-exact-split-content-v1-length-prefixed-sha256",
                "tensor_values": (
                    "core_computed_canonical_detached_fp32_shape_plus_f32le_bytes"
                ),
                "decision_policies": (
                    "exact_nuisance_discovery_confirmation_config_digests_"
                    "prevent_posthoc_threshold_relaxation"
                ),
                "discovery": (
                    "actual_positive_negative_null_tensors_plus_ordered_groups_"
                    "labels_nuisance_sketch_binding_and_bank"
                ),
                "confirmation": (
                    "actual_correct_negative_null_tensors_plus_ordered_groups_"
                    "labels_nuisance_sketch_binding_bank_and_discovery_digest"
                ),
                "external_bank_digest_scope": (
                    "externally_authenticated_artifact_sha_chained_but_not_"
                    "reconstructed_by_geometry_core"
                ),
                "upstream_query_scope": (
                    "same_state_owner_prompt_semantics_are_asserted_by_an_"
                    "externally_authenticated_upstream_query_receipt_and_only_"
                    "its_sha_is_chained_by_this_geometry_core"
                ),
            },
        },
        "semantic_content_audit": (
            "external_event_audit_required_labels_bind_structure_not_video_content"
        ),
        "null_floor": "fit_on_discovery_null_controls_then_frozen",
        "factorial_groups": "explicit_unique_discovery_confirmation_disjoint",
        "confirmation": [
            "signed_prototype_cosine",
            "grassmann_similarity",
            "per_negative_semantic_ordering",
            "null_floor",
        ],
        "decision_scope": {
            "reported_gate": "local_geometry_eligible",
            "fitq_go_authorized_always_false": True,
            "required_external_go_gates": [
                "engineering",
                "nondegeneracy",
                "semantic",
                "homotopy",
                "nuisance",
                "cross_mode",
                "frozen_causal",
            ],
        },
    }


__all__ = [
    "ABSENT_NUISANCE_AUDIT_DIGEST",
    "ConfirmationConfig",
    "ConfirmationMetrics",
    "DiscoverySubspaceConfig",
    "DiscoveryContrastDiagnostics",
    "DISCOVERY_EVIDENCE_PROFILES",
    "EVIDENCE_BINDING_SCHEMA",
    "EVIDENCE_CROSS_MODE_CONTRACTS",
    "EVIDENCE_HOOK_SITES",
    "EVIDENCE_MODES",
    "EvidenceBinding",
    "EXPECTED_HIDDEN_SIZE",
    "EXPECTED_PHASES",
    "FixedDiscoveryActionSubspace",
    "HardNegativeTemporalResiduals",
    "InternalTemporalQuotientError",
    "METHOD_NAME",
    "MIN_HARD_NEGATIVES",
    "NUISANCE_AUDIT_SCHEMA",
    "NuisanceAuditEvidence",
    "NuisanceBasisConfig",
    "NuisanceLeaveOneOutDiagnostics",
    "ObservableNuisanceProjection",
    "PINNED_BERNINI_REVISION",
    "PINNED_CHECKPOINT_TREE_SHA256",
    "PINNED_VEOMNI_REVISION",
    "SCHEMA_VERSION",
    "SCIENTIFIC_MIN_CONFIRMATION_EPISODES",
    "SCIENTIFIC_MIN_DISCOVERY_EPISODES",
    "SCIENTIFIC_LATENT_GEOMETRY",
    "SCIENTIFIC_PATCH_GEOMETRY",
    "SCIENTIFIC_PATCH_HEIGHT",
    "SCIENTIFIC_PATCH_TOKENS",
    "SCIENTIFIC_PATCH_WIDTH",
    "SCIENTIFIC_REQUIRED_NEGATIVE_LABELS",
    "SCIENTIFIC_REQUIRED_NUISANCE_TYPES",
    "SCIENTIFIC_SPATIAL_SKETCH_COORDINATES",
    "SPATIAL_DESCRIPTOR_POLICIES",
    "TEMPORAL_FEATURE_STEPS",
    "TEMPORAL_BLOCK_SPECS",
    "TEMPORAL_LAGS",
    "TERMINAL_HOLD_WINDOW",
    "TERMINAL_HOLD_PHASES",
    "TemporalBundle",
    "TypedObservableNuisanceBasis",
    "build_fixed_discovery_action_subspace",
    "build_temporal_bundle",
    "build_typed_observable_nuisance_basis",
    "canonical_fitq_split_content_digest",
    "canonical_confirmation_config_digest",
    "canonical_discovery_subspace_config_digest",
    "canonical_nuisance_basis_config_digest",
    "canonical_nuisance_audit_evidence_digest",
    "canonical_tensor_raw_value_digest",
    "compute_hard_negative_temporal_residuals",
    "evaluate_fixed_action_subspace_confirmation",
    "internal_temporal_quotient_contract_receipt",
    "project_observable_nuisance",
    "weight_temporal_direct_sum",
]
