"""Contrastive self-generated motion geometry for CAGE.

This module deliberately does *not* distil a generated T2V video into an
RV2V target.  It accepts only detached, CPU LoRA-B probe gradients computed
from a frozen T2V model.  For every event-qualified generated clip, the same
registered scalar probe must be evaluated on six counterfactual views:

``action, reverse, freeze, shuffle, camera, appearance``.

The first four views identify a temporal-order contrast.  The last two define
a nuisance span.  After removing that span, a block is retained only when:

* action-vs-freeze and action-vs-shuffle agree;
* the reverse contrast has the opposite sign and comparable norm;
* seed directions agree inside every identity;
* identity centroids agree inside every action family.

The surviving family centroids form a small empirical contrastive
``Motion-Fisher proxy``.  It is a PSD second moment of *contrast gradients*,
not statistical Fisher information.  Its eigenspace is computed through an
FP64 family-by-family Gram matrix.  The only downstream operation exposed by
the module is projection of a native RV2V action gradient onto that signless
subspace, plus an alignment audit.  No T2V pixel, latent, hidden state, noise,
or per-sample gradient is returned as an update target.

The implementation is intentionally CPU-only.  GPU gradients should be
copied to detached FP32 CPU vectors after the distributed LoRA-B reduction.
This keeps the small-matrix geometry deterministic and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

import torch


SCHEMA_VERSION = "bernini-cage-self-generated-motion-fisher-v1"
OBSERVATION_ORIGIN = "frozen_t2v_registered_scalar_lora_b_probe_v1"
RV2V_GRADIENT_ORIGIN = "native_rv2v_vi_cond_lora_b_action_gradient_v1"
REQUIRED_TRANSFORMS = (
    "action",
    "reverse",
    "freeze",
    "shuffle",
    "camera",
    "appearance",
)
BAND_SELECTION_RULE = "maximin_then_mean_then_earliest_exact_contiguous_band_v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NUMERICAL_EPSILON = 1.0e-12
_ORTHOGONALITY_TOLERANCE = 2.0e-8


class CAGEMotionFisherError(RuntimeError):
    """The registered gradient geometry is malformed or cannot be audited."""


@dataclass(frozen=True)
class MotionGradientObservation:
    """One frozen-T2V LoRA-B probe gradient and its closed metadata.

    ``event_receipt_digest`` must be identical for all transforms and blocks
    belonging to the same ``(family, identity, seed)`` generated clip.
    ``coordinate_digest`` binds the flattened B-parameter names, shapes,
    ordering, fixed A matrices, and probe revision for one block.
    """

    block_index: int
    action_family: str
    identity_key: str
    seed_key: str
    transform: str
    coordinate_digest: str
    event_receipt_digest: str
    event_qualified: bool
    origin: str
    gradient: torch.Tensor


@dataclass(frozen=True)
class MotionFisherRegistration:
    """Scientific thresholds that must be fixed before fitting the proxy."""

    candidate_block_indices: tuple[int, ...]
    required_action_families: tuple[str, ...]
    minimum_identity_count: int
    minimum_seeds_per_identity: int
    maximum_nuisance_rank: int
    nuisance_relative_eigenvalue_floor: float
    minimum_nuisance_residual_ratio: float
    minimum_temporal_order_cosine: float
    maximum_reverse_cosine: float
    minimum_reverse_norm_ratio: float
    maximum_reverse_norm_ratio: float
    minimum_seed_coherence: float
    minimum_identity_coherence: float
    minimum_identity_alignment: float
    maximum_motion_rank: int
    motion_relative_eigenvalue_floor: float
    motion_explained_variance_target: float
    minimum_rank_boundary_relative_gap: float


@dataclass(frozen=True)
class MotionSampleGate:
    block_index: int
    action_family: str
    identity_key: str
    seed_key: str
    nuisance_residual_ratio: Optional[float]
    temporal_order_cosine: Optional[float]
    reverse_cosine: Optional[float]
    reverse_norm_ratio: Optional[float]
    passed: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class FamilyMotionConsensus:
    action_family: str
    direction: torch.Tensor
    minimum_seed_coherence: float
    identity_coherence: float
    minimum_identity_alignment: float
    identity_count: int
    sample_count: int


@dataclass(frozen=True)
class BlockMotionSubspace:
    block_index: int
    coordinate_digest: str
    parameter_dimension: int
    qualified: bool
    rejection_reasons: tuple[str, ...]
    nuisance_basis: torch.Tensor
    motion_basis: torch.Tensor
    nuisance_eigenvalues: tuple[float, ...]
    motion_eigenvalues: tuple[float, ...]
    motion_rank: int
    explained_motion_fraction: float
    quality_score: float
    sample_gates: tuple[MotionSampleGate, ...]
    family_consensus: tuple[FamilyMotionConsensus, ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class MotionFisherFit:
    registration: MotionFisherRegistration
    blocks: tuple[BlockMotionSubspace, ...]
    qualified_block_indices: tuple[int, ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class RV2VGradientObservation:
    """A current native-RV2V student action gradient in the same B coordinates."""

    block_index: int
    action_family: str
    coordinate_digest: str
    student_state_receipt_digest: str
    branch_lock_receipt_digest: str
    origin: str
    gradient: torch.Tensor


@dataclass(frozen=True)
class RV2VMotionProjection:
    block_index: int
    action_family: str
    projected_gradient: torch.Tensor
    original_norm: float
    nuisance_clean_norm: float
    projected_norm: float
    projection_fraction: float
    nuisance_clean_projection_fraction: float
    signed_family_alignment: float
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class ContinuousBandRegistration:
    candidate_block_indices: tuple[int, ...]
    required_action_families: tuple[str, ...]
    exact_band_length: int
    minimum_fit_quality_score: float
    minimum_projection_fraction: float
    minimum_family_alignment: float
    selection_rule: str


@dataclass(frozen=True)
class ContinuousBandSelection:
    selected_block_indices: tuple[int, ...]
    band_selection_authorized: bool
    block_scores: tuple[tuple[int, float], ...]
    projections: tuple[RV2VMotionProjection, ...]
    block_reason: Optional[str]
    receipt: Mapping[str, Any]


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise CAGEMotionFisherError(
            "receipt is not canonical finite ASCII JSON"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    # A clone owns exactly the logical tensor bytes (no parent-storage prefix
    # or suffix).  Reading the CPU storage directly avoids a NumPy dependency,
    # which is important for lean audit environments and NumPy ABI mismatches.
    owned = value.detach().contiguous().clone()
    raw = bytes(owned.untyped_storage())
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise CAGEMotionFisherError(f"{label} must be a lowercase SHA256")
    return value


def _token(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not value.isascii()
        or any(character.isspace() for character in value)
    ):
        raise CAGEMotionFisherError(
            f"{label} must be a nonempty whitespace-free ASCII token"
        )
    return value


def _integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CAGEMotionFisherError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_range(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CAGEMotionFisherError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CAGEMotionFisherError(f"{label} must be a finite number")
    lower_bad = result < minimum if minimum_inclusive else result <= minimum
    upper_bad = result > maximum if maximum_inclusive else result >= maximum
    if lower_bad or upper_bad:
        left = "[" if minimum_inclusive else "("
        right = "]" if maximum_inclusive else ")"
        raise CAGEMotionFisherError(
            f"{label} must be in {left}{minimum}, {maximum}{right}"
        )
    return result


def _validate_sorted_unique_blocks(
    value: Any, *, label: str
) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise CAGEMotionFisherError(f"{label} must be a nonempty tuple")
    blocks = tuple(_integer(item, label=label, minimum=0) for item in value)
    if blocks != tuple(sorted(set(blocks))):
        raise CAGEMotionFisherError(f"{label} must be unique and ascending")
    return blocks


def _validate_sorted_unique_families(
    value: Any, *, label: str
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise CAGEMotionFisherError(f"{label} must be a nonempty tuple")
    families = tuple(_token(item, label=label) for item in value)
    if families != tuple(sorted(set(families))):
        raise CAGEMotionFisherError(
            f"{label} must be unique and lexicographically ascending"
        )
    return families


def _validate_registration(value: Any) -> MotionFisherRegistration:
    if not isinstance(value, MotionFisherRegistration):
        raise CAGEMotionFisherError(
            "registration must be MotionFisherRegistration"
        )
    _validate_sorted_unique_blocks(
        value.candidate_block_indices, label="candidate_block_indices"
    )
    _validate_sorted_unique_families(
        value.required_action_families, label="required_action_families"
    )
    _integer(value.minimum_identity_count, label="minimum_identity_count", minimum=2)
    _integer(
        value.minimum_seeds_per_identity,
        label="minimum_seeds_per_identity",
        minimum=2,
    )
    _integer(value.maximum_nuisance_rank, label="maximum_nuisance_rank", minimum=1)
    _finite_range(
        value.nuisance_relative_eigenvalue_floor,
        label="nuisance_relative_eigenvalue_floor",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    _finite_range(
        value.minimum_nuisance_residual_ratio,
        label="minimum_nuisance_residual_ratio",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    _finite_range(
        value.minimum_temporal_order_cosine,
        label="minimum_temporal_order_cosine",
        minimum=-1.0,
        maximum=1.0,
    )
    _finite_range(
        value.maximum_reverse_cosine,
        label="maximum_reverse_cosine",
        minimum=-1.0,
        maximum=0.0,
        maximum_inclusive=False,
    )
    min_ratio = _finite_range(
        value.minimum_reverse_norm_ratio,
        label="minimum_reverse_norm_ratio",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    max_ratio = _finite_range(
        value.maximum_reverse_norm_ratio,
        label="maximum_reverse_norm_ratio",
        minimum=1.0,
        maximum=float("inf"),
    )
    if min_ratio > max_ratio:
        raise CAGEMotionFisherError(
            "minimum_reverse_norm_ratio exceeds maximum_reverse_norm_ratio"
        )
    for label, scalar in (
        ("minimum_seed_coherence", value.minimum_seed_coherence),
        ("minimum_identity_coherence", value.minimum_identity_coherence),
        ("minimum_identity_alignment", value.minimum_identity_alignment),
    ):
        _finite_range(
            scalar,
            label=label,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
    _integer(value.maximum_motion_rank, label="maximum_motion_rank", minimum=1)
    _finite_range(
        value.motion_relative_eigenvalue_floor,
        label="motion_relative_eigenvalue_floor",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    _finite_range(
        value.motion_explained_variance_target,
        label="motion_explained_variance_target",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    _finite_range(
        value.minimum_rank_boundary_relative_gap,
        label="minimum_rank_boundary_relative_gap",
        minimum=0.0,
        maximum=1.0,
    )
    return value


def _validate_vector(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.dtype not in (torch.float32, torch.float64)
        or value.ndim != 1
        or value.numel() == 0
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise CAGEMotionFisherError(
            f"{label} must be a detached finite nonempty strided CPU FP32/FP64 vector"
        )
    return value.detach().contiguous()


def _registration_payload(value: MotionFisherRegistration) -> Mapping[str, Any]:
    return {
        "candidate_block_indices": list(value.candidate_block_indices),
        "required_action_families": list(value.required_action_families),
        "minimum_identity_count": value.minimum_identity_count,
        "minimum_seeds_per_identity": value.minimum_seeds_per_identity,
        "maximum_nuisance_rank": value.maximum_nuisance_rank,
        "nuisance_relative_eigenvalue_floor": value.nuisance_relative_eigenvalue_floor,
        "minimum_nuisance_residual_ratio": value.minimum_nuisance_residual_ratio,
        "minimum_temporal_order_cosine": value.minimum_temporal_order_cosine,
        "maximum_reverse_cosine": value.maximum_reverse_cosine,
        "minimum_reverse_norm_ratio": value.minimum_reverse_norm_ratio,
        "maximum_reverse_norm_ratio": value.maximum_reverse_norm_ratio,
        "minimum_seed_coherence": value.minimum_seed_coherence,
        "minimum_identity_coherence": value.minimum_identity_coherence,
        "minimum_identity_alignment": value.minimum_identity_alignment,
        "maximum_motion_rank": value.maximum_motion_rank,
        "motion_relative_eigenvalue_floor": value.motion_relative_eigenvalue_floor,
        "motion_explained_variance_target": value.motion_explained_variance_target,
        "minimum_rank_boundary_relative_gap": value.minimum_rank_boundary_relative_gap,
    }


def _validate_observations(
    observations: Any,
    registration: MotionFisherRegistration,
) -> tuple[
    dict[tuple[int, str, str, str], dict[str, torch.Tensor]],
    dict[int, str],
    dict[int, int],
    list[Mapping[str, Any]],
]:
    if not isinstance(observations, Sequence) or isinstance(
        observations, (str, bytes)
    ):
        raise CAGEMotionFisherError("observations must be a sequence")
    if not observations:
        raise CAGEMotionFisherError("observations cannot be empty")

    allowed_blocks = set(registration.candidate_block_indices)
    allowed_families = set(registration.required_action_families)
    grouped: dict[tuple[int, str, str, str], dict[str, torch.Tensor]] = {}
    coordinate_by_block: dict[int, str] = {}
    dimension_by_block: dict[int, int] = {}
    event_by_sample: dict[tuple[str, str, str], str] = {}
    manifest: list[Mapping[str, Any]] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, MotionGradientObservation):
            raise CAGEMotionFisherError(
                f"observations[{index}] must be MotionGradientObservation"
            )
        block = _integer(
            observation.block_index,
            label=f"observations[{index}].block_index",
            minimum=0,
        )
        if block not in allowed_blocks:
            raise CAGEMotionFisherError(
                f"observations[{index}] uses an unregistered block"
            )
        family = _token(
            observation.action_family,
            label=f"observations[{index}].action_family",
        )
        if family not in allowed_families:
            raise CAGEMotionFisherError(
                f"observations[{index}] uses an unregistered action family"
            )
        identity = _token(
            observation.identity_key,
            label=f"observations[{index}].identity_key",
        )
        seed = _token(
            observation.seed_key, label=f"observations[{index}].seed_key"
        )
        if observation.transform not in REQUIRED_TRANSFORMS:
            raise CAGEMotionFisherError(
                f"observations[{index}].transform is outside the closed registry"
            )
        if observation.origin != OBSERVATION_ORIGIN:
            raise CAGEMotionFisherError(
                f"observations[{index}].origin is not the frozen T2V probe origin"
            )
        if observation.event_qualified is not True:
            raise CAGEMotionFisherError(
                f"observations[{index}] is not event-qualified"
            )
        coordinate = _digest(
            observation.coordinate_digest,
            label=f"observations[{index}].coordinate_digest",
        )
        event = _digest(
            observation.event_receipt_digest,
            label=f"observations[{index}].event_receipt_digest",
        )
        gradient = _validate_vector(
            observation.gradient, label=f"observations[{index}].gradient"
        )
        prior_coordinate = coordinate_by_block.setdefault(block, coordinate)
        prior_dimension = dimension_by_block.setdefault(block, int(gradient.numel()))
        if prior_coordinate != coordinate or prior_dimension != int(gradient.numel()):
            raise CAGEMotionFisherError(
                f"block {block} does not have one coordinate digest and dimension"
            )
        sample = (family, identity, seed)
        prior_event = event_by_sample.setdefault(sample, event)
        if prior_event != event:
            raise CAGEMotionFisherError(
                f"sample {sample!r} does not share one event receipt across blocks"
            )
        key = (block, family, identity, seed)
        bucket = grouped.setdefault(key, {})
        if observation.transform in bucket:
            raise CAGEMotionFisherError(
                f"duplicate transform {observation.transform!r} for {key!r}"
            )
        bucket[observation.transform] = gradient
        manifest.append(
            {
                "block_index": block,
                "action_family": family,
                "identity_key": identity,
                "seed_key": seed,
                "transform": observation.transform,
                "coordinate_digest": coordinate,
                "event_receipt_digest": event,
                "gradient_dtype": str(gradient.dtype).removeprefix("torch."),
                "gradient_numel": int(gradient.numel()),
                "gradient_sha256": _tensor_sha256(gradient),
            }
        )

    expected_transforms = set(REQUIRED_TRANSFORMS)
    for key, bucket in grouped.items():
        if set(bucket) != expected_transforms:
            missing = sorted(expected_transforms - set(bucket))
            extra = sorted(set(bucket) - expected_transforms)
            raise CAGEMotionFisherError(
                f"sample {key!r} does not close the transform registry; "
                f"missing={missing}, extra={extra}"
            )

    support_by_block: dict[int, set[tuple[str, str, str]]] = {}
    for block, family, identity, seed in grouped:
        support_by_block.setdefault(block, set()).add((family, identity, seed))
    if set(support_by_block) != allowed_blocks:
        raise CAGEMotionFisherError("at least one registered block has no observations")
    reference_support = support_by_block[registration.candidate_block_indices[0]]
    if any(support != reference_support for support in support_by_block.values()):
        raise CAGEMotionFisherError(
            "all registered blocks must use the identical family/identity/seed support"
        )
    if {item[0] for item in reference_support} != allowed_families:
        raise CAGEMotionFisherError("the required action-family registry is incomplete")
    for family in registration.required_action_families:
        family_rows = [item for item in reference_support if item[0] == family]
        identities = sorted({item[1] for item in family_rows})
        if len(identities) < registration.minimum_identity_count:
            raise CAGEMotionFisherError(
                f"family {family!r} has fewer than the registered identity count"
            )
        for identity in identities:
            seeds = {item[2] for item in family_rows if item[1] == identity}
            if len(seeds) < registration.minimum_seeds_per_identity:
                raise CAGEMotionFisherError(
                    f"family {family!r}, identity {identity!r} has too few seeds"
                )
    manifest.sort(
        key=lambda row: (
            row["block_index"],
            row["action_family"],
            row["identity_key"],
            row["seed_key"],
            REQUIRED_TRANSFORMS.index(row["transform"]),
        )
    )
    return grouped, coordinate_by_block, dimension_by_block, manifest


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item())


def _unit(value: torch.Tensor) -> Optional[torch.Tensor]:
    norm = _norm(value)
    if norm <= _NUMERICAL_EPSILON:
        return None
    return value / norm


def _cosine(left: torch.Tensor, right: torch.Tensor) -> Optional[float]:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm <= _NUMERICAL_EPSILON or right_norm <= _NUMERICAL_EPSILON:
        return None
    result = float(torch.dot(left, right).item() / (left_norm * right_norm))
    return max(-1.0, min(1.0, result))


def _orthonormalize_columns_small_gram(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise CAGEMotionFisherError("basis candidate must be rank two")
    if value.shape[1] == 0:
        return value.detach().contiguous()
    gram = value.T @ value
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    if float(eigenvalues[0].item()) <= _NUMERICAL_EPSILON:
        raise CAGEMotionFisherError("basis candidate is rank deficient")
    inverse_root = eigenvectors @ torch.diag(torch.rsqrt(eigenvalues)) @ eigenvectors.T
    return (value @ inverse_root).detach().contiguous()


def _canonicalize_subspace(value: torch.Tensor) -> torch.Tensor:
    """Make a basis deterministic from its projector, including degeneracies."""

    orthogonal = _orthonormalize_columns_small_gram(value.to(torch.float64))
    dimension, rank = orthogonal.shape
    if rank == 0:
        return orthogonal
    remaining = torch.eye(rank, dtype=torch.float64)
    vectors: list[torch.Tensor] = []
    for _ in range(rank):
        # Leverage of every coordinate in the remaining subspace.  ``argmax``
        # chooses the lowest coordinate on an exact tie.
        row_projection = orthogonal @ remaining
        leverage = torch.sum(row_projection * orthogonal, dim=1)
        pivot = int(torch.argmax(leverage).item())
        coefficient = remaining @ orthogonal[pivot]
        coefficient_norm = _norm(coefficient)
        if coefficient_norm <= _NUMERICAL_EPSILON:
            raise CAGEMotionFisherError("cannot canonicalize a numerical subspace")
        coefficient = coefficient / coefficient_norm
        vector = orthogonal @ coefficient
        if float(vector[pivot].item()) < 0.0:
            coefficient = -coefficient
            vector = -vector
        vectors.append(vector)
        remaining = remaining - torch.outer(coefficient, coefficient)
        remaining = 0.5 * (remaining + remaining.T)
    result = _orthonormalize_columns_small_gram(torch.stack(vectors, dim=1))
    error = torch.max(
        torch.abs(result.T @ result - torch.eye(rank, dtype=torch.float64))
    )
    if float(error.item()) > _ORTHOGONALITY_TOLERANCE:
        raise CAGEMotionFisherError("canonical basis is not orthonormal")
    if result.shape != (dimension, rank):
        raise CAGEMotionFisherError("canonical basis changed geometry")
    return result.detach().contiguous()


def _basis_from_rows(
    rows: Sequence[torch.Tensor],
    *,
    relative_eigenvalue_floor: float,
    maximum_rank: int,
) -> tuple[torch.Tensor, tuple[float, ...], bool]:
    """Return the full significant row span via a small FP64 Gram matrix."""

    if not rows:
        raise CAGEMotionFisherError("basis rows cannot be empty")
    matrix = torch.stack(tuple(row.to(torch.float64) for row in rows), dim=0)
    dimension = int(matrix.shape[1])
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = torch.clamp(eigenvalues[order], min=0.0)
    eigenvectors = eigenvectors[:, order]
    top = float(eigenvalues[0].item()) if eigenvalues.numel() else 0.0
    if top <= _NUMERICAL_EPSILON:
        return (
            torch.zeros((dimension, 0), dtype=torch.float64),
            tuple(float(item) for item in eigenvalues.tolist()),
            False,
        )
    cutoff = max(_NUMERICAL_EPSILON, top * relative_eigenvalue_floor)
    rank = int(torch.sum(eigenvalues >= cutoff).item())
    if rank > maximum_rank:
        return (
            torch.zeros((dimension, 0), dtype=torch.float64),
            tuple(float(item) for item in eigenvalues.tolist()),
            True,
        )
    selected_values = eigenvalues[:rank]
    selected_vectors = eigenvectors[:, :rank]
    raw = matrix.T @ selected_vectors @ torch.diag(torch.rsqrt(selected_values))
    return (
        _canonicalize_subspace(raw),
        tuple(float(item) for item in eigenvalues.tolist()),
        False,
    )


def _project_out(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    value = value.to(torch.float64)
    if basis.shape[1] == 0:
        return value.detach().contiguous()
    return (value - basis @ (basis.T @ value)).detach().contiguous()


def _motion_fisher_basis(
    family_directions: Sequence[torch.Tensor],
    registration: MotionFisherRegistration,
) -> tuple[Optional[torch.Tensor], tuple[float, ...], int, float, Optional[str]]:
    matrix = torch.stack(
        tuple(direction.to(torch.float64) for direction in family_directions), dim=0
    )
    family_count = int(matrix.shape[0])
    gram = (matrix @ matrix.T) / float(family_count)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = torch.clamp(eigenvalues[order], min=0.0)
    eigenvectors = eigenvectors[:, order]
    eigenvalue_tuple = tuple(float(item) for item in eigenvalues.tolist())
    top = float(eigenvalues[0].item()) if eigenvalues.numel() else 0.0
    total = float(torch.sum(eigenvalues).item())
    if top <= _NUMERICAL_EPSILON or total <= _NUMERICAL_EPSILON:
        return None, eigenvalue_tuple, 0, 0.0, "zero_motion_fisher_proxy"
    cutoff = max(
        _NUMERICAL_EPSILON,
        top * registration.motion_relative_eigenvalue_floor,
    )
    eligible = int(torch.sum(eigenvalues >= cutoff).item())
    maximum = min(eligible, registration.maximum_motion_rank)
    selected_rank = 0
    explained = 0.0
    for rank in range(1, maximum + 1):
        explained = float(torch.sum(eigenvalues[:rank]).item() / total)
        if explained >= registration.motion_explained_variance_target:
            selected_rank = rank
            break
    if selected_rank == 0:
        return (
            None,
            eigenvalue_tuple,
            0,
            explained,
            "motion_rank_cannot_reach_registered_explained_fraction",
        )
    if selected_rank < int(eigenvalues.numel()):
        boundary = float(eigenvalues[selected_rank - 1].item())
        next_value = float(eigenvalues[selected_rank].item())
        if next_value > _NUMERICAL_EPSILON:
            gap = (boundary - next_value) / max(boundary, _NUMERICAL_EPSILON)
            if gap < registration.minimum_rank_boundary_relative_gap:
                return (
                    None,
                    eigenvalue_tuple,
                    0,
                    explained,
                    "ambiguous_motion_rank_boundary",
                )
    values = eigenvalues[:selected_rank]
    vectors = eigenvectors[:, :selected_rank]
    raw = matrix.T @ vectors @ torch.diag(
        torch.rsqrt(values * float(family_count))
    )
    basis = _canonicalize_subspace(raw)
    return basis, eigenvalue_tuple, selected_rank, explained, None


def _rejected_block(
    *,
    block: int,
    coordinate: str,
    dimension: int,
    nuisance_basis: torch.Tensor,
    nuisance_eigenvalues: tuple[float, ...],
    reasons: Sequence[str],
    sample_gates: Sequence[MotionSampleGate],
    family_consensus: Sequence[FamilyMotionConsensus],
    observation_manifest_digest: str,
) -> BlockMotionSubspace:
    unique_reasons = tuple(sorted(set(reasons)))
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "block_index": block,
        "coordinate_digest": coordinate,
        "parameter_dimension": dimension,
        "qualified": False,
        "rejection_reasons": list(unique_reasons),
        "nuisance_rank": int(nuisance_basis.shape[1]),
        "nuisance_basis_sha256": _tensor_sha256(nuisance_basis),
        "motion_rank": 0,
        "motion_basis_sha256": _tensor_sha256(
            torch.zeros((dimension, 0), dtype=torch.float64)
        ),
        "quality_score": 0.0,
        "observation_manifest_digest": observation_manifest_digest,
        "optimizer_update_authorized": False,
    }
    receipt = {**unsigned, "digest": _object_sha256(unsigned)}
    return BlockMotionSubspace(
        block_index=block,
        coordinate_digest=coordinate,
        parameter_dimension=dimension,
        qualified=False,
        rejection_reasons=unique_reasons,
        nuisance_basis=nuisance_basis.detach().contiguous(),
        motion_basis=torch.zeros((dimension, 0), dtype=torch.float64),
        nuisance_eigenvalues=nuisance_eigenvalues,
        motion_eigenvalues=(),
        motion_rank=0,
        explained_motion_fraction=0.0,
        quality_score=0.0,
        sample_gates=tuple(sample_gates),
        family_consensus=tuple(family_consensus),
        receipt=receipt,
    )


def fit_self_generated_motion_fisher(
    observations: Sequence[MotionGradientObservation],
    registration: MotionFisherRegistration,
) -> MotionFisherFit:
    """Fit nuisance-orthogonal per-block contrastive motion subspaces.

    Scientific gate failures reject only the affected block and return an
    auditable result.  Malformed or incomplete contracts raise immediately.
    Neither outcome authorizes an optimizer update.
    """

    registration = _validate_registration(registration)
    grouped, coordinates, dimensions, manifest = _validate_observations(
        observations, registration
    )
    manifest_digest = _object_sha256(manifest)
    blocks: list[BlockMotionSubspace] = []
    for block in registration.candidate_block_indices:
        keys = sorted(key for key in grouped if key[0] == block)
        nuisance_rows: list[torch.Tensor] = []
        for key in keys:
            rows = grouped[key]
            neutral = 0.5 * (
                rows["freeze"].to(torch.float64)
                + rows["shuffle"].to(torch.float64)
            )
            nuisance_rows.append(rows["camera"].to(torch.float64) - neutral)
            nuisance_rows.append(rows["appearance"].to(torch.float64) - neutral)
        nuisance_basis, nuisance_eigenvalues, nuisance_rank_overflow = (
            _basis_from_rows(
                nuisance_rows,
                relative_eigenvalue_floor=registration.nuisance_relative_eigenvalue_floor,
                maximum_rank=registration.maximum_nuisance_rank,
            )
        )
        if nuisance_rank_overflow:
            blocks.append(
                _rejected_block(
                    block=block,
                    coordinate=coordinates[block],
                    dimension=dimensions[block],
                    nuisance_basis=nuisance_basis,
                    nuisance_eigenvalues=nuisance_eigenvalues,
                    reasons=("nuisance_rank_exceeds_registered_maximum",),
                    sample_gates=(),
                    family_consensus=(),
                    observation_manifest_digest=manifest_digest,
                )
            )
            continue

        block_reasons: list[str] = []
        sample_gates: list[MotionSampleGate] = []
        units_by_family_identity: dict[str, dict[str, list[torch.Tensor]]] = {}
        gate_quality_values: list[float] = []
        for key in keys:
            _, family, identity, seed = key
            rows = grouped[key]
            action = rows["action"].to(torch.float64)
            reverse = rows["reverse"].to(torch.float64)
            freeze = rows["freeze"].to(torch.float64)
            shuffle = rows["shuffle"].to(torch.float64)
            action_freeze = _project_out(action - freeze, nuisance_basis)
            action_shuffle = _project_out(action - shuffle, nuisance_basis)
            reverse_freeze = _project_out(reverse - freeze, nuisance_basis)
            reverse_shuffle = _project_out(reverse - shuffle, nuisance_basis)
            forward = 0.5 * (action_freeze + action_shuffle)
            reverse_direction = 0.5 * (reverse_freeze + reverse_shuffle)
            raw_forward = action - 0.5 * (freeze + shuffle)
            raw_norm = _norm(raw_forward)
            forward_norm = _norm(forward)
            reverse_norm = _norm(reverse_direction)
            residual_ratio = (
                forward_norm / raw_norm
                if raw_norm > _NUMERICAL_EPSILON
                else None
            )
            action_order_cosine = _cosine(action_freeze, action_shuffle)
            reverse_order_cosine = _cosine(reverse_freeze, reverse_shuffle)
            temporal_cosine = (
                min(action_order_cosine, reverse_order_cosine)
                if action_order_cosine is not None
                and reverse_order_cosine is not None
                else None
            )
            reverse_cosine = _cosine(forward, reverse_direction)
            reverse_ratio = (
                reverse_norm / forward_norm
                if forward_norm > _NUMERICAL_EPSILON
                else None
            )
            reasons: list[str] = []
            if residual_ratio is None:
                reasons.append("zero_raw_action_temporal_contrast")
            elif residual_ratio < registration.minimum_nuisance_residual_ratio:
                reasons.append("action_contrast_collapses_into_nuisance_span")
            if temporal_cosine is None:
                reasons.append("zero_temporal_order_contrast")
            elif temporal_cosine < registration.minimum_temporal_order_cosine:
                reasons.append("freeze_shuffle_temporal_order_disagreement")
            if reverse_cosine is None:
                reasons.append("zero_reverse_contrast")
            elif reverse_cosine > registration.maximum_reverse_cosine:
                reasons.append("reverse_is_not_opposite_signed")
            if reverse_ratio is None:
                reasons.append("undefined_reverse_norm_ratio")
            elif not (
                registration.minimum_reverse_norm_ratio
                <= reverse_ratio
                <= registration.maximum_reverse_norm_ratio
            ):
                reasons.append("reverse_norm_ratio_outside_registered_range")
            unit = _unit(forward)
            if unit is None:
                reasons.append("zero_nuisance_clean_forward_direction")
            if not reasons and unit is not None:
                units_by_family_identity.setdefault(family, {}).setdefault(
                    identity, []
                ).append(unit)
                gate_quality_values.extend(
                    (
                        float(residual_ratio),
                        0.5 * (float(temporal_cosine) + 1.0),
                        0.5 * (1.0 - float(reverse_cosine)),
                        min(float(reverse_ratio), 1.0 / float(reverse_ratio)),
                    )
                )
            prefixed = tuple(f"{family}/{identity}/{seed}:{item}" for item in reasons)
            block_reasons.extend(prefixed)
            sample_gates.append(
                MotionSampleGate(
                    block_index=block,
                    action_family=family,
                    identity_key=identity,
                    seed_key=seed,
                    nuisance_residual_ratio=residual_ratio,
                    temporal_order_cosine=temporal_cosine,
                    reverse_cosine=reverse_cosine,
                    reverse_norm_ratio=reverse_ratio,
                    passed=not reasons,
                    rejection_reasons=tuple(reasons),
                )
            )

        family_consensus: list[FamilyMotionConsensus] = []
        consensus_quality_values: list[float] = []
        for family in registration.required_action_families:
            identity_rows = units_by_family_identity.get(family, {})
            identity_centroids: list[torch.Tensor] = []
            seed_coherences: list[float] = []
            sample_count = 0
            for identity in sorted(identity_rows):
                seed_units = identity_rows[identity]
                sample_count += len(seed_units)
                mean = torch.mean(torch.stack(seed_units, dim=0), dim=0)
                coherence = _norm(mean)
                seed_coherences.append(coherence)
                centroid = _unit(mean)
                if centroid is None:
                    block_reasons.append(
                        f"{family}/{identity}:zero_seed_consensus"
                    )
                else:
                    identity_centroids.append(centroid)
                if coherence < registration.minimum_seed_coherence:
                    block_reasons.append(
                        f"{family}/{identity}:seed_coherence_below_registered_floor"
                    )
            if len(identity_centroids) < registration.minimum_identity_count:
                block_reasons.append(
                    f"{family}:insufficient_identity_consensus_after_sample_gates"
                )
                continue
            identity_mean = torch.mean(torch.stack(identity_centroids, dim=0), dim=0)
            identity_coherence = _norm(identity_mean)
            direction = _unit(identity_mean)
            if direction is None:
                block_reasons.append(f"{family}:zero_cross_identity_consensus")
                continue
            alignments = [
                float(torch.dot(centroid, direction).item())
                for centroid in identity_centroids
            ]
            minimum_alignment = min(alignments)
            if identity_coherence < registration.minimum_identity_coherence:
                block_reasons.append(
                    f"{family}:identity_coherence_below_registered_floor"
                )
            if minimum_alignment < registration.minimum_identity_alignment:
                block_reasons.append(
                    f"{family}:identity_alignment_below_registered_floor"
                )
            minimum_seed = min(seed_coherences) if seed_coherences else 0.0
            consensus_quality_values.extend(
                (minimum_seed, identity_coherence, minimum_alignment)
            )
            family_consensus.append(
                FamilyMotionConsensus(
                    action_family=family,
                    direction=direction.detach().contiguous(),
                    minimum_seed_coherence=minimum_seed,
                    identity_coherence=identity_coherence,
                    minimum_identity_alignment=minimum_alignment,
                    identity_count=len(identity_centroids),
                    sample_count=sample_count,
                )
            )

        if block_reasons or len(family_consensus) != len(
            registration.required_action_families
        ):
            blocks.append(
                _rejected_block(
                    block=block,
                    coordinate=coordinates[block],
                    dimension=dimensions[block],
                    nuisance_basis=nuisance_basis,
                    nuisance_eigenvalues=nuisance_eigenvalues,
                    reasons=block_reasons
                    or ("required_family_consensus_is_incomplete",),
                    sample_gates=sample_gates,
                    family_consensus=family_consensus,
                    observation_manifest_digest=manifest_digest,
                )
            )
            continue

        motion_basis, motion_eigenvalues, motion_rank, explained, fisher_reason = (
            _motion_fisher_basis(
                [item.direction for item in family_consensus], registration
            )
        )
        if fisher_reason is not None or motion_basis is None:
            blocks.append(
                _rejected_block(
                    block=block,
                    coordinate=coordinates[block],
                    dimension=dimensions[block],
                    nuisance_basis=nuisance_basis,
                    nuisance_eigenvalues=nuisance_eigenvalues,
                    reasons=(fisher_reason or "unknown_motion_fisher_failure",),
                    sample_gates=sample_gates,
                    family_consensus=family_consensus,
                    observation_manifest_digest=manifest_digest,
                )
            )
            continue
        nuisance_overlap = (
            torch.max(torch.abs(nuisance_basis.T @ motion_basis))
            if nuisance_basis.shape[1] and motion_basis.shape[1]
            else torch.zeros((), dtype=torch.float64)
        )
        if float(nuisance_overlap.item()) > _ORTHOGONALITY_TOLERANCE:
            raise CAGEMotionFisherError(
                f"block {block} motion basis is not nuisance-orthogonal"
            )
        quality_score = min(
            gate_quality_values + consensus_quality_values + [explained]
        )
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "block_index": block,
            "coordinate_digest": coordinates[block],
            "parameter_dimension": dimensions[block],
            "qualified": True,
            "rejection_reasons": [],
            "nuisance_rank": int(nuisance_basis.shape[1]),
            "nuisance_basis_sha256": _tensor_sha256(nuisance_basis),
            "motion_rank": motion_rank,
            "motion_basis_sha256": _tensor_sha256(motion_basis),
            "family_direction_sha256": {
                item.action_family: _tensor_sha256(item.direction)
                for item in family_consensus
            },
            "motion_eigenvalues": list(motion_eigenvalues),
            "explained_motion_fraction": explained,
            "quality_score": quality_score,
            "observation_manifest_digest": manifest_digest,
            "fisher_interpretation": "empirical_contrastive_psd_second_moment_not_statistical_fisher",
            "optimizer_update_authorized": False,
        }
        receipt = {**unsigned, "digest": _object_sha256(unsigned)}
        blocks.append(
            BlockMotionSubspace(
                block_index=block,
                coordinate_digest=coordinates[block],
                parameter_dimension=dimensions[block],
                qualified=True,
                rejection_reasons=(),
                nuisance_basis=nuisance_basis.detach().contiguous(),
                motion_basis=motion_basis.detach().contiguous(),
                nuisance_eigenvalues=nuisance_eigenvalues,
                motion_eigenvalues=motion_eigenvalues,
                motion_rank=motion_rank,
                explained_motion_fraction=explained,
                quality_score=quality_score,
                sample_gates=tuple(sample_gates),
                family_consensus=tuple(family_consensus),
                receipt=receipt,
            )
        )

    qualified = tuple(item.block_index for item in blocks if item.qualified)
    unsigned_fit = {
        "schema_version": SCHEMA_VERSION,
        "registration": _registration_payload(registration),
        "registration_digest": _object_sha256(_registration_payload(registration)),
        "observation_manifest_digest": manifest_digest,
        "observation_count": len(manifest),
        "qualified_block_indices": list(qualified),
        "block_receipt_digests": [item.receipt["digest"] for item in blocks],
        "input_media_accepted": False,
        "optimizer_update_authorized": False,
    }
    fit_receipt = {**unsigned_fit, "digest": _object_sha256(unsigned_fit)}
    return MotionFisherFit(
        registration=registration,
        blocks=tuple(blocks),
        qualified_block_indices=qualified,
        receipt=fit_receipt,
    )


def _block(fit: MotionFisherFit, block_index: int) -> BlockMotionSubspace:
    if not isinstance(fit, MotionFisherFit):
        raise CAGEMotionFisherError("fit must be MotionFisherFit")
    matches = [item for item in fit.blocks if item.block_index == block_index]
    if len(matches) != 1:
        raise CAGEMotionFisherError("RV2V gradient block is absent from the fit")
    if not matches[0].qualified:
        raise CAGEMotionFisherError("RV2V projection is closed for a rejected block")
    return matches[0]


def _family(
    block: BlockMotionSubspace, action_family: str
) -> FamilyMotionConsensus:
    matches = [
        item for item in block.family_consensus if item.action_family == action_family
    ]
    if len(matches) != 1:
        raise CAGEMotionFisherError(
            "RV2V gradient family is absent from the fitted block"
        )
    return matches[0]


def _validate_rv2v_observation(
    value: Any, *, label: str
) -> tuple[RV2VGradientObservation, torch.Tensor]:
    if not isinstance(value, RV2VGradientObservation):
        raise CAGEMotionFisherError(f"{label} must be RV2VGradientObservation")
    _integer(value.block_index, label=f"{label}.block_index", minimum=0)
    _token(value.action_family, label=f"{label}.action_family")
    _digest(value.coordinate_digest, label=f"{label}.coordinate_digest")
    _digest(
        value.student_state_receipt_digest,
        label=f"{label}.student_state_receipt_digest",
    )
    _digest(
        value.branch_lock_receipt_digest,
        label=f"{label}.branch_lock_receipt_digest",
    )
    if value.origin != RV2V_GRADIENT_ORIGIN:
        raise CAGEMotionFisherError(f"{label}.origin is not native RV2V VI_cond")
    gradient = _validate_vector(value.gradient, label=f"{label}.gradient")
    return value, gradient


def project_native_rv2v_gradient(
    fit: MotionFisherFit,
    observation: RV2VGradientObservation,
) -> RV2VMotionProjection:
    """Project an RV2V-owned gradient; never synthesize an update from T2V."""

    observation, gradient = _validate_rv2v_observation(
        observation, label="observation"
    )
    block = _block(fit, observation.block_index)
    family = _family(block, observation.action_family)
    if observation.coordinate_digest != block.coordinate_digest:
        raise CAGEMotionFisherError("RV2V and T2V probe coordinates are not identical")
    if int(gradient.numel()) != block.parameter_dimension:
        raise CAGEMotionFisherError("RV2V gradient dimension differs from the fit")
    value = gradient.to(torch.float64)
    original_norm = _norm(value)
    if original_norm <= _NUMERICAL_EPSILON:
        raise CAGEMotionFisherError("RV2V gradient is numerically zero")
    nuisance_clean = _project_out(value, block.nuisance_basis)
    projected = block.motion_basis @ (block.motion_basis.T @ nuisance_clean)
    nuisance_clean_norm = _norm(nuisance_clean)
    projected_norm = _norm(projected)
    projection_fraction = projected_norm / original_norm
    nuisance_clean_fraction = (
        projected_norm / nuisance_clean_norm
        if nuisance_clean_norm > _NUMERICAL_EPSILON
        else 0.0
    )
    alignment = _cosine(projected, family.direction)
    signed_alignment = float(alignment) if alignment is not None else 0.0
    output_dtype = gradient.dtype
    projected_output = projected.to(output_dtype).detach().contiguous()
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "block_index": block.block_index,
        "action_family": observation.action_family,
        "coordinate_digest": block.coordinate_digest,
        "student_state_receipt_digest": observation.student_state_receipt_digest,
        "branch_lock_receipt_digest": observation.branch_lock_receipt_digest,
        "fit_receipt_digest": fit.receipt["digest"],
        "input_gradient_sha256": _tensor_sha256(gradient),
        "projected_gradient_sha256": _tensor_sha256(projected_output),
        "original_norm": original_norm,
        "nuisance_clean_norm": nuisance_clean_norm,
        "projected_norm": projected_norm,
        "projection_fraction": projection_fraction,
        "nuisance_clean_projection_fraction": nuisance_clean_fraction,
        "signed_family_alignment": signed_alignment,
        "projection_operator": "signless_motion_subspace_projector",
        "optimizer_update_authorized": False,
    }
    receipt = {**unsigned, "digest": _object_sha256(unsigned)}
    return RV2VMotionProjection(
        block_index=block.block_index,
        action_family=observation.action_family,
        projected_gradient=projected_output,
        original_norm=original_norm,
        nuisance_clean_norm=nuisance_clean_norm,
        projected_norm=projected_norm,
        projection_fraction=projection_fraction,
        nuisance_clean_projection_fraction=nuisance_clean_fraction,
        signed_family_alignment=signed_alignment,
        receipt=receipt,
    )


def _validate_band_registration(
    value: Any, fit: MotionFisherFit
) -> ContinuousBandRegistration:
    if not isinstance(value, ContinuousBandRegistration):
        raise CAGEMotionFisherError(
            "band registration must be ContinuousBandRegistration"
        )
    blocks = _validate_sorted_unique_blocks(
        value.candidate_block_indices, label="band.candidate_block_indices"
    )
    families = _validate_sorted_unique_families(
        value.required_action_families, label="band.required_action_families"
    )
    if blocks != fit.registration.candidate_block_indices:
        raise CAGEMotionFisherError(
            "band candidate blocks differ from the pre-registered fit blocks"
        )
    if families != fit.registration.required_action_families:
        raise CAGEMotionFisherError(
            "band action families differ from the pre-registered fit families"
        )
    length = _integer(
        value.exact_band_length, label="band.exact_band_length", minimum=2
    )
    if length > len(blocks):
        raise CAGEMotionFisherError("exact band length exceeds candidate blocks")
    for label, scalar in (
        ("minimum_fit_quality_score", value.minimum_fit_quality_score),
        ("minimum_projection_fraction", value.minimum_projection_fraction),
        ("minimum_family_alignment", value.minimum_family_alignment),
    ):
        _finite_range(
            scalar,
            label=f"band.{label}",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
    if value.selection_rule != BAND_SELECTION_RULE:
        raise CAGEMotionFisherError("band selection rule is not the closed rule")
    return value


def select_registered_contiguous_block_band(
    fit: MotionFisherFit,
    rv2v_gradients: Sequence[RV2VGradientObservation],
    registration: ContinuousBandRegistration,
) -> ContinuousBandSelection:
    """Select one exact-length contiguous band by a pre-registered rule.

    A selected band only authorizes the *module band*.  The returned receipt
    explicitly does not authorize an optimizer step; CAGE source-safe QP and
    decoded rollback gates remain mandatory.
    """

    if not isinstance(fit, MotionFisherFit):
        raise CAGEMotionFisherError("fit must be MotionFisherFit")
    registration = _validate_band_registration(registration, fit)
    if not isinstance(rv2v_gradients, Sequence) or isinstance(
        rv2v_gradients, (str, bytes)
    ):
        raise CAGEMotionFisherError("rv2v_gradients must be a sequence")
    expected = {
        (block, family)
        for block in registration.candidate_block_indices
        for family in registration.required_action_families
    }
    indexed: dict[tuple[int, str], RV2VGradientObservation] = {}
    branch_digests: set[str] = set()
    state_digest_by_family: dict[str, str] = {}
    for index, raw in enumerate(rv2v_gradients):
        observation, _ = _validate_rv2v_observation(
            raw, label=f"rv2v_gradients[{index}]"
        )
        key = (observation.block_index, observation.action_family)
        if key not in expected:
            raise CAGEMotionFisherError("RV2V gradient is outside the closed registry")
        if key in indexed:
            raise CAGEMotionFisherError("duplicate RV2V block/family gradient")
        indexed[key] = observation
        branch_digests.add(observation.branch_lock_receipt_digest)
        prior_state = state_digest_by_family.setdefault(
            observation.action_family,
            observation.student_state_receipt_digest,
        )
        if prior_state != observation.student_state_receipt_digest:
            raise CAGEMotionFisherError(
                "one action family must use the same RV2V student state across blocks"
            )
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise CAGEMotionFisherError(
            f"RV2V block/family gradient registry is incomplete: {missing}"
        )
    if len(branch_digests) != 1:
        raise CAGEMotionFisherError(
            "all RV2V gradients must share one branch-lock receipt"
        )

    projections: list[RV2VMotionProjection] = []
    scores: dict[int, float] = {}
    eligible: set[int] = set()
    for block in fit.blocks:
        if not block.qualified:
            scores[block.block_index] = 0.0
            continue
        family_projections = [
            project_native_rv2v_gradient(
                fit, indexed[(block.block_index, family)]
            )
            for family in registration.required_action_families
        ]
        projections.extend(family_projections)
        score = min(
            [block.quality_score]
            + [item.projection_fraction for item in family_projections]
            + [max(0.0, item.signed_family_alignment) for item in family_projections]
        )
        scores[block.block_index] = score
        if (
            block.quality_score >= registration.minimum_fit_quality_score
            and all(
                item.projection_fraction >= registration.minimum_projection_fraction
                and item.signed_family_alignment
                >= registration.minimum_family_alignment
                for item in family_projections
            )
        ):
            eligible.add(block.block_index)

    candidates: list[tuple[float, float, int, tuple[int, ...]]] = []
    blocks = registration.candidate_block_indices
    length = registration.exact_band_length
    for offset in range(0, len(blocks) - length + 1):
        window = blocks[offset : offset + length]
        if any(right != left + 1 for left, right in zip(window, window[1:])):
            continue
        if not set(window).issubset(eligible):
            continue
        values = [scores[item] for item in window]
        candidates.append((min(values), sum(values) / len(values), window[0], window))
    if candidates:
        # Highest worst-block score, then highest mean, then earliest start.
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected = candidates[0][3]
        authorized = True
        reason = None
    else:
        selected = ()
        authorized = False
        reason = "no_exact_contiguous_band_passes_all_registered_gates"

    block_scores = tuple((block, float(scores[block])) for block in blocks)
    band_payload = {
        "candidate_block_indices": list(registration.candidate_block_indices),
        "required_action_families": list(registration.required_action_families),
        "exact_band_length": registration.exact_band_length,
        "minimum_fit_quality_score": registration.minimum_fit_quality_score,
        "minimum_projection_fraction": registration.minimum_projection_fraction,
        "minimum_family_alignment": registration.minimum_family_alignment,
        "selection_rule": registration.selection_rule,
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "fit_receipt_digest": fit.receipt["digest"],
        "band_registration": band_payload,
        "band_registration_digest": _object_sha256(band_payload),
        "selected_block_indices": list(selected),
        "band_selection_authorized": authorized,
        "block_scores": [[block, score] for block, score in block_scores],
        "projection_receipt_digests": [
            item.receipt["digest"] for item in projections
        ],
        "block_reason": reason,
        "optimizer_update_authorized": False,
    }
    receipt = {**unsigned, "digest": _object_sha256(unsigned)}
    return ContinuousBandSelection(
        selected_block_indices=selected,
        band_selection_authorized=authorized,
        block_scores=block_scores,
        projections=tuple(projections),
        block_reason=reason,
        receipt=receipt,
    )


def contract_receipt() -> Mapping[str, Any]:
    """Return the closed scientific boundary of this geometry core."""

    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "accepted_payload": "detached_cpu_lora_b_probe_gradient_vectors_and_closed_metadata",
        "required_transforms": list(REQUIRED_TRANSFORMS),
        "teacher_role": "identify_a_signless_cross_identity_motion_parameter_subspace",
        "fisher_interpretation": "empirical_contrastive_psd_second_moment_not_statistical_fisher",
        "rv2v_role": "supply_the_only_gradient_that_can_be_projected_downstream",
        "forbidden_carriers": [
            "t2v_pixel",
            "t2v_latent",
            "t2v_hidden_state",
            "t2v_velocity",
            "t2v_noise",
            "t2v_per_sample_gradient_as_parameter_step",
            "pseudo_ground_truth_video",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
        ],
        "band_selection_rule": BAND_SELECTION_RULE,
        "optimizer_update_authorized": False,
        "remaining_required_gates": [
            "native_same_state_student_vjp",
            "source_safe_parameter_space_qp",
            "branch_lock",
            "fresh_exact81_decoded_rollback",
        ],
    }
    return {**unsigned, "digest": _object_sha256(unsigned)}
