"""Source-aware, actor/camera-factorized action representation primitives.

R5 deliberately does not reuse the prompt-only ``PromptActionEncoder``.  Its
student predicts a source-conditioned edit in two independent factors:

``(source actor state, target intent) -> actor motion delta``
``(source camera state, target intent) -> camera motion delta``

The target intent can be an instruction embedding, a target-reference motion
descriptor, or both.  Reference availability is explicit so text-only and
reference-conditioned evaluation cannot be silently mixed.

This module is checkpoint/generator independent.  It produces typed temporal
conditioning tokens for a future denoiser integration; feeding the output back
into Lucy's global V10 rank gate would discard the spatial/temporal distinction
that R5 is intended to preserve.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from torch.nn import Module as _TorchModuleBase
except ImportError:
    # Keep split/teacher/control utilities importable on CPU audit hosts that
    # do not install PyTorch. Instantiating a model still fails through _nn().
    class _TorchModuleBase:  # type: ignore[no-redef]
        pass


R5_SCHEMA_VERSION = "motive-source-aware-factorized-action-repr-v1"
R5_CONTENT_SPLIT_VERSION = "source-visual-cluster-v1"
R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION = "source-phash-near-cluster-v1"
R5_TOKEN_ROLES = (
    "source_actor",
    "actor_delta",
    "source_camera",
    "camera_delta",
)
VALID_SPLITS = frozenset({"train", "validation", "test"})
_INT_BIT_COUNT = getattr(int, "bit_count", None)


def _popcount(value: int) -> int:
    return (
        int(_INT_BIT_COUNT(value))
        if _INT_BIT_COUNT is not None
        else bin(value).count("1")
    )


@dataclass(frozen=True)
class R5ExperimentSeeds:
    """Seeds are named by causal role and must be persisted separately."""

    data_seed: int
    model_seed: int

    def __post_init__(self) -> None:
        for name in ("data_seed", "model_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "data_seed": int(self.data_seed),
            "model_seed": int(self.model_seed),
        }


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the R5 representation") from error
    return torch


def _nn() -> Any:
    return _torch().nn


def _as_finite_matrix(
    values: Any,
    *,
    name: str,
    rows: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N,D] with D >= 1")
    if rows is not None and len(matrix) != rows:
        raise ValueError(f"{name} row count mismatch")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ContentSplitAudit:
    """Result of a fail-closed content-group split audit."""

    samples: int
    groups: int
    split_counts: dict[str, int]
    group_counts: dict[str, int]
    split_version: str
    cross_split_group_collisions: tuple[str, ...]
    cross_split_near_duplicate_pairs: tuple[tuple[int, int, float], ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_hex_hash(value: str, *, index: int) -> bytes:
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(
            f"perceptual_hashes[{index}] is not hexadecimal"
        ) from error
    if not decoded:
        raise ValueError(f"perceptual_hashes[{index}] is empty")
    return decoded


def audit_content_disjoint_splits(
    *,
    splits: Sequence[str],
    content_group_ids: Sequence[str],
    split_versions: Sequence[str] | None = None,
    perceptual_hashes: Sequence[str] | None = None,
    maximum_cross_split_hamming_fraction: float = 0.10,
    require_visual_clusters: bool = True,
) -> ContentSplitAudit:
    """Verify that content groups cannot cross train/validation/test.

    ``source-sampled-phash-v1`` is not accepted as a production split by
    default: exact pHash equality does not cluster near duplicates or repeated
    subjects across shots. ``source-phash-near-cluster-v1`` can be inspected
    in diagnostic mode by setting ``require_visual_clusters=False`` and
    providing hashes, in which case an explicit cross-split Hamming audit is
    also performed. Diagnostic acceptance is not a production attestation.
    """

    split_values = tuple(str(value) for value in splits)
    group_values = tuple(str(value).strip() for value in content_group_ids)
    if not split_values:
        raise ValueError("content split audit requires at least one sample")
    if len(group_values) != len(split_values):
        raise ValueError("content_group_ids length mismatch")
    invalid_splits = sorted(set(split_values) - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"invalid split values: {invalid_splits}")
    if any(not value for value in group_values):
        raise ValueError("content_group_ids must be non-empty")

    if split_versions is None:
        version_values = (R5_CONTENT_SPLIT_VERSION,) * len(split_values)
    else:
        version_values = tuple(str(value) for value in split_versions)
        if len(version_values) != len(split_values):
            raise ValueError("split_versions length mismatch")
    unique_versions = sorted(set(version_values))
    if len(unique_versions) != 1:
        raise ValueError(
            "one R5 experiment cannot mix split provenance versions: "
            f"{unique_versions}"
        )
    split_version = unique_versions[0]
    if require_visual_clusters and split_version != R5_CONTENT_SPLIT_VERSION:
        raise ValueError(
            "R5 production training requires source subject/scene visual "
            f"clusters ({R5_CONTENT_SPLIT_VERSION!r}); got {split_version!r}"
        )
    if (
        not require_visual_clusters
        and split_version != R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION
    ):
        raise ValueError(
            "R5 diagnostic training requires the explicit near-pHash split "
            f"version {R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION!r}; got "
            f"{split_version!r}"
        )
    if not require_visual_clusters and perceptual_hashes is None:
        raise ValueError(
            "diagnostic near-pHash splits require perceptual_hashes for a "
            "cross-split near-duplicate audit"
        )

    group_splits: dict[str, set[str]] = {}
    for split, group_id in zip(split_values, group_values):
        group_splits.setdefault(group_id, set()).add(split)
    collisions = tuple(
        sorted(group_id for group_id, values in group_splits.items() if len(values) > 1)
    )

    near_duplicates: list[tuple[int, int, float]] = []
    if perceptual_hashes is not None:
        if len(perceptual_hashes) != len(split_values):
            raise ValueError("perceptual_hashes length mismatch")
        if not 0.0 <= maximum_cross_split_hamming_fraction < 1.0:
            raise ValueError(
                "maximum_cross_split_hamming_fraction must be in [0,1)"
            )
        decoded = [
            _validate_hex_hash(str(value), index=index)
            for index, value in enumerate(perceptual_hashes)
        ]
        lengths = {len(value) for value in decoded}
        if len(lengths) != 1:
            raise ValueError("perceptual hashes must have a common byte length")
        bit_count = 8 * next(iter(lengths))
        for left in range(len(decoded)):
            for right in range(left + 1, len(decoded)):
                if split_values[left] == split_values[right]:
                    continue
                distance = _popcount(
                    int.from_bytes(decoded[left], "big")
                    ^ int.from_bytes(decoded[right], "big")
                ) / float(bit_count)
                if distance <= maximum_cross_split_hamming_fraction:
                    near_duplicates.append((left, right, float(distance)))

    split_counts = {
        split: int(sum(value == split for value in split_values))
        for split in sorted(VALID_SPLITS)
    }
    group_counts = {
        split: int(
            len(
                {
                    group_id
                    for group_id, value in zip(group_values, split_values)
                    if value == split
                }
            )
        )
        for split in sorted(VALID_SPLITS)
    }
    passed = not collisions and not near_duplicates
    audit = ContentSplitAudit(
        samples=len(split_values),
        groups=len(group_splits),
        split_counts=split_counts,
        group_counts=group_counts,
        split_version=split_version,
        cross_split_group_collisions=collisions,
        cross_split_near_duplicate_pairs=tuple(near_duplicates),
        passed=passed,
    )
    if not passed:
        raise ValueError(
            "content-disjoint split audit failed: "
            f"group_collisions={list(collisions)[:8]} "
            f"near_duplicate_pairs={near_duplicates[:8]}"
        )
    return audit


def stable_splits_from_content_groups(
    content_group_ids: Sequence[str],
    *,
    data_seed: int = 260108828,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[str, ...]:
    """Assign whole visual-content clusters to deterministic splits."""

    if (
        not math.isfinite(train_fraction)
        or not math.isfinite(validation_fraction)
        or train_fraction <= 0.0
        or validation_fraction < 0.0
        or train_fraction + validation_fraction >= 1.0
    ):
        raise ValueError("invalid train/validation fractions")
    output: list[str] = []
    train_threshold = int(round(train_fraction * 10_000))
    validation_threshold = int(
        round((train_fraction + validation_fraction) * 10_000)
    )
    for index, raw_group_id in enumerate(content_group_ids):
        group_id = str(raw_group_id).strip()
        if not group_id:
            raise ValueError(f"content_group_ids[{index}] is empty")
        digest = hashlib.sha256(
            f"{int(data_seed)}\0{group_id}".encode("utf-8")
        ).digest()
        bucket = int.from_bytes(digest[:8], "little") % 10_000
        output.append(
            "train"
            if bucket < train_threshold
            else ("validation" if bucket < validation_threshold else "test")
        )
    return tuple(output)


@dataclass(frozen=True)
class R5EndpointBatch:
    """Endpoint features required by an R5 representation experiment.

    Delta-only archives cannot instantiate this type by construction.  Actor
    and camera endpoints remain separate so camera changes cannot rescale or
    rotate the actor teacher.
    """

    iids: tuple[str, ...]
    source_actor: np.ndarray
    source_camera: np.ndarray
    target_actor: np.ndarray
    target_camera: np.ndarray
    instruction_features: np.ndarray
    splits: tuple[str, ...]
    content_group_ids: tuple[str, ...]
    action_signatures: tuple[str, ...]
    split_versions: tuple[str, ...]
    perceptual_hashes: tuple[str, ...] | None
    maximum_cross_split_hamming_fraction: float | None

    @classmethod
    def create(
        cls,
        *,
        iids: Sequence[str],
        source_actor: Any,
        source_camera: Any,
        target_actor: Any,
        target_camera: Any,
        instruction_features: Any,
        splits: Sequence[str],
        content_group_ids: Sequence[str],
        action_signatures: Sequence[str],
        split_versions: Sequence[str],
        perceptual_hashes: Sequence[str] | None = None,
        require_visual_clusters: bool = True,
        maximum_cross_split_hamming_fraction: float = 0.10,
    ) -> "R5EndpointBatch":
        iid_values = tuple(str(value) for value in iids)
        if not iid_values or len(set(iid_values)) != len(iid_values):
            raise ValueError("R5 iids must be non-empty and unique")
        rows = len(iid_values)
        matrices = {
            "source_actor": _as_finite_matrix(
                source_actor, name="source_actor", rows=rows
            ),
            "source_camera": _as_finite_matrix(
                source_camera, name="source_camera", rows=rows
            ),
            "target_actor": _as_finite_matrix(
                target_actor, name="target_actor", rows=rows
            ),
            "target_camera": _as_finite_matrix(
                target_camera, name="target_camera", rows=rows
            ),
            "instruction_features": _as_finite_matrix(
                instruction_features,
                name="instruction_features",
                rows=rows,
            ),
        }
        if matrices["source_actor"].shape != matrices["target_actor"].shape:
            raise ValueError("source/target actor endpoint dimensions differ")
        if matrices["source_camera"].shape != matrices["target_camera"].shape:
            raise ValueError("source/target camera endpoint dimensions differ")
        split_values = tuple(str(value) for value in splits)
        group_values = tuple(str(value) for value in content_group_ids)
        signature_values = tuple(str(value).strip() for value in action_signatures)
        version_values = tuple(str(value) for value in split_versions)
        hash_values = (
            None
            if perceptual_hashes is None
            else tuple(str(value).strip().lower() for value in perceptual_hashes)
        )
        for name, values in (
            ("splits", split_values),
            ("content_group_ids", group_values),
            ("action_signatures", signature_values),
            ("split_versions", version_values),
        ):
            if len(values) != rows:
                raise ValueError(f"{name} length mismatch")
        if hash_values is not None and len(hash_values) != rows:
            raise ValueError("perceptual_hashes length mismatch")
        if any(not value for value in signature_values):
            raise ValueError("action_signatures must be non-empty")
        audit_content_disjoint_splits(
            splits=split_values,
            content_group_ids=group_values,
            split_versions=version_values,
            perceptual_hashes=hash_values,
            maximum_cross_split_hamming_fraction=(
                maximum_cross_split_hamming_fraction
            ),
            require_visual_clusters=require_visual_clusters,
        )
        return cls(
            iids=iid_values,
            source_actor=matrices["source_actor"],
            source_camera=matrices["source_camera"],
            target_actor=matrices["target_actor"],
            target_camera=matrices["target_camera"],
            instruction_features=matrices["instruction_features"],
            splits=split_values,
            content_group_ids=group_values,
            action_signatures=signature_values,
            split_versions=version_values,
            perceptual_hashes=hash_values,
            maximum_cross_split_hamming_fraction=(
                float(maximum_cross_split_hamming_fraction)
                if hash_values is not None
                else None
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        require_visual_clusters: bool = True,
        maximum_cross_split_hamming_fraction: float = 0.10,
    ) -> "R5EndpointBatch":
        required = {
            "iids",
            "source_actor",
            "source_camera",
            "target_actor",
            "target_camera",
            "instruction_features",
            "splits",
            "content_group_ids",
            "action_signatures",
            "split_versions",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                "R5 endpoint payload is incomplete (delta-only archives are "
                f"unsupported); missing={missing}"
            )
        plural_hashes = payload.get("perceptual_hashes")
        singular_hashes = payload.get("source_perceptual_hash")
        if plural_hashes is not None and singular_hashes is not None:
            plural_values = tuple(str(value) for value in plural_hashes)
            singular_values = tuple(str(value) for value in singular_hashes)
            if plural_values != singular_values:
                raise ValueError(
                    "perceptual_hashes/source_perceptual_hash aliases differ"
                )
        hash_values = (
            plural_hashes
            if plural_hashes is not None
            else singular_hashes
        )
        return cls.create(
            **{name: payload[name] for name in sorted(required)},
            perceptual_hashes=hash_values,
            require_visual_clusters=require_visual_clusters,
            maximum_cross_split_hamming_fraction=(
                maximum_cross_split_hamming_fraction
            ),
        )

    def indices(self, split: str) -> np.ndarray:
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split {split!r}")
        return np.flatnonzero(np.asarray(self.splits) == split).astype(np.int64)


@dataclass(frozen=True)
class Standardizer:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    scale_floor: float

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        *,
        scale_floor_fraction: float = 0.01,
        eps: float = 1e-6,
    ) -> "Standardizer":
        matrix = _as_finite_matrix(values, name="standardizer values")
        if not len(matrix):
            raise ValueError("cannot fit a standardizer without samples")
        if scale_floor_fraction < 0.0:
            raise ValueError("scale_floor_fraction must be non-negative")
        mean = np.mean(matrix, axis=0)
        raw_scale = np.std(matrix, axis=0)
        maximum = float(np.max(raw_scale))
        floor = max(eps, maximum * scale_floor_fraction)
        scale = np.maximum(raw_scale, floor)
        return cls(
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
            scale_floor=float(floor),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = _as_finite_matrix(values, name="standardizer input")
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        if matrix.shape[1] != len(mean):
            raise ValueError("standardizer feature dimension mismatch")
        return ((matrix - mean) / scale).astype(np.float32)


@dataclass(frozen=True)
class DeltaWhitening:
    """Train-only PCA whitening for one motion factor."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    components: tuple[tuple[float, ...], ...]
    component_scale: tuple[float, ...]
    output_dim: int
    stable_rank: int
    minimum_energy: float = 1e-6

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        *,
        output_dim: int,
        minimum_relative_variance: float = 1e-4,
        whitening_ridge_fraction: float = 0.05,
        eps: float = 1e-6,
    ) -> "DeltaWhitening":
        matrix = _as_finite_matrix(values, name="delta whitening values")
        if len(matrix) < 2:
            raise ValueError("delta whitening requires at least two train samples")
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        standardizer = Standardizer.fit(matrix, eps=eps)
        standardized = standardizer.transform(matrix).astype(np.float64)
        _, singular_values, right = np.linalg.svd(
            standardized,
            full_matrices=False,
        )
        variance = singular_values**2 / max(len(matrix) - 1, 1)
        maximum = float(variance[0]) if len(variance) else 0.0
        if maximum <= eps**2:
            raise ValueError("motion delta has no stable train-split variance")
        stable_rank = int(
            np.count_nonzero(
                variance >= max(maximum * minimum_relative_variance, eps**2)
            )
        )
        rank = max(1, min(int(output_dim), right.shape[0], stable_rank))
        ridge = max(eps, whitening_ridge_fraction * math.sqrt(maximum))
        component_scale = np.sqrt(variance[:rank] + ridge**2)
        return cls(
            mean=standardizer.mean,
            scale=standardizer.scale,
            components=tuple(
                tuple(float(value) for value in row)
                for row in right[:rank].astype(np.float32)
            ),
            component_scale=tuple(float(value) for value in component_scale),
            output_dim=int(output_dim),
            stable_rank=rank,
        )

    def transform(
        self,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = _as_finite_matrix(values, name="delta whitening input")
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        components = np.asarray(self.components, dtype=np.float32)
        component_scale = np.asarray(self.component_scale, dtype=np.float32)
        if matrix.shape[1] != len(mean):
            raise ValueError("delta whitening feature dimension mismatch")
        # The PCA basis is fitted on centered train deltas, but the physical
        # no-edit origin must remain exactly zero. Subtracting the train mean
        # here would turn static/suppression examples into a non-zero action.
        whitened = (matrix / scale) @ components.T
        whitened = whitened / component_scale
        if whitened.shape[1] < self.output_dim:
            whitened = np.pad(
                whitened,
                ((0, 0), (0, self.output_dim - whitened.shape[1])),
            )
        energy = np.linalg.norm(whitened, axis=1).astype(np.float32)
        direction = np.divide(
            whitened,
            np.maximum(energy[:, None], self.minimum_energy),
            out=np.zeros_like(whitened, dtype=np.float32),
            where=energy[:, None] >= self.minimum_energy,
        )
        return direction.astype(np.float32), np.log1p(energy).astype(np.float32)


@dataclass(frozen=True)
class FactorizedR5Targets:
    actor_direction: np.ndarray
    actor_log_magnitude: np.ndarray
    camera_direction: np.ndarray
    camera_log_magnitude: np.ndarray

    def subset(self, indices: np.ndarray) -> "FactorizedR5Targets":
        return FactorizedR5Targets(
            actor_direction=self.actor_direction[indices],
            actor_log_magnitude=self.actor_log_magnitude[indices],
            camera_direction=self.camera_direction[indices],
            camera_log_magnitude=self.camera_log_magnitude[indices],
        )


@dataclass(frozen=True)
class R5FeatureTransform:
    """All preprocessing fitted strictly on train content groups."""

    actor_endpoint: Standardizer
    camera_endpoint: Standardizer
    instruction: Standardizer
    actor_delta: DeltaWhitening
    camera_delta: DeltaWhitening
    train_iid_digest: str
    schema_version: str = R5_SCHEMA_VERSION

    @classmethod
    def fit(
        cls,
        batch: R5EndpointBatch,
        *,
        condition_dim: int = 128,
    ) -> "R5FeatureTransform":
        train = batch.indices("train")
        if len(train) < 2:
            raise ValueError("R5 transform requires at least two train samples")
        actor_endpoints = np.concatenate(
            (batch.source_actor[train], batch.target_actor[train]),
            axis=0,
        )
        camera_endpoints = np.concatenate(
            (batch.source_camera[train], batch.target_camera[train]),
            axis=0,
        )
        actor_delta = batch.target_actor[train] - batch.source_actor[train]
        camera_delta = batch.target_camera[train] - batch.source_camera[train]
        return cls(
            actor_endpoint=Standardizer.fit(actor_endpoints),
            camera_endpoint=Standardizer.fit(camera_endpoints),
            instruction=Standardizer.fit(batch.instruction_features[train]),
            actor_delta=DeltaWhitening.fit(
                actor_delta,
                output_dim=condition_dim,
            ),
            camera_delta=DeltaWhitening.fit(
                camera_delta,
                output_dim=condition_dim,
            ),
            train_iid_digest=_canonical_digest(
                sorted(batch.iids[int(index)] for index in train)
            ),
        )

    def source_inputs(
        self,
        batch: R5EndpointBatch,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.actor_endpoint.transform(batch.source_actor),
            self.camera_endpoint.transform(batch.source_camera),
            self.instruction.transform(batch.instruction_features),
        )

    def reference_inputs(
        self,
        batch: R5EndpointBatch,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.actor_endpoint.transform(batch.target_actor),
            self.camera_endpoint.transform(batch.target_camera),
        )

    def targets(self, batch: R5EndpointBatch) -> FactorizedR5Targets:
        actor_direction, actor_magnitude = self.actor_delta.transform(
            batch.target_actor - batch.source_actor
        )
        camera_direction, camera_magnitude = self.camera_delta.transform(
            batch.target_camera - batch.source_camera
        )
        return FactorizedR5Targets(
            actor_direction=actor_direction,
            actor_log_magnitude=actor_magnitude,
            camera_direction=camera_direction,
            camera_log_magnitude=camera_magnitude,
        )

    def digest(self) -> str:
        return _canonical_digest(asdict(self))


class _FactorBranch(_TorchModuleBase):
    """One factor branch; it never receives the other factor's state."""

    def __init__(
        self,
        *,
        state_dim: int,
        instruction_dim: int,
        condition_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        nn = _nn()

        def tower(input_dim: int) -> Any:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )

        self.source = tower(state_dim)
        self.instruction = tower(instruction_dim)
        self.reference = tower(state_dim)
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.direction = nn.Linear(hidden_dim, condition_dim)
        self.log_magnitude = nn.Linear(hidden_dim, 1)
        self.source_token = nn.Linear(hidden_dim, condition_dim)

    def forward(
        self,
        source_state: Any,
        instruction: Any,
        reference_state: Any,
        reference_mask: Any,
    ) -> tuple[Any, Any, Any]:
        torch = _torch()
        source_hidden = self.source(source_state)
        instruction_hidden = self.instruction(instruction)
        reference_hidden = self.reference(reference_state)
        reference_hidden = reference_hidden * reference_mask
        hidden = self.fusion(
            torch.cat(
                (
                    source_hidden,
                    instruction_hidden,
                    reference_hidden,
                    reference_mask,
                ),
                dim=-1,
            )
        )
        direction = torch.nn.functional.normalize(
            self.direction(hidden).float(),
            dim=-1,
        ).to(hidden.dtype)
        # log1p magnitude targets are non-negative.
        log_magnitude = torch.nn.functional.softplus(
            self.log_magnitude(hidden).float()
        ).to(hidden.dtype)
        return direction, log_magnitude, self.source_token(source_hidden)


class SourceAwareFactorizedR5(_TorchModuleBase):
    """Minimal R5 student with structurally separated actor/camera branches."""

    def __init__(
        self,
        *,
        actor_state_dim: int,
        camera_state_dim: int,
        instruction_dim: int,
        condition_dim: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(
            actor_state_dim,
            camera_state_dim,
            instruction_dim,
            condition_dim,
            hidden_dim,
        ) < 1:
            raise ValueError("all R5 dimensions must be positive")
        self.actor_state_dim = int(actor_state_dim)
        self.camera_state_dim = int(camera_state_dim)
        self.instruction_dim = int(instruction_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.actor = _FactorBranch(
            state_dim=self.actor_state_dim,
            instruction_dim=self.instruction_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.hidden_dim,
        )
        self.camera = _FactorBranch(
            state_dim=self.camera_state_dim,
            instruction_dim=self.instruction_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.hidden_dim,
        )
        self.token_role = _nn().Embedding(len(R5_TOKEN_ROLES), self.condition_dim)

    @staticmethod
    def _validate_tensor(name: str, value: Any, *, width: int) -> None:
        torch = _torch()
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{name} must have shape [B,{width}]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")

    def forward(
        self,
        *,
        source_actor: Any,
        source_camera: Any,
        instruction_features: Any,
        reference_actor: Any | None = None,
        reference_camera: Any | None = None,
        reference_mask: Any | None = None,
    ) -> dict[str, Any]:
        torch = _torch()
        self._validate_tensor(
            "source_actor", source_actor, width=self.actor_state_dim
        )
        self._validate_tensor(
            "source_camera", source_camera, width=self.camera_state_dim
        )
        self._validate_tensor(
            "instruction_features",
            instruction_features,
            width=self.instruction_dim,
        )
        batch = source_actor.shape[0]
        if source_camera.shape[0] != batch or instruction_features.shape[0] != batch:
            raise ValueError("R5 input batch dimensions differ")
        has_reference = reference_actor is not None
        if has_reference != (reference_camera is not None):
            raise ValueError(
                "reference_actor and reference_camera must be supplied together"
            )
        if reference_mask is not None and not has_reference:
            raise ValueError("reference_mask requires reference endpoints")
        if reference_actor is None:
            reference_actor = torch.zeros_like(source_actor)
            reference_camera = torch.zeros_like(source_camera)
            mask = source_actor.new_zeros((batch, 1))
        else:
            self._validate_tensor(
                "reference_actor",
                reference_actor,
                width=self.actor_state_dim,
            )
            self._validate_tensor(
                "reference_camera",
                reference_camera,
                width=self.camera_state_dim,
            )
            if reference_actor.shape[0] != batch or reference_camera.shape[0] != batch:
                raise ValueError("R5 reference batch dimensions differ")
            if reference_mask is None:
                mask = source_actor.new_ones((batch, 1))
            else:
                if (
                    not torch.is_tensor(reference_mask)
                    or reference_mask.shape not in {(batch,), (batch, 1)}
                ):
                    raise ValueError("reference_mask must have shape [B] or [B,1]")
                mask = reference_mask.reshape(batch, 1).to(
                    device=source_actor.device,
                    dtype=source_actor.dtype,
                )
                if (
                    not bool(torch.isfinite(mask).all())
                    or bool(((mask < 0.0) | (mask > 1.0)).any())
                ):
                    raise ValueError("reference_mask must contain values in [0,1]")
        actor_direction, actor_magnitude, source_actor_token = self.actor(
            source_actor,
            instruction_features,
            reference_actor,
            mask,
        )
        camera_direction, camera_magnitude, source_camera_token = self.camera(
            source_camera,
            instruction_features,
            reference_camera,
            mask,
        )
        delta_scale_actor = torch.expm1(actor_magnitude.float()).to(
            actor_direction.dtype
        )
        delta_scale_camera = torch.expm1(camera_magnitude.float()).to(
            camera_direction.dtype
        )
        role_ids = torch.arange(
            len(R5_TOKEN_ROLES),
            device=source_actor.device,
        )
        roles = self.token_role(role_ids).unsqueeze(0)
        tokens = torch.stack(
            (
                source_actor_token,
                actor_direction * delta_scale_actor,
                source_camera_token,
                camera_direction * delta_scale_camera,
            ),
            dim=1,
        )
        tokens = tokens + roles.to(dtype=tokens.dtype)
        return {
            "actor_direction": actor_direction,
            "actor_log_magnitude": actor_magnitude,
            "camera_direction": camera_direction,
            "camera_log_magnitude": camera_magnitude,
            "conditioning_tokens": tokens,
            "token_roles": R5_TOKEN_ROLES,
            "reference_mask": mask,
        }


def factorized_r5_loss(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    direction_weight: float = 1.0,
    magnitude_weight: float = 0.25,
    active_log_magnitude_threshold: float = 1e-4,
) -> dict[str, Any]:
    """Direction + amplitude loss that keeps static/suppression rows useful."""

    torch = _torch()
    if direction_weight < 0.0 or magnitude_weight < 0.0:
        raise ValueError("R5 loss weights must be non-negative")
    losses: dict[str, Any] = {}
    total = None
    for factor in ("actor", "camera"):
        predicted_direction = prediction[f"{factor}_direction"].float()
        target_direction = target[f"{factor}_direction"].float()
        predicted_magnitude = prediction[f"{factor}_log_magnitude"].float().reshape(
            -1
        )
        target_magnitude = target[f"{factor}_log_magnitude"].float().reshape(-1)
        if predicted_direction.shape != target_direction.shape:
            raise ValueError(f"{factor} direction shape mismatch")
        if len(predicted_direction) != len(predicted_magnitude):
            raise ValueError(f"{factor} magnitude shape mismatch")
        active = target_magnitude > active_log_magnitude_threshold
        cosine = 1.0 - torch.sum(
            predicted_direction * target_direction,
            dim=-1,
        )
        direction_loss = (
            cosine[active].mean()
            if bool(active.any())
            else cosine.new_zeros(())
        )
        magnitude_loss = torch.nn.functional.smooth_l1_loss(
            predicted_magnitude,
            target_magnitude,
        )
        factor_loss = (
            float(direction_weight) * direction_loss
            + float(magnitude_weight) * magnitude_loss
        )
        losses[f"{factor}_direction"] = direction_loss
        losses[f"{factor}_magnitude"] = magnitude_loss
        losses[f"{factor}_active_fraction"] = active.float().mean()
        total = factor_loss if total is None else total + factor_loss
    losses["loss"] = total
    return losses


def make_matched_random_control(
    model: SourceAwareFactorizedR5,
    *,
    model_seed: int,
) -> SourceAwareFactorizedR5:
    """Return a deterministic random model with the exact R5 architecture."""

    torch = _torch()
    control = copy.deepcopy(model)
    devices = sorted(
        {
            parameter.device.index
            for parameter in control.parameters()
            if parameter.is_cuda and parameter.device.index is not None
        }
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(model_seed))

        def reset(module: Any) -> None:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        control.apply(reset)
    return control


def source_shuffled_indices(
    *,
    splits: Sequence[str],
    action_signatures: Sequence[str],
    content_group_ids: Sequence[str],
    data_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a within-split/signature, cross-content source derangement.

    Rows in a stratum without a perfect cross-content derangement are marked
    invalid and must not contribute to the shuffled-control metric.
    """

    split_values = tuple(str(value) for value in splits)
    signature_values = tuple(str(value) for value in action_signatures)
    group_values = tuple(str(value) for value in content_group_ids)
    if not (
        len(split_values) == len(signature_values) == len(group_values)
    ):
        raise ValueError("source shuffle metadata lengths differ")
    rng = np.random.default_rng(int(data_seed))
    output = np.arange(len(split_values), dtype=np.int64)
    valid = np.zeros(len(split_values), dtype=bool)
    strata: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(zip(split_values, signature_values)):
        strata.setdefault(key, []).append(index)

    for key in sorted(strata):
        anchors = list(strata[key])
        if len(anchors) < 2:
            continue
        candidate_order = {
            anchor: [
                int(value)
                for value in rng.permutation(anchors)
                if value != anchor
                and group_values[int(value)] != group_values[anchor]
            ]
            for anchor in anchors
        }
        matched_candidate_to_anchor: dict[int, int] = {}

        def augment(anchor: int, seen: set[int]) -> bool:
            for candidate in candidate_order[anchor]:
                if candidate in seen:
                    continue
                seen.add(candidate)
                previous = matched_candidate_to_anchor.get(candidate)
                if previous is None or augment(previous, seen):
                    matched_candidate_to_anchor[candidate] = anchor
                    return True
            return False

        full = all(augment(anchor, set()) for anchor in anchors)
        if not full or len(matched_candidate_to_anchor) != len(anchors):
            continue
        for candidate, anchor in matched_candidate_to_anchor.items():
            output[anchor] = candidate
            valid[anchor] = True
    return output, valid


def prompt_shuffled_indices(
    *,
    splits: Sequence[str],
    action_signatures: Sequence[str],
    content_group_ids: Sequence[str],
    data_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a within-split prompt derangement across action signatures.

    The source endpoint and teacher stay fixed; only the target instruction
    feature is replaced.  Requiring a different signature makes this an
    informative target-intent control instead of a paraphrase control.  Rows
    without a perfect derangement are excluded through the returned mask.
    """

    split_values = tuple(str(value) for value in splits)
    signature_values = tuple(str(value) for value in action_signatures)
    group_values = tuple(str(value) for value in content_group_ids)
    if not (
        len(split_values) == len(signature_values) == len(group_values)
    ):
        raise ValueError("prompt shuffle metadata lengths differ")
    rng = np.random.default_rng(int(data_seed))
    output = np.arange(len(split_values), dtype=np.int64)
    valid = np.zeros(len(split_values), dtype=bool)
    split_rows: dict[str, list[int]] = {}
    for index, split in enumerate(split_values):
        split_rows.setdefault(split, []).append(index)

    for split in sorted(split_rows):
        anchors = list(split_rows[split])
        if len(anchors) < 2:
            continue
        candidate_order = {
            anchor: [
                int(value)
                for value in rng.permutation(anchors)
                if value != anchor
                and group_values[int(value)] != group_values[anchor]
                and signature_values[int(value)] != signature_values[anchor]
            ]
            for anchor in anchors
        }
        matched_candidate_to_anchor: dict[int, int] = {}

        def augment(anchor: int, seen: set[int]) -> bool:
            for candidate in candidate_order[anchor]:
                if candidate in seen:
                    continue
                seen.add(candidate)
                previous = matched_candidate_to_anchor.get(candidate)
                if previous is None or augment(previous, seen):
                    matched_candidate_to_anchor[candidate] = anchor
                    return True
            return False

        full = all(augment(anchor, set()) for anchor in anchors)
        if not full or len(matched_candidate_to_anchor) != len(anchors):
            continue
        for candidate, anchor in matched_candidate_to_anchor.items():
            output[anchor] = candidate
            valid[anchor] = True
    return output, valid


@dataclass(frozen=True)
class FactorizedCentroidControl:
    """Train-only action-signature centroid, with train-global fallback."""

    actor_centroids: dict[str, tuple[float, ...]]
    camera_centroids: dict[str, tuple[float, ...]]
    actor_magnitude_centroids: dict[str, float]
    camera_magnitude_centroids: dict[str, float]
    global_actor: tuple[float, ...]
    global_camera: tuple[float, ...]
    global_actor_magnitude: float
    global_camera_magnitude: float
    train_samples: int

    @staticmethod
    def _unit(values: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(values))
        return (
            np.zeros_like(values, dtype=np.float32)
            if norm <= 1e-8
            else (values / norm).astype(np.float32)
        )

    @classmethod
    def fit(
        cls,
        *,
        targets: FactorizedR5Targets,
        action_signatures: Sequence[str],
        train_indices: Sequence[int],
    ) -> "FactorizedCentroidControl":
        train = np.asarray(train_indices, dtype=np.int64)
        signatures = np.asarray([str(value) for value in action_signatures])
        if train.ndim != 1 or not len(train):
            raise ValueError("centroid control requires train indices")
        if int(train.min()) < 0 or int(train.max()) >= len(signatures):
            raise ValueError("centroid train indices are out of bounds")

        def direction_centroids(values: np.ndarray) -> tuple[
            dict[str, tuple[float, ...]],
            tuple[float, ...],
        ]:
            global_value = cls._unit(np.mean(values[train], axis=0))
            per_signature: dict[str, tuple[float, ...]] = {}
            for signature in sorted(set(signatures[train].tolist())):
                indices = train[signatures[train] == signature]
                per_signature[signature] = tuple(
                    float(value)
                    for value in cls._unit(np.mean(values[indices], axis=0))
                )
            return per_signature, tuple(float(value) for value in global_value)

        def magnitude_centroids(values: np.ndarray) -> tuple[dict[str, float], float]:
            flattened = np.asarray(values, dtype=np.float32).reshape(-1)
            return (
                {
                    signature: float(
                        np.mean(
                            flattened[
                                train[signatures[train] == signature]
                            ]
                        )
                    )
                    for signature in sorted(set(signatures[train].tolist()))
                },
                float(np.mean(flattened[train])),
            )

        actor, global_actor = direction_centroids(targets.actor_direction)
        camera, global_camera = direction_centroids(targets.camera_direction)
        actor_magnitude, global_actor_magnitude = magnitude_centroids(
            targets.actor_log_magnitude
        )
        camera_magnitude, global_camera_magnitude = magnitude_centroids(
            targets.camera_log_magnitude
        )
        return cls(
            actor_centroids=actor,
            camera_centroids=camera,
            actor_magnitude_centroids=actor_magnitude,
            camera_magnitude_centroids=camera_magnitude,
            global_actor=global_actor,
            global_camera=global_camera,
            global_actor_magnitude=global_actor_magnitude,
            global_camera_magnitude=global_camera_magnitude,
            train_samples=len(train),
        )

    def predict(
        self,
        action_signatures: Sequence[str],
    ) -> FactorizedR5Targets:
        signatures = [str(value) for value in action_signatures]
        return FactorizedR5Targets(
            actor_direction=np.asarray(
                [
                    self.actor_centroids.get(signature, self.global_actor)
                    for signature in signatures
                ],
                dtype=np.float32,
            ),
            actor_log_magnitude=np.asarray(
                [
                    self.actor_magnitude_centroids.get(
                        signature,
                        self.global_actor_magnitude,
                    )
                    for signature in signatures
                ],
                dtype=np.float32,
            ),
            camera_direction=np.asarray(
                [
                    self.camera_centroids.get(signature, self.global_camera)
                    for signature in signatures
                ],
                dtype=np.float32,
            ),
            camera_log_magnitude=np.asarray(
                [
                    self.camera_magnitude_centroids.get(
                        signature,
                        self.global_camera_magnitude,
                    )
                    for signature in signatures
                ],
                dtype=np.float32,
            ),
        )


def factorized_representation_metrics(
    prediction: FactorizedR5Targets,
    target: FactorizedR5Targets,
    *,
    active_log_magnitude_threshold: float = 1e-4,
) -> dict[str, float]:
    """Report factor-specific direction and amplitude without pooling them."""

    result: dict[str, float] = {}
    for factor in ("actor", "camera"):
        predicted_direction = np.asarray(
            getattr(prediction, f"{factor}_direction"),
            dtype=np.float32,
        )
        target_direction = np.asarray(
            getattr(target, f"{factor}_direction"),
            dtype=np.float32,
        )
        predicted_magnitude = np.asarray(
            getattr(prediction, f"{factor}_log_magnitude"),
            dtype=np.float32,
        ).reshape(-1)
        target_magnitude = np.asarray(
            getattr(target, f"{factor}_log_magnitude"),
            dtype=np.float32,
        ).reshape(-1)
        if (
            predicted_direction.shape != target_direction.shape
            or len(predicted_direction) != len(predicted_magnitude)
            or len(predicted_magnitude) != len(target_magnitude)
        ):
            raise ValueError(f"{factor} prediction/target shape mismatch")
        active = target_magnitude > active_log_magnitude_threshold
        cosine = np.sum(predicted_direction * target_direction, axis=1)
        result[f"{factor}_active_count"] = int(np.count_nonzero(active))
        result[f"{factor}_mean_cosine_active"] = (
            float(np.mean(cosine[active])) if np.any(active) else 0.0
        )
        result[f"{factor}_magnitude_mae"] = float(
            np.mean(np.abs(predicted_magnitude - target_magnitude))
        )
    return result


def factorized_per_sample_metrics(
    prediction: FactorizedR5Targets,
    target: FactorizedR5Targets,
    *,
    iids: Sequence[str],
    splits: Sequence[str],
    action_signatures: Sequence[str],
    valid_mask: Sequence[bool] | None = None,
    arm: str,
    data_seed: int,
    model_seed: int,
) -> list[dict[str, Any]]:
    """Return paired raw rows suitable for macro metrics and bootstrap CIs."""

    actor_prediction = _as_finite_matrix(
        prediction.actor_direction,
        name="actor prediction",
    )
    actor_target = _as_finite_matrix(target.actor_direction, name="actor target")
    camera_prediction = _as_finite_matrix(
        prediction.camera_direction,
        name="camera prediction",
    )
    camera_target = _as_finite_matrix(
        target.camera_direction,
        name="camera target",
    )
    rows = len(actor_prediction)
    if (
        actor_target.shape != actor_prediction.shape
        or len(camera_prediction) != rows
        or camera_target.shape != camera_prediction.shape
    ):
        raise ValueError("factorized per-sample direction shapes differ")
    actor_prediction_magnitude = np.asarray(
        prediction.actor_log_magnitude,
        dtype=np.float32,
    ).reshape(-1)
    actor_target_magnitude = np.asarray(
        target.actor_log_magnitude,
        dtype=np.float32,
    ).reshape(-1)
    camera_prediction_magnitude = np.asarray(
        prediction.camera_log_magnitude,
        dtype=np.float32,
    ).reshape(-1)
    camera_target_magnitude = np.asarray(
        target.camera_log_magnitude,
        dtype=np.float32,
    ).reshape(-1)
    metadata = (
        tuple(str(value) for value in iids),
        tuple(str(value) for value in splits),
        tuple(str(value) for value in action_signatures),
    )
    if any(len(values) != rows for values in metadata) or any(
        len(values) != rows
        for values in (
            actor_prediction_magnitude,
            actor_target_magnitude,
            camera_prediction_magnitude,
            camera_target_magnitude,
        )
    ):
        raise ValueError("factorized per-sample metadata/magnitude length mismatch")
    if valid_mask is None:
        validity = np.ones(rows, dtype=bool)
    else:
        validity = np.asarray(valid_mask, dtype=bool)
        if validity.shape != (rows,):
            raise ValueError("valid_mask must have shape [N]")
    actor_cosine = np.sum(actor_prediction * actor_target, axis=1)
    camera_cosine = np.sum(camera_prediction * camera_target, axis=1)
    output: list[dict[str, Any]] = []
    for index in range(rows):
        output.append(
            {
                "iid": metadata[0][index],
                "split": metadata[1][index],
                "action_signature": metadata[2][index],
                "arm": str(arm),
                "data_seed": int(data_seed),
                "model_seed": int(model_seed),
                "control_valid": bool(validity[index]),
                "actor_direction_cosine": float(actor_cosine[index]),
                "actor_log_magnitude_absolute_error": float(
                    abs(
                        actor_prediction_magnitude[index]
                        - actor_target_magnitude[index]
                    )
                ),
                "actor_target_log_magnitude": float(
                    actor_target_magnitude[index]
                ),
                "camera_direction_cosine": float(camera_cosine[index]),
                "camera_log_magnitude_absolute_error": float(
                    abs(
                        camera_prediction_magnitude[index]
                        - camera_target_magnitude[index]
                    )
                ),
                "camera_target_log_magnitude": float(
                    camera_target_magnitude[index]
                ),
            }
        )
    return output
