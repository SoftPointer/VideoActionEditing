"""Deterministic joint visual-component splits for R7.

R7 must not split a paired edit, an exact duplicate, a perceptual near
duplicate, or a frozen-DINO near duplicate across train/validation/test.
This module builds one graph over *both* source and target assets, computes
connected components with a deterministic disjoint-set union (DSU), and
assigns complete components to a stable split.

The core intentionally consumes already-extracted per-frame features.  It
does not import a DINO model, decode videos, or perform file I/O.  All graph
relations are recomputed by :func:`audit_r7_visual_split`; callers cannot
attest disjointness merely by supplying component identifiers.

Fresh-IID policy
----------------
The caller must explicitly provide the complete set of IIDs exposed in prior
R5/R6 development/evaluation.  A component containing any such IID is forced
to train.  Because source and target are joined before visual edges, this also
quarantines every current IID visually connected to a previously seen IID.
The canonical prior-IID ledger digest is persisted in result provenance.

This protects IID freshness only for prior IIDs represented in the current
graph.  Detecting a renamed copy of a historical video requires including its
features in the candidate graph (or pre-joining it upstream); an IID ledger
alone cannot establish that fact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np


R7_VISUAL_SPLIT_SCHEMA = "motive-r7-joint-visual-split-v1"
R7_VISUAL_SPLIT_VERSION = (
    "joint-source-target-exact-phash-dino-dsu-v1"
)
R7_DINO_PROVENANCE_SCHEMA = "motive-r7-frozen-dino-features-v1"
R7_FRESHNESS_POLICY_VERSION = "prior-iid-component-forced-train-v1"
R7_IMPLEMENTATION_VERSION = "r7-visual-split-numpy-v1"
VALID_SPLITS = frozenset({"train", "validation", "test"})
VALID_ROLES = frozenset({"source", "target"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_INT_BIT_COUNT = getattr(int, "bit_count", None)
_Node = tuple[str, str]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value.astype("<f4", copy=False))
    return _canonical_digest(
        {
            "dtype": "float32-little-endian",
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    )


def _popcount(value: int) -> int:
    return (
        int(_INT_BIT_COUNT(value))
        if _INT_BIT_COUNT is not None
        else bin(value).count("1")
    )


def _normalized_iid(value: Any, *, name: str) -> str:
    iid = str(value).strip()
    if not iid:
        raise ValueError(f"{name} must be non-empty")
    if "\x00" in iid:
        raise ValueError(f"{name} must not contain NUL")
    return iid


def _validated_sha256(value: Any, *, name: str) -> str:
    raw = str(value).strip()
    digest = raw.lower()
    if _SHA256_RE.fullmatch(digest) is None or raw != digest:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _validated_phash(value: Any, *, name: str) -> str:
    encoded = str(value).strip().lower()
    if not encoded or len(encoded) % 2:
        raise ValueError(f"{name} must be non-empty even-length hexadecimal")
    try:
        bytes.fromhex(encoded)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return encoded


@dataclass(frozen=True)
class R7DinoProvenance:
    """Immutable provenance shared by every per-frame DINO feature."""

    encoder_id: str
    encoder_revision: str
    weights_sha256: str
    frame_sampling_version: str
    preprocessing_version: str
    pooling: str
    embedding_dim: int
    dtype: str = "float32"
    normalization: str = "cosine_l2_at_split"
    frozen_encoder: bool = True
    schema_version: str = R7_DINO_PROVENANCE_SCHEMA

    def validate(self) -> None:
        for name in (
            "encoder_id",
            "frame_sampling_version",
            "preprocessing_version",
            "pooling",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"DINO provenance {name} is empty")
        raw_revision = str(self.encoder_revision).strip()
        revision = raw_revision.lower()
        if (
            _IMMUTABLE_REVISION_RE.fullmatch(revision) is None
            or raw_revision != revision
        ):
            raise ValueError(
                "DINO encoder_revision must be an immutable hexadecimal "
                "revision"
            )
        _validated_sha256(
            self.weights_sha256,
            name="DINO provenance weights_sha256",
        )
        if (
            isinstance(self.embedding_dim, bool)
            or not isinstance(self.embedding_dim, int)
            or self.embedding_dim < 1
        ):
            raise ValueError("DINO embedding_dim must be a positive integer")
        if self.dtype != "float32":
            raise ValueError("R7 visual split requires float32 DINO features")
        if self.normalization != "cosine_l2_at_split":
            raise ValueError(
                "unsupported DINO normalization contract; expected "
                "'cosine_l2_at_split'"
            )
        if self.frozen_encoder is not True:
            raise ValueError("R7 visual split requires a frozen DINO encoder")
        if self.schema_version != R7_DINO_PROVENANCE_SCHEMA:
            raise ValueError("unsupported R7 DINO provenance schema")

    def digest(self) -> str:
        self.validate()
        return _canonical_digest(asdict(self))


@dataclass(frozen=True)
class R7VisualAsset:
    """Features for one source or target video on common sampled frames."""

    video_sha256: str
    frame_indices: tuple[int, ...]
    perceptual_hashes: tuple[str, ...]
    dino_embeddings: np.ndarray

    @classmethod
    def create(
        cls,
        *,
        video_sha256: str,
        frame_indices: Sequence[int],
        perceptual_hashes: Sequence[str],
        dino_embeddings: Any,
    ) -> "R7VisualAsset":
        digest = _validated_sha256(
            video_sha256,
            name="video_sha256",
        )
        indices: list[int] = []
        for offset, value in enumerate(frame_indices):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"frame_indices[{offset}] must be an integer")
            index = int(value)
            if index < 0:
                raise ValueError(
                    f"frame_indices[{offset}] must be non-negative"
                )
            indices.append(index)
        if not indices:
            raise ValueError("frame_indices must contain at least one frame")
        if any(
            right <= left for left, right in zip(indices, indices[1:])
        ):
            raise ValueError("frame_indices must be strictly increasing")

        hashes = tuple(
            _validated_phash(value, name=f"perceptual_hashes[{offset}]")
            for offset, value in enumerate(perceptual_hashes)
        )
        if len(hashes) != len(indices):
            raise ValueError(
                "perceptual_hashes must align one-to-one with frame_indices"
            )
        hash_lengths = {len(value) for value in hashes}
        if len(hash_lengths) != 1:
            raise ValueError(
                "perceptual_hashes within one asset must share a bit length"
            )

        raw = np.asarray(dino_embeddings)
        if raw.dtype.kind != "f":
            raise ValueError("dino_embeddings must use a floating dtype")
        matrix = np.asarray(raw, dtype=np.float32)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != len(indices)
            or matrix.shape[1] < 1
        ):
            raise ValueError(
                "dino_embeddings must have shape "
                "[len(frame_indices), embedding_dim]"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("dino_embeddings contain non-finite values")
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if bool((norms <= 1e-12).any()):
            raise ValueError("dino_embeddings contain a zero frame vector")
        frozen = np.ascontiguousarray(matrix)
        frozen.setflags(write=False)
        return cls(
            video_sha256=digest,
            frame_indices=tuple(indices),
            perceptual_hashes=hashes,
            dino_embeddings=frozen,
        )

    def digest(self) -> str:
        return _canonical_digest(
            {
                "video_sha256": self.video_sha256,
                "frame_indices": list(self.frame_indices),
                "perceptual_hashes": list(self.perceptual_hashes),
                "dino_embeddings": _array_digest(self.dino_embeddings),
            }
        )


@dataclass(frozen=True)
class R7VisualPair:
    """One edit IID and its source/target visual assets."""

    iid: str
    source: R7VisualAsset
    target: R7VisualAsset

    @classmethod
    def create(
        cls,
        *,
        iid: str,
        source: R7VisualAsset,
        target: R7VisualAsset,
    ) -> "R7VisualPair":
        if not isinstance(source, R7VisualAsset):
            raise TypeError("source must be an R7VisualAsset")
        if not isinstance(target, R7VisualAsset):
            raise TypeError("target must be an R7VisualAsset")
        return cls(
            iid=_normalized_iid(iid, name="iid"),
            source=source,
            target=target,
        )

    def digest(self) -> str:
        return _canonical_digest(
            {
                "iid": self.iid,
                "source": self.source.digest(),
                "target": self.target.digest(),
            }
        )


@dataclass(frozen=True)
class R7VisualSplitConfig:
    """Thresholds and deterministic split policy."""

    data_seed: int = 260108828
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    maximum_phash_hamming_fraction: float = 0.10
    minimum_dino_cosine: float = 0.95
    split_version: str = R7_VISUAL_SPLIT_VERSION
    freshness_policy_version: str = R7_FRESHNESS_POLICY_VERSION

    def validate(self) -> None:
        if isinstance(self.data_seed, bool) or not isinstance(
            self.data_seed, int
        ):
            raise ValueError("data_seed must be an integer")
        if self.data_seed < 0:
            raise ValueError("data_seed must be non-negative")
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.maximum_phash_hamming_fraction,
            self.minimum_dino_cosine,
        )
        if any(isinstance(value, bool) for value in fractions):
            raise ValueError("R7 visual split thresholds must not be boolean")
        if not all(math.isfinite(float(value)) for value in fractions):
            raise ValueError("R7 visual split thresholds must be finite")
        if (
            self.train_fraction <= 0.0
            or self.validation_fraction <= 0.0
            or self.train_fraction + self.validation_fraction >= 1.0
        ):
            raise ValueError(
                "train_fraction and validation_fraction must be positive "
                "and leave a positive test fraction"
            )
        if not 0.0 <= self.maximum_phash_hamming_fraction < 1.0:
            raise ValueError(
                "maximum_phash_hamming_fraction must be in [0,1)"
            )
        if not 0.0 < self.minimum_dino_cosine <= 1.0:
            raise ValueError("minimum_dino_cosine must be in (0,1]")
        if self.split_version != R7_VISUAL_SPLIT_VERSION:
            raise ValueError("unsupported R7 visual split version")
        if self.freshness_policy_version != R7_FRESHNESS_POLICY_VERSION:
            raise ValueError("unsupported R7 freshness policy version")

    def digest(self) -> str:
        self.validate()
        return _canonical_digest(asdict(self))


@dataclass(frozen=True, order=True)
class R7VisualEdge:
    """One auditable graph relation between two visual assets."""

    left_iid: str
    left_role: str
    right_iid: str
    right_role: str
    relation: str
    value: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class R7VisualAssignment:
    iid: str
    split: str
    component_id: str
    evaluation_fresh: bool
    forced_train_by_seen_component: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class R7VisualComponent:
    component_id: str
    member_nodes: tuple[tuple[str, str], ...]
    member_iids: tuple[str, ...]
    split: str
    seen_iids: tuple[str, ...]
    forced_train_by_seen_component: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class R7VisualSplitAudit:
    samples: int
    assets: int
    components: int
    split_counts: tuple[tuple[str, int], ...]
    component_split_counts: tuple[tuple[str, int], ...]
    edge_counts: tuple[tuple[str, int], ...]
    cross_split_component_ids: tuple[str, ...]
    cross_split_relation_edges: tuple[R7VisualEdge, ...]
    assignment_component_mismatches: tuple[str, ...]
    stable_split_mismatches: tuple[str, ...]
    seen_component_evaluation_iids: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cross_split_relation_edges"] = [
            edge.to_dict() for edge in self.cross_split_relation_edges
        ]
        return value


@dataclass(frozen=True)
class R7VisualSplitProvenance:
    schema_version: str
    implementation_version: str
    split_version: str
    freshness_policy_version: str
    config_digest: str
    dino_provenance_digest: str
    input_pairs_digest: str
    prior_seen_iid_ledger_digest: str
    prior_seen_iid_count: int
    matched_prior_seen_iids: tuple[str, ...]
    edges_digest: str
    components_digest: str
    assignments_digest: str
    audit_digest: str
    provenance_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class R7VisualSplitResult:
    assignments: tuple[R7VisualAssignment, ...]
    components: tuple[R7VisualComponent, ...]
    edges: tuple[R7VisualEdge, ...]
    audit: R7VisualSplitAudit
    provenance: R7VisualSplitProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignments": [
                assignment.to_dict() for assignment in self.assignments
            ],
            "components": [
                component.to_dict() for component in self.components
            ],
            "edges": [edge.to_dict() for edge in self.edges],
            "audit": self.audit.to_dict(),
            "provenance": self.provenance.to_dict(),
        }


class _DeterministicDsu:
    def __init__(self, nodes: Sequence[_Node]) -> None:
        self._parent: dict[_Node, _Node] = {node: node for node in nodes}

    def find(self, node: _Node) -> _Node:
        parent = self._parent[node]
        while parent != self._parent[parent]:
            parent = self._parent[parent]
        while node != parent:
            previous = self._parent[node]
            self._parent[node] = parent
            node = previous
        return parent

    def union(self, left: _Node, right: _Node) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        lower, upper = sorted((left_root, right_root))
        self._parent[upper] = lower


def _normalize_prior_seen_iids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            "previously_seen_iids must be a sequence, not one string"
        )
    normalized = tuple(
        _normalized_iid(value, name=f"previously_seen_iids[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("previously_seen_iids must be unique")
    return tuple(sorted(normalized))


def _normalize_pairs(
    pairs: Sequence[R7VisualPair],
    *,
    dino_provenance: R7DinoProvenance,
) -> tuple[R7VisualPair, ...]:
    dino_provenance.validate()
    values = tuple(pairs)
    if not values:
        raise ValueError("R7 visual split requires at least one pair")
    if any(not isinstance(pair, R7VisualPair) for pair in values):
        raise TypeError("pairs must contain only R7VisualPair values")
    iids = tuple(pair.iid for pair in values)
    if any(_normalized_iid(iid, name="pair iid") != iid for iid in iids):
        raise ValueError("pair IID is not normalized; use R7VisualPair.create")
    if len(set(iids)) != len(iids):
        raise ValueError("R7 visual pair IIDs must be unique")

    phash_lengths: set[int] = set()
    for pair in values:
        for role in ("source", "target"):
            asset = getattr(pair, role)
            if not isinstance(asset, R7VisualAsset):
                raise TypeError(f"{pair.iid} {role} is not R7VisualAsset")
            # Revalidate fields so direct dataclass construction cannot bypass
            # the fail-closed create contract.
            _validated_sha256(
                asset.video_sha256,
                name=f"{pair.iid}.{role}.video_sha256",
            )
            if asset.video_sha256 != asset.video_sha256.lower():
                raise ValueError(
                    f"{pair.iid}.{role} video SHA-256 is not normalized"
                )
            if (
                not asset.frame_indices
                or len(asset.frame_indices) != len(asset.perceptual_hashes)
                or len(asset.frame_indices) != len(asset.dino_embeddings)
            ):
                raise ValueError(
                    f"{pair.iid}.{role} frame feature lengths differ"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
                for value in asset.frame_indices
            ):
                raise ValueError(
                    f"{pair.iid}.{role} has invalid frame_indices"
                )
            if any(
                right <= left
                for left, right in zip(
                    asset.frame_indices, asset.frame_indices[1:]
                )
            ):
                raise ValueError(
                    f"{pair.iid}.{role} frame_indices are not increasing"
                )
            for index, phash in enumerate(asset.perceptual_hashes):
                normalized = _validated_phash(
                    phash,
                    name=f"{pair.iid}.{role}.perceptual_hashes[{index}]",
                )
                if normalized != phash:
                    raise ValueError(
                        f"{pair.iid}.{role} pHash is not normalized"
                    )
                phash_lengths.add(len(phash))
            matrix = np.asarray(asset.dino_embeddings)
            if (
                matrix.dtype != np.dtype(np.float32)
                or matrix.ndim != 2
                or matrix.shape[1] != dino_provenance.embedding_dim
                or not np.isfinite(matrix).all()
            ):
                raise ValueError(
                    f"{pair.iid}.{role} DINO matrix violates provenance"
                )
            if matrix.flags.writeable:
                raise ValueError(
                    f"{pair.iid}.{role} DINO matrix must be immutable; "
                    "use R7VisualAsset.create"
                )
            norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
            if bool((norms <= 1e-12).any()):
                raise ValueError(
                    f"{pair.iid}.{role} DINO matrix contains a zero row"
                )
    if len(phash_lengths) != 1:
        raise ValueError(
            "all source/target perceptual hashes must share a bit length"
        )
    return tuple(sorted(values, key=lambda pair: pair.iid))


def _asset_by_node(
    pairs: Sequence[R7VisualPair],
) -> tuple[tuple[str, str, R7VisualAsset], ...]:
    return tuple(
        (pair.iid, role, getattr(pair, role))
        for pair in pairs
        for role in ("source", "target")
    )  # type: ignore[return-value]


def _minimum_phash_distance(
    left: R7VisualAsset,
    right: R7VisualAsset,
) -> float:
    bits = len(left.perceptual_hashes[0]) * 4
    left_values = [
        int(value, 16) for value in left.perceptual_hashes
    ]
    right_values = [
        int(value, 16) for value in right.perceptual_hashes
    ]
    minimum = min(
        _popcount(left_value ^ right_value)
        for left_value in left_values
        for right_value in right_values
    )
    return float(minimum / bits)


def _maximum_dino_cosine(
    left: R7VisualAsset,
    right: R7VisualAsset,
) -> float:
    left_matrix = left.dino_embeddings.astype(np.float64)
    right_matrix = right.dino_embeddings.astype(np.float64)
    left_matrix = left_matrix / np.linalg.norm(
        left_matrix, axis=1, keepdims=True
    )
    right_matrix = right_matrix / np.linalg.norm(
        right_matrix, axis=1, keepdims=True
    )
    maximum = float(np.max(left_matrix @ right_matrix.T))
    return float(min(1.0, max(-1.0, maximum)))


def _ordered_edge(
    left: _Node,
    right: _Node,
    *,
    relation: str,
    value: float | None,
) -> R7VisualEdge:
    if right < left:
        left, right = right, left
    return R7VisualEdge(
        left_iid=left[0],
        left_role=left[1],
        right_iid=right[0],
        right_role=right[1],
        relation=relation,
        value=(
            None if value is None else float(round(float(value), 12))
        ),
    )


def _relation_edges(
    pairs: Sequence[R7VisualPair],
    *,
    config: R7VisualSplitConfig,
) -> tuple[R7VisualEdge, ...]:
    edges: list[R7VisualEdge] = []
    for pair in pairs:
        edges.append(
            _ordered_edge(
                (pair.iid, "source"),
                (pair.iid, "target"),
                relation="paired_sample",
                value=None,
            )
        )
    nodes_with_assets = _asset_by_node(pairs)
    for left_index in range(len(nodes_with_assets)):
        left_iid, left_role, left_asset = nodes_with_assets[left_index]
        left_node = (left_iid, left_role)
        for right_index in range(left_index + 1, len(nodes_with_assets)):
            right_iid, right_role, right_asset = nodes_with_assets[right_index]
            right_node = (right_iid, right_role)
            if left_asset.video_sha256 == right_asset.video_sha256:
                edges.append(
                    _ordered_edge(
                        left_node,
                        right_node,
                        relation="exact_digest",
                        value=1.0,
                    )
                )
            phash_distance = _minimum_phash_distance(
                left_asset, right_asset
            )
            if phash_distance <= config.maximum_phash_hamming_fraction:
                edges.append(
                    _ordered_edge(
                        left_node,
                        right_node,
                        relation="phash",
                        value=phash_distance,
                    )
                )
            dino_cosine = _maximum_dino_cosine(left_asset, right_asset)
            if dino_cosine >= config.minimum_dino_cosine:
                edges.append(
                    _ordered_edge(
                        left_node,
                        right_node,
                        relation="dino_cosine",
                        value=dino_cosine,
                    )
                )
    return tuple(sorted(set(edges)))


def _nodes(pairs: Sequence[R7VisualPair]) -> tuple[_Node, ...]:
    return tuple(
        (pair.iid, role)
        for pair in pairs
        for role in ("source", "target")
    )


def _components_from_edges(
    pairs: Sequence[R7VisualPair],
    edges: Sequence[R7VisualEdge],
) -> tuple[tuple[_Node, ...], ...]:
    dsu = _DeterministicDsu(_nodes(pairs))
    for edge in edges:
        dsu.union(
            (edge.left_iid, edge.left_role),
            (edge.right_iid, edge.right_role),
        )
    grouped: dict[_Node, list[_Node]] = {}
    for node in _nodes(pairs):
        grouped.setdefault(dsu.find(node), []).append(node)
    return tuple(
        sorted(
            (tuple(sorted(members)) for members in grouped.values()),
            key=lambda members: members,
        )
    )


def _component_id(member_nodes: Sequence[_Node]) -> str:
    digest = _canonical_digest(
        {
            "split_version": R7_VISUAL_SPLIT_VERSION,
            "member_nodes": [list(node) for node in sorted(member_nodes)],
        }
    )
    return f"r7vc-{digest}"


def _stable_split(
    component_id: str,
    *,
    config: R7VisualSplitConfig,
) -> str:
    digest = hashlib.sha256(
        f"{config.data_seed}\0{component_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "little") % 1_000_000
    train_end = int(round(config.train_fraction * 1_000_000))
    validation_end = int(
        round(
            (config.train_fraction + config.validation_fraction)
            * 1_000_000
        )
    )
    if bucket < train_end:
        return "train"
    if bucket < validation_end:
        return "validation"
    return "test"


def _expected_components_and_assignments(
    pairs: Sequence[R7VisualPair],
    edges: Sequence[R7VisualEdge],
    *,
    config: R7VisualSplitConfig,
    previously_seen_iids: Sequence[str],
) -> tuple[
    tuple[R7VisualComponent, ...],
    tuple[R7VisualAssignment, ...],
]:
    seen = set(previously_seen_iids)
    components: list[R7VisualComponent] = []
    assignments: list[R7VisualAssignment] = []
    for member_nodes in _components_from_edges(pairs, edges):
        component_id = _component_id(member_nodes)
        member_iids = tuple(sorted({iid for iid, _role in member_nodes}))
        seen_iids = tuple(sorted(set(member_iids) & seen))
        forced = bool(seen_iids)
        split = (
            "train"
            if forced
            else _stable_split(component_id, config=config)
        )
        components.append(
            R7VisualComponent(
                component_id=component_id,
                member_nodes=member_nodes,
                member_iids=member_iids,
                split=split,
                seen_iids=seen_iids,
                forced_train_by_seen_component=forced,
            )
        )
        assignments.extend(
            R7VisualAssignment(
                iid=iid,
                split=split,
                component_id=component_id,
                evaluation_fresh=not forced,
                forced_train_by_seen_component=forced,
            )
            for iid in member_iids
        )
    return (
        tuple(sorted(components, key=lambda value: value.component_id)),
        tuple(sorted(assignments, key=lambda value: value.iid)),
    )


def audit_r7_visual_split(
    pairs: Sequence[R7VisualPair],
    assignments: Sequence[R7VisualAssignment],
    *,
    config: R7VisualSplitConfig,
    dino_provenance: R7DinoProvenance,
    previously_seen_iids: Sequence[str],
    raise_on_failure: bool = True,
) -> R7VisualSplitAudit:
    """Independently recompute graph relations and audit a proposed split.

    Missing/extra assignments, malformed features, wrong component IDs,
    non-stable split assignments, seen-IID evaluation, and every cross-split
    relation fail closed.
    """

    config.validate()
    normalized_pairs = _normalize_pairs(
        pairs, dino_provenance=dino_provenance
    )
    seen_iids = _normalize_prior_seen_iids(previously_seen_iids)
    proposed = tuple(assignments)
    if any(
        not isinstance(value, R7VisualAssignment) for value in proposed
    ):
        raise TypeError(
            "assignments must contain only R7VisualAssignment values"
        )
    proposed_iids = [assignment.iid for assignment in proposed]
    if any(
        _normalized_iid(iid, name="assignment iid") != iid
        for iid in proposed_iids
    ):
        raise ValueError("assignment IIDs must be normalized")
    if any(
        type(assignment.evaluation_fresh) is not bool
        or type(assignment.forced_train_by_seen_component) is not bool
        for assignment in proposed
    ):
        raise ValueError("assignment freshness flags must be booleans")
    expected_iids = [pair.iid for pair in normalized_pairs]
    if (
        len(set(proposed_iids)) != len(proposed_iids)
        or set(proposed_iids) != set(expected_iids)
    ):
        raise ValueError(
            "assignments must contain every pair IID exactly once"
        )
    invalid_splits = sorted(
        {
            assignment.split
            for assignment in proposed
            if assignment.split not in VALID_SPLITS
        }
    )
    if invalid_splits:
        raise ValueError(f"invalid R7 split values: {invalid_splits}")

    edges = _relation_edges(normalized_pairs, config=config)
    expected_components, expected_assignments = (
        _expected_components_and_assignments(
            normalized_pairs,
            edges,
            config=config,
            previously_seen_iids=seen_iids,
        )
    )
    expected_by_iid = {
        assignment.iid: assignment
        for assignment in expected_assignments
    }
    proposed_by_iid = {
        assignment.iid: assignment for assignment in proposed
    }

    assignment_component_mismatches: list[str] = []
    stable_split_mismatches: list[str] = []
    for iid in expected_iids:
        expected = expected_by_iid[iid]
        actual = proposed_by_iid[iid]
        if (
            actual.component_id != expected.component_id
            or actual.evaluation_fresh != expected.evaluation_fresh
            or actual.forced_train_by_seen_component
            != expected.forced_train_by_seen_component
        ):
            assignment_component_mismatches.append(iid)
        if actual.split != expected.split:
            stable_split_mismatches.append(iid)

    component_splits: dict[str, set[str]] = {}
    for assignment in proposed:
        component_splits.setdefault(
            expected_by_iid[assignment.iid].component_id, set()
        ).add(assignment.split)
    cross_split_components = tuple(
        sorted(
            component_id
            for component_id, split_values in component_splits.items()
            if len(split_values) > 1
        )
    )

    cross_split_edges: list[R7VisualEdge] = []
    for edge in edges:
        left_split = proposed_by_iid[edge.left_iid].split
        right_split = proposed_by_iid[edge.right_iid].split
        if left_split != right_split:
            cross_split_edges.append(edge)

    component_by_id = {
        component.component_id: component
        for component in expected_components
    }
    seen_component_evaluation_iids = tuple(
        sorted(
            assignment.iid
            for assignment in proposed
            if assignment.split != "train"
            and component_by_id[
                expected_by_iid[assignment.iid].component_id
            ].forced_train_by_seen_component
        )
    )

    edge_counter: dict[str, int] = {}
    for edge in edges:
        edge_counter[edge.relation] = (
            edge_counter.get(edge.relation, 0) + 1
        )
    split_counts = tuple(
        (split, sum(value.split == split for value in proposed))
        for split in sorted(VALID_SPLITS)
    )
    component_split_counts = tuple(
        (
            split,
            sum(component.split == split for component in expected_components),
        )
        for split in sorted(VALID_SPLITS)
    )
    passed = not any(
        (
            cross_split_components,
            cross_split_edges,
            assignment_component_mismatches,
            stable_split_mismatches,
            seen_component_evaluation_iids,
        )
    )
    audit = R7VisualSplitAudit(
        samples=len(normalized_pairs),
        assets=2 * len(normalized_pairs),
        components=len(expected_components),
        split_counts=split_counts,
        component_split_counts=component_split_counts,
        edge_counts=tuple(sorted(edge_counter.items())),
        cross_split_component_ids=cross_split_components,
        cross_split_relation_edges=tuple(cross_split_edges),
        assignment_component_mismatches=tuple(
            sorted(assignment_component_mismatches)
        ),
        stable_split_mismatches=tuple(
            sorted(stable_split_mismatches)
        ),
        seen_component_evaluation_iids=(
            seen_component_evaluation_iids
        ),
        passed=passed,
    )
    if raise_on_failure and not passed:
        raise ValueError(
            "R7 visual split audit failed: "
            f"component_collisions={list(cross_split_components)[:8]} "
            f"cross_split_edges="
            f"{[edge.to_dict() for edge in cross_split_edges[:8]]} "
            f"component_mismatches="
            f"{assignment_component_mismatches[:8]} "
            f"stable_split_mismatches={stable_split_mismatches[:8]} "
            f"seen_component_evaluation="
            f"{list(seen_component_evaluation_iids)[:8]}"
        )
    return audit


def build_r7_visual_split(
    pairs: Sequence[R7VisualPair],
    *,
    config: R7VisualSplitConfig,
    dino_provenance: R7DinoProvenance,
    previously_seen_iids: Sequence[str],
) -> R7VisualSplitResult:
    """Build and audit a deterministic R7 joint visual split."""

    config.validate()
    dino_provenance.validate()
    normalized_pairs = _normalize_pairs(
        pairs, dino_provenance=dino_provenance
    )
    seen_iids = _normalize_prior_seen_iids(previously_seen_iids)
    edges = _relation_edges(normalized_pairs, config=config)
    components, assignments = _expected_components_and_assignments(
        normalized_pairs,
        edges,
        config=config,
        previously_seen_iids=seen_iids,
    )
    audit = audit_r7_visual_split(
        normalized_pairs,
        assignments,
        config=config,
        dino_provenance=dino_provenance,
        previously_seen_iids=seen_iids,
    )

    input_pairs_digest = _canonical_digest(
        [
            {"iid": pair.iid, "pair_digest": pair.digest()}
            for pair in normalized_pairs
        ]
    )
    prior_ledger_digest = _canonical_digest(list(seen_iids))
    edges_digest = _canonical_digest(
        [edge.to_dict() for edge in edges]
    )
    components_digest = _canonical_digest(
        [component.to_dict() for component in components]
    )
    assignments_digest = _canonical_digest(
        [assignment.to_dict() for assignment in assignments]
    )
    audit_digest = _canonical_digest(audit.to_dict())
    provenance_base: dict[str, Any] = {
        "schema_version": R7_VISUAL_SPLIT_SCHEMA,
        "implementation_version": R7_IMPLEMENTATION_VERSION,
        "split_version": config.split_version,
        "freshness_policy_version": config.freshness_policy_version,
        "config_digest": config.digest(),
        "dino_provenance_digest": dino_provenance.digest(),
        "input_pairs_digest": input_pairs_digest,
        "prior_seen_iid_ledger_digest": prior_ledger_digest,
        "prior_seen_iid_count": len(seen_iids),
        "matched_prior_seen_iids": tuple(
            sorted(set(pair.iid for pair in normalized_pairs) & set(seen_iids))
        ),
        "edges_digest": edges_digest,
        "components_digest": components_digest,
        "assignments_digest": assignments_digest,
        "audit_digest": audit_digest,
    }
    provenance = R7VisualSplitProvenance(
        **provenance_base,
        provenance_digest=_canonical_digest(provenance_base),
    )
    return R7VisualSplitResult(
        assignments=assignments,
        components=components,
        edges=edges,
        audit=audit,
        provenance=provenance,
    )


def assignments_by_iid(
    result: R7VisualSplitResult,
) -> Mapping[str, R7VisualAssignment]:
    """Return a read-only-by-convention IID lookup for integration code."""

    if not isinstance(result, R7VisualSplitResult):
        raise TypeError("result must be R7VisualSplitResult")
    return {
        assignment.iid: assignment for assignment in result.assignments
    }


def with_assignment_split(
    assignment: R7VisualAssignment,
    split: str,
) -> R7VisualAssignment:
    """Small explicit helper useful for negative audit tests/tools."""

    return replace(assignment, split=str(split))
