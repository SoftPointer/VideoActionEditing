"""Scalable deterministic visual-component graph for R7 expansion.

This module is a pure algorithmic core.  It consumes already verified video
SHA-256 values, six 64-bit dHashes per asset, and externally proposed DINO
asset edges.  It performs no file I/O, video decoding, model inference, or
all-assets quadratic scan.

The graph has two asset nodes for every IID (``source`` and ``target``).
Those nodes are joined first, so an edit pair is atomic under every later
operation.  Exact-video relations are grouped by SHA-256.  Perceptual
relations are discovered exactly with a BK-tree over the 64-bit Hamming
metric: an asset is related when *any* of its six dHashes is at distance at
most six from any frame of another asset.  DINO edges are validated against
the complete asset registry and the configured cosine threshold before they
are joined.

Only deterministic spanning relation edges are retained.  This is enough to
reconstruct the exact connected-component partition while avoiding a
quadratic edge materialization for duplicate-heavy data.  Statistics name
the indexed candidate counts separately from retained spanning-edge counts.

Freshness policy
----------------
Any component containing an ``anchor_iid`` or a currently represented
``previously_seen_iid`` is forced to ``train`` and marked ``fresh=False``.
Every other complete component is assigned by a stable seeded 80/10/10 hash.
Unmatched historical IIDs remain bound in provenance but, naturally, cannot
join a current component without a supplied current visual node.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


R7_INDEXED_VISUAL_GRAPH_SCHEMA = "motive-r7-indexed-visual-graph-v1"
R7_INDEXED_VISUAL_GRAPH_VERSION = (
    "pair-exact-sha-bktree-dhash6-external-dino-dsu-v1"
)
R7_INDEXED_VISUAL_GRAPH_IMPLEMENTATION = (
    "r7-indexed-visual-graph-pure-python-v1"
)
R7_INDEXED_VISUAL_COMPONENT_VERSION = "r7-indexed-visual-component-v1"
R7_INDEXED_VISUAL_SPLIT_VERSION = "seeded-component-80-10-10-v1"

ASSET_ROLES = ("source", "target")
VALID_ASSET_ROLES = frozenset(ASSET_ROLES)
VALID_SPLITS = frozenset({"train", "validation", "test"})
DHASHES_PER_ASSET = 6
DHASH_BITS = 64
MAXIMUM_DHASH_HAMMING = 6
DINO_HARD_THRESHOLD = 0.96
GIANT_COMPONENT_FRACTION = 0.05

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DHASH_RE = re.compile(r"^[0-9a-f]{16}$")
_Node = tuple[str, str]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _normalized_iid(value: Any, *, context: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{context} must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError(
            f"{context} must be a canonical non-empty IID"
        )
    return value


def _normalized_role(value: Any, *, context: str) -> str:
    if type(value) is not str or value not in VALID_ASSET_ROLES:
        raise ValueError(
            f"{context} must be one of {sorted(VALID_ASSET_ROLES)}"
        )
    return value


def _normalized_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{context} must be a lowercase SHA-256 digest"
        )
    return value


def _normalized_dhash(value: Any, *, context: str) -> str:
    if type(value) is not str or _DHASH_RE.fullmatch(value) is None:
        raise ValueError(
            f"{context} must be a lowercase 64-bit hexadecimal dHash"
        )
    return value


def _normalized_iid_ledger(
    values: Sequence[str],
    *,
    context: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{context} must be a sequence, not a string")
    normalized = tuple(
        _normalized_iid(value, context=f"{context}[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{context} must not contain duplicate IIDs")
    return tuple(sorted(normalized))


def _node_to_list(node: _Node) -> list[str]:
    return [node[0], node[1]]


@dataclass(frozen=True)
class R7IndexedVisualAsset:
    """One verified video asset and its six sampled-frame dHashes."""

    video_sha256: str
    dhashes: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        video_sha256: str,
        dhashes: Sequence[str],
    ) -> "R7IndexedVisualAsset":
        digest = _normalized_sha256(
            video_sha256,
            context="video_sha256",
        )
        if isinstance(dhashes, (str, bytes)):
            raise TypeError("dhashes must be a sequence, not a string")
        normalized = tuple(
            _normalized_dhash(
                value,
                context=f"dhashes[{index}]",
            )
            for index, value in enumerate(dhashes)
        )
        if len(normalized) != DHASHES_PER_ASSET:
            raise ValueError(
                f"dhashes must contain exactly {DHASHES_PER_ASSET} "
                "64-bit values"
            )
        return cls(video_sha256=digest, dhashes=normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_sha256": self.video_sha256,
            "dhashes": list(self.dhashes),
        }

    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class R7IndexedVisualPair:
    """One IID with exactly one source and one target asset."""

    iid: str
    source: R7IndexedVisualAsset
    target: R7IndexedVisualAsset

    @classmethod
    def create(
        cls,
        *,
        iid: str,
        source: R7IndexedVisualAsset,
        target: R7IndexedVisualAsset,
    ) -> "R7IndexedVisualPair":
        normalized_iid = _normalized_iid(iid, context="iid")
        if not isinstance(source, R7IndexedVisualAsset):
            raise TypeError("source must be R7IndexedVisualAsset")
        if not isinstance(target, R7IndexedVisualAsset):
            raise TypeError("target must be R7IndexedVisualAsset")
        # Recreate to prevent direct dataclass construction from bypassing
        # the immutable canonical asset contract.
        normalized_source = R7IndexedVisualAsset.create(
            video_sha256=source.video_sha256,
            dhashes=source.dhashes,
        )
        normalized_target = R7IndexedVisualAsset.create(
            video_sha256=target.video_sha256,
            dhashes=target.dhashes,
        )
        return cls(
            iid=normalized_iid,
            source=normalized_source,
            target=normalized_target,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "iid": self.iid,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
        }

    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class R7IndexedDinoEdge:
    """One externally proposed asset-level DINO cosine relation."""

    left_iid: str
    left_role: str
    right_iid: str
    right_role: str
    cosine: float

    @classmethod
    def create(
        cls,
        *,
        left_iid: str,
        left_role: str,
        right_iid: str,
        right_role: str,
        cosine: float,
    ) -> "R7IndexedDinoEdge":
        left = (
            _normalized_iid(left_iid, context="left_iid"),
            _normalized_role(left_role, context="left_role"),
        )
        right = (
            _normalized_iid(right_iid, context="right_iid"),
            _normalized_role(right_role, context="right_role"),
        )
        if left == right:
            raise ValueError("DINO edge endpoints must be distinct assets")
        if isinstance(cosine, bool) or not isinstance(
            cosine, (int, float)
        ):
            raise ValueError("DINO edge cosine must be numeric")
        score = float(cosine)
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise ValueError(
                "DINO edge cosine must be finite and in [-1,1]"
            )
        if right < left:
            left, right = right, left
        return cls(
            left_iid=left[0],
            left_role=left[1],
            right_iid=right[0],
            right_role=right[1],
            cosine=score,
        )

    @property
    def left_node(self) -> _Node:
        return (self.left_iid, self.left_role)

    @property
    def right_node(self) -> _Node:
        return (self.right_iid, self.right_role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_iid": self.left_iid,
            "left_role": self.left_role,
            "right_iid": self.right_iid,
            "right_role": self.right_role,
            "cosine": self.cosine,
        }


@dataclass(frozen=True)
class R7IndexedVisualGraphConfig:
    """Fixed dHash/split policy plus the external-DINO threshold."""

    data_seed: int = 260108828
    maximum_dhash_hamming: int = MAXIMUM_DHASH_HAMMING
    minimum_dino_cosine: float = DINO_HARD_THRESHOLD
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    split_version: str = R7_INDEXED_VISUAL_SPLIT_VERSION

    def validate(self) -> None:
        if (
            isinstance(self.data_seed, bool)
            or not isinstance(self.data_seed, int)
            or self.data_seed < 0
        ):
            raise ValueError("data_seed must be a non-negative integer")
        if (
            isinstance(self.maximum_dhash_hamming, bool)
            or not isinstance(self.maximum_dhash_hamming, int)
            or self.maximum_dhash_hamming != MAXIMUM_DHASH_HAMMING
        ):
            raise ValueError(
                "R7 indexed graph requires maximum_dhash_hamming=6"
            )
        if isinstance(self.minimum_dino_cosine, bool) or not isinstance(
            self.minimum_dino_cosine, (int, float)
        ):
            raise ValueError("minimum_dino_cosine must be numeric")
        if (
            not math.isfinite(float(self.minimum_dino_cosine))
            or not 0.0 < float(self.minimum_dino_cosine) <= 1.0
        ):
            raise ValueError("minimum_dino_cosine must be in (0,1]")
        if (
            isinstance(self.train_fraction, bool)
            or isinstance(self.validation_fraction, bool)
            or not isinstance(self.train_fraction, (int, float))
            or not isinstance(self.validation_fraction, (int, float))
            or not math.isfinite(float(self.train_fraction))
            or not math.isfinite(float(self.validation_fraction))
            or float(self.train_fraction) != 0.8
            or float(self.validation_fraction) != 0.1
        ):
            raise ValueError(
                "R7 indexed graph requires a fixed 80/10/10 split"
            )
        if self.split_version != R7_INDEXED_VISUAL_SPLIT_VERSION:
            raise ValueError("unsupported indexed visual split version")

    def thresholds_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "maximum_dhash_hamming": self.maximum_dhash_hamming,
            "minimum_dino_cosine": float(self.minimum_dino_cosine),
            "giant_component_fraction": GIANT_COMPONENT_FRACTION,
        }

    def split_policy_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "data_seed": self.data_seed,
            "train_fraction": float(self.train_fraction),
            "validation_fraction": float(self.validation_fraction),
            "test_fraction": 0.1,
            "split_version": self.split_version,
        }

    def digest(self) -> str:
        self.validate()
        return _digest(
            {
                "thresholds": self.thresholds_dict(),
                "split_policy": self.split_policy_dict(),
            }
        )


@dataclass(frozen=True)
class R7IndexedRelationEdge:
    """One retained edge in the deterministic spanning relation graph."""

    left_iid: str
    left_role: str
    right_iid: str
    right_role: str
    relation: str
    value: int | float | None

    @property
    def left_node(self) -> _Node:
        return (self.left_iid, self.left_role)

    @property
    def right_node(self) -> _Node:
        return (self.right_iid, self.right_role)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class R7IndexedVisualAssignment:
    iid: str
    component_id: str
    split: str
    fresh: bool
    forced_train: bool
    forced_by_anchor: bool
    forced_by_previously_seen: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class R7IndexedVisualComponent:
    component_id: str
    member_assets: tuple[_Node, ...]
    member_iids: tuple[str, ...]
    split: str
    fresh: bool
    forced_train: bool
    anchor_iids: tuple[str, ...]
    previously_seen_iids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "member_assets": [
                _node_to_list(node) for node in self.member_assets
            ],
            "member_iids": list(self.member_iids),
            "split": self.split,
            "fresh": self.fresh,
            "forced_train": self.forced_train,
            "anchor_iids": list(self.anchor_iids),
            "previously_seen_iids": list(self.previously_seen_iids),
        }


@dataclass(frozen=True)
class R7IndexedVisualGraphStatistics:
    pair_count: int
    asset_count: int
    component_count: int
    relation_counts: tuple[tuple[str, int], ...]
    indexed_candidate_relation_counts: tuple[tuple[str, int], ...]
    split_iid_counts: tuple[tuple[str, int], ...]
    split_component_counts: tuple[tuple[str, int], ...]
    forced_component_count: int
    forced_iid_count: int
    fresh_iid_count: int
    dino_input_edge_count: int
    dino_above_threshold_count: int
    dino_below_threshold_count: int
    dhash_index_unique_values: int
    dhash_index_queries: int
    dhash_index_distance_evaluations: int
    largest_component_iids: int
    largest_component_assets: int
    largest_component_fraction: float
    giant_component_threshold: float
    giant_component_warning: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in (
            "relation_counts",
            "indexed_candidate_relation_counts",
            "split_iid_counts",
            "split_component_counts",
        ):
            value[field] = dict(getattr(self, field))
        return value


@dataclass(frozen=True)
class R7IndexedVisualGraphProvenance:
    schema_version: str
    graph_version: str
    implementation_version: str
    config_digest: str
    input_pairs_digest: str
    dino_input_edges_digest: str
    anchor_iids_digest: str
    previously_seen_iids_digest: str
    input_digest: str
    threshold_digest: str
    edge_digest: str
    component_digest: str
    assignment_digest: str
    statistics_digest: str
    provenance_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class R7IndexedVisualGraphResult:
    assignments: tuple[R7IndexedVisualAssignment, ...]
    components: tuple[R7IndexedVisualComponent, ...]
    spanning_edges: tuple[R7IndexedRelationEdge, ...]
    statistics: R7IndexedVisualGraphStatistics
    provenance: R7IndexedVisualGraphProvenance

    @property
    def edges(self) -> tuple[R7IndexedRelationEdge, ...]:
        """Compatibility alias emphasizing that only spanning edges persist."""

        return self.spanning_edges

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignments": [
                assignment.to_dict() for assignment in self.assignments
            ],
            "components": [
                component.to_dict() for component in self.components
            ],
            "spanning_edges": [
                edge.to_dict() for edge in self.spanning_edges
            ],
            "statistics": self.statistics.to_dict(),
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

    def union(self, left: _Node, right: _Node) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        lower, upper = sorted((left_root, right_root))
        self._parent[upper] = lower
        return True


class _BkNode:
    __slots__ = ("value", "representative", "children")

    def __init__(self, value: int, representative: _Node) -> None:
        self.value = value
        self.representative = representative
        self.children: dict[int, "_BkNode"] = {}


class _HammingBkTree:
    """BK-tree retaining one sufficient representative per exact dHash."""

    def __init__(self) -> None:
        self._root: _BkNode | None = None
        self.unique_values = 0
        self.queries = 0
        self.distance_evaluations = 0

    @staticmethod
    def _distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def insert(self, value: int, representative: _Node) -> None:
        if self._root is None:
            self._root = _BkNode(value, representative)
            self.unique_values = 1
            return
        node = self._root
        while True:
            distance = self._distance(value, node.value)
            if distance == 0:
                # Every asset carrying this exact value is connected when it
                # queries the existing representative.  Keeping the first
                # deterministic representative is therefore sufficient.
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BkNode(value, representative)
                self.unique_values += 1
                return
            node = child

    def query(
        self,
        value: int,
        maximum_distance: int,
    ) -> tuple[tuple[_Node, int], ...]:
        self.queries += 1
        if self._root is None:
            return ()
        matches: list[tuple[_Node, int]] = []
        pending = [self._root]
        while pending:
            node = pending.pop()
            distance = self._distance(value, node.value)
            self.distance_evaluations += 1
            if distance <= maximum_distance:
                matches.append((node.representative, distance))
            lower = distance - maximum_distance
            upper = distance + maximum_distance
            for child_distance in sorted(node.children, reverse=True):
                if lower <= child_distance <= upper:
                    pending.append(node.children[child_distance])
        return tuple(matches)


def _normalize_pairs(
    pairs: Sequence[R7IndexedVisualPair],
) -> tuple[R7IndexedVisualPair, ...]:
    if isinstance(pairs, (str, bytes)):
        raise TypeError("pairs must be a sequence")
    values = tuple(pairs)
    if not values:
        raise ValueError("indexed visual graph requires at least one pair")
    normalized: list[R7IndexedVisualPair] = []
    for index, pair in enumerate(values):
        if not isinstance(pair, R7IndexedVisualPair):
            raise TypeError(
                f"pairs[{index}] must be R7IndexedVisualPair"
            )
        normalized.append(
            R7IndexedVisualPair.create(
                iid=pair.iid,
                source=pair.source,
                target=pair.target,
            )
        )
    iids = [pair.iid for pair in normalized]
    if len(set(iids)) != len(iids):
        raise ValueError("pair IIDs must be unique")
    return tuple(sorted(normalized, key=lambda value: value.iid))


def _normalize_dino_edges(
    edges: Sequence[R7IndexedDinoEdge],
    *,
    valid_nodes: set[_Node],
) -> tuple[R7IndexedDinoEdge, ...]:
    if isinstance(edges, (str, bytes)):
        raise TypeError("dino_edges must be a sequence")
    normalized: list[R7IndexedDinoEdge] = []
    endpoint_pairs: set[tuple[_Node, _Node]] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, R7IndexedDinoEdge):
            raise TypeError(
                f"dino_edges[{index}] must be R7IndexedDinoEdge"
            )
        value = R7IndexedDinoEdge.create(
            left_iid=edge.left_iid,
            left_role=edge.left_role,
            right_iid=edge.right_iid,
            right_role=edge.right_role,
            cosine=edge.cosine,
        )
        if value.left_node not in valid_nodes:
            raise ValueError(
                f"DINO edge {index} has unknown left asset "
                f"{value.left_node!r}"
            )
        if value.right_node not in valid_nodes:
            raise ValueError(
                f"DINO edge {index} has unknown right asset "
                f"{value.right_node!r}"
            )
        key = (value.left_node, value.right_node)
        if key in endpoint_pairs:
            raise ValueError(
                "DINO edges must contain each unordered asset pair once"
            )
        endpoint_pairs.add(key)
        normalized.append(value)
    return tuple(
        sorted(
            normalized,
            key=lambda value: (
                value.left_node,
                value.right_node,
                value.cosine,
            ),
        )
    )


def _ordered_relation_edge(
    left: _Node,
    right: _Node,
    *,
    relation: str,
    value: int | float | None,
) -> R7IndexedRelationEdge:
    if right < left:
        left, right = right, left
    if isinstance(value, float):
        value = float(round(value, 12))
    return R7IndexedRelationEdge(
        left_iid=left[0],
        left_role=left[1],
        right_iid=right[0],
        right_role=right[1],
        relation=relation,
        value=value,
    )


def _component_id(member_assets: Sequence[_Node]) -> str:
    value = {
        "component_version": R7_INDEXED_VISUAL_COMPONENT_VERSION,
        "member_assets": [
            _node_to_list(node) for node in sorted(member_assets)
        ],
    }
    return f"r7ivc-{_digest(value)}"


def _stable_split(
    component_id: str,
    *,
    config: R7IndexedVisualGraphConfig,
) -> str:
    payload = (
        f"{config.split_version}\0{config.data_seed}\0{component_id}"
    ).encode("utf-8")
    bucket = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
    ) % 1_000_000
    if bucket < 800_000:
        return "train"
    if bucket < 900_000:
        return "validation"
    return "test"


def build_r7_indexed_visual_graph(
    pairs: Sequence[R7IndexedVisualPair],
    *,
    dino_edges: Sequence[R7IndexedDinoEdge] = (),
    anchor_iids: Sequence[str] = (),
    previously_seen_iids: Sequence[str] = (),
    config: R7IndexedVisualGraphConfig | None = None,
) -> R7IndexedVisualGraphResult:
    """Build a deterministic indexed graph and whole-component split.

    The BK-tree is queried only against assets already inserted in canonical
    node order.  For an exact stored dHash value it retains one representative
    asset, which is sufficient because every later carrier is joined to that
    representative.  Thus the resulting partition is identical to the full
    any-frame pair relation without enumerating all asset pairs.
    """

    active_config = config or R7IndexedVisualGraphConfig()
    if not isinstance(active_config, R7IndexedVisualGraphConfig):
        raise TypeError("config must be R7IndexedVisualGraphConfig")
    active_config.validate()
    normalized_pairs = _normalize_pairs(pairs)
    anchors = _normalized_iid_ledger(
        anchor_iids,
        context="anchor_iids",
    )
    seen = _normalized_iid_ledger(
        previously_seen_iids,
        context="previously_seen_iids",
    )

    asset_by_node: dict[_Node, R7IndexedVisualAsset] = {
        (pair.iid, role): getattr(pair, role)
        for pair in normalized_pairs
        for role in ASSET_ROLES
    }
    nodes = tuple(sorted(asset_by_node))
    valid_nodes = set(nodes)
    normalized_dino_edges = _normalize_dino_edges(
        dino_edges,
        valid_nodes=valid_nodes,
    )

    dsu = _DeterministicDsu(nodes)
    spanning_edges: list[R7IndexedRelationEdge] = []
    spanning_counts: dict[str, int] = {
        "paired_sample": 0,
        "exact_sha256": 0,
        "dhash_hamming": 0,
        "dino_cosine": 0,
    }
    candidate_counts: dict[str, int] = {
        "paired_sample": 0,
        "exact_sha256": 0,
        "dhash_hamming": 0,
        "dino_cosine": 0,
    }

    def connect(
        left: _Node,
        right: _Node,
        *,
        relation: str,
        value: int | float | None,
    ) -> bool:
        if not dsu.union(left, right):
            return False
        spanning_edges.append(
            _ordered_relation_edge(
                left,
                right,
                relation=relation,
                value=value,
            )
        )
        spanning_counts[relation] = spanning_counts.get(relation, 0) + 1
        return True

    # Pair atomicity is established before every visual relation.
    for pair in normalized_pairs:
        candidate_counts["paired_sample"] += 1
        connected = connect(
            (pair.iid, "source"),
            (pair.iid, "target"),
            relation="paired_sample",
            value=None,
        )
        if not connected:  # pragma: no cover - defensive invariant
            raise AssertionError("new pair assets were already connected")

    # Exact SHA groups need only a deterministic star, not a clique.
    nodes_by_sha: dict[str, list[_Node]] = {}
    for node in nodes:
        nodes_by_sha.setdefault(
            asset_by_node[node].video_sha256,
            [],
        ).append(node)
    for video_sha256 in sorted(nodes_by_sha):
        members = tuple(sorted(nodes_by_sha[video_sha256]))
        if len(members) < 2:
            continue
        representative = members[0]
        for member in members[1:]:
            candidate_counts["exact_sha256"] += 1
            connect(
                representative,
                member,
                relation="exact_sha256",
                value=None,
            )

    # Incremental exact radius search in 64-bit Hamming space.
    index = _HammingBkTree()
    for node in nodes:
        hashes = tuple(
            int(value, 16) for value in asset_by_node[node].dhashes
        )
        candidate_minimum_distance: dict[_Node, int] = {}
        for value in hashes:
            for candidate, distance in index.query(
                value,
                active_config.maximum_dhash_hamming,
            ):
                previous = candidate_minimum_distance.get(candidate)
                if previous is None or distance < previous:
                    candidate_minimum_distance[candidate] = distance
        for candidate in sorted(candidate_minimum_distance):
            candidate_counts["dhash_hamming"] += 1
            connect(
                candidate,
                node,
                relation="dhash_hamming",
                value=candidate_minimum_distance[candidate],
            )
        for value in sorted(set(hashes)):
            index.insert(value, node)

    dino_above_threshold = 0
    dino_below_threshold = 0
    for edge in normalized_dino_edges:
        if edge.cosine < active_config.minimum_dino_cosine:
            dino_below_threshold += 1
            continue
        dino_above_threshold += 1
        candidate_counts["dino_cosine"] += 1
        connect(
            edge.left_node,
            edge.right_node,
            relation="dino_cosine",
            value=edge.cosine,
        )

    grouped: dict[_Node, list[_Node]] = {}
    for node in nodes:
        grouped.setdefault(dsu.find(node), []).append(node)
    member_groups = tuple(
        sorted(
            (tuple(sorted(members)) for members in grouped.values()),
            key=lambda members: members,
        )
    )

    current_iids = {pair.iid for pair in normalized_pairs}
    matched_anchors = set(anchors) & current_iids
    matched_seen = set(seen) & current_iids
    components: list[R7IndexedVisualComponent] = []
    assignments: list[R7IndexedVisualAssignment] = []
    for member_assets in member_groups:
        member_iids = tuple(
            sorted({iid for iid, _role in member_assets})
        )
        if len(member_assets) != 2 * len(member_iids):
            raise AssertionError(
                "pair atomicity invariant failed inside component"
            )
        component_anchors = tuple(
            sorted(set(member_iids) & matched_anchors)
        )
        component_seen = tuple(
            sorted(set(member_iids) & matched_seen)
        )
        forced_by_anchor = bool(component_anchors)
        forced_by_seen = bool(component_seen)
        forced = forced_by_anchor or forced_by_seen
        identifier = _component_id(member_assets)
        split = (
            "train"
            if forced
            else _stable_split(identifier, config=active_config)
        )
        fresh = not forced
        components.append(
            R7IndexedVisualComponent(
                component_id=identifier,
                member_assets=member_assets,
                member_iids=member_iids,
                split=split,
                fresh=fresh,
                forced_train=forced,
                anchor_iids=component_anchors,
                previously_seen_iids=component_seen,
            )
        )
        assignments.extend(
            R7IndexedVisualAssignment(
                iid=iid,
                component_id=identifier,
                split=split,
                fresh=fresh,
                forced_train=forced,
                forced_by_anchor=forced_by_anchor,
                forced_by_previously_seen=forced_by_seen,
            )
            for iid in member_iids
        )

    ordered_components = tuple(
        sorted(components, key=lambda value: value.component_id)
    )
    ordered_assignments = tuple(
        sorted(assignments, key=lambda value: value.iid)
    )
    ordered_edges = tuple(
        sorted(
            spanning_edges,
            key=lambda value: (
                value.left_node,
                value.right_node,
                value.relation,
                -2.0 if value.value is None else float(value.value),
            ),
        )
    )

    if len(ordered_edges) != len(nodes) - len(ordered_components):
        raise AssertionError(
            "retained relation edges are not a global spanning forest"
        )
    if {assignment.iid for assignment in ordered_assignments} != current_iids:
        raise AssertionError("assignment conservation failed")

    split_iid_counts = tuple(
        (
            split,
            sum(
                assignment.split == split
                for assignment in ordered_assignments
            ),
        )
        for split in sorted(VALID_SPLITS)
    )
    split_component_counts = tuple(
        (
            split,
            sum(
                component.split == split
                for component in ordered_components
            ),
        )
        for split in sorted(VALID_SPLITS)
    )
    largest_iids = max(
        len(component.member_iids) for component in ordered_components
    )
    largest_assets = max(
        len(component.member_assets) for component in ordered_components
    )
    largest_fraction = largest_iids / len(normalized_pairs)
    statistics = R7IndexedVisualGraphStatistics(
        pair_count=len(normalized_pairs),
        asset_count=len(nodes),
        component_count=len(ordered_components),
        relation_counts=tuple(sorted(spanning_counts.items())),
        indexed_candidate_relation_counts=tuple(
            sorted(candidate_counts.items())
        ),
        split_iid_counts=split_iid_counts,
        split_component_counts=split_component_counts,
        forced_component_count=sum(
            component.forced_train for component in ordered_components
        ),
        forced_iid_count=sum(
            assignment.forced_train
            for assignment in ordered_assignments
        ),
        fresh_iid_count=sum(
            assignment.fresh for assignment in ordered_assignments
        ),
        dino_input_edge_count=len(normalized_dino_edges),
        dino_above_threshold_count=dino_above_threshold,
        dino_below_threshold_count=dino_below_threshold,
        dhash_index_unique_values=index.unique_values,
        dhash_index_queries=index.queries,
        dhash_index_distance_evaluations=index.distance_evaluations,
        largest_component_iids=largest_iids,
        largest_component_assets=largest_assets,
        largest_component_fraction=float(
            round(largest_fraction, 12)
        ),
        giant_component_threshold=GIANT_COMPONENT_FRACTION,
        giant_component_warning=(
            largest_fraction > GIANT_COMPONENT_FRACTION
        ),
    )

    pairs_value = [
        {
            "iid": pair.iid,
            "pair_digest": pair.digest(),
            "pair": pair.to_dict(),
        }
        for pair in normalized_pairs
    ]
    dino_value = [
        edge.to_dict() for edge in normalized_dino_edges
    ]
    input_pairs_digest = _digest(pairs_value)
    dino_edges_digest = _digest(dino_value)
    anchor_digest = _digest(list(anchors))
    seen_digest = _digest(list(seen))
    threshold_digest = _digest(active_config.thresholds_dict())
    config_digest = active_config.digest()
    input_digest = _digest(
        {
            "input_pairs_digest": input_pairs_digest,
            "dino_input_edges_digest": dino_edges_digest,
            "anchor_iids_digest": anchor_digest,
            "previously_seen_iids_digest": seen_digest,
        }
    )
    edge_digest = _digest(
        [edge.to_dict() for edge in ordered_edges]
    )
    component_digest = _digest(
        [component.to_dict() for component in ordered_components]
    )
    assignment_digest = _digest(
        [
            assignment.to_dict()
            for assignment in ordered_assignments
        ]
    )
    statistics_digest = _digest(statistics.to_dict())
    provenance_base = {
        "schema_version": R7_INDEXED_VISUAL_GRAPH_SCHEMA,
        "graph_version": R7_INDEXED_VISUAL_GRAPH_VERSION,
        "implementation_version": (
            R7_INDEXED_VISUAL_GRAPH_IMPLEMENTATION
        ),
        "config_digest": config_digest,
        "input_pairs_digest": input_pairs_digest,
        "dino_input_edges_digest": dino_edges_digest,
        "anchor_iids_digest": anchor_digest,
        "previously_seen_iids_digest": seen_digest,
        "input_digest": input_digest,
        "threshold_digest": threshold_digest,
        "edge_digest": edge_digest,
        "component_digest": component_digest,
        "assignment_digest": assignment_digest,
        "statistics_digest": statistics_digest,
    }
    provenance = R7IndexedVisualGraphProvenance(
        **provenance_base,
        provenance_digest=_digest(provenance_base),
    )
    return R7IndexedVisualGraphResult(
        assignments=ordered_assignments,
        components=ordered_components,
        spanning_edges=ordered_edges,
        statistics=statistics,
        provenance=provenance,
    )


def verify_r7_indexed_visual_graph(
    result: R7IndexedVisualGraphResult,
    pairs: Sequence[R7IndexedVisualPair],
    *,
    dino_edges: Sequence[R7IndexedDinoEdge] = (),
    anchor_iids: Sequence[str] = (),
    previously_seen_iids: Sequence[str] = (),
    config: R7IndexedVisualGraphConfig | None = None,
) -> bool:
    """Recompute the full deterministic result and reject any tampering."""

    if not isinstance(result, R7IndexedVisualGraphResult):
        raise TypeError("result must be R7IndexedVisualGraphResult")
    expected = build_r7_indexed_visual_graph(
        pairs,
        dino_edges=dino_edges,
        anchor_iids=anchor_iids,
        previously_seen_iids=previously_seen_iids,
        config=config,
    )
    if result.to_dict() != expected.to_dict():
        raise ValueError(
            "indexed visual graph result differs from deterministic "
            "recomputation"
        )
    return True


def assignments_by_iid(
    result: R7IndexedVisualGraphResult,
) -> Mapping[str, R7IndexedVisualAssignment]:
    if not isinstance(result, R7IndexedVisualGraphResult):
        raise TypeError("result must be R7IndexedVisualGraphResult")
    return {
        assignment.iid: assignment for assignment in result.assignments
    }
