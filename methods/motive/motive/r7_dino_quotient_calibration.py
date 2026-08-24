"""Exact quotient-pair evidence derived during the R7 DINO scan.

This module is deliberately independent of filesystem publication.  It
provides the in-memory, digest-bound core that can be connected to
``r7_expansion_dino_edges.match_rank_arrays`` without performing a second
DINO pass.

The graph contract is intentionally narrow: assets are in canonical
lexicographic IID order and every IID owns two adjacent assets, source then
target.  The existing exact-eight partition assigns an asset ``a`` to
``a % 8``.  For an IID pair ``i < j`` this means that exactly two rank
partials exist:

* source(i) reduced over source(j), target(j); and
* target(i) reduced over source(j), target(j).

The exact-eight merge reduces those two partials and therefore records the
exact maximum over all four asset-role combinations.  All comparisons are
on finite float32 values.  Ties are resolved by the lexicographically
smallest ``(asset_a, asset_b, frame_a, frame_b)`` witness.

The second reduction maps IIDs to *pre-DINO* base components.  It produces
one exact maximum for every unordered base-component pair.  Statistical
calibration then samples these quotient pairs, rather than correlated asset
pairs, with fixed score bins and seeded SHA-256 bottom-k selection.

No output from this module authorizes training or asserts that a threshold
has been calibrated by humans.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from . import r7_artifact_permissions as artifact_permissions


WORLD_SIZE = 8
DINO_FRAMES = 6

RANK_PARTIAL_SCHEMA = "motive-r7-dino-quotient-rank-partial-v2"
IID_PAIR_MAXIMA_SCHEMA = "motive-r7-dino-iid-pair-maxima-v2"
COMPONENT_PAIR_MAXIMA_SCHEMA = (
    "motive-r7-dino-base-component-pair-maxima-v2"
)
QUOTIENT_SAMPLE_SCHEMA = "motive-r7-dino-quotient-calibration-sample-v2"
GRAPH_COMMIT_BINDING_SCHEMA = (
    "motive-r7-validated-graph-input-commit-binding-v1"
)
ARTIFACT_METADATA_SCHEMA = (
    "motive-r7-dino-quotient-array-artifact-metadata-v1"
)
ARTIFACT_DONE_SCHEMA = (
    "motive-r7-dino-quotient-array-artifact-done-v1"
)

ARTIFACT_METADATA_NAME = "metadata.json"
ARTIFACT_ARRAYS_NAME = "arrays.npz"
ARTIFACT_DONE_NAME = "done.json"

GRAPH_ORDER_VERSION = "lexicographic-iid-source-before-target-v1"
GRAPH_DIGEST_VERSION = "complete-canonical-graph-rows-v2"
PARTITION_VERSION = "smaller-asset-index-modulo-exactly-8-v1"
PARTIAL_REDUCTION_VERSION = (
    "asset-a-by-candidate-iid-two-role-float32-max-v2"
)
IID_REDUCTION_VERSION = "iid-pair-four-role-float32-max-v2"
COMPONENT_REDUCTION_VERSION = (
    "pre-dino-base-component-pair-exact-float32-max-v2"
)
WITNESS_TIE_BREAK = (
    "lowest-asset-a-then-asset-b-then-frame-a-then-frame-b"
)

SAMPLE_SEED = 260108830
DEFAULT_SAMPLE_PER_STRATUM = 256
SAMPLE_METHOD = "seeded-sha256-quotient-pair-bottom-k-v1"
SAMPLE_PAIR_ID_VERSION = "pre-dino-base-component-pair-v1"

# These are the fixed audit bins already used by the DINO edge diagnostic.
# Comparisons are deliberately against float32 cut points.
SCORE_BIN_NAMES = (
    "low_complement",
    "mid_complement",
    "near_audit_complement",
    "audit_lower",
    "audit_upper",
    "hard",
)
_SCORE_CUTS = np.asarray(
    [0.80, 0.90, 0.92, 0.94, 0.96],
    dtype=np.float32,
)
_SCORE_BOUNDS = np.asarray(
    [-1.0, 0.80, 0.90, 0.92, 0.94, 0.96, 1.0],
    dtype=np.float32,
)

ENDPOINT_CLASS_NAMES = ("AA", "AC", "CC")
ENDPOINT_CLASS_CODES = {
    "AA": 0,
    "AC": 1,
    "CC": 2,
}

_GRAPH_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "asset_index",
        "iid",
        "role",
        "anchor",
        "cohort",
        "video_sha256",
        "dhashes",
        "source_artifact_digest",
        "source_input_index",
        "source_index_digest",
    }
)
_GRAPH_ARTIFACT_HASH_NAMES = frozenset(
    {"manifest", "archive", "summary", "done"}
)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def make_graph_commit_binding(
    *,
    artifact_digest: str,
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Create the path-free binding for one validated graph-input commit."""

    value = {
        "schema_version": GRAPH_COMMIT_BINDING_SCHEMA,
        "artifact_digest": artifact_digest,
        "artifact_hashes": dict(artifact_hashes),
    }
    return _normalize_graph_commit_binding(value)


def _normalize_graph_commit_binding(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {"schema_version", "artifact_digest", "artifact_hashes"}
        or value.get("schema_version") != GRAPH_COMMIT_BINDING_SCHEMA
        or not _is_sha256(value.get("artifact_digest"))
        or not isinstance(value.get("artifact_hashes"), Mapping)
        or set(value["artifact_hashes"])
        != _GRAPH_ARTIFACT_HASH_NAMES
        or any(
            not _is_sha256(value["artifact_hashes"].get(name))
            for name in _GRAPH_ARTIFACT_HASH_NAMES
        )
    ):
        raise ValueError("validated graph commit binding differs")
    return {
        "schema_version": GRAPH_COMMIT_BINDING_SCHEMA,
        "artifact_digest": value["artifact_digest"],
        "artifact_hashes": {
            name: value["artifact_hashes"][name]
            for name in sorted(_GRAPH_ARTIFACT_HASH_NAMES)
        },
    }


@dataclass(frozen=True)
class _Graph:
    iids: tuple[str, ...]
    anchors: tuple[bool, ...]
    assets: int
    digest: str

    @property
    def n_iids(self) -> int:
        return len(self.iids)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object_digest(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        raise ValueError("artifact array is not C-contiguous")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _array_descriptors(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": np.asarray(arrays[name]).dtype.name,
            "shape": list(np.asarray(arrays[name]).shape),
            "sha256": _array_digest(np.asarray(arrays[name])),
        }
        for name in sorted(arrays)
    }


def _make_artifact(
    *,
    schema: str,
    contract: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    owned_arrays = {
        name: np.ascontiguousarray(value).copy()
        for name, value in arrays.items()
    }
    descriptors = _array_descriptors(owned_arrays)
    digest = _object_digest(
        {
            "schema_version": schema,
            "contract": dict(contract),
            "array_descriptors": descriptors,
        }
    )
    return {
        "schema_version": schema,
        "contract": dict(contract),
        "array_descriptors": descriptors,
        "arrays": owned_arrays,
        "artifact_digest": digest,
    }


def _validate_artifact_envelope(
    artifact: Mapping[str, Any],
    *,
    schema: str,
    array_dtypes: Mapping[str, np.dtype[Any] | str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if (
        not isinstance(artifact, Mapping)
        or set(artifact)
        != {
            "schema_version",
            "contract",
            "array_descriptors",
            "arrays",
            "artifact_digest",
        }
        or artifact.get("schema_version") != schema
        or not isinstance(artifact.get("contract"), Mapping)
        or not isinstance(artifact.get("array_descriptors"), Mapping)
        or not isinstance(artifact.get("arrays"), Mapping)
        or type(artifact.get("artifact_digest")) is not str
    ):
        raise ValueError(f"{schema} artifact envelope differs")
    arrays_raw = artifact["arrays"]
    if set(arrays_raw) != set(array_dtypes):
        raise ValueError(f"{schema} artifact array names differ")
    arrays: dict[str, np.ndarray] = {}
    for name, expected_dtype in array_dtypes.items():
        value = arrays_raw[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype(expected_dtype)
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{schema} array {name} dtype/layout differs"
            )
        arrays[name] = value
    descriptors = _array_descriptors(arrays)
    if dict(artifact["array_descriptors"]) != descriptors:
        raise ValueError(f"{schema} artifact array digest differs")
    expected_digest = _object_digest(
        {
            "schema_version": schema,
            "contract": dict(artifact["contract"]),
            "array_descriptors": descriptors,
        }
    )
    if artifact["artifact_digest"] != expected_digest:
        raise ValueError(f"{schema} artifact digest differs")
    return dict(artifact["contract"]), arrays


def _canonical_graph(rows: Sequence[Mapping[str, Any]]) -> _Graph:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or len(rows) < 4
        or len(rows) % 2
    ):
        raise ValueError(
            "canonical graph must contain at least two paired IIDs"
        )
    iids: list[str] = []
    anchors: list[bool] = []
    canonical_rows: list[dict[str, Any]] = []
    for offset in range(0, len(rows), 2):
        source = rows[offset]
        target = rows[offset + 1]
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            raise ValueError("canonical graph rows must be mappings")
        iid = source.get("iid")
        source_index = source.get("asset_index")
        target_index = target.get("asset_index")
        source_anchor = source.get("anchor")
        target_anchor = target.get("anchor")
        source_cohort = source.get("cohort")
        source_artifact = source.get("source_artifact_digest")
        source_input_index = source.get("source_input_index")
        if (
            set(source) != _GRAPH_ROW_FIELDS
            or set(target) != _GRAPH_ROW_FIELDS
            or type(source_index) is not int
            or source_index != offset
            or type(target_index) is not int
            or target_index != offset + 1
            or type(source.get("schema_version")) is not str
            or not source["schema_version"]
            or target.get("schema_version")
            != source["schema_version"]
            or type(iid) is not str
            or not iid
            or iid.strip() != iid
            or "\x00" in iid
            or source.get("role") != "source"
            or target.get("role") != "target"
            or target.get("iid") != iid
            or type(source_anchor) is not bool
            or target_anchor is not source_anchor
            or type(source_cohort) is not str
            or not source_cohort
            or target.get("cohort") != source_cohort
            or source_anchor != source_cohort.startswith("anchor_")
            or not _is_sha256(source.get("video_sha256"))
            or not _is_sha256(target.get("video_sha256"))
            or not _is_sha256(source_artifact)
            or target.get("source_artifact_digest") != source_artifact
            or type(source_input_index) is not int
            or source_input_index < 0
            or target.get("source_input_index") != source_input_index
            or not _is_sha256(source.get("source_index_digest"))
            or not _is_sha256(target.get("source_index_digest"))
            or any(
                not isinstance(row.get("dhashes"), list)
                or len(row["dhashes"]) != DINO_FRAMES
                or any(
                    type(value) is not str
                    or len(value) != 16
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                    for value in row["dhashes"]
                )
                for row in (source, target)
            )
        ):
            raise ValueError(
                f"canonical source/target binding differs at {offset}"
            )
        iids.append(iid)
        anchors.append(source_anchor)
        canonical_rows.extend([dict(source), dict(target)])
    if len(set(iids)) != len(iids) or iids != sorted(iids):
        raise ValueError(
            "IIDs must be unique and in canonical lexicographic order"
        )
    return _Graph(
        iids=tuple(iids),
        anchors=tuple(anchors),
        assets=len(rows),
        digest=_object_digest(
            {
                "graph_digest_version": GRAPH_DIGEST_VERSION,
                "asset_order": GRAPH_ORDER_VERSION,
                "rows": canonical_rows,
            }
        ),
    )


def _validate_rank(rank: int, world_size: int) -> None:
    if (
        type(rank) is not int
        or type(world_size) is not int
        or world_size != WORLD_SIZE
        or not 0 <= rank < WORLD_SIZE
    ):
        raise ValueError("rank partition must be exactly ranks 0..7")


def _pair_count(items: int) -> int:
    if type(items) is not int or items < 0:
        raise ValueError("pair-count input differs")
    return items * (items - 1) // 2


def _pair_ordinal(
    left: np.ndarray,
    right: np.ndarray,
    items: int,
) -> np.ndarray:
    a = np.asarray(left, dtype=np.int64)
    b = np.asarray(right, dtype=np.int64)
    if (
        a.shape != b.shape
        or np.any(a < 0)
        or np.any(b <= a)
        or np.any(b >= items)
    ):
        raise ValueError("unordered pair indices differ")
    return (
        a * (2 * items - a - 1) // 2 + (b - a - 1)
    ).astype(np.int64, copy=False)


def _enumerated_pairs(items: int) -> tuple[np.ndarray, np.ndarray]:
    count = _pair_count(items)
    left = np.empty(count, dtype=np.int32)
    right = np.empty(count, dtype=np.int32)
    cursor = 0
    for item in range(items - 1):
        width = items - item - 1
        left[cursor : cursor + width] = item
        right[cursor : cursor + width] = np.arange(
            item + 1,
            items,
            dtype=np.int32,
        )
        cursor += width
    if cursor != count:
        raise RuntimeError("internal pair enumeration differs")
    return left, right


def _expected_rank_layout(
    graph: _Graph,
    rank: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, int],
]:
    owned = np.arange(rank, graph.assets, WORLD_SIZE, dtype=np.int32)
    partials = sum(
        graph.n_iids - int(asset) // 2 - 1
        for asset in owned.tolist()
    )
    iid_a = np.empty(partials, dtype=np.int32)
    iid_b = np.empty(partials, dtype=np.int32)
    asset_a = np.empty(partials, dtype=np.int32)
    offsets: dict[int, int] = {}
    cursor = 0
    for raw_asset in owned.tolist():
        asset = int(raw_asset)
        left_iid = asset // 2
        width = graph.n_iids - left_iid - 1
        offsets[asset] = cursor
        if width:
            iid_a[cursor : cursor + width] = left_iid
            iid_b[cursor : cursor + width] = np.arange(
                left_iid + 1,
                graph.n_iids,
                dtype=np.int32,
            )
            asset_a[cursor : cursor + width] = asset
        cursor += width
    if cursor != partials:
        raise RuntimeError("internal rank layout differs")
    return iid_a, iid_b, asset_a, offsets


def _owned_asset_digest(owned: np.ndarray) -> str:
    return _object_digest([int(value) for value in owned.tolist()])


class RankQuotientAccumulator:
    """Accumulate quotient partials alongside one exact-eight DINO rank.

    ``consume_block`` must be called immediately after the existing block
    matcher returns.  Candidate blocks for each owned asset must exactly
    cover ``asset_a + 1 .. assets - 1`` in order.  This streaming condition
    detects gaps, duplicate blocks, and silently truncated rank scans.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        graph_binding: Mapping[str, Any],
        rank: int,
        world_size: int = WORLD_SIZE,
    ) -> None:
        _validate_rank(rank, world_size)
        self._graph = _canonical_graph(rows)
        self._graph_binding = _normalize_graph_commit_binding(
            graph_binding
        )
        self.rank = rank
        self.world_size = world_size
        (
            self._iid_a,
            self._iid_b,
            self._asset_a,
            self._offsets,
        ) = _expected_rank_layout(self._graph, rank)
        self._owned = np.arange(
            rank,
            self._graph.assets,
            WORLD_SIZE,
            dtype=np.int32,
        )
        self._next = {
            int(asset): int(asset) + 1
            for asset in self._owned.tolist()
        }
        count = len(self._iid_a)
        self._score = np.full(count, -np.inf, dtype=np.float32)
        self._asset_b = np.full(count, -1, dtype=np.int32)
        self._frame_a = np.full(count, 255, dtype=np.uint8)
        self._frame_b = np.full(count, 255, dtype=np.uint8)
        self._candidate_roles_seen = np.zeros(count, dtype=np.uint8)
        self._observed_asset_pairs = 0
        self._observed_cross_asset_pairs = 0
        self._finalized = False

    def consume_block(
        self,
        *,
        asset_a: int,
        candidate_indices: np.ndarray,
        scores: np.ndarray,
        frames_a: np.ndarray,
        frames_b: np.ndarray,
    ) -> None:
        if self._finalized:
            raise RuntimeError("rank quotient accumulator is finalized")
        if type(asset_a) is not int or asset_a not in self._next:
            raise ValueError("asset_a is not owned by this rank")
        values = np.asarray(scores)
        candidates = np.asarray(candidate_indices)
        fa = np.asarray(frames_a)
        fb = np.asarray(frames_b)
        count = len(candidates) if candidates.ndim == 1 else -1
        if (
            candidates.dtype != np.dtype("int64")
            or candidates.ndim != 1
            or count < 1
            or values.dtype != np.dtype("float32")
            or values.shape != (count,)
            or fa.dtype != np.dtype("int64")
            or fa.shape != (count,)
            or fb.dtype != np.dtype("int64")
            or fb.shape != (count,)
            or not candidates.flags.c_contiguous
            or not values.flags.c_contiguous
            or not fa.flags.c_contiguous
            or not fb.flags.c_contiguous
            or not np.isfinite(values).all()
            or np.any(values < np.float32(-1.0))
            or np.any(values > np.float32(1.0))
            or np.any(fa < 0)
            or np.any(fa >= DINO_FRAMES)
            or np.any(fb < 0)
            or np.any(fb >= DINO_FRAMES)
        ):
            raise ValueError("DINO quotient block dtype/shape/value differs")
        expected_begin = self._next[asset_a]
        expected = np.arange(
            expected_begin,
            expected_begin + count,
            dtype=np.int64,
        )
        if (
            not np.array_equal(candidates, expected)
            or int(candidates[-1]) >= self._graph.assets
        ):
            raise ValueError(
                "candidate block is not the next exact contiguous range"
            )
        left_iid = asset_a // 2
        offset_base = self._offsets[asset_a]
        self._observed_asset_pairs += count
        for position in range(count):
            asset_b = int(candidates[position])
            right_iid = asset_b // 2
            if right_iid == left_iid:
                continue
            self._observed_cross_asset_pairs += 1
            slot = offset_base + right_iid - left_iid - 1
            role_bit = np.uint8(1 << (asset_b % 2))
            if self._candidate_roles_seen[slot] & role_bit:
                raise RuntimeError("candidate role was observed twice")
            self._candidate_roles_seen[slot] |= role_bit
            score = np.float32(values[position])
            candidate_key = (
                asset_b,
                int(fa[position]),
                int(fb[position]),
            )
            current_key = (
                int(self._asset_b[slot]),
                int(self._frame_a[slot]),
                int(self._frame_b[slot]),
            )
            if (
                score > self._score[slot]
                or (
                    score == self._score[slot]
                    and candidate_key < current_key
                )
            ):
                self._score[slot] = score
                self._asset_b[slot] = asset_b
                self._frame_a[slot] = int(fa[position])
                self._frame_b[slot] = int(fb[position])
        self._next[asset_a] = expected_begin + count

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("rank quotient accumulator is finalized")
        incomplete = [
            asset
            for asset, cursor in self._next.items()
            if cursor != self._graph.assets
        ]
        if incomplete:
            raise RuntimeError(
                "rank quotient coverage is incomplete for assets "
                + ",".join(str(value) for value in incomplete[:8])
            )
        expected_asset_pairs = sum(
            self._graph.assets - int(asset) - 1
            for asset in self._owned.tolist()
        )
        expected_cross_asset_pairs = 2 * len(self._iid_a)
        if (
            self._observed_asset_pairs != expected_asset_pairs
            or self._observed_cross_asset_pairs
            != expected_cross_asset_pairs
            or np.any(self._candidate_roles_seen != np.uint8(3))
            or not np.isfinite(self._score).all()
            or np.any(self._asset_b < 0)
        ):
            raise RuntimeError("rank quotient exact coverage proof differs")
        arrays = {
            "iid_a": self._iid_a,
            "iid_b": self._iid_b,
            "asset_a": self._asset_a,
            "asset_b": self._asset_b,
            "score": self._score,
            "frame_a": self._frame_a,
            "frame_b": self._frame_b,
        }
        contract = {
            "schema_version": RANK_PARTIAL_SCHEMA,
            "graph_digest": self._graph.digest,
            "graph_digest_version": GRAPH_DIGEST_VERSION,
            "graph_order": GRAPH_ORDER_VERSION,
            "graph_commit_binding": self._graph_binding,
            "graph_commit_binding_sha256": _object_digest(
                self._graph_binding
            ),
            "assets": self._graph.assets,
            "iids": self._graph.n_iids,
            "iid_identifiers_sha256": _object_digest(
                list(self._graph.iids)
            ),
            "rank": self.rank,
            "world_size": self.world_size,
            "partition": PARTITION_VERSION,
            "partial_reduction": PARTIAL_REDUCTION_VERSION,
            "compute_dtype": "float32",
            "witness_tie_break": WITNESS_TIE_BREAK,
            "owned_asset_indices_sha256": _owned_asset_digest(
                self._owned
            ),
            "expected_asset_pairs": expected_asset_pairs,
            "observed_asset_pairs": self._observed_asset_pairs,
            "expected_cross_asset_pairs": expected_cross_asset_pairs,
            "observed_cross_asset_pairs":
                self._observed_cross_asset_pairs,
            "expected_partials": len(self._iid_a),
            "observed_partials": len(self._iid_a),
            "candidate_roles_per_partial": 2,
            "candidate_role_coverage_complete": True,
            "coverage_complete": True,
            "human_labels_asserted": False,
            "training_authorized": False,
        }
        artifact = _make_artifact(
            schema=RANK_PARTIAL_SCHEMA,
            contract=contract,
            arrays=arrays,
        )
        validate_rank_partial_artifact(
            self._graph,
            artifact,
            graph_binding=self._graph_binding,
            expected_rank=self.rank,
        )
        self._finalized = True
        return artifact


_RANK_ARRAY_DTYPES = {
    "iid_a": "int32",
    "iid_b": "int32",
    "asset_a": "int32",
    "asset_b": "int32",
    "score": "float32",
    "frame_a": "uint8",
    "frame_b": "uint8",
}


def _coerce_graph(
    rows_or_graph: Sequence[Mapping[str, Any]] | _Graph,
) -> _Graph:
    if isinstance(rows_or_graph, _Graph):
        return rows_or_graph
    return _canonical_graph(rows_or_graph)


def validate_rank_partial_artifact(
    rows: Sequence[Mapping[str, Any]] | _Graph,
    artifact: Mapping[str, Any],
    *,
    graph_binding: Mapping[str, Any],
    expected_rank: int | None = None,
) -> dict[str, Any]:
    graph = _coerce_graph(rows)
    binding = _normalize_graph_commit_binding(graph_binding)
    contract, arrays = _validate_artifact_envelope(
        artifact,
        schema=RANK_PARTIAL_SCHEMA,
        array_dtypes=_RANK_ARRAY_DTYPES,
    )
    rank = contract.get("rank")
    if type(rank) is not int:
        raise ValueError("rank partial rank differs")
    _validate_rank(rank, contract.get("world_size"))
    if expected_rank is not None and rank != expected_rank:
        raise ValueError("rank partial is from an unexpected rank")
    iid_a, iid_b, asset_a, unused_offsets = _expected_rank_layout(
        graph,
        rank,
    )
    del unused_offsets
    owned = np.arange(rank, graph.assets, WORLD_SIZE, dtype=np.int32)
    expected_asset_pairs = sum(
        graph.assets - int(asset) - 1
        for asset in owned.tolist()
    )
    expected_cross = 2 * len(iid_a)
    expected_contract = {
        "schema_version": RANK_PARTIAL_SCHEMA,
        "graph_digest": graph.digest,
        "graph_digest_version": GRAPH_DIGEST_VERSION,
        "graph_order": GRAPH_ORDER_VERSION,
        "graph_commit_binding": binding,
        "graph_commit_binding_sha256": _object_digest(binding),
        "assets": graph.assets,
        "iids": graph.n_iids,
        "iid_identifiers_sha256": _object_digest(list(graph.iids)),
        "rank": rank,
        "world_size": WORLD_SIZE,
        "partition": PARTITION_VERSION,
        "partial_reduction": PARTIAL_REDUCTION_VERSION,
        "compute_dtype": "float32",
        "witness_tie_break": WITNESS_TIE_BREAK,
        "owned_asset_indices_sha256": _owned_asset_digest(owned),
        "expected_asset_pairs": expected_asset_pairs,
        "observed_asset_pairs": expected_asset_pairs,
        "expected_cross_asset_pairs": expected_cross,
        "observed_cross_asset_pairs": expected_cross,
        "expected_partials": len(iid_a),
        "observed_partials": len(iid_a),
        "candidate_roles_per_partial": 2,
        "candidate_role_coverage_complete": True,
        "coverage_complete": True,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    if contract != expected_contract:
        raise ValueError("rank partial contract differs")
    count = len(iid_a)
    if any(value.shape != (count,) for value in arrays.values()):
        raise ValueError("rank partial array shape differs")
    if (
        not np.array_equal(arrays["iid_a"], iid_a)
        or not np.array_equal(arrays["iid_b"], iid_b)
        or not np.array_equal(arrays["asset_a"], asset_a)
        or not np.isfinite(arrays["score"]).all()
        or np.any(arrays["score"] < np.float32(-1.0))
        or np.any(arrays["score"] > np.float32(1.0))
        or np.any(arrays["frame_a"] >= DINO_FRAMES)
        or np.any(arrays["frame_b"] >= DINO_FRAMES)
        or np.any(
            arrays["asset_b"].astype(np.int64) // 2
            != arrays["iid_b"].astype(np.int64)
        )
    ):
        raise ValueError("rank partial semantic arrays differ")
    return dict(artifact)


_IID_ARRAY_DTYPES = {
    "iid_a": "int32",
    "iid_b": "int32",
    "asset_a": "int32",
    "asset_b": "int32",
    "score": "float32",
    "frame_a": "uint8",
    "frame_b": "uint8",
}


def _witness_key(
    *,
    asset_a: np.ndarray,
    asset_b: np.ndarray,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    assets: int,
) -> np.ndarray:
    return (
        (
            (
                asset_a.astype(np.int64) * assets
                + asset_b.astype(np.int64)
            )
            * DINO_FRAMES
            + frame_a.astype(np.int64)
        )
        * DINO_FRAMES
        + frame_b.astype(np.int64)
    )


def merge_exact8_rank_partials(
    rows: Sequence[Mapping[str, Any]],
    partials: Sequence[Mapping[str, Any]],
    *,
    graph_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge exactly eight complete rank partials into all IID-pair maxima."""

    graph = _canonical_graph(rows)
    binding = _normalize_graph_commit_binding(graph_binding)
    if (
        not isinstance(partials, Sequence)
        or isinstance(partials, (str, bytes))
        or len(partials) != WORLD_SIZE
    ):
        raise ValueError("exact quotient merge requires eight partials")
    by_rank: dict[int, Mapping[str, Any]] = {}
    for artifact in partials:
        contract = artifact.get("contract") if isinstance(
            artifact, Mapping
        ) else None
        rank = contract.get("rank") if isinstance(contract, Mapping) else None
        if type(rank) is not int or rank in by_rank:
            raise ValueError("rank partial set has duplicate/invalid ranks")
        validate_rank_partial_artifact(
            graph,
            artifact,
            graph_binding=binding,
            expected_rank=rank,
        )
        by_rank[rank] = artifact
    if set(by_rank) != set(range(WORLD_SIZE)):
        raise ValueError("rank partial set is not exactly ranks 0..7")
    rank_bindings = {
        _canonical_json(
            by_rank[rank]["contract"]["graph_commit_binding"]
        )
        for rank in range(WORLD_SIZE)
    }
    if rank_bindings != {_canonical_json(binding)}:
        raise ValueError(
            "exact-eight rank graph commit bindings are not identical"
        )

    pair_count = _pair_count(graph.n_iids)
    output_score = np.full(pair_count, -np.inf, dtype=np.float32)
    output_asset_a = np.full(pair_count, -1, dtype=np.int32)
    output_asset_b = np.full(pair_count, -1, dtype=np.int32)
    output_frame_a = np.full(pair_count, 255, dtype=np.uint8)
    output_frame_b = np.full(pair_count, 255, dtype=np.uint8)
    roles_seen = np.zeros(pair_count, dtype=np.uint8)
    observed_partials = 0
    for rank in range(WORLD_SIZE):
        arrays = by_rank[rank]["arrays"]
        left = arrays["iid_a"]
        right = arrays["iid_b"]
        ordinal = _pair_ordinal(left, right, graph.n_iids)
        role = (
            arrays["asset_a"].astype(np.int64)
            - 2 * left.astype(np.int64)
        )
        if (
            np.any(role < 0)
            or np.any(role > 1)
            or len(np.unique(ordinal)) != len(ordinal)
        ):
            raise ValueError("rank partial IID/role ownership differs")
        bits = np.left_shift(
            np.uint8(1),
            role.astype(np.uint8),
        )
        if np.any(np.bitwise_and(roles_seen[ordinal], bits)):
            raise ValueError("IID pair has duplicate source/target partial")
        old_key = _witness_key(
            asset_a=output_asset_a[ordinal],
            asset_b=output_asset_b[ordinal],
            frame_a=output_frame_a[ordinal],
            frame_b=output_frame_b[ordinal],
            assets=graph.assets,
        )
        new_key = _witness_key(
            asset_a=arrays["asset_a"],
            asset_b=arrays["asset_b"],
            frame_a=arrays["frame_a"],
            frame_b=arrays["frame_b"],
            assets=graph.assets,
        )
        better = arrays["score"] > output_score[ordinal]
        tied_better = np.logical_and(
            arrays["score"] == output_score[ordinal],
            new_key < old_key,
        )
        replace = np.logical_or(better, tied_better)
        selected = ordinal[replace]
        output_score[selected] = arrays["score"][replace]
        output_asset_a[selected] = arrays["asset_a"][replace]
        output_asset_b[selected] = arrays["asset_b"][replace]
        output_frame_a[selected] = arrays["frame_a"][replace]
        output_frame_b[selected] = arrays["frame_b"][replace]
        roles_seen[ordinal] = np.bitwise_or(roles_seen[ordinal], bits)
        observed_partials += len(ordinal)
    if (
        observed_partials != 2 * pair_count
        or np.any(roles_seen != np.uint8(3))
        or not np.isfinite(output_score).all()
    ):
        raise RuntimeError("exact-eight IID quotient coverage differs")
    iid_a, iid_b = _enumerated_pairs(graph.n_iids)
    rank_digests = [
        {
            "rank": rank,
            "artifact_digest": by_rank[rank]["artifact_digest"],
        }
        for rank in range(WORLD_SIZE)
    ]
    contract = {
        "schema_version": IID_PAIR_MAXIMA_SCHEMA,
        "graph_digest": graph.digest,
        "graph_digest_version": GRAPH_DIGEST_VERSION,
        "graph_order": GRAPH_ORDER_VERSION,
        "graph_commit_binding": binding,
        "graph_commit_binding_sha256": _object_digest(binding),
        "assets": graph.assets,
        "iids": graph.n_iids,
        "iid_identifiers": list(graph.iids),
        "iid_anchor_flags": list(graph.anchors),
        "iid_identifiers_sha256": _object_digest(list(graph.iids)),
        "partition": PARTITION_VERSION,
        "world_size": WORLD_SIZE,
        "required_ranks": list(range(WORLD_SIZE)),
        "rank_partial_artifacts": rank_digests,
        "rank_partial_artifacts_sha256": _object_digest(rank_digests),
        "partial_reduction": PARTIAL_REDUCTION_VERSION,
        "iid_reduction": IID_REDUCTION_VERSION,
        "compute_dtype": "float32",
        "witness_tie_break": WITNESS_TIE_BREAK,
        "expected_iid_pairs": pair_count,
        "observed_iid_pairs": pair_count,
        "partials_per_iid_pair": 2,
        "expected_partials": 2 * pair_count,
        "observed_partials": observed_partials,
        "source_target_partial_coverage_complete": True,
        "coverage_complete": True,
        "thresholds_human_calibrated": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    artifact = _make_artifact(
        schema=IID_PAIR_MAXIMA_SCHEMA,
        contract=contract,
        arrays={
            "iid_a": iid_a,
            "iid_b": iid_b,
            "asset_a": output_asset_a,
            "asset_b": output_asset_b,
            "score": output_score,
            "frame_a": output_frame_a,
            "frame_b": output_frame_b,
        },
    )
    validate_iid_pair_maxima(
        rows,
        artifact,
        graph_binding=binding,
    )
    return artifact


def validate_iid_pair_maxima(
    rows: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    *,
    graph_binding: Mapping[str, Any],
) -> dict[str, Any]:
    graph = _canonical_graph(rows)
    binding = _normalize_graph_commit_binding(graph_binding)
    contract, arrays = _validate_artifact_envelope(
        artifact,
        schema=IID_PAIR_MAXIMA_SCHEMA,
        array_dtypes=_IID_ARRAY_DTYPES,
    )
    count = _pair_count(graph.n_iids)
    iid_a, iid_b = _enumerated_pairs(graph.n_iids)
    required_contract_values = {
        "schema_version": IID_PAIR_MAXIMA_SCHEMA,
        "graph_digest": graph.digest,
        "graph_digest_version": GRAPH_DIGEST_VERSION,
        "graph_order": GRAPH_ORDER_VERSION,
        "graph_commit_binding": binding,
        "graph_commit_binding_sha256": _object_digest(binding),
        "assets": graph.assets,
        "iids": graph.n_iids,
        "iid_identifiers": list(graph.iids),
        "iid_anchor_flags": list(graph.anchors),
        "iid_identifiers_sha256": _object_digest(list(graph.iids)),
        "partition": PARTITION_VERSION,
        "world_size": WORLD_SIZE,
        "required_ranks": list(range(WORLD_SIZE)),
        "partial_reduction": PARTIAL_REDUCTION_VERSION,
        "iid_reduction": IID_REDUCTION_VERSION,
        "compute_dtype": "float32",
        "witness_tie_break": WITNESS_TIE_BREAK,
        "expected_iid_pairs": count,
        "observed_iid_pairs": count,
        "partials_per_iid_pair": 2,
        "expected_partials": 2 * count,
        "observed_partials": 2 * count,
        "source_target_partial_coverage_complete": True,
        "coverage_complete": True,
        "thresholds_human_calibrated": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    if (
        set(contract)
        != set(required_contract_values)
        | {
            "rank_partial_artifacts",
            "rank_partial_artifacts_sha256",
        }
    ):
        raise ValueError("IID maxima contract fields differ")
    for key, expected in required_contract_values.items():
        if contract.get(key) != expected:
            raise ValueError(f"IID maxima contract {key} differs")
    rank_digests = contract.get("rank_partial_artifacts")
    if (
        not isinstance(rank_digests, list)
        or len(rank_digests) != WORLD_SIZE
        or [
            item.get("rank")
            for item in rank_digests
            if isinstance(item, Mapping)
        ]
        != list(range(WORLD_SIZE))
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"rank", "artifact_digest"}
            or not _is_sha256(item.get("artifact_digest"))
            for item in rank_digests
        )
        or contract.get("rank_partial_artifacts_sha256")
        != _object_digest(rank_digests)
    ):
        raise ValueError("IID maxima rank provenance differs")
    if any(value.shape != (count,) for value in arrays.values()):
        raise ValueError("IID maxima array shape differs")
    if (
        not np.array_equal(arrays["iid_a"], iid_a)
        or not np.array_equal(arrays["iid_b"], iid_b)
        or not np.isfinite(arrays["score"]).all()
        or np.any(arrays["score"] < np.float32(-1.0))
        or np.any(arrays["score"] > np.float32(1.0))
        or np.any(arrays["frame_a"] >= DINO_FRAMES)
        or np.any(arrays["frame_b"] >= DINO_FRAMES)
        or np.any(
            arrays["asset_a"].astype(np.int64) // 2
            != iid_a.astype(np.int64)
        )
        or np.any(
            arrays["asset_b"].astype(np.int64) // 2
            != iid_b.astype(np.int64)
        )
    ):
        raise ValueError("IID maxima semantic arrays differ")
    return dict(artifact)


_COMPONENT_ARRAY_DTYPES = {
    "component_a": "int32",
    "component_b": "int32",
    "endpoint_class": "uint8",
    "score": "float32",
    "witness_iid_a": "int32",
    "witness_iid_b": "int32",
    "asset_a": "int32",
    "asset_b": "int32",
    "frame_a": "uint8",
    "frame_b": "uint8",
}


def _component_inputs(
    iid_artifact: Mapping[str, Any],
    iid_to_base_component: Mapping[str, str],
    iid_anchor_flags: Mapping[str, bool],
) -> tuple[
    list[str],
    np.ndarray,
    list[str],
    np.ndarray,
    list[bool],
]:
    contract = iid_artifact["contract"]
    iids = list(contract["iid_identifiers"])
    if (
        not isinstance(iid_to_base_component, Mapping)
        or set(iid_to_base_component) != set(iids)
        or not isinstance(iid_anchor_flags, Mapping)
        or set(iid_anchor_flags) != set(iids)
    ):
        raise ValueError("IID component/anchor mappings are not exhaustive")
    components_by_iid: list[str] = []
    anchors_by_iid: list[bool] = []
    for index, iid in enumerate(iids):
        component = iid_to_base_component[iid]
        anchor = iid_anchor_flags[iid]
        if (
            type(component) is not str
            or not component
            or component.strip() != component
            or "\x00" in component
            or type(anchor) is not bool
            or anchor is not contract["iid_anchor_flags"][index]
        ):
            raise ValueError(f"IID component/anchor value differs: {iid}")
        components_by_iid.append(component)
        anchors_by_iid.append(anchor)
    component_ids = sorted(set(components_by_iid))
    if len(component_ids) < 2:
        raise ValueError("at least two base components are required")
    component_index = {
        component: index for index, component in enumerate(component_ids)
    }
    iid_component_indices = np.asarray(
        [component_index[value] for value in components_by_iid],
        dtype=np.int32,
    )
    component_anchor = np.zeros(len(component_ids), dtype=np.bool_)
    for index, anchor in zip(
        iid_component_indices.tolist(),
        anchors_by_iid,
        strict=True,
    ):
        component_anchor[int(index)] |= anchor
    return (
        iids,
        iid_component_indices,
        component_ids,
        component_anchor,
        anchors_by_iid,
    )


def aggregate_base_component_pairs(
    iid_pair_maxima: Mapping[str, Any],
    *,
    iid_to_base_component: Mapping[str, str],
    iid_anchor_flags: Mapping[str, bool],
) -> dict[str, Any]:
    """Reduce complete IID-pair maxima to exact base-component maxima."""

    if iid_pair_maxima.get("schema_version") != IID_PAIR_MAXIMA_SCHEMA:
        raise ValueError("input is not an IID-pair maxima artifact")
    # Envelope validation without graph rows is still possible here because
    # the graph-bound contract and typed arrays are self-contained.
    iid_contract, iid_arrays = _validate_artifact_envelope(
        iid_pair_maxima,
        schema=IID_PAIR_MAXIMA_SCHEMA,
        array_dtypes=_IID_ARRAY_DTYPES,
    )
    if (
        iid_contract.get("coverage_complete") is not True
        or iid_contract.get("source_target_partial_coverage_complete")
        is not True
        or iid_contract.get("training_authorized") is not False
        or iid_contract.get("graph_digest_version")
        != GRAPH_DIGEST_VERSION
    ):
        raise ValueError("IID-pair maxima coverage contract differs")
    graph_binding = _normalize_graph_commit_binding(
        iid_contract.get("graph_commit_binding")
    )
    if iid_contract.get("graph_commit_binding_sha256") != _object_digest(
        graph_binding
    ):
        raise ValueError("IID-pair graph commit binding digest differs")
    (
        iids,
        iid_component,
        component_ids,
        component_anchor,
        anchors_by_iid,
    ) = _component_inputs(
        iid_pair_maxima,
        iid_to_base_component,
        iid_anchor_flags,
    )
    components = len(component_ids)
    count = _pair_count(components)
    iid_left = iid_arrays["iid_a"].astype(np.int64)
    iid_right = iid_arrays["iid_b"].astype(np.int64)
    component_left = iid_component[iid_left].astype(np.int64)
    component_right = iid_component[iid_right].astype(np.int64)
    cross = component_left != component_right
    low = np.minimum(component_left[cross], component_right[cross])
    high = np.maximum(component_left[cross], component_right[cross])
    ordinals = _pair_ordinal(low, high, components)
    scores = iid_arrays["score"][cross]
    max_scores = np.full(count, -np.inf, dtype=np.float32)
    np.maximum.at(max_scores, ordinals, scores)
    if not np.isfinite(max_scores).all():
        raise RuntimeError("a base-component pair has no IID-pair witness")
    candidate_indices = np.flatnonzero(cross)
    maximal = scores == max_scores[ordinals]
    maximal_input = candidate_indices[maximal]
    maximal_ordinals = ordinals[maximal]
    keys = _witness_key(
        asset_a=iid_arrays["asset_a"][maximal_input],
        asset_b=iid_arrays["asset_b"][maximal_input],
        frame_a=iid_arrays["frame_a"][maximal_input],
        frame_b=iid_arrays["frame_b"][maximal_input],
        assets=2 * len(iids),
    )
    best_keys = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(best_keys, maximal_ordinals, keys)
    selected = keys == best_keys[maximal_ordinals]
    selected_input = maximal_input[selected]
    selected_ordinals = maximal_ordinals[selected]
    if (
        len(selected_ordinals) != count
        or len(np.unique(selected_ordinals)) != count
    ):
        raise RuntimeError("component maximum witness tie-break differs")
    component_a, component_b = _enumerated_pairs(components)
    endpoint_class = np.empty(count, dtype=np.uint8)
    left_anchor = component_anchor[component_a]
    right_anchor = component_anchor[component_b]
    endpoint_class[np.logical_and(left_anchor, right_anchor)] = (
        ENDPOINT_CLASS_CODES["AA"]
    )
    endpoint_class[np.logical_xor(left_anchor, right_anchor)] = (
        ENDPOINT_CLASS_CODES["AC"]
    )
    endpoint_class[np.logical_not(
        np.logical_or(left_anchor, right_anchor)
    )] = ENDPOINT_CLASS_CODES["CC"]
    output: dict[str, np.ndarray] = {
        "component_a": component_a,
        "component_b": component_b,
        "endpoint_class": endpoint_class,
        "score": max_scores,
        "witness_iid_a": np.empty(count, dtype=np.int32),
        "witness_iid_b": np.empty(count, dtype=np.int32),
        "asset_a": np.empty(count, dtype=np.int32),
        "asset_b": np.empty(count, dtype=np.int32),
        "frame_a": np.empty(count, dtype=np.uint8),
        "frame_b": np.empty(count, dtype=np.uint8),
    }
    for target, source in (
        ("witness_iid_a", "iid_a"),
        ("witness_iid_b", "iid_b"),
        ("asset_a", "asset_a"),
        ("asset_b", "asset_b"),
        ("frame_a", "frame_a"),
        ("frame_b", "frame_b"),
    ):
        output[target][selected_ordinals] = iid_arrays[source][
            selected_input
        ]
    component_mapping = [
        {
            "iid": iid,
            "base_component": iid_to_base_component[iid],
            "anchor": iid_anchor_flags[iid],
        }
        for iid in iids
    ]
    contract = {
        "schema_version": COMPONENT_PAIR_MAXIMA_SCHEMA,
        "source_iid_pair_artifact_digest":
            iid_pair_maxima["artifact_digest"],
        "source_graph_digest": iid_contract["graph_digest"],
        "source_graph_digest_version": GRAPH_DIGEST_VERSION,
        "source_graph_commit_binding": graph_binding,
        "source_graph_commit_binding_sha256": _object_digest(
            graph_binding
        ),
        "iid_identifiers": iids,
        "iid_anchor_flags": anchors_by_iid,
        "iid_to_base_component": component_mapping,
        "iid_to_base_component_sha256": _object_digest(
            component_mapping
        ),
        "base_component_identifiers": component_ids,
        "base_component_anchor_flags": [
            bool(value) for value in component_anchor.tolist()
        ],
        "base_components": components,
        "component_reduction": COMPONENT_REDUCTION_VERSION,
        "compute_dtype": "float32",
        "witness_tie_break": WITNESS_TIE_BREAK,
        "endpoint_class_codes": dict(ENDPOINT_CLASS_CODES),
        "endpoint_class_semantics":
            "component-is-A-iff-it-contains-an-anchor-IID",
        "expected_component_pairs": count,
        "observed_component_pairs": count,
        "coverage_complete": True,
        "statistical_unit": "pre-dino-base-component-pair",
        "thresholds_human_calibrated": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    artifact = _make_artifact(
        schema=COMPONENT_PAIR_MAXIMA_SCHEMA,
        contract=contract,
        arrays=output,
    )
    validate_component_pair_maxima(
        iid_pair_maxima,
        artifact,
        iid_to_base_component=iid_to_base_component,
        iid_anchor_flags=iid_anchor_flags,
    )
    return artifact


def validate_component_pair_maxima(
    iid_pair_maxima: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    iid_to_base_component: Mapping[str, str],
    iid_anchor_flags: Mapping[str, bool],
) -> dict[str, Any]:
    iid_contract, unused_iid_arrays = _validate_artifact_envelope(
        iid_pair_maxima,
        schema=IID_PAIR_MAXIMA_SCHEMA,
        array_dtypes=_IID_ARRAY_DTYPES,
    )
    del unused_iid_arrays
    graph_binding = _normalize_graph_commit_binding(
        iid_contract.get("graph_commit_binding")
    )
    if (
        iid_contract.get("graph_digest_version")
        != GRAPH_DIGEST_VERSION
        or iid_contract.get("graph_commit_binding_sha256")
        != _object_digest(graph_binding)
    ):
        raise ValueError("source IID graph commit binding differs")
    contract, arrays = _validate_artifact_envelope(
        artifact,
        schema=COMPONENT_PAIR_MAXIMA_SCHEMA,
        array_dtypes=_COMPONENT_ARRAY_DTYPES,
    )
    (
        iids,
        iid_component,
        component_ids,
        component_anchor,
        anchors_by_iid,
    ) = _component_inputs(
        iid_pair_maxima,
        iid_to_base_component,
        iid_anchor_flags,
    )
    components = len(component_ids)
    count = _pair_count(components)
    component_a, component_b = _enumerated_pairs(components)
    mapping = [
        {
            "iid": iid,
            "base_component": iid_to_base_component[iid],
            "anchor": iid_anchor_flags[iid],
        }
        for iid in iids
    ]
    expected_contract = {
        "schema_version": COMPONENT_PAIR_MAXIMA_SCHEMA,
        "source_iid_pair_artifact_digest":
            iid_pair_maxima["artifact_digest"],
        "source_graph_digest": iid_contract["graph_digest"],
        "source_graph_digest_version": GRAPH_DIGEST_VERSION,
        "source_graph_commit_binding": graph_binding,
        "source_graph_commit_binding_sha256": _object_digest(
            graph_binding
        ),
        "iid_identifiers": iids,
        "iid_anchor_flags": anchors_by_iid,
        "iid_to_base_component": mapping,
        "iid_to_base_component_sha256": _object_digest(mapping),
        "base_component_identifiers": component_ids,
        "base_component_anchor_flags": [
            bool(value) for value in component_anchor.tolist()
        ],
        "base_components": components,
        "component_reduction": COMPONENT_REDUCTION_VERSION,
        "compute_dtype": "float32",
        "witness_tie_break": WITNESS_TIE_BREAK,
        "endpoint_class_codes": dict(ENDPOINT_CLASS_CODES),
        "endpoint_class_semantics":
            "component-is-A-iff-it-contains-an-anchor-IID",
        "expected_component_pairs": count,
        "observed_component_pairs": count,
        "coverage_complete": True,
        "statistical_unit": "pre-dino-base-component-pair",
        "thresholds_human_calibrated": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    if contract != expected_contract:
        raise ValueError("component maxima contract differs")
    if any(value.shape != (count,) for value in arrays.values()):
        raise ValueError("component maxima array shape differs")
    expected_endpoint = np.empty(count, dtype=np.uint8)
    left_anchor = component_anchor[component_a]
    right_anchor = component_anchor[component_b]
    expected_endpoint[np.logical_and(left_anchor, right_anchor)] = 0
    expected_endpoint[np.logical_xor(left_anchor, right_anchor)] = 1
    expected_endpoint[
        np.logical_not(np.logical_or(left_anchor, right_anchor))
    ] = 2
    witness_left = arrays["witness_iid_a"].astype(np.int64)
    witness_right = arrays["witness_iid_b"].astype(np.int64)
    if (
        not np.array_equal(arrays["component_a"], component_a)
        or not np.array_equal(arrays["component_b"], component_b)
        or not np.array_equal(
            arrays["endpoint_class"],
            expected_endpoint,
        )
        or not np.isfinite(arrays["score"]).all()
        or np.any(arrays["score"] < np.float32(-1.0))
        or np.any(arrays["score"] > np.float32(1.0))
        or np.any(witness_left < 0)
        or np.any(witness_right <= witness_left)
        or np.any(witness_right >= len(iids))
        or np.any(
            arrays["asset_a"].astype(np.int64) // 2
            != witness_left
        )
        or np.any(
            arrays["asset_b"].astype(np.int64) // 2
            != witness_right
        )
        or np.any(arrays["frame_a"] >= DINO_FRAMES)
        or np.any(arrays["frame_b"] >= DINO_FRAMES)
    ):
        raise ValueError("component maxima semantic arrays differ")
    witness_component_left = iid_component[witness_left]
    witness_component_right = iid_component[witness_right]
    if (
        np.any(
            np.minimum(
                witness_component_left,
                witness_component_right,
            )
            != component_a
        )
        or np.any(
            np.maximum(
                witness_component_left,
                witness_component_right,
            )
            != component_b
        )
    ):
        raise ValueError("component maxima witness endpoints differ")
    return dict(artifact)


def _score_bin_indices(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores)
    if (
        values.dtype != np.dtype("float32")
        or values.ndim != 1
        or not np.isfinite(values).all()
        or np.any(values < np.float32(-1.0))
        or np.any(values > np.float32(1.0))
    ):
        raise ValueError("quotient calibration scores differ")
    return np.searchsorted(
        _SCORE_CUTS,
        values,
        side="right",
    ).astype(np.uint8)


def _score_bin_contract() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(SCORE_BIN_NAMES):
        rows.append(
            {
                "index": index,
                "name": name,
                "lower": float(_SCORE_BOUNDS[index]),
                "lower_operator": ">=",
                "upper": float(_SCORE_BOUNDS[index + 1]),
                "upper_operator": "<=" if index == 5 else "<",
                "comparison_dtype": "float32",
            }
        )
    return rows


def _priority(
    *,
    seed: int,
    component_a: str,
    component_b: str,
) -> bytes:
    payload = _canonical_json(
        [
            SAMPLE_PAIR_ID_VERSION,
            seed,
            component_a,
            component_b,
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


_SAMPLE_ARRAY_DTYPES = {
    "component_a": "int32",
    "component_b": "int32",
    "endpoint_class": "uint8",
    "score_bin": "uint8",
    "sampling_stratum": "uint8",
    "score": "float32",
    "witness_iid_a": "int32",
    "witness_iid_b": "int32",
    "asset_a": "int32",
    "asset_b": "int32",
    "frame_a": "uint8",
    "frame_b": "uint8",
    "hash_priority_sha256": "uint8",
    "sample_rank_within_stratum": "int32",
    "N_h": "int64",
    "n_h": "int32",
    "sampling_probability": "float64",
    "sampling_weight": "float64",
}


def _derive_sample(
    component_artifact: Mapping[str, Any],
    *,
    seed: int,
    samples_per_stratum: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    contract = component_artifact["contract"]
    arrays = component_artifact["arrays"]
    component_ids = contract["base_component_identifiers"]
    score_bins = _score_bin_indices(arrays["score"])
    endpoint = arrays["endpoint_class"]
    strata_count = len(SCORE_BIN_NAMES) * len(ENDPOINT_CLASS_NAMES)
    stratum = (
        endpoint.astype(np.int64) * len(SCORE_BIN_NAMES)
        + score_bins.astype(np.int64)
    )
    populations = np.bincount(
        stratum,
        minlength=strata_count,
    ).astype(np.int64)
    heaps: list[list[tuple[int, int, int, int, bytes]]] = [
        [] for _ in range(strata_count)
    ]
    for row_index in range(len(arrays["score"])):
        left = int(arrays["component_a"][row_index])
        right = int(arrays["component_b"][row_index])
        digest = _priority(
            seed=seed,
            component_a=component_ids[left],
            component_b=component_ids[right],
        )
        priority = int.from_bytes(digest, byteorder="big", signed=False)
        key = (priority, left, right)
        heap = heaps[int(stratum[row_index])]
        if len(heap) >= samples_per_stratum:
            worst = (-heap[0][0], -heap[0][1], -heap[0][2])
            if key >= worst:
                continue
        item = (-priority, -left, -right, row_index, digest)
        if len(heap) < samples_per_stratum:
            heapq.heappush(heap, item)
        else:
            heapq.heapreplace(heap, item)
    selected: list[tuple[int, int, bytes]] = []
    strata_metadata: list[dict[str, Any]] = []
    for endpoint_code, endpoint_name in enumerate(ENDPOINT_CLASS_NAMES):
        for bin_index, bin_name in enumerate(SCORE_BIN_NAMES):
            index = endpoint_code * len(SCORE_BIN_NAMES) + bin_index
            population = int(populations[index])
            sample_size = min(samples_per_stratum, population)
            bucket = [
                (item[3], item[4])
                for item in heaps[index]
            ]
            bucket.sort(
                key=lambda item: (
                    item[1],
                    int(arrays["component_a"][item[0]]),
                    int(arrays["component_b"][item[0]]),
                )
            )
            if len(bucket) != sample_size:
                raise RuntimeError("quotient bottom-k bucket differs")
            probability = (
                0.0 if population == 0 else sample_size / population
            )
            weight = (
                None
                if sample_size == 0
                else population / sample_size
            )
            strata_metadata.append(
                {
                    "index": index,
                    "endpoint_class": endpoint_name,
                    "endpoint_class_code": endpoint_code,
                    "score_bin": bin_name,
                    "score_bin_index": bin_index,
                    "N_h": population,
                    "n_h": sample_size,
                    "sampling_probability": probability,
                    "sampling_weight": weight,
                }
            )
            for rank, (row_index, digest) in enumerate(
                bucket,
                start=1,
            ):
                selected.append((row_index, rank, digest))
    sample_count = len(selected)
    output: dict[str, np.ndarray] = {}
    source_fields = (
        "component_a",
        "component_b",
        "endpoint_class",
        "score",
        "witness_iid_a",
        "witness_iid_b",
        "asset_a",
        "asset_b",
        "frame_a",
        "frame_b",
    )
    selected_indices = np.asarray(
        [item[0] for item in selected],
        dtype=np.int64,
    )
    for field in source_fields:
        output[field] = np.ascontiguousarray(
            arrays[field][selected_indices]
        )
    output["score_bin"] = np.asarray(
        [int(score_bins[item[0]]) for item in selected],
        dtype=np.uint8,
    )
    output["sampling_stratum"] = np.asarray(
        [
            int(stratum[item[0]])
            for item in selected
        ],
        dtype=np.uint8,
    )
    output["hash_priority_sha256"] = np.ascontiguousarray(
        np.asarray(
            [list(item[2]) for item in selected],
            dtype=np.uint8,
        ).reshape(sample_count, 32)
    )
    output["sample_rank_within_stratum"] = np.asarray(
        [item[1] for item in selected],
        dtype=np.int32,
    )
    output["N_h"] = np.asarray(
        [
            strata_metadata[int(stratum[item[0]])]["N_h"]
            for item in selected
        ],
        dtype=np.int64,
    )
    output["n_h"] = np.asarray(
        [
            strata_metadata[int(stratum[item[0]])]["n_h"]
            for item in selected
        ],
        dtype=np.int32,
    )
    output["sampling_probability"] = np.asarray(
        [
            strata_metadata[int(stratum[item[0]])][
                "sampling_probability"
            ]
            for item in selected
        ],
        dtype=np.float64,
    )
    output["sampling_weight"] = np.asarray(
        [
            strata_metadata[int(stratum[item[0]])]["sampling_weight"]
            for item in selected
        ],
        dtype=np.float64,
    )
    return output, strata_metadata


def build_quotient_calibration_sample(
    component_pair_maxima: Mapping[str, Any],
    *,
    seed: int = SAMPLE_SEED,
    samples_per_stratum: int = DEFAULT_SAMPLE_PER_STRATUM,
) -> dict[str, Any]:
    """Create a deterministic fixed-bin bottom-k quotient-pair sample."""

    component_contract, unused_arrays = _validate_artifact_envelope(
        component_pair_maxima,
        schema=COMPONENT_PAIR_MAXIMA_SCHEMA,
        array_dtypes=_COMPONENT_ARRAY_DTYPES,
    )
    del unused_arrays
    graph_binding = _normalize_graph_commit_binding(
        component_contract.get("source_graph_commit_binding")
    )
    if (
        type(seed) is not int
        or seed < 0
        or type(samples_per_stratum) is not int
        or samples_per_stratum < 1
        or component_contract.get("coverage_complete") is not True
        or component_contract.get("statistical_unit")
        != "pre-dino-base-component-pair"
        or component_contract.get(
            "source_graph_commit_binding_sha256"
        )
        != _object_digest(graph_binding)
    ):
        raise ValueError("quotient sampling contract input differs")
    arrays, strata = _derive_sample(
        component_pair_maxima,
        seed=seed,
        samples_per_stratum=samples_per_stratum,
    )
    population = int(
        component_contract["observed_component_pairs"]
    )
    if sum(item["N_h"] for item in strata) != population:
        raise RuntimeError("quotient stratum population is not conserved")
    contract = {
        "schema_version": QUOTIENT_SAMPLE_SCHEMA,
        "source_component_pair_artifact_digest":
            component_pair_maxima["artifact_digest"],
        "source_graph_commit_binding": graph_binding,
        "source_graph_commit_binding_sha256": _object_digest(
            graph_binding
        ),
        "source_component_pairs": population,
        "statistical_unit": "pre-dino-base-component-pair",
        "statistical_unit_alias": "quotient_pair",
        "sampling_method": SAMPLE_METHOD,
        "pair_id_version": SAMPLE_PAIR_ID_VERSION,
        "hash": "sha256",
        "hash_input":
            "canonical-json([pair-id-version,seed,component-a,component-b])",
        "seed": seed,
        "samples_per_stratum": samples_per_stratum,
        "score_bins": _score_bin_contract(),
        "endpoint_class_codes": dict(ENDPOINT_CLASS_CODES),
        "stratum_index":
            "endpoint-class-code * 6 + score-bin-index",
        "strata": strata,
        "strata_sha256": _object_digest(strata),
        "population_count": population,
        "sample_count": len(arrays["score"]),
        "selection_complete": True,
        "intended_use": "human-threshold-calibration-label-collection",
        "thresholds_human_calibrated": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    artifact = _make_artifact(
        schema=QUOTIENT_SAMPLE_SCHEMA,
        contract=contract,
        arrays=arrays,
    )
    validate_quotient_calibration_sample(
        component_pair_maxima,
        artifact,
    )
    return artifact


def validate_quotient_calibration_sample(
    component_pair_maxima: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    component_contract, unused_component_arrays = (
        _validate_artifact_envelope(
            component_pair_maxima,
            schema=COMPONENT_PAIR_MAXIMA_SCHEMA,
            array_dtypes=_COMPONENT_ARRAY_DTYPES,
        )
    )
    del unused_component_arrays
    graph_binding = _normalize_graph_commit_binding(
        component_contract.get("source_graph_commit_binding")
    )
    if component_contract.get(
        "source_graph_commit_binding_sha256"
    ) != _object_digest(graph_binding):
        raise ValueError("component graph commit binding differs")
    contract, arrays = _validate_artifact_envelope(
        artifact,
        schema=QUOTIENT_SAMPLE_SCHEMA,
        array_dtypes=_SAMPLE_ARRAY_DTYPES,
    )
    seed = contract.get("seed")
    per_stratum = contract.get("samples_per_stratum")
    if (
        type(seed) is not int
        or seed < 0
        or type(per_stratum) is not int
        or per_stratum < 1
    ):
        raise ValueError("quotient sample seed/size differs")
    expected_arrays, strata = _derive_sample(
        component_pair_maxima,
        seed=seed,
        samples_per_stratum=per_stratum,
    )
    population = int(
        component_contract["observed_component_pairs"]
    )
    expected_contract = {
        "schema_version": QUOTIENT_SAMPLE_SCHEMA,
        "source_component_pair_artifact_digest":
            component_pair_maxima["artifact_digest"],
        "source_graph_commit_binding": graph_binding,
        "source_graph_commit_binding_sha256": _object_digest(
            graph_binding
        ),
        "source_component_pairs": population,
        "statistical_unit": "pre-dino-base-component-pair",
        "statistical_unit_alias": "quotient_pair",
        "sampling_method": SAMPLE_METHOD,
        "pair_id_version": SAMPLE_PAIR_ID_VERSION,
        "hash": "sha256",
        "hash_input":
            "canonical-json([pair-id-version,seed,component-a,component-b])",
        "seed": seed,
        "samples_per_stratum": per_stratum,
        "score_bins": _score_bin_contract(),
        "endpoint_class_codes": dict(ENDPOINT_CLASS_CODES),
        "stratum_index":
            "endpoint-class-code * 6 + score-bin-index",
        "strata": strata,
        "strata_sha256": _object_digest(strata),
        "population_count": population,
        "sample_count": len(expected_arrays["score"]),
        "selection_complete": True,
        "intended_use": "human-threshold-calibration-label-collection",
        "thresholds_human_calibrated": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    if contract != expected_contract:
        raise ValueError("quotient sample contract differs")
    if set(arrays) != set(expected_arrays):
        raise ValueError("quotient sample arrays differ")
    for name in arrays:
        if (
            arrays[name].shape != expected_arrays[name].shape
            or arrays[name].dtype != expected_arrays[name].dtype
            or not np.array_equal(
                arrays[name],
                expected_arrays[name],
            )
        ):
            raise ValueError(f"quotient sample array {name} differs")
    if (
        sum(item["N_h"] for item in strata) != population
        or contract["thresholds_human_calibrated"] is not False
        or contract["training_authorized"] is not False
    ):
        raise ValueError("quotient sample safety/population differs")
    return dict(artifact)


_SCHEMA_ARRAY_DTYPES: dict[str, Mapping[str, str]] = {
    RANK_PARTIAL_SCHEMA: _RANK_ARRAY_DTYPES,
    IID_PAIR_MAXIMA_SCHEMA: _IID_ARRAY_DTYPES,
    COMPONENT_PAIR_MAXIMA_SCHEMA: _COMPONENT_ARRAY_DTYPES,
    QUOTIENT_SAMPLE_SCHEMA: _SAMPLE_ARRAY_DTYPES,
}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _write_exclusive(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o444,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _npy_bytes(array: np.ndarray) -> bytes:
    value = np.asarray(array)
    if (
        value.dtype.hasobject
        or value.dtype.kind in {"O", "V"}
        or not value.flags.c_contiguous
    ):
        raise ValueError("artifact arrays must be C-contiguous numeric data")
    output = io.BytesIO()
    np.lib.format.write_array(
        output,
        value,
        allow_pickle=False,
    )
    return output.getvalue()


def _write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w+b", closefd=False) as handle:
            with zipfile.ZipFile(
                handle,
                mode="w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as archive:
                for name in sorted(arrays):
                    if (
                        type(name) is not str
                        or not name
                        or "/" in name
                        or "\\" in name
                        or "\x00" in name
                    ):
                        raise ValueError("artifact array name is unsafe")
                    member = zipfile.ZipInfo(
                        filename=f"{name}.npy",
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    member.compress_type = zipfile.ZIP_STORED
                    member.create_system = 3
                    member.external_attr = (stat.S_IFREG | 0o444) << 16
                    archive.writestr(
                        member,
                        _npy_bytes(np.asarray(arrays[name])),
                    )
            handle.flush()
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file(path: Path, label: str) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"artifact {label} is missing") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"artifact {label} is not a regular file")


def _artifact_directory(path: Path) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("artifact directory is missing") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError(
            "artifact directory is a symlink or is not a directory"
        )
    expected = {
        ARTIFACT_METADATA_NAME,
        ARTIFACT_ARRAYS_NAME,
        ARTIFACT_DONE_NAME,
    }
    observed = {entry.name for entry in os.scandir(path)}
    if observed != expected:
        raise ValueError("artifact directory entries differ")
    for name in sorted(expected):
        _regular_file(path / name, name)


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as error:
        raise ValueError(f"artifact {label} JSON is invalid") from error
    if (
        not isinstance(value, dict)
        or payload != _canonical_json_bytes(value)
    ):
        raise ValueError(f"artifact {label} JSON is not canonical")
    return value


def _validate_npz_members(
    path: Path,
    *,
    expected_arrays: Mapping[str, str],
) -> None:
    expected = [f"{name}.npy" for name in sorted(expected_arrays)]
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if (
                names != expected
                or len(set(names)) != len(names)
                or any(
                    member.is_dir()
                    or member.compress_type != zipfile.ZIP_STORED
                    or member.flag_bits & 0x1
                    or member.filename.startswith("/")
                    or ".." in Path(member.filename).parts
                    for member in members
                )
            ):
                raise ValueError("artifact NPZ member set/layout differs")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(
                    f"artifact NPZ CRC differs: {bad_member}"
                )
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("artifact NPZ is invalid") from error


def _load_npz(
    path: Path,
    *,
    expected_arrays: Mapping[str, str],
) -> dict[str, np.ndarray]:
    _validate_npz_members(path, expected_arrays=expected_arrays)
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if (
                archive.files != sorted(expected_arrays)
                or len(set(archive.files)) != len(archive.files)
            ):
                raise ValueError("artifact NPZ array names differ")
            for name in sorted(expected_arrays):
                value = archive[name]
                if (
                    not isinstance(value, np.ndarray)
                    or value.dtype != np.dtype(expected_arrays[name])
                    or value.dtype.hasobject
                    or value.dtype.kind in {"O", "V"}
                    or not value.flags.c_contiguous
                ):
                    raise ValueError(
                        f"artifact NPZ array {name} dtype/layout differs"
                    )
                arrays[name] = value.copy(order="C")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("artifact NPZ arrays are invalid") from error
    return arrays


def validate_artifact_by_schema(
    artifact: Mapping[str, Any],
    *,
    graph_binding: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]] | None = None,
    iid_pair_maxima: Mapping[str, Any] | None = None,
    component_pair_maxima: Mapping[str, Any] | None = None,
    iid_to_base_component: Mapping[str, str] | None = None,
    iid_anchor_flags: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Dispatch strict semantic validation for every quotient artifact.

    Context is mandatory because rank/IID artifacts are bound to canonical
    graph rows, while downstream artifacts are bound to their exact source
    artifact and pre-DINO component mapping.
    """

    if not isinstance(artifact, Mapping):
        raise ValueError("quotient artifact is not a mapping")
    binding = _normalize_graph_commit_binding(graph_binding)
    schema = artifact.get("schema_version")
    contract = artifact.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("quotient artifact contract is missing")
    binding_field = (
        "graph_commit_binding"
        if schema in {RANK_PARTIAL_SCHEMA, IID_PAIR_MAXIMA_SCHEMA}
        else "source_graph_commit_binding"
    )
    if (
        contract.get(binding_field) != binding
        or contract.get(f"{binding_field}_sha256")
        != _object_digest(binding)
    ):
        raise ValueError("quotient artifact graph commit binding differs")
    if schema == RANK_PARTIAL_SCHEMA:
        if rows is None:
            raise ValueError("rank partial validation requires graph rows")
        return validate_rank_partial_artifact(
            rows,
            artifact,
            graph_binding=binding,
        )
    if schema == IID_PAIR_MAXIMA_SCHEMA:
        if rows is None:
            raise ValueError("IID maxima validation requires graph rows")
        return validate_iid_pair_maxima(
            rows,
            artifact,
            graph_binding=binding,
        )
    if schema == COMPONENT_PAIR_MAXIMA_SCHEMA:
        if (
            iid_pair_maxima is None
            or iid_to_base_component is None
            or iid_anchor_flags is None
        ):
            raise ValueError(
                "component maxima validation requires IID artifact "
                "and mappings"
            )
        return validate_component_pair_maxima(
            iid_pair_maxima,
            artifact,
            iid_to_base_component=iid_to_base_component,
            iid_anchor_flags=iid_anchor_flags,
        )
    if schema == QUOTIENT_SAMPLE_SCHEMA:
        if component_pair_maxima is None:
            raise ValueError(
                "calibration sample validation requires component artifact"
            )
        return validate_quotient_calibration_sample(
            component_pair_maxima,
            artifact,
        )
    raise ValueError(f"unsupported quotient artifact schema: {schema!r}")


def publish_artifact_directory(
    directory: str | os.PathLike[str],
    artifact: Mapping[str, Any],
    *,
    graph_binding: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]] | None = None,
    iid_pair_maxima: Mapping[str, Any] | None = None,
    component_pair_maxima: Mapping[str, Any] | None = None,
    iid_to_base_component: Mapping[str, str] | None = None,
    iid_anchor_flags: Mapping[str, bool] | None = None,
    _include_permission_contract: bool = True,
) -> dict[str, Any]:
    """Atomically publish one validated ndarray artifact directory.

    Publication is create-only.  It writes a deterministic uncompressed NPZ,
    canonical metadata, and a canonical done record into a sibling staging
    directory before one directory rename.
    """

    validate_artifact_by_schema(
        artifact,
        graph_binding=graph_binding,
        rows=rows,
        iid_pair_maxima=iid_pair_maxima,
        component_pair_maxima=component_pair_maxima,
        iid_to_base_component=iid_to_base_component,
        iid_anchor_flags=iid_anchor_flags,
    )
    raw_target = Path(directory).expanduser()
    if not raw_target.name or raw_target.name in {".", ".."}:
        raise ValueError("artifact output directory name is invalid")
    parent = raw_target.parent.resolve(strict=True)
    target = parent / raw_target.name
    if os.path.lexists(target):
        raise FileExistsError(f"artifact output already exists: {target}")
    schema = artifact["schema_version"]
    expected_dtypes = _SCHEMA_ARRAY_DTYPES.get(schema)
    if expected_dtypes is None:
        raise ValueError("artifact schema has no disk representation")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=parent,
        )
    )
    published = False
    try:
        arrays_path = staging / ARTIFACT_ARRAYS_NAME
        _write_deterministic_npz(arrays_path, artifact["arrays"])
        arrays_sha256 = _sha256_file(arrays_path)
        metadata = {
            "schema_version": ARTIFACT_METADATA_SCHEMA,
            "artifact_schema": schema,
            "artifact_digest": artifact["artifact_digest"],
            "contract": dict(artifact["contract"]),
            "array_descriptors": dict(artifact["array_descriptors"]),
            "arrays_filename": ARTIFACT_ARRAYS_NAME,
            "arrays_npz_sha256": arrays_sha256,
            "storage_format": "deterministic-uncompressed-npz-v1",
        }
        metadata_path = staging / ARTIFACT_METADATA_NAME
        _write_exclusive(
            metadata_path,
            _canonical_json_bytes(metadata),
        )
        metadata_sha256 = _sha256_file(metadata_path)
        done = {
            "schema_version": ARTIFACT_DONE_SCHEMA,
            "status": "complete",
            "artifact_schema": schema,
            "artifact_digest": artifact["artifact_digest"],
            "metadata_filename": ARTIFACT_METADATA_NAME,
            "metadata_sha256": metadata_sha256,
            "arrays_filename": ARTIFACT_ARRAYS_NAME,
            "arrays_npz_sha256": arrays_sha256,
            "files": sorted(
                {
                    ARTIFACT_METADATA_NAME,
                    ARTIFACT_ARRAYS_NAME,
                    ARTIFACT_DONE_NAME,
                }
            ),
        }
        if _include_permission_contract:
            done["permission_contract"] = (
                artifact_permissions.permission_contract()
            )
        _write_exclusive(
            staging / ARTIFACT_DONE_NAME,
            _canonical_json_bytes(done),
        )
        _fsync_directory(staging)
        # Prove the staged bytes pass storage and semantic validation before
        # the directory can become visible at its committed name.
        load_artifact_directory(
            staging,
            graph_binding=graph_binding,
            rows=rows,
            iid_pair_maxima=iid_pair_maxima,
            component_pair_maxima=component_pair_maxima,
            iid_to_base_component=iid_to_base_component,
            iid_anchor_flags=iid_anchor_flags,
            _verify_permissions=False,
        )
        if _include_permission_contract:
            artifact_permissions.seal_staging_tree(
                staging,
                leave_root_writable=True,
            )
            artifact_permissions.assert_sealed_tree(
                staging,
                allow_writable_root=True,
            )
        if os.path.lexists(target):
            raise FileExistsError(
                f"artifact output appeared during publish: {target}"
            )
        os.rename(staging, target)
        if _include_permission_contract:
            artifact_permissions.seal_published_root(target)
        published = True
        _fsync_directory(parent)
        return done
    finally:
        if not published and staging.exists():
            artifact_permissions.remove_staging_tree(staging)


def load_artifact_directory(
    directory: str | os.PathLike[str],
    *,
    graph_binding: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]] | None = None,
    iid_pair_maxima: Mapping[str, Any] | None = None,
    component_pair_maxima: Mapping[str, Any] | None = None,
    iid_to_base_component: Mapping[str, str] | None = None,
    iid_anchor_flags: Mapping[str, bool] | None = None,
    _verify_permissions: bool = True,
) -> dict[str, Any]:
    """Load, storage-verify, and semantically validate an artifact."""

    path = Path(directory).expanduser()
    _artifact_directory(path)
    done_path = path / ARTIFACT_DONE_NAME
    metadata_path = path / ARTIFACT_METADATA_NAME
    arrays_path = path / ARTIFACT_ARRAYS_NAME
    done = _load_canonical_json(done_path, "done")
    expected_done_fields = {
        "schema_version",
        "status",
        "artifact_schema",
        "artifact_digest",
        "metadata_filename",
        "metadata_sha256",
        "arrays_filename",
        "arrays_npz_sha256",
        "files",
    }
    observed_done_fields = set(done)
    sealed = "permission_contract" in observed_done_fields
    if sealed:
        expected_done_fields.add("permission_contract")
    if (
        observed_done_fields != expected_done_fields
        or done.get("schema_version") != ARTIFACT_DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("artifact_schema") not in _SCHEMA_ARRAY_DTYPES
        or not _is_sha256(done.get("artifact_digest"))
        or done.get("metadata_filename") != ARTIFACT_METADATA_NAME
        or not _is_sha256(done.get("metadata_sha256"))
        or done.get("arrays_filename") != ARTIFACT_ARRAYS_NAME
        or not _is_sha256(done.get("arrays_npz_sha256"))
        or done.get("files")
        != sorted(
            {
                ARTIFACT_METADATA_NAME,
                ARTIFACT_ARRAYS_NAME,
                ARTIFACT_DONE_NAME,
            }
        )
    ):
        raise ValueError("artifact done contract differs")
    if sealed:
        artifact_permissions.validate_permission_contract(
            done["permission_contract"]
        )
        if _verify_permissions:
            artifact_permissions.assert_sealed_tree(path)
    if _sha256_file(metadata_path) != done["metadata_sha256"]:
        raise ValueError("artifact metadata file digest differs")
    if _sha256_file(arrays_path) != done["arrays_npz_sha256"]:
        raise ValueError("artifact NPZ file digest differs")
    metadata = _load_canonical_json(metadata_path, "metadata")
    expected_metadata_fields = {
        "schema_version",
        "artifact_schema",
        "artifact_digest",
        "contract",
        "array_descriptors",
        "arrays_filename",
        "arrays_npz_sha256",
        "storage_format",
    }
    if (
        set(metadata) != expected_metadata_fields
        or metadata.get("schema_version") != ARTIFACT_METADATA_SCHEMA
        or metadata.get("artifact_schema") != done["artifact_schema"]
        or metadata.get("artifact_digest") != done["artifact_digest"]
        or metadata.get("arrays_filename") != ARTIFACT_ARRAYS_NAME
        or metadata.get("arrays_npz_sha256")
        != done["arrays_npz_sha256"]
        or metadata.get("storage_format")
        != "deterministic-uncompressed-npz-v1"
        or not isinstance(metadata.get("contract"), dict)
        or not isinstance(metadata.get("array_descriptors"), dict)
    ):
        raise ValueError("artifact metadata contract differs")
    expected_dtypes = _SCHEMA_ARRAY_DTYPES[done["artifact_schema"]]
    arrays = _load_npz(
        arrays_path,
        expected_arrays=expected_dtypes,
    )
    artifact = {
        "schema_version": done["artifact_schema"],
        "contract": metadata["contract"],
        "array_descriptors": metadata["array_descriptors"],
        "arrays": arrays,
        "artifact_digest": done["artifact_digest"],
    }
    validate_artifact_by_schema(
        artifact,
        graph_binding=graph_binding,
        rows=rows,
        iid_pair_maxima=iid_pair_maxima,
        component_pair_maxima=component_pair_maxima,
        iid_to_base_component=iid_to_base_component,
        iid_anchor_flags=iid_anchor_flags,
    )
    return artifact


__all__ = [
    "ARTIFACT_ARRAYS_NAME",
    "ARTIFACT_DONE_NAME",
    "ARTIFACT_METADATA_NAME",
    "COMPONENT_PAIR_MAXIMA_SCHEMA",
    "DEFAULT_SAMPLE_PER_STRATUM",
    "ENDPOINT_CLASS_CODES",
    "ENDPOINT_CLASS_NAMES",
    "GRAPH_COMMIT_BINDING_SCHEMA",
    "GRAPH_DIGEST_VERSION",
    "IID_PAIR_MAXIMA_SCHEMA",
    "QUOTIENT_SAMPLE_SCHEMA",
    "RANK_PARTIAL_SCHEMA",
    "RankQuotientAccumulator",
    "SAMPLE_SEED",
    "SCORE_BIN_NAMES",
    "WORLD_SIZE",
    "aggregate_base_component_pairs",
    "build_quotient_calibration_sample",
    "load_artifact_directory",
    "make_graph_commit_binding",
    "merge_exact8_rank_partials",
    "publish_artifact_directory",
    "validate_artifact_by_schema",
    "validate_component_pair_maxima",
    "validate_iid_pair_maxima",
    "validate_quotient_calibration_sample",
    "validate_rank_partial_artifact",
]
