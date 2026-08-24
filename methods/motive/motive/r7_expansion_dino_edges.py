"""Exact, hash-bound DINO edge matching for the R7 expansion graph.

The input is an immutable graph-input directory containing one row per
visual asset and one six-frame DINOv2 tensor per row.  Pair ownership is
defined by the smaller asset index: rank ``r`` evaluates every ``(a, b)``
with ``a < b`` and ``a % 8 == r``.  Consequently the union of exactly eight
valid shards is the complete upper triangle, without communication between
workers.

Hard edges are exhaustive at maximum six-by-six frame cosine >= 0.96.
The [0.92, 0.96) diagnostic band is explicitly non-hard.  Each shard keeps
the best K incident band edges per asset; the finalizer merges those local
candidate sets and computes the exact global top K per asset.  A local top-K
union is sufficient for a global top K because an item excluded from a
partition's top K cannot enter the top K of the union.

In addition, every evaluated pair is assigned to one fixed, preregistered
score stratum.  A shard records the exact population count and retains the
SHA-256 bottom-k asset-pair identifiers in each stratum.  The finalizer sums
the exact counts and applies the same bottom-k operation to the union of
local candidates, which is exactly the global bottom-k.  These statistical
calibration samples are separate from the diagnostic top-K and are always
marked as unlabelled and unauthorized for training.

The same normalized matcher output is also streamed exactly once into one
rank-local quotient accumulator.  Rank commits contain exact per-left-role
IID-pair partial maxima; the exact-eight finalizer reduces the two required
partials per IID pair and publishes every IID-pair maximum.  No second DINO
matcher call is made.  This IID artifact enables the downstream pre-DINO
base-component quotient, which remains the required statistical unit for
formal threshold calibration.

All commits are immutable directory renames.  ``resume`` only validates an
existing commit.  A matcher/runtime exception occurs before publication, so
it cannot publish a partial shard.  Cross-rank all-or-nothing behavior is a
launcher responsibility: the finalizer must only run after torchrun exits
successfully for all eight ranks, and itself requires all eight shards.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import r7_artifact_permissions as artifact_permissions
from . import r7_visual_graph_input as visual_graph_input
from . import r7_dino_quotient_calibration as quotient_calibration


MATCHER_SCHEMA = "motive-r7-expansion-dino-edge-matcher-v3"
HARD_EDGE_SCHEMA = "motive-r7-expansion-dino-hard-edge-v1"
AUDIT_EDGE_SCHEMA = "motive-r7-expansion-dino-audit-edge-v1"
CALIBRATION_EDGE_SCHEMA = (
    "motive-r7-expansion-dino-calibration-edge-v1"
)
CALIBRATION_METADATA_SCHEMA = (
    "motive-r7-expansion-dino-calibration-metadata-v1"
)
SHARD_SUMMARY_SCHEMA = "motive-r7-expansion-dino-edge-shard-summary-v3"
SHARD_DONE_SCHEMA = "motive-r7-expansion-dino-edge-shard-done-v3"
FINAL_SUMMARY_SCHEMA = "motive-r7-expansion-dino-edge-final-summary-v3"
FINAL_DONE_SCHEMA = "motive-r7-expansion-dino-edge-final-done-v3"

WORLD_SIZE = 8
DINO_FRAMES = 6
DINO_DIM = 768
HARD_THRESHOLD = 0.96
AUDIT_THRESHOLD = 0.92
COSINE_ROUND_DECIMALS = 8
DEFAULT_AUDIT_TOP_K = 20
DEFAULT_CALIBRATION_PER_STRATUM = 256
DEFAULT_BLOCK_SIZE = 256
PARTITION_VERSION = "smaller-asset-index-modulo-exactly-8-v1"
ALGORITHM_VERSION = "exact-max-6x6-float32-cosine-v3"
CALIBRATION_STRATA_VERSION = (
    "fixed-dino-score-strata-0p80-0p90-0p92-0p94-0p96-v1"
)
CALIBRATION_BOTTOM_K_VERSION = (
    "sha256-asset-index-pair-bottom-k-v1"
)
CALIBRATION_PAIR_ID_VERSION = "asset-index-pair-v1"
CALIBRATION_SAMPLING_SEED = 260108832
CALIBRATION_PRIORITY_BITS = 256
CALIBRATION_HISTOGRAM_SCHEMA = (
    "motive-r7-expansion-dino-score-histogram-v1"
)
CALIBRATION_HISTOGRAM_BIN_WIDTH = 0.001
CALIBRATION_HISTOGRAM_BIN_COUNT = 2000
CALIBRATION_PAIR_RELATIONS = ("same_iid", "cross_iid")
PROGRESS_OWNED_ASSET_INTERVAL = 128

# Fixed before looking at labels.  Adjacent intervals are lower-inclusive and
# upper-exclusive, except for the final interval which includes 1.0.  The
# intervals are non-overlapping and cover the complete valid cosine support.
_CALIBRATION_STRATUM_SPECS: tuple[
    tuple[str, float, float, str], ...
] = (
    ("low_complement", -1.0, 0.80, "<"),
    ("mid_complement", 0.80, 0.90, "<"),
    ("near_audit_complement", 0.90, 0.92, "<"),
    ("audit_lower", 0.92, 0.94, "<"),
    ("audit_upper", 0.94, 0.96, "<"),
    ("hard", 0.96, 1.0, "<="),
)
_CALIBRATION_CUTS_FLOAT32 = np.asarray(
    [spec[2] for spec in _CALIBRATION_STRATUM_SPECS[:-1]],
    dtype=np.float32,
)

SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
HARD_EDGES_NAME = "hard_edges.jsonl"
AUDIT_EDGES_NAME = "audit_edges.jsonl"
CALIBRATION_EDGES_NAME = "calibration_edges.jsonl"
QUOTIENT_RANK_PARTIAL_NAME = "quotient_rank_partial"
IID_PAIR_MAXIMA_NAME = "iid_pair_maxima"

_OUTPUT_FILE_NAMES = frozenset(
    {
        HARD_EDGES_NAME,
        AUDIT_EDGES_NAME,
        CALIBRATION_EDGES_NAME,
        SUMMARY_NAME,
        DONE_NAME,
    }
)
_QUOTIENT_ARTIFACT_FILES = frozenset(
    {
        quotient_calibration.ARTIFACT_METADATA_NAME,
        quotient_calibration.ARTIFACT_ARRAYS_NAME,
        quotient_calibration.ARTIFACT_DONE_NAME,
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "torch_version",
        "torch_cuda_version",
        "torch_hip_version",
        "device_type",
        "device_name",
        "tf32_allowed",
    }
)
_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "input_directory",
        "input_artifact_digest",
        "input_artifacts",
        "input_rows",
        "dino_contract",
        "dino_contract_sha256",
        "algorithm",
        "algorithm_sha256",
        "implementation",
        "runtime",
        "rank",
        "world_size",
        "device",
    }
)
_HARD_EDGE_FIELDS = frozenset(
    {
        "schema_version",
        "edge_type",
        "hard_edge",
        "asset_a",
        "asset_b",
        "iid_a",
        "role_a",
        "video_sha256_a",
        "iid_b",
        "role_b",
        "video_sha256_b",
        "cosine",
        "cosine_float32_hex",
        "frame_a",
        "frame_b",
        "owner_rank",
        "world_size",
    }
)
_AUDIT_EDGE_FIELDS = _HARD_EDGE_FIELDS | frozenset(
    {"selected_for_asset_indices"}
)
_CALIBRATION_EDGE_FIELDS = _HARD_EDGE_FIELDS | frozenset(
    {
        "score_stratum",
        "score_stratum_index",
        "score_stratum_lower",
        "score_stratum_lower_operator",
        "score_stratum_upper",
        "score_stratum_upper_operator",
        "pair_relation",
        "sampling_stratum",
        "sampling_stratum_index",
        "pair_id",
        "pair_id_sha256",
        "bottom_k_key_sha256",
        "hash_priority_sha256",
        "sampling_method",
        "sampling_scope",
        "sample_rank_within_stratum",
        "stratum_population_count",
        "stratum_sample_size",
        "sampling_probability",
        "sampling_weight",
        "human_labels_asserted",
        "training_authorized",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DHASH_RE = re.compile(r"^[0-9a-f]{16}$")

BlockMatcher = Callable[
    [np.ndarray, np.ndarray, int, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return (
        "".join(_canonical_json(dict(row)) + "\n" for row in rows)
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _object_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_canonical_jsonl(
    path: Path,
    *,
    allow_empty: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            try:
                value = json.loads(line)
            except Exception as error:
                raise ValueError(
                    f"{path}:{line_number} is invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} is not a JSON object"
                )
            if line != _canonical_json(value) + "\n":
                raise ValueError(
                    f"{path}:{line_number} is not canonical JSONL"
                )
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"{path} contains no rows")
    return rows


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_directory(
    target: Path,
    writer: Callable[[Path], Any],
) -> Any:
    target = target.expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
    )
    try:
        result = writer(staging)
        artifact_permissions.seal_staging_tree(
            staging,
            leave_root_writable=True,
        )
        artifact_permissions.assert_sealed_tree(
            staging,
            allow_writable_root=True,
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"commit target appeared during publication: {target}"
            )
        os.rename(staging, target)
        artifact_permissions.seal_published_root(target)
        _fsync_directory(target.parent)
        return result
    finally:
        if staging.exists():
            artifact_permissions.remove_staging_tree(staging)


def _regular_exact_files(
    directory: Path,
    expected_files: frozenset[str],
    *,
    expected_directories: frozenset[str] = frozenset(),
) -> tuple[dict[str, Path], dict[str, Path]]:
    directory = directory.expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError(directory)
    actual = {entry.name for entry in directory.iterdir()}
    expected = expected_files | expected_directories
    if actual != expected:
        raise ValueError(
            f"artifact set differs in {directory}: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    paths = {name: directory / name for name in expected_files}
    for path in paths.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact is not a regular file: {path}")
    directories = {
        name: directory / name for name in expected_directories
    }
    for path in directories.values():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(
                f"artifact directory is not a real directory: {path}"
            )
    return paths, directories


def _validate_sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_dino_contract(value: Any) -> dict[str, Any]:
    expected_fields = set(visual_graph_input.DINO_COMPARISON_FIELDS)
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("graph-input DINO contract fields differ")
    contract = dict(value)
    revision = contract.get("encoder_revision")
    if (
        contract.get("encoder_id") != "facebook/dinov2-base"
        or type(revision) is not str
        or re.fullmatch(r"[0-9a-f]{7,64}", revision) is None
        or type(contract.get("model_file_count")) is not int
        or contract["model_file_count"] < 1
        or contract.get("embedding_dim") != DINO_DIM
        or contract.get("dtype") != "float32"
        or contract.get("normalization") != "l2-per-frame"
        or contract.get("frozen_encoder") is not True
        or contract.get("local_files_only") is not True
    ):
        raise ValueError("graph-input DINO contract differs")
    _validate_sha(contract.get("model_tree_sha256"), "model tree digest")
    _validate_sha(contract.get("weights_sha256"), "weights digest")
    for field in (
        "frame_sampling_version",
        "preprocessing_version",
        "pooling",
    ):
        if type(contract.get(field)) is not str or not contract[field]:
            raise ValueError(f"DINO contract {field} is invalid")
    return contract


def _validate_asset_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("graph input contains no assets")
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for position, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"asset row {position} is not a mapping")
        if set(raw) != set(visual_graph_input.ROW_FIELDS):
            raise ValueError(
                f"asset row {position} does not match the external "
                "visual-graph row schema"
            )
        row = dict(raw)
        # This also rejects NaN and non-JSON extension values.
        _canonical_json(row)
        if (
            row.get("schema_version") != visual_graph_input.ROW_SCHEMA
            or type(row.get("asset_index")) is not int
            or row["asset_index"] != position
        ):
            raise ValueError(
                f"asset row {position} index/schema binding differs"
            )
        iid = row.get("iid")
        role = row.get("role")
        if (
            type(iid) is not str
            or not iid
            or iid.strip() != iid
            or "\x00" in iid
            or type(role) is not str
            or role not in visual_graph_input.ROLES
        ):
            raise ValueError(f"asset row {position} identity is invalid")
        identity = (iid, role)
        if identity in seen:
            raise ValueError(f"duplicate asset identity: {identity}")
        seen.add(identity)
        if type(row.get("anchor")) is not bool:
            raise ValueError(f"asset row {position} anchor is not boolean")
        cohort = row.get("cohort")
        source_input_index = row.get("source_input_index")
        if (
            cohort
            not in {
                "pseudo_positive",
                "pseudo_negative",
                "anchor_positive",
                "anchor_negative",
            }
            or row["anchor"] != str(cohort).startswith("anchor_")
            or type(source_input_index) is not int
            or source_input_index < 0
        ):
            raise ValueError(f"asset row {position} cohort/source differs")
        _validate_sha(
            row.get("video_sha256"),
            f"asset row {position} video digest",
        )
        _validate_sha(
            row.get("source_artifact_digest"),
            f"asset row {position} source artifact digest",
        )
        _validate_sha(
            row.get("source_index_digest"),
            f"asset row {position} source index digest",
        )
        hashes = row.get("dhashes")
        if (
            not isinstance(hashes, list)
            or len(hashes) != DINO_FRAMES
            or any(
                type(item) is not str
                or _DHASH_RE.fullmatch(item) is None
                for item in hashes
            )
        ):
            raise ValueError(f"asset row {position} dHashes are invalid")
        validated.append(row)
    if len(validated) % 2:
        raise ValueError("asset rows are not source/target paired")
    expected_order: list[tuple[str, int]] = []
    for offset in range(0, len(validated), 2):
        source, target = validated[offset : offset + 2]
        if (
            source["role"] != "source"
            or target["role"] != "target"
            or source["iid"] != target["iid"]
            or source["anchor"] is not target["anchor"]
            or source["cohort"] != target["cohort"]
            or source["source_artifact_digest"]
            != target["source_artifact_digest"]
            or source["source_input_index"]
            != target["source_input_index"]
        ):
            raise ValueError("asset source/target pair binding differs")
        expected_order.extend(
            [(str(source["iid"]), 0), (str(target["iid"]), 1)]
        )
    if expected_order != sorted(expected_order):
        raise ValueError("asset rows are not in canonical graph order")
    return validated


def _validate_feature_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    rows: int,
) -> dict[str, np.ndarray]:
    if set(arrays) != {"asset_indices", "dino_cls"}:
        raise ValueError(
            "graph-input feature arrays must be exactly "
            "asset_indices and dino_cls"
        )
    indices = np.asarray(arrays["asset_indices"])
    dino = np.asarray(arrays["dino_cls"])
    if (
        indices.dtype != np.dtype("int64")
        or indices.shape != (rows,)
        or not np.array_equal(
            indices,
            np.arange(rows, dtype=np.int64),
        )
    ):
        raise ValueError(
            "asset_indices must be int64, input ordered, and contiguous"
        )
    if (
        dino.dtype != np.dtype("float32")
        or dino.shape != (rows, DINO_FRAMES, DINO_DIM)
        or not np.isfinite(dino).all()
    ):
        raise ValueError(
            "dino_cls must be finite float32 [A,6,768]"
        )
    norms = np.linalg.norm(dino.astype(np.float64), axis=2)
    if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
        raise ValueError("dino_cls rows are not L2 normalized")
    return {
        "asset_indices": np.ascontiguousarray(indices),
        "dino_cls": np.ascontiguousarray(dino),
    }


def validate_graph_input(directory: Path) -> dict[str, Any]:
    """Consume the canonical consolidation commit without copying features."""

    return visual_graph_input.validate_graph_input_commit(directory)


def _graph_commit_binding(
    graph_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind quotient evidence to the exact validated graph-input commit."""

    return quotient_calibration.make_graph_commit_binding(
        artifact_digest=graph_input["artifact_digest"],
        artifact_hashes=graph_input["artifact_hashes"],
    )


def _validate_rank(rank: Any, world_size: Any) -> tuple[int, int]:
    if (
        type(rank) is not int
        or type(world_size) is not int
        or world_size != WORLD_SIZE
        or not 0 <= rank < WORLD_SIZE
    ):
        raise ValueError(
            f"rank/world_size must be rank in [0,{WORLD_SIZE}) "
            f"with exact world size {WORLD_SIZE}"
        )
    return rank, world_size


def shard_directory(output_root: Path, rank: int) -> Path:
    _validate_rank(rank, WORLD_SIZE)
    return (
        output_root.expanduser()
        / "shards"
        / f"rank-{rank:05d}-of-{WORLD_SIZE:05d}"
    )


def _calibration_strata_contract() -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "name": name,
            "lower": lower,
            "lower_operator": ">=",
            "upper": upper,
            "upper_operator": upper_operator,
        }
        for index, (
            name,
            lower,
            upper,
            upper_operator,
        ) in enumerate(_CALIBRATION_STRATUM_SPECS)
    ]


def _validate_calibration_strata_contract(value: Any) -> list[dict[str, Any]]:
    expected = _calibration_strata_contract()
    if value != expected:
        raise ValueError("calibration score strata contract differs")
    # Defensive proof independent of the literal declaration above.
    if expected[0]["lower"] != -1.0 or expected[-1]["upper"] != 1.0:
        raise AssertionError("calibration strata do not cover [-1,1]")
    for index, stratum in enumerate(expected):
        if (
            stratum["index"] != index
            or stratum["lower_operator"] != ">="
            or stratum["upper_operator"]
            != ("<=" if index == len(expected) - 1 else "<")
            or stratum["lower"] >= stratum["upper"]
            or index > 0
            and stratum["lower"] != expected[index - 1]["upper"]
        ):
            raise AssertionError("calibration strata are not a partition")
    return expected


def _calibration_sampling_strata_contract() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for score in _calibration_strata_contract():
        for relation in CALIBRATION_PAIR_RELATIONS:
            result.append(
                {
                    "index": len(result),
                    "name": f"{score['name']}__{relation}",
                    "score_stratum_index": score["index"],
                    "score_stratum": score["name"],
                    "pair_relation": relation,
                    "lower": score["lower"],
                    "lower_operator": score["lower_operator"],
                    "upper": score["upper"],
                    "upper_operator": score["upper_operator"],
                }
            )
    return result


def _validate_calibration_sampling_strata_contract(
    value: Any,
) -> list[dict[str, Any]]:
    expected = _calibration_sampling_strata_contract()
    if value != expected:
        raise ValueError("calibration sampling strata contract differs")
    return expected


def _calibration_histogram_contract() -> dict[str, Any]:
    return {
        "schema_version": CALIBRATION_HISTOGRAM_SCHEMA,
        "score_dtype": "float32",
        "lower": -1.0,
        "lower_operator": ">=",
        "upper": 1.0,
        "upper_operator": "<=",
        "bin_width": CALIBRATION_HISTOGRAM_BIN_WIDTH,
        "bin_count": CALIBRATION_HISTOGRAM_BIN_COUNT,
        "bin_semantics":
            "left-closed-right-open-except-final-right-closed",
        "relation_counts": ["all", *CALIBRATION_PAIR_RELATIONS],
    }


def _algorithm_contract(
    *,
    block_size: int,
    audit_top_k: int,
    calibration_per_stratum: int = DEFAULT_CALIBRATION_PER_STRATUM,
) -> dict[str, Any]:
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if type(audit_top_k) is not int or audit_top_k < 1:
        raise ValueError("audit_top_k must be a positive integer")
    if (
        type(calibration_per_stratum) is not int
        or calibration_per_stratum < 1
    ):
        raise ValueError(
            "calibration_per_stratum must be a positive integer"
        )
    if quotient_calibration.WORLD_SIZE != WORLD_SIZE:
        raise RuntimeError("DINO/quotient exact-eight world size differs")
    return {
        "version": ALGORITHM_VERSION,
        "metric": "l2-normalized-frame-dot-product",
        "pair_reduction": "maximum-over-ordered-6x6-frame-pairs",
        "argmax_tie_break": "lowest-frame-a-then-lowest-frame-b",
        "frames_per_asset": DINO_FRAMES,
        "embedding_dim": DINO_DIM,
        "compute_dtype": "float32",
        "hard_threshold": HARD_THRESHOLD,
        "hard_operator": ">=",
        "audit_lower_threshold": AUDIT_THRESHOLD,
        "audit_lower_operator": ">=",
        "audit_upper_threshold": HARD_THRESHOLD,
        "audit_upper_operator": "<",
        "audit_is_hard": False,
        "audit_top_k_per_asset": audit_top_k,
        "audit_top_k_order":
            "descending-exact-float32-cosine-then-stable-neighbor-identity"
            "-and-frames",
        "calibration_strata_version": CALIBRATION_STRATA_VERSION,
        "calibration_score_strata": _calibration_strata_contract(),
        "calibration_pair_relations": list(CALIBRATION_PAIR_RELATIONS),
        "calibration_sampling_strata":
            _calibration_sampling_strata_contract(),
        "calibration_samples_per_stratum": calibration_per_stratum,
        "calibration_sampling_method": CALIBRATION_BOTTOM_K_VERSION,
        "calibration_pair_id_version": CALIBRATION_PAIR_ID_VERSION,
        "calibration_hash": "sha256",
        "calibration_hash_input":
            "canonical-json(schema,seed,input_artifact_digest,"
            "pair_id_sha256)",
        "calibration_sampling_seed": CALIBRATION_SAMPLING_SEED,
        "calibration_priority_bits": CALIBRATION_PRIORITY_BITS,
        "calibration_local_candidate_rule":
            "each-rank-retains-bottom-k-per-stratum",
        "calibration_final_rule":
            "global-bottom-k-of-exact8-local-bottom-k-union",
        "calibration_histogram": _calibration_histogram_contract(),
        "calibration_statistical_unit": "asset_pair",
        "calibration_intended_use":
            "diagnostic_not_threshold_calibrating",
        "formal_threshold_statistical_unit_required":
            "pre-dino-base-component-pair",
        "calibration_multiple_comparison_adjusted": False,
        "thresholds_human_calibrated": False,
        "calibration_human_labels_asserted": False,
        "calibration_training_authorized": False,
        "quotient_accumulation_enabled": True,
        "quotient_accumulation_optional": False,
        "quotient_world_size": quotient_calibration.WORLD_SIZE,
        "quotient_consume_rule":
            "same-normalized-contiguous-candidate-block-before-any-"
            "downstream-score-processing",
        "quotient_rank_partial_schema":
            quotient_calibration.RANK_PARTIAL_SCHEMA,
        "quotient_rank_partial_directory": QUOTIENT_RANK_PARTIAL_NAME,
        "quotient_exact8_merge":
            "merge_exact8_rank_partials-v1",
        "quotient_iid_pair_maxima_schema":
            quotient_calibration.IID_PAIR_MAXIMA_SCHEMA,
        "quotient_iid_pair_maxima_directory": IID_PAIR_MAXIMA_NAME,
        "quotient_graph_commit_binding_schema":
            quotient_calibration.GRAPH_COMMIT_BINDING_SCHEMA,
        "quotient_partials_per_iid_pair": 2,
        "cosine_round_decimals": COSINE_ROUND_DECIMALS,
        "exact_float32_bits_recorded": True,
        "block_size": block_size,
        "partition": PARTITION_VERSION,
        "pair_owner": "smaller_asset_index_mod_world_size",
        "candidate_direction": "asset_b_greater_than_asset_a",
    }


def _implementation_provenance() -> dict[str, Any]:
    module = Path(__file__).resolve(strict=True)
    quotient_module = Path(
        quotient_calibration.__file__
    ).resolve(strict=True)
    return {
        "module": module.name,
        "module_sha256": _file_digest(module),
        "quotient_module": quotient_module.name,
        "quotient_module_sha256": _file_digest(quotient_module),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def _validate_runtime(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RUNTIME_FIELDS:
        raise ValueError("matcher runtime provenance fields differ")
    runtime = dict(value)
    for field in ("torch_version", "device_type", "device_name"):
        if type(runtime.get(field)) is not str or not runtime[field]:
            raise ValueError(f"matcher runtime {field} is invalid")
    for field in ("torch_cuda_version", "torch_hip_version"):
        if runtime.get(field) is not None and (
            type(runtime[field]) is not str or not runtime[field]
        ):
            raise ValueError(f"matcher runtime {field} is invalid")
    if type(runtime.get("tf32_allowed")) is not bool:
        raise ValueError("matcher runtime tf32 flag is invalid")
    return runtime


def _build_contract(
    graph_input: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    device: str,
    block_size: int,
    audit_top_k: int,
    calibration_per_stratum: int,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_rank(rank, world_size)
    if type(device) is not str or not device:
        raise ValueError("device must be a non-empty string")
    algorithm = _algorithm_contract(
        block_size=block_size,
        audit_top_k=audit_top_k,
        calibration_per_stratum=calibration_per_stratum,
    )
    dino_contract = dict(graph_input["dino_contract"])
    return {
        "schema_version": MATCHER_SCHEMA,
        "input_directory": str(graph_input["directory"]),
        "input_artifact_digest": graph_input["artifact_digest"],
        "input_artifacts": dict(graph_input["artifact_hashes"]),
        "input_rows": len(graph_input["rows"]),
        "dino_contract": dino_contract,
        "dino_contract_sha256": _object_digest(dino_contract),
        "algorithm": algorithm,
        "algorithm_sha256": _object_digest(algorithm),
        "implementation": _implementation_provenance(),
        "runtime": _validate_runtime(runtime),
        "rank": rank,
        "world_size": world_size,
        "device": device,
    }


def _validate_contract(value: Any, *, common: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("matcher contract is not a mapping")
    contract = dict(value)
    expected = set(_CONTRACT_FIELDS)
    if common:
        expected -= {"rank", "device"}
    if set(contract) != expected:
        raise ValueError("matcher contract field set differs")
    if (
        contract.get("schema_version") != MATCHER_SCHEMA
        or type(contract.get("input_directory")) is not str
        or not Path(contract["input_directory"]).is_absolute()
        or str(
            Path(contract["input_directory"]).resolve(strict=False)
        )
        != contract["input_directory"]
        or type(contract.get("input_rows")) is not int
        or contract["input_rows"] < 1
        or contract.get("world_size") != WORLD_SIZE
    ):
        raise ValueError("matcher contract scalar fields differ")
    _validate_sha(
        contract.get("input_artifact_digest"),
        "matcher input artifact digest",
    )
    artifacts = contract.get("input_artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != {"manifest", "archive", "summary", "done"}
    ):
        raise ValueError("matcher input artifact hashes differ")
    for name, digest in artifacts.items():
        _validate_sha(digest, f"matcher input {name} digest")
    dino_contract = _validate_dino_contract(
        contract.get("dino_contract")
    )
    if contract.get("dino_contract_sha256") != _object_digest(
        dino_contract
    ):
        raise ValueError("matcher DINO contract digest differs")
    algorithm = contract.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise ValueError("matcher algorithm is missing")
    block_size = algorithm.get("block_size")
    top_k = algorithm.get("audit_top_k_per_asset")
    calibration_per_stratum = algorithm.get(
        "calibration_samples_per_stratum"
    )
    expected_algorithm = _algorithm_contract(
        block_size=block_size,
        audit_top_k=top_k,
        calibration_per_stratum=calibration_per_stratum,
    )
    _validate_calibration_strata_contract(
        algorithm.get("calibration_score_strata")
    )
    _validate_calibration_sampling_strata_contract(
        algorithm.get("calibration_sampling_strata")
    )
    if (
        algorithm.get("calibration_pair_relations")
        != list(CALIBRATION_PAIR_RELATIONS)
        or algorithm.get("calibration_histogram")
        != _calibration_histogram_contract()
    ):
        raise ValueError("calibration relation/histogram contract differs")
    if dict(algorithm) != expected_algorithm:
        raise ValueError("matcher algorithm contract differs")
    if contract.get("algorithm_sha256") != _object_digest(
        expected_algorithm
    ):
        raise ValueError("matcher algorithm digest differs")
    implementation = contract.get("implementation")
    if (
        not isinstance(implementation, Mapping)
        or set(implementation)
        != {
            "module",
            "module_sha256",
            "quotient_module",
            "quotient_module_sha256",
            "python",
            "numpy",
        }
        or implementation.get("module") != Path(__file__).name
        or implementation.get("quotient_module")
        != Path(quotient_calibration.__file__).name
        or type(implementation.get("python")) is not str
        or not implementation["python"]
        or type(implementation.get("numpy")) is not str
        or not implementation["numpy"]
    ):
        raise ValueError("matcher implementation provenance differs")
    _validate_sha(
        implementation.get("module_sha256"),
        "matcher implementation digest",
    )
    _validate_sha(
        implementation.get("quotient_module_sha256"),
        "quotient implementation digest",
    )
    _validate_runtime(contract.get("runtime"))
    if not common:
        _validate_rank(contract.get("rank"), contract.get("world_size"))
        if type(contract.get("device")) is not str or not contract["device"]:
            raise ValueError("matcher device differs")
    return contract


def _common_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    common = dict(contract)
    common.pop("rank")
    common.pop("device")
    _validate_contract(common, common=True)
    return common


def _float32_bits(value: Any) -> str:
    scalar = np.float32(value)
    return struct.pack(">f", float(scalar)).hex()


def _float32_from_bits(value: Any) -> np.float32:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{8}", value) is None:
        raise ValueError("cosine_float32_hex is invalid")
    return np.float32(struct.unpack(">f", bytes.fromhex(value))[0])


def _rounded_cosine(value: np.float32) -> float:
    return round(float(value), COSINE_ROUND_DECIMALS)


def _edge_base(
    *,
    rows: Sequence[Mapping[str, Any]],
    asset_a: int,
    asset_b: int,
    score: np.float32,
    frame_a: int,
    frame_b: int,
    hard: bool,
) -> dict[str, Any]:
    first = rows[asset_a]
    second = rows[asset_b]
    return {
        "schema_version": (
            HARD_EDGE_SCHEMA if hard else AUDIT_EDGE_SCHEMA
        ),
        "edge_type": "hard_dino" if hard else "audit_calibration",
        "hard_edge": hard,
        "asset_a": asset_a,
        "asset_b": asset_b,
        "iid_a": first["iid"],
        "role_a": first["role"],
        "video_sha256_a": first["video_sha256"],
        "iid_b": second["iid"],
        "role_b": second["role"],
        "video_sha256_b": second["video_sha256"],
        "cosine": _rounded_cosine(score),
        "cosine_float32_hex": _float32_bits(score),
        "frame_a": frame_a,
        "frame_b": frame_b,
        "owner_rank": asset_a % WORLD_SIZE,
        "world_size": WORLD_SIZE,
    }


def _score_stratum_index(score: np.float32) -> int:
    scalar = np.float32(score)
    if (
        not np.isfinite(scalar)
        or scalar < np.float32(-1.0)
        or scalar > np.float32(1.0)
    ):
        raise ValueError("calibration score is outside [-1,1]")
    index = int(
        np.searchsorted(
            _CALIBRATION_CUTS_FLOAT32,
            scalar,
            side="right",
        )
    )
    if not 0 <= index < len(_CALIBRATION_STRATUM_SPECS):
        raise AssertionError("calibration stratum lookup failed")
    return index


def _sampling_stratum_index(
    *,
    score_index: int,
    relation: str,
) -> int:
    if (
        type(score_index) is not int
        or not 0 <= score_index < len(_CALIBRATION_STRATUM_SPECS)
        or relation not in CALIBRATION_PAIR_RELATIONS
    ):
        raise ValueError("calibration sampling stratum input differs")
    return (
        score_index * len(CALIBRATION_PAIR_RELATIONS)
        + CALIBRATION_PAIR_RELATIONS.index(relation)
    )


def _pair_identifier(asset_a: int, asset_b: int) -> tuple[str, str]:
    if (
        type(asset_a) is not int
        or type(asset_b) is not int
        or not 0 <= asset_a < asset_b
    ):
        raise ValueError("calibration asset pair is invalid")
    pair_id = f"{CALIBRATION_PAIR_ID_VERSION}:{asset_a}:{asset_b}"
    return pair_id, _sha256_bytes(pair_id.encode("utf-8"))


def _calibration_hash_priority(
    *,
    pair_id_sha256: str,
    population_digest: str,
) -> str:
    _validate_sha(pair_id_sha256, "calibration pair ID digest")
    _validate_sha(population_digest, "calibration population digest")
    return _object_digest(
        {
            "schema_version": CALIBRATION_BOTTOM_K_VERSION,
            "seed": CALIBRATION_SAMPLING_SEED,
            "input_artifact_digest": population_digest,
            "pair_id_sha256": pair_id_sha256,
        }
    )


def _calibration_core_edge(
    *,
    rows: Sequence[Mapping[str, Any]],
    asset_a: int,
    asset_b: int,
    score: np.float32,
    frame_a: int,
    frame_b: int,
    score_index: int,
    relation: str,
    population_digest: str,
) -> dict[str, Any]:
    score_contract = _calibration_strata_contract()[score_index]
    sampling_index = _sampling_stratum_index(
        score_index=score_index,
        relation=relation,
    )
    sampling_contract = _calibration_sampling_strata_contract()[
        sampling_index
    ]
    pair_id, pair_digest = _pair_identifier(asset_a, asset_b)
    priority = _calibration_hash_priority(
        pair_id_sha256=pair_digest,
        population_digest=population_digest,
    )
    edge = _edge_base(
        rows=rows,
        asset_a=asset_a,
        asset_b=asset_b,
        score=score,
        frame_a=frame_a,
        frame_b=frame_b,
        hard=bool(score >= np.float32(HARD_THRESHOLD)),
    )
    edge.update(
        {
            "schema_version": CALIBRATION_EDGE_SCHEMA,
            "edge_type": "asset_pair_statistical_diagnostic",
            "score_stratum": score_contract["name"],
            "score_stratum_index": score_index,
            "score_stratum_lower": score_contract["lower"],
            "score_stratum_lower_operator":
                score_contract["lower_operator"],
            "score_stratum_upper": score_contract["upper"],
            "score_stratum_upper_operator":
                score_contract["upper_operator"],
            "pair_relation": relation,
            "sampling_stratum": sampling_contract["name"],
            "sampling_stratum_index": sampling_index,
            "pair_id": pair_id,
            "pair_id_sha256": pair_digest,
            "bottom_k_key_sha256": priority,
            "hash_priority_sha256": priority,
        }
    )
    return edge


def _offer_calibration_pair(
    heaps: list[list[tuple[int, int, int, dict[str, Any]]]],
    *,
    rows: Sequence[Mapping[str, Any]],
    asset_a: int,
    asset_b: int,
    score: np.float32,
    frame_a: int,
    frame_b: int,
    score_index: int,
    relation: str,
    sample_size: int,
    population_digest: str,
) -> None:
    sampling_index = _sampling_stratum_index(
        score_index=score_index,
        relation=relation,
    )
    pair_id, pair_digest = _pair_identifier(asset_a, asset_b)
    del pair_id
    priority = _calibration_hash_priority(
        pair_id_sha256=pair_digest,
        population_digest=population_digest,
    )
    priority_integer = int(priority, 16)
    key = (priority_integer, asset_a, asset_b)
    heap = heaps[sampling_index]
    if len(heap) >= sample_size:
        current_worst = (
            -heap[0][0],
            -heap[0][1],
            -heap[0][2],
        )
        if key >= current_worst:
            return
    edge = _calibration_core_edge(
        rows=rows,
        asset_a=asset_a,
        asset_b=asset_b,
        score=score,
        frame_a=frame_a,
        frame_b=frame_b,
        score_index=score_index,
        relation=relation,
        population_digest=population_digest,
    )
    entry = (-priority_integer, -asset_a, -asset_b, edge)
    if len(heap) < sample_size:
        heapq.heappush(heap, entry)
    else:
        heapq.heapreplace(heap, entry)


_CALIBRATION_DYNAMIC_FIELDS = frozenset(
    {
        "sampling_method",
        "sampling_scope",
        "sample_rank_within_stratum",
        "stratum_population_count",
        "stratum_sample_size",
        "sampling_probability",
        "sampling_weight",
        "human_labels_asserted",
        "training_authorized",
    }
)


def _calibration_core_from_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in _CALIBRATION_DYNAMIC_FIELDS
    }


def _calibration_rows_and_metadata(
    candidates: Iterable[Mapping[str, Any]],
    *,
    population_counts: Sequence[int],
    samples_per_stratum: int,
    sampling_scope: str,
    population_digest: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata_contract = _calibration_sampling_strata_contract()
    if (
        sampling_scope not in {"rank_partition", "global_upper_triangle"}
        or type(samples_per_stratum) is not int
        or samples_per_stratum < 1
        or len(population_counts) != len(strata_contract)
        or any(type(value) is not int or value < 0 for value in population_counts)
    ):
        raise ValueError("calibration sampling metadata input differs")
    buckets: list[list[dict[str, Any]]] = [
        [] for _ in strata_contract
    ]
    seen_pairs: set[tuple[int, int]] = set()
    for raw in candidates:
        core = _calibration_core_from_row(raw)
        index = core.get("sampling_stratum_index")
        if type(index) is not int or not 0 <= index < len(buckets):
            raise ValueError("calibration candidate stratum differs")
        pair = _edge_pair(core)
        if pair in seen_pairs:
            raise ValueError(f"duplicate calibration candidate: {pair}")
        seen_pairs.add(pair)
        buckets[index].append(core)
    rows: list[dict[str, Any]] = []
    metadata_strata: list[dict[str, Any]] = []
    for index, (contract, bucket, population) in enumerate(
        zip(strata_contract, buckets, population_counts, strict=True)
    ):
        bucket.sort(
            key=lambda row: (
                row["hash_priority_sha256"],
                int(row["asset_a"]),
                int(row["asset_b"]),
            )
        )
        expected_sample = min(samples_per_stratum, population)
        if len(bucket) < expected_sample:
            raise ValueError(
                f"calibration stratum {index} lacks bottom-k candidates"
            )
        selected = bucket[:expected_sample]
        probability = (
            0.0 if population == 0 else expected_sample / population
        )
        weight: float | None = (
            None
            if expected_sample == 0
            else population / expected_sample
        )
        metadata_strata.append(
            {
                **contract,
                "N_h": population,
                "n_h": expected_sample,
                "sampling_probability": probability,
                "sampling_weight": weight,
            }
        )
        for rank, core in enumerate(selected, start=1):
            row = dict(core)
            row.update(
                {
                    "sampling_method": CALIBRATION_BOTTOM_K_VERSION,
                    "sampling_scope": sampling_scope,
                    "sample_rank_within_stratum": rank,
                    "stratum_population_count": population,
                    "stratum_sample_size": expected_sample,
                    "sampling_probability": probability,
                    "sampling_weight": weight,
                    "human_labels_asserted": False,
                    "training_authorized": False,
                }
            )
            rows.append(row)
    metadata = {
        "schema_version": CALIBRATION_METADATA_SCHEMA,
        "sampling_method": CALIBRATION_BOTTOM_K_VERSION,
        "sampling_scope": sampling_scope,
        "sampling_seed": CALIBRATION_SAMPLING_SEED,
        "priority_bits": CALIBRATION_PRIORITY_BITS,
        "input_artifact_digest": _validate_sha(
            population_digest,
            "calibration population digest",
        ),
        "hash_input":
            "canonical-json(schema,seed,input_artifact_digest,"
            "pair_id_sha256)",
        "statistical_unit": "asset_pair",
        "intended_use": "diagnostic_not_threshold_calibrating",
        "formal_threshold_statistical_unit_required":
            "pre-dino-base-component-pair",
        "multiple_comparison_adjusted": False,
        "thresholds_human_calibrated": False,
        "samples_per_stratum_requested": samples_per_stratum,
        "population_count": sum(population_counts),
        "sample_count": len(rows),
        "strata": metadata_strata,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    return rows, metadata


def _empty_histogram_counts() -> dict[str, np.ndarray]:
    return {
        name: np.zeros(
            CALIBRATION_HISTOGRAM_BIN_COUNT,
            dtype=np.int64,
        )
        for name in ("all", *CALIBRATION_PAIR_RELATIONS)
    }


def _histogram_bin_indices(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    indices = np.floor(
        (values.astype(np.float64) + 1.0)
        / CALIBRATION_HISTOGRAM_BIN_WIDTH
    ).astype(np.int64)
    return np.clip(
        indices,
        0,
        CALIBRATION_HISTOGRAM_BIN_COUNT - 1,
    )


def _histogram_metadata(
    counts: Mapping[str, Sequence[int] | np.ndarray],
) -> dict[str, Any]:
    expected_names = {"all", *CALIBRATION_PAIR_RELATIONS}
    if set(counts) != expected_names:
        raise ValueError("calibration histogram relation set differs")
    normalized: dict[str, list[int]] = {}
    for name in ("all", *CALIBRATION_PAIR_RELATIONS):
        values = np.asarray(counts[name])
        if (
            values.shape != (CALIBRATION_HISTOGRAM_BIN_COUNT,)
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
        ):
            raise ValueError("calibration histogram counts differ")
        normalized[name] = [int(value) for value in values.tolist()]
    if any(
        all_count != same_count + cross_count
        for all_count, same_count, cross_count in zip(
            normalized["all"],
            normalized["same_iid"],
            normalized["cross_iid"],
            strict=True,
        )
    ):
        raise ValueError("calibration histogram relation conservation fails")
    return {
        **_calibration_histogram_contract(),
        "counts": normalized,
        "population_count": sum(normalized["all"]),
    }


def _semantic_band_counts(
    population_counts: Sequence[int],
) -> dict[str, Any]:
    expected = len(
        _CALIBRATION_STRATUM_SPECS
    ) * len(CALIBRATION_PAIR_RELATIONS)
    if (
        len(population_counts) != expected
        or any(type(value) is not int or value < 0 for value in population_counts)
    ):
        raise ValueError("semantic band population counts differ")
    by_relation: dict[str, dict[str, int]] = {}
    for relation_index, relation in enumerate(
        CALIBRATION_PAIR_RELATIONS
    ):
        score_counts = [
            int(
                population_counts[
                    score_index * len(CALIBRATION_PAIR_RELATIONS)
                    + relation_index
                ]
            )
            for score_index in range(len(_CALIBRATION_STRATUM_SPECS))
        ]
        by_relation[relation] = {
            "below_0p92": sum(score_counts[:3]),
            "at_least_0p92_below_0p96": sum(score_counts[3:5]),
            "at_least_0p96": score_counts[5],
        }
    return {
        "boundary_semantics": {
            "below_0p92": "< float32(0.92)",
            "at_least_0p92_below_0p96":
                ">= float32(0.92) and < float32(0.96)",
            "at_least_0p96": ">= float32(0.96)",
        },
        "all": {
            name: sum(
                by_relation[relation][name]
                for relation in CALIBRATION_PAIR_RELATIONS
            )
            for name in (
                "below_0p92",
                "at_least_0p92_below_0p96",
                "at_least_0p96",
            )
        },
        **by_relation,
    }


def _edge_pair(edge: Mapping[str, Any]) -> tuple[int, int]:
    return int(edge["asset_a"]), int(edge["asset_b"])


def _audit_rank_key(
    edge: Mapping[str, Any],
    asset_index: int,
) -> tuple[Any, ...]:
    a, b = _edge_pair(edge)
    if asset_index == a:
        neighbor = b
        neighbor_identity = (
            edge["iid_b"],
            edge["role_b"],
            edge["video_sha256_b"],
        )
        own_frame = int(edge["frame_a"])
        neighbor_frame = int(edge["frame_b"])
    elif asset_index == b:
        neighbor = a
        neighbor_identity = (
            edge["iid_a"],
            edge["role_a"],
            edge["video_sha256_a"],
        )
        own_frame = int(edge["frame_b"])
        neighbor_frame = int(edge["frame_a"])
    else:
        raise ValueError("audit edge is not incident to requested asset")
    score = float(_float32_from_bits(edge["cosine_float32_hex"]))
    return (
        -score,
        *neighbor_identity,
        own_frame,
        neighbor_frame,
        # Asset identities are unique, so this is only a defensive final
        # ordering term and does not make semantic top-K input-order
        # dependent.
        neighbor,
        a,
        b,
    )


def _offer_local_audit(
    buckets: dict[int, list[dict[str, Any]]],
    *,
    asset_index: int,
    edge: dict[str, Any],
    top_k: int,
) -> None:
    bucket = buckets.setdefault(asset_index, [])
    bucket.append(edge)
    bucket.sort(key=lambda item: _audit_rank_key(item, asset_index))
    if len(bucket) > top_k:
        del bucket[top_k:]


def _normalise_matcher_output(
    result: Any,
    *,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError("block matcher must return scores/frame_a/frame_b")
    scores = np.asarray(result[0], dtype=np.float32)
    frames_a = np.asarray(result[1])
    frames_b = np.asarray(result[2])
    if (
        scores.shape != (count,)
        or frames_a.shape != (count,)
        or frames_b.shape != (count,)
        or not np.isfinite(scores).all()
        or np.any(scores < np.float32(-1.0001))
        or np.any(scores > np.float32(1.0001))
        or not np.issubdtype(frames_a.dtype, np.integer)
        or not np.issubdtype(frames_b.dtype, np.integer)
        or np.any(frames_a < 0)
        or np.any(frames_a >= DINO_FRAMES)
        or np.any(frames_b < 0)
        or np.any(frames_b >= DINO_FRAMES)
    ):
        raise RuntimeError("block matcher violated its output contract")
    return (
        np.clip(scores, -1.0, 1.0).astype(np.float32),
        frames_a.astype(np.int64),
        frames_b.astype(np.int64),
    )


class _TorchBlockMatcher:
    def __init__(
        self,
        local_rank: int,
        dino_cls: np.ndarray,
    ) -> None:
        try:
            import torch
        except Exception as error:
            raise RuntimeError("PyTorch is required for GPU matching") from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is unavailable")
        if type(local_rank) is not int or local_rank < 0:
            raise ValueError("local_rank must be a non-negative integer")
        torch.cuda.set_device(local_rank)
        if hasattr(torch.backends, "cuda") and hasattr(
            torch.backends.cuda, "matmul"
        ):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        self.torch = torch
        self.device = torch.device(f"cuda:{local_rank}")
        # The full normalized graph is small relative to an accelerator
        # (roughly 123 MiB for 6,700 assets) and is reused for every query.
        # Keeping it resident avoids quadratic host-to-device traffic.
        self.features = torch.from_numpy(
            np.ascontiguousarray(dino_cls)
        ).to(device=self.device, dtype=torch.float32)
        properties = torch.cuda.get_device_properties(self.device)
        self.device_label = str(self.device)
        self.runtime = {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": (
                None
                if getattr(torch.version, "cuda", None) is None
                else str(torch.version.cuda)
            ),
            "torch_hip_version": (
                None
                if getattr(torch.version, "hip", None) is None
                else str(torch.version.hip)
            ),
            "device_type": "cuda",
            "device_name": str(properties.name),
            "tf32_allowed": False,
        }

    def __call__(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        query_index: int,
        candidate_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del query, candidates
        torch = self.torch
        indices = np.asarray(candidate_indices, dtype=np.int64)
        if (
            indices.ndim != 1
            or not len(indices)
            or not np.array_equal(
                indices,
                np.arange(
                    int(indices[0]),
                    int(indices[-1]) + 1,
                    dtype=np.int64,
                ),
            )
        ):
            raise RuntimeError(
                "GPU matcher requires a non-empty contiguous asset block"
            )
        with torch.inference_mode():
            query_tensor = self.features[query_index]
            candidate_tensor = self.features[
                int(indices[0]) : int(indices[-1]) + 1
            ]
            count = int(candidate_tensor.shape[0])
            flat = candidate_tensor.reshape(
                count * DINO_FRAMES,
                DINO_DIM,
            )
            similarity = torch.matmul(query_tensor, flat.T)
            similarity = similarity.reshape(
                DINO_FRAMES,
                count,
                DINO_FRAMES,
            ).permute(1, 0, 2)
            maxima, argmax = similarity.reshape(count, -1).max(dim=1)
            frames_a = torch.div(
                argmax,
                DINO_FRAMES,
                rounding_mode="floor",
            )
            frames_b = argmax.remainder(DINO_FRAMES)
            return (
                maxima.cpu().numpy().astype(np.float32, copy=False),
                frames_a.cpu().numpy().astype(np.int64, copy=False),
                frames_b.cpu().numpy().astype(np.int64, copy=False),
            )


def numpy_block_matcher(
    query: np.ndarray,
    candidates: np.ndarray,
    query_index: int,
    candidate_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference CPU matcher used by tests and small offline verification."""

    del query_index, candidate_indices
    similarities = np.einsum(
        "fd,bgd->bfg",
        np.asarray(query, dtype=np.float32),
        np.asarray(candidates, dtype=np.float32),
        dtype=np.float32,
        optimize=False,
    )
    flattened = similarities.reshape(similarities.shape[0], -1)
    argmax = np.argmax(flattened, axis=1)
    maxima = flattened[np.arange(len(flattened)), argmax]
    return (
        maxima.astype(np.float32, copy=False),
        (argmax // DINO_FRAMES).astype(np.int64),
        (argmax % DINO_FRAMES).astype(np.int64),
    )


def match_rank_arrays(
    *,
    rows: Sequence[Mapping[str, Any]],
    dino_cls: np.ndarray,
    graph_binding: Mapping[str, Any],
    rank: int,
    world_size: int,
    block_size: int,
    audit_top_k: int,
    block_matcher: BlockMatcher,
    calibration_per_stratum: int = DEFAULT_CALIBRATION_PER_STRATUM,
    calibration_population_digest: str = "0" * 64,
) -> dict[str, Any]:
    """Compute one complete rank partition before any filesystem commit."""

    _validate_rank(rank, world_size)
    _algorithm_contract(
        block_size=block_size,
        audit_top_k=audit_top_k,
        calibration_per_stratum=calibration_per_stratum,
    )
    asset_rows = _validate_asset_rows(rows)
    quotient_accumulator = quotient_calibration.RankQuotientAccumulator(
        asset_rows,
        graph_binding=graph_binding,
        rank=rank,
        world_size=world_size,
    )
    arrays = _validate_feature_arrays(
        {
            "asset_indices": np.arange(len(asset_rows), dtype=np.int64),
            "dino_cls": np.asarray(dino_cls),
        },
        rows=len(asset_rows),
    )
    features = arrays["dino_cls"]
    _validate_sha(
        calibration_population_digest,
        "calibration population digest",
    )
    hard_edges: list[dict[str, Any]] = []
    audit_buckets: dict[int, list[dict[str, Any]]] = {}
    sampling_strata = _calibration_sampling_strata_contract()
    calibration_heaps: list[
        list[tuple[int, int, int, dict[str, Any]]]
    ] = [[] for _ in sampling_strata]
    calibration_population_counts = [0 for _ in sampling_strata]
    histogram_counts = _empty_histogram_counts()
    owned = list(range(rank, len(asset_rows), WORLD_SIZE))
    compared_pairs = 0
    hard_threshold = np.float32(HARD_THRESHOLD)
    audit_threshold = np.float32(AUDIT_THRESHOLD)
    started = time.monotonic()
    for owned_position, asset_a in enumerate(owned, start=1):
        query = features[asset_a]
        for begin in range(asset_a + 1, len(asset_rows), block_size):
            stop = min(begin + block_size, len(asset_rows))
            candidate_indices = np.arange(
                begin,
                stop,
                dtype=np.int64,
            )
            scores, frames_a, frames_b = _normalise_matcher_output(
                block_matcher(
                    query,
                    features[begin:stop],
                    asset_a,
                    candidate_indices,
                ),
                count=stop - begin,
            )
            quotient_accumulator.consume_block(
                asset_a=asset_a,
                candidate_indices=candidate_indices,
                scores=scores,
                frames_a=frames_a,
                frames_b=frames_b,
            )
            compared_pairs += stop - begin
            score_indices = np.searchsorted(
                _CALIBRATION_CUTS_FLOAT32,
                scores,
                side="right",
            ).astype(np.int64)
            same_iid = np.fromiter(
                (
                    asset_rows[asset_a]["iid"]
                    == asset_rows[int(asset_b)]["iid"]
                    for asset_b in candidate_indices
                ),
                dtype=np.bool_,
                count=len(candidate_indices),
            )
            sampling_indices = (
                score_indices * len(CALIBRATION_PAIR_RELATIONS)
                + np.where(same_iid, 0, 1)
            )
            block_population = np.bincount(
                sampling_indices,
                minlength=len(sampling_strata),
            )
            calibration_population_counts = [
                current + int(increment)
                for current, increment in zip(
                    calibration_population_counts,
                    block_population.tolist(),
                    strict=True,
                )
            ]
            histogram_indices = _histogram_bin_indices(scores)
            histogram_counts["all"] += np.bincount(
                histogram_indices,
                minlength=CALIBRATION_HISTOGRAM_BIN_COUNT,
            )
            for relation, mask in (
                ("same_iid", same_iid),
                ("cross_iid", ~same_iid),
            ):
                histogram_counts[relation] += np.bincount(
                    histogram_indices[mask],
                    minlength=CALIBRATION_HISTOGRAM_BIN_COUNT,
                )
            for offset in range(stop - begin):
                relation = (
                    "same_iid" if bool(same_iid[offset]) else "cross_iid"
                )
                _offer_calibration_pair(
                    calibration_heaps,
                    rows=asset_rows,
                    asset_a=asset_a,
                    asset_b=begin + offset,
                    score=scores[offset],
                    frame_a=int(frames_a[offset]),
                    frame_b=int(frames_b[offset]),
                    score_index=int(score_indices[offset]),
                    relation=relation,
                    sample_size=calibration_per_stratum,
                    population_digest=calibration_population_digest,
                )
            candidate_offsets = np.flatnonzero(
                scores >= audit_threshold
            )
            for offset_value in candidate_offsets.tolist():
                offset = int(offset_value)
                asset_b = begin + offset
                score = scores[offset]
                if score >= hard_threshold:
                    hard_edges.append(
                        _edge_base(
                            rows=asset_rows,
                            asset_a=asset_a,
                            asset_b=asset_b,
                            score=score,
                            frame_a=int(frames_a[offset]),
                            frame_b=int(frames_b[offset]),
                            hard=True,
                        )
                    )
                elif score >= audit_threshold:
                    edge = _edge_base(
                        rows=asset_rows,
                        asset_a=asset_a,
                        asset_b=asset_b,
                        score=score,
                        frame_a=int(frames_a[offset]),
                        frame_b=int(frames_b[offset]),
                        hard=False,
                    )
                    _offer_local_audit(
                        audit_buckets,
                        asset_index=asset_a,
                        edge=edge,
                        top_k=audit_top_k,
                    )
                    _offer_local_audit(
                        audit_buckets,
                        asset_index=asset_b,
                        edge=edge,
                        top_k=audit_top_k,
                    )
        if (
            owned_position % PROGRESS_OWNED_ASSET_INTERVAL == 0
            or owned_position == len(owned)
        ):
            progress = {
                "rank": rank,
                "completed_owned": owned_position,
                "total_owned": len(owned),
                "compared_pairs": compared_pairs,
                "elapsed_seconds": round(
                    time.monotonic() - started,
                    3,
                ),
            }
            print(
                "[r7-dino-edge-progress] "
                + _canonical_json(progress),
                flush=True,
            )
    expected_pairs = sum(len(asset_rows) - asset - 1 for asset in owned)
    if compared_pairs != expected_pairs:
        raise RuntimeError("rank pair coverage differs from partition")
    quotient_rank_partial = quotient_accumulator.finalize()
    selected: dict[tuple[int, int], set[int]] = {}
    audit_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for asset_index, bucket in audit_buckets.items():
        for edge in bucket:
            pair = _edge_pair(edge)
            audit_by_pair[pair] = edge
            selected.setdefault(pair, set()).add(asset_index)
    audit_edges: list[dict[str, Any]] = []
    for pair in sorted(audit_by_pair):
        edge = dict(audit_by_pair[pair])
        edge["selected_for_asset_indices"] = sorted(selected[pair])
        audit_edges.append(edge)
    hard_edges.sort(key=_edge_pair)
    calibration_candidates = [
        entry[3]
        for heap in calibration_heaps
        for entry in heap
    ]
    calibration_edges, calibration_sampling = (
        _calibration_rows_and_metadata(
            calibration_candidates,
            population_counts=calibration_population_counts,
            samples_per_stratum=calibration_per_stratum,
            sampling_scope="rank_partition",
            population_digest=calibration_population_digest,
        )
    )
    score_histogram = _histogram_metadata(histogram_counts)
    semantic_band_counts = _semantic_band_counts(
        calibration_population_counts
    )
    if (
        calibration_sampling["population_count"] != compared_pairs
        or score_histogram["population_count"] != compared_pairs
        or sum(semantic_band_counts["all"].values())
        != compared_pairs
    ):
        raise RuntimeError("calibration population conservation differs")
    return {
        "hard_edges": hard_edges,
        "audit_edges": audit_edges,
        "calibration_edges": calibration_edges,
        "calibration_sampling": calibration_sampling,
        "score_histogram": score_histogram,
        "semantic_band_counts": semantic_band_counts,
        "quotient_rank_partial": quotient_rank_partial,
        "owned_asset_indices": owned,
        "compared_pairs": compared_pairs,
        "expected_compared_pairs": expected_pairs,
        "audit_endpoint_selections": sum(
            len(edge["selected_for_asset_indices"])
            for edge in audit_edges
        ),
    }


def _quotient_artifact_binding(
    directory: Path,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        directory.name
        not in {QUOTIENT_RANK_PARTIAL_NAME, IID_PAIR_MAXIMA_NAME}
        or directory.is_symlink()
        or not directory.is_dir()
        or set(entry.name for entry in directory.iterdir())
        != _QUOTIENT_ARTIFACT_FILES
    ):
        raise ValueError("quotient artifact directory closure differs")
    files: dict[str, dict[str, Any]] = {}
    for name in sorted(_QUOTIENT_ARTIFACT_FILES):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"quotient artifact is not a regular file: {path}"
            )
        files[name] = {
            "sha256": _file_digest(path),
            "bytes": int(path.stat().st_size),
        }
    core = {
        "directory": directory.name,
        "artifact_schema": artifact["schema_version"],
        "artifact_digest": artifact["artifact_digest"],
        "contract_sha256": _object_digest(artifact["contract"]),
        "files": files,
    }
    return {
        **core,
        "binding_sha256": _object_digest(core),
    }


def _same_quotient_artifact_bytes(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Compare two validated in-memory artifacts without ndarray coercion."""

    envelope_fields = {
        "schema_version",
        "contract",
        "array_descriptors",
        "arrays",
        "artifact_digest",
    }
    if (
        set(left) != envelope_fields
        or set(right) != envelope_fields
        or any(
            left[field] != right[field]
            for field in envelope_fields - {"arrays"}
        )
        or not isinstance(left["arrays"], Mapping)
        or not isinstance(right["arrays"], Mapping)
        or set(left["arrays"]) != set(right["arrays"])
    ):
        return False
    for name in sorted(left["arrays"]):
        left_array = left["arrays"][name]
        right_array = right["arrays"][name]
        if (
            not isinstance(left_array, np.ndarray)
            or not isinstance(right_array, np.ndarray)
            or left_array.dtype != right_array.dtype
            or left_array.shape != right_array.shape
            or left_array.tobytes(order="C")
            != right_array.tobytes(order="C")
        ):
            return False
    return True


def _load_quotient_artifact(
    directory: Path,
    *,
    graph_rows: Sequence[Mapping[str, Any]],
    graph_binding: Mapping[str, Any],
    expected_schema: str,
    expected_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = quotient_calibration.load_artifact_directory(
        directory,
        graph_binding=graph_binding,
        rows=graph_rows,
    )
    if artifact.get("schema_version") != expected_schema:
        raise ValueError("quotient artifact schema differs")
    binding = _quotient_artifact_binding(directory, artifact)
    if expected_binding is not None and binding != dict(expected_binding):
        raise ValueError("quotient artifact binding differs")
    return artifact, binding


def _publish_output(
    directory: Path,
    *,
    hard_edges: Sequence[Mapping[str, Any]],
    audit_edges: Sequence[Mapping[str, Any]],
    calibration_edges: Sequence[Mapping[str, Any]],
    quotient_artifact: Mapping[str, Any],
    quotient_directory_name: str,
    graph_rows: Sequence[Mapping[str, Any]],
    graph_binding: Mapping[str, Any],
    summary_base: Mapping[str, Any],
    done_base: Mapping[str, Any],
) -> dict[str, Any]:
    hard_payload = _jsonl_bytes(hard_edges)
    audit_payload = _jsonl_bytes(audit_edges)
    calibration_payload = _jsonl_bytes(calibration_edges)

    def writer(staging: Path) -> dict[str, Any]:
        hard_path = staging / HARD_EDGES_NAME
        audit_path = staging / AUDIT_EDGES_NAME
        calibration_path = staging / CALIBRATION_EDGES_NAME
        _write_exclusive(hard_path, hard_payload)
        _write_exclusive(audit_path, audit_payload)
        _write_exclusive(calibration_path, calibration_payload)
        quotient_directory = staging / quotient_directory_name
        quotient_calibration.publish_artifact_directory(
            quotient_directory,
            quotient_artifact,
            graph_binding=graph_binding,
            rows=graph_rows,
        )
        loaded_quotient, quotient_binding = _load_quotient_artifact(
            quotient_directory,
            graph_rows=graph_rows,
            graph_binding=graph_binding,
            expected_schema=quotient_artifact["schema_version"],
        )
        if (
            loaded_quotient["artifact_digest"]
            != quotient_artifact["artifact_digest"]
        ):
            raise ValueError("published quotient logical digest differs")
        summary = dict(summary_base)
        summary["hard_edges_sha256"] = _file_digest(hard_path)
        summary["audit_edges_sha256"] = _file_digest(audit_path)
        summary["calibration_edges_sha256"] = _file_digest(
            calibration_path
        )
        summary["quotient_artifact"] = quotient_binding
        summary_path = staging / SUMMARY_NAME
        _write_exclusive(summary_path, _pretty_json_bytes(summary))
        artifacts = {
            "hard_edges": {
                "filename": HARD_EDGES_NAME,
                "sha256": _file_digest(hard_path),
            },
            "audit_edges": {
                "filename": AUDIT_EDGES_NAME,
                "sha256": _file_digest(audit_path),
            },
            "calibration_edges": {
                "filename": CALIBRATION_EDGES_NAME,
                "sha256": _file_digest(calibration_path),
            },
            "summary": {
                "filename": SUMMARY_NAME,
                "sha256": _file_digest(summary_path),
            },
        }
        done = dict(done_base)
        done["artifacts"] = artifacts
        done["quotient_artifact"] = quotient_binding
        done["permission_contract"] = (
            artifact_permissions.permission_contract()
        )
        _write_exclusive(
            staging / DONE_NAME,
            _pretty_json_bytes(done),
        )
        return done

    return _atomic_directory(directory, writer)


def _partition_proof(
    *,
    owned: Sequence[int],
    compared_pairs: int,
    expected_pairs: int,
) -> dict[str, Any]:
    return {
        "version": PARTITION_VERSION,
        "owner_rule": "asset_a % 8",
        "candidate_rule": "asset_b > asset_a",
        "owned_asset_indices_sha256": _object_digest(list(owned)),
        "observed_compared_pairs": compared_pairs,
        "expected_compared_pairs": expected_pairs,
        "complete": compared_pairs == expected_pairs,
    }


def extract_rank(
    *,
    input_directory: Path,
    output_root: Path,
    rank: int,
    world_size: int = WORLD_SIZE,
    local_rank: int = 0,
    block_size: int = DEFAULT_BLOCK_SIZE,
    audit_top_k: int = DEFAULT_AUDIT_TOP_K,
    calibration_per_stratum: int = DEFAULT_CALIBRATION_PER_STRATUM,
    resume: bool = False,
    block_matcher: BlockMatcher | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Compute and atomically publish one exact rank shard."""

    _validate_rank(rank, world_size)
    graph_input = validate_graph_input(input_directory)
    target = shard_directory(output_root, rank)
    if target.exists() or target.is_symlink():
        if not resume:
            raise FileExistsError(target)
        validated = validate_shard(
            target,
            input_directory=input_directory,
        )
        committed = validated["contract"]
        if (
            committed["rank"] != rank
            or committed["world_size"] != world_size
            or committed["algorithm"]
            != _algorithm_contract(
                block_size=block_size,
                audit_top_k=audit_top_k,
                calibration_per_stratum=calibration_per_stratum,
            )
            or committed["implementation"]
            != _implementation_provenance()
            or runtime_provenance is not None
            and committed["runtime"]
            != _validate_runtime(runtime_provenance)
            or device is not None
            and committed["device"] != device
        ):
            raise ValueError(
                "resume arguments/implementation differ from committed shard"
            )
        # Verification-only resume: do not initialize torch, allocate a GPU
        # tensor, or execute the injected matcher.
        return validated["done"]
    torch_matcher: _TorchBlockMatcher | None = None
    if block_matcher is None:
        torch_matcher = _TorchBlockMatcher(
            local_rank,
            graph_input["arrays"]["dino_cls"],
        )
        block_matcher = torch_matcher
        if runtime_provenance is not None or device is not None:
            raise ValueError(
                "runtime/device overrides are only valid for an injected "
                "test matcher"
            )
        runtime_provenance = torch_matcher.runtime
        device = torch_matcher.device_label
    else:
        if runtime_provenance is None:
            runtime_provenance = {
                "torch_version": "injected-test-matcher",
                "torch_cuda_version": None,
                "torch_hip_version": None,
                "device_type": "cpu-test",
                "device_name": "injected-block-matcher",
                "tf32_allowed": False,
            }
        if device is None:
            device = "cpu-test"
    contract = _build_contract(
        graph_input,
        rank=rank,
        world_size=world_size,
        device=device,
        block_size=block_size,
        audit_top_k=audit_top_k,
        calibration_per_stratum=calibration_per_stratum,
        runtime=runtime_provenance,
    )
    # All matching is deliberately completed before the staging directory is
    # created.  A global device/runtime failure therefore publishes nothing.
    result = match_rank_arrays(
        rows=graph_input["rows"],
        dino_cls=graph_input["arrays"]["dino_cls"],
        graph_binding=_graph_commit_binding(graph_input),
        rank=rank,
        world_size=world_size,
        block_size=block_size,
        audit_top_k=audit_top_k,
        block_matcher=block_matcher,
        calibration_per_stratum=calibration_per_stratum,
        calibration_population_digest=graph_input["artifact_digest"],
    )
    proof = _partition_proof(
        owned=result["owned_asset_indices"],
        compared_pairs=result["compared_pairs"],
        expected_pairs=result["expected_compared_pairs"],
    )
    if not proof["complete"]:
        raise RuntimeError("rank partition proof is incomplete")
    contract_sha = _object_digest(contract)
    summary = {
        "schema_version": SHARD_SUMMARY_SCHEMA,
        "status": "complete",
        "contract": contract,
        "contract_sha256": contract_sha,
        "rank": rank,
        "world_size": world_size,
        "input_rows": len(graph_input["rows"]),
        "owned_asset_indices": result["owned_asset_indices"],
        "compared_pairs": result["compared_pairs"],
        "expected_compared_pairs": result["expected_compared_pairs"],
        "hard_edges": len(result["hard_edges"]),
        "audit_edges": len(result["audit_edges"]),
        "calibration_edges": len(result["calibration_edges"]),
        "audit_endpoint_selections":
            result["audit_endpoint_selections"],
        "calibration_sampling": result["calibration_sampling"],
        "score_histogram": result["score_histogram"],
        "semantic_band_counts": result["semantic_band_counts"],
        "quotient_rank_partial_rows": len(
            result["quotient_rank_partial"]["arrays"]["score"]
        ),
        "partition_proof": proof,
    }
    done = {
        "schema_version": SHARD_DONE_SCHEMA,
        "status": "complete",
        "contract_sha256": contract_sha,
        "rank": rank,
        "world_size": world_size,
        "hard_edges": len(result["hard_edges"]),
        "audit_edges": len(result["audit_edges"]),
        "calibration_edges": len(result["calibration_edges"]),
        "quotient_rank_partial_rows": len(
            result["quotient_rank_partial"]["arrays"]["score"]
        ),
        "training_authorized": False,
    }
    published = _publish_output(
        target,
        hard_edges=result["hard_edges"],
        audit_edges=result["audit_edges"],
        calibration_edges=result["calibration_edges"],
        quotient_artifact=result["quotient_rank_partial"],
        quotient_directory_name=QUOTIENT_RANK_PARTIAL_NAME,
        graph_rows=graph_input["rows"],
        graph_binding=_graph_commit_binding(graph_input),
        summary_base=summary,
        done_base=done,
    )
    validate_shard(
        target,
        input_directory=input_directory,
        expected_contract=contract,
    )
    return published


def _validate_output_artifacts(
    directory: Path,
    *,
    done_schema: str,
) -> tuple[dict[str, Path], dict[str, Any], Path]:
    if done_schema == SHARD_DONE_SCHEMA:
        nested_name = QUOTIENT_RANK_PARTIAL_NAME
    elif done_schema == FINAL_DONE_SCHEMA:
        nested_name = IID_PAIR_MAXIMA_NAME
    else:
        raise ValueError("unsupported DINO edge output schema")
    paths, directories = _regular_exact_files(
        directory,
        _OUTPUT_FILE_NAMES,
        expected_directories=frozenset({nested_name}),
    )
    done = _load_json(paths[DONE_NAME])
    if (
        done.get("schema_version") != done_schema
        or done.get("status") != "complete"
    ):
        raise ValueError("DINO edge done marker differs")
    if "permission_contract" in done:
        artifact_permissions.validate_permission_contract(
            done["permission_contract"]
        )
        artifact_permissions.assert_sealed_tree(directory)
    artifacts = done.get("artifacts")
    registry = {
        "hard_edges": HARD_EDGES_NAME,
        "audit_edges": AUDIT_EDGES_NAME,
        "calibration_edges": CALIBRATION_EDGES_NAME,
        "summary": SUMMARY_NAME,
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        registry
    ):
        raise ValueError("DINO edge artifact registry differs")
    for name, filename in registry.items():
        record = artifacts.get(name)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"filename", "sha256"}
            or record.get("filename") != filename
            or record.get("sha256") != _file_digest(paths[filename])
        ):
            raise ValueError(f"DINO edge {name} digest mismatch")
    return paths, done, directories[nested_name]


def _validate_edge_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    graph_rows: Sequence[Mapping[str, Any]],
    hard: bool,
    expected_rank: int | None,
    audit_top_k: int,
) -> None:
    seen: set[tuple[int, int]] = set()
    prior_pair: tuple[int, int] | None = None
    endpoint_selections: dict[int, int] = {}
    hard_threshold = np.float32(HARD_THRESHOLD)
    audit_threshold = np.float32(AUDIT_THRESHOLD)
    expected_fields = (
        _HARD_EDGE_FIELDS if hard else _AUDIT_EDGE_FIELDS
    )
    for position, raw in enumerate(rows):
        if set(raw) != expected_fields:
            raise ValueError(f"edge row {position} field set differs")
        if (
            raw.get("schema_version")
            != (HARD_EDGE_SCHEMA if hard else AUDIT_EDGE_SCHEMA)
            or raw.get("edge_type")
            != ("hard_dino" if hard else "audit_calibration")
            or raw.get("hard_edge") is not hard
            or raw.get("world_size") != WORLD_SIZE
        ):
            raise ValueError(f"edge row {position} class fields differ")
        a = raw.get("asset_a")
        b = raw.get("asset_b")
        if (
            type(a) is not int
            or type(b) is not int
            or not 0 <= a < b < len(graph_rows)
        ):
            raise ValueError(f"edge row {position} pair is invalid")
        pair = (a, b)
        if pair in seen or (
            prior_pair is not None and pair <= prior_pair
        ):
            raise ValueError("edge rows are duplicate or not sorted")
        seen.add(pair)
        prior_pair = pair
        owner = a % WORLD_SIZE
        if (
            raw.get("owner_rank") != owner
            or expected_rank is not None
            and owner != expected_rank
        ):
            raise ValueError(f"edge row {position} owner differs")
        first = graph_rows[a]
        second = graph_rows[b]
        identity = {
            "iid_a": first["iid"],
            "role_a": first["role"],
            "video_sha256_a": first["video_sha256"],
            "iid_b": second["iid"],
            "role_b": second["role"],
            "video_sha256_b": second["video_sha256"],
        }
        if any(raw.get(name) != value for name, value in identity.items()):
            raise ValueError(f"edge row {position} identity differs")
        score = _float32_from_bits(raw.get("cosine_float32_hex"))
        if (
            not np.isfinite(score)
            or score < np.float32(-1.0)
            or score > np.float32(1.0)
            or type(raw.get("cosine")) not in {int, float}
            or float(raw["cosine"]) != _rounded_cosine(score)
        ):
            raise ValueError(f"edge row {position} cosine differs")
        if hard:
            if score < hard_threshold:
                raise ValueError("hard edge is below hard threshold")
        elif not audit_threshold <= score < hard_threshold:
            raise ValueError("audit edge is outside the non-hard band")
        for field in ("frame_a", "frame_b"):
            if (
                type(raw.get(field)) is not int
                or not 0 <= raw[field] < DINO_FRAMES
            ):
                raise ValueError(f"edge row {position} frame differs")
        if not hard:
            selected = raw.get("selected_for_asset_indices")
            if (
                not isinstance(selected, list)
                or not selected
                or selected != sorted(set(selected))
                or any(item not in pair for item in selected)
            ):
                raise ValueError(
                    f"audit edge row {position} selection differs"
                )
            for asset_index in selected:
                endpoint_selections[asset_index] = (
                    endpoint_selections.get(asset_index, 0) + 1
                )
    if any(count > audit_top_k for count in endpoint_selections.values()):
        raise ValueError("audit endpoint has more than top-K selections")


def _validate_calibration_edge_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    graph_rows: Sequence[Mapping[str, Any]],
    expected_rank: int | None,
    expected_scope: str,
    population_digest: str,
) -> None:
    prior_key: tuple[int, str, int, int] | None = None
    seen_pairs: set[tuple[int, int]] = set()
    strata = _calibration_sampling_strata_contract()
    for position, raw in enumerate(rows):
        if set(raw) != _CALIBRATION_EDGE_FIELDS:
            raise ValueError(
                f"calibration edge row {position} field set differs"
            )
        if (
            raw.get("schema_version") != CALIBRATION_EDGE_SCHEMA
            or raw.get("edge_type")
            != "asset_pair_statistical_diagnostic"
            or raw.get("sampling_method")
            != CALIBRATION_BOTTOM_K_VERSION
            or raw.get("sampling_scope") != expected_scope
            or raw.get("human_labels_asserted") is not False
            or raw.get("training_authorized") is not False
            or raw.get("world_size") != WORLD_SIZE
        ):
            raise ValueError(
                f"calibration edge row {position} class fields differ"
            )
        a = raw.get("asset_a")
        b = raw.get("asset_b")
        if (
            type(a) is not int
            or type(b) is not int
            or not 0 <= a < b < len(graph_rows)
        ):
            raise ValueError(
                f"calibration edge row {position} pair differs"
            )
        pair = (a, b)
        if pair in seen_pairs:
            raise ValueError("calibration edge pair is duplicated")
        seen_pairs.add(pair)
        owner = a % WORLD_SIZE
        if (
            raw.get("owner_rank") != owner
            or expected_rank is not None and owner != expected_rank
        ):
            raise ValueError(
                f"calibration edge row {position} owner differs"
            )
        first = graph_rows[a]
        second = graph_rows[b]
        identity = {
            "iid_a": first["iid"],
            "role_a": first["role"],
            "video_sha256_a": first["video_sha256"],
            "iid_b": second["iid"],
            "role_b": second["role"],
            "video_sha256_b": second["video_sha256"],
        }
        if any(raw.get(name) != value for name, value in identity.items()):
            raise ValueError(
                f"calibration edge row {position} identity differs"
            )
        relation = (
            "same_iid"
            if first["iid"] == second["iid"]
            else "cross_iid"
        )
        score = _float32_from_bits(raw.get("cosine_float32_hex"))
        if (
            not np.isfinite(score)
            or score < np.float32(-1.0)
            or score > np.float32(1.0)
            or type(raw.get("cosine")) not in {int, float}
            or float(raw["cosine"]) != _rounded_cosine(score)
            or raw.get("hard_edge")
            is not bool(score >= np.float32(HARD_THRESHOLD))
        ):
            raise ValueError(
                f"calibration edge row {position} score differs"
            )
        score_index = _score_stratum_index(score)
        sampling_index = _sampling_stratum_index(
            score_index=score_index,
            relation=relation,
        )
        contract = strata[sampling_index]
        repeated = {
            "score_stratum": contract["score_stratum"],
            "score_stratum_index": contract["score_stratum_index"],
            "score_stratum_lower": contract["lower"],
            "score_stratum_lower_operator":
                contract["lower_operator"],
            "score_stratum_upper": contract["upper"],
            "score_stratum_upper_operator":
                contract["upper_operator"],
            "pair_relation": contract["pair_relation"],
            "sampling_stratum": contract["name"],
            "sampling_stratum_index": contract["index"],
        }
        if any(raw.get(name) != value for name, value in repeated.items()):
            raise ValueError(
                f"calibration edge row {position} stratum differs"
            )
        pair_id, pair_digest = _pair_identifier(a, b)
        priority = _calibration_hash_priority(
            pair_id_sha256=pair_digest,
            population_digest=population_digest,
        )
        if (
            raw.get("pair_id") != pair_id
            or raw.get("pair_id_sha256") != pair_digest
            or raw.get("bottom_k_key_sha256") != priority
            or raw.get("hash_priority_sha256") != priority
        ):
            raise ValueError(
                f"calibration edge row {position} hash priority differs"
            )
        for field in ("frame_a", "frame_b"):
            if (
                type(raw.get(field)) is not int
                or not 0 <= raw[field] < DINO_FRAMES
            ):
                raise ValueError(
                    f"calibration edge row {position} frame differs"
                )
        population = raw.get("stratum_population_count")
        sample_size = raw.get("stratum_sample_size")
        rank = raw.get("sample_rank_within_stratum")
        if (
            type(population) is not int
            or population < 1
            or type(sample_size) is not int
            or not 1 <= sample_size <= population
            or type(rank) is not int
            or not 1 <= rank <= sample_size
            or raw.get("sampling_probability")
            != sample_size / population
            or raw.get("sampling_weight")
            != population / sample_size
        ):
            raise ValueError(
                f"calibration edge row {position} sampling weight differs"
            )
        order_key = (sampling_index, priority, a, b)
        if prior_key is not None and order_key <= prior_key:
            raise ValueError("calibration edge rows are not sorted")
        prior_key = order_key


def _validate_calibration_metadata(
    value: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    compared_pairs: int,
    samples_per_stratum: int,
    sampling_scope: str,
    population_digest: str,
) -> list[int]:
    if not isinstance(value, Mapping):
        raise ValueError("calibration sampling metadata is missing")
    metadata = dict(value)
    raw_strata = metadata.get("strata")
    contracts = _calibration_sampling_strata_contract()
    if (
        not isinstance(raw_strata, list)
        or len(raw_strata) != len(contracts)
    ):
        raise ValueError("calibration sampling metadata strata differ")
    population_counts: list[int] = []
    for index, (raw, contract) in enumerate(
        zip(raw_strata, contracts, strict=True)
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("calibration metadata stratum is not a mapping")
        expected_fields = {
            *contract,
            "N_h",
            "n_h",
            "sampling_probability",
            "sampling_weight",
        }
        if set(raw) != expected_fields or any(
            raw.get(name) != expected
            for name, expected in contract.items()
        ):
            raise ValueError(
                f"calibration metadata stratum {index} contract differs"
            )
        population = raw.get("N_h")
        sample_size = raw.get("n_h")
        if (
            type(population) is not int
            or population < 0
            or type(sample_size) is not int
            or sample_size != min(samples_per_stratum, population)
            or raw.get("sampling_probability")
            != (0.0 if population == 0 else sample_size / population)
            or raw.get("sampling_weight")
            != (
                None
                if sample_size == 0
                else population / sample_size
            )
        ):
            raise ValueError(
                f"calibration metadata stratum {index} counts differ"
            )
        population_counts.append(population)
    expected_rows, expected_metadata = _calibration_rows_and_metadata(
        rows,
        population_counts=population_counts,
        samples_per_stratum=samples_per_stratum,
        sampling_scope=sampling_scope,
        population_digest=population_digest,
    )
    if list(rows) != expected_rows or metadata != expected_metadata:
        raise ValueError("calibration bottom-k sample/metadata differs")
    if (
        metadata["population_count"] != compared_pairs
        or sum(population_counts) != compared_pairs
    ):
        raise ValueError("calibration population count differs")
    return population_counts


def _validate_score_histogram(
    value: Any,
    *,
    compared_pairs: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(
        value.get("counts"), Mapping
    ):
        raise ValueError("calibration score histogram is missing")
    expected = _histogram_metadata(value["counts"])
    if dict(value) != expected or expected["population_count"] != compared_pairs:
        raise ValueError("calibration score histogram differs")
    return expected


def validate_shard(
    directory: Path,
    *,
    input_directory: Path,
    expected_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly revalidate an immutable matcher shard."""

    graph_input = validate_graph_input(input_directory)
    paths, done, quotient_directory = _validate_output_artifacts(
        directory,
        done_schema=SHARD_DONE_SCHEMA,
    )
    expected_done_fields = {
        "schema_version",
        "status",
        "contract_sha256",
        "rank",
        "world_size",
        "hard_edges",
        "audit_edges",
        "calibration_edges",
        "quotient_rank_partial_rows",
        "training_authorized",
        "quotient_artifact",
        "artifacts",
    }
    observed_done_fields = set(done)
    if observed_done_fields not in (
        expected_done_fields,
        expected_done_fields | {"permission_contract"},
    ):
        raise ValueError("DINO edge shard done fields differ")
    summary = _load_json(paths[SUMMARY_NAME])
    expected_summary_fields = {
        "schema_version",
        "status",
        "contract",
        "contract_sha256",
        "rank",
        "world_size",
        "input_rows",
        "owned_asset_indices",
        "compared_pairs",
        "expected_compared_pairs",
        "hard_edges",
        "audit_edges",
        "calibration_edges",
        "audit_endpoint_selections",
        "calibration_sampling",
        "score_histogram",
        "semantic_band_counts",
        "quotient_rank_partial_rows",
        "quotient_artifact",
        "partition_proof",
        "hard_edges_sha256",
        "audit_edges_sha256",
        "calibration_edges_sha256",
    }
    if set(summary) != expected_summary_fields:
        raise ValueError("DINO edge shard summary fields differ")
    if (
        summary.get("schema_version") != SHARD_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
    ):
        raise ValueError("DINO edge shard summary differs")
    contract = _validate_contract(summary.get("contract"))
    contract_sha = _object_digest(contract)
    if (
        summary.get("contract_sha256") != contract_sha
        or done.get("contract_sha256") != contract_sha
    ):
        raise ValueError("DINO edge shard contract digest differs")
    if expected_contract is not None and contract != dict(
        expected_contract
    ):
        raise ValueError("resume contract differs from committed shard")
    if (
        contract["input_directory"] != str(graph_input["directory"])
        or contract["input_artifact_digest"]
        != graph_input["artifact_digest"]
        or contract["input_artifacts"]
        != graph_input["artifact_hashes"]
        or contract["input_rows"] != len(graph_input["rows"])
        or contract["dino_contract"] != graph_input["dino_contract"]
    ):
        raise ValueError("DINO edge shard input binding differs")
    rank, world_size = _validate_rank(
        contract["rank"],
        contract["world_size"],
    )
    quotient_rank_partial, quotient_binding = _load_quotient_artifact(
        quotient_directory,
        graph_rows=graph_input["rows"],
        graph_binding=_graph_commit_binding(graph_input),
        expected_schema=quotient_calibration.RANK_PARTIAL_SCHEMA,
        expected_binding=done.get("quotient_artifact"),
    )
    if (
        summary.get("quotient_artifact") != quotient_binding
        or quotient_rank_partial["contract"].get("rank") != rank
    ):
        raise ValueError("DINO edge shard quotient binding differs")
    top_k = contract["algorithm"]["audit_top_k_per_asset"]
    calibration_per_stratum = contract["algorithm"][
        "calibration_samples_per_stratum"
    ]
    hard_rows = _load_canonical_jsonl(
        paths[HARD_EDGES_NAME],
        allow_empty=True,
    )
    audit_rows = _load_canonical_jsonl(
        paths[AUDIT_EDGES_NAME],
        allow_empty=True,
    )
    calibration_rows = _load_canonical_jsonl(
        paths[CALIBRATION_EDGES_NAME],
        allow_empty=True,
    )
    _validate_edge_rows(
        hard_rows,
        graph_rows=graph_input["rows"],
        hard=True,
        expected_rank=rank,
        audit_top_k=top_k,
    )
    _validate_edge_rows(
        audit_rows,
        graph_rows=graph_input["rows"],
        hard=False,
        expected_rank=rank,
        audit_top_k=top_k,
    )
    _validate_calibration_edge_rows(
        calibration_rows,
        graph_rows=graph_input["rows"],
        expected_rank=rank,
        expected_scope="rank_partition",
        population_digest=graph_input["artifact_digest"],
    )
    hard_pairs = {_edge_pair(row) for row in hard_rows}
    audit_pairs = {_edge_pair(row) for row in audit_rows}
    if hard_pairs & audit_pairs:
        raise ValueError("hard and audit edge sets overlap")
    owned = list(range(rank, len(graph_input["rows"]), WORLD_SIZE))
    expected_pairs = sum(
        len(graph_input["rows"]) - asset - 1 for asset in owned
    )
    iids = len(graph_input["rows"]) // 2
    expected_quotient_partials = sum(
        iids - asset // 2 - 1 for asset in owned
    )
    endpoint_selections = sum(
        len(row["selected_for_asset_indices"]) for row in audit_rows
    )
    proof = _partition_proof(
        owned=owned,
        compared_pairs=expected_pairs,
        expected_pairs=expected_pairs,
    )
    population_counts = _validate_calibration_metadata(
        summary.get("calibration_sampling"),
        rows=calibration_rows,
        compared_pairs=expected_pairs,
        samples_per_stratum=calibration_per_stratum,
        sampling_scope="rank_partition",
        population_digest=graph_input["artifact_digest"],
    )
    _validate_score_histogram(
        summary.get("score_histogram"),
        compared_pairs=expected_pairs,
    )
    if summary.get("semantic_band_counts") != _semantic_band_counts(
        population_counts
    ):
        raise ValueError("calibration semantic band counts differ")
    if (
        summary.get("rank") != rank
        or summary.get("world_size") != world_size
        or summary.get("input_rows") != len(graph_input["rows"])
        or summary.get("owned_asset_indices") != owned
        or summary.get("compared_pairs") != expected_pairs
        or summary.get("expected_compared_pairs") != expected_pairs
        or summary.get("hard_edges") != len(hard_rows)
        or summary.get("audit_edges") != len(audit_rows)
        or summary.get("calibration_edges") != len(calibration_rows)
        or summary.get("quotient_rank_partial_rows")
        != expected_quotient_partials
        or summary.get("audit_endpoint_selections")
        != endpoint_selections
        or summary.get("partition_proof") != proof
        or summary.get("hard_edges_sha256")
        != _file_digest(paths[HARD_EDGES_NAME])
        or summary.get("audit_edges_sha256")
        != _file_digest(paths[AUDIT_EDGES_NAME])
        or summary.get("calibration_edges_sha256")
        != _file_digest(paths[CALIBRATION_EDGES_NAME])
        or done.get("rank") != rank
        or done.get("world_size") != world_size
        or done.get("hard_edges") != len(hard_rows)
        or done.get("audit_edges") != len(audit_rows)
        or done.get("calibration_edges") != len(calibration_rows)
        or done.get("quotient_rank_partial_rows")
        != expected_quotient_partials
        or len(quotient_rank_partial["arrays"]["score"])
        != expected_quotient_partials
        or done.get("training_authorized") is not False
    ):
        raise ValueError("DINO edge shard summary/count binding differs")
    return {
        "paths": paths,
        "done": done,
        "summary": summary,
        "contract": contract,
        "hard_edges": hard_rows,
        "audit_edges": audit_rows,
        "calibration_edges": calibration_rows,
        "quotient_rank_partial": quotient_rank_partial,
        "quotient_binding": quotient_binding,
    }


def _deduplicate_edges(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates = 0
    for raw in rows:
        row = dict(raw)
        pair = _edge_pair(row)
        prior = by_pair.get(pair)
        if prior is None:
            by_pair[pair] = row
        elif prior == row:
            duplicates += 1
        else:
            raise ValueError(f"conflicting duplicate edge: {pair}")
    return [by_pair[pair] for pair in sorted(by_pair)], duplicates


def _global_audit_top_k(
    candidates: Sequence[Mapping[str, Any]],
    *,
    assets: int,
    top_k: int,
) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {
        asset: [] for asset in range(assets)
    }
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in candidates:
        row = dict(raw)
        row.pop("selected_for_asset_indices", None)
        pair = _edge_pair(row)
        prior = by_pair.get(pair)
        if prior is not None and prior != row:
            raise ValueError(f"conflicting audit candidate edge: {pair}")
        by_pair[pair] = row
    for edge in by_pair.values():
        a, b = _edge_pair(edge)
        buckets[a].append(edge)
        buckets[b].append(edge)
    selected: dict[tuple[int, int], set[int]] = {}
    for asset_index, bucket in buckets.items():
        bucket.sort(key=lambda edge: _audit_rank_key(edge, asset_index))
        for edge in bucket[:top_k]:
            selected.setdefault(_edge_pair(edge), set()).add(asset_index)
    result: list[dict[str, Any]] = []
    for pair in sorted(selected):
        row = dict(by_pair[pair])
        row["selected_for_asset_indices"] = sorted(selected[pair])
        result.append(row)
    return result


def _expected_shard_names() -> set[str]:
    return {
        f"rank-{rank:05d}-of-{WORLD_SIZE:05d}"
        for rank in range(WORLD_SIZE)
    }


def _load_all_shards(
    *,
    input_directory: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    shard_root = output_root.expanduser() / "shards"
    if shard_root.is_symlink() or not shard_root.is_dir():
        raise FileNotFoundError(shard_root)
    actual = {entry.name for entry in shard_root.iterdir()}
    expected = _expected_shard_names()
    if actual != expected:
        raise ValueError(
            "exactly eight canonical shard directories are required: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    results = []
    for rank in range(WORLD_SIZE):
        results.append(
            validate_shard(
                shard_directory(output_root, rank),
                input_directory=input_directory,
            )
        )
    return results


def _merge_shards(
    *,
    graph_input: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(shards) != WORLD_SIZE:
        raise ValueError("exactly eight validated shards are required")
    common: dict[str, Any] | None = None
    seen_ranks: set[int] = set()
    owned: list[int] = []
    observed_pairs = 0
    hard_input: list[Mapping[str, Any]] = []
    audit_input: list[Mapping[str, Any]] = []
    calibration_input: list[Mapping[str, Any]] = []
    calibration_population_counts = [
        0 for _ in _calibration_sampling_strata_contract()
    ]
    histogram_counts = _empty_histogram_counts()
    quotient_rank_partials: list[Mapping[str, Any]] = []
    shard_bindings: list[dict[str, Any]] = []
    for shard in shards:
        contract = shard["contract"]
        rank = int(contract["rank"])
        if rank in seen_ranks:
            raise ValueError(f"duplicate shard rank: {rank}")
        seen_ranks.add(rank)
        shard_common = _common_contract(contract)
        if common is None:
            common = shard_common
        elif common != shard_common:
            raise ValueError("shard global contracts differ")
        owned.extend(shard["summary"]["owned_asset_indices"])
        observed_pairs += int(shard["summary"]["compared_pairs"])
        hard_input.extend(shard["hard_edges"])
        audit_input.extend(shard["audit_edges"])
        calibration_input.extend(shard["calibration_edges"])
        quotient_rank_partials.append(shard["quotient_rank_partial"])
        for index, stratum in enumerate(
            shard["summary"]["calibration_sampling"]["strata"]
        ):
            calibration_population_counts[index] += int(stratum["N_h"])
        shard_histogram = shard["summary"]["score_histogram"]["counts"]
        for name in histogram_counts:
            histogram_counts[name] += np.asarray(
                shard_histogram[name],
                dtype=np.int64,
            )
        shard_bindings.append(
            {
                "rank": rank,
                "done_sha256": _file_digest(
                    shard["paths"][DONE_NAME]
                ),
                "summary_sha256": _file_digest(
                    shard["paths"][SUMMARY_NAME]
                ),
                "hard_edges_sha256": _file_digest(
                    shard["paths"][HARD_EDGES_NAME]
                ),
                "audit_edges_sha256": _file_digest(
                    shard["paths"][AUDIT_EDGES_NAME]
                ),
                "calibration_edges_sha256": _file_digest(
                    shard["paths"][CALIBRATION_EDGES_NAME]
                ),
                "quotient_rank_partial":
                    shard["quotient_binding"],
                "contract_sha256": shard["summary"][
                    "contract_sha256"
                ],
            }
        )
    if common is None or seen_ranks != set(range(WORLD_SIZE)):
        raise ValueError("rank coverage is incomplete")
    assets = len(graph_input["rows"])
    expected_owned = list(range(assets))
    if sorted(owned) != expected_owned or len(owned) != assets:
        raise ValueError("asset ownership is not an exact partition")
    expected_pairs = assets * (assets - 1) // 2
    if observed_pairs != expected_pairs:
        raise ValueError("global upper-triangle coverage is incomplete")
    iid_pair_maxima = quotient_calibration.merge_exact8_rank_partials(
        graph_input["rows"],
        quotient_rank_partials,
        graph_binding=_graph_commit_binding(graph_input),
    )
    expected_iid_pairs = (assets // 2) * (assets // 2 - 1) // 2
    if (
        len(iid_pair_maxima["arrays"]["score"]) != expected_iid_pairs
        or iid_pair_maxima["contract"].get("expected_iid_pairs")
        != expected_iid_pairs
        or iid_pair_maxima["contract"].get("observed_iid_pairs")
        != expected_iid_pairs
        or iid_pair_maxima["contract"].get("partials_per_iid_pair") != 2
        or iid_pair_maxima["contract"].get("observed_partials")
        != 2 * expected_iid_pairs
    ):
        raise ValueError("global quotient IID-pair coverage differs")
    hard_edges, hard_duplicates = _deduplicate_edges(hard_input)
    audit_candidates, audit_duplicates = _deduplicate_edges(audit_input)
    hard_pairs = {_edge_pair(row) for row in hard_edges}
    if hard_pairs & {_edge_pair(row) for row in audit_candidates}:
        raise ValueError("hard edges overlap audit candidates")
    top_k = common["algorithm"]["audit_top_k_per_asset"]
    audit_edges = _global_audit_top_k(
        audit_candidates,
        assets=assets,
        top_k=top_k,
    )
    calibration_edges, calibration_sampling = (
        _calibration_rows_and_metadata(
            calibration_input,
            population_counts=calibration_population_counts,
            samples_per_stratum=common["algorithm"][
                "calibration_samples_per_stratum"
            ],
            sampling_scope="global_upper_triangle",
            population_digest=graph_input["artifact_digest"],
        )
    )
    score_histogram = _histogram_metadata(histogram_counts)
    semantic_band_counts = _semantic_band_counts(
        calibration_population_counts
    )
    if (
        calibration_sampling["population_count"] != expected_pairs
        or score_histogram["population_count"] != expected_pairs
        or sum(semantic_band_counts["all"].values()) != expected_pairs
    ):
        raise ValueError("global calibration population differs")
    shard_bindings.sort(key=lambda item: item["rank"])
    coverage = {
        "partition": PARTITION_VERSION,
        "ranks": list(range(WORLD_SIZE)),
        "rank_count": WORLD_SIZE,
        "asset_count": assets,
        "owned_asset_indices_sha256": _object_digest(expected_owned),
        "observed_compared_pairs": observed_pairs,
        "expected_compared_pairs": expected_pairs,
        "complete_upper_triangle": observed_pairs == expected_pairs,
        "hard_duplicate_records_removed": hard_duplicates,
        "audit_duplicate_records_removed": audit_duplicates,
        "audit_local_candidate_edges": len(audit_candidates),
        "audit_global_top_k_edges": len(audit_edges),
        "audit_edges_are_nonhard": True,
        "calibration_statistical_unit": "asset_pair",
        "calibration_intended_use":
            "diagnostic_not_threshold_calibrating",
        "calibration_local_candidate_edges": len(calibration_input),
        "calibration_global_bottom_k_edges": len(calibration_edges),
        "calibration_population_complete": True,
        "thresholds_human_calibrated": False,
        "training_authorized": False,
        "iid_count": assets // 2,
        "expected_iid_pairs": expected_iid_pairs,
        "observed_iid_pairs": expected_iid_pairs,
        "quotient_partials_per_iid_pair": 2,
        "quotient_observed_partials": 2 * expected_iid_pairs,
        "quotient_coverage_complete": True,
    }
    return {
        "common_contract": common,
        "hard_edges": hard_edges,
        "audit_edges": audit_edges,
        "calibration_edges": calibration_edges,
        "calibration_sampling": calibration_sampling,
        "score_histogram": score_histogram,
        "semantic_band_counts": semantic_band_counts,
        "iid_pair_maxima": iid_pair_maxima,
        "coverage": coverage,
        "shards": shard_bindings,
    }


def finalize_shards(
    *,
    input_directory: Path,
    output_root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate exactly eight shards and atomically publish the final edge set."""

    graph_input = validate_graph_input(input_directory)
    shards = _load_all_shards(
        input_directory=input_directory,
        output_root=output_root,
    )
    merged = _merge_shards(graph_input=graph_input, shards=shards)
    target = output_root.expanduser() / "final"
    if target.exists() or target.is_symlink():
        if not resume:
            raise FileExistsError(target)
        return validate_final(
            target,
            input_directory=input_directory,
            output_root=output_root,
        )["done"]
    common_sha = _object_digest(merged["common_contract"])
    endpoint_selections = sum(
        len(row["selected_for_asset_indices"])
        for row in merged["audit_edges"]
    )
    summary = {
        "schema_version": FINAL_SUMMARY_SCHEMA,
        "status": "complete",
        "contract": merged["common_contract"],
        "contract_sha256": common_sha,
        "input_rows": len(graph_input["rows"]),
        "rank_count": WORLD_SIZE,
        "hard_edges": len(merged["hard_edges"]),
        "audit_edges": len(merged["audit_edges"]),
        "calibration_edges": len(merged["calibration_edges"]),
        "audit_endpoint_selections": endpoint_selections,
        "calibration_sampling": merged["calibration_sampling"],
        "score_histogram": merged["score_histogram"],
        "semantic_band_counts": merged["semantic_band_counts"],
        "iid_pair_maxima_rows": len(
            merged["iid_pair_maxima"]["arrays"]["score"]
        ),
        "coverage_proof": merged["coverage"],
        "shards": merged["shards"],
    }
    done = {
        "schema_version": FINAL_DONE_SCHEMA,
        "status": "complete",
        "contract_sha256": common_sha,
        "hard_edges": len(merged["hard_edges"]),
        "audit_edges": len(merged["audit_edges"]),
        "calibration_edges": len(merged["calibration_edges"]),
        "iid_pair_maxima_rows": len(
            merged["iid_pair_maxima"]["arrays"]["score"]
        ),
        "audit_edges_are_nonhard": True,
        "calibration_intended_use":
            "diagnostic_not_threshold_calibrating",
        "thresholds_human_calibrated": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    published = _publish_output(
        target,
        hard_edges=merged["hard_edges"],
        audit_edges=merged["audit_edges"],
        calibration_edges=merged["calibration_edges"],
        quotient_artifact=merged["iid_pair_maxima"],
        quotient_directory_name=IID_PAIR_MAXIMA_NAME,
        graph_rows=graph_input["rows"],
        graph_binding=_graph_commit_binding(graph_input),
        summary_base=summary,
        done_base=done,
    )
    validate_final(
        target,
        input_directory=input_directory,
        output_root=output_root,
    )
    return published


def validate_final(
    directory: Path,
    *,
    input_directory: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Revalidate final artifacts against the eight immutable shards."""

    graph_input = validate_graph_input(input_directory)
    unresolved = directory.expanduser()
    if unresolved.is_symlink():
        raise ValueError("final edge directory must not be a symlink")
    final_directory = unresolved.resolve(strict=True)
    if output_root is None:
        output_root = final_directory.parent
    shards = _load_all_shards(
        input_directory=input_directory,
        output_root=output_root,
    )
    merged = _merge_shards(graph_input=graph_input, shards=shards)
    paths, done, quotient_directory = _validate_output_artifacts(
        final_directory,
        done_schema=FINAL_DONE_SCHEMA,
    )
    expected_done_fields = {
        "schema_version",
        "status",
        "contract_sha256",
        "hard_edges",
        "audit_edges",
        "calibration_edges",
        "iid_pair_maxima_rows",
        "audit_edges_are_nonhard",
        "calibration_intended_use",
        "thresholds_human_calibrated",
        "human_labels_asserted",
        "training_authorized",
        "quotient_artifact",
        "artifacts",
    }
    observed_done_fields = set(done)
    if observed_done_fields not in (
        expected_done_fields,
        expected_done_fields | {"permission_contract"},
    ):
        raise ValueError("DINO edge final done fields differ")
    summary = _load_json(paths[SUMMARY_NAME])
    expected_summary_fields = {
        "schema_version",
        "status",
        "contract",
        "contract_sha256",
        "input_rows",
        "rank_count",
        "hard_edges",
        "audit_edges",
        "calibration_edges",
        "audit_endpoint_selections",
        "calibration_sampling",
        "score_histogram",
        "semantic_band_counts",
        "iid_pair_maxima_rows",
        "quotient_artifact",
        "coverage_proof",
        "shards",
        "hard_edges_sha256",
        "audit_edges_sha256",
        "calibration_edges_sha256",
    }
    if set(summary) != expected_summary_fields:
        raise ValueError("DINO edge final summary fields differ")
    if (
        summary.get("schema_version") != FINAL_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
    ):
        raise ValueError("DINO edge final summary differs")
    contract = _validate_contract(summary.get("contract"), common=True)
    contract_sha = _object_digest(contract)
    if (
        contract != merged["common_contract"]
        or summary.get("contract_sha256") != contract_sha
        or done.get("contract_sha256") != contract_sha
    ):
        raise ValueError("DINO edge final contract differs")
    iid_pair_maxima, quotient_binding = _load_quotient_artifact(
        quotient_directory,
        graph_rows=graph_input["rows"],
        graph_binding=_graph_commit_binding(graph_input),
        expected_schema=quotient_calibration.IID_PAIR_MAXIMA_SCHEMA,
        expected_binding=done.get("quotient_artifact"),
    )
    if (
        summary.get("quotient_artifact") != quotient_binding
        or not _same_quotient_artifact_bytes(
            iid_pair_maxima,
            merged["iid_pair_maxima"],
        )
    ):
        raise ValueError("DINO edge final quotient binding differs")
    hard_rows = _load_canonical_jsonl(
        paths[HARD_EDGES_NAME],
        allow_empty=True,
    )
    audit_rows = _load_canonical_jsonl(
        paths[AUDIT_EDGES_NAME],
        allow_empty=True,
    )
    calibration_rows = _load_canonical_jsonl(
        paths[CALIBRATION_EDGES_NAME],
        allow_empty=True,
    )
    top_k = contract["algorithm"]["audit_top_k_per_asset"]
    _validate_edge_rows(
        hard_rows,
        graph_rows=graph_input["rows"],
        hard=True,
        expected_rank=None,
        audit_top_k=top_k,
    )
    _validate_edge_rows(
        audit_rows,
        graph_rows=graph_input["rows"],
        hard=False,
        expected_rank=None,
        audit_top_k=top_k,
    )
    _validate_calibration_edge_rows(
        calibration_rows,
        graph_rows=graph_input["rows"],
        expected_rank=None,
        expected_scope="global_upper_triangle",
        population_digest=graph_input["artifact_digest"],
    )
    if hard_rows != merged["hard_edges"]:
        raise ValueError("final hard edges differ from shard merge")
    if audit_rows != merged["audit_edges"]:
        raise ValueError("final audit edges differ from global top-K merge")
    if calibration_rows != merged["calibration_edges"]:
        raise ValueError(
            "final calibration edges differ from global bottom-k merge"
        )
    endpoint_selections = sum(
        len(row["selected_for_asset_indices"]) for row in audit_rows
    )
    if (
        summary.get("input_rows") != len(graph_input["rows"])
        or summary.get("rank_count") != WORLD_SIZE
        or summary.get("hard_edges") != len(hard_rows)
        or summary.get("audit_edges") != len(audit_rows)
        or summary.get("calibration_edges") != len(calibration_rows)
        or summary.get("iid_pair_maxima_rows")
        != len(merged["iid_pair_maxima"]["arrays"]["score"])
        or summary.get("audit_endpoint_selections")
        != endpoint_selections
        or summary.get("calibration_sampling")
        != merged["calibration_sampling"]
        or summary.get("score_histogram")
        != merged["score_histogram"]
        or summary.get("semantic_band_counts")
        != merged["semantic_band_counts"]
        or summary.get("coverage_proof") != merged["coverage"]
        or summary.get("shards") != merged["shards"]
        or summary.get("hard_edges_sha256")
        != _file_digest(paths[HARD_EDGES_NAME])
        or summary.get("audit_edges_sha256")
        != _file_digest(paths[AUDIT_EDGES_NAME])
        or summary.get("calibration_edges_sha256")
        != _file_digest(paths[CALIBRATION_EDGES_NAME])
        or done.get("hard_edges") != len(hard_rows)
        or done.get("audit_edges") != len(audit_rows)
        or done.get("calibration_edges") != len(calibration_rows)
        or done.get("iid_pair_maxima_rows")
        != len(merged["iid_pair_maxima"]["arrays"]["score"])
        or len(iid_pair_maxima["arrays"]["score"])
        != len(merged["iid_pair_maxima"]["arrays"]["score"])
        or done.get("audit_edges_are_nonhard") is not True
        or done.get("calibration_intended_use")
        != "diagnostic_not_threshold_calibrating"
        or done.get("thresholds_human_calibrated") is not False
        or done.get("human_labels_asserted") is not False
        or done.get("training_authorized") is not False
    ):
        raise ValueError("DINO edge final summary/count binding differs")
    return {
        "paths": paths,
        "done": done,
        "summary": summary,
        "contract": contract,
        "hard_edges": hard_rows,
        "audit_edges": audit_rows,
        "calibration_edges": calibration_rows,
        "iid_pair_maxima": iid_pair_maxima,
        "quotient_binding": quotient_binding,
    }


def _torchrun_coordinates() -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "extract requires torchrun RANK/WORLD_SIZE/LOCAL_RANK"
        ) from error
    _validate_rank(rank, world_size)
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be non-negative")
    return rank, world_size, local_rank


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact eight-rank DINO edge matcher",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--input-directory", type=Path, required=True)
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
    )
    extract.add_argument(
        "--audit-top-k",
        type=int,
        default=DEFAULT_AUDIT_TOP_K,
    )
    extract.add_argument(
        "--calibration-per-stratum",
        type=int,
        default=DEFAULT_CALIBRATION_PER_STRATUM,
    )
    extract.add_argument("--resume", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--input-directory", type=Path, required=True)
    finalize.add_argument("--output-root", type=Path, required=True)
    finalize.add_argument("--resume", action="store_true")
    validate_input = subparsers.add_parser("validate-input")
    validate_input.add_argument(
        "--input-directory",
        type=Path,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "extract":
        rank, world_size, local_rank = _torchrun_coordinates()
        result = extract_rank(
            input_directory=args.input_directory,
            output_root=args.output_root,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            block_size=args.block_size,
            audit_top_k=args.audit_top_k,
            calibration_per_stratum=args.calibration_per_stratum,
            resume=args.resume,
        )
    elif args.command == "finalize":
        result = finalize_shards(
            input_directory=args.input_directory,
            output_root=args.output_root,
            resume=args.resume,
        )
    else:
        validated = validate_graph_input(args.input_directory)
        result = {
            "status": "complete",
            "input_artifact_digest": validated["artifact_digest"],
            "rows": len(validated["rows"]),
        }
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
