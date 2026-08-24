"""Blind, provenance-bound review tooling for R7 visual-neighbour audits.

The source manifest contains graph evidence needed by a later statistical
report.  None of that evidence is copied into reviewer-facing templates.
Templates contain only opaque assignment IDs, two complete byte-bound videos,
the frozen question, and label fields.  Primary labels are never an input to
secondary template preparation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .r7_neighbor_audit_policy import (
    COHORT_DOUBLE_REVIEW_TARGETS,
    COHORT_PRIMARY_TARGETS,
    LABEL_COMMIT_SCHEMA,
    MERGED_REVIEW_SCHEMA,
    POPULATION_COMMIT_SCHEMA,
    POPULATION_CONTEXT_SCHEMA,
    POPULATION_ROW_SCHEMA,
    REASON_CODES,
    REVIEW_TEMPLATE_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    VERDICTS,
    canonical_json,
    policy_payload,
    policy_sha256,
)


REVIEW_NAME = "review.jsonl"
REVIEW_BUNDLE_DONE_NAME = "review_bundle.done.json"
POPULATION_NAME = "population.jsonl"
POPULATION_CONTEXT_NAME = "population_context.json"
POPULATION_DONE_NAME = "population.done.json"
LABELS_NAME = "labels.jsonl"
LABELS_DONE_NAME = "labels.done.json"
MEDIA_BINDING_SCHEMA = "motive-r7-neighbor-opaque-media-binding-v1"
REVIEW_BUNDLE_SCHEMA = "motive-r7-neighbor-review-bundle-v1"
STATISTICAL_UNIT_SCHEMA = "motive-r7-base-component-pair-v1"

QUESTION = (
    "After watching both complete videos, must these two examples be kept "
    "in the same dataset split to prevent visual/content leakage?"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVIEWER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._@+-]{0,127}$")
_RFC3339_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)
_SAFE_COMPONENT_RE = re.compile(r"^[^\x00\r\n]{1,256}$")
_SAFE_BIN_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,127}$")
_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})

_POPULATION_FIELDS = frozenset(
    {
        "schema_version",
        "policy_sha256",
        "thresholds_human_calibrated",
        "statistical_unit_id",
        "base_component_pair",
        "source_bindings",
        "hidden_context",
        "witness",
        "media",
    }
)
_POPULATION_CONTEXT_FIELDS = frozenset(
    {
        "schema_version",
        "policy_sha256",
        "thresholds_human_calibrated",
        "population_rows",
        "population_sha256",
        "canonical_order",
        "canonical_encoding",
        "statistical_unit",
        "cohort_precedence",
        "cohort_population_counts",
        "source_bindings",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "policy_sha256",
        "thresholds_human_calibrated",
        "statistical_unit_id",
        "cohort",
        "cohort_rank",
        "sample_order",
        "selection_sha256",
        "population_sha256",
        "population_row_sha256",
        "sampling_design",
        "base_component_pair",
        "source_bindings",
        "hidden_context",
        "witness",
        "media",
    }
)
_SAMPLING_DESIGN_FIELDS = frozenset(
    {
        "design",
        "population_size_N_h",
        "sample_size_n_h",
        "inclusion_probability_pi_h",
        "design_weight",
        "selection_rule",
    }
)
_SOURCE_BINDING_FIELDS = frozenset(
    {
        "indexed_graph_artifact_digest",
        "dino_edges_artifact_digest",
        "sampling_population_sha256",
        "validated_quotient_artifact_digest",
        "base_component_population_sha256",
    }
)
_HIDDEN_CONTEXT_FIELDS = frozenset(
    {
        "score",
        "score_bin",
        "threshold_relation",
        "anchor_flags",
        "qwen_evidence_sha256",
        "iid_pair",
        "provisional_split_pair",
    }
)
_WITNESS_FIELDS = frozenset(
    {
        "high_impact",
        "hard_edge",
        "top_neighbor",
        "msf_witness",
        "cross_component",
        "cross_split",
        "priority",
        "top_merge_witness",
        "large_component_witness",
    }
)
_SOURCE_MEDIA_FIELDS = frozenset(
    {
        "relative_path",
        "sha256",
        "size_bytes",
        "complete_video",
    }
)
_TEMPLATE_FIELDS = frozenset(
    {
        "schema_version",
        "policy_sha256",
        "review_instance_id",
        "assignment_set_digest",
        "assigned_reviewer_id",
        "question",
        "media",
        "allowed_verdicts",
        "allowed_reason_codes",
        "label_scope",
        "thresholds_human_calibrated",
        "training_authorized",
        "direct_training_supervision_allowed",
        "verdict",
        "reason_codes",
        "notes",
        "completed_at_utc",
        "review_attestation",
    }
)
_OPAQUE_MEDIA_FIELDS = frozenset(
    {
        "schema_version",
        "opaque_media_id",
        "relative_path",
        "sha256",
        "size_bytes",
        "complete_video",
        "whole_file_byte_copy_verified",
    }
)
_ATTESTATION_FIELDS = frozenset(
    {
        "video_1_reviewed_in_full",
        "video_2_reviewed_in_full",
        "independent_judgment",
        "other_reviewer_result_not_observed",
    }
)
_IMMUTABLE_TEMPLATE_FIELDS = _TEMPLATE_FIELDS - {
    "verdict",
    "reason_codes",
    "notes",
    "completed_at_utc",
    "review_attestation",
}

_MUST_SAME_SPLIT_REASONS = frozenset(
    {
        "same_clip_or_transcode",
        "temporal_overlap",
        "same_generation_lineage",
        "same_scene_different_action_edit",
    }
)
_INDEPENDENT_CONTENT_REASONS = frozenset(
    {
        "same_subject_background_only",
        "same_action_only",
        "common_overlay_or_border",
        "unrelated",
    }
)
_UNCERTAIN_REASONS = frozenset(REASON_CODES) - {"media_failure"}
_POLICY = policy_payload()
_PRIMARY_SEED = int(_POLICY["sampling_design"]["primary_sampling_seed"])
_DOUBLE_SEED = int(
    _POLICY["sampling_design"]["double_review_sampling_seed"]
)
del _POLICY


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object_sha256(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _require_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be 64 lowercase hexadecimal bytes")
    return value


def upstream_bindings_sha256(bindings: Mapping[str, Any]) -> str:
    """Digest the exact upstream receipt fields anchored outside this audit."""

    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != _SOURCE_BINDING_FIELDS
    ):
        raise ValueError("upstream source bindings differ")
    validated = {
        field: _require_sha256(
            bindings.get(field),
            context=f"upstream source binding {field}",
        )
        for field in sorted(_SOURCE_BINDING_FIELDS)
    }
    return _object_sha256(validated)


def population_context_sha256(context: Mapping[str, Any]) -> str:
    """Return the canonical digest independently anchored outside the commit."""

    if not isinstance(context, Mapping):
        raise ValueError("population context must be a mapping")
    return _object_sha256(context)


def normalize_reviewer_id(value: Any, *, context: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{context} reviewer ID must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    if _REVIEWER_ID_RE.fullmatch(normalized) is None:
        raise ValueError(
            f"{context} reviewer ID must match "
            "[a-z0-9][a-z0-9._@+-]{0,127}"
        )
    if value != normalized:
        raise ValueError(f"{context} reviewer ID must already be normalized")
    return normalized


def statistical_unit_id(base_component_pair: Sequence[str]) -> str:
    """Return the canonical ID for one unordered base-component pair."""

    if (
        isinstance(base_component_pair, (str, bytes))
        or len(base_component_pair) != 2
    ):
        raise ValueError("base component pair must contain exactly two IDs")
    components = list(base_component_pair)
    if any(
        type(value) is not str
        or _SAFE_COMPONENT_RE.fullmatch(value) is None
        for value in components
    ):
        raise ValueError("base component IDs are invalid")
    if components[0] >= components[1]:
        raise ValueError(
            "base component pair must be distinct and lexicographically sorted"
        )
    return _object_sha256(
        {
            "schema_version": STATISTICAL_UNIT_SCHEMA,
            "base_component_pair": components,
        }
    )


def selection_sha256(cohort: str, unit_id: str) -> str:
    """Return the frozen primary selection key for a sampled unit."""

    if cohort not in COHORT_PRIMARY_TARGETS:
        raise ValueError(f"unsupported neighbor audit cohort: {cohort!r}")
    _require_sha256(unit_id, context="statistical unit ID")
    return _object_sha256(
        {
            "schema_version": SOURCE_MANIFEST_SCHEMA,
            "seed": _PRIMARY_SEED,
            "cohort": cohort,
            "statistical_unit_id": unit_id,
        }
    )


def _double_review_sha256(cohort: str, unit_id: str) -> str:
    return _object_sha256(
        {
            "schema_version": REVIEW_TEMPLATE_SCHEMA,
            "seed": _DOUBLE_SEED,
            "cohort": cohort,
            "statistical_unit_id": unit_id,
        }
    )


def _validate_relative_media_path(value: Any, *, context: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{context} relative path must be a string")
    if "\\" in value:
        raise ValueError(f"{context} relative path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{context} relative path is not normalized")
    if path.suffix.casefold() not in _VIDEO_SUFFIXES:
        raise ValueError(f"{context} does not have an allowed video suffix")
    if canonical_json(value) != canonical_json(path.as_posix()):
        raise ValueError(f"{context} relative path is not canonical")
    return path.as_posix()


def _regular_media_path(
    root: Path,
    relative_path: str,
    *,
    context: str,
) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{context} media root must be a non-symlink directory")
    root_resolved = root.resolve(strict=True)
    current = root_resolved
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{context} media path contains a symlink")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{context} media escapes its root") from error
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{context} media must be a regular file")
    return resolved


def _validate_bound_media(
    value: Any,
    *,
    media_root: Path,
    context: str,
    opaque: bool,
) -> dict[str, Any]:
    expected_fields = _OPAQUE_MEDIA_FIELDS if opaque else _SOURCE_MEDIA_FIELDS
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{context} media fields differ")
    media = dict(value)
    relative_path = _validate_relative_media_path(
        media.get("relative_path"),
        context=context,
    )
    expected_sha256 = _require_sha256(
        media.get("sha256"),
        context=f"{context} sha256",
    )
    size_bytes = media.get("size_bytes")
    if type(size_bytes) is not int or size_bytes <= 0:
        raise ValueError(f"{context} size_bytes must be positive")
    if media.get("complete_video") is not True:
        raise ValueError(f"{context} must bind a complete video")
    if opaque:
        if media.get("schema_version") != MEDIA_BINDING_SCHEMA:
            raise ValueError(f"{context} opaque media schema differs")
        if media.get("opaque_media_id") not in {"video_1", "video_2"}:
            raise ValueError(f"{context} opaque media ID differs")
        if media.get("whole_file_byte_copy_verified") is not True:
            raise ValueError(f"{context} byte-copy verification differs")
    path = _regular_media_path(
        media_root,
        relative_path,
        context=context,
    )
    before = path.stat()
    if before.st_size != size_bytes:
        raise ValueError(f"{context} media size differs")
    actual_sha256 = _file_sha256(path)
    after = path.stat()
    if (
        before.st_ino != after.st_ino
        or before.st_dev != after.st_dev
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"{context} media changed during validation")
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{context} media SHA-256 differs")
    return media


def _validate_media_binding_syntax(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_MEDIA_FIELDS:
        raise ValueError(f"{context} media fields differ")
    media = dict(value)
    _validate_relative_media_path(
        media.get("relative_path"),
        context=context,
    )
    _require_sha256(media.get("sha256"), context=f"{context} sha256")
    if type(media.get("size_bytes")) is not int or media["size_bytes"] <= 0:
        raise ValueError(f"{context} size_bytes must be positive")
    if media.get("complete_video") is not True:
        raise ValueError(f"{context} must bind a complete video")
    return media


def _derive_cohort(row: Mapping[str, Any]) -> str:
    witness = row["witness"]
    score = float(row["hidden_context"]["score"])
    if witness["priority"]:
        return "component_risk"
    if score >= 0.96:
        return "hard"
    if score >= 0.92:
        return "boundary"
    if witness["top_neighbor"]:
        return "below_floor"
    return "far_negative"


def _component_risk_priority_tier(row: Mapping[str, Any]) -> int:
    witness = row["witness"]
    if (
        witness["high_impact"]
        and witness["hard_edge"]
        and witness["msf_witness"]
    ):
        return 0
    if (
        witness["high_impact"]
        and not witness["hard_edge"]
        and (witness["cross_component"] or witness["cross_split"])
    ):
        return 1
    if witness["large_component_witness"]:
        return 2
    if witness["top_merge_witness"]:
        return 3
    if witness["priority"]:
        return 4
    raise ValueError("non-priority row has no component-risk priority tier")


def _validate_population_row_structure(
    value: Any,
    *,
    context: str,
    media_root: Path | None,
    validate_media_bytes: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _POPULATION_FIELDS:
        raise ValueError(f"{context} population-row fields differ")
    row = copy.deepcopy(dict(value))
    if row.get("schema_version") != POPULATION_ROW_SCHEMA:
        raise ValueError(f"{context} population-row schema differs")
    if row.get("policy_sha256") != policy_sha256():
        raise ValueError(f"{context} policy digest differs")
    if row.get("thresholds_human_calibrated") is not False:
        raise ValueError(f"{context} threshold calibration flag differs")
    expected_unit_id = statistical_unit_id(row.get("base_component_pair"))
    if row.get("statistical_unit_id") != expected_unit_id:
        raise ValueError(f"{context} statistical unit ID differs")

    bindings = row.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _SOURCE_BINDING_FIELDS:
        raise ValueError(f"{context} source bindings differ")
    upstream_bindings_sha256(bindings)

    hidden = row.get("hidden_context")
    if not isinstance(hidden, Mapping) or set(hidden) != _HIDDEN_CONTEXT_FIELDS:
        raise ValueError(f"{context} hidden context fields differ")
    score = hidden.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not -1.0 <= float(score) <= 1.0
    ):
        raise ValueError(f"{context} score differs")
    score = float(score)
    score_bin = hidden.get("score_bin")
    if type(score_bin) is not str or _SAFE_BIN_RE.fullmatch(score_bin) is None:
        raise ValueError(f"{context} score bin differs")
    if (
        type(hidden.get("threshold_relation")) is not str
        or not hidden["threshold_relation"]
    ):
        raise ValueError(f"{context} threshold relation differs")
    anchor_flags = hidden.get("anchor_flags")
    if (
        not isinstance(anchor_flags, list)
        or len(anchor_flags) != 2
        or any(type(flag) is not bool for flag in anchor_flags)
    ):
        raise ValueError(f"{context} anchor flags differ")
    qwen = hidden.get("qwen_evidence_sha256")
    if not isinstance(qwen, list) or len(qwen) != 2:
        raise ValueError(f"{context} Qwen evidence binding differs")
    for index, digest in enumerate(qwen):
        if digest is not None:
            _require_sha256(
                digest,
                context=f"{context} Qwen evidence {index}",
            )
    iids = hidden.get("iid_pair")
    if (
        not isinstance(iids, list)
        or len(iids) != 2
        or any(type(iid) is not str or not iid for iid in iids)
    ):
        raise ValueError(f"{context} IID pair differs")
    splits = hidden.get("provisional_split_pair")
    if (
        not isinstance(splits, list)
        or len(splits) != 2
        or any(
            split not in {None, "train", "validation", "test"}
            for split in splits
        )
    ):
        raise ValueError(f"{context} provisional split pair differs")

    witness = row.get("witness")
    if not isinstance(witness, Mapping) or set(witness) != _WITNESS_FIELDS:
        raise ValueError(f"{context} witness fields differ")
    if any(type(witness[field]) is not bool for field in _WITNESS_FIELDS):
        raise ValueError(f"{context} witness values must be booleans")
    if witness["hard_edge"] is not (score >= 0.96):
        raise ValueError(f"{context} hard-edge/score semantics differ")
    if witness["msf_witness"] and not witness["hard_edge"]:
        raise ValueError(f"{context} non-hard MSF witness is invalid")
    if witness["high_impact"] and not witness["priority"]:
        raise ValueError(f"{context} high-impact witness must be priority")
    if (
        witness["large_component_witness"]
        or witness["top_merge_witness"]
    ) and not witness["priority"]:
        raise ValueError(f"{context} graph-risk witness must be priority")
    derived_cohort = _derive_cohort(row)
    if hidden["threshold_relation"] != derived_cohort:
        raise ValueError(f"{context} frozen cohort precedence differs")
    if derived_cohort == "component_risk":
        _component_risk_priority_tier(row)

    media = row.get("media")
    if not isinstance(media, list) or len(media) != 2:
        raise ValueError(f"{context} must contain exactly two videos")
    for index, item in enumerate(media):
        if validate_media_bytes:
            if media_root is None:
                raise ValueError("media_root is required to validate bytes")
            _validate_bound_media(
                item,
                media_root=media_root,
                context=f"{context} media[{index}]",
                opaque=False,
            )
        else:
            _validate_media_binding_syntax(
                item,
                context=f"{context} media[{index}]",
            )
    return row


def _canonical_population_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    hasher = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["statistical_unit_id"]):
        hasher.update(canonical_json(row).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _validated_population_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    media_root: Path | None,
    validate_media_bytes: bool,
) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("population manifest must be a sequence")
    validated = [
        _validate_population_row_structure(
            row,
            context=f"population row {index}",
            media_root=media_root,
            validate_media_bytes=validate_media_bytes,
        )
        for index, row in enumerate(rows)
    ]
    unit_ids = [str(row["statistical_unit_id"]) for row in validated]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("population repeats a statistical unit")
    pairs = [tuple(row["base_component_pair"]) for row in validated]
    if len(set(pairs)) != len(pairs):
        raise ValueError("population repeats a base-component pair")
    bindings = {
        canonical_json(row["source_bindings"]) for row in validated
    }
    if len(bindings) != 1:
        raise ValueError("population source artifact bindings differ")
    return sorted(validated, key=lambda row: row["statistical_unit_id"])


def build_population_context(
    population_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the context that must be externally frozen before sampling."""

    validated = _validated_population_rows(
        population_rows,
        media_root=None,
        validate_media_bytes=False,
    )
    cohort_counts = Counter(_derive_cohort(row) for row in validated)
    return {
        "schema_version": POPULATION_CONTEXT_SCHEMA,
        "policy_sha256": policy_sha256(),
        "thresholds_human_calibrated": False,
        "population_rows": len(validated),
        "population_sha256": _canonical_population_sha256(validated),
        "canonical_order": "statistical_unit_id",
        "canonical_encoding":
            "canonical-json-one-row-per-line-with-terminal-newline",
        "statistical_unit": "unordered_base_component_pair",
        "cohort_precedence": [
            "component_risk",
            "hard",
            "boundary",
            "below_floor",
            "far_negative",
        ],
        "cohort_population_counts": {
            cohort: int(cohort_counts.get(cohort, 0))
            for cohort in COHORT_PRIMARY_TARGETS
        },
        "source_bindings": copy.deepcopy(validated[0]["source_bindings"])
        if validated
        else None,
    }


def validate_population_manifest(
    population_rows: Sequence[Mapping[str, Any]],
    *,
    population_context: Mapping[str, Any],
    expected_population_sha256: str,
    expected_population_context_sha256: str,
    expected_upstream_bindings_sha256: str,
    media_root: Path | None = None,
    validate_media_bytes: bool = False,
) -> dict[str, Any]:
    """Validate the population against three independent external anchors."""

    if (
        not isinstance(population_context, Mapping)
        or set(population_context) != _POPULATION_CONTEXT_FIELDS
    ):
        raise ValueError("population context fields differ")
    expected_population_digest = _require_sha256(
        expected_population_sha256,
        context="expected population",
    )
    expected_context_digest = _require_sha256(
        expected_population_context_sha256,
        context="expected population context",
    )
    expected_upstream_digest = _require_sha256(
        expected_upstream_bindings_sha256,
        context="expected upstream bindings",
    )
    if population_context_sha256(population_context) != expected_context_digest:
        raise ValueError("external population-context anchor differs")
    validated = _validated_population_rows(
        population_rows,
        media_root=media_root,
        validate_media_bytes=validate_media_bytes,
    )
    expected_context = build_population_context(validated)
    actual_upstream_digest = upstream_bindings_sha256(
        expected_context["source_bindings"]
    )
    if actual_upstream_digest != expected_upstream_digest:
        raise ValueError("external upstream-bindings anchor differs")
    if (
        expected_context["population_sha256"]
        != expected_population_digest
    ):
        raise ValueError("external population anchor differs")
    if canonical_json(population_context) != canonical_json(expected_context):
        raise ValueError("population context or complete population digest differs")
    for cohort, target in COHORT_PRIMARY_TARGETS.items():
        population_size = int(
            expected_context["cohort_population_counts"][cohort]
        )
        if population_size < target:
            raise ValueError(
                f"population cohort {cohort} has {population_size} rows; "
                f"requires at least {target}"
            )
    cohorts: dict[str, list[dict[str, Any]]] = {
        cohort: [] for cohort in COHORT_PRIMARY_TARGETS
    }
    for row in validated:
        cohorts[_derive_cohort(row)].append(row)
    return {
        "rows": validated,
        "context": expected_context,
        "population_sha256": expected_context["population_sha256"],
        "cohorts": cohorts,
    }


def _population_commit_done(
    *,
    population: Mapping[str, Any],
    context_sha256: str,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    source_bindings = copy.deepcopy(
        population["context"]["source_bindings"]
    )
    core = {
        "schema_version": POPULATION_COMMIT_SCHEMA,
        "policy_sha256": policy_sha256(),
        "status": "complete",
        "population_rows": len(population["rows"]),
        "population_sha256": population["population_sha256"],
        "population_context_sha256": context_sha256,
        "source_bindings": source_bindings,
        "upstream_bindings_sha256":
            upstream_bindings_sha256(source_bindings),
        "thresholds_human_calibrated": False,
        "training_authorized": False,
        "formal_report": False,
        "exact_recursive_closure": copy.deepcopy(closure),
        "closure_sha256": _object_sha256(closure),
    }
    return {
        **core,
        "artifact_digest": _object_sha256(core),
    }


def commit_population_manifest(
    population_rows: Sequence[Mapping[str, Any]],
    *,
    population_context: Mapping[str, Any],
    expected_population_sha256: str,
    expected_population_context_sha256: str,
    expected_upstream_bindings_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Publish the externally anchored population as a create-only commit."""

    population = validate_population_manifest(
        population_rows,
        population_context=population_context,
        expected_population_sha256=expected_population_sha256,
        expected_population_context_sha256=
            expected_population_context_sha256,
        expected_upstream_bindings_sha256=
            expected_upstream_bindings_sha256,
    )
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(
            f"population commit already exists: {output_directory}"
        )
    _reject_nested_commit_output(output_directory)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    try:
        _write_exclusive(
            stage / POPULATION_NAME,
            _jsonl_bytes(population["rows"]),
        )
        _write_exclusive(
            stage / POPULATION_CONTEXT_NAME,
            (
                canonical_json(population["context"]) + "\n"
            ).encode("utf-8"),
        )
        closure = _recursive_closure(
            stage,
            excluded_root_files=frozenset({POPULATION_DONE_NAME}),
            require_root_mode=False,
        )
        done = _population_commit_done(
            population=population,
            context_sha256=expected_population_context_sha256,
            closure=closure,
        )
        _write_exclusive(
            stage / POPULATION_DONE_NAME,
            (canonical_json(done) + "\n").encode("utf-8"),
        )
        os.replace(stage, output_directory)
        stage = None
        os.chmod(output_directory, 0o555)
        return done
    finally:
        if stage is not None and stage.exists():
            _make_tree_removable(stage)
            shutil.rmtree(stage)


def validate_population_commit(
    population_commit_root: Path,
    *,
    expected_population_sha256: str,
    expected_population_context_sha256: str,
    expected_upstream_bindings_sha256: str,
    expected_population_commit_digest: str,
) -> dict[str, Any]:
    """Validate an exact population closure against four external anchors."""

    expected_commit = _require_sha256(
        expected_population_commit_digest,
        context="expected population commit",
    )
    closure = _recursive_closure(
        population_commit_root,
        excluded_root_files=frozenset({POPULATION_DONE_NAME}),
        require_root_mode=True,
    )
    if closure["directories"]:
        raise ValueError("population commit contains extra directories")
    if {
        record["relative_path"] for record in closure["files"]
    } != {POPULATION_NAME, POPULATION_CONTEXT_NAME}:
        raise ValueError("population commit has missing or extra files")
    population_rows = load_jsonl(
        population_commit_root / POPULATION_NAME
    )
    context = _load_canonical_object(
        population_commit_root / POPULATION_CONTEXT_NAME,
        context="population context",
    )
    population = validate_population_manifest(
        population_rows,
        population_context=context,
        expected_population_sha256=expected_population_sha256,
        expected_population_context_sha256=
            expected_population_context_sha256,
        expected_upstream_bindings_sha256=
            expected_upstream_bindings_sha256,
    )
    done = _load_canonical_object(
        population_commit_root / POPULATION_DONE_NAME,
        context="population done",
    )
    expected_done = _population_commit_done(
        population=population,
        context_sha256=expected_population_context_sha256,
        closure=closure,
    )
    if canonical_json(done) != canonical_json(expected_done):
        raise ValueError("population commit done or closure differs")
    if done.get("artifact_digest") != expected_commit:
        raise ValueError("external population-commit anchor differs")
    return {
        **population,
        "done": done,
        "commit_digest": expected_commit,
    }


def _sampling_design(cohort: str, population_size: int) -> dict[str, Any]:
    sample_size = int(COHORT_PRIMARY_TARGETS[cohort])
    if cohort == "component_risk":
        return {
            "design": "nonprobability_purposive_priority",
            "population_size_N_h": population_size,
            "sample_size_n_h": sample_size,
            "inclusion_probability_pi_h": None,
            "design_weight": None,
            "selection_rule":
                "priority-tier-then-global-bottom-sha256",
        }
    return {
        "design": "SRSWOR",
        "population_size_N_h": population_size,
        "sample_size_n_h": sample_size,
        "inclusion_probability_pi_h": sample_size / population_size,
        "design_weight": population_size / sample_size,
        "selection_rule": "global-bottom-sha256-within-frozen-cohort",
    }


def _derive_source_manifest(
    population: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    population_digest = str(population["population_sha256"])
    for cohort, target in COHORT_PRIMARY_TARGETS.items():
        candidates = list(population["cohorts"][cohort])
        if cohort == "component_risk":
            candidates.sort(
                key=lambda row: (
                    _component_risk_priority_tier(row),
                    selection_sha256(
                        cohort,
                        str(row["statistical_unit_id"]),
                    ),
                    row["statistical_unit_id"],
                )
            )
        else:
            candidates.sort(
                key=lambda row: (
                    selection_sha256(
                        cohort,
                        str(row["statistical_unit_id"]),
                    ),
                    row["statistical_unit_id"],
                )
            )
        design = _sampling_design(cohort, len(candidates))
        for cohort_rank, population_row in enumerate(candidates[:target]):
            result.append(
                {
                    "schema_version": SOURCE_MANIFEST_SCHEMA,
                    "policy_sha256": policy_sha256(),
                    "thresholds_human_calibrated": False,
                    "statistical_unit_id":
                        population_row["statistical_unit_id"],
                    "cohort": cohort,
                    "cohort_rank": cohort_rank,
                    "sample_order": len(result),
                    "selection_sha256": selection_sha256(
                        cohort,
                        str(population_row["statistical_unit_id"]),
                    ),
                    "population_sha256": population_digest,
                    "population_row_sha256": _object_sha256(population_row),
                    "sampling_design": copy.deepcopy(design),
                    "base_component_pair": copy.deepcopy(
                        population_row["base_component_pair"]
                    ),
                    "source_bindings": copy.deepcopy(
                        population_row["source_bindings"]
                    ),
                    "hidden_context": copy.deepcopy(
                        population_row["hidden_context"]
                    ),
                    "witness": copy.deepcopy(population_row["witness"]),
                    "media": copy.deepcopy(population_row["media"]),
                }
            )
    return result


def build_source_manifest(
    population_commit_root: Path,
    *,
    expected_population_sha256: str,
    expected_population_context_sha256: str,
    expected_upstream_bindings_sha256: str,
    expected_population_commit_digest: str,
) -> list[dict[str, Any]]:
    """Derive the sole valid 800-row formal sample from the full population."""

    population = validate_population_commit(
        population_commit_root,
        expected_population_sha256=expected_population_sha256,
        expected_population_context_sha256=
            expected_population_context_sha256,
        expected_upstream_bindings_sha256=
            expected_upstream_bindings_sha256,
        expected_population_commit_digest=
            expected_population_commit_digest,
    )
    return _derive_source_manifest(population)


def validate_source_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    population_commit_root: Path,
    expected_population_sha256: str,
    expected_population_context_sha256: str,
    expected_upstream_bindings_sha256: str,
    expected_population_commit_digest: str,
    media_root: Path,
) -> list[dict[str, Any]]:
    """Recompute and validate the exact sample from the full population."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("source manifest must be a sequence")
    population = validate_population_commit(
        population_commit_root,
        expected_population_sha256=expected_population_sha256,
        expected_population_context_sha256=
            expected_population_context_sha256,
        expected_upstream_bindings_sha256=
            expected_upstream_bindings_sha256,
        expected_population_commit_digest=
            expected_population_commit_digest,
    )
    expected = _derive_source_manifest(population)
    if len(rows) != len(expected):
        raise ValueError(
            f"source manifest must contain exactly {len(expected)} rows"
        )
    for index, (actual, frozen) in enumerate(zip(rows, expected)):
        if not isinstance(actual, Mapping) or set(actual) != _SOURCE_FIELDS:
            raise ValueError(
                f"source manifest row {index} fields differ"
            )
        if canonical_json(actual) != canonical_json(frozen):
            raise ValueError(
                f"source manifest row {index} is not the globally selected "
                "population row"
            )
        design = actual["sampling_design"]
        if not isinstance(design, Mapping) or set(design) != _SAMPLING_DESIGN_FIELDS:
            raise ValueError(
                f"source manifest row {index} sampling design differs"
            )
        for media_index, binding in enumerate(actual["media"]):
            _validate_bound_media(
                binding,
                media_root=media_root,
                context=(
                    f"source manifest row {index} media[{media_index}]"
                ),
                opaque=False,
            )
    return copy.deepcopy(expected)


def _selected_secondary_unit_ids(
    rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    selected: set[str] = set()
    for cohort, target in COHORT_DOUBLE_REVIEW_TARGETS.items():
        candidates = [row for row in rows if row["cohort"] == cohort]
        candidates.sort(
            key=lambda row: (
                _double_review_sha256(
                    cohort,
                    str(row["statistical_unit_id"]),
                ),
                row["statistical_unit_id"],
            )
        )
        selected.update(
            str(row["statistical_unit_id"]) for row in candidates[:target]
        )
    if len(selected) != sum(COHORT_DOUBLE_REVIEW_TARGETS.values()):
        raise AssertionError("double-review selection count differs")
    return selected


def _review_instance_id(
    *,
    source_row: Mapping[str, Any],
    reviewer_id: str,
    slot: str,
) -> str:
    return _object_sha256(
        {
            "schema_version": REVIEW_TEMPLATE_SCHEMA,
            "policy_sha256": policy_sha256(),
            "source_row_sha256": _object_sha256(source_row),
            "assigned_reviewer_id": reviewer_id,
            "internal_annotator_slot": slot,
        }
    )


def _assignment_entries(
    rows: Sequence[dict[str, Any]],
    *,
    reviewer_id: str,
    role: str,
) -> list[dict[str, Any]]:
    if role not in {"primary", "secondary"}:
        raise ValueError("reviewer role must be primary or secondary")
    selected_units = (
        {str(row["statistical_unit_id"]) for row in rows}
        if role == "primary"
        else _selected_secondary_unit_ids(rows)
    )
    entries: list[dict[str, Any]] = []
    for row in rows:
        if row["statistical_unit_id"] in selected_units:
            entries.append(
                {
                    "source_row": row,
                    "slot": role,
                    "reviewer_id": reviewer_id,
                    "review_instance_id": _review_instance_id(
                        source_row=row,
                        reviewer_id=reviewer_id,
                        slot=role,
                    ),
                }
            )
    entries.sort(key=lambda item: item["review_instance_id"])
    expected = (
        sum(COHORT_PRIMARY_TARGETS.values())
        if role == "primary"
        else sum(COHORT_DOUBLE_REVIEW_TARGETS.values())
    )
    if len(entries) != expected:
        raise AssertionError(f"{role} assignment count differs")
    return entries


def _assignment_set_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    return _object_sha256(
        {
            "schema_version": REVIEW_BUNDLE_SCHEMA,
            "policy_sha256": policy_sha256(),
            "review_instance_ids": [
                entry["review_instance_id"] for entry in entries
            ],
        }
    )


def _media_source_order(review_instance_id: str) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"motive-r7-neighbor-media-order-v1:{review_instance_id}".encode(
            "utf-8"
        )
    ).digest()
    return (0, 1) if digest[0] % 2 == 0 else (1, 0)


def _expected_media_bindings(
    entry: Mapping[str, Any],
    *,
    source_media_root: Path,
    bundle_root: Path,
    copy_media: bool,
) -> list[dict[str, Any]]:
    source_row = entry["source_row"]
    review_instance_id = str(entry["review_instance_id"])
    result: list[dict[str, Any]] = []
    for output_index, source_index in enumerate(
        _media_source_order(review_instance_id),
        start=1,
    ):
        source_binding = _validate_bound_media(
            source_row["media"][source_index],
            media_root=source_media_root,
            context=(
                f"source unit {source_row['statistical_unit_id']} "
                f"media[{source_index}]"
            ),
            opaque=False,
        )
        suffix = PurePosixPath(source_binding["relative_path"]).suffix.casefold()
        relative = (
            f"media/{review_instance_id}/video_{output_index}{suffix}"
        )
        destination = bundle_root / PurePosixPath(relative)
        if copy_media:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = _regular_media_path(
                source_media_root,
                source_binding["relative_path"],
                context="review-bundle source",
            )
            before = source.stat()
            hasher = hashlib.sha256()
            with source.open("rb") as reader, destination.open("xb") as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(block)
                    hasher.update(block)
                writer.flush()
                os.fsync(writer.fileno())
            os.chmod(destination, 0o444)
            after = source.stat()
            if (
                before.st_ino != after.st_ino
                or before.st_dev != after.st_dev
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or hasher.hexdigest() != source_binding["sha256"]
                or destination.stat().st_size != source_binding["size_bytes"]
            ):
                raise RuntimeError("source media changed during whole-file copy")
        result.append(
            {
                "schema_version": MEDIA_BINDING_SCHEMA,
                "opaque_media_id": f"video_{output_index}",
                "relative_path": relative,
                "sha256": source_binding["sha256"],
                "size_bytes": source_binding["size_bytes"],
                "complete_video": True,
                "whole_file_byte_copy_verified": True,
            }
        )
    return result


def _template_row(
    entry: Mapping[str, Any],
    *,
    assignment_set_digest: str,
    source_media_root: Path,
    bundle_root: Path,
    copy_media: bool,
) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_TEMPLATE_SCHEMA,
        "policy_sha256": policy_sha256(),
        "thresholds_human_calibrated": False,
        "review_instance_id": entry["review_instance_id"],
        "assignment_set_digest": assignment_set_digest,
        "assigned_reviewer_id": entry["reviewer_id"],
        "question": QUESTION,
        "media": _expected_media_bindings(
            entry,
            source_media_root=source_media_root,
            bundle_root=bundle_root,
            copy_media=copy_media,
        ),
        "allowed_verdicts": list(VERDICTS),
        "allowed_reason_codes": list(REASON_CODES),
        "label_scope": "split_threshold_audit_only",
        "training_authorized": False,
        "direct_training_supervision_allowed": False,
        "verdict": None,
        "reason_codes": [],
        "notes": None,
        "completed_at_utc": None,
        "review_attestation": {
            "video_1_reviewed_in_full": None,
            "video_2_reviewed_in_full": None,
            "independent_judgment": None,
            "other_reviewer_result_not_observed": None,
        },
    }


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        canonical_json(row).encode("utf-8") + b"\n" for row in rows
    )


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _reject_nested_commit_output(output_directory: Path) -> None:
    markers = (
        POPULATION_DONE_NAME,
        REVIEW_BUNDLE_DONE_NAME,
        LABELS_DONE_NAME,
    )
    if any(
        any((ancestor / marker).exists() for marker in markers)
        for ancestor in output_directory.parents
    ):
        raise ValueError(
            "artifact output cannot be nested inside another commit"
        )


def _recursive_closure(
    root: Path,
    *,
    excluded_root_files: frozenset[str],
    require_root_mode: bool,
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("review closure root must be a non-symlink directory")
    if (
        require_root_mode
        and stat.S_IMODE(root.stat().st_mode) != 0o555
    ):
        raise ValueError("artifact closure root mode differs from 0555")
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    for directory, child_directories, child_files in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(directory)
        for child in sorted(child_directories):
            path = current / child
            if path.is_symlink() or not path.is_dir():
                raise ValueError("review closure contains a directory symlink")
            if stat.S_IMODE(path.stat().st_mode) != 0o555:
                raise ValueError(
                    "artifact closure directory mode differs from 0555"
                )
            directories.append(path.relative_to(root).as_posix())
        for child in sorted(child_files):
            path = current / child
            relative = path.relative_to(root).as_posix()
            if relative in excluded_root_files:
                continue
            if path.is_symlink():
                raise ValueError("review closure contains a file symlink")
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("review closure contains a non-regular file")
            if metadata.st_nlink != 1:
                raise ValueError("artifact closure contains a hard-linked file")
            mode = stat.S_IMODE(metadata.st_mode)
            if mode != 0o444:
                raise ValueError("review closure file mode differs from 0444")
            digest = _file_sha256(path)
            after = path.stat()
            if (
                metadata.st_dev != after.st_dev
                or metadata.st_ino != after.st_ino
                or metadata.st_size != after.st_size
                or metadata.st_mtime_ns != after.st_mtime_ns
                or metadata.st_mode != after.st_mode
                or after.st_nlink != 1
            ):
                raise RuntimeError(
                    "artifact closure file changed during hashing"
                )
            files.append(
                {
                    "relative_path": relative,
                    "sha256": digest,
                    "size_bytes": metadata.st_size,
                    "mode_octal": "0444",
                }
            )
    directories.sort()
    files.sort(key=lambda item: item["relative_path"])
    return {
        "root_mode_octal": "0555",
        "directories": directories,
        "files": files,
    }


def _seal_child_directories(root: Path) -> None:
    directories = [
        path
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    ]
    for path in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        os.chmod(path, 0o555)


def _make_tree_removable(root: Path) -> None:
    if root.exists() and not root.is_symlink():
        os.chmod(root, 0o755)
        for path in root.rglob("*"):
            if path.is_symlink():
                continue
            os.chmod(path, 0o755 if path.is_dir() else 0o644)


def _review_bundle_done(
    *,
    reviewer_id: str,
    templates: Sequence[Mapping[str, Any]],
    template_payload: bytes,
    assignment_set_digest: str,
    source_rows: Sequence[Mapping[str, Any]],
    population_context_sha256: str,
    upstream_bindings_digest: str,
    population_commit_digest: str,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": REVIEW_BUNDLE_SCHEMA,
        "policy_sha256": policy_sha256(),
        "status": "complete",
        "assigned_reviewer_id": reviewer_id,
        "review_count": len(templates),
        "assignment_set_digest": assignment_set_digest,
        "review_template_sha256": _sha256_bytes(template_payload),
        "source_manifest_sha256": _object_sha256(source_rows),
        "population_sha256": source_rows[0]["population_sha256"],
        "population_context_sha256": population_context_sha256,
        "upstream_bindings_sha256": upstream_bindings_digest,
        "population_commit_digest": population_commit_digest,
        "label_scope": "split_threshold_audit_only",
        "thresholds_human_calibrated": False,
        "training_authorized": False,
        "direct_training_supervision_allowed": False,
        "exact_recursive_closure": copy.deepcopy(closure),
        "closure_sha256": _object_sha256(closure),
    }
    return {
        **core,
        "artifact_digest": _object_sha256(core),
    }


def prepare_reviewer_bundle(
    rows: Sequence[Mapping[str, Any]],
    *,
    population_commit_root: Path,
    expected_population_sha256: str,
    expected_population_context_sha256: str,
    expected_upstream_bindings_sha256: str,
    expected_population_commit_digest: str,
    source_media_root: Path,
    output_directory: Path,
    reviewer_role: str,
    reviewer_id: str,
) -> dict[str, Any]:
    """Create one independent, exact-closure blind reviewer package."""

    if reviewer_role not in {"primary", "secondary"}:
        raise ValueError("reviewer role must be primary or secondary")
    normalized_reviewer = normalize_reviewer_id(
        reviewer_id,
        context=reviewer_role,
    )
    validated = validate_source_manifest(
        rows,
        population_commit_root=population_commit_root,
        expected_population_sha256=expected_population_sha256,
        expected_population_context_sha256=
            expected_population_context_sha256,
        expected_upstream_bindings_sha256=
            expected_upstream_bindings_sha256,
        expected_population_commit_digest=
            expected_population_commit_digest,
        media_root=source_media_root,
    )
    entries = _assignment_entries(
        validated,
        reviewer_id=normalized_reviewer,
        role=reviewer_role,
    )
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"review output already exists: {output_directory}")
    _reject_nested_commit_output(output_directory)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    try:
        assignment_digest = _assignment_set_digest(entries)
        templates = [
            _template_row(
                entry,
                assignment_set_digest=assignment_digest,
                source_media_root=source_media_root,
                bundle_root=stage,
                copy_media=True,
            )
            for entry in entries
        ]
        payload = _jsonl_bytes(templates)
        _write_exclusive(stage / REVIEW_NAME, payload)
        _seal_child_directories(stage)
        closure = _recursive_closure(
            stage,
            excluded_root_files=frozenset({REVIEW_BUNDLE_DONE_NAME}),
            require_root_mode=False,
        )
        done = _review_bundle_done(
            reviewer_id=normalized_reviewer,
            templates=templates,
            template_payload=payload,
            assignment_set_digest=assignment_digest,
            source_rows=validated,
            population_context_sha256=
                expected_population_context_sha256,
            upstream_bindings_digest=
                expected_upstream_bindings_sha256,
            population_commit_digest=expected_population_commit_digest,
            closure=closure,
        )
        _write_exclusive(
            stage / REVIEW_BUNDLE_DONE_NAME,
            (canonical_json(done) + "\n").encode("utf-8"),
        )
        os.replace(stage, output_directory)
        stage = None
        os.chmod(output_directory, 0o555)
        return done
    finally:
        if stage is not None and stage.exists():
            _make_tree_removable(stage)
            shutil.rmtree(stage)


def _validate_review_row_shape(
    value: Any,
    *,
    bundle_root: Path,
    context: str,
    completed: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TEMPLATE_FIELDS:
        raise ValueError(f"{context} review-template fields differ")
    row = copy.deepcopy(dict(value))
    if row.get("schema_version") != REVIEW_TEMPLATE_SCHEMA:
        raise ValueError(f"{context} review schema differs")
    if row.get("policy_sha256") != policy_sha256():
        raise ValueError(f"{context} policy digest differs")
    for field in ("review_instance_id", "assignment_set_digest"):
        _require_sha256(row.get(field), context=f"{context} {field}")
    normalize_reviewer_id(
        row.get("assigned_reviewer_id"),
        context=f"{context} assigned",
    )
    if row.get("question") != QUESTION:
        raise ValueError(f"{context} question differs")
    if row.get("allowed_verdicts") != list(VERDICTS):
        raise ValueError(f"{context} allowed verdicts differ")
    if row.get("allowed_reason_codes") != list(REASON_CODES):
        raise ValueError(f"{context} allowed reason codes differ")
    if row.get("label_scope") != "split_threshold_audit_only":
        raise ValueError(f"{context} label scope differs")
    if row.get("thresholds_human_calibrated") is not False:
        raise ValueError(f"{context} threshold calibration flag differs")
    if row.get("training_authorized") is not False:
        raise ValueError(f"{context} training authorization differs")
    if row.get("direct_training_supervision_allowed") is not False:
        raise ValueError(f"{context} direct-supervision flag differs")
    media = row.get("media")
    if not isinstance(media, list) or len(media) != 2:
        raise ValueError(f"{context} must expose exactly two videos")
    validated_media = [
        _validate_bound_media(
            item,
            media_root=bundle_root,
            context=f"{context} media[{index}]",
            opaque=True,
        )
        for index, item in enumerate(media)
    ]
    if [
        item["opaque_media_id"] for item in validated_media
    ] != ["video_1", "video_2"]:
        raise ValueError(f"{context} opaque media order differs")

    attestation = row.get("review_attestation")
    if not isinstance(attestation, Mapping) or set(attestation) != _ATTESTATION_FIELDS:
        raise ValueError(f"{context} review attestation fields differ")
    if not completed:
        if (
            row.get("verdict") is not None
            or row.get("reason_codes") != []
            or row.get("notes") is not None
            or row.get("completed_at_utc") is not None
            or any(value is not None for value in attestation.values())
        ):
            raise ValueError(f"{context} template contains a review result")
        return row

    verdict = row.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"{context} verdict differs")
    reasons = row.get("reason_codes")
    if (
        not isinstance(reasons, list)
        or not reasons
        or len(set(reasons)) != len(reasons)
        or any(reason not in REASON_CODES for reason in reasons)
    ):
        raise ValueError(f"{context} reason codes differ")
    expected_reason_order = sorted(
        reasons,
        key=lambda reason: REASON_CODES.index(reason),
    )
    if reasons != expected_reason_order:
        raise ValueError(f"{context} reason codes are not policy ordered")
    reason_set = set(reasons)
    if verdict == "must_same_split" and (
        not reason_set <= _MUST_SAME_SPLIT_REASONS
    ):
        raise ValueError(f"{context} must-same-split reasons differ")
    if verdict == "independent_content" and (
        not reason_set <= _INDEPENDENT_CONTENT_REASONS
    ):
        raise ValueError(f"{context} independent-content reasons differ")
    if verdict == "uncertain" and not reason_set <= _UNCERTAIN_REASONS:
        raise ValueError(f"{context} uncertain reasons differ")
    if verdict == "unreviewable" and reason_set != {"media_failure"}:
        raise ValueError(f"{context} unreviewable reasons differ")

    notes = row.get("notes")
    if notes is not None and (
        type(notes) is not str
        or len(notes) > 2000
        or "\x00" in notes
    ):
        raise ValueError(f"{context} notes differ")
    completed_at = row.get("completed_at_utc")
    if (
        type(completed_at) is not str
        or _RFC3339_UTC_RE.fullmatch(completed_at) is None
    ):
        raise ValueError(f"{context} completion timestamp differs")
    if attestation.get("independent_judgment") is not True:
        raise ValueError(f"{context} independent judgment is not attested")
    if attestation.get("other_reviewer_result_not_observed") is not True:
        raise ValueError(f"{context} blind review is not attested")
    watched = (
        attestation.get("video_1_reviewed_in_full"),
        attestation.get("video_2_reviewed_in_full"),
    )
    if any(type(value) is not bool for value in watched):
        raise ValueError(f"{context} media review attestations differ")
    if verdict != "unreviewable" and watched != (True, True):
        raise ValueError(
            f"{context} conclusive/uncertain review requires both full videos"
        )
    if verdict == "unreviewable" and watched == (True, True):
        raise ValueError(
            f"{context} media failure cannot attest both complete views"
        )
    return row


def _validate_review_templates(
    rows: Sequence[Mapping[str, Any]],
    *,
    bundle_root: Path,
    expected_reviewer_id: str,
    expected_count: int,
    completed: bool,
) -> list[dict[str, Any]]:
    """Strictly validate one blind reviewer file and all bound media bytes."""

    reviewer_id = normalize_reviewer_id(
        expected_reviewer_id,
        context="expected",
    )
    if len(rows) != expected_count:
        raise ValueError(
            f"review file must contain exactly {expected_count} rows"
        )
    validated = [
        _validate_review_row_shape(
            row,
            bundle_root=bundle_root,
            context=f"review row {index}",
            completed=completed,
        )
        for index, row in enumerate(rows)
    ]
    if any(
        row["assigned_reviewer_id"] != reviewer_id for row in validated
    ):
        raise ValueError("review file assigned reviewer differs")
    review_ids = [str(row["review_instance_id"]) for row in validated]
    if review_ids != sorted(review_ids) or len(set(review_ids)) != len(review_ids):
        raise ValueError("review file order or review IDs differ")
    assignment_digests = {
        str(row["assignment_set_digest"]) for row in validated
    }
    if len(assignment_digests) != 1:
        raise ValueError("review assignment-set digests differ")
    expected_digest = _object_sha256(
        {
            "schema_version": REVIEW_BUNDLE_SCHEMA,
            "policy_sha256": policy_sha256(),
            "review_instance_ids": review_ids,
        }
    )
    if assignment_digests != {expected_digest}:
        raise ValueError("review assignment-set digest differs")
    return validated


def _load_canonical_object(
    path: Path,
    *,
    context: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be regular and non-symlink")
    before = path.stat()
    if before.st_nlink != 1:
        raise ValueError(f"{context} must not be hard-linked")
    if stat.S_IMODE(before.st_mode) != 0o444:
        raise ValueError(f"{context} mode differs from 0444")
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_mode != after.st_mode
        or after.st_nlink != 1
    ):
        raise RuntimeError(f"{context} changed during validation")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    expected = (canonical_json(value) + "\n").encode("utf-8")
    if payload != expected:
        raise ValueError(f"{context} is not canonically encoded")
    return value


def validate_reviewer_bundle(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    population_commit_root: Path,
    expected_population_sha256: str,
    expected_population_context_sha256: str,
    expected_upstream_bindings_sha256: str,
    expected_population_commit_digest: str,
    source_media_root: Path,
    bundle_root: Path,
    reviewer_role: str,
    reviewer_id: str,
) -> list[dict[str, Any]]:
    """Validate one immutable reviewer directory and its exact closure."""

    if reviewer_role not in {"primary", "secondary"}:
        raise ValueError("reviewer role must be primary or secondary")
    normalized = normalize_reviewer_id(
        reviewer_id,
        context=reviewer_role,
    )
    source = validate_source_manifest(
        source_rows,
        population_commit_root=population_commit_root,
        expected_population_sha256=expected_population_sha256,
        expected_population_context_sha256=
            expected_population_context_sha256,
        expected_upstream_bindings_sha256=
            expected_upstream_bindings_sha256,
        expected_population_commit_digest=
            expected_population_commit_digest,
        media_root=source_media_root,
    )
    entries = _assignment_entries(
        source,
        reviewer_id=normalized,
        role=reviewer_role,
    )
    expected = _expected_templates(
        entries,
        source_media_root=source_media_root,
        bundle_root=bundle_root,
    )
    review_path = bundle_root / REVIEW_NAME
    templates = load_jsonl(review_path)
    validated = _validate_review_templates(
        templates,
        bundle_root=bundle_root,
        expected_reviewer_id=normalized,
        expected_count=len(entries),
        completed=False,
    )
    if canonical_json(validated) != canonical_json(expected):
        raise ValueError("review bundle templates differ from frozen assignments")
    template_payload = _jsonl_bytes(expected)
    if review_path.read_bytes() != template_payload:
        raise ValueError("review template JSONL is not canonically encoded")
    closure = _recursive_closure(
        bundle_root,
        excluded_root_files=frozenset({REVIEW_BUNDLE_DONE_NAME}),
        require_root_mode=True,
    )
    done = _load_canonical_object(
        bundle_root / REVIEW_BUNDLE_DONE_NAME,
        context="review bundle done",
    )
    expected_done = _review_bundle_done(
        reviewer_id=normalized,
        templates=expected,
        template_payload=template_payload,
        assignment_set_digest=_assignment_set_digest(entries),
        source_rows=source,
        population_context_sha256=expected_population_context_sha256,
        upstream_bindings_digest=expected_upstream_bindings_sha256,
        population_commit_digest=expected_population_commit_digest,
        closure=closure,
    )
    if canonical_json(done) != canonical_json(expected_done):
        raise ValueError("review bundle done or closure digest differs")
    expected_paths = {
        REVIEW_NAME,
        *(
            binding["relative_path"]
            for row in expected
            for binding in row["media"]
        ),
    }
    observed_paths = {
        record["relative_path"] for record in closure["files"]
    }
    if observed_paths != expected_paths:
        raise ValueError("review bundle has missing or extra closure files")
    expected_directories = {
        "media",
        *(
            str(PurePosixPath(binding["relative_path"]).parent)
            for row in expected
            for binding in row["media"]
        ),
    }
    if set(closure["directories"]) != expected_directories:
        raise ValueError("review bundle has missing or extra directories")
    return validated


def _expected_templates(
    entries: Sequence[Mapping[str, Any]],
    *,
    source_media_root: Path,
    bundle_root: Path,
) -> list[dict[str, Any]]:
    digest = _assignment_set_digest(entries)
    return [
        _template_row(
            entry,
            assignment_set_digest=digest,
            source_media_root=source_media_root,
            bundle_root=bundle_root,
            copy_media=False,
        )
        for entry in entries
    ]


def _validate_labels_against_expected(
    labels: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    bundle_root: Path,
    reviewer_id: str,
) -> list[dict[str, Any]]:
    validated = _validate_review_templates(
        labels,
        bundle_root=bundle_root,
        expected_reviewer_id=reviewer_id,
        expected_count=len(expected),
        completed=True,
    )
    for index, (actual, frozen) in enumerate(zip(validated, expected)):
        for field in _IMMUTABLE_TEMPLATE_FIELDS:
            if actual[field] != frozen[field]:
                raise ValueError(
                    f"review row {index} immutable field {field} differs"
                )
    return validated


def _label_commit_done(
    *,
    labels: Sequence[Mapping[str, Any]],
    label_payload: bytes,
    reviewer_id: str,
    bundle_done: Mapping[str, Any],
    bundle_root: Path,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": LABEL_COMMIT_SCHEMA,
        "policy_sha256": policy_sha256(),
        "status": "complete",
        "assigned_reviewer_id": reviewer_id,
        "label_count": len(labels),
        "labels_sha256": _sha256_bytes(label_payload),
        "assignment_set_digest": bundle_done["assignment_set_digest"],
        "review_bundle_artifact_digest": bundle_done["artifact_digest"],
        "review_bundle_done_sha256": _file_sha256(
            bundle_root / REVIEW_BUNDLE_DONE_NAME
        ),
        "review_bundle_closure_sha256": bundle_done["closure_sha256"],
        "review_template_sha256": bundle_done["review_template_sha256"],
        "source_manifest_sha256": bundle_done["source_manifest_sha256"],
        "population_sha256": bundle_done["population_sha256"],
        "population_context_sha256":
            bundle_done["population_context_sha256"],
        "upstream_bindings_sha256":
            bundle_done["upstream_bindings_sha256"],
        "population_commit_digest":
            bundle_done["population_commit_digest"],
        "label_scope": "split_threshold_audit_only",
        "thresholds_human_calibrated": False,
        "training_authorized": False,
        "direct_training_supervision_allowed": False,
        "formal_report": False,
        "exact_recursive_closure": copy.deepcopy(closure),
        "closure_sha256": _object_sha256(closure),
    }
    return {
        **core,
        "artifact_digest": _object_sha256(core),
    }


def commit_reviewer_labels(
    completed_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    *,
    population_commit_root: Path,
    expected_population_sha256: str,
    expected_population_context_sha256: str,
    expected_upstream_bindings_sha256: str,
    expected_population_commit_digest: str,
    source_media_root: Path,
    reviewer_bundle_root: Path,
    reviewer_role: str,
    reviewer_id: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Publish one reviewer's completed labels as a separate commit."""

    templates = validate_reviewer_bundle(
        source_rows,
        population_commit_root=population_commit_root,
        expected_population_sha256=expected_population_sha256,
        expected_population_context_sha256=
            expected_population_context_sha256,
        expected_upstream_bindings_sha256=
            expected_upstream_bindings_sha256,
        expected_population_commit_digest=
            expected_population_commit_digest,
        source_media_root=source_media_root,
        bundle_root=reviewer_bundle_root,
        reviewer_role=reviewer_role,
        reviewer_id=reviewer_id,
    )
    normalized = normalize_reviewer_id(
        reviewer_id,
        context=reviewer_role,
    )
    labels = _validate_labels_against_expected(
        completed_rows,
        templates,
        bundle_root=reviewer_bundle_root,
        reviewer_id=normalized,
    )
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(
            f"label commit already exists: {output_directory}"
        )
    _reject_nested_commit_output(output_directory)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    try:
        payload = _jsonl_bytes(labels)
        _write_exclusive(stage / LABELS_NAME, payload)
        closure = _recursive_closure(
            stage,
            excluded_root_files=frozenset({LABELS_DONE_NAME}),
            require_root_mode=False,
        )
        bundle_done = _load_canonical_object(
            reviewer_bundle_root / REVIEW_BUNDLE_DONE_NAME,
            context="review bundle done",
        )
        done = _label_commit_done(
            labels=labels,
            label_payload=payload,
            reviewer_id=normalized,
            bundle_done=bundle_done,
            bundle_root=reviewer_bundle_root,
            closure=closure,
        )
        _write_exclusive(
            stage / LABELS_DONE_NAME,
            (canonical_json(done) + "\n").encode("utf-8"),
        )
        os.replace(stage, output_directory)
        stage = None
        os.chmod(output_directory, 0o555)
        return done
    finally:
        if stage is not None and stage.exists():
            _make_tree_removable(stage)
            shutil.rmtree(stage)


def validate_label_commit(
    label_commit_root: Path,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    population_commit_root: Path,
    expected_population_sha256: str,
    expected_population_context_sha256: str,
    expected_upstream_bindings_sha256: str,
    expected_population_commit_digest: str,
    source_media_root: Path,
    reviewer_bundle_root: Path,
    reviewer_role: str,
    reviewer_id: str,
    expected_label_commit_digest: str,
) -> dict[str, Any]:
    """Validate one completed-label closure against its external receipt."""

    expected_commit = _require_sha256(
        expected_label_commit_digest,
        context=f"expected {reviewer_role} label commit",
    )
    templates = validate_reviewer_bundle(
        source_rows,
        population_commit_root=population_commit_root,
        expected_population_sha256=expected_population_sha256,
        expected_population_context_sha256=
            expected_population_context_sha256,
        expected_upstream_bindings_sha256=
            expected_upstream_bindings_sha256,
        expected_population_commit_digest=
            expected_population_commit_digest,
        source_media_root=source_media_root,
        bundle_root=reviewer_bundle_root,
        reviewer_role=reviewer_role,
        reviewer_id=reviewer_id,
    )
    closure = _recursive_closure(
        label_commit_root,
        excluded_root_files=frozenset({LABELS_DONE_NAME}),
        require_root_mode=True,
    )
    if closure["directories"]:
        raise ValueError("label commit contains extra directories")
    if {
        record["relative_path"] for record in closure["files"]
    } != {LABELS_NAME}:
        raise ValueError("label commit has missing or extra files")
    labels = load_jsonl(label_commit_root / LABELS_NAME)
    normalized = normalize_reviewer_id(
        reviewer_id,
        context=reviewer_role,
    )
    validated = _validate_labels_against_expected(
        labels,
        templates,
        bundle_root=reviewer_bundle_root,
        reviewer_id=normalized,
    )
    payload = _jsonl_bytes(validated)
    if (label_commit_root / LABELS_NAME).read_bytes() != payload:
        raise ValueError("committed labels JSONL is not canonical")
    bundle_done = _load_canonical_object(
        reviewer_bundle_root / REVIEW_BUNDLE_DONE_NAME,
        context="review bundle done",
    )
    done = _load_canonical_object(
        label_commit_root / LABELS_DONE_NAME,
        context="labels done",
    )
    expected_done = _label_commit_done(
        labels=validated,
        label_payload=payload,
        reviewer_id=normalized,
        bundle_done=bundle_done,
        bundle_root=reviewer_bundle_root,
        closure=closure,
    )
    if canonical_json(done) != canonical_json(expected_done):
        raise ValueError("label commit done or closure differs")
    if done.get("artifact_digest") != expected_commit:
        raise ValueError("external label-commit anchor differs")
    return {
        "labels": validated,
        "done": done,
        "commit_digest": expected_commit,
    }


def merge_review_labels(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    population_commit_root: Path,
    expected_population_sha256: str,
    expected_population_context_sha256: str,
    expected_upstream_bindings_sha256: str,
    expected_population_commit_digest: str,
    source_media_root: Path,
    primary_bundle_root: Path,
    secondary_bundle_root: Path,
    primary_label_commit_root: Path,
    secondary_label_commit_root: Path,
    expected_primary_label_commit_digest: str,
    expected_secondary_label_commit_digest: str,
    primary_reviewer_id: str,
    secondary_reviewer_id: str,
) -> list[dict[str, Any]]:
    """Join only two externally anchored, committed label roots."""

    primary_id = normalize_reviewer_id(
        primary_reviewer_id,
        context="primary",
    )
    secondary_id = normalize_reviewer_id(
        secondary_reviewer_id,
        context="secondary",
    )
    if primary_id == secondary_id:
        raise ValueError("primary and secondary reviewer IDs must differ")
    source = validate_source_manifest(
        source_rows,
        population_commit_root=population_commit_root,
        expected_population_sha256=expected_population_sha256,
        expected_population_context_sha256=
            expected_population_context_sha256,
        expected_upstream_bindings_sha256=
            expected_upstream_bindings_sha256,
        expected_population_commit_digest=
            expected_population_commit_digest,
        media_root=source_media_root,
    )
    primary_commit = validate_label_commit(
        primary_label_commit_root,
        source,
        population_commit_root=population_commit_root,
        expected_population_sha256=expected_population_sha256,
        expected_population_context_sha256=
            expected_population_context_sha256,
        expected_upstream_bindings_sha256=
            expected_upstream_bindings_sha256,
        expected_population_commit_digest=
            expected_population_commit_digest,
        source_media_root=source_media_root,
        reviewer_bundle_root=primary_bundle_root,
        reviewer_role="primary",
        reviewer_id=primary_id,
        expected_label_commit_digest=
            expected_primary_label_commit_digest,
    )
    secondary_commit = validate_label_commit(
        secondary_label_commit_root,
        source,
        population_commit_root=population_commit_root,
        expected_population_sha256=expected_population_sha256,
        expected_population_context_sha256=
            expected_population_context_sha256,
        expected_upstream_bindings_sha256=
            expected_upstream_bindings_sha256,
        expected_population_commit_digest=
            expected_population_commit_digest,
        source_media_root=source_media_root,
        reviewer_bundle_root=secondary_bundle_root,
        reviewer_role="secondary",
        reviewer_id=secondary_id,
        expected_label_commit_digest=
            expected_secondary_label_commit_digest,
    )
    primary_entries = _assignment_entries(
        source,
        reviewer_id=primary_id,
        role="primary",
    )
    secondary_entries = _assignment_entries(
        source,
        reviewer_id=secondary_id,
        role="secondary",
    )
    primary = primary_commit["labels"]
    secondary = secondary_commit["labels"]
    primary_by_unit = {
        entry["source_row"]["statistical_unit_id"]: label
        for entry, label in zip(primary_entries, primary)
    }
    secondary_by_unit = {
        entry["source_row"]["statistical_unit_id"]: label
        for entry, label in zip(secondary_entries, secondary)
    }

    def label_projection(label: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "review_instance_id": label["review_instance_id"],
            "assigned_reviewer_id": label["assigned_reviewer_id"],
            "verdict": label["verdict"],
            "reason_codes": list(label["reason_codes"]),
            "notes": label["notes"],
            "completed_at_utc": label["completed_at_utc"],
            "review_attestation": dict(label["review_attestation"]),
        }

    merged: list[dict[str, Any]] = []
    for source_row in source:
        unit_id = str(source_row["statistical_unit_id"])
        primary_label = primary_by_unit[unit_id]
        secondary_label = secondary_by_unit.get(unit_id)
        merged.append(
            {
                "schema_version": MERGED_REVIEW_SCHEMA,
                "policy_sha256": policy_sha256(),
                "statistical_unit_id": unit_id,
                "source_row_sha256": _object_sha256(source_row),
                "source": copy.deepcopy(source_row),
                "primary_review": label_projection(primary_label),
                "secondary_review": (
                    label_projection(secondary_label)
                    if secondary_label is not None
                    else None
                ),
                "label_scope": "split_threshold_audit_only",
                "thresholds_human_calibrated": False,
                "training_authorized": False,
                "direct_training_supervision_allowed": False,
            }
        )
    if (
        len(merged) != sum(COHORT_PRIMARY_TARGETS.values())
        or sum(row["secondary_review"] is not None for row in merged)
        != sum(COHORT_DOUBLE_REVIEW_TARGETS.values())
    ):
        raise AssertionError("merged neighbor review conservation failed")
    return merged


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL strictly; useful for callers without weakening validators."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSONL input must be regular and non-symlink: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows
