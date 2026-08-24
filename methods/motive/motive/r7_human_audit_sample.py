"""Deterministic, provenance-bound sampling for the R7 human audit.

The expansion manifest contains three semantically different populations:

* strict Qwen pseudo-positives, whose precision must be estimated;
* trusted Qwen pseudo-negatives, whose false-negative rate must be estimated;
* unresolved review rows, which are sampled for high-value case finding.

The first two populations use stratified simple random sampling without
replacement, implemented with a frozen SHA-256 ordering.  Every sampled row
records its stratum inclusion probability and inverse-probability weight.
The review population is explicitly marked as purposive and must not be used
for population-rate estimates.

Sanitized ``audit_only`` negatives are validated but can never enter the
sample.  This module does not assign a train/validation/test split and does
not authorize training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .r7_build_expansion_manifest import (
    _classify as _recompute_manifest_classification,
)
from .r7_build_expansion_manifest import (
    _validate_qwen_evidence as _validate_manifest_qwen_evidence,
)
from .human_review import (
    R7_ASSIGNMENT_FIELD,
    R7_ASSIGNMENT_SCHEMA,
    R7_MEDIA_FIELD,
    R7_MEDIA_SCHEMA,
    R7_REVIEW_ITEM_DIGEST_FIELDS,
    _review_item_digest as _human_review_item_digest,
    normalize_reviewer_id,
)
from .r7_human_audit_policy import (
    implementation_bundle_payload,
    implementation_bundle_sha256,
    policy_payload,
    policy_sha256,
)
from . import r7_human_audit_policy as _policy_module


SCHEMA_VERSION = "motive-r7-human-audit-sample-v2"
ROW_SCHEMA = "motive-r7-human-audit-sample-row-v2"
LEDGER_ROW_SCHEMA = "motive-r7-human-audit-ledger-row-v2"
SUMMARY_SCHEMA = "motive-r7-human-audit-sample-summary-v2"
DONE_SCHEMA = "motive-r7-human-audit-sample-done-v2"
DESIGN_VERSION = (
    "stratified-srs-media-assignment-policy-sha256-v2"
)

SOURCE_SUMMARY_SCHEMA = "motive-r7-expansion-manifest-v2"
SOURCE_DONE_SCHEMA = "motive-r7-expansion-manifest-done-v2"
SOURCE_ROW_SCHEMA = "motive-r7-expansion-manifest-row-v2"
HUMAN_REVIEW_SCHEMA = "motive-action-human-review-v1"
REVIEW_ITEM_DIGEST_FIELDS = R7_REVIEW_ITEM_DIGEST_FIELDS

POSITIVES_NAME = "positives.jsonl"
NEGATIVES_NAME = "negatives.jsonl"
REVIEW_NAME = "review.jsonl"
SOURCE_SUMMARY_NAME = "summary.json"
SOURCE_DONE_NAME = "done.json"
SAMPLED_MANIFEST_NAME = "sampled_manifest.jsonl"
SECONDARY_MANIFEST_NAME = "secondary_manifest.jsonl"
SAMPLING_LEDGER_NAME = "sampling_ledger.jsonl"
PRIMARY_REVIEW_NAME = "primary_review.blind.jsonl"
SECONDARY_REVIEW_NAME = "secondary_review.blind.jsonl"
REVIEWER_ASSIGNMENTS_NAME = "reviewer_assignments.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"

_SAMPLING_POLICY = policy_payload()["sampling_design"]
DEFAULT_SEED = int(_SAMPLING_POLICY["seed"])
DEFAULT_POSITIVE_SAMPLE = int(
    _SAMPLING_POLICY["positive_sample_target"]
)
DEFAULT_PSEUDO_NEGATIVE_SAMPLE = int(
    _SAMPLING_POLICY["pseudo_negative_sample_target"]
)
DEFAULT_REVIEW_SAMPLE = int(
    _SAMPLING_POLICY["review_sample_target"]
)
DEFAULT_DOUBLE_REVIEW_FRACTION = float(
    _SAMPLING_POLICY["double_review_fraction"]
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_NAMES = frozenset(
    {
        POSITIVES_NAME,
        NEGATIVES_NAME,
        REVIEW_NAME,
        SOURCE_SUMMARY_NAME,
        SOURCE_DONE_NAME,
    }
)
_OUTPUT_NAMES = frozenset(
    {
        SAMPLED_MANIFEST_NAME,
        SECONDARY_MANIFEST_NAME,
        SAMPLING_LEDGER_NAME,
        PRIMARY_REVIEW_NAME,
        SECONDARY_REVIEW_NAME,
        REVIEWER_ASSIGNMENTS_NAME,
        SUMMARY_NAME,
        DONE_NAME,
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _stat_snapshot(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _validated_data_root(data_root: Path) -> Path:
    expanded = data_root.expanduser()
    if expanded.is_symlink():
        raise ValueError("data root must not be a symlink")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def _resolved_media_path(
    *,
    data_root: Path,
    manifest_path: Any,
    context: str,
) -> tuple[Path, str]:
    if (
        not isinstance(manifest_path, str)
        or not manifest_path
        or "\x00" in manifest_path
    ):
        raise ValueError(f"{context} media path is invalid")
    raw = Path(manifest_path).expanduser()
    if ".." in raw.parts:
        raise ValueError(f"{context} media path contains parent traversal")
    lexical_root = Path(os.path.abspath(data_root))
    candidate = raw if raw.is_absolute() else lexical_root / raw
    lexical_candidate = Path(os.path.abspath(candidate))
    resolved = lexical_candidate.resolve(strict=True)
    if resolved == data_root or data_root not in resolved.parents:
        raise ValueError(f"{context} media path escapes data root")
    try:
        relative_lexical = lexical_candidate.relative_to(lexical_root)
    except ValueError:
        raise ValueError(
            f"{context} media path is lexically outside data root"
        ) from None
    current = lexical_root
    for part in relative_lexical.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{context} media path contains a symlink")
    metadata = resolved.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{context} media file must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{context} media path is not a regular file")
    return resolved, resolved.relative_to(data_root).as_posix()


def _stable_media_file_binding(
    *,
    data_root: Path,
    manifest_path: Any,
    context: str,
) -> dict[str, Any]:
    resolved, relative = _resolved_media_path(
        data_root=data_root,
        manifest_path=manifest_path,
        context=context,
    )
    before_path = resolved.lstat()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{context} media file is not regular")
        if (
            before.st_dev != before_path.st_dev
            or before.st_ino != before_path.st_ino
        ):
            raise RuntimeError(f"{context} media changed before hashing")
        hasher = hashlib.sha256()
        byte_count = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            hasher.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = resolved.lstat()
    before_snapshot = _stat_snapshot(before)
    after_snapshot = _stat_snapshot(after)
    if (
        before_snapshot != after_snapshot
        or _stat_snapshot(after_path) != after_snapshot
        or byte_count != before.st_size
    ):
        raise RuntimeError(f"{context} media changed while being hashed")
    re_resolved, re_relative = _resolved_media_path(
        data_root=data_root,
        manifest_path=manifest_path,
        context=context,
    )
    if re_resolved != resolved or re_relative != relative:
        raise RuntimeError(f"{context} media path changed while being hashed")
    return {
        "manifest_path": str(manifest_path),
        "relative_path": relative,
        "sha256": hasher.hexdigest(),
        "bytes": byte_count,
        "stat": before_snapshot,
    }


def build_media_binding(
    row: Mapping[str, Any],
    *,
    data_root: Path | None,
    diagnostic_unbound_media: bool,
) -> dict[str, Any]:
    iid = _iid(row, context="media binding row")
    if data_root is None:
        if not diagnostic_unbound_media:
            raise ValueError(
                "formal sampling requires data_root; use the explicit "
                "diagnostic_unbound_media flag only for diagnostics"
            )
        return {
            "schema_version": R7_MEDIA_SCHEMA,
            "media_bytes_bound": False,
            "data_root": None,
            "src_video": None,
            "tgt_video": None,
        }
    if diagnostic_unbound_media:
        raise ValueError(
            "diagnostic_unbound_media cannot be combined with data_root"
        )
    root = _validated_data_root(data_root)
    return {
        "schema_version": R7_MEDIA_SCHEMA,
        "media_bytes_bound": True,
        "data_root": str(root),
        "src_video": _stable_media_file_binding(
            data_root=root,
            manifest_path=row.get("src_video"),
            context=f"iid={iid} src_video",
        ),
        "tgt_video": _stable_media_file_binding(
            data_root=root,
            manifest_path=row.get("tgt_video"),
            context=f"iid={iid} tgt_video",
        ),
    }


def validate_media_binding(
    row: Mapping[str, Any],
    *,
    expected_data_root: Path | None,
    allow_diagnostic_unbound: bool,
) -> bool:
    iid = _iid(row, context="media-bound row")
    binding = row.get(R7_MEDIA_FIELD)
    if not isinstance(binding, Mapping):
        raise ValueError(f"iid={iid} lacks R7 media binding")
    expected_fields = {
        "schema_version",
        "media_bytes_bound",
        "data_root",
        "src_video",
        "tgt_video",
    }
    if set(binding) != expected_fields:
        raise ValueError(f"iid={iid} R7 media fields differ")
    if binding.get("schema_version") != R7_MEDIA_SCHEMA:
        raise ValueError(f"iid={iid} R7 media schema differs")
    if binding.get("media_bytes_bound") is False:
        if (
            not allow_diagnostic_unbound
            or expected_data_root is not None
            or binding.get("data_root") is not None
            or binding.get("src_video") is not None
            or binding.get("tgt_video") is not None
        ):
            raise ValueError(f"iid={iid} invalid diagnostic media binding")
        return False
    if binding.get("media_bytes_bound") is not True:
        raise ValueError(f"iid={iid} media bound flag differs")
    if expected_data_root is None:
        raise ValueError(f"iid={iid} bound media lacks expected data root")
    root = _validated_data_root(expected_data_root)
    if binding.get("data_root") != str(root):
        raise ValueError(f"iid={iid} media data root differs")
    rebuilt = build_media_binding(
        row,
        data_root=root,
        diagnostic_unbound_media=False,
    )
    if dict(binding) != rebuilt:
        raise ValueError(f"iid={iid} media bytes or provenance differ")
    return True


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} is not a JSON object"
                )
            if line != _canonical_json(value) + "\n":
                raise ValueError(
                    f"{path}:{line_number} is not canonical JSONL"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        _canonical_json(dict(row)) + "\n" for row in rows
    ).encode("utf-8")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _iid(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("iid")
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise ValueError(f"{context} has an invalid IID")
    return value


def _manifest_label(
    row: Mapping[str, Any],
    *,
    expected_bucket: str,
    context: str,
) -> Mapping[str, Any]:
    label = row.get("r7_expansion_manifest")
    if not isinstance(label, Mapping):
        raise ValueError(f"{context} lacks r7_expansion_manifest")
    if label.get("schema_version") != SOURCE_ROW_SCHEMA:
        raise ValueError(f"{context} has an unexpected row schema")
    if label.get("bucket") != expected_bucket:
        raise ValueError(f"{context} bucket differs")
    if label.get("split_assigned") is not False:
        raise ValueError(f"{context} unexpectedly assigns a split")
    if label.get("human_label") is not False:
        raise ValueError(f"{context} unexpectedly asserts a human label")
    if label.get("formal_evidence") is not False:
        raise ValueError(f"{context} unexpectedly asserts formal evidence")
    if "split" in row or "split_provenance" in row:
        raise ValueError(f"{context} contains a quarantined legacy split")
    return label


def _qwen_result(row: Mapping[str, Any], *, context: str) -> Mapping[str, Any]:
    evidence = row.get("qwen_evidence")
    visual = (
        evidence.get("visual")
        if isinstance(evidence, Mapping)
        else None
    )
    result = (
        visual.get("result")
        if isinstance(visual, Mapping)
        else None
    )
    if not isinstance(result, Mapping):
        raise ValueError(f"{context} lacks Qwen visual result")
    return result


def _validate_source_directory(
    source_dir: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    dict[str, Any],
]:
    expanded = source_dir.expanduser()
    if expanded.is_symlink():
        raise ValueError("source manifest directory must not be a symlink")
    root = expanded.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    actual = {entry.name for entry in root.iterdir()}
    if actual != _SOURCE_NAMES:
        raise ValueError(
            "source manifest directory artifact set differs: "
            f"missing={sorted(_SOURCE_NAMES - actual)}, "
            f"extra={sorted(actual - _SOURCE_NAMES)}"
        )
    initial_source_digests: dict[str, str] = {}
    for name in sorted(_SOURCE_NAMES):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"source artifact must be a regular non-symlink file: {name}"
            )
        initial_source_digests[name] = _file_digest(path)
    summary = _load_json(root / SOURCE_SUMMARY_NAME)
    done = _load_json(root / SOURCE_DONE_NAME)
    if summary.get("schema_version") != SOURCE_SUMMARY_SCHEMA:
        raise ValueError("source summary schema differs")
    if done.get("schema_version") != SOURCE_DONE_SCHEMA:
        raise ValueError("source done schema differs")
    if summary.get("status") != "complete" or done.get("status") != "complete":
        raise ValueError("source manifest is not complete")
    semantics = summary.get("semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("source summary lacks semantics")
    required_false = (
        "split_assigned",
        "human_labels_asserted",
        "formal_evidence",
        "production_eligible",
    )
    if any(semantics.get(field) is not False for field in required_false):
        raise ValueError("source semantics do not fail closed")
    for field in (
        "split_assigned",
        "human_labels_asserted",
        "formal_evidence",
    ):
        if done.get(field) is not False:
            raise ValueError(f"source done {field} does not fail closed")

    output_sha = done.get("output_sha256")
    output_rows = done.get("output_rows")
    summary_outputs = summary.get("outputs")
    if not all(
        isinstance(value, Mapping)
        for value in (output_sha, output_rows, summary_outputs)
    ):
        raise ValueError("source artifact bindings are incomplete")
    if (
        _require_sha256(
            output_sha.get(SOURCE_SUMMARY_NAME),
            label="source summary digest",
        )
        != _file_digest(root / SOURCE_SUMMARY_NAME)
    ):
        raise ValueError("source summary digest differs")
    bound_digests = {
        name: _require_sha256(
            output_sha.get(name),
            label=f"source {name} done digest",
        )
        for name in (
            NEGATIVES_NAME,
            POSITIVES_NAME,
            REVIEW_NAME,
            SOURCE_SUMMARY_NAME,
        )
    }
    if done.get("artifact_digest") != _object_digest(bound_digests):
        raise ValueError("source done artifact_digest differs")

    rows_by_bucket: dict[str, list[dict[str, Any]]] = {}
    all_iids: set[str] = set()
    for bucket, name in (
        ("positive", POSITIVES_NAME),
        ("negative", NEGATIVES_NAME),
        ("review", REVIEW_NAME),
    ):
        path = root / name
        if path.is_symlink():
            raise ValueError(f"source artifact must not be a symlink: {name}")
        digest = _file_digest(path)
        expected_done_digest = _require_sha256(
            output_sha.get(name),
            label=f"source {name} done digest",
        )
        summary_binding = summary_outputs.get(name)
        if not isinstance(summary_binding, Mapping):
            raise ValueError(f"source summary lacks {name} binding")
        if (
            digest != expected_done_digest
            or digest
            != _require_sha256(
                summary_binding.get("sha256"),
                label=f"source {name} summary digest",
            )
        ):
            raise ValueError(f"source {name} digest differs")
        rows = _load_canonical_jsonl(path)
        expected_rows = output_rows.get(name)
        if (
            isinstance(expected_rows, bool)
            or not isinstance(expected_rows, int)
            or expected_rows != len(rows)
            or summary_binding.get("rows") != len(rows)
        ):
            raise ValueError(f"source {name} row count differs")
        for line_number, row in enumerate(rows, start=1):
            context = f"{path}:{line_number}"
            iid = _iid(row, context=context)
            if iid in all_iids:
                raise ValueError(f"source buckets duplicate IID={iid}")
            all_iids.add(iid)
            label = _manifest_label(
                row,
                expected_bucket=bucket,
                context=context,
            )
            visual, observation, result = _validate_manifest_qwen_evidence(
                row,
                iid=iid,
            )
            decision = _recompute_manifest_classification(
                visual=visual,
                observation=observation,
                result=result,
            )
            expected_fields = {
                "bucket": decision["bucket"],
                "classification_reason": decision["reason"],
                "verdict": result["verdict"],
                "observation_validated_from": visual[
                    "observation_validated_from"
                ],
                "result_validated_from": visual[
                    "result_validated_from"
                ],
            }
            for field in (
                "action_signature",
                "negative_type",
                "negative_role",
                "quality_failures",
            ):
                if field in decision:
                    expected_fields[field] = decision[field]
            for field, expected in expected_fields.items():
                if label.get(field) != expected:
                    raise ValueError(
                        f"{context} recomputed {field} differs"
                    )
            optional_decision_fields = {
                "action_signature",
                "negative_type",
                "negative_role",
                "quality_failures",
            }
            unexpected = sorted(
                field
                for field in optional_decision_fields
                if field not in decision and field in label
            )
            if unexpected:
                raise ValueError(
                    f"{context} has unexpected classification fields: "
                    f"{unexpected}"
                )
        rows_by_bucket[bucket] = rows

    bucket_counts = summary.get("bucket_counts")
    if not isinstance(bucket_counts, Mapping):
        raise ValueError("source summary lacks bucket counts")
    for bucket, rows in rows_by_bucket.items():
        if bucket_counts.get(bucket) != len(rows):
            raise ValueError(f"source summary {bucket} count differs")
    for name, initial_digest in initial_source_digests.items():
        if _file_digest(root / name) != initial_digest:
            raise RuntimeError(
                f"source artifact changed while being validated: {name}"
            )
    return rows_by_bucket, summary, done


def _stable_rank(seed: int, scope: str, value: str) -> str:
    return hashlib.sha256(
        f"{DESIGN_VERSION}\0{seed}\0{scope}\0{value}".encode("utf-8")
    ).hexdigest()


def _primary_family(row: Mapping[str, Any]) -> str:
    label = row["r7_expansion_manifest"]
    value = label.get("primary_family")
    if not isinstance(value, str) or not value:
        raise ValueError(f"iid={row.get('iid')} lacks primary_family")
    return value


def _confidence(row: Mapping[str, Any]) -> str:
    result = _qwen_result(row, context=f"iid={row.get('iid')}")
    value = result.get("confidence")
    if value not in {"low", "medium", "high"}:
        raise ValueError(f"iid={row.get('iid')} has invalid confidence")
    return str(value)


def _verdict(row: Mapping[str, Any]) -> str:
    result = _qwen_result(row, context=f"iid={row.get('iid')}")
    value = result.get("verdict")
    if not isinstance(value, str) or not value:
        raise ValueError(f"iid={row.get('iid')} has invalid verdict")
    return value


def _result_validation_source(row: Mapping[str, Any]) -> str:
    label = row["r7_expansion_manifest"]
    value = label.get("result_validated_from")
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"iid={row.get('iid')} lacks result validation source"
        )
    return value


def _auto_rule_tier(row: Mapping[str, Any]) -> str:
    rule = row.get("auto_rule")
    value = rule.get("tier") if isinstance(rule, Mapping) else None
    if value not in {"possible", "high"}:
        raise ValueError(f"iid={row.get('iid')} has invalid auto-rule tier")
    return str(value)


def _family_frequency_band(
    row: Mapping[str, Any],
    family_counts: Mapping[str, int],
) -> str:
    count = family_counts[_primary_family(row)]
    return (
        "rare_1_to_10"
        if count <= 10
        else "medium_11_to_49"
        if count <= 49
        else "large_50_plus"
    )


def _positive_stratum(
    row: Mapping[str, Any],
    family_counts: Mapping[str, int],
) -> tuple[str, ...]:
    return (
        _confidence(row),
        _auto_rule_tier(row),
        _family_frequency_band(row, family_counts),
    )


def _negative_stratum(
    row: Mapping[str, Any],
    family_counts: Mapping[str, int],
) -> tuple[str, ...]:
    return (
        _verdict(row),
        _confidence(row),
        _family_frequency_band(row, family_counts),
    )


def _stratum_id(cohort: str, key: Sequence[str]) -> str:
    return _object_digest({"cohort": cohort, "key": list(key)})


def _allocate_with_estimable_minimum(
    populations: Mapping[tuple[str, ...], int],
    *,
    target: int,
) -> dict[tuple[str, ...], int]:
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("sample target must be a positive integer")
    total = sum(populations.values())
    if target > total:
        raise ValueError(
            f"sample target {target} exceeds population {total}"
        )
    keys = sorted(populations)
    # A non-census stratum with one observation has no estimable within-
    # stratum variance.  Census every smaller stratum and require at least
    # five observations from every other pre-registered stratum.
    minimums = {
        key: min(5, populations[key])
        for key in keys
    }
    minimum_total = sum(minimums.values())
    if target < minimum_total:
        raise ValueError(
            f"sample target {target} is smaller than the estimable-strata "
            f"minimum {minimum_total}; increase the target"
        )
    allocations = dict(minimums)
    remaining = target - minimum_total
    while remaining:
        capacities = {
            key: populations[key] - allocations[key]
            for key in keys
            if populations[key] > allocations[key]
        }
        capacity_total = sum(capacities.values())
        if capacity_total < remaining:
            raise AssertionError("stratum allocation capacity underflow")
        exact = {
            key: remaining * capacity / capacity_total
            for key, capacity in capacities.items()
        }
        added = 0
        for key in sorted(capacities):
            increment = min(capacities[key], int(math.floor(exact[key])))
            allocations[key] += increment
            added += increment
        remaining -= added
        if not remaining:
            break
        candidates = sorted(
            (
                (exact[key] - math.floor(exact[key]), key)
                for key in capacities
                if allocations[key] < populations[key]
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not candidates:
            raise AssertionError("stratum allocation stalled")
        for _fraction, key in candidates:
            if not remaining:
                break
            allocations[key] += 1
            remaining -= 1
    if sum(allocations.values()) != target:
        raise AssertionError("stratum allocation total differs")
    return allocations


def _probability_sample(
    rows: Sequence[dict[str, Any]],
    *,
    cohort: str,
    target: int,
    seed: int,
    stratum_function: Any,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    by_stratum: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[stratum_function(row)].append(row)
    populations = {
        key: len(values) for key, values in sorted(by_stratum.items())
    }
    allocations = _allocate_with_estimable_minimum(
        populations,
        target=target,
    )
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    stratum_summary: list[dict[str, Any]] = []
    for key in sorted(by_stratum):
        population_rows = by_stratum[key]
        ordered = sorted(
            population_rows,
            key=lambda row: (
                _stable_rank(seed, f"{cohort}:row", str(row["iid"])),
                str(row["iid"]),
            ),
        )
        sample_size = allocations[key]
        probability = sample_size / len(ordered)
        weight = len(ordered) / sample_size
        sid = _stratum_id(cohort, key)
        for rank, row in enumerate(ordered[:sample_size], start=1):
            selected.append(
                (
                    row,
                    {
                        "sampling_mode": "probability",
                        "estimand": (
                            "pseudo_positive_precision"
                            if cohort == "pseudo_positive"
                            else "pseudo_negative_false_negative_rate"
                        ),
                        "stratum_id": sid,
                        "stratum_key": list(key),
                        "stratum_population": len(ordered),
                        "stratum_sample_size": sample_size,
                        "selection_probability": probability,
                        "design_weight": weight,
                        "within_stratum_rank": rank,
                    },
                )
            )
        stratum_summary.append(
            {
                "stratum_id": sid,
                "stratum_key": list(key),
                "population": len(ordered),
                "sample": sample_size,
                "selection_probability": probability,
                "design_weight": weight,
            }
        )
    return selected, {
        "sampling_mode": "probability",
        "population": len(rows),
        "sample": len(selected),
        "strata": stratum_summary,
    }


def _review_priority(row: Mapping[str, Any]) -> str:
    source = _result_validation_source(row)
    verdict = _verdict(row)
    if source.startswith("repair_"):
        return "schema_repair"
    if verdict in {"valid_action", "valid_suppression"}:
        return "positive_quality_failure"
    if (
        source == "original"
        and verdict in {"endpoint_only", "uncertain"}
    ):
        return "original_endpoint_or_uncertain"
    if source == "fallback_uncertain":
        return "fallback_uncertain"
    return "original_low_confidence_negative"


def _balanced_casefinding_rows(
    rows: Sequence[dict[str, Any]],
    *,
    target: int,
    seed: int,
    scope: str,
) -> list[dict[str, Any]]:
    if target >= len(rows):
        return sorted(
            rows,
            key=lambda row: (
                _stable_rank(seed, scope, str(row["iid"])),
                str(row["iid"]),
            ),
        )
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[_primary_family(row)].append(row)
    for family, values in by_family.items():
        by_family[family] = sorted(
            values,
            key=lambda row: (
                _stable_rank(
                    seed,
                    f"{scope}:{family}",
                    str(row["iid"]),
                ),
                str(row["iid"]),
            ),
        )
    family_order = sorted(
        by_family,
        key=lambda family: (
            _stable_rank(seed, f"{scope}:family", family),
            family,
        ),
    )
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < target:
        progress = False
        for family in family_order:
            values = by_family[family]
            if offset < len(values):
                selected.append(values[offset])
                progress = True
                if len(selected) == target:
                    break
        if not progress:
            raise AssertionError("review case-finding selection stalled")
        offset += 1
    return selected


def _review_casefinding_sample(
    rows: Sequence[dict[str, Any]],
    *,
    target: int,
    seed: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("review target must be a positive integer")
    if target > len(rows):
        raise ValueError("review target exceeds population")
    by_priority: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_priority[_review_priority(row)].append(row)
    order_and_caps = (
        ("positive_quality_failure", 20),
        ("schema_repair", 20),
        ("original_endpoint_or_uncertain", 20),
        ("fallback_uncertain", 20),
        ("original_low_confidence_negative", target),
    )
    remaining = target
    quotas: dict[str, int] = {}
    for category, cap in order_and_caps:
        available = len(by_priority.get(category, ()))
        quota = min(available, cap, remaining)
        quotas[category] = quota
        remaining -= quota
    if remaining:
        # Fill unused capacity in reverse priority order without silently
        # changing the pre-registered category precedence.
        for category, _cap in reversed(order_and_caps):
            available = len(by_priority.get(category, ()))
            extra = min(available - quotas[category], remaining)
            quotas[category] += extra
            remaining -= extra
            if not remaining:
                break
    if remaining:
        raise AssertionError("review selection capacity underflow")

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    category_summary: list[dict[str, Any]] = []
    for category, _cap in order_and_caps:
        population_rows = by_priority.get(category, [])
        chosen = _balanced_casefinding_rows(
            population_rows,
            target=quotas[category],
            seed=seed,
            scope=f"review:{category}",
        )
        for rank, row in enumerate(chosen, start=1):
            selected.append(
                (
                    row,
                    {
                        "sampling_mode": "purposive_casefinding",
                        "estimand": None,
                        "priority_category": category,
                        "category_population": len(population_rows),
                        "category_sample_size": len(chosen),
                        "selection_probability": None,
                        "design_weight": None,
                        "within_category_rank": rank,
                    },
                )
            )
        category_summary.append(
            {
                "priority_category": category,
                "population": len(population_rows),
                "sample": len(chosen),
                "population_inference_allowed": False,
            }
        )
    return selected, {
        "sampling_mode": "purposive_casefinding",
        "population": len(rows),
        "sample": len(selected),
        "categories": category_summary,
        "population_inference_allowed": False,
    }


def _attach_double_review(
    selected: Sequence[tuple[dict[str, Any], dict[str, Any], str]],
    *,
    seed: int,
    fraction: float,
) -> set[str]:
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("double-review fraction must be in [0,1]")
    by_cohort: dict[str, list[str]] = defaultdict(list)
    for row, _metadata, cohort in selected:
        by_cohort[cohort].append(str(row["iid"]))
    chosen: set[str] = set()
    for cohort, iids in sorted(by_cohort.items()):
        count = int(math.floor(len(iids) * fraction + 0.5))
        ordered = sorted(
            iids,
            key=lambda iid: (
                _stable_rank(seed, f"double-review:{cohort}", iid),
                iid,
            ),
        )
        chosen.update(ordered[:count])
    return chosen


def _assignment_core(
    *,
    iid: str,
    slot: str,
    assigned_reviewer_id: str,
    seed: int,
    policy_digest: str,
) -> dict[str, Any]:
    if slot not in {"primary", "secondary"}:
        raise ValueError("review assignment slot differs")
    return {
        "schema_version": R7_ASSIGNMENT_SCHEMA,
        "review_instance_id": _object_digest(
            {
                "schema_version": R7_ASSIGNMENT_SCHEMA,
                "design_version": DESIGN_VERSION,
                "seed": seed,
                "iid": iid,
                "annotator_slot": slot,
                "assigned_reviewer_id": assigned_reviewer_id,
                "policy_sha256": policy_digest,
            }
        ),
        "iid": iid,
        "annotator_slot": slot,
        "assigned_reviewer_id": assigned_reviewer_id,
        "independent_review_required": slot == "secondary",
        "policy_sha256": policy_digest,
    }


def _freeze_assignments(
    *,
    sampled_rows: Sequence[dict[str, Any]],
    double_review_iids: set[str],
    primary_reviewer_id: str,
    secondary_reviewer_id: str,
    seed: int,
    policy_digest: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    primary_cores = [
        _assignment_core(
            iid=str(row["iid"]),
            slot="primary",
            assigned_reviewer_id=primary_reviewer_id,
            seed=seed,
            policy_digest=policy_digest,
        )
        for row in sampled_rows
    ]
    secondary_cores = [
        _assignment_core(
            iid=str(row["iid"]),
            slot="secondary",
            assigned_reviewer_id=secondary_reviewer_id,
            seed=seed,
            policy_digest=policy_digest,
        )
        for row in sampled_rows
        if str(row["iid"]) in double_review_iids
    ]
    cores = [*primary_cores, *secondary_cores]
    assignment_set_digest = _object_digest(
        {
            "schema_version": "motive-r7-review-assignment-set-v1",
            "design_version": DESIGN_VERSION,
            "seed": seed,
            "assignments": cores,
        }
    )

    def committed(core: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **dict(core),
            "assignment_set_digest": assignment_set_digest,
        }

    primary_by_iid = {
        str(core["iid"]): committed(core) for core in primary_cores
    }
    secondary_by_iid = {
        str(core["iid"]): committed(core) for core in secondary_cores
    }
    primary_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    for source in sampled_rows:
        iid = str(source["iid"])
        primary = dict(source)
        primary[R7_ASSIGNMENT_FIELD] = primary_by_iid[iid]
        primary_rows.append(primary)
        if iid in double_review_iids:
            secondary = dict(source)
            secondary[R7_ASSIGNMENT_FIELD] = secondary_by_iid[iid]
            secondary_rows.append(secondary)
    return (
        primary_rows,
        secondary_rows,
        [committed(core) for core in cores],
        assignment_set_digest,
    )


def _review_item_digest(
    row: Mapping[str, Any],
    *,
    context: str,
) -> str:
    return _human_review_item_digest(dict(row), context=context)


def _blind_review_template(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact hint-free template accepted by human_review.merge."""

    iid = _iid(row, context="sampled review row")
    input_digest = _require_sha256(
        row.get("input_digest"),
        label=f"iid={iid} input_digest",
    )
    visible: dict[str, Any] = {}
    for field in ("prompt", "src_video", "tgt_video"):
        value = row.get(field)
        if not isinstance(value, str):
            raise ValueError(f"iid={iid} has invalid {field}")
        visible[field] = value
    template: dict[str, Any] = {
        "schema_version": HUMAN_REVIEW_SCHEMA,
        "iid": iid,
        "input_digest": input_digest,
        "verdict": "",
        "reviewer": "",
        "action_signature": "",
        "event_type": "",
        "actor": "",
        "actor_valid": "",
        "instruction_aligned": "",
        "complete_temporal_event": "",
        "source_action": "",
        "target_action": "",
        "direction": "",
        "speed": "",
        "phase": "",
        "contact_or_interaction": "",
        "camera_motion": "",
        "preservation_ok": "",
        "event_start_frame": "",
        "event_end_frame": "",
        "review_confidence": "",
        "secondary_reviewer": "",
        "adjudication": "",
        "notes": "",
        **visible,
    }
    for field in (R7_ASSIGNMENT_FIELD, R7_MEDIA_FIELD):
        value = row.get(field)
        if not isinstance(value, Mapping):
            raise ValueError(f"iid={iid} lacks immutable {field}")
        template[field] = json.loads(
            json.dumps(value, ensure_ascii=False)
        )
    template["review_item_digest"] = _review_item_digest(
        template,
        context=f"iid={iid}",
    )
    forbidden = {
        "qwen_evidence",
        "auto_rule",
        "r7_expansion_manifest",
        "r7_human_audit_sampling",
        "automation_hints",
        "cohort",
        "bucket",
        "verdict_hint",
    }
    if forbidden & set(template):
        raise AssertionError("blind template leaks automation metadata")
    return template


def _family_coverage_supplement(
    population_rows: Sequence[dict[str, Any]],
    probability_selected: Sequence[
        tuple[dict[str, Any], dict[str, Any]]
    ],
    *,
    cohort: str,
    seed: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    represented = {
        _primary_family(row) for row, _metadata in probability_selected
    }
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in population_rows:
        by_family[_primary_family(row)].append(row)
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing = sorted(set(by_family) - represented)
    for family in missing:
        ordered = sorted(
            by_family[family],
            key=lambda row: (
                _stable_rank(
                    seed,
                    f"{cohort}:family-coverage:{family}",
                    str(row["iid"]),
                ),
                str(row["iid"]),
            ),
        )
        row = ordered[0]
        selected.append(
            (
                row,
                {
                    "sampling_mode": "purposive_family_coverage",
                    "estimand": None,
                    "family": family,
                    "family_population": len(ordered),
                    "selection_probability": None,
                    "design_weight": None,
                },
            )
        )
    return selected, {
        "sampling_mode": "purposive_family_coverage",
        "population_families": len(by_family),
        "families_already_represented_by_probability_sample": len(
            represented
        ),
        "supplemental_families": missing,
        "sample": len(selected),
        "population_inference_allowed": False,
    }


def _atomic_directory(
    output_dir: Path,
    *,
    files: Mapping[str, bytes],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    published: list[Path] = []
    target_created = False
    try:
        for name, payload in files.items():
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        # mkdir is the no-replace publication claim.  Hard links then publish
        # each already-fsynced inode without any overwrite window.  A crash
        # can leave a partial directory, but resume rejects partial state and
        # never repairs it.
        output_dir.mkdir()
        target_created = True
        for name in sorted(files):
            destination = output_dir / name
            os.link(temporary / name, destination)
            published.append(destination)
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        directory_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if target_created:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            output_dir.rmdir()
        raise
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()


def _strict_resume(
    output_dir: Path,
    *,
    expected: Mapping[str, bytes],
) -> None:
    actual_names = {entry.name for entry in output_dir.iterdir()}
    if actual_names != _OUTPUT_NAMES:
        raise ValueError(
            "resume artifact set differs: "
            f"missing={sorted(_OUTPUT_NAMES - actual_names)}, "
            f"extra={sorted(actual_names - _OUTPUT_NAMES)}"
    )
    for name, payload in expected.items():
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"resume artifact is not a regular non-symlink file: {name}"
            )
        if path.read_bytes() != payload:
            raise ValueError(f"resume artifact differs: {name}")


def build_human_audit_sample(
    *,
    source_dir: Path,
    output_dir: Path,
    data_root: Path | None,
    primary_reviewer_id: str,
    secondary_reviewer_id: str,
    expected_implementation_bundle_digest: str | None = None,
    expected_source_artifact_digest: str | None = None,
    expected_source_input_sha256: str | None = None,
    diagnostic_unbound_media: bool = False,
    positive_sample: int = DEFAULT_POSITIVE_SAMPLE,
    pseudo_negative_sample: int = DEFAULT_PSEUDO_NEGATIVE_SAMPLE,
    review_sample: int = DEFAULT_REVIEW_SAMPLE,
    double_review_fraction: float = DEFAULT_DOUBLE_REVIEW_FRACTION,
    seed: int = DEFAULT_SEED,
    resume: bool = False,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(double_review_fraction, bool)
        or not isinstance(double_review_fraction, (int, float))
        or not math.isfinite(float(double_review_fraction))
        or not 0.0 <= float(double_review_fraction) <= 1.0
    ):
        raise ValueError("double-review fraction must be in [0,1]")
    primary_reviewer = normalize_reviewer_id(
        primary_reviewer_id,
        context="primary",
    )
    secondary_reviewer = normalize_reviewer_id(
        secondary_reviewer_id,
        context="secondary",
    )
    if double_review_fraction > 0.0 and (
        primary_reviewer == secondary_reviewer
    ):
        raise ValueError(
            "primary and secondary reviewer IDs must differ when "
            "double review is enabled"
        )
    if data_root is None:
        if not diagnostic_unbound_media:
            raise ValueError(
                "formal sampling requires data_root; pass "
                "diagnostic_unbound_media=True only for diagnostics"
            )
        resolved_data_root: Path | None = None
    else:
        if diagnostic_unbound_media:
            raise ValueError(
                "diagnostic_unbound_media cannot be combined with data_root"
            )
        resolved_data_root = _validated_data_root(data_root)
    frozen_policy = policy_payload()
    frozen_policy_sha256 = policy_sha256()
    sampling_policy = frozen_policy["sampling_design"]
    requested_design = {
        "seed": seed,
        "positive_sample_target": positive_sample,
        "pseudo_negative_sample_target": pseudo_negative_sample,
        "review_sample_target": review_sample,
        "double_review_fraction": float(double_review_fraction),
    }
    frozen_formal_design = {
        key: sampling_policy[key] for key in requested_design
    }
    if (
        resolved_data_root is not None
        and requested_design != frozen_formal_design
    ):
        raise ValueError(
            "formal sampling parameters must exactly equal the immutable "
            f"policy design: expected={frozen_formal_design}, "
            f"requested={requested_design}; custom designs are "
            "diagnostic-only"
        )
    current_implementation_bundle = implementation_bundle_payload()
    current_implementation_bundle_digest = (
        implementation_bundle_sha256()
    )
    if (
        current_implementation_bundle.get("bundle_sha256")
        != current_implementation_bundle_digest
    ):
        raise RuntimeError("current implementation bundle digest differs")
    if expected_implementation_bundle_digest is None:
        if resolved_data_root is not None:
            raise ValueError(
                "formal sampling requires an external expected "
                "implementation bundle digest"
            )
        implementation_bundle_anchor_verified = False
    else:
        expected_bundle_digest = _require_sha256(
            expected_implementation_bundle_digest,
            label="expected implementation bundle digest",
        )
        if expected_bundle_digest != current_implementation_bundle_digest:
            raise ValueError(
                "current implementation bundle differs from the external "
                "expected digest"
            )
        implementation_bundle_anchor_verified = True
    if (
        resolved_data_root is not None
        and (
            expected_source_artifact_digest is None
            or expected_source_input_sha256 is None
        )
    ):
        raise ValueError(
            "formal sampling requires external expected source artifact "
            "and fused-input SHA-256 digests"
        )
    normalized_expected_source_artifact = (
        _require_sha256(
            expected_source_artifact_digest,
            label="expected source artifact digest",
        )
        if expected_source_artifact_digest is not None
        else None
    )
    normalized_expected_source_input = (
        _require_sha256(
            expected_source_input_sha256,
            label="expected source fused-input digest",
        )
        if expected_source_input_sha256 is not None
        else None
    )
    expanded_source = source_dir.expanduser()
    if expanded_source.is_symlink():
        raise ValueError("source manifest directory must not be a symlink")
    source_root = expanded_source.resolve(strict=True)
    expanded_output = output_dir.expanduser()
    if expanded_output.is_symlink():
        raise ValueError("output directory must not be a symlink")
    target = expanded_output.resolve(strict=False)
    if target == source_root or source_root in target.parents:
        raise ValueError("output directory cannot contain source artifacts")
    if resume:
        if not target.is_dir():
            raise FileNotFoundError(
                "--resume is verification-only and requires an output"
            )
    elif target.exists():
        raise FileExistsError(target)

    rows_by_bucket, source_summary, source_done = (
        _validate_source_directory(source_root)
    )
    source_input = source_summary.get("input")
    if not isinstance(source_input, Mapping):
        raise ValueError("source summary lacks fused-input provenance")
    live_source_artifact_digest = _require_sha256(
        source_done.get("artifact_digest"),
        label="live source artifact digest",
    )
    live_source_input_sha256 = _require_sha256(
        source_input.get("sha256"),
        label="live source fused-input digest",
    )
    if source_done.get("input_sha256") != live_source_input_sha256:
        raise ValueError("source summary/done fused-input digest differs")
    if (
        normalized_expected_source_artifact is not None
        and normalized_expected_source_artifact
        != live_source_artifact_digest
    ):
        raise ValueError(
            "live source artifact differs from the external expected digest"
        )
    if (
        normalized_expected_source_input is not None
        and normalized_expected_source_input != live_source_input_sha256
    ):
        raise ValueError(
            "source fused input differs from the external expected digest"
        )
    source_external_anchor_verified = (
        normalized_expected_source_artifact
        == live_source_artifact_digest
        and normalized_expected_source_input
        == live_source_input_sha256
    )
    positives = rows_by_bucket["positive"]
    negative_rows = rows_by_bucket["negative"]
    pseudo_negatives: list[dict[str, Any]] = []
    audit_only = 0
    for row in negative_rows:
        role = row["r7_expansion_manifest"].get("negative_role")
        if role == "pseudo_negative":
            pseudo_negatives.append(row)
        elif role == "audit_only":
            audit_only += 1
        else:
            raise ValueError(f"iid={row.get('iid')} has unknown negative role")
    expected_roles = source_summary.get("negative_role_counts")
    if (
        not isinstance(expected_roles, Mapping)
        or expected_roles.get("pseudo_negative") != len(pseudo_negatives)
        or expected_roles.get("audit_only") != audit_only
    ):
        raise ValueError("source negative-role counts differ")

    positive_family_counts = Counter(
        _primary_family(row) for row in positives
    )
    negative_family_counts = Counter(
        _primary_family(row) for row in pseudo_negatives
    )
    positive_selected, positive_design = _probability_sample(
        positives,
        cohort="pseudo_positive",
        target=positive_sample,
        seed=seed,
        stratum_function=lambda row: _positive_stratum(
            row, positive_family_counts
        ),
    )
    negative_selected, negative_design = _probability_sample(
        pseudo_negatives,
        cohort="pseudo_negative",
        target=pseudo_negative_sample,
        seed=seed,
        stratum_function=lambda row: _negative_stratum(
            row, negative_family_counts
        ),
    )
    positive_supplement, positive_supplement_design = (
        _family_coverage_supplement(
            positives,
            positive_selected,
            cohort="pseudo_positive",
            seed=seed,
        )
    )
    negative_supplement, negative_supplement_design = (
        _family_coverage_supplement(
            pseudo_negatives,
            negative_selected,
            cohort="pseudo_negative",
            seed=seed,
        )
    )
    review_selected, review_design = _review_casefinding_sample(
        rows_by_bucket["review"],
        target=review_sample,
        seed=seed,
    )
    combined: list[tuple[dict[str, Any], dict[str, Any], str]] = [
        *(
            (row, metadata, "pseudo_positive")
            for row, metadata in positive_selected
        ),
        *(
            (row, metadata, "pseudo_negative")
            for row, metadata in negative_selected
        ),
        *(
            (row, metadata, "pseudo_positive_family_coverage")
            for row, metadata in positive_supplement
        ),
        *(
            (row, metadata, "pseudo_negative_family_coverage")
            for row, metadata in negative_supplement
        ),
        *(
            (row, metadata, "priority_review")
            for row, metadata in review_selected
        ),
    ]
    double_review_iids = _attach_double_review(
        combined,
        seed=seed,
        fraction=double_review_fraction,
    )
    ordered = sorted(
        combined,
        key=lambda item: (
            _stable_rank(seed, "blind-review-order", str(item[0]["iid"])),
            str(item[0]["iid"]),
        ),
    )
    sampled_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for sample_order, (source_row, metadata, cohort) in enumerate(
        ordered, start=1
    ):
        iid = str(source_row["iid"])
        sampling = {
            "schema_version": ROW_SCHEMA,
            "design_version": DESIGN_VERSION,
            "seed": seed,
            "cohort": cohort,
            "sample_order": sample_order,
            "double_review": iid in double_review_iids,
            "split_assigned": False,
            "human_label": False,
            "training_eligible": False,
            **metadata,
        }
        row = dict(source_row)
        if any(
            field in row
            for field in (
                "r7_human_audit_sampling",
                R7_ASSIGNMENT_FIELD,
                R7_MEDIA_FIELD,
            )
        ):
            raise ValueError(
                f"iid={iid} already has an R7 human-audit field"
            )
        row["r7_human_audit_sampling"] = sampling
        row[R7_MEDIA_FIELD] = build_media_binding(
            row,
            data_root=resolved_data_root,
            diagnostic_unbound_media=diagnostic_unbound_media,
        )
        sampled_rows.append(row)
        ledger_rows.append(
            {
                "schema_version": LEDGER_ROW_SCHEMA,
                "iid": iid,
                "input_digest": row.get("input_digest"),
                "source_bucket": row["r7_expansion_manifest"]["bucket"],
                **sampling,
            }
        )

    (
        sampled_rows,
        secondary_rows,
        assignment_cores,
        assignment_set_digest,
    ) = _freeze_assignments(
        sampled_rows=sampled_rows,
        double_review_iids=double_review_iids,
        primary_reviewer_id=primary_reviewer,
        secondary_reviewer_id=secondary_reviewer,
        seed=seed,
        policy_digest=frozen_policy_sha256,
    )
    sampled_bytes = _jsonl_bytes(sampled_rows)
    secondary_manifest_bytes = _jsonl_bytes(secondary_rows)
    ledger_bytes = _jsonl_bytes(ledger_rows)
    primary_templates = [
        _blind_review_template(row) for row in sampled_rows
    ]
    secondary_templates = [
        _blind_review_template(row) for row in secondary_rows
    ]
    primary_review_bytes = _jsonl_bytes(primary_templates)
    secondary_review_bytes = _jsonl_bytes(secondary_templates)
    template_by_task = {
        ("primary", str(template["iid"])): (
            PRIMARY_REVIEW_NAME,
            str(template["review_item_digest"]),
        )
        for template in primary_templates
    }
    template_by_task.update(
        {
            ("secondary", str(template["iid"])): (
                SECONDARY_REVIEW_NAME,
                str(template["review_item_digest"]),
            )
            for template in secondary_templates
        }
    )
    reviewer_assignments: list[dict[str, Any]] = []
    for assignment in assignment_cores:
        task = (
            str(assignment["annotator_slot"]),
            str(assignment["iid"]),
        )
        template_name, review_digest = template_by_task[task]
        reviewer_assignments.append(
            {
                **assignment,
                "blind_template": template_name,
                "review_item_digest": review_digest,
            }
        )
    assignments_bytes = _jsonl_bytes(reviewer_assignments)
    implementation_sha256 = _file_digest(Path(__file__).resolve())
    policy_module_sha256 = _file_digest(
        Path(_policy_module.__file__).resolve()
    )
    media_bytes_bound = resolved_data_root is not None
    media_binding_set_digest = _object_digest(
        [
            {
                "iid": row["iid"],
                R7_MEDIA_FIELD: row[R7_MEDIA_FIELD],
            }
            for row in sampled_rows
        ]
    )
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "design_version": DESIGN_VERSION,
        "seed": seed,
        "implementation_sha256": implementation_sha256,
        "policy": frozen_policy,
        "policy_sha256": frozen_policy_sha256,
        "policy_module_sha256": policy_module_sha256,
        "implementation_bundle": current_implementation_bundle,
        "expected_implementation_bundle_digest":
            expected_implementation_bundle_digest,
        "implementation_bundle_external_anchor_verified":
            implementation_bundle_anchor_verified,
        "sampling_design": {
            "mode": (
                "formal_policy_locked"
                if resolved_data_root is not None
                else "diagnostic_customizable"
            ),
            "requested": requested_design,
            "immutable_policy_design": frozen_formal_design,
            "matches_immutable_policy":
                requested_design == frozen_formal_design,
        },
        "source": {
            "directory": str(source_root),
            "summary_sha256": _file_digest(
                source_root / SOURCE_SUMMARY_NAME
            ),
            "done_sha256": _file_digest(source_root / SOURCE_DONE_NAME),
            "artifact_digest": source_done.get("artifact_digest"),
            "fused_input_sha256": live_source_input_sha256,
            "expected_artifact_digest":
                normalized_expected_source_artifact,
            "expected_fused_input_sha256":
                normalized_expected_source_input,
            "external_anchor_verified":
                source_external_anchor_verified,
            "populations": {
                "pseudo_positive": len(positives),
                "pseudo_negative": len(pseudo_negatives),
                "priority_review_source": len(rows_by_bucket["review"]),
                "audit_only_excluded": audit_only,
            },
        },
        "media": {
            "schema_version": "motive-r7-media-commit-v1",
            "mode": (
                "formal_bound"
                if media_bytes_bound
                else "diagnostic_unbound"
            ),
            "data_root": (
                str(resolved_data_root)
                if resolved_data_root is not None
                else None
            ),
            "media_bytes_bound": media_bytes_bound,
            "selected_iids": len(sampled_rows),
            "bound_files": (
                2 * len(sampled_rows) if media_bytes_bound else 0
            ),
            "media_binding_set_digest": media_binding_set_digest,
        },
        "reviewer_assignment": {
            "schema_version": "motive-r7-review-assignment-set-v1",
            "primary_reviewer_id": primary_reviewer,
            "secondary_reviewer_id": secondary_reviewer,
            "assignment_set_digest": assignment_set_digest,
            "assignments": len(reviewer_assignments),
            "core_digest_excludes_assignment_set_digest": True,
            "distinct_assigned_reviewer_ids": (
                primary_reviewer != secondary_reviewer
                if secondary_rows
                else None
            ),
            "cryptographic_reviewer_identity_verified": False,
            "independent_humans_attested": False,
            "external_independent_reviewer_attestation_required":
                bool(secondary_rows),
        },
        "selected": {
            "pseudo_positive_probability": len(positive_selected),
            "pseudo_negative_probability": len(negative_selected),
            "pseudo_positive_family_coverage": len(
                positive_supplement
            ),
            "pseudo_negative_family_coverage": len(
                negative_supplement
            ),
            "priority_review": len(review_selected),
            "total": len(sampled_rows),
            "double_review": len(double_review_iids),
            "review_tasks": len(reviewer_assignments),
        },
        "designs": {
            "pseudo_positive": positive_design,
            "pseudo_negative": negative_design,
            "pseudo_positive_family_coverage":
                positive_supplement_design,
            "pseudo_negative_family_coverage":
                negative_supplement_design,
            "priority_review": review_design,
        },
        "double_review": {
            "fraction": double_review_fraction,
            "selection": "cohort-stratified-frozen-sha256-order",
            "counts": dict(
                sorted(
                    Counter(
                        row["r7_human_audit_sampling"]["cohort"]
                        for row in sampled_rows
                        if row["r7_human_audit_sampling"][
                            "double_review"
                        ]
                    ).items()
                )
            ),
        },
        "outputs": {
            SAMPLED_MANIFEST_NAME: {
                "rows": len(sampled_rows),
                "sha256": _sha256_bytes(sampled_bytes),
                "order": "frozen_randomized_order",
                "reviewer_facing": False,
                "contains_automation_metadata": True,
            },
            SECONDARY_MANIFEST_NAME: {
                "rows": len(secondary_rows),
                "sha256": _sha256_bytes(secondary_manifest_bytes),
                "order":
                    "sampled_manifest_filtered_with_secondary_assignment",
                "reviewer_facing": False,
                "contains_automation_metadata": True,
            },
            SAMPLING_LEDGER_NAME: {
                "rows": len(ledger_rows),
                "sha256": _sha256_bytes(ledger_bytes),
                "order": "matches_sampled_manifest",
            },
            PRIMARY_REVIEW_NAME: {
                "rows": len(primary_templates),
                "sha256": _sha256_bytes(primary_review_bytes),
                "reviewer_facing": True,
                "automation_hints_included": False,
            },
            SECONDARY_REVIEW_NAME: {
                "rows": len(secondary_templates),
                "sha256": _sha256_bytes(secondary_review_bytes),
                "reviewer_facing": True,
                "automation_hints_included": False,
            },
            REVIEWER_ASSIGNMENTS_NAME: {
                "rows": len(reviewer_assignments),
                "sha256": _sha256_bytes(assignments_bytes),
                "primary_tasks": len(primary_templates),
                "secondary_tasks": len(secondary_templates),
            },
        },
        "blind_review_contract": {
            "reviewer_facing_files": [
                PRIMARY_REVIEW_NAME,
                SECONDARY_REVIEW_NAME,
            ],
            "automation_hints_included": False,
            "assignment_set_digest": assignment_set_digest,
            "policy_sha256": frozen_policy_sha256,
            "media_bytes_bound": media_bytes_bound,
            "review_item_digest_fields": list(
                REVIEW_ITEM_DIGEST_FIELDS
            ),
            "secondary_rows_are_distinct_review_instances": True,
            "secondary_must_be_reviewed_independently_before_adjudication":
                True,
            "merge_manifests": {
                "primary": SAMPLED_MANIFEST_NAME,
                "secondary": SECONDARY_MANIFEST_NAME,
            },
        },
        "semantics": {
            "probability_estimands": [
                "pseudo_positive_precision",
                "pseudo_negative_false_negative_rate",
            ],
            "family_coverage_population_inference_allowed": False,
            "priority_review_population_inference_allowed": False,
            "audit_only_rows_sampled": 0,
            "split_assigned": False,
            "human_labels_asserted": False,
            "formal_evidence": False,
            "formal_gate_input_eligible": (
                media_bytes_bound
                and implementation_bundle_anchor_verified
                and source_external_anchor_verified
                and requested_design == frozen_formal_design
            ),
            "label_scope": "rate_audit_only",
            "direct_training_supervision_allowed": False,
            "training_authorized": False,
        },
        "threat_model": {
            "name":
                "controlled_immutable_storage_without_active_concurrent_path_swap",
            "active_concurrent_writer_resistant": False,
        },
    }
    summary_bytes = _pretty_bytes(summary)
    done: dict[str, Any] = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "design_version": DESIGN_VERSION,
        "implementation_sha256": implementation_sha256,
        "policy": frozen_policy,
        "policy_sha256": frozen_policy_sha256,
        "policy_module_sha256": policy_module_sha256,
        "implementation_bundle": current_implementation_bundle,
        "expected_implementation_bundle_digest":
            expected_implementation_bundle_digest,
        "implementation_bundle_external_anchor_verified":
            implementation_bundle_anchor_verified,
        "sampling_design": {
            "mode": (
                "formal_policy_locked"
                if resolved_data_root is not None
                else "diagnostic_customizable"
            ),
            "requested": requested_design,
            "immutable_policy_design": frozen_formal_design,
            "matches_immutable_policy":
                requested_design == frozen_formal_design,
        },
        "source_artifact_digest": source_done.get("artifact_digest"),
        "source_input_sha256": live_source_input_sha256,
        "expected_source_artifact_digest":
            normalized_expected_source_artifact,
        "expected_source_input_sha256":
            normalized_expected_source_input,
        "source_external_anchor_verified":
            source_external_anchor_verified,
        "source_summary_sha256": _file_digest(
            source_root / SOURCE_SUMMARY_NAME
        ),
        "source_done_sha256": _file_digest(
            source_root / SOURCE_DONE_NAME
        ),
        "media_bytes_bound": media_bytes_bound,
        "data_root": (
            str(resolved_data_root)
            if resolved_data_root is not None
            else None
        ),
        "media_binding_set_digest": media_binding_set_digest,
        "assignment_set_digest": assignment_set_digest,
        "primary_reviewer_id": primary_reviewer,
        "secondary_reviewer_id": secondary_reviewer,
        "rows": len(sampled_rows),
        "output_sha256": {
            SAMPLED_MANIFEST_NAME: _sha256_bytes(sampled_bytes),
            SECONDARY_MANIFEST_NAME: _sha256_bytes(
                secondary_manifest_bytes
            ),
            SAMPLING_LEDGER_NAME: _sha256_bytes(ledger_bytes),
            PRIMARY_REVIEW_NAME: _sha256_bytes(primary_review_bytes),
            SECONDARY_REVIEW_NAME: _sha256_bytes(
                secondary_review_bytes
            ),
            REVIEWER_ASSIGNMENTS_NAME: _sha256_bytes(
                assignments_bytes
            ),
            SUMMARY_NAME: _sha256_bytes(summary_bytes),
        },
        "artifact_digest": _object_digest(
            {
                SAMPLED_MANIFEST_NAME: _sha256_bytes(sampled_bytes),
                SECONDARY_MANIFEST_NAME: _sha256_bytes(
                    secondary_manifest_bytes
                ),
                SAMPLING_LEDGER_NAME: _sha256_bytes(ledger_bytes),
                PRIMARY_REVIEW_NAME: _sha256_bytes(
                    primary_review_bytes
                ),
                SECONDARY_REVIEW_NAME: _sha256_bytes(
                    secondary_review_bytes
                ),
                REVIEWER_ASSIGNMENTS_NAME: _sha256_bytes(
                    assignments_bytes
                ),
                SUMMARY_NAME: _sha256_bytes(summary_bytes),
            }
        ),
        "audit_only_rows_sampled": 0,
        "split_assigned": False,
        "human_labels_asserted": False,
        "formal_evidence": False,
        "formal_gate_input_eligible": (
            media_bytes_bound
            and implementation_bundle_anchor_verified
            and source_external_anchor_verified
            and requested_design == frozen_formal_design
        ),
        "label_scope": "rate_audit_only",
        "direct_training_supervision_allowed": False,
        "training_authorized": False,
        "threat_model": {
            "name":
                "controlled_immutable_storage_without_active_concurrent_path_swap",
            "active_concurrent_writer_resistant": False,
        },
    }
    done_bytes = _pretty_bytes(done)
    files = {
        SAMPLED_MANIFEST_NAME: sampled_bytes,
        SECONDARY_MANIFEST_NAME: secondary_manifest_bytes,
        SAMPLING_LEDGER_NAME: ledger_bytes,
        PRIMARY_REVIEW_NAME: primary_review_bytes,
        SECONDARY_REVIEW_NAME: secondary_review_bytes,
        REVIEWER_ASSIGNMENTS_NAME: assignments_bytes,
        SUMMARY_NAME: summary_bytes,
        DONE_NAME: done_bytes,
    }
    if resume:
        _strict_resume(target, expected=files)
    else:
        _atomic_directory(target, files=files)
    returned = dict(summary)
    returned["resume_verified"] = bool(resume)
    return returned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the provenance-bound R7 human-audit sample"
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "required for formal sampling; every selected source/target "
            "video is contained and SHA-256 bound under this root"
        ),
    )
    parser.add_argument("--primary-reviewer-id", required=True)
    parser.add_argument("--secondary-reviewer-id", required=True)
    parser.add_argument(
        "--expected-implementation-bundle-digest",
        help=(
            "externally recorded current implementation-bundle SHA-256; "
            "required for formal sampling"
        ),
    )
    parser.add_argument(
        "--expected-source-artifact-digest",
        help=(
            "externally recorded source manifest-v2 done artifact digest; "
            "required for formal sampling"
        ),
    )
    parser.add_argument(
        "--expected-source-input-sha256",
        help=(
            "externally recorded SHA-256 of the fused input committed by "
            "the source manifest; required for formal sampling"
        ),
    )
    parser.add_argument(
        "--diagnostic-unbound-media",
        action="store_true",
        help=(
            "explicit diagnostic-only escape hatch; incompatible with "
            "--data-root and never eligible for a formal gate"
        ),
    )
    parser.add_argument(
        "--positive-sample",
        type=int,
        default=DEFAULT_POSITIVE_SAMPLE,
    )
    parser.add_argument(
        "--pseudo-negative-sample",
        type=int,
        default=DEFAULT_PSEUDO_NEGATIVE_SAMPLE,
    )
    parser.add_argument(
        "--review-sample",
        type=int,
        default=DEFAULT_REVIEW_SAMPLE,
    )
    parser.add_argument(
        "--double-review-fraction",
        type=float,
        default=DEFAULT_DOUBLE_REVIEW_FRACTION,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate exact existing outputs; never repair or overwrite",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_human_audit_sample(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        data_root=args.data_root,
        primary_reviewer_id=args.primary_reviewer_id,
        secondary_reviewer_id=args.secondary_reviewer_id,
        expected_implementation_bundle_digest=(
            args.expected_implementation_bundle_digest
        ),
        expected_source_artifact_digest=(
            args.expected_source_artifact_digest
        ),
        expected_source_input_sha256=args.expected_source_input_sha256,
        diagnostic_unbound_media=args.diagnostic_unbound_media,
        positive_sample=args.positive_sample,
        pseudo_negative_sample=args.pseudo_negative_sample,
        review_sample=args.review_sample,
        double_review_fraction=args.double_review_fraction,
        seed=args.seed,
        resume=args.resume,
    )
    print(_canonical_json(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
