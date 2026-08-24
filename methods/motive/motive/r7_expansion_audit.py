"""Strict, read-only post-hoc audit for the R7 expansion commit pair.

The auditor consumes the immutable outputs of ``r7_qwen_merge``
(``fused_v2``) and ``r7_build_expansion_manifest`` (``manifest_v2``).
It independently checks their artifact/hash chains and re-derives the
manifest classification.  It never edits either input directory.

An optional JSON report is committed atomically without overwrite.
``--resume`` is verification-only and requires that report to exist before
either input is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import qwen_filter as qwen_filter_module
from .qwen_filter import (
    OBSERVATION_SCHEMA_VERSION,
    VISUAL_SCHEMA_VERSION,
    _validate_observation,
    _validate_visual,
)


AUDIT_SCHEMA = "motive-r7-expansion-posthoc-audit-v1"
FUSED_SUMMARY_SCHEMA = "motive-r7-qwen-visual-merge-v2"
FUSED_DONE_SCHEMA = "motive-r7-qwen-visual-merge-done-v2"
MANIFEST_SUMMARY_SCHEMA = "motive-r7-expansion-manifest-v2"
MANIFEST_DONE_SCHEMA = "motive-r7-expansion-manifest-done-v2"
MANIFEST_ROW_SCHEMA = "motive-r7-expansion-manifest-row-v2"
MANIFEST_POLICY = "r7-strict-qwen-pseudolabel-v2"
SELECTION_SCHEMA = "motive-r7-expansion-selection-v1"
LEGACY_SPLIT_POLICY = "r7-legacy-caption-or-path-split-quarantine-v1"
LEGACY_SPLIT_VALUES = frozenset({"train", "validation", "test"})
LEGACY_SPLIT_PROVENANCE = {
    "seed": 260108828,
    "version": "caption-or-path-fallback-v1",
}
PUBLIC_LEGACY_SPLIT_FIELDS = (
    "removed",
    "removed_by_builder",
    "quarantine_stage",
    "canonical_sha256",
    "selection_upstream_attestation",
    "source_top_level_fields_removed",
    "quarantine_policy_version",
)

FUSED_ARTIFACTS = ("fused.jsonl", "summary.json", "done.json")
BUCKET_FILES = {
    "positive": "positives.jsonl",
    "negative": "negatives.jsonl",
    "review": "review.jsonl",
}
MANIFEST_ARTIFACTS = (
    "positives.jsonl",
    "negatives.jsonl",
    "review.jsonl",
    "summary.json",
    "done.json",
)

POSITIVE_VERDICTS = frozenset({"valid_action", "valid_suppression"})
NEGATIVE_VERDICTS = frozenset(
    {
        "endpoint_only",
        "appearance_only",
        "camera_motion",
        "background_motion",
        "static",
        "instruction_mismatch",
        "artifact",
    }
)
VISIBLE_TARGET_MOTION = frozenset({"clear", "weak"})
ACCEPTED_CONFIDENCE = frozenset({"medium", "high"})
POSITIVE_QUALITY = {
    "camera_dominance": "low",
    "background_dominance": "low",
    "artifact_level": "low",
    "preservation_quality": "acceptable",
}

# Fixed and exhaustive.  The final bin intentionally exposes out-of-range
# rule scores rather than silently folding them into [0, 1].
SCORE_BINS = (
    ("lt_0.60", None, 0.60, False),
    ("0.60_to_lt_0.70", 0.60, 0.70, False),
    ("0.70_to_lt_0.80", 0.70, 0.80, False),
    ("0.80_to_lt_0.90", 0.80, 0.90, False),
    ("0.90_to_1.00", 0.90, 1.00, True),
    ("gt_1.00", 1.00, None, False),
)
REQUIRED_QWEN_SHARDS = 8
QWEN_PARTITION_VERSION = "line_modulo_v1"
QWEN_SHARD_MARKER_SCHEMA = "motive-qwen-shard-manifest-v2"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _object_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _report_bytes(value: Mapping[str, Any]) -> bytes:
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


def _auditor_implementation() -> dict[str, Any]:
    paths = {
        "r7_expansion_audit.py": Path(__file__).resolve(strict=True),
        "qwen_filter.py": Path(
            qwen_filter_module.__file__
        ).resolve(strict=True),
    }
    files = {
        name: {"sha256": _sha256_file(path)}
        for name, path in sorted(paths.items())
    }
    return {
        "files": files,
        "bundle_sha256": _object_digest(
            {name: value["sha256"] for name, value in files.items()}
        ),
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} is not UTF-8") from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{context} is not strict JSON: {error}") from error


def _load_json(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = _parse_json(raw, context=context)
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain one JSON object")
    return value, raw


def _load_jsonl(
    path: Path,
    *,
    context: str,
    allow_empty: bool,
) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"{context} must end with a newline")
    if not raw:
        if allow_empty:
            return [], raw
        raise ValueError(f"{context} is empty")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(
                f"{context}:{line_number} contains a blank JSONL line"
            )
        value = _parse_json(
            line,
            context=f"{context}:{line_number}",
        )
        if not isinstance(value, dict):
            raise ValueError(
                f"{context}:{line_number} is not a JSON object"
            )
        rows.append(value)
    return rows, raw


def _strict_directory(
    raw_path: Path,
    *,
    expected_names: Sequence[str],
    description: str,
) -> tuple[Path, dict[str, Path]]:
    expanded = raw_path.expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise FileNotFoundError(
            f"{description} must be a regular non-symlink directory: "
            f"{expanded}"
        )
    directory = expanded.resolve(strict=True)
    expected = set(expected_names)
    entries = list(directory.iterdir())
    actual = {entry.name for entry in entries}
    if actual != expected:
        raise ValueError(
            f"{description} artifact set mismatch: "
            f"missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    artifacts: dict[str, Path] = {}
    for name in expected_names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"{description} artifact must be a regular non-symlink "
                f"file: {path}"
            )
        artifacts[name] = path
    return directory, artifacts


def _expect_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _expect_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _expect_int(value: Any, *, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _expect_false(value: Any, *, context: str) -> None:
    if value is not False:
        raise ValueError(f"{context} must be the JSON boolean false")


def _expect_sha(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _same_json(left: Any, right: Any) -> bool:
    return _canonical_bytes(left) == _canonical_bytes(right)


def _require_same(left: Any, right: Any, *, context: str) -> None:
    if not _same_json(left, right):
        raise ValueError(f"{context} mismatch")


def _iid(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("iid")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} has no non-empty IID")
    if value != value.strip() or "\x00" in value:
        raise ValueError(f"{context} has a non-canonical IID")
    return value


def _primary_family(row: Mapping[str, Any], *, iid: str) -> str:
    selection = row.get("r7_expansion_selection")
    if isinstance(selection, Mapping):
        value = selection.get("primary_family")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    rule = row.get("auto_rule")
    if isinstance(rule, Mapping):
        values = rule.get("action_families")
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()
    return "unknown"


def _expected_legacy_split_quarantine(
    row: Mapping[str, Any],
    *,
    iid: str,
) -> dict[str, Any]:
    selection = _expect_mapping(
        row.get("r7_expansion_selection"),
        context=f"iid={iid} r7_expansion_selection",
    )
    has_split = "split" in row
    has_provenance = "split_provenance" in row
    upstream_present = "legacy_split_quarantine" in selection
    if upstream_present:
        upstream = selection.get("legacy_split_quarantine")
        if type(upstream) is not dict or set(upstream) != {
            "present",
            "canonical_sha256",
        }:
            raise ValueError(
                f"iid={iid} upstream legacy quarantine is non-canonical"
            )
        present = upstream.get("present")
        canonical_sha = upstream.get("canonical_sha256")
        if type(present) is not bool:
            raise ValueError(
                f"iid={iid} upstream legacy present is not boolean"
            )
        if has_split or has_provenance:
            raise ValueError(
                f"iid={iid} has upstream quarantine and top-level split"
            )
        if present:
            canonical_sha = _expect_sha(
                canonical_sha,
                context=f"iid={iid} upstream legacy canonical SHA",
            )
            return {
                "removed": True,
                "removed_by_builder": False,
                "quarantine_stage": "selection_upstream",
                "canonical_sha256": canonical_sha,
                "selection_upstream_attestation": True,
                "source_top_level_fields_removed": [
                    "split",
                    "split_provenance",
                ],
                "quarantine_policy_version": LEGACY_SPLIT_POLICY,
            }
        if canonical_sha is not None:
            raise ValueError(
                f"iid={iid} absent upstream legacy pair has a digest"
            )
        return {
            "removed": False,
            "removed_by_builder": False,
            "quarantine_stage": "none",
            "canonical_sha256": None,
            "selection_upstream_attestation": True,
            "source_top_level_fields_removed": [],
            "quarantine_policy_version": LEGACY_SPLIT_POLICY,
        }

    if has_split != has_provenance:
        raise ValueError(f"iid={iid} has a partial top-level legacy split pair")
    if not has_split:
        return {
            "removed": False,
            "removed_by_builder": False,
            "quarantine_stage": "none",
            "canonical_sha256": None,
            "selection_upstream_attestation": False,
            "source_top_level_fields_removed": [],
            "quarantine_policy_version": LEGACY_SPLIT_POLICY,
        }
    split = row.get("split")
    provenance = row.get("split_provenance")
    if type(split) is not str or split not in LEGACY_SPLIT_VALUES:
        raise ValueError(f"iid={iid} legacy split value is non-canonical")
    if (
        type(provenance) is not dict
        or set(provenance) != set(LEGACY_SPLIT_PROVENANCE)
        or type(provenance.get("seed")) is not int
        or provenance.get("seed") != LEGACY_SPLIT_PROVENANCE["seed"]
        or type(provenance.get("version")) is not str
        or provenance.get("version")
        != LEGACY_SPLIT_PROVENANCE["version"]
    ):
        raise ValueError(f"iid={iid} legacy split provenance is non-canonical")
    canonical_provenance = dict(LEGACY_SPLIT_PROVENANCE)
    pair = {
        "split": split,
        "split_provenance": canonical_provenance,
    }
    return {
        "removed": True,
        "removed_by_builder": True,
        "quarantine_stage": "builder_legacy",
        "canonical_sha256": _object_digest(pair),
        "selection_upstream_attestation": False,
        "source_top_level_fields_removed": [
            "split",
            "split_provenance",
        ],
        "legacy_split_value": split,
        "legacy_split_provenance": canonical_provenance,
        "legacy_split_provenance_sha256": _object_digest(
            canonical_provenance
        ),
        "legacy_split_pair_sha256": _object_digest(pair),
        "quarantine_policy_version": LEGACY_SPLIT_POLICY,
    }


def _visual_fields(
    row: Mapping[str, Any],
    *,
    iid: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    evidence = _expect_mapping(
        row.get("qwen_evidence"),
        context=f"iid={iid} qwen_evidence",
    )
    visual = _expect_mapping(
        evidence.get("visual"),
        context=f"iid={iid} qwen_evidence.visual",
    )
    if visual.get("iid") != iid:
        raise ValueError(f"iid={iid} Qwen evidence IID mismatch")
    if visual.get("status") != "ok" or visual.get("mode") != "visual":
        raise ValueError(f"iid={iid} Qwen visual evidence is not successful")
    observation = _expect_mapping(
        visual.get("observation"),
        context=f"iid={iid} Qwen observation",
    )
    result = _expect_mapping(
        visual.get("result"),
        context=f"iid={iid} Qwen result",
    )
    input_digest = _expect_sha(
        row.get("input_digest"),
        context=f"iid={iid} input_digest",
    )
    if visual.get("input_digest") != input_digest:
        raise ValueError(f"iid={iid} Qwen input digest mismatch")
    try:
        _validate_observation(dict(observation))
        _validate_visual(dict(result), observation=dict(observation))
    except (TypeError, ValueError) as error:
        raise ValueError(f"iid={iid} Qwen schema validation failed") from error
    if visual.get("observation_digest") != _object_digest(observation):
        raise ValueError(f"iid={iid} Qwen observation digest mismatch")
    if visual.get("result_digest") != _object_digest(result):
        raise ValueError(f"iid={iid} Qwen result digest mismatch")
    for field in (
        "visual_input_digest",
        "run_config_digest",
        "config_digest",
        "implementation_digest",
        "execution_manifest_sha256",
    ):
        _expect_sha(
            visual.get(field),
            context=f"iid={iid} Qwen {field}",
        )
    for field in (
        "model_revision",
        "transformers_version",
        "execution_manifest",
    ):
        _expect_string(
            visual.get(field),
            context=f"iid={iid} Qwen {field}",
        )
    shard_index = visual.get("execution_shard_index")
    shard_count = visual.get("execution_shard_count")
    if (
        type(shard_count) is not int
        or shard_count != REQUIRED_QWEN_SHARDS
        or type(shard_index) is not int
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError(f"iid={iid} Qwen shard provenance is invalid")
    return visual, observation, result


def _validation_source(
    visual: Mapping[str, Any],
    *,
    stage: str,
    iid: str,
) -> str:
    source = _expect_string(
        visual.get(f"{stage}_validated_from"),
        context=f"iid={iid} {stage}_validated_from",
    )
    repairs_field = (
        "observation_repairs" if stage == "observation" else "alignment_repairs"
    )
    repairs = visual.get(repairs_field)
    if not isinstance(repairs, list) or not all(
        isinstance(item, Mapping) for item in repairs
    ):
        raise ValueError(f"iid={iid} {repairs_field} must be a list of objects")
    if stage == "result" and any(
        item.get("authoritative_context_digest")
        != _object_digest(visual["observation"])
        for item in repairs
    ):
        raise ValueError(f"iid={iid} result repair context digest mismatch")
    if source == "original":
        if repairs:
            raise ValueError(f"iid={iid} original {stage} has repairs")
        return source
    if source == "original_sanitized":
        if stage != "result" or not any(
            item.get("attempt") == 0
            and item.get("status") == "ok"
            and item.get("repair_generation_called") is False
            for item in repairs
        ):
            raise ValueError(
                f"iid={iid} original_sanitized provenance is invalid"
            )
        return source
    if source.startswith("repair_"):
        suffix = source.removeprefix("repair_")
        if (
            not suffix.isdigit()
            or int(suffix) < 1
            or not any(
                item.get("attempt") == int(suffix)
                and item.get("status") == "ok"
                and item.get("repair_generation_called") is True
                for item in repairs
            )
        ):
            raise ValueError(f"iid={iid} {stage} repair provenance is invalid")
        return source
    if source == "fallback_uncertain":
        fallback = visual.get(f"{stage}_fallback")
        digest = _object_digest(visual[stage if stage == "observation" else "result"])
        if (
            not isinstance(fallback, Mapping)
            or fallback.get("fallback_digest") != digest
        ):
            raise ValueError(
                f"iid={iid} {stage} fallback provenance is invalid"
            )
        if stage == "result" and (
            fallback.get("authoritative_context_digest")
            != _object_digest(visual["observation"])
        ):
            raise ValueError(
                f"iid={iid} result fallback context digest mismatch"
            )
        return source
    raise ValueError(f"iid={iid} unsupported {stage} validation source")


def _positive_quality_failures(
    observation: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    motion = observation.get("target_actor_motion")
    if motion not in VISIBLE_TARGET_MOTION:
        failures.append(f"target_actor_motion={motion}")
    for field, expected in POSITIVE_QUALITY.items():
        actual = observation.get(field)
        if actual != expected:
            failures.append(f"{field}={actual}")
    confidence = result.get("confidence")
    if confidence not in ACCEPTED_CONFIDENCE:
        failures.append(f"confidence={confidence}")
    if observation.get("uncertainty_codes"):
        failures.append("observation_uncertainty")
    if result.get("uncertainty_codes"):
        failures.append("result_uncertainty")
    return failures


def _negative_quality_failures(
    observation: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    confidence = result.get("confidence")
    if confidence not in ACCEPTED_CONFIDENCE:
        failures.append(f"confidence={confidence}")
    if observation.get("uncertainty_codes"):
        failures.append("observation_uncertainty")
    if result.get("uncertainty_codes"):
        failures.append("result_uncertainty")
    return failures


def _expected_decision(
    *,
    observation_source: str,
    result_source: str,
    observation: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    verdict = _expect_string(result.get("verdict"), context="Qwen verdict")
    if observation_source != "original":
        return {
            "bucket": "review",
            "classification_reason": (
                f"observation_validation_source:{observation_source}"
            ),
        }
    if result_source == "original_sanitized":
        if verdict in NEGATIVE_VERDICTS:
            return {
                "bucket": "negative",
                "classification_reason": (
                    "deterministic_sanitized_audit_negative"
                ),
                "negative_type": verdict,
                "negative_role": "audit_only",
            }
        return {
            "bucket": "review",
            "classification_reason": (
                f"sanitized_nonnegative_verdict:{verdict}"
            ),
        }
    if result_source != "original":
        return {
            "bucket": "review",
            "classification_reason": (
                f"result_validation_source:{result_source}"
            ),
        }
    if verdict == "uncertain":
        return {
            "bucket": "review",
            "classification_reason": "verdict:uncertain",
        }
    if verdict in POSITIVE_VERDICTS:
        failures = _positive_quality_failures(observation, result)
        if failures:
            return {
                "bucket": "review",
                "classification_reason": "positive_quality_gate_failed",
                "quality_failures": failures,
            }
        return {
            "bucket": "positive",
            "classification_reason": (
                "strict_original_qwen_pseudo_positive"
            ),
            "action_signature": _expect_string(
                result.get("action_signature"),
                context="positive Qwen action_signature",
            ),
        }
    if verdict in NEGATIVE_VERDICTS:
        failures = _negative_quality_failures(observation, result)
        if failures:
            return {
                "bucket": "review",
                "classification_reason": "negative_trust_gate_failed",
                "quality_failures": failures,
            }
        return {
            "bucket": "negative",
            "classification_reason": "trusted_original_qwen_negative",
            "negative_type": verdict,
            "negative_role": "pseudo_negative",
        }
    raise ValueError(f"unsupported validated Qwen verdict: {verdict}")


def _score_bin(score: float) -> str:
    for name, lower, upper, upper_inclusive in SCORE_BINS:
        if lower is not None:
            if name == "gt_1.00":
                if score <= lower:
                    continue
            elif score < lower:
                continue
        if upper is not None:
            outside_upper = (
                score > upper if upper_inclusive else score >= upper
            )
            if outside_upper:
                continue
        return name
    raise AssertionError(f"score did not enter an exhaustive bin: {score}")


def _normalize_signature(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    pieces: list[str] = []
    previous_space = True
    for character in normalized:
        if character.isalnum():
            pieces.append(character)
            previous_space = False
        elif not previous_space:
            pieces.append(" ")
            previous_space = True
    return "".join(pieces).strip()


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _nested_counter_dict(
    values: Mapping[str, Counter[str]],
) -> dict[str, dict[str, int]]:
    return {
        key: _counter_dict(counter)
        for key, counter in sorted(values.items())
    }


def _three_dimensional_counts(
    values: Mapping[str, Mapping[str, Counter[str]]],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        outer: {
            middle: _counter_dict(counter)
            for middle, counter in sorted(middles.items())
        }
        for outer, middles in sorted(values.items())
    }


def _signature_summary(
    counter: Counter[str],
    *,
    normalization: str,
) -> dict[str, Any]:
    total = sum(counter.values())
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    coverage: dict[str, dict[str, Any]] = {}
    for top_k in (1, 5, 10, 20):
        covered = sum(count for _, count in ranked[:top_k])
        coverage[f"top_{top_k}"] = {
            "rows": covered,
            "fraction": 0.0 if total == 0 else covered / total,
        }
    return {
        "normalization": normalization,
        "rows": total,
        "unique": len(counter),
        "top_coverage": coverage,
        "top_signatures": [
            {"signature": signature, "rows": count}
            for signature, count in ranked[:20]
        ],
    }


def _verify_fused_summary_semantics(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    if summary.get("partition_version") != QWEN_PARTITION_VERSION:
        raise ValueError("fused_v2 partition version mismatch")
    if summary.get("shard_marker_schema") != QWEN_SHARD_MARKER_SCHEMA:
        raise ValueError("fused_v2 shard marker schema mismatch")
    if (
        type(summary.get("shard_count")) is not int
        or summary.get("shard_count") != REQUIRED_QWEN_SHARDS
    ):
        raise ValueError("fused_v2 must attest exactly eight shards")
    _expect_string(
        summary.get("qwen_root"),
        context="fused_v2 qwen_root",
    )

    verdict_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    verdict_family: dict[str, Counter[str]] = {}
    fallback_counts: Counter[str] = Counter(
        {"observation": 0, "result": 0}
    )
    repair_counts: Counter[str] = Counter(
        {
            "observation_rows": 0,
            "observation_attempts": 0,
            "alignment_rows": 0,
            "alignment_attempts": 0,
        }
    )
    validation_counts: Counter[str] = Counter()
    repair_generation_counts: Counter[str] = Counter()
    sanitization_counts: Counter[str] = Counter()
    shard_rows: Counter[int] = Counter()
    shard_configs: dict[int, set[str]] = {
        index: set() for index in range(REQUIRED_QWEN_SHARDS)
    }
    shard_manifests: dict[int, set[str]] = {
        index: set() for index in range(REQUIRED_QWEN_SHARDS)
    }
    shard_manifest_paths: dict[int, set[str]] = {
        index: set() for index in range(REQUIRED_QWEN_SHARDS)
    }
    contracts: set[bytes] = set()
    run_configs: set[str] = set()

    for line_number, row in enumerate(rows, start=1):
        iid = _iid(row, context=f"fused_v2 line {line_number}")
        visual, _observation, result = _visual_fields(row, iid=iid)
        observation_source = _validation_source(
            visual,
            stage="observation",
            iid=iid,
        )
        result_source = _validation_source(
            visual,
            stage="result",
            iid=iid,
        )
        shard_index = int(visual["execution_shard_index"])
        if shard_index != (line_number - 1) % REQUIRED_QWEN_SHARDS:
            raise ValueError(
                f"iid={iid} violates line-modulo shard partition"
            )
        shard_rows[shard_index] += 1
        shard_configs[shard_index].add(str(visual["config_digest"]))
        shard_manifests[shard_index].add(
            str(visual["execution_manifest_sha256"])
        )
        shard_manifest_paths[shard_index].add(
            str(visual["execution_manifest"])
        )
        run_configs.add(str(visual["run_config_digest"]))
        contract = {
            field: str(visual[field])
            for field in (
                "implementation_digest",
                "model_revision",
                "transformers_version",
                "mode",
            )
        }
        contracts.add(_canonical_bytes(contract))

        verdict = str(result["verdict"])
        family = _primary_family(row, iid=iid)
        verdict_counts[verdict] += 1
        family_counts[family] += 1
        verdict_family.setdefault(family, Counter())[verdict] += 1
        fallback_counts["observation"] += int(
            observation_source == "fallback_uncertain"
        )
        fallback_counts["result"] += int(
            result_source == "fallback_uncertain"
        )
        observation_repairs = visual["observation_repairs"]
        alignment_repairs = visual["alignment_repairs"]
        repair_counts["observation_rows"] += int(bool(observation_repairs))
        repair_counts["observation_attempts"] += len(observation_repairs)
        repair_counts["alignment_rows"] += int(bool(alignment_repairs))
        repair_counts["alignment_attempts"] += len(alignment_repairs)
        validation_counts[
            f"observation:{observation_source}"
        ] += 1
        validation_counts[f"result:{result_source}"] += 1
        for stage, attempts in (
            ("observation", observation_repairs),
            ("alignment", alignment_repairs),
        ):
            for attempt in attempts:
                generated = attempt.get("repair_generation_called")
                if generated is True:
                    repair_generation_counts[f"{stage}:generated"] += 1
                elif generated is False:
                    repair_generation_counts[
                        f"{stage}:deterministic"
                    ] += 1
                elif generated is not None:
                    raise ValueError(
                        f"iid={iid} repair generation flag is invalid"
                    )
                events = attempt.get("repair_sanitizations", [])
                if not isinstance(events, list):
                    raise ValueError(
                        f"iid={iid} repair sanitizations are invalid"
                    )
                for event in events:
                    if (
                        not isinstance(event, Mapping)
                        or not isinstance(event.get("action"), str)
                        or not event["action"]
                    ):
                        raise ValueError(
                            f"iid={iid} sanitization event is invalid"
                        )
                    sanitization_counts[
                        f"{stage}:{event['action']}"
                    ] += 1

    if len(contracts) != 1:
        raise ValueError("fused_v2 Qwen semantic contract differs across rows")
    if len(run_configs) != 1:
        raise ValueError("fused_v2 Qwen run configuration differs across rows")
    if any(shard_rows[index] == 0 for index in range(REQUIRED_QWEN_SHARDS)):
        raise ValueError("fused_v2 contains an empty Qwen shard")
    if any(
        len(shard_configs[index]) != 1
        or len(shard_manifests[index]) != 1
        or len(shard_manifest_paths[index]) != 1
        for index in range(REQUIRED_QWEN_SHARDS)
    ):
        raise ValueError("fused_v2 has mixed per-shard provenance")

    contract = json.loads(next(iter(contracts)).decode("utf-8"))
    contract.update(
        {
            "run_config_digest": next(iter(run_configs)),
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "visual_schema_version": VISUAL_SCHEMA_VERSION,
        }
    )
    _require_same(
        summary.get("qwen_contract"),
        contract,
        context="fused_v2 qwen_contract",
    )
    expected_counts: dict[str, Any] = {
        "verdict_counts": _counter_dict(verdict_counts),
        "family_counts": _counter_dict(family_counts),
        "verdict_family_counts": {
            family: _counter_dict(counts)
            for family, counts in sorted(verdict_family.items())
        },
        "fallback_counts": _counter_dict(fallback_counts),
        "repair_counts": _counter_dict(repair_counts),
        "validation_source_counts": _counter_dict(validation_counts),
        "repair_generation_counts": _counter_dict(
            repair_generation_counts
        ),
        "sanitization_counts": _counter_dict(sanitization_counts),
    }
    for field, expected in expected_counts.items():
        _require_same(
            summary.get(field),
            expected,
            context=f"fused_v2 summary {field}",
        )

    shards = summary.get("shards")
    if not isinstance(shards, list) or len(shards) != REQUIRED_QWEN_SHARDS:
        raise ValueError("fused_v2 shard summaries are incomplete")
    run_config = next(iter(run_configs))
    for shard_index, raw_shard in enumerate(shards):
        shard = _expect_mapping(
            raw_shard,
            context=f"fused_v2 shard {shard_index}",
        )
        if set(shard) != {
            "shard_index",
            "manifest_rows",
            "manifest_sha256",
            "marker_sha256",
            "output_rows",
            "output_sha256",
            "config_digest",
            "run_config_digest",
        }:
            raise ValueError(
                f"fused_v2 shard {shard_index} key set mismatch"
            )
        expected_rows = shard_rows[shard_index]
        expected_manifest_sha = next(iter(shard_manifests[shard_index]))
        expected_config = next(iter(shard_configs[shard_index]))
        if (
            shard.get("shard_index") != shard_index
            or shard.get("manifest_rows") != expected_rows
            or shard.get("output_rows") != expected_rows
            or shard.get("manifest_sha256") != expected_manifest_sha
            or shard.get("config_digest") != expected_config
            or shard.get("run_config_digest") != run_config
        ):
            raise ValueError(
                f"fused_v2 shard {shard_index} semantic mismatch"
            )
        _expect_sha(
            shard.get("marker_sha256"),
            context=f"fused_v2 shard {shard_index} marker SHA",
        )
        _expect_sha(
            shard.get("output_sha256"),
            context=f"fused_v2 shard {shard_index} output SHA",
        )
    return contract


def _verify_fused_commit(
    artifacts: Mapping[str, Path],
) -> tuple[
    list[dict[str, Any]],
    bytes,
    dict[str, Any],
    dict[str, Any],
    bytes,
    bytes,
]:
    rows, fused_raw = _load_jsonl(
        artifacts["fused.jsonl"],
        context="fused_v2/fused.jsonl",
        allow_empty=False,
    )
    summary, summary_raw = _load_json(
        artifacts["summary.json"],
        context="fused_v2/summary.json",
    )
    done, done_raw = _load_json(
        artifacts["done.json"],
        context="fused_v2/done.json",
    )
    if summary.get("schema_version") != FUSED_SUMMARY_SCHEMA:
        raise ValueError("unexpected fused_v2 summary schema")
    if done.get("schema_version") != FUSED_DONE_SCHEMA:
        raise ValueError("unexpected fused_v2 done schema")
    if done.get("status") != "complete":
        raise ValueError("fused_v2 done status is not complete")
    if set(done) != {
        "schema_version",
        "status",
        "input_rows",
        "input_sha256",
        "fused_rows",
        "fused_sha256",
        "summary_sha256",
        "artifact_digest",
    }:
        raise ValueError("fused_v2 done key set mismatch")

    row_count = len(rows)
    fused_sha = _sha256_bytes(fused_raw)
    summary_sha = _sha256_bytes(summary_raw)
    fused_meta = _expect_mapping(
        summary.get("fused"),
        context="fused_v2 summary.fused",
    )
    if fused_meta.get("name") != "fused.jsonl":
        raise ValueError("fused_v2 summary names the wrong fused artifact")
    if _expect_int(fused_meta.get("rows"), context="fused rows") != row_count:
        raise ValueError("fused_v2 summary row count mismatch")
    if _expect_sha(
        fused_meta.get("sha256"),
        context="fused summary SHA",
    ) != fused_sha:
        raise ValueError("fused_v2 summary fused SHA mismatch")
    input_meta = _expect_mapping(
        summary.get("input"),
        context="fused_v2 summary.input",
    )
    input_rows = _expect_int(input_meta.get("rows"), context="fused input rows")
    input_sha = _expect_sha(
        input_meta.get("sha256"),
        context="fused input SHA",
    )
    if input_rows != row_count:
        raise ValueError("fused_v2 did not conserve input row count")

    if _expect_int(
        done.get("input_rows"),
        context="fused done input_rows",
    ) != row_count:
        raise ValueError("fused_v2 done input_rows mismatch")
    if _expect_int(
        done.get("fused_rows"),
        context="fused done fused_rows",
    ) != row_count:
        raise ValueError("fused_v2 done fused_rows mismatch")
    expected_done = {
        "input_sha256": input_sha,
        "fused_sha256": fused_sha,
        "summary_sha256": summary_sha,
        "artifact_digest": _object_digest(
            {
                "fused.jsonl": fused_sha,
                "summary.json": summary_sha,
            }
        ),
    }
    for field, expected in expected_done.items():
        if done.get(field) != expected:
            raise ValueError(f"fused_v2 done {field} mismatch")
    _verify_fused_summary_semantics(rows, summary)
    return rows, fused_raw, summary, done, summary_raw, done_raw


def _verify_manifest_chain(
    artifacts: Mapping[str, Path],
    *,
    fused_sha: str,
    fused_rows: int,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, bytes],
    dict[str, Any],
    dict[str, Any],
    bytes,
    bytes,
]:
    bucket_rows: dict[str, list[dict[str, Any]]] = {}
    bucket_raw: dict[str, bytes] = {}
    for bucket, name in BUCKET_FILES.items():
        rows, raw = _load_jsonl(
            artifacts[name],
            context=f"manifest_v2/{name}",
            allow_empty=True,
        )
        bucket_rows[bucket] = rows
        bucket_raw[name] = raw
    summary, summary_raw = _load_json(
        artifacts["summary.json"],
        context="manifest_v2/summary.json",
    )
    done, done_raw = _load_json(
        artifacts["done.json"],
        context="manifest_v2/done.json",
    )
    if summary.get("schema_version") != MANIFEST_SUMMARY_SCHEMA:
        raise ValueError("unexpected manifest_v2 summary schema")
    if summary.get("status") != "complete":
        raise ValueError("manifest_v2 summary status is not complete")
    if done.get("schema_version") != MANIFEST_DONE_SCHEMA:
        raise ValueError("unexpected manifest_v2 done schema")
    if done.get("status") != "complete":
        raise ValueError("manifest_v2 done status is not complete")
    if set(done) != {
        "schema_version",
        "status",
        "input_rows",
        "input_sha256",
        "implementation_sha256",
        "output_sha256",
        "output_rows",
        "artifact_digest",
        "split_assigned",
        "legacy_split_rows_removed",
        "human_labels_asserted",
        "formal_evidence",
    }:
        raise ValueError("manifest_v2 done key set mismatch")
    implementation = _expect_mapping(
        summary.get("implementation"),
        context="manifest_v2 summary.implementation",
    )
    implementation_sha = _expect_sha(
        implementation.get("bundle_sha256"),
        context="manifest implementation bundle SHA",
    )
    implementation_files = _expect_mapping(
        implementation.get("files"),
        context="manifest implementation files",
    )
    if set(implementation_files) != {
        "qwen_filter.py",
        "r7_build_expansion_manifest.py",
    }:
        raise ValueError("manifest implementation file set mismatch")
    implementation_file_shas: dict[str, str] = {}
    for name, raw_metadata in implementation_files.items():
        metadata = _expect_mapping(
            raw_metadata,
            context=f"manifest implementation {name}",
        )
        if set(metadata) != {"path", "sha256"}:
            raise ValueError(
                f"manifest implementation {name} key set mismatch"
            )
        _expect_string(
            metadata.get("path"),
            context=f"manifest implementation {name} path",
        )
        implementation_file_shas[name] = _expect_sha(
            metadata.get("sha256"),
            context=f"manifest implementation {name} SHA",
        )
    if implementation_sha != _object_digest(implementation_file_shas):
        raise ValueError("manifest implementation bundle SHA mismatch")

    input_meta = _expect_mapping(
        summary.get("input"),
        context="manifest_v2 summary.input",
    )
    if _expect_int(
        input_meta.get("rows"),
        context="manifest input rows",
    ) != fused_rows:
        raise ValueError("manifest_v2 input row count differs from fused_v2")
    if _expect_sha(
        input_meta.get("sha256"),
        context="manifest input SHA",
    ) != fused_sha:
        raise ValueError("manifest input SHA differs from fused SHA")

    output_meta = _expect_mapping(
        summary.get("outputs"),
        context="manifest_v2 summary.outputs",
    )
    if set(output_meta) != set(BUCKET_FILES.values()):
        raise ValueError("manifest_v2 summary output artifact set mismatch")
    output_sha: dict[str, str] = {}
    output_rows: dict[str, int] = {}
    for bucket, name in BUCKET_FILES.items():
        metadata = _expect_mapping(
            output_meta[name],
            context=f"manifest summary output {name}",
        )
        actual_sha = _sha256_bytes(bucket_raw[name])
        actual_rows = len(bucket_rows[bucket])
        if _expect_sha(
            metadata.get("sha256"),
            context=f"manifest {name} SHA",
        ) != actual_sha:
            raise ValueError(f"manifest_v2 {name} SHA mismatch")
        if _expect_int(
            metadata.get("rows"),
            context=f"manifest {name} rows",
        ) != actual_rows:
            raise ValueError(f"manifest_v2 {name} row count mismatch")
        output_sha[name] = actual_sha
        output_rows[name] = actual_rows

    summary_sha = _sha256_bytes(summary_raw)
    expected_output_sha = {
        **dict(sorted(output_sha.items())),
        "summary.json": summary_sha,
    }
    expected_output_rows = dict(sorted(output_rows.items()))
    if _expect_int(
        done.get("input_rows"),
        context="manifest done input_rows",
    ) != fused_rows:
        raise ValueError("manifest_v2 done input_rows mismatch")
    if done.get("input_sha256") != fused_sha:
        raise ValueError("manifest_v2 done input SHA mismatch")
    _require_same(
        done.get("output_sha256"),
        expected_output_sha,
        context="manifest_v2 done output SHA map",
    )
    _require_same(
        done.get("output_rows"),
        expected_output_rows,
        context="manifest_v2 done output row map",
    )
    if done.get("artifact_digest") != _object_digest(expected_output_sha):
        raise ValueError("manifest_v2 done artifact_digest mismatch")
    if done.get("implementation_sha256") != implementation_sha:
        raise ValueError("manifest_v2 implementation SHA chain mismatch")
    quarantine_summary = _expect_mapping(
        summary.get("legacy_split_quarantine"),
        context="manifest_v2 summary legacy_split_quarantine",
    )
    removed_rows = _expect_int(
        quarantine_summary.get("rows_with_pair_removed"),
        context="manifest summary legacy rows removed",
    )
    if _expect_int(
        done.get("legacy_split_rows_removed"),
        context="manifest done legacy rows removed",
    ) != removed_rows:
        raise ValueError("manifest_v2 legacy quarantine count chain mismatch")
    _expect_false(
        done.get("split_assigned"),
        context="manifest done split_assigned",
    )
    _expect_false(
        done.get("human_labels_asserted"),
        context="manifest done human_labels_asserted",
    )
    _expect_false(
        done.get("formal_evidence"),
        context="manifest done formal_evidence",
    )
    return (
        bucket_rows,
        bucket_raw,
        summary,
        done,
        summary_raw,
        done_raw,
    )


def _prepare_output(
    output_path: Path | None,
    *,
    resume: bool,
) -> Path | None:
    if resume and output_path is None:
        raise ValueError("--resume requires --output")
    if output_path is None:
        return None
    expanded = output_path.expanduser()
    if resume:
        # This check deliberately precedes every input read.
        if expanded.is_symlink() or not expanded.is_file():
            raise FileNotFoundError(
                "--resume requires an existing regular non-symlink output"
            )
        return expanded.resolve(strict=True)
    if expanded.exists() or expanded.is_symlink():
        raise FileExistsError(f"output already exists: {expanded}")
    return expanded.resolve(strict=False)


def _commit_report(path: Path, payload: bytes) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise FileNotFoundError(
            f"output parent must be an existing non-symlink directory: {parent}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-linking a complete same-filesystem temp file gives atomic
        # create-if-absent semantics; unlike rename(), it cannot overwrite.
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def audit_expansion(
    *,
    fused_dir: Path,
    manifest_dir: Path,
    output_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Audit one fused/manifest pair and optionally commit the JSON report."""

    prepared_output = _prepare_output(output_path, resume=resume)
    fused_root, fused_artifacts = _strict_directory(
        fused_dir,
        expected_names=FUSED_ARTIFACTS,
        description="fused_v2",
    )
    manifest_root, manifest_artifacts = _strict_directory(
        manifest_dir,
        expected_names=MANIFEST_ARTIFACTS,
        description="manifest_v2",
    )
    if fused_root == manifest_root:
        raise ValueError("fused_v2 and manifest_v2 must be different directories")
    if prepared_output is not None and (
        prepared_output == fused_root
        or fused_root in prepared_output.parents
        or prepared_output == manifest_root
        or manifest_root in prepared_output.parents
    ):
        raise ValueError("audit output must be outside both input directories")

    (
        fused_rows,
        fused_raw,
        fused_summary,
        _fused_done,
        fused_summary_raw,
        fused_done_raw,
    ) = _verify_fused_commit(fused_artifacts)
    fused_sha = _sha256_bytes(fused_raw)
    (
        bucket_rows,
        bucket_raw,
        manifest_summary,
        _manifest_done,
        manifest_summary_raw,
        manifest_done_raw,
    ) = _verify_manifest_chain(
        manifest_artifacts,
        fused_sha=fused_sha,
        fused_rows=len(fused_rows),
    )
    manifest_implementation = _expect_mapping(
        manifest_summary.get("implementation"),
        context="manifest_v2 summary.implementation",
    )
    manifest_implementation_sha = _expect_sha(
        manifest_implementation.get("bundle_sha256"),
        context="manifest implementation bundle SHA",
    )

    fused_by_iid: dict[str, tuple[int, dict[str, Any]]] = {}
    for line_number, row in enumerate(fused_rows, start=1):
        iid = _iid(row, context=f"fused_v2 line {line_number}")
        if iid in fused_by_iid:
            raise ValueError(f"duplicate fused IID: {iid}")
        fused_by_iid[iid] = (line_number, row)

    bucket_verdict_family: dict[
        str, dict[str, Counter[str]]
    ] = {}
    bucket_stage_source_confidence: dict[
        str, dict[str, Counter[str]]
    ] = {}
    bucket_source_pair_confidence: dict[
        str, dict[str, Counter[str]]
    ] = {}
    bucket_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    validation_source_counts: Counter[str] = Counter()
    bucket_verdict_counts: dict[str, Counter[str]] = {}
    bucket_family_counts: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    negative_type_counts: Counter[str] = Counter()
    negative_role_counts: Counter[str] = Counter()
    review_quality_failures: Counter[str] = Counter()
    review_with_quality = 0
    exact_signatures: Counter[str] = Counter()
    normalized_signatures: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    score_bin_counts: Counter[str] = Counter()
    unstratifiable_score_rows = 0
    bucket_tier_score: dict[str, dict[str, Counter[str]]] = {}
    quarantine_stage_counts: Counter[str] = Counter()
    legacy_split_value_counts: Counter[str] = Counter()
    legacy_provenance_sha_counts: Counter[str] = Counter()
    legacy_pair_sha_counts: Counter[str] = Counter()
    legacy_canonical_sha_counts: Counter[str] = Counter()
    legacy_upstream_sha_counts: Counter[str] = Counter()
    legacy_rows_removed = 0
    legacy_rows_removed_by_builder = 0

    seen_manifest: dict[str, str] = {}
    sanitized_rows = 0
    repair_rows = 0
    fallback_rows = 0
    for bucket, rows in bucket_rows.items():
        previous_source_line = 0
        for artifact_line, row in enumerate(rows, start=1):
            iid = _iid(
                row,
                context=f"manifest_v2 {BUCKET_FILES[bucket]}:{artifact_line}",
            )
            if iid in seen_manifest:
                raise ValueError(
                    f"manifest IID appears in multiple rows/buckets: {iid}"
                )
            seen_manifest[iid] = bucket
            if iid not in fused_by_iid:
                raise ValueError(f"manifest IID is absent from fused_v2: {iid}")
            source_line, fused_row = fused_by_iid[iid]
            label = _expect_mapping(
                row.get("r7_expansion_manifest"),
                context=f"iid={iid} r7_expansion_manifest",
            )
            if label.get("schema_version") != MANIFEST_ROW_SCHEMA:
                raise ValueError(f"iid={iid} has unexpected manifest row schema")
            if label.get("policy_version") != MANIFEST_POLICY:
                raise ValueError(f"iid={iid} has unexpected manifest policy")
            if (
                type(label.get("source_line_number")) is not int
                or label.get("source_line_number") != source_line
            ):
                raise ValueError(f"iid={iid} source line provenance mismatch")
            if source_line <= previous_source_line:
                raise ValueError(
                    f"{BUCKET_FILES[bucket]} is not in fused source order"
                )
            previous_source_line = source_line
            if label.get("source_fused_sha256") != fused_sha:
                raise ValueError(f"iid={iid} source fused SHA mismatch")
            if (
                label.get("builder_implementation_sha256")
                != manifest_implementation_sha
            ):
                raise ValueError(
                    f"iid={iid} builder implementation SHA mismatch"
                )
            _expect_false(
                label.get("split_assigned"),
                context=f"iid={iid} split_assigned",
            )
            _expect_false(
                label.get("human_label"),
                context=f"iid={iid} human_label",
            )
            _expect_false(
                label.get("formal_evidence"),
                context=f"iid={iid} formal_evidence",
            )
            selection = _expect_mapping(
                row.get("r7_expansion_selection"),
                context=f"iid={iid} r7_expansion_selection",
            )
            if selection.get("schema_version") != SELECTION_SCHEMA:
                raise ValueError(
                    f"iid={iid} expansion selection schema mismatch"
                )
            _expect_false(
                selection.get("split_assigned"),
                context=f"iid={iid} selection split_assigned",
            )
            if "split" in row or "split_provenance" in row:
                raise ValueError(
                    f"iid={iid} manifest output retains top-level split fields"
                )
            expected_quarantine = _expected_legacy_split_quarantine(
                fused_row,
                iid=iid,
            )
            public_quarantine = {
                field: expected_quarantine[field]
                for field in PUBLIC_LEGACY_SPLIT_FIELDS
            }
            _require_same(
                label.get("legacy_split_quarantine"),
                public_quarantine,
                context=f"iid={iid} legacy split quarantine",
            )
            base = dict(row)
            del base["r7_expansion_manifest"]
            expected_base = dict(fused_row)
            expected_base.pop("split", None)
            expected_base.pop("split_provenance", None)
            if not _same_json(base, expected_base):
                raise ValueError(f"iid={iid} manifest row differs from fused row")

            visual, observation, result = _visual_fields(row, iid=iid)
            observation_source = _validation_source(
                visual,
                stage="observation",
                iid=iid,
            )
            result_source = _validation_source(
                visual,
                stage="result",
                iid=iid,
            )
            expected = _expected_decision(
                observation_source=observation_source,
                result_source=result_source,
                observation=observation,
                result=result,
            )
            if expected["bucket"] != bucket:
                raise ValueError(
                    f"iid={iid} safety/classification violation: "
                    f"expected bucket={expected['bucket']} actual={bucket}"
                )
            family = _primary_family(row, iid=iid)
            verdict = _expect_string(
                result.get("verdict"),
                context=f"iid={iid} verdict",
            )
            confidence = _expect_string(
                result.get("confidence"),
                context=f"iid={iid} confidence",
            )
            expected_label_fields = {
                "bucket": bucket,
                "classification_reason": expected["classification_reason"],
                "verdict": verdict,
                "primary_family": family,
                "observation_validated_from": observation_source,
                "result_validated_from": result_source,
            }
            for field, value in expected.items():
                if field != "bucket":
                    expected_label_fields[field] = value
            for field, value in expected_label_fields.items():
                if not _same_json(label.get(field), value):
                    raise ValueError(
                        f"iid={iid} manifest label field {field} mismatch"
                    )
            conditional = {
                "action_signature",
                "negative_type",
                "negative_role",
                "quality_failures",
            }
            for field in conditional - set(expected):
                if field in label:
                    raise ValueError(
                        f"iid={iid} has unexpected manifest field {field}"
                    )
            expected_label_keys = {
                "schema_version",
                "policy_version",
                "bucket",
                "classification_reason",
                "source_line_number",
                "source_fused_sha256",
                "builder_implementation_sha256",
                "verdict",
                "primary_family",
                "observation_validated_from",
                "result_validated_from",
                "split_assigned",
                "human_label",
                "formal_evidence",
                "legacy_split_quarantine",
            } | (conditional & set(expected))
            if set(label) != expected_label_keys:
                raise ValueError(
                    f"iid={iid} manifest label key set mismatch"
                )

            if (
                observation_source == "original_sanitized"
                or result_source == "original_sanitized"
            ):
                sanitized_rows += 1
                if bucket == "positive":
                    raise ValueError(
                        f"iid={iid} sanitized evidence entered positives"
                    )
                if bucket == "negative" and (
                    label.get("negative_role") != "audit_only"
                ):
                    raise ValueError(
                        f"iid={iid} sanitized negative is not audit_only"
                    )
            sources = (observation_source, result_source)
            has_repair = any(source.startswith("repair_") for source in sources)
            has_fallback = "fallback_uncertain" in sources
            repair_rows += int(has_repair)
            fallback_rows += int(has_fallback)
            if (has_repair or has_fallback) and bucket != "review":
                raise ValueError(
                    f"iid={iid} repaired/fallback evidence was auto-labeled"
                )

            source_pair = (
                f"observation={observation_source}|result={result_source}"
            )
            bucket_verdict_family.setdefault(bucket, {}).setdefault(
                verdict,
                Counter(),
            )[family] += 1
            bucket_source_pair_confidence.setdefault(
                bucket,
                {},
            ).setdefault(
                source_pair,
                Counter(),
            )[confidence] += 1
            for stage, source in (
                ("observation", observation_source),
                ("result", result_source),
            ):
                bucket_stage_source_confidence.setdefault(
                    bucket,
                    {},
                ).setdefault(
                    f"{stage}:{source}",
                    Counter(),
                )[confidence] += 1
            bucket_counts[bucket] += 1
            verdict_counts[verdict] += 1
            family_counts[family] += 1
            bucket_verdict_counts.setdefault(bucket, Counter())[verdict] += 1
            bucket_family_counts.setdefault(bucket, Counter())[family] += 1
            reason_counts[str(expected["classification_reason"])] += 1
            validation_source_counts[
                f"observation:{observation_source}"
            ] += 1
            validation_source_counts[f"result:{result_source}"] += 1
            if "negative_type" in expected:
                negative_type_counts[str(expected["negative_type"])] += 1
            if "negative_role" in expected:
                negative_role_counts[str(expected["negative_role"])] += 1
            if bucket == "positive":
                signature = str(expected["action_signature"])
                normalized = _normalize_signature(signature)
                if not normalized:
                    raise ValueError(
                        f"iid={iid} positive signature normalizes to empty"
                    )
                exact_signatures[signature] += 1
                normalized_signatures[normalized] += 1
            if bucket == "review" and "quality_failures" in expected:
                review_with_quality += 1
                for failure in expected["quality_failures"]:
                    review_quality_failures[str(failure)] += 1

            raw_rule = row.get("auto_rule")
            rule = raw_rule if isinstance(raw_rule, Mapping) else {}
            raw_tier = rule.get("tier")
            tier = (
                raw_tier.strip().lower()
                if isinstance(raw_tier, str) and raw_tier.strip()
                else "<missing>"
            )
            raw_score = rule.get("score")
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(float(raw_score))
            ):
                score_bin = "<missing_or_invalid>"
                unstratifiable_score_rows += 1
            else:
                score_bin = _score_bin(float(raw_score))
            tier_counts[tier] += 1
            score_bin_counts[score_bin] += 1
            bucket_tier_score.setdefault(bucket, {}).setdefault(
                tier,
                Counter(),
            )[score_bin] += 1

            quarantine_stage = str(
                expected_quarantine["quarantine_stage"]
            )
            quarantine_stage_counts[quarantine_stage] += 1
            if expected_quarantine["removed"] is True:
                legacy_rows_removed += 1
                legacy_canonical_sha_counts[
                    str(expected_quarantine["canonical_sha256"])
                ] += 1
            if quarantine_stage == "builder_legacy":
                legacy_rows_removed_by_builder += 1
                legacy_split_value_counts[
                    str(expected_quarantine["legacy_split_value"])
                ] += 1
                legacy_provenance_sha_counts[
                    str(
                        expected_quarantine[
                            "legacy_split_provenance_sha256"
                        ]
                    )
                ] += 1
                legacy_pair_sha_counts[
                    str(expected_quarantine["legacy_split_pair_sha256"])
                ] += 1
            elif quarantine_stage == "selection_upstream":
                legacy_upstream_sha_counts[
                    str(expected_quarantine["canonical_sha256"])
                ] += 1

    if set(seen_manifest) != set(fused_by_iid):
        raise ValueError(
            "manifest bucket union does not equal fused IID set: "
            f"missing={sorted(set(fused_by_iid) - set(seen_manifest))[:20]} "
            f"extra={sorted(set(seen_manifest) - set(fused_by_iid))[:20]}"
        )
    if len(seen_manifest) != len(fused_rows):
        raise ValueError("manifest N conservation failed")

    expected_producer_counts = {
        "bucket_counts": {
            bucket: bucket_counts.get(bucket, 0)
            for bucket in sorted(BUCKET_FILES)
        },
        "verdict_counts": _counter_dict(verdict_counts),
        "family_counts": _counter_dict(family_counts),
        "validation_source_counts": _counter_dict(
            validation_source_counts
        ),
        "bucket_verdict_counts": _nested_counter_dict(
            bucket_verdict_counts
        ),
        "bucket_family_counts": _nested_counter_dict(bucket_family_counts),
        "classification_reason_counts": _counter_dict(reason_counts),
        "negative_type_counts": _counter_dict(negative_type_counts),
        "negative_role_counts": _counter_dict(negative_role_counts),
    }
    for field, expected in expected_producer_counts.items():
        _require_same(
            manifest_summary.get(field),
            expected,
            context=f"manifest_v2 summary {field}",
        )
    expected_quarantine_summary = {
        "policy_version": LEGACY_SPLIT_POLICY,
        "rows_with_pair_removed": legacy_rows_removed,
        "rows_removed_by_builder": legacy_rows_removed_by_builder,
        "rows_removed_by_selection_upstream": (
            legacy_rows_removed - legacy_rows_removed_by_builder
        ),
        "rows_with_no_legacy_pair_attested": (
            len(fused_rows) - legacy_rows_removed
        ),
        "input_rows_without_top_level_legacy_fields": (
            len(fused_rows) - legacy_rows_removed_by_builder
        ),
        "quarantine_stage_counts": _counter_dict(
            quarantine_stage_counts
        ),
        "quarantined_source_top_level_fields": [
            "split",
            "split_provenance",
        ],
        "builder_legacy": {
            "rows_removed": legacy_rows_removed_by_builder,
            "accepted_split_values": sorted(LEGACY_SPLIT_VALUES),
            "accepted_split_provenance": dict(
                LEGACY_SPLIT_PROVENANCE
            ),
            "accepted_split_provenance_sha256": _object_digest(
                LEGACY_SPLIT_PROVENANCE
            ),
            "split_value_counts": _counter_dict(
                legacy_split_value_counts
            ),
            "split_provenance_sha256_counts": _counter_dict(
                legacy_provenance_sha_counts
            ),
            "split_pair_sha256_counts": _counter_dict(
                legacy_pair_sha_counts
            ),
        },
        "selection_upstream": {
            "rows_removed": (
                legacy_rows_removed - legacy_rows_removed_by_builder
            ),
            "attestation_required_keys": [
                "canonical_sha256",
                "present",
            ],
            "canonical_sha256_counts": _counter_dict(
                legacy_upstream_sha_counts
            ),
        },
        "all_removed_pair_canonical_sha256_counts": _counter_dict(
            legacy_canonical_sha_counts
        ),
        "output_rows_have_top_level_split": False,
        "qwen_input_digest_or_evidence_rewritten": False,
    }
    _require_same(
        manifest_summary.get("legacy_split_quarantine"),
        expected_quarantine_summary,
        context="manifest_v2 summary legacy_split_quarantine",
    )
    if manifest_summary.get("policy_version") != MANIFEST_POLICY:
        raise ValueError("manifest_v2 policy version mismatch")
    expected_policy = {
        "positive_verdicts": sorted(POSITIVE_VERDICTS),
        "negative_verdicts": sorted(NEGATIVE_VERDICTS),
        "positive_target_actor_motion": sorted(VISIBLE_TARGET_MOTION),
        "positive_quality": dict(sorted(POSITIVE_QUALITY.items())),
        "accepted_confidence": sorted(ACCEPTED_CONFIDENCE),
        "original_sanitized_positive_allowed": False,
        "repair_or_fallback_auto_label_allowed": False,
    }
    _require_same(
        manifest_summary.get("policy"),
        expected_policy,
        context="manifest_v2 summary policy",
    )
    semantics = _expect_mapping(
        manifest_summary.get("semantics"),
        context="manifest_v2 summary.semantics",
    )
    for field in (
        "split_assigned",
        "human_labels_asserted",
        "formal_evidence",
        "production_eligible",
    ):
        _expect_false(
            semantics.get(field),
            context=f"manifest semantics {field}",
        )

    score_bin_definition = [
        {
            "name": name,
            "lower": lower,
            "upper": upper,
            "upper_inclusive": upper_inclusive,
        }
        for name, lower, upper, upper_inclusive in SCORE_BINS
    ]
    summary: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "complete",
        "auditor_implementation": _auditor_implementation(),
        "qwen_contract": dict(fused_summary["qwen_contract"]),
        "inputs": {
            "fused_v2": {
                "rows": len(fused_rows),
                "artifacts": {
                    "fused.jsonl": fused_sha,
                    "summary.json": _sha256_bytes(fused_summary_raw),
                    "done.json": _sha256_bytes(fused_done_raw),
                },
            },
            "manifest_v2": {
                "rows": len(seen_manifest),
                "artifacts": {
                    **{
                        name: _sha256_bytes(raw)
                        for name, raw in sorted(bucket_raw.items())
                    },
                    "summary.json": _sha256_bytes(manifest_summary_raw),
                    "done.json": _sha256_bytes(manifest_done_raw),
                },
            },
        },
        "conservation": {
            "fused_rows": len(fused_rows),
            "manifest_rows": len(seen_manifest),
            "unique_fused_iids": len(fused_by_iid),
            "unique_manifest_iids": len(seen_manifest),
            "bucket_disjoint_union_exact": True,
            "manifest_input_sha_equals_fused_sha": True,
        },
        "bucket_verdict_family_counts": _three_dimensional_counts(
            bucket_verdict_family
        ),
        "bucket_validation_source_confidence_counts": (
            _three_dimensional_counts(bucket_stage_source_confidence)
        ),
        "bucket_validation_source_pair_confidence_counts": (
            _three_dimensional_counts(bucket_source_pair_confidence)
        ),
        "safety_assertions": {
            "all_passed": True,
            "sanitized_rows": sanitized_rows,
            "sanitized_positive_rows": 0,
            "sanitized_non_audit_negative_rows": 0,
            "repair_rows": repair_rows,
            "fallback_rows": fallback_rows,
            "repair_or_fallback_auto_labeled_rows": 0,
            "split_assigned_rows": 0,
            "human_labeled_rows": 0,
            "formal_evidence_rows": 0,
        },
        "legacy_split_quarantine": {
            "all_manifest_top_level_split_fields_absent": True,
            "policy_version": LEGACY_SPLIT_POLICY,
            "rows_removed": legacy_rows_removed,
            "rows_removed_by_builder": legacy_rows_removed_by_builder,
            "rows_removed_by_selection_upstream": (
                legacy_rows_removed - legacy_rows_removed_by_builder
            ),
            "quarantine_stage_counts": _counter_dict(
                quarantine_stage_counts
            ),
            "legacy_split_value_counts": _counter_dict(
                legacy_split_value_counts
            ),
            "canonical_sha256_counts": _counter_dict(
                legacy_canonical_sha_counts
            ),
        },
        "negative_roles": {
            "rows": sum(negative_role_counts.values()),
            "counts": _counter_dict(negative_role_counts),
        },
        "positive_signatures": {
            "exact": _signature_summary(
                exact_signatures,
                normalization="none",
            ),
            "diagnostic_normalized": _signature_summary(
                normalized_signatures,
                normalization=(
                    "unicode_nfkc_casefold_non_alnum_to_space_v1"
                ),
            ),
            "diagnostic_normalization_is_not_a_semantic_label": True,
        },
        "review_quality_failures": {
            "review_rows": bucket_counts.get("review", 0),
            "rows_with_quality_failures": review_with_quality,
            "rows_without_quality_failures": (
                bucket_counts.get("review", 0) - review_with_quality
            ),
            "failure_occurrences": sum(review_quality_failures.values()),
            "counts": _counter_dict(review_quality_failures),
        },
        "auto_rule_strata": {
            "fixed_score_bins": score_bin_definition,
            "tier_counts": _counter_dict(tier_counts),
            "score_bin_counts": _counter_dict(score_bin_counts),
            "unstratifiable_score_rows": unstratifiable_score_rows,
            "bucket_tier_score_bin_counts": _three_dimensional_counts(
                bucket_tier_score
            ),
        },
    }
    payload = _report_bytes(summary)
    if prepared_output is not None:
        if resume:
            if prepared_output.read_bytes() != payload:
                raise RuntimeError(
                    "resume output differs from fresh deterministic audit"
                )
        else:
            _commit_report(prepared_output, payload)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict read-only R7 expansion post-hoc audit."
    )
    parser.add_argument("--fused-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Verification-only: output must already exist and match exactly."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = audit_expansion(
        fused_dir=args.fused_dir,
        manifest_dir=args.manifest_dir,
        output_path=args.output,
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
