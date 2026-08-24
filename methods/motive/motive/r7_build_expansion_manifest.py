"""Build a strict, provenance-bound R7 post-Qwen pseudo-label manifest.

The input must be ``fused.jsonl`` produced by :mod:`motive.r7_qwen_merge`.
Every Qwen observation/result is revalidated with the authoritative
``qwen_filter`` validators and its digest/validation-source audit is checked
again.  This stage deliberately does not create data splits and does not
assert human or formal labels.

The output is an atomic immutable-style directory containing:

``positives.jsonl``
    Conservative Qwen pseudo-positives only.
``negatives.jsonl``
    Trustworthy original non-action/wrong-action pseudo-negatives, plus
    deterministic ``original_sanitized`` audit-only negatives.
``review.jsonl``
    Repairs, fallbacks, uncertainty, and positive-looking rows that fail the
    quality gate.
``summary.json``
    Input, implementation, policy, count, and output-hash provenance.
``done.json``
    Terminal commit marker written last.

Existing outputs are never overwritten.  ``--resume`` is verification-only:
it requires an existing output directory and compares every byte with a fresh
in-memory derivation from the supplied input and current implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import qwen_filter as qwen_filter_module
from .qwen_filter import (
    _object_digest,
    _validate_observation,
    _validate_visual,
)


ROW_SCHEMA = "motive-r7-expansion-manifest-row-v2"
SUMMARY_SCHEMA = "motive-r7-expansion-manifest-v2"
DONE_SCHEMA = "motive-r7-expansion-manifest-done-v2"
POLICY_VERSION = "r7-strict-qwen-pseudolabel-v2"
SELECTION_SCHEMA = "motive-r7-expansion-selection-v1"
LEGACY_SPLIT_QUARANTINE_POLICY_VERSION = (
    "r7-legacy-caption-or-path-split-quarantine-v1"
)
LEGACY_SPLIT_VALUES = frozenset({"train", "validation", "test"})
LEGACY_SPLIT_PROVENANCE = {
    "seed": 260108828,
    "version": "caption-or-path-fallback-v1",
}
PUBLIC_LEGACY_SPLIT_AUDIT_FIELDS = (
    "removed",
    "removed_by_builder",
    "quarantine_stage",
    "canonical_sha256",
    "selection_upstream_attestation",
    "source_top_level_fields_removed",
    "quarantine_policy_version",
)

POSITIVES_NAME = "positives.jsonl"
NEGATIVES_NAME = "negatives.jsonl"
REVIEW_NAME = "review.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
ARTIFACT_NAMES = (
    POSITIVES_NAME,
    NEGATIVES_NAME,
    REVIEW_NAME,
    SUMMARY_NAME,
    DONE_NAME,
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
REQUIRED_EXECUTION_SHARDS = 8


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _strict_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _regular_file(path: Path, *, description: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise FileNotFoundError(
            f"{description} must be a regular non-symlink file: {expanded}"
        )
    return expanded.resolve(strict=True)


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    resolved = _regular_file(path, description="R7 fused Qwen input")
    raw = resolved.read_bytes()
    physical_lines = raw.splitlines(keepends=True)
    if not physical_lines:
        raise ValueError(f"R7 fused Qwen input is empty: {resolved}")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(physical_lines, start=1):
        if not raw_line.strip():
            raise ValueError(
                f"blank JSONL line is forbidden: {resolved}:{line_number}"
            )
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"input is not UTF-8: {resolved}:{line_number}"
            ) from error
        try:
            value = json.loads(
                text,
                parse_constant=_strict_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"input is not strict JSON: {resolved}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"input row is not an object: {resolved}:{line_number}"
            )
        rows.append(value)
    return rows, raw


def _sha256_field(value: Any, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _iid(row: Mapping[str, Any], *, line_number: int) -> str:
    value = row.get("iid")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"input line {line_number} has no non-empty IID")
    iid = value.strip()
    if iid != value or "\x00" in iid:
        raise ValueError(f"input line {line_number} has a non-canonical IID")
    return iid


def _validate_legacy_split_pair(
    row: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    iid: str,
) -> dict[str, Any] | None:
    """Validate a known pre-R7 split pair for explicit quarantine.

    The expansion selection deliberately asserts ``split_assigned=false``.
    Some source rows nevertheless carry the older caption/path split at the
    top level.  Only the exact known legacy schema is accepted; partial,
    extended, or type-confused variants fail closed.  The returned internal
    audit may contain aggregate-counter inputs and must be projected through
    :func:`_public_legacy_split_audit` before it is placed on an output row.
    """

    has_split = "split" in row
    has_provenance = "split_provenance" in row
    upstream_key_present = "legacy_split_quarantine" in selection
    upstream = selection.get("legacy_split_quarantine")
    if upstream_key_present:
        if type(upstream) is not dict or set(upstream) != {
            "present",
            "canonical_sha256",
        }:
            raise ValueError(
                f"iid={iid} selection legacy_split_quarantine is not "
                "canonical"
            )
        present = upstream.get("present")
        canonical_sha256 = upstream.get("canonical_sha256")
        if type(present) is not bool:
            raise ValueError(
                f"iid={iid} selection legacy split present must be boolean"
            )
        if present:
            _sha256_field(
                canonical_sha256,
                description=(
                    f"selection legacy split canonical_sha256 for iid={iid}"
                ),
            )
        elif canonical_sha256 is not None:
            raise ValueError(
                f"iid={iid} selection legacy split canonical_sha256 must "
                "be null when present=false"
            )
        if has_split or has_provenance:
            raise ValueError(
                f"iid={iid} has both selection-upstream quarantine metadata "
                "and top-level legacy split fields"
            )
        if not present:
            return {
                "removed": False,
                "removed_by_builder": False,
                "quarantine_stage": "none",
                "canonical_sha256": None,
                "selection_upstream_attestation": True,
                "source_top_level_fields_removed": [],
                "quarantine_policy_version": (
                    LEGACY_SPLIT_QUARANTINE_POLICY_VERSION
                ),
            }
        return {
            "removed": True,
            "removed_by_builder": False,
            "quarantine_stage": "selection_upstream",
            "canonical_sha256": canonical_sha256,
            "selection_upstream_attestation": True,
            "source_top_level_fields_removed": [
                "split",
                "split_provenance",
            ],
            "quarantine_policy_version": (
                LEGACY_SPLIT_QUARANTINE_POLICY_VERSION
            ),
        }

    if has_split != has_provenance:
        missing = "split_provenance" if has_split else "split"
        raise ValueError(
            f"iid={iid} has a partial legacy split pair; missing {missing}"
        )
    if not has_split:
        return None

    split = row["split"]
    if type(split) is not str or split not in LEGACY_SPLIT_VALUES:
        raise ValueError(
            f"iid={iid} legacy split is not canonical: {split!r}"
        )
    provenance = row["split_provenance"]
    if type(provenance) is not dict:
        raise ValueError(
            f"iid={iid} legacy split_provenance must be an object"
        )
    if set(provenance) != set(LEGACY_SPLIT_PROVENANCE):
        raise ValueError(
            f"iid={iid} legacy split_provenance keys are not canonical"
        )
    if (
        type(provenance.get("seed")) is not int
        or provenance.get("seed") != LEGACY_SPLIT_PROVENANCE["seed"]
        or type(provenance.get("version")) is not str
        or provenance.get("version")
        != LEGACY_SPLIT_PROVENANCE["version"]
    ):
        raise ValueError(
            f"iid={iid} legacy split_provenance values are not canonical"
        )

    canonical_provenance = dict(LEGACY_SPLIT_PROVENANCE)
    source_pair = {
        "split": split,
        "split_provenance": canonical_provenance,
    }
    return {
        "removed": True,
        "removed_by_builder": True,
        "quarantine_stage": "builder_legacy",
        "canonical_sha256": _object_digest(source_pair),
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
        "legacy_split_pair_sha256": _object_digest(source_pair),
        "quarantine_policy_version": (
            LEGACY_SPLIT_QUARANTINE_POLICY_VERSION
        ),
    }


def _public_legacy_split_audit(
    audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return only non-semantic quarantine metadata safe for output rows."""

    if audit is None:
        audit = {
            "removed": False,
            "removed_by_builder": False,
            "quarantine_stage": "none",
            "canonical_sha256": None,
            "selection_upstream_attestation": False,
            "source_top_level_fields_removed": [],
            "quarantine_policy_version": (
                LEGACY_SPLIT_QUARANTINE_POLICY_VERSION
            ),
        }
    return {
        field: audit[field] for field in PUBLIC_LEGACY_SPLIT_AUDIT_FIELDS
    }


def _primary_family(row: Mapping[str, Any]) -> str:
    selection = row.get("r7_expansion_selection")
    if isinstance(selection, Mapping):
        value = selection.get("primary_family")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    rule = row.get("auto_rule")
    if isinstance(rule, Mapping):
        families = rule.get("action_families")
        if (
            isinstance(families, Sequence)
            and not isinstance(families, (str, bytes))
        ):
            for value in families:
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()
    return "unknown"


def _validate_source_audit(
    *,
    stage: str,
    source: Any,
    repairs: Any,
    object_digest: str,
    fallback: Any,
) -> str:
    if not isinstance(repairs, list) or not all(
        isinstance(item, dict) for item in repairs
    ):
        raise ValueError(f"Qwen {stage} repairs must be a list of objects")
    if source == "original":
        if repairs:
            raise ValueError(f"Qwen {stage}=original cannot have repairs")
        return source
    if source == "original_sanitized":
        if stage != "result" or not any(
            repair.get("attempt") == 0
            and repair.get("status") == "ok"
            and repair.get("repair_generation_called") is False
            for repair in repairs
        ):
            raise ValueError(
                f"Qwen {stage}=original_sanitized has invalid audit"
            )
        return source
    if isinstance(source, str) and source.startswith("repair_"):
        suffix = source.removeprefix("repair_")
        if (
            not suffix.isdigit()
            or int(suffix) < 1
            or not any(
                repair.get("attempt") == int(suffix)
                and repair.get("status") == "ok"
                and repair.get("repair_generation_called") is True
                for repair in repairs
            )
        ):
            raise ValueError(f"Qwen {stage} repair provenance is invalid")
        return source
    if source == "fallback_uncertain":
        if (
            not isinstance(fallback, Mapping)
            or fallback.get("fallback_digest") != object_digest
        ):
            raise ValueError(f"Qwen {stage} fallback provenance is invalid")
        return source
    raise ValueError(f"Qwen {stage}_validated_from is invalid: {source!r}")


def _validate_qwen_evidence(
    row: Mapping[str, Any],
    *,
    iid: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence = row.get("qwen_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError(f"iid={iid} has no qwen_evidence object")
    visual = evidence.get("visual")
    if not isinstance(visual, dict):
        raise ValueError(f"iid={iid} has no qwen_evidence.visual object")
    if visual.get("iid") != iid:
        raise ValueError(f"iid={iid} disagrees with Qwen evidence IID")
    input_digest = _sha256_field(
        row.get("input_digest"),
        description=f"input_digest for iid={iid}",
    )
    if visual.get("input_digest") != input_digest:
        raise ValueError(f"iid={iid} Qwen input_digest mismatch")
    if visual.get("status") != "ok" or visual.get("mode") != "visual":
        raise ValueError(f"iid={iid} Qwen evidence is not a successful visual row")

    observation = visual.get("observation")
    result = visual.get("result")
    if not isinstance(observation, dict) or not isinstance(result, dict):
        raise ValueError(f"iid={iid} lacks structured observation/result")
    try:
        _validate_observation(observation)
        _validate_visual(result, observation=observation)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"iid={iid} Qwen schema validation failed: {error}"
        ) from error

    observation_digest = _object_digest(observation)
    result_digest = _object_digest(result)
    if visual.get("observation_digest") != observation_digest:
        raise ValueError(f"iid={iid} Qwen observation_digest mismatch")
    if visual.get("result_digest") != result_digest:
        raise ValueError(f"iid={iid} Qwen result_digest mismatch")

    for field in (
        "visual_input_digest",
        "run_config_digest",
        "config_digest",
        "implementation_digest",
        "execution_manifest_sha256",
    ):
        _sha256_field(
            visual.get(field),
            description=f"Qwen {field} for iid={iid}",
        )
    for field in (
        "model_revision",
        "transformers_version",
        "execution_manifest",
    ):
        if not isinstance(visual.get(field), str) or not visual[field]:
            raise ValueError(f"iid={iid} Qwen {field} is missing")
    shard_index = visual.get("execution_shard_index")
    shard_count = visual.get("execution_shard_count")
    if (
        type(shard_count) is not int
        or shard_count != REQUIRED_EXECUTION_SHARDS
        or type(shard_index) is not int
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError(f"iid={iid} Qwen execution shard provenance is invalid")

    observation_repairs = visual.get("observation_repairs")
    alignment_repairs = visual.get("alignment_repairs")
    observation_source = _validate_source_audit(
        stage="observation",
        source=visual.get("observation_validated_from"),
        repairs=observation_repairs,
        object_digest=observation_digest,
        fallback=visual.get("observation_fallback"),
    )
    result_source = _validate_source_audit(
        stage="result",
        source=visual.get("result_validated_from"),
        repairs=alignment_repairs,
        object_digest=result_digest,
        fallback=visual.get("result_fallback"),
    )
    if isinstance(alignment_repairs, list):
        for repair in alignment_repairs:
            if (
                repair.get("authoritative_context_digest")
                != observation_digest
            ):
                raise ValueError(
                    f"iid={iid} alignment repair context digest mismatch"
                )
    result_fallback = visual.get("result_fallback")
    if result_source == "fallback_uncertain" and (
        not isinstance(result_fallback, Mapping)
        or result_fallback.get("authoritative_context_digest")
        != observation_digest
    ):
        raise ValueError(f"iid={iid} result fallback context digest mismatch")

    # Return a shallow copy so downstream code cannot mutate the source object.
    checked_visual = dict(visual)
    checked_visual["observation_validated_from"] = observation_source
    checked_visual["result_validated_from"] = result_source
    return checked_visual, observation, result


def _positive_quality_failures(
    observation: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    target_motion = observation.get("target_actor_motion")
    if target_motion not in VISIBLE_TARGET_MOTION:
        failures.append(f"target_actor_motion={target_motion}")
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


def _negative_trust_failures(
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


def _classify(
    *,
    visual: Mapping[str, Any],
    observation: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    verdict = str(result["verdict"])
    observation_source = str(visual["observation_validated_from"])
    result_source = str(visual["result_validated_from"])

    # Any observation repair/fallback makes both positive and negative
    # semantics unsuitable for automatic learning.
    if observation_source != "original":
        return {
            "bucket": "review",
            "reason": f"observation_validation_source:{observation_source}",
        }

    if result_source == "original_sanitized":
        # Deterministic sanitization is never allowed to create a positive.
        # The current sanitizer normally produces static; retain a non-action
        # result only as an explicitly audit-only negative.
        if verdict in NEGATIVE_VERDICTS:
            return {
                "bucket": "negative",
                "reason": "deterministic_sanitized_audit_negative",
                "negative_type": verdict,
                "negative_role": "audit_only",
            }
        return {
            "bucket": "review",
            "reason": f"sanitized_nonnegative_verdict:{verdict}",
        }

    if result_source != "original":
        return {
            "bucket": "review",
            "reason": f"result_validation_source:{result_source}",
        }
    if verdict == "uncertain":
        return {"bucket": "review", "reason": "verdict:uncertain"}
    if verdict in POSITIVE_VERDICTS:
        failures = _positive_quality_failures(observation, result)
        if failures:
            return {
                "bucket": "review",
                "reason": "positive_quality_gate_failed",
                "quality_failures": failures,
            }
        return {
            "bucket": "positive",
            "reason": "strict_original_qwen_pseudo_positive",
            "action_signature": str(result["action_signature"]),
        }
    if verdict in NEGATIVE_VERDICTS:
        failures = _negative_trust_failures(observation, result)
        if failures:
            return {
                "bucket": "review",
                "reason": "negative_trust_gate_failed",
                "quality_failures": failures,
            }
        return {
            "bucket": "negative",
            "reason": "trusted_original_qwen_negative",
            "negative_type": verdict,
            "negative_role": "pseudo_negative",
        }
    raise ValueError(f"unsupported validated Qwen verdict: {verdict}")


def _implementation_provenance() -> dict[str, Any]:
    paths = {
        "r7_build_expansion_manifest.py": Path(__file__).resolve(strict=True),
        "qwen_filter.py": Path(qwen_filter_module.__file__).resolve(strict=True),
    }
    files = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in sorted(paths.items())
    }
    return {
        "files": files,
        "bundle_sha256": _object_digest(
            {name: value["sha256"] for name, value in files.items()}
        ),
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


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


def _write_atomic_directory(
    output_dir: Path,
    *,
    files: Mapping[str, bytes],
) -> None:
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=parent,
        )
    )
    try:
        # done.json is the terminal commit marker and is always written last.
        for name in ARTIFACT_NAMES:
            path = staging / name
            with path.open("xb") as handle:
                handle.write(files[name])
                handle.flush()
                os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                f"output directory appeared during commit: {output_dir}"
            )
        os.rename(staging, output_dir)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _strict_resume(
    output_dir: Path,
    *,
    expected: Mapping[str, bytes],
) -> None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise FileExistsError(
            f"--resume requires a regular output directory: {output_dir}"
        )
    expected_names = set(expected)
    actual_names = {path.name for path in output_dir.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError(
            "strict resume artifact set mismatch: "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )
    for name, payload in expected.items():
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"resume artifact is not a regular file: {path}")
        if path.read_bytes() != payload:
            raise RuntimeError(
                f"resume artifact differs from fresh derivation: {path}"
            )


def build_expansion_manifest(
    *,
    input_path: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate and atomically classify one fused R7 Qwen manifest."""

    input_path = _regular_file(input_path, description="R7 fused Qwen input")
    expanded_output = output_dir.expanduser()
    if expanded_output.is_symlink():
        raise ValueError("output directory must not be a symlink")
    output_dir = expanded_output.resolve(strict=False)
    if resume:
        if not output_dir.exists():
            raise FileNotFoundError(
                "--resume is verification-only and requires existing output"
            )
    elif output_dir.exists():
        raise FileExistsError(
            f"{output_dir} exists; use a fresh directory or --resume to verify"
        )
    if output_dir == input_path or input_path in output_dir.parents:
        raise ValueError("output directory cannot contain the input file")

    rows, input_raw = _load_jsonl(input_path)
    input_sha256 = _sha256_bytes(input_raw)
    implementation = _implementation_provenance()
    implementation_sha256 = str(implementation["bundle_sha256"])

    outputs: dict[str, list[dict[str, Any]]] = {
        "positive": [],
        "negative": [],
        "review": [],
    }
    seen_iids: set[str] = set()
    verdict_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    validation_source_counts: Counter[str] = Counter()
    bucket_verdict_counts: dict[str, Counter[str]] = {}
    bucket_family_counts: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    negative_type_counts: Counter[str] = Counter()
    negative_role_counts: Counter[str] = Counter()
    legacy_split_value_counts: Counter[str] = Counter()
    legacy_split_provenance_digest_counts: Counter[str] = Counter()
    legacy_split_pair_digest_counts: Counter[str] = Counter()
    legacy_split_canonical_digest_counts: Counter[str] = Counter()
    legacy_split_upstream_digest_counts: Counter[str] = Counter()
    legacy_split_quarantine_stage_counts: Counter[str] = Counter()
    legacy_split_rows_removed = 0
    legacy_split_rows_removed_by_builder = 0

    for line_number, source_row in enumerate(rows, start=1):
        iid = _iid(source_row, line_number=line_number)
        if iid in seen_iids:
            raise ValueError(f"duplicate input IID: {iid}")
        seen_iids.add(iid)
        selection = source_row.get("r7_expansion_selection")
        if not isinstance(selection, Mapping):
            raise ValueError(
                f"iid={iid} has no r7_expansion_selection object"
            )
        if selection.get("schema_version") != SELECTION_SCHEMA:
            raise ValueError(
                f"iid={iid} has an unexpected expansion selection schema"
            )
        if selection.get("split_assigned") is not False:
            raise ValueError(
                f"iid={iid} selection does not assert split_assigned=false"
            )
        legacy_split_audit = _validate_legacy_split_pair(
            source_row,
            selection=selection,
            iid=iid,
        )
        if "r7_expansion_manifest" in source_row:
            raise ValueError(
                f"iid={iid} already has r7_expansion_manifest metadata"
            )

        visual, observation, result = _validate_qwen_evidence(
            source_row,
            iid=iid,
        )
        decision = _classify(
            visual=visual,
            observation=observation,
            result=result,
        )
        bucket = str(decision["bucket"])
        verdict = str(result["verdict"])
        family = _primary_family(source_row)
        row = dict(source_row)
        # Do not reinterpret or propagate an old caption/path split as the new
        # experiment split.  Qwen input_digest/evidence remain byte-for-byte
        # bound to the original fused row; only these top-level legacy fields
        # are removed from the derived output row.
        row.pop("split", None)
        row.pop("split_provenance", None)
        label: dict[str, Any] = {
            "schema_version": ROW_SCHEMA,
            "policy_version": POLICY_VERSION,
            "bucket": bucket,
            "classification_reason": decision["reason"],
            "source_line_number": line_number,
            "source_fused_sha256": input_sha256,
            "builder_implementation_sha256": implementation_sha256,
            "verdict": verdict,
            "primary_family": family,
            "observation_validated_from": visual[
                "observation_validated_from"
            ],
            "result_validated_from": visual["result_validated_from"],
            "split_assigned": False,
            "human_label": False,
            "formal_evidence": False,
            "legacy_split_quarantine": _public_legacy_split_audit(
                legacy_split_audit
            ),
        }
        for field in (
            "action_signature",
            "negative_type",
            "negative_role",
            "quality_failures",
        ):
            if field in decision:
                label[field] = decision[field]
        row["r7_expansion_manifest"] = label
        outputs[bucket].append(row)

        verdict_counts[verdict] += 1
        family_counts[family] += 1
        reason_counts[str(decision["reason"])] += 1
        bucket_verdict_counts.setdefault(bucket, Counter())[verdict] += 1
        bucket_family_counts.setdefault(bucket, Counter())[family] += 1
        for stage in ("observation", "result"):
            source = str(visual[f"{stage}_validated_from"])
            validation_source_counts[f"{stage}:{source}"] += 1
        if "negative_type" in decision:
            negative_type_counts[str(decision["negative_type"])] += 1
        if "negative_role" in decision:
            negative_role_counts[str(decision["negative_role"])] += 1
        quarantine_stage = (
            str(legacy_split_audit["quarantine_stage"])
            if legacy_split_audit is not None
            else "none"
        )
        legacy_split_quarantine_stage_counts[quarantine_stage] += 1
        if (
            legacy_split_audit is not None
            and legacy_split_audit["removed"] is True
        ):
            legacy_split_rows_removed += 1
            legacy_split_canonical_digest_counts[
                str(legacy_split_audit["canonical_sha256"])
            ] += 1
        if (
            legacy_split_audit is not None
            and legacy_split_audit["quarantine_stage"] == "builder_legacy"
        ):
            legacy_split_rows_removed_by_builder += 1
            legacy_split_value_counts[
                str(legacy_split_audit["legacy_split_value"])
            ] += 1
            legacy_split_provenance_digest_counts[
                str(
                    legacy_split_audit[
                        "legacy_split_provenance_sha256"
                    ]
                )
            ] += 1
            legacy_split_pair_digest_counts[
                str(legacy_split_audit["legacy_split_pair_sha256"])
            ] += 1
        elif (
            legacy_split_audit is not None
            and legacy_split_audit["quarantine_stage"]
            == "selection_upstream"
        ):
            legacy_split_upstream_digest_counts[
                str(legacy_split_audit["canonical_sha256"])
            ] += 1

    data_bytes = {
        POSITIVES_NAME: _jsonl_bytes(outputs["positive"]),
        NEGATIVES_NAME: _jsonl_bytes(outputs["negative"]),
        REVIEW_NAME: _jsonl_bytes(outputs["review"]),
    }
    artifact_rows = {
        POSITIVES_NAME: len(outputs["positive"]),
        NEGATIVES_NAME: len(outputs["negative"]),
        REVIEW_NAME: len(outputs["review"]),
    }
    data_artifacts = {
        name: {
            "rows": artifact_rows[name],
            "sha256": _sha256_bytes(payload),
            "order": "source_fused_order_within_bucket",
        }
        for name, payload in data_bytes.items()
    }
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "policy_version": POLICY_VERSION,
        "input": {
            "path": str(input_path),
            "rows": len(rows),
            "sha256": input_sha256,
            "expected_stage": "motive.r7_qwen_merge/fused.jsonl",
        },
        "implementation": implementation,
        "outputs": data_artifacts,
        "bucket_counts": {
            bucket: len(values) for bucket, values in sorted(outputs.items())
        },
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "validation_source_counts": dict(
            sorted(validation_source_counts.items())
        ),
        "bucket_verdict_counts": {
            bucket: dict(sorted(counts.items()))
            for bucket, counts in sorted(bucket_verdict_counts.items())
        },
        "bucket_family_counts": {
            bucket: dict(sorted(counts.items()))
            for bucket, counts in sorted(bucket_family_counts.items())
        },
        "classification_reason_counts": dict(sorted(reason_counts.items())),
        "negative_type_counts": dict(sorted(negative_type_counts.items())),
        "negative_role_counts": dict(sorted(negative_role_counts.items())),
        "legacy_split_quarantine": {
            "policy_version": LEGACY_SPLIT_QUARANTINE_POLICY_VERSION,
            "rows_with_pair_removed": legacy_split_rows_removed,
            "rows_removed_by_builder": (
                legacy_split_rows_removed_by_builder
            ),
            "rows_removed_by_selection_upstream": (
                legacy_split_rows_removed
                - legacy_split_rows_removed_by_builder
            ),
            "rows_with_no_legacy_pair_attested": (
                len(rows) - legacy_split_rows_removed
            ),
            "input_rows_without_top_level_legacy_fields": (
                len(rows) - legacy_split_rows_removed_by_builder
            ),
            "quarantine_stage_counts": dict(
                sorted(legacy_split_quarantine_stage_counts.items())
            ),
            "quarantined_source_top_level_fields": [
                "split",
                "split_provenance",
            ],
            "builder_legacy": {
                "rows_removed": legacy_split_rows_removed_by_builder,
                "accepted_split_values": sorted(LEGACY_SPLIT_VALUES),
                "accepted_split_provenance": dict(
                    LEGACY_SPLIT_PROVENANCE
                ),
                "accepted_split_provenance_sha256": _object_digest(
                    LEGACY_SPLIT_PROVENANCE
                ),
                "split_value_counts": dict(
                    sorted(legacy_split_value_counts.items())
                ),
                "split_provenance_sha256_counts": dict(
                    sorted(
                        legacy_split_provenance_digest_counts.items()
                    )
                ),
                "split_pair_sha256_counts": dict(
                    sorted(legacy_split_pair_digest_counts.items())
                ),
            },
            "selection_upstream": {
                "rows_removed": (
                    legacy_split_rows_removed
                    - legacy_split_rows_removed_by_builder
                ),
                "attestation_required_keys": [
                    "canonical_sha256",
                    "present",
                ],
                "canonical_sha256_counts": dict(
                    sorted(legacy_split_upstream_digest_counts.items())
                ),
            },
            "all_removed_pair_canonical_sha256_counts": dict(
                sorted(legacy_split_canonical_digest_counts.items())
            ),
            "output_rows_have_top_level_split": False,
            "qwen_input_digest_or_evidence_rewritten": False,
        },
        "policy": {
            "positive_verdicts": sorted(POSITIVE_VERDICTS),
            "negative_verdicts": sorted(NEGATIVE_VERDICTS),
            "positive_target_actor_motion": sorted(
                VISIBLE_TARGET_MOTION
            ),
            "positive_quality": dict(sorted(POSITIVE_QUALITY.items())),
            "accepted_confidence": sorted(ACCEPTED_CONFIDENCE),
            "original_sanitized_positive_allowed": False,
            "repair_or_fallback_auto_label_allowed": False,
        },
        "semantics": {
            "positives": "strict Qwen pseudo-positive; not human ground truth",
            "negatives": (
                "Qwen pseudo-negative or deterministic audit-only negative"
            ),
            "review": "not automatically labeled",
            "split_assigned": False,
            "human_labels_asserted": False,
            "formal_evidence": False,
            "production_eligible": False,
        },
    }
    summary_bytes = _pretty_json_bytes(summary)
    summary_sha256 = _sha256_bytes(summary_bytes)
    done: dict[str, Any] = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "input_rows": len(rows),
        "input_sha256": input_sha256,
        "implementation_sha256": implementation_sha256,
        "output_sha256": {
            **{
                name: value["sha256"]
                for name, value in sorted(data_artifacts.items())
            },
            SUMMARY_NAME: summary_sha256,
        },
        "output_rows": {
            name: artifact_rows[name] for name in sorted(artifact_rows)
        },
        "artifact_digest": _object_digest(
            {
                **{
                    name: value["sha256"]
                    for name, value in sorted(data_artifacts.items())
                },
                SUMMARY_NAME: summary_sha256,
            }
        ),
        "split_assigned": False,
        "legacy_split_rows_removed": legacy_split_rows_removed,
        "human_labels_asserted": False,
        "formal_evidence": False,
    }
    done_bytes = _pretty_json_bytes(done)
    files = {
        **data_bytes,
        SUMMARY_NAME: summary_bytes,
        DONE_NAME: done_bytes,
    }
    if resume:
        _strict_resume(output_dir, expected=files)
    else:
        _write_atomic_directory(output_dir, files=files)
    returned = dict(summary)
    returned["resume_verified"] = bool(resume)
    return returned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an unsplit strict R7 pseudo-label manifest from fused Qwen "
            "evidence."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verification-only; never creates or modifies output artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_expansion_manifest(
        input_path=args.input,
        output_dir=args.output_dir,
        resume=bool(args.resume),
    )
    counts = summary["bucket_counts"]
    print(
        "[motive-r7-expansion-manifest] "
        f"positive={counts['positive']} "
        f"negative={counts['negative']} "
        f"review={counts['review']} "
        f"resume_verified={summary['resume_verified']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
