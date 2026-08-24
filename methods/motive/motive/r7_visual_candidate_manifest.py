"""Build an immutable visual-learning candidate manifest from R7 expansion.

This stage deliberately performs no training and assigns no data split.  It
keeps only strict pseudo-positives and trustworthy ``pseudo_negative`` rows
from an already committed ``r7_build_expansion_manifest`` directory.  Review
rows and deterministic ``audit_only`` negatives are validated but excluded.

The input commit is revalidated at three levels:

* exact artifact set and SHA-256 chains in ``summary.json``/``done.json``;
* canonical JSONL, IID/source-line conservation, and implementation binding;
* authoritative Qwen evidence plus a fresh expansion-policy classification
  for every row, including rows that are not selected.

The output directory contains canonical ``candidates.jsonl``, ``summary.json``
and a terminal ``done.json`` marker.  Existing paths are never overwritten.
``--resume`` is verification-only and compares every byte with a fresh
derivation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import qwen_filter as qwen_filter_module
from . import r7_artifact_permissions as artifact_permissions
from . import r7_build_expansion_manifest as expansion_module
from .r7_build_expansion_manifest import (
    DONE_SCHEMA as SOURCE_DONE_SCHEMA,
    LEGACY_SPLIT_QUARANTINE_POLICY_VERSION,
    NEGATIVES_NAME,
    POLICY_VERSION as SOURCE_POLICY_VERSION,
    POSITIVES_NAME,
    PUBLIC_LEGACY_SPLIT_AUDIT_FIELDS,
    REVIEW_NAME,
    ROW_SCHEMA as SOURCE_ROW_SCHEMA,
    SUMMARY_NAME as SOURCE_SUMMARY_NAME,
    SUMMARY_SCHEMA as SOURCE_SUMMARY_SCHEMA,
    _classify,
    _primary_family,
    _validate_qwen_evidence,
)


ROW_SCHEMA = "motive-r7-visual-candidate-row-v1"
SUMMARY_SCHEMA = "motive-r7-visual-candidate-manifest-v1"
DONE_SCHEMA = "motive-r7-visual-candidate-manifest-done-v1"
POLICY_VERSION = "r7-visual-candidate-positive-plus-trusted-negative-v1"

CANDIDATES_NAME = "candidates.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
SOURCE_ARTIFACT_NAMES = (
    POSITIVES_NAME,
    NEGATIVES_NAME,
    REVIEW_NAME,
    SOURCE_SUMMARY_NAME,
    DONE_NAME,
)
OUTPUT_ARTIFACT_NAMES = (CANDIDATES_NAME, SUMMARY_NAME, DONE_NAME)
SOURCE_DATA_NAMES = (POSITIVES_NAME, NEGATIVES_NAME, REVIEW_NAME)
SOURCE_BUCKETS = {
    POSITIVES_NAME: "positive",
    NEGATIVES_NAME: "negative",
    REVIEW_NAME: "review",
}
CANDIDATE_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "iid",
        "input_digest",
        "prompt",
        "src_video",
        "tgt_video",
        "cohort",
        "primary_family",
        "source_row_sha256",
        "source_artifact_digest",
        "split_assigned",
        "human_label",
        "training_eligible",
    }
)


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


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


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


def _load_canonical_jsonl(
    path: Path,
    *,
    context: str,
) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"{context} must end with a newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(f"{context}:{line_number} is blank")
        value = _parse_json(line, context=f"{context}:{line_number}")
        if not isinstance(value, dict):
            raise ValueError(
                f"{context}:{line_number} is not a JSON object"
            )
        if line != _canonical_bytes(value):
            raise ValueError(
                f"{context}:{line_number} is not canonical JSON"
            )
        rows.append(value)
    return rows, raw


def _sha256_field(value: Any, *, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _plain_string(value: Any, *, context: str) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
    ):
        raise ValueError(f"{context} must be a canonical non-empty string")
    return value


def _strict_directory(
    raw_path: Path,
    *,
    names: Sequence[str],
    context: str,
) -> tuple[Path, dict[str, Path]]:
    expanded = raw_path.expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise FileNotFoundError(
            f"{context} must be a regular non-symlink directory: {expanded}"
        )
    directory = expanded.resolve(strict=True)
    expected = set(names)
    actual = {entry.name for entry in directory.iterdir()}
    if actual != expected:
        raise ValueError(
            f"{context} artifact set mismatch: "
            f"missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    artifacts: dict[str, Path] = {}
    for name in names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"{context} artifact is not a regular file: {path}"
            )
        artifacts[name] = path
    return directory, artifacts


def _implementation_provenance() -> dict[str, Any]:
    paths = {
        "qwen_filter.py": Path(qwen_filter_module.__file__).resolve(
            strict=True
        ),
        "r7_build_expansion_manifest.py": Path(
            expansion_module.__file__
        ).resolve(strict=True),
        "r7_visual_candidate_manifest.py": Path(__file__).resolve(strict=True),
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


def _validate_source_implementation(
    summary: Mapping[str, Any],
) -> str:
    implementation = summary.get("implementation")
    if type(implementation) is not dict:
        raise ValueError("source summary implementation must be an object")
    if set(implementation) != {"files", "bundle_sha256"}:
        raise ValueError("source summary implementation keys are not exact")
    files = implementation.get("files")
    expected_paths = {
        "qwen_filter.py": Path(qwen_filter_module.__file__).resolve(
            strict=True
        ),
        "r7_build_expansion_manifest.py": Path(
            expansion_module.__file__
        ).resolve(strict=True),
    }
    if type(files) is not dict or set(files) != set(expected_paths):
        raise ValueError("source implementation file set is not exact")
    digests: dict[str, str] = {}
    for name, path in sorted(expected_paths.items()):
        entry = files[name]
        if type(entry) is not dict or set(entry) != {"path", "sha256"}:
            raise ValueError(
                f"source implementation metadata is invalid for {name}"
            )
        digest = _sha256_field(
            entry.get("sha256"),
            context=f"source implementation {name}",
        )
        if digest != _sha256_file(path):
            raise ValueError(
                f"source implementation does not match current {name}"
            )
        if type(entry.get("path")) is not str or not entry["path"]:
            raise ValueError(f"source implementation path missing for {name}")
        digests[name] = digest
    bundle = _sha256_field(
        implementation.get("bundle_sha256"),
        context="source implementation bundle_sha256",
    )
    if bundle != _object_digest(digests):
        raise ValueError("source implementation bundle digest mismatch")
    return bundle


def _validate_legacy_audit(value: Any, *, iid: str) -> None:
    if type(value) is not dict or set(value) != set(
        PUBLIC_LEGACY_SPLIT_AUDIT_FIELDS
    ):
        raise ValueError(
            f"iid={iid} legacy split audit fields are not exact"
        )
    for field in (
        "removed",
        "removed_by_builder",
        "selection_upstream_attestation",
    ):
        if type(value.get(field)) is not bool:
            raise ValueError(
                f"iid={iid} legacy split audit {field} must be boolean"
            )
    if (
        value.get("quarantine_policy_version")
        != LEGACY_SPLIT_QUARANTINE_POLICY_VERSION
    ):
        raise ValueError(f"iid={iid} legacy split policy mismatch")
    stage = value.get("quarantine_stage")
    if stage not in {"none", "builder_legacy", "selection_upstream"}:
        raise ValueError(f"iid={iid} legacy split stage is invalid")
    fields = value.get("source_top_level_fields_removed")
    if type(fields) is not list or fields not in (
        [],
        ["split", "split_provenance"],
    ):
        raise ValueError(f"iid={iid} legacy removed fields are invalid")
    removed = value["removed"]
    canonical = value.get("canonical_sha256")
    if removed:
        _sha256_field(
            canonical,
            context=f"iid={iid} legacy canonical_sha256",
        )
        if stage == "none" or fields != ["split", "split_provenance"]:
            raise ValueError(f"iid={iid} inconsistent legacy removal audit")
    elif canonical is not None or stage != "none" or fields:
        raise ValueError(f"iid={iid} inconsistent no-removal audit")
    if value["removed_by_builder"] is not (stage == "builder_legacy"):
        raise ValueError(f"iid={iid} removed_by_builder is inconsistent")
    attested = value["selection_upstream_attestation"]
    # A legacy builder row has no upstream attestation; an upstream removal
    # must have one.  A clean row may have the explicit present=false
    # attestation or may predate that metadata.
    if (
        (stage == "builder_legacy" and attested)
        or (stage == "selection_upstream" and not attested)
    ):
        raise ValueError(
            f"iid={iid} selection upstream attestation is inconsistent"
        )


def _validate_source_row(
    row: Mapping[str, Any],
    *,
    bucket: str,
    source_line_number: int,
    source_fused_sha256: str,
    builder_implementation_sha256: str,
) -> tuple[str, str]:
    iid = _plain_string(
        row.get("iid"),
        context=f"source line {source_line_number} IID",
    )
    if "split" in row or "split_provenance" in row:
        raise ValueError(f"iid={iid} leaks a top-level split")
    label = row.get("r7_expansion_manifest")
    if type(label) is not dict:
        raise ValueError(f"iid={iid} has no expansion manifest label")
    visual, observation, result = _validate_qwen_evidence(row, iid=iid)
    decision = _classify(
        visual=visual,
        observation=observation,
        result=result,
    )
    family = _primary_family(row)
    expected: dict[str, Any] = {
        "schema_version": SOURCE_ROW_SCHEMA,
        "policy_version": SOURCE_POLICY_VERSION,
        "bucket": decision["bucket"],
        "classification_reason": decision["reason"],
        "source_line_number": source_line_number,
        "source_fused_sha256": source_fused_sha256,
        "builder_implementation_sha256": builder_implementation_sha256,
        "verdict": result["verdict"],
        "primary_family": family,
        "observation_validated_from": visual[
            "observation_validated_from"
        ],
        "result_validated_from": visual["result_validated_from"],
        "split_assigned": False,
        "human_label": False,
        "formal_evidence": False,
    }
    for field in (
        "action_signature",
        "negative_type",
        "negative_role",
        "quality_failures",
    ):
        if field in decision:
            expected[field] = decision[field]
    legacy = label.get("legacy_split_quarantine")
    _validate_legacy_audit(legacy, iid=iid)
    expected["legacy_split_quarantine"] = legacy
    if label != expected:
        raise ValueError(
            f"iid={iid} expansion label differs from fresh classification"
        )
    if decision["bucket"] != bucket:
        raise ValueError(
            f"iid={iid} is in {bucket}, freshly classified as "
            f"{decision['bucket']}"
        )
    return iid, family


def _validate_source_commit(
    manifest_dir: Path,
) -> tuple[
    Path,
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, str],
]:
    directory, artifacts = _strict_directory(
        manifest_dir,
        names=SOURCE_ARTIFACT_NAMES,
        context="R7 expansion manifest",
    )
    summary, summary_raw = _load_json(
        artifacts[SOURCE_SUMMARY_NAME],
        context="source summary.json",
    )
    done, done_raw = _load_json(
        artifacts[DONE_NAME],
        context="source done.json",
    )
    if (
        summary.get("schema_version") != SOURCE_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("policy_version") != SOURCE_POLICY_VERSION
    ):
        raise ValueError("source summary schema/status/policy mismatch")
    if (
        done.get("schema_version") != SOURCE_DONE_SCHEMA
        or done.get("status") != "complete"
    ):
        raise ValueError("source done schema/status mismatch")
    builder_digest = _validate_source_implementation(summary)

    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    raws: dict[str, bytes] = {}
    for name in SOURCE_DATA_NAMES:
        rows, raw = _load_canonical_jsonl(
            artifacts[name],
            context=f"source {name}",
        )
        rows_by_name[name] = rows
        raws[name] = raw

    outputs = summary.get("outputs")
    if type(outputs) is not dict or set(outputs) != set(SOURCE_DATA_NAMES):
        raise ValueError("source summary output artifact set mismatch")
    expected_output_sha: dict[str, str] = {}
    expected_output_rows: dict[str, int] = {}
    for name in SOURCE_DATA_NAMES:
        entry = outputs[name]
        if type(entry) is not dict or set(entry) != {
            "rows",
            "sha256",
            "order",
        }:
            raise ValueError(f"source summary output entry invalid: {name}")
        digest = _sha256_bytes(raws[name])
        count = len(rows_by_name[name])
        if (
            type(entry.get("rows")) is not int
            or entry["rows"] != count
            or entry.get("sha256") != digest
            or entry.get("order") != "source_fused_order_within_bucket"
        ):
            raise ValueError(f"source summary output mismatch: {name}")
        expected_output_sha[name] = digest
        expected_output_rows[name] = count
    summary_sha256 = _sha256_bytes(summary_raw)
    expected_output_sha[SOURCE_SUMMARY_NAME] = summary_sha256

    if done.get("output_sha256") != {
        name: expected_output_sha[name]
        for name in sorted(expected_output_sha)
    }:
        raise ValueError("source done output SHA chain mismatch")
    if done.get("output_rows") != {
        name: expected_output_rows[name]
        for name in sorted(expected_output_rows)
    }:
        raise ValueError("source done output row counts mismatch")
    if done.get("artifact_digest") != _object_digest(
        {
            name: expected_output_sha[name]
            for name in sorted(expected_output_sha)
        }
    ):
        raise ValueError("source done artifact_digest mismatch")
    source_artifact_digest = _sha256_field(
        done.get("artifact_digest"),
        context="source done artifact_digest",
    )
    input_summary = summary.get("input")
    if type(input_summary) is not dict:
        raise ValueError("source summary input is not an object")
    source_fused_sha256 = _sha256_field(
        input_summary.get("sha256"),
        context="source fused SHA-256",
    )
    total_rows = sum(expected_output_rows.values())
    if (
        type(input_summary.get("rows")) is not int
        or input_summary["rows"] != total_rows
        or done.get("input_rows") != total_rows
        or done.get("input_sha256") != source_fused_sha256
        or done.get("implementation_sha256") != builder_digest
    ):
        raise ValueError("source input/implementation chain mismatch")
    if summary.get("bucket_counts") != {
        SOURCE_BUCKETS[name]: expected_output_rows[name]
        for name in sorted(SOURCE_DATA_NAMES)
    }:
        raise ValueError("source bucket_counts mismatch")
    if (
        done.get("split_assigned") is not False
        or done.get("human_labels_asserted") is not False
        or done.get("formal_evidence") is not False
    ):
        raise ValueError("source done asserts forbidden labels or split")

    seen_iids: set[str] = set()
    seen_lines: set[int] = set()
    negative_roles: Counter[str] = Counter()
    for name in SOURCE_DATA_NAMES:
        bucket = SOURCE_BUCKETS[name]
        previous_line = 0
        for row in rows_by_name[name]:
            label = row.get("r7_expansion_manifest")
            if type(label) is not dict:
                raise ValueError(f"source {name} row has no label")
            line_number = label.get("source_line_number")
            if (
                type(line_number) is not int
                or line_number <= previous_line
                or not 1 <= line_number <= total_rows
            ):
                raise ValueError(
                    f"source {name} has invalid source-line ordering"
                )
            previous_line = line_number
            iid, _ = _validate_source_row(
                row,
                bucket=bucket,
                source_line_number=line_number,
                source_fused_sha256=source_fused_sha256,
                builder_implementation_sha256=builder_digest,
            )
            if iid in seen_iids:
                raise ValueError(f"duplicate source IID: {iid}")
            if line_number in seen_lines:
                raise ValueError(
                    f"duplicate source line number: {line_number}"
                )
            seen_iids.add(iid)
            seen_lines.add(line_number)
            if bucket == "negative":
                negative_roles[str(label.get("negative_role"))] += 1
    if seen_lines != set(range(1, total_rows + 1)):
        raise ValueError("source line numbers are not an exact conservation")
    if summary.get("negative_role_counts") != dict(
        sorted(negative_roles.items())
    ):
        raise ValueError("source negative_role_counts mismatch")
    captured_sha256 = {
        **{
            name: _sha256_bytes(raws[name])
            for name in SOURCE_DATA_NAMES
        },
        SOURCE_SUMMARY_NAME: _sha256_bytes(summary_raw),
        DONE_NAME: _sha256_bytes(done_raw),
    }
    for name, digest in captured_sha256.items():
        if _sha256_file(artifacts[name]) != digest:
            raise RuntimeError(
                f"source {name} changed while it was validated"
            )
    return (
        directory,
        rows_by_name,
        summary,
        done,
        source_artifact_digest,
        captured_sha256,
    )


def _candidate_from_source(
    row: Mapping[str, Any],
    *,
    cohort: str,
    source_artifact_digest: str,
) -> tuple[int, dict[str, Any]]:
    label = row["r7_expansion_manifest"]
    line_number = int(label["source_line_number"])
    iid = _plain_string(row.get("iid"), context="candidate IID")
    input_digest = _sha256_field(
        row.get("input_digest"),
        context=f"candidate input_digest for iid={iid}",
    )
    candidate: dict[str, Any] = {
        "schema_version": ROW_SCHEMA,
        "iid": iid,
        "input_digest": input_digest,
        "prompt": _plain_string(
            row.get("prompt"),
            context=f"candidate prompt for iid={iid}",
        ),
        "src_video": _plain_string(
            row.get("src_video"),
            context=f"candidate src_video for iid={iid}",
        ),
        "tgt_video": _plain_string(
            row.get("tgt_video"),
            context=f"candidate tgt_video for iid={iid}",
        ),
        "cohort": cohort,
        "primary_family": _plain_string(
            label.get("primary_family"),
            context=f"candidate primary_family for iid={iid}",
        ),
        "source_row_sha256": _object_digest(row),
        "source_artifact_digest": source_artifact_digest,
        "split_assigned": False,
        "human_label": False,
        "training_eligible": False,
    }
    if set(candidate) != CANDIDATE_ROW_FIELDS:
        raise RuntimeError("internal candidate projection field mismatch")
    return line_number, candidate


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
        for name in OUTPUT_ARTIFACT_NAMES:
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
        artifact_permissions.seal_staging_tree(
            staging,
            leave_root_writable=True,
        )
        artifact_permissions.assert_sealed_tree(
            staging,
            allow_writable_root=True,
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                f"output directory appeared during commit: {output_dir}"
            )
        os.rename(staging, output_dir)
        artifact_permissions.seal_published_root(output_dir)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if staging.exists():
            artifact_permissions.remove_staging_tree(staging)


def _strict_resume(
    output_dir: Path,
    *,
    expected: Mapping[str, bytes],
) -> None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise FileExistsError(
            f"--resume requires a regular output directory: {output_dir}"
        )
    artifact_permissions.assert_sealed_tree(output_dir)
    actual = {path.name for path in output_dir.iterdir()}
    if actual != set(expected):
        raise RuntimeError(
            "strict resume artifact set mismatch: "
            f"missing={sorted(set(expected) - actual)} "
            f"extra={sorted(actual - set(expected))}"
        )
    for name, payload in expected.items():
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"resume artifact is not regular: {path}")
        if path.read_bytes() != payload:
            raise RuntimeError(
                f"resume artifact differs from fresh derivation: {path}"
            )


def build_visual_candidate_manifest(
    *,
    manifest_dir: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate an expansion commit and project conservative candidates."""

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

    (
        source_dir,
        rows_by_name,
        source_summary,
        source_done,
        source_artifact_digest,
        source_file_sha256,
    ) = _validate_source_commit(manifest_dir)
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("output directory cannot be inside the input commit")

    ordered: list[tuple[int, dict[str, Any]]] = []
    for row in rows_by_name[POSITIVES_NAME]:
        ordered.append(
            _candidate_from_source(
                row,
                cohort="pseudo_positive",
                source_artifact_digest=source_artifact_digest,
            )
        )
    negative_role_counts: Counter[str] = Counter()
    for row in rows_by_name[NEGATIVES_NAME]:
        label = row["r7_expansion_manifest"]
        role = str(label["negative_role"])
        negative_role_counts[role] += 1
        if role == "pseudo_negative":
            ordered.append(
                _candidate_from_source(
                    row,
                    cohort="pseudo_negative",
                    source_artifact_digest=source_artifact_digest,
                )
            )
        elif role != "audit_only":
            raise ValueError(f"unsupported negative role: {role!r}")
    ordered.sort(key=lambda item: item[0])
    if len({line for line, _ in ordered}) != len(ordered):
        raise RuntimeError("candidate source-line order is not unique")
    candidates = [row for _, row in ordered]
    cohort_counts = Counter(row["cohort"] for row in candidates)
    candidate_raw = _jsonl_bytes(candidates)
    candidate_sha256 = _sha256_bytes(candidate_raw)
    implementation = _implementation_provenance()

    source_rows = int(source_summary["input"]["rows"])
    audit_only_count = negative_role_counts["audit_only"]
    review_count = len(rows_by_name[REVIEW_NAME])
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "policy_version": POLICY_VERSION,
        "input": {
            "path": str(source_dir),
            "schema_version": SOURCE_SUMMARY_SCHEMA,
            "rows": source_rows,
            "artifact_digest": source_artifact_digest,
            "summary_sha256": source_file_sha256[SOURCE_SUMMARY_NAME],
            "done_sha256": source_file_sha256[DONE_NAME],
            "source_output_sha256": source_done["output_sha256"],
        },
        "implementation": implementation,
        "output": {
            "name": CANDIDATES_NAME,
            "rows": len(candidates),
            "sha256": candidate_sha256,
            "order": "ascending_source_line_number",
            "row_encoding": "canonical_json_utf8_lf",
            "source_row_digest": "canonical_json_sha256",
        },
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "excluded_counts": {
            "audit_only_negative": audit_only_count,
            "review": review_count,
        },
        "conservation": {
            "source_rows": source_rows,
            "candidate_rows": len(candidates),
            "excluded_rows": audit_only_count + review_count,
            "exact": (
                source_rows
                == len(candidates) + audit_only_count + review_count
            ),
        },
        "policy": {
            "included": [
                "strict_original_qwen_pseudo_positive",
                "trusted_original_qwen_negative:pseudo_negative",
            ],
            "excluded": [
                "deterministic_sanitized_audit_negative:audit_only",
                "review",
            ],
        },
        "semantics": {
            "split_assigned": False,
            "human_labels_asserted": False,
            "training_eligible": False,
            "candidate_labels_are_human_truth": False,
        },
    }
    if not summary["conservation"]["exact"]:
        raise RuntimeError("candidate/exclusion conservation failed")
    summary_raw = _pretty_json_bytes(summary)
    summary_sha256 = _sha256_bytes(summary_raw)
    output_sha = {
        CANDIDATES_NAME: candidate_sha256,
        SUMMARY_NAME: summary_sha256,
    }
    done: dict[str, Any] = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "input_rows": source_rows,
        "input_artifact_digest": source_artifact_digest,
        "implementation_sha256": implementation["bundle_sha256"],
        "output_rows": len(candidates),
        "output_sha256": output_sha,
        "artifact_digest": _object_digest(output_sha),
        "split_assigned": False,
        "human_labels_asserted": False,
        "training_eligible": False,
    }
    files = {
        CANDIDATES_NAME: candidate_raw,
        SUMMARY_NAME: summary_raw,
        DONE_NAME: _pretty_json_bytes(done),
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
            "Build an unsplit visual-learning candidate projection from a "
            "strict R7 expansion manifest commit."
        )
    )
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verification-only; never creates or modifies output artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_visual_candidate_manifest(
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
        resume=bool(args.resume),
    )
    counts = summary["cohort_counts"]
    print(
        "[motive-r7-visual-candidates] "
        f"pseudo_positive={counts.get('pseudo_positive', 0)} "
        f"pseudo_negative={counts.get('pseudo_negative', 0)} "
        f"resume_verified={summary['resume_verified']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
