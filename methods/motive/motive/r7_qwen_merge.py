"""Strictly validate and fuse the eight R7 Qwen visual shards.

The merge is intentionally fail-closed.  It verifies the byte-level
line-modulo manifests, their v2 source markers, per-IID input binding,
execution-shard provenance, and the current visual schemas before attaching
the complete Qwen record as ``qwen_evidence.visual``.

The output directory is a small immutable-style commit:

``fused.jsonl``
    Input rows in their original order with visual evidence attached.
``summary.json``
    Deterministic provenance, artifact hashes, and verdict/family counts.
``done.json``
    Written last and bound to both preceding files by SHA-256.

No existing output is silently reused.  ``--resume`` is verification-only:
all three committed files must already exist and be byte-identical to a fresh
merge of the currently supplied inputs.
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

from .qwen_filter import (
    OBSERVATION_SCHEMA_VERSION,
    VISUAL_SCHEMA_VERSION,
    _object_digest,
    _validate_observation,
    _validate_visual,
)


R7_QWEN_MERGE_SCHEMA = "motive-r7-qwen-visual-merge-v2"
R7_QWEN_DONE_SCHEMA = "motive-r7-qwen-visual-merge-done-v2"
SHARD_MARKER_SCHEMA = "motive-qwen-shard-manifest-v2"
PARTITION_VERSION = "line_modulo_v1"
REQUIRED_SHARD_COUNT = 8
FUSED_NAME = "fused.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_json(raw: bytes, *, path: Path, line_number: int | None = None) -> Any:
    location = str(path)
    if line_number is not None:
        location += f":{line_number}"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{location} is not valid UTF-8") from error
    try:
        return json.loads(text, parse_constant=_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{location} is not strict JSON") from error


def _regular_file(path: Path, *, description: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise FileNotFoundError(
            f"{description} must be an existing regular non-symlink file: "
            f"{expanded}"
        )
    return expanded.resolve(strict=True)


def _load_jsonl(
    path: Path,
    *,
    description: str,
) -> tuple[list[dict[str, Any]], bytes, list[bytes]]:
    resolved = _regular_file(path, description=description)
    raw = resolved.read_bytes()
    physical_lines = raw.splitlines(keepends=True)
    if not physical_lines:
        raise ValueError(f"{description} is empty: {resolved}")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(physical_lines, start=1):
        if not raw_line.strip():
            raise ValueError(
                f"{description} contains a blank line: "
                f"{resolved}:{line_number}"
            )
        value = _parse_json(
            raw_line,
            path=resolved,
            line_number=line_number,
        )
        if not isinstance(value, dict):
            raise ValueError(
                f"{description} row is not an object: "
                f"{resolved}:{line_number}"
            )
        rows.append(value)
    return rows, raw, physical_lines


def _iid(
    row: Mapping[str, Any],
    *,
    source: bool,
    description: str,
) -> str:
    value = row.get("iid")
    if value is None and source:
        value = row.get("id")
    if not isinstance(value, str) or not value.strip():
        suffix = "iid/id" if source else "iid"
        raise ValueError(f"{description} has no non-empty string {suffix}")
    iid = value.strip()
    if "\x00" in iid:
        raise ValueError(f"{description} IID contains NUL")
    return iid


def _sha256_field(value: Any, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{description} must be a lowercase SHA-256 digest"
        )
    return value


def _unique_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: bool,
    description: str,
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    iids: list[str] = []
    by_iid: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        iid = _iid(row, source=source, description=description)
        if iid in by_iid:
            duplicates.append(iid)
        else:
            by_iid[iid] = row
        iids.append(iid)
    if duplicates:
        raise ValueError(
            f"{description} contains duplicate IIDs: "
            f"{sorted(set(duplicates))[:10]}"
        )
    return iids, by_iid


def _expected_manifest_bytes(
    input_lines: Sequence[bytes],
    *,
    shard_index: int,
    shard_count: int,
) -> bytes:
    # GNU awk, used by the shard launcher, emits one newline per selected
    # record even if the source's final physical line lacks a newline.
    selected = []
    for line in input_lines[shard_index::shard_count]:
        selected.append(line if line.endswith(b"\n") else line + b"\n")
    return b"".join(selected)


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


def _validate_qwen_visual(
    row: Mapping[str, Any],
    *,
    iid: str,
) -> None:
    if row.get("status") != "ok":
        raise ValueError(f"Qwen row is not successful: iid={iid}")
    if row.get("mode") != "visual":
        raise ValueError(f"Qwen row mode is not visual: iid={iid}")
    observation = row.get("observation")
    result = row.get("result")
    if not isinstance(observation, dict):
        raise ValueError(f"Qwen row lacks observation object: iid={iid}")
    if not isinstance(result, dict):
        raise ValueError(f"Qwen row lacks result object: iid={iid}")
    try:
        _validate_observation(observation)
        _validate_visual(result, observation=observation)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Qwen visual schema validation failed: iid={iid}: {error}"
        ) from error

    expected_observation_digest = _object_digest(observation)
    if row.get("observation_digest") != expected_observation_digest:
        raise ValueError(f"Qwen observation digest mismatch: iid={iid}")
    expected_result_digest = _object_digest(result)
    if row.get("result_digest") != expected_result_digest:
        raise ValueError(f"Qwen result digest mismatch: iid={iid}")
    _sha256_field(
        row.get("visual_input_digest"),
        description=f"Qwen visual_input_digest for iid={iid}",
    )

    for key in ("observation_repairs", "alignment_repairs"):
        repairs = row.get(key)
        if not isinstance(repairs, list) or not all(
            isinstance(item, dict) for item in repairs
        ):
            raise ValueError(f"Qwen {key} must be a list of objects: iid={iid}")
    for repair in row["alignment_repairs"]:
        if (
            repair.get("authoritative_context_digest")
            != expected_observation_digest
        ):
            raise ValueError(
                "Qwen alignment repair context digest mismatch: "
                f"iid={iid}"
            )

    for stage, repairs_key in (
        ("observation", "observation_repairs"),
        ("result", "alignment_repairs"),
    ):
        source = row.get(f"{stage}_validated_from")
        repairs = row[repairs_key]
        if source == "original":
            if repairs:
                raise ValueError(
                    f"Qwen {stage} claims original with repair audit: iid={iid}"
                )
        elif source == "original_sanitized":
            if stage != "result" or not any(
                repair.get("attempt") == 0
                and repair.get("status") == "ok"
                and repair.get("repair_generation_called") is False
                for repair in repairs
            ):
                raise ValueError(
                    f"Qwen {stage} sanitized provenance is invalid: iid={iid}"
                )
        elif isinstance(source, str) and source.startswith("repair_"):
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
                raise ValueError(
                    f"Qwen {stage} repair provenance is invalid: iid={iid}"
                )
        elif source != "fallback_uncertain":
            raise ValueError(
                f"Qwen {stage}_validated_from is invalid: iid={iid}"
            )

    observation_is_fallback = (
        row.get("observation_validated_from") == "fallback_uncertain"
    )
    result_is_fallback = (
        row.get("result_validated_from") == "fallback_uncertain"
    )
    if observation_is_fallback:
        fallback = row.get("observation_fallback")
        if (
            not isinstance(fallback, Mapping)
            or fallback.get("fallback_digest")
            != expected_observation_digest
        ):
            raise ValueError(
                f"Qwen observation fallback audit mismatch: iid={iid}"
            )
    if result_is_fallback:
        fallback = row.get("result_fallback")
        if (
            not isinstance(fallback, Mapping)
            or fallback.get("fallback_digest") != expected_result_digest
            or fallback.get("authoritative_context_digest")
            != expected_observation_digest
        ):
            raise ValueError(
                f"Qwen result fallback audit mismatch: iid={iid}"
            )


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
        # done.json is the terminal commit marker and is written last.
        for name in (FUSED_NAME, SUMMARY_NAME, DONE_NAME):
            path = staging / name
            payload = files[name]
            with path.open("xb") as handle:
                handle.write(payload)
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
            "strict resume output set mismatch: "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )
    for name, payload in expected.items():
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"strict resume artifact is not a regular file: {path}"
            )
        if path.read_bytes() != payload:
            raise RuntimeError(
                f"strict resume artifact differs from fresh merge: {path}"
            )


def merge_qwen_shards(
    *,
    input_path: Path,
    qwen_root: Path,
    output_dir: Path,
    shard_count: int = REQUIRED_SHARD_COUNT,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate shards and atomically emit the deterministic fused commit."""

    if isinstance(shard_count, bool) or shard_count != REQUIRED_SHARD_COUNT:
        raise ValueError(
            f"R7 Qwen merge requires exactly {REQUIRED_SHARD_COUNT} shards"
        )
    expanded_output_dir = output_dir.expanduser()
    if expanded_output_dir.is_symlink():
        raise ValueError("output directory must not be a symlink")
    output_dir = expanded_output_dir.resolve(strict=False)
    if resume:
        if not output_dir.exists():
            raise FileNotFoundError(
                "--resume is verification-only and requires an existing "
                f"output directory: {output_dir}"
            )
    elif output_dir.exists():
        raise FileExistsError(
            f"{output_dir} exists; use a new directory or strict --resume"
        )

    input_path = _regular_file(
        input_path,
        description="R7 selection input",
    )
    expanded_qwen_root = qwen_root.expanduser()
    if expanded_qwen_root.is_symlink() or not expanded_qwen_root.is_dir():
        raise FileNotFoundError(
            f"qwen root must be a regular directory: {expanded_qwen_root}"
        )
    qwen_root = expanded_qwen_root.resolve(strict=True)
    if output_dir == qwen_root or qwen_root in output_dir.parents:
        raise ValueError("output directory must not be inside qwen-root")

    input_rows, input_raw, input_lines = _load_jsonl(
        input_path,
        description="R7 selection input",
    )
    input_iids, input_by_iid = _unique_rows(
        input_rows,
        source=True,
        description="R7 selection input",
    )
    input_digests: dict[str, str] = {}
    for iid in input_iids:
        input_digests[iid] = _sha256_field(
            input_by_iid[iid].get("input_digest"),
            description=f"selection input_digest for iid={iid}",
        )
    input_sha256 = _sha256_bytes(input_raw)

    all_outputs: dict[str, dict[str, Any]] = {}
    manifest_coverage: set[str] = set()
    shard_summaries: list[dict[str, Any]] = []
    global_run_config_digest: str | None = None
    global_contract: dict[str, str] | None = None

    for shard_index in range(shard_count):
        tag = f"{shard_index:03d}"
        manifest_path = qwen_root / "manifests" / f"shard-{tag}.jsonl"
        marker_path = qwen_root / "manifests" / f"shard-{tag}.jsonl.source"
        output_path = qwen_root / "shards" / f"qwen-{tag}.jsonl"
        manifest_rows, manifest_raw, _ = _load_jsonl(
            manifest_path,
            description=f"Qwen shard {shard_index} manifest",
        )
        output_rows, output_raw, _ = _load_jsonl(
            output_path,
            description=f"Qwen shard {shard_index} output",
        )
        marker_path = _regular_file(
            marker_path,
            description=f"Qwen shard {shard_index} marker",
        )

        expected_manifest = _expected_manifest_bytes(
            input_lines,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        if manifest_raw != expected_manifest:
            raise ValueError(
                "shard manifest bytes do not match line-modulo partition: "
                f"shard={shard_index}"
            )
        manifest_sha256 = _sha256_bytes(manifest_raw)
        marker_raw = marker_path.read_bytes()
        marker = _parse_json(marker_raw, path=marker_path)
        expected_marker = {
            "partition": PARTITION_VERSION,
            "schema_version": SHARD_MARKER_SCHEMA,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "shard_rows": len(manifest_rows),
            "shard_sha256": manifest_sha256,
            "source_rows": len(input_rows),
            "source_sha256": input_sha256,
        }
        if marker != expected_marker:
            raise ValueError(
                f"shard marker v2 mismatch: shard={shard_index}"
            )

        manifest_iids, manifest_by_iid = _unique_rows(
            manifest_rows,
            source=True,
            description=f"Qwen shard {shard_index} manifest",
        )
        output_iids, output_by_iid_raw = _unique_rows(
            output_rows,
            source=False,
            description=f"Qwen shard {shard_index} output",
        )
        expected_iids = input_iids[shard_index::shard_count]
        if manifest_iids != expected_iids:
            raise ValueError(
                f"shard manifest IID order mismatch: shard={shard_index}"
            )
        if set(output_iids) != set(manifest_iids):
            missing = sorted(set(manifest_iids) - set(output_iids))[:10]
            extra = sorted(set(output_iids) - set(manifest_iids))[:10]
            raise ValueError(
                f"Qwen shard output coverage mismatch: shard={shard_index} "
                f"missing={missing} extra={extra}"
            )
        overlap = manifest_coverage.intersection(manifest_iids)
        if overlap:
            raise ValueError(
                f"IIDs occur in multiple shard manifests: {sorted(overlap)[:10]}"
            )
        manifest_coverage.update(manifest_iids)

        shard_run_config: str | None = None
        shard_config_digests: set[str] = set()
        for iid in manifest_iids:
            manifest_digest = _sha256_field(
                manifest_by_iid[iid].get("input_digest"),
                description=(
                    f"manifest input_digest for shard={shard_index} iid={iid}"
                ),
            )
            if manifest_digest != input_digests[iid]:
                raise ValueError(
                    f"manifest input_digest mismatch: "
                    f"shard={shard_index} iid={iid}"
                )
            output_row = dict(output_by_iid_raw[iid])
            if output_row.get("input_digest") != input_digests[iid]:
                raise ValueError(
                    f"Qwen output input_digest mismatch: "
                    f"shard={shard_index} iid={iid}"
                )
            if (
                type(output_row.get("execution_shard_index")) is not int
                or output_row["execution_shard_index"] != shard_index
                or type(output_row.get("execution_shard_count")) is not int
                or output_row["execution_shard_count"] != shard_count
            ):
                raise ValueError(
                    f"Qwen execution shard index/count mismatch: iid={iid}"
                )
            execution_manifest = output_row.get("execution_manifest")
            if not isinstance(execution_manifest, str):
                raise ValueError(
                    f"Qwen execution_manifest is missing: iid={iid}"
                )
            try:
                execution_manifest_path = (
                    Path(execution_manifest).expanduser().resolve(strict=True)
                )
            except (FileNotFoundError, OSError) as error:
                raise ValueError(
                    f"Qwen execution_manifest cannot be resolved: iid={iid}"
                ) from error
            if (
                execution_manifest_path != manifest_path.resolve(strict=True)
                or output_row.get("execution_manifest_sha256")
                != manifest_sha256
            ):
                raise ValueError(
                    f"Qwen execution manifest provenance mismatch: iid={iid}"
                )

            run_config_digest = _sha256_field(
                output_row.get("run_config_digest"),
                description=f"Qwen run_config_digest for iid={iid}",
            )
            config_digest = _sha256_field(
                output_row.get("config_digest"),
                description=f"Qwen config_digest for iid={iid}",
            )
            _sha256_field(
                output_row.get("implementation_digest"),
                description=f"Qwen implementation_digest for iid={iid}",
            )
            if shard_run_config is None:
                shard_run_config = run_config_digest
            elif shard_run_config != run_config_digest:
                raise ValueError(
                    f"mixed run_config_digest in shard={shard_index}"
                )
            shard_config_digests.add(config_digest)

            contract = {}
            for key in (
                "implementation_digest",
                "model_revision",
                "transformers_version",
                "mode",
            ):
                value = output_row.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"Qwen row has missing/invalid {key}: iid={iid}"
                    )
                contract[key] = value
            if global_contract is None:
                global_contract = contract
            elif global_contract != contract:
                raise ValueError(
                    f"Qwen semantic contract differs across rows: iid={iid}"
                )

            _validate_qwen_visual(output_row, iid=iid)
            all_outputs[iid] = output_row

        if len(shard_config_digests) != 1:
            raise ValueError(
                f"mixed config_digest in shard={shard_index}"
            )
        if shard_run_config is None:
            raise ValueError(f"empty Qwen shard is forbidden: {shard_index}")
        if global_run_config_digest is None:
            global_run_config_digest = shard_run_config
        elif global_run_config_digest != shard_run_config:
            raise ValueError(
                f"Qwen run configuration differs across shard={shard_index}"
            )
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "manifest_rows": len(manifest_rows),
                "manifest_sha256": manifest_sha256,
                "marker_sha256": _sha256_bytes(marker_raw),
                "output_rows": len(output_rows),
                "output_sha256": _sha256_bytes(output_raw),
                "config_digest": next(iter(shard_config_digests)),
                "run_config_digest": shard_run_config,
            }
        )

    if manifest_coverage != set(input_iids) or set(all_outputs) != set(input_iids):
        raise ValueError("eight-shard IID coverage does not equal selection input")

    fused_rows: list[dict[str, Any]] = []
    verdict_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    verdict_family_counts: dict[str, Counter[str]] = {}
    fallback_counts: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()
    validation_source_counts: Counter[str] = Counter()
    repair_generation_counts: Counter[str] = Counter()
    sanitization_counts: Counter[str] = Counter()
    for input_row, iid in zip(input_rows, input_iids):
        fused = dict(input_row)
        existing_evidence = fused.get("qwen_evidence")
        if existing_evidence is None:
            evidence: dict[str, Any] = {}
        elif isinstance(existing_evidence, Mapping):
            evidence = dict(existing_evidence)
        else:
            raise ValueError(
                f"input qwen_evidence is not an object: iid={iid}"
            )
        if evidence.get("visual") is not None:
            raise ValueError(
                f"refusing to overwrite existing qwen_evidence.visual: iid={iid}"
            )
        visual = all_outputs[iid]
        evidence["visual"] = visual
        fused["qwen_evidence"] = evidence
        fused_rows.append(fused)

        result = visual["result"]
        verdict = str(result["verdict"])
        family = _primary_family(input_row)
        verdict_counts[verdict] += 1
        family_counts[family] += 1
        verdict_family_counts.setdefault(family, Counter())[verdict] += 1
        fallback_counts["observation"] += (
            visual.get("observation_validated_from")
            == "fallback_uncertain"
        )
        fallback_counts["result"] += (
            visual.get("result_validated_from")
            == "fallback_uncertain"
        )
        for stage in ("observation", "result"):
            source = visual.get(f"{stage}_validated_from")
            validation_source_counts[f"{stage}:{source}"] += 1
        repair_counts["observation_rows"] += bool(
            visual["observation_repairs"]
        )
        repair_counts["observation_attempts"] += len(
            visual["observation_repairs"]
        )
        repair_counts["alignment_rows"] += bool(
            visual["alignment_repairs"]
        )
        repair_counts["alignment_attempts"] += len(
            visual["alignment_repairs"]
        )
        for stage, attempts in (
            ("observation", visual["observation_repairs"]),
            ("alignment", visual["alignment_repairs"]),
        ):
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    raise ValueError(
                        f"Qwen {stage} repair audit is malformed: iid={iid}"
                    )
                generated = attempt.get("repair_generation_called")
                if generated is True:
                    repair_generation_counts[f"{stage}:generated"] += 1
                elif generated is False:
                    repair_generation_counts[
                        f"{stage}:deterministic"
                    ] += 1
                elif generated is not None:
                    raise ValueError(
                        f"Qwen {stage} repair generation flag is malformed: "
                        f"iid={iid}"
                    )
                sanitizations = attempt.get("repair_sanitizations", [])
                if not isinstance(sanitizations, list):
                    raise ValueError(
                        f"Qwen {stage} sanitization audit is malformed: "
                        f"iid={iid}"
                    )
                for event in sanitizations:
                    if (
                        not isinstance(event, Mapping)
                        or not isinstance(event.get("action"), str)
                        or not event["action"]
                    ):
                        raise ValueError(
                            f"Qwen {stage} sanitization event is malformed: "
                            f"iid={iid}"
                        )
                    sanitization_counts[
                        f"{stage}:{event['action']}"
                    ] += 1

    fused_bytes = _jsonl_bytes(fused_rows)
    fused_sha256 = _sha256_bytes(fused_bytes)
    qwen_contract = dict(global_contract or {})
    qwen_contract.update(
        {
            "run_config_digest": global_run_config_digest,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "visual_schema_version": VISUAL_SCHEMA_VERSION,
        }
    )
    summary: dict[str, Any] = {
        "schema_version": R7_QWEN_MERGE_SCHEMA,
        "partition_version": PARTITION_VERSION,
        "shard_marker_schema": SHARD_MARKER_SCHEMA,
        "input": {
            "path": str(input_path),
            "rows": len(input_rows),
            "sha256": input_sha256,
        },
        "qwen_root": str(qwen_root),
        "shard_count": shard_count,
        "shards": shard_summaries,
        "qwen_contract": qwen_contract,
        "fused": {
            "name": FUSED_NAME,
            "rows": len(fused_rows),
            "sha256": fused_sha256,
            "order": "original_selection_input",
            "evidence_field": "qwen_evidence.visual",
        },
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "verdict_family_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(verdict_family_counts.items())
        },
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "repair_counts": dict(sorted(repair_counts.items())),
        "validation_source_counts": dict(
            sorted(validation_source_counts.items())
        ),
        "repair_generation_counts": dict(
            sorted(repair_generation_counts.items())
        ),
        "sanitization_counts": dict(sorted(sanitization_counts.items())),
    }
    summary_bytes = _pretty_json_bytes(summary)
    summary_sha256 = _sha256_bytes(summary_bytes)
    done: dict[str, Any] = {
        "schema_version": R7_QWEN_DONE_SCHEMA,
        "status": "complete",
        "input_rows": len(input_rows),
        "input_sha256": input_sha256,
        "fused_rows": len(fused_rows),
        "fused_sha256": fused_sha256,
        "summary_sha256": summary_sha256,
        "artifact_digest": _object_digest(
            {
                FUSED_NAME: fused_sha256,
                SUMMARY_NAME: summary_sha256,
            }
        ),
    }
    done_bytes = _pretty_json_bytes(done)
    files = {
        FUSED_NAME: fused_bytes,
        SUMMARY_NAME: summary_bytes,
        DONE_NAME: done_bytes,
    }
    if output_dir.exists() or output_dir.is_symlink():
        _strict_resume(output_dir, expected=files)
        summary["resume_verified"] = True
        return summary
    _write_atomic_directory(output_dir, files=files)
    summary["resume_verified"] = False
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly validate and fuse eight R7 Qwen visual shards."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--qwen-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=REQUIRED_SHARD_COUNT,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Verification-only reuse: existing fused/summary/done bytes must "
            "exactly match a fresh validation."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = merge_qwen_shards(
        input_path=args.input,
        qwen_root=args.qwen_root,
        output_dir=args.output_dir,
        shard_count=args.shard_count,
        resume=args.resume,
    )
    print(
        "[motive-r7-qwen-merge] "
        f"rows={summary['fused']['rows']} "
        f"sha256={summary['fused']['sha256']} "
        f"resume_verified={summary['resume_verified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
