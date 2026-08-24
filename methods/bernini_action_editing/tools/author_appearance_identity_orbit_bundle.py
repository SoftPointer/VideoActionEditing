#!/usr/bin/env python3
"""Publish externally reviewed appearance-orbit seals and a closed RV2V-4 v3 spec.

The input decision file is content addressed and explicitly records every
full-video qualification gate.  This authoring step is separate from both the
native Bernini generator and the VAE materializer.  It never infers a pass: a
missing or false gate fails closed.

Publication is create-only.  Qualification files and the materialization spec
are written first; ``receipt.json`` is written last and is the commit marker.
An interrupted output directory is deliberately left incomplete and must not
be reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import appearance_counterfactual_identity_orbit as orbit  # noqa: E402
import source_self_runtime as runtime  # noqa: E402


DECISION_SCHEMA = "bernini-appearance-identity-orbit-review-decisions-v1"
RECEIPT_SCHEMA = "bernini-appearance-identity-orbit-authoring-receipt-v2"
_SAFE_OUTPUT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_TOP_KEYS = frozenset(
    {
        "schema_version",
        "qualifier_id",
        "protocol_sha256",
        "downstream_training_results_seen",
        "rows",
    }
)
_ROW_KEYS = frozenset(
    {"iid", "source", "variant_a", "variant_b", "qualification_gates"}
)
_SOURCE_KEYS = frozenset({"video_path", "video_sha256"})
_VARIANT_KEYS = frozenset(
    {
        "video_path",
        "video_sha256",
        "native_arm",
        "native_receipt_path",
        "native_receipt_file_sha256",
        "native_receipt_digest",
    }
)


class OrbitAuthoringError(RuntimeError):
    """Raised before an ambiguous decision or mutable output is accepted."""


def _closed(value: Any, keys: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise OrbitAuthoringError(
            f"{label} field closure differs: expected={sorted(keys)} actual={actual}"
        )
    return value


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise OrbitAuthoringError(f"{label} contains non-finite constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise OrbitAuthoringError(f"{label} contains duplicate key {key!r}")
            output[key] = value
        return output

    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OrbitAuthoringError(f"cannot decode {label}: {error}") from error
    if not isinstance(value, dict):
        raise OrbitAuthoringError(f"{label} root must be one object")
    return value


def _read_bound_json(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OrbitAuthoringError(f"{label} must be a plain file")
    try:
        expected = runtime.require_sha256(expected_sha256, label=f"{label} SHA-256")
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except (OSError, runtime.SourceSelfRuntimeError) as error:
        raise OrbitAuthoringError(str(error)) from error
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise OrbitAuthoringError(f"{label} changed while reading")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise OrbitAuthoringError(f"{label} SHA-256 differs")
    return _strict_json(raw, label=label)


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = orbit.canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _output_root(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or requested.suffix
        or _SAFE_OUTPUT.fullmatch(requested.name) is None
    ):
        raise OrbitAuthoringError("output must be an absolute safe suffix-free directory")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise OrbitAuthoringError(f"output parent is unavailable: {error}") from error
    if parent.is_symlink() or not parent.is_dir() or requested != parent / requested.name:
        raise OrbitAuthoringError("output path/parent is not canonical")
    if requested.exists() or requested.is_symlink():
        raise OrbitAuthoringError("output is create-only")
    return requested


def _validate_decisions(value: Mapping[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    decisions = _closed(value, _TOP_KEYS, label="review decisions")
    if decisions.get("schema_version") != DECISION_SCHEMA:
        raise OrbitAuthoringError("review decision schema differs")
    qualifier_id = decisions.get("qualifier_id")
    if type(qualifier_id) is not str or not qualifier_id.strip() or "\x00" in qualifier_id:
        raise OrbitAuthoringError("qualifier_id is invalid")
    try:
        protocol_sha = runtime.require_sha256(
            decisions.get("protocol_sha256"), label="protocol SHA-256"
        )
    except runtime.SourceSelfRuntimeError as error:
        raise OrbitAuthoringError(str(error)) from error
    if decisions.get("downstream_training_results_seen") is not False:
        raise OrbitAuthoringError("review must be blind to downstream training results")
    raw_rows = decisions.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise OrbitAuthoringError("review decisions require at least one row")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(raw_rows):
        row = _closed(raw_row, _ROW_KEYS, label=f"review row {index}")
        iid = row.get("iid")
        if type(iid) is not str or iid in seen:
            raise OrbitAuthoringError(f"review row {index} IID is invalid/duplicate")
        seen.add(iid)
        source = dict(_closed(row.get("source"), _SOURCE_KEYS, label=f"{iid} source"))
        variants = {
            name: dict(_closed(row.get(name), _VARIANT_KEYS, label=f"{iid} {name}"))
            for name in orbit.GENERATED_MEMBER_NAMES
        }
        members = {
            "source": source,
            **{
                name: {
                    "video_path": member["video_path"],
                    "video_sha256": member["video_sha256"],
                    "native_arm": member["native_arm"],
                }
                for name, member in variants.items()
            },
        }
        expected_seal = orbit.qualification_seal_body(
            iid=iid,
            members=members,
            qualifier_id=qualifier_id,
            protocol_sha256=protocol_sha,
        )
        if row.get("qualification_gates") != expected_seal["qualification_gates"]:
            raise OrbitAuthoringError(f"{iid} has a missing/false/extra qualification gate")
        rows.append(
            {
                "iid": iid,
                "source": source,
                "variant_a": variants["variant_a"],
                "variant_b": variants["variant_b"],
                "seal": expected_seal,
            }
        )
    return qualifier_id, protocol_sha, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--expected-decisions-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    decisions_path = args.decisions.expanduser().resolve(strict=True)
    protocol_path = args.protocol.expanduser().resolve(strict=True)
    decisions = _read_bound_json(
        decisions_path, args.expected_decisions_sha256, label="review decisions"
    )
    qualifier_id, protocol_sha, rows = _validate_decisions(decisions)
    if args.expected_protocol_sha256 != protocol_sha:
        raise OrbitAuthoringError("CLI and decision protocol SHA-256 differ")
    _read_bound_json(protocol_path, protocol_sha, label="qualification protocol")
    output = _output_root(args.output)
    output.mkdir(mode=0o750, exist_ok=False)
    runtime.fsync_directory(output.parent)

    spec_rows: list[dict[str, Any]] = []
    qualification_bindings: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        qualification_path = output / f"{row['iid']}.qualification.json"
        file_sha = _write_exclusive_json(qualification_path, row["seal"])
        binding = {
            "path": str(qualification_path),
            "file_sha256": file_sha,
            "digest": row["seal"]["receipt_digest"],
        }
        qualification_bindings[row["iid"]] = binding
        spec_rows.append(
            {
                "iid": row["iid"],
                "source": row["source"],
                "variant_a": row["variant_a"],
                "variant_b": row["variant_b"],
                "qualification": binding,
            }
        )

    spec = orbit.build_materialization_spec(spec_rows)
    spec_path = output / "materialization_spec.json"
    spec_file_sha = _write_exclusive_json(spec_path, spec)
    audit = orbit.FileMutationAudit()
    loaded = orbit.load_materialization_spec(
        spec_path, expected_sha256=spec_file_sha, audit=audit
    )
    mutation_records = audit.finalize()
    if len(loaded.rows) != len(rows) or any(
        not row.scientific_use_authorized for row in loaded.rows
    ):
        raise OrbitAuthoringError("published materialization spec did not revalidate")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "complete": True,
        "create_only": True,
        "receipt_is_commit_marker": True,
        "authoring_program": {
            "path": str(Path(__file__).resolve(strict=True)),
            "file_sha256": runtime.file_sha256(Path(__file__).resolve(strict=True)),
        },
        "review_decisions": {
            "path": str(decisions_path),
            "file_sha256": args.expected_decisions_sha256,
            "qualifier_id": qualifier_id,
            "downstream_training_results_seen": False,
        },
        "qualification_protocol": {
            "path": str(protocol_path),
            "file_sha256": protocol_sha,
        },
        "qualification_bindings": qualification_bindings,
        "materialization_spec": {
            "path": str(spec_path),
            "file_sha256": spec_file_sha,
            "digest": loaded.digest,
            "rows": len(loaded.rows),
            "schema_version": orbit.SPEC_SCHEMA,
            "reference_encoding_contract": dict(
                loaded.reference_encoding_contract
            ),
        },
        "input_mutation_audit": {
            "files": list(mutation_records),
            "all_files_stable": all(
                item["pre_post_stat_and_hash_stable"] for item in mutation_records
            ),
        },
        "authorized_claim": "appearance-counterfactual factor-exchange pretext only",
        "action_editing_claim_authorized": False,
    }
    receipt["receipt_digest"] = orbit.object_sha256(receipt)
    _write_exclusive_json(output / "receipt.json", receipt)
    runtime.fsync_directory(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "spec": str(spec_path),
                "spec_sha256": spec_file_sha,
                "receipt_digest": receipt["receipt_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DECISION_SCHEMA",
    "OrbitAuthoringError",
    "RECEIPT_SCHEMA",
    "build_parser",
    "main",
]
