#!/usr/bin/env python3
"""Materialize a no-authority source x branch matrix from SAIC anchor triage."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, NoReturn


SCHEMA = "saic-t2v-anchor-semantics-matrix-v2"
RECORD_SCHEMA = "saic-t2v-branch-semantics-qwen-record-v6"
SUMMARY_SCHEMA = "saic-t2v-branch-semantics-qwen-summary-v6"
SOURCE_SCHEMA = "bernini-saic-reversible-source-set-v1"
BRANCHES = ("forward", "reverse", "noop")
SPLITS = ("fit", "confirmation")
FAMILIES = ("dog", "human")
EXPECTED_SOURCES = 8
EXPECTED_RECORDS = 60
AUTHORITY = {
    "human_review": False,
    "data_selection": False,
    "training": False,
    "optimizer": False,
    "scientific_claim": False,
}


def die(message: str) -> NoReturn:
    raise SystemExit(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def plain_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        die(f"{label} must resolve to a plain file")
    return resolved


def verify_sealed(row: Mapping[str, Any], *, label: str) -> None:
    digest = row.get("receipt_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        die(f"{label} receipt digest differs")
    unsigned = {key: value for key, value in row.items() if key != "receipt_digest"}
    if object_sha256(unsigned) != digest:
        die(f"{label} receipt seal differs")


def load_sources(path: Path) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, str]]:
    path = plain_file(path, label="source manifest")
    root = json.loads(path.read_text(encoding="ascii"))
    rows = root.get("rows")
    if root.get("schema_version") != SOURCE_SCHEMA or not isinstance(rows, list):
        die("source manifest boundary differs")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            die("source row differs")
        iid = row.get("iid")
        if (
            not isinstance(iid, str)
            or iid in indexed
            or row.get("analysis_split") not in SPLITS
            or row.get("actor_family") not in FAMILIES
            or not isinstance(row.get("action_family_id"), str)
            or not isinstance(row.get("rollout_seeds"), list)
            or len(row["rollout_seeds"])
            != (2 if row["analysis_split"] == "fit" else 3)
        ):
            die("source row boundary differs")
        indexed[iid] = row
    if len(indexed) != EXPECTED_SOURCES:
        die("source count differs")
    return indexed, {"path": str(path), "sha256": file_sha256(path)}


def load_audit(
    records_path: Path, summary_path: Path
) -> tuple[list[Mapping[str, Any]], Mapping[str, str]]:
    records_path = plain_file(records_path, label="audit records")
    summary_path = plain_file(summary_path, label="audit summary")
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="ascii").splitlines()
    ]
    if len(records) != EXPECTED_RECORDS:
        die("audit record count differs")
    ids: set[str] = set()
    for row in records:
        candidate_id = row.get("candidate_id")
        if (
            row.get("schema_version") != RECORD_SCHEMA
            or row.get("branch_set") != "anchor"
            or row.get("branch") not in BRANCHES
            or row.get("analysis_split") not in SPLITS
            or row.get("actor_family") not in FAMILIES
            or not isinstance(candidate_id, str)
            or candidate_id in ids
            or row.get("authority") != AUTHORITY
            or not isinstance(row.get("deterministic_failure_codes"), list)
            or not isinstance(row.get("deterministic_branch_gate_passed"), bool)
        ):
            die("audit record boundary differs")
        verify_sealed(row, label="audit record")
        ids.add(candidate_id)
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("branch_set") != "anchor"
        or summary.get("record_count") != EXPECTED_RECORDS
        or summary.get("output_jsonl_sha256") != file_sha256(records_path)
        or summary.get("authority") != AUTHORITY
    ):
        die("audit summary boundary differs")
    verify_sealed(summary, label="audit summary")
    return records, {
        "records_path": str(records_path),
        "records_sha256": file_sha256(records_path),
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "summary_receipt_digest": str(summary["receipt_digest"]),
    }


def is_event_evidence_candidate(record: Mapping[str, Any]) -> bool:
    """Gate event completeness separately from preservation-safe target use.

    Forward/reverse action donors may carry appearance drift that makes them
    invalid synthetic targets.  This diagnostic view deliberately ignores only
    appearance change; it still requires an exact, complete, temporally coherent
    action without camera, identity-geometry, or scene shortcuts.  Noop controls
    retain the original strict branch gate.
    """
    if record.get("branch") == "noop":
        return record.get("deterministic_branch_gate_passed") is True
    observation = record.get("validated_observation")
    if not isinstance(observation, Mapping):
        return False
    expected = {
        "start_state_match": "yes",
        "requested_branch_change_present": "yes",
        "requested_change_fidelity": "exact",
        "target_action_progress": "full",
        "terminal_state_reached": "yes",
        "temporal_order_coherent": "yes",
        "identity_geometry_stable": "yes",
        "protected_scene_stable": "yes",
        "camera_motion_level": "none",
    }
    return all(observation.get(key) == value for key, value in expected.items())


def build_matrix(
    sources: Mapping[str, Mapping[str, Any]],
    records: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    by_source: dict[str, list[Mapping[str, Any]]] = {iid: [] for iid in sources}
    for record in records:
        iid = record.get("iid")
        source = sources.get(iid) if isinstance(iid, str) else None
        if (
            source is None
            or record.get("analysis_split") != source["analysis_split"]
            or record.get("actor_family") != source["actor_family"]
            or record.get("action_family_id") != source["action_family_id"]
            or record.get("seed") not in source["rollout_seeds"]
            or record.get("candidate_id")
            != f"saic-{iid}-{record['branch']}-s{record['seed']}"
        ):
            die("audit-to-source binding differs")
        by_source[iid].append(record)
    matrix = []
    for iid, source in sorted(sources.items()):
        rows = by_source[iid]
        expected = len(source["rollout_seeds"])
        branch_rows = {branch: [r for r in rows if r["branch"] == branch] for branch in BRANCHES}
        if any(
            len(branch_rows[branch]) != expected
            or {r["seed"] for r in branch_rows[branch]}
            != set(source["rollout_seeds"])
            for branch in BRANCHES
        ):
            die("source branch/seed closure differs")
        matrix.append({
            "iid": iid,
            "analysis_split": source["analysis_split"],
            "actor_family": source["actor_family"],
            "action_family_id": source["action_family_id"],
            "registered_seeds": list(source["rollout_seeds"]),
            "branches": {
                branch: {
                    "record_count": len(branch_rows[branch]),
                    "valid_model_output_count": sum(
                        row.get("validated_observation") is not None
                        for row in branch_rows[branch]
                    ),
                    "diagnostic_gate_pass_count": sum(
                        row["deterministic_branch_gate_passed"]
                        for row in branch_rows[branch]
                    ),
                    "diagnostic_gate_pass_candidate_ids": [
                        row["candidate_id"]
                        for row in branch_rows[branch]
                        if row["deterministic_branch_gate_passed"]
                    ],
                    "event_evidence_candidate_count": sum(
                        is_event_evidence_candidate(row)
                        for row in branch_rows[branch]
                    ),
                    "event_evidence_candidate_ids": [
                        row["candidate_id"]
                        for row in branch_rows[branch]
                        if is_event_evidence_candidate(row)
                    ],
                    "appearance_confounded_event_count": sum(
                        is_event_evidence_candidate(row)
                        and isinstance(row.get("validated_observation"), Mapping)
                        and row["validated_observation"].get(
                            "appearance_change_level"
                        ) != "none"
                        for row in branch_rows[branch]
                    ),
                    "failure_code_counts": dict(sorted(Counter(
                        code
                        for row in branch_rows[branch]
                        for code in row["deterministic_failure_codes"]
                    ).items())),
                }
                for branch in BRANCHES
            },
        })
    aggregates = {
        "by_split": {
            split: {
                branch: sum(
                    item["branches"][branch]["diagnostic_gate_pass_count"]
                    for item in matrix if item["analysis_split"] == split
                )
                for branch in BRANCHES
            }
            for split in SPLITS
        },
        "by_actor_family": {
            family: {
                branch: sum(
                    item["branches"][branch]["diagnostic_gate_pass_count"]
                    for item in matrix if item["actor_family"] == family
                )
                for branch in BRANCHES
            }
            for family in FAMILIES
        },
        "source_coverage": {
            branch: sum(
                item["branches"][branch]["diagnostic_gate_pass_count"] > 0
                for item in matrix
            )
            for branch in BRANCHES
        },
        "event_evidence_source_coverage": {
            branch: sum(
                item["branches"][branch]["event_evidence_candidate_count"] > 0
                for item in matrix
            )
            for branch in BRANCHES
        },
        "event_evidence_candidate_count": {
            branch: sum(
                item["branches"][branch]["event_evidence_candidate_count"]
                for item in matrix
            )
            for branch in BRANCHES
        },
        "event_evidence_by_split": {
            split: {
                branch: sum(
                    item["branches"][branch]["event_evidence_candidate_count"]
                    for item in matrix if item["analysis_split"] == split
                )
                for branch in BRANCHES
            }
            for split in SPLITS
        },
        "event_evidence_by_actor_family": {
            family: {
                branch: sum(
                    item["branches"][branch]["event_evidence_candidate_count"]
                    for item in matrix if item["actor_family"] == family
                )
                for branch in BRANCHES
            }
            for family in FAMILIES
        },
        "event_evidence_source_coverage_by_split": {
            split: {
                branch: sum(
                    item["branches"][branch]["event_evidence_candidate_count"] > 0
                    for item in matrix if item["analysis_split"] == split
                )
                for branch in BRANCHES
            }
            for split in SPLITS
        },
        "event_evidence_source_coverage_by_actor_family": {
            family: {
                branch: sum(
                    item["branches"][branch]["event_evidence_candidate_count"] > 0
                    for item in matrix if item["actor_family"] == family
                )
                for branch in BRANCHES
            }
            for family in FAMILIES
        },
        "appearance_confounded_event_count": {
            branch: sum(
                item["branches"][branch]["appearance_confounded_event_count"]
                for item in matrix
            )
            for branch in BRANCHES
        },
    }
    return matrix, aggregates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        die(f"output already exists: {args.output}")
    sources, source_binding = load_sources(args.source_manifest)
    records, audit_binding = load_audit(args.records, args.summary)
    matrix, aggregates = build_matrix(sources, records)
    unsigned = {
        "schema_version": SCHEMA,
        "status": "diagnostic_source_branch_seed_matrix_no_authority",
        "source_binding": source_binding,
        "audit_binding": audit_binding,
        "source_count": len(matrix),
        "record_count": len(records),
        "matrix": matrix,
        "aggregates": aggregates,
        "diagnostic_all_sources_have_forward_candidate": (
            aggregates["source_coverage"]["forward"] == EXPECTED_SOURCES
        ),
        "diagnostic_all_sources_have_forward_event_evidence": (
            aggregates["event_evidence_source_coverage"]["forward"]
            == EXPECTED_SOURCES
        ),
        "event_evidence_is_synthetic_target_authority": False,
        "event_evidence_requires_appearance_deconfounding": True,
        "diagnostic_all_sources_have_noop_control": (
            aggregates["source_coverage"]["noop"] == EXPECTED_SOURCES
        ),
        "seed_selection_performed": False,
        "basis_or_training_row_admission_performed": False,
        "manual_full_video_review_required": True,
        "authority": AUTHORITY,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "output": str(args.output),
        "receipt_digest": receipt["receipt_digest"],
        "source_coverage": aggregates["source_coverage"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
