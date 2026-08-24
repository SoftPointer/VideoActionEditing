#!/usr/bin/env python3
"""Replay the labelled local 7-PASS/13-FAIL cohort through quality gate v3.

This is labelled replay, not independent validation.  The two route-pair rows
are intentionally fail-closed when their manifest-authenticated source/base
references are unavailable locally; no case00 reference is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import checkpoint_visual_quality_gate_v3 as gate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _rows(public: Path) -> list[dict[str, Any]]:
    case00_source = public / "assets/media/case00-source.mp4"
    case00_base = public / "assets/media/case00-base.mp4"
    pass_candidates = (
        ("diagnostic-s1-case00", public / "diagnosis/media/direct-routeoff-s00000001-case00.mp4", 1),
        ("diagnostic-s4-case00", public / "diagnosis/media/direct-routeoff-s00000004-case00.mp4", 4),
        ("diagnostic-s8-case00", public / "diagnosis/media/direct-routeoff-s00000008-case00.mp4", 8),
        ("v16r4-s1-case00", public / "v16r4-s1/case00-v16r4-s1-routeoff-recovery.mp4", 1),
        ("v16r4-s8-case00", public / "v16r4-s8/direct-routeoff-s00000008-case00.mp4", 8),
        ("v16r5-s8-case00", public / "v16r5-s8/direct-routeoff-s00000008-case00.mp4", 8),
        ("v16r5-s32-case00", public / "v16r5-s32/direct-routeoff-s00000032-case00.mp4", 32),
    )
    rows = [
        {
            "case_id": name,
            "expected": "PASS",
            "source": case00_source,
            "base": case00_base,
            "candidate": candidate,
            "step": step,
        }
        for name, candidate, step in pass_candidates
    ]
    for index in range(8):
        rows.append(
            {
                "case_id": f"heldout-case{index:02d}-v16r3",
                "expected": "FAIL",
                "source": public / f"assets/media/case{index:02d}-source.mp4",
                "base": public / f"assets/media/case{index:02d}-base.mp4",
                "candidate": public / f"assets/media/case{index:02d}-v16r3.mp4",
                "step": 644,
            }
        )
    for step in (32, 64, 128):
        rows.append(
            {
                "case_id": f"diagnostic-s{step}-case00",
                "expected": "FAIL",
                "source": case00_source,
                "base": case00_base,
                "candidate": public / f"diagnosis/media/direct-routeoff-s{step:08d}-case00.mp4",
                "step": step,
            }
        )
    for arm in ("off", "on"):
        rows.append(
            {
                "case_id": f"routepair-s279-s644-{arm}",
                "expected": "FAIL",
                "candidate": public / f"diagnosis/media/routepair-s279-s00000644-{arm}.mp4",
                "step": 644,
                "routepair_arm": arm,
            }
        )
    if len(rows) != 20:
        raise AssertionError("labelled replay must contain exactly 20 rows")
    return rows


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    scale_values = {
        key: value["scalars"]
        for key, value in report.get("features", {}).get("scales", {}).items()
    }
    return {
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "passed": report.get("passed") is True,
        "hard_artifact_failure": report.get("hard_artifact_failure"),
        "unresolved": report.get("unresolved"),
        "failure_codes": report.get("failure_codes", []),
        "unresolved_codes": report.get("unresolved_codes", []),
        "evidence_families": report.get("decision", {}).get("evidence_families", {}),
        "scalars_by_scale": scale_values,
    }


def replay(public: Path) -> dict[str, Any]:
    public = public.expanduser().resolve(strict=True)
    manifest_path = public / "diagnosis/manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    decoded_cache = {}

    def decode_reference(path: Path):
        resolved = path.resolve(strict=True)
        key = str(resolved)
        if "case00-" not in resolved.name:
            return gate.decode_video_exact81_multiscale(resolved)
        if key not in decoded_cache:
            decoded_cache[key] = gate.decode_video_exact81_multiscale(resolved)
        return decoded_cache[key]

    results = []
    for row in _rows(public):
        candidate = row["candidate"].resolve(strict=True)
        candidate_sha = _sha256(candidate)
        if "routepair_arm" in row:
            routepair = manifest.get("route_pair_s279", {})
            source_row = routepair.get("source", {})
            failure_codes = []
            if source_row.get("copied") is not True or not source_row.get("local_path"):
                failure_codes.append("authenticated_routepair_source_not_available_locally")
            if routepair.get("frozen_base_supplied_to_gate") is not True:
                failure_codes.append("authenticated_routepair_frozen_base_not_supplied")
            report = {
                "schema_version": gate.SCHEMA_VERSION,
                "status": "error",
                "passed": False,
                "hard_artifact_failure": None,
                "unresolved": True,
                "failure_codes": failure_codes,
                "unresolved_codes": ["quality_gate_authenticated_references_unavailable"],
            }
            reference_resolution = {
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "source_expected_sha256": source_row.get("remote_sha256"),
                "source_copied": source_row.get("copied"),
                "frozen_base_supplied_to_old_gate": routepair.get("frozen_base_supplied_to_gate"),
                "case00_reference_substitution_forbidden": True,
                "resolved": False,
            }
        else:
            source_frames, source_identity = decode_reference(row["source"])
            base_frames, base_identity = decode_reference(row["base"])
            candidate_frames, candidate_identity = gate.decode_video_exact81_multiscale(
                candidate
            )
            report = gate.evaluate_visual_quality(
                source_frames,
                candidate_frames,
                frozen_base_frames_by_scale=base_frames,
                metadata={
                    "sample_id": row["case_id"],
                    "checkpoint_step": row["step"],
                    "inputs": {
                        "source": source_identity,
                        "candidate": candidate_identity,
                        "frozen_base": base_identity,
                    },
                },
            )
            reference_resolution = {
                "resolved": True,
                "source_path": source_identity["path"],
                "source_sha256": source_identity["sha256"],
                "base_path": base_identity["path"],
                "base_sha256": base_identity["sha256"],
            }
        predicted = "PASS" if report.get("passed") is True else "FAIL"
        results.append(
            {
                "case_id": row["case_id"],
                "expected": row["expected"],
                "predicted_fail_closed_binary": predicted,
                "correct": predicted == row["expected"],
                "candidate_path": str(candidate),
                "candidate_sha256": candidate_sha,
                "reference_resolution": reference_resolution,
                "gate": _summary(report),
            }
        )

    matrix = {
        "expected_PASS_predicted_PASS": sum(
            item["expected"] == "PASS" and item["predicted_fail_closed_binary"] == "PASS"
            for item in results
        ),
        "expected_PASS_predicted_FAIL": sum(
            item["expected"] == "PASS" and item["predicted_fail_closed_binary"] == "FAIL"
            for item in results
        ),
        "expected_FAIL_predicted_PASS": sum(
            item["expected"] == "FAIL" and item["predicted_fail_closed_binary"] == "PASS"
            for item in results
        ),
        "expected_FAIL_predicted_FAIL": sum(
            item["expected"] == "FAIL" and item["predicted_fail_closed_binary"] == "FAIL"
            for item in results
        ),
    }
    return {
        "schema_version": "bernini-checkpoint-visual-quality-gate-v3-labelled-replay-v1",
        "validation_status": "labelled_replay_not_independent_validation",
        "label_source": "task-provided audit labels: 7 early clean PASS, 13 invalid FAIL",
        "gate_schema_version": gate.SCHEMA_VERSION,
        "fail_closed_binary_rule": "PASS only when gate passed=true; fail, unresolved, and error map to FAIL",
        "row_count": len(results),
        "expected_pass_count": 7,
        "expected_fail_count": 13,
        "confusion_matrix": matrix,
        "all_labels_replayed_correctly": all(item["correct"] for item in results),
        "rows": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = replay(args.site_public)
    gate.write_json_atomic(args.output, report)
    return 0 if report["all_labels_replayed_correctly"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
