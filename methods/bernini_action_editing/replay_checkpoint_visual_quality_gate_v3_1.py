#!/usr/bin/env python3
"""Replay gate v3.1 on the frozen v3 cohort and explicit counterexamples.

The legacy 7-PASS/13-FAIL inventory is reused only as labelled replay, not as
independent validation.  An optional counterexample manifest evaluates clear
videos under a separate NOT_HARD/HARD matrix so an ``unresolved`` result is
not misreported as either a hard artifact or a promotion pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import checkpoint_visual_quality_gate_v3_1 as gate
import replay_checkpoint_visual_quality_gate_v3 as v3_replay


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "feature_extractor_schema_version": report.get(
            "feature_extractor_schema_version"
        ),
        "feature_extractor_tool_sha256": report.get(
            "feature_extractor_tool_sha256"
        ),
        "tool_sha256": report.get("tool_sha256"),
        "status": report.get("status"),
        "passed": report.get("passed") is True,
        "hard_artifact_failure": report.get("hard_artifact_failure"),
        "unresolved": report.get("unresolved"),
        "failure_codes": report.get("failure_codes", []),
        "unresolved_codes": report.get("unresolved_codes", []),
        "evidence_families": report.get("decision", {}).get(
            "evidence_families", {}
        ),
        "scalars_by_scale": {
            key: value["scalars"]
            for key, value in report.get("features", {})
            .get("scales", {})
            .items()
        },
    }


def _decoder_cache():
    cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    def decode(path: Path):
        resolved = path.expanduser().resolve(strict=True)
        key = str(resolved)
        if key not in cache:
            cache[key] = gate.decode_video_exact81_multiscale(resolved)
        return cache[key]

    return decode


def replay_labelled(public: Path) -> dict[str, Any]:
    public = public.expanduser().resolve(strict=True)
    manifest_path = public / "diagnosis/manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    decode = _decoder_cache()
    results = []

    for row in v3_replay._rows(public):
        candidate = row["candidate"].resolve(strict=True)
        candidate_sha = _sha256(candidate)
        if "routepair_arm" in row:
            routepair = manifest.get("route_pair_s279", {})
            source_row = routepair.get("source", {})
            failure_codes = []
            if source_row.get("copied") is not True or not source_row.get(
                "local_path"
            ):
                failure_codes.append(
                    "authenticated_routepair_source_not_available_locally"
                )
            if routepair.get("frozen_base_supplied_to_gate") is not True:
                failure_codes.append(
                    "authenticated_routepair_frozen_base_not_supplied"
                )
            report = {
                "schema_version": gate.SCHEMA_VERSION,
                "feature_extractor_schema_version": (
                    gate.FEATURE_EXTRACTOR_SCHEMA_VERSION
                ),
                "feature_extractor_tool_sha256": (
                    gate.FEATURE_EXTRACTOR_TOOL_SHA256
                ),
                "tool_sha256": gate.TOOL_SHA256,
                "status": "error",
                "passed": False,
                "hard_artifact_failure": None,
                "unresolved": True,
                "failure_codes": failure_codes,
                "unresolved_codes": [
                    "quality_gate_authenticated_references_unavailable"
                ],
            }
            reference_resolution = {
                "manifest_sha256": _sha256(manifest_path),
                "source_expected_sha256": source_row.get("remote_sha256"),
                "source_copied": source_row.get("copied"),
                "frozen_base_supplied_to_old_gate": routepair.get(
                    "frozen_base_supplied_to_gate"
                ),
                "case00_reference_substitution_forbidden": True,
                "resolved": False,
            }
        else:
            source_frames, source_identity = decode(row["source"])
            base_frames, base_identity = decode(row["base"])
            candidate_frames, candidate_identity = decode(candidate)
            report = gate.evaluate_visual_quality(
                source_frames,
                candidate_frames,
                frozen_base_frames_by_scale=base_frames,
                metadata={
                    "sample_id": row["case_id"],
                    "checkpoint_step": row["step"],
                    "input_sha256": {
                        "source": source_identity["sha256"],
                        "candidate": candidate_identity["sha256"],
                        "frozen_base": base_identity["sha256"],
                    },
                },
            )
            reference_resolution = {
                "resolved": True,
                "source_sha256": source_identity["sha256"],
                "base_sha256": base_identity["sha256"],
            }
        predicted = "PASS" if report.get("passed") is True else "FAIL"
        results.append(
            {
                "case_id": row["case_id"],
                "expected": row["expected"],
                "predicted_fail_closed_binary": predicted,
                "correct": predicted == row["expected"],
                "candidate_sha256": candidate_sha,
                "reference_resolution": reference_resolution,
                "gate": _summary(report),
            }
        )

    matrix = {
        "expected_PASS_predicted_PASS": sum(
            row["expected"] == "PASS"
            and row["predicted_fail_closed_binary"] == "PASS"
            for row in results
        ),
        "expected_PASS_predicted_FAIL": sum(
            row["expected"] == "PASS"
            and row["predicted_fail_closed_binary"] == "FAIL"
            for row in results
        ),
        "expected_FAIL_predicted_PASS": sum(
            row["expected"] == "FAIL"
            and row["predicted_fail_closed_binary"] == "PASS"
            for row in results
        ),
        "expected_FAIL_predicted_FAIL": sum(
            row["expected"] == "FAIL"
            and row["predicted_fail_closed_binary"] == "FAIL"
            for row in results
        ),
    }
    return {
        "schema_version": (
            "bernini-checkpoint-visual-quality-gate-v3.1-labelled-replay-v1"
        ),
        "validation_status": "labelled_replay_not_independent_validation",
        "gate_schema_version": gate.SCHEMA_VERSION,
        "fail_closed_binary_rule": (
            "PASS only when gate passed=true; fail, unresolved, and error map to FAIL"
        ),
        "row_count": len(results),
        "expected_pass_count": 7,
        "expected_fail_count": 13,
        "confusion_matrix": matrix,
        "all_labels_replayed_correctly": all(row["correct"] for row in results),
        "rows": results,
    }


def _counterexample_rows(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.expanduser().resolve(strict=True).read_text("utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("rows"), list):
        raise ValueError("counterexample manifest must contain a rows array")
    rows = manifest["rows"]
    if not rows:
        raise ValueError("counterexample manifest rows must not be empty")
    required = {
        "case_id",
        "source",
        "candidate",
        "frozen_base",
        "checkpoint_step",
        "expected_hard_artifact_failure",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"counterexample row {index} is incomplete")
        if row["expected_hard_artifact_failure"] is not False:
            raise ValueError(
                f"counterexample row {index} must expect no hard artifact"
            )
    return rows


def replay_counterexamples(manifest_path: Path) -> dict[str, Any]:
    decode = _decoder_cache()
    results = []
    for row in _counterexample_rows(manifest_path):
        paths = {
            "source": Path(row["source"]).expanduser().resolve(strict=True),
            "candidate": Path(row["candidate"]).expanduser().resolve(strict=True),
            "frozen_base": Path(row["frozen_base"])
            .expanduser()
            .resolve(strict=True),
        }
        decoded = {name: decode(path) for name, path in paths.items()}
        identities = {name: value[1] for name, value in decoded.items()}
        for name, expected_sha in row.get("expected_sha256", {}).items():
            if identities[name]["sha256"] != expected_sha:
                raise ValueError(
                    f"{row['case_id']}: {name} SHA differs from manifest"
                )
        report = gate.evaluate_visual_quality(
            decoded["source"][0],
            decoded["candidate"][0],
            frozen_base_frames_by_scale=decoded["frozen_base"][0],
            metadata={
                "sample_id": row["case_id"],
                "checkpoint_step": int(row["checkpoint_step"]),
                "input_sha256": {
                    name: identity["sha256"]
                    for name, identity in identities.items()
                },
                "audit_label": "clear_counterexample_not_hard_artifact",
            },
        )
        predicted_hard = report.get("hard_artifact_failure") is True
        results.append(
            {
                "case_id": row["case_id"],
                "expected": "NOT_HARD",
                "predicted": "HARD" if predicted_hard else "NOT_HARD",
                "correct": not predicted_hard,
                "input_sha256": {
                    name: identity["sha256"]
                    for name, identity in identities.items()
                },
                "gate": _summary(report),
            }
        )
    matrix = {
        "expected_NOT_HARD_predicted_NOT_HARD": sum(
            row["predicted"] == "NOT_HARD" for row in results
        ),
        "expected_NOT_HARD_predicted_HARD": sum(
            row["predicted"] == "HARD" for row in results
        ),
    }
    return {
        "schema_version": (
            "bernini-checkpoint-visual-quality-gate-v3.1-counterexample-replay-v1"
        ),
        "validation_status": "task_labelled_clear_counterexample_replay",
        "gate_schema_version": gate.SCHEMA_VERSION,
        "row_count": len(results),
        "confusion_matrix": matrix,
        "all_counterexamples_avoid_hard_failure": all(
            row["correct"] for row in results
        ),
        "rows": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-public", type=Path)
    parser.add_argument("--counterexample-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.site_public is None and args.counterexample_manifest is None:
        raise SystemExit("at least one replay input is required")
    report: dict[str, Any] = {
        "schema_version": (
            "bernini-checkpoint-visual-quality-gate-v3.1-replay-suite-v1"
        ),
        "gate_schema_version": gate.SCHEMA_VERSION,
        "gate_tool_sha256": gate.TOOL_SHA256,
        "feature_extractor_schema_version": (
            gate.FEATURE_EXTRACTOR_SCHEMA_VERSION
        ),
        "feature_extractor_tool_sha256": gate.FEATURE_EXTRACTOR_TOOL_SHA256,
    }
    passed = True
    if args.site_public is not None:
        report["labelled_replay"] = replay_labelled(args.site_public)
        passed &= bool(report["labelled_replay"]["all_labels_replayed_correctly"])
    if args.counterexample_manifest is not None:
        report["counterexample_replay"] = replay_counterexamples(
            args.counterexample_manifest
        )
        passed &= bool(
            report["counterexample_replay"][
                "all_counterexamples_avoid_hard_failure"
            ]
        )
    report["all_requested_replays_passed"] = passed
    gate.write_json_atomic(args.output, report)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
