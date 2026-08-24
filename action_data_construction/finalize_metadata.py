#!/usr/bin/env python3
"""Merge rule and Qwen evidence into train-oriented JSONL/CSV metadata."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import file_sha256, iter_jsonl, object_sha256, write_json, write_jsonl


def _result_path(audit_root: Path, pair_id: str) -> Path:
    return audit_root / "results" / pair_id[:2] / f"{pair_id}.json"


def _load_result(audit_root: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    path = _result_path(audit_root, row["pair_id"])
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("pair_id") != row["pair_id"] or value.get("input_sha256") != object_sha256(row):
        raise RuntimeError(f"Qwen result identity differs: {path}")
    return value


def _training_row(candidate: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    audit = result["audit"]
    identity = {
        "schema_version": "mev-action-edit-paired-training-candidate-v1",
        "pair_id": candidate["pair_id"],
        "source_video_path": candidate["source"]["video_path"],
        "target_video_path": candidate["target"]["video_path"],
        "instruction": audit["action_instruction"],
    }
    return {
        "schema_version": "mev-action-edit-paired-training-candidate-v1",
        "row_id": object_sha256(identity),
        "pair_id": candidate["pair_id"],
        "uuid": candidate["uuid"],
        "split": candidate["split"],
        "mode": "paired_with_real_adjacent_target",
        "source_video_path": candidate["source"]["video_path"],
        "target_video_path": candidate["target"]["video_path"],
        "instruction": audit["action_instruction"],
        "source_event_caption": candidate["source"]["event_caption"],
        "target_event_caption": candidate["target"]["event_caption"],
        "source_event_id": candidate["source"]["event_id"],
        "target_event_id": candidate["target"]["event_id"],
        "target": {
            "provenance": "real-adjacent-segment",
            "semantic_truth_class": "continuation-derived",
            "qualification_status": "qwen-auto-accepted-pending-human",
            "qwen_result_path": str(_result_path(Path(result["_audit_root"]), candidate["pair_id"])),
            "qwen_result_sha256": result["_result_sha256"],
        },
        "automatic_audit": audit,
        "training_use": "sft_candidate_pending_human_qualification",
        "is_strict_counterfactual_ground_truth": False,
        "videos_copied": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_id",
        "pair_id",
        "uuid",
        "split",
        "source_video_path",
        "target_video_path",
        "instruction",
        "source_event_id",
        "target_event_id",
        "qualification_status",
        "training_use",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "qualification_status": row["target"]["qualification_status"],
                }
            )
    temporary.replace(path)


def finalize(queue_path: Path, audit_root: Path, output_root: Path, allow_incomplete: bool) -> dict[str, Any]:
    queue_path = queue_path.resolve(strict=True)
    audit_root = audit_root.resolve(strict=True)
    output_root = output_root.resolve()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    verdicts: Counter[str] = Counter()
    for candidate in iter_jsonl(queue_path):
        result = _load_result(audit_root, candidate)
        if result is None:
            missing.append(candidate)
            continue
        result["_audit_root"] = str(audit_root)
        result["_result_sha256"] = file_sha256(_result_path(audit_root, candidate["pair_id"]))
        if result.get("status") != "ok" or not isinstance(result.get("audit"), dict):
            failures.append({"candidate": candidate, "qwen_result": result})
            verdicts["error"] += 1
            continue
        verdict = result["audit"]["verdict"]
        verdicts[verdict] += 1
        evidence = {"candidate": candidate, "qwen_result": result}
        if verdict == "accept":
            accepted.append(_training_row(candidate, result))
        elif verdict == "reject":
            rejected.append(evidence)
        else:
            uncertain.append(evidence)
    if missing and not allow_incomplete:
        raise RuntimeError(f"audit is incomplete: {len(missing)} missing rows")

    output_root.mkdir(parents=True, exist_ok=True)
    counts = {
        "paired_training_candidates": write_jsonl(output_root / "paired_training_candidates.jsonl", accepted),
        "paired_qwen_rejected": write_jsonl(output_root / "paired_qwen_rejected.jsonl", rejected),
        "paired_uncertain_review": write_jsonl(output_root / "paired_uncertain_review.jsonl", uncertain),
        "paired_qwen_failures": write_jsonl(output_root / "paired_qwen_failures.jsonl", failures),
        "paired_missing_audit": write_jsonl(output_root / "paired_missing_audit.jsonl", missing),
    }
    _write_csv(output_root / "paired_training_candidates.csv", accepted)
    summary = {
        "schema_version": "mev-action-edit-finalization-v1",
        "queue_path": str(queue_path),
        "queue_sha256": file_sha256(queue_path),
        "audit_root": str(audit_root),
        "output_root": str(output_root),
        "complete": not missing,
        "counts": counts,
        "qwen_verdicts": dict(sorted(verdicts.items())),
        "strict_counterfactual_claim": False,
        "accepted_qualification": "qwen-auto-accepted-pending-human",
        "formal_sft_authorized": False,
        "video_copying_performed": False,
    }
    summary["summary_digest"] = object_sha256(summary)
    write_json(output_root / "finalization_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    summary = finalize(args.queue, args.audit_root, args.output_root, args.allow_incomplete)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
