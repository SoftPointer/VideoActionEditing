#!/usr/bin/env python3
"""Finalize visual audits using mev.json-derived editing instructions as authority."""

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


def _load_result(audit_root: Path, candidate: dict[str, Any]) -> dict[str, Any] | None:
    path = _result_path(audit_root, candidate["pair_id"])
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("pair_id") != candidate["pair_id"] or result.get("input_sha256") != object_sha256(candidate):
        raise RuntimeError(f"Qwen result identity differs: {path}")
    result["_path"] = str(path)
    result["_sha256"] = file_sha256(path)
    return result


def _load_semantics(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        pair_id = row["pair_id"]
        if pair_id in result:
            raise ValueError(f"duplicate annotation semantics for {pair_id}")
        result[pair_id] = row
    return result


def _semantic_ref(row: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "pair_id": row["pair_id"],
        "semantic_row_sha256": row["semantic_row_sha256"],
        "instruction_source": row["instruction_source"],
    }


def _training_row(candidate: dict[str, Any], result: dict[str, Any], semantic: dict[str, Any], semantic_path: Path) -> dict[str, Any]:
    if semantic["candidate_sha256"] != object_sha256(candidate):
        raise RuntimeError(f"annotation semantics are not bound to candidate {candidate['pair_id']}")
    instruction = semantic["instruction"]
    identity = {
        "schema_version": "mev-action-edit-paired-training-candidate-v2",
        "pair_id": candidate["pair_id"],
        "source_video_path": semantic["source_video_path"],
        "target_video_path": semantic["target_video_path"],
        "instruction": instruction,
        "instruction_semantic_row_sha256": semantic["semantic_row_sha256"],
    }
    audit = dict(result["audit"])
    visual_proposal = audit.pop("action_instruction", None)
    return {
        "schema_version": "mev-action-edit-paired-training-candidate-v2",
        "row_id": object_sha256(identity),
        "pair_id": candidate["pair_id"],
        "uuid": candidate["uuid"],
        "split": candidate["split"],
        "mode": "paired_with_real_adjacent_target",
        "source_video_path": semantic["source_video_path"],
        "target_video_path": semantic["target_video_path"],
        "instruction": instruction,
        "instruction_source": semantic["instruction_source"],
        "instruction_derivation": semantic["instruction_derivation"],
        "instruction_semantic_override_by_qwen_allowed": False,
        "source_action_caption": semantic["source_action_caption"],
        "target_action_caption": semantic["target_action_caption"],
        "global_prompt": semantic["global_prompt"],
        "source_event_annotation": semantic["source_event_annotation"],
        "target_event_annotation": semantic["target_event_annotation"],
        "source_annotation_provenance": semantic["source_annotation_provenance"],
        "target_annotation_provenance": semantic["target_annotation_provenance"],
        "annotation_semantics_ref": _semantic_ref(semantic, semantic_path),
        "target": {
            "provenance": "real-adjacent-segment",
            "semantic_truth_class": "continuation-derived",
            "qualification_status": "qwen-visual-accepted-annotation-instruction-pending-human",
            "qwen_result_path": result["_path"],
            "qwen_result_sha256": result["_sha256"],
        },
        "automatic_visual_audit": audit,
        "non_authoritative_qwen_instruction_proposal": visual_proposal,
        "training_use": "sft_candidate_pending_human_qualification",
        "is_strict_counterfactual_ground_truth": False,
        "formal_sft_authorized": False,
        "videos_copied": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "row_id", "pair_id", "uuid", "split", "source_video_path", "target_video_path",
        "instruction", "instruction_source", "source_action_caption", "target_action_caption",
        "qualification_status", "training_use",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = {field: row.get(field) for field in fields}
            value["qualification_status"] = row["target"]["qualification_status"]
            writer.writerow(value)
    temporary.replace(path)


def finalize(
    queue_path: Path,
    audit_root: Path,
    annotation_semantics_path: Path,
    output_root: Path,
    allow_incomplete: bool,
) -> dict[str, Any]:
    queue_path = queue_path.resolve(strict=True)
    audit_root = audit_root.resolve(strict=True)
    annotation_semantics_path = annotation_semantics_path.resolve(strict=True)
    semantics = _load_semantics(annotation_semantics_path)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    verdicts: Counter[str] = Counter()
    queue_ids: set[str] = set()
    for candidate in iter_jsonl(queue_path):
        pair_id = candidate["pair_id"]
        queue_ids.add(pair_id)
        semantic = semantics.get(pair_id)
        if semantic is None:
            raise RuntimeError(f"missing mev.json semantics for {pair_id}")
        result = _load_result(audit_root, candidate)
        if result is None:
            missing.append(candidate)
            continue
        evidence = {
            "candidate": candidate,
            "annotation_semantics_ref": _semantic_ref(semantic, annotation_semantics_path),
            "qwen_result": result,
        }
        if result.get("status") != "ok" or not isinstance(result.get("audit"), dict):
            failures.append(evidence)
            verdicts["error"] += 1
            continue
        verdict = result["audit"]["verdict"]
        verdicts[verdict] += 1
        if verdict == "accept":
            accepted.append(_training_row(candidate, result, semantic, annotation_semantics_path))
        elif verdict == "reject":
            rejected.append(evidence)
        else:
            uncertain.append(evidence)
    if missing and not allow_incomplete:
        raise RuntimeError(f"audit is incomplete: {len(missing)} missing rows")
    if not queue_ids.issubset(semantics):
        raise RuntimeError("annotation semantics do not cover the full Qwen queue")

    output_root = output_root.resolve()
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
        "schema_version": "mev-action-edit-finalization-v2",
        "queue_path": str(queue_path),
        "queue_sha256": file_sha256(queue_path),
        "audit_root": str(audit_root),
        "annotation_semantics_path": str(annotation_semantics_path),
        "annotation_semantics_sha256": file_sha256(annotation_semantics_path),
        "output_root": str(output_root),
        "complete": not missing,
        "counts": counts,
        "qwen_verdicts": dict(sorted(verdicts.items())),
        "instruction_authority": "MEV annotations/mev.json target event caption",
        "qwen_instruction_is_authoritative": False,
        "strict_counterfactual_claim": False,
        "accepted_qualification": "qwen-visual-accepted-annotation-instruction-pending-human",
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
    parser.add_argument("--annotation-semantics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = finalize(args.queue, args.audit_root, args.annotation_semantics, args.output_root, args.allow_incomplete)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
