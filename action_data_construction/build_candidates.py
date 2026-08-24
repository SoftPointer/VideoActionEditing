#!/usr/bin/env python3
"""Build rule-routed MEV paired candidates and event-1 no-target metadata."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import (
    bool_field,
    ensure_disjoint,
    file_sha256,
    float_field,
    object_sha256,
    source_inventory,
    split_for_uuid,
    write_json,
    write_jsonl,
)


BUILD_SCHEMA = "mev-action-edit-candidate-build-v1"
PAIR_SCHEMA = "mev-action-edit-adjacent-pair-candidate-v1"
NO_TARGET_SCHEMA = "mev-action-edit-no-target-source-v1"

STATE_CHANGE_RE = re.compile(
    r"\b(?:walk|run|move|cross|enter|exit|leave|arrive|approach|depart|"
    r"sit|stand|lie|fall|jump|climb|turn|open|close|pick|take|grab|hold|"
    r"put|place|drop|remove|give|receive|pour|eat|drink|load|unload|"
    r"mount|dismount|park|drive|ride|throw|catch|push|pull|carry)\w*\b",
    re.IGNORECASE,
)

BOOLEAN_FIELDS = (
    "has_appearance",
    "has_disappearance",
    "has_camera_motion",
    "has_focus_shift",
    "is_multi_person",
    "has_environmental_change",
    "has_occlusion",
    "has_lighting_change",
    "has_animal",
    "is_cooking",
)

QUALITY_FIELDS = (
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_imaging_quality",
    "vbench_aesthetic_quality",
)


def _event(row: dict[str, str], source_root: Path) -> dict[str, Any]:
    event_id = int(row["event_id"])
    filename = row["original_filename"].strip()
    if not filename or Path(filename).name != filename or not filename.endswith(".mp4"):
        raise ValueError(f"unsafe media filename: {filename!r}")
    path = source_root / "videos" / filename
    flags = {field: bool_field(row.get(field)) for field in BOOLEAN_FIELDS}
    quality = {field[len("vbench_") :]: float_field(row.get(field)) for field in QUALITY_FIELDS}
    return {
        "uuid": row["uuid"].strip(),
        "event_id": event_id,
        "video_path": str(path),
        "original_filename": filename,
        "event_caption": row.get("event_caption", "").strip(),
        "global_short_caption": row.get("global_short_caption", "").strip(),
        "subject_profile": row.get("subject_profile", "").strip(),
        "background": row.get("background", "").strip(),
        "start_time": float_field(row.get("start_time")),
        "end_time": float_field(row.get("end_time")),
        "duration": float_field(row.get("duration")),
        "focus_object": row.get("focus_object", "").strip() or None,
        "camera_motion_label": row.get("camera_motion_label", "").strip() or None,
        "camera_motion_desc": row.get("camera_motion_desc", "").strip() or None,
        "flags": flags,
        "quality": quality,
        "media_exists": path.is_file(),
    }


def _quality_failures(event: dict[str, Any], role: str) -> list[str]:
    failures: list[str] = []
    duration = event["duration"]
    if duration is None or duration < 1.0:
        failures.append(f"{role}_duration_below_1s")
    if duration is not None and duration > 20.0:
        failures.append(f"{role}_duration_above_20s")
    thresholds = {
        "subject_consistency": 0.70,
        "background_consistency": 0.70,
        "temporal_flickering": 0.90,
        "motion_smoothness": 0.90,
        "imaging_quality": 0.45,
    }
    for field, threshold in thresholds.items():
        value = event["quality"].get(field)
        if value is not None and value < threshold:
            failures.append(f"{role}_{field}_below_{threshold:g}")
    if not event["media_exists"]:
        failures.append(f"{role}_media_missing")
    return failures


def pair_rule(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    hard = _quality_failures(source, "source") + _quality_failures(target, "target")
    if target["event_id"] != source["event_id"] + 1:
        hard.append("non_consecutive_event_ids")
    # If the source segment introduces/removes an entity or changes illumination,
    # T0 almost certainly encodes an outcome unavailable at S0. These are
    # deterministic dependency failures, not Qwen judgments.
    for field in ("has_appearance", "has_disappearance", "has_lighting_change"):
        if source["flags"].get(field) is True:
            hard.append(f"source_{field}")

    advisory: list[str] = []
    for field in ("has_environmental_change", "has_focus_shift", "has_occlusion"):
        if source["flags"].get(field) is True:
            advisory.append(f"source_{field}")
    if source["flags"].get("has_camera_motion") is True:
        advisory.append("source_camera_motion")
    if target["flags"].get("has_camera_motion") is True:
        advisory.append("target_camera_motion")
    if STATE_CHANGE_RE.search(source["event_caption"]):
        advisory.append("source_caption_state_change_verb")
    if target["flags"].get("has_appearance") is True:
        advisory.append("target_has_appearance")
    if target["flags"].get("has_disappearance") is True:
        advisory.append("target_has_disappearance")

    return {
        "decision": "rule_reject" if hard else "qwen_required",
        "hard_reason_codes": sorted(set(hard)),
        "advisory_reason_codes": sorted(set(advisory)),
        "dependency_risk_score": min(10, len(set(advisory))),
    }


def _pair_row(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "schema_version": PAIR_SCHEMA,
        "uuid": source["uuid"],
        "source_event_id": source["event_id"],
        "target_event_id": target["event_id"],
        "source_video_path": source["video_path"],
        "target_video_path": target["video_path"],
    }
    return {
        "schema_version": PAIR_SCHEMA,
        "pair_id": object_sha256(identity),
        "uuid": source["uuid"],
        "split": split_for_uuid(source["uuid"]),
        "mode": "paired_adjacent_continuation_candidate",
        "semantic_truth_class": "continuation-derived",
        "source": source,
        "target": target,
        "rule_audit": pair_rule(source, target),
        "qwen_audit_status": "not_applicable" if pair_rule(source, target)["decision"] == "rule_reject" else "pending",
    }


def _no_target_row(event: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "schema_version": NO_TARGET_SCHEMA,
        "uuid": event["uuid"],
        "event_id": event["event_id"],
        "video_path": event["video_path"],
    }
    return {
        "schema_version": NO_TARGET_SCHEMA,
        "row_id": object_sha256(identity),
        "uuid": event["uuid"],
        "split": split_for_uuid(event["uuid"]),
        "mode": "no_target_self_generated_action_anchor",
        "source": event,
        "target": None,
        "observed_source_action": event["event_caption"],
        "self_generated_anchor_contract": {
            "source_role": "identity_scene_anchor",
            "action_instruction": None,
            "action_anchor_video": None,
            "permitted_training_uses": [
                "self_generated_action_anchor",
                "action_representation_contrastive",
                "source_preservation_or_noop",
            ],
            "forbidden_training_uses": ["flow_matching_sft_without_qualified_target"],
        },
        "qualification_status": "source_only",
    }


def _choose_smoke(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    low_to_high = sorted(
        rows,
        key=lambda row: (
            int(row["rule_audit"]["dependency_risk_score"]),
            row["target"]["event_id"],
            row["pair_id"],
        ),
    )
    high_to_low = list(reversed(low_to_high))
    # Exercise both likely accepts and likely continuation-dependency rejects,
    # and both first transitions (1->2) and later transitions.
    slots = (
        (high_to_low, False),
        (low_to_high, True),
        (high_to_low, True),
        (low_to_high, False),
    )
    while len(chosen) < count:
        progressed = False
        for pool, first_transition in slots:
            if len(chosen) >= count:
                break
            for row in pool:
                if row["uuid"] in used:
                    continue
                if (row["source"]["event_id"] == 1) != first_transition:
                    continue
                chosen.append(row)
                used.add(row["uuid"])
                progressed = True
                break
        if not progressed:
            break
    for row in low_to_high:
        if len(chosen) >= count:
            break
        if row["uuid"] not in used:
            chosen.append(row)
            used.add(row["uuid"])
    return chosen


def build(source_root: Path, output_root: Path, smoke_count: int) -> dict[str, Any]:
    source_root, output_root = ensure_disjoint(source_root, output_root)
    metadata_path = source_root / "metadata" / "events.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    filenames: list[str] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            event = _event(raw, source_root)
            if not event["uuid"]:
                raise ValueError("empty UUID")
            grouped[event["uuid"]].append(event)
            filenames.append(event["original_filename"])

    pairs: list[dict[str, Any]] = []
    no_target: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    for uuid, events in sorted(grouped.items()):
        ordered = sorted(events, key=lambda event: event["event_id"])
        event_ids = [event["event_id"] for event in ordered]
        if len(set(event_ids)) != len(event_ids):
            anomalies.append({"uuid": uuid, "reason": "duplicate_event_id", "event_ids": event_ids})
            continue
        event1 = next((event for event in ordered if event["event_id"] == 1), None)
        if event1 is None:
            anomalies.append({"uuid": uuid, "reason": "missing_event1", "event_ids": event_ids})
        else:
            no_target.append(_no_target_row(event1))
        for source, target in zip(ordered, ordered[1:]):
            pairs.append(_pair_row(source, target))

    queue = [row for row in pairs if row["rule_audit"]["decision"] == "qwen_required"]
    rejected = [row for row in pairs if row["rule_audit"]["decision"] == "rule_reject"]
    smoke = _choose_smoke(queue, min(smoke_count, len(queue)))
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "raw_paired_candidates": write_jsonl(metadata_dir / "raw_paired_candidates.jsonl", pairs),
        "qwen_audit_queue": write_jsonl(metadata_dir / "qwen_audit_queue.jsonl", queue),
        "rule_rejected_pairs": write_jsonl(metadata_dir / "rule_rejected_pairs.jsonl", rejected),
        "qwen_smoke": write_jsonl(metadata_dir / f"qwen_smoke{len(smoke)}.jsonl", smoke),
        "no_target_sources": write_jsonl(metadata_dir / "no_target_sources.jsonl", no_target),
        "source_anomalies": write_jsonl(metadata_dir / "source_anomalies.jsonl", anomalies),
    }
    split_counts = {
        name: dict(sorted(Counter(row["split"] for row in rows).items()))
        for name, rows in (("pairs", pairs), ("qwen_queue", queue), ("no_target", no_target))
    }
    summary = {
        "schema_version": BUILD_SCHEMA,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_is_read_only_contract": True,
        "video_copying_performed": False,
        "metadata_path": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "source_inventory": source_inventory(source_root, filenames),
        "uuid_count": len(grouped),
        "event_count": len(filenames),
        "counts": counts,
        "split_counts": split_counts,
        "rule_rejection_reasons": dict(sorted(Counter(code for row in rejected for code in row["rule_audit"]["hard_reason_codes"]).items())),
        "notes": [
            "Adjacent segments are continuation-derived candidates, not strict counterfactual ground truth.",
            "Every non-rule-rejected pair requires Qwen visual audit.",
            "No-target rows are never valid flow-matching SFT rows without a qualified target.",
        ],
    }
    summary["build_digest"] = object_sha256(summary)
    write_json(metadata_dir / "build_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-count", type=int, default=8)
    args = parser.parse_args()
    if args.smoke_count <= 0:
        parser.error("--smoke-count must be positive")
    print(json.dumps(build(args.source_root, args.output_root, args.smoke_count), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
