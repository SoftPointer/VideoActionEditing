#!/usr/bin/env python3
"""Extract annotation-authoritative action metadata directly from MEV mev.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import ensure_disjoint, file_sha256, iter_jsonl, object_sha256, write_json, write_jsonl


PAIR_SCHEMA = "mev-action-edit-annotation-pair-v2"
NO_TARGET_SCHEMA = "mev-action-edit-no-target-annotation-v2"
SUMMARY_SCHEMA = "mev-action-edit-annotation-extraction-v2"
INSTRUCTION_DERIVATION = "deterministic_template_edit_action_so_that_v1"


def editing_instruction(event_caption: str) -> str:
    """Turn the exact annotation clause into an imperative without adding semantics."""

    caption = " ".join(event_caption.strip().split())
    if not caption:
        raise ValueError("target event caption is empty")
    clause = caption.rstrip(".?!").strip()
    clause = clause[:1].lower() + clause[1:]
    return f"Edit the action so that {clause}."


def _load_mev(path: Path) -> tuple[list[dict[str, Any]], dict[str, tuple[int, dict[str, Any], dict[int, tuple[int, dict[str, Any]]]]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("videos"), list):
        raise ValueError("mev.json must contain a videos list")
    videos = value["videos"]
    if value.get("total_videos") != len(videos):
        raise ValueError("mev.json total_videos does not match videos length")
    index: dict[str, tuple[int, dict[str, Any], dict[int, tuple[int, dict[str, Any]]]]] = {}
    for video_index, video in enumerate(videos):
        if not isinstance(video, dict) or not isinstance(video.get("events"), list):
            raise ValueError(f"invalid video record at index {video_index}")
        uuid = str(video.get("uuid", "")).strip()
        if not uuid or uuid in index:
            raise ValueError(f"empty or duplicate uuid: {uuid!r}")
        if not isinstance(video.get("global_prompt"), dict):
            raise ValueError(f"missing global_prompt for {uuid}")
        events: dict[int, tuple[int, dict[str, Any]]] = {}
        for event_index, event in enumerate(video["events"]):
            event_id = int(event["event_id"])
            if event_id in events:
                raise ValueError(f"duplicate event {uuid}/{event_id}")
            if not str(event.get("caption", "")).strip() or not str(event.get("filename", "")).strip():
                raise ValueError(f"event lacks caption or filename: {uuid}/{event_id}")
            events[event_id] = (event_index, event)
        if int(video.get("total_events", -1)) != len(events):
            raise ValueError(f"total_events differs for {uuid}")
        index[uuid] = (video_index, video, events)
    return videos, index


def _bind_event(candidate_event: dict[str, Any], annotation: dict[str, Any], label: str) -> None:
    expected_filename = str(annotation["filename"])
    expected_caption = str(annotation["caption"]).strip()
    if candidate_event.get("original_filename") != expected_filename:
        raise ValueError(f"{label} filename differs from mev.json")
    if candidate_event.get("event_caption") != expected_caption:
        raise ValueError(f"{label} caption differs from mev.json")
    if Path(candidate_event["video_path"]).name != expected_filename:
        raise ValueError(f"{label} video path is not bound to mev.json filename")


def _provenance(
    mev_json: Path,
    mev_sha256: str,
    video_index: int,
    event_index: int,
    uuid: str,
    event_id: int,
    event: dict[str, Any],
) -> dict[str, Any]:
    return {
        "authority": "MEV annotations/mev.json",
        "mev_json_path": str(mev_json),
        "mev_json_sha256": mev_sha256,
        "json_pointer": f"/videos/{video_index}/events/{event_index}",
        "uuid": uuid,
        "event_id": event_id,
        "event_annotation_sha256": object_sha256(event),
    }


def extract(mev_json: Path, raw_pairs: Path, no_target: Path, output_root: Path) -> dict[str, Any]:
    mev_json = mev_json.resolve(strict=True)
    source_root = mev_json.parent.parent
    _, output_root = ensure_disjoint(source_root, output_root)
    raw_pairs = raw_pairs.resolve(strict=True)
    no_target = no_target.resolve(strict=True)
    videos, index = _load_mev(mev_json)
    mev_sha256 = file_sha256(mev_json)

    paired_rows: list[dict[str, Any]] = []
    for candidate in iter_jsonl(raw_pairs):
        uuid = candidate["uuid"]
        video_index, video, events = index[uuid]
        source_id = int(candidate["source"]["event_id"])
        target_id = int(candidate["target"]["event_id"])
        source_event_index, source_event = events[source_id]
        target_event_index, target_event = events[target_id]
        _bind_event(candidate["source"], source_event, "source")
        _bind_event(candidate["target"], target_event, "target")
        global_prompt = video["global_prompt"]
        if candidate["source"].get("global_short_caption") != global_prompt.get("short_caption", ""):
            raise ValueError(f"global short caption differs for {uuid}")
        instruction = editing_instruction(target_event["caption"])
        row = {
            "schema_version": PAIR_SCHEMA,
            "pair_id": candidate["pair_id"],
            "candidate_sha256": object_sha256(candidate),
            "uuid": uuid,
            "split": candidate["split"],
            "source_event_id": source_id,
            "target_event_id": target_id,
            "source_video_path": candidate["source"]["video_path"],
            "target_video_path": candidate["target"]["video_path"],
            "instruction": instruction,
            "instruction_source": "mev.json target event caption",
            "instruction_derivation": INSTRUCTION_DERIVATION,
            "instruction_semantic_override_by_qwen_allowed": False,
            "source_action_caption": source_event["caption"],
            "target_action_caption": target_event["caption"],
            "global_prompt": global_prompt,
            "source_event_annotation": source_event,
            "target_event_annotation": target_event,
            "source_annotation_provenance": _provenance(
                mev_json, mev_sha256, video_index, source_event_index, uuid, source_id, source_event
            ),
            "target_annotation_provenance": _provenance(
                mev_json, mev_sha256, video_index, target_event_index, uuid, target_id, target_event
            ),
            "visual_audit_role": "validate_pair_compatibility_and_annotation_consistency_only",
        }
        row["semantic_row_sha256"] = object_sha256(row)
        paired_rows.append(row)

    no_target_rows: list[dict[str, Any]] = []
    for source_row in iter_jsonl(no_target):
        uuid = source_row["uuid"]
        video_index, video, events = index[uuid]
        event_id = int(source_row["source"]["event_id"])
        event_index, event = events[event_id]
        if event_id != 1:
            raise ValueError(f"no-target source is not event1: {uuid}/{event_id}")
        _bind_event(source_row["source"], event, "no-target source")
        row = {
            "schema_version": NO_TARGET_SCHEMA,
            "row_id": source_row["row_id"],
            "uuid": uuid,
            "split": source_row["split"],
            "mode": "no_target_self_generated_action_anchor",
            "source_video_path": source_row["source"]["video_path"],
            "target": None,
            "observed_source_action": event["caption"],
            "global_prompt": video["global_prompt"],
            "source_event_annotation": event,
            "source_annotation_provenance": _provenance(
                mev_json, mev_sha256, video_index, event_index, uuid, event_id, event
            ),
            "self_generated_anchor_contract": source_row["self_generated_anchor_contract"],
            "qualification_status": "source-only-annotation-bound",
        }
        row["semantic_row_sha256"] = object_sha256(row)
        no_target_rows.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    paired_path = output_root / "paired_annotation_semantics.jsonl"
    no_target_path = output_root / "no_target_sources_annotation_v2.jsonl"
    paired_count = write_jsonl(paired_path, paired_rows)
    no_target_count = write_jsonl(no_target_path, no_target_rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "mev_json_path": str(mev_json),
        "mev_json_sha256": mev_sha256,
        "mev_video_records": len(videos),
        "mev_event_records": sum(len(video["events"]) for video in videos),
        "raw_pairs_path": str(raw_pairs),
        "raw_pairs_sha256": file_sha256(raw_pairs),
        "no_target_input_path": str(no_target),
        "no_target_input_sha256": file_sha256(no_target),
        "paired_annotation_semantics": paired_count,
        "paired_annotation_semantics_path": str(paired_path),
        "paired_annotation_semantics_sha256": file_sha256(paired_path),
        "no_target_sources_annotation_v2": no_target_count,
        "no_target_sources_annotation_v2_path": str(no_target_path),
        "no_target_sources_annotation_v2_sha256": file_sha256(no_target_path),
        "instruction_authority": "mev.json target event caption",
        "instruction_derivation": INSTRUCTION_DERIVATION,
        "qwen_instruction_is_authoritative": False,
        "video_copying_performed": False,
    }
    summary["summary_digest"] = object_sha256(summary)
    write_json(output_root / "annotation_extraction_summary.json", summary)
    return summary


def verify(mev_json: Path, summary_path: Path) -> dict[str, Any]:
    mev_json = mev_json.resolve(strict=True)
    summary = json.loads(summary_path.resolve(strict=True).read_text(encoding="utf-8"))
    observed = file_sha256(mev_json)
    checks = {
        "mev_json_sha256": observed == summary.get("mev_json_sha256"),
        "paired_semantics_sha256": file_sha256(Path(summary["paired_annotation_semantics_path"]))
        == summary.get("paired_annotation_semantics_sha256"),
        "no_target_semantics_sha256": file_sha256(Path(summary["no_target_sources_annotation_v2_path"]))
        == summary.get("no_target_sources_annotation_v2_sha256"),
    }
    result = {"schema_version": "mev-action-edit-annotation-verification-v1", "checks": checks, "unchanged": all(checks.values())}
    if not result["unchanged"]:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("extract")
    build.add_argument("--mev-json", type=Path, required=True)
    build.add_argument("--raw-pairs", type=Path, required=True)
    build.add_argument("--no-target", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    check = subparsers.add_parser("verify")
    check.add_argument("--mev-json", type=Path, required=True)
    check.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        result = extract(args.mev_json, args.raw_pairs, args.no_target, args.output_root)
    else:
        result = verify(args.mev_json, args.summary)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
