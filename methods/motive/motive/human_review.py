"""Prepare and merge provenance-bound human action-review labels.

Review templates are blind to upstream automation by default.  Rule, feature,
and Qwen-derived hints are only copied when explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .train_action_repr import (
    HUMAN_APPROVED_VERDICTS,
    HUMAN_REJECTED_VERDICTS,
    HUMAN_REVIEW_SCHEMA,
)

OPTIONAL_REVIEW_TEXT_FIELDS = (
    "event_type",
    "actor",
    "actor_valid",
    "instruction_aligned",
    "complete_temporal_event",
    "source_action",
    "target_action",
    "direction",
    "speed",
    "phase",
    "contact_or_interaction",
    "camera_motion",
    "preservation_ok",
    "review_confidence",
    "secondary_reviewer",
    "adjudication",
)

REVIEW_ITEM_DIGEST_FIELDS = (
    "schema_version",
    "iid",
    "input_digest",
    "prompt",
    "src_video",
    "tgt_video",
)
R7_ASSIGNMENT_FIELD = "r7_review_assignment"
R7_MEDIA_FIELD = "r7_media_binding"
R7_REVIEW_ITEM_DIGEST_FIELDS = (
    *REVIEW_ITEM_DIGEST_FIELDS,
    R7_ASSIGNMENT_FIELD,
    R7_MEDIA_FIELD,
)
R7_ASSIGNMENT_SCHEMA = "motive-r7-review-assignment-v2"
R7_MEDIA_SCHEMA = "motive-r7-review-media-binding-v1"
R7_RATE_AUDIT_REVIEW_SCHEMA = "motive-r7-rate-audit-review-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVIEWER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._@+-]{0,127}$")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _validated_input_digest(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{context} has missing/invalid input_digest; expected "
            "64 lowercase hex characters"
        )
    return value


def normalize_reviewer_id(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} reviewer ID must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    if _REVIEWER_ID_RE.fullmatch(normalized) is None:
        raise ValueError(
            f"{context} reviewer ID must match "
            "[a-z0-9][a-z0-9._@+-]{0,127}"
        )
    return normalized


def _validated_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _validated_r7_contract(
    row: Mapping[str, Any],
    *,
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    has_assignment = R7_ASSIGNMENT_FIELD in row
    has_media = R7_MEDIA_FIELD in row
    if not has_assignment and not has_media:
        return None
    if not has_assignment or not has_media:
        raise ValueError(f"{context} has an incomplete R7 review contract")
    assignment = row.get(R7_ASSIGNMENT_FIELD)
    media = row.get(R7_MEDIA_FIELD)
    if not isinstance(assignment, Mapping):
        raise ValueError(f"{context} has invalid R7 review assignment")
    if not isinstance(media, Mapping):
        raise ValueError(f"{context} has invalid R7 media binding")
    assignment_copy = dict(assignment)
    media_copy = dict(media)
    expected_assignment_fields = {
        "schema_version",
        "review_instance_id",
        "iid",
        "annotator_slot",
        "assigned_reviewer_id",
        "independent_review_required",
        "assignment_set_digest",
        "policy_sha256",
    }
    if set(assignment_copy) != expected_assignment_fields:
        raise ValueError(f"{context} R7 assignment fields differ")
    if assignment_copy.get("schema_version") != R7_ASSIGNMENT_SCHEMA:
        raise ValueError(f"{context} R7 assignment schema differs")
    iid = row.get("iid")
    if assignment_copy.get("iid") != iid:
        raise ValueError(f"{context} R7 assignment IID differs")
    slot = assignment_copy.get("annotator_slot")
    if slot not in {"primary", "secondary"}:
        raise ValueError(f"{context} R7 assignment slot differs")
    expected_independent = slot == "secondary"
    if (
        assignment_copy.get("independent_review_required")
        is not expected_independent
    ):
        raise ValueError(f"{context} R7 independence flag differs")
    assigned_reviewer = normalize_reviewer_id(
        assignment_copy.get("assigned_reviewer_id"),
        context=f"{context} assigned",
    )
    if assignment_copy.get("assigned_reviewer_id") != assigned_reviewer:
        raise ValueError(f"{context} assigned reviewer ID is not normalized")
    for field in (
        "review_instance_id",
        "assignment_set_digest",
        "policy_sha256",
    ):
        _validated_sha256(
            assignment_copy.get(field),
            context=f"{context} R7 assignment {field}",
        )
    if media_copy.get("schema_version") != R7_MEDIA_SCHEMA:
        raise ValueError(f"{context} R7 media schema differs")
    if not isinstance(media_copy.get("media_bytes_bound"), bool):
        raise ValueError(f"{context} R7 media bound flag differs")
    return assignment_copy, media_copy


def review_item_digest_fields(row: Mapping[str, Any]) -> list[str]:
    return list(
        R7_REVIEW_ITEM_DIGEST_FIELDS
        if _validated_r7_contract(row, context="review item") is not None
        else REVIEW_ITEM_DIGEST_FIELDS
    )


def _review_item_payload(
    row: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    iid = row.get("iid")
    if not isinstance(iid, str) or not iid:
        raise ValueError(f"{context} has missing/invalid iid")
    values: dict[str, str] = {}
    for field in ("prompt", "src_video", "tgt_video"):
        value = row.get(field)
        if not isinstance(value, str):
            raise ValueError(f"{context} has missing/invalid {field}")
        values[field] = value
    payload: dict[str, Any] = {
        "schema_version": HUMAN_REVIEW_SCHEMA,
        "iid": iid,
        "input_digest": _validated_input_digest(
            row.get("input_digest"),
            context=context,
        ),
        **values,
    }
    r7_contract = _validated_r7_contract(row, context=context)
    if r7_contract is not None:
        assignment, media = r7_contract
        payload[R7_ASSIGNMENT_FIELD] = assignment
        payload[R7_MEDIA_FIELD] = media
    return payload


def _review_item_digest(
    row: dict[str, Any],
    *,
    context: str,
) -> str:
    payload = _review_item_payload(row, context=context)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _r7_manifest_contract_summary(
    rows: Sequence[dict[str, Any]],
    *,
    context: str,
) -> dict[str, Any]:
    contracts = [
        _validated_r7_contract(row, context=f"{context} iid={row.get('iid')}")
        for row in rows
    ]
    present = [contract for contract in contracts if contract is not None]
    if present and len(present) != len(rows):
        raise ValueError(f"{context} mixes R7 and legacy review contracts")
    if not present:
        return {
            "r7_contract_bound": False,
            "media_bytes_bound": False,
            "assignment_set_digest": None,
            "policy_sha256": None,
            "annotator_slots": [],
        }
    assignments = [contract[0] for contract in present]
    media = [contract[1] for contract in present]
    assignment_digests = {
        str(assignment["assignment_set_digest"])
        for assignment in assignments
    }
    policy_digests = {
        str(assignment["policy_sha256"]) for assignment in assignments
    }
    if len(assignment_digests) != 1 or len(policy_digests) != 1:
        raise ValueError(f"{context} R7 contract digests differ")
    return {
        "r7_contract_bound": True,
        "media_bytes_bound": all(
            binding["media_bytes_bound"] is True for binding in media
        ),
        "assignment_set_digest": next(iter(assignment_digests)),
        "policy_sha256": next(iter(policy_digests)),
        "annotator_slots": sorted(
            {
                str(assignment["annotator_slot"])
                for assignment in assignments
            }
        ),
    }


def _atomic_jsonl(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite intentionally")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _automation_hints(row: dict[str, Any]) -> dict[str, Any]:
    qwen = row.get("qwen_evidence", {}).get("visual")
    qwen_result = (
        qwen.get("result", {})
        if isinstance(qwen, dict) and qwen.get("status") == "ok"
        else {}
    )
    qwen_observation = (
        qwen.get("observation", {})
        if isinstance(qwen, dict) and qwen.get("status") == "ok"
        else {}
    )
    return {
        "automatic_decision": row.get("final_triage", {}).get("decision"),
        "qwen_visual_verdict": qwen_result.get("verdict"),
        "qwen_confidence": qwen_result.get("confidence"),
        "qwen_action_signature": qwen_result.get("action_signature"),
        "qwen_source_action": qwen_observation.get("source_action"),
        "qwen_target_action": qwen_observation.get("target_action"),
        "qwen_source_actor_motion": qwen_observation.get(
            "source_actor_motion"
        ),
        "qwen_target_actor_motion": qwen_observation.get(
            "target_actor_motion"
        ),
        "qwen_edit_effect": qwen_result.get("edit_effect"),
        "qwen_camera_dominance": qwen_observation.get("camera_dominance"),
        "qwen_preservation_quality": qwen_observation.get(
            "preservation_quality"
        ),
        "rule_action_families": row.get("auto_rule", {}).get(
            "action_families", []
        ),
    }


def prepare(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    include_automation_hints = bool(
        getattr(args, "include_automation_hints", False)
    )
    rows = list(_iter_jsonl(input_path))
    contract_summary = _r7_manifest_contract_summary(
        rows,
        context=str(input_path),
    )
    seen: set[str] = set()
    templates: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=1):
        context = f"{input_path}:{line_number}"
        iid = row.get("iid")
        if not isinstance(iid, str) or not iid:
            raise ValueError(f"{context} has missing/invalid iid")
        if iid in seen:
            raise ValueError(f"duplicate iid={iid} in {input_path}")
        seen.add(iid)
        template: dict[str, Any] = {
            "schema_version": HUMAN_REVIEW_SCHEMA,
            "iid": iid,
            "input_digest": _validated_input_digest(
                row.get("input_digest"),
                context=context,
            ),
            "verdict": "",
            "reviewer": "",
            "action_signature": "",
            "event_type": "",
            "actor": "",
            "actor_valid": "",
            "instruction_aligned": "",
            "complete_temporal_event": "",
            "source_action": "",
            "target_action": "",
            "direction": "",
            "speed": "",
            "phase": "",
            "contact_or_interaction": "",
            "camera_motion": "",
            "preservation_ok": "",
            "event_start_frame": "",
            "event_end_frame": "",
            "review_confidence": "",
            "secondary_reviewer": "",
            "adjudication": "",
            "notes": "",
            "prompt": row.get("prompt"),
            "src_video": row.get("src_video"),
            "tgt_video": row.get("tgt_video"),
        }
        if contract_summary["r7_contract_bound"]:
            for field in (R7_ASSIGNMENT_FIELD, R7_MEDIA_FIELD):
                template[field] = json.loads(
                    json.dumps(row[field], ensure_ascii=False)
                )
        template["review_item_digest"] = _review_item_digest(
            template,
            context=context,
        )
        if include_automation_hints:
            template["automation_hints"] = _automation_hints(row)
        templates.append(template)
    _atomic_jsonl(output_path, templates, overwrite=args.overwrite)
    summary = {
        "schema_version": HUMAN_REVIEW_SCHEMA,
        "stage": "prepare",
        "input": str(input_path),
        "input_sha256": _file_digest(input_path),
        "rows": len(templates),
        "output": str(output_path),
        "output_sha256": _file_digest(output_path),
        "review_mode": (
            "automation-assisted" if include_automation_hints else "blind"
        ),
        "automation_hints_included": include_automation_hints,
        "review_item_digest_fields": (
            list(R7_REVIEW_ITEM_DIGEST_FIELDS)
            if contract_summary["r7_contract_bound"]
            else list(REVIEW_ITEM_DIGEST_FIELDS)
        ),
        **contract_summary,
        **(
            {
                "label_scope": "rate_audit_only",
                "direct_training_supervision_allowed": False,
                "training_authorized": False,
            }
            if contract_summary["r7_contract_bound"]
            else {}
        ),
        "verdicts": sorted(
            HUMAN_APPROVED_VERDICTS | HUMAN_REJECTED_VERDICTS
        ),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[motive-human-review prepare] rows={len(templates)} "
        f"mode={summary['review_mode']} output={output_path}",
        flush=True,
    )
    return 0


def merge(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser()
    labels_path = args.labels.expanduser()
    output_path = args.output.expanduser()
    manifest_sha256 = _file_digest(manifest_path)
    labels_sha256 = _file_digest(labels_path)
    manifest_rows = list(_iter_jsonl(manifest_path))
    contract_summary = _r7_manifest_contract_summary(
        manifest_rows,
        context=str(manifest_path),
    )
    by_id: dict[str, dict[str, Any]] = {}
    manifest_review_item_digests: dict[str, str] = {}
    for line_number, row in enumerate(manifest_rows, start=1):
        context = f"{manifest_path}:{line_number}"
        iid = row.get("iid")
        if not isinstance(iid, str) or not iid:
            raise ValueError(f"{context} has missing/invalid iid")
        if iid in by_id:
            raise ValueError(f"duplicate iid={iid} in {manifest_path}")
        by_id[iid] = row
        manifest_review_item_digests[iid] = _review_item_digest(
            row,
            context=context,
        )

    labels: dict[str, dict[str, Any]] = {}
    incomplete = 0
    for line_number, label in enumerate(_iter_jsonl(labels_path), start=1):
        context = f"{labels_path}:{line_number}"
        iid = label.get("iid")
        if not isinstance(iid, str) or not iid:
            raise ValueError(f"{context} has missing/invalid iid")
        if iid in labels:
            raise ValueError(f"duplicate iid={iid} in {labels_path}")
        if iid not in by_id:
            raise ValueError(f"{labels_path}:{line_number} has unknown iid={iid}")
        if label.get("schema_version") != HUMAN_REVIEW_SCHEMA:
            raise ValueError(
                f"{labels_path}:{line_number} has unsupported schema"
            )
        label_input_digest = _validated_input_digest(
            label.get("input_digest"),
            context=context,
        )
        if label_input_digest != by_id[iid].get("input_digest"):
            raise ValueError(
                f"{labels_path}:{line_number} input_digest mismatch for iid={iid}"
            )
        supplied_review_item_digest = label.get("review_item_digest")
        if (
            not isinstance(supplied_review_item_digest, str)
            or _SHA256_RE.fullmatch(supplied_review_item_digest) is None
        ):
            raise ValueError(
                f"{context} has missing/invalid review_item_digest; expected "
                "64 lowercase hex characters"
            )
        label_review_item_digest = _review_item_digest(
            label,
            context=context,
        )
        manifest_review_item_digest = manifest_review_item_digests[iid]
        if (
            supplied_review_item_digest != label_review_item_digest
            or supplied_review_item_digest != manifest_review_item_digest
        ):
            raise ValueError(
                f"{context} review_item_digest mismatch for iid={iid}; "
                "immutable review metadata changed"
            )
        verdict = str(label.get("verdict") or "").strip()
        if not verdict:
            incomplete += 1
            continue
        if verdict not in HUMAN_APPROVED_VERDICTS | HUMAN_REJECTED_VERDICTS:
            raise ValueError(
                f"{labels_path}:{line_number} has invalid verdict={verdict!r}"
            )
        reviewer = str(label.get("reviewer") or "").strip()
        if not reviewer:
            raise ValueError(
                f"{labels_path}:{line_number} reviewer is required for iid={iid}"
            )
        r7_contract = _validated_r7_contract(
            by_id[iid],
            context=f"{context} manifest contract",
        )
        if r7_contract is not None:
            assignment, _media = r7_contract
            assigned_reviewer = str(assignment["assigned_reviewer_id"])
            if reviewer != assigned_reviewer:
                raise ValueError(
                    f"{context} reviewer must exactly equal assigned R7 "
                    f"reviewer ID={assigned_reviewer!r}"
                )
        frame_values: dict[str, int | None] = {}
        for field in ("event_start_frame", "event_end_frame"):
            value = label.get(field)
            if value is None or value == "":
                frame_values[field] = None
                continue
            if isinstance(value, bool):
                raise ValueError(
                    f"{labels_path}:{line_number} {field} must be a "
                    "non-negative integer or blank"
                )
            try:
                integer = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{labels_path}:{line_number} {field} must be a "
                    "non-negative integer or blank"
                ) from error
            if integer < 0 or str(integer) != str(value).strip():
                raise ValueError(
                    f"{labels_path}:{line_number} {field} must be a "
                    "non-negative integer or blank"
                )
            frame_values[field] = integer
        start = frame_values["event_start_frame"]
        end = frame_values["event_end_frame"]
        if start is not None and end is not None and end < start:
            raise ValueError(
                f"{labels_path}:{line_number} event_end_frame precedes "
                "event_start_frame"
            )
        review = {
            "schema_version": (
                R7_RATE_AUDIT_REVIEW_SCHEMA
                if r7_contract is not None
                else HUMAN_REVIEW_SCHEMA
            ),
            "verdict": verdict,
            "reviewer": reviewer,
            "action_signature": str(
                label.get("action_signature") or ""
            ).strip(),
            "notes": str(label.get("notes") or "").strip(),
            "review_item_digest": supplied_review_item_digest,
            "label_source_sha256": labels_sha256,
        }
        review.update(
            {
                field: str(label.get(field) or "").strip()
                for field in OPTIONAL_REVIEW_TEXT_FIELDS
            }
        )
        review.update(frame_values)
        if r7_contract is not None:
            assignment, media = r7_contract
            review.update(
                {
                    "review_instance_id": assignment[
                        "review_instance_id"
                    ],
                    "annotator_slot": assignment["annotator_slot"],
                    "assigned_reviewer_id": assignment[
                        "assigned_reviewer_id"
                    ],
                    "assignment_set_digest": assignment[
                        "assignment_set_digest"
                    ],
                    "policy_sha256": assignment["policy_sha256"],
                    "media_binding_sha256": hashlib.sha256(
                        json.dumps(
                            media,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                        ).hexdigest(),
                    "label_scope": "rate_audit_only",
                    "direct_training_supervision_allowed": False,
                    "training_authorized": False,
                }
            )
        labels[iid] = review
    if _file_digest(manifest_path) != manifest_sha256:
        raise RuntimeError(f"{manifest_path} changed while it was being read")
    if _file_digest(labels_path) != labels_sha256:
        raise RuntimeError(f"{labels_path} changed while it was being read")

    merged: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in manifest_rows:
        iid = str(row["iid"])
        review = labels.get(iid)
        if review is None:
            continue
        output_row = dict(row)
        output_row["human_review"] = review
        merged.append(output_row)
        verdict = str(review["verdict"])
        counts[verdict] = counts.get(verdict, 0) + 1
    _atomic_jsonl(output_path, merged, overwrite=args.overwrite)
    summary = {
        "schema_version": HUMAN_REVIEW_SCHEMA,
        "stage": "merge",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "labels": str(labels_path),
        "labels_sha256": labels_sha256,
        "completed": len(merged),
        "incomplete": incomplete,
        "verdicts": dict(sorted(counts.items())),
        "review_item_digest_fields": (
            list(R7_REVIEW_ITEM_DIGEST_FIELDS)
            if contract_summary["r7_contract_bound"]
            else list(REVIEW_ITEM_DIGEST_FIELDS)
        ),
        **contract_summary,
        **(
            {
                "label_scope": "rate_audit_only",
                "direct_training_supervision_allowed": False,
                "training_authorized": False,
            }
            if contract_summary["r7_contract_bound"]
            else {}
        ),
        "output": str(output_path),
        "output_sha256": _file_digest(output_path),
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[motive-human-review merge] completed={len(merged)} "
        f"incomplete={incomplete} output={output_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or merge human action-review labels."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument(
        "--include-automation-hints",
        action="store_true",
        help=(
            "Include rule/feature/Qwen decisions under automation_hints. "
            "The default blind template omits them to avoid reviewer anchoring."
        ),
    )
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(handler=prepare)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--manifest", required=True, type=Path)
    merge_parser.add_argument("--labels", required=True, type=Path)
    merge_parser.add_argument("--output", required=True, type=Path)
    merge_parser.add_argument("--overwrite", action="store_true")
    merge_parser.set_defaults(handler=merge)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
