#!/usr/bin/env python3
"""Diagnostic Qwen3-VL audit of SAIC counterfactual T2V branch semantics.

The audit is intentionally independent of the generation receipt's pending
semantic flags.  It shows one chronological frame mosaic plus the untrusted
registered branch specification to a frozen local VLM, validates closed JSON,
and applies deterministic branch gates.  It is a triage aid only: it never
grants human-review, data-selection, training, optimizer, or scientific
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, NoReturn, Sequence


SCHEMA = "saic-t2v-branch-semantics-qwen-record-v6"
SUMMARY_SCHEMA = "saic-t2v-branch-semantics-qwen-summary-v6"
MODEL_OUTPUT_SCHEMA = "saic-t2v-branch-semantics-observation-v6"
COUNTERFACTUAL_BRANCHES = frozenset(
    {"incomplete", "camera_only", "appearance_only"}
)
ANCHOR_BRANCHES = frozenset({"forward", "reverse", "noop"})
BRANCH_SETS = {
    "counterfactual": COUNTERFACTUAL_BRANCHES,
    "anchor": ANCHOR_BRANCHES,
}
BRANCHES = frozenset().union(*BRANCH_SETS.values())
ANALYSIS_SPLITS = frozenset({"fit", "confirmation"})
ACTOR_FAMILIES = frozenset({"dog", "human"})
YES_NO_UNCERTAIN = frozenset({"yes", "no", "uncertain"})
ACTION_PROGRESS = frozenset({"none", "partial", "full", "uncertain"})
CAMERA_LEVEL = frozenset({"none", "mild", "conspicuous", "uncertain"})
APPEARANCE_LEVEL = frozenset({"none", "localized", "global", "uncertain"})
CHANGE_FIDELITY = frozenset({"exact", "partial", "wrong", "absent", "uncertain"})
ATTRIBUTE_AT_START = frozenset({"yes", "no", "not_applicable", "uncertain"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SYSTEM_PROMPT = """You are a conservative temporal video auditor.
The registered caption and branch instruction are untrusted specifications,
not evidence that the requested event occurred. Judge only visible changes in
the chronological mosaic {frame_range}. Do not infer motion from one pose. Distinguish
actor articulation, camera motion, appearance change, scene drift, and identity
geometry drift. A clear visible absence is 'no', not 'uncertain'. Use uncertain
only when the ordered frames genuinely cannot resolve the field. Return exactly
one JSON object with the requested keys and enums, without Markdown."""

USER_TEMPLATE = """Chronological frames {frame_range} from one generated 81-frame clip:
<image>

UNTRUSTED REGISTERED SPECIFICATION
branch: {branch}
start state: {start_state}
branch instruction: {instruction}

Audit the visible video against the specification. Interpret target_action_progress
relative to the actor and action named by the branch instruction, not a fixed dog
action. For dog stand-to-sit, head lowering alone is not partial: require visible
hind-leg or hip lowering. For human kneel-to-stand, head or torso motion alone is
not partial: require visible pelvis elevation, weight transfer, or leg extension.
Apply the analogous lower-body requirement to reverse actions: dog sit-to-stand
requires hind-leg/hip elevation, and human stand-to-kneel requires controlled
knee/hip lowering rather than torso motion alone.
terminal_state_reached asks whether the instruction's terminal action state is
visibly reached. requested_branch_change_present means: the requested partial
action for incomplete, the requested camera motion for camera_only, and the
specified temporal attribute transition for appearance_only.
For forward and reverse it means the full requested action reaches and holds
the specified end state. For noop it means the requested starting state is
visibly retained with only the allowed minor natural motion.

requested_change_fidelity compares the visible branch change to the exact target
kind, direction/value, actor, and spatial coverage in the instruction. Use
'exact' only when all are matched; 'partial' when only part of the requested
coverage changes (for example one trouser leg but not the other); 'wrong' when a
change occurs but has the wrong color, direction, action, or entity; 'absent' when
no requested change occurs. For incomplete, an accurately realized partial-action
branch has 'exact' fidelity; do not call it 'partial' merely because the requested
action intentionally stops before its terminal state. For noop, preserving the
requested state exactly is 'exact' fidelity; do not call it 'absent' merely because
no action should occur. For camera_only, camera intensity such as 'conspicuous'
belongs only in camera_motion_level: requested_change_fidelity must still be one
of exact, partial, wrong, absent, or uncertain. For appearance_only,
requested_attribute_already_present_at_start is 'yes' if the target appearance is
already visible at F0, 'no' if not, and otherwise 'uncertain'. For non-appearance
branches it must be exactly 'not_applicable'; do not inspect an appearance target
that those branches did not request.

For target_action_progress, forward and reverse require 'full'; incomplete
requires 'partial'; noop, camera_only, and appearance_only require 'none'. For
terminal_state_reached, forward and reverse ask whether their requested terminal
action state is reached and held; every other branch requires 'no'.

Return exactly:
{{
  "schema_version": "saic-t2v-branch-semantics-observation-v6",
  "start_state_match": "yes|no|uncertain",
  "requested_branch_change_present": "yes|no|uncertain",
  "requested_change_fidelity": "exact|partial|wrong|absent|uncertain",
  "requested_attribute_already_present_at_start": "yes|no|not_applicable|uncertain",
  "target_action_progress": "none|partial|full|uncertain",
  "terminal_state_reached": "yes|no|uncertain",
  "temporal_order_coherent": "yes|no|uncertain",
  "identity_geometry_stable": "yes|no|uncertain",
  "protected_scene_stable": "yes|no|uncertain",
  "camera_motion_level": "none|mild|conspicuous|uncertain",
  "appearance_change_level": "none|localized|global|uncertain",
  "observed_evidence": ["1 to 4 short literal F-indexed observations"]
}}"""


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


def _closed(value: Any, keys: Iterable[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} keys differ")
    return value


def parse_one_json_object(raw: str) -> Mapping[str, Any]:
    stripped = raw.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    value = json.loads(stripped)
    return validate_model_output(value)


def validate_model_output(value: Any) -> Mapping[str, Any]:
    keys = {
        "schema_version", "start_state_match",
        "requested_branch_change_present", "requested_change_fidelity",
        "requested_attribute_already_present_at_start", "target_action_progress",
        "terminal_state_reached", "temporal_order_coherent",
        "identity_geometry_stable", "protected_scene_stable",
        "camera_motion_level", "appearance_change_level", "observed_evidence",
    }
    row = _closed(value, keys, label="model output")
    if row["schema_version"] != MODEL_OUTPUT_SCHEMA:
        raise ValueError("model output schema differs")
    for field in (
        "start_state_match", "requested_branch_change_present",
        "terminal_state_reached", "temporal_order_coherent",
        "identity_geometry_stable", "protected_scene_stable",
    ):
        if row[field] not in YES_NO_UNCERTAIN:
            raise ValueError(f"model output {field} enum differs")
    if row["target_action_progress"] not in ACTION_PROGRESS:
        raise ValueError("model output target_action_progress enum differs")
    if row["camera_motion_level"] not in CAMERA_LEVEL:
        raise ValueError("model output camera_motion_level enum differs")
    if row["appearance_change_level"] not in APPEARANCE_LEVEL:
        raise ValueError("model output appearance_change_level enum differs")
    if row["requested_change_fidelity"] not in CHANGE_FIDELITY:
        raise ValueError("model output requested_change_fidelity enum differs")
    if row["requested_attribute_already_present_at_start"] not in ATTRIBUTE_AT_START:
        raise ValueError(
            "model output requested_attribute_already_present_at_start enum differs"
        )
    evidence = row["observed_evidence"]
    if (
        not isinstance(evidence, list)
        or not 1 <= len(evidence) <= 4
        or any(not isinstance(item, str) or not item.strip() for item in evidence)
    ):
        raise ValueError("model output observed_evidence differs")
    return row


def deterministic_branch_gate(
    branch: str, observation: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    if branch not in BRANCHES:
        raise ValueError("branch differs")
    failures: list[str] = []
    required_yes = {
        "start_state_match": "start_state_mismatch",
        "requested_branch_change_present": "requested_branch_change_absent",
        "temporal_order_coherent": "temporal_order_incoherent",
        "identity_geometry_stable": "identity_geometry_drift",
        "protected_scene_stable": "protected_scene_drift",
    }
    for field, code in required_yes.items():
        if observation[field] != "yes":
            failures.append(
                "insufficient_visual_evidence"
                if observation[field] == "uncertain" else code
            )
    expected_terminal = "yes" if branch in {"forward", "reverse"} else "no"
    if observation["terminal_state_reached"] != expected_terminal:
        failures.append(
            "insufficient_visual_evidence"
            if observation["terminal_state_reached"] == "uncertain"
            else (
                "terminal_state_missing"
                if expected_terminal == "yes"
                else "terminal_state_leakage"
            )
        )
    progress = observation["target_action_progress"]
    camera = observation["camera_motion_level"]
    appearance = observation["appearance_change_level"]
    fidelity = observation["requested_change_fidelity"]
    if fidelity != "exact":
        fidelity_failures = {
            "partial": "requested_branch_change_spatially_incomplete",
            "wrong": "requested_branch_change_wrong",
            "absent": "requested_branch_change_absent",
            "uncertain": "insufficient_visual_evidence",
        }
        failures.append(fidelity_failures[fidelity])
    attribute_at_start = observation[
        "requested_attribute_already_present_at_start"
    ]
    if branch == "appearance_only":
        if attribute_at_start != "no":
            failures.append({
                "yes": "requested_attribute_present_at_start",
                "not_applicable": "appearance_start_field_misapplied",
                "uncertain": "insufficient_visual_evidence",
            }[attribute_at_start])
    if branch == "incomplete":
        if progress != "partial":
            failures.append(
                "insufficient_visual_evidence"
                if progress == "uncertain" else (
                    "target_action_absent" if progress == "none"
                    else "terminal_state_leakage"
                )
            )
        if camera != "none":
            failures.append(
                "insufficient_visual_evidence"
                if camera == "uncertain" else "unexpected_camera_motion"
            )
        if appearance != "none":
            failures.append(
                "insufficient_visual_evidence"
                if appearance == "uncertain" else "unexpected_appearance_change"
            )
    elif branch == "camera_only":
        if progress != "none":
            failures.append(
                "insufficient_visual_evidence"
                if progress == "uncertain" else "terminal_state_leakage"
            )
        if camera != "conspicuous":
            failures.append(
                "insufficient_visual_evidence"
                if camera == "uncertain" else "camera_change_missing"
            )
        if appearance != "none":
            failures.append(
                "insufficient_visual_evidence"
                if appearance == "uncertain" else "unexpected_appearance_change"
            )
    elif branch == "appearance_only":
        if progress != "none":
            failures.append(
                "insufficient_visual_evidence"
                if progress == "uncertain" else "terminal_state_leakage"
            )
        if camera != "none":
            failures.append(
                "insufficient_visual_evidence"
                if camera == "uncertain" else "unexpected_camera_motion"
            )
        if appearance not in {"localized", "global"}:
            failures.append(
                "insufficient_visual_evidence"
                if appearance == "uncertain" else "appearance_change_missing"
            )
    elif branch in {"forward", "reverse"}:
        if progress != "full":
            failures.append(
                "insufficient_visual_evidence"
                if progress == "uncertain" else (
                    "target_action_absent"
                    if progress == "none"
                    else "target_action_incomplete"
                )
            )
        if camera != "none":
            failures.append(
                "insufficient_visual_evidence"
                if camera == "uncertain" else "unexpected_camera_motion"
            )
        if appearance != "none":
            failures.append(
                "insufficient_visual_evidence"
                if appearance == "uncertain" else "unexpected_appearance_change"
            )
    else:
        if progress != "none":
            failures.append(
                "insufficient_visual_evidence"
                if progress == "uncertain" else "unexpected_target_action_progress"
            )
        if camera != "none":
            failures.append(
                "insufficient_visual_evidence"
                if camera == "uncertain" else "unexpected_camera_motion"
            )
        if appearance != "none":
            failures.append(
                "insufficient_visual_evidence"
                if appearance == "uncertain" else "unexpected_appearance_change"
            )
    return not failures, sorted(set(failures))


def chronological_mosaic(
    video_path: Path, *, nframes: int, tile_width: int, columns: int
) -> tuple[Any, str]:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < nframes:
        capture.release()
        raise RuntimeError(f"video has too few frames: {total}")
    indices = np.linspace(0, total - 1, nframes).round().astype(int).tolist()
    tiles = []
    try:
        for display_index, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"cannot read video frame {frame_index}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            height = max(1, round(image.height * tile_width / image.width))
            image = image.resize((tile_width, height), Image.Resampling.LANCZOS)
            label_height = 28
            tile = Image.new("RGB", (tile_width, height + label_height), "black")
            tile.paste(image, (0, label_height))
            ImageDraw.Draw(tile).text(
                (7, 5), f"F{display_index} / frame {frame_index}", fill="white"
            )
            tiles.append(tile)
    finally:
        capture.release()
    rows = (len(tiles) + columns - 1) // columns
    cell_height = max(tile.height for tile in tiles)
    mosaic = Image.new("RGB", (columns * tile_width, rows * cell_height), "black")
    for index, tile in enumerate(tiles):
        mosaic.paste(tile, ((index % columns) * tile_width, (index // columns) * cell_height))
    buffer = io.BytesIO()
    mosaic.save(buffer, format="PNG")
    return mosaic, hashlib.sha256(buffer.getvalue()).hexdigest()


class QwenAuditor:
    def __init__(self, model_path: Path, *, max_new_tokens: int) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, device_map="auto",
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )

    def generate(
        self, *, image: Any, system_prompt: str, user_prompt: str
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt.replace("<image>\n", "")},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def load_generation_receipts(
    root: Path, *, branch_set: str
) -> list[tuple[Path, Mapping[str, Any]]]:
    if branch_set not in BRANCH_SETS:
        die("branch set differs")
    receipt_name = (
        "saic-event-topup-generation-receipt.json"
        if branch_set == "counterfactual"
        else "saic-event-generation-receipt.json"
    )
    branches = BRANCH_SETS[branch_set]
    paths = sorted(root.glob(f"*/{receipt_name}"))
    rows = []
    for path in paths:
        value = json.loads(path.read_text(encoding="ascii"))
        candidate = value.get("candidate")
        if not isinstance(candidate, dict):
            die(f"candidate missing: {path}")
        candidate_id = candidate.get("candidate_id")
        branch = candidate.get("branch")
        analysis_split = candidate.get("analysis_split")
        actor_family = candidate.get("actor_family")
        if (
            not isinstance(candidate_id, str)
            or path.parent.name != candidate_id
            or branch not in branches
            or analysis_split not in ANALYSIS_SPLITS
            or actor_family not in ACTOR_FAMILIES
            or not isinstance(candidate.get("iid"), str)
            or not isinstance(candidate.get("action_family_id"), str)
            or candidate.get("event_verified") is not False
            or candidate.get("event_audit_status")
            != "pending_detached_full81_review"
        ):
            die(f"candidate receipt boundary differs: {path}")
        video = path.parent / "t2v.mp4"
        if not video.is_file() or video.is_symlink():
            die(f"candidate video differs: {video}")
        rows.append((path, value))
    if not rows:
        die("no generation receipts found")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--nframes", type=int, default=9)
    parser.add_argument("--tile-width", type=int, default=384)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--branch-set", choices=sorted(BRANCH_SETS), default="counterfactual"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.nframes < 5 or args.tile_width < 128 or args.columns < 1:
        die("mosaic geometry differs")
    attempts_root = args.attempts_root.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    selected_branches = BRANCH_SETS[args.branch_set]
    receipts = load_generation_receipts(
        attempts_root, branch_set=args.branch_set
    )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            die("max samples must be positive")
        receipts = receipts[:args.max_samples]
    for output in (args.output_jsonl, args.summary):
        if output.exists() or output.is_symlink():
            die(f"output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    auditor = QwenAuditor(model_path, max_new_tokens=args.max_new_tokens)
    frame_range = f"F0..F{args.nframes - 1}"
    system_prompt = SYSTEM_PROMPT.format(frame_range=frame_range)
    records: list[Mapping[str, Any]] = []
    with args.output_jsonl.open("x", encoding="ascii") as handle:
        for receipt_path, receipt in receipts:
            candidate = receipt["candidate"]
            video_path = receipt_path.parent / "t2v.mp4"
            image, visual_digest = chronological_mosaic(
                video_path, nframes=args.nframes,
                tile_width=args.tile_width, columns=args.columns,
            )
            prompt = USER_TEMPLATE.format(
                frame_range=frame_range,
                branch=candidate["branch"],
                start_state=candidate["branch_start_state_caption"],
                instruction=candidate["branch_instruction"],
            )
            raw = auditor.generate(
                image=image, system_prompt=system_prompt, user_prompt=prompt
            )
            parse_error = None
            observation = None
            try:
                observation = parse_one_json_object(raw)
            except (ValueError, json.JSONDecodeError) as error:
                parse_error = str(error)
            if observation is None:
                gate_passed = False
                failures = ["invalid_model_output"]
            else:
                gate_passed, failures = deterministic_branch_gate(
                    candidate["branch"], observation
                )
            unsigned = {
                "schema_version": SCHEMA,
                "branch_set": args.branch_set,
                "candidate_id": candidate["candidate_id"],
                "iid": candidate["iid"],
                "branch": candidate["branch"],
                "actor_family": candidate["actor_family"],
                "action_family_id": candidate["action_family_id"],
                "analysis_split": candidate["analysis_split"],
                "seed": candidate["seed"],
                "video_path": str(video_path),
                "video_sha256": file_sha256(video_path),
                "generation_receipt_path": str(receipt_path),
                "generation_receipt_sha256": file_sha256(receipt_path),
                "visual_input": {
                    "kind": "chronological_labeled_mosaic",
                    "nframes": args.nframes,
                    "tile_width": args.tile_width,
                    "columns": args.columns,
                    "sha256": visual_digest,
                },
                "registered_specification": {
                    "start_state": candidate["branch_start_state_caption"],
                    "branch_instruction": candidate["branch_instruction"],
                },
                "raw_response": raw,
                "validated_observation": observation,
                "parse_error": parse_error,
                "deterministic_branch_gate_passed": gate_passed,
                "deterministic_failure_codes": failures,
                "authority": {
                    "human_review": False,
                    "data_selection": False,
                    "training": False,
                    "optimizer": False,
                    "scientific_claim": False,
                },
            }
            record = {**unsigned, "receipt_digest": object_sha256(unsigned)}
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            records.append(record)
            print(json.dumps({
                "candidate_id": candidate["candidate_id"],
                "gate_passed": gate_passed, "failure_codes": failures,
                "parse_error": parse_error,
            }, sort_keys=True), flush=True)
    model_index = model_path / "model.safetensors.index.json"
    config_path = model_path / "config.json"
    unsigned_summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "diagnostic_vlm_triage_no_authority",
        "attempts_root": str(attempts_root),
        "branch_set": args.branch_set,
        "record_count": len(records),
        "valid_model_output_count": sum(
            row["validated_observation"] is not None for row in records
        ),
        "deterministic_gate_pass_count": sum(
            row["deterministic_branch_gate_passed"] is True for row in records
        ),
        "records_by_branch": {
            branch: sum(row["branch"] == branch for row in records)
            for branch in sorted(selected_branches)
        },
        "passes_by_branch": {
            branch: sum(
                row["branch"] == branch
                and row["deterministic_branch_gate_passed"] is True
                for row in records
            )
            for branch in sorted(selected_branches)
        },
        "records_by_analysis_split": {
            split: sum(row["analysis_split"] == split for row in records)
            for split in sorted(ANALYSIS_SPLITS)
        },
        "passes_by_analysis_split": {
            split: sum(
                row["analysis_split"] == split
                and row["deterministic_branch_gate_passed"] is True
                for row in records
            )
            for split in sorted(ANALYSIS_SPLITS)
        },
        "records_by_actor_family": {
            family: sum(row["actor_family"] == family for row in records)
            for family in sorted(ACTOR_FAMILIES)
        },
        "passes_by_actor_family": {
            family: sum(
                row["actor_family"] == family
                and row["deterministic_branch_gate_passed"] is True
                for row in records
            )
            for family in sorted(ACTOR_FAMILIES)
        },
        "failure_code_counts": {
            code: sum(
                code in row["deterministic_failure_codes"] for row in records
            )
            for code in sorted({
                code
                for row in records
                for code in row["deterministic_failure_codes"]
            })
        },
        "output_jsonl": str(args.output_jsonl),
        "output_jsonl_sha256": file_sha256(args.output_jsonl),
        "model": {
            "path": str(model_path),
            "config_sha256": file_sha256(config_path),
            "index_sha256": file_sha256(model_index),
            "transformers_version": auditor.transformers_version,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "authority": {
            "human_review": False,
            "data_selection": False,
            "training": False,
            "optimizer": False,
            "scientific_claim": False,
        },
    }
    summary = {
        **unsigned_summary, "receipt_digest": object_sha256(unsigned_summary)
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
