#!/usr/bin/env python3
"""Corrected, source-hidden action audit for the MEV anchor-gap calibration set.

This v2 evaluator deliberately does not show SOURCE or REAL TARGET alongside a
candidate.  Qwen receives one native video and a frozen, human-authored atomic
action contract.  SOURCE, REAL TARGET, reversed target, and temporally shuffled
target are evaluated as separately anonymized controls.  This prevents layout
and identity similarity from masquerading as successful action editing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from statistics import mean
from typing import Any, Mapping, NoReturn, Sequence

from .audit import MANIFEST_SCHEMA, MEV_PROTECTED_ROOT, assert_not_protected_write, file_sha256


CONTRACT_SCHEMA = "mev-action-anchor-human-contracts-v2"
CONTROL_SCHEMA = "mev-action-anchor-controls-v2"
OBSERVATION_SCHEMA = "mev-action-single-video-observation-v2"
TRACE_SCHEMA = "mev-neutral-video-trace-v2"
RECORD_SCHEMA = "mev-action-single-video-record-v2"
SUMMARY_SCHEMA = "mev-action-single-video-summary-v2"
INTERNVIDEO_SCHEMA = "mev-action-internvideo2-text-diagnostic-v2"
ROLES = (
    "anchor", "frozen_base", "target_forward", "source_noop",
    "target_reverse", "target_shuffle",
)
YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}


SYSTEM_PROMPT = """You are a conservative temporal action verifier. You see
exactly one anonymized video. Judge only visible temporal evidence in that
video. Do not infer an action from the scene, identity, objects, a single pose,
or the requested text. A required transition is false when only its endpoint
pose is visible. Report uncertainty instead of inventing contact, release,
pickup, rotation, direction, ordering, or completion. Return exactly one JSON
object and no Markdown."""


TRACE_SYSTEM_PROMPT = """You are a neutral video evidence recorder. You are
not told any requested action, edit instruction, source video, target video, or
candidate role. Record only literal changes visibly established across the
complete chronological video. Never infer an intended action from the scene or
an endpoint pose. Return exactly one JSON object and no Markdown."""


TRACE_USER_PROMPT = """Inspect the attached chronological video from beginning
to end and build an instruction-free evidence trace.

Track the main actor's facing orientation at multiple times (front, back,
left_profile, right_profile, other, or uncertain), literal torso/limb motion,
each manipulated object's held/surface/contact state, releases and pickups, and
the order of visible contacts or transfers. A return to the starting pose is
not evidence of a 360-degree rotation unless intermediate observations
establish a continuous path through front, one side, back, the opposite side,
and front. Rapid front/back pose replacement, identity morphing, or an
orientation jump without visible intermediate states is discontinuous and is
not a physical turn. Object presence is not evidence of pickup or placement. Mark
occluded or unresolved facts as ambiguous instead of guessing.

Populate dense_temporal_observations with exactly 12 chronological checkpoints
numbered 0 through 11 across the full clip. For every checkpoint, record the
visible body pose, actor orientation, hand/object state, and whether motion from
the previous checkpoint is physically continuous. Do not compress checkpoints
into only early/middle/late endpoints.

Return exactly:
{
  "schema_version": "mev-neutral-video-trace-v2",
  "visual_quality": "yes|no|uncertain",
  "dense_temporal_observations": [
    {"index": 0, "phase": "early|middle|late", "actor_orientation": "front|back|left_profile|right_profile|other|uncertain", "body_pose": "at most 12 words", "hands_and_objects": "at most 12 words", "continuity_from_previous": "start|continuous|discontinuous|uncertain", "observation": "at most 16 literal words"}
  ],
  "ambiguous_or_unseen": ["zero to four short facts that cannot be established"],
  "neutral_summary": "at most 30 literal words without guessing intent"
}"""


TRACE_JUDGE_TEMPLATE = """Use only the neutral trace below as evidence. You do
not have access to the video, source, real target, layout, or candidate role.
Do not add any transition absent from the trace. A predicate is yes only when
the trace explicitly establishes every part; no when the trace contradicts it;
otherwise uncertain. A checkpoint marked discontinuous cannot establish the
physical transition that crosses it. Completion cannot be yes unless all
required predicates are yes. The ordering of predicates below is arbitrary.

Neutral trace:
{trace}

Requested action: {target_action}

Required predicates:
{required}

Forbidden substitutes or failures:
{forbidden}

Return exactly this shape (retain every supplied id exactly once):
{{
  "schema_version": "mev-action-single-video-observation-v2",
  "action_observable": "yes|no|uncertain",
  "requested_action_complete": "yes|no|uncertain",
  "required_predicates": [
    {{"id": "supplied_id", "result": "yes|no|uncertain", "evidence": [{{"phase": "early|middle|late", "observation": "quote or closely paraphrase only the neutral trace"}}]}}
  ],
  "forbidden_behaviors": [
    {{"id": "supplied_id", "result": "yes|no|uncertain", "evidence": [{{"phase": "early|middle|late", "observation": "quote or closely paraphrase only the neutral trace"}}]}}
  ],
  "summary": "one short trace-grounded conclusion"
}}"""


USER_TEMPLATE = """Inspect the attached video from beginning to end.

Requested action: {target_action}

Required visible predicates, in the order used only for this audit pass:
{required}

Forbidden substitutes or failures:
{forbidden}

For each predicate, answer yes only when the transition is visibly established
across time; answer no when contradicted or absent; answer uncertain when the
video cannot establish it. Evidence must name early, middle, or late and state
a literal observation. requested_action_complete is yes only if the whole
requested action, including ordering and terminal state/full cycle, completes.

Return exactly this shape (retain every supplied id exactly once):
{{
  "schema_version": "mev-action-single-video-observation-v2",
  "action_observable": "yes|no|uncertain",
  "requested_action_complete": "yes|no|uncertain",
  "required_predicates": [
    {{"id": "supplied_id", "result": "yes|no|uncertain", "evidence": [{{"phase": "early|middle|late", "observation": "literal observation"}}]}}
  ],
  "forbidden_behaviors": [
    {{"id": "supplied_id", "result": "yes|no|uncertain", "evidence": [{{"phase": "early|middle|late", "observation": "literal observation"}}]}}
  ],
  "summary": "one short literal conclusion"
}}"""


def die(message: str) -> NoReturn:
    raise SystemExit(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    assert_not_protected_write(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    assert_not_protected_write(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_contracts(path: str | Path) -> dict[str, Mapping[str, Any]]:
    payload = load_json(path)
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("manual contract schema differs")
    rows = payload.get("samples")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("manual contracts must contain exactly 16 samples")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        prefix = row.get("pair_prefix")
        if not isinstance(prefix, str) or prefix in indexed:
            raise ValueError("manual contract prefix is missing or duplicated")
        if row.get("manual_winner") not in {"anchor", "tie"}:
            raise ValueError(f"unsupported manual winner for {prefix}")
        for field in ("required_predicates", "forbidden_behaviors"):
            items = row.get(field)
            if not isinstance(items, list) or not items:
                raise ValueError(f"{prefix} {field} must be nonempty")
            ids = [item.get("id") for item in items]
            if any(not isinstance(item_id, str) or not item_id for item_id in ids) or len(ids) != len(set(ids)):
                raise ValueError(f"{prefix} {field} ids differ")
        indexed[prefix] = row
    return indexed


def validate_inputs(manifest: Mapping[str, Any], contracts: Mapping[str, Mapping[str, Any]]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema differs")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 16:
        raise ValueError("manifest must contain exactly 16 samples")
    prefixes = {row.get("pair_prefix") for row in samples}
    if prefixes != set(contracts):
        raise ValueError("manifest and manual contract pair sets differ")


def _probe(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid video geometry: {path}")
    return {"frame_count": frames, "fps": fps, "duration": frames / fps, "width": width, "height": height}


def _run_ffmpeg(command: list[str], output: Path) -> None:
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output}: {process.stderr[-3000:]}")
    geometry = _probe(output)
    if geometry["frame_count"] != 81 or abs(geometry["fps"] - 25.0) > 1.0e-6:
        raise RuntimeError(f"control geometry differs for {output}: {geometry}")


def _normalize(ffmpeg: Path, source: Path, output: Path) -> dict[str, Any]:
    source_probe = _probe(source)
    pts_scale = 3.2 / source_probe["duration"]
    vf = (
        f"setpts={pts_scale:.12f}*PTS,fps=25,tpad=stop_mode=clone:stop_duration=1,"
        "scale=960:-2:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source), "-map", "0:v:0", "-an", "-vf", vf,
        "-frames:v", "81", "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], output)
    return {
        "path": str(output.resolve()), "sha256": file_sha256(output),
        "derived_from_path": str(source), "derived_from_sha256": file_sha256(source),
        "probe": _probe(output),
    }


def _temporal_transform(ffmpeg: Path, source: Path, output: Path, kind: str) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if kind == "reverse":
        filter_arg = "reverse,setpts=PTS-STARTPTS,fps=25"
    elif kind == "shuffle":
        filter_arg = (
            "[0:v]trim=start_frame=0:end_frame=27,setpts=PTS-STARTPTS[a];"
            "[0:v]trim=start_frame=27:end_frame=54,setpts=PTS-STARTPTS[b];"
            "[0:v]trim=start_frame=54:end_frame=81,setpts=PTS-STARTPTS[c];"
            "[b][a][c]concat=n=3:v=1:a=0,fps=25[out]"
        )
    else:
        raise ValueError(kind)
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source),
        "-an",
    ]
    if kind == "reverse":
        command += ["-vf", filter_arg, "-map", "0:v:0"]
    else:
        command += ["-filter_complex", filter_arg, "-map", "[out]"]
    command += [
        "-frames:v", "81", "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    _run_ffmpeg(command, output)
    return {
        "path": str(output.resolve()), "sha256": file_sha256(output),
        "derived_from_path": str(source.resolve()), "derived_from_sha256": file_sha256(source),
        "transform": kind, "probe": _probe(output),
    }


def build_controls(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve(strict=True)
    contract_path = Path(args.contracts).resolve(strict=True)
    manifest = load_json(manifest_path)
    contracts = load_contracts(contract_path)
    validate_inputs(manifest, contracts)
    ffmpeg = Path(args.ffmpeg).resolve(strict=True)
    if not os.access(ffmpeg, os.X_OK):
        raise ValueError("ffmpeg is not executable")
    output_root = Path(args.output_dir).resolve(strict=False)
    assert_not_protected_write(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"control output already exists: {output_root}")
    output_root.mkdir(parents=True)
    rows = []
    for sample in manifest["samples"]:
        prefix = sample["pair_prefix"]
        case_root = output_root / prefix
        target = _normalize(ffmpeg, Path(sample["real_target"]["path"]), case_root / "target-forward.mp4")
        source = _normalize(ffmpeg, Path(sample["source"]["path"]), case_root / "source-noop.mp4")
        reverse = _temporal_transform(ffmpeg, Path(target["path"]), case_root / "target-reverse.mp4", "reverse")
        shuffle = _temporal_transform(ffmpeg, Path(target["path"]), case_root / "target-shuffle.mp4", "shuffle")
        rows.append({
            "pair_id": sample["pair_id"], "pair_prefix": prefix,
            "roles": {
                "anchor": {"path": sample["generation"]["anchor"]["path"], "sha256": file_sha256(sample["generation"]["anchor"]["path"])},
                "frozen_base": {"path": sample["generation"]["frozen_base"]["path"], "sha256": file_sha256(sample["generation"]["frozen_base"]["path"])},
                "target_forward": target, "source_noop": source,
                "target_reverse": reverse, "target_shuffle": shuffle,
            },
        })
        print(json.dumps({"pair_prefix": prefix, "controls": "complete"}), flush=True)
    payload = {
        "schema_version": CONTROL_SCHEMA, "created_at": utc_now(),
        "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
        "manifest_digest": manifest["manifest_digest"],
        "contracts_path": str(contract_path), "contracts_sha256": file_sha256(contract_path),
        "protected_mev_tree_modified": False,
        "normalization": {"frames": 81, "fps": 25, "duration_seconds": 3.2},
        "samples": rows,
    }
    write_json(output_root / "control_manifest.json", payload)
    return 0


def _format_items(items: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(f"- {item['id']}: {item['description']}" for item in items)


def make_prompt(contract: Mapping[str, Any], reverse_order: bool) -> str:
    required = list(contract["required_predicates"])
    forbidden = list(contract["forbidden_behaviors"])
    if reverse_order:
        required.reverse()
        forbidden.reverse()
    return USER_TEMPLATE.format(
        target_action=contract["target_action"],
        required=_format_items(required), forbidden=_format_items(forbidden),
    )


def make_trace_judge_prompt(contract: Mapping[str, Any], trace: Mapping[str, Any], reverse_order: bool) -> str:
    required = list(contract["required_predicates"])
    forbidden = list(contract["forbidden_behaviors"])
    if reverse_order:
        required.reverse()
        forbidden.reverse()
    return TRACE_JUDGE_TEMPLATE.format(
        trace=json.dumps(trace, sort_keys=True, ensure_ascii=False),
        target_action=contract["target_action"],
        required=_format_items(required), forbidden=_format_items(forbidden),
    )


def _extract_json(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def _validate_evidence(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("evidence must be nonempty")
    for item in value:
        if not isinstance(item, dict) or set(item) != {"phase", "observation"}:
            raise ValueError("evidence shape differs")
        if item["phase"] not in {"early", "middle", "late"} or not isinstance(item["observation"], str) or not item["observation"].strip():
            raise ValueError("evidence value differs")


def validate_observation(value: Any, contract: Mapping[str, Any]) -> Mapping[str, Any]:
    expected_keys = {
        "schema_version", "action_observable", "requested_action_complete",
        "required_predicates", "forbidden_behaviors", "summary",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("observation keys differ")
    if value["schema_version"] != OBSERVATION_SCHEMA:
        raise ValueError("observation schema differs")
    for field in ("action_observable", "requested_action_complete"):
        if value[field] not in YES_NO_UNCERTAIN:
            raise ValueError(f"{field} enum differs")
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise ValueError("summary differs")
    for field in ("required_predicates", "forbidden_behaviors"):
        expected_ids = {item["id"] for item in contract[field]}
        rows = value[field]
        if not isinstance(rows, list) or {item.get("id") for item in rows if isinstance(item, dict)} != expected_ids or len(rows) != len(expected_ids):
            raise ValueError(f"{field} ids differ")
        for item in rows:
            if set(item) != {"id", "result", "evidence"} or item["result"] not in YES_NO_UNCERTAIN:
                raise ValueError(f"{field} item differs")
            _validate_evidence(item["evidence"])
    return value


def parse_observation(raw: str, contract: Mapping[str, Any]) -> Mapping[str, Any]:
    return validate_observation(_extract_json(raw), contract)


def validate_trace(value: Any) -> Mapping[str, Any]:
    keys = {
        "schema_version", "visual_quality", "dense_temporal_observations",
        "ambiguous_or_unseen", "neutral_summary",
    }
    if not isinstance(value, dict) or set(value) != keys or value["schema_version"] != TRACE_SCHEMA:
        raise ValueError("neutral trace shape differs")
    if value["visual_quality"] not in YES_NO_UNCERTAIN:
        raise ValueError("neutral trace visual quality differs")
    if not isinstance(value["neutral_summary"], str) or not value["neutral_summary"].strip():
        raise ValueError("neutral trace summary differs")
    if not isinstance(value["ambiguous_or_unseen"], list) or any(not isinstance(item, str) for item in value["ambiguous_or_unseen"]):
        raise ValueError("neutral trace ambiguity list differs")
    dense = value["dense_temporal_observations"]
    dense_keys = {"index", "phase", "actor_orientation", "body_pose", "hands_and_objects", "continuity_from_previous", "observation"}
    if not isinstance(dense, list) or len(dense) != 12 or [item.get("index") for item in dense if isinstance(item, dict)] != list(range(12)):
        raise ValueError("neutral trace must contain checkpoints 0..11")
    for item in dense:
        if set(item) != dense_keys or item["phase"] not in {"early", "middle", "late"}:
            raise ValueError("neutral trace dense checkpoint shape differs")
        if item["actor_orientation"] not in {"front", "back", "left_profile", "right_profile", "other", "uncertain"}:
            raise ValueError("neutral trace dense orientation differs")
        if item["continuity_from_previous"] not in {"start", "continuous", "discontinuous", "uncertain"}:
            raise ValueError("neutral trace continuity differs")
        if any(not isinstance(item[field], str) or not item[field].strip() for field in ("body_pose", "hands_and_objects", "observation")):
            raise ValueError("neutral trace dense text differs")
    return value


def parse_trace(raw: str) -> Mapping[str, Any]:
    return validate_trace(_extract_json(raw))


class NativeVideoQwenJudge:
    def __init__(self, model_path: Path, max_new_tokens: int, num_frames: int) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.max_new_tokens = max_new_tokens
        self.num_frames = num_frames
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, device_map="auto",
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)

    def generate(
        self, video_path: Path, prompt: str, *, system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int | None = None,
    ) -> str:
        # The AUH Qwen environment intentionally has no torchvision video
        # decoder. Decode a deterministic chronological tensor ourselves and
        # pass it through the processor's native video branch (T,C,H,W), not as
        # an image mosaic.
        video_frames = _read_frames(video_path, self.num_frames)
        geometry = _probe(video_path)
        sampled_fps = self.num_frames / geometry["duration"]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "video", "video": video_frames},
                {"type": "text", "text": prompt},
            ]},
        ]
        kwargs: dict[str, Any] = {
            "tokenize": True, "add_generation_prompt": True,
            "return_dict": True, "return_tensors": "pt",
            "processor_kwargs": {
                "do_sample_frames": False,
                "video_metadata": [{
                    "total_num_frames": self.num_frames,
                    "fps": sampled_fps,
                    "duration": geometry["duration"],
                    "frames_indices": list(range(self.num_frames)),
                    "height": int(video_frames.shape[2]),
                    "width": int(video_frames.shape[3]),
                }],
            },
        }
        inputs = self.processor.apply_chat_template(messages, **kwargs)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        converted = {}
        for key, value in inputs.items():
            if isinstance(value, self.torch.Tensor):
                value = value.to(device)
                if "pixel_values" in key:
                    value = value.to(dtype)
            converted[key] = value
        with self.torch.inference_mode():
            output = self.model.generate(
                **converted, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False,
            )
        input_length = converted["input_ids"].shape[1]
        return self.processor.batch_decode(
            output[:, input_length:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def generate_text(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [tokens[len(input_ids):] for input_ids, tokens in zip(inputs.input_ids, output)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _control_index(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if payload.get("schema_version") != CONTROL_SCHEMA:
        raise ValueError("control schema differs")
    return {row["pair_prefix"]: row for row in payload["samples"]}


def qwen_evaluate(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    contracts = load_contracts(args.contracts)
    validate_inputs(manifest, contracts)
    controls_payload = load_json(args.controls)
    controls = _control_index(controls_payload)
    if controls_payload.get("manifest_digest") != manifest.get("manifest_digest"):
        raise ValueError("controls are not bound to this manifest")
    if set(controls) != set(contracts):
        raise ValueError("controls and contracts pair sets differ")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index differs")
    samples = manifest["samples"]
    if args.pair_prefix:
        requested_prefixes = set(args.pair_prefix)
        samples = [row for row in samples if row["pair_prefix"] in requested_prefixes]
        if {row["pair_prefix"] for row in samples} != requested_prefixes:
            raise ValueError("one or more pair prefixes did not resolve exactly once")
    samples = [row for row in samples if row["ordinal"] % args.num_shards == args.shard_index]
    selected_roles = tuple(args.role) if args.role else ROLES
    allowed_prefixes = {row["pair_prefix"] for row in samples}
    records = _read_jsonl(args.output) if args.resume and Path(args.output).is_file() else []
    existing_keys = set()
    for row in records:
        key = (row["pair_prefix"], row["role"], row["pass_index"])
        if row["pair_prefix"] not in allowed_prefixes or row["role"] not in selected_roles:
            raise ValueError(f"resume record outside selected shard: {key}")
        if key in existing_keys:
            raise ValueError(f"duplicate resume record: {key}")
        existing_keys.add(key)
    judge = NativeVideoQwenJudge(Path(args.model).resolve(strict=True), args.max_new_tokens, args.num_frames)
    for sample in samples:
        prefix = sample["pair_prefix"]
        contract = contracts[prefix]
        for role in selected_roles:
            role_info = controls[prefix]["roles"][role]
            video_path = Path(role_info["path"]).resolve(strict=True)
            if file_sha256(video_path) != role_info["sha256"]:
                raise ValueError(f"video digest differs for {prefix}/{role}")
            pass_count = 2 if role in {"anchor", "frozen_base"} else 1
            completed = [row for row in records if row["pair_prefix"] == prefix and row["role"] == role]
            if len(completed) > pass_count:
                raise ValueError(f"too many resume records for {prefix}/{role}")
            if len(completed) == pass_count:
                if any(row["video_sha256"] != role_info["sha256"] for row in completed):
                    raise ValueError(f"resume video digest differs for {prefix}/{role}")
                print(json.dumps({"pair_prefix": prefix, "role": role, "resumed_passes": pass_count}), flush=True)
                continue
            if completed:
                reference = completed[0]
                trace_raw = reference["neutral_trace_raw_output"]
                neutral_trace = reference["neutral_trace"]
                trace_parse_error = reference["neutral_trace_parse_error"]
            else:
                trace_raw = judge.generate(
                    video_path, TRACE_USER_PROMPT, system_prompt=TRACE_SYSTEM_PROMPT,
                    max_new_tokens=args.trace_max_new_tokens,
                )
                try:
                    neutral_trace = parse_trace(trace_raw)
                    trace_parse_error = None
                except Exception as error:
                    neutral_trace = None
                    trace_parse_error = f"{type(error).__name__}: {error}"
            for pass_index in range(pass_count):
                key = (prefix, role, pass_index)
                if key in existing_keys:
                    continue
                if neutral_trace is None:
                    prompt, raw, observation = "", "", None
                    parse_error = f"neutral trace invalid: {trace_parse_error}"
                else:
                    prompt = make_trace_judge_prompt(contract, neutral_trace, reverse_order=bool(pass_index))
                    raw = judge.generate_text(prompt)
                    try:
                        observation = parse_observation(raw, contract)
                        parse_error = None
                    except Exception as error:
                        observation = None
                        parse_error = f"{type(error).__name__}: {error}"
                records.append({
                    "schema_version": RECORD_SCHEMA,
                    "pair_id": sample["pair_id"], "pair_prefix": prefix,
                    "role": role, "role_disclosed_to_model": False,
                    "source_or_reference_shown_with_candidate": False,
                    "target_instruction_disclosed_during_video_trace": False,
                    "neutral_trace_schema": TRACE_SCHEMA,
                    "neutral_trace_raw_output": trace_raw,
                    "neutral_trace_parse_error": trace_parse_error,
                    "neutral_trace": neutral_trace,
                    "pass_index": pass_index, "predicate_order": "reversed" if pass_index else "normal",
                    "video_path": str(video_path), "video_sha256": role_info["sha256"],
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "raw_output": raw, "parse_error": parse_error, "observation": observation,
                    "model": str(Path(args.model).resolve()),
                    "transformers_version": judge.transformers_version,
                    "native_video_frames_requested": args.num_frames,
                })
                existing_keys.add(key)
                # Persist after every pass so a preempted long-running shard
                # retains inspectable evidence instead of losing the shard.
                write_jsonl(args.output, records)
                print(json.dumps({"pair_prefix": prefix, "role": role, "pass": pass_index, "parsed": observation is not None}), flush=True)
    write_jsonl(args.output, records)
    return 0


def observation_components(observation: Mapping[str, Any]) -> list[int]:
    score = {"yes": 4, "uncertain": 2, "no": 0}
    inverse = {"yes": 0, "uncertain": 2, "no": 4}
    components = [
        score[observation["action_observable"]],
        score[observation["requested_action_complete"]],
    ]
    components.extend(score[item["result"]] for item in observation["required_predicates"])
    components.extend(inverse[item["result"]] for item in observation["forbidden_behaviors"])
    return components


def observation_gate(observation: Mapping[str, Any]) -> int:
    return min(observation_components(observation))


def observation_coverage(observation: Mapping[str, Any]) -> float:
    """Secondary relative evidence score; never changes the strict pass gate.

    The minimum gate deliberately collapses any hard failure to zero.  That is
    correct for deciding whether a candidate fully passes the action contract,
    but it cannot distinguish two failed candidates when one visibly completes
    more of the requested atomic action.  Coverage is therefore used only as a
    lexicographic tie-break after equal gates, never to compensate for a lower
    gate.
    """
    return mean(observation_components(observation))


def _winner(anchor: tuple[int, float], base: tuple[int, float]) -> str:
    if anchor > base:
        return "anchor"
    if base > anchor:
        return "frozen_base"
    return "tie"


def qwen_summarize(args: argparse.Namespace) -> int:
    contracts_payload = load_json(args.contracts)
    contracts = load_contracts(args.contracts)
    records = []
    for path in sorted(Path(args.records_dir).glob("qwen-v2-shard-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        prefix: {role: [] for role in ROLES} for prefix in contracts
    }
    for row in records:
        if row.get("schema_version") != RECORD_SCHEMA:
            raise ValueError("record schema differs")
        grouped[row["pair_prefix"]][row["role"]].append(row)
    pairs = []
    for prefix, contract in contracts.items():
        role_rows = grouped[prefix]
        expected = {role: (2 if role in {"anchor", "frozen_base"} else 1) for role in ROLES}
        if any(len(role_rows[role]) != count for role, count in expected.items()):
            raise ValueError(f"incomplete Qwen records for {prefix}")
        scores: dict[str, list[int | None]] = {}
        coverage: dict[str, list[float | None]] = {}
        for role, rows in role_rows.items():
            rows.sort(key=lambda row: row["pass_index"])
            scores[role] = [observation_gate(row["observation"]) if row["observation"] is not None else None for row in rows]
            coverage[role] = [observation_coverage(row["observation"]) if row["observation"] is not None else None for row in rows]
        pass_winners = []
        for pass_index in range(2):
            a_score = scores["anchor"][pass_index]
            b_score = scores["frozen_base"][pass_index]
            a_coverage = coverage["anchor"][pass_index]
            b_coverage = coverage["frozen_base"][pass_index]
            pass_winners.append(
                "abstain"
                if a_score is None or b_score is None or a_coverage is None or b_coverage is None
                else _winner((a_score, a_coverage), (b_score, b_coverage))
            )
        winner = pass_winners[0] if pass_winners[0] == pass_winners[1] else "abstain"
        manual = contract["manual_winner"]
        pairs.append({
            "pair_prefix": prefix, "manual_winner": manual,
            "qwen_winner": winner, "agrees_with_manual": winner == manual,
            "pass_winners": pass_winners, "gate_scores": scores,
            "coverage_scores": coverage,
            "roles": role_rows, "human_note": contract["human_note"],
            "target_action": contract["target_action"],
        })
    counts = Counter(row["qwen_winner"] for row in pairs)
    agreement = sum(row["agrees_with_manual"] for row in pairs)
    control = {
        "target_forward_gate_mean": mean(row["gate_scores"]["target_forward"][0] for row in pairs),
        "source_noop_gate_mean": mean(row["gate_scores"]["source_noop"][0] for row in pairs),
        "target_reverse_gate_mean": mean(row["gate_scores"]["target_reverse"][0] for row in pairs),
        "target_shuffle_gate_mean": mean(row["gate_scores"]["target_shuffle"][0] for row in pairs),
        "target_forward_strict_pass_count": sum(row["gate_scores"]["target_forward"][0] == 4 for row in pairs),
        "source_noop_strict_pass_count": sum(row["gate_scores"]["source_noop"][0] == 4 for row in pairs),
        "reverse_below_forward_count": sum(row["gate_scores"]["target_reverse"][0] < row["gate_scores"]["target_forward"][0] for row in pairs),
        "shuffle_below_forward_count": sum(row["gate_scores"]["target_shuffle"][0] < row["gate_scores"]["target_forward"][0] for row in pairs),
    }
    payload = {
        "schema_version": SUMMARY_SCHEMA, "created_at": utc_now(),
        "contracts_sha256": file_sha256(args.contracts),
        "label_source": contracts_payload["label_source"],
        "evaluation_role": contracts_payload["evaluation_role"],
        "decision_rule": "lexicographic (noncompensatory minimum gate, then atomic evidence coverage only when gates tie); same winner under predicate-order reversal",
        "candidate_context": "instruction-free native-video trace, then trace-only atomic contract judgment; SOURCE and REAL TARGET hidden",
        "pair_count": len(pairs), "winner_counts": dict(counts),
        "manual_agreement_count": agreement,
        "manual_agreement_rate": agreement / len(pairs),
        "control_calibration": control, "pairs": pairs,
    }
    write_json(args.output, payload)
    return 0


def _read_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if value.get("schema_version") != RECORD_SCHEMA:
                raise ValueError("record schema differs")
            rows.append(value)
    return rows


def merge_qwen_repair(args: argparse.Namespace) -> int:
    """Atomically replace complete per-pair records in a finished shard."""
    base_rows = _read_jsonl(args.base)
    repair_rows = _read_jsonl(args.repair)
    if not base_rows or not repair_rows:
        raise ValueError("base and repair records must both be nonempty")
    repair_prefixes = {row["pair_prefix"] for row in repair_rows}
    expected = {role: (2 if role in {"anchor", "frozen_base"} else 1) for role in ROLES}
    for prefix in repair_prefixes:
        counts = Counter(row["role"] for row in repair_rows if row["pair_prefix"] == prefix)
        if counts != Counter(expected):
            raise ValueError(f"incomplete repair records for {prefix}: {dict(counts)}")
        if not any(row["pair_prefix"] == prefix for row in base_rows):
            raise ValueError(f"repair pair absent from base shard: {prefix}")
    merged = [row for row in base_rows if row["pair_prefix"] not in repair_prefixes] + repair_rows
    keys = [(row["pair_prefix"], row["role"], row["pass_index"]) for row in merged]
    if len(keys) != len(set(keys)):
        raise ValueError("merged records contain duplicate pair/role/pass keys")
    if len(merged) != len(base_rows):
        raise ValueError("repair changed shard record count")
    output = Path(args.output)
    assert_not_protected_write(output)
    temporary = output.with_name(output.name + ".merge-tmp")
    write_jsonl(temporary, merged)
    temporary.replace(output)
    print(json.dumps({"repaired_pairs": sorted(repair_prefixes), "records": len(merged)}), flush=True)
    return 0


def qwen_rescore_contract(args: argparse.Namespace) -> int:
    """Re-run only trace-to-contract judgments on frozen neutral traces."""
    contracts = load_contracts(args.contracts)
    requested_prefixes = set(args.pair_prefix)
    if not requested_prefixes or not requested_prefixes <= set(contracts):
        raise ValueError("rescore pair prefixes differ from contracts")
    source_rows = []
    for path in args.records:
        source_rows.extend(_read_jsonl(path))
    output_dir = Path(args.output_dir)
    assert_not_protected_write(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judge = NativeVideoQwenJudge(Path(args.model).resolve(strict=True), args.max_new_tokens, 32)
    expected = {role: (2 if role in {"anchor", "frozen_base"} else 1) for role in ROLES}
    for prefix in sorted(requested_prefixes):
        rows = [row for row in source_rows if row["pair_prefix"] == prefix]
        counts = Counter(row["role"] for row in rows)
        if counts != Counter(expected):
            raise ValueError(f"incomplete source records for rescore {prefix}: {dict(counts)}")
        keys = [(row["role"], row["pass_index"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate source records for rescore {prefix}")
        destination = output_dir / f"{prefix}.jsonl"
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"rescore output already exists: {destination}")
        rescored = []
        order = {role: index for index, role in enumerate(ROLES)}
        rows.sort(key=lambda row: (order[row["role"]], row["pass_index"]))
        for row in rows:
            trace = validate_trace(row["neutral_trace"])
            prompt = make_trace_judge_prompt(
                contracts[prefix], trace, reverse_order=bool(row["pass_index"]),
            )
            raw = judge.generate_text(prompt)
            try:
                observation = parse_observation(raw, contracts[prefix])
                parse_error = None
            except Exception as error:
                observation = None
                parse_error = f"{type(error).__name__}: {error}"
            updated = dict(row)
            updated["contract_rescore"] = {
                "rescored_at": utc_now(),
                "previous_prompt_sha256": row["prompt_sha256"],
                "previous_raw_output_sha256": hashlib.sha256(row["raw_output"].encode("utf-8")).hexdigest(),
                "previous_parse_error": row["parse_error"],
                "previous_observation": row["observation"],
                "neutral_trace_reused_without_video_inference": True,
            }
            updated.update({
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "raw_output": raw, "parse_error": parse_error, "observation": observation,
            })
            rescored.append(updated)
            write_jsonl(destination, rescored)
            print(json.dumps({
                "pair_prefix": prefix, "role": row["role"], "pass": row["pass_index"],
                "rescored": True, "parsed": observation is not None,
            }), flush=True)
    return 0


def _middle_indices(total: int, count: int) -> list[int]:
    import numpy as np
    intervals = np.linspace(0, total, count + 1).astype(int)
    return [(int(left) + int(right) - 1) // 2 for left, right in zip(intervals[:-1], intervals[1:])]


def _read_frames(path: Path, count: int) -> Any:
    import cv2
    import numpy as np
    import torch

    capture = cv2.VideoCapture(str(path))
    total = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if total <= 0:
        capture.release()
        raise ValueError(f"no frames: {path}")
    frames = []
    for index in _middle_indices(total, count):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"cannot decode frame {index}: {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    array = np.stack(frames, axis=0)
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def _cosine(a: Any, b: Any) -> float:
    import torch.nn.functional as functional
    return float(functional.cosine_similarity(a.float(), b.float(), dim=-1).item())


def _load_pinned_internvideo_classes(model_root: Path) -> tuple[type, type]:
    """Load the pinned local HF implementation without the dynamic-module cache.

    Transformers' offline dynamic-module copier drops second-level relative
    imports for this repository (notably ``pos_embed.py``).  A private package
    namespace keeps every relative import rooted in the pinned, hashed vendor
    directory and avoids mutating either the model tree or site-packages.
    """
    import importlib.util
    import sys
    import types

    package_name = "action_gap_pinned_internvideo2_clip_s"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_root)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    def load_module(name: str, filename: str) -> Any:
        qualified = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, model_root / filename)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load pinned InternVideo2 module {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        return module

    config_module = load_module("config", "config.py")
    model_module = load_module("modeling_internvideo2encoder", "modeling_internvideo2encoder.py")
    return config_module.InternVideo2Config, model_module.InternVideo2_CLIP_small


def internvideo_evaluate(args: argparse.Namespace) -> int:
    if args.extra_site_packages:
        import sys
        sys.path.insert(0, str(Path(args.extra_site_packages).resolve(strict=True)))
    import torch

    manifest = load_json(args.manifest)
    contracts = load_contracts(args.contracts)
    validate_inputs(manifest, contracts)
    controls_payload = load_json(args.controls)
    controls = _control_index(controls_payload)
    model_root = Path(args.model).resolve(strict=True)
    config_class, model_class = _load_pinned_internvideo_classes(model_root)
    config = config_class.from_pretrained(str(model_root), local_files_only=True)
    model = model_class.from_pretrained(
        str(model_root), config=config, local_files_only=True,
    ).eval().to(args.device)
    revision_files = []
    for path in sorted(model_root.iterdir()):
        if path.is_file():
            revision_files.append({"name": path.name, "sha256": file_sha256(path)})
    selected_samples = manifest["samples"]
    if args.pair_prefix:
        selected_samples = [row for row in selected_samples if row["pair_prefix"] == args.pair_prefix]
        if len(selected_samples) != 1:
            raise ValueError("pair prefix did not resolve exactly once")
    rows = []
    with torch.inference_mode():
        for sample in selected_samples:
            prefix = sample["pair_prefix"]
            target_text = contracts[prefix]["target_action"]
            source_text = sample["source_action_caption"]
            text_input = model.tokenizer([target_text, source_text]).to(model.device)
            text_features = model.encode_text(text_input)
            role_scores = {}
            for role in ROLES:
                role_info = controls[prefix]["roles"][role]
                video_path = Path(role_info["path"]).resolve(strict=True)
                frames = _read_frames(video_path, args.num_frames)
                video_input = model.transform(frames).unsqueeze(0).to(model.device)
                video_feature = model.encode_vision(video_input, test=True)
                target_score = _cosine(video_feature, text_features[0:1])
                source_score = _cosine(video_feature, text_features[1:2])
                role_scores[role] = {
                    "target_action_text_cosine": target_score,
                    "source_action_text_cosine": source_score,
                    "target_minus_source_action_margin": target_score - source_score,
                    "video_sha256": role_info["sha256"],
                }
            rows.append({
                "pair_prefix": prefix, "manual_winner": contracts[prefix]["manual_winner"],
                "target_action_text": target_text, "source_action_text": source_text,
                "role_scores": role_scores,
            })
            print(json.dumps({"pair_prefix": prefix, "internvideo": "complete"}), flush=True)
    forward_over_reverse = sum(
        row["role_scores"]["target_forward"]["target_minus_source_action_margin"]
        > row["role_scores"]["target_reverse"]["target_minus_source_action_margin"] + args.control_epsilon
        for row in rows
    )
    forward_over_shuffle = sum(
        row["role_scores"]["target_forward"]["target_minus_source_action_margin"]
        > row["role_scores"]["target_shuffle"]["target_minus_source_action_margin"] + args.control_epsilon
        for row in rows
    )
    forward_over_source = sum(
        row["role_scores"]["target_forward"]["target_minus_source_action_margin"]
        > row["role_scores"]["source_noop"]["target_minus_source_action_margin"] + args.control_epsilon
        for row in rows
    )
    required_passes = min(args.min_control_passes, len(rows))
    admission = (
        forward_over_reverse >= required_passes
        and forward_over_shuffle >= required_passes
        and forward_over_source >= required_passes
    )
    payload = {
        "schema_version": INTERNVIDEO_SCHEMA, "created_at": utc_now(),
        "official_model": "OpenGVLab/InternVideo2_CLIP_S",
        "pinned_revision": "1f9fca1389fd883defc652634d95a21121c85a8c",
        "model_root": str(model_root), "model_files": revision_files,
        "num_frames": args.num_frames,
        "metric": "cos(video,target-action-text)-cos(video,source-action-text)",
        "candidate_target_video_cosine_used": False,
        "admission_rule": f"forward exceeds reverse, shuffle, and source by >{args.control_epsilon} on >= {required_passes}/{len(rows)} each",
        "calibration": {
            "forward_over_reverse_count": forward_over_reverse,
            "forward_over_shuffle_count": forward_over_shuffle,
            "forward_over_source_count": forward_over_source,
            "admitted_for_candidate_ranking": admission,
        },
        "interpretation": "auxiliary candidate ranking" if admission else "rejected as action-ranking evidence; diagnostic only",
        "pairs": rows,
    }
    write_json(args.output, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    controls = sub.add_parser("build-controls")
    controls.add_argument("--manifest", required=True)
    controls.add_argument("--contracts", required=True)
    controls.add_argument("--ffmpeg", required=True)
    controls.add_argument("--output-dir", required=True)
    controls.set_defaults(function=build_controls)

    qwen = sub.add_parser("qwen-evaluate")
    qwen.add_argument("--manifest", required=True)
    qwen.add_argument("--contracts", required=True)
    qwen.add_argument("--controls", required=True)
    qwen.add_argument("--model", required=True)
    qwen.add_argument("--shard-index", type=int, required=True)
    qwen.add_argument("--num-shards", type=int, default=2)
    qwen.add_argument("--pair-prefix", action="append")
    qwen.add_argument("--role", action="append", choices=ROLES)
    qwen.add_argument("--num-frames", type=int, default=32)
    qwen.add_argument("--max-new-tokens", type=int, default=1536)
    qwen.add_argument("--trace-max-new-tokens", type=int, default=2048)
    qwen.add_argument("--resume", action="store_true")
    qwen.add_argument("--output", required=True)
    qwen.set_defaults(function=qwen_evaluate)

    summary = sub.add_parser("qwen-summarize")
    summary.add_argument("--contracts", required=True)
    summary.add_argument("--records-dir", required=True)
    summary.add_argument("--output", required=True)
    summary.set_defaults(function=qwen_summarize)

    merge = sub.add_parser("merge-qwen-repair")
    merge.add_argument("--base", required=True)
    merge.add_argument("--repair", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(function=merge_qwen_repair)

    rescore = sub.add_parser("qwen-rescore-contract")
    rescore.add_argument("--contracts", required=True)
    rescore.add_argument("--records", action="append", required=True)
    rescore.add_argument("--pair-prefix", action="append", required=True)
    rescore.add_argument("--model", required=True)
    rescore.add_argument("--max-new-tokens", type=int, default=1536)
    rescore.add_argument("--output-dir", required=True)
    rescore.set_defaults(function=qwen_rescore_contract)

    intern = sub.add_parser("internvideo-evaluate")
    intern.add_argument("--manifest", required=True)
    intern.add_argument("--contracts", required=True)
    intern.add_argument("--controls", required=True)
    intern.add_argument("--model", required=True)
    intern.add_argument("--device", default="cuda")
    intern.add_argument("--extra-site-packages")
    intern.add_argument("--pair-prefix")
    intern.add_argument("--num-frames", type=int, default=8)
    intern.add_argument("--control-epsilon", type=float, default=0.005)
    intern.add_argument("--min-control-passes", type=int, default=12)
    intern.add_argument("--output", required=True)
    intern.set_defaults(function=internvideo_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
