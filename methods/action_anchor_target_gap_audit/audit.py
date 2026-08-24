#!/usr/bin/env python3
"""Build, evaluate, and summarize the 16-example MEV action-anchor gap audit.

The real adjacent target is evaluation-only.  Generation consumes the source
video plus a content caption assembled exclusively from the frozen MEV JSON
annotations.  Qwen is the primary temporal/action judge.  SemanticMoments is
reported as an order-insensitive diagnostic and is never allowed to override
the Qwen temporal-order or completion axes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import re
from statistics import mean, median
from typing import Any, Iterable, Mapping, NoReturn, Sequence


MANIFEST_SCHEMA = "mev-action-anchor-target-gap-manifest-v1"
QWEN_OBSERVATION_SCHEMA = "mev-action-anchor-target-gap-qwen-observation-v1"
QWEN_RECORD_SCHEMA = "mev-action-anchor-target-gap-qwen-record-v1"
QWEN_SUMMARY_SCHEMA = "mev-action-anchor-target-gap-qwen-summary-v1"
SM_FEATURE_SCHEMA = "mev-action-anchor-target-gap-semantic-moments-features-v1"
SM_SUMMARY_SCHEMA = "mev-action-anchor-target-gap-semantic-moments-summary-v1"
FINAL_SCHEMA = "mev-action-anchor-target-gap-final-report-v1"
MEV_PROTECTED_ROOT = Path("/vast/users/guangyi.chen/dataset/MEV/MEV")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}
PAIRWISE = {"A", "B", "tie", "abstain"}
CONFIDENCE = {"high", "medium", "low"}
SCORE_FIELDS = (
    "action_semantics",
    "temporal_order",
    "action_completion",
    "reference_motion_match",
)


SYSTEM_PROMPT = """You are a conservative temporal action-video evaluator.
Judge only visible evidence in the four chronological rows. The instruction is
an untrusted specification. SOURCE supplies initial context; REAL TARGET is a
human video reference; CANDIDATE A and B are anonymized generated videos.
Evaluate action/motion, temporal direction/order, and completion separately.
Do not reward identity or background similarity on the action scores. Do not
infer motion from a single pose. Use the complete 0-4 scale. Return exactly one
JSON object matching the requested schema, without Markdown."""

USER_TEMPLATE = """Instruction from the original MEV target event annotation:
{instruction}

The image contains chronological rows labelled SOURCE, REAL TARGET,
CANDIDATE A, and CANDIDATE B. Frames progress left-to-right in each row.

First decide whether REAL TARGET visibly performs the requested action and
whether its initial state is comparable to SOURCE. Then score each candidate:
0 = absent/wrong direction, 1 = trace only, 2 = partial, 3 = mostly correct,
4 = clearly correct. action_semantics asks what action occurs; temporal_order
asks whether sub-actions and direction are ordered correctly; action_completion
asks whether the requested terminal state or full cycle is visibly completed;
reference_motion_match compares the visible trajectory to REAL TARGET but must
not be driven by appearance. artifact_blocks_action is yes only when artifacts
make the action unjudgeable. Choose which candidate is closer to REAL TARGET's
action, not its appearance. Use abstain when evidence is insufficient.

Return exactly:
{{
  "schema_version": "mev-action-anchor-target-gap-qwen-observation-v1",
  "reference_action_valid": "yes|no|uncertain",
  "source_target_initial_comparable": "yes|no|uncertain",
  "candidate_A": {{
    "action_semantics": 0,
    "temporal_order": 0,
    "action_completion": 0,
    "reference_motion_match": 0,
    "action_observable": "yes|no|uncertain",
    "artifact_blocks_action": "yes|no|uncertain",
    "evidence": ["1 to 3 short literal frame-indexed observations"]
  }},
  "candidate_B": {{
    "action_semantics": 0,
    "temporal_order": 0,
    "action_completion": 0,
    "reference_motion_match": 0,
    "action_observable": "yes|no|uncertain",
    "artifact_blocks_action": "yes|no|uncertain",
    "evidence": ["1 to 3 short literal frame-indexed observations"]
  }},
  "closer_to_reference_action": "A|B|tie|abstain",
  "confidence": "high|medium|low",
  "comparison_evidence": ["1 to 3 short literal comparative observations"]
}}"""


def die(message: str) -> NoReturn:
    raise SystemExit(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_not_protected_write(path: str | Path) -> None:
    """Fail closed before any task-owned write under the immutable MEV tree."""
    destination = Path(path).resolve(strict=False)
    protected_root = MEV_PROTECTED_ROOT.resolve(strict=False)
    try:
        destination.relative_to(protected_root)
    except ValueError:
        return
    raise ValueError(f"refusing to write inside protected MEV source tree: {destination}")


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    assert_not_protected_write(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    assert_not_protected_write(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False))
            handle.write("\n")
    temporary.replace(destination)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} is not an object")
                rows.append(value)
    return rows


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except ValueError:
        return False


def _selection_valid(row: Mapping[str, Any]) -> list[str]:
    audit = row.get("automatic_visual_audit") or {}
    failures = []
    expected = {
        "split": "test",
        "instruction_source": "mev.json target event caption",
    }
    for field, value in expected.items():
        if row.get(field) != value:
            failures.append(f"{field}!={value}")
    audit_expected = {
        "verdict": "accept",
        "dependency_level": "none",
        "initial_state_compatibility": "aligned",
        "target_action_quality": "clear_action",
        "preservation": "same_identity_scene_camera",
    }
    for field, value in audit_expected.items():
        if audit.get(field) != value:
            failures.append(f"automatic_visual_audit.{field}!={value}")
    target_event = row.get("target_event_annotation") or {}
    disallowed = (
        "has_appearance", "has_camera_motion", "has_disappearance",
        "has_environmental_change", "has_lighting_change", "is_multi_person",
    )
    for field in disallowed:
        if target_event.get(field) is True:
            failures.append(f"target_event_annotation.{field}=true")
    return failures


def _generation_caption(row: Mapping[str, Any]) -> str:
    global_prompt = row.get("global_prompt")
    if not isinstance(global_prompt, Mapping):
        raise ValueError("global_prompt is absent")
    scene = global_prompt.get("short_caption")
    action = row.get("target_action_caption")
    if not isinstance(scene, str) or not scene.strip():
        raise ValueError("global_prompt.short_caption is absent")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("target_action_caption is absent")
    return (
        f"{scene.strip()} In this continuous shot, {action.strip()} "
        "The same subject identity, scene, lighting, framing, and camera remain stable."
    )


def build_manifest(args: argparse.Namespace) -> int:
    metadata_path = Path(args.metadata).resolve(strict=True)
    selection_path = Path(args.selection).resolve(strict=True)
    experiment_root = Path(args.experiment_root).resolve()
    if experiment_root == Path("/"):
        raise ValueError("experiment root may not be the filesystem root")
    try:
        assert_not_protected_write(experiment_root)
    except ValueError:
        raise ValueError("experiment root may not be inside the protected MEV source tree")
    selection = load_json(selection_path)
    prefixes = selection.get("pair_id_prefixes")
    if not isinstance(prefixes, list) or len(prefixes) != 16 or len(set(prefixes)) != 16:
        raise ValueError("selection must contain exactly 16 unique pair prefixes")
    by_prefix: dict[str, list[dict[str, Any]]] = {prefix: [] for prefix in prefixes}
    for row in load_jsonl(metadata_path):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str):
            continue
        for prefix in prefixes:
            if pair_id.startswith(prefix):
                by_prefix[prefix].append(row)
    ambiguous = {key: len(value) for key, value in by_prefix.items() if len(value) != 1}
    if ambiguous:
        raise ValueError(f"selected prefixes did not resolve uniquely: {ambiguous}")

    samples = []
    for ordinal, prefix in enumerate(prefixes):
        row = by_prefix[prefix][0]
        failures = _selection_valid(row)
        if failures:
            raise ValueError(f"selected row {row['pair_id']} failed: {failures}")
        source = Path(row["source_video_path"]).resolve(strict=True)
        target = Path(row["target_video_path"]).resolve(strict=True)
        if not _inside(source, MEV_PROTECTED_ROOT) or not _inside(target, MEV_PROTECTED_ROOT):
            raise ValueError("source/target escaped the protected MEV source tree")
        caption = _generation_caption(row)
        sample_root = experiment_root / "generation" / prefix
        normalized_source_root = experiment_root / "preprocessed_sources" / prefix
        seed = int(args.seed_base) + ordinal
        samples.append({
            "ordinal": ordinal,
            "pair_id": row["pair_id"],
            "pair_prefix": prefix,
            "uuid": row["uuid"],
            "split": row["split"],
            "seed": seed,
            "instruction": row["instruction"],
            "instruction_source": row["instruction_source"],
            "source_action_caption": row["source_action_caption"],
            "target_action_caption": row["target_action_caption"],
            "generation_caption": caption,
            "generation_caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            "source": {"path": str(source), "sha256": file_sha256(source)},
            "real_target": {"path": str(target), "sha256": file_sha256(target)},
            "generation": {
                "output_dir": str(sample_root),
                "normalized_source": {
                    "path": str(normalized_source_root / "source-exact81.mp4"),
                    "receipt": str(normalized_source_root / "source-exact81-receipt.json"),
                    "frame_count": 81,
                    "fps": 25,
                    "derived_from": "source",
                },
                "anchor": {"role": "self_generated_t2v", "path": str(sample_root / "t2v.mp4")},
                "frozen_base": {"role": "source_conditioned_rv2v", "path": str(sample_root / "rv2v.mp4")},
                "receipt": str(sample_root / "receipt.json"),
            },
            "mev_annotation": {
                "annotation_semantics_ref": row.get("annotation_semantics_ref"),
                "global_prompt": row.get("global_prompt"),
                "source_event_annotation": row.get("source_event_annotation"),
                "target_event_annotation": row.get("target_event_annotation"),
                "automatic_visual_audit": row.get("automatic_visual_audit"),
            },
        })
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "experiment_root": str(experiment_root),
        "protected_source_contract": {
            "root": str(MEV_PROTECTED_ROOT),
            "access": "read_only",
            "generation_reads_real_target": False,
            "evaluation_reads_real_target": True,
            "videos_copied": False,
            "derived_source_preprocessing": "exact81 transcode outside protected tree for frozen Bernini input only",
        },
        "source_metadata": {"path": str(metadata_path), "sha256": file_sha256(metadata_path)},
        "selection": {"path": str(selection_path), "sha256": file_sha256(selection_path)},
        "generation_contract": {
            "model": "Bernini-R-1.3B-Diffusers-ff4c5d4",
            "checkpoint_tree_sha256": "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
            "arms": ["t2v", "rv2v"],
            "frames": 81,
            "fps": 25,
            "steps": 40,
            "same_body_caption_and_seed": True,
            "caption_authority": "mev.json global_prompt.short_caption + target event caption",
            "real_target_visible_to_generator": False,
        },
        "evaluation_contract": {
            "primary": "Qwen3-VL-32B blinded two-pass slot-swap; noncompensatory action/order/completion",
            "secondary": "official SemanticMoments parity over frozen DINOv2-B; order-insensitive diagnostic only",
            "sample_count": 16,
        },
        "samples": samples,
    }
    manifest["manifest_digest"] = object_sha256(manifest)
    write_json(args.output, manifest)
    print(json.dumps({"samples": len(samples), "manifest_digest": manifest["manifest_digest"]}))
    return 0


def _sample_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count < 2:
        raise ValueError("invalid frame sampling request")
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def _video_strip(path: Path, *, count: int, width: int, label: str) -> Any:
    import cv2
    from PIL import Image, ImageDraw

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_indices(total, count)
    tiles = []
    try:
        for display_index, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"cannot decode {path} frame {frame_index}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            height = max(1, round(image.height * width / image.width))
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (width, height + 23), "black")
            tile.paste(image, (0, 23))
            ImageDraw.Draw(tile).text((5, 4), f"{display_index}", fill="white")
            tiles.append(tile)
    finally:
        capture.release()
    cell_height = max(tile.height for tile in tiles)
    left = 132
    strip = Image.new("RGB", (left + width * count, cell_height), "black")
    draw = ImageDraw.Draw(strip)
    draw.text((8, 8), label, fill="white")
    draw.text((8, 31), "left -> right", fill="white")
    for index, tile in enumerate(tiles):
        strip.paste(tile, (left + width * index, 0))
    return strip


def comparison_mosaic(
    sample: Mapping[str, Any], *, slot_map: Mapping[str, str], frame_count: int,
    tile_width: int,
) -> tuple[Any, str]:
    from PIL import Image, ImageDraw

    roles = {
        "anchor": Path(sample["generation"]["anchor"]["path"]),
        "frozen_base": Path(sample["generation"]["frozen_base"]["path"]),
    }
    rows = [
        _video_strip(Path(sample["source"]["path"]), count=frame_count, width=tile_width, label="SOURCE"),
        _video_strip(Path(sample["real_target"]["path"]), count=frame_count, width=tile_width, label="REAL TARGET"),
        _video_strip(roles[slot_map["A"]], count=frame_count, width=tile_width, label="CANDIDATE A"),
        _video_strip(roles[slot_map["B"]], count=frame_count, width=tile_width, label="CANDIDATE B"),
    ]
    gap = 5
    mosaic = Image.new(
        "RGB", (max(row.width for row in rows), sum(row.height for row in rows) + gap * 3), "#303030"
    )
    y = 0
    for row in rows:
        mosaic.paste(row, (0, y))
        y += row.height + gap
    buffer = io.BytesIO()
    mosaic.save(buffer, format="PNG")
    return mosaic, hashlib.sha256(buffer.getvalue()).hexdigest()


def _closed(value: Any, keys: Iterable[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} keys differ: {sorted(value) if isinstance(value, dict) else type(value)}")
    return value


def validate_qwen_observation(value: Any) -> Mapping[str, Any]:
    keys = {
        "schema_version", "reference_action_valid", "source_target_initial_comparable",
        "candidate_A", "candidate_B", "closer_to_reference_action", "confidence",
        "comparison_evidence",
    }
    row = _closed(value, keys, label="Qwen observation")
    if row["schema_version"] != QWEN_OBSERVATION_SCHEMA:
        raise ValueError("Qwen observation schema differs")
    for field in ("reference_action_valid", "source_target_initial_comparable"):
        if row[field] not in YES_NO_UNCERTAIN:
            raise ValueError(f"{field} enum differs")
    candidate_keys = set(SCORE_FIELDS) | {"action_observable", "artifact_blocks_action", "evidence"}
    for slot in ("A", "B"):
        candidate = _closed(row[f"candidate_{slot}"], candidate_keys, label=f"candidate {slot}")
        for field in SCORE_FIELDS:
            if type(candidate[field]) is not int or not 0 <= candidate[field] <= 4:
                raise ValueError(f"candidate {slot} {field} score differs")
        for field in ("action_observable", "artifact_blocks_action"):
            if candidate[field] not in YES_NO_UNCERTAIN:
                raise ValueError(f"candidate {slot} {field} enum differs")
        _validate_evidence(candidate["evidence"], label=f"candidate {slot} evidence")
    if row["closer_to_reference_action"] not in PAIRWISE:
        raise ValueError("pairwise enum differs")
    if row["confidence"] not in CONFIDENCE:
        raise ValueError("confidence enum differs")
    _validate_evidence(row["comparison_evidence"], label="comparison evidence")
    return row


def _validate_evidence(value: Any, *, label: str) -> None:
    if (
        not isinstance(value, list) or not 1 <= len(value) <= 3
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{label} differs")


def parse_json_object(raw: str) -> Mapping[str, Any]:
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return validate_qwen_observation(json.loads(text))


class QwenJudge:
    def __init__(self, model_path: Path, max_new_tokens: int) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.transformers_version = transformers.__version__
        self.max_new_tokens = max_new_tokens
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, device_map="auto",
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)

    def generate(self, image: Any, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _slot_maps(pair_id: str) -> list[dict[str, str]]:
    first_anchor = int(hashlib.sha256(pair_id.encode("ascii")).hexdigest()[:8], 16) % 2 == 0
    first = {"A": "anchor", "B": "frozen_base"} if first_anchor else {"A": "frozen_base", "B": "anchor"}
    return [first, {"A": first["B"], "B": first["A"]}]


def qwen_evaluate(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema differs")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index differs")
    samples = manifest["samples"]
    if args.pair_prefix:
        samples = [row for row in samples if row["pair_prefix"] == args.pair_prefix]
        if len(samples) != 1:
            raise ValueError("--pair-prefix did not resolve exactly one manifest sample")
    selected = [row for row in samples if row["ordinal"] % args.num_shards == args.shard_index]
    judge = QwenJudge(Path(args.model).resolve(strict=True), args.max_new_tokens)
    mosaic_root = Path(args.output).parent / "mosaics"
    mosaic_root.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in selected:
        passes = []
        for pass_index, slot_map in enumerate(_slot_maps(sample["pair_id"])):
            image, mosaic_sha = comparison_mosaic(
                sample, slot_map=slot_map, frame_count=args.frame_count, tile_width=args.tile_width
            )
            mosaic_path = mosaic_root / f"{sample['pair_prefix']}-pass-{pass_index}.png"
            image.save(mosaic_path, format="PNG")
            if file_sha256(mosaic_path) != mosaic_sha:
                raise RuntimeError("saved Qwen mosaic SHA-256 differs")
            raw = judge.generate(image, USER_TEMPLATE.format(instruction=sample["instruction"]))
            try:
                observation = parse_json_object(raw)
                parse_error = None
            except Exception as error:
                observation = None
                parse_error = f"{type(error).__name__}: {error}"
            passes.append({
                "pass_index": pass_index,
                "slot_map": slot_map,
                "mosaic_path": str(mosaic_path.resolve()),
                "mosaic_sha256": mosaic_sha,
                "raw_output": raw,
                "parse_error": parse_error,
                "observation": observation,
            })
        records.append({
            "schema_version": QWEN_RECORD_SCHEMA,
            "pair_id": sample["pair_id"],
            "pair_prefix": sample["pair_prefix"],
            "instruction": sample["instruction"],
            "passes": passes,
        })
        print(json.dumps({"pair_prefix": sample["pair_prefix"], "passes": len(passes)}), flush=True)
    write_jsonl(args.output, records)
    return 0


def _load_official_semantic_moments_embedder(semantic_root: Path) -> type:
    """Load only the pinned official moment implementation from its source tree."""
    base_path = semantic_root / "src/semantic_moments/embedders/base.py"
    if not base_path.is_file():
        raise FileNotFoundError(f"official SemanticMoments base.py is absent: {base_path}")
    spec = importlib.util.spec_from_file_location(
        "action_gap_official_semantic_moments_base", base_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official SemanticMoments base module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Embedder


def _uniform_frame_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        raise ValueError("video frame count and requested count must be positive")
    if count == 1:
        return [0]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def _load_video_frames(path: Path, count: int) -> list[Any]:
    """Uniformly sample by sequential decode so frame seeking cannot drift."""
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"video has no frames: {path}")
        targets = _uniform_frame_indices(total, count)
        positions: dict[int, list[int]] = {}
        for output_index, frame_index in enumerate(targets):
            positions.setdefault(frame_index, []).append(output_index)
        frames: list[Any | None] = [None] * count
        for frame_index in range(targets[-1] + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"cannot decode frame {frame_index} from {path}")
            for output_index in positions.get(frame_index, ()):
                frames[output_index] = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
    finally:
        capture.release()
    if any(frame is None for frame in frames):
        raise RuntimeError(f"sequential decode did not fill every sample for {path}")
    return list(frames)


def _semantic_moment_components(video_embedding: Any) -> Any:
    """Match official eb4ec98 moment math and unbiased torch.std exactly."""
    import torch
    import torch.nn.functional as F

    if video_embedding.ndim != 3 or video_embedding.shape[0] < 2:
        raise ValueError("expected at least two temporal samples in [T,P,D]")
    values = video_embedding.float()
    first = values.mean(dim=0)
    standard_deviation = values.std(dim=0)
    centered = values - first
    skew = (centered**3).mean(dim=0) / (standard_deviation**3 + 1.0e-6)
    return torch.stack(
        [
            F.normalize(first.mean(dim=0), dim=0, eps=1.0e-8),
            F.normalize(standard_deviation.mean(dim=0), dim=0, eps=1.0e-8),
            F.normalize(skew.mean(dim=0), dim=0, eps=1.0e-8),
        ],
        dim=0,
    )


def _compose_semantic_moments(components: Any, weights: Sequence[float]) -> Any:
    import torch
    import torch.nn.functional as F

    if components.ndim != 2 or components.shape[0] != 3 or len(weights) != 3:
        raise ValueError("expected three moment components and three weights")
    weight_tensor = torch.tensor(weights, dtype=torch.float32).view(3, 1)
    return F.normalize(
        (components.float() * weight_tensor).flatten(), dim=0, eps=1.0e-8
    )


class _LocalDINOv2:
    """Per-frame patch-token extractor backed only by a local DINOv2 tree."""

    def __init__(self, model_root: Path, device: str, frame_batch_size: int) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = torch.device(device)
        self.frame_batch_size = frame_batch_size
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_root), local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            str(model_root), local_files_only=True
        ).to(self.device).eval()
        self.num_register_tokens = int(
            getattr(self.model.config, "num_register_tokens", 0) or 0
        )

    def extract(self, frames: Sequence[Any]) -> Any:
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(frames), self.frame_batch_size):
                batch = frames[start : start + self.frame_batch_size]
                inputs = self.processor(images=list(batch), return_tensors="pt")
                pixels = inputs["pixel_values"].to(self.device)
                hidden = self.model(pixel_values=pixels).last_hidden_state
                patch_start = 1 + self.num_register_tokens
                outputs.append(hidden[:, patch_start:].detach().float().cpu())
        result = self.torch.cat(outputs, dim=0)
        if result.shape[0] != len(frames) or result.ndim != 3:
            raise RuntimeError(f"unexpected DINO feature geometry: {result.shape}")
        return result


def semantic_moments_extract(args: argparse.Namespace) -> int:
    import torch

    manifest = load_json(args.manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema differs")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index differs")
    samples = manifest["samples"]
    if args.pair_prefix:
        samples = [row for row in samples if row["pair_prefix"] == args.pair_prefix]
        if len(samples) != 1:
            raise ValueError("--pair-prefix did not resolve exactly one manifest sample")
    items = []
    for sample in samples:
        roles = {
            "source": sample["source"],
            "real_target": sample["real_target"],
            "anchor": sample["generation"]["anchor"],
            "frozen_base": sample["generation"]["frozen_base"],
        }
        for role, artifact in roles.items():
            path = Path(artifact["path"]).resolve(strict=True)
            items.append((sample, role, path))
    selected = [item for ordinal, item in enumerate(items) if ordinal % args.num_shards == args.shard_index]
    semantic_root = Path(args.semantic_moments_root).resolve(strict=True)
    model_root = Path(args.model_root).resolve(strict=True)
    official_class = _load_official_semantic_moments_embedder(semantic_root)
    official = official_class(alpha1=1.0, alpha2=8.0, alpha3=4.0, aggregation="concat")
    extractor = _LocalDINOv2(model_root, args.device, args.frame_batch_size)
    records = []
    for sample, role, path in selected:
        frames = _load_video_frames(path, args.num_frames)
        tokens = extractor.extract(frames)
        components = _semantic_moment_components(tokens)
        default = _compose_semantic_moments(components, (1.0, 8.0, 4.0))
        parity = float(torch.max(torch.abs(official.compute_moments(tokens).cpu() - default.cpu())))
        if parity > 2.0e-6:
            raise RuntimeError(f"official SemanticMoments parity failed: {parity}")
        records.append({
            "pair_id": sample["pair_id"], "pair_prefix": sample["pair_prefix"],
            "role": role, "path": str(path), "sha256": file_sha256(path),
            "components": components.cpu(), "official_default_parity_max_abs": parity,
            "feature_geometry": list(tokens.shape),
        })
        print(json.dumps({"pair_prefix": sample["pair_prefix"], "role": role}), flush=True)
    payload = {
        "schema_version": SM_FEATURE_SCHEMA, "created_at": utc_now(),
        "manifest_sha256": file_sha256(args.manifest), "manifest_digest": manifest["manifest_digest"],
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "num_frames": args.num_frames, "records": records,
        "semantic_moments_root": str(semantic_root), "model_root": str(model_root),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return 0


def _cosine(left: Any, right: Any) -> float:
    import torch.nn.functional as F
    return float(F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0))


def _compose(components: Any, weights: tuple[float, float, float]) -> Any:
    import torch
    import torch.nn.functional as F
    w = torch.tensor(weights, dtype=torch.float32).view(3, 1)
    return F.normalize((components.float() * w).flatten(), dim=0, eps=1e-8)


def _bootstrap(values: Sequence[float], seed: int = 20260819, draws: int = 4000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    estimates = [mean(rng.choice(values) for _ in values) for _ in range(draws)]
    estimates.sort()
    return [estimates[round(0.025 * (draws - 1))], estimates[round(0.975 * (draws - 1))]]


def semantic_moments_summarize(args: argparse.Namespace) -> int:
    import torch

    feature_paths = sorted(Path(args.features_dir).glob("features-shard-*.pt"))
    if not feature_paths:
        raise ValueError("no SemanticMoments feature shards found")
    records = []
    manifest_digest = None
    for path in feature_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != SM_FEATURE_SCHEMA:
            raise ValueError(f"feature schema differs: {path}")
        if manifest_digest is None:
            manifest_digest = payload["manifest_digest"]
        elif manifest_digest != payload["manifest_digest"]:
            raise ValueError("feature shards bind different manifests")
        records.extend(payload["records"])
    by_pair = defaultdict(dict)
    for row in records:
        if row["role"] in by_pair[row["pair_id"]]:
            raise ValueError("duplicate SemanticMoments feature role")
        by_pair[row["pair_id"]][row["role"]] = row
    weights = {
        "m1": (1.0, 0.0, 0.0), "m2": (0.0, 1.0, 0.0),
        "m3": (0.0, 0.0, 1.0), "m23": (0.0, 8.0, 4.0),
        "m123": (1.0, 8.0, 4.0),
    }
    pair_rows = []
    for pair_id, roles in sorted(by_pair.items()):
        if set(roles) != {"source", "real_target", "anchor", "frozen_base"}:
            raise ValueError(f"incomplete SemanticMoments roles for {pair_id}: {sorted(roles)}")
        similarities = {}
        for name, weight in weights.items():
            vectors = {role: _compose(value["components"], weight) for role, value in roles.items()}
            anchor = _cosine(vectors["anchor"], vectors["real_target"])
            base = _cosine(vectors["frozen_base"], vectors["real_target"])
            similarities[name] = {
                "anchor_to_target": anchor,
                "frozen_base_to_target": base,
                "anchor_minus_frozen_base": anchor - base,
                "source_to_target": _cosine(vectors["source"], vectors["real_target"]),
            }
        pair_rows.append({
            "pair_id": pair_id, "pair_prefix": roles["source"]["pair_prefix"],
            "similarities": similarities,
        })
    aggregate = {}
    for name in weights:
        deltas = [row["similarities"][name]["anchor_minus_frozen_base"] for row in pair_rows]
        aggregate[name] = {
            "mean_anchor_minus_frozen_base": mean(deltas),
            "median_anchor_minus_frozen_base": median(deltas),
            "bootstrap_95_ci_mean": _bootstrap(deltas),
            "anchor_wins": sum(value > 0 for value in deltas),
            "frozen_base_wins": sum(value < 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
        }
    result = {
        "schema_version": SM_SUMMARY_SCHEMA, "created_at": utc_now(),
        "manifest_digest": manifest_digest, "pair_count": len(pair_rows),
        "interpretation": "order-insensitive diagnostic only; cannot establish temporal direction or completion",
        "pairs": pair_rows, "aggregate": aggregate,
    }
    write_json(args.output, result)
    return 0


def _mapped_candidate(pass_row: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    slot_map = pass_row["slot_map"]
    slot = next(slot for slot, mapped_role in slot_map.items() if mapped_role == role)
    return pass_row["observation"][f"candidate_{slot}"]


def _mapped_pairwise(pass_row: Mapping[str, Any]) -> str:
    choice = pass_row["observation"]["closer_to_reference_action"]
    if choice in {"tie", "abstain"}:
        return choice
    return pass_row["slot_map"][choice]


def _candidate_gate(candidate: Mapping[str, Any]) -> int:
    return min(candidate[field] for field in ("action_semantics", "temporal_order", "action_completion"))


def qwen_summarize(args: argparse.Namespace) -> int:
    records = []
    for path in sorted(Path(args.records_dir).glob("qwen-shard-*.jsonl")):
        records.extend(load_jsonl(path))
    if not records:
        raise ValueError("no Qwen records found")
    pair_rows = []
    for row in sorted(records, key=lambda value: value["pair_prefix"]):
        valid_passes = [value for value in row["passes"] if value.get("observation") is not None]
        if len(valid_passes) != 2:
            pair_rows.append({
                "pair_id": row["pair_id"], "pair_prefix": row["pair_prefix"],
                "winner": "abstain", "reason": "one_or_more_qwen_passes_invalid",
                "valid_passes": len(valid_passes), "passes": row["passes"],
            })
            continue
        reference_valid = all(value["observation"]["reference_action_valid"] == "yes" for value in valid_passes)
        initial_comparable = all(value["observation"]["source_target_initial_comparable"] == "yes" for value in valid_passes)
        role_stats = {}
        for role in ("anchor", "frozen_base"):
            candidates = [_mapped_candidate(value, role) for value in valid_passes]
            role_stats[role] = {
                field: mean(candidate[field] for candidate in candidates) for field in SCORE_FIELDS
            }
            role_stats[role]["gate_score"] = mean(_candidate_gate(candidate) for candidate in candidates)
            role_stats[role]["both_passes_observable"] = all(candidate["action_observable"] == "yes" for candidate in candidates)
            role_stats[role]["artifact_blocks_any_pass"] = any(candidate["artifact_blocks_action"] == "yes" for candidate in candidates)
            role_stats[role]["evidence"] = [candidate["evidence"] for candidate in candidates]
        pass_winners = []
        for value in valid_passes:
            anchor_gate = _candidate_gate(_mapped_candidate(value, "anchor"))
            base_gate = _candidate_gate(_mapped_candidate(value, "frozen_base"))
            pass_winners.append("anchor" if anchor_gate > base_gate else "frozen_base" if base_gate > anchor_gate else "tie")
        direct_winners = [_mapped_pairwise(value) for value in valid_passes]
        if not reference_valid or not initial_comparable:
            winner, reason = "abstain", "reference_invalid_or_initial_state_incomparable"
        elif pass_winners[0] != pass_winners[1]:
            winner, reason = "abstain", "slot_swap_gate_winner_unstable"
        elif pass_winners[0] == "tie":
            winner, reason = "tie", "noncompensatory_gate_tie"
        else:
            winner, reason = pass_winners[0], "noncompensatory_gate_consistent_across_slot_swap"
        pair_rows.append({
            "pair_id": row["pair_id"], "pair_prefix": row["pair_prefix"],
            "reference_valid_both_passes": reference_valid,
            "initial_comparable_both_passes": initial_comparable,
            "role_scores": role_stats, "gate_pass_winners": pass_winners,
            "direct_pairwise_winners": direct_winners, "winner": winner,
            "reason": reason, "passes": row["passes"],
        })
    counts = Counter(row["winner"] for row in pair_rows)
    gate_deltas = [
        row["role_scores"]["anchor"]["gate_score"] - row["role_scores"]["frozen_base"]["gate_score"]
        for row in pair_rows if "role_scores" in row
    ]
    result = {
        "schema_version": QWEN_SUMMARY_SCHEMA, "created_at": utc_now(),
        "pair_count": len(pair_rows), "winner_counts": dict(sorted(counts.items())),
        "mean_anchor_minus_frozen_base_gate": mean(gate_deltas) if gate_deltas else None,
        "bootstrap_95_ci_mean_gate_delta": _bootstrap(gate_deltas) if gate_deltas else None,
        "decision_rule": "per-pass min(action_semantics, temporal_order, action_completion); require same winner after A/B slot swap",
        "pairs": pair_rows,
    }
    write_json(args.output, result)
    return 0


def final_report(args: argparse.Namespace) -> int:
    qwen = load_json(args.qwen_summary)
    sm = load_json(args.semantic_moments_summary)
    if qwen.get("schema_version") != QWEN_SUMMARY_SCHEMA or sm.get("schema_version") != SM_SUMMARY_SCHEMA:
        raise ValueError("summary schema differs")
    sm_by_pair = {row["pair_id"]: row for row in sm["pairs"]}
    pairs = []
    conflicts = 0
    for row in qwen["pairs"]:
        sm_row = sm_by_pair.get(row["pair_id"])
        sm_m3_winner = None
        if sm_row is not None:
            delta = sm_row["similarities"]["m3"]["anchor_minus_frozen_base"]
            sm_m3_winner = "anchor" if delta > 0 else "frozen_base" if delta < 0 else "tie"
        qwen_winner = row["winner"]
        conflict = qwen_winner in {"anchor", "frozen_base"} and sm_m3_winner in {"anchor", "frozen_base"} and qwen_winner != sm_m3_winner
        conflicts += int(conflict)
        pairs.append({
            "pair_id": row["pair_id"], "pair_prefix": row["pair_prefix"],
            "qwen_primary_winner": qwen_winner, "qwen_reason": row["reason"],
            "semantic_moments_m3_diagnostic_winner": sm_m3_winner,
            "qwen_sm_conflict": conflict,
        })
    report = {
        "schema_version": FINAL_SCHEMA, "created_at": utc_now(),
        "primary_conclusion": qwen["winner_counts"],
        "qwen_mean_anchor_minus_frozen_base_gate": qwen["mean_anchor_minus_frozen_base_gate"],
        "qwen_bootstrap_95_ci_mean_gate_delta": qwen["bootstrap_95_ci_mean_gate_delta"],
        "semantic_moments_m3": sm["aggregate"]["m3"],
        "qwen_semantic_moments_conflict_count": conflicts,
        "authority": {
            "winner": "Qwen3-VL temporal/action rubric with blinded slot swap",
            "semantic_moments": "diagnostic only because moments are invariant to time permutation/reversal",
        },
        "pairs": pairs,
    }
    write_json(args.output, report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-manifest")
    build.add_argument("--metadata", required=True)
    build.add_argument("--selection", required=True)
    build.add_argument("--experiment-root", required=True)
    build.add_argument("--seed-base", type=int, default=2026081900)
    build.add_argument("--output", required=True)
    build.set_defaults(function=build_manifest)

    qwen = sub.add_parser("qwen-evaluate")
    qwen.add_argument("--manifest", required=True)
    qwen.add_argument("--model", required=True)
    qwen.add_argument("--shard-index", type=int, required=True)
    qwen.add_argument("--num-shards", type=int, default=2)
    qwen.add_argument("--pair-prefix")
    qwen.add_argument("--frame-count", type=int, default=10)
    qwen.add_argument("--tile-width", type=int, default=144)
    qwen.add_argument("--max-new-tokens", type=int, default=1536)
    qwen.add_argument("--output", required=True)
    qwen.set_defaults(function=qwen_evaluate)

    extract = sub.add_parser("semantic-moments-extract")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--semantic-moments-root", required=True)
    extract.add_argument("--model-root", required=True)
    extract.add_argument("--shard-index", type=int, required=True)
    extract.add_argument("--num-shards", type=int, default=8)
    extract.add_argument("--pair-prefix")
    extract.add_argument("--num-frames", type=int, default=32)
    extract.add_argument("--frame-batch-size", type=int, default=8)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--output", required=True)
    extract.set_defaults(function=semantic_moments_extract)

    sm_sum = sub.add_parser("semantic-moments-summarize")
    sm_sum.add_argument("--features-dir", required=True)
    sm_sum.add_argument("--output", required=True)
    sm_sum.set_defaults(function=semantic_moments_summarize)

    qwen_sum = sub.add_parser("qwen-summarize")
    qwen_sum.add_argument("--records-dir", required=True)
    qwen_sum.add_argument("--output", required=True)
    qwen_sum.set_defaults(function=qwen_summarize)

    final = sub.add_parser("final-report")
    final.add_argument("--qwen-summary", required=True)
    final.add_argument("--semantic-moments-summary", required=True)
    final.add_argument("--output", required=True)
    final.set_defaults(function=final_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
