"""Committed R5 endpoint extraction and content-cluster split pipeline.

The two array workers in this module only extract raw source/target endpoint
descriptors.  They never assign a split.  ``finalize`` first validates every
committed worker artifact, globally merges exact/near source perceptual hashes
with a DSU, and only then assigns whole clusters to deterministic splits.

``final.npz`` exposes both the singular trainer contract
(``split``, ``content_group_id``, ``split_version``) and plural aliases expected
by :class:`motive.source_aware_repr.R5EndpointBatch`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .action_repr import PROMPT_HASH_VERSION
from .descriptor import DescriptorConfig, encode_action_descriptor
from .geometry import MotionAnalysis, MotionConfig, analyze_video
from .source_aware_repr import (
    R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION,
    R5EndpointBatch,
    R5_SCHEMA_VERSION,
    audit_content_disjoint_splits,
    stable_splits_from_content_groups,
)


R5_FEATURE_SCHEMA = "motive-r5-endpoint-features-v1"
R5_TASK_SCHEMA = "motive-r5-endpoint-task-v1"
R5_FINAL_SCHEMA = "motive-r5-endpoint-final-v1"
R5_ENDPOINT_LAYOUT = "raw-action-descriptor-actor-then-camera-v1"
R5_PHASH_SPLIT_VERSION = R5_DIAGNOSTIC_CONTENT_SPLIT_VERSION
DEFAULT_CAMERA_DIMS = 8
DEFAULT_TASK_COUNT = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'_-]*")
_INT_BIT_COUNT = getattr(int, "bit_count", None)


def _popcount(value: int) -> int:
    return (
        int(_INT_BIT_COUNT(value))
        if _INT_BIT_COUNT is not None
        else bin(value).count("1")
    )


@dataclass(frozen=True)
class R5FeatureConfig:
    analysis_frames: int = 20
    resize_width: int = 160
    active_speed_threshold: float = 0.005
    camera_dims: int = DEFAULT_CAMERA_DIMS
    perceptual_hash_frames: int = 6
    instruction_dim: int = 512

    def validate(self) -> None:
        if self.analysis_frames < 3:
            raise ValueError("analysis_frames must be >= 3")
        if self.resize_width < 32:
            raise ValueError("resize_width must be >= 32")
        if not math.isfinite(self.active_speed_threshold):
            raise ValueError("active_speed_threshold must be finite")
        if self.active_speed_threshold < 0.0:
            raise ValueError("active_speed_threshold must be non-negative")
        if self.camera_dims != DEFAULT_CAMERA_DIMS:
            raise ValueError(
                "R5 endpoint schema requires the descriptor's final 8 camera dims"
            )
        if self.perceptual_hash_frames < 1:
            raise ValueError("perceptual_hash_frames must be positive")
        if self.instruction_dim < 1:
            raise ValueError("instruction_dim must be positive")


@dataclass(frozen=True)
class ClusterResult:
    group_ids: tuple[str, ...]
    groups: int
    exact_digest_unions: int
    exact_phash_unions: int
    near_phash_unions: int
    near_pairs_examined: int
    maximum_hamming_fraction: float
    maximum_group_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _DSU:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        return True


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _root_contract(root: Path) -> dict[str, str]:
    resolved = root.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return {
        "kind": "resolved-data-root-v1",
        "resolved_path": str(resolved),
    }


def _implementation_digest() -> str:
    package = Path(__file__).resolve().parent
    names = (
        "source_aware_features.py",
        "source_aware_repr.py",
        "descriptor.py",
        "geometry.py",
        "action_repr.py",
    )
    return _object_digest({name: _file_digest(package / name) for name in names})


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _validated_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _select_string(
    row: Mapping[str, Any],
    names: Sequence[str],
    *,
    context: str,
    allow_empty: bool = False,
) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and (allow_empty or value.strip()):
            return value.strip()
    raise ValueError(f"{context} is missing one of {list(names)}")


def _normalized_row(
    raw: Mapping[str, Any],
    *,
    input_index: int,
    context: str,
) -> dict[str, Any]:
    iid = _select_string(raw, ("iid", "case_id", "id"), context=context)
    prompt = _select_string(
        raw,
        ("prompt", "instruction_en", "instruction"),
        context=context,
    )
    source_video = _select_string(
        raw,
        ("src_video", "source_video"),
        context=context,
    )
    target_video = _select_string(
        raw,
        ("tgt_video", "target_video", "edited_video"),
        context=context,
    )
    supplied_input_digest = raw.get("input_digest")
    if supplied_input_digest is None:
        input_digest = _object_digest(
            {
                "iid": iid,
                "prompt": prompt,
                "src_video": source_video,
                "tgt_video": target_video,
                "source_caption": str(raw.get("source_caption") or ""),
                "edited_caption": str(raw.get("edited_caption") or ""),
            }
        )
        input_digest_origin = "derived-r5-core-fields-v1"
    else:
        input_digest = _validated_sha256(
            supplied_input_digest,
            name=f"{context} input_digest",
        )
        input_digest_origin = "upstream"
    human_review = raw.get("human_review")
    if human_review is not None and not isinstance(human_review, dict):
        raise ValueError(f"{context} human_review must be an object or null")
    qwen_evidence = raw.get("qwen_evidence")
    if qwen_evidence is not None and not isinstance(qwen_evidence, dict):
        raise ValueError(f"{context} qwen_evidence must be an object or null")
    pilot_label = _r5_pilot_label(raw, context=context)
    action_signature = _action_signature(raw, pilot_label=pilot_label)
    return {
        "input_index": int(input_index),
        "iid": iid,
        "prompt": prompt,
        "src_video": source_video,
        "tgt_video": target_video,
        "input_digest": input_digest,
        "input_digest_origin": input_digest_origin,
        "row_digest": _object_digest(raw),
        "human_review": human_review,
        "qwen_evidence": qwen_evidence,
        "r5_pilot_label": pilot_label["raw"],
        "label_role": pilot_label["label_role"],
        "label_type": pilot_label["label_type"],
        "eligible_positive": pilot_label["eligible_positive"],
        "production_eligible": pilot_label["production_eligible"],
        "pilot_action_signature": pilot_label["pilot_action_signature"],
        "action_signature": action_signature,
        "action_family": _action_family(
            raw,
            pilot_label=pilot_label,
            action_signature=action_signature,
        ),
    }


def _r5_pilot_label(
    row: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    raw = row.get("r5_pilot_label")
    if raw is None:
        return {
            "raw": None,
            "label_role": "unlabeled",
            "label_type": "unlabeled",
            "eligible_positive": False,
            "production_eligible": False,
            "pilot_action_signature": "",
        }
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} r5_pilot_label must be an object")
    label_class = str(raw.get("class") or "").strip()
    if label_class not in {"positive", "negative"}:
        raise ValueError(
            f"{context} r5_pilot_label.class must be positive or negative"
        )
    production_eligible = raw.get("production_eligible")
    if not isinstance(production_eligible, bool):
        raise ValueError(
            f"{context} r5_pilot_label.production_eligible must be boolean"
        )
    if production_eligible:
        raise ValueError(
            f"{context} pHash-only pilot cannot be production_eligible"
        )
    pilot_action_signature = str(raw.get("action_signature") or "").strip()
    negative_type = str(raw.get("negative_type") or "").strip()
    if label_class == "positive":
        if not pilot_action_signature:
            raise ValueError(
                f"{context} positive r5_pilot_label requires action_signature"
            )
        label_role = "positive_delta"
        label_type = "positive"
        eligible_positive = True
    else:
        if not negative_type:
            raise ValueError(
                f"{context} negative r5_pilot_label requires negative_type"
            )
        label_role = "negative_audit"
        label_type = negative_type
        eligible_positive = False
    return {
        "raw": dict(raw),
        "label_role": label_role,
        "label_type": label_type,
        "eligible_positive": eligible_positive,
        "production_eligible": production_eligible,
        "pilot_action_signature": pilot_action_signature,
    }


def _action_signature(
    row: Mapping[str, Any],
    *,
    pilot_label: Mapping[str, Any] | None = None,
) -> str:
    if pilot_label is not None:
        if pilot_label["label_role"] == "negative_audit":
            return f"negative:{pilot_label['label_type']}"
        pilot_signature = str(pilot_label["pilot_action_signature"]).strip()
        if pilot_signature:
            return pilot_signature
    candidates: list[Any] = [row.get("action_signature")]
    for field in ("human_review", "final_triage"):
        nested = row.get(field)
        if isinstance(nested, Mapping):
            candidates.append(nested.get("action_signature"))
    qwen = row.get("qwen_evidence")
    if isinstance(qwen, Mapping):
        for branch in ("visual", "text"):
            record = qwen.get(branch)
            if isinstance(record, Mapping):
                result = record.get("result")
                if isinstance(result, Mapping):
                    candidates.append(result.get("action_signature"))
    rule = row.get("auto_rule")
    if isinstance(rule, Mapping):
        families = rule.get("action_families")
        if isinstance(families, Sequence) and not isinstance(families, (str, bytes)):
            candidates.append("+".join(str(value) for value in families if value))
        candidates.append(rule.get("label"))
    semantics = row.get("instruction_semantics")
    if isinstance(semantics, Mapping):
        candidates.append(semantics.get("label"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "unknown"


def _action_family(
    row: Mapping[str, Any],
    *,
    pilot_label: Mapping[str, Any],
    action_signature: str,
) -> str:
    pilot_raw = pilot_label.get("raw")
    if isinstance(pilot_raw, Mapping):
        value = pilot_raw.get("action_family")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if pilot_label.get("label_role") == "negative_audit":
        return "negative"
    rule = row.get("auto_rule")
    if isinstance(rule, Mapping):
        families = rule.get("action_families")
        if isinstance(families, Sequence) and not isinstance(families, (str, bytes)):
            values = [str(value).strip() for value in families if str(value).strip()]
            if values:
                return values[0]
        label = rule.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    normalized = action_signature.strip().lower()
    return normalized.split(":", 1)[0] if normalized else "unknown"


def _read_source_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    path = path.expanduser().resolve(strict=True)
    digest_before = _file_digest(path)
    raw_rows = _load_jsonl(path)
    digest_after = _file_digest(path)
    if digest_after != digest_before:
        raise RuntimeError(f"{path} changed while it was being read")
    if not raw_rows:
        raise ValueError(f"{path} has no JSONL rows")
    rows = [
        _normalized_row(
            raw,
            input_index=index,
            context=f"{path}:{index + 1}",
        )
        for index, raw in enumerate(raw_rows)
    ]
    iids = [row["iid"] for row in rows]
    if len(set(iids)) != len(iids):
        duplicates = sorted(
            iid for iid in set(iids) if iids.count(iid) > 1
        )
        raise ValueError(f"{path} contains duplicate iids: {duplicates[:8]}")
    return rows, digest_before


def _resolve_video(root: Path, value: str) -> Path:
    resolved_root = root.expanduser().resolve(strict=True)
    candidate = Path(value).expanduser()
    candidate = candidate if candidate.is_absolute() else resolved_root / candidate
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"video path escapes the declared data root: {resolved}"
        ) from error
    return resolved


def instruction_hash_features(
    prompts: Sequence[str],
    *,
    feature_dim: int = 512,
) -> np.ndarray:
    """NumPy implementation of Lucy's signed unigram/bigram/trigram hash."""

    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    features = np.zeros((len(prompts), feature_dim), dtype=np.float32)
    for row_index, prompt in enumerate(prompts):
        tokens = _TOKEN_RE.findall(str(prompt).lower())
        grams: list[str] = list(tokens)
        grams.extend(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
        grams.extend(
            f"{a} {b} {c}"
            for a, b, c in zip(tokens, tokens[1:], tokens[2:])
        )
        if not grams:
            grams = ["<empty>"]
        scale = float(len(grams)) ** -0.5
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little", signed=False)
            column = value % feature_dim
            sign = 1.0 if ((value >> 63) & 1) == 0 else -1.0
            features[row_index, column] += sign * scale
        norm = float(np.linalg.norm(features[row_index]))
        if norm > 1e-6:
            features[row_index] /= norm
    return features


def source_perceptual_fingerprint(
    frames_gray: np.ndarray,
    *,
    max_frames: int = 6,
) -> dict[str, str]:
    """Return a codec-tolerant multi-frame pHash plus a stronger exact digest."""

    frames = np.asarray(frames_gray)
    if frames.ndim != 3 or len(frames) < 1:
        raise ValueError("frames_gray must have shape [T,H,W] with T >= 1")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    indices = np.rint(
        np.linspace(0, len(frames) - 1, num=min(max_frames, len(frames)))
    ).astype(np.int64)
    exact = hashlib.sha256()
    perceptual = bytearray()
    for frame_index in indices:
        frame = np.asarray(frames[int(frame_index)], dtype=np.uint8)
        compact = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        exact.update((compact >> 4).tobytes())
        coefficients = cv2.dct(compact.astype(np.float32) / 255.0)[:8, :8]
        low_frequency = coefficients.reshape(-1)[1:]
        median = float(np.median(low_frequency))
        perceptual.extend(
            np.packbits(
                (low_frequency >= median).astype(np.uint8),
                bitorder="little",
            ).tobytes()
        )
    return {
        "sampled_frame_digest": exact.hexdigest(),
        "perceptual_hash": bytes(perceptual).hex(),
    }


def endpoint_blocks(
    analysis: MotionAnalysis,
    *,
    descriptor_config: DescriptorConfig,
    camera_dims: int = DEFAULT_CAMERA_DIMS,
) -> tuple[np.ndarray, np.ndarray]:
    descriptor = encode_action_descriptor(
        analysis.residual_flows,
        analysis.frame_times,
        analysis.frames_gray.shape[2],
        global_flows=analysis.global_flows,
        config=descriptor_config,
        normalize=False,
    )
    if descriptor.ndim != 1 or len(descriptor) <= camera_dims:
        raise ValueError("action descriptor is too small for the camera block")
    actor = np.asarray(descriptor[:-camera_dims], dtype=np.float32)
    camera = np.asarray(descriptor[-camera_dims:], dtype=np.float32)
    if not np.isfinite(actor).all() or not np.isfinite(camera).all():
        raise ValueError("endpoint descriptor contains non-finite values")
    return actor, camera


def _extract_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    row = dict(payload["row"])
    root = Path(str(payload["data_root"]))
    config = R5FeatureConfig(**dict(payload["config"]))
    config.validate()
    motion_config = MotionConfig(
        analysis_frames=config.analysis_frames,
        resize_width=config.resize_width,
        active_speed_threshold=config.active_speed_threshold,
    )
    descriptor_config = DescriptorConfig(
        active_speed_threshold=config.active_speed_threshold
    )
    source_path = _resolve_video(root, row["src_video"])
    target_path = _resolve_video(root, row["tgt_video"])
    source_sha256 = _file_digest(source_path)
    target_sha256 = _file_digest(target_path)
    source = analyze_video(source_path, motion_config)
    target = analyze_video(target_path, motion_config)
    if _file_digest(source_path) != source_sha256:
        raise RuntimeError(f"{source_path} changed while it was decoded")
    if _file_digest(target_path) != target_sha256:
        raise RuntimeError(f"{target_path} changed while it was decoded")
    source_actor, source_camera = endpoint_blocks(
        source,
        descriptor_config=descriptor_config,
        camera_dims=config.camera_dims,
    )
    target_actor, target_camera = endpoint_blocks(
        target,
        descriptor_config=descriptor_config,
        camera_dims=config.camera_dims,
    )
    if source_actor.shape != target_actor.shape:
        raise ValueError(f"{row['iid']} actor endpoint dimensions differ")
    if source_camera.shape != target_camera.shape:
        raise ValueError(f"{row['iid']} camera endpoint dimensions differ")
    fingerprint = source_perceptual_fingerprint(
        source.frames_gray,
        max_frames=config.perceptual_hash_frames,
    )
    manifest = {
        **row,
        "source_resolved_path": str(source_path),
        "target_resolved_path": str(target_path),
        "source_video_sha256": source_sha256,
        "target_video_sha256": target_sha256,
        "source_sampled_frame_digest": fingerprint["sampled_frame_digest"],
        "source_perceptual_hash": fingerprint["perceptual_hash"],
        "source_motion_label": source.label,
        "target_motion_label": target.label,
    }
    return {
        "manifest": manifest,
        "source_actor": source_actor,
        "source_camera": source_camera,
        "target_actor": target_actor,
        "target_camera": target_camera,
    }


def _feature_config_digest(config: R5FeatureConfig) -> str:
    config.validate()
    return _object_digest(
        {
            "schema_version": R5_FEATURE_SCHEMA,
            "endpoint_layout": R5_ENDPOINT_LAYOUT,
            "prompt_hash_version": PROMPT_HASH_VERSION,
            "config": asdict(config),
        }
    )


def _task_paths(task_dir: Path) -> dict[str, Path]:
    return {
        "archive": task_dir / "features.npz",
        "manifest": task_dir / "manifest.jsonl",
        "summary": task_dir / "summary.json",
        "done": task_dir / "done.json",
    }


def _ensure_fresh_or_committed(task_dir: Path, *, resume: bool) -> bool:
    paths = _task_paths(task_dir)
    existing = [name for name, path in paths.items() if path.exists()]
    if not existing:
        return False
    if resume and len(existing) == len(paths) and paths["done"].is_file():
        _load_committed_task(task_dir)
        return True
    raise RuntimeError(
        f"{task_dir} contains an incomplete/conflicting artifact set: {existing}"
    )


def _stack_results(
    results: Sequence[Mapping[str, Any]],
    *,
    instruction_dim: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if not results:
        raise ValueError("an extraction task must contain at least one row")
    manifests: list[dict[str, Any]] = []
    matrix_names = (
        "source_actor",
        "source_camera",
        "target_actor",
        "target_camera",
    )
    arrays: dict[str, np.ndarray] = {}
    for name in matrix_names:
        arrays[name] = np.stack(
            [np.asarray(result[name], dtype=np.float32) for result in results]
        )
        if arrays[name].ndim != 2 or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must be a finite [N,D] matrix")
    if arrays["source_actor"].shape != arrays["target_actor"].shape:
        raise ValueError("source/target actor endpoint matrices differ")
    if arrays["source_camera"].shape != arrays["target_camera"].shape:
        raise ValueError("source/target camera endpoint matrices differ")
    for feature_index, result in enumerate(results):
        manifest = dict(result["manifest"])
        manifest["feature_index"] = feature_index
        manifests.append(manifest)
    prompts = [str(row["prompt"]) for row in manifests]
    arrays["instruction_features"] = instruction_hash_features(
        prompts,
        feature_dim=instruction_dim,
    )
    arrays["iids"] = np.asarray([row["iid"] for row in manifests], dtype=str)
    arrays["source_perceptual_hash"] = np.asarray(
        [row["source_perceptual_hash"] for row in manifests],
        dtype=str,
    )
    arrays["source_sampled_frame_digest"] = np.asarray(
        [row["source_sampled_frame_digest"] for row in manifests],
        dtype=str,
    )
    return arrays, manifests


def _commit_task(
    task_dir: Path,
    *,
    results: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _task_paths(task_dir)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"refusing to overwrite task artifacts in {task_dir}")
    config = R5FeatureConfig(**dict(provenance["config"]))
    arrays, manifests = _stack_results(
        results,
        instruction_dim=config.instruction_dim,
    )
    actor_dim = int(arrays["source_actor"].shape[1])
    camera_dim = int(arrays["source_camera"].shape[1])
    metadata = {
        "schema_version": R5_TASK_SCHEMA,
        "endpoint_layout": R5_ENDPOINT_LAYOUT,
        "actor_dim": actor_dim,
        "camera_dim": camera_dim,
        "instruction_dim": int(arrays["instruction_features"].shape[1]),
        "rows": len(manifests),
        "provenance": dict(provenance),
        "row_contract_digest": _object_digest(
            [
                {
                    "input_index": row["input_index"],
                    "iid": row["iid"],
                    "input_digest": row["input_digest"],
                    "row_digest": row["row_digest"],
                    "source_video_sha256": row["source_video_sha256"],
                    "target_video_sha256": row["target_video_sha256"],
                    "source_perceptual_hash": row["source_perceptual_hash"],
                }
                for row in manifests
            ]
        ),
    }
    arrays["metadata_json"] = np.asarray(_canonical_json(metadata))
    task_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(paths["archive"], arrays)
    _atomic_jsonl(paths["manifest"], manifests)
    summary = {
        "schema_version": R5_TASK_SCHEMA,
        "stage": "extract",
        "rows": len(manifests),
        "actor_dim": actor_dim,
        "camera_dim": camera_dim,
        "instruction_dim": int(arrays["instruction_features"].shape[1]),
        "endpoint_layout": R5_ENDPOINT_LAYOUT,
        "provenance": dict(provenance),
        "archive_sha256": _file_digest(paths["archive"]),
        "manifest_sha256": _file_digest(paths["manifest"]),
    }
    _atomic_json(paths["summary"], summary)
    done = {
        "schema_version": R5_TASK_SCHEMA,
        "rows": len(manifests),
        "archive_sha256": summary["archive_sha256"],
        "manifest_sha256": summary["manifest_sha256"],
        "summary_sha256": _file_digest(paths["summary"]),
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "data_root_digest": provenance["data_root_digest"],
        "config_digest": provenance["config_digest"],
        "implementation_digest": provenance["implementation_digest"],
        "task_index": provenance["task_index"],
        "task_count": provenance["task_count"],
    }
    _atomic_json(paths["done"], done)
    return done


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _load_committed_task(task_dir: Path) -> dict[str, Any]:
    paths = _task_paths(task_dir)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    done = _load_json(paths["done"])
    summary = _load_json(paths["summary"])
    if done.get("schema_version") != R5_TASK_SCHEMA:
        raise ValueError(f"{paths['done']} has an incompatible schema")
    expected_digests = {
        "archive_sha256": _file_digest(paths["archive"]),
        "manifest_sha256": _file_digest(paths["manifest"]),
        "summary_sha256": _file_digest(paths["summary"]),
    }
    for name, actual in expected_digests.items():
        if done.get(name) != actual:
            raise ValueError(f"{task_dir} {name} mismatch")
    if summary.get("archive_sha256") != expected_digests["archive_sha256"]:
        raise ValueError(f"{task_dir} summary archive digest mismatch")
    if summary.get("manifest_sha256") != expected_digests["manifest_sha256"]:
        raise ValueError(f"{task_dir} summary manifest digest mismatch")
    arrays = _load_npz(paths["archive"])
    required = {
        "iids",
        "source_actor",
        "source_camera",
        "target_actor",
        "target_camera",
        "instruction_features",
        "source_perceptual_hash",
        "source_sampled_frame_digest",
        "metadata_json",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{paths['archive']} is missing {missing}")
    metadata_value = arrays["metadata_json"]
    if metadata_value.ndim != 0:
        raise ValueError("task metadata_json must be scalar")
    metadata = json.loads(str(metadata_value.item()))
    if metadata.get("schema_version") != R5_TASK_SCHEMA:
        raise ValueError("task archive metadata schema mismatch")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict) or provenance != summary.get("provenance"):
        raise ValueError("task archive/summary provenance mismatch")
    if _feature_config_digest(R5FeatureConfig(**provenance["config"])) != provenance.get(
        "config_digest"
    ):
        raise ValueError("task config digest mismatch")
    if _object_digest(provenance["data_root_contract"]) != provenance.get(
        "data_root_digest"
    ):
        raise ValueError("task data-root digest mismatch")
    for name in (
        "source_manifest_sha256",
        "data_root_digest",
        "config_digest",
        "implementation_digest",
        "task_index",
        "task_count",
    ):
        if done.get(name) != provenance.get(name):
            raise ValueError(f"task done/provenance {name} mismatch")
    if _implementation_digest() != provenance.get("implementation_digest"):
        raise ValueError("task implementation changed after extraction")
    source_manifest = Path(str(provenance["source_manifest"]))
    if _file_digest(source_manifest) != provenance.get("source_manifest_sha256"):
        raise ValueError("source manifest changed after task extraction")
    root = Path(str(provenance["data_root_contract"]["resolved_path"]))
    if _root_contract(root) != provenance["data_root_contract"]:
        raise ValueError("data root identity changed after task extraction")
    rows = _load_jsonl(paths["manifest"])
    count = len(rows)
    if count < 1 or count != int(done.get("rows", -1)):
        raise ValueError("task row count is empty or disagrees with done marker")
    if count != int(metadata.get("rows", -1)):
        raise ValueError("task row count disagrees with archive metadata")
    for name in required - {"metadata_json"}:
        if len(arrays[name]) != count:
            raise ValueError(f"task {name} row count mismatch")
    for name in (
        "source_actor",
        "source_camera",
        "target_actor",
        "target_camera",
        "instruction_features",
    ):
        values = np.asarray(arrays[name], dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"task {name} is not a finite matrix")
        arrays[name] = values
    if arrays["source_actor"].shape != arrays["target_actor"].shape:
        raise ValueError("task actor endpoint shapes differ")
    if arrays["source_camera"].shape != arrays["target_camera"].shape:
        raise ValueError("task camera endpoint shapes differ")
    if int(metadata.get("actor_dim", -1)) != arrays["source_actor"].shape[1]:
        raise ValueError("task actor_dim mismatch")
    if int(metadata.get("camera_dim", -1)) != arrays["source_camera"].shape[1]:
        raise ValueError("task camera_dim mismatch")
    if int(metadata.get("instruction_dim", -1)) != arrays[
        "instruction_features"
    ].shape[1]:
        raise ValueError("task instruction_dim mismatch")
    if metadata.get("endpoint_layout") != R5_ENDPOINT_LAYOUT:
        raise ValueError("task endpoint layout mismatch")
    for index, row in enumerate(rows):
        if int(row.get("feature_index", -1)) != index:
            raise ValueError("task feature_index is not contiguous")
        if str(arrays["iids"][index]) != str(row.get("iid")):
            raise ValueError("task iid/archive alignment mismatch")
        if str(arrays["source_perceptual_hash"][index]) != str(
            row.get("source_perceptual_hash")
        ):
            raise ValueError("task perceptual hash/archive alignment mismatch")
    row_contract_digest = _object_digest(
        [
            {
                "input_index": row["input_index"],
                "iid": row["iid"],
                "input_digest": row["input_digest"],
                "row_digest": row["row_digest"],
                "source_video_sha256": row["source_video_sha256"],
                "target_video_sha256": row["target_video_sha256"],
                "source_perceptual_hash": row["source_perceptual_hash"],
            }
            for row in rows
        ]
    )
    if metadata.get("row_contract_digest") != row_contract_digest:
        raise ValueError("task row-contract digest mismatch")
    return {
        "arrays": arrays,
        "rows": rows,
        "metadata": metadata,
        "summary": summary,
        "done": done,
    }


def cluster_source_hashes(
    *,
    exact_digests: Sequence[str],
    perceptual_hashes: Sequence[str],
    maximum_hamming_fraction: float = 0.10,
) -> ClusterResult:
    """Globally merge exact and near source hashes into deterministic groups."""

    if len(exact_digests) != len(perceptual_hashes) or not exact_digests:
        raise ValueError("exact/perceptual hashes require equal non-zero lengths")
    if not 0.0 <= maximum_hamming_fraction < 1.0:
        raise ValueError("maximum_hamming_fraction must be in [0,1)")
    exact_values = [
        _validated_sha256(value, name=f"exact_digests[{index}]")
        for index, value in enumerate(exact_digests)
    ]
    decoded: list[bytes] = []
    for index, value in enumerate(perceptual_hashes):
        try:
            raw = bytes.fromhex(str(value))
        except ValueError as error:
            raise ValueError(f"perceptual_hashes[{index}] is not hex") from error
        if not raw:
            raise ValueError(f"perceptual_hashes[{index}] is empty")
        decoded.append(raw)
    lengths = {len(value) for value in decoded}
    if len(lengths) != 1:
        raise ValueError("perceptual hashes must have a common length")

    dsu = _DSU(len(decoded))
    exact_digest_unions = 0
    exact_phash_unions = 0
    representatives: dict[str, int] = {}
    for index, value in enumerate(exact_values):
        if value in representatives:
            exact_digest_unions += int(dsu.union(index, representatives[value]))
        else:
            representatives[value] = index
    phash_representatives: dict[bytes, int] = {}
    for index, value in enumerate(decoded):
        if value in phash_representatives:
            exact_phash_unions += int(
                dsu.union(index, phash_representatives[value])
            )
        else:
            phash_representatives[value] = index

    unique = sorted(phash_representatives.items(), key=lambda item: item[0])
    bit_count = 8 * next(iter(lengths))
    near_phash_unions = 0
    near_pairs_examined = 0
    for left in range(len(unique)):
        left_hash, left_index = unique[left]
        left_integer = int.from_bytes(left_hash, "big")
        for right in range(left + 1, len(unique)):
            near_pairs_examined += 1
            right_hash, right_index = unique[right]
            distance = _popcount(
                left_integer ^ int.from_bytes(right_hash, "big")
            ) / float(bit_count)
            if distance <= maximum_hamming_fraction:
                near_phash_unions += int(dsu.union(left_index, right_index))

    components: dict[int, list[int]] = {}
    for index in range(len(decoded)):
        components.setdefault(dsu.find(index), []).append(index)
    group_by_root: dict[int, str] = {}
    for root, indices in components.items():
        content = sorted(
            {
                f"{exact_values[index]}:{decoded[index].hex()}"
                for index in indices
            }
        )
        group_by_root[root] = f"srcvis-{_object_digest(content)[:24]}"
    group_ids = tuple(group_by_root[dsu.find(index)] for index in range(len(decoded)))
    return ClusterResult(
        group_ids=group_ids,
        groups=len(components),
        exact_digest_unions=exact_digest_unions,
        exact_phash_unions=exact_phash_unions,
        near_phash_unions=near_phash_unions,
        near_pairs_examined=near_pairs_examined,
        maximum_hamming_fraction=float(maximum_hamming_fraction),
        maximum_group_size=max(len(indices) for indices in components.values()),
    )


def extract_task(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve(strict=True)
    data_root = args.data_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser()
    task_index = int(args.task_index)
    task_count = int(args.task_count)
    if not 1 <= task_count <= DEFAULT_TASK_COUNT:
        raise ValueError("task_count must be 1 or 2")
    if not 0 <= task_index < task_count:
        raise ValueError("task_index must be in [0, task_count)")
    if int(args.workers) < 1:
        raise ValueError("workers must be positive")
    config = R5FeatureConfig(
        analysis_frames=int(args.analysis_frames),
        resize_width=int(args.resize_width),
        active_speed_threshold=float(args.active_speed_threshold),
        camera_dims=DEFAULT_CAMERA_DIMS,
        perceptual_hash_frames=int(args.perceptual_hash_frames),
        instruction_dim=int(args.instruction_dim),
    )
    config.validate()
    rows, manifest_digest = _read_source_rows(input_path)
    selected = [row for index, row in enumerate(rows) if index % task_count == task_index]
    if not selected:
        raise ValueError(
            f"task {task_index}/{task_count} is empty; reduce task_count"
        )
    task_dir = output_dir / "tasks" / f"task-{task_index:03d}"
    root_contract = _root_contract(data_root)
    implementation_digest = _implementation_digest()
    provenance: dict[str, Any] = {
        "source_manifest": str(input_path),
        "source_manifest_sha256": manifest_digest,
        "source_rows": len(rows),
        "data_root_contract": root_contract,
        "data_root_digest": _object_digest(root_contract),
        "config": asdict(config),
        "config_digest": _feature_config_digest(config),
        "implementation_digest": implementation_digest,
        "task_index": task_index,
        "task_count": task_count,
        "partition": "input-index-modulo-task-count-v1",
    }
    if _ensure_fresh_or_committed(task_dir, resume=bool(args.resume)):
        committed = _load_committed_task(task_dir)
        if committed["metadata"]["provenance"] != provenance:
            raise ValueError(
                f"{task_dir} was committed with different extraction arguments"
            )
        print(f"[r5-features] task={task_index}/{task_count} already committed")
        return 0
    payloads = [
        {
            "row": row,
            "data_root": str(data_root),
            "config": asdict(config),
        }
        for row in selected
    ]
    if int(args.workers) == 1:
        results = [_extract_one(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            results = list(executor.map(_extract_one, payloads, chunksize=1))
    if _file_digest(input_path) != manifest_digest:
        raise RuntimeError("source manifest changed during feature extraction")
    done = _commit_task(
        task_dir,
        results=results,
        provenance=provenance,
    )
    print(
        f"[r5-features] committed task={task_index}/{task_count} "
        f"rows={done['rows']} output={task_dir}",
        flush=True,
    )
    return 0


def _final_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "archive": output_dir / "final.npz",
        "manifest": output_dir / "manifest.jsonl",
        "summary": output_dir / "summary.json",
        "done": output_dir / "done.json",
    }


def _validate_final(output_dir: Path) -> dict[str, Any]:
    paths = _final_paths(output_dir)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    done = _load_json(paths["done"])
    if done.get("schema_version") != R5_FINAL_SCHEMA:
        raise ValueError("final done schema mismatch")
    for name in ("archive", "manifest", "summary"):
        actual = _file_digest(paths[name])
        if done.get(f"{name}_sha256") != actual:
            raise ValueError(f"final {name} digest mismatch")
    arrays = _load_npz(paths["archive"])
    required = {
        "iids",
        "source_actor",
        "source_camera",
        "target_actor",
        "target_camera",
        "instruction_features",
        "action_signatures",
        "action_family",
        "label_role",
        "label_type",
        "eligible_positive",
        "production_eligible",
        "split",
        "content_group_id",
        "split_version",
        "source_perceptual_hash",
        "splits",
        "content_group_ids",
        "split_versions",
        "metadata_json",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"final archive is missing {missing}")
    if not np.array_equal(arrays["split"], arrays["splits"]):
        raise ValueError("singular/plural split aliases differ")
    if not np.array_equal(
        arrays["content_group_id"], arrays["content_group_ids"]
    ):
        raise ValueError("singular/plural content-group aliases differ")
    if not np.array_equal(arrays["split_version"], arrays["split_versions"]):
        raise ValueError("singular/plural split-version aliases differ")
    metadata_value = arrays["metadata_json"]
    if metadata_value.ndim != 0:
        raise ValueError("final metadata_json must be scalar")
    metadata = json.loads(str(metadata_value.item()))
    if metadata.get("schema_version") != R5_FINAL_SCHEMA:
        raise ValueError("final archive metadata schema mismatch")
    if metadata.get("endpoint_layout") != R5_ENDPOINT_LAYOUT:
        raise ValueError("final endpoint layout mismatch")
    if metadata.get("split_version") != R5_PHASH_SPLIT_VERSION:
        raise ValueError("final pHash split version mismatch")
    if metadata.get("production_eligible") is not False:
        raise ValueError("pHash-only final archive cannot be production eligible")
    for name in (
        "source_manifest_sha256",
        "data_root_digest",
        "config_digest",
        "implementation_digest",
        "cluster_config_digest",
    ):
        if metadata.get(name) != done.get(name):
            raise ValueError(f"final metadata/done {name} mismatch")
    if _object_digest(metadata.get("cluster_config")) != metadata.get(
        "cluster_config_digest"
    ):
        raise ValueError("final cluster-config digest mismatch")
    feature_config = R5FeatureConfig(**dict(metadata["feature_config"]))
    if _feature_config_digest(feature_config) != metadata.get("config_digest"):
        raise ValueError("final feature-config digest mismatch")
    rows = _load_jsonl(paths["manifest"])
    if len(rows) != len(arrays["iids"]) or not rows:
        raise ValueError("final manifest/archive row count mismatch")
    for index, row in enumerate(rows):
        if int(row.get("feature_index", -1)) != index:
            raise ValueError("final manifest feature_index mismatch")
        if str(row.get("iid")) != str(arrays["iids"][index]):
            raise ValueError("final manifest iid mismatch")
        for row_name, array_name in (
            ("split", "split"),
            ("content_group_id", "content_group_id"),
            ("split_version", "split_version"),
            ("source_perceptual_hash", "source_perceptual_hash"),
            ("action_family", "action_family"),
            ("label_role", "label_role"),
            ("label_type", "label_type"),
        ):
            if str(row.get(row_name)) != str(arrays[array_name][index]):
                raise ValueError(f"final manifest {row_name} mismatch")
        for row_name in ("eligible_positive", "production_eligible"):
            if bool(row.get(row_name)) != bool(arrays[row_name][index]):
                raise ValueError(f"final manifest {row_name} mismatch")
        if str(row.get("action_signature")) != str(
            arrays["action_signatures"][index]
        ):
            raise ValueError("final manifest action_signature mismatch")
    if bool(np.any(np.asarray(arrays["production_eligible"], dtype=np.bool_))):
        raise ValueError("pHash-only final rows cannot be production eligible")
    if int(metadata.get("rows", -1)) != len(rows):
        raise ValueError("final metadata row count mismatch")
    for name, expected in (
        ("actor_dim", np.asarray(arrays["source_actor"]).shape[1]),
        ("camera_dim", np.asarray(arrays["source_camera"]).shape[1]),
        (
            "instruction_dim",
            np.asarray(arrays["instruction_features"]).shape[1],
        ),
    ):
        if int(metadata.get(name, -1)) != int(expected):
            raise ValueError(f"final metadata {name} mismatch")
    rebuilt_row_contract = _object_digest(
        [
            (
                row["iid"],
                row["input_digest"],
                row["source_video_sha256"],
                row["target_video_sha256"],
                row["source_perceptual_hash"],
                row["content_group_id"],
                row["split"],
                row["label_role"],
                row["label_type"],
            )
            for row in rows
        ]
    )
    if metadata.get("row_contract_digest") != rebuilt_row_contract:
        raise ValueError("final row-contract digest mismatch")
    source_manifest = Path(str(metadata["source_manifest"]))
    if _file_digest(source_manifest) != metadata["source_manifest_sha256"]:
        raise ValueError("final source manifest changed")
    if _object_digest(metadata["data_root_contract"]) != metadata[
        "data_root_digest"
    ]:
        raise ValueError("final data-root digest mismatch")
    if _root_contract(
        Path(str(metadata["data_root_contract"]["resolved_path"]))
    ) != metadata["data_root_contract"]:
        raise ValueError("final data-root identity changed")
    if _implementation_digest() != metadata["implementation_digest"]:
        raise ValueError("final implementation changed after extraction")
    summary = _load_json(paths["summary"])
    for name in (
        "source_manifest_sha256",
        "data_root_digest",
        "config_digest",
        "implementation_digest",
        "cluster_config_digest",
    ):
        if summary.get(name) != metadata.get(name):
            raise ValueError(f"final summary/metadata {name} mismatch")
    if summary.get("cluster_config") != metadata.get("cluster_config"):
        raise ValueError("final summary/metadata cluster config mismatch")
    batch = R5EndpointBatch.from_mapping(
        arrays,
        require_visual_clusters=False,
        maximum_cross_split_hamming_fraction=float(
            done["maximum_hamming_fraction"]
        ),
    )
    if len(batch.iids) != len(rows):
        raise ValueError("R5EndpointBatch row count mismatch")
    audit_content_disjoint_splits(
        splits=[str(value) for value in arrays["split"]],
        content_group_ids=[str(value) for value in arrays["content_group_id"]],
        split_versions=[str(value) for value in arrays["split_version"]],
        perceptual_hashes=[
            str(value) for value in arrays["source_perceptual_hash"]
        ],
        maximum_cross_split_hamming_fraction=float(
            done["maximum_hamming_fraction"]
        ),
        require_visual_clusters=False,
    )
    return {
        "done": done,
        "summary": summary,
        "arrays": arrays,
        "rows": rows,
    }


def finalize_tasks(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser()
    task_count = int(args.task_count)
    if not 1 <= task_count <= DEFAULT_TASK_COUNT:
        raise ValueError("task_count must be 1 or 2")
    requested_cluster_config = {
        "algorithm": "global-dsu-exact-and-near-phash-v1",
        "maximum_hamming_fraction": float(args.maximum_hamming_fraction),
        "data_seed": int(args.data_seed),
        "train_fraction": float(args.train_fraction),
        "validation_fraction": float(args.validation_fraction),
        "minimum_positive_split_counts": {
            "train": int(getattr(args, "minimum_positive_train", 2)),
            "validation": int(
                getattr(args, "minimum_positive_validation", 1)
            ),
            "test": int(getattr(args, "minimum_positive_test", 1)),
        },
    }
    maximum_hamming_fraction = requested_cluster_config[
        "maximum_hamming_fraction"
    ]
    if not 0.0 <= maximum_hamming_fraction < 1.0:
        raise ValueError("maximum_hamming_fraction must be in [0,1)")
    train_fraction = requested_cluster_config["train_fraction"]
    validation_fraction = requested_cluster_config["validation_fraction"]
    if (
        not math.isfinite(train_fraction)
        or not math.isfinite(validation_fraction)
        or train_fraction <= 0.0
        or validation_fraction < 0.0
        or train_fraction + validation_fraction >= 1.0
    ):
        raise ValueError("invalid train/validation fractions")
    minimum_positive_split_counts = requested_cluster_config[
        "minimum_positive_split_counts"
    ]
    if any(value < 0 for value in minimum_positive_split_counts.values()):
        raise ValueError("minimum positive split counts must be non-negative")
    final_paths = _final_paths(output_dir)
    existing = [name for name, path in final_paths.items() if path.exists()]
    if existing:
        if bool(args.resume) and len(existing) == len(final_paths):
            validated = _validate_final(output_dir)
            if (
                validated["summary"].get("cluster_config")
                != requested_cluster_config
            ):
                raise ValueError(
                    "committed final archive uses different cluster/split arguments"
                )
            print(
                f"[r5-features-finalize] already committed rows="
                f"{len(validated['rows'])} output={output_dir}"
            )
            return 0
        raise RuntimeError(
            f"{output_dir} contains incomplete/conflicting final artifacts: {existing}"
        )
    tasks = [
        _load_committed_task(output_dir / "tasks" / f"task-{index:03d}")
        for index in range(task_count)
    ]
    reference_provenance = tasks[0]["metadata"]["provenance"]
    invariant_names = (
        "source_manifest",
        "source_manifest_sha256",
        "source_rows",
        "data_root_contract",
        "data_root_digest",
        "config",
        "config_digest",
        "implementation_digest",
        "task_count",
        "partition",
    )
    for expected_index, task in enumerate(tasks):
        provenance = task["metadata"]["provenance"]
        if int(provenance.get("task_index", -1)) != expected_index:
            raise ValueError(f"task {expected_index} provenance index mismatch")
        for name in invariant_names:
            if provenance.get(name) != reference_provenance.get(name):
                raise ValueError(f"task {expected_index} mixed provenance: {name}")
    if int(reference_provenance["task_count"]) != task_count:
        raise ValueError("requested task_count disagrees with extraction")

    combined: list[tuple[int, dict[str, Any], dict[str, np.ndarray], int]] = []
    for task_index, task in enumerate(tasks):
        for local_index, row in enumerate(task["rows"]):
            input_index = int(row["input_index"])
            if input_index % task_count != task_index:
                raise ValueError("task modulo partition provenance is invalid")
            combined.append((input_index, row, task["arrays"], local_index))
    combined.sort(key=lambda item: item[0])
    expected_rows = int(reference_provenance["source_rows"])
    if [item[0] for item in combined] != list(range(expected_rows)):
        raise ValueError("array tasks do not cover every source row exactly once")
    iids = [str(item[1]["iid"]) for item in combined]
    if len(set(iids)) != len(iids):
        raise ValueError("array tasks contain duplicate iids")

    def merged_matrix(name: str) -> np.ndarray:
        values = [
            np.asarray(item[2][name][item[3]], dtype=np.float32)
            for item in combined
        ]
        matrix = np.stack(values)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError(f"merged {name} is not a finite matrix")
        return matrix

    source_actor = merged_matrix("source_actor")
    source_camera = merged_matrix("source_camera")
    target_actor = merged_matrix("target_actor")
    target_camera = merged_matrix("target_camera")
    instruction_features = merged_matrix("instruction_features")
    if source_actor.shape != target_actor.shape:
        raise ValueError("merged actor endpoint shapes differ")
    if source_camera.shape != target_camera.shape:
        raise ValueError("merged camera endpoint shapes differ")
    perceptual_hashes = [
        str(item[1]["source_perceptual_hash"]) for item in combined
    ]
    exact_digests = [
        str(item[1]["source_sampled_frame_digest"]) for item in combined
    ]
    cluster = cluster_source_hashes(
        exact_digests=exact_digests,
        perceptual_hashes=perceptual_hashes,
        maximum_hamming_fraction=float(args.maximum_hamming_fraction),
    )
    splits = stable_splits_from_content_groups(
        cluster.group_ids,
        data_seed=int(args.data_seed),
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
    )
    split_versions = (R5_PHASH_SPLIT_VERSION,) * len(combined)
    audit = audit_content_disjoint_splits(
        splits=splits,
        content_group_ids=cluster.group_ids,
        split_versions=split_versions,
        perceptual_hashes=perceptual_hashes,
        maximum_cross_split_hamming_fraction=float(
            args.maximum_hamming_fraction
        ),
        require_visual_clusters=False,
    )
    action_signatures = [
        str(item[1]["action_signature"]) for item in combined
    ]
    action_families = [str(item[1]["action_family"]) for item in combined]
    label_roles = [str(item[1]["label_role"]) for item in combined]
    label_types = [str(item[1]["label_type"]) for item in combined]
    eligible_positive = [
        bool(item[1]["eligible_positive"]) for item in combined
    ]
    production_eligible = [
        bool(item[1]["production_eligible"]) for item in combined
    ]
    if any(production_eligible):
        raise ValueError("pHash-only pilot rows cannot be production eligible")
    positive_split_counts = {
        split_name: int(
            sum(
                role == "positive_delta" and split == split_name
                for role, split in zip(label_roles, splits)
            )
        )
        for split_name in ("train", "validation", "test")
    }
    insufficient_positive_splits = {
        split_name: {
            "actual": positive_split_counts[split_name],
            "minimum": minimum_positive_split_counts[split_name],
        }
        for split_name in ("train", "validation", "test")
        if positive_split_counts[split_name]
        < minimum_positive_split_counts[split_name]
    }
    if insufficient_positive_splits:
        raise ValueError(
            "R5 feature split lacks positive rows: "
            f"{insufficient_positive_splits}; choose a predeclared data seed "
            "or add content-disjoint positives"
        )
    final_rows: list[dict[str, Any]] = []
    for feature_index, ((_, source_row, _, _), group_id, split) in enumerate(
        zip(combined, cluster.group_ids, splits)
    ):
        final_rows.append(
            {
                **source_row,
                "feature_index": feature_index,
                "content_group_id": group_id,
                "split": split,
                "split_version": R5_PHASH_SPLIT_VERSION,
            }
        )
    cluster_config = requested_cluster_config
    metadata = {
        "schema_version": R5_FINAL_SCHEMA,
        "r5_batch_schema_version": R5_SCHEMA_VERSION,
        "endpoint_layout": R5_ENDPOINT_LAYOUT,
        "actor_dim": int(source_actor.shape[1]),
        "camera_dim": int(source_camera.shape[1]),
        "instruction_dim": int(instruction_features.shape[1]),
        "camera_tail_dims": DEFAULT_CAMERA_DIMS,
        "rows": len(final_rows),
        "source_manifest": reference_provenance["source_manifest"],
        "source_manifest_sha256": reference_provenance[
            "source_manifest_sha256"
        ],
        "data_root_contract": reference_provenance["data_root_contract"],
        "data_root_digest": reference_provenance["data_root_digest"],
        "feature_config": reference_provenance["config"],
        "config_digest": reference_provenance["config_digest"],
        "implementation_digest": reference_provenance[
            "implementation_digest"
        ],
        "prompt_hash_version": PROMPT_HASH_VERSION,
        "cluster_config": cluster_config,
        "cluster_config_digest": _object_digest(cluster_config),
        "split_version": R5_PHASH_SPLIT_VERSION,
        "production_eligible": False,
        "row_contract_digest": _object_digest(
            [
                (
                    row["iid"],
                    row["input_digest"],
                    row["source_video_sha256"],
                    row["target_video_sha256"],
                    row["source_perceptual_hash"],
                    row["content_group_id"],
                    row["split"],
                    row["label_role"],
                    row["label_type"],
                )
                for row in final_rows
            ]
        ),
    }
    arrays: dict[str, Any] = {
        "iids": np.asarray(iids, dtype=str),
        "source_actor": source_actor,
        "source_camera": source_camera,
        "target_actor": target_actor,
        "target_camera": target_camera,
        "instruction_features": instruction_features,
        "action_signatures": np.asarray(action_signatures, dtype=str),
        "action_family": np.asarray(action_families, dtype=str),
        "label_role": np.asarray(label_roles, dtype=str),
        "label_type": np.asarray(label_types, dtype=str),
        "eligible_positive": np.asarray(eligible_positive, dtype=np.bool_),
        "production_eligible": np.asarray(
            production_eligible,
            dtype=np.bool_,
        ),
        "split": np.asarray(splits, dtype=str),
        "content_group_id": np.asarray(cluster.group_ids, dtype=str),
        "split_version": np.asarray(split_versions, dtype=str),
        "source_perceptual_hash": np.asarray(perceptual_hashes, dtype=str),
        # Direct R5EndpointBatch aliases.
        "splits": np.asarray(splits, dtype=str),
        "content_group_ids": np.asarray(cluster.group_ids, dtype=str),
        "split_versions": np.asarray(split_versions, dtype=str),
        "metadata_json": np.asarray(_canonical_json(metadata)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(final_paths["archive"], arrays)
    _atomic_jsonl(final_paths["manifest"], final_rows)
    split_counts = {
        name: int(sum(value == name for value in splits))
        for name in ("train", "validation", "test")
    }
    summary = {
        "schema_version": R5_FINAL_SCHEMA,
        "stage": "finalize",
        "rows": len(final_rows),
        "actor_dim": int(source_actor.shape[1]),
        "camera_dim": int(source_camera.shape[1]),
        "instruction_dim": int(instruction_features.shape[1]),
        "endpoint_layout": R5_ENDPOINT_LAYOUT,
        "splits": split_counts,
        "positive_splits": positive_split_counts,
        "label_roles": {
            role: int(sum(value == role for value in label_roles))
            for role in sorted(set(label_roles))
        },
        "label_types": {
            label_type: int(sum(value == label_type for value in label_types))
            for label_type in sorted(set(label_types))
        },
        "cluster": cluster.to_dict(),
        "content_split_audit": audit.to_dict(),
        "source_manifest": metadata["source_manifest"],
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "data_root_contract": metadata["data_root_contract"],
        "data_root_digest": metadata["data_root_digest"],
        "feature_config": metadata["feature_config"],
        "config_digest": metadata["config_digest"],
        "implementation_digest": metadata["implementation_digest"],
        "cluster_config": cluster_config,
        "cluster_config_digest": metadata["cluster_config_digest"],
        "archive_sha256": _file_digest(final_paths["archive"]),
        "manifest_sha256": _file_digest(final_paths["manifest"]),
    }
    _atomic_json(final_paths["summary"], summary)
    done = {
        "schema_version": R5_FINAL_SCHEMA,
        "rows": len(final_rows),
        "archive_sha256": summary["archive_sha256"],
        "manifest_sha256": summary["manifest_sha256"],
        "summary_sha256": _file_digest(final_paths["summary"]),
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "data_root_digest": metadata["data_root_digest"],
        "config_digest": metadata["config_digest"],
        "implementation_digest": metadata["implementation_digest"],
        "cluster_config_digest": metadata["cluster_config_digest"],
        "maximum_hamming_fraction": float(args.maximum_hamming_fraction),
    }
    _atomic_json(final_paths["done"], done)
    _validate_final(output_dir)
    print(
        f"[r5-features-finalize] committed rows={len(final_rows)} "
        f"groups={cluster.groups} splits={split_counts} output={output_dir}",
        flush=True,
    )
    return 0


def validate(args: argparse.Namespace) -> int:
    result = _validate_final(args.output_dir.expanduser())
    print(
        f"[r5-features-validate] rows={len(result['rows'])} "
        f"archive={args.output_dir.expanduser() / 'final.npz'}",
        flush=True,
    )
    return 0


def _add_common_feature_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--analysis-frames", type=int, default=20)
    parser.add_argument("--resize-width", type=int, default=160)
    parser.add_argument("--active-speed-threshold", type=float, default=0.005)
    parser.add_argument("--perceptual-hash-frames", type=int, default=6)
    parser.add_argument("--instruction-dim", type=int, default=512)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and globally cluster source-aware R5 endpoints."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--input", required=True, type=Path)
    extract_parser.add_argument("--data-root", required=True, type=Path)
    extract_parser.add_argument("--output-dir", required=True, type=Path)
    extract_parser.add_argument("--task-index", required=True, type=int)
    extract_parser.add_argument(
        "--task-count",
        type=int,
        default=DEFAULT_TASK_COUNT,
    )
    extract_parser.add_argument("--workers", type=int, default=8)
    extract_parser.add_argument("--resume", action="store_true")
    _add_common_feature_arguments(extract_parser)
    extract_parser.set_defaults(handler=extract_task)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-dir", required=True, type=Path)
    finalize_parser.add_argument(
        "--task-count",
        type=int,
        default=DEFAULT_TASK_COUNT,
    )
    finalize_parser.add_argument("--data-seed", type=int, default=260108828)
    finalize_parser.add_argument("--train-fraction", type=float, default=0.8)
    finalize_parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )
    finalize_parser.add_argument(
        "--maximum-hamming-fraction",
        type=float,
        default=0.10,
    )
    finalize_parser.add_argument("--minimum-positive-train", type=int, default=2)
    finalize_parser.add_argument(
        "--minimum-positive-validation",
        type=int,
        default=1,
    )
    finalize_parser.add_argument("--minimum-positive-test", type=int, default=1)
    finalize_parser.add_argument("--resume", action="store_true")
    finalize_parser.set_defaults(handler=finalize_tasks)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-dir", required=True, type=Path)
    validate_parser.set_defaults(handler=validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
