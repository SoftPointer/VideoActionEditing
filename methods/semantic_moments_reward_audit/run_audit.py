#!/usr/bin/env python3
"""Audit SemanticMoments as a reward signal for dual-anchor action editing.

The script deliberately separates feature extraction from analysis so an AUH
node can extract independent shards on several GPUs without sharing model
state.  It consumes the official SemanticMoments ``Embedder.compute_moments``
implementation for a parity check, while using a locally pinned Hugging Face
DINOv2 checkpoint to avoid network-dependent model loading.

This is an evaluator audit.  It does not select training data, update a video
model, or authorize an action-editing reward.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = "semantic-moments-action-reward-audit-v1"
FEATURE_SCHEMA_VERSION = "semantic-moments-action-reward-features-v1"
DEFAULT_WEIGHTS = {
    "m1": (1.0, 0.0, 0.0),
    "m2": (0.0, 1.0, 0.0),
    "m3": (0.0, 0.0, 1.0),
    "m23": (0.0, 8.0, 4.0),
    "m123": (1.0, 8.0, 4.0),
}
PROJECT_BRANCHES = ("forward", "reverse", "noop")
EPS = 1.0e-8


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unit(value: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(value.float(), dim=dim, eps=EPS)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"cosine shape mismatch: {left.shape} != {right.shape}")
    return float(F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0))


def temporal_components(video_embedding: torch.Tensor) -> torch.Tensor:
    """Return the three normalized, spatially pooled official moment blocks.

    ``video_embedding`` is ``[T, P, D]``.  The use of unbiased ``torch.std``
    intentionally matches official commit eb4ec98 rather than silently
    substituting the population standard deviation.
    """
    if video_embedding.ndim != 3 or video_embedding.shape[0] < 2:
        raise ValueError("expected at least two temporal samples in [T,P,D]")
    values = video_embedding.float()
    mean = values.mean(dim=0)
    std = values.std(dim=0)
    centered = values - mean
    skew = (centered**3).mean(dim=0) / (std**3 + 1.0e-6)
    return torch.stack(
        [
            unit(mean.mean(dim=0), dim=0),
            unit(std.mean(dim=0), dim=0),
            unit(skew.mean(dim=0), dim=0),
        ],
        dim=0,
    )


def compose_moments(
    components: torch.Tensor, weights: Sequence[float]
) -> torch.Tensor:
    if components.ndim != 2 or components.shape[0] != 3 or len(weights) != 3:
        raise ValueError("expected three moment components and three weights")
    weight_tensor = torch.tensor(weights, dtype=torch.float32).view(3, 1)
    return unit((components.float() * weight_tensor).flatten(), dim=0)


def load_official_embedder_class(semantic_moments_root: str | Path) -> type:
    base_path = Path(semantic_moments_root) / "src/semantic_moments/embedders/base.py"
    if not base_path.is_file():
        raise FileNotFoundError(f"official SemanticMoments base.py is absent: {base_path}")
    spec = importlib.util.spec_from_file_location("semantic_moments_official_base", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official SemanticMoments base module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Embedder


def uniform_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        raise ValueError("video frame count and requested count must be positive")
    return [int(round(value)) for value in np.linspace(0, total - 1, count)]


def load_video_frames(path: str | Path, count: int) -> list[Any]:
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise ValueError(f"video has no frames: {path}")
    targets = uniform_indices(total, count)
    positions: dict[int, list[int]] = {}
    for output_index, frame_index in enumerate(targets):
        positions.setdefault(frame_index, []).append(output_index)
    frames: list[Any | None] = [None] * count
    last_target = targets[-1]
    frame_index = 0
    while frame_index <= last_target:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"cannot decode frame {frame_index} from {path}")
        if frame_index in positions:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for output_index in positions[frame_index]:
                frames[output_index] = image.copy()
        frame_index += 1
    capture.release()
    if any(frame is None for frame in frames):
        raise RuntimeError(f"sequential decode did not fill every sample for {path}")
    return list(frames)


class LocalDINOv2:
    """Per-frame DINOv2 patch extractor backed by a local checkpoint."""

    def __init__(self, model_root: str | Path, device: str, frame_batch_size: int):
        from transformers import AutoImageProcessor, AutoModel

        self.model_root = str(Path(model_root).resolve())
        self.device = torch.device(device)
        self.frame_batch_size = frame_batch_size
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_root, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_root, local_files_only=True
        ).to(self.device).eval()
        self.num_register_tokens = int(
            getattr(self.model.config, "num_register_tokens", 0) or 0
        )

    @torch.inference_mode()
    def extract(self, frames: Sequence[Any]) -> torch.Tensor:
        outputs = []
        for start in range(0, len(frames), self.frame_batch_size):
            batch = frames[start : start + self.frame_batch_size]
            inputs = self.processor(images=list(batch), return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device)
            hidden = self.model(pixel_values=pixels).last_hidden_state
            patch_start = 1 + self.num_register_tokens
            outputs.append(hidden[:, patch_start:].detach().float().cpu())
        result = torch.cat(outputs, dim=0)
        if result.shape[0] != len(frames) or result.ndim != 3:
            raise RuntimeError(f"unexpected DINO feature geometry: {result.shape}")
        return result


def dtw_cost(left: torch.Tensor, right: torch.Tensor) -> float:
    """Monotonic DTW over appearance-centered frame descriptors."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("DTW expects [T,D] sequences with a shared D")
    left = unit(left.float() - left.float().mean(dim=0, keepdim=True), dim=1)
    right = unit(right.float() - right.float().mean(dim=0, keepdim=True), dim=1)
    local = (1.0 - left @ right.T).clamp(min=0.0, max=2.0).numpy()
    rows, columns = local.shape
    table = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    steps = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    table[0, 0] = 0.0
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            candidates = (
                (table[i - 1, j - 1], steps[i - 1, j - 1]),
                (table[i - 1, j], steps[i - 1, j]),
                (table[i, j - 1], steps[i, j - 1]),
            )
            previous_cost, previous_steps = min(candidates, key=lambda item: item[0])
            table[i, j] = previous_cost + float(local[i - 1, j - 1])
            steps[i, j] = previous_steps + 1
    return float(table[rows, columns] / max(int(steps[rows, columns]), 1))


def order_margin(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Positive when candidate aligns better forward than after reversal."""
    forward = dtw_cost(reference, candidate)
    reverse = dtw_cost(reference, torch.flip(candidate, dims=(0,)))
    return float((reverse - forward) / (reverse + forward + EPS))


def endpoint_score(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    left = reference[-1] - reference[0]
    right = candidate[-1] - candidate[0]
    return cosine(left, right)


def deterministic_permutation(identifier: str, count: int) -> torch.Tensor:
    seed = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16], 16)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(count, generator=generator)


def variant_indices(identifier: str, count: int) -> dict[str, torch.Tensor]:
    natural = torch.arange(count)
    half = count // 2
    block = max(1, count // 4)
    blocks = [natural[i : min(i + block, count)] for i in range(0, count, block)]
    block_order = list(range(len(blocks)))
    random.Random(int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)).shuffle(
        block_order
    )
    first = natural[:half]
    last = natural[-half:]
    tail_hold = torch.cat([first, first[-1:].repeat(count - len(first))])
    return {
        "reverse": torch.flip(natural, dims=(0,)),
        "random_shuffle": deterministic_permutation(identifier, count),
        "block_shuffle": torch.cat([blocks[index] for index in block_order])[:count],
        "first_half_repeat": torch.cat([first, first])[:count],
        "last_half_repeat": torch.cat([last, last])[:count],
        "tail_hold": tail_hold[:count],
    }


def build_manifest(args: argparse.Namespace) -> int:
    items: list[dict[str, Any]] = []
    simmotion_root = Path(args.simmotion_real_root)
    examples_root = simmotion_root / "examples"
    if not examples_root.is_dir():
        raise FileNotFoundError(f"SimMotion-Real examples are absent: {examples_root}")
    for example in sorted(path for path in examples_root.iterdir() if path.is_dir()):
        for role in ("ref", "positive", "negative"):
            path = example / f"{role}.mp4"
            if not path.is_file():
                raise FileNotFoundError(path)
            items.append(
                {
                    "item_id": f"simmotion:{example.name}:{role}",
                    "group": "simmotion_real",
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "metadata": {"example_id": example.name, "role": role},
                }
            )

    project_receipt = load_json(args.project_bank_receipt)
    project_rows = project_receipt.get("candidate_rows")
    if not isinstance(project_rows, list) or len(project_rows) != 60:
        raise ValueError("project bank receipt must contain the exact 60 candidate rows")
    for row in project_rows:
        path = Path(row["video_path"])
        expected = row["video_sha256"]
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"project video hash differs: {path}")
        metadata = {
            key: row[key]
            for key in (
                "candidate_id",
                "iid",
                "seed",
                "branch",
                "actor_family",
                "action_family_id",
                "analysis_split",
            )
        }
        items.append(
            {
                "item_id": f"project:{row['candidate_id']}",
                "group": "project_saic_bank",
                "path": str(path.resolve()),
                "sha256": expected,
                "metadata": metadata,
            }
        )

    for probe in args.probe_video:
        name, separator, raw_path = probe.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--probe-video must use NAME=/absolute/path.mp4")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        items.append(
            {
                "item_id": f"probe:{name}",
                "group": "project_probe",
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "metadata": {"name": name},
            }
        )

    identifiers = [row["item_id"] for row in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("manifest item identifiers are not unique")
    groups = Counter(row["group"] for row in items)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "authority": "evaluator_audit_only",
        "sources": {
            "simmotion_real_root": str(simmotion_root.resolve()),
            "project_bank_receipt": str(Path(args.project_bank_receipt).resolve()),
            "project_bank_receipt_sha256": file_sha256(args.project_bank_receipt),
        },
        "counts": {"total": len(items), "by_group": dict(sorted(groups.items()))},
        "items": items,
    }
    manifest["manifest_digest"] = object_sha256(manifest)
    write_json(args.output, manifest)
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


@dataclass
class ExtractionContext:
    official_embedder: Any
    extractor: LocalDINOv2
    num_frames: int


def extract_one(item: Mapping[str, Any], context: ExtractionContext) -> dict[str, Any]:
    path = Path(item["path"])
    if file_sha256(path) != item["sha256"]:
        raise ValueError(f"input video changed: {path}")
    frames = load_video_frames(path, context.num_frames)
    tokens = context.extractor.extract(frames)
    components = temporal_components(tokens)
    official = context.official_embedder.compute_moments(tokens)
    local_default = compose_moments(components, DEFAULT_WEIGHTS["m123"])
    parity_max_abs = float(torch.max(torch.abs(official.cpu() - local_default.cpu())))
    if parity_max_abs > 2.0e-6:
        raise RuntimeError(f"official formula parity failed: max_abs={parity_max_abs}")

    base_embedding = local_default
    frame_sequence = unit(tokens.mean(dim=1), dim=1)
    variants = {}
    for name, indices in variant_indices(item["item_id"], tokens.shape[0]).items():
        selected_tokens = tokens[indices]
        selected_sequence = frame_sequence[indices]
        embedding = compose_moments(
            temporal_components(selected_tokens), DEFAULT_WEIGHTS["m123"]
        )
        variants[name] = {
            "semantic_moments_cosine": cosine(base_embedding, embedding),
            "semantic_moments_max_abs": float(
                torch.max(torch.abs(base_embedding - embedding))
            ),
            "dtw_order_margin": order_margin(frame_sequence, selected_sequence),
            "endpoint_score": endpoint_score(frame_sequence, selected_sequence),
        }

    return {
        "item_id": item["item_id"],
        "group": item["group"],
        "path": item["path"],
        "sha256": item["sha256"],
        "metadata": dict(item["metadata"]),
        "feature_geometry": {
            "num_frames": int(tokens.shape[0]),
            "num_patches": int(tokens.shape[1]),
            "dimension": int(tokens.shape[2]),
        },
        "components": components.cpu(),
        "frame_sequence": frame_sequence.cpu(),
        "official_default_parity_max_abs": parity_max_abs,
        "variants": variants,
    }


def runtime_identity(model_root: Path, semantic_moments_root: Path) -> dict[str, Any]:
    try:
        rocm_devices = subprocess.run(
            ["/opt/rocm/bin/rocm-smi", "--showuniqueid"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        rocm_devices = ""
    vendor_commit = subprocess.run(
        ["git", "-C", str(semantic_moments_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    model_files = sorted(path for path in model_root.rglob("*") if path.is_file())
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "cuda_api_available": torch.cuda.is_available(),
        "visible_device_count": torch.cuda.device_count(),
        "visible_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "semantic_moments_root": str(semantic_moments_root),
        "semantic_moments_commit": vendor_commit,
        "semantic_moments_base_sha256": file_sha256(
            semantic_moments_root / "src/semantic_moments/embedders/base.py"
        ),
        "model_root": str(model_root),
        "model_files": [
            {"relative_path": str(path.relative_to(model_root)), "sha256": file_sha256(path)}
            for path in model_files
        ],
        "rocm_smi_unique_ids": rocm_devices.strip().splitlines(),
    }


def extract_shard(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("manifest schema differs")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard index is outside [0, num_shards)")
    selected = [
        item
        for ordinal, item in enumerate(manifest["items"])
        if ordinal % args.num_shards == args.shard_index
    ]
    semantic_root = Path(args.semantic_moments_root).resolve()
    model_root = Path(args.model_root).resolve()
    official_class = load_official_embedder_class(semantic_root)
    context = ExtractionContext(
        official_embedder=official_class(
            alpha1=1.0, alpha2=8.0, alpha3=4.0, aggregation="concat"
        ),
        extractor=LocalDINOv2(model_root, args.device, args.frame_batch_size),
        num_frames=args.num_frames,
    )
    records = []
    for ordinal, item in enumerate(selected):
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "ordinal": ordinal,
                    "count": len(selected),
                    "item_id": item["item_id"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        records.append(extract_one(item, context))
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_digest": manifest["manifest_digest"],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_frames": args.num_frames,
        "record_count": len(records),
        "runtime": runtime_identity(model_root, semantic_root),
        "records": records,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(destination)
    print(f"wrote {destination} records={len(records)}")
    return 0


def bootstrap_interval(
    values: Sequence[float], statistic: str, seed: int = 20260815, draws: int = 4000
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    samples = rng.choice(array, size=(draws, len(array)), replace=True)
    if statistic == "mean":
        estimates = samples.mean(axis=1)
    else:
        raise ValueError(statistic)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def binary_preference_summary(margins: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in margins]
    wins = sum(value > 0 for value in values)
    ties = sum(abs(value) <= 1.0e-8 for value in values)
    return {
        "count": len(values),
        "wins": wins,
        "ties": ties,
        "accuracy": wins / len(values) if values else 0.0,
        "accuracy_bootstrap_95pct": bootstrap_interval(
            [float(value > 0) for value in values], "mean"
        ),
        "mean_margin": float(np.mean(values)) if values else 0.0,
        "median_margin": float(np.median(values)) if values else 0.0,
        "minimum_margin": min(values) if values else 0.0,
        "maximum_margin": max(values) if values else 0.0,
    }


def representation(record: Mapping[str, Any], name: str) -> torch.Tensor:
    return compose_moments(record["components"], DEFAULT_WEIGHTS[name])


def group_by_metadata(
    records: Iterable[Mapping[str, Any]], key: str
) -> dict[Any, list[Mapping[str, Any]]]:
    output: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        output[row["metadata"][key]].append(row)
    return dict(output)


def simmotion_analysis(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_example = group_by_metadata(records, "example_id")
    output: dict[str, Any] = {
        "example_count": len(by_example),
        "video_count": len(records),
        "protocol": {
            "pairwise": "positive cosine must exceed the paired negative cosine",
            "within_dataset_recall_at_1": (
                "official SimMotionReal.evaluate rule without optional external "
                "distractors: positive must beat the paired negative and all 117 "
                "videos from the other 39 triplets"
            ),
            "external_kinetics_distractors_used": False,
        },
        "representations": {},
    }
    for name in DEFAULT_WEIGHTS:
        margins = []
        rows = []
        retrieval_margins = []
        retrieval_rows = []
        for example_id, example_rows in sorted(by_example.items()):
            by_role = {row["metadata"]["role"]: row for row in example_rows}
            reference = representation(by_role["ref"], name)
            positive = cosine(reference, representation(by_role["positive"], name))
            negative = cosine(reference, representation(by_role["negative"], name))
            margin = positive - negative
            margins.append(margin)
            rows.append(
                {
                    "example_id": example_id,
                    "positive_cosine": positive,
                    "negative_cosine": negative,
                    "margin": margin,
                    "correct": margin > 0,
                }
            )

            other_candidates = [
                row
                for row in records
                if row["metadata"]["example_id"] != example_id
            ]
            if len(other_candidates) != len(records) - 3:
                raise ValueError(f"unexpected SimMotion candidate count: {example_id}")
            other_scores = [
                (
                    cosine(reference, representation(row, name)),
                    row["item_id"],
                )
                for row in other_candidates
            ]
            max_other, max_other_id = max(other_scores)
            hardest_distractor = max(negative, max_other)
            retrieval_margin = positive - hardest_distractor
            retrieval_margins.append(retrieval_margin)
            retrieval_rows.append(
                {
                    "example_id": example_id,
                    "positive_cosine": positive,
                    "negative_cosine": negative,
                    "max_other_cosine": max_other,
                    "max_other_item_id": max_other_id,
                    "margin": retrieval_margin,
                    "correct": retrieval_margin > 0,
                }
            )
        output["representations"][name] = {
            "pairwise_positive_over_negative": {
                **binary_preference_summary(margins),
                "rows": rows,
            },
            "within_dataset_recall_at_1": {
                **binary_preference_summary(retrieval_margins),
                "candidate_count_per_query": len(records) - 1,
                "other_triplet_video_count": len(records) - 3,
                "rows": retrieval_rows,
            },
        }

    dtw_margins = []
    endpoint_margins = []
    for _, example_rows in sorted(by_example.items()):
        by_role = {row["metadata"]["role"]: row for row in example_rows}
        reference = by_role["ref"]["frame_sequence"]
        dtw_margins.append(
            order_margin(reference, by_role["positive"]["frame_sequence"])
            - order_margin(reference, by_role["negative"]["frame_sequence"])
        )
        endpoint_margins.append(
            endpoint_score(reference, by_role["positive"]["frame_sequence"])
            - endpoint_score(reference, by_role["negative"]["frame_sequence"])
        )
    output["order_diagnostics"] = {
        "dtw_margin_positive_over_negative": binary_preference_summary(dtw_margins),
        "endpoint_positive_over_negative": binary_preference_summary(endpoint_margins),
    }
    return output


def nearest_branch_rate(
    rows: Sequence[Mapping[str, Any]], name: str, candidate_policy: str
) -> dict[str, Any]:
    decisions = []
    for query in rows:
        meta = query["metadata"]
        if candidate_policy == "same_iid_other_seed":
            candidates = [
                row
                for row in rows
                if row["metadata"]["iid"] == meta["iid"]
                and row["metadata"]["seed"] != meta["seed"]
            ]
        elif candidate_policy == "same_family_other_iid":
            candidates = [
                row
                for row in rows
                if row["metadata"]["actor_family"] == meta["actor_family"]
                and row["metadata"]["iid"] != meta["iid"]
            ]
        else:
            raise ValueError(candidate_policy)
        if not candidates:
            continue
        query_embedding = representation(query, name)
        best = max(
            candidates, key=lambda row: cosine(query_embedding, representation(row, name))
        )
        decisions.append(
            {
                "query": meta["candidate_id"],
                "neighbor": best["metadata"]["candidate_id"],
                "query_branch": meta["branch"],
                "neighbor_branch": best["metadata"]["branch"],
                "correct": meta["branch"] == best["metadata"]["branch"],
                "cosine": cosine(query_embedding, representation(best, name)),
            }
        )
    return {
        "policy": candidate_policy,
        "count": len(decisions),
        "correct": sum(row["correct"] for row in decisions),
        "accuracy": (
            sum(row["correct"] for row in decisions) / len(decisions)
            if decisions
            else 0.0
        ),
        "decisions": decisions,
    }


def global_nearest_neighbor_audit(
    rows: Sequence[Mapping[str, Any]], name: str, exclude_same_seed: bool
) -> dict[str, Any]:
    decisions = []
    pool_priors: dict[str, list[float]] = {
        "same_iid": [],
        "same_branch": [],
        "same_actor_family": [],
        "same_action_family": [],
    }
    for query in rows:
        meta = query["metadata"]
        candidates = [
            row
            for row in rows
            if row["metadata"]["candidate_id"] != meta["candidate_id"]
            and (
                not exclude_same_seed
                or row["metadata"]["seed"] != meta["seed"]
            )
        ]
        if not candidates:
            continue
        query_embedding = representation(query, name)
        best = max(
            candidates, key=lambda row: cosine(query_embedding, representation(row, name))
        )
        best_meta = best["metadata"]
        attributes = {
            "same_iid": best_meta["iid"] == meta["iid"],
            "same_branch": best_meta["branch"] == meta["branch"],
            "same_actor_family": best_meta["actor_family"] == meta["actor_family"],
            "same_action_family": (
                best_meta["action_family_id"] == meta["action_family_id"]
            ),
        }
        for key in pool_priors:
            if key == "same_action_family":
                metadata_key = "action_family_id"
            else:
                metadata_key = key.removeprefix("same_")
            pool_priors[key].append(
                sum(
                    row["metadata"][metadata_key] == meta[metadata_key]
                    for row in candidates
                )
                / len(candidates)
            )
        decisions.append(
            {
                "query": meta["candidate_id"],
                "neighbor": best_meta["candidate_id"],
                "cosine": cosine(query_embedding, representation(best, name)),
                **attributes,
            }
        )
    rates = {
        key: sum(row[key] for row in decisions) / len(decisions)
        for key in pool_priors
    }
    return {
        "exclude_same_seed": exclude_same_seed,
        "count": len(decisions),
        "rates": rates,
        "mean_candidate_pool_priors": {
            key: float(np.mean(values)) for key, values in pool_priors.items()
        },
        "lift_over_pool_prior": {
            key: rates[key] / max(float(np.mean(pool_priors[key])), EPS)
            for key in pool_priors
        },
        "decisions": decisions,
    }


def centroid_branch_rate(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    fit = [row for row in rows if row["metadata"]["analysis_split"] == "fit"]
    confirmation = [
        row for row in rows if row["metadata"]["analysis_split"] == "confirmation"
    ]
    centroids = {}
    for family in sorted({row["metadata"]["actor_family"] for row in rows}):
        for branch in PROJECT_BRANCHES:
            members = [
                representation(row, name)
                for row in fit
                if row["metadata"]["actor_family"] == family
                and row["metadata"]["branch"] == branch
            ]
            if not members:
                raise ValueError(f"empty centroid: {family}:{branch}")
            centroids[(family, branch)] = unit(torch.stack(members).mean(dim=0), dim=0)
    decisions = []
    for row in confirmation:
        meta = row["metadata"]
        embedding = representation(row, name)
        scores = {
            branch: cosine(embedding, centroids[(meta["actor_family"], branch)])
            for branch in PROJECT_BRANCHES
        }
        predicted = max(PROJECT_BRANCHES, key=lambda branch: scores[branch])
        decisions.append(
            {
                "candidate_id": meta["candidate_id"],
                "truth": meta["branch"],
                "predicted": predicted,
                "correct": predicted == meta["branch"],
                "scores": scores,
            }
        )
    confusion = {branch: Counter() for branch in PROJECT_BRANCHES}
    for row in decisions:
        confusion[row["truth"]][row["predicted"]] += 1
    return {
        "fit_count": len(fit),
        "confirmation_count": len(confirmation),
        "correct": sum(row["correct"] for row in decisions),
        "accuracy": sum(row["correct"] for row in decisions) / len(decisions),
        "confusion": {
            truth: {predicted: confusion[truth][predicted] for predicted in PROJECT_BRANCHES}
            for truth in PROJECT_BRANCHES
        },
        "decisions": decisions,
    }


def forward_preference(
    rows: Sequence[Mapping[str, Any]], name: str | None, order: bool = False
) -> dict[str, Any]:
    margins_by_negative: dict[str, list[float]] = {"reverse": [], "noop": []}
    detail = []
    for query in rows:
        meta = query["metadata"]
        if meta["branch"] != "forward":
            continue
        siblings = [
            row
            for row in rows
            if row["metadata"]["iid"] == meta["iid"]
            and row["metadata"]["seed"] != meta["seed"]
        ]
        positives = [row for row in siblings if row["metadata"]["branch"] == "forward"]
        if not positives:
            continue
        for negative_branch in ("reverse", "noop"):
            negatives = [
                row
                for row in siblings
                if row["metadata"]["branch"] == negative_branch
            ]
            if not negatives:
                continue
            if order:
                positive_score = max(
                    order_margin(query["frame_sequence"], row["frame_sequence"])
                    for row in positives
                )
                negative_score = max(
                    order_margin(query["frame_sequence"], row["frame_sequence"])
                    for row in negatives
                )
            else:
                assert name is not None
                query_embedding = representation(query, name)
                positive_score = max(
                    cosine(query_embedding, representation(row, name))
                    for row in positives
                )
                negative_score = max(
                    cosine(query_embedding, representation(row, name))
                    for row in negatives
                )
            margin = positive_score - negative_score
            margins_by_negative[negative_branch].append(margin)
            detail.append(
                {
                    "query": meta["candidate_id"],
                    "negative_branch": negative_branch,
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "margin": margin,
                    "correct": margin > 0,
                }
            )
    return {
        "against_reverse": binary_preference_summary(margins_by_negative["reverse"]),
        "against_noop": binary_preference_summary(margins_by_negative["noop"]),
        "decisions": detail,
    }


def project_analysis(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["metadata"]["branch"] for row in records)
    output: dict[str, Any] = {
        "candidate_count": len(records),
        "branch_counts": dict(sorted(counts.items())),
        "labels_are_generation_contracts_not_human_event_verdicts": True,
        "representations": {},
    }
    for name in DEFAULT_WEIGHTS:
        output["representations"][name] = {
            "global_top1": global_nearest_neighbor_audit(
                records, name, exclude_same_seed=False
            ),
            "global_other_seed_top1": global_nearest_neighbor_audit(
                records, name, exclude_same_seed=True
            ),
            "same_iid_other_seed_top1_branch": nearest_branch_rate(
                records, name, "same_iid_other_seed"
            ),
            "same_family_other_iid_top1_branch": nearest_branch_rate(
                records, name, "same_family_other_iid"
            ),
            "fit_centroid_confirmation_branch": centroid_branch_rate(records, name),
            "forward_anchor_preference": forward_preference(records, name),
        }
    output["order_diagnostic"] = {
        "forward_anchor_preference": forward_preference(records, None, order=True)
    }
    return output


def variant_analysis(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(next(iter(records))["variants"])
    output = {}
    for name in names:
        moment_values = [
            float(row["variants"][name]["semantic_moments_cosine"]) for row in records
        ]
        max_abs_values = [
            float(row["variants"][name]["semantic_moments_max_abs"]) for row in records
        ]
        dtw_values = [float(row["variants"][name]["dtw_order_margin"]) for row in records]
        endpoint_values = [float(row["variants"][name]["endpoint_score"]) for row in records]
        output[name] = {
            "count": len(records),
            "semantic_moments_cosine": {
                "mean": float(np.mean(moment_values)),
                "minimum": min(moment_values),
                "maximum": max(moment_values),
            },
            "semantic_moments_max_abs_maximum": max(max_abs_values),
            "dtw_order_margin": {
                "mean": float(np.mean(dtw_values)),
                "minimum": min(dtw_values),
                "maximum": max(dtw_values),
                "positive_rate": sum(value > 0 for value in dtw_values) / len(dtw_values),
            },
            "endpoint_score": {
                "mean": float(np.mean(endpoint_values)),
                "minimum": min(endpoint_values),
                "maximum": max(endpoint_values),
            },
        }
    return output


def analyze(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    expected_manifest_sha = file_sha256(args.manifest)
    shard_paths = sorted(Path(args.features_dir).glob("features-shard-*.pt"))
    if not shard_paths:
        raise FileNotFoundError("no features-shard-*.pt files found")
    records: dict[str, Mapping[str, Any]] = {}
    shard_receipts = []
    shard_indices = set()
    expected_num_shards = None
    for path in shard_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema_version") != FEATURE_SCHEMA_VERSION
            or payload.get("manifest_sha256") != expected_manifest_sha
            or payload.get("manifest_digest") != manifest["manifest_digest"]
        ):
            raise ValueError(f"feature shard provenance differs: {path}")
        if expected_num_shards is None:
            expected_num_shards = payload["num_shards"]
        if payload["num_shards"] != expected_num_shards:
            raise ValueError("feature shards disagree on num_shards")
        shard_indices.add(payload["shard_index"])
        for record in payload["records"]:
            if record["item_id"] in records:
                raise ValueError(f"duplicate feature record: {record['item_id']}")
            records[record["item_id"]] = record
        shard_receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "shard_index": payload["shard_index"],
                "record_count": payload["record_count"],
                "runtime": payload["runtime"],
            }
        )
    expected_ids = [row["item_id"] for row in manifest["items"]]
    if set(records) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(records))
        extra = sorted(set(records) - set(expected_ids))
        raise ValueError(f"feature coverage differs: missing={missing}, extra={extra}")
    if shard_indices != set(range(int(expected_num_shards))):
        raise ValueError(f"feature shard coverage differs: {sorted(shard_indices)}")
    ordered_records = [records[identifier] for identifier in expected_ids]
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in ordered_records:
        by_group[record["group"]].append(record)
    if set(by_group) != {"simmotion_real", "project_saic_bank", "project_probe"}:
        raise ValueError(f"unexpected data groups: {sorted(by_group)}")

    parity_values = [
        float(record["official_default_parity_max_abs"]) for record in ordered_records
    ]
    results = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "evaluator_audit_complete_no_training_authority",
        "manifest": {
            "path": str(Path(args.manifest).resolve()),
            "sha256": expected_manifest_sha,
            "digest": manifest["manifest_digest"],
            "counts": manifest["counts"],
        },
        "feature_shards": shard_receipts,
        "implementation_parity": {
            "official_compute_moments_checked_for_every_video": True,
            "maximum_absolute_difference": max(parity_values),
            "paper_default_weights": [1.0, 8.0, 4.0],
            "official_uses_unbiased_torch_std": True,
        },
        "simmotion_real_pairwise": simmotion_analysis(by_group["simmotion_real"]),
        "project_saic_prompt_branch_bank": project_analysis(
            by_group["project_saic_bank"]
        ),
        "controlled_temporal_variants_all_videos": variant_analysis(ordered_records),
        "controlled_temporal_variants_project_probes": variant_analysis(
            by_group["project_probe"]
        ),
        "authority": {
            "metric_audit_only": True,
            "human_event_labels_for_project_bank_consumed": False,
            "reward_authorized": False,
            "reranking_authorized": False,
            "preference_data_authorized": False,
            "optimizer_update_authorized": False,
        },
    }
    unsigned = dict(results)
    results["receipt_digest"] = object_sha256(unsigned)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "aggregate-results.json", results)
    print(json.dumps({"receipt_digest": results["receipt_digest"]}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest")
    manifest_parser.add_argument("--simmotion-real-root", required=True)
    manifest_parser.add_argument("--project-bank-receipt", required=True)
    manifest_parser.add_argument("--probe-video", action="append", default=[])
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.set_defaults(function=build_manifest)

    extract_parser = subparsers.add_parser("extract-shard")
    extract_parser.add_argument("--manifest", required=True)
    extract_parser.add_argument("--semantic-moments-root", required=True)
    extract_parser.add_argument("--model-root", required=True)
    extract_parser.add_argument("--shard-index", type=int, required=True)
    extract_parser.add_argument("--num-shards", type=int, required=True)
    extract_parser.add_argument("--num-frames", type=int, default=32)
    extract_parser.add_argument("--frame-batch-size", type=int, default=8)
    extract_parser.add_argument("--device", default="cuda:0")
    extract_parser.add_argument("--output", required=True)
    extract_parser.set_defaults(function=extract_shard)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--manifest", required=True)
    analyze_parser.add_argument("--features-dir", required=True)
    analyze_parser.add_argument("--output-root", required=True)
    analyze_parser.set_defaults(function=analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
