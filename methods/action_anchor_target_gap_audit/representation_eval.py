#!/usr/bin/env python3
"""Control-gated VideoPrism and V-JEPA2 diagnostics for MEV action editing.

Neither representation may vote for an editing candidate unless the frozen
real-target forward control beats the source, reversed-target, and shuffled-
target controls on at least 12 of 16 pairs.  This makes temporal admission an
empirical property of the metric rather than an assumption about the model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from .audit import MANIFEST_SCHEMA, assert_not_protected_write, file_sha256
from .corrected_eval import CONTROL_SCHEMA, ROLES, load_contracts, load_json, validate_inputs, write_json, write_jsonl


RECORD_SCHEMA = "mev-action-representation-record-v3"
SUMMARY_SCHEMA = "mev-action-representation-summary-v3"
METRIC_ROLES = ROLES
CONTROL_ROLES = ("target_reverse", "target_shuffle", "source_noop")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def l2_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot normalize an empty/nonfinite representation")
    return value / norm


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(l2_normalize(left), l2_normalize(right)))


def ordered_residual_descriptor(tokens: np.ndarray) -> np.ndarray:
    """Retain ordered dynamics while suppressing the temporally constant layout.

    The spatially pooled V-JEPA tokens are centered over time, then augmented
    with directed residuals at three temporal scales. Reversing or permuting a
    motion therefore changes the descriptor; a global token mean does not.
    """

    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[0] < 5 or tokens.shape[1] < 1:
        raise ValueError("ordered tokens must have shape [time>=5, channels]")
    centered = tokens - tokens.mean(axis=0, keepdims=True)
    pieces = [centered.reshape(-1)]
    for stride in (1, 2, 4):
        residual = tokens[stride:] - tokens[:-stride]
        pieces.append(residual.reshape(-1))
    return l2_normalize(np.concatenate(pieces))


def admission_counts(
    pairs: Sequence[Mapping[str, Any]], metric: str, epsilon: float
) -> dict[str, Any]:
    comparisons: dict[str, int] = {}
    for control in CONTROL_ROLES:
        comparisons[f"forward_over_{control}"] = sum(
            float(pair["scores"][metric]["target_forward"])
            > float(pair["scores"][metric][control]) + epsilon
            for pair in pairs
        )
    threshold = 12
    admitted = len(pairs) == 16 and all(value >= threshold for value in comparisons.values())
    return {
        "epsilon": epsilon,
        "pair_count": len(pairs),
        "required_count_per_control": threshold,
        "counts": comparisons,
        "admitted_for_candidate_voting": admitted,
    }


def candidate_winner(anchor: float, frozen_base: float, epsilon: float) -> str:
    if anchor > frozen_base + epsilon:
        return "anchor"
    if frozen_base > anchor + epsilon:
        return "frozen_base"
    return "tie"


def _load_inputs(
    manifest_path: str | Path,
    controls_path: str | Path,
    contracts_path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    manifest = load_json(manifest_path)
    contracts = load_contracts(contracts_path)
    validate_inputs(manifest, contracts)
    controls = load_json(controls_path)
    if controls.get("schema_version") not in {None, CONTROL_SCHEMA}:
        raise ValueError("control manifest schema differs")
    control_rows = {row["pair_prefix"]: row for row in controls.get("samples", [])}
    if set(control_rows) != set(contracts):
        raise ValueError("control and contract pair sets differ")
    for row in control_rows.values():
        if set(row.get("roles", {})) != set(METRIC_ROLES):
            raise ValueError(f"control roles differ for {row.get('pair_prefix')}")
    return manifest, controls, contracts


def _sample_rows(
    manifest: Mapping[str, Any], controls: Mapping[str, Any], shard_index: int, num_shards: int
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard index differs")
    control_rows = {row["pair_prefix"]: row for row in controls["samples"]}
    selected = [row for row in manifest["samples"] if row["ordinal"] % num_shards == shard_index]
    return [(row, control_rows[row["pair_prefix"]]) for row in selected]


def _read_video(path: str | Path) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return frames


def _sample_indices(length: int, count: int, view: str) -> np.ndarray:
    if length <= 0 or count <= 0 or view not in {"reference", "evaluation"}:
        raise ValueError("sampling arguments differ")
    if view == "reference":
        positions = np.linspace(0.0, float(length), num=count, endpoint=False) + 0.25 * length / count
    else:
        positions = np.linspace(0.0, float(length), num=count, endpoint=False) + 0.75 * length / count
    return np.clip(positions.astype(np.int64), 0, length - 1)


def _center_crop_resize(frame: np.ndarray, size: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    scale = size / min(height, width)
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    y = (resized.shape[0] - size) // 2
    x = (resized.shape[1] - size) // 2
    return resized[y:y + size, x:x + size]


def _video_frames(path: str | Path, count: int, size: int, view: str) -> list[np.ndarray]:
    decoded = _read_video(path)
    indices = _sample_indices(len(decoded), count, view)
    return [_center_crop_resize(decoded[index], size) for index in indices]


def _record_base(
    model_kind: str,
    model_revision: str,
    manifest_path: Path,
    controls_path: Path,
    manifest_row: Mapping[str, Any],
    control_row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA,
        "created_at": utc_now(),
        "model_kind": model_kind,
        "model_revision": model_revision,
        "manifest_sha256": file_sha256(manifest_path),
        "control_manifest_sha256": file_sha256(controls_path),
        "pair_prefix": manifest_row["pair_prefix"],
        "ordinal": manifest_row["ordinal"],
        "source_action_caption": manifest_row["source_action_caption"],
        "target_action_caption": manifest_row["target_action_caption"],
        "video_sha256": {role: control_row["roles"][role]["sha256"] for role in METRIC_ROLES},
    }


class VideoPrismExtractor:
    def __init__(self, checkpoint: Path, tokenizer_model: Path):
        import jax
        import jax.numpy as jnp
        from videoprism import models as vp
        from videoprism.tokenizers import SentencePieceTokenizer

        self.jnp = jnp
        model_name = "videoprism_lvt_public_v1_base"
        self.model = vp.get_model(model_name)
        self.state = vp.load_pretrained_weights(model_name, checkpoint_path=str(checkpoint))
        self.tokenizer = SentencePieceTokenizer(str(tokenizer_model))
        self.tokenize_texts = vp.tokenize_texts

        @jax.jit
        def forward(inputs: Any, token_ids: Any, paddings: Any) -> Any:
            return self.model.apply(self.state, inputs, token_ids, paddings, train=False)

        self.forward = forward

    def text_margin(self, frames: Sequence[np.ndarray], target: str, source: str) -> tuple[float, dict[str, float]]:
        video = np.asarray(frames, dtype=np.float32)[None, ...] / 255.0
        prompts = [f"a video of {target}", f"a video of {source}"]
        token_ids, paddings = self.tokenize_texts(self.tokenizer, prompts)
        video_embeddings, text_embeddings, _ = self.forward(
            self.jnp.asarray(video), self.jnp.asarray(token_ids), self.jnp.asarray(paddings)
        )
        video_vector = np.asarray(video_embeddings).reshape(-1, np.asarray(video_embeddings).shape[-1])[0]
        text_vectors = np.asarray(text_embeddings).reshape(-1, np.asarray(text_embeddings).shape[-1])
        target_similarity = cosine(video_vector, text_vectors[0])
        source_similarity = cosine(video_vector, text_vectors[1])
        return target_similarity - source_similarity, {
            "target_text_similarity": target_similarity,
            "source_text_similarity": source_similarity,
        }


class VJEPA2Extractor:
    def __init__(self, model_path: Path):
        import torch
        from transformers import AutoModel, AutoVideoProcessor

        self.torch = torch
        self.device = torch.device("cuda")
        self.processor = AutoVideoProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(
            str(model_path), local_files_only=True, dtype=torch.float16
        ).to(self.device).eval()
        config = self.model.config
        self.spatial_patches = (int(config.image_size) // int(config.patch_size)) ** 2

    def tokens(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        from PIL import Image

        images = [Image.fromarray(frame) for frame in frames]
        inputs = self.processor(images, return_tensors="pt")
        inputs = {
            key: (
                value.to(self.device, dtype=self.torch.float16)
                if value.is_floating_point() else value.to(self.device)
            )
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            output = self.model(**inputs, skip_predictor=True)
        hidden = output.last_hidden_state.float().squeeze(0)
        if hidden.shape[0] % self.spatial_patches:
            raise ValueError("V-JEPA token count is not divisible by spatial patches")
        temporal = hidden.shape[0] // self.spatial_patches
        pooled = hidden.reshape(temporal, self.spatial_patches, hidden.shape[-1]).mean(dim=1)
        return pooled.cpu().numpy()


def videoprism_shard(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve(strict=True)
    controls_path = Path(args.controls).resolve(strict=True)
    manifest, controls, _ = _load_inputs(manifest_path, controls_path, args.contracts)
    extractor = VideoPrismExtractor(
        Path(args.checkpoint).resolve(strict=True), Path(args.tokenizer_model).resolve(strict=True)
    )
    output = Path(args.output)
    assert_not_protected_write(output)
    rows: list[Mapping[str, Any]] = []
    for manifest_row, control_row in _sample_rows(
        manifest, controls, args.shard_index, args.num_shards
    ):
        scores: dict[str, float] = {}
        components: dict[str, Mapping[str, float]] = {}
        for role in METRIC_ROLES:
            frames = _video_frames(control_row["roles"][role]["path"], 16, 288, "evaluation")
            score, detail = extractor.text_margin(
                frames, manifest_row["target_action_caption"], manifest_row["source_action_caption"]
            )
            scores[role] = score
            components[role] = detail
        row = _record_base(
            "videoprism_lvt_base_text_margin", args.model_revision,
            manifest_path, controls_path, manifest_row, control_row,
        )
        row.update({
            "preprocessing": "16 uniform three-quarter-bin frames; center crop; 288x288; RGB [0,1]",
            "score_definition": "cos(video,target-action-text)-cos(video,source-action-text)",
            "scores": {"text_margin": scores},
            "components": components,
        })
        rows.append(row)
        write_jsonl(output, rows)
        print(json.dumps({"pair_prefix": row["pair_prefix"], "model": row["model_kind"]}), flush=True)
    return 0


def vjepa2_shard(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve(strict=True)
    controls_path = Path(args.controls).resolve(strict=True)
    manifest, controls, _ = _load_inputs(manifest_path, controls_path, args.contracts)
    extractor = VJEPA2Extractor(Path(args.model).resolve(strict=True))
    output = Path(args.output)
    assert_not_protected_write(output)
    rows: list[Mapping[str, Any]] = []
    for manifest_row, control_row in _sample_rows(
        manifest, controls, args.shard_index, args.num_shards
    ):
        role_paths = control_row["roles"]
        target_reference = extractor.tokens(
            _video_frames(role_paths["target_forward"]["path"], 64, 256, "reference")
        )
        source_reference = extractor.tokens(
            _video_frames(role_paths["source_noop"]["path"], 64, 256, "reference")
        )
        target_descriptors = {
            "ordered_residual": ordered_residual_descriptor(target_reference),
            "global_mean": l2_normalize(target_reference.mean(axis=0)),
        }
        source_descriptors = {
            "ordered_residual": ordered_residual_descriptor(source_reference),
            "global_mean": l2_normalize(source_reference.mean(axis=0)),
        }
        scores = {metric: {} for metric in target_descriptors}
        components = {metric: {} for metric in target_descriptors}
        for role in METRIC_ROLES:
            tokens = extractor.tokens(
                _video_frames(role_paths[role]["path"], 64, 256, "evaluation")
            )
            descriptors = {
                "ordered_residual": ordered_residual_descriptor(tokens),
                "global_mean": l2_normalize(tokens.mean(axis=0)),
            }
            for metric, descriptor in descriptors.items():
                target_similarity = cosine(descriptor, target_descriptors[metric])
                source_similarity = cosine(descriptor, source_descriptors[metric])
                scores[metric][role] = target_similarity - source_similarity
                components[metric][role] = {
                    "target_video_similarity": target_similarity,
                    "source_video_similarity": source_similarity,
                }
        row = _record_base(
            "vjepa2_vitl_video_margin", args.model_revision,
            manifest_path, controls_path, manifest_row, control_row,
        )
        row.update({
            "preprocessing": "64 deterministic reference/evaluation views; official processor; 256x256",
            "score_definition": "cos(candidate,target-reference)-cos(candidate,source-reference)",
            "ordered_residual_definition": "time-centered spatial means plus directed token residuals at strides 1,2,4",
            "scores": scores,
            "components": components,
        })
        rows.append(row)
        write_jsonl(output, rows)
        print(json.dumps({"pair_prefix": row["pair_prefix"], "model": row["model_kind"]}), flush=True)
    return 0


def summarize(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve(strict=True)
    controls_path = Path(args.controls).resolve(strict=True)
    manifest, _, contracts = _load_inputs(manifest_path, controls_path, args.contracts)
    rows: list[Mapping[str, Any]] = []
    for path in sorted(Path(args.records_dir).glob(f"{args.prefix}*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if len(rows) != 16 or any(row.get("schema_version") != RECORD_SCHEMA for row in rows):
        raise ValueError(f"representation records differ: expected 16, received {len(rows)}")
    prefixes = [row["pair_prefix"] for row in rows]
    if len(prefixes) != len(set(prefixes)) or set(prefixes) != set(contracts):
        raise ValueError("representation pair set differs")
    ordinal = {row["pair_prefix"]: row["ordinal"] for row in manifest["samples"]}
    rows.sort(key=lambda row: ordinal[row["pair_prefix"]])
    metric_names = set(rows[0]["scores"])
    if any(set(row["scores"]) != metric_names for row in rows):
        raise ValueError("representation metric sets differ")
    metrics: dict[str, Any] = {}
    for metric in sorted(metric_names):
        admission = admission_counts(rows, metric, args.epsilon)
        winners = []
        for row in rows:
            winner = candidate_winner(
                float(row["scores"][metric]["anchor"]),
                float(row["scores"][metric]["frozen_base"]),
                args.epsilon,
            )
            winners.append({
                "pair_prefix": row["pair_prefix"],
                "diagnostic_winner": winner,
                "voting_winner": winner if admission["admitted_for_candidate_voting"] else "rejected_no_vote",
                "manual_winner": contracts[row["pair_prefix"]]["manual_winner"],
                "agrees_with_manual_if_admitted": winner == contracts[row["pair_prefix"]]["manual_winner"],
                "anchor_score": row["scores"][metric]["anchor"],
                "frozen_base_score": row["scores"][metric]["frozen_base"],
            })
        metrics[metric] = {
            "admission": admission,
            "diagnostic_winner_counts": dict(Counter(row["diagnostic_winner"] for row in winners)),
            "manual_agreement_count_if_admitted": sum(row["agrees_with_manual_if_admitted"] for row in winners),
            "mean_scores": {
                role: mean(float(row["scores"][metric][role]) for row in rows)
                for role in METRIC_ROLES
            },
            "candidate_diagnostics": winners,
        }
    payload = {
        "schema_version": SUMMARY_SCHEMA,
        "created_at": utc_now(),
        "model_kind": rows[0]["model_kind"],
        "model_revision": rows[0]["model_revision"],
        "manifest_sha256": file_sha256(manifest_path),
        "control_manifest_sha256": file_sha256(controls_path),
        "record_count": len(rows),
        "voting_policy": "no candidate vote unless real-target forward beats reverse, shuffle, and source on >=12/16 for that metric",
        "metrics": metrics,
        "pairs": rows,
    }
    write_json(args.output, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", required=True)
    common.add_argument("--controls", required=True)
    common.add_argument("--contracts", required=True)
    common.add_argument("--model-revision", required=True)
    common.add_argument("--shard-index", type=int, required=True)
    common.add_argument("--num-shards", type=int, required=True)
    common.add_argument("--output", required=True)

    videoprism = sub.add_parser("videoprism-shard", parents=[common])
    videoprism.add_argument("--checkpoint", required=True)
    videoprism.add_argument("--tokenizer-model", required=True)
    videoprism.set_defaults(function=videoprism_shard)

    vjepa = sub.add_parser("vjepa2-shard", parents=[common])
    vjepa.add_argument("--model", required=True)
    vjepa.set_defaults(function=vjepa2_shard)

    summary = sub.add_parser("summarize")
    summary.add_argument("--manifest", required=True)
    summary.add_argument("--controls", required=True)
    summary.add_argument("--contracts", required=True)
    summary.add_argument("--records-dir", required=True)
    summary.add_argument("--prefix", required=True)
    summary.add_argument("--epsilon", type=float, default=0.005)
    summary.add_argument("--output", required=True)
    summary.set_defaults(function=summarize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
