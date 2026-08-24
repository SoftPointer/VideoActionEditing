#!/usr/bin/env python3
"""Evaluate the official pretrained TEAM matcher on the frozen audit videos.

TEAM is episodic: it ranks support classes for a query and does not expose an
independent pairwise scalar.  This runner therefore reports group-relative
ranking, two-way decisions, and context-instability separately.  Dataset and
generation-contract labels are not promoted to action ground truth.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TVF


FEATURE_SCHEMA = "team-action-matcher-features-v1"
RESULT_SCHEMA = "team-action-matcher-audit-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


@dataclass
class TeamArgs:
    backbone: str = "ViT"
    seq_len: int = 8
    agg_num: int = 50
    num_gpus: int = 1


def load_team_model(
    team_root: str | Path, checkpoint: str | Path, device: str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    team_root = Path(team_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not (team_root / "model.py").is_file():
        raise FileNotFoundError(f"TEAM model.py missing: {team_root}")
    sys.path.insert(0, str(team_root))
    try:
        timm = importlib.import_module("timm")
        original_create_model = timm.create_model

        def offline_create_model(name: str, *args: Any, **kwargs: Any) -> Any:
            kwargs["pretrained"] = False
            return original_create_model(name, *args, **kwargs)

        timm.create_model = offline_create_model
        try:
            team_module = importlib.import_module("model")
            model = team_module.TEAM(TeamArgs())
        finally:
            timm.create_model = original_create_model
    finally:
        sys.path.pop(0)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_iteration = int(payload["iteration"])
    state = payload["model_state_dict"]
    if any(key.startswith("module.") for key in state):
        state = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"TEAM checkpoint mismatch: {incompatible}")
    del state, payload
    model = model.to(torch.device(device)).eval()
    receipt = {
        "team_root": str(team_root),
        "team_commit": subprocess.run(
            ["git", "-C", str(team_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_iteration": checkpoint_iteration,
        "strict_load": True,
        "config": {
            "dataset": "SSv2-Small",
            "backbone": "ViT-B/16",
            "shot": 1,
            "seq_len": 8,
            "agg_num": 50,
        },
    }
    return model, receipt


def uniform_center_indices(total: int, count: int) -> list[int]:
    if total < count:
        return [int(round(value)) for value in np.linspace(0, total - 1, count)]
    interval = total // count
    return [int((index * interval + index * interval + interval - 1) / 2) for index in range(count)]


def load_team_frames(path: str | Path, count: int = 8) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise ValueError(f"video contains no frames: {path}")
    tensors = []
    for index in uniform_center_indices(total, count):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"cannot read frame {index}: {path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = TVF.resize(image, 256)
        image = TVF.center_crop(image, [224, 224])
        tensor = TVF.to_tensor(image)
        tensor = TVF.normalize(tensor, mean=[0.45] * 3, std=[0.225] * 3)
        tensors.append(tensor)
    capture.release()
    return torch.stack(tensors)


@torch.inference_mode()
def extract_video(model: torch.nn.Module, path: str | Path, device: str) -> torch.Tensor:
    frames = load_team_frames(path).to(torch.device(device))
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        features = model.backbone(frames)
    if features.shape != (8, 768):
        raise RuntimeError(f"unexpected TEAM ViT feature shape: {features.shape}")
    return features.detach().float().cpu()


def runtime_identity(device: str) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "device": device,
        "visible_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
    }


def extract_shard(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid shard index")
    selected = [
        item
        for ordinal, item in enumerate(manifest["items"])
        if ordinal % args.num_shards == args.shard_index
    ]
    model, model_receipt = load_team_model(args.team_root, args.checkpoint, args.device)
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
        records.append(
            {
                "item_id": item["item_id"],
                "group": item["group"],
                "path": item["path"],
                "sha256": item["sha256"],
                "metadata": item["metadata"],
                "frame_sequence": extract_video(model, item["path"], args.device),
            }
        )
    payload = {
        "schema_version": FEATURE_SCHEMA,
        "created_at": utc_now(),
        "manifest_digest": manifest["manifest_digest"],
        "manifest_sha256": file_sha256(args.manifest),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "record_count": len(records),
        "model_receipt": model_receipt,
        "runtime": runtime_identity(args.device),
        "records": records,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(destination)
    print(f"wrote {destination} records={len(records)}")
    return 0


def load_feature_records(feature_root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(Path(feature_root).glob("team-features-shard-*.pt"))
    if not paths:
        raise FileNotFoundError(f"no TEAM feature shards: {feature_root}")
    records = []
    receipts = []
    model_receipt = None
    manifest_digest = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["schema_version"] != FEATURE_SCHEMA:
            raise ValueError(f"invalid TEAM feature shard: {path}")
        if model_receipt is None:
            model_receipt = payload["model_receipt"]
            manifest_digest = payload["manifest_digest"]
        elif payload["model_receipt"] != model_receipt or payload["manifest_digest"] != manifest_digest:
            raise ValueError("TEAM feature receipts differ across shards")
        records.extend(payload["records"])
        receipts.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    if len(records) != len({row["item_id"] for row in records}):
        raise ValueError("duplicate TEAM feature item IDs")
    return records, {
        "model": model_receipt,
        "manifest_digest": manifest_digest,
        "feature_shards": receipts,
        "record_count": len(records),
    }


@torch.inference_mode()
def team_logits_from_features(
    model: torch.nn.Module,
    supports: torch.Tensor,
    query: torch.Tensor,
    device: str,
) -> list[float]:
    if supports.ndim != 3 or query.ndim != 2 or supports.shape[1:] != query.shape:
        raise ValueError("TEAM feature matcher expects supports [W,T,D], query [T,D]")
    if len(supports) < 2:
        raise ValueError("TEAM discriminative matching requires at least two supports")
    supports = supports.to(torch.device(device))
    query = query.unsqueeze(0).to(torch.device(device))
    labels = torch.arange(len(supports), device=torch.device(device))
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        spt, tar = model.reshape(supports, query, labels)
        spt, tar = spt.mean(dim=2), tar.mean(dim=2)
        spt_pos, spt_neg = model.DPM(spt)
        spt_pos, spt_neg = model.relu(spt_pos), model.relu(spt_neg)
        tar_pos, tar_neg = model.DPM(tar)
        tar_pos, tar_neg = model.relu(tar_pos), model.relu(tar_neg)
        spt_pos_sim = model.get_spt_sim(spt_pos)
        spt_neg_sim = model.get_spt_sim(spt_neg)
        spt_other = model.get_other(spt)
        spt_disc_pos, spt_disc_neg = model.DPM(
            spt, spt_other, spt_pos_sim, spt_neg_sim
        )
        spt_disc_pos = model.relu(spt_disc_pos).mean(dim=2)
        spt_disc_neg = model.relu(spt_disc_neg).mean(dim=2)
        cumulative, _, _ = model.get_cum_dists(
            spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg
        )
        logits = -cumulative
    values = logits[0].detach().float().cpu().tolist()
    if len(values) != len(supports) or not all(np.isfinite(values)):
        raise RuntimeError(f"invalid TEAM logits: {values}")
    return [float(value) for value in values]


def controlled_variants(sequence_module: Any, item_id: str, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
    return sequence_module.controlled_variants(item_id, sequence)


def controlled_ranking(
    records: Sequence[Mapping[str, Any]], model: torch.nn.Module, sequence_module: Any, device: str
) -> dict[str, Any]:
    names = [
        "speed_ease_in",
        "speed_ease_out",
        "reverse",
        "reverse_speed",
        "random_shuffle",
        "noop_first_frame",
        "incomplete_tail_hold",
    ]
    positives = {"speed_ease_in", "speed_ease_out"}
    rows = []
    pairwise_by_negative: dict[str, list[bool]] = defaultdict(list)
    context_flips = []
    for record in records:
        query = record["frame_sequence"].float()
        variants = controlled_variants(sequence_module, record["item_id"], query)
        supports = torch.stack([variants[name] for name in names])
        all_logits = team_logits_from_features(model, supports, query, device)
        top_name = names[int(np.argmax(all_logits))]
        pairwise = {}
        for negative_name in names:
            if negative_name in positives:
                continue
            two_names = ["speed_ease_in", negative_name]
            two_logits = team_logits_from_features(
                model, torch.stack([variants[name] for name in two_names]), query, device
            )
            correct = two_logits[0] > two_logits[1]
            pairwise[negative_name] = {
                "correct": correct,
                "margin": two_logits[0] - two_logits[1],
            }
            pairwise_by_negative[negative_name].append(correct)
            all_margin = all_logits[names.index("speed_ease_in")] - all_logits[names.index(negative_name)]
            context_flips.append((all_margin > 0) != correct)
        rows.append(
            {
                "item_id": record["item_id"],
                "group": record["group"],
                "top1": top_name,
                "top1_is_positive": top_name in positives,
                "all_way_logits": dict(zip(names, all_logits)),
                "two_way": pairwise,
            }
        )
    return {
        "record_count": len(rows),
        "candidate_count": len(names),
        "positive_count": len(positives),
        "top1_positive_wins": sum(row["top1_is_positive"] for row in rows),
        "top1_positive_rate": float(np.mean([row["top1_is_positive"] for row in rows])),
        "two_way_accuracy_by_negative": {
            name: float(np.mean(values)) for name, values in sorted(pairwise_by_negative.items())
        },
        "two_way_by_negative": {
            name: {
                "count": len(values),
                "wins": sum(values),
                "accuracy": float(np.mean(values)),
            }
            for name, values in sorted(pairwise_by_negative.items())
        },
        "context_comparison_count": len(context_flips),
        "context_flip_count": sum(context_flips),
        "two_way_vs_all_way_order_flip_rate": float(np.mean(context_flips)),
        "rows": rows,
    }


def group_by_metadata(records: Sequence[Mapping[str, Any]], key: str) -> dict[Any, list[Mapping[str, Any]]]:
    output: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        output[row["metadata"][key]].append(row)
    return dict(output)


def simmotion_designated(
    records: Sequence[Mapping[str, Any]], model: torch.nn.Module, device: str
) -> dict[str, Any]:
    rows = []
    reverse_rows = []
    context_flips = []
    by_example = group_by_metadata(records, "example_id")
    ordered_examples = sorted(by_example)
    for index, example_id in enumerate(ordered_examples):
        by_role = {row["metadata"]["role"]: row for row in by_example[example_id]}
        query = by_role["ref"]["frame_sequence"].float()
        positive = by_role["positive"]["frame_sequence"].float()
        negative = by_role["negative"]["frame_sequence"].float()
        two_logits = team_logits_from_features(model, torch.stack([positive, negative]), query, device)
        other_example = ordered_examples[(index + 1) % len(ordered_examples)]
        other = next(
            row["frame_sequence"].float()
            for row in by_example[other_example]
            if row["metadata"]["role"] == "positive"
        )
        three_logits = team_logits_from_features(
            model, torch.stack([positive, negative, other]), query, device
        )
        rows.append(
            {
                "example_id": example_id,
                "two_way_margin": two_logits[0] - two_logits[1],
                "three_way_margin": three_logits[0] - three_logits[1],
                "dataset_agreement": two_logits[0] > two_logits[1],
            }
        )
        context_flips.append((two_logits[0] > two_logits[1]) != (three_logits[0] > three_logits[1]))
        reverse_logits = team_logits_from_features(
            model, torch.stack([positive, torch.flip(positive, dims=(0,))]), query, device
        )
        reverse_rows.append(
            {
                "example_id": example_id,
                "margin": reverse_logits[0] - reverse_logits[1],
                "forward_positive_wins": reverse_logits[0] > reverse_logits[1],
            }
        )
    return {
        "label_authority": "SimMotion dataset designation; not action correctness truth",
        "example_count": len(rows),
        "dataset_designated_pairwise_wins": sum(row["dataset_agreement"] for row in rows),
        "dataset_designated_pairwise_agreement": float(np.mean([row["dataset_agreement"] for row in rows])),
        "positive_over_exact_reverse_wins": sum(
            row["forward_positive_wins"] for row in reverse_rows
        ),
        "positive_over_exact_reverse_of_positive": float(
            np.mean([row["forward_positive_wins"] for row in reverse_rows])
        ),
        "context_flip_count": sum(context_flips),
        "two_way_vs_three_way_pair_order_flip_rate": float(np.mean(context_flips)),
        "rows": rows,
        "reverse_rows": reverse_rows,
    }


def project_contract(
    records: Sequence[Mapping[str, Any]], model: torch.nn.Module, device: str
) -> dict[str, Any]:
    rows = []
    top1 = []
    pairwise_reverse = []
    pairwise_noop = []
    for query in records:
        meta = query["metadata"]
        if meta["branch"] != "forward":
            continue
        candidates = [
            row
            for row in records
            if row["metadata"]["iid"] == meta["iid"]
            and row["metadata"]["seed"] != meta["seed"]
        ]
        if len(candidates) < 3:
            continue
        features = torch.stack([row["frame_sequence"].float() for row in candidates])
        logits = team_logits_from_features(model, features, query["frame_sequence"].float(), device)
        winner = candidates[int(np.argmax(logits))]["metadata"]["branch"]
        top1.append(winner == "forward")
        pair_rows = []
        positives = [row for row in candidates if row["metadata"]["branch"] == "forward"]
        for branch, sink in (("reverse", pairwise_reverse), ("noop", pairwise_noop)):
            negatives = [row for row in candidates if row["metadata"]["branch"] == branch]
            decisions = []
            for positive in positives:
                for negative in negatives:
                    values = team_logits_from_features(
                        model,
                        torch.stack([
                            positive["frame_sequence"].float(),
                            negative["frame_sequence"].float(),
                        ]),
                        query["frame_sequence"].float(),
                        device,
                    )
                    decisions.append(values[0] > values[1])
            sink.extend(decisions)
            pair_rows.append({"negative_branch": branch, "accuracy": float(np.mean(decisions))})
        rows.append(
            {
                "query": meta["candidate_id"],
                "top1_branch": winner,
                "top1_forward": winner == "forward",
                "candidate_branches": [row["metadata"]["branch"] for row in candidates],
                "logits": logits,
                "two_way": pair_rows,
            }
        )
    return {
        "label_authority": "generation branch contract; not human action truth",
        "query_count": len(rows),
        "all_candidate_top1_forward_wins": sum(top1),
        "all_candidate_top1_forward_rate": float(np.mean(top1)),
        "two_way_forward_over_reverse_count": len(pairwise_reverse),
        "two_way_forward_over_reverse_wins": sum(pairwise_reverse),
        "two_way_forward_over_reverse": float(np.mean(pairwise_reverse)),
        "two_way_forward_over_noop_count": len(pairwise_noop),
        "two_way_forward_over_noop_wins": sum(pairwise_noop),
        "two_way_forward_over_noop": float(np.mean(pairwise_noop)),
        "rows": rows,
    }


def analyze(args: argparse.Namespace) -> int:
    sequence_spec = importlib.util.spec_from_file_location(
        "sequence_audit_for_team", args.sequence_audit
    )
    if sequence_spec is None or sequence_spec.loader is None:
        raise RuntimeError("cannot load sequence audit module")
    sequence_module = importlib.util.module_from_spec(sequence_spec)
    sequence_spec.loader.exec_module(sequence_module)
    records, feature_receipt = load_feature_records(args.feature_root)
    if len(records) != 182:
        raise ValueError(f"expected 182 TEAM feature records, got {len(records)}")
    model, model_receipt = load_team_model(args.team_root, args.checkpoint, args.device)
    if model_receipt != feature_receipt["model"]:
        raise ValueError("analysis model receipt differs from extraction model receipt")
    simmotion = [row for row in records if row["group"] == "simmotion_real"]
    project = [row for row in records if row["group"] == "project_saic_bank"]
    result = {
        "schema_version": RESULT_SCHEMA,
        "created_at": utc_now(),
        "authority": {
            "reward_authorized": False,
            "reranking_authorized": False,
            "preference_data_authorized": False,
            "optimizer_update_authorized": False,
        },
        "protocol": {
            "model": "official TEAM SSV2-Small ViT 1-shot checkpoint",
            "output_semantics": (
                "episodic group-relative logits; not an absolute pairwise score or calibrated reward"
            ),
            "controlled_authority": "deterministic temporal transformations",
            "simmotion_authority": "dataset designation only",
            "project_authority": "generation contract only",
        },
        "feature_receipt": feature_receipt,
        "controlled_ranking": controlled_ranking(records, model, sequence_module, args.device),
        "simmotion_designated": simmotion_designated(simmotion, model, args.device),
        "project_contract": project_contract(project, model, args.device),
    }
    result["result_digest"] = object_sha256(result)
    write_json(args.output, result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "digest": result["result_digest"]}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract-shard")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--team-root", required=True)
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--shard-index", type=int, required=True)
    extract.add_argument("--num-shards", type=int, required=True)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--output", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--team-root", required=True)
    analyze_parser.add_argument("--checkpoint", required=True)
    analyze_parser.add_argument("--feature-root", required=True)
    analyze_parser.add_argument("--sequence-audit", required=True)
    analyze_parser.add_argument("--device", default="cuda:0")
    analyze_parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "extract-shard":
        return extract_shard(args)
    if args.command == "analyze":
        return analyze(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
