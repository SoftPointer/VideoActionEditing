#!/usr/bin/env python3
"""Audit order-aware sequence matchers for dual-anchor action editing.

The audit consumes the frozen per-frame DINO descriptors produced by the
SemanticMoments audit.  It deliberately separates objective temporal
counterfactuals from dataset-designated or generation-contract labels.

No score produced here is authorized as a training reward.  The script is a
calibration and falsification tool for candidate evaluators.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = "action-matcher-sequence-audit-v1"
EPS = 1.0e-8


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


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


def unit(value: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(value.float(), dim=dim, eps=EPS)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0)
    )


def centered(sequence: torch.Tensor) -> torch.Tensor:
    values = sequence.float()
    return unit(values - values.mean(dim=0, keepdim=True), dim=1)


def raw_unit(sequence: torch.Tensor) -> torch.Tensor:
    return unit(sequence.float(), dim=1)


def derivative(sequence: torch.Tensor) -> torch.Tensor:
    values = centered(sequence)
    return unit(values[1:] - values[:-1], dim=1)


def local_cosine_cost(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    return (1.0 - left @ right.T).clamp(min=0.0, max=2.0).cpu().numpy()


def hard_dtw_cost(
    left: torch.Tensor, right: torch.Tensor, *, open_boundary: bool = False
) -> float:
    """Return path-length-normalized DTW cost.

    ``open_boundary`` implements a symmetric OTAM-style relaxed-boundary
    diagnostic.  It is not claimed to be an official pretrained OTAM model.
    """

    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("DTW expects [T,D] sequences with a shared D")
    local = local_cosine_cost(left, right)
    rows, columns = local.shape
    table = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    steps = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    table[0, 0] = 0.0
    if open_boundary:
        table[0, 1:] = 0.0
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
    if open_boundary:
        endings = [
            (table[rows, j], steps[rows, j]) for j in range(1, columns + 1)
        ]
        total, count = min(endings, key=lambda item: item[0] / max(item[1], 1))
    else:
        total, count = table[rows, columns], steps[rows, columns]
    return float(total / max(int(count), 1))


def score_frame_diagonal(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if len(reference) != len(candidate):
        indices = torch.linspace(0, len(candidate) - 1, len(reference)).round().long()
        candidate = candidate[indices]
    return float((centered(reference) * centered(candidate)).sum(dim=1).mean())


def score_dtw_raw(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return -hard_dtw_cost(raw_unit(reference), raw_unit(candidate))


def score_dtw_centered(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return -hard_dtw_cost(centered(reference), centered(candidate))


def score_dtw_derivative(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return -hard_dtw_cost(derivative(reference), derivative(candidate))


def score_otam_style(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    left = centered(reference)
    right = centered(candidate)
    return -0.5 * (
        hard_dtw_cost(left, right, open_boundary=True)
        + hard_dtw_cost(right, left, open_boundary=True)
    )


def score_endpoint(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return cosine(reference[-1] - reference[0], candidate[-1] - candidate[0])


METRICS: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
    "frame_diagonal_centered": score_frame_diagonal,
    "dtw_raw": score_dtw_raw,
    "dtw_centered": score_dtw_centered,
    "dtw_derivative": score_dtw_derivative,
    "otam_style_centered": score_otam_style,
    "endpoint": score_endpoint,
}


def deterministic_permutation(identifier: str, count: int) -> torch.Tensor:
    seed = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16], 16)
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed))


def warp_indices(count: int, power: float) -> torch.Tensor:
    phase = torch.linspace(0.0, 1.0, count)
    return (phase.pow(power) * (count - 1)).round().long()


def controlled_variants(identifier: str, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
    count = len(sequence)
    if count < 8:
        raise ValueError("controlled variants require at least eight frames")
    ease_in = sequence[warp_indices(count, 1.45)]
    ease_out = sequence[(1.0 - (1.0 - torch.linspace(0.0, 1.0, count)).pow(1.45)).mul(count - 1).round().long()]
    cutoff = max(2, int(round(count * 0.60)))
    incomplete = torch.cat(
        [sequence[:cutoff], sequence[cutoff - 1 : cutoff].repeat(count - cutoff, 1)],
        dim=0,
    )
    noop = sequence[0:1].repeat(count, 1)
    permutation = deterministic_permutation(identifier, count)
    return {
        "speed_ease_in": ease_in,
        "speed_ease_out": ease_out,
        "reverse": torch.flip(sequence, dims=(0,)),
        "reverse_speed": torch.flip(ease_in, dims=(0,)),
        "random_shuffle": sequence[permutation],
        "noop_first_frame": noop,
        "incomplete_tail_hold": incomplete,
    }


def load_records(feature_root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(Path(feature_root).glob("features-shard-*.pt"))
    if not paths:
        raise FileNotFoundError(f"no feature shards in {feature_root}")
    records: list[dict[str, Any]] = []
    receipts = []
    manifest_digest = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != "semantic-moments-action-reward-features-v1":
            raise ValueError(f"unexpected feature schema: {path}")
        if manifest_digest is None:
            manifest_digest = payload["manifest_digest"]
        elif manifest_digest != payload["manifest_digest"]:
            raise ValueError("feature shards use different manifests")
        records.extend(payload["records"])
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "record_count": payload["record_count"],
                "shard_index": payload["shard_index"],
            }
        )
    item_ids = [row["item_id"] for row in records]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("duplicate item_id across feature shards")
    return records, {
        "manifest_digest": manifest_digest,
        "feature_shards": receipts,
        "record_count": len(records),
    }


def group_by_metadata(
    records: Iterable[Mapping[str, Any]], key: str
) -> dict[Any, list[Mapping[str, Any]]]:
    output: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        output[row["metadata"][key]].append(row)
    return dict(output)


def split_name(item_id: str) -> str:
    value = int(hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:8], 16)
    return "fit" if value % 2 == 0 else "heldout"


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def choose_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    unique = sorted(set(float(value) for value in scores))
    candidates = [unique[0] - 1.0e-6]
    candidates.extend((a + b) / 2 for a, b in zip(unique[:-1], unique[1:]))
    candidates.append(unique[-1] + 1.0e-6)
    best = max(
        candidates,
        key=lambda threshold: (
            np.mean([int((score >= threshold) == bool(label)) for label, score in zip(labels, scores)]),
            threshold,
        ),
    )
    return float(best)


def classification_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fit = [row for row in rows if row["split"] == "fit"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    threshold = choose_threshold(
        [int(row["label"]) for row in fit], [float(row["score"]) for row in fit]
    )
    output: dict[str, Any] = {
        "fit_count": len(fit),
        "heldout_count": len(heldout),
        "threshold_from_fit": threshold,
    }
    for name, subset in (("fit", fit), ("heldout", heldout)):
        labels = [int(row["label"]) for row in subset]
        scores = [float(row["score"]) for row in subset]
        decisions = [int(score >= threshold) for score in scores]
        output[name] = {
            "accuracy": float(np.mean([a == b for a, b in zip(labels, decisions)])),
            "auc": auc(labels, scores),
            "positive_acceptance": float(
                np.mean([decision for decision, label in zip(decisions, labels) if label])
            ),
            "negative_rejection": float(
                np.mean([1 - decision for decision, label in zip(decisions, labels) if not label])
            ),
        }
        by_variant = defaultdict(list)
        for row, decision in zip(subset, decisions):
            by_variant[row["variant"]].append(
                int(decision == int(row["label"]))
            )
        output[name]["accuracy_by_variant"] = {
            key: float(np.mean(values)) for key, values in sorted(by_variant.items())
        }
    return output


def controlled_open_set(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive_names = {"speed_ease_in", "speed_ease_out"}
    rows_by_metric: dict[str, list[dict[str, Any]]] = {name: [] for name in METRICS}
    for record in records:
        reference = record["frame_sequence"].float()
        variants = controlled_variants(record["item_id"], reference)
        for metric_name, metric in METRICS.items():
            for variant_name, candidate in variants.items():
                rows_by_metric[metric_name].append(
                    {
                        "item_id": record["item_id"],
                        "group": record["group"],
                        "split": split_name(record["item_id"]),
                        "variant": variant_name,
                        "label": int(variant_name in positive_names),
                        "score": metric(reference, candidate),
                    }
                )
    return {
        name: {**classification_summary(rows), "rows": rows}
        for name, rows in rows_by_metric.items()
    }


def binary_summary(margins: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in margins]
    return {
        "count": len(values),
        "wins": sum(value > 0 for value in values),
        "ties": sum(abs(value) <= EPS for value in values),
        "accuracy": float(np.mean([value > 0 for value in values])) if values else 0.0,
        "mean_margin": float(np.mean(values)) if values else 0.0,
        "median_margin": float(np.median(values)) if values else 0.0,
    }


def simmotion_designated(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_example = group_by_metadata(records, "example_id")
    output: dict[str, Any] = {}
    for metric_name, metric in METRICS.items():
        pairwise_rows = []
        reverse_rows = []
        for example_id, rows in sorted(by_example.items()):
            by_role = {row["metadata"]["role"]: row for row in rows}
            reference = by_role["ref"]["frame_sequence"].float()
            positive = by_role["positive"]["frame_sequence"].float()
            negative = by_role["negative"]["frame_sequence"].float()
            positive_score = metric(reference, positive)
            negative_score = metric(reference, negative)
            pairwise_rows.append(
                {
                    "example_id": example_id,
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "margin": positive_score - negative_score,
                }
            )
            reverse_score = metric(reference, torch.flip(positive, dims=(0,)))
            reverse_rows.append(
                {
                    "example_id": example_id,
                    "positive_score": positive_score,
                    "reversed_positive_score": reverse_score,
                    "margin": positive_score - reverse_score,
                }
            )
        output[metric_name] = {
            "dataset_designated_positive_over_negative": {
                **binary_summary([row["margin"] for row in pairwise_rows]),
                "rows": pairwise_rows,
            },
            "positive_over_exact_reverse_of_positive": {
                **binary_summary([row["margin"] for row in reverse_rows]),
                "rows": reverse_rows,
            },
        }
    return output


def project_contract(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric_name, metric in METRICS.items():
        margins: dict[str, list[float]] = {"reverse": [], "noop": []}
        detail = []
        for query in records:
            meta = query["metadata"]
            if meta["branch"] != "forward":
                continue
            siblings = [
                row
                for row in records
                if row["metadata"]["iid"] == meta["iid"]
                and row["metadata"]["seed"] != meta["seed"]
            ]
            positives = [row for row in siblings if row["metadata"]["branch"] == "forward"]
            if not positives:
                continue
            reference = query["frame_sequence"].float()
            positive_score = max(
                metric(reference, row["frame_sequence"].float()) for row in positives
            )
            for branch in ("reverse", "noop"):
                negatives = [row for row in siblings if row["metadata"]["branch"] == branch]
                if not negatives:
                    continue
                negative_score = max(
                    metric(reference, row["frame_sequence"].float()) for row in negatives
                )
                margin = positive_score - negative_score
                margins[branch].append(margin)
                detail.append(
                    {
                        "query": meta["candidate_id"],
                        "negative_branch": branch,
                        "positive_score": positive_score,
                        "negative_score": negative_score,
                        "margin": margin,
                    }
                )
        output[metric_name] = {
            "label_authority": "generation branch contract; not human action truth",
            "forward_over_reverse": binary_summary(margins["reverse"]),
            "forward_over_noop": binary_summary(margins["noop"]),
            "rows": detail,
        }
    return output


def nearest_neighbor_leakage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric_name, metric in METRICS.items():
        decisions = []
        for query in records:
            qmeta = query["metadata"]
            candidates = [
                row
                for row in records
                if row["metadata"]["candidate_id"] != qmeta["candidate_id"]
                and row["metadata"]["seed"] != qmeta["seed"]
            ]
            best = max(
                candidates,
                key=lambda row: metric(
                    query["frame_sequence"].float(), row["frame_sequence"].float()
                ),
            )
            bmeta = best["metadata"]
            decisions.append(
                {
                    "query": qmeta["candidate_id"],
                    "neighbor": bmeta["candidate_id"],
                    "same_iid": bmeta["iid"] == qmeta["iid"],
                    "same_branch": bmeta["branch"] == qmeta["branch"],
                }
            )
        output[metric_name] = {
            "count": len(decisions),
            "same_iid_rate": float(np.mean([row["same_iid"] for row in decisions])),
            "same_branch_rate": float(np.mean([row["same_branch"] for row in decisions])),
            "rows": decisions,
        }
    return output


def calibrated_transfer(
    controlled: Mapping[str, Mapping[str, Any]],
    simmotion: Mapping[str, Mapping[str, Any]],
    project: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply fit-only controlled thresholds unchanged to cross-video scores.

    This deliberately tests abstention calibration.  Pairwise preference can be
    useful even when absolute score scales do not transfer; those are reported
    as different properties rather than collapsed into an accuracy number.
    """

    output: dict[str, Any] = {}
    for metric_name in METRICS:
        threshold = float(controlled[metric_name]["threshold_from_fit"])
        sim_rows = simmotion[metric_name][
            "dataset_designated_positive_over_negative"
        ]["rows"]
        outcomes = Counter()
        for row in sim_rows:
            positive = float(row["positive_score"]) >= threshold
            negative = float(row["negative_score"]) >= threshold
            if positive and not negative:
                outcomes["designated_positive_only"] += 1
            elif positive and negative:
                outcomes["both_accepted"] += 1
            elif not positive and not negative:
                outcomes["neither_accepted"] += 1
            else:
                outcomes["designated_negative_only"] += 1

        project_rows = project[metric_name]["rows"]
        project_by_branch: dict[str, Any] = {}
        for branch in ("reverse", "noop"):
            subset = [row for row in project_rows if row["negative_branch"] == branch]
            forward_pass = [float(row["positive_score"]) >= threshold for row in subset]
            negative_pass = [float(row["negative_score"]) >= threshold for row in subset]
            project_by_branch[branch] = {
                "comparison_count": len(subset),
                "forward_accept_count": sum(forward_pass),
                "negative_accept_count": sum(negative_pass),
                "forward_accept_negative_reject_count": sum(
                    left and not right
                    for left, right in zip(forward_pass, negative_pass)
                ),
            }
        output[metric_name] = {
            "threshold_from_controlled_fit": threshold,
            "simmotion_pair_outcomes": {
                key: int(outcomes[key])
                for key in (
                    "designated_positive_only",
                    "both_accepted",
                    "neither_accepted",
                    "designated_negative_only",
                )
            },
            "project_pair_outcomes": project_by_branch,
        }
    return output


def analyze(args: argparse.Namespace) -> int:
    records, feature_receipt = load_records(args.feature_root)
    simmotion = [row for row in records if row["group"] == "simmotion_real"]
    project = [row for row in records if row["group"] == "project_saic_bank"]
    probes = [row for row in records if row["group"] == "project_probe"]
    if (len(simmotion), len(project), len(probes)) != (120, 60, 2):
        raise ValueError(
            f"unexpected frozen population: {len(simmotion)}, {len(project)}, {len(probes)}"
        )
    controlled = controlled_open_set(records)
    simmotion_results = simmotion_designated(simmotion)
    project_results = project_contract(project)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "authority": {
            "reward_authorized": False,
            "reranking_authorized": False,
            "preference_data_authorized": False,
            "optimizer_update_authorized": False,
        },
        "protocol": {
            "objective_open_set": (
                "fit threshold on hash-split videos using order-preserving speed warps "
                "as positives and reverse/shuffle/noop/incomplete as negatives; report "
                "held-out acceptance and rejection"
            ),
            "simmotion_authority": (
                "dataset-designated retrieval labels only; not action correctness truth"
            ),
            "project_authority": (
                "generation branch contracts only; not human action correctness truth"
            ),
            "otam_style_status": (
                "training-free relaxed-boundary algorithmic diagnostic; not an official "
                "OTAM checkpoint"
            ),
        },
        "feature_receipt": feature_receipt,
        "metric_names": list(METRICS),
        "controlled_open_set": controlled,
        "simmotion_designated": simmotion_results,
        "project_contract": project_results,
        "calibrated_transfer": calibrated_transfer(
            controlled, simmotion_results, project_results
        ),
        "project_nearest_neighbor_leakage": nearest_neighbor_leakage(project),
    }
    result["result_digest"] = object_sha256(result)
    write_json(args.output, result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "digest": result["result_digest"]}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    return analyze(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
