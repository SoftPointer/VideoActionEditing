#!/usr/bin/env python3
"""Frozen-DINO probe for appearance-deconfounded SAIC event representations.

The complete registered 8-source x 3-anchor-branch x seed bank is consumed
without selecting rows.  Frozen DINO frame features are transformed into raw,
temporally centered, endpoint, velocity, and temporal-signature views.  Fit
sources define branch centroids and action-minus-noop contrast directions;
confirmation sources are read only for evaluation.  This is a representation
diagnostic, not an event audit, target admission, optimizer, or training run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np


SCHEMA = "saic-anchor-dinov2-temporal-representation-v1"
SOURCE_SCHEMA = "bernini-saic-reversible-source-set-v1"
RECEIPT_NAME = "saic-event-generation-receipt.json"
BRANCHES = ("forward", "reverse", "noop")
ACTION_BRANCHES = ("forward", "reverse")
FAMILIES = ("dog", "human")
SPLITS = ("fit", "confirmation")
EXPECTED_SOURCES = 8
EXPECTED_CANDIDATES = 60
EPS = 1.0e-12
AUTHORITY = {
    "event_verified": False,
    "human_review": False,
    "data_selection": False,
    "representation_selection": False,
    "training": False,
    "optimizer": False,
    "scientific_claim": False,
}


def die(message: str) -> NoReturn:
    raise SystemExit(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value.astype(np.float32, copy=False))
    metadata = {"shape": list(array.shape), "dtype": str(array.dtype)}
    return hashlib.sha256(
        canonical_bytes(metadata) + b"\x00" + array.tobytes()
    ).hexdigest()


def unit(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if not math.isfinite(norm):
        raise ValueError("representation norm is non-finite")
    if norm <= EPS:
        return np.zeros_like(flat)
    return np.ascontiguousarray(flat / norm)


def temporal_representations(
    global_feature: np.ndarray, dense_feature: np.ndarray
) -> dict[str, np.ndarray]:
    """Build fixed, no-learned-parameter views of one 17-frame trajectory."""
    global_feature = np.asarray(global_feature, dtype=np.float32)
    dense_feature = np.asarray(dense_feature, dtype=np.float32)
    if (
        global_feature.ndim != 2
        or dense_feature.ndim != 3
        or global_feature.shape[0] != 17
        or dense_feature.shape[0] != 17
        or global_feature.shape[-1] != dense_feature.shape[-1]
        or not np.isfinite(global_feature).all()
        or not np.isfinite(dense_feature).all()
    ):
        raise ValueError("DINO trajectory geometry differs")
    centered = global_feature - global_feature.mean(axis=0, keepdims=True)
    velocity = np.diff(global_feature, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    similarity = global_feature @ global_feature.T
    upper = similarity[np.triu_indices(global_feature.shape[0], k=1)]
    dense_velocity = np.diff(dense_feature, axis=0)
    dense_speed = np.median(np.linalg.norm(dense_velocity, axis=-1), axis=1)
    dense_lag = np.asarray([
        np.median(np.sum(
            dense_feature[:-lag] * dense_feature[lag:], axis=-1
        ))
        for lag in range(1, 9)
    ], dtype=np.float32)
    return {
        "appearance_mean": unit(global_feature.mean(axis=0)),
        "endpoint_arrow": unit(global_feature[-1] - global_feature[0]),
        "centered_trajectory": unit(centered),
        "velocity_trajectory": unit(velocity),
        "speed_profile": unit(speed - speed.mean()),
        "temporal_self_similarity": unit(upper - upper.mean()),
        "dense_speed_profile": unit(dense_speed - dense_speed.mean()),
        "dense_lag_profile": unit(dense_lag - dense_lag.mean()),
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("cosine geometry differs")
    value = float(np.dot(left, right))
    if not math.isfinite(value):
        raise ValueError("cosine is non-finite")
    return max(-1.0, min(1.0, value))


def centroid(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("empty centroid")
    return unit(np.stack(rows, axis=0).mean(axis=0))


def branch_centroid_evaluation(
    rows: Sequence[Mapping[str, Any]], representation: str
) -> dict[str, Any]:
    fit = [row for row in rows if row["analysis_split"] == "fit"]
    confirmation = [
        row for row in rows if row["analysis_split"] == "confirmation"
    ]
    prototypes = {
        (family, branch): centroid([
            row["representations"][representation]
            for row in fit
            if row["actor_family"] == family and row["branch"] == branch
        ])
        for family in FAMILIES for branch in BRANCHES
    }
    confusion = {truth: Counter() for truth in BRANCHES}
    correct_by_family = Counter()
    total_by_family = Counter()
    predictions = []
    for row in confirmation:
        scores = {
            branch: cosine(
                row["representations"][representation],
                prototypes[(row["actor_family"], branch)],
            )
            for branch in BRANCHES
        }
        predicted = max(BRANCHES, key=lambda branch: (scores[branch], branch))
        truth = row["branch"]
        correct = predicted == truth
        confusion[truth][predicted] += 1
        total_by_family[row["actor_family"]] += 1
        correct_by_family[row["actor_family"]] += int(correct)
        predictions.append({
            "candidate_id": row["candidate_id"],
            "truth": truth,
            "predicted": predicted,
            "correct": correct,
            "scores": scores,
        })
    correct_count = sum(row["correct"] for row in predictions)
    return {
        "fit_row_count": len(fit),
        "confirmation_row_count": len(confirmation),
        "accuracy": correct_count / len(predictions),
        "correct_count": correct_count,
        "confusion": {
            truth: {branch: confusion[truth][branch] for branch in BRANCHES}
            for truth in BRANCHES
        },
        "accuracy_by_actor_family": {
            family: correct_by_family[family] / total_by_family[family]
            for family in FAMILIES
        },
        "predictions": predictions,
    }


def nearest_neighbor_diagnostic(
    rows: Sequence[Mapping[str, Any]], representation: str
) -> dict[str, Any]:
    results = []
    for row in rows:
        others = [other for other in rows if other["candidate_id"] != row["candidate_id"]]
        best = max(
            others,
            key=lambda other: cosine(
                row["representations"][representation],
                other["representations"][representation],
            ),
        )
        results.append({
            "candidate_id": row["candidate_id"],
            "neighbor_id": best["candidate_id"],
            "cosine": cosine(
                row["representations"][representation],
                best["representations"][representation],
            ),
            "same_iid": row["iid"] == best["iid"],
            "same_branch": row["branch"] == best["branch"],
            "same_actor_family": row["actor_family"] == best["actor_family"],
        })
    return {
        "row_count": len(results),
        "same_iid_top1_rate": sum(row["same_iid"] for row in results) / len(results),
        "same_branch_top1_rate": sum(row["same_branch"] for row in results) / len(results),
        "same_actor_family_top1_rate": sum(
            row["same_actor_family"] for row in results
        ) / len(results),
        "neighbors": results,
    }


def action_noop_contrast_evaluation(
    rows: Sequence[Mapping[str, Any]], representation: str
) -> dict[str, Any]:
    indexed = {
        (row["iid"], row["seed"], row["branch"]): row for row in rows
    }
    contrasts = []
    for row in rows:
        if row["branch"] not in ACTION_BRANCHES:
            continue
        noop = indexed[(row["iid"], row["seed"], "noop")]
        contrasts.append({
            **{key: row[key] for key in (
                "candidate_id", "iid", "seed", "branch", "actor_family",
                "analysis_split",
            )},
            "vector": unit(
                row["representations"][representation]
                - noop["representations"][representation]
            ),
        })
    prototypes = {
        (family, branch): centroid([
            row["vector"] for row in contrasts
            if row["analysis_split"] == "fit"
            and row["actor_family"] == family and row["branch"] == branch
        ])
        for family in FAMILIES for branch in ACTION_BRANCHES
    }
    confirmation = []
    for row in contrasts:
        if row["analysis_split"] != "confirmation":
            continue
        value = cosine(row["vector"], prototypes[(row["actor_family"], row["branch"])])
        confirmation.append({
            "candidate_id": row["candidate_id"],
            "iid": row["iid"],
            "actor_family": row["actor_family"],
            "branch": row["branch"],
            "cosine_to_fit_prototype": value,
            "positive": value > 0.0,
        })
    by_cell = {}
    for family in FAMILIES:
        for branch in ACTION_BRANCHES:
            values = [
                row["cosine_to_fit_prototype"] for row in confirmation
                if row["actor_family"] == family and row["branch"] == branch
            ]
            by_cell[f"{family}:{branch}"] = {
                "count": len(values),
                "mean_cosine": float(np.mean(values)),
                "min_cosine": min(values),
                "positive_count": sum(value > 0.0 for value in values),
            }
    by_source = {}
    for iid in sorted({row["iid"] for row in confirmation}):
        values = [
            row["cosine_to_fit_prototype"] for row in confirmation
            if row["iid"] == iid
        ]
        by_source[iid] = {
            "count": len(values),
            "mean_cosine": float(np.mean(values)),
            "min_cosine": min(values),
            "all_positive": all(value > 0.0 for value in values),
        }
    return {
        "fit_contrast_count": sum(
            row["analysis_split"] == "fit" for row in contrasts
        ),
        "confirmation_contrast_count": len(confirmation),
        "by_actor_family_and_branch": by_cell,
        "by_confirmation_source": by_source,
        "all_confirmation_contrasts_positive": all(
            row["positive"] for row in confirmation
        ),
        "confirmation": confirmation,
    }


def orthonormal_basis(rows: Sequence[np.ndarray], rank: int) -> np.ndarray:
    if not rows or rank <= 0:
        raise ValueError("subspace basis boundary differs")
    matrix = np.stack(rows, axis=0).astype(np.float32, copy=False)
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    numerical_rank = int(np.sum(singular_values > EPS))
    effective = min(rank, numerical_rank)
    if effective <= 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.float32)
    return np.ascontiguousarray(right[:effective].astype(np.float32))


def projection_score(vector: np.ndarray, basis: np.ndarray) -> float:
    if basis.ndim != 2 or basis.shape[1] != vector.size:
        raise ValueError("projection geometry differs")
    if basis.shape[0] == 0:
        return 0.0
    value = float(np.linalg.norm(basis @ vector))
    if not math.isfinite(value):
        raise ValueError("projection score is non-finite")
    return value


def typed_subspace_evaluation(
    rows: Sequence[Mapping[str, Any]], representation: str
) -> dict[str, Any]:
    """Evaluate a fixed rank sweep of fit-only family x branch subspaces."""
    indexed = {
        (row["iid"], row["seed"], row["branch"]): row for row in rows
    }
    contrasts = []
    for row in rows:
        if row["branch"] not in ACTION_BRANCHES:
            continue
        noop = indexed[(row["iid"], row["seed"], "noop")]
        contrasts.append({
            **{key: row[key] for key in (
                "candidate_id", "iid", "seed", "branch", "actor_family",
                "analysis_split",
            )},
            "route": f"{row['actor_family']}:{row['branch']}",
            "vector": unit(
                row["representations"][representation]
                - noop["representations"][representation]
            ),
        })
    routes = tuple(
        f"{family}:{branch}"
        for family in FAMILIES for branch in ACTION_BRANCHES
    )
    ranks = {}
    for rank in range(1, 5):
        bases = {
            route: orthonormal_basis([
                row["vector"] for row in contrasts
                if row["analysis_split"] == "fit" and row["route"] == route
            ], rank)
            for route in routes
        }
        results = []
        for row in contrasts:
            if row["analysis_split"] != "confirmation":
                continue
            scores = {
                route: projection_score(row["vector"], bases[route])
                for route in routes
            }
            predicted = max(routes, key=lambda route: (scores[route], route))
            correct = row["route"]
            correct_value = scores[correct]
            wrong_value = max(
                score for route, score in scores.items() if route != correct
            )
            predicted_family, predicted_branch = predicted.split(":")
            results.append({
                "candidate_id": row["candidate_id"],
                "iid": row["iid"],
                "correct_route": correct,
                "predicted_route": predicted,
                "route_correct": predicted == correct,
                "family_correct": predicted_family == row["actor_family"],
                "branch_correct": predicted_branch == row["branch"],
                "correct_projection": correct_value,
                "max_wrong_projection": wrong_value,
                "correct_margin": correct_value - wrong_value,
                "scores": scores,
            })
        route_count = sum(result["route_correct"] for result in results)
        family_count = sum(result["family_correct"] for result in results)
        branch_count = sum(result["branch_correct"] for result in results)
        margins = [result["correct_margin"] for result in results]
        correct_scores = [result["correct_projection"] for result in results]
        per_route = {}
        for route in routes:
            selected = [
                result for result in results if result["correct_route"] == route
            ]
            per_route[route] = {
                "count": len(selected),
                "route_correct_count": sum(
                    result["route_correct"] for result in selected
                ),
                "mean_correct_projection": float(np.mean([
                    result["correct_projection"] for result in selected
                ])),
                "mean_margin": float(np.mean([
                    result["correct_margin"] for result in selected
                ])),
            }
        ranks[str(rank)] = {
            "registered_rank": rank,
            "fit_rows_per_route": 4,
            "confirmation_count": len(results),
            "typed_route_accuracy": route_count / len(results),
            "actor_family_accuracy": family_count / len(results),
            "action_branch_accuracy": branch_count / len(results),
            "mean_correct_projection": float(np.mean(correct_scores)),
            "min_correct_projection": min(correct_scores),
            "mean_correct_margin": float(np.mean(margins)),
            "min_correct_margin": min(margins),
            "all_correct_margins_positive": all(value > 0.0 for value in margins),
            "per_route": per_route,
            "results": results,
        }
    return {
        "route_definition": "actor_family_x_action_branch",
        "routes": list(routes),
        "fit_only_basis": True,
        "confirmation_vectors_never_extend_basis": True,
        "rank_sweep_registered_before_rerun": [1, 2, 3, 4],
        "ranks": ranks,
    }


def load_sources(path: Path) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    path = path.resolve(strict=True)
    root = json.loads(path.read_text(encoding="ascii"))
    rows = root.get("rows")
    if root.get("schema_version") != SOURCE_SCHEMA or not isinstance(rows, list):
        die("source manifest differs")
    indexed = {row["iid"]: row for row in rows}
    if len(indexed) != EXPECTED_SOURCES:
        die("source closure differs")
    return indexed, {"path": str(path), "sha256": file_sha256(path)}


def load_candidates(
    attempts_root: Path, sources: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    root = attempts_root.resolve(strict=True)
    rows = []
    for receipt_path in sorted(root.glob(f"*/{RECEIPT_NAME}")):
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        candidate = receipt.get("candidate")
        if not isinstance(candidate, dict):
            die("candidate receipt differs")
        iid = candidate.get("iid")
        branch = candidate.get("branch")
        seed = candidate.get("seed")
        source = sources.get(iid)
        candidate_id = candidate.get("candidate_id")
        if (
            source is None or branch not in BRANCHES
            or seed not in source["rollout_seeds"]
            or candidate_id != f"saic-{iid}-{branch}-s{seed}"
            or receipt_path.parent.name != candidate_id
            or candidate.get("analysis_split") != source["analysis_split"]
            or candidate.get("actor_family") != source["actor_family"]
            or candidate.get("action_family_id") != source["action_family_id"]
            or candidate.get("event_verified") is not False
            or candidate.get("optimizer_authorized") is not False
        ):
            die(f"candidate boundary differs: {receipt_path}")
        video = receipt_path.parent / "t2v.mp4"
        declared = receipt.get("artifacts", {}).get("mp4", {})
        if (
            not video.is_file() or video.is_symlink()
            or declared.get("path") != str(video)
            or declared.get("sha256") != file_sha256(video)
        ):
            die(f"video binding differs: {receipt_path}")
        rows.append({
            "candidate_id": candidate_id,
            "iid": iid,
            "seed": seed,
            "branch": branch,
            "actor_family": candidate["actor_family"],
            "analysis_split": candidate["analysis_split"],
            "action_family_id": candidate["action_family_id"],
            "video_path": str(video),
            "video_sha256": declared["sha256"],
            "generation_receipt_path": str(receipt_path),
            "generation_receipt_sha256": file_sha256(receipt_path),
        })
    if len(rows) != EXPECTED_CANDIDATES:
        die(f"candidate count differs: {len(rows)}")
    observed = Counter((row["iid"], row["branch"]) for row in rows)
    for iid, source in sources.items():
        for branch in BRANCHES:
            if observed[(iid, branch)] != len(source["rollout_seeds"]):
                die("source x branch x seed closure differs")
    return rows


def load_scorer(path: Path) -> Any:
    path = path.resolve(strict=True)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("sealed_dino_scorer", path)
    if spec is None or spec.loader is None:
        die("cannot load sealed DINO scorer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-spec", type=Path, required=True)
    parser.add_argument("--visual-scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        die(f"output exists: {args.output}")
    sources, source_binding = load_sources(args.source_manifest)
    candidates = load_candidates(args.attempts_root, sources)
    scorer = load_scorer(args.visual_scorer)
    evaluator_spec = json.loads(args.evaluator_spec.read_text(encoding="utf-8"))
    checkpoint_evidence = scorer.verify_checkpoint_content(
        args.checkpoint, args.checkpoint_manifest, evaluator_spec=evaluator_spec
    )
    import torch

    if not torch.cuda.is_available():
        die("one GPU is required")
    device = torch.device("cuda:0")
    model, loading_counts = scorer.load_frozen_model(
        checkpoint_evidence, device=device
    )
    processor = checkpoint_evidence["processor"]
    model_spec = evaluator_spec["model"]
    runtime_rows = []
    for ordinal, row in enumerate(candidates):
        frames, decode = scorer.decode_exact81_rgb(
            row["video_path"], expected_sha256=row["video_sha256"]
        )
        _, pixels = scorer.preprocess_selected_rgb(frames, processor)
        global_feature, dense_feature, feature_evidence = scorer.extract_features(
            model, pixels, device=device,
            num_register_tokens=model_spec["num_register_tokens"],
            evaluation_image_size=model_spec["preprocessor_golden_output_shape"][-1],
            patch_size=model_spec["patch_size"],
        )
        representations = temporal_representations(
            global_feature.numpy(), dense_feature.numpy()
        )
        runtime_rows.append({
            **row,
            "ordinal": ordinal,
            "decoded_rgb_sha256": decode["decoded_rgb_sha256"],
            "selected_rgb_sha256": decode["selected_rgb_sha256"],
            "global_feature_sha256": feature_evidence["global_feature_sha256"],
            "dense_feature_sha256": feature_evidence["dense_feature_sha256"],
            "representation_sha256": {
                name: array_sha256(value)
                for name, value in representations.items()
            },
            "representations": representations,
        })
        print(json.dumps({
            "ordinal": ordinal, "candidate_id": row["candidate_id"]
        }, sort_keys=True), flush=True)
    names = tuple(sorted(runtime_rows[0]["representations"]))
    diagnostics = {
        name: {
            "dimension": int(runtime_rows[0]["representations"][name].size),
            "heldout_branch_nearest_centroid": branch_centroid_evaluation(
                runtime_rows, name
            ),
            "all_row_nearest_neighbor": nearest_neighbor_diagnostic(
                runtime_rows, name
            ),
            "heldout_action_minus_noop_transport": (
                action_noop_contrast_evaluation(runtime_rows, name)
            ),
            "fit_only_typed_subspace_transport": typed_subspace_evaluation(
                runtime_rows, name
            ),
        }
        for name in names
    }
    public_rows = [{
        key: value for key, value in row.items() if key != "representations"
    } for row in runtime_rows]
    unsigned = {
        "schema_version": SCHEMA,
        "status": "frozen_representation_diagnostic_no_authority",
        "source_binding": source_binding,
        "attempts_root": str(args.attempts_root.resolve(strict=True)),
        "candidate_count": len(public_rows),
        "candidate_rows": public_rows,
        "candidate_rows_digest": object_sha256(public_rows),
        "representation_names": list(names),
        "diagnostics": diagnostics,
        "frozen_visual_model": {
            "checkpoint": str(args.checkpoint.resolve(strict=True)),
            "checkpoint_manifest": str(args.checkpoint_manifest.resolve(strict=True)),
            "checkpoint_manifest_sha256": file_sha256(args.checkpoint_manifest),
            "evaluator_spec": str(args.evaluator_spec.resolve(strict=True)),
            "evaluator_spec_sha256": file_sha256(args.evaluator_spec),
            "visual_scorer": str(args.visual_scorer.resolve(strict=True)),
            "visual_scorer_sha256": file_sha256(args.visual_scorer),
            "all_parameters_frozen": True,
            "loading_counts": loading_counts,
            "runtime_versions": scorer.runtime_versions(),
        },
        "split_policy": {
            "fit_sources_define_centroids_and_contrast_prototypes": True,
            "confirmation_sources_contribute_to_fit": False,
            "all_registered_rows_consumed": True,
            "seed_selection_performed": False,
        },
        "limitations": {
            "dinov2_is_image_not_video_pretraining": True,
            "global_and_dense_features_are_proxy_representations": True,
            "no_human_event_labels_consumed": True,
            "no_threshold_registered": True,
            "decoded_video_review_still_required": True,
        },
        "authority": AUTHORITY,
    }
    output = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "output": str(args.output),
        "receipt_digest": output["receipt_digest"],
        "candidate_count": len(public_rows),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
