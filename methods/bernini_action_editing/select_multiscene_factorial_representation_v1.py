#!/usr/bin/env python3
"""Fit-only multi-scene selection of frozen-DINO temporal event bases.

Each action family contains exactly three decoded, human-admitted sources.  A
candidate is a typed temporal representation, rank-1/2 family event basis, and
rank-0/2/4 observed nuisance nullspace.  Selection is source leave-one-out and
uses only ordinal margins; no confirmation example or absolute threshold is
consulted.  This program never updates Bernini or any evaluator parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parent
BASE_NAME = "probe_factorial_axis_controls_v3_dinov2.py"
BASE_SHA256 = "04fb0f2daf8d2ce5eeda6e4e58593ddff94ed28acd0469f4940f6a3479e65659"
BASE_PATH = METHOD_ROOT / BASE_NAME
if (
    not BASE_PATH.is_file()
    or BASE_PATH.is_symlink()
    or hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256
):
    raise RuntimeError("pinned factorial-axis v3 dependency differs")

import probe_factorial_axis_controls_v3_dinov2 as v3  # noqa: E402


SCHEMA_VERSION = "bernini-multiscene-factorial-representation-selection-v1"
FRAME_INDICES = tuple(range(0, 81, 5))
FAMILIES = ("dog", "human")
BRANCHES = (
    "normalized_noop",
    "normalized_forward",
    "reverse_from_forward",
    "incomplete_phasewarp",
    "camera_right_push",
    "camera_center_push",
    "camera_vertical_push",
    "camera_center_pull",
    "appearance_hue_ramp",
)
NUISANCE_BRANCHES = BRANCHES[4:]
REPRESENTATIONS = (
    "appearance_mean",
    "centered_trajectory",
    "endpoint_arrow",
    "speed_profile",
    "temporal_self_similarity",
    "velocity_trajectory",
)
ACTION_RANKS = (1, 2)
NUISANCE_RANKS = (0, 2, 4)
AUTHORITY = {
    "fit_only_representation_selection": True,
    "source_leave_one_out": True,
    "ordinal_margins_only": True,
    "absolute_threshold_defined": False,
    "confirmation_accessed": False,
    "representation_selection_authorized": True,
    "training_target_authorized": False,
    "optimizer_or_parameter_update_authorized": False,
    "method_success_claimed": False,
}


class MultiSceneSelectionError(RuntimeError):
    """Raised before incomplete inputs or invalid selection can be emitted."""


def fail(message: str) -> NoReturn:
    raise MultiSceneSelectionError(message)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def finite(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} is non-finite")
    return result


def _read_cell(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MultiSceneSelectionError(f"cannot read cell receipt: {path}") from error
    if (
        type(value) is not dict
        or value.get("schema_version") != "bernini-multiscene-factorial-cell-r1"
        or value.get("family") not in FAMILIES
        or value.get("branches") != len(BRANCHES)
        or value.get("authority", {}).get("optimizer_step_authorized") is not False
    ):
        fail(f"cell receipt differs: {path}")
    return value


def sealed_cells(root: Path) -> list[dict[str, Any]]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        fail("input root differs")
    result = []
    for cell_root in sorted(path for path in resolved.iterdir() if path.is_dir()):
        if cell_root.is_symlink() or not (cell_root / "COMPLETE").is_file():
            fail(f"cell completion differs: {cell_root}")
        cell = _read_cell(cell_root / "cell.json")
        if cell_root.name != cell["cell_id"]:
            fail("cell directory identity differs")
        declared: dict[str, str] = {}
        manifest = cell_root / "media.sha256"
        if not manifest.is_file() or manifest.is_symlink():
            fail("media manifest differs")
        for line in manifest.read_text(encoding="ascii").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) != 2 or len(fields[0]) != 64:
                fail("media manifest row differs")
            declared[Path(fields[1]).name] = fields[0]
        expected = {f"{branch}.mp4" for branch in BRANCHES}
        if set(declared) != expected:
            fail("cell branch closure differs")
        media = {}
        for branch in BRANCHES:
            path = cell_root / f"{branch}.mp4"
            if (
                not path.is_file() or path.is_symlink()
                or file_sha256(path) != declared[path.name]
            ):
                fail(f"cell media binding differs: {path}")
            media[branch] = {"path": str(path), "sha256": declared[path.name]}
        result.append({**cell, "root": str(cell_root), "media": media})
    counts = {family: sum(row["family"] == family for row in result) for family in FAMILIES}
    if len(result) != 6 or counts != {"dog": 3, "human": 3}:
        fail("fit6 family/source closure differs")
    return result


def representation(features: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != len(FRAME_INDICES):
        fail("feature trajectory geometry differs")
    if name == "appearance_mean":
        result = value.mean(axis=0)
    elif name == "centered_trajectory":
        result = (value - value.mean(axis=0, keepdims=True)).reshape(-1)
    elif name == "endpoint_arrow":
        result = value[-1] - value[0]
    elif name == "speed_profile":
        result = np.linalg.norm(np.diff(value, axis=0), axis=1)
    elif name == "temporal_self_similarity":
        norms = np.linalg.norm(value, axis=1, keepdims=True)
        normalized = value / np.maximum(norms, 1.0e-12)
        result = (normalized @ normalized.T).reshape(-1)
    elif name == "velocity_trajectory":
        result = np.diff(value, axis=0).reshape(-1)
    else:
        fail(f"unknown representation: {name}")
    if result.ndim != 1 or not np.isfinite(result).all():
        fail("representation vector differs")
    return result


def _row_basis(rows: np.ndarray, rank: int) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all() or rank < 0:
        fail("basis input differs")
    if rank == 0:
        return np.zeros((value.shape[1], 0), dtype=np.float64)
    _, singular, vh = np.linalg.svd(value, full_matrices=False)
    available = int(np.sum(singular > 1.0e-10))
    keep = min(rank, available)
    return vh[:keep].T.copy() if keep else np.zeros((value.shape[1], 0), dtype=np.float64)


def _project_null(value: np.ndarray, nuisance_basis: np.ndarray) -> np.ndarray:
    if nuisance_basis.shape[1] == 0:
        return value.copy()
    return value - nuisance_basis @ (nuisance_basis.T @ value)


def loo_candidate(
    vectors: Mapping[str, Mapping[str, np.ndarray]], *, action_rank: int,
    nuisance_rank: int,
) -> dict[str, Any]:
    cell_ids = sorted(vectors)
    if len(cell_ids) != 3:
        fail("LOO family requires three source cells")
    folds = []
    for heldout in cell_ids:
        training = [cell for cell in cell_ids if cell != heldout]
        nuisance_rows = []
        for cell in training:
            noop = vectors[cell]["normalized_noop"]
            nuisance_rows.extend(vectors[cell][branch] - noop for branch in NUISANCE_BRANCHES)
        nuisance_basis = _row_basis(np.stack(nuisance_rows), nuisance_rank)
        forward_rows = []
        for cell in training:
            delta = vectors[cell]["normalized_forward"] - vectors[cell]["normalized_noop"]
            forward_rows.append(_project_null(delta, nuisance_basis))
        action_basis = _row_basis(np.stack(forward_rows), action_rank)
        if action_basis.shape[1] == 0:
            fail("action basis is degenerate")
        forward_matrix = np.stack(forward_rows)
        centroid = (forward_matrix @ action_basis).mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm <= 1.0e-12:
            fail("action centroid is degenerate")
        scores = {}
        noop = vectors[heldout]["normalized_noop"]
        for branch in BRANCHES:
            delta = _project_null(vectors[heldout][branch] - noop, nuisance_basis)
            norm = np.linalg.norm(delta)
            scores[branch] = 0.0 if norm <= 1.0e-12 else finite(
                np.dot(delta @ action_basis, centroid) / (norm * centroid_norm),
                label="candidate branch score",
            )
        nuisance_abs_max = max(abs(scores[name]) for name in NUISANCE_BRANCHES)
        margins = {
            "forward_gt_noop": scores["normalized_forward"] - scores["normalized_noop"],
            "forward_gt_reverse": scores["normalized_forward"] - scores["reverse_from_forward"],
            "forward_gt_incomplete": scores["normalized_forward"] - scores["incomplete_phasewarp"],
            "forward_gt_abs_nuisance": scores["normalized_forward"] - nuisance_abs_max,
        }
        folds.append(
            {
                "heldout_cell": heldout,
                "training_cells": training,
                "realized_nuisance_rank": nuisance_basis.shape[1],
                "realized_action_rank": action_basis.shape[1],
                "scores": scores,
                "margins": {key: finite(value, label=key) for key, value in margins.items()},
                "passes": {key: value > 0.0 for key, value in margins.items()},
            }
        )
    return {"folds": folds}


def select_candidates(feature_bank: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    families = {
        family: {cell: feature_bank[cell] for cell in feature_bank if cell.startswith(f"{family}:")}
        for family in FAMILIES
    }
    candidates = []
    for name in REPRESENTATIONS:
        vectors = {
            family: {
                cell: {branch: representation(features, name) for branch, features in branches.items()}
                for cell, branches in family_cells.items()
            }
            for family, family_cells in families.items()
        }
        for action_rank in ACTION_RANKS:
            for nuisance_rank in NUISANCE_RANKS:
                family_results = {
                    family: loo_candidate(
                        vectors[family], action_rank=action_rank,
                        nuisance_rank=nuisance_rank,
                    )
                    for family in FAMILIES
                }
                folds = [fold for result in family_results.values() for fold in result["folds"]]
                pass_counts = {
                    margin: sum(fold["passes"][margin] for fold in folds)
                    for margin in (
                        "forward_gt_noop", "forward_gt_reverse",
                        "forward_gt_incomplete", "forward_gt_abs_nuisance",
                    )
                }
                min_margins = {
                    margin: min(fold["margins"][margin] for fold in folds)
                    for margin in pass_counts
                }
                candidate_id = f"{name}-ar{action_rank}-nr{nuisance_rank}"
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "representation": name,
                        "requested_action_rank": action_rank,
                        "requested_nuisance_rank": nuisance_rank,
                        "families": family_results,
                        "pass_counts": pass_counts,
                        "minimum_margins": min_margins,
                    }
                )
    if len(candidates) != len(REPRESENTATIONS) * len(ACTION_RANKS) * len(NUISANCE_RANKS):
        fail("candidate grid closure differs")
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        counts = row["pass_counts"]
        margins = row["minimum_margins"]
        return (
            counts["forward_gt_noop"], counts["forward_gt_reverse"],
            counts["forward_gt_incomplete"], counts["forward_gt_abs_nuisance"],
            margins["forward_gt_noop"], margins["forward_gt_reverse"],
            margins["forward_gt_incomplete"], margins["forward_gt_abs_nuisance"],
            -row["requested_nuisance_rank"], -row["requested_action_rank"],
            row["candidate_id"],
        )
    selected = max(candidates, key=key)
    return {
        "candidate_count": len(candidates),
        "selection_rule": (
            "lexicographic_fit_LOO_pass_counts_then_worst_margins_then_lower_ranks;"
            "no_absolute_threshold"
        ),
        "selected_candidate_id": selected["candidate_id"],
        "selected_all_four_margins_all_six_folds": all(
            count == 6 for count in selected["pass_counts"].values()
        ),
        "candidates": candidates,
    }


def summary(report: Mapping[str, Any]) -> str:
    selection = report["selection"]
    selected = next(
        row for row in selection["candidates"]
        if row["candidate_id"] == selection["selected_candidate_id"]
    )
    lines = [
        "# Multi-scene factorial representation selection", "",
        "Fit-only source leave-one-out; frozen DINO; no optimizer and no confirmation access.", "",
        f"Selected: `{selected['candidate_id']}`", "",
        "| margin | pass / 6 | worst margin |",
        "| --- | ---: | ---: |",
    ]
    for name, count in selected["pass_counts"].items():
        lines.append(f"| {name} | {count}/6 | {selected['minimum_margins'][name]:.6f} |")
    lines += [
        "", f"All four ordinal margins on all six folds: `{selection['selected_all_four_margins_all_six_folds']}`.",
        "", "This selects a fit representation candidate only. Calibration, decoded no-update evaluation, and optimizer authorization remain closed.", "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-spec", type=Path, required=True)
    parser.add_argument("--visual-scorer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_sha = file_sha256(Path(__file__).resolve())
    if source_sha != args.expected_source_sha256:
        fail("selection source SHA-256 differs")
    cells = sealed_cells(args.input_root)
    scorer = v3.base.load_module(args.visual_scorer, "multiscene_factorial_dino_scorer")
    spec = json.loads(args.evaluator_spec.read_text(encoding="utf-8"))
    checkpoint = scorer.verify_checkpoint_content(
        args.checkpoint, args.checkpoint_manifest, evaluator_spec=spec
    )
    processor = checkpoint.pop("processor")

    import av
    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        fail("selection requires exactly one visible GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    model, loading_counts = scorer.load_frozen_model(checkpoint, device=device)
    if any(loading_counts.values()):
        fail("frozen DINO loading counts differ")

    feature_bank: dict[str, dict[str, np.ndarray]] = {}
    bindings = {}
    for cell in cells:
        key = f"{cell['family']}:{cell['cell_id']}"
        feature_bank[key], bindings[key] = {}, {}
        for branch in BRANCHES:
            row = cell["media"][branch]
            frames, decode = scorer.decode_exact81_rgb(row["path"], expected_sha256=row["sha256"])
            _, normalized = scorer.preprocess_selected_rgb(frames, processor)
            global_feature, _, evidence = scorer.extract_features(
                model, normalized, device=device, num_register_tokens=0
            )
            if file_sha256(row["path"]) != row["sha256"]:
                fail("video changed while extracting features")
            feature_bank[key][branch] = global_feature.numpy().copy()
            bindings[key][branch] = {**row, "decode": decode, "features": evidence}

    selection = select_candidates(feature_bank)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "selection_source_sha256": source_sha,
        "base_dependency_sha256": BASE_SHA256,
        "input_root": str(args.input_root.resolve()),
        "cell_count": len(cells),
        "cells": [{key: value for key, value in cell.items() if key != "media"} for cell in cells],
        "selected_frame_indices": list(FRAME_INDICES),
        "branches": list(BRANCHES),
        "selection": selection,
        "media_bindings": bindings,
        "evaluator": {
            "checkpoint_root": str(args.checkpoint.resolve()),
            "checkpoint_manifest_sha256": file_sha256(args.checkpoint_manifest),
            "evaluator_spec_sha256": file_sha256(args.evaluator_spec),
            "visual_scorer_sha256": file_sha256(args.visual_scorer),
            "loading_counts": loading_counts,
            "model_frozen_eval": not model.training and not any(
                parameter.requires_grad for parameter in model.parameters()
            ),
            "runtime": {
                "python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "av": av.__version__,
                "numpy": np.__version__,
            },
        },
        "authority": dict(AUTHORITY),
    }
    report = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    output = args.output_root
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        fail("output root must be fresh")
    output.mkdir(mode=0o700)
    (output / "report.json").write_bytes(canonical_bytes(report) + b"\n")
    (output / "summary.md").write_text(summary(report), encoding="utf-8")
    print(json.dumps({"selected": selection["selected_candidate_id"], "candidates": selection["candidate_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MultiSceneSelectionError, v3.AxisV3ProbeError) as error:
        print(f"[multiscene-representation] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
