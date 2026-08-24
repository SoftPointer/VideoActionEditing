#!/usr/bin/env python3
"""Calibrate one fit-selected two-head event representation without reselection.

The two heads, ranks, and mixture weights are read from a content-addressed
fit-only receipt.  Bases are fit once on all admitted fit sources, then applied
unchanged to the disjoint calibration cells.  The gate boundary is the
semantic ordinal boundary (margin > 0), not a tuned raw-embedding threshold.
This program never inspects confirmation data and never updates Bernini.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parent
SELECTION_NAME = "select_multiscene_factorial_representation_v1.py"
SELECTION_SHA256 = "7e82f5ba30f0a69cf8b044d1e86ff0a582cc77883d5dfda0ab1ff478f7bb1f9c"
SELECTION_PATH = METHOD_ROOT / SELECTION_NAME
if (
    not SELECTION_PATH.is_file() or SELECTION_PATH.is_symlink()
    or hashlib.sha256(SELECTION_PATH.read_bytes()).hexdigest() != SELECTION_SHA256
):
    raise RuntimeError("pinned multi-scene selection dependency differs")

import select_multiscene_factorial_representation_v1 as selection  # noqa: E402


SCHEMA_VERSION = "bernini-frozen-multiscene-factorial-calibration-v1"
TWO_HEAD_SCHEMA = "bernini-multiscene-factorial-two-head-selection-v1"
MARGINS = (
    "forward_gt_noop",
    "forward_gt_reverse",
    "forward_gt_incomplete",
    "forward_gt_abs_nuisance",
)
_CANDIDATE = re.compile(
    r"(?P<representation>[a-z_]+)-ar(?P<action_rank>[12])-nr(?P<nuisance_rank>[024])\Z"
)
AUTHORITY_BASE = {
    "fit_only_model_fitting": True,
    "calibration_accessed": True,
    "confirmation_accessed": False,
    "heads_or_weights_reselected": False,
    "raw_embedding_threshold_tuned": False,
    "ordinal_zero_boundary_frozen": True,
    "training_target_authorized": False,
    "optimizer_step_authorized": False,
    "method_success_claimed": False,
}


class FrozenCalibrationError(RuntimeError):
    """Raised before an incomplete or adaptive calibration can be emitted."""


def fail(message: str) -> NoReturn:
    raise FrozenCalibrationError(message)


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FrozenCalibrationError(f"cannot read {label}") from error
    if type(value) is not dict:
        fail(f"{label} must contain one object")
    return value


def _candidate(value: str) -> dict[str, Any]:
    if type(value) is not str:
        fail("candidate ID differs")
    match = _CANDIDATE.fullmatch(value)
    if match is None or match.group("representation") not in selection.REPRESENTATIONS:
        fail("candidate ID differs")
    return {
        "candidate_id": value,
        "representation": match.group("representation"),
        "action_rank": int(match.group("action_rank")),
        "nuisance_rank": int(match.group("nuisance_rank")),
    }


def frozen_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    selected = value.get("selected")
    authority = value.get("authority")
    if (
        value.get("schema_version") != TWO_HEAD_SCHEMA
        or not isinstance(selected, Mapping)
        or not isinstance(authority, Mapping)
        or authority.get("fit_only_representation_selection") is not True
        or authority.get("calibration_accessed") is not False
        or authority.get("confirmation_accessed") is not False
        or authority.get("calibration_may_not_reselect_heads_or_weights") is not True
        or authority.get("optimizer_step_authorized") is not False
        or selected.get("all_four_margins_all_six_folds") is not True
    ):
        fail("fit-only two-head selection receipt differs")
    head_a = _candidate(selected.get("head_a"))
    head_b = _candidate(selected.get("head_b"))
    weight_a = float(selected.get("weight_a", math.nan))
    weight_b = float(selected.get("weight_b", math.nan))
    if (
        not math.isfinite(weight_a) or not math.isfinite(weight_b)
        or not 0.0 < weight_a < 1.0 or not 0.0 < weight_b < 1.0
        or not math.isclose(weight_a + weight_b, 1.0, abs_tol=1.0e-12)
        or head_a["candidate_id"] == head_b["candidate_id"]
    ):
        fail("frozen two-head weights differ")
    return {
        "head_a": head_a,
        "head_b": head_b,
        "weight_a": weight_a,
        "weight_b": weight_b,
    }


def _family_cells(
    feature_bank: Mapping[str, Mapping[str, np.ndarray]], family: str
) -> dict[str, Mapping[str, np.ndarray]]:
    result = {
        key: value for key, value in feature_bank.items()
        if key.startswith(f"{family}:")
    }
    if len(result) != 3:
        fail(f"{family} source closure differs")
    return result


def fit_family_model(
    features: Mapping[str, Mapping[str, np.ndarray]], head: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    vectors = {
        cell: {
            branch: selection.representation(value, head["representation"])
            for branch, value in branches.items()
        }
        for cell, branches in features.items()
    }
    nuisance_rows = []
    for cell in sorted(vectors):
        noop = vectors[cell]["normalized_noop"]
        nuisance_rows.extend(
            vectors[cell][branch] - noop for branch in selection.NUISANCE_BRANCHES
        )
    nuisance_basis = selection._row_basis(
        np.stack(nuisance_rows), int(head["nuisance_rank"])
    )
    forward_rows = []
    for cell in sorted(vectors):
        delta = vectors[cell]["normalized_forward"] - vectors[cell]["normalized_noop"]
        forward_rows.append(selection._project_null(delta, nuisance_basis))
    forward_matrix = np.stack(forward_rows)
    action_basis = selection._row_basis(forward_matrix, int(head["action_rank"]))
    if action_basis.shape[1] == 0:
        fail("fit action basis is degenerate")
    centroid = (forward_matrix @ action_basis).mean(axis=0)
    if np.linalg.norm(centroid) <= 1.0e-12:
        fail("fit action centroid is degenerate")
    return {
        "nuisance_basis": nuisance_basis,
        "action_basis": action_basis,
        "centroid": centroid,
    }


def score_cell(
    features: Mapping[str, np.ndarray], head: Mapping[str, Any],
    model: Mapping[str, np.ndarray],
) -> dict[str, float]:
    vectors = {
        branch: selection.representation(value, head["representation"])
        for branch, value in features.items()
    }
    noop = vectors["normalized_noop"]
    action_basis = model["action_basis"]
    centroid = model["centroid"]
    centroid_norm = np.linalg.norm(centroid)
    scores = {}
    for branch in selection.BRANCHES:
        delta = selection._project_null(
            vectors[branch] - noop, model["nuisance_basis"]
        )
        norm = np.linalg.norm(delta)
        score = 0.0 if norm <= 1.0e-12 else float(
            np.dot(delta @ action_basis, centroid) / (norm * centroid_norm)
        )
        if not math.isfinite(score):
            fail("calibration score is non-finite")
        scores[branch] = score
    return scores


def ordinal_margins(scores: Mapping[str, float]) -> dict[str, float]:
    nuisance_abs_max = max(abs(scores[name]) for name in selection.NUISANCE_BRANCHES)
    result = {
        "forward_gt_noop": scores["normalized_forward"] - scores["normalized_noop"],
        "forward_gt_reverse": scores["normalized_forward"] - scores["reverse_from_forward"],
        "forward_gt_incomplete": scores["normalized_forward"] - scores["incomplete_phasewarp"],
        "forward_gt_abs_nuisance": scores["normalized_forward"] - nuisance_abs_max,
    }
    if any(not math.isfinite(value) for value in result.values()):
        fail("calibration margin is non-finite")
    return result


def calibrate(
    fit_bank: Mapping[str, Mapping[str, np.ndarray]],
    calibration_bank: Mapping[str, Mapping[str, np.ndarray]],
    frozen: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    if set(fit_bank) & set(calibration_bank):
        fail("fit and calibration source cells overlap")
    models: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for family in selection.FAMILIES:
        fit_cells = _family_cells(fit_bank, family)
        calibration_cells = _family_cells(calibration_bank, family)
        family_models = {}
        for label in ("head_a", "head_b"):
            key = f"{label}:{family}"
            family_models[label] = fit_family_model(fit_cells, frozen[label])
            models[key] = family_models[label]
        for cell in sorted(calibration_cells):
            scores_by_head = {
                label: score_cell(calibration_cells[cell], frozen[label], family_models[label])
                for label in ("head_a", "head_b")
            }
            mixed = {
                branch: (
                    frozen["weight_a"] * scores_by_head["head_a"][branch]
                    + frozen["weight_b"] * scores_by_head["head_b"][branch]
                )
                for branch in selection.BRANCHES
            }
            margins = ordinal_margins(mixed)
            rows.append({
                "family": family,
                "cell": cell,
                "scores_by_head": scores_by_head,
                "mixed_scores": mixed,
                "margins": margins,
                "passes": {name: value > 0.0 for name, value in margins.items()},
            })
    minimum_margins = {
        name: min(row["margins"][name] for row in rows) for name in MARGINS
    }
    pass_counts = {
        name: sum(row["passes"][name] for row in rows) for name in MARGINS
    }
    all_pass = all(count == 6 for count in pass_counts.values())
    result = {
        "cell_count": len(rows),
        "ordinal_thresholds": {name: 0.0 for name in MARGINS},
        "threshold_rule": (
            "semantic_strict_order_boundary_zero;calibration_validates_positive_slack;"
            "no_raw_embedding_threshold_search"
        ),
        "pass_counts": pass_counts,
        "minimum_margins": minimum_margins,
        "all_four_margins_all_six_cells": all_pass,
        "confirmation_evaluation_authorized": all_pass,
        "cells": rows,
    }
    return result, models


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def model_manifest(models: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    return {
        key: {
            name: {"shape": list(value.shape), "float64_bytes_sha256": _array_sha256(value)}
            for name, value in sorted(parts.items())
        }
        for key, parts in sorted(models.items())
    }


def summary(report: Mapping[str, Any]) -> str:
    result = report["calibration"]
    lines = [
        "# Frozen two-head factorial calibration", "",
        "Fit-selected heads and weights were applied unchanged to six disjoint calibration sources.", "",
        "| margin | pass / 6 | worst calibration margin | frozen boundary |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in MARGINS:
        lines.append(
            f"| {name} | {result['pass_counts'][name]}/6 | "
            f"{result['minimum_margins'][name]:.6f} | 0.000000 |"
        )
    lines += [
        "",
        f"Confirmation evaluation authorized: `{result['confirmation_evaluation_authorized']}`.",
        "",
        "The zero boundary is defined by the ordinal claim itself. Calibration only tests positive slack; it does not search an embedding-scale cutoff. Optimizer authority remains closed.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--two-head-receipt", type=Path, required=True)
    parser.add_argument("--expected-two-head-receipt-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-spec", type=Path, required=True)
    parser.add_argument("--visual-scorer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_sha = selection.file_sha256(Path(__file__).resolve())
    if source_sha != args.expected_source_sha256:
        fail("calibration source SHA-256 differs")
    if selection.file_sha256(args.two_head_receipt) != args.expected_two_head_receipt_sha256:
        fail("two-head selection receipt SHA-256 differs")
    receipt = _read_object(args.two_head_receipt, label="two-head selection receipt")
    frozen = frozen_selection(receipt)
    fit_cells = selection.sealed_cells(args.fit_root)
    calibration_cells = selection.sealed_cells(args.calibration_root)

    scorer = selection.v3.base.load_module(
        args.visual_scorer, "frozen_multiscene_calibration_dino_scorer"
    )
    spec = json.loads(args.evaluator_spec.read_text(encoding="utf-8"))
    checkpoint = scorer.verify_checkpoint_content(
        args.checkpoint, args.checkpoint_manifest, evaluator_spec=spec
    )
    processor = checkpoint.pop("processor")

    import av
    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        fail("calibration requires exactly one visible GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    model, loading_counts = scorer.load_frozen_model(checkpoint, device=device)
    if any(loading_counts.values()):
        fail("frozen DINO loading counts differ")

    def extract(cells: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        bank: dict[str, Any] = {}
        bindings: dict[str, Any] = {}
        for cell in cells:
            key = f"{cell['family']}:{cell['cell_id']}"
            bank[key], bindings[key] = {}, {}
            for branch in selection.BRANCHES:
                row = cell["media"][branch]
                frames, decode = scorer.decode_exact81_rgb(
                    row["path"], expected_sha256=row["sha256"]
                )
                _, normalized = scorer.preprocess_selected_rgb(frames, processor)
                global_feature, _, evidence = scorer.extract_features(
                    model, normalized, device=device, num_register_tokens=0
                )
                if selection.file_sha256(row["path"]) != row["sha256"]:
                    fail("video changed while extracting features")
                bank[key][branch] = global_feature.numpy().copy()
                bindings[key][branch] = {**row, "decode": decode, "features": evidence}
        return bank, bindings

    fit_bank, fit_bindings = extract(fit_cells)
    calibration_bank, calibration_bindings = extract(calibration_cells)
    result, models = calibrate(fit_bank, calibration_bank, frozen)
    output = args.output_root
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        fail("output root must be fresh")
    output.mkdir(mode=0o700)
    arrays = {
        f"{key.replace(':', '__')}__{name}": value
        for key, parts in sorted(models.items()) for name, value in sorted(parts.items())
    }
    np.savez(output / "frozen_fit_model.npz", **arrays)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_sha,
        "selection_dependency_sha256": SELECTION_SHA256,
        "two_head_receipt": str(args.two_head_receipt.resolve()),
        "two_head_receipt_sha256": args.expected_two_head_receipt_sha256,
        "frozen_selection": frozen,
        "fit_root": str(args.fit_root.resolve()),
        "calibration_root": str(args.calibration_root.resolve()),
        "fit_cells": [{key: value for key, value in cell.items() if key != "media"} for cell in fit_cells],
        "calibration_cells": [{key: value for key, value in cell.items() if key != "media"} for cell in calibration_cells],
        "calibration": result,
        "fit_model_manifest": model_manifest(models),
        "frozen_fit_model_npz_sha256": selection.file_sha256(output / "frozen_fit_model.npz"),
        "media_bindings": {"fit": fit_bindings, "calibration": calibration_bindings},
        "evaluator": {
            "checkpoint_root": str(args.checkpoint.resolve()),
            "checkpoint_manifest_sha256": selection.file_sha256(args.checkpoint_manifest),
            "evaluator_spec_sha256": selection.file_sha256(args.evaluator_spec),
            "visual_scorer_sha256": selection.file_sha256(args.visual_scorer),
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
        "authority": {
            **AUTHORITY_BASE,
            "confirmation_evaluation_authorized": result["confirmation_evaluation_authorized"],
        },
    }
    report = {**unsigned, "receipt_digest": selection.object_sha256(unsigned)}
    (output / "report.json").write_bytes(selection.canonical_bytes(report) + b"\n")
    (output / "summary.md").write_text(summary(report), encoding="utf-8")
    print(json.dumps({
        "confirmation_evaluation_authorized": result["confirmation_evaluation_authorized"],
        "minimum_margins": result["minimum_margins"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FrozenCalibrationError, selection.MultiSceneSelectionError) as error:
        print(f"[frozen-multiscene-calibration] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
