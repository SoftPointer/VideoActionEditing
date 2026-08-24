#!/usr/bin/env python3
"""Fit-only selection and disclosed replay of a source-local anchor quotient.

Unlike the failed family basis, every cell uses its own admitted self-generated
forward as the rank-1 action anchor.  Camera/appearance controls define a
within-cell nuisance nullspace.  Scores are signed coefficients relative to
the projected anchor, so incomplete magnitude and time direction are retained.
The second population is an explicitly disclosed development replay, not a new
independent calibration.
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
BASE_NAME = "select_multiscene_factorial_representation_v1.py"
BASE_SHA256 = "7e82f5ba30f0a69cf8b044d1e86ff0a582cc77883d5dfda0ab1ff478f7bb1f9c"
BASE_PATH = METHOD_ROOT / BASE_NAME
if (
    not BASE_PATH.is_file() or BASE_PATH.is_symlink()
    or hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256
):
    raise RuntimeError("pinned multi-scene representation dependency differs")

import select_multiscene_factorial_representation_v1 as base  # noqa: E402


SCHEMA_VERSION = "bernini-source-local-anchor-quotient-probe-v1"
CELL_SCHEMA = "bernini-source-local-hold-cell-r1"
BRANCHES = (
    "normalized_noop", "normalized_forward", "reverse_from_forward",
    "incomplete_hold16", "incomplete_hold24", "camera_right_push",
    "camera_center_push", "camera_vertical_push", "camera_center_pull",
    "appearance_hue_ramp",
)
NUISANCE_BRANCHES = BRANCHES[5:]
MARGINS = (
    "forward_gt_noop", "forward_gt_reverse", "forward_gt_incomplete_hold16",
    "forward_gt_incomplete_hold24", "forward_gt_abs_nuisance",
)
NUISANCE_RANKS = (0, 2, 4)
AUTHORITY = {
    "fit_only_representation_selection": True,
    "source_local_self_generated_forward_anchor": True,
    "development_calibration_replay_accessed": True,
    "independent_calibration_claimed": False,
    "confirmation_accessed": False,
    "training_target_authorized": False,
    "optimizer_step_authorized": False,
    "method_success_claimed": False,
}


class SourceLocalProbeError(RuntimeError):
    """Raised before invalid source-local evidence can be emitted."""


def fail(message: str) -> NoReturn:
    raise SourceLocalProbeError(message)


def sealed_cells(root: Path, *, expected_split: str) -> list[dict[str, Any]]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        fail("input root differs")
    result = []
    for cell_root in sorted(path for path in resolved.iterdir() if path.is_dir()):
        if cell_root.is_symlink() or not (cell_root / "COMPLETE").is_file():
            fail(f"cell completion differs: {cell_root}")
        try:
            cell = json.loads((cell_root / "cell.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SourceLocalProbeError(f"cannot read cell: {cell_root}") from error
        if (
            type(cell) is not dict or cell.get("schema_version") != CELL_SCHEMA
            or cell.get("cell_id") != cell_root.name
            or cell.get("family") not in base.FAMILIES
            or cell.get("analysis_split") != expected_split
            or cell.get("branches") != len(BRANCHES)
            or cell.get("authority", {}).get("optimizer_step_authorized") is not False
        ):
            fail(f"cell receipt differs: {cell_root}")
        declared = {}
        for line in (cell_root / "media.sha256").read_text(encoding="ascii").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) != 2 or len(fields[0]) != 64:
                fail("media manifest row differs")
            declared[Path(fields[1]).name] = fields[0]
        expected = {f"{branch}.mp4" for branch in BRANCHES}
        if set(declared) != expected:
            fail("media branch closure differs")
        media = {}
        for branch in BRANCHES:
            path = cell_root / f"{branch}.mp4"
            if (
                not path.is_file() or path.is_symlink()
                or base.file_sha256(path) != declared[path.name]
            ):
                fail(f"cell media binding differs: {path}")
            media[branch] = {"path": str(path), "sha256": declared[path.name]}
        result.append({**cell, "root": str(cell_root), "media": media})
    counts = {family: sum(row["family"] == family for row in result) for family in base.FAMILIES}
    if len(result) != 6 or counts != {"dog": 3, "human": 3}:
        fail("six-source population closure differs")
    return result


def cell_scores(
    features: Mapping[str, np.ndarray], *, representation: str, nuisance_rank: int,
) -> dict[str, Any]:
    vectors = {
        branch: base.representation(value, representation)
        for branch, value in features.items()
    }
    noop = vectors["normalized_noop"]
    nuisance_rows = np.stack([vectors[name] - noop for name in NUISANCE_BRANCHES])
    nuisance_basis = base._row_basis(nuisance_rows, nuisance_rank)
    anchor = base._project_null(vectors["normalized_forward"] - noop, nuisance_basis)
    denominator = float(np.dot(anchor, anchor))
    if not math.isfinite(denominator) or denominator <= 1.0e-12:
        fail("projected source-local action anchor is degenerate")
    scores = {}
    for branch in BRANCHES:
        delta = base._project_null(vectors[branch] - noop, nuisance_basis)
        score = float(np.dot(delta, anchor) / denominator)
        if not math.isfinite(score):
            fail("source-local branch score is non-finite")
        scores[branch] = score
    nuisance_abs_max = max(abs(scores[name]) for name in NUISANCE_BRANCHES)
    margins = {
        "forward_gt_noop": scores["normalized_forward"] - scores["normalized_noop"],
        "forward_gt_reverse": scores["normalized_forward"] - scores["reverse_from_forward"],
        "forward_gt_incomplete_hold16": scores["normalized_forward"] - scores["incomplete_hold16"],
        "forward_gt_incomplete_hold24": scores["normalized_forward"] - scores["incomplete_hold24"],
        "forward_gt_abs_nuisance": scores["normalized_forward"] - nuisance_abs_max,
    }
    return {
        "realized_nuisance_rank": nuisance_basis.shape[1],
        "anchor_squared_norm": denominator,
        "scores": scores,
        "margins": margins,
        "passes": {name: value > 0.0 for name, value in margins.items()},
    }


def evaluate_candidate(
    bank: Mapping[str, Mapping[str, np.ndarray]], *, representation: str,
    nuisance_rank: int,
) -> dict[str, Any]:
    cells = [
        {"cell": cell, **cell_scores(features, representation=representation, nuisance_rank=nuisance_rank)}
        for cell, features in sorted(bank.items())
    ]
    pass_counts = {name: sum(row["passes"][name] for row in cells) for name in MARGINS}
    minimum_margins = {name: min(row["margins"][name] for row in cells) for name in MARGINS}
    return {
        "candidate_id": f"{representation}-source-local-nr{nuisance_rank}",
        "representation": representation,
        "requested_nuisance_rank": nuisance_rank,
        "pass_counts": pass_counts,
        "minimum_margins": minimum_margins,
        "all_five_margins_all_six_cells": all(value == 6 for value in pass_counts.values()),
        "cells": cells,
    }


def select_and_replay(
    fit_bank: Mapping[str, Mapping[str, np.ndarray]],
    replay_bank: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    if set(fit_bank) & set(replay_bank):
        fail("fit and development replay cells overlap")
    candidates = [
        evaluate_candidate(fit_bank, representation=representation, nuisance_rank=rank)
        for representation in base.REPRESENTATIONS for rank in NUISANCE_RANKS
    ]
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            *(row["pass_counts"][name] for name in MARGINS),
            *(row["minimum_margins"][name] for name in MARGINS),
            -row["requested_nuisance_rank"], row["candidate_id"],
        )
    selected_fit = max(candidates, key=key)
    replay = evaluate_candidate(
        replay_bank,
        representation=selected_fit["representation"],
        nuisance_rank=selected_fit["requested_nuisance_rank"],
    )
    return {
        "selection_rule": (
            "fit_only_lexicographic_five_margin_pass_counts_then_worst_margins;"
            "source_local_signed_anchor_coefficient"
        ),
        "candidate_count": len(candidates),
        "selected_candidate_id": selected_fit["candidate_id"],
        "fit": selected_fit,
        "development_calibration_replay": replay,
        "candidates": candidates,
        "independent_calibration_claimed": False,
    }


def summary(report: Mapping[str, Any]) -> str:
    result = report["result"]
    lines = [
        "# Source-local self-generated anchor quotient", "",
        f"Selected on fit only: `{result['selected_candidate_id']}`", "",
        "| population | margin | pass / 6 | worst margin |", "| --- | --- | ---: | ---: |",
    ]
    for population in ("fit", "development_calibration_replay"):
        row = result[population]
        for name in MARGINS:
            lines.append(f"| {population} | {name} | {row['pass_counts'][name]}/6 | {row['minimum_margins'][name]:.6f} |")
    lines += [
        "", "The replay population was already observed in the preceding failed calibration and is development evidence only. A new independent calibration is still required.", "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--development-replay-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-spec", type=Path, required=True)
    parser.add_argument("--visual-scorer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_sha = base.file_sha256(Path(__file__).resolve())
    if source_sha != args.expected_source_sha256:
        fail("probe source SHA-256 differs")
    fit_cells = sealed_cells(args.fit_root, expected_split="fit")
    replay_cells = sealed_cells(
        args.development_replay_root, expected_split="development_calibration_replay"
    )
    scorer = base.v3.base.load_module(args.visual_scorer, "source_local_anchor_dino_scorer")
    spec = json.loads(args.evaluator_spec.read_text(encoding="utf-8"))
    checkpoint = scorer.verify_checkpoint_content(
        args.checkpoint, args.checkpoint_manifest, evaluator_spec=spec
    )
    processor = checkpoint.pop("processor")
    import av
    import torch
    import transformers
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        fail("probe requires exactly one visible GPU")
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
        bank, bindings = {}, {}
        for cell in cells:
            key = f"{cell['family']}:{cell['cell_id']}"
            bank[key], bindings[key] = {}, {}
            for branch in BRANCHES:
                row = cell["media"][branch]
                frames, decode = scorer.decode_exact81_rgb(row["path"], expected_sha256=row["sha256"])
                _, normalized = scorer.preprocess_selected_rgb(frames, processor)
                global_feature, _, evidence = scorer.extract_features(
                    model, normalized, device=device, num_register_tokens=0
                )
                if base.file_sha256(row["path"]) != row["sha256"]:
                    fail("video changed while extracting features")
                bank[key][branch] = global_feature.numpy().copy()
                bindings[key][branch] = {**row, "decode": decode, "features": evidence}
        return bank, bindings

    fit_bank, fit_bindings = extract(fit_cells)
    replay_bank, replay_bindings = extract(replay_cells)
    result = select_and_replay(fit_bank, replay_bank)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_sha,
        "base_dependency_sha256": BASE_SHA256,
        "fit_root": str(args.fit_root.resolve()),
        "development_replay_root": str(args.development_replay_root.resolve()),
        "result": result,
        "media_bindings": {"fit": fit_bindings, "development_calibration_replay": replay_bindings},
        "evaluator": {
            "checkpoint_root": str(args.checkpoint.resolve()),
            "checkpoint_manifest_sha256": base.file_sha256(args.checkpoint_manifest),
            "evaluator_spec_sha256": base.file_sha256(args.evaluator_spec),
            "visual_scorer_sha256": base.file_sha256(args.visual_scorer),
            "loading_counts": loading_counts,
            "model_frozen_eval": not model.training and not any(parameter.requires_grad for parameter in model.parameters()),
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "av": av.__version__, "numpy": np.__version__},
        },
        "authority": dict(AUTHORITY),
    }
    report = {**unsigned, "receipt_digest": base.object_sha256(unsigned)}
    output = args.output_root
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        fail("output root must be fresh")
    output.mkdir(mode=0o700)
    (output / "report.json").write_bytes(base.canonical_bytes(report) + b"\n")
    (output / "summary.md").write_text(summary(report), encoding="utf-8")
    print(json.dumps({"selected": result["selected_candidate_id"], "fit_all": result["fit"]["all_five_margins_all_six_cells"], "replay_all": result["development_calibration_replay"]["all_five_margins_all_six_cells"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SourceLocalProbeError, base.MultiSceneSelectionError) as error:
        print(f"[source-local-anchor-quotient] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
