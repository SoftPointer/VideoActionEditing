#!/usr/bin/env python3
"""Frozen-DINO shortcut probe for diversified factorial-axis v3 controls.

This no-update diagnostic compares hold and phase-warp incomplete controls and
tests four camera transforms plus one global appearance nuisance against the
same forward event axis.  It has no representation-selection authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, NoReturn

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parent
BASE_NAME = "probe_factorial_axis_controls_v2_dinov2.py"
BASE_SHA256 = "3c97e23fb17022699c394293ce11549ab9a925e5cd39b01cf1596aed824893d0"
SCHEMA_VERSION = "bernini-factorial-axis-controls-v3-dinov2-probe-v1"
FAMILIES = ("dog", "human")
BRANCHES = (
    "normalized_noop",
    "normalized_forward",
    "reverse_from_forward",
    "incomplete_hold",
    "incomplete_phasewarp",
    "camera_right_push",
    "camera_center_push",
    "camera_vertical_push",
    "camera_center_pull",
    "appearance_hue_ramp",
)
CAMERA_BRANCHES = (
    "camera_right_push", "camera_center_push",
    "camera_vertical_push", "camera_center_pull",
)
NUISANCE_BRANCHES = CAMERA_BRANCHES + ("appearance_hue_ramp",)
AUTHORITY = {
    "diagnostic_only": True,
    "event_authority": False,
    "preservation_authority": False,
    "representation_selection_authorized": False,
    "training_target_authorized": False,
    "optimizer_or_parameter_update_authorized": False,
    "scientific_claim_authorized": False,
}


class AxisV3ProbeError(RuntimeError):
    """Raised when a v3 diagnostic input or numerical contract differs."""


def fail(message: str) -> NoReturn:
    raise AxisV3ProbeError(message)


def load_source(path: Path, name: str) -> Any:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        fail(f"module source differs: {path}")
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        fail(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE_PATH = METHOD_ROOT / BASE_NAME
if not BASE_PATH.is_file() or hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError("pinned factorial-axis v2 probe dependency differs")
base = load_source(BASE_PATH, "factorial_axis_v2_probe_dependency")


def sealed_media(root: Path, expected_manifest_sha256: str) -> dict[str, dict[str, Any]]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        fail("input root differs")
    manifest = resolved / "media.sha256"
    if (
        not manifest.is_file() or manifest.is_symlink()
        or base.file_sha256(manifest) != expected_manifest_sha256
    ):
        fail("media manifest differs")
    declared: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            fail("media manifest row differs")
        relative = "/".join(Path(fields[1]).parts[-2:])
        declared[relative] = fields[0]
    expected = {
        f"{family}/{branch}.mp4"
        for family in FAMILIES for branch in BRANCHES
    }
    if set(declared) != expected:
        fail("media branch closure differs")
    rows: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        rows[family] = {}
        for branch in BRANCHES:
            relative = f"{family}/{branch}.mp4"
            video = resolved / relative
            if (
                not video.is_file() or video.is_symlink()
                or base.file_sha256(video) != declared[relative]
            ):
                fail(f"video binding differs: {relative}")
            rows[family][branch] = {
                "path": str(video), "sha256": declared[relative],
            }
    return rows


def mean(values: list[float], *, label: str) -> float:
    if not values:
        fail(f"{label} is empty")
    return base.finite(np.mean(values), label=label)


def static_fraction(speeds: list[float], forward_mean: float) -> tuple[float, float]:
    threshold = max(1.0e-4, 0.05 * forward_mean)
    return (
        base.finite(sum(value <= threshold for value in speeds) / len(speeds), label="static fraction"),
        threshold,
    )


def analyze_family(features: Mapping[str, np.ndarray], *, family: str) -> dict[str, Any]:
    if family not in FAMILIES or set(features) != set(BRANCHES):
        fail("family/branch closure differs")
    shape = features[BRANCHES[0]].shape
    if (
        len(shape) != 2 or shape[0] != len(base.FRAME_INDICES)
        or any(value.shape != shape for value in features.values())
        or any(not np.isfinite(value).all() for value in features.values())
    ):
        fail("feature geometry differs")

    vectors = {name: value.astype(np.float64) for name, value in features.items()}
    arrows = {name: value[-1] - value[0] for name, value in vectors.items()}
    event_axis, event_norm = base.unit(arrows["normalized_forward"])
    if event_norm <= 1.0e-12:
        fail("forward endpoint arrow is degenerate")
    arrow_scores = {
        name: base.cosine(vector, event_axis) for name, vector in arrows.items()
    }
    arrow_norms = {
        name: base.finite(np.linalg.norm(vector), label="arrow norm")
        for name, vector in arrows.items()
    }
    speeds = {name: base.speed_profile(value) for name, value in vectors.items()}
    forward_mean = mean(speeds["normalized_forward"], label="forward mean speed")
    incomplete = {}
    for name in ("incomplete_hold", "incomplete_phasewarp"):
        fraction, threshold = static_fraction(speeds[name], forward_mean)
        incomplete[name] = {
            "endpoint_cosine_to_forward_axis": arrow_scores[name],
            "endpoint_arrow_norm": arrow_norms[name],
            "mean_speed": mean(speeds[name], label=f"{name} mean speed"),
            "near_static_interval_fraction": fraction,
            "near_static_threshold": threshold,
            "threshold_is_relative_diagnostic_only": True,
        }
    hold_endpoint = vectors["incomplete_hold"][-1]
    warp_endpoint = vectors["incomplete_phasewarp"][-1]
    incomplete["endpoint_feature_cosine_hold_vs_phasewarp"] = base.cosine(
        hold_endpoint, warp_endpoint
    )

    forward = vectors["normalized_forward"]
    reverse = vectors["reverse_from_forward"]
    reverse_alignment = base.aligned_cosines(forward, reverse[::-1])
    reverse_tss = base.temporal_self_similarity(reverse)
    forward_tss = base.temporal_self_similarity(forward)
    reverse_rmse = base.finite(
        np.sqrt(np.mean((forward_tss - reverse_tss[::-1, ::-1]) ** 2)),
        label="reverse TSS RMSE",
    )

    noop = vectors["normalized_noop"]
    nuisance = {}
    for name in NUISANCE_BRANCHES:
        aligned = base.aligned_cosines(noop, vectors[name])
        nuisance[name] = {
            "endpoint_cosine_to_forward_axis": arrow_scores[name],
            "endpoint_arrow_norm": arrow_norms[name],
            "mean_frame_aligned_cosine_to_noop": mean(aligned, label="nuisance alignment"),
            "mean_feature_cosine_to_noop": base.cosine(
                vectors[name].mean(axis=0), noop.mean(axis=0)
            ),
            "exploratory_abs_event_projection_alert": (
                arrow_scores[name] is not None and abs(arrow_scores[name]) >= 0.5
            ),
            "alert_threshold_is_unvalidated_heuristic": True,
        }
    alerts = [
        name for name in NUISANCE_BRANCHES
        if nuisance[name]["exploratory_abs_event_projection_alert"]
    ]
    return {
        "family": family,
        "selected_frame_indices": list(base.FRAME_INDICES),
        "feature_shape": list(shape),
        "endpoint_arrow_cosine_to_forward_axis": arrow_scores,
        "endpoint_arrow_norms": arrow_norms,
        "reverse_integrity": {
            "mean_forward_vs_time_reversed_reverse_alignment": mean(
                reverse_alignment, label="reverse alignment"
            ),
            "minimum_alignment": min(reverse_alignment),
            "temporal_self_similarity_reversal_rmse": reverse_rmse,
        },
        "incomplete_comparison": incomplete,
        "nuisance_controls": nuisance,
        "nuisance_alerts": alerts,
        "speed_profiles": speeds,
        "interpretation": {
            "phasewarp_is_optical_flow_interpolation_not_real_motion": True,
            "appearance_hue_ramp_is_global_not_actor_localized": True,
            "camera_set_is_one_transform_family": True,
            "all_thresholds_diagnostic_only": True,
        },
    }


def markdown_summary(report: Mapping[str, Any]) -> str:
    lines = [
        "# Factorial-axis v3 frozen-DINO diagnostic", "",
        "No-update shortcut probe; no target, representation, or optimizer authority.", "",
        "| family | reverse align | reverse arrow | hold static frac | phasewarp static frac | hold/warp endpoint | max nuisance | alerts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for family in FAMILIES:
        row = report["families"][family]
        incomplete = row["incomplete_comparison"]
        nuisance_values = {
            name: value["endpoint_cosine_to_forward_axis"]
            for name, value in row["nuisance_controls"].items()
        }
        max_nuisance = max(nuisance_values, key=lambda name: abs(nuisance_values[name]))
        lines.append(
            f"| {family} | {row['reverse_integrity']['mean_forward_vs_time_reversed_reverse_alignment']:.6f} | "
            f"{row['endpoint_arrow_cosine_to_forward_axis']['reverse_from_forward']:.6f} | "
            f"{incomplete['incomplete_hold']['near_static_interval_fraction']:.6f} | "
            f"{incomplete['incomplete_phasewarp']['near_static_interval_fraction']:.6f} | "
            f"{incomplete['endpoint_feature_cosine_hold_vs_phasewarp']:.6f} | "
            f"{max_nuisance}={nuisance_values[max_nuisance]:.6f} | "
            f"{','.join(row['nuisance_alerts']) or 'none'} |"
        )
    lines += [
        "", "- The phase-warp arm is a construction/shortcut diagnostic, not real generated motion.",
        "- The global hue ramp is an appearance nuisance, not localized actor appearance editing.",
        "- The 0.5 alert is an uncalibrated exploratory flag and is not a universal threshold.",
        "- No output authorizes representation selection or training.", "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-media-manifest-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-spec", type=Path, required=True)
    parser.add_argument("--visual-scorer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_sha = base.file_sha256(Path(__file__).resolve())
    if source_sha != args.expected_source_sha256:
        fail("diagnostic source SHA-256 differs")
    media = sealed_media(args.input_root, args.expected_media_manifest_sha256)
    scorer = base.load_module(args.visual_scorer, "factorial_axis_v3_sealed_dino_scorer")
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

    feature_bank: dict[str, dict[str, np.ndarray]] = {}
    bindings: dict[str, Any] = {}
    for family in FAMILIES:
        feature_bank[family], bindings[family] = {}, {}
        for branch in BRANCHES:
            row = media[family][branch]
            frames, decode = scorer.decode_exact81_rgb(
                row["path"], expected_sha256=row["sha256"]
            )
            _, normalized = scorer.preprocess_selected_rgb(frames, processor)
            global_feature, _, evidence = scorer.extract_features(
                model, normalized, device=device, num_register_tokens=0
            )
            if base.file_sha256(row["path"]) != row["sha256"]:
                fail("video changed while extracting features")
            feature_bank[family][branch] = global_feature.numpy().copy()
            bindings[family][branch] = {**row, "decode": decode, "features": evidence}

    families = {
        family: analyze_family(feature_bank[family], family=family)
        for family in FAMILIES
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_source_sha256": source_sha,
        "base_dependency_sha256": BASE_SHA256,
        "input_root": str(args.input_root.resolve()),
        "media_manifest_sha256": args.expected_media_manifest_sha256,
        "families": families,
        "media_bindings": bindings,
        "evaluator": {
            "checkpoint_root": str(args.checkpoint.resolve()),
            "checkpoint_manifest_sha256": base.file_sha256(args.checkpoint_manifest),
            "evaluator_spec_sha256": base.file_sha256(args.evaluator_spec),
            "visual_scorer_sha256": base.file_sha256(args.visual_scorer),
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
    report = {**unsigned, "receipt_digest": base.object_sha256(unsigned)}
    output = base.create_output_root(args.output_root)
    base.write_create_only(output / "report.json", base.canonical_bytes(report) + b"\n")
    base.write_create_only(output / "summary.md", markdown_summary(report).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
