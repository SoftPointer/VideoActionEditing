#!/usr/bin/env python3
"""Frozen-DINO diagnostic for the prospective factorial-axis v2 canary.

The probe measures whether simple temporal representations are vulnerable to
the deterministic construction used by the v2 control videos.  It performs no
training, target admission, representation selection, or optimizer update.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, NoReturn

import numpy as np


SCHEMA_VERSION = "bernini-factorial-axis-controls-v2-dinov2-probe-v1"
FAMILIES = ("dog", "human")
BRANCHES = (
    "normalized_noop",
    "normalized_forward",
    "reverse_from_forward",
    "incomplete_from_forward",
    "camera_from_noop",
)
FRAME_INDICES = tuple(range(0, 81, 5))
FAMILY_CUTS = {"dog": 10, "human": 24}
AUTHORITY = {
    "diagnostic_only": True,
    "event_authority": False,
    "preservation_authority": False,
    "representation_selection_authorized": False,
    "training_target_authorized": False,
    "optimizer_or_parameter_update_authorized": False,
    "scientific_claim_authorized": False,
}


class AxisProbeError(RuntimeError):
    """Raised when the diagnostic boundary or numerical contract differs."""


def fail(message: str) -> NoReturn:
    raise AxisProbeError(message)


def file_sha256(path: str | Path) -> str:
    value = Path(path)
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} is non-finite")
    return result


def unit(vector: np.ndarray) -> tuple[np.ndarray, float]:
    value = np.asarray(vector, dtype=np.float64)
    norm = finite(np.linalg.norm(value), label="vector norm")
    if norm <= 1.0e-12:
        return np.zeros_like(value), norm
    return value / norm, norm


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    lhs, lhs_norm = unit(left)
    rhs, rhs_norm = unit(right)
    if lhs_norm <= 1.0e-12 or rhs_norm <= 1.0e-12:
        return None
    return finite(np.dot(lhs, rhs), label="cosine")


def aligned_cosines(left: np.ndarray, right: np.ndarray) -> list[float]:
    if left.shape != right.shape or left.ndim != 2:
        fail("aligned feature geometry differs")
    values = np.sum(left.astype(np.float64) * right.astype(np.float64), axis=-1)
    return [finite(item, label="aligned cosine") for item in values]


def speed_profile(features: np.ndarray) -> list[float]:
    if features.ndim != 2 or features.shape[0] != len(FRAME_INDICES):
        fail("temporal feature geometry differs")
    values = np.linalg.norm(np.diff(features.astype(np.float64), axis=0), axis=-1)
    return [finite(item, label="speed") for item in values]


def temporal_self_similarity(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] != len(FRAME_INDICES):
        fail("temporal self-similarity geometry differs")
    return features.astype(np.float64) @ features.astype(np.float64).T


def analyze_family(
    features: Mapping[str, np.ndarray], *, family: str, cut_frame: int,
) -> dict[str, Any]:
    if family not in FAMILIES or set(features) != set(BRANCHES):
        fail("family or branch closure differs")
    shape = features[BRANCHES[0]].shape
    if len(shape) != 2 or shape[0] != len(FRAME_INDICES):
        fail("feature tensor geometry differs")
    if any(value.shape != shape for value in features.values()):
        fail("branch feature geometry differs")
    if any(not np.isfinite(value).all() for value in features.values()):
        fail("branch feature contains a non-finite value")

    forward = features["normalized_forward"].astype(np.float64)
    reverse = features["reverse_from_forward"].astype(np.float64)
    incomplete = features["incomplete_from_forward"].astype(np.float64)
    noop = features["normalized_noop"].astype(np.float64)
    camera = features["camera_from_noop"].astype(np.float64)

    arrows = {name: value[-1] - value[0] for name, value in features.items()}
    event_axis, event_axis_norm = unit(arrows["normalized_forward"])
    if event_axis_norm <= 1.0e-12:
        fail("forward endpoint arrow is degenerate")
    endpoint_scores = {
        name: cosine(vector, event_axis) for name, vector in arrows.items()
    }
    arrow_norms = {
        name: finite(np.linalg.norm(vector), label="arrow norm")
        for name, vector in arrows.items()
    }

    reverse_alignment = aligned_cosines(forward, reverse[::-1])
    forward_tss = temporal_self_similarity(forward)
    reverse_tss = temporal_self_similarity(reverse)
    reversal_tss_rmse = finite(
        np.sqrt(np.mean((forward_tss - reverse_tss[::-1, ::-1]) ** 2)),
        label="reversal TSS RMSE",
    )

    prefix_positions = [
        index for index, frame in enumerate(FRAME_INDICES) if frame <= cut_frame
    ]
    first_hold_position = next(
        index for index, frame in enumerate(FRAME_INDICES) if frame >= cut_frame
    )
    prefix_alignment = aligned_cosines(
        incomplete[prefix_positions], forward[prefix_positions]
    )
    incomplete_speeds = speed_profile(incomplete)
    forward_speeds = speed_profile(forward)
    tail_speeds = incomplete_speeds[first_hold_position:]
    forward_mean_speed = finite(np.mean(forward_speeds), label="forward mean speed")
    tail_mean_speed = finite(np.mean(tail_speeds), label="tail mean speed")
    hold_ratio = (
        tail_mean_speed / forward_mean_speed if forward_mean_speed > 1.0e-12 else None
    )
    hold_shortcut = (
        None if hold_ratio is None else finite(1.0 - min(1.0, hold_ratio), label="hold shortcut")
    )

    noop_camera_alignment = aligned_cosines(noop, camera)
    appearance_mean_similarity = {
        name: cosine(value.mean(axis=0), noop.mean(axis=0))
        for name, value in features.items()
    }
    speed_profiles = {name: speed_profile(value) for name, value in features.items()}
    camera_event_score = endpoint_scores["camera_from_noop"]

    return {
        "family": family,
        "cut_frame": cut_frame,
        "selected_frame_indices": list(FRAME_INDICES),
        "feature_shape": list(shape),
        "endpoint_arrow": {
            "forward_axis_norm": event_axis_norm,
            "branch_arrow_norms": arrow_norms,
            "cosine_to_forward_axis": endpoint_scores,
        },
        "reverse_integrity": {
            "forward_vs_time_reversed_reverse_cosines": reverse_alignment,
            "mean_alignment": finite(np.mean(reverse_alignment), label="reverse alignment"),
            "minimum_alignment": min(reverse_alignment),
            "temporal_self_similarity_reversal_rmse": reversal_tss_rmse,
        },
        "incomplete_hold": {
            "prefix_selected_positions": prefix_positions,
            "prefix_alignment_cosines": prefix_alignment,
            "prefix_mean_alignment": finite(np.mean(prefix_alignment), label="prefix alignment"),
            "first_hold_selected_position": first_hold_position,
            "tail_mean_speed": tail_mean_speed,
            "forward_mean_speed": forward_mean_speed,
            "tail_to_forward_speed_ratio": hold_ratio,
            "static_tail_shortcut_severity": hold_shortcut,
        },
        "camera_control": {
            "noop_vs_camera_aligned_cosines": noop_camera_alignment,
            "mean_noop_alignment": finite(np.mean(noop_camera_alignment), label="camera alignment"),
            "endpoint_cosine_to_forward_event_axis": camera_event_score,
            "exploratory_endpoint_false_positive_alert": (
                camera_event_score is not None and abs(camera_event_score) >= 0.5
            ),
            "alert_threshold_is_unvalidated_heuristic": True,
        },
        "appearance_mean_cosine_to_noop": appearance_mean_similarity,
        "speed_profiles": speed_profiles,
        "diagnostic_interpretation": {
            "reverse_is_deterministic_transform_not_independent_generalization": True,
            "incomplete_static_tail_is_known_shortcut": True,
            "camera_transform_family_is_single_deterministic_pattern": True,
            "thresholds_have_no_selection_or_scientific_authority": True,
        },
    }


def load_module(path: Path, name: str) -> Any:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        fail(f"module source is not one plain file: {path}")
    if str(resolved.parent) not in sys.path:
        sys.path.insert(0, str(resolved.parent))
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        fail(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sealed_media(root: Path, expected_manifest_sha256: str) -> dict[str, dict[str, Any]]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        fail("input root is not one plain directory")
    manifest = resolved / "media.sha256"
    if not manifest.is_file() or manifest.is_symlink():
        fail("media manifest differs")
    manifest_sha = file_sha256(manifest)
    if manifest_sha != expected_manifest_sha256:
        fail("media manifest SHA-256 differs")
    declared: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            fail("media manifest row differs")
        relative = "/".join(Path(fields[1]).parts[-2:])
        declared[relative] = fields[0]
    rows: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        rows[family] = {}
        for branch in BRANCHES:
            video = resolved / family / f"{branch}.mp4"
            relative = f"{family}/{branch}.mp4"
            if (
                not video.is_file() or video.is_symlink()
                or declared.get(relative) != file_sha256(video)
            ):
                fail(f"video binding differs: {relative}")
            rows[family][branch] = {
                "path": str(video), "sha256": declared[relative],
            }
    if len(declared) != len(FAMILIES) * len(BRANCHES):
        fail("media manifest branch closure differs")
    return rows


def create_output_root(path: Path) -> Path:
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        fail("output root must be fresh, absolute, and non-root")
    path.mkdir(mode=0o700, parents=False)
    return path


def write_create_only(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def markdown_summary(report: Mapping[str, Any]) -> str:
    lines = [
        "# Factorial-axis v2 frozen-DINO diagnostic", "",
        "This is a no-update shortcut probe, not a representation selection result.", "",
        "| family | reverse alignment | reverse arrow | incomplete hold shortcut | camera event-axis cosine | camera alert |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for family in FAMILIES:
        row = report["families"][family]
        reverse = row["reverse_integrity"]["mean_alignment"]
        reverse_arrow = row["endpoint_arrow"]["cosine_to_forward_axis"]["reverse_from_forward"]
        shortcut = row["incomplete_hold"]["static_tail_shortcut_severity"]
        camera = row["camera_control"]["endpoint_cosine_to_forward_event_axis"]
        alert = row["camera_control"]["exploratory_endpoint_false_positive_alert"]
        lines.append(
            f"| {family} | {reverse:.6f} | {reverse_arrow:.6f} | "
            f"{shortcut:.6f} | {camera:.6f} | {str(alert).lower()} |"
        )
    lines += [
        "", "## Interpretation", "",
        "- Reverse integrity is expected because reverse is derived from the same trajectory; it is not an independent generalization test.",
        "- A large incomplete hold-shortcut value means a representation can separate incomplete by static-tail duration instead of event phase.",
        "- The camera alert is an exploratory |cosine| >= 0.5 flag only; it is not a calibrated or universal threshold.",
        "- No output authorizes training targets, representation selection, or optimizer steps.", "",
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
    source_path = Path(__file__).resolve()
    source_sha = file_sha256(source_path)
    if source_sha != args.expected_source_sha256:
        fail("diagnostic source SHA-256 differs")
    media = sealed_media(args.input_root, args.expected_media_manifest_sha256)
    scorer = load_module(args.visual_scorer, "factorial_axis_v2_sealed_dino_scorer")
    evaluator_spec = json.loads(args.evaluator_spec.read_text(encoding="utf-8"))
    checkpoint = scorer.verify_checkpoint_content(
        args.checkpoint, args.checkpoint_manifest, evaluator_spec=evaluator_spec
    )
    processor = checkpoint.pop("processor")

    import torch
    import transformers
    import av

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
        feature_bank[family] = {}
        bindings[family] = {}
        for branch in BRANCHES:
            row = media[family][branch]
            frames, decode = scorer.decode_exact81_rgb(
                row["path"], expected_sha256=row["sha256"]
            )
            _, normalized = scorer.preprocess_selected_rgb(frames, processor)
            global_feature, _, evidence = scorer.extract_features(
                model, normalized, device=device, num_register_tokens=0
            )
            if file_sha256(row["path"]) != row["sha256"]:
                fail("video changed during feature extraction")
            feature_bank[family][branch] = global_feature.numpy().copy()
            bindings[family][branch] = {**row, "decode": decode, "features": evidence}

    families = {
        family: analyze_family(
            feature_bank[family], family=family, cut_frame=FAMILY_CUTS[family]
        )
        for family in FAMILIES
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_source_sha256": source_sha,
        "input_root": str(args.input_root.resolve()),
        "media_manifest_sha256": args.expected_media_manifest_sha256,
        "frame_indices": list(FRAME_INDICES),
        "families": families,
        "media_bindings": bindings,
        "evaluator": {
            "checkpoint_root": str(Path(args.checkpoint).resolve()),
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
    output = create_output_root(args.output_root)
    write_create_only(output / "report.json", canonical_bytes(report) + b"\n")
    write_create_only(output / "summary.md", markdown_summary(report).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
