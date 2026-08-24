#!/usr/bin/env python3
"""Run a target-pair-only SPT oracle execution diagnostic on one parquet row.

This tool does not train and does not establish inference quality.  It answers
the prerequisite question: can local transport from the complete clean source
bank reconstruct enough of a paired target to justify training a student?
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent
for value in (str(HERE), str(METHOD_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from phase_transport import (  # noqa: E402
    PhaseTransportConfig,
    build_oracle_plan,
    make_proxy_target,
    packed_to_video,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPT-v2 paired oracle diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-temperature", type=float, default=0.08)
    parser.add_argument("--teacher-generate-threshold", type=float, default=0.35)
    parser.add_argument("--teacher-transport-margin", type=float, default=0.05)
    parser.add_argument(
        "--disable-cycle-gate",
        action="store_true",
        help="ablation only: report cycle inconsistency but do not reject it",
    )
    parser.add_argument(
        "--feature-channels",
        type=int,
        default=64,
        help=(
            "fixed DCT-IV output width; every output uses all packed input "
            "channels (64 preserves full-channel L2)"
        ),
    )
    parser.add_argument(
        "--allow-lossy-projection-ablation",
        action="store_true",
        help="explicit diagnostic ablation; forbidden by default/main training path",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    import torch
    # Keep parser/contract inspection independent of the Bernini environment.
    # The two pinned-v1 helpers are required only for an actual AUH diagnostic.
    import motion_residual as motion
    import train_lora as legacy

    args = build_parser().parse_args(argv)
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    if not 0 <= args.row_index < len(dataset):
        raise SystemExit(f"row-index must lie in [0,{len(dataset) - 1}]")
    row = dataset[args.row_index]
    sample = legacy.sanitize_preprocessed_row(row)
    shape, _ = legacy.validate_81_frame_latents(sample)
    vae_mean, vae_std, _ = legacy._vae_statistics(checkpoint)
    source_mode = motion.unpack_clean_mode(
        sample["video_vae_latents"][0], vae_mean, vae_std, max_frames=21
    )
    target_mode = motion.unpack_clean_mode(
        sample["video_vae_latents"][1], vae_mean, vae_std, max_frames=21
    )
    source_packed = motion.flatten_velocity_patches(source_mode.unsqueeze(0)).float()
    target_packed = motion.flatten_velocity_patches(target_mode.unsqueeze(0)).float()
    source = packed_to_video(source_packed, height=shape[3] // 2, width=shape[4] // 2)
    target = packed_to_video(target_packed, height=shape[3] // 2, width=shape[4] // 2)
    device = torch.device(args.device)
    source = source.to(device)
    target = target.to(device)
    config = PhaseTransportConfig(
        latent_channels=int(source.shape[-1]),
        teacher_temperature=args.teacher_temperature,
        teacher_generate_threshold=args.teacher_generate_threshold,
        teacher_transport_margin=args.teacher_transport_margin,
        teacher_require_cycle=not args.disable_cycle_gate,
        teacher_allow_lossy_projection=args.allow_lossy_projection_ablation,
    )
    oracle = build_oracle_plan(
        source, target, config, feature_channels=args.feature_channels
    )
    proxy = make_proxy_target(source, target, oracle)

    def mse(left, right) -> float:
        return float(torch.mean((left.float() - right.float()) ** 2).item())

    iid = row.get("iid", row.get("id"))
    report = {
        "schema_version": "bernini-spt-v2-oracle-diagnostic-v2",
        "row_index": args.row_index,
        "iid": str(iid),
        "latent_shape": list(source.shape),
        "teacher_is_training_only": True,
        "inference_target_condition_forbidden": True,
        "gate_fraction": {
            "preserve": float(oracle.gate_probs[:, 0].mean().item()),
            "transport": float(oracle.gate_probs[:, 1].mean().item()),
            "generate": float(oracle.gate_probs[:, 2].mean().item()),
        },
        "rejection_fraction": {
            "valid": float(oracle.diagnostics["valid_reject_fraction"]),
            "cycle": float(oracle.diagnostics["cycle_reject_fraction"]),
            "margin": float(oracle.diagnostics["margin_reject_fraction"]),
        },
        "consistency": {
            "cycle_inconsistent_fraction": float(
                oracle.diagnostics["cycle_inconsistent_fraction"]
            ),
            "hard_executor_candidate_mse": float(
                oracle.diagnostics["hard_executor_candidate_mse"]
            ),
        },
        "mse": {
            "source_to_target": mse(source, target),
            "proxy_to_target": mse(proxy, target),
            "proxy_to_source": mse(proxy, source),
        },
        "mean_absolute_offset_cells": {
            "dt": float(oracle.offsets[:, 0].abs().mean().item()),
            "dy": float(oracle.offsets[:, 1].abs().mean().item()),
            "dx": float(oracle.offsets[:, 2].abs().mean().item()),
        },
        "config": {
            **asdict(config),
            "feature_channels": args.feature_channels,
            "device": str(device),
            "candidate_count": len(config.teacher_temporal_offsets)
            * len(config.teacher_spatial_offsets) ** 2,
            "projection": oracle.diagnostics["projection"],
            "teacher_temperature_note": (
                "retained for checkpoint compatibility; hard oracle gates do not "
                "use temperature"
            ),
        },
        "oracle_metrics": dict(oracle.diagnostics),
        "interpretation": (
            "Proceed to student training only if transport is non-trivial and "
            "proxy-to-target improves over source-to-target without opening an "
            "excessive generate fraction, and only if hard-executor candidate "
            "MSE is numerically zero. This is not an inference result."
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite: {output}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
