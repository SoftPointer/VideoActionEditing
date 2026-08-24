#!/usr/bin/env python3
"""Audit PACT's Wan flow tables against one fixed DiffSynth checkout.

The command imports only ``diffsynth/diffusion/flow_match.py`` from the
user-supplied checkout.  It verifies the pinned Git revision, requires exact
FP32 equality for sigma/timestep/BSMNTW tables, checks the add-noise and
velocity target equations, and reports the bin collapse caused by the
reference training entry's later BF16 timestep cast.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pact.flow import flow_noisy_latent, velocity_target  # noqa: E402
from pact.training import DiffSynthWanTrainingScheduler  # noqa: E402


REFERENCE_REVISION = "ab12bf4119b7c9a23ff3359eefb41ba54a658ccb"
AUDIT_FORMAT = "pact-diffsynth-wan-parity-audit-v1"


class ParityAuditError(RuntimeError):
    """Raised when the fixed DiffSynth reference and PACT differ."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ParityAuditError(
            f"cannot resolve DiffSynth revision: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _load_flow_match(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("pact_diffsynth_flow_match", path)
    if spec is None or spec.loader is None:
        raise ParityAuditError(f"cannot import reference source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_diffsynth_wan_parity(diffsynth_root: Path) -> dict[str, Any]:
    root = diffsynth_root.expanduser().resolve()
    source = root / "diffsynth" / "diffusion" / "flow_match.py"
    if not source.is_file() or source.is_symlink():
        raise ParityAuditError(
            f"reference flow_match.py must be a regular non-symlink file: {source}"
        )
    revision = _revision(root)
    if revision != REFERENCE_REVISION:
        raise ParityAuditError(
            f"DiffSynth revision must be {REFERENCE_REVISION}, got {revision}"
        )

    reference_module = _load_flow_match(source)
    reference = reference_module.FlowMatchScheduler("Wan")
    reference.set_timesteps(1000, training=True)
    pact = DiffSynthWanTrainingScheduler(shift=5.0)

    exact = {
        "sigmas": torch.equal(reference.sigmas, pact.sigmas),
        "timesteps": torch.equal(reference.timesteps, pact.timesteps),
        "bsmntw_weights": torch.equal(
            reference.linear_timesteps_weights, pact.flow_weights
        ),
    }
    if not all(exact.values()):
        raise ParityAuditError(f"FP32 table parity failed: {exact}")

    generator = torch.Generator(device="cpu").manual_seed(20260803)
    x0 = torch.randn((1, 16, 3, 4, 4), generator=generator)
    noise = torch.randn(x0.shape, generator=generator)
    equation_checks: dict[str, bool] = {}
    for timestep_id in (0, 317, 999):
        sample = pact.at(timestep_id, 1)
        reference_x_t = reference.add_noise(x0, noise, sample.timestep)
        reference_target = reference.training_target(x0, noise, sample.timestep)
        equation_checks[f"add_noise_bin_{timestep_id}"] = torch.equal(
            reference_x_t, flow_noisy_latent(x0, noise, sample.sigma)
        )
        equation_checks[f"target_bin_{timestep_id}"] = torch.equal(
            reference_target, velocity_target(x0, noise)
        )
    if not all(equation_checks.values()):
        raise ParityAuditError(f"flow equation parity failed: {equation_checks}")

    # DiffSynth's Wan training loss casts a sampled timestep to pipe.torch_dtype
    # (BF16 by default), then its scheduler finds the nearest FP32 table entry.
    # Quantify this literal execution behavior without adopting it in PACT.
    rounded_timesteps = reference.timesteps.to(torch.bfloat16).float()
    remapped_ids = torch.stack(
        [
            torch.argmin((reference.timesteps - timestep).abs())
            for timestep in rounded_timesteps
        ]
    )
    nominal_ids = torch.arange(reference.timesteps.numel())
    remapped = remapped_ids != nominal_ids

    return {
        "format": AUDIT_FORMAT,
        "status": "verified",
        "diffsynth_revision": revision,
        "flow_match_sha256": _sha256(source),
        "fp32_table_exact": exact,
        "flow_equations_exact": equation_checks,
        "pact_policy": {
            "flow_path_dtype": "float32",
            "reason": "preserve the selected FP32 bin and formal RF path",
        },
        "diffsynth_literal_bf16_lookup": {
            "nominal_bins": int(reference.timesteps.numel()),
            "effective_bins": int(remapped_ids.unique().numel()),
            "remapped_draws": int(remapped.sum()),
            "max_absolute_id_shift": int(
                (remapped_ids - nominal_ids).abs().max()
            ),
            "max_sigma_absolute_difference": float(
                (reference.sigmas[remapped_ids] - reference.sigmas).abs().max()
            ),
            "max_bsmntw_absolute_difference": float(
                (
                    reference.linear_timesteps_weights[remapped_ids]
                    - reference.linear_timesteps_weights
                )
                .abs()
                .max()
            ),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffsynth-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional create-only JSON receipt; the receipt is always printed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = audit_diffsynth_wan_parity(args.diffsynth_root)
        encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
        print(encoded)
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "format": AUDIT_FORMAT,
                    "status": "rejected",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
