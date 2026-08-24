#!/usr/bin/env python3
"""Prove an archived authority cannot reach an old runner's model or optimizer."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence


class ModelSentinel(RuntimeError):
    pass


class OptimizerSentinel(RuntimeError):
    pass


class ArchiveGateUnsafe(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--original-authority-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--g1-receipt", required=True)
    parser.add_argument("--g2a-receipt", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    method_root = Path(args.method_root).resolve(strict=True)
    authority = Path(args.authority).resolve(strict=True)
    output = Path(args.output).absolute()
    if output.exists() or output.is_symlink():
        raise RuntimeError("negative-test output must be fresh")
    sys.path.insert(0, str(method_root))
    import torch
    import train_action_repr_target_t0_canary_v1 as runner

    # Execute the old runner's exact arbitrary-path authority load and external
    # SHA gate independently of upstream G1 replay.  Reaching the end would mean
    # that the archived bytes can still impersonate the original ACTIVE seal.
    gate_failure: Exception | None = None
    try:
        _, _, observed_authority_sha = runner.g2a_world4.read_json(
            authority, label="archived authority through old runner gate"
        )
        runner.g2a_world4.require_sha256(
            args.original_authority_sha256, label="original old-runner authority seal"
        )
        if observed_authority_sha != args.original_authority_sha256:
            runner.fail("external Stage-B T0 authority seal differs from addendum bytes")
        raise ArchiveGateUnsafe(
            "archived authority passed the old runner read_json and original-SHA gate"
        )
    except Exception as error:
        gate_failure = error
    if isinstance(gate_failure, ArchiveGateUnsafe):
        raise gate_failure

    model_attempted = False
    optimizer_attempted = False
    original_import = builtins.__import__
    original_adamw = torch.optim.AdamW

    def guarded_import(name, *positional, **keyword):
        nonlocal model_attempted
        if name == "bernini.models.renderer" or name.startswith("bernini.models.renderer."):
            model_attempted = True
            raise ModelSentinel("old runner reached BerniniRendererModel import")
        return original_import(name, *positional, **keyword)

    def guarded_adamw(*positional, **keyword):
        nonlocal optimizer_attempted
        optimizer_attempted = True
        raise OptimizerSentinel("old runner reached AdamW")

    raw_mapping = "0,1,2,3"
    os.environ["ROCR_VISIBLE_DEVICES"] = raw_mapping
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES"] = raw_mapping
    os.environ["ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_SHA256"] = hashlib.sha256(
        raw_mapping.encode("ascii")
    ).hexdigest()
    os.environ["ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_COUNT"] = "4"
    os.environ["ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_PRESERVED"] = "true"
    os.environ["STAGE_B_T0_AUTHORITY_SHA256"] = args.original_authority_sha256
    builtins.__import__ = guarded_import
    torch.optim.AdamW = guarded_adamw
    failure: Exception | None = None
    try:
        runner.main(
            [
                "--authorization-addendum",
                str(authority),
                "--manifest",
                args.manifest,
                "--g1-admission-receipt",
                args.g1_receipt,
                "--g2a-receipt",
                args.g2a_receipt,
                "--bernini-root",
                args.bernini_root,
                "--veomni-root",
                args.veomni_root,
                "--checkpoint",
                args.checkpoint,
                "--output",
                str(output),
            ]
        )
    except Exception as error:
        failure = error
    finally:
        builtins.__import__ = original_import
        torch.optim.AdamW = original_adamw
    if failure is None:
        raise RuntimeError("superseded authority unexpectedly passed the old runner")
    if isinstance(failure, (ModelSentinel, OptimizerSentinel)):
        raise failure
    if model_attempted or optimizer_attempted or output.exists() or output.is_symlink():
        raise RuntimeError("old runner crossed the pre-model/pre-optimizer failure boundary")
    evidence = {
        "schema_version": "bernini-superseded-authority-direct-runner-negative-v1",
        "authority_path": str(authority),
        "authority_file_sha256": sha256_file(authority),
        "original_active_authority_sha256_supplied_as_external_seal": args.original_authority_sha256,
        "old_runner_path": str(method_root / "train_action_repr_target_t0_canary_v1.py"),
        "old_runner_arbitrary_path_authority_gate_executed": True,
        "old_runner_authority_gate_failure_type": type(gate_failure).__name__,
        "old_runner_authority_gate_failure_message": str(gate_failure),
        "archive_passed_original_sha_gate": False,
        "failure_type": type(failure).__name__,
        "failure_message": str(failure),
        "model_import_attempted": model_attempted,
        "optimizer_creation_attempted": optimizer_attempted,
        "output_created": False,
        "passed": True,
    }
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
