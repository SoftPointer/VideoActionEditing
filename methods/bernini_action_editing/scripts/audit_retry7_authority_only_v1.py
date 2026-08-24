#!/usr/bin/env python3
"""Run the retry7 real authority chain without distributed init, model, or optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--g1-receipt", required=True)
    parser.add_argument("--g2a-receipt", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--claim", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    method_root = Path(args.method_root).resolve(strict=True)
    output = Path(args.output).absolute()
    claim = Path(args.claim).absolute()
    if output.exists() or output.is_symlink() or claim.exists() or claim.is_symlink():
        raise RuntimeError("authority-only audit requires fresh output and claim")
    sys.path.insert(0, str(method_root))
    import torch
    import train_action_repr_target_t0_canary_retry7_v1 as runner

    optimizer_attempted = False
    original_adamw = torch.optim.AdamW

    def forbidden_adamw(*args, **kwargs):
        nonlocal optimizer_attempted
        optimizer_attempted = True
        raise RuntimeError("authority-only audit reached AdamW")

    torch.optim.AdamW = forbidden_adamw
    os.environ["STAGE_B_T0_AUTHORITY_SHA256"] = args.authority_sha256
    try:
        authority = runner.authorize_preoptimizer_inputs(
            manifest=args.manifest,
            g1_admission_receipt=args.g1_receipt,
            g2a_receipt=args.g2a_receipt,
            authorization_addendum=args.authority,
            bernini_root=args.bernini_root,
            veomni_root=args.veomni_root,
            checkpoint=args.checkpoint,
            output=output,
        )
    finally:
        torch.optim.AdamW = original_adamw
    if optimizer_attempted or output.exists() or output.is_symlink() or claim.exists() or claim.is_symlink():
        raise RuntimeError("authority-only audit crossed its zero-mutation boundary")
    evidence = {
        "schema_version": "bernini-action-repr-retry7-authority-only-audit-v1",
        "passed": True,
        "authorization_addendum_sha256": authority.authorization_sha256,
        "source_hash_pins_digest": authority.authorization["source_hash_pins_digest"],
        "g1_receipt_sha256": authority.g1_receipt_sha256,
        "g2a_receipt_sha256": authority.g2a_file_sha256,
        "g1_replay_runtime": authority.g1_replay_runtime,
        "optimizer_created": False,
        "model_loaded": False,
        "claim_created": False,
        "output_created": False,
        "process_pid": os.getpid(),
        "torch_hip": str(torch.version.hip),
    }
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    evidence["evidence_digest"] = hashlib.sha256(payload.encode("ascii")).hexdigest()
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
