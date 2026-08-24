#!/usr/bin/env python3
"""Sequential SP4 controller for one subset of Stage-0 flow-noise arms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ARMS = {
    "raw": ("raw", "0.0"),
    "raw_deg030": ("raw", "0.3"),
    "camera": ("camera_residual", "0.0"),
    "camera_deg030": ("camera_residual", "0.3"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--flow-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=2026081601)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--master-port-base", type=int, required=True)
    args = parser.parse_args()
    method_root = Path(args.method_root).resolve(strict=True)
    manifest = json.loads(Path(args.manifest).read_text())
    flow_root = Path(args.flow_root).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runner = method_root / "infer_flow_noise_action_canary_v1.py"
    python_bin = Path(sys.executable).resolve(strict=True)
    fixed = manifest["runtime"]
    media_root = Path(manifest["remote_media_root"])
    run_index = 0
    for row in manifest["rows"]:
        iid = row["iid"]
        source = media_root / iid / "source.mp4"
        flow_bundle = flow_root / f"{iid}.safetensors"
        for arm in args.arms:
            field, degradation = ARMS[arm]
            output = output_root / iid / f"{arm}__s{args.seed}.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(python_bin),
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                f"--master_port={args.master_port_base + run_index}",
                str(runner),
                "--flow-bundle",
                str(flow_bundle),
                "--flow-field",
                field,
                "--flow-degradation",
                degradation,
                "--bernini-root",
                fixed["bernini_root"],
                "--veomni-root",
                fixed["veomni_root"],
                "--checkpoint",
                fixed["checkpoint"],
                "--base-only",
                "--source-video",
                str(source),
                "--instruction",
                row["instruction"],
                "--output",
                str(output),
                "--num-inference-steps",
                "40",
                "--seed",
                str(args.seed),
                "--source-onset-policy",
                "hard1_every_step",
                "--expected-bernini-commit",
                fixed["bernini_commit"],
                "--expected-veomni-commit",
                fixed["veomni_commit"],
                "--expected-checkpoint-tree-sha256",
                fixed["checkpoint_tree_sha256"],
                "--method-source-revision",
                args.method_source_revision,
                "--method-source-archive-sha256",
                args.method_source_archive_sha256,
            ]
            print(json.dumps({"iid": iid, "arm": arm, "command": command}), flush=True)
            subprocess.run(command, check=True, env=os.environ.copy())
            run_index += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
