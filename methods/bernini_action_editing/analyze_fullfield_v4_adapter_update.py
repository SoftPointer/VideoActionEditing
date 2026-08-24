#!/usr/bin/env python3
"""Measure the actual merged rank-256 correction against its base weights."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
from typing import Sequence


PREFIX = "base_model.model.diff_dec.transformer."
BLOCK = re.compile(r"^blocks\.(\d+)\.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-transformer", required=True)
    args = parser.parse_args(argv)

    import torch
    from safetensors import safe_open

    checkpoint = Path(args.checkpoint).resolve(strict=True)
    adapter_path = checkpoint / "adapter" / "adapter_model.safetensors"
    base_root = Path(args.base_transformer).resolve(strict=True)
    index = json.loads(
        (base_root / "diffusion_pytorch_model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )["weight_map"]
    if not torch.cuda.is_available():
        raise RuntimeError("adapter correction analysis requires one GPU")
    device = torch.device("cuda", 0)
    delta_sq = 0.0
    base_sq = 0.0
    element_count = 0
    block_delta: dict[int, float] = {index: 0.0 for index in range(30)}
    block_base: dict[int, float] = {index: 0.0 for index in range(30)}

    with ExitStack() as stack:
        adapter = stack.enter_context(safe_open(adapter_path, framework="pt", device="cpu"))
        shards = {
            name: stack.enter_context(
                safe_open(base_root / name, framework="pt", device="cpu")
            )
            for name in sorted(set(index.values()))
        }
        a_keys = sorted(key for key in adapter.keys() if key.endswith(".lora_A.weight"))
        if len(a_keys) != 240:
            raise RuntimeError(f"expected 240 LoRA modules, found {len(a_keys)}")
        for a_key in a_keys:
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            module = a_key.removeprefix(PREFIX).removesuffix(".lora_A.weight")
            base_key = f"{module}.weight"
            match = BLOCK.match(module)
            if match is None or base_key not in index:
                raise RuntimeError(f"base mapping differs for {a_key}")
            block = int(match.group(1))
            a = adapter.get_tensor(a_key).to(device=device, dtype=torch.float32)
            b = adapter.get_tensor(b_key).to(device=device, dtype=torch.float32)
            weight = shards[index[base_key]].get_tensor(base_key).to(
                device=device, dtype=torch.float32
            )
            correction = b @ a
            if tuple(correction.shape) != tuple(weight.shape):
                raise RuntimeError(f"merged shape differs for {module}")
            local_delta = float(correction.double().square().sum().item())
            local_base = float(weight.double().square().sum().item())
            delta_sq += local_delta
            base_sq += local_base
            element_count += int(weight.numel())
            block_delta[block] += local_delta
            block_base[block] += local_base
            del a, b, weight, correction
    if set(block_delta) != set(range(30)) or min(block_delta.values()) <= 0:
        raise RuntimeError("merged correction does not cover all 30 blocks")
    result = {
        "schema_version": "action-fullfield-v4-merged-update-analysis-v1",
        "checkpoint": str(checkpoint),
        "module_count": 240,
        "block_count": 30,
        "merged_lora_frobenius": math.sqrt(delta_sq),
        "base_target_frobenius": math.sqrt(base_sq),
        "merged_to_base_ratio": math.sqrt(delta_sq / base_sq),
        "merged_correction_rms": math.sqrt(delta_sq / element_count),
        "block_merged_to_base_ratio": {
            str(block): math.sqrt(block_delta[block] / block_base[block])
            for block in range(30)
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
