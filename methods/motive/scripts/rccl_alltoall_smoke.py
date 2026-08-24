#!/usr/bin/env python3
"""Small fail-fast RCCL all-to-all diagnostic for AUH MI210 nodes."""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=4_915_200)
    parser.add_argument("--iterations", type=int, default=16)
    args = parser.parse_args()
    if args.elements <= 0 or args.iterations <= 0:
        raise SystemExit("elements and iterations must be positive")

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.elements % world_size:
        raise SystemExit("elements must be divisible by world size")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        input_tensor = torch.full(
            (args.elements,),
            rank,
            dtype=torch.int32,
            device=f"cuda:{local_rank}",
        )
        output_tensor = torch.empty_like(input_tensor)
        chunk = args.elements // world_size
        for _ in range(args.iterations):
            dist.all_to_all_single(output_tensor, input_tensor)
            torch.cuda.synchronize(local_rank)
        for sender_rank in range(world_size):
            actual = output_tensor[sender_rank * chunk : (sender_rank + 1) * chunk]
            if not bool(torch.all(actual == sender_rank).item()):
                raise RuntimeError(
                    f"all-to-all payload mismatch from sender {sender_rank}"
                )
        dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "backend": "nccl",
                        "elements_per_rank": args.elements,
                        "iterations": args.iterations,
                        "status": "pass",
                        "world_size": world_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
