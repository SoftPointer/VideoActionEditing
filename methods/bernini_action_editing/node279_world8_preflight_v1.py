#!/usr/bin/env python3
"""Create-only WORLD8 ROCm preflight for the 0817 action-editing program.

This program performs no model forward, backward, optimizer update, media
generation, or scheduler action.  It only proves that one torchrun process is
bound to each of eight GPUs and that BF16 compute plus an NCCL/RCCL collective
work inside the caller's existing Slurm allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any


SCHEMA = "bernini-0817-node279-world8-preflight-v1"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit("node279 WORLD8 preflight refused: " + message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-job-id", required=True)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        fail("output must be a fresh absolute path")
    if os.environ.get("SLURM_JOB_ID") != args.expected_job_id:
        fail("Slurm job identity differs")
    hostname = socket.gethostname().split(".", 1)[0]
    if hostname != args.expected_node:
        fail("node identity differs")

    try:
        import torch
        import torch.distributed as dist
    except Exception as error:  # pragma: no cover - exercised on AUH
        fail(f"PyTorch import failed: {error}")

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    if any(name not in os.environ for name in required):
        fail("torchrun rank environment is incomplete")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8 or not 0 <= rank < 8 or not 0 <= local_rank < 8:
        fail("WORLD8 rank envelope differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        fail("exactly eight ROCm devices are not visible")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        if dist.get_world_size() != 8 or dist.get_rank() != rank:
            fail("distributed process group differs")
        device = torch.device("cuda", local_rank)
        properties = torch.cuda.get_device_properties(local_rank)
        if "MI210" not in properties.name.upper():
            fail("GPU model is not MI210")
        if int(properties.total_memory) < 60_000_000_000:
            fail("GPU memory envelope is too small")

        left = torch.arange(1024, device=device, dtype=torch.bfloat16).reshape(32, 32)
        product = left @ left.transpose(0, 1)
        if not bool(torch.isfinite(product.float()).all().item()):
            fail("BF16 compute is non-finite")
        collective = torch.tensor([float(rank + 1)], device=device, dtype=torch.float32)
        dist.all_reduce(collective, op=dist.ReduceOp.SUM)
        if float(collective.item()) != 36.0:
            fail("WORLD8 all-reduce differs")
        free_bytes, total_bytes = torch.cuda.mem_get_info(local_rank)
        local = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": hostname,
            "device_name": properties.name,
            "device_total_bytes": int(properties.total_memory),
            "memory_free_bytes_after_probe": int(free_bytes),
            "memory_total_bytes_after_probe": int(total_bytes),
            "bf16_matmul_finite": True,
            "all_reduce_sum": float(collective.item()),
        }
        gathered: list[Any] = [None] * 8
        dist.all_gather_object(gathered, local)
        dist.barrier()
        if rank == 0:
            ranks = [item.get("rank") for item in gathered]
            local_ranks = [item.get("local_rank") for item in gathered]
            if ranks != list(range(8)) or local_ranks != list(range(8)):
                fail("rank-to-device mapping is not exact")
            payload = {
                "schema_version": SCHEMA,
                "status": "PASS",
                "scope": "hardware-and-distributed-runtime-only",
                "optimizer_updates": 0,
                "model_forward_calls": 0,
                "training_claim_authorized": False,
                "scientific_claim_authorized": False,
                "expected_job_id": args.expected_job_id,
                "expected_node": args.expected_node,
                "world_size": 8,
                "backend": dist.get_backend(),
                "torch_version": torch.__version__,
                "hip_version": getattr(torch.version, "hip", None),
                "all_reduce_expected_sum": 36.0,
                "ranks": gathered,
                "completed_unix_ns": time.time_ns(),
            }
            raw_without_digest = canonical_bytes(payload)
            payload["payload_sha256"] = hashlib.sha256(raw_without_digest).hexdigest()
            raw = canonical_bytes(payload)
            output.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            try:
                os.write(descriptor, raw)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directory = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            print(raw.decode("utf-8"), end="", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
