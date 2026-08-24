#!/usr/bin/env python3
"""WORLD8 DP2/SP4 RCCL and preservation-role admission smoke."""

from __future__ import annotations

import json
import os

import preservation_source_role_v1 as role


def _integer_environment(name: str) -> int:
    value = os.environ.get(name, "")
    if not value.isdecimal():
        raise RuntimeError(f"{name} differs")
    return int(value)


def main() -> int:
    import torch
    import torch.distributed as dist

    world = _integer_environment("WORLD_SIZE")
    rank = _integer_environment("RANK")
    local_rank = _integer_environment("LOCAL_RANK")
    local_world = _integer_environment("LOCAL_WORLD_SIZE")
    if (
        world != 8
        or local_world != 8
        or rank != local_rank
        or not 0 <= rank < world
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 8
        or getattr(torch.version, "hip", None) is None
    ):
        raise RuntimeError("WORLD8 single-node ROCm contract differs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    sp_members = ((0, 1, 2, 3), (4, 5, 6, 7))
    dp_members = ((0, 4), (1, 5), (2, 6), (3, 7))
    sp_groups = [dist.new_group(list(members)) for members in sp_members]
    dp_groups = [dist.new_group(list(members)) for members in dp_members]
    arm = rank // 4
    sp_rank = rank % 4
    sp_group = sp_groups[arm]
    dp_group = dp_groups[sp_rank]

    world_probe = torch.tensor(float(rank), device="cuda")
    dist.all_reduce(world_probe)
    if float(world_probe.item()) != 28.0:
        raise RuntimeError("WORLD8 RCCL sum differs")

    layout = role.TokenRoleLayout.contiguous(
        donor_tokens=5, reference_tokens=(2, 2, 2), target_tokens=7
    )
    invocation = role.RouteInvocation(
        layout,
        sequence_parallel_rank=sp_rank,
        sequence_parallel_size=4,
    )
    local_roles = invocation.local_roles(device=torch.device("cuda"))
    gathered = [torch.empty_like(local_roles) for _ in range(4)]
    dist.all_gather(gathered, local_roles, group=sp_group)
    reconstructed = torch.cat(gathered)
    expected = torch.tensor(
        layout.roles
        + (role.ROLE_PADDING,) * (invocation.local_length * 4 - layout.total_tokens),
        dtype=torch.int64,
        device="cuda",
    )
    if not torch.equal(reconstructed, expected):
        raise RuntimeError("SP4 role selector differs")

    dp_mean = torch.tensor(1.0 if arm == 0 else 3.0, device="cuda")
    dist.all_reduce(dp_mean, group=sp_group)
    dp_mean.div_(4.0)
    dist.all_reduce(dp_mean, group=dp_group)
    dp_mean.div_(2.0)
    if float(dp_mean.item()) != 2.0:
        raise RuntimeError("SP-then-DP mean differs")

    dist.barrier()
    if rank == 0:
        print(
            json.dumps(
                {
                    "world8_rccl": True,
                    "topology": "world8-dp2-sp4",
                    "sp4_role_selector": True,
                    "sp_then_dp_mean": 2.0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
