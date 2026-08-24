#!/usr/bin/env python3
"""Closed runtime helpers for the source-self exact81 engineering canary.

This module is intentionally independent of experimental training programs.
It contains the stable WORLD8 DP2 x Ulysses-SP4 default plus an explicit
WORLD4 DP1 x Ulysses-SP4 engineering profile, hashing, tokenization,
gradient-consensus, and create-only publication primitives consumed by
``train_source_self_role_repaint.py``.  Existing callers retain the single-node
WORLD8 default; callers must explicitly select any alternate topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
SP_GROUP_RANKS = ((0, 1, 2, 3), (4, 5, 6, 7))
DP_GROUP_RANKS = ((0, 4), (1, 5), (2, 6), (3, 7))
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_OUTPUT_STAGE_IDENTITIES: dict[str, tuple[int, int, int]] = {}


class SourceSelfRuntimeError(RuntimeError):
    """Raised before an ambiguous distributed update or publication."""


@dataclass(frozen=True)
class ParallelTopology:
    """Closed physical process topology; logical training arms live upstream."""

    profile: str
    world_size: int
    sp_size: int
    dp_size: int
    sp_group_ranks: tuple[tuple[int, ...], ...]
    dp_group_ranks: tuple[tuple[int, ...], ...]


WORLD8_DP2_SP4 = ParallelTopology(
    profile="world8-dp2-sp4",
    world_size=WORLD_SIZE,
    sp_size=SP_SIZE,
    dp_size=DP_SIZE,
    sp_group_ranks=SP_GROUP_RANKS,
    dp_group_ranks=DP_GROUP_RANKS,
)
WORLD4_DP1_SP4 = ParallelTopology(
    profile="world4-dp1-sp4",
    world_size=4,
    sp_size=4,
    dp_size=1,
    sp_group_ranks=((0, 1, 2, 3),),
    dp_group_ranks=((0,), (1,), (2,), (3,)),
)
PARALLEL_TOPOLOGIES = {
    WORLD8_DP2_SP4.profile: WORLD8_DP2_SP4,
    WORLD4_DP1_SP4.profile: WORLD4_DP1_SP4,
}


def parallel_topology(profile: str) -> ParallelTopology:
    try:
        return PARALLEL_TOPOLOGIES[profile]
    except (KeyError, TypeError) as error:
        raise SourceSelfRuntimeError(
            f"unsupported source-self topology profile: {profile!r}"
        ) from error


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SourceSelfRuntimeError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SourceSelfRuntimeError(f"{label} must be a lowercase SHA-256")
    return value


def _read_bound_bytes(
    path: Path, expected_sha256: str, *, label: str
) -> bytes:
    expected = require_sha256(expected_sha256, label=f"{label} expected SHA")
    try:
        before = path.stat()
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        raw = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise SourceSelfRuntimeError(f"cannot read {label}: {error}") from error
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(raw) != before.st_size:
        raise SourceSelfRuntimeError(f"{label} changed while reading")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SourceSelfRuntimeError(f"{label} SHA-256 differs")
    return raw


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
        named = path.stat()
    except OSError as error:
        raise SourceSelfRuntimeError(f"cannot hash {path}: {error}") from error
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or identity != (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
    ):
        raise SourceSelfRuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def tensor_sha256(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise SourceSelfRuntimeError(
            "tensor digest requires one materialized tensor"
        )
    tensor = value.detach().contiguous().cpu()
    metadata = canonical_json_bytes(
        {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    )
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def packed_output_field(patches: Any) -> Any:
    import torch

    if (
        not isinstance(patches, torch.Tensor)
        or patches.ndim != 5
        or tuple(int(item) for item in patches.shape[1:]) != (16, 1, 2, 2)
    ):
        raise SourceSelfRuntimeError(
            "packed output requires [N,16,1,2,2]"
        )
    return (
        patches.permute(0, 2, 3, 4, 1)
        .reshape(1, int(patches.shape[0]), 64)
        .contiguous()
    )


@dataclass(frozen=True)
class DistributedContract:
    world_size: int
    rank: int
    local_rank: int
    local_world_size: int
    topology: ParallelTopology

    @property
    def arm_index(self) -> int:
        return self.rank // self.topology.sp_size

    @property
    def sp_rank(self) -> int:
        return self.rank % self.topology.sp_size


@dataclass(frozen=True)
class ParallelContext:
    contract: DistributedContract
    world_group: Any
    sp_group: Any
    dp_group: Any


def distributed_contract(
    environment: Mapping[str, str] = os.environ,
    *,
    allow_multinode_dp2_sp4: bool = False,
    topology: ParallelTopology = WORLD8_DP2_SP4,
) -> DistributedContract:
    values: dict[str, int] = {}
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        raw = environment.get(name)
        if raw is None or not raw.isdecimal():
            raise SourceSelfRuntimeError(f"{name} must be a decimal integer")
        values[name] = int(raw)
    if topology not in PARALLEL_TOPOLOGIES.values():
        raise SourceSelfRuntimeError("source-self topology is not registered")
    if values["WORLD_SIZE"] != topology.world_size:
        raise SourceSelfRuntimeError(
            f"source-self requires exact WORLD{topology.world_size}"
        )
    if not 0 <= values["RANK"] < topology.world_size:
        raise SourceSelfRuntimeError(
            f"global rank differs from WORLD{topology.world_size}"
        )
    if allow_multinode_dp2_sp4:
        if topology != WORLD8_DP2_SP4:
            raise SourceSelfRuntimeError(
                "multinode admission is only defined for WORLD8 DP2 x SP4"
            )
        if values["LOCAL_WORLD_SIZE"] not in (
            topology.sp_size,
            topology.world_size,
        ):
            raise SourceSelfRuntimeError(
                "read-only DP2 x SP4 requires local world size 4 or 8"
            )
        if (
            not 0 <= values["LOCAL_RANK"] < values["LOCAL_WORLD_SIZE"]
            or values["RANK"] % values["LOCAL_WORLD_SIZE"]
            != values["LOCAL_RANK"]
        ):
            raise SourceSelfRuntimeError(
                "read-only DP2 x SP4 rank/local-rank mapping differs"
            )
    elif topology == WORLD4_DP1_SP4:
        if (
            values["LOCAL_WORLD_SIZE"] not in (2, topology.world_size)
            or not 0 <= values["LOCAL_RANK"] < values["LOCAL_WORLD_SIZE"]
            or values["RANK"] % values["LOCAL_WORLD_SIZE"]
            != values["LOCAL_RANK"]
        ):
            raise SourceSelfRuntimeError(
                "WORLD4 DP1 x SP4 requires exact 2x2 or 1x4 placement"
            )
    elif (
        values["LOCAL_WORLD_SIZE"] != topology.world_size
        or values["RANK"] != values["LOCAL_RANK"]
    ):
        raise SourceSelfRuntimeError(
            f"source-self requires one exact WORLD{topology.world_size} node"
        )
    return DistributedContract(
        values["WORLD_SIZE"],
        values["RANK"],
        values["LOCAL_RANK"],
        values["LOCAL_WORLD_SIZE"],
        topology,
    )


def initialise_distributed(contract: DistributedContract) -> Any:
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise SourceSelfRuntimeError(
            f"WORLD{contract.topology.world_size} source-self requires "
            "ROCm-visible accelerators"
        )
    if torch.cuda.device_count() != contract.local_world_size:
        raise SourceSelfRuntimeError(
            "visible accelerator count differs from local world size"
        )
    torch.cuda.set_device(contract.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    if (
        dist.get_world_size() != contract.topology.world_size
        or dist.get_rank() != contract.rank
    ):
        raise SourceSelfRuntimeError("initialized RCCL world differs from torchrun")
    return torch.device("cuda", contract.local_rank)


def _group_members(group: Any, expected: tuple[int, ...]) -> None:
    import torch.distributed as dist

    gathered: list[Any] = [None] * len(expected)
    dist.all_gather_object(gathered, dist.get_rank(), group=group)
    if tuple(gathered) != expected:
        raise SourceSelfRuntimeError(
            f"process group members differ: {gathered} != {expected}"
        )


def validate_parallel_state(
    contract: DistributedContract, state: Any
) -> ParallelContext:
    import torch.distributed as dist

    topology = contract.topology
    if (
        getattr(state, "world_size", None) != topology.world_size
        or getattr(state, "ulysses_size", None) != topology.sp_size
        or getattr(state, "dp_size", None) != topology.dp_size
        or getattr(state, "rank", None) != contract.rank
        or getattr(state, "ulysses_rank", None) != contract.sp_rank
        or getattr(state, "dp_rank", None) != contract.arm_index
    ):
        raise SourceSelfRuntimeError(
            f"Bernini {topology.profile} state differs"
        )
    _group_members(
        state.ulysses_group, topology.sp_group_ranks[contract.arm_index]
    )
    if topology.dp_size > 1:
        _group_members(state.dp_group, topology.dp_group_ranks[contract.sp_rank])
    return ParallelContext(
        contract, dist.group.WORLD, state.ulysses_group, state.dp_group
    )


def world_all_true(value: bool, *, group: Any) -> bool:
    import torch
    import torch.distributed as dist

    probe = torch.tensor(int(value), dtype=torch.int32, device="cuda")
    dist.all_reduce(probe, op=dist.ReduceOp.MIN, group=group)
    return bool(probe.item())


def digest_consensus(
    value: str, *, group: Any, expected_count: int, label: str
) -> str:
    import torch.distributed as dist

    gathered: list[Any] = [None] * expected_count
    dist.all_gather_object(gathered, value, group=group)
    if any(item != value for item in gathered):
        raise SourceSelfRuntimeError(f"{label} differs across replicated ranks")
    return value


def trainable_parameters_digest(
    named: Sequence[tuple[str, Any]],
) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in named:
        tensor = parameter.detach().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def synchronize_initial_parameters(
    named: Sequence[tuple[str, Any]],
    world_group: Any,
    *,
    expected_count: int = WORLD_SIZE,
) -> str:
    import torch.distributed as dist

    if not named:
        raise SourceSelfRuntimeError("adapter trainable parameter set is empty")
    for _, parameter in named:
        dist.broadcast(parameter.data, src=0, group=world_group)
    return digest_consensus(
        trainable_parameters_digest(named),
        group=world_group,
        expected_count=expected_count,
        label="initial adapter",
    )


def synchronize_gradients(
    named: Sequence[tuple[str, Any]], parallel: ParallelContext
) -> float:
    """Average over SP first, then over DP when the physical DP size exceeds one."""

    import torch
    import torch.distributed as dist

    if not named:
        raise SourceSelfRuntimeError("adapter trainable parameter set is empty")
    ready = all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        for _, parameter in named
    )
    if not world_all_true(ready, group=parallel.world_group):
        raise SourceSelfRuntimeError(
            "at least one adapter gradient is missing/non-finite"
        )
    topology = parallel.contract.topology
    squared = torch.zeros((), dtype=torch.float32, device="cuda")
    for _, parameter in named:
        assert parameter.grad is not None
        dist.all_reduce(
            parameter.grad, op=dist.ReduceOp.SUM, group=parallel.sp_group
        )
        parameter.grad.div_(float(topology.sp_size))
        if topology.dp_size > 1:
            dist.all_reduce(
                parameter.grad, op=dist.ReduceOp.SUM, group=parallel.dp_group
            )
            parameter.grad.div_(float(topology.dp_size))
        squared.add_(parameter.grad.float().square().sum())
    norm = float(squared.sqrt().item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise SourceSelfRuntimeError(
            "synchronized adapter gradient norm is zero/non-finite"
        )
    return norm


def parameter_consensus(
    named: Sequence[tuple[str, Any]],
    world_group: Any,
    label: str,
    *,
    expected_count: int = WORLD_SIZE,
) -> str:
    return digest_consensus(
        trainable_parameters_digest(named),
        group=world_group,
        expected_count=expected_count,
        label=label,
    )


def tokenize_generic_instruction(
    tokenizer: Any, instruction: str, device: Any
) -> dict[str, Any]:
    from bernini.training.data import encode_renderer_messages

    messages = [
        {"type": "video", "has_loss": 0},
        {"type": "text", "has_loss": 0, "text": instruction},
        {"type": "video_gen", "has_loss": 1},
    ]
    tokenized = encode_renderer_messages(
        messages, tokenizer, "mv2v", False, False, False
    )
    if (
        tokenized["vae_type_list"].tolist() != [1, 1]
        or tokenized["video_vit_mask"].tolist() != [False, True]
        or tokenized["video_drop_mask"].tolist() != [False, False]
    ):
        raise SourceSelfRuntimeError(
            "official tokenizer did not preserve source/target video roles"
        )
    return {
        "input_ids": tokenized["input_ids"].unsqueeze(0).to(device),
        "attention_mask": tokenized["attention_mask"].unsqueeze(0).to(device),
        "t5_input_lens": tokenized["t5_input_lens"].unsqueeze(0).to(device),
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_file_replace(temporary: Path, destination: Path) -> None:
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    fsync_directory(destination.parent)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        durable_file_replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    import torch

    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(value), temporary)
        durable_file_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise SourceSelfRuntimeError(
            f"{label} contains non-finite constant {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SourceSelfRuntimeError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SourceSelfRuntimeError(f"cannot decode {label}: {error}") from error
    if not isinstance(value, dict):
        raise SourceSelfRuntimeError(f"{label} root must be one object")
    return value


def verify_staged_run_bundle(stage: Path, receipt: Mapping[str, Any]) -> None:
    expected_files = {
        "adapter.safetensors",
        "optimizer.pt",
        "history.json",
        "receipt.json",
    }
    entries = list(stage.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise SourceSelfRuntimeError(
            "staged output contains a non-plain artifact"
        )
    if {path.name for path in entries} != expected_files:
        raise SourceSelfRuntimeError("staged output artifact set differs")
    expected_receipt = canonical_json_bytes(receipt) + b"\n"
    actual_receipt = _read_bound_bytes(
        stage / "receipt.json",
        hashlib.sha256(expected_receipt).hexdigest(),
        label="staged run receipt",
    )
    if actual_receipt != expected_receipt:
        raise SourceSelfRuntimeError(
            "staged receipt bytes differ from in-memory receipt"
        )
    parsed = _strict_json(actual_receipt, label="staged run receipt")
    if parsed != dict(receipt):
        raise SourceSelfRuntimeError(
            "staged receipt object differs from in-memory receipt"
        )
    declared = require_sha256(
        parsed.get("receipt_digest"), label="run receipt digest"
    )
    unsigned = dict(parsed)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        raise SourceSelfRuntimeError(
            "staged run receipt embedded digest differs"
        )
    artifacts = parsed.get("artifacts")
    expected_artifacts = expected_files - {"receipt.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise SourceSelfRuntimeError(
            "staged receipt artifact manifest differs"
        )
    for name in sorted(expected_artifacts):
        expected = require_sha256(
            artifacts[name], label=f"{name} artifact SHA"
        )
        if file_sha256(stage / name) != expected:
            raise SourceSelfRuntimeError(
                f"staged {name} differs from receipt"
            )


def prepare_output_transaction(
    path: str | Path, rank: int, world_group: Any
) -> tuple[Path, Path]:
    import torch.distributed as dist

    # Every rank validates only the lexical path and its already-existing
    # parent.  Nonzero ranks must never prime negative metadata entries for
    # output/staging on a remote VAST/NFS client.
    local: dict[str, Any]
    try:
        requested = Path(path).expanduser()
        if (
            not requested.is_absolute()
            or requested == Path("/")
            or requested.suffix
            or _SAFE_OUTPUT_NAME.fullmatch(requested.name) is None
        ):
            raise SourceSelfRuntimeError(
                "output must be an absolute safe suffix-free directory"
            )
        parent = requested.parent.resolve(strict=True)
        if not parent.is_dir() or requested != parent / requested.name:
            raise SourceSelfRuntimeError("output parent/path is not canonical")
        staging = parent / f".{requested.name}.staging"
        local = {
            "ok": True,
            "requested": str(requested),
            "parent": str(parent),
            "staging": str(staging),
        }
    except Exception as error:
        local = {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    gathered: list[Any] = [None] * dist.get_world_size(group=world_group)
    dist.all_gather_object(gathered, local, group=world_group)
    if any(not isinstance(item, dict) or item.get("ok") is not True for item in gathered):
        raise SourceSelfRuntimeError(
            f"output path validation failed across ranks: {gathered!r}"
        )
    identity = {
        (item["requested"], item["parent"], item["staging"])
        for item in gathered
    }
    if len(identity) != 1:
        raise SourceSelfRuntimeError("output path differs across ranks")
    requested = Path(local["requested"])
    parent = Path(local["parent"])
    staging = Path(local["staging"])

    reservation: list[Any] = [None]
    if rank == 0:
        try:
            _lstat_absent(requested, label="training output")
            os.mkdir(staging, mode=0o750)
            _lstat_absent(requested, label="training output")
            stage_stat = os.lstat(staging)
            if not stat.S_ISDIR(stage_stat.st_mode):
                raise SourceSelfRuntimeError("reserved output stage is not a directory")
            fsync_directory(parent)
            reservation[0] = {
                "ok": True,
                "requested": str(requested),
                "staging": str(staging),
                "stage_identity": [
                    int(stage_stat.st_dev),
                    int(stage_stat.st_ino),
                    stat.S_IMODE(stage_stat.st_mode),
                ],
            }
            _OUTPUT_STAGE_IDENTITIES[str(staging)] = (
                int(stage_stat.st_dev),
                int(stage_stat.st_ino),
                stat.S_IMODE(stage_stat.st_mode),
            )
        except Exception as error:
            reservation[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(reservation, src=0, group=world_group)
    result = reservation[0]
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise SourceSelfRuntimeError(f"cannot reserve output stage: {result!r}")
    if result.get("requested") != str(requested) or result.get("staging") != str(staging):
        raise SourceSelfRuntimeError("rank-zero output reservation path differs")
    return requested, staging


def _lstat_absent(path: Path, *, label: str) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise SourceSelfRuntimeError(f"{label} already exists")


def _try_renameat2_noreplace(
    parent_fd: int, source_name: str, destination_name: str
) -> int:
    """Return zero on success or the exact errno without mutating on failure."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return errno.ENOSYS
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(destination_name),
        1,
    )
    return 0 if result == 0 else ctypes.get_errno()


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish a sibling directory without clobbering an existing path.

    Linux NFS rejects ``RENAME_NOREPLACE`` with ``EINVAL``.  In that case an
    empty, inode-pinned directory at the final name is used as the create-only
    reservation before a same-parent rename.  A failed fallback deliberately
    retains that reservation as a tombstone; it never performs path-based
    cleanup of an inode that another process may have replaced.
    """
    if (
        not source.is_absolute()
        or not destination.is_absolute()
        or source.parent != destination.parent
        or source == destination
    ):
        raise SourceSelfRuntimeError("atomic publication paths differ")
    before = os.lstat(source)
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise SourceSelfRuntimeError("atomic publication source is not a directory")
    _lstat_absent(destination, label="training output")
    parent = source.parent
    parent_before = os.lstat(parent)
    if not stat.S_ISDIR(parent_before.st_mode) or stat.S_ISLNK(parent_before.st_mode):
        raise SourceSelfRuntimeError("atomic publication parent is not a directory")
    parent_fd = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        parent_opened = os.fstat(parent_fd)
        parent_identity = (parent_opened.st_dev, parent_opened.st_ino)
        if parent_identity != (parent_before.st_dev, parent_before.st_ino):
            raise SourceSelfRuntimeError("atomic publication parent changed")
        error_number = _try_renameat2_noreplace(
            parent_fd, source.name, destination.name
        )
        unsupported = {
            errno.EINVAL,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
            errno.ENOSYS,
        }
        if error_number in unsupported:
            reservation_fd: Optional[int] = None
            reservation_identity: Optional[tuple[int, ...]] = None
            try:
                os.mkdir(destination.name, 0o700, dir_fd=parent_fd)
                reservation_fd = os.open(
                    destination.name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                reservation = os.fstat(reservation_fd)
                if (
                    not stat.S_ISDIR(reservation.st_mode)
                    or stat.S_ISLNK(reservation.st_mode)
                    or stat.S_IMODE(reservation.st_mode) != 0o700
                    or reservation.st_nlink < 2
                    or os.listdir(reservation_fd)
                ):
                    raise SourceSelfRuntimeError(
                        "atomic publication reservation admission differs"
                    )
                reservation_identity = (
                    reservation.st_dev,
                    reservation.st_ino,
                    reservation.st_mode,
                    reservation.st_nlink,
                )
                source_now = os.stat(
                    source.name, dir_fd=parent_fd, follow_symlinks=False
                )
                destination_now = os.stat(
                    destination.name, dir_fd=parent_fd, follow_symlinks=False
                )
                parent_now = os.fstat(parent_fd)
                if (
                    (
                        source_now.st_dev,
                        source_now.st_ino,
                        source_now.st_mode,
                        source_now.st_nlink,
                    )
                    != (before.st_dev, before.st_ino, before.st_mode, before.st_nlink)
                    or (
                        destination_now.st_dev,
                        destination_now.st_ino,
                        destination_now.st_mode,
                        destination_now.st_nlink,
                    )
                    != reservation_identity
                    or (parent_now.st_dev, parent_now.st_ino) != parent_identity
                    or os.listdir(reservation_fd)
                ):
                    raise SourceSelfRuntimeError(
                        "atomic publication reservation changed before rename"
                    )
                # Linux permits the parent-directory rename while the empty
                # reservation itself is inert.  Darwin additionally requires
                # access to the destination directory for this replacement;
                # keep 0700 there so the model-free fallback test can exercise
                # the same inode checks.  Production AUH nodes are Linux.
                inert_mode = 0o000 if sys.platform.startswith("linux") else 0o700
                os.fchmod(reservation_fd, inert_mode)
                reservation = os.fstat(reservation_fd)
                reservation_identity = (
                    reservation.st_dev,
                    reservation.st_ino,
                    reservation.st_mode,
                    reservation.st_nlink,
                )
                source_now = os.stat(
                    source.name, dir_fd=parent_fd, follow_symlinks=False
                )
                destination_now = os.stat(
                    destination.name, dir_fd=parent_fd, follow_symlinks=False
                )
                parent_now = os.fstat(parent_fd)
                if (
                    stat.S_IMODE(reservation.st_mode) != inert_mode
                    or (
                        source_now.st_dev,
                        source_now.st_ino,
                        source_now.st_mode,
                        source_now.st_nlink,
                    )
                    != (before.st_dev, before.st_ino, before.st_mode, before.st_nlink)
                    or (
                        destination_now.st_dev,
                        destination_now.st_ino,
                        destination_now.st_mode,
                        destination_now.st_nlink,
                    )
                    != reservation_identity
                    or (parent_now.st_dev, parent_now.st_ino) != parent_identity
                ):
                    raise SourceSelfRuntimeError(
                        "atomic publication inert reservation changed before rename"
                    )
                os.fsync(parent_fd)
                os.rename(
                    source.name,
                    destination.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                error_number = 0
            finally:
                if reservation_fd is not None:
                    os.close(reservation_fd)
        if error_number == errno.EEXIST:
            raise SourceSelfRuntimeError(
                "training output appeared before publication"
            )
        if error_number != 0:
            raise OSError(error_number, os.strerror(error_number), str(destination))
        os.fsync(parent_fd)
        parent_after = os.lstat(parent)
        if (parent_after.st_dev, parent_after.st_ino) != parent_identity:
            raise SourceSelfRuntimeError("atomic publication parent identity changed")
    finally:
        os.close(parent_fd)
    after = os.lstat(destination)
    try:
        os.lstat(source)
    except FileNotFoundError:
        pass
    else:
        raise SourceSelfRuntimeError("atomic publication source remains after rename")
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise SourceSelfRuntimeError("atomic publication identity differs after rename")


def publish_output_transaction(
    output: Path,
    stage: Path,
    receipt: Optional[Mapping[str, Any]],
    rank: int,
    world_group: Any,
    *,
    rank_zero_error: Optional[str] = None,
) -> None:
    """Publish on rank zero and broadcast commit; remote ranks never stat paths."""
    import torch.distributed as dist

    local_identity = (str(output), str(stage))
    gathered: list[Any] = [None] * dist.get_world_size(group=world_group)
    dist.all_gather_object(gathered, local_identity, group=world_group)
    if any(item != local_identity for item in gathered):
        raise SourceSelfRuntimeError("publication path differs across ranks")
    publication: list[Any] = [None]
    if rank == 0:
        try:
            if rank_zero_error is not None:
                raise SourceSelfRuntimeError(rank_zero_error)
            if receipt is None:
                raise SourceSelfRuntimeError("rank zero publication receipt is missing")
            stage_stat = os.lstat(stage)
            identity = (
                int(stage_stat.st_dev),
                int(stage_stat.st_ino),
                stat.S_IMODE(stage_stat.st_mode),
            )
            if (
                not stat.S_ISDIR(stage_stat.st_mode)
                or _OUTPUT_STAGE_IDENTITIES.get(str(stage)) != identity
            ):
                raise SourceSelfRuntimeError("output stage identity differs")
            _lstat_absent(output, label="training output")
            verify_staged_run_bundle(stage, receipt)
            fsync_directory(stage)
            _rename_directory_noreplace(stage, output)
            fsync_directory(output.parent)
            output_stat = os.lstat(output)
            if not stat.S_ISDIR(output_stat.st_mode):
                raise SourceSelfRuntimeError("published output is not a directory")
            _lstat_absent(stage, label="training output stage")
            verify_staged_run_bundle(output, receipt)
            _OUTPUT_STAGE_IDENTITIES.pop(str(stage), None)
            publication[0] = {
                "ok": True,
                "output": str(output),
                "receipt_digest": receipt.get("receipt_digest"),
            }
        except Exception as error:
            publication[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(publication, src=0, group=world_group)
    result = publication[0]
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise SourceSelfRuntimeError(f"cannot publish output transaction: {result!r}")
    if result.get("output") != str(output):
        raise SourceSelfRuntimeError("rank-zero publication output differs")


__all__ = [
    "DP_GROUP_RANKS",
    "DP_SIZE",
    "DistributedContract",
    "PARALLEL_TOPOLOGIES",
    "ParallelContext",
    "ParallelTopology",
    "SP_GROUP_RANKS",
    "SP_SIZE",
    "SourceSelfRuntimeError",
    "WORLD_SIZE",
    "WORLD4_DP1_SP4",
    "WORLD8_DP2_SP4",
    "atomic_json",
    "atomic_torch_save",
    "digest_consensus",
    "distributed_contract",
    "durable_file_replace",
    "file_sha256",
    "fsync_directory",
    "initialise_distributed",
    "packed_output_field",
    "parallel_topology",
    "parameter_consensus",
    "prepare_output_transaction",
    "publish_output_transaction",
    "synchronize_gradients",
    "synchronize_initial_parameters",
    "tensor_sha256",
    "tokenize_generic_instruction",
    "trainable_parameters_digest",
    "validate_parallel_state",
    "verify_staged_run_bundle",
    "world_all_true",
]
