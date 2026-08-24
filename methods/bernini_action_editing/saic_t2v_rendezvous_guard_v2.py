#!/usr/bin/env python3
"""Fail-closed dynamic rendezvous lifecycle v2 for the SAIC full60 top-up.

The unchanged scientific generator is imported only after four torch-elastic
workers have authenticated one kernel-selected c10d port, one permanent
job-local create-only port claim, and one exact WORLD4 admission decision.
Completed and collision receipts carry no scientific or resume authority.

Version 2 makes receipt publication explicit on shared filesystems: writers
reopen and verify their public path after close, while readers treat only a
same-inode 0600/empty publication as temporarily not ready.  Every identity,
type, link-count, mode, canonical-JSON, schema, or digest discrepancy remains
an immediate fail-closed error.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence


RANK_SCHEMA_VERSION = "saic-t2v-topup-rendezvous-rank-v2"
CLAIM_SCHEMA_VERSION = "saic-t2v-topup-rendezvous-port-claim-v2"
DECISION_SCHEMA_VERSION = "saic-t2v-topup-rendezvous-admission-v2"
COLLISION_SCHEMA_VERSION = "saic-t2v-topup-rendezvous-port-collision-v2"
COMPLETION_SCHEMA_VERSION = "saic-t2v-topup-rendezvous-completion-v2"
JOB_AUDIT_SCHEMA_VERSION = "saic-t2v-topup-rendezvous-job-audit-v2"
PLAN_SCHEMA_VERSION = "saic-t2v-topup-rendezvous-dynamic-plan-v2"
ATTEMPT_SCHEMA_VERSION = "bernini-saic-pure-t2v-event-topup-generation-receipt-v2"
ATTEMPT_RECEIPT_BASENAME = "saic-event-topup-generation-receipt.json"
EXPECTED_RUNTIME_SHA256 = "3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36"
COLLISION_EXIT_STATUS = 86
WORLD_SIZE = 4
MAX_LAUNCH_ORDINAL = 16
WAIT_SECONDS = 90.0
MAX_RECEIPT_BYTES = 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")


class SAICT2VRendezvousGuardError(RuntimeError):
    """The operational rendezvous boundary failed closed."""


class _EvidenceNotReady(RuntimeError):
    """A same-inode create-only receipt is not publicly sealed yet."""

    def __init__(
        self, message: str, *, identity: Optional[tuple[int, int]] = None
    ) -> None:
        super().__init__(message)
        self.identity = identity


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_digest"] = object_sha256(result)
    return result


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    reserved_identity: tuple[int, int]
    try:
        observed = os.fstat(descriptor)
        reserved_identity = (observed.st_dev, observed.st_ino)
        leaf = path.lstat()
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o600
            or not stat.S_ISREG(leaf.st_mode)
            or stat.S_ISLNK(leaf.st_mode)
            or leaf.st_nlink != 1
            or (leaf.st_dev, leaf.st_ino) != reserved_identity
        ):
            raise SAICT2VRendezvousGuardError("receipt inode differs")
        offset = 0
        while offset < len(payload):
            wrote = os.write(descriptor, payload[offset:])
            if wrote <= 0:
                raise SAICT2VRendezvousGuardError("receipt write stalled")
            offset += wrote
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(payload) + 1) != payload:
            raise SAICT2VRendezvousGuardError("receipt payload differs before sealing")
        staged = os.fstat(descriptor)
        public = path.lstat()
        if (
            (staged.st_dev, staged.st_ino) != reserved_identity
            or not stat.S_ISREG(staged.st_mode)
            or staged.st_nlink != 1
            or stat.S_IMODE(staged.st_mode) != 0o600
            or staged.st_size != len(payload)
            or not stat.S_ISREG(public.st_mode)
            or stat.S_ISLNK(public.st_mode)
            or public.st_nlink != 1
            or (public.st_dev, public.st_ino) != reserved_identity
        ):
            raise SAICT2VRendezvousGuardError("staged receipt identity differs")
        _fsync_parent(path)
    finally:
        os.close(descriptor)

    # Reopen the public pathname while it is still 0600.  All checks and I/O
    # that can fail happen before the terminal publication transition.  Once
    # fchmod(0444) succeeds, no later failure is allowed to make the caller
    # report failure beside a success-looking sealed receipt.
    public_descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        public = os.fstat(public_descriptor)
        leaf = path.lstat()
        if (
            not stat.S_ISREG(public.st_mode)
            or public.st_nlink != 1
            or stat.S_IMODE(public.st_mode) != 0o600
            or public.st_size != len(payload)
            or (public.st_dev, public.st_ino) != reserved_identity
            or not stat.S_ISREG(leaf.st_mode)
            or stat.S_ISLNK(leaf.st_mode)
            or leaf.st_nlink != 1
            or (leaf.st_dev, leaf.st_ino) != reserved_identity
        ):
            raise SAICT2VRendezvousGuardError(
                "public staged receipt identity differs"
            )
        os.lseek(public_descriptor, 0, os.SEEK_SET)
        if os.read(public_descriptor, len(payload) + 1) != payload:
            raise SAICT2VRendezvousGuardError(
                "public staged receipt payload differs"
            )
    except BaseException:
        os.close(public_descriptor)
        raise
    os.fchmod(public_descriptor, 0o444)
    try:
        os.close(public_descriptor)
    except OSError:
        # Publication is already terminal.  A close error cannot revoke it or
        # create a contradictory failure terminal.
        pass


def exact_plain_file(path: Path, *, label: str, mode: Optional[int] = None) -> Path:
    if not path.is_absolute():
        raise SAICT2VRendezvousGuardError(f"{label} is not absolute")
    info = path.lstat()
    resolved = path.resolve(strict=True)
    if (
        resolved != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or (mode is not None and stat.S_IMODE(info.st_mode) != mode)
    ):
        raise SAICT2VRendezvousGuardError(f"{label} is not an exact plain file")
    return path


def _read_ready_bytes_once(
    path: Path,
    *,
    label: str,
    expected_identity: Optional[tuple[int, int]] = None,
) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError as error:
        if expected_identity is not None:
            raise SAICT2VRendezvousGuardError(
                f"{label} pinned pathname disappeared"
            ) from error
        raise _EvidenceNotReady(f"{label} is absent", identity=None) from error
    except OSError as error:
        raise SAICT2VRendezvousGuardError(f"{label} cannot be opened exactly") from error
    try:
        before = os.fstat(descriptor)
        try:
            leaf_before = path.lstat()
        except OSError as error:
            raise SAICT2VRendezvousGuardError(
                f"{label} pathname disappeared after open"
            ) from error
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not stat.S_ISREG(leaf_before.st_mode)
            or stat.S_ISLNK(leaf_before.st_mode)
            or leaf_before.st_nlink != 1
            or (leaf_before.st_dev, leaf_before.st_ino)
            != (before.st_dev, before.st_ino)
            or (
                expected_identity is not None
                and (before.st_dev, before.st_ino) != expected_identity
            )
        ):
            raise SAICT2VRendezvousGuardError(f"{label} pathname/inode differs")
        mode = stat.S_IMODE(before.st_mode)
        if mode == 0o600:
            raise _EvidenceNotReady(
                f"{label} same-inode publication is incomplete",
                identity=(before.st_dev, before.st_ino),
            )
        if mode != 0o444:
            raise SAICT2VRendezvousGuardError(f"{label} ready metadata differs")
        if before.st_size == 0:
            raise _EvidenceNotReady(
                f"{label} same-inode sealed size is not visible",
                identity=(before.st_dev, before.st_ino),
            )
        if before.st_size > MAX_RECEIPT_BYTES:
            raise SAICT2VRendezvousGuardError(f"{label} ready metadata differs")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65536))
            if not chunk:
                raise SAICT2VRendezvousGuardError(f"{label} truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SAICT2VRendezvousGuardError(f"{label} grew during read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        leaf = path.lstat()
    except OSError as error:
        raise SAICT2VRendezvousGuardError(f"{label} leaf disappeared") from error
    identity = lambda info: (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
    )
    if (
        identity(before) != identity(after)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or stat.S_IMODE(after.st_mode) != 0o444
        or not stat.S_ISREG(leaf.st_mode)
        or stat.S_ISLNK(leaf.st_mode)
        or leaf.st_nlink != 1
        or (leaf.st_dev, leaf.st_ino) != (before.st_dev, before.st_ino)
    ):
        raise SAICT2VRendezvousGuardError(f"{label} inode changed during read")
    return b"".join(chunks)


def read_ready_bytes(path: Path, *, label: str) -> bytes:
    try:
        return _read_ready_bytes_once(path, label=label)
    except _EvidenceNotReady as error:
        raise SAICT2VRendezvousGuardError(f"{label} is not sealed") from error


def wait_ready_bytes(
    path: Path,
    *,
    label: str,
    expected_identity: Optional[tuple[int, int]] = None,
) -> bytes:
    deadline = time.monotonic() + WAIT_SECONDS
    pinned_identity = expected_identity
    while time.monotonic() < deadline:
        try:
            return _read_ready_bytes_once(
                path, label=label, expected_identity=pinned_identity
            )
        except _EvidenceNotReady as error:
            if error.identity is not None:
                if pinned_identity is None:
                    pinned_identity = error.identity
                elif error.identity != pinned_identity:
                    raise SAICT2VRendezvousGuardError(
                        f"{label} provisional inode changed"
                    ) from error
            time.sleep(0.02)
    raise SAICT2VRendezvousGuardError(f"timed out waiting for {label}")


def ready_file_sha256(path: Path, *, label: str) -> str:
    return hashlib.sha256(read_ready_bytes(path, label=label)).hexdigest()


def wait_ready_file_sha256(path: Path, *, label: str) -> str:
    return hashlib.sha256(wait_ready_bytes(path, label=label)).hexdigest()


def _parse_identity(value: str, *, label: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise SAICT2VRendezvousGuardError(f"{label} identity differs")
    return int(parts[0]), int(parts[1])


def exact_directory(
    value: str | Path,
    *,
    label: str,
    expected_identity: Optional[str] = None,
    required_mode: Optional[int] = 0o700,
    allow_device_remap_same_inode: bool = False,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise SAICT2VRendezvousGuardError(f"{label} is not absolute")
    info = path.lstat()
    resolved = path.resolve(strict=True)
    if (
        resolved != path
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or (
            required_mode is not None
            and stat.S_IMODE(info.st_mode) != required_mode
        )
        or (
            required_mode is None
            and stat.S_IMODE(info.st_mode) & 0o022 != 0
        )
    ):
        raise SAICT2VRendezvousGuardError(f"{label} is not an exact private directory")
    if expected_identity is not None:
        expected_device, expected_inode = _parse_identity(
            expected_identity, label=label
        )
        if (
            info.st_ino != expected_inode
            or (not allow_device_remap_same_inode and info.st_dev != expected_device)
        ):
            raise SAICT2VRendezvousGuardError(f"{label} inode changed")
    return path


def load_sealed(
    path: Path,
    *,
    schema_version: str,
    exact_fields: Optional[set[str]] = None,
) -> dict[str, Any]:
    raw = read_ready_bytes(path, label="rendezvous evidence")
    return _decode_sealed(
        raw, schema_version=schema_version, exact_fields=exact_fields
    )


def _decode_sealed(
    raw: bytes,
    *,
    schema_version: str,
    exact_fields: Optional[set[str]] = None,
) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SAICT2VRendezvousGuardError(
            "rendezvous evidence encoding differs"
        ) from error
    if not isinstance(value, dict) or value.get("schema_version") != schema_version:
        raise SAICT2VRendezvousGuardError("rendezvous evidence schema differs")
    if exact_fields is not None and set(value) != exact_fields:
        raise SAICT2VRendezvousGuardError("rendezvous evidence fields differ")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    if not isinstance(claimed, str) or claimed != object_sha256(unsigned):
        raise SAICT2VRendezvousGuardError("rendezvous evidence digest differs")
    if raw != canonical_json_bytes(value) + b"\n":
        raise SAICT2VRendezvousGuardError("rendezvous evidence is not canonical")
    return value


def wait_load_sealed(
    path: Path,
    *,
    schema_version: str,
    exact_fields: Optional[set[str]] = None,
    label: str,
) -> dict[str, Any]:
    raw = wait_ready_bytes(path, label=label)
    return _decode_sealed(
        raw, schema_version=schema_version, exact_fields=exact_fields
    )


AUTHORITY = {
    "scientific_selection": False,
    "training": False,
    "optimizer": False,
    "resume_or_partial_reuse": False,
}
CLAIM_FIELDS = {
    "schema_version", "status", "slurm_job_id", "group_id", "candidate_index",
    "candidate_id", "launch_ordinal", "rdzv_backend", "rdzv_endpoint_request",
    "rdzv_id", "actual_master_addr", "actual_master_port",
    "lifecycle_dir", "lifecycle_dir_identity", "admission_receipt_path",
    "torch_disable_share_rdzv_tcp_store", "shared_tcp_store_bootstrap",
    "kernel_selected_free_port", "port_claim_create_only_across_both_groups_for_this_job",
    "generation_runtime_entered", "scientific_spec_changed", "authority", "receipt_digest",
}
RANK_FIELDS = {
    "schema_version", "status", "slurm_job_id", "group_id", "candidate_index",
    "candidate_id", "launch_ordinal", "rdzv_backend", "rdzv_endpoint_request",
    "rdzv_id", "actual_master_addr", "actual_master_port", "rank", "local_rank",
    "world_size", "local_world_size", "port_claim_receipt_digest", "runtime_sha256",
    "generation_runtime_entered_before_admission", "scientific_spec_changed", "authority",
    "torch_disable_share_rdzv_tcp_store", "shared_tcp_store_bootstrap",
    "receipt_digest",
}
DECISION_FIELDS = {
    "schema_version", "status", "slurm_job_id", "group_id", "candidate_index",
    "candidate_id", "launch_ordinal", "rdzv_id", "actual_master_addr",
    "actual_master_port", "world_size", "rank_order", "rank_packet_digests",
    "port_claim_receipt_digest", "runtime_sha256", "all_four_ranks_admitted",
    "generation_runtime_entry_authorized", "scientific_spec_changed", "authority",
    "torch_disable_share_rdzv_tcp_store", "shared_tcp_store_bootstrap",
    "receipt_digest",
}
COLLISION_FIELDS = {
    "schema_version", "status", "slurm_job_id", "group_id", "candidate_index",
    "candidate_id", "launch_ordinal", "rdzv_id", "actual_master_port",
    "existing_claim_receipt_digest", "existing_claim_sha256",
    "existing_admission_receipt_path", "existing_admission_receipt_digest",
    "generation_runtime_entered", "candidate_output_reuse_authorized", "authority",
    "receipt_digest",
}
COMPLETION_FIELDS = {
    "schema_version", "status", "slurm_job_id", "group_id", "candidate_index",
    "candidate_id", "launch_ordinal", "rdzv_id", "rdzv_backend",
    "rdzv_endpoint_request", "actual_master_addr", "actual_master_port",
    "kernel_selected_and_atomically_bound", "permanent_job_local_port_claim_path",
    "permanent_job_local_port_claim_sha256", "port_claim_receipt_digest",
    "exact_world4_rank_order", "rank_packet_digests", "admission_receipt_digest",
    "runtime_sha256", "candidate_output", "attempt_receipt_path",
    "attempt_receipt_sha256", "attempt_receipt_digest",
    "collision_retry_used_for_scientific_selection", "candidate_output_reused",
    "scientific_spec_changed", "authority", "receipt_digest",
    "torch_disable_share_rdzv_tcp_store", "shared_tcp_store_bootstrap",
}
PLAN_FIELDS = {
    "schema_version", "group_id", "slurm_job_id", "candidate_count",
    "fixed_order", "rdzv_backend", "rdzv_endpoint_request",
    "kernel_atomic_port_allocation", "numeric_ports_preregistered",
    "torch_disable_share_rdzv_tcp_store", "shared_tcp_store_bootstrap_required",
    "permanent_create_only_claim_across_both_groups",
    "maximum_operational_launches_per_candidate", "retry_condition",
    "scientific_candidate_set_or_order_changed_by_retry", "candidate_rows",
    "authority", "receipt_digest",
}
PLAN_ROW_FIELDS = {
    "candidate_index", "candidate_id", "envelope_path", "envelope_sha256",
    "candidate_id_sha256", "rdzv_id_prefix", "requested_rendezvous_endpoint",
    "numeric_port_preregistered",
}


def validate_identity(args: argparse.Namespace) -> None:
    if TOKEN.fullmatch(args.candidate_id) is None:
        raise SAICT2VRendezvousGuardError("candidate identity differs")
    candidate_digest = hashlib.sha256(args.candidate_id.encode("ascii")).hexdigest()
    exact_rdzv_id = (
        f"saic-{args.slurm_job_id}-{args.group_id}-c{args.candidate_index:02d}-"
        f"{candidate_digest[:16]}-l{args.launch_ordinal:02d}"
    )
    if (
        not args.slurm_job_id.isdigit()
        or args.group_id not in {"sp4-a", "sp4-b"}
        or not 0 <= args.candidate_index < 30
        or not 1 <= args.launch_ordinal <= MAX_LAUNCH_ORDINAL
        or TOKEN.fullmatch(args.expected_rdzv_id) is None
        or args.expected_rdzv_id != exact_rdzv_id
        or SHA256.fullmatch(args.expected_runtime_sha256) is None
        or args.expected_runtime_sha256 != EXPECTED_RUNTIME_SHA256
    ):
        raise SAICT2VRendezvousGuardError("rendezvous identity differs")


def identity_core(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "slurm_job_id": args.slurm_job_id,
        "group_id": args.group_id,
        "candidate_index": args.candidate_index,
        "candidate_id": args.candidate_id,
        "launch_ordinal": args.launch_ordinal,
        "rdzv_id": args.expected_rdzv_id,
    }


def _load_runtime(path_value: str, expected_sha256: str) -> ModuleType:
    path = Path(path_value)
    exact_plain_file(path, label="generation runtime")
    if file_sha256(path) != expected_sha256:
        raise SAICT2VRendezvousGuardError("generation runtime bytes differ")
    spec = importlib.util.spec_from_file_location("_sealed_saic_t2v_topup_runtime_v2", path)
    if spec is None or spec.loader is None:
        raise SAICT2VRendezvousGuardError("generation runtime import spec differs")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "main", None)):
        raise SAICT2VRendezvousGuardError("generation runtime main differs")
    return module


def _wait_for(path: Path, *, label: str) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if path.exists() or path.is_symlink():
            return
        time.sleep(0.02)
    raise SAICT2VRendezvousGuardError(f"timed out waiting for {label}")


def _validate_claim(value: Mapping[str, Any], *, port: int) -> None:
    if (
        value.get("status") != "reserved_before_generation_runtime"
        or value.get("actual_master_port") != port
        or value.get("rdzv_backend") != "c10d"
        or value.get("rdzv_endpoint_request") != "127.0.0.1:0"
        or value.get("actual_master_addr") != "127.0.0.1"
        or value.get("torch_disable_share_rdzv_tcp_store") != "0"
        or value.get("shared_tcp_store_bootstrap") is not True
        or not isinstance(value.get("lifecycle_dir"), str)
        or not isinstance(value.get("lifecycle_dir_identity"), str)
        or value.get("admission_receipt_path")
        != f"{value.get('lifecycle_dir')}/admission.json"
        or value.get("kernel_selected_free_port") is not True
        or value.get("port_claim_create_only_across_both_groups_for_this_job") is not True
        or value.get("generation_runtime_entered") is not False
        or value.get("scientific_spec_changed") is not False
        or value.get("authority") != AUTHORITY
    ):
        raise SAICT2VRendezvousGuardError("port claim content differs")


def _validate_generic_admission(
    claim: Mapping[str, Any],
    *,
    require_rank_packets: bool,
    allow_cross_mount_device_remap: bool = False,
) -> dict[str, Any]:
    lifecycle = exact_directory(
        str(claim["lifecycle_dir"]),
        label="existing admitted lifecycle directory",
        expected_identity=str(claim["lifecycle_dir_identity"]),
        allow_device_remap_same_inode=allow_cross_mount_device_remap,
    )
    admission_path = Path(str(claim["admission_receipt_path"]))
    if admission_path != lifecycle / "admission.json":
        raise SAICT2VRendezvousGuardError("existing admission path differs")
    decision = wait_load_sealed(
        admission_path,
        schema_version=DECISION_SCHEMA_VERSION,
        exact_fields=DECISION_FIELDS,
        label="prior same-job admission decision",
    )
    identity_fields = (
        "slurm_job_id", "group_id", "candidate_index", "candidate_id",
        "launch_ordinal", "rdzv_id", "actual_master_addr", "actual_master_port",
    )
    if (
        any(decision.get(key) != claim.get(key) for key in identity_fields)
        or decision.get("status") != "exact_world4_admitted_before_generation_runtime"
        or decision.get("world_size") != WORLD_SIZE
        or decision.get("rank_order") != list(range(WORLD_SIZE))
        or not isinstance(decision.get("rank_packet_digests"), list)
        or len(decision["rank_packet_digests"]) != WORLD_SIZE
        or any(SHA256.fullmatch(item) is None for item in decision["rank_packet_digests"])
        or decision.get("port_claim_receipt_digest") != claim.get("receipt_digest")
        or decision.get("runtime_sha256") != EXPECTED_RUNTIME_SHA256
        or decision.get("torch_disable_share_rdzv_tcp_store") != "0"
        or decision.get("shared_tcp_store_bootstrap") is not True
        or decision.get("all_four_ranks_admitted") is not True
        or decision.get("generation_runtime_entry_authorized") is not True
        or decision.get("scientific_spec_changed") is not False
        or decision.get("authority") != AUTHORITY
    ):
        raise SAICT2VRendezvousGuardError("existing admitted lifecycle differs")
    if require_rank_packets:
        observed_digests = []
        for rank in range(WORLD_SIZE):
            packet_path = lifecycle / f"rank-{rank}.json"
            packet = wait_load_sealed(
                packet_path,
                schema_version=RANK_SCHEMA_VERSION,
                exact_fields=RANK_FIELDS,
                label=f"prior same-job rank {rank} packet",
            )
            if (
                any(packet.get(key) != claim.get(key) for key in identity_fields)
                or packet.get("status") != "prepared_before_generation_runtime"
                or packet.get("rdzv_backend") != "c10d"
                or packet.get("rdzv_endpoint_request") != "127.0.0.1:0"
                or packet.get("rank") != rank
                or packet.get("local_rank") != rank
                or packet.get("world_size") != WORLD_SIZE
                or packet.get("local_world_size") != WORLD_SIZE
                or packet.get("port_claim_receipt_digest") != claim.get("receipt_digest")
                or packet.get("runtime_sha256") != EXPECTED_RUNTIME_SHA256
                or packet.get("torch_disable_share_rdzv_tcp_store") != "0"
                or packet.get("shared_tcp_store_bootstrap") is not True
                or packet.get("generation_runtime_entered_before_admission") is not False
                or packet.get("scientific_spec_changed") is not False
                or packet.get("authority") != AUTHORITY
            ):
                raise SAICT2VRendezvousGuardError("existing rank packet differs")
            observed_digests.append(packet["receipt_digest"])
        if observed_digests != decision["rank_packet_digests"]:
            raise SAICT2VRendezvousGuardError("existing WORLD4 packet linkage differs")
    return decision


def _claim_for_worker(
    *,
    args: argparse.Namespace,
    claim_root: Path,
    lifecycle: Path,
    port: int,
    address: str,
    shared_store_environment_value: str,
) -> tuple[Optional[dict[str, Any]], bool]:
    claim_path = claim_root / f"port-{port}.json"
    collision_path = lifecycle / "collision.json"
    expected_claim = seal(
        {
            "schema_version": CLAIM_SCHEMA_VERSION,
            "status": "reserved_before_generation_runtime",
            **identity_core(args),
            "rdzv_backend": "c10d",
            "rdzv_endpoint_request": "127.0.0.1:0",
            "actual_master_addr": address,
            "actual_master_port": port,
            "lifecycle_dir": str(lifecycle),
            "lifecycle_dir_identity": args.lifecycle_dir_identity,
            "admission_receipt_path": str(lifecycle / "admission.json"),
            "torch_disable_share_rdzv_tcp_store": shared_store_environment_value,
            "shared_tcp_store_bootstrap": True,
            "kernel_selected_free_port": True,
            "port_claim_create_only_across_both_groups_for_this_job": True,
            "generation_runtime_entered": False,
            "scientific_spec_changed": False,
            "authority": AUTHORITY,
        }
    )
    try:
        write_create_only(claim_path, expected_claim)
        return expected_claim, False
    except FileExistsError:
        existing = wait_load_sealed(
            claim_path,
            schema_version=CLAIM_SCHEMA_VERSION,
            exact_fields=CLAIM_FIELDS,
            label="existing same-job port claim final publication",
        )
        _validate_claim(existing, port=port)
        if existing.get("slurm_job_id") != args.slurm_job_id:
            raise SAICT2VRendezvousGuardError("foreign-job port claim is not retryable")
        if all(existing.get(key) == value for key, value in identity_core(args).items()):
            raise SAICT2VRendezvousGuardError("duplicate launch attempted the same port claim")
        existing_admission = _validate_generic_admission(
            existing,
            require_rank_packets=True,
        )
        collision = seal(
            {
                "schema_version": COLLISION_SCHEMA_VERSION,
                "status": "kernel_port_already_claimed_in_this_job_before_runtime",
                **identity_core(args),
                "actual_master_port": port,
                "existing_claim_receipt_digest": existing["receipt_digest"],
                "existing_claim_sha256": wait_ready_file_sha256(
                    claim_path, label="existing port claim"
                ),
                "existing_admission_receipt_path": existing["admission_receipt_path"],
                "existing_admission_receipt_digest": existing_admission["receipt_digest"],
                "generation_runtime_entered": False,
                "candidate_output_reuse_authorized": False,
                "authority": AUTHORITY,
            }
        )
        write_create_only(collision_path, collision)
        return None, True


def _validate_rank_packet(
    packet: Mapping[str, Any], *, args: argparse.Namespace, rank: int, port: int, claim_digest: str
) -> None:
    if (
        any(packet.get(key) != value for key, value in identity_core(args).items())
        or packet.get("status") != "prepared_before_generation_runtime"
        or packet.get("rdzv_backend") != "c10d"
        or packet.get("rdzv_endpoint_request") != "127.0.0.1:0"
        or packet.get("actual_master_addr") != "127.0.0.1"
        or packet.get("actual_master_port") != port
        or packet.get("rank") != rank
        or packet.get("local_rank") != rank
        or packet.get("world_size") != WORLD_SIZE
        or packet.get("local_world_size") != WORLD_SIZE
        or packet.get("port_claim_receipt_digest") != claim_digest
        or packet.get("runtime_sha256") != EXPECTED_RUNTIME_SHA256
        or packet.get("torch_disable_share_rdzv_tcp_store") != "0"
        or packet.get("shared_tcp_store_bootstrap") is not True
        or packet.get("generation_runtime_entered_before_admission") is not False
        or packet.get("scientific_spec_changed") is not False
        or packet.get("authority") != AUTHORITY
    ):
        raise SAICT2VRendezvousGuardError("rank admission packet differs")


def worker(args: argparse.Namespace) -> int:
    validate_identity(args)
    runtime_path = Path(args.runtime)
    exact_plain_file(runtime_path, label="generation runtime")
    if file_sha256(runtime_path) != args.expected_runtime_sha256:
        raise SAICT2VRendezvousGuardError("generation runtime bytes differ")
    claim_root = exact_directory(
        args.claim_root, label="port claim root", expected_identity=args.claim_root_identity
    )
    lifecycle = exact_directory(
        args.lifecycle_dir,
        label="candidate launch lifecycle directory",
        expected_identity=args.lifecycle_dir_identity,
    )
    environment = {
        name: os.environ.get(name, "")
        for name in (
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
            "MASTER_PORT", "TORCHELASTIC_RUN_ID",
            "TORCH_DISABLE_SHARE_RDZV_TCP_STORE",
        )
    }
    numeric = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_PORT")
    if (
        not all(environment[name].isdigit() for name in numeric)
        or int(environment["WORLD_SIZE"]) != WORLD_SIZE
        or int(environment["LOCAL_WORLD_SIZE"]) != WORLD_SIZE
        or not 0 <= int(environment["RANK"]) < WORLD_SIZE
        or int(environment["LOCAL_RANK"]) != int(environment["RANK"])
        or environment["MASTER_ADDR"] != "127.0.0.1"
        or not 1024 <= int(environment["MASTER_PORT"]) <= 65535
        or environment["TORCHELASTIC_RUN_ID"] != args.expected_rdzv_id
        or environment["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] != "0"
    ):
        raise SAICT2VRendezvousGuardError("torch elastic rendezvous environment differs")
    rank = int(environment["RANK"])
    port = int(environment["MASTER_PORT"])
    claim_path = claim_root / f"port-{port}.json"
    collision_path = lifecycle / "collision.json"
    if rank == 0:
        claim, collided = _claim_for_worker(
            args=args,
            claim_root=claim_root,
            lifecycle=lifecycle,
            port=port,
            address=environment["MASTER_ADDR"],
            shared_store_environment_value=environment[
                "TORCH_DISABLE_SHARE_RDZV_TCP_STORE"
            ],
        )
        if collided:
            return COLLISION_EXIT_STATUS
        assert claim is not None
    else:
        candidate = wait_load_sealed(
            claim_path,
            schema_version=CLAIM_SCHEMA_VERSION,
            exact_fields=CLAIM_FIELDS,
            label="rank-zero port claim",
        )
        _validate_claim(candidate, port=port)
        if any(
            candidate.get(key) != value
            for key, value in identity_core(args).items()
        ):
            if candidate.get("slurm_job_id") != args.slurm_job_id:
                raise SAICT2VRendezvousGuardError(
                    "foreign-job port claim is not retryable"
                )
            collision = wait_load_sealed(
                collision_path,
                schema_version=COLLISION_SCHEMA_VERSION,
                exact_fields=COLLISION_FIELDS,
                label="sealed same-job port collision",
            )
            if (
                any(
                    collision.get(key) != value
                    for key, value in identity_core(args).items()
                )
                or collision.get("existing_claim_receipt_digest")
                != candidate["receipt_digest"]
            ):
                raise SAICT2VRendezvousGuardError(
                    "worker collision evidence differs"
                )
            return COLLISION_EXIT_STATUS
        claim = candidate

    rank_packet = seal(
        {
            "schema_version": RANK_SCHEMA_VERSION,
            "status": "prepared_before_generation_runtime",
            **identity_core(args),
            "rdzv_backend": "c10d",
            "rdzv_endpoint_request": "127.0.0.1:0",
            "actual_master_addr": environment["MASTER_ADDR"],
            "actual_master_port": port,
            "rank": rank,
            "local_rank": int(environment["LOCAL_RANK"]),
            "world_size": WORLD_SIZE,
            "local_world_size": WORLD_SIZE,
            "port_claim_receipt_digest": claim["receipt_digest"],
            "runtime_sha256": args.expected_runtime_sha256,
            "torch_disable_share_rdzv_tcp_store": environment[
                "TORCH_DISABLE_SHARE_RDZV_TCP_STORE"
            ],
            "shared_tcp_store_bootstrap": True,
            "generation_runtime_entered_before_admission": False,
            "scientific_spec_changed": False,
            "authority": AUTHORITY,
        }
    )
    write_create_only(lifecycle / f"rank-{rank}.json", rank_packet)
    decision_path = lifecycle / "admission.json"
    if rank == 0:
        packets = []
        for expected_rank in range(WORLD_SIZE):
            packet_path = lifecycle / f"rank-{expected_rank}.json"
            packet = wait_load_sealed(
                packet_path,
                schema_version=RANK_SCHEMA_VERSION,
                exact_fields=RANK_FIELDS,
                label=f"rank {expected_rank} admission packet",
            )
            _validate_rank_packet(
                packet,
                args=args,
                rank=expected_rank,
                port=port,
                claim_digest=claim["receipt_digest"],
            )
            packets.append(packet)
        decision = seal(
            {
                "schema_version": DECISION_SCHEMA_VERSION,
                "status": "exact_world4_admitted_before_generation_runtime",
                **identity_core(args),
                "actual_master_addr": environment["MASTER_ADDR"],
                "actual_master_port": port,
                "world_size": WORLD_SIZE,
                "rank_order": list(range(WORLD_SIZE)),
                "rank_packet_digests": [packet["receipt_digest"] for packet in packets],
                "port_claim_receipt_digest": claim["receipt_digest"],
                "runtime_sha256": args.expected_runtime_sha256,
                "torch_disable_share_rdzv_tcp_store": claim[
                    "torch_disable_share_rdzv_tcp_store"
                ],
                "shared_tcp_store_bootstrap": True,
                "all_four_ranks_admitted": True,
                "generation_runtime_entry_authorized": True,
                "scientific_spec_changed": False,
                "authority": AUTHORITY,
            }
        )
        write_create_only(decision_path, decision)
    decision = wait_load_sealed(
        decision_path,
        schema_version=DECISION_SCHEMA_VERSION,
        exact_fields=DECISION_FIELDS,
        label="exact WORLD4 admission decision",
    )
    expected_digests = []
    for expected_rank in range(WORLD_SIZE):
        packet = wait_load_sealed(
            lifecycle / f"rank-{expected_rank}.json",
            schema_version=RANK_SCHEMA_VERSION,
            exact_fields=RANK_FIELDS,
            label=f"admitted rank {expected_rank} packet",
        )
        _validate_rank_packet(
            packet,
            args=args,
            rank=expected_rank,
            port=port,
            claim_digest=claim["receipt_digest"],
        )
        expected_digests.append(packet["receipt_digest"])
    expected_decision = seal(
        {
            "schema_version": DECISION_SCHEMA_VERSION,
            "status": "exact_world4_admitted_before_generation_runtime",
            **identity_core(args),
            "actual_master_addr": environment["MASTER_ADDR"],
            "actual_master_port": port,
            "world_size": WORLD_SIZE,
            "rank_order": list(range(WORLD_SIZE)),
            "rank_packet_digests": expected_digests,
            "port_claim_receipt_digest": claim["receipt_digest"],
            "runtime_sha256": args.expected_runtime_sha256,
            "torch_disable_share_rdzv_tcp_store": claim[
                "torch_disable_share_rdzv_tcp_store"
            ],
            "shared_tcp_store_bootstrap": True,
            "all_four_ranks_admitted": True,
            "generation_runtime_entry_authorized": True,
            "scientific_spec_changed": False,
            "authority": AUTHORITY,
        }
    )
    if decision != expected_decision:
        raise SAICT2VRendezvousGuardError("WORLD4 admission decision differs")
    runtime_args = list(args.runtime_args)
    if runtime_args[:1] == ["--"]:
        runtime_args = runtime_args[1:]
    if not runtime_args:
        raise SAICT2VRendezvousGuardError("generation runtime arguments are absent")
    module = _load_runtime(args.runtime, args.expected_runtime_sha256)
    return int(module.main(runtime_args))


def admit_collision(args: argparse.Namespace) -> int:
    validate_identity(args)
    claim_root = exact_directory(
        args.claim_root, label="port claim root", expected_identity=args.claim_root_identity
    )
    lifecycle = exact_directory(
        args.lifecycle_dir,
        label="candidate launch lifecycle directory",
        expected_identity=args.lifecycle_dir_identity,
    )
    output = Path(args.candidate_output)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise SAICT2VRendezvousGuardError("collision retry requires absent candidate output")
    collision = wait_load_sealed(
        lifecycle / "collision.json",
        schema_version=COLLISION_SCHEMA_VERSION,
        exact_fields=COLLISION_FIELDS,
        label="terminal collision receipt",
    )
    if any(collision.get(key) != value for key, value in identity_core(args).items()):
        raise SAICT2VRendezvousGuardError("collision identity differs")
    port = collision.get("actual_master_port")
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        raise SAICT2VRendezvousGuardError("collision port differs")
    claim_path = claim_root / f"port-{port}.json"
    claim = wait_load_sealed(
        claim_path,
        schema_version=CLAIM_SCHEMA_VERSION,
        exact_fields=CLAIM_FIELDS,
        label="collision port claim",
    )
    _validate_claim(claim, port=port)
    if claim.get("slurm_job_id") != args.slurm_job_id or all(
        claim.get(key) == value for key, value in identity_core(args).items()
    ):
        raise SAICT2VRendezvousGuardError("collision claim is foreign or duplicate")
    existing_admission = _validate_generic_admission(
        claim,
        require_rank_packets=True,
    )
    if (
        collision.get("status") != "kernel_port_already_claimed_in_this_job_before_runtime"
        or collision.get("existing_claim_receipt_digest") != claim["receipt_digest"]
        or collision.get("existing_claim_sha256")
        != wait_ready_file_sha256(claim_path, label="collision port claim")
        or collision.get("existing_admission_receipt_path")
        != claim["admission_receipt_path"]
        or collision.get("existing_admission_receipt_digest")
        != existing_admission["receipt_digest"]
        or collision.get("generation_runtime_entered") is not False
        or collision.get("candidate_output_reuse_authorized") is not False
        or collision.get("authority") != AUTHORITY
        or any(lifecycle.glob("rank-*.json"))
        or (lifecycle / "admission.json").exists()
        or (lifecycle / "admission.json").is_symlink()
    ):
        raise SAICT2VRendezvousGuardError("collision is not retry-authorizing evidence")
    print(canonical_json_bytes(collision).decode("ascii"))
    return 0


def assemble(args: argparse.Namespace) -> int:
    validate_identity(args)
    claim_root = exact_directory(
        args.claim_root, label="port claim root", expected_identity=args.claim_root_identity
    )
    lifecycle = exact_directory(
        args.lifecycle_dir,
        label="candidate launch lifecycle directory",
        expected_identity=args.lifecycle_dir_identity,
    )
    if (lifecycle / "collision.json").exists() or (lifecycle / "collision.json").is_symlink():
        raise SAICT2VRendezvousGuardError("successful lifecycle has collision evidence")
    packet_paths = [lifecycle / f"rank-{rank}.json" for rank in range(WORLD_SIZE)]
    if sorted(path.name for path in lifecycle.glob("rank-*.json")) != [
        path.name for path in packet_paths
    ]:
        raise SAICT2VRendezvousGuardError("successful lifecycle rank coverage differs")
    packets = [
        wait_load_sealed(
            path,
            schema_version=RANK_SCHEMA_VERSION,
            exact_fields=RANK_FIELDS,
            label=f"successful lifecycle rank {rank} packet",
        )
        for rank, path in enumerate(packet_paths)
    ]
    ports = {packet.get("actual_master_port") for packet in packets}
    if len(ports) != 1:
        raise SAICT2VRendezvousGuardError("successful lifecycle port consensus differs")
    port = next(iter(ports))
    if not isinstance(port, int):
        raise SAICT2VRendezvousGuardError("successful lifecycle port differs")
    claim_path = claim_root / f"port-{port}.json"
    claim = wait_load_sealed(
        claim_path,
        schema_version=CLAIM_SCHEMA_VERSION,
        exact_fields=CLAIM_FIELDS,
        label="successful lifecycle port claim",
    )
    _validate_claim(claim, port=port)
    if any(claim.get(key) != value for key, value in identity_core(args).items()):
        raise SAICT2VRendezvousGuardError("successful lifecycle claim identity differs")
    for rank, packet in enumerate(packets):
        _validate_rank_packet(
            packet,
            args=args,
            rank=rank,
            port=port,
            claim_digest=claim["receipt_digest"],
        )
    decision = wait_load_sealed(
        lifecycle / "admission.json",
        schema_version=DECISION_SCHEMA_VERSION,
        exact_fields=DECISION_FIELDS,
        label="successful lifecycle admission",
    )
    expected_decision = seal(
        {
            "schema_version": DECISION_SCHEMA_VERSION,
            "status": "exact_world4_admitted_before_generation_runtime",
            **identity_core(args),
            "actual_master_addr": "127.0.0.1",
            "actual_master_port": port,
            "world_size": WORLD_SIZE,
            "rank_order": list(range(WORLD_SIZE)),
            "rank_packet_digests": [packet["receipt_digest"] for packet in packets],
            "port_claim_receipt_digest": claim["receipt_digest"],
            "runtime_sha256": args.expected_runtime_sha256,
            "torch_disable_share_rdzv_tcp_store": claim[
                "torch_disable_share_rdzv_tcp_store"
            ],
            "shared_tcp_store_bootstrap": True,
            "all_four_ranks_admitted": True,
            "generation_runtime_entry_authorized": True,
            "scientific_spec_changed": False,
            "authority": AUTHORITY,
        }
    )
    if decision != expected_decision:
        raise SAICT2VRendezvousGuardError("successful lifecycle admission differs")
    candidate_output = exact_directory(
        args.candidate_output, label="candidate output", required_mode=None
    )
    attempt_path = candidate_output / ATTEMPT_RECEIPT_BASENAME
    attempt_raw = wait_ready_bytes(attempt_path, label="scientific attempt receipt")
    attempt = json.loads(attempt_raw.decode("ascii"))
    if not isinstance(attempt, dict):
        raise SAICT2VRendezvousGuardError("scientific attempt receipt differs")
    unsigned_attempt = dict(attempt)
    attempt_digest = unsigned_attempt.pop("receipt_digest", None)
    if (
        attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        or attempt_digest != object_sha256(unsigned_attempt)
        or attempt.get("group_id") != args.group_id
        or not isinstance(attempt.get("candidate"), dict)
        or attempt["candidate"].get("candidate_id") != args.candidate_id
        or attempt.get("event_verified") is not False
        or attempt.get("identity_preservation_verified") is not False
        or attempt.get("seed_selection_authorized") is not False
        or attempt.get("training_target_authorized") is not False
        or attempt.get("optimizer_or_parameter_update_authorized") is not False
    ):
        raise SAICT2VRendezvousGuardError("scientific attempt receipt binding differs")
    completion_path = Path(args.completion_receipt)
    if not completion_path.is_absolute() or completion_path.parent != lifecycle:
        raise SAICT2VRendezvousGuardError("completion receipt path differs")
    completion = seal(
        {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "status": "generation_runtime_completed_after_exact_world4_admission",
            **identity_core(args),
            "rdzv_backend": "c10d",
            "rdzv_endpoint_request": "127.0.0.1:0",
            "actual_master_addr": "127.0.0.1",
            "actual_master_port": port,
            "kernel_selected_and_atomically_bound": True,
            "permanent_job_local_port_claim_path": str(claim_path),
            "permanent_job_local_port_claim_sha256": wait_ready_file_sha256(
                claim_path, label="completed port claim"
            ),
            "port_claim_receipt_digest": claim["receipt_digest"],
            "exact_world4_rank_order": list(range(WORLD_SIZE)),
            "rank_packet_digests": [packet["receipt_digest"] for packet in packets],
            "admission_receipt_digest": decision["receipt_digest"],
            "runtime_sha256": args.expected_runtime_sha256,
            "torch_disable_share_rdzv_tcp_store": claim[
                "torch_disable_share_rdzv_tcp_store"
            ],
            "shared_tcp_store_bootstrap": True,
            "candidate_output": str(candidate_output),
            "attempt_receipt_path": str(attempt_path),
            "attempt_receipt_sha256": hashlib.sha256(attempt_raw).hexdigest(),
            "attempt_receipt_digest": attempt_digest,
            "collision_retry_used_for_scientific_selection": False,
            "candidate_output_reused": False,
            "scientific_spec_changed": False,
            "authority": AUTHORITY,
        }
    )
    write_create_only(completion_path, completion)
    print(canonical_json_bytes(completion).decode("ascii"))
    return 0


def _validate_completion(
    completion_path: Path, *, claim_root: Path, slurm_job_id: str
) -> dict[str, Any]:
    completion = wait_load_sealed(
        completion_path,
        schema_version=COMPLETION_SCHEMA_VERSION,
        exact_fields=COMPLETION_FIELDS,
        label="terminal completion receipt",
    )
    port = completion.get("actual_master_port")
    claim_path = claim_root / f"port-{port}.json"
    if (
        completion.get("status")
        != "generation_runtime_completed_after_exact_world4_admission"
        or completion.get("slurm_job_id") != slurm_job_id
        or completion.get("group_id") not in {"sp4-a", "sp4-b"}
        or not isinstance(completion.get("candidate_index"), int)
        or not 0 <= completion["candidate_index"] < 30
        or not isinstance(completion.get("candidate_id"), str)
        or TOKEN.fullmatch(completion["candidate_id"]) is None
        or not isinstance(completion.get("launch_ordinal"), int)
        or not 1 <= completion["launch_ordinal"] <= MAX_LAUNCH_ORDINAL
        or TOKEN.fullmatch(str(completion.get("rdzv_id", ""))) is None
        or completion.get("rdzv_backend") != "c10d"
        or completion.get("rdzv_endpoint_request") != "127.0.0.1:0"
        or completion.get("actual_master_addr") != "127.0.0.1"
        or not isinstance(port, int)
        or not 1024 <= port <= 65535
        or completion.get("kernel_selected_and_atomically_bound") is not True
        or completion.get("permanent_job_local_port_claim_path") != str(claim_path)
        or completion.get("exact_world4_rank_order") != list(range(WORLD_SIZE))
        or not isinstance(completion.get("rank_packet_digests"), list)
        or len(completion["rank_packet_digests"]) != WORLD_SIZE
        or any(SHA256.fullmatch(item) is None for item in completion["rank_packet_digests"])
        or completion.get("runtime_sha256") != EXPECTED_RUNTIME_SHA256
        or completion.get("torch_disable_share_rdzv_tcp_store") != "0"
        or completion.get("shared_tcp_store_bootstrap") is not True
        or completion.get("collision_retry_used_for_scientific_selection") is not False
        or completion.get("candidate_output_reused") is not False
        or completion.get("scientific_spec_changed") is not False
        or completion.get("authority") != AUTHORITY
    ):
        raise SAICT2VRendezvousGuardError("completed rendezvous lifecycle differs")
    claim = wait_load_sealed(
        claim_path,
        schema_version=CLAIM_SCHEMA_VERSION,
        exact_fields=CLAIM_FIELDS,
        label="terminal completion port claim",
    )
    _validate_claim(claim, port=port)
    identity_fields = (
        "slurm_job_id", "group_id", "candidate_index", "candidate_id",
        "launch_ordinal", "rdzv_id",
    )
    if (
        any(completion.get(key) != claim.get(key) for key in identity_fields)
        or claim.get("lifecycle_dir") != str(completion_path.parent)
        or claim.get("admission_receipt_path")
        != str(completion_path.parent / "admission.json")
        or completion.get("permanent_job_local_port_claim_sha256")
        != wait_ready_file_sha256(claim_path, label="audited port claim")
        or completion.get("port_claim_receipt_digest") != claim["receipt_digest"]
    ):
        raise SAICT2VRendezvousGuardError("completion-to-claim linkage differs")
    decision = _validate_generic_admission(
        claim,
        require_rank_packets=True,
        # VAST presents the same canonical path/inode under a different st_dev
        # on login and compute nodes.  Terminal auditors bind the canonical
        # path, inode, sealed claim/admission bytes, and child hashes; st_dev is
        # therefore allowed to remap only on this read-only verification path.
        allow_cross_mount_device_remap=True,
    )
    if (
        completion.get("rank_packet_digests") != decision["rank_packet_digests"]
        or completion.get("admission_receipt_digest") != decision["receipt_digest"]
    ):
        raise SAICT2VRendezvousGuardError("completion-to-WORLD4 linkage differs")
    candidate_output = exact_directory(
        str(completion.get("candidate_output")),
        label="completed candidate output",
        required_mode=None,
    )
    attempt_path = Path(str(completion.get("attempt_receipt_path")))
    if not attempt_path.is_absolute():
        raise SAICT2VRendezvousGuardError("completion attempt receipt is not absolute")
    if attempt_path != candidate_output / ATTEMPT_RECEIPT_BASENAME:
        raise SAICT2VRendezvousGuardError("completion attempt receipt path differs")
    attempt_raw = wait_ready_bytes(attempt_path, label="audited attempt receipt")
    attempt = json.loads(attempt_raw.decode("ascii"))
    unsigned_attempt = dict(attempt)
    attempt_digest = unsigned_attempt.pop("receipt_digest", None)
    if (
        attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        or attempt_digest != object_sha256(unsigned_attempt)
        or attempt.get("group_id") != completion["group_id"]
        or not isinstance(attempt.get("candidate"), dict)
        or attempt["candidate"].get("candidate_id") != completion["candidate_id"]
        or attempt.get("event_verified") is not False
        or attempt.get("identity_preservation_verified") is not False
        or attempt.get("seed_selection_authorized") is not False
        or attempt.get("training_target_authorized") is not False
        or attempt.get("optimizer_or_parameter_update_authorized") is not False
        or completion.get("attempt_receipt_sha256")
        != hashlib.sha256(attempt_raw).hexdigest()
        or completion.get("attempt_receipt_digest") != attempt_digest
    ):
        raise SAICT2VRendezvousGuardError("completion scientific receipt linkage differs")
    return completion


def _validate_collision_for_job_audit(
    collision_path: Path, *, claim_root: Path, slurm_job_id: str
) -> dict[str, Any]:
    collision = wait_load_sealed(
        collision_path,
        schema_version=COLLISION_SCHEMA_VERSION,
        exact_fields=COLLISION_FIELDS,
        label="terminal collision audit receipt",
    )
    port = collision.get("actual_master_port")
    if not isinstance(port, int):
        raise SAICT2VRendezvousGuardError("audited collision port differs")
    claim_path = claim_root / f"port-{port}.json"
    claim = wait_load_sealed(
        claim_path,
        schema_version=CLAIM_SCHEMA_VERSION,
        exact_fields=CLAIM_FIELDS,
        label="terminal collision prior claim",
    )
    _validate_claim(claim, port=port)
    current_identity = {
        key: collision.get(key)
        for key in (
            "slurm_job_id", "group_id", "candidate_index", "candidate_id",
            "launch_ordinal", "rdzv_id",
        )
    }
    old_identity = {key: claim.get(key) for key in current_identity}
    prior_admission = _validate_generic_admission(
        claim,
        require_rank_packets=True,
        allow_cross_mount_device_remap=True,
    )
    if (
        collision.get("status")
        != "kernel_port_already_claimed_in_this_job_before_runtime"
        or collision.get("slurm_job_id") != slurm_job_id
        or claim.get("slurm_job_id") != slurm_job_id
        or current_identity == old_identity
        or collision.get("existing_claim_receipt_digest") != claim["receipt_digest"]
        or collision.get("existing_claim_sha256")
        != wait_ready_file_sha256(claim_path, label="audited collision claim")
        or collision.get("existing_admission_receipt_path")
        != claim["admission_receipt_path"]
        or collision.get("existing_admission_receipt_digest")
        != prior_admission["receipt_digest"]
        or collision.get("generation_runtime_entered") is not False
        or collision.get("candidate_output_reuse_authorized") is not False
        or collision.get("authority") != AUTHORITY
    ):
        raise SAICT2VRendezvousGuardError("audited collision linkage differs")
    return collision


def _load_job_plans(
    output_root: Path, *, slurm_job_id: str
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for group_id in ("sp4-a", "sp4-b"):
        plan_path = output_root / "logs" / f"{group_id}-rendezvous-dynamic-plan-v1.json"
        plan = load_sealed(
            plan_path, schema_version=PLAN_SCHEMA_VERSION, exact_fields=PLAN_FIELDS
        )
        rows = plan.get("candidate_rows")
        if (
            plan.get("group_id") != group_id
            or plan.get("slurm_job_id") != slurm_job_id
            or plan.get("candidate_count") != 30
            or plan.get("fixed_order") != "lexicographic_envelope_basename"
            or plan.get("rdzv_backend") != "c10d"
            or plan.get("rdzv_endpoint_request") != "127.0.0.1:0"
            or plan.get("kernel_atomic_port_allocation") is not True
            or plan.get("numeric_ports_preregistered") is not False
            or plan.get("torch_disable_share_rdzv_tcp_store") != "0"
            or plan.get("shared_tcp_store_bootstrap_required") is not True
            or plan.get("permanent_create_only_claim_across_both_groups") is not True
            or plan.get("maximum_operational_launches_per_candidate")
            != MAX_LAUNCH_ORDINAL
            or plan.get("retry_condition")
            != "sealed_same_job_prior_admitted_port_claim_and_candidate_output_absent_only"
            or plan.get("scientific_candidate_set_or_order_changed_by_retry") is not False
            or plan.get("authority") != AUTHORITY
            or not isinstance(rows, list)
            or len(rows) != 30
        ):
            raise SAICT2VRendezvousGuardError("sealed dynamic rendezvous plan differs")
        plan_dir = exact_directory(
            output_root / "plan" / group_id,
            label=f"{group_id} scientific plan directory",
            required_mode=None,
        )
        expected_envelopes = sorted(plan_dir.glob("*.json"), key=lambda item: item.name)
        if len(expected_envelopes) != 30:
            raise SAICT2VRendezvousGuardError("sealed scientific plan count differs")
        for index, (row, envelope_path) in enumerate(zip(rows, expected_envelopes)):
            if not isinstance(row, dict) or set(row) != PLAN_ROW_FIELDS:
                raise SAICT2VRendezvousGuardError("dynamic rendezvous plan row fields differ")
            exact_plain_file(envelope_path, label="candidate envelope", mode=0o444)
            envelope_raw = read_ready_bytes(envelope_path, label="candidate envelope")
            envelope = json.loads(envelope_raw.decode("ascii"))
            candidate_id = envelope.get("candidate", {}).get("candidate_id")
            if not isinstance(candidate_id, str) or TOKEN.fullmatch(candidate_id) is None:
                raise SAICT2VRendezvousGuardError("candidate envelope identity differs")
            candidate_digest = hashlib.sha256(candidate_id.encode("ascii")).hexdigest()
            expected_prefix = (
                f"saic-{slurm_job_id}-{group_id}-c{index:02d}-{candidate_digest[:16]}"
            )
            if (
                row.get("candidate_index") != index
                or row.get("candidate_id") != candidate_id
                or row.get("envelope_path") != str(envelope_path)
                or row.get("envelope_sha256") != hashlib.sha256(envelope_raw).hexdigest()
                or row.get("candidate_id_sha256") != candidate_digest
                or row.get("rdzv_id_prefix") != expected_prefix
                or row.get("requested_rendezvous_endpoint") != "127.0.0.1:0"
                or row.get("numeric_port_preregistered") is not False
            ):
                raise SAICT2VRendezvousGuardError("dynamic rendezvous plan linkage differs")
        result[group_id] = rows
    return result


def audit_job(args: argparse.Namespace) -> int:
    if not args.slurm_job_id.isdigit() or args.expected_runtime_sha256 != EXPECTED_RUNTIME_SHA256:
        raise SAICT2VRendezvousGuardError("job audit identity differs")
    output_root = exact_directory(
        args.output_root, label="job output root", required_mode=None
    )
    claim_root = exact_directory(
        args.claim_root, label="port claim root", expected_identity=args.claim_root_identity
    )
    rendezvous_root = exact_directory(
        claim_root.parent, label="job rendezvous root"
    )
    if rendezvous_root != output_root / "logs/rendezvous":
        raise SAICT2VRendezvousGuardError("job rendezvous root escaped output")
    if sorted(item.name for item in rendezvous_root.iterdir()) != [
        "port-claims", "sp4-a", "sp4-b"
    ]:
        raise SAICT2VRendezvousGuardError("job rendezvous root has unknown lifecycle")
    plans = _load_job_plans(output_root, slurm_job_id=args.slurm_job_id)
    rows = []
    all_launch_rdzv_ids: list[str] = []
    referenced_claims: set[Path] = set()
    observed_candidates: dict[str, set[int]] = {"sp4-a": set(), "sp4-b": set()}
    for group_id in ("sp4-a", "sp4-b"):
        group_root = exact_directory(
            rendezvous_root / group_id, label=f"{group_id} rendezvous group"
        )
        candidate_dirs = sorted(group_root.iterdir(), key=lambda item: item.name)
        if len(candidate_dirs) != 30:
            raise SAICT2VRendezvousGuardError("job candidate lifecycle count differs")
        for candidate_dir in candidate_dirs:
            exact_directory(candidate_dir, label="candidate lifecycle root")
            launch_dirs = sorted(candidate_dir.iterdir(), key=lambda item: item.name)
            if not launch_dirs or len(launch_dirs) > MAX_LAUNCH_ORDINAL:
                raise SAICT2VRendezvousGuardError("candidate launch count differs")
            for expected_ordinal, launch_dir in enumerate(launch_dirs, start=1):
                if launch_dir.name != f"launch-{expected_ordinal:02d}":
                    raise SAICT2VRendezvousGuardError("candidate launch order differs")
                exact_directory(launch_dir, label="candidate launch lifecycle directory")
            completion_paths = [
                launch_dir / "completion.json"
                for launch_dir in launch_dirs
                if (launch_dir / "completion.json").exists()
                or (launch_dir / "completion.json").is_symlink()
            ]
            if len(completion_paths) != 1 or completion_paths[0].parent != launch_dirs[-1]:
                raise SAICT2VRendezvousGuardError("candidate completion placement differs")
            completion = _validate_completion(
                completion_paths[0], claim_root=claim_root, slurm_job_id=args.slurm_job_id
            )
            if (
                completion["group_id"] != group_id
                or completion["launch_ordinal"] != len(launch_dirs)
                or completion["candidate_index"] in observed_candidates[group_id]
                or completion["candidate_id"]
                != plans[group_id][completion["candidate_index"]]["candidate_id"]
                or completion["candidate_output"]
                != str(output_root / "attempts" / completion["candidate_id"])
                or completion["rdzv_id"]
                != f"{plans[group_id][completion['candidate_index']]['rdzv_id_prefix']}-l{completion['launch_ordinal']:02d}"
                or candidate_dir.name
                != f"candidate-{completion['candidate_index']:02d}-{hashlib.sha256(completion['candidate_id'].encode('ascii')).hexdigest()[:16]}"
            ):
                raise SAICT2VRendezvousGuardError("candidate lifecycle identity differs")
            observed_candidates[group_id].add(completion["candidate_index"])
            all_launch_rdzv_ids.append(completion["rdzv_id"])
            for launch_dir in launch_dirs[:-1]:
                entries = sorted(item.name for item in launch_dir.iterdir())
                if entries != ["collision.json"]:
                    raise SAICT2VRendezvousGuardError("retry launch contains non-collision evidence")
                collision = _validate_collision_for_job_audit(
                    launch_dir / "collision.json",
                    claim_root=claim_root,
                    slurm_job_id=args.slurm_job_id,
                )
                if (
                    collision["group_id"] != group_id
                    or collision["candidate_index"] != completion["candidate_index"]
                    or collision["candidate_id"] != completion["candidate_id"]
                    or collision["launch_ordinal"] != int(launch_dir.name[-2:])
                    or collision["rdzv_id"]
                    != f"{plans[group_id][completion['candidate_index']]['rdzv_id_prefix']}-l{collision['launch_ordinal']:02d}"
                ):
                    raise SAICT2VRendezvousGuardError("collision retry order differs")
                all_launch_rdzv_ids.append(collision["rdzv_id"])
            success_entries = sorted(item.name for item in launch_dirs[-1].iterdir())
            if success_entries != [
                "admission.json", "assemble.stdout", "completion.json",
                "rank-0.json", "rank-1.json", "rank-2.json", "rank-3.json",
            ]:
                raise SAICT2VRendezvousGuardError("successful launch evidence closure differs")
            referenced_claims.add(
                Path(completion["permanent_job_local_port_claim_path"])
            )
            rows.append(
                {
                    "group_id": group_id,
                    "candidate_index": completion["candidate_index"],
                    "candidate_id": completion["candidate_id"],
                    "successful_launch_ordinal": completion["launch_ordinal"],
                    "rdzv_id": completion["rdzv_id"],
                    "actual_master_port": completion["actual_master_port"],
                    "completion_receipt_digest": completion["receipt_digest"],
                }
            )
    if any(indexes != set(range(30)) for indexes in observed_candidates.values()):
        raise SAICT2VRendezvousGuardError("job candidate index coverage differs")
    claim_paths = set(claim_root.glob("port-*.json"))
    if (
        len(rows) != 60
        or len({row["candidate_id"] for row in rows}) != 60
        or len({row["rdzv_id"] for row in rows}) != 60
        or len({row["actual_master_port"] for row in rows}) != 60
        or len(claim_paths) != 60
        or claim_paths != referenced_claims
        or set(claim_root.iterdir()) != claim_paths
        or len(all_launch_rdzv_ids) != len(set(all_launch_rdzv_ids))
    ):
        raise SAICT2VRendezvousGuardError("job port/candidate exact60 uniqueness differs")
    receipt_path = Path(args.receipt)
    logs_root = exact_directory(
        output_root / "logs", label="job log root", required_mode=None
    )
    if not receipt_path.is_absolute() or receipt_path.parent != logs_root:
        raise SAICT2VRendezvousGuardError("job audit receipt path differs")
    rows.sort(key=lambda row: (row["group_id"], row["candidate_index"]))
    receipt = seal(
        {
            "schema_version": JOB_AUDIT_SCHEMA_VERSION,
            "status": "exact60_dynamic_rendezvous_lifecycles_admitted",
            "slurm_job_id": args.slurm_job_id,
            "topology": "two_concurrent_world4_sp4_groups_on_one_8gpu_node",
            "candidate_count": 60,
            "completion_receipt_count": 60,
            "rank_packet_count": 240,
            "group_candidate_counts": {"sp4-a": 30, "sp4-b": 30},
            "unique_rdzv_id_count": 60,
            "all_launch_rdzv_id_count": len(all_launch_rdzv_ids),
            "all_launch_rdzv_ids_unique": True,
            "unique_actual_master_port_count": 60,
            "permanent_port_claim_count": 60,
            "rdzv_backend": "c10d",
            "rdzv_endpoint_request": "127.0.0.1:0",
            "torch_disable_share_rdzv_tcp_store": "0",
            "shared_tcp_store_bootstrap": True,
            "fixed_candidate_order_preserved": True,
            "runtime_sha256": EXPECTED_RUNTIME_SHA256,
            "candidate_rows": rows,
            "scientific_spec_changed": False,
            "authority": AUTHORITY,
        }
    )
    write_create_only(receipt_path, receipt)
    print(canonical_json_bytes(receipt).decode("ascii"))
    return 0


def add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--group-id", choices=("sp4-a", "sp4-b"), required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--launch-ordinal", type=int, required=True)
    parser.add_argument("--expected-rdzv-id", required=True)
    parser.add_argument("--claim-root", required=True)
    parser.add_argument("--claim-root-identity", required=True)
    parser.add_argument("--lifecycle-dir", required=True)
    parser.add_argument("--lifecycle-dir-identity", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker_parser = commands.add_parser("worker")
    add_identity_arguments(worker_parser)
    worker_parser.add_argument("--runtime", required=True)
    worker_parser.add_argument("runtime_args", nargs=argparse.REMAINDER)
    collision_parser = commands.add_parser("admit-collision")
    add_identity_arguments(collision_parser)
    collision_parser.add_argument("--candidate-output", required=True)
    assemble_parser = commands.add_parser("assemble")
    add_identity_arguments(assemble_parser)
    assemble_parser.add_argument("--candidate-output", required=True)
    assemble_parser.add_argument("--completion-receipt", required=True)
    audit_parser = commands.add_parser("audit-job")
    audit_parser.add_argument("--expected-runtime-sha256", required=True)
    audit_parser.add_argument("--slurm-job-id", required=True)
    audit_parser.add_argument("--output-root", required=True)
    audit_parser.add_argument("--claim-root", required=True)
    audit_parser.add_argument("--claim-root-identity", required=True)
    audit_parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "worker":
        return worker(args)
    if args.command == "admit-collision":
        return admit_collision(args)
    if args.command == "assemble":
        return assemble(args)
    if args.command == "audit-job":
        return audit_job(args)
    raise SAICT2VRendezvousGuardError("rendezvous command differs")


if __name__ == "__main__":
    raise SystemExit(main())
