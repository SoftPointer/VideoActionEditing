#!/usr/bin/env python3
"""Fail-closed live-device guard for BRAID Stage-0 packet materialization.

``materialize_qmosaic_editor_runtime_v1.py`` deliberately owns the packet
mathematics.  This tiny entry point runs immediately before that materializer
inside each torchrun rank.  It observes the real process environment without
importing PyTorch, writes one create-only environment receipt, and then
``exec`` replaces itself with the authenticated materializer.

The guard is not a GPU result or a training authority.  It only establishes
that the dog and human WORLD4 processes saw their preregistered physical ROCm
visibility and no competing HIP/CUDA visibility alias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-braid-stage0-editor-packet-live-device-v1"
WORLD_SIZE = 4
CELL_VISIBLE_DEVICES = {
    "dog": (0, 1, 2, 3),
    "human": (4, 5, 6, 7),
}
FORBIDDEN_VISIBILITY_ALIASES = (
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class BraidStage0EditorPacketRankGuardError(RuntimeError):
    """Raised before the packet materializer can import PyTorch."""


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
        raise BraidStage0EditorPacketRankGuardError(
            "live-device evidence is not canonical JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise BraidStage0EditorPacketRankGuardError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _plain_file(value: str | Path, *, label: str, executable: bool = False) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or (executable and not os.access(path, os.X_OK))
    ):
        raise BraidStage0EditorPacketRankGuardError(
            f"{label} must be one canonical absolute plain file"
        )
    return path


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or not path.is_dir()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise BraidStage0EditorPacketRankGuardError(
            f"{label} must be one canonical absolute plain directory"
        )
    return path


def validate_live_environment(
    *,
    cell_id: str,
    expected_rocr_visible_devices: str,
    expected_guard_source_sha256: str,
    expected_materializer_source_sha256: str,
    environment: Optional[Mapping[str, str]] = None,
    imported_modules: Optional[Mapping[str, Any]] = None,
    guard_source: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Read and seal the exact per-rank environment before torch import.

    ``environment`` and ``imported_modules`` are test seams.  Production
    ``main`` deliberately omits both, so evidence comes from ``os.environ``
    and ``sys.modules`` in the rank process that immediately execs the
    materializer.
    """

    if cell_id not in CELL_VISIBLE_DEVICES:
        raise BraidStage0EditorPacketRankGuardError("unknown fixed packet cell")
    fixed_devices = CELL_VISIBLE_DEVICES[cell_id]
    fixed_rocr = ",".join(str(item) for item in fixed_devices)
    if expected_rocr_visible_devices != fixed_rocr:
        raise BraidStage0EditorPacketRankGuardError(
            "requested ROCR mapping differs from the fixed cell"
        )

    live = os.environ if environment is None else environment
    modules = sys.modules if imported_modules is None else imported_modules
    if "torch" in modules:
        raise BraidStage0EditorPacketRankGuardError(
            "PyTorch was imported before live-device validation"
        )
    if live.get("ROCR_VISIBLE_DEVICES") != fixed_rocr:
        raise BraidStage0EditorPacketRankGuardError(
            "live ROCR_VISIBLE_DEVICES differs from the fixed cell"
        )
    polluted = [name for name in FORBIDDEN_VISIBILITY_ALIASES if name in live]
    if polluted:
        raise BraidStage0EditorPacketRankGuardError(
            f"live device environment contains forbidden aliases: {polluted}"
        )

    parsed: dict[str, int] = {}
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        raw = live.get(name)
        if type(raw) is not str or not raw.isascii() or not raw.isdigit():
            raise BraidStage0EditorPacketRankGuardError(
                f"live torchrun {name} is absent or non-canonical"
            )
        parsed[name] = int(raw)
        if str(parsed[name]) != raw:
            raise BraidStage0EditorPacketRankGuardError(
                f"live torchrun {name} is non-canonical"
            )
    if (
        parsed["WORLD_SIZE"] != WORLD_SIZE
        or parsed["RANK"] not in range(WORLD_SIZE)
        or parsed["LOCAL_RANK"] != parsed["RANK"]
    ):
        raise BraidStage0EditorPacketRankGuardError(
            "live WORLD4 single-node rank topology differs"
        )

    guard_path = _plain_file(
        Path(__file__).resolve(strict=True) if guard_source is None else guard_source,
        label="rank guard source",
    )
    guard_sha = file_sha256(guard_path)
    if guard_sha != _sha256(
        expected_guard_source_sha256, label="rank guard source"
    ):
        raise BraidStage0EditorPacketRankGuardError(
            "executing rank guard source bytes differ"
        )
    materializer_sha = _sha256(
        expected_materializer_source_sha256,
        label="packet materializer source",
    )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id,
        "rank": parsed["RANK"],
        "local_rank": parsed["LOCAL_RANK"],
        "world_size": parsed["WORLD_SIZE"],
        "rocr_visible_devices": fixed_rocr,
        "physical_visible_devices": list(fixed_devices),
        "hip_visible_devices_unset": True,
        "cuda_visible_devices_unset": True,
        "gpu_device_ordinal_unset": True,
        "observed_before_torch_import": True,
        "rank_guard_source_sha256": guard_sha,
        "materializer_source_sha256": materializer_sha,
        "next_process_role": "qmosaic_editor_runtime_materializer_world4_rank",
        "decode_backward_optimizer_update_authority": False,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


_EVIDENCE_KEYS = {
    "schema_version",
    "cell_id",
    "rank",
    "local_rank",
    "world_size",
    "rocr_visible_devices",
    "physical_visible_devices",
    "hip_visible_devices_unset",
    "cuda_visible_devices_unset",
    "gpu_device_ordinal_unset",
    "observed_before_torch_import",
    "rank_guard_source_sha256",
    "materializer_source_sha256",
    "next_process_role",
    "decode_backward_optimizer_update_authority",
    "receipt_digest",
}


def validate_live_environment_receipt(
    value: Any,
    *,
    cell_id: str,
    rank: int,
    expected_guard_source_sha256: str,
    expected_materializer_source_sha256: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _EVIDENCE_KEYS:
        raise BraidStage0EditorPacketRankGuardError(
            "live-device receipt field closure differs"
        )
    row = dict(value)
    digest = row.pop("receipt_digest")
    fixed = CELL_VISIBLE_DEVICES.get(cell_id)
    if (
        fixed is None
        or digest != object_sha256(row)
        or value["schema_version"] != SCHEMA_VERSION
        or value["cell_id"] != cell_id
        or value["rank"] != rank
        or value["local_rank"] != rank
        or value["world_size"] != WORLD_SIZE
        or value["rocr_visible_devices"]
        != ",".join(str(item) for item in fixed)
        or value["physical_visible_devices"] != list(fixed)
        or value["hip_visible_devices_unset"] is not True
        or value["cuda_visible_devices_unset"] is not True
        or value["gpu_device_ordinal_unset"] is not True
        or value["observed_before_torch_import"] is not True
        or value["rank_guard_source_sha256"]
        != _sha256(expected_guard_source_sha256, label="rank guard source")
        or value["materializer_source_sha256"]
        != _sha256(
            expected_materializer_source_sha256,
            label="packet materializer source",
        )
        or value["next_process_role"]
        != "qmosaic_editor_runtime_materializer_world4_rank"
        or value["decode_backward_optimizer_update_authority"] is not False
    ):
        raise BraidStage0EditorPacketRankGuardError(
            "live-device receipt semantics or seal differs"
        )
    return dict(value)


def write_create_only_evidence(
    evidence_dir: str | Path, receipt: Mapping[str, Any]
) -> Path:
    cell_id = receipt.get("cell_id")
    rank = receipt.get("rank")
    if cell_id not in CELL_VISIBLE_DEVICES or type(rank) is not int:
        raise BraidStage0EditorPacketRankGuardError(
            "live-device evidence target coordinate differs"
        )
    root = _plain_directory(evidence_dir, label="live-device evidence directory")
    target = root / f"rank-{rank}.environment.json"
    if target.exists() or target.is_symlink():
        raise BraidStage0EditorPacketRankGuardError(
            "live-device evidence target is not fresh"
        )
    payload = canonical_json_bytes(dict(receipt)) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o400)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise BraidStage0EditorPacketRankGuardError(
            "cannot publish create-only live-device evidence"
        ) from error
    if target.read_bytes() != payload:
        raise BraidStage0EditorPacketRankGuardError(
            "reopened live-device evidence bytes differ"
        )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", choices=tuple(CELL_VISIBLE_DEVICES), required=True)
    parser.add_argument("--expected-rocr-visible-devices", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--materializer", required=True)
    parser.add_argument("--expected-guard-source-sha256", required=True)
    parser.add_argument("--expected-materializer-source-sha256", required=True)
    parser.add_argument("materializer_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    python_bin = _plain_file(args.python_bin, label="Python executable", executable=True)
    if Path(sys.executable).resolve(strict=True) != python_bin:
        raise BraidStage0EditorPacketRankGuardError(
            "rank guard interpreter differs from the authenticated launcher"
        )
    materializer = _plain_file(args.materializer, label="packet materializer")
    expected_materializer_sha = _sha256(
        args.expected_materializer_source_sha256,
        label="packet materializer source",
    )
    if file_sha256(materializer) != expected_materializer_sha:
        raise BraidStage0EditorPacketRankGuardError(
            "packet materializer source bytes differ before exec"
        )
    receipt = validate_live_environment(
        cell_id=args.cell_id,
        expected_rocr_visible_devices=args.expected_rocr_visible_devices,
        expected_guard_source_sha256=args.expected_guard_source_sha256,
        expected_materializer_source_sha256=expected_materializer_sha,
    )
    write_create_only_evidence(args.evidence_dir, receipt)
    materializer_args = list(args.materializer_args)
    if materializer_args[:1] == ["--"]:
        materializer_args.pop(0)
    if not materializer_args:
        raise BraidStage0EditorPacketRankGuardError(
            "packet materializer arguments are absent"
        )
    os.execv(
        python_bin,
        [str(python_bin), "-B", str(materializer), *materializer_args],
    )
    raise AssertionError("os.execv unexpectedly returned")


if __name__ == "__main__":  # pragma: no cover - exec boundary
    raise SystemExit(main())


__all__ = [
    "BraidStage0EditorPacketRankGuardError",
    "CELL_VISIBLE_DEVICES",
    "FORBIDDEN_VISIBILITY_ALIASES",
    "SCHEMA_VERSION",
    "WORLD_SIZE",
    "build_parser",
    "canonical_json_bytes",
    "file_sha256",
    "main",
    "object_sha256",
    "validate_live_environment",
    "validate_live_environment_receipt",
    "write_create_only_evidence",
]
