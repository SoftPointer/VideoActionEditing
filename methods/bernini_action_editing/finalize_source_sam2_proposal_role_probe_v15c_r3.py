#!/usr/bin/env python3
"""Verify the v15c-r3 sealed snapshot and atomically publish COMPLETE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


RELEASE_SCHEMA = "bernini-source-object-proposal-role-v15c-r3-release"
COMPLETE_SCHEMA = "bernini-source-object-proposal-role-v15c-r3-complete"
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_MEMBER_COUNT = 8


class FinalizeV15CR3Error(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_PATTERN.fullmatch(value) is None:
        raise FinalizeV15CR3Error(f"{label} is not lowercase SHA256")
    return value


def read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FinalizeV15CR3Error("JSON input must be one regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise FinalizeV15CR3Error("JSON input differs") from error
    if type(value) is not dict:
        raise FinalizeV15CR3Error("JSON input is not an object")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str) -> None:
    payload = dict(value)
    claimed = payload.pop(field, None)
    require_sha(claimed, field)
    if claimed != object_sha256(payload):
        raise FinalizeV15CR3Error(f"{field} self-hash differs")


def verify_release(root: Path, release_path: Path, expected_sha: str) -> Mapping[str, Any]:
    expected_sha = require_sha(expected_sha, "expected release file hash")
    root = root.resolve(strict=True)
    release_path = release_path.resolve(strict=True)
    if file_sha256(release_path) != expected_sha:
        raise FinalizeV15CR3Error("release file pin differs")
    release = read_json(release_path)
    if set(release) != {
        "schema_version",
        "tag",
        "member_count",
        "members",
        "dependency_files",
        "observer_only",
        "route_authorized",
        "release_sha256",
    }:
        raise FinalizeV15CR3Error("release exact keys differ")
    verify_self_hash(release, "release_sha256")
    members = release["members"]
    if (
        release["schema_version"] != RELEASE_SCHEMA
        or release["tag"] != "v15c-r3"
        or release["member_count"] != EXPECTED_MEMBER_COUNT
        or type(members) is not list
        or len(members) != EXPECTED_MEMBER_COUNT
        or release["dependency_files"]
        != ["methods/bernini_action_editing/assets/e00_source_sam2_proposal_role_probe_v15c_r3_release.json"]
        or release["observer_only"] is not True
        or release["route_authorized"] is not False
    ):
        raise FinalizeV15CR3Error("release semantics differ")
    paths = []
    for row in members:
        if type(row) is not dict or set(row) != {"path", "sha256", "size"}:
            raise FinalizeV15CR3Error("release member exact keys differ")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise FinalizeV15CR3Error("release member path differs")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or root not in path.resolve().parents
            or file_sha256(path) != require_sha(row["sha256"], "member hash")
            or path.stat().st_size != row["size"]
        ):
            raise FinalizeV15CR3Error("release member bytes differ")
        paths.append(row["path"])
    if paths != sorted(paths) or len(set(paths)) != EXPECTED_MEMBER_COUNT:
        raise FinalizeV15CR3Error("release member order differs")
    dependency = root / release["dependency_files"][0]
    if not dependency.is_file() or dependency.is_symlink():
        raise FinalizeV15CR3Error("release dependency differs")
    if dependency.resolve() != release_path:
        raise FinalizeV15CR3Error("release dependency path differs")
    return release


def _output_manifest(run_root: Path) -> Mapping[str, Any]:
    files = {}
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.name in {"COMPLETE.manifest.json", ".COMPLETE.tmp"}:
            continue
        if path.is_symlink() or run_root not in path.resolve().parents:
            raise FinalizeV15CR3Error("output member differs")
        relative = str(path.relative_to(run_root))
        files[relative] = {"sha256": file_sha256(path), "size": path.stat().st_size}
    return files


def write_complete(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve(strict=True)
    snapshot = args.snapshot.resolve(strict=True)
    release_path = args.release_manifest.resolve(strict=True)
    if snapshot.parent != run_root or snapshot.name != "sealed_code_snapshot":
        raise FinalizeV15CR3Error("snapshot placement differs")
    release = verify_release(snapshot, release_path, args.release_sha256)
    if (
        args.job_id != "143808"
        or args.node != "auh7-1b-gpu-292"
        or args.gpu_index != "0"
        or args.gpu_name != "AMD Instinct MI210"
        or (snapshot.stat().st_mode & 0o777) != 0o700
    ):
        raise FinalizeV15CR3Error("execution authority differs")
    required_outputs = {
        "tracks/track_receipt.json",
        "tracks/phase_coverage.safetensors",
        "tracks/output_manifest.json",
        "result.json",
        "postflight.json",
        "review/index.html",
        "review/overlay_receipt.json",
        "review/media_validation.json",
    }
    for relative in required_outputs:
        path = run_root / relative
        if not path.is_file() or path.is_symlink():
            raise FinalizeV15CR3Error(f"required output absent: {relative}")
    result = read_json(run_root / "result.json")
    postflight = read_json(run_root / "postflight.json")
    overlay = read_json(run_root / "review/overlay_receipt.json")
    for value, field in ((result, "receipt_sha256"), (postflight, "receipt_sha256"), (overlay, "receipt_sha256")):
        verify_self_hash(value, field)
    if (
        result.get("route_authorized") is not False
        or result.get("training_authorized") is not False
        or result.get("decode_authorized") is not False
        or postflight.get("human_audit_action") != "reject_only"
        or postflight.get("human_audit_may_authorize_route") is not False
        or overlay.get("human_audit_action") != "reject_only"
        or overlay.get("approve_action_available") is not False
    ):
        raise FinalizeV15CR3Error("observer-only output authority differs")
    inputs = {}
    for label, raw_path in (
        ("source", args.source),
        ("sam2_checkpoint", args.checkpoint),
        ("sam2_config", args.config),
        ("r6_receipt", args.r6_receipt),
        ("r6_tensors", args.r6_tensors),
    ):
        path = raw_path.resolve(strict=True)
        inputs[label] = {"path": str(path), "sha256": file_sha256(path), "size": path.stat().st_size}
    expected_input_hashes = {
        "source": "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de",
        "sam2_checkpoint": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
        "sam2_config": "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107",
        "r6_receipt": "8f081c990edd84a64ca35e78ca1de3d4ea6cf4b80bfcdec70bf54c51dc9ed959",
        "r6_tensors": "2535193d41a3405460bd152cd77bc61db7ef8ea6ba7cefd98f514f0787acc553",
    }
    if any(inputs[key]["sha256"] != digest for key, digest in expected_input_hashes.items()):
        raise FinalizeV15CR3Error("final input binding differs")
    complete = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "COMPLETE_OBSERVER_ONLY_REJECT_ONLY",
        "execution": {
            "parent_job_id": int(args.job_id),
            "node": args.node,
            "visible_gpu_index": int(args.gpu_index),
            "visible_gpu_count": 1,
            "visible_gpu_name": args.gpu_name,
            "fresh_run_root": str(run_root),
        },
        "code": {
            "snapshot_root": str(snapshot),
            "snapshot_mode": "0700",
            "release_manifest_file_sha256": file_sha256(release_path),
            "release_manifest_internal_sha256": release["release_sha256"],
            "members": release["members"],
        },
        "inputs": inputs,
        "outputs": _output_manifest(run_root),
        "human_audit_action": "reject_only",
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
    }
    complete["complete_sha256"] = object_sha256(complete)
    destination = run_root / "COMPLETE.manifest.json"
    temporary = run_root / ".COMPLETE.tmp"
    if destination.exists() or destination.is_symlink() or temporary.exists():
        raise FinalizeV15CR3Error("COMPLETE path is not fresh")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(complete))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    reopened = read_json(destination)
    verify_self_hash(reopened, "complete_sha256")
    if reopened != complete:
        raise FinalizeV15CR3Error("COMPLETE reopen differs")


def verify_complete(run_root: Path) -> None:
    run_root = run_root.resolve(strict=True)
    complete = read_json(run_root / "COMPLETE.manifest.json")
    if set(complete) != {
        "schema_version",
        "status",
        "execution",
        "code",
        "inputs",
        "outputs",
        "human_audit_action",
        "route_authorized",
        "decode_authorized",
        "training_authorized",
        "complete_sha256",
    }:
        raise FinalizeV15CR3Error("COMPLETE exact keys differ")
    verify_self_hash(complete, "complete_sha256")
    execution = complete.get("execution")
    code = complete.get("code")
    inputs = complete.get("inputs")
    outputs = complete.get("outputs")
    if (
        complete["schema_version"] != COMPLETE_SCHEMA
        or complete["status"] != "COMPLETE_OBSERVER_ONLY_REJECT_ONLY"
        or complete["human_audit_action"] != "reject_only"
        or complete["route_authorized"] is not False
        or complete["decode_authorized"] is not False
        or complete["training_authorized"] is not False
        or type(execution) is not dict
        or set(execution)
        != {
            "parent_job_id",
            "node",
            "visible_gpu_index",
            "visible_gpu_count",
            "visible_gpu_name",
            "fresh_run_root",
        }
        or execution["parent_job_id"] != 143808
        or execution["node"] != "auh7-1b-gpu-292"
        or execution["visible_gpu_index"] != 0
        or execution["visible_gpu_count"] != 1
        or execution["visible_gpu_name"] != "AMD Instinct MI210"
        or execution["fresh_run_root"] != str(run_root)
        or type(code) is not dict
        or set(code)
        != {
            "snapshot_root",
            "snapshot_mode",
            "release_manifest_file_sha256",
            "release_manifest_internal_sha256",
            "members",
        }
        or type(inputs) is not dict
        or set(inputs)
        != {"source", "sam2_checkpoint", "sam2_config", "r6_receipt", "r6_tensors"}
        or type(outputs) is not dict
        or not outputs
        or outputs != _output_manifest(run_root)
    ):
        raise FinalizeV15CR3Error("COMPLETE output replay differs")
    for relative, row in outputs.items():
        relative_path = Path(relative)
        path = run_root / relative_path
        if (
            type(row) is not dict
            or set(row) != {"sha256", "size"}
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not path.is_file()
            or path.is_symlink()
            or file_sha256(path) != require_sha(row["sha256"], "complete output hash")
            or path.stat().st_size != row["size"]
        ):
            raise FinalizeV15CR3Error("COMPLETE output member differs")
    for row in inputs.values():
        path = Path(row["path"]).resolve(strict=True)
        if set(row) != {"path", "sha256", "size"} or (
            file_sha256(path) != require_sha(row["sha256"], "complete input hash")
            or path.stat().st_size != row["size"]
        ):
            raise FinalizeV15CR3Error("COMPLETE input replay differs")
    release = verify_release(
        Path(code["snapshot_root"]),
        Path(code["snapshot_root"])
        / "methods/bernini_action_editing/assets/e00_source_sam2_proposal_role_probe_v15c_r3_release.json",
        code["release_manifest_file_sha256"],
    )
    if (
        code["snapshot_root"] != str(run_root / "sealed_code_snapshot")
        or code["snapshot_mode"] != "0700"
        or code["release_manifest_internal_sha256"] != release["release_sha256"]
        or code["members"] != release["members"]
    ):
        raise FinalizeV15CR3Error("COMPLETE code binding differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-release")
    verify.add_argument("--root", required=True, type=Path)
    verify.add_argument("--release-manifest", required=True, type=Path)
    verify.add_argument("--expected-sha256", required=True)
    complete = subparsers.add_parser("complete")
    complete.add_argument("--run-root", required=True, type=Path)
    complete.add_argument("--snapshot", required=True, type=Path)
    complete.add_argument("--release-manifest", required=True, type=Path)
    complete.add_argument("--release-sha256", required=True)
    complete.add_argument("--job-id", required=True)
    complete.add_argument("--node", required=True)
    complete.add_argument("--gpu-index", required=True)
    complete.add_argument("--gpu-name", required=True)
    complete.add_argument("--source", required=True, type=Path)
    complete.add_argument("--checkpoint", required=True, type=Path)
    complete.add_argument("--config", required=True, type=Path)
    complete.add_argument("--r6-receipt", required=True, type=Path)
    complete.add_argument("--r6-tensors", required=True, type=Path)
    verify_done = subparsers.add_parser("verify-complete")
    verify_done.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "verify-release":
        verify_release(args.root, args.release_manifest, args.expected_sha256)
    elif args.command == "complete":
        write_complete(args)
    elif args.command == "verify-complete":
        verify_complete(args.run_root)
    else:
        raise FinalizeV15CR3Error("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
