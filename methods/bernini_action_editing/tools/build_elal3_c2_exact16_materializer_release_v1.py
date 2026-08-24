#!/usr/bin/env python3
"""Build the deterministic, create-only ELAL-3 C2 exact16 VAE release."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-elal3-c2-exact16-materializer-release-v1"
ARCHIVE_FORMAT = "fixed-ustar-ascii-sorted-owner0-mtime0-record10240-v1"
MATERIALIZER_SHA256 = "b9142b3da63499163623248d902c51d01bf5c5295ff171125e3c614dea788c0f"
PACKET_MANIFEST_SHA256 = "2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"
DERIVATIVE_SHA256 = "543aedd714c7a48c48b4dcc19d1dd6a8bba37d1edda9b1fa195083659380c64a"
MODEL_SHA256 = "312d74a830ebec675af39b74e31c696ca188068a0e0ac9058745a537961c260d"
CONTRACT_SHA256 = "92d700bde0ff9c644f998344d3fecb48bc7c0361f6e948a93c42b924245b25f8"
CONTRACT_DIGEST = "18462dcfbeb017e48a7ed6816559667fa8de1911081261cdc103bc6dd9a229d6"

RUNTIME_PINS: Mapping[str, tuple[str, int]] = {
    "methods/bernini_action_editing/elal3_c0_v1.py": (
        "70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862", 31330),
    "methods/bernini_action_editing/elal3_simulator_c2_label_v1.py": (
        "1f09670a3dd2eae09cd27dbb5fe28c913f618d096a04a64fb0cf1dc9b6e1ec11", 76939),
    "methods/bernini_action_editing/materialize_elal3_simulator_c2_vae_v1.py": (
        MATERIALIZER_SHA256, 50334),
    "methods/bernini_action_editing/train_lora.py": (
        "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5", 66931),
    "methods/bernini_action_editing/tools/build_renderer_dataset.py": (
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5", 31012),
    "methods/bernini_action_editing/tools/materialize_vae.py": (
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0", 32195),
}
CONTROL_PINS: Mapping[str, tuple[str, int]] = {
    "md/action_editing/20260817_box/evidence/elal3_c2_real_model_authority_v1.json": (MODEL_SHA256, 3292),
    "md/action_editing/20260817_box/evidence/elal3_c2_role_binding_experiment_contract_v1.json": (CONTRACT_SHA256, 8553),
    "md/action_editing/20260817_box/evidence/elal3_c2_simulator_optimizer_diagnostic_authority_v1.json": (DERIVATIVE_SHA256, 1900),
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class ReleaseError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ReleaseError("non-canonical JSON value") from error


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def stable_file(path: Path, *, maximum: int = 8 << 20) -> bytes:
    require(path.is_absolute() and not path.is_symlink(), f"unsafe file: {path}")
    before_name = path.lstat()
    require(stat.S_ISREG(before_name.st_mode) and before_name.st_nlink == 1,
            f"non-plain file: {path}")
    require(before_name.st_size <= maximum, f"oversize file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) |
                 getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        first = os.read(fd, before.st_size + 1)
        os.lseek(fd, 0, os.SEEK_SET)
        second = os.read(fd, before.st_size + 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    ident = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                         row.st_uid, row.st_gid, row.st_size, row.st_mtime_ns,
                         row.st_ctime_ns)
    require(ident(before_name) == ident(before) == ident(after) == ident(path.lstat()),
            f"file identity changed: {path}")
    require(first == second and len(first) == before.st_size, f"file replay differs: {path}")
    return first


def strict_control(raw: bytes, *, seal: str, expected_digest: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"invalid {label} JSON") from error
    require(type(value) is dict, f"{label} is not an object")
    unsigned = dict(value)
    stored = unsigned.pop(seal, None)
    require(stored == expected_digest == digest(unsigned), f"{label} digest differs")
    return value


def tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o444
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info


def build_payload(repo_root: Path, train_lora_source: Path) -> tuple[bytes, dict[str, Any]]:
    root = repo_root.resolve(strict=True)
    require(root.is_dir() and not repo_root.is_symlink(), "repo root differs")
    payloads: dict[str, bytes] = {}
    for name, (expected_sha, expected_size) in {**RUNTIME_PINS, **CONTROL_PINS}.items():
        require(_SHA.fullmatch(expected_sha) is not None, f"invalid pin: {name}")
        path = train_lora_source if name.endswith("/train_lora.py") else root.joinpath(*PurePosixPath(name).parts)
        require(path.is_absolute(), f"source path is not absolute: {name}")
        raw = stable_file(path)
        require(len(raw) == expected_size and hashlib.sha256(raw).hexdigest() == expected_sha,
                f"SHA/size differs: {name}")
        if name.endswith(".py"):
            compile(raw, name, "exec")
        payloads[name] = raw
    derivative = strict_control(
        payloads["md/action_editing/20260817_box/evidence/elal3_c2_simulator_optimizer_diagnostic_authority_v1.json"],
        seal="authority_digest", expected_digest="936e91cf3d1d39dd7f45d5f7a4d510dadcbcb4c2f89a8d22581638fccdefd599",
        label="derivative authority")
    model = strict_control(
        payloads["md/action_editing/20260817_box/evidence/elal3_c2_real_model_authority_v1.json"],
        seal="authority_digest", expected_digest="c2c0c9037dea2fd56aa13ac56416bf38c6167686c75b69f0b4b568c82e670c1f",
        label="model authority")
    contract = strict_control(
        payloads["md/action_editing/20260817_box/evidence/elal3_c2_role_binding_experiment_contract_v1.json"],
        seal="contract_digest", expected_digest=CONTRACT_DIGEST, label="experiment contract")
    require(derivative.get("authorized_row_ids") == ["c2-three-entity-blocking-response", "c2-three-entity-handover-occlusion"],
            "derivative row scope differs")
    require(model.get("authorized_row_ids") == derivative["authorized_row_ids"], "model row scope differs")
    require(contract.get("authorized_row_ids") == derivative["authorized_row_ids"], "contract row scope differs")
    require(derivative.get("packet_manifest_sha256") == PACKET_MANIFEST_SHA256 and
            contract.get("packet_manifest_sha256") == PACKET_MANIFEST_SHA256,
            "packet binding differs")
    names = sorted(payloads, key=lambda value: value.encode("ascii"))
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in names:
            archive.addfile(tar_info(name, len(payloads[name])), io.BytesIO(payloads[name]))
    archive_raw = stream.getvalue()
    require(len(archive_raw) % 10240 == 0, "archive record boundary differs")
    rows = [{"path": name, "sha256": hashlib.sha256(payloads[name]).hexdigest(),
             "size": len(payloads[name]), "mode": "0444"} for name in names]
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "archive_format": ARCHIVE_FORMAT,
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "archive_size": len(archive_raw),
        "file_count": len(rows),
        "files": rows,
        "runtime_pins": {name: {"sha256": sha, "size": size} for name, (sha, size) in RUNTIME_PINS.items()},
        "authority_bindings": {
            "derivative_authority_sha256": DERIVATIVE_SHA256,
            "model_authority_sha256": MODEL_SHA256,
            "experiment_contract_sha256": CONTRACT_SHA256,
            "experiment_contract_digest": CONTRACT_DIGEST,
            "packet_manifest_sha256": PACKET_MANIFEST_SHA256,
        },
        "authorized_holder": {"holder_job_id": "141620", "node": "auh7-1b-gpu-226"},
        "materialization_scope": "frozen_bernini_vae_encode_c2_exact16_only",
        "archive_member_mode": "0444",
        "fresh_runtime_extract_file_mode_required_by_consumer": "0644",
        "fresh_runtime_extract_root_mode": "0555",
        "simulator_oracle_q_diagnostic_authorized": True,
        "formal_c2_authorized": False,
        "exact160_authorized": False,
        "source_instruction_inference_authorized": False,
        "real_video_generalization_authorized": False,
        "scientific_claim_authorized": False,
    }
    manifest = {**unsigned, "manifest_digest": digest(unsigned)}
    return archive_raw, manifest


def write_create_only(path: Path, payload: bytes, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            require(count > 0, "write made no progress")
            view = view[count:]
        os.fchmod(fd, mode)
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(repo_root: Path, train_lora_source: Path, output: Path) -> dict[str, Any]:
    require(output.is_absolute() and not output.exists() and not output.is_symlink(), "output must be fresh absolute path")
    archive, manifest = build_payload(repo_root, train_lora_source)
    os.mkdir(output, 0o700)
    manifest_raw = canonical(manifest) + b"\n"
    write_create_only(output / "source.tar", archive, 0o444)
    write_create_only(output / "source.manifest.json", manifest_raw, 0o444)
    os.chmod(output, 0o555)
    require(stable_file(output / "source.tar", maximum=len(archive)) == archive, "archive replay differs")
    require(stable_file(output / "source.manifest.json", maximum=len(manifest_raw)) == manifest_raw, "manifest replay differs")
    return {"archive_sha256": manifest["archive_sha256"], "archive_size": len(archive),
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(), "manifest_size": len(manifest_raw),
            "manifest_digest": manifest["manifest_digest"], "output": str(output)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--train-lora-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = publish(args.repo_root, args.train_lora_source, args.output)
    except (ReleaseError, OSError, SyntaxError) as error:
        print(f"[elal3-c2-materializer-release] ERROR: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
