#!/usr/bin/env python3
"""Build/audit the deterministic seed-20260817 four-holder release."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any, Mapping, Sequence


SCHEMA = "bernini-self-generated-action-quotient-seed2-release-v1"
ENVELOPE_SCHEMA = "bernini-self-generated-action-quotient-seed2-deployment-v1"
RELEASE_GENERATION = "seed20260817-four-holder-r1"
MEMBER_ROOT = "methods/bernini_action_editing"
SEED = 20260817
SOURCE_DATA_MANIFEST_SHA256 = "62fee73b3d84015f2e72edcd4da14b51f7695980a4ba892420ca137aa50e9ad8"
SOURCE_DATA_MANIFEST_DIGEST = "2fb367ed6f06275705e0b71020dd87fd68e13a010e80ef0bd2a122c94070f503"
ANCESTRY_ARCHIVE_SHA256 = "6f44caf7ce865933dffdeb61bd69f89741b7381abe4d805dd6643becc61f1ada"
FORBIDDEN_SEED1_CACHE_SHA256 = "d96253fba3dac1b9602cc55bc71704f386c5ad17d4078992231df05da9b64a41"
DETACHED_CONTROLLER = "auh_launch_self_generated_action_quotient_seed2_four_holder_v1.sh"
FILES_AND_MODES: Mapping[str, int] = {
    "full30_action_learning_v1.py": 0o444,
    "materialize_self_generated_action_quotient_v1.py": 0o444,
    "scripts/auh_run_self_generated_action_quotient_v1.sh": 0o555,
    "self_generated_action_quotient_v1.py": 0o444,
    "train_lora.py": 0o444,
    "train_self_generated_action_quotient_v1.py": 0o444,
}
EXPECTED_CARRIED_SHA256 = {
    "full30_action_learning_v1.py": "67275ae09e7cb7b1e7e8fc43ce2928031b3fe8aabe213e8626000f37abad4ead",
    "materialize_self_generated_action_quotient_v1.py": "8b477d1bed3776737025b90c452fcf84286356b8213d69cefe8dd60b338aaf73",
    "self_generated_action_quotient_v1.py": "a9bfec2816ec1b6ccb2a336ea25600f15f22557aea76b1ea0605bbeb737b501c",
    "train_lora.py": "eae8eaac25197112637f466e611ba7eae574266d4cd1b83e625195fb22b0476e",
}
EXPECTED_REPLACEMENT_SHA256 = {
    "scripts/auh_run_self_generated_action_quotient_v1.sh": "d2861c22c6758879cf842a57479eb82a2e6372a4f35a51ab1cc2d491fa6cb85f",
    "train_self_generated_action_quotient_v1.py": "dd0a2a84272b015476b37a76af37bc1995f094d16d0d464cfa1c9b5ca668490c",
}


class ReleaseError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def method_root() -> Path:
    return Path(__file__).resolve().parents[1]


def payload_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    expected = {**EXPECTED_CARRIED_SHA256, **EXPECTED_REPLACEMENT_SHA256}
    for relative in sorted(FILES_AND_MODES):
        source = root / relative
        require(source.is_file() and not source.is_symlink(), f"member source differs: {relative}")
        raw = source.read_bytes()
        digest = sha256(raw)
        require(digest == expected[relative], f"member SHA differs: {relative}")
        payloads[relative] = raw
        rows.append(
            {"path": relative, "mode": FILES_AND_MODES[relative], "size": len(raw), "sha256": digest}
        )
    return rows, payloads


def content_revision(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha1(canonical(list(rows))).hexdigest()


def make_manifest(rows: list[dict[str, Any]], controller_sha: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "release_generation": RELEASE_GENERATION,
        "member_root": MEMBER_ROOT,
        "archive_format": "python-tarfile-ustar-sorted-owner0-mtime0-v1",
        "file_count": len(rows),
        "exact_member_closure": True,
        "files": rows,
        "content_revision": content_revision(rows),
        "allowed_entrypoints": [
            "scripts/auh_run_self_generated_action_quotient_v1.sh",
            DETACHED_CONTROLLER,
        ],
        "authority": {
            "training_kind": "real_optimizer_action_quotient_seed_replication",
            "seed": SEED,
            "initialization_seed": SEED,
            "teacher_cache_seed": SEED,
            "teacher_cache_must_be_fresh": True,
            "source_data_manifest_sha256": SOURCE_DATA_MANIFEST_SHA256,
            "source_data_manifest_digest": SOURCE_DATA_MANIFEST_DIGEST,
            "ancestry_archive_sha256": ANCESTRY_ARCHIVE_SHA256,
            "ancestry_archive_runtime_consumed": False,
            "forbidden_seed1_cache_sha256": FORBIDDEN_SEED1_CACHE_SHA256,
            "max_steps_per_arm": 160,
            "arms": [
                "action_only", "action_only_lowlr", "action_noop", "action_start",
                "action_nuisance", "action_start_nuisance",
                "action_start_nuisance_noop", "action_start_nuisance_border",
            ],
            "holders": {
                "136719": "auh7-1b-gpu-306", "136141": "auh7-1b-gpu-299",
                "136309": "auh7-1b-gpu-280", "136140": "auh7-1b-gpu-215",
            },
            "parent_cancel_release_requeue_forbidden": True,
            "automatic_retry": False,
            "scientific_claim_authorized": False,
            "experimental_training": True,
        },
        "component_sha256": {
            "trainer": EXPECTED_REPLACEMENT_SHA256["train_self_generated_action_quotient_v1.py"],
            "node_runner": EXPECTED_REPLACEMENT_SHA256["scripts/auh_run_self_generated_action_quotient_v1.sh"],
            "detached_controller": controller_sha,
        },
    }
    value["manifest_digest"] = sha256(canonical(value))
    return value


def make_archive(payloads: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for relative in sorted(payloads):
            raw = payloads[relative]
            info = tarfile.TarInfo(f"{MEMBER_ROOT}/{relative}")
            info.size = len(raw)
            info.mode = FILES_AND_MODES[relative]
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(raw))
    return output.getvalue()


def make_envelope(archive_raw: bytes, manifest_raw: bytes, manifest: Mapping[str, Any], controller_raw: bytes) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA,
        "release_generation": RELEASE_GENERATION,
        "seed": SEED,
        "remote_release_exact_entries": [
            DETACHED_CONTROLLER, "deployment-envelope.json", "source.manifest.json", "source.tar"
        ],
        "source_archive": {"basename": "source.tar", "sha256": sha256(archive_raw), "mode": 0o444},
        "source_manifest": {
            "basename": "source.manifest.json", "sha256": sha256(manifest_raw),
            "manifest_digest": manifest["manifest_digest"], "content_revision": manifest["content_revision"],
            "file_count": manifest["file_count"], "mode": 0o444,
        },
        "detached_controller": {"basename": DETACHED_CONTROLLER, "sha256": sha256(controller_raw), "mode": 0o555},
        "create_only_deployment_required": True,
        "fresh_experiment_root_required": True,
        "launch_authorized_by_user_request": True,
    }
    value["envelope_digest"] = sha256(canonical(value))
    return value


def write_create_only(path: Path, raw: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short write: {path}")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build(release_dir: Path) -> dict[str, Any]:
    root = method_root()
    controller_source = root / "scripts" / DETACHED_CONTROLLER
    require(controller_source.is_file() and not controller_source.is_symlink(), "detached controller source differs")
    controller_raw = controller_source.read_bytes()
    rows, payloads = payload_rows(root)
    manifest = make_manifest(rows, sha256(controller_raw))
    manifest_raw = canonical(manifest) + b"\n"
    archive_raw = make_archive(payloads)
    envelope = make_envelope(archive_raw, manifest_raw, manifest, controller_raw)
    envelope_raw = canonical(envelope) + b"\n"
    require(not release_dir.exists() and not release_dir.is_symlink(), "release directory is not fresh")
    release_dir.mkdir(mode=0o700, parents=False)
    write_create_only(release_dir / "source.tar", archive_raw, 0o444)
    write_create_only(release_dir / "source.manifest.json", manifest_raw, 0o444)
    write_create_only(release_dir / DETACHED_CONTROLLER, controller_raw, 0o555)
    write_create_only(release_dir / "deployment-envelope.json", envelope_raw, 0o444)
    descriptor = os.open(release_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return audit(release_dir, against_workspace=True)


def audit(release_dir: Path, *, against_workspace: bool) -> dict[str, Any]:
    expected_names = {"source.tar", "source.manifest.json", DETACHED_CONTROLLER, "deployment-envelope.json"}
    require(release_dir.is_dir() and not release_dir.is_symlink(), "release directory differs")
    require({path.name for path in release_dir.iterdir()} == expected_names, "release entry closure differs")
    for name, mode in {
        "source.tar": 0o444, "source.manifest.json": 0o444,
        DETACHED_CONTROLLER: 0o555, "deployment-envelope.json": 0o444,
    }.items():
        path = release_dir / name
        stat = path.stat()
        require(path.is_file() and not path.is_symlink() and stat.st_nlink == 1, f"release topology differs: {name}")
        require(stat.st_mode & 0o777 == mode, f"release mode differs: {name}")
    manifest_raw = (release_dir / "source.manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    unsigned = dict(manifest); declared = unsigned.pop("manifest_digest")
    require(declared == sha256(canonical(unsigned)), "manifest digest differs")
    require(manifest["schema_version"] == SCHEMA and manifest["release_generation"] == RELEASE_GENERATION, "manifest identity differs")
    require(manifest["file_count"] == len(FILES_AND_MODES), "manifest member count differs")
    rows = manifest["files"]
    require([row["path"] for row in rows] == sorted(FILES_AND_MODES), "manifest order differs")
    require(manifest["content_revision"] == content_revision(rows), "content revision differs")
    archive_raw = (release_dir / "source.tar").read_bytes()
    payloads: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as archive:
        members = archive.getmembers()
        expected_members = [f"{MEMBER_ROOT}/{relative}" for relative in sorted(FILES_AND_MODES)]
        require([member.name for member in members] == expected_members, "archive member closure/order differs")
        for member, row in zip(members, rows):
            require(member.isfile() and not member.linkname, f"archive member kind differs: {member.name}")
            require(member.uid == member.gid == member.mtime == 0, f"archive metadata differs: {member.name}")
            require(member.mode == row["mode"] and member.size == row["size"], f"archive row differs: {member.name}")
            handle = archive.extractfile(member); require(handle is not None, "archive payload absent")
            raw = handle.read(); require(sha256(raw) == row["sha256"], f"archive payload SHA differs: {member.name}")
            payloads[row["path"]] = raw
    if against_workspace:
        workspace_rows, workspace_payloads = payload_rows(method_root())
        require(rows == workspace_rows and payloads == workspace_payloads, "workspace payload closure differs")
        require((release_dir / DETACHED_CONTROLLER).read_bytes() == (method_root() / "scripts" / DETACHED_CONTROLLER).read_bytes(), "detached controller differs")
    envelope_raw = (release_dir / "deployment-envelope.json").read_bytes()
    envelope = json.loads(envelope_raw)
    envelope_unsigned = dict(envelope); envelope_declared = envelope_unsigned.pop("envelope_digest")
    require(envelope_declared == sha256(canonical(envelope_unsigned)), "envelope digest differs")
    require(envelope["source_archive"]["sha256"] == sha256(archive_raw), "envelope archive SHA differs")
    require(envelope["source_manifest"]["sha256"] == sha256(manifest_raw), "envelope manifest SHA differs")
    require(envelope["detached_controller"]["sha256"] == sha256((release_dir / DETACHED_CONTROLLER).read_bytes()), "envelope controller SHA differs")
    return {
        "static_audit_go": True,
        "release_dir": str(release_dir),
        "archive_sha256": sha256(archive_raw),
        "manifest_sha256": sha256(manifest_raw),
        "manifest_digest": manifest["manifest_digest"],
        "content_revision": manifest["content_revision"],
        "controller_sha256": envelope["detached_controller"]["sha256"],
        "envelope_sha256": sha256(envelope_raw),
        "envelope_digest": envelope["envelope_digest"],
        "file_count": len(rows),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--release-dir", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--release-dir", required=True)
    audit_parser.add_argument("--against-workspace", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    release_dir = Path(args.release_dir).resolve()
    result = build(release_dir) if args.command == "build" else audit(release_dir, against_workspace=args.against_workspace)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
