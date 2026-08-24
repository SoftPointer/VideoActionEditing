#!/usr/bin/env python3
"""Deterministic exact-member executable release for checkpoint review v2."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tarfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-packed-preservation-checkpoint-review-release-v2"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-exact-modes-v2"
MEMBER_ROOT = "methods/bernini_action_editing"
ARCHIVE_NAME = "method.tar"
MANIFEST_NAME = "manifest.json"
RUNNER_MEMBER = "infer_packed_preservation_checkpoint_review_v2.py"
LAUNCHER_MEMBER = "scripts/auh_decode_packed_preservation_checkpoint_v2_job136309.sh"
FILES_AND_MODES = {
    "clean_source_visual_context_adapter_v1.py": 0o444,
    "clean_source_visual_context_checkpoint_review_contract_v1.py": 0o444,
    "clean_source_visual_context_training_v1.py": 0o444,
    "infer_lora.py": 0o444,
    "infer_native_identity_generation_canary.py": 0o444,
    "infer_native_v_axis_exact81_probe_v1.py": 0o444,
    "infer_orderless_source_frame_set_noise_canary.py": 0o444,
    "infer_packed_preservation_checkpoint_review_v2.py": 0o444,
    "infer_source_kv_carrier_oracle.py": 0o444,
    "infer_source_value_residual_oracle.py": 0o444,
    "native_i_axis_guidance.py": 0o444,
    "native_v_axis_guidance_v1.py": 0o444,
    "orderless_source_frame_set_noise.py": 0o444,
    "packed_preservation_checkpoint_review_release_v2.py": 0o444,
    "packed_preservation_checkpoint_review_v2.py": 0o444,
    "packed_preservation_lora_v2.py": 0o444,
    "source_kv_replay.py": 0o444,
    "source_kv_route_batches.py": 0o444,
    "source_value_residual.py": 0o444,
    "train_lora.py": 0o444,
    "tri_branch_unipc.py": 0o444,
    "tools/build_renderer_dataset.py": 0o444,
    "tools/materialize_vae.py": 0o444,
    "scripts/auh_packed_preservation_review_rank_exec_v2.sh": 0o555,
    LAUNCHER_MEMBER: 0o555,
}
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ReviewReleaseError(RuntimeError):
    """Raised when archive, manifest, and executed files are not one closure."""


def fail(message: str) -> NoReturn:
    raise ReviewReleaseError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ReviewReleaseError("release JSON is not canonical") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    named = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after) or identity(before) != identity(named):
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not stat.S_ISREG(resolved.lstat().st_mode):
        fail(f"{label} must be one canonical plain file")
    return resolved


def _method_root(value: str | Path, *, exact: bool) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail("method root must be an absolute non-symlink directory")
    root = requested.resolve(strict=True)
    if root != requested or not root.is_dir():
        fail("method root differs")
    for relative in FILES_AND_MODES:
        path = root / relative
        if path.resolve(strict=True) != path or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            fail(f"release member is not a canonical plain file: {relative}")
    if exact:
        observed: set[str] = set()
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                fail(f"executed method root contains symlink: {relative}")
            if stat.S_ISREG(mode):
                observed.add(relative)
            elif not stat.S_ISDIR(mode):
                fail(f"executed method root contains special entry: {relative}")
        if observed != set(FILES_AND_MODES):
            fail("executed method root is not the exact release file closure")
    return root


def _rows(root: Path) -> list[Mapping[str, Any]]:
    return [
        {
            "path": relative,
            "mode": FILES_AND_MODES[relative],
            "size": len(payload),
            "sha256": bytes_sha256(payload),
        }
        for relative in sorted(FILES_AND_MODES)
        for payload in [(root / relative).read_bytes()]
    ]


def content_closure_sha1(rows: Sequence[Mapping[str, Any]]) -> str:
    value = {"schema_version": SCHEMA_VERSION, "member_root": MEMBER_ROOT, "files": list(rows)}
    return hashlib.sha1(canonical_json_bytes(value)).hexdigest()


def build_release(*, method_root: str | Path, release_root: str | Path) -> Mapping[str, Any]:
    """Create a deterministic archive and adjacent canonical manifest."""

    root = _method_root(method_root, exact=False)
    release = Path(release_root).expanduser()
    if not release.is_absolute() or release.exists() or release.is_symlink() or release.parent.resolve(strict=True) != release.parent:
        fail("release root must be one fresh absolute path")
    release.mkdir(mode=0o700)
    archive = release / ARCHIVE_NAME
    manifest = release / MANIFEST_NAME
    rows = _rows(root)
    with archive.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as bundle:
            for row in rows:
                payload = (root / str(row["path"])).read_bytes()
                info = tarfile.TarInfo(f"{MEMBER_ROOT}/{row['path']}")
                info.size = len(payload)
                info.mode = int(row["mode"])
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.type = tarfile.REGTYPE
                bundle.addfile(info, io.BytesIO(payload))
        raw.flush()
        os.fsync(raw.fileno())
    archive_sha = file_sha256(archive)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "archive_format": ARCHIVE_FORMAT,
        "archive_file": ARCHIVE_NAME,
        "member_root": MEMBER_ROOT,
        "revision_kind": "content-closure-sha1",
        "method_revision": content_closure_sha1(rows),
        "archive_sha256": archive_sha,
        "exact_member_closure": True,
        "executed_root_required": True,
        "runner_member": RUNNER_MEMBER,
        "launcher_member": LAUNCHER_MEMBER,
        "file_count": len(rows),
        "files": rows,
    }
    value = {**unsigned, "manifest_digest": object_sha256(unsigned)}
    with manifest.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "release_root": str(release),
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "manifest_digest": value["manifest_digest"],
        "method_revision": value["method_revision"],
    }


def _manifest(path: Path) -> Mapping[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReviewReleaseError("cannot decode release manifest") from error
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value) + b"\n":
        fail("release manifest is not one canonical JSON object")
    return value


def validate_executed_release(
    *, executed_file: str | Path, executed_launcher: str | Path,
    manifest: str | Path, expected_manifest_sha256: str,
    expected_archive_sha256: str, expected_method_revision: str,
) -> Mapping[str, Any]:
    """Bind the running file to its adjacent archive and exact extracted root."""

    if (
        _SHA256.fullmatch(expected_manifest_sha256) is None
        or _SHA256.fullmatch(expected_archive_sha256) is None
        or _SHA1.fullmatch(expected_method_revision) is None
    ):
        fail("expected release identity differs")
    manifest_path = _plain_file(manifest, label="method manifest")
    if file_sha256(manifest_path) != expected_manifest_sha256:
        fail("method manifest SHA differs")
    value = _manifest(manifest_path)
    required = {
        "schema_version", "archive_format", "archive_file", "member_root",
        "revision_kind", "method_revision", "archive_sha256",
        "exact_member_closure", "executed_root_required", "runner_member",
        "launcher_member", "file_count", "files", "manifest_digest",
    }
    unsigned = dict(value)
    digest = unsigned.pop("manifest_digest", None)
    rows = value.get("files")
    if (
        set(value) != required
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("archive_format") != ARCHIVE_FORMAT
        or value.get("archive_file") != ARCHIVE_NAME
        or value.get("member_root") != MEMBER_ROOT
        or value.get("revision_kind") != "content-closure-sha1"
        or _SHA1.fullmatch(str(value.get("method_revision"))) is None
        or _SHA256.fullmatch(str(value.get("archive_sha256"))) is None
        or value.get("exact_member_closure") is not True
        or value.get("executed_root_required") is not True
        or value.get("runner_member") != RUNNER_MEMBER
        or value.get("launcher_member") != LAUNCHER_MEMBER
        or value.get("archive_sha256") != expected_archive_sha256
        or value.get("method_revision") != expected_method_revision
        or value.get("file_count") != len(FILES_AND_MODES)
        or not isinstance(rows, list)
        or digest != object_sha256(unsigned)
    ):
        fail("release manifest schema/digest differs")
    release = manifest_path.parent
    archive = _plain_file(release / ARCHIVE_NAME, label="adjacent method archive")
    if file_sha256(archive) != value["archive_sha256"]:
        fail("adjacent method archive SHA differs")
    root = _method_root(release / MEMBER_ROOT, exact=True)
    executed = _plain_file(executed_file, label="executed review runner")
    launcher = _plain_file(executed_launcher, label="executed review launcher")
    if executed != root / RUNNER_MEMBER:
        fail("executed runner is outside the derived exact release root")
    if launcher != root / LAUNCHER_MEMBER:
        fail("executed launcher is outside the derived exact release root")
    expected_rows = _rows(root)
    if rows != expected_rows or content_closure_sha1(expected_rows) != value["method_revision"]:
        fail("executed method-root bytes differ from release manifest")
    for row in expected_rows:
        if stat.S_IMODE((root / str(row["path"])).stat().st_mode) != row["mode"]:
            fail(f"executed release mode differs: {row['path']}")
    with tarfile.open(archive, mode="r:") as bundle:
        members = bundle.getmembers()
        names = [f"{MEMBER_ROOT}/{row['path']}" for row in expected_rows]
        if [member.name for member in members] != names:
            fail("archive exact member order/set differs")
        for member, row in zip(members, expected_rows):
            if (
                not member.isfile() or member.mode != row["mode"]
                or member.uid != 0 or member.gid != 0 or member.uname != ""
                or member.gname != "" or member.mtime != 0
                or member.size != row["size"] or member.pax_headers
            ):
                fail(f"archive member metadata differs: {member.name}")
            handle = bundle.extractfile(member)
            payload = b"" if handle is None else handle.read()
            if len(payload) != row["size"] or bytes_sha256(payload) != row["sha256"]:
                fail(f"archive member bytes differ: {member.name}")
    receipt = {
        "method_root": str(root),
        "archive": str(archive),
        "archive_sha256": value["archive_sha256"],
        "manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "manifest_digest": digest,
        "method_revision": value["method_revision"],
        "exact_member_count": len(expected_rows),
        "archive_members_verified": True,
        "executed_root_exact_closure_verified": True,
        "executed_file_bound": True,
        "executed_launcher_bound": True,
    }
    return {**receipt, "digest": object_sha256(receipt)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--release-root", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(build_release(method_root=args.method_root, release_root=args.release_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
