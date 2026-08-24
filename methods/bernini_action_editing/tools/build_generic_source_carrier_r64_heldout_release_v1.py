#!/usr/bin/env python3
"""Build/audit a deterministic exact-member R64 held-out decode release."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-generic-source-carrier-r64-heldout-release-v1"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-exact-modes-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
FILES_AND_MODES: Mapping[str, int] = {
    "clean_source_visual_context_adapter_v1.py": 0o444,
    "clean_source_visual_context_checkpoint_decode_runtime_v1.py": 0o444,
    "clean_source_visual_context_training_v1.py": 0o444,
    "generic_source_carrier_r64_heldout_contract_v1.py": 0o444,
    "infer_generic_source_carrier_r64_heldout_v1.py": 0o444,
    "infer_lora.py": 0o444,
    "infer_native_identity_generation_canary.py": 0o444,
    "infer_native_v_axis_exact81_probe_v1.py": 0o444,
    "infer_orderless_source_frame_set_noise_canary.py": 0o444,
    "infer_source_kv_carrier_oracle.py": 0o444,
    "infer_source_value_residual_oracle.py": 0o444,
    "native_i_axis_guidance.py": 0o444,
    "native_v_axis_guidance_v1.py": 0o444,
    "orderless_source_frame_set_noise.py": 0o444,
    "source_kv_replay.py": 0o444,
    "source_kv_route_batches.py": 0o444,
    "source_value_residual.py": 0o444,
    "train_lora.py": 0o444,
    "tri_branch_unipc.py": 0o444,
    "tools/build_generic_source_carrier_r64_heldout_html_v1.py": 0o444,
    "tools/build_generic_source_carrier_r64_heldout_release_v1.py": 0o444,
    "tools/build_renderer_dataset.py": 0o444,
    "tools/materialize_vae.py": 0o444,
    "tools/preflight_generic_source_carrier_r64_heldout_sources_v1.py": 0o444,
    "scripts/auh_generic_source_carrier_r64_heldout_holder_v1.sh": 0o555,
    "scripts/auh_generic_source_carrier_r64_heldout_rank_exec_v1.sh": 0o555,
}
COMPONENT_FILES = {
    "runner_sha256": "infer_generic_source_carrier_r64_heldout_v1.py",
    "contract_sha256": "generic_source_carrier_r64_heldout_contract_v1.py",
    "html_builder_sha256": (
        "tools/build_generic_source_carrier_r64_heldout_html_v1.py"
    ),
    "source_preflight_sha256": (
        "tools/preflight_generic_source_carrier_r64_heldout_sources_v1.py"
    ),
    "rank_exec_sha256": (
        "scripts/auh_generic_source_carrier_r64_heldout_rank_exec_v1.sh"
    ),
    "launcher_sha256": (
        "scripts/auh_generic_source_carrier_r64_heldout_holder_v1.sh"
    ),
}


class R64HeldoutReleaseError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise R64HeldoutReleaseError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise R64HeldoutReleaseError("release is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_plain_bytes(path: Path) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        fail("release member must be absolute and non-symlink")
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if resolved != path or not stat.S_ISREG(before.st_mode):
        fail("release member must be one canonical plain file")
    raw = resolved.read_bytes()
    after = resolved.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size or not raw
    ):
        fail("release member changed while reading or is empty")
    return raw


def build_manifest(method_root: Path) -> tuple[Mapping[str, Any], Mapping[str, bytes]]:
    root = method_root.resolve(strict=True)
    if root != method_root or root.is_symlink() or not root.is_dir():
        fail("method root must be one canonical directory")
    rows: list[Mapping[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for relative in sorted(FILES_AND_MODES):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            fail("release member path differs")
        raw = _stable_plain_bytes(root / relative)
        payloads[relative] = raw
        rows.append(
            {
                "path": relative, "mode": FILES_AND_MODES[relative],
                "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    by_path = {str(row["path"]): row for row in rows}
    pins = {
        label: by_path[relative]["sha256"]
        for label, relative in COMPONENT_FILES.items()
    }
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    closure_bytes = canonical_json_bytes(closure)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "archive_format": ARCHIVE_FORMAT,
        "member_root": MEMBER_ROOT,
        "file_count": len(rows), "files": rows,
        "component_pins": pins,
        "revision_kind": "content-closure-sha1",
        "content_closure_sha1": hashlib.sha1(closure_bytes).hexdigest(),
        "content_closure_sha256": hashlib.sha256(closure_bytes).hexdigest(),
        "exact_member_closure": True,
        "release_scope": "r64-heldout-preservation-only-8x2-exact40-exact81",
        "remote_launch_authorized": False,
        "complete_action_result": False,
        "action_claim_forbidden": True,
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}, payloads


def build_archive(manifest: Mapping[str, Any], payloads: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as bundle:
        for row in manifest["files"]:
            relative = str(row["path"])
            raw = payloads[relative]
            member = tarfile.TarInfo(f"{MEMBER_ROOT}/{relative}")
            member.size = len(raw)
            member.mode = int(row["mode"])
            member.uid = member.gid = member.mtime = 0
            member.uname = member.gname = ""
            member.type = tarfile.REGTYPE
            bundle.addfile(member, io.BytesIO(raw))
    return output.getvalue()


def verify_archive(raw: bytes, manifest: Mapping[str, Any]) -> None:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        fail("release manifest rows differ")
    expected = [f"{MEMBER_ROOT}/{row['path']}" for row in rows]
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as bundle:
            members = bundle.getmembers()
            if [member.name for member in members] != expected:
                fail("release archive exact member order/set differs")
            for member, row in zip(members, rows):
                opened = bundle.extractfile(member)
                payload = b"" if opened is None else opened.read()
                if (
                    not member.isfile() or member.issym() or member.islnk()
                    or member.uid != 0 or member.gid != 0
                    or member.uname != "" or member.gname != ""
                    or member.mtime != 0 or member.mode != row["mode"]
                    or len(payload) != row["size"]
                    or hashlib.sha256(payload).hexdigest() != row["sha256"]
                ):
                    fail(f"release archive member differs: {row['path']}")
    except (tarfile.TarError, OSError) as error:
        raise R64HeldoutReleaseError("cannot verify release archive") from error


def _load_manifest(path: Path, *, expected_sha256: str) -> Mapping[str, Any]:
    if (
        not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path
        or not path.is_file()
    ):
        fail("release manifest must be one canonical plain file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        fail("release manifest bytes differ")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise R64HeldoutReleaseError("cannot decode release manifest") from error
    required = {
        "schema_version", "archive_format", "member_root", "file_count", "files",
        "component_pins", "revision_kind", "content_closure_sha1",
        "content_closure_sha256",
        "exact_member_closure", "release_scope", "remote_launch_authorized",
        "complete_action_result", "action_claim_forbidden", "manifest_digest",
    }
    unsigned = dict(value) if isinstance(value, Mapping) else {}
    declared = unsigned.pop("manifest_digest", None)
    rows = value.get("files") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or raw != canonical_json_bytes(value) + b"\n"
        or set(value) != required
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("archive_format") != ARCHIVE_FORMAT
        or value.get("member_root") != MEMBER_ROOT
        or value.get("file_count") != len(FILES_AND_MODES)
        or not isinstance(rows, list)
        or [row.get("path") if isinstance(row, Mapping) else None for row in rows]
        != sorted(FILES_AND_MODES)
        or value.get("component_pins")
        != {
            label: next(row["sha256"] for row in rows if row["path"] == relative)
            for label, relative in COMPONENT_FILES.items()
        }
        or value.get("revision_kind") != "content-closure-sha1"
        or value.get("exact_member_closure") is not True
        or value.get("release_scope")
        != "r64-heldout-preservation-only-8x2-exact40-exact81"
        or value.get("remote_launch_authorized") is not False
        or value.get("complete_action_result") is not False
        or value.get("action_claim_forbidden") is not True
        or declared != object_sha256(unsigned)
    ):
        fail("release manifest schema/digest differs")
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    closure_bytes = canonical_json_bytes(closure)
    if (
        hashlib.sha1(closure_bytes).hexdigest() != value["content_closure_sha1"]
        or hashlib.sha256(closure_bytes).hexdigest()
        != value["content_closure_sha256"]
    ):
        fail("release content closure differs")
    return value


def validate_executed_release(
    *, method_root: Path, manifest_path: Path, expected_manifest_sha256: str
) -> Mapping[str, Any]:
    """Hash every exact extracted member before any model-facing import."""

    if (
        not method_root.is_absolute() or method_root.is_symlink()
        or method_root.resolve(strict=True) != method_root or not method_root.is_dir()
    ):
        fail("executed method root must be one canonical directory")
    if len(expected_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_manifest_sha256
    ):
        fail("expected release manifest SHA differs")
    manifest = _load_manifest(
        manifest_path, expected_sha256=expected_manifest_sha256
    )
    rows = manifest["files"]
    observed: set[str] = set()
    for path in method_root.rglob("*"):
        relative = path.relative_to(method_root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            fail(f"executed release contains a symlink: {relative}")
        if stat.S_ISREG(mode):
            observed.add(relative)
        elif not stat.S_ISDIR(mode):
            fail(f"executed release contains a special entry: {relative}")
    if observed != set(FILES_AND_MODES):
        fail("executed release exact member set differs")
    for row in rows:
        path = method_root / str(row["path"])
        raw = _stable_plain_bytes(path)
        if (
            set(row) != {"path", "mode", "size", "sha256"}
            or row["mode"] != FILES_AND_MODES[str(row["path"])]
            or stat.S_IMODE(path.stat().st_mode) != row["mode"]
            or len(raw) != row["size"]
            or hashlib.sha256(raw).hexdigest() != row["sha256"]
        ):
            fail(f"executed release member differs: {row['path']}")
    return {
        "manifest_sha256": expected_manifest_sha256,
        "manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "content_closure_sha256": manifest["content_closure_sha256"],
        "component_pins": dict(manifest["component_pins"]),
        "exact_member_closure": True,
        "remote_launch_authorized": False,
    }


def build_release(method_root: Path, output: Path) -> Mapping[str, Any]:
    if (
        not output.is_absolute() or output.exists() or output.is_symlink()
        or not output.parent.is_dir() or output.parent.is_symlink()
        or output.parent.resolve(strict=True) != output.parent
    ):
        fail("release output must be one fresh canonical directory")
    manifest, payloads = build_manifest(method_root)
    archive = build_archive(manifest, payloads)
    verify_archive(archive, manifest)
    output.mkdir(mode=0o700)
    archive_path = output / "source.tar"
    manifest_path = output / "source.manifest.json"
    for path, raw, mode in (
        (archive_path, archive, 0o400),
        (manifest_path, canonical_json_bytes(manifest) + b"\n", 0o400),
    ):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    return {
        "release_root": str(output),
        "archive": str(archive_path),
        "archive_sha256": file_sha256(archive_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "content_closure_sha256": manifest["content_closure_sha256"],
        "component_pins": dict(manifest["component_pins"]),
        "file_count": manifest["file_count"],
        "remote_launch_authorized": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    value = build_release(
        Path(args.method_root).expanduser(), Path(args.output).expanduser()
    )
    print(canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FILES_AND_MODES", "R64HeldoutReleaseError", "build_archive",
    "build_manifest", "build_release", "main", "validate_executed_release",
    "verify_archive",
]
