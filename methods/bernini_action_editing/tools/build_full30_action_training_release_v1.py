#!/usr/bin/env python3
"""Build or audit the deterministic full-30 action-training source release.

The executed trainer validates a deliberately small manifest schema.  This
builder emits that exact schema, derives ``release_sha256`` from the canonical
ordered file closure, and stores the same bytes in a deterministic USTAR
archive.  It never reads model weights, data, review authority, or optimizer
state and it does not authorize a launch.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-full30-action-training-release-v1"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-mode0444-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
RELEASE_FILES = (
    "clean_source_visual_context_training_v1.py",
    "dclr_runtime_contract.py",
    "dual_conditional_ratio_core.py",
    "full30_action_amplitude_authority_v1.py",
    "full30_action_checkpoint_v1.py",
    "full30_action_data_teacher_authority_v1.py",
    "full30_action_learning_v1.py",
    "full30_action_mechanism_canary_authority_v1.py",
    "full30_action_optimizer_v1.py",
    "full30_action_psiout_materializer_v1.py",
    "full30_action_runtime_v1.py",
    "full30_action_training_step_v1.py",
    "graft_phase_a_native_training_closure_v1.py",
    "infer_dclr_reward_runtime_smoke.py",
    "infer_lora.py",
    "infer_source_kv_carrier_oracle.py",
    "inference_sigma_strata.py",
    "motion_residual.py",
    "packed_preservation_lora_v2.py",
    "packed_preservation_release_v2.py",
    "source_kv_replay.py",
    "source_kv_route_batches.py",
    "source_self_runtime.py",
    "train_full30_action_lora_v1.py",
    "train_lora.py",
    "train_packed_preservation_lora_v2.py",
)
RELEASE_FILE_SET = frozenset(RELEASE_FILES)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Full30ActionReleaseError(RuntimeError):
    """Raised before an ambiguous or mutable source release is accepted."""


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Full30ActionReleaseError(message)


def _sha256(value: Any, *, label: str) -> str:
    _require(type(value) is str and _SHA256.fullmatch(value) is not None, f"{label} differs")
    return value


def _plain_directory(path: Path, *, label: str) -> Path:
    _require(path.is_absolute() and not path.is_symlink(), f"{label} must be an absolute non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as error:
        raise Full30ActionReleaseError(f"cannot resolve {label}: {error}") from error
    _require(resolved == path and stat.S_ISDIR(metadata.st_mode), f"{label} must be canonical")
    return resolved


def _plain_file(path: Path, *, label: str) -> Path:
    _require(path.is_absolute() and not path.is_symlink(), f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as error:
        raise Full30ActionReleaseError(f"cannot resolve {label}: {error}") from error
    _require(
        resolved == path and stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
        f"{label} must be one canonical plain file",
    )
    return resolved


def _read_stable(path: Path, *, label: str, maximum_bytes: int = 32 * 1024 * 1024) -> bytes:
    source = _plain_file(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(source, flags)
        before = os.fstat(descriptor)
        _require(before.st_size <= maximum_bytes, f"{label} exceeds the byte limit")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            _require(bool(block), f"{label} was truncated")
            chunks.append(block)
            remaining -= len(block)
        _require(os.read(descriptor, 1) == b"", f"{label} grew while reading")
        after = os.fstat(descriptor)
    except OSError as error:
        raise Full30ActionReleaseError(f"cannot read {label}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    named = source.lstat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
        item.st_nlink,
    )
    _require(identity(before) == identity(after) == identity(named), f"{label} changed while reading")
    return b"".join(chunks)


def _decode_canonical_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            _require(key not in result, f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30ActionReleaseError(f"cannot decode {label}") from error
    _require(type(value) is dict and raw == canonical_json_bytes(value) + b"\n", f"{label} is not canonical")
    return value


def _write_create_only(path: Path, payload: bytes) -> None:
    _require(path.is_absolute() and not path.exists() and not path.is_symlink(), "output path must be fresh and absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    _plain_directory(path.parent, label="output parent")
    descriptor: Optional[int] = None
    temporary: Optional[Path] = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        _require(_read_stable(path, label=str(path), maximum_bytes=max(len(payload), 1)) == payload, "published output differs")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = 0o444
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    member.type = tarfile.REGTYPE
    return member


def _release_payload(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "files": list(rows)}


def _validate_trainer_release_contract(raw: bytes) -> None:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename="train_full30_action_lora_v1.py")
    except (SyntaxError, UnicodeError) as error:
        raise Full30ActionReleaseError("trainer source cannot be parsed") from error
    required: Optional[frozenset[str]] = None
    schema: Optional[str] = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {
            target.id for target in node.targets if isinstance(target, ast.Name)
        }
        if "REQUIRED_RELEASE_FILES" in names:
            try:
                _require(
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "frozenset"
                    and len(node.value.args) == 1,
                    "trainer required-release declaration differs",
                )
                required = frozenset(ast.literal_eval(node.value.args[0]))
            except (ValueError, TypeError) as error:
                raise Full30ActionReleaseError(
                    "trainer required-release declaration cannot be decoded"
                ) from error
        if "RELEASE_SCHEMA_VERSION" in names:
            try:
                schema = ast.literal_eval(node.value)
            except (ValueError, TypeError) as error:
                raise Full30ActionReleaseError(
                    "trainer release schema declaration cannot be decoded"
                ) from error
    _require(required == RELEASE_FILE_SET, "builder/trainer required-release closure differs")
    _require(schema == SCHEMA_VERSION, "builder/trainer release schema differs")


def build_manifest(method_root: Path) -> tuple[dict[str, Any], Mapping[str, bytes]]:
    root = _plain_directory(method_root, label="method root")
    _require(tuple(sorted(RELEASE_FILES)) == RELEASE_FILES, "release files must be canonical UTF-8 order")
    payloads: dict[str, bytes] = {}
    rows: list[dict[str, str]] = []
    for relative in RELEASE_FILES:
        parts = PurePosixPath(relative)
        _require(not parts.is_absolute() and ".." not in parts.parts and len(parts.parts) == 1, "release member path differs")
        raw = _read_stable(root / relative, label=f"release member {relative}")
        payloads[relative] = raw
        rows.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
    _validate_trainer_release_contract(payloads["train_full30_action_lora_v1.py"])
    release_sha = object_sha256(_release_payload(rows))
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "exact_member_closure": True,
        "files": rows,
        "release_sha256": release_sha,
    }
    manifest = {**unsigned, "manifest_digest": object_sha256(unsigned)}
    return manifest, MappingProxyType(payloads)


def validate_manifest(value: Any) -> dict[str, Any]:
    _require(type(value) is dict, "release manifest must be an object")
    _require(
        set(value) == {"schema_version", "exact_member_closure", "files", "release_sha256", "manifest_digest"},
        "release manifest field closure differs",
    )
    unsigned = dict(value)
    declared = unsigned.pop("manifest_digest")
    rows = value.get("files")
    _require(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("exact_member_closure") is True
        and type(rows) is list
        and declared == object_sha256(unsigned),
        "release manifest seal differs",
    )
    observed: list[dict[str, str]] = []
    for item in rows:
        _require(type(item) is dict and set(item) == {"path", "sha256"}, "release row field closure differs")
        relative = item["path"]
        _require(type(relative) is str and relative not in {row["path"] for row in observed}, "release row path differs")
        observed.append({"path": relative, "sha256": _sha256(item["sha256"], label=f"release member {relative} SHA")})
    _require(tuple(row["path"] for row in observed) == RELEASE_FILES, "release member closure/order differs")
    _require(value.get("release_sha256") == object_sha256(_release_payload(observed)), "release SHA differs")
    return value


def build_archive(manifest: Mapping[str, Any], payloads: Mapping[str, bytes]) -> bytes:
    validated = validate_manifest(manifest)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for row in validated["files"]:
            relative = row["path"]
            raw = payloads[relative]
            _require(hashlib.sha256(raw).hexdigest() == row["sha256"], f"payload differs: {relative}")
            archive.addfile(_tar_info(f"{MEMBER_ROOT}/{relative}", len(raw)), io.BytesIO(raw))
    return buffer.getvalue()


def validate_archive_bytes(raw: bytes, manifest: Mapping[str, Any]) -> None:
    validated = validate_manifest(manifest)
    expected = [f"{MEMBER_ROOT}/{row['path']}" for row in validated["files"]]
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
            _require([member.name for member in members] == expected, "archive member closure/order differs")
            trainer_raw: Optional[bytes] = None
            for member, row in zip(members, validated["files"]):
                handle = archive.extractfile(member)
                payload = handle.read() if handle is not None else b""
                _require(
                    member.isfile()
                    and not member.issym()
                    and not member.islnk()
                    and member.uid == 0
                    and member.gid == 0
                    and member.mtime == 0
                    and stat.S_IMODE(member.mode) == 0o444
                    and handle is not None
                    and hashlib.sha256(payload).hexdigest() == row["sha256"],
                    f"archive member differs: {member.name}",
                )
                if row["path"] == "train_full30_action_lora_v1.py":
                    trainer_raw = payload
            _require(trainer_raw is not None, "archive omits trainer source")
            _validate_trainer_release_contract(trainer_raw)
    except (OSError, tarfile.TarError) as error:
        raise Full30ActionReleaseError(f"cannot validate archive: {error}") from error


def build(method_root: Path, archive_path: Path, manifest_path: Path) -> Mapping[str, Any]:
    manifest, payloads = build_manifest(method_root)
    archive_raw = build_archive(manifest, payloads)
    validate_archive_bytes(archive_raw, manifest)
    _require(build_archive(manifest, payloads) == archive_raw, "archive rebuild is not byte-identical")
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(archive_path, archive_raw)
    _write_create_only(manifest_path, manifest_raw)
    return MappingProxyType(
        {
            "schema_version": SCHEMA_VERSION,
            "archive_format": ARCHIVE_FORMAT,
            "archive": str(archive_path),
            "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
            "manifest": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "manifest_digest": manifest["manifest_digest"],
            "release_sha256": manifest["release_sha256"],
            "file_count": len(RELEASE_FILES),
            "exact_member_closure": True,
            "launch_authorized": False,
        }
    )


def audit(archive_path: Path, manifest_path: Path, *, expected_archive_sha256: str, expected_manifest_sha256: str) -> Mapping[str, Any]:
    archive_raw = _read_stable(archive_path, label="release archive", maximum_bytes=256 * 1024 * 1024)
    manifest_raw = _read_stable(manifest_path, label="release manifest")
    _require(hashlib.sha256(archive_raw).hexdigest() == _sha256(expected_archive_sha256, label="archive SHA"), "archive SHA differs")
    _require(hashlib.sha256(manifest_raw).hexdigest() == _sha256(expected_manifest_sha256, label="manifest SHA"), "manifest SHA differs")
    manifest = validate_manifest(_decode_canonical_json(manifest_raw, label="release manifest"))
    validate_archive_bytes(archive_raw, manifest)
    return MappingProxyType(
        {
            "schema_version": SCHEMA_VERSION,
            "archive_sha256": expected_archive_sha256,
            "manifest_sha256": expected_manifest_sha256,
            "manifest_digest": manifest["manifest_digest"],
            "release_sha256": manifest["release_sha256"],
            "file_count": len(RELEASE_FILES),
            "exact_member_closure": True,
            "audit_passed": True,
            "launch_authorized": False,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser_ = subparsers.add_parser("build")
    build_parser_.add_argument("--method-root", required=True)
    build_parser_.add_argument("--archive", required=True)
    build_parser_.add_argument("--manifest", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--archive", required=True)
    audit_parser.add_argument("--manifest", required=True)
    audit_parser.add_argument("--expected-archive-sha256", required=True)
    audit_parser.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        result = build(Path(args.method_root), Path(args.archive), Path(args.manifest))
    else:
        result = audit(
            Path(args.archive),
            Path(args.manifest),
            expected_archive_sha256=args.expected_archive_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    print(canonical_json_bytes(dict(result)).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
