#!/usr/bin/env python3
"""Create/audit the pinned exact-14 preservation-v2 decoded-eval release."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


SCHEMA = "bernini-action-preservation-decoded-eval-source-release-v2"
ENVELOPE_SCHEMA = "bernini-action-preservation-decoded-eval-deployment-v1"
RELEASE_GENERATION = "preservation-v2-decoded-eval-exact14-r1"
MEMBER_ROOT = "methods/bernini_action_editing"
ARCHIVE_FORMAT = "fixed-ustar-ascii-zero-dev-sorted-owner0-mtime0-record10240-v1"
BLOCK_SIZE = 512
RECORD_SIZE = 10240
RUNTIME = "action_preservation_decoded_eval_verified_release_v1.py"

FILES_AND_MODES: Mapping[str, int] = {
    "infer_lora.py": 0o444,
    "train_lora.py": 0o444,
    "self_generated_action_preservation_v2.py": 0o444,
    "action_preservation_gate_v1.py": 0o444,
    "action_preservation_decoded_eval_plan_v1.py": 0o444,
    "action_preservation_decoded_eval_bridge_v1.py": 0o444,
    "action_preservation_decoded_eval_decoder_adapter_v1.py": 0o555,
    "action_preservation_decoded_eval_executor_v1.py": 0o444,
    "action_preservation_decoded_eval_launcher_v1.py": 0o444,
    "action_preservation_decoded_eval_aggregate_v1.py": 0o444,
    "action_preservation_loop_controller_v1.py": 0o444,
    "tools/materialize_vae.py": 0o444,
    "tools/build_renderer_dataset.py": 0o444,
    RUNTIME: 0o444,
}
MEMBER_ORDER = tuple(FILES_AND_MODES)
EXPECTED_SHA256: Mapping[str, str] = {
    "infer_lora.py": "3dd890e60d4427fefd8a9619fc8b918210b88670737a470fd53dae5538e5cead",
    "train_lora.py": "eae8eaac25197112637f466e611ba7eae574266d4cd1b83e625195fb22b0476e",
    "self_generated_action_preservation_v2.py": "11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c",
    "action_preservation_gate_v1.py": "9c8aad2a0f96d021e3bd9164952462faf105998839b5835d1e8dca859ce6e24f",
    "action_preservation_decoded_eval_plan_v1.py": "287efb71142c91bd0ad78354f6f72948a7aebc5b746c96fb32f701aa7158072b",
    "action_preservation_decoded_eval_bridge_v1.py": "a545e62d0528ec6d425caf92709a5158c8a6f95e2dff5cb4699dfc70770d4cd0",
    "action_preservation_decoded_eval_decoder_adapter_v1.py": "d73d1f0a5aae8dbf9359a552bd24f9806250a14a5460031eec5883f3f20fdd45",
    "action_preservation_decoded_eval_executor_v1.py": "e9f55f039c0f27da677c1211b7b9d368a7289b2e5f178f3af1e1f2ab6f82c718",
    "action_preservation_decoded_eval_launcher_v1.py": "21b294236a418e82c54896a670f95957a1f5c83b9c1652c7521e860cd1d1fd34",
    "action_preservation_decoded_eval_aggregate_v1.py": "7af794411789b4d6defb354169cdc82525af7e8ffddcf5bc22213f76fb8c7c4c",
    "action_preservation_loop_controller_v1.py": "b070cd82c11251b9b638ff1f39a3c346e8347a0137b8b1e17f8aa2a67661db6c",
    "tools/materialize_vae.py": "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
    "tools/build_renderer_dataset.py": "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
    RUNTIME: "4fb1b7bf474951cfe4d8fa9a083ab0b1caf17e5b981fa73faa156547945812ac",
}
EXPECTED_SIZE: Mapping[str, int] = {
    "infer_lora.py": 88623,
    "train_lora.py": 84216,
    "self_generated_action_preservation_v2.py": 11334,
    "action_preservation_gate_v1.py": 75465,
    "action_preservation_decoded_eval_plan_v1.py": 49347,
    "action_preservation_decoded_eval_bridge_v1.py": 88640,
    "action_preservation_decoded_eval_decoder_adapter_v1.py": 28832,
    "action_preservation_decoded_eval_executor_v1.py": 64880,
    "action_preservation_decoded_eval_launcher_v1.py": 27965,
    "action_preservation_decoded_eval_aggregate_v1.py": 38127,
    "action_preservation_loop_controller_v1.py": 49068,
    "tools/materialize_vae.py": 32195,
    "tools/build_renderer_dataset.py": 31012,
    RUNTIME: 110765,
}
ALLOWED_ENTRYPOINTS = tuple(sorted({
    "infer_lora.py", "action_preservation_gate_v1.py",
    "action_preservation_decoded_eval_plan_v1.py",
    "action_preservation_decoded_eval_bridge_v1.py",
    "action_preservation_decoded_eval_decoder_adapter_v1.py",
    "action_preservation_decoded_eval_executor_v1.py",
    "action_preservation_decoded_eval_launcher_v1.py",
    "action_preservation_decoded_eval_aggregate_v1.py",
    "action_preservation_loop_controller_v1.py", "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py",
}))
AUTHORITY = {
    "evaluation_kind": "preservation-v2-full-video-decoded-exact264",
    "candidate_count": 264, "full_video_frame_count": 81,
    "fps_num": 25, "fps_den": 1,
    "source_identity_background_camera_are_conjunctive": True,
    "training_loss_is_not_evaluation_evidence": True,
    "missing_calibration_requires_abstain": True,
    "distinct_blind_reviewers_required": 2,
    "automatic_scientific_promotion_authorized": False,
}


class EvalReleaseBuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvalReleaseBuildError(message)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise EvalReleaseBuildError("value is not canonical JSON") from error


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def object_sha256(value: Any) -> str:
    return sha256(canonical(value))


def content_revision(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha1(canonical(list(rows))).hexdigest()


def _identity(value: os.stat_result) -> Tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid, value.st_mode,
        value.st_nlink, value.st_rdev, value.st_size,
        getattr(value, "st_blocks", 0), value.st_mtime_ns, value.st_ctime_ns,
    )


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def stable_capture(
    path: Path, *, label: str, expected_sha256: str | None = None,
    expected_mode: int | None = None,
) -> Tuple[bytes, os.stat_result]:
    require(path.is_absolute() and not path.is_symlink(), f"{label} path differs")
    try:
        require(path.resolve(strict=True) == path, f"{label} is not canonical")
    except OSError as error:
        raise EvalReleaseBuildError(f"{label} is unavailable") from error
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    require(
        stat.S_ISREG(before.st_mode) and before.st_nlink == 1
        and _identity(before) == _identity(middle) == _identity(after) == _identity(named)
        and first == second and len(first) == before.st_size,
        f"{label} physical identity changed or differs",
    )
    require(
        expected_mode is None or stat.S_IMODE(before.st_mode) == expected_mode,
        f"{label} mode differs",
    )
    require(
        expected_sha256 is None or sha256(first) == expected_sha256,
        f"{label} SHA differs",
    )
    return first, before


def _unique_pairs(pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def load_canonical_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw, object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise EvalReleaseBuildError(f"{label} is not strict JSON") from error
    require(type(value) is dict and canonical(value) + b"\n" == raw, f"{label} is not canonical")
    return value


def method_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[1]


def expected_rows() -> list[dict[str, Any]]:
    require(
        tuple(FILES_AND_MODES) == tuple(EXPECTED_SHA256) == tuple(EXPECTED_SIZE),
        "builder pin closure/order differs",
    )
    return [
        {
            "path": relative, "mode": FILES_AND_MODES[relative],
            "size": EXPECTED_SIZE[relative], "sha256": EXPECTED_SHA256[relative],
        }
        for relative in MEMBER_ORDER
    ]


def payload_rows(root: Path) -> Tuple[list[dict[str, Any]], Dict[str, bytes]]:
    rows = expected_rows()
    payloads: Dict[str, bytes] = {}
    for row in rows:
        relative = row["path"]
        raw, _ = stable_capture(
            root / relative, label=f"workspace member {relative}",
            expected_sha256=row["sha256"],
        )
        require(len(raw) == row["size"], f"workspace member size differs: {relative}")
        payloads[relative] = raw
    return rows, payloads


def make_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(list(rows) == expected_rows(), "manifest cannot self-sign workspace drift")
    value: dict[str, Any] = {
        "schema_version": SCHEMA, "release_generation": RELEASE_GENERATION,
        "archive_format": ARCHIVE_FORMAT, "member_root": MEMBER_ROOT,
        "exact_member_closure": True, "file_count": len(MEMBER_ORDER),
        "files": list(rows), "content_revision": content_revision(rows),
        "allowed_entrypoints": list(ALLOWED_ENTRYPOINTS), "authority": AUTHORITY,
        "component_sha256": {row["path"]: row["sha256"] for row in rows},
    }
    value["manifest_digest"] = object_sha256(value)
    return value


def validate_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = make_manifest(expected_rows())
    require(value == expected, "manifest exact field/value closure differs")
    return value


def _ustar_text(value: str, width: int, label: str) -> bytes:
    require(type(value) is str and "\0" not in value, f"{label} differs")
    try:
        raw = value.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise EvalReleaseBuildError(f"{label} is not USTAR ASCII") from error
    require(len(raw) <= width, f"{label} exceeds USTAR width")
    return raw + b"\0" * (width - len(raw))


def _ustar_octal(value: int, width: int, label: str) -> bytes:
    require(type(value) is int and 0 <= value < 8 ** (width - 1), f"{label} differs")
    return f"{value:0{width - 1}o}".encode("ascii") + b"\0"


def _ustar_name_fields(value: str) -> Tuple[bytes, bytes]:
    try:
        encoded = value.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise EvalReleaseBuildError("USTAR name is not ASCII") from error
    if len(encoded) <= 100:
        return _ustar_text(value, 100, "USTAR name"), b"\0" * 155
    for index in range(len(value) - 1, -1, -1):
        if value[index] != "/":
            continue
        prefix, basename = value[:index], value[index + 1:]
        if (
            prefix and basename and len(prefix.encode("ascii")) <= 155
            and len(basename.encode("ascii")) <= 100
        ):
            return (
                _ustar_text(basename, 100, "USTAR name"),
                _ustar_text(prefix, 155, "USTAR prefix"),
            )
    raise EvalReleaseBuildError("USTAR name cannot be represented")


def fixed_ustar_header(name: str, *, size: int, mode: int) -> bytes:
    name_field, prefix_field = _ustar_name_fields(name)
    header = bytearray(BLOCK_SIZE)
    header[0:100] = name_field
    header[100:108] = _ustar_octal(mode, 8, "USTAR mode")
    header[108:116] = _ustar_octal(0, 8, "USTAR uid")
    header[116:124] = _ustar_octal(0, 8, "USTAR gid")
    header[124:136] = _ustar_octal(size, 12, "USTAR size")
    header[136:148] = _ustar_octal(0, 12, "USTAR mtime")
    header[148:156] = b" " * 8
    header[156:157] = b"0"
    header[157:257] = b"\0" * 100
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    header[265:329] = b"\0" * 64
    header[329:337] = _ustar_octal(0, 8, "USTAR devmajor")
    header[337:345] = _ustar_octal(0, 8, "USTAR devminor")
    header[345:500] = prefix_field
    header[500:512] = b"\0" * 12
    checksum = sum(header)
    require(checksum < 8 ** 6, "USTAR checksum differs")
    header[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    return bytes(header)


def make_archive(payloads: Mapping[str, bytes]) -> bytes:
    require(set(payloads) == set(MEMBER_ORDER), "archive payload closure differs")
    output = bytearray()
    for relative in MEMBER_ORDER:
        raw = payloads[relative]
        require(
            len(raw) == EXPECTED_SIZE[relative] and sha256(raw) == EXPECTED_SHA256[relative],
            f"archive payload differs: {relative}",
        )
        output.extend(fixed_ustar_header(
            f"{MEMBER_ROOT}/{relative}", size=len(raw), mode=FILES_AND_MODES[relative]
        ))
        output.extend(raw)
        output.extend(b"\0" * (-len(raw) % BLOCK_SIZE))
    output.extend(b"\0" * (2 * BLOCK_SIZE))
    output.extend(b"\0" * (-len(output) % RECORD_SIZE))
    require(len(output) % RECORD_SIZE == 0, "archive record boundary differs")
    return bytes(output)


def verify_archive(raw: bytes, manifest: Mapping[str, Any]) -> Dict[str, bytes]:
    rows = manifest["files"]
    expected_names = [f"{MEMBER_ROOT}/{relative}" for relative in MEMBER_ORDER]
    payloads: Dict[str, bytes] = {}
    expected_offset = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
            require([member.name for member in members] == expected_names, "archive closure differs")
            for member, row in zip(members, rows):
                require(
                    member.type == tarfile.REGTYPE and member.isfile()
                    and not member.issym() and not member.islnk() and not member.linkname
                    and not member.pax_headers and member.mode == row["mode"]
                    and member.uid == member.gid == member.mtime == 0
                    and member.uname == member.gname == ""
                    and member.size == row["size"] and member.offset == expected_offset
                    and member.offset_data == expected_offset + BLOCK_SIZE,
                    f"archive metadata differs: {member.name}",
                )
                require(
                    raw[member.offset:member.offset + BLOCK_SIZE]
                    == fixed_ustar_header(member.name, size=row["size"], mode=row["mode"]),
                    f"archive header differs: {member.name}",
                )
                stream = archive.extractfile(member)
                payload = b"" if stream is None else stream.read()
                require(
                    len(payload) == row["size"] and sha256(payload) == row["sha256"],
                    f"archive payload differs: {member.name}",
                )
                payloads[row["path"]] = payload
                expected_offset = member.offset_data + (
                    (member.size + BLOCK_SIZE - 1) // BLOCK_SIZE
                ) * BLOCK_SIZE
    except (OSError, tarfile.TarError) as error:
        raise EvalReleaseBuildError("archive is not strict USTAR") from error
    require(
        len(raw) % RECORD_SIZE == 0 and len(raw[expected_offset:]) >= 2 * BLOCK_SIZE
        and not any(raw[expected_offset:]) and raw == make_archive(payloads),
        "archive canonical trailer/bytes differ",
    )
    return payloads


def make_envelope(
    archive_raw: bytes, manifest_raw: bytes, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    validate_manifest(manifest)
    require(canonical(manifest) + b"\n" == manifest_raw, "manifest bytes differ")
    value: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA,
        "release_generation": RELEASE_GENERATION,
        "remote_release_exact_entries": [
            "deployment-envelope.json", "source.manifest.json", "source.tar"
        ],
        "source_archive": {
            "basename": "source.tar", "sha256": sha256(archive_raw), "mode": 0o444,
        },
        "source_manifest": {
            "basename": "source.manifest.json", "sha256": sha256(manifest_raw),
            "manifest_digest": manifest["manifest_digest"],
            "content_revision": manifest["content_revision"],
            "file_count": len(MEMBER_ORDER), "mode": 0o444,
        },
        "create_only_deployment_required": True,
        "fresh_materialized_root_required": True,
        "verified_runtime_required": True,
        "detached_controller_authority_receipt_required": True,
        "automatic_scientific_promotion_authorized": False,
    }
    value["envelope_digest"] = object_sha256(value)
    return value


def validate_envelope(
    value: Mapping[str, Any], *, archive_raw: bytes, manifest_raw: bytes,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    require(
        value == make_envelope(archive_raw, manifest_raw, manifest),
        "envelope exact field/value closure differs",
    )
    return value


def _directory_flags() -> int:
    require(hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW"), "safe directory flags absent")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _write_at(parent_fd: int, name: str, raw: bytes, mode: int) -> None:
    descriptor = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0), mode, dir_fd=parent_fd,
    )
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            require(count > 0, f"write made no progress: {name}")
            offset += count
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        details = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        require(
            stat.S_ISREG(details.st_mode) and details.st_nlink == 1
            and _identity(details) == _identity(named),
            f"created release identity differs: {name}",
        )
    finally:
        os.close(descriptor)
    os.fsync(parent_fd)


def _fresh_release_parent(release_dir: Path) -> Tuple[Path, Path]:
    require(
        release_dir.is_absolute() and not release_dir.exists()
        and not release_dir.is_symlink() and release_dir.name not in ("", ".", ".."),
        "release directory must be fresh absolute path",
    )
    parent = release_dir.parent.resolve(strict=True)
    require(
        parent == release_dir.parent and not release_dir.parent.is_symlink(),
        "release parent is not canonical",
    )
    return release_dir, parent


def build(release_dir: Path) -> dict[str, Any]:
    root = method_root()
    rows, payloads = payload_rows(root)
    manifest = make_manifest(rows)
    manifest_raw = canonical(manifest) + b"\n"
    archive_raw = make_archive(payloads)
    envelope = make_envelope(archive_raw, manifest_raw, manifest)
    envelope_raw = canonical(envelope) + b"\n"
    destination, parent = _fresh_release_parent(release_dir)
    parent_fd = os.open(parent, _directory_flags())
    release_fd = -1
    try:
        os.mkdir(destination.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        release_fd = os.open(destination.name, _directory_flags(), dir_fd=parent_fd)
        _write_at(release_fd, "source.tar", archive_raw, 0o444)
        _write_at(release_fd, "source.manifest.json", manifest_raw, 0o444)
        _write_at(release_fd, "deployment-envelope.json", envelope_raw, 0o444)
        os.fchmod(release_fd, 0o555)
        os.fsync(release_fd)
        os.fsync(parent_fd)
    except FileExistsError as error:
        raise EvalReleaseBuildError("create-only release build collided") from error
    finally:
        if release_fd >= 0:
            os.close(release_fd)
        os.close(parent_fd)
    return audit(destination, against_workspace=True)


def audit(release_dir: Path, *, against_workspace: bool) -> dict[str, Any]:
    require(
        release_dir.is_absolute() and not release_dir.is_symlink()
        and release_dir.resolve(strict=True) == release_dir,
        "release directory path differs",
    )
    details = release_dir.lstat()
    require(
        stat.S_ISDIR(details.st_mode) and stat.S_IMODE(details.st_mode) == 0o555,
        "release directory is not sealed 0555",
    )
    expected_names = {
        "source.tar", "source.manifest.json", "deployment-envelope.json"
    }
    require(
        {entry.name for entry in os.scandir(release_dir)} == expected_names,
        "release exact entry closure differs",
    )
    archive_raw, _ = stable_capture(
        release_dir / "source.tar", label="release archive", expected_mode=0o444
    )
    manifest_raw, _ = stable_capture(
        release_dir / "source.manifest.json", label="release manifest", expected_mode=0o444
    )
    envelope_raw, _ = stable_capture(
        release_dir / "deployment-envelope.json", label="deployment envelope", expected_mode=0o444
    )
    manifest = load_canonical_json(manifest_raw, label="release manifest")
    validate_manifest(manifest)
    payloads = verify_archive(archive_raw, manifest)
    envelope = load_canonical_json(envelope_raw, label="deployment envelope")
    validate_envelope(
        envelope, archive_raw=archive_raw, manifest_raw=manifest_raw, manifest=manifest
    )
    if against_workspace:
        workspace_rows, workspace_payloads = payload_rows(method_root())
        require(
            workspace_rows == manifest["files"] and workspace_payloads == payloads,
            "workspace differs from independently pinned release",
        )
    return {
        "static_audit_go": True, "release_dir": str(release_dir),
        "archive_sha256": sha256(archive_raw),
        "manifest_sha256": sha256(manifest_raw),
        "manifest_digest": manifest["manifest_digest"],
        "content_revision": manifest["content_revision"],
        "envelope_sha256": sha256(envelope_raw),
        "envelope_digest": envelope["envelope_digest"],
        "file_count": len(MEMBER_ORDER), "exact_member_closure": True,
        "release_directory_mode": "0555",
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--release-dir", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--release-dir", required=True)
    audit_parser.add_argument("--against-workspace", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    release_dir = Path(args.release_dir).resolve(strict=False)
    result = (
        build(release_dir) if args.command == "build"
        else audit(release_dir, against_workspace=args.against_workspace)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
