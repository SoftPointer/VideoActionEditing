#!/usr/bin/env python3
"""Build/audit the exact-member blind-review + Phi authority overlay.

The overlay is intentionally not a generation or training release.  Its only
runtime entrypoints build/ingest an externally authored full160 blind review
and run reviewed block-22 Phi extraction.  Existing materializer dependencies
are not duplicated; their exact bytes are recorded as required base pins and
must be present beside the overlay before either entrypoint is used.
"""

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


SCHEMA_VERSION = "bernini-generic-action-phi-v1-authority-overlay-release-v1"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-exact-modes-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
FILES_AND_MODES: Mapping[str, int] = {
    "tools/generic_action_blind_review_authority_v1.py": 0o444,
    "tools/materialize_phi_v1_authority_controller_v1.py": 0o444,
    "tools/build_generic_action_phi_v1_authority_release_v1.py": 0o444,
    "assets/pair_v5_t2v_calibration_first8_authoring_v1.json": 0o444,
    "assets/mosaic_event_population_compact6_topup20_v1.json": 0o444,
    "assets/action_source_q0_authority_first8_v1.json": 0o444,
}
REQUIRED_BASE_FILES = (
    "tools/generic_action_manifest_v1.py",
    "tools/materialize_phi_v1_sidecars_sp4.py",
    "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py",
    "generic_source_anchored_action_v1.py",
    "temporal_counterfactual_action_scorer_v1.py",
    "materialize_latent_temporal_event_critic_core4.py",
)
ENTRYPOINTS = (
    "tools/generic_action_blind_review_authority_v1.py",
    "tools/materialize_phi_v1_authority_controller_v1.py",
)


class PhiAuthorityReleaseError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise PhiAuthorityReleaseError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PhiAuthorityReleaseError("release is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_plain_bytes(path: Path) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        fail("release input must be absolute and non-symlink")
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if resolved != path or not stat.S_ISREG(before.st_mode):
        fail("release input must be one canonical plain file")
    raw = resolved.read_bytes()
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or len(raw) != before.st_size or not raw:
        fail("release input changed while reading or is empty")
    return raw


def build_manifest(method_root: Path) -> tuple[Mapping[str, Any], Mapping[str, bytes]]:
    root = method_root.resolve(strict=True)
    if root != method_root or root.is_symlink() or not root.is_dir():
        fail("method root must be one canonical directory")
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for relative in sorted(FILES_AND_MODES):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            fail("release member path differs")
        raw = _stable_plain_bytes(root / relative)
        payloads[relative] = raw
        rows.append({"path": relative, "mode": FILES_AND_MODES[relative], "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    base_pins = {relative: hashlib.sha256(_stable_plain_bytes(root / relative)).hexdigest() for relative in REQUIRED_BASE_FILES}
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    unsigned = {
        "schema_version": SCHEMA_VERSION, "release_generation": "r1",
        "archive_format": ARCHIVE_FORMAT, "member_root": MEMBER_ROOT,
        "file_count": len(rows), "files": rows,
        "allowed_entrypoints": list(ENTRYPOINTS),
        "required_base_file_sha256": base_pins,
        "revision_kind": "content-closure-sha1",
        "content_closure_sha1": hashlib.sha1(canonical_json_bytes(closure)).hexdigest(),
        "exact_member_closure": True,
        "authority": {
            "existing_core4_clip_count": 80, "missing_reserve4_clip_count": 80,
            "full_first8_external_review_required": 160,
            "same_runner_self_certification_allowed": False,
            "packet_sealed_before_review_required": True,
            "precommitted_reviewer_tool_source_required": True,
            "precommitted_ed25519_public_key_required": True,
            "signed_execution_credential_required": True,
            "private_reviewer_key_embedded": False,
            "reviewer_authority_template_only_until_external_key_provisioned": True,
            "authority_replay_full81_decode_required": True,
            "base_closure_before_base_import_required": True,
            "phi_block_index": 22, "teacher_exact40_index": 29,
            "fit_operator_coordinate_row_count": 16,
            "core4_only_operator_coordinate_row_count": 8,
            "camera_appearance_coordinate_shape": [21, 32],
            "camera_appearance_weighted_metric_mixing": False,
            "pinned_split80_coordinate_reconstruction_required": True,
            "generated_rgb_or_latent_is_editor_input_or_target": False,
            "training_entrypoint_present": False, "optimizer_authorized": False,
        },
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}, payloads


def validate_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    expected_fields = {"schema_version", "release_generation", "archive_format", "member_root", "file_count", "files", "allowed_entrypoints", "required_base_file_sha256", "revision_kind", "content_closure_sha1", "exact_member_closure", "authority", "manifest_digest"}
    if set(value) != expected_fields:
        fail("release manifest field closure differs")
    unsigned = dict(value); declared = unsigned.pop("manifest_digest")
    rows = value["files"]
    if value["schema_version"] != SCHEMA_VERSION or value["release_generation"] != "r1" or value["archive_format"] != ARCHIVE_FORMAT or value["member_root"] != MEMBER_ROOT or value["file_count"] != len(FILES_AND_MODES) or value["allowed_entrypoints"] != list(ENTRYPOINTS) or value["exact_member_closure"] is not True or declared != object_sha256(unsigned) or [row["path"] for row in rows] != sorted(FILES_AND_MODES):
        fail("release manifest identity differs")
    if set(value["required_base_file_sha256"]) != set(REQUIRED_BASE_FILES) or any(not isinstance(item, str) or len(item) != 64 for item in value["required_base_file_sha256"].values()):
        fail("required base pin closure differs")
    for row in rows:
        if set(row) != {"path", "mode", "size", "sha256"} or row["mode"] != FILES_AND_MODES[row["path"]] or type(row["size"]) is not int or row["size"] <= 0 or type(row["sha256"]) is not str or len(row["sha256"]) != 64:
            fail("release member row differs")
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    if value["revision_kind"] != "content-closure-sha1" or value["content_closure_sha1"] != hashlib.sha1(canonical_json_bytes(closure)).hexdigest():
        fail("release content closure differs")
    expected_authority = {
        "existing_core4_clip_count": 80, "missing_reserve4_clip_count": 80,
        "full_first8_external_review_required": 160,
        "same_runner_self_certification_allowed": False,
        "packet_sealed_before_review_required": True,
        "precommitted_reviewer_tool_source_required": True,
        "precommitted_ed25519_public_key_required": True,
        "signed_execution_credential_required": True,
        "private_reviewer_key_embedded": False,
        "reviewer_authority_template_only_until_external_key_provisioned": True,
        "authority_replay_full81_decode_required": True,
        "base_closure_before_base_import_required": True,
        "phi_block_index": 22, "teacher_exact40_index": 29,
        "fit_operator_coordinate_row_count": 16,
        "core4_only_operator_coordinate_row_count": 8,
        "camera_appearance_coordinate_shape": [21, 32],
        "camera_appearance_weighted_metric_mixing": False,
        "pinned_split80_coordinate_reconstruction_required": True,
        "generated_rgb_or_latent_is_editor_input_or_target": False,
        "training_entrypoint_present": False, "optimizer_authorized": False,
    }
    if value["authority"] != expected_authority:
        fail("release authority differs")
    return value


def build_archive(manifest: Mapping[str, Any], payloads: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for row in manifest["files"]:
            raw = payloads[row["path"]]
            member = tarfile.TarInfo(f"{MEMBER_ROOT}/{row['path']}")
            member.size = len(raw); member.mode = row["mode"]; member.uid = 0; member.gid = 0
            member.uname = ""; member.gname = ""; member.mtime = 0; member.type = tarfile.REGTYPE
            tar.addfile(member, io.BytesIO(raw))
    return output.getvalue()


def verify_archive(raw: bytes, manifest: Mapping[str, Any]) -> None:
    expected = [f"{MEMBER_ROOT}/{row['path']}" for row in manifest["files"]]
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            members = tar.getmembers()
            if [member.name for member in members] != expected:
                fail("archive member order/set differs")
            for member, row in zip(members, manifest["files"]):
                handle = tar.extractfile(member); payload = b"" if handle is None else handle.read()
                if not member.isfile() or member.issym() or member.islnk() or member.uid != 0 or member.gid != 0 or member.uname or member.gname or member.mtime != 0 or member.mode != row["mode"] or member.size != row["size"] or member.pax_headers or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                    fail(f"archive member differs: {member.name}")
    except (OSError, tarfile.TarError) as error:
        raise PhiAuthorityReleaseError(f"cannot verify archive: {error}") from error


def _create(path: Path, raw: bytes, mode: int) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail("release output must be a fresh absolute file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name); handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, mode); os.link(temporary, path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def build(method_root: Path, archive: Path, manifest_path: Path) -> Mapping[str, Any]:
    manifest, payloads = build_manifest(method_root); validate_manifest(manifest)
    archive_raw = build_archive(manifest, payloads); verify_archive(archive_raw, manifest)
    if build_archive(manifest, payloads) != archive_raw: fail("archive rebuild differs")
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _create(archive, archive_raw, 0o444); _create(manifest_path, manifest_raw, 0o444)
    unsigned = {"schema_version": SCHEMA_VERSION, "archive": str(archive), "archive_sha256": hashlib.sha256(archive_raw).hexdigest(), "manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(), "manifest_digest": manifest["manifest_digest"], "file_count": len(FILES_AND_MODES), "optimizer_authorized": False}
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def audit(archive: Path, manifest_path: Path, expected_archive_sha256: str, expected_manifest_sha256: str) -> Mapping[str, Any]:
    archive_raw = _stable_plain_bytes(archive); manifest_raw = _stable_plain_bytes(manifest_path)
    if hashlib.sha256(archive_raw).hexdigest() != expected_archive_sha256 or hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha256: fail("release file SHA-256 differs")
    try: manifest = json.loads(manifest_raw)
    except (UnicodeError, json.JSONDecodeError) as error: raise PhiAuthorityReleaseError("cannot decode manifest") from error
    if manifest_raw != canonical_json_bytes(manifest) + b"\n": fail("manifest bytes are not canonical")
    validate_manifest(manifest); verify_archive(archive_raw, manifest); return manifest


def validate_installed_closure(method_root: Path, manifest_path: Path, expected_manifest_sha256: str) -> Mapping[str, Any]:
    root = method_root.resolve(strict=True)
    if root != method_root or root.is_symlink() or not root.is_dir(): fail("installed method root differs")
    raw = _stable_plain_bytes(manifest_path)
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256: fail("installed release manifest SHA-256 differs")
    try: manifest = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error: raise PhiAuthorityReleaseError("cannot decode installed release manifest") from error
    if raw != canonical_json_bytes(manifest) + b"\n": fail("installed release manifest is not canonical")
    validate_manifest(manifest)
    for row in manifest["files"]:
        observed = hashlib.sha256(_stable_plain_bytes(root / row["path"])).hexdigest()
        if observed != row["sha256"]: fail(f"installed overlay member differs: {row['path']}")
    for relative, expected in manifest["required_base_file_sha256"].items():
        observed = hashlib.sha256(_stable_plain_bytes(root / relative)).hexdigest()
        if observed != expected: fail(f"installed required base member differs: {relative}")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build"); create.add_argument("--method-root", required=True); create.add_argument("--archive", required=True); create.add_argument("--manifest", required=True)
    check = commands.add_parser("audit"); check.add_argument("--archive", required=True); check.add_argument("--manifest", required=True); check.add_argument("--expected-archive-sha256", required=True); check.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "build": value = build(Path(args.method_root), Path(args.archive), Path(args.manifest))
    else: value = audit(Path(args.archive), Path(args.manifest), args.expected_archive_sha256, args.expected_manifest_sha256)
    print(canonical_json_bytes(value).decode("ascii"), flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
