#!/usr/bin/env python3
"""Build and audit the deterministic BOX-EXP-014 source7 release."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full30_action_source7_reencode_plan_v1 as plan_contract  # noqa: E402


SCHEMA_VERSION = "bernini-full30-action-source7-reencode-release-v1"
RELEASE_GENERATION = "r4"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-exact-modes-zero-devfields-v2"
MEMBER_ROOT = "methods/bernini_action_editing"
FILES_AND_MODES: Mapping[str, int] = {
    "full30_action_source7_reencode_plan_v1.py": 0o444,
    "full30_action_source7_reencode_controller_v1.py": 0o444,
    "source_self_role_repaint.py": 0o444,
    "tools/materialize_full30_action_source7_reencode_v1.py": 0o444,
    "tools/materialize_source_self_role_repaint.py": 0o444,
    "tools/materialize_ramp_motion_analogy_vae.py": 0o444,
    "tools/materialize_vae.py": 0o444,
    "tools/build_renderer_dataset.py": 0o444,
    "tools/full30_action_source7_reencode_runtime_cache_v1.py": 0o444,
    "tools/build_full30_action_source7_reencode_release_v1.py": 0o444,
    "scripts/auh_full30_action_source7_reencode_136141_v1.sh": 0o555,
}
COMPONENT_FILES: Mapping[str, str] = {
    "plan_sha256": "full30_action_source7_reencode_plan_v1.py",
    "controller_sha256": "full30_action_source7_reencode_controller_v1.py",
    "materializer_sha256": "tools/materialize_full30_action_source7_reencode_v1.py",
    "official_source_self_materializer_sha256": "tools/materialize_source_self_role_repaint.py",
    "pinned_wan_encoder_sha256": "tools/materialize_ramp_motion_analogy_vae.py",
    "official_vae_io_sha256": "tools/materialize_vae.py",
    "release_builder_sha256": "tools/build_full30_action_source7_reencode_release_v1.py",
    "runtime_cache_preflight_sha256": "tools/full30_action_source7_reencode_runtime_cache_v1.py",
    "launcher_sha256": "scripts/auh_full30_action_source7_reencode_136141_v1.sh",
}
ENTRYPOINTS = (
    "full30_action_source7_reencode_controller_v1.py",
    "scripts/auh_full30_action_source7_reencode_136141_v1.sh",
)


class Source7ReencodeReleaseError(RuntimeError):
    """Raised before a mutable or incomplete release can pass."""


def fail(message: str) -> NoReturn:
    raise Source7ReencodeReleaseError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    return plan_contract.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return plan_contract.object_sha256(value)


def _stable_plain_bytes(path: Path) -> bytes:
    require(path.is_absolute() and not path.is_symlink(), "release input must be absolute and non-symlink")
    try:
        resolved = path.resolve(strict=True)
        before = path.stat()
    except OSError as error:
        raise Source7ReencodeReleaseError("release input is unavailable") from error
    require(resolved == path and stat.S_ISREG(before.st_mode), "release input must be a canonical plain file")
    raw = path.read_bytes()
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    require(before_identity == after_identity and len(raw) == before.st_size and bool(raw), "release input changed or is empty")
    return raw


def _authority() -> Mapping[str, Any]:
    plan = plan_contract.validate_plan(plan_contract.canonical_plan())
    return {
        "experiment_id": plan_contract.EXPERIMENT_ID,
        "purpose": plan["purpose"],
        "scientific_target": plan["scientific_target"],
        "learning_target": plan["learning_target"],
        "numeric_target": plan["numeric_target"],
        "dataset": plan["dataset"],
        "steps": plan["steps"],
        "baseline": plan["baseline"],
        "core_validation": plan["core_validation"],
        "plan_digest": plan["plan_digest"],
        "exact7_iids": [row["iid"] for row in plan["rows"]],
        "external_existing_index0_iid": "2d2e28871a5a4856",
        "source_only_reencode_from_source_video": True,
        "vae_encode_calls_per_source": 1,
        "paired_dataset_accessed": False,
        "legacy_source_target_container_opened": False,
        "synthetic_target_index1_path_read": False,
        "synthetic_target_index1_bytes_read": False,
        "synthetic_target_index1_decoded": False,
        "synthetic_target_index1_filtered_on": False,
        "synthetic_target_index1_hashed": False,
        "target_video_path_present": False,
        "target_video_accessed": False,
        "external_existing_index0_reencoded": False,
        "inventory_snapshot_only": True,
        "exact8_authority_go_claimed": False,
        "teacher_cross_disjointness_pending": True,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "training_authorized": False,
        "miopen_runtime_cache_preflight_required": True,
        "sqlite_commit_reopen_probe_required": True,
        "cuda_miopen_wan_resample_three_geometry_smoke_before_source_open_required": True,
        "cuda_miopen_smoke_exact_field_and_digest_closure_required": True,
        "miopen_custom_cache_fresh_gfx90a68_ukdb_required": True,
        "miopen_custom_cache_kern_db_nonempty_required": True,
        "miopen_custom_cache_wal_absent_or_empty_required": True,
        "miopen_custom_cache_sqlite_immutable_readonly_validation_required": True,
        "miopen_custom_cache_validation_inventory_exact_stability_required": True,
        "miopen_user_db_path_write_required": False,
        "miopen_user_db_allowed_plaintext_main_basenames": [
            "gfx90a68.HIP.3_3_0_a85ca8a54-dirty.udb.txt",
            "gfx90a68.HIP.3_3_0_a85ca8a54-dirty.ufdb.txt",
        ],
        "miopen_user_db_plaintext_main_mode_0777_required_if_present": True,
        "miopen_user_db_optional_time_sidecar_modes_recorded_not_pinned": True,
        "tmpdir_export_before_runtime_prepare_required": True,
        "tmpdir_scoped_under_fresh_runtime_cache_required": True,
        "cpp_temp_directory_path_scoped_lock_activity_required": True,
        "miopen_lock_basename_prefix": "md5(canonical-absolute-MIOPEN_USER_DB_PATH)_",
        "miopen_lock_basenames_path_hash_bound": True,
        "scoped_miopen_temp_lock_directory_and_file_mode_0777_required": True,
        "global_miopen_lock_root_authoritative": False,
        "global_miopen_lock_root_members_scanned": False,
        "global_miopen_lock_root_cleanup_allowed": False,
        "vae_config_sha256": "f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527",
        "torch_version": "2.7.1+rocm6.3",
        "torch_hip_version": "6.3.42131-fa1d09cbd",
        "miopen_backend_version": 3003000,
        "miopen_bundled_library_resolved_path": "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/torch/lib/libMIOpen.so",
        "miopen_bundled_library_size": 690355265,
        "miopen_bundled_library_sha256": "1e6cc33ca21951dce12795e6c5d99578e8f2f1754b84a703508df44426b44b52",
        "miopen_embedded_version": "3.3.0.a85ca8a54-dirty",
        "r2_failure_step": "136141.115",
        "r2_failure_stack_location": "diffusers/models/autoencoders/autoencoder_kl_wan.py:298",
        "home_must_remain_unchanged": True,
        "phase_failure_create_only_terminal_required": True,
        "phase_failure_shell_fallback_required_if_runtime_tool_terminal_fails": True,
        "cleanup_receipt_publication_failure_success_claim_forbidden": True,
        "retained_failure_scoped_lock_root_present_absent_observation_required": True,
        "post_srun_parent_recomputes_prepare_completion_cleanup_digests": True,
        "post_srun_parent_revalidates_completion_negative_access_and_exact7_authority": True,
        "final_marker_binds_release_and_all_runtime_receipt_hashes_and_digests": True,
        "failed_r1_r2_exact_and_descendant_paths_forbidden": True,
        "failed_r3_generation_paths_forbidden": True,
        "path_dot_component_and_symlink_aliases_forbidden": True,
        "ustar_regular_member_devmajor": 0,
        "ustar_regular_member_devminor": 0,
        "ustar_header_fields_explicitly_normalized": True,
        "ustar_regular_member_devfields_encoding": "sixteen-nul-bytes-checksum-recomputed",
        "ustar_raw_headers_checksums_offsets_and_zero_trailer_reverified": True,
    }


def _topology() -> Mapping[str, Any]:
    return {
        "holder_job_id": 136141,
        "holder_node": "auh7-1b-gpu-299",
        "run_generation": "r4",
        "runtime_cache_root_pattern": "/tmp/BOX-EXP-014-r4-${SLURM_JOB_ID}-${SLURM_STEP_ID}",
        "runtime_tmpdir_pattern": "/tmp/BOX-EXP-014-r4-${SLURM_JOB_ID}-${SLURM_STEP_ID}/tmp",
        "runtime_scoped_miopen_lock_root_pattern": "/tmp/BOX-EXP-014-r4-${SLURM_JOB_ID}-${SLURM_STEP_ID}/tmp/miopen-lockfiles",
        "runtime_cache_statfs_type": "ext2/ext3",
        "runtime_cache_statfs_magic_hex": "0xef53",
        "runtime_cache_mount_fstype": "ext4",
        "runtime_cache_mount_point": "/",
        "runtime_cache_mount_source": "/dev/mapper/vgroot-lvroot",
        "runtime_cache_cleanup": "same-compute-child-after-controller-success-before-numbered-step-exit;retain-on-failure;never-reuse",
        "runtime_cache_prepare_cleanup_node": "auh7-1b-gpu-299",
        "parent_must_not_inspect_or_remove_compute_node_tmp": True,
        "global_tmp_miopen_lock_root_must_not_be_enumerated_or_mutated": True,
        "phase_failure_terminals_written_in_shared_run_root": True,
        "retained_parent_required": True,
        "numbered_child_count": 1,
        "gpu_count": 1,
        "source_rows_encoded_serially": True,
        "fresh_run_root_required": True,
        "parent_cancel_forbidden": True,
        "parent_release_forbidden": True,
        "parent_requeue_forbidden": True,
        "upload_authorized_by_static_release": False,
        "launch_authorized_by_static_release": False,
    }


def build_manifest(method_root: Path) -> tuple[Mapping[str, Any], Mapping[str, bytes]]:
    root = method_root.resolve(strict=True)
    require(root == method_root and root.is_dir() and not root.is_symlink(), "method root must be one canonical directory")
    payloads: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for relative in sorted(FILES_AND_MODES):
        pure = PurePosixPath(relative)
        require(not pure.is_absolute() and ".." not in pure.parts and pure.as_posix() == relative, "release member path differs")
        raw = _stable_plain_bytes(root / relative)
        payloads[relative] = raw
        rows.append(
            {
                "path": relative,
                "mode": FILES_AND_MODES[relative],
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    by_path = {row["path"]: row for row in rows}
    component_pins = {
        name: by_path[path]["sha256"] for name, path in COMPONENT_FILES.items()
    }
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_generation": RELEASE_GENERATION,
        "archive_format": ARCHIVE_FORMAT,
        "member_root": MEMBER_ROOT,
        "file_count": len(rows),
        "files": rows,
        "component_pins": component_pins,
        "allowed_entrypoints": list(ENTRYPOINTS),
        "revision_kind": "content-closure-sha1",
        "content_closure_sha1": hashlib.sha1(canonical_json_bytes(closure)).hexdigest(),
        "git_commit_claimed": False,
        "exact_member_closure": True,
        "release_scope": "BOX-EXP-014-source-only-exact7-reencode-only",
        "authority": _authority(),
        "topology": _topology(),
    }
    manifest = {**unsigned, "manifest_digest": object_sha256(unsigned)}
    validate_manifest(manifest)
    return manifest, payloads


def validate_manifest(value: Any) -> Mapping[str, Any]:
    require(type(value) is dict, "release manifest must be one object")
    required = {
        "schema_version", "release_generation", "archive_format", "member_root",
        "file_count", "files", "component_pins", "allowed_entrypoints",
        "revision_kind", "content_closure_sha1", "git_commit_claimed",
        "exact_member_closure", "release_scope", "authority", "topology",
        "manifest_digest",
    }
    require(set(value) == required, "release manifest field closure differs")
    require(
        value["schema_version"] == SCHEMA_VERSION
        and value["release_generation"] == RELEASE_GENERATION
        and value["archive_format"] == ARCHIVE_FORMAT
        and value["member_root"] == MEMBER_ROOT
        and value["git_commit_claimed"] is False
        and value["exact_member_closure"] is True,
        "release manifest identity differs",
    )
    unsigned = dict(value)
    declared = unsigned.pop("manifest_digest", None)
    require(declared == object_sha256(unsigned), "release manifest digest differs")
    rows = value["files"]
    require(type(rows) is list and len(rows) == len(FILES_AND_MODES) == value["file_count"], "release file count differs")
    expected_paths = sorted(FILES_AND_MODES)
    require([row.get("path") for row in rows] == expected_paths, "release file order/closure differs")
    for row in rows:
        require(
            type(row) is dict
            and set(row) == {"path", "mode", "size", "sha256"}
            and row["mode"] == FILES_AND_MODES[row["path"]]
            and type(row["size"]) is int
            and row["size"] > 0
            and type(row["sha256"]) is str
            and len(row["sha256"]) == 64,
            f"release row differs: {row!r}",
        )
    by_path = {row["path"]: row for row in rows}
    require(
        value["component_pins"]
        == {name: by_path[path]["sha256"] for name, path in COMPONENT_FILES.items()},
        "release component pins differ",
    )
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    require(
        value["content_closure_sha1"]
        == hashlib.sha1(canonical_json_bytes(closure)).hexdigest(),
        "release content closure differs",
    )
    require(value["allowed_entrypoints"] == list(ENTRYPOINTS), "release entrypoints differ")
    require(value["authority"] == _authority(), "release authority differs")
    require(value["topology"] == _topology(), "release topology differs")
    return value


def _tar_info(name: str, size: int, mode: int) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name=name)
    member.size = size
    member.mode = mode
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    member.mtime = 0
    member.devmajor = 0
    member.devminor = 0
    member.pax_headers = {}
    member.type = tarfile.REGTYPE
    return member


def build_archive(manifest: Mapping[str, Any], payloads: Mapping[str, bytes]) -> bytes:
    validate_manifest(manifest)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for row in manifest["files"]:
            raw = payloads[row["path"]]
            require(len(raw) == row["size"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "archive payload differs")
            archive.addfile(
                _tar_info(f"{MEMBER_ROOT}/{row['path']}", len(raw), row["mode"]),
                io.BytesIO(raw),
            )
    raw = bytearray(buffer.getvalue())
    offset = 0
    for row in manifest["files"]:
        header = bytearray(raw[offset : offset + 512])
        require(len(header) == 512, "USTAR header is truncated")
        header[329:345] = b"\x00" * 16
        header[148:156] = b" " * 8
        checksum = sum(header)
        checksum_field = f"{checksum:06o}\x00 ".encode("ascii")
        require(len(checksum_field) == 8, "USTAR checksum field differs")
        header[148:156] = checksum_field
        raw[offset : offset + 512] = header
        offset += 512 + ((row["size"] + 511) // 512) * 512
    require(
        raw[offset:] == b"\x00" * (len(raw) - offset),
        "USTAR trailer differs",
    )
    return bytes(raw)


def _verify_raw_ustar_headers(raw: bytes, manifest: Mapping[str, Any]) -> None:
    """Verify byte-level fields that tarfile intentionally normalizes away."""

    offset = 0
    for row in manifest["files"]:
        require(offset + 512 <= len(raw), "raw USTAR header is truncated")
        header = raw[offset : offset + 512]
        require(
            header[329:345] == b"\x00" * 16,
            "raw USTAR devmajor/devminor fields are not sixteen NUL bytes",
        )
        checksum_header = bytearray(header)
        checksum_header[148:156] = b" " * 8
        checksum = sum(checksum_header)
        expected_checksum = f"{checksum:06o}\x00 ".encode("ascii")
        require(
            len(expected_checksum) == 8
            and header[148:156] == expected_checksum,
            "raw USTAR header checksum differs",
        )
        expected_name = f"{MEMBER_ROOT}/{row['path']}".encode("ascii")
        require(
            len(expected_name) <= 100
            and header[0:100].split(b"\x00", 1)[0] == expected_name,
            "raw USTAR member name/offset differs",
        )
        data_start = offset + 512
        data_end = data_start + row["size"]
        padded_end = data_start + ((row["size"] + 511) // 512) * 512
        require(
            data_end <= len(raw)
            and hashlib.sha256(raw[data_start:data_end]).hexdigest()
            == row["sha256"]
            and raw[data_end:padded_end] == b"\x00" * (padded_end - data_end),
            "raw USTAR payload boundary/padding differs",
        )
        offset = padded_end
    expected_archive_size = (
        (offset + 1024 + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE
    ) * tarfile.RECORDSIZE
    require(
        len(raw) == expected_archive_size
        and raw[offset:] == b"\x00" * (len(raw) - offset),
        "raw USTAR zero trailer differs",
    )


def verify_archive_bytes(raw: bytes, manifest: Mapping[str, Any]) -> None:
    validate_manifest(manifest)
    _verify_raw_ustar_headers(raw, manifest)
    expected = [f"{MEMBER_ROOT}/{row['path']}" for row in manifest["files"]]
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
            require([member.name for member in members] == expected, "archive member closure differs")
            for member, row in zip(members, manifest["files"]):
                handle = archive.extractfile(member)
                require(
                    member.isfile()
                    and not member.issym()
                    and not member.islnk()
                    and member.uid == member.gid == member.mtime == 0
                    and member.devmajor == member.devminor == 0
                    and stat.S_IMODE(member.mode) == row["mode"]
                    and member.size == row["size"]
                    and handle is not None
                    and hashlib.sha256(handle.read()).hexdigest() == row["sha256"],
                    f"archive member differs: {member.name}",
                )
    except (OSError, tarfile.TarError) as error:
        raise Source7ReencodeReleaseError("cannot verify release archive") from error


def _write_create_only(path: Path, raw: bytes, mode: int) -> None:
    require(path.is_absolute() and not path.exists() and not path.is_symlink(), "release output must be a fresh absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: Optional[int] = None
    temporary: Optional[Path] = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(method_root: Path, archive_path: Path, manifest_path: Path) -> Mapping[str, Any]:
    manifest, payloads = build_manifest(method_root)
    archive = build_archive(manifest, payloads)
    verify_archive_bytes(archive, manifest)
    require(build_archive(manifest, payloads) == archive, "release archive rebuild is not byte-identical")
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(archive_path, archive, 0o444)
    _write_create_only(manifest_path, manifest_raw, 0o444)
    require(archive_path.read_bytes() == archive and manifest_path.read_bytes() == manifest_raw, "published release replay differs")
    return {
        "schema_version": SCHEMA_VERSION,
        "release_generation": RELEASE_GENERATION,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "file_count": manifest["file_count"],
        "upload_authorized": False,
        "launch_authorized": False,
    }


def _load_json(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    raw = _stable_plain_bytes(path)
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "release manifest file SHA differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Source7ReencodeReleaseError("release manifest is not valid JSON") from error
    require(type(value) is dict and raw == canonical_json_bytes(value) + b"\n", "release manifest is not canonical JSON")
    return validate_manifest(value)


def audit(
    *, archive_path: Path, expected_archive_sha256: str,
    manifest_path: Path, expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    manifest = _load_json(manifest_path, expected_manifest_sha256)
    archive = _stable_plain_bytes(archive_path)
    require(hashlib.sha256(archive).hexdigest() == expected_archive_sha256, "release archive SHA differs")
    verify_archive_bytes(archive, manifest)
    return {
        "schema_version": "bernini-full30-action-source7-reencode-release-audit-v1",
        "archive_sha256": expected_archive_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "manifest_digest": manifest["manifest_digest"],
        "exact_member_closure": True,
        "static_audit_go": True,
        "upload_authorized": False,
        "launch_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--method-root", required=True)
    builder.add_argument("--archive", required=True)
    builder.add_argument("--manifest", required=True)
    auditor = commands.add_parser("audit")
    auditor.add_argument("--archive", required=True)
    auditor.add_argument("--expected-archive-sha256", required=True)
    auditor.add_argument("--manifest", required=True)
    auditor.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        result = build(Path(args.method_root), Path(args.archive), Path(args.manifest))
    else:
        result = audit(
            archive_path=Path(args.archive),
            expected_archive_sha256=args.expected_archive_sha256,
            manifest_path=Path(args.manifest),
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
