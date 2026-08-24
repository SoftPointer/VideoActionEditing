#!/usr/bin/env python3
"""Build the isolated Stage-A schedule x block localization release.

The audited Stage-B r3 inference release is treated as an immutable base.  The
diagnostic policy/core, Stage-A entry point, its eight exact identity-runtime
dependencies, and two authoring registries are the only appended members.
Publication is deterministic, exact-closure, and create-only.  This builder
grants no optimizer, routing, or scientific-selection authority.
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
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-schedule-block-causal-localization-release-v3"
RELEASE_GENERATION = "r3"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-mode0444-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
BASE_ARCHIVE_SHA256 = "e3880934c3e6cfcb0dfe56aa34a03f3ffbb2cb192a262fdb8ae1734a02f183ca"
BASE_MANIFEST_SHA256 = "6849ed11ad214e4c49f72731e4beb88948f2abf26e79f0ff5cf8c4e2814e62a3"
BASE_SCHEMA = "bernini-source-noised-carrier-stage-b-inference-release-v3"
BASE_GENERATION = "r3"
BASE_FILES = (
    "source_self_role_repaint.py",
    "source_self_runtime.py",
    "train_source_self_role_repaint.py",
    "train_lora.py",
    "assets/source_self_role_repaint_canary_spec_v2.json",
    "tools/materialize_source_self_role_repaint.py",
    "tools/materialize_ramp_motion_analogy_vae.py",
    "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py",
    "inference_sigma_strata.py",
    "source_noised_ladder_v1.py",
    "train_source_noised_carrier_strata_v1.py",
    "infer_lora.py",
    "infer_source_noised_carrier_stage_b_v1.py",
)
ADDED_FILES = (
    "schedule_block_causal_policy_v1.py",
    "schedule_block_target_row_prompt_swap_v1.py",
    "infer_schedule_block_causal_localization_v1.py",
    "train_source_self_identity_orbit_v4.py",
    "source_self_native_ref_contrastive_v3.py",
    "appearance_counterfactual_identity_orbit.py",
    "source_self_identity_orbit_v4.py",
    "source_self_native_rv2v_guidance.py",
    "source_self_native_target_adapter.py",
    "tools/materialize_appearance_counterfactual_identity_orbit.py",
    "mdr_exact_motion_analogy.py",
    "assets/pair_v5_t2v_calibration_first8_authoring_v1.json",
    "assets/appearance_identity_orbit_portrait2_review_v1.json",
)
RELEASE_FILES = BASE_FILES + ADDED_FILES
EXPECTED_FILE_SHA256 = {
    "source_self_role_repaint.py": "bf212ac4effcd5b3975eefc61e01c71cba366969ec92cf2ff186765ddec43f2e",
    "source_self_runtime.py": "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f",
    "train_source_self_role_repaint.py": "357ba5310a297c042e1c1bd10bef35bb69e483e18ff15b5ba4cc2bd65a07c80d",
    "train_lora.py": "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
    "assets/source_self_role_repaint_canary_spec_v2.json": "62468b24d4a57ec03d42ce8c006a707cbcf56588ef62d10632089eb5ad457920",
    "tools/materialize_source_self_role_repaint.py": "8065cafc34c15d7e8e6fc8e3abb13551b2cbe20c925ab8415267be5b3993cc80",
    "tools/materialize_ramp_motion_analogy_vae.py": "ca9b4620ad7dc6cd03e70b180f68d83aad05c21cef574fe6467bdaa1202bb93a",
    "tools/materialize_vae.py": "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
    "tools/build_renderer_dataset.py": "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
    "inference_sigma_strata.py": "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
    "source_noised_ladder_v1.py": "eb8653a5e98d0744c9fd7066f3aefc4c5e0dfcd8f70320e86a2e669a376fef98",
    "train_source_noised_carrier_strata_v1.py": "39c3fad7e8d710eedd453e75b1acf7fb35f30c0ccba4dee71d336efec5274704",
    "infer_lora.py": "babd6d63287723ccd14b2bbe43bd4550c30b4feaa794d17c66f5a5ddefe979fe",
    "infer_source_noised_carrier_stage_b_v1.py": "7e6cdba95c62d2ae9bbe81cfa123ac208c2ca890f134cfe6d0538cefea68db50",
    "schedule_block_causal_policy_v1.py": "1be281b0419a23254d51556a41eda0d014ecd75cb044caaf5e3ceb96f7c54998",
    # Frozen only after the independently-owned core passed AUH Torch 2.7.1
    # dynamic tests and an independent hostile review.
    "schedule_block_target_row_prompt_swap_v1.py": "385cc2321da888f75d5aff5017175b85acf06174969aaa39210b802cc14695c5",
    # Frozen only after model-free, AUH dynamic, and independent hostile audits.
    "infer_schedule_block_causal_localization_v1.py": "05b62e8575a2421b535f533530f5e075a12f34814394408fa03f2f51f891c9da",
    "train_source_self_identity_orbit_v4.py": "5f86af928be2d087d178fb7f106a06e5a523dd4bf1152da52eb9cb54064fba2d",
    "source_self_native_ref_contrastive_v3.py": "d8825bc167c64e497f8d29c807d9b0a69d9a9a59de09afee863b7fc9df2bdeb0",
    "appearance_counterfactual_identity_orbit.py": "81c6c2f3b579d77c74143bcf7badd0c3b74d06b34cb0511d0929972db25a89fc",
    "source_self_identity_orbit_v4.py": "40ed67f23cde219bc39abdabb9059a2a05a76edc67204f6125aee181f6b580bd",
    "source_self_native_rv2v_guidance.py": "8737c286af650df0dc29bab2cfd512afcdd8167ba061dc15c275e363e1b16359",
    "source_self_native_target_adapter.py": "9a5c2dda4cc87089c5ff05c155e72b1081b80577bf5128ffc97051f0044b6c47",
    "tools/materialize_appearance_counterfactual_identity_orbit.py": "aac6b3e7708e462bca410e90b028fc04865e83a5ef81000250d22ce1711693b6",
    "mdr_exact_motion_analogy.py": "b76c01439d45965d9eb043fdacbd4f8a04351fcc3083b1ef82ad4b55e6528cfb",
    "assets/pair_v5_t2v_calibration_first8_authoring_v1.json": "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c",
    "assets/appearance_identity_orbit_portrait2_review_v1.json": "dc2d83322357196cec84418ddf4318d9fc7d1eb41269cb216739bae7c6169651",
}


class ReleaseError(RuntimeError):
    """Raised before an ambiguous or mutable release can be published."""


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


def _plain_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ReleaseError(f"{label} must be an absolute non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path or not stat.S_ISREG(resolved.lstat().st_mode):
        raise ReleaseError(f"{label} must be a canonical plain file")
    return resolved


def _stable_bytes(path: Path, *, label: str) -> bytes:
    path = _plain_file(path, label=label)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise ReleaseError(f"{label} changed while reading")
    return raw


def _validated_base_payloads(
    archive_path: Path, manifest_path: Path
) -> tuple[dict[str, bytes], dict[str, Any]]:
    archive_raw = _stable_bytes(archive_path, label="base archive")
    manifest_raw = _stable_bytes(manifest_path, label="base manifest")
    if hashlib.sha256(archive_raw).hexdigest() != BASE_ARCHIVE_SHA256:
        raise ReleaseError("base archive SHA differs")
    if hashlib.sha256(manifest_raw).hexdigest() != BASE_MANIFEST_SHA256:
        raise ReleaseError("base manifest SHA differs")
    try:
        manifest = json.loads(manifest_raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseError("cannot parse base manifest") from error
    unsigned = dict(manifest)
    declared = unsigned.pop("manifest_digest", None)
    rows = manifest.get("files")
    if (
        manifest.get("schema_version") != BASE_SCHEMA
        or manifest.get("release_generation") != BASE_GENERATION
        or manifest.get("archive_format") != ARCHIVE_FORMAT
        or manifest.get("member_root") != MEMBER_ROOT
        or manifest.get("file_count") != len(BASE_FILES)
        or manifest.get("exact_member_closure") is not True
        or manifest.get("git_commit_claimed") is not False
        or declared != object_sha256(unsigned)
        or not isinstance(rows, list)
        or [row.get("path") for row in rows if isinstance(row, Mapping)]
        != list(BASE_FILES)
    ):
        raise ReleaseError("base manifest contract differs")
    payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as archive:
            members = archive.getmembers()
            expected_names = [f"{MEMBER_ROOT}/{name}" for name in BASE_FILES]
            if [member.name for member in members] != expected_names:
                raise ReleaseError("base archive member closure differs")
            for member, row, relative in zip(members, rows, BASE_FILES):
                stream = archive.extractfile(member)
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.uid != 0
                    or member.gid != 0
                    or member.mtime != 0
                    or stat.S_IMODE(member.mode) != 0o444
                    or stream is None
                ):
                    raise ReleaseError(f"base member metadata differs: {relative}")
                raw = stream.read()
                digest = hashlib.sha256(raw).hexdigest()
                if (
                    digest != row.get("sha256")
                    or digest != EXPECTED_FILE_SHA256[relative]
                    or len(raw) != row.get("size")
                    or row.get("mode") != "0444"
                ):
                    raise ReleaseError(f"base member bytes differ: {relative}")
                payloads[relative] = raw
    except (OSError, tarfile.TarError) as error:
        raise ReleaseError("cannot verify base archive") from error
    return payloads, manifest


def build_manifest(
    method_root: Path, base_archive: Path, base_manifest: Path
) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = method_root.resolve(strict=True)
    if root != method_root or not root.is_dir() or root.is_symlink():
        raise ReleaseError("method root must be a canonical plain directory")
    payloads, base = _validated_base_payloads(base_archive, base_manifest)
    for relative in ADDED_FILES:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise ReleaseError("release path traversal is forbidden")
        raw = _stable_bytes(root / relative, label=relative)
        if hashlib.sha256(raw).hexdigest() != EXPECTED_FILE_SHA256[relative]:
            raise ReleaseError(f"frozen Stage-A source SHA differs: {relative}")
        payloads[relative] = raw
    rows = [
        {
            "path": name,
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            "size": len(payloads[name]),
            "mode": "0444",
        }
        for name in RELEASE_FILES
    ]
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "release_generation": RELEASE_GENERATION,
        "archive_format": ARCHIVE_FORMAT,
        "member_root": MEMBER_ROOT,
        "file_count": len(rows),
        "files": rows,
        "revision_kind": "content-closure-sha1",
        "content_closure_sha1": hashlib.sha1(canonical_json_bytes(closure)).hexdigest(),
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "base_manifest_digest": base["manifest_digest"],
        "formal_profile": "smoke-then-full-fixed",
        "single_model_load_required": True,
        "engineering_c0_decoded_output_count": 6,
        "engineering_c0_plan_digest": "d11dbd0cfca34f26ea5f72bdd2f5ed8b21c512387410b659ade9f217d866c923",
        "preregistered_full_grid_decoded_output_count": 112,
        "preregistered_full_grid_plan_digest": "6fd3299a1af84968bebe12cd6f1b2a84feb0fb28a07d29619fbcfac66bf4d2e8",
        "formal_total_decoded_output_count": 118,
        "formal_full_continuation_automatic_after_c0_pass": True,
        "c0_failure_forbids_full_grid": True,
        "c0_gate_engineering_only": True,
        "engineering_c0_has_no_visual_or_scientific_selection": True,
        "prompt_calibration_action_reverse_direction_passed": True,
        "prompt_calibration_noop_incomplete_semantics_passed": False,
        "negative_cluster_semantically_validated": False,
        "negative_cluster_scientific_veto_authorized": False,
        "full_grid_cells_retained_without_deletion": True,
        "diagnostic_only": True,
        "optimizer_authorized": False,
        "parameter_update_authorized": False,
        "scientific_selection_authorized": False,
        "git_commit_claimed": False,
        "exact_member_closure": True,
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}, payloads


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = 0o444
    member.uid = member.gid = member.mtime = 0
    member.uname = member.gname = ""
    member.type = tarfile.REGTYPE
    return member


def build_archive(
    manifest: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for row in manifest["files"]:
            raw = payloads[str(row["path"])]
            archive.addfile(
                _tar_info(f"{MEMBER_ROOT}/{row['path']}", len(raw)),
                io.BytesIO(raw),
            )
    return buffer.getvalue()


def verify_archive_bytes(raw: bytes, manifest: Mapping[str, Any]) -> None:
    expected = [f"{MEMBER_ROOT}/{row['path']}" for row in manifest["files"]]
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != expected:
                raise ReleaseError("Stage-A archive closure differs")
            for member, row in zip(members, manifest["files"]):
                stream = archive.extractfile(member)
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or stream is None
                    or member.uid != 0
                    or member.gid != 0
                    or member.mtime != 0
                    or stat.S_IMODE(member.mode) != 0o444
                    or member.size != row["size"]
                    or hashlib.sha256(stream.read()).hexdigest() != row["sha256"]
                ):
                    raise ReleaseError(f"Stage-A archive member differs: {member.name}")
    except (OSError, tarfile.TarError) as error:
        raise ReleaseError("cannot verify Stage-A archive") from error


def _write_create_only(path: Path, raw: bytes) -> None:
    _require_fresh_output(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_fresh_output(path: Path) -> None:
    if (
        not path.is_absolute()
        or Path(os.path.normpath(path)) != path
        or path.exists()
        or path.is_symlink()
    ):
        raise ReleaseError("release outputs must be fresh canonical absolute paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve(strict=True) != path.parent:
        raise ReleaseError("release output parent must be a canonical plain directory")


def build(
    method_root: Path,
    base_archive: Path,
    base_manifest: Path,
    archive: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest, payloads = build_manifest(method_root, base_archive, base_manifest)
    raw = build_archive(manifest, payloads)
    verify_archive_bytes(raw, manifest)
    if build_archive(manifest, payloads) != raw:
        raise ReleaseError("Stage-A archive rebuild is not byte-identical")
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    if archive == manifest_path:
        raise ReleaseError("archive and manifest outputs must be distinct")
    # Pair-level preflight prevents a known existing peer from leaving a
    # newly-published half release.  The individual link operations remain
    # create-only and repeat the check to fail closed on a publication race.
    _require_fresh_output(archive)
    _require_fresh_output(manifest_path)
    _write_create_only(archive, raw)
    _write_create_only(manifest_path, manifest_raw)
    return {
        "schema_version": SCHEMA_VERSION,
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "content_closure_sha1": manifest["content_closure_sha1"],
        "file_count": manifest["file_count"],
        "diagnostic_only": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--base-archive", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    result = build(
        *(Path(value) for value in (
            args.method_root,
            args.base_archive,
            args.base_manifest,
            args.archive,
            args.manifest,
        ))
    )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
