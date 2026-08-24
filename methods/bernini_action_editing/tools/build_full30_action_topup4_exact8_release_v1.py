#!/usr/bin/env python3
"""Build/audit the deterministic BOX-EXP-010 topup4 exact8 release."""

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
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-full30-action-topup4-exact8-release-v1"
RELEASE_GENERATION = "r1"
ARCHIVE_FORMAT = "ustar-owner0-mtime0-exact-modes-v1"
MEMBER_ROOT = "methods/bernini_action_editing"
RESOURCE_SOURCE = "tools/reserve4_fixed_generation_sp4_v1.py"
RESOURCE_SPECIALIZED = (
    "tools/reserve4_fixed_generation_sp4_136140_specialized_v1.py"
)
RESOURCE_SOURCE_SHA256 = (
    "be722e4020040ba446f290f07378e870e2d3c1a4228ec997c3447770fcb53d5d"
)
RESOURCE_SPECIALIZED_SHA256 = (
    "aa2f5c01c9d231ad5340cbb572c1523546fa2e148143ee1b5bf04f53f005f017"
)
RESOURCE_SIZE = 166_064
SELECTION_MEMBER = "assets/full30_action_minimal_cross_anchor_topup4_v1.json"
SELECTION_SHA256 = (
    "72a1d58ede5381f57d2fa8ef895a7e9d5c11b3872e87ecbc3e08fec0cc5ef38e"
)
PARENT_REGISTRY_MEMBER = (
    "assets/mosaic_event_population_compact6_topup20_v1.json"
)
PARENT_REGISTRY_SHA256 = (
    "71906510d162e6626338b5785fd1cf55b437de5ba77d9b9b122ad761694f8e62"
)
R10_R13_PARITY_EVIDENCE: Mapping[str, Any] = {
    "schema_version": "bernini-generic-action-fit40-r10-parity-evidence-v1",
    "compile_smoke_receipt": {
        "path": (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/"
            "bernini_generic_source_anchored_action_v1_20260814/data_prep/"
            "fit40-generation-136141-r10-f5551895-r1/logs/"
            "compile-smoke-receipt.json"
        ),
        "file_sha256": (
            "e1b23a75258fac7dfcae0528c0a62c789365f683c1f096f5c2ba36ca7b1f34a3"
        ),
    },
    "generation_log": {
        "path": (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/"
            "bernini_generic_source_anchored_action_v1_20260814/data_prep/"
            "fit40-generation-136141-r10-f5551895-r1/logs/"
            "generation-fit-all8-serial4.log"
        ),
        "file_sha256": (
            "2998413eb2e37821b55dfdbfa43486063d0d97188de84680aceb9bf339a3d8dc"
        ),
    },
    "fit_r13_resource_preimage_sha256": RESOURCE_SOURCE_SHA256,
    "tensor_values_handwritten": False,
}

_RUNTIME_FILES = (
    "infer_lora.py",
    "infer_native_identity_generation_canary.py",
    "infer_pair_v5_t2v_calibration_bank.py",
    "infer_source_kv_carrier_oracle.py",
    "infer_source_value_residual_oracle.py",
    "pair_v5_t2v_calibration_bank_spec.py",
    "source_kv_replay.py",
    "source_kv_route_batches.py",
    "source_value_residual.py",
    "train_lora.py",
)
_PLAN_AND_CONTROL_FILES = (
    "mosaic_event_population_authoring.py",
    "full30_action_minimal_cross_anchor_topup4_plan_v1.py",
    "full30_action_topup4_exact8_plan_v1.py",
    "full30_action_topup4_exact8_generator_v1.py",
    "full30_action_topup4_exact8_controller_v1.py",
)
_TOOLS = (
    "tools/build_full30_action_topup4_exact8_release_v1.py",
    "tools/build_pair_v5_t2v_seed2_bank.py",
    "tools/build_renderer_dataset.py",
    "tools/materialize_vae.py",
    RESOURCE_SOURCE,
    RESOURCE_SPECIALIZED,
)
_ASSETS = (SELECTION_MEMBER, PARENT_REGISTRY_MEMBER)
FILES_AND_MODES: Mapping[str, int] = {
    **{
        path: 0o444
        for path in _RUNTIME_FILES + _PLAN_AND_CONTROL_FILES + _TOOLS + _ASSETS
    },
    "scripts/auh_full30_action_topup4_exact8_136140_world4_v1.sh": 0o555,
    "scripts/auh_generic_action_data_prep_rank_exec_v1.sh": 0o555,
}
COMPONENT_FILES: Mapping[str, str] = {
    "selection_sha256": SELECTION_MEMBER,
    "parent_registry_sha256": PARENT_REGISTRY_MEMBER,
    "source_plan_sha256": "full30_action_minimal_cross_anchor_topup4_plan_v1.py",
    "exact8_plan_sha256": "full30_action_topup4_exact8_plan_v1.py",
    "generator_sha256": "full30_action_topup4_exact8_generator_v1.py",
    "controller_sha256": "full30_action_topup4_exact8_controller_v1.py",
    "release_builder_sha256": (
        "tools/build_full30_action_topup4_exact8_release_v1.py"
    ),
    "launcher_sha256": (
        "scripts/auh_full30_action_topup4_exact8_136140_world4_v1.sh"
    ),
    "rank_cache_wrapper_sha256": (
        "scripts/auh_generic_action_data_prep_rank_exec_v1.sh"
    ),
    "fit_r13_resource_preimage_sha256": RESOURCE_SOURCE,
    "resource_136140_specialization_sha256": RESOURCE_SPECIALIZED,
}
ENTRYPOINTS = (
    "full30_action_topup4_exact8_controller_v1.py",
    "scripts/auh_full30_action_topup4_exact8_136140_world4_v1.sh",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Topup4Exact8ReleaseError(RuntimeError):
    """Raised before an ambiguous or mutable release can pass."""


def fail(message: str) -> NoReturn:
    raise Topup4Exact8ReleaseError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise Topup4Exact8ReleaseError(
            "release is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_plain_bytes(path: Path) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        fail("release input must be absolute and non-symlink")
    try:
        resolved = path.resolve(strict=True)
        before = path.stat()
    except OSError as error:
        raise Topup4Exact8ReleaseError("release input is unavailable") from error
    if resolved != path or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        fail("release input must be one canonical single-link plain file")
    raw = path.read_bytes()
    after = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if (
        identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size
        or not raw
    ):
        fail("release input changed while reading or is empty")
    return raw


def specialize_resource_bytes(raw: bytes) -> bytes:
    if (
        len(raw) != RESOURCE_SIZE
        or hashlib.sha256(raw).hexdigest() != RESOURCE_SOURCE_SHA256
        or raw.count(b"136141") != 7
        or raw.count(b"136140") != 0
    ):
        fail("fit-r13 resource specialization preimage differs")
    specialized = raw.replace(b"136141", b"136140")
    if (
        len(specialized) != len(raw)
        or hashlib.sha256(specialized).hexdigest() != RESOURCE_SPECIALIZED_SHA256
        or specialized.count(b"136141") != 0
        or specialized.count(b"136140") != 7
        or specialized.replace(b"136140", b"136141") != raw
    ):
        fail("136140 resource specialization postimage differs")
    return specialized


def _authority() -> Mapping[str, Any]:
    return {
        "experiment_id": "BOX-EXP-010",
        "purpose": "author four missing cross-anchor comparator cells",
        "learning_target": "N/A; representation evidence only",
        "numeric_target": "N/A; q and a_min remain downstream materializer outputs",
        "dataset": "minimal_cross_anchor_topup4_exact8",
        "analysis_split": "fit",
        "comparator_cell_count": 4,
        "formal_candidate_count": 8,
        "formal_branch_order": ["action", "incomplete"],
        "diagnostic_task_count": 0,
        "diagnostic_generation_allowed": False,
        "selected_cells": [
            ["human-arms-raised-to-hands-on-hips", "man-gym-raised-arms", 2026081205, "sp4-a"],
            ["human-head-turn-forward-and-smile", "young-man-library-angled-head", 2026081209, "sp4-a"],
            ["human-peace-sign-to-open-palm-wave", "woman-art-studio-peace", 2026081213, "sp4-b"],
            ["human-left-fist-to-forward-palm-down", "woman-office-left-fist", 2026081217, "sp4-b"],
        ],
        "action_and_incomplete_each_require_independent_full81_pass": True,
        "same_gaussian_action_incomplete_required_per_cell": True,
        "same_state_controls_source": "official_full30_action_psiout_materializer",
        "generated_media_can_enter_q_before_review_and_materializer": False,
        "generated_media_can_enter_a_min_before_review_and_materializer": False,
        "generated_media_can_enter_optimizer": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "training_authorized": False,
    }


def _topology() -> Mapping[str, Any]:
    return {
        "holder": {"job_id": 136140, "node": "auh7-1b-gpu-215"},
        "slurm_child_gpu_count": 8,
        "compute_world_size": 4,
        "parallelism": "dp1_sp4_one_model_replica_at_a_time",
        "formal_shard_count": 2,
        "formal_world4_model_invocation_count": 8,
        "compile_smoke_world4_model_invocation_count": 1,
        "total_required_native_model_invocation_count": 9,
        "diagnostic_model_invocation_count": 0,
        "all_required_model_invocations_strictly_serial": True,
        "physical_island_order": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "fresh_run_root_required": True,
        "host_memory_request_gib": 60,
        "host_sampled_current_safe_ceiling_gib": 56,
        "host_cgroup_sample_interval_ns": 10_000_000,
        "host_cgroup_max_sample_gap_ns": 100_000_000,
        "t2v_rank_gpu_memory_limit_gib": 52,
        "per_rank_node_local_cache_wrapper": True,
        "nfs_comgr_tmp_rejected": True,
        "serialized_world4_host_checkpoint_load": True,
        "world4_all_renderer_loads_complete_before_first_sampling": True,
        "t2v_text_encoder_rank_gpu_residency_required": True,
        "physical_safetensors_safe_open_required": True,
        "r10_r13_mp4_gaussian_clean_latent_parity_required": True,
        "disposable_smoke_formal_candidate_count_at_gate": 0,
        "terminal_zero_oom_and_oom_kill_required": True,
    }


def build_manifest(
    method_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, bytes]]:
    root = method_root.resolve(strict=True)
    if root != method_root or root.is_symlink() or not root.is_dir():
        fail("method root must be one canonical directory")
    payloads: dict[str, bytes] = {}
    source_bytes = _stable_plain_bytes(root / RESOURCE_SOURCE)
    specialized_bytes = specialize_resource_bytes(source_bytes)
    rows: list[dict[str, Any]] = []
    for relative in sorted(FILES_AND_MODES):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            fail("release member path differs")
        raw = (
            specialized_bytes
            if relative == RESOURCE_SPECIALIZED
            else _stable_plain_bytes(root / relative)
        )
        payloads[relative] = raw
        rows.append(
            {
                "path": relative,
                "mode": FILES_AND_MODES[relative],
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    row_by_path = {row["path"]: row for row in rows}
    if row_by_path[SELECTION_MEMBER]["sha256"] != SELECTION_SHA256:
        fail("topup4 selection asset differs")
    if row_by_path[PARENT_REGISTRY_MEMBER]["sha256"] != PARENT_REGISTRY_SHA256:
        fail("topup4 parent registry differs")
    component_pins = {
        label: row_by_path[relative]["sha256"]
        for label, relative in COMPONENT_FILES.items()
    }
    closure = {"member_root": MEMBER_ROOT, "files": rows}
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "release_generation": RELEASE_GENERATION,
        "archive_format": ARCHIVE_FORMAT,
        "member_root": MEMBER_ROOT,
        "file_count": len(rows),
        "files": rows,
        "component_pins": component_pins,
        "allowed_entrypoints": list(ENTRYPOINTS),
        "revision_kind": "content-closure-sha1",
        "content_closure_sha1": hashlib.sha1(
            canonical_json_bytes(closure)
        ).hexdigest(),
        "exact_member_closure": True,
        "release_scope": "BOX-EXP-010-minimal-cross-anchor-topup4-exact8-only",
        "resource_specialization": {
            "source_member": RESOURCE_SOURCE,
            "source_sha256": RESOURCE_SOURCE_SHA256,
            "source_holder_job": 136141,
            "specialized_member": RESOURCE_SPECIALIZED,
            "specialized_sha256": RESOURCE_SPECIALIZED_SHA256,
            "topup_holder_job": 136140,
            "replacement_count": 7,
            "same_length_required": True,
            "all_non_holder_bytes_identical_required": True,
            "fit_r13_source_modified": False,
        },
        "external_evidence": R10_R13_PARITY_EVIDENCE,
        "topology": _topology(),
        "authority": _authority(),
    }
    manifest = {**unsigned, "manifest_digest": object_sha256(unsigned)}
    validate_manifest(manifest)
    return manifest, payloads


def build_archive(
    manifest: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for row in manifest["files"]:
            relative = row["path"]
            raw = payloads[relative]
            info = tarfile.TarInfo(f"{MEMBER_ROOT}/{relative}")
            info.size = len(raw)
            info.mode = row["mode"]
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(raw))
    result = stream.getvalue()
    verify_archive(result, manifest)
    return result


def verify_archive(raw: bytes, manifest: Mapping[str, Any]) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
            expected_names = [
                f"{MEMBER_ROOT}/{row['path']}" for row in manifest["files"]
            ]
            if [member.name for member in members] != expected_names:
                fail("archive exact member order differs")
            for member, row in zip(members, manifest["files"]):
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.uid != 0
                    or member.gid != 0
                    or member.mtime != 0
                    or member.mode != row["mode"]
                    or member.size != row["size"]
                ):
                    fail(f"archive member metadata differs: {member.name}")
                handle = archive.extractfile(member)
                if handle is None:
                    fail(f"archive member cannot be read: {member.name}")
                payload = handle.read()
                if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                    fail(f"archive member content differs: {member.name}")
    except (tarfile.TarError, OSError) as error:
        raise Topup4Exact8ReleaseError("archive is invalid") from error


def validate_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    unsigned = dict(value)
    declared = unsigned.pop("manifest_digest", None)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("release_generation") != RELEASE_GENERATION
        or value.get("archive_format") != ARCHIVE_FORMAT
        or value.get("member_root") != MEMBER_ROOT
        or value.get("exact_member_closure") is not True
        or value.get("allowed_entrypoints") != list(ENTRYPOINTS)
        or not isinstance(declared, str)
        or SHA256_RE.fullmatch(declared) is None
        or object_sha256(unsigned) != declared
    ):
        fail("release manifest schema/digest differs")
    rows = value.get("files")
    if (
        not isinstance(rows, list)
        or len(rows) != len(FILES_AND_MODES)
        or value.get("file_count") != len(rows)
        or [row.get("path") for row in rows] != sorted(FILES_AND_MODES)
    ):
        fail("release manifest file closure differs")
    seen: set[str] = set()
    for row in rows:
        path = row.get("path")
        if (
            not isinstance(path, str)
            or path in seen
            or row.get("mode") != FILES_AND_MODES.get(path)
            or type(row.get("size")) is not int
            or row["size"] <= 0
            or SHA256_RE.fullmatch(str(row.get("sha256"))) is None
        ):
            fail("release manifest file row differs")
        seen.add(path)
    row_by_path = {row["path"]: row for row in rows}
    expected_components = {
        label: row_by_path[path]["sha256"]
        for label, path in COMPONENT_FILES.items()
    }
    specialization = value.get("resource_specialization", {})
    authority = value.get("authority", {})
    topology = value.get("topology", {})
    if (
        value.get("component_pins") != expected_components
        or row_by_path[RESOURCE_SOURCE]["sha256"] != RESOURCE_SOURCE_SHA256
        or row_by_path[RESOURCE_SPECIALIZED]["sha256"]
        != RESOURCE_SPECIALIZED_SHA256
        or row_by_path[SELECTION_MEMBER]["sha256"] != SELECTION_SHA256
        or row_by_path[PARENT_REGISTRY_MEMBER]["sha256"]
        != PARENT_REGISTRY_SHA256
        or specialization.get("fit_r13_source_modified") is not False
        or specialization.get("replacement_count") != 7
        or specialization.get("topup_holder_job") != 136140
        or authority != _authority()
        or topology != _topology()
    ):
        fail("release authority/topology differs")
    return value


def _write_create_only(path: Path, raw: bytes, *, mode: int) -> None:
    if (
        not path.is_absolute()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.exists()
        or path.is_symlink()
    ):
        fail("release output must be a fresh absolute path")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def build(method_root: Path, archive: Path, manifest_path: Path) -> Mapping[str, Any]:
    manifest, payloads = build_manifest(method_root)
    archive_raw = build_archive(manifest, payloads)
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_create_only(archive, archive_raw, mode=0o400)
    try:
        _write_create_only(manifest_path, manifest_raw, mode=0o400)
    except Exception:
        archive.unlink(missing_ok=True)
        raise
    verify_archive(archive.read_bytes(), manifest)
    return {
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "file_count": manifest["file_count"],
    }


def _unique_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def audit(
    archive: Path, expected_archive_sha256: str, manifest_path: Path,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    archive_raw = _stable_plain_bytes(archive)
    manifest_raw = _stable_plain_bytes(manifest_path)
    if hashlib.sha256(archive_raw).hexdigest() != expected_archive_sha256:
        fail("release archive SHA-256 differs")
    if hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha256:
        fail("release manifest SHA-256 differs")
    try:
        manifest = json.loads(
            manifest_raw,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Topup4Exact8ReleaseError(
            "release manifest is invalid JSON"
        ) from error
    if canonical_json_bytes(manifest) + b"\n" != manifest_raw:
        fail("release manifest bytes are not canonical")
    validate_manifest(manifest)
    verify_archive(archive_raw, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--method-root", required=True)
    create.add_argument("--archive", required=True)
    create.add_argument("--manifest", required=True)
    check = commands.add_parser("audit")
    check.add_argument("--archive", required=True)
    check.add_argument("--expected-archive-sha256", required=True)
    check.add_argument("--manifest", required=True)
    check.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        value = build(Path(args.method_root), Path(args.archive), Path(args.manifest))
    else:
        manifest = audit(
            Path(args.archive),
            args.expected_archive_sha256,
            Path(args.manifest),
            args.expected_manifest_sha256,
        )
        value = {
            "archive": args.archive,
            "archive_sha256": args.expected_archive_sha256,
            "manifest": args.manifest,
            "manifest_sha256": args.expected_manifest_sha256,
            "manifest_digest": manifest["manifest_digest"],
            "file_count": manifest["file_count"],
        }
    print(canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
