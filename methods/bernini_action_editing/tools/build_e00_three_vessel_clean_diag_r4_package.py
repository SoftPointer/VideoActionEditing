#!/usr/bin/env python3
"""Build and verify the cache-free, overlay-pinned R4 E00 package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Optional, Sequence

try:
    import build_e00_three_vessel_fresh_keyed_legacy_package_v1 as legacy_package
    import e00_three_vessel_clean_diag_r4_overlay_pins as pin_authority
except ModuleNotFoundError:
    from tools import build_e00_three_vessel_fresh_keyed_legacy_package_v1 as legacy_package
    from tools import e00_three_vessel_clean_diag_r4_overlay_pins as pin_authority


SCHEMA = "bernini-e00-clean-diagnostic-r4-immutable-package-v4"
REVISION_TAG = "E00_DFIX2_CLEAN_DIAG_R4_OVERLAY_CACHE_CLOSURE_20260821"
DFIX2_REVISION = "online-anchor-targetowned-qk-v14r3-gradient-geometry-decodefix2-20260820"
MANIFEST_NAME = "e00-clean-diagnostic-r4-package.manifest.json"
REVIEW_MARKER = "R4_EXECUTION_REVIEW_REQUIRED.json"
PIN_AUTHORITY_FILE = "methods/bernini_action_editing/tools/e00_three_vessel_clean_diag_r4_overlay_pins.py"
CORE_PINS = {
    "methods/bernini_action_editing/anchor_sga_anc_controller.py": "1427a4908e0a4239e95a353d3406c41cb77fdb7f0be81727126a2cfd23f1f3ad",
    "methods/bernini_action_editing/anchor_qk_transport.py": "37941e30853b16fa242a7c91940620069f87a1a975d2ecf610f3cde800557a99",
    "methods/bernini_action_editing/infer_anchor_sga_anc_event_v1.py": "dd3558a4c38c5541ba6b7ad455ac599f43eb48b1b56f207a07776c9e1819145f",
    "methods/bernini_action_editing/infer_anchor_sga_anc_trained_editor_decode_v1.py": "4ed2f22df876613ecfc720a662a48f8e028eb89fe9778e491bc962a4f8f68ab1",
    "methods/bernini_action_editing/scripts/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh": "0365aacb88d976fcdc1f9bf169384f5d336bd4abe2f3c899ab6bdd502a580034",
    "methods/bernini_action_editing/guided_source_aligned_controller.py": "3e7f8e449447c8cc0f2678da82b9e298d84d0b5b9f729281a5b19369cba7ddc6",
    "methods/bernini_action_editing/differential_sampler.py": "16738e7bfa48d6b44dfc35fc395d55068e3794212baabaefa2b2876c8774916f",
    "methods/bernini_action_editing/source_aligned_controller.py": "e8601c82d1fcf7e4df11daa658b9f237e01eabc489f77a88610fcab6ad3cf4a8",
    "methods/bernini_action_editing/infer_source_aligned_controller_oracle.py": "9ae3a41e52f520f66ebcddba331b26837a5c8291426d13379eaa4c8a01a80e02",
    "methods/bernini_action_editing/infer_lora.py": "0c79faa8417a40a5735571db3a5ba828d6aa977d7d0507a5bfcb63368c07728d",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py": "bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42",
}
OVERLAY_PINS = dict(pin_authority.OVERLAY_PINS)
OVERLAY_FILES = tuple(OVERLAY_PINS) + (PIN_AUTHORITY_FILE,)
ARM_ORDER = [
    "pure_noobserver_output_routeoff",
    "observer_matched_output_routeoff",
    "old_pureqk_temporal_routeon",
]
A_LAUNCHER = "methods/bernini_action_editing/scripts/auh_launch_e00_three_vessel_clean_diag_r4_phase_a_only_node292.sh"
BC_LAUNCHER = "methods/bernini_action_editing/scripts/auh_launch_e00_three_vessel_clean_diag_r4_phase_bc_node292.sh"


class PackageError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    return legacy_package._sha256(path)


def reject_cache_bytecode(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise PackageError(f"cache scan root must be a plain directory: {root}")
    for path in root.rglob("*"):
        if path.is_dir() and path.name == "__pycache__":
            raise PackageError(f"package contains a __pycache__ directory: {path}")
        if path.is_file() and path.suffix == ".pyc":
            raise PackageError(f"package contains a .pyc file: {path}")


def _verify_overlay_pins(root: Path) -> None:
    if set(OVERLAY_PINS) != set(OVERLAY_FILES) - {PIN_AUTHORITY_FILE}:
        raise PackageError("R4 overlay pin/file closure differs")
    for relative, expected_sha in OVERLAY_PINS.items():
        path = root / relative
        try:
            legacy_package._plain_file(path, f"R4 pinned overlay {relative}")
        except legacy_package.PackageError as error:
            raise PackageError(str(error)) from error
        if _sha256(path) != expected_sha:
            raise PackageError(f"R4 overlay current bytes differ: {relative}")


def _review_marker() -> dict[str, Any]:
    return {
        "schema_version": "bernini-e00-clean-diagnostic-r4-execution-review-required-v4",
        "revision_tag": REVISION_TAG,
        "execution_authorized": False,
        "gpu_run_started": False,
        "training_performed": False,
        "independent_package_audit_passed": False,
        "required_node": "auh7-1b-gpu-292",
        "fixed_parent_job_id": "143808",
        "full_arm_order": list(ARM_ORDER),
        "authorized_launchers_only": {"phase_a_only": A_LAUNCHER, "phase_bc": BC_LAUNCHER},
        "overlay_pins_checked_at_build_and_verify": True,
        "builder_sha256_pinned_by_both_launchers": True,
        "package_cache_bytecode_forbidden": True,
        "phase_cache_scan_required": True,
        "phase_a_must_stop": True,
        "current_ab_gate_required_before_c": True,
        "final_current_receipt_closure_required": True,
        "reason": "R4 remains execution-disabled until package review and separate phase authorization.",
    }


def build_package(*, dfix2_source_tree: Path, overlay_root: Path, output: Path) -> Mapping[str, Any]:
    if output.exists() or output.is_symlink():
        raise PackageError(f"refusing to overwrite R4 package: {output}")
    reject_cache_bytecode(dfix2_source_tree)
    if not overlay_root.is_dir() or overlay_root.is_symlink():
        raise PackageError("overlay root must be a plain directory")
    try:
        legacy_package._verify_core(dfix2_source_tree, CORE_PINS)
    except legacy_package.PackageError as error:
        raise PackageError(str(error)) from error
    _verify_overlay_pins(overlay_root)
    pin_file = overlay_root / PIN_AUTHORITY_FILE
    legacy_package._plain_file(pin_file, "R4 overlay pin authority")
    source_files = legacy_package._iter_plain_files(dfix2_source_tree)
    output.mkdir(parents=True)
    try:
        for source in source_files:
            relative = source.relative_to(dfix2_source_tree)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for relative in OVERLAY_FILES:
            source = overlay_root / relative
            legacy_package._plain_file(source, f"R4 overlay {relative}")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        reject_cache_bytecode(output)
        _verify_overlay_pins(output)
        marker_path = output / REVIEW_MARKER
        marker_path.write_bytes(_canonical(_review_marker()) + b"\n")
        rows = []
        for path in legacy_package._iter_plain_files(output):
            if path.name == MANIFEST_NAME:
                continue
            relative = path.relative_to(output).as_posix()
            rows.append({"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size})
        manifest = {
            "schema_version": SCHEMA,
            "revision_tag": REVISION_TAG,
            "complete": True,
            "execution_authorized": False,
            "gpu_run_started": False,
            "training_performed": False,
            "independent_package_audit_passed": False,
            "dfix2_revision": DFIX2_REVISION,
            "dfix2_core_pins": dict(CORE_PINS),
            "overlay_files": list(OVERLAY_FILES),
            "overlay_pins": dict(OVERLAY_PINS),
            "overlay_pin_authority": {"path": PIN_AUTHORITY_FILE, "sha256": _sha256(output / PIN_AUTHORITY_FILE)},
            "authorized_entrypoints_only": [A_LAUNCHER, BC_LAUNCHER],
            "full_arm_order": list(ARM_ORDER),
            "cache_bytecode_forbidden": True,
            "review_marker_sha256": _sha256(marker_path),
            "files": rows,
            "content_digest": hashlib.sha256(_canonical(rows)).hexdigest(),
        }
        manifest_path = output / MANIFEST_NAME
        manifest_path.write_bytes(_canonical(manifest) + b"\n")
        verify_package(package_root=output, manifest_path=manifest_path)
        return manifest
    except BaseException:
        if output.exists():
            shutil.rmtree(output)
        raise


def verify_package(*, package_root: Path, manifest_path: Path) -> Mapping[str, Any]:
    reject_cache_bytecode(package_root)
    legacy_package._plain_file(manifest_path, "R4 package manifest")
    if manifest_path.parent.resolve() != package_root.resolve():
        raise PackageError("R4 manifest must live at package root")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field, expected in (
        ("schema_version", SCHEMA), ("revision_tag", REVISION_TAG), ("complete", True),
        ("execution_authorized", False), ("gpu_run_started", False),
        ("training_performed", False), ("independent_package_audit_passed", False),
        ("dfix2_revision", DFIX2_REVISION), ("dfix2_core_pins", CORE_PINS),
        ("overlay_files", list(OVERLAY_FILES)), ("overlay_pins", OVERLAY_PINS),
        ("authorized_entrypoints_only", [A_LAUNCHER, BC_LAUNCHER]),
        ("full_arm_order", ARM_ORDER), ("cache_bytecode_forbidden", True),
    ):
        if value.get(field) != expected:
            raise PackageError(f"R4 manifest {field} differs")
    pin_binding = value.get("overlay_pin_authority")
    if pin_binding != {"path": PIN_AUTHORITY_FILE, "sha256": _sha256(package_root / PIN_AUTHORITY_FILE)}:
        raise PackageError("R4 overlay pin authority binding differs")
    _verify_overlay_pins(package_root)
    rows = value.get("files")
    if not isinstance(rows, list) or not rows or value.get("content_digest") != hashlib.sha256(_canonical(rows)).hexdigest():
        raise PackageError("R4 file-row digest differs")
    expected_paths = {row.get("path") for row in rows}
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in legacy_package._iter_plain_files(package_root)
        if path.resolve() != manifest_path.resolve()
    }
    if expected_paths != actual_paths or len(expected_paths) != len(rows):
        raise PackageError("R4 package file closure differs")
    if not set(OVERLAY_FILES).issubset(expected_paths) or REVIEW_MARKER not in expected_paths:
        raise PackageError("R4 package omits a registered overlay or review marker")
    for row in rows:
        relative = row.get("path")
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise PackageError("unsafe R4 package path")
        path = package_root / relative
        legacy_package._plain_file(path, f"R4 package file {relative}")
        if row.get("sha256") != _sha256(path) or row.get("bytes") != path.stat().st_size:
            raise PackageError(f"R4 package file identity differs: {relative}")
    marker_path = package_root / REVIEW_MARKER
    if json.loads(marker_path.read_text(encoding="utf-8")) != _review_marker() or value.get("review_marker_sha256") != _sha256(marker_path):
        raise PackageError("R4 review marker differs")
    try:
        legacy_package._verify_core(package_root, CORE_PINS)
    except legacy_package.PackageError as error:
        raise PackageError(str(error)) from error
    reject_cache_bytecode(package_root)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--dfix2-source-tree", required=True); build.add_argument("--overlay-root", required=True); build.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--package-root", required=True); verify.add_argument("--manifest", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    value = (
        build_package(dfix2_source_tree=Path(args.dfix2_source_tree), overlay_root=Path(args.overlay_root), output=Path(args.output))
        if args.command == "build"
        else verify_package(package_root=Path(args.package_root), manifest_path=Path(args.manifest))
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
