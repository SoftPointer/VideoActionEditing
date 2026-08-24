#!/usr/bin/env python3
"""Build and verify the bootstrap-rooted, cache-free R5 E00 package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence

try:
    import build_e00_three_vessel_fresh_keyed_legacy_package_v1 as legacy_package
except ModuleNotFoundError:
    from tools import build_e00_three_vessel_fresh_keyed_legacy_package_v1 as legacy_package


SCHEMA = "bernini-e00-clean-diagnostic-r5-bootstrap-package-v5"
BOOTSTRAP_SCHEMA = "bernini-e00-clean-diagnostic-r5-bootstrap-root-v5"
REVISION_TAG = "E00_DFIX2_CLEAN_DIAG_R5_ACYCLIC_BOOTSTRAP_20260821"
DFIX2_REVISION = "online-anchor-targetowned-qk-v14r3-gradient-geometry-decodefix2-20260820"
MANIFEST_NAME = "e00-clean-diagnostic-r5-package.manifest.json"
REVIEW_MARKER = "R5_EXECUTION_REVIEW_REQUIRED.json"
BOOTSTRAP_ROOT_FILE = "methods/bernini_action_editing/assets/e00_three_vessel_clean_diag_r5_BOOTSTRAP_ROOT.json"
PIN_AUTHORITY_FILE = "methods/bernini_action_editing/tools/e00_three_vessel_clean_diag_r5_overlay_pins.py"
BUILDER_FILE = "methods/bernini_action_editing/tools/build_e00_three_vessel_clean_diag_r5_package.py"
VALIDATOR_FILE = "methods/bernini_action_editing/validate_e00_three_vessel_clean_diag_r5.py"
BRIDGE = "methods/bernini_action_editing/scripts/auh_e00_three_vessel_clean_diag_r5_bridge.sh"
A_LAUNCHER = "methods/bernini_action_editing/scripts/auh_launch_e00_three_vessel_clean_diag_r5_phase_a_only_node292.sh"
BC_LAUNCHER = "methods/bernini_action_editing/scripts/auh_launch_e00_three_vessel_clean_diag_r5_phase_bc_node292.sh"
ROOT_PINNED_FILES = (PIN_AUTHORITY_FILE, BUILDER_FILE, VALIDATOR_FILE)
BOOTSTRAP_CONSUMERS = (BRIDGE, A_LAUNCHER, BC_LAUNCHER)
UNPINNED_BY_AUTHORITY = {
    BOOTSTRAP_ROOT_FILE,
    PIN_AUTHORITY_FILE,
    BUILDER_FILE,
    VALIDATOR_FILE,
    *BOOTSTRAP_CONSUMERS,
}
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
ARM_ORDER = [
    "pure_noobserver_output_routeoff",
    "observer_matched_output_routeoff",
    "old_pureqk_temporal_routeon",
]
HEX = set("0123456789abcdef")


class PackageError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    return legacy_package._sha256(path)


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value):
        raise PackageError(f"{label} is not a lowercase SHA-256")
    return value


def reject_cache_bytecode(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise PackageError(f"cache scan root must be a plain directory: {root}")
    for path in root.rglob("*"):
        if path.is_dir() and path.name == "__pycache__":
            raise PackageError(f"package contains a __pycache__ directory: {path}")
        if path.is_file() and path.suffix == ".pyc":
            raise PackageError(f"package contains a .pyc file: {path}")


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise PackageError(f"{label} path is unsafe")
    return value


def validate_bootstrap_root(
    *, content_root: Path, bootstrap_path: Path, expected_bootstrap_sha256: str
) -> Mapping[str, Any]:
    _hex(expected_bootstrap_sha256, "expected bootstrap root SHA-256")
    legacy_package._plain_file(bootstrap_path, "R5 bootstrap root")
    if bootstrap_path.resolve() != (content_root / BOOTSTRAP_ROOT_FILE).resolve():
        raise PackageError("R5 bootstrap root path differs")
    if _sha256(bootstrap_path) != expected_bootstrap_sha256:
        raise PackageError("R5 bootstrap root bytes differ")
    try:
        value = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackageError("R5 bootstrap root is unreadable") from error
    expected_fields = {
        "schema_version": BOOTSTRAP_SCHEMA,
        "revision_tag": REVISION_TAG,
        "complete": True,
        "immutable": True,
        "root_does_not_pin_consumers": True,
        "authority_does_not_pin_bootstrap_files_or_consumers": True,
        "bootstrap_consumers": list(BOOTSTRAP_CONSUMERS),
    }
    for field, expected in expected_fields.items():
        if value.get(field) != expected:
            raise PackageError(f"R5 bootstrap root {field} differs")
    pins = value.get("pins")
    if not isinstance(pins, dict) or tuple(pins) != ROOT_PINNED_FILES:
        raise PackageError("R5 bootstrap root pin closure differs")
    for relative, expected_sha in pins.items():
        _safe_relative(relative, "bootstrap pin")
        _hex(expected_sha, f"bootstrap pin {relative}")
        path = content_root / relative
        legacy_package._plain_file(path, f"R5 bootstrap-pinned file {relative}")
        if _sha256(path) != expected_sha:
            raise PackageError(f"R5 bootstrap-pinned current bytes differ: {relative}")
    return value


def _load_authority(content_root: Path, bootstrap: Mapping[str, Any]) -> tuple[ModuleType, dict[str, str]]:
    path = content_root / PIN_AUTHORITY_FILE
    if _sha256(path) != bootstrap["pins"][PIN_AUTHORITY_FILE]:
        raise PackageError("R5 pin authority differs after bootstrap validation")
    spec = importlib.util.spec_from_file_location("e00_r5_trusted_overlay_pins", path)
    if spec is None or spec.loader is None:
        raise PackageError("R5 pin authority loader is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = getattr(module, "OVERLAY_PINS", None)
    if not isinstance(raw, dict) or not raw:
        raise PackageError("R5 OVERLAY_PINS is absent")
    pins: dict[str, str] = {}
    for relative, expected_sha in raw.items():
        relative = _safe_relative(relative, "overlay pin")
        _hex(expected_sha, f"overlay pin {relative}")
        if relative in pins:
            raise PackageError("R5 overlay pin is duplicated")
        pins[relative] = expected_sha
    if set(pins) & UNPINNED_BY_AUTHORITY:
        raise PackageError("R5 authority pins a bootstrap file or consumer")
    return module, pins


def _overlay_files(pins: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(pins) + (
        BOOTSTRAP_ROOT_FILE,
        PIN_AUTHORITY_FILE,
        BUILDER_FILE,
        VALIDATOR_FILE,
        BRIDGE,
        A_LAUNCHER,
        BC_LAUNCHER,
    )


def _verify_overlay_pins(root: Path, pins: Mapping[str, str]) -> None:
    for relative, expected_sha in pins.items():
        path = root / relative
        legacy_package._plain_file(path, f"R5 pinned overlay {relative}")
        if _sha256(path) != expected_sha:
            raise PackageError(f"R5 overlay current bytes differ: {relative}")


def _review_marker(*, bootstrap_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "bernini-e00-clean-diagnostic-r5-execution-review-required-v5",
        "revision_tag": REVISION_TAG,
        "execution_authorized": False,
        "gpu_run_started": False,
        "training_performed": False,
        "independent_package_audit_passed": False,
        "required_node": "auh7-1b-gpu-292",
        "fixed_parent_job_id": "143808",
        "bootstrap_root": {"path": BOOTSTRAP_ROOT_FILE, "sha256": bootstrap_sha256},
        "authorized_launchers_only": {"phase_a_only": A_LAUNCHER, "phase_bc": BC_LAUNCHER},
        "full_arm_order": list(ARM_ORDER),
        "phase_a_must_stop": True,
        "current_ab_gate_required_before_c": True,
        "final_current_receipt_closure_required": True,
        "package_cache_bytecode_forbidden": True,
        "reason": "R5 remains disabled until bootstrap/package review and separate phase authorization.",
    }


def build_package(
    *, dfix2_source_tree: Path, overlay_root: Path, output: Path,
    expected_bootstrap_root_sha256: str,
) -> Mapping[str, Any]:
    if output.exists() or output.is_symlink():
        raise PackageError(f"refusing to overwrite R5 package: {output}")
    reject_cache_bytecode(dfix2_source_tree)
    if not overlay_root.is_dir() or overlay_root.is_symlink():
        raise PackageError("overlay root must be a plain directory")
    try:
        legacy_package._verify_core(dfix2_source_tree, CORE_PINS)
    except legacy_package.PackageError as error:
        raise PackageError(str(error)) from error
    bootstrap_path = overlay_root / BOOTSTRAP_ROOT_FILE
    bootstrap = validate_bootstrap_root(
        content_root=overlay_root, bootstrap_path=bootstrap_path,
        expected_bootstrap_sha256=expected_bootstrap_root_sha256,
    )
    _, overlay_pins = _load_authority(overlay_root, bootstrap)
    overlay_files = _overlay_files(overlay_pins)
    _verify_overlay_pins(overlay_root, overlay_pins)
    for relative in overlay_files:
        legacy_package._plain_file(overlay_root / relative, f"R5 overlay {relative}")
    source_files = legacy_package._iter_plain_files(dfix2_source_tree)
    output.mkdir(parents=True)
    try:
        for source in source_files:
            relative = source.relative_to(dfix2_source_tree)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for relative in overlay_files:
            source = overlay_root / relative
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        reject_cache_bytecode(output)
        package_bootstrap = validate_bootstrap_root(
            content_root=output, bootstrap_path=output / BOOTSTRAP_ROOT_FILE,
            expected_bootstrap_sha256=expected_bootstrap_root_sha256,
        )
        _, package_pins = _load_authority(output, package_bootstrap)
        if package_pins != overlay_pins:
            raise PackageError("R5 package authority content differs")
        _verify_overlay_pins(output, package_pins)
        marker_path = output / REVIEW_MARKER
        marker_path.write_bytes(_canonical(_review_marker(bootstrap_sha256=expected_bootstrap_root_sha256)) + b"\n")
        rows = []
        for path in legacy_package._iter_plain_files(output):
            if path.name == MANIFEST_NAME:
                continue
            rows.append({
                "path": path.relative_to(output).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            })
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
            "bootstrap_root": {"path": BOOTSTRAP_ROOT_FILE, "sha256": expected_bootstrap_root_sha256},
            "bootstrap_pins": dict(bootstrap["pins"]),
            "bootstrap_consumers": list(BOOTSTRAP_CONSUMERS),
            "overlay_files": list(overlay_files),
            "overlay_pins": dict(overlay_pins),
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
        verify_package(
            package_root=output, manifest_path=manifest_path,
            expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        )
        return manifest
    except BaseException:
        if output.exists():
            shutil.rmtree(output)
        raise


def verify_package(
    *, package_root: Path, manifest_path: Path, expected_bootstrap_root_sha256: str,
) -> Mapping[str, Any]:
    reject_cache_bytecode(package_root)
    bootstrap = validate_bootstrap_root(
        content_root=package_root, bootstrap_path=package_root / BOOTSTRAP_ROOT_FILE,
        expected_bootstrap_sha256=expected_bootstrap_root_sha256,
    )
    _, overlay_pins = _load_authority(package_root, bootstrap)
    overlay_files = _overlay_files(overlay_pins)
    _verify_overlay_pins(package_root, overlay_pins)
    legacy_package._plain_file(manifest_path, "R5 package manifest")
    if manifest_path.parent.resolve() != package_root.resolve():
        raise PackageError("R5 manifest must live at package root")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = (
        ("schema_version", SCHEMA), ("revision_tag", REVISION_TAG), ("complete", True),
        ("execution_authorized", False), ("gpu_run_started", False),
        ("training_performed", False), ("independent_package_audit_passed", False),
        ("dfix2_revision", DFIX2_REVISION), ("dfix2_core_pins", CORE_PINS),
        ("bootstrap_root", {"path": BOOTSTRAP_ROOT_FILE, "sha256": expected_bootstrap_root_sha256}),
        ("bootstrap_pins", dict(bootstrap["pins"])),
        ("bootstrap_consumers", list(BOOTSTRAP_CONSUMERS)),
        ("overlay_files", list(overlay_files)), ("overlay_pins", overlay_pins),
        ("authorized_entrypoints_only", [A_LAUNCHER, BC_LAUNCHER]),
        ("full_arm_order", ARM_ORDER), ("cache_bytecode_forbidden", True),
    )
    for field, expected in expected_fields:
        if value.get(field) != expected:
            raise PackageError(f"R5 manifest {field} differs")
    authority_binding = {"path": PIN_AUTHORITY_FILE, "sha256": _sha256(package_root / PIN_AUTHORITY_FILE)}
    if value.get("overlay_pin_authority") != authority_binding:
        raise PackageError("R5 authority manifest binding differs")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows or value.get("content_digest") != hashlib.sha256(_canonical(rows)).hexdigest():
        raise PackageError("R5 file-row digest differs")
    expected_paths = {row.get("path") for row in rows}
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in legacy_package._iter_plain_files(package_root)
        if path.resolve() != manifest_path.resolve()
    }
    if expected_paths != actual_paths or len(expected_paths) != len(rows):
        raise PackageError("R5 package file closure differs")
    if not set(overlay_files).issubset(expected_paths) or REVIEW_MARKER not in expected_paths:
        raise PackageError("R5 package omits a registered file")
    for row in rows:
        relative = _safe_relative(row.get("path"), "manifest row")
        path = package_root / relative
        legacy_package._plain_file(path, f"R5 package file {relative}")
        if row.get("sha256") != _sha256(path) or row.get("bytes") != path.stat().st_size:
            raise PackageError(f"R5 package file identity differs: {relative}")
    marker_path = package_root / REVIEW_MARKER
    if json.loads(marker_path.read_text(encoding="utf-8")) != _review_marker(bootstrap_sha256=expected_bootstrap_root_sha256):
        raise PackageError("R5 review marker differs")
    if value.get("review_marker_sha256") != _sha256(marker_path):
        raise PackageError("R5 review marker binding differs")
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
    build.add_argument("--dfix2-source-tree", required=True)
    build.add_argument("--overlay-root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--expected-bootstrap-root-sha256", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--package-root", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--expected-bootstrap-root-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        value = build_package(
            dfix2_source_tree=Path(args.dfix2_source_tree), overlay_root=Path(args.overlay_root),
            output=Path(args.output), expected_bootstrap_root_sha256=args.expected_bootstrap_root_sha256,
        )
    else:
        value = verify_package(
            package_root=Path(args.package_root), manifest_path=Path(args.manifest),
            expected_bootstrap_root_sha256=args.expected_bootstrap_root_sha256,
        )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
