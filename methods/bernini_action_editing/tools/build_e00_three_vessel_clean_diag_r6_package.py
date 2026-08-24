#!/usr/bin/env python3
"""Build and verify the one-way-rooted R6 diagnostic package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "bernini-e00-clean-diagnostic-r6-external-root-package-v6"
ROOT_SCHEMA = "bernini-e00-clean-diagnostic-r6-external-bootstrap-root-v6"
REVISION_TAG = "E00_DFIX2_CLEAN_DIAG_R6_EXTERNAL_BOOTSTRAP_20260821"
DFIX2_REVISION = "online-anchor-targetowned-qk-v14r3-gradient-geometry-decodefix2-20260820"
MANIFEST_NAME = "e00-clean-diagnostic-r6-package.manifest.json"
REVIEW_MARKER = "R6_EXECUTION_REVIEW_REQUIRED.json"
ROOT_FILE = "methods/bernini_action_editing/assets/e00_three_vessel_clean_diag_r6_EXTERNAL_ROOT.json"
LEGACY_HELPER_FILE = "methods/bernini_action_editing/tools/build_e00_three_vessel_fresh_keyed_legacy_package_v1.py"
BRIDGE = "methods/bernini_action_editing/scripts/auh_e00_three_vessel_clean_diag_r6_bridge.sh"
A_LAUNCHER = "methods/bernini_action_editing/scripts/auh_launch_e00_three_vessel_clean_diag_r6_phase_a_only_node292.sh"
BC_LAUNCHER = "methods/bernini_action_editing/scripts/auh_launch_e00_three_vessel_clean_diag_r6_phase_bc_node292.sh"
RUNTIME_CONSUMERS = [BRIDGE, A_LAUNCHER, BC_LAUNCHER]
HEX = set("0123456789abcdef")


class PackageError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value):
        raise PackageError(f"{label} is not a lowercase SHA-256")
    return value


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise PackageError(f"{label} path is unsafe")
    return value


def _plain_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise PackageError(f"{label} is not a plain file: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_cache_bytecode(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise PackageError(f"cache scan root must be a plain directory: {root}")
    for path in root.rglob("*"):
        if path.is_dir() and path.name == "__pycache__":
            raise PackageError(f"package contains a __pycache__ directory: {path}")
        if path.is_file() and path.suffix == ".pyc":
            raise PackageError(f"package contains a .pyc file: {path}")


def validate_root(
    *, content_root: Path, root_path: Path, expected_root_sha256: str,
    reject_unpinned_cache_bytecode: bool = True,
) -> Mapping[str, Any]:
    _hex(expected_root_sha256, "expected R6 root SHA-256")
    _plain_file(root_path, "R6 root")
    if root_path.resolve() != (content_root / ROOT_FILE).resolve():
        raise PackageError("R6 root path differs")
    if _sha256(root_path) != expected_root_sha256:
        raise PackageError("R6 root bytes differ")
    try:
        value = json.loads(root_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackageError("R6 root is unreadable") from error
    for field, expected in (
        ("schema_version", ROOT_SCHEMA), ("revision_tag", REVISION_TAG),
        ("complete", True), ("immutable", True), ("one_way_root_pins", True),
        ("consumers_pin_root", False), ("runtime_diagnostic_only", True),
        ("property_preservation_fix_claimed", False),
        ("dfix2_revision", DFIX2_REVISION), ("runtime_consumers", RUNTIME_CONSUMERS),
    ):
        if value.get(field) != expected:
            raise PackageError(f"R6 root {field} differs")
    pins = value.get("pins"); core_paths = value.get("core_paths")
    if not isinstance(pins, dict) or not pins or value.get("pinned_file_count") != len(pins):
        raise PackageError("R6 root pin closure differs")
    if not isinstance(core_paths, list) or not core_paths or any(path not in pins for path in core_paths):
        raise PackageError("R6 root core path closure differs")
    if any(path not in pins for path in RUNTIME_CONSUMERS) or LEGACY_HELPER_FILE not in pins:
        raise PackageError("R6 root omits a runtime consumer or legacy helper")
    # A package must be globally cache-free.  The local overlay repository is
    # only a build input and may contain unrelated artifacts outside this
    # root's pin closure, so at that boundary we verify every pinned byte but
    # defer the global scan until the fresh package has been materialized.
    if reject_unpinned_cache_bytecode:
        reject_cache_bytecode(content_root)
    for relative, expected_sha in pins.items():
        relative = _safe_relative(relative, "R6 root pin")
        _hex(expected_sha, f"R6 root pin {relative}")
        path = content_root / relative
        _plain_file(path, f"R6 pinned file {relative}")
        if _sha256(path) != expected_sha:
            raise PackageError(f"R6 pinned current bytes differ: {relative}")
    if reject_unpinned_cache_bytecode:
        reject_cache_bytecode(content_root)
    return value


def _load_legacy_helper(content_root: Path, root: Mapping[str, Any]) -> ModuleType:
    expected_sha = root["pins"][LEGACY_HELPER_FILE]
    helper_path = content_root / LEGACY_HELPER_FILE
    _plain_file(helper_path, "R6 legacy package helper")
    if _sha256(helper_path) != expected_sha:
        raise PackageError("R6 legacy helper bytes differ before lazy import")
    spec = importlib.util.spec_from_file_location("e00_r6_verified_legacy_package_helper", helper_path)
    if spec is None or spec.loader is None:
        raise PackageError("R6 legacy helper loader is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _review_marker(root_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "bernini-e00-clean-diagnostic-r6-execution-review-required-v6",
        "revision_tag": REVISION_TAG,
        "execution_authorized": False,
        "gpu_run_started": False,
        "training_performed": False,
        "independent_package_audit_passed": False,
        "root": {"path": ROOT_FILE, "sha256": root_sha256},
        "authorized_launchers_only": {"phase_a_only": A_LAUNCHER, "phase_bc": BC_LAUNCHER},
        "runtime_consumers": list(RUNTIME_CONSUMERS),
        "old_qk_route_white_leakage_diagnostic_only": True,
        "property_preservation_fix_claimed": False,
        "a_b_bit_exact_or_stop_without_c": True,
        "only_independent_a_phase_review_may_follow": True,
        "reason": "R6 remains disabled pending independent package and phase-A review.",
    }


def build_package(
    *, dfix2_source_tree: Path, overlay_root: Path, output: Path,
    expected_root_sha256: str,
) -> Mapping[str, Any]:
    if output.exists() or output.is_symlink():
        raise PackageError(f"refusing to overwrite R6 package: {output}")
    if not overlay_root.is_dir() or overlay_root.is_symlink():
        raise PackageError("R6 overlay root is not a plain directory")
    reject_cache_bytecode(dfix2_source_tree)
    root = validate_root(
        content_root=overlay_root, root_path=overlay_root / ROOT_FILE,
        expected_root_sha256=expected_root_sha256,
        reject_unpinned_cache_bytecode=False,
    )
    legacy = _load_legacy_helper(overlay_root, root)
    for relative in root["core_paths"]:
        source = dfix2_source_tree / relative
        legacy._plain_file(source, f"R6 dfix2 core {relative}")
        if _sha256(source) != root["pins"][relative]:
            raise PackageError(f"R6 dfix2 core bytes differ: {relative}")
    source_files = legacy._iter_plain_files(dfix2_source_tree)
    overlay_files = list(root["pins"]) + [ROOT_FILE]
    output.mkdir(parents=True)
    try:
        for source in source_files:
            relative = source.relative_to(dfix2_source_tree)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for relative in overlay_files:
            source = overlay_root / relative
            _plain_file(source, f"R6 overlay {relative}")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        reject_cache_bytecode(output)
        package_root_value = validate_root(
            content_root=output, root_path=output / ROOT_FILE,
            expected_root_sha256=expected_root_sha256,
        )
        _load_legacy_helper(output, package_root_value)
        marker_path = output / REVIEW_MARKER
        marker_path.write_bytes(_canonical(_review_marker(expected_root_sha256)) + b"\n")
        rows = []
        for path in legacy._iter_plain_files(output):
            if path.name == MANIFEST_NAME:
                continue
            rows.append({"path": path.relative_to(output).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size})
        manifest = {
            "schema_version": SCHEMA,
            "revision_tag": REVISION_TAG,
            "complete": True,
            "execution_authorized": False,
            "gpu_run_started": False,
            "training_performed": False,
            "independent_package_audit_passed": False,
            "root": {"path": ROOT_FILE, "sha256": expected_root_sha256},
            "root_pins": dict(root["pins"]),
            "pinned_file_count": len(root["pins"]),
            "runtime_consumers": list(RUNTIME_CONSUMERS),
            "dfix2_revision": DFIX2_REVISION,
            "core_paths": list(root["core_paths"]),
            "overlay_files": overlay_files,
            "cache_bytecode_forbidden": True,
            "old_qk_route_white_leakage_diagnostic_only": True,
            "property_preservation_fix_claimed": False,
            "authorized_entrypoints_only": [A_LAUNCHER, BC_LAUNCHER],
            "review_marker_sha256": _sha256(marker_path),
            "files": rows,
            "content_digest": hashlib.sha256(_canonical(rows)).hexdigest(),
        }
        manifest_path = output / MANIFEST_NAME
        manifest_path.write_bytes(_canonical(manifest) + b"\n")
        verify_package(
            package_root=output, manifest_path=manifest_path,
            expected_root_sha256=expected_root_sha256,
        )
        return manifest
    except BaseException:
        if output.exists():
            shutil.rmtree(output)
        raise


def verify_package(
    *, package_root: Path, manifest_path: Path, expected_root_sha256: str,
) -> Mapping[str, Any]:
    reject_cache_bytecode(package_root)
    root = validate_root(
        content_root=package_root, root_path=package_root / ROOT_FILE,
        expected_root_sha256=expected_root_sha256,
    )
    legacy = _load_legacy_helper(package_root, root)
    legacy._plain_file(manifest_path, "R6 package manifest")
    if manifest_path.parent.resolve() != package_root.resolve():
        raise PackageError("R6 manifest must live at package root")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    overlay_files = list(root["pins"]) + [ROOT_FILE]
    for field, expected in (
        ("schema_version", SCHEMA), ("revision_tag", REVISION_TAG), ("complete", True),
        ("execution_authorized", False), ("gpu_run_started", False),
        ("training_performed", False), ("independent_package_audit_passed", False),
        ("root", {"path": ROOT_FILE, "sha256": expected_root_sha256}),
        ("root_pins", dict(root["pins"])), ("pinned_file_count", len(root["pins"])),
        ("runtime_consumers", RUNTIME_CONSUMERS), ("dfix2_revision", DFIX2_REVISION),
        ("core_paths", list(root["core_paths"])), ("overlay_files", overlay_files),
        ("cache_bytecode_forbidden", True),
        ("old_qk_route_white_leakage_diagnostic_only", True),
        ("property_preservation_fix_claimed", False),
        ("authorized_entrypoints_only", [A_LAUNCHER, BC_LAUNCHER]),
    ):
        if value.get(field) != expected:
            raise PackageError(f"R6 manifest {field} differs")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows or value.get("content_digest") != hashlib.sha256(_canonical(rows)).hexdigest():
        raise PackageError("R6 manifest file-row digest differs")
    expected_paths = {row.get("path") for row in rows}
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in legacy._iter_plain_files(package_root)
        if path.resolve() != manifest_path.resolve()
    }
    if expected_paths != actual_paths or len(expected_paths) != len(rows):
        raise PackageError("R6 package file closure differs")
    if not set(overlay_files).issubset(expected_paths) or REVIEW_MARKER not in expected_paths:
        raise PackageError("R6 package omits a rooted file")
    for row in rows:
        relative = _safe_relative(row.get("path"), "R6 manifest row")
        path = package_root / relative
        legacy._plain_file(path, f"R6 package file {relative}")
        if row.get("sha256") != _sha256(path) or row.get("bytes") != path.stat().st_size:
            raise PackageError(f"R6 package file identity differs: {relative}")
    marker_path = package_root / REVIEW_MARKER
    if json.loads(marker_path.read_text(encoding="utf-8")) != _review_marker(expected_root_sha256):
        raise PackageError("R6 review marker differs")
    if value.get("review_marker_sha256") != _sha256(marker_path):
        raise PackageError("R6 review marker binding differs")
    reject_cache_bytecode(package_root)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--dfix2-source-tree", required=True); build.add_argument("--overlay-root", required=True)
    build.add_argument("--output", required=True); build.add_argument("--expected-root-sha256", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--package-root", required=True); verify.add_argument("--manifest", required=True)
    verify.add_argument("--expected-root-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        value = build_package(
            dfix2_source_tree=Path(args.dfix2_source_tree), overlay_root=Path(args.overlay_root),
            output=Path(args.output), expected_root_sha256=args.expected_root_sha256,
        )
    else:
        value = verify_package(
            package_root=Path(args.package_root), manifest_path=Path(args.manifest),
            expected_root_sha256=args.expected_root_sha256,
        )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
