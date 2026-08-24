#!/usr/bin/env python3
"""Build/verify an immutable dfix2 + E00 diagnostic package.

Building copies a plain-file dfix2 source tree, overlays only the registered
E00 closure files, hashes every package byte, and leaves execution review
required.  It never invokes Torch, Slurm, SSH, a renderer, or a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "bernini-e00-fresh-keyed-legacy-immutable-package-v1"
MANIFEST_NAME = "e00-fresh-keyed-legacy-package.manifest.json"
REVIEW_MARKER = "EXECUTION_REVIEW_REQUIRED.json"
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
OVERLAY_FILES = (
    "methods/bernini_action_editing/assets/e00_three_vessel_clean_matched_route_probe_v1.json",
    "methods/bernini_action_editing/assets/e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.json",
    "methods/bernini_action_editing/assets/e00_source_frame0_static81_25fps_704x1056_v1.mp4",
    "methods/bernini_action_editing/assets/e00_three_vessel_execution_authorization_v1.template.json",
    "methods/bernini_action_editing/e00_three_vessel_clean_matched_route_probe_spec_v1.py",
    "methods/bernini_action_editing/e00_legacy_infer_fork_rng_wrapper_v1.py",
    "methods/bernini_action_editing/validate_e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.py",
    "methods/bernini_action_editing/tools/build_e00_three_vessel_fresh_keyed_legacy_package_v1.py",
    "methods/bernini_action_editing/scripts/auh_e00_three_vessel_fresh_keyed_legacy_bridge_v1.sh",
    "methods/bernini_action_editing/scripts/auh_launch_e00_three_vessel_fresh_keyed_legacy_diag_node292_v1.sh",
    "methods/bernini_action_editing/tests/test_e00_three_vessel_clean_matched_route_probe_spec_v1.py",
    "methods/bernini_action_editing/tests/test_e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.py",
)


class PackageError(RuntimeError):
    pass


def _plain_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise PackageError(f"{label} is not a plain file: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _iter_plain_files(root: Path) -> list[Path]:
    rows: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {"__pycache__", ".git"} for part in relative.parts) or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise PackageError(f"package input contains a symlink: {path}")
        if path.is_file():
            _plain_file(path, "package input")
            rows.append(path)
        elif not path.is_dir():
            raise PackageError(f"package input contains a non-file: {path}")
    return rows


def _verify_core(root: Path, expected_core: Mapping[str, str]) -> None:
    for relative, expected in expected_core.items():
        path = root / relative
        _plain_file(path, f"dfix2 core {relative}")
        if _sha256(path) != expected:
            raise PackageError(f"dfix2 core SHA-256 differs: {relative}")


def build_package(
    *,
    dfix2_source_tree: Path,
    overlay_root: Path,
    output: Path,
    expected_core: Mapping[str, str] = CORE_PINS,
    overlay_files: Sequence[str] = OVERLAY_FILES,
) -> Mapping[str, Any]:
    if output.exists() or output.is_symlink():
        raise PackageError(f"refusing to overwrite package output: {output}")
    if not dfix2_source_tree.is_dir() or dfix2_source_tree.is_symlink():
        raise PackageError("dfix2 source tree must be a plain directory")
    if not overlay_root.is_dir() or overlay_root.is_symlink():
        raise PackageError("overlay root must be a plain directory")
    _verify_core(dfix2_source_tree, expected_core)
    source_files = _iter_plain_files(dfix2_source_tree)
    overlay_paths = []
    for relative in overlay_files:
        path = overlay_root / relative
        _plain_file(path, f"overlay {relative}")
        overlay_paths.append((relative, path))

    output.mkdir(parents=True)
    try:
        for source in source_files:
            relative = source.relative_to(dfix2_source_tree)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for relative, source in overlay_paths:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        review = {
            "schema_version": "bernini-e00-execution-review-required-v1",
            "execution_authorized": False,
            "gpu_run_started": False,
            "required_node": "auh7-1b-gpu-292",
            "required_serial_order": list(("observer_matched_output_routeoff", "old_pureqk_temporal_routeon")),
            "reason": "User/parent review is required after immutable package construction; node292 remains reserved for SP4 observer until separately authorized.",
        }
        (output / REVIEW_MARKER).write_bytes(_canonical(review) + b"\n")
        rows = []
        for path in _iter_plain_files(output):
            if path.name == MANIFEST_NAME:
                continue
            relative = path.relative_to(output).as_posix()
            rows.append({"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size})
        manifest = {
            "schema_version": SCHEMA,
            "complete": True,
            "execution_authorized": False,
            "dfix2_revision": "online-anchor-targetowned-qk-v14r3-gradient-geometry-decodefix2-20260820",
            "dfix2_core_pins": dict(expected_core),
            "overlay_files": list(overlay_files),
            "files": rows,
            "content_digest": hashlib.sha256(_canonical(rows)).hexdigest(),
        }
        manifest_path = output / MANIFEST_NAME
        manifest_path.write_bytes(_canonical(manifest) + b"\n")
        verify_package(package_root=output, manifest_path=manifest_path)
        return manifest
    except BaseException:
        # A failed draft is never considered a package.  Refuse to guess at a
        # partial recovery; the caller chose a fresh, non-existing output.
        if output.exists():
            shutil.rmtree(output)
        raise


def verify_package(*, package_root: Path, manifest_path: Path) -> Mapping[str, Any]:
    if not package_root.is_dir() or package_root.is_symlink():
        raise PackageError("package root must be a plain directory")
    _plain_file(manifest_path, "package manifest")
    if manifest_path.parent.resolve() != package_root.resolve():
        raise PackageError("package manifest must live at package root")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA or value.get("complete") is not True:
        raise PackageError("package manifest schema/completion differs")
    if value.get("execution_authorized") is not False:
        raise PackageError("immutable draft must remain execution-unauthorized")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise PackageError("package manifest has no files")
    if value.get("content_digest") != hashlib.sha256(_canonical(rows)).hexdigest():
        raise PackageError("package content digest differs")
    expected_paths = {row.get("path") for row in rows}
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in _iter_plain_files(package_root)
        if path.resolve() != manifest_path.resolve()
    }
    if expected_paths != actual_paths or len(expected_paths) != len(rows):
        raise PackageError("package file closure differs")
    for row in rows:
        relative = row.get("path")
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise PackageError("unsafe package manifest path")
        path = package_root / relative
        _plain_file(path, f"package file {relative}")
        if row.get("sha256") != _sha256(path) or row.get("bytes") != path.stat().st_size:
            raise PackageError(f"package file identity differs: {relative}")
    _verify_core(package_root, value.get("dfix2_core_pins", {}))
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--dfix2-source-tree", required=True)
    build.add_argument("--overlay-root", required=True)
    build.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--package-root", required=True)
    verify.add_argument("--manifest", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        value = build_package(
            dfix2_source_tree=Path(args.dfix2_source_tree),
            overlay_root=Path(args.overlay_root),
            output=Path(args.output),
        )
    else:
        value = verify_package(
            package_root=Path(args.package_root), manifest_path=Path(args.manifest)
        )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
