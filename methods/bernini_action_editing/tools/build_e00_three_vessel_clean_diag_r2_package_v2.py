#!/usr/bin/env python3
"""Build/verify the immutable, two-phase E00 clean diagnostic R2 package.

The emitted review marker names the complete A -> B -> C order and the hard
stop/gate edges.  Package construction never authorizes execution, invokes
Slurm, starts a GPU process, or edits the pinned legacy model sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Optional, Sequence

try:
    import build_e00_three_vessel_fresh_keyed_legacy_package_v1 as legacy_package
except ModuleNotFoundError:  # Imported as ``tools.<module>`` by unit tests.
    from tools import build_e00_three_vessel_fresh_keyed_legacy_package_v1 as legacy_package


SCHEMA = "bernini-e00-clean-diagnostic-r2-immutable-package-v2"
REVISION_TAG = "E00_DFIX2_CLEAN_DIAG_R2_FIXED_RNG_TWO_PHASE_20260821"
MANIFEST_NAME = "e00-clean-diagnostic-r2-package.manifest.json"
REVIEW_MARKER = "EXECUTION_REVIEW_REQUIRED.json"
ARM_ORDER = [
    "pure_noobserver_output_routeoff",
    "observer_matched_output_routeoff",
    "old_pureqk_temporal_routeon",
]
A_LAUNCHER = (
    "methods/bernini_action_editing/scripts/"
    "auh_launch_e00_three_vessel_clean_diag_r2_phase_a_only_node292.sh"
)
BC_LAUNCHER = (
    "methods/bernini_action_editing/scripts/"
    "auh_launch_e00_three_vessel_clean_diag_r2_phase_bc_node292.sh"
)
OVERLAY_FILES = (
    "methods/bernini_action_editing/assets/e00_three_vessel_clean_matched_route_probe_v1.json",
    "methods/bernini_action_editing/assets/e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.json",
    "methods/bernini_action_editing/assets/e00_source_frame0_static81_25fps_704x1056_v1.mp4",
    "methods/bernini_action_editing/assets/e00_three_vessel_fresh_keyed_two_phase_diagnostic_v2.json",
    "methods/bernini_action_editing/assets/e00_three_vessel_clean_diag_r2_phase_a_authorization.template.json",
    "methods/bernini_action_editing/assets/e00_three_vessel_clean_diag_r2_phase_bc_authorization.template.json",
    "methods/bernini_action_editing/e00_legacy_infer_fork_rng_wrapper_v1.py",
    "methods/bernini_action_editing/validate_e00_three_vessel_fresh_keyed_legacy_diagnostic_v1.py",
    "methods/bernini_action_editing/e00_legacy_infer_fixed_rng_wrapper_v2.py",
    "methods/bernini_action_editing/validate_e00_three_vessel_fresh_keyed_two_phase_diagnostic_v2.py",
    "methods/bernini_action_editing/tools/build_e00_three_vessel_fresh_keyed_legacy_package_v1.py",
    "methods/bernini_action_editing/tools/build_e00_three_vessel_clean_diag_r2_package_v2.py",
    "methods/bernini_action_editing/scripts/auh_e00_three_vessel_clean_diag_r2_bridge.sh",
    A_LAUNCHER,
    BC_LAUNCHER,
    "methods/bernini_action_editing/tests/test_e00_three_vessel_clean_diag_r2_two_phase_v2.py",
)


class PackageError(RuntimeError):
    pass


def _review_marker() -> dict[str, Any]:
    return {
        "schema_version": (
            "bernini-e00-clean-diagnostic-r2-execution-review-required-v2"
        ),
        "revision_tag": REVISION_TAG,
        "execution_authorized": False,
        "gpu_run_started": False,
        "training_performed": False,
        "independent_package_audit_passed": False,
        "required_node": "auh7-1b-gpu-292",
        "parent_job_id": "143808",
        "full_arm_order": list(ARM_ORDER),
        "authorized_launchers_only": {
            "phase_a_only": A_LAUNCHER,
            "phase_bc": BC_LAUNCHER,
        },
        "required_state_machine": [
            "external_A_only_authorization",
            "run_A",
            "hard_stop_after_A_for_review",
            "external_BC_authorization_bound_to_A_marker_audit_and_mp4_bytes",
            "run_B",
            "fresh_A_B_latent_mp4_rng_noise_schedule_frozen_bit_exact_gate",
            "run_C_only_if_gate_passes",
            "final_ABC_audit",
        ],
        "separate_a_and_bc_authorizations_required": True,
        "phase_a_must_stop": True,
        "ab_predecode_latent_bit_exact_required_before_c": True,
        "ab_mp4_bit_exact_required_before_c": True,
        "same_per_rank_explicit_cpu_cuda_initial_rng_required_all_arms": True,
        "on_ab_gate_failure": "STOP_WITHOUT_C",
        "reason": (
            "Independent review must precede phase A; phase A then stops.  A second "
            "byte-bound review must precede B, and C remains unreachable until the "
            "fresh A/B bit-exact gate passes."
        ),
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return legacy_package._sha256(path)


def _verify_review_marker(path: Path) -> Mapping[str, Any]:
    legacy_package._plain_file(path, "R2 review marker")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = _review_marker()
    if value != expected:
        raise PackageError("R2 review marker or complete A -> B -> C state machine differs")
    return value


def build_package(
    *,
    dfix2_source_tree: Path,
    overlay_root: Path,
    output: Path,
    expected_core: Mapping[str, str] = legacy_package.CORE_PINS,
    overlay_files: Sequence[str] = OVERLAY_FILES,
) -> Mapping[str, Any]:
    if output.exists() or output.is_symlink():
        raise PackageError(f"refusing to overwrite package output: {output}")
    if not dfix2_source_tree.is_dir() or dfix2_source_tree.is_symlink():
        raise PackageError("dfix2 source tree must be a plain directory")
    if not overlay_root.is_dir() or overlay_root.is_symlink():
        raise PackageError("overlay root must be a plain directory")
    try:
        legacy_package._verify_core(dfix2_source_tree, expected_core)
    except legacy_package.PackageError as error:
        raise PackageError(str(error)) from error
    source_files = legacy_package._iter_plain_files(dfix2_source_tree)
    overlay_paths = []
    for relative in overlay_files:
        path = overlay_root / relative
        try:
            legacy_package._plain_file(path, f"overlay {relative}")
        except legacy_package.PackageError as error:
            raise PackageError(str(error)) from error
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
        marker_path = output / REVIEW_MARKER
        marker_path.write_bytes(_canonical(_review_marker()) + b"\n")

        rows = []
        for path in legacy_package._iter_plain_files(output):
            if path.name == MANIFEST_NAME:
                continue
            relative = path.relative_to(output).as_posix()
            rows.append(
                {
                    "path": relative,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        manifest = {
            "schema_version": SCHEMA,
            "revision_tag": REVISION_TAG,
            "complete": True,
            "execution_authorized": False,
            "gpu_run_started": False,
            "training_performed": False,
            "independent_package_audit_passed": False,
            "dfix2_revision": (
                "online-anchor-targetowned-qk-v14r3-gradient-geometry-decodefix2-20260820"
            ),
            "dfix2_core_pins": dict(expected_core),
            "overlay_files": list(overlay_files),
            "authorized_entrypoints_only": [A_LAUNCHER, BC_LAUNCHER],
            "full_arm_order": list(ARM_ORDER),
            "phase_a_hard_stop_required": True,
            "ab_bit_exact_gate_required_before_c": True,
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
    if not package_root.is_dir() or package_root.is_symlink():
        raise PackageError("package root must be a plain directory")
    try:
        legacy_package._plain_file(manifest_path, "R2 package manifest")
    except legacy_package.PackageError as error:
        raise PackageError(str(error)) from error
    if manifest_path.parent.resolve() != package_root.resolve():
        raise PackageError("package manifest must live at package root")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field, expected in (
        ("schema_version", SCHEMA),
        ("revision_tag", REVISION_TAG),
        ("complete", True),
        ("execution_authorized", False),
        ("gpu_run_started", False),
        ("training_performed", False),
        ("independent_package_audit_passed", False),
        ("authorized_entrypoints_only", [A_LAUNCHER, BC_LAUNCHER]),
        ("full_arm_order", ARM_ORDER),
        ("phase_a_hard_stop_required", True),
        ("ab_bit_exact_gate_required_before_c", True),
    ):
        if value.get(field) != expected:
            raise PackageError(f"R2 package manifest {field} differs")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise PackageError("R2 package manifest has no files")
    if value.get("content_digest") != hashlib.sha256(_canonical(rows)).hexdigest():
        raise PackageError("R2 package content digest differs")
    expected_paths = {row.get("path") for row in rows}
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in legacy_package._iter_plain_files(package_root)
        if path.resolve() != manifest_path.resolve()
    }
    if expected_paths != actual_paths or len(expected_paths) != len(rows):
        raise PackageError("R2 package file closure differs")
    if not {A_LAUNCHER, BC_LAUNCHER, REVIEW_MARKER}.issubset(expected_paths):
        raise PackageError("R2 package omits an authorized launcher or review marker")
    for row in rows:
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            raise PackageError("unsafe R2 package manifest path")
        path = package_root / relative
        try:
            legacy_package._plain_file(path, f"R2 package file {relative}")
        except legacy_package.PackageError as error:
            raise PackageError(str(error)) from error
        if row.get("sha256") != _sha256(path) or row.get("bytes") != path.stat().st_size:
            raise PackageError(f"R2 package file identity differs: {relative}")
    marker_path = package_root / REVIEW_MARKER
    _verify_review_marker(marker_path)
    if value.get("review_marker_sha256") != _sha256(marker_path):
        raise PackageError("R2 review marker SHA-256 binding differs")
    try:
        legacy_package._verify_core(package_root, value.get("dfix2_core_pins", {}))
    except legacy_package.PackageError as error:
        raise PackageError(str(error)) from error
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
