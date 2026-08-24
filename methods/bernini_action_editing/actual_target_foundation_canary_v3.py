#!/usr/bin/env python3
"""Fail-closed V3 authority and non-compensable development-canary gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, NoReturn, Optional, Sequence

import actual_target_foundation_canary_v1 as base


METHOD_ROOT = Path(os.path.abspath(__file__)).parent
AUTHORITY_PATH = METHOD_ROOT / "assets" / "actual_target_foundation_canary_authority_v3.json"
SCHEMA_VERSION = "actual-target-foundation-canary-v3"
EXPERIMENT_ID = "actual_target_foundation_canary_v3"
PHASES = 8
FAMILIES = base.FAMILIES
INPUT_CONTROLS = base.INPUT_CONTROLS
BRANCHES = base.BRANCHES
REAL_GPU_LAUNCH_AUTHORIZED = True

BRANCH_FIELDS = {
    "frozen_base": {
        "all_models_eval_frozen",
        "source_and_weight_closure_unchanged",
        "parameter_updates",
        "generator_forward_calls",
        "actual_forward_hook_delta",
        "full_model_closure_deferred_to_run_receipt",
    },
    "node": {
        "dustbin_used",
        "unbalanced_phase_pair_count",
        "dustbin_unmatched_count",
        "dustbin_transport_mass",
        "forced_nonempty_slot_used",
        "anonymous_slot_relabel_invariant",
        "phase_cardinalities",
        "mechanically_valid_phases",
        "positive_similarity",
        "input_margins",
        "mask_descriptor_binding_break_margin",
    },
    "track": {
        "assigned_track_count",
        "assigned_point_count",
        "minimum_same_track_member_phases_observed",
        "visible_and_member_fraction",
        "per_phase_visible_member_counts",
        "ambiguous_overlap_observation_count",
        "out_of_bounds_observation_count",
        "nonfinite_observation_count",
        "vote_tie_abstain_count",
        "insufficient_membership_abstain_count",
        "state_counts",
        "lifecycle_counts",
        "dynamic_nonentry_lifecycle_observed",
        "valid_adjacent_velocity_count",
        "positive_similarity",
        "input_margins",
        "cross_phase_track_identity_break_margin",
    },
    "edge": {
        "per_phase_active_counts",
        "per_phase_birth_counts",
        "per_phase_persist_counts",
        "per_phase_death_counts",
        "per_phase_valid_velocity_counts",
        "per_phase_qualified_lifecycle_counts",
        "evaluated_pairwise_edge_count",
        "real_per_phase_lifecycle_channels",
        "positive_similarity",
        "input_margins",
        "drop_edge_margin",
        "drop_edge_removed_count",
        "drop_edge_control_norm",
        "drop_edge_control_similarity",
        "drop_edge_positive_l2_distance",
    },
    "ordered_phase": {"input_margins"},
}


class CanaryV3Error(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise CanaryV3Error(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise CanaryV3Error("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> NoReturn:
    fail(f"nonfinite JSON constant is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes) -> Any:
    try:
        text = payload.decode("ascii", errors="strict")
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except CanaryV3Error:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CanaryV3Error("strict finite ASCII JSON parse failed") from error


def _plain_file(path: Path) -> None:
    if not path.is_absolute():
        fail(f"path is not absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            row = current.lstat()
        except OSError as error:
            raise CanaryV3Error(f"path component is unavailable: {current}") from error
        if stat.S_ISLNK(row.st_mode):
            fail(f"symlink component is forbidden: {current}")
    row = path.lstat()
    if not stat.S_ISREG(row.st_mode):
        fail(f"path is not a regular file: {path}")


def _plain_directory(path: Path) -> None:
    if not path.is_absolute():
        fail(f"path is not absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            row = current.lstat()
        except OSError as error:
            raise CanaryV3Error(f"path component is unavailable: {current}") from error
        if stat.S_ISLNK(row.st_mode):
            fail(f"symlink component is forbidden: {current}")
    if not stat.S_ISDIR(path.lstat().st_mode):
        fail(f"path is not a directory: {path}")


def stable_file_bytes(path: Path) -> bytes:
    _plain_file(path)
    before = path.stat()
    with path.open("rb") as handle:
        payload = handle.read()
        inside = os.fstat(handle.fileno())
    after = path.stat()
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    if identity(before) != identity(inside) or identity(before) != identity(after):
        fail(f"file changed during stable read: {path}")
    return payload


def _source_tree_rows(root: Path, suffix: str) -> list[Mapping[str, Any]]:
    _plain_directory(root)
    rows = []
    for current_text, directories, files in os.walk(root, followlinks=False):
        current = Path(current_text)
        current_row = current.lstat()
        if stat.S_ISLNK(current_row.st_mode) or not stat.S_ISDIR(current_row.st_mode):
            fail(f"source tree directory differs: {current}")
        for name in directories:
            child = current / name
            child_row = child.lstat()
            if stat.S_ISLNK(child_row.st_mode) or not stat.S_ISDIR(child_row.st_mode):
                fail(f"source tree contains symlink/non-directory: {child}")
        for name in files:
            child = current / name
            child_row = child.lstat()
            if stat.S_ISLNK(child_row.st_mode) or not stat.S_ISREG(child_row.st_mode):
                fail(f"source tree contains symlink/non-file: {child}")
            if child.suffix != suffix:
                continue
            payload = stable_file_bytes(child)
            rows.append(
                {
                    "relative_path": str(child.relative_to(root)),
                    "byte_count": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "mode": stat.S_IMODE(child_row.st_mode),
                }
            )
    return sorted(rows, key=lambda row: row["relative_path"])


def foundation_source_tree_closure() -> Mapping[str, Any]:
    specs = load_authority()["foundation_source_tree_authority"]
    if not isinstance(specs, list) or len(specs) != 3:
        fail("foundation source tree authority differs")
    output = []
    seen_roles = set()
    for spec in specs:
        if not isinstance(spec, Mapping) or set(spec) != {
            "role",
            "root",
            "suffix",
            "file_count",
            "manifest_sha256",
        }:
            fail("foundation source tree authority row differs")
        if spec["role"] in seen_roles or spec["suffix"] != ".py":
            fail("foundation source tree role/suffix differs")
        seen_roles.add(spec["role"])
        root = Path(spec["root"])
        rows = _source_tree_rows(root, spec["suffix"])
        if rows != _source_tree_rows(root, spec["suffix"]):
            fail(f"foundation source tree changed during double scan: {spec['role']}")
        digest = object_sha256(rows)
        if len(rows) != spec["file_count"] or digest != spec["manifest_sha256"]:
            fail(f"foundation source tree manifest differs: {spec['role']}")
        output.append(
            {
                "role": spec["role"],
                "root": str(root),
                "suffix": spec["suffix"],
                "file_count": len(rows),
                "manifest_sha256": digest,
                "files": rows,
                "double_scan_stable": True,
                "no_symlinks": True,
            }
        )
    value = {"verified": True, "trees": output}
    return {**value, "digest": object_sha256(value)}


def _validate_legacy_tree_authority(spec: Any, label: str) -> None:
    if not isinstance(spec, Mapping) or not {
        "root",
        "rows",
        "canonical_tree_digest",
    }.issubset(spec):
        fail(f"{label} tree authority is incomplete")
    root = spec.get("root")
    rows = spec.get("rows")
    if (
        not isinstance(root, str)
        or not Path(root).is_absolute()
        or not isinstance(rows, list)
        or not rows
    ):
        fail(f"{label} tree authority root/rows differ")
    paths = []
    for row in rows:
        if not isinstance(row, Mapping):
            fail(f"{label} tree row is not an object")
        relative = row.get("relative_path")
        kind = row.get("kind")
        if (
            not isinstance(relative, str)
            or (
                relative != "."
                and (
                    Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                )
            )
            or kind not in {"directory", "file"}
            or not isinstance(row.get("device"), int)
            or isinstance(row.get("device"), bool)
            or not isinstance(row.get("inode"), int)
            or isinstance(row.get("inode"), bool)
            or row.get("inode", 0) <= 0
            or not isinstance(row.get("mode"), int)
            or isinstance(row.get("mode"), bool)
        ):
            fail(f"{label} tree row identity differs")
        expected_fields = {
            "relative_path",
            "kind",
            "device",
            "inode",
            "mode",
        }
        if kind == "file":
            expected_fields |= {"byte_count", "sha256"}
            if (
                not isinstance(row.get("byte_count"), int)
                or isinstance(row.get("byte_count"), bool)
                or row.get("byte_count", -1) < 0
                or not isinstance(row.get("sha256"), str)
                or len(row.get("sha256", "")) != 64
            ):
                fail(f"{label} file closure differs")
        if set(row) != expected_fields:
            fail(f"{label} tree row schema differs")
        paths.append(relative)
    if (
        paths != sorted(paths)
        or len(paths) != len(set(paths))
        or paths[0] != "."
        or spec.get("canonical_tree_digest")
        != object_sha256({"root": root, "rows": rows})
    ):
        fail(f"{label} canonical tree closure differs")


def load_authority() -> Mapping[str, Any]:
    value = strict_json_bytes(stable_file_bytes(AUTHORITY_PATH))
    if not isinstance(value, Mapping):
        fail("V3 authority must be one object")
    body = dict(value)
    claim = body.pop("authority_self_sha256", None)
    if claim != object_sha256(body):
        fail("V3 authority self hash differs")
    if value.get("experiment_id") != EXPERIMENT_ID or value.get("sealed_before_gpu_execution") is not True:
        fail("V3 authority identity differs")
    boundaries = value.get("boundaries")
    if not isinstance(boundaries, Mapping) or any(
        (
            boundaries.get("real_gpu_launch_authorized") is not REAL_GPU_LAUNCH_AUTHORIZED,
            boundaries.get("independent_preflip_audit_required") is not True,
            boundaries.get("training_performed") is not False,
            boundaries.get("parameter_updates") != 0,
            boundaries.get("generator_loaded") is not False,
            boundaries.get("generator_forward_calls") != 0,
            boundaries.get("representation_admission_hard_false") is not True,
            boundaries.get("scientific_evidence_claimed") is not False,
        )
    ):
        fail("V3 hard boundary differs")
    fixed = value.get("fixed_paths")
    prior = value.get("prior_failed_engineering_attempt")
    v3r2_failed = value.get("v3r2_failed_engineering_attempt")
    v3r3_failed = value.get("v3r3_failed_engineering_attempt")
    repair = value.get("v3r2_engineering_repair_contract")
    repair_v3r3 = value.get("v3r3_engineering_repair_contract")
    repair_v3r4 = value.get("v3r4_engineering_repair_contract")
    sam_layout_source = (
        repair_v3r3.get("sam_pinned_binary_mask_source_evidence")
        if isinstance(repair_v3r3, Mapping)
        else None
    )
    if (
        not isinstance(fixed, Mapping)
        or not str(fixed.get("planned_preflip_snapshot_root", "")).endswith(
            "v3r4"
        )
        or not str(fixed.get("fresh_formal_run_root", "")).endswith("v3r4")
        or fixed.get("miopen_user_dirname") != "miopen-user"
        or fixed.get("miopen_custom_cache_dirname") != "miopen-custom"
        or fixed.get("miopen_scratch_closure_filename")
        != "miopen_scratch_closure.json"
        or not isinstance(prior, Mapping)
        or prior.get("run_root") == fixed.get("fresh_formal_run_root")
        or prior.get("immutable_preservation_required") is not True
        or prior.get("relaunch_or_reuse_forbidden") is not True
        or prior.get("candidate_present") is not False
        or prior.get("completion_seal_present") is not False
        or any(
            not isinstance(prior.get(name), Mapping)
            or prior[name].get("mode") != 0o444
            or not isinstance(prior[name].get("sha256"), str)
            or len(prior[name]["sha256"]) != 64
            for name in ("formal_log", "attempt_ledger")
        )
        or not isinstance(repair, Mapping)
        or repair.get("old_snapshot_and_run_are_immutable_and_forbidden")
        is not True
        or repair.get("fresh_run_root_create_only") is not True
        or repair.get("miopen_disable_cache_forbidden") is not True
        or repair.get("zero_gpu_controller_task_internal_mask_reset_required")
        is not True
        or repair.get("controller_wrapper_sha256")
        != file_sha256(
            METHOD_ROOT
            / str(repair.get("controller_wrapper_relative_path", ""))
        )
        or not isinstance(v3r2_failed, Mapping)
        or v3r2_failed.get("run_root") == fixed.get("fresh_formal_run_root")
        or v3r2_failed.get("immutable_preservation_required") is not True
        or v3r2_failed.get("relaunch_or_reuse_forbidden") is not True
        or v3r2_failed.get("candidate_present") is not False
        or v3r2_failed.get("completion_seal_present") is not False
        or v3r2_failed.get("scientific_evidence_claimed") is not False
        or not isinstance(v3r2_failed.get("failure_closure_receipt"), Mapping)
        or v3r2_failed["failure_closure_receipt"].get("mode") != 0o444
        or not isinstance(v3r3_failed, Mapping)
        or v3r3_failed.get("run_root") == fixed.get("fresh_formal_run_root")
        or v3r3_failed.get("immutable_preservation_required") is not True
        or v3r3_failed.get("relaunch_or_reuse_forbidden") is not True
        or v3r3_failed.get("candidate_present") is not False
        or v3r3_failed.get("completion_seal_present") is not False
        or v3r3_failed.get("scientific_evidence_claimed") is not False
        or not isinstance(v3r3_failed.get("failure_closure_receipt"), Mapping)
        or v3r3_failed["failure_closure_receipt"].get("mode") != 0o444
        or not isinstance(repair_v3r3, Mapping)
        or repair_v3r3.get(
            "v3r1_and_v3r2_snapshot_run_and_receipt_are_immutable_and_forbidden"
        )
        is not True
        or repair_v3r3.get("fresh_run_root_create_only") is not True
        or repair_v3r3.get("sam_external_mask_schema_is_checked_per_field")
        is not True
        or repair_v3r3.get(
            "sam_external_mask_full_backing_storage_span_required"
        )
        is not True
        or repair_v3r3.get("sam_per_mask_claim_copy_release_order_required")
        is not True
        or repair_v3r3.get(
            "sam_arbitrary_strided_and_partial_base_views_rejected"
        )
        is not True
        or repair_v3r3.get(
            "sam_c_contiguous_copy_is_unconditional_and_separately_owned"
        )
        is not True
        or repair_v3r3.get("sam_external_mask_is_immediately_zeroized_after_copy")
        is not True
        or repair_v3r3.get("sam_ascontiguousarray_alias_shortcut_forbidden")
        is not True
        or not isinstance(sam_layout_source, Mapping)
        or sam_layout_source.get("claim_boundary")
        != (
            "the full-storage transpose layout is derived from pinned SAM2 "
            "source bytes, not inferred from the V3R2 compound failure log"
        )
        or not isinstance(sam_layout_source.get("sources"), list)
        or len(sam_layout_source["sources"]) != 2
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "role",
                "path",
                "sha256",
                "line_start",
                "line_end",
                "line_span_sha256",
            }
            for row in sam_layout_source["sources"]
        )
        or {
            row.get("role")
            for row in sam_layout_source["sources"]
            if isinstance(row, Mapping)
        }
        != {"uncompressed_rle_to_mask", "automatic_binary_mask_return"}
        or repair_v3r3.get("controller_wrapper_sha256")
        != file_sha256(
            METHOD_ROOT
            / str(repair_v3r3.get("controller_wrapper_relative_path", ""))
        )
        or not isinstance(repair_v3r4, Mapping)
        or repair_v3r4.get(
            "v3r1_v3r2_v3r3_snapshots_runs_and_receipts_are_immutable_and_forbidden"
        )
        is not True
        or repair_v3r4.get("fresh_run_root_create_only") is not True
        or repair_v3r4.get("failure_class")
        != "numpy_sequence_truthiness_in_metric_reduction"
        or repair_v3r4.get("failure_location")
        != "actual_target_foundation_runtime_v3.py:_cosine"
        or repair_v3r4.get("sequence_truthiness_forbidden") is not True
        or repair_v3r4.get("sequence_lengths_are_obtained_in_guarded_operations")
        is not True
        or repair_v3r4.get("empty_sequence_check_uses_explicit_length")
        is not True
        or repair_v3r4.get("numeric_conversion_and_reduction_fail_closed")
        is not True
        or repair_v3r4.get(
            "numpy_nonempty_empty_zero_nonfinite_shape_mismatch_and_scalar_tests_required"
        )
        is not True
        or repair_v3r4.get(
            "full_numpy_node_track_edge_drop_edge_phase_case_path_test_required"
        )
        is not True
        or repair_v3r4.get("sam_v3r3_ownership_repair_inherited_unchanged")
        is not True
        or repair_v3r4.get("new_immutable_snapshot_and_fresh_run_required")
        is not True
    ):
        fail("V3R4 fresh-root/failed-attempt engineering boundary differs")
    legacy_snapshot = prior.get("legacy_snapshot")
    legacy_run = prior.get("legacy_run_tree")
    _validate_legacy_tree_authority(legacy_snapshot, "legacy snapshot")
    _validate_legacy_tree_authority(legacy_run, "legacy run")
    snapshot_rows = legacy_snapshot["rows"]
    run_rows = legacy_run["rows"]
    if (
        legacy_snapshot.get("root")
        == fixed.get("planned_preflip_snapshot_root")
        or legacy_snapshot.get("manifest_relative_path")
        != "snapshot_manifest_v3.json"
        or legacy_snapshot.get("manifest_schema_version")
        != "actual-target-foundation-immutable-snapshot-v3"
        or legacy_snapshot.get("snapshot_file_count") != 16
        or len(snapshot_rows) != 21
        or len([row for row in snapshot_rows if row["kind"] == "file"])
        != 17
        or any(
            row["mode"] != 0o555
            if row["kind"] == "directory"
            else row["mode"] not in {0o444, 0o555}
            for row in snapshot_rows
        )
        or legacy_snapshot.get("manifest_file_sha256")
        != next(
            (
                row.get("sha256")
                for row in snapshot_rows
                if row["relative_path"] == "snapshot_manifest_v3.json"
            ),
            None,
        )
        or legacy_run.get("root") != prior.get("run_root")
        or len(run_rows) != 8
        or [row["relative_path"] for row in run_rows]
        != [
            ".",
            "attempt_ledger.json",
            "cache",
            "controller_argv.nul",
            "formal.log",
            "rank_argv.nul",
            "srun_argv.nul",
            "step_meta.json",
        ]
        or any(
            row["mode"] != (0o555 if row["kind"] == "directory" else 0o444)
            for row in run_rows
        )
        or legacy_run.get("candidate_absent") is not True
        or legacy_run.get("completion_seal_absent") is not True
    ):
        fail("V3R2 legacy snapshot/run authority matrix differs")
    base_authority = value.get("base_authority")
    expected = {
        "canary_module_sha256": file_sha256(METHOD_ROOT / "actual_target_foundation_canary_v1.py"),
        "prereg_file_sha256": file_sha256(base.PREREG_PATH),
        "availability_file_sha256": file_sha256(base.AVAILABILITY_PATH),
        "decode_receipt_file_sha256": file_sha256(base.DECODE_RECEIPT_PATH),
    }
    if not isinstance(base_authority, Mapping) or any(base_authority.get(key) != digest for key, digest in expected.items()):
        fail("V3 base authority file binding differs")
    prereg = base.load_preregistration()
    availability = base.load_availability()
    decode = base.load_decode_receipt()
    if (
        base_authority.get("prereg_self_sha256") != prereg.get("prereg_self_sha256")
        or base_authority.get("availability_self_sha256") != availability.get("availability_self_sha256")
        or base_authority.get("decode_receipt_self_sha256") != decode.get("decode_receipt_self_sha256")
    ):
        fail("V3 base authority self-hash binding differs")
    return value


def load_preregistration() -> Mapping[str, Any]:
    load_authority()
    return base.load_preregistration()


def load_availability() -> Mapping[str, Any]:
    load_authority()
    return base.load_availability()


def load_decode_receipt() -> Mapping[str, Any]:
    load_authority()
    return base.load_decode_receipt()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _at_least(value: Any, minimum: float) -> bool:
    return _finite(value) and float(value) >= float(minimum)


def _finite_difference_exceeds(left: Any, right: Any, minimum: float) -> bool:
    return (
        _finite(left)
        and _finite(right)
        and abs(float(left) - float(right)) > float(minimum)
    )


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _phase_counts(value: Any) -> bool:
    return isinstance(value, list) and len(value) == PHASES and all(_nonnegative_int(item) for item in value)


def _phase_counts_bounded(value: Any, maximum: int) -> bool:
    return _phase_counts(value) and all(item <= maximum for item in value)


def _phase_sum_at_least(value: Any, minimum: int) -> bool:
    return _phase_counts(value) and sum(value) >= minimum


def _exact_margins(value: Any, minimum: float) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(INPUT_CONTROLS)
        and all(_at_least(value[name], minimum) for name in INPUT_CONTROLS)
    )


def _canonical_copy(value: Any) -> Any:
    return strict_json_bytes(canonical_json_bytes(value))


@dataclass(frozen=True)
class CaseEvidenceV3:
    family: str
    pair_id: str
    branches: Mapping[str, Mapping[str, Any]]

    def to_mapping(self) -> Mapping[str, Any]:
        return _canonical_copy(
            {"family": self.family, "pair_id": self.pair_id, "branches": self.branches}
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "CaseEvidenceV3":
        if not isinstance(value, Mapping) or set(value) != {"family", "pair_id", "branches"}:
            fail("mechanical CaseEvidenceV3 top-level schema differs")
        branches = value.get("branches")
        if not isinstance(branches, Mapping) or set(branches) != set(BRANCHES):
            fail("mechanical CaseEvidenceV3 branch matrix differs")
        for name, fields in BRANCH_FIELDS.items():
            row = branches.get(name)
            if not isinstance(row, Mapping) or set(row) != fields:
                fail(f"mechanical CaseEvidenceV3 {name} fields differ")
        family = value.get("family")
        pair_id = value.get("pair_id")
        if not isinstance(family, str) or not isinstance(pair_id, str):
            fail("mechanical CaseEvidenceV3 identity differs")
        return cls(family=family, pair_id=pair_id, branches=_canonical_copy(branches))


def evaluate_case(
    evidence: CaseEvidenceV3,
    prereg: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    spec = dict(prereg or load_preregistration())
    CaseEvidenceV3.from_mapping(evidence.to_mapping())
    pair_authority = {row["family"]: row["pair_id"] for row in spec["pairs"]}
    if evidence.family not in FAMILIES or pair_authority.get(evidence.family) != evidence.pair_id:
        fail("case is not bound to its preregistered family/pair")
    gates = spec["fixed_gates"]
    frozen = evidence.branches["frozen_base"]
    node = evidence.branches["node"]
    track = evidence.branches["track"]
    edge = evidence.branches["edge"]
    phase = evidence.branches["ordered_phase"]

    hook_delta = frozen.get("actual_forward_hook_delta")
    frozen_pass = all(
        (
            frozen.get("all_models_eval_frozen") is True,
            frozen.get("source_and_weight_closure_unchanged") is True,
            frozen.get("parameter_updates") == 0,
            frozen.get("generator_forward_calls") == 0,
            frozen.get("full_model_closure_deferred_to_run_receipt") is True,
            hook_delta == {"sam2_image_encoder": 24, "dinov2": 24, "cotracker": 5, "vjepa2": 5},
        )
    )
    node_pass = all(
        (
            node.get("dustbin_used") is True,
            node.get("unbalanced_phase_pair_count") == 7,
            _nonnegative_int(node.get("dustbin_unmatched_count")),
            _at_least(node.get("dustbin_transport_mass"), 0.0),
            node.get("forced_nonempty_slot_used") is False,
            node.get("anonymous_slot_relabel_invariant") is True,
            _phase_counts_bounded(node.get("phase_cardinalities"), 12),
            _at_least(node.get("mechanically_valid_phases"), gates["minimum_mechanically_valid_phases"]),
            _at_least(node.get("positive_similarity"), gates["node_positive_similarity_min"]),
            _exact_margins(node.get("input_margins"), gates["node_margin_each_input_control_min"]),
            _at_least(node.get("mask_descriptor_binding_break_margin"), gates["mask_descriptor_binding_break_node_margin_min"]),
        )
    )

    state_counts = track.get("state_counts")
    lifecycle = track.get("lifecycle_counts")
    exact_states = {"ABSENT", "VISIBLE_MEMBER", "OCCLUDED", "VISIBLE_OUTSIDE_MASK"}
    exact_lifecycle = {"entry", "occlusion", "membership_loss", "reentry", "death"}
    lifecycle_valid = (
        isinstance(state_counts, Mapping)
        and set(state_counts) == exact_states
        and all(_nonnegative_int(item) for item in state_counts.values())
        and isinstance(lifecycle, Mapping)
        and set(lifecycle) == exact_lifecycle
        and all(_nonnegative_int(item) for item in lifecycle.values())
    )
    dynamic_nonentry = lifecycle_valid and sum(lifecycle[name] for name in ("occlusion", "membership_loss", "reentry", "death")) >= 1
    track_pass = all(
        (
            _at_least(track.get("assigned_track_count"), 1),
            _at_least(track.get("assigned_point_count"), 1),
            _at_least(track.get("minimum_same_track_member_phases_observed"), 3),
            _at_least(track.get("visible_and_member_fraction"), gates["track_visible_fraction_min"]),
            _phase_counts(track.get("per_phase_visible_member_counts")),
            all(
                _nonnegative_int(track.get(name))
                for name in (
                    "ambiguous_overlap_observation_count",
                    "out_of_bounds_observation_count",
                    "nonfinite_observation_count",
                    "vote_tie_abstain_count",
                    "insufficient_membership_abstain_count",
                )
            ),
            lifecycle_valid,
            track.get("dynamic_nonentry_lifecycle_observed") is True,
            dynamic_nonentry,
            _at_least(track.get("valid_adjacent_velocity_count"), 1),
            _at_least(track.get("positive_similarity"), gates["track_positive_similarity_min"]),
            _exact_margins(track.get("input_margins"), gates["track_margin_each_input_control_min"]),
            _at_least(track.get("cross_phase_track_identity_break_margin"), gates["cross_phase_track_identity_break_margin_min"]),
        )
    )

    edge_pass = all(
        (
            _phase_counts(edge.get("per_phase_active_counts")),
            _phase_counts(edge.get("per_phase_birth_counts")),
            _phase_counts(edge.get("per_phase_persist_counts")),
            _phase_counts(edge.get("per_phase_death_counts")),
            _phase_counts(edge.get("per_phase_valid_velocity_counts")),
            _phase_counts(edge.get("per_phase_qualified_lifecycle_counts")),
            edge.get("real_per_phase_lifecycle_channels") is True,
            _at_least(edge.get("evaluated_pairwise_edge_count"), 1),
            (
                _phase_counts(edge.get("per_phase_persist_counts"))
                and _phase_counts(edge.get("per_phase_death_counts"))
                and sum(edge["per_phase_persist_counts"])
                + sum(edge["per_phase_death_counts"])
                >= 1
            ),
            _phase_sum_at_least(
                edge.get("per_phase_valid_velocity_counts"), 1
            ),
            _phase_sum_at_least(
                edge.get("per_phase_qualified_lifecycle_counts"), 1
            ),
            _at_least(edge.get("positive_similarity"), gates["edge_positive_similarity_min"]),
            _exact_margins(edge.get("input_margins"), gates["edge_margin_each_input_control_min"]),
            _at_least(edge.get("drop_edge_margin"), gates["drop_edge_margin_min"]),
            _at_least(edge.get("drop_edge_removed_count"), 1),
            _at_least(edge.get("drop_edge_control_norm"), 1e-12),
            _finite(edge.get("drop_edge_control_similarity")),
            _at_least(edge.get("drop_edge_positive_l2_distance"), 1e-12),
            _finite(edge.get("drop_edge_margin")),
            _finite(edge.get("positive_similarity")),
            _finite_difference_exceeds(
                edge.get("drop_edge_margin"),
                edge.get("positive_similarity"),
                1e-12,
            ),
        )
    )
    phase_pass = _exact_margins(
        phase.get("input_margins"), gates["vjepa_phase_margin_each_input_control_min"]
    )
    branch_pass = {
        "frozen_base": frozen_pass,
        "node": node_pass,
        "track": track_pass,
        "edge": edge_pass,
        "ordered_phase": phase_pass,
    }
    value = {
        "family": evidence.family,
        "pair_id": evidence.pair_id,
        "mechanical_evidence_digest": object_sha256(evidence.to_mapping()),
        "branch_pass": branch_pass,
        "case_formula": "frozen_base AND node AND track AND edge AND ordered_phase",
        "case_pass": all(branch_pass.values()),
        "branch_compensation_permitted": False,
        "control_compensation_permitted": False,
        "representation_admitted": False,
        "scalar_metrics": _canonical_copy({name: evidence.branches[name] for name in BRANCHES}),
    }
    return {**value, "digest": object_sha256(value)}


def aggregate_canary(
    rows: Sequence[Mapping[str, Any]],
    evidences: Sequence[CaseEvidenceV3],
    prereg: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    spec = dict(prereg or load_preregistration())
    if len(rows) != 4 or len(evidences) != 4 or {row.get("family") for row in rows} != set(FAMILIES):
        fail("aggregate requires exactly one row per development family")
    evidence_by_family = {row.family: row for row in evidences}
    if set(evidence_by_family) != set(FAMILIES):
        fail("aggregate mechanical evidence family matrix differs")
    family_pass = {}
    for row in rows:
        evidence = evidence_by_family[row["family"]]
        rebuilt = evaluate_case(evidence, spec)
        if _canonical_copy(row) != rebuilt:
            fail("aggregate row is not exact recomputed case evidence")
        family_pass[row["family"]] = row.get("case_pass") is True and all(
            row["branch_pass"].get(name) is True for name in BRANCHES
        )
    value = {
        "development_only": True,
        "locked_validation_claimed": False,
        "scientific_evidence_claimed": False,
        "family_pass": family_pass,
        "passed_case_count": sum(family_pass.values()),
        "diagnostic_canary_pass": all(family_pass.values()),
        "canary_formula": "all 4/4 seen development cases pass every non-compensable branch and control",
        "representation_admitted": False,
        "stable_transferable_action_representation_established": False,
        "generator_connection_authorized": False,
    }
    return {**value, "digest": object_sha256(value)}


def verify_remote_assets() -> Mapping[str, Any]:
    base_receipt = base.verify_remote_assets()
    source_trees = foundation_source_tree_closure()
    authority = load_authority()
    hydra = authority["sam_hydra_authority"]
    path = Path(hydra["runtime_config_path"])
    payload = stable_file_bytes(path)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != hydra["runtime_config_sha256"]:
        fail("actual packaged SAM Hydra config SHA differs")
    files = list(base_receipt["files"])
    files.append(
        {
            "role": "sam2:actual_hydra_runtime_config",
            "path": str(path),
            "sha256": digest,
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
    )
    config_authority = authority["preprocessor_and_nontensor_config_authority"]
    for name in ("sam_build_function", "dinov2_processor", "vjepa2_processor"):
        row = config_authority[name]
        source = Path(row["source_path"])
        source_payload = stable_file_bytes(source)
        source_digest = hashlib.sha256(source_payload).hexdigest()
        if source_digest != row["source_sha256"]:
            fail(f"{name} source SHA differs")
        files.append(
            {
                "role": f"runtime_config_source:{name}",
                "path": str(source),
                "sha256": source_digest,
                "mode": stat.S_IMODE(source.stat().st_mode),
            }
        )
    value = {
        "status": "PASS",
        "verified_file_count": len(files),
        "files": files,
        "base_asset_digest": base_receipt["digest"],
        "foundation_source_trees": source_trees,
        "gpu_used": False,
        "model_forward_calls": 0,
    }
    return {**value, "digest": object_sha256(value)}


def contract() -> Mapping[str, Any]:
    authority = load_authority()
    value = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "implementation_status": "V3R4_NUMPY_METRIC_REPAIR_SOURCE_INDEPENDENT_AUDIT_PASS_LAUNCH_AUTHORIZED",
        "authority_file": str(AUTHORITY_PATH),
        "authority_file_sha256": file_sha256(AUTHORITY_PATH),
        "authority_self_sha256": authority["authority_self_sha256"],
        "development_case_count": 4,
        "locked_validation_case_count": 0,
        "families": list(FAMILIES),
        "branches": list(BRANCHES),
        "branchwise_AND_without_compensation": True,
        "raw_inventory_categories": authority["raw_inventory_required_categories"],
        "raw_ownership_contract": authority["raw_ownership_contract"],
        "track_contract": authority["track_contract"],
        "edge_contract": authority["edge_contract"],
        "model_closure": authority["model_closure"],
        "completion": authority["completion"],
        "fixed_paths": authority["fixed_paths"],
        "snapshot_payload_relative_paths": authority["snapshot_payload_relative_paths"],
        "training_performed": False,
        "parameter_updates": 0,
        "generator_loaded": False,
        "generator_forward_calls": 0,
        "representation_admission_hard_false": True,
        "real_gpu_launch_authorized": REAL_GPU_LAUNCH_AUTHORIZED,
        "independent_preflip_audit_required": True,
    }
    return {**value, "digest": object_sha256(value)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contract", action="store_true")
    mode.add_argument("--verify-remote-assets", action="store_true")
    mode.add_argument("--run-real", action="store_true")
    args = parser.parse_args(argv)
    if args.print_contract:
        print(json.dumps(contract(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.verify_remote_assets:
        print(json.dumps(verify_remote_assets(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    fail("V3 authority module is not an execution entry point; use the reviewed immutable runtime launcher")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_PATH",
    "BRANCHES",
    "BRANCH_FIELDS",
    "CanaryV3Error",
    "CaseEvidenceV3",
    "EXPERIMENT_ID",
    "FAMILIES",
    "INPUT_CONTROLS",
    "PHASES",
    "REAL_GPU_LAUNCH_AUTHORIZED",
    "aggregate_canary",
    "canonical_json_bytes",
    "contract",
    "evaluate_case",
    "fail",
    "file_sha256",
    "foundation_source_tree_closure",
    "load_authority",
    "load_availability",
    "load_decode_receipt",
    "load_preregistration",
    "object_sha256",
    "stable_file_bytes",
    "strict_json_bytes",
    "verify_remote_assets",
]
