#!/usr/bin/env python3
"""Fail-closed prelaunch authority for the actual-target foundation canary.

This module deliberately contains no generator, optimizer, training, or GPU
implementation.  It freezes the four seen development cases, verifies the
offline foundation files, and implements the exact non-compensable admission
logic that a later independently audited GPU runner must call.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = METHOD_ROOT / "assets"
PREREG_PATH = ASSET_ROOT / "actual_target_foundation_canary_prereg_v1.json"
AVAILABILITY_PATH = (
    ASSET_ROOT / "actual_target_foundation_canary_availability_v1.json"
)
DECODE_RECEIPT_PATH = ASSET_ROOT / "actual_target_foundation_decode_receipt_v2.json"
SCHEMA_VERSION = "actual-target-foundation-canary-runner-v1"
EXPERIMENT_ID = "actual_target_foundation_canary_v1"
FAMILIES = (
    "articulated_ordered_motion",
    "contact_transfer",
    "lifecycle_entry_exit",
    "multi_entity_interaction",
)
INPUT_CONTROLS = (
    "target_reverse",
    "target_deterministic_shuffle",
    "source_noop",
)
COUNTERFACTUAL_CONTROLS = (
    "mask_descriptor_binding_break",
    "cross_phase_track_identity_break",
    "drop_edge",
)
BRANCHES = ("frozen_base", "node", "track", "edge", "ordered_phase")
GPU_SMOKE_AUTHORIZED = False
FORMAL_GPU_EXECUTION_AUTHORIZED = False


class CanaryError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise CanaryError(message)


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
        raise CanaryError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_self_hashed(path: Path, field: str) -> Mapping[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        fail(f"authority is not a plain file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CanaryError(f"authority cannot be parsed: {path}") from error
    if not isinstance(value, dict):
        fail("authority must be one JSON object")
    claim = value.get(field)
    body = dict(value)
    body.pop(field, None)
    if not isinstance(claim, str) or claim != object_sha256(body):
        fail(f"authority self hash differs: {path.name}")
    return value


def load_preregistration() -> Mapping[str, Any]:
    value = _load_self_hashed(PREREG_PATH, "prereg_self_sha256")
    if (
        value.get("experiment_id") != EXPERIMENT_ID
        or value.get("sealed_before_gpu_execution") is not True
    ):
        fail("preregistration identity differs")
    scope = value.get("scope")
    if not isinstance(scope, Mapping) or any(
        (
            scope.get("development_only") is not True,
            scope.get("previously_seen_in_r1b") is not True,
            scope.get("locked_validation_claim_permitted") is not False,
            scope.get("scientific_representation_claim_permitted") is not False,
        )
    ):
        fail("development-only claim boundary differs")
    pairs = value.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 4:
        fail("canary requires exactly four pairs")
    if {row.get("family") for row in pairs if isinstance(row, Mapping)} != set(
        FAMILIES
    ):
        fail("canary requires exactly one pair per family")
    if len({row.get("pair_id") for row in pairs}) != 4 or len(
        {row.get("uuid") for row in pairs}
    ) != 4:
        fail("canary pair identities are not unique")
    if {row.get("r1b_ordinal") for row in pairs} != {2, 6, 7, 11}:
        fail("seen R1b row authority differs")
    representation = value.get("proposal_and_representation")
    if not isinstance(representation, Mapping) or any(
        (
            representation.get("manual_boxes_permitted") is not False,
            representation.get("semantic_role_names_permitted") is not False,
            representation.get("variable_cardinality_range") != [0, 12],
            representation.get("forced_nonempty_slot_permitted") is not False,
            representation.get("dustbin_required") is not True,
        )
    ):
        fail("anonymous variable-cardinality authority differs")
    closure=value.get("v2_closure"); decoded=value.get("decoded_media_authority")
    if not isinstance(closure,Mapping) or any((
        closure.get("unbalanced_matching")!={"algorithm":"KL-relaxed Sinkhorn with explicit final row/column dustbin","epsilon":0.08,"rho":0.35,"dustbin_cost":0.42,"iterations":80},
        closure.get("edge_channels")!=["mask_boundary_gap","mask_overlap_iou","relative_velocity","pairwise_birth_death_lifecycle"],
        "exactly invariant" not in closure.get("anonymous_relabel_rule",""),
        "external CPU postflight" not in closure.get("completion",""),
    )): fail("v2 graph/ownership/completion closure differs")
    if not isinstance(decoded,Mapping) or any((
        decoded.get("receipt_file")!="assets/actual_target_foundation_decode_receipt_v2.json",
        decoded.get("receipt_self_sha256")!="210ac14e5d9ebf9043f931bad0630f394c7108118ea7dc93f0008b05e841cea1",
        decoded.get("receipt_file_sha256")!=file_sha256(DECODE_RECEIPT_PATH),
        not isinstance(decoded.get("rows"),list),
        len(decoded.get("rows",()))!=8,
    )): fail("decoded RGB preregistration closure differs")
    admission = value.get("admission")
    boundary = value.get("boundaries")
    if not isinstance(admission, Mapping) or any(
        (
            admission.get("branch_compensation_permitted") is not False,
            admission.get("control_compensation_permitted") is not False,
            admission.get("representation_admission_hard_false") is not True,
        )
    ):
        fail("non-compensable admission authority differs")
    if not isinstance(boundary, Mapping) or any(
        (
            boundary.get("training_performed") is not False,
            boundary.get("optimizer_created") is not False,
            boundary.get("parameter_updates") != 0,
            boundary.get("generator_loaded") is not False,
            boundary.get("generator_forward_calls") != 0,
            boundary.get("gpu_smoke_authorized") is not GPU_SMOKE_AUTHORIZED,
            boundary.get("formal_gpu_execution_authorized")
            is not FORMAL_GPU_EXECUTION_AUTHORIZED,
            boundary.get("independent_audit_required_before_gpu") is not True,
        )
    ):
        fail("execution boundary differs")
    return value


def load_availability() -> Mapping[str, Any]:
    value = _load_self_hashed(AVAILABILITY_PATH, "availability_self_sha256")
    if value.get("gpu_smoke_authorized") is not False or value.get(
        "formal_gpu_execution_authorized"
    ) is not False:
        fail("availability receipt crosses GPU boundary")
    foundations = value.get("foundations")
    if not isinstance(foundations, Mapping) or set(foundations) != {
        "sam2",
        "cotracker",
        "dinov2",
        "vjepa2",
    }:
        fail("foundation availability matrix differs")
    for name, row in foundations.items():
        if not isinstance(row, Mapping):
            fail(f"{name} availability row differs")
    classes=value.get("runtime_class_authority")
    if not isinstance(classes,list) or len(classes)!=5 or {(row.get("module"),row.get("class")) for row in classes if isinstance(row,Mapping)}!={("sam2.modeling.sam2_base","SAM2Base"),("sam2.automatic_mask_generator","SAM2AutomaticMaskGenerator"),("cotracker.predictor","CoTrackerPredictor"),("transformers.models.dinov2.modeling_dinov2","Dinov2Model"),("transformers.models.vjepa2.modeling_vjepa2","VJEPA2Model")}:
        fail("runtime class source authority differs")
    return value


def load_decode_receipt() -> Mapping[str, Any]:
    value=_load_self_hashed(DECODE_RECEIPT_PATH,"decode_receipt_self_sha256")
    rows=value.get("rows")
    if not isinstance(rows,list) or len(rows)!=8 or value.get("all_frames_dtype")!="uint8" or value.get("all_frames_shape_hwc")!=[720,1280,3]:
        fail("decoded RGB authority differs")
    prereg=load_preregistration(); expected={(row["r1b_ordinal"],role):row[f"{role}_video_sha256"] for row in prereg["pairs"] for role in ("source","target")}
    observed={(row.get("r1b_ordinal"),row.get("role")):row.get("compressed_sha256") for row in rows if isinstance(row,Mapping)}
    if observed!=expected: fail("decoded RGB rows do not bind preregistered media")
    return value


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _at_least(value: Any, minimum: float) -> bool:
    return _finite(value) and float(value) >= float(minimum)


def _json_scalar(value: Any) -> Optional[float]:
    return float(value) if _finite(value) else None


def _scalar_margins(value: Any) -> Mapping[str, Optional[float]]:
    if not isinstance(value, Mapping):
        return {name: None for name in INPUT_CONTROLS}
    return {name: _json_scalar(value.get(name)) for name in INPUT_CONTROLS}


def _exact_margins(
    values: Any, *, minimum: float, controls: Sequence[str]
) -> bool:
    return (
        isinstance(values, Mapping)
        and set(values) == set(controls)
        and all(_at_least(values[name], minimum) for name in controls)
    )


@dataclass(frozen=True)
class CaseEvidenceV1:
    family: str
    pair_id: str
    branches: Mapping[str, Mapping[str, Any]]


def evaluate_case(
    evidence: CaseEvidenceV1, prereg: Optional[Mapping[str, Any]] = None
) -> Mapping[str, Any]:
    """Apply every fixed branch and control as a strict conjunction."""

    spec = dict(prereg or load_preregistration())
    gates = spec["fixed_gates"]
    if evidence.family not in FAMILIES or set(evidence.branches) != set(BRANCHES):
        fail("case evidence matrix differs")
    pair_authority = {
        row["family"]: row["pair_id"] for row in spec["pairs"]
    }
    if pair_authority.get(evidence.family) != evidence.pair_id:
        fail("case pair is not bound to its preregistered family")
    frozen = evidence.branches["frozen_base"]
    node = evidence.branches["node"]
    track = evidence.branches["track"]
    edge = evidence.branches["edge"]
    phase = evidence.branches["ordered_phase"]
    frozen_pass = all(
        (
            frozen.get("all_models_eval_frozen") is True,
            frozen.get("source_and_weight_closure_unchanged") is True,
            frozen.get("parameter_updates") == 0,
            frozen.get("generator_forward_calls") == 0,
        )
    )
    node_pass = all(
        (
            node.get("dustbin_used") is True,
            node.get("unbalanced_phase_pair_count") == 7,
            isinstance(node.get("dustbin_unmatched_count"), int),
            not isinstance(node.get("dustbin_unmatched_count"), bool),
            node.get("dustbin_unmatched_count", -1) >= 0,
            _at_least(node.get("dustbin_transport_mass"), 0.0),
            node.get("forced_nonempty_slot_used") is False,
            node.get("anonymous_slot_relabel_invariant") is True,
            isinstance(node.get("phase_cardinalities"), list),
            len(node.get("phase_cardinalities", ()))
            == spec["views_and_controls"]["phase_keyframe_count"],
            all(
                isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 12
                for value in node.get("phase_cardinalities", ())
            ),
            _at_least(
                node.get("mechanically_valid_phases"),
                gates["minimum_mechanically_valid_phases"],
            ),
            _at_least(
                node.get("positive_similarity"),
                gates["node_positive_similarity_min"],
            ),
            _exact_margins(
                node.get("input_margins"),
                minimum=gates["node_margin_each_input_control_min"],
                controls=INPUT_CONTROLS,
            ),
            _at_least(
                node.get("mask_descriptor_binding_break_margin"),
                gates["mask_descriptor_binding_break_node_margin_min"],
            ),
        )
    )
    track_pass = all(
        (
            isinstance(track.get("assigned_track_count"), int),
            not isinstance(track.get("assigned_track_count"), bool),
            track.get("assigned_track_count", 0) >= 1,
            _at_least(
                track.get("visible_fraction"),
                gates["track_visible_fraction_min"],
            ),
            _at_least(
                track.get("positive_similarity"),
                gates["track_positive_similarity_min"],
            ),
            _exact_margins(
                track.get("input_margins"),
                minimum=gates["track_margin_each_input_control_min"],
                controls=INPUT_CONTROLS,
            ),
            _at_least(
                track.get("cross_phase_track_identity_break_margin"),
                gates["cross_phase_track_identity_break_margin_min"],
            ),
        )
    )
    edge_pass = all(
        (
            edge.get("dynamic_lifecycle_observed") is True,
            isinstance(edge.get("pairwise_lifecycle_count"), int),
            not isinstance(edge.get("pairwise_lifecycle_count"), bool),
            edge.get("pairwise_lifecycle_count", 0) >= 1,
            isinstance(edge.get("evaluated_pairwise_edge_count"), int),
            not isinstance(edge.get("evaluated_pairwise_edge_count"), bool),
            edge.get("evaluated_pairwise_edge_count", 0) >= 1,
            _at_least(
                edge.get("positive_similarity"),
                gates["edge_positive_similarity_min"],
            ),
            _exact_margins(
                edge.get("input_margins"),
                minimum=gates["edge_margin_each_input_control_min"],
                controls=INPUT_CONTROLS,
            ),
            _at_least(
                edge.get("drop_edge_margin"),
                gates["drop_edge_margin_min"],
            ),
        )
    )
    phase_pass = _exact_margins(
        phase.get("input_margins"),
        minimum=gates["vjepa_phase_margin_each_input_control_min"],
        controls=INPUT_CONTROLS,
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
        "branch_pass": branch_pass,
        "case_formula": "frozen_base AND node AND track AND edge AND ordered_phase",
        "case_pass": all(branch_pass.values()),
        "branch_compensation_permitted": False,
        "representation_admitted": False,
        "scalar_metrics": {
            "node": {
                "phase_cardinalities": list(node.get("phase_cardinalities", ())),
                "mechanically_valid_phases": node.get("mechanically_valid_phases"),
                "unbalanced_phase_pair_count": node.get("unbalanced_phase_pair_count"),
                "dustbin_unmatched_count": node.get("dustbin_unmatched_count"),
                "dustbin_transport_mass": _json_scalar(node.get("dustbin_transport_mass")),
                "positive_similarity": _json_scalar(node.get("positive_similarity")),
                "input_margins": _scalar_margins(node.get("input_margins")),
                "mask_descriptor_binding_break_margin": _json_scalar(node.get("mask_descriptor_binding_break_margin")),
            },
            "track": {
                "assigned_track_count": track.get("assigned_track_count"),
                "visible_fraction": _json_scalar(track.get("visible_fraction")),
                "positive_similarity": _json_scalar(track.get("positive_similarity")),
                "input_margins": _scalar_margins(track.get("input_margins")),
                "cross_phase_track_identity_break_margin": _json_scalar(track.get("cross_phase_track_identity_break_margin")),
            },
            "edge": {
                "evaluated_pairwise_edge_count": edge.get("evaluated_pairwise_edge_count"),
                "pairwise_lifecycle_count": edge.get("pairwise_lifecycle_count"),
                "positive_similarity": _json_scalar(edge.get("positive_similarity")),
                "input_margins": _scalar_margins(edge.get("input_margins")),
                "drop_edge_margin": _json_scalar(edge.get("drop_edge_margin")),
            },
            "ordered_phase": {"input_margins": _scalar_margins(phase.get("input_margins"))},
        },
    }
    return {**value, "digest": object_sha256(value)}


def aggregate_canary(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(rows) != 4 or {row.get("family") for row in rows} != set(FAMILIES):
        fail("aggregate requires exactly one row per family")
    if any(set(row.get("branch_pass", {})) != set(BRANCHES) for row in rows):
        fail("aggregate branch matrix differs")
    family_pass = {
        row["family"]: (
            row.get("case_pass") is True
            and all(row["branch_pass"].get(branch) is True for branch in BRANCHES)
        )
        for row in rows
    }
    diagnostic_pass = all(family_pass.values())
    value = {
        "development_only": True,
        "locked_validation_claimed": False,
        "family_pass": family_pass,
        "passed_case_count": sum(family_pass.values()),
        "canary_formula": "all 4/4 seen development cases pass every branch and every control",
        "diagnostic_canary_pass": diagnostic_pass,
        "representation_admitted": False,
        "stable_transferable_action_representation_established": False,
        "generator_connection_authorized": False,
    }
    return {**value, "digest": object_sha256(value)}


def verify_remote_assets() -> Mapping[str, Any]:
    """Hash all AUH assets; this performs no model loading or GPU work."""

    prereg = load_preregistration()
    availability = load_availability()
    rows = []

    def bind(path_text: str, expected: str, role: str) -> None:
        path = Path(path_text)
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            fail(f"{role} is not an absolute plain file")
        before = path.stat()
        observed = file_sha256(path)
        after = path.stat()
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        if identity(before) != identity(after):
            fail(f"{role} changed during closure hashing")
        if observed != expected:
            fail(f"{role} SHA-256 differs")
        rows.append({"role": role, "path": str(path), "sha256": observed})

    for pair in prereg["pairs"]:
        bind(pair["source_video_path"], pair["source_video_sha256"], f'{pair["family"]}:source')
        bind(pair["target_video_path"], pair["target_video_sha256"], f'{pair["family"]}:target')
    foundations = availability["foundations"]
    sam = foundations["sam2"]
    bind(sam["checkpoint_path"], sam["checkpoint_sha256"], "sam2:checkpoint")
    bind(sam["config_path"], sam["config_sha256"], "sam2:config")
    for path, digest in sam["source_files"].items():
        bind(path, digest, f"sam2:source:{Path(path).name}")
    cotracker = foundations["cotracker"]
    bind(
        cotracker["checkpoint_path"],
        cotracker["checkpoint_sha256"],
        "cotracker:checkpoint",
    )
    for path, digest in cotracker["source_files"].items():
        bind(path, digest, f"cotracker:source:{Path(path).name}")
    for foundation in ("dinov2", "vjepa2"):
        row = foundations[foundation]
        for name, digest in row["required_files"].items():
            bind(str(Path(row["model_root"]) / name), digest, f"{foundation}:{name}")
    value = {
        "status": "PASS",
        "verified_file_count": len(rows),
        "files": rows,
        "gpu_used": False,
        "model_forward_calls": 0,
    }
    return {**value, "digest": object_sha256(value)}


def contract() -> Mapping[str, Any]:
    prereg = load_preregistration()
    availability = load_availability()
    value = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "implementation_status": "V2_CLOSURE_IMPLEMENTED_UNEXECUTED_PRE_FLIP_NO",
        "preregistration": prereg,
        "availability": availability,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "development_case_count": 4,
        "locked_validation_case_count": 0,
        "families": list(FAMILIES),
        "branches": list(BRANCHES),
        "input_controls": list(INPUT_CONTROLS),
        "counterfactual_controls": list(COUNTERFACTUAL_CONTROLS),
        "manual_boxes_used": False,
        "semantic_role_names_used": False,
        "variable_cardinality_and_dustbin_required": True,
        "branchwise_AND_without_compensation": True,
        "training_performed": False,
        "parameter_updates": 0,
        "generator_loaded": False,
        "generator_forward_calls": 0,
        "gpu_smoke_authorized": GPU_SMOKE_AUTHORIZED,
        "formal_gpu_execution_authorized": FORMAL_GPU_EXECUTION_AUTHORIZED,
        "independent_audit_required_before_gpu": True,
        "representation_admission_hard_false": True,
        "executor": {
            "path": str(METHOD_ROOT / "actual_target_foundation_runtime_v1.py"),
            "sha256": file_sha256(
                METHOD_ROOT / "actual_target_foundation_runtime_v1.py"
            ),
            "rank_wrapper_path": str(
                METHOD_ROOT
                / "scripts"
                / "auh_actual_target_foundation_canary_rank_wrapper_v1.sh"
            ),
            "rank_wrapper_sha256": file_sha256(
                METHOD_ROOT
                / "scripts"
                / "auh_actual_target_foundation_canary_rank_wrapper_v1.sh"
            ),
            "real_gpu_launch_authorized": False,
        },
        "gpu_smoke_plan": {
            "device": "one audited MI210",
            "cases": 4,
            "sam2_automatic_keyframe_calls": 96,
            "dinov2_keyframe_calls": 96,
            "cotracker_video_calls": 20,
            "vjepa2_video_calls": 20,
            "receipt_mode": "absolute absent create-only JSON",
            "first_stage": "one-case engineering smoke, then four-case canary only after independent audit",
        },
    }
    return {**value, "digest": object_sha256(value)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contract", action="store_true")
    mode.add_argument("--verify-remote-assets", action="store_true")
    mode.add_argument("--run-gpu-smoke", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_contract:
        sys.stdout.write(json.dumps(contract(), indent=2, sort_keys=True) + "\n")
        return 0
    if args.verify_remote_assets:
        sys.stdout.write(
            json.dumps(verify_remote_assets(), indent=2, sort_keys=True) + "\n"
        )
        return 0
    if not GPU_SMOKE_AUTHORIZED or not FORMAL_GPU_EXECUTION_AUTHORIZED:
        fail("GPU execution is blocked pending a new independent audit")
    fail("use the separately audited actual_target_foundation_runtime_v1 executor")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AVAILABILITY_PATH",
    "BRANCHES",
    "COUNTERFACTUAL_CONTROLS",
    "DECODE_RECEIPT_PATH",
    "CanaryError",
    "CaseEvidenceV1",
    "EXPERIMENT_ID",
    "FAMILIES",
    "FORMAL_GPU_EXECUTION_AUTHORIZED",
    "GPU_SMOKE_AUTHORIZED",
    "INPUT_CONTROLS",
    "PREREG_PATH",
    "aggregate_canary",
    "contract",
    "evaluate_case",
    "load_availability",
    "load_decode_receipt",
    "load_preregistration",
    "verify_remote_assets",
]
