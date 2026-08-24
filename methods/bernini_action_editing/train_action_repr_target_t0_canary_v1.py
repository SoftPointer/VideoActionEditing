#!/usr/bin/env python3
"""One-step target-T0 canary for the authenticated action-representation route.

This is a real optimizer runner, intentionally limited to fit case
``0be6494dfac3`` and one WORLD4/SP4 update.  It consumes passed G1-target and
production-G2a receipts, never accepts target media on its CLI, and publishes
only create-once adapter states plus execution receipts.  A successful run is
an optimizer/integration canary, not an ``Ours`` or decoded-quality claim.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, ContextManager, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch

import action_repr_g2a_adapter_v1 as g2a
import action_representation_joint_objective_v1 as objective
import audit_action_repr_g2a_world4_v1 as g2a_world4
import materialize_decoded_middle_action_repr_v1 as middle_extractor


SCHEMA_VERSION = "bernini-action-repr-target-t0-one-step-receipt-v1"
STEP_SCHEMA_VERSION = "bernini-action-repr-target-t0-adapter-state-receipt-v1"
METHOD = "bernini-action-repr-target-t0-one-step-canary-v1"
FIXED_CASE_ID = "0be6494dfac3"
BLOCK_INDICES = (6, 12, 18, 24)
HIDDEN_WIDTH = 1536
BOTTLENECK_WIDTH = 256
LEARNING_RATE = 1.0e-4
ADAPTER_SEED = 2026082403
CONTROL_TO_ROUTE = {
    "zero_or_noop": "zero",
    "temporal_shuffle": "temporal_shuffle",
    "reverse": "reverse",
    "incomplete": "incomplete",
    "wrong_action_energy_matched": "wrong_action",
}
FORBIDDEN_MEDIA_KEYS = frozenset(
    {
        "target_video",
        "target_video_path",
        "anchor_video",
        "anchor_video_path",
        "target_rgb",
        "target_vae_latent",
        "target_clean_latent",
        "target_latent",
        "absolute_target_hidden",
        "raw_target_query",
        "raw_target_key",
        "raw_target_value",
    }
)


class TargetT0CanaryError(RuntimeError):
    """Fail-closed T0 authorization or execution error."""


def fail(message: str) -> None:
    raise TargetT0CanaryError(message)


# The implementation below is deliberately split into a CPU-testable optimizer
# kernel and a production-only WORLD4 entry point.  No fake renderer is exposed
# by the CLI.


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
        raise TargetT0CanaryError("T0 evidence is not finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return g2a_world4.file_sha256(path)


def _parse_slurm_rocr_visible_devices(raw: Any) -> list[str]:
    if not isinstance(raw, str) or not raw:
        fail("Slurm ROCR_VISIBLE_DEVICES is absent or empty")
    if "/" in raw or "\\" in raw:
        fail("Slurm ROCR_VISIBLE_DEVICES contains a path separator")
    if any(character.isspace() for character in raw):
        fail("Slurm ROCR_VISIBLE_DEVICES contains whitespace")
    devices = raw.split(",")
    if len(devices) != 4 or any(not token for token in devices):
        fail("Slurm ROCR_VISIBLE_DEVICES must contain exactly four nonempty tokens")
    if any(len(token) != 1 or token not in "01234567" for token in devices):
        fail("Slurm ROCR_VISIBLE_DEVICES tokens must be canonical AUH devices 0..7")
    if len(set(devices)) != 4:
        fail("Slurm ROCR_VISIBLE_DEVICES contains duplicate devices")
    return devices


def validate_slurm_visible_device_receipt(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "raw",
        "devices",
        "device_count",
        "sha256",
        "mapping_preserved",
        "hip_visible_devices_nonempty",
        "cuda_visible_devices_nonempty",
    }:
        fail("Slurm ROCR visible-device receipt closure differs")
    raw = value.get("raw")
    devices = _parse_slurm_rocr_visible_devices(raw)
    expected_sha256 = hashlib.sha256(raw.encode("ascii")).hexdigest()
    if (
        value.get("devices") != devices
        or value.get("device_count") != 4
        or value.get("sha256") != expected_sha256
        or value.get("mapping_preserved") is not True
        or value.get("hip_visible_devices_nonempty") is not False
        or value.get("cuda_visible_devices_nonempty") is not False
    ):
        fail("Slurm ROCR visible-device receipt differs")
    return value


def validate_slurm_visible_device_environment(
    environ: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    env = os.environ if environ is None else environ
    raw = env.get("ROCR_VISIBLE_DEVICES")
    devices = _parse_slurm_rocr_visible_devices(raw)
    if env.get("HIP_VISIBLE_DEVICES", ""):
        fail("nonempty HIP_VISIBLE_DEVICES would override the Slurm ROCR mapping")
    if env.get("CUDA_VISIBLE_DEVICES", ""):
        fail("nonempty CUDA_VISIBLE_DEVICES would override the Slurm ROCR mapping")
    digest = hashlib.sha256(raw.encode("ascii")).hexdigest()
    if (
        env.get("ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES") != raw
        or env.get("ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_SHA256") != digest
        or env.get("ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_COUNT") != "4"
        or env.get("ACTION_REPR_T0_SLURM_ROCR_VISIBLE_DEVICES_PRESERVED") != "true"
    ):
        fail("Slurm ROCR visible-device launcher audit differs")
    receipt = {
        "raw": raw,
        "devices": devices,
        "device_count": 4,
        "sha256": digest,
        "mapping_preserved": True,
        "hip_visible_devices_nonempty": False,
        "cuda_visible_devices_nonempty": False,
    }
    validate_slurm_visible_device_receipt(receipt)
    return receipt


def reject_forbidden_media_fields(value: Any, *, label: str) -> None:
    """Reject media-bearing receipt additions before an optimizer can exist."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in FORBIDDEN_MEDIA_KEYS:
                fail(f"{label} contains forbidden target/anchor media field: {key}")
            reject_forbidden_media_fields(nested, label=label)
    elif isinstance(value, list):
        for nested in value:
            reject_forbidden_media_fields(nested, label=label)


@dataclass(frozen=True)
class FixedFitCase:
    manifest_path: Path
    manifest_sha256: str
    case_id: str
    instruction: str
    seed: int
    source_path: Path
    source_sha256: str


def load_fixed_fit_case(manifest_value: Path | str) -> FixedFitCase:
    path, manifest, digest = g2a_world4.read_json(
        manifest_value, label="T0 experiment manifest"
    )
    if manifest.get("schema_version") != "mev-target-selfgen-flow-calibration-manifest-v1":
        fail("T0 manifest schema differs")
    cases = manifest.get("cases")
    splits = manifest.get("splits")
    if not isinstance(cases, list) or not isinstance(splits, Mapping):
        fail("T0 manifest cases/splits are absent")
    selected = [row for row in cases if isinstance(row, Mapping) and row.get("case_id") == FIXED_CASE_ID]
    fit = splits.get("fit")
    heldout = splits.get("heldout")
    if (
        len(selected) != 1
        or selected[0].get("split") != "fit"
        or not isinstance(fit, list)
        or fit.count(FIXED_CASE_ID) != 1
        or not isinstance(heldout, list)
        or FIXED_CASE_ID in heldout
    ):
        fail("initial T0 canary must be the uniquely registered fit case, never heldout")
    row = selected[0]
    source = row.get("source")
    instruction = row.get("instruction")
    seed = row.get("seed")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"path", "sha256"}
        or type(instruction) is not str
        or not instruction
        or instruction != instruction.strip()
        or type(seed) is not int
        or seed < 0
    ):
        fail("fixed T0 fit-case source/instruction/seed authority differs")
    source_sha = g2a_world4.require_sha256(source.get("sha256"), label="manifest source")
    return FixedFitCase(
        manifest_path=path,
        manifest_sha256=digest,
        case_id=FIXED_CASE_ID,
        instruction=instruction,
        seed=seed,
        source_path=Path(str(source["path"])).expanduser().absolute(),
        source_sha256=source_sha,
    )


@dataclass(frozen=True)
class PreoptimizerAuthority:
    case: FixedFitCase
    g1: g2a_world4.TargetG1Authority
    g1_receipt_sha256: str
    g2a_path: Path
    g2a_file_sha256: str
    g2a_receipt: Mapping[str, Any]
    sigma_index: int
    authorization_path: Optional[Path] = None
    authorization_sha256: Optional[str] = None
    authorization: Optional[Mapping[str, Any]] = None


def _projection_authority_row(
    authority: g2a_world4.TargetG1Authority,
) -> Mapping[str, Any]:
    ref = authority.middle_receipt.get("external_caches", {}).get("correct")
    if not isinstance(ref, Mapping):
        fail("G1 correct middle projection authority is absent")
    _, upstream, _ = g2a_world4.read_json(
        ref.get("receipt_path"),
        label="correct middle upstream receipt",
        expected_sha256=ref.get("receipt_sha256"),
    )
    row = upstream.get("representation", {}).get("projection")
    if (
        not isinstance(row, Mapping)
        or row.get("kind") != "case_independent_fixed_rademacher_jl"
        or row.get("fitted_on_input_video") is not False
        or row.get("width") != authority.projection_width
        or type(row.get("seed")) is not int
        or type(row.get("sha256")) is not str
    ):
        fail("fixed JL projection authority differs")
    g2a_world4.require_sha256(row["sha256"], label="fixed JL projection")
    return dict(row)


def validate_authorization_addendum(
    value: Mapping[str, Any],
    *,
    addendum_path: Path,
    case: FixedFitCase,
    g1_path: Path,
    g1_sha256: str,
    g2a_path: Path,
    g2a_sha256: str,
    g2a_receipt_digest: str,
    projection: Mapping[str, Any],
    bernini_root: Path | str,
    veomni_root: Path | str,
    checkpoint: Path | str,
) -> Mapping[str, Any]:
    """Validate the independent, frozen Stage-B grant before optimizer setup."""

    if not isinstance(value, Mapping):
        fail("Stage-B authorization addendum must be an object")
    row = dict(value)
    expected = {
        "schema_version",
        "experiment_id",
        "created_on",
        "document_role",
        "activation",
        "canonical_preregistration",
        "prior_quantized_energy_match_addendum",
        "upstream_gate_evidence",
        "canary_scope",
        "representation_contract",
        "counterfactual_gradient_contract",
        "optimizer_contract",
        "parameter_firewall",
        "distributed_contract",
        "source_hash_pins",
        "runtime_paths",
        "output_contract",
        "next_order_boundary",
        "claim_boundary",
    }
    if set(row) != expected:
        fail("Stage-B authorization field closure differs")
    activation = row.get("activation")
    upstream = row.get("upstream_gate_evidence")
    canary = row.get("canary_scope")
    representation = row.get("representation_contract")
    counterfactual = row.get("counterfactual_gradient_contract")
    optimizer_row = row.get("optimizer_contract")
    firewall = row.get("parameter_firewall")
    distributed = row.get("distributed_contract")
    output_contract = row.get("output_contract")
    next_order = row.get("next_order_boundary")
    runtime_paths = row.get("runtime_paths")
    pins = row.get("source_hash_pins")
    expected_pin_paths = {
        "methods/bernini_action_editing/train_action_repr_target_t0_canary_v1.py",
        "methods/bernini_action_editing/action_representation_joint_objective_v1.py",
        "methods/bernini_action_editing/action_repr_g2a_adapter_v1.py",
        "methods/bernini_action_editing/audit_action_repr_g2a_world4_v1.py",
        "methods/bernini_action_editing/materialize_decoded_middle_action_repr_v1.py",
        "methods/bernini_action_editing/dense_flow_token_adapter_v1.py",
        "methods/bernini_action_editing/exact_local_video_materializer_v1.py",
        "methods/bernini_action_editing/train_lora.py",
        "methods/bernini_action_editing/train_self_generated_action_quotient_v1.py",
        "methods/bernini_action_editing/scripts/auh_stage_b_t0_single_update_20260824_retry5.sh",
        "methods/bernini_action_editing/tests/test_train_action_repr_target_t0_canary_v1.py",
        "tests/test_auh_stage_b_t0_single_update_20260824_v1.py",
    }
    source_root = METHOD_ROOT.parents[1]
    if not isinstance(pins, Mapping) or set(pins) != expected_pin_paths:
        fail("Stage-B source hash pin closure differs")
    for relative, expected_sha in pins.items():
        g2a_world4.require_sha256(expected_sha, label=f"Stage-B source pin {relative}")
        path = source_root / relative
        if not path.is_file() or file_sha256(path.resolve()) != expected_sha:
            fail(f"Stage-B source hash pin differs: {relative}")
    fixed_jl = representation.get("fixed_jl") if isinstance(representation, Mapping) else None
    phase = representation.get("phase_activity") if isinstance(representation, Mapping) else None
    energy = representation.get("energy_rms") if isinstance(representation, Mapping) else None
    if (
        row.get("schema_version")
        != "bernini-action-repr-stage-b-t0-single-update-authority-addendum-v1"
        or row.get("experiment_id") != "action_repr_target_selfgen_middle_g1_20260824_v2"
        or row.get("created_on") != "2026-08-24"
        or row.get("document_role")
        != "create_once_stage_b_target_t0_single_update_authority"
        or not isinstance(activation, Mapping)
        or activation
        != {
            "state": "ACTIVE_CREATE_ONCE_AUTHORITY",
            "create_once": True,
            "all_placeholders_replaced": True,
            "activation_rule": (
                "copy_to_stage_b_t0_single_update_authority_addendum.json_"
                "once_only_after_runner_tests_and_launcher_are_final_then_"
                "replace_every_explicit_sha256_placeholder_and_set_state_"
                "ACTIVE_CREATE_ONCE_AUTHORITY"
            ),
            "template_itself_authorizes_optimizer_creation": False,
        }
        or row.get("canonical_preregistration")
        != {
            "path": "stage1_v2_preregistration.json",
            "sha256": "294168e596212bd61e8d555e72702ceeeb993fb18c7fa7536a43d0b00ad592b3",
            "modified": False,
        }
        or row.get("prior_quantized_energy_match_addendum")
        != {
            "path": "stage1_v2_quantized_energy_match_addendum.json",
            "sha256": "39a2879c35bdc0fc87c67f05adc11e5766f7dae61792c75f44653450b7ee04da",
            "modified": False,
        }
        or not isinstance(upstream, Mapping)
        or upstream.get("manifest")
        != {"path": str(case.manifest_path), "sha256": case.manifest_sha256}
        or upstream.get("g1_target")
        != {
            "path": str(g1_path),
            "receipt_sha256": g1_sha256,
            "required_status": "passed",
            "selfgen_status_required": "not_evaluated",
            "optimizer_creation_authorized_by_receipt": False,
        }
        or upstream.get("production_g2a")
        != {
            "path": str(g2a_path),
            "receipt_sha256": g2a_sha256,
            "receipt_digest": g2a_receipt_digest,
            "required_status": "PASSED",
            "six_routes_exact_native_bits_required": True,
            "renderer_base_unchanged_required": True,
            "optimizer_created": False,
            "optimization_steps": 0,
        }
        or upstream.get("required_order_closed_before_optimizer")
        != ["G0_integrity", "G1_target_8_of_8_flow_AND_middle", "production_WORLD4_six_route_G2a"]
        or not isinstance(canary, Mapping)
        or any(
            canary.get(name) is not expected_value
            for name, expected_value in {
                "parameter_updates_required": True,
                "target_representation": True,
                "TP": False,
                "sourcecopy": False,
                "selfgen": False,
                "graph": False,
                "automatic_expansion": False,
                "longer_training_authorized": False,
                "decode_authorized_by_this_addendum": False,
            }.items()
        )
        or canary.get("arm") != "T0"
        or canary.get("case_id") != FIXED_CASE_ID
        or canary.get("case_split") != "fit"
        or canary.get("world_size") != 4
        or canary.get("sequence_parallel_size") != 4
        or canary.get("optimization_steps") != 1
        or not isinstance(representation, Mapping)
        or representation.get("accepted_payloads")
        != [
            "authenticated_detached_dense_flow",
            "authenticated_detached_projected_action_minus_noop_middle_residual",
        ]
        or representation.get("target_rgb_allowed_in_trainer") is not False
        or representation.get("target_video_cli_argument_allowed") is not False
        or representation.get("target_vae_or_clean_latent_allowed_in_trainer") is not False
        or representation.get("absolute_target_hidden_or_qkv_allowed_in_trainer") is not False
        or fixed_jl
        != {
            "kind": projection["kind"],
            "seed": projection["seed"],
            "input_width": HIDDEN_WIDTH,
            "output_width": projection["width"],
            "tensor_sha256": projection["sha256"],
            "fitted_on_input_video": False,
            "applied_differentiably_to_student_trace": True,
            "teacher_cache_remains_detached": True,
        }
        or phase != {
            "phase0_active": False,
            "onset_active": True,
            "terminal_active": True,
            "phase0_may_be_relabelled_active": False,
        }
        or energy != {
            "stable_floor_used_for_backward": True,
            "raw_rms_recorded_separately": True,
            "exact_zero_student_nonzero_teacher_requires_finite_nonzero_gradient": True,
        }
        or not isinstance(counterfactual, Mapping)
        or counterfactual.get("required_controls_in_order")
        != ["zero", "temporal_shuffle", "reverse", "incomplete", "wrong_action"]
        or counterfactual.get("no_grad_hinge_prepass") is not True
        or counterfactual.get("correct_side_gradient_passes") != 1
        or counterfactual.get("separate_control_gradient_passes") != 5
        or counterfactual.get("detached_control_scores_without_control_side_gradient_are_sufficient") is not False
        or counterfactual.get("all_gradients_finite") is not True
        or counterfactual.get("all_control_hinges_and_gradient_passes_must_be_receipted") is not True
        or not isinstance(optimizer_row, Mapping)
        or optimizer_row.get("optimizer_creation_authorized_by_final_active_addendum") is not True
        or optimizer_row.get("kind") != "AdamW"
        or optimizer_row.get("learning_rate") != LEARNING_RATE
        or optimizer_row.get("weight_decay") != 0.0
        or optimizer_row.get("steps_exact") != 1
        or optimizer_row.get("gradient_norm_max") != 1.0
        or optimizer_row.get("state_before_and_after_required") is not True
        or optimizer_row.get("at_least_one_allowlisted_parameter_element_must_change") is not True
        or optimizer_row.get("second_step_forbidden") is not True
        or optimizer_row.get("resume_forbidden") is not True
        or not isinstance(firewall, Mapping)
        or firewall.get("base_generator_frozen") is not True
        or firewall.get("vae_frozen") is not True
        or firewall.get("text_encoder_frozen") is not True
        or firewall.get("lora_enabled") is not False
        or firewall.get("trainable_roles_exact") != ["motion_adapter", "middle_projector"]
        or firewall.get("optimizer_parameter_ids_must_equal_exact_allowlist") is not True
        or firewall.get("base_parameter_identity_version_and_bytes_unchanged_required") is not True
        or not isinstance(distributed, Mapping)
        or distributed.get("world_size") != 4
        or distributed.get("sequence_parallel_size") != 4
        or distributed.get("runtime_backend") != "nccl/rccl"
        or distributed.get("rank0_initial_parameter_broadcast") is not True
        or distributed.get("explicit_gradient_all_reduce") is not True
        or distributed.get("post_update_parameter_digest_consensus") is not True
        or distributed.get("rank_divergence_fails_closed") is not True
        or distributed.get("slurm_rocr_visible_devices_mapping_preserved") is not True
        or distributed.get("slurm_rocr_visible_devices_mapping_receipted") is not True
        or distributed.get("slurm_rocr_visible_devices_exact_device_count") != 4
        or distributed.get("slurm_rocr_visible_devices_physical_range_inclusive") != [0, 7]
        or distributed.get("hip_visible_devices_must_be_empty") is not True
        or distributed.get("cuda_visible_devices_must_be_empty") is not True
        or not isinstance(runtime_paths, Mapping)
        or runtime_paths.get("bernini_root") != str(Path(bernini_root).expanduser().absolute())
        or runtime_paths.get("veomni_root") != str(Path(veomni_root).expanduser().absolute())
        or runtime_paths.get("checkpoint") != str(Path(checkpoint).expanduser().absolute())
        or runtime_paths.get("fresh_source_root_name") != "source_stage_b_t0_retry5"
        or runtime_paths.get("fresh_stage_root_name") != "stage_b_t0_retry5"
        or runtime_paths.get("fresh_log_root_name") != "logs/stage_b_t0_retry5"
        or not isinstance(output_contract, Mapping)
        or output_contract.get("fresh_create_only") is not True
        or output_contract.get("step0_and_step1_adapter_states_required") is not True
        or output_contract.get("receipt_create_only") is not True
        or output_contract.get("receipt_must_bind_authority_file_sha256") is not True
        or output_contract.get("checkpoint_contains_allowlisted_adapter_state_only") is not True
        or output_contract.get("trained_video_created") is not False
        or output_contract.get("review_webpage_created") is not False
        or not isinstance(next_order, Mapping)
        or next_order.get("immediate_next_if_single_update_passes")
        != "matched_step0_step1_decode_quality_gate"
        or next_order.get("longer_T0_before_matched_decode") is not False
        or next_order.get("TP_authorized") is not False
        or next_order.get("selfgen_requires_independent_G1") is not True
        or next_order.get("graph_requires_independent_G3") is not True
        or row.get("claim_boundary")
        != "optimizer_integration_canary_only_not_ours_not_quality_not_decoded_video"
    ):
        fail("Stage-B T0 authorization contract differs")
    reject_forbidden_media_fields(value, label="Stage-B authorization addendum")
    if addendum_path.name == "receipt.json":
        fail("authorization addendum must be distinct from an experiment receipt")
    return value


def authorize_preoptimizer_inputs(
    *,
    manifest: Path | str,
    g1_admission_receipt: Path | str,
    g2a_receipt: Path | str,
    authorization_addendum: Path | str,
    bernini_root: Path | str,
    veomni_root: Path | str,
    checkpoint: Path | str,
    output: Path | str,
) -> PreoptimizerAuthority:
    """Close every data/receipt gate.  This function cannot create an optimizer."""

    output_path = Path(output).expanduser().absolute()
    if output_path.exists() or output_path.is_symlink():
        fail("T0 output must be a fresh path")
    case = load_fixed_fit_case(manifest)
    _, raw_g1, g1_file_sha = g2a_world4.read_json(
        g1_admission_receipt, label="passed target G1 receipt"
    )
    reject_forbidden_media_fields(raw_g1, label="G1 receipt")
    try:
        authority = g2a_world4.resolve_target_g1_authority(
            g1_admission_receipt, case_id=FIXED_CASE_ID
        )
    except Exception as error:
        raise TargetT0CanaryError("target G1 receipt did not replay") from error
    if authority.admission_sha256 != g1_file_sha:
        fail("G1 authority byte binding differs")
    g2a_path, g2a_value, g2a_file_sha = g2a_world4.read_json(
        g2a_receipt, label="passed production G2a receipt"
    )
    reject_forbidden_media_fields(g2a_value, label="G2a receipt")
    try:
        g2a_world4.validate_world4_receipt(g2a_value)
    except Exception as error:
        raise TargetT0CanaryError("production G2a receipt did not validate") from error
    g2a_g1 = g2a_value.get("g1_authority")
    runtime = g2a_value.get("runtime")
    source = g2a_value.get("source_owned_native_input")
    training = g2a_value.get("training_authority")
    sigma_index = runtime.get("selected_sigma_index") if isinstance(runtime, Mapping) else None
    if (
        g2a_value.get("passed") is not True
        or g2a_value.get("case_id") != FIXED_CASE_ID
        or not isinstance(g2a_g1, Mapping)
        or g2a_g1.get("admission_sha256") != g1_file_sha
        or g2a_g1.get("evaluation_sha256") != authority.evaluation_sha256
        or g2a_g1.get("flow_cohort_sha256") != authority.flow_cohort_sha256
        or g2a_g1.get("middle_cohort_sha256") != authority.middle_cohort_sha256
        or not isinstance(runtime, Mapping)
        or runtime.get("world_size") != 4
        or runtime.get("ulysses_size") != 4
        or runtime.get("exact_transformer_block_count") != 30
        or runtime.get("hidden_width") != HIDDEN_WIDTH
        or type(sigma_index) is not int
        or not 0 <= sigma_index < len(authority.sigmas)
        or float(runtime.get("selected_sigma", -1.0)) != authority.sigmas[sigma_index]
        or not isinstance(source, Mapping)
        or source.get("source_video_sha256") != case.source_sha256
        or source.get("source_video_sha256") != authority.source_video_sha256
        or source.get("target_or_anchor_media_accessed") is not False
        or training
        != {
            "optimizer_created": False,
            "backward_calls": 0,
            "optimization_steps": 0,
            "parameter_updates": 0,
            "stage_b_training_started": False,
            "optimizer_authorized_by_this_receipt": False,
        }
    ):
        fail("G1/G2a/manifest T0 authority cross-binding differs")
    if hashlib.sha256(case.instruction.encode("utf-8")).hexdigest() != authority.instruction_sha256:
        fail("manifest instruction differs from authenticated target representation")
    source_path = g2a_world4.regular_file(case.source_path, label="manifest source video")
    if file_sha256(source_path) != case.source_sha256:
        fail("manifest source video bytes differ")
    projection_row = _projection_authority_row(authority)
    authorization_path, authorization, authorization_sha = g2a_world4.read_json(
        authorization_addendum, label="frozen Stage-B T0 authorization addendum"
    )
    external_seal = os.environ.get("STAGE_B_T0_AUTHORITY_SHA256", "")
    g2a_world4.require_sha256(external_seal, label="external Stage-B T0 authority seal")
    if external_seal != authorization_sha:
        fail("external Stage-B T0 authority seal differs from addendum bytes")
    validate_authorization_addendum(
        authorization,
        addendum_path=authorization_path,
        case=case,
        g1_path=authority.admission_path,
        g1_sha256=g1_file_sha,
        g2a_path=g2a_path,
        g2a_sha256=g2a_file_sha,
        g2a_receipt_digest=g2a_value["receipt_digest"],
        projection=projection_row,
        bernini_root=bernini_root,
        veomni_root=veomni_root,
        checkpoint=checkpoint,
    )
    return PreoptimizerAuthority(
        case=replace(case, source_path=source_path),
        g1=authority,
        g1_receipt_sha256=g1_file_sha,
        g2a_path=g2a_path,
        g2a_file_sha256=g2a_file_sha,
        g2a_receipt=g2a_value,
        sigma_index=sigma_index,
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha,
        authorization=authorization,
    )


def resolve_fixed_projection(
    authority: g2a_world4.TargetG1Authority, *, device: torch.device
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    ref = authority.middle_receipt["external_caches"]["correct"]
    projection_row = _projection_authority_row(authority)
    projection = middle_extractor.deterministic_projection(
        HIDDEN_WIDTH,
        authority.projection_width,
        seed=projection_row["seed"],
        device=device,
    )
    observed = middle_extractor.tensor_sha256(projection)
    if observed != projection_row.get("sha256"):
        fail("reconstructed fixed JL projection SHA-256 differs")
    return projection, {
        "kind": projection_row["kind"],
        "seed": projection_row["seed"],
        "width": projection_row["width"],
        "sha256": observed,
        "upstream_receipt_sha256": ref["receipt_sha256"],
        "student_native_width": HIDDEN_WIDTH,
        "student_projection_applied_differentiably": True,
    }


def _state_manifest(state: Mapping[str, torch.Tensor]) -> Mapping[str, Any]:
    rows = {
        name: {
            "shape": list(map(int, value.shape)),
            "dtype": str(value.dtype),
            "sha256": g2a.tensor_sha256(value),
        }
        for name, value in sorted(state.items())
    }
    return {"tensor_count": len(rows), "rows": rows, "state_digest": object_sha256(rows)}


def _owned_adapter_state(
    handle: g2a.G2APatchHandle,
) -> Mapping[str, torch.Tensor]:
    """Own state bytes even when the adapter itself is already on CPU."""

    return {
        name: parameter.detach().float().cpu().clone().contiguous()
        for name, parameter in handle.trainable_named_parameters()
    }


def _dist_world_size() -> int:
    import torch.distributed as dist

    return int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1


def _broadcast_parameters(named: Sequence[tuple[str, torch.nn.Parameter]]) -> None:
    import torch.distributed as dist

    if _dist_world_size() > 1:
        for _, parameter in named:
            dist.broadcast(parameter.data, src=0)


def _raw_gradients(
    loss: torch.Tensor,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    retain_graph: bool,
    label: str,
) -> tuple[torch.Tensor, ...]:
    """Materialize one unprojected gradient arm without assigning ``.grad``."""

    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        fail(f"{label} loss must be one scalar tensor")
    raw = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named],
        retain_graph=retain_graph,
        allow_unused=True,
    )
    values = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for (_, parameter), gradient in zip(named, raw)
    )
    for (name, _), gradient in zip(named, values):
        if not bool(torch.isfinite(gradient).all().item()):
            fail(f"{label} raw gradient is non-finite: {name}")
    return values


def _all_reduce_gradient_values(
    gradients: Sequence[torch.Tensor],
    named: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    label: str,
) -> tuple[torch.Tensor, ...]:
    """Average raw gradients before any nonlinear PCGrad/clip operation."""

    import torch.distributed as dist

    values = tuple(gradient.detach().clone() for gradient in gradients)
    if len(values) != len(named):
        fail(f"{label} gradient/parameter closure differs")
    world = _dist_world_size()
    for (name, parameter), gradient in zip(named, values):
        if gradient.shape != parameter.shape or not bool(
            torch.isfinite(gradient).all().item()
        ):
            fail(f"{label} raw gradient geometry/finiteness differs: {name}")
        if world > 1:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(float(world))
        if not bool(torch.isfinite(gradient).all().item()):
            fail(f"{label} all-reduced raw gradient is non-finite: {name}")
    return values


def _consensus(value: Any, *, label: str) -> None:
    import torch.distributed as dist

    if _dist_world_size() == 1:
        return
    rows: list[Any] = [None for _ in range(_dist_world_size())]
    dist.all_gather_object(rows, value)
    if len({canonical_json_bytes(row) for row in rows}) != 1:
        fail(f"WORLD4 ranks disagree on {label}")


def _clone_route_with_trace(
    route: g2a.ActionRepresentationRoute,
    *,
    projection: torch.Tensor,
    optimizer_step: int,
) -> tuple[g2a.ActionRepresentationRoute, g2a.ResidualTraceCollector]:
    trace = g2a.ResidualTraceCollector(
        BLOCK_INDICES,
        gather_sequence_parallel=_dist_world_size() > 1,
        feature_projection=projection,
    )
    return replace(route, optimizer_step=optimizer_step, trace=trace), trace


def _trace_zero_facts(trace: g2a.ResidualTraceCollector) -> Mapping[str, Any]:
    values = trace.require_complete()
    rows = {}
    for index in BLOCK_INDICES:
        value = values[index]
        if bool(torch.count_nonzero(value.detach()).item()):
            fail("step-zero control residual is not exact numeric zero")
        rows[str(index)] = g2a.tensor_sha256(value.detach())
    return {"all_four_projected_traces_exact_zero": True, "trace_sha256s": rows}


def _mean(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        fail("cannot average an empty objective sequence")
    return torch.stack(tuple(values)).mean()


@dataclass(frozen=True)
class OneStepResult:
    step0_state: Mapping[str, torch.Tensor]
    step1_state: Mapping[str, torch.Tensor]
    facts: Mapping[str, Any]


def run_one_step_optimizer_canary(
    *,
    model: torch.nn.Module,
    forward_native: Callable[[], torch.Tensor],
    input_digest: Callable[[], str],
    routes: Mapping[str, g2a.ActionRepresentationRoute],
    feature_projection: torch.Tensor,
    hidden_width: int,
    middle_width: int,
    expected_input_digest: str,
    expected_base_digest: str,
    expected_native_output_digest: str,
    bottleneck_width: int = BOTTLENECK_WIDTH,
    learning_rate: float = LEARNING_RATE,
    adapter_seed: int = ADAPTER_SEED,
    serial_cpu_audit: Callable[[], ContextManager[Any]] = nullcontext,
) -> OneStepResult:
    """Execute one real update with streamed counterfactual gradients."""

    if set(routes) != set(g2a.STEP0_REQUIRED_ROUTES):
        fail("T0 route closure differs")
    if (
        feature_projection.ndim != 2
        or tuple(feature_projection.shape) != (hidden_width, middle_width)
        or feature_projection.requires_grad
        or not bool(torch.isfinite(feature_projection).all().item())
    ):
        fail("T0 differentiable fixed projection geometry differs")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        fail("T0 learning rate must be finite and positive")
    matched_input = g2a_world4.require_sha256(input_digest(), label="source-only FM batch")
    expected_input = g2a_world4.require_sha256(
        expected_input_digest, label="production G2a source-only FM batch"
    )
    if matched_input != expected_input:
        fail("T0 source-only FM batch differs from the passed production G2a batch")
    _consensus(matched_input, label="source-only FM batch digest before optimizer")
    with serial_cpu_audit():
        base_before = g2a_world4.renderer_base_snapshot(model)
    expected_base = g2a_world4.require_sha256(
        expected_base_digest, label="production G2a frozen renderer base"
    )
    if base_before.digest != expected_base:
        fail("T0 frozen renderer base differs from the passed production G2a base")
    expected_native = g2a_world4.require_sha256(
        expected_native_output_digest, label="production G2a native post-head output"
    )

    # Close the G2a/T0 runtime-equivalence gate before a trainable module,
    # gradient graph, or optimizer can exist.
    if input_digest() != matched_input:
        fail("source-only FM batch changed before pre-adapter native forward")
    with torch.inference_mode():
        native_output = forward_native()
    if input_digest() != matched_input:
        fail("source-only FM batch changed during pre-adapter native forward")
    if not isinstance(native_output, torch.Tensor) or not bool(
        torch.isfinite(native_output.detach()).all().item()
    ):
        fail("pre-adapter native output is not one finite tensor")
    native = native_output.detach()
    del native_output
    native_sha = g2a.tensor_sha256(native)
    if native_sha != expected_native:
        fail("T0 native post-head output differs from production G2a")
    _consensus(native_sha, label="production G2a native post-head output before adapter")
    torch.manual_seed(int(adapter_seed))
    handle: Optional[g2a.G2APatchHandle] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    try:
        with serial_cpu_audit():
            handle = g2a.install_action_repr_g2a_adapter(
                model,
                block_indices=BLOCK_INDICES,
                hidden_width=hidden_width,
                flow_width=g2a.DEFAULT_FLOW_WIDTH,
                bottleneck_width=bottleneck_width,
                middle_width=middle_width,
                enable_source_copy_adapter=False,
            )
        named = handle.trainable_named_parameters()
        allowlist = handle.parameter_allowlist()
        if allowlist[g2a.TRAINABLE_ROLES[2]]:
            fail("target T0 must install source_copy=false")
        _broadcast_parameters(named)
        step0_state = _owned_adapter_state(handle)
        step0_manifest = _state_manifest(step0_state)
        _consensus(step0_manifest, label="broadcast step0 adapter state")
        if not handle.output_gates_are_byte_zero():
            fail("T0 step0 adapter output gates are not byte zero")

        def checked_forward(label: str, *, grad: bool) -> torch.Tensor:
            if input_digest() != matched_input:
                fail(f"source-only FM batch changed before {label}")
            context = nullcontext() if grad else torch.inference_mode()
            with context:
                value = forward_native()
            if input_digest() != matched_input:
                fail(f"source-only FM batch changed during {label}")
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value.detach()).all().item()):
                fail(f"{label} output is not finite")
            return value

        route_off_step0 = checked_forward("route_off_step0", grad=False).detach()
        if not g2a.tensor_bits_equal(native, route_off_step0):
            fail("installed zero-init adapter changed the native route-off output")
        del route_off_step0
        control_prepass: dict[str, Any] = {}
        for control_name, route_kind in CONTROL_TO_ROUTE.items():
            traced_route, trace = _clone_route_with_trace(
                routes[route_kind], projection=feature_projection, optimizer_step=0
            )
            with g2a.action_representation_route(traced_route):
                output = checked_forward(f"{route_kind}_step0_prepass", grad=False)
            if not g2a.tensor_bits_equal(native, output):
                fail(f"step0 {route_kind} is not exact route-off bits")
            control_prepass[control_name] = _trace_zero_facts(trace)
            del output, trace, traced_route

        # Determine every hinge from the no-grad, exact-zero student state
        # before any trainable graph or optimizer exists.
        prepass_hinges: dict[str, list[float]] = {
            name: [] for name in CONTROL_TO_ROUTE
        }
        correct_activity = routes["correct"].activity
        if correct_activity is None:
            fail("correct route lacks authenticated action activity")
        with torch.no_grad():
            for index in BLOCK_INDICES:
                teacher_flat = routes["correct"].middle_by_block[index]
                teacher = teacher_flat.reshape(
                    int(teacher_flat.shape[0]),
                    int(routes["correct"].layout.phase_count),
                    int(routes["correct"].layout.tokens_per_phase),
                    int(teacher_flat.shape[-1]),
                ).to(device=feature_projection.device, dtype=torch.float32).detach()
                zero = torch.zeros_like(teacher)
                zero_correct = zero.clone().requires_grad_(True)
                activity = correct_activity.reshape(
                    int(correct_activity.shape[0]),
                    int(routes["correct"].layout.phase_count),
                    int(routes["correct"].layout.tokens_per_phase),
                    1,
                ).to(device=feature_projection.device)
                prepass = objective.compute_joint_objectives(
                    objective.JointObjectiveInputs(
                        student_correct=zero_correct,
                        student_controls={name: zero for name in CONTROL_TO_ROUTE},
                        detached_teacher_correct=teacher,
                        student_route_off=zero,
                        detached_frozen_route_off=zero,
                        action_activity=activity,
                    )
                )
                for name, margin in prepass.diagnostics[
                    "independent_control_margins"
                ].items():
                    prepass_hinges[name].append(float(margin.item()))
                del teacher, zero, zero_correct, activity, prepass
        if any(
            len(values) != len(BLOCK_INDICES)
            or any(not math.isfinite(value) or value <= 0.0 for value in values)
            for values in prepass_hinges.values()
        ):
            fail("all five independent counterfactual hinges must be active at step zero")

        correct_route, correct_trace = _clone_route_with_trace(
            routes["correct"], projection=feature_projection, optimizer_step=0
        )
        with g2a.action_representation_route(correct_route):
            correct_output = checked_forward("correct_step0_gradient", grad=True)
        if not g2a.tensor_bits_equal(native, correct_output.detach()):
            fail("step0 correct route is not exact route-off bits")
        del correct_output, correct_route

        block_results = []
        for index in BLOCK_INDICES:
            student = correct_trace.for_block(index)
            teacher = routes["correct"].middle_by_block[index].to(
                device=student.device, dtype=student.dtype
            ).reshape_as(student).detach()
            zeros = torch.zeros_like(student.detach())
            result = objective.compute_joint_objectives(
                objective.JointObjectiveInputs(
                    student_correct=student,
                    student_controls={name: zeros for name in CONTROL_TO_ROUTE},
                    detached_teacher_correct=teacher,
                    student_route_off=zeros,
                    detached_frozen_route_off=zeros,
                    action_activity=correct_trace.activity_for_block(index),
                )
            )
            block_results.append(result)
        aggregate = objective.JointObjectiveResult(
            action=_mean([row.action for row in block_results]),
            preservation=_mean([row.preservation for row in block_results]),
            action_components={"four_block_mean": _mean([row.action for row in block_results])},
            preservation_components={
                name: _mean([row.preservation_components[name] for row in block_results])
                for name in block_results[0].preservation_components
            },
            diagnostics={},
        )
        correct_activities = {
            index: correct_trace.activity_for_block(index).detach()
            for index in BLOCK_INDICES
        }
        if not bool(torch.isfinite(aggregate.action).item()) or not bool(torch.isfinite(aggregate.preservation).item()):
            fail("T0 correct-side objective is non-finite")

        config = objective.JointObjectiveConfig()
        replay = aggregate.preservation_components.get("exact_zero_route_replay")
        replay_value = float(replay.detach().item()) if isinstance(replay, torch.Tensor) else math.nan
        preservation_value = float(aggregate.preservation.detach().item())
        if (
            not math.isfinite(replay_value)
            or not 0.0 <= replay_value <= float(config.noop_trust_radius)
            or not math.isfinite(preservation_value)
            or not 0.0 <= preservation_value <= float(config.noop_trust_radius)
        ):
            fail("T0 initialization lies outside the no-op preservation trust boundary")
        _consensus(
            {
                "action": float(aggregate.action.detach().item()),
                "preservation": preservation_value,
                "noop_replay": replay_value,
            },
            label="unprojected correct objective scalars",
        )
        correct_action_local = _raw_gradients(
            aggregate.action,
            named,
            retain_graph=True,
            label="correct action",
        )
        preservation_local = _raw_gradients(
            aggregate.preservation,
            named,
            retain_graph=False,
            label="preservation",
        )
        accumulated = list(
            _all_reduce_gradient_values(
                correct_action_local,
                named,
                label="correct action",
            )
        )
        global_preservation = _all_reduce_gradient_values(
            preservation_local,
            named,
            label="preservation",
        )
        if not any(bool(torch.count_nonzero(value).item()) for value in accumulated):
            fail("globally reduced correct action objective produced no gradient")
        del aggregate, block_results, correct_trace

        control_gradient_rows: dict[str, Any] = {}
        for control_name, route_kind in CONTROL_TO_ROUTE.items():
            traced_route, trace = _clone_route_with_trace(
                routes[route_kind], projection=feature_projection, optimizer_step=0
            )
            with g2a.action_representation_route(traced_route):
                routed_output = checked_forward(f"{route_kind}_step0_gradient", grad=True)
            if not g2a.tensor_bits_equal(native, routed_output.detach()):
                fail(f"gradient pass {route_kind} is not exact step0 bits")
            del routed_output, traced_route
            route_actions = []
            for index in BLOCK_INDICES:
                current = trace.for_block(index)
                teacher = routes["correct"].middle_by_block[index].to(
                    device=current.device, dtype=current.dtype
                ).reshape_as(current).detach()
                zero = torch.zeros_like(current.detach())
                detached_correct = zero.clone().requires_grad_(True)
                controls = {name: zero for name in CONTROL_TO_ROUTE}
                controls[control_name] = current
                row = objective.compute_joint_objectives(
                    objective.JointObjectiveInputs(
                        student_correct=detached_correct,
                        student_controls=controls,
                        detached_teacher_correct=teacher,
                        student_route_off=zero,
                        detached_frozen_route_off=zero,
                        # Counterfactual errors are judged on the authenticated
                        # correct action tube.  In particular the zero/no-op
                        # route has no payload activity of its own.
                        action_activity=correct_activities[index],
                    )
                )
                route_actions.append(row.action)
            route_action = _mean(route_actions)
            raw_local = _raw_gradients(
                route_action,
                named,
                retain_graph=False,
                label=f"{control_name} control action",
            )
            raw_global = _all_reduce_gradient_values(
                raw_local,
                named,
                label=f"{control_name} control action",
            )
            local_nonzero = sum(
                int(torch.count_nonzero(value.detach()).item()) for value in raw_local
            )
            global_nonzero = sum(
                int(torch.count_nonzero(value.detach()).item()) for value in raw_global
            )
            for position, value in enumerate(raw_global):
                accumulated[position].add_(value)
            control_gradient_rows[control_name] = {
                "gradient_pass_executed": True,
                "hinge_active_all_four_blocks": all(value > 0.0 for value in prepass_hinges[control_name]),
                "hinge_values": prepass_hinges[control_name],
                "local_nonzero_gradient_elements_before_all_reduce": local_nonzero,
                "global_nonzero_gradient_elements_after_all_reduce": global_nonzero,
                "raw_gradient_all_reduced_before_pcgrad": True,
            }
            del trace, route_actions, route_action, raw_local, raw_global

        action_sq = sum(value.float().square().sum() for value in accumulated)
        if not bool(torch.isfinite(action_sq).item()) or float(action_sq.item()) <= 0.0:
            fail("streamed correct+five-control objective produced no finite gradient")
        preservation_gradient_nonzero = any(
            bool(torch.count_nonzero(value).item()) for value in global_preservation
        )
        projected = objective.project_action_against_preservation_gradients(
            accumulated,
            global_preservation,
            epsilon=float(config.pcgrad_epsilon),
            maximum_gradient_norm=float(config.maximum_gradient_norm),
        )
        pcgrad_facts = {
            **dict(projected.diagnostics),
            "gradient_combination_mode": (
                "global_preservation_priority_pcgrad"
                if preservation_gradient_nonzero
                else "global_initial_zero_preservation_action_only_failsafe"
            ),
            "zero_preservation_gradient_fallback": not preservation_gradient_nonzero,
            "preservation_gradient_nonzero": preservation_gradient_nonzero,
            "preservation_scalar": preservation_value,
            "preservation_scalar_within_noop_trust": True,
            "fallback_establishes_tp_preservation": False,
            "noop_replay": replay_value,
            "noop_replay_within_trust": True,
            "noop_trust_radius": float(config.noop_trust_radius),
            "raw_action_and_preservation_all_reduced_before_pcgrad": True,
            "pcgrad_and_norm_clip_applied_exactly_once_after_global_reduction": True,
        }

        # Optimizer creation occurs only after authority validation, the six
        # exact-noop forwards, all six raw action-gradient passes, global raw
        # gradient reduction, and the one global PCGrad/clip operation.
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            lr=float(learning_rate),
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
        )
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != {id(parameter) for _, parameter in named}:
            fail("optimizer parameter set differs from exact adapter allowlist")
        for (_, parameter), gradient in zip(named, projected.gradients):
            parameter.grad = gradient.detach().clone()
        gradient_manifest = {
            name: g2a.tensor_sha256(parameter.grad.detach())
            for name, parameter in named
            if parameter.grad is not None
        }
        if not any(bool(torch.count_nonzero(parameter.grad).item()) for _, parameter in named):
            fail("all-reduced allowlisted gradient is identically zero")
        _consensus(gradient_manifest, label="all-reduced allowlisted gradients")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step1_state = _owned_adapter_state(handle)
        step1_manifest = _state_manifest(step1_state)
        _consensus(step1_manifest, label="step1 adapter parameters")
        updated_tensors = 0
        updated_elements = 0
        updated_names: list[str] = []
        for name in step0_state:
            changed = step0_state[name] != step1_state[name]
            count = int(torch.count_nonzero(changed).item())
            updated_tensors += int(count > 0)
            updated_elements += count
            if count > 0:
                updated_names.append(name)
        if updated_elements <= 0 or step0_manifest["state_digest"] == step1_manifest["state_digest"]:
            fail("optimizer.step did not update an allowlisted parameter")
        expected_first_step_names = {
            f"blocks.{index}.{g2a.MODULE_NAME}.motion_adapter.output.weight"
            for index in BLOCK_INDICES
        }
        if set(updated_names) != expected_first_step_names:
            fail("T0 first-step changed-parameter closure differs from zero-init theory")
        handle.audit_parameters(deep_base_bytes=True)

        route_off_step1 = checked_forward("route_off_step1", grad=False)
        zero_step1 = replace(routes["zero"], optimizer_step=1)
        with g2a.action_representation_route(zero_step1):
            zero_output = checked_forward("zero_step1", grad=False)
        if not g2a.tensor_bits_equal(native, route_off_step1) or not g2a.tensor_bits_equal(native, zero_output):
            fail("route-off/zero hard bypass changed after optimizer step")
        post_correct, post_trace = _clone_route_with_trace(
            routes["correct"], projection=feature_projection, optimizer_step=1
        )
        with g2a.action_representation_route(post_correct):
            post_correct_output = checked_forward("correct_step1", grad=False)
        residual_nonzero = any(
            bool(torch.count_nonzero(value).item())
            for value in post_trace.require_complete().values()
        )
        if not residual_nonzero:
            fail("correct route remained an internal no-op after the real update")
        post_correct_changed = not g2a.tensor_bits_equal(native, post_correct_output)
        if not post_correct_changed:
            fail("real T0 update did not reach the renderer post-head output")
        del route_off_step1, zero_output, post_correct_output, post_trace, post_correct
        handle.restore()
        handle = None
        with serial_cpu_audit():
            base_after = g2a_world4.renderer_base_snapshot(model)
        if base_before != base_after:
            fail("renderer base identity/version/bytes changed during T0 update")
        facts = {
            "optimizer_created": True,
            "optimization_steps": 1,
            "parameter_updates": updated_elements,
            "updated_parameter_tensors": updated_tensors,
            "updated_parameter_names": sorted(updated_names),
            "first_step_updated_role": "motion_adapter_output_projection_only",
            "middle_projector_parameter_updates": 0,
            "middle_projector_trained_claimed": False,
            "optimizer_kind": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": 0.0,
            "optimizer_parameter_ids_equal_exact_allowlist": True,
            "allowlist_names": [name for name, _ in named],
            "source_copy_adapter_enabled": False,
            "initial_parameter_broadcast": True,
            "gradient_all_reduce": True,
            "world_size": _dist_world_size(),
            "step0_state": step0_manifest,
            "step1_state": step1_manifest,
            "gradient_manifest_digest": object_sha256(gradient_manifest),
            "parameter_digest_consensus": True,
            "global_pcgrad": {
                key: (
                    bool(value)
                    if isinstance(value, bool)
                    else float(value.item())
                    if isinstance(value, torch.Tensor) and value.numel() == 1
                    else str(value)
                )
                for key, value in pcgrad_facts.items()
            },
            "control_no_grad_prepass": control_prepass,
            "control_gradient_passes": control_gradient_rows,
            "control_gradient_pass_count": len(control_gradient_rows),
            "all_five_control_gradient_passes_executed": len(control_gradient_rows) == 5,
            "all_gradients_finite_before_and_after_all_reduce": True,
            "raw_gradients_all_reduced_before_global_pcgrad": True,
            "global_pcgrad_and_norm_clip_count": 1,
            "matched_source_owned_batch_sha256": matched_input,
            "matched_production_g2a_source_batch": matched_input == expected_input,
            "native_step0_output_sha256": native_sha,
            "matched_production_g2a_native_output": native_sha == expected_native,
            "step0_correct_and_five_controls_exact_native": True,
            "route_off_step1_exact_native": True,
            "zero_step1_exact_native": True,
            "correct_step1_internal_residual_nonzero": True,
            "correct_step1_post_head_changed": post_correct_changed,
            "renderer_base_snapshot_digest_before": base_before.digest,
            "renderer_base_snapshot_digest_after": base_after.digest,
            "matched_production_g2a_renderer_base": base_before.digest == expected_base,
            "renderer_base_identity_versions_bytes_unchanged": True,
        }
        del correct_activities
        return OneStepResult(step0_state=step0_state, step1_state=step1_state, facts=facts)
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if handle is not None and not handle.restored:
            handle.restore()


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                fail("create-only publication made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _state_receipt(
    *,
    step: int,
    filename: str,
    file_digest: str,
    state_manifest: Mapping[str, Any],
    authority_digest: str,
    parameter_updates: int,
) -> Mapping[str, Any]:
    row: dict[str, Any] = {
        "schema_version": STEP_SCHEMA_VERSION,
        "optimizer_step": step,
        "adapter_state": {
            "filename": filename,
            "sha256": file_digest,
            "state_digest": state_manifest["state_digest"],
            "tensor_count": state_manifest["tensor_count"],
        },
        "authorization_addendum_sha256": authority_digest,
        "optimizer_created": step == 1,
        "optimization_steps": step,
        "parameter_updates": parameter_updates if step == 1 else 0,
        "create_only": True,
        "quality_success_claimed": False,
    }
    row["receipt_digest"] = object_sha256(row)
    return row


def publish_create_only_result(
    *,
    output: Path | str,
    result: OneStepResult,
    authority: PreoptimizerAuthority,
    projection: Mapping[str, Any],
    runtime: Mapping[str, Any],
    source_lock: Mapping[str, str],
) -> Mapping[str, Any]:
    """Publish into one atomically claimed directory using only O_EXCL files."""

    target = Path(output).expanduser().absolute()
    if target.exists() or target.is_symlink():
        fail("T0 output publication is create-only")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir(mode=0o700)
    except FileExistsError as error:
        raise TargetT0CanaryError("T0 output directory claim lost") from error
    try:
        from safetensors.torch import save as save_safetensors
    except ImportError as error:  # pragma: no cover - production dependency
        raise TargetT0CanaryError("safetensors is required for T0 state publication") from error
    step_rows = {}
    for step, state in ((0, result.step0_state), (1, result.step1_state)):
        step_dir = target / f"step{step:04d}"
        step_dir.mkdir(mode=0o700)
        model_name = "adapter_model.safetensors"
        model_path = step_dir / model_name
        payload = save_safetensors(
            dict(state),
            metadata={
                "schema_version": "bernini-action-repr-target-t0-adapter-state-v1",
                "optimizer_step": str(step),
                "case_id": FIXED_CASE_ID,
                "contains_adapter_allowlist_only": "true",
            },
        )
        _write_exclusive(model_path, payload)
        model_sha = file_sha256(model_path)
        step_receipt = _state_receipt(
            step=step,
            filename=model_name,
            file_digest=model_sha,
            state_manifest=result.facts[f"step{step}_state"],
            authority_digest=str(authority.authorization_sha256),
            parameter_updates=int(result.facts["parameter_updates"]),
        )
        _write_exclusive(
            step_dir / "receipt.json", canonical_json_bytes(step_receipt) + b"\n"
        )
        step_rows[str(step)] = {
            "state_path": f"step{step:04d}/{model_name}",
            "state_sha256": model_sha,
            "state_digest": result.facts[f"step{step}_state"]["state_digest"],
            "receipt_path": f"step{step:04d}/receipt.json",
            "receipt_sha256": file_sha256(step_dir / "receipt.json"),
            "receipt_digest": step_receipt["receipt_digest"],
        }
    receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "complete": True,
            "canary_execution_passed": True,
            "case_id": FIXED_CASE_ID,
            "arm": "T0_target_representation",
            "upstream_authority": {
                "authorization_addendum_sha256": authority.authorization_sha256,
                "manifest_sha256": authority.case.manifest_sha256,
                "g1_target_receipt_sha256": authority.g1_receipt_sha256,
                "production_g2a_receipt_sha256": authority.g2a_file_sha256,
                "production_g2a_receipt_digest": authority.g2a_receipt["receipt_digest"],
                "production_g2a_matched_native_batch_sha256": authority.g2a_receipt[
                    "source_owned_native_input"
                ]["matched_native_batch_sha256"],
                "production_g2a_renderer_base_snapshot_digest": authority.g2a_receipt[
                    "parameter_firewall"
                ]["renderer_base_snapshot_digest_before"],
                "production_g2a_native_post_head_tensor_sha256": authority.g2a_receipt[
                    "parameter_firewall"
                ]["native_post_head_tensor_sha256"],
            },
            "projection": dict(projection),
            "runtime": dict(runtime),
            "training": dict(result.facts),
            "adapter_states": step_rows,
            "information_firewall": {
                "target_or_anchor_media_cli_accepted": False,
                "target_or_anchor_media_opened_by_trainer": False,
                "target_rgb_vae_clean_latent_absolute_hidden_received": False,
                "detached_authenticated_target_action_representation_only": True,
                "source_owned_native_fm_batch_only": True,
                "heldout_entered_optimizer": False,
            },
            "source_lock": dict(source_lock),
            "optimizer_created": True,
            "optimization_steps": 1,
            "parameter_updates": int(result.facts["parameter_updates"]),
            "decoded_video_generated": False,
            "ours_model_claimed": False,
            "quality_success_claimed": False,
            "claim_scope": "one_step_optimizer_execution_canary_only_not_ours_or_quality_success",
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    validate_t0_receipt(receipt)
    # The final receipt is the completion marker and is always published last.
    _write_exclusive(target / "receipt.json", canonical_json_bytes(receipt) + b"\n")
    return receipt


def _validate_state_manifest(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{label} state manifest is absent")
    rows = value.get("rows")
    count = value.get("tensor_count")
    digest = value.get("state_digest")
    g2a_world4.require_sha256(digest, label=f"{label} state")
    if (
        not isinstance(rows, Mapping)
        or type(count) is not int
        or count <= 0
        or count != len(rows)
        or object_sha256(rows) != digest
    ):
        fail(f"{label} state manifest closure differs")
    for name, tensor in rows.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(tensor, Mapping)
            or not isinstance(tensor.get("shape"), list)
            or not tensor["shape"]
            or any(type(size) is not int or size <= 0 for size in tensor["shape"])
            or tensor.get("dtype") != "torch.float32"
        ):
            fail(f"{label} state tensor manifest differs: {name}")
        g2a_world4.require_sha256(
            tensor.get("sha256"), label=f"{label} state tensor {name}"
        )
    return value


def _validate_step_receipt(
    value: Any,
    *,
    step: int,
    authority_sha256: str,
    state_sha256: str,
    state_manifest: Mapping[str, Any],
    parameter_updates: int,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"step {step} receipt must be an object")
    row = dict(value)
    declared = row.pop("receipt_digest", None)
    g2a_world4.require_sha256(declared, label=f"step {step} receipt")
    adapter = row.get("adapter_state")
    if (
        object_sha256(row) != declared
        or set(row)
        != {
            "schema_version",
            "optimizer_step",
            "adapter_state",
            "authorization_addendum_sha256",
            "optimizer_created",
            "optimization_steps",
            "parameter_updates",
            "create_only",
            "quality_success_claimed",
        }
        or row.get("schema_version") != STEP_SCHEMA_VERSION
        or row.get("optimizer_step") != step
        or adapter
        != {
            "filename": "adapter_model.safetensors",
            "sha256": state_sha256,
            "state_digest": state_manifest["state_digest"],
            "tensor_count": state_manifest["tensor_count"],
        }
        or row.get("authorization_addendum_sha256") != authority_sha256
        or row.get("optimizer_created") is not (step == 1)
        or row.get("optimization_steps") != step
        or row.get("parameter_updates") != (parameter_updates if step == 1 else 0)
        or row.get("create_only") is not True
        or row.get("quality_success_claimed") is not False
    ):
        fail(f"step {step} receipt contract differs")
    return value


def validate_t0_receipt(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail("T0 receipt must be an object")
    row = dict(value)
    declared = row.pop("receipt_digest", None)
    g2a_world4.require_sha256(declared, label="T0 receipt")
    if object_sha256(row) != declared:
        fail("T0 receipt digest differs")
    if set(row) != {
        "schema_version",
        "method",
        "complete",
        "canary_execution_passed",
        "case_id",
        "arm",
        "upstream_authority",
        "projection",
        "runtime",
        "training",
        "adapter_states",
        "information_firewall",
        "source_lock",
        "optimizer_created",
        "optimization_steps",
        "parameter_updates",
        "decoded_video_generated",
        "ours_model_claimed",
        "quality_success_claimed",
        "claim_scope",
    }:
        fail("T0 receipt field closure differs")
    authority = row.get("upstream_authority")
    projection = row.get("projection")
    runtime = row.get("runtime")
    training = row.get("training")
    states = row.get("adapter_states")
    firewall = row.get("information_firewall")
    source_lock = row.get("source_lock")
    updates = row.get("parameter_updates")
    expected_updated_names = {
        f"blocks.{index}.{g2a.MODULE_NAME}.motion_adapter.output.weight"
        for index in BLOCK_INDICES
    }
    if not isinstance(authority, Mapping) or set(authority) != {
        "authorization_addendum_sha256",
        "manifest_sha256",
        "g1_target_receipt_sha256",
        "production_g2a_receipt_sha256",
        "production_g2a_receipt_digest",
        "production_g2a_matched_native_batch_sha256",
        "production_g2a_renderer_base_snapshot_digest",
        "production_g2a_native_post_head_tensor_sha256",
    }:
        fail("T0 upstream authority closure differs")
    for name, digest in authority.items():
        g2a_world4.require_sha256(digest, label=f"T0 upstream authority {name}")
    if (
        not isinstance(projection, Mapping)
        or projection.get("kind") != "case_independent_fixed_rademacher_jl"
        or projection.get("seed") != 2026082401
        or projection.get("width") != 256
        or projection.get("student_native_width") != HIDDEN_WIDTH
        or projection.get("student_projection_applied_differentiably") is not True
    ):
        fail("T0 projection receipt differs")
    for name in ("sha256", "upstream_receipt_sha256"):
        g2a_world4.require_sha256(projection.get(name), label=f"T0 projection {name}")
    slurm_rocr = (
        runtime.get("slurm_rocr_visible_devices")
        if isinstance(runtime, Mapping)
        else None
    )
    validate_slurm_visible_device_receipt(slurm_rocr)
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("world_size") != 4
        or runtime.get("ulysses_size") != 4
        or runtime.get("backend") != "nccl/rccl"
        or runtime.get("exact_transformer_block_count") != 30
        or runtime.get("source_owned_native_batch") is not True
        or type(runtime.get("selected_sigma_index")) is not int
        or not isinstance(runtime.get("patch_grid"), list)
        or len(runtime["patch_grid"]) != 3
    ):
        fail("T0 WORLD4 runtime receipt differs")
    for name in ("checkpoint_tree_sha256", "route_facts_digest", "source_posterior_tensor_sha256"):
        g2a_world4.require_sha256(runtime.get(name), label=f"T0 runtime {name}")
    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("method") != METHOD
        or row.get("complete") is not True
        or row.get("canary_execution_passed") is not True
        or row.get("case_id") != FIXED_CASE_ID
        or row.get("arm") != "T0_target_representation"
        or row.get("optimizer_created") is not True
        or row.get("optimization_steps") != 1
        or type(updates) is not int
        or updates <= 0
        or not isinstance(training, Mapping)
        or training.get("optimizer_created") is not True
        or training.get("optimization_steps") != 1
        or training.get("parameter_updates") != updates
        or training.get("updated_parameter_tensors") != len(BLOCK_INDICES)
        or set(training.get("updated_parameter_names", ())) != expected_updated_names
        or training.get("first_step_updated_role")
        != "motion_adapter_output_projection_only"
        or training.get("middle_projector_parameter_updates") != 0
        or training.get("middle_projector_trained_claimed") is not False
        or training.get("optimizer_kind") != "AdamW"
        or training.get("learning_rate") != LEARNING_RATE
        or training.get("weight_decay") != 0.0
        or training.get("optimizer_parameter_ids_equal_exact_allowlist") is not True
        or training.get("source_copy_adapter_enabled") is not False
        or training.get("initial_parameter_broadcast") is not True
        or training.get("gradient_all_reduce") is not True
        or training.get("world_size") != 4
        or training.get("parameter_digest_consensus") is not True
        or training.get("control_gradient_pass_count") != 5
        or training.get("all_five_control_gradient_passes_executed") is not True
        or training.get("all_gradients_finite_before_and_after_all_reduce") is not True
        or training.get("raw_gradients_all_reduced_before_global_pcgrad") is not True
        or training.get("global_pcgrad_and_norm_clip_count") != 1
        or training.get("matched_production_g2a_source_batch") is not True
        or training.get("matched_production_g2a_renderer_base") is not True
        or training.get("matched_production_g2a_native_output") is not True
        or training.get("step0_correct_and_five_controls_exact_native") is not True
        or training.get("renderer_base_identity_versions_bytes_unchanged") is not True
        or training.get("route_off_step1_exact_native") is not True
        or training.get("zero_step1_exact_native") is not True
        or training.get("correct_step1_internal_residual_nonzero") is not True
        or training.get("correct_step1_post_head_changed") is not True
    ):
        fail("T0 optimizer/training receipt contract differs")
    for name in (
        "gradient_manifest_digest",
        "matched_source_owned_batch_sha256",
        "native_step0_output_sha256",
        "renderer_base_snapshot_digest_before",
        "renderer_base_snapshot_digest_after",
    ):
        g2a_world4.require_sha256(training.get(name), label=f"T0 training {name}")
    if (
        training["matched_source_owned_batch_sha256"]
        != authority["production_g2a_matched_native_batch_sha256"]
        or training["native_step0_output_sha256"]
        != authority["production_g2a_native_post_head_tensor_sha256"]
        or training["renderer_base_snapshot_digest_before"]
        != authority["production_g2a_renderer_base_snapshot_digest"]
        or training["renderer_base_snapshot_digest_after"]
        != training["renderer_base_snapshot_digest_before"]
    ):
        fail("T0 training/G2a batch or frozen-base binding differs")
    global_pcgrad = training.get("global_pcgrad")
    if (
        not isinstance(global_pcgrad, Mapping)
        or global_pcgrad.get("raw_action_and_preservation_all_reduced_before_pcgrad") is not True
        or global_pcgrad.get("pcgrad_and_norm_clip_applied_exactly_once_after_global_reduction") is not True
        or global_pcgrad.get("fallback_establishes_tp_preservation") is not False
        or global_pcgrad.get("noop_replay_within_trust") is not True
        or global_pcgrad.get("preservation_scalar_within_noop_trust") is not True
    ):
        fail("T0 global PCGrad receipt differs")
    controls = training.get("control_gradient_passes")
    prepass = training.get("control_no_grad_prepass")
    if (
        not isinstance(controls, Mapping)
        or set(controls) != set(CONTROL_TO_ROUTE)
        or not isinstance(prepass, Mapping)
        or set(prepass) != set(CONTROL_TO_ROUTE)
    ):
        fail("T0 counterfactual receipt closure differs")
    for name, control in controls.items():
        hinges = control.get("hinge_values") if isinstance(control, Mapping) else None
        if (
            not isinstance(control, Mapping)
            or control.get("gradient_pass_executed") is not True
            or control.get("hinge_active_all_four_blocks") is not True
            or control.get("raw_gradient_all_reduced_before_pcgrad") is not True
            or not isinstance(hinges, list)
            or len(hinges) != len(BLOCK_INDICES)
            or any(not math.isfinite(float(item)) or float(item) <= 0.0 for item in hinges)
            or type(control.get("global_nonzero_gradient_elements_after_all_reduce")) is not int
            or control["global_nonzero_gradient_elements_after_all_reduce"] < 0
            or (
                name != "zero_or_noop"
                and control["global_nonzero_gradient_elements_after_all_reduce"] <= 0
            )
        ):
            fail(f"T0 counterfactual gradient receipt differs: {name}")
        prepass_row = prepass[name]
        if (
            not isinstance(prepass_row, Mapping)
            or prepass_row.get("all_four_projected_traces_exact_zero") is not True
            or not isinstance(prepass_row.get("trace_sha256s"), Mapping)
            or set(prepass_row["trace_sha256s"]) != {str(index) for index in BLOCK_INDICES}
        ):
            fail(f"T0 counterfactual prepass receipt differs: {name}")
        for digest in prepass_row["trace_sha256s"].values():
            g2a_world4.require_sha256(digest, label=f"T0 prepass trace {name}")
    step0 = _validate_state_manifest(training.get("step0_state"), label="step0")
    step1 = _validate_state_manifest(training.get("step1_state"), label="step1")
    if step0["state_digest"] == step1["state_digest"]:
        fail("T0 step0/step1 state digests are identical")
    if not isinstance(states, Mapping) or set(states) != {"0", "1"}:
        fail("T0 adapter-state publication closure differs")
    for step, manifest in ((0, step0), (1, step1)):
        state = states[str(step)]
        if (
            not isinstance(state, Mapping)
            or state.get("state_path")
            != f"step{step:04d}/adapter_model.safetensors"
            or state.get("receipt_path") != f"step{step:04d}/receipt.json"
            or state.get("state_digest") != manifest["state_digest"]
        ):
            fail(f"T0 adapter-state row differs: step {step}")
        for name in ("state_sha256", "receipt_sha256", "receipt_digest"):
            g2a_world4.require_sha256(state.get(name), label=f"T0 step {step} {name}")
    if firewall != {
        "target_or_anchor_media_cli_accepted": False,
        "target_or_anchor_media_opened_by_trainer": False,
        "target_rgb_vae_clean_latent_absolute_hidden_received": False,
        "detached_authenticated_target_action_representation_only": True,
        "source_owned_native_fm_batch_only": True,
        "heldout_entered_optimizer": False,
    }:
        fail("T0 information firewall receipt differs")
    expected_source_names = {
        "train_action_repr_target_t0_canary_v1.py",
        "action_repr_g2a_adapter_v1.py",
        "action_representation_joint_objective_v1.py",
        "audit_action_repr_g2a_world4_v1.py",
        "materialize_decoded_middle_action_repr_v1.py",
        "dense_flow_token_adapter_v1.py",
        "exact_local_video_materializer_v1.py",
        "train_lora.py",
        "train_self_generated_action_quotient_v1.py",
    }
    if not isinstance(source_lock, Mapping) or set(source_lock) != expected_source_names:
        fail("T0 runtime source-lock closure differs")
    for name, digest in source_lock.items():
        g2a_world4.require_sha256(digest, label=f"T0 runtime source {name}")
    if (
        row.get("decoded_video_generated") is not False
        or row.get("ours_model_claimed") is not False
        or row.get("quality_success_claimed") is not False
        or row.get("claim_scope")
        != "one_step_optimizer_execution_canary_only_not_ours_or_quality_success"
    ):
        fail("T0 claim boundary differs")
    reject_forbidden_media_fields(value, label="T0 receipt")
    return value


def validate_published_t0_output(output: Path | str) -> Mapping[str, Any]:
    """Replay the complete create-only state/receipt tree from disk."""

    root = Path(output).expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        fail("published T0 output must be one real directory")
    if {path.name for path in root.iterdir()} != {"receipt.json", "step0000", "step0001"}:
        fail("published T0 output file closure differs")
    receipt_path = root / "receipt.json"
    _, receipt, _ = g2a_world4.read_json(receipt_path, label="published T0 receipt")
    validate_t0_receipt(receipt)
    validate_slurm_visible_device_receipt(
        receipt["runtime"]["slurm_rocr_visible_devices"]
    )
    authority_sha = receipt["upstream_authority"]["authorization_addendum_sha256"]
    updates = receipt["parameter_updates"]
    try:
        from safetensors.torch import load_file as load_safetensors
    except ImportError as error:  # pragma: no cover - production dependency
        raise TargetT0CanaryError("safetensors is required for T0 state replay") from error
    for step in (0, 1):
        step_dir = root / f"step{step:04d}"
        if (
            not step_dir.is_dir()
            or step_dir.is_symlink()
            or {path.name for path in step_dir.iterdir()}
            != {"adapter_model.safetensors", "receipt.json"}
        ):
            fail(f"published T0 step {step} file closure differs")
        state_path = step_dir / "adapter_model.safetensors"
        step_receipt_path = step_dir / "receipt.json"
        if state_path.is_symlink() or step_receipt_path.is_symlink():
            fail(f"published T0 step {step} contains a symlink")
        state_row = receipt["adapter_states"][str(step)]
        if (
            file_sha256(state_path) != state_row["state_sha256"]
            or file_sha256(step_receipt_path) != state_row["receipt_sha256"]
        ):
            fail(f"published T0 step {step} file SHA-256 differs")
        _, step_receipt, _ = g2a_world4.read_json(
            step_receipt_path, label=f"published T0 step {step} receipt"
        )
        if step_receipt.get("receipt_digest") != state_row["receipt_digest"]:
            fail(f"published T0 step {step} receipt digest binding differs")
        manifest = _state_manifest(load_safetensors(str(state_path), device="cpu"))
        expected_manifest = receipt["training"][f"step{step}_state"]
        if canonical_json_bytes(manifest) != canonical_json_bytes(expected_manifest):
            fail(f"published T0 step {step} tensor manifest differs")
        _validate_step_receipt(
            step_receipt,
            step=step,
            authority_sha256=authority_sha,
            state_sha256=state_row["state_sha256"],
            state_manifest=expected_manifest,
            parameter_updates=updates,
        )
    return receipt


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-addendum", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--g1-admission-receipt", required=True)
    parser.add_argument("--g2a-receipt", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    slurm_visible_devices = validate_slurm_visible_device_environment()
    authorization = authorize_preoptimizer_inputs(
        manifest=args.manifest,
        g1_admission_receipt=args.g1_admission_receipt,
        g2a_receipt=args.g2a_receipt,
        authorization_addendum=args.authorization_addendum,
        bernini_root=args.bernini_root,
        veomni_root=args.veomni_root,
        checkpoint=args.checkpoint,
        output=args.output,
    )

    import torch.distributed as dist
    from transformers import AutoTokenizer

    import train_lora as legacy
    import train_self_generated_action_quotient_v1 as data

    bernini_root, veomni_root, bernini_revision, veomni_revision = (
        legacy.validate_source_trees(args.bernini_root, args.veomni_root)
    )
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    legacy.activate_source_trees(bernini_root, veomni_root)
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.training.data import NoiseScheduler

    contract = legacy.distributed_contract()
    if contract.world_size != 4 or contract.ulysses_size != 4:
        fail("target T0 canary requires WORLD4/Ulysses-SP4")
    device, backend = legacy.initialise_distributed(contract)
    if (
        backend != "nccl/rccl"
        or dist.get_backend() != "nccl"
        or not getattr(torch.version, "hip", None)
    ):
        fail("Stage-B authorization requires the ROCm NCCL backend contract")
    init_parallel_state(ulysses_size=4)
    if canonical_json_bytes(validate_slurm_visible_device_environment()) != canonical_json_bytes(
        slurm_visible_devices
    ):
        fail("Slurm ROCR visible-device mapping changed during distributed initialization")
    _consensus(
        {
            "authorization_sha256": authorization.authorization_sha256,
            "g1_sha256": authorization.g1_receipt_sha256,
            "g2a_sha256": authorization.g2a_file_sha256,
            "case_id": FIXED_CASE_ID,
            "slurm_rocr_visible_devices": slurm_visible_devices,
        },
        label="preoptimizer authority",
    )

    flow_maps, middle_maps = g2a_world4.load_authenticated_route_cache_maps(
        authorization.g1
    )
    routes, route_facts = g2a_world4.assemble_global_route_payloads(
        authority=authorization.g1,
        flow_maps=flow_maps,
        middle_maps=middle_maps,
        sigma_index=authorization.sigma_index,
    )
    _consensus(route_facts, label="authenticated global route payloads")
    projection, projection_facts = resolve_fixed_projection(
        authorization.g1, device=device
    )
    _consensus(projection_facts, label="fixed JL projection")

    source_blob, posterior_facts = g2a_world4._source_posterior_world4(
        source_video=authorization.case.source_path,
        checkpoint=checkpoint,
        device=device,
        rank=contract.rank,
        max_pixels=245_760,
        stride=16,
        serialized_model_load=data.serialized_model_load,
    )
    _consensus(posterior_facts, label="source-owned posterior")
    posterior_shape = tuple(posterior_facts["posterior_shape"])
    spatial_shape = (
        1,
        16,
        g2a_world4.PHASES,
        int(posterior_shape[3]),
        int(posterior_shape[4]),
    )
    patch_grid = (
        g2a_world4.PHASES,
        int(spatial_shape[-2]) // 2,
        int(spatial_shape[-1]) // 2,
    )
    if patch_grid != authorization.g1.patch_grid:
        fail("source-only batch and target representation patch grids differ")

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    with middle_extractor._model_load_guard(data.serialized_model_load):
        renderer = BerniniRendererModel(config)
        renderer.eval().requires_grad_(False)
        renderer.t5_text_encoder.eval()
        renderer.to(device)
        middle_extractor.trim_runtime_memory(device=device)
    transformer = renderer.diff_dec.transformer
    if transformer is None or len(tuple(getattr(transformer, "blocks", ()))) != 30:
        fail("target T0 requires the exact30 Bernini transformer")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    mean, std, _ = legacy._vae_statistics(checkpoint)
    scheduler = NoiseScheduler(**legacy.noise_scheduler_kwargs())
    transform = data.build_transform(
        tokenizer=tokenizer,
        rope=rope,
        mean=mean,
        std=std,
        scheduler=scheduler,
        device=device,
    )
    native_batch = transform(
        data.make_sample(
            instruction=authorization.case.instruction,
            source_blob=None,
            target_blob=source_blob,
        ),
        authorization.case.seed,
    )
    del source_blob, tokenizer, rope, mean, std, scheduler, transform
    matched = middle_extractor.recover_matched_patch_pair(
        native_batch,
        native_batch,
        spatial_shape=spatial_shape,
        patches_to_spatial=data.patches_to_spatial,
    )
    audit_batch = middle_extractor.retime_fm_batch(
        native_batch,
        clean=matched.action_clean,
        gaussian=matched.gaussian,
        selector=matched.selector,
        sigma=authorization.g1.sigmas[authorization.sigma_index],
    )
    del matched, native_batch

    def forward_native() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return data.predicted_target_velocity(
                renderer, audit_batch, spatial_shape=spatial_shape
            )

    result = run_one_step_optimizer_canary(
        model=renderer,
        forward_native=forward_native,
        input_digest=lambda: g2a_world4.renderer_batch_sha256(audit_batch),
        routes=routes,
        feature_projection=projection,
        hidden_width=HIDDEN_WIDTH,
        middle_width=authorization.g1.projection_width,
        serial_cpu_audit=lambda: middle_extractor._model_load_guard(
            data.serialized_model_load
        ),
        expected_input_digest=authorization.g2a_receipt[
            "source_owned_native_input"
        ]["matched_native_batch_sha256"],
        expected_base_digest=authorization.g2a_receipt["parameter_firewall"][
            "renderer_base_snapshot_digest_before"
        ],
        expected_native_output_digest=authorization.g2a_receipt["parameter_firewall"][
            "native_post_head_tensor_sha256"
        ],
    )
    runtime = {
        "world_size": contract.world_size,
        "ulysses_size": contract.ulysses_size,
        "backend": backend,
        "slurm_rocr_visible_devices": slurm_visible_devices,
        "exact_transformer_block_count": 30,
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
        "checkpoint_tree_sha256": legacy.CHECKPOINT_TREE_SHA256,
        "source_owned_native_batch": True,
        "selected_sigma_index": authorization.sigma_index,
        "selected_sigma": authorization.g1.sigmas[authorization.sigma_index],
        "patch_grid": list(patch_grid),
        "route_facts_digest": object_sha256(route_facts),
        "source_posterior_tensor_sha256": posterior_facts[
            "source_posterior_tensor_sha256"
        ],
    }
    source_lock = {
        Path(__file__).name: file_sha256(Path(__file__).resolve()),
        Path(g2a.__file__).name: file_sha256(Path(g2a.__file__).resolve()),
        Path(objective.__file__).name: file_sha256(Path(objective.__file__).resolve()),
        Path(g2a_world4.__file__).name: file_sha256(Path(g2a_world4.__file__).resolve()),
        Path(middle_extractor.__file__).name: file_sha256(Path(middle_extractor.__file__).resolve()),
        Path(g2a_world4.dense_flow.__file__).name: file_sha256(
            Path(g2a_world4.dense_flow.__file__).resolve()
        ),
        Path(g2a_world4.exact_video.__file__).name: file_sha256(
            Path(g2a_world4.exact_video.__file__).resolve()
        ),
        Path(legacy.__file__).name: file_sha256(Path(legacy.__file__).resolve()),
        Path(data.__file__).name: file_sha256(Path(data.__file__).resolve()),
    }
    if contract.rank == 0:
        publish_create_only_result(
            output=args.output,
            result=result,
            authority=authorization,
            projection=projection_facts,
            runtime=runtime,
            source_lock=source_lock,
        )
    dist.barrier()
    published = validate_published_t0_output(args.output)
    _consensus(published, label="fully replayed published T0 output")
    if contract.rank == 0:
        print(
            json.dumps(
                {
                    "complete": True,
                    "canary_execution_passed": True,
                    "case_id": FIXED_CASE_ID,
                    "optimization_steps": 1,
                    "parameter_updates": published["parameter_updates"],
                    "decoded_video_generated": False,
                    "ours_or_quality_success_claimed": False,
                    "receipt": str(Path(args.output).expanduser().absolute() / "receipt.json"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    del renderer, transformer, audit_batch, routes, flow_maps, middle_maps, projection
    middle_extractor.trim_runtime_memory(device=device)
    dist.barrier()
    dist.destroy_process_group()
    return 0


# Stable launcher-facing name.
validate_receipt = validate_t0_receipt


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIXED_CASE_ID",
    "OneStepResult",
    "PreoptimizerAuthority",
    "TargetT0CanaryError",
    "authorize_preoptimizer_inputs",
    "load_fixed_fit_case",
    "main",
    "publish_create_only_result",
    "reject_forbidden_media_fields",
    "resolve_fixed_projection",
    "run_one_step_optimizer_canary",
    "validate_authorization_addendum",
    "validate_published_t0_output",
    "validate_receipt",
    "validate_slurm_visible_device_environment",
    "validate_slurm_visible_device_receipt",
    "validate_t0_receipt",
]
