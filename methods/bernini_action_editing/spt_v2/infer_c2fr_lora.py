#!/usr/bin/env python3
"""Strict 81-frame projected-bridge C2FR LoRA inference on Bernini UniPC.

This entry point is the adapter counterpart of :mod:`spt_v2.infer_c2fr`.  It
keeps that frozen baseline untouched and changes exactly one model component:
an unmerged LoRA checkpoint trained by ``train_delta_lora.py``'s v4 projected
source/target bridge-consistent clean-field objective is activated at unit
scale.

At every one of the forty official UniPC steps, negative, action, and fixed
semantic-noop branches see one identical noisy state.  The executor applies
the robust Q0 projection ``d(t)-d(0)`` to
``d=action_condition_clean-noop_condition_clean``.  This is precisely the
representation supervised at both bridge endpoints: it removes a
time-constant appearance offset without low-pass filtering the requested
motion.  No binary top-k support is inserted between the dense training field
and dense inference field.  APG is retained as the exact official action-path
certificate and is not used to silently redefine the learned counterfactual
field.

The only external model conditions are an 81-frame source video and an edit
instruction.  There is no target, mask, tracker, optical flow, pose,
trajectory, planner, or first-frame-anchor input.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


SPT_ROOT = Path(__file__).resolve().parent
METHOD_ROOT = SPT_ROOT.parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_delta_lora as adapter_loader  # noqa: E402
import inference_sigma_strata as sigma_strata  # noqa: E402
import motion_residual as motion  # noqa: E402
import train_delta_lora as delta_train  # noqa: E402
import train_lora as trainer  # noqa: E402
import tri_branch_unipc as tri  # noqa: E402
from spt_v2 import generator_native_sparse_router as sparse_router  # noqa: E402
from spt_v2 import infer_c2fr as frozen  # noqa: E402


INFERENCE_RECEIPT_SCHEMA = (
    "bernini-c2fr-projected-bridge-robust-q0-dense-lora-inference-receipt-v4"
)
METHOD_NAME = (
    "projected-bridge-consistent-robust-counterfactual-clean-field-lora-v4"
)
REQUIRED_BRANCH_STATE_MODE = "source_target_bridge_clean_field"
REQUIRED_LORA_SCOPE = "cross_q"
REQUIRED_TARGET_MODULE_COUNT = 30
REQUIRED_MOTION_OBJECTIVE = "causal_boundary_charbonnier"
REQUIRED_MOTION_REPRESENTATION = "source-relative-causal-boundary-charbonnier-v1"
REQUIRED_TARGET_PROJECTION = "executable_target=source+Q0(raw_target-source)"
ADAPTER_SCALE = 1.0
OFFICIAL_LAST_POSITIVE_SIGMA = sigma_strata.PINNED_POSITIVE_SIGMAS[-1]
REQUIRED_INVERSE_SIGMA_WEIGHT_FLOOR = OFFICIAL_LAST_POSITIVE_SIGMA
REQUIRED_BRIDGE_CONSISTENCY_WEIGHT = 0.1
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class C2FRLoRAInferenceError(RuntimeError):
    """Raised before generation when the same-state adapter contract differs."""


def build_parser() -> argparse.ArgumentParser:
    parser = frozen.build_parser()
    parser.description = (
        "Run Bernini-R 1.3B same-state C2FR LoRA on one exact 81-frame source"
    )
    parser.add_argument("--adapter-checkpoint", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    try:
        frozen.validate_cli(args)
    except frozen.C2FRInferenceError as error:
        raise C2FRLoRAInferenceError(str(error)) from error
    if not isinstance(args.adapter_checkpoint, str) or not args.adapter_checkpoint:
        raise C2FRLoRAInferenceError("adapter_checkpoint must be non-empty")
    if float(args.alpha) != ADAPTER_SCALE:
        raise C2FRLoRAInferenceError(
            "formal v4 projected-Q0 inference requires alpha exactly 1.0"
        )
    if (
        float(args.max_generate_fraction) != frozen.DEFAULT_GENERATE_CAP
        or float(args.energy_coverage) != frozen.DEFAULT_ENERGY_COVERAGE
    ):
        raise C2FRLoRAInferenceError(
            "v4 dense inference forbids legacy sparse-router control overrides"
        )


def dense_causal_boundary_runtime_contract() -> dict[str, Any]:
    """Describe the bridge-trained robust-Q0 execution representation."""

    return {
        "method": METHOD_NAME,
        "external_inference_conditions": [
            "source_video",
            "action_instruction",
        ],
        "internal_fixed_controls": [
            "semantic_noop_instruction",
            "negative_prompt",
        ],
        "same_state_input": (
            "raw_action_condition_clean_minus_raw_noop_condition_clean"
        ),
        "execution_field": (
            "dense_causal_boundary_action_minus_noop_clean_field"
        ),
        "execution_field_formula": "delta_exec(t)=delta_raw(t)-delta_raw(0)",
        "target_projection": REQUIRED_TARGET_PROJECTION,
        "first_phase_exact_zero": True,
        "callback_clean_first_phase_bit_exact": True,
        "final_generated_latent_first_phase_bit_exact_claimed": False,
        "decoded_first_frame_bit_exact_claimed": False,
        "temporal_mean_subtraction_at_execution": False,
        "temporal_low_pass_at_execution": False,
        "support_operator": "dense_generator_native_field_no_binary_gate",
        "training_inference_support_operator_aligned": True,
        "counterfactual_formula": "source_clean+alpha*delta_exec",
        "official_apg_role": "parity_certificate_only_not_routed_delta",
        "latent_phases": frozen.base.LATENT_FRAME_COUNT,
        "forbidden_conditions": [
            "target_video",
            "paired_target",
            "mask",
            "track",
            "pose",
            "optical_flow",
            "trajectory",
            "first_frame_anchor",
        ],
        "learned_localizer": False,
        "external_localizer": False,
    }


@dataclass(frozen=True)
class DenseCausalBoundaryStepRecord:
    """Tensor-free certificate for one dense causal-boundary execution."""

    step_index: int
    timestep: float
    sigma: float
    raw_field_rms: float
    causal_field_rms: float
    executed_change_rms: float
    first_phase_max_abs: float


@dataclass
class DenseCausalBoundaryExecutionTrace:
    """All dense field executions made by one official Bernini sample call."""

    alpha: float
    records: list[DenseCausalBoundaryStepRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": METHOD_NAME,
            "alpha": self.alpha,
            "contract": dense_causal_boundary_runtime_contract(),
            "step_count": len(self.records),
            "steps": [asdict(record) for record in self.records],
        }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise C2FRLoRAInferenceError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise C2FRLoRAInferenceError(f"{label} must contain one JSON object")
    return value


def _audited_attention_projection_names() -> list[str]:
    return sorted(
        f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
        for block in range(30)
        for attention in (1, 2)
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    )


def _validate_serialized_target_coverage(
    serialized: Any, *, targets: Sequence[str]
) -> tuple[str, ...]:
    """Validate PEFT's possibly compact suffix list against receipt authority.

    PEFT may serialize exact module names as shorter suffixes.  The runtime
    loader never trusts those suffixes as construction authority: it replaces
    them with the receipt's fully-qualified set.  Nevertheless, every suffix
    must name a receipt target and the suffix set must cover every target.
    """

    if (
        not isinstance(serialized, list)
        or not serialized
        or not all(isinstance(name, str) and name for name in serialized)
        or len(serialized) != len(set(serialized))
    ):
        raise C2FRLoRAInferenceError(
            "adapter serialized target_modules must be a unique non-empty string list"
        )
    target_set = set(targets)
    matched: set[str] = set()
    for suffix in serialized:
        candidates = {
            target
            for target in target_set
            if target == suffix or target.endswith(f".{suffix}")
        }
        if not candidates:
            raise C2FRLoRAInferenceError(
                "adapter serialized target_modules exceed receipt scope"
            )
        matched.update(candidates)
    if matched != target_set:
        raise C2FRLoRAInferenceError(
            "adapter serialized target_modules do not cover the receipt scope"
        )
    return tuple(sorted(serialized))


def validate_same_state_training_adapter_contract(
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Reject legacy/two-state adapters before constructing the renderer."""

    # Constants are checked explicitly so a future alias cannot make a v1
    # receipt acceptable by changing one imported name.
    if receipt.get("schema_version") != "bernini-r-1p3b-c2fr-lora-receipt-v4":
        raise C2FRLoRAInferenceError(
            "only the bridge-consistent C2FR LoRA v4 receipt is supported"
        )
    if receipt.get("method") != (
        "projected-bridge-consistent-robust-counterfactual-clean-field-lora-v4"
    ):
        raise C2FRLoRAInferenceError(
            "only the bridge-consistent robust C2FR LoRA method is supported"
        )

    receipt_without_digest = dict(receipt)
    receipt_digest = receipt_without_digest.pop("receipt_digest", None)
    if (
        not isinstance(receipt_digest, str)
        or _SHA256_RE.fullmatch(receipt_digest) is None
        or trainer.object_sha256(receipt_without_digest) != receipt_digest
    ):
        raise C2FRLoRAInferenceError("training receipt digest differs")
    if receipt.get("bernini_commit") != trainer.BERNINI_OFFICIAL_COMMIT:
        raise C2FRLoRAInferenceError("training Bernini revision differs")
    if receipt.get("veomni_commit") != trainer.VEOMNI_TESTED_COMMIT:
        raise C2FRLoRAInferenceError("training VeOmni revision differs")
    checkpoint = receipt.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("tree_sha256") != expected_checkpoint_tree_sha256
    ):
        raise C2FRLoRAInferenceError("training checkpoint tree differs")
    global_step = receipt.get("global_step")
    if type(global_step) is not int or global_step < sigma_strata.NUM_INFERENCE_STEPS:
        raise C2FRLoRAInferenceError(
            "formal v4 inference requires at least one complete 40-sigma cycle"
        )
    expected_sigma_receipt = sigma_strata.build_sigma_strata_receipt(
        completed_optimizer_steps=global_step
    )
    if receipt.get("inference_sigma_strata") != expected_sigma_receipt:
        raise C2FRLoRAInferenceError(
            "training inference-sigma strata receipt differs"
        )

    adapter = receipt.get("adapter")
    immutable = receipt.get("immutable_contract")
    value = immutable.get("value") if isinstance(immutable, Mapping) else None
    supervision = receipt.get("supervision")
    if not isinstance(adapter, Mapping) or not isinstance(value, Mapping):
        raise C2FRLoRAInferenceError("training receipt lacks adapter/immutable identity")
    if not isinstance(supervision, Mapping):
        raise C2FRLoRAInferenceError("training receipt lacks supervision identity")

    if immutable.get("digest") != trainer.object_sha256(value):
        raise C2FRLoRAInferenceError("training immutable contract digest differs")
    scope = adapter.get("scope")
    targets = adapter.get("target_modules")
    if scope != REQUIRED_LORA_SCOPE:
        raise C2FRLoRAInferenceError(
            "formal v4 training must use the cross_q-only LoRA scope"
        )
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(name, str) for name in targets)
        or targets != sorted(set(targets))
        or value.get("target_modules") != targets
        or value.get("lora_scope") != scope
        or value.get("checkpoint_tree_sha256")
        != expected_checkpoint_tree_sha256
    ):
        raise C2FRLoRAInferenceError(
            "training immutable adapter target scope differs"
        )
    distributed = receipt.get("distributed")
    if (
        not isinstance(distributed, Mapping)
        or distributed.get("world_size") != 4
        or distributed.get("ulysses_size") != 4
        or distributed.get("same_pair_all_ranks") is not True
        or distributed.get("explicit_lora_gradient_all_reduce") is not True
    ):
        raise C2FRLoRAInferenceError(
            "training must use the audited four-rank Ulysses/all-reduce contract"
        )
    if receipt.get("production_claim_forbidden") is not True:
        raise C2FRLoRAInferenceError(
            "training receipt lost production restriction"
        )
    if receipt.get("scientific_claim_authorized") is not False:
        raise C2FRLoRAInferenceError(
            "training receipt carries an unsupported scientific claim"
        )
    dataset = receipt.get("dataset")
    routing = dataset.get("routing") if isinstance(dataset, Mapping) else None
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("rows") != 644
        or not isinstance(routing, Mapping)
        or routing.get("default_tier") != "reject"
        or routing.get("explicit_route_counts")
        != {"full_pair": 0, "motion_only": 359, "reject": 285}
        or routing.get("file_sha256") != value.get("routing_file_sha256")
        or value.get("expected_routing_jsonl_sha256")
        != value.get("routing_file_sha256")
        or value.get("eligible_route_stream_count") != 359
    ):
        raise C2FRLoRAInferenceError(
            "training did not use the hash-bound strict-359 cohort"
        )

    if adapter_config.get("peft_type") != "LORA":
        raise C2FRLoRAInferenceError("adapter is not LoRA")
    if adapter_config.get("r") != trainer.LORA_RANK:
        raise C2FRLoRAInferenceError("adapter rank differs")
    try:
        serialized_alpha = float(adapter_config.get("lora_alpha", -1))
        serialized_dropout = float(adapter_config.get("lora_dropout", -1))
    except (TypeError, ValueError) as error:
        raise C2FRLoRAInferenceError(
            "adapter alpha/dropout are invalid"
        ) from error
    if serialized_alpha != trainer.LORA_ALPHA:
        raise C2FRLoRAInferenceError("adapter alpha differs")
    if serialized_dropout != 0.0:
        raise C2FRLoRAInferenceError("adapter dropout differs")
    if adapter_config.get("bias") != "none":
        raise C2FRLoRAInferenceError("adapter bias differs")
    if adapter_config.get("modules_to_save") not in (None, []):
        raise C2FRLoRAInferenceError("modules_to_save are forbidden")

    targets = list(targets)
    expected_targets = motion.select_lora_scope(
        _audited_attention_projection_names(), REQUIRED_LORA_SCOPE
    )
    if (
        len(expected_targets) != REQUIRED_TARGET_MODULE_COUNT
        or targets != expected_targets
    ):
        raise C2FRLoRAInferenceError(
            "training target modules differ from the audited 30-module cross_q scope"
        )
    try:
        receipt_alpha = float(adapter.get("alpha", -1.0))
    except (TypeError, ValueError) as error:
        raise C2FRLoRAInferenceError("training adapter alpha is invalid") from error
    if (
        adapter.get("rank") != trainer.LORA_RANK
        or receipt_alpha != trainer.LORA_ALPHA
        or adapter.get("target_module_count") != len(expected_targets)
        or adapter.get("target_modules_sha256")
        != trainer.object_sha256(expected_targets)
    ):
        raise C2FRLoRAInferenceError("training adapter scope/rank/hash differs")
    initialization_digest = adapter.get("initialization_digest")
    if (
        not isinstance(initialization_digest, str)
        or _SHA256_RE.fullmatch(initialization_digest) is None
    ):
        raise C2FRLoRAInferenceError("training adapter initialization hash is invalid")
    checkpoint_parameter_digest = adapter.get("checkpoint_parameter_digest")
    if (
        not isinstance(checkpoint_parameter_digest, str)
        or _SHA256_RE.fullmatch(checkpoint_parameter_digest) is None
    ):
        raise C2FRLoRAInferenceError(
            "training adapter checkpoint parameter digest is invalid"
        )

    expected_immutable = {
        "method": (
            "projected-bridge-consistent-robust-counterfactual-clean-field-lora-v4"
        ),
        "branch_state_mode": REQUIRED_BRANCH_STATE_MODE,
        "shared_source_sigma_noise": True,
        "exact_same_noisy_query": True,
        "paired_cells": [
            "source_query_action",
            "source_query_noop",
            "executable_target_query_action",
            "executable_target_query_noop",
        ],
        "posterior_statistic": "mode",
        "clean_reconstruction_formula": "x_clean = y - sigma * velocity",
        "clean_reconstruction_numeric_program": (
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
        ),
        "training_sigma_representation": "cpu_fp32_0d",
        "branch_prediction_dtype_before_clean_reconstruction": "bfloat16",
        "predicted_clean_delta_formula": "-sigma * (v_action - v_noop)",
        "target_clean_delta_formula": "executable_target_clean - source_clean",
        "target_projection_idempotent": True,
        "motion_loss_multiplier": "1 / sigma",
        "copy_boundary_loss_multiplier": "not_enabled",
        "motion_objective": REQUIRED_MOTION_OBJECTIVE,
        "motion_representation": REQUIRED_MOTION_REPRESENTATION,
        "target_projection": REQUIRED_TARGET_PROJECTION,
        "boundary_gauge": (
            "zero_first_latent_phase_of_raw_predicted_clean_delta"
        ),
        "bridge_fractions": [0.0, 1.0],
        "bridge_consistency_weight": REQUIRED_BRIDGE_CONSISTENCY_WEIGHT,
        "causal_ema_decay": None,
        "charbonnier_scale": 0.1,
        "inference_sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "inference_sigma_selector": "absolute_global_step_mod_40",
        "copy_loss_weight": 0.0,
        "boundary_gauge_loss_weight": 0.0,
        "high_noise_floor": 1.0,
        "high_noise_power": 2.0,
    }
    for name, expected in expected_immutable.items():
        if value.get(name) != expected:
            raise C2FRLoRAInferenceError(
                f"training same-state immutable field differs: {name}"
            )
    minimum_sigma = value.get("minimum_training_sigma")
    if (
        isinstance(minimum_sigma, bool)
        or not isinstance(minimum_sigma, (int, float))
        or not 0.0 < float(minimum_sigma) <= OFFICIAL_LAST_POSITIVE_SIGMA
        or not math.isfinite(float(minimum_sigma))
    ):
        raise C2FRLoRAInferenceError(
            "training minimum sigma does not cover the 40-step shift-5 schedule"
        )
    inverse_sigma_weight_floor = value.get("inverse_sigma_weight_floor")
    if (
        isinstance(inverse_sigma_weight_floor, bool)
        or not isinstance(inverse_sigma_weight_floor, (int, float))
        or not math.isfinite(float(inverse_sigma_weight_floor))
        or float(inverse_sigma_weight_floor)
        != REQUIRED_INVERSE_SIGMA_WEIGHT_FLOOR
    ):
        raise C2FRLoRAInferenceError(
            "training inverse-sigma weight floor is invalid"
        )
    expected_weight_range = [1.0, 1.0 / float(inverse_sigma_weight_floor)]
    if value.get("clean_field_loss_weight_range") != expected_weight_range:
        raise C2FRLoRAInferenceError(
            "training inverse-sigma weight range differs"
        )
    boundary_gauge_weight = value.get("boundary_gauge_loss_weight")
    if (
        isinstance(boundary_gauge_weight, bool)
        or not isinstance(boundary_gauge_weight, (int, float))
        or not math.isfinite(float(boundary_gauge_weight))
        or float(boundary_gauge_weight) != 0.0
    ):
        raise C2FRLoRAInferenceError(
            "v4 uses exact execution projection and requires zero soft boundary weight"
        )

    expected_supervision = {
        "inference_conditions": ["source_video", "edit_instruction"],
        "target_used_as_condition": False,
        "external_mask_track_pose_trajectory": False,
        "paired_action_noop_forward_every_optimizer_step": True,
        "action_noop_forwards_per_optimizer_step": 4,
        "counterfactual_noop_forward": True,
        "branch_state_mode": REQUIRED_BRANCH_STATE_MODE,
        "exact_same_noisy_query": True,
        "only_text_condition_differs": True,
        "shared_source_posterior_mode": True,
        "shared_sigma": True,
        "shared_diffusion_noise": True,
        "unreviewed_full_target_weight": 0.0,
        "clean_reconstruction_formula": "x_clean = y - sigma * velocity",
        "clean_reconstruction_numeric_program": (
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
        ),
        "training_sigma_representation": "cpu_fp32_0d",
        "branch_prediction_dtype_before_clean_reconstruction": "bfloat16",
        "predicted_clean_delta_formula": "-sigma * (v_action - v_noop)",
        "target_clean_delta_formula": "executable_target_clean - source_clean",
        "target_projection_idempotent": True,
        "motion_loss_multiplier": "1 / sigma",
        "copy_boundary_loss_multiplier": "not_enabled",
        "motion_objective": REQUIRED_MOTION_OBJECTIVE,
        "causal_boundary_quotient_enabled": True,
        "causal_boundary_projection_enabled": True,
        "target_projection": REQUIRED_TARGET_PROJECTION,
        "temporal_quotient_enabled": False,
        "boundary_gauge_enabled": False,
        "boundary_gauge_field": (
            "raw_predicted_action_minus_noop_clean_field"
        ),
        "boundary_gauge_target": "zero_first_latent_phase",
        "boundary_gauge_uses_target_appearance": False,
        "copy_calibration_enabled": False,
        "copy_calibration_weight": 0.0,
        "bridge_endpoints": [0.0, 1.0],
        "bridge_consistency_enabled": True,
        "bridge_consistency_weight": REQUIRED_BRIDGE_CONSISTENCY_WEIGHT,
        "bridge_query_formula": (
            "y_beta=(1-sigma)*((1-beta)*source+beta*executable_target)"
            "+sigma*epsilon"
        ),
        "causal_ema_enabled": False,
        "causal_ema_decay": None,
        "charbonnier_scale": 0.1,
        "inference_sigma_stratification": "exact_40_step_flow_shift_5_cycle",
        "inference_sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
    }
    for name, expected in expected_supervision.items():
        if supervision.get(name) != expected:
            raise C2FRLoRAInferenceError(
                f"training same-state supervision field differs: {name}"
            )
    try:
        supervision_minimum_sigma = float(
            supervision.get("minimum_training_sigma", -1.0)
        )
        supervision_weight_floor = float(
            supervision.get("inverse_sigma_weight_floor", -1.0)
        )
    except (TypeError, ValueError) as error:
        raise C2FRLoRAInferenceError(
            "training supervision sigma floors are invalid"
        ) from error
    if (
        supervision_minimum_sigma != float(minimum_sigma)
        or supervision_weight_floor != float(inverse_sigma_weight_floor)
        or supervision.get("boundary_gauge_loss_weight")
        != float(boundary_gauge_weight)
        or supervision.get("causal_boundary_gauge_loss_weight")
        != float(boundary_gauge_weight)
    ):
        raise C2FRLoRAInferenceError(
            "training supervision sigma floors differ from immutable contract"
        )

    expected_noop_sha256 = hashlib.sha256(
        motion.DEFAULT_NOOP_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    if value.get("noop_instruction_sha256") != expected_noop_sha256:
        raise C2FRLoRAInferenceError(
            "training no-op instruction differs from fixed C2FR inference control"
        )
    if adapter_config.get("use_dora") not in (None, False):
        raise C2FRLoRAInferenceError("DoRA is outside the C2FR LoRA contract")
    if adapter_config.get("use_rslora") not in (None, False):
        raise C2FRLoRAInferenceError("RS-LoRA is outside the C2FR LoRA contract")
    serialized = _validate_serialized_target_coverage(
        adapter_config.get("target_modules"), targets=targets
    )

    for name, pattern in (
        ("method_source_revision", _SHA1_RE),
        ("method_source_archive_sha256", _SHA256_RE),
    ):
        candidate = value.get(name)
        if not isinstance(candidate, str) or pattern.fullmatch(candidate) is None:
            raise C2FRLoRAInferenceError(f"training {name} is invalid")
    transformers_version = receipt.get("transformers_version")
    if not isinstance(transformers_version, str) or not transformers_version:
        raise C2FRLoRAInferenceError("training Transformers version is missing")

    return {
        "receipt_digest": receipt_digest,
        "global_step": global_step,
        "scope": scope,
        "target_modules_sha256": adapter["target_modules_sha256"],
        "transformers_version": transformers_version,
        "targets": targets,
        "serialized_target_modules": list(serialized),
        "initialization_digest": initialization_digest,
        "checkpoint_parameter_digest": checkpoint_parameter_digest,
        "branch_state_mode": REQUIRED_BRANCH_STATE_MODE,
        "minimum_training_sigma": float(minimum_sigma),
        "inverse_sigma_weight_floor": float(inverse_sigma_weight_floor),
        "boundary_gauge_loss_weight": float(boundary_gauge_weight),
        "bridge_consistency_weight": float(value["bridge_consistency_weight"]),
        "charbonnier_scale": float(value["charbonnier_scale"]),
        "inference_sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "motion_representation": value["motion_representation"],
        "official_last_positive_sigma": OFFICIAL_LAST_POSITIVE_SIGMA,
        "training_method_source_revision": value["method_source_revision"],
        "training_method_source_archive_sha256": value[
            "method_source_archive_sha256"
        ],
    }


def validate_loaded_adapter_parameter_digest(
    model: Any, *, expected_digest: str
) -> str:
    """Hash the exact reloaded LoRA parameters using the trainer's algorithm."""

    if (
        not isinstance(expected_digest, str)
        or _SHA256_RE.fullmatch(expected_digest) is None
    ):
        raise C2FRLoRAInferenceError("expected adapter parameter digest is invalid")
    if not hasattr(model, "named_parameters"):
        raise C2FRLoRAInferenceError("loaded adapter model lacks named_parameters")
    named_lora = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if trainer.is_lora_parameter_name(name)
    ]
    if not named_lora:
        raise C2FRLoRAInferenceError("loaded adapter has no LoRA parameters")
    try:
        actual = delta_train._checkpoint_parameter_digest(named_lora)
    except delta_train.DeltaTrainingError as error:
        raise C2FRLoRAInferenceError(str(error)) from error
    if actual != expected_digest:
        raise C2FRLoRAInferenceError(
            "reloaded LoRA parameter digest differs from training receipt"
        )
    return actual


def _strict_load_same_state_adapter(
    *,
    base_model: Any,
    bundle: Any,
    adapter_config: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> tuple[Any, int, int]:
    """Load exact tensors, then certify the active PEFT module locations."""

    targets = list(identity["targets"])
    _validate_serialized_target_coverage(
        adapter_config.get("target_modules"), targets=targets
    )
    try:
        model, tensor_count = adapter_loader._strict_load_adapter(
            base_model=base_model,
            adapter_dir=bundle.adapter_dir,
            adapter_model_path=bundle.adapter_model_path,
            targets=targets,
        )
    except adapter_loader.DeltaInferenceError as error:
        raise C2FRLoRAInferenceError(str(error)) from error

    mapped: list[str] = []
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        scaling = getattr(module, "scaling", None)
        # PEFT stores A/B in torch.nn.ModuleDict, which intentionally is not a
        # collections.abc.Mapping despite exposing the mapping protocol.
        if lora_a is None and lora_b is None:
            continue
        try:
            complete = (
                lora_a is not None
                and "default" in lora_a
                and lora_b is not None
                and "default" in lora_b
                and scaling is not None
                and "default" in scaling
            )
        except (TypeError, AttributeError):
            complete = False
        if not complete:
            raise C2FRLoRAInferenceError("runtime LoRA module is incomplete")
        matches = [
            target
            for target in targets
            if name == target or name.endswith(f".{target}")
        ]
        if len(matches) != 1:
            raise C2FRLoRAInferenceError(
                f"runtime LoRA module is outside receipt scope: {name}"
            )
        if not math.isclose(
            float(scaling["default"]), ADAPTER_SCALE, rel_tol=0.0, abs_tol=0.0
        ):
            raise C2FRLoRAInferenceError("runtime LoRA scale differs from unit scale")
        mapped.append(matches[0])
    if sorted(mapped) != targets or len(mapped) != len(set(mapped)):
        raise C2FRLoRAInferenceError(
            "runtime active LoRA module set differs from receipt target set"
        )
    if tensor_count != 2 * len(targets):
        raise C2FRLoRAInferenceError("runtime LoRA tensor count differs")
    validate_loaded_adapter_parameter_digest(
        model, expected_digest=identity["checkpoint_parameter_digest"]
    )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise C2FRLoRAInferenceError("inference model contains trainable parameters")
    model.eval()
    return model, tensor_count, len(mapped)


def _tensor_stat(value: Any, *, label: str) -> float:
    try:
        numeric = float(value.detach().float().cpu().item())
    except Exception as error:
        raise C2FRLoRAInferenceError(f"cannot serialize {label}") from error
    if not math.isfinite(numeric) or numeric < 0.0:
        raise C2FRLoRAInferenceError(f"{label} must be finite and non-negative")
    return numeric


def validate_runtime_schedule_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the real inference scheduler to the pinned float32 schedule."""

    expected = {
        "schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "timesteps": list(sigma_strata.PINNED_TIMESTEPS),
        "positive_sigmas": list(sigma_strata.PINNED_POSITIVE_SIGMAS),
        "positive_sigmas_float32_be_hex": list(
            sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
        ),
        "terminal_sigma": 0.0,
        "terminal_sigma_float32_be_hex": (
            sigma_strata.TERMINAL_SIGMA_FLOAT32_HEX
        ),
    }
    if not isinstance(audit, Mapping) or dict(audit) != expected:
        raise C2FRLoRAInferenceError(
            "runtime UniPC schedule audit differs from the pinned 40-step grid"
        )
    return expected


class TracedDenseCausalBoundaryCallback:
    """Execute the trained dense robust-Q0 field and retain diagnostics."""

    def __init__(
        self,
        *,
        source_clean: Any,
        layout: tri.PackedLatentLayout,
        alpha: float,
    ) -> None:
        try:
            alpha_value = float(alpha)
        except (TypeError, ValueError) as error:
            raise C2FRLoRAInferenceError(
                "formal v4 dense callback alpha is invalid"
            ) from error
        if not math.isfinite(alpha_value) or alpha_value != ADAPTER_SCALE:
            raise C2FRLoRAInferenceError(
                "formal v4 dense callback requires alpha exactly 1.0"
            )
        self.source_phase = sparse_router.source_to_phase_video(
            source_clean, layout=layout
        ).float()
        self.layout = layout
        self.alpha = alpha_value
        self.trace = DenseCausalBoundaryExecutionTrace(alpha=self.alpha)

    def __call__(self, fields: tri.CleanFieldStep) -> Any:
        if not isinstance(fields, tri.CleanFieldStep):
            raise C2FRLoRAInferenceError(
                "dense callback requires one tri-branch clean-field step"
            )
        import torch

        action_phase = sparse_router.spatial_to_phase_video(
            fields.action_condition_clean, layout=self.layout
        ).float()
        noop_phase = sparse_router.spatial_to_phase_video(
            fields.noop_condition_clean, layout=self.layout
        ).float()
        raw_field = action_phase - noop_phase
        causal_field = sparse_router.causal_boundary_projection(raw_field)
        executed_phase = self.source_phase + self.alpha * causal_field
        if not bool(torch.isfinite(executed_phase).all()):
            raise C2FRLoRAInferenceError("dense executed clean field is non-finite")
        first_phase_max_abs = _tensor_stat(
            causal_field[:, :1].abs().max(),
            label="causal first-phase maximum",
        )
        if first_phase_max_abs != 0.0:
            raise C2FRLoRAInferenceError(
                "dense causal-boundary field is not exactly zero at phase zero"
            )
        if not bool(
            torch.equal(executed_phase[:, :1], self.source_phase[:, :1])
        ):
            raise C2FRLoRAInferenceError(
                "dense execution did not preserve the source first phase exactly"
            )
        self.trace.records.append(
            DenseCausalBoundaryStepRecord(
                step_index=int(fields.step_index),
                timestep=float(fields.timestep),
                sigma=float(fields.sigma),
                raw_field_rms=_tensor_stat(
                    raw_field.square().mean().sqrt(),
                    label="raw field RMS",
                ),
                causal_field_rms=_tensor_stat(
                    causal_field.square().mean().sqrt(),
                    label="causal field RMS",
                ),
                executed_change_rms=_tensor_stat(
                    (executed_phase - self.source_phase).square().mean().sqrt(),
                    label="executed change RMS",
                ),
                first_phase_max_abs=first_phase_max_abs,
            )
        )
        return sparse_router.phase_video_to_spatial(
            executed_phase, layout=self.layout
        )


def validate_dense_execution_trace(
    tri_trace: tri.TriBranchTrace,
    dense_trace: DenseCausalBoundaryExecutionTrace,
    *,
    runtime_schedule_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Require forty exact APG steps and forty aligned dense executions."""

    if not isinstance(tri_trace, tri.TriBranchTrace):
        raise C2FRLoRAInferenceError("tri_trace must be a TriBranchTrace")
    if not isinstance(dense_trace, DenseCausalBoundaryExecutionTrace):
        raise C2FRLoRAInferenceError(
            "dense_trace must be a DenseCausalBoundaryExecutionTrace"
        )
    branch_records = list(tri_trace.records)
    dense_records = list(dense_trace.records)
    if tri_trace.sample_calls != 1:
        raise C2FRLoRAInferenceError(
            "tri-branch hook must observe exactly one sample call"
        )
    if (
        len(branch_records) != frozen.NUM_INFERENCE_STEPS
        or len(dense_records) != frozen.NUM_INFERENCE_STEPS
    ):
        raise C2FRLoRAInferenceError(
            "dense C2FR must certify all 40 official UniPC steps"
        )
    if dense_trace.alpha != ADAPTER_SCALE:
        raise C2FRLoRAInferenceError(
            "formal v4 dense trace requires alpha exactly 1.0"
        )
    audited_schedule = validate_runtime_schedule_audit(runtime_schedule_audit)
    sigmas: list[float] = []
    for expected_index, (branch, dense) in enumerate(
        zip(branch_records, dense_records)
    ):
        if branch.step_index != expected_index or dense.step_index != expected_index:
            raise C2FRLoRAInferenceError(
                "dense C2FR step indices are incomplete or reordered"
            )
        selected = sigma_strata.select_sigma_stratum(expected_index)
        try:
            sigma_strata.assert_selected_timestep_sigma(
                timestep=branch.timestep,
                sigma=branch.sigma,
                selected=selected,
            )
        except sigma_strata.InferenceSigmaStrataError as error:
            raise C2FRLoRAInferenceError(
                f"runtime UniPC trace differs at schedule index {expected_index}"
            ) from error
        if branch.model_id != "transformer_1":
            raise C2FRLoRAInferenceError(
                "Bernini-R 1.3B must remain single-expert"
            )
        if (
            branch.transformer_forwards != 3
            or branch.shared_negative_forwards != 1
            or branch.action_forwards != 1
            or branch.noop_forwards != 1
            or branch.original_scheduler_calls != 1
        ):
            raise C2FRLoRAInferenceError(
                "each step must use three branches and one original UniPC call"
            )
        if (
            branch.official_action_exact_parity is not True
            or branch.official_action_parity_rms_error != 0.0
            or branch.official_action_parity_max_abs_error != 0.0
        ):
            raise C2FRLoRAInferenceError(
                "official action APG exact certificate failed"
            )
        if branch.effective_guidance_scale != frozen.base.OMEGA_TEXT:
            raise C2FRLoRAInferenceError("action guidance scale differs")
        for label, value in (
            ("sigma", branch.sigma),
            ("callback correction", branch.callback_correction_rms),
            ("raw action-noop delta", branch.raw_action_noop_delta_rms),
            ("guided action-noop delta", branch.guided_action_noop_delta_rms),
            ("guided action-noop L2", branch.guided_action_noop_delta_l2),
        ):
            if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
                raise C2FRLoRAInferenceError(
                    f"tri-branch {label} diagnostic is invalid"
                )
        if branch.sigma <= 0.0:
            raise C2FRLoRAInferenceError(
                "every intercepted UniPC sigma must be positive"
            )
        if branch.sigma != dense.sigma or branch.timestep != dense.timestep:
            raise C2FRLoRAInferenceError(
                "tri-branch and dense execution traces differ"
            )
        for value in (
            dense.raw_field_rms,
            dense.causal_field_rms,
            dense.executed_change_rms,
            dense.first_phase_max_abs,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise C2FRLoRAInferenceError(
                    "dense field diagnostic is invalid"
                )
        if dense.first_phase_max_abs != 0.0:
            raise C2FRLoRAInferenceError(
                "dense causal field first phase differs from exact zero"
            )
        sigmas.append(float(branch.sigma))
    if any(following >= current for current, following in zip(sigmas, sigmas[1:])):
        raise C2FRLoRAInferenceError(
            "official UniPC sigma trace must be strictly descending"
        )
    payload = {
        "tri_branch": tri_trace.as_dict(),
        "dense_causal_boundary": dense_trace.as_dict(),
        "runtime_unipc_schedule_audit": audited_schedule,
        "certificate": {
            "step_count": frozen.NUM_INFERENCE_STEPS,
            "official_action_apg_exact_steps": frozen.NUM_INFERENCE_STEPS,
            "original_unipc_calls": frozen.NUM_INFERENCE_STEPS,
            "transformer_forwards": 3 * frozen.NUM_INFERENCE_STEPS,
            "dense_execution_steps": frozen.NUM_INFERENCE_STEPS,
            "binary_support_operator": False,
            "custom_integrator": False,
        },
    }
    payload["trace_digest"] = frozen.base.object_sha256(payload)
    return payload


def _method_hashes() -> dict[str, str]:
    hashes = frozen._method_hashes()
    paths = {
        "spt_v2/infer_c2fr_lora.py": SPT_ROOT / "infer_c2fr_lora.py",
        "train_delta_lora.py": METHOD_ROOT / "train_delta_lora.py",
        "infer_delta_lora.py": METHOD_ROOT / "infer_delta_lora.py",
        "inference_sigma_strata.py": METHOD_ROOT / "inference_sigma_strata.py",
    }
    hashes.update({name: frozen.base.file_sha256(path) for name, path in paths.items()})
    return hashes


def build_inference_receipt(
    *,
    args: argparse.Namespace,
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    output_path: Path,
    output_sha256: str,
    noop_identity: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    inference_file_hashes: Mapping[str, str],
    wan_diffusion_path: Path,
    wan_diffusion_sha256: str,
    runtime_versions: Mapping[str, str],
    adapter_bundle: Any,
    adapter_identity: Mapping[str, Any],
    adapter_config_sha256: str,
    adapter_model_sha256: str,
    training_receipt_file_sha256: str,
    adapter_tensor_count: int,
    active_lora_module_count: int,
) -> dict[str, Any]:
    runtime_schedule_audit = validate_runtime_schedule_audit(
        execution_trace.get("runtime_unipc_schedule_audit", {})
    )
    receipt = frozen.build_inference_receipt(
        args=args,
        source_path=source_path,
        source_sha256=source_sha256,
        source_metadata=source_metadata,
        output_path=output_path,
        output_sha256=output_sha256,
        noop_identity=noop_identity,
        execution_trace=execution_trace,
        bernini_revision=bernini_revision,
        veomni_revision=veomni_revision,
        inference_file_hashes=inference_file_hashes,
        wan_diffusion_path=wan_diffusion_path,
        wan_diffusion_sha256=wan_diffusion_sha256,
        runtime_versions=runtime_versions,
    )
    receipt.pop("receipt_digest", None)
    receipt["schema_version"] = INFERENCE_RECEIPT_SCHEMA
    receipt["method"] = METHOD_NAME
    receipt["method_files_sha256"] = _method_hashes()
    receipt["base_model"].update(
        {
            "frozen": True,
            "base_weights_frozen": True,
            "lora_or_peft_loaded": True,
            "adapter_loaded": True,
            "all_runtime_parameters_require_grad_false": True,
        }
    )
    receipt["adapter"] = {
        "checkpoint_root": str(adapter_bundle.checkpoint_root),
        "adapter_config_path": str(adapter_bundle.adapter_config_path),
        "adapter_config_sha256": adapter_config_sha256,
        "adapter_model_path": str(adapter_bundle.adapter_model_path),
        "adapter_model_sha256": adapter_model_sha256,
        "training_receipt_path": str(adapter_bundle.training_receipt_path),
        "training_receipt_file_sha256": training_receipt_file_sha256,
        "training_receipt_digest": adapter_identity["receipt_digest"],
        "training_global_step": adapter_identity["global_step"],
        "training_method_source_revision": adapter_identity[
            "training_method_source_revision"
        ],
        "training_method_source_archive_sha256": adapter_identity[
            "training_method_source_archive_sha256"
        ],
        "scope": adapter_identity["scope"],
        "target_module_count": len(adapter_identity["targets"]),
        "target_modules_sha256": adapter_identity["target_modules_sha256"],
        "serialized_target_modules": list(
            adapter_identity["serialized_target_modules"]
        ),
        "initialization_digest": adapter_identity["initialization_digest"],
        "checkpoint_parameter_digest": adapter_identity[
            "checkpoint_parameter_digest"
        ],
        "tensor_count": int(adapter_tensor_count),
        "active_lora_module_count": int(active_lora_module_count),
        "strict_tensor_reload_equal": True,
        "parameter_digest_verified_after_safetensors_reload": True,
        "target_modules_rebound_from_receipt": True,
        "merged": False,
        "scale": ADAPTER_SCALE,
    }
    receipt["training_inference_alignment"] = {
        "training_receipt_schema": delta_train.RECEIPT_SCHEMA,
        "training_method": delta_train.METHOD_NAME,
        "branch_state_mode": REQUIRED_BRANCH_STATE_MODE,
        "training_query": (
            "identical_action_noop_y_at_source_and_target_bridge_endpoints"
        ),
        "inference_query": "identical_y_sigma_source_rope_timestep",
        "only_action_noop_text_differs": True,
        "closed_loop_query_gap_mitigated_by_endpoint_consistency": True,
        "closed_loop_query_gap_proven_closed": False,
        "bridge_endpoints": [0.0, 1.0],
        "bridge_consistency_weight": adapter_identity[
            "bridge_consistency_weight"
        ],
        "training_clean_delta": "-sigma * (v_action - v_noop)",
        "inference_clean_delta": (
            "action_condition_clean - noop_condition_clean"
        ),
        "raw_condition_field_routed": False,
        "causal_boundary_projected_field_executed": True,
        "official_apg_role": "exact_action_path_certificate_only",
        "minimum_training_sigma": adapter_identity["minimum_training_sigma"],
        "official_last_positive_sigma": OFFICIAL_LAST_POSITIVE_SIGMA,
        "training_covers_every_positive_inference_sigma": True,
        "training_sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "runtime_sigma_schedule_sha256": runtime_schedule_audit[
            "schedule_sha256"
        ],
        "runtime_schedule_bit_exact": True,
        "inverse_sigma_weight_floor": adapter_identity[
            "inverse_sigma_weight_floor"
        ],
        "training_motion_loss_multiplier": "1 / sigma",
        "training_copy_boundary_loss_multiplier": "not_enabled",
        "clean_reconstruction_numeric_program": (
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
        ),
        "training_sigma_representation": "cpu_fp32_0d",
        "inference_sigma_representation": "cpu_fp32_0d",
        "branch_prediction_dtype_before_clean_reconstruction": "bfloat16",
        "training_motion_representation": (
            REQUIRED_MOTION_REPRESENTATION
        ),
        "training_motion_objective": REQUIRED_MOTION_OBJECTIVE,
        "target_projection": REQUIRED_TARGET_PROJECTION,
        "training_boundary_gauge": (
            "zero_first_latent_phase_of_raw_predicted_clean_delta"
        ),
        "training_boundary_gauge_loss_weight": adapter_identity[
            "boundary_gauge_loss_weight"
        ],
        "gauge_requires_no_inference_condition": True,
        "inference_field_execution": (
            "dense_causal_boundary_action_minus_noop_clean_field"
        ),
        "inference_field_formula": "d_exec(t)=d_raw(t)-d_raw(0)",
        "first_phase_exact_zero_by_projection": True,
        "callback_clean_first_phase_bit_exact": True,
        "final_generated_latent_first_phase_bit_exact_claimed": False,
        "decoded_first_frame_bit_exact_claimed": False,
        "temporal_mean_subtraction_at_execution": False,
        "temporal_low_pass_at_execution": False,
        "dense_training_and_inference_support_operator": True,
        "binary_support_operator": False,
        "same_noisy_state_gap": False,
        "motion_representation_gap": False,
        "support_execution_gap": False,
    }
    receipt["sampling"].pop("router_config", None)
    receipt["sampling"].update(
        {
            "adapter_scale": ADAPTER_SCALE,
            "adapter_merged": False,
            "raw_condition_clean_delta_routing": False,
            "routing_contract": dense_causal_boundary_runtime_contract(),
            "dense_causal_boundary_clean_delta_execution": True,
            "binary_support_operator": False,
            "temporal_mean_subtraction_at_execution": False,
            "temporal_low_pass_at_execution": False,
            "runtime_unipc_schedule_audit": runtime_schedule_audit,
        }
    )
    receipt["receipt_digest"] = frozen.base.object_sha256(receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    frozen.configure_rank_local_caches()
    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute():
        raise C2FRLoRAInferenceError("source video must be an absolute path")
    try:
        source_path = frozen.base._plain_file(
            source_requested.resolve(strict=True), label="source video"
        )
        output_path, receipt_path = frozen.base._resolve_output(args.output)
        bundle = frozen.base.resolve_adapter_bundle(args.adapter_checkpoint)
    except frozen.base.InferenceContractError as error:
        raise C2FRLoRAInferenceError(str(error)) from error

    adapter_config = _read_json(bundle.adapter_config_path, label="adapter config")
    training_receipt = _read_json(
        bundle.training_receipt_path, label="training receipt"
    )
    adapter_identity = validate_same_state_training_adapter_contract(
        adapter_config,
        training_receipt,
        expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
    )
    if (
        args.method_source_revision
        != adapter_identity["training_method_source_revision"]
        or args.method_source_archive_sha256
        != adapter_identity["training_method_source_archive_sha256"]
    ):
        raise C2FRLoRAInferenceError(
            "inference source archive must exactly match the training archive"
        )
    adapter_config_sha256 = frozen.base.file_sha256(bundle.adapter_config_path)
    adapter_model_sha256 = frozen.base.file_sha256(bundle.adapter_model_path)
    training_receipt_file_sha256 = frozen.base.file_sha256(
        bundle.training_receipt_path
    )

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = trainer.validate_checkpoint(args.checkpoint)
        inference_file_hashes = frozen.base.validate_inference_source_files(
            bernini_root
        )
    except (frozen.base.InferenceContractError, trainer.TrainingContractError) as error:
        raise C2FRLoRAInferenceError(str(error)) from error
    if transformer_config["num_attention_heads"] % frozen.base.ULYSSES_SIZE:
        raise C2FRLoRAInferenceError(
            "Bernini-R 1.3B heads are not divisible by Ulysses=4"
        )
    wan_diffusion_path = (
        bernini_root / "bernini/models/wan_diffusion.py"
    ).resolve(strict=True)
    try:
        wan_diffusion_sha256 = tri.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
        )
    except tri.TriBranchHookError as error:
        raise C2FRLoRAInferenceError(str(error)) from error
    trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    import peft
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if transformers_version != adapter_identity["transformers_version"]:
        raise C2FRLoRAInferenceError(
            "Transformers version differs from same-state training"
        )
    if SYSTEM_PROMPTS.get("mv2v") != frozen.base.MV2V_SYSTEM_PROMPT:
        raise C2FRLoRAInferenceError("runtime Bernini mv2v system prompt differs")
    if DEFAULT_NEG_PROMPT != frozen.base.DEFAULT_NEGATIVE_PROMPT:
        raise C2FRLoRAInferenceError("runtime Bernini negative prompt differs")
    distributed = frozen.base.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise C2FRLoRAInferenceError("C2FR LoRA requires four AUH ROCm-visible GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    try:
        source_tensor, source_metadata = frozen.base.prepare_exact_source(source_path)
    except frozen.base.InferenceContractError as error:
        raise C2FRLoRAInferenceError(str(error)) from error
    source_sha256 = frozen.base.file_sha256(source_path)
    action_prompt = frozen.base.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = frozen.base.build_training_prompt(
        motion.DEFAULT_NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **frozen.base.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except trainer.TrainingContractError as error:
        raise C2FRLoRAInferenceError(str(error)) from error
    if float(config.shift) != frozen.base.FLOW_SHIFT or config.use_unipc is not True:
        raise C2FRLoRAInferenceError(
            "renderer must use official UniPC with flow shift 5"
        )
    base_model = BerniniRendererModel(config)
    if any("lora_" in name.lower() for name, _ in base_model.named_modules()):
        raise C2FRLoRAInferenceError("base renderer unexpectedly contains LoRA modules")
    base_model.requires_grad_(False)
    base_model.eval()
    model, adapter_tensor_count, active_lora_module_count = (
        _strict_load_same_state_adapter(
            base_model=base_model,
            bundle=bundle,
            adapter_config=adapter_config,
            identity=adapter_identity,
        )
    )
    renderer = model.get_base_model()
    renderer.requires_grad_(False)
    renderer.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **frozen.base.tokenizer_load_kwargs()
    )
    if (
        tokenizer.padding_side != "right"
        or tokenizer.init_kwargs.get("fix_mistral_regex") is not True
    ):
        raise C2FRLoRAInferenceError(
            "tokenizer lost fix_mistral_regex/right-padding"
        )
    action_ids, action_mask = frozen.base._tokenize_training_prompt(
        tokenizer, action_prompt
    )
    noop_ids, noop_mask = frozen.base._tokenize_training_prompt(tokenizer, noop_prompt)
    negative_ids, negative_mask = frozen.base._tokenize_renderer_negative(
        tokenizer, frozen.base.DEFAULT_NEGATIVE_PROMPT
    )

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    with torch.no_grad():
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        )
    bucket = source_metadata["source_derived_bucket_hw"]
    expected_latent_shape = (
        1,
        int(vae.config.z_dim),
        frozen.base.LATENT_FRAME_COUNT,
        int(bucket[0]) // 8,
        int(bucket[1]) // 8,
    )
    if tuple(int(value) for value in source_latent.shape) != expected_latent_shape:
        raise C2FRLoRAInferenceError(
            "source VAE latent differs from exact 81f geometry"
        )
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    noop_embeddings, noop_identity = frozen.encode_semantic_noop_prompt(
        renderer, noop_ids, noop_mask, device=device
    )
    layout = tri.PackedLatentLayout.from_spatial_shape(expected_latent_shape)
    callback = TracedDenseCausalBoundaryCallback(
        source_clean=source_latent,
        layout=layout,
        alpha=float(args.alpha),
    )
    sampling = frozen.exact_sampler_contract(seed=args.seed)
    try:
        diffusion = tri.resolve_diffusion_core(renderer)
        pre_sample_schedule_audit = sigma_strata.audit_runtime_unipc_schedule(
            diffusion.scheduler, initialize=True
        )
        with tri.tri_branch_unipc_hook(
            renderer,
            noop_prompt_embeds=noop_embeddings,
            latent_shape=expected_latent_shape,
            clean_field_callback=callback,
            bernini_commit=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
            expected_steps=frozen.NUM_INFERENCE_STEPS,
            expected_flow_shift=frozen.base.FLOW_SHIFT,
        ) as tri_trace:
            with torch.no_grad():
                generated_latent = renderer.sample(
                    input_ids=action_ids.to(device),
                    attention_mask=action_mask.to(device),
                    uncond_input_ids=negative_ids.to(device),
                    uncond_attention_mask=negative_mask.to(device),
                    image_vae_latents=None,
                    multi_video_vae_latents=[source_latent],
                    multi_image_vae_latents=None,
                    width=int(bucket[1]),
                    height=int(bucket[0]),
                    device=device,
                    **sampling,
                )
        post_sample_schedule_audit = sigma_strata.audit_runtime_unipc_schedule(
            diffusion.scheduler, initialize=False
        )
    except (
        tri.TriBranchHookError,
        sparse_router.GeneratorNativeSparseRouterError,
        sigma_strata.InferenceSigmaStrataError,
    ) as error:
        raise C2FRLoRAInferenceError(str(error)) from error
    if pre_sample_schedule_audit != post_sample_schedule_audit:
        raise C2FRLoRAInferenceError(
            "official sample changed the bit-exact audited UniPC schedule"
        )
    execution_trace = validate_dense_execution_trace(
        tri_trace,
        callback.trace,
        runtime_schedule_audit=post_sample_schedule_audit,
    )
    if tuple(int(value) for value in generated_latent.shape) != expected_latent_shape:
        raise C2FRLoRAInferenceError(
            "generated latent differs from exact 81f geometry"
        )
    model.to("cpu")
    del noop_embeddings, callback, source_latent
    torch.cuda.empty_cache()

    if distributed.rank == 0:
        vae.to(device)
        with torch.no_grad():
            output = _vae_decode(vae, generated_latent)
        vae.to("cpu")
        expected_output_shape = (
            frozen.base.FRAME_COUNT,
            int(bucket[0]),
            int(bucket[1]),
            3,
        )
        if tuple(int(value) for value in output.shape) != expected_output_shape:
            raise C2FRLoRAInferenceError(
                "decoded output differs from exact 81f geometry"
            )
        temporary_output = output_path.with_name(
            f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
        )
        if temporary_output.exists() or temporary_output.is_symlink():
            raise C2FRLoRAInferenceError(
                f"stale temporary output exists: {temporary_output}"
            )
        save_output(output, str(temporary_output), fps=int(frozen.base.FPS))
        os.replace(temporary_output, output_path)
        from tools import materialize_vae

        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(
            output_path
        )
        try:
            frozen.base.validate_exact_video_metadata(
                int(encoded.shape[0]), encoded_fps
            )
        except frozen.base.InferenceContractError as error:
            raise C2FRLoRAInferenceError(str(error)) from error
        if tuple(encoded_hw) != tuple(bucket):
            raise C2FRLoRAInferenceError(
                "encoded output geometry differs from source bucket"
            )
        receipt = build_inference_receipt(
            args=args,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata=source_metadata,
            output_path=output_path,
            output_sha256=frozen.base.file_sha256(output_path),
            noop_identity=noop_identity,
            execution_trace=execution_trace,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            inference_file_hashes=inference_file_hashes,
            wan_diffusion_path=wan_diffusion_path,
            wan_diffusion_sha256=wan_diffusion_sha256,
            runtime_versions={
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
                "peft": peft.__version__,
            },
            adapter_bundle=bundle,
            adapter_identity=adapter_identity,
            adapter_config_sha256=adapter_config_sha256,
            adapter_model_sha256=adapter_model_sha256,
            training_receipt_file_sha256=training_receipt_file_sha256,
            adapter_tensor_count=adapter_tensor_count,
            active_lora_module_count=active_lora_module_count,
        )
        frozen.base._atomic_write_json(receipt_path, receipt)
        print(frozen.base.canonical_json_bytes(receipt).decode("utf-8"), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
