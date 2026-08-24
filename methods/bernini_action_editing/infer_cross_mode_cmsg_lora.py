#!/usr/bin/env python3
"""Fail-closed Bernini Cross-Mode CMSG LoRA v6 inference integration.

The frozen target-only T2V generator used by v6 is a *training teacher*.  It
is deliberately absent here.  Deployment observes only the source video and
the action instruction and evaluates four editor branches at every one of the
official forty UniPC states:

* frozen negative;
* frozen semantic no-op;
* frozen action; and
* adapted action.

The official frozen-action APG output is first reconstructed bit-exactly.  In
clean-field coordinates the two causal editor directions are

``B0 = Q0(frozen_action - frozen_noop)`` and
``Btheta = Q0(adapted_action - frozen_noop)``.

The operator shared with training releases ``Btheta`` early, cosine blends it
to ``B0``, and aliases ``B0`` exactly at zero release.  The scheduler-boundary
clean field is reconstructed as ``frozen_action + (executed - B0)``.  This is
important: ``frozen_noop + executed`` would silently replace the official
action phase-zero gauge and would not recover the official Bernini object at
late steps.  At every zero-release step this module therefore sends the exact
same ``official_model_output`` Python tensor object to the untouched official
UniPC scheduler.

This file integrates a reusable four-branch hook and strict receipt/operator
validation.  It intentionally does not claim a standalone end-to-end CLI;
without ``--preflight-only`` its command-line entry point fails closed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import cross_mode_motion_spectrum as spectrum  # noqa: E402
import infer_prior_tangent_lora as v5  # noqa: E402
import train_cross_mode_cmsg_auh as v6_auh  # noqa: E402
import train_cross_mode_cmsg_lora as v6_train  # noqa: E402


tri = v5.tri
trainer = v5.trainer
sigma_strata = v5.sigma_strata
frozen = v5.frozen

METHOD_NAME = v6_auh.METHOD_NAME
TRAINING_RECEIPT_SCHEMA = v6_auh.RECEIPT_SCHEMA
INFERENCE_RECEIPT_SCHEMA = "bernini-cross-mode-cmsg-lora-inference-receipt-v6"
REQUIRED_LORA_SCOPE = v6_train.LORA_SCOPE
REQUIRED_TARGET_MODULE_COUNT = v6_train.EXPECTED_LORA_MODULES
REQUIRED_LORA_RANK = v6_train.LORA_RANK
REQUIRED_LORA_ALPHA = v6_train.LORA_ALPHA
ADAPTER_SCALE = 1.0
NUM_FRAMES = v6_train.NUM_FRAMES
LATENT_PHASES = v6_train.LATENT_PHASES
NUM_DENOISING_STEPS = spectrum.NUM_DENOISING_STEPS
LATE_EXACT_STEPS = tuple(
    step
    for step in range(NUM_DENOISING_STEPS)
    if spectrum.release_rho(step) == 0.0
)
FORMAL_ADAPTER_OFF_STEPS = tuple(range(spectrum.ZERO_RELEASE_FIRST_STEP, 40))

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class CrossModeCMSGInferenceError(RuntimeError):
    """Raised before a v6 adapter or scheduler boundary can be trusted."""


def expected_lora_targets() -> list[str]:
    """Return the immutable, fully qualified 46-module v6 LoRA scope."""

    try:
        targets = v6_train.select_cmsg_lora_targets(
            v6_train.canonical_attention_modules()
        )
    except v6_train.CrossModeCMSGTrainingError as error:
        raise CrossModeCMSGInferenceError(str(error)) from error
    if targets != sorted(set(targets)) or len(targets) != REQUIRED_TARGET_MODULE_COUNT:
        raise CrossModeCMSGInferenceError("v6 canonical LoRA scope is not exact-46")
    return targets


def runtime_contract() -> dict[str, Any]:
    """Return the complete editor-only deployment contract."""

    schedule = list(spectrum.release_rho_schedule())
    return {
        "method": METHOD_NAME,
        "external_inference_conditions": list(v6_train.INFERENCE_CONDITIONS),
        "internal_fixed_controls": [
            "official_negative_prompt",
            "semantic_noop_instruction",
        ],
        "training_only_conditions": list(v6_train.TRAINING_ONLY_CONDITIONS),
        "inference_generator_forwards": 0,
        "inference_generator_loaded": False,
        "per_step_editor_branches": [
            "frozen_negative",
            "frozen_noop",
            "frozen_action",
            "adapted_action",
        ],
        "frozen_branch_adapter_state": "disabled",
        "adapted_action_adapter_state": "enabled_unmerged_unit_scale",
        "all_branch_autograd": False,
        "apg": "official_normalized_guidance",
        "apg_momentum": 0.0,
        "packed_field_shape": "[B,N,D]",
        "operator_field_shape": "[B,21,S,D]",
        "latent_phases": LATENT_PHASES,
        "frozen_editor_direction": "B0=Q0(frozen_action-frozen_noop)",
        "adapted_editor_direction": "Btheta=Q0(adapted_action-frozen_noop)",
        "shared_training_operator": (
            "execute_distilled_editor(B0,Btheta,step_index)"
        ),
        "scheduler_clean": "frozen_action_clean+(executed_direction-B0)",
        "release_schedule": schedule,
        "release_schedule_sha256": trainer.object_sha256(schedule),
        "release_contract": (
            "steps 0-19 adapted; steps 20-31 inclusive cosine; "
            "steps 32-39 formal adapter-off"
        ),
        "zero_release_steps": list(LATE_EXACT_STEPS),
        "formal_adapter_off_steps": list(FORMAL_ADAPTER_OFF_STEPS),
        "zero_release_scheduler_boundary": (
            "exact_same_official_frozen_action_model_output_object"
        ),
        "official_unipc_calls_per_step": 1,
        "custom_integrator": False,
        "first_frame_anchor": False,
        "forbidden_conditions": list(v6_train.FORBIDDEN_INFERENCE_CONDITIONS),
    }


def expected_immutable_training_contract(
    *, checkpoint_tree_sha256: str = trainer.CHECKPOINT_TREE_SHA256
) -> dict[str, Any]:
    """Return the fixed subset emitted by the real AUH trainer.

    Dataset hashes, routing stream identity, seed, and source revisions are
    dynamic and are validated separately.  Every item returned here maps
    directly to :func:`train_cross_mode_cmsg_auh._immutable_contract`; this is
    not a speculative future receipt schema.
    """

    targets = expected_lora_targets()
    schedule = list(spectrum.release_rho_schedule())
    return {
        "method": METHOD_NAME,
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "frames": NUM_FRAMES,
        "latent_phases": LATENT_PHASES,
        "learning_rate": v6_auh.LEARNING_RATE,
        "lora": {
            "scope": REQUIRED_LORA_SCOPE,
            "rank": REQUIRED_LORA_RANK,
            "alpha": REQUIRED_LORA_ALPHA,
            "dropout": 0.0,
            "bias": "none",
            "target_modules": targets,
            "target_modules_sha256": trainer.object_sha256(targets),
            "target_module_count": REQUIRED_TARGET_MODULE_COUNT,
        },
        "training_bridge_endpoint": v6_auh.TRAINING_BRIDGE_ENDPOINT,
        "target_endpoint_teacher_leakage_forbidden": True,
        "forward_cell_order": list(v6_auh.FORWARD_CELL_ORDER),
        "forwards_per_candidate": 6,
        "graph_forwards_per_candidate": 1,
        "training_editor_branches": [
            "frozen_negative_adapter_off_no_grad",
            "frozen_noop_adapter_off_no_grad",
            "frozen_action_adapter_off_no_grad",
            "adapted_action_adapter_on_grad",
        ],
        "inference_editor_branches": [
            "frozen_negative_adapter_off_no_grad",
            "frozen_noop_adapter_off_no_grad",
            "frozen_action_adapter_off_no_grad",
            "adapted_action_adapter_on_no_grad",
        ],
        "editor_guidance": {
            "mode": "official_momentum_zero_apg",
            "guidance_scale": v6_auh.v5.APG_GUIDANCE_SCALE,
            "eta": v6_auh.v5.APG_ETA,
            "norm_threshold": v6_auh.v5.APG_NORM_THRESHOLD,
            "momentum": v6_auh.v5.APG_MOMENTUM,
        },
        "generator_guidance": {
            "mode": "official_t2v_plain_cfg",
            "native_velocity_formula": "v_negative+4*(v_action-v_negative)",
            "scale": v6_auh.T2V_GUIDANCE_SCALE,
            "combine_before_fp32_clean_reconstruction": True,
        },
        "text_contract": {
            "editor_action": "official_mv2v_system_prompt_plus_prompt_clean",
            "generator_action": "official_t2v_system_prompt_plus_prompt_clean",
            "generator_t2v_system_prompt_sha256": v6_auh.T2V_SYSTEM_PROMPT_SHA256,
            "generator_negative": "official_negative_verbatim",
            "generator_negative_sha256": v6_auh.v5.NEGATIVE_PROMPT_SHA256,
        },
        "target_motion_teacher": "Q0(target_clean-source_clean)",
        "target_used_as_model_condition": False,
        "t2v_rope_parity": {
            "official_pack_rule": "vae_mask=True -> source_id=0",
            "mv2v_target_source_id": 0,
            "native_t2v_target_source_id": 0,
            "same_target_shape_required": True,
            "per_candidate_exact_tensor_equality_required": True,
            "generator_uses_direct_editor_target_tail_view": True,
        },
        "generator_target_tail": (
            "direct GPU storage view of editor noisy target; no transform/noise resample"
        ),
        "loss_config": asdict(v6_train.CMSGTrainingLossConfig()),
        "spectrum_config": asdict(spectrum.CrossModeMotionSpectrumConfig()),
        "release_schedule": schedule,
        "release_schedule_sha256": trainer.object_sha256(schedule),
        "sigma_schedule": "exact_40_step_flow_shift_5_cycle",
        "sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "sigma_selector": "absolute_global_step_mod_40",
        "inverse_sigma_weight_floor": float(
            sigma_strata.PINNED_POSITIVE_SIGMAS[-1]
        ),
        "candidate_seed_formula": "step_seed(base_seed,attempt_ordinal,row_index)",
        "inference_conditions": list(v6_train.INFERENCE_CONDITIONS),
        "training_only_conditions": list(v6_train.TRAINING_ONLY_CONDITIONS),
        "forbidden_inference_conditions": list(
            v6_train.FORBIDDEN_INFERENCE_CONDITIONS
        ),
        "inference_generator_forwards": 0,
        "frozen_editor_direction": "Q0(frozen_action-frozen_noop)",
        "adapted_editor_direction": "Q0(adapted_action-frozen_noop)",
        "inference_execution": (
            "frozen_action_clean+(execute_distilled_editor(B0,Btheta,k)-B0)"
        ),
        "phase_zero_contract": "official_frozen_action_phase_zero_exactly_preserved",
        "release_contract": (
            "0-19 adapted; 20-31 inclusive cosine with rho31=0; "
            "32-39 exact official adapter-off replay"
        ),
        "zero_release_steps": list(LATE_EXACT_STEPS),
        "formal_adapter_off_steps": list(FORMAL_ADAPTER_OFF_STEPS),
        "late_scheduler_boundary": (
            "exact_same_official_frozen_action_model_output_object"
        ),
        "resume_integrated": False,
    }


def expected_training_supervision_contract() -> dict[str, Any]:
    """Return the real trainer's completed-step supervision receipt."""

    try:
        return dict(v6_auh._supervision_receipt(global_step=NUM_DENOISING_STEPS))
    except v6_auh.CMSGauhTrainingError as error:  # pragma: no cover - invariant
        raise CrossModeCMSGInferenceError(str(error)) from error


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossModeCMSGInferenceError(f"training receipt lacks {label}")
    return value


def _validate_sha(value: Any, *, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise CrossModeCMSGInferenceError(f"{label} is invalid")
    return value


def validate_training_adapter_contract(
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Accept only a completed receipt emitted by the real AUH trainer."""

    if not isinstance(adapter_config, Mapping) or not isinstance(receipt, Mapping):
        raise CrossModeCMSGInferenceError("adapter config and receipt must be mappings")
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    _validate_sha(digest, label="training receipt digest", pattern=_SHA256_RE)
    if trainer.object_sha256(candidate) != digest:
        raise CrossModeCMSGInferenceError("training receipt digest differs")
    if receipt.get("schema_version") != v6_auh.RECEIPT_SCHEMA:
        raise CrossModeCMSGInferenceError("training receipt schema differs from AUH v6")
    if receipt.get("method") != v6_auh.METHOD_NAME:
        raise CrossModeCMSGInferenceError("training method differs from AUH v6")

    global_step = receipt.get("global_step")
    max_steps = receipt.get("max_steps")
    accepted_count = receipt.get("accepted_count")
    attempt_ordinal = receipt.get("attempt_ordinal")
    rejected_count = receipt.get("rejected_count")
    if (
        type(global_step) is not int
        or global_step < NUM_DENOISING_STEPS
        or type(max_steps) is not int
        or max_steps < global_step
        or accepted_count != global_step
        or type(attempt_ordinal) is not int
        or type(rejected_count) is not int
        or rejected_count < 0
        or attempt_ordinal != global_step + rejected_count
    ):
        raise CrossModeCMSGInferenceError(
            "AUH v6 checkpoint lacks one completed, auditable 40-sigma cycle"
        )
    gate_audit = receipt.get("gate_audit")
    if (
        not isinstance(gate_audit, list)
        or len(gate_audit) != attempt_ordinal
        or receipt.get("gate_audit_sha256") != trainer.object_sha256(gate_audit)
    ):
        raise CrossModeCMSGInferenceError("training gate audit differs")
    expected_sigma_receipt = sigma_strata.build_sigma_strata_receipt(
        completed_optimizer_steps=global_step
    )
    if receipt.get("inference_sigma_strata") != expected_sigma_receipt:
        raise CrossModeCMSGInferenceError("training sigma-strata receipt differs")

    if receipt.get("bernini_commit") != trainer.BERNINI_OFFICIAL_COMMIT:
        raise CrossModeCMSGInferenceError("training Bernini revision differs")
    if receipt.get("veomni_commit") != trainer.VEOMNI_TESTED_COMMIT:
        raise CrossModeCMSGInferenceError("training VeOmni revision differs")
    checkpoint = _require_mapping(receipt.get("checkpoint"), label="checkpoint identity")
    if checkpoint.get("tree_sha256") != expected_checkpoint_tree_sha256:
        raise CrossModeCMSGInferenceError("training checkpoint tree differs")

    immutable = _require_mapping(
        receipt.get("immutable_contract"), label="immutable contract"
    )
    value = _require_mapping(immutable.get("value"), label="immutable value")
    if immutable.get("digest") != trainer.object_sha256(value):
        raise CrossModeCMSGInferenceError("training immutable contract digest differs")
    expected_immutable = expected_immutable_training_contract(
        checkpoint_tree_sha256=expected_checkpoint_tree_sha256
    )
    for name, expected in expected_immutable.items():
        if value.get(name) != expected:
            raise CrossModeCMSGInferenceError(
                f"training AUH v6 immutable field differs: {name}"
            )
    for name, pattern in (
        ("method_source_revision", _SHA1_RE),
        ("method_source_archive_sha256", _SHA256_RE),
        ("dataset_summary_sha256", _SHA256_RE),
        ("dataset_index_sha256", _SHA256_RE),
        ("routing_digest", _SHA256_RE),
        ("routing_file_sha256", _SHA256_RE),
        ("eligible_route_stream_sha256", _SHA256_RE),
    ):
        _validate_sha(value.get(name), label=f"training {name}", pattern=pattern)
    if (
        value.get("bernini_commit") != receipt.get("bernini_commit")
        or value.get("veomni_commit") != receipt.get("veomni_commit")
        or value.get("checkpoint_tree_sha256") != checkpoint.get("tree_sha256")
        or value.get("checkpoint_path") != checkpoint.get("path")
        or value.get("eligible_route_count") != 359
        or value.get("routing_file_sha256") != v6_auh.v5.STRICT_ROUTING_SHA256
        or type(value.get("seed")) is not int
    ):
        raise CrossModeCMSGInferenceError("training source/data identity differs")
    for name in ("weight_decay", "max_grad_norm"):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise CrossModeCMSGInferenceError(f"training {name} is invalid")
    if float(value["weight_decay"]) < 0.0 or float(value["max_grad_norm"]) <= 0.0:
        raise CrossModeCMSGInferenceError("training optimizer scalar differs")
    gate = _require_mapping(value.get("gate"), label="immutable gate")
    if (
        gate.get("enforced") is not True
        or type(gate.get("max_attempts_per_accepted_step")) is not int
        or not 1 <= gate["max_attempts_per_accepted_step"] <= 359
        or gate.get("attempt_stream") != "global_attempt_ordinal"
        or gate.get("complete_record_exact_across_all_four_ranks") is not True
        or gate.get("rho_zero_teacher_inactive_no_rejection") is not True
    ):
        raise CrossModeCMSGInferenceError("training frozen-prior gate differs")

    dataset = _require_mapping(receipt.get("dataset"), label="dataset identity")
    summary = _require_mapping(dataset.get("summary"), label="dataset summary")
    routing = _require_mapping(dataset.get("routing"), label="routing identity")
    dataset_signature = value.get("dataset_signature")
    if (
        not isinstance(dataset_signature, str)
        or not dataset_signature
        or dataset.get("rows") != 644
        or dataset.get("signature") != dataset_signature
        or summary.get("sha256") != value.get("dataset_summary_sha256")
        or summary.get("index_sha256") != value.get("dataset_index_sha256")
        or routing.get("default_tier") != "reject"
        or routing.get("explicit_route_counts")
        != {"full_pair": 0, "motion_only": 359, "reject": 285}
        or routing.get("file_sha256") != value.get("routing_file_sha256")
        or routing.get("routing_digest") != value.get("routing_digest")
    ):
        raise CrossModeCMSGInferenceError("training strict-359 dataset receipt differs")

    supervision = _require_mapping(receipt.get("supervision"), label="supervision")
    if dict(supervision) != expected_training_supervision_contract():
        raise CrossModeCMSGInferenceError("training AUH v6 supervision differs")
    if (
        receipt.get("inference_conditions") != list(v6_train.INFERENCE_CONDITIONS)
        or receipt.get("training_only_generator_and_target") is not True
        or receipt.get("experimental_training") is not True
        or receipt.get("canary_gate_disabled") is not False
        or receipt.get("resume_integrated") is not False
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
        or receipt.get("inference_loader_parity_pending") is not False
    ):
        raise CrossModeCMSGInferenceError("training AUH v6 publication state differs")

    targets = expected_lora_targets()
    expected_adapter = {
        "rank": REQUIRED_LORA_RANK,
        "alpha": REQUIRED_LORA_ALPHA,
        "scope": REQUIRED_LORA_SCOPE,
        "target_module_count": REQUIRED_TARGET_MODULE_COUNT,
        "target_modules": targets,
        "target_modules_sha256": trainer.object_sha256(targets),
    }
    adapter = _require_mapping(receipt.get("adapter"), label="adapter identity")
    immutable_lora = _require_mapping(value.get("lora"), label="immutable LoRA")
    for name, expected in expected_adapter.items():
        if adapter.get(name) != expected or immutable_lora.get(name) != expected:
            raise CrossModeCMSGInferenceError(
                f"training adapter exact-46 field differs: {name}"
            )
    for name in (
        "initialization_digest",
        "checkpoint_parameter_digest",
        "parameter_names_sha256",
    ):
        _validate_sha(adapter.get(name), label=f"training adapter {name}", pattern=_SHA256_RE)
    if type(adapter.get("trainable_parameter_count")) is not int or adapter[
        "trainable_parameter_count"
    ] <= 0:
        raise CrossModeCMSGInferenceError("training adapter parameter count is invalid")

    optimizer = _require_mapping(receipt.get("optimizer"), label="optimizer identity")
    parameter_names = optimizer.get("parameter_names")
    if (
        optimizer.get("type") != "AdamW"
        or float(optimizer.get("learning_rate", -1.0)) != v6_auh.LEARNING_RATE
        or float(optimizer.get("weight_decay", -1.0)) != float(value["weight_decay"])
        or float(optimizer.get("max_gradient_norm", -1.0))
        != float(value["max_grad_norm"])
        or not isinstance(parameter_names, list)
        or not parameter_names
        or not all(isinstance(name, str) and name for name in parameter_names)
        or len(parameter_names) != len(set(parameter_names))
        or adapter.get("parameter_names_sha256")
        != trainer.object_sha256(parameter_names)
    ):
        raise CrossModeCMSGInferenceError("training optimizer/LoRA identity differs")
    _validate_sha(
        optimizer.get("checkpoint_state_digest"),
        label="optimizer checkpoint state digest",
        pattern=_SHA256_RE,
    )
    distributed = _require_mapping(receipt.get("distributed"), label="distributed identity")
    if (
        distributed.get("world_size") != 4
        or distributed.get("ulysses_size") != 4
        or distributed.get("same_pair_all_ranks") is not True
        or distributed.get("explicit_lora_gradient_all_reduce") is not True
        or not isinstance(distributed.get("backend"), str)
        or not distributed["backend"]
    ):
        raise CrossModeCMSGInferenceError("training four-rank Ulysses contract differs")

    if adapter_config.get("peft_type") != "LORA":
        raise CrossModeCMSGInferenceError("adapter is not LoRA")
    if adapter_config.get("r") != REQUIRED_LORA_RANK:
        raise CrossModeCMSGInferenceError("adapter rank differs")
    try:
        alpha = float(adapter_config.get("lora_alpha", -1))
        dropout = float(adapter_config.get("lora_dropout", -1))
    except (TypeError, ValueError) as error:
        raise CrossModeCMSGInferenceError("adapter alpha/dropout are invalid") from error
    if alpha != float(REQUIRED_LORA_ALPHA) or dropout != 0.0:
        raise CrossModeCMSGInferenceError("adapter alpha/dropout differ")
    if adapter_config.get("bias") != "none":
        raise CrossModeCMSGInferenceError("adapter bias differs")
    if adapter_config.get("modules_to_save") not in (None, []):
        raise CrossModeCMSGInferenceError("modules_to_save are forbidden")
    if adapter_config.get("use_dora") not in (None, False):
        raise CrossModeCMSGInferenceError("DoRA is outside the v6 contract")
    if adapter_config.get("use_rslora") not in (None, False):
        raise CrossModeCMSGInferenceError("RS-LoRA is outside the v6 contract")
    serialized = adapter_config.get("target_modules")
    if (
        not isinstance(serialized, list)
        or len(serialized) != REQUIRED_TARGET_MODULE_COUNT
        or not all(isinstance(name, str) and name for name in serialized)
        or len(serialized) != len(set(serialized))
        or set(serialized) != set(targets)
    ):
        raise CrossModeCMSGInferenceError(
            "serialized target_modules must cover the exact unique 46-module scope"
        )

    transformers_version = receipt.get("transformers_version")
    if not isinstance(transformers_version, str) or not transformers_version:
        raise CrossModeCMSGInferenceError("training Transformers version is missing")
    return {
        "receipt_digest": digest,
        "global_step": global_step,
        "scope": REQUIRED_LORA_SCOPE,
        "targets": targets,
        "serialized_target_modules": sorted(serialized),
        "target_modules_sha256": trainer.object_sha256(targets),
        "initialization_digest": adapter["initialization_digest"],
        "checkpoint_parameter_digest": adapter["checkpoint_parameter_digest"],
        "transformers_version": transformers_version,
        "training_method_source_revision": value["method_source_revision"],
        "training_method_source_archive_sha256": value[
            "method_source_archive_sha256"
        ],
    }


def strict_load_adapter(
    *,
    base_model: Any,
    bundle: Any,
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    expected_checkpoint_tree_sha256: str = trainer.CHECKPOINT_TREE_SHA256,
) -> tuple[Any, int, int, dict[str, Any]]:
    """Validate the v6 receipt, then reuse the pinned strict PEFT loader."""

    identity = validate_training_adapter_contract(
        adapter_config,
        receipt,
        expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
    )
    try:
        model, tensor_count, active_count = v5._strict_load_adapter(
            base_model=base_model,
            bundle=bundle,
            adapter_config=adapter_config,
            identity=identity,
        )
    except v5.PriorTangentInferenceError as error:
        raise CrossModeCMSGInferenceError(str(error)) from error
    if tensor_count != 2 * REQUIRED_TARGET_MODULE_COUNT:
        raise CrossModeCMSGInferenceError("v6 adapter tensor count differs from 92")
    if active_count != REQUIRED_TARGET_MODULE_COUNT:
        raise CrossModeCMSGInferenceError("v6 active LoRA scope differs from exact-46")
    return model, tensor_count, active_count, identity


def packed_to_phase_grid(packed: Any, *, layout: Any) -> Any:
    """Reshape exact Wan token order ``[B,N,D]`` to ``[B,21,S,D]``."""

    shape = tuple(int(value) for value in getattr(packed, "shape", ()))
    if shape != layout.packed_shape:
        raise CrossModeCMSGInferenceError(
            f"packed field shape {shape} differs from {layout.packed_shape}"
        )
    if layout.frames != LATENT_PHASES or layout.tokens % LATENT_PHASES:
        raise CrossModeCMSGInferenceError("v6 requires exactly 21 latent phases")
    cells = layout.tokens // LATENT_PHASES
    return packed.reshape(layout.batch, LATENT_PHASES, cells, layout.packed_channels)


def phase_grid_to_packed(grid: Any, *, layout: Any) -> Any:
    cells = layout.tokens // LATENT_PHASES
    expected = (layout.batch, LATENT_PHASES, cells, layout.packed_channels)
    shape = tuple(int(value) for value in getattr(grid, "shape", ()))
    if shape != expected:
        raise CrossModeCMSGInferenceError(
            f"phase field shape {shape} differs from {expected}"
        )
    return grid.reshape(layout.packed_shape)


@dataclass(frozen=True)
class RawCrossModeCMSGStep:
    """Only the four deployed editor branches at one official UniPC state."""

    step_index: int
    timestep: Any
    timestep_float: float
    sigma: Any
    sigma_float: float
    model_id: str
    sample_packed: Any
    official_model_output: Any
    frozen_negative_velocity_packed: Any
    frozen_noop_velocity_packed: Any
    frozen_action_velocity_packed: Any
    adapted_action_velocity_packed: Any
    apg: Any
    layout: Any


@dataclass(frozen=True)
class CrossModeCMSGStepRecord:
    step_index: int
    timestep: float
    sigma: float
    rho: float
    model_id: str
    transformer_forwards: int
    frozen_negative_forwards: int
    frozen_noop_forwards: int
    frozen_action_forwards: int
    adapted_action_forwards: int
    original_scheduler_calls: int
    official_frozen_action_apg_exact: bool
    official_frozen_action_apg_rms_error: float
    official_frozen_action_apg_max_abs_error: float
    frozen_editor_direction_rms: float
    adapted_editor_direction_rms: float
    raw_direction_delta_rms: float
    executed_direction_correction_rms: float
    executed_first_phase_max_abs: float
    scheduler_boundary_correction_rms: float
    phase_cells: int
    exact_official_model_output_object: bool
    adapter_loaded: bool
    generator_forwards: int


@dataclass
class CrossModeCMSGTrace:
    adapter_loaded: bool
    records: list[CrossModeCMSGStepRecord] = field(default_factory=list)
    sample_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": runtime_contract(),
            "adapter_loaded": self.adapter_loaded,
            "sample_calls": self.sample_calls,
            "step_count": len(self.records),
            "steps": [asdict(record) for record in self.records],
        }


def _tensor_stat(value: Any, *, label: str) -> float:
    try:
        result = float(value.detach().float().cpu().item())
    except Exception as error:
        raise CrossModeCMSGInferenceError(f"cannot serialize {label}") from error
    if not math.isfinite(result) or result < 0.0:
        raise CrossModeCMSGInferenceError(f"{label} must be finite and non-negative")
    return result


def project_cross_mode_cmsg_step(
    raw: RawCrossModeCMSGStep,
    *,
    frozen_action_momentum: Any,
    frozen_noop_momentum: Any,
    adapted_action_momentum: Any,
) -> tuple[Any, CrossModeCMSGStepRecord]:
    """Project four editor branches into one official UniPC model output."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH has torch
        raise CrossModeCMSGInferenceError("v6 projection requires PyTorch") from error

    if not isinstance(raw, RawCrossModeCMSGStep):
        raise CrossModeCMSGInferenceError("raw step has the wrong contract type")
    if type(raw.step_index) is not int or not 0 <= raw.step_index < NUM_DENOISING_STEPS:
        raise CrossModeCMSGInferenceError("step_index must be an integer in [0,40)")
    if raw.model_id != "transformer_1":
        raise CrossModeCMSGInferenceError("v6 exact-1.3B path requires transformer_1")
    if not math.isfinite(float(raw.timestep_float)):
        raise CrossModeCMSGInferenceError("timestep must be finite")
    if raw.apg.momentum != 0.0:
        raise CrossModeCMSGInferenceError("official APG momentum must be exactly zero")
    tensors = (
        raw.sample_packed,
        raw.official_model_output,
        raw.frozen_negative_velocity_packed,
        raw.frozen_noop_velocity_packed,
        raw.frozen_action_velocity_packed,
        raw.adapted_action_velocity_packed,
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise CrossModeCMSGInferenceError("all four branches must be torch tensors")
    if any(tuple(int(v) for v in tensor.shape) != raw.layout.packed_shape for tensor in tensors):
        raise CrossModeCMSGInferenceError("four-branch packed shapes differ")
    sigma = raw.sigma
    if (
        not isinstance(sigma, torch.Tensor)
        or sigma.ndim != 0
        or sigma.device.type != "cpu"
        or sigma.dtype != torch.float32
        or not bool(torch.isfinite(sigma))
        or not bool(sigma > 0)
    ):
        raise CrossModeCMSGInferenceError(
            "pinned UniPC sigma must be one positive CPU fp32 scalar"
        )
    if not math.isfinite(float(raw.sigma_float)) or not math.isclose(
        float(raw.sigma_float), float(sigma.item()), rel_tol=0.0, abs_tol=1.0e-7
    ):
        raise CrossModeCMSGInferenceError("serialized sigma differs from pinned tensor")

    sample = tri._packed_to_spatial(raw.sample_packed, raw.layout)
    official_velocity = tri._packed_to_spatial(raw.official_model_output, raw.layout)
    negative_velocity = tri._packed_to_spatial(
        raw.frozen_negative_velocity_packed, raw.layout
    )
    noop_velocity = tri._packed_to_spatial(
        raw.frozen_noop_velocity_packed, raw.layout
    )
    frozen_action_velocity = tri._packed_to_spatial(
        raw.frozen_action_velocity_packed, raw.layout
    )
    adapted_action_velocity = tri._packed_to_spatial(
        raw.adapted_action_velocity_packed, raw.layout
    )
    negative_clean = tri.pinned_raw_condition_clean(sample, negative_velocity, sigma)
    noop_condition_clean = tri.pinned_raw_condition_clean(sample, noop_velocity, sigma)
    frozen_action_condition_clean = tri.pinned_raw_condition_clean(
        sample, frozen_action_velocity, sigma
    )
    adapted_action_condition_clean = tri.pinned_raw_condition_clean(
        sample, adapted_action_velocity, sigma
    )
    guidance = raw.apg.guidance_scale_for(raw.model_id)
    frozen_action_clean = tri._normalized_guidance(
        frozen_action_condition_clean,
        negative_clean,
        guidance,
        frozen_action_momentum,
        raw.apg.eta,
        raw.apg.norm_threshold,
    )
    frozen_noop_clean = tri._normalized_guidance(
        noop_condition_clean,
        negative_clean,
        guidance,
        frozen_noop_momentum,
        raw.apg.eta,
        raw.apg.norm_threshold,
    )
    adapted_action_clean = tri._normalized_guidance(
        adapted_action_condition_clean,
        negative_clean,
        guidance,
        adapted_action_momentum,
        raw.apg.eta,
        raw.apg.norm_threshold,
    )

    rebuilt_official = tri._spatial_to_packed(
        (sample - frozen_action_clean) / sigma, raw.layout
    ).to(
        device=raw.official_model_output.device,
        dtype=raw.official_model_output.dtype,
    )
    parity_error = rebuilt_official.float() - raw.official_model_output.float()
    parity_rms = _tensor_stat(parity_error.square().mean().sqrt(), label="APG parity RMS")
    parity_max = _tensor_stat(parity_error.abs().max(), label="APG parity maximum")
    parity_exact = bool(torch.equal(rebuilt_official, raw.official_model_output))
    if not parity_exact:
        raise CrossModeCMSGInferenceError(
            "locally rebuilt frozen-action APG differs from official model_output"
        )

    # Only after the bit-exact certificate may the official object become the
    # authoritative frozen-action clean field.
    frozen_action_clean = sample - sigma * official_velocity
    frozen_action_phase = packed_to_phase_grid(
        tri._spatial_to_packed(frozen_action_clean, raw.layout).float(),
        layout=raw.layout,
    )
    frozen_noop_phase = packed_to_phase_grid(
        tri._spatial_to_packed(frozen_noop_clean, raw.layout).float(),
        layout=raw.layout,
    )
    adapted_action_phase = packed_to_phase_grid(
        tri._spatial_to_packed(adapted_action_clean, raw.layout).float(),
        layout=raw.layout,
    )
    try:
        frozen_direction = spectrum.q0(frozen_action_phase - frozen_noop_phase)
        adapted_direction = spectrum.q0(adapted_action_phase - frozen_noop_phase)
        execution = v6_train.execute_distilled_editor(
            frozen_direction,
            adapted_direction,
            step_index=raw.step_index,
        )
    except (
        spectrum.CrossModeMotionSpectrumError,
        v6_train.CrossModeCMSGTrainingError,
    ) as error:
        raise CrossModeCMSGInferenceError(str(error)) from error
    rho = spectrum.release_rho(raw.step_index)
    if execution.rho != rho:
        raise CrossModeCMSGInferenceError("shared train/inference release rho differs")
    if rho == 0.0 and execution.executed_field is not frozen_direction:
        raise CrossModeCMSGInferenceError("zero-release operator lost frozen alias")
    if rho == 1.0 and execution.executed_field is not adapted_direction:
        raise CrossModeCMSGInferenceError("full-release operator lost adapted alias")

    direction_correction = execution.executed_field - frozen_direction
    zero_phase = torch.zeros_like(direction_correction[:, 0])
    phase0_max = _tensor_stat(
        direction_correction[:, :1].abs().max(),
        label="executed first-phase maximum",
    )
    if phase0_max != 0.0 or not bool(torch.equal(direction_correction[:, 0], zero_phase)):
        raise CrossModeCMSGInferenceError("v6 direction correction changed phase zero")

    if rho == 0.0:
        # This object-identity path is the late-detail guarantee.  Do not
        # perform a clean->velocity round trip here.
        executed_velocity = raw.official_model_output
    else:
        executed_clean_phase = frozen_action_phase + direction_correction
        if not bool(
            torch.equal(
                executed_clean_phase[:, :1], frozen_action_phase[:, :1]
            )
        ):
            raise CrossModeCMSGInferenceError(
                "scheduler clean reconstruction changed frozen phase zero"
            )
        executed_clean_packed = phase_grid_to_packed(
            executed_clean_phase, layout=raw.layout
        )
        executed_velocity = ((raw.sample_packed - executed_clean_packed) / sigma).to(
            device=raw.official_model_output.device,
            dtype=raw.official_model_output.dtype,
        )
    if not bool(torch.isfinite(executed_velocity).all()):
        raise CrossModeCMSGInferenceError("scheduler model_output is non-finite")
    exact_official_object = executed_velocity is raw.official_model_output
    if exact_official_object is not (rho == 0.0):
        raise CrossModeCMSGInferenceError(
            "scheduler object identity differs from the zero-release contract"
        )

    scheduler_correction_rms = _tensor_stat(
        executed_velocity.float()
        .sub(raw.official_model_output.float())
        .square()
        .mean()
        .sqrt(),
        label="scheduler-boundary correction RMS",
    )
    record = CrossModeCMSGStepRecord(
        step_index=raw.step_index,
        timestep=float(raw.timestep_float),
        sigma=float(raw.sigma_float),
        rho=float(rho),
        model_id=raw.model_id,
        transformer_forwards=4,
        frozen_negative_forwards=1,
        frozen_noop_forwards=1,
        frozen_action_forwards=1,
        adapted_action_forwards=1,
        original_scheduler_calls=1,
        official_frozen_action_apg_exact=True,
        official_frozen_action_apg_rms_error=parity_rms,
        official_frozen_action_apg_max_abs_error=parity_max,
        frozen_editor_direction_rms=_tensor_stat(
            frozen_direction.square().mean().sqrt(), label="frozen direction RMS"
        ),
        adapted_editor_direction_rms=_tensor_stat(
            adapted_direction.square().mean().sqrt(), label="adapted direction RMS"
        ),
        raw_direction_delta_rms=_tensor_stat(
            (adapted_direction - frozen_direction).square().mean().sqrt(),
            label="raw direction delta RMS",
        ),
        executed_direction_correction_rms=_tensor_stat(
            direction_correction.square().mean().sqrt(),
            label="executed direction correction RMS",
        ),
        executed_first_phase_max_abs=phase0_max,
        scheduler_boundary_correction_rms=scheduler_correction_rms,
        phase_cells=raw.layout.tokens // LATENT_PHASES,
        exact_official_model_output_object=exact_official_object,
        adapter_loaded=True,
        generator_forwards=0,
    )
    projected = tri.ProjectedVelocity(
        model_output=executed_velocity,
        correction_rms=scheduler_correction_rms,
        effective_guidance_scale=guidance,
        official_action_parity_rms_error=parity_rms,
        official_action_parity_max_abs_error=parity_max,
        official_action_exact_parity=parity_exact,
        sample_dtype=str(sample.dtype),
        branch_velocity_dtype=str(frozen_action_velocity.dtype),
        official_model_output_dtype=str(raw.official_model_output.dtype),
    )
    return projected, record


class _InstalledCrossModeCMSG(v5._InstalledFourBranch):
    """Pinned v5 capture/APG wrapper with only the v6 scheduler operator changed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, execution_arm="main", **kwargs)
        if not self.adapter_loaded:
            raise CrossModeCMSGInferenceError("v6 hook requires one loaded adapter")
        self.trace = CrossModeCMSGTrace(adapter_loaded=True)

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise CrossModeCMSGInferenceError("scheduler.step ran outside sample")
        if any(
            value is None
            for value in (
                state.pending_negative,
                state.pending_base_action,
                state.pending_noop,
                state.pending_adapted_action,
            )
        ):
            raise CrossModeCMSGInferenceError(
                "scheduler.step arrived before all four editor branches"
            )
        official = tri._extract_argument(args, kwargs, index=0, name="model_output")
        timestep = tri._extract_argument(args, kwargs, index=1, name="timestep")
        sample = tri._extract_argument(args, kwargs, index=2, name="sample")
        step_index, sigma, sigma_float = tri._resolve_sigma(self.scheduler, timestep)
        base_action = state.pending_base_action
        assert base_action is not None and state.pending_negative is not None
        raw = RawCrossModeCMSGStep(
            step_index=step_index,
            timestep=timestep,
            timestep_float=tri._coerce_scalar(timestep, label="timestep"),
            sigma=sigma,
            sigma_float=sigma_float,
            model_id=str(base_action.bound["model_id"]),
            sample_packed=sample,
            official_model_output=official,
            frozen_negative_velocity_packed=state.pending_negative.prediction,
            frozen_noop_velocity_packed=state.pending_noop,
            frozen_action_velocity_packed=base_action.prediction,
            adapted_action_velocity_packed=state.pending_adapted_action,
            apg=state.apg,
            layout=self.layout,
        )
        projected, record = project_cross_mode_cmsg_step(
            raw,
            frozen_action_momentum=state.base_action_momentum,
            frozen_noop_momentum=state.base_noop_momentum,
            adapted_action_momentum=state.adapted_action_momentum,
        )
        call_args, call_kwargs = tri._replace_argument(
            self._original_scheduler_step,
            args,
            kwargs,
            name="model_output",
            value=projected.model_output,
        )
        result = self._original_scheduler_step(*call_args, **call_kwargs)
        state.integrated_steps += 1
        self.trace.records.append(record)
        state.pending_negative = None
        state.pending_base_action = None
        state.pending_noop = None
        state.pending_adapted_action = None
        return result


@contextmanager
def cross_mode_cmsg_unipc_hook(
    renderer_or_diffusion: Any,
    *,
    adapter_model: Any,
    source_clean: Any,
    noop_prompt_embeds: Any,
    latent_shape: Sequence[int],
    bernini_commit: str,
    wan_diffusion_path: str | Path,
    expected_steps: int = NUM_DENOISING_STEPS,
    expected_flow_shift: float = 5.0,
) -> Iterator[CrossModeCMSGTrace]:
    """Install the v6 operator into the audited official four-branch hook.

    ``source_clean`` is the full source-video latent already required by the
    official editor.  It is accepted only because the pinned v5 capture layer
    validates source geometry; the v6 projector never uses it as an anchor.
    """

    if adapter_model is None:
        raise CrossModeCMSGInferenceError("v6 inference requires a loaded adapter")
    if expected_steps != NUM_DENOISING_STEPS:
        raise CrossModeCMSGInferenceError("v6 requires exactly 40 official UniPC steps")
    diffusion = tri.resolve_diffusion_core(renderer_or_diffusion)
    try:
        bridge = _InstalledCrossModeCMSG(
            diffusion,
            adapter_model=adapter_model,
            source_clean=source_clean,
            noop_prompt_embeds=noop_prompt_embeds,
            noop_prompt_embeds_t2=None,
            latent_shape=latent_shape,
            clean_field_callback=lambda fields: fields.action_guided_clean,
            expected_steps=expected_steps,
            expected_flow_shift=expected_flow_shift,
            projector=tri.project_clean_fields,
            bernini_commit=bernini_commit,
            wan_diffusion_path=wan_diffusion_path,
        )
        bridge.install()
    except (v5.PriorTangentInferenceError, tri.TriBranchHookError) as error:
        raise CrossModeCMSGInferenceError(str(error)) from error
    try:
        yield bridge.trace
    finally:
        bridge.restore()


def validate_runtime_schedule_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "timesteps": list(sigma_strata.PINNED_TIMESTEPS),
        "positive_sigmas": list(sigma_strata.PINNED_POSITIVE_SIGMAS),
        "positive_sigmas_float32_be_hex": list(
            sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
        ),
        "terminal_sigma": 0.0,
        "terminal_sigma_float32_be_hex": sigma_strata.TERMINAL_SIGMA_FLOAT32_HEX,
    }
    if not isinstance(audit, Mapping) or dict(audit) != expected:
        raise CrossModeCMSGInferenceError("runtime UniPC schedule differs")
    return expected


def validate_execution_trace(
    trace: CrossModeCMSGTrace,
    *,
    runtime_schedule_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Certify one exact 40-step, four-editor-forward official UniPC sample."""

    if not isinstance(trace, CrossModeCMSGTrace) or trace.sample_calls != 1:
        raise CrossModeCMSGInferenceError("v6 hook must observe exactly one sample")
    if trace.adapter_loaded is not True:
        raise CrossModeCMSGInferenceError("v6 trace lacks its exact-46 adapter")
    if len(trace.records) != NUM_DENOISING_STEPS:
        raise CrossModeCMSGInferenceError("v6 must certify all 40 UniPC steps")
    audited = validate_runtime_schedule_audit(runtime_schedule_audit)
    sigmas: list[float] = []
    exact_steps: list[int] = []
    for expected_index, record in enumerate(trace.records):
        selected = sigma_strata.select_sigma_stratum(expected_index)
        try:
            sigma_strata.assert_selected_timestep_sigma(
                timestep=record.timestep,
                sigma=record.sigma,
                selected=selected,
            )
        except sigma_strata.InferenceSigmaStrataError as error:
            raise CrossModeCMSGInferenceError(
                f"runtime schedule differs at step {expected_index}"
            ) from error
        if record.step_index != expected_index or record.model_id != "transformer_1":
            raise CrossModeCMSGInferenceError("v6 step order/expert differs")
        if (
            record.transformer_forwards != 4
            or record.frozen_negative_forwards != 1
            or record.frozen_noop_forwards != 1
            or record.frozen_action_forwards != 1
            or record.adapted_action_forwards != 1
            or record.original_scheduler_calls != 1
            or record.generator_forwards != 0
        ):
            raise CrossModeCMSGInferenceError("v6 per-step branch call count differs")
        if record.adapter_loaded is not True:
            raise CrossModeCMSGInferenceError("v6 step lost its adapter")
        if (
            record.official_frozen_action_apg_exact is not True
            or record.official_frozen_action_apg_rms_error != 0.0
            or record.official_frozen_action_apg_max_abs_error != 0.0
        ):
            raise CrossModeCMSGInferenceError("frozen-action APG certificate failed")
        expected_rho = spectrum.release_rho(expected_index)
        if record.rho != expected_rho:
            raise CrossModeCMSGInferenceError("v6 release schedule differs")
        expected_exact = expected_rho == 0.0
        if record.exact_official_model_output_object is not expected_exact:
            raise CrossModeCMSGInferenceError(
                "late official scheduler object identity differs"
            )
        if expected_exact:
            exact_steps.append(expected_index)
            if (
                record.executed_direction_correction_rms != 0.0
                or record.scheduler_boundary_correction_rms != 0.0
            ):
                raise CrossModeCMSGInferenceError(
                    "zero-release step is not exact official replay"
                )
        if record.executed_first_phase_max_abs != 0.0 or record.phase_cells <= 0:
            raise CrossModeCMSGInferenceError("causal phase-zero contract differs")
        for item in (
            record.sigma,
            record.frozen_editor_direction_rms,
            record.adapted_editor_direction_rms,
            record.raw_direction_delta_rms,
            record.executed_direction_correction_rms,
            record.scheduler_boundary_correction_rms,
        ):
            if not math.isfinite(float(item)) or float(item) < 0.0:
                raise CrossModeCMSGInferenceError("v6 trace diagnostic is invalid")
        sigmas.append(float(record.sigma))
    if any(following >= current for current, following in zip(sigmas, sigmas[1:])):
        raise CrossModeCMSGInferenceError("UniPC sigmas must strictly descend")
    if tuple(exact_steps) != LATE_EXACT_STEPS:
        raise CrossModeCMSGInferenceError("v6 exact official replay steps differ")
    payload = {
        "cross_mode_cmsg": trace.as_dict(),
        "runtime_unipc_schedule_audit": audited,
        "certificate": {
            "step_count": NUM_DENOISING_STEPS,
            "original_unipc_calls": NUM_DENOISING_STEPS,
            "editor_transformer_forwards": 4 * NUM_DENOISING_STEPS,
            "generator_forwards": 0,
            "exact_official_model_output_steps": list(LATE_EXACT_STEPS),
            "formal_adapter_off_steps": list(FORMAL_ADAPTER_OFF_STEPS),
            "custom_integrator": False,
            "mask_flow_pose_inputs": False,
            "target_input": False,
            "generator_loaded": False,
        },
    }
    payload["trace_digest"] = trainer.object_sha256(payload)
    return payload


def preflight_contract() -> dict[str, Any]:
    """Validate the insertable operator without claiming full CLI execution."""

    targets = expected_lora_targets()
    contract = runtime_contract()
    schedule = tuple(contract["release_schedule"])
    if (
        len(schedule) != NUM_DENOISING_STEPS
        or schedule[:20] != (1.0,) * 20
        or schedule[20] != 1.0
        or schedule[31] != 0.0
        or schedule[32:] != (0.0,) * 8
        or LATE_EXACT_STEPS != tuple(range(31, 40))
    ):
        raise CrossModeCMSGInferenceError("v6 immutable release schedule differs")
    return {
        "method": METHOD_NAME,
        "training_receipt_schema": TRAINING_RECEIPT_SCHEMA,
        "inference_receipt_schema": INFERENCE_RECEIPT_SCHEMA,
        "strict_operator_ready": True,
        "four_branch_hook_ready": True,
        "strict_receipt_loader_ready": True,
        "standalone_full_cli_integrated": False,
        "production_inference_claim_authorized": False,
        "frames": NUM_FRAMES,
        "latent_phases": LATENT_PHASES,
        "lora_scope": REQUIRED_LORA_SCOPE,
        "lora_target_count": len(targets),
        "lora_targets": targets,
        "runtime": contract,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight the insertable Bernini Cross-Mode CMSG v6 inference operator"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate contracts without claiming a standalone video run",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.preflight_only:
        raise CrossModeCMSGInferenceError(
            "standalone Bernini v6 CLI is not integrated; use the audited "
            "cross_mode_cmsg_unipc_hook or run --preflight-only"
        )
    print(json.dumps(preflight_contract(), sort_keys=True))
    return 0


__all__ = [
    "CrossModeCMSGInferenceError",
    "CrossModeCMSGStepRecord",
    "CrossModeCMSGTrace",
    "RawCrossModeCMSGStep",
    "build_parser",
    "cross_mode_cmsg_unipc_hook",
    "expected_immutable_training_contract",
    "expected_lora_targets",
    "expected_training_supervision_contract",
    "main",
    "packed_to_phase_grid",
    "phase_grid_to_packed",
    "preflight_contract",
    "project_cross_mode_cmsg_step",
    "runtime_contract",
    "strict_load_adapter",
    "validate_execution_trace",
    "validate_runtime_schedule_audit",
    "validate_training_adapter_contract",
]


if __name__ == "__main__":
    raise SystemExit(main())
