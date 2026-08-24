#!/usr/bin/env python3
"""Strict adapter contract for a finalized Bernini RS-FQT LoRA v8 pilot.

The v8 trainer serializes a pending receipt.  A separate finalizer must prove
that the exact PEFT artifact can be reconstructed on a fresh pinned Bernini-R
1.3B base before changing that receipt to ready.  This module is the
fail-closed consumer of the resulting receipt: it accepts neither a v7
checkpoint nor a pending/canary v8 checkpoint, and delegates tensor loading to
the shared exact-target loader only after the complete v8 contract is proven.

There is no video, prompt, target, mask, tracking, or sampling interface here.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import feasible_quotient_objective as objective  # noqa: E402
import infer_relational_motion_commutator as rmc  # noqa: E402
import motion_commutator as commutator  # noqa: E402
import train_feasible_quotient_auh as v8_train  # noqa: E402


legacy = v8_train.legacy
sigma_strata = v8_train.sigma_strata

METHOD_NAME = v8_train.METHOD_NAME
TRAINING_RECEIPT_SCHEMA = v8_train.RECEIPT_SCHEMA
ARTIFACT_VALIDATION_SCHEMA = v8_train.ARTIFACT_VALIDATION_SCHEMA
INFERENCE_RECEIPT_SCHEMA = v8_train.INFERENCE_RECEIPT_SCHEMA
FORMAL_GLOBAL_STEP = v8_train.MAX_STEPS
PENDING_ARTIFACT_STATUS = "pending_post_save_strict_reload"
READY_ARTIFACT_STATUS = "post_save_strict_reload_complete"
LOADER_MODULE = Path(__file__).name
FINALIZER_MODULE = v8_train.INFERENCE_FINALIZER_MODULE

REQUIRED_LORA_SCOPE = v8_train.v6_scope.LORA_SCOPE
REQUIRED_TARGET_MODULE_COUNT = v8_train.v6_scope.EXPECTED_LORA_MODULES
REQUIRED_LORA_RANK = v8_train.v6_scope.LORA_RANK
REQUIRED_LORA_ALPHA = v8_train.v6_scope.LORA_ALPHA

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class FeasibleQuotientInferenceError(RuntimeError):
    """Raised before an unfinalized or altered v8 adapter can be loaded."""


def expected_lora_targets() -> list[str]:
    """Return the canonical exact-46 target list shared with v8 training."""

    targets = rmc.expected_lora_targets()
    if (
        targets != sorted(set(targets))
        or len(targets) != REQUIRED_TARGET_MODULE_COUNT
    ):
        raise FeasibleQuotientInferenceError(
            "canonical v8 LoRA target set is not sorted exact-46"
        )
    return targets


def expected_serialized_target_patterns() -> list[str]:
    """Return PEFT 0.19.1's audited 17-pattern exact-46 serialization."""

    patterns = rmc.expected_serialized_target_patterns()
    if len(patterns) != 17 or patterns != sorted(set(patterns)):
        raise FeasibleQuotientInferenceError(
            "canonical v8 serialized target set is not exact-17"
        )
    return patterns


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeasibleQuotientInferenceError(f"receipt lacks {label}")
    return value


def _require_sha(
    value: Any, *, label: str, pattern: re.Pattern[str]
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise FeasibleQuotientInferenceError(f"{label} is invalid")
    return value


def _validate_receipt_digest(receipt: Mapping[str, Any]) -> str:
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    _require_sha(digest, label="training receipt digest", pattern=_SHA256_RE)
    if legacy.object_sha256(candidate) != digest:
        raise FeasibleQuotientInferenceError("training receipt digest differs")
    return digest


def _expected_parity_contract() -> dict[str, Any]:
    return {
        "verified": True,
        "verification_stage": (
            "immutable_v8_loader_finalizer_and_runner_preflight_before_model_load"
        ),
        "loader_module": v8_train.INFERENCE_LOADER_MODULE,
        "runner_module": v8_train.INFERENCE_RUNNER_MODULE,
        "finalizer_module": v8_train.INFERENCE_FINALIZER_MODULE,
        "training_receipt_schema": TRAINING_RECEIPT_SCHEMA,
        "inference_receipt_schema": INFERENCE_RECEIPT_SCHEMA,
        "contract_tests": list(v8_train.INFERENCE_PARITY_TESTS),
        "source_revision_and_archive_bound": True,
        "strict_loader_rejects_pending_canary_and_incomplete_cycle": True,
        "post_save_v8_loader_finalization_required": True,
    }


def _expected_fixed_immutable_fields() -> dict[str, Any]:
    loss_config = objective.FeasibleQuotientLossConfig()
    release = list(commutator.release_rho_schedule())
    return {
        "method": METHOD_NAME,
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "frames": v8_train.v7.NUM_FRAMES,
        "latent_phases": v8_train.v7.LATENT_PHASES,
        "checkpoint_content_manifest_sha256": (
            v8_train.CHECKPOINT_CONTENT_MANIFEST_SHA256
        ),
        "checkpoint_content_file_count": (
            v8_train.CHECKPOINT_CONTENT_FILE_COUNT
        ),
        "checkpoint_content_validation": (
            "pinned_sha256sum_manifest_before_torchrun_and_finalizer"
        ),
        "learning_rate": v8_train.LEARNING_RATE,
        "teacher_mode": "paired_displacement_only",
        "loss_config": asdict(loss_config),
        "objective_contract": objective.immutable_objective_contract(
            loss_config
        ),
        "training_diffusion_query": "target(beta=1)",
        "training_diffusion_query_formula": (
            "x_sigma=(1-sigma)*paired_target+sigma*epsilon"
        ),
        "paired_target_constructs_training_diffusion_state": True,
        "paired_target_used_as_external_model_condition": False,
        "forward_cell_order": list(v8_train.FORWARD_CELL_ORDER),
        "forwards_per_candidate": 5,
        "graph_forwards_per_candidate": 2,
        "graph_branch_order": list(objective.GRAPH_BRANCHES),
        "inference_forward_order": list(v8_train.INFERENCE_FORWARD_ORDER),
        "inference_forwards_per_step": 5,
        "editor_guidance": {
            "mode": "official_momentum_zero_apg",
            "guidance_scale": v8_train.v5.APG_GUIDANCE_SCALE,
            "eta": v8_train.v5.APG_ETA,
            "norm_threshold": v8_train.v5.APG_NORM_THRESHOLD,
            "momentum": v8_train.v5.APG_MOMENTUM,
            "clean_reconstruction": (
                "fp32_noisy_minus_sigma_times_native_bf16_velocity"
            ),
            "v8_frozen_action_section": (
                "local_fp32_apg_after_bit_exact_native_bf16_official_parity"
            ),
        },
        "target_motion_teacher": (
            "Q0(target_clean-stopgrad(frozen_noop_section))"
        ),
        "target_section_reference": (
            "same_query_frozen_noop_prevents_beta1_double_count"
        ),
        "target_used_as_model_condition": False,
        "training_operator": (
            "FIR(DQ0(Atheta-Ntheta)) with feasible canonical target"
        ),
        "deployment_operator": (
            "N0+rho*Integrate(Project_r(FIR(DQ0(Atheta-Ntheta))))"
        ),
        "deployment_velocity_precision": (
            "fp32_exact_clean_transport_with_post_boundary_radius_certificate"
        ),
        "appearance_carrier": "frozen_noop_reconstruction_section",
        "source_only_radius": (
            "max(frozen_quotient_rms,0.25*noop_dynamics_rms,1e-3)"
        ),
        "metrics_timing": v8_train.METRICS_TIMING,
        "release_schedule": release,
        "release_schedule_sha256": legacy.object_sha256(release),
        "zero_release_steps": [
            index for index, rho in enumerate(release) if rho == 0.0
        ],
        "zero_release_semantics": (
            "adam_moments_reset_before_suffix_then_current_noop_gradient_only"
        ),
        "zero_release_optimizer_boundary": {
            "first_zero_release_schedule_index": (
                v8_train.ZERO_RELEASE_START_INDEX
            ),
            "reset_before_optimizer_step": (
                v8_train.ZERO_RELEASE_START_INDEX + 1
            ),
            "reset_state": ["step", "exp_avg", "exp_avg_sq"],
            "reset_count": 1,
            "weight_decay": 0.0,
        },
        "sigma_schedule": "exact_40_step_flow_shift_5_cycle",
        "sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "sigma_selector": "absolute_global_step_mod_40",
        "inference_conditions": ["source_video", "action_instruction"],
        "training_only_conditions": ["paired_target_video"],
        "forbidden_inference_conditions": list(
            objective.FORBIDDEN_INFERENCE_CONDITIONS
        ),
        "training_generator_forwards": 0,
        "inference_generator_forwards": 0,
        "pilot_scope": "exact40_fixed_lr_falsification",
        "inference_loader_parity": _expected_parity_contract(),
        "resume_integrated": False,
    }


def _validate_artifact_transition(
    receipt: Mapping[str, Any],
    *,
    immutable_value: Mapping[str, Any],
    expected_checkpoint_tree_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    artifact = _require_mapping(
        receipt.get("artifact_validation"), label="artifact validation"
    )
    artifact_value = dict(artifact)
    artifact_digest = artifact_value.pop("digest", None)
    _require_sha(
        artifact_digest,
        label="artifact validation digest",
        pattern=_SHA256_RE,
    )
    if legacy.object_sha256(artifact_value) != artifact_digest:
        raise FeasibleQuotientInferenceError(
            "artifact validation digest differs"
        )
    expected = {
        "schema_version": ARTIFACT_VALIDATION_SCHEMA,
        "verified": True,
        "status": READY_ARTIFACT_STATUS,
        "serialized_target_pattern_count": 17,
        "expanded_target_module_count": 46,
        "adapter_tensor_count": 92,
        "active_lora_module_count": 46,
        "strict_tensor_reload_equal": True,
        "parameter_digest_verified_after_safetensors_reload": True,
        "validator_method_source_revision": immutable_value[
            "method_source_revision"
        ],
        "validator_method_source_archive_sha256": immutable_value[
            "method_source_archive_sha256"
        ],
        "bernini_commit": receipt["bernini_commit"],
        "veomni_commit": receipt["veomni_commit"],
        "checkpoint_tree_sha256": expected_checkpoint_tree_sha256,
        "checkpoint_content_manifest_sha256": (
            v8_train.CHECKPOINT_CONTENT_MANIFEST_SHA256
        ),
        "checkpoint_content_file_count": (
            v8_train.CHECKPOINT_CONTENT_FILE_COUNT
        ),
        "loader_module": LOADER_MODULE,
        "finalizer_module": FINALIZER_MODULE,
    }
    for name, wanted in expected.items():
        if artifact_value.get(name) != wanted:
            raise FeasibleQuotientInferenceError(
                f"post-save v8 artifact validation differs: {name}"
            )
    for name in (
        "adapter_config_sha256",
        "adapter_model_sha256",
        "checkpoint_parameter_digest",
        "pending_receipt_digest",
    ):
        _require_sha(
            artifact_value.get(name),
            label=f"artifact {name}",
            pattern=_SHA256_RE,
        )
    expected_keys = set(expected) | {
        "adapter_config_sha256",
        "adapter_model_sha256",
        "checkpoint_parameter_digest",
        "pending_receipt_digest",
    }
    if set(artifact_value) != expected_keys:
        raise FeasibleQuotientInferenceError(
            "artifact validation contains an unaudited field"
        )

    # Prove that finalization changed exactly the release gate and artifact
    # proof.  Every scientific/training field must reconstruct the hash-bound
    # pending receipt byte-for-byte at canonical-JSON level.
    reconstructed = copy.deepcopy(dict(receipt))
    reconstructed.pop("receipt_digest", None)
    reconstructed["inference_loader_parity_pending"] = True
    reconstructed["artifact_validation"] = {
        "schema_version": ARTIFACT_VALIDATION_SCHEMA,
        "verified": False,
        "status": PENDING_ARTIFACT_STATUS,
    }
    if legacy.object_sha256(reconstructed) != artifact_value[
        "pending_receipt_digest"
    ]:
        raise FeasibleQuotientInferenceError(
            "ready v8 receipt is not an exact pending-to-ready transition"
        )
    return artifact_value, artifact_digest


def validate_training_adapter_contract(
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = legacy.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Validate one finalized target-only exact-40 RS-FQT checkpoint."""

    if not isinstance(adapter_config, Mapping) or not isinstance(receipt, Mapping):
        raise FeasibleQuotientInferenceError(
            "adapter config and training receipt must be mappings"
        )
    receipt_digest = _validate_receipt_digest(receipt)
    if receipt.get("schema_version") != TRAINING_RECEIPT_SCHEMA:
        raise FeasibleQuotientInferenceError(
            "training receipt is not the RS-FQT v8 schema"
        )
    if receipt.get("method") != METHOD_NAME:
        raise FeasibleQuotientInferenceError(
            "training method is not RS-FQT v8"
        )
    if (
        receipt.get("global_step") != FORMAL_GLOBAL_STEP
        or receipt.get("max_steps") != FORMAL_GLOBAL_STEP
        or receipt.get("formal_40_sigma_cycle_complete") is not True
        or receipt.get("accepted_sigma_schedule_indices")
        != list(range(FORMAL_GLOBAL_STEP))
    ):
        raise FeasibleQuotientInferenceError(
            "v8 checkpoint is not one formal exact-40 sigma cycle"
        )
    step_audit = receipt.get("step_audit")
    if (
        not isinstance(step_audit, list)
        or len(step_audit) != FORMAL_GLOBAL_STEP
        or receipt.get("step_audit_sha256") != legacy.object_sha256(step_audit)
    ):
        raise FeasibleQuotientInferenceError(
            "v8 step audit is incomplete or altered"
        )
    for index, record in enumerate(step_audit):
        if (
            not isinstance(record, Mapping)
            or record.get("optimizer_step") != index + 1
            or record.get("sigma_schedule_index") != index
            or record.get("teacher_mode") != "paired_displacement_only"
            or record.get("metrics_timing") != v8_train.METRICS_TIMING
            or not math.isclose(
                float(record.get("rho", math.nan)),
                commutator.release_rho(index),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise FeasibleQuotientInferenceError(
                f"v8 step audit differs at index {index}"
            )

    immutable = _require_mapping(
        receipt.get("immutable_contract"), label="immutable contract"
    )
    value = _require_mapping(immutable.get("value"), label="immutable value")
    if immutable.get("digest") != legacy.object_sha256(value):
        raise FeasibleQuotientInferenceError(
            "immutable v8 training contract digest differs"
        )
    for name, wanted in _expected_fixed_immutable_fields().items():
        if value.get(name) != wanted:
            raise FeasibleQuotientInferenceError(
                f"immutable v8 field differs: {name}"
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
        _require_sha(value.get(name), label=f"immutable {name}", pattern=pattern)
    if (
        receipt.get("bernini_commit") != legacy.BERNINI_OFFICIAL_COMMIT
        or receipt.get("veomni_commit") != legacy.VEOMNI_TESTED_COMMIT
        or value.get("bernini_commit") != receipt.get("bernini_commit")
        or value.get("veomni_commit") != receipt.get("veomni_commit")
        or value.get("checkpoint_tree_sha256")
        != expected_checkpoint_tree_sha256
        or value.get("eligible_route_count") != 359
        or value.get("routing_file_sha256") != v8_train.v5.STRICT_ROUTING_SHA256
        or type(value.get("seed")) is not int
        or not isinstance(value.get("dataset_signature"), str)
        or not value.get("dataset_signature")
    ):
        raise FeasibleQuotientInferenceError(
            "v8 source, data, or routing identity differs"
        )
    checkpoint = _require_mapping(
        receipt.get("checkpoint"), label="base checkpoint identity"
    )
    if (
        checkpoint.get("tree_sha256") != expected_checkpoint_tree_sha256
        or checkpoint.get("path") != value.get("checkpoint_path")
    ):
        raise FeasibleQuotientInferenceError(
            "v8 base checkpoint identity differs"
        )
    dataset = _require_mapping(receipt.get("dataset"), label="dataset identity")
    summary = _require_mapping(dataset.get("summary"), label="dataset summary")
    routing = _require_mapping(dataset.get("routing"), label="routing identity")
    if (
        dataset.get("rows") != 644
        or dataset.get("signature") != value.get("dataset_signature")
        or summary.get("sha256") != value.get("dataset_summary_sha256")
        or summary.get("index_sha256") != value.get("dataset_index_sha256")
        or routing.get("default_tier") != "reject"
        or routing.get("explicit_route_counts")
        != {"full_pair": 0, "motion_only": 359, "reject": 285}
        or routing.get("file_sha256") != value.get("routing_file_sha256")
        or routing.get("routing_digest") != value.get("routing_digest")
    ):
        raise FeasibleQuotientInferenceError(
            "v8 strict-359 dataset receipt differs"
        )

    if (
        receipt.get("inference_conditions")
        != ["source_video", "action_instruction"]
        or receipt.get("training_only_generator_and_target") is not False
        or receipt.get("training_only_paired_target") is not True
        or receipt.get("training_generator_forwards") != 0
        or receipt.get("inference_generator_forwards") != 0
        or receipt.get("teacher_mode") != "paired_displacement_only"
        or receipt.get("pilot_scope") != "exact40_fixed_lr_falsification"
        or receipt.get("external_mask_track_flow_pose_trajectory") is not False
        or receipt.get("first_frame_anchor") is not False
        or receipt.get("experimental_training") is not True
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
        or receipt.get("resume_integrated") is not False
        or receipt.get("inference_loader_parity_pending") is not False
        or receipt.get("metrics_timing") != v8_train.METRICS_TIMING
        or receipt.get("inference_loader_parity") != _expected_parity_contract()
        or receipt.get("inference_loader_parity")
        != value.get("inference_loader_parity")
    ):
        raise FeasibleQuotientInferenceError(
            "v8 publication or inference-boundary state differs"
        )

    artifact, artifact_digest = _validate_artifact_transition(
        receipt,
        immutable_value=value,
        expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
    )

    targets = expected_lora_targets()
    target_digest = legacy.object_sha256(targets)
    immutable_lora = _require_mapping(value.get("lora"), label="immutable LoRA")
    adapter = _require_mapping(receipt.get("adapter"), label="adapter identity")
    expected_adapter = {
        "rank": 8,
        "alpha": 8,
        "scope": REQUIRED_LORA_SCOPE,
        "target_module_count": 46,
        "target_modules": targets,
        "target_modules_sha256": target_digest,
    }
    for name, wanted in expected_adapter.items():
        if immutable_lora.get(name) != wanted or adapter.get(name) != wanted:
            raise FeasibleQuotientInferenceError(
                f"v8 exact-46 adapter field differs: {name}"
            )
    if (
        immutable_lora.get("dropout") != 0.0
        or immutable_lora.get("bias") != "none"
    ):
        raise FeasibleQuotientInferenceError("immutable v8 LoRA extras differ")
    for name in (
        "initialization_digest",
        "checkpoint_parameter_digest",
        "parameter_names_sha256",
    ):
        _require_sha(adapter.get(name), label=f"adapter {name}", pattern=_SHA256_RE)
    if (
        artifact.get("checkpoint_parameter_digest")
        != adapter.get("checkpoint_parameter_digest")
        or type(adapter.get("trainable_parameter_count")) is not int
        or adapter["trainable_parameter_count"] <= 0
    ):
        raise FeasibleQuotientInferenceError(
            "v8 adapter parameter identity differs"
        )

    optimizer = _require_mapping(receipt.get("optimizer"), label="optimizer")
    zero_release_reset = _require_mapping(
        optimizer.get("zero_release_moment_reset"),
        label="zero-release optimizer moment reset",
    )
    parameter_names = optimizer.get("parameter_names")
    weight_decay = value.get("weight_decay")
    max_grad_norm = value.get("max_grad_norm")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(float(weight_decay))
        or float(weight_decay) != 0.0
        or isinstance(max_grad_norm, bool)
        or not isinstance(max_grad_norm, (int, float))
        or not math.isfinite(float(max_grad_norm))
        or float(max_grad_norm) <= 0.0
        or optimizer.get("type") != "AdamW"
        or float(optimizer.get("learning_rate", math.nan))
        != v8_train.LEARNING_RATE
        or float(optimizer.get("weight_decay", math.nan)) != float(weight_decay)
        or float(optimizer.get("max_gradient_norm", math.nan))
        != float(max_grad_norm)
        or not isinstance(parameter_names, list)
        or not parameter_names
        or not all(isinstance(name, str) and name for name in parameter_names)
        or len(parameter_names) != len(set(parameter_names))
        or adapter.get("parameter_names_sha256")
        != legacy.object_sha256(parameter_names)
        or zero_release_reset
        != {
            "first_zero_release_schedule_index": 31,
            "reset_before_optimizer_step": 32,
            "completed_optimizer_steps": 40,
            "reset_count": 1,
            "state_step_after_reset_suffix": 9,
            "state_step_values": [9],
            "state_parameter_count": len(parameter_names),
            "weight_decay": 0.0,
        }
    ):
        raise FeasibleQuotientInferenceError(
            "v8 optimizer or parameter-name identity differs"
        )
    _require_sha(
        optimizer.get("checkpoint_state_digest"),
        label="optimizer checkpoint digest",
        pattern=_SHA256_RE,
    )
    distributed = _require_mapping(
        receipt.get("distributed"), label="distributed identity"
    )
    if (
        distributed.get("world_size") != 4
        or distributed.get("ulysses_size") != 4
        or distributed.get("same_pair_all_ranks") is not True
        or distributed.get("explicit_lora_gradient_all_reduce") is not True
        or not isinstance(distributed.get("backend"), str)
        or not distributed.get("backend")
    ):
        raise FeasibleQuotientInferenceError(
            "v8 four-rank Ulysses identity differs"
        )

    if adapter_config.get("peft_type") != "LORA":
        raise FeasibleQuotientInferenceError("v8 adapter is not LoRA")
    if adapter_config.get("r") != REQUIRED_LORA_RANK:
        raise FeasibleQuotientInferenceError("v8 adapter rank differs")
    try:
        alpha = float(adapter_config.get("lora_alpha", math.nan))
        dropout = float(adapter_config.get("lora_dropout", math.nan))
    except (TypeError, ValueError) as error:
        raise FeasibleQuotientInferenceError(
            "v8 adapter alpha or dropout is invalid"
        ) from error
    if alpha != float(REQUIRED_LORA_ALPHA) or dropout != 0.0:
        raise FeasibleQuotientInferenceError(
            "v8 adapter alpha or dropout differs"
        )
    if (
        adapter_config.get("bias") != "none"
        or adapter_config.get("modules_to_save") not in (None, [])
        or adapter_config.get("use_dora") not in (None, False)
        or adapter_config.get("use_rslora") not in (None, False)
    ):
        raise FeasibleQuotientInferenceError(
            "v8 adapter contains unsupported PEFT features"
        )
    serialized = adapter_config.get("target_modules")
    patterns = expected_serialized_target_patterns()
    if (
        not isinstance(serialized, list)
        or len(serialized) != len(patterns)
        or not all(isinstance(name, str) and name for name in serialized)
        or len(serialized) != len(set(serialized))
        or set(serialized) != set(patterns)
        or rmc._expand_serialized_target_patterns(serialized) != targets
    ):
        raise FeasibleQuotientInferenceError(
            "v8 serialized targets are not the exact17-to-exact46 expansion"
        )
    transformers_version = receipt.get("transformers_version")
    if not isinstance(transformers_version, str) or not transformers_version:
        raise FeasibleQuotientInferenceError(
            "v8 training Transformers version is missing"
        )
    return {
        "receipt_digest": receipt_digest,
        "training_receipt_schema": TRAINING_RECEIPT_SCHEMA,
        "training_method": METHOD_NAME,
        "projection_consistent_objective": True,
        "global_step": FORMAL_GLOBAL_STEP,
        "scope": REQUIRED_LORA_SCOPE,
        "targets": targets,
        "serialized_target_modules": sorted(serialized),
        "target_modules_sha256": target_digest,
        "initialization_digest": adapter["initialization_digest"],
        "checkpoint_parameter_digest": adapter[
            "checkpoint_parameter_digest"
        ],
        "transformers_version": transformers_version,
        "training_method_source_revision": value["method_source_revision"],
        "training_method_source_archive_sha256": value[
            "method_source_archive_sha256"
        ],
        "artifact_validation_digest": artifact_digest,
        "adapter_config_sha256": artifact["adapter_config_sha256"],
        "adapter_model_sha256": artifact["adapter_model_sha256"],
    }


def _bundle_file_hashes(bundle: Any) -> tuple[str, str]:
    try:
        config_path = Path(bundle.adapter_config_path)
        model_path = Path(bundle.adapter_model_path)
        return legacy.file_sha256(config_path), legacy.file_sha256(model_path)
    except (AttributeError, OSError, TypeError) as error:
        raise FeasibleQuotientInferenceError(
            "cannot hash the resolved v8 adapter bundle"
        ) from error


def strict_load_adapter(
    *,
    base_model: Any,
    bundle: Any,
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    expected_checkpoint_tree_sha256: str = legacy.CHECKPOINT_TREE_SHA256,
) -> tuple[Any, int, int, dict[str, Any]]:
    """Strict-reload exact46/92 and prove stable serialized artifact hashes."""

    identity = validate_training_adapter_contract(
        adapter_config,
        receipt,
        expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
    )
    hashes_before = _bundle_file_hashes(bundle)
    if hashes_before != (
        identity["adapter_config_sha256"],
        identity["adapter_model_sha256"],
    ):
        raise FeasibleQuotientInferenceError(
            "v8 adapter files differ from finalized artifact hashes"
        )
    try:
        model, tensor_count, active_count = rmc.v5._strict_load_adapter(
            base_model=base_model,
            bundle=bundle,
            adapter_config=adapter_config,
            identity=identity,
        )
    except rmc.v5.PriorTangentInferenceError as error:
        raise FeasibleQuotientInferenceError(str(error)) from error
    if tensor_count != 92 or active_count != 46:
        raise FeasibleQuotientInferenceError(
            "reloaded v8 adapter is not exact-46/92-tensor LoRA"
        )
    if _bundle_file_hashes(bundle) != hashes_before:
        raise FeasibleQuotientInferenceError(
            "v8 adapter files changed during strict reload"
        )
    return model, tensor_count, active_count, identity


__all__ = [
    "ARTIFACT_VALIDATION_SCHEMA",
    "FINALIZER_MODULE",
    "FORMAL_GLOBAL_STEP",
    "FeasibleQuotientInferenceError",
    "INFERENCE_RECEIPT_SCHEMA",
    "LOADER_MODULE",
    "METHOD_NAME",
    "PENDING_ARTIFACT_STATUS",
    "READY_ARTIFACT_STATUS",
    "TRAINING_RECEIPT_SCHEMA",
    "expected_lora_targets",
    "expected_serialized_target_patterns",
    "strict_load_adapter",
    "validate_training_adapter_contract",
]
