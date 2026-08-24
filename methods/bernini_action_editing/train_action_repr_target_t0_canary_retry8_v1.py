#!/usr/bin/env python3
"""Retry8 target-T0 canary with a same-runtime G2a equivalence gate.

Retry7 correctly failed before AdamW because a source VAE posterior that was
rematerialized on a different AUH run was not bit-identical to the historical
production-G2a posterior.  That historical digest remains authenticated
evidence, but it is not a sound cross-run bitwise gate for ROCm BF16 VAE
kernels.  Retry8 therefore replays the native route and all six zero-init G2a
routes on the *exact same in-memory source-owned FM batch* that enters the one
optimizer update.  Historical G2a comparisons are recorded honestly and are
never relabelled as same-run equality.

The sealed retry7 implementation remains unchanged.  This additive wrapper
reuses its optimizer/publication core while replacing only revision paths,
authority validation, the runtime-equivalence boundary, and receipt checks.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any, Callable, ContextManager, Mapping, Optional


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch

import train_action_repr_target_t0_canary_retry7_v1 as retry7


SCHEMA_VERSION = "bernini-action-repr-target-t0-one-step-retry8-receipt-v1"
STEP_SCHEMA_VERSION = "bernini-action-repr-target-t0-retry8-adapter-state-receipt-v1"
METHOD = "bernini-action-repr-target-t0-one-step-retry8-canary-v1"
ATTEMPT_CLAIM_SCHEMA_VERSION = (
    "bernini-action-repr-target-t0-retry8-preoptimizer-attempt-claim-v1"
)
ATTEMPT_CLAIM_MARKER_NAME = ".single_update.retry8.attempt_claim.json"
EXPECTED_CANONICAL_OUTPUT_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/action_repr_target_selfgen_middle_g1_20260824_v2/"
    "stage_b_t0_retry8/target_t0/0be6494dfac3/single_update"
)
EXPECTED_ATTEMPT_CLAIM_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/action_repr_target_selfgen_middle_g1_20260824_v2/"
    "stage_b_t0_retry8/target_t0/0be6494dfac3/"
    ".single_update.retry8.attempt_claim.json"
)
AUTHORITY_SCHEMA_VERSION = (
    "bernini-action-repr-stage-b-t0-single-update-retry8-authority-addendum-v1"
)
AUTHORITY_DOCUMENT_ROLE = (
    "create_once_stage_b_target_t0_single_update_retry8_authority"
)
AUTHORITY_ACTIVATION_RULE = (
    "copy_to_stage_b_t0_single_update_retry8_authority_addendum.json_once_only_"
    "after_runner_tests_launcher_and_batch_replay_diagnostic_are_final_then_"
    "replace_every_explicit_sha256_placeholder_and_set_state_"
    "ACTIVE_CREATE_ONCE_AUTHORITY"
)

EXPECTED_SOURCE_PIN_PATHS = frozenset(
    {
        "methods/bernini_action_editing/train_action_repr_target_t0_canary_retry8_v1.py",
        "methods/bernini_action_editing/train_action_repr_target_t0_canary_retry7_v1.py",
        "methods/bernini_action_editing/action_representation_joint_objective_v1.py",
        "methods/bernini_action_editing/action_repr_g2a_adapter_v1.py",
        "methods/bernini_action_editing/audit_action_repr_g2a_world4_v1.py",
        "methods/bernini_action_editing/score_g1_joint_action_repr_admission_v1.py",
        "methods/bernini_action_editing/evaluate_g1_action_repr_selectivity_v1.py",
        "methods/bernini_action_editing/materialize_g1_flow_control_cohort_v1.py",
        "methods/bernini_action_editing/materialize_g1_middle_control_cohort_v1.py",
        "methods/bernini_action_editing/materialize_decoded_middle_action_repr_v1.py",
        "methods/bernini_action_editing/dense_flow_token_adapter_v1.py",
        "methods/bernini_action_editing/exact_local_video_materializer_v1.py",
        "methods/bernini_action_editing/train_lora.py",
        "methods/bernini_action_editing/train_self_generated_action_quotient_v1.py",
        "methods/bernini_action_editing/scripts/diagnose_stage_b_t0_batch_replay_v1.py",
        "methods/bernini_action_editing/scripts/auh_stage_b_t0_single_update_20260824_retry8.sh",
        "methods/bernini_action_editing/tests/test_train_action_repr_target_t0_canary_retry8_v1.py",
        "tests/test_auh_stage_b_t0_single_update_20260824_retry8.py",
    }
)


_ORIGINAL_PREVALIDATE_AUTHORITY = retry7.prevalidate_authorization_replay_contract
_ORIGINAL_VALIDATE_AUTHORITY = retry7.validate_authorization_addendum
_ORIGINAL_RUN_ONE_STEP = retry7.run_one_step_optimizer_canary
_ORIGINAL_VALIDATE_RECEIPT = retry7.validate_t0_receipt


def _require_retry8_authority_shape(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        retry7.fail("retry8 authority must be one object")
    row = dict(value)
    activation = row.get("activation")
    representation = row.get("representation_contract")
    diagnostic = (
        representation.get("batch_replay_diagnostic")
        if isinstance(representation, Mapping)
        else None
    )
    same_runtime = (
        representation.get("same_runtime_g2a_contract")
        if isinstance(representation, Mapping)
        else None
    )
    runtime_paths = row.get("runtime_paths")
    output = row.get("output_contract")
    if (
        row.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or row.get("document_role") != AUTHORITY_DOCUMENT_ROLE
        or not isinstance(activation, Mapping)
        or activation.get("activation_rule") != AUTHORITY_ACTIVATION_RULE
        or not isinstance(runtime_paths, Mapping)
        or runtime_paths.get("fresh_source_root_name") != "source_stage_b_t0_retry8"
        or runtime_paths.get("fresh_stage_root_name") != "stage_b_t0_retry8"
        or runtime_paths.get("fresh_log_root_name") != "logs/stage_b_t0_retry8"
        or not isinstance(output, Mapping)
        or output.get("attempt_claim_marker_name") != ATTEMPT_CLAIM_MARKER_NAME
        or output.get("canonical_output_path") != EXPECTED_CANONICAL_OUTPUT_PATH
        or output.get("attempt_claim_marker_path") != EXPECTED_ATTEMPT_CLAIM_PATH
        or output.get("next_revision_after_claim_failure") != "retry9"
        or not isinstance(diagnostic, Mapping)
        or diagnostic.get("schema_version")
        != "bernini-action-repr-t0-batch-replay-diagnostic-v1"
        or diagnostic.get("passed") is not True
        or diagnostic.get("diagnostic_only") is not True
        or diagnostic.get("renderer_model_loaded") is not False
        or diagnostic.get("optimizer_created") is not False
        or diagnostic.get("optimization_steps") != 0
        or diagnostic.get("tensor_payload_persisted") is not False
        or diagnostic.get("historical_posterior_match") is not False
        or diagnostic.get("historical_batch_match") is not False
        or not isinstance(same_runtime, Mapping)
        or same_runtime
        != {
            "historical_g2a_receipt_remains_authenticated_reference": True,
            "historical_rematerialized_batch_bitwise_equality_required": False,
            "same_in_memory_source_owned_batch_required": True,
            "pre_adapter_native_baseline_required": True,
            "route_off_and_six_zero_init_routes_exact_native_bits_required": True,
            "same_batch_digest_stable_through_all_forwards_required": True,
            "target_or_anchor_media_accessed_by_renderer": False,
        }
    ):
        retry7.fail("retry8 same-runtime G2a authority contract differs")
    diagnostic_path, diagnostic_value, diagnostic_sha = retry7.g2a_world4.read_json(
        diagnostic.get("path"),
        label="retry8 batch replay diagnostic",
        expected_sha256=diagnostic.get("file_sha256"),
    )
    if (
        str(diagnostic_path) != diagnostic.get("path")
        or diagnostic_sha != diagnostic.get("file_sha256")
        or diagnostic_value.get("schema_version") != diagnostic.get("schema_version")
        or diagnostic_value.get("passed") is not True
        or diagnostic_value.get("diagnostic_only") is not True
        or diagnostic_value.get("renderer_model_loaded") is not False
        or diagnostic_value.get("optimizer_created") is not False
        or diagnostic_value.get("optimization_steps") != 0
        or diagnostic_value.get("tensor_payload_persisted") is not False
        or diagnostic_value.get("receipt_digest") != diagnostic.get("receipt_digest")
        or diagnostic_value.get("rematerialized", {}).get(
            "source_posterior_matches_historical"
        )
        is not False
        or diagnostic_value.get("rematerialized", {}).get(
            "matched_native_batch_matches_historical"
        )
        is not False
    ):
        retry7.fail("retry8 batch replay diagnostic evidence differs")
    return row


def _as_retry7_authority_for_shared_validation(value: Any) -> Mapping[str, Any]:
    row = deepcopy(_require_retry8_authority_shape(value))
    row["schema_version"] = (
        "bernini-action-repr-stage-b-t0-single-update-retry7-authority-addendum-v1"
    )
    row["document_role"] = (
        "create_once_stage_b_target_t0_single_update_retry7_authority"
    )
    row["activation"]["activation_rule"] = (
        "copy_to_stage_b_t0_single_update_retry7_authority_addendum.json_"
        "once_only_after_runner_tests_and_launcher_are_final_then_"
        "replace_every_explicit_sha256_placeholder_and_set_state_"
        "ACTIVE_CREATE_ONCE_AUTHORITY"
    )
    row["runtime_paths"]["fresh_source_root_name"] = "source_stage_b_t0_retry7"
    row["runtime_paths"]["fresh_stage_root_name"] = "stage_b_t0_retry7"
    row["runtime_paths"]["fresh_log_root_name"] = "logs/stage_b_t0_retry7"
    row["output_contract"]["next_revision_after_claim_failure"] = "retry8"
    return row


def prevalidate_authorization_replay_contract(value: Any) -> Mapping[str, Any]:
    _ORIGINAL_PREVALIDATE_AUTHORITY(_as_retry7_authority_for_shared_validation(value))
    return value


def validate_authorization_addendum(value: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
    _ORIGINAL_VALIDATE_AUTHORITY(
        _as_retry7_authority_for_shared_validation(value), **kwargs
    )
    return value


def run_one_step_optimizer_canary(
    *,
    model: torch.nn.Module,
    forward_native: Callable[[], torch.Tensor],
    input_digest: Callable[[], str],
    routes: Mapping[str, Any],
    feature_projection: torch.Tensor,
    hidden_width: int,
    middle_width: int,
    expected_input_digest: str,
    expected_base_digest: str,
    expected_native_output_digest: str,
    bottleneck_width: int = retry7.BOTTLENECK_WIDTH,
    learning_rate: float = retry7.LEARNING_RATE,
    adapter_seed: int = retry7.ADAPTER_SEED,
    serial_cpu_audit: Callable[[], ContextManager[Any]] = nullcontext,
) -> retry7.OneStepResult:
    """Bind retry8 to an exact same-runtime native/G2a baseline."""

    historical_input = retry7.g2a_world4.require_sha256(
        expected_input_digest, label="historical production G2a batch"
    )
    historical_base = retry7.g2a_world4.require_sha256(
        expected_base_digest, label="historical production G2a renderer base"
    )
    historical_native = retry7.g2a_world4.require_sha256(
        expected_native_output_digest, label="historical production G2a native output"
    )
    same_input = retry7.g2a_world4.require_sha256(
        input_digest(), label="same-runtime source-only FM batch"
    )
    with serial_cpu_audit():
        same_base = retry7.g2a_world4.renderer_base_snapshot(model).digest
    if input_digest() != same_input:
        retry7.fail("retry8 source-only batch changed before same-runtime native baseline")
    with torch.inference_mode():
        native = forward_native()
    if input_digest() != same_input:
        retry7.fail("retry8 source-only batch changed during same-runtime native baseline")
    if not isinstance(native, torch.Tensor) or not bool(
        torch.isfinite(native.detach()).all().item()
    ):
        retry7.fail("retry8 same-runtime native baseline is not finite")
    same_native = retry7.g2a.tensor_sha256(native.detach())
    retry7._consensus(
        {
            "source_batch_sha256": same_input,
            "renderer_base_sha256": same_base,
            "native_output_sha256": same_native,
        },
        label="retry8 same-runtime pre-adapter G2a baseline",
    )
    del native

    result = _ORIGINAL_RUN_ONE_STEP(
        model=model,
        forward_native=forward_native,
        input_digest=input_digest,
        routes=routes,
        feature_projection=feature_projection,
        hidden_width=hidden_width,
        middle_width=middle_width,
        expected_input_digest=same_input,
        expected_base_digest=same_base,
        expected_native_output_digest=same_native,
        bottleneck_width=bottleneck_width,
        learning_rate=learning_rate,
        adapter_seed=adapter_seed,
        serial_cpu_audit=serial_cpu_audit,
    )
    facts = dict(result.facts)
    facts["matched_production_g2a_source_batch"] = same_input == historical_input
    facts["matched_production_g2a_renderer_base"] = same_base == historical_base
    facts["matched_production_g2a_native_output"] = same_native == historical_native
    facts["cross_run_historical_match_required"] = False
    facts["historical_production_g2a_reference"] = {
        "source_batch_sha256": historical_input,
        "renderer_base_sha256": historical_base,
        "native_output_sha256": historical_native,
        "source_batch_matches_same_runtime": same_input == historical_input,
        "renderer_base_matches_same_runtime": same_base == historical_base,
        "native_output_matches_same_runtime": same_native == historical_native,
        "authenticated_reference_only": True,
    }
    facts["same_runtime_g2a_gate"] = {
        "source_batch_sha256": same_input,
        "renderer_base_sha256": same_base,
        "native_output_sha256": same_native,
        "pre_adapter_native_baseline_executed": True,
        "same_batch_used_by_optimizer_canary": True,
        "route_off_and_six_zero_init_routes_exact_native_bits": True,
        "batch_digest_stable_through_all_forwards": True,
    }
    return retry7.OneStepResult(
        step0_state=result.step0_state,
        step1_state=result.step1_state,
        facts=facts,
    )


def validate_t0_receipt(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        retry7.fail("retry8 T0 receipt must be one object")
    receipt = dict(value)
    training = receipt.get("training")
    authority = receipt.get("upstream_authority")
    source_lock = receipt.get("source_lock")
    if not isinstance(training, Mapping) or not isinstance(authority, Mapping):
        retry7.fail("retry8 receipt training/authority is absent")
    historical = training.get("historical_production_g2a_reference")
    same = training.get("same_runtime_g2a_gate")
    if (
        training.get("cross_run_historical_match_required") is not False
        or not isinstance(historical, Mapping)
        or historical.get("authenticated_reference_only") is not True
        or historical.get("source_batch_sha256")
        != authority.get("production_g2a_matched_native_batch_sha256")
        or historical.get("renderer_base_sha256")
        != authority.get("production_g2a_renderer_base_snapshot_digest")
        or historical.get("native_output_sha256")
        != authority.get("production_g2a_native_post_head_tensor_sha256")
        or historical.get("source_batch_matches_same_runtime")
        is not (
            training.get("matched_source_owned_batch_sha256")
            == historical.get("source_batch_sha256")
        )
        or historical.get("renderer_base_matches_same_runtime")
        is not (
            training.get("renderer_base_snapshot_digest_before")
            == historical.get("renderer_base_sha256")
        )
        or historical.get("native_output_matches_same_runtime")
        is not (
            training.get("native_step0_output_sha256")
            == historical.get("native_output_sha256")
        )
        or training.get("matched_production_g2a_source_batch")
        is not historical.get("source_batch_matches_same_runtime")
        or training.get("matched_production_g2a_renderer_base")
        is not historical.get("renderer_base_matches_same_runtime")
        or training.get("matched_production_g2a_native_output")
        is not historical.get("native_output_matches_same_runtime")
        or not isinstance(same, Mapping)
        or same
        != {
            "source_batch_sha256": training.get("matched_source_owned_batch_sha256"),
            "renderer_base_sha256": training.get(
                "renderer_base_snapshot_digest_before"
            ),
            "native_output_sha256": training.get("native_step0_output_sha256"),
            "pre_adapter_native_baseline_executed": True,
            "same_batch_used_by_optimizer_canary": True,
            "route_off_and_six_zero_init_routes_exact_native_bits": True,
            "batch_digest_stable_through_all_forwards": True,
        }
    ):
        retry7.fail("retry8 historical/same-runtime G2a receipt boundary differs")
    expected_source_names = {
        "train_action_repr_target_t0_canary_retry8_v1.py",
        "action_repr_g2a_adapter_v1.py",
        "action_representation_joint_objective_v1.py",
        "audit_action_repr_g2a_world4_v1.py",
        "score_g1_joint_action_repr_admission_v1.py",
        "evaluate_g1_action_repr_selectivity_v1.py",
        "materialize_g1_flow_control_cohort_v1.py",
        "materialize_g1_middle_control_cohort_v1.py",
        "materialize_decoded_middle_action_repr_v1.py",
        "dense_flow_token_adapter_v1.py",
        "exact_local_video_materializer_v1.py",
        "train_lora.py",
        "train_self_generated_action_quotient_v1.py",
    }
    if not isinstance(source_lock, Mapping) or set(source_lock) != expected_source_names:
        retry7.fail("retry8 runtime source-lock closure differs")
    if source_lock["train_action_repr_target_t0_canary_retry8_v1.py"] != retry7.file_sha256(
        Path(__file__).resolve()
    ):
        retry7.fail("retry8 runtime runner source lock differs")

    # Reuse the mature retry7 structural validator after projecting only the
    # three intentionally changed cross-run semantics back to its legacy view.
    projected = deepcopy(receipt)
    projected_training = projected["training"]
    projected_training.pop("historical_production_g2a_reference")
    projected_training.pop("same_runtime_g2a_gate")
    projected_training.pop("cross_run_historical_match_required")
    projected_training["matched_production_g2a_source_batch"] = True
    projected_training["matched_production_g2a_renderer_base"] = True
    projected_training["matched_production_g2a_native_output"] = True
    projected_training["matched_source_owned_batch_sha256"] = authority[
        "production_g2a_matched_native_batch_sha256"
    ]
    projected_training["native_step0_output_sha256"] = authority[
        "production_g2a_native_post_head_tensor_sha256"
    ]
    projected_training["renderer_base_snapshot_digest_before"] = authority[
        "production_g2a_renderer_base_snapshot_digest"
    ]
    projected_training["renderer_base_snapshot_digest_after"] = authority[
        "production_g2a_renderer_base_snapshot_digest"
    ]
    projected_lock = projected["source_lock"]
    projected_lock["train_action_repr_target_t0_canary_retry7_v1.py"] = projected_lock.pop(
        "train_action_repr_target_t0_canary_retry8_v1.py"
    )
    projected.pop("receipt_digest", None)
    projected["receipt_digest"] = retry7.object_sha256(projected)
    _ORIGINAL_VALIDATE_RECEIPT(projected)
    return value


# Patch only the revision-specific extension points used by retry7.main.
retry7.SCHEMA_VERSION = SCHEMA_VERSION
retry7.STEP_SCHEMA_VERSION = STEP_SCHEMA_VERSION
retry7.METHOD = METHOD
retry7.ATTEMPT_CLAIM_SCHEMA_VERSION = ATTEMPT_CLAIM_SCHEMA_VERSION
retry7.ATTEMPT_CLAIM_MARKER_NAME = ATTEMPT_CLAIM_MARKER_NAME
retry7.EXPECTED_CANONICAL_OUTPUT_PATH = EXPECTED_CANONICAL_OUTPUT_PATH
retry7.EXPECTED_ATTEMPT_CLAIM_PATH = EXPECTED_ATTEMPT_CLAIM_PATH
retry7.EXPECTED_SOURCE_PIN_PATHS = EXPECTED_SOURCE_PIN_PATHS
retry7.__file__ = str(Path(__file__).resolve())
retry7.prevalidate_authorization_replay_contract = prevalidate_authorization_replay_contract
retry7.validate_authorization_addendum = validate_authorization_addendum
retry7.run_one_step_optimizer_canary = run_one_step_optimizer_canary
retry7.validate_t0_receipt = validate_t0_receipt
retry7.validate_receipt = validate_t0_receipt


validate_published_t0_output = retry7.validate_published_t0_output
main = retry7.main


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_CLAIM_MARKER_NAME",
    "ATTEMPT_CLAIM_SCHEMA_VERSION",
    "EXPECTED_ATTEMPT_CLAIM_PATH",
    "EXPECTED_CANONICAL_OUTPUT_PATH",
    "EXPECTED_SOURCE_PIN_PATHS",
    "SCHEMA_VERSION",
    "STEP_SCHEMA_VERSION",
    "main",
    "run_one_step_optimizer_canary",
    "validate_published_t0_output",
    "validate_t0_receipt",
]
