#!/usr/bin/env python3
"""Strict source-only inference for SEER same-state FM+motion+copy LoRA.

The sampler and exact 60-module/120-factor PEFT loader are shared with
``infer_seer_scoped_lora.py``.  This entry point differs only in the training
receipt it admits: the full same-noisy-state action/no-op trainer rather than
the four-step B0 standard-FM trainer.  Keeping separate entry points prevents
either receipt contract from being widened implicitly.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_seer_scoped_lora as scoped  # noqa: E402
import train_seer_event_erasure_fm as trainer  # noqa: E402


base = scoped.base
TRAINING_RECEIPT_SCHEMA = trainer.RECEIPT_SCHEMA
TRAINING_METHOD = trainer.METHOD_NAME
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SeerSameStateInferenceError(scoped.SeerInferenceError):
    """Raised before generation when the full SEER receipt differs."""


def _positive(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise SeerSameStateInferenceError(f"{label} must be finite and positive")
    return float(value)


def _validate_adapter_contract_at_step(
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    required_global_step: int,
    step_error: str,
    expected_checkpoint_tree_sha256: str = base.trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Validate a genuine same-state update at one caller-pinned step."""

    scoped._validate_serialized_target_coverage(adapter_config.get("target_modules"))
    if (
        adapter_config.get("peft_type") != "LORA"
        or adapter_config.get("r") != 8
        or float(adapter_config.get("lora_alpha", -1)) != 8.0
        or float(adapter_config.get("lora_dropout", -1)) != 0.0
        or adapter_config.get("bias") != "none"
        or adapter_config.get("modules_to_save") not in (None, [])
        or adapter_config.get("use_dora") not in (None, False)
        or adapter_config.get("use_rslora") not in (None, False)
    ):
        raise SeerSameStateInferenceError("SEER same-state PEFT config differs")
    candidate = dict(receipt)
    declared_digest = candidate.pop("receipt_digest", None)
    if (
        not isinstance(declared_digest, str)
        or _SHA256.fullmatch(declared_digest) is None
        or base.object_sha256(candidate) != declared_digest
    ):
        raise SeerSameStateInferenceError("SEER same-state receipt digest differs")
    if (
        receipt.get("schema_version") != TRAINING_RECEIPT_SCHEMA
        or receipt.get("method") != TRAINING_METHOD
        or receipt.get("bernini_commit") != base.trainer.BERNINI_OFFICIAL_COMMIT
        or receipt.get("veomni_commit") != base.trainer.VEOMNI_TESTED_COMMIT
    ):
        raise SeerSameStateInferenceError("SEER same-state method/source identity differs")
    step = receipt.get("global_step")
    if type(step) is not int or step != required_global_step:
        raise SeerSameStateInferenceError(step_error)
    checkpoint = receipt.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("tree_sha256") != expected_checkpoint_tree_sha256
    ):
        raise SeerSameStateInferenceError("SEER same-state checkpoint tree differs")

    expected_targets = scoped.expected_lora_target_modules()
    adapter = receipt.get("adapter")
    immutable = receipt.get("immutable_contract")
    value = immutable.get("value") if isinstance(immutable, Mapping) else None
    supervision = receipt.get("supervision")
    if (
        not isinstance(adapter, Mapping)
        or adapter.get("rank") != 8
        or float(adapter.get("alpha", -1)) != 8.0
        or adapter.get("scope") != "cross_q_out"
        or adapter.get("target_module_count") != 60
        or adapter.get("target_modules") != expected_targets
        or adapter.get("target_modules_sha256")
        != base.object_sha256(expected_targets)
    ):
        raise SeerSameStateInferenceError("SEER same-state exact adapter scope differs")
    manifest_sha256 = scoped._sha(
        value.get("expected_seer_manifest_sha256") if isinstance(value, Mapping) else None,
        label="same-state SEER manifest digest",
    )
    owner_sha256 = scoped._sha(
        value.get("expected_seer_owner_spec_sha256") if isinstance(value, Mapping) else None,
        label="same-state SEER owner digest",
    )
    method_archive_sha256 = scoped._sha(
        value.get("method_source_archive_sha256") if isinstance(value, Mapping) else None,
        label="same-state method archive digest",
    )
    method_revision = scoped._sha(
        value.get("method_source_revision") if isinstance(value, Mapping) else None,
        label="same-state method revision",
        sha1=True,
    )
    if (
        not isinstance(immutable, Mapping)
        or not isinstance(value, Mapping)
        or immutable.get("digest") != base.object_sha256(value)
        or value.get("method") != TRAINING_METHOD
        or value.get("branch_state_mode") != "shared_noisy_clean_field"
        or value.get("exact_same_noisy_query") is not True
        or value.get("lora_scope") != "cross_q_out"
        or value.get("target_modules") != expected_targets
        or value.get("full_pair_flow_matching_weight") != 1.0
        or value.get("same_state_causal_motion_weight") != 0.5
        or value.get("same_state_noop_copy_weight") != 0.5
        or value.get("same_generated_video_coordinate") is not True
        or value.get("event_erasure_source_excludes_transition_and_terminal") is not True
        or value.get("rejected_cmsg_cross_identity_gate_reused") is not False
        or value.get("seer_authority") != trainer.AUTHORITY
        or value.get("training_completion_is_method_success") is not False
        or value.get("heldout_decoded_review_required") is not True
        or immutable.get("expected_seer_manifest_sha256")
        != manifest_sha256
        or immutable.get("expected_seer_owner_spec_sha256") != owner_sha256
        or immutable.get("method_source_archive_sha256")
        != method_archive_sha256
    ):
        raise SeerSameStateInferenceError("SEER same-state immutable contract differs")
    if (
        not isinstance(supervision, Mapping)
        or supervision.get("exact_same_noisy_query") is not True
        or supervision.get("self_generated_target_supervision") is not True
        or supervision.get("event_erased_source_supervision") is not True
        or supervision.get("full_pair_flow_matching_enabled") is not True
        or supervision.get("full_pair_flow_matching_weight") != 1.0
        or supervision.get("same_state_causal_motion_weight") != 0.5
        or supervision.get("same_state_noop_copy_weight") != 0.5
        or supervision.get("training_completion_is_method_success") is not False
        or supervision.get("heldout_decoded_review_required") is not True
    ):
        raise SeerSameStateInferenceError("SEER same-state supervision differs")
    distributed = receipt.get("distributed")
    if (
        not isinstance(distributed, Mapping)
        or distributed.get("world_size") != 4
        or distributed.get("ulysses_size") != 4
        or distributed.get("same_pair_all_ranks") is not True
        or distributed.get("explicit_lora_gradient_all_reduce") is not True
    ):
        raise SeerSameStateInferenceError("SEER same-state distributed contract differs")

    update = receipt.get("parameter_update_evidence")
    seer = receipt.get("seer")
    metrics = receipt.get("last_metrics")
    if (
        not isinstance(update, Mapping)
        or update.get("exact_parameter_bytes_changed") is not True
        or update.get("engineering_execution_success") is not True
        or update.get("method_success_claimed") is not False
        or not isinstance(seer, Mapping)
        or seer.get("owner_spec_sha256") != owner_sha256
        or seer.get("dataset_manifest_sha256") != manifest_sha256
        or seer.get("row_count") != value.get("seer_row_count")
        or seer.get("self_generated_target_supervision") is not True
        or seer.get("event_erased_source_supervision") is not True
        or seer.get("training_completion_is_method_success") is not False
        or seer.get("heldout_decoded_review_required") is not True
        or not isinstance(metrics, Mapping)
    ):
        raise SeerSameStateInferenceError("SEER same-state update evidence differs")
    initial = scoped._sha(
        adapter.get("initialization_digest"), label="same-state initialization digest"
    )
    final = scoped._sha(
        adapter.get("checkpoint_parameter_digest"), label="same-state final digest"
    )
    if initial == final:
        raise SeerSameStateInferenceError("SEER same-state parameters equal initialization")
    gradient_norm = _positive(
        metrics.get("preclip_gradient_norm"), label="same-state gradient norm"
    )
    if (
        update.get("initial_trainable_parameter_digest") != initial
        or update.get("final_trainable_parameter_digest") != final
        or update.get("final_preclip_gradient_norm") != gradient_norm
    ):
        raise SeerSameStateInferenceError(
            "SEER same-state parameter-update evidence does not cross-bind"
        )
    if (
        receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
    ):
        raise SeerSameStateInferenceError("SEER same-state receipt claims method success")
    transformers_version = receipt.get("transformers_version")
    if not isinstance(transformers_version, str) or not transformers_version:
        raise SeerSameStateInferenceError("SEER same-state Transformers version is absent")
    return {
        "global_step": step,
        "receipt_digest": declared_digest,
        "target_modules_sha256": base.object_sha256(expected_targets),
        "transformers_version": transformers_version,
        "method_source_revision": method_revision,
        "method_source_archive_sha256": method_archive_sha256,
        "seer_manifest_sha256": manifest_sha256,
        "seer_owner_spec_sha256": owner_sha256,
    }


def validate_adapter_contract(
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = base.trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Validate only the historical four-step same-state checkpoint.

    A separate entry point owns the full-160 contract.  Keeping the step pin
    here explicit prevents that production-sized receipt from widening this
    four-step engineering helper (or vice versa).
    """

    return _validate_adapter_contract_at_step(
        adapter_config,
        receipt,
        required_global_step=scoped.REQUIRED_GLOBAL_STEP,
        step_error=(
            "SEER same-state held-out decode requires the four-step checkpoint"
        ),
        expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
    )


def _install_specialization() -> None:
    base.expected_lora_target_modules = scoped.expected_lora_target_modules
    base.expected_adapter_state_keys = scoped.expected_adapter_state_keys
    base.EXPECTED_ADAPTER_TENSOR_COUNT = scoped.ADAPTER_TENSOR_COUNT
    base.validate_adapter_contract = validate_adapter_contract
    base.validate_adapter_state_dicts = scoped.validate_adapter_state_dicts
    base._strict_load_and_merge_adapter = scoped._strict_load_and_merge_adapter


def main(argv: Optional[Sequence[str]] = None) -> int:
    _install_specialization()
    preliminary = base.build_parser().parse_args(argv)
    if preliminary.adapter_checkpoint:
        bundle = base.resolve_adapter_bundle(preliminary.adapter_checkpoint)
        scoped.validate_scoped_safetensors(bundle.adapter_model_path)
    return base.main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SeerSameStateInferenceError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
