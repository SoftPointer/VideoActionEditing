#!/usr/bin/env python3
"""Run a zero-training directed-source-attention probe on legacy full644.

This entry point is intentionally a thin wrapper around :mod:`infer_lora`.
The legacy harness remains authoritative for the exact-81-frame source-video
path, the full644 adapter receipt, strict 480-tensor PEFT reload, safe merge,
prompt construction, VAE geometry, v2v APG sampling, and four-rank Ulysses.

Only after ``infer_lora._strict_load_and_merge_adapter`` has completed do we
replace selected ``attn1`` processors with the fail-closed directed-source
processor.  The replacement is an untrained architecture oracle and must not
be reported as a production method or scientific result.  Its only external
conditions remain the source video and edit instruction accepted by the
legacy harness.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import directed_source_attention as directed  # noqa: E402
import infer_lora as legacy_inference  # noqa: E402


INFERENCE_RECEIPT_SCHEMA = (
    "bernini-r-1p3b-full644-directed-source-attention-oracle-receipt-v1"
)
FULL644_TRAINING_STEP = 644
FULL644_ADAPTER_TENSOR_COUNT = 480
FULL644_ADAPTER_SHA256 = (
    "9217ff653e47f915105fe8fa64856037d63811562cec1e9fd53ae9e4613a9774"
)
EXPECTED_INFERENCE_STEPS = 40
EXPECTED_EDITOR_FORWARDS_PER_STEP = 2
EXPECTED_PROCESSOR_CALLS = (
    EXPECTED_INFERENCE_STEPS * EXPECTED_EDITOR_FORWARDS_PER_STEP
)
PROBE_ARM_BY_SELECTION = {
    "mid": "B_mid_attn1",
    "late": "C_late_attn1",
    "all": "BC_all_attn1",
}


class DirectedOracleInferenceError(RuntimeError):
    """Raised before making a claim from an invalid oracle execution."""


def split_oracle_arguments(
    argv: Sequence[str],
) -> tuple[argparse.Namespace, list[str]]:
    """Remove the one oracle-only switch before the legacy parser runs."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--directed-attn-blocks",
        required=True,
        choices=directed.BLOCK_SELECTIONS,
    )
    oracle_args, legacy_args = parser.parse_known_args(list(argv))
    return oracle_args, legacy_args


def _validate_full644_base_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != legacy_inference.INFERENCE_RECEIPT_SCHEMA:
        raise DirectedOracleInferenceError(
            "refusing to augment an unknown legacy inference receipt"
        )
    adapter = receipt.get("adapter")
    if not isinstance(adapter, Mapping):
        raise DirectedOracleInferenceError("legacy receipt lacks adapter identity")
    if adapter.get("training_global_step") != FULL644_TRAINING_STEP:
        raise DirectedOracleInferenceError(
            "directed oracle requires the completed legacy full644 adapter"
        )
    if adapter.get("adapter_model_sha256") != FULL644_ADAPTER_SHA256:
        raise DirectedOracleInferenceError(
            "legacy full644 adapter weight digest differs"
        )
    if adapter.get("tensor_count") != FULL644_ADAPTER_TENSOR_COUNT:
        raise DirectedOracleInferenceError(
            "legacy full644 strict adapter tensor count differs"
        )
    if adapter.get("strictly_reloaded") is not True:
        raise DirectedOracleInferenceError("adapter was not strictly reloaded")
    if adapter.get("safe_merged_for_inference") is not True:
        raise DirectedOracleInferenceError("adapter was not safely merged")

    model_input = receipt.get("input")
    if not isinstance(model_input, Mapping):
        raise DirectedOracleInferenceError("legacy receipt lacks input contract")
    if model_input.get("accepted_model_conditions") != [
        "source_video",
        "edit_instruction",
    ]:
        raise DirectedOracleInferenceError(
            "oracle conditions differ from source video plus instruction"
        )
    false_fields = (
        "target_video_argument",
        "target_accessed_by_inference",
        "external_mask_or_swept_tube",
        "external_tracking_pose_or_trajectory",
        "reference_image_or_video",
        "external_shared_i0",
    )
    if any(model_input.get(name) is not False for name in false_fields):
        raise DirectedOracleInferenceError(
            "legacy input receipt contains a privileged external condition"
        )
    if receipt.get("production_claim_forbidden") is not True:
        raise DirectedOracleInferenceError(
            "legacy receipt lost its production-claim restriction"
        )
    if receipt.get("scientific_claim_authorized") is not False:
        raise DirectedOracleInferenceError(
            "legacy receipt contains an unsupported scientific claim"
        )


def validate_full644_adapter_bundle(
    adapter: legacy_inference.AdapterBundle,
) -> dict[str, str]:
    """Pin the loader itself to the completed, audited full644 artifact."""

    actual_sha256 = legacy_inference.file_sha256(adapter.adapter_model_path)
    if actual_sha256 != FULL644_ADAPTER_SHA256:
        raise DirectedOracleInferenceError(
            "legacy full644 adapter weight digest differs before strict load"
        )
    receipt = legacy_inference._read_json(
        adapter.training_receipt_path,
        label="legacy full644 training receipt",
    )
    if receipt.get("global_step") != FULL644_TRAINING_STEP:
        raise DirectedOracleInferenceError(
            "adapter training receipt is not the completed full644 step"
        )
    training_revision = receipt.get("method_source_revision")
    training_archive = receipt.get("method_source_archive_sha256")
    if (
        not isinstance(training_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", training_revision) is None
        or not isinstance(training_archive, str)
        or re.fullmatch(r"[0-9a-f]{64}", training_archive) is None
    ):
        raise DirectedOracleInferenceError(
            "full644 training receipt lacks immutable method source identity"
        )
    return {
        "training_method_source_revision": training_revision,
        "training_method_source_archive_sha256": training_archive,
    }


def validate_runtime_attention_certificate(
    handle: directed.DirectedAttentionPatchHandle,
) -> dict[str, Any]:
    """Prove every selected processor executed on the full Ulysses layout."""

    if handle.restored is not True:
        raise DirectedOracleInferenceError(
            "directed attention must be restored before receipt finalization"
        )
    try:
        value = handle.receipt()
    except (AttributeError, directed.DirectedAttentionContractError) as error:
        raise DirectedOracleInferenceError(
            "cannot serialize directed-attention runtime evidence"
        ) from error
    if not isinstance(value, Mapping):
        raise DirectedOracleInferenceError(
            "directed-attention runtime evidence must be a mapping"
        )
    expected_indices = list(
        directed.resolve_block_indices(
            directed.EXPECTED_BLOCK_COUNT, handle.selection
        )
    )
    runtime = value.get("runtime")
    if (
        value.get("block_indices") != expected_indices
        or not isinstance(runtime, Mapping)
        or runtime.get("installed_block_count") != len(expected_indices)
        or runtime.get("restored") is not True
    ):
        raise DirectedOracleInferenceError(
            "directed-attention installed scope or restore evidence differs"
        )
    per_block = runtime.get("per_block")
    if not isinstance(per_block, list) or len(per_block) != len(
        expected_indices
    ):
        raise DirectedOracleInferenceError(
            "directed-attention per-block evidence is incomplete"
        )
    common_geometry: Optional[tuple[int, int, int]] = None
    for expected_index, item in zip(expected_indices, per_block):
        if not isinstance(item, Mapping):
            raise DirectedOracleInferenceError(
                "directed-attention block evidence is not a mapping"
            )
        full = item.get("full_sequence_length")
        source = item.get("source_sequence_length")
        target = item.get("target_sequence_length")
        if (
            item.get("block_index") != expected_index
            or item.get("call_count") != EXPECTED_PROCESSOR_CALLS
            or type(full) is not int
            or type(source) is not int
            or type(target) is not int
            or source <= 0
            or target != source
            or full != source + target
            or item.get("ulysses_observed") is not True
        ):
            raise DirectedOracleInferenceError(
                f"directed-attention block {expected_index} lacks complete "
                "80-call equal-pair Ulysses evidence"
            )
        geometry = (full, source, target)
        if common_geometry not in (None, geometry):
            raise DirectedOracleInferenceError(
                "directed-attention sequence geometry changed across blocks"
            )
        common_geometry = geometry
    certificate = {
        "validated": True,
        "expected_inference_steps": EXPECTED_INFERENCE_STEPS,
        "expected_editor_forwards_per_step": (
            EXPECTED_EDITOR_FORWARDS_PER_STEP
        ),
        "expected_calls_per_selected_block": EXPECTED_PROCESSOR_CALLS,
        "selected_block_count": len(expected_indices),
        "selected_block_indices": expected_indices,
        "common_full_sequence_length": common_geometry[0],
        "common_source_sequence_length": common_geometry[1],
        "common_target_sequence_length": common_geometry[2],
        "ulysses_observed_on_every_selected_block": True,
    }
    return {"attention": dict(value), "certificate": certificate}


def augment_inference_receipt(
    receipt: Mapping[str, Any],
    *,
    handle: directed.DirectedAttentionPatchHandle,
    training_source_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Bind the untrained oracle scope and runtime statistics to the receipt."""

    _validate_full644_base_receipt(receipt)
    if set(training_source_identity) != {
        "training_method_source_revision",
        "training_method_source_archive_sha256",
    }:
        raise DirectedOracleInferenceError(
            "full644 training source identity is incomplete"
        )
    runtime_evidence = validate_runtime_attention_certificate(handle)
    value = copy.deepcopy(dict(receipt))
    value.pop("receipt_digest", None)
    value["schema_version"] = INFERENCE_RECEIPT_SCHEMA
    value["method"] = "legacy_full644_directed_source_attention_oracle"
    value["oracle"] = {
        "classification": "untrained_inference_only_architecture_oracle",
        "probe_arm": PROBE_ARM_BY_SELECTION[handle.selection],
        "zero_training": True,
        "trained_parameters": 0,
        "full644_adapter_frozen": True,
        **dict(training_source_identity),
        "strict_full644_load_then_safe_merge_before_install": True,
        "source_and_instruction_only": True,
        "production_claim_forbidden": True,
        "scientific_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "directed_source_attention": runtime_evidence["attention"],
        "runtime_execution_certificate": runtime_evidence["certificate"],
    }
    value["experimental_inference"] = True
    value["untrained_oracle"] = True
    value["source_and_instruction_only"] = True
    value["production_claim_forbidden"] = True
    value["scientific_claim_forbidden"] = True
    value["scientific_claim_authorized"] = False
    value["receipt_digest"] = legacy_inference.object_sha256(value)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    oracle_args, legacy_args = split_oracle_arguments(raw)
    state: dict[str, Any] = {}
    original_loader = legacy_inference._strict_load_and_merge_adapter
    original_writer = legacy_inference._atomic_write_json

    def strict_load_merge_then_patch(
        base_model: Any,
        adapter: legacy_inference.AdapterBundle,
        expected_targets: Sequence[str],
    ) -> tuple[Any, int]:
        if state:
            raise DirectedOracleInferenceError(
                "legacy strict adapter loader ran more than once"
            )
        training_source_identity = validate_full644_adapter_bundle(adapter)
        # This call is the contract boundary: do not install a processor until
        # infer_lora has validated all 480 tensors and safely merged the LoRA.
        merged_model, tensor_count = original_loader(
            base_model,
            adapter,
            expected_targets,
        )
        if tensor_count != FULL644_ADAPTER_TENSOR_COUNT:
            raise DirectedOracleInferenceError(
                "strict full644 loader returned an unexpected tensor count"
            )
        try:
            handle = directed.install_directed_source_attention(
                merged_model,
                selection=oracle_args.directed_attn_blocks,
            )
        except directed.DirectedAttentionContractError as error:
            raise DirectedOracleInferenceError(str(error)) from error
        state["handle"] = handle
        state["training_source_identity"] = training_source_identity
        return merged_model, tensor_count

    def restore_and_write(path: Any, receipt: Mapping[str, Any]) -> None:
        handle = state.get("handle")
        if not isinstance(handle, directed.DirectedAttentionPatchHandle):
            raise DirectedOracleInferenceError(
                "directed attention was not installed before receipt write"
            )
        try:
            handle.restore()
        except directed.DirectedAttentionContractError as error:
            raise DirectedOracleInferenceError(str(error)) from error
        training_source_identity = state.get("training_source_identity")
        if not isinstance(training_source_identity, Mapping):
            raise DirectedOracleInferenceError(
                "full644 training source identity was not retained"
            )
        augmented = augment_inference_receipt(
            receipt,
            handle=handle,
            training_source_identity=training_source_identity,
        )
        # infer_lora prints this same object immediately after the atomic write;
        # update it so persisted JSON and stdout carry one digest.
        if isinstance(receipt, dict):
            receipt.clear()
            receipt.update(augmented)
            original_writer(path, receipt)
        else:
            original_writer(path, augmented)

    legacy_inference._strict_load_and_merge_adapter = strict_load_merge_then_patch
    legacy_inference._atomic_write_json = restore_and_write
    try:
        return legacy_inference.main(legacy_args)
    except directed.DirectedAttentionContractError as error:
        raise DirectedOracleInferenceError(str(error)) from error
    finally:
        legacy_inference._strict_load_and_merge_adapter = original_loader
        legacy_inference._atomic_write_json = original_writer
        handle = state.get("handle")
        if isinstance(handle, directed.DirectedAttentionPatchHandle):
            try:
                handle.restore()
            except directed.DirectedAttentionContractError as error:
                raise DirectedOracleInferenceError(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
