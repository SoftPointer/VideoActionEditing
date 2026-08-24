#!/usr/bin/env python3
"""Strict source-only decode for the isolated v16r4 S1 checkpoint.

The sampler, exact source materialization, retained-FD model consumption,
strict 480-tensor PEFT replay, safe merge, and output publication are delegated
unchanged to :mod:`infer_lora`.  This wrapper replaces only the checkpoint
identity boundary and records that one route-off case is a diagnostic, not the
full Heldout8 dual-arm promotion sidecar required by the trainer receipt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as delegated  # noqa: E402
import v16r4_s1_checkpoint_contract as contract  # noqa: E402


INFERENCE_RECEIPT_SCHEMA = (
    "bernini-online-anchor-v16r4-s1-checkpoint-source-only-inference-receipt-v1"
)

V16R4S1CheckpointInferenceError = contract.V16R4S1CheckpointContractError

_ACTIVE_IDENTITY: dict[str, Any] | None = None
_DELEGATED_BUILD_PARSER = delegated.build_parser
_DELEGATED_VALIDATE_CLI = delegated.validate_cli
_DELEGATED_ACTIVATE_MODEL_CONSUMPTION_AUTHORITY = (
    delegated.activate_model_consumption_authority
)
_DELEGATED_BUILD_INFERENCE_RECEIPT = delegated.build_inference_receipt
_DELEGATED_STRICT_LOAD_AND_MERGE = delegated._strict_load_and_merge_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = _DELEGATED_BUILD_PARSER()
    parser.allow_abbrev = False
    identity = parser.add_argument_group("required v16r4 S1 checkpoint identity")
    identity.add_argument(
        "--expected-training-global-step",
        type=int,
        required=True,
        choices=contract.SAVE_STEPS,
    )
    identity.add_argument("--expected-adapter-config-sha256", required=True)
    identity.add_argument("--expected-adapter-model-sha256", required=True)
    identity.add_argument("--expected-training-receipt-sha256", required=True)
    identity.add_argument(
        "--expected-training-method-source-revision", required=True
    )
    identity.add_argument(
        "--expected-training-method-source-archive-sha256", required=True
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    _DELEGATED_VALIDATE_CLI(args)
    contract.validate_runtime_versions()
    if bool(getattr(args, "base_only", False)):
        raise V16R4S1CheckpointInferenceError(
            "the v16r4 S1 wrapper requires --adapter-checkpoint"
        )
    if getattr(args, "adapter_checkpoint_manifest", None) is not None or getattr(
        args, "adapter_checkpoint_manifest_sha256", None
    ) is not None:
        raise V16R4S1CheckpointInferenceError(
            "the v16r4 S1 three-file checkpoint has no legacy checkpoint manifest"
        )
    contract.require_save_step(getattr(args, "expected_training_global_step", None))
    for name, label in (
        ("expected_adapter_config_sha256", "expected adapter config SHA-256"),
        ("expected_adapter_model_sha256", "expected adapter model SHA-256"),
        ("expected_training_receipt_sha256", "expected training receipt SHA-256"),
        (
            "expected_training_method_source_revision",
            "expected training method source revision",
        ),
        (
            "expected_training_method_source_archive_sha256",
            "expected training method source archive SHA-256",
        ),
    ):
        contract.require_sha256(getattr(args, name, None), label=label)


def _identity_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return contract.authenticate_checkpoint(
        args.adapter_checkpoint,
        expected_global_step=args.expected_training_global_step,
        expected_adapter_config_sha256=args.expected_adapter_config_sha256,
        expected_adapter_model_sha256=args.expected_adapter_model_sha256,
        expected_training_receipt_sha256=args.expected_training_receipt_sha256,
        expected_training_method_source_revision=(
            args.expected_training_method_source_revision
        ),
        expected_training_method_source_archive_sha256=(
            args.expected_training_method_source_archive_sha256
        ),
        expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        # Bind both JSON inputs now.  Authenticate the 755-MiB model immediately
        # before the strict PEFT load, avoiding a redundant full-file hash pass.
        include_model=False,
    )


def activate_model_consumption_authority(
    args: argparse.Namespace,
) -> Mapping[str, Any] | None:
    global _ACTIVE_IDENTITY

    evidence = _DELEGATED_ACTIVATE_MODEL_CONSUMPTION_AUTHORITY(args)
    _ACTIVE_IDENTITY = _identity_from_args(args)
    return evidence


def _active_identity() -> dict[str, Any]:
    if _ACTIVE_IDENTITY is None:
        raise V16R4S1CheckpointInferenceError(
            "v16r4 expected S1 checkpoint identity was not activated"
        )
    return _ACTIVE_IDENTITY


def validate_adapter_contract(
    adapter_config: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = delegated.trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    active = _active_identity()
    bundle = active["bundle"]
    contract.verify_checkpoint_hashes(
        bundle,
        expected_adapter_config_sha256=active["adapter_config_sha256"],
        expected_adapter_model_sha256=active["adapter_model_sha256"],
        expected_training_receipt_sha256=active["training_receipt_sha256"],
        include_model=False,
    )
    return contract.validate_v16r4_s1_checkpoint_contract(
        adapter_config,
        training_receipt,
        expected_global_step=active["global_step"],
        expected_adapter_config_sha256=active["adapter_config_sha256"],
        expected_adapter_model_sha256=active["adapter_model_sha256"],
        expected_training_receipt_sha256=active["training_receipt_sha256"],
        expected_training_method_source_revision=active["method_source_revision"],
        expected_training_method_source_archive_sha256=active[
            "method_source_archive_sha256"
        ],
        expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
    )


def _strict_load_and_merge_adapter(
    base_model: Any,
    adapter: delegated.AdapterBundle,
    expected_targets: Sequence[str],
    *,
    route_scope: Optional[str] = None,
    require_zero_effect: bool = False,
) -> tuple[Any, int]:
    active = _active_identity()
    bundle = active["bundle"]
    contract.assert_same_bundle(adapter, bundle)
    contract.verify_checkpoint_hashes(
        bundle,
        expected_adapter_config_sha256=active["adapter_config_sha256"],
        expected_adapter_model_sha256=active["adapter_model_sha256"],
        expected_training_receipt_sha256=active["training_receipt_sha256"],
        include_model=True,
    )
    inventory = contract.validate_adapter_safetensors_inventory(
        bundle.adapter_model_path
    )
    if inventory["tensor_count"] != contract.ADAPTER_TENSOR_COUNT:
        raise V16R4S1CheckpointInferenceError(
            "v16r4 S1 adapter tensor inventory changed before PEFT loading"
        )
    result = _DELEGATED_STRICT_LOAD_AND_MERGE(
        base_model,
        adapter,
        expected_targets,
        route_scope=route_scope,
        require_zero_effect=require_zero_effect,
    )
    contract.verify_checkpoint_hashes(
        bundle,
        expected_adapter_config_sha256=active["adapter_config_sha256"],
        expected_adapter_model_sha256=active["adapter_model_sha256"],
        expected_training_receipt_sha256=active["training_receipt_sha256"],
        include_model=False,
    )
    return result


def _annotate_inference_receipt(
    receipt: Mapping[str, Any],
    *,
    adapter_identity: Mapping[str, Any],
    wrapper_source_sha256: str,
) -> dict[str, Any]:
    annotated = dict(receipt)
    adapter = annotated.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("enabled") is not True:
        raise V16R4S1CheckpointInferenceError(
            "v16r4 inference receipt does not contain an enabled adapter"
        )
    if adapter.get("adapter_model_sha256") != adapter_identity.get(
        "adapter_model_sha256"
    ):
        raise V16R4S1CheckpointInferenceError(
            "v16r4 inference receipt adapter model identity differs"
        )
    if (
        adapter.get("tensor_count") != contract.ADAPTER_TENSOR_COUNT
        or adapter.get("strictly_reloaded") is not True
        or adapter.get("safe_merged_for_inference") is not True
    ):
        raise V16R4S1CheckpointInferenceError(
            "v16r4 inference lost strict 480-tensor replay/safe merge"
        )
    legacy_digest = adapter.pop("training_receipt_digest", None)
    if legacy_digest != adapter_identity.get("training_receipt_sha256"):
        raise V16R4S1CheckpointInferenceError(
            "delegated receipt lost its external v16r4 receipt binding"
        )
    training_identity = {
        "schema_version": adapter_identity["training_schema_version"],
        "method": adapter_identity["training_method"],
        "global_step": contract.GLOBAL_STEP,
        "max_steps": contract.MAX_STEPS,
        "checkpoint_complete": True,
        "terminal_full644_checkpoint": False,
        "exact644_training_complete": False,
        "lora_rank": adapter_identity["lora_rank"],
        "lora_alpha": adapter_identity["lora_alpha"],
        "lora_scope": adapter_identity["lora_scope"],
        "target_module_count": adapter_identity["target_module_count"],
        "adapter_tensor_count": contract.ADAPTER_TENSOR_COUNT,
        "target_modules_sha256": adapter_identity["target_modules_sha256"],
        "adapter_config_sha256": adapter_identity["adapter_config_sha256"],
        "adapter_model_sha256": adapter_identity["adapter_model_sha256"],
        "training_receipt_sha256": adapter_identity["training_receipt_sha256"],
        "training_method_source_revision": adapter_identity[
            "method_source_revision"
        ],
        "training_method_source_archive_sha256": adapter_identity[
            "method_source_archive_sha256"
        ],
        "decoded_canary_manifest_sha256": contract.HELDOUT8_MANIFEST_SHA256,
        "claim_scope": adapter_identity["claim_scope"],
        "scientific_claim_authorized": False,
    }
    adapter.update(
        {
            "training_receipt_sha256": adapter_identity[
                "training_receipt_sha256"
            ],
            "adapter_config_sha256": adapter_identity["adapter_config_sha256"],
            "training_identity": training_identity,
            "v16r4": {
                "decode_mode": "adapter_only_direct_rv2v",
                "online_anchor_route_applied": False,
                "global_step": contract.GLOBAL_STEP,
                "max_steps": contract.MAX_STEPS,
                "terminal_full644_checkpoint": False,
                "exact644_training_complete": False,
                "training_receipt_schema": contract.TRAINING_RECEIPT_SCHEMA,
                "adapter_model_sha256": adapter_identity[
                    "adapter_model_sha256"
                ],
                "adapter_config_sha256": adapter_identity[
                    "adapter_config_sha256"
                ],
                "training_receipt_sha256": adapter_identity[
                    "training_receipt_sha256"
                ],
                "full_heldout8_dual_arm_promotion_complete": False,
            },
        }
    )
    annotated["schema_version"] = INFERENCE_RECEIPT_SCHEMA
    annotated["delegated_inference"] = {
        "schema_version": delegated.INFERENCE_RECEIPT_SCHEMA,
        "infer_lora_source_sha256": annotated.get("infer_lora_source_sha256"),
        "sampling_and_strict_lora_loader_reused": True,
    }
    annotated["v16r4_s1_checkpoint_inference_wrapper"] = {
        "source_sha256": wrapper_source_sha256,
        "training_receipt_schema": contract.TRAINING_RECEIPT_SCHEMA,
        "expected_artifact_sha256_required_on_cli": True,
        "training_receipt_identity_kind": "external_file_sha256",
        "registered_save_steps": list(contract.SAVE_STEPS),
        "diagnostic_arm": "adapter_only_route_off",
        "full_heldout8_dual_arm_promotion_complete": False,
        "per_sample_manual_review_required": False,
    }
    annotated["scientific_claim_authorized"] = False
    annotated.pop("receipt_digest", None)
    annotated["receipt_digest"] = delegated.object_sha256(annotated)
    return annotated


def build_inference_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    adapter_identity = kwargs.get("adapter_identity")
    if not isinstance(adapter_identity, Mapping) or adapter_identity.get(
        "v16r4_online_anchor"
    ) is not True:
        raise V16R4S1CheckpointInferenceError(
            "v16r4 inference receipt lacks its validated S1 identity"
        )
    receipt = _DELEGATED_BUILD_INFERENCE_RECEIPT(*args, **kwargs)
    wrapper_sha = contract.shared.file_sha256(
        Path(__file__).resolve(), label="v16r4 S1 inference wrapper source"
    )
    return _annotate_inference_receipt(
        receipt,
        adapter_identity=adapter_identity,
        wrapper_source_sha256=wrapper_sha,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _ACTIVE_IDENTITY

    originals = (
        delegated.build_parser,
        delegated.validate_cli,
        delegated.activate_model_consumption_authority,
        delegated.validate_adapter_contract,
        delegated._strict_load_and_merge_adapter,
        delegated.build_inference_receipt,
    )
    _ACTIVE_IDENTITY = None
    delegated.build_parser = build_parser
    delegated.validate_cli = validate_cli
    delegated.activate_model_consumption_authority = activate_model_consumption_authority
    delegated.validate_adapter_contract = validate_adapter_contract
    delegated._strict_load_and_merge_adapter = _strict_load_and_merge_adapter
    delegated.build_inference_receipt = build_inference_receipt
    try:
        return delegated.main(argv)
    finally:
        (
            delegated.build_parser,
            delegated.validate_cli,
            delegated.activate_model_consumption_authority,
            delegated.validate_adapter_contract,
            delegated._strict_load_and_merge_adapter,
            delegated.build_inference_receipt,
        ) = originals
        _ACTIVE_IDENTITY = None


if __name__ == "__main__":
    raise SystemExit(main())
