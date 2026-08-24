#!/usr/bin/env python3
"""Source-only inference for the exact v16r3 online-anchor S644 adapter.

This module deliberately delegates preprocessing, Bernini construction,
sampling, strict PEFT state replay, safe merge, decoding, and publication to
``infer_lora.py``.  The v16r3 trainer emits a different, externally hash-bound
training receipt, so this wrapper replaces only the adapter identity boundary
and annotates the resulting inference receipt with that identity.

The adapter, PEFT config, and training receipt SHA-256 values are mandatory
CLI inputs.  This is intentional: the v16r3 receipt binds the first two files
but has no internal receipt digest with which it could bind its own bytes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as delegated  # noqa: E402


TRAINING_RECEIPT_SCHEMA = (
    "bernini-online-anchor-full644-dynamic-static-routed-teacher-receipt-v16r3"
)
TRAINING_METHOD = (
    "bernini-online-anchor-full644-dynamic-static-routed-teacher-v16r3"
)
INFERENCE_RECEIPT_SCHEMA = (
    "bernini-online-anchor-v16r3-source-only-inference-receipt-v1"
)
TRAINING_CLAIM_SCOPE = (
    "engineering_training_run_only_non_scientific_until_held_out_evaluation"
)
LORA_RANK = 256
LORA_ALPHA = 256
TARGET_MODULE_COUNT = 240
ADAPTER_TENSOR_COUNT = 480
TRAINABLE_PARAMETER_COUNT = 188_743_680
GLOBAL_STEP = 644
PEFT_VERSION = "0.19.1"
TRANSFORMERS_VERSION = "5.5.4"
LORA_SCOPE = "all_30_blocks_attn1_attn2_qkvo"
ZERO_RMS_POLICY = "exact_forward_zero_rms_zero_subgradient_v1"
ZERO_RMS_SCOPE = ["current_temporal_rms", "route_rms"]
TARGET_MODULES_SHA256 = (
    "d253ba3f11ec5ac26710a829d543a18b939c6f111c64be785264fcd852f3f35a"
)
S279_EXPECTED_CALLS = [
    {"role": "action_micro_0", "seed": 1656484053, "timestep": 1000.0},
    {"role": "raw_replay_micro_0", "seed": 1657484056, "timestep": 580.0},
    {"role": "action_micro_1", "seed": 718898016, "timestep": 764.0},
    {"role": "raw_replay_micro_1", "seed": 719898019, "timestep": 880.0},
]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class V16R3InferenceContractError(delegated.InferenceContractError):
    """Raised before generation when the v16r3 identity differs."""


@dataclass(frozen=True)
class ExpectedArtifactIdentity:
    checkpoint_root: Path
    adapter_dir: Path
    adapter_config_path: Path
    adapter_model_path: Path
    training_receipt_path: Path
    adapter_config_sha256: str
    adapter_model_sha256: str
    training_receipt_sha256: str


_ACTIVE_EXPECTED_IDENTITY: ExpectedArtifactIdentity | None = None

_DELEGATED_BUILD_PARSER = delegated.build_parser
_DELEGATED_VALIDATE_CLI = delegated.validate_cli
_DELEGATED_ACTIVATE_MODEL_CONSUMPTION_AUTHORITY = (
    delegated.activate_model_consumption_authority
)
_DELEGATED_BUILD_INFERENCE_RECEIPT = delegated.build_inference_receipt
_DELEGATED_STRICT_LOAD_AND_MERGE = delegated._strict_load_and_merge_adapter


def expected_target_modules() -> list[str]:
    targets = delegated.expected_lora_target_modules()
    if len(targets) != TARGET_MODULE_COUNT:
        raise V16R3InferenceContractError(
            "v16r3 delegated target registry no longer has 240 modules"
        )
    if delegated.object_sha256(targets) != TARGET_MODULES_SHA256:
        raise V16R3InferenceContractError(
            "v16r3 delegated target registry digest differs"
        )
    return targets


def _strict_equal(observed: Any, expected: Any) -> bool:
    return delegated.canonical_json_bytes(observed) == delegated.canonical_json_bytes(
        expected
    )


def _require_field(
    mapping: Mapping[str, Any], key: str, expected: Any, *, label: str
) -> None:
    if key not in mapping or not _strict_equal(mapping[key], expected):
        raise V16R3InferenceContractError(
            f"{label} differs for {key}: {mapping.get(key)!r}"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise V16R3InferenceContractError(
            f"{label} must be an explicit lowercase SHA-256"
        )
    return value


def _expected_peft_config_without_targets() -> dict[str, Any]:
    return {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": {
            "base_model_class": "BerniniRendererModel",
            "parent_library": "bernini.models.renderer",
        },
        "base_model_name_or_path": "",
        "bias": "none",
        "corda_config": None,
        "ensure_weight_tying": False,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": LORA_ALPHA,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "lora_ga_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": PEFT_VERSION,
        "qalora_group_size": 16,
        "r": LORA_RANK,
        "rank_pattern": {},
        "revision": None,
        "target_parameters": None,
        "task_type": None,
        "trainable_token_indices": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }


def _validate_peft_config(adapter_config: Mapping[str, Any]) -> None:
    if set(adapter_config) != delegated.FULL644_PEFT_CONFIG_FIELDS:
        raise V16R3InferenceContractError(
            "v16r3 adapter PEFT 0.19.1 field closure differs"
        )
    targets = adapter_config.get("target_modules")
    if not isinstance(targets, list) or not all(
        isinstance(item, str) for item in targets
    ):
        raise V16R3InferenceContractError(
            "v16r3 adapter target_modules must be an explicit string list"
        )
    serialized = set(targets)
    if len(serialized) != len(targets) or serialized not in (
        set(expected_target_modules()),
        set(delegated.PEFT_COMPACT_TARGET_MODULES),
    ):
        raise V16R3InferenceContractError(
            "v16r3 adapter is not exact all-30-block attn1/attn2 q/k/v/out"
        )
    observed = dict(adapter_config)
    observed.pop("target_modules")
    if not _strict_equal(observed, _expected_peft_config_without_targets()):
        raise V16R3InferenceContractError(
            "v16r3 adapter PEFT semantic closure differs"
        )


def validate_v16r3_adapter_contract(
    adapter_config: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    *,
    expected_adapter_config_sha256: str,
    expected_adapter_model_sha256: str,
    expected_training_receipt_sha256: str,
    expected_checkpoint_tree_sha256: str = delegated.trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Validate the externally hash-bound v16r3 S644 adapter contract.

    This function is deliberately free of Torch, PEFT, and GPU imports so the
    receipt boundary can be unit-tested on a login machine.
    """

    config_sha = _require_sha256(
        expected_adapter_config_sha256, label="expected adapter config SHA-256"
    )
    model_sha = _require_sha256(
        expected_adapter_model_sha256, label="expected adapter model SHA-256"
    )
    receipt_sha = _require_sha256(
        expected_training_receipt_sha256,
        label="expected training receipt SHA-256",
    )
    if expected_checkpoint_tree_sha256 != delegated.trainer.CHECKPOINT_TREE_SHA256:
        raise V16R3InferenceContractError(
            "v16r3 inference supports only the audited Bernini base checkpoint tree"
        )
    if not isinstance(adapter_config, Mapping) or not isinstance(
        training_receipt, Mapping
    ):
        raise V16R3InferenceContractError(
            "v16r3 adapter config and training receipt must be JSON objects"
        )
    _validate_peft_config(adapter_config)

    top_level = {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "adapter_config_sha256": config_sha,
        "adapter_model_sha256": model_sha,
        "bernini_commit": delegated.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": delegated.trainer.VEOMNI_TESTED_COMMIT,
        "global_step": GLOBAL_STEP,
        "max_steps": GLOBAL_STEP,
        "complete": True,
        "scientific_claim_authorized": False,
        "claim_scope": TRAINING_CLAIM_SCOPE,
        "last_reporting_scalar_is_not_a_joint_backpropagated_objective": True,
    }
    for key, expected in top_level.items():
        _require_field(training_receipt, key, expected, label="v16r3 receipt")
    for key in ("method_source_revision", "method_source_archive_sha256"):
        _require_sha256(training_receipt.get(key), label=f"v16r3 receipt {key}")

    contract = training_receipt.get("training_contract")
    if not isinstance(contract, Mapping):
        raise V16R3InferenceContractError(
            "v16r3 receipt lacks its training_contract"
        )
    expected_contract = {
        "method": TRAINING_METHOD,
        "training_objective": (
            "real_source_target_owned_routed_teacher_delta_v14r2"
        ),
        "route_operator": "self_target_owned_activity_kernel25_v14r2",
        "route_transport": (
            "self_target_owned_activity_kernel25_attn_output_v14r2"
        ),
        "dynaedit_sga_anc_reserved_for_decode_solver": True,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_scope": LORA_SCOPE,
        "lora_target_module_count": TARGET_MODULE_COUNT,
        "lora_target_modules_sha256": TARGET_MODULES_SHA256,
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "full_attention_lora_enabled": True,
        "full644_optimizer_schedule": "exact644_unique_rows_once",
        "all_full644_rows_targeted_exactly_once": True,
        "single_continuous_fresh_from_base_exact644_run": True,
        "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
        "starts_from_frozen_base_checkpoint_not_prior_adapter": True,
        "qk_only_zero_rms_backward_policy": ZERO_RMS_POLICY,
        "qk_only_zero_rms_backward_scope": ZERO_RMS_SCOPE,
        "qk_only_zero_rms_forward_values_changed": False,
        "qk_only_zero_rms_zero_subgradient": 0.0,
        "s279_endpoint_canary_covered": True,
        "sample_retry_or_skip_for_v16r3": False,
        "seed_or_timestep_changed_for_v16r3": False,
        "loss_scale_changed_for_v16r3": False,
        "component_preallreduce_finite_gate_relaxed": False,
        "nonfinite_gradient_committed": False,
        "manual_or_visual_review_controls_optimizer_admission": False,
        "scientific_claim_authorized": False,
    }
    for key, expected in expected_contract.items():
        _require_field(contract, key, expected, label="v16r3 training contract")

    summary = training_receipt.get("v16r3_zero_rms_backward_summary")
    if not isinstance(summary, Mapping):
        raise V16R3InferenceContractError(
            "v16r3 zero-RMS backward summary is absent"
        )
    expected_summary = {
        "policy": ZERO_RMS_POLICY,
        "scope": ZERO_RMS_SCOPE,
        "finite_nonnegative_forward_values_bit_exact": True,
        "zero_forward_value": 0.0,
        "zero_backward_subgradient": 0.0,
        "positive_backward_matches_standard_sqrt": True,
        "negative_or_nonfinite_values_masked": False,
        "loss_scale_changed": False,
        "seed_or_timestep_changed": False,
        "sample_retry_or_skip": False,
        "component_preallreduce_finite_gate_relaxed": False,
        "nonfinite_gradient_committed": False,
        "policy_fixed_from_step_one": True,
        "single_continuous_fresh_from_base_exact644": True,
        "scientific_claim_authorized": False,
    }
    for key, expected in expected_summary.items():
        _require_field(summary, key, expected, label="v16r3 zero-RMS summary")
    canary = summary.get("s279_endpoint_canary")
    if not isinstance(canary, Mapping):
        raise V16R3InferenceContractError("v16r3 S279 canary is absent")
    expected_canary = {
        "step": 279,
        "target_iid": "4aeb0557a94b4db3",
        "target_family": "fall",
        "expected_calls": S279_EXPECTED_CALLS,
        "observed_calls": S279_EXPECTED_CALLS,
        "covered_by_checkpoint": True,
    }
    for key, expected in expected_canary.items():
        _require_field(canary, key, expected, label="v16r3 S279 canary")

    memory = training_receipt.get("memory_gate")
    if not isinstance(memory, Mapping):
        raise V16R3InferenceContractError("v16r3 memory gate is absent")
    for key, expected in {
        "passed": True,
        "dummy_or_padding_allocations": False,
        "true_training_tensors_only": True,
    }.items():
        _require_field(memory, key, expected, label="v16r3 memory gate")
    minimum_reserved = memory.get("minimum_reserved_fraction")
    if (
        isinstance(minimum_reserved, bool)
        or not isinstance(minimum_reserved, (int, float))
        or not 0.5 < float(minimum_reserved) <= 1.0
    ):
        raise V16R3InferenceContractError(
            "v16r3 real-memory gate is not strictly above 50%"
        )

    coverage = training_receipt.get("gradient_coverage")
    if not isinstance(coverage, Mapping):
        raise V16R3InferenceContractError("v16r3 gradient coverage is absent")
    for key, expected in {
        "tensor_count": ADAPTER_TENSOR_COUNT,
        "nonzero_tensor_count": ADAPTER_TENSOR_COUNT,
    }.items():
        _require_field(coverage, key, expected, label="v16r3 gradient coverage")

    return {
        "global_step": GLOBAL_STEP,
        # ``infer_lora`` historically calls this field a receipt digest.  The
        # wrapper relabels it as a file SHA in the published receipt because
        # the v16r3 training receipt has no internal self-digest.
        "receipt_digest": receipt_sha,
        "training_receipt_sha256": receipt_sha,
        "adapter_config_sha256": config_sha,
        "adapter_model_sha256": model_sha,
        "target_modules_sha256": TARGET_MODULES_SHA256,
        "target_modules": expected_target_modules(),
        "target_module_count": TARGET_MODULE_COUNT,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "peft_version": PEFT_VERSION,
        "transformers_version": TRANSFORMERS_VERSION,
        "training_schema_version": TRAINING_RECEIPT_SCHEMA,
        "training_method": TRAINING_METHOD,
        "training_complete": True,
        "claim_scope": TRAINING_CLAIM_SCOPE,
        "method_source_revision": training_receipt["method_source_revision"],
        "method_source_archive_sha256": training_receipt[
            "method_source_archive_sha256"
        ],
        "v16r3_online_anchor": True,
    }


def _file_sha256(path: Path, *, label: str) -> str:
    if (
        str(path) in delegated._AUTHORIZED_FD_VIEW_FILES
        and delegated._ACTIVE_INHERITED_FDS is not None
    ):
        return delegated.file_sha256(path)
    path = delegated._plain_file(path, label=label)
    try:
        before = path.lstat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            fd_before = os.fstat(handle.fileno())
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            fd_after = os.fstat(handle.fileno())
        after = path.lstat()
    except OSError as error:
        raise V16R3InferenceContractError(
            f"cannot stably hash {label}: {path}: {error}"
        ) from error
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if not (
        identity(before)
        == identity(fd_before)
        == identity(fd_after)
        == identity(after)
    ):
        raise V16R3InferenceContractError(f"{label} changed while hashing: {path}")
    return digest.hexdigest()


def _identity_from_args(args: argparse.Namespace) -> ExpectedArtifactIdentity:
    if bool(getattr(args, "base_only", False)):
        raise V16R3InferenceContractError(
            "the v16r3 wrapper requires --adapter-checkpoint; base-only is unsupported"
        )
    if getattr(args, "adapter_checkpoint_manifest", None) is not None or getattr(
        args, "adapter_checkpoint_manifest_sha256", None
    ) is not None:
        raise V16R3InferenceContractError(
            "the v16r3 three-file checkpoint has no legacy checkpoint manifest"
        )
    config_sha = _require_sha256(
        getattr(args, "expected_adapter_config_sha256", None),
        label="expected adapter config SHA-256",
    )
    model_sha = _require_sha256(
        getattr(args, "expected_adapter_model_sha256", None),
        label="expected adapter model SHA-256",
    )
    receipt_sha = _require_sha256(
        getattr(args, "expected_training_receipt_sha256", None),
        label="expected training receipt SHA-256",
    )
    bundle = delegated.resolve_adapter_bundle(args.adapter_checkpoint)
    identity = ExpectedArtifactIdentity(
        checkpoint_root=bundle.checkpoint_root,
        adapter_dir=bundle.adapter_dir,
        adapter_config_path=bundle.adapter_config_path,
        adapter_model_path=bundle.adapter_model_path,
        training_receipt_path=bundle.training_receipt_path,
        adapter_config_sha256=config_sha,
        adapter_model_sha256=model_sha,
        training_receipt_sha256=receipt_sha,
    )
    _verify_artifact_hashes(identity, include_model=False)
    return identity


def _verify_artifact_hashes(
    identity: ExpectedArtifactIdentity, *, include_model: bool
) -> None:
    rows = [
        (
            identity.adapter_config_path,
            identity.adapter_config_sha256,
            "v16r3 adapter config",
        ),
        (
            identity.training_receipt_path,
            identity.training_receipt_sha256,
            "v16r3 training receipt",
        ),
    ]
    if include_model:
        rows.append(
            (
                identity.adapter_model_path,
                identity.adapter_model_sha256,
                "v16r3 adapter model",
            )
        )
    for path, expected, label in rows:
        actual = _file_sha256(path, label=label)
        if actual != expected:
            raise V16R3InferenceContractError(
                f"{label} SHA-256 differs: expected={expected} actual={actual}"
            )


def _active_identity() -> ExpectedArtifactIdentity:
    if _ACTIVE_EXPECTED_IDENTITY is None:
        raise V16R3InferenceContractError(
            "v16r3 expected artifact identity was not activated"
        )
    return _ACTIVE_EXPECTED_IDENTITY


def _assert_same_bundle(
    bundle: delegated.AdapterBundle, identity: ExpectedArtifactIdentity
) -> None:
    observed = (
        bundle.checkpoint_root,
        bundle.adapter_dir,
        bundle.adapter_config_path,
        bundle.adapter_model_path,
        bundle.training_receipt_path,
    )
    expected = (
        identity.checkpoint_root,
        identity.adapter_dir,
        identity.adapter_config_path,
        identity.adapter_model_path,
        identity.training_receipt_path,
    )
    if observed != expected:
        raise V16R3InferenceContractError(
            "v16r3 adapter bundle changed after CLI identity binding"
        )


def _validate_runtime_versions() -> None:
    for distribution, expected in (
        ("transformers", TRANSFORMERS_VERSION),
        ("peft", PEFT_VERSION),
    ):
        try:
            actual = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as error:
            raise V16R3InferenceContractError(
                f"required runtime distribution is absent: {distribution}"
            ) from error
        if actual != expected:
            raise V16R3InferenceContractError(
                f"v16r3 runtime {distribution} must be {expected}, got {actual}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = _DELEGATED_BUILD_PARSER()
    parser.allow_abbrev = False
    identity = parser.add_argument_group("required v16r3 artifact identity")
    identity.add_argument("--expected-adapter-config-sha256", required=True)
    identity.add_argument("--expected-adapter-model-sha256", required=True)
    identity.add_argument(
        "--expected-training-receipt-sha256",
        "--expected-adapter-receipt-sha256",
        dest="expected_training_receipt_sha256",
        required=True,
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    _DELEGATED_VALIDATE_CLI(args)
    _validate_runtime_versions()
    if bool(getattr(args, "base_only", False)):
        raise V16R3InferenceContractError(
            "the v16r3 wrapper requires --adapter-checkpoint; base-only is unsupported"
        )
    if getattr(args, "adapter_checkpoint_manifest", None) is not None or getattr(
        args, "adapter_checkpoint_manifest_sha256", None
    ) is not None:
        raise V16R3InferenceContractError(
            "the v16r3 three-file checkpoint has no legacy checkpoint manifest"
        )
    _require_sha256(
        getattr(args, "expected_adapter_config_sha256", None),
        label="expected adapter config SHA-256",
    )
    _require_sha256(
        getattr(args, "expected_adapter_model_sha256", None),
        label="expected adapter model SHA-256",
    )
    _require_sha256(
        getattr(args, "expected_training_receipt_sha256", None),
        label="expected training receipt SHA-256",
    )


def activate_model_consumption_authority(
    args: argparse.Namespace,
) -> Mapping[str, Any] | None:
    global _ACTIVE_EXPECTED_IDENTITY

    evidence = _DELEGATED_ACTIVATE_MODEL_CONSUMPTION_AUTHORITY(args)
    # Bind after delegated FD-view activation so both direct checkpoints and
    # retained model-consumption views use the same three-SHA boundary.
    _ACTIVE_EXPECTED_IDENTITY = _identity_from_args(args)
    return evidence


def validate_adapter_contract(
    adapter_config: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = delegated.trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    identity = _active_identity()
    # Recheck the two small JSON inputs immediately after ``infer_lora`` read
    # them; the 755-MiB model is checked immediately before strict PEFT load.
    _verify_artifact_hashes(identity, include_model=False)
    return validate_v16r3_adapter_contract(
        adapter_config,
        training_receipt,
        expected_adapter_config_sha256=identity.adapter_config_sha256,
        expected_adapter_model_sha256=identity.adapter_model_sha256,
        expected_training_receipt_sha256=identity.training_receipt_sha256,
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
    identity = _active_identity()
    _assert_same_bundle(adapter, identity)
    _verify_artifact_hashes(identity, include_model=True)
    result = _DELEGATED_STRICT_LOAD_AND_MERGE(
        base_model,
        adapter,
        expected_targets,
        route_scope=route_scope,
        require_zero_effect=require_zero_effect,
    )
    # The original loader compares every loaded A/B tensor to safetensors.
    # Recheck the small semantic inputs after that potentially long read.
    _verify_artifact_hashes(identity, include_model=False)
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
        raise V16R3InferenceContractError(
            "v16r3 inference receipt does not contain an enabled adapter"
        )
    if adapter.get("adapter_model_sha256") != adapter_identity.get(
        "adapter_model_sha256"
    ):
        raise V16R3InferenceContractError(
            "v16r3 inference receipt adapter model identity differs"
        )
    if (
        adapter.get("tensor_count") != ADAPTER_TENSOR_COUNT
        or adapter.get("strictly_reloaded") is not True
        or adapter.get("safe_merged_for_inference") is not True
    ):
        raise V16R3InferenceContractError(
            "v16r3 inference receipt lost strict 480-tensor reload/safe merge"
        )
    legacy_digest = adapter.pop("training_receipt_digest", None)
    if legacy_digest != adapter_identity.get("training_receipt_sha256"):
        raise V16R3InferenceContractError(
            "v16r3 delegated receipt lost its externally bound training receipt"
        )
    adapter.update(
        {
            "training_receipt_sha256": adapter_identity[
                "training_receipt_sha256"
            ],
            "adapter_config_sha256": adapter_identity["adapter_config_sha256"],
            "training_identity": {
                "schema_version": adapter_identity["training_schema_version"],
                "method": adapter_identity["training_method"],
                "global_step": adapter_identity["global_step"],
                "complete": adapter_identity["training_complete"],
                "lora_rank": adapter_identity["lora_rank"],
                "lora_alpha": adapter_identity["lora_alpha"],
                "lora_scope": LORA_SCOPE,
                "target_module_count": adapter_identity[
                    "target_module_count"
                ],
                "target_modules_sha256": adapter_identity[
                    "target_modules_sha256"
                ],
                "adapter_config_sha256": adapter_identity[
                    "adapter_config_sha256"
                ],
                "adapter_model_sha256": adapter_identity[
                    "adapter_model_sha256"
                ],
                "training_receipt_sha256": adapter_identity[
                    "training_receipt_sha256"
                ],
                "training_method_source_revision": adapter_identity[
                    "method_source_revision"
                ],
                "training_method_source_archive_sha256": adapter_identity[
                    "method_source_archive_sha256"
                ],
                "claim_scope": adapter_identity["claim_scope"],
                "scientific_claim_authorized": False,
            },
            "v16r3": {
                "decode_mode": "adapter_only_direct_rv2v",
                "online_anchor_route_applied": False,
                "lora_rank": adapter_identity["lora_rank"],
                "lora_alpha": adapter_identity["lora_alpha"],
                "target_module_count": adapter_identity[
                    "target_module_count"
                ],
                "training_receipt_schema": adapter_identity[
                    "training_schema_version"
                ],
                "adapter_model_sha256": adapter_identity[
                    "adapter_model_sha256"
                ],
                "adapter_config_sha256": adapter_identity[
                    "adapter_config_sha256"
                ],
                "training_receipt_sha256": adapter_identity[
                    "training_receipt_sha256"
                ],
                # Short aliases are retained for runner-side jq postchecks;
                # the explicit names above remain the authoritative labels.
                "adapter_sha256": adapter_identity["adapter_model_sha256"],
                "config_sha256": adapter_identity["adapter_config_sha256"],
                "receipt_sha256": adapter_identity[
                    "training_receipt_sha256"
                ],
            },
        }
    )
    annotated["schema_version"] = INFERENCE_RECEIPT_SCHEMA
    annotated["delegated_inference"] = {
        "schema_version": delegated.INFERENCE_RECEIPT_SCHEMA,
        "infer_lora_source_sha256": annotated.get("infer_lora_source_sha256"),
        "sampling_and_strict_lora_loader_reused": True,
    }
    annotated["v16r3_inference_wrapper"] = {
        "source_sha256": wrapper_source_sha256,
        "training_receipt_schema": TRAINING_RECEIPT_SCHEMA,
        "expected_artifact_sha256_required_on_cli": True,
        "training_receipt_identity_kind": "external_file_sha256",
    }
    annotated.pop("receipt_digest", None)
    annotated["receipt_digest"] = delegated.object_sha256(annotated)
    return annotated


def build_inference_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    adapter_identity = kwargs.get("adapter_identity")
    if not isinstance(adapter_identity, Mapping) or adapter_identity.get(
        "v16r3_online_anchor"
    ) is not True:
        raise V16R3InferenceContractError(
            "v16r3 inference receipt lacks its validated adapter identity"
        )
    receipt = _DELEGATED_BUILD_INFERENCE_RECEIPT(*args, **kwargs)
    wrapper_sha = _file_sha256(
        Path(__file__).resolve(), label="v16r3 inference wrapper source"
    )
    return _annotate_inference_receipt(
        receipt,
        adapter_identity=adapter_identity,
        wrapper_source_sha256=wrapper_sha,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _ACTIVE_EXPECTED_IDENTITY

    originals = (
        delegated.build_parser,
        delegated.validate_cli,
        delegated.activate_model_consumption_authority,
        delegated.validate_adapter_contract,
        delegated._strict_load_and_merge_adapter,
        delegated.build_inference_receipt,
    )
    _ACTIVE_EXPECTED_IDENTITY = None
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
        _ACTIVE_EXPECTED_IDENTITY = None


if __name__ == "__main__":
    raise SystemExit(main())
