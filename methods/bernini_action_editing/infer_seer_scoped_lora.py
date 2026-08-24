#!/usr/bin/env python3
"""Strict source-only inference for the 60-module SEER B0 adapter.

``infer_lora.py`` intentionally remains the validator for the original
240-projection action LoRA.  SEER B0 instead trains rank-8 LoRA factors only
on ``attn2`` (text cross-attention) Q/O in all 30 Wan blocks.  This wrapper
reuses the same audited Bernini sampler while replacing only the adapter
scope/receipt validators and the strict PEFT loader.

The target video used during SEER training is never accepted by this CLI.
Runtime conditions remain exactly one source video and one edit instruction.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as base  # noqa: E402

# When invoked by file path, keep the canonical module name bound to this
# exact object.  The lazily imported same-state receipt validator imports the
# canonical name to share this loader; without the alias Python would execute
# this wrapper a second time and create a different exception hierarchy.
if __name__ == "__main__":
    sys.modules.setdefault("infer_seer_scoped_lora", sys.modules[__name__])


TRAINING_METHOD = "self-generated-event-erasure-flow-matching-b0"
TRAINING_RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-receipt-v2"
TRAINING_SCOPE = "all_30_blocks_attn2_cross_attention_q_and_out_rank8"
TARGET_MODULE_COUNT = 60
ADAPTER_TENSOR_COUNT = 2 * TARGET_MODULE_COUNT
REQUIRED_GLOBAL_STEP = 4
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CROSS_Q_OUT = re.compile(
    r".+\.blocks\.(?P<block>\d+)\.attn2\.(?:to_q|to_out\.0)\Z"
)


class SeerInferenceError(base.InferenceContractError):
    """Raised before generation when the scoped SEER contract differs."""


def expected_lora_target_modules() -> list[str]:
    """Exact fully-qualified target-row Q/O scope emitted by SEER B0."""

    return sorted(
        f"diff_dec.transformer.blocks.{block}.attn2.{projection}"
        for block in range(30)
        for projection in ("to_q", "to_out.0")
    )


def expected_adapter_state_keys() -> list[str]:
    return sorted(
        f"base_model.model.{module}.lora_{factor}.weight"
        for module in expected_lora_target_modules()
        for factor in ("A", "B")
    )


def _sha(value: Any, *, label: str, sha1: bool = False) -> str:
    pattern = _SHA1 if sha1 else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        kind = "SHA-1" if sha1 else "SHA-256"
        raise SeerInferenceError(f"{label} is not a lowercase {kind}")
    return value


def _finite_positive(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise SeerInferenceError(f"{label} must be finite and positive")
    return float(value)


def _validate_serialized_target_coverage(serialized: Any) -> list[str]:
    """Accept PEFT's compact suffixes only when they cover exactly SEER Q/O."""

    if (
        not isinstance(serialized, list)
        or not serialized
        or any(not isinstance(name, str) or not name for name in serialized)
        or len(serialized) != len(set(serialized))
    ):
        raise SeerInferenceError(
            "adapter target_modules must be a unique non-empty string list"
        )
    expected = set(expected_lora_target_modules())
    covered: set[str] = set()
    for suffix in serialized:
        matches = {
            target
            for target in expected
            if target == suffix or target.endswith(f".{suffix}")
        }
        if not matches:
            raise SeerInferenceError(
                "adapter serialized target_modules exceed the SEER Q/O scope"
            )
        covered.update(matches)
    if covered != expected:
        raise SeerInferenceError(
            "adapter serialized target_modules do not cover all 60 SEER modules"
        )
    return sorted(serialized)


def validate_b0_adapter_contract(
    adapter_config: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = base.trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Validate the B0 receipt without weakening ``infer_lora``'s contract."""

    _validate_serialized_target_coverage(adapter_config.get("target_modules"))
    if adapter_config.get("peft_type") != "LORA":
        raise SeerInferenceError("adapter peft_type must be LORA")
    if adapter_config.get("r") != 8 or float(adapter_config.get("lora_alpha", -1)) != 8.0:
        raise SeerInferenceError("SEER adapter rank/alpha must both be 8")
    if float(adapter_config.get("lora_dropout", -1)) != 0.0:
        raise SeerInferenceError("SEER adapter dropout must be zero")
    if adapter_config.get("bias") != "none":
        raise SeerInferenceError("SEER adapter bias must be none")
    if adapter_config.get("modules_to_save") not in (None, []):
        raise SeerInferenceError("SEER adapter modules_to_save are forbidden")
    if adapter_config.get("use_dora") not in (None, False):
        raise SeerInferenceError("DoRA is outside the SEER B0 contract")
    if adapter_config.get("use_rslora") not in (None, False):
        raise SeerInferenceError("RS-LoRA is outside the SEER B0 contract")

    receipt = dict(training_receipt)
    declared_digest = receipt.pop("receipt_digest", None)
    if (
        not isinstance(declared_digest, str)
        or _SHA256.fullmatch(declared_digest) is None
        or base.object_sha256(receipt) != declared_digest
    ):
        raise SeerInferenceError("SEER training receipt digest differs")
    if training_receipt.get("schema_version") != TRAINING_RECEIPT_SCHEMA:
        raise SeerInferenceError("SEER training receipt schema differs")
    if training_receipt.get("method") != TRAINING_METHOD:
        raise SeerInferenceError("SEER training method identity differs")
    if training_receipt.get("global_step") != REQUIRED_GLOBAL_STEP:
        raise SeerInferenceError("SEER held-out decode requires the four-step checkpoint")
    if training_receipt.get("bernini_commit") != base.trainer.BERNINI_OFFICIAL_COMMIT:
        raise SeerInferenceError("SEER training Bernini revision differs")
    if training_receipt.get("veomni_commit") != base.trainer.VEOMNI_TESTED_COMMIT:
        raise SeerInferenceError("SEER training VeOmni revision differs")
    if training_receipt.get("checkpoint_tree_sha256") != expected_checkpoint_tree_sha256:
        raise SeerInferenceError("SEER training checkpoint tree differs")
    if training_receipt.get("bernini_training_files_index_sha256") != base.object_sha256(
        base.trainer.BERNINI_PINNED_FILE_HASHES
    ):
        raise SeerInferenceError("SEER training source-file index differs")

    expected_targets = expected_lora_target_modules()
    if (
        training_receipt.get("target_module_count") != TARGET_MODULE_COUNT
        or training_receipt.get("target_modules_sha256")
        != base.object_sha256(expected_targets)
    ):
        raise SeerInferenceError("SEER receipt target-row Q/O scope differs")
    contract = training_receipt.get("training_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("lora_scope") != TRAINING_SCOPE
        or contract.get("lora_rank") != 8
        or float(contract.get("lora_alpha", -1)) != 8.0
        or contract.get("conditioning") != [
            "clean_source_video_vae",
            "edit_instruction",
        ]
        or contract.get("target_embedding_or_caption_conditioning") is not False
        or contract.get("external_spatial_mask") is not False
        or contract.get("external_tracking_or_swept_tube") is not False
        or contract.get("num_frames") != 81
        or contract.get("latent_frames") != 21
    ):
        raise SeerInferenceError("SEER source-only training contract differs")
    transformers_version = contract.get("transformers_version")
    if not isinstance(transformers_version, str) or not transformers_version:
        raise SeerInferenceError("SEER receipt lacks Transformers version")

    distributed = training_receipt.get("distributed")
    if (
        not isinstance(distributed, Mapping)
        or distributed.get("world_size") != 4
        or distributed.get("ulysses_size") != 4
        or distributed.get("explicit_lora_gradient_all_reduce") is not True
    ):
        raise SeerInferenceError("SEER adapter lacks the four-rank training contract")
    seer = training_receipt.get("seer")
    update = training_receipt.get("parameter_update_evidence")
    if (
        not isinstance(seer, Mapping)
        or _SHA256.fullmatch(str(seer.get("owner_spec_sha256"))) is None
        or seer.get("self_generated_target_supervision") is not True
        or seer.get("training_completion_is_method_success") is not False
        or seer.get("heldout_decoded_review_required") is not True
        or not isinstance(update, Mapping)
        or update.get("exact_parameter_bytes_changed") is not True
        or update.get("method_success_claimed") is not False
    ):
        raise SeerInferenceError("SEER authority/parameter-update evidence differs")
    initial = _sha(
        update.get("initial_trainable_parameter_digest"),
        label="SEER initialization digest",
    )
    final = _sha(
        update.get("final_trainable_parameter_digest"),
        label="SEER final parameter digest",
    )
    if initial == final:
        raise SeerInferenceError("SEER saved parameters equal initialization")
    _finite_positive(
        training_receipt.get("last_preclip_gradient_norm"),
        label="SEER final gradient norm",
    )
    if (
        training_receipt.get("production_claim_forbidden") is not True
        or training_receipt.get("scientific_claim_authorized") is not False
    ):
        raise SeerInferenceError("SEER receipt carries an unauthorized success claim")
    method_revision = _sha(
        training_receipt.get("method_source_revision"),
        label="SEER method source revision",
        sha1=True,
    )
    method_archive = _sha(
        training_receipt.get("method_source_archive_sha256"),
        label="SEER method source archive",
    )
    return {
        "global_step": REQUIRED_GLOBAL_STEP,
        "receipt_digest": declared_digest,
        "target_modules_sha256": base.object_sha256(expected_targets),
        "transformers_version": transformers_version,
        "method_source_revision": method_revision,
        "method_source_archive_sha256": method_archive,
        "seer_owner_spec_sha256": seer["owner_spec_sha256"],
    }


def validate_adapter_contract(
    adapter_config: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = base.trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Dispatch between two closed SEER receipt layouts, never by aliases.

    B0 is the standard-FM engineering canary.  The second admitted layout is
    the genuine same-noisy-state FM+motion+copy update.  Both retain the exact
    same 60-module/120-factor loader below; only their training receipts differ.
    """

    schema = training_receipt.get("schema_version")
    method = training_receipt.get("method")
    if schema == TRAINING_RECEIPT_SCHEMA and method == TRAINING_METHOD:
        return validate_b0_adapter_contract(
            adapter_config,
            training_receipt,
            expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
        )
    # Import lazily after this module has initialized.  The same-state
    # validator imports this module only for the shared exact-scope loader, so
    # a top-level import would otherwise create a needless cycle.
    from infer_seer_same_state_lora import (
        TRAINING_METHOD as SAME_STATE_METHOD,
        TRAINING_RECEIPT_SCHEMA as SAME_STATE_SCHEMA,
        validate_adapter_contract as validate_same_state,
    )

    if schema == SAME_STATE_SCHEMA and method == SAME_STATE_METHOD:
        return validate_same_state(
            adapter_config,
            training_receipt,
            expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
        )
    raise SeerInferenceError("training receipt is not one admitted SEER layout")


def _plain_safetensors(path: Path) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SeerInferenceError(f"adapter safetensors is unavailable: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise SeerInferenceError("adapter safetensors must be a plain file")
    return path


def inspect_safetensors_tensor_keys(path: Path) -> list[str]:
    """Parse the real safetensors header without importing GPU dependencies."""

    file_path = _plain_safetensors(path)
    size = file_path.stat().st_size
    if size < 10:
        raise SeerInferenceError("adapter safetensors is truncated")
    with file_path.open("rb") as handle:
        prefix = handle.read(8)
        header_length = int.from_bytes(prefix, "little", signed=False)
        if header_length <= 1 or header_length > size - 8:
            raise SeerInferenceError("adapter safetensors header length differs")
        try:
            header = json.loads(handle.read(header_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SeerInferenceError(f"adapter safetensors header is invalid: {error}") from error
    if not isinstance(header, dict):
        raise SeerInferenceError("adapter safetensors header must be an object")
    data_bytes = size - 8 - header_length
    spans: list[tuple[int, int, str]] = []
    for key, descriptor in header.items():
        if key == "__metadata__":
            if not isinstance(descriptor, dict):
                raise SeerInferenceError("adapter safetensors metadata differs")
            continue
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(descriptor, Mapping)
            or not isinstance(descriptor.get("dtype"), str)
            or not isinstance(descriptor.get("shape"), list)
            or any(type(value) is not int or value < 0 for value in descriptor["shape"])
            or not isinstance(descriptor.get("data_offsets"), list)
            or len(descriptor["data_offsets"]) != 2
            or any(type(value) is not int for value in descriptor["data_offsets"])
        ):
            raise SeerInferenceError("adapter safetensors tensor descriptor differs")
        start, end = descriptor["data_offsets"]
        if start < 0 or end <= start or end > data_bytes:
            raise SeerInferenceError("adapter safetensors data offsets differ")
        spans.append((start, end, key))
    ordered = sorted(spans)
    if not ordered or ordered[0][0] != 0 or ordered[-1][1] != data_bytes:
        raise SeerInferenceError("adapter safetensors data region is not fully bound")
    for left, right in zip(ordered, ordered[1:]):
        if left[1] != right[0]:
            raise SeerInferenceError("adapter safetensors tensor spans overlap or have gaps")
    return sorted(key for _, _, key in ordered)


def validate_scoped_safetensors(path: Path) -> list[str]:
    actual = inspect_safetensors_tensor_keys(path)
    expected = expected_adapter_state_keys()
    if actual != expected or len(actual) != ADAPTER_TENSOR_COUNT:
        raise SeerInferenceError(
            "adapter safetensors keys differ from the exact 120-factor SEER scope"
        )
    return actual


def validate_adapter_state_dicts(
    saved: Mapping[str, Any],
    loaded: Mapping[str, Any],
    *,
    tensor_equal: Any,
) -> int:
    expected = set(expected_adapter_state_keys())
    if set(saved) != expected or set(loaded) != expected:
        raise SeerInferenceError(
            "strict PEFT reload keys differ from the exact SEER Q/O scope"
        )
    unequal = [key for key in sorted(expected) if not tensor_equal(saved[key], loaded[key])]
    if unequal:
        raise SeerInferenceError(f"strict SEER PEFT reload tensors differ: {unequal[:4]}")
    return ADAPTER_TENSOR_COUNT


def _strict_load_and_merge_adapter(
    base_model: Any,
    adapter: base.AdapterBundle,
    expected_targets: Sequence[str],
) -> tuple[Any, int]:
    import torch
    from peft import LoraConfig, PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file as load_safetensors

    if list(expected_targets) != expected_lora_target_modules():
        raise SeerInferenceError("requested runtime SEER target scope differs")
    all_targets = base.trainer.select_attention_projection_names(base_model)
    actual_targets = sorted(name for name in all_targets if _CROSS_Q_OUT.fullmatch(name))
    if actual_targets != list(expected_targets):
        raise SeerInferenceError("runtime Bernini SEER target module set differs")
    validate_scoped_safetensors(adapter.adapter_model_path)
    config = LoraConfig.from_pretrained(str(adapter.adapter_dir), local_files_only=True)
    config.target_modules = set(expected_targets)
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter.adapter_dir),
        is_trainable=False,
        config=config,
        local_files_only=True,
    )
    saved = load_safetensors(str(adapter.adapter_model_path), device="cpu")
    loaded = get_peft_model_state_dict(peft_model, adapter_name="default")
    count = validate_adapter_state_dicts(
        saved,
        loaded,
        tensor_equal=lambda left, right: bool(torch.equal(left.cpu(), right.cpu())),
    )
    merged = peft_model.merge_and_unload(safe_merge=True)
    if any("lora_" in name for name, _ in merged.named_modules()):
        raise SeerInferenceError("SEER LoRA modules remain after safe merge")
    merged.requires_grad_(False)
    merged.eval()
    return merged, count


def _install_specialization() -> None:
    base.expected_lora_target_modules = expected_lora_target_modules
    base.expected_adapter_state_keys = expected_adapter_state_keys
    base.EXPECTED_ADAPTER_TENSOR_COUNT = ADAPTER_TENSOR_COUNT
    base.validate_adapter_contract = validate_adapter_contract
    base.validate_adapter_state_dicts = validate_adapter_state_dicts
    base._strict_load_and_merge_adapter = _strict_load_and_merge_adapter


def main(argv: Optional[Sequence[str]] = None) -> int:
    _install_specialization()
    preliminary = base.build_parser().parse_args(argv)
    if preliminary.adapter_checkpoint:
        bundle = base.resolve_adapter_bundle(preliminary.adapter_checkpoint)
        validate_scoped_safetensors(bundle.adapter_model_path)
    return base.main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SeerInferenceError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
