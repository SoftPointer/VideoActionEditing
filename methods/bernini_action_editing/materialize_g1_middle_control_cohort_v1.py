#!/usr/bin/env python3
"""Materialize fail-closed G1 controls for projected ``Delta H_middle``.

Inputs are firewall-safe caches emitted by
``materialize_decoded_middle_action_repr_v1.py``.  Correct, temporal-shuffle
and reverse caches must be extracted independently with the same frozen model,
projection and matched-noise contract.  This utility adds exact zero/noop, a
prefix-only incomplete cache, and a different-action cache scaled by one
common scalar to match correct residual RMS.  Absolute hidden states, RGB and
latents are never accepted by this ABI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
from typing import Any, Mapping, Sequence


COHORT_SCHEMA_VERSION = "bernini-g1-middle-control-cohort-v2"
BUNDLE_RECEIPT_SCHEMA_VERSION = "bernini-g1-middle-control-bundle-receipt-v2"
UPSTREAM_CACHE_SCHEMA = "bernini-decoded-middle-action-representation-cache-v1"
UPSTREAM_RECEIPT_SCHEMA = "bernini-decoded-middle-action-representation-receipt-v1"
BLOCK_INDICES = (6, 12, 18, 24)
REQUIRED_TENSORS = tuple(f"middle_block_{index:02d}" for index in BLOCK_INDICES)
ANCHOR_KINDS = ("target", "selfgen")
EXTERNAL_ROLES = ("correct", "temporal_shuffle", "reverse", "wrong_action_donor")
TARGET_EXTERNAL_ROLE_MAP = {
    "correct": "real_forward",
    "temporal_shuffle": "temporal_shuffle",
    "reverse": "reverse",
    "wrong_action_donor": "real_forward",
}
SELFGEN_EXTERNAL_ROLE_MAP = {
    "correct": "self_generated",
    "temporal_shuffle": "self_generated_temporal_shuffle",
    "reverse": "self_generated_reverse",
    "wrong_action_donor": "self_generated",
}
EXTERNAL_ROLE_MAP_BY_ANCHOR = {
    "target": TARGET_EXTERNAL_ROLE_MAP,
    "selfgen": SELFGEN_EXTERNAL_ROLE_MAP,
}
UPSTREAM_SELFGEN_ROLES = frozenset(SELFGEN_EXTERNAL_ROLE_MAP.values())
UPSTREAM_CONTROL_ROLES = frozenset(
    (*TARGET_EXTERNAL_ROLE_MAP.values(), *SELFGEN_EXTERNAL_ROLE_MAP.values())
)
GENERATED_CONTROLS = (
    "zero_or_noop",
    "incomplete",
    "wrong_action_energy_matched",
)
PHASES = 21
EXPLICIT_GAUSSIAN_DOMAIN = (
    "bernini-decoded-middle-explicit-prepack-gaussian-v1"
)
EXPLICIT_GAUSSIAN_AUTHORITY_KIND = (
    "rank0_domain_seeded_explicit_prepack_fp32_gaussian"
)
EXPLICIT_GAUSSIAN_MATCH_CRITERION = (
    "same_canonical_raw_fp32_tensor_injected_exactly_once_per_branch"
)
DETERMINISTIC_VAE_AUTHORITY_KIND = (
    "rank0_local_strict_deterministic_vae_encode"
)
DETERMINISTIC_VAE_POLICY = (
    "rank0_two_branch_vae_encode_in_local_strict_deterministic_scope_"
    "with_exact_flag_restoration_v1"
)
DETERMINISTIC_VAE_SCOPE = (
    "action_and_first_frame_repeat_encode_calls_only"
)
PINNED_BERNINI_DATA_SHA256 = (
    "29aa4f89579c7771cb9f78706fde4f0dca0de954fdb2f5e2de1abacd8a0d6c65"
)
PINNED_PACK_VAE_LATENTS_SOURCE_SHA256 = (
    "445893fee2cca1f745265cea857740937f338a04b67e9f895fef943948c49c9f"
)
PINNED_PROCESS_RENDERER_SAMPLE_SOURCE_SHA256 = (
    "9e8532898267ea167f0776a71a30233cbfada4f94132e0b546f1740115ee372e"
)
ENERGY_DEFINITION = "sqrt(sum(all_block_residual_squared)/number_of_residual_scalars)"
ENERGY_EPSILON = 1.0e-12
ENERGY_MATCH_RTOL = 2.0e-5
MAX_ENERGY_SCALE = 100.0
ENERGY_SCALE_CALIBRATION_ITERATIONS = 32
ENERGY_SCALE_CALIBRATION_COMPUTE_DTYPE = "torch.float64"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class G1MiddleControlError(RuntimeError):
    """Raised when a middle-representation control cannot be proven safe."""


@dataclass(frozen=True)
class LoadedMiddle:
    path: Path
    sha256: str
    receipt_path: Path
    receipt_sha256: str
    receipt: dict[str, Any]
    metadata: dict[str, str]
    tensors: dict[str, Any]


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            text = json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ) + "\n"
        else:
            text = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
    except (TypeError, ValueError, UnicodeError) as error:
        raise G1MiddleControlError("value is not finite canonical ASCII JSON") from error
    return text.encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise G1MiddleControlError(f"{label} must be a sealed identifier")
    return value


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise G1MiddleControlError(f"{label} must be lowercase SHA-256")
    return value


def _regular_path(value: Path | str, *, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise G1MiddleControlError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _read_json(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any], str]:
    resolved = _regular_path(path, label=label)
    payload = resolved.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise G1MiddleControlError(f"{label} must be ASCII JSON") from error
    if not isinstance(value, dict):
        raise G1MiddleControlError(f"{label} must be a JSON object")
    return resolved, value, _sha256_bytes(payload)


def _tensor_sha256(value: Any) -> str:
    import torch

    owned = value.detach().to(device="cpu").clone(memory_format=torch.contiguous_format).contiguous()
    header = _canonical_json_bytes(
        {"dtype": str(owned.dtype), "shape": list(map(int, owned.shape))}
    )
    payload = owned.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\0" + payload).hexdigest()


def _validate_explicit_gaussian_match(
    representation: Mapping[str, Any], *, label: str
) -> None:
    """Authenticate the extractor's pre-interpolation Gaussian authority.

    G1 must never silently consume the legacy inverse reconstruction from
    ``input_vae_latents``/``target_velocity``.  It accepts only the canonical
    FP32 Gaussian injected into each private pre-pack branch exactly once.
    """

    gaussian_match = representation.get("gaussian_match")
    if not isinstance(gaussian_match, Mapping):
        raise G1MiddleControlError(
            f"{label} explicit pre-pack Gaussian match authority is missing"
        )
    expected_match_keys = {
        "comparison_stage",
        "criterion",
        "inverse_recovery_numerical_fields_applicable",
        "canonical_gaussian_sha256",
        "both_branches_retimed_from_canonical_gaussian",
        "fixed_absolute_tolerance_is_authority",
        "authority",
    }
    authority = gaussian_match.get("authority")
    if not isinstance(authority, Mapping):
        raise G1MiddleControlError(
            f"{label} explicit pre-pack Gaussian authority record is missing"
        )
    expected_authority_keys = {
        "authority_kind",
        "domain",
        "producer_rank",
        "base_seed",
        "derived_seed",
        "dtype",
        "shape",
        "canonical_gaussian_sha256",
        "broadcast_transport",
        "world_size",
        "world4_raw_sha256_consensus",
        "action_injection_count",
        "noop_injection_count",
        "action_gaussian_sha256",
        "noop_gaussian_sha256",
        "raw_noise_sigma_dtype",
        "raw_noise_sigma_shape",
        "action_raw_noise_sigma_sha256",
        "noop_raw_noise_sigma_sha256",
        "clean_capture_stage",
        "packed_state_original_op_order_bit_exact",
        "target_velocity_bit_exact",
        "recovered_from_x_or_velocity",
        "vendor_data_file_sha256",
        "pack_vae_latents_source_sha256",
        "process_renderer_sample_source_sha256",
        "vendor_module_mutated",
        "original_function_globals_mutated",
        "trainer_received_authority",
    }
    canonical_sha = gaussian_match.get("canonical_gaussian_sha256")
    shape = authority.get("shape")
    sigma_shape = authority.get("raw_noise_sigma_shape")
    if (
        set(gaussian_match) != expected_match_keys
        or gaussian_match.get("comparison_stage") != "before_fm_interpolation"
        or gaussian_match.get("criterion") != EXPLICIT_GAUSSIAN_MATCH_CRITERION
        or gaussian_match.get("inverse_recovery_numerical_fields_applicable")
        is not False
        or type(canonical_sha) is not str
        or _SHA256.fullmatch(str(canonical_sha)) is None
        or gaussian_match.get("both_branches_retimed_from_canonical_gaussian")
        is not True
        or gaussian_match.get("fixed_absolute_tolerance_is_authority") is not False
        or set(authority) != expected_authority_keys
        or authority.get("authority_kind") != EXPLICIT_GAUSSIAN_AUTHORITY_KIND
        or authority.get("domain") != EXPLICIT_GAUSSIAN_DOMAIN
        or authority.get("producer_rank") != 0
        or type(authority.get("base_seed")) is not int
        or int(authority.get("base_seed", -1)) < 0
        or type(authority.get("derived_seed")) is not int
        or not 0 < int(authority.get("derived_seed", 0)) <= 2**63 - 1
        or authority.get("dtype") != "torch.float32"
        or not isinstance(shape, list)
        or len(shape) != 5
        or any(type(value) is not int or value <= 0 for value in (shape or []))
        or tuple((shape or [])[1:]) != (16, 1, 2, 2)
        or authority.get("canonical_gaussian_sha256") != canonical_sha
        or authority.get("broadcast_transport")
        != "torch_distributed_nccl_fp32_tensor_broadcast"
        or authority.get("world_size") != 4
        or authority.get("world4_raw_sha256_consensus") is not True
        or authority.get("action_injection_count") != 1
        or authority.get("noop_injection_count") != 1
        or authority.get("action_gaussian_sha256") != canonical_sha
        or authority.get("noop_gaussian_sha256") != canonical_sha
        or authority.get("raw_noise_sigma_dtype")
        not in {"torch.bfloat16", "torch.float16", "torch.float32", "torch.float64"}
        or not isinstance(sigma_shape, list)
        or any(
            type(value) is not int or value <= 0 for value in (sigma_shape or [])
        )
        or math.prod(sigma_shape or []) != 1
        or type(authority.get("action_raw_noise_sigma_sha256")) is not str
        or _SHA256.fullmatch(
            str(authority.get("action_raw_noise_sigma_sha256", ""))
        )
        is None
        or authority.get("noop_raw_noise_sigma_sha256")
        != authority.get("action_raw_noise_sigma_sha256")
        or authority.get("clean_capture_stage")
        != "inside_cloned_pack_before_fm_interpolation"
        or authority.get("packed_state_original_op_order_bit_exact") is not True
        or authority.get("target_velocity_bit_exact") is not True
        or authority.get("recovered_from_x_or_velocity") is not False
        or authority.get("vendor_data_file_sha256") != PINNED_BERNINI_DATA_SHA256
        or authority.get("pack_vae_latents_source_sha256")
        != PINNED_PACK_VAE_LATENTS_SOURCE_SHA256
        or authority.get("process_renderer_sample_source_sha256")
        != PINNED_PROCESS_RENDERER_SAMPLE_SOURCE_SHA256
        or authority.get("vendor_module_mutated") is not False
        or authority.get("original_function_globals_mutated") is not False
        or authority.get("trainer_received_authority") is not False
        or representation.get("noise_max_abs_error") != 0.0
        or "noise_match_atol" in representation
    ):
        raise G1MiddleControlError(
            f"{label} explicit pre-pack Gaussian authority differs"
        )


def _validate_deterministic_vae_authority(
    representation: Mapping[str, Any], *, label: str
) -> None:
    """Authenticate the extractor's unmodified deterministic VAE posterior.

    The G1 consumer intentionally duplicates this small receipt ABI instead of
    importing the extractor.  A legacy receipt, a partially deterministic
    encode, a repaired/spliced posterior, or a nonzero phase-0 tolerance must
    therefore fail before any cohort artifact can be published.
    """

    authority = representation.get("deterministic_vae_authority")
    if not isinstance(authority, Mapping):
        raise G1MiddleControlError(
            f"{label} deterministic VAE authority is missing"
        )
    expected_authority_keys = {
        "authority_kind",
        "policy",
        "producer_rank",
        "encode_call_count",
        "scope",
        "before_flags",
        "during_flags",
        "restored_flags",
        "flags_restored_exact",
        "posterior_phase0_max_abs_error",
        "posterior_phase0_bit_exact",
        "action_phase0_posterior_sha256",
        "noop_phase0_posterior_sha256",
        "posterior_modified_after_encode",
        "posterior_copy_or_splice_used",
        "trainer_received_posterior",
    }
    flag_keys = {
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
    }
    expected_during = {
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    before = authority.get("before_flags")
    during = authority.get("during_flags")
    restored = authority.get("restored_flags")
    posterior_error = authority.get("posterior_phase0_max_abs_error")
    action_phase0_sha = authority.get("action_phase0_posterior_sha256")
    noop_phase0_sha = authority.get("noop_phase0_posterior_sha256")
    clean_phase0_error = representation.get("phase0_clean_max_abs_error")
    clean_phase0_atol = representation.get("phase0_match_atol")
    if (
        set(authority) != expected_authority_keys
        or authority.get("authority_kind") != DETERMINISTIC_VAE_AUTHORITY_KIND
        or authority.get("policy") != DETERMINISTIC_VAE_POLICY
        or type(authority.get("producer_rank")) is not int
        or authority.get("producer_rank") != 0
        or type(authority.get("encode_call_count")) is not int
        or authority.get("encode_call_count") != 2
        or authority.get("scope") != DETERMINISTIC_VAE_SCOPE
        or not isinstance(before, Mapping)
        or set(before) != flag_keys
        or any(type(value) is not bool for value in before.values())
        or not isinstance(during, Mapping)
        or set(during) != flag_keys
        or dict(during) != expected_during
        or not isinstance(restored, Mapping)
        or set(restored) != flag_keys
        or any(type(value) is not bool for value in restored.values())
        or dict(restored) != dict(before)
        or authority.get("flags_restored_exact") is not True
        or isinstance(posterior_error, bool)
        or not isinstance(posterior_error, (int, float))
        or not math.isfinite(float(posterior_error))
        or float(posterior_error) != 0.0
        or authority.get("posterior_phase0_bit_exact") is not True
        or type(action_phase0_sha) is not str
        or _SHA256.fullmatch(action_phase0_sha) is None
        or noop_phase0_sha != action_phase0_sha
        or authority.get("posterior_modified_after_encode") is not False
        or authority.get("posterior_copy_or_splice_used") is not False
        or authority.get("trainer_received_posterior") is not False
        or isinstance(clean_phase0_error, bool)
        or not isinstance(clean_phase0_error, (int, float))
        or not math.isfinite(float(clean_phase0_error))
        or float(clean_phase0_error) != 0.0
        or isinstance(clean_phase0_atol, bool)
        or not isinstance(clean_phase0_atol, (int, float))
        or not math.isfinite(float(clean_phase0_atol))
        or float(clean_phase0_atol) != 0.0
    ):
        raise G1MiddleControlError(
            f"{label} deterministic VAE authority differs"
        )


def _validate_upstream_receipt(receipt: Mapping[str, Any], *, label: str) -> None:
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    cache = receipt.get("cache")
    representation = receipt.get("representation")
    firewall = receipt.get("information_firewall")
    training = receipt.get("training_authority")
    projection = representation.get("projection") if isinstance(representation, Mapping) else None
    sigmas = representation.get("sigmas") if isinstance(representation, Mapping) else None
    role = receipt.get("anchor_source_role")
    is_selfgen = role in UPSTREAM_SELFGEN_ROLES
    if isinstance(representation, Mapping):
        _validate_explicit_gaussian_match(representation, label=label)
        _validate_deterministic_vae_authority(representation, label=label)
    else:
        raise G1MiddleControlError(f"{label} upstream representation is missing")
    if (
        receipt.get("schema_version") != UPSTREAM_RECEIPT_SCHEMA
        or receipt.get("method") != "bernini-decoded-middle-action-representation-v1"
        or receipt.get("complete") is not True
        or receipt.get("scientific_claim_authorized") is not False
        or receipt.get("representation_origin") != "decoded_video_reencode"
        or role not in UPSTREAM_CONTROL_ROLES
        or type(receipt.get("case_id")) is not str
        or _IDENTIFIER.fullmatch(receipt.get("case_id", "")) is None
        or type(receipt.get("input_video_sha256")) is not str
        or _SHA256.fullmatch(receipt.get("input_video_sha256", "")) is None
        or type(receipt.get("instruction_sha256")) is not str
        or _SHA256.fullmatch(receipt.get("instruction_sha256", "")) is None
        or type(receipt.get("method_source_sha256")) is not str
        or _SHA256.fullmatch(receipt.get("method_source_sha256", "")) is None
        or _sha256_bytes(_canonical_json_bytes(candidate)) != digest
        or not isinstance(cache, Mapping)
        or cache.get("filename") != "middle_repr.safetensors"
        or cache.get("schema_version") != UPSTREAM_CACHE_SCHEMA
        or cache.get("tensor_key_allowlist") != sorted(REQUIRED_TENSORS)
        or not isinstance(representation, Mapping)
        or representation.get("blocks") != list(BLOCK_INDICES)
        or representation.get("capture") != "post_transformer_block_output"
        or representation.get("contrast") != "decoded_action_minus_exact_first_frame_repeat"
        or representation.get("decoded_video_reencode") is not True
        or representation.get("selfgen_native_trajectory") is not False
        or representation.get("noop_constructed_inside_extractor") is not True
        or representation.get("first_frame_repeat_rgb_exact") is not True
        or representation.get("same_caption") is not True
        or representation.get("same_gaussian") is not True
        or representation.get("same_timestep") is not True
        or representation.get("same_rotary") is not True
        or not isinstance(sigmas, list)
        or not sigmas
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
            for value in (sigmas or [])
        )
        or not isinstance(projection, Mapping)
        or projection.get("kind") != "case_independent_fixed_rademacher_jl"
        or projection.get("fitted_on_input_video") is not False
        or isinstance(projection.get("width"), bool)
        or not isinstance(projection.get("width"), int)
        or projection.get("width", 0) <= 0
        or isinstance(projection.get("seed"), bool)
        or not isinstance(projection.get("seed"), int)
        or type(projection.get("sha256")) is not str
        or _SHA256.fullmatch(projection.get("sha256", "")) is None
        or not isinstance(firewall, Mapping)
        or firewall.get("input_video_accessed_by_frozen_extractor") is not True
        or firewall.get("target_video_accessed_by_extractor") is not (not is_selfgen)
        or firewall.get("target_rgb_or_vae_used_by_frozen_extractor") is not (not is_selfgen)
        or firewall.get("trainer_receives_detached_representation_cache_only") is not True
        or firewall.get("target_video_accessed_by_trainer") is not False
        or firewall.get("target_rgb_or_vae_target_used_by_trainer") is not False
        or firewall.get("anchor_role") != "detached_action_representation_only"
        or firewall.get("input_video_path_persisted") is not False
        or firewall.get("input_rgb_frames_persisted") is not False
        or firewall.get("input_vae_or_clean_latent_persisted") is not False
        or firewall.get("absolute_action_hidden_persisted") is not False
        or firewall.get("absolute_noop_hidden_persisted") is not False
        or firewall.get("raw_q_or_k_or_value_persisted") is not False
        or firewall.get("model_endpoint_or_velocity_persisted") is not False
        or firewall.get("self_generated_rgb_or_latent_copied_to_output") is not False
        or firewall.get("ephemeral_absolute_hidden_zero_reference_released_before_publication")
        is not True
        or not isinstance(training, Mapping)
        or training.get("optimizer_created") is not False
        or training.get("optimization_steps") != 0
        or training.get("generator_parameters_updated") is not False
        or training.get("cache_is_not_a_flow_matching_target") is not True
        or not isinstance(receipt.get("model_identity"), Mapping)
        or receipt.get("model_identity", {}).get("base_frozen") is not True
        or not isinstance(receipt.get("runtime_identity"), Mapping)
    ):
        raise G1MiddleControlError(f"{label} upstream receipt closure differs")


def _load_middle(
    cache_path: Path | str,
    receipt_path: Path | str,
    *,
    label: str,
) -> LoadedMiddle:
    cache = _regular_path(cache_path, label=f"{label} cache")
    if cache.suffix != ".safetensors":
        raise G1MiddleControlError(f"{label} cache must end in .safetensors")
    receipt_file, receipt, receipt_sha = _read_json(receipt_path, label=f"{label} receipt")
    _validate_upstream_receipt(receipt, label=label)
    cache_sha = _sha256_file(cache)
    cache_row = receipt["cache"]
    if cache_row.get("filename") != cache.name or cache_row.get("sha256") != cache_sha:
        raise G1MiddleControlError(f"{label} cache binding differs")
    try:
        from safetensors import safe_open
        with safe_open(str(cache), framework="pt", device="cpu") as handle:
            keys = tuple(handle.keys())
            metadata = dict(handle.metadata() or {})
            tensors = {key: handle.get_tensor(key).contiguous() for key in keys}
    except Exception as error:
        raise G1MiddleControlError(f"cannot load {label} middle cache") from error
    import torch
    if set(tensors) != set(REQUIRED_TENSORS):
        raise G1MiddleControlError(f"{label} middle tensor-key closure differs")
    reference_shape: tuple[int, ...] | None = None
    reference_dtype: Any = None
    tensor_receipts = cache_row.get("tensors")
    if not isinstance(tensor_receipts, Mapping) or set(tensor_receipts) != set(REQUIRED_TENSORS):
        raise G1MiddleControlError(f"{label} middle tensor receipts differ")
    for key in REQUIRED_TENSORS:
        tensor = tensors[key]
        if (
            tensor.ndim != 4
            or int(tensor.shape[1]) != PHASES
            or int(tensor.shape[0]) <= 0
            or int(tensor.shape[2]) <= 0
            or int(tensor.shape[3]) <= 0
            or tensor.dtype not in (torch.float16, torch.float32)
            or not bool(torch.isfinite(tensor).all().item())
            or bool(tensor[:, 0].any().item())
        ):
            raise G1MiddleControlError(f"{label}.{key} tensor contract differs")
        shape = tuple(map(int, tensor.shape))
        if reference_shape is None:
            reference_shape, reference_dtype = shape, tensor.dtype
        elif shape != reference_shape or tensor.dtype != reference_dtype:
            raise G1MiddleControlError(f"{label} middle block geometry differs")
        recorded = tensor_receipts[key]
        if (
            not isinstance(recorded, Mapping)
            or recorded.get("shape") != list(shape)
            or recorded.get("dtype") != str(tensor.dtype)
            or recorded.get("sha256") != _tensor_sha256(tensor)
            or recorded.get("detached") is not True
            or recorded.get("phase0_hard_zero") is not True
        ):
            raise G1MiddleControlError(f"{label}.{key} tensor receipt does not replay")
    required_metadata = {
        "schema_version": UPSTREAM_CACHE_SCHEMA,
        "method": "bernini-decoded-middle-action-representation-v1",
        "representation_origin": "decoded_video_reencode",
        "anchor_source_role": receipt["anchor_source_role"],
        "blocks": ",".join(map(str, BLOCK_INDICES)),
        "sigmas": ",".join(
            f"{float(value):.9g}" for value in receipt["representation"]["sigmas"]
        ),
        "projection_width": str(receipt["representation"]["projection"]["width"]),
        "contains_detached_projected_residuals_only": "true",
        "contains_rgb_latent_absolute_hidden_qkv_or_endpoint": "false",
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise G1MiddleControlError(f"{label} cache metadata differs")
    return LoadedMiddle(
        path=cache,
        sha256=cache_sha,
        receipt_path=receipt_file,
        receipt_sha256=receipt_sha,
        receipt=receipt,
        metadata=metadata,
        tensors=tensors,
    )


def _middle_ref(value: LoadedMiddle) -> dict[str, Any]:
    return {
        "path": str(value.path),
        "sha256": value.sha256,
        "receipt_path": str(value.receipt_path),
        "receipt_sha256": value.receipt_sha256,
        "case_id": value.receipt["case_id"],
        "anchor_source_role": value.receipt["anchor_source_role"],
        "instruction_sha256": value.receipt["instruction_sha256"],
        "tensor_shapes": {
            key: list(map(int, value.tensors[key].shape)) for key in REQUIRED_TENSORS
        },
    }


def _energy(tensors: Mapping[str, Any]) -> float:
    total = 0.0
    count = 0
    for key in REQUIRED_TENSORS:
        tensor = tensors[key].double()
        total += float(tensor.square().sum().item())
        count += tensor.numel()
    if count <= 0:
        raise G1MiddleControlError("middle residual energy has no values")
    value = math.sqrt(total / count)
    if not math.isfinite(value) or value <= ENERGY_EPSILON:
        raise G1MiddleControlError("middle residual energy must be finite and nonzero")
    return value


def _quantized_scaled_energy(tensors: Mapping[str, Any], scale: float) -> float:
    """Measure RMS after high-precision scaling and output-dtype quantization."""

    import torch

    total = 0.0
    count = 0
    for key in REQUIRED_TENSORS:
        tensor = tensors[key]
        quantized = (tensor.to(dtype=torch.float64) * scale).to(
            dtype=tensor.dtype
        )
        if not bool(torch.isfinite(quantized).all().item()):
            return math.inf
        total += float(quantized.double().square().sum().item())
        count += quantized.numel()
    if count <= 0:
        raise G1MiddleControlError("middle residual energy has no values")
    return math.sqrt(total / count)


def _quantized_scale(
    tensors: Mapping[str, Any], scale: float
) -> dict[str, Any]:
    """Scale in FP64, then immediately restore each published tensor dtype."""

    import torch

    return {
        key: (
            tensors[key].to(dtype=torch.float64)
            .mul(scale)
            .to(dtype=tensors[key].dtype)
            .contiguous()
        )
        for key in REQUIRED_TENSORS
    }


def _calibrate_quantized_energy_scale(
    tensors: Mapping[str, Any],
    *,
    target_energy: float,
    analytic_scale: float,
) -> tuple[float, float, float]:
    """Find one common scalar against the quantized, published-domain RMS.

    FP16 scalar multiplication can quantize the scalar before multiplication,
    while even a high-precision multiplication is quantized when its result is
    cast back to FP16.  Search the latter monotone energy function directly.
    The fixed iteration count and total candidate ordering make replay exact.
    """

    lower = max(1.0 / MAX_ENERGY_SCALE, analytic_scale / 2.0)
    upper = min(MAX_ENERGY_SCALE, analytic_scale * 2.0)
    lower_energy = _quantized_scaled_energy(tensors, lower)
    upper_energy = _quantized_scaled_energy(tensors, upper)
    if lower_energy > target_energy or upper_energy < target_energy:
        raise G1MiddleControlError(
            "wrong-action quantized energy calibration does not bracket target"
        )

    initial_energy = _quantized_scaled_energy(tensors, analytic_scale)
    candidates = [
        (lower, lower_energy),
        (upper, upper_energy),
        (analytic_scale, initial_energy),
    ]
    for _ in range(ENERGY_SCALE_CALIBRATION_ITERATIONS):
        midpoint = lower + (upper - lower) / 2.0
        midpoint_energy = _quantized_scaled_energy(tensors, midpoint)
        candidates.append((midpoint, midpoint_energy))
        if midpoint_energy < target_energy:
            lower = midpoint
        else:
            upper = midpoint

    finite_candidates = [
        (candidate_scale, candidate_energy)
        for candidate_scale, candidate_energy in candidates
        if math.isfinite(candidate_energy)
    ]
    if not finite_candidates:
        raise G1MiddleControlError(
            "wrong-action quantized energy calibration has no finite candidate"
        )
    final_scale, final_energy = min(
        finite_candidates,
        key=lambda candidate: (
            abs(candidate[1] - target_energy) / target_energy,
            abs(candidate[0] - analytic_scale),
            candidate[0],
        ),
    )
    return initial_energy, final_scale, final_energy


def _signature(value: LoadedMiddle) -> str:
    representation = value.receipt["representation"]
    payload = {
        "blocks": representation["blocks"],
        "sigmas": representation["sigmas"],
        "patch_grid": representation["patch_grid"],
        "projection": representation["projection"],
        "model_identity": value.receipt["model_identity"],
        "method_source_sha256": value.receipt["method_source_sha256"],
        "tensor_shapes": {
            key: list(map(int, value.tensors[key].shape)) for key in REQUIRED_TENSORS
        },
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _same_case_gaussian_signature(value: LoadedMiddle) -> str:
    """Bind temporal controls to one pre-pack Gaussian/noise-sigma authority."""

    authority = value.receipt["representation"]["gaussian_match"]["authority"]
    payload = {
        "authority_kind": authority["authority_kind"],
        "domain": authority["domain"],
        "base_seed": authority["base_seed"],
        "derived_seed": authority["derived_seed"],
        "dtype": authority["dtype"],
        "shape": authority["shape"],
        "canonical_gaussian_sha256": authority[
            "canonical_gaussian_sha256"
        ],
        "raw_noise_sigma_dtype": authority["raw_noise_sigma_dtype"],
        "raw_noise_sigma_shape": authority["raw_noise_sigma_shape"],
        "raw_noise_sigma_sha256": authority[
            "action_raw_noise_sigma_sha256"
        ],
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _build_transforms(
    correct: LoadedMiddle,
    wrong: LoadedMiddle,
    *,
    incomplete_action_phases: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import torch

    if not 1 <= incomplete_action_phases < PHASES - 1:
        raise G1MiddleControlError("incomplete action phases must lie in [1,19]")
    cutoff = 1 + incomplete_action_phases
    zero = {key: torch.zeros_like(correct.tensors[key]).contiguous() for key in REQUIRED_TENSORS}
    incomplete = {key: correct.tensors[key].clone().contiguous() for key in REQUIRED_TENSORS}
    removed: dict[str, int] = {}
    for key in REQUIRED_TENSORS:
        removed[key] = int(torch.count_nonzero(incomplete[key][:, cutoff:]).item())
        incomplete[key][:, cutoff:] = 0
    if sum(removed.values()) == 0:
        raise G1MiddleControlError("incomplete middle control removes no tail residual")
    correct_energy = _energy(correct.tensors)
    donor_energy = _energy(wrong.tensors)
    initial_scale = correct_energy / donor_energy
    if (
        not math.isfinite(initial_scale)
        or not 1.0 / MAX_ENERGY_SCALE
        <= initial_scale
        <= MAX_ENERGY_SCALE
    ):
        raise G1MiddleControlError("wrong-action middle energy scale is outside bound")
    initial_matched_energy, scale, calibrated_energy = (
        _calibrate_quantized_energy_scale(
            wrong.tensors,
            target_energy=correct_energy,
            analytic_scale=initial_scale,
        )
    )
    if not 1.0 / MAX_ENERGY_SCALE <= scale <= MAX_ENERGY_SCALE:
        raise G1MiddleControlError("wrong-action middle energy scale is outside bound")
    matched = _quantized_scale(wrong.tensors, scale)
    matched_energy = _energy(matched)
    if matched_energy != calibrated_energy:
        raise G1MiddleControlError(
            "wrong-action quantized energy calibration does not replay"
        )
    initial_relative_error = (
        abs(initial_matched_energy - correct_energy) / correct_energy
    )
    relative_error = abs(matched_energy - correct_energy) / correct_energy
    if relative_error > ENERGY_MATCH_RTOL:
        raise G1MiddleControlError("wrong-action middle energy matching failed")
    return (
        {
            "zero_or_noop": zero,
            "incomplete": incomplete,
            "wrong_action_energy_matched": matched,
        },
        {
            "energy_definition": ENERGY_DEFINITION,
            "correct_energy": correct_energy,
            "wrong_action_donor_energy_before_scale": donor_energy,
            "wrong_action_initial_scale": initial_scale,
            "wrong_action_initial_energy_after_quantization": (
                initial_matched_energy
            ),
            "wrong_action_initial_relative_energy_error": (
                initial_relative_error
            ),
            "wrong_action_scale": scale,
            "wrong_action_final_scale": scale,
            "wrong_action_energy_after_scale": matched_energy,
            "wrong_action_relative_energy_error": relative_error,
            "wrong_action_quantization_calibrated": True,
            "wrong_action_scale_calibration_iterations": (
                ENERGY_SCALE_CALIBRATION_ITERATIONS
            ),
            "wrong_action_scale_calibration_compute_dtype": (
                ENERGY_SCALE_CALIBRATION_COMPUTE_DTYPE
            ),
            "wrong_action_output_dtype": str(
                wrong.tensors[REQUIRED_TENSORS[0]].dtype
            ),
            "energy_match_rtol": ENERGY_MATCH_RTOL,
            "maximum_energy_scale": MAX_ENERGY_SCALE,
            "incomplete_action_phases_retained": incomplete_action_phases,
            "incomplete_tail_phases_zeroed": PHASES - cutoff,
            "incomplete_removed_nonzero_counts": removed,
        },
    )


def _save_bytes(
    tensors: Mapping[str, Any],
    metadata: Mapping[str, str],
    *,
    case_id: str,
    anchor_kind: str,
    control_kind: str,
    correct_sha256: str,
) -> bytes:
    from safetensors.torch import load as load_safetensors
    from safetensors.torch import save as save_safetensors
    reserved = {
        "bernini_g1_schema_version": COHORT_SCHEMA_VERSION,
        "bernini_g1_case_id": case_id,
        "bernini_g1_anchor_kind": anchor_kind,
        "bernini_g1_control_kind": control_kind,
        "bernini_g1_correct_sha256": correct_sha256,
    }
    if set(metadata).intersection(reserved):
        raise G1MiddleControlError("middle metadata collides with reserved G1 keys")
    output_metadata = dict(metadata)
    output_metadata.update(reserved)
    payload = save_safetensors(dict(tensors), metadata=output_metadata)
    replay = load_safetensors(payload)
    import torch
    if set(replay) != set(REQUIRED_TENSORS) or any(
        not torch.equal(replay[key], tensors[key]) for key in REQUIRED_TENSORS
    ):
        raise G1MiddleControlError("middle control safetensors round trip differs")
    return payload


def _write_create_only(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise G1MiddleControlError(f"refusing to overwrite {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def materialize_cohort(
    *,
    correct_cache: Path | str,
    correct_receipt: Path | str,
    temporal_shuffle_cache: Path | str,
    temporal_shuffle_receipt: Path | str,
    reverse_cache: Path | str,
    reverse_receipt: Path | str,
    wrong_action_cache: Path | str,
    wrong_action_receipt: Path | str,
    output_dir: Path | str,
    case_id: str,
    anchor_kind: str,
    action_family: str,
    wrong_case_id: str,
    wrong_action_family: str,
    incomplete_action_phases: int = 10,
) -> dict[str, Any]:
    case_id = _identifier(case_id, label="case_id")
    wrong_case_id = _identifier(wrong_case_id, label="wrong_case_id")
    action_family = _identifier(action_family, label="action_family")
    wrong_action_family = _identifier(wrong_action_family, label="wrong_action_family")
    if anchor_kind not in ANCHOR_KINDS:
        raise G1MiddleControlError(f"anchor_kind must be one of {ANCHOR_KINDS}")
    if case_id == wrong_case_id or action_family == wrong_action_family:
        raise G1MiddleControlError("wrong action must use a different case and action family")
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise G1MiddleControlError(f"refusing to overwrite output directory: {output}")
    external = {
        "correct": _load_middle(correct_cache, correct_receipt, label="correct"),
        "temporal_shuffle": _load_middle(
            temporal_shuffle_cache, temporal_shuffle_receipt, label="temporal_shuffle"
        ),
        "reverse": _load_middle(reverse_cache, reverse_receipt, label="reverse"),
        "wrong_action_donor": _load_middle(
            wrong_action_cache, wrong_action_receipt, label="wrong_action_donor"
        ),
    }
    expected_roles = EXTERNAL_ROLE_MAP_BY_ANCHOR[anchor_kind]
    for slot, value in external.items():
        expected_case = wrong_case_id if slot == "wrong_action_donor" else case_id
        if (
            value.receipt["case_id"] != expected_case
            or value.receipt["anchor_source_role"] != expected_roles[slot]
        ):
            raise G1MiddleControlError(f"{slot} case/anchor provenance differs")
    if len({value.sha256 for value in external.values()}) != len(EXTERNAL_ROLES):
        raise G1MiddleControlError("external middle caches alias by SHA-256")
    signatures = {_signature(value) for value in external.values()}
    if len(signatures) != 1:
        raise G1MiddleControlError("middle extractor/model/projection geometry differs")
    instruction = external["correct"].receipt["instruction_sha256"]
    if any(
        external[role].receipt["instruction_sha256"] != instruction
        for role in ("temporal_shuffle", "reverse")
    ):
        raise G1MiddleControlError("middle temporal controls are not instruction matched")
    same_case_authorities = {
        _same_case_gaussian_signature(external[role])
        for role in ("correct", "temporal_shuffle", "reverse")
    }
    if len(same_case_authorities) != 1:
        raise G1MiddleControlError(
            "same-case middle temporal controls do not share one explicit "
            "Gaussian authority"
        )
    if external["wrong_action_donor"].receipt["instruction_sha256"] == instruction:
        raise G1MiddleControlError("wrong-action middle donor instruction is not different")
    generated, diagnostics = _build_transforms(
        external["correct"],
        external["wrong_action_donor"],
        incomplete_action_phases=incomplete_action_phases,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{secrets.token_hex(6)}"
    staging.mkdir(mode=0o700)
    published = False
    try:
        generated_rows: dict[str, Any] = {}
        for control in GENERATED_CONTROLS:
            cache_path = staging / f"{control}.safetensors"
            sidecar_path = staging / f"{control}.json"
            payload = _save_bytes(
                generated[control],
                external["correct"].metadata,
                case_id=case_id,
                anchor_kind=anchor_kind,
                control_kind=control,
                correct_sha256=external["correct"].sha256,
            )
            sidecar = {
                "schema_version": BUNDLE_RECEIPT_SCHEMA_VERSION,
                "case_id": case_id,
                "anchor_kind": anchor_kind,
                "control_kind": control,
                "correct_sha256": external["correct"].sha256,
                "wrong_action_donor_sha256": (
                    external["wrong_action_donor"].sha256
                    if control == "wrong_action_energy_matched"
                    else None
                ),
                "tensor_shapes": {
                    key: list(map(int, generated[control][key].shape))
                    for key in REQUIRED_TENSORS
                },
                "transform_diagnostics": diagnostics,
                "contains_detached_projected_residuals_only": True,
                "target_rgb_vae_clean_latent_or_absolute_hidden_accessed": False,
                "optimizer_created": False,
                "current_experiment_optimization_steps": 0,
            }
            sidecar_payload = _canonical_json_bytes(sidecar, pretty=True)
            _write_create_only(cache_path, payload)
            _write_create_only(sidecar_path, sidecar_payload)
            generated_rows[control] = {
                "path": str(output / cache_path.name),
                "sha256": _sha256_bytes(payload),
                "sidecar_path": str(output / sidecar_path.name),
                "sidecar_sha256": _sha256_bytes(sidecar_payload),
            }
        receipt = {
            "schema_version": COHORT_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "case_id": case_id,
            "anchor_kind": anchor_kind,
            "correct_role": expected_roles["correct"],
            "action_family": action_family,
            "wrong_case_id": wrong_case_id,
            "wrong_action_family": wrong_action_family,
            "external_caches": {role: _middle_ref(external[role]) for role in EXTERNAL_ROLES},
            "generated_controls": generated_rows,
            "diagnostics": diagnostics,
            "contracts": {
                "abi_component": "delta_h_middle",
                "external_role_contract": dict(expected_roles),
                "required_controls": [
                    "zero_or_noop",
                    "temporal_shuffle",
                    "reverse",
                    "incomplete",
                    "wrong_action_energy_matched",
                ],
                "target_and_selfgen_judged_separately": True,
                "same_case_temporal_controls_share_explicit_gaussian_authority": True,
                "weighted_compensation_forbidden": True,
                "detached_projected_residual_only": True,
                "target_rgb_vae_clean_latent_or_absolute_hidden_accessed": False,
                "optimizer_created": False,
                "current_experiment_optimization_steps": 0,
                "atomic_directory_publication": True,
            },
        }
        _write_create_only(staging / "cohort_receipt.json", _canonical_json_bytes(receipt, pretty=True))
        if output.exists() or output.is_symlink():
            raise G1MiddleControlError("output appeared during publication")
        os.rename(staging, output)
        published = True
        final_receipt = output / "cohort_receipt.json"
        verify_cohort_receipt(final_receipt)
        return json.loads(final_receipt.read_text(encoding="ascii"))
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def verify_cohort_receipt(path: Path | str) -> dict[str, Any]:
    _, receipt, _ = _read_json(path, label="middle cohort receipt")
    required = {
        "schema_version",
        "created_at_utc",
        "case_id",
        "anchor_kind",
        "correct_role",
        "action_family",
        "wrong_case_id",
        "wrong_action_family",
        "external_caches",
        "generated_controls",
        "diagnostics",
        "contracts",
    }
    if set(receipt) != required or receipt["schema_version"] != COHORT_SCHEMA_VERSION:
        raise G1MiddleControlError("middle cohort receipt closure differs")
    case_id = _identifier(receipt["case_id"], label="case_id")
    wrong_case_id = _identifier(receipt["wrong_case_id"], label="wrong_case_id")
    anchor_kind = receipt["anchor_kind"]
    action_family = _identifier(receipt["action_family"], label="action_family")
    wrong_family = _identifier(receipt["wrong_action_family"], label="wrong_action_family")
    if (
        anchor_kind not in ANCHOR_KINDS
        or case_id == wrong_case_id
        or action_family == wrong_family
        or receipt["correct_role"]
        != EXTERNAL_ROLE_MAP_BY_ANCHOR.get(anchor_kind, {}).get("correct")
    ):
        raise G1MiddleControlError("middle cohort identity differs")
    expected_contracts = {
        "abi_component": "delta_h_middle",
        "external_role_contract": dict(EXTERNAL_ROLE_MAP_BY_ANCHOR[anchor_kind]),
        "required_controls": [
            "zero_or_noop",
            "temporal_shuffle",
            "reverse",
            "incomplete",
            "wrong_action_energy_matched",
        ],
        "target_and_selfgen_judged_separately": True,
        "same_case_temporal_controls_share_explicit_gaussian_authority": True,
        "weighted_compensation_forbidden": True,
        "detached_projected_residual_only": True,
        "target_rgb_vae_clean_latent_or_absolute_hidden_accessed": False,
        "optimizer_created": False,
        "current_experiment_optimization_steps": 0,
        "atomic_directory_publication": True,
    }
    if receipt["contracts"] != expected_contracts:
        raise G1MiddleControlError("middle cohort contracts differ")
    refs = receipt["external_caches"]
    if not isinstance(refs, Mapping) or set(refs) != set(EXTERNAL_ROLES):
        raise G1MiddleControlError("middle external cache closure differs")
    external: dict[str, LoadedMiddle] = {}
    ref_fields = {
        "path",
        "sha256",
        "receipt_path",
        "receipt_sha256",
        "case_id",
        "anchor_source_role",
        "instruction_sha256",
        "tensor_shapes",
    }
    for role in EXTERNAL_ROLES:
        ref = refs[role]
        if not isinstance(ref, Mapping) or set(ref) != ref_fields:
            raise G1MiddleControlError(f"{role} middle reference closure differs")
        value = _load_middle(ref["path"], ref["receipt_path"], label=role)
        if (
            value.sha256 != ref["sha256"]
            or value.receipt_sha256 != ref["receipt_sha256"]
            or value.receipt["case_id"] != ref["case_id"]
            or value.receipt["anchor_source_role"] != ref["anchor_source_role"]
            or value.receipt["instruction_sha256"] != ref["instruction_sha256"]
            or ref["tensor_shapes"]
            != {key: list(map(int, value.tensors[key].shape)) for key in REQUIRED_TENSORS}
        ):
            raise G1MiddleControlError(f"{role} middle reference does not replay")
        external[role] = value
    expected_roles = EXTERNAL_ROLE_MAP_BY_ANCHOR[anchor_kind]
    for slot, value in external.items():
        expected_case = wrong_case_id if slot == "wrong_action_donor" else case_id
        if (
            value.receipt["case_id"] != expected_case
            or value.receipt["anchor_source_role"] != expected_roles[slot]
        ):
            raise G1MiddleControlError(f"{slot} case/anchor provenance differs")
    if len({value.sha256 for value in external.values()}) != len(EXTERNAL_ROLES):
        raise G1MiddleControlError("external middle caches alias by SHA-256")
    if len({_signature(value) for value in external.values()}) != 1:
        raise G1MiddleControlError("middle extractor/model/projection geometry differs")
    instruction = external["correct"].receipt["instruction_sha256"]
    if any(
        external[slot].receipt["instruction_sha256"] != instruction
        for slot in ("temporal_shuffle", "reverse")
    ):
        raise G1MiddleControlError("middle temporal controls are not instruction matched")
    if len(
        {
            _same_case_gaussian_signature(external[slot])
            for slot in ("correct", "temporal_shuffle", "reverse")
        }
    ) != 1:
        raise G1MiddleControlError(
            "same-case middle temporal controls do not share one explicit "
            "Gaussian authority"
        )
    if external["wrong_action_donor"].receipt["instruction_sha256"] == instruction:
        raise G1MiddleControlError("wrong-action middle donor instruction is not different")
    keep = receipt["diagnostics"].get("incomplete_action_phases_retained")
    if isinstance(keep, bool) or not isinstance(keep, int):
        raise G1MiddleControlError("middle incomplete phase count differs")
    expected, diagnostics = _build_transforms(
        external["correct"],
        external["wrong_action_donor"],
        incomplete_action_phases=keep,
    )
    if _canonical_json_bytes(receipt["diagnostics"]) != _canonical_json_bytes(diagnostics):
        raise G1MiddleControlError("middle transform diagnostics do not replay")
    rows = receipt["generated_controls"]
    if not isinstance(rows, Mapping) or set(rows) != set(GENERATED_CONTROLS):
        raise G1MiddleControlError("middle generated control closure differs")
    import torch
    for control in GENERATED_CONTROLS:
        row = rows[control]
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "sidecar_path", "sidecar_sha256"}:
            raise G1MiddleControlError(f"{control} middle output reference differs")
        cache = _regular_path(row["path"], label=control)
        if _sha256_file(cache) != _sha(row["sha256"], label=f"{control}.sha256"):
            raise G1MiddleControlError(f"{control} middle output SHA differs")
        _, sidecar_value, sidecar_sha = _read_json(row["sidecar_path"], label=f"{control} sidecar")
        if sidecar_sha != row["sidecar_sha256"] or sidecar_value.get("control_kind") != control:
            raise G1MiddleControlError(f"{control} middle sidecar differs")
        from safetensors.torch import load_file
        tensors = load_file(str(cache), device="cpu")
        if set(tensors) != set(REQUIRED_TENSORS):
            raise G1MiddleControlError(f"{control} tensor closure differs")
        for key in REQUIRED_TENSORS:
            if not torch.equal(tensors[key], expected[control][key]):
                raise G1MiddleControlError(f"{control}.{key} transform does not replay")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize or verify a G1 Delta-H-middle control cohort.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    for role in ("correct", "temporal-shuffle", "reverse", "wrong-action"):
        materialize.add_argument(f"--{role}-cache", required=True)
        materialize.add_argument(f"--{role}-receipt", required=True)
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--case-id", required=True)
    materialize.add_argument("--anchor-kind", required=True, choices=ANCHOR_KINDS)
    materialize.add_argument("--action-family", required=True)
    materialize.add_argument("--wrong-case-id", required=True)
    materialize.add_argument("--wrong-action-family", required=True)
    materialize.add_argument("--incomplete-action-phases", type=int, default=10)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        receipt = materialize_cohort(
            correct_cache=args.correct_cache,
            correct_receipt=args.correct_receipt,
            temporal_shuffle_cache=args.temporal_shuffle_cache,
            temporal_shuffle_receipt=args.temporal_shuffle_receipt,
            reverse_cache=args.reverse_cache,
            reverse_receipt=args.reverse_receipt,
            wrong_action_cache=args.wrong_action_cache,
            wrong_action_receipt=args.wrong_action_receipt,
            output_dir=args.output_dir,
            case_id=args.case_id,
            anchor_kind=args.anchor_kind,
            action_family=args.action_family,
            wrong_case_id=args.wrong_case_id,
            wrong_action_family=args.wrong_action_family,
            incomplete_action_phases=args.incomplete_action_phases,
        )
    else:
        receipt = verify_cohort_receipt(args.receipt)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
