#!/usr/bin/env python3
"""Post-write audit for the preservation-v2 teacher cache and 20-step arms."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_generated_action_preservation_v2 as preservation
import train_self_generated_action_quotient_v1 as trainer


ARMS = tuple(preservation.ARM_NAMES)
CHECKPOINT_STEPS = (0, 5, 10, 20)
SOURCE_MANIFEST_DIGEST = "2fb367ed6f06275705e0b71020dd87fd68e13a010e80ef0bd2a122c94070f503"
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
TRANSFORMERS_VERSION = "5.5.4"
LORA_RANK = 8
LORA_WIDTH = 1536
EXPECTED_ADAPTER_CONFIG_SHA256_BY_SCOPE = {
    "all_attention": "d6d676d03c05175edda3dd0c9ed99787a749146b123ee4dfcc47ebd9d7fbc896",
    "cross_attn2_qo": "eb6fed8af2d1884782db4461760da6ccf356ec13e16bbb6487df41d8914fb139",
}


class PreservationAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreservationAuditError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain_single_file(path: Path, *, mode: int | None = None) -> Path:
    details = path.lstat()
    require(stat.S_ISREG(details.st_mode), f"not a regular file: {path}")
    require(
        not path.is_symlink()
        and details.st_nlink == 1
        and details.st_uid == os.getuid(),
        f"file topology differs: {path}",
    )
    if mode is not None:
        require(stat.S_IMODE(details.st_mode) == mode, f"file mode differs: {path}")
    return path


def stable_file_bytes(
    path: Path, *, mode: int | None = None,
) -> tuple[bytes, str]:
    plain_single_file(path, mode=mode)
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        first = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            first.extend(block)
        middle = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            second.extend(block)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_uid,
            value.st_gid,
            stat.S_IMODE(value.st_mode),
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        named = path.lstat()
        require(
            identity(before) == identity(middle) == identity(after) == identity(named),
            f"file changed during stable audit: {path}",
        )
        raw = bytes(first)
        require(
            raw == bytes(second) and len(raw) == before.st_size,
            f"file bytes changed during stable audit: {path}",
        )
        return raw, hashlib.sha256(raw).hexdigest()
    finally:
        os.close(descriptor)


def _unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(token: str) -> Any:
    raise PreservationAuditError(f"non-finite JSON constant: {token}")


def load_canonical_receipt(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, PreservationAuditError) as error:
        raise PreservationAuditError(f"{label} receipt is not strict JSON") from error
    require(isinstance(value, dict), f"{label} receipt JSON differs")
    require(
        trainer.canonical(value) + b"\n" == raw,
        f"{label} receipt bytes are not canonical",
    )
    return value


def validate_digest(value: Mapping[str, Any], *, label: str) -> None:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    require(
        isinstance(declared, str) and trainer.object_sha(unsigned) == declared,
        f"{label} receipt digest differs",
    )


def expected_targets(scope: str) -> list[str]:
    all_targets = [
        f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
        for block in range(30)
        for attention in (1, 2)
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    ]
    return preservation.select_projection_scope(all_targets, scope=scope)


def expected_training_contract(
    *, arm: str, transformers_version: str = TRANSFORMERS_VERSION,
) -> dict[str, Any]:
    spec = preservation.arm_spec(arm)
    return {
        "model": "Bernini-R-1.3B-Diffusers renderer-only",
        "single_expert": "transformer_1",
        "mv2v_flow_shift": 5.0,
        "num_frames": 81,
        "latent_frames": 21,
        "task_source_name": trainer.legacy.TASK_SOURCE_NAME,
        "external_spatial_mask": False,
        "external_tracking_or_swept_tube": False,
        "conditioning": ["clean_source_video_vae", "edit_instruction"],
        "target_embedding_or_caption_conditioning": False,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_RANK,
        "tokenizer_fix_mistral_regex": True,
        "transformers_version": transformers_version,
        "objective": trainer.METHOD_V2,
        "objective_family": "preservation_v2",
        "arm": arm,
        "weights": {
            "noop": spec.noop_weight,
            "onset": spec.onset_weight,
            "nuisance": spec.nuisance_weight,
            "functional": spec.functional_weight,
        },
        "onset_latent_phase_weights": list(preservation.ONSET_WEIGHTS),
        "functional_components": [
            "teacher_direction_exempt_post_head_orthogonal_drift",
            "post_onset_temporal_dc_clean_latent_drift",
        ],
        "lora_route_scope": spec.route_scope,
        "lora_route_scope_semantics": (
            "observable Wan attention topology only; no temporal-only or "
            "source-only route claim"
        ),
        "sigma_bins": [list(item) for item in trainer.V2_SIGMA_BINS],
        "checkpoint_updates": list(CHECKPOINT_STEPS),
        "rv2v_supervision_target": "source_video_only",
        "self_generated_anchor_role": "detached_post_head_action_phase_code_only",
        "historical_selected_target_reachable": False,
        "decoded_identity_background_camera_claim_authorized": False,
        "post_decode_gate_schema": "bernini-action-preservation-decision-v1",
        "blind_full_video_review_required_for_promotion": True,
    }


def validate_checkpoint_receipt(
    receipt: Mapping[str, Any], *, arm: str, step: int,
    cache_sha256: str, source_manifest_sha256: str,
    method_source_revision: str, method_source_archive_sha256: str,
    targets: Sequence[str],
) -> tuple[str, float, float]:
    expected_fields = {
        "schema_version", "global_step", "max_steps", "last_loss",
        "last_preclip_gradient_norm", "bernini_commit",
        "bernini_training_files_index_sha256", "veomni_commit",
        "method_source_revision", "method_source_archive_sha256",
        "initialization_seed", "teacher_cache_seed", "checkpoint_tree_sha256",
        "training_contract", "source_manifest_digest", "source_manifest_sha256",
        "teacher_cache_sha256", "optimizer", "distributed",
        "target_module_count", "target_modules_sha256",
        "trainable_parameter_count", "production_claim_forbidden",
        "scientific_claim_authorized", "experimental_training",
        "objective_family", "last_loss_components", "target_modules",
        "decoded_preservation_evidence_present",
        "automatic_scientific_promotion_authorized", "receipt_digest",
    }
    require(set(receipt) == expected_fields, f"checkpoint receipt field closure differs: {arm}@{step}")
    require(receipt["schema_version"] == trainer.RECEIPT_SCHEMA_V2, "checkpoint schema differs")
    require(receipt["global_step"] == step and receipt["max_steps"] == 20, "checkpoint step differs")
    require(receipt["bernini_commit"] == BERNINI_COMMIT, "checkpoint Bernini revision differs")
    require(receipt["veomni_commit"] == VEOMNI_COMMIT, "checkpoint VeOmni revision differs")
    require(
        receipt["bernini_training_files_index_sha256"]
        == trainer.legacy.object_sha256(trainer.legacy.BERNINI_PINNED_FILE_HASHES),
        "checkpoint Bernini training index differs",
    )
    require(receipt["checkpoint_tree_sha256"] == trainer.legacy.CHECKPOINT_TREE_SHA256, "checkpoint model tree differs")
    require(receipt["initialization_seed"] == receipt["teacher_cache_seed"] == trainer.V2_CANARY_SEED, "checkpoint seed differs")
    require(receipt["teacher_cache_sha256"] == cache_sha256, "checkpoint cache differs")
    require(receipt["source_manifest_sha256"] == source_manifest_sha256, "checkpoint manifest differs")
    require(receipt["source_manifest_digest"] == SOURCE_MANIFEST_DIGEST, "checkpoint manifest digest differs")
    require(receipt["method_source_revision"] == method_source_revision, "checkpoint revision differs")
    require(receipt["method_source_archive_sha256"] == method_source_archive_sha256, "checkpoint archive differs")
    require(receipt["objective_family"] == "preservation_v2", "checkpoint objective differs")
    require(receipt["target_modules"] == list(targets), "checkpoint target module list differs")
    require(receipt["target_module_count"] == len(targets), "checkpoint target count differs")
    require(receipt["target_modules_sha256"] == trainer.legacy.object_sha256(list(targets)), "checkpoint target digest differs")
    require(
        receipt["trainable_parameter_count"] == len(targets) * 2 * LORA_RANK * LORA_WIDTH,
        "checkpoint trainable parameter count differs",
    )
    require(
        receipt["training_contract"] == expected_training_contract(arm=arm),
        "checkpoint training contract differs",
    )
    require(
        receipt["optimizer"] == {
            "type": "AdamW",
            "learning_rate": preservation.arm_spec(arm).learning_rate,
            "weight_decay": trainer.V2_WEIGHT_DECAY,
        },
        "checkpoint optimizer contract differs",
    )
    distributed = receipt["distributed"]
    require(isinstance(distributed, dict), "checkpoint distributed contract differs")
    require(
        distributed == {
            "world_size": 4,
            "ulysses_size": 4,
            "backend": "nccl/rccl",
            "same_sample_all_ranks": True,
            "same_seed_all_ranks": True,
            "explicit_lora_gradient_all_reduce": True,
            "lora_initialization_digest": distributed.get("lora_initialization_digest"),
        },
        "checkpoint distributed contract differs",
    )
    initial_digest = distributed["lora_initialization_digest"]
    require(
        isinstance(initial_digest, str)
        and len(initial_digest) == 64
        and all(character in "0123456789abcdef" for character in initial_digest),
        "checkpoint initialization digest differs",
    )
    require(receipt["production_claim_forbidden"] is True, "checkpoint production gate differs")
    require(receipt["scientific_claim_authorized"] is False, "checkpoint scientific overclaim")
    require(receipt["experimental_training"] is True, "checkpoint experimental gate differs")
    require(receipt["decoded_preservation_evidence_present"] is False, "checkpoint decoded evidence overclaim")
    require(receipt["automatic_scientific_promotion_authorized"] is False, "checkpoint auto-promotes")
    components = receipt["last_loss_components"]
    expected_components = {
        "action", "onset", "nuisance", "noop", "functional_code",
        "functional_temporal_dc", "functional_total",
    }
    require(isinstance(components, dict) and set(components) == expected_components, "checkpoint loss component closure differs")
    require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in components.values()
        ),
        "checkpoint loss component is invalid",
    )
    require(
        math.isclose(
            float(components["functional_total"]),
            float(components["functional_code"])
            + float(components["functional_temporal_dc"]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ),
        "checkpoint functional sum differs",
    )
    spec = preservation.arm_spec(arm)
    reconstructed = (
        float(components["action"])
        + spec.onset_weight * float(components["onset"])
        + spec.nuisance_weight * float(components["nuisance"])
        + spec.noop_weight * float(components["noop"])
        + spec.functional_weight * float(components["functional_total"])
    )
    loss = receipt["last_loss"]
    gradient = receipt["last_preclip_gradient_norm"]
    require(
        isinstance(loss, (int, float))
        and not isinstance(loss, bool)
        and math.isfinite(float(loss)),
        "checkpoint loss is non-finite",
    )
    require(
        isinstance(gradient, (int, float))
        and not isinstance(gradient, bool)
        and math.isfinite(float(gradient)),
        "checkpoint gradient is non-finite",
    )
    if step == 0:
        require(float(loss) == float(gradient) == reconstructed == 0.0, "checkpoint zero is not pre-update")
        require(all(float(value) == 0.0 for value in components.values()), "checkpoint zero components differ")
    else:
        require(float(loss) > 0.0 and float(gradient) > 0.0, "trained checkpoint lacks positive update evidence")
        require(
            math.isclose(float(loss), reconstructed, rel_tol=1.0e-5, abs_tol=1.0e-5),
            "checkpoint loss does not match preregistered objective",
        )
    return initial_digest, float(loss), float(gradient)


def expected_adapter_tensor_keys(targets: Sequence[str]) -> list[str]:
    return sorted(
        f"base_model.model.{target}.lora_{side}.weight"
        for target in targets
        for side in ("A", "B")
    )


def validate_adapter_artifacts(
    *, adapter: Path, targets: Sequence[str], route_scope: str, step: int,
) -> dict[str, Any]:
    require(
        adapter.is_dir()
        and not adapter.is_symlink()
        and adapter.lstat().st_uid == os.getuid()
        and stat.S_IMODE(adapter.lstat().st_mode) == 0o555,
        "checkpoint adapter directory differs",
    )
    require(
        {path.name for path in adapter.iterdir()}
        == trainer.CHECKPOINT_ADAPTER_ENTRY_NAMES,
        "checkpoint adapter entry closure differs",
    )
    import torch
    from safetensors.torch import load as load_safetensors

    config_path = adapter / "adapter_config.json"
    model_path = adapter / "adapter_model.safetensors"
    config_raw, config_sha256 = stable_file_bytes(config_path, mode=0o444)
    model_raw, model_sha256 = stable_file_bytes(model_path, mode=0o444)
    require(config_raw and model_raw, "checkpoint adapter file is empty")
    config = json.loads(config_raw)
    require(isinstance(config, dict), "adapter config JSON differs")
    expected_config_fields = {
        "alora_invocation_tokens", "alpha_pattern", "arrow_config", "auto_mapping",
        "base_model_name_or_path", "bias", "corda_config", "ensure_weight_tying",
        "eva_config", "exclude_modules", "fan_in_fan_out", "inference_mode",
        "init_lora_weights", "layer_replication", "layers_pattern",
        "layers_to_transform", "loftq_config", "lora_alpha", "lora_bias",
        "lora_dropout", "lora_ga_config", "megatron_config", "megatron_core",
        "modules_to_save", "peft_type", "peft_version", "qalora_group_size", "r",
        "rank_pattern", "revision", "target_modules", "target_parameters",
        "task_type", "trainable_token_indices", "use_bdlora", "use_dora",
        "use_qalora", "use_rslora",
    }
    require(set(config) == expected_config_fields, "adapter config field closure differs")
    config_without_targets = dict(config)
    config_without_targets.pop("target_modules")
    expected_config_without_targets = {
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
        "lora_alpha": LORA_RANK,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "lora_ga_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": "0.19.1",
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
    require(
        trainer.canonical(config_without_targets)
        == trainer.canonical(expected_config_without_targets),
        "adapter config semantic closure differs",
    )
    expected_serialized_targets = list(
        trainer.expected_peft_serialized_target_modules(route_scope)
    )
    require(
        config["target_modules"] == expected_serialized_targets,
        "adapter serialized target scope/order differs",
    )
    expected_config = dict(expected_config_without_targets)
    expected_config["target_modules"] = expected_serialized_targets
    expected_config_raw = trainer.canonical(expected_config) + b"\n"
    expected_config_sha256 = hashlib.sha256(expected_config_raw).hexdigest()
    require(
        expected_config_sha256
        == EXPECTED_ADAPTER_CONFIG_SHA256_BY_SCOPE.get(route_scope),
        "adapter config exact scope SHA-256 authority differs",
    )
    require(
        config_raw == expected_config_raw,
        "adapter config canonical byte closure differs",
    )
    require(
        config_sha256 == EXPECTED_ADAPTER_CONFIG_SHA256_BY_SCOPE[route_scope],
        "adapter config exact scope SHA-256 differs",
    )
    expected_keys = expected_adapter_tensor_keys(targets)
    nonzero_a = nonzero_b = False
    tensor_shapes: list[tuple[int, ...]] = []
    require(len(model_raw) >= 8, "adapter safetensors header is absent")
    header_size = int.from_bytes(model_raw[:8], "little")
    require(0 < header_size <= len(model_raw) - 8, "adapter safetensors header size differs")
    header = json.loads(model_raw[8 : 8 + header_size])
    require(isinstance(header, dict), "adapter safetensors header differs")
    metadata = header.pop("__metadata__", None)
    require(metadata == {"format": "pt"}, "adapter safetensors metadata differs")
    require(sorted(header) == expected_keys, "adapter safetensors header key closure differs")
    tensors = load_safetensors(model_raw)
    keys = sorted(tensors)
    require(keys == expected_keys, "adapter tensor key closure differs")
    for key in keys:
        tensor = tensors[key]
        expected_shape = (
            (LORA_RANK, LORA_WIDTH)
            if key.endswith(".lora_A.weight")
            else (LORA_WIDTH, LORA_RANK)
        )
        require(
            tensor.dtype == torch.float32
            and tuple(int(item) for item in tensor.shape) == expected_shape
            and bool(torch.isfinite(tensor).all().item()),
            f"adapter tensor differs: {key}",
        )
        is_nonzero = bool(torch.count_nonzero(tensor).item())
        if key.endswith(".lora_A.weight"):
            nonzero_a = nonzero_a or is_nonzero
        else:
            nonzero_b = nonzero_b or is_nonzero
        tensor_shapes.append(expected_shape)
    require(nonzero_a, "adapter LoRA-A initialization is zero")
    if step == 0:
        require(not nonzero_b, "checkpoint zero LoRA-B is not zero initialized")
    else:
        require(nonzero_b, "trained checkpoint LoRA-B did not update")
    return {
        "config_sha256": config_sha256,
        "adapter_sha256": model_sha256,
        "tensor_count": len(expected_keys),
        "tensor_shapes": sorted(tensor_shapes),
    }


def validate_optimizer_artifact(
    *, path: Path, step: int, arm: str, tensor_count: int,
    tensor_shapes: Sequence[tuple[int, ...]],
) -> str:
    import torch

    optimizer_raw, optimizer_sha256 = stable_file_bytes(path, mode=0o444)
    require(optimizer_raw, "checkpoint optimizer is empty")
    try:
        payload = torch.load(
            io.BytesIO(optimizer_raw), map_location="cpu", weights_only=True
        )
    except TypeError as error:
        raise PreservationAuditError(
            "checkpoint audit requires torch.load(weights_only=True)"
        ) from error
    require(isinstance(payload, dict) and set(payload) == {"global_step", "optimizer"}, "optimizer payload closure differs")
    require(payload["global_step"] == step, "optimizer global step differs")
    optimizer = payload["optimizer"]
    require(isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"}, "optimizer state closure differs")
    groups = optimizer["param_groups"]
    require(isinstance(groups, list) and len(groups) == 1 and isinstance(groups[0], dict), "optimizer parameter groups differ")
    group = groups[0]
    expected_group_fields = {
        "lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
        "foreach", "capturable", "differentiable", "fused",
        "decoupled_weight_decay", "params",
    }
    require(set(group) == expected_group_fields, "optimizer parameter group field closure differs")
    require(float(group["lr"]) == preservation.arm_spec(arm).learning_rate, "optimizer learning rate differs")
    require(tuple(group["betas"]) == (0.9, 0.999) and float(group["eps"]) == 1.0e-8, "optimizer moments differ")
    require(float(group["weight_decay"]) == trainer.V2_WEIGHT_DECAY, "optimizer weight decay differs")
    require(
        group["amsgrad"] is False
        and group["maximize"] is False
        and group["foreach"] is None
        and group["capturable"] is False
        and group["differentiable"] is False
        and group["fused"] is None
        and group["decoupled_weight_decay"] is True,
        "optimizer AdamW options differ",
    )
    params = group["params"]
    require(
        isinstance(params, list)
        and params == list(range(tensor_count)),
        "optimizer parameter identity/order differs",
    )
    state = optimizer["state"]
    require(isinstance(state, dict), "optimizer tensor state differs")
    if step == 0:
        require(state == {}, "checkpoint zero optimizer is not pristine")
    else:
        require(set(state) == set(params), "trained optimizer state coverage differs")
        observed_shapes: list[tuple[int, ...]] = []
        any_first_moment = any_second_moment = False
        for parameter_id in params:
            row = state[parameter_id]
            require(isinstance(row, dict) and set(row) == {"step", "exp_avg", "exp_avg_sq"}, "optimizer tensor row closure differs")
            counter, first, second = row["step"], row["exp_avg"], row["exp_avg_sq"]
            require(
                isinstance(counter, torch.Tensor)
                and counter.ndim == 0
                and counter.dtype == torch.float32
                and bool(torch.isfinite(counter).item())
                and float(counter.item()) == float(step),
                "optimizer step tensor differs",
            )
            require(
                isinstance(first, torch.Tensor)
                and isinstance(second, torch.Tensor)
                and first.dtype == second.dtype == torch.float32
                and first.shape == second.shape
                and bool(torch.isfinite(first).all().item())
                and bool(torch.isfinite(second).all().item()),
                "optimizer moment tensor differs",
            )
            require(
                bool((second >= 0.0).all().item())
                and bool(torch.count_nonzero(first).item())
                and bool(torch.count_nonzero(second).item()),
                "optimizer moment values differ",
            )
            observed_shapes.append(tuple(int(item) for item in first.shape))
            any_first_moment = any_first_moment or bool(torch.count_nonzero(first).item())
            any_second_moment = any_second_moment or bool(torch.count_nonzero(second).item())
        require(sorted(observed_shapes) == sorted(tensor_shapes), "optimizer moment shape coverage differs")
        require(any_first_moment and any_second_moment, "trained optimizer moments are all zero")
    return optimizer_sha256


def validate_cache(
    *, cache_path: Path, expected_sha256: str, source_manifest: Path,
    source_manifest_sha256: str, method_source_revision: str,
    method_source_archive_sha256: str,
) -> dict[str, Any]:
    import torch

    cache_raw, cache_observed_sha256 = stable_file_bytes(
        cache_path, mode=0o444
    )
    require(cache_observed_sha256 == expected_sha256, "teacher cache SHA differs")
    manifest, rows = trainer.load_manifest(
        source_manifest, source_manifest_sha256
    )
    cache = torch.load(
        io.BytesIO(cache_raw), map_location="cpu", weights_only=True
    )
    expected_cache_fields = {
        "schema_version", "objective_family", "manifest_digest",
        "source_manifest_sha256", "method_source_revision",
        "method_source_archive_sha256", "slots", "sigma_bins", "seed",
        "initialization_seed", "teacher_cache_seed", "cells", "teacher_graph",
        "frozen_source_action_velocity", "anchor_role",
        "decoded_identity_background_camera_claim_authorized",
    }
    require(isinstance(cache, dict) and set(cache) == expected_cache_fields, "cache field closure differs")
    require(cache["schema_version"] == trainer.CACHE_SCHEMA_V2, "cache schema differs")
    require(cache["objective_family"] == "preservation_v2", "cache objective differs")
    require(cache["manifest_digest"] == manifest["manifest_digest"], "cache manifest digest differs")
    require(cache["source_manifest_sha256"] == source_manifest_sha256, "cache manifest SHA differs")
    require(cache["method_source_revision"] == method_source_revision, "cache revision differs")
    require(cache["method_source_archive_sha256"] == method_source_archive_sha256, "cache archive differs")
    require(
        cache["seed"] == cache["initialization_seed"] == cache["teacher_cache_seed"] == trainer.V2_CANARY_SEED,
        "cache seed closure differs",
    )
    require(cache["teacher_graph"] == "detached", "cache teacher graph differs")
    require(
        cache["frozen_source_action_velocity"]
        == "cpu_float32_sha256_bound",
        "cache frozen velocity authority differs",
    )
    require(cache["anchor_role"] == "action_phase_representation_only", "cache anchor role differs")
    require(cache["decoded_identity_background_camera_claim_authorized"] is False, "cache overclaims decoded preservation")
    cells, by_key = trainer.validate_teacher_cache_cells_v2(
        cache, expected_seed=trainer.V2_CANARY_SEED
    )
    for (row_index, _), cell in by_key.items():
        require(cell["iid"] == rows[row_index]["iid"], "cache IID binding differs")

    receipt_path = cache_path.with_suffix(cache_path.suffix + ".receipt.json")
    receipt_raw, receipt_sha256 = stable_file_bytes(receipt_path, mode=0o444)
    receipt = load_canonical_receipt(receipt_raw, label="cache")
    validate_digest(receipt, label="cache")
    expected_receipt_fields = {
        "schema_version", "objective_family", "cell_count", "manifest_digest",
        "cache_sha256", "source_manifest_sha256", "seed", "initialization_seed",
        "teacher_cache_seed", "method_source_revision",
        "method_source_archive_sha256", "historical_selected_target_reachable",
        "sigma_bins", "frozen_source_action_velocity",
        "decoded_identity_background_camera_claim_authorized", "receipt_digest",
    }
    require(set(receipt) == expected_receipt_fields, "cache receipt field closure differs")
    require(receipt["schema_version"] == trainer.CACHE_SCHEMA_V2, "cache receipt schema differs")
    require(receipt["objective_family"] == "preservation_v2", "cache receipt objective differs")
    require(receipt["cell_count"] == 20, "cache receipt count differs")
    require(receipt["cache_sha256"] == expected_sha256, "cache receipt SHA differs")
    require(receipt["manifest_digest"] == manifest["manifest_digest"], "cache receipt manifest differs")
    require(receipt["source_manifest_sha256"] == source_manifest_sha256, "cache receipt manifest SHA differs")
    require(receipt["method_source_revision"] == method_source_revision, "cache receipt revision differs")
    require(receipt["method_source_archive_sha256"] == method_source_archive_sha256, "cache receipt archive differs")
    require(receipt["seed"] == receipt["initialization_seed"] == receipt["teacher_cache_seed"] == trainer.V2_CANARY_SEED, "cache receipt seed differs")
    require(receipt["historical_selected_target_reachable"] is False, "cache receipt target reachability differs")
    require(
        receipt["sigma_bins"]
        == [list(item) for item in trainer.V2_SIGMA_BINS],
        "cache receipt sigma bins differ",
    )
    require(
        receipt["frozen_source_action_velocity"]
        == "cpu_float32_sha256_bound",
        "cache receipt frozen velocity authority differs",
    )
    require(receipt["decoded_identity_background_camera_claim_authorized"] is False, "cache receipt overclaims preservation")
    return {
        "cache_audit_go": True,
        "cache_sha256": expected_sha256,
        "cache_receipt_sha256": receipt_sha256,
        "cell_count": len(cells),
        "iid_count": len(rows),
        "sigma_bin_count": len(trainer.V2_SIGMA_BINS),
    }


def validate_training(
    *, experiment_root: Path, cache_sha256: str, source_manifest_sha256: str,
    method_source_revision: str, method_source_archive_sha256: str,
) -> dict[str, Any]:
    runs_root = experiment_root / "runs"
    runs_details = runs_root.lstat()
    require(
        runs_root.is_dir()
        and not runs_root.is_symlink()
        and runs_details.st_uid == os.getuid()
        and stat.S_IMODE(runs_details.st_mode) == 0o555,
        "runs root differs",
    )
    require({path.name for path in runs_root.iterdir()} == set(ARMS), "arm root closure differs")
    initialization_by_scope: dict[str, set[str]] = {}
    initial_adapter_by_scope: dict[str, set[str]] = {}
    config_by_scope: dict[str, set[str]] = {}
    receipt_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_root = runs_root / arm
        arm_details = arm_root.lstat()
        require(
            arm_root.is_dir()
            and not arm_root.is_symlink()
            and arm_details.st_uid == os.getuid()
            and stat.S_IMODE(arm_details.st_mode) == 0o555,
            f"arm root differs: {arm}",
        )
        expected_names = {f"checkpoint-{step:08d}" for step in CHECKPOINT_STEPS}
        require({path.name for path in arm_root.iterdir()} == expected_names, f"checkpoint closure differs: {arm}")
        spec = preservation.arm_spec(arm)
        targets = expected_targets(spec.route_scope)
        initialization_by_scope.setdefault(spec.route_scope, set())
        initial_adapter_by_scope.setdefault(spec.route_scope, set())
        config_by_scope.setdefault(spec.route_scope, set())
        adapter_hashes: list[str] = []
        optimizer_hashes: list[str] = []
        for step in CHECKPOINT_STEPS:
            checkpoint = arm_root / f"checkpoint-{step:08d}"
            require(
                checkpoint.is_dir()
                and not checkpoint.is_symlink()
                and checkpoint.lstat().st_uid == os.getuid()
                and stat.S_IMODE(checkpoint.lstat().st_mode) == 0o555,
                "checkpoint directory differs",
            )
            require(
                {path.name for path in checkpoint.iterdir()}
                == trainer.CHECKPOINT_ENTRY_NAMES,
                "checkpoint entry closure differs",
            )
            receipt_path = checkpoint / "receipt.json"
            receipt_raw, receipt_sha256 = stable_file_bytes(
                receipt_path, mode=0o444
            )
            receipt = load_canonical_receipt(
                receipt_raw, label=f"checkpoint {arm}@{step}"
            )
            validate_digest(receipt, label=f"{arm}@{step}")
            digest, loss, gradient = validate_checkpoint_receipt(
                receipt,
                arm=arm,
                step=step,
                cache_sha256=cache_sha256,
                source_manifest_sha256=source_manifest_sha256,
                method_source_revision=method_source_revision,
                method_source_archive_sha256=method_source_archive_sha256,
                targets=targets,
            )
            initialization_by_scope[spec.route_scope].add(digest)
            adapter_evidence = validate_adapter_artifacts(
                adapter=checkpoint / "adapter",
                targets=targets,
                route_scope=spec.route_scope,
                step=step,
            )
            config_by_scope[spec.route_scope].add(adapter_evidence["config_sha256"])
            adapter_hashes.append(adapter_evidence["adapter_sha256"])
            if step == 0:
                initial_adapter_by_scope[spec.route_scope].add(
                    adapter_evidence["adapter_sha256"]
                )
            optimizer_sha = validate_optimizer_artifact(
                path=plain_single_file(
                    checkpoint / "optimizer.pt", mode=0o444
                ),
                step=step,
                arm=arm,
                tensor_count=adapter_evidence["tensor_count"],
                tensor_shapes=adapter_evidence["tensor_shapes"],
            )
            optimizer_hashes.append(optimizer_sha)
            receipt_rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "receipt_sha256": receipt_sha256,
                    "adapter_sha256": adapter_evidence["adapter_sha256"],
                    "adapter_config_sha256": adapter_evidence["config_sha256"],
                    "optimizer_sha256": optimizer_sha,
                    "loss": loss,
                    "preclip_gradient_norm": gradient,
                }
            )
        require(len(set(adapter_hashes)) == len(CHECKPOINT_STEPS), f"adapter did not change at every checkpoint: {arm}")
        require(len(set(optimizer_hashes)) == len(CHECKPOINT_STEPS), f"optimizer did not change at every checkpoint: {arm}")
    require(all(len(values) == 1 for values in initialization_by_scope.values()), "initialization digest differs within route scope")
    require(all(len(values) == 1 for values in initial_adapter_by_scope.values()), "checkpoint-zero adapter differs within route scope")
    require(all(len(values) == 1 for values in config_by_scope.values()), "adapter config differs within route scope")
    return {
        "training_audit_go": True,
        "arm_count": len(ARMS),
        "checkpoint_count": len(receipt_rows),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "route_scopes": sorted(initialization_by_scope),
        "initialization_digest_by_scope": {
            scope: next(iter(values)) for scope, values in sorted(initialization_by_scope.items())
        },
        "checkpoint_zero_adapter_sha256_by_scope": {
            scope: next(iter(values))
            for scope, values in sorted(initial_adapter_by_scope.items())
        },
        "adapter_config_sha256_by_scope": {
            scope: next(iter(values))
            for scope, values in sorted(config_by_scope.items())
        },
        "receipt_rows": receipt_rows,
        "decoded_evaluation_complete": False,
        "scientific_promotion_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("validate-cache")
    cache.add_argument("--cache", required=True)
    cache.add_argument("--expected-cache-sha256", required=True)
    cache.add_argument("--source-manifest", required=True)
    cache.add_argument("--source-manifest-sha256", required=True)
    cache.add_argument("--method-source-revision", required=True)
    cache.add_argument("--method-source-archive-sha256", required=True)
    training = sub.add_parser("validate-training")
    training.add_argument("--experiment-root", required=True)
    training.add_argument("--cache-sha256", required=True)
    training.add_argument("--source-manifest-sha256", required=True)
    training.add_argument("--method-source-revision", required=True)
    training.add_argument("--method-source-archive-sha256", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate-cache":
        result = validate_cache(
            cache_path=Path(args.cache).resolve(strict=True),
            expected_sha256=args.expected_cache_sha256,
            source_manifest=Path(args.source_manifest).resolve(strict=True),
            source_manifest_sha256=args.source_manifest_sha256,
            method_source_revision=args.method_source_revision,
            method_source_archive_sha256=args.method_source_archive_sha256,
        )
    else:
        result = validate_training(
            experiment_root=Path(args.experiment_root).resolve(strict=True),
            cache_sha256=args.cache_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
            method_source_revision=args.method_source_revision,
            method_source_archive_sha256=args.method_source_archive_sha256,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
