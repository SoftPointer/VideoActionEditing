#!/usr/bin/env python3
"""Execute exactly one endpoint-qualified PAIR-v5 WORLD8 flow-DPO update.

This executable consumes only ``bernini-pair-v5-one-update-manifest-v3``.  It
does not import the v1/v2 calibrator, selector, manifest, or trainer.  All
evidence is replayed before model construction.  Two independent data-parallel
arms each use one qualified source while every arm is Ulysses-SP4.

The reference is the exact frozen parent policy that generated the rollouts:
base Bernini for round zero, or the previous Action-LoRA for later rounds.  An
optional CIO declared by that same parent policy is loaded into both student
and reference.  One or more microbatches may accumulate gradients, but there
is exactly one optimizer construction and exactly one ``optimizer.step()``.
The output cannot be reused as another step on the same rollout generation;
the next update requires a fresh next-round rollout manifest.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_endpoint_one_update_v3 as endpoint_contract  # noqa: E402

# Heavy Torch/Bernini modules are loaded only after the evidence-only preflight.
action_adapter: Any = None
flow_dpo: Any = None
native_bridge: Any = None
native: Any = None
guidance: Any = None
cio_adapter: Any = None
runtime: Any = None
legacy: Any = None


METHOD_NAME = "bernini-pair-v5-endpoint-qualified-one-update-v3"
RUN_RECEIPT_SCHEMA = "bernini-pair-v5-endpoint-one-update-receipt-v3"
HISTORY_SCHEMA = "bernini-pair-v5-endpoint-one-update-history-v3"
ADAPTER_CHECKPOINT_SCHEMA = "bernini-pair-v5-action-lora-checkpoint-v3"
CIO_ADAPTER_CHECKPOINT_SCHEMA = "bernini-native-target-row-qo-lora-checkpoint-v2"

WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
FRAME_COUNT = 81
FPS = 25.0
LATENT_CHANNELS = 16
LATENT_PHASES = 21
REFERENCE_INDICES = (0, 27, 53, 80)
ACTION_SIGMA_INDICES = tuple(range(38))
DEFAULT_SIGMA_INDEX = 20
DEFAULT_SEED = 20260808
DEFAULT_LEARNING_RATE = 1.0e-6
DEFAULT_BETA = 1000.0
DEFAULT_MAX_GRAD_NORM = 1.0
VJP_REPLAY_RTOL = 2.0e-5
VJP_REPLAY_ATOL = 2.0e-5
BERNINI_OFFICIAL_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_TESTED_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"


class PairV5EndpointOneUpdateError(RuntimeError):
    """A preflight, parent-policy, or exact-one-update invariant failed."""


def _load_runtime_modules() -> None:
    global action_adapter, flow_dpo, native_bridge, native, guidance
    global cio_adapter, runtime, legacy
    if runtime is not None:
        return
    import pair_v5_action_adapter as loaded_action_adapter
    import pair_v5_flow_dpo as loaded_flow_dpo
    import pair_v5_native_bridge as loaded_native_bridge
    import source_self_native_ref_contrastive_v3 as loaded_native
    import source_self_native_rv2v_guidance as loaded_guidance
    import source_self_native_target_adapter as loaded_cio_adapter
    import source_self_runtime as loaded_runtime
    import train_lora as loaded_legacy

    action_adapter = loaded_action_adapter
    flow_dpo = loaded_flow_dpo
    native_bridge = loaded_native_bridge
    native = loaded_native
    guidance = loaded_guidance
    cio_adapter = loaded_cio_adapter
    runtime = loaded_runtime
    legacy = loaded_legacy


@dataclass(frozen=True)
class PreparedPair:
    pair: Mapping[str, Any]
    source_video: Any
    image_references: tuple[Any, ...]
    winner_clean: Any
    loser_clean: Any
    conditional: Any
    unconditional: Any
    tensor_digest: str


def object_sha256(value: Any) -> str:
    return endpoint_contract.object_sha256(value)


def _load_manifest_file(path: str, expected_sha256: str) -> tuple[dict[str, Any], dict[str, str]]:
    binding = {"path": str(Path(path).resolve()), "sha256": expected_sha256}
    raw, checked = endpoint_contract.load_bound_json(
        binding, label="endpoint one-update manifest"
    )
    return raw, checked


def _parent_policy_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    policies: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        for endpoint_name in ("winner", "loser"):
            rollout_file = pair[endpoint_name]["rollout_evidence_file"]
            raw, _ = endpoint_contract.load_bound_json(
                rollout_file, label=f"{endpoint_name} rollout evidence"
            )
            evidence = endpoint_contract.validate_rollout_evidence(
                raw, replay_files=True
            )
            policies.append(evidence["parent_policy"])
    if (
        not policies
        or any(item != policies[0] for item in policies[1:])
        or policies[0]["policy_digest"] != manifest["parent_policy_digest"]
        or policies[0]["generation_round"] != manifest["generation_round"]
    ):
        raise PairV5EndpointOneUpdateError(
            "manifest endpoints do not share one replayed parent policy"
        )
    return policies[0]


def validate_cli(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.ack_experimental_no_action_success_claim is not True:
        raise PairV5EndpointOneUpdateError(
            "experimental no-success-claim acknowledgement is required"
        )
    if type(args.expected_generation_round) is not int or args.expected_generation_round < 0:
        raise PairV5EndpointOneUpdateError("expected generation round differs")
    endpoint_contract._sha256(
        args.expected_parent_policy_digest,
        label="expected parent policy digest",
    )
    endpoint_contract._sha256(
        args.expected_manifest_sha256, label="expected manifest SHA-256"
    )
    endpoint_contract._sha256(
        args.expected_checkpoint_tree_sha256,
        label="expected checkpoint tree SHA-256",
    )
    endpoint_contract._sha256(
        args.method_source_archive_sha256, label="method archive SHA-256"
    )
    if (
        type(args.sigma_index) is not int
        or args.sigma_index not in ACTION_SIGMA_INDICES
    ):
        raise PairV5EndpointOneUpdateError(
            "one-update sigma must be a registered high/mid exact40 index"
        )
    if (
        type(args.gradient_accumulation_steps) is not int
        or not 1 <= args.gradient_accumulation_steps <= 4
    ):
        raise PairV5EndpointOneUpdateError(
            "gradient accumulation must lie in [1,4]"
        )
    for name in ("learning_rate", "beta", "max_grad_norm"):
        value = getattr(args, name)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise PairV5EndpointOneUpdateError(f"{name} must be finite positive")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise PairV5EndpointOneUpdateError("seed must lie in [0,2^63)")
    return {
        "world_size": WORLD_SIZE,
        "data_parallel_size": DP_SIZE,
        "sequence_parallel_size": SP_SIZE,
        "optimizer_update_count": 1,
        "optimizer_step_index": 0,
        "sigma_index": args.sigma_index,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }


def preflight_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], Mapping[str, Any]]:
    run_contract = validate_cli(args)
    raw, manifest_file = _load_manifest_file(
        args.manifest, args.expected_manifest_sha256
    )
    try:
        manifest = endpoint_contract.authorize_manifest_for_single_step(
            raw,
            expected_generation_round=args.expected_generation_round,
            expected_parent_policy_digest=args.expected_parent_policy_digest,
            optimizer_step_index=0,
            replay_files=True,
        )
    except endpoint_contract.PairV5EndpointV3Error as error:
        raise PairV5EndpointOneUpdateError(str(error)) from error
    parent_policy = _parent_policy_from_manifest(manifest)
    if (
        parent_policy["checkpoint_tree_sha256"]
        != args.expected_checkpoint_tree_sha256
        or manifest["source_count"] < 2
        or manifest["pair_count"] < 2
        or manifest["optimizer_update_count"] != 1
    ):
        raise PairV5EndpointOneUpdateError(
            "checkpoint, independent-source, or one-update preflight differs"
        )
    return manifest, parent_policy, manifest_file, run_contract


def assigned_pair_indices(
    *, pair_count: int, dp_rank: int, accumulation_steps: int
) -> tuple[int, ...]:
    if (
        type(pair_count) is not int
        or pair_count < 2
        or dp_rank not in (0, 1)
        or type(accumulation_steps) is not int
        or not 1 <= accumulation_steps <= 4
    ):
        raise PairV5EndpointOneUpdateError("DP pair assignment inputs differ")
    return tuple(
        (accumulation_index * DP_SIZE + dp_rank) % pair_count
        for accumulation_index in range(accumulation_steps)
    )


def fresh_noise_seed(
    *,
    base_seed: int,
    manifest_digest: str,
    pair_digest: str,
    dp_rank: int,
    accumulation_index: int,
) -> int:
    material = (
        f"{base_seed}\x00pair-v5-endpoint-v3\x00{manifest_digest}\x00"
        f"{pair_digest}\x00{dp_rank}\x00{accumulation_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63


def _load_optional_frozen_cio(
    transformer: Any, adapter_binding: Optional[Mapping[str, Any]]
) -> tuple[Optional[Any], Mapping[str, Any], Optional[Mapping[str, str]]]:
    """Load optional CIO and always return the audited three-value tuple."""

    if adapter_binding is None:
        return None, {"loaded": False, "active_in_student_and_reference": False}, None
    import torch
    from safetensors import safe_open

    binding = endpoint_contract.validate_file_binding(
        adapter_binding, label="parent CIO adapter", verify_bytes=True
    )
    handle = cio_adapter.install_native_target_adapter(
        transformer,
        rank=8,
        alpha=8.0,
        block_indices=cio_adapter.DEFAULT_BLOCK_INDICES,
    )
    named = dict(handle.trainable_named_parameters())
    with safe_open(binding["path"], framework="pt", device="cpu") as opened:
        state = {
            name: opened.get_tensor(name).float().contiguous()
            for name in opened.keys()
        }
        metadata = dict(opened.metadata() or {})
    if set(state) != set(named):
        raise PairV5EndpointOneUpdateError("parent CIO key closure differs")
    if metadata.get("schema_version") != CIO_ADAPTER_CHECKPOINT_SCHEMA:
        raise PairV5EndpointOneUpdateError("parent CIO metadata schema differs")
    with torch.no_grad():
        for name, parameter in named.items():
            value = state[name]
            if (
                value.dtype != torch.float32
                or tuple(value.shape) != tuple(parameter.shape)
                or not bool(torch.isfinite(value).all().item())
            ):
                raise PairV5EndpointOneUpdateError(f"parent CIO tensor differs: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise PairV5EndpointOneUpdateError("parent CIO left parameters trainable")
    return handle, {
        "loaded": True,
        "file": binding,
        "metadata": metadata,
        "active_in_student_and_reference": True,
        "optimized": False,
    }, binding


def _load_parent_action_lora(
    handle: action_adapter.PairV5ActionAdapterHandle,
    parent_policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    binding = parent_policy["action_lora"]
    if binding is None:
        if parent_policy["generation_round"] != 0:
            raise PairV5EndpointOneUpdateError("round>0 parent Action-LoRA is absent")
        return {
            "loaded": False,
            "round0_zero_delta_parent": True,
            "active_in_student_and_reference": True,
        }
    from safetensors import safe_open

    checked = endpoint_contract.validate_file_binding(
        binding, label="parent Action-LoRA", verify_bytes=True
    )
    with safe_open(checked["path"], framework="pt", device="cpu") as opened:
        state = {
            name: opened.get_tensor(name).float().contiguous()
            for name in opened.keys()
        }
        metadata = dict(opened.metadata() or {})
    if (
        metadata.get("schema_version") != ADAPTER_CHECKPOINT_SCHEMA
        or metadata.get("next_generation_round")
        != str(parent_policy["generation_round"])
    ):
        raise PairV5EndpointOneUpdateError("parent Action-LoRA lineage metadata differs")
    load_receipt = handle.load_trainable_state_dict(state)
    return {
        "loaded": True,
        "file": checked,
        "metadata": metadata,
        "load_receipt": dict(load_receipt),
        "active_in_student_and_reference": True,
    }


def _native_rows(
    pack: native.NativeRV2VPack, *, conditional: Any, unconditional: Any
) -> tuple[tuple[str, Any, Any, float], ...]:
    rows = (
        (
            "none_uncond",
            pack.none,
            unconditional,
            native_bridge.EXPANDED_GUIDANCE_COEFFICIENTS["none_uncond"],
        ),
        (
            "V_uncond",
            pack.video,
            unconditional,
            native_bridge.EXPANDED_GUIDANCE_COEFFICIENTS["V_uncond"],
        ),
        (
            "VI_uncond",
            pack.video_image,
            unconditional,
            native_bridge.EXPANDED_GUIDANCE_COEFFICIENTS["VI_uncond"],
        ),
        (
            "VI_cond",
            pack.video_image,
            conditional,
            native_bridge.EXPANDED_GUIDANCE_COEFFICIENTS["VI_cond"],
        ),
    )
    if tuple(item[0] for item in rows) != tuple(
        guidance.guidance_receipt()["forward_order"]
    ) or not math.isclose(
        sum(item[3] for item in rows), 1.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise PairV5EndpointOneUpdateError("native RV2V branch registry differs")
    return rows


def _route_stack(
    stack: ExitStack,
    *,
    branch: Any,
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Optional[Any],
    sp_rank: int,
    sigma_index: int,
) -> None:
    if cio_handle is not None:
        stack.enter_context(
            cio_handle.route(
                cio_adapter.NativeTargetRoute(
                    total_tokens=branch.total_tokens,
                    condition_tokens=branch.condition_tokens,
                    sequence_parallel_rank=sp_rank,
                    sequence_parallel_size=SP_SIZE,
                    branch_name=branch.name,
                    enabled=True,
                )
            )
        )
    # The same parent Action-LoRA route is active for student and reference.
    stack.enter_context(
        action_handle.route(
            action_adapter.PairV5ActionRoute(
                total_tokens=branch.total_tokens,
                condition_tokens=branch.condition_tokens,
                sequence_parallel_rank=sp_rank,
                sequence_parallel_size=SP_SIZE,
                branch_name=branch.name,
                sigma_schedule_index=sigma_index,
                enabled=True,
            )
        )
    )


def _build_pack(
    transformer: Any,
    source_video: Any,
    references: Sequence[Any],
    state: Any,
) -> native.NativeRV2VPack:
    import torch

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return native.build_native_rv2v_pack(
            transformer,
            donor_video=source_video,
            image_references=references,
            noisy_target=state,
        )


def _forward_branch(
    diffusion: Any,
    branch: Any,
    *,
    timestep: Any,
    text: Any,
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Optional[Any],
    sp_rank: int,
    sigma_index: int,
) -> Any:
    with ExitStack() as stack:
        _route_stack(
            stack,
            branch=branch,
            action_handle=action_handle,
            cio_handle=cio_handle,
            sp_rank=sp_rank,
            sigma_index=sigma_index,
        )
        return native.forward_native_target_branch(
            diffusion, branch, timestep=timestep, cond_embeds=text
        )


def _guided_prediction_no_grad(
    diffusion: Any,
    pack: native.NativeRV2VPack,
    *,
    timestep: Any,
    conditional: Any,
    unconditional: Any,
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Optional[Any],
    sp_rank: int,
    sigma_index: int,
    video_shape: Sequence[int],
) -> Any:
    import torch

    components: dict[str, Any] = {}
    with torch.no_grad():
        for name, branch, text, _ in _native_rows(
            pack, conditional=conditional, unconditional=unconditional
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                components[name] = _forward_branch(
                    diffusion,
                    branch,
                    timestep=timestep,
                    text=text,
                    action_handle=action_handle,
                    cio_handle=cio_handle,
                    sp_rank=sp_rank,
                    sigma_index=sigma_index,
                )
    guided = (
        components["none_uncond"]
        + guidance.OMEGA_VIDEO
        * (components["V_uncond"] - components["none_uncond"])
        + guidance.OMEGA_IMAGE
        * (components["VI_uncond"] - components["V_uncond"])
        + guidance.OMEGA_TEXT
        * (components["VI_cond"] - components["VI_uncond"])
    )
    return native_bridge._unpack_spatial_velocity(
        guided.float(), video_shape=video_shape
    ).detach()


def _reference_and_student_predictions(
    diffusion: Any,
    transformer: Any,
    pair: PreparedPair,
    state: Any,
    *,
    timestep: Any,
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Optional[Any],
    sp_rank: int,
    sigma_index: int,
) -> tuple[Any, Any, str]:
    """Evaluate reference before any step with the same active parent policy."""

    before = runtime.trainable_parameters_digest(
        action_handle.trainable_named_parameters()
    )
    pack = _build_pack(
        transformer, pair.source_video, pair.image_references, state
    )
    try:
        reference = _guided_prediction_no_grad(
            diffusion,
            pack,
            timestep=timestep,
            conditional=pair.conditional,
            unconditional=pair.unconditional,
            action_handle=action_handle,
            cio_handle=cio_handle,
            sp_rank=sp_rank,
            sigma_index=sigma_index,
            video_shape=state.shape,
        )
        student = _guided_prediction_no_grad(
            diffusion,
            pack,
            timestep=timestep,
            conditional=pair.conditional,
            unconditional=pair.unconditional,
            action_handle=action_handle,
            cio_handle=cio_handle,
            sp_rank=sp_rank,
            sigma_index=sigma_index,
            video_shape=state.shape,
        )
    finally:
        del pack
    after = runtime.trainable_parameters_digest(
        action_handle.trainable_named_parameters()
    )
    if before != after:
        raise PairV5EndpointOneUpdateError(
            "reference evaluation changed frozen parent Action-LoRA"
        )
    return reference.detach(), student.detach(), before


def _replay_prediction_vjp(
    diffusion: Any,
    transformer: Any,
    pair: PreparedPair,
    state: Any,
    *,
    timestep: Any,
    action_handle: action_adapter.PairV5ActionAdapterHandle,
    cio_handle: Optional[Any],
    sp_rank: int,
    sigma_index: int,
    output_cotangent: Any,
    expected_guided: Any,
) -> float:
    import torch

    if (
        output_cotangent.shape != state.shape
        or output_cotangent.requires_grad
        or not bool(torch.isfinite(output_cotangent).all().item())
    ):
        raise PairV5EndpointOneUpdateError("guided output cotangent differs")
    pack = _build_pack(
        transformer, pair.source_video, pair.image_references, state
    )
    replay: dict[str, Any] = {}
    for name, branch, text, coefficient in _native_rows(
        pack, conditional=pair.conditional, unconditional=pair.unconditional
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packed = _forward_branch(
                diffusion,
                branch,
                timestep=timestep,
                text=text,
                action_handle=action_handle,
                cio_handle=cio_handle,
                sp_rank=sp_rank,
                sigma_index=sigma_index,
            )
            spatial = native_bridge._unpack_spatial_velocity(
                packed, video_shape=state.shape
            )
        replay[name] = spatial.detach()
        torch.autograd.backward(
            spatial,
            grad_tensors=output_cotangent.to(spatial.dtype) * float(coefficient),
        )
    guided = (
        replay["none_uncond"]
        + guidance.OMEGA_VIDEO * (replay["V_uncond"] - replay["none_uncond"])
        + guidance.OMEGA_IMAGE * (replay["VI_uncond"] - replay["V_uncond"])
        + guidance.OMEGA_TEXT * (replay["VI_cond"] - replay["VI_uncond"])
    ).float()
    maximum = float((guided - expected_guided.float()).abs().max().item())
    scale = float(expected_guided.float().abs().max().item())
    if maximum > VJP_REPLAY_ATOL + VJP_REPLAY_RTOL * scale:
        raise PairV5EndpointOneUpdateError(
            f"native VJP replay changed prediction: max={maximum}, scale={scale}"
        )
    return maximum


def _tokenize_positive(tokenizer: Any, text: str) -> tuple[Any, Any]:
    import torch

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids, mask = encoded.input_ids, encoded.attention_mask
    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[0] != 1:
        raise PairV5EndpointOneUpdateError("positive tokenization differs")
    if ids.shape[1] >= 512:
        return ids[:, :512], mask[:, :512]
    padding = 512 - ids.shape[1]
    return (
        torch.cat((ids, ids.new_zeros((1, padding))), dim=1),
        torch.cat((mask, mask.new_zeros((1, padding))), dim=1),
    )


def _tokenize_negative(tokenizer: Any, text: str) -> tuple[Any, Any]:
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    if tuple(encoded.input_ids.shape) != (1, 512):
        raise PairV5EndpointOneUpdateError("negative tokenization differs")
    return encoded.input_ids, encoded.attention_mask


def _broadcast_sp(value: Any, *, parallel: runtime.ParallelContext) -> None:
    import torch.distributed as dist

    source_rank = runtime.SP_GROUP_RANKS[parallel.contract.arm_index][0]
    dist.broadcast(value, src=source_rank, group=parallel.sp_group)


def _load_clean_latent(endpoint: Mapping[str, Any]) -> Any:
    import torch
    from safetensors import safe_open

    binding = endpoint_contract.validate_file_binding(
        endpoint["clean_latent"], label="endpoint clean latent", verify_bytes=True
    )
    with safe_open(binding["path"], framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [endpoint["clean_latent_tensor_key"]]:
            raise PairV5EndpointOneUpdateError("clean latent key closure differs")
        value = opened.get_tensor(endpoint["clean_latent_tensor_key"]).float().contiguous()
    if (
        tuple(value.shape) != tuple(endpoint["clean_latent_shape"])
        or value.dtype != torch.float32
        or tuple(value.shape[:3]) != (1, LATENT_CHANNELS, LATENT_PHASES)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PairV5EndpointOneUpdateError("clean latent tensor differs")
    return value.detach()


def _prepare_visual_pairs(
    manifest: Mapping[str, Any],
    pair_indices: Sequence[int],
    *,
    vae: Any,
    device: Any,
    parallel: runtime.ParallelContext,
    source_audit: Any,
    vae_encode: Any,
) -> dict[int, tuple[Any, tuple[Any, ...], Any, Any, str]]:
    import torch

    cache: dict[int, tuple[Any, tuple[Any, ...], Any, Any, str]] = {}
    vae.to(device)
    for pair_index in sorted(set(pair_indices)):
        pair = manifest["pairs"][pair_index]
        source_binding = endpoint_contract.validate_file_binding(
            pair["source_video"], label="pair source", verify_bytes=True
        )
        pixels, metadata, observed_sha = source_audit.prepare_hashed_source_snapshot(
            Path(source_binding["path"])
        )
        if (
            observed_sha != source_binding["sha256"]
            or metadata.get("frame_count") != FRAME_COUNT
            or float(metadata.get("fps", -1.0)) != FPS
        ):
            raise PairV5EndpointOneUpdateError("source video is not exact81/25fps")
        pixels = pixels.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            source = vae_encode(vae, pixels).float().contiguous()
            references = tuple(
                vae_encode(
                    vae, pixels[:, :, index : index + 1].contiguous()
                ).float().contiguous()
                for index in REFERENCE_INDICES
            )
        winner = _load_clean_latent(pair["winner"])
        loser = _load_clean_latent(pair["loser"])
        if (
            tuple(source.shape) != tuple(winner.shape)
            or tuple(winner.shape) != tuple(loser.shape)
            or any(
                tuple(reference.shape)
                != (1, LATENT_CHANNELS, 1, *tuple(winner.shape[3:]))
                for reference in references
            )
            or torch.equal(winner, loser)
        ):
            raise PairV5EndpointOneUpdateError("source/endpoint geometry differs")
        for tensor in (source, *references):
            _broadcast_sp(tensor, parallel=parallel)
        digest = object_sha256(
            [
                runtime.tensor_sha256(tensor)
                for tensor in (source, *references, winner, loser)
            ]
        )
        runtime.digest_consensus(
            digest,
            group=parallel.sp_group,
            expected_count=SP_SIZE,
            label=f"visual pair {pair['pair_id']}",
        )
        cache[pair_index] = (
            source.cpu().contiguous(),
            tuple(item.cpu().contiguous() for item in references),
            winner.cpu().contiguous(),
            loser.cpu().contiguous(),
            digest,
        )
        del pixels, source, references
        torch.cuda.empty_cache()
    vae.to("cpu")
    return cache


def _attach_text(
    manifest: Mapping[str, Any],
    visual: Mapping[int, tuple[Any, tuple[Any, ...], Any, Any, str]],
    *,
    renderer: Any,
    tokenizer: Any,
    device: Any,
    parallel: runtime.ParallelContext,
    build_task_prompt: Any,
    prompt_cleaner: Any,
) -> dict[int, PreparedPair]:
    import torch

    negative_ids, negative_mask = _tokenize_negative(
        tokenizer, legacy.DEFAULT_NEGATIVE_PROMPT
    )
    with torch.inference_mode():
        unconditional = renderer.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach()
    _broadcast_sp(unconditional, parallel=parallel)
    result: dict[int, PreparedPair] = {}
    for pair_index, values in visual.items():
        pair = manifest["pairs"][pair_index]
        prompt = build_task_prompt(
            "rv2v", pair["complete_caption"], prompt_cleaner=prompt_cleaner
        )
        ids, mask = _tokenize_positive(tokenizer, prompt)
        with torch.inference_mode():
            conditional = renderer.encode_prompt(ids.to(device), mask.to(device)).detach()
        _broadcast_sp(conditional, parallel=parallel)
        if (
            tuple(conditional.shape) != (1, 512, 4096)
            or tuple(unconditional.shape) != (1, 512, 4096)
        ):
            raise PairV5EndpointOneUpdateError("frozen text embedding differs")
        source, references, winner, loser, digest = values
        result[pair_index] = PreparedPair(
            pair,
            source,
            references,
            winner,
            loser,
            conditional.cpu().contiguous(),
            unconditional.cpu().contiguous(),
            digest,
        )
    return result


def _fresh_epsilon(shape: Sequence[int], *, seed: int, device: Any) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        tuple(shape), generator=generator, dtype=torch.float32
    ).to(device=device).contiguous().detach()


def _save_action_adapter(
    path: Path,
    handle: action_adapter.PairV5ActionAdapterHandle,
    *,
    generation_round: int,
    parent_policy_digest: str,
    manifest_digest: str,
) -> Mapping[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    state = dict(handle.state_dict_for_save())
    metadata = {
        "schema_version": ADAPTER_CHECKPOINT_SCHEMA,
        "trained_from_generation_round": str(generation_round),
        "next_generation_round": str(generation_round + 1),
        "parent_policy_digest": parent_policy_digest,
        "one_update_manifest_digest": manifest_digest,
        "action_adapter_contract_digest": str(handle.receipt()["digest"]),
        "flow_dpo_contract_digest": str(flow_dpo.contract_receipt()["digest"]),
        "native_bridge_contract_digest": str(
            native_bridge.bridge_contract_receipt()["digest"]
        ),
        "optimizer_update_count": "1",
        "reference_policy": "same_frozen_parent_action_lora_active",
    }
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".safetensors", delete=False
    ) as handle_file:
        temporary = Path(handle_file.name)
    try:
        save_file(state, str(temporary), metadata=metadata)
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            loaded = {
                name: opened.get_tensor(name).contiguous()
                for name in opened.keys()
            }
            loaded_metadata = dict(opened.metadata() or {})
        if loaded_metadata != metadata or set(loaded) != set(state) or any(
            loaded[name].dtype != torch.float32
            or not torch.equal(loaded[name], state[name])
            for name in state
        ):
            raise PairV5EndpointOneUpdateError(
                "Action-LoRA safetensors roundtrip differs"
            )
        runtime.durable_file_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "file_sha256": runtime.file_sha256(path),
        "tensor_count": len(state),
        "metadata": metadata,
        "roundtrip_exact": True,
    }


def _publish_create_only(stage: Path, output: Path) -> None:
    expected = {"adapter.safetensors", "optimizer.pt", "history.json", "receipt.json"}
    if {item.name for item in stage.iterdir()} != expected:
        raise PairV5EndpointOneUpdateError("staged artifact closure differs")
    output.mkdir(mode=0o750)
    for name in sorted(expected - {"receipt.json"}):
        os.link(stage / name, output / name)
    os.link(stage / "receipt.json", output / "receipt.json")
    runtime.fsync_directory(output)
    runtime.fsync_directory(output.parent)
    for name in expected:
        (stage / name).unlink()
    stage.rmdir()
    runtime.fsync_directory(output.parent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-generation-round", type=int, required=True)
    parser.add_argument("--expected-parent-policy-digest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sigma-index", type=int, default=DEFAULT_SIGMA_INDEX)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-bernini-commit", default=BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=VEOMNI_TESTED_COMMIT
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--ack-experimental-no-action-success-claim", action="store_true"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, parent_policy, manifest_file, run_contract = preflight_inputs(args)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "manifest_digest": manifest["manifest_digest"],
                    "generation_round": manifest["generation_round"],
                    "parent_policy_digest": manifest["parent_policy_digest"],
                    "pair_count": manifest["pair_count"],
                    "source_count": manifest["source_count"],
                    "optimizer_update_count": 1,
                    "cio_optional": parent_policy["cio_adapter"] is None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    _load_runtime_modules()
    if (runtime.WORLD_SIZE, runtime.SP_SIZE, runtime.DP_SIZE) != (
        WORLD_SIZE,
        SP_SIZE,
        DP_SIZE,
    ):
        raise PairV5EndpointOneUpdateError("WORLD8 DP2xSP4 runtime differs")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise PairV5EndpointOneUpdateError(str(error)) from error
    if (
        transformer_config.get("num_attention_heads") != 12
        or args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256
    ):
        raise PairV5EndpointOneUpdateError("pinned Bernini checkpoint differs")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    import infer_native_identity_generation_canary as native_canary

    distributed = runtime.distributed_contract()
    device = runtime.initialise_distributed(distributed)
    parallel = runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=SP_SIZE)
    )
    output, stage = runtime.prepare_output_transaction(
        args.output, distributed.rank, parallel.world_group
    )
    pair_indices = assigned_pair_indices(
        pair_count=manifest["pair_count"],
        dp_rank=distributed.arm_index,
        accumulation_steps=args.gradient_accumulation_steps,
    )

    legacy.seed_same_sample(args.seed)
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config)
    renderer.requires_grad_(False)
    renderer.eval()

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False)
    visual = _prepare_visual_pairs(
        manifest,
        pair_indices,
        vae=vae,
        device=device,
        parallel=parallel,
        source_audit=native_canary.source_audit,
        vae_encode=_vae_encode,
    )
    del vae
    torch.cuda.empty_cache()

    renderer.to(device)
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise PairV5EndpointOneUpdateError("one-update requires transformer_1 only")
    disable = getattr(renderer, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    if bool(getattr(transformer, "gradient_checkpointing", False)) or bool(
        getattr(transformer, "is_gradient_checkpointing", False)
    ):
        raise PairV5EndpointOneUpdateError("gradient checkpointing remains enabled")

    cio_handle, cio_receipt, cio_snapshot = _load_optional_frozen_cio(
        transformer, parent_policy["cio_adapter"]
    )
    action_handle = action_adapter.install_pair_v5_action_adapter(transformer)
    parent_action_receipt = _load_parent_action_lora(action_handle, parent_policy)
    trainable = action_handle.trainable_named_parameters()
    if not action_handle.base_parameters_frozen():
        raise PairV5EndpointOneUpdateError("Action-LoRA trainability closure differs")
    initial_digest = runtime.synchronize_initial_parameters(
        trainable, parallel.world_group
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    prepared = _attach_text(
        manifest,
        visual,
        renderer=renderer,
        tokenizer=tokenizer,
        device=device,
        parallel=parallel,
        build_task_prompt=native_canary.build_task_prompt,
        prompt_cleaner=prompt_clean,
    )
    renderer.t5_text_encoder.to("cpu")
    del tokenizer, visual
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    sigma_index = args.sigma_index
    sigma = torch.tensor(
        [native.NATIVE_UNIPC40_SIGMAS[sigma_index]],
        dtype=torch.float32,
        device=device,
    ).detach()
    timestep = torch.tensor(
        [native.NATIVE_UNIPC40_TIMESTEPS[sigma_index]],
        dtype=torch.float32,
        device=device,
    ).detach()
    micro_records = []
    reference_parent_digests = set()
    replay_max = 0.0
    for accumulation_index, pair_index in enumerate(pair_indices):
        cached = prepared[pair_index]
        pair = PreparedPair(
            cached.pair,
            cached.source_video.to(device).contiguous().detach(),
            tuple(item.to(device).contiguous().detach() for item in cached.image_references),
            cached.winner_clean.to(device).contiguous().detach(),
            cached.loser_clean.to(device).contiguous().detach(),
            cached.conditional.to(device).contiguous().detach(),
            cached.unconditional.to(device).contiguous().detach(),
            cached.tensor_digest,
        )
        for tensor in (
            pair.source_video,
            *pair.image_references,
            pair.winner_clean,
            pair.loser_clean,
            pair.conditional,
            pair.unconditional,
        ):
            _broadcast_sp(tensor, parallel=parallel)
        seed = fresh_noise_seed(
            base_seed=args.seed,
            manifest_digest=manifest["manifest_digest"],
            pair_digest=pair.pair["pair_digest"],
            dp_rank=distributed.arm_index,
            accumulation_index=accumulation_index,
        )
        epsilon = _fresh_epsilon(pair.winner_clean.shape, seed=seed, device=device)
        _broadcast_sp(epsilon, parallel=parallel)
        sigma_view = sigma.reshape(1, 1, 1, 1, 1)
        winner_state = (
            (1.0 - sigma_view) * pair.winner_clean + sigma_view * epsilon
        ).detach()
        loser_state = (
            (1.0 - sigma_view) * pair.loser_clean + sigma_view * epsilon
        ).detach()
        reference: dict[str, Any] = {}
        student: dict[str, Any] = {}
        for name, state in (("winner", winner_state), ("loser", loser_state)):
            ref, stu, parent_digest = _reference_and_student_predictions(
                diffusion,
                transformer,
                pair,
                state,
                timestep=timestep,
                action_handle=action_handle,
                cio_handle=cio_handle,
                sp_rank=distributed.sp_rank,
                sigma_index=sigma_index,
            )
            reference[name] = ref
            student[name] = stu
            reference_parent_digests.add(parent_digest)
        winner_leaf = student["winner"].clone().requires_grad_(True)
        loser_leaf = student["loser"].clone().requires_grad_(True)
        result = flow_dpo.reference_corrected_flow_dpo(
            pair.winner_clean,
            pair.loser_clean,
            epsilon,
            sigma,
            winner_leaf,
            loser_leaf,
            reference["winner"],
            reference["loser"],
            beta=args.beta,
        )
        if not runtime.world_all_true(
            bool(torch.isfinite(result.loss.detach()).item()),
            group=parallel.world_group,
        ):
            raise PairV5EndpointOneUpdateError("non-finite DPO loss blocked update")
        (result.loss / float(args.gradient_accumulation_steps)).backward()
        if winner_leaf.grad is None or loser_leaf.grad is None:
            raise PairV5EndpointOneUpdateError("DPO leaves have no cotangent")
        for state, cotangent, expected in (
            (winner_state, winner_leaf.grad.detach(), student["winner"]),
            (loser_state, loser_leaf.grad.detach(), student["loser"]),
        ):
            replay_max = max(
                replay_max,
                _replay_prediction_vjp(
                    diffusion,
                    transformer,
                    pair,
                    state,
                    timestep=timestep,
                    action_handle=action_handle,
                    cio_handle=cio_handle,
                    sp_rank=distributed.sp_rank,
                    sigma_index=sigma_index,
                    output_cotangent=cotangent,
                    expected_guided=expected,
                ),
            )
        micro_records.append(
            {
                "accumulation_index": accumulation_index,
                "dp_rank": distributed.arm_index,
                "pair_index": pair_index,
                "pair_id": pair.pair["pair_id"],
                "pair_digest": pair.pair["pair_digest"],
                "source_video_sha256": pair.pair["source_video"]["sha256"],
                "noise_seed": seed,
                "fresh_epsilon_sha256": runtime.tensor_sha256(epsilon),
                "same_epsilon_sigma_for_winner_loser": True,
                "reference_evaluated_before_optimizer_step": True,
                "reference_parent_action_lora_active": True,
                "loss": float(result.loss.detach().item()),
                "advantage": float(result.advantage.detach().item()),
                "student_gap": float(result.student_gap.detach().item()),
                "reference_gap": float(result.reference_gap.detach().item()),
            }
        )
        del pair, epsilon, winner_state, loser_state, reference, student, result
        torch.cuda.empty_cache()
    if reference_parent_digests != {initial_digest}:
        raise PairV5EndpointOneUpdateError(
            "frozen reference did not use the synchronized parent policy"
        )
    preclip_norm = runtime.synchronize_gradients(trainable, parallel)
    clipped = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in trainable], args.max_grad_norm
    )
    if not math.isfinite(float(clipped)):
        raise PairV5EndpointOneUpdateError("gradient clipping is non-finite")
    optimizer.step()  # The only optimizer step in this executable.
    optimizer_update_count = 1
    final_digest = runtime.parameter_consensus(
        trainable, parallel.world_group, "PAIR-v5 endpoint v3 one update"
    )
    if final_digest == initial_digest:
        raise PairV5EndpointOneUpdateError("one optimizer step changed no parameter")
    endpoint_contract.validate_one_update_manifest(manifest, replay_files=True)
    if cio_snapshot is not None:
        endpoint_contract.validate_file_binding(
            cio_snapshot, label="post-update CIO", verify_bytes=True
        )
    dist.barrier(group=parallel.world_group)

    local_history = {
        "optimizer_step_index": 0,
        "optimizer_step_called": True,
        "optimizer_update_count": optimizer_update_count,
        "dp_rank": distributed.arm_index,
        "sp_rank": distributed.sp_rank,
        "sigma_index": sigma_index,
        "micro_records": micro_records,
        "preclip_gradient_norm": preclip_norm,
        "vjp_replay_max_abs": replay_max,
        "initial_parent_parameter_digest": initial_digest,
        "final_parameter_digest": final_digest,
    }
    projection = {key: value for key, value in local_history.items() if key != "sp_rank"}
    runtime.digest_consensus(
        object_sha256(projection),
        group=parallel.sp_group,
        expected_count=SP_SIZE,
        label="endpoint v3 SP history",
    )
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local_history, group=parallel.world_group)
    history = {"dp_records": [gathered[0], gathered[4]]}

    if distributed.rank == 0:
        adapter_path = stage / "adapter.safetensors"
        optimizer_path = stage / "optimizer.pt"
        history_path = stage / "history.json"
        adapter_roundtrip = _save_action_adapter(
            adapter_path,
            action_handle,
            generation_round=manifest["generation_round"],
            parent_policy_digest=manifest["parent_policy_digest"],
            manifest_digest=manifest["manifest_digest"],
        )
        runtime.atomic_torch_save(
            optimizer_path,
            {
                "schema_version": RUN_RECEIPT_SCHEMA,
                "optimizer": optimizer.state_dict(),
                "optimizer_update_count": 1,
                "manifest_digest": manifest["manifest_digest"],
                "initial_parent_parameter_digest": initial_digest,
                "final_parameter_digest": final_digest,
            },
        )
        runtime.atomic_json(
            history_path,
            {
                "schema_version": HISTORY_SCHEMA,
                "optimizer_update_count": 1,
                "records": [history],
            },
        )
        receipt: dict[str, Any] = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "complete": True,
            "generation_round": manifest["generation_round"],
            "next_generation_round": manifest["expected_next_generation_round"],
            "parent_policy": parent_policy,
            "parent_policy_digest": manifest["parent_policy_digest"],
            "manifest_file": manifest_file,
            "manifest_digest": manifest["manifest_digest"],
            "gate_policy_digest": manifest["gate_policy_digest"],
            "source_count": manifest["source_count"],
            "pair_count": manifest["pair_count"],
            "run_contract": dict(run_contract),
            "optimizer_update_count": 1,
            "optimizer_step_indices": [0],
            "static_rollout_reused_for_multiple_steps": False,
            "fresh_next_round_rollout_required": True,
            "reference_policy": {
                "definition": "same_frozen_parent_policy_that_generated_rollouts",
                "parent_action_lora_active": True,
                "cio": dict(cio_receipt),
                "parent_action_lora": dict(parent_action_receipt),
                "parameter_digest": initial_digest,
                "evaluated_before_optimizer_step": True,
                "detached": True,
            },
            "objective": {
                "flow_dpo_contract": dict(flow_dpo.contract_receipt()),
                "beta": args.beta,
                "same_fresh_epsilon_sigma_for_each_pair": True,
                "serial_exact_linear_vjp": True,
            },
            "adapter": {
                **dict(action_handle.receipt()),
                "initial_parent_parameter_digest": initial_digest,
                "final_parameter_digest": final_digest,
                "changed_by_exactly_one_optimizer_step": True,
                "safetensors_roundtrip": dict(adapter_roundtrip),
            },
            "distributed": {
                "world_size": WORLD_SIZE,
                "data_parallel_size": DP_SIZE,
                "sequence_parallel_size": SP_SIZE,
                "all_eight_gpus_used": True,
                "sp_groups": [list(item) for item in runtime.SP_GROUP_RANKS],
                "dp_groups": [list(item) for item in runtime.DP_GROUP_RANKS],
            },
            "history_summary": history,
            "model": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                "single_expert": "transformer_1",
            },
            "runtime": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
            },
            "artifacts": {
                "adapter.safetensors": runtime.file_sha256(adapter_path),
                "optimizer.pt": runtime.file_sha256(optimizer_path),
                "history.json": runtime.file_sha256(history_path),
            },
            "engineering_experiment_only": True,
            "semantic_action_editing_success": False,
            "scientific_generalization_claim_authorized": False,
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
        }
        receipt["receipt_digest"] = runtime.object_sha256(receipt)
        runtime.atomic_json(stage / "receipt.json", receipt)
        runtime.verify_staged_run_bundle(stage, receipt)
        runtime.fsync_directory(stage)
        _publish_create_only(stage, output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "generation_round": manifest["generation_round"],
                    "next_generation_round": manifest["expected_next_generation_round"],
                    "optimizer_update_count": 1,
                    "adapter_parameter_digest": final_digest,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=parallel.world_group)
    if not output.is_dir() or output.is_symlink() or stage.exists():
        raise PairV5EndpointOneUpdateError("atomic output publication differs")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_SIGMA_INDICES",
    "ADAPTER_CHECKPOINT_SCHEMA",
    "PairV5EndpointOneUpdateError",
    "assigned_pair_indices",
    "build_parser",
    "fresh_noise_seed",
    "preflight_inputs",
    "validate_cli",
]
