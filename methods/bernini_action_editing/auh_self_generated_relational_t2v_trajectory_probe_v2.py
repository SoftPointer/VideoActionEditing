#!/usr/bin/env python3
"""Real WORLD4 self-generated intermediate T2V representation probe.

This create-only AUH runner evolves three pure-T2V action trajectories from
one authenticated, source-independent Gaussian.  At exact UniPC indices
18/32/38 it freezes the current *intermediate trajectory state* and evaluates
five branches on the same tensor objects:

* B0 frozen base, observer absent, action prompt;
* observer action;
* observer role-matched no-op;
* observer reverse;
* observer static.

The action observer output must be bit-exact to B0.  Q/K and the explicitly
derived role-responsibility proxy are reduced immediately to sparse,
pre-registered relative graph edges and then zeroized.  No final anchor video,
target, source video tensor, decoder, optimizer, adapter, route, or parameter
update is available to the scientific probe path.

The AUH site adapter is reused only to authenticate/load the frozen checkpoint
and official WORLD4 implementation.  Its E00 bootstrap tensors are explicitly
scrubbed before any trajectory step and cannot enter a probe forward.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.distributed as dist


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import anchor_sga_anc_controller as trajectory  # noqa: E402
import auh_native_relational_attention_parity_smoke_v1 as parity  # noqa: E402
import auh_source_owned_role_locator_v15_adapter as site  # noqa: E402
import differential_sampler as cdf  # noqa: E402
import guided_source_aligned_controller as guided  # noqa: E402
import infer_native_self_generated_relational_graph_observer_v1 as native  # noqa: E402
import native_relational_attention_hook_v1 as attention_hook  # noqa: E402
import self_generated_intermediate_action_anchor_v1 as anchor_core  # noqa: E402
import self_generated_relational_action_graph_observer_v1 as observer  # noqa: E402
import self_generated_relational_t2v_probe_registry_v2 as registry  # noqa: E402
import source_owned_role_locator_v15 as locator  # noqa: E402


METHOD = "bernini-auh-self-generated-relational-t2v-trajectory-probe-v2"
SCHEMA_VERSION = "bernini-auh-self-generated-relational-t2v-trajectory-probe-v2"
WORLD_SIZE = 4
TEXT_LENGTH = 512
TEXT_WIDTH = 4096
EXPECTED_CAPTURE_COUNT = 144


class AUHSelfGeneratedRelationalProbeError(RuntimeError):
    """Fail-closed runtime, provenance, parity, or representation violation."""


def probe_contract() -> Mapping[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "registry": dict(registry.registry_receipt()),
        "checkpoint_tree_sha256": site.CHECKPOINT_TREE_SHA256,
        "official_transformer_source_sha256": (
            attention_hook.OFFICIAL_TRANSFORMER_SOURCE_SHA256
        ),
        "world_size": WORLD_SIZE,
        "num_inference_steps": 40,
        "guidance": "official_t2v_apg_momentum_zero",
        "trajectory": "native_unipc_action_prompt_intermediate_states",
        "capture_blocks": list(native.BLOCKS),
        "capture_count": EXPECTED_CAPTURE_COUNT,
        "frozen_base_arm": "B0_FROZEN_BASE",
        "frozen_base_per_appearance_sigma": True,
        "frozen_base_graph_observation_supplied": False,
        "observer_action_must_equal_frozen_base_bit_exact": True,
        "source_bootstrap_tensor_consumed_by_trajectory_or_probe_forward": False,
        "target_inputs_consumed": False,
        "final_anchor_video_decode": False,
        "decoder_available_to_probe": False,
        "adapter_or_lora_loaded": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "route_or_injection_called": False,
        "candidate_output_modified": False,
        "backend_attention_weights_observed": False,
        "responsibility_kind": attention_hook.RESPONSIBILITY_KIND,
        "persistent_raw_tensor_artifact": False,
        "scientific_claim_authorized": False,
        "causal_generation_claimed": False,
    }
    return {**value, "digest": registry.object_sha256(value)}


def remote_launch_template() -> Mapping[str, Any]:
    value = {
        "python": "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12",
        "launcher": "python -m torch.distributed.run",
        "nproc_per_node": WORLD_SIZE,
        "entrypoint": Path(__file__).name,
        "arguments": ["--run", "--output", "ABSENT_CREATE_ONLY_JSON_PATH"],
        "required_environment": {
            "MODELING_BACKEND": "hf",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        "launch_executed": False,
    }
    return {**value, "digest": registry.object_sha256(value)}


def _all_rank_rows(value: Any) -> list[Any]:
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, value)
    return rows


def _require_all_rank_equal(value: Any, *, label: str) -> None:
    if _all_rank_rows(value) != [value] * WORLD_SIZE:
        raise AUHSelfGeneratedRelationalProbeError(f"{label} differs across WORLD4")


def _check_world4() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise AUHSelfGeneratedRelationalProbeError("WORLD4 is not initialized")
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        env_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        env_world = int(os.environ["WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as error:
        raise AUHSelfGeneratedRelationalProbeError("torchrun rank environment differs") from error
    if world != WORLD_SIZE or env_world != WORLD_SIZE or rank != env_rank:
        raise AUHSelfGeneratedRelationalProbeError("runtime is not exact WORLD4")
    if not 0 <= local_rank < WORLD_SIZE:
        raise AUHSelfGeneratedRelationalProbeError("LOCAL_RANK is outside WORLD4")
    return rank, local_rank


def _padded_offsets_and_partition(
    *,
    tokenizer: Any,
    legacy: Any,
    model_prompt: str,
    role_phrases: Mapping[str, str],
    role_ids: Sequence[str] = registry.ROLE_IDS,
) -> tuple[torch.Tensor, torch.Tensor, attention_hook.ExhaustiveTextRolePartition, Mapping[str, Any]]:
    """Tokenize once and bind every text key to an exact semantic/null role."""

    encoded = tokenizer(
        model_prompt,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    raw_ids = encoded.input_ids
    raw_mask = encoded.attention_mask
    offsets = encoded.offset_mapping
    padded_ids, padded_mask = legacy._tokenize_training_prompt(tokenizer, model_prompt)
    if (
        raw_ids.ndim != 2
        or tuple(raw_ids.shape) != tuple(raw_mask.shape)
        or offsets.ndim != 3
        or tuple(offsets.shape[:2]) != tuple(raw_ids.shape)
    ):
        raise AUHSelfGeneratedRelationalProbeError("tokenizer offset ABI differs")
    retained = min(int(raw_ids.shape[1]), TEXT_LENGTH)
    if (
        tuple(padded_ids.shape) != (1, TEXT_LENGTH)
        or tuple(padded_mask.shape) != (1, TEXT_LENGTH)
        or not torch.equal(padded_ids[:, :retained], raw_ids[:, :retained])
        or not torch.equal(padded_mask[:, :retained], raw_mask[:, :retained])
    ):
        raise AUHSelfGeneratedRelationalProbeError("prompt token/offset authority differs")
    padded_offsets = [(0, 0)] * TEXT_LENGTH
    for index, pair in enumerate(offsets[0, :retained].tolist()):
        padded_offsets[index] = (int(pair[0]), int(pair[1]))

    role_names = tuple(role_ids)
    null_index = role_names.index("null_context")
    owner = [null_index] * TEXT_LENGTH
    support: dict[str, list[int]] = {name: [] for name in role_names}
    folded_prompt = model_prompt.casefold()
    character_ranges: dict[str, list[tuple[int, int]]] = {}
    for role_name in role_names[:-1]:
        phrase = role_phrases.get(role_name)
        if not isinstance(phrase, str) or not phrase:
            raise AUHSelfGeneratedRelationalProbeError("role phrase registry differs")
        folded_phrase = phrase.casefold()
        starts: list[tuple[int, int]] = []
        cursor = 0
        while True:
            start = folded_prompt.find(folded_phrase, cursor)
            if start < 0:
                break
            starts.append((start, start + len(folded_phrase)))
            cursor = start + len(folded_phrase)
        if not starts:
            raise AUHSelfGeneratedRelationalProbeError(
                f"model prompt lacks role phrase {role_name}"
            )
        character_ranges[role_name] = starts

    for token_index, (token_start, token_end) in enumerate(padded_offsets):
        if token_end <= token_start:
            support["null_context"].append(token_index)
            continue
        matches = []
        for role_index, role_name in enumerate(role_names[:-1]):
            if any(
                token_start < char_end and token_end > char_start
                for char_start, char_end in character_ranges[role_name]
            ):
                matches.append((role_index, role_name))
        if len(matches) > 1:
            raise AUHSelfGeneratedRelationalProbeError("semantic role token spans overlap")
        if matches:
            owner[token_index] = matches[0][0]
            support[matches[0][1]].append(token_index)
        else:
            support["null_context"].append(token_index)
    if any(not support[name] for name in role_names):
        raise AUHSelfGeneratedRelationalProbeError("one text role lacks token support")
    partition = attention_hook.ExhaustiveTextRolePartition(
        role_names=role_names,
        token_to_role=tuple(owner),
    )
    receipt = {
        "model_prompt_sha256": hashlib.sha256(model_prompt.encode("utf-8")).hexdigest(),
        "input_ids_sha256": locator.tensor_sha256(padded_ids),
        "attention_mask_sha256": locator.tensor_sha256(padded_mask),
        "active_attention_mask_tokens": int(padded_mask.sum().item()),
        "role_partition_sha256": partition.digest,
        "role_token_counts": {name: len(support[name]) for name in role_names},
        "all_512_text_keys_exhaustively_owned": True,
        "null_context_includes_special_padding_and_nonrole_tokens": True,
    }
    return padded_ids, padded_mask, partition, {
        **receipt,
        "digest": registry.object_sha256(receipt),
    }


def _encode_prompt_bank(
    runtime: Any,
    *,
    rank: int,
    registry_module: Any = registry,
) -> tuple[
    Mapping[str, Mapping[str, torch.Tensor]],
    torch.Tensor,
    Mapping[str, Mapping[str, attention_hook.ExhaustiveTextRolePartition]],
    Mapping[str, Any],
]:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import UMT5EncoderModel
    from bernini.cli import DEFAULT_NEG_PROMPT

    names = [
        f"{appearance.appearance_id}:{arm}"
        for appearance in registry_module.APPEARANCES
        for arm in registry_module.ARMS
    ]
    token_rows: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    partitions: dict[str, dict[str, attention_hook.ExhaustiveTextRolePartition]] = {}
    prompt_receipts: dict[str, Any] = {}
    model_prompts: dict[str, str] = {}
    token_status: list[Any] = [None]
    if rank == 0:
        try:
            for appearance in registry_module.APPEARANCES:
                partitions[appearance.appearance_id] = {}
                prompt_receipts[appearance.appearance_id] = {}
                for arm in registry_module.ARMS:
                    cleaned = prompt_clean(appearance.captions[arm])
                    model_prompt = (
                        "You are a helpful assistant specialized in text-to-video generation."
                        + cleaned
                    )
                    key = f"{appearance.appearance_id}:{arm}"
                    ids, mask, partition, receipt = _padded_offsets_and_partition(
                        tokenizer=runtime._tokenizer,
                        legacy=runtime._legacy,
                        model_prompt=model_prompt,
                        role_phrases=appearance.role_phrases,
                        role_ids=registry_module.ROLE_IDS,
                    )
                    token_rows[key] = (ids, mask)
                    partitions[appearance.appearance_id][arm] = partition
                    prompt_receipts[appearance.appearance_id][arm] = receipt
                    model_prompts[key] = model_prompt
            negative_ids, negative_mask = runtime._legacy._tokenize_renderer_negative(
                runtime._tokenizer, DEFAULT_NEG_PROMPT
            )
            token_rows["negative"] = (negative_ids, negative_mask)
            prompt_receipts["negative"] = {
                "text_sha256": hashlib.sha256(DEFAULT_NEG_PROMPT.encode("utf-8")).hexdigest(),
                "input_ids_sha256": locator.tensor_sha256(negative_ids),
                "attention_mask_sha256": locator.tensor_sha256(negative_mask),
            }
            token_status[0] = {
                "ok": True,
                "partition_rows": {
                    appearance_id: {
                        arm: {
                            "role_names": list(partition.role_names),
                            "token_to_role": list(partition.token_to_role),
                        }
                        for arm, partition in arm_rows.items()
                    }
                    for appearance_id, arm_rows in partitions.items()
                },
                "prompt_receipts": prompt_receipts,
                "model_prompts": model_prompts,
            }
        except Exception as error:
            token_status[0] = {
                "ok": False,
                "type": type(error).__name__,
                "message": str(error),
            }
    dist.broadcast_object_list(token_status, src=0)
    status = token_status[0]
    if not isinstance(status, Mapping) or status.get("ok") is not True:
        raise AUHSelfGeneratedRelationalProbeError(f"rank-zero prompt binding failed: {status}")
    if rank != 0:
        prompt_receipts = dict(status["prompt_receipts"])
        model_prompts = dict(status["model_prompts"])
        partitions = {
            appearance_id: {
                arm: attention_hook.ExhaustiveTextRolePartition(
                    role_names=tuple(row["role_names"]),
                    token_to_role=tuple(row["token_to_role"]),
                )
                for arm, row in arm_rows.items()
            }
            for appearance_id, arm_rows in status["partition_rows"].items()
        }

    # The site adapter has already retired its constructor-time T5.  Reload it
    # on rank zero only; peers allocate only the 12+1 BF16 result tensors.
    if getattr(runtime.model, "t5_text_encoder", None) is not None:
        raise AUHSelfGeneratedRelationalProbeError("site T5 was not retired")
    embeddings_local: dict[str, torch.Tensor] = {}
    encode_status: list[Any] = [None]
    dist.barrier()
    if rank == 0:
        try:
            runtime.model.t5_text_encoder = UMT5EncoderModel.from_pretrained(
                str(site.CHECKPOINT),
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            runtime.model.t5_text_encoder.to(runtime.device)
            with torch.inference_mode():
                for name in (*names, "negative"):
                    ids, mask = token_rows[name]
                    value = runtime.model.encode_prompt(
                        ids.to(runtime.device), mask.to(runtime.device)
                    ).detach().contiguous()
                    if tuple(value.shape) != (1, TEXT_LENGTH, TEXT_WIDTH) or value.dtype != torch.bfloat16:
                        raise AUHSelfGeneratedRelationalProbeError(
                            f"prompt embedding geometry differs for {name}"
                        )
                    embeddings_local[name] = value
            encode_status[0] = {"ok": True}
        except Exception as error:
            encode_status[0] = {
                "ok": False,
                "type": type(error).__name__,
                "message": str(error),
            }
        finally:
            if getattr(runtime.model, "t5_text_encoder", None) is not None:
                runtime._event_runtime._retire_t5_text_encoder(
                    runtime.model, torch_module=torch
                )
    dist.broadcast_object_list(encode_status, src=0)
    if not isinstance(encode_status[0], Mapping) or encode_status[0].get("ok") is not True:
        raise AUHSelfGeneratedRelationalProbeError(
            f"rank-zero prompt encoding failed: {encode_status[0]}"
        )

    broadcast_bank: dict[str, torch.Tensor] = {}
    embedding_sha: dict[str, str] = {}
    for name in (*names, "negative"):
        value = embeddings_local.get(name)
        if rank != 0:
            value = torch.empty(
                (1, TEXT_LENGTH, TEXT_WIDTH),
                dtype=torch.bfloat16,
                device=runtime.device,
            )
        if value is None:
            raise AUHSelfGeneratedRelationalProbeError("rank-zero embedding is absent")
        dist.broadcast(value, src=0)
        value = value.detach().contiguous()
        digest = locator.tensor_sha256(value)
        _require_all_rank_equal(digest, label=f"{name} prompt embedding")
        broadcast_bank[name] = value
        embedding_sha[name] = digest

    prompt_bank = {
        appearance.appearance_id: {
            arm: broadcast_bank[f"{appearance.appearance_id}:{arm}"]
            for arm in registry_module.ARMS
        }
        for appearance in registry_module.APPEARANCES
    }
    value = {
        "prompt_receipts": prompt_receipts,
        "model_prompt_sha256": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in sorted(model_prompts.items())
        },
        "embedding_sha256": embedding_sha,
        "rank_zero_only_t5_load": True,
        "t5_retired_before_trajectory": True,
        "role_partitions_exhaustive": True,
    }
    return prompt_bank, broadcast_bank["negative"], partitions, {
        **value,
        "digest": registry_module.object_sha256(value),
    }


def _scrub_site_bootstrap_tensors(runtime: Any) -> Mapping[str, Any]:
    names = (
        "_raw_source_text",
        "_timestep",
        "_noisy_source",
        "_visual_tokens",
        "_visual_rotary",
    )
    rows = []
    with torch.inference_mode():
        for name in names:
            value = getattr(runtime, name, None)
            if not isinstance(value, torch.Tensor):
                raise AUHSelfGeneratedRelationalProbeError(
                    f"site bootstrap tensor {name} is absent"
                )
            shape = [int(item) for item in value.shape]
            value.zero_()
            if int(torch.count_nonzero(value).item()) != 0:
                raise AUHSelfGeneratedRelationalProbeError(
                    f"site bootstrap tensor {name} did not scrub"
                )
            setattr(runtime, name, None)
            rows.append({"name": name, "shape": shape, "zeroized": True})
    value = {
        "rows": rows,
        "source_bootstrap_tensor_count": len(rows),
        "all_zeroized_and_references_cleared_before_trajectory": True,
        "source_bootstrap_tensor_consumed_by_probe_forward": False,
    }
    return {**value, "digest": registry.object_sha256(value)}


def _load_initial_gaussian(runtime: Any) -> tuple[torch.Tensor, Mapping[str, Any]]:
    from safetensors.torch import load_file

    rows = load_file(str(site.NOISE), device="cpu")
    if tuple(rows) != ("official_initial_gaussian",):
        raise AUHSelfGeneratedRelationalProbeError("initial Gaussian key differs")
    value = rows["official_initial_gaussian"].float().contiguous()
    digest = locator.tensor_sha256(value)
    if tuple(value.shape) != site.LATENT_SHAPE or digest != site.NOISE_TENSOR_SHA256:
        raise AUHSelfGeneratedRelationalProbeError("initial Gaussian bytes differ")
    value = value.to(runtime.device).contiguous()
    _require_all_rank_equal(locator.tensor_sha256(value), label="initial Gaussian")
    receipt = {
        "file_sha256": site.NOISE_FILE_SHA256,
        "tensor_sha256": digest,
        "shape": list(site.LATENT_SHAPE),
        "source_or_target_derived": False,
        "captured_from_native_sampler": True,
        "same_bytes_reused_across_three_appearances": True,
    }
    return value, {**receipt, "digest": registry.object_sha256(receipt)}


def _real_capture_plan(diffusion: Any, device: torch.device) -> tuple[native.CapturePlan, Any, Any]:
    config = cdf.DifferentialFlowConfig(num_inference_steps=40, flow_shift=5.0, seed=20260823)
    timesteps, raw_intervals = cdf._set_scheduler_timesteps(diffusion, config, device)
    intervals = guided.validate_pinned_sigma_intervals(raw_intervals)
    sigma_tensors, sigma_digest = guided.capture_pinned_scheduler_sigma_scalars(
        diffusion, intervals
    )
    if sigma_digest != guided.PINNED_UNIPC_SIGMA_FP32_DIGEST:
        raise AUHSelfGeneratedRelationalProbeError("pinned sigma vector differs")
    cells = []
    for band in native.SIGMA_BAND_ORDER:
        index = int(registry.SIGMA_CELL_INDICES[band])
        cells.append(native.SigmaCell(band, index, float(sigma_tensors[index].item())))
    plan = native.CapturePlan(tuple(cells))
    return plan, timesteps, sigma_tensors


def _observer_stream() -> tuple[tuple[Any, ...], Any]:
    roles = (
        observer.RoleSpec(
            "agent", "self_generated_anchor_owned", semantic_role="human_agent"
        ),
        observer.RoleSpec(
            "moving_object", "self_generated_anchor_owned", semantic_role="moving_object"
        ),
        observer.RoleSpec(
            "start_support", "self_generated_anchor_owned", semantic_role="support_surface"
        ),
        observer.RoleSpec(
            "end_support", "self_generated_anchor_owned", semantic_role="support_surface"
        ),
        observer.RoleSpec(
            "null_context",
            "self_generated_anchor_owned",
            semantic_role="distractor",
            critical=False,
        ),
    )
    edge_specs = tuple(observer.EdgeSpec(**row) for row in registry.registry_receipt()["typed_edges"])
    stream = observer.StreamingRelationalObserver(
        roles=roles,
        config=observer.ObserverConfig(edge_specs=edge_specs),
    )
    return roles, stream


def _capture_arm(
    *,
    runtime: Any,
    authority: native.RoleMatchedT2VFourArmForwardAuthority,
    arm: str,
    partition: attention_hook.ExhaustiveTextRolePartition,
    rank_bank: attention_hook.InMemoryWorld4RankShardBank,
    native_bank: native.InMemoryNativeCaptureBank,
    stream: Any,
    roles: Sequence[Any],
    rank: int,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    invocation = native.CaptureInvocation(
        authority.appearance_id,
        arm,
        authority.sigma_cell,
        authority.state_tensor_sha256["noisy_latents"],
        authority.state_tensor_sha256["timesteps"],
        authority.state_tensor_sha256["rotary_embs"],
        runtime.source_geometry.height,
        runtime.source_geometry.width,
    )
    rank_invocation = attention_hook.RankCaptureInvocation(
        invocation,
        attention_hook.World4RankLayout(
            rank, runtime.source_geometry.height, runtime.source_geometry.width
        ),
        partition,
    )
    with torch.inference_mode(), rank_bank.observe(rank_invocation):
        output = authority.call(arm)
    output = output.detach()
    local_shards = rank_bank.take_rank(rank_invocation)
    gathered = []
    for shard in local_shards:
        gathered.extend(
            parity._gather_one_block_after_forward(
                shard,
                invocation=invocation,
                role_partition=partition,
            )
        )
    commit = attention_hook.commit_world4_shards_to_native_bank(
        native_bank=native_bank,
        invocation=invocation,
        rank_shards=gathered,
    )
    captures = native_bank.consume(invocation)
    summary = [
        {
            "block_index": item.block_index,
            "query_shape": [int(value) for value in item.query.shape],
            "proxy_shape": [
                int(value)
                for value in item.derived_qk_role_responsibility_proxy.shape
            ],
        }
        for item in captures
    ]
    native._stream_native_capture_group(
        observer=observer,
        stream=stream,
        captures=captures,
        roles=roles,
        prompt_sha256=authority.prompt_embedding_sha256[arm],
    )
    native_bank.zeroize(captures)
    if any(
        int(torch.count_nonzero(tensor).item()) != 0
        for item in captures
        for tensor in (
            item.query,
            item.key,
            item.derived_qk_role_responsibility_proxy,
        )
    ):
        raise AUHSelfGeneratedRelationalProbeError("upstream capture did not zeroize")
    receipt = {
        "arm": arm,
        "output_sha256": parity._tensor_digest(output, label=f"{arm} output"),
        "role_partition_sha256": partition.digest,
        "commit": dict(commit),
        "capture_summary": summary,
        "raw_capture_zeroized": True,
    }
    return output, {**receipt, "digest": registry.object_sha256(receipt)}


def run_real_world4_probe(output: Path) -> Mapping[str, Any]:
    rank, _local_rank = _check_world4()
    if not output.is_absolute() or output.is_symlink() or output.exists():
        raise AUHSelfGeneratedRelationalProbeError("output must be an absolute absent plain path")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise AUHSelfGeneratedRelationalProbeError("output parent must be a plain directory")

    runtime = site.create_auh_bernini_source_role_adapter({})
    binding_receipt = runtime.binding_receipt()
    prompt_bank, negative, partitions, prompt_receipt = _encode_prompt_bank(
        runtime, rank=rank, registry_module=registry
    )
    bootstrap_scrub = _scrub_site_bootstrap_tensors(runtime)
    initial_gaussian, gaussian_receipt = _load_initial_gaussian(runtime)
    plan, timesteps, sigmas = _real_capture_plan(runtime.model.diff_dec, runtime.device)
    state_before = parity._module_state_version_receipt(runtime._transformer)
    roles, stream = _observer_stream()
    rank_bank = attention_hook.InMemoryWorld4RankShardBank()
    native_bank = native.InMemoryNativeCaptureBank()
    base_cells = []
    authority_receipts = []
    capture_receipts = []
    trajectory_steps = []

    layout = cdf.validate_latent_shape(tuple(initial_gaussian.shape))
    initial_packed = cdf._pack_spatial_latent(initial_gaussian, layout)
    selected_by_index = {cell.step_index: cell for cell in plan.sigma_cells}
    compute_dtype = runtime._transformer.patch_embedding.weight.dtype
    if compute_dtype != torch.bfloat16:
        raise AUHSelfGeneratedRelationalProbeError("transformer compute dtype differs")

    for appearance in registry.APPEARANCES:
        state_packed = initial_packed.clone()
        solver = copy.deepcopy(runtime.model.diff_dec.scheduler)
        solver.set_timesteps(40)
        selected_seen = []
        for step_index, timestep in enumerate(timesteps):
            spatial_state = cdf._unpack_spatial_latent(state_packed, layout)
            velocity = trajectory._guided_source_free_apg_velocity(
                diffusion=runtime.model.diff_dec,
                transformer=runtime._transformer,
                query_state=spatial_state,
                condition_prompt_embeds=prompt_bank[appearance.appearance_id]["action"],
                negative_prompt_embeds=negative,
                timestep=timestep,
                sigma=sigmas[step_index],
                branch="anchor_action_trajectory",
                adapter_controller=runtime.model,
            )
            if step_index in selected_by_index:
                cell = selected_by_index[step_index]
                with torch.inference_mode():
                    noisy_tokens, rotary = runtime._transformer.patch_vae_latent(
                        spatial_state.to(dtype=compute_dtype), source_id=cdf.QUERY_ID
                    )
                noisy_tokens = noisy_tokens.detach().contiguous()
                rotary = rotary.detach()
                timestep_object = timestep.expand(1)
                model_prompts = {
                    arm: (
                        "You are a helpful assistant specialized in text-to-video generation."
                        + __import__(
                            "diffusers.pipelines.wan.pipeline_wan",
                            fromlist=["prompt_clean"],
                        ).prompt_clean(appearance.captions[arm])
                    )
                    for arm in registry.ARMS
                }
                authority = native.seal_role_matched_t2v_four_arm_forward(
                    appearance_id=appearance.appearance_id,
                    sigma_cell=cell,
                    shared_step=runtime.model.diff_dec.shared_step,
                    action_kwargs={
                        "model_id": "transformer_1",
                        "noisy_latents": noisy_tokens,
                        "timesteps": timestep_object,
                        "cond_embeds": prompt_bank[appearance.appearance_id]["action"],
                        "rotary_embs": rotary,
                        "batch_vae_seqlen": [int(noisy_tokens.shape[1])],
                        "batch_text_seqlen": [TEXT_LENGTH],
                    },
                    prompt_embeds=prompt_bank[appearance.appearance_id],
                    instructions=model_prompts,
                    role_phrases=appearance.role_phrases,
                )
                authority_receipts.append(dict(authority.receipt()))

                # B0 is explicit and has no hook or graph observation.
                with torch.inference_mode():
                    base_output = authority.call("action").detach()
                base_sha = parity._tensor_digest(
                    base_output, label="B0 frozen base output"
                )
                handle = attention_hook.install_native_relational_attention_hook(
                    runtime._transformer, rank_bank=rank_bank
                )
                arm_rows = []
                action_observed = None
                try:
                    for arm in registry.ARMS:
                        arm_output, arm_receipt = _capture_arm(
                            runtime=runtime,
                            authority=authority,
                            arm=arm,
                            partition=partitions[appearance.appearance_id][arm],
                            rank_bank=rank_bank,
                            native_bank=native_bank,
                            stream=stream,
                            roles=roles,
                            rank=rank,
                        )
                        arm_rows.append(dict(arm_receipt))
                        if arm == "action":
                            action_observed = arm_output
                        else:
                            del arm_output
                finally:
                    handle.restore()
                if action_observed is None:
                    raise AUHSelfGeneratedRelationalProbeError("action observer output is absent")
                action_sha = parity._tensor_digest(
                    action_observed, label="observer action output"
                )
                if not torch.equal(base_output, action_observed) or base_sha != action_sha:
                    raise AUHSelfGeneratedRelationalProbeError(
                        "observer changed the frozen-base action output"
                    )
                if any(
                    wrapper.base_calls != len(registry.ARMS)
                    or wrapper.observer_calls != len(registry.ARMS)
                    for wrapper in (*handle.attn1_wrappers, *handle.attn2_wrappers)
                ):
                    raise AUHSelfGeneratedRelationalProbeError("hook call count differs")
                base_row = {
                    "arm": "B0_FROZEN_BASE",
                    "appearance_id": appearance.appearance_id,
                    "sigma_band": cell.band,
                    "step_index": cell.step_index,
                    "sigma": cell.sigma,
                    "output_sha256": base_sha,
                    "observer_action_output_sha256": action_sha,
                    "observer_action_bit_exact": True,
                    "graph_observation_supplied": False,
                    "graph_success": None,
                    "used_as_graph_positive": False,
                }
                base_cells.append({**base_row, "digest": registry.object_sha256(base_row)})
                capture_receipts.extend(arm_rows)
                selected_seen.append(step_index)
                del base_output, action_observed, noisy_tokens, rotary

            state_packed = trajectory._native_unipc_step(
                solver,
                velocity_packed=velocity,
                timestep=timestep,
                state_packed=state_packed,
            )
            trajectory_steps.append(
                {
                    "appearance_id": appearance.appearance_id,
                    "step_index": step_index,
                    "sigma": float(sigmas[step_index].item()),
                    "captured": step_index in selected_by_index,
                }
            )
            del spatial_state, velocity
        if selected_seen != [
            int(registry.SIGMA_CELL_INDICES[band]) for band in native.SIGMA_BAND_ORDER
        ]:
            raise AUHSelfGeneratedRelationalProbeError("appearance capture cells differ")
        del state_packed, solver

    relational = native._finalize_streaming_observer(stream)
    rank_bank_receipt = rank_bank.receipt()
    native_bank_receipt = native_bank.receipt()
    state_after = parity._module_state_version_receipt(runtime._transformer)
    if state_after != state_before:
        raise AUHSelfGeneratedRelationalProbeError("frozen transformer state changed")
    if (
        native_bank_receipt["capture_count"] != EXPECTED_CAPTURE_COUNT
        or native_bank_receipt["zeroized_count"] != EXPECTED_CAPTURE_COUNT
        or native_bank_receipt["resident_invocation_count"] != 0
        or rank_bank_receipt["resident_rank_invocations"] != 0
    ):
        raise AUHSelfGeneratedRelationalProbeError("capture banks did not close")
    if len(base_cells) != 9 or len(authority_receipts) != 9 or len(capture_receipts) != 36:
        raise AUHSelfGeneratedRelationalProbeError("probe receipt matrix differs")

    relational_digest = relational.get("representation_digest")
    _require_all_rank_equal(relational_digest, label="relational representation")
    local_summary = {
        "rank": rank,
        "frozen_state_digest": state_before["digest"],
        "relational_representation_digest": relational_digest,
        "base_cell_digest": registry.object_sha256(base_cells),
        "capture_bank_digest": native_bank_receipt["digest"],
    }
    local_summary = {**local_summary, "digest": registry.object_sha256(local_summary)}
    rank_summaries = _all_rank_rows(local_summary)

    status = (
        "REAL_INTERMEDIATE_REPRESENTATION_MECHANICALLY_ADMITTED_NOT_CAUSAL"
        if relational.get("status") == "MECHANICALLY_ADMITTED"
        else "REAL_INTERMEDIATE_REPRESENTATION_REJECTED"
    )
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "contract": dict(probe_contract()),
        "site_binding": binding_receipt,
        "site_bootstrap": dict(bootstrap_scrub),
        "initial_gaussian": dict(gaussian_receipt),
        "prompt_bank": dict(prompt_receipt),
        "capture_plan": dict(plan.receipt()),
        "trajectory_step_registry": trajectory_steps,
        "trajectory_model_forward_count": 3 * 40 * 2,
        "trajectory_unipc_step_count": 3 * 40,
        "frozen_base_probe_forward_count": 3 * 3,
        "observer_probe_forward_count": 3 * 3 * 4,
        "total_frozen_transformer_forward_count": 3 * 40 * 2 + 3 * 3 * 5,
        "frozen_base_cells": base_cells,
        "role_matched_four_arm_authorities": authority_receipts,
        "capture_receipts": capture_receipts,
        "rank_summaries": rank_summaries,
        "capture_bank": dict(native_bank_receipt),
        "rank_capture_bank": dict(rank_bank_receipt),
        "relational_observer": dict(relational),
        "frozen_transformer_state_unchanged": True,
        "frozen_transformer_state_digest": state_before["digest"],
        "all_nine_observer_action_outputs_equal_frozen_base_bit_exact": True,
        "frozen_base_graph_observation_supplied": False,
        "raw_qk_and_role_proxies_zeroized": True,
        "persistent_raw_tensor_artifact_created": False,
        "source_bootstrap_tensor_consumed_by_probe_forward": False,
        "target_inputs_consumed": False,
        "final_anchor_video_decoded": False,
        "output_video_created": False,
        "decoder_called": False,
        "adapter_or_lora_loaded": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "route_or_injection_called": False,
        "candidate_output_modified": False,
        "backend_attention_weights_observed": False,
        "mechanical_representation_admission_passed": (
            relational.get("status") == "MECHANICALLY_ADMITTED"
        ),
        "causal_generation_executed": False,
        "scientific_claim_authorized": False,
        "stable_transferable_action_representation_claimed": False,
    }
    value = {**value, "digest": registry.object_sha256(value)}

    write_status: list[Any] = [None]
    if rank == 0:
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ) + "\n"
            with output.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            write_status[0] = {"ok": True, "receipt_digest": value["digest"]}
        except Exception as error:
            write_status[0] = {
                "ok": False,
                "type": type(error).__name__,
                "message": str(error),
            }
    dist.broadcast_object_list(write_status, src=0)
    if not isinstance(write_status[0], Mapping) or write_status[0].get("ok") is not True:
        raise AUHSelfGeneratedRelationalProbeError(
            f"create-only receipt write failed: {write_status[0]}"
        )
    dist.barrier()
    return value


def _initialize_world4() -> None:
    if dist.is_initialized():
        raise AUHSelfGeneratedRelationalProbeError("process group initialized too early")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise AUHSelfGeneratedRelationalProbeError("live probe requires torchrun WORLD4")
    dist.init_process_group(backend="nccl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contract", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_contract:
        if args.output is not None:
            raise AUHSelfGeneratedRelationalProbeError("contract print accepts no output")
        sys.stdout.write(
            json.dumps(
                {
                    "contract": probe_contract(),
                    "remote_launch_template": remote_launch_template(),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ) + "\n"
        )
        return 0
    if args.output is None:
        raise AUHSelfGeneratedRelationalProbeError("live probe requires --output")
    _initialize_world4()
    try:
        run_real_world4_probe(args.output)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
