#!/usr/bin/env python3
"""Real WORLD4 branch-trajectory support-frame representation probe.

For every appearance, action/noop/reverse/static independently evolve from
the same authenticated source-independent Gaussian with their own exact40
UniPC history and their own negative/positive APG forwards.  At pre-registered
steps 18/32/38, each branch state receives an explicit observer-absent Frozen
Base forward and an observer-present replay on the exact same tensor objects.

The observer reduces ephemeral middle-layer Q/K role proxies to a canonical
support-frame object graph and immediately scrubs all raw tensors.  There is no
target, source tensor in a probe forward, final video, decoder, optimizer,
adapter, route, injection or parameter update.  This can only admit a
mechanical representation candidate; causal editing is a later experiment.
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
import auh_self_generated_relational_t2v_trajectory_probe_v2 as v2  # noqa: E402
import differential_sampler as cdf  # noqa: E402
import infer_native_self_generated_relational_graph_observer_v1 as native  # noqa: E402
import native_relational_attention_hook_v1 as attention_hook  # noqa: E402
import self_generated_branch_interaction_graph_observer_v3 as graph  # noqa: E402
import self_generated_intermediate_action_anchor_v1 as anchor_core  # noqa: E402
import self_generated_relational_action_graph_observer_v1 as legacy_observer  # noqa: E402
import self_generated_relational_t2v_probe_registry_v3 as registry  # noqa: E402


METHOD = "bernini-auh-self-generated-branch-graph-trajectory-probe-v3"
SCHEMA_VERSION = "bernini-auh-self-generated-branch-graph-trajectory-probe-v3"
WORLD_SIZE = 4
TEXT_LENGTH = v2.TEXT_LENGTH
EXPECTED_CAPTURE_COUNT = 144
EXPECTED_FROZEN_BASE_CELLS = 36
EXPECTED_TRAJECTORY_FORWARDS = 3 * 4 * 40 * 2
EXPECTED_TRAJECTORY_STEPS = 3 * 4 * 40
EXPECTED_TOTAL_FORWARDS = EXPECTED_TRAJECTORY_FORWARDS + 36 + 36


class AUHBranchGraphProbeError(RuntimeError):
    """Fail-closed runtime, state-lineage, parity or representation violation."""


def source_manifest() -> Mapping[str, Any]:
    modules = (
        ("runner", sys.modules[__name__]),
        ("branch_graph_observer", graph),
        ("v3_registry", registry),
        ("trajectory_helper", trajectory),
        ("v2_runtime_helpers", v2),
        ("native_capture", native),
        ("native_attention_hook", attention_hook),
        ("parity_helper", parity),
        ("differential_sampler", cdf),
        ("anchor_core", anchor_core),
        ("legacy_capture_schema", legacy_observer),
        ("site_adapter", v2.site),
    )
    rows = []
    names = set()
    for role, module in modules:
        path = Path(module.__file__).resolve()
        if not path.is_file() or path.is_symlink() or path.name in names:
            raise AUHBranchGraphProbeError("source manifest path differs")
        names.add(path.name)
        payload = path.read_bytes()
        rows.append(
            {
                "role": role,
                "file": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    value = {
        "files": rows,
        "file_count": len(rows),
        "all_plain_nonsymlink_files": True,
    }
    return {**value, "digest": registry.object_sha256(value)}


def probe_contract() -> Mapping[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "source_manifest": dict(source_manifest()),
        "registry": dict(registry.registry_receipt()),
        "checkpoint_tree_sha256": v2.site.CHECKPOINT_TREE_SHA256,
        "official_transformer_source_sha256": (
            attention_hook.OFFICIAL_TRANSFORMER_SOURCE_SHA256
        ),
        "world_size": WORLD_SIZE,
        "num_inference_steps": 40,
        "guidance": "official_t2v_apg_momentum_zero",
        "trajectory": "four_independent_explicit_bernini_field_unipc_histories",
        "auditable_attempt": "v3_r2_or_later",
        "earlier_v3_r1_diagnostic_authorized_for_admission": False,
        "same_initial_gaussian_across_all_branches": True,
        "same_state_prompt_overlay_used": False,
        "branch_negative_prediction_shared_after_divergence": False,
        "capture_steps": dict(registry.SIGMA_CELL_INDICES),
        "capture_blocks": list(native.BLOCKS),
        "capture_count": EXPECTED_CAPTURE_COUNT,
        "frozen_base_arm": "B0_FROZEN_BASE_OBSERVER_ABSENT",
        "frozen_base_per_appearance_arm_sigma": True,
        "frozen_base_cell_count": EXPECTED_FROZEN_BASE_CELLS,
        "frozen_base_graph_observation_supplied": False,
        "observer_output_must_equal_own_frozen_base_bit_exact": True,
        "support_frame_reference_edge_contributes_to_reward": False,
        "appearance_transfer_thresholds_unchanged_from_v2": {
            "cosine_min": 0.95,
            "distance_max": 0.15,
        },
        "source_bootstrap_tensor_consumed_by_trajectory_or_probe_forward": False,
        "target_inputs_consumed": False,
        "final_anchor_video_decode": False,
        "decoder_available_to_probe": False,
        "adapter_or_lora_loaded": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "route_or_injection_called": False,
        "candidate_output_modified": False,
        "persistent_raw_tensor_artifact": False,
        "official_sampler_terminal_parity_executed": False,
        "strict_native_sampler_trajectory_claimed": False,
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
        "launch_executed": False,
    }
    return {**value, "digest": registry.object_sha256(value)}


def _check_world4() -> tuple[int, int]:
    try:
        return v2._check_world4()
    except Exception as error:
        raise AUHBranchGraphProbeError(str(error)) from error


def _require_all_rank_equal(value: Any, *, label: str) -> None:
    try:
        v2._require_all_rank_equal(value, label=label)
    except Exception as error:
        raise AUHBranchGraphProbeError(str(error)) from error


def _roles_and_stream() -> tuple[tuple[Any, ...], graph.StreamingBranchInteractionGraphObserver]:
    roles = tuple(
        legacy_observer.RoleSpec(
            role_id,
            "self_generated_anchor_owned",
            semantic_role={
                "agent": "human_agent",
                "moving_object": "moving_object",
                "start_support": "support_surface",
                "end_support": "support_surface",
                "null_context": "distractor",
            }[role_id],
            critical=role_id != "null_context",
        )
        for role_id in registry.ROLE_IDS
    )
    return roles, graph.StreamingBranchInteractionGraphObserver(roles=roles)


def _model_prompts() -> Mapping[str, Mapping[str, str]]:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    prefix = "You are a helpful assistant specialized in text-to-video generation."
    return {
        appearance.appearance_id: {
            arm: prefix + prompt_clean(appearance.captions[arm])
            for arm in registry.ARMS
        }
        for appearance in registry.APPEARANCES
    }


def _validate_canonical_partition_receipt(prompt_receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = prompt_receipt.get("prompt_receipts")
    if not isinstance(rows, Mapping):
        raise AUHBranchGraphProbeError("prompt partition receipt is absent")
    core_roles = tuple(registry.ROLE_IDS[:-1])
    checks = []
    for arm in registry.ARMS:
        signatures = []
        for appearance in registry.APPEARANCE_IDS:
            value = rows[appearance][arm]
            signatures.append(tuple(int(value["role_token_counts"][role]) for role in core_roles))
        if len(set(signatures)) != 1:
            raise AUHBranchGraphProbeError(
                f"canonical role token capacity differs across appearances for {arm}"
            )
        checks.append({"arm": arm, "core_role_token_counts": list(signatures[0])})
    value = {
        "canonical_role_aliases": dict(registry.ROLE_PHRASES),
        "core_role_token_counts_equal_across_appearances_per_arm": True,
        "appearance_words_owned_by_null_context": True,
        "rows": checks,
    }
    return {**value, "digest": registry.object_sha256(value)}


def _branch_kwargs(
    *,
    runtime: Any,
    spatial_state: torch.Tensor,
    timestep: torch.Tensor,
    prompt: torch.Tensor,
) -> tuple[Mapping[str, Any], Mapping[str, str]]:
    compute_dtype = runtime._transformer.patch_embedding.weight.dtype
    with torch.inference_mode():
        noisy_tokens, rotary = runtime._transformer.patch_vae_latent(
            spatial_state.to(dtype=compute_dtype), source_id=cdf.QUERY_ID
        )
    noisy_tokens = noisy_tokens.detach().contiguous()
    rotary = rotary.detach()
    timestep_object = timestep.expand(1)
    kwargs = {
        "model_id": "transformer_1",
        "noisy_latents": noisy_tokens,
        "timesteps": timestep_object,
        "cond_embeds": prompt,
        "rotary_embs": rotary,
        "batch_vae_seqlen": [int(noisy_tokens.shape[1])],
        "batch_text_seqlen": [TEXT_LENGTH],
    }
    anchor_core.assert_target_isolation_payload(kwargs, path="branch_graph_probe")
    digests = {
        name: anchor_core.tensor_sha256(kwargs[name])
        for name in ("noisy_latents", "timesteps", "rotary_embs", "cond_embeds")
    }
    return kwargs, digests


def _capture_branch(
    *,
    runtime: Any,
    appearance_id: str,
    arm: str,
    cell: native.SigmaCell,
    kwargs: Mapping[str, Any],
    tensor_digests: Mapping[str, str],
    prompt_sha256: str,
    partition: attention_hook.ExhaustiveTextRolePartition,
    rank_bank: attention_hook.InMemoryWorld4RankShardBank,
    native_bank: native.InMemoryNativeCaptureBank,
    stream: graph.StreamingBranchInteractionGraphObserver,
    roles: Sequence[Any],
    rank: int,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    invocation = native.CaptureInvocation(
        appearance_id,
        arm,
        cell,
        tensor_digests["noisy_latents"],
        tensor_digests["timesteps"],
        tensor_digests["rotary_embs"],
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
        output = runtime.model.diff_dec.shared_step(**kwargs).detach()
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
    capture_summary = [
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
        observer=legacy_observer,
        stream=stream,
        captures=captures,
        roles=roles,
        prompt_sha256=prompt_sha256,
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
        raise AUHBranchGraphProbeError("upstream capture did not zeroize")
    receipt = {
        "appearance_id": appearance_id,
        "arm": arm,
        "sigma_band": cell.band,
        "step_index": cell.step_index,
        "branch_state_sha256": invocation.state_sha256,
        "state_tensor_sha256": dict(tensor_digests),
        "prompt_embedding_sha256": prompt_sha256,
        "role_partition_sha256": partition.digest,
        "output_sha256": parity._tensor_digest(output, label=f"{arm} observer output"),
        "commit": dict(commit),
        "capture_summary": capture_summary,
        "raw_capture_zeroized": True,
    }
    return output, {**receipt, "digest": registry.object_sha256(receipt)}


def run_real_world4_probe(output: Path) -> Mapping[str, Any]:
    rank, _local_rank = _check_world4()
    if not output.is_absolute() or output.is_symlink() or output.exists():
        raise AUHBranchGraphProbeError("output must be an absolute absent plain path")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise AUHBranchGraphProbeError("output parent must be a plain directory")

    runtime = v2.site.create_auh_bernini_source_role_adapter({})
    binding_receipt = runtime.binding_receipt()
    prompt_bank, negative, partitions, prompt_receipt = v2._encode_prompt_bank(
        runtime,
        rank=rank,
        registry_module=registry,
    )
    partition_receipt = _validate_canonical_partition_receipt(prompt_receipt)
    bootstrap_scrub = v2._scrub_site_bootstrap_tensors(runtime)
    initial_gaussian, gaussian_receipt = v2._load_initial_gaussian(runtime)
    plan, timesteps, sigmas = v2._real_capture_plan(
        runtime.model.diff_dec, runtime.device
    )
    state_before = parity._module_state_version_receipt(runtime._transformer)
    roles, stream = _roles_and_stream()
    rank_bank = attention_hook.InMemoryWorld4RankShardBank()
    native_bank = native.InMemoryNativeCaptureBank()
    model_prompts = _model_prompts()

    layout = cdf.validate_latent_shape(tuple(initial_gaussian.shape))
    initial_packed = cdf._pack_spatial_latent(initial_gaussian, layout)
    initial_packed_sha256 = anchor_core.tensor_sha256(initial_packed)
    _require_all_rank_equal(initial_packed_sha256, label="initial packed Gaussian")
    selected_by_index = {cell.step_index: cell for cell in plan.sigma_cells}
    frozen_base_cells = []
    capture_receipts = []
    trajectory_rows = []
    terminal_rows = []

    for appearance in registry.APPEARANCES:
        for arm in registry.ARMS:
            state_packed = initial_packed.clone()
            if anchor_core.tensor_sha256(state_packed) != initial_packed_sha256:
                raise AUHBranchGraphProbeError("branch initial state differs")
            solver = copy.deepcopy(runtime.model.diff_dec.scheduler)
            solver.set_timesteps(40)
            selected_seen = []
            branch_rows = []
            for step_index, timestep in enumerate(timesteps):
                pre_state_sha256 = anchor_core.tensor_sha256(state_packed)
                spatial_state = cdf._unpack_spatial_latent(state_packed, layout)
                velocity = trajectory._guided_source_free_apg_velocity(
                    diffusion=runtime.model.diff_dec,
                    transformer=runtime._transformer,
                    query_state=spatial_state,
                    condition_prompt_embeds=prompt_bank[appearance.appearance_id][arm],
                    negative_prompt_embeds=negative,
                    timestep=timestep,
                    sigma=sigmas[step_index],
                    branch=f"anchor_{arm}_trajectory",
                    adapter_controller=runtime.model,
                )
                captured = step_index in selected_by_index
                capture_state_digest = None
                if captured:
                    cell = selected_by_index[step_index]
                    kwargs, tensor_digests = _branch_kwargs(
                        runtime=runtime,
                        spatial_state=spatial_state,
                        timestep=timestep,
                        prompt=prompt_bank[appearance.appearance_id][arm],
                    )
                    capture_state_digest = registry.object_sha256(
                        {
                            "noisy_state_sha256": tensor_digests["noisy_latents"],
                            "timestep_sha256": tensor_digests["timesteps"],
                            "rotary_sha256": tensor_digests["rotary_embs"],
                        }
                    )
                    _require_all_rank_equal(
                        capture_state_digest,
                        label=f"{appearance.appearance_id}:{arm}:{cell.band} branch state",
                    )

                    # Frozen Base is an explicit hook-absent forward for every
                    # branch state, never a graph-positive or an admission arm.
                    with torch.inference_mode():
                        base_output = runtime.model.diff_dec.shared_step(**kwargs).detach()
                    base_sha = parity._tensor_digest(
                        base_output, label=f"{arm} B0 frozen base output"
                    )
                    handle = attention_hook.install_native_relational_attention_hook(
                        runtime._transformer, rank_bank=rank_bank
                    )
                    try:
                        observed_output, observed_receipt = _capture_branch(
                            runtime=runtime,
                            appearance_id=appearance.appearance_id,
                            arm=arm,
                            cell=cell,
                            kwargs=kwargs,
                            tensor_digests=tensor_digests,
                            prompt_sha256=tensor_digests["cond_embeds"],
                            partition=partitions[appearance.appearance_id][arm],
                            rank_bank=rank_bank,
                            native_bank=native_bank,
                            stream=stream,
                            roles=roles,
                            rank=rank,
                        )
                    finally:
                        handle.restore()
                    observed_sha = parity._tensor_digest(
                        observed_output, label=f"{arm} observed output"
                    )
                    if (
                        not torch.equal(base_output, observed_output)
                        or base_sha != observed_sha
                    ):
                        raise AUHBranchGraphProbeError(
                            "observer changed its own Frozen Base output"
                        )
                    if any(
                        wrapper.base_calls != 1 or wrapper.observer_calls != 1
                        for wrapper in (*handle.attn1_wrappers, *handle.attn2_wrappers)
                    ):
                        raise AUHBranchGraphProbeError("hook call count differs")
                    for name in ("noisy_latents", "timesteps", "rotary_embs", "cond_embeds"):
                        if anchor_core.tensor_sha256(kwargs[name]) != tensor_digests[name]:
                            raise AUHBranchGraphProbeError(
                                f"Frozen Base replay mutated {name}"
                            )
                    base_row = {
                        "arm": "B0_FROZEN_BASE_OBSERVER_ABSENT",
                        "trajectory_arm": arm,
                        "appearance_id": appearance.appearance_id,
                        "sigma_band": cell.band,
                        "step_index": cell.step_index,
                        "sigma": cell.sigma,
                        "branch_state_sha256": capture_state_digest,
                        "output_sha256": base_sha,
                        "observer_output_sha256": observed_sha,
                        "observer_output_bit_exact": True,
                        "graph_observation_supplied": False,
                        "graph_success": None,
                        "used_as_graph_positive": False,
                    }
                    frozen_base_cells.append(
                        {**base_row, "digest": registry.object_sha256(base_row)}
                    )
                    capture_receipts.append(dict(observed_receipt))
                    selected_seen.append(step_index)
                    del kwargs, base_output, observed_output

                state_packed = trajectory._native_unipc_step(
                    solver,
                    velocity_packed=velocity,
                    timestep=timestep,
                    state_packed=state_packed,
                )
                post_state_sha256 = anchor_core.tensor_sha256(state_packed)
                branch_rows.append(
                    {
                        "step_index": step_index,
                        "timestep_sha256": anchor_core.tensor_sha256(timestep),
                        "sigma": float(sigmas[step_index].item()),
                        "pre_state_sha256": pre_state_sha256,
                        "post_state_sha256": post_state_sha256,
                        "captured": captured,
                        "capture_state_sha256": capture_state_digest,
                    }
                )
                del spatial_state, velocity
            expected_selected = [
                int(registry.SIGMA_CELL_INDICES[band])
                for band in native.SIGMA_BAND_ORDER
            ]
            if selected_seen != expected_selected:
                raise AUHBranchGraphProbeError("branch capture cells differ")
            branch_value = {
                "appearance_id": appearance.appearance_id,
                "arm": arm,
                "initial_packed_state_sha256": initial_packed_sha256,
                "prompt_embedding_sha256": anchor_core.tensor_sha256(
                    prompt_bank[appearance.appearance_id][arm]
                ),
                "independent_solver_history": True,
                "negative_prediction_shared_after_divergence": False,
                "rows": branch_rows,
            }
            branch_value = {
                **branch_value,
                "digest": registry.object_sha256(branch_value),
            }
            trajectory_rows.append(branch_value)
            terminal_rows.append(
                {
                    "appearance_id": appearance.appearance_id,
                    "arm": arm,
                    "terminal_predecode_packed_state_sha256": anchor_core.tensor_sha256(
                        state_packed
                    ),
                    "decoded": False,
                }
            )
            del state_packed, solver

    trajectory_registry_digest = registry.object_sha256(trajectory_rows)
    _require_all_rank_equal(
        trajectory_registry_digest, label="all branch trajectory lineages"
    )
    relational = stream.finalize()
    relational_digest = relational.get("representation_digest")
    _require_all_rank_equal(relational_digest, label="branch graph representation")
    rank_bank_receipt = rank_bank.receipt()
    native_bank_receipt = native_bank.receipt()
    state_after = parity._module_state_version_receipt(runtime._transformer)
    if state_after != state_before:
        raise AUHBranchGraphProbeError("frozen transformer state changed")
    if (
        native_bank_receipt["capture_count"] != EXPECTED_CAPTURE_COUNT
        or native_bank_receipt["zeroized_count"] != EXPECTED_CAPTURE_COUNT
        or native_bank_receipt["resident_invocation_count"] != 0
        or rank_bank_receipt["resident_rank_invocations"] != 0
    ):
        raise AUHBranchGraphProbeError("capture banks did not close")
    if (
        len(frozen_base_cells) != EXPECTED_FROZEN_BASE_CELLS
        or len(capture_receipts) != EXPECTED_FROZEN_BASE_CELLS
        or len(trajectory_rows) != 12
        or len(terminal_rows) != 12
    ):
        raise AUHBranchGraphProbeError("probe receipt matrix differs")

    local_summary = {
        "rank": rank,
        "frozen_state_digest": state_before["digest"],
        "trajectory_registry_digest": trajectory_registry_digest,
        "relational_representation_digest": relational_digest,
        "frozen_base_cell_digest": registry.object_sha256(frozen_base_cells),
        "capture_bank_digest": native_bank_receipt["digest"],
    }
    local_summary = {
        **local_summary,
        "digest": registry.object_sha256(local_summary),
    }
    rank_summaries = v2._all_rank_rows(local_summary)
    mechanically_admitted = relational.get("status") == "MECHANICALLY_ADMITTED"
    status = (
        "REAL_BRANCH_GRAPH_REPRESENTATION_MECHANICALLY_ADMITTED_NOT_CAUSAL"
        if mechanically_admitted
        else "REAL_BRANCH_GRAPH_REPRESENTATION_REJECTED"
    )
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "contract": dict(probe_contract()),
        "source_manifest": dict(source_manifest()),
        "site_binding": binding_receipt,
        "site_bootstrap": dict(bootstrap_scrub),
        "initial_gaussian": dict(gaussian_receipt),
        "initial_packed_state_sha256": initial_packed_sha256,
        "prompt_bank": dict(prompt_receipt),
        "canonical_partition": dict(partition_receipt),
        "capture_plan": dict(plan.receipt()),
        "trajectory_lineages": trajectory_rows,
        "trajectory_registry_digest": trajectory_registry_digest,
        "terminal_predecode_states": terminal_rows,
        "trajectory_model_forward_count": EXPECTED_TRAJECTORY_FORWARDS,
        "trajectory_unipc_step_count": EXPECTED_TRAJECTORY_STEPS,
        "frozen_base_probe_forward_count": EXPECTED_FROZEN_BASE_CELLS,
        "observer_probe_forward_count": EXPECTED_FROZEN_BASE_CELLS,
        "total_frozen_transformer_forward_count": EXPECTED_TOTAL_FORWARDS,
        "frozen_base_cells": frozen_base_cells,
        "capture_receipts": capture_receipts,
        "rank_summaries": rank_summaries,
        "capture_bank": dict(native_bank_receipt),
        "rank_capture_bank": dict(rank_bank_receipt),
        "branch_interaction_graph_observer": dict(relational),
        "frozen_transformer_state_unchanged": True,
        "frozen_transformer_state_digest": state_before["digest"],
        "all_36_observer_outputs_equal_own_frozen_base_bit_exact": True,
        "frozen_base_graph_observation_supplied": False,
        "raw_qk_and_role_proxies_zeroized": True,
        "persistent_raw_tensor_artifact_created": False,
        "source_bootstrap_tensor_consumed_by_trajectory_or_probe_forward": False,
        "target_inputs_consumed": False,
        "final_anchor_video_decoded": False,
        "output_video_created": False,
        "decoder_called": False,
        "adapter_or_lora_loaded": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "route_or_injection_called": False,
        "candidate_output_modified": False,
        "official_sampler_terminal_parity_executed": False,
        "strict_native_sampler_trajectory_claimed": False,
        "mechanical_representation_admission_passed": mechanically_admitted,
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
        raise AUHBranchGraphProbeError(
            f"create-only receipt write failed: {write_status[0]}"
        )
    dist.barrier()
    return value


def _initialize_world4() -> None:
    if dist.is_initialized():
        raise AUHBranchGraphProbeError("process group initialized too early")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise AUHBranchGraphProbeError("live probe requires torchrun WORLD4")
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
            raise AUHBranchGraphProbeError("contract print accepts no output")
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
        raise AUHBranchGraphProbeError("live probe requires --output")
    _initialize_world4()
    try:
        run_real_world4_probe(args.output)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
