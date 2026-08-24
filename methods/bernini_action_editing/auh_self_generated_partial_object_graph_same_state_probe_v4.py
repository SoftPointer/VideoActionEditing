#!/usr/bin/env python3
"""Real WORLD4 same-state partial-object-graph observer probe, v4.

Three source-independent action-prompt trajectories are evolved from one
authenticated Gaussian.  At UniPC indices 18/32/38, an explicit observer-
absent Frozen Base action forward is followed by action/noop/reverse/static
observer forwards on the exact same state.  Each prompt arm's native post-
RoPE Q/K and derived role proxy is reduced and zeroized before the next arm is
allowed to run.  Four arms of raw Q/K are therefore never resident together.

This entry point has no decoder, renderer, route, optimizer, target input or
parameter-update path.  Even a positive four-arm component result is not a
stable/transferable action-representation admission: the shuffled-prompt,
persistent identity and contact-state gates are not executed here.
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
import self_generated_partial_object_graph_observer_v4 as graph  # noqa: E402
import self_generated_partial_object_graph_registry_v4 as graph_registry_v4  # noqa: E402
import self_generated_relational_t2v_probe_registry_v3 as prompt_registry  # noqa: E402


METHOD = "bernini-auh-self-generated-partial-object-graph-same-state-probe-v4"
SCHEMA_VERSION = METHOD
WORLD_SIZE = 4
TEXT_LENGTH = v2.TEXT_LENGTH
EXPECTED_PATCHES = 925
EXPECTED_HEADS = 12
EXPECTED_HEAD_DIM = 128
EXPECTED_PHASES = 21
EXPECTED_CAPTURE_COUNT = 3 * 3 * 4 * 4
EXPECTED_FROZEN_BASE_CELLS = 3 * 3
EXPECTED_OBSERVER_FORWARDS = 3 * 3 * 4
EXPECTED_TRAJECTORY_FORWARDS = 3 * 40 * 2
EXPECTED_TRAJECTORY_STEPS = 3 * 40
EXPECTED_TOTAL_FORWARDS = (
    EXPECTED_TRAJECTORY_FORWARDS
    + EXPECTED_FROZEN_BASE_CELLS
    + EXPECTED_OBSERVER_FORWARDS
)


class AUHPartialObjectGraphProbeV4Error(RuntimeError):
    """A WORLD4, provenance, residency, parity or claim gate failed."""


def _rank_wrapper_path() -> Path:
    name = (
        "auh_self_generated_partial_object_graph_same_state_"
        "probe_rank_wrapper_v4.sh"
    )
    candidates = (
        METHOD_ROOT / "scripts" / name,
        METHOD_ROOT.parent / "scripts" / name,
    )
    rows = [path.resolve() for path in candidates if path.is_file()]
    if len(rows) != 1 or rows[0].is_symlink():
        raise AUHPartialObjectGraphProbeV4Error(
            "rank wrapper source authority differs"
        )
    return rows[0]


def make_graph_registry() -> graph_registry_v4.ObserverRegistryV4:
    """Bind all v3 role aliases to an explicit v4 partial graph."""

    roles = (
        graph_registry_v4.RoleSpecV4("agent", "actor_root"),
        graph_registry_v4.RoleSpecV4(
            "moving_object", "manipulated_object"
        ),
        graph_registry_v4.RoleSpecV4(
            "start_support",
            "support_surface",
            support_frame_role="start",
        ),
        graph_registry_v4.RoleSpecV4(
            "end_support",
            "support_surface",
            support_frame_role="end",
        ),
        # This is the exhaustive text partition's residual role.  It may
        # receive text-derived score but can never be declared a visual slot.
        graph_registry_v4.RoleSpecV4(
            "null_context",
            "context_residual",
            evidence_mode="instruction_only",
            critical=False,
        ),
    )
    edges = (
        graph_registry_v4.EdgeSpecV4(
            "agent", "moving_object", "latent_affinity"
        ),
        graph_registry_v4.EdgeSpecV4(
            "moving_object", "start_support", "receding"
        ),
        graph_registry_v4.EdgeSpecV4(
            "moving_object", "end_support", "approaching"
        ),
        graph_registry_v4.EdgeSpecV4(
            "null_context",
            "moving_object",
            "instruction_relation_unresolved",
            critical=False,
        ),
    )
    value = graph_registry_v4.make_registry_v4(
        roles,
        edges,
        phases=EXPECTED_PHASES,
        requires_support_frame=True,
    )
    if value.role_ids != prompt_registry.ROLE_IDS:
        raise AUHPartialObjectGraphProbeV4Error(
            "graph roles differ from native role-proxy order"
        )
    return value


def source_manifest() -> Mapping[str, Any]:
    modules = (
        ("runner", sys.modules[__name__]),
        ("partial_graph_observer", graph),
        ("partial_graph_registry", graph_registry_v4),
        ("v3_prompt_registry", prompt_registry),
        ("v2_runtime_helpers", v2),
        ("trajectory_helper", trajectory),
        ("native_capture", native),
        ("native_attention_hook", attention_hook),
        ("parity_helper", parity),
        ("differential_sampler", cdf),
        ("site_adapter", v2.site),
    )
    rows = []
    names = set()
    for role, module in modules:
        path = Path(module.__file__).resolve()
        if not path.is_file() or path.is_symlink() or path.name in names:
            raise AUHPartialObjectGraphProbeV4Error(
                "source manifest path differs"
            )
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
    wrapper = _rank_wrapper_path()
    payload = wrapper.read_bytes()
    if wrapper.name in names:
        raise AUHPartialObjectGraphProbeV4Error(
            "rank wrapper duplicates a source name"
        )
    rows.append(
        {
            "role": "rank_wrapper",
            "file": wrapper.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    value = {
        "files": rows,
        "file_count": len(rows),
        "all_plain_nonsymlink_files": True,
    }
    return {
        **value,
        "digest": graph_registry_v4.object_sha256(value),
    }


def probe_contract() -> Mapping[str, Any]:
    graph_registry = make_graph_registry()
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "source_manifest": dict(source_manifest()),
        "prompt_registry": dict(prompt_registry.registry_receipt()),
        "graph_registry": graph_registry.as_dict(),
        "graph_registry_digest": graph_registry.digest,
        "checkpoint_tree_sha256": v2.site.CHECKPOINT_TREE_SHA256,
        "official_transformer_source_sha256": (
            attention_hook.OFFICIAL_TRANSFORMER_SOURCE_SHA256
        ),
        "world_size": WORLD_SIZE,
        "num_inference_steps": 40,
        "guidance": "official_t2v_apg_momentum_zero",
        "trajectory": "three_action_prompt_bernini_field_unipc_histories",
        "same_state_prompt_overlay_used": True,
        "capture_steps": dict(prompt_registry.SIGMA_CELL_INDICES),
        "capture_blocks": list(native.BLOCKS),
        "capture_count": EXPECTED_CAPTURE_COUNT,
        "native_raw_qk_shape": [1, 21, 925, 12, 128],
        "native_role_proxy_shape": [1, 21, len(graph_registry.roles), 925],
        "frozen_base_arm": "B0_FROZEN_BASE_OBSERVER_ABSENT",
        "frozen_base_per_appearance_sigma": True,
        "frozen_base_cell_count": EXPECTED_FROZEN_BASE_CELLS,
        "frozen_base_graph_observation_supplied": False,
        "observer_action_must_equal_frozen_base_bit_exact": True,
        "per_arm_immediate_reduce_and_zeroize_required": True,
        "maximum_simultaneously_resident_raw_prompt_arms": 1,
        "four_arm_raw_qk_bundle_permitted": False,
        "compact_arm_evidence_may_persist_until_cell_close": True,
        "partial_assignment_has_unrestricted_dustbin": True,
        "native_proxy_simplex_required": True,
        "native_proxy_simplex_tolerance": {"atol": 2.0e-4, "rtol": 2.0e-4},
        "token_prior_correction": (
            "u=p/pi;q=u/sum_role(u);gate_q_logu_and_second_margin"
        ),
        "prior_equalized_probability_gate": True,
        "absolute_evidence_before_zscore_or_topk": True,
        "failed_absolute_evidence_kernel_exact_zero": True,
        "duplicate_role_phase_abstains": True,
        "shared_frame_sources": ["noop", "static"],
        "action_or_reverse_defines_shared_frame": False,
        "shared_frame_closure_gates": [
            "endpoint_rms",
            "direction_cosine",
            "log_scale_error",
        ],
        "failed_shared_frame_phase_abstains_all_four_arms": True,
        "four_arm_common_edge_domain_required": True,
        "reverse_endpoint_topology_gate_required": True,
        "roles_explicit": [role.as_dict() for role in graph_registry.roles],
        "edges_explicit": [edge.as_dict() for edge in graph_registry.edges],
        "shuffled_prompt_control_executed": False,
        "shuffled_prompt_robustness_claimed": False,
        "missing_shuffled_prompt_gate_counts_as_representation_failure": True,
        "component_admission_can_imply_representation_admission": False,
        "representation_admission_hard_false": True,
        "observer_only_diagnostic_launch_authorized": True,
        "launch_blocked_pending_failure_path_audit": False,
        "representation_or_renderer_launch_authorized": False,
        "source_bootstrap_tensor_consumed_by_trajectory_or_probe_forward": False,
        "target_inputs_consumed": False,
        "final_anchor_video_decode": False,
        "entrypoint_exercised_call_path_decoder_called": False,
        "claims_apply_to_exercised_entrypoint_call_path_only": True,
        "renderer_called": False,
        "adapter_or_lora_loaded": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "route_or_injection_called": False,
        "candidate_output_modified": False,
        "persistent_raw_tensor_artifact": False,
        "failure_path_raw_capture_finally_scrub_required": True,
        "successful_exit_raw_capture_zeroization_required": True,
        "explicit_capture_ownership_boundary_exception_scrub_required": True,
        "uncovered_exception_requires_nonzero_exit_without_receipt": True,
        "all_allocation_failure_zeroization_claimed": False,
        "backend_attention_weights_observed": False,
        "scientific_claim_authorized": False,
        "stable_transferable_action_representation_claimed": False,
        "causal_generation_claimed": False,
    }
    return {
        **value,
        "digest": graph_registry_v4.object_sha256(value),
    }


def remote_launch_template() -> Mapping[str, Any]:
    value = {
        "python": "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12",
        "launcher": "python -m torch.distributed.run",
        "nproc_per_node": WORLD_SIZE,
        "entrypoint": Path(__file__).name,
        "rank_wrapper": (
            "scripts/auh_self_generated_partial_object_graph_same_state_"
            "probe_rank_wrapper_v4.sh"
        ),
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
    return {
        **value,
        "digest": graph_registry_v4.object_sha256(value),
    }


def _check_world4() -> tuple[int, int]:
    try:
        return v2._check_world4()
    except Exception as error:
        raise AUHPartialObjectGraphProbeV4Error(str(error)) from error


def _require_all_rank_equal(value: Any, *, label: str) -> None:
    try:
        v2._require_all_rank_equal(value, label=label)
    except Exception as error:
        raise AUHPartialObjectGraphProbeV4Error(str(error)) from error


def _model_prompts() -> Mapping[str, Mapping[str, str]]:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    prefix = "You are a helpful assistant specialized in text-to-video generation."
    return {
        appearance.appearance_id: {
            arm: prefix + prompt_clean(appearance.captions[arm])
            for arm in prompt_registry.ARMS
        }
        for appearance in prompt_registry.APPEARANCES
    }


def _validate_canonical_partitions(
    prompt_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    rows = prompt_receipt.get("prompt_receipts")
    if not isinstance(rows, Mapping):
        raise AUHPartialObjectGraphProbeV4Error(
            "prompt partition receipt is absent"
        )
    core_roles = tuple(prompt_registry.ROLE_IDS[:-1])
    checks = []
    for arm in prompt_registry.ARMS:
        signatures = []
        for appearance in prompt_registry.APPEARANCE_IDS:
            row = rows[appearance][arm]
            signatures.append(
                tuple(int(row["role_token_counts"][role]) for role in core_roles)
            )
        if len(set(signatures)) != 1:
            raise AUHPartialObjectGraphProbeV4Error(
                f"canonical role token capacity differs for {arm}"
            )
        checks.append(
            {"arm": arm, "core_role_token_counts": list(signatures[0])}
        )
    value = {
        "canonical_role_aliases": dict(prompt_registry.ROLE_PHRASES),
        "core_role_token_counts_equal_across_appearances_per_arm": True,
        "appearance_words_owned_by_null_context": True,
        "rows": checks,
    }
    return {
        **value,
        "digest": graph_registry_v4.object_sha256(value),
    }


def _reduce_native_capture_group_finally_scrubbed(
    *,
    captures: Sequence[native.NativeBlockCapture],
    native_bank: native.InMemoryNativeCaptureBank,
    invocation: native.CaptureInvocation,
    authority: native.RoleMatchedT2VFourArmForwardAuthority,
    arm: str,
    partition: attention_hook.ExhaustiveTextRolePartition,
    assemblers: Mapping[int, graph.SameStateCellAssemblerV4],
    graph_registry: graph_registry_v4.ObserverRegistryV4,
) -> tuple[list[Mapping[str, Any]], tuple[int, ...]]:
    """Reduce one prompt arm; scrub its complete native group on every exit."""

    rows = tuple(captures)
    try:
        expected_qk = (
            1,
            EXPECTED_PHASES,
            EXPECTED_PATCHES,
            EXPECTED_HEADS,
            EXPECTED_HEAD_DIM,
        )
        expected_proxy = (
            1,
            EXPECTED_PHASES,
            len(graph_registry.roles),
            EXPECTED_PATCHES,
        )
        if tuple(partition.role_names) != graph_registry.role_ids:
            raise AUHPartialObjectGraphProbeV4Error(
                "native partition role order differs from graph registry"
            )
        role_token_counts = tuple(
            partition.token_to_role.count(index)
            for index in range(len(partition.role_names))
        )
        if (
            any(count < 1 for count in role_token_counts)
            or sum(role_token_counts) != graph_registry_v4.TEXT_TOKEN_COUNT
        ):
            raise AUHPartialObjectGraphProbeV4Error(
                "native partition token prior differs"
            )
        summary: list[Mapping[str, Any]] = []
        for capture in rows:
            if tuple(capture.query.shape) != expected_qk or (
                tuple(capture.key.shape) != expected_qk
            ):
                raise AUHPartialObjectGraphProbeV4Error(
                    "native raw Q/K geometry differs"
                )
            if tuple(
                capture.derived_qk_role_responsibility_proxy.shape
            ) != expected_proxy:
                raise AUHPartialObjectGraphProbeV4Error(
                    "native role-proxy geometry differs"
                )
            summary.append(
                {
                    "block_index": capture.block_index,
                    "query_shape": list(expected_qk),
                    "key_shape": list(expected_qk),
                    "proxy_shape": list(expected_proxy),
                }
            )
            observation = graph.MiddleObservationV4.create(
                appearance_id=invocation.appearance_id,
                arm=arm,
                sigma_band=invocation.sigma_cell.band,
                block_index=capture.block_index,
                state_sha256=invocation.state_sha256,
                timestep_sha256=invocation.timestep_sha256,
                rotary_sha256=invocation.rotary_sha256,
                prompt_sha256=authority.prompt_embedding_sha256[arm],
                role_order=partition.role_names,
                role_partition_sha256=partition.digest,
                role_token_counts=role_token_counts,
                patch_height=invocation.patch_height,
                patch_width=invocation.patch_width,
                queries=capture.query,
                keys=capture.key,
                role_scores=capture.derived_qk_role_responsibility_proxy,
                metadata={
                    "evidence_kind": (
                        "native_derived_qk_role_responsibility_proxy"
                    ),
                    "backend_attention_weights_observed": False,
                    "prompt_arm": arm,
                },
            )
            compact = graph.reduce_one_arm_v4(
                observation, graph_registry=graph_registry
            )
            if not observation.consumed or any(
                int(torch.count_nonzero(value).item()) != 0
                for value in (
                    capture.query,
                    capture.key,
                    capture.derived_qk_role_responsibility_proxy,
                )
            ):
                raise AUHPartialObjectGraphProbeV4Error(
                    "one-arm raw capture remained after compact reduction"
                )
            assemblers[capture.block_index].add(compact)
        return summary, role_token_counts
    finally:
        # This closes both resident bytes and native-bank accounting on
        # success, validation failure, reducer failure or assembler failure.
        native_bank.zeroize(rows)


def _abort_native_bank_invocation(
    bank: native.InMemoryNativeCaptureBank,
    invocation: native.CaptureInvocation,
) -> None:
    """Failure-only scrub for a committed invocation not yet consumed."""

    resident = getattr(bank, "_captures", None)
    if not isinstance(resident, dict):
        raise AUHPartialObjectGraphProbeV4Error(
            "native capture failure scrub ABI differs"
        )
    block_rows = resident.pop(invocation.key, None)
    if block_rows is None:
        return
    captures = tuple(block_rows[index] for index in sorted(block_rows))
    bank.consumed_count += len(captures)
    bank.zeroize(captures)


def _abort_rank_bank_invocation(
    bank: attention_hook.InMemoryWorld4RankShardBank,
    invocation: attention_hook.RankCaptureInvocation,
) -> None:
    """Failure-only scrub for hook rows that never reached ``take_rank``."""

    resident = getattr(bank, "_rows", None)
    if not isinstance(resident, dict):
        raise AUHPartialObjectGraphProbeV4Error(
            "rank capture failure scrub ABI differs"
        )
    block_rows = resident.pop(invocation.key, None)
    if block_rows is None:
        return
    for row in block_rows.values():
        for value in (getattr(row, "qk", None), getattr(row, "role", None)):
            if callable(getattr(value, "zeroize", None)):
                value.zeroize()


def _abort_compact_ownership(
    assemblers: Mapping[int, graph.SameStateCellAssemblerV4],
    graph_stream: graph.PartialObjectGraphObserverV4,
) -> None:
    """Close both pending arm rows and previously committed reduced cells."""

    errors: list[BaseException] = []
    for assembler in assemblers.values():
        try:
            assembler.abort()
        except BaseException as error:
            errors.append(error)
    try:
        graph_stream.abort()
    except BaseException as error:
        errors.append(error)
    if errors:
        raise AUHPartialObjectGraphProbeV4Error(
            "compact ownership failure scrub did not close"
        ) from errors[0]


def _capture_one_arm(
    *,
    runtime: Any,
    authority: native.RoleMatchedT2VFourArmForwardAuthority,
    arm: str,
    partition: attention_hook.ExhaustiveTextRolePartition,
    rank_bank: attention_hook.InMemoryWorld4RankShardBank,
    native_bank: native.InMemoryNativeCaptureBank,
    assemblers: Mapping[int, graph.SameStateCellAssemblerV4],
    graph_registry: graph_registry_v4.ObserverRegistryV4,
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
    try:
        with torch.inference_mode(), rank_bank.observe(rank_invocation):
            output = authority.call(arm).detach()
    except BaseException:
        _abort_rank_bank_invocation(rank_bank, rank_invocation)
        raise
    local_shards: tuple[Any, ...] = ()
    gathered: list[Any] = []
    native_pending: tuple[Any, ...] = ()
    committed_not_consumed = False
    try:
        local_shards = rank_bank.take_rank(rank_invocation)
        for shard in local_shards:
            gathered.extend(
                parity._gather_one_block_after_forward(
                    shard,
                    invocation=invocation,
                    role_partition=partition,
                )
            )
        # Ownership may transfer into the native bank before the helper
        # returns.  Arm the failure scrub before entering that boundary.
        committed_not_consumed = True
        commit = attention_hook.commit_world4_shards_to_native_bank(
            native_bank=native_bank,
            invocation=invocation,
            rank_shards=gathered,
        )
        native_pending = native_bank.consume(invocation)
        committed_not_consumed = False
        transferred = native_pending
        native_pending = ()
        summary, role_token_counts = (
            _reduce_native_capture_group_finally_scrubbed(
                captures=transferred,
                native_bank=native_bank,
                invocation=invocation,
                authority=authority,
                arm=arm,
                partition=partition,
                assemblers=assemblers,
                graph_registry=graph_registry,
            )
        )
        captures = transferred
    except BaseException:
        for assembler in assemblers.values():
            assembler.abort()
        raise
    finally:
        for shard in (*local_shards, *gathered):
            if callable(getattr(shard, "zeroize", None)):
                shard.zeroize()
        if native_pending:
            native_bank.zeroize(native_pending)
        if committed_not_consumed:
            _abort_native_bank_invocation(native_bank, invocation)
    if any(
        int(torch.count_nonzero(value).item()) != 0
        for capture in captures
        for value in (
            capture.query,
            capture.key,
            capture.derived_qk_role_responsibility_proxy,
        )
    ):
        raise AUHPartialObjectGraphProbeV4Error(
            "native capture bank did not remain zeroized"
        )
    receipt = {
        "appearance_id": authority.appearance_id,
        "arm": arm,
        "sigma_band": authority.sigma_cell.band,
        "step_index": authority.sigma_cell.step_index,
        "state_sha256": invocation.state_sha256,
        "prompt_embedding_sha256": authority.prompt_embedding_sha256[arm],
        "role_partition_sha256": partition.digest,
        "role_order": list(partition.role_names),
        "role_token_counts": list(role_token_counts),
        "role_prior": [
            count / float(sum(role_token_counts))
            for count in role_token_counts
        ],
        "output_sha256": parity._tensor_digest(
            output, label=f"{arm} observer output"
        ),
        "commit": dict(commit),
        "capture_summary": summary,
        "block_count_reduced": len(summary),
        "raw_prompt_arm_zeroized_before_return": True,
        "resident_raw_prompt_arm_count_after_return": 0,
        "compact_evidence_only_retained": True,
        "failure_path_native_capture_group_finally_scrubbed": True,
    }
    return output, {
        **receipt,
        "digest": graph_registry_v4.object_sha256(receipt),
    }


def run_real_world4_probe(output: Path) -> Mapping[str, Any]:
    rank, _local_rank = _check_world4()
    if not output.is_absolute() or output.is_symlink() or output.exists():
        raise AUHPartialObjectGraphProbeV4Error(
            "output must be an absolute absent plain path"
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise AUHPartialObjectGraphProbeV4Error(
            "output parent must be a plain directory"
        )

    runtime = v2.site.create_auh_bernini_source_role_adapter({})
    binding_receipt = runtime.binding_receipt()
    prompt_bank, negative, partitions, prompt_receipt = v2._encode_prompt_bank(
        runtime,
        rank=rank,
        registry_module=prompt_registry,
    )
    partition_receipt = _validate_canonical_partitions(prompt_receipt)
    bootstrap_scrub = v2._scrub_site_bootstrap_tensors(runtime)
    initial_gaussian, gaussian_receipt = v2._load_initial_gaussian(runtime)
    plan, timesteps, sigmas = v2._real_capture_plan(
        runtime.model.diff_dec, runtime.device
    )
    if {
        cell.band: cell.step_index for cell in plan.sigma_cells
    } != dict(prompt_registry.SIGMA_CELL_INDICES):
        raise AUHPartialObjectGraphProbeV4Error("capture schedule differs")
    graph_registry = make_graph_registry()
    graph_stream = graph.PartialObjectGraphObserverV4(graph_registry)
    state_before = parity._module_state_version_receipt(runtime._transformer)
    rank_bank = attention_hook.InMemoryWorld4RankShardBank()
    native_bank = native.InMemoryNativeCaptureBank()
    base_cells = []
    authority_receipts = []
    capture_receipts = []
    trajectory_steps = []
    cell_residency_receipts = []

    layout = cdf.validate_latent_shape(tuple(initial_gaussian.shape))
    initial_packed = cdf._pack_spatial_latent(initial_gaussian, layout)
    selected_by_index = {cell.step_index: cell for cell in plan.sigma_cells}
    compute_dtype = runtime._transformer.patch_embedding.weight.dtype
    if compute_dtype != torch.bfloat16:
        raise AUHPartialObjectGraphProbeV4Error(
            "transformer compute dtype differs"
        )
    model_prompts = _model_prompts()

    for appearance in prompt_registry.APPEARANCES:
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
                condition_prompt_embeds=prompt_bank[
                    appearance.appearance_id
                ]["action"],
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
                        spatial_state.to(dtype=compute_dtype),
                        source_id=cdf.QUERY_ID,
                    )
                noisy_tokens = noisy_tokens.detach().contiguous()
                rotary = rotary.detach()
                timestep_object = timestep.expand(1)
                authority = native.seal_role_matched_t2v_four_arm_forward(
                    appearance_id=appearance.appearance_id,
                    sigma_cell=cell,
                    shared_step=runtime.model.diff_dec.shared_step,
                    action_kwargs={
                        "model_id": "transformer_1",
                        "noisy_latents": noisy_tokens,
                        "timesteps": timestep_object,
                        "cond_embeds": prompt_bank[
                            appearance.appearance_id
                        ]["action"],
                        "rotary_embs": rotary,
                        "batch_vae_seqlen": [int(noisy_tokens.shape[1])],
                        "batch_text_seqlen": [TEXT_LENGTH],
                    },
                    prompt_embeds=prompt_bank[appearance.appearance_id],
                    instructions=model_prompts[appearance.appearance_id],
                    role_phrases=appearance.role_phrases,
                )
                authority_receipts.append(dict(authority.receipt()))

                # Explicit B0: no hook, no graph observer, exactly once per
                # appearance/sigma cell.
                with torch.inference_mode():
                    base_output = authority.call("action").detach()
                base_sha = parity._tensor_digest(
                    base_output, label="B0 frozen base output"
                )
                assemblers = {
                    block: graph.SameStateCellAssemblerV4(graph_registry)
                    for block in native.BLOCKS
                }
                handle = attention_hook.install_native_relational_attention_hook(
                    runtime._transformer, rank_bank=rank_bank
                )
                arm_rows = []
                action_observed = None
                try:
                    for arm in prompt_registry.ARMS:
                        arm_output, arm_receipt = _capture_one_arm(
                            runtime=runtime,
                            authority=authority,
                            arm=arm,
                            partition=partitions[
                                appearance.appearance_id
                            ][arm],
                            rank_bank=rank_bank,
                            native_bank=native_bank,
                            assemblers=assemblers,
                            graph_registry=graph_registry,
                            rank=rank,
                        )
                        arm_rows.append(dict(arm_receipt))
                        if arm == "action":
                            action_observed = arm_output
                        else:
                            del arm_output
                except BaseException:
                    _abort_compact_ownership(assemblers, graph_stream)
                    raise
                finally:
                    try:
                        handle.restore()
                    except BaseException:
                        _abort_compact_ownership(assemblers, graph_stream)
                        raise
                if action_observed is None:
                    _abort_compact_ownership(assemblers, graph_stream)
                    raise AUHPartialObjectGraphProbeV4Error(
                        "action observer output is absent"
                    )
                action_sha = parity._tensor_digest(
                    action_observed, label="observer action output"
                )
                if not torch.equal(base_output, action_observed) or (
                    base_sha != action_sha
                ):
                    _abort_compact_ownership(assemblers, graph_stream)
                    raise AUHPartialObjectGraphProbeV4Error(
                        "observer changed the frozen-base action output"
                    )
                if any(
                    wrapper.base_calls != len(prompt_registry.ARMS)
                    or wrapper.observer_calls != len(prompt_registry.ARMS)
                    for wrapper in (
                        *handle.attn1_wrappers,
                        *handle.attn2_wrappers,
                    )
                ):
                    _abort_compact_ownership(assemblers, graph_stream)
                    raise AUHPartialObjectGraphProbeV4Error(
                        "hook call count differs"
                    )
                try:
                    for block in native.BLOCKS:
                        graph_stream.add(assemblers[block].finalize())
                except BaseException:
                    _abort_compact_ownership(assemblers, graph_stream)
                    raise
                residency_row = {
                    "appearance_id": appearance.appearance_id,
                    "sigma_band": cell.band,
                    "arms_executed_in_order": list(prompt_registry.ARMS),
                    "each_arm_zeroized_before_next_arm_returned": True,
                    "maximum_simultaneously_resident_raw_prompt_arms": 1,
                    "four_arm_raw_qk_bundle_created": False,
                    "compact_block_cells_added": len(native.BLOCKS),
                }
                cell_residency_receipts.append(
                    {
                        **residency_row,
                        "digest": graph_registry_v4.object_sha256(
                            residency_row
                        ),
                    }
                )
                base_row = {
                    "arm": "B0_FROZEN_BASE_OBSERVER_ABSENT",
                    "appearance_id": appearance.appearance_id,
                    "sigma_band": cell.band,
                    "step_index": cell.step_index,
                    "sigma": cell.sigma,
                    "output_sha256": base_sha,
                    "observer_action_output_sha256": action_sha,
                    "observer_action_bit_exact": True,
                    "graph_observation_supplied": False,
                    "component_success": None,
                    "used_as_graph_positive": False,
                }
                base_cells.append(
                    {
                        **base_row,
                        "digest": graph_registry_v4.object_sha256(base_row),
                    }
                )
                capture_receipts.extend(arm_rows)
                selected_seen.append(step_index)
                del (
                    base_output,
                    action_observed,
                    noisy_tokens,
                    rotary,
                    timestep_object,
                    authority,
                    assemblers,
                )

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
            int(prompt_registry.SIGMA_CELL_INDICES[band])
            for band in native.SIGMA_BAND_ORDER
        ]:
            raise AUHPartialObjectGraphProbeV4Error(
                "appearance capture cells differ"
            )
        del state_packed, solver

    graph_result = graph_stream.finalize()
    graph_public = graph_result.public_payload()
    graph_receipt = graph_result.receipt()
    rank_bank_receipt = rank_bank.receipt()
    native_bank_receipt = native_bank.receipt()
    state_after = parity._module_state_version_receipt(runtime._transformer)
    if state_after != state_before:
        raise AUHPartialObjectGraphProbeV4Error(
            "frozen transformer state changed"
        )
    if (
        native_bank_receipt["capture_count"] != EXPECTED_CAPTURE_COUNT
        or native_bank_receipt["consumed_count"] != EXPECTED_CAPTURE_COUNT
        or native_bank_receipt["zeroized_count"] != EXPECTED_CAPTURE_COUNT
        or native_bank_receipt["resident_invocation_count"] != 0
        or rank_bank_receipt["resident_rank_invocations"] != 0
    ):
        raise AUHPartialObjectGraphProbeV4Error(
            "capture banks did not close"
        )
    if (
        len(base_cells) != EXPECTED_FROZEN_BASE_CELLS
        or len(authority_receipts) != EXPECTED_FROZEN_BASE_CELLS
        or len(capture_receipts) != EXPECTED_OBSERVER_FORWARDS
        or len(cell_residency_receipts) != EXPECTED_FROZEN_BASE_CELLS
    ):
        raise AUHPartialObjectGraphProbeV4Error(
            "probe receipt matrix differs"
        )
    if not all(
        row["observer_action_bit_exact"] for row in base_cells
    ):
        raise AUHPartialObjectGraphProbeV4Error(
            "Frozen Base parity matrix differs"
        )
    if graph_receipt["representation_admitted"] is not False or (
        graph_receipt["shuffled_prompt_control_observed"] is not False
    ):
        raise AUHPartialObjectGraphProbeV4Error(
            "component crossed the representation claim boundary"
        )

    _require_all_rank_equal(
        graph_result.receipt_digest, label="partial object graph component"
    )
    local_summary = {
        "rank": rank,
        "frozen_state_digest": state_before["digest"],
        "graph_component_receipt_digest": graph_result.receipt_digest,
        "base_cell_digest": graph_registry_v4.object_sha256(base_cells),
        "capture_bank_digest": native_bank_receipt["digest"],
    }
    local_summary = {
        **local_summary,
        "digest": graph_registry_v4.object_sha256(local_summary),
    }
    rank_summaries = v2._all_rank_rows(local_summary)

    component_admitted = bool(graph_result.admitted)
    status = (
        "REAL_SAME_STATE_PARTIAL_OBJECT_COMPONENT_ADMITTED_"
        "REPRESENTATION_NOT_ADMITTED"
        if component_admitted
        else "REAL_SAME_STATE_PARTIAL_OBJECT_COMPONENT_REJECTED"
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
        "prompt_bank": dict(prompt_receipt),
        "canonical_prompt_partition": dict(partition_receipt),
        "capture_plan": dict(plan.receipt()),
        "trajectory_step_registry": trajectory_steps,
        "trajectory_model_forward_count": EXPECTED_TRAJECTORY_FORWARDS,
        "trajectory_unipc_step_count": EXPECTED_TRAJECTORY_STEPS,
        "frozen_base_probe_forward_count": EXPECTED_FROZEN_BASE_CELLS,
        "observer_probe_forward_count": EXPECTED_OBSERVER_FORWARDS,
        "total_frozen_transformer_forward_count": EXPECTED_TOTAL_FORWARDS,
        "frozen_base_cells": base_cells,
        "role_matched_four_arm_authorities": authority_receipts,
        "capture_receipts": capture_receipts,
        "raw_residency_receipts": cell_residency_receipts,
        "rank_summaries": rank_summaries,
        "capture_bank": dict(native_bank_receipt),
        "rank_capture_bank": dict(rank_bank_receipt),
        "partial_object_graph_public": graph_public,
        "partial_object_graph_receipt": graph_receipt,
        "frozen_transformer_state_unchanged": True,
        "frozen_transformer_state_digest": state_before["digest"],
        "all_nine_observer_action_outputs_equal_frozen_base_bit_exact": True,
        "frozen_base_graph_observation_supplied": False,
        "per_arm_immediate_reduce_and_zeroize_executed": True,
        "native_proxy_simplex_validated": True,
        "role_order_partition_and_token_priors_bound": True,
        "absolute_role_vs_dustbin_abstention_executed": True,
        "failed_absolute_evidence_kernel_exact_zero": True,
        "shared_frame_sources": ["noop", "static"],
        "action_or_reverse_defined_shared_frame": False,
        "shared_frame_endpoint_direction_scale_closure_executed": True,
        "failed_shared_frame_phase_abstained_all_four_arms": True,
        "four_arm_common_edge_domain_executed": True,
        "reverse_endpoint_topology_gate_executed": True,
        "maximum_simultaneously_resident_raw_prompt_arms": 1,
        "four_arm_raw_qk_bundle_created": False,
        "raw_qk_and_role_proxies_zeroized": True,
        "failure_path_native_capture_group_finally_scrubbed": True,
        "successful_exit_raw_capture_zeroization_observed": True,
        "explicit_capture_ownership_boundary_exception_scrub_covered": True,
        "uncovered_exception_requires_nonzero_exit_without_receipt": True,
        "all_allocation_failure_zeroization_claimed": False,
        "persistent_raw_tensor_artifact_created": False,
        "component_four_arm_mechanical_admitted": component_admitted,
        "representation_admitted": False,
        "full_oceg_representation_admitted": False,
        "shuffled_prompt_control_executed": False,
        "shuffled_prompt_control_observed": False,
        "shuffled_prompt_robustness_claimed": False,
        "missing_representation_gates_count_as_failure": True,
        "observer_only_diagnostic_launch_authorized": True,
        "launch_blocked_pending_failure_path_audit": False,
        "representation_or_renderer_launch_authorized": False,
        "source_bootstrap_tensor_consumed_by_probe_forward": False,
        "target_inputs_consumed": False,
        "final_anchor_video_decoded": False,
        "output_video_created": False,
        "decoder_called": False,
        "claims_apply_to_exercised_entrypoint_call_path_only": True,
        "renderer_called": False,
        "adapter_or_lora_loaded": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "route_or_injection_called": False,
        "candidate_output_modified": False,
        "backend_attention_weights_observed": False,
        "causal_generation_executed": False,
        "scientific_claim_authorized": False,
        "stable_transferable_action_representation_claimed": False,
    }
    value = {
        **value,
        "digest": graph_registry_v4.object_sha256(value),
    }

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
            write_status[0] = {
                "ok": True,
                "receipt_digest": value["digest"],
            }
        except Exception as error:
            write_status[0] = {
                "ok": False,
                "type": type(error).__name__,
                "message": str(error),
            }
    dist.broadcast_object_list(write_status, src=0)
    if not isinstance(write_status[0], Mapping) or (
        write_status[0].get("ok") is not True
    ):
        raise AUHPartialObjectGraphProbeV4Error(
            f"create-only receipt write failed: {write_status[0]}"
        )
    dist.barrier()
    return value


def _initialize_world4() -> None:
    if dist.is_initialized():
        raise AUHPartialObjectGraphProbeV4Error(
            "process group initialized too early"
        )
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise AUHPartialObjectGraphProbeV4Error(
            "live probe requires torchrun WORLD4"
        )
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
            raise AUHPartialObjectGraphProbeV4Error(
                "contract print accepts no output"
            )
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
        raise AUHPartialObjectGraphProbeV4Error(
            "live probe requires --output"
        )
    _initialize_world4()
    try:
        run_real_world4_probe(args.output)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
