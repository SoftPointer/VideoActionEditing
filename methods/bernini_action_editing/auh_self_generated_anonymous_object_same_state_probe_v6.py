#!/usr/bin/env python3
"""Create-only WORLD4 runner for the V6 anonymous-object diagnostic.

The live call path is frozen and read-only.  It follows one native action T2V
trajectory per appearance and, at high/mid/low cells, evaluates eight text
arms on the exact same non-text state.  B0 runs once per cell without a hook.
The observer-on action output must be bit-exact.  Only projected visual
intermediates leave the hook; all discovery is anonymous and cross-fitted.

GPU launch remains fail-closed until an independent audit changes the single
launch authority and the frozen preregistration together.  Contract printing
does not initialize distributed state or CUDA.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
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
import anchor_cross_attention_transport as anchor_cross_transport  # noqa: E402
import anchor_qk_transport as anchor_qk_transport  # noqa: E402
import anonymous_visual_projection_hook_v6 as projection_hook  # noqa: E402
import auh_native_relational_attention_parity_smoke_v1 as parity  # noqa: E402
import auh_self_generated_relational_t2v_trajectory_probe_v2 as v2  # noqa: E402
import differential_sampler as cdf  # noqa: E402
import guided_source_aligned_controller as guided_source_controller  # noqa: E402
import infer_native_self_generated_intermediate_anchor_canary_v1 as intermediate_canary  # noqa: E402
import infer_native_self_generated_relational_graph_observer_v1 as native_graph_runtime  # noqa: E402
import native_relational_attention_hook_v1 as native_attention_hook  # noqa: E402
import self_generated_anonymous_object_observer_v6 as observer  # noqa: E402
import self_generated_anonymous_object_registry_v6 as registry  # noqa: E402
import self_generated_intermediate_action_anchor_v1 as intermediate_anchor  # noqa: E402
import self_generated_relational_action_graph_observer_v1 as relational_graph  # noqa: E402
import self_generated_relational_t2v_probe_registry_v2 as trajectory_registry  # noqa: E402
import source_aligned_controller as source_aligned_controller  # noqa: E402
import source_kv_replay as source_kv_replay  # noqa: E402
import source_owned_role_locator_v15 as locator  # noqa: E402
import source_owned_role_locator_v15b_e00_asset as site_role_asset  # noqa: E402
import tri_branch_unipc as tri_branch_unipc  # noqa: E402


METHOD = registry.METHOD
SCHEMA_VERSION = "bernini-auh-self-generated-anonymous-object-probe-v6"
WORLD_SIZE = 4
TEXT_LENGTH = 512
TEXT_WIDTH = 4096
EXPECTED_B0_CELLS = len(registry.APPEARANCE_IDS) * len(registry.SIGMA_CELL_INDICES)
EXPECTED_OBSERVER_FORWARDS = EXPECTED_B0_CELLS * len(registry.ARMS)
EXPECTED_PROJECTED_BLOCK_CAPTURES = EXPECTED_OBSERVER_FORWARDS * len(registry.BLOCKS)
EXPECTED_TRAJECTORY_STEPS = len(registry.APPEARANCE_IDS) * 40
EXPECTED_TRAJECTORY_FORWARDS = EXPECTED_TRAJECTORY_STEPS * 2
EXPECTED_TOTAL_FROZEN_FORWARDS = (
    EXPECTED_TRAJECTORY_FORWARDS + EXPECTED_B0_CELLS + EXPECTED_OBSERVER_FORWARDS
)
EXPECTED_PRELAUNCH_CPU_TESTS = 48
GPU_LAUNCH_AUTHORIZED = True
LAUNCH_BLOCKED_PENDING_INDEPENDENT_AUDIT = False


class AUHAnonymousObjectProbeV6Error(RuntimeError):
    """A WORLD4, same-state, ownership, B0, or claim boundary failed."""


def _rank_wrapper_path() -> Path:
    return (
        METHOD_ROOT
        / "scripts"
        / "auh_self_generated_anonymous_object_same_state_probe_rank_wrapper_v6.sh"
    )


def _source_row(role: str, path: Path) -> Mapping[str, Any]:
    original = Path(path).absolute()
    if not original.is_file() or original.is_symlink():
        raise AUHAnonymousObjectProbeV6Error(f"source {role} must be a plain file")
    try:
        canonical = original.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise AUHAnonymousObjectProbeV6Error(
            f"source {role} cannot be resolved"
        ) from error
    if original != canonical:
        raise AUHAnonymousObjectProbeV6Error(
            f"source {role} path is not canonical"
        )
    payload = original.read_bytes()
    return {
        "role": role,
        "file": original.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _official_transformer_source_row() -> Mapping[str, Any]:
    expected = (
        Path(v2.site.BERNINI_ROOT) / "bernini" / "models" / "transformer_wan.py"
    )
    canonical = projection_hook.validate_official_transformer_source_file_v6(
        expected
    )
    return {
        **_source_row("actual_official_transformer_source", canonical),
        "canonical_path": str(canonical),
    }


def source_manifest() -> Mapping[str, Any]:
    modules = (
        ("runner", sys.modules[__name__]),
        ("anonymous_observer", observer),
        ("anonymous_registry", registry),
        ("visual_projection_hook", projection_hook),
        ("v2_schedule_and_runtime_helpers", v2),
        ("trajectory_helper", trajectory),
        ("parity_helper", parity),
        ("differential_sampler", cdf),
        ("site_adapter", v2.site),
        ("tensor_digest_authority", locator),
        ("anchor_cross_attention_transport_import_closure", anchor_cross_transport),
        ("anchor_qk_transport_import_closure", anchor_qk_transport),
        ("guided_source_aligned_controller_active_APG", guided_source_controller),
        ("intermediate_anchor_canary_import_closure", intermediate_canary),
        ("native_relational_graph_capture_plan", native_graph_runtime),
        ("native_relational_attention_hook_import_closure", native_attention_hook),
        ("intermediate_action_anchor_import_closure", intermediate_anchor),
        ("relational_action_graph_import_closure", relational_graph),
        ("trajectory_registry_capture_indices", trajectory_registry),
        ("source_aligned_controller_import_closure", source_aligned_controller),
        ("source_kv_replay_import_closure", source_kv_replay),
        ("site_role_asset_import_closure", site_role_asset),
        ("tri_branch_unipc_active_APG", tri_branch_unipc),
    )
    rows = [
        _source_row(role, Path(module.__file__)) for role, module in modules
    ]
    rows.extend(
        (
            _official_transformer_source_row(),
            _source_row("frozen_preregistration", registry.PREREG_PATH),
            _source_row("rank_wrapper", _rank_wrapper_path()),
        )
    )
    names = [row["file"] for row in rows]
    if len(names) != len(set(names)):
        raise AUHAnonymousObjectProbeV6Error("source manifest names collide")
    value = {
        "files": rows,
        "file_count": len(rows),
        "all_plain_nonsymlink_files": True,
    }
    return {**value, "digest": registry.object_sha256(value)}


def test_source_manifest() -> Mapping[str, Any]:
    """Bind the exact prelaunch CPU suite without mixing tests into runtime code."""

    test_root = METHOD_ROOT / "tests"
    rows = [
        _source_row(role, test_root / filename)
        for role, filename in (
            (
                "registry_and_preregistration_tests",
                "test_self_generated_anonymous_object_registry_v6.py",
            ),
            (
                "official_hook_and_projection_tests",
                "test_anonymous_visual_projection_hook_v6.py",
            ),
            (
                "anonymous_observer_and_reducer_tests",
                "test_self_generated_anonymous_object_observer_v6.py",
            ),
            (
                "real_runner_and_contract_tests",
                "test_auh_self_generated_anonymous_object_same_state_probe_v6.py",
            ),
        )
    ]
    value = {
        "files": rows,
        "file_count": len(rows),
        "expected_unittest_case_count": EXPECTED_PRELAUNCH_CPU_TESTS,
        "all_plain_nonsymlink_files": True,
        "execution_claimed_by_gpu_receipt": False,
    }
    return {**value, "digest": registry.object_sha256(value)}


def probe_contract() -> Mapping[str, Any]:
    prereg = dict(registry.load_preregistration())
    completion_authority = dict(prereg["receipt_completion_authority"])
    site_geometry = tuple(int(item) for item in v2.site.SOURCE_GEOMETRY)
    if site_geometry != (registry.PHASES, registry.PATCH_HEIGHT, registry.PATCH_WIDTH):
        raise AUHAnonymousObjectProbeV6Error(
            "registry patch geometry differs from pinned site geometry"
        )
    projection = projection_hook.ProjectionAuthorityV6.create()
    projection.validate()
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "source_manifest": dict(source_manifest()),
        "test_source_manifest": dict(test_source_manifest()),
        "preregistration": prereg,
        "registry": dict(registry.registry_receipt()),
        "projection": dict(projection.receipt()),
        "checkpoint_tree_sha256": v2.site.CHECKPOINT_TREE_SHA256,
        "official_transformer_source_sha256": (
            projection_hook.OFFICIAL_TRANSFORMER_SOURCE_SHA256
        ),
        "world_size": WORLD_SIZE,
        "site_source_geometry": list(site_geometry),
        "registry_patch_geometry": [
            registry.PHASES,
            registry.PATCH_HEIGHT,
            registry.PATCH_WIDTH,
        ],
        "num_inference_steps": 40,
        "trajectory": "three_action_prompt_bernini_field_unipc_histories",
        "trajectory_guidance_transformer_forwards_per_step": 2,
        "trajectory_transformer_forward_count": EXPECTED_TRAJECTORY_FORWARDS,
        "trajectory_unipc_step_count": EXPECTED_TRAJECTORY_STEPS,
        "same_state_nontext_overlay": True,
        "capture_blocks": list(registry.BLOCKS),
        "capture_steps": dict(registry.SIGMA_CELL_INDICES),
        "prompt_arms": list(registry.ARMS),
        "observer_forward_count": EXPECTED_OBSERVER_FORWARDS,
        "projected_block_capture_count": EXPECTED_PROJECTED_BLOCK_CAPTURES,
        "frozen_base_cell_count": EXPECTED_B0_CELLS,
        "total_frozen_transformer_forward_count": EXPECTED_TOTAL_FROZEN_FORWARDS,
        "B0_observer_absent_once_per_cell": True,
        "B0_action_observer_bit_exact_required": True,
        "B0_may_supply_graph_evidence": False,
        "post_rope_visual_query_observed": True,
        "attn1_hidden_intermediate_observed": True,
        "caption_token_offsets_computed": False,
        "caption_role_partition_computed": False,
        "text_key_or_value_observed": False,
        "fixed_semantic_role_inventory_used": False,
        "action_noop_residual_usage": "detached_proposal_support_only",
        "action_noop_residual_as_descriptor": False,
        "action_noop_residual_as_reward": False,
        "prompt_neutral_visual_correspondence": True,
        "crossfit_layer_folds": {
            name: list(rows) for name, rows in observer.LAYER_FOLDS.items()
        },
        "crossfit_phase_pairs": {
            name: [list(pair) for pair in rows]
            for name, rows in observer.CROSS_FIT_PHASE_PAIRS.items()
        },
        "proposal_evaluation_layer_and_time_disjoint": True,
        "controls_executed_in_live_call_path": list(observer.CONTROL_ARMS),
        "source_swap_exact_next_appearance_action": True,
        "branchwise_diagnostic_formula": "A_to_B AND B_to_A",
        "overall_diagnostic_formula": "all 9/9 cells pass both branches",
        "graph_abstention_R0_compensation_permitted": False,
        "raw_query_and_hidden_observer_clones_zeroized_after_projection": True,
        "only_projected_sketches_retained_until_cell_reduction": True,
        "projected_sketches_zeroized_at_cell_reduction": True,
        "success_path_capture_ownership_zeroization_required": True,
        "explicit_capture_ownership_boundary_exception_scrub_required": True,
        "uncovered_exception_policy": (
            "nonzero_exit; any candidate file remains non_authoritative "
            "without external completion seal"
        ),
        "all_allocation_failure_zeroization_claimed": False,
        "receipt_completion_authority": completion_authority,
        "manifest_claim_scope": (
            "entrypoint_exercised_call_path_only; broad helper modules may expose "
            "decoder, route, optimizer, or role facilities that this call path does not call"
        ),
        "persistent_tensor_artifact_authorized": False,
        "target_inputs_consumed": False,
        "site_source_bootstrap_tensors_created": True,
        "site_source_bootstrap_tensors_scrubbed_before_trajectory": True,
        "source_bootstrap_tensor_consumed_by_probe_forward": False,
        "decoder_called": False,
        "renderer_called": False,
        "adapter_or_lora_loaded": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "route_or_injection_called": False,
        "candidate_output_modified": False,
        "prompt_shuffle_control_executed": False,
        "heldout_transfer_control_executed": False,
        "representation_admission_hard_false": True,
        "stable_transferable_action_representation_claimed": False,
        "scientific_claim_authorized": False,
        "gpu_launch_authorized": GPU_LAUNCH_AUTHORIZED,
        "launch_blocked_pending_independent_audit": (
            LAUNCH_BLOCKED_PENDING_INDEPENDENT_AUDIT
        ),
    }
    if (
        value["gpu_launch_authorized"]
        != prereg["claims"]["gpu_launch_authorized"]
        or value["launch_blocked_pending_independent_audit"]
        != prereg["claims"]["launch_blocked_pending_independent_audit"]
    ):
        raise AUHAnonymousObjectProbeV6Error(
            "runner and preregistration launch authority differ"
        )
    ownership = prereg.get("ownership_and_failure_claims")
    for name in (
        "success_path_capture_ownership_zeroization_required",
        "explicit_capture_ownership_boundary_exception_scrub_required",
        "uncovered_exception_policy",
        "all_allocation_failure_zeroization_claimed",
    ):
        if not isinstance(ownership, Mapping) or value[name] != ownership.get(name):
            raise AUHAnonymousObjectProbeV6Error(
                "runner and preregistration ownership claims differ"
            )
    return {**value, "digest": registry.object_sha256(value)}


def remote_launch_template() -> Mapping[str, Any]:
    completion_authority = dict(
        registry.load_preregistration()["receipt_completion_authority"]
    )
    value = {
        "python": "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12",
        "launcher": "python -m torch.distributed.run",
        "torchrun_arguments": ["--standalone", "--nproc_per_node=4", "--no-python"],
        "rank_wrapper": _rank_wrapper_path().name,
        "rank_wrapper_arguments": [
            "ABSENT_CREATE_ONLY_JSON_PATH",
            Path(__file__).name,
        ],
        "required_environment": {
            "MODELING_BACKEND": "hf",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "MIOPEN_CACHE_ROOT": "ABSOLUTE_FRESH_DIRECTORY",
        },
        "required_slurm_isolation": {
            "nonoverlapping_ROCR_VISIBLE_DEVICES": True,
            "no_concurrent_step_on_any_selected_gpu": True,
            "srun_flags": ["--exclusive", "--exact", "--kill-on-bad-exit=1"],
        },
        "probe_output_kind": "candidate_only",
        "receipt_completion_authority": completion_authority,
        "external_controller_postflight_required_after_srun_exit_zero": True,
        "launch_executed": False,
        "gpu_launch_authorized": GPU_LAUNCH_AUTHORIZED,
        "launch_blocked_pending_independent_audit": (
            LAUNCH_BLOCKED_PENDING_INDEPENDENT_AUDIT
        ),
    }
    return {**value, "digest": registry.object_sha256(value)}


@dataclass(frozen=True)
class SigmaCellV6:
    band: str
    step_index: int
    sigma: float

    def __post_init__(self) -> None:
        if (
            self.band not in registry.SIGMA_CELL_INDICES
            or self.step_index != registry.SIGMA_CELL_INDICES[self.band]
            or not math.isfinite(float(self.sigma))
            or not 0.0 < float(self.sigma) <= 1.0
        ):
            raise AUHAnonymousObjectProbeV6Error("V6 sigma cell differs")

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "band": self.band,
            "step_index": self.step_index,
            "sigma": float(self.sigma),
        }
        return {**value, "digest": registry.object_sha256(value)}


def _check_world4() -> tuple[int, int]:
    try:
        return v2._check_world4()
    except Exception as error:
        raise AUHAnonymousObjectProbeV6Error(str(error)) from error


def _all_rank_rows(value: Any) -> list[Any]:
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, value)
    return rows


def _require_all_rank_equal(value: Any, *, label: str) -> None:
    if _all_rank_rows(value) != [value] * WORLD_SIZE:
        raise AUHAnonymousObjectProbeV6Error(f"{label} differs across WORLD4")


def _assert_no_target_payload(value: Any, *, path: str = "payload") -> None:
    forbidden = ("target_video", "target_frames", "target_latent", "teacher_target")
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).casefold()
            if any(token in name for token in forbidden):
                raise AUHAnonymousObjectProbeV6Error(
                    f"target payload is forbidden at {path}.{key}"
                )
            _assert_no_target_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_no_target_payload(item, path=f"{path}[{index}]")


def _model_prompt(caption: str) -> str:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    return (
        "You are a helpful assistant specialized in text-to-video generation."
        + prompt_clean(caption)
    )


def _encode_anonymous_prompt_bank(
    runtime: Any, *, rank: int
) -> tuple[Mapping[str, Mapping[str, torch.Tensor]], torch.Tensor, Mapping[str, Any]]:
    from bernini.cli import DEFAULT_NEG_PROMPT
    from transformers import UMT5EncoderModel

    names = [
        f"{appearance.appearance_id}:{arm}"
        for appearance in registry.APPEARANCES
        for arm in registry.ARMS
    ]
    token_rows: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    model_prompts: dict[str, str] = {}
    token_receipts: dict[str, Any] = {}
    status: list[Any] = [None]
    if rank == 0:
        try:
            for appearance in registry.APPEARANCES:
                for arm in registry.ARMS:
                    name = f"{appearance.appearance_id}:{arm}"
                    prompt = _model_prompt(appearance.captions[arm])
                    ids, mask = runtime._legacy._tokenize_training_prompt(
                        runtime._tokenizer, prompt
                    )
                    if tuple(ids.shape) != (1, TEXT_LENGTH) or tuple(mask.shape) != (
                        1,
                        TEXT_LENGTH,
                    ):
                        raise AUHAnonymousObjectProbeV6Error(
                            "anonymous prompt token geometry differs"
                        )
                    token_rows[name] = (ids, mask)
                    model_prompts[name] = prompt
                    token_receipts[name] = {
                        "prompt_sha256": registry.text_sha256(prompt),
                        "input_ids_sha256": locator.tensor_sha256(ids),
                        "attention_mask_sha256": locator.tensor_sha256(mask),
                        "caption_offsets_requested": False,
                        "role_partition_created": False,
                    }
            negative_ids, negative_mask = runtime._legacy._tokenize_renderer_negative(
                runtime._tokenizer, DEFAULT_NEG_PROMPT
            )
            token_rows["negative"] = (negative_ids, negative_mask)
            token_receipts["negative"] = {
                "prompt_sha256": registry.text_sha256(DEFAULT_NEG_PROMPT),
                "input_ids_sha256": locator.tensor_sha256(negative_ids),
                "attention_mask_sha256": locator.tensor_sha256(negative_mask),
            }
            status[0] = {
                "ok": True,
                "model_prompts": model_prompts,
                "token_receipts": token_receipts,
            }
        except Exception as error:
            status[0] = {
                "ok": False,
                "type": type(error).__name__,
                "message": str(error),
            }
    dist.broadcast_object_list(status, src=0)
    if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
        raise AUHAnonymousObjectProbeV6Error(
            f"rank-zero anonymous tokenization failed: {status[0]}"
        )
    if rank != 0:
        model_prompts = dict(status[0]["model_prompts"])
        token_receipts = dict(status[0]["token_receipts"])

    if getattr(runtime.model, "t5_text_encoder", None) is not None:
        raise AUHAnonymousObjectProbeV6Error("site T5 was not retired")
    embeddings_local: dict[str, torch.Tensor] = {}
    encode_status: list[Any] = [None]
    dist.barrier()
    if rank == 0:
        try:
            runtime.model.t5_text_encoder = UMT5EncoderModel.from_pretrained(
                str(v2.site.CHECKPOINT),
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            runtime.model.t5_text_encoder.to(runtime.device)
            with torch.inference_mode():
                for name in (*names, "negative"):
                    ids, mask = token_rows[name]
                    embedding = runtime.model.encode_prompt(
                        ids.to(runtime.device), mask.to(runtime.device)
                    ).detach().contiguous()
                    if tuple(embedding.shape) != (1, TEXT_LENGTH, TEXT_WIDTH) or (
                        embedding.dtype != torch.bfloat16
                    ):
                        raise AUHAnonymousObjectProbeV6Error(
                            "anonymous prompt embedding geometry differs"
                        )
                    embeddings_local[name] = embedding
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
        raise AUHAnonymousObjectProbeV6Error(
            f"rank-zero anonymous prompt encoding failed: {encode_status[0]}"
        )

    broadcast: dict[str, torch.Tensor] = {}
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
            raise AUHAnonymousObjectProbeV6Error("rank-zero embedding is absent")
        dist.broadcast(value, src=0)
        value = value.detach().contiguous()
        digest = locator.tensor_sha256(value)
        _require_all_rank_equal(digest, label=f"{name} prompt embedding")
        broadcast[name] = value
        embedding_sha[name] = digest
    prompt_bank = {
        appearance.appearance_id: {
            arm: broadcast[f"{appearance.appearance_id}:{arm}"]
            for arm in registry.ARMS
        }
        for appearance in registry.APPEARANCES
    }
    source_swap_embedding_checks, neutral_sha = _validate_anonymous_embedding_controls(
        embedding_sha
    )
    value = {
        "token_receipts": token_receipts,
        "model_prompt_sha256": {
            name: registry.text_sha256(text)
            for name, text in sorted(model_prompts.items())
        },
        "embedding_sha256": embedding_sha,
        "rank_zero_only_T5_load": True,
        "T5_retired_before_trajectory": True,
        "caption_offsets_requested": False,
        "role_partition_created": False,
        "source_swap_exact_cycle": True,
        "source_swap_embedding_equals_next_action_embedding": True,
        "source_swap_embedding_checks": source_swap_embedding_checks,
        "identical_neutral_caption_embedding_unique_count": 1,
        "neutral_embedding_sha256": neutral_sha,
    }
    return prompt_bank, broadcast["negative"], {
        **value,
        "digest": registry.object_sha256(value),
    }


def _validate_anonymous_embedding_controls(
    embedding_sha: Mapping[str, str],
) -> tuple[list[Mapping[str, Any]], str]:
    source_swap_embedding_checks = []
    for index, appearance in enumerate(registry.APPEARANCES):
        next_appearance = registry.APPEARANCES[(index + 1) % len(registry.APPEARANCES)]
        source_name = f"{appearance.appearance_id}:source_swap"
        next_name = f"{next_appearance.appearance_id}:action"
        equal = embedding_sha[source_name] == embedding_sha[next_name]
        if not equal:
            raise AUHAnonymousObjectProbeV6Error(
                "source-swap embedding is not the exact next-appearance action embedding"
            )
        source_swap_embedding_checks.append(
            {
                "source_swap_appearance": appearance.appearance_id,
                "next_action_appearance": next_appearance.appearance_id,
                "embedding_sha256": embedding_sha[source_name],
                "equal": True,
            }
        )
    neutral_embedding_sha = {
        embedding_sha[f"{appearance.appearance_id}:neutral"]
        for appearance in registry.APPEARANCES
    }
    if len(neutral_embedding_sha) != 1:
        raise AUHAnonymousObjectProbeV6Error(
            "identical neutral captions produced different embeddings"
        )
    return source_swap_embedding_checks, next(iter(neutral_embedding_sha))


@dataclass(frozen=True)
class AnonymousSameStateArmAuthorityV6:
    appearance_id: str
    sigma_cell: SigmaCellV6
    shared_step: Any
    arm_args: Mapping[str, tuple[Any, ...]]
    arm_kwargs: Mapping[str, Mapping[str, Any]]
    prompt_embedding_sha256: Mapping[str, str]
    instruction_sha256: Mapping[str, str]
    state_tensor_sha256: Mapping[str, str]

    def validate(self) -> None:
        if self.appearance_id not in registry.APPEARANCE_IDS or set(
            self.arm_args
        ) != set(registry.ARMS) or set(self.arm_kwargs) != set(registry.ARMS):
            raise AUHAnonymousObjectProbeV6Error("anonymous arm authority differs")
        bounds = {
            arm: inspect.signature(self.shared_step).bind(
                *self.arm_args[arm], **dict(self.arm_kwargs[arm])
            )
            for arm in registry.ARMS
        }
        for bound in bounds.values():
            bound.apply_defaults()
        required = {
            "model_id",
            "noisy_latents",
            "timesteps",
            "cond_embeds",
            "rotary_embs",
            "batch_vae_seqlen",
            "batch_text_seqlen",
        }
        action = bounds["action"]
        if not required.issubset(action.arguments):
            raise AUHAnonymousObjectProbeV6Error("shared step ABI differs")
        _assert_no_target_payload(action.arguments)
        text_fields = {"cond_embeds", "batch_text_seqlen"}
        for arm, bound in bounds.items():
            if not required.issubset(bound.arguments):
                raise AUHAnonymousObjectProbeV6Error("arm call ABI differs")
            for name, action_value in action.arguments.items():
                if name in text_fields:
                    continue
                value = bound.arguments.get(name)
                if isinstance(action_value, Mapping) and not action_value and isinstance(value, Mapping):
                    if value:
                        raise AUHAnonymousObjectProbeV6Error("arm kwargs differ")
                elif value is not action_value:
                    raise AUHAnonymousObjectProbeV6Error(
                        f"{arm} differs outside text fields: {name}"
                    )
        prompts = {arm: bounds[arm].arguments["cond_embeds"] for arm in registry.ARMS}
        if len({id(value) for value in prompts.values()}) != len(registry.ARMS):
            raise AUHAnonymousObjectProbeV6Error("prompt objects must be distinct")
        for arm, prompt in prompts.items():
            if (
                not isinstance(prompt, torch.Tensor)
                or tuple(prompt.shape) != (1, TEXT_LENGTH, TEXT_WIDTH)
                or prompt.requires_grad
                or prompt.grad_fn is not None
                or locator.tensor_sha256(prompt) != self.prompt_embedding_sha256[arm]
            ):
                raise AUHAnonymousObjectProbeV6Error(f"{arm} prompt differs")
        for name, digest in self.state_tensor_sha256.items():
            if locator.tensor_sha256(action.arguments[name]) != digest:
                raise AUHAnonymousObjectProbeV6Error(f"{name} state changed")

    def call(self, arm: str) -> Any:
        self.validate()
        if arm not in registry.ARMS:
            raise AUHAnonymousObjectProbeV6Error("anonymous arm is unknown")
        return self.shared_step(*self.arm_args[arm], **dict(self.arm_kwargs[arm]))

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        value = {
            "appearance_id": self.appearance_id,
            "sigma_cell": dict(self.sigma_cell.receipt()),
            "arms": list(registry.ARMS),
            "same_noisy_timestep_rotary_and_nontext_objects": True,
            "only_replaced_fields": ["cond_embeds", "batch_text_seqlen"],
            "semantic_role_authority_used": False,
            "prompt_embedding_sha256": dict(self.prompt_embedding_sha256),
            "instruction_sha256": dict(self.instruction_sha256),
            "state_tensor_sha256": dict(self.state_tensor_sha256),
        }
        return {**value, "digest": registry.object_sha256(value)}


def seal_anonymous_same_state_arms_v6(
    *,
    appearance_id: str,
    sigma_cell: SigmaCellV6,
    shared_step: Any,
    action_kwargs: Mapping[str, Any],
    prompt_embeds: Mapping[str, torch.Tensor],
    instructions: Mapping[str, str],
) -> AnonymousSameStateArmAuthorityV6:
    if set(prompt_embeds) != set(registry.ARMS) or set(instructions) != set(
        registry.ARMS
    ):
        raise AUHAnonymousObjectProbeV6Error("exact anonymous arm set is required")
    signature = inspect.signature(shared_step)
    base = signature.bind(**dict(action_kwargs))
    base.apply_defaults()
    if base.arguments.get("cond_embeds") is not prompt_embeds["action"]:
        raise AUHAnonymousObjectProbeV6Error("action prompt ownership differs")
    arm_args: dict[str, tuple[Any, ...]] = {}
    arm_kwargs: dict[str, Mapping[str, Any]] = {}
    for arm in registry.ARMS:
        bound = signature.bind(**dict(action_kwargs))
        bound.apply_defaults()
        bound.arguments["cond_embeds"] = prompt_embeds[arm]
        bound.arguments["batch_text_seqlen"] = [int(prompt_embeds[arm].shape[1])]
        arm_args[arm] = tuple(bound.args)
        arm_kwargs[arm] = dict(bound.kwargs)
    authority = AnonymousSameStateArmAuthorityV6(
        appearance_id,
        sigma_cell,
        shared_step,
        arm_args,
        arm_kwargs,
        {arm: locator.tensor_sha256(prompt_embeds[arm]) for arm in registry.ARMS},
        {arm: registry.text_sha256(instructions[arm]) for arm in registry.ARMS},
        {
            name: locator.tensor_sha256(base.arguments[name])
            for name in ("noisy_latents", "timesteps", "rotary_embs")
        },
    )
    authority.validate()
    return authority


def _gather_projected_block_after_forward(
    shard: projection_hook.ProjectedVisualRankShardV6,
) -> projection_hook.ProjectedVisualCaptureV6:
    query_local: Optional[torch.Tensor] = None
    hidden_local: Optional[torch.Tensor] = None
    query_flat: Optional[torch.Tensor] = None
    hidden_flat: Optional[torch.Tensor] = None
    query_rank_major: Optional[torch.Tensor] = None
    hidden_rank_major: Optional[torch.Tensor] = None
    result: Optional[projection_hook.ProjectedVisualCaptureV6] = None
    succeeded = False
    try:
        query_local, hidden_local, metadata = shard.collective_payload_and_zeroize()
        query_flat = torch.empty(
            (WORLD_SIZE * query_local.shape[0], *query_local.shape[1:]),
            dtype=query_local.dtype,
            device=query_local.device,
        )
        hidden_flat = torch.empty(
            (WORLD_SIZE * hidden_local.shape[0], *hidden_local.shape[1:]),
            dtype=hidden_local.dtype,
            device=hidden_local.device,
        )
        metadata_rows: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_into_tensor(query_flat, query_local)
        dist.all_gather_into_tensor(hidden_flat, hidden_local)
        dist.all_gather_object(metadata_rows, metadata)
        query_rank_major = query_flat.reshape(
            WORLD_SIZE, *query_local.shape
        ).contiguous()
        hidden_rank_major = hidden_flat.reshape(
            WORLD_SIZE, *hidden_local.shape
        ).contiguous()
        result = projection_hook.reconstruct_projected_world4_block_v6(
            identity=shard.invocation.identity,
            block_index=shard.block_index,
            projection_digest=shard.invocation.projection_digest,
            query_rank_major=query_rank_major,
            hidden_rank_major=hidden_rank_major,
            rank_metadata=metadata_rows,
        )
        succeeded = True
        return result
    finally:
        with torch.inference_mode():
            for value in (
                query_local,
                hidden_local,
                query_flat,
                hidden_flat,
                query_rank_major,
                hidden_rank_major,
            ):
                if isinstance(value, torch.Tensor):
                    value.zero_()
        if not succeeded and result is not None:
            result.zeroize()


def _capture_one_arm(
    *,
    runtime: Any,
    authority: AnonymousSameStateArmAuthorityV6,
    arm: str,
    rank: int,
    projection: projection_hook.ProjectionAuthorityV6,
    rank_bank: projection_hook.InMemoryProjectedRankBankV6,
) -> tuple[torch.Tensor, Mapping[int, observer.AnonymousProjectedArmV6], Mapping[str, Any]]:
    identity = projection_hook.AnonymousCaptureIdentityV6(
        authority.appearance_id,
        arm,
        authority.sigma_cell.band,
        authority.sigma_cell.step_index,
        authority.state_tensor_sha256["noisy_latents"],
        authority.state_tensor_sha256["timesteps"],
        authority.state_tensor_sha256["rotary_embs"],
        runtime.source_geometry.height,
        runtime.source_geometry.width,
    )
    invocation = projection_hook.AnonymousRankInvocationV6(
        identity,
        projection_hook.World4VisualLayoutV6(
            rank, runtime.source_geometry.height, runtime.source_geometry.width
        ),
        projection.digest,
    )
    try:
        with torch.inference_mode(), rank_bank.observe(invocation):
            output = authority.call(arm).detach()
    except BaseException:
        rank_bank.abort(invocation)
        raise
    local_shards: tuple[projection_hook.ProjectedVisualRankShardV6, ...] = ()
    global_rows: list[projection_hook.ProjectedVisualCaptureV6] = []
    transferred: dict[int, observer.AnonymousProjectedArmV6] = {}
    succeeded = False
    try:
        local_shards = rank_bank.take_rank(invocation)
        for shard in local_shards:
            capture = _gather_projected_block_after_forward(shard)
            global_rows.append(capture)
            transferred[capture.block_index] = observer.AnonymousProjectedArmV6.from_capture(
                capture
            )
        if set(transferred) != set(registry.BLOCKS):
            raise AUHAnonymousObjectProbeV6Error("projected arm blocks differ")
        succeeded = True
    finally:
        for shard in local_shards:
            if not shard.consumed:
                shard.zeroize()
        if not succeeded:
            for row in transferred.values():
                row.zeroize()
            for capture in global_rows:
                if not capture.consumed:
                    capture.zeroize()
    value = {
        "appearance_id": authority.appearance_id,
        "arm": arm,
        "sigma_band": authority.sigma_cell.band,
        "step_index": authority.sigma_cell.step_index,
        "state_sha256": identity.state_sha256,
        "prompt_embedding_sha256": authority.prompt_embedding_sha256[arm],
        "output_sha256": parity._tensor_digest(output, label=f"V6 {arm} output"),
        "block_count": len(transferred),
        "query_sketch_shape": [
            1,
            registry.PHASES,
            registry.PATCHES,
            projection_hook.QUERY_SKETCH_DIM,
        ],
        "hidden_sketch_shape": [
            1,
            registry.PHASES,
            registry.PATCHES,
            projection_hook.HIDDEN_SKETCH_DIM,
        ],
        "projection_digest": projection.digest,
        "raw_query_and_hidden_clones_zeroized_before_return": True,
        "caption_role_partition_used": False,
    }
    return output, transferred, {**value, "digest": registry.object_sha256(value)}


def _abort_projected_cell(
    rows: Mapping[str, Mapping[int, observer.AnonymousProjectedArmV6]]
) -> None:
    for arm_rows in rows.values():
        for row in arm_rows.values():
            if not row.consumed:
                row.zeroize()


def run_real_world4_probe(output: Path) -> Mapping[str, Any]:
    if not GPU_LAUNCH_AUTHORIZED or LAUNCH_BLOCKED_PENDING_INDEPENDENT_AUDIT:
        raise AUHAnonymousObjectProbeV6Error(
            "V6 GPU launch is blocked pending independent audit"
        )
    rank, _local_rank = _check_world4()
    if not output.is_absolute() or output.is_symlink() or output.exists():
        raise AUHAnonymousObjectProbeV6Error(
            "output must be an absolute absent plain path"
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise AUHAnonymousObjectProbeV6Error("output parent must be plain")

    runtime = v2.site.create_auh_bernini_source_role_adapter({})
    if (
        registry.PHASES,
        runtime.source_geometry.height,
        runtime.source_geometry.width,
    ) != tuple(v2.site.SOURCE_GEOMETRY):
        raise AUHAnonymousObjectProbeV6Error(
            "live runtime geometry differs from registry/site authority"
        )
    binding_receipt = runtime.binding_receipt()
    prompt_bank, negative, prompt_receipt = _encode_anonymous_prompt_bank(
        runtime, rank=rank
    )
    bootstrap_scrub = v2._scrub_site_bootstrap_tensors(runtime)
    initial_gaussian, gaussian_receipt = v2._load_initial_gaussian(runtime)
    native_plan, timesteps, sigmas = v2._real_capture_plan(
        runtime.model.diff_dec, runtime.device
    )
    band_map = {"high": "high", "mid": "mid", "mid_low": "low"}
    cells = tuple(
        SigmaCellV6(band_map[cell.band], cell.step_index, cell.sigma)
        for cell in native_plan.sigma_cells
    )
    if {cell.band: cell.step_index for cell in cells} != dict(
        registry.SIGMA_CELL_INDICES
    ):
        raise AUHAnonymousObjectProbeV6Error("capture schedule differs")
    projection = projection_hook.ProjectionAuthorityV6.create()
    projection.validate()
    _require_all_rank_equal(projection.digest, label="V6 projection authority")
    rank_bank = projection_hook.InMemoryProjectedRankBankV6()
    stream = observer.AnonymousObjectObserverV6()
    state_before = parity._module_state_version_receipt(runtime._transformer)
    base_cells = []
    authority_receipts = []
    capture_receipts = []
    trajectory_steps = []
    cell_receipts = []

    layout = cdf.validate_latent_shape(tuple(initial_gaussian.shape))
    initial_packed = cdf._pack_spatial_latent(initial_gaussian, layout)
    selected_by_index = {cell.step_index: cell for cell in cells}
    compute_dtype = runtime._transformer.patch_embedding.weight.dtype
    if compute_dtype != torch.bfloat16:
        raise AUHAnonymousObjectProbeV6Error("transformer dtype differs")

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
                instructions = {
                    arm: _model_prompt(appearance.captions[arm])
                    for arm in registry.ARMS
                }
                authority = seal_anonymous_same_state_arms_v6(
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
                    instructions=instructions,
                )
                authority_receipts.append(dict(authority.receipt()))

                with torch.inference_mode():
                    base_output = authority.call("action").detach()
                base_sha = parity._tensor_digest(base_output, label="V6 B0 output")
                handle = projection_hook.install_anonymous_visual_projection_hook_v6(
                    runtime._transformer,
                    rank_bank=rank_bank,
                    projection=projection,
                )
                projected_by_arm: dict[
                    str, Mapping[int, observer.AnonymousProjectedArmV6]
                ] = {}
                action_observed: Optional[torch.Tensor] = None
                arm_receipts = []
                try:
                    for arm in registry.ARMS:
                        arm_output, projected, arm_receipt = _capture_one_arm(
                            runtime=runtime,
                            authority=authority,
                            arm=arm,
                            rank=rank,
                            projection=projection,
                            rank_bank=rank_bank,
                        )
                        projected_by_arm[arm] = projected
                        arm_receipts.append(dict(arm_receipt))
                        if arm == "action":
                            action_observed = arm_output
                        else:
                            del arm_output
                except BaseException:
                    _abort_projected_cell(projected_by_arm)
                    raise
                finally:
                    try:
                        handle.restore()
                    except BaseException:
                        _abort_projected_cell(projected_by_arm)
                        raise
                if action_observed is None:
                    _abort_projected_cell(projected_by_arm)
                    raise AUHAnonymousObjectProbeV6Error("observer action is absent")
                action_sha = parity._tensor_digest(
                    action_observed, label="V6 observer action output"
                )
                if not torch.equal(base_output, action_observed) or base_sha != action_sha:
                    _abort_projected_cell(projected_by_arm)
                    raise AUHAnonymousObjectProbeV6Error(
                        "V6 observer changed frozen action output"
                    )
                if any(
                    wrapper.base_calls != len(registry.ARMS)
                    or wrapper.observer_calls != len(registry.ARMS)
                    for wrapper in handle.wrappers
                ):
                    _abort_projected_cell(projected_by_arm)
                    raise AUHAnonymousObjectProbeV6Error("V6 hook calls differ")
                try:
                    reduced = observer.reduce_anonymous_cell_v6(projected_by_arm)
                    stream.add(reduced)
                except BaseException:
                    _abort_projected_cell(projected_by_arm)
                    stream.abort()
                    raise
                cell_receipts.append(dict(reduced.receipt()))
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
                    "used_as_positive_evidence": False,
                }
                base_cells.append(
                    {**base_row, "digest": registry.object_sha256(base_row)}
                )
                capture_receipts.extend(arm_receipts)
                selected_seen.append(step_index)
                del base_output, action_observed, noisy_tokens, rotary, authority

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
        if selected_seen != sorted(selected_by_index):
            raise AUHAnonymousObjectProbeV6Error("appearance capture cells differ")
        del state_packed, solver

    anonymous_result = stream.finalize()
    rank_bank_receipt = rank_bank.receipt()
    state_after = parity._module_state_version_receipt(runtime._transformer)
    if state_after != state_before:
        raise AUHAnonymousObjectProbeV6Error("frozen transformer state changed")
    if (
        rank_bank_receipt["projected_rank_shard_count"]
        != EXPECTED_PROJECTED_BLOCK_CAPTURES
        or rank_bank_receipt["taken_rank_invocation_count"]
        != EXPECTED_OBSERVER_FORWARDS
        or rank_bank_receipt["resident_rank_invocation_count"] != 0
    ):
        raise AUHAnonymousObjectProbeV6Error("V6 rank bank did not close")
    if (
        len(base_cells) != EXPECTED_B0_CELLS
        or len(authority_receipts) != EXPECTED_B0_CELLS
        or len(capture_receipts) != EXPECTED_OBSERVER_FORWARDS
        or len(cell_receipts) != EXPECTED_B0_CELLS
    ):
        raise AUHAnonymousObjectProbeV6Error("V6 receipt matrix differs")
    if anonymous_result["representation_admitted"] is not False:
        raise AUHAnonymousObjectProbeV6Error("V6 crossed representation boundary")

    _require_all_rank_equal(anonymous_result["digest"], label="V6 anonymous result")
    local_summary = {
        "rank": rank,
        "frozen_state_digest": state_before["digest"],
        "anonymous_result_digest": anonymous_result["digest"],
        "base_cell_digest": registry.object_sha256(base_cells),
        "rank_bank_digest": rank_bank_receipt["digest"],
    }
    local_summary = {
        **local_summary,
        "digest": registry.object_sha256(local_summary),
    }
    rank_summaries = _all_rank_rows(local_summary)
    diagnostic_admitted = bool(anonymous_result["diagnostic_component_admitted"])
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "ANONYMOUS_COMPONENT_ADMITTED_9_OF_9_REPRESENTATION_NOT_ADMITTED"
            if diagnostic_admitted
            else "ANONYMOUS_COMPONENT_REJECTED_REPRESENTATION_NOT_ADMITTED"
        ),
        "contract": dict(probe_contract()),
        "source_manifest": dict(source_manifest()),
        "test_source_manifest": dict(test_source_manifest()),
        "site_binding": binding_receipt,
        "site_bootstrap": dict(bootstrap_scrub),
        "initial_gaussian": dict(gaussian_receipt),
        "prompt_bank": dict(prompt_receipt),
        "capture_plan": [dict(cell.receipt()) for cell in cells],
        "trajectory_step_registry": trajectory_steps,
        "trajectory_model_forward_count": EXPECTED_TRAJECTORY_FORWARDS,
        "trajectory_unipc_step_count": EXPECTED_TRAJECTORY_STEPS,
        "frozen_base_probe_forward_count": EXPECTED_B0_CELLS,
        "observer_probe_forward_count": EXPECTED_OBSERVER_FORWARDS,
        "total_frozen_transformer_forward_count": EXPECTED_TOTAL_FROZEN_FORWARDS,
        "frozen_base_cells": base_cells,
        "anonymous_same_state_authorities": authority_receipts,
        "projected_capture_receipts": capture_receipts,
        "reduced_cell_receipts": cell_receipts,
        "rank_summaries": rank_summaries,
        "rank_projection_bank": dict(rank_bank_receipt),
        "anonymous_object_result": anonymous_result,
        "all_nine_B0_action_outputs_observer_bit_exact": True,
        "raw_query_and_hidden_clones_zeroized_after_projection": True,
        "projected_sketches_zeroized_after_cell_reduction": True,
        "success_path_capture_ownership_zeroization_required": True,
        "explicit_capture_ownership_boundary_exception_scrub_required": True,
        "uncovered_exception_policy": (
            "nonzero_exit; any candidate file remains non_authoritative "
            "without external completion seal"
        ),
        "all_allocation_failure_zeroization_claimed": False,
        "receipt_authority_status": "CANDIDATE_PENDING_EXTERNAL_COMPLETION_SEAL",
        "candidate_file_presence_is_completion_authority": False,
        "external_completion_seal_written_by_probe": False,
        "caption_role_token_localization_used": False,
        "fixed_semantic_role_inventory_used": False,
        "all_controls_executed": anonymous_result["all_controls_executed"],
        "diagnostic_component_admitted": diagnostic_admitted,
        "representation_admitted": False,
        "prompt_shuffle_control_executed": False,
        "heldout_transfer_control_executed": False,
        "stable_transferable_action_representation_claimed": False,
        "scientific_claim_authorized": False,
        "site_source_bootstrap_tensors_created_and_scrubbed": True,
        "source_bootstrap_tensor_consumed_by_probe_forward": False,
        "target_inputs_consumed": False,
        "decoder_called": False,
        "renderer_called": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "route_or_injection_called": False,
        "candidate_output_modified": False,
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
        raise AUHAnonymousObjectProbeV6Error(
            f"create-only receipt write failed: {write_status[0]}"
        )
    dist.barrier()
    return value


def _initialize_world4() -> None:
    if dist.is_initialized():
        raise AUHAnonymousObjectProbeV6Error("process group initialized too early")
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise AUHAnonymousObjectProbeV6Error("live probe requires WORLD4")
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
            raise AUHAnonymousObjectProbeV6Error(
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
            )
            + "\n"
        )
        return 0
    if args.output is None:
        raise AUHAnonymousObjectProbeV6Error("live probe requires --output")
    if not GPU_LAUNCH_AUTHORIZED or LAUNCH_BLOCKED_PENDING_INDEPENDENT_AUDIT:
        raise AUHAnonymousObjectProbeV6Error(
            "V6 launch remains blocked pending independent audit"
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


__all__ = [
    "AUHAnonymousObjectProbeV6Error",
    "AnonymousSameStateArmAuthorityV6",
    "EXPECTED_B0_CELLS",
    "EXPECTED_OBSERVER_FORWARDS",
    "EXPECTED_PRELAUNCH_CPU_TESTS",
    "EXPECTED_PROJECTED_BLOCK_CAPTURES",
    "GPU_LAUNCH_AUTHORIZED",
    "LAUNCH_BLOCKED_PENDING_INDEPENDENT_AUDIT",
    "METHOD",
    "SCHEMA_VERSION",
    "SigmaCellV6",
    "probe_contract",
    "remote_launch_template",
    "run_real_world4_probe",
    "seal_anonymous_same_state_arms_v6",
    "source_manifest",
    "test_source_manifest",
]
