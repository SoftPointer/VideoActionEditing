#!/usr/bin/env python3
"""Auditable bridge and CPU dry-run for intermediate action-anchor inference.

The production seam is intentionally narrow: an outer native ``v2v_apg``
adapter calls :class:`NativeIntermediateStepBridge` around the positive
``shared_step`` at selected exact40 steps, then makes one extra canonical
no-op ``shared_step`` using the exact same noisy/timestep/rotary/source-state
objects.  The teacher terminal latent is discarded and never decoded.  A
second native sample calls ``inject_student`` at the corresponding positive
branch.

This file does not silently launch a checkpoint job.  ``--dry-run`` exercises
the complete capture -> object/graph packet -> guarded injection -> P0 replay
lifecycle with a frozen 30-block synthetic transformer.  The same bridge and
SP4 assembler are the code paths intended for the audited AUH wrapper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch
from torch import nn

import self_generated_intermediate_action_anchor_v1 as anchor


METHOD = "bernini-native-intermediate-action-anchor-canary-v1"
SCHEMA_VERSION = "bernini-native-intermediate-action-anchor-canary-receipt-v1"


class NativeIntermediateAnchorCanaryError(RuntimeError):
    """Raised before a dry-run/native adapter may publish evidence."""


_PAIRED_STATE_TOKEN = object()


def _callable_identity(value: Any) -> Mapping[str, Any]:
    function = getattr(value, "__func__", value)
    owner = getattr(value, "__self__", None)
    try:
        signature = str(inspect.signature(value))
    except (TypeError, ValueError) as error:
        raise NativeIntermediateAnchorCanaryError(
            "shared_step signature is not inspectable"
        ) from error
    return {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", None),
        "function_object_id": id(function),
        "owner_object_id": None if owner is None else id(owner),
        "signature": signature,
    }


@dataclass(frozen=True)
class PairedStateForwardAuthority:
    """One callable, one state, and a no-op call differing in text only."""

    shared_step: Callable[..., Any]
    action_args: tuple[Any, ...]
    action_kwargs: Mapping[str, Any]
    noop_args: tuple[Any, ...]
    noop_kwargs: Mapping[str, Any]
    callable_identity: Mapping[str, Any]
    state_object_ids: Mapping[str, int]
    canonical_noop_instruction_sha256: str
    canonical_noop_embedding_sha256: str
    _token: Any = None

    def validate(self) -> None:
        if self._token is not _PAIRED_STATE_TOKEN:
            raise NativeIntermediateAnchorCanaryError(
                "paired-state authority is not authentic"
            )
        anchor.validate_canonical_noop(
            anchor.CANONICAL_NOOP_INSTRUCTION,
            self.canonical_noop_instruction_sha256,
        )
        if _callable_identity(self.shared_step) != dict(self.callable_identity):
            raise NativeIntermediateAnchorCanaryError(
                "paired shared_step callable identity changed"
            )
        signature = inspect.signature(self.shared_step)
        try:
            action = signature.bind(*self.action_args, **dict(self.action_kwargs))
            noop = signature.bind(*self.noop_args, **dict(self.noop_kwargs))
        except TypeError as error:
            raise NativeIntermediateAnchorCanaryError(
                "sealed paired call arguments changed"
            ) from error
        action.apply_defaults()
        noop.apply_defaults()
        required = {
            "model_id",
            "noisy_latents",
            "timesteps",
            "cond_embeds",
            "rotary_embs",
            "batch_vae_seqlen",
            "batch_text_seqlen",
        }
        if not required.issubset(action.arguments) or not required.issubset(
            noop.arguments
        ):
            raise NativeIntermediateAnchorCanaryError(
                "paired shared_step ABI lacks required Bernini arguments"
            )
        if action.arguments["model_id"] != "transformer_1" or noop.arguments[
            "model_id"
        ] != "transformer_1":
            raise NativeIntermediateAnchorCanaryError("paired model route differs")
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            if (
                action.arguments[name] is not noop.arguments[name]
                or id(action.arguments[name]) != self.state_object_ids.get(name)
            ):
                raise NativeIntermediateAnchorCanaryError(
                    f"paired action/no-op {name} is not the sealed object"
                )
        for name, action_value in action.arguments.items():
            if name in {"cond_embeds", "batch_text_seqlen"}:
                continue
            if noop.arguments.get(name) is not action_value:
                raise NativeIntermediateAnchorCanaryError(
                    f"paired calls differ outside text field: {name}"
                )
        if action.arguments["cond_embeds"] is noop.arguments["cond_embeds"]:
            raise NativeIntermediateAnchorCanaryError(
                "action and canonical no-op prompts must be distinct objects"
            )
        for label, prompt in (
            ("action", action.arguments["cond_embeds"]),
            ("no-op", noop.arguments["cond_embeds"]),
        ):
            if (
                not isinstance(prompt, torch.Tensor)
                or prompt.ndim != 3
                or int(prompt.shape[0]) != 1
                or prompt.requires_grad
                or prompt.grad_fn is not None
                or not bool(torch.isfinite(prompt).all().item())
            ):
                raise NativeIntermediateAnchorCanaryError(
                    f"paired {label} prompt tensor differs"
                )
        expected_noop_length = (int(noop.arguments["cond_embeds"].shape[1]),)
        try:
            noop_text_length = tuple(int(value) for value in noop.arguments["batch_text_seqlen"])
        except Exception as error:
            raise NativeIntermediateAnchorCanaryError(
                "paired no-op text metadata differs"
            ) from error
        if noop_text_length != expected_noop_length:
            raise NativeIntermediateAnchorCanaryError(
                "paired no-op text length differs from its embedding"
            )
        if (
            anchor.tensor_sha256(noop.arguments["cond_embeds"])
            != self.canonical_noop_embedding_sha256
        ):
            raise NativeIntermediateAnchorCanaryError(
                "sealed canonical no-op embedding bytes changed"
            )

    def action_call(self) -> Any:
        self.validate()
        return self.shared_step(*self.action_args, **dict(self.action_kwargs))

    def noop_call(self) -> Any:
        self.validate()
        return self.shared_step(*self.noop_args, **dict(self.noop_kwargs))

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        signature = inspect.signature(self.shared_step)
        action = signature.bind(*self.action_args, **dict(self.action_kwargs))
        noop = signature.bind(*self.noop_args, **dict(self.noop_kwargs))
        value = {
            "callable_identity": dict(self.callable_identity),
            "state_object_ids": dict(self.state_object_ids),
            "same_original_callable": True,
            "same_noisy_timestep_rotary_objects": True,
            "only_replaced_fields": ["cond_embeds", "batch_text_seqlen"],
            "action_prompt_object_id": id(action.arguments["cond_embeds"]),
            "noop_prompt_object_id": id(noop.arguments["cond_embeds"]),
            "canonical_noop_instruction_sha256": (
                self.canonical_noop_instruction_sha256
            ),
            "canonical_noop_embedding_sha256": (
                self.canonical_noop_embedding_sha256
            ),
        }
        return {**value, "digest": anchor.object_sha256(value)}


def seal_paired_state_forward(
    *,
    shared_step: Callable[..., Any],
    action_args: Sequence[Any] = (),
    action_kwargs: Optional[Mapping[str, Any]] = None,
    canonical_noop_embeds: torch.Tensor,
    canonical_noop_instruction: str,
    canonical_noop_instruction_sha256: str,
) -> PairedStateForwardAuthority:
    """Derive both calls from one bound action call; no arbitrary no-op closure."""

    if not callable(shared_step):
        raise NativeIntermediateAnchorCanaryError("shared_step must be callable")
    anchor.validate_canonical_noop(
        canonical_noop_instruction, canonical_noop_instruction_sha256
    )
    kwargs = dict(action_kwargs or {})
    signature = inspect.signature(shared_step)
    try:
        action = signature.bind(*tuple(action_args), **kwargs)
        action.apply_defaults()
    except TypeError as error:
        raise NativeIntermediateAnchorCanaryError(
            "action call does not bind to shared_step"
        ) from error
    required = {
        "model_id",
        "noisy_latents",
        "timesteps",
        "cond_embeds",
        "rotary_embs",
        "batch_vae_seqlen",
        "batch_text_seqlen",
    }
    if not required.issubset(action.arguments):
        raise NativeIntermediateAnchorCanaryError(
            "action call lacks pinned shared_step fields"
        )
    noop = signature.bind(*tuple(action_args), **kwargs)
    noop.apply_defaults()
    noop.arguments["cond_embeds"] = canonical_noop_embeds
    noop.arguments["batch_text_seqlen"] = [int(canonical_noop_embeds.shape[1])]
    packet = PairedStateForwardAuthority(
        shared_step=shared_step,
        action_args=tuple(action.args),
        action_kwargs=dict(action.kwargs),
        noop_args=tuple(noop.args),
        noop_kwargs=dict(noop.kwargs),
        callable_identity=_callable_identity(shared_step),
        state_object_ids={
            name: id(action.arguments[name])
            for name in ("noisy_latents", "timesteps", "rotary_embs")
        },
        canonical_noop_instruction_sha256=canonical_noop_instruction_sha256,
        canonical_noop_embedding_sha256=anchor.tensor_sha256(
            canonical_noop_embeds
        ),
        _token=_PAIRED_STATE_TOKEN,
    )
    packet.validate()
    return packet


def native_runtime_contract(config: anchor.AnchorConfig) -> Mapping[str, Any]:
    """Return the exact experiment/control contract before checkpoint loading."""

    config.validate()
    patch_positions = config.patch_positions
    vi_condition_tokens = 25 * patch_positions
    total_tokens = vi_condition_tokens + config.phases * patch_positions
    local_tokens = (total_tokens + 3) // 4
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "model": "Bernini-R 1.3B single transformer_1",
        "base_frozen": True,
        "optimizer": None,
        "parameter_updates": 0,
        "scientific_candidate": False,
        "gpu_launch_authorized": False,
        "guidance_mode": "v2v_apg",
        "num_inference_steps": 40,
        "teacher": {
            "pass": 1,
            "source_conditioned": True,
            "action_prompt": True,
            "same_state_canonical_noop_query_at_selected_steps": True,
            "paired_state_authority": (
                "one sealed original shared_step; only cond_embeds and "
                "batch_text_seqlen may differ"
            ),
            "canonical_noop_string_and_embedding_bytes_sealed": True,
            "capture_blocks": [config.geometry_block, config.semantic_block],
            "capture_steps": list(config.capture_steps),
            "teacher_terminal_latent_discarded": True,
            "teacher_video_decoded": False,
            "teacher_rgb_or_latent_saved": False,
        },
        "representation": {
            "block15": "object activity, soft slots, dynamic centers",
            "block22": "action-minus-noop semantic residual",
            "appearance_null": "remove per-patch temporal DC",
            "camera_null": "remove per-phase spatial common mode",
            "interaction_graph": "phase-wise proximity+semantic row-stochastic graph",
            "packet_abi": [
                f"responsibilities[1,21,{patch_positions},{config.object_slots}]",
                f"activity_gate[1,21,{patch_positions},1]",
                f"slot_values[1,21,{config.object_slots},{config.hidden_size}]",
                f"graph[1,21,{config.object_slots},{config.object_slots}]",
            ],
        },
        "student": {
            "pass": 2,
            "native_source_conditions_reused": True,
            "injection_block": config.semantic_block,
            "injection_branch": "positive_action_only",
            "injection_rows": "target_rows_only; phase0/source/reference/padding protected",
            "scale": config.default_scale,
            "sigma_bandpass": [
                config.sigma_zero_below,
                config.sigma_full_from,
                config.sigma_full_to,
                config.sigma_zero_above,
            ],
            "hidden_rms_ceiling": config.max_injection_to_hidden_rms,
            "temporal_rms_ceiling": config.max_injection_to_temporal_rms,
        },
        "world4_tensor_geometry": {
            "vi_condition_prefix_tokens": vi_condition_tokens,
            "target_tokens": config.phases * patch_positions,
            "global_tokens": total_tokens,
            "padded_local_hidden_shape": [1, local_tokens, config.hidden_size],
            "hook_output_dtype": "native bf16 expected; packet FP32",
        },
        "controls": [
            "FROZEN_BASE_P0a",
            "TEACHER_OBSERVER_ONLY_no_decode",
            "INTERMEDIATE_ANCHOR_CANARY_not_ours",
            "FROZEN_BASE_P0b_exact_replay",
        ],
        "admission_required_before_injection": {
            "distinct_source_reference_views": 2,
            "same_persistent_source_sha256": True,
            "canonical_noop_control": True,
            "teacher_observer_output_bit_exact": True,
            "frozen_state_unchanged_through_teacher": True,
            "target_inputs_absent": True,
        },
        "known_representation_limits": {
            "classification": "compressed_hidden_residual_canary_not_OCEG",
            "fixed_slot_count": config.object_slots,
            "learned_persistent_entity_binding": False,
            "typed_interaction_edges": False,
            "entity_state_machine": False,
            "cross_case_transfer_established": False,
            "native_tokenizer_to_noop_embedding_callgraph_authenticated": False,
        },
        "target_isolation": {
            "real_target_inputs": False,
            "target_media_features_latents_qkv_masks_tracks_flow": False,
            "only_source_video_action_text_and_canonical_noop": True,
        },
        "config_digest": config.receipt()["digest"],
    }
    return {**value, "digest": anchor.object_sha256(value)}


def distributed_assemble_local_delta(
    local_delta: torch.Tensor,
    layout: anchor.LocalTokenLayout,
    *,
    process_group: Any = None,
) -> torch.Tensor:
    """NCCL-friendly SP4 gather of one target-only action-minus-noop shard.

    Each rank receives the same assembled detached grid.  Only one block delta
    should be gathered at a time; callers release the temporary padded lists
    before gathering the next block.
    """

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        if layout.sp_size != 1:
            raise NativeIntermediateAnchorCanaryError(
                "SP4 assembly requires initialized torch.distributed"
            )
        return anchor.assemble_sp_target_grid([local_delta], [layout])
    world_size = dist.get_world_size(group=process_group)
    rank = dist.get_rank(group=process_group)
    if world_size != 4 or layout.sp_size != 4 or layout.sp_rank != rank:
        raise NativeIntermediateAnchorCanaryError(
            "distributed assembler requires authenticated WORLD4/SP4 ranks"
        )
    if (
        not isinstance(local_delta, torch.Tensor)
        or local_delta.ndim != 3
        or int(local_delta.shape[0]) != 1
        or int(local_delta.shape[1]) != int(layout.local_target_indices.numel())
        or local_delta.device.type != "cuda"
        or local_delta.requires_grad
        or not bool(torch.isfinite(local_delta).all().item())
    ):
        raise NativeIntermediateAnchorCanaryError(
            "distributed local target delta geometry differs"
        )
    count = torch.tensor(
        [int(local_delta.shape[1])], dtype=torch.int64, device=local_delta.device
    )
    count_rows = [torch.empty_like(count) for _ in range(world_size)]
    dist.all_gather(count_rows, count, group=process_group)
    counts = [int(row.item()) for row in count_rows]
    maximum = max(counts)
    padded = torch.zeros(
        (1, maximum, int(local_delta.shape[2])),
        dtype=local_delta.dtype,
        device=local_delta.device,
    )
    padded[:, : int(local_delta.shape[1]), :].copy_(local_delta)
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=process_group)
    layouts = [
        anchor.LocalTokenLayout.build(
            condition_tokens=layout.condition_tokens,
            patch_height=layout.patch_height,
            patch_width=layout.patch_width,
            phases=layout.phases,
            sp_rank=candidate_rank,
            sp_size=4,
        )
        for candidate_rank in range(4)
    ]
    shards = [
        gathered[candidate_rank][:, : counts[candidate_rank], :].contiguous()
        for candidate_rank in range(4)
    ]
    result = anchor.assemble_sp_target_grid(shards, layouts)
    del gathered, padded, shards
    return result.detach().contiguous()


class NativeIntermediateStepBridge:
    """Pair same-state block captures and inject a packet on a second pass."""

    def __init__(
        self,
        controller: anchor.IntermediateAnchorHookController,
        *,
        assembler: Callable[[torch.Tensor, anchor.LocalTokenLayout], torch.Tensor],
    ) -> None:
        if not isinstance(controller, anchor.IntermediateAnchorHookController):
            raise NativeIntermediateAnchorCanaryError("hook controller differs")
        if not callable(assembler):
            raise NativeIntermediateAnchorCanaryError("SP assembler must be callable")
        self.controller = controller
        self.config = controller.config
        self.assembler = assembler
        self.bank = anchor.IntermediateAnchorTrajectoryBank(self.config)
        self.teacher_steps = 0
        self.student_steps = 0
        self.same_state_noop_queries = 0
        self.paired_state_receipts: list[Mapping[str, Any]] = []

    def capture_teacher_step(
        self,
        *,
        step_index: int,
        sigma: float,
        layout: anchor.LocalTokenLayout,
        paired_state: PairedStateForwardAuthority,
    ) -> Any:
        """Return the untouched action prediction and retain only its packet."""

        if step_index not in self.config.capture_steps:
            raise NativeIntermediateAnchorCanaryError(
                "bridge called on an unselected teacher step"
            )
        paired_state.validate()
        with self.controller.invoke(
            anchor.HookInvocation(
                mode="capture_action",
                step_index=step_index,
                sigma=sigma,
                layout=layout,
            )
        ):
            action_prediction = paired_state.action_call()
        action_capture = self.controller.pop_captures()
        with self.controller.invoke(
            anchor.HookInvocation(
                mode="capture_noop",
                step_index=step_index,
                sigma=sigma,
                layout=layout,
            )
        ):
            # The no-op prediction is deliberately discarded.  Its block
            # observations exist only long enough to form action-minus-noop.
            paired_state.noop_call()
        noop_capture = self.controller.pop_captures()
        grids: dict[int, torch.Tensor] = {}
        for block_index in (self.config.geometry_block, self.config.semantic_block):
            local_delta = (
                action_capture[block_index] - noop_capture[block_index]
            ).detach().contiguous()
            grids[block_index] = self.assembler(local_delta, layout)
        zeros_geometry = torch.zeros_like(grids[self.config.geometry_block])
        zeros_semantic = torch.zeros_like(grids[self.config.semantic_block])
        packet = anchor.build_intermediate_action_anchor(
            geometry_action=grids[self.config.geometry_block].detach(),
            geometry_noop=zeros_geometry,
            semantic_action=grids[self.config.semantic_block].detach(),
            semantic_noop=zeros_semantic,
            config=self.config,
            step_index=step_index,
            sigma=sigma,
        )
        self.bank.add(packet)
        self.teacher_steps += 1
        self.same_state_noop_queries += 1
        self.paired_state_receipts.append(paired_state.receipt())
        del action_capture, noop_capture, grids
        return action_prediction

    def inject_student_step(
        self,
        *,
        step_index: int,
        sigma: float,
        layout: anchor.LocalTokenLayout,
        scale: float,
        admission: anchor.MultiViewControlAdmission,
        source_video_sha256: str,
        student_forward: Callable[[], Any],
    ) -> Any:
        packet = self.bank.get(step_index)
        if packet is None:
            raise NativeIntermediateAnchorCanaryError(
                "student step has no exact-step teacher packet"
            )
        if scale <= 0.0:
            raise NativeIntermediateAnchorCanaryError(
                "P0/scale-zero must bypass hook installation entirely"
            )
        with self.controller.invoke(
            anchor.HookInvocation(
                mode="inject_student",
                step_index=step_index,
                sigma=sigma,
                layout=layout,
                scale=scale,
                packet=packet,
                admission=admission,
                source_video_sha256=source_video_sha256,
            )
        ):
            result = student_forward()
        self.student_steps += 1
        return result

    def receipt(self, *, require_complete: bool) -> Mapping[str, Any]:
        if require_complete:
            self.bank.assert_complete()
        packet_steps = sorted(self.bank._packets)
        audits = self.controller.pop_audits()
        value = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "teacher_step_count": self.teacher_steps,
            "student_step_count": self.student_steps,
            "same_state_noop_query_count": self.same_state_noop_queries,
            "paired_state_authorities": list(self.paired_state_receipts),
            "packet_steps": packet_steps,
            "trajectory_complete": packet_steps == list(self.config.capture_steps),
            "injection_audits": [row.receipt() for row in audits],
            "teacher_action_prediction_forwarded_unchanged": True,
            "noop_prediction_discarded": True,
            "teacher_terminal_latent_retained": False,
            "teacher_video_decoded": False,
        }
        return {**value, "digest": anchor.object_sha256(value)}


class _DryBlock(nn.Module):
    def __init__(self, width: int, index: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.full((width,), float(index) * 1.0e-3))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.bias.reshape(1, 1, -1).to(hidden.dtype)


class _DryTransformer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_DryBlock(width, index) for index in range(30)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


def _dry_run(scale: float) -> Mapping[str, Any]:
    config = anchor.AnchorConfig(
        patch_height=6,
        patch_width=8,
        hidden_size=32,
        object_slots=4,
        capture_steps=(8,),
        default_scale=scale,
        min_retained_fraction=1.0e-7,
    )
    config.validate()
    transformer = _DryTransformer(config.hidden_size).eval().requires_grad_(False)
    freeze_before = anchor.frozen_module_certificate(transformer)
    condition_tokens = 25 * config.patch_positions
    layout = anchor.LocalTokenLayout.build(
        condition_tokens=condition_tokens,
        patch_height=config.patch_height,
        patch_width=config.patch_width,
        observed_local_length=46 * config.patch_positions,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260822)
    source_state = torch.randn(
        (1, layout.local_length, config.hidden_size), generator=generator
    )
    p0a = transformer(source_state.clone()).detach()

    controller = anchor.IntermediateAnchorHookController(transformer, config)
    controller.install()
    bridge = NativeIntermediateStepBridge(
        controller,
        assembler=lambda local, local_layout: anchor.assemble_sp_target_grid(
            [local], [local_layout]
        ),
    )
    action_input = source_state.clone()
    noop_input = source_state.clone()
    for phase in range(1, config.phases):
        first_patch = 1 * config.patch_width + min(config.patch_width - 1, phase // 3)
        second_patch = 4 * config.patch_width + max(0, 7 - phase // 4)
        for patch_index, channel_start, sign in (
            (first_patch, 0, 1.0),
            (second_patch, 16, -1.0),
        ):
            global_row = condition_tokens + phase * config.patch_positions + patch_index
            action_input[0, global_row, channel_start : channel_start + 8] += (
                sign * (0.4 + phase / 30.0)
            )
    shared_noisy = torch.randn(1, 2, 3, generator=generator)
    shared_timestep = torch.tensor([550.0])
    shared_rotary = torch.randn(1, 1, 2, 4, generator=generator)

    def shared_step(
        model_id: str,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_embeds: torch.Tensor,
        rotary_embs: torch.Tensor,
        batch_vae_seqlen: Sequence[int],
        batch_text_seqlen: Sequence[int],
    ) -> torch.Tensor:
        del noisy_latents, timesteps, rotary_embs, batch_vae_seqlen, batch_text_seqlen
        if model_id != "transformer_1":
            raise NativeIntermediateAnchorCanaryError("dry model route differs")
        return transformer(cond_embeds)

    paired_state = seal_paired_state_forward(
        shared_step=shared_step,
        action_kwargs={
            "model_id": "transformer_1",
            "noisy_latents": shared_noisy,
            "timesteps": shared_timestep,
            "cond_embeds": action_input,
            "rotary_embs": shared_rotary,
            "batch_vae_seqlen": [layout.total_tokens],
            "batch_text_seqlen": [int(action_input.shape[1])],
        },
        canonical_noop_embeds=noop_input,
        canonical_noop_instruction=anchor.CANONICAL_NOOP_INSTRUCTION,
        canonical_noop_instruction_sha256=anchor.CANONICAL_NOOP_SHA256,
    )
    teacher_prediction = bridge.capture_teacher_step(
        step_index=8,
        sigma=0.55,
        layout=layout,
        paired_state=paired_state,
    )
    unhooked_teacher_prediction = transformer(action_input)
    if not anchor.bits_equal(teacher_prediction, unhooked_teacher_prediction):
        raise NativeIntermediateAnchorCanaryError(
            "read-only teacher observation changed the action prediction"
        )
    # A second source-derived reference view carries a shared static offset in
    # both action/no-op calls.  The action contrast should survive while that
    # view-specific appearance cancels.  This is only a synthetic admission
    # dry-run, not evidence that the real checkpoint passes the gate.
    alternate_bridge = NativeIntermediateStepBridge(
        controller,
        assembler=lambda local, local_layout: anchor.assemble_sp_target_grid(
            [local], [local_layout]
        ),
    )
    static_view_offset = torch.linspace(
        -0.05, 0.05, config.hidden_size
    ).reshape(1, 1, -1)
    noop_input_alternate = noop_input + static_view_offset
    action_input_alternate = noop_input_alternate + 1.01 * (
        action_input - noop_input
    )
    alternate_noisy = torch.randn(1, 2, 3, generator=generator)
    alternate_timestep = torch.tensor([550.0])
    alternate_rotary = torch.randn(1, 1, 2, 4, generator=generator)
    alternate_paired = seal_paired_state_forward(
        shared_step=shared_step,
        action_kwargs={
            "model_id": "transformer_1",
            "noisy_latents": alternate_noisy,
            "timesteps": alternate_timestep,
            "cond_embeds": action_input_alternate,
            "rotary_embs": alternate_rotary,
            "batch_vae_seqlen": [layout.total_tokens],
            "batch_text_seqlen": [int(action_input_alternate.shape[1])],
        },
        canonical_noop_embeds=noop_input_alternate,
        canonical_noop_instruction=anchor.CANONICAL_NOOP_INSTRUCTION,
        canonical_noop_instruction_sha256=anchor.CANONICAL_NOOP_SHA256,
    )
    alternate_bridge.capture_teacher_step(
        step_index=8,
        sigma=0.55,
        layout=layout,
        paired_state=alternate_paired,
    )
    freeze_after_teacher = anchor.frozen_module_certificate(transformer)
    if freeze_before["digest"] != freeze_after_teacher["digest"]:
        raise NativeIntermediateAnchorCanaryError(
            "frozen base changed during multi-view teacher capture"
        )
    source_sha256 = "0" * 64
    primary_packet = bridge.bank.get(8)
    alternate_packet = alternate_bridge.bank.get(8)
    if primary_packet is None or alternate_packet is None:
        raise NativeIntermediateAnchorCanaryError("dry multi-view packet is absent")
    admission = anchor.admit_multiview_control(
        primary=anchor.SourceViewPacketEvidence(
            view_id="source_refs_0_27_53_80",
            source_video_sha256=source_sha256,
            reference_frame_indices=(0, 27, 53, 80),
            packet=primary_packet,
        ),
        alternate=anchor.SourceViewPacketEvidence(
            view_id="source_refs_0_20_60_80",
            source_video_sha256=source_sha256,
            reference_frame_indices=(0, 20, 60, 80),
            packet=alternate_packet,
        ),
        controls=anchor.MultiViewControlEvidence(
            noop_vs_noop_delta_rms=0.0,
            action_delta_rms_reference=float(
                primary_packet.quality["teacher_delta_rms"]
            ),
            teacher_observer_action_output_bit_exact=True,
            frozen_state_before_sha256=freeze_before["digest"],
            frozen_state_after_teacher_sha256=freeze_after_teacher["digest"],
            target_inputs_absent=True,
        ),
    )
    student_native = transformer(source_state.clone())
    student_injected = bridge.inject_student_step(
        step_index=8,
        sigma=0.55,
        layout=layout,
        scale=scale,
        admission=admission,
        source_video_sha256=source_sha256,
        student_forward=lambda: transformer(source_state.clone()),
    )
    if anchor.bits_equal(student_native, student_injected):
        raise NativeIntermediateAnchorCanaryError(
            "nonzero dry-run injection did not change any admitted target row"
        )
    controller.remove()
    p0b = transformer(source_state.clone()).detach()
    replay = anchor.assert_p0_exact_replay(p0a, p0b)
    freeze_after = anchor.frozen_module_certificate(transformer)
    if freeze_before["digest"] != freeze_after["digest"]:
        raise NativeIntermediateAnchorCanaryError("frozen base state changed")
    bridge_receipt = bridge.receipt(require_complete=True)
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "dry_run": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "native_contract": native_runtime_contract(config),
        "teacher_student_bridge": bridge_receipt,
        "alternate_view_teacher_bridge": alternate_bridge.receipt(
            require_complete=True
        ),
        "multi_view_control_admission": admission.receipt(),
        "p0_exact_replay": replay,
        "freeze_before_digest": freeze_before["digest"],
        "freeze_after_digest": freeze_after["digest"],
        "frozen_base_unchanged": True,
        "teacher_observer_output_exact": True,
        "student_changed": True,
        "target_inputs_consumed": False,
        "teacher_decode_called": False,
        "gpu_launch_authorized": False,
        "representation_classification": (
            "compressed_hidden_residual_canary_not_OCEG"
        ),
    }
    return {**value, "digest": anchor.object_sha256(value)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the frozen synthetic capture/injection/P0 lifecycle",
    )
    parser.add_argument(
        "--print-native-contract",
        action="store_true",
        help="print the production 368x656 WORLD4 tensor/control contract",
    )
    parser.add_argument("--scale", type=float, default=0.06)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run == args.print_native_contract:
        raise NativeIntermediateAnchorCanaryError(
            "select exactly one of --dry-run/--print-native-contract"
        )
    if not 0.0 < args.scale <= 0.25:
        raise NativeIntermediateAnchorCanaryError("scale must lie in (0,.25]")
    if args.dry_run:
        value = _dry_run(float(args.scale))
    else:
        config = anchor.AnchorConfig(patch_height=23, patch_width=41)
        value = native_runtime_contract(config)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output_json is not None:
        output = args.output_json.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
