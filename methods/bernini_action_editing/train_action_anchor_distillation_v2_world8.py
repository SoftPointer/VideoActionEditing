#!/usr/bin/env python3
"""Executable WORLD8 adapter for the frozen action-anchor V2 canary.

This is the missing model/runtime entry point.  It authenticates the complete
16-record sidecar before importing or allocating Bernini, builds the same
Bernini-R/large-LoRA/typed-patch/exact30-hook runtime as the 0817 runner, and
then calls the sealed two-update V2 core.  It never creates teachers or grants
qualification.  A missing or malformed q_y/q_anchor authority closure fails
before distributed initialization, model allocation, or optimizer creation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_action_anchor_distillation_v2_canary as v2


METHOD = "bernini-action-anchor-distillation-v2-world8"
RECEIPT_SCHEMA = "bernini-action-anchor-v2-world8-canary-receipt-v1"
AUTHORITY = v2.AUTHORITY
LORA_SCOPE = "all-attention"


class ActionAnchorV2World8Error(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise ActionAnchorV2World8Error(message)


def _sidecar_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "renderer_release_manifest_path": Path(args.release_manifest),
        "expected_manifest_file_sha256": args.expected_sidecar_manifest_file_sha256,
        "expected_renderer_release_manifest_sha256": (
            args.expected_release_manifest_sha256
        ),
        "expected_teacher_authority_file_sha256": (
            args.expected_teacher_authority_file_sha256
        ),
        "expected_teacher_authority_sha256": args.expected_teacher_authority_sha256,
        "expected_classification_authority_sha256": (
            args.expected_classification_authority_sha256
        ),
        "expected_predictor_source_sha256": args.expected_predictor_source_sha256,
        "expected_distillation_source_sha256": (
            args.expected_distillation_source_sha256
        ),
        "expected_renderer_runner_source_sha256": (
            args.expected_renderer_runner_source_sha256
        ),
        "expected_v2_runner_source_sha256": args.expected_v2_runner_source_sha256,
        "expected_schedule_source_sha256": args.expected_schedule_source_sha256,
        "expected_packed_core_source_sha256": args.expected_packed_core_source_sha256,
        "expected_runtime_source_sha256": args.expected_runtime_source_sha256,
        "expected_legacy_loader_source_sha256": (
            args.expected_legacy_loader_source_sha256
        ),
        "expected_world8_adapter_source_sha256": (
            args.expected_world8_adapter_source_sha256
        ),
        "expected_inference_sigma_source_sha256": (
            args.expected_inference_sigma_source_sha256
        ),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--checkpoint-content-manifest", required=True)
    value.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    value.add_argument("--release-manifest", required=True)
    value.add_argument("--expected-release-manifest-sha256", required=True)
    value.add_argument("--expected-bernini-commit", required=True)
    value.add_argument("--expected-veomni-commit", required=True)
    value.add_argument("--expected-checkpoint-tree-sha256", required=True)
    value.add_argument("--sidecar-manifest", required=True)
    value.add_argument("--expected-sidecar-manifest-file-sha256", required=True)
    value.add_argument("--expected-teacher-authority-file-sha256", required=True)
    value.add_argument("--expected-teacher-authority-sha256", required=True)
    value.add_argument("--expected-classification-authority-sha256", required=True)
    value.add_argument("--expected-predictor-source-sha256", required=True)
    value.add_argument("--expected-distillation-source-sha256", required=True)
    value.add_argument("--expected-renderer-runner-source-sha256", required=True)
    value.add_argument("--expected-v2-runner-source-sha256", required=True)
    value.add_argument("--expected-schedule-source-sha256", required=True)
    value.add_argument("--expected-packed-core-source-sha256", required=True)
    value.add_argument("--expected-runtime-source-sha256", required=True)
    value.add_argument("--expected-legacy-loader-source-sha256", required=True)
    value.add_argument("--expected-world8-adapter-source-sha256", required=True)
    value.add_argument("--expected-inference-sigma-source-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--learning-rate", type=float, default=1.0e-4)
    value.add_argument("--max-grad-norm", type=float, default=1.0)
    value.add_argument("--seed", type=int, default=20260818)
    value.add_argument("--smooth-l1-weight", type=float, default=1.0)
    value.add_argument("--cosine-weight", type=float, default=1.0)
    value.add_argument("--infonce-weight", type=float, default=1.0)
    value.add_argument("--flow-weight", type=float, default=1.0)
    value.add_argument("--temperature", type=float, default=0.07)
    value.add_argument("--ack-exploratory-two-update-only", action="store_true")
    value.add_argument("--ack-no-formal-or-scientific-claim", action="store_true")
    value.add_argument("--ack-terminal-non-resumable-canary", action="store_true")
    return value


def validate_args(args: argparse.Namespace) -> None:
    if not (
        args.ack_exploratory_two_update_only
        and args.ack_no_formal_or_scientific_claim
        and args.ack_terminal_non_resumable_canary
    ):
        fail("all V2 exploratory canary acknowledgements are mandatory")
    for name in (
        "expected_checkpoint_content_manifest_sha256",
        "expected_release_manifest_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_sidecar_manifest_file_sha256",
        "expected_teacher_authority_file_sha256",
        "expected_teacher_authority_sha256",
        "expected_classification_authority_sha256",
        "expected_predictor_source_sha256",
        "expected_distillation_source_sha256",
        "expected_renderer_runner_source_sha256",
        "expected_v2_runner_source_sha256",
        "expected_schedule_source_sha256",
        "expected_packed_core_source_sha256",
        "expected_runtime_source_sha256",
        "expected_legacy_loader_source_sha256",
        "expected_world8_adapter_source_sha256",
        "expected_inference_sigma_source_sha256",
    ):
        v2._sha256(getattr(args, name), label=name, authority=True)
    if (
        type(args.seed) is not int
        or not 0 <= args.seed < 2**63
        or any(
            not math.isfinite(float(value))
            for value in (
                args.learning_rate,
                args.max_grad_norm,
                args.smooth_l1_weight,
                args.cosine_weight,
                args.infonce_weight,
                args.flow_weight,
                args.temperature,
            )
        )
    ):
        fail("V2 numeric launch contract differs")
    v2._validate_loss_config_v2(
        {
            "smooth_l1_weight": args.smooth_l1_weight,
            "cosine_weight": args.cosine_weight,
            "infonce_weight": args.infonce_weight,
            "flow_weight": args.flow_weight,
            "temperature": args.temperature,
        }
    )
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        fail("V2 optimizer hyperparameters must be positive")
    output = Path(args.output).expanduser()
    if (
        not output.is_absolute()
        or output == Path("/")
        or output.suffix
        or "action_anchor_v2_canary" not in output.name
    ):
        fail("output must be one absolute suffix-free action_anchor_v2_canary path")


def _materialize_local_inputs(
    *,
    preflight: v2.FrozenSidecarPreflightV2,
    distributed: Any,
    base_renderer: Any,
    tokenizer: Any,
    runtime: Any,
    renderer_v1: Any,
    device: Any,
) -> tuple[v2.RuntimeRecordInputV2, ...]:
    import torch

    local: list[v2.RuntimeRecordInputV2] = []
    for record in preflight.records[distributed.arm_index :: v2.DP_SIZE]:
        payload = v2.load_frozen_runtime_payload_v2(record)
        tokenized = runtime.tokenize_generic_instruction(
            tokenizer, payload.instruction, device
        )
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                text_lens, text_embs = base_renderer.get_t5_text_embeddings(
                    tokenized["input_ids"],
                    tokenized["attention_mask"],
                    tokenized["t5_input_lens"],
                )
        if isinstance(text_embs, (list, tuple)):
            if len(text_embs) != 1:
                fail("official UMT5 operator view must contain one embedding")
            text_embs = text_embs[0]
        actual_length = int(tokenized["t5_input_lens"].item())
        if (
            int(getattr(base_renderer, "max_sequence_length", 0)) != 512
            or [int(value) for value in text_lens] != [512]
            or list(text_embs.shape) != [1, 512, 4096]
            or text_embs.dtype != torch.bfloat16
            or text_embs.requires_grad
            or not torch.is_inference(text_embs)
            or not bool(torch.isfinite(text_embs).all().item())
            or not 0 < actual_length <= 512
        ):
            fail("frozen UMT5 renderer/tokenizer contract differs")
        text_embs = renderer_v1.materialize_training_text_embedding(text_embs)
        if isinstance(text_lens, torch.Tensor):
            text_lens = text_lens.detach().clone(memory_format=torch.contiguous_format)
        instruction_tokens = renderer_v1.canonical_instruction_tokens(
            text_embs, tokenized["t5_input_lens"]
        )
        local.append(
            v2.RuntimeRecordInputV2(
                logical_record=record.logical_record,
                dataset_iid=record.dataset_iid,
                dataset_row_index=record.dataset_row_index,
                source_media_path=record.source_media_path,
                target_media_path=record.target_media_path,
                source_mode=payload.source_mode,
                target_mode=payload.target_mode,
                instruction=payload.instruction,
                text_lens=text_lens,
                text_embs=text_embs,
                instruction_tokens=instruction_tokens,
            )
        )
    return tuple(local)


def _persist_terminal_result(
    *,
    output: Path,
    stage: Path,
    result: v2.CanaryRunResultV2,
    model: Any,
    conditioner: Any,
    preflight: v2.FrozenSidecarPreflightV2,
    args: argparse.Namespace,
    distributed: Any,
    parallel: Any,
    runtime: Any,
    renderer_v1: Any,
    release_closure: Mapping[str, Any],
    checkpoint_content: Mapping[str, Any],
) -> None:
    import torch.distributed as dist

    local_evidence = {
        "world_rank": distributed.rank,
        "dp_arm": distributed.arm_index,
        "sp_rank": distributed.sp_rank,
        "history": list(result.history),
        "parameter_sha256_p0_p1_p2": list(result.parameter_sha256_p0_p1_p2),
    }
    world: list[Any] = [None] * v2.WORLD_SIZE
    dist.all_gather_object(world, local_evidence, group=parallel.world_group)
    _validate_terminal_world8_evidence_v2(world)
    launch_contract = {
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "loss": dict(
            v2._validate_loss_config_v2(
                {
                    "smooth_l1_weight": args.smooth_l1_weight,
                    "cosine_weight": args.cosine_weight,
                    "infonce_weight": args.infonce_weight,
                    "flow_weight": args.flow_weight,
                    "temperature": args.temperature,
                }
            )
        ),
        "bernini_commit": args.expected_bernini_commit,
        "veomni_commit": args.expected_veomni_commit,
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_content_manifest_sha256": (
            args.expected_checkpoint_content_manifest_sha256
        ),
        "checkpoint_content": dict(checkpoint_content),
        "renderer_release_closure": dict(release_closure),
    }

    receipt: Optional[Mapping[str, Any]] = None
    rank_zero_error: Optional[str] = None
    if distributed.rank == 0:
        try:
            from safetensors.torch import save_file

            named = renderer_v1.exact_trainable_named_parameters(model, conditioner)
            terminal_state = dict(renderer_v1.export_trainable_state(named))
            save_file(terminal_state, str(stage / "adapter.safetensors"))
            runtime.atomic_torch_save(
                stage / "optimizer.pt",
                {
                    "schema_version": RECEIPT_SCHEMA,
                    "optimizer_state_retained": False,
                    "resumable": False,
                    "reason": "terminal_exact_two_update_exploratory_canary",
                },
            )
            runtime.atomic_json(
                stage / "history.json",
                {
                    "schema_version": RECEIPT_SCHEMA,
                    "launch_contract": launch_contract,
                    "world8": world,
                },
            )
            artifacts = {
                name: runtime.file_sha256(stage / name)
                for name in ("adapter.safetensors", "optimizer.pt", "history.json")
            }
            unsigned = {
                "schema_version": RECEIPT_SCHEMA,
                "method": METHOD,
                "authority": AUTHORITY,
                "complete": True,
                "exploratory_only": True,
                "formal_training_authorized": False,
                "scientific_claim_authorized": False,
                "optimizer_updates": v2.MAX_UPDATES,
                "optimizer_state_retained": False,
                "resumable": False,
                "parameter_sha256_p0_p1_p2": list(
                    result.parameter_sha256_p0_p1_p2
                ),
                "launch_contract": launch_contract,
                "sidecar_preflight": preflight.receipt(),
                "world8_evidence_sha256": v2.object_sha256(world),
                "artifacts": artifacts,
            }
            receipt = {**unsigned, "receipt_digest": v2.object_sha256(unsigned)}
            runtime.atomic_json(stage / "receipt.json", receipt)
        except Exception as error:
            rank_zero_error = f"{type(error).__name__}: {error}"
    runtime.publish_output_transaction(
        output,
        stage,
        receipt,
        distributed.rank,
        parallel.world_group,
        rank_zero_error=rank_zero_error,
    )


def _expected_local_phase_digest_v2(route: Mapping[str, Any], sp_rank: int) -> str:
    source = route.get("source_tokens")
    target = route.get("target_tokens")
    if (
        type(source) is not int
        or type(target) is not int
        or source <= 0
        or target != source
        or target % v2.PHASE_COUNT
        or type(sp_rank) is not int
        or not 0 <= sp_rank < v2.SP_SIZE
    ):
        fail("V2 terminal route geometry differs")
    local_length = math.ceil((source + target) / v2.SP_SIZE)
    spatial = target // v2.PHASE_COUNT
    phases = []
    for global_index in range(sp_rank * local_length, (sp_rank + 1) * local_length):
        target_index = global_index - source
        phases.append(target_index // spatial if 0 <= target_index < target else -1)
    return v2.object_sha256(phases)


def _validate_terminal_world8_evidence_v2(world: Sequence[Mapping[str, Any]]) -> None:
    """Validate rank-common evidence while preserving rank-specific SP routes."""

    route_fields = {
        "row_identity",
        "source_tokens",
        "target_tokens",
        "spatial_tokens_per_phase",
        "sequence_parallel_rank",
        "sequence_parallel_size",
        "local_phase_indices_sha256",
        "block_call_counts",
        "checkpoint_context_captures",
        "checkpoint_forward_contexts",
        "checkpoint_recompute_contexts",
        "checkpoint_recompute_calls_per_block",
        "exact_block_set_0_through_29",
        "source_or_padding_written",
    }
    if (
        type(world) not in (list, tuple)
        or len(world) != v2.WORLD_SIZE
        or any(type(row) is not dict for row in world)
        or [row.get("world_rank") for row in world] != list(range(v2.WORLD_SIZE))
    ):
        fail("V2 terminal WORLD8 evidence rank closure differs")
    reference_parameters = world[0].get("parameter_sha256_p0_p1_p2")
    if (
        type(reference_parameters) is not list
        or len(reference_parameters) != 3
        or len(set(reference_parameters)) != 3
        or any(v2._SHA256.fullmatch(str(value)) is None for value in reference_parameters)
    ):
        fail("V2 terminal P0/P1/P2 digest closure differs")
    observed: list[int] = []
    for rank, row in enumerate(world):
        arm = rank // v2.SP_SIZE
        sp_rank = rank % v2.SP_SIZE
        history = row.get("history")
        if (
            row.get("dp_arm") != arm
            or row.get("sp_rank") != sp_rank
            or row.get("parameter_sha256_p0_p1_p2") != reference_parameters
            or type(history) is not list
            or len(history) != v2.MAX_UPDATES
        ):
            fail("V2 terminal WORLD8 parameter/topology evidence differs")
        reference_history = world[arm * v2.SP_SIZE].get("history")
        if type(reference_history) is not list or len(reference_history) != v2.MAX_UPDATES:
            fail("V2 terminal SP4 reference history differs")
        for step_index, (step, reference_step) in enumerate(
            zip(history, reference_history)
        ):
            if type(step) is not dict or type(reference_step) is not dict:
                fail("V2 terminal step evidence differs")
            logical = list(
                range(
                    step_index * v2.DP_SIZE * v2.GRADIENT_ACCUMULATION + arm,
                    (step_index + 1) * v2.DP_SIZE * v2.GRADIENT_ACCUMULATION,
                    v2.DP_SIZE,
                )
            )
            if step.get("logical_records") != logical:
                fail("V2 terminal DP-arm logical records differ")
            if rank == arm * v2.SP_SIZE:
                observed.extend(logical)
            common = dict(step)
            reference_common = dict(reference_step)
            objectives = common.pop("microbatch_objectives", None)
            reference_objectives = reference_common.pop(
                "microbatch_objectives", None
            )
            if (
                common != reference_common
                or type(objectives) is not list
                or type(reference_objectives) is not list
                or len(objectives) != v2.GRADIENT_ACCUMULATION
                or len(reference_objectives) != v2.GRADIENT_ACCUMULATION
            ):
                fail("V2 terminal SP4 common step/objective evidence differs")
            for logical_record, objective, reference_objective in zip(
                logical, objectives, reference_objectives
            ):
                if type(objective) is not dict or type(reference_objective) is not dict:
                    fail("V2 terminal microbatch objective evidence differs")
                objective_common = dict(objective)
                reference_objective_common = dict(reference_objective)
                route = objective_common.pop("route", None)
                reference_route = reference_objective_common.pop("route", None)
                if (
                    objective_common != reference_objective_common
                    or objective.get("logical_record") != logical_record
                    or objective.get("point_pair_count") != 1
                    or type(objective.get("contrastive_positive_pair_count"))
                    is not int
                    or objective.get("contrastive_positive_pair_count") < 0
                    or type(objective.get("contrastive_negative_pair_count"))
                    is not int
                    or objective.get("contrastive_negative_pair_count") < 1
                    or type(objective.get("excluded_pair_count")) is not int
                    or objective.get("excluded_pair_count") < 0
                    or objective.get("active_q_anchor_infonce") is not True
                    or type(route) is not dict
                    or type(reference_route) is not dict
                    or set(route) != route_fields
                    or set(reference_route) != route_fields
                ):
                    fail("V2 terminal SP4 common microbatch evidence differs")
                route_common = dict(route)
                reference_route_common = dict(reference_route)
                route_common.pop("sequence_parallel_rank")
                route_common.pop("local_phase_indices_sha256")
                reference_route_common.pop("sequence_parallel_rank")
                reference_route_common.pop("local_phase_indices_sha256")
                if (
                    route_common != reference_route_common
                    or route.get("row_identity") != objective.get("row_id")
                    or route.get("sequence_parallel_rank") != sp_rank
                    or route.get("sequence_parallel_size") != v2.SP_SIZE
                    or route.get("spatial_tokens_per_phase")
                    != route.get("target_tokens") // v2.PHASE_COUNT
                    or route.get("local_phase_indices_sha256")
                    != _expected_local_phase_digest_v2(route, sp_rank)
                    or route.get("block_call_counts")
                    != {str(index): 2 for index in range(v2.TRANSFORMER_BLOCKS)}
                    or route.get("checkpoint_context_captures")
                    != v2.TRANSFORMER_BLOCKS
                    or route.get("checkpoint_forward_contexts")
                    != v2.TRANSFORMER_BLOCKS
                    or route.get("checkpoint_recompute_contexts")
                    != v2.TRANSFORMER_BLOCKS
                    or route.get("checkpoint_recompute_calls_per_block") != 1
                    or route.get("exact_block_set_0_through_29") is not True
                    or route.get("source_or_padding_written") is not False
                ):
                    fail("V2 terminal rank-specific exact30 SP route differs")
    if sorted(observed) != list(range(v2.GLOBAL_RECORDS)):
        fail("V2 terminal two-update global logical-record closure differs")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    validate_args(args)

    # The complete external teacher closure is validated on CPU before any
    # Bernini source activation, distributed initialization, or model memory.
    preflight = v2.preflight_frozen_sidecars_v2(
        Path(args.sidecar_manifest), **_sidecar_kwargs(args)
    )
    if len(v2.materialize_and_validate_teachers_v2(preflight, device="cpu")) != (
        v2.GLOBAL_RECORDS
    ):
        fail("V2 CPU teacher preflight cardinality differs")

    import packed_preservation_release_v2 as release_contract
    import train_action_edit_large_lora_0817_v1 as renderer_v1
    import train_lora as legacy

    if (
        args.expected_bernini_commit != renderer_v1.BERNINI_COMMIT
        or args.expected_veomni_commit != renderer_v1.VEOMNI_COMMIT
        or args.expected_checkpoint_tree_sha256 != renderer_v1.CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != renderer_v1.CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        fail("V2 Bernini/VeOmni/checkpoint external pins differ from 0817 runtime")
    release_closure = renderer_v1.validate_release_manifest(
        Path(args.release_manifest),
        expected_sha256=args.expected_release_manifest_sha256,
    )
    try:
        bernini_root, veomni_root, _, _ = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise ActionAnchorV2World8Error(str(error)) from error
    if (
        transformer_config.get("num_layers") != v2.TRANSFORMER_BLOCKS
        or transformer_config.get("attention_head_dim") != 128
        or transformer_config.get("num_attention_heads") != 12
    ):
        fail("V2 Bernini-R 1.3B geometry differs")
    checkpoint_manifest = Path(args.checkpoint_content_manifest).resolve(strict=True)
    if v2.file_sha256(checkpoint_manifest) != args.expected_checkpoint_content_manifest_sha256:
        fail("V2 checkpoint content manifest SHA differs")
    checkpoint_content = release_contract.validate_checkpoint_content(
        checkpoint,
        checkpoint_manifest,
        expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
    )
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    import action_anchor_distillation_v1 as distillation_module
    import action_plan_predictor_v1 as action_plan_module
    from action_plan_predictor_v1 import ActionPlanConditionerV1, ActionPlanPredictorConfig
    import clean_source_visual_context_stage_b_contract_v1 as schedule_contract
    import inference_sigma_strata as sigma_strata_module
    import packed_preservation_lora_v2 as packed_core
    import source_self_runtime as runtime

    for module, name, expected in (
        (renderer_v1, v2.RENDERER_RUNNER_SOURCE_NAME, preflight.renderer_runner_source_sha256),
        (schedule_contract, v2.SCHEDULE_SOURCE_NAME, preflight.schedule_source_sha256),
        (packed_core, v2.PACKED_CORE_SOURCE_NAME, preflight.packed_core_source_sha256),
        (runtime, v2.RUNTIME_SOURCE_NAME, preflight.runtime_source_sha256),
        (legacy, v2.LEGACY_LOADER_SOURCE_NAME, preflight.legacy_loader_source_sha256),
        (
            distillation_module,
            v2.DISTILLATION_SOURCE_NAME,
            preflight.distillation_source_sha256,
        ),
        (v2, Path(v2.__file__).name, preflight.v2_runner_source_sha256),
        (sys.modules[__name__], Path(__file__).name, preflight.world8_adapter_source_sha256),
    ):
        v2._require_exact_import_v2(module, name, expected)
    renderer_v1.validate_imported_release_modules(
        release_closure,
        {
            "action_plan_predictor_v1.py": action_plan_module,
            "clean_source_visual_context_stage_b_contract_v1.py": schedule_contract,
            "inference_sigma_strata.py": sigma_strata_module,
            "packed_preservation_lora_v2.py": packed_core,
            "packed_preservation_release_v2.py": release_contract,
            "source_self_runtime.py": runtime,
            "train_action_edit_large_lora_0817_v1.py": renderer_v1,
            "train_lora.py": legacy,
        },
    )

    distributed = runtime.distributed_contract()
    device = runtime.initialise_distributed(distributed)
    parallel = runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=v2.SP_SIZE)
    )
    global_launch = v2.object_sha256(
        {
            "preflight": preflight.receipt(),
            "release": release_closure,
            "checkpoint_content": checkpoint_content,
            "output": args.output,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "max_grad_norm": args.max_grad_norm,
        }
    )
    runtime.digest_consensus(
        global_launch,
        group=parallel.world_group,
        expected_count=v2.WORLD_SIZE,
        label="V2 WORLD8 launch envelope",
    )
    output, stage = runtime.prepare_output_transaction(
        args.output, distributed.rank, parallel.world_group
    )
    renderer_v1.seed_everything(args.seed)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    with renderer_v1.serialized_model_load():
        renderer = BerniniRendererModel(config)
        renderer.requires_grad_(False)
        renderer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "context_fn": renderer_v1.action_route_checkpoint_context_fn,
            }
        )
        specs = packed_core.select_projection_specs(renderer, LORA_SCOPE)
        model = get_peft_model(
            renderer,
            LoraConfig(
                r=packed_core.LORA_RANK,
                lora_alpha=packed_core.LORA_ALPHA,
                lora_dropout=0.0,
                bias="none",
                target_modules=[item.name for item in specs],
            ),
        )
        transformer = model.get_base_model().diff_dec.transformer
        renderer_v1.validate_action_route_checkpointing_installation(transformer)
        packed_core.install_typed_patch_embedding(transformer)
        conditioner = ActionPlanConditionerV1(ActionPlanPredictorConfig())
        hook_handle = renderer_v1.install_action_plan_hooks(transformer, conditioner)
        model.to(device)
    model.train()
    base_renderer = model.get_base_model()
    base_renderer.t5_text_encoder.eval()
    named = renderer_v1.exact_trainable_named_parameters(model, conditioner)
    packed_core.validate_lora_installation(model, specs)
    if any(parameter.dtype != torch.float32 or parameter.device != device for _, parameter in named):
        fail("V2 initialized trainable dtype/device differs")
    runtime.synchronize_initial_parameters(
        named, parallel.world_group, expected_count=v2.WORLD_SIZE
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    local_inputs = _materialize_local_inputs(
        preflight=preflight,
        distributed=distributed,
        base_renderer=base_renderer,
        tokenizer=tokenizer,
        runtime=runtime,
        renderer_v1=renderer_v1,
        device=device,
    )
    base_renderer.t5_text_encoder = None
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if base_renderer.t5_text_encoder is not None:
        fail("V2 frozen T5 was not retired")

    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    execution = v2.prepare_world8_canary_v2(
        preflight,
        model=model,
        base_renderer=base_renderer,
        transformer=transformer,
        conditioner=conditioner,
        hook_handle=hook_handle,
        parallel=parallel,
        distributed=distributed,
        rope=rope,
        device=device,
        local_inputs=local_inputs,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        loss_kwargs={
            "smooth_l1_weight": args.smooth_l1_weight,
            "cosine_weight": args.cosine_weight,
            "infonce_weight": args.infonce_weight,
            "flow_weight": args.flow_weight,
            "temperature": args.temperature,
        },
    )
    result = v2.run_exact_two_updates_v2(execution)
    _persist_terminal_result(
        output=output,
        stage=stage,
        result=result,
        model=model,
        conditioner=conditioner,
        preflight=preflight,
        args=args,
        distributed=distributed,
        parallel=parallel,
        runtime=runtime,
        renderer_v1=renderer_v1,
        release_closure=release_closure,
        checkpoint_content=checkpoint_content,
    )
    if distributed.rank == 0:
        print(
            json.dumps(
                {
                    "complete": True,
                    "output": str(output),
                    "optimizer_updates": result.optimizer_updates,
                    "formal_training_authorized": False,
                    "scientific_claim_authorized": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=parallel.world_group)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
