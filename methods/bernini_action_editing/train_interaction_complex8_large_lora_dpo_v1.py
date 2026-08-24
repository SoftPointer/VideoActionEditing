#!/usr/bin/env python3
"""Train large native-RV2V action preference with optional identity replay.

This runner consumes only source-preserving RV2V candidate preferences.  The
pure-T2V anchor is absent from every model call and endpoint.  The student is
an all-30-block rank-256 q/k/v/out LoRA (188.7M parameters); the reference is
the same native renderer with PEFT disabled.  Chosen/rejected endpoints share
fresh Gaussian noise and physical sigma in a reference-corrected flow-DPO
objective.  Optional preservation replay uses source->source under an explicit
identity instruction, never under the requested action instruction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import interaction_large_action_adapter_v1 as large_adapter
import pair_v5_flow_dpo as flow_dpo
import source_self_runtime as runtime
import train_lora as legacy
import train_pair_v5_action_preference as pair_runtime


# Reuse the already validated native RV2V pack/VJP implementation with the
# large adapter's API-compatible route and exact reference-disable context.
pair_runtime.action_adapter = large_adapter

SCHEMA_VERSION = "bernini-interaction-complex8-large-lora-dpo-run-v1"
MANIFEST_SCHEMA = "bernini-interaction-complex8-preference-manifest-v1"
PAIR_SCHEMA = "bernini-interaction-complex8-preference-row-v1"
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
FRAME_COUNT = 81
IDENTITY_CAPTION = (
    "Keep the source video exactly unchanged, including every subject, identity, "
    "clothing or fur, objects, background, camera, timing, composition and motion."
)
DEFAULT_SEED = 20260817


class Complex8TrainingError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return runtime.file_sha256(path)


@dataclass(frozen=True)
class Complex8Manifest:
    snapshot: pair_runtime.FileSnapshot
    manifest_digest: str
    critic_validation: Mapping[str, Any]
    selection_policy: Mapping[str, Any]
    input_closure: Mapping[str, Any]
    rows: tuple[pair_runtime.PreferenceRow, ...]

    def assert_unchanged(self) -> None:
        self.snapshot.assert_unchanged(label="complex8 preference manifest")
        for row in self.rows:
            row.assert_unchanged()


def read_manifest(path_value: str, expected_sha256: str) -> Complex8Manifest:
    snapshot = pair_runtime.FileSnapshot.capture(
        path_value,
        expected_sha256=expected_sha256,
        label="complex8 preference manifest",
    )
    value = json.loads(snapshot.path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != MANIFEST_SCHEMA:
        raise Complex8TrainingError("complex8 preference manifest schema differs")
    declared = value.get("manifest_digest")
    unsigned = dict(value)
    unsigned.pop("manifest_digest", None)
    if not isinstance(declared, str) or object_sha256(unsigned) != declared:
        raise Complex8TrainingError("complex8 preference manifest digest differs")
    critic = value.get("action_critic_validation")
    policy = value.get("selection_policy")
    closure = value.get("input_closure")
    raw_rows = value.get("pairs")
    if (
        value.get("complete") is not True
        or not isinstance(critic, Mapping)
        or critic.get("passed") is not True
        or float(critic.get("pairwise_accuracy", 0.0)) < 0.75
        or not isinstance(policy, Mapping)
        or policy.get("both_endpoints_must_preserve_source") is not True
        or policy.get("weighted_action_appearance_score_used") is not False
        or not isinstance(closure, Mapping)
        or closure.get("pure_t2v_anchor_used_as_training_endpoint") is not False
        or closure.get("pure_t2v_anchor_appearance_enters_candidate_selection") is not False
        or closure.get("native_rv2v_candidate_latents_are_dpo_endpoints") is not True
        or closure.get("qwen_or_vlm_used") is not False
        or not isinstance(raw_rows, list)
        or not 6 <= len(raw_rows) <= 8
    ):
        raise Complex8TrainingError("manifest optimizer authorization closure differs")
    rows: list[pair_runtime.PreferenceRow] = []
    seen_events: set[int] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or raw.get("schema_version") != PAIR_SCHEMA:
            raise Complex8TrainingError("complex8 preference row schema differs")
        row_unsigned = dict(raw)
        pair_digest = row_unsigned.pop("pair_digest", None)
        if not isinstance(pair_digest, str) or object_sha256(row_unsigned) != pair_digest:
            raise Complex8TrainingError("complex8 preference row digest differs")
        event = raw.get("event_ordinal")
        selection = raw.get("selection")
        source_binding = raw.get("source_video")
        caption = raw.get("complete_caption")
        caption_sha = raw.get("complete_caption_sha256")
        if (
            type(event) is not int
            or not 0 <= event < 8
            or event in seen_events
            or not isinstance(selection, Mapping)
            or selection.get("both_endpoints_pass_all_source_preservation_gates") is not True
            or selection.get("weighted_action_appearance_score_used") is not False
            or not float(selection.get("strict_action_margin", 0.0)) > 0.0
            or not isinstance(source_binding, Mapping)
            or source_binding.get("schema_version") != pair_runtime.FILE_BINDING_SCHEMA
            or not isinstance(caption, str)
            or hashlib.sha256(caption.encode("utf-8")).hexdigest() != caption_sha
            or raw.get("sample_weight") != 1.0
        ):
            raise Complex8TrainingError("complex8 preference row authorization differs")
        source = pair_runtime.FileSnapshot.capture(
            source_binding.get("path"),
            expected_sha256=source_binding.get("sha256"),
            label=f"complex8 event {event} source",
        )
        chosen = pair_runtime._validate_rollout_binding(
            raw.get("chosen_rollout"),
            source_sha256=source.sha256,
            caption=caption,
            caption_sha256=caption_sha,
            label=f"complex8 event {event} chosen",
        )
        rejected = pair_runtime._validate_rollout_binding(
            raw.get("rejected_rollout"),
            source_sha256=source.sha256,
            caption=caption,
            caption_sha256=caption_sha,
            label=f"complex8 event {event} rejected",
        )
        if chosen.candidate_id == rejected.candidate_id:
            raise Complex8TrainingError("chosen and rejected candidate IDs are equal")
        rows.append(
            pair_runtime.PreferenceRow(
                pair_id=str(raw.get("pair_id")),
                source_video_snapshot=source,
                complete_caption=caption,
                complete_caption_sha256=caption_sha,
                chosen=chosen,
                rejected=rejected,
                sample_weight=1.0,
                pair_digest=pair_digest,
            )
        )
        seen_events.add(event)
    snapshot.assert_unchanged(label="complex8 preference manifest after validation")
    return Complex8Manifest(snapshot, declared, dict(critic), dict(policy), dict(closure), tuple(rows))


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def identity_rows(rows: Sequence[pair_runtime.PreferenceRow]) -> tuple[pair_runtime.PreferenceRow, ...]:
    digest = hashlib.sha256(IDENTITY_CAPTION.encode("utf-8")).hexdigest()
    return tuple(
        replace(
            row,
            pair_id=f"identity-{row.pair_id}",
            complete_caption=IDENTITY_CAPTION,
            complete_caption_sha256=digest,
        )
        for row in rows
    )


def save_adapter(path: Path, handle: large_adapter.LargeActionAdapterHandle, *, step: int) -> Mapping[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    state = dict(handle.state_dict_for_save())
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "step": str(step),
        "adapter_contract_digest": str(handle.receipt()["digest"]),
        "flow_dpo_contract_digest": str(flow_dpo.contract_receipt()["digest"]),
        "pure_t2v_anchor_is_absent": "true",
    }
    save_file(state, str(path), metadata=metadata)
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if set(opened.keys()) != set(state) or dict(opened.metadata() or {}) != metadata:
            raise Complex8TrainingError("large adapter checkpoint roundtrip differs")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "tensor_count": len(state),
        "bytes": path.stat().st_size,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preference-manifest", required=True)
    parser.add_argument("--expected-preference-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, choices=(2, 4, 8, 16), default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--preservation-weight", type=float, default=0.0)
    parser.add_argument("--minimum-peak-memory-ratio", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT)
    parser.add_argument("--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("learning_rate", "beta", "max_grad_norm"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise Complex8TrainingError(f"{name} must be finite and positive")
    if (
        not math.isfinite(args.preservation_weight)
        or not 0.0 <= args.preservation_weight <= 1.0
        or not math.isfinite(args.minimum_peak_memory_ratio)
        or not 0.0 <= args.minimum_peak_memory_ratio <= 1.0
        or type(args.seed) is not int
        or not 0 <= args.seed < 2**63
    ):
        raise Complex8TrainingError("preservation/memory/seed argument differs")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    manifest = read_manifest(
        args.preference_manifest, args.expected_preference_manifest_sha256
    )
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise Complex8TrainingError(str(error)) from error
    if (
        transformer_config.get("num_layers") != 30
        or transformer_config.get("num_attention_heads") != 12
        or args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256
    ):
        raise Complex8TrainingError("Bernini checkpoint/transformer geometry differs")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    import infer_native_identity_generation_canary as native_canary
    import source_self_native_ref_contrastive_v3 as native

    distributed = runtime.distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.local_world_size != WORLD_SIZE
    ):
        raise Complex8TrainingError("trainer requires one-node WORLD8 DP2xSP4")
    device = runtime.initialise_distributed(distributed)
    parallel = runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=SP_SIZE)
    )
    seed_everything(args.seed)

    output = Path(args.output)
    status: list[Any] = [None]
    if distributed.rank == 0:
        status[0] = bool(
            output.is_absolute()
            and output != Path("/")
            and not output.exists()
            and not output.is_symlink()
        )
        if status[0]:
            output.mkdir(parents=True, mode=0o750)
            (output / "checkpoints").mkdir(mode=0o750)
    dist.broadcast_object_list(status, src=0, group=parallel.world_group)
    if status[0] is not True:
        raise Complex8TrainingError("output must be a fresh absolute non-root directory")
    dist.barrier(group=parallel.world_group)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config).requires_grad_(False).eval()

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    ).eval().requires_grad_(False)
    source_cache, source_receipts = pair_runtime._source_condition_cache(
        vae,
        manifest.rows,
        device=device,
        parallel=parallel,
        source_audit=native_canary.source_audit,
        vae_encode=_vae_encode,
    )
    del vae
    torch.cuda.empty_cache()

    renderer.to(device)
    disable_checkpointing = getattr(renderer, "gradient_checkpointing_disable", None)
    if callable(disable_checkpointing):
        disable_checkpointing()
    handle = large_adapter.install_pair_v5_action_adapter(renderer)
    model = handle.model.to(device)
    base_renderer = model.get_base_model()
    diffusion = base_renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise Complex8TrainingError("trainer requires frozen Bernini transformer_1 base")
    trainable = handle.trainable_named_parameters()
    if not handle.base_parameters_frozen() or any(
        parameter.device != device or parameter.dtype != torch.float32
        for _, parameter in trainable
    ):
        raise Complex8TrainingError("large adapter trainability closure differs")
    initial_digest = runtime.synchronize_initial_parameters(trainable, parallel.world_group)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    all_text_rows = tuple(manifest.rows) + identity_rows(manifest.rows)
    text_cache, text_receipts = pair_runtime._frozen_text_embedding_cache(
        base_renderer,
        tokenizer,
        all_text_rows,
        device=device,
        parallel=parallel,
        build_task_prompt=native_canary.build_task_prompt,
        prompt_cleaner=prompt_clean,
    )
    base_renderer.t5_text_encoder.to("cpu")
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    endpoints = {
        row.pair_id: (
            pair_runtime._load_clean_latent(row.chosen),
            pair_runtime._load_clean_latent(row.rejected),
        )
        for row in manifest.rows
    }
    for row in manifest.rows:
        chosen, rejected = endpoints[row.pair_id]
        source, references = source_cache[row.pair_id]
        if (
            chosen.shape != rejected.shape
            or chosen.shape != source.shape
            or torch.equal(chosen, rejected)
            or any(reference.shape[3:] != source.shape[3:] for reference in references)
        ):
            raise Complex8TrainingError("preference endpoint/source geometry differs")

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    checkpoint_steps = sorted({1, 2, args.max_steps // 2, args.max_steps})
    checkpoint_records: list[Mapping[str, Any]] = []
    history: list[Mapping[str, Any]] = []
    memory_world_history: list[Any] = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    for global_step in range(args.max_steps):
        row_index = (global_step * DP_SIZE + distributed.arm_index) % len(manifest.rows)
        row = manifest.rows[row_index]
        chosen_cpu, rejected_cpu = endpoints[row.pair_id]
        source_cpu, refs_cpu = source_cache[row.pair_id]
        cond_cpu, uncond_cpu = text_cache[row.pair_id]
        identity_cond_cpu, identity_uncond_cpu = text_cache[f"identity-{row.pair_id}"]
        chosen = chosen_cpu.to(device).contiguous().detach()
        rejected = rejected_cpu.to(device).contiguous().detach()
        source = source_cpu.to(device).contiguous().detach()
        refs = tuple(value.to(device).contiguous().detach() for value in refs_cpu)
        conditional = cond_cpu.to(device).contiguous().detach()
        unconditional = uncond_cpu.to(device).contiguous().detach()
        identity_conditional = identity_cond_cpu.to(device).contiguous().detach()
        identity_unconditional = identity_uncond_cpu.to(device).contiguous().detach()
        for tensor in (
            chosen,
            rejected,
            source,
            *refs,
            conditional,
            unconditional,
            identity_conditional,
            identity_unconditional,
        ):
            pair_runtime._broadcast_within_sp(tensor, parallel=parallel)

        sigma_index = pair_runtime.registered_action_sigma_index(
            seed=args.seed,
            step=global_step,
            pair_digest=row.pair_digest,
            dp_rank=distributed.arm_index,
        )
        sigma = torch.tensor(
            [native.NATIVE_UNIPC40_SIGMAS[sigma_index]], dtype=torch.float32, device=device
        ).detach()
        timestep = torch.tensor(
            [native.NATIVE_UNIPC40_TIMESTEPS[sigma_index]], dtype=torch.float32, device=device
        ).detach()
        noise_seed = pair_runtime._fresh_noise_seed(
            args.seed, global_step, row.pair_digest, distributed.arm_index
        )
        epsilon = pair_runtime.fresh_shared_epsilon(chosen.shape, seed=noise_seed, device=device)
        pair_runtime._broadcast_within_sp(epsilon, parallel=parallel)
        sigma_view = sigma.reshape(1, 1, 1, 1, 1)
        chosen_x = ((1.0 - sigma_view) * chosen + sigma_view * epsilon).detach()
        rejected_x = ((1.0 - sigma_view) * rejected + sigma_view * epsilon).detach()

        student: dict[str, Any] = {}
        reference: dict[str, Any] = {}
        for name, state in (("chosen", chosen_x), ("rejected", rejected_x)):
            pack = pair_runtime._build_pack(transformer, source, refs, state)
            student[name] = pair_runtime._guided_prediction_no_grad(
                diffusion,
                pack,
                timestep=timestep,
                cond_embeds=conditional,
                uncond_embeds=unconditional,
                action_handle=handle,
                cio_handle=None,
                sp_rank=distributed.sp_rank,
                sigma_index=sigma_index,
                action_enabled=True,
                video_shape=state.shape,
            )
            reference[name] = pair_runtime._guided_prediction_no_grad(
                diffusion,
                pack,
                timestep=timestep,
                cond_embeds=conditional,
                uncond_embeds=unconditional,
                action_handle=handle,
                cio_handle=None,
                sp_rank=distributed.sp_rank,
                sigma_index=sigma_index,
                action_enabled=False,
                video_shape=state.shape,
            )
            del pack

        chosen_leaf = student["chosen"].detach().clone().requires_grad_(True)
        rejected_leaf = student["rejected"].detach().clone().requires_grad_(True)
        student_chosen = chosen_leaf + torch.zeros(
            (), dtype=chosen_leaf.dtype, device=device
        )
        student_rejected = rejected_leaf + torch.zeros(
            (), dtype=rejected_leaf.dtype, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        result = flow_dpo.reference_corrected_flow_dpo(
            chosen,
            rejected,
            epsilon,
            sigma,
            student_chosen,
            student_rejected,
            reference["chosen"],
            reference["rejected"],
            beta=args.beta,
            sample_weight=torch.ones(1, dtype=torch.float32, device=device),
        )
        if not runtime.world_all_true(
            bool(torch.isfinite(result.loss.detach()).item()), group=parallel.world_group
        ):
            raise Complex8TrainingError("non-finite reference-corrected flow-DPO loss")
        result.loss.backward()
        replay_max = 0.0
        for state, cotangent, expected in (
            (chosen_x, chosen_leaf.grad.detach(), student["chosen"]),
            (rejected_x, rejected_leaf.grad.detach(), student["rejected"]),
        ):
            replay_max = max(
                replay_max,
                pair_runtime._replay_prediction_vjp(
                    diffusion,
                    transformer,
                    source_video=source,
                    references=refs,
                    x_sigma=state,
                    timestep=timestep,
                    cond_embeds=conditional,
                    uncond_embeds=unconditional,
                    action_handle=handle,
                    cio_handle=None,
                    sp_rank=distributed.sp_rank,
                    sigma_index=sigma_index,
                    output_cotangent=cotangent,
                    expected_guided=expected,
                ),
            )

        preservation_loss_value = 0.0
        if args.preservation_weight > 0.0:
            identity_x = ((1.0 - sigma_view) * source + sigma_view * epsilon).detach()
            identity_pack = pair_runtime._build_pack(transformer, source, refs, identity_x)
            identity_student = pair_runtime._guided_prediction_no_grad(
                diffusion,
                identity_pack,
                timestep=timestep,
                cond_embeds=identity_conditional,
                uncond_embeds=identity_unconditional,
                action_handle=handle,
                cio_handle=None,
                sp_rank=distributed.sp_rank,
                sigma_index=sigma_index,
                action_enabled=True,
                video_shape=identity_x.shape,
            )
            identity_leaf = identity_student.detach().requires_grad_(True)
            identity_target = (epsilon - source).detach()
            preservation_loss = (
                torch.nn.functional.mse_loss(identity_leaf.float(), identity_target.float())
                * args.preservation_weight
            )
            preservation_loss.backward()
            preservation_loss_value = float(preservation_loss.detach().item())
            replay_max = max(
                replay_max,
                pair_runtime._replay_prediction_vjp(
                    diffusion,
                    transformer,
                    source_video=source,
                    references=refs,
                    x_sigma=identity_x,
                    timestep=timestep,
                    cond_embeds=identity_conditional,
                    uncond_embeds=identity_unconditional,
                    action_handle=handle,
                    cio_handle=None,
                    sp_rank=distributed.sp_rank,
                    sigma_index=sigma_index,
                    output_cotangent=identity_leaf.grad.detach(),
                    expected_guided=identity_student,
                ),
            )
            del identity_x, identity_pack, identity_student, identity_leaf, identity_target, preservation_loss

        preclip_norm = runtime.synchronize_gradients(trainable, parallel)
        clipped = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in trainable], args.max_grad_norm
        )
        if not math.isfinite(float(clipped)):
            raise Complex8TrainingError("large-adapter gradient norm is non-finite")
        optimizer.step()
        torch.cuda.synchronize(device)
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        memory = {
            "world_rank": distributed.rank,
            "step": global_step + 1,
            "allocated": int(torch.cuda.memory_allocated(device)),
            "reserved": int(torch.cuda.memory_reserved(device)),
            "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
            "total": total_memory,
            "peak_allocated_ratio": float(torch.cuda.max_memory_allocated(device) / total_memory),
            "peak_reserved_ratio": float(torch.cuda.max_memory_reserved(device) / total_memory),
        }
        gathered_memory: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_memory, memory, group=parallel.world_group)
        memory_world_history.append(gathered_memory)
        if global_step == 0 and min(
            row_memory["peak_allocated_ratio"] for row_memory in gathered_memory
        ) < args.minimum_peak_memory_ratio:
            raise Complex8TrainingError(
                "first update used less than the required physical-memory fraction; "
                f"minimum={min(row_memory['peak_allocated_ratio'] for row_memory in gathered_memory):.4f}, "
                f"required={args.minimum_peak_memory_ratio:.4f}"
            )

        local = {
            "step": global_step + 1,
            "world_rank": distributed.rank,
            "dp_rank": distributed.arm_index,
            "sp_rank": distributed.sp_rank,
            "pair_id": row.pair_id,
            "chosen_candidate_id": row.chosen.candidate_id,
            "rejected_candidate_id": row.rejected.candidate_id,
            "sigma_schedule_index": sigma_index,
            "noise_seed": noise_seed,
            "dpo_loss": float(result.loss.detach().item()),
            "advantage": float(result.advantage.detach().item()),
            "student_gap": float(result.student_gap.detach().item()),
            "reference_gap": float(result.reference_gap.detach().item()),
            "preservation_loss_weighted": preservation_loss_value,
            "preclip_gradient_norm_world_average": preclip_norm,
            "vjp_replay_max_abs": replay_max,
            "peak_allocated_ratio": memory["peak_allocated_ratio"],
        }
        gathered: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local, group=parallel.world_group)
        history.append({"step": global_step + 1, "dp_records": [gathered[0], gathered[4]]})
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "step": global_step + 1,
                        "loss_dp0": gathered[0]["dpo_loss"],
                        "loss_dp1": gathered[4]["dpo_loss"],
                        "preservation_weight": args.preservation_weight,
                        "preclip_norm": preclip_norm,
                        "min_peak_allocated_ratio": min(
                            row_memory["peak_allocated_ratio"] for row_memory in gathered_memory
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if global_step + 1 in checkpoint_steps:
            digest = runtime.parameter_consensus(
                trainable,
                parallel.world_group,
                f"complex8 large adapter step {global_step + 1}",
            )
            if distributed.rank == 0:
                record = save_adapter(
                    output / "checkpoints" / f"step_{global_step + 1:04d}.safetensors",
                    handle,
                    step=global_step + 1,
                )
                checkpoint_records.append({**record, "parameter_digest": digest})
        dist.barrier(group=parallel.world_group)
        del (
            chosen,
            rejected,
            source,
            refs,
            conditional,
            unconditional,
            identity_conditional,
            identity_unconditional,
            epsilon,
            chosen_x,
            rejected_x,
            student,
            reference,
            chosen_leaf,
            rejected_leaf,
            student_chosen,
            student_rejected,
            result,
        )
        gc.collect()
        torch.cuda.empty_cache()

    final_digest = runtime.parameter_consensus(
        trainable, parallel.world_group, "complex8 final large adapter"
    )
    if final_digest == initial_digest:
        raise Complex8TrainingError("optimizer did not change the large adapter")
    manifest.assert_unchanged()
    if distributed.rank == 0:
        receipt_unsigned = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "optimizer_steps": args.max_steps,
            "elapsed_seconds": time.monotonic() - started,
            "manifest": {
                "path": str(manifest.snapshot.path),
                "sha256": manifest.snapshot.sha256,
                "manifest_digest": manifest.manifest_digest,
                "pair_count": len(manifest.rows),
                "critic_validation": dict(manifest.critic_validation),
                "selection_policy": dict(manifest.selection_policy),
                "input_closure": dict(manifest.input_closure),
            },
            "objective": {
                "name": "reference_corrected_shared_noise_flow_dpo",
                "beta": args.beta,
                "preservation_replay": (
                    "source_to_source_explicit_identity_instruction"
                    if args.preservation_weight > 0.0
                    else "disabled"
                ),
                "preservation_weight": args.preservation_weight,
                "pure_t2v_anchor_in_model_call": False,
                "full_anchor_latent_or_pixel_regression": False,
                "weighted_action_appearance_reward": False,
            },
            "adapter": {
                **dict(handle.receipt()),
                "initial_parameter_digest": initial_digest,
                "final_parameter_digest": final_digest,
                "changed_by_optimizer": True,
            },
            "optimizer": {
                "type": "AdamW",
                "learning_rate": args.learning_rate,
                "betas": [0.9, 0.95],
                "weight_decay": 0.0,
                "max_gradient_norm": args.max_grad_norm,
            },
            "distributed": {
                "world_size": WORLD_SIZE,
                "data_parallel_size": DP_SIZE,
                "sequence_parallel_size": SP_SIZE,
                "all_eight_gpus_used": True,
            },
            "memory": {
                "minimum_required_peak_allocated_ratio": args.minimum_peak_memory_ratio,
                "per_step_world": memory_world_history,
                "requirement_passed": True,
            },
            "history": history,
            "checkpoints": checkpoint_records,
            "source_conditions": source_receipts,
            "text_conditions": text_receipts,
            "model": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
            },
            "seed": args.seed,
        }
        receipt = {**receipt_unsigned, "receipt_digest": object_sha256(receipt_unsigned)}
        (output / "receipt.json").write_bytes(canonical_bytes(receipt) + b"\n")
        (output / "COMPLETE").touch()
    dist.barrier(group=parallel.world_group)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
