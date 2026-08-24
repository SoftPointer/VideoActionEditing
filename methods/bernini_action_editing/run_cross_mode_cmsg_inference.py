#!/usr/bin/env python3
"""End-to-end AUH runner for Bernini Cross-Mode CMSG LoRA v6.

This is the executable deployment half of the v6 experiment.  It deliberately
reuses the audited Bernini v5 source-only entry path (exact 81-frame source
preparation, official MV2V prompt rendering, semantic no-op encoding, frozen
negative APG, 40-step shift-5 UniPC, four-rank Ulysses, VAE decode, and atomic
receipt publication), while replacing the v5 adapter contract and scheduler
operator with the v6 Cross-Mode CMSG implementations.

The target-only T2V branch is a training teacher and is never constructed by
this runner.  The complete external edit condition is therefore exactly one
source video plus one action instruction.  No target, generator prompt, mask,
track, flow, pose, trajectory, or first-frame anchor argument exists.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_cross_mode_cmsg_lora as cmsg  # noqa: E402
import infer_prior_tangent_lora as v5  # noqa: E402


frozen = v5.frozen
trainer = v5.trainer
tri = v5.tri
sigma_strata = v5.sigma_strata

METHOD_NAME = cmsg.METHOD_NAME
INFERENCE_RECEIPT_SCHEMA = cmsg.INFERENCE_RECEIPT_SCHEMA
NUM_FRAMES = cmsg.NUM_FRAMES
LATENT_PHASES = cmsg.LATENT_PHASES
NUM_INFERENCE_STEPS = cmsg.NUM_DENOISING_STEPS
ULYSSES_SIZE = frozen.base.ULYSSES_SIZE


class CrossModeCMSGRunnerError(RuntimeError):
    """Raised before an invalid v6 run can publish an output or receipt."""


def _remove_inherited_option(
    parser: argparse.ArgumentParser, *, destination: str, default: Any
) -> None:
    """Hide a fixed v5-only control while retaining the audited parser.

    ``argparse`` has no public remove API.  We keep this tiny operation local
    and assert its postcondition so a future parser change fails during module
    tests instead of silently exposing an irrelevant v5 execution knob.
    """

    matches = [action for action in parser._actions if action.dest == destination]
    if len(matches) != 1:
        raise CrossModeCMSGRunnerError(
            f"inherited parser lacks exactly one {destination!r} option"
        )
    action = matches[0]
    parser._remove_action(action)
    for group in parser._action_groups:
        group_actions = getattr(group, "_group_actions", None)
        if group_actions is not None and action in group_actions:
            group_actions.remove(action)
    for option in action.option_strings:
        parser._option_string_actions.pop(option, None)
    parser.set_defaults(**{destination: default})
    if any(action.dest == destination for action in parser._actions):
        raise CrossModeCMSGRunnerError(
            f"failed to fix inherited parser option {destination!r}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Reuse the v5 parser while exposing only meaningful v6 controls."""

    parser = v5.build_parser()
    parser.description = (
        "Run Bernini-R 1.3B Cross-Mode CMSG v6 on one exact 81-frame source"
    )
    # These v5 routing/ablation controls do not exist in the v6 deployment
    # operator.  Fixing them internally also prevents launcher drift.
    for destination, default in (
        ("execution_arm", "main"),
        ("alpha", v5.ADAPTER_SCALE),
        ("max_generate_fraction", frozen.DEFAULT_GENERATE_CAP),
        ("energy_coverage", frozen.DEFAULT_ENERGY_COVERAGE),
    ):
        _remove_inherited_option(
            parser, destination=destination, default=default
        )
    adapter_actions = [
        action for action in parser._actions if action.dest == "adapter_checkpoint"
    ]
    if len(adapter_actions) != 1:
        raise CrossModeCMSGRunnerError(
            "inherited parser lacks exactly one adapter checkpoint option"
        )
    adapter_actions[0].required = True
    adapter_actions[0].help = (
        "completed Cross-Mode CMSG v6 checkpoint root (or its adapter/ directory)"
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    """Validate both the inherited source-only contract and v6 restrictions."""

    try:
        v5.validate_cli(args)
    except v5.PriorTangentInferenceError as error:
        raise CrossModeCMSGRunnerError(str(error)) from error
    if args.execution_arm != "main":
        raise CrossModeCMSGRunnerError("v6 runner supports only the main execution arm")
    if not isinstance(args.adapter_checkpoint, str) or not args.adapter_checkpoint:
        raise CrossModeCMSGRunnerError("v6 inference requires --adapter-checkpoint")
    if int(args.num_inference_steps) != NUM_INFERENCE_STEPS:
        raise CrossModeCMSGRunnerError("v6 requires exactly 40 official UniPC steps")


def launcher_contract() -> dict[str, Any]:
    """Machine-readable contract for the four-GPU AUH launcher."""

    return {
        "launcher": "torchrun",
        "nproc_per_node": ULYSSES_SIZE,
        "world_size": ULYSSES_SIZE,
        "ulysses_size": ULYSSES_SIZE,
        "entrypoint": Path(__file__).name,
        "frames": NUM_FRAMES,
        "latent_phases": LATENT_PHASES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "required_external_conditions": ["source_video", "action_instruction"],
        "required_model_inputs": [
            "bernini_root",
            "veomni_root",
            "checkpoint",
            "adapter_checkpoint",
        ],
        "required_output": "output",
        "generator_loaded": False,
        "target_argument": False,
        "mask_flow_pose_track_anchor_arguments": False,
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CrossModeCMSGRunnerError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CrossModeCMSGRunnerError(f"{label} must contain one JSON object")
    return value


def _method_hashes() -> dict[str, str]:
    paths = {
        "run_cross_mode_cmsg_inference.py": Path(__file__).resolve(),
        "infer_cross_mode_cmsg_lora.py": METHOD_ROOT
        / "infer_cross_mode_cmsg_lora.py",
        "train_cross_mode_cmsg_auh.py": METHOD_ROOT / "train_cross_mode_cmsg_auh.py",
        "train_cross_mode_cmsg_lora.py": METHOD_ROOT
        / "train_cross_mode_cmsg_lora.py",
        "cross_mode_motion_spectrum.py": METHOD_ROOT
        / "cross_mode_motion_spectrum.py",
        "infer_prior_tangent_lora.py": METHOD_ROOT / "infer_prior_tangent_lora.py",
        "tri_branch_unipc.py": METHOD_ROOT / "tri_branch_unipc.py",
        "inference_sigma_strata.py": METHOD_ROOT / "inference_sigma_strata.py",
        "infer_delta_lora.py": METHOD_ROOT / "infer_delta_lora.py",
    }
    return {name: frozen.base.file_sha256(path) for name, path in paths.items()}


def build_inference_receipt(
    *,
    args: argparse.Namespace,
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    output_path: Path,
    output_sha256: str,
    noop_identity: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    inference_file_hashes: Mapping[str, str],
    wan_diffusion_path: Path,
    wan_diffusion_sha256: str,
    runtime_versions: Mapping[str, str],
    adapter_bundle: Any,
    adapter_identity: Mapping[str, Any],
    adapter_config_sha256: str,
    adapter_model_sha256: str,
    training_receipt_file_sha256: str,
    adapter_tensor_count: int,
    active_lora_module_count: int,
) -> dict[str, Any]:
    """Build a v6 receipt on the audited v5/frozen publication path."""

    audited_schedule = cmsg.validate_runtime_schedule_audit(
        execution_trace.get("runtime_unipc_schedule_audit", {})
    )
    receipt = frozen.build_inference_receipt(
        args=args,
        source_path=source_path,
        source_sha256=source_sha256,
        source_metadata=source_metadata,
        output_path=output_path,
        output_sha256=output_sha256,
        noop_identity=noop_identity,
        execution_trace=execution_trace,
        bernini_revision=bernini_revision,
        veomni_revision=veomni_revision,
        inference_file_hashes=inference_file_hashes,
        wan_diffusion_path=wan_diffusion_path,
        wan_diffusion_sha256=wan_diffusion_sha256,
        runtime_versions=runtime_versions,
    )
    receipt.pop("receipt_digest", None)
    receipt["schema_version"] = INFERENCE_RECEIPT_SCHEMA
    receipt["method"] = METHOD_NAME
    receipt["method_files_sha256"] = _method_hashes()
    receipt["launcher_contract"] = launcher_contract()
    receipt["base_model"].update(
        {
            "frozen": True,
            "base_weights_frozen": True,
            "lora_or_peft_loaded": True,
            "adapter_loaded": True,
            "all_runtime_parameters_require_grad_false": True,
        }
    )
    receipt["input"].update(
        {
            "accepted_external_conditions": [
                "source_video",
                "action_instruction",
            ],
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "generator_prompt_argument": False,
            "external_mask_track_pose_flow_trajectory": False,
            "first_frame_anchor": False,
        }
    )
    receipt["adapter"] = {
        "loaded": True,
        "checkpoint_root": str(adapter_bundle.checkpoint_root),
        "adapter_config_path": str(adapter_bundle.adapter_config_path),
        "adapter_config_sha256": adapter_config_sha256,
        "adapter_model_path": str(adapter_bundle.adapter_model_path),
        "adapter_model_sha256": adapter_model_sha256,
        "training_receipt_path": str(adapter_bundle.training_receipt_path),
        "training_receipt_file_sha256": training_receipt_file_sha256,
        "training_receipt_digest": adapter_identity["receipt_digest"],
        "training_global_step": adapter_identity["global_step"],
        "training_method_source_revision": adapter_identity[
            "training_method_source_revision"
        ],
        "training_method_source_archive_sha256": adapter_identity[
            "training_method_source_archive_sha256"
        ],
        "scope": adapter_identity["scope"],
        "target_module_count": len(adapter_identity["targets"]),
        "target_modules_sha256": adapter_identity["target_modules_sha256"],
        "serialized_target_modules": list(
            adapter_identity["serialized_target_modules"]
        ),
        "initialization_digest": adapter_identity["initialization_digest"],
        "checkpoint_parameter_digest": adapter_identity[
            "checkpoint_parameter_digest"
        ],
        "tensor_count": int(adapter_tensor_count),
        "active_lora_module_count": int(active_lora_module_count),
        "strict_tensor_reload_equal": True,
        "parameter_digest_verified_after_safetensors_reload": True,
        "target_modules_rebound_from_receipt": True,
        "merged": False,
        "scale": cmsg.ADAPTER_SCALE,
    }
    receipt["training_inference_alignment"] = {
        "training_receipt_schema": cmsg.TRAINING_RECEIPT_SCHEMA,
        "training_method": METHOD_NAME,
        "training_teacher": "target_only_frozen_t2v",
        "training_teacher_loaded_at_inference": False,
        "inference_generator_forwards": 0,
        "four_same_state_editor_branches": True,
        "frozen_negative_noop_action_adapter_disabled": True,
        "adapted_action_adapter_enabled_unmerged": True,
        "all_inference_branch_forwards_no_grad": True,
        "apg_momentum": 0.0,
        "packed_to_phase_shape": "[B,N,D]->[B,21,S,D]",
        "shared_training_operator": (
            "execute_distilled_editor(B0,Btheta,step_index)"
        ),
        "frozen_direction_formula": "B0=Q0(frozen_action-frozen_noop)",
        "adapted_direction_formula": "Btheta=Q0(adapted_action-frozen_noop)",
        "scheduler_clean_formula": (
            "frozen_action_clean+(executed_direction-B0)"
        ),
        "release_schedule": list(cmsg.spectrum.release_rho_schedule()),
        "release_schedule_sha256": cmsg.trainer.object_sha256(
            list(cmsg.spectrum.release_rho_schedule())
        ),
        "zero_release_exact_official_model_output_steps": list(
            cmsg.LATE_EXACT_STEPS
        ),
        "formal_adapter_off_steps": list(cmsg.FORMAL_ADAPTER_OFF_STEPS),
        "runtime_sigma_schedule_sha256": audited_schedule["schedule_sha256"],
        "training_sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "mask_flow_pose_train_test_gap": False,
        "first_frame_anchor": False,
    }
    receipt["sampling"].pop("router_config", None)
    receipt["sampling"].pop("routing_contract", None)
    receipt["sampling"].pop("alpha", None)
    receipt["sampling"].update(
        {
            "adapter_loaded": True,
            "adapter_scale": cmsg.ADAPTER_SCALE,
            "adapter_merged": False,
            "cross_mode_cmsg_contract": cmsg.runtime_contract(),
            "transformer_forwards_per_step": 4,
            "generator_forwards_per_step": 0,
            "legacy_binary_router": False,
            "runtime_unipc_schedule_audit": audited_schedule,
        }
    )
    receipt["experimental_inference"] = True
    receipt["production_claim_forbidden"] = True
    receipt["scientific_claim_authorized"] = False
    receipt["receipt_digest"] = trainer.object_sha256(receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    frozen.configure_rank_local_caches()

    requested_source = Path(args.source_video).expanduser()
    if not requested_source.is_absolute():
        raise CrossModeCMSGRunnerError("source video must be absolute")
    try:
        source_path = frozen.base._plain_file(
            requested_source.resolve(strict=True), label="source video"
        )
        output_path, receipt_path = frozen.base._resolve_output(args.output)
        bundle = frozen.base.resolve_adapter_bundle(args.adapter_checkpoint)
    except frozen.base.InferenceContractError as error:
        raise CrossModeCMSGRunnerError(str(error)) from error

    adapter_config = _read_json(bundle.adapter_config_path, label="adapter config")
    training_receipt = _read_json(
        bundle.training_receipt_path, label="training receipt"
    )
    try:
        identity = cmsg.validate_training_adapter_contract(
            adapter_config,
            training_receipt,
            expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        )
    except cmsg.CrossModeCMSGInferenceError as error:
        raise CrossModeCMSGRunnerError(str(error)) from error
    if (
        args.method_source_revision != identity["training_method_source_revision"]
        or args.method_source_archive_sha256
        != identity["training_method_source_archive_sha256"]
    ):
        raise CrossModeCMSGRunnerError(
            "inference source archive must exactly match training archive"
        )
    adapter_config_sha256 = frozen.base.file_sha256(bundle.adapter_config_path)
    adapter_model_sha256 = frozen.base.file_sha256(bundle.adapter_model_path)
    training_receipt_file_sha256 = frozen.base.file_sha256(
        bundle.training_receipt_path
    )

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = trainer.validate_checkpoint(args.checkpoint)
        inference_file_hashes = frozen.base.validate_inference_source_files(
            bernini_root
        )
    except (frozen.base.InferenceContractError, trainer.TrainingContractError) as error:
        raise CrossModeCMSGRunnerError(str(error)) from error
    if transformer_config["num_attention_heads"] % ULYSSES_SIZE:
        raise CrossModeCMSGRunnerError(
            "attention heads are not divisible by four-rank Ulysses"
        )
    wan_diffusion_path = (
        bernini_root / "bernini/models/wan_diffusion.py"
    ).resolve(strict=True)
    try:
        wan_diffusion_sha256 = tri.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
        )
    except tri.TriBranchHookError as error:
        raise CrossModeCMSGRunnerError(str(error)) from error
    trainer.activate_source_trees(bernini_root, veomni_root)

    import peft
    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if transformers_version != identity["transformers_version"]:
        raise CrossModeCMSGRunnerError("Transformers version differs from training")
    if SYSTEM_PROMPTS.get("mv2v") != frozen.base.MV2V_SYSTEM_PROMPT:
        raise CrossModeCMSGRunnerError("runtime MV2V system prompt differs")
    if DEFAULT_NEG_PROMPT != frozen.base.DEFAULT_NEGATIVE_PROMPT:
        raise CrossModeCMSGRunnerError("runtime negative prompt differs")
    try:
        distributed = frozen.base.inference_distributed_contract()
    except frozen.base.InferenceContractError as error:
        raise CrossModeCMSGRunnerError(str(error)) from error
    if (
        distributed.world_size != ULYSSES_SIZE
        or distributed.ulysses_size != ULYSSES_SIZE
    ):
        raise CrossModeCMSGRunnerError("v6 requires exactly four Ulysses ranks")
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise CrossModeCMSGRunnerError("v6 requires four AUH ROCm-visible GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    try:
        source_tensor, source_metadata = frozen.base.prepare_exact_source(source_path)
    except frozen.base.InferenceContractError as error:
        raise CrossModeCMSGRunnerError(str(error)) from error
    source_sha256 = frozen.base.file_sha256(source_path)
    action_prompt = frozen.base.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = frozen.base.build_training_prompt(
        v5.motion.DEFAULT_NOOP_INSTRUCTION,
        prompt_cleaner=prompt_clean,
    )
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **frozen.base.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except trainer.TrainingContractError as error:
        raise CrossModeCMSGRunnerError(str(error)) from error
    if float(config.shift) != frozen.base.FLOW_SHIFT or config.use_unipc is not True:
        raise CrossModeCMSGRunnerError("renderer must use official shift-5 UniPC")
    base_model = BerniniRendererModel(config)
    if any("lora_" in name.lower() for name, _ in base_model.named_modules()):
        raise CrossModeCMSGRunnerError("base renderer unexpectedly contains LoRA")
    base_model.requires_grad_(False)
    base_model.eval()
    try:
        model, adapter_tensor_count, active_lora_module_count, loaded_identity = (
            cmsg.strict_load_adapter(
                base_model=base_model,
                bundle=bundle,
                adapter_config=adapter_config,
                receipt=training_receipt,
                expected_checkpoint_tree_sha256=(
                    args.expected_checkpoint_tree_sha256
                ),
            )
        )
    except cmsg.CrossModeCMSGInferenceError as error:
        raise CrossModeCMSGRunnerError(str(error)) from error
    if loaded_identity != identity:
        raise CrossModeCMSGRunnerError("validated/reloaded adapter identities differ")
    renderer = model.get_base_model()
    renderer.requires_grad_(False)
    renderer.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        **frozen.base.tokenizer_load_kwargs(),
    )
    if (
        tokenizer.padding_side != "right"
        or tokenizer.init_kwargs.get("fix_mistral_regex") is not True
    ):
        raise CrossModeCMSGRunnerError("tokenizer contract differs")
    action_ids, action_mask = frozen.base._tokenize_training_prompt(
        tokenizer, action_prompt
    )
    noop_ids, noop_mask = frozen.base._tokenize_training_prompt(
        tokenizer, noop_prompt
    )
    negative_ids, negative_mask = frozen.base._tokenize_renderer_negative(
        tokenizer, frozen.base.DEFAULT_NEGATIVE_PROMPT
    )

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    with torch.no_grad():
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        )
    bucket = source_metadata["source_derived_bucket_hw"]
    expected_latent_shape = (
        1,
        int(vae.config.z_dim),
        LATENT_PHASES,
        int(bucket[0]) // 8,
        int(bucket[1]) // 8,
    )
    if tuple(int(value) for value in source_latent.shape) != expected_latent_shape:
        raise CrossModeCMSGRunnerError("source latent differs from exact 81f geometry")
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    noop_embeddings, noop_identity = frozen.encode_semantic_noop_prompt(
        renderer, noop_ids, noop_mask, device=device
    )
    sampling = frozen.exact_sampler_contract(seed=args.seed)
    if (
        sampling.get("momentum") != 0.0
        or sampling.get("num_frames") != NUM_FRAMES
        or sampling.get("num_inference_steps") != NUM_INFERENCE_STEPS
    ):
        raise CrossModeCMSGRunnerError("official sampler frame/step/APG contract differs")
    try:
        diffusion = tri.resolve_diffusion_core(renderer)
        pre_schedule = sigma_strata.audit_runtime_unipc_schedule(
            diffusion.scheduler, initialize=True
        )
        with cmsg.cross_mode_cmsg_unipc_hook(
            renderer,
            adapter_model=model,
            source_clean=source_latent,
            noop_prompt_embeds=noop_embeddings,
            latent_shape=expected_latent_shape,
            bernini_commit=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
            expected_steps=NUM_INFERENCE_STEPS,
            expected_flow_shift=frozen.base.FLOW_SHIFT,
        ) as trace:
            with torch.no_grad():
                generated_latent = renderer.sample(
                    input_ids=action_ids.to(device),
                    attention_mask=action_mask.to(device),
                    uncond_input_ids=negative_ids.to(device),
                    uncond_attention_mask=negative_mask.to(device),
                    image_vae_latents=None,
                    multi_video_vae_latents=[source_latent],
                    multi_image_vae_latents=None,
                    width=int(bucket[1]),
                    height=int(bucket[0]),
                    device=device,
                    **sampling,
                )
        post_schedule = sigma_strata.audit_runtime_unipc_schedule(
            diffusion.scheduler, initialize=False
        )
    except (
        cmsg.CrossModeCMSGInferenceError,
        v5.PriorTangentInferenceError,
        tri.TriBranchHookError,
        sigma_strata.InferenceSigmaStrataError,
        cmsg.spectrum.CrossModeMotionSpectrumError,
        cmsg.v6_train.CrossModeCMSGTrainingError,
    ) as error:
        raise CrossModeCMSGRunnerError(str(error)) from error
    if pre_schedule != post_schedule:
        raise CrossModeCMSGRunnerError("official sample changed pinned UniPC schedule")
    try:
        execution_trace = cmsg.validate_execution_trace(
            trace, runtime_schedule_audit=post_schedule
        )
    except cmsg.CrossModeCMSGInferenceError as error:
        raise CrossModeCMSGRunnerError(str(error)) from error
    if tuple(int(value) for value in generated_latent.shape) != expected_latent_shape:
        raise CrossModeCMSGRunnerError("generated latent differs from 81f geometry")
    model.to("cpu")
    del noop_embeddings, source_latent
    torch.cuda.empty_cache()

    if distributed.rank == 0:
        vae.to(device)
        with torch.no_grad():
            output = _vae_decode(vae, generated_latent)
        vae.to("cpu")
        expected_output_shape = (NUM_FRAMES, int(bucket[0]), int(bucket[1]), 3)
        if tuple(int(value) for value in output.shape) != expected_output_shape:
            raise CrossModeCMSGRunnerError("decoded output differs from 81f geometry")
        temporary_output = output_path.with_name(
            f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
        )
        if temporary_output.exists() or temporary_output.is_symlink():
            raise CrossModeCMSGRunnerError(
                f"stale temporary output: {temporary_output}"
            )
        save_output(output, str(temporary_output), fps=int(frozen.base.FPS))
        os.replace(temporary_output, output_path)
        from tools import materialize_vae

        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(
            output_path
        )
        try:
            frozen.base.validate_exact_video_metadata(
                int(encoded.shape[0]), encoded_fps
            )
        except frozen.base.InferenceContractError as error:
            raise CrossModeCMSGRunnerError(str(error)) from error
        if tuple(encoded_hw) != tuple(bucket):
            raise CrossModeCMSGRunnerError("encoded output geometry differs")
        receipt = build_inference_receipt(
            args=args,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata=source_metadata,
            output_path=output_path,
            output_sha256=frozen.base.file_sha256(output_path),
            noop_identity=noop_identity,
            execution_trace=execution_trace,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            inference_file_hashes=inference_file_hashes,
            wan_diffusion_path=wan_diffusion_path,
            wan_diffusion_sha256=wan_diffusion_sha256,
            runtime_versions={
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
                "peft": peft.__version__,
            },
            adapter_bundle=bundle,
            adapter_identity=identity,
            adapter_config_sha256=adapter_config_sha256,
            adapter_model_sha256=adapter_model_sha256,
            training_receipt_file_sha256=training_receipt_file_sha256,
            adapter_tensor_count=adapter_tensor_count,
            active_lora_module_count=active_lora_module_count,
        )
        frozen.base._atomic_write_json(receipt_path, receipt)
        print(frozen.base.canonical_json_bytes(receipt).decode("utf-8"), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


__all__ = [
    "CrossModeCMSGRunnerError",
    "build_inference_receipt",
    "build_parser",
    "launcher_contract",
    "main",
    "validate_cli",
]


if __name__ == "__main__":
    raise SystemExit(main())
