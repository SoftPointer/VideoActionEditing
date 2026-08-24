#!/usr/bin/env python3
"""Plan-aware source-only inference for Bernini P3T-LoRA.

Two samplers are exposed for a controlled ablation:

``differential``
    DynaEdit-inspired source-state integration of
    ``V(action | source) - V(no-op | source)``.  It uses one fixed noise and no
    ANC, target video, mask, track, or first-frame anchor.

``standard``
    The official Bernini ``v2v_apg`` sampler with the same adapter.  This
    separates gains from the paired CDF-LoRA objective and gains from the new
    solver.

``motion_strength=0`` is an exact latent-level source bypass in both modes.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import differential_sampler  # noqa: E402
import infer_lora as legacy_infer  # noqa: E402
import motion_residual as motion  # noqa: E402
import p3t  # noqa: E402
import train_p3t_lora as delta_train  # noqa: E402
import train_lora as legacy_train  # noqa: E402


INFERENCE_RECEIPT_SCHEMA = "bernini-r-1p3b-p3t-lora-inference-receipt-v1"
SAMPLING_MODES = ("differential", "standard")


class DeltaInferenceError(RuntimeError):
    """Raised before output when a P3T-LoRA inference contract is violated."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeltaInferenceError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise DeltaInferenceError(f"{label} must contain one JSON object")
    return value


def _validate_receipt_digest(receipt: Mapping[str, Any]) -> str:
    candidate = dict(receipt)
    declared = candidate.pop("receipt_digest", None)
    if (
        not isinstance(declared, str)
        or re.fullmatch(r"[0-9a-f]{64}", declared) is None
        or legacy_train.object_sha256(candidate) != declared
    ):
        raise DeltaInferenceError("training receipt digest differs")
    return declared


def validate_training_adapter_contract(
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = legacy_train.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    digest = _validate_receipt_digest(receipt)
    if receipt.get("schema_version") != delta_train.RECEIPT_SCHEMA:
        raise DeltaInferenceError("training receipt schema differs")
    if receipt.get("method") != delta_train.METHOD_NAME:
        raise DeltaInferenceError("training method identity differs")
    if receipt.get("bernini_commit") != legacy_train.BERNINI_OFFICIAL_COMMIT:
        raise DeltaInferenceError("training Bernini revision differs")
    if receipt.get("veomni_commit") != legacy_train.VEOMNI_TESTED_COMMIT:
        raise DeltaInferenceError("training VeOmni revision differs")
    checkpoint = receipt.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("tree_sha256") != expected_checkpoint_tree_sha256
    ):
        raise DeltaInferenceError("training checkpoint tree differs")
    step = receipt.get("global_step")
    if type(step) is not int or step <= 0:
        raise DeltaInferenceError("training global_step must be positive")
    adapter = receipt.get("adapter")
    if not isinstance(adapter, dict):
        raise DeltaInferenceError("training receipt lacks adapter contract")
    scope = adapter.get("scope")
    targets = adapter.get("target_modules")
    if scope not in motion.MODULE_SCOPES:
        raise DeltaInferenceError("training LoRA scope is invalid")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(name, str) for name in targets)
        or targets != sorted(set(targets))
        or adapter.get("target_module_count") != len(targets)
        or adapter.get("target_modules_sha256")
        != legacy_train.object_sha256(targets)
    ):
        raise DeltaInferenceError("training exact target-module set differs")
    immutable = receipt.get("immutable_contract")
    if not isinstance(immutable, dict):
        raise DeltaInferenceError("training receipt lacks immutable contract")
    value = immutable.get("value")
    if (
        not isinstance(value, dict)
        or immutable.get("digest") != legacy_train.object_sha256(value)
        or value.get("target_modules") != targets
        or value.get("lora_scope") != scope
        or value.get("checkpoint_tree_sha256") != expected_checkpoint_tree_sha256
    ):
        raise DeltaInferenceError("training immutable contract differs")
    supervision = receipt.get("supervision")
    if (
        not isinstance(supervision, dict)
        or supervision.get("target_used_as_condition") is not False
        or supervision.get("external_mask_track_pose_trajectory") is not False
        or supervision.get("unreviewed_full_target_weight") != 0.0
        or supervision.get("shared_source_posterior_mode") is not True
        or supervision.get("shared_sigma") is not True
        or supervision.get("shared_diffusion_noise") is not True
    ):
        raise DeltaInferenceError("training source-only/paired supervision contract differs")
    distributed = receipt.get("distributed")
    if not isinstance(distributed, dict) or distributed.get("ulysses_size") not in (1, 4):
        raise DeltaInferenceError("training Ulysses contract differs")
    if receipt.get("production_claim_forbidden") is not True:
        raise DeltaInferenceError("training receipt lost production restriction")
    if receipt.get("scientific_claim_authorized") is not False:
        raise DeltaInferenceError("training receipt carries an unsupported scientific claim")

    if adapter_config.get("peft_type") != "LORA":
        raise DeltaInferenceError("adapter is not LoRA")
    if adapter_config.get("r") != legacy_train.LORA_RANK:
        raise DeltaInferenceError("adapter rank differs")
    if float(adapter_config.get("lora_alpha", -1)) != legacy_train.LORA_ALPHA:
        raise DeltaInferenceError("adapter alpha differs")
    if float(adapter_config.get("lora_dropout", -1)) != 0.0:
        raise DeltaInferenceError("adapter dropout differs")
    if adapter_config.get("bias") != "none":
        raise DeltaInferenceError("adapter bias differs")
    if adapter_config.get("modules_to_save") not in (None, []):
        raise DeltaInferenceError("modules_to_save are forbidden")
    serialized = adapter_config.get("target_modules")
    if not isinstance(serialized, list) or not all(isinstance(name, str) for name in serialized):
        raise DeltaInferenceError("adapter serialized target_modules are invalid")
    # PEFT may compact exact fully-qualified names.  The receipt remains the
    # authority and is rebound before construction; reject anything that could
    # not plausibly be a suffix set of the receipt targets.
    if not serialized or any(
        not any(target == name or target.endswith(f".{name}") for target in targets)
        for name in serialized
    ):
        raise DeltaInferenceError("adapter compact target_modules exceed receipt scope")
    return {
        "receipt_digest": digest,
        "global_step": step,
        "scope": scope,
        "targets": targets,
        "target_modules_sha256": adapter["target_modules_sha256"],
        "transformers_version": receipt.get("transformers_version"),
        "noop_instruction_sha256": value.get("noop_instruction_sha256"),
        "method_source_revision": value.get("method_source_revision"),
        "method_source_archive_sha256": value.get("method_source_archive_sha256"),
    }


def expected_adapter_state_keys(targets: Sequence[str]) -> set[str]:
    return {
        f"base_model.model.{target}.lora_{factor}.weight"
        for target in targets
        for factor in ("A", "B")
    }


def _strict_load_adapter(
    *,
    base_model: Any,
    adapter_dir: Path,
    adapter_model_path: Path,
    targets: Sequence[str],
) -> tuple[Any, int]:
    import torch
    from peft import LoraConfig, PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file as load_safetensors

    available = legacy_train.select_attention_projection_names(base_model)
    scope = None
    for candidate in sorted(motion.MODULE_SCOPES):
        try:
            if motion.select_lora_scope(available, candidate) == list(targets):
                scope = candidate
                break
        except motion.MotionContractError:
            continue
    if scope is None:
        raise DeltaInferenceError("runtime Bernini cannot reproduce receipt LoRA scope")
    config = LoraConfig.from_pretrained(str(adapter_dir), local_files_only=True)
    config.target_modules = set(targets)
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=False,
        config=config,
        local_files_only=True,
    )
    saved = load_safetensors(str(adapter_model_path), device="cpu")
    loaded = get_peft_model_state_dict(model, adapter_name="default")
    expected = expected_adapter_state_keys(targets)
    if set(saved) != expected or set(loaded) != expected:
        raise DeltaInferenceError(
            "strict adapter state scope differs: "
            f"saved_delta={len(set(saved) ^ expected)} "
            f"loaded_delta={len(set(loaded) ^ expected)}"
        )
    unequal = [
        key
        for key in sorted(expected)
        if not bool(torch.equal(saved[key].cpu(), loaded[key].cpu()))
    ]
    if unequal:
        raise DeltaInferenceError(f"strict adapter tensor reload differs: {unequal[:4]}")
    model.requires_grad_(False)
    model.eval()
    return model, len(expected)


def apply_adapter_strength(model: Any, strength: float) -> int:
    """Scale every active PEFT LoRA branch once, without merging."""

    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or float(strength) < 0.0
    ):
        raise DeltaInferenceError("motion strength must be finite and non-negative")
    count = 0
    for _, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if not isinstance(scaling, dict) or "default" not in scaling:
            continue
        if "default" not in lora_a or "default" not in lora_b:
            raise DeltaInferenceError("PEFT LoRA layer has incomplete default adapter")
        base = float(scaling["default"])
        scaling["default"] = base * float(strength)
        count += 1
    if count <= 0:
        raise DeltaInferenceError("no PEFT LoRA layers were scaled")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run plan-aware source-only Bernini P3T-LoRA inference"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--compiled-plan-text", default=None,
        help="oracle/frozen internal plan for diagnosis only",
    )
    parser.add_argument("--oracle-plan-diagnostic", action="store_true")
    parser.add_argument("--noop-instruction", default=motion.DEFAULT_NOOP_INSTRUCTION)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sampling-mode", choices=SAMPLING_MODES, default="standard")
    parser.add_argument("--motion-strength", type=float, default=1.0)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=legacy_infer.NUM_INFERENCE_STEPS_DEFAULT,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy_train.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy_train.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy_train.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    try:
        legacy_infer.validate_cli(args)
    except legacy_infer.InferenceContractError as error:
        raise DeltaInferenceError(str(error)) from error
    if args.sampling_mode not in SAMPLING_MODES:
        raise DeltaInferenceError("unknown sampling mode")
    if (
        isinstance(args.motion_strength, bool)
        or not math.isfinite(float(args.motion_strength))
        or float(args.motion_strength) < 0.0
    ):
        raise DeltaInferenceError("motion strength must be finite and non-negative")
    if not isinstance(args.noop_instruction, str) or not args.noop_instruction.strip() or "\x00" in args.noop_instruction:
        raise DeltaInferenceError("no-op instruction must be non-empty text")
    if args.compiled_plan_text is not None and (
        not isinstance(args.compiled_plan_text, str)
        or not args.compiled_plan_text.strip()
        or "\x00" in args.compiled_plan_text
    ):
        raise DeltaInferenceError("compiled plan text must be non-empty text without NUL")
    if (args.compiled_plan_text is not None) != bool(args.oracle_plan_diagnostic):
        raise DeltaInferenceError(
            "compiled-plan-text and oracle-plan-diagnostic must be supplied together; "
            "oracle plans cannot be reported as source-only results"
        )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = legacy_train.canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute():
        raise DeltaInferenceError("source video must be an absolute path")
    try:
        source_path = legacy_infer._plain_file(
            source_requested.resolve(strict=True), label="source video"
        )
        output_path, output_receipt_path = legacy_infer._resolve_output(args.output)
        bundle = legacy_infer.resolve_adapter_bundle(args.adapter_checkpoint)
    except legacy_infer.InferenceContractError as error:
        raise DeltaInferenceError(str(error)) from error
    adapter_config = _read_json(bundle.adapter_config_path, label="adapter config")
    training_receipt = _read_json(bundle.training_receipt_path, label="training receipt")
    adapter_identity = validate_training_adapter_contract(
        adapter_config,
        training_receipt,
        expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
    )
    expected_noop_hash = adapter_identity["noop_instruction_sha256"]
    actual_noop_hash = hashlib.sha256(args.noop_instruction.encode("utf-8")).hexdigest()
    if actual_noop_hash != expected_noop_hash:
        raise DeltaInferenceError("inference no-op instruction differs from training")
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy_train.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = legacy_train.validate_checkpoint(args.checkpoint)
        inference_files = legacy_infer.validate_inference_source_files(bernini_root)
    except (legacy_train.TrainingContractError, legacy_infer.InferenceContractError) as error:
        raise DeltaInferenceError(str(error)) from error
    if transformer_config["num_attention_heads"] % legacy_infer.ULYSSES_SIZE:
        raise DeltaInferenceError("attention heads are not divisible by Ulysses=4")
    legacy_train.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    import peft
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode

    if transformers_version != adapter_identity["transformers_version"]:
        raise DeltaInferenceError(
            "Transformers version differs from training: "
            f"{transformers_version} != {adapter_identity['transformers_version']}"
        )
    distributed = legacy_infer.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise DeltaInferenceError("CDF inference requires AUH ROCm-visible GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    source_tensor, source_metadata = legacy_infer.prepare_exact_source(source_path)
    source_sha256 = legacy_infer.file_sha256(source_path)
    config_dir = bernini_root / "configs/bernini_renderer_wan21_1p3b"
    config = BerniniRendererConfig.from_pretrained(
        str(config_dir),
        local_files_only=True,
        **legacy_infer.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy_train.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    base_model = BerniniRendererModel(config)
    base_model.requires_grad_(False)
    base_model.eval()
    model, adapter_tensor_count = _strict_load_adapter(
        base_model=base_model,
        adapter_dir=bundle.adapter_dir,
        adapter_model_path=bundle.adapter_model_path,
        targets=adapter_identity["targets"],
    )
    # Differential mode scales the complete action-minus-no-op field below;
    # scaling the adapter here as well would accidentally square the control.
    adapter_strength = (
        float(args.motion_strength) if args.sampling_mode == "standard" else 1.0
    )
    scaled_layer_count = apply_adapter_strength(model, adapter_strength)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        **legacy_infer.tokenizer_load_kwargs(),
    )
    model_instruction = (
        args.compiled_plan_text
        if args.oracle_plan_diagnostic
        else p3t.compile_generic_phase_wrapper(args.instruction)
    )
    action_prompt = legacy_infer.build_training_prompt(
        model_instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = legacy_infer.build_training_prompt(
        args.noop_instruction, prompt_cleaner=prompt_clean
    )
    action_ids, action_mask = legacy_infer._tokenize_training_prompt(
        tokenizer, action_prompt
    )
    noop_ids, noop_mask = legacy_infer._tokenize_training_prompt(tokenizer, noop_prompt)
    negative_ids, negative_mask = legacy_infer._tokenize_renderer_negative(
        tokenizer, legacy_infer.DEFAULT_NEGATIVE_PROMPT
    )

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.requires_grad_(False)
    vae.eval().to(device)
    with torch.no_grad():
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        )
    bucket = source_metadata["source_derived_bucket_hw"]
    expected_shape = (
        1,
        int(vae.config.z_dim),
        legacy_infer.LATENT_FRAME_COUNT,
        int(bucket[0]) // 8,
        int(bucket[1]) // 8,
    )
    if tuple(int(value) for value in source_latent.shape) != expected_shape:
        raise DeltaInferenceError("source latent geometry differs")
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    if float(args.motion_strength) == 0.0:
        generated_latent = source_latent
        trace = differential_sampler.DifferentialFlowTrace(True, (), ())
    elif args.sampling_mode == "differential":
        renderer = model.get_base_model()
        renderer.t5_text_encoder.to(device)
        with torch.no_grad():
            action_embeds = renderer.encode_prompt(
                action_ids.to(device), action_mask.to(device)
            )
            noop_embeds = renderer.encode_prompt(noop_ids.to(device), noop_mask.to(device))
        renderer.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()
        # Stage only the active 1.3B DiT after VAE encode and T5 prompt encode.
        # Moving the entire renderer before VAE encoding creates an unnecessary
        # VAE+T5+DiT peak and has caused avoidable OOMs in similar jobs.
        renderer.diff_dec.transformer.to(device)
        generated_latent, trace = differential_sampler.sample_differential_flow(
            model,
            source_latent=source_latent,
            action_prompt_embeds=action_embeds,
            noop_prompt_embeds=noop_embeds,
            config=differential_sampler.DifferentialFlowConfig(
                num_inference_steps=args.num_inference_steps,
                flow_shift=legacy_infer.FLOW_SHIFT,
                seed=args.seed,
                # LoRA strength is already applied above.  This scale controls
                # the entire action/no-op velocity difference.
                motion_scale=float(args.motion_strength),
            ),
            return_trace=True,
        )
    else:
        sampling = legacy_infer.sampler_contract(
            steps=args.num_inference_steps, seed=args.seed
        )
        with torch.no_grad():
            generated_latent = model.sample(
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
        trace = None
    if tuple(int(value) for value in generated_latent.shape) != expected_shape:
        raise DeltaInferenceError("generated latent geometry differs")
    model.to("cpu")
    del source_latent
    torch.cuda.empty_cache()

    if distributed.rank == 0:
        vae.to(device)
        with torch.no_grad():
            output = _vae_decode(vae, generated_latent)
        vae.to("cpu")
        expected_video_shape = (
            legacy_infer.FRAME_COUNT,
            int(bucket[0]),
            int(bucket[1]),
            3,
        )
        if tuple(int(value) for value in output.shape) != expected_video_shape:
            raise DeltaInferenceError("decoded video geometry differs")
        temporary_output = output_path.with_name(
            f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
        )
        save_output(output, str(temporary_output), fps=int(legacy_infer.FPS))
        os.replace(temporary_output, output_path)
        from tools import materialize_vae

        encoded, fps, encoded_hw = materialize_vae._decode_exact_video(output_path)
        legacy_infer.validate_exact_video_metadata(int(encoded.shape[0]), fps)
        if tuple(encoded_hw) != tuple(bucket):
            raise DeltaInferenceError("encoded output geometry differs")
        trace_value = None
        if trace is not None:
            trace_value = {
                "identity_bypassed": trace.identity_bypassed,
                "sigmas": list(trace.sigmas),
                "delta_rms": list(trace.delta_rms),
            }
        receipt: dict[str, Any] = {
            "schema_version": INFERENCE_RECEIPT_SCHEMA,
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
            "bernini_inference_files": inference_files,
            "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
            "adapter": {
                "path": str(bundle.checkpoint_root),
                "sha256": legacy_infer.file_sha256(bundle.adapter_model_path),
                "training_receipt_digest": adapter_identity["receipt_digest"],
                "training_global_step": adapter_identity["global_step"],
                "scope": adapter_identity["scope"],
                "target_modules_sha256": adapter_identity["target_modules_sha256"],
                "tensor_count": adapter_tensor_count,
                "strictly_reloaded": True,
                "merged": False,
                "scaled_layer_count": scaled_layer_count,
                "adapter_strength": adapter_strength,
            },
            "input": {
                "source_video_path": str(source_path),
                "source_video_sha256": source_sha256,
                "instruction_utf8_sha256": hashlib.sha256(
                    args.instruction.encode("utf-8")
                ).hexdigest(),
                "model_conditioning_text_sha256": hashlib.sha256(
                    model_instruction.encode("utf-8")
                ).hexdigest(),
                "internal_compiled_plan_supplied": args.compiled_plan_text is not None,
                "conditioning_plan_mode": (
                    "oracle_frozen_diagnostic"
                    if args.oracle_plan_diagnostic
                    else "source_only_deterministic_generic_21phase"
                ),
                "accepted_external_conditions": ["source_video", "edit_instruction"],
                "target_video": False,
                "mask_track_pose_trajectory": False,
                "first_frame_anchor": False,
            },
            "preprocessing": source_metadata,
            "sampling": {
                "mode": args.sampling_mode,
                "motion_strength": float(args.motion_strength),
                "num_inference_steps": args.num_inference_steps,
                "seed": args.seed,
                "flow_shift": legacy_infer.FLOW_SHIFT,
                "differential_contract": differential_sampler.sampler_contract()
                if args.sampling_mode == "differential"
                else None,
                "trace": trace_value,
            },
            "output": {
                "path": str(output_path),
                "sha256": legacy_infer.file_sha256(output_path),
                "frame_count": legacy_infer.FRAME_COUNT,
                "fps": legacy_infer.FPS,
                "height": int(bucket[0]),
                "width": int(bucket[1]),
            },
            "runtime_versions": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
                "peft": peft.__version__,
            },
            "experimental_inference": True,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
        }
        receipt["receipt_digest"] = legacy_train.object_sha256(receipt)
        _atomic_write_json(output_receipt_path, receipt)
        print(legacy_train.canonical_json_bytes(receipt).decode("utf-8"), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
