#!/usr/bin/env python3
"""Run native Bernini inference with the dense-flow token adapter enabled."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import types
from typing import Any, Optional

import dense_flow_token_adapter_v1 as adapter_core
import dense_flow_preservation_adapter_v1 as preservation_core
import dense_flow_source_copy_adapter_v1 as source_copy_core
import infer_lora as native


SCHEMA_VERSION = "bernini-dense-flow-token-adapter-inference-v1"


class DenseFlowInferenceError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_registry_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def option(argv: list[str], name: str) -> str:
    try:
        index = argv.index(name)
        return argv[index + 1]
    except (ValueError, IndexError) as error:
        raise DenseFlowInferenceError(f"native arguments lack {name}") from error


def source_copy_schedule_weight(
    *, shared_call_index: int, schedule_start: int, inference_steps: int
) -> float:
    """Return an exact late-denoise source-copy route for native CFG calls.

    Bernini performs two authenticated ``shared_step`` forwards (negative and
    action) per UniPC cell.  Both forwards in a cell must receive the same
    route so classifier-free guidance is not changed by an asymmetric branch.
    """

    if (
        isinstance(shared_call_index, bool)
        or not isinstance(shared_call_index, int)
        or isinstance(schedule_start, bool)
        or not isinstance(schedule_start, int)
        or isinstance(inference_steps, bool)
        or not isinstance(inference_steps, int)
        or inference_steps <= 0
        or not 0 <= shared_call_index < 2 * inference_steps
        or not 0 <= schedule_start <= inference_steps
    ):
        raise DenseFlowInferenceError("source-copy schedule coordinate differs")
    schedule_index = shared_call_index // 2
    return 1.0 if schedule_index >= schedule_start else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dense-flow-checkpoint", required=True)
    parser.add_argument("--flow-bundle", required=True)
    parser.add_argument("--dense-flow-scale", type=float, default=1.0)
    parser.add_argument("--dense-flow-reference-checkpoint")
    parser.add_argument("--dense-flow-reference-mix", type=float, default=1.0)
    parser.add_argument("--dense-flow-preservation-checkpoint")
    parser.add_argument("--dense-flow-preservation-scale", type=float, default=1.0)
    parser.add_argument("--dense-flow-source-copy-checkpoint")
    parser.add_argument("--dense-flow-source-copy-scale", type=float, default=1.0)
    parser.add_argument(
        "--dense-flow-source-copy-schedule-start", type=int, default=0
    )
    parser.add_argument(
        "--dense-flow-source-copy-mode", choices=source_copy_core.MODES
    )
    parser.add_argument(
        "--dense-flow-hard-source-mode",
        choices=("phase0_broadcast", *source_copy_core.FLOW_WARP_FEATURE_OFFSETS),
    )
    parser.add_argument("--dense-flow-hard-source-scale", type=float, default=0.0)
    parser.add_argument(
        "--dense-flow-hard-source-schedule-start", type=int, default=0
    )
    parser.add_argument(
        "--dense-flow-hard-source-block-indices", default="18,22,26,29"
    )
    args, native_argv = parser.parse_known_args()
    if not math.isfinite(args.dense_flow_scale) or not 0.0 < args.dense_flow_scale <= 2.0:
        raise DenseFlowInferenceError("dense-flow scale must be finite in (0,2]")
    if (
        not math.isfinite(args.dense_flow_reference_mix)
        or not 0.0 <= args.dense_flow_reference_mix <= 3.0
    ):
        raise DenseFlowInferenceError("dense-flow reference mix must be finite in [0,3]")
    if not args.dense_flow_reference_checkpoint and args.dense_flow_reference_mix != 1.0:
        raise DenseFlowInferenceError("reference mix requires a reference checkpoint")
    if (
        not math.isfinite(args.dense_flow_preservation_scale)
        or not 0.0 < args.dense_flow_preservation_scale <= 3.0
    ):
        raise DenseFlowInferenceError(
            "dense-flow preservation scale must be finite in (0,3]"
        )
    if (
        not args.dense_flow_preservation_checkpoint
        and args.dense_flow_preservation_scale != 1.0
    ):
        raise DenseFlowInferenceError(
            "preservation scale requires a preservation checkpoint"
        )
    if bool(args.dense_flow_source_copy_checkpoint) != bool(
        args.dense_flow_source_copy_mode
    ):
        raise DenseFlowInferenceError(
            "source-copy checkpoint and mode must be set together"
        )
    if (
        not math.isfinite(args.dense_flow_source_copy_scale)
        or not 0.0 < args.dense_flow_source_copy_scale <= 3.0
    ):
        raise DenseFlowInferenceError(
            "dense-flow source-copy scale must be finite in (0,3]"
        )
    if (
        not args.dense_flow_source_copy_checkpoint
        and args.dense_flow_source_copy_scale != 1.0
    ):
        raise DenseFlowInferenceError(
            "source-copy scale requires a source-copy checkpoint"
        )
    if (
        not args.dense_flow_source_copy_checkpoint
        and args.dense_flow_source_copy_schedule_start != 0
    ):
        raise DenseFlowInferenceError(
            "source-copy schedule requires a source-copy checkpoint"
        )
    if (
        args.dense_flow_preservation_checkpoint
        and args.dense_flow_source_copy_checkpoint
    ):
        raise DenseFlowInferenceError(
            "generic preservation and source-copy branches are mutually exclusive"
        )
    if bool(args.dense_flow_hard_source_mode) != bool(
        args.dense_flow_hard_source_scale > 0.0
    ):
        raise DenseFlowInferenceError(
            "hard source mode and a positive hard source scale must be set together"
        )
    if (
        not math.isfinite(args.dense_flow_hard_source_scale)
        or not 0.0 <= args.dense_flow_hard_source_scale <= 1.0
    ):
        raise DenseFlowInferenceError(
            "hard source scale must be finite in [0,1]"
        )
    if args.dense_flow_hard_source_mode and (
        args.dense_flow_preservation_checkpoint
        or args.dense_flow_source_copy_checkpoint
    ):
        raise DenseFlowInferenceError(
            "hard source transport is mutually exclusive with learned preservation"
        )
    checkpoint = Path(args.dense_flow_checkpoint).expanduser().resolve(strict=True)
    model_path = checkpoint / "adapter_model.safetensors"
    receipt_path = checkpoint / "receipt.json"
    if not model_path.is_file() or not receipt_path.is_file():
        raise DenseFlowInferenceError("dense-flow checkpoint closure is incomplete")
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    contract = receipt.get("training_contract", {})
    if (
        receipt.get("schema_version")
        != "bernini-same-video-dense-flow-adapter-receipt-v1"
        or contract.get("method") != "bernini-same-video-dense-flow-adapter-v1"
        or contract.get("base_transformer_frozen") is not True
        or contract.get("native_iid_initial_noise_unchanged") is not True
        or contract.get("dense_flow_feature_width") != adapter_core.FEATURE_WIDTH
    ):
        raise DenseFlowInferenceError("dense-flow training receipt contract differs")
    full_attention_lora = bool(contract.get("full_attention_lora_enabled", False))
    lora_dir = checkpoint / "adapter"
    if full_attention_lora:
        if (
            contract.get("lora_rank") != 256
            or contract.get("lora_alpha") != 256
            or contract.get("lora_scope") != "all_30_blocks_attn1_attn2_qkvo"
            or contract.get("lora_target_module_count") != 240
            or not (lora_dir / "adapter_config.json").is_file()
            or not (lora_dir / "adapter_model.safetensors").is_file()
        ):
            raise DenseFlowInferenceError("joint LoRA checkpoint closure differs")
    dense_flow_mode = contract.get("dense_flow_mode", "local_mlp")
    if dense_flow_mode not in adapter_core.MODES:
        raise DenseFlowInferenceError("dense-flow training mode differs")
    dense_flow_block_indices = tuple(
        int(item) for item in contract.get("adapter_block_indices", ())
    )
    if (
        not dense_flow_block_indices
        or dense_flow_block_indices
        != tuple(sorted(set(dense_flow_block_indices)))
        or any(
            item < 0 or item >= adapter_core.EXPECTED_BLOCK_COUNT
            for item in dense_flow_block_indices
        )
    ):
        raise DenseFlowInferenceError("dense-flow adapter block contract differs")
    from safetensors.torch import load_file

    flow_path = Path(args.flow_bundle).expanduser().resolve(strict=True)
    features, activity = adapter_core.load_dense_flow_features(flow_path)
    flow_values = load_file(str(flow_path), device="cpu")
    flow_height, flow_width = map(
        int, flow_values["backward_raw"].shape[-2:]
    )
    source_copy_spatial_shape = (flow_height // 2, flow_width // 2)
    del flow_values

    state = load_file(str(model_path), device="cpu")
    if not state or not any(bool(value.count_nonzero().item()) for value in state.values()):
        raise DenseFlowInferenceError("dense-flow checkpoint has no learned nonzero state")
    reference: Optional[dict[str, Any]] = None
    reference_checkpoint: Optional[Path] = None
    reference_receipt: Optional[dict[str, Any]] = None
    reference_model_path: Optional[Path] = None
    if args.dense_flow_reference_checkpoint:
        reference_checkpoint = Path(
            args.dense_flow_reference_checkpoint
        ).expanduser().resolve(strict=True)
        reference_model_path = reference_checkpoint / "adapter_model.safetensors"
        reference_receipt_path = reference_checkpoint / "receipt.json"
        if not reference_model_path.is_file() or not reference_receipt_path.is_file():
            raise DenseFlowInferenceError("reference checkpoint closure is incomplete")
        reference_receipt = json.loads(reference_receipt_path.read_text(encoding="ascii"))
        if (
            reference_receipt.get("schema_version") != receipt.get("schema_version")
            or reference_receipt.get("global_step") != receipt.get("global_step")
            or reference_receipt.get("initialization_digest")
            != receipt.get("initialization_digest")
            or reference_receipt.get("pair_manifest_digest")
            != receipt.get("pair_manifest_digest")
            or reference_receipt.get("optimizer") != receipt.get("optimizer")
        ):
            raise DenseFlowInferenceError(
                "reference/main checkpoints are not a matched weight-space pair"
            )
        reference = load_file(str(reference_model_path), device="cpu")
        if set(reference) != set(state):
            raise DenseFlowInferenceError("reference/main state-key closure differs")
        mixed = {}
        for name, target_value in state.items():
            reference_value = reference[name]
            if reference_value.shape != target_value.shape:
                raise DenseFlowInferenceError("reference/main tensor geometry differs")
            value = reference_value.float().add(
                target_value.float().sub(reference_value.float()).mul(
                    args.dense_flow_reference_mix
                )
            )
            if not bool(value.isfinite().all().item()):
                raise DenseFlowInferenceError("mixed dense-flow state is non-finite")
            mixed[name] = value.to(target_value.dtype)
        state = mixed
    if args.dense_flow_scale != 1.0:
        state = {
            name: (
                value.float().mul(args.dense_flow_scale).to(value.dtype)
                if name.endswith("output.weight") else value
            )
            for name, value in state.items()
        }
    preservation_checkpoint: Optional[Path] = None
    preservation_model_path: Optional[Path] = None
    preservation_state: Optional[dict[str, Any]] = None
    preservation_receipt: Optional[dict[str, Any]] = None
    if args.dense_flow_preservation_checkpoint:
        preservation_checkpoint = Path(
            args.dense_flow_preservation_checkpoint
        ).expanduser().resolve(strict=True)
        preservation_model_path = preservation_checkpoint / "adapter_model.safetensors"
        preservation_receipt_path = preservation_checkpoint / "receipt.json"
        if not preservation_model_path.is_file() or not preservation_receipt_path.is_file():
            raise DenseFlowInferenceError(
                "dense-flow preservation checkpoint closure is incomplete"
            )
        preservation_receipt = json.loads(
            preservation_receipt_path.read_text(encoding="ascii")
        )
        preservation_contract = preservation_receipt.get("training_contract", {})
        if (
            preservation_receipt.get("schema_version")
            != "bernini-same-video-dense-flow-adapter-receipt-v1"
            or preservation_contract.get("method")
            != "bernini-same-video-dense-flow-adapter-v1"
            or preservation_contract.get("source_reconstruction_only") is not True
            or preservation_contract.get(
                "source_reconstruction_motion_features_exact_zero"
            )
            is not True
            or preservation_contract.get("base_transformer_frozen") is not True
            or tuple(preservation_contract.get("adapter_block_indices", ()))
            != adapter_core.BLOCK_INDICES
        ):
            raise DenseFlowInferenceError(
                "dense-flow preservation training contract differs"
            )
        preservation_state = load_file(str(preservation_model_path), device="cpu")
        if not preservation_state or not any(
            bool(value.count_nonzero().item())
            for value in preservation_state.values()
        ):
            raise DenseFlowInferenceError(
                "dense-flow preservation checkpoint has no learned state"
            )
    source_copy_checkpoint: Optional[Path] = None
    source_copy_model_path: Optional[Path] = None
    source_copy_state: Optional[dict[str, Any]] = None
    source_copy_receipt: Optional[dict[str, Any]] = None
    source_copy_block_indices: tuple[int, ...] = adapter_core.BLOCK_INDICES
    if args.dense_flow_source_copy_checkpoint:
        source_copy_checkpoint = Path(
            args.dense_flow_source_copy_checkpoint
        ).expanduser().resolve(strict=True)
        source_copy_model_path = source_copy_checkpoint / "adapter_model.safetensors"
        source_copy_receipt_path = source_copy_checkpoint / "receipt.json"
        if not source_copy_model_path.is_file() or not source_copy_receipt_path.is_file():
            raise DenseFlowInferenceError(
                "dense-flow source-copy checkpoint closure is incomplete"
            )
        source_copy_receipt = json.loads(
            source_copy_receipt_path.read_text(encoding="ascii")
        )
        source_copy_contract = source_copy_receipt.get("training_contract", {})
        source_copy_block_indices = tuple(
            source_copy_contract.get(
                "source_copy_block_indices",
                source_copy_contract.get("adapter_block_indices", ()),
            )
        )
        if (
            source_copy_receipt.get("schema_version")
            != "bernini-same-video-dense-flow-adapter-receipt-v1"
            or source_copy_contract.get("method")
            != "bernini-same-video-dense-flow-adapter-v1"
            or source_copy_contract.get("source_copy_mode")
            != args.dense_flow_source_copy_mode
            or source_copy_contract.get(
                "source_copy_explicit_source_hidden_tokens"
            )
            is not True
            or source_copy_contract.get(
                "source_copy_trained_with_motion_branch_active"
            )
            is not True
            or source_copy_contract.get("base_transformer_frozen") is not True
            or not source_copy_block_indices
            or source_copy_block_indices
            != tuple(sorted(set(source_copy_block_indices)))
            or any(
                int(item) < 0 or int(item) >= adapter_core.EXPECTED_BLOCK_COUNT
                for item in source_copy_block_indices
            )
        ):
            raise DenseFlowInferenceError(
                "dense-flow source-copy training contract differs"
            )
        if (
            source_copy_contract.get("frozen_motion_model_sha256")
            != file_sha256(model_path)
        ):
            raise DenseFlowInferenceError(
                "source-copy checkpoint was trained with another motion adapter"
            )
        source_copy_state = load_file(str(source_copy_model_path), device="cpu")
        if not source_copy_state or not any(
            bool(value.count_nonzero().item()) for value in source_copy_state.values()
        ):
            raise DenseFlowInferenceError(
                "dense-flow source-copy checkpoint has no learned state"
            )
    try:
        hard_source_block_indices = tuple(
            int(item.strip())
            for item in args.dense_flow_hard_source_block_indices.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise DenseFlowInferenceError(
            "hard source block indices must be integers"
        ) from error
    if args.dense_flow_hard_source_mode and (
        not hard_source_block_indices
        or hard_source_block_indices
        != tuple(sorted(set(hard_source_block_indices)))
        or any(
            item < 0 or item >= adapter_core.EXPECTED_BLOCK_COUNT
            for item in hard_source_block_indices
        )
    ):
        raise DenseFlowInferenceError(
            "hard source block indices must be sorted unique in [0,29]"
        )
    output = Path(option(native_argv, "--output"))
    if not output.is_absolute():
        raise DenseFlowInferenceError("native output must be absolute")
    sidecar = output.with_name(output.name + ".dense-flow.json")
    if sidecar.exists() or sidecar.is_symlink():
        raise DenseFlowInferenceError("refusing to overwrite dense-flow sidecar")

    # The wrapper owns both learned routes. Native inference remains in its
    # audited base-only transport path; immediately after construction we
    # strictly load+merge the jointly trained LoRA and then install the motion
    # adapter into that same model call.
    native_argv = [value for value in native_argv if value != "--base-only"]
    if "--adapter-checkpoint" in native_argv:
        raise DenseFlowInferenceError("dense-flow inference cannot combine a PEFT LoRA")
    native_argv.append("--base-only")
    inference_steps = int(option(native_argv, "--num-inference-steps"))
    if not 0 <= args.dense_flow_source_copy_schedule_start <= inference_steps:
        raise DenseFlowInferenceError(
            "source-copy schedule start must lie in [0,num-inference-steps]"
        )
    if not 0 <= args.dense_flow_hard_source_schedule_start <= inference_steps:
        raise DenseFlowInferenceError(
            "hard source schedule start must lie in [0,num-inference-steps]"
        )
    capture: dict[str, Any] = {"constructed": 0, "sample_calls": 0}
    original_activate = native.trainer.activate_source_trees

    def patched_activate(*activate_args: Any, **activate_kwargs: Any) -> Any:
        result = original_activate(*activate_args, **activate_kwargs)
        import bernini.models.renderer as renderer_module

        original_class = renderer_module.BerniniRendererModel

        def constructed(config: Any) -> Any:
            model = original_class(config)
            if full_attention_lora:
                from peft import LoraConfig, PeftModel

                targets = native.trainer.select_attention_projection_names(model)
                if (
                    len(targets) != 240
                    or native.trainer.object_sha256(targets)
                    != contract.get("lora_target_modules_sha256")
                ):
                    raise DenseFlowInferenceError(
                        "runtime full-attention LoRA target registry differs"
                    )
                lora_config = LoraConfig.from_pretrained(
                    str(lora_dir), local_files_only=True
                )
                lora_config.target_modules = set(targets)
                peft_model = PeftModel.from_pretrained(
                    model,
                    str(lora_dir),
                    is_trainable=False,
                    config=lora_config,
                    local_files_only=True,
                )
                model = peft_model.merge_and_unload(safe_merge=True)
                if any("lora_" in name for name, _ in model.named_modules()):
                    raise DenseFlowInferenceError("LoRA modules remained after merge")
            handle = adapter_core.install_dense_flow_adapter(
                model,
                mode=dense_flow_mode,
                block_indices=dense_flow_block_indices,
            )
            handle.load_state_dict_strict(state)
            if handle.zero_effect():
                raise DenseFlowInferenceError("loaded dense-flow state remained zero effect")
            original_sample = model.sample
            preservation_handle = None
            if preservation_state is not None:
                preservation_handle = preservation_core.install_preservation_adapter(
                    model
                )
                preservation_handle.load_dense_flow_state_strict(
                    preservation_state,
                    output_scale=args.dense_flow_preservation_scale,
                )
                if preservation_handle.zero_effect():
                    raise DenseFlowInferenceError(
                        "loaded preservation state remained zero effect"
                    )
            source_copy_handle = None
            if source_copy_state is not None:
                source_copy_handle = source_copy_core.install_source_copy_adapter(
                    model,
                    mode=args.dense_flow_source_copy_mode,
                    block_indices=source_copy_block_indices,
                )
                source_copy_handle.load_state_dict_strict(
                    source_copy_state,
                    output_scale=args.dense_flow_source_copy_scale,
                )
                if source_copy_handle.zero_effect():
                    raise DenseFlowInferenceError(
                        "loaded source-copy state remained zero effect"
                    )
            hard_source_handle = None
            if args.dense_flow_hard_source_mode:
                hard_source_handle = source_copy_core.install_hard_source_transport(
                    model,
                    mode=args.dense_flow_hard_source_mode,
                    scale=args.dense_flow_hard_source_scale,
                    block_indices=hard_source_block_indices,
                )

            def sampled(*sample_args: Any, **sample_kwargs: Any) -> Any:
                capture["sample_calls"] += 1
                invocation = adapter_core.DenseFlowInvocation(
                    features,
                    activity,
                    mode=dense_flow_mode,
                    spatial_shape=source_copy_spatial_shape,
                )
                with contextlib.ExitStack() as stack:
                    stack.enter_context(adapter_core.dense_flow_invocation(invocation))
                    if preservation_handle is not None:
                        stack.enter_context(
                            preservation_core.preservation_invocation(
                                preservation_core.PreservationInvocation(activity)
                            )
                        )
                    source_transport_handle = (
                        source_copy_handle
                        if source_copy_handle is not None
                        else hard_source_handle
                    )
                    source_transport_mode = (
                        args.dense_flow_source_copy_mode
                        if source_copy_handle is not None
                        else args.dense_flow_hard_source_mode
                    )
                    if source_transport_handle is not None:
                        stack.enter_context(
                            source_copy_core.source_copy_invocation(
                                source_copy_core.SourceCopyInvocation(
                                    activity,
                                    mode=source_transport_mode,
                                    spatial_shape=(
                                        source_copy_spatial_shape
                                        if source_transport_mode
                                        in source_copy_core.SPATIAL_MODES
                                        else None
                                    ),
                                    motion_features=(
                                        features
                                        if source_transport_mode
                                        in source_copy_core.FLOW_WARP_FEATURE_OFFSETS
                                        else None
                                    ),
                                )
                            )
                        )
                    if source_transport_handle is None:
                        return original_sample(*sample_args, **sample_kwargs)

                    diffusion = getattr(model, "diff_dec", None)
                    original_shared_step = getattr(diffusion, "shared_step", None)
                    if diffusion is None or not callable(original_shared_step):
                        raise DenseFlowInferenceError(
                            "native source-copy schedule lacks diffusion.shared_step"
                        )
                    instance = vars(diffusion)
                    had_instance = "shared_step" in instance
                    previous_instance = instance.get("shared_step")
                    shared_calls = 0

                    def scheduled_shared_step(*shared_args: Any, **shared_kwargs: Any) -> Any:
                        nonlocal shared_calls
                        schedule_start = (
                            args.dense_flow_source_copy_schedule_start
                            if source_copy_handle is not None
                            else args.dense_flow_hard_source_schedule_start
                        )
                        weight = source_copy_schedule_weight(
                            shared_call_index=shared_calls,
                            schedule_start=schedule_start,
                            inference_steps=inference_steps,
                        )
                        shared_calls += 1
                        with source_copy_core.source_copy_denoise_weight(weight):
                            return original_shared_step(*shared_args, **shared_kwargs)

                    try:
                        setattr(diffusion, "shared_step", scheduled_shared_step)
                        result = original_sample(*sample_args, **sample_kwargs)
                    finally:
                        if had_instance:
                            setattr(diffusion, "shared_step", previous_instance)
                        else:
                            delattr(diffusion, "shared_step")
                    if shared_calls != 2 * inference_steps:
                        raise DenseFlowInferenceError(
                            "native source-copy schedule did not observe two forwards per step"
                        )
                    capture["source_copy_shared_step_calls"] = shared_calls
                    return result

            model.sample = sampled
            capture["constructed"] += 1
            capture["full_attention_lora_enabled"] = full_attention_lora
            capture["trainable_parameter_count"] = sum(
                int(parameter.numel())
                for _, parameter in handle.trainable_named_parameters()
            )
            capture["preservation_parameter_count"] = (
                sum(
                    int(parameter.numel())
                    for _, parameter in preservation_handle.trainable_named_parameters()
                )
                if preservation_handle is not None
                else 0
            )
            capture["source_copy_parameter_count"] = (
                sum(
                    int(parameter.numel())
                    for _, parameter in source_copy_handle.trainable_named_parameters()
                )
                if source_copy_handle is not None
                else 0
            )
            capture["hard_source_transport_enabled"] = hard_source_handle is not None
            return model

        renderer_module.BerniniRendererModel = constructed
        return result

    native.trainer.activate_source_trees = patched_activate
    status = native.main(native_argv)
    if capture["constructed"] != 1 or capture["sample_calls"] != 1:
        raise DenseFlowInferenceError(
            "dense-flow adapter was not invoked exactly once in native sampling"
        )
    if int(os.environ.get("RANK", "0")) == 0:
        native_receipt = output.with_name(output.name + ".receipt.json")
        if not output.is_file() or not native_receipt.is_file():
            raise DenseFlowInferenceError("native output/receipt closure is incomplete")
        value = {
            "schema_version": SCHEMA_VERSION,
            "output": str(output),
            "output_sha256": file_sha256(output),
            "native_runtime_receipt": str(native_receipt),
            "native_runtime_receipt_sha256": file_sha256(native_receipt),
            "native_runtime_base_only_label_is_transport_only": True,
            "dense_flow_checkpoint": str(checkpoint),
            "dense_flow_checkpoint_step": int(receipt["global_step"]),
            "dense_flow_model_sha256": file_sha256(model_path),
            "dense_flow_state_digest": tensor_registry_sha256(state),
            "dense_flow_output_scale": float(args.dense_flow_scale),
            "dense_flow_mode": dense_flow_mode,
            "dense_flow_block_indices": list(dense_flow_block_indices),
            "dense_flow_attention_memory_shape": (
                list(adapter_core.ATTENTION_MEMORY_SHAPES[dense_flow_mode])
                if dense_flow_mode in adapter_core.ATTENTION_MEMORY_SHAPES
                else None
            ),
            "full_attention_lora_enabled": full_attention_lora,
            "full_attention_lora_model_sha256": (
                file_sha256(lora_dir / "adapter_model.safetensors")
                if full_attention_lora else None
            ),
            "dense_flow_source_copy_schedule_start": int(
                args.dense_flow_source_copy_schedule_start
            ),
            "dense_flow_source_copy_shared_step_calls": int(
                capture.get("source_copy_shared_step_calls", 0)
            ),
            "dense_flow_scale_scope": "output_projection_only",
            "dense_flow_reference_checkpoint": (
                str(reference_checkpoint) if reference_checkpoint else None
            ),
            "dense_flow_reference_model_sha256": (
                file_sha256(reference_model_path) if reference_model_path else None
            ),
            "dense_flow_reference_mix": float(args.dense_flow_reference_mix),
            "dense_flow_reference_mix_formula": (
                "reference + mix * (main - reference)"
                if reference_checkpoint else None
            ),
            "dense_flow_preservation_checkpoint": (
                str(preservation_checkpoint) if preservation_checkpoint else None
            ),
            "dense_flow_preservation_model_sha256": (
                file_sha256(preservation_model_path)
                if preservation_model_path else None
            ),
            "dense_flow_preservation_scale": float(
                args.dense_flow_preservation_scale
            ),
            "dense_flow_preservation_formula": (
                "motion_residual + scaled_independent_zero_flow_hidden_residual"
                if preservation_checkpoint else None
            ),
            "dense_flow_source_copy_checkpoint": (
                str(source_copy_checkpoint) if source_copy_checkpoint else None
            ),
            "dense_flow_source_copy_model_sha256": (
                file_sha256(source_copy_model_path)
                if source_copy_model_path else None
            ),
            "dense_flow_source_copy_scale": float(
                args.dense_flow_source_copy_scale
            ),
            "dense_flow_source_copy_mode": args.dense_flow_source_copy_mode,
            "dense_flow_source_copy_block_indices": (
                list(source_copy_block_indices) if source_copy_checkpoint else None
            ),
            "dense_flow_source_copy_formula": (
                "frozen_motion_residual + scaled_source_token_residual"
                if source_copy_checkpoint else None
            ),
            "dense_flow_hard_source_mode": args.dense_flow_hard_source_mode,
            "dense_flow_hard_source_scale": float(
                args.dense_flow_hard_source_scale
            ),
            "dense_flow_hard_source_schedule_start": int(
                args.dense_flow_hard_source_schedule_start
            ),
            "dense_flow_hard_source_block_indices": (
                list(hard_source_block_indices)
                if args.dense_flow_hard_source_mode else None
            ),
            "dense_flow_hard_source_formula": (
                "target_hidden = lerp(target_hidden, "
                "flow_warp(source_phase0_hidden), scale) on active target tokens"
                if args.dense_flow_hard_source_mode else None
            ),
            "flow_bundle": str(flow_path),
            "flow_bundle_sha256": file_sha256(flow_path),
            "motion_features_shape": list(map(int, features.shape)),
            "active_target_token_count": int(activity.count_nonzero().item()),
            "adapter_constructed_count": capture["constructed"],
            "adapter_sample_call_count": capture["sample_calls"],
            "trainable_parameter_count": capture["trainable_parameter_count"],
            "preservation_parameter_count": capture[
                "preservation_parameter_count"
            ],
            "source_copy_parameter_count": capture[
                "source_copy_parameter_count"
            ],
            "base_weights_modified": False,
            "native_iid_initial_noise_unchanged": True,
            "anchor_rgb_or_vae_latent_used_by_adapter": False,
            "qwen_used": False,
        }
        value["receipt_digest"] = native.object_sha256(value)
        temporary = sidecar.with_name(sidecar.name + f".tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, sidecar)
        print(json.dumps(value, sort_keys=True), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
