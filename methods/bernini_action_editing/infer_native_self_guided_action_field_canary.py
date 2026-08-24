#!/usr/bin/env python3
"""Exact81 frozen Bernini self-guided action-field composition canary.

Each cell compares native RV2V with two fixed strengths of a same-state frozen
guided-T2V action-field quotient.  All arms receive the same full source video,
four independently VAE-encoded source frames, RV2V target caption, official
Gaussian seed, and exact40 UniPC schedule.  The only difference is whether the
native target-token velocity receives::

    lambda(sigma) * (T2V-APG[target action] - T2V-APG[source action])

Both counterfactual T2V fields use a third target-only negative query, the
active UniPC sigma, and independent Bernini APG momentum buffers.  Composition
happens after native RV2V APG and immediately before the one official
``scheduler.step`` call.

No generated owner media, target video, mask, track, pose, flow, custom initial
noise, optimizer, or parameter update is consumed.  A successful arm is a
teacher-field canary for later LoRA distillation, not a trained checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import infer_source_value_residual_oracle as value_audit  # noqa: E402
from self_guided_action_field_v1 import (  # noqa: E402
    ActionFieldConfig,
    NativeRV2VActionFieldPatch,
)
import tri_branch_unipc as sampler_contract  # noqa: E402


METHOD = "frozen-bernini-guided-t2v-action-field-canary"
SCHEMA_VERSION = "bernini-self-guided-action-field-canary-receipt-v3"
NATIVE_GUIDANCE_MODE = "v2v_apg"
FRAME_COUNT = 81
LATENT_PHASES = 21
REFERENCE_INDICES = (0, 27, 53, 80)
NUM_INFERENCE_STEPS = 40
ULYSSES_SIZE = 4
FPS = 25
ARM_SCALES = (
    ("native-rv2v", None),
    ("action-field-075", 0.75),
    ("action-field-150", 1.50),
)

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ActionFieldCanaryError(RuntimeError):
    """Raised before incomplete or ambiguous evidence is published."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_digest(value: Any) -> str:
    return _sha256_bytes(native.legacy.canonical_json_bytes(value))


def _strong_model_freeze_certificate(model: Any) -> dict[str, Any]:
    """Hash every frozen parameter and buffer, not only ``requires_grad``.

    The older shared oracle certificate proves that an optimizer cannot see a
    trainable tensor, but it does not prove that an inference hook left tensor
    values unchanged.  This certificate deliberately excludes device/storage
    addresses so it remains stable when Bernini moves T5 and the transformer
    between CPU and GPU.  Names, module topology, dtype, shape and exact raw
    bytes are all included in the digest.
    """

    try:
        import torch
    except Exception as error:  # pragma: no cover - AUH runtime dependency
        raise ActionFieldCanaryError(
            "strong freeze certificate requires PyTorch"
        ) from error

    if not isinstance(model, torch.nn.Module) or bool(model.training):
        raise ActionFieldCanaryError("frozen model must be one eval torch module")
    modules = [
        (
            str(name),
            f"{type(module).__module__}.{type(module).__qualname__}",
        )
        for name, module in model.named_modules()
    ]
    module_names = [name for name, _ in modules]
    if len(module_names) != len(set(module_names)):
        raise ActionFieldCanaryError("frozen model module names repeat")
    adapter_modules = [
        name
        for name, class_name in modules
        if "lora" in name.lower() or "lora" in class_name.lower()
    ]
    if adapter_modules:
        raise ActionFieldCanaryError("frozen model contains LoRA/adapter modules")

    named_state = [
        *(('parameter', name, value) for name, value in model.named_parameters()),
        *(('buffer', name, value) for name, value in model.named_buffers()),
    ]
    qualified_names = [f"{kind}.{name}" for kind, name, _ in named_state]
    if len(qualified_names) != len(set(qualified_names)):
        raise ActionFieldCanaryError("frozen model state names repeat")
    content = hashlib.sha256()
    metadata_rows: list[dict[str, Any]] = []
    counts = {"parameter": 0, "buffer": 0}
    elements = {"parameter": 0, "buffer": 0}
    byte_counts = {"parameter": 0, "buffer": 0}
    for kind, name, value in named_state:
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise ActionFieldCanaryError(
                f"frozen model {kind} {name} is not materialized"
            )
        if kind == "parameter" and (
            bool(value.requires_grad) or value.grad is not None
        ):
            raise ActionFieldCanaryError(
                f"frozen model parameter {name} is trainable or retains a gradient"
            )
        detached = value.detach().to(device="cpu").contiguous()
        # ``Tensor.view(dtype)`` rejects a zero-dimensional tensor even though
        # scalar buffers are perfectly valid module state.  Flatten first so
        # parameters and buffers of every rank share one exact raw-byte path.
        raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        row = {
            "kind": kind,
            "name": str(name),
            "shape": [int(item) for item in detached.shape],
            "dtype": str(detached.dtype),
            "numel": int(detached.numel()),
            "byte_count": len(raw),
            "raw_storage_sha256": _sha256_bytes(raw),
        }
        encoded = native.legacy.canonical_json_bytes(row)
        content.update(encoded)
        content.update(b"\0")
        content.update(raw)
        counts[kind] += 1
        elements[kind] += int(detached.numel())
        byte_counts[kind] += len(raw)
        metadata_rows.append(row)

    topology_digest = _canonical_digest(modules)
    metadata_digest = _canonical_digest(metadata_rows)
    return {
        "base_frozen": True,
        "model_eval": True,
        "adapter_modules_absent": True,
        "module_count": len(modules),
        "module_topology_sha256": topology_digest,
        "parameter_tensor_count": counts["parameter"],
        "parameter_element_count": elements["parameter"],
        "parameter_byte_count": byte_counts["parameter"],
        "buffer_tensor_count": counts["buffer"],
        "buffer_element_count": elements["buffer"],
        "buffer_byte_count": byte_counts["buffer"],
        "state_metadata_sha256": metadata_digest,
        "state_content_sha256": content.hexdigest(),
        "device_and_storage_address_excluded": True,
        "exact_parameter_and_buffer_bytes_hashed": True,
    }


def _plain_json(path_value: str | Path, *, label: str) -> tuple[Path, Mapping[str, Any]]:
    requested = Path(path_value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise ActionFieldCanaryError(f"{label} path differs")
    path = requested.resolve(strict=True)
    if path != requested or not path.is_file() or path.is_symlink():
        raise ActionFieldCanaryError(f"{label} must be a plain absolute file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ActionFieldCanaryError(f"{label} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ActionFieldCanaryError(f"{label} root differs")
    return path, value


def _registry_cell(
    registry: Mapping[str, Any], *, cell_id: str
) -> Mapping[str, Any]:
    if registry.get("schema_version") != "bernini-self-guided-action-field-core2-v1":
        raise ActionFieldCanaryError("registry schema differs")
    if registry.get("arm_scales") != {name: scale for name, scale in ARM_SCALES}:
        raise ActionFieldCanaryError("registry arm scales differ")
    contract = registry.get("contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("native_guidance_mode") != NATIVE_GUIDANCE_MODE
    ):
        raise ActionFieldCanaryError("registry native guidance mode differs")
    cells = registry.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise ActionFieldCanaryError("registry cells differ")
    matches = [row for row in cells if isinstance(row, Mapping) and row.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise ActionFieldCanaryError("registry cell lookup differs")
    cell = matches[0]
    for key in (
        "source_video",
        "source_video_sha256",
        "source_action_caption",
        "source_action_caption_sha256",
        "target_action_caption",
        "target_action_caption_sha256",
        "seed",
        "bucket_hw",
        "latent_shape",
    ):
        if key not in cell:
            raise ActionFieldCanaryError(f"registry cell lacks {key}")
    if _sha256_text(str(cell["source_action_caption"])) != cell["source_action_caption_sha256"]:
        raise ActionFieldCanaryError("source action caption digest differs")
    if _sha256_text(str(cell["target_action_caption"])) != cell["target_action_caption_sha256"]:
        raise ActionFieldCanaryError("target action caption digest differs")
    if not isinstance(cell["seed"], int) or not 0 <= cell["seed"] < 2**63:
        raise ActionFieldCanaryError("cell seed differs")
    if len(cell["bucket_hw"]) != 2 or len(cell["latent_shape"]) != 5:
        raise ActionFieldCanaryError("cell geometry differs")
    if list(cell["latent_shape"])[0:3] != [1, 16, LATENT_PHASES]:
        raise ActionFieldCanaryError("cell latent phase geometry differs")
    return cell


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--cell-id", choices=("dog", "human"), required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=native.legacy.trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=native.legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=native.legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "expected_registry_sha256",
        "runtime_source_archive_sha256",
        "launcher_source_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            raise ActionFieldCanaryError(f"{name} differs")
    for name in (
        "runtime_source_revision",
        "expected_bernini_commit",
        "expected_veomni_commit",
    ):
        if _SHA1.fullmatch(str(getattr(args, name))) is None:
            raise ActionFieldCanaryError(f"{name} differs")
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise ActionFieldCanaryError("Bernini revision differs")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise ActionFieldCanaryError("VeOmni revision differs")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise ActionFieldCanaryError("checkpoint tree differs")


def _prompt_tokens(tokenizer: Any, prompt: str) -> tuple[Any, Any]:
    return native.legacy._tokenize_training_prompt(tokenizer, prompt)


def _action_field_sampling_contract(*, steps: int, seed: int) -> dict[str, Any]:
    """Pin VR2V task conditioning to Bernini's two-forward APG sampler.

    ``native_sampling_contract("rv2v")`` intentionally selects the separate
    four-forward linear ``rv2v`` guidance mode.  SGAF observes exactly the
    two calls made by ``v2v_apg`` while retaining the full video/reference
    conditions and the VR2V task prompt, so the override must be explicit and
    independently testable.
    """

    contract = native.native_sampling_contract("rv2v", steps=steps, seed=seed)
    if contract.get("guidance_mode") != "rv2v":
        raise ActionFieldCanaryError("native RV2V sampling contract changed")
    contract["guidance_mode"] = NATIVE_GUIDANCE_MODE
    return contract


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    registry_path, registry = _plain_json(args.registry, label="registry")
    if native.legacy.file_sha256(registry_path) != args.expected_registry_sha256:
        raise ActionFieldCanaryError("registry file digest differs")
    cell = _registry_cell(registry, cell_id=args.cell_id)
    output_dir = native._resolve_fresh_output_dir(args.output_dir)
    source_requested = Path(str(cell["source_video"])).expanduser()
    if not source_requested.is_absolute() or source_requested.is_symlink():
        raise ActionFieldCanaryError("source path differs")
    source_path = source_requested.resolve(strict=True)
    if (
        source_path != source_requested
        or not source_path.is_file()
        or native.legacy.file_sha256(source_path) != cell["source_video_sha256"]
    ):
        raise ActionFieldCanaryError("source bytes differ")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise ActionFieldCanaryError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise ActionFieldCanaryError("attention heads do not divide Ulysses4")
    inference_file_hashes = native.legacy.validate_inference_source_files(bernini_root)
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("t2v") != native.TASK_SYSTEM_PROMPTS["t2v"]:
        raise ActionFieldCanaryError("runtime T2V system prompt differs")
    if SYSTEM_PROMPTS.get("vr2v") != native.TASK_SYSTEM_PROMPTS["vr2v"]:
        raise ActionFieldCanaryError("runtime VR2V system prompt differs")
    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise ActionFieldCanaryError("runtime negative prompt differs")
    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != ULYSSES_SIZE
        or distributed.ulysses_size != ULYSSES_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise ActionFieldCanaryError("runtime requires AUH WORLD4/Ulysses4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=ULYSSES_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
        raise ActionFieldCanaryError(f"checkpoint validation failed: {checkpoint_rows[0]}")
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    source_tensor, source_metadata, source_sha = source_audit.prepare_hashed_source_snapshot(
        source_path
    )
    bucket_hw = tuple(int(item) for item in cell["bucket_hw"])
    latent_shape = tuple(int(item) for item in cell["latent_shape"])
    if (
        source_sha != cell["source_video_sha256"]
        or source_metadata.get("frame_count") != FRAME_COUNT
        or tuple(source_metadata.get("source_derived_bucket_hw", ())) != bucket_hw
    ):
        raise ActionFieldCanaryError("source exact81 geometry differs")

    target_caption = str(cell["target_action_caption"])
    source_caption = str(cell["source_action_caption"])
    rv2v_target_prompt = native.build_task_prompt(
        "rv2v", target_caption, prompt_cleaner=prompt_clean
    )
    t2v_target_prompt = native.build_task_prompt(
        "t2v", target_caption, prompt_cleaner=prompt_clean
    )
    t2v_source_prompt = native.build_task_prompt(
        "t2v", source_caption, prompt_cleaner=prompt_clean
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
    )
    rv2v_ids, rv2v_mask = _prompt_tokens(tokenizer, rv2v_target_prompt)
    t2v_target_ids, t2v_target_mask = _prompt_tokens(tokenizer, t2v_target_prompt)
    t2v_source_ids, t2v_source_mask = _prompt_tokens(tokenizer, t2v_source_prompt)
    negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
        tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
        raise ActionFieldCanaryError("renderer scheduler differs")
    model = BerniniRendererModel(config)
    model.eval().requires_grad_(False)
    freeze_before = _strong_model_freeze_certificate(model)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    reference_shape = (1, 16, 1, latent_shape[3], latent_shape[4])
    if distributed.rank == 0:
        source_pixels = source_tensor.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            source_latent = _vae_encode(vae, source_pixels).contiguous()
            references = {
                index: _vae_encode(
                    vae, source_pixels[:, :, index : index + 1].contiguous()
                ).contiguous()
                for index in REFERENCE_INDICES
            }
        del source_pixels
    else:
        source_latent = torch.empty(latent_shape, device=device, dtype=torch.float32)
        references = {
            index: torch.empty(reference_shape, device=device, dtype=torch.float32)
            for index in REFERENCE_INDICES
        }
    dist.broadcast(source_latent, src=0)
    for index in REFERENCE_INDICES:
        dist.broadcast(references[index], src=0)
    if tuple(source_latent.shape) != latent_shape or any(
        tuple(value.shape) != reference_shape for value in references.values()
    ):
        raise ActionFieldCanaryError("source condition geometry differs")
    condition_identities = {
        "source_video": native._all_rank_tensor_identity(
            source_latent, label=f"{args.cell_id}_source_video", world_size=ULYSSES_SIZE
        ),
        "references": {
            str(index): native._all_rank_tensor_identity(
                value,
                label=f"{args.cell_id}_source_reference_{index}",
                world_size=ULYSSES_SIZE,
            )
            for index, value in references.items()
        },
    }
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    model.to(device)
    model.t5_text_encoder.to(device)
    with torch.inference_mode():
        rv2v_target_embeds = model.encode_prompt(
            rv2v_ids.to(device), rv2v_mask.to(device)
        ).detach()
        t2v_target_embeds = model.encode_prompt(
            t2v_target_ids.to(device), t2v_target_mask.to(device)
        ).detach()
        t2v_source_embeds = model.encode_prompt(
            t2v_source_ids.to(device), t2v_source_mask.to(device)
        ).detach()
        uncond_embeds = model.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach()
    model.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()
    diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
    sampler_contract._validate_scheduler_contract(
        diffusion.scheduler, expected_flow_shift=native.FLOW_SHIFT
    )
    if diffusion.transformer_2 is not None:
        raise ActionFieldCanaryError(
            "canary is pinned to the single-expert Bernini-R 1.3B checkpoint"
        )
    target_patch_tokens = LATENT_PHASES * (bucket_hw[0] // 16) * (bucket_hw[1] // 16)

    generated: dict[str, Any] = {}
    generated_identities: dict[str, Any] = {}
    initial_noise: dict[str, Any] = {}
    initial_noise_identities: dict[str, Any] = {}
    action_field_receipts: dict[str, Any] = {}
    with torch.inference_mode():
        for arm, scale in ARM_SCALES:
            patch = None
            if scale is not None:
                patch = NativeRV2VActionFieldPatch(
                    diffusion,
                    target_t2v_embeds=t2v_target_embeds,
                    source_t2v_embeds=t2v_source_embeds,
                    config=ActionFieldConfig(
                        target_patch_tokens=target_patch_tokens,
                        effective_scale=float(scale),
                        target_latent_shape=latent_shape,
                        expected_condition_prefix_tokens=(
                            target_patch_tokens
                            + 4 * (target_patch_tokens // LATENT_PHASES)
                        ),
                        expected_steps=NUM_INFERENCE_STEPS,
                        native_text_guidance_scale=native.OMEGA_TEXT,
                        sigma_zero_below=0.20,
                        sigma_full_above=0.55,
                        maximum_delta_to_native_text_rms=1.50,
                    ),
                )
                patch.install()
            sample_kwargs = {
                "prompt_embeds": rv2v_target_embeds,
                "uncond_prompt_embeds": uncond_embeds,
                "image_vae_latents": None,
                "multi_video_vae_latents": [source_latent],
                "multi_image_vae_latents": [references[index] for index in REFERENCE_INDICES],
                "width": bucket_hw[1],
                "height": bucket_hw[0],
                "device": device,
                **_action_field_sampling_contract(
                    steps=NUM_INFERENCE_STEPS, seed=int(cell["seed"])
                ),
            }
            try:
                result, capture = native._sample_with_native_initial_noise_observer(
                    sample_fn=lambda kw=sample_kwargs: diffusion.sample(**kw),
                    wan_diffusion_module=wan_diffusion,
                    expected_shape=latent_shape,
                    expected_device=device,
                    expected_seed=int(cell["seed"]),
                )
            finally:
                if patch is not None and patch.installed and not patch.restored:
                    patch.restore()
            if patch is not None:
                action_field_receipts[arm] = dict(patch.finalize())
            else:
                action_field_receipts[arm] = {
                    "native_baseline": True,
                    "frozen_t2v_teacher_forwards": 0,
                    "effective_scale": 0.0,
                }
            if (
                not isinstance(result, torch.Tensor)
                or tuple(result.shape) != latent_shape
                or result.dtype != torch.float32
                or result.requires_grad
                or result.grad_fn is not None
                or not bool(torch.isfinite(result).all().item())
            ):
                raise ActionFieldCanaryError("native sampler result differs")
            stored = result.detach().to(device="cpu").contiguous()
            generated[arm] = stored
            generated_identities[arm] = native._all_rank_tensor_identity(
                stored, label=f"{args.cell_id}_{arm}", world_size=ULYSSES_SIZE
            )
            initial_noise[arm] = capture
            initial_noise_identities[arm] = native._all_rank_tensor_identity(
                capture.tensor,
                label=f"{args.cell_id}_{arm}_official_initial_gaussian",
                world_size=ULYSSES_SIZE,
            )

    if len({capture.raw_value_sha256 for capture in initial_noise.values()}) != 1:
        raise ActionFieldCanaryError("arms did not share one official Gaussian")
    local_trace_digest = _canonical_digest(action_field_receipts)
    trace_rows: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(trace_rows, local_trace_digest)
    if len(set(trace_rows)) != 1:
        raise ActionFieldCanaryError("action-field traces differ across SP4 ranks")
    freeze_after = _strong_model_freeze_certificate(model)
    if freeze_after != freeze_before or any(p.requires_grad for p in model.parameters()):
        raise ActionFieldCanaryError("frozen model changed")
    model.to("cpu")
    torch.cuda.empty_cache()

    if distributed.rank == 0:
        output_dir.mkdir(parents=False, exist_ok=False)
        noise_artifacts = {
            arm: native._save_initial_noise_atomically(
                output_dir / f"{arm}.official-initial-gaussian.safetensors",
                initial_noise[arm],
                all_rank_identity=initial_noise_identities[arm],
            )
            for arm, _ in ARM_SCALES
        }
        generated_for_decode = {
            arm: value.to(device=device).contiguous() for arm, value in generated.items()
        }
        try:
            outputs = native._save_outputs(
                output_dir=output_dir,
                generated=generated_for_decode,
                vae=vae,
                bucket_hw=bucket_hw,
                device=device,
                save_output_fn=save_output,
            )
        finally:
            generated_for_decode.clear()
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "cell_id": args.cell_id,
            "input": {
                "source_video": str(source_path),
                "source_video_sha256": source_sha,
                "source_action_caption": source_caption,
                "source_action_caption_sha256": _sha256_text(source_caption),
                "target_action_caption": target_caption,
                "target_action_caption_sha256": _sha256_text(target_caption),
                "mask_track_pose_flow": False,
                "generated_owner_media": False,
                "target_video": False,
            },
            "prompts": {
                "rv2v_target_full_prompt_sha256": _sha256_text(rv2v_target_prompt),
                "t2v_target_full_prompt_sha256": _sha256_text(t2v_target_prompt),
                "t2v_source_full_prompt_sha256": _sha256_text(t2v_source_prompt),
                "teacher_difference": (
                    "same_state_t2v_apg_target_minus_t2v_apg_source"
                ),
                "teacher_negative": "same_target_only_negative_prompt",
            },
            "sampling": {
                "frame_count": FRAME_COUNT,
                "latent_phases": LATENT_PHASES,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_mode": NATIVE_GUIDANCE_MODE,
                "seed": int(cell["seed"]),
                "arms": [name for name, _ in ARM_SCALES],
                "same_official_gaussian_all_arms": True,
                "source_reference_indices": list(REFERENCE_INDICES),
                "source_rich_initial_noise": False,
                "native_target_initialization": native.TARGET_INITIALIZATION,
            },
            "action_field": action_field_receipts,
            "condition_identities": condition_identities,
            "generated_identities": generated_identities,
            "initial_noise_artifacts": noise_artifacts,
            "outputs": outputs,
            "checkpoint": checkpoint_identity,
            "freeze_certificate": freeze_after,
            "source_revisions": {
                "bernini": bernini_revision,
                "veomni": veomni_revision,
                "runtime_method": args.runtime_source_revision,
                "runtime_source_archive_sha256": args.runtime_source_archive_sha256,
                "launcher_source_sha256": args.launcher_source_sha256,
                "inference_files": inference_file_hashes,
            },
            "runtime_versions": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "diffusers": diffusers_version,
                "transformers": transformers_version,
            },
            "training_performed": False,
            "parameter_update": False,
            "prospective_teacher_for_lora_distillation_only": True,
            "scientific_or_action_editing_claim_authorized": False,
        }
        receipt["receipt_digest"] = _canonical_digest(receipt)
        value_audit.write_receipt_atomically(output_dir / "receipt.json", receipt)
        print(native.legacy.canonical_json_bytes(receipt).decode("ascii"), flush=True)

    dist.barrier()
    del source_latent, references, generated, initial_noise
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_SCALES",
    "ActionFieldCanaryError",
    "METHOD",
    "NATIVE_GUIDANCE_MODE",
    "_action_field_sampling_contract",
    "_registry_cell",
    "build_parser",
]
