#!/usr/bin/env python3
"""Single-GPU adapter training entry for OmniVideo2-1.3B + PACT.

All expensive encoders are offline: this entry consumes digest-bound
``AtomicLatentDataset`` payloads and trains only LoRA, the prompt router, and
explicitly enabled conditioning adapters.  It never serializes base weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import logging
import os
import pickle
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.utils.checkpoint  # noqa: F401 -- required by the upstream Wan module
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pact.dataset import AtomicLatentDataset, collate_atomic_latents
from pact.lora import (
    expected_lora_module_count,
    inject_lora,
    lora_scope_target_regex,
    lora_state_dict,
)
from pact.router import PromptConditionedMaskRouter
from pact.training import (
    DiffSynthWanTrainingScheduler,
    budget_source_condition_preserving_first_frame,
    nonvisual_token_counts,
    pact_training_losses,
    prepare_pact_flow_batch,
    validate_training_config,
    wan_sequence_length,
)


LOGGER = logging.getLogger("pact.train")
SPECIAL_TOKEN_KEYS = ("<img_st>", "<img_ed>", "<prp_st>", "<prp_ed>")
DIFFSYNTH_REFERENCE_REVISION = "ab12bf4119b7c9a23ff3359eefb41ba54a658ccb"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--payload-root",
        type=Path,
        default=None,
        help="Allowed root for payload-relative paths (default: manifest parent).",
    )
    parser.add_argument(
        "--omnivideo-root",
        type=Path,
        default=Path("../Omni-Video"),
        help="Clean upstream SAIS-FUXI/Omni-Video checkout.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Fudan-FUXI/OmniVideo2-1.3B directory; training loads transformer/pytorch_model.pt.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--dry-run-contract",
        action="store_true",
        help="Validate a CPU subset without importing OmniVideo or creating output.",
    )
    parser.add_argument(
        "--dry-run-samples",
        type=int,
        default=1,
        help="Payload count for dry-run; 0 validates every payload.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return validate_training_config(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_dataset(args: argparse.Namespace) -> AtomicLatentDataset:
    manifest = args.manifest.expanduser().resolve()
    payload_root = (
        args.payload_root.expanduser().resolve() if args.payload_root is not None else None
    )
    return AtomicLatentDataset(
        manifest,
        payload_root=payload_root,
        require_training_authorized=True,
        require_payload_digest=True,
        verify_payload_digest=True,
    )


def _to_batch(sample: Mapping[str, Any]) -> dict[str, Any]:
    batch = collate_atomic_latents([sample])
    if batch["source_latent"].shape[1] != 16:
        raise ValueError("OmniVideo2-1.3B requires 16-channel VAE latents")
    return batch


def _dry_run(
    args: argparse.Namespace, config: Mapping[str, Any], dataset: AtomicLatentDataset
) -> None:
    if args.dry_run_samples < 0:
        raise ValueError("--dry-run-samples must be non-negative")
    requested = len(dataset) if args.dry_run_samples == 0 else args.dry_run_samples
    count = min(len(dataset), requested)
    if count == 0:
        raise ValueError("dry-run selected no payloads")
    summaries: list[dict[str, Any]] = []
    model_cfg = config["model"]
    mask_cfg = config["mask"]
    for index in range(count):
        batch = _to_batch(dataset[index])
        source = batch["source_latent"].float()
        target = batch["global_target_latent"].float()
        source_mask = batch["source_component_mask"].float()
        target_mask = batch["target_component_mask"].float()
        sigma = torch.full((1,), 0.5)
        prepared = prepare_pact_flow_batch(
            source,
            target,
            source_mask,
            target_mask,
            sigma,
            dilation_radius=mask_cfg["dilation_radius"],
            feather_radius=mask_cfg["feather_radius"],
            source_erasure_mode=model_cfg["source_erasure_mode"],
            noise=torch.zeros_like(source),
        )
        # Contract-only mode deliberately does not open the checkpoint. The
        # checked-in model contract assumes the official four one-row tokens;
        # real training below counts the serialized tensors' actual rows.
        special_count = 4 if model_cfg["require_special_tokens"] else 0
        token_counts = nonvisual_token_counts(
            batch["text_context"],
            batch["vlm_context"],
            special_token_count=special_count,
        )
        budgeted, budget = budget_source_condition_preserving_first_frame(
            prepared.source_condition,
            max_context_len=model_cfg["max_context_len"],
            nonvisual_tokens=max(token_counts),
            visual_patch_size=model_cfg["visual_patch_size"],
        )
        seq_len = wan_sequence_length(prepared.x_t, model_cfg["wan_patch_size"])
        router = PromptConditionedMaskRouter(
            in_channels=source.shape[1],
            prompt_dim=2048,
            hidden_channels=config["router"]["hidden_channels"],
            depth=config["router"]["depth"],
        )
        prompt = torch.stack([item.float().mean(dim=0) for item in batch["vlm_context"]])
        router_logits = router(source, prompt)
        losses = pact_training_losses(
            prepared.local_velocity,
            prepared,
            router_logits=router_logits,
            router_target_mask=prepared.source_erase_mask,
            weights=config["loss_weights"],
        )
        if not bool(torch.isfinite(losses["total"])):
            raise ValueError(f"non-finite dry-run loss for atom {batch['atom_id'][0]}")
        summaries.append(
            {
                "atom_id": batch["atom_id"][0],
                "latent_shape": list(source.shape),
                "wan_sequence_length": seq_len,
                "nonvisual_tokens": token_counts[0],
                "source_visual_tokens_before": budget.original_visual_tokens,
                "source_visual_tokens_after": budget.output_visual_tokens,
                "source_condition_frames_after": budgeted.shape[2],
                "ideal_velocity_loss": float(
                    losses["velocity_edit"] + losses["velocity_preserve"]
                ),
                "finite_total_with_random_router": float(losses["total"]),
            }
        )
    print(
        json.dumps(
            {
                "status": "contract-ok",
                "dataset_rows_digest_verified": len(dataset),
                "payloads_fully_validated": count,
                "samples": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _require_single_gpu(device_index: int) -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError("this entry is deliberately single-process; WORLD_SIZE must equal 1")
    if not isinstance(device_index, int) or device_index < 0:
        raise ValueError("--device must be a non-negative CUDA index")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OmniVideo2-1.3B training")
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device_index} does not exist; found {torch.cuda.device_count()} device(s)"
        )
    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")


def _checkpoint_state_dict(path: Path, dtype: torch.dtype) -> Mapping[str, Any]:
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError(
            "this PyTorch version cannot safely load the checkpoint with weights_only=True"
        )
    load_kwargs["weights_only"] = True
    value = torch.load(path, **load_kwargs)
    if not isinstance(value, Mapping):
        raise ValueError("official transformer checkpoint must contain a mapping")
    if "module" in value:
        value = value["module"]
    elif "model" in value:
        value = value["model"]
    if not isinstance(value, Mapping) or not value:
        raise ValueError("unwrapped transformer checkpoint is not a non-empty state dict")
    return {
        key: tensor.to(dtype) if isinstance(tensor, Tensor) else tensor
        for key, tensor in value.items()
    }


def _load_official_model(
    omnivideo_root: Path,
    checkpoint_dir: Path,
    config: Mapping[str, Any],
) -> tuple[nn.Module, Any, Path]:
    root = omnivideo_root.expanduser().resolve()
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    required_source = root / "omnivideo" / "modules" / "unified_model.py"
    checkpoint = checkpoint_dir / "transformer" / "pytorch_model.pt"
    if not required_source.is_file():
        raise FileNotFoundError(f"not an OmniVideo checkout: {root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "the only accepted base checkpoint path is "
            f"<checkpoint-dir>/transformer/pytorch_model.pt; missing {checkpoint}"
        )
    sys.path.insert(0, str(root))
    from omnivideo.configs import WAN_CONFIGS  # type: ignore[import-not-found]
    from omnivideo.modules.unified_model import (  # type: ignore[import-not-found]
        UnifiedWanWithMixedConditionModel,
    )

    official = copy.deepcopy(WAN_CONFIGS["t2v-1.3B"])
    expected_visual_patch = tuple(config["model"]["visual_patch_size"])
    expected_wan_patch = tuple(config["model"]["wan_patch_size"])
    if tuple(official.visual_context_adapter_patch_size) != expected_visual_patch:
        raise ValueError("config visual patch size differs from official OmniVideo2-1.3B")
    if tuple(official.patch_size) != expected_wan_patch:
        raise ValueError("config Wan patch size differs from official OmniVideo2-1.3B")
    if int(official.vlm_in_dim) != 2048:
        raise ValueError("official 1.3B config no longer has the expected 2048-D VLM input")

    model = UnifiedWanWithMixedConditionModel(
        wan_config=official,
        vlm_in_dim=official.vlm_in_dim,
        precision_dtype=official.param_dtype,
        device_id="cpu",
        rank=0,
        dit_fsdp=False,
        use_usp=False,
        use_visual_context_adapter=official.use_visual_context_adapter,
        visual_context_adapter_patch_size=official.visual_context_adapter_patch_size,
        max_context_len=config["model"]["max_context_len"],
        skip_init=True,
    ).to(official.param_dtype)
    state = _checkpoint_state_dict(checkpoint, official.param_dtype)
    incompatible = model.load_state_dict(state, strict=False)
    del state
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/model mismatch is forbidden: "
            f"missing={incompatible.missing_keys[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    return model, official, checkpoint


def _load_special_tokens(
    checkpoint_dir: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
    required: bool,
) -> tuple[dict[str, Tensor] | None, int]:
    path = checkpoint_dir.expanduser().resolve() / "special_tokens.pkl"
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required official special-token file is missing: {path}")
        return None, 0
    # This mirrors the official loader. Only use a checkpoint directory from a
    # trusted source: pickle is not a safe interchange format.
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError("special_tokens.pkl must contain a dictionary")
    missing = set(SPECIAL_TOKEN_KEYS) - set(value)
    if missing:
        raise ValueError(f"special_tokens.pkl lacks required keys: {sorted(missing)}")
    result: dict[str, Tensor] = {}
    token_rows = 0
    for key in SPECIAL_TOKEN_KEYS:
        tensor = value[key]
        if not isinstance(tensor, Tensor) or tensor.ndim not in {1, 2} or tensor.shape[-1] != 4096:
            raise ValueError(f"special token {key!r} must have shape [4096] or [L, 4096]")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"special token {key!r} contains NaN or Inf")
        result[key] = tensor.to(device=device, dtype=dtype)
        token_rows += 1 if tensor.ndim == 1 else int(tensor.shape[0])
    return result, token_rows


def _enable_adapters(
    model: nn.Module, config: Mapping[str, Any]
) -> tuple[list[str], list[nn.Parameter]]:
    model.requires_grad_(False)
    lora_cfg = config["lora"]
    injected = inject_lora(
        model,
        lora_scope_target_regex(lora_cfg["scope"]),
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg["dropout"],
        freeze_base=True,
        adapter_dtype=torch.float32,
    )
    expected = expected_lora_module_count(
        lora_cfg["scope"], int(model.wan_model.num_layers)
    )
    if len(injected) != expected:
        raise RuntimeError(
            f"LoRA scope {lora_cfg['scope']!r} count changed: "
            f"expected {expected}, got {len(injected)}"
        )
    model_cfg = config["model"]
    if model_cfg["train_visual_adapter"]:
        if model.visual_context_adapter is None:
            raise RuntimeError("official model has no visual_context_adapter")
        model.visual_context_adapter.float()
        model.visual_context_adapter.requires_grad_(True)
    if model_cfg["train_vlm_projection"]:
        model.vlm_norm.float()
        model.vlm_proj.float()
        model.vlm_norm.requires_grad_(True)
        model.vlm_proj.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no model adapter parameters are trainable")
    return injected, parameters


def _cpu_module_state(module: nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def _adapter_gradient_groups(
    model: nn.Module, router: nn.Module
) -> dict[str, list[nn.Parameter]]:
    """Partition every trainable tensor for connectivity diagnostics."""

    groups: dict[str, list[nn.Parameter]] = {
        "lora": [],
        "visual_adapter": [],
        "vlm_projection": [],
        "router": [parameter for parameter in router.parameters() if parameter.requires_grad],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lora_A." in name or ".lora_B." in name:
            group = "lora"
        elif name.startswith("visual_context_adapter."):
            group = "visual_adapter"
        elif name.startswith("vlm_norm.") or name.startswith("vlm_proj."):
            group = "vlm_projection"
        else:
            raise RuntimeError(f"unclassified trainable model parameter: {name}")
        groups[group].append(parameter)
    empty = [name for name, parameters in groups.items() if not parameters]
    if empty:
        raise RuntimeError(f"empty required trainable gradient groups: {empty}")
    return groups


def _finite_gradient_stats(
    groups: Mapping[str, list[nn.Parameter]],
) -> dict[str, dict[str, float | int]]:
    """Fail on disconnected/non-finite groups and return pre-clip L2 norms."""

    result: dict[str, dict[str, float | int]] = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters]
        missing = sum(gradient is None for gradient in gradients)
        if missing:
            raise RuntimeError(
                f"gradient group {name!r} has {missing} disconnected parameter tensor(s)"
            )
        present = [gradient for gradient in gradients if gradient is not None]
        if any(not bool(torch.isfinite(gradient).all()) for gradient in present):
            raise FloatingPointError(f"gradient group {name!r} contains NaN or Inf")
        squared_norm = torch.stack(
            [gradient.detach().float().square().sum() for gradient in present]
        ).sum()
        result[name] = {
            "parameter_tensors": len(parameters),
            "parameter_elements": sum(parameter.numel() for parameter in parameters),
            "pre_clip_l2_norm": float(torch.sqrt(squared_norm)),
        }
    return result


def _save_adapter_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    router: nn.Module,
    config: Mapping[str, Any],
    injected: list[str],
    step: int,
    config_sha256: str,
    base_checkpoint_sha256: str,
    manifest_sha256: str,
    special_tokens_sha256: str | None,
    encoder_contract_sha256: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    payload: dict[str, Any] = {
        "format": "pact-omnivideo2-adapters-v2",
        "step": step,
        "config_sha256": config_sha256,
        "validated_config": dict(config),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "special_tokens_sha256": special_tokens_sha256,
        "encoder_contract_sha256": encoder_contract_sha256,
        "lora_modules": injected,
        "lora_state_dict": lora_state_dict(model),
        "router_state_dict": _cpu_module_state(router),
    }
    if config["model"]["train_visual_adapter"]:
        payload["visual_context_adapter_state_dict"] = _cpu_module_state(
            model.visual_context_adapter
        )
    if config["model"]["train_vlm_projection"]:
        payload["vlm_norm_state_dict"] = _cpu_module_state(model.vlm_norm)
        payload["vlm_proj_state_dict"] = _cpu_module_state(model.vlm_proj)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary checkpoint exists: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _create_output(
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    args: argparse.Namespace,
    config_sha256: str,
    checkpoint_sha256: str,
    manifest_sha256: str,
    special_tokens_sha256: str | None,
    encoder_contract_sha256: str,
) -> Path:
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    run = {
        "format": "pact-omnivideo2-run-v2",
        "config": config,
        "config_sha256": config_sha256,
        "manifest": str(args.manifest.expanduser().resolve()),
        "manifest_sha256": manifest_sha256,
        "payload_root": str(args.payload_root.expanduser().resolve()) if args.payload_root else None,
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "special_tokens_sha256": special_tokens_sha256,
        "encoder_contract_sha256": encoder_contract_sha256,
        "diffsynth_reference_revision": DIFFSYNTH_REFERENCE_REVISION,
        "flow_master_dtype": "float32",
        "trainable_master_dtype": "float32",
        "base_model_dtype": "bfloat16",
        "base_weights_saved": False,
        "single_gpu": args.device,
    }
    (destination / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def _train(
    args: argparse.Namespace, config: Mapping[str, Any], dataset: AtomicLatentDataset
) -> None:
    if args.checkpoint_dir is None or args.output_dir is None:
        raise ValueError("training requires --checkpoint-dir and --output-dir")
    device = _require_single_gpu(args.device)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    first = _to_batch(dataset[0])
    encoder_contract_digest = first["encoder_contract_sha256"]
    model, official, checkpoint = _load_official_model(
        args.omnivideo_root, args.checkpoint_dir, config
    )
    model_dtype = official.param_dtype
    if model_dtype != torch.bfloat16:
        raise RuntimeError(f"expected official bfloat16 parameters, got {model_dtype}")
    injected, model_parameters = _enable_adapters(model, config)
    model.to(device)
    special_tokens, special_count = _load_special_tokens(
        args.checkpoint_dir,
        dtype=model_dtype,
        device=device,
        required=config["model"]["require_special_tokens"],
    )
    router = PromptConditionedMaskRouter(
        in_channels=first["source_latent"].shape[1],
        prompt_dim=2048,
        hidden_channels=config["router"]["hidden_channels"],
        depth=config["router"]["depth"],
    ).to(device=device, dtype=torch.float32)
    if config["model"]["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    model.train()
    router.train()

    config_path = args.config.expanduser().resolve()
    config_digest = _sha256(config_path)
    checkpoint_digest = _sha256(checkpoint)
    manifest_digest = _sha256(args.manifest.expanduser().resolve())
    special_tokens_path = args.checkpoint_dir.expanduser().resolve() / "special_tokens.pkl"
    special_tokens_digest = (
        _sha256(special_tokens_path) if special_tokens_path.is_file() else None
    )
    output_dir = _create_output(
        args.output_dir,
        config=config,
        args=args,
        config_sha256=config_digest,
        checkpoint_sha256=checkpoint_digest,
        manifest_sha256=manifest_digest,
        special_tokens_sha256=special_tokens_digest,
        encoder_contract_sha256=encoder_contract_digest,
    )
    LOGGER.info("create-only output: %s", output_dir)
    LOGGER.info("trainable model adapter parameters: %d", sum(p.numel() for p in model_parameters))
    LOGGER.info("trainable router parameters: %d", sum(p.numel() for p in router.parameters()))

    train_cfg = config["training"]
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_atomic_latents,
        generator=loader_generator,
        pin_memory=True,
        drop_last=False,
    )
    optimizer_cfg = config["optimizer"]
    trainable = model_parameters + list(router.parameters())
    non_fp32_trainable = [
        parameter.dtype for parameter in trainable if parameter.dtype != torch.float32
    ]
    if non_fp32_trainable:
        raise RuntimeError(
            "all trainable adapters/router parameters must retain FP32 master "
            f"weights; found dtypes={sorted({str(item) for item in non_fp32_trainable})}"
        )
    gradient_groups = _adapter_gradient_groups(model, router)
    optimizer_groups = [
        {
            "params": gradient_groups["lora"] + gradient_groups["router"],
            "lr": optimizer_cfg["learning_rate"],
            "name": "lora_router",
        },
        {
            "params": gradient_groups["visual_adapter"]
            + gradient_groups["vlm_projection"],
            "lr": optimizer_cfg["pretrained_adapter_learning_rate"],
            "name": "pretrained_condition_adapters",
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(optimizer_cfg["beta1"], optimizer_cfg["beta2"]),
        eps=optimizer_cfg["eps"],
        weight_decay=optimizer_cfg["weight_decay"],
    )
    optimizer.zero_grad(set_to_none=True)
    log_path = output_dir / "metrics.jsonl"
    global_step = 0
    max_steps = int(train_cfg["max_steps"])
    accumulation = int(train_cfg["gradient_accumulation_steps"])
    stop = False
    start_time = time.monotonic()
    model_cfg = config["model"]
    mask_cfg = config["mask"]
    flow_scheduler = DiffSynthWanTrainingScheduler(shift=config["flow"]["shift"])
    if flow_scheduler.num_train_timesteps != config["flow"]["num_train_timesteps"]:
        raise RuntimeError("validated flow timestep count differs from scheduler")
    flow_generator = torch.Generator(device="cpu").manual_seed(seed)

    for epoch in range(int(train_cfg["epochs"])):
        for batch_index, batch in enumerate(loader):
            if batch["encoder_contract_sha256"] != encoder_contract_digest:
                raise ValueError(
                    "training manifest mixes incompatible offline encoder contracts"
                )
            # Keep the complete flow path in FP32. Casting x0/noise/sigma to
            # BF16 before interpolation makes the actual x_t disagree with the
            # FP32 timestep sent to Wan and leaves a non-zero oracle x0 loss.
            source = batch["source_latent"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            target = batch["global_target_latent"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            if source.shape[1] != int(model.wan_model.in_dim):
                raise ValueError("payload latent channel count differs from official Wan model")
            source_mask = batch["source_component_mask"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            target_mask = batch["target_component_mask"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            flow_sample = flow_scheduler.sample(
                source.shape[0],
                generator=flow_generator,
                device=device,
                dtype=torch.float32,
            )
            sigma = flow_sample.sigma
            prepared = prepare_pact_flow_batch(
                source,
                target,
                source_mask,
                target_mask,
                sigma,
                dilation_radius=mask_cfg["dilation_radius"],
                feather_radius=mask_cfg["feather_radius"],
                source_erasure_mode=model_cfg["source_erasure_mode"],
            )
            text_context = [item.to(device=device, dtype=model_dtype) for item in batch["text_context"]]
            vlm_context = [item.to(device=device, dtype=model_dtype) for item in batch["vlm_context"]]
            token_counts = nonvisual_token_counts(
                text_context, vlm_context, special_token_count=special_count
            )
            source_condition, budget = budget_source_condition_preserving_first_frame(
                prepared.source_condition,
                max_context_len=model_cfg["max_context_len"],
                nonvisual_tokens=max(token_counts),
                visual_patch_size=model_cfg["visual_patch_size"],
            )
            if budget.output_total_tokens > model_cfg["max_context_len"]:
                raise RuntimeError("source visual condition would be silently truncated")
            seq_len = wan_sequence_length(prepared.x_t, model_cfg["wan_patch_size"])
            prompt = torch.stack([item.float().mean(dim=0) for item in vlm_context])
            router_logits = router(source.float(), prompt)
            timestep = flow_sample.timestep
            with torch.amp.autocast("cuda", dtype=model_dtype):
                outputs = model(
                    list(prepared.x_t.to(dtype=model_dtype).unbind(0)),
                    t=timestep,
                    context=text_context,
                    ar_vision_input=vlm_context,
                    visual_emb=source_condition.to(dtype=model_dtype),
                    seq_len=seq_len,
                    special_token_dict=special_tokens,
                    classifier_free_ratio=0.0,
                    condition_mode="full",
                )
            if not isinstance(outputs, list) or len(outputs) != source.shape[0]:
                raise RuntimeError("OmniVideo model returned an unexpected batch structure")
            prediction = torch.stack(outputs, dim=0).float()
            if prediction.shape != source.shape:
                raise RuntimeError(
                    f"OmniVideo output shape {tuple(prediction.shape)} differs from latent {tuple(source.shape)}"
                )
            losses = pact_training_losses(
                prediction,
                prepared,
                router_logits=router_logits,
                router_target_mask=prepared.source_erase_mask.float(),
                weights=config["loss_weights"],
                flow_weight=flow_sample.flow_weight,
            )
            if not bool(torch.isfinite(losses["total"]).detach()):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch} batch={batch_index} atoms={batch['atom_id']}"
                )

            remaining = len(loader) - (batch_index // accumulation) * accumulation
            window_size = min(accumulation, remaining)
            (losses["total"] / window_size).backward()
            end_window = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if not end_window:
                continue
            gradient_stats = _finite_gradient_stats(gradient_groups)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=float(optimizer_cfg["max_grad_norm"])
            )
            if not bool(torch.isfinite(grad_norm).detach()):
                raise FloatingPointError("non-finite adapter gradient norm")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            record = {
                "step": global_step,
                "epoch": epoch,
                "batch": batch_index,
                "atom_ids": batch["atom_id"],
                "loss": {key: float(value.detach()) for key, value in losses.items()},
                "grad_norm": float(grad_norm.detach()),
                "gradient_groups": gradient_stats,
                "timestep_id": flow_sample.timestep_id,
                "timestep_mean": float(timestep.mean()),
                "sigma_mean": float(sigma.mean()),
                "flow_training_weight": float(flow_sample.flow_weight),
                "learning_rates": {
                    str(group["name"]): float(group["lr"])
                    for group in optimizer.param_groups
                },
                "source_visual_tokens": budget.output_visual_tokens,
                "elapsed_seconds": time.monotonic() - start_time,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if global_step % int(train_cfg["log_every"]) == 0:
                LOGGER.info("step=%d total=%.6f", global_step, record["loss"]["total"])
            if global_step % int(train_cfg["checkpoint_every"]) == 0:
                _save_adapter_checkpoint(
                    output_dir / f"adapters_step_{global_step:08d}.pt",
                    model=model,
                    router=router,
                    config=config,
                    injected=injected,
                    step=global_step,
                    config_sha256=config_digest,
                    base_checkpoint_sha256=checkpoint_digest,
                    manifest_sha256=manifest_digest,
                    special_tokens_sha256=special_tokens_digest,
                    encoder_contract_sha256=encoder_contract_digest,
                )
            if max_steps and global_step >= max_steps:
                stop = True
                break
        if stop:
            break
    if global_step == 0:
        raise RuntimeError("training produced zero optimizer steps")
    _save_adapter_checkpoint(
        output_dir / f"adapters_final_step_{global_step:08d}.pt",
        model=model,
        router=router,
        config=config,
        injected=injected,
        step=global_step,
        config_sha256=config_digest,
        base_checkpoint_sha256=checkpoint_digest,
        manifest_sha256=manifest_digest,
        special_tokens_sha256=special_tokens_digest,
        encoder_contract_sha256=encoder_contract_digest,
    )
    final_checkpoint = output_dir / f"adapters_final_step_{global_step:08d}.pt"
    receipt = {
        "format": "pact-omnivideo2-training-done-v2",
        "optimizer_steps": global_step,
        "final_adapter_checkpoint": final_checkpoint.name,
        "final_adapter_sha256": _sha256(final_checkpoint),
        "config_sha256": config_digest,
        "manifest_sha256": manifest_digest,
        "base_checkpoint_sha256": checkpoint_digest,
        "special_tokens_sha256": special_tokens_digest,
        "encoder_contract_sha256": encoder_contract_digest,
        "diffsynth_reference_revision": DIFFSYNTH_REFERENCE_REVISION,
        "flow_master_dtype": "float32",
        "trainable_master_dtype": "float32",
        "base_weights_saved": False,
        "lora_module_count": len(injected),
        "trainable_model_adapter_parameters": sum(
            parameter.numel() for parameter in model_parameters
        ),
        "trainable_router_parameters": sum(
            parameter.numel() for parameter in router.parameters()
        ),
        "elapsed_seconds": time.monotonic() - start_time,
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "accelerator_name": torch.cuda.get_device_name(device),
        "accelerator_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(
            device
        ),
        "accelerator_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(
            device
        ),
    }
    (output_dir / "done.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("finished at optimizer step %d", global_step)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    config = _load_config(args.config)
    dataset = _make_dataset(args)
    if args.dry_run_contract:
        _dry_run(args, config, dataset)
        return
    _train(args, config, dataset)


if __name__ == "__main__":
    main()
