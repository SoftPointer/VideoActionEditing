"""Strict loader for adapter-only PACT/OmniVideo2 training checkpoints."""

from __future__ import annotations

import hashlib
import inspect
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .lora import (
    inject_lora,
    iter_lora_modules,
    load_lora_state_dict,
    lora_config,
    lora_scope_target_regex,
)
from .router import PromptConditionedMaskRouter
from .training import validate_training_config


class AdapterCheckpointError(ValueError):
    """Raised when an adapter bundle cannot be restored exactly."""


@dataclass(frozen=True)
class LoadedPactAdapters:
    router: PromptConditionedMaskRouter
    config: dict[str, Any]
    step: int
    lora_modules: tuple[str, ...]
    checkpoint_sha256: str
    manifest_sha256: str
    special_tokens_sha256: str | None
    encoder_contract_sha256: str


_UNSET = object()


def _digest(value: Any, *, name: str, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AdapterCheckpointError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _load_same_bytes(path: Path) -> tuple[Mapping[str, Any], str]:
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise AdapterCheckpointError(
            "this PyTorch lacks safe weights_only loading; upgrade PyTorch"
        )
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise AdapterCheckpointError("adapter checkpoint must contain a mapping")
    return value, digest


def _strict_module_load(module: nn.Module, state: Any, *, name: str) -> None:
    if not isinstance(state, Mapping):
        raise AdapterCheckpointError(f"{name} state must be a mapping")
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise AdapterCheckpointError(f"{name} state does not match: {exc}") from exc


def load_pact_adapter_bundle(
    model: nn.Module,
    path: str | Path,
    *,
    expected_base_checkpoint_sha256: str,
    expected_manifest_sha256: str | None = None,
    expected_special_tokens_sha256: str | None | object = _UNSET,
    expected_encoder_contract_sha256: str | None = None,
) -> LoadedPactAdapters:
    """Inject and restore every trainable PACT module into an official base.

    ``expected_base_checkpoint_sha256`` is mandatory so an adapter can never be
    silently applied to a different OmniVideo2 checkpoint. Manifest and special
    token identities can additionally be enforced for exact experiment replay.
    The function rejects unknown fields, including accidental serialized base
    weights.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module")
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise AdapterCheckpointError(
            f"adapter checkpoint must be a regular non-symlink file: {checkpoint_path}"
        )
    payload, checkpoint_digest = _load_same_bytes(checkpoint_path)
    if payload.get("format") != "pact-omnivideo2-adapters-v2":
        raise AdapterCheckpointError("unsupported PACT adapter checkpoint format")

    base_digest = _digest(
        expected_base_checkpoint_sha256,
        name="expected_base_checkpoint_sha256",
    )
    if payload.get("base_checkpoint_sha256") != base_digest:
        raise AdapterCheckpointError(
            "adapter bundle was trained from a different base checkpoint"
        )
    manifest_digest = _digest(payload.get("manifest_sha256"), name="manifest_sha256")
    special_digest = _digest(
        payload.get("special_tokens_sha256"),
        name="special_tokens_sha256",
        allow_none=True,
    )
    encoder_contract_digest = _digest(
        payload.get("encoder_contract_sha256"),
        name="encoder_contract_sha256",
    )
    _digest(payload.get("config_sha256"), name="config_sha256")
    if expected_manifest_sha256 is not None:
        expected_manifest = _digest(
            expected_manifest_sha256, name="expected_manifest_sha256"
        )
        if manifest_digest != expected_manifest:
            raise AdapterCheckpointError("adapter manifest digest differs")
    if expected_special_tokens_sha256 is not _UNSET:
        expected_special = _digest(
            expected_special_tokens_sha256,
            name="expected_special_tokens_sha256",
            allow_none=True,
        )
        if special_digest != expected_special:
            raise AdapterCheckpointError("adapter special-token digest differs")
    if expected_encoder_contract_sha256 is not None:
        expected_encoder_contract = _digest(
            expected_encoder_contract_sha256,
            name="expected_encoder_contract_sha256",
        )
        if encoder_contract_digest != expected_encoder_contract:
            raise AdapterCheckpointError("adapter offline encoder contract differs")

    try:
        config = validate_training_config(payload.get("validated_config"))
    except (TypeError, ValueError) as exc:
        raise AdapterCheckpointError(f"invalid embedded training config: {exc}") from exc
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise AdapterCheckpointError("adapter step must be a positive integer")

    common_fields = {
        "format",
        "step",
        "config_sha256",
        "validated_config",
        "base_checkpoint_sha256",
        "manifest_sha256",
        "special_tokens_sha256",
        "encoder_contract_sha256",
        "lora_modules",
        "lora_state_dict",
        "router_state_dict",
    }
    expected_fields = set(common_fields)
    if config["model"]["train_visual_adapter"]:
        expected_fields.add("visual_context_adapter_state_dict")
    if config["model"]["train_vlm_projection"]:
        expected_fields.update({"vlm_norm_state_dict", "vlm_proj_state_dict"})
    if set(payload) != expected_fields:
        raise AdapterCheckpointError(
            "adapter fields differ: "
            f"missing={sorted(expected_fields - set(payload))}, "
            f"unknown={sorted(set(payload) - expected_fields)}"
        )

    saved_names = payload.get("lora_modules")
    if (
        not isinstance(saved_names, list)
        or not saved_names
        or any(not isinstance(name, str) or not name for name in saved_names)
        or len(set(saved_names)) != len(saved_names)
    ):
        raise AdapterCheckpointError("lora_modules must be a non-empty unique list")
    existing_names = [name for name, _ in iter_lora_modules(model)]
    lora_cfg = config["lora"]
    if existing_names:
        if existing_names != saved_names:
            raise AdapterCheckpointError("existing injected LoRA modules differ")
    else:
        model.requires_grad_(False)
        injected = inject_lora(
            model,
            lora_scope_target_regex(lora_cfg["scope"]),
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            freeze_base=True,
            adapter_dtype=torch.float32,
        )
        if injected != saved_names:
            raise AdapterCheckpointError(
                "embedded LoRA module list differs from the current base model"
            )
    for name, module_config in lora_config(model).items():
        if (
            name not in saved_names
            or module_config["rank"] != lora_cfg["rank"]
            or module_config["alpha"] != float(lora_cfg["alpha"])
            or module_config["dropout"] != float(lora_cfg["dropout"])
        ):
            raise AdapterCheckpointError("injected LoRA configuration differs")
    try:
        loaded_names = load_lora_state_dict(model, payload["lora_state_dict"])
    except ValueError as exc:
        raise AdapterCheckpointError(str(exc)) from exc
    if loaded_names != saved_names:
        raise AdapterCheckpointError("loaded LoRA module order differs")

    try:
        in_channels = int(model.wan_model.in_dim)
        device = next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError, ValueError) as exc:
        raise AdapterCheckpointError("model lacks the expected Wan attributes") from exc
    router = PromptConditionedMaskRouter(
        in_channels=in_channels,
        prompt_dim=2048,
        hidden_channels=config["router"]["hidden_channels"],
        depth=config["router"]["depth"],
    ).to(device=device, dtype=torch.float32)
    _strict_module_load(router, payload["router_state_dict"], name="router")

    if config["model"]["train_visual_adapter"]:
        visual_adapter = getattr(model, "visual_context_adapter", None)
        if not isinstance(visual_adapter, nn.Module):
            raise AdapterCheckpointError("base model lacks visual_context_adapter")
        # Training keeps every mutable conditioning adapter as an FP32 master
        # module even though the frozen Wan base is BF16.  Upcast the clean
        # base module before loading so ``load_state_dict`` cannot silently
        # round an FP32 adapter checkpoint back to BF16.
        visual_adapter.float()
        _strict_module_load(
            visual_adapter,
            payload["visual_context_adapter_state_dict"],
            name="visual_context_adapter",
        )
    if config["model"]["train_vlm_projection"]:
        for attribute, field in (
            ("vlm_norm", "vlm_norm_state_dict"),
            ("vlm_proj", "vlm_proj_state_dict"),
        ):
            module = getattr(model, attribute, None)
            if not isinstance(module, nn.Module):
                raise AdapterCheckpointError(f"base model lacks {attribute}")
            module.float()
            _strict_module_load(module, payload[field], name=attribute)

    return LoadedPactAdapters(
        router=router,
        config=config,
        step=step,
        lora_modules=tuple(saved_names),
        checkpoint_sha256=checkpoint_digest,
        manifest_sha256=manifest_digest,
        special_tokens_sha256=special_digest,
        encoder_contract_sha256=encoder_contract_digest,
    )
