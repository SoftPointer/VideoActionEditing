"""Dependency-free ``nn.Linear`` LoRA injection and adapter serialization."""

from __future__ import annotations

import math
import os
import re
import inspect
from pathlib import Path
from typing import Any, Iterator, Mapping, Pattern

import torch
from torch import Tensor, nn


LORA_SCOPE_PATTERNS = {
    "cross_qo": (
        r"^wan_model\.blocks\.\d+\.cross_attn\.(?:q|o)$"
    ),
    "all_attn": (
        r"^wan_model\.blocks\.\d+\.(?:self_attn|cross_attn)\.(?:q|k|v|o)$"
    ),
    "diffsynth_full": (
        r"^wan_model\.blocks\.\d+\."
        r"(?:(?:self_attn|cross_attn)\.(?:q|k|v|o)|ffn\.(?:0|2))$"
    ),
}

LORA_SCOPE_MODULES_PER_BLOCK = {
    "cross_qo": 2,
    "all_attn": 8,
    "diffsynth_full": 10,
}


def lora_scope_target_regex(scope: str) -> str:
    """Return the closed OmniVideo2/Wan LoRA target regex for ``scope``."""

    if not isinstance(scope, str) or scope not in LORA_SCOPE_PATTERNS:
        raise ValueError(
            f"lora scope must be one of {sorted(LORA_SCOPE_PATTERNS)}, got {scope!r}"
        )
    return LORA_SCOPE_PATTERNS[scope]


def expected_lora_module_count(scope: str, num_wan_blocks: int) -> int:
    """Return the exact module count expected for a closed Wan LoRA scope."""

    lora_scope_target_regex(scope)
    if (
        not isinstance(num_wan_blocks, int)
        or isinstance(num_wan_blocks, bool)
        or num_wan_blocks <= 0
    ):
        raise ValueError("num_wan_blocks must be a positive integer")
    return LORA_SCOPE_MODULES_PER_BLOCK[scope] * num_wan_blocks


class LoRALinear(nn.Module):
    """Wrap an existing linear layer with a zero-initialized low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
        freeze_base: bool = True,
        adapter_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be nn.Linear")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if alpha is None:
            alpha = float(rank)
        if not math.isfinite(float(alpha)) or float(alpha) <= 0:
            raise ValueError("alpha must be finite and positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout_p = float(dropout)
        self.dropout = nn.Dropout(self.dropout_p) if dropout else nn.Identity()
        if adapter_dtype is None:
            adapter_dtype = base.weight.dtype
        if not isinstance(adapter_dtype, torch.dtype) or not adapter_dtype.is_floating_point:
            raise TypeError("adapter_dtype must be a floating torch.dtype")
        factory_kwargs = {"device": base.weight.device, "dtype": adapter_dtype}
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, **factory_kwargs)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, **factory_kwargs)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # Optional per-sample adapter gate.  It is deliberately a transient
        # plain attribute (not a parameter/buffer), so old checkpoints remain
        # byte-for-byte compatible.  ``None`` preserves the historical LoRA
        # behavior.  The mask-free action trainer sets a detached [B] gate and
        # keeps it installed through gradient-checkpoint recomputation.
        self.adapter_gate: Tensor | None = None
        if freeze_base:
            self.base.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: Tensor) -> Tensor:
        base_output = self.base(inputs)
        adapter_inputs = self.dropout(inputs).to(dtype=self.lora_A.weight.dtype)
        update = self.lora_B(self.lora_A(adapter_inputs))
        if self.adapter_gate is not None:
            gate = self.adapter_gate.to(device=update.device, dtype=update.dtype)
            if gate.ndim != 1 or update.ndim < 2 or gate.shape[0] != update.shape[0]:
                raise RuntimeError(
                    "LoRA adapter_gate must have shape [B] matching the first "
                    "dimension of the selected Wan activation"
                )
            gate = gate.reshape(gate.shape[0], *([1] * (update.ndim - 1)))
            update = update * gate
        return base_output + self.scaling * update.to(dtype=base_output.dtype)


def _replace_child(parent: nn.Module, child_name: str, replacement: nn.Module) -> None:
    if isinstance(parent, nn.ModuleList) and child_name.isdigit():
        parent[int(child_name)] = replacement
    elif isinstance(parent, nn.ModuleDict):
        parent[child_name] = replacement
    else:
        setattr(parent, child_name, replacement)


def inject_lora(
    model: nn.Module,
    target_regex: str | Pattern[str],
    *,
    rank: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
    freeze_base: bool = True,
    adapter_dtype: torch.dtype | None = None,
) -> list[str]:
    """Replace regex-selected linear submodules and return their full names.

    Regex matching uses :func:`re.Pattern.search` on fully qualified module
    names. Existing ``LoRALinear`` subtrees are skipped, making repeated calls
    safe from recursively wrapping adapter projections.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module")
    pattern = re.compile(target_regex) if isinstance(target_regex, str) else target_regex
    if not hasattr(pattern, "search"):
        raise TypeError("target_regex must be a string or compiled regex")

    existing_prefixes = [
        name for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]
    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if any(name.startswith(f"{prefix}.") for prefix in existing_prefixes):
            continue
        if pattern.search(name):
            candidates.append((name, module))
    if not candidates:
        raise ValueError(f"target_regex {pattern.pattern!r} matched no nn.Linear modules")

    injected: list[str] = []
    for name, module in candidates:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = LoRALinear(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            freeze_base=freeze_base,
            adapter_dtype=adapter_dtype,
        )
        _replace_child(parent, child_name, replacement)
        injected.append(name)
    return injected


def iter_lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return detached CPU copies of adapter weights only."""

    state: dict[str, Tensor] = {}
    for name, module in iter_lora_modules(model):
        prefix = f"{name}." if name else ""
        state[f"{prefix}lora_A.weight"] = module.lora_A.weight.detach().cpu().clone()
        state[f"{prefix}lora_B.weight"] = module.lora_B.weight.detach().cpu().clone()
    return state


def lora_config(model: nn.Module) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "rank": module.rank,
            "alpha": module.alpha,
            "dropout": module.dropout_p,
            "in_features": module.in_features,
            "out_features": module.out_features,
        }
        for name, module in iter_lora_modules(model)
    }


def load_lora_state_dict(
    model: nn.Module, saved_state: Mapping[str, Any]
) -> list[str]:
    """Strictly copy an adapter-only state mapping into injected modules."""

    if not isinstance(saved_state, Mapping):
        raise ValueError("LoRA state_dict must be a mapping")
    expected: dict[str, Tensor] = {}
    for name, module in iter_lora_modules(model):
        prefix = f"{name}." if name else ""
        expected[f"{prefix}lora_A.weight"] = module.lora_A.weight
        expected[f"{prefix}lora_B.weight"] = module.lora_B.weight
    if not expected:
        raise ValueError("model contains no injected LoRA modules")
    if set(saved_state) != set(expected):
        missing = sorted(set(expected) - set(saved_state))
        unexpected = sorted(set(saved_state) - set(expected))
        raise ValueError(
            f"LoRA state keys differ: missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for key, destination in expected.items():
            source = saved_state[key]
            if not isinstance(source, Tensor) or source.shape != destination.shape:
                raise ValueError(f"LoRA tensor shape differs for {key}")
            destination.copy_(
                source.to(device=destination.device, dtype=destination.dtype)
            )
    return [name for name, _ in iter_lora_modules(model)]


def _checkpoint_digest(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def save_lora_weights(
    model: nn.Module,
    path: str | os.PathLike[str],
    *,
    base_checkpoint_sha256: str,
) -> Path:
    """Save adapter-only weights and enough configuration to audit them."""

    state = lora_state_dict(model)
    if not state:
        raise ValueError("model contains no LoRA modules")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: Mapping[str, Any] = {
        "format": "pact-linear-lora-v1",
        "base_checkpoint_sha256": _checkpoint_digest(
            base_checkpoint_sha256, name="base_checkpoint_sha256"
        ),
        "config": lora_config(model),
        "state_dict": state,
    }
    torch.save(payload, destination)
    return destination


def load_lora_weights(
    model: nn.Module,
    path: str | os.PathLike[str],
    *,
    expected_base_checkpoint_sha256: str,
) -> list[str]:
    """Strictly restore adapters into already-injected ``LoRALinear`` modules.

    The caller must first construct the same base model and inject LoRA using
    the saved module configuration. Every config/state key and tensor shape is
    consumed exactly once; partial or surplus checkpoints are rejected.
    """

    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError(
            "this PyTorch lacks safe weights_only loading; upgrade PyTorch"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format") != "pact-linear-lora-v1":
        raise ValueError("unsupported LoRA checkpoint format")
    expected_digest = _checkpoint_digest(
        expected_base_checkpoint_sha256,
        name="expected_base_checkpoint_sha256",
    )
    if payload.get("base_checkpoint_sha256") != expected_digest:
        raise ValueError("LoRA checkpoint was built for a different base checkpoint")
    saved_config = payload.get("config")
    saved_state = payload.get("state_dict")
    current_config = lora_config(model)
    if not isinstance(saved_config, Mapping) or dict(saved_config) != current_config:
        raise ValueError("LoRA checkpoint config does not match injected modules")
    loaded = load_lora_state_dict(model, saved_state)
    if loaded != list(current_config):
        raise RuntimeError("internal error: loaded LoRA module order changed")
    return loaded


save_lora = save_lora_weights
