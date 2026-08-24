"""Strict OmniVideo2-1.3B integration for mask-free action fine-tuning.

This module intentionally has no dependency on the legacy PACT mask/tube
stack.  It owns the small amount of upstream glue needed by the action
trainer: loading the exact official transformer, adding action LoRA adapters,
loading the four required learned delimiter entries (whose tensors can span
multiple context rows), and budgeting the *unaltered* source-video condition
before OmniVideo's tight context concatenation.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import math
import pickle
import subprocess
import sys
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .checkpoint_contract import (
    OMNIVIDEO2_1_3B_ACTIVE_SPECIAL_TOKEN_ROWS,
    OMNIVIDEO2_1_3B_SPECIAL_TOKEN_LAYOUT,
    OMNIVIDEO2_1_3B_SPECIAL_TOKENS_SHA256,
    OMNIVIDEO2_1_3B_TRANSFORMER_SHA256,
    OMNIVIDEO2_1_3B_UPSTREAM_REVISION,
)
from pact.lora import (
    LoRALinear,
    expected_lora_module_count,
    inject_lora,
    lora_scope_target_regex,
)


SPECIAL_TOKEN_KEYS = tuple(
    key
    for key, _shape, active in OMNIVIDEO2_1_3B_SPECIAL_TOKEN_LAYOUT
    if active
)


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest without loading the file at once."""

    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_checkpoint_state_dict(
    path: Path,
    dtype: torch.dtype,
    *,
    expected_sha256: str,
) -> Mapping[str, Any]:
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RuntimeError(
            "this PyTorch cannot safely load the official checkpoint with "
            "weights_only=True"
        )
    # Hash and deserialize through the same open file descriptor.  A path swap
    # between two separate opens must not bypass the pinned checkpoint digest.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(
                "official transformer checkpoint digest differs from the pinned "
                "OmniVideo2-1.3B contract"
            )
        handle.seek(0)
        value = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError("official transformer checkpoint must contain a mapping")
    if "module" in value:
        value = value["module"]
    elif "model" in value:
        value = value["model"]
    if not isinstance(value, Mapping) or not value:
        raise ValueError("unwrapped transformer checkpoint is empty or invalid")
    return {
        key: tensor.to(dtype=dtype) if isinstance(tensor, Tensor) else tensor
        for key, tensor in value.items()
    }


def _validate_pinned_upstream_checkout(root: Path) -> None:
    """Require the exact tracked-clean OmniVideo2 source revision."""

    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_changes = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot verify the pinned OmniVideo2 Git checkout") from error
    if revision != OMNIVIDEO2_1_3B_UPSTREAM_REVISION:
        raise ValueError("OmniVideo2 upstream revision differs from checkpoint contract")
    if tracked_changes:
        raise ValueError("OmniVideo2 upstream tracked files are dirty")


def load_official_omnivideo2_1_3b(
    omnivideo_root: str | Path,
    checkpoint_dir: str | Path,
    *,
    max_context_len: int,
    visual_patch_size: Sequence[int] = (1, 4, 4),
    wan_patch_size: Sequence[int] = (1, 2, 2),
) -> tuple[nn.Module, Any, Path]:
    """Load the exact upstream 1.3B architecture and strict base state.

    ``strict=False`` is used only because PyTorch returns a useful mismatch
    object; every missing and unexpected key is then rejected explicitly.
    """

    root = Path(omnivideo_root).expanduser().resolve()
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    required_source = root / "omnivideo" / "modules" / "unified_model.py"
    checkpoint = checkpoint_root / "transformer" / "pytorch_model.pt"
    if not required_source.is_file():
        raise FileNotFoundError(f"not an OmniVideo checkout: {root}")
    _validate_pinned_upstream_checkout(root)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "expected the official transformer at "
            f"<checkpoint-dir>/transformer/pytorch_model.pt: {checkpoint}"
        )
    if checkpoint.is_symlink():
        raise ValueError("official transformer checkpoint must not be a symlink")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from omnivideo.configs import WAN_CONFIGS  # type: ignore[import-not-found]
    from omnivideo.modules.unified_model import (  # type: ignore[import-not-found]
        UnifiedWanWithMixedConditionModel,
    )

    official = copy.deepcopy(WAN_CONFIGS["t2v-1.3B"])
    if tuple(official.visual_context_adapter_patch_size) != tuple(visual_patch_size):
        raise ValueError("visual patch size differs from official OmniVideo2-1.3B")
    if tuple(official.patch_size) != tuple(wan_patch_size):
        raise ValueError("Wan patch size differs from official OmniVideo2-1.3B")
    if int(official.vlm_in_dim) != 2048:
        raise ValueError("official 1.3B VLM input is no longer 2048-D")
    if official.param_dtype != torch.bfloat16:
        raise ValueError(f"official base dtype changed: {official.param_dtype}")

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
        max_context_len=int(max_context_len),
        skip_init=True,
    ).to(official.param_dtype)
    state = _safe_checkpoint_state_dict(
        checkpoint,
        official.param_dtype,
        expected_sha256=OMNIVIDEO2_1_3B_TRANSFORMER_SHA256,
    )
    incompatible = model.load_state_dict(state, strict=False)
    del state
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/model mismatch is forbidden: "
            f"missing={incompatible.missing_keys[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    return model, official, checkpoint


def _load_special_token_payload(
    payload: bytes,
    *,
    expected_sha256: str,
    dtype: torch.dtype,
    device: torch.device,
    unpickler: Any = pickle.loads,
) -> tuple[dict[str, Tensor], int, str]:
    """Digest-check bytes before unpickling, then validate the full layout."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("special_tokens.pkl payload must be non-empty bytes")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            "special_tokens.pkl digest differs from the pinned checkpoint contract"
        )
    # The exact digest makes this otherwise-unsafe upstream pickle trusted.
    value = unpickler(payload)
    if not isinstance(value, dict):
        raise ValueError("special_tokens.pkl must contain a dictionary")
    expected_keys = {key for key, _shape, _active in OMNIVIDEO2_1_3B_SPECIAL_TOKEN_LAYOUT}
    if set(value) != expected_keys:
        raise ValueError(
            "special_tokens.pkl keys differ from the pinned six-entry layout: "
            f"missing={sorted(expected_keys - set(value))}, "
            f"unexpected={sorted(set(value) - expected_keys)}"
        )
    result: dict[str, Tensor] = {}
    rows = 0
    for key, expected_shape, active in OMNIVIDEO2_1_3B_SPECIAL_TOKEN_LAYOUT:
        tensor = value[key]
        if (
            not isinstance(tensor, Tensor)
            or tuple(tensor.shape) != expected_shape
            or tensor.dtype != torch.bfloat16
            or tensor.device.type != "cpu"
        ):
            raise ValueError(
                f"special token {key!r} must be CPU BF16 {expected_shape}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"special token {key!r} contains NaN/Inf")
        if active:
            result[key] = tensor.to(device=device, dtype=dtype)
            rows += int(tensor.shape[0])
    if rows != OMNIVIDEO2_1_3B_ACTIVE_SPECIAL_TOKEN_ROWS:
        raise RuntimeError("internal active special-token row contract is inconsistent")
    return result, rows, digest


def load_special_tokens(
    checkpoint_dir: str | Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
    required: bool,
) -> tuple[dict[str, Tensor] | None, int, str | None]:
    """Load the exact official delimiter tensors after digest/layout checks."""

    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    path = checkpoint_root / "special_tokens.pkl"
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required official special tokens missing: {path}")
        return None, 0, None
    if path.is_symlink():
        raise ValueError("official special_tokens.pkl must not be a symlink")
    return _load_special_token_payload(
        path.read_bytes(),
        expected_sha256=OMNIVIDEO2_1_3B_SPECIAL_TOKENS_SHA256,
        dtype=dtype,
        device=device,
    )


def enable_action_lora(
    model: nn.Module,
    *,
    scope: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> tuple[list[str], list[nn.Parameter]]:
    """Freeze OmniVideo completely, then add only the action LoRA weights.

    In particular, the pretrained visual-context adapter and Qwen projection
    stay frozen.  Non-action inference retains the exact native model simply
    by not loading this separately named action adapter.
    """

    model.requires_grad_(False)
    injected = inject_lora(
        model,
        lora_scope_target_regex(scope),
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        freeze_base=True,
        adapter_dtype=torch.float32,
    )
    expected = expected_lora_module_count(scope, int(model.wan_model.num_layers))
    if len(injected) != expected:
        raise RuntimeError(
            f"closed LoRA scope {scope!r} changed: expected {expected}, got {len(injected)}"
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.dtype != torch.float32 for parameter in trainable):
        raise RuntimeError("action LoRA must have non-empty FP32 master weights")
    forbidden = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and ".lora_A." not in name
        and ".lora_B." not in name
    ]
    if forbidden:
        raise RuntimeError(f"non-LoRA OmniVideo weights became trainable: {forbidden[:20]}")
    return injected, trainable


def set_action_lora_gate(model: nn.Module, gate: Tensor | None) -> int:
    """Install a transient per-sample action gate on every injected LoRA.

    A zero row gives the native frozen renderer path for an explicit isolation
    probe; trainable action, identity, and replay rows use one. The caller must
    not replace/clear the gate between forward and backward when gradient
    checkpointing is enabled.
    """

    if gate is not None:
        if not isinstance(gate, Tensor) or gate.ndim != 1:
            raise ValueError("action LoRA gate must be a rank-1 tensor or None")
        if not gate.is_floating_point() or not bool(torch.isfinite(gate).all()):
            raise ValueError("action LoRA gate must be finite floating point")
        if not bool(((gate == 0.0) | (gate == 1.0)).all()):
            raise ValueError("action LoRA gate values must be exactly zero or one")
        gate = gate.detach()
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.adapter_gate = gate
            count += 1
    if count == 0:
        raise RuntimeError("cannot set action gate before LoRA injection")
    return count


@dataclass(frozen=True)
class SourceBudget:
    original_shape: tuple[int, int, int, int, int]
    output_shape: tuple[int, int, int, int, int]
    nonvisual_tokens: int
    original_visual_tokens: int
    output_visual_tokens: int
    max_context_len: int
    compressed: bool

    @property
    def output_total_tokens(self) -> int:
        return self.nonvisual_tokens + self.output_visual_tokens


@dataclass(frozen=True)
class FullSourceContextBudget:
    """Exact token accounting before Omni can truncate or pad a row."""

    source_shape: tuple[int, int, int, int, int]
    nonvisual_tokens: int
    visual_tokens: int
    total_tokens: int
    budget_tokens: int
    fixed_budget_padding_tokens: int
    fits: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_shape": list(self.source_shape),
            "nonvisual_tokens": self.nonvisual_tokens,
            "visual_tokens": self.visual_tokens,
            "total_tokens": self.total_tokens,
            "budget_tokens": self.budget_tokens,
            "fixed_budget_padding_tokens": self.fixed_budget_padding_tokens,
            "fits": self.fits,
            "source_truncated": False,
        }


class SourceContextBudgetError(ValueError):
    """Raised before model execution when a full source row cannot fit."""

    def __init__(
        self,
        budget: FullSourceContextBudget,
        *,
        sample_id: str | None = None,
        task_type: str | None = None,
    ) -> None:
        self.budget = budget
        self.sample_id = sample_id
        self.task_type = task_type
        identity = (
            f"sample_id={sample_id!r}, task_type={task_type!r}, "
            if sample_id is not None or task_type is not None
            else ""
        )
        super().__init__(
            "full source context exceeds budget; source truncation is forbidden: "
            f"{identity}nonvisual={budget.nonvisual_tokens}, "
            f"visual={budget.visual_tokens}, total={budget.total_tokens}, "
            f"budget={budget.budget_tokens}"
        )


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def full_source_context_budget(
    source_latent: Tensor,
    *,
    max_context_len: int,
    nonvisual_tokens: int,
    visual_patch_size: Sequence[int] = (1, 4, 4),
) -> FullSourceContextBudget:
    """Count a complete source row without selecting, pooling, or truncating it."""

    if not isinstance(source_latent, Tensor) or source_latent.ndim != 5:
        raise ValueError("source_latent must be [B,C,T,H,W]")
    if not source_latent.is_floating_point() or min(source_latent.shape) <= 0:
        raise ValueError("source_latent must be a non-empty floating tensor")
    if source_latent.shape[0] != 1:
        raise ValueError("full source context budgeting is per row and requires B=1")
    patch = tuple(
        _positive_int(item, f"visual_patch_size[{index}]")
        for index, item in enumerate(visual_patch_size)
    )
    if len(patch) != 3 or patch[0] != 1:
        raise ValueError("Omni full-source budgeting requires patch size [1,H,W]")
    max_context_len = _positive_int(max_context_len, "max_context_len")
    if not isinstance(nonvisual_tokens, Integral) or isinstance(nonvisual_tokens, bool):
        raise ValueError("nonvisual_tokens must be an integer")
    nonvisual_tokens = int(nonvisual_tokens)
    if nonvisual_tokens < 0:
        raise ValueError("nonvisual_tokens must be non-negative")
    grid = tuple(int(size) for size in source_latent.shape[2:])
    if any(size % stride for size, stride in zip(grid, patch)):
        raise ValueError(
            f"source latent grid {grid} must divide visual patch {patch}"
        )
    visual_tokens = math.prod(
        size // stride for size, stride in zip(grid, patch)
    )
    total_tokens = nonvisual_tokens + visual_tokens
    return FullSourceContextBudget(
        source_shape=tuple(int(size) for size in source_latent.shape),
        nonvisual_tokens=nonvisual_tokens,
        visual_tokens=visual_tokens,
        total_tokens=total_tokens,
        budget_tokens=max_context_len,
        fixed_budget_padding_tokens=max(0, max_context_len - total_tokens),
        fits=total_tokens <= max_context_len,
    )


def require_full_source_context(
    source_latent: Tensor,
    *,
    max_context_len: int,
    nonvisual_tokens: int,
    visual_patch_size: Sequence[int] = (1, 4, 4),
    sample_id: str | None = None,
    task_type: str | None = None,
) -> FullSourceContextBudget:
    """Return exact accounting or reject the row before Omni sees it."""

    budget = full_source_context_budget(
        source_latent,
        max_context_len=max_context_len,
        nonvisual_tokens=nonvisual_tokens,
        visual_patch_size=visual_patch_size,
    )
    if not budget.fits:
        raise SourceContextBudgetError(
            budget, sample_id=sample_id, task_type=task_type
        )
    return budget


def set_exact_omni_context_length(
    model: nn.Module, *, exact_context_len: int, max_context_len: int
) -> int:
    """Ablation-only alternative to upstream fixed-budget context padding.

    This must not be used for the official-compatible first feasibility run.
    Omni's Wan implementation pads every row to ``wan_model.text_len`` while
    passing ``context_lens=None`` to cross-attention.  With the closed local
    batch size of one, setting that scalar to the exact row length removes all
    artificial padding while retaining ``model.max_context_len`` as the hard
    truncation ceiling.
    """

    exact_context_len = _positive_int(exact_context_len, "exact_context_len")
    max_context_len = _positive_int(max_context_len, "max_context_len")
    if exact_context_len > max_context_len:
        raise ValueError(
            f"exact context {exact_context_len} exceeds budget {max_context_len}"
        )
    if int(getattr(model, "max_context_len", -1)) != max_context_len:
        raise RuntimeError("Omni max_context_len differs from the closed config")
    reset = getattr(model, "reset_wan_text_len", None)
    wan_model = getattr(model, "wan_model", None)
    if not callable(reset) or wan_model is None or not hasattr(wan_model, "text_len"):
        raise RuntimeError("Omni model lacks the exact context-length interface")
    reset(exact_context_len)
    if int(wan_model.text_len) != exact_context_len:
        raise RuntimeError("Omni failed to install the exact context length")
    return exact_context_len


def budget_untouched_source_preserving_first_frame(
    source_latent: Tensor,
    *,
    max_context_len: int,
    nonvisual_tokens: int,
    visual_patch_size: Sequence[int] = (1, 4, 4),
) -> tuple[Tensor, SourceBudget]:
    """Fit complete source conditioning without a mask, erasure, or crop.

    Only temporal average pooling is allowed when the context is too long;
    the exact first latent frame is concatenated back.  Spatial source tokens
    are never selected by actor location.
    """

    if not isinstance(source_latent, Tensor) or source_latent.ndim != 5:
        raise ValueError("source_latent must be [B,C,T,H,W]")
    if not source_latent.is_floating_point() or min(source_latent.shape) <= 0:
        raise ValueError("source_latent must be a non-empty floating tensor")
    patch = tuple(_positive_int(item, f"visual_patch_size[{index}]") for index, item in enumerate(visual_patch_size))
    if len(patch) != 3 or patch[0] != 1:
        raise ValueError("Omni first-frame-safe budgeting requires patch size [1,H,W]")
    max_context_len = _positive_int(max_context_len, "max_context_len")
    if not isinstance(nonvisual_tokens, Integral) or isinstance(nonvisual_tokens, bool):
        raise ValueError("nonvisual_tokens must be an integer")
    nonvisual_tokens = int(nonvisual_tokens)
    if nonvisual_tokens < 0:
        raise ValueError("nonvisual_tokens must be non-negative")
    _, _, frames, height, width = source_latent.shape
    if any(size % stride for size, stride in zip((frames, height, width), patch)):
        raise ValueError(
            f"source latent grid {(frames, height, width)} must divide visual patch {patch}"
        )
    spatial = (height // patch[1]) * (width // patch[2])
    original_visual = frames * spatial
    available = max_context_len - nonvisual_tokens
    output_frames = min(frames, available // spatial)
    if output_frames < 1:
        raise ValueError("context cannot hold even one complete source latent frame")
    if output_frames == frames:
        output = source_latent
    elif output_frames == 1:
        output = source_latent[:, :, :1]
    else:
        tail = F.adaptive_avg_pool3d(
            source_latent[:, :, 1:],
            output_size=(output_frames - 1, height, width),
        )
        output = torch.cat((source_latent[:, :, :1], tail), dim=2)
    if not torch.equal(output[:, :, 0], source_latent[:, :, 0]):
        raise RuntimeError("source first latent frame changed during token budgeting")
    budget = SourceBudget(
        original_shape=tuple(source_latent.shape),
        output_shape=tuple(output.shape),
        nonvisual_tokens=nonvisual_tokens,
        original_visual_tokens=original_visual,
        output_visual_tokens=output_frames * spatial,
        max_context_len=max_context_len,
        compressed=output_frames != frames,
    )
    if budget.output_total_tokens > max_context_len:
        raise RuntimeError("source token budgeting exceeded OmniVideo context")
    return output, budget


def nonvisual_token_counts(
    text_context: Sequence[Tensor],
    source_vlm_context: Sequence[Tensor],
    *,
    motion_plan_tokens: int,
    special_token_count: int,
) -> list[int]:
    """Count Qwen + predicted-plan + T5 + delimiter tokens exactly."""

    if len(text_context) != len(source_vlm_context) or not text_context:
        raise ValueError("text and VLM contexts must be non-empty/equal length")
    motion_plan_tokens = _positive_int(motion_plan_tokens, "motion_plan_tokens")
    if not isinstance(special_token_count, int) or special_token_count < 0:
        raise ValueError("special_token_count must be a non-negative integer")
    counts: list[int] = []
    for index, (text, vlm) in enumerate(zip(text_context, source_vlm_context)):
        if not isinstance(text, Tensor) or text.ndim != 2 or text.shape[1] != 4096:
            raise ValueError(f"text_context[{index}] must be [L,4096]")
        if not isinstance(vlm, Tensor) or vlm.ndim != 2 or vlm.shape[1] != 2048:
            raise ValueError(f"source_vlm_context[{index}] must be [L,2048]")
        counts.append(
            int(text.shape[0] + vlm.shape[0] + motion_plan_tokens + special_token_count)
        )
    return counts


def wan_sequence_length(
    latent: Tensor | Sequence[int], patch_size: Sequence[int] = (1, 2, 2)
) -> int:
    """Return exact no-padding Wan patch count for a latent batch."""

    shape = tuple(latent.shape) if isinstance(latent, Tensor) else tuple(latent)
    if len(shape) != 5 or any(not isinstance(item, int) or item <= 0 for item in shape):
        raise ValueError("latent shape must be positive [B,C,T,H,W]")
    patch = tuple(patch_size)
    if len(patch) != 3 or any(not isinstance(item, int) or item <= 0 for item in patch):
        raise ValueError("patch_size must contain three positive integers")
    grid = shape[2:]
    if any(size % stride for size, stride in zip(grid, patch)):
        raise ValueError(f"latent grid {grid} is not divisible by Wan patch {patch}")
    return math.prod(size // stride for size, stride in zip(grid, patch))
