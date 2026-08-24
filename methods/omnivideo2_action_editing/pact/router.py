"""A compact prompt-conditioned spatiotemporal edit-mask router."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .masks import validate_video_mask


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _Residual3DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.conv(F.silu(self.norm(hidden)))


class PromptConditionedMaskRouter(nn.Module):
    """Predict one edit-support logit per latent voxel.

    Prompt embeddings modulate normalized video features with FiLM. The module
    is deliberately small enough to train alongside a frozen video backbone.
    """

    def __init__(
        self,
        in_channels: int,
        prompt_dim: int,
        *,
        hidden_channels: int = 32,
        depth: int = 2,
    ) -> None:
        super().__init__()
        for name, value in {
            "in_channels": in_channels,
            "prompt_dim": prompt_dim,
            "hidden_channels": hidden_channels,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        self.in_channels = in_channels
        self.prompt_dim = prompt_dim
        self.stem = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.condition_norm = nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
        self.prompt_film = nn.Linear(prompt_dim, 2 * hidden_channels)
        self.blocks = nn.ModuleList(
            _Residual3DBlock(hidden_channels) for _ in range(depth)
        )
        self.output = nn.Conv3d(hidden_channels, 1, kernel_size=1)

    def forward(self, video_features: Tensor, prompt_embedding: Tensor) -> Tensor:
        if not isinstance(video_features, Tensor) or video_features.ndim != 5:
            raise ValueError("video_features must have shape [B, C, T, H, W]")
        if not video_features.is_floating_point():
            raise TypeError("video_features must have a floating dtype")
        if video_features.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} video channels, got {video_features.shape[1]}"
            )
        if not isinstance(prompt_embedding, Tensor) or prompt_embedding.ndim != 2:
            raise ValueError("prompt_embedding must have shape [B, D]")
        if not prompt_embedding.is_floating_point():
            raise TypeError("prompt_embedding must have a floating dtype")
        if prompt_embedding.shape != (video_features.shape[0], self.prompt_dim):
            raise ValueError(
                f"prompt_embedding must have shape [{video_features.shape[0]}, "
                f"{self.prompt_dim}]"
            )
        if prompt_embedding.device != video_features.device:
            raise ValueError("video_features and prompt_embedding must share a device")

        hidden = self.stem(video_features)
        scale, shift = self.prompt_film(prompt_embedding).chunk(2, dim=-1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        hidden = F.silu(self.condition_norm(hidden) * (1.0 + scale) + shift)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def router_loss_components(
    logits: Tensor,
    target_mask: Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    smooth: float = 1.0,
) -> dict[str, Tensor]:
    """Return BCE, soft Dice, and weighted total router losses."""

    if not isinstance(logits, Tensor) or logits.ndim != 5 or logits.shape[1] != 1:
        raise ValueError("logits must have shape [B, 1, T, H, W]")
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating dtype")
    validate_video_mask(
        target_mask,
        name="target_mask",
        batch_size=logits.shape[0],
        frames=logits.shape[2],
        height=logits.shape[3],
        width=logits.shape[4],
    )
    if target_mask.device != logits.device:
        raise ValueError("logits and target_mask must share a device")
    if (
        not math.isfinite(float(bce_weight))
        or not math.isfinite(float(dice_weight))
        or bce_weight < 0
        or dice_weight < 0
    ):
        raise ValueError("loss weights must be finite and non-negative")
    if not math.isfinite(float(smooth)) or smooth <= 0:
        raise ValueError("smooth must be finite and positive")

    target = target_mask.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probabilities = logits.sigmoid().flatten(1)
    flat_target = target.flatten(1)
    intersection = (probabilities * flat_target).sum(dim=1)
    denominator = probabilities.sum(dim=1) + flat_target.sum(dim=1)
    dice = (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()
    total = float(bce_weight) * bce + float(dice_weight) * dice
    return {"total": total, "bce": bce, "dice": dice}


def bce_dice_loss(
    logits: Tensor,
    target_mask: Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    smooth: float = 1.0,
) -> Tensor:
    """Return the scalar weighted BCE + Dice objective."""

    return router_loss_components(
        logits,
        target_mask,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        smooth=smooth,
    )["total"]
