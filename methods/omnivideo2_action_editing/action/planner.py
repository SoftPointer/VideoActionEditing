"""Source-conditioned temporal motion-plan prediction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MOTION_TOKEN_DIM, VLM_DIM


def _positive_int(value: int, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _source_batch(
    source_vlm_context: Tensor | Sequence[Tensor], *, input_dim: int
) -> tuple[Tensor, Tensor | None]:
    """Create a padded batch while keeping padding internal to the planner."""

    if isinstance(source_vlm_context, Tensor):
        if source_vlm_context.ndim == 2:
            source_vlm_context = source_vlm_context.unsqueeze(0)
        if source_vlm_context.ndim != 3:
            raise ValueError(
                "source_vlm_context must have shape [B, L, D] or [L, D]"
            )
        if source_vlm_context.shape[-1] != input_dim:
            raise ValueError(
                f"source_vlm_context last dimension must be {input_dim}"
            )
        if min(source_vlm_context.shape) <= 0:
            raise ValueError("source_vlm_context cannot have an empty dimension")
        if not source_vlm_context.is_floating_point():
            raise TypeError("source_vlm_context must have a floating dtype")
        return source_vlm_context, None

    if not isinstance(source_vlm_context, Sequence) or not source_vlm_context:
        raise TypeError(
            "source_vlm_context must be a tensor or a non-empty tensor sequence"
        )
    contexts = list(source_vlm_context)
    for index, context in enumerate(contexts):
        if not isinstance(context, Tensor):
            raise TypeError(f"source_vlm_context[{index}] must be a tensor")
        if context.ndim != 2 or context.shape[-1] != input_dim:
            raise ValueError(
                f"source_vlm_context[{index}] must have shape [L, {input_dim}]"
            )
        if context.shape[0] <= 0:
            raise ValueError(f"source_vlm_context[{index}] cannot be empty")
        if not context.is_floating_point():
            raise TypeError(f"source_vlm_context[{index}] must be floating point")
        if context.device != contexts[0].device or context.dtype != contexts[0].dtype:
            raise ValueError("all source VLM contexts must share device and dtype")

    batch_size = len(contexts)
    max_length = max(context.shape[0] for context in contexts)
    batch = contexts[0].new_zeros((batch_size, max_length, input_dim))
    padding = torch.ones(
        (batch_size, max_length), dtype=torch.bool, device=contexts[0].device
    )
    for index, context in enumerate(contexts):
        length = context.shape[0]
        batch[index, :length] = context
        padding[index, :length] = False
    return batch, padding


class _MotionPlanLayer(nn.Module):
    def __init__(
        self, hidden_dim: int, *, num_heads: int, mlp_ratio: float, dropout: float
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.source_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        feedforward_dim = max(hidden_dim, int(round(hidden_dim * mlp_ratio)))
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, queries: Tensor, source: Tensor, padding: Tensor | None
    ) -> Tensor:
        normalized_source = self.source_norm(source)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            normalized_source,
            normalized_source,
            key_padding_mask=padding,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class TemporalMotionPlanPredictor(nn.Module):
    """Predict ``K`` 2048-wide motion tokens from source VLM features only.

    Ground-truth motion tokens are intentionally absent from ``forward``. They
    are privileged training labels consumed only by :func:`motion_plan_loss`,
    so training and inference use the same planner inputs. The supplied source
    context is expected to have been encoded jointly from the source video and
    edit instruction by OmniVideo2's VLM path.
    """

    def __init__(
        self,
        num_tokens: int,
        *,
        input_dim: int = VLM_DIM,
        hidden_dim: int = 512,
        depth: int = 2,
        output_dim: int = MOTION_TOKEN_DIM,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tokens = _positive_int(num_tokens, name="num_tokens")
        self.input_dim = _positive_int(input_dim, name="input_dim")
        self.hidden_dim = _positive_int(hidden_dim, name="hidden_dim")
        self.output_dim = _positive_int(output_dim, name="output_dim")
        depth = _positive_int(depth, name="depth")
        num_heads = _positive_int(num_heads, name="num_heads")
        if self.input_dim != VLM_DIM:
            raise ValueError(f"input_dim must be exactly {VLM_DIM}")
        if self.output_dim != MOTION_TOKEN_DIM:
            raise ValueError(f"output_dim must be exactly {MOTION_TOKEN_DIM}")
        if self.hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not isinstance(mlp_ratio, (int, float)) or isinstance(mlp_ratio, bool):
            raise TypeError("mlp_ratio must be a number")
        mlp_ratio = float(mlp_ratio)
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be finite and positive")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise TypeError("dropout must be a number")
        dropout = float(dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.source_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.motion_queries = nn.Parameter(
            torch.empty(self.num_tokens, self.hidden_dim)
        )
        self.layers = nn.ModuleList(
            _MotionPlanLayer(
                self.hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.normal_(self.motion_queries, mean=0.0, std=0.02)

    def forward(self, source_vlm_context: Tensor | Sequence[Tensor]) -> Tensor:
        source, padding = _source_batch(
            source_vlm_context, input_dim=self.input_dim
        )
        source = self.source_projection(source)
        queries = self.motion_queries.to(dtype=source.dtype).unsqueeze(0).expand(
            source.shape[0], -1, -1
        )
        for layer in self.layers:
            queries = layer(queries, source, padding)
        return self.output_projection(self.output_norm(queries))


def motion_plan_loss(
    predicted_motion_tokens: Tensor,
    target_motion_tokens: Tensor,
    *,
    cosine_weight: float = 0.1,
) -> Tensor:
    """Distill offline target motion tokens without feeding them to the model."""

    for value, name in (
        (predicted_motion_tokens, "predicted_motion_tokens"),
        (target_motion_tokens, "target_motion_tokens"),
    ):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 3 or value.shape[-1] != MOTION_TOKEN_DIM:
            raise ValueError(f"{name} must have shape [B, K, {MOTION_TOKEN_DIM}]")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating dtype")
    if predicted_motion_tokens.shape != target_motion_tokens.shape:
        raise ValueError("predicted and target motion token shapes must match")
    if predicted_motion_tokens.device != target_motion_tokens.device:
        raise ValueError("predicted and target motion tokens must share a device")
    if not isinstance(cosine_weight, (int, float)) or isinstance(
        cosine_weight, bool
    ):
        raise TypeError("cosine_weight must be a number")
    cosine_weight = float(cosine_weight)
    if not math.isfinite(cosine_weight) or cosine_weight < 0.0:
        raise ValueError("cosine_weight must be finite and non-negative")

    target = target_motion_tokens.detach().to(dtype=predicted_motion_tokens.dtype)
    regression = F.smooth_l1_loss(predicted_motion_tokens.float(), target.float())
    if cosine_weight == 0.0:
        return regression
    cosine = 1.0 - F.cosine_similarity(
        predicted_motion_tokens.float(), target.float(), dim=-1, eps=1e-8
    )
    return regression + cosine_weight * cosine.mean()


__all__ = ["TemporalMotionPlanPredictor", "motion_plan_loss"]
