#!/usr/bin/env python3
"""Phase-query source+instruction planner for SPT-v2.

The planner deliberately has only two semantic inputs: the complete clean
source latent and the complete *unpadded* contextual T5 token sequence.  It
does not accept a target, mask, track, flow, pose, or trajectory.

Twenty-one learned queries, each augmented with a fixed sinusoidal timestamp,
cross-attend to every instruction token in two consecutive attention blocks.
The resulting per-phase states FiLM-modulate a position-aware source volume
before dense offset and preserve/transport/generate heads are evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:
    from . import phase_transport as spt
except ImportError:  # direct execution/import from the spt_v2 directory
    import phase_transport as spt  # type: ignore


ARCHITECTURE_NAME = "phase_query_v2"
CROSS_ATTENTION_LAYERS = 2
POSITION_CHANNELS = 3


class PhaseQueryPlannerError(spt.PhaseTransportError):
    """Raised when the phase-query planner contract is violated."""


@dataclass(frozen=True)
class PhaseQueryPlannerConfig:
    architecture: str = ARCHITECTURE_NAME
    latent_channels: int = 64
    text_channels: int = 4096
    hidden_channels: int = 128
    attention_heads: int = 8
    cross_attention_layers: int = CROSS_ATTENTION_LAYERS
    feedforward_multiplier: int = 4
    latent_phases: int = spt.LATENT_PHASES
    max_temporal_offset: float = 2.0
    max_spatial_offset: float = 4.0
    source_bank_detach: bool = True

    def validate(self) -> None:
        if self.architecture != ARCHITECTURE_NAME:
            raise PhaseQueryPlannerError(
                f"planner architecture must be {ARCHITECTURE_NAME!r}"
            )
        if self.latent_phases != spt.LATENT_PHASES:
            raise PhaseQueryPlannerError("phase-query planner requires exactly 21 phases")
        if self.cross_attention_layers != CROSS_ATTENTION_LAYERS:
            raise PhaseQueryPlannerError("phase-query planner requires exactly two cross-attention layers")
        for name in (
            "latent_channels",
            "text_channels",
            "hidden_channels",
            "attention_heads",
            "feedforward_multiplier",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise PhaseQueryPlannerError(f"{name} must be a positive integer")
        if self.hidden_channels % self.attention_heads:
            raise PhaseQueryPlannerError("hidden_channels must be divisible by attention_heads")
        groups = min(32, self.hidden_channels)
        if self.hidden_channels % groups:
            raise PhaseQueryPlannerError(
                "hidden_channels must be divisible by its GroupNorm group count"
            )
        for name in ("max_temporal_offset", "max_spatial_offset"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise PhaseQueryPlannerError(f"{name} must be finite and positive")
        if type(self.source_bank_detach) is not bool:
            raise PhaseQueryPlannerError("source_bank_detach must be boolean")


def _axis_coordinates(length: int, *, device: Any) -> Any:
    import torch

    if type(length) is not int or length <= 0:
        raise PhaseQueryPlannerError("coordinate axis length must be positive")
    if length == 1:
        return torch.zeros(1, device=device, dtype=torch.float32)
    return torch.linspace(-1.0, 1.0, length, device=device, dtype=torch.float32)


def normalized_position_channels(source: Any) -> Any:
    """Return explicit ``(t,y,x)`` channels in ``[-1,1]`` as ``[B,3,T,H,W]``."""

    spt._validate_video(source, label="source")
    import torch

    batch, phases, height, width, _ = map(int, source.shape)
    t = _axis_coordinates(phases, device=source.device)
    y = _axis_coordinates(height, device=source.device)
    x = _axis_coordinates(width, device=source.device)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    return (
        torch.stack((tt, yy, xx), dim=0)
        .unsqueeze(0)
        .expand(batch, -1, -1, -1, -1)
    )


def sinusoidal_phase_encoding(phases: int, channels: int) -> Any:
    """Build a fixed, explicit timestamp encoding for the latent phases."""

    import torch
    import torch.nn.functional as functional

    if type(phases) is not int or phases <= 0 or type(channels) is not int or channels <= 0:
        raise PhaseQueryPlannerError("phase encoding shape must be positive")
    half = max(channels // 2, 1)
    positions = torch.arange(phases, dtype=torch.float32).unsqueeze(1)
    denominator = max(half - 1, 1)
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / denominator
    ).unsqueeze(0)
    encoding = torch.cat((torch.sin(positions * frequencies), torch.cos(positions * frequencies)), dim=1)
    if int(encoding.shape[1]) < channels:
        encoding = functional.pad(encoding, (0, channels - int(encoding.shape[1])))
    return encoding[:, :channels].contiguous()


try:
    import torch
    from torch import nn

    class PhaseTextCrossAttentionBlock(nn.Module):
        """One phase-query to full-token cross-attention/MLP residual block."""

        def __init__(self, hidden_channels: int, attention_heads: int, multiplier: int):
            super().__init__()
            self.query_norm = nn.LayerNorm(hidden_channels)
            self.token_norm = nn.LayerNorm(hidden_channels)
            self.cross_attention = nn.MultiheadAttention(
                hidden_channels,
                attention_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.ff_norm = nn.LayerNorm(hidden_channels)
            self.feedforward = nn.Sequential(
                nn.Linear(hidden_channels, multiplier * hidden_channels),
                nn.GELU(approximate="tanh"),
                nn.Linear(multiplier * hidden_channels, hidden_channels),
            )

        def forward(self, phase_states: Any, instruction_tokens: Any) -> Any:
            tokens = self.token_norm(instruction_tokens)
            attended, _ = self.cross_attention(
                self.query_norm(phase_states),
                tokens,
                tokens,
                need_weights=False,
            )
            phase_states = phase_states + attended
            return phase_states + self.feedforward(self.ff_norm(phase_states))


    class PhaseQueryPlanner(nn.Module):
        """Predict a dense phase transport plan from source + instruction tokens."""

        architecture = ARCHITECTURE_NAME

        def __init__(self, config: PhaseQueryPlannerConfig):
            super().__init__()
            config.validate()
            self.config = config
            hidden = config.hidden_channels
            self.source_in = nn.Conv3d(
                config.latent_channels + POSITION_CHANNELS,
                hidden,
                kernel_size=1,
            )
            self.text_in = nn.Linear(config.text_channels, hidden)
            self.phase_queries = nn.Parameter(torch.empty(config.latent_phases, hidden))
            nn.init.normal_(self.phase_queries, mean=0.0, std=hidden ** -0.5)
            self.register_buffer(
                "phase_time_encoding",
                sinusoidal_phase_encoding(config.latent_phases, hidden),
                persistent=True,
            )
            self.cross_attention_blocks = nn.ModuleList(
                PhaseTextCrossAttentionBlock(
                    hidden,
                    config.attention_heads,
                    config.feedforward_multiplier,
                )
                for _ in range(config.cross_attention_layers)
            )
            self.phase_film = nn.Linear(hidden, 2 * hidden)
            self.body = nn.Sequential(
                nn.GroupNorm(min(32, hidden), hidden),
                nn.SiLU(),
                nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.offset_head = nn.Conv3d(hidden, 3, kernel_size=1)
            self.gate_head = nn.Conv3d(hidden, 3, kernel_size=1)
            # Exact zero transport at initialization.  The zero gate kernel
            # makes the spatial prediction independent of random features,
            # while the bias starts overwhelmingly on source preservation.
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)
            nn.init.zeros_(self.gate_head.weight)
            with torch.no_grad():
                self.gate_head.bias.copy_(torch.tensor((4.0, -2.0, -4.0)))

        def _validate_instruction_tokens(self, source: Any, instruction_tokens: Any) -> None:
            if getattr(instruction_tokens, "ndim", None) != 3:
                raise PhaseQueryPlannerError(
                    "instruction_tokens must be the full unpadded [B,L,text_channels] sequence"
                )
            if int(instruction_tokens.shape[0]) != int(source.shape[0]):
                raise PhaseQueryPlannerError("source/instruction batch sizes differ")
            if int(instruction_tokens.shape[1]) <= 0:
                raise PhaseQueryPlannerError("instruction token sequence is empty")
            if int(instruction_tokens.shape[2]) != self.config.text_channels:
                raise PhaseQueryPlannerError("instruction token channel count differs")
            if not bool(torch.isfinite(instruction_tokens).all()):
                raise PhaseQueryPlannerError("instruction tokens contain non-finite values")
            # The pinned Bernini T5 path represents padded rows as exact zeros.
            # Reject them here so accidental global/padded conditioning cannot
            # silently re-enter through a different caller.
            if bool((instruction_tokens.float().abs().sum(dim=-1) == 0).any()):
                raise PhaseQueryPlannerError("instruction_tokens contains padded zero rows")

        def forward(self, source: Any, instruction_tokens: Any) -> spt.PhasePlan:
            spt._validate_video(source, label="source")
            if int(source.shape[-1]) != self.config.latent_channels:
                raise PhaseQueryPlannerError("student source channel count differs")
            self._validate_instruction_tokens(source, instruction_tokens)
            bank = source.detach() if self.config.source_bank_detach else source
            coordinates = normalized_position_channels(bank)
            source_channels = bank.permute(0, 4, 1, 2, 3).float()
            hidden = self.source_in(torch.cat((source_channels, coordinates), dim=1))

            tokens = self.text_in(instruction_tokens.float())
            batch = int(source.shape[0])
            phase_states = (
                self.phase_queries + self.phase_time_encoding
            ).unsqueeze(0).expand(batch, -1, -1)
            for block in self.cross_attention_blocks:
                phase_states = block(phase_states, tokens)
            scale, shift = self.phase_film(phase_states).chunk(2, dim=-1)
            scale = scale.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            shift = shift.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            hidden = hidden * (1.0 + scale) + shift
            hidden = self.body(hidden)

            raw_offsets = torch.tanh(self.offset_head(hidden))
            limits = raw_offsets.new_tensor(
                (
                    self.config.max_temporal_offset,
                    self.config.max_spatial_offset,
                    self.config.max_spatial_offset,
                )
            ).view(1, 3, 1, 1, 1)
            offsets = raw_offsets * limits
            gates = torch.softmax(self.gate_head(hidden), dim=1)
            plan = spt.PhasePlan(offsets=offsets, gate_probs=gates, provenance="student")
            plan.validate(source)
            return plan

except ImportError:  # pragma: no cover - contract-only local environments

    class PhaseTextCrossAttentionBlock:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise PhaseQueryPlannerError("PhaseQueryPlanner requires PyTorch")


    class PhaseQueryPlanner:  # type: ignore[no-redef]
        architecture = ARCHITECTURE_NAME

        def __init__(self, config: PhaseQueryPlannerConfig):
            config.validate()
            raise PhaseQueryPlannerError("PhaseQueryPlanner requires PyTorch")

        def forward(self, source: Any, instruction_tokens: Any) -> spt.PhasePlan:
            raise PhaseQueryPlannerError("PhaseQueryPlanner requires PyTorch")
