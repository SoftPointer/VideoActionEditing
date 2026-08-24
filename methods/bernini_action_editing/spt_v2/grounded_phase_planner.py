#!/usr/bin/env python3
"""Source-grounded cell/phase planner for SPT-v3.

The deployable API remains deliberately weak: a clean source-video latent and
the complete unpadded instruction-token sequence.  Paired targets, masks,
tracks, optical flow, pose, and trajectories are not accepted.  Unlike the
global-FiLM v2 planner, every coarse source cell queries the instruction and
grounded edit slots, then all 21 latent phases communicate through axial
temporal attention before a source-skipped coarse-to-fine decoder predicts a
plan.

Two constraints are architectural rather than regularization-only:

* preserve/transport/generate gates are factorized into change and conditional
  novelty, with rejected novelty mass returned to preserve so every sample and
  every phase has soft Generate mass <= 0.12;
* transport is selected from the same 5 x 5 x 5 integer candidate lattice as
  the hardened paired oracle, using a source-bank correlation volume plus a
  learned residual.  Straight-through selection executes an exact candidate
  while retaining categorical gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:
    from . import phase_transport as spt
    from .phase_query_planner import normalized_position_channels, sinusoidal_phase_encoding
except ImportError:  # direct import from the spt_v2 directory
    import phase_transport as spt  # type: ignore
    from phase_query_planner import normalized_position_channels, sinusoidal_phase_encoding  # type: ignore


ARCHITECTURE_NAME = "grounded_cell_phase_transport_v3"
GLOBAL_TEXT_LAYERS = 1
DENSE_TEXT_LAYERS = 1
TEMPORAL_ATTENTION_LAYERS = 1
DEFAULT_EDIT_SLOTS = 8
DEFAULT_MATCH_CHANNELS = 32
MAX_GENERATE_FRACTION_PER_PHASE = 0.12
SEMANTIC_RESIDUAL_INITIAL_SCALE = 0.05
TEMPORAL_RESIDUAL_INITIAL_SCALE = 0.05
SLOT_SELF_INITIAL_SCALE = 0.01
TEMPORAL_CANDIDATES = (-2, -1, 0, 1, 2)
SPATIAL_CANDIDATES = (-4, -2, 0, 2, 4)


class GroundedPlannerError(spt.PhaseTransportError):
    """Raised when the source-grounded planner contract differs."""


@dataclass(frozen=True)
class GroundedPhasePlannerConfig:
    architecture: str = ARCHITECTURE_NAME
    latent_channels: int = 64
    text_channels: int = 4096
    hidden_channels: int = 192
    attention_heads: int = 8
    match_channels: int = DEFAULT_MATCH_CHANNELS
    edit_slots: int = DEFAULT_EDIT_SLOTS
    latent_phases: int = spt.LATENT_PHASES
    dense_query_chunk_size: int = 4096
    global_text_layers: int = GLOBAL_TEXT_LAYERS
    dense_text_layers: int = DENSE_TEXT_LAYERS
    temporal_attention_layers: int = TEMPORAL_ATTENTION_LAYERS
    max_generate_fraction_per_phase: float = MAX_GENERATE_FRACTION_PER_PHASE
    source_bank_detach: bool = True

    def validate(self) -> None:
        if self.architecture != ARCHITECTURE_NAME:
            raise GroundedPlannerError(
                f"planner architecture must be {ARCHITECTURE_NAME!r}"
            )
        if self.latent_phases != spt.LATENT_PHASES:
            raise GroundedPlannerError("grounded planner requires exactly 21 phases")
        for name in (
            "latent_channels",
            "text_channels",
            "hidden_channels",
            "attention_heads",
            "match_channels",
            "edit_slots",
            "dense_query_chunk_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise GroundedPlannerError(f"{name} must be a positive integer")
        if self.hidden_channels % self.attention_heads:
            raise GroundedPlannerError(
                "hidden_channels must be divisible by attention_heads"
            )
        groups = min(32, self.hidden_channels)
        if self.hidden_channels % groups:
            raise GroundedPlannerError(
                "hidden_channels must be divisible by its GroupNorm group count"
            )
        if (
            self.global_text_layers != GLOBAL_TEXT_LAYERS
            or self.dense_text_layers != DENSE_TEXT_LAYERS
            or self.temporal_attention_layers != TEMPORAL_ATTENTION_LAYERS
        ):
            raise GroundedPlannerError("grounded planner depth is fixed for v3")
        if self.max_generate_fraction_per_phase != MAX_GENERATE_FRACTION_PER_PHASE:
            raise GroundedPlannerError("grounded planner fixes per-phase Generate to 0.12")
        if type(self.source_bank_detach) is not bool or not self.source_bank_detach:
            raise GroundedPlannerError("grounded planner requires a detached source bank")


def candidate_lattice() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (dt, dy, dx)
        for dt in TEMPORAL_CANDIDATES
        for dy in SPATIAL_CANDIDATES
        for dx in SPATIAL_CANDIDATES
    )


def budgeted_factorized_gates(
    change_logits: Any,
    novelty_logits: Any,
    *,
    maximum: float = MAX_GENERATE_FRACTION_PER_PHASE,
) -> tuple[Any, Any, Any]:
    """Compose P/T/G gates with a structural per-sample, per-phase G cap.

    ``change`` means content may leave the immutable same-cell source path;
    ``novelty`` divides that mass into transport and generation.  If raw
    generation exceeds the phase budget, the rejected mass returns to
    preserve.  It is never silently reclassified as transport.
    """

    import torch

    if (
        getattr(change_logits, "ndim", None) != 5
        or tuple(change_logits.shape) != tuple(novelty_logits.shape)
        or int(change_logits.shape[1]) != 1
    ):
        raise GroundedPlannerError("change/novelty logits must share [B,1,T,H,W]")
    if not math.isfinite(float(maximum)) or not 0.0 < float(maximum) <= 0.12:
        raise GroundedPlannerError("Generate budget must lie in (0,0.12]")
    change = torch.sigmoid(change_logits.float())
    raw_generate = change * torch.sigmoid(novelty_logits.float())
    raw_phase_mass = raw_generate.mean(dim=(-2, -1), keepdim=True)
    scale = torch.clamp(
        float(maximum) / raw_phase_mass.clamp_min(1.0e-12),
        max=1.0,
    )
    generate = raw_generate * scale
    rejected = raw_generate - generate
    transport = change - raw_generate
    preserve = 1.0 - change + rejected
    gates = torch.cat((preserve, transport, generate), dim=1)
    if not bool(torch.isfinite(gates).all()):
        raise GroundedPlannerError("factorized gates are non-finite")
    phase_mass = gates[:, spt.GATE_GENERATE].mean(dim=(-2, -1))
    if bool((phase_mass > float(maximum) + 2.0e-6).any()):
        raise GroundedPlannerError("factorized gates violated the per-phase budget")
    return gates, raw_generate, scale


try:
    import torch
    import torch.nn.functional as functional
    from torch import nn

    class Residual3DBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.norm = nn.GroupNorm(min(32, channels), channels)
            self.depthwise = nn.Conv3d(
                channels, channels, kernel_size=3, padding=1, groups=channels
            )
            self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
            nn.init.zeros_(self.pointwise.weight)
            nn.init.zeros_(self.pointwise.bias)

        def forward(self, value: Any) -> Any:
            delta = self.pointwise(
                functional.silu(self.depthwise(functional.silu(self.norm(value))))
            )
            return value + delta


    class CrossAttentionResidual(nn.Module):
        def __init__(
            self,
            channels: int,
            heads: int,
            *,
            residual_scale: float = SEMANTIC_RESIDUAL_INITIAL_SCALE,
        ):
            super().__init__()
            self.query_norm = nn.LayerNorm(channels)
            self.context_norm = nn.LayerNorm(channels)
            self.attention = nn.MultiheadAttention(
                channels, heads, batch_first=True, dropout=0.0
            )
            self.ff_norm = nn.LayerNorm(channels)
            self.feedforward = nn.Sequential(
                nn.Linear(channels, 4 * channels),
                nn.GELU(approximate="tanh"),
                nn.Linear(4 * channels, channels),
            )
            self.attention_scale = nn.Parameter(
                torch.full((1,), float(residual_scale))
            )
            self.feedforward_scale = nn.Parameter(
                torch.full((1,), float(residual_scale))
            )

        def forward(self, query: Any, context: Any) -> Any:
            attended, _ = self.attention(
                self.query_norm(query),
                self.context_norm(context),
                self.context_norm(context),
                need_weights=False,
            )
            query = query + self.attention_scale * attended
            return query + self.feedforward_scale * self.feedforward(
                self.ff_norm(query)
            )


    class ChunkedCrossAttentionResidual(CrossAttentionResidual):
        def __init__(self, channels: int, heads: int, chunk_size: int):
            super().__init__(channels, heads)
            self.chunk_size = int(chunk_size)

        def forward(self, query: Any, context: Any) -> Any:
            chunks = []
            for start in range(0, int(query.shape[1]), self.chunk_size):
                stop = min(start + self.chunk_size, int(query.shape[1]))
                chunks.append(super().forward(query[:, start:stop], context))
            return torch.cat(chunks, dim=1)


    class TemporalAxialBlock(nn.Module):
        def __init__(self, channels: int, heads: int):
            super().__init__()
            self.norm = nn.LayerNorm(channels)
            self.attention = nn.MultiheadAttention(
                channels, heads, batch_first=True, dropout=0.0
            )
            self.scale = nn.Parameter(
                torch.full((1,), TEMPORAL_RESIDUAL_INITIAL_SCALE)
            )

        def forward(self, volume: Any) -> Any:
            batch, channels, phases, height, width = map(int, volume.shape)
            sequence = (
                volume.permute(0, 3, 4, 2, 1)
                .reshape(batch * height * width, phases, channels)
            )
            normalized = self.norm(sequence)
            attended, _ = self.attention(
                normalized, normalized, normalized, need_weights=False
            )
            sequence = sequence + self.scale * attended
            return (
                sequence.reshape(batch, height, width, phases, channels)
                .permute(0, 4, 3, 1, 2)
                .contiguous()
            )


    class ZeroFusion(nn.Module):
        """Fuse a coarse semantic delta while starting from the source skip."""

        def __init__(self, channels: int):
            super().__init__()
            self.norm = nn.GroupNorm(min(32, channels), 2 * channels)
            self.delta = nn.Sequential(
                nn.Conv3d(2 * channels, channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv3d(channels, channels, kernel_size=1),
            )
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)

        def forward(self, source_skip: Any, semantic: Any) -> Any:
            if tuple(source_skip.shape[2:]) != tuple(semantic.shape[2:]):
                semantic = functional.interpolate(
                    semantic,
                    size=tuple(map(int, source_skip.shape[2:])),
                    mode="trilinear",
                    align_corners=False,
                )
            joined = torch.cat((source_skip, semantic), dim=1)
            return source_skip + self.delta(functional.silu(self.norm(joined)))


    class GroundedPhasePlanner(nn.Module):
        architecture = ARCHITECTURE_NAME

        def __init__(self, config: GroundedPhasePlannerConfig):
            super().__init__()
            config.validate()
            self.config = config
            hidden = config.hidden_channels
            self.source_in = nn.Conv3d(
                config.latent_channels + 3, hidden, kernel_size=1
            )
            self.fine_source_block = Residual3DBlock(hidden)
            self.mid_source_block = Residual3DBlock(hidden)
            self.coarse_source_block = Residual3DBlock(hidden)
            self.text_in = nn.Linear(config.text_channels, hidden)
            self.text_norm = nn.LayerNorm(hidden)

            self.edit_slots = nn.Parameter(torch.empty(config.edit_slots, hidden))
            nn.init.normal_(self.edit_slots, mean=0.0, std=hidden ** -0.5)
            self.slot_text = CrossAttentionResidual(hidden, config.attention_heads)
            self.slot_source = CrossAttentionResidual(hidden, config.attention_heads)
            self.slot_self_norm = nn.LayerNorm(hidden)
            self.slot_self_attention = nn.MultiheadAttention(
                hidden, config.attention_heads, batch_first=True, dropout=0.0
            )
            self.slot_self_scale = nn.Parameter(
                torch.full((1,), SLOT_SELF_INITIAL_SCALE)
            )

            self.register_buffer(
                "phase_time_encoding",
                sinusoidal_phase_encoding(config.latent_phases, hidden),
                persistent=True,
            )
            self.phase_projection = nn.Linear(hidden, hidden)
            self.cell_text = ChunkedCrossAttentionResidual(
                hidden, config.attention_heads, config.dense_query_chunk_size
            )
            self.cell_slots = ChunkedCrossAttentionResidual(
                hidden, config.attention_heads, config.dense_query_chunk_size
            )
            self.temporal = TemporalAxialBlock(hidden, config.attention_heads)
            self.mid_fusion = ZeroFusion(hidden)
            self.fine_fusion = ZeroFusion(hidden)
            self.mid_block = Residual3DBlock(hidden)
            self.fine_block = Residual3DBlock(hidden)
            self.fine_ground_block = Residual3DBlock(hidden)

            self.coarse_change_head = nn.Conv3d(hidden, 1, kernel_size=1)
            self.mid_change_head = nn.Conv3d(hidden, 1, kernel_size=1)
            self.change_head = nn.Conv3d(hidden, 1, kernel_size=1)
            self.novelty_head = nn.Conv3d(hidden, 1, kernel_size=1)
            change_bias = math.log(0.08 / 0.92)
            for head in (
                self.coarse_change_head,
                self.mid_change_head,
                self.change_head,
            ):
                nn.init.zeros_(head.weight)
                nn.init.constant_(head.bias, change_bias)
            nn.init.zeros_(self.novelty_head.weight)
            nn.init.zeros_(self.novelty_head.bias)

            self.offset_query = nn.Conv3d(hidden, config.match_channels, kernel_size=1)
            self.offset_key = nn.Conv3d(hidden, config.match_channels, kernel_size=1)
            with torch.no_grad():
                self.offset_key.weight.copy_(self.offset_query.weight)
                self.offset_key.bias.copy_(self.offset_query.bias)
            # Start source correlation at 10% strength so the explicit zero
            # prior wins deterministically before training.  The bounded
            # sigmoid scale can then grow as categorical supervision proves
            # that a shifted source candidate is useful.
            self.offset_correlation_logit = nn.Parameter(
                torch.full((1,), math.log(0.1 / 0.9))
            )
            candidates = torch.tensor(candidate_lattice(), dtype=torch.float32)
            self.register_buffer("offset_candidates", candidates, persistent=True)
            self.offset_residual = nn.Conv3d(
                hidden, int(candidates.shape[0]), kernel_size=1
            )
            nn.init.zeros_(self.offset_residual.weight)
            nn.init.zeros_(self.offset_residual.bias)
            zero_index = candidate_lattice().index((0, 0, 0))
            with torch.no_grad():
                self.offset_residual.bias[zero_index] = 2.0

        def _validate_instruction_tokens(self, source: Any, tokens: Any) -> None:
            if getattr(tokens, "ndim", None) != 3:
                raise GroundedPlannerError(
                    "instruction_tokens must be full unpadded [B,L,text_channels]"
                )
            if (
                int(tokens.shape[0]) != int(source.shape[0])
                or int(tokens.shape[1]) <= 0
                or int(tokens.shape[2]) != self.config.text_channels
            ):
                raise GroundedPlannerError("instruction token geometry differs")
            if not bool(torch.isfinite(tokens).all()):
                raise GroundedPlannerError("instruction tokens are non-finite")
            if bool((tokens.float().abs().sum(dim=-1) == 0).any()):
                raise GroundedPlannerError("instruction tokens contain padded zero rows")

        @staticmethod
        def _spatial_pool(volume: Any) -> Any:
            return functional.avg_pool3d(
                volume,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
                ceil_mode=True,
            )

        def _grounded_slots(self, coarse: Any, text: Any) -> Any:
            batch = int(coarse.shape[0])
            slots = self.edit_slots.unsqueeze(0).expand(batch, -1, -1)
            slots = self.slot_text(slots, text)
            source_tokens = coarse.permute(0, 2, 3, 4, 1).reshape(
                batch, -1, self.config.hidden_channels
            )
            slots = self.slot_source(slots, source_tokens)
            normalized = self.slot_self_norm(slots)
            attended, _ = self.slot_self_attention(
                normalized, normalized, normalized, need_weights=False
            )
            return slots + self.slot_self_scale * attended

        def _dense_grounding(self, coarse: Any, text: Any, slots: Any) -> Any:
            batch, channels, phases, height, width = map(int, coarse.shape)
            dense = coarse.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)
            phase = self.phase_projection(self.phase_time_encoding).view(
                1, phases, 1, 1, channels
            )
            phase = phase.expand(batch, -1, height, width, -1).reshape(
                batch, -1, channels
            )
            dense = dense + phase
            dense = self.cell_text(dense, text)
            dense = self.cell_slots(dense, slots)
            return dense.reshape(batch, phases, height, width, channels).permute(
                0, 4, 1, 2, 3
            ).contiguous()

        @staticmethod
        def _valid_candidate_mask(
            phases: int,
            height: int,
            width: int,
            candidates: Any,
            *,
            device: Any,
        ) -> Any:
            tt = torch.arange(phases, device=device).view(1, phases, 1, 1)
            yy = torch.arange(height, device=device).view(1, 1, height, 1)
            xx = torch.arange(width, device=device).view(1, 1, 1, width)
            masks = []
            for dt, dy, dx in candidates.tolist():
                masks.append(
                    (tt + int(dt) >= 0)
                    & (tt + int(dt) < phases)
                    & (yy + int(dy) >= 0)
                    & (yy + int(dy) < height)
                    & (xx + int(dx) >= 0)
                    & (xx + int(dx) < width)
                )
            return torch.cat(masks, dim=0)

        def _offset_logits(self, decoded: Any, source_fine: Any) -> Any:
            query = functional.normalize(self.offset_query(decoded).float(), dim=1)
            key = functional.normalize(self.offset_key(source_fine).float(), dim=1)
            batch, _, phases, height, width = map(int, query.shape)
            temporal_pad = max(abs(value) for value in TEMPORAL_CANDIDATES)
            spatial_pad = max(abs(value) for value in SPATIAL_CANDIDATES)
            padded = functional.pad(
                key,
                (spatial_pad, spatial_pad, spatial_pad, spatial_pad, temporal_pad, temporal_pad),
            )
            correlations = []
            for dt, dy, dx in candidate_lattice():
                shifted = padded[
                    :,
                    :,
                    temporal_pad + dt : temporal_pad + dt + phases,
                    spatial_pad + dy : spatial_pad + dy + height,
                    spatial_pad + dx : spatial_pad + dx + width,
                ]
                correlations.append((query * shifted).sum(dim=1))
            correlation_scale = torch.sigmoid(self.offset_correlation_logit.float())
            logits = 5.0 * correlation_scale * torch.stack(correlations, dim=1)
            logits = logits + self.offset_residual(decoded).float()
            valid = self._valid_candidate_mask(
                phases,
                height,
                width,
                self.offset_candidates,
                device=decoded.device,
            ).unsqueeze(0)
            return logits.masked_fill(~valid, -1.0e4)

        def forward(self, source: Any, instruction_tokens: Any) -> spt.PhasePlan:
            spt._validate_video(source, label="source")
            if int(source.shape[-1]) != self.config.latent_channels:
                raise GroundedPlannerError("source latent channel count differs")
            self._validate_instruction_tokens(source, instruction_tokens)
            bank = source.detach()
            positions = normalized_position_channels(bank)
            source_channels = bank.permute(0, 4, 1, 2, 3).float()
            source_fine = self.fine_source_block(
                self.source_in(torch.cat((source_channels, positions), dim=1))
            )
            source_mid = self.mid_source_block(self._spatial_pool(source_fine))
            source_coarse = self.coarse_source_block(self._spatial_pool(source_mid))
            text = self.text_norm(self.text_in(instruction_tokens.float()))
            slots = self._grounded_slots(source_coarse, text)
            coarse = self.temporal(
                self._dense_grounding(source_coarse, text, slots)
            )
            mid = self.mid_block(self.mid_fusion(source_mid, coarse))
            decoded = self.fine_block(self.fine_fusion(source_fine, mid))
            # Coarse attention supplies global semantics, while this second
            # chunked pass gives every output-resolution cell its own direct
            # full-token and grounded-slot query before the execution heads.
            decoded = self.fine_ground_block(
                self._dense_grounding(decoded, text, slots)
            )

            change_logits = self.change_head(decoded)
            novelty_logits = self.novelty_head(decoded)
            gates, raw_generate, budget_scale = budgeted_factorized_gates(
                change_logits,
                novelty_logits,
                maximum=self.config.max_generate_fraction_per_phase,
            )
            offset_logits = self._offset_logits(decoded, source_fine)
            offset_probs = torch.softmax(offset_logits, dim=1)
            soft_offsets = torch.einsum(
                "bkthw,kc->bcthw",
                offset_probs,
                self.offset_candidates.float(),
            )
            hard_index = offset_probs.argmax(dim=1)
            hard = functional.one_hot(
                hard_index, num_classes=int(self.offset_candidates.shape[0])
            ).permute(0, 4, 1, 2, 3).float()
            straight_through = hard + (offset_probs - offset_probs.detach())
            offsets = torch.einsum(
                "bkthw,kc->bcthw",
                straight_through,
                self.offset_candidates.float(),
            )
            diagnostics = {
                "architecture": ARCHITECTURE_NAME,
                "change_logits": change_logits,
                "novelty_logits": novelty_logits,
                "prebudget_generate_probs": raw_generate,
                "generate_budget_scale": budget_scale,
                "offset_candidate_logits": offset_logits,
                "offset_candidates": self.offset_candidates,
                "soft_offsets": soft_offsets,
                "offset_correlation_scale": torch.sigmoid(
                    self.offset_correlation_logit.float()
                ),
                "coarse_change_logits": self.coarse_change_head(coarse),
                "mid_change_logits": self.mid_change_head(mid),
            }
            plan = spt.PhasePlan(
                offsets=offsets,
                gate_probs=gates,
                provenance="student",
                diagnostics=diagnostics,
            )
            plan.validate(source)
            return plan

except ImportError:  # pragma: no cover - contract-only environments

    class GroundedPhasePlanner:  # type: ignore[no-redef]
        architecture = ARCHITECTURE_NAME

        def __init__(self, config: GroundedPhasePlannerConfig):
            config.validate()
            raise GroundedPlannerError("GroundedPhasePlanner requires PyTorch")

        def forward(self, source: Any, instruction_tokens: Any) -> spt.PhasePlan:
            raise GroundedPlannerError("GroundedPhasePlanner requires PyTorch")
