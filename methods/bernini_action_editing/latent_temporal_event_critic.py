#!/usr/bin/env python3
"""Small text-conditioned event head on frozen Bernini hidden residuals.

The head never sees a generated RGB target or a raw appearance embedding.  For
one owner clean latent, frozen Bernini is queried at an exact shared ``x_sigma``
under the target-action and scene-matched no-op conditions.  The head consumes
only ``H_action(x_sigma) - H_noop(x_sigma)`` at one preregistered hook.

Spatial evidence is retained by a fixed, content-independent Rademacher sketch
(not a mask).  Temporal evidence is a direct sum of phase-zero boundary,
lag-1/2/4 differences, and a four-phase terminal hold.  A learned head below
one million parameters performs within-cell group ranking against every named
hard negative.  When used as an RV2V reward, the frozen Bernini forwards remain
in the autograd graph, so the score differentiates with respect to the current
RV2V clean latent while critic/Bernini parameters stay frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as functional

import latent_temporal_event_critic_dataset as data_contract


SCHEMA_VERSION = "bernini-frozen-hidden-latent-temporal-event-critic-v1"
BACKEND_ID = "frozen_text_conditioned_temporal_event_critic_raw_score_vjp_v1"
EXPECTED_PHASES = data_contract.LATENT_PHASES
EXPECTED_HIDDEN_SIZE = data_contract.HIDDEN_SIZE
TEMPORAL_LAGS = (1, 2, 4)
TERMINAL_HOLD_START = 17
TEMPORAL_FEATURE_STEPS = EXPECTED_PHASES * 4 + 1
MILESTONE_NAMES = (
    "actor_object_binding",
    "transition",
    "chronology",
    "terminal_hold",
)
TYPED_NUISANCE_NAMES = (
    "actor",
    "scene",
    "camera",
    "appearance",
    "seed_quality",
)

ROLE_MILESTONES = {
    "same_video_reverse": ("chronology", "terminal_hold"),
    "same_video_freeze_first": ("transition", "terminal_hold"),
    "same_video_phase_shuffle": ("chronology",),
    "semantic_noop": ("transition", "terminal_hold"),
    "semantic_incomplete": ("transition", "terminal_hold"),
    "semantic_reverse": ("chronology", "terminal_hold"),
    "semantic_shuffle": ("chronology",),
    "semantic_wrong_actor": ("actor_object_binding",),
    "semantic_wrong_object": ("actor_object_binding",),
    "semantic_camera_only": tuple(MILESTONE_NAMES),
    "semantic_appearance_only": tuple(MILESTONE_NAMES),
    "semantic_generic_wrong_motion": ("transition", "chronology"),
}


class LatentTemporalEventCriticError(RuntimeError):
    """A hidden pair, fixed feature map, loss group, or VJP failed closed."""


def apply_registered_temporal_transform(
    clean_latent: torch.Tensor, transform: str
) -> torch.Tensor:
    """Apply a fixed 21-phase transform without breaking an input graph."""

    if (
        not isinstance(clean_latent, torch.Tensor)
        or clean_latent.ndim != 5
        or int(clean_latent.shape[2]) != EXPECTED_PHASES
        or not clean_latent.is_floating_point()
        or clean_latent.device.type == "meta"
        or not bool(torch.isfinite(clean_latent).all().item())
    ):
        raise LatentTemporalEventCriticError(
            "temporal transform input must be finite [B,C,21,H,W]"
        )
    maps = {
        "chronological": tuple(range(EXPECTED_PHASES)),
        "reverse": tuple(range(EXPECTED_PHASES - 1, -1, -1)),
        "freeze_first": (0,) * EXPECTED_PHASES,
        "phase_shuffle": tuple((8 * index) % EXPECTED_PHASES for index in range(EXPECTED_PHASES)),
    }
    if transform not in maps:
        raise LatentTemporalEventCriticError("temporal transform is not registered")
    indices = torch.tensor(maps[transform], dtype=torch.long, device=clean_latent.device)
    return clean_latent.index_select(2, indices)


def _finite_positive(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise LatentTemporalEventCriticError(f"{label} must be positive finite")
    return float(value)


def _canonical_tensor_digest(value: torch.Tensor, *, label: str) -> str:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise LatentTemporalEventCriticError(
            f"{label} must be detached finite FP32"
        )
    owned = value.detach().cpu().contiguous().clone()
    header = (
        f"bernini-ltec-f32le-v1|shape={','.join(str(int(x)) for x in owned.shape)}|"
    ).encode("ascii")
    # PyTorch CPU float32 storage is little endian on the supported AUH hosts.
    raw = bytes(owned.untyped_storage())
    return hashlib.sha256(header + raw).hexdigest()


def make_fixed_spatial_sketch(
    *,
    patch_positions: int,
    coordinates: int = 16,
    seed: int = 20260808017,
) -> torch.Tensor:
    """Create the preregistered counter-hash Rademacher spatial sketch.

    Every coordinate covers every patch.  No video, actor location, mask,
    detector, or annotation influences a weight.
    """

    if (
        type(patch_positions) is not int
        or patch_positions < 2
        or type(coordinates) is not int
        or not 2 <= coordinates <= patch_positions
        or type(seed) is not int
        or seed < 0
    ):
        raise LatentTemporalEventCriticError("spatial sketch dimensions/seed differ")
    scale = 1.0 / math.sqrt(float(patch_positions))
    matrix = torch.empty(coordinates, patch_positions, dtype=torch.float32)
    for row in range(coordinates):
        for column in range(patch_positions):
            token = f"{seed}:{row}:{column}".encode("ascii")
            positive = hashlib.sha256(token).digest()[0] & 1
            matrix[row, column] = scale if positive else -scale
    if int(torch.linalg.matrix_rank(matrix).item()) != coordinates:
        raise LatentTemporalEventCriticError("fixed spatial sketch is rank deficient")
    return matrix


@dataclass(frozen=True)
class NuisanceBasisFit:
    type_names: tuple[str, ...]
    basis: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    observation_count_by_type: tuple[int, ...]
    basis_digest: str


def fit_typed_nuisance_basis(
    typed_directions: Mapping[str, torch.Tensor],
    *,
    rank_rtol: float = 1.0e-5,
    maximum_rank: int = 64,
) -> NuisanceBasisFit:
    """Fit a discovery-only actor/scene/camera/appearance/seed span.

    Each value is ``[N,1536]`` and must provide at least two named donor groups.
    Callers must construct the directions from the train split only and bind
    their donor IDs in the dataset receipt.  Validation/test may only reuse the
    returned frozen values.
    """

    if not isinstance(typed_directions, Mapping) or set(typed_directions) != set(
        TYPED_NUISANCE_NAMES
    ):
        raise LatentTemporalEventCriticError(
            "typed nuisance directions require exact actor/scene/camera/appearance/seed closure"
        )
    rtol = _finite_positive(rank_rtol, label="rank_rtol")
    if rtol >= 1.0 or type(maximum_rank) is not int or maximum_rank < 1:
        raise LatentTemporalEventCriticError("nuisance rank policy differs")
    normalized = []
    counts = []
    reference_device = None
    for name in TYPED_NUISANCE_NAMES:
        value = typed_directions[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.ndim != 2
            or tuple(value.shape)[1:] != (EXPECTED_HIDDEN_SIZE,)
            or int(value.shape[0]) < 2
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise LatentTemporalEventCriticError(
                f"nuisance {name} must be detached finite FP32 [N>=2,1536]"
            )
        if reference_device is None:
            reference_device = value.device
        elif value.device != reference_device:
            raise LatentTemporalEventCriticError("nuisance donor devices differ")
        norms = torch.linalg.vector_norm(value, dim=1)
        if bool((norms <= 1.0e-8).any().item()):
            raise LatentTemporalEventCriticError(f"nuisance {name} has a null donor")
        normalized.append(value / norms[:, None])
        counts.append(int(value.shape[0]))
    columns = torch.cat(normalized, dim=0).transpose(0, 1).contiguous()
    left, singular_values, _ = torch.linalg.svd(columns, full_matrices=False)
    active = singular_values > singular_values[0] * rtol
    rank = min(int(active.sum().item()), maximum_rank)
    if rank < 1 or rank >= EXPECTED_HIDDEN_SIZE:
        raise LatentTemporalEventCriticError("nuisance basis leaves no valid complement")
    basis = left[:, :rank].detach().contiguous()
    eye = torch.eye(rank, dtype=torch.float32, device=basis.device)
    if float((basis.transpose(0, 1) @ basis - eye).abs().max().item()) > 2.0e-4:
        raise LatentTemporalEventCriticError("nuisance basis is not orthonormal")
    return NuisanceBasisFit(
        type_names=TYPED_NUISANCE_NAMES,
        basis=basis,
        singular_values=singular_values.detach(),
        rank=rank,
        observation_count_by_type=tuple(counts),
        basis_digest=_canonical_tensor_digest(basis, label="nuisance basis"),
    )


@dataclass(frozen=True)
class CriticConfig:
    hidden_size: int = EXPECTED_HIDDEN_SIZE
    # ``None`` means infer P from the episode-specific sketch passed to the
    # constructor.  P is not a learned/head dimension: all geometries compress
    # to ``spatial_coordinates`` before any trainable layer.
    patch_positions: Optional[int] = None
    spatial_coordinates: int = 16
    spatial_sketch_seed: int = 20260808017
    projected_size: int = 48
    model_size: int = 96
    attention_heads: int = 4
    transformer_layers: int = 1
    softmin_temperature: float = 0.25
    dropout: float = 0.0
    require_nuisance_basis: bool = False
    production_geometry: bool = True

    def validate(self) -> None:
        integer_fields = {
            "hidden_size": self.hidden_size,
            "spatial_coordinates": self.spatial_coordinates,
            "spatial_sketch_seed": self.spatial_sketch_seed,
            "projected_size": self.projected_size,
            "model_size": self.model_size,
            "attention_heads": self.attention_heads,
            "transformer_layers": self.transformer_layers,
        }
        if any(type(value) is not int or value <= 0 for value in integer_fields.values()):
            raise LatentTemporalEventCriticError("critic dimensions must be positive integers")
        if self.patch_positions is not None and (
            type(self.patch_positions) is not int
            or self.patch_positions < self.spatial_coordinates
        ):
            raise LatentTemporalEventCriticError(
                "patch_positions must be None or an integer no smaller than K"
            )
        if self.production_geometry and (
            self.hidden_size != EXPECTED_HIDDEN_SIZE
            or self.spatial_coordinates != 16
            or self.spatial_sketch_seed != 20260808017
        ):
            raise LatentTemporalEventCriticError("production critic geometry differs from Bernini")
        if self.model_size % self.attention_heads != 0:
            raise LatentTemporalEventCriticError("model size must divide attention heads")
        _finite_positive(self.softmin_temperature, label="softmin temperature")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise LatentTemporalEventCriticError("dropout must lie in [0,1)")
        if type(self.require_nuisance_basis) is not bool:
            raise LatentTemporalEventCriticError("require_nuisance_basis must be bool")
        if type(self.production_geometry) is not bool:
            raise LatentTemporalEventCriticError("production_geometry must be bool")


@dataclass(frozen=True)
class CriticOutput:
    milestone_scores: torch.Tensor
    score: torch.Tensor
    temporal_tokens: torch.Tensor


class FrozenHiddenTemporalEventCritic(nn.Module):
    """A mask-free small head over a frozen same-state Bernini residual."""

    def __init__(
        self,
        spatial_sketch: torch.Tensor,
        *,
        config: CriticConfig = CriticConfig(),
        nuisance_basis: Optional[torch.Tensor] = None,
        expected_spatial_sketch_digest: Optional[str] = None,
        expected_nuisance_basis_digest: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not isinstance(spatial_sketch, torch.Tensor) or spatial_sketch.ndim != 2:
            raise LatentTemporalEventCriticError(
                "spatial sketch must be a detached FP32 matrix [K,P]"
            )
        inferred_positions = int(spatial_sketch.shape[1])
        if config.patch_positions is None:
            config = replace(config, patch_positions=inferred_positions)
        config.validate()
        if (
            spatial_sketch.dtype != torch.float32
            or tuple(spatial_sketch.shape)
            != (config.spatial_coordinates, config.patch_positions)
            or spatial_sketch.requires_grad
            or spatial_sketch.grad_fn is not None
            or not bool(torch.isfinite(spatial_sketch).all().item())
        ):
            raise LatentTemporalEventCriticError(
                "spatial sketch must be detached FP32 [K,P] for this episode geometry"
            )
        sketch_digest = _canonical_tensor_digest(spatial_sketch, label="spatial sketch")
        if expected_spatial_sketch_digest is not None and sketch_digest != expected_spatial_sketch_digest:
            raise LatentTemporalEventCriticError("spatial sketch digest differs")
        if config.production_geometry:
            registered = make_fixed_spatial_sketch(
                patch_positions=int(config.patch_positions),
                coordinates=config.spatial_coordinates,
                seed=config.spatial_sketch_seed,
            ).to(device=spatial_sketch.device)
            if not torch.equal(spatial_sketch, registered):
                raise LatentTemporalEventCriticError(
                    "production sketch is outside the registered dynamic-P family"
                )
        basis = nuisance_basis
        if basis is None:
            if config.require_nuisance_basis:
                raise LatentTemporalEventCriticError(
                    "scientific critic requires a discovery-frozen nuisance basis"
                )
            basis = torch.empty(config.hidden_size, 0, dtype=torch.float32)
        if (
            not isinstance(basis, torch.Tensor)
            or basis.dtype != torch.float32
            or basis.ndim != 2
            or int(basis.shape[0]) != config.hidden_size
            or basis.requires_grad
            or basis.grad_fn is not None
            or not bool(torch.isfinite(basis).all().item())
        ):
            raise LatentTemporalEventCriticError("nuisance basis shape/value differs")
        if int(basis.shape[1]) > 0:
            gram = basis.transpose(0, 1) @ basis
            eye = torch.eye(int(basis.shape[1]), dtype=torch.float32, device=basis.device)
            if float((gram - eye).abs().max().item()) > 2.0e-4:
                raise LatentTemporalEventCriticError("nuisance basis is not orthonormal")
            basis_digest = _canonical_tensor_digest(basis, label="nuisance basis")
            if expected_nuisance_basis_digest is not None and basis_digest != expected_nuisance_basis_digest:
                raise LatentTemporalEventCriticError("nuisance basis digest differs")
        elif expected_nuisance_basis_digest is not None:
            raise LatentTemporalEventCriticError("declared nuisance digest has no basis")

        self.config = config
        self.spatial_sketch_digest = sketch_digest
        self.nuisance_basis_digest = (
            _canonical_tensor_digest(basis, label="nuisance basis")
            if int(basis.shape[1]) > 0
            else None
        )
        self.register_buffer("spatial_sketch", spatial_sketch.detach().clone())
        self.register_buffer("nuisance_basis", basis.detach().clone())

        self.channel_projection = nn.Linear(
            config.hidden_size, config.projected_size, bias=False
        )
        self.channel_norm = nn.LayerNorm(config.projected_size)
        self.sketch_mixer = nn.Sequential(
            nn.Linear(
                config.spatial_coordinates * config.projected_size,
                config.model_size,
            ),
            nn.GELU(),
            nn.LayerNorm(config.model_size),
        )
        self.temporal_position = nn.Parameter(
            torch.zeros(TEMPORAL_FEATURE_STEPS, config.model_size)
        )
        self.temporal_block = nn.Parameter(torch.zeros(5, config.model_size))
        block_ids = (
            [0] * EXPECTED_PHASES
            + [1] * EXPECTED_PHASES
            + [2] * EXPECTED_PHASES
            + [3] * EXPECTED_PHASES
            + [4]
        )
        self.register_buffer(
            "temporal_block_ids",
            torch.tensor(block_ids, dtype=torch.long),
            persistent=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_size,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.model_size,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=config.transformer_layers
        )
        self.milestone_queries = nn.Parameter(
            torch.zeros(len(MILESTONE_NAMES), config.model_size)
        )
        self.milestone_attention = nn.MultiheadAttention(
            config.model_size,
            config.attention_heads,
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.milestone_norm = nn.LayerNorm(config.model_size)
        self.milestone_head = nn.Linear(config.model_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.temporal_position, std=0.01)
        nn.init.normal_(self.temporal_block, std=0.01)
        nn.init.normal_(self.milestone_queries, std=0.02)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _canonical_hidden(self, value: Any, *, label: str, require_input_grad: bool) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type == "meta"
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or value.ndim not in (4, 5)
            or int(value.shape[1]) != EXPECTED_PHASES
            or int(value.shape[-1]) != self.config.hidden_size
            or not bool(torch.isfinite(value).all().item())
        ):
            raise LatentTemporalEventCriticError(
                f"{label} must be finite [B,21,P,1536] or [B,21,H,W,1536]"
            )
        if require_input_grad and (not value.requires_grad or value.grad_fn is None):
            raise LatentTemporalEventCriticError(
                f"{label} live reward hidden must remain graph-connected"
            )
        if value.ndim == 5:
            batch, phases, height, width, channels = map(int, value.shape)
            result = value.reshape(batch, phases, height * width, channels)
        else:
            result = value
        if int(result.shape[2]) != int(self.spatial_sketch.shape[1]):
            raise LatentTemporalEventCriticError(
                f"{label} patch count differs from the frozen spatial sketch"
            )
        return result.float()

    def _project_nuisance(self, value: torch.Tensor) -> torch.Tensor:
        if int(self.nuisance_basis.shape[1]) == 0:
            return value
        basis = self.nuisance_basis
        return value - (value @ basis) @ basis.transpose(0, 1)

    def sketch_same_state_hidden_residual(
        self,
        action_hidden: torch.Tensor,
        noop_hidden: torch.Tensor,
        *,
        require_input_grad: bool = False,
    ) -> torch.Tensor:
        """Subtract then irreversibly compress a same-state hidden pair.

        This is the preferred critic-dataset artifact: ``[B,21,K,1536]`` is
        roughly sixty times smaller than storing both ``[B,21,P,1536]``
        hidden tensors.  Because nuisance projection acts on the channel axis,
        it commutes exactly with this fixed spatial linear sketch and can be
        fitted later from the train split.
        """

        action = self._canonical_hidden(
            action_hidden,
            label="action hidden",
            require_input_grad=require_input_grad,
        )
        noop = self._canonical_hidden(
            noop_hidden,
            label="no-op hidden",
            require_input_grad=require_input_grad,
        )
        if action.shape != noop.shape or action.device != noop.device:
            raise LatentTemporalEventCriticError("action/no-op hidden geometry differs")
        return torch.einsum(
            "btpd,kp->btkd", action - noop, self.spatial_sketch
        )

    def _canonical_sketched_residual(
        self, value: Any, *, require_input_grad: bool
    ) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type == "meta"
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or value.ndim != 4
            or tuple(map(int, value.shape[1:3]))
            != (EXPECTED_PHASES, self.config.spatial_coordinates)
            or int(value.shape[-1]) != self.config.hidden_size
            or not bool(torch.isfinite(value).all().item())
        ):
            raise LatentTemporalEventCriticError(
                "sketched residual must be finite [B,21,K,D] with registered geometry"
            )
        if require_input_grad and (not value.requires_grad or value.grad_fn is None):
            raise LatentTemporalEventCriticError(
                "live sketched residual must remain graph-connected"
            )
        return value.float()

    @staticmethod
    def _lag(value: torch.Tensor, amount: int) -> torch.Tensor:
        leading = torch.zeros_like(value[:, :amount])
        return torch.cat((leading, value[:, amount:] - value[:, :-amount]), dim=1)

    def _temporal_bundle(self, value: torch.Tensor) -> torch.Tensor:
        causal = value - value[:, :1]
        causal = torch.cat((torch.zeros_like(causal[:, :1]), causal[:, 1:]), dim=1)
        lag1 = self._lag(causal, 1)
        lag2 = self._lag(causal, 2)
        lag4 = self._lag(causal, 4)
        terminal = causal[:, TERMINAL_HOLD_START:].mean(dim=1, keepdim=True)
        result = torch.cat((causal, lag1, lag2, lag4, terminal), dim=1)
        if int(result.shape[1]) != TEMPORAL_FEATURE_STEPS:
            raise LatentTemporalEventCriticError("temporal direct sum differs")
        return result

    def forward_sketched_residual(
        self,
        sketched_residual: torch.Tensor,
        *,
        require_input_grad: bool = False,
    ) -> CriticOutput:
        sketched = self._canonical_sketched_residual(
            sketched_residual, require_input_grad=require_input_grad
        )
        sketched = self._project_nuisance(sketched)
        temporal = self._temporal_bundle(sketched)
        projected = self.channel_norm(self.channel_projection(temporal))
        batch = int(projected.shape[0])
        tokens = self.sketch_mixer(projected.reshape(batch, TEMPORAL_FEATURE_STEPS, -1))
        tokens = (
            tokens
            + self.temporal_position[None]
            + self.temporal_block[self.temporal_block_ids][None]
        )
        tokens = self.temporal_encoder(tokens)
        queries = self.milestone_queries[None].expand(batch, -1, -1)
        attended, _ = self.milestone_attention(
            queries, tokens, tokens, need_weights=False
        )
        milestone_scores = self.milestone_head(
            self.milestone_norm(attended + queries)
        ).squeeze(-1)
        temperature = float(self.config.softmin_temperature)
        score = -temperature * torch.logsumexp(
            -milestone_scores / temperature, dim=-1
        )
        if require_input_grad and (not score.requires_grad or score.grad_fn is None):
            raise LatentTemporalEventCriticError("critic score detached from live hidden")
        return CriticOutput(
            milestone_scores=milestone_scores,
            score=score,
            temporal_tokens=tokens,
        )

    def forward(
        self,
        action_hidden: torch.Tensor,
        noop_hidden: torch.Tensor,
        *,
        require_input_grad: bool = False,
    ) -> CriticOutput:
        sketched = self.sketch_same_state_hidden_residual(
            action_hidden,
            noop_hidden,
            require_input_grad=require_input_grad,
        )
        return self.forward_sketched_residual(
            sketched, require_input_grad=require_input_grad
        )


@dataclass(frozen=True)
class RankingLossConfig:
    global_margin: float = 0.50
    milestone_margin: float = 0.25
    temperature: float = 0.50

    def validate(self) -> None:
        _finite_positive(self.global_margin, label="global margin")
        _finite_positive(self.milestone_margin, label="milestone margin")
        _finite_positive(self.temperature, label="ranking temperature")


@dataclass(frozen=True)
class GroupRankingLoss:
    loss: torch.Tensor
    positive_score: torch.Tensor
    negative_scores: torch.Tensor
    global_margins: torch.Tensor
    per_role_global_losses: torch.Tensor
    milestone_losses: torch.Tensor
    role_order: tuple[str, ...]


def group_ranking_loss(
    critic: FrozenHiddenTemporalEventCritic,
    positive_pair: tuple[torch.Tensor, torch.Tensor],
    negative_pairs: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    config: RankingLossConfig = RankingLossConfig(),
) -> GroupRankingLoss:
    """Rank one event-qualified owner over every hard negative in its cell.

    There is intentionally no absolute positive/negative classification loss:
    only within-cell differences can update the head, removing identity/scene/
    seed-specific score offsets.
    """

    if not isinstance(critic, FrozenHiddenTemporalEventCritic):
        raise LatentTemporalEventCriticError("critic type differs")
    config.validate()
    if not isinstance(negative_pairs, Mapping) or tuple(negative_pairs) != data_contract.NEGATIVE_ROLES:
        raise LatentTemporalEventCriticError("negative pair order/closure differs")
    if (
        not isinstance(positive_pair, tuple)
        or len(positive_pair) != 2
        or any(
            not isinstance(pair, tuple) or len(pair) != 2
            for pair in negative_pairs.values()
        )
    ):
        raise LatentTemporalEventCriticError("hidden pairs must be explicit action/no-op tuples")
    positive = critic(*positive_pair)
    negatives = [critic(*negative_pairs[role]) for role in data_contract.NEGATIVE_ROLES]
    negative_scores = torch.stack([row.score for row in negatives], dim=0)
    global_margins = positive.score[None] - negative_scores
    temperature = float(config.temperature)
    global_losses = functional.softplus(
        (float(config.global_margin) - global_margins) / temperature
    )
    milestone_index = {name: index for index, name in enumerate(MILESTONE_NAMES)}
    milestone_losses = []
    for role, negative in zip(data_contract.NEGATIVE_ROLES, negatives):
        indices = [milestone_index[name] for name in ROLE_MILESTONES[role]]
        margin = (
            positive.milestone_scores[:, indices]
            - negative.milestone_scores[:, indices]
        )
        milestone_losses.append(
            functional.softplus(
                (float(config.milestone_margin) - margin) / temperature
            ).mean()
        )
    milestone_loss_tensor = torch.stack(milestone_losses)
    loss = global_losses.mean() + milestone_loss_tensor.mean()
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise LatentTemporalEventCriticError("group ranking loss is not finite scalar")
    return GroupRankingLoss(
        loss=loss,
        positive_score=positive.score,
        negative_scores=negative_scores,
        global_margins=global_margins,
        per_role_global_losses=global_losses,
        milestone_losses=milestone_loss_tensor,
        role_order=data_contract.NEGATIVE_ROLES,
    )


def group_ranking_loss_from_sketched_residuals(
    critic: FrozenHiddenTemporalEventCritic,
    positive_residual: torch.Tensor,
    negative_residuals: Mapping[str, torch.Tensor],
    *,
    config: RankingLossConfig = RankingLossConfig(),
) -> GroupRankingLoss:
    """Memory-efficient equivalent using sealed fixed-sketch artifacts."""

    if not isinstance(critic, FrozenHiddenTemporalEventCritic):
        raise LatentTemporalEventCriticError("critic type differs")
    config.validate()
    if (
        not isinstance(negative_residuals, Mapping)
        or tuple(negative_residuals) != data_contract.NEGATIVE_ROLES
    ):
        raise LatentTemporalEventCriticError("negative residual order/closure differs")
    positive = critic.forward_sketched_residual(positive_residual)
    negatives = [
        critic.forward_sketched_residual(negative_residuals[role])
        for role in data_contract.NEGATIVE_ROLES
    ]
    negative_scores = torch.stack([row.score for row in negatives], dim=0)
    global_margins = positive.score[None] - negative_scores
    temperature = float(config.temperature)
    global_losses = functional.softplus(
        (float(config.global_margin) - global_margins) / temperature
    )
    milestone_index = {name: index for index, name in enumerate(MILESTONE_NAMES)}
    milestone_losses = []
    for role, negative in zip(data_contract.NEGATIVE_ROLES, negatives):
        indices = [milestone_index[name] for name in ROLE_MILESTONES[role]]
        margin = (
            positive.milestone_scores[:, indices]
            - negative.milestone_scores[:, indices]
        )
        milestone_losses.append(
            functional.softplus(
                (float(config.milestone_margin) - margin) / temperature
            ).mean()
        )
    milestone_loss_tensor = torch.stack(milestone_losses)
    loss = global_losses.mean() + milestone_loss_tensor.mean()
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise LatentTemporalEventCriticError("group ranking loss is not finite scalar")
    return GroupRankingLoss(
        loss=loss,
        positive_score=positive.score,
        negative_scores=negative_scores,
        global_margins=global_margins,
        per_role_global_losses=global_losses,
        milestone_losses=milestone_loss_tensor,
        role_order=data_contract.NEGATIVE_ROLES,
    )


@dataclass(frozen=True)
class FrozenHiddenQuery:
    backend_id: str
    role: str
    condition: Any
    x_sigma: torch.Tensor
    sigma: float
    native_timestep: int
    hook_coordinate: str
    condition_mode: str
    adapter_enabled: bool
    source_condition: None


@dataclass(frozen=True)
class LiveCriticScore:
    output: CriticOutput
    x_sigma: torch.Tensor
    action_hidden: torch.Tensor
    noop_hidden: torch.Tensor
    call_order: tuple[str, str]


def score_current_rv2v_clean_latent(
    critic: FrozenHiddenTemporalEventCritic,
    clean_latent: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    action_condition: Any,
    noop_condition: Any,
    frozen_hidden_callback: Callable[[FrozenHiddenQuery], torch.Tensor],
    sigma: float = data_contract.PILOT_HIDDEN_QUERY["sigma"],
    native_timestep: int = data_contract.PILOT_HIDDEN_QUERY["native_timestep"],
    hook_coordinate: str = data_contract.PILOT_HIDDEN_QUERY["hook_coordinate"],
) -> LiveCriticScore:
    """Evaluate a live RV2V candidate through frozen T2V hidden cross-query.

    ``clean_latent`` is the current editor output, not a generated teacher
    target.  The callback API deliberately has no source/mask/flow/pose slot.
    Both calls receive the same Python ``x_sigma`` object and remain in the
    input graph.  A caller should use ``autograd.grad(score.sum(), clean)`` and
    must separately enforce source identity/camera/quality constraints.
    """

    if not isinstance(critic, FrozenHiddenTemporalEventCritic):
        raise LatentTemporalEventCriticError("critic type differs")
    if (
        not isinstance(clean_latent, torch.Tensor)
        or clean_latent.dtype != torch.float32
        or clean_latent.ndim != 5
        or tuple(map(int, clean_latent.shape[:3]))
        != (1, data_contract.LATENT_CHANNELS, EXPECTED_PHASES)
        or not clean_latent.requires_grad
        or not bool(torch.isfinite(clean_latent).all().item())
    ):
        raise LatentTemporalEventCriticError(
            "live clean latent must be finite graph-connected FP32 [1,16,21,H,W]"
        )
    try:
        native_geometry = data_contract.derive_native_geometry(
            [int(item) for item in clean_latent.shape]
        )
    except data_contract.LatentTemporalEventDatasetError as error:
        raise LatentTemporalEventCriticError(str(error)) from error
    if critic.config.patch_positions != native_geometry["patch_positions"]:
        raise LatentTemporalEventCriticError(
            "live clean latent patch count differs from the critic sketch instance"
        )
    if (
        not isinstance(epsilon, torch.Tensor)
        or epsilon.dtype != torch.float32
        or epsilon.shape != clean_latent.shape
        or epsilon.device != clean_latent.device
        or epsilon.requires_grad
        or epsilon.grad_fn is not None
        or not bool(torch.isfinite(epsilon).all().item())
    ):
        raise LatentTemporalEventCriticError(
            "epsilon must be detached finite FP32 with exact clean geometry"
        )
    sigma_value = _finite_positive(sigma, label="sigma")
    if sigma_value >= 1.0 or type(native_timestep) is not int or native_timestep <= 0:
        raise LatentTemporalEventCriticError("live query coordinate differs")
    if not isinstance(hook_coordinate, str) or not hook_coordinate:
        raise LatentTemporalEventCriticError("hook coordinate must be nonempty")
    if not callable(frozen_hidden_callback):
        raise LatentTemporalEventCriticError("frozen hidden callback must be callable")
    x_sigma = (1.0 - sigma_value) * clean_latent + sigma_value * epsilon
    before = x_sigma.detach().clone()

    def query(role: str, condition: Any) -> torch.Tensor:
        request = FrozenHiddenQuery(
            backend_id=BACKEND_ID,
            role=role,
            condition=condition,
            x_sigma=x_sigma,
            sigma=sigma_value,
            native_timestep=native_timestep,
            hook_coordinate=hook_coordinate,
            condition_mode="t2v_same_state_target_tail",
            adapter_enabled=False,
            source_condition=None,
        )
        value = frozen_hidden_callback(request)
        if not torch.equal(x_sigma.detach(), before):
            raise LatentTemporalEventCriticError("hidden callback mutated shared x_sigma")
        return value

    action_hidden = query("action", action_condition)
    noop_hidden = query("noop", noop_condition)
    output = critic(action_hidden, noop_hidden, require_input_grad=True)
    if not output.score.requires_grad or output.score.grad_fn is None:
        raise LatentTemporalEventCriticError("live critic score detached from clean latent")
    return LiveCriticScore(
        output=output,
        x_sigma=x_sigma,
        action_hidden=action_hidden,
        noop_hidden=noop_hidden,
        call_order=("action", "noop"),
    )


def critic_contract_receipt(
    critic: FrozenHiddenTemporalEventCritic,
) -> dict[str, Any]:
    """Return the static interpretation and parameter budget of one head."""

    if not isinstance(critic, FrozenHiddenTemporalEventCritic):
        raise LatentTemporalEventCriticError("critic type differs")
    return {
        "schema_version": SCHEMA_VERSION,
        "backend_id": BACKEND_ID,
        "hidden_query_contract": dict(data_contract.PILOT_HIDDEN_QUERY),
        "spatial_policy": "fixed_full-support_counter-rademacher_no_mask_v1",
        "patch_positions": critic.config.patch_positions,
        "spatial_sketch_seed": critic.config.spatial_sketch_seed,
        "patch_positions_are_episode_specific": True,
        "spatial_sketch_digest": critic.spatial_sketch_digest,
        "nuisance_basis_digest": critic.nuisance_basis_digest,
        "nuisance_basis_required": critic.config.require_nuisance_basis,
        "production_geometry": critic.config.production_geometry,
        "temporal_features": [
            "phase0_boundary",
            "lag1",
            "lag2",
            "lag4",
            "terminal_hold_phases_17_20",
        ],
        "milestone_order": list(MILESTONE_NAMES),
        "negative_role_order": list(data_contract.NEGATIVE_ROLES),
        "trainable_parameter_count": critic.trainable_parameter_count,
        "generated_video_or_latent_is_editor_target": False,
        "score_is_differentiable_wrt_current_rv2v_clean_latent": True,
        "score_alone_authorizes_editor_optimizer": False,
    }
