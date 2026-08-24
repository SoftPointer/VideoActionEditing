#!/usr/bin/env python3
"""Minimal optimizer surface for the frozen-hidden Bernini event critic.

Only the small critic head is trainable.  Bernini hidden artifacts are detached
inputs during critic fitting; generated videos/latents never become editor
targets.  At reward time the fitted head and Bernini are both frozen and
``latent_temporal_event_critic.score_current_rv2v_clean_latent`` rebuilds the
input graph to the current RV2V clean latent.

This file intentionally has no Slurm launcher and no end-to-end editor trainer.
The current core4 population may run only the two-cell fit/two-cell
confirmation pilot gate and can never authorize an editor update.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

import latent_temporal_event_critic as critic_core
import latent_temporal_event_critic_dataset as data_contract


SCHEMA_VERSION = "bernini-latent-temporal-event-critic-minimal-trainer-v1"


class LatentTemporalEventTrainerError(RuntimeError):
    """The critic-only optimizer or group batch violated its closure."""


@dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-2
    maximum_gradient_norm: float = 1.0
    global_margin: float = 0.50
    milestone_margin: float = 0.25
    ranking_temperature: float = 0.50

    def validate(self) -> None:
        for name in (
            "learning_rate",
            "maximum_gradient_norm",
            "global_margin",
            "milestone_margin",
            "ranking_temperature",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise LatentTemporalEventTrainerError(f"{name} must be positive finite")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(float(self.weight_decay))
            or float(self.weight_decay) < 0.0
        ):
            raise LatentTemporalEventTrainerError("weight_decay must be finite nonnegative")


@dataclass(frozen=True)
class CriticGroupBatch:
    """One cell: an event-qualified positive and all twelve hard negatives."""

    episode_id: str
    positive_sketched_residual: torch.Tensor
    negative_sketched_residuals: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class CriticTrainStep:
    loss: float
    gradient_norm: float
    minimum_group_margin: float
    episode_ids: tuple[str, ...]
    optimizer_step_performed: bool
    editor_parameter_present: bool


def build_critic_optimizer(
    critic: critic_core.FrozenHiddenTemporalEventCritic,
    *,
    config: TrainerConfig = TrainerConfig(),
) -> torch.optim.Optimizer:
    """Build AdamW over exactly the small head's parameters."""

    if not isinstance(critic, critic_core.FrozenHiddenTemporalEventCritic):
        raise LatentTemporalEventTrainerError("critic type differs")
    config.validate()
    parameters = [parameter for parameter in critic.parameters() if parameter.requires_grad]
    if not parameters or sum(parameter.numel() for parameter in parameters) != critic.trainable_parameter_count:
        raise LatentTemporalEventTrainerError("critic trainable parameter closure differs")
    return torch.optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )


def _validate_optimizer_scope(
    critic: critic_core.FrozenHiddenTemporalEventCritic,
    optimizer: torch.optim.Optimizer,
) -> None:
    expected = {id(parameter) for parameter in critic.parameters() if parameter.requires_grad}
    observed = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    }
    if observed != expected:
        raise LatentTemporalEventTrainerError(
            "optimizer must contain every and only critic-head parameter"
        )


def _validate_sketched_residual(
    value: Any,
    *,
    label: str,
    hidden_size: int,
    spatial_coordinates: int,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 4
        or tuple(map(int, value.shape[1:3]))
        != (data_contract.LATENT_PHASES, spatial_coordinates)
        or int(value.shape[-1]) != hidden_size
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise LatentTemporalEventTrainerError(
            f"{label} must be a detached finite FP32 fixed-sketch residual"
        )
    return value


def _validate_group(
    group: Any, *, hidden_size: int, spatial_coordinates: int
) -> CriticGroupBatch:
    if not isinstance(group, CriticGroupBatch):
        raise LatentTemporalEventTrainerError("group must be CriticGroupBatch")
    if (
        not isinstance(group.episode_id, str)
        or not group.episode_id
        or tuple(group.negative_sketched_residuals) != data_contract.NEGATIVE_ROLES
    ):
        raise LatentTemporalEventTrainerError("group identity/negative closure differs")
    _validate_sketched_residual(
        group.positive_sketched_residual,
        label=f"{group.episode_id}/positive",
        hidden_size=hidden_size,
        spatial_coordinates=spatial_coordinates,
    )
    for role in data_contract.NEGATIVE_ROLES:
        _validate_sketched_residual(
            group.negative_sketched_residuals[role],
            label=f"{group.episode_id}/{role}",
            hidden_size=hidden_size,
            spatial_coordinates=spatial_coordinates,
        )
    return group


def train_critic_groups_one_step(
    critic: critic_core.FrozenHiddenTemporalEventCritic,
    optimizer: torch.optim.Optimizer,
    groups: Sequence[CriticGroupBatch],
    *,
    config: TrainerConfig = TrainerConfig(),
) -> CriticTrainStep:
    """Perform one critic-only update over complete within-cell groups."""

    if not isinstance(critic, critic_core.FrozenHiddenTemporalEventCritic):
        raise LatentTemporalEventTrainerError("critic type differs")
    config.validate()
    _validate_optimizer_scope(critic, optimizer)
    checked = [
        _validate_group(
            group,
            hidden_size=critic.config.hidden_size,
            spatial_coordinates=critic.config.spatial_coordinates,
        )
        for group in groups
    ]
    if not checked or len({group.episode_id for group in checked}) != len(checked):
        raise LatentTemporalEventTrainerError("training step requires unique nonempty groups")
    ranking_config = critic_core.RankingLossConfig(
        global_margin=float(config.global_margin),
        milestone_margin=float(config.milestone_margin),
        temperature=float(config.ranking_temperature),
    )
    critic.train()
    optimizer.zero_grad(set_to_none=True)
    results = [
        critic_core.group_ranking_loss_from_sketched_residuals(
            critic,
            group.positive_sketched_residual,
            group.negative_sketched_residuals,
            config=ranking_config,
        )
        for group in checked
    ]
    loss = torch.stack([result.loss for result in results]).mean()
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise LatentTemporalEventTrainerError("aggregate critic loss is invalid")
    loss.backward()
    parameters = [parameter for parameter in critic.parameters() if parameter.requires_grad]
    if not any(parameter.grad is not None for parameter in parameters):
        raise LatentTemporalEventTrainerError("critic loss produced no parameter gradient")
    if any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
    ):
        raise LatentTemporalEventTrainerError("critic gradient is non-finite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters, float(config.maximum_gradient_norm)
    )
    if not bool(torch.isfinite(gradient_norm).item()) or float(gradient_norm.item()) <= 0.0:
        raise LatentTemporalEventTrainerError("critic gradient norm is invalid")
    optimizer.step()
    minimum_margin = min(
        float(result.global_margins.detach().min().item()) for result in results
    )
    return CriticTrainStep(
        loss=float(loss.detach().item()),
        gradient_norm=float(gradient_norm.detach().item()),
        minimum_group_margin=minimum_margin,
        episode_ids=tuple(group.episode_id for group in checked),
        optimizer_step_performed=True,
        editor_parameter_present=False,
    )


def freeze_fitted_critic_for_reward(
    critic: critic_core.FrozenHiddenTemporalEventCritic,
) -> critic_core.FrozenHiddenTemporalEventCritic:
    """Freeze/eval the fitted head before it can score an editor candidate."""

    if not isinstance(critic, critic_core.FrozenHiddenTemporalEventCritic):
        raise LatentTemporalEventTrainerError("critic type differs")
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in critic.parameters()):
        raise LatentTemporalEventTrainerError("critic did not freeze")
    return critic


def audit_current_clean_latent_gradient(
    live_score: critic_core.LiveCriticScore,
    clean_latent: torch.Tensor,
    *,
    minimum_norm: float = 1.0e-12,
) -> dict[str, Any]:
    """Prove that the frozen live reward has a finite nonzero input VJP."""

    threshold = float(minimum_norm)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise LatentTemporalEventTrainerError("minimum gradient norm must be positive")
    if not isinstance(live_score, critic_core.LiveCriticScore):
        raise LatentTemporalEventTrainerError("live score type differs")
    gradient = torch.autograd.grad(
        live_score.output.score.sum(), clean_latent, retain_graph=False, create_graph=False
    )[0]
    norm = torch.linalg.vector_norm(gradient.float())
    finite = bool(torch.isfinite(gradient).all().item()) and bool(torch.isfinite(norm).item())
    nonzero = finite and float(norm.item()) >= threshold
    return {
        "schema_version": "bernini-ltec-current-clean-latent-gradient-audit-v1",
        "gradient_shape": list(gradient.shape),
        "gradient_norm": float(norm.item()) if finite else None,
        "finite": finite,
        "nonzero": nonzero,
        "minimum_norm": threshold,
        "passed": finite and nonzero,
        "generated_t2v_target_consumed": False,
        "editor_optimizer_authorized": False,
    }


def trainer_contract_receipt(
    critic: critic_core.FrozenHiddenTemporalEventCritic,
    *,
    config: TrainerConfig = TrainerConfig(),
) -> dict[str, Any]:
    config.validate()
    return {
        "schema_version": SCHEMA_VERSION,
        "critic_contract": critic_core.critic_contract_receipt(critic),
        "optimizer": "AdamW_critic_head_only",
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "maximum_gradient_norm": float(config.maximum_gradient_norm),
        "ranking": {
            "positive_source": "externally_event_qualified_action_owner_only",
            "negative_role_order": list(data_contract.NEGATIVE_ROLES),
            "within_cell_only": True,
            "absolute_classification_loss": False,
            "global_margin": float(config.global_margin),
            "milestone_margin": float(config.milestone_margin),
            "temperature": float(config.ranking_temperature),
        },
        "generated_media_is_editor_target": False,
        "core4_may_authorize": "fixed_topup_recommendation_only",
        "core4_may_authorize_editor_optimizer": False,
    }
