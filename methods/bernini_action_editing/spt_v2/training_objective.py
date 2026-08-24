#!/usr/bin/env python3
"""Training objective shared by SPT oracle and student experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .phase_transport import (
    PhasePlan,
    PhaseTransportAdapter,
    PhaseTransportConfig,
    PhaseTransportError,
    build_oracle_plan,
    clean_from_velocity,
    execute_clean_plan,
    make_proxy_target,
    plan_distillation_loss,
    velocity_from_clean,
)


@dataclass(frozen=True)
class SPTLossConfig:
    flow_weight: float = 1.0
    gate_distill_weight: float = 0.25
    offset_distill_weight: float = 0.10
    plan_smooth_weight: float = 0.01
    generation_budget_weight: float = 0.02
    low_noise_floor: float = 0.25
    low_noise_power: float = 2.0

    def validate(self) -> None:
        for name in (
            "flow_weight",
            "gate_distill_weight",
            "offset_distill_weight",
            "plan_smooth_weight",
            "generation_budget_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise PhaseTransportError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.low_noise_floor <= 1.0:
            raise PhaseTransportError("low_noise_floor must lie in [0,1]")
        if not math.isfinite(self.low_noise_power) or self.low_noise_power <= 0.0:
            raise PhaseTransportError("low_noise_power must be finite and positive")


def low_noise_weight(sigma: Any, *, floor: float, power: float) -> Any:
    """Favor texture/topology-forming late denoising while retaining all sigma."""

    return floor + (1.0 - floor) * (1.0 - sigma.float().clamp(0.0, 1.0)).pow(power)


def compute_spt_loss(
    *,
    source_clean: Any,
    target_clean: Any,
    noisy_target: Any,
    sigma: Any,
    base_velocity: Any,
    student_plan: PhasePlan,
    oracle_plan: PhasePlan,
    config: SPTLossConfig,
) -> tuple[Any, dict[str, Any]]:
    """Train what is integrated: executed raw velocity, not a quotient proxy.

    The full synthetic target is never the default flow target.  The oracle
    proxy takes explainable content from the source bank and exposes target
    latents only through the teacher's generate gate.
    """

    import torch

    config.validate()
    generated_clean = clean_from_velocity(noisy_target, base_velocity, sigma)
    proxy_clean = make_proxy_target(source_clean, target_clean, oracle_plan)
    executed_clean = execute_clean_plan(source_clean, generated_clean, student_plan)
    executed_velocity = velocity_from_clean(noisy_target, executed_clean, sigma)
    proxy_velocity = velocity_from_clean(noisy_target, proxy_clean, sigma)
    weight = low_noise_weight(
        sigma, floor=config.low_noise_floor, power=config.low_noise_power
    ).mean()
    flow = torch.mean((executed_velocity.float() - proxy_velocity.float()) ** 2)
    distilled = plan_distillation_loss(student_plan, oracle_plan, source_clean)
    generate_budget = student_plan.gate_probs[:, 2].float().mean()
    total = (
        config.flow_weight * weight * flow
        + config.gate_distill_weight * distilled["gate"]
        + config.offset_distill_weight * distilled["offset"]
        + config.plan_smooth_weight * distilled["smooth"]
        + config.generation_budget_weight * generate_budget
    )
    return total, {
        "flow_proxy": flow,
        "low_noise_weight": weight,
        "gate_distill": distilled["gate"],
        "offset_distill": distilled["offset"],
        "plan_smooth": distilled["smooth"],
        "generation_budget": generate_budget,
        "oracle_generate_fraction": oracle_plan.gate_probs[:, 2].float().mean(),
    }


try:
    from torch import nn

    class SPTTrainingHead(nn.Module):
        """Trainable small planner; Bernini LoRA remains a separate PEFT model."""

        def __init__(
            self,
            planner_config: PhaseTransportConfig,
            loss_config: SPTLossConfig = SPTLossConfig(),
        ):
            super().__init__()
            self.planner = PhaseTransportAdapter(planner_config)
            self.planner_config = planner_config
            self.loss_config = loss_config
            loss_config.validate()

        def forward(
            self,
            *,
            source_clean: Any,
            target_clean: Any,
            noisy_target: Any,
            sigma: Any,
            base_velocity: Any,
            instruction_embedding: Any,
            plan_mode: str = "student",
        ) -> tuple[Any, dict[str, Any], PhasePlan]:
            if plan_mode not in ("student", "oracle"):
                raise PhaseTransportError("plan_mode must be student or oracle")
            oracle = build_oracle_plan(source_clean, target_clean, self.planner_config)
            student = self.planner(source_clean, instruction_embedding)
            execution_plan = student if plan_mode == "student" else oracle
            total, metrics = compute_spt_loss(
                source_clean=source_clean,
                target_clean=target_clean,
                noisy_target=noisy_target,
                sigma=sigma,
                base_velocity=base_velocity,
                student_plan=student,
                oracle_plan=oracle,
                config=self.loss_config,
            )
            # Oracle mode is an execution upper-bound diagnostic; the same
            # student distillation loss is reported/trained, while callers use
            # execution_plan for video sampling.
            metrics["plan_mode_oracle"] = total.new_tensor(float(plan_mode == "oracle"))
            return total, metrics, execution_plan

except ImportError:

    class SPTTrainingHead:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise PhaseTransportError("SPTTrainingHead requires PyTorch")
