"""Pure-PyTorch motion-only objective for few-shot Bernini teacher inversion.

The privileged target is available only while inverting a compact action code.
It is never returned as a condition and this module has no API for a target
mask, optical flow, pose, track, trajectory, or full-target flow-matching
loss.  Instead, the target supplies a source-relative temporal feature:

1. subtract the clean source latent;
2. remove the temporal DC component (``Q0``);
3. pool every latent phase to an 8 x 8 spatial grid;
4. take temporal differences at lags 1, 2, and 4; and
5. normalize each resulting phase by its RMS and clip outliers.

All objective arithmetic is FP32.  The first executable teacher uses the
auditable 36-dimensional ``phase20 + block16`` action code.  Attention-head
gating belongs at the projected-head boundary and is deliberately outside this
module; adjacent chunks of a pre-projection 1536-vector are not attention
heads.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F


METHOD_NAME = "fewshot-privileged-motion-teacher-objective"
SCHEMA_VERSION = "bernini-fewshot-teacher-objective-v1"

LATENT_CHANNELS = 16
LATENT_PHASES = 21
NONBOUNDARY_PHASES = LATENT_PHASES - 1
MOTION_BLOCKS = 16
ACTION_CODE_DIM = NONBOUNDARY_PHASES + MOTION_BLOCKS

POOL_HEIGHT = 8
POOL_WIDTH = 8
TEMPORAL_LAGS = (1, 2, 4)
RMS_EPSILON = 1.0e-6
CHARBONNIER_EPSILON = 1.0e-3
NORMALIZED_FEATURE_CLIP = 4.0
GATE_SATURATION_THRESHOLD = 0.95

FEATURE_MATCH_WEIGHT = 1.0
TEMPORAL_DC_WEIGHT = 0.25
PHASE0_PARITY_WEIGHT = 0.25
PHASE_RMS_WEIGHT = 0.10
GATE_L2_WEIGHT = 1.0e-3

# Generated previews that saw only the source first frame must not become a
# full-target reconstruction objective.  There is intentionally no API through
# which a caller can supply or weight a flow-matching loss.
FULL_TARGET_FLOW_MATCHING_WEIGHT = 0.0

GO_MIN_ZERO_IMPROVEMENT = 0.15
GO_MIN_CONTROL_IMPROVEMENT = 0.05
GO_MAX_SATURATION_FRACTION = 0.25  # strict: saturation must be < 25%
GO_MIN_SUPPORT_CODE_COSINE = 0.60

REVERSE_PHASE_INDICES = (0, *tuple(range(LATENT_PHASES - 1, 0, -1)))
SHUFFLE_PHASE_INDICES = (
    0,
    17,
    18,
    1,
    6,
    16,
    4,
    12,
    11,
    7,
    13,
    19,
    2,
    15,
    8,
    3,
    9,
    20,
    5,
    10,
    14,
)


class FewShotTeacherObjectiveError(ValueError):
    """A latent, gate, metric, or frozen objective contract is invalid."""


def _require_fp32_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise FewShotTeacherObjectiveError(f"{name} must be a torch.Tensor")
    if value.device.type == "meta":
        raise FewShotTeacherObjectiveError(f"{name} cannot be a meta tensor")
    if value.dtype != torch.float32:
        raise FewShotTeacherObjectiveError(f"{name} must be float32")
    if not bool(torch.isfinite(value).all().item()):
        raise FewShotTeacherObjectiveError(f"{name} contains NaN or infinity")
    return value


def _validate_clean_video(
    name: str,
    value: Any,
    *,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    result = _require_fp32_tensor(name, value)
    if result.ndim != 5:
        raise FewShotTeacherObjectiveError(
            f"{name} must have exact layout [B,16,21,H,W]"
        )
    batch, channels, phases, height, width = map(int, result.shape)
    if batch < 1 or channels != LATENT_CHANNELS or phases != LATENT_PHASES:
        raise FewShotTeacherObjectiveError(
            f"{name} must have exact layout [B,16,21,H,W]"
        )
    if height < POOL_HEIGHT or width < POOL_WIDTH:
        raise FewShotTeacherObjectiveError(
            f"{name} spatial dimensions must both be at least 8"
        )
    if reference is not None:
        if result.shape != reference.shape:
            raise FewShotTeacherObjectiveError(
                f"{name} shape differs from source_clean"
            )
        if result.device != reference.device:
            raise FewShotTeacherObjectiveError(
                f"{name} device differs from source_clean"
            )
    return result


def _positive_zero(name: str, value: torch.Tensor) -> None:
    if int(torch.count_nonzero(value).item()) != 0:
        raise FewShotTeacherObjectiveError(f"{name} must be exact zero")
    payload = value.detach().reshape(-1).repeat(1).view(torch.uint8)
    if int(torch.count_nonzero(payload).item()) != 0:
        raise FewShotTeacherObjectiveError(
            f"{name} must be byte-exact positive zero"
        )


def _validate_gates(
    phase_gates: Any,
    block_gates: Any,
    *,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = _require_fp32_tensor("phase_gates", phase_gates)
    block = _require_fp32_tensor("block_gates", block_gates)
    if phase.ndim != 2 or tuple(phase.shape[1:]) != (LATENT_PHASES,):
        raise FewShotTeacherObjectiveError("phase_gates must be [B,21]")
    if block.ndim != 2 or tuple(block.shape[1:]) != (MOTION_BLOCKS,):
        raise FewShotTeacherObjectiveError("block_gates must be [B,16]")
    if int(phase.shape[0]) < 1 or phase.shape[0] != block.shape[0]:
        raise FewShotTeacherObjectiveError("gate batch dimensions must agree")
    if batch_size is not None and int(phase.shape[0]) != int(batch_size):
        raise FewShotTeacherObjectiveError("gate batch differs from video batch")
    if phase.device != block.device:
        raise FewShotTeacherObjectiveError("gate tensors must share one device")
    if bool((phase.abs() > 1.0).any().item()) or bool(
        (block.abs() > 1.0).any().item()
    ):
        raise FewShotTeacherObjectiveError("gates must remain in [-1,1]")
    _positive_zero("phase_gates[:,0]", phase[:, 0])
    return phase, block


def _charbonnier_difference(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return a zero-at-equality robust Charbonnier mean."""

    difference = left - right
    epsilon = difference.new_tensor(CHARBONNIER_EPSILON)
    return (torch.sqrt(difference.square() + epsilon.square()) - epsilon).mean()


@dataclass(frozen=True)
class SourceRelativeMotionFeatures:
    """Normalized temporal features and their pre-normalization phase RMS."""

    pooled_q0: torch.Tensor
    normalized_by_lag: tuple[torch.Tensor, ...]
    phase_rms_by_lag: tuple[torch.Tensor, ...]
    lags: tuple[int, ...] = TEMPORAL_LAGS

    def validate(self) -> None:
        pooled = _require_fp32_tensor("pooled_q0", self.pooled_q0)
        if pooled.ndim != 5 or tuple(pooled.shape[1:3]) != (
            LATENT_CHANNELS,
            LATENT_PHASES,
        ) or tuple(pooled.shape[-2:]) != (POOL_HEIGHT, POOL_WIDTH):
            raise FewShotTeacherObjectiveError(
                "pooled_q0 must be [B,16,21,8,8]"
            )
        if self.lags != TEMPORAL_LAGS:
            raise FewShotTeacherObjectiveError("feature lag registry differs")
        if len(self.normalized_by_lag) != len(self.lags) or len(
            self.phase_rms_by_lag
        ) != len(self.lags):
            raise FewShotTeacherObjectiveError("feature tuple lengths differ")
        batch = int(pooled.shape[0])
        for lag, normalized, phase_rms in zip(
            self.lags, self.normalized_by_lag, self.phase_rms_by_lag
        ):
            _require_fp32_tensor(f"normalized_lag_{lag}", normalized)
            _require_fp32_tensor(f"phase_rms_lag_{lag}", phase_rms)
            expected_phases = LATENT_PHASES - lag
            if tuple(normalized.shape) != (
                batch,
                LATENT_CHANNELS,
                expected_phases,
                POOL_HEIGHT,
                POOL_WIDTH,
            ):
                raise FewShotTeacherObjectiveError(
                    f"normalized lag-{lag} feature shape differs"
                )
            if tuple(phase_rms.shape) != (batch, expected_phases):
                raise FewShotTeacherObjectiveError(
                    f"lag-{lag} phase RMS shape differs"
                )
            if bool((phase_rms < 0).any().item()):
                raise FewShotTeacherObjectiveError("phase RMS cannot be negative")
            if bool((normalized.abs() > NORMALIZED_FEATURE_CLIP).any().item()):
                raise FewShotTeacherObjectiveError(
                    "normalized temporal feature escaped the clip bound"
                )


def source_relative_motion_features(
    candidate_clean: torch.Tensor,
    source_clean: torch.Tensor,
) -> SourceRelativeMotionFeatures:
    """Compute the frozen source-relative motion feature ``F``.

    Both videos use exact ``[B,16,21,H,W]`` FP32 layout.  The returned tensors
    remain differentiable with respect to ``candidate_clean``.
    """

    source = _validate_clean_video("source_clean", source_clean)
    candidate = _validate_clean_video(
        "candidate_clean", candidate_clean, reference=source
    )
    relative = candidate - source
    q0 = relative - relative.mean(dim=2, keepdim=True)
    batch, channels, phases, height, width = map(int, q0.shape)
    pooled = F.adaptive_avg_pool2d(
        q0.permute(0, 2, 1, 3, 4).reshape(
            batch * phases, channels, height, width
        ),
        (POOL_HEIGHT, POOL_WIDTH),
    )
    pooled = pooled.reshape(
        batch, phases, channels, POOL_HEIGHT, POOL_WIDTH
    ).permute(0, 2, 1, 3, 4).contiguous()

    normalized_values: list[torch.Tensor] = []
    rms_values: list[torch.Tensor] = []
    for lag in TEMPORAL_LAGS:
        temporal_difference = pooled[:, :, lag:] - pooled[:, :, :-lag]
        # ``sqrt(mean(x**2))`` has an undefined derivative at exactly zero.
        # The epsilon-inside-root form keeps the inversion gradient finite,
        # while subtracting epsilon keeps the reported amplitude exactly zero
        # for a motionless phase.
        safe_rms = (
            temporal_difference.square().mean(dim=(1, 3, 4))
            + RMS_EPSILON**2
        ).sqrt()
        phase_rms = safe_rms - RMS_EPSILON
        normalized = temporal_difference / safe_rms[
            :, None, :, None, None
        ]
        normalized = normalized.clamp(
            min=-NORMALIZED_FEATURE_CLIP,
            max=NORMALIZED_FEATURE_CLIP,
        )
        normalized_values.append(normalized)
        rms_values.append(phase_rms)

    result = SourceRelativeMotionFeatures(
        pooled_q0=pooled,
        normalized_by_lag=tuple(normalized_values),
        phase_rms_by_lag=tuple(rms_values),
    )
    result.validate()
    return result


def motion_feature_match_loss(
    predicted: SourceRelativeMotionFeatures,
    target: SourceRelativeMotionFeatures,
) -> torch.Tensor:
    """Robustly match normalized motion features with equal lag weighting."""

    predicted.validate()
    target.validate()
    losses: list[torch.Tensor] = []
    for pred, wanted in zip(
        predicted.normalized_by_lag, target.normalized_by_lag
    ):
        if pred.shape != wanted.shape or pred.device != wanted.device:
            raise FewShotTeacherObjectiveError(
                "predicted and target motion feature layouts differ"
            )
        losses.append(_charbonnier_difference(pred, wanted.detach()))
    result = torch.stack(losses).mean()
    if result.ndim != 0 or not bool(torch.isfinite(result).item()):
        raise FewShotTeacherObjectiveError("motion feature loss is invalid")
    return result


def phase_rms_match_loss(
    predicted: SourceRelativeMotionFeatures,
    target: SourceRelativeMotionFeatures,
) -> torch.Tensor:
    """Match motion amplitude discarded by per-phase feature normalization."""

    predicted.validate()
    target.validate()
    losses: list[torch.Tensor] = []
    for pred, wanted in zip(predicted.phase_rms_by_lag, target.phase_rms_by_lag):
        if pred.shape != wanted.shape or pred.device != wanted.device:
            raise FewShotTeacherObjectiveError(
                "predicted and target phase-RMS layouts differ"
            )
        losses.append(_charbonnier_difference(pred, wanted.detach()))
    result = torch.stack(losses).mean()
    if result.ndim != 0 or not bool(torch.isfinite(result).item()):
        raise FewShotTeacherObjectiveError("phase RMS loss is invalid")
    return result


def temporal_dc_residual_penalty(
    predicted_clean: torch.Tensor,
    source_clean: torch.Tensor,
) -> torch.Tensor:
    source = _validate_clean_video("source_clean", source_clean)
    predicted = _validate_clean_video(
        "predicted_clean", predicted_clean, reference=source
    )
    temporal_dc = (predicted - source).mean(dim=2)
    return _charbonnier_difference(temporal_dc, torch.zeros_like(temporal_dc))


def target_phase0_base_parity_penalty(
    predicted_clean: torch.Tensor,
    source_clean: torch.Tensor,
) -> torch.Tensor:
    """Keep the predicted target's first latent phase at the source base."""

    source = _validate_clean_video("source_clean", source_clean)
    predicted = _validate_clean_video(
        "predicted_clean", predicted_clean, reference=source
    )
    return _charbonnier_difference(predicted[:, :, 0], source[:, :, 0])


def gate_l2_penalty(
    phase_gates: torch.Tensor,
    block_gates: torch.Tensor,
    *,
    batch_size: int | None = None,
) -> torch.Tensor:
    phase, block = _validate_gates(
        phase_gates, block_gates, batch_size=batch_size
    )
    flattened = torch.cat((phase[:, 1:], block), dim=1)
    result = flattened.square().mean()
    if result.ndim != 0 or not bool(torch.isfinite(result).item()):
        raise FewShotTeacherObjectiveError("gate L2 penalty is invalid")
    return result


@dataclass(frozen=True)
class TeacherObjectiveResult:
    total: torch.Tensor
    feature_match: torch.Tensor
    temporal_dc_residual: torch.Tensor
    target_phase0_base_parity: torch.Tensor
    phase_rms_match: torch.Tensor
    gate_l2: torch.Tensor
    full_target_flow_matching_weight: float = FULL_TARGET_FLOW_MATCHING_WEIGHT

    def detached_statistics(self) -> dict[str, float | str]:
        result: dict[str, float | str] = {
            "method": METHOD_NAME,
            "schema_version": SCHEMA_VERSION,
            "full_target_flow_matching_weight": self.full_target_flow_matching_weight,
        }
        for name in (
            "total",
            "feature_match",
            "temporal_dc_residual",
            "target_phase0_base_parity",
            "phase_rms_match",
            "gate_l2",
        ):
            value = getattr(self, name)
            if value.ndim != 0 or not bool(torch.isfinite(value).item()):
                raise FewShotTeacherObjectiveError(
                    f"objective statistic {name} is not a finite scalar"
                )
            result[name] = float(value.detach().item())
        return result


def fewshot_teacher_objective(
    source_clean: torch.Tensor,
    predicted_clean: torch.Tensor,
    target_clean: torch.Tensor,
    phase_gates: torch.Tensor,
    block_gates: torch.Tensor,
) -> TeacherObjectiveResult:
    """Compute the complete motion-only privileged teacher objective.

    ``target_clean`` supplies supervision here and nowhere in the inference
    path.  No absolute target reconstruction or full-target flow-matching term
    exists in this function.
    """

    source = _validate_clean_video("source_clean", source_clean)
    predicted = _validate_clean_video(
        "predicted_clean", predicted_clean, reference=source
    )
    target = _validate_clean_video("target_clean", target_clean, reference=source)
    phase, block = _validate_gates(
        phase_gates, block_gates, batch_size=int(source.shape[0])
    )
    if phase.device != source.device:
        raise FewShotTeacherObjectiveError("gates and clean videos must share one device")

    predicted_features = source_relative_motion_features(predicted, source)
    with torch.no_grad():
        target_features = source_relative_motion_features(target.detach(), source.detach())
    feature_match = motion_feature_match_loss(predicted_features, target_features)
    phase_rms = phase_rms_match_loss(predicted_features, target_features)
    temporal_dc = temporal_dc_residual_penalty(predicted, source)
    phase0 = target_phase0_base_parity_penalty(predicted, source)
    gate_l2 = gate_l2_penalty(phase, block, batch_size=int(source.shape[0]))

    total = (
        FEATURE_MATCH_WEIGHT * feature_match
        + TEMPORAL_DC_WEIGHT * temporal_dc
        + PHASE0_PARITY_WEIGHT * phase0
        + PHASE_RMS_WEIGHT * phase_rms
        + GATE_L2_WEIGHT * gate_l2
    )
    if total.ndim != 0 or not bool(torch.isfinite(total).item()):
        raise FewShotTeacherObjectiveError("teacher objective is invalid")
    return TeacherObjectiveResult(
        total=total,
        feature_match=feature_match,
        temporal_dc_residual=temporal_dc,
        target_phase0_base_parity=phase0,
        phase_rms_match=phase_rms,
        gate_l2=gate_l2,
    )


def _validate_phase_permutation(indices: Sequence[int]) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in indices)
    except (TypeError, ValueError, OverflowError) as error:
        raise FewShotTeacherObjectiveError("phase permutation must be integral") from error
    if len(result) != LATENT_PHASES or result[0] != 0:
        raise FewShotTeacherObjectiveError(
            "phase permutation must have length 21 and preserve phase 0"
        )
    if tuple(sorted(result)) != tuple(range(LATENT_PHASES)):
        raise FewShotTeacherObjectiveError("phase permutation is not bijective")
    return result


def permute_nonboundary_phases(
    clean_video: torch.Tensor,
    phase_indices: Sequence[int],
) -> torch.Tensor:
    """Apply an audited phase control while preserving phase 0."""

    video = _validate_clean_video("clean_video", clean_video)
    indices = _validate_phase_permutation(phase_indices)
    if indices not in (REVERSE_PHASE_INDICES, SHUFFLE_PHASE_INDICES):
        raise FewShotTeacherObjectiveError(
            "only the frozen reverse and shuffle controls are supported"
        )
    index = torch.tensor(indices, dtype=torch.long, device=video.device)
    return video.index_select(2, index)


def reverse_nonboundary_phases(clean_video: torch.Tensor) -> torch.Tensor:
    return permute_nonboundary_phases(clean_video, REVERSE_PHASE_INDICES)


def shuffle_nonboundary_phases(clean_video: torch.Tensor) -> torch.Tensor:
    return permute_nonboundary_phases(clean_video, SHUFFLE_PHASE_INDICES)


def _require_scalar_loss(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise FewShotTeacherObjectiveError(f"{name} must be a scalar tensor")
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).item()):
        raise FewShotTeacherObjectiveError(f"{name} must be finite floating point")
    if bool((value < 0).item()):
        raise FewShotTeacherObjectiveError(f"{name} cannot be negative")
    return value


@dataclass(frozen=True)
class ControlContrastResult:
    total: torch.Tensor
    reverse_hinge: torch.Tensor
    shuffle_hinge: torch.Tensor
    relative_margin: float


def reverse_shuffle_contrast_margin(
    correct_loss: torch.Tensor,
    reverse_loss: torch.Tensor,
    shuffle_loss: torch.Tensor,
    *,
    relative_margin: float = GO_MIN_CONTROL_IMPROVEMENT,
) -> ControlContrastResult:
    """Require each control loss to exceed correct by a relative margin."""

    correct = _require_scalar_loss("correct_loss", correct_loss)
    reverse = _require_scalar_loss("reverse_loss", reverse_loss)
    shuffle = _require_scalar_loss("shuffle_loss", shuffle_loss)
    if correct.device != reverse.device or correct.device != shuffle.device:
        raise FewShotTeacherObjectiveError("control losses must share one device")
    if (
        isinstance(relative_margin, bool)
        or not isinstance(relative_margin, (int, float))
        or not math.isfinite(float(relative_margin))
        or not 0.0 < float(relative_margin) < 1.0
    ):
        raise FewShotTeacherObjectiveError("relative_margin must lie in (0,1)")
    wanted = correct * (1.0 + float(relative_margin))
    reverse_hinge = F.relu(wanted - reverse)
    shuffle_hinge = F.relu(wanted - shuffle)
    total = 0.5 * (reverse_hinge + shuffle_hinge)
    return ControlContrastResult(
        total=total,
        reverse_hinge=reverse_hinge,
        shuffle_hinge=shuffle_hinge,
        relative_margin=float(relative_margin),
    )


def flatten_phase_block_code(
    phase_gates: torch.Tensor,
    block_gates: torch.Tensor,
) -> torch.Tensor:
    phase, block = _validate_gates(phase_gates, block_gates)
    return torch.cat((phase[:, 1:], block), dim=1)


def gate_saturation_fraction(
    phase_gates: torch.Tensor,
    block_gates: torch.Tensor,
) -> float:
    flattened = flatten_phase_block_code(phase_gates, block_gates)
    saturated = flattened.detach().abs() > GATE_SATURATION_THRESHOLD
    return float(saturated.to(torch.float32).mean().item())


def minimum_pairwise_support_code_cosine(support_codes: torch.Tensor) -> float:
    codes = _require_fp32_tensor("support_codes", support_codes)
    if codes.ndim != 2 or int(codes.shape[0]) < 2 or int(codes.shape[1]) != ACTION_CODE_DIM:
        raise FewShotTeacherObjectiveError(
            f"support_codes must be [K,{ACTION_CODE_DIM}] with K>=2"
        )
    norms = codes.square().sum(dim=1).sqrt()
    if bool((norms <= RMS_EPSILON).any().item()):
        raise FewShotTeacherObjectiveError("support code cosine is undefined at zero")
    normalized = F.normalize(codes, dim=1, eps=RMS_EPSILON)
    similarity = normalized @ normalized.transpose(0, 1)
    indices = torch.triu_indices(
        int(codes.shape[0]), int(codes.shape[0]), offset=1, device=codes.device
    )
    minimum = similarity[indices[0], indices[1]].min()
    return float(minimum.item())


def _finite_nonnegative_number(name: str, value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise FewShotTeacherObjectiveError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FewShotTeacherObjectiveError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise FewShotTeacherObjectiveError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class HeldNoiseStatistics:
    zero_loss: float
    correct_loss: float
    reverse_loss: float
    shuffle_loss: float
    zero_improvement_fraction: float
    reverse_improvement_fraction: float
    shuffle_improvement_fraction: float
    gate_saturation_fraction: float
    minimum_support_code_cosine: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "held_noise": True,
            "zero_loss": self.zero_loss,
            "correct_loss": self.correct_loss,
            "reverse_loss": self.reverse_loss,
            "shuffle_loss": self.shuffle_loss,
            "zero_improvement_fraction": self.zero_improvement_fraction,
            "reverse_improvement_fraction": self.reverse_improvement_fraction,
            "shuffle_improvement_fraction": self.shuffle_improvement_fraction,
            "gate_saturation_fraction": self.gate_saturation_fraction,
            "minimum_support_code_cosine": self.minimum_support_code_cosine,
        }


def build_held_noise_statistics(
    *,
    zero_loss: float | torch.Tensor,
    correct_loss: float | torch.Tensor,
    reverse_loss: float | torch.Tensor,
    shuffle_loss: float | torch.Tensor,
    phase_gates: torch.Tensor,
    block_gates: torch.Tensor,
    support_codes: torch.Tensor,
) -> HeldNoiseStatistics:
    """Build held-noise metrics without consulting the optimization trajectory."""

    zero = _finite_nonnegative_number("zero_loss", zero_loss)
    correct = _finite_nonnegative_number("correct_loss", correct_loss)
    reverse = _finite_nonnegative_number("reverse_loss", reverse_loss)
    shuffle = _finite_nonnegative_number("shuffle_loss", shuffle_loss)
    if zero <= RMS_EPSILON:
        raise FewShotTeacherObjectiveError(
            "zero_loss must be positive to define held-noise improvement"
        )

    zero_improvement = (zero - correct) / zero
    reverse_improvement = (reverse - correct) / max(reverse, RMS_EPSILON)
    shuffle_improvement = (shuffle - correct) / max(shuffle, RMS_EPSILON)
    saturation = gate_saturation_fraction(phase_gates, block_gates)
    cosine = minimum_pairwise_support_code_cosine(support_codes)
    return HeldNoiseStatistics(
        zero_loss=zero,
        correct_loss=correct,
        reverse_loss=reverse,
        shuffle_loss=shuffle,
        zero_improvement_fraction=zero_improvement,
        reverse_improvement_fraction=reverse_improvement,
        shuffle_improvement_fraction=shuffle_improvement,
        gate_saturation_fraction=saturation,
        minimum_support_code_cosine=cosine,
    )


@dataclass(frozen=True)
class TeacherGoDecision:
    go: bool
    checks: Mapping[str, bool]
    failed_checks: tuple[str, ...]
    thresholds: Mapping[str, float]


def evaluate_teacher_go(stats: HeldNoiseStatistics) -> TeacherGoDecision:
    """Apply the frozen minimum teacher-representability GO thresholds."""

    if not isinstance(stats, HeldNoiseStatistics):
        raise FewShotTeacherObjectiveError(
            "stats must be HeldNoiseStatistics built from held noise"
        )
    values = stats.as_dict()
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FewShotTeacherObjectiveError(f"held-noise statistic {key} is non-finite")
    checks = {
        "zero_improvement_at_least_15pct": (
            stats.zero_improvement_fraction >= GO_MIN_ZERO_IMPROVEMENT
        ),
        "reverse_improvement_at_least_5pct": (
            stats.reverse_improvement_fraction >= GO_MIN_CONTROL_IMPROVEMENT
        ),
        "shuffle_improvement_at_least_5pct": (
            stats.shuffle_improvement_fraction >= GO_MIN_CONTROL_IMPROVEMENT
        ),
        "gate_saturation_strictly_below_25pct": (
            stats.gate_saturation_fraction < GO_MAX_SATURATION_FRACTION
        ),
        "support_code_cosine_at_least_0p6": (
            stats.minimum_support_code_cosine >= GO_MIN_SUPPORT_CODE_COSINE
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return TeacherGoDecision(
        go=not failed,
        checks=checks,
        failed_checks=failed,
        thresholds={
            "minimum_zero_improvement_fraction": GO_MIN_ZERO_IMPROVEMENT,
            "minimum_control_improvement_fraction": GO_MIN_CONTROL_IMPROVEMENT,
            "maximum_gate_saturation_fraction_exclusive": GO_MAX_SATURATION_FRACTION,
            "minimum_support_code_cosine": GO_MIN_SUPPORT_CODE_COSINE,
        },
    )


def objective_contract() -> dict[str, Any]:
    return {
        "method": METHOD_NAME,
        "schema_version": SCHEMA_VERSION,
        "clean_video_shape": "[B,16,21,H,W]",
        "feature": {
            "source_relative": True,
            "temporal_dc_removed": True,
            "spatial_pool": [POOL_HEIGHT, POOL_WIDTH],
            "temporal_lags": list(TEMPORAL_LAGS),
            "per_phase_rms_normalization": True,
            "normalized_clip": NORMALIZED_FEATURE_CLIP,
            "match": "zero-shifted_charbonnier",
        },
        "code": {
            "first_canary": "phase20_plus_block16",
            "dimension": ACTION_CODE_DIM,
            "preprojection_channel_chunks_claimed_as_heads": False,
        },
        "weights": {
            "feature_match": FEATURE_MATCH_WEIGHT,
            "temporal_dc_residual": TEMPORAL_DC_WEIGHT,
            "target_phase0_base_parity": PHASE0_PARITY_WEIGHT,
            "phase_rms_match": PHASE_RMS_WEIGHT,
            "gate_l2": GATE_L2_WEIGHT,
            "full_target_flow_matching": FULL_TARGET_FLOW_MATCHING_WEIGHT,
        },
        "forbidden_conditions": ["mask", "flow", "pose", "track", "trajectory"],
        "held_noise_go_thresholds": {
            "zero_improvement_min": GO_MIN_ZERO_IMPROVEMENT,
            "control_improvement_min": GO_MIN_CONTROL_IMPROVEMENT,
            "gate_saturation_max_exclusive": GO_MAX_SATURATION_FRACTION,
            "support_code_cosine_min": GO_MIN_SUPPORT_CODE_COSINE,
        },
    }


__all__ = [
    "ACTION_CODE_DIM",
    "CHARBONNIER_EPSILON",
    "ControlContrastResult",
    "FEATURE_MATCH_WEIGHT",
    "FULL_TARGET_FLOW_MATCHING_WEIGHT",
    "FewShotTeacherObjectiveError",
    "GATE_L2_WEIGHT",
    "GATE_SATURATION_THRESHOLD",
    "GO_MAX_SATURATION_FRACTION",
    "GO_MIN_CONTROL_IMPROVEMENT",
    "GO_MIN_SUPPORT_CODE_COSINE",
    "GO_MIN_ZERO_IMPROVEMENT",
    "HeldNoiseStatistics",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "METHOD_NAME",
    "MOTION_BLOCKS",
    "NORMALIZED_FEATURE_CLIP",
    "NONBOUNDARY_PHASES",
    "PHASE0_PARITY_WEIGHT",
    "PHASE_RMS_WEIGHT",
    "POOL_HEIGHT",
    "POOL_WIDTH",
    "REVERSE_PHASE_INDICES",
    "RMS_EPSILON",
    "SCHEMA_VERSION",
    "SHUFFLE_PHASE_INDICES",
    "SourceRelativeMotionFeatures",
    "TEMPORAL_DC_WEIGHT",
    "TEMPORAL_LAGS",
    "TeacherGoDecision",
    "TeacherObjectiveResult",
    "build_held_noise_statistics",
    "evaluate_teacher_go",
    "fewshot_teacher_objective",
    "flatten_phase_block_code",
    "gate_l2_penalty",
    "gate_saturation_fraction",
    "minimum_pairwise_support_code_cosine",
    "motion_feature_match_loss",
    "objective_contract",
    "permute_nonboundary_phases",
    "phase_rms_match_loss",
    "reverse_nonboundary_phases",
    "reverse_shuffle_contrast_margin",
    "shuffle_nonboundary_phases",
    "source_relative_motion_features",
    "target_phase0_base_parity_penalty",
    "temporal_dc_residual_penalty",
]
