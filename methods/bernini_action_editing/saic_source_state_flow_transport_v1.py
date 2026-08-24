#!/usr/bin/env python3
"""Fail-closed source-state flow transport for a frozen Bernini prior.

This module implements only an inference-time clean-latent transport.  It
does not grant semantic, evaluator, training, optimizer, or checkpoint
authority.  A native adapter supplies one *guided* velocity for each frozen
request in one preregistered T2V/R2V/V2V APG field regime.  Every candidate
therefore has two *observed callback queries*.  A T2V/V2V guided query requires
the native negative/conditional pair; a true Bernini R2V guided query requires
the native no-visual-negative, image-negative, image-conditional chain.  Thus
the registered raw-forward requirement is four per T2V/V2V candidate and six
per R2V-I0 candidate.  Model identity, raw calls, and the APG program remain
requirements on the native adapter, not facts this black-box core can observe
or certify.

The core pins a complete 41-value/40-cell rollout and exactly two candidate
schedules: K1=(1,)*40 and K5=(5,5,5)+(1,)*37.  K5 uses the candidate-zero
K-to-one continuation from the existing guided controller.  Exact40 execution
verifies the SHA-256 digest of the actual ordered noise tensors against the
pre-registered bank digest.  Noise
Gaussianity and generator provenance still cannot be established by inspecting
those samples; generator identity and seed remain explicit external claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import struct
from types import MappingProxyType
from typing import Any, Callable, Optional, Sequence

import torch


SCHEMA_VERSION = "bernini-saic-source-state-flow-transport-v1"
REQUEST_SCHEMA_VERSION = f"{SCHEMA_VERSION}/guided-velocity-request-v1"
EXPECTED_BATCH = 1
EXPECTED_CHANNELS = 16
EXPECTED_PHASES = 21
EXPECTED_STEPS = 40
DEFAULT_EARLY_CANDIDATES = 5
DEFAULT_EARLY_STEPS = 3
DEFAULT_SGA_TEMPERATURE = 0.01
ANC_FULL_RETENTION_TIME = 0.25
EXACT_CANDIDATE_SCHEDULE = (
    (DEFAULT_EARLY_CANDIDATES,) * DEFAULT_EARLY_STEPS
    + (1,) * (EXPECTED_STEPS - DEFAULT_EARLY_STEPS)
)
EXACT_SINGLE_CANDIDATE_SCHEDULE = (1,) * EXPECTED_STEPS
REGISTERED_CANDIDATE_SCHEDULES = (
    EXACT_SINGLE_CANDIDATE_SCHEDULE,
    EXACT_CANDIDATE_SCHEDULE,
)
REGISTERED_AGGREGATION_MODES = (
    "uniform",
    "source_similarity_softmax",
)
CANDIDATE_CONTINUATION_POLICY = "candidate_zero"
REGISTERED_FIELD_REGIMES = (
    "t2v_apg",
    "r2v_apg_source_i0",
    "v2v_apg_full_source",
)
GUIDANCE_MODE_BY_FIELD_REGIME = MappingProxyType({
    "t2v_apg": "t2v_apg",
    "r2v_apg_source_i0": "r2v_apg",
    "v2v_apg_full_source": "v2v_apg",
})
VISUAL_CONDITION_SCOPE_BY_FIELD_REGIME = MappingProxyType({
    "t2v_apg": "none",
    "r2v_apg_source_i0": "source_frame0_only_no_future_motion",
    "v2v_apg_full_source": "full_source_video_negative_control",
})
EXPECTED_GUIDANCE_SCALE = 4.0
EXPECTED_IMAGE_GUIDANCE_SCALE = 4.5
EXPECTED_APG_ETA = 0.5
EXPECTED_APG_NORM_THRESHOLD = 50.0
EXPECTED_APG_MOMENTUM = 0.0
EXPECTED_T2V_V2V_BRANCH_ORDER = (
    "target_negative",
    "target_condition",
    "source_negative",
    "source_condition",
)
EXPECTED_R2V_I0_BRANCH_ORDER = (
    "target_none_negative",
    "target_i0_negative",
    "target_i0_condition",
    "source_none_negative",
    "source_i0_negative",
    "source_i0_condition",
)
EXPECTED_BRANCH_ORDER = EXPECTED_T2V_V2V_BRANCH_ORDER
T2V_V2V_RAW_TRANSFORMER_FORWARDS_PER_GUIDED_QUERY = 2
R2V_I0_RAW_TRANSFORMER_FORWARDS_PER_GUIDED_QUERY = 3
T2V_V2V_RAW_TRANSFORMER_FORWARDS_PER_CANDIDATE = 4
R2V_I0_RAW_TRANSFORMER_FORWARDS_PER_CANDIDATE = 6
RAW_TRANSFORMER_FORWARDS_PER_CANDIDATE = (
    T2V_V2V_RAW_TRANSFORMER_FORWARDS_PER_CANDIDATE
)
GUIDED_VELOCITY_QUERIES_PER_CANDIDATE = 2


def _registered_candidate_schedule() -> tuple[int, ...]:
    """Return the literal K5 schedule used by similarity-guided arms."""

    return (5, 5, 5) + (1,) * 37


def _registered_single_candidate_schedule() -> tuple[int, ...]:
    """Return the literal K1 schedule used by the first smoke arms."""

    return (1,) * 40


def _validate_candidate_schedule(value: Any) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or len(value) != 40
        or any(type(item) is not int for item in value)
        or value not in (
            _registered_single_candidate_schedule(),
            _registered_candidate_schedule(),
        )
    ):
        raise SAICSourceStateFlowTransportError(
            "candidate_schedule must be registered K1=(1,)*40 or "
            "K5=(5,5,5)+(1,)*37"
        )
    return value


def _registered_guidance_mode(field_regime: str) -> str:
    if field_regime == "t2v_apg":
        return "t2v_apg"
    if field_regime == "r2v_apg_source_i0":
        return "r2v_apg"
    if field_regime == "v2v_apg_full_source":
        return "v2v_apg"
    raise SAICSourceStateFlowTransportError(
        "field_regime must be one of "
        "('t2v_apg', 'r2v_apg_source_i0', 'v2v_apg_full_source')"
    )


def _registered_visual_condition_scope(field_regime: str) -> str:
    if field_regime == "t2v_apg":
        return "none"
    if field_regime == "r2v_apg_source_i0":
        return "source_frame0_only_no_future_motion"
    if field_regime == "v2v_apg_full_source":
        return "full_source_video_negative_control"
    raise SAICSourceStateFlowTransportError(
        "unregistered field_regime has no visual-condition scope"
    )


def _registered_native_guidance_program(field_regime: str) -> dict[str, Any]:
    """Return literal per-regime Bernini APG-chain requirements.

    The R2V image axis is not ordinary two-forward text CFG.  Its first
    guidance stage compares an unconditioned visual pack against the I0 pack,
    and its second stage adds the role caption on that same I0 pack.
    """

    if field_regime in ("t2v_apg", "v2v_apg_full_source"):
        return {
            "image_guidance_scale": 0.0,
            "guidance_chain_scales": (4.0,),
            "apg_norm_thresholds": (50.0,),
            "apg_momenta": (0.0,),
            "branch_order": (
                "target_negative",
                "target_condition",
                "source_negative",
                "source_condition",
            ),
            "raw_transformer_forwards_per_guided_query": 2,
            "raw_transformer_forwards_per_candidate": 4,
        }
    if field_regime == "r2v_apg_source_i0":
        return {
            "image_guidance_scale": 4.5,
            "guidance_chain_scales": (4.5, 4.0),
            "apg_norm_thresholds": (50.0, 50.0),
            "apg_momenta": (0.0, 0.0),
            "branch_order": (
                "target_none_negative",
                "target_i0_negative",
                "target_i0_condition",
                "source_none_negative",
                "source_i0_negative",
                "source_i0_condition",
            ),
            "raw_transformer_forwards_per_guided_query": 3,
            "raw_transformer_forwards_per_candidate": 6,
        }
    raise SAICSourceStateFlowTransportError(
        "unregistered field_regime has no native guidance program"
    )


class SAICSourceStateFlowTransportError(RuntimeError):
    """The clean-state transport contract was violated."""


@dataclass(frozen=True)
class NativeGuidanceBinding:
    """Expected frozen model and APG program at the native-adapter boundary."""

    model_id: str
    checkpoint_sha256: str
    negative_prompt_sha256: str
    field_regime: str
    guidance_mode: str
    guidance_contract_sha256: str
    guidance_scale: float = EXPECTED_GUIDANCE_SCALE
    image_guidance_scale: float = 0.0
    guidance_chain_scales: tuple[float, ...] = (4.0,)
    apg_eta: float = EXPECTED_APG_ETA
    apg_norm_threshold: float = EXPECTED_APG_NORM_THRESHOLD
    apg_norm_thresholds: tuple[float, ...] = (50.0,)
    apg_momentum: float = EXPECTED_APG_MOMENTUM
    apg_momenta: tuple[float, ...] = (0.0,)
    branch_order: tuple[str, ...] = EXPECTED_BRANCH_ORDER
    raw_transformer_forwards_per_candidate: int = (
        RAW_TRANSFORMER_FORWARDS_PER_CANDIDATE
    )

    def validate(self) -> "NativeGuidanceBinding":
        _stripped(self.model_id, label="model_id")
        _sha256(self.checkpoint_sha256, label="checkpoint_sha256")
        _sha256(self.negative_prompt_sha256, label="negative_prompt_sha256")
        _sha256(self.guidance_contract_sha256, label="guidance_contract_sha256")
        _stripped(self.field_regime, label="field_regime")
        _stripped(self.guidance_mode, label="guidance_mode")
        registered_guidance_mode = _registered_guidance_mode(self.field_regime)
        registered_program = _registered_native_guidance_program(self.field_regime)
        exact = {
            "guidance_mode": (
                self.guidance_mode,
                registered_guidance_mode,
            ),
            "guidance_scale": (self.guidance_scale, 4.0),
            "image_guidance_scale": (
                self.image_guidance_scale,
                registered_program["image_guidance_scale"],
            ),
            "guidance_chain_scales": (
                self.guidance_chain_scales,
                registered_program["guidance_chain_scales"],
            ),
            "apg_eta": (self.apg_eta, 0.5),
            "apg_norm_threshold": (
                self.apg_norm_threshold,
                50.0,
            ),
            "apg_norm_thresholds": (
                self.apg_norm_thresholds,
                registered_program["apg_norm_thresholds"],
            ),
            "apg_momentum": (self.apg_momentum, 0.0),
            "apg_momenta": (
                self.apg_momenta,
                registered_program["apg_momenta"],
            ),
            "branch_order": (
                self.branch_order,
                registered_program["branch_order"],
            ),
            "raw_transformer_forwards_per_candidate": (
                self.raw_transformer_forwards_per_candidate,
                registered_program["raw_transformer_forwards_per_candidate"],
            ),
        }
        for label, (actual, expected) in exact.items():
            if type(expected) is float:
                valid = (
                    type(actual) in (int, float)
                    and not isinstance(actual, bool)
                    and math.isfinite(float(actual))
                    and float(actual) == expected
                )
            else:
                valid = type(actual) is type(expected) and actual == expected
            if not valid:
                raise SAICSourceStateFlowTransportError(
                    f"{label} must equal the registered value {expected!r}"
                )
        return self

    @property
    def visual_condition_scope(self) -> str:
        return _registered_visual_condition_scope(self.field_regime)


@dataclass(frozen=True)
class FlowTransportRolloutConfig:
    """Immutable exact40 mechanism and native closure."""

    native: NativeGuidanceBinding
    anc_enabled: bool
    noise_generator_id: str
    master_seed: int
    noise_bank_sha256: str
    sigma_schedule: tuple[float, ...]
    candidate_schedule: tuple[int, ...] = EXACT_CANDIDATE_SCHEDULE
    candidate_continuation: str = CANDIDATE_CONTINUATION_POLICY
    aggregation_mode: str = "source_similarity_softmax"
    temperature: Optional[float] = DEFAULT_SGA_TEMPERATURE
    anchor_latent_phase_zero: bool = False

    def validate(self) -> "FlowTransportRolloutConfig":
        if type(self.native) is not _NATIVE_GUIDANCE_BINDING_TYPE:
            raise SAICSourceStateFlowTransportError(
                "native must be an exact NativeGuidanceBinding"
            )
        self.native.validate()
        if type(self.anc_enabled) is not bool:
            raise SAICSourceStateFlowTransportError("anc_enabled must be bool")
        _stripped(self.noise_generator_id, label="noise_generator_id")
        if type(self.master_seed) is not int or self.master_seed < 0:
            raise SAICSourceStateFlowTransportError(
                "master_seed must be a nonnegative integer"
            )
        _sha256(self.noise_bank_sha256, label="noise_bank_sha256")
        registered_schedule = _validate_candidate_schedule(self.candidate_schedule)
        if (
            type(self.candidate_continuation) is not str
            or self.candidate_continuation != "candidate_zero"
        ):
            raise SAICSourceStateFlowTransportError(
                "K-to-one continuation must keep candidate zero; weighted noise collapse is unsupported"
            )
        if type(self.aggregation_mode) is not str:
            raise SAICSourceStateFlowTransportError(
                "aggregation_mode must be an exact string"
            )
        if self.aggregation_mode == "uniform":
            if self.temperature is not None:
                raise SAICSourceStateFlowTransportError(
                    "uniform aggregation requires temperature=None"
                )
        elif self.aggregation_mode == "source_similarity_softmax":
            _positive_finite(self.temperature, label="temperature")
        else:
            raise SAICSourceStateFlowTransportError(
                "aggregation_mode must be 'uniform' or 'source_similarity_softmax'"
            )
        if (
            registered_schedule == _registered_single_candidate_schedule()
            and self.aggregation_mode != "uniform"
        ):
            raise SAICSourceStateFlowTransportError(
                "registered K1 arms require uniform aggregation"
            )
        if type(self.sigma_schedule) is not tuple:
            raise SAICSourceStateFlowTransportError(
                "sigma_schedule must be the immutable registered 41-value tuple"
            )
        _validate_sigma_schedule(self.sigma_schedule)
        if type(self.anchor_latent_phase_zero) is not bool:
            raise SAICSourceStateFlowTransportError(
                "anchor_latent_phase_zero must be bool"
            )
        return self

    @property
    def sigma_schedule_sha256(self) -> str:
        return _sigma_schedule_sha256(self.sigma_schedule)


@dataclass(frozen=True)
class FlowTransportStepBinding:
    """Exact scheduler/mechanism binding attached to every callback request."""

    step_index: int
    sigma: float
    next_sigma: float
    time: float
    next_time: float
    candidate_count: int
    anc_enabled: bool
    anc_retention: float
    candidate_continuation: str
    candidate_schedule: tuple[int, ...]
    aggregation_mode: str
    temperature: Optional[float]
    sigma_schedule: tuple[float, ...]
    sigma_schedule_sha256: str
    native: NativeGuidanceBinding
    noise_generator_id: str
    master_seed: int
    noise_bank_sha256: str
    time_parameterization: str = "flow_time_equals_sigma"
    guided_velocity_queries_per_candidate: int = (
        GUIDED_VELOCITY_QUERIES_PER_CANDIDATE
    )
    raw_transformer_forwards_per_candidate: int = (
        RAW_TRANSFORMER_FORWARDS_PER_CANDIDATE
    )


@dataclass(frozen=True)
class VelocityQueryRequest:
    """Only object a frozen guided-velocity callback is allowed to consume."""

    state: torch.Tensor
    caption: str
    role: str
    candidate_index: int
    step: FlowTransportStepBinding
    state_sha256: str
    request_schema: str = REQUEST_SCHEMA_VERSION
    expected_raw_transformer_forwards: int = 0


@dataclass(frozen=True)
class FlowTransportDiagnostics:
    """Detached tensor-free diagnostics for one clean-state Euler step.

    ``raw_transformer_forward_count`` is the count required by the registered
    adapter contract.  This core observes callback invocations, not native
    transformer calls, so the corresponding verification flag is always
    false here.
    """

    step_index: int
    sigma: float
    next_sigma: float
    time: float
    next_time: float
    anc_enabled: bool
    anc_retention: float
    candidate_count: int
    candidate_continuation: str
    aggregation_mode: str
    temperature: Optional[float]
    source_target_query_states_equal_by_candidate: tuple[bool, ...]
    source_similarity_by_candidate: tuple[float, ...]
    aggregation_weights: tuple[float, ...]
    latent_phase_zero_anchor_requested: bool
    latent_phase_zero_anchored: bool
    exact_caption_noop_bypass: bool
    guided_velocity_query_count: int
    raw_transformer_forward_count: int
    raw_transformer_forward_count_verified: bool = False
    native_request_execution_verified: bool = False
    model_checkpoint_use_verified: bool = False
    noise_bank_digest_verified: bool = False
    noise_distribution_verified: bool = False
    optimizer_step_allowed: bool = False
    training_update_allowed: bool = False
    semantic_action_success: bool = False


@dataclass(frozen=True)
class FlowTransportStep:
    """The next clean edit state and truthful ANC states for the next step."""

    edit_clean: torch.Tensor
    correlated_noises: tuple[torch.Tensor, ...]
    diagnostics: FlowTransportDiagnostics


@dataclass(frozen=True)
class FlowTransportRolloutDiagnostics:
    """Tensor-free exact40 execution evidence."""

    sigma_schedule: tuple[float, ...]
    sigma_schedule_sha256: str
    candidate_counts: tuple[int, ...]
    candidate_continuation: str
    aggregation_mode: str
    temperature: Optional[float]
    anc_enabled: bool
    guided_velocity_query_count: int
    raw_transformer_forward_count: int
    step_diagnostics: tuple[FlowTransportDiagnostics, ...]
    model_id: str
    checkpoint_sha256: str
    field_regime: str
    visual_condition_scope: str
    guidance_mode: str
    guidance_contract_sha256: str
    negative_prompt_sha256: str
    guidance_scale: float
    image_guidance_scale: float
    guidance_chain_scales: tuple[float, ...]
    apg_eta: float
    apg_norm_threshold: float
    apg_norm_thresholds: tuple[float, ...]
    apg_momentum: float
    apg_momenta: tuple[float, ...]
    branch_order: tuple[str, ...]
    noise_generator_id: str
    master_seed: int
    noise_bank_sha256: str
    noise_bank_digest_verified: bool
    raw_transformer_forward_count_verified: bool = False
    native_request_execution_verified: bool = False
    model_checkpoint_use_verified: bool = False
    noise_distribution_verified: bool = False
    optimizer_step_allowed: bool = False
    training_update_allowed: bool = False
    semantic_action_success: bool = False


@dataclass(frozen=True)
class FlowTransportRollout:
    """Final exact40 edit state and non-authoritative diagnostics."""

    edit_clean: torch.Tensor
    final_correlated_noises: tuple[torch.Tensor, ...]
    diagnostics: FlowTransportRolloutDiagnostics


# Capture concrete implementation types once.  Exported names remain useful
# for callers, but rebinding those public globals cannot make runtime type
# validation accept attacker-selected replacement classes.
_NATIVE_GUIDANCE_BINDING_TYPE = NativeGuidanceBinding
_ROLLOUT_CONFIG_TYPE = FlowTransportRolloutConfig
_STEP_BINDING_TYPE = FlowTransportStepBinding
_VELOCITY_QUERY_REQUEST_TYPE = VelocityQueryRequest
_STEP_DIAGNOSTICS_TYPE = FlowTransportDiagnostics
_STEP_RESULT_TYPE = FlowTransportStep
_ROLLOUT_DIAGNOSTICS_TYPE = FlowTransportRolloutDiagnostics
_ROLLOUT_RESULT_TYPE = FlowTransportRollout


VelocityQuery = Callable[[VelocityQueryRequest], torch.Tensor]


def _candidate_count_for_step(
    step_index: int,
    candidate_schedule: tuple[int, ...],
) -> int:
    if type(step_index) is not int or not 0 <= step_index < 40:
        raise SAICSourceStateFlowTransportError(
            "step_index must be an integer in [0,39]"
        )
    return _validate_candidate_schedule(candidate_schedule)[step_index]


def candidate_count_for_step(
    step_index: int,
    *,
    candidate_schedule: tuple[int, ...] = EXACT_CANDIDATE_SCHEDULE,
) -> int:
    """Return a cell from one of the two registered exact40 schedules."""

    return _candidate_count_for_step(step_index, candidate_schedule)


def _anc_retention(time: float) -> float:
    value = _unit_time(time, label="time")
    if value <= 0.25:
        return 1.0
    return (1.0 - value) / 0.75


def anc_retention(time: float) -> float:
    """Return retained variance; this does not prove sampled Gaussianity."""

    return _anc_retention(time)


def correlate_noise(
    fresh_noise: torch.Tensor,
    *,
    previous_noise: Optional[torch.Tensor],
    retention: float,
) -> torch.Tensor:
    """Mix fresh and previous states, always returning owned storage."""

    fresh = _clean_tensor(fresh_noise, label="fresh_noise")
    amount = _unit_time(retention, label="retention")
    if amount == 0.0:
        return fresh.detach().clone().contiguous()
    if previous_noise is None:
        raise SAICSourceStateFlowTransportError(
            "positive ANC retention requires a previous noise state"
        )
    previous = _clean_tensor(
        previous_noise,
        label="previous_noise",
        expected_shape=tuple(map(int, fresh.shape)),
        expected_device=fresh.device,
    )
    if amount == 1.0:
        return previous.detach().clone().contiguous()
    with torch.no_grad():
        mixed = (
            math.sqrt(amount) * previous.float()
            + math.sqrt(1.0 - amount) * fresh.float()
        )
    return mixed.detach().clone().contiguous()


_CORRELATE_NOISE_IMPL = correlate_noise


def build_source_target_query_states(
    source_clean: torch.Tensor,
    edit_clean: torch.Tensor,
    noise: torch.Tensor,
    *,
    time: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct source/target queries with one shared noise residual."""

    source, edit, candidate_noise = _validate_state_triplet(
        source_clean, edit_clean, noise
    )
    value = _positive_time(time, label="time")
    with torch.no_grad():
        source_state = (
            (1.0 - value) * source.float() + value * candidate_noise.float()
        ).contiguous()
        target_state = (edit.float() + source_state - source.float()).contiguous()
    return source_state, target_state


_BUILD_SOURCE_TARGET_QUERY_STATES_IMPL = build_source_target_query_states


def similarity_guided_delta(
    source_clean: torch.Tensor,
    edit_clean: torch.Tensor,
    candidate_deltas: Sequence[torch.Tensor],
    *,
    time: float,
    temperature: float,
) -> tuple[torch.Tensor, tuple[float, ...], tuple[float, ...]]:
    """Aggregate candidate velocity differences by full-latent cosine SGA."""

    source_input = _clean_state(source_clean, label="source_clean")
    edit_input = _clean_state(
        edit_clean,
        label="edit_clean",
        expected_shape=tuple(map(int, source_input.shape)),
        expected_device=source_input.device,
    )
    # Work only on owned state.  A Python callback can perform arbitrary side
    # effects on caller-owned closure variables (which this module cannot
    # sandbox), but those handles must not mutate the core's numerical state.
    source = source_input.detach().clone().contiguous()
    edit = edit_input.detach().clone().contiguous()
    value = _positive_time(time, label="time")
    tau = _positive_finite(temperature, label="temperature")
    if isinstance(candidate_deltas, (str, bytes)) or not isinstance(
        candidate_deltas, Sequence
    ) or not candidate_deltas:
        raise SAICSourceStateFlowTransportError(
            "candidate_deltas must be a nonempty sequence"
        )
    deltas = tuple(
        _clean_tensor(
            delta,
            label=f"candidate_delta[{index}]",
            expected_shape=tuple(map(int, source.shape)),
            expected_device=source.device,
        ).float()
        for index, delta in enumerate(candidate_deltas)
    )
    with torch.no_grad():
        projected = torch.stack(
            [edit.float() - value * delta for delta in deltas], dim=0
        )
        source_vector = source.float().reshape(1, -1)
        projected_vectors = projected.reshape(len(deltas), -1)
        source_norm = source_vector.norm(dim=1).clamp_min(1.0e-12)
        projected_norm = projected_vectors.norm(dim=1).clamp_min(1.0e-12)
        similarities = (
            projected_vectors @ source_vector.transpose(0, 1)
        ).squeeze(1) / (projected_norm * source_norm.squeeze(0))
        weights = torch.softmax(similarities / tau, dim=0)
        projected_mean = torch.sum(
            weights.reshape((-1,) + (1,) * edit.ndim) * projected, dim=0
        )
        aggregate = ((edit.float() - projected_mean) / value).contiguous()
    if not bool(torch.isfinite(aggregate).all().item()):
        raise SAICSourceStateFlowTransportError(
            "similarity-guided aggregate became non-finite"
        )
    return (
        aggregate,
        tuple(float(item) for item in similarities.cpu().tolist()),
        tuple(float(item) for item in weights.cpu().tolist()),
    )


_SIMILARITY_GUIDED_DELTA_IMPL = similarity_guided_delta


def _aggregate_candidate_deltas(
    source_clean: torch.Tensor,
    edit_clean: torch.Tensor,
    candidate_deltas: Sequence[torch.Tensor],
    *,
    time: float,
    aggregation_mode: str,
    temperature: Optional[float],
) -> tuple[torch.Tensor, tuple[float, ...], tuple[float, ...]]:
    if aggregation_mode == "source_similarity_softmax":
        if temperature is None:
            raise SAICSourceStateFlowTransportError(
                "source-similarity aggregation requires a temperature"
            )
        return _SIMILARITY_GUIDED_DELTA_IMPL(
            source_clean,
            edit_clean,
            candidate_deltas,
            time=time,
            temperature=temperature,
        )
    if aggregation_mode != "uniform" or temperature is not None:
        raise SAICSourceStateFlowTransportError(
            "unregistered aggregation mode/temperature pair"
        )
    source_input = _clean_state(source_clean, label="source_clean")
    source = source_input.detach().clone().contiguous()
    _clean_state(
        edit_clean,
        label="edit_clean",
        expected_shape=tuple(map(int, source.shape)),
        expected_device=source.device,
    )
    _positive_time(time, label="time")
    if (
        isinstance(candidate_deltas, (str, bytes))
        or not isinstance(candidate_deltas, Sequence)
        or not candidate_deltas
    ):
        raise SAICSourceStateFlowTransportError(
            "candidate_deltas must be a nonempty sequence"
        )
    deltas = tuple(
        _clean_tensor(
            delta,
            label=f"candidate_delta[{index}]",
            expected_shape=tuple(map(int, source.shape)),
            expected_device=source.device,
        ).float()
        for index, delta in enumerate(candidate_deltas)
    )
    with torch.no_grad():
        aggregate = torch.stack(deltas, dim=0).mean(dim=0).contiguous()
    if not bool(torch.isfinite(aggregate).all().item()):
        raise SAICSourceStateFlowTransportError(
            "uniform aggregate became non-finite"
        )
    weight = 1.0 / len(deltas)
    return aggregate, (), (weight,) * len(deltas)


def source_state_flow_step(
    *,
    config: FlowTransportRolloutConfig,
    step_index: int,
    source_clean: torch.Tensor,
    edit_clean: torch.Tensor,
    source_caption: str,
    target_caption: str,
    sigma: float,
    next_sigma: float,
    time: float,
    next_time: float,
    fresh_noises: Sequence[torch.Tensor],
    previous_noises: Optional[Sequence[torch.Tensor]],
    velocity_query: VelocityQuery,
) -> FlowTransportStep:
    """Execute one registered exact40 frozen-generator clean edit-ODE cell."""

    runtime = _config(config)
    binding = _make_step_binding(
        runtime,
        step_index=step_index,
        sigma=sigma,
        next_sigma=next_sigma,
        time=time,
        next_time=next_time,
    )
    source_input = _clean_state(source_clean, label="source_clean")
    edit_input = _clean_state(
        edit_clean,
        label="edit_clean",
        expected_shape=tuple(map(int, source_input.shape)),
        expected_device=source_input.device,
    )
    source = source_input.detach().clone().contiguous()
    edit = edit_input.detach().clone().contiguous()
    source_text = _caption(source_caption, label="source_caption")
    target_text = _caption(target_caption, label="target_caption")
    if not callable(velocity_query):
        raise SAICSourceStateFlowTransportError("velocity_query must be callable")
    noise_inputs = _tensor_sequence(
        fresh_noises,
        label="fresh_noises",
        expected_shape=tuple(map(int, source.shape)),
        expected_device=source.device,
    )
    if len(noise_inputs) != binding.candidate_count:
        raise SAICSourceStateFlowTransportError(
            f"step {step_index} requires exactly {binding.candidate_count} fresh candidates"
        )
    noises = tuple(item.detach().clone().contiguous() for item in noise_inputs)

    previous = _previous_noise_states(
        config=runtime,
        step_index=step_index,
        fresh_noises=noises,
        previous_noises=previous_noises,
        expected_shape=tuple(map(int, source.shape)),
        expected_device=source.device,
    )
    retention = binding.anc_retention if runtime.anc_enabled else 0.0
    correlated = tuple(
        _CORRELATE_NOISE_IMPL(fresh, previous_noise=old, retention=retention)
        for fresh, old in zip(noises, previous)
    )

    # The no-op still advances the registered ANC chain.  Thus the returned
    # noise state is truthful and may be used by the next cell.  No phase-zero
    # anchor is reported because no edit update/anchor operation was executed.
    if source_text == target_text:
        return _STEP_RESULT_TYPE(
            edit_clean=edit.detach().clone().contiguous(),
            correlated_noises=tuple(item.clone() for item in correlated),
            diagnostics=_step_diagnostics(
                binding,
                anchor_requested=runtime.anchor_latent_phase_zero,
                anchor_applied=False,
                noop=True,
                equal_states=(),
                similarities=(),
                weights=(),
                guided_queries=0,
                raw_forwards=0,
            ),
        )

    deltas: list[torch.Tensor] = []
    equal_states: list[bool] = []
    # Version checks catch ordinary in-place closure mutation of caller inputs;
    # owned copies ensure ``.data`` writes cannot change the transport itself.
    protected = (source_input, edit_input) + noise_inputs + (source, edit) + noises + correlated
    for candidate_index, candidate_noise in enumerate(correlated):
        source_state, target_state = _BUILD_SOURCE_TARGET_QUERY_STATES_IMPL(
            source, edit, candidate_noise, time=binding.time
        )
        equal_states.append(bool(torch.equal(source_state, target_state)))
        # Preserve the preregistered target-first APG order used by the
        # existing guided controller: target neg+cond, then source neg+cond.
        target_velocity = _guarded_velocity_query(
            velocity_query,
            state=target_state,
            caption=target_text,
            role="target",
            candidate_index=candidate_index,
            step=binding,
            protected=protected + (source_state, target_state),
        )
        source_velocity = _guarded_velocity_query(
            velocity_query,
            state=source_state,
            caption=source_text,
            role="source",
            candidate_index=candidate_index,
            step=binding,
            protected=protected + (source_state, target_state),
        )
        deltas.append((target_velocity - source_velocity).contiguous())

    aggregate, similarities, weights = _aggregate_candidate_deltas(
        source,
        edit,
        deltas,
        time=binding.time,
        aggregation_mode=runtime.aggregation_mode,
        temperature=runtime.temperature,
    )
    with torch.no_grad():
        updated = (
            edit.float() + (binding.next_time - binding.time) * aggregate.float()
        ).contiguous()
        anchor_applied = runtime.anchor_latent_phase_zero
        if anchor_applied:
            updated[:, :, 0].copy_(source.float()[:, :, 0])
    if not bool(torch.isfinite(updated).all().item()):
        raise SAICSourceStateFlowTransportError(
            "clean-state Euler update became non-finite"
        )
    candidate_count = len(correlated)
    return _STEP_RESULT_TYPE(
        edit_clean=updated.detach().clone().contiguous(),
        correlated_noises=tuple(item.clone() for item in correlated),
        diagnostics=_step_diagnostics(
            binding,
            anchor_requested=runtime.anchor_latent_phase_zero,
            anchor_applied=anchor_applied,
            noop=False,
            equal_states=tuple(equal_states),
            similarities=similarities,
            weights=weights,
            guided_queries=2 * candidate_count,
            raw_forwards=(
                binding.raw_transformer_forwards_per_candidate * candidate_count
            ),
        ),
    )


_SOURCE_STATE_FLOW_STEP_IMPL = source_state_flow_step


def run_exact40_source_state_flow_transport(
    *,
    config: FlowTransportRolloutConfig,
    source_clean: torch.Tensor,
    source_caption: str,
    target_caption: str,
    sigma_schedule: Sequence[float],
    fresh_noise_schedule: Sequence[Sequence[torch.Tensor]],
    velocity_query: VelocityQuery,
) -> FlowTransportRollout:
    """Run all 40 registered cells from the clean source state.

    The caller owns schedule creation and noise generation.  This core checks
    the complete schedule against the config binding and checks the canonical
    digest of every ordered noise tensor against ``noise_bank_sha256``.  It
    deliberately reports ``noise_distribution_verified=False`` because tensor
    content cannot establish Gaussianity, generator implementation, or seed
    provenance.
    """

    runtime = _config(config)
    source_input = _clean_state(source_clean, label="source_clean")
    source = source_input.detach().clone().contiguous()
    source_text = _caption(source_caption, label="source_caption")
    target_text = _caption(target_caption, label="target_caption")
    sigmas = _validate_sigma_schedule(sigma_schedule)
    if sigmas != runtime.sigma_schedule:
        raise SAICSourceStateFlowTransportError(
            "runtime sigma_schedule does not equal the config-registered exact40 schedule"
        )
    actual_noise_bank_sha256, noise_manifest = _noise_bank_manifest(
        fresh_noise_schedule,
        candidate_schedule=runtime.candidate_schedule,
    )
    if actual_noise_bank_sha256 != runtime.noise_bank_sha256:
        raise SAICSourceStateFlowTransportError(
            "actual ordered noise-bank digest does not match noise_bank_sha256"
        )

    edit = source.detach().clone().contiguous()
    previous: Optional[tuple[torch.Tensor, ...]] = None
    step_diagnostics: list[FlowTransportDiagnostics] = []
    for step_index in range(40):
        fresh = fresh_noise_schedule[step_index]
        # A prior callback can close over caller-owned future noise tensors.
        # Rechecking the registered per-tensor digest immediately before use
        # prevents such mutation from silently changing the rollout bank.
        _assert_noise_cell_manifest(
            fresh, noise_manifest[step_index], step_index=step_index
        )
        result = _SOURCE_STATE_FLOW_STEP_IMPL(
            config=runtime,
            step_index=step_index,
            source_clean=source,
            edit_clean=edit,
            source_caption=source_text,
            target_caption=target_text,
            sigma=sigmas[step_index],
            next_sigma=sigmas[step_index + 1],
            time=sigmas[step_index],
            next_time=sigmas[step_index + 1],
            fresh_noises=fresh,
            previous_noises=previous if runtime.anc_enabled else None,
            velocity_query=velocity_query,
        )
        edit = result.edit_clean
        previous = result.correlated_noises
        step_diagnostics.append(result.diagnostics)

    guided_queries = sum(item.guided_velocity_query_count for item in step_diagnostics)
    raw_forwards = sum(item.raw_transformer_forward_count for item in step_diagnostics)
    diagnostics = _ROLLOUT_DIAGNOSTICS_TYPE(
        sigma_schedule=sigmas,
        sigma_schedule_sha256=_sigma_schedule_sha256(sigmas),
        candidate_counts=tuple(
            item.candidate_count for item in step_diagnostics
        ),
        candidate_continuation=runtime.candidate_continuation,
        aggregation_mode=runtime.aggregation_mode,
        temperature=runtime.temperature,
        anc_enabled=runtime.anc_enabled,
        guided_velocity_query_count=guided_queries,
        raw_transformer_forward_count=raw_forwards,
        step_diagnostics=tuple(step_diagnostics),
        model_id=runtime.native.model_id,
        checkpoint_sha256=runtime.native.checkpoint_sha256,
        field_regime=runtime.native.field_regime,
        visual_condition_scope=runtime.native.visual_condition_scope,
        guidance_mode=runtime.native.guidance_mode,
        guidance_contract_sha256=runtime.native.guidance_contract_sha256,
        negative_prompt_sha256=runtime.native.negative_prompt_sha256,
        guidance_scale=runtime.native.guidance_scale,
        image_guidance_scale=runtime.native.image_guidance_scale,
        guidance_chain_scales=runtime.native.guidance_chain_scales,
        apg_eta=runtime.native.apg_eta,
        apg_norm_threshold=runtime.native.apg_norm_threshold,
        apg_norm_thresholds=runtime.native.apg_norm_thresholds,
        apg_momentum=runtime.native.apg_momentum,
        apg_momenta=runtime.native.apg_momenta,
        branch_order=runtime.native.branch_order,
        noise_generator_id=runtime.noise_generator_id,
        master_seed=runtime.master_seed,
        noise_bank_sha256=runtime.noise_bank_sha256,
        noise_bank_digest_verified=True,
    )
    return _ROLLOUT_RESULT_TYPE(
        edit_clean=edit.detach().clone().contiguous(),
        final_correlated_noises=tuple(item.clone() for item in (previous or ())),
        diagnostics=diagnostics,
    )


def _config(value: Any) -> FlowTransportRolloutConfig:
    if type(value) is not _ROLLOUT_CONFIG_TYPE:
        raise SAICSourceStateFlowTransportError(
            "config must be an exact FlowTransportRolloutConfig"
        )
    value.validate()
    native = value.native
    owned_native = _NATIVE_GUIDANCE_BINDING_TYPE(
        model_id=native.model_id,
        checkpoint_sha256=native.checkpoint_sha256,
        negative_prompt_sha256=native.negative_prompt_sha256,
        field_regime=native.field_regime,
        guidance_mode=native.guidance_mode,
        guidance_contract_sha256=native.guidance_contract_sha256,
        guidance_scale=float(native.guidance_scale),
        image_guidance_scale=float(native.image_guidance_scale),
        guidance_chain_scales=tuple(native.guidance_chain_scales),
        apg_eta=float(native.apg_eta),
        apg_norm_threshold=float(native.apg_norm_threshold),
        apg_norm_thresholds=tuple(native.apg_norm_thresholds),
        apg_momentum=float(native.apg_momentum),
        apg_momenta=tuple(native.apg_momenta),
        branch_order=tuple(native.branch_order),
        raw_transformer_forwards_per_candidate=int(
            native.raw_transformer_forwards_per_candidate
        ),
    ).validate()
    # Return an owned canonical copy.  ``frozen=True`` is not a security
    # boundary against ``object.__setattr__``; a callback must not be able to
    # corrupt the caller's reusable config through its request.
    return _ROLLOUT_CONFIG_TYPE(
        native=owned_native,
        anc_enabled=value.anc_enabled,
        noise_generator_id=value.noise_generator_id,
        master_seed=value.master_seed,
        noise_bank_sha256=value.noise_bank_sha256,
        sigma_schedule=tuple(value.sigma_schedule),
        candidate_schedule=tuple(value.candidate_schedule),
        candidate_continuation=value.candidate_continuation,
        aggregation_mode=value.aggregation_mode,
        temperature=(
            None if value.temperature is None else float(value.temperature)
        ),
        anchor_latent_phase_zero=value.anchor_latent_phase_zero,
    ).validate()


def _make_step_binding(
    config: FlowTransportRolloutConfig,
    *,
    step_index: int,
    sigma: Any,
    next_sigma: Any,
    time: Any,
    next_time: Any,
) -> FlowTransportStepBinding:
    count = _candidate_count_for_step(step_index, config.candidate_schedule)
    current_sigma = _positive_time(sigma, label="sigma")
    following_sigma = _unit_time(next_sigma, label="next_sigma")
    current_time = _positive_time(time, label="time")
    following_time = _unit_time(next_time, label="next_time")
    if following_sigma >= current_sigma or following_time >= current_time:
        raise SAICSourceStateFlowTransportError(
            "next sigma/time must be strictly smaller than current sigma/time"
        )
    if current_sigma != current_time or following_sigma != following_time:
        raise SAICSourceStateFlowTransportError(
            "v1 requires flow time to equal scheduler sigma exactly"
        )
    if (
        current_sigma != config.sigma_schedule[step_index]
        or following_sigma != config.sigma_schedule[step_index + 1]
    ):
        raise SAICSourceStateFlowTransportError(
            "step sigma/time pair does not match the registered full exact40 schedule"
        )
    retention = _anc_retention(current_time) if config.anc_enabled else 0.0
    return _STEP_BINDING_TYPE(
        step_index=step_index,
        sigma=current_sigma,
        next_sigma=following_sigma,
        time=current_time,
        next_time=following_time,
        candidate_count=count,
        anc_enabled=config.anc_enabled,
        anc_retention=retention,
        candidate_continuation=config.candidate_continuation,
        candidate_schedule=config.candidate_schedule,
        aggregation_mode=config.aggregation_mode,
        temperature=config.temperature,
        sigma_schedule=config.sigma_schedule,
        sigma_schedule_sha256=config.sigma_schedule_sha256,
        native=config.native,
        noise_generator_id=config.noise_generator_id,
        master_seed=config.master_seed,
        noise_bank_sha256=config.noise_bank_sha256,
        raw_transformer_forwards_per_candidate=(
            config.native.raw_transformer_forwards_per_candidate
        ),
    )


def _previous_noise_states(
    *,
    config: FlowTransportRolloutConfig,
    step_index: int,
    fresh_noises: tuple[torch.Tensor, ...],
    previous_noises: Optional[Sequence[torch.Tensor]],
    expected_shape: tuple[int, ...],
    expected_device: torch.device,
) -> tuple[Optional[torch.Tensor], ...]:
    current_count = config.candidate_schedule[step_index]
    if not config.anc_enabled:
        if previous_noises is not None:
            raise SAICSourceStateFlowTransportError(
                "ANC-off rollout must not provide previous noise state"
            )
        return (None,) * current_count

    if step_index == 0:
        if previous_noises is not None:
            raise SAICSourceStateFlowTransportError(
                "ANC step zero uses the registered zero-initialized predecessor"
            )
        return tuple(torch.zeros_like(item) for item in fresh_noises)

    if previous_noises is None:
        raise SAICSourceStateFlowTransportError(
            "positive ANC rollout step requires previous noise state"
        )
    validated = _tensor_sequence(
        previous_noises,
        label="previous_noises",
        expected_shape=expected_shape,
        expected_device=expected_device,
    )
    expected_previous_count = config.candidate_schedule[step_index - 1]
    if len(validated) != expected_previous_count:
        raise SAICSourceStateFlowTransportError(
            f"step {step_index} requires {expected_previous_count} registered predecessor candidates"
        )
    if expected_previous_count == current_count:
        return tuple(validated)
    if expected_previous_count == 5 and current_count == 1:
        # This is the only registered K transition.  It deliberately retains
        # candidate zero rather than inventing a weighted noise-chain collapse.
        return (validated[0],)
    raise SAICSourceStateFlowTransportError("unregistered K transition")


def _guarded_velocity_query(
    velocity_query: VelocityQuery,
    *,
    state: torch.Tensor,
    caption: str,
    role: str,
    candidate_index: int,
    step: FlowTransportStepBinding,
    protected: Sequence[torch.Tensor],
) -> torch.Tensor:
    callback_state = state.detach().clone().contiguous()
    state_snapshot = _tensor_snapshot(callback_state)
    protected_versions = tuple(_tensor_version(item) for item in protected)
    request = _VELOCITY_QUERY_REQUEST_TYPE(
        state=callback_state,
        caption=caption,
        role=role,
        candidate_index=candidate_index,
        step=step,
        state_sha256=state_snapshot[1],
        expected_raw_transformer_forwards=(
            step.raw_transformer_forwards_per_candidate
            // step.guided_velocity_queries_per_candidate
        ),
    )
    metadata_snapshot = _request_metadata(request)
    try:
        with torch.inference_mode():
            raw_velocity = velocity_query(request)
    except Exception as error:
        raise SAICSourceStateFlowTransportError(
            f"frozen guided velocity query failed for step {step.step_index} "
            f"candidate {candidate_index} role {role}: {error}"
        ) from error

    if request.state is not callback_state or _request_metadata(request) != metadata_snapshot:
        raise SAICSourceStateFlowTransportError(
            "velocity callback mutated/rebound its immutable request"
        )
    _assert_snapshot(callback_state, state_snapshot, label="callback input state")
    for index, (item, version) in enumerate(zip(protected, protected_versions)):
        current_version = _tensor_version(item)
        if version is not None and current_version != version:
            raise SAICSourceStateFlowTransportError(
                f"protected tensor[{index}] was mutated by callback"
            )
    native = _velocity(raw_velocity, like=state, label="guided_velocity")
    if _shares_storage(native, callback_state) or any(
        _shares_storage(native, item) for item in protected
    ):
        raise SAICSourceStateFlowTransportError(
            "guided velocity must own storage and may not alias callback/core state"
        )
    # Clone immediately after validation/alias rejection so the callback can
    # retain no writable handle to the velocity consumed by the core.
    owned = native.detach().to(dtype=torch.float32).clone().contiguous()
    if not bool(torch.isfinite(owned).all().item()):
        raise SAICSourceStateFlowTransportError("guided velocity became non-finite")
    return owned


def _request_metadata(request: VelocityQueryRequest) -> tuple[Any, ...]:
    native = request.step.native
    step = request.step
    return (
        request.caption,
        request.role,
        request.candidate_index,
        step.step_index,
        step.sigma,
        step.next_sigma,
        step.time,
        step.next_time,
        step.candidate_count,
        step.anc_enabled,
        step.anc_retention,
        step.candidate_continuation,
        step.candidate_schedule,
        step.aggregation_mode,
        step.temperature,
        step.sigma_schedule,
        step.sigma_schedule_sha256,
        step.noise_generator_id,
        step.master_seed,
        step.noise_bank_sha256,
        step.time_parameterization,
        step.guided_velocity_queries_per_candidate,
        step.raw_transformer_forwards_per_candidate,
        native.model_id,
        native.checkpoint_sha256,
        native.negative_prompt_sha256,
        native.field_regime,
        native.guidance_mode,
        native.guidance_contract_sha256,
        native.guidance_scale,
        native.image_guidance_scale,
        native.guidance_chain_scales,
        native.apg_eta,
        native.apg_norm_threshold,
        native.apg_norm_thresholds,
        native.apg_momentum,
        native.apg_momenta,
        native.branch_order,
        native.raw_transformer_forwards_per_candidate,
        request.state_sha256,
        request.request_schema,
        request.expected_raw_transformer_forwards,
    )


def _tensor_version(value: torch.Tensor) -> Optional[int]:
    try:
        return int(value._version)
    except RuntimeError:
        # Tensors created inside torch.inference_mode intentionally have no
        # version counter.  Byte/equality guards remain authoritative.
        return None


def _tensor_snapshot(
    value: torch.Tensor,
) -> tuple[Optional[int], str, torch.Tensor]:
    # The digest crosses the native-adapter request boundary once.  The owned
    # guard keeps the post-callback integrity check on device, avoiding a
    # second full GPU-to-CPU copy for every guided query.
    return (
        _tensor_version(value),
        _tensor_content_sha256(value),
        value.detach().clone().contiguous(),
    )


def _assert_snapshot(
    value: torch.Tensor,
    snapshot: tuple[Optional[int], str, torch.Tensor],
    *,
    label: str,
) -> None:
    initial_version, _content_sha256, owned_guard = snapshot
    current_version = _tensor_version(value)
    if (
        (initial_version is not None and current_version != initial_version)
        or not bool(torch.equal(value, owned_guard))
    ):
        raise SAICSourceStateFlowTransportError(f"{label} was mutated by callback")


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device:
        return False
    try:
        return _storage_pointer(left) == _storage_pointer(right)
    except RuntimeError as error:
        raise SAICSourceStateFlowTransportError(
            "could not establish guided-velocity storage independence"
        ) from error


def _storage_pointer(value: torch.Tensor) -> int:
    if hasattr(value, "untyped_storage"):
        return int(value.untyped_storage().data_ptr())
    return int(value.storage().data_ptr())


def _step_diagnostics(
    binding: FlowTransportStepBinding,
    *,
    anchor_requested: bool,
    anchor_applied: bool,
    noop: bool,
    equal_states: tuple[bool, ...],
    similarities: tuple[float, ...],
    weights: tuple[float, ...],
    guided_queries: int,
    raw_forwards: int,
) -> FlowTransportDiagnostics:
    return _STEP_DIAGNOSTICS_TYPE(
        step_index=binding.step_index,
        sigma=binding.sigma,
        next_sigma=binding.next_sigma,
        time=binding.time,
        next_time=binding.next_time,
        anc_enabled=binding.anc_enabled,
        anc_retention=binding.anc_retention,
        candidate_count=binding.candidate_count,
        candidate_continuation=binding.candidate_continuation,
        aggregation_mode=binding.aggregation_mode,
        temperature=binding.temperature,
        source_target_query_states_equal_by_candidate=equal_states,
        source_similarity_by_candidate=similarities,
        aggregation_weights=weights,
        latent_phase_zero_anchor_requested=anchor_requested,
        latent_phase_zero_anchored=anchor_applied,
        exact_caption_noop_bypass=noop,
        guided_velocity_query_count=guided_queries,
        raw_transformer_forward_count=raw_forwards,
    )


def _validate_sigma_schedule(value: Any) -> tuple[float, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 41
    ):
        raise SAICSourceStateFlowTransportError(
            "sigma_schedule must contain exactly 41 values"
        )
    result = tuple(_unit_time(item, label=f"sigma_schedule[{i}]") for i, item in enumerate(value))
    if result[0] <= 0.0 or result[-1] != 0.0:
        raise SAICSourceStateFlowTransportError(
            "sigma schedule must start positive and terminate at exact zero"
        )
    if any(following >= current for current, following in zip(result, result[1:])):
        raise SAICSourceStateFlowTransportError(
            "sigma schedule must be strictly decreasing"
        )
    return result


def _sigma_schedule_sha256(value: Any) -> str:
    schedule = _validate_sigma_schedule(value)
    digest = hashlib.sha256(b"saic-exact40-sigma-schedule-v1\0")
    for index, sigma in enumerate(schedule):
        digest.update(struct.pack(">Id", index, sigma))
    return digest.hexdigest()


def sigma_schedule_sha256(value: Any) -> str:
    """Return the canonical digest of a validated 41-value exact40 schedule."""

    return _sigma_schedule_sha256(value)


def _tensor_content_sha256(value: torch.Tensor) -> str:
    byte_tensor = (
        value.detach().contiguous().reshape(-1).view(torch.uint8).to(device="cpu")
    )
    try:
        data = byte_tensor.numpy().tobytes()
    except RuntimeError:
        data = bytes(byte_tensor.tolist())
    return hashlib.sha256(data).hexdigest()


def _noise_bank_manifest(
    value: Any,
    *,
    candidate_schedule: tuple[int, ...],
) -> tuple[str, tuple[tuple[str, ...], ...]]:
    registered_schedule = _validate_candidate_schedule(candidate_schedule)
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 40
    ):
        raise SAICSourceStateFlowTransportError(
            "fresh_noise_schedule must contain exactly 40 cells"
        )
    digest = hashlib.sha256(b"saic-exact40-ordered-noise-bank-v1\0")
    digest.update(bytes(registered_schedule))
    manifest: list[tuple[str, ...]] = []
    expected_shape: Optional[tuple[int, ...]] = None
    expected_device: Optional[torch.device] = None
    for step_index, cell in enumerate(value):
        if (
            isinstance(cell, (str, bytes))
            or not isinstance(cell, Sequence)
            or len(cell) != registered_schedule[step_index]
        ):
            raise SAICSourceStateFlowTransportError(
                f"noise-bank cell {step_index} must contain exactly "
                f"{registered_schedule[step_index]} candidates"
            )
        cell_manifest: list[str] = []
        for candidate_index, item in enumerate(cell):
            tensor = _clean_state(
                item,
                label=f"fresh_noise_schedule[{step_index}][{candidate_index}]",
                expected_shape=expected_shape,
                expected_device=expected_device,
            )
            if expected_shape is None:
                expected_shape = tuple(map(int, tensor.shape))
                expected_device = tensor.device
            content_sha256 = _tensor_content_sha256(tensor)
            cell_manifest.append(content_sha256)
            digest.update(struct.pack(">II", step_index, candidate_index))
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(struct.pack(">I", tensor.ndim))
            for extent in tensor.shape:
                digest.update(struct.pack(">Q", int(extent)))
            digest.update(bytes.fromhex(content_sha256))
        manifest.append(tuple(cell_manifest))
    return digest.hexdigest(), tuple(manifest)


def noise_bank_sha256(
    value: Any,
    *,
    candidate_schedule: tuple[int, ...] = EXACT_CANDIDATE_SCHEDULE,
) -> str:
    """Hash actual ordered FP32 noise tensors and exact candidate cells.

    This proves content equality to a pre-registered bank.  It does not prove
    Gaussianity, generator implementation, or seed provenance.
    """

    return _noise_bank_manifest(
        value, candidate_schedule=candidate_schedule
    )[0]


def _assert_noise_cell_manifest(
    value: Sequence[torch.Tensor],
    expected: tuple[str, ...],
    *,
    step_index: int,
) -> None:
    if len(value) != len(expected):
        raise SAICSourceStateFlowTransportError(
            f"noise-bank cell {step_index} cardinality changed after registration"
        )
    for candidate_index, (tensor, content_sha256) in enumerate(zip(value, expected)):
        if _tensor_content_sha256(tensor) != content_sha256:
            raise SAICSourceStateFlowTransportError(
                f"noise-bank tensor [{step_index}][{candidate_index}] changed "
                "after digest registration"
            )


def _finite(value: Any, *, label: str) -> float:
    if type(value) not in (int, float):
        raise SAICSourceStateFlowTransportError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SAICSourceStateFlowTransportError(f"{label} must be finite")
    return result


def _unit_time(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise SAICSourceStateFlowTransportError(f"{label} must be in [0,1]")
    return result


def _positive_time(value: Any, *, label: str) -> float:
    result = _unit_time(value, label=label)
    if result == 0.0:
        raise SAICSourceStateFlowTransportError(f"{label} must be positive")
    return result


def _positive_finite(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if result <= 0.0:
        raise SAICSourceStateFlowTransportError(f"{label} must be positive")
    return result


def _stripped(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SAICSourceStateFlowTransportError(
            f"{label} must be a nonempty stripped string"
        )
    return value


def _sha256(value: Any, *, label: str) -> str:
    result = _stripped(value, label=label)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise SAICSourceStateFlowTransportError(
            f"{label} must be a lowercase SHA-256 hex digest"
        )
    return result


def _caption(value: Any, *, label: str) -> str:
    return _stripped(value, label=label)


def _clean_tensor(
    value: Any,
    *,
    label: str,
    expected_shape: Optional[tuple[int, ...]] = None,
    expected_device: Optional[torch.device] = None,
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or (expected_shape is not None and tuple(map(int, value.shape)) != expected_shape)
        or (expected_device is not None and value.device != expected_device)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SAICSourceStateFlowTransportError(
            f"{label} must be a detached finite FP32 tensor with registered geometry"
        )
    return value


def _clean_state(
    value: Any,
    *,
    label: str,
    expected_shape: Optional[tuple[int, ...]] = None,
    expected_device: Optional[torch.device] = None,
) -> torch.Tensor:
    result = _clean_tensor(
        value,
        label=label,
        expected_shape=expected_shape,
        expected_device=expected_device,
    )
    if (
        result.ndim != 5
        or int(result.shape[0]) != 1
        or int(result.shape[1]) != 16
        or int(result.shape[2]) != 21
        or int(result.shape[-2]) <= 0
        or int(result.shape[-1]) <= 0
    ):
        raise SAICSourceStateFlowTransportError(
            f"{label} must have exact [1,16,21,H,W] Bernini geometry"
        )
    return result


def _validate_state_triplet(
    source_clean: Any, edit_clean: Any, noise: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = _clean_state(source_clean, label="source_clean")
    shape = tuple(map(int, source.shape))
    edit = _clean_state(
        edit_clean,
        label="edit_clean",
        expected_shape=shape,
        expected_device=source.device,
    )
    candidate_noise = _clean_state(
        noise,
        label="noise",
        expected_shape=shape,
        expected_device=source.device,
    )
    return source, edit, candidate_noise


def _tensor_sequence(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, ...],
    expected_device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise SAICSourceStateFlowTransportError(
            f"{label} must be a nonempty tensor sequence"
        )
    return tuple(
        _clean_state(
            item,
            label=f"{label}[{index}]",
            expected_shape=expected_shape,
            expected_device=expected_device,
        )
        for index, item in enumerate(value)
    )


def _velocity(value: Any, *, like: torch.Tensor, label: str) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or tuple(map(int, value.shape)) != tuple(map(int, like.shape))
        or value.device != like.device
        or value.dtype not in (torch.float32, torch.bfloat16)
        or value.layout != torch.strided
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SAICSourceStateFlowTransportError(
            f"{label} must be a detached finite FP32/BF16 guided velocity"
        )
    return value


__all__ = [
    "ANC_FULL_RETENTION_TIME",
    "CANDIDATE_CONTINUATION_POLICY",
    "DEFAULT_EARLY_CANDIDATES",
    "DEFAULT_EARLY_STEPS",
    "DEFAULT_SGA_TEMPERATURE",
    "EXACT_CANDIDATE_SCHEDULE",
    "EXACT_SINGLE_CANDIDATE_SCHEDULE",
    "EXPECTED_IMAGE_GUIDANCE_SCALE",
    "EXPECTED_R2V_I0_BRANCH_ORDER",
    "EXPECTED_STEPS",
    "EXPECTED_T2V_V2V_BRANCH_ORDER",
    "FlowTransportDiagnostics",
    "FlowTransportRollout",
    "FlowTransportRolloutConfig",
    "FlowTransportRolloutDiagnostics",
    "FlowTransportStep",
    "FlowTransportStepBinding",
    "GUIDANCE_MODE_BY_FIELD_REGIME",
    "GUIDED_VELOCITY_QUERIES_PER_CANDIDATE",
    "NativeGuidanceBinding",
    "R2V_I0_RAW_TRANSFORMER_FORWARDS_PER_CANDIDATE",
    "R2V_I0_RAW_TRANSFORMER_FORWARDS_PER_GUIDED_QUERY",
    "RAW_TRANSFORMER_FORWARDS_PER_CANDIDATE",
    "REGISTERED_AGGREGATION_MODES",
    "REGISTERED_CANDIDATE_SCHEDULES",
    "REGISTERED_FIELD_REGIMES",
    "SAICSourceStateFlowTransportError",
    "SCHEMA_VERSION",
    "T2V_V2V_RAW_TRANSFORMER_FORWARDS_PER_CANDIDATE",
    "T2V_V2V_RAW_TRANSFORMER_FORWARDS_PER_GUIDED_QUERY",
    "VelocityQueryRequest",
    "VISUAL_CONDITION_SCOPE_BY_FIELD_REGIME",
    "anc_retention",
    "build_source_target_query_states",
    "candidate_count_for_step",
    "correlate_noise",
    "noise_bank_sha256",
    "run_exact40_source_state_flow_transport",
    "similarity_guided_delta",
    "sigma_schedule_sha256",
    "source_state_flow_step",
]
