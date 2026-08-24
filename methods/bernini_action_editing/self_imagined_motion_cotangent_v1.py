#!/usr/bin/env python3
"""Per-query self-imagined motion cotangent for frozen Bernini.

This module is deliberately smaller than an editor trainer.  For one public
``source_video + instruction`` query, the same frozen Bernini first produces
one pure-T2V action owner with a pre-registered seed.  At two pre-registered
hidden-query seeds, the owner and the current RV2V candidate are independently
queried at native schedule index 33:

    R = H_15(x_sigma, action) - H_15(x_sigma, scene_matched_noop).

Only ``Phi(R_owner)`` -- a detached prompt-relative temporal quotient -- may
cross from the owner graph into the editor-side scalar.  Owner RGB, clean
latent, Gaussian, text condition, velocity, reference, target and donor are
forbidden editor inputs.  The per-seed editor scalar is

    S_s = cos(Phi(R_editor_s), stopgrad(Phi(R_owner_s)))

and the live bridge supplies ``q_s = grad(clean_editor, S_s)``.  The two seeds
remain separate.  They are neither ranked nor averaged.  Each projected
cotangent creates one fixed-dose ``+q_s/-q_s`` latent pair which must be fully
decoded to exact81 and judged symmetrically.

The code below implements the strict tensor core, registry validation, a
minimal adapter for ``STARCLiveVJPBridgeV1``, the mask-free nuisance quotient,
and the two-seed runtime orchestration boundary.  It has no optimizer, no
parameter update, no mask/track/pose/flow input and no outcome authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-self-imagined-motion-cotangent-v1"
REGISTRY_SCHEMA_VERSION = "bernini-self-imagined-motion-cotangent-core2-registry-v1"
TEMPLATE_SCHEMA_VERSION = "bernini-self-imagined-motion-template-v1"
PAIR_SCHEMA_VERSION = "bernini-self-imagined-motion-symmetric-pair-v1"
GATE_SCHEMA_VERSION = "bernini-self-imagined-motion-direction-gate-v1"
PLAN_SCHEMA_VERSION = "bernini-self-imagined-motion-dual4-plan-v1"
SPECIFICITY_SCHEMA_VERSION = "bernini-self-imagined-motion-specificity-gate-v1"

HOOK_COORDINATE = "block.15.output"
SCHEDULE_INDEX = 33
NATIVE_TIMESTEP = 516
NATIVE_SIGMA = 0.5161304473876953
RESIDUAL_SHAPE = (1, 21, 16, 1536)
LATENT_PHASES = 21
HIDDEN_SIZE = 1536
QUERY_SEED_COUNT = 2
SP_SIZE = 4
EXPECTED_CELL_IDS = ("dog", "human")
TEMPORAL_LAGS = (1, 2, 4)
HOLD_PHASES = 4

EXTERNAL_INFERENCE_INPUTS = ("source_video", "instruction")
FORBIDDEN_AUXILIARY_INPUTS = (
    "mask",
    "track",
    "pose",
    "flow",
    "detector_box",
    "swept_tube",
)
FORBIDDEN_OWNER_TO_EDITOR_CHANNELS = (
    "rgb",
    "clean_latent",
    "gaussian",
    "text_condition",
    "velocity",
    "reference",
    "initial_noise",
    "target",
    "donor",
    "parameter_gradient",
)
ALLOWED_OWNER_TO_EDITOR_CHANNEL = (
    "detached_normalized_prompt_relative_temporal_hidden_quotient"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class SelfImaginedCotangentContractError(RuntimeError):
    """A registry, tensor, live proof, intervention or direction gate failed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SelfImaginedCotangentContractError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SelfImaginedCotangentContractError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise SelfImaginedCotangentContractError(
            f"{label} must be a path-safe identifier"
        )
    return value


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency-light hosts
        raise SelfImaginedCotangentContractError(
            "PyTorch is required for the motion-cotangent tensor core"
        ) from error
    return torch


def tensor_value_digest(value: Any, *, label: str) -> str:
    """Hash a detached tensor with dtype/shape and exact contiguous bytes."""

    torch = _require_torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SelfImaginedCotangentContractError(
            f"{label} must be a detached finite real tensor"
        )
    owned = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "dtype": str(owned.dtype),
        "shape": list(map(int, owned.shape)),
        "layout": str(owned.layout),
    }
    if hasattr(owned, "untyped_storage"):
        raw = bytes(owned.untyped_storage())
    else:  # Torch 1.12 contract-test environments predate untyped_storage.
        payload = io.BytesIO()
        typed_storage = owned.storage()
        untyped = typed_storage._untyped()
        untyped._write_file(payload, False, False, 1)
        raw = payload.getvalue()
    expected = int(owned.numel()) * int(owned.element_size())
    if len(raw) != expected:
        raise SelfImaginedCotangentContractError(f"{label} storage closure differs")
    return hashlib.sha256(canonical_json_bytes(metadata) + b"\x00" + raw).hexdigest()


def _finite_positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelfImaginedCotangentContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SelfImaginedCotangentContractError(
            f"{label} must be positive finite"
        )
    return result


def _finite_cosine_threshold(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelfImaginedCotangentContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise SelfImaginedCotangentContractError(
            f"{label} must be finite in [-1,1]"
        )
    return result


def _exact_keys(value: Any, keys: Iterable[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise SelfImaginedCotangentContractError(f"{label} field closure differs")
    return dict(value)


@dataclass(frozen=True)
class MotionQuotientConfig:
    """Fixed, training-free temporal feature quotient."""

    temporal_lags: tuple[int, ...] = TEMPORAL_LAGS
    hold_phases: int = HOLD_PHASES
    level_weight: float = 1.0
    lag_weight: float = 1.0
    boundary_weight: float = 1.0
    hold_weight: float = 0.5
    spatial_rms_epsilon: float = 1.0e-8
    minimum_feature_norm: float = 1.0e-8

    def __post_init__(self) -> None:
        if (
            self.temporal_lags != TEMPORAL_LAGS
            or type(self.hold_phases) is not int
            or not 1 <= self.hold_phases <= LATENT_PHASES // 2
        ):
            raise SelfImaginedCotangentContractError(
                "temporal quotient lag/hold contract differs"
            )
        for name in (
            "level_weight",
            "lag_weight",
            "boundary_weight",
            "hold_weight",
            "spatial_rms_epsilon",
            "minimum_feature_norm",
        ):
            _finite_positive(getattr(self, name), label=name)

    def receipt(self) -> dict[str, Any]:
        return {
            "temporal_dc_removed": True,
            "spatial_orderless_signature": [
                "signed_channel_mean",
                "centered_channel_rms_second_moment",
            ],
            "spatial_rms_epsilon": self.spatial_rms_epsilon,
            "temporal_lags": list(self.temporal_lags),
            "hold_phases": self.hold_phases,
            "component_weights": {
                "level": self.level_weight,
                "lag_each": self.lag_weight,
                "start_to_terminal_boundary": self.boundary_weight,
                "initial_and_terminal_hold_each": self.hold_weight,
            },
            "per_component_length_normalization": True,
            "minimum_feature_norm": self.minimum_feature_norm,
            "learned_parameters": 0,
        }


def _validate_residual(
    residual: Any, *, label: str, require_input_grad: bool
) -> Any:
    torch = _require_torch()
    if (
        not isinstance(residual, torch.Tensor)
        or residual.ndim != 4
        or int(residual.shape[0]) != 1
        or int(residual.shape[1]) != LATENT_PHASES
        or int(residual.shape[2]) <= 0
        or int(residual.shape[3]) != RESIDUAL_SHAPE[-1]
        or residual.dtype != torch.float32
        or residual.device.type == "meta"
        or not bool(torch.isfinite(residual).all().item())
    ):
        raise SelfImaginedCotangentContractError(
            f"{label} must be finite FP32 [1,{LATENT_PHASES},K,{RESIDUAL_SHAPE[-1]}] with K>0"
        )
    if require_input_grad and (
        not residual.requires_grad or residual.grad_fn is None
    ):
        raise SelfImaginedCotangentContractError(
            f"{label} must remain connected to the current RV2V graph"
        )
    if not require_input_grad and (residual.requires_grad or residual.grad_fn is not None):
        raise SelfImaginedCotangentContractError(
            f"{label} must be detached from the pure-T2V owner graph"
        )
    return residual


def _length_normalize_component(value: Any, *, temporal_length: int) -> Any:
    return value.reshape(value.shape[0], -1) / math.sqrt(float(temporal_length))


def temporal_motion_quotient(
    residual: Any,
    *,
    config: MotionQuotientConfig = MotionQuotientConfig(),
    require_input_grad: bool = False,
) -> Any:
    """Return a spatial-orderless, temporal-DC-invariant feature bundle.

    ``R`` is already prompt-relative (action minus scene-matched no-op) at one
    exact noisy state.  ``Phi`` additionally removes every temporally static
    hidden coordinate, pools spatial coordinates into signed channel mean plus
    centered channel RMS/second moment, retains ordered lag-1/2/4 changes, adds
    an explicit start-to-terminal boundary, and represents deviations within
    the initial and terminal hold windows.  No spatially varying mask,
    matching, OT or content-derived routing is used.
    """

    torch = _require_torch()
    value = _validate_residual(
        residual, label="prompt-relative hidden residual", require_input_grad=require_input_grad
    )
    if not isinstance(config, MotionQuotientConfig):
        raise SelfImaginedCotangentContractError("motion quotient config differs")

    # Remove static actor/scene coordinates *before* the nonlinear second
    # moment so a temporally constant identity term cannot leak through RMS.
    centered_hidden = value - value.mean(dim=1, keepdim=True)
    signed_mean = centered_hidden.mean(dim=2)
    centered_rms = torch.sqrt(
        centered_hidden.square().mean(dim=2) + config.spatial_rms_epsilon
    ) - math.sqrt(config.spatial_rms_epsilon)
    # Both statistics are invariant to any permutation/cyclic shift of the
    # spatial axis.  The signed mean keeps directional channel information;
    # RMS alone is explicitly not treated as an action representation.
    value = torch.cat((signed_mean, centered_rms), dim=2)
    centered = value - value.mean(dim=1, keepdim=True)
    components = [
        config.level_weight
        * _length_normalize_component(centered, temporal_length=LATENT_PHASES)
    ]
    for lag in config.temporal_lags:
        prefix = torch.zeros_like(value[:, :lag])
        difference = torch.cat(
            (prefix, value[:, lag:] - value[:, :-lag]), dim=1
        )
        components.append(
            config.lag_weight
            * _length_normalize_component(
                difference, temporal_length=LATENT_PHASES - lag
            )
        )

    hold = config.hold_phases
    initial = value[:, :hold]
    terminal = value[:, -hold:]
    initial_mean = initial.mean(dim=1, keepdim=True)
    terminal_mean = terminal.mean(dim=1, keepdim=True)
    boundary = terminal_mean - initial_mean
    initial_deviation = initial - initial_mean
    terminal_deviation = terminal - terminal_mean
    components.extend(
        (
            config.boundary_weight
            * _length_normalize_component(boundary, temporal_length=1),
            config.hold_weight
            * _length_normalize_component(initial_deviation, temporal_length=hold),
            config.hold_weight
            * _length_normalize_component(terminal_deviation, temporal_length=hold),
        )
    )
    feature = torch.cat(components, dim=1).float().contiguous()
    if (
        feature.ndim != 2
        or feature.shape[0] != 1
        or not bool(torch.isfinite(feature).all().item())
        or (require_input_grad and (not feature.requires_grad or feature.grad_fn is None))
    ):
        raise SelfImaginedCotangentContractError(
            "temporal motion quotient graph/value closure differs"
        )
    return feature


@dataclass(frozen=True)
class FrozenOwnerTemplate:
    query_seed: int
    unit_feature: Any
    owner_spatial_coordinates: int
    hidden_size: int
    raw_feature_norm: float
    unit_feature_digest: str
    quotient_config_digest: str
    owner_provenance_digest: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": TEMPLATE_SCHEMA_VERSION,
            "query_seed": self.query_seed,
            "template_shape": list(map(int, self.unit_feature.shape)),
            "template_dtype": str(self.unit_feature.dtype),
            "raw_feature_norm": self.raw_feature_norm,
            "owner_spatial_coordinates": self.owner_spatial_coordinates,
            "hidden_size": self.hidden_size,
            "candidate_spatial_coordinates_may_differ": True,
            "spatial_orderless_signature": [
                "signed_channel_mean",
                "centered_channel_rms_second_moment",
            ],
            "unit_feature_digest": self.unit_feature_digest,
            "quotient_config_digest": self.quotient_config_digest,
            "owner_provenance_digest": self.owner_provenance_digest,
            "hook_coordinate": HOOK_COORDINATE,
            "prompt_relative_action_minus_noop": True,
            "stop_gradient": True,
            "allowed_owner_to_editor_channel": ALLOWED_OWNER_TO_EDITOR_CHANNEL,
            "forbidden_owner_to_editor_channels": list(
                FORBIDDEN_OWNER_TO_EDITOR_CHANNELS
            ),
            "owner_media_or_primal_tensor_retained": False,
            "learned_global_critic": False,
        }


def build_frozen_owner_template(
    owner_residual: Any,
    *,
    query_seed: int,
    owner_provenance: Mapping[str, Any],
    config: MotionQuotientConfig = MotionQuotientConfig(),
) -> FrozenOwnerTemplate:
    """Collapse one detached owner residual into its only permitted channel."""

    torch = _require_torch()
    if type(query_seed) is not int or query_seed < 0:
        raise SelfImaginedCotangentContractError("query seed must be a nonnegative int")
    provenance = _exact_keys(
        owner_provenance,
        (
            "cell_id",
            "owner_generation_seed",
            "query_seed",
            "owner_mode",
            "owner_exact81_action_audit_passed",
            "owner_used_source_video_condition",
        ),
        label="owner provenance",
    )
    if (
        provenance["query_seed"] != query_seed
        or type(provenance["owner_generation_seed"]) is not int
        or provenance["owner_generation_seed"] < 0
        or provenance["owner_mode"] != "frozen_bernini_pure_t2v"
        or provenance["owner_exact81_action_audit_passed"] is not True
        or provenance["owner_used_source_video_condition"] is not False
    ):
        raise SelfImaginedCotangentContractError("owner provenance contract differs")
    _safe_id(provenance["cell_id"], label="owner cell ID")
    feature = temporal_motion_quotient(
        owner_residual, config=config, require_input_grad=False
    ).detach()
    norm = torch.linalg.vector_norm(feature)
    if (
        not bool(torch.isfinite(norm).item())
        or float(norm.item()) < config.minimum_feature_norm
    ):
        raise SelfImaginedCotangentContractError(
            "owner prompt-relative temporal quotient is degenerate"
        )
    unit = (feature / norm).detach().contiguous()
    if unit.requires_grad or unit.grad_fn is not None:
        raise SelfImaginedCotangentContractError("owner template did not detach")
    return FrozenOwnerTemplate(
        query_seed=query_seed,
        unit_feature=unit,
        owner_spatial_coordinates=int(owner_residual.shape[2]),
        hidden_size=int(owner_residual.shape[3]),
        raw_feature_norm=float(norm.item()),
        unit_feature_digest=tensor_value_digest(unit, label="owner unit template"),
        quotient_config_digest=object_sha256(config.receipt()),
        owner_provenance_digest=object_sha256(provenance),
    )


@dataclass(frozen=True)
class PerQueryScoreOutput:
    score: Any
    candidate_feature_norm: Any


def make_frozen_per_query_scorer(
    template: FrozenOwnerTemplate,
    *,
    config: MotionQuotientConfig = MotionQuotientConfig(),
) -> Any:
    """Create the frozen ``forward_sketched_residual`` bridge adapter."""

    torch = _require_torch()
    if not isinstance(template, FrozenOwnerTemplate):
        raise SelfImaginedCotangentContractError("owner template type differs")
    if template.quotient_config_digest != object_sha256(config.receipt()):
        raise SelfImaginedCotangentContractError(
            "owner template and scorer quotient configs differ"
        )

    class FrozenPerQueryMotionScorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "owner_unit_feature",
                template.unit_feature.detach().contiguous().clone(),
                persistent=True,
            )
            self.query_seed = template.query_seed
            self.template_digest = template.unit_feature_digest
            self.config = config
            self.requires_grad_(False)
            self.eval()

        def forward_sketched_residual(
            self, residual: Any, *, require_input_grad: bool
        ) -> PerQueryScoreOutput:
            feature = temporal_motion_quotient(
                residual,
                config=self.config,
                require_input_grad=require_input_grad,
            )
            norm = torch.linalg.vector_norm(feature, dim=1, keepdim=True)
            if (
                not bool(torch.isfinite(norm).all().item())
                or float(norm.min().item()) < self.config.minimum_feature_norm
            ):
                raise SelfImaginedCotangentContractError(
                    "current RV2V temporal quotient is degenerate"
                )
            owner = self.owner_unit_feature.to(
                device=feature.device, dtype=feature.dtype
            )
            if owner.shape != feature.shape:
                raise SelfImaginedCotangentContractError(
                    "owner/current temporal quotient shapes differ"
                )
            score = ((feature / norm) * owner).sum(dim=1)
            if (
                score.numel() != 1
                or not bool(torch.isfinite(score).all().item())
                or (require_input_grad and (not score.requires_grad or score.grad_fn is None))
            ):
                raise SelfImaginedCotangentContractError(
                    "per-query cosine score graph differs"
                )
            return PerQueryScoreOutput(
                score=score.reshape(()), candidate_feature_norm=norm.reshape(())
            )

    return FrozenPerQueryMotionScorer()


@dataclass(frozen=True)
class PromptSpecificityAudit:
    """Non-compensating A-vs-reverse/null margin at one exact x_sigma."""

    query_seed: int
    same_x_sigma_binding_digest: str
    action_score: float
    reverse_wrong_family_score: float
    common_scene_null_score: float
    reverse_wrong_family_margin: float
    common_scene_null_margin: float
    minimum_margin: float
    passed: bool

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SPECIFICITY_SCHEMA_VERSION,
            "query_seed": self.query_seed,
            "same_x_sigma_binding_digest": self.same_x_sigma_binding_digest,
            "query_order": ["action", "reverse_wrong_family", "common_scene_null"],
            "action_score": self.action_score,
            "reverse_wrong_family_score": self.reverse_wrong_family_score,
            "common_scene_null_score": self.common_scene_null_score,
            "reverse_wrong_family_margin": self.reverse_wrong_family_margin,
            "common_scene_null_margin": self.common_scene_null_margin,
            "minimum_margin": self.minimum_margin,
            "all_margins_pass_without_compensation": self.passed,
            "generic_motion_reward_sufficient": False,
            "same_topology_reverse_query_required": True,
        }


def _detached_template_score(
    residual: Any,
    template: FrozenOwnerTemplate,
    *,
    config: MotionQuotientConfig,
    allow_exact_zero: bool,
    label: str,
) -> float:
    torch = _require_torch()
    value = _validate_residual(residual, label=label, require_input_grad=False)
    if allow_exact_zero and int(torch.count_nonzero(value).item()) == 0:
        return 0.0
    feature = temporal_motion_quotient(
        value, config=config, require_input_grad=False
    )
    norm = torch.linalg.vector_norm(feature)
    if (
        not bool(torch.isfinite(norm).item())
        or float(norm.item()) < config.minimum_feature_norm
    ):
        raise SelfImaginedCotangentContractError(f"{label} quotient is degenerate")
    if feature.shape != template.unit_feature.shape:
        raise SelfImaginedCotangentContractError(f"{label} template shape differs")
    unit = feature / norm
    score = (unit * template.unit_feature.to(unit.device)).sum()
    if not bool(torch.isfinite(score).item()):
        raise SelfImaginedCotangentContractError(f"{label} score is non-finite")
    return float(score.item())


def audit_prompt_specificity(
    template: FrozenOwnerTemplate,
    *,
    action_residual: Any,
    reverse_wrong_family_residual: Any,
    common_scene_null_residual: Any,
    same_x_sigma_binding_digest: str,
    minimum_margin: float,
    config: MotionQuotientConfig = MotionQuotientConfig(),
) -> PromptSpecificityAudit:
    """Reject a per-query template that merely rewards generic motion energy.

    All three prompts must be evaluated on the same owner or current-candidate
    noisy state.  The caller seals that state in ``same_x_sigma_binding_digest``.
    The scene-null residual must be byte-exact zero (the common-null branch
    minus itself), while the same-topology reverse/wrong-family residual must be
    independently non-degenerate.  Both margins pass separately; no averaging
    is permitted.
    """

    if not isinstance(template, FrozenOwnerTemplate):
        raise SelfImaginedCotangentContractError("specificity template differs")
    binding = _sha256(
        same_x_sigma_binding_digest, label="same-x-sigma binding digest"
    )
    margin = _finite_positive(minimum_margin, label="specificity margin")
    if margin >= 2.0:
        raise SelfImaginedCotangentContractError(
            "cosine specificity margin must be below two"
        )
    action_score = _detached_template_score(
        action_residual,
        template,
        config=config,
        allow_exact_zero=False,
        label="action residual",
    )
    reverse_score = _detached_template_score(
        reverse_wrong_family_residual,
        template,
        config=config,
        allow_exact_zero=False,
        label="same-topology reverse/wrong-family residual",
    )
    torch = _require_torch()
    null_value = _validate_residual(
        common_scene_null_residual,
        label="common-scene-null residual",
        require_input_grad=False,
    )
    if int(torch.count_nonzero(null_value).item()) != 0:
        raise SelfImaginedCotangentContractError(
            "common-scene-null residual must be exact H(c0)-H(c0)=0"
        )
    null_score = _detached_template_score(
        null_value,
        template,
        config=config,
        allow_exact_zero=True,
        label="common-scene-null residual",
    )
    reverse_margin = action_score - reverse_score
    null_margin = action_score - null_score
    passed = reverse_margin >= margin and null_margin >= margin
    return PromptSpecificityAudit(
        query_seed=template.query_seed,
        same_x_sigma_binding_digest=binding,
        action_score=action_score,
        reverse_wrong_family_score=reverse_score,
        common_scene_null_score=null_score,
        reverse_wrong_family_margin=reverse_margin,
        common_scene_null_margin=null_margin,
        minimum_margin=margin,
        passed=passed,
    )


def cosine_similarity(value_a: Any, value_b: Any, *, label: str) -> float:
    torch = _require_torch()
    if (
        not isinstance(value_a, torch.Tensor)
        or not isinstance(value_b, torch.Tensor)
        or value_a.shape != value_b.shape
        or value_a.numel() == 0
        or not bool(torch.isfinite(value_a).all().item())
        or not bool(torch.isfinite(value_b).all().item())
    ):
        raise SelfImaginedCotangentContractError(f"{label} tensor closure differs")
    a = value_a.detach().float().reshape(-1)
    b = value_b.detach().float().reshape(-1)
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if not bool(torch.isfinite(denominator).item()) or float(denominator.item()) <= 0.0:
        raise SelfImaginedCotangentContractError(f"{label} contains zero norm")
    result = float((torch.dot(a, b) / denominator).item())
    if not math.isfinite(result):
        raise SelfImaginedCotangentContractError(f"{label} cosine is non-finite")
    return max(-1.0, min(1.0, result))


@dataclass(frozen=True)
class TwoSeedTemplateAudit:
    ordered_query_seeds: tuple[int, int]
    cosine: float
    minimum_cosine: float
    passed: bool
    no_seed_selection: bool = True

    def receipt(self) -> dict[str, Any]:
        return {
            "ordered_query_seeds": list(self.ordered_query_seeds),
            "owner_template_cosine": self.cosine,
            "minimum_owner_template_cosine": self.minimum_cosine,
            "passed": self.passed,
            "seed_ranking_or_selection": False,
            "seed_averaging": False,
        }


def audit_two_seed_templates(
    templates: Sequence[FrozenOwnerTemplate], *, minimum_cosine: float
) -> TwoSeedTemplateAudit:
    threshold = _finite_cosine_threshold(
        minimum_cosine, label="minimum owner template cosine"
    )
    if len(templates) != QUERY_SEED_COUNT or any(
        not isinstance(row, FrozenOwnerTemplate) for row in templates
    ):
        raise SelfImaginedCotangentContractError(
            "exactly two frozen owner templates are required"
        )
    seeds = tuple(row.query_seed for row in templates)
    if len(set(seeds)) != QUERY_SEED_COUNT:
        raise SelfImaginedCotangentContractError("owner template query seeds alias")
    value = cosine_similarity(
        templates[0].unit_feature,
        templates[1].unit_feature,
        label="two-seed owner templates",
    )
    return TwoSeedTemplateAudit(
        ordered_query_seeds=seeds,  # type: ignore[arg-type]
        cosine=value,
        minimum_cosine=threshold,
        passed=value >= threshold,
    )


@dataclass(frozen=True)
class ProjectedCotangent:
    tensor: Any
    raw_norm: float
    projected_norm: float
    projection_survival_cosine: float
    phase0_max_abs: float
    temporal_sum_max_abs: float
    spatial_affine_max_abs_dot: float
    tensor_digest: str

    def receipt(self) -> dict[str, Any]:
        return {
            "raw_norm": self.raw_norm,
            "projected_norm": self.projected_norm,
            "projection_survival_cosine": self.projection_survival_cosine,
            "phase0_max_abs": self.phase0_max_abs,
            "temporal_sum_max_abs": self.temporal_sum_max_abs,
            "spatial_affine_max_abs_dot": self.spatial_affine_max_abs_dot,
            "projected_tensor_digest": self.tensor_digest,
            "phase0_exactly_protected": True,
            "temporal_dc_removed_over_phases_1_to_20": True,
            "global_spatial_basis_removed": ["constant", "x_ramp", "y_ramp"],
            "mask_or_content_derived_projection": False,
            "identity_or_camera_preservation_proven": False,
        }


def _orthonormal_spatial_affine_basis(
    height: int, width: int, *, device: Any, dtype: Any
) -> Any:
    torch = _require_torch()
    if type(height) is not int or type(width) is not int or height < 2 or width < 2:
        raise SelfImaginedCotangentContractError(
            "spatial affine projection requires H,W >= 2"
        )
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rows = torch.stack((torch.ones_like(xx), xx, yy), dim=0).reshape(3, -1)
    rows = rows / torch.linalg.vector_norm(rows, dim=1, keepdim=True)
    gram = rows @ rows.transpose(0, 1)
    if not bool(
        torch.allclose(
            gram,
            torch.eye(3, device=device, dtype=dtype),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
    ):
        raise SelfImaginedCotangentContractError(
            "spatial affine basis did not orthonormalize"
        )
    return rows


def project_mask_free_nuisance_cotangent(
    gradient: Any,
    *,
    minimum_norm: float = 1.0e-12,
    minimum_survival_cosine: float = 0.0,
) -> ProjectedCotangent:
    """Project away registered nuisance modes without a spatial mask.

    This is an orthogonal projection on separate tensor axes.  Phase zero is
    fixed exactly; phases 1..20 have zero temporal mean at every C/H/W; and
    each C/T slice is orthogonal to the global spatial constant/x/y basis.
    These constraints reduce static appearance and low-order global drift but
    do *not* prove identity or camera preservation; exact81 outcome gates do.
    """

    torch = _require_torch()
    threshold = _finite_positive(minimum_norm, label="minimum projected norm")
    survival_threshold = _finite_cosine_threshold(
        minimum_survival_cosine, label="minimum projection survival cosine"
    )
    if (
        not isinstance(gradient, torch.Tensor)
        or gradient.ndim != 5
        or tuple(map(int, gradient.shape[:3]))[:2] != (1, 16)
        or int(gradient.shape[2]) != LATENT_PHASES
        or gradient.dtype != torch.float32
        or gradient.device.type == "meta"
        or not bool(torch.isfinite(gradient).all().item())
    ):
        raise SelfImaginedCotangentContractError(
            "latent cotangent must be finite FP32 [1,16,21,H,W]"
        )
    raw = gradient.detach().float().contiguous()
    raw_norm_tensor = torch.linalg.vector_norm(raw)
    if (
        not bool(torch.isfinite(raw_norm_tensor).item())
        or float(raw_norm_tensor.item()) < threshold
    ):
        raise SelfImaginedCotangentContractError("raw latent cotangent is degenerate")

    # Orthogonal projection onto {q_0=0, sum_{t=1}^{20} q_t=0}.
    later = raw[:, :, 1:] - raw[:, :, 1:].mean(dim=2, keepdim=True)
    temporal = torch.cat((torch.zeros_like(raw[:, :, :1]), later), dim=2)

    # Orthogonal projection off the content-independent spatial affine basis.
    height, width = int(raw.shape[-2]), int(raw.shape[-1])
    basis = _orthonormal_spatial_affine_basis(
        height, width, device=raw.device, dtype=raw.dtype
    )
    flat = temporal.reshape(-1, height * width)
    coefficients = flat @ basis.transpose(0, 1)
    projected_flat = flat - coefficients @ basis
    projected = projected_flat.reshape_as(raw).contiguous()
    projected[:, :, 0].zero_()

    projected_norm_tensor = torch.linalg.vector_norm(projected)
    if (
        not bool(torch.isfinite(projected_norm_tensor).item())
        or float(projected_norm_tensor.item()) < threshold
    ):
        raise SelfImaginedCotangentContractError(
            "nuisance quotient removed the entire cotangent"
        )
    survival = cosine_similarity(raw, projected, label="raw/projected cotangent")
    if survival < survival_threshold:
        raise SelfImaginedCotangentContractError(
            "action cotangent did not survive the nuisance projection"
        )
    temporal_sum_max = float(projected[:, :, 1:].sum(dim=2).abs().max().item())
    affine_dot_max = float(
        (projected.reshape(-1, height * width) @ basis.transpose(0, 1))
        .abs()
        .max()
        .item()
    )
    phase0_max = float(projected[:, :, 0].abs().max().item())
    owned = projected.detach().contiguous()
    return ProjectedCotangent(
        tensor=owned,
        raw_norm=float(raw_norm_tensor.item()),
        projected_norm=float(projected_norm_tensor.item()),
        projection_survival_cosine=survival,
        phase0_max_abs=phase0_max,
        temporal_sum_max_abs=temporal_sum_max,
        spatial_affine_max_abs_dot=affine_dot_max,
        tensor_digest=tensor_value_digest(owned, label="projected cotangent"),
    )


@dataclass(frozen=True)
class SymmetricInterventionPair:
    query_seed: int
    plus: Any
    minus: Any
    delta: Any
    relative_l2_dose: float
    base_norm: float
    delta_norm: float
    plus_digest: str
    minus_digest: str
    delta_digest: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": PAIR_SCHEMA_VERSION,
            "query_seed": self.query_seed,
            "arm_order": ["plus", "minus"],
            "relative_l2_dose": self.relative_l2_dose,
            "base_norm": self.base_norm,
            "delta_norm": self.delta_norm,
            "plus_tensor_digest": self.plus_digest,
            "minus_tensor_digest": self.minus_digest,
            "delta_tensor_digest": self.delta_digest,
            "same_base_and_exact_opposite_delta": True,
            "full_exact81_decode_required": True,
            "seed_or_arm_selection": False,
            "optimizer_or_parameter_update": False,
        }


def make_symmetric_intervention_pair(
    base_clean_latent: Any,
    cotangent: ProjectedCotangent,
    *,
    query_seed: int,
    relative_l2_dose: float,
) -> SymmetricInterventionPair:
    torch = _require_torch()
    dose = _finite_positive(relative_l2_dose, label="relative L2 dose")
    if dose >= 1.0:
        raise SelfImaginedCotangentContractError("relative L2 dose must be below one")
    if type(query_seed) is not int or query_seed < 0:
        raise SelfImaginedCotangentContractError("query seed must be nonnegative int")
    if (
        not isinstance(cotangent, ProjectedCotangent)
        or not isinstance(base_clean_latent, torch.Tensor)
        or tuple(base_clean_latent.shape) != tuple(cotangent.tensor.shape)
        or base_clean_latent.dtype != torch.float32
        or base_clean_latent.device != cotangent.tensor.device
        or not bool(torch.isfinite(base_clean_latent).all().item())
    ):
        raise SelfImaginedCotangentContractError(
            "base clean latent/projected cotangent closure differs"
        )
    base = base_clean_latent.detach().float().contiguous()
    base_norm_tensor = torch.linalg.vector_norm(base)
    if not bool(torch.isfinite(base_norm_tensor).item()) or float(base_norm_tensor.item()) <= 0.0:
        raise SelfImaginedCotangentContractError("base clean latent is degenerate")
    direction_norm = torch.linalg.vector_norm(cotangent.tensor)
    target_delta_norm = dose * float(base_norm_tensor.item())
    delta = (
        cotangent.tensor
        * (target_delta_norm / float(direction_norm.item()))
    ).detach().contiguous()
    plus = (base + delta).detach().contiguous()
    minus = (base - delta).detach().contiguous()
    if not bool(torch.isfinite(plus).all().item()) or not bool(torch.isfinite(minus).all().item()):
        raise SelfImaginedCotangentContractError("symmetric intervention is non-finite")
    observed_delta_norm = float(torch.linalg.vector_norm(delta).item())
    if not math.isclose(
        observed_delta_norm, target_delta_norm, rel_tol=2.0e-6, abs_tol=1.0e-8
    ):
        raise SelfImaginedCotangentContractError("fixed-dose intervention differs")
    return SymmetricInterventionPair(
        query_seed=query_seed,
        plus=plus,
        minus=minus,
        delta=delta,
        relative_l2_dose=dose,
        base_norm=float(base_norm_tensor.item()),
        delta_norm=observed_delta_norm,
        plus_digest=tensor_value_digest(plus, label="plus intervention"),
        minus_digest=tensor_value_digest(minus, label="minus intervention"),
        delta_digest=tensor_value_digest(delta, label="intervention delta"),
    )


@dataclass(frozen=True)
class LiveSeedProbe:
    query_seed: int
    live_proof: Any
    projected_cotangent: ProjectedCotangent
    intervention: SymmetricInterventionPair


@dataclass(frozen=True)
class LiveTwoSeedProbe:
    ordered_query_seeds: tuple[int, int]
    template_audit: TwoSeedTemplateAudit
    specificity_audits: tuple[PromptSpecificityAudit, PromptSpecificityAudit]
    seed_probes: tuple[LiveSeedProbe, LiveSeedProbe]
    projected_cotangent_cosine: float
    minimum_projected_cotangent_cosine: float
    two_seed_direction_consistent: bool
    optimizer_or_parameter_update: bool = False
    scientific_or_action_editing_claim_authorized: bool = False

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ordered_query_seeds": list(self.ordered_query_seeds),
            "template_audit": self.template_audit.receipt(),
            "specificity_audits": [
                row.receipt() for row in self.specificity_audits
            ],
            "per_seed": [
                {
                    "query_seed": row.query_seed,
                    "live_gradient_norm": float(row.live_proof.gradient_norm),
                    "live_score": float(row.live_proof.critic_score),
                    "projected_cotangent": row.projected_cotangent.receipt(),
                    "intervention": row.intervention.receipt(),
                }
                for row in self.seed_probes
            ],
            "projected_cotangent_cosine": self.projected_cotangent_cosine,
            "minimum_projected_cotangent_cosine": (
                self.minimum_projected_cotangent_cosine
            ),
            "two_seed_direction_consistent": self.two_seed_direction_consistent,
            "seed_averaging": False,
            "seed_ranking_or_selection": False,
            "optimizer_or_parameter_update": False,
            "scientific_or_action_editing_claim_authorized": False,
            "decoded_direction_gate_pending": True,
        }


def run_two_query_seed_live_probe(
    *,
    templates: Sequence[FrozenOwnerTemplate],
    specificity_audits_by_seed: Mapping[int, PromptSpecificityAudit],
    current_clean_latent: Any,
    noises_by_seed: Mapping[int, Any],
    bridge_factory: Callable[[int, Any], Any],
    relative_l2_dose: float,
    minimum_template_cosine: float,
    minimum_projected_cotangent_cosine: float,
    minimum_projection_survival_cosine: float,
    minimum_vjp_norm: float = 1.0e-12,
    quotient_config: MotionQuotientConfig = MotionQuotientConfig(),
) -> LiveTwoSeedProbe:
    """Connect two frozen templates to two independent live bridge VJPs.

    ``bridge_factory(seed, scorer)`` must return a configured
    ``STARCLiveVJPBridgeV1``-compatible object.  The function intentionally
    calls each seed once and retains both results.  A failed template consensus
    stops before any candidate VJP.  A failed q-consensus remains a visible
    NO-GO result; it is never repaired by seed selection or averaging.
    """

    torch = _require_torch()
    template_audit = audit_two_seed_templates(
        templates, minimum_cosine=minimum_template_cosine
    )
    if not template_audit.passed:
        raise SelfImaginedCotangentContractError(
            "two owner query seeds do not define a stable motion template"
        )
    seeds = template_audit.ordered_query_seeds
    if set(noises_by_seed) != set(seeds):
        raise SelfImaginedCotangentContractError(
            "native query noise mapping must contain exactly both registered seeds"
        )
    if set(specificity_audits_by_seed) != set(seeds) or any(
        not isinstance(specificity_audits_by_seed[seed], PromptSpecificityAudit)
        or specificity_audits_by_seed[seed].query_seed != seed
        or not specificity_audits_by_seed[seed].passed
        for seed in seeds
    ):
        raise SelfImaginedCotangentContractError(
            "both query seeds require non-compensating A-vs-reverse/null specificity"
        )
    if (
        not callable(bridge_factory)
        or not isinstance(current_clean_latent, torch.Tensor)
        or current_clean_latent.dtype != torch.float32
        or current_clean_latent.ndim != 5
        or int(current_clean_latent.shape[2]) != LATENT_PHASES
        or not bool(torch.isfinite(current_clean_latent).all().item())
    ):
        raise SelfImaginedCotangentContractError("current clean latent runtime differs")
    q_threshold = _finite_cosine_threshold(
        minimum_projected_cotangent_cosine,
        label="minimum projected cotangent cosine",
    )
    probes = []
    for template in templates:
        seed = template.query_seed
        scorer = make_frozen_per_query_scorer(template, config=quotient_config)
        scorer.to(device=current_clean_latent.device)
        scorer.requires_grad_(False).eval()
        bridge = bridge_factory(seed, scorer)
        if not callable(getattr(bridge, "prove_current_clean_latent_vjp", None)):
            raise SelfImaginedCotangentContractError("live bridge factory result differs")
        clean = (
            current_clean_latent.detach().clone().contiguous().requires_grad_(True)
        )
        noise = noises_by_seed[seed]
        if (
            not isinstance(noise, torch.Tensor)
            or tuple(noise.shape) != tuple(clean.shape)
            or noise.dtype != torch.float32
            or noise.device != clean.device
            or noise.requires_grad
            or noise.grad_fn is not None
            or not bool(torch.isfinite(noise).all().item())
        ):
            raise SelfImaginedCotangentContractError(
                f"query seed {seed} noise tensor differs"
            )
        proof = bridge.prove_current_clean_latent_vjp(
            clean, noise, minimum_norm=minimum_vjp_norm
        )
        gradient = getattr(proof, "gradient", None)
        if (
            not isinstance(gradient, torch.Tensor)
            or tuple(gradient.shape) != tuple(clean.shape)
            or gradient.dtype != torch.float32
            or not bool(torch.isfinite(gradient).all().item())
            or not bool(getattr(proof, "real_sp4_autograd_collective", False))
            or tuple(getattr(proof, "hook_call_order", ())) != ("action", "noop")
        ):
            raise SelfImaginedCotangentContractError(
                f"query seed {seed} live bridge proof differs"
            )
        projected = project_mask_free_nuisance_cotangent(
            gradient,
            minimum_norm=minimum_vjp_norm,
            minimum_survival_cosine=minimum_projection_survival_cosine,
        )
        pair = make_symmetric_intervention_pair(
            current_clean_latent,
            projected,
            query_seed=seed,
            relative_l2_dose=relative_l2_dose,
        )
        probes.append(
            LiveSeedProbe(
                query_seed=seed,
                live_proof=proof,
                projected_cotangent=projected,
                intervention=pair,
            )
        )
    q_cosine = cosine_similarity(
        probes[0].projected_cotangent.tensor,
        probes[1].projected_cotangent.tensor,
        label="two-seed projected cotangents",
    )
    return LiveTwoSeedProbe(
        ordered_query_seeds=seeds,
        template_audit=template_audit,
        specificity_audits=tuple(
            specificity_audits_by_seed[seed] for seed in seeds
        ),  # type: ignore[arg-type]
        seed_probes=tuple(probes),  # type: ignore[arg-type]
        projected_cotangent_cosine=q_cosine,
        minimum_projected_cotangent_cosine=q_threshold,
        two_seed_direction_consistent=q_cosine >= q_threshold,
    )


_ROOT_FIELDS = (
    "schema_version",
    "probe_id",
    "contract",
    "cells",
)
_CONTRACT_FIELDS = (
    "method",
    "external_inference_inputs",
    "forbidden_auxiliary_inputs",
    "owner_mode",
    "owner_generation_count_per_cell",
    "owner_to_editor_allowed_channel",
    "owner_to_editor_forbidden_channels",
    "hook_coordinate",
    "prompt_relative_pair",
    "spatial_signature",
    "owner_editor_spatial_geometry_may_differ",
    "content_selected_mask_matching_or_ot",
    "native_schedule_index",
    "native_timestep",
    "sigma",
    "query_seed_count_per_cell",
    "frame_count",
    "latent_phases",
    "num_inference_steps",
    "world_topology",
    "relative_l2_dose",
    "minimum_owner_template_cosine",
    "minimum_same_topology_specificity_margin",
    "minimum_projected_cotangent_cosine",
    "minimum_projection_survival_cosine",
    "symmetric_arm_order",
    "same_topology_reverse_query_required",
    "plus_must_not_improve_reverse_rubric",
    "seed_selection",
    "seed_averaging",
    "optimizer",
    "parameter_update",
    "failure_policy",
    "scientific_or_action_editing_claim_authorized",
)
_CELL_FIELDS = (
    "cell_id",
    "actor_kind",
    "source_iid",
    "source_video",
    "source_video_sha256",
    "latent_shape",
    "action_caption",
    "action_caption_utf8_sha256",
    "noop_caption",
    "noop_caption_utf8_sha256",
    "action_family_id",
    "reverse_wrong_family_id",
    "reverse_wrong_family_caption",
    "reverse_wrong_family_caption_utf8_sha256",
    "owner_generation_seed",
    "query_seeds",
    "selected_before_generation",
)


@dataclass(frozen=True)
class ProbeCellSpec:
    cell_id: str
    actor_kind: str
    source_iid: str
    source_video: str
    source_video_sha256: str
    latent_shape: tuple[int, int, int, int, int]
    action_caption: str
    action_caption_utf8_sha256: str
    noop_caption: str
    noop_caption_utf8_sha256: str
    action_family_id: str
    reverse_wrong_family_id: str
    reverse_wrong_family_caption: str
    reverse_wrong_family_caption_utf8_sha256: str
    owner_generation_seed: int
    query_seeds: tuple[int, int]


@dataclass(frozen=True)
class ProbeRegistry:
    probe_id: str
    contract: Mapping[str, Any]
    cells: tuple[ProbeCellSpec, ProbeCellSpec]
    content_digest: str

    def cell(self, cell_id: str) -> ProbeCellSpec:
        matches = [row for row in self.cells if row.cell_id == cell_id]
        if len(matches) != 1:
            raise SelfImaginedCotangentContractError("registry cell lookup differs")
        return matches[0]


def validate_probe_registry(value: Any) -> ProbeRegistry:
    root = _exact_keys(value, _ROOT_FIELDS, label="probe registry")
    if root["schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise SelfImaginedCotangentContractError("probe registry schema differs")
    probe_id = _safe_id(root["probe_id"], label="probe ID")
    contract = _exact_keys(root["contract"], _CONTRACT_FIELDS, label="probe contract")
    expected_contract = {
        "method": "per-query-frozen-bernini-self-imagined-motion-cotangent",
        "external_inference_inputs": list(EXTERNAL_INFERENCE_INPUTS),
        "forbidden_auxiliary_inputs": list(FORBIDDEN_AUXILIARY_INPUTS),
        "owner_mode": "frozen_bernini_pure_t2v",
        "owner_generation_count_per_cell": 1,
        "owner_to_editor_allowed_channel": ALLOWED_OWNER_TO_EDITOR_CHANNEL,
        "owner_to_editor_forbidden_channels": list(
            FORBIDDEN_OWNER_TO_EDITOR_CHANNELS
        ),
        "hook_coordinate": HOOK_COORDINATE,
        "prompt_relative_pair": "action_minus_scene_matched_noop_same_x_sigma",
        "spatial_signature": [
            "signed_channel_mean",
            "centered_channel_rms_second_moment",
        ],
        "owner_editor_spatial_geometry_may_differ": True,
        "content_selected_mask_matching_or_ot": False,
        "native_schedule_index": SCHEDULE_INDEX,
        "native_timestep": NATIVE_TIMESTEP,
        "sigma": NATIVE_SIGMA,
        "query_seed_count_per_cell": QUERY_SEED_COUNT,
        "frame_count": 81,
        "latent_phases": LATENT_PHASES,
        "num_inference_steps": 40,
        "world_topology": "two_concurrent_world4_sp4_groups_on_one_8gpu_node",
        "symmetric_arm_order": ["plus", "minus"],
        "same_topology_reverse_query_required": True,
        "plus_must_not_improve_reverse_rubric": True,
        "seed_selection": False,
        "seed_averaging": False,
        "optimizer": False,
        "parameter_update": False,
        "failure_policy": "null_no_intervention_no_decode_no_update",
        "scientific_or_action_editing_claim_authorized": False,
    }
    for name, expected in expected_contract.items():
        observed = contract.get(name)
        if name == "sigma":
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or float(observed).hex() != float(expected).hex()
            ):
                raise SelfImaginedCotangentContractError(
                    "probe contract native sigma differs"
                )
        elif observed != expected:
            raise SelfImaginedCotangentContractError(
                f"probe contract {name} differs"
            )
    dose = _finite_positive(contract["relative_l2_dose"], label="relative L2 dose")
    if dose >= 1.0:
        raise SelfImaginedCotangentContractError("relative L2 dose must be below one")
    specificity_margin = _finite_positive(
        contract["minimum_same_topology_specificity_margin"],
        label="minimum same-topology specificity margin",
    )
    if specificity_margin >= 2.0:
        raise SelfImaginedCotangentContractError(
            "minimum same-topology specificity margin must be below two"
        )
    for name in (
        "minimum_owner_template_cosine",
        "minimum_projected_cotangent_cosine",
        "minimum_projection_survival_cosine",
    ):
        _finite_cosine_threshold(contract[name], label=name)

    raw_cells = root["cells"]
    if not isinstance(raw_cells, list) or len(raw_cells) != 2:
        raise SelfImaginedCotangentContractError("registry must contain exact core2")
    cells = []
    all_seeds: list[int] = []
    for index, expected_cell_id in enumerate(EXPECTED_CELL_IDS):
        row = _exact_keys(raw_cells[index], _CELL_FIELDS, label=f"cell {index}")
        cell_id = _safe_id(row["cell_id"], label="cell ID")
        if cell_id != expected_cell_id or row["actor_kind"] != expected_cell_id:
            raise SelfImaginedCotangentContractError("core2 cell order/kind differs")
        source_iid = _safe_id(row["source_iid"], label=f"{cell_id} source IID")
        if not isinstance(row["source_video"], str) or not row["source_video"].startswith("/"):
            raise SelfImaginedCotangentContractError(
                f"{cell_id} source video must be an absolute AUH path"
            )
        source_sha = _sha256(
            row["source_video_sha256"], label=f"{cell_id} source video SHA-256"
        )
        latent_shape_raw = row["latent_shape"]
        if (
            not isinstance(latent_shape_raw, list)
            or len(latent_shape_raw) != 5
            or any(type(item) is not int or item <= 0 for item in latent_shape_raw)
            or tuple(latent_shape_raw[:3]) != (1, 16, LATENT_PHASES)
            or latent_shape_raw[3] % 2
            or latent_shape_raw[4] % 2
        ):
            raise SelfImaginedCotangentContractError(
                f"{cell_id} latent shape differs"
            )
        for prompt_name in ("action_caption", "noop_caption"):
            prompt = row[prompt_name]
            prompt_sha = row[f"{prompt_name}_utf8_sha256"]
            if (
                not isinstance(prompt, str)
                or not prompt
                or prompt != prompt.strip()
                or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                != _sha256(prompt_sha, label=f"{cell_id} {prompt_name} SHA-256")
            ):
                raise SelfImaginedCotangentContractError(
                    f"{cell_id} {prompt_name} bytes differ"
                )
        if row["action_caption"] == row["noop_caption"]:
            raise SelfImaginedCotangentContractError(
                f"{cell_id} action/no-op prompts alias"
            )
        for family_name in ("action_family_id", "reverse_wrong_family_id"):
            _safe_id(row[family_name], label=f"{cell_id} {family_name}")
        reverse_prompt = row["reverse_wrong_family_caption"]
        reverse_sha = row["reverse_wrong_family_caption_utf8_sha256"]
        if (
            not isinstance(reverse_prompt, str)
            or not reverse_prompt
            or reverse_prompt != reverse_prompt.strip()
            or hashlib.sha256(reverse_prompt.encode("utf-8")).hexdigest()
            != _sha256(reverse_sha, label=f"{cell_id} reverse prompt SHA-256")
            or reverse_prompt in (row["action_caption"], row["noop_caption"])
            or row["action_family_id"] == row["reverse_wrong_family_id"]
        ):
            raise SelfImaginedCotangentContractError(
                f"{cell_id} same-topology reverse/wrong-family prompt differs"
            )
        owner_seed = row["owner_generation_seed"]
        query_seeds_raw = row["query_seeds"]
        if (
            type(owner_seed) is not int
            or owner_seed < 0
            or not isinstance(query_seeds_raw, list)
            or len(query_seeds_raw) != QUERY_SEED_COUNT
            or any(type(seed) is not int or seed < 0 for seed in query_seeds_raw)
            or len(set(query_seeds_raw)) != QUERY_SEED_COUNT
            or owner_seed in query_seeds_raw
            or row["selected_before_generation"] is not True
        ):
            raise SelfImaginedCotangentContractError(
                f"{cell_id} prospective seed contract differs"
            )
        all_seeds.extend((owner_seed, *query_seeds_raw))
        cells.append(
            ProbeCellSpec(
                cell_id=cell_id,
                actor_kind=row["actor_kind"],
                source_iid=source_iid,
                source_video=row["source_video"],
                source_video_sha256=source_sha,
                latent_shape=tuple(latent_shape_raw),  # type: ignore[arg-type]
                action_caption=row["action_caption"],
                action_caption_utf8_sha256=row["action_caption_utf8_sha256"],
                noop_caption=row["noop_caption"],
                noop_caption_utf8_sha256=row["noop_caption_utf8_sha256"],
                action_family_id=row["action_family_id"],
                reverse_wrong_family_id=row["reverse_wrong_family_id"],
                reverse_wrong_family_caption=reverse_prompt,
                reverse_wrong_family_caption_utf8_sha256=reverse_sha,
                owner_generation_seed=owner_seed,
                query_seeds=tuple(query_seeds_raw),  # type: ignore[arg-type]
            )
        )
    if len(set(all_seeds)) != len(all_seeds):
        raise SelfImaginedCotangentContractError(
            "owner/query seeds must be unique across the sealed core2 probe"
        )
    return ProbeRegistry(
        probe_id=probe_id,
        contract=MappingProxyType(dict(contract)),
        cells=tuple(cells),  # type: ignore[arg-type]
        content_digest=object_sha256(root),
    )


def load_probe_registry(
    path: str | Path, *, expected_file_sha256: Optional[str] = None
) -> ProbeRegistry:
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise SelfImaginedCotangentContractError(
            "probe registry must be an absolute plain file"
        )
    raw = source.read_bytes()
    if expected_file_sha256 is not None and hashlib.sha256(raw).hexdigest() != _sha256(
        expected_file_sha256, label="expected registry file SHA-256"
    ):
        raise SelfImaginedCotangentContractError("probe registry file SHA-256 differs")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelfImaginedCotangentContractError(
                    f"probe registry contains duplicate key {key}"
                )
            result[key] = value
        return result

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SelfImaginedCotangentContractError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelfImaginedCotangentContractError(
            "probe registry is not strict UTF-8 JSON"
        ) from error
    return validate_probe_registry(decoded)


def build_auh_dual4_execution_plan(registry: ProbeRegistry) -> dict[str, Any]:
    if not isinstance(registry, ProbeRegistry):
        raise SelfImaginedCotangentContractError("probe registry type differs")
    groups = []
    for index, cell in enumerate(registry.cells):
        groups.append(
            {
                "cell_id": cell.cell_id,
                "sp4_ranks": list(range(index * SP_SIZE, (index + 1) * SP_SIZE)),
                "owner_generation_seed": cell.owner_generation_seed,
                "ordered_query_seeds": list(cell.query_seeds),
                "query_execution": "sequential_no_selection",
                "same_x_sigma_prompt_order_per_query_seed": [
                    "action",
                    "reverse_wrong_family",
                    "common_scene_null",
                ],
                "symmetric_exact81_arms": [
                    f"q{seed}:{sign}"
                    for seed in cell.query_seeds
                    for sign in ("plus", "minus")
                ],
            }
        )
    unsigned = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "probe_id": registry.probe_id,
        "registry_content_digest": registry.content_digest,
        "world_size": 8,
        "topology": "DP2_cells_x_SP4_no_cross_cell_averaging",
        "groups": groups,
        "owner_generation_count": 2,
        "owner_hidden_prompt_forward_count": 12,
        "candidate_hidden_prompt_forward_count": 12,
        "symmetric_exact81_decode_count": 8,
        "seed_selection": False,
        "seed_averaging": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "authority": "mechanism_and_direction_probe_only",
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


_OUTCOME_FIELDS = (
    "query_seed",
    "frame_count_plus",
    "frame_count_minus",
    "plus_action_better_than_minus",
    "plus_reverse_wrong_family_not_improved_vs_minus_or_base",
    "plus_source_identity_noninferior_to_base",
    "plus_camera_noninferior_to_base",
    "plus_background_noninferior_to_base",
    "plus_quality_noninferior_to_base",
    "plus_temporal_consistency_noninferior_to_base",
    "audited_without_seed_or_arm_selection",
)


def evaluate_two_seed_direction_gate(
    outcomes: Sequence[Mapping[str, Any]], *, ordered_query_seeds: Sequence[int]
) -> dict[str, Any]:
    seeds = tuple(ordered_query_seeds)
    if (
        len(seeds) != QUERY_SEED_COUNT
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != QUERY_SEED_COUNT
        or not isinstance(outcomes, Sequence)
        or len(outcomes) != QUERY_SEED_COUNT
    ):
        raise SelfImaginedCotangentContractError(
            "direction gate requires exactly both ordered query seeds"
        )
    checked = []
    boolean_fields = _OUTCOME_FIELDS[3:]
    for index, seed in enumerate(seeds):
        row = _exact_keys(
            outcomes[index], _OUTCOME_FIELDS, label=f"query seed {seed} outcome"
        )
        if (
            row["query_seed"] != seed
            or row["frame_count_plus"] != 81
            or row["frame_count_minus"] != 81
            or any(type(row[name]) is not bool for name in boolean_fields)
        ):
            raise SelfImaginedCotangentContractError(
                f"query seed {seed} exact81 outcome closure differs"
            )
        passed = all(row[name] is True for name in boolean_fields)
        checked.append({**row, "passed": passed})
    passed = all(row["passed"] for row in checked)
    unsigned = {
        "schema_version": GATE_SCHEMA_VERSION,
        "ordered_query_seeds": list(seeds),
        "per_seed": checked,
        "required_pass_count": QUERY_SEED_COUNT,
        "observed_pass_count": sum(int(row["passed"]) for row in checked),
        "passed": passed,
        "seed_selection": False,
        "seed_averaging": False,
        "parameter_update_authorized": False,
        "action_editing_success_claim_authorized": False,
        "next_authority_if_passed": (
            "identity_camera_jacobian_integration_probe_only"
            if passed
            else "null_stop"
        ),
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


__all__ = [
    "ALLOWED_OWNER_TO_EDITOR_CHANNEL",
    "EXPECTED_CELL_IDS",
    "EXTERNAL_INFERENCE_INPUTS",
    "FORBIDDEN_AUXILIARY_INPUTS",
    "FORBIDDEN_OWNER_TO_EDITOR_CHANNELS",
    "FrozenOwnerTemplate",
    "GATE_SCHEMA_VERSION",
    "HOOK_COORDINATE",
    "HIDDEN_SIZE",
    "LATENT_PHASES",
    "LiveSeedProbe",
    "LiveTwoSeedProbe",
    "MotionQuotientConfig",
    "NATIVE_SIGMA",
    "NATIVE_TIMESTEP",
    "PAIR_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "PerQueryScoreOutput",
    "PromptSpecificityAudit",
    "ProbeCellSpec",
    "ProbeRegistry",
    "ProjectedCotangent",
    "QUERY_SEED_COUNT",
    "REGISTRY_SCHEMA_VERSION",
    "RESIDUAL_SHAPE",
    "SCHEDULE_INDEX",
    "SCHEMA_VERSION",
    "SPECIFICITY_SCHEMA_VERSION",
    "SelfImaginedCotangentContractError",
    "SymmetricInterventionPair",
    "TEMPLATE_SCHEMA_VERSION",
    "TwoSeedTemplateAudit",
    "audit_two_seed_templates",
    "audit_prompt_specificity",
    "build_auh_dual4_execution_plan",
    "build_frozen_owner_template",
    "canonical_json_bytes",
    "cosine_similarity",
    "evaluate_two_seed_direction_gate",
    "load_probe_registry",
    "make_frozen_per_query_scorer",
    "make_symmetric_intervention_pair",
    "object_sha256",
    "project_mask_free_nuisance_cotangent",
    "run_two_query_seed_live_probe",
    "temporal_motion_quotient",
    "tensor_value_digest",
    "validate_probe_registry",
]
