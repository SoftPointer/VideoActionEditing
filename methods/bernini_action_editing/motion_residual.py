#!/usr/bin/env python3
"""Pure contracts for Bernini counterfactual delta-field LoRA training.

The representation in this module is deliberately renderer-native.  It is a
dense, signed velocity residual on Wan's packed latent grid, not a pose, mask,
track, or target-derived inference condition.  Target video latents are used
only to build training losses.

The key quotient discards the temporally constant component of a source-to-
target field and compares its non-DC temporal modes at several lags.  A static
appearance replacement is therefore not directly rewarded, while transitions,
articulation, and sustained motions with an onset remain observable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence


ROUTING_SCHEMA = "bernini-cdf-routing-v1"
ROUTING_TIERS = frozenset(("full_pair", "motion_only", "reject"))
MOTION_OBJECTIVES = frozenset(
    (
        "raw_delta",
        "quotient_multilag",
        "causal_boundary_multilag",
        "causal_boundary_charbonnier",
        "causal_ema_charbonnier",
    )
)
BRANCH_STATE_MODES = frozenset(
    (
        "separate_clean_paths",
        "shared_noisy_clean_field",
        "source_target_bridge_clean_field",
    )
)
DEFAULT_NOOP_INSTRUCTION = (
    "Keep the source video exactly unchanged, including every subject, "
    "appearance, action, camera motion, background, timing, and composition."
)
MODULE_SCOPES = frozenset(
    (
        "all_qkvo",
        "q_out",
        "self_q_out",
        "cross_q",
        "cross_q_out",
        "mid_q_out",
    )
)
_MODULE_RE = re.compile(
    r"^(?P<prefix>.+\.blocks\.(?P<block>\d+)\.attn(?P<attention>[12]))\."
    r"(?P<projection>to_q|to_k|to_v|to_out\.0)$"
)


class MotionContractError(RuntimeError):
    """Raised when a motion-training invariant is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MotionContractError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_lora_scope(
    available: Sequence[str],
    scope: str,
    *,
    middle_blocks: tuple[int, int] = (7, 22),
) -> list[str]:
    """Select exact fully-qualified Wan attention projection names.

    ``q_out`` is the primary low-destructiveness arm: K/V memories remain
    frozen while both target/source self-attention and instruction
    cross-attention can change how those memories are queried and written.
    ``cross_q_out`` is the more conservative instruction-only ablation.
    """

    if scope not in MODULE_SCOPES:
        raise MotionContractError(f"unknown LoRA scope: {scope!r}")
    start, end = middle_blocks
    if type(start) is not int or type(end) is not int or not 0 <= start <= end < 30:
        raise MotionContractError("middle block range must lie in [0, 29]")
    selected: list[str] = []
    for name in available:
        match = _MODULE_RE.fullmatch(name)
        if match is None:
            raise MotionContractError(f"unexpected attention module name: {name}")
        block = int(match.group("block"))
        attention = int(match.group("attention"))
        projection = match.group("projection")
        is_q_out = projection in ("to_q", "to_out.0")
        keep = {
            "all_qkvo": True,
            "q_out": is_q_out,
            "self_q_out": attention == 1 and is_q_out,
            "cross_q": attention == 2 and projection == "to_q",
            "cross_q_out": attention == 2 and is_q_out,
            "mid_q_out": start <= block <= end and is_q_out,
        }[scope]
        if keep:
            selected.append(name)
    selected = sorted(selected)
    expected = {
        "all_qkvo": 240,
        "q_out": 120,
        "self_q_out": 60,
        "cross_q": 30,
        "cross_q_out": 60,
        "mid_q_out": (end - start + 1) * 4,
    }[scope]
    if len(selected) != expected:
        raise MotionContractError(
            f"scope {scope} selected {len(selected)} modules, expected {expected}"
        )
    return selected


@dataclass(frozen=True)
class Route:
    iid: str
    tier: str
    full_target_weight: float
    review: Optional[str] = None


class ReviewRouter:
    """Hash-bound per-IID supervision routing.

    Unreviewed pairs default to ``motion_only``.  This is intentional: the
    current 644-row preview release has no post-video identity acceptance, so
    it must not silently receive framewise full-target supervision.
    """

    def __init__(
        self,
        routes: Mapping[str, Route],
        *,
        source_path: Optional[Path],
        default_tier: str = "motion_only",
    ):
        if default_tier not in ("motion_only", "reject"):
            raise MotionContractError("default tier must be motion_only or reject")
        self._routes = dict(routes)
        self.source_path = source_path
        self.default_tier = default_tier
        serial = {
            iid: {
                "tier": route.tier,
                "full_target_weight": route.full_target_weight,
                "review": route.review,
            }
            for iid, route in sorted(self._routes.items())
        }
        self.digest = object_sha256(
            {"schema": ROUTING_SCHEMA, "default_tier": default_tier, "routes": serial}
        )
        self.file_sha256 = file_sha256(source_path) if source_path is not None else None

    @classmethod
    def load(
        cls,
        value: Optional[str | Path],
        *,
        default_tier: str = "motion_only",
    ) -> "ReviewRouter":
        if value is None:
            return cls({}, source_path=None, default_tier=default_tier)
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise MotionContractError("routing JSONL must be an absolute path")
        try:
            path = path.resolve(strict=True)
        except OSError as error:
            raise MotionContractError(f"routing JSONL is unavailable: {error}") from error
        if not path.is_file() or path.is_symlink():
            raise MotionContractError("routing JSONL must be a plain file")
        routes: dict[str, Route] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise MotionContractError(
                            f"blank routing row at line {line_number}"
                        )
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise MotionContractError(
                            f"routing row {line_number} is not an object"
                        )
                    if value.get("schema_version") != ROUTING_SCHEMA:
                        raise MotionContractError(
                            f"routing row {line_number} schema differs"
                        )
                    iid = value.get("iid")
                    tier = value.get("tier")
                    if not isinstance(iid, str) or not iid or "\x00" in iid:
                        raise MotionContractError(
                            f"routing row {line_number} has invalid IID"
                        )
                    if iid in routes:
                        raise MotionContractError(f"duplicate routing IID: {iid}")
                    if tier not in ROUTING_TIERS:
                        raise MotionContractError(
                            f"routing row {line_number} has invalid tier: {tier!r}"
                        )
                    weight = value.get(
                        "full_target_weight", 1.0 if tier == "full_pair" else 0.0
                    )
                    if (
                        not isinstance(weight, (int, float))
                        or isinstance(weight, bool)
                        or not math.isfinite(float(weight))
                        or not 0.0 <= float(weight) <= 1.0
                    ):
                        raise MotionContractError(
                            f"routing row {line_number} has invalid full_target_weight"
                        )
                    if tier != "full_pair" and float(weight) != 0.0:
                        raise MotionContractError(
                            "only full_pair routes may enable full-target loss"
                        )
                    review = value.get("review")
                    if review is not None and not isinstance(review, str):
                        raise MotionContractError(
                            f"routing row {line_number} review must be text"
                        )
                    routes[iid] = Route(iid, tier, float(weight), review)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MotionContractError(f"cannot read routing JSONL: {error}") from error
        return cls(routes, source_path=path, default_tier=default_tier)

    def route(self, iid: str) -> Route:
        if not isinstance(iid, str) or not iid:
            raise MotionContractError("dataset row is missing a non-empty IID")
        existing = self._routes.get(iid)
        if existing is not None:
            return existing
        return Route(iid, self.default_tier, 0.0, "unreviewed_default")

    def receipt(self) -> dict[str, Any]:
        counts = {tier: 0 for tier in sorted(ROUTING_TIERS)}
        for route in self._routes.values():
            counts[route.tier] += 1
        return {
            "schema_version": ROUTING_SCHEMA,
            "path": str(self.source_path) if self.source_path is not None else None,
            "file_sha256": self.file_sha256,
            "routing_digest": self.digest,
            "default_tier": self.default_tier,
            "explicit_route_counts": counts,
        }


def replace_edit_instruction(sample: Mapping[str, Any], instruction: str) -> dict[str, Any]:
    if not isinstance(instruction, str) or not instruction.strip() or "\x00" in instruction:
        raise MotionContractError("replacement instruction must be non-empty text")
    try:
        messages = json.loads(str(sample["inputs"]))
    except (KeyError, json.JSONDecodeError) as error:
        raise MotionContractError(f"cannot decode renderer messages: {error}") from error
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or not isinstance(messages[1], dict)
        or messages[1].get("type") != "text"
    ):
        raise MotionContractError("renderer messages do not contain one edit instruction")
    messages[1] = dict(messages[1])
    messages[1]["text"] = instruction.strip()
    result = dict(sample)
    result["inputs"] = canonical_json_bytes(messages).decode("utf-8")
    return result


def high_noise_weight(
    sigma: Any,
    *,
    floor: float = 0.25,
    power: float = 2.0,
) -> Any:
    """Differentiable scalar weight favoring DynaEdit's early/high-noise regime."""

    if not 0.0 <= floor <= 1.0 or not math.isfinite(floor):
        raise MotionContractError("high-noise floor must lie in [0, 1]")
    if power <= 0.0 or not math.isfinite(power):
        raise MotionContractError("high-noise power must be finite and positive")
    return floor + (1.0 - floor) * sigma.float().clamp(0.0, 1.0).pow(power)


def clean_field_inverse_sigma_weight(
    sigma: Any,
    *,
    weight_floor: float = 0.25,
) -> Any:
    """Condition clean-field gradients without a low-sigma explosion.

    Recovering ``x=y-sigma*v`` makes a clean-space squared loss contribute a
    factor of ``sigma`` to the velocity gradient.  Multiplying the loss by
    ``1/sigma`` removes that first-order attenuation.  The audited training
    denominator is clamped independently of the training range, so the weight
    remains in ``[1, 4]`` by default while training still covers the final
    positive sigma of the 40-step inference schedule.
    """

    import torch

    if (
        not math.isfinite(float(weight_floor))
        or not 0.0 < float(weight_floor) <= 1.0
    ):
        raise MotionContractError("weight_floor must lie in (0, 1]")
    value = sigma.float()
    if not bool(torch.isfinite(value).all()):
        raise MotionContractError("sigma must be finite")
    if bool((value <= 0.0).any()) or bool((value > 1.0).any()):
        raise MotionContractError("sigma must lie in (0, 1] for clean-field training")
    return value.clamp_min(float(weight_floor)).reciprocal()


def _as_temporal_grid(packed: Any, *, latent_frames: int) -> Any:
    """View ``[B,N,D]`` packed Wan tokens as ``[B,T,HW,D]``."""

    if getattr(packed, "ndim", None) != 3:
        raise MotionContractError("packed prediction must have shape [B,N,D]")
    if type(latent_frames) is not int or latent_frames <= 1:
        raise MotionContractError("latent_frames must be an integer greater than one")
    tokens = int(packed.shape[1])
    if tokens <= 0 or tokens % latent_frames:
        raise MotionContractError(
            f"packed token count {tokens} is not divisible by {latent_frames} frames"
        )
    return packed.reshape(int(packed.shape[0]), latent_frames, tokens // latent_frames, int(packed.shape[2]))


def temporal_quotient(field: Any, *, latent_frames: int = 21) -> Any:
    """Remove only the per-spatial-location temporal DC component."""

    grid = _as_temporal_grid(field, latent_frames=latent_frames)
    return grid - grid.mean(dim=1, keepdim=True)


def causal_boundary_quotient(field: Any, *, latent_frames: int = 21) -> Any:
    """Represent a field relative to its first latent phase.

    Unlike a zero-mean temporal quotient, this causal gauge preserves a
    sustained terminal action: a step from zero to ``c`` remains exactly zero
    before onset and ``c`` afterwards.  It removes time-constant appearance
    replacement without leaking a negative pre-action ghost.
    """

    grid = _as_temporal_grid(field, latent_frames=latent_frames)
    return grid - grid[:, :1]


def causal_ema_boundary_projection(
    field: Any,
    *,
    latent_frames: int = 21,
    decay: float = 0.5,
) -> Any:
    """Return a causal, low-pass counterfactual field with an exact boundary.

    A symmetric temporal kernel would leak a future action into earlier latent
    phases.  This one-sided exponential filter cannot create that pre-action
    ghost: phase ``t`` depends only on phases ``<= t``.  Applying the exact
    phase-zero projection after filtering removes a time-constant appearance
    offset, while a persistent step action approaches its full terminal value.

    The same operator is used as the v4 training representation and as the
    inference execution operator.  It consumes only the model's internal
    action-minus-no-op field and introduces no mask, flow, pose, or tracker.
    """

    import torch

    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not math.isfinite(float(decay))
        or not 0.0 <= float(decay) < 1.0
    ):
        raise MotionContractError("causal EMA decay must lie in [0, 1)")
    grid = _as_temporal_grid(field.float(), latent_frames=latent_frames)
    phases = [grid[:, 0]]
    decay_value = float(decay)
    update_value = 1.0 - decay_value
    for phase_index in range(1, latent_frames):
        phases.append(
            decay_value * phases[-1] + update_value * grid[:, phase_index]
        )
    filtered = torch.stack(phases, dim=1)
    projected = filtered - filtered[:, :1]
    if not bool(torch.isfinite(projected).all()):
        raise MotionContractError("causal EMA projection is non-finite")
    if not bool(
        torch.equal(projected[:, :1], torch.zeros_like(projected[:, :1]))
    ):
        raise MotionContractError(
            "causal EMA projection did not zero the first phase exactly"
        )
    return projected


def charbonnier_distance(
    predicted: Any,
    target: Any,
    *,
    scale: float = 0.1,
) -> Any:
    """Robust mean distance that does not square synthetic target outliers."""

    import torch

    if tuple(predicted.shape) != tuple(target.shape):
        raise MotionContractError("Charbonnier operands have different shapes")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise MotionContractError("Charbonnier scale must be finite and positive")
    residual = predicted.float() - target.float()
    value = torch.sqrt(residual.square() + float(scale) ** 2) - float(scale)
    if not bool(torch.isfinite(value).all()):
        raise MotionContractError("Charbonnier distance is non-finite")
    return value.mean()


def causal_ema_motion_loss(
    predicted_field: Any,
    target_field: Any,
    *,
    latent_frames: int = 21,
    decay: float = 0.5,
    charbonnier_scale: float = 0.1,
) -> tuple[Any, dict[str, Any]]:
    """Compare the exact v4 executable field with a robust penalty."""

    predicted = causal_ema_boundary_projection(
        predicted_field, latent_frames=latent_frames, decay=decay
    )
    target = causal_ema_boundary_projection(
        target_field, latent_frames=latent_frames, decay=decay
    )
    loss = charbonnier_distance(
        predicted, target, scale=charbonnier_scale
    )
    return loss, {
        "predicted_causal_ema": predicted,
        "target_causal_ema": target,
    }


def causal_boundary_charbonnier_loss(
    predicted_field: Any,
    target_field: Any,
    *,
    latent_frames: int = 21,
    charbonnier_scale: float = 0.1,
) -> tuple[Any, dict[str, Any]]:
    """Robustly compare the idempotent executable ``Q0`` fields.

    Unlike a temporal low-pass, ``Q0(d)=d-d(0)`` does not delay action onset or
    turn a one-phase change into a tail.  It is idempotent, which lets the
    trainer project a noisy synthetic target onto the exact inference-reachable
    clean manifold before constructing bridge queries.
    """

    predicted = causal_boundary_quotient(
        predicted_field.float(), latent_frames=latent_frames
    )
    target = causal_boundary_quotient(
        target_field.float(), latent_frames=latent_frames
    )
    loss = charbonnier_distance(
        predicted, target, scale=charbonnier_scale
    )
    return loss, {
        "predicted_causal_boundary": predicted,
        "target_causal_boundary": target,
    }


def multiscale_temporal_difference_loss(
    predicted_field: Any,
    target_field: Any,
    *,
    latent_frames: int = 21,
    lags: Sequence[int] = (1, 2, 4),
) -> Any:
    """Compare dense signed temporal changes at several latent-frame scales."""

    import torch

    predicted = _as_temporal_grid(predicted_field.float(), latent_frames=latent_frames)
    target = _as_temporal_grid(target_field.float(), latent_frames=latent_frames)
    if tuple(predicted.shape) != tuple(target.shape):
        raise MotionContractError("predicted and target fields have different shapes")
    if not lags:
        raise MotionContractError("at least one temporal lag is required")
    losses = []
    seen: set[int] = set()
    for lag in lags:
        if type(lag) is not int or lag <= 0 or lag >= latent_frames:
            raise MotionContractError(f"invalid temporal lag: {lag!r}")
        if lag in seen:
            raise MotionContractError(f"duplicate temporal lag: {lag}")
        seen.add(lag)
        predicted_delta = predicted[:, lag:] - predicted[:, :-lag]
        target_delta = target[:, lag:] - target[:, :-lag]
        losses.append(torch.mean((predicted_delta - target_delta) ** 2))
    return torch.stack(losses).mean()


def counterfactual_motion_loss(
    predicted_velocity: Any,
    source_velocity: Any,
    target_velocity: Any,
    *,
    latent_frames: int = 21,
    lags: Sequence[int] = (1, 2, 4),
    quotient_weight: float = 0.5,
) -> tuple[Any, dict[str, Any]]:
    """Loss on source-relative, appearance-quotiented velocity fields.

    ``source_velocity`` uses the *same diffusion noise* as the action target.
    Consequently the ground-truth residual is exactly ``source_clean -
    target_clean``; no external tracker or representation model is involved.
    """

    import torch

    if not 0.0 <= quotient_weight <= 1.0 or not math.isfinite(quotient_weight):
        raise MotionContractError("quotient_weight must lie in [0, 1]")
    for left, right, label in (
        (predicted_velocity, source_velocity, "predicted/source"),
        (target_velocity, source_velocity, "target/source"),
    ):
        if tuple(left.shape) != tuple(right.shape):
            raise MotionContractError(f"{label} velocity shapes differ")
    predicted_field = predicted_velocity.float() - source_velocity.float()
    target_field = target_velocity.float() - source_velocity.float()
    predicted_q = temporal_quotient(predicted_field, latent_frames=latent_frames)
    target_q = temporal_quotient(target_field, latent_frames=latent_frames)
    quotient = torch.mean((predicted_q - target_q) ** 2)
    differences = multiscale_temporal_difference_loss(
        predicted_field,
        target_field,
        latent_frames=latent_frames,
        lags=lags,
    )
    total = quotient_weight * quotient + (1.0 - quotient_weight) * differences
    return total, {
        "temporal_quotient": quotient,
        "multiscale_difference": differences,
    }


def differential_motion_loss(
    action_prediction: Any,
    noop_prediction: Any,
    action_target_velocity: Any,
    source_target_velocity: Any,
    *,
    latent_frames: int = 21,
    lags: Sequence[int] = (1, 2, 4),
    quotient_weight: float = 0.5,
    objective: str = "causal_boundary_multilag",
    causal_ema_decay: float = 0.5,
    charbonnier_scale: float = 0.1,
) -> tuple[Any, dict[str, Any]]:
    """Match the exact action-minus-no-op field used by inference.

    The two predictions must be evaluated with the same clean source, sigma,
    and diffusion noise.  The target field remains analytic:
    ``(epsilon-G) - (epsilon-S) == S-G``.  Unlike
    :func:`counterfactual_motion_loss`, this function also learns/calibrates
    errors in Bernini's no-op branch instead of assuming it is the exact source
    velocity.
    """

    import torch

    if objective not in MOTION_OBJECTIVES:
        raise MotionContractError(f"unknown differential motion objective: {objective!r}")
    if not 0.0 <= quotient_weight <= 1.0 or not math.isfinite(quotient_weight):
        raise MotionContractError("quotient_weight must lie in [0, 1]")
    shapes = {
        tuple(value.shape)
        for value in (
            action_prediction,
            noop_prediction,
            action_target_velocity,
            source_target_velocity,
        )
    }
    if len(shapes) != 1:
        raise MotionContractError("differential prediction/target shapes differ")
    predicted_field = action_prediction.float() - noop_prediction.float()
    target_field = action_target_velocity.float() - source_target_velocity.float()
    raw_delta = torch.mean((predicted_field - target_field) ** 2)
    predicted_q = temporal_quotient(predicted_field, latent_frames=latent_frames)
    target_q = temporal_quotient(target_field, latent_frames=latent_frames)
    quotient = torch.mean((predicted_q - target_q) ** 2)
    predicted_causal = causal_boundary_quotient(
        predicted_field, latent_frames=latent_frames
    )
    target_causal = causal_boundary_quotient(
        target_field, latent_frames=latent_frames
    )
    causal_boundary = torch.mean((predicted_causal - target_causal) ** 2)
    causal_boundary_charbonnier, _ = causal_boundary_charbonnier_loss(
        predicted_field,
        target_field,
        latent_frames=latent_frames,
        charbonnier_scale=charbonnier_scale,
    )
    causal_ema, _ = causal_ema_motion_loss(
        predicted_field,
        target_field,
        latent_frames=latent_frames,
        decay=causal_ema_decay,
        charbonnier_scale=charbonnier_scale,
    )
    differences = multiscale_temporal_difference_loss(
        predicted_field,
        target_field,
        latent_frames=latent_frames,
        lags=lags,
    )
    if objective == "raw_delta":
        total = raw_delta
    elif objective == "causal_boundary_multilag":
        total = (
            quotient_weight * causal_boundary
            + (1.0 - quotient_weight) * differences
        )
    elif objective == "causal_boundary_charbonnier":
        total = causal_boundary_charbonnier
    elif objective == "causal_ema_charbonnier":
        total = causal_ema
    else:
        total = quotient_weight * quotient + (1.0 - quotient_weight) * differences
    return total, {
        "raw_delta": raw_delta,
        "temporal_quotient": quotient,
        "causal_boundary": causal_boundary,
        "causal_boundary_charbonnier": causal_boundary_charbonnier,
        "causal_ema_charbonnier": causal_ema,
        "multiscale_difference": differences,
    }


def same_state_clean_predictions(
    action_velocity: Any,
    noop_velocity: Any,
    shared_noisy: Any,
    sigma: Any,
) -> tuple[Any, Any]:
    """Recover action/no-op clean fields from one identical noisy query."""

    shapes = {
        tuple(value.shape)
        for value in (action_velocity, noop_velocity, shared_noisy)
    }
    if len(shapes) != 1 or getattr(action_velocity, "ndim", None) != 3:
        raise MotionContractError(
            "same-state action/no-op/noisy tensors must share [B,N,D]"
        )
    if int(action_velocity.shape[0]) != 1:
        raise MotionContractError(
            "pinned Bernini same-state clean reconstruction requires batch size one"
        )
    try:
        from tri_branch_unipc import (
            TriBranchHookError,
            pinned_raw_condition_clean,
        )

        return (
            pinned_raw_condition_clean(shared_noisy, action_velocity, sigma),
            pinned_raw_condition_clean(shared_noisy, noop_velocity, sigma),
        )
    except TriBranchHookError as error:
        raise MotionContractError(str(error)) from error


def differential_clean_motion_loss(
    action_velocity: Any,
    noop_velocity: Any,
    shared_noisy: Any,
    sigma: Any,
    target_clean: Any,
    source_clean: Any,
    *,
    latent_frames: int = 21,
    lags: Sequence[int] = (1, 2, 4),
    quotient_weight: float = 0.5,
    objective: str = "causal_boundary_multilag",
    causal_ema_decay: float = 0.5,
    charbonnier_scale: float = 0.1,
) -> tuple[Any, dict[str, Any]]:
    """Train the exact inference clean delta on a shared noisy state.

    Since ``x = y - sigma*v``, the predicted field is
    ``x_action-x_noop = -sigma*(v_action-v_noop)`` and is matched to
    ``target_clean-source_clean``.  This removes the previous branch-state gap
    and avoids a velocity target that diverges as sigma approaches zero.
    """

    import torch

    if objective not in MOTION_OBJECTIVES:
        raise MotionContractError(f"unknown differential motion objective: {objective!r}")
    if not 0.0 <= quotient_weight <= 1.0 or not math.isfinite(quotient_weight):
        raise MotionContractError("quotient_weight must lie in [0, 1]")
    action_clean, noop_clean = same_state_clean_predictions(
        action_velocity, noop_velocity, shared_noisy, sigma
    )
    if tuple(target_clean.shape) != tuple(action_clean.shape):
        raise MotionContractError("target clean/action prediction shapes differ")
    if tuple(source_clean.shape) != tuple(noop_clean.shape):
        raise MotionContractError("source clean/no-op prediction shapes differ")
    predicted_field = action_clean - noop_clean
    target_field = target_clean.float() - source_clean.float()
    raw_delta = torch.mean((predicted_field - target_field) ** 2)
    predicted_q = temporal_quotient(predicted_field, latent_frames=latent_frames)
    target_q = temporal_quotient(target_field, latent_frames=latent_frames)
    quotient = torch.mean((predicted_q - target_q) ** 2)
    predicted_causal = causal_boundary_quotient(
        predicted_field, latent_frames=latent_frames
    )
    target_causal = causal_boundary_quotient(
        target_field, latent_frames=latent_frames
    )
    causal_boundary = torch.mean((predicted_causal - target_causal) ** 2)
    causal_boundary_charbonnier, causal_boundary_parts = (
        causal_boundary_charbonnier_loss(
            predicted_field,
            target_field,
            latent_frames=latent_frames,
            charbonnier_scale=charbonnier_scale,
        )
    )
    causal_ema, causal_ema_parts = causal_ema_motion_loss(
        predicted_field,
        target_field,
        latent_frames=latent_frames,
        decay=causal_ema_decay,
        charbonnier_scale=charbonnier_scale,
    )
    differences = multiscale_temporal_difference_loss(
        predicted_field,
        target_field,
        latent_frames=latent_frames,
        lags=lags,
    )
    if objective == "raw_delta":
        total = raw_delta
    elif objective == "causal_boundary_multilag":
        total = (
            quotient_weight * causal_boundary
            + (1.0 - quotient_weight) * differences
        )
    elif objective == "causal_boundary_charbonnier":
        total = causal_boundary_charbonnier
    elif objective == "causal_ema_charbonnier":
        total = causal_ema
    else:
        total = quotient_weight * quotient + (1.0 - quotient_weight) * differences
    return total, {
        "raw_delta": raw_delta,
        "temporal_quotient": quotient,
        "causal_boundary": causal_boundary,
        "causal_boundary_charbonnier": causal_boundary_charbonnier,
        "predicted_causal_boundary": causal_boundary_parts[
            "predicted_causal_boundary"
        ],
        "target_causal_boundary": causal_boundary_parts[
            "target_causal_boundary"
        ],
        "causal_ema_charbonnier": causal_ema,
        "predicted_causal_ema": causal_ema_parts["predicted_causal_ema"],
        "target_causal_ema": causal_ema_parts["target_causal_ema"],
        "multiscale_difference": differences,
        "action_clean": action_clean,
        "noop_clean": noop_clean,
        "predicted_clean_delta": predicted_field,
        "target_clean_delta": target_field,
    }


def flatten_velocity_patches(velocity: Any) -> Any:
    """Match official GEN_Wanx22's ``(pt ph pw c)`` flatten order."""

    if getattr(velocity, "ndim", None) != 6:
        raise MotionContractError("velocity patches must have shape [B,N,C,pt,ph,pw]")
    return velocity.permute(0, 1, 3, 4, 5, 2).reshape(
        int(velocity.shape[0]), int(velocity.shape[1]), -1
    )


def unflatten_velocity_patches(packed: Any, *, reference: Any) -> Any:
    """Invert :func:`flatten_velocity_patches` using audited patch geometry."""

    if getattr(packed, "ndim", None) != 3:
        raise MotionContractError("packed velocity must have shape [B,N,D]")
    if getattr(reference, "ndim", None) != 6:
        raise MotionContractError(
            "reference velocity patches must have shape [B,N,C,pt,ph,pw]"
        )
    batch, tokens, channels, pt, ph, pw = (
        int(value) for value in reference.shape
    )
    expected = (batch, tokens, channels * pt * ph * pw)
    if tuple(int(value) for value in packed.shape) != expected:
        raise MotionContractError(
            f"packed velocity shape {tuple(packed.shape)} differs from {expected}"
        )
    return (
        packed.reshape(batch, tokens, pt, ph, pw, channels)
        .permute(0, 1, 5, 2, 3, 4)
        .contiguous()
    )


def project_executable_target_mode(
    source_mode: Any,
    target_mode: Any,
    *,
    latent_frames: int = 21,
) -> Any:
    """Project a synthetic target onto the exact Q0 inference fixed point.

    The dense executor can reach ``source + Q0(raw_target-source)``, not the
    raw synthetic target when that pair contains a time-constant appearance
    gauge.  Since Q0 is idempotent, using this executable target for both the
    action clean target and the source/target bridge closes the training loop:

    ``x_dagger = source + Q0(target-source)`` and
    ``source + Q0(x_dagger-source) = x_dagger``.

    This is an offline supervision transformation.  It adds no inference mask,
    tracker, flow, pose, target video, or first-frame input.
    """

    import torch

    if (
        getattr(source_mode, "ndim", None) != 5
        or tuple(source_mode.shape) != tuple(target_mode.shape)
    ):
        raise MotionContractError(
            "source/target modes must share [N,C,pt,ph,pw] geometry"
        )
    source_packed = flatten_velocity_patches(source_mode.unsqueeze(0)).float()
    target_packed = flatten_velocity_patches(target_mode.unsqueeze(0)).float()
    projected_grid = causal_boundary_quotient(
        target_packed - source_packed,
        latent_frames=latent_frames,
    )
    projected = projected_grid.reshape_as(source_packed)
    executable_packed = source_packed + projected
    executable = unflatten_velocity_patches(
        executable_packed, reference=source_mode.unsqueeze(0)
    ).squeeze(0).to(dtype=source_mode.dtype)

    executed_field = causal_boundary_quotient(
        flatten_velocity_patches(executable.unsqueeze(0)).float()
        - source_packed,
        latent_frames=latent_frames,
    ).reshape_as(source_packed)
    if not bool(torch.allclose(executed_field, projected, atol=2e-6, rtol=2e-6)):
        raise MotionContractError("executable target projection is not idempotent")
    projected_grid = projected.reshape(
        int(projected.shape[0]),
        latent_frames,
        int(projected.shape[1]) // latent_frames,
        int(projected.shape[2]),
    )
    if not bool(
        torch.equal(projected_grid[:, :1], torch.zeros_like(projected_grid[:, :1]))
    ):
        raise MotionContractError(
            "executable target projection did not preserve phase zero exactly"
        )
    return executable


def unpack_clean_mode(blob: Any, vae_mean: Any, vae_std: Any, *, max_frames: int = 21) -> Any:
    """Decode one posterior distribution by mode and pack it like Bernini."""

    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from einops import rearrange

    # Import lazily so pure contract tests do not require torch/diffusers.
    try:
        from train_lora import _load_tensor_blob
    except ImportError:  # package-style import
        from .train_lora import _load_tensor_blob  # type: ignore

    distribution = DiagonalGaussianDistribution(_load_tensor_blob(blob))
    clean = distribution.mode().squeeze(0)
    clean = (clean - vae_mean) / vae_std
    clean = clean[:, :max_frames]
    return rearrange(
        clean,
        "c (t pt) (h ph) (w pw) -> (t h w) c pt ph pw",
        pt=1,
        ph=2,
        pw=2,
    )


def rebuild_paired_batches_from_modes(
    action_batch: Mapping[str, Any],
    copy_batch: Mapping[str, Any],
    *,
    source_mode: Any,
    target_mode: Any,
    sigma: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build action/copy cells with one source, sigma, and diffusion noise.

    The original transformed target supplies only ``epsilon``.  Both posterior
    samples are replaced by posterior modes.  The copy target then reuses the
    exact same source mode as its clean condition, eliminating the otherwise
    hidden two-posterior-sample no-op bug.
    """

    import torch

    action = dict(action_batch)
    copy = dict(copy_batch)
    selector = action["vae_latents_mask"].squeeze(0).bool()
    source_count = int((~selector).sum().item())
    target_count = int(selector.sum().item())
    if source_count <= 0 or source_count != target_count:
        raise MotionContractError("paired source/target token spans differ")
    if tuple(source_mode.shape) != tuple(target_mode.shape):
        raise MotionContractError("source/target posterior modes have different shapes")
    if int(source_mode.shape[0]) != source_count:
        raise MotionContractError("posterior mode token count differs from renderer batch")
    target_noisy_old = action["input_vae_latents"][selector]
    target_velocity_old = action["target_velocity"]
    sigma_value = sigma.float().reshape(-1)
    if sigma_value.numel() != 1:
        raise MotionContractError("paired one-sample batch requires one sigma")
    shape = [1] * target_noisy_old.ndim
    shape[0] = 1
    sigma_broadcast = sigma_value.reshape(shape)
    # x_sigma = x_0 + sigma * v; epsilon = x_sigma + (1-sigma) * v.
    epsilon = target_noisy_old.float() + (1.0 - sigma_broadcast) * target_velocity_old.float()
    source_mode = source_mode.to(dtype=target_noisy_old.dtype)
    target_mode = target_mode.to(dtype=target_noisy_old.dtype)
    action_velocity = epsilon - target_mode.float()
    copy_velocity = epsilon - source_mode.float()
    action_noisy = target_mode.float() + sigma_broadcast * action_velocity
    copy_noisy = source_mode.float() + sigma_broadcast * copy_velocity

    def _replace(batch: dict[str, Any], noisy: Any, velocity: Any) -> None:
        latent = batch["input_vae_latents"].clone()
        latent[~selector] = source_mode.to(dtype=latent.dtype)
        latent[selector] = noisy.to(dtype=latent.dtype)
        batch["input_vae_latents"] = latent
        batch["target_velocity"] = velocity.to(dtype=batch["target_velocity"].dtype)
        batch["timesteps"] = action["timesteps"].clone()
        # Geometry/source IDs are content-independent.  Reusing the action rope
        # makes the two cells differ only in text and clean target content.
        batch["input_vae_rope"] = action["input_vae_rope"].clone()
        batch["vae_latents_mask"] = action["vae_latents_mask"].clone()
        batch["vae_seqlen"] = action["vae_seqlen"].clone()
        batch["target_lens"] = action["target_lens"].clone()

    _replace(action, action_noisy, action_velocity)
    _replace(copy, copy_noisy, copy_velocity)
    copy_clean_recovered = copy_noisy.float() - sigma_broadcast * copy_velocity
    if not bool(torch.equal(copy_clean_recovered, source_mode.float())):
        # Floating arithmetic can be non-bitwise for general sigma; exact
        # equality is not expected, but a strict numerical bound is.
        if not bool(torch.allclose(copy_clean_recovered, source_mode.float(), atol=2e-6, rtol=2e-6)):
            raise MotionContractError("copy-cell clean target does not reconstruct source")
    return action, copy, {
        "source_velocity": copy_velocity,
        "target_velocity": action_velocity,
        "epsilon": epsilon,
        "sigma": sigma_value,
    }


def rebuild_same_state_batches_from_modes(
    action_batch: Mapping[str, Any],
    noop_batch: Mapping[str, Any],
    *,
    source_mode: Any,
    target_mode: Any,
    sigma: Any,
    minimum_sigma: float = 0.1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build action/no-op cells on one identical noisy target state.

    Inference compares action and semantic-noop predictions at the same noisy
    query ``y``.  The older paired builder instead placed the two branches on
    separate target/source diffusion paths.  This builder removes that gap:

    ``y = target + sigma * (epsilon - target)``
    ``v_action* = (y - target) / sigma``
    ``v_noop*   = (y - source) / sigma``

    The stable training quantity is the clean difference
    ``-sigma * (v_action - v_noop) == target - source``.  Target video remains
    supervision only; it is not an inference condition.
    """

    import torch

    if not math.isfinite(float(minimum_sigma)) or minimum_sigma <= 0.0:
        raise MotionContractError("minimum_sigma must be finite and positive")
    action = dict(action_batch)
    noop = dict(noop_batch)
    selector = action["vae_latents_mask"].squeeze(0).bool()
    if not torch.equal(selector, noop["vae_latents_mask"].squeeze(0).bool()):
        raise MotionContractError("action/no-op target selectors differ")
    source_count = int((~selector).sum().item())
    target_count = int(selector.sum().item())
    if source_count <= 0 or source_count != target_count:
        raise MotionContractError("same-state source/target token spans differ")
    if tuple(source_mode.shape) != tuple(target_mode.shape):
        raise MotionContractError("source/target posterior modes have different shapes")
    if int(source_mode.shape[0]) != source_count:
        raise MotionContractError("posterior mode token count differs from renderer batch")

    sigma_value = sigma.float().reshape(-1)
    if sigma_value.numel() != 1 or not bool(torch.isfinite(sigma_value).all()):
        raise MotionContractError("same-state one-sample batch requires one finite sigma")
    if float(sigma_value.item()) < float(minimum_sigma):
        raise MotionContractError(
            f"same-state clean-field training requires sigma >= {minimum_sigma}"
        )
    target_noisy_old = action["input_vae_latents"][selector]
    target_velocity_old = action["target_velocity"]
    sigma_shape = [1] * target_noisy_old.ndim
    sigma_broadcast = sigma_value.reshape(sigma_shape)
    epsilon = (
        target_noisy_old.float()
        + (1.0 - sigma_broadcast) * target_velocity_old.float()
    )
    source_mode = source_mode.to(dtype=target_noisy_old.dtype)
    target_mode = target_mode.to(dtype=target_noisy_old.dtype)
    action_velocity = epsilon - target_mode.float()
    shared_noisy = target_mode.float() + sigma_broadcast * action_velocity

    def _replace(batch: dict[str, Any], target_velocity: Any) -> None:
        latent = action["input_vae_latents"].clone()
        latent[~selector] = source_mode.to(dtype=latent.dtype)
        latent[selector] = shared_noisy.to(dtype=latent.dtype)
        batch["input_vae_latents"] = latent
        batch["target_velocity"] = target_velocity.to(
            dtype=batch["target_velocity"].dtype
        )
        batch["timesteps"] = action["timesteps"].clone()
        batch["input_vae_rope"] = action["input_vae_rope"].clone()
        batch["vae_latents_mask"] = action["vae_latents_mask"].clone()
        batch["vae_seqlen"] = action["vae_seqlen"].clone()
        batch["target_lens"] = action["target_lens"].clone()

    # Quantize the query exactly once to the renderer input dtype, then derive
    # both analytic targets from that same value.  This prevents an implicit
    # FP32-vs-BF16 state mismatch in clean reconstruction.
    shared_noisy = shared_noisy.to(dtype=action["input_vae_latents"].dtype)
    action_velocity = (
        shared_noisy.float() - target_mode.float()
    ) / sigma_broadcast
    noop_velocity = (
        shared_noisy.float() - source_mode.float()
    ) / sigma_broadcast
    _replace(action, action_velocity)
    _replace(noop, noop_velocity)
    if not torch.equal(action["input_vae_latents"], noop["input_vae_latents"]):
        raise MotionContractError("action/no-op noisy query tensors are not identical")
    exact_shared_noisy = action["input_vae_latents"][selector].float()
    action_clean = exact_shared_noisy - sigma_broadcast * action_velocity
    noop_clean = exact_shared_noisy - sigma_broadcast * noop_velocity
    if not bool(torch.allclose(action_clean, target_mode.float(), atol=2e-6, rtol=2e-6)):
        raise MotionContractError("same-state action target does not recover target clean")
    if not bool(torch.allclose(noop_clean, source_mode.float(), atol=2e-6, rtol=2e-6)):
        raise MotionContractError("same-state no-op target does not recover source clean")

    return action, noop, {
        "source_clean": flatten_velocity_patches(source_mode.unsqueeze(0)).float(),
        "target_clean": flatten_velocity_patches(target_mode.unsqueeze(0)).float(),
        "shared_noisy": flatten_velocity_patches(
            exact_shared_noisy.unsqueeze(0)
        ).float(),
        "action_target_velocity": flatten_velocity_patches(
            action_velocity.unsqueeze(0)
        ).float(),
        "noop_target_velocity": flatten_velocity_patches(
            noop_velocity.unsqueeze(0)
        ).float(),
        "epsilon": epsilon,
        "sigma": sigma_value.reshape(()),
        "branch_state_mode": "shared_noisy_clean_field",
        "same_state_formula": {
            "noisy": "y=(1-sigma)*target+sigma*epsilon",
            "clean": "x=y-sigma*velocity",
            "predicted_delta": "-sigma*(v_action-v_noop)",
            "target_delta": "target-source",
        },
    }


def rebuild_bridge_state_batches_from_modes(
    action_batch: Mapping[str, Any],
    noop_batch: Mapping[str, Any],
    *,
    source_mode: Any,
    target_mode: Any,
    epsilon: Any,
    sigma: Any,
    timestep: Any,
    bridge_fraction: float,
    minimum_sigma: float = 0.1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build one exact action/no-op pair on a source-target bridge query.

    The v3 trainer evaluated only the target diffusion path, although C2FR
    inference starts on the source and then closes the loop on its own edited
    state.  v4 uses the same epsilon, sigma, timestep, source tokens, and text
    geometry at both bridge endpoints:

    ``b_beta=(1-beta)*source+beta*target``
    ``y_beta=(1-sigma)*b_beta+sigma*epsilon``.

    Action and no-op are still compared on one bit-identical ``y_beta``.  The
    clean counterfactual supervision remains ``target-source`` for every beta;
    target video is never added as a model condition at inference.
    """

    import torch

    if (
        isinstance(bridge_fraction, bool)
        or not isinstance(bridge_fraction, (int, float))
        or not math.isfinite(float(bridge_fraction))
        or not 0.0 <= float(bridge_fraction) <= 1.0
    ):
        raise MotionContractError("bridge_fraction must lie in [0, 1]")
    if not math.isfinite(float(minimum_sigma)) or minimum_sigma <= 0.0:
        raise MotionContractError("minimum_sigma must be finite and positive")
    action = dict(action_batch)
    noop = dict(noop_batch)
    selector = action["vae_latents_mask"].squeeze(0).bool()
    if not torch.equal(selector, noop["vae_latents_mask"].squeeze(0).bool()):
        raise MotionContractError("bridge action/no-op target selectors differ")
    source_count = int((~selector).sum().item())
    target_count = int(selector.sum().item())
    if source_count <= 0 or source_count != target_count:
        raise MotionContractError("bridge source/target token spans differ")
    shapes = {
        tuple(value.shape) for value in (source_mode, target_mode, epsilon)
    }
    if len(shapes) != 1 or int(source_mode.shape[0]) != source_count:
        raise MotionContractError("bridge clean/noise tensor shapes differ")
    sigma_value = sigma.detach().float().reshape(-1)
    if (
        sigma_value.numel() != 1
        or sigma_value.device.type != "cpu"
        or not bool(torch.isfinite(sigma_value).all())
        or float(sigma_value.item()) < float(minimum_sigma)
        or float(sigma_value.item()) > 1.0
    ):
        raise MotionContractError(
            "bridge sigma must be one finite CPU fp32 value in the training range"
        )
    timestep_value = timestep.detach().reshape(-1)
    if (
        timestep_value.numel() != 1
        or timestep_value.device.type != "cpu"
        or timestep_value.dtype != torch.int64
    ):
        raise MotionContractError("bridge timestep must be one CPU int64 value")

    renderer_dtype = action["input_vae_latents"].dtype
    source_mode = source_mode.to(dtype=renderer_dtype)
    target_mode = target_mode.to(dtype=renderer_dtype)
    epsilon = epsilon.to(dtype=torch.float32)
    beta = float(bridge_fraction)
    bridge_clean = (
        (1.0 - beta) * source_mode.float() + beta * target_mode.float()
    )
    sigma_shape = [1] * bridge_clean.ndim
    sigma_shape[0] = 1
    sigma_broadcast = sigma_value.reshape(sigma_shape)
    shared_noisy = (
        (1.0 - sigma_broadcast) * bridge_clean
        + sigma_broadcast * epsilon
    ).to(dtype=renderer_dtype)
    action_velocity = (
        shared_noisy.float() - target_mode.float()
    ) / sigma_broadcast
    noop_velocity = (
        shared_noisy.float() - source_mode.float()
    ) / sigma_broadcast

    def _replace(batch: dict[str, Any], target_velocity: Any) -> None:
        latent = action["input_vae_latents"].clone()
        latent[~selector] = source_mode.to(dtype=latent.dtype)
        latent[selector] = shared_noisy.to(dtype=latent.dtype)
        batch["input_vae_latents"] = latent
        batch["target_velocity"] = target_velocity.to(
            dtype=batch["target_velocity"].dtype
        )
        # The upstream training scheduler emits BF16 timesteps. Reusing that
        # dtype would round official UniPC values (notably 999 -> 1000) before
        # the renderer sees them. Preserve the inference grid exactly as a CPU
        # int64 tensor; the normal batch transfer moves it to the model device
        # later without changing its value.
        batch["timesteps"] = torch.full(
            tuple(action["timesteps"].shape),
            int(timestep_value.item()),
            dtype=torch.int64,
            device="cpu",
        )
        batch["input_vae_rope"] = action["input_vae_rope"].clone()
        batch["vae_latents_mask"] = action["vae_latents_mask"].clone()
        batch["vae_seqlen"] = action["vae_seqlen"].clone()
        batch["target_lens"] = action["target_lens"].clone()

    _replace(action, action_velocity)
    _replace(noop, noop_velocity)
    if not torch.equal(action["input_vae_latents"], noop["input_vae_latents"]):
        raise MotionContractError(
            "bridge action/no-op noisy query tensors are not identical"
        )
    exact_shared_noisy = action["input_vae_latents"][selector].float()
    action_clean = exact_shared_noisy - sigma_broadcast * action_velocity
    noop_clean = exact_shared_noisy - sigma_broadcast * noop_velocity
    if not bool(
        torch.allclose(action_clean, target_mode.float(), atol=2e-6, rtol=2e-6)
    ):
        raise MotionContractError("bridge action target does not recover target clean")
    if not bool(
        torch.allclose(noop_clean, source_mode.float(), atol=2e-6, rtol=2e-6)
    ):
        raise MotionContractError("bridge no-op target does not recover source clean")
    return action, noop, {
        "source_clean": flatten_velocity_patches(source_mode.unsqueeze(0)).float(),
        "target_clean": flatten_velocity_patches(target_mode.unsqueeze(0)).float(),
        "shared_noisy": flatten_velocity_patches(
            exact_shared_noisy.unsqueeze(0)
        ).float(),
        "action_target_velocity": flatten_velocity_patches(
            action_velocity.unsqueeze(0)
        ).float(),
        "noop_target_velocity": flatten_velocity_patches(
            noop_velocity.unsqueeze(0)
        ).float(),
        "epsilon": epsilon,
        "sigma": sigma_value.reshape(()),
        "timestep": timestep_value.reshape(()),
        "bridge_fraction": beta,
        "branch_state_mode": "source_target_bridge_clean_field",
        "same_state_formula": {
            "bridge_clean": "b=(1-beta)*source+beta*target",
            "noisy": "y=(1-sigma)*b+sigma*epsilon",
            "clean": "x=y-sigma*velocity",
            "predicted_delta": "-sigma*(v_action-v_noop)",
            "target_delta": "target-source",
        },
    }


def renderer_velocity_prediction(renderer: Any, batch: Mapping[str, Any]) -> Any:
    """Return raw packed target velocity from the pinned official renderer.

    This is a faithful extraction of ``GEN_Wanx22.forward`` up to, but not
    including, its elementwise MSE.  Keeping it in our wrapper preserves the
    byte-pinned upstream source tree.
    """

    if hasattr(batch.get("target_velocity"), "tensor"):
        raise MotionContractError("DTensor target wrappers are not supported here")
    text_lens, text_embs = renderer.get_t5_text_embeddings(
        batch["input_ids"], batch["attention_mask"], batch["t5_input_lens"]
    )
    valid_samples = len(text_lens)
    vae_seqlen = batch["vae_seqlen"].squeeze(0)
    vae_seqlen = vae_seqlen[vae_seqlen > 0].unsqueeze(0)
    timesteps = batch["timesteps"].squeeze(0)[:valid_samples].unsqueeze(0)
    decoder = renderer.diff_dec
    if decoder.transformer is not None and decoder.transformer_2 is not None:
        raise MotionContractError("delta trainer requires exactly one Wan expert")
    if decoder.transformer_2 is None:
        model_id, transformer = "transformer_1", decoder.transformer
    else:
        model_id, transformer = "transformer_2", decoder.transformer_2
    if transformer is None:
        raise MotionContractError("active Wan transformer is unavailable")
    inputs = batch["input_vae_latents"].unsqueeze(0)
    inputs = transformer.patch_embedding(inputs.squeeze(0)).flatten(1).unsqueeze(0)
    rope = batch["input_vae_rope"].permute(1, 0, 2).unsqueeze(0)
    target_indices = batch["vae_latents_mask"].squeeze(0).nonzero().squeeze(-1)
    prediction = decoder.shared_step(
        model_id=model_id,
        noisy_latents=inputs,
        timesteps=timesteps.squeeze(0),
        cond_embeds=text_embs,
        rotary_embs=rope,
        batch_vae_seqlen=vae_seqlen.squeeze(0).tolist(),
        batch_text_seqlen=text_lens,
    )
    return prediction[:, target_indices, :]
