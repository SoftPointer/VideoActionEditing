"""Pure-tensor core for dual-conditional Bernini reward diagnostics.

The quantities in this module are *denoising-error / energy-ratio proxies*.
They are useful for contrasting two conditions on the same candidate, noise,
and rectified-flow state, but they are **not exact likelihoods** and must not be
reported as such.  In particular, this module never calls a model and never
assumes that a velocity-MSE ratio is a calibrated video probability ratio.

The intended comparison is dual:

* the action proxy prefers a T2V action condition over a no-op condition; and
* the preservation proxy prefers the correct source condition over a wrong
  source condition.

Both proxies use the sign convention ``positive == preferred condition has
lower denoising error``.  Candidate acceptance remains a constrained decision:
action and preservation must both pass their hard gates.  The two proxies are
therefore deliberately not collapsed into an unconstrained weighted sum.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


SCHEMA_VERSION = "bernini-dual-conditional-denoising-ratio-proxy-v1"
DEFAULT_ENERGY_EPSILON = 1.0e-8


class DualConditionalRatioError(ValueError):
    """A tensor, mask, weight, or candidate-selection contract is invalid."""


@dataclass(frozen=True)
class RectifiedFlowState:
    """A noised rectified-flow state and its exact training velocity target."""

    x_sigma: torch.Tensor
    true_velocity: torch.Tensor


@dataclass(frozen=True)
class LexicographicSelection:
    """Fail-closed result of hard-gated candidate selection.

    ``index`` is ``-1`` for every batch item without a jointly feasible
    candidate.  For unbatched candidate scores it is a scalar tensor; for
    ``[B, K]`` scores it is ``[B]``.  The pass masks retain the full candidate
    shape.
    """

    index: torch.Tensor
    has_feasible: torch.Tensor
    action_pass: torch.Tensor
    preservation_pass: torch.Tensor
    joint_pass: torch.Tensor


@dataclass(frozen=True)
class MultiSigmaRatioDiagnostics:
    """Aggregated proxy plus auditable per-sigma error diagnostics.

    The primary ``proxy`` is computed from the two *aggregated errors*.  It is
    intentionally not a weighted mean of ``per_sigma_proxy``.
    """

    proxy: torch.Tensor
    preferred_error: torch.Tensor
    contrast_error: torch.Tensor
    preferred_error_by_sigma: torch.Tensor
    contrast_error_by_sigma: torch.Tensor
    per_sigma_proxy: torch.Tensor
    normalized_sigma_weights: torch.Tensor


@dataclass(frozen=True)
class MultiAxisLexicographicSelection:
    """Fail-closed multi-axis action/preservation gate diagnostics."""

    index: torch.Tensor
    has_feasible: torch.Tensor
    action_axis_pass: torch.Tensor
    preservation_axis_pass: torch.Tensor
    action_pass: torch.Tensor
    preservation_pass: torch.Tensor
    joint_pass: torch.Tensor
    worst_action_calibrated_margin: torch.Tensor
    worst_preservation_calibrated_margin: torch.Tensor


def _require_floating_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DualConditionalRatioError(f"{name} must be a torch.Tensor")
    if value.device.type == "meta":
        raise DualConditionalRatioError(f"{name} cannot be a meta tensor")
    if not value.is_floating_point():
        raise DualConditionalRatioError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all().item()):
        raise DualConditionalRatioError(f"{name} contains NaN or infinity")
    return value


def _require_same_tensor_contract(
    name: str,
    value: Any,
    reference_name: str,
    reference: torch.Tensor,
    *,
    require_same_dtype: bool = True,
) -> torch.Tensor:
    result = _require_floating_tensor(name, value)
    if result.shape != reference.shape:
        raise DualConditionalRatioError(
            f"{name} shape differs from {reference_name}"
        )
    if result.device != reference.device:
        raise DualConditionalRatioError(
            f"{name} device differs from {reference_name}"
        )
    if require_same_dtype and result.dtype != reference.dtype:
        raise DualConditionalRatioError(
            f"{name} dtype differs from {reference_name}"
        )
    return result


def _common_accumulation_dtype(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.dtype:
    result = torch.promote_types(left.dtype, right.dtype)
    if result in (torch.float16, torch.bfloat16):
        return torch.float32
    return result


def _as_scalar_or_batch_parameter(
    name: str,
    value: Any,
    *,
    batch_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        result = result.expand(batch_size)
    elif result.ndim == 1 and int(result.shape[0]) == batch_size:
        pass
    else:
        raise DualConditionalRatioError(
            f"{name} must be a scalar or exact [B] tensor"
        )
    if not bool(torch.isfinite(result).all().item()):
        raise DualConditionalRatioError(f"{name} contains NaN or infinity")
    return result


def rectified_flow_state(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: float | torch.Tensor,
) -> RectifiedFlowState:
    """Return ``x_sigma=(1-sigma)*clean+sigma*noise`` and ``noise-clean``.

    ``clean`` and ``noise`` use a batch-first layout ``[B, ...]``.  ``sigma``
    is either one scalar or ``[B]`` and follows the convention ``0 == clean``
    and ``1 == noise``.  No stochastic sampling occurs in this function.
    """

    clean_tensor = _require_floating_tensor("clean", clean)
    if clean_tensor.ndim < 2 or int(clean_tensor.shape[0]) < 1:
        raise DualConditionalRatioError("clean must have batch-first [B, ...] layout")
    noise_tensor = _require_same_tensor_contract(
        "noise", noise, "clean", clean_tensor
    )
    sigma_tensor = _as_scalar_or_batch_parameter(
        "sigma",
        sigma,
        batch_size=int(clean_tensor.shape[0]),
        reference=clean_tensor,
    )
    if bool(((sigma_tensor < 0.0) | (sigma_tensor > 1.0)).any().item()):
        raise DualConditionalRatioError("sigma must remain in [0, 1]")
    broadcast_sigma = sigma_tensor.reshape(
        int(clean_tensor.shape[0]), *([1] * (clean_tensor.ndim - 1))
    )
    x_sigma = clean_tensor + broadcast_sigma * (noise_tensor - clean_tensor)
    return RectifiedFlowState(
        x_sigma=x_sigma,
        true_velocity=noise_tensor - clean_tensor,
    )


def masked_per_sample_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute one mask-normalized MSE energy per batch item.

    The mask may be boolean or floating point and must broadcast to the full
    prediction shape.  Its denominator is computed *after* broadcasting, so a
    ``[B,1,T,H,W]`` mask correctly counts every latent channel.  Every sample
    must retain positive mask mass.  Half/bfloat inputs accumulate in FP32.
    """

    predicted = _require_floating_tensor("prediction", prediction)
    if predicted.ndim < 2 or int(predicted.shape[0]) < 1:
        raise DualConditionalRatioError(
            "prediction must have batch-first [B, ...] layout"
        )
    expected = _require_same_tensor_contract(
        "target",
        target,
        "prediction",
        predicted,
        require_same_dtype=False,
    )
    accumulation_dtype = _common_accumulation_dtype(predicted, expected)
    squared_error = (
        predicted.to(accumulation_dtype) - expected.to(accumulation_dtype)
    ).square()
    if mask is None:
        return squared_error.flatten(start_dim=1).mean(dim=1)

    if not isinstance(mask, torch.Tensor):
        raise DualConditionalRatioError("mask must be a torch.Tensor or None")
    if mask.device != predicted.device:
        raise DualConditionalRatioError("mask device differs from prediction")
    if mask.device.type == "meta":
        raise DualConditionalRatioError("mask cannot be a meta tensor")
    if mask.dtype == torch.bool:
        weights = mask.to(dtype=accumulation_dtype)
    elif mask.is_floating_point():
        weights = mask.to(dtype=accumulation_dtype)
    else:
        raise DualConditionalRatioError("mask must be boolean or floating point")
    if not bool(torch.isfinite(weights).all().item()):
        raise DualConditionalRatioError("mask contains NaN or infinity")
    if bool((weights < 0.0).any().item()):
        raise DualConditionalRatioError("mask weights must be nonnegative")
    try:
        expanded_weights = torch.broadcast_to(weights, squared_error.shape)
    except RuntimeError as error:
        raise DualConditionalRatioError(
            "mask is not broadcastable to prediction"
        ) from error
    flat_weights = expanded_weights.flatten(start_dim=1)
    denominator = flat_weights.sum(dim=1)
    if bool((denominator <= 0.0).any().item()):
        raise DualConditionalRatioError(
            "every sample must retain positive mask mass"
        )
    numerator = (squared_error * expanded_weights).flatten(start_dim=1).sum(dim=1)
    return numerator / denominator


def denoising_error_log_ratio_proxy(
    preferred_error: torch.Tensor,
    contrast_error: torch.Tensor,
    *,
    epsilon: float = DEFAULT_ENERGY_EPSILON,
) -> torch.Tensor:
    """Return ``log((contrast_error+eps)/(preferred_error+eps))``.

    Positive values mean that the preferred condition has lower denoising MSE.
    This is an energy/error-ratio proxy, not an exact log-likelihood ratio.
    """

    preferred = _require_floating_tensor("preferred_error", preferred_error)
    contrast = _require_same_tensor_contract(
        "contrast_error",
        contrast_error,
        "preferred_error",
        preferred,
        require_same_dtype=False,
    )
    if bool((preferred < 0.0).any().item()) or bool(
        (contrast < 0.0).any().item()
    ):
        raise DualConditionalRatioError("denoising errors must be nonnegative")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise DualConditionalRatioError("epsilon must be a positive finite scalar")
    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise DualConditionalRatioError("epsilon must be a positive finite scalar")
    accumulation_dtype = _common_accumulation_dtype(preferred, contrast)
    preferred_accumulated = preferred.to(dtype=accumulation_dtype)
    contrast_accumulated = contrast.to(dtype=accumulation_dtype)
    epsilon_tensor = preferred_accumulated.new_tensor(epsilon_value)
    return torch.log(contrast_accumulated + epsilon_tensor) - torch.log(
        preferred_accumulated + epsilon_tensor
    )


def action_t2v_cond_vs_noop_proxy(
    action_condition_velocity: torch.Tensor,
    noop_condition_velocity: torch.Tensor,
    true_velocity: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    epsilon: float = DEFAULT_ENERGY_EPSILON,
) -> torch.Tensor:
    """Contrast a T2V action condition against a T2V no-op condition.

    Returns one scalar proxy per sample.  All predictions must correspond to
    the same candidate ``x_sigma``, noise realization, sigma, and true velocity.
    """

    action_error = masked_per_sample_mse(
        action_condition_velocity, true_velocity, mask
    )
    noop_error = masked_per_sample_mse(
        noop_condition_velocity, true_velocity, mask
    )
    return denoising_error_log_ratio_proxy(
        action_error, noop_error, epsilon=epsilon
    )


def source_correct_vs_wrong_proxy(
    correct_source_velocity: torch.Tensor,
    wrong_source_velocity: torch.Tensor,
    true_velocity: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    epsilon: float = DEFAULT_ENERGY_EPSILON,
) -> torch.Tensor:
    """Contrast the correct source condition against a wrong-source control.

    Returns one scalar proxy per sample.  The two predictions must use the same
    candidate ``x_sigma``, noise realization, sigma, and true velocity.
    """

    correct_error = masked_per_sample_mse(
        correct_source_velocity, true_velocity, mask
    )
    wrong_error = masked_per_sample_mse(
        wrong_source_velocity, true_velocity, mask
    )
    return denoising_error_log_ratio_proxy(
        correct_error, wrong_error, epsilon=epsilon
    )


def aggregate_multi_sigma(
    proxy_by_sigma: torch.Tensor,
    sigma_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a weighted mean over the leading sigma axis.

    ``proxy_by_sigma`` has shape ``[S, ...]``.  ``sigma_weights`` must have exact
    shape ``[S]`` and is shared across every sample and candidate.  Per-sample
    or candidate-specific sigma weights are forbidden.  This generic helper is
    retained for diagnostics; :func:`multi_sigma_denoising_error_ratio_proxy`
    is the primary error-ratio API.  The returned shape is ``[...]``.
    """

    proxy = _require_floating_tensor("proxy_by_sigma", proxy_by_sigma)
    if proxy.ndim < 1 or int(proxy.shape[0]) < 1:
        raise DualConditionalRatioError(
            "proxy_by_sigma must have a nonempty leading sigma axis"
        )
    if sigma_weights is None:
        accumulation_dtype = (
            torch.float32
            if proxy.dtype in (torch.float16, torch.bfloat16)
            else proxy.dtype
        )
        return proxy.to(dtype=accumulation_dtype).mean(dim=0)
    weights = _require_floating_tensor("sigma_weights", sigma_weights)
    if weights.device != proxy.device:
        raise DualConditionalRatioError(
            "sigma_weights device differs from proxy_by_sigma"
        )
    if weights.ndim != 1 or tuple(weights.shape) != (int(proxy.shape[0]),):
        raise DualConditionalRatioError(
            "sigma_weights must have exact shared shape [S]"
        )
    if bool((weights < 0.0).any().item()):
        raise DualConditionalRatioError("sigma_weights must be nonnegative")
    accumulation_dtype = _common_accumulation_dtype(proxy, weights)
    accumulated_proxy = proxy.to(dtype=accumulation_dtype)
    accumulated_weights = weights.to(dtype=accumulation_dtype)
    denominator = accumulated_weights.sum()
    if bool((denominator <= 0.0).item()):
        raise DualConditionalRatioError(
            "shared sigma weights must have positive total mass"
        )
    broadcast_weights = accumulated_weights.reshape(
        int(accumulated_weights.shape[0]),
        *([1] * (accumulated_proxy.ndim - 1)),
    )
    return (accumulated_proxy * broadcast_weights).sum(dim=0) / denominator


def multi_sigma_denoising_error_ratio_proxy(
    preferred_error_by_sigma: torch.Tensor,
    contrast_error_by_sigma: torch.Tensor,
    sigma_weights: torch.Tensor,
    *,
    epsilon: float = DEFAULT_ENERGY_EPSILON,
) -> MultiSigmaRatioDiagnostics:
    """Aggregate both condition errors first, then form one log-ratio proxy.

    Both error tensors have shape ``[S, ...]`` and the shared weights have exact
    shape ``[S]``.  This ordering is material: averaging per-sigma log ratios
    would define a different objective and can overemphasize a low-energy sigma.
    The returned diagnostics retain both raw errors and per-sigma ratios for
    auditing, but only ``proxy`` is the primary multi-sigma score.
    """

    preferred = _require_floating_tensor(
        "preferred_error_by_sigma", preferred_error_by_sigma
    )
    if preferred.ndim < 1 or int(preferred.shape[0]) < 1:
        raise DualConditionalRatioError(
            "preferred_error_by_sigma must have nonempty [S, ...] layout"
        )
    contrast = _require_same_tensor_contract(
        "contrast_error_by_sigma",
        contrast_error_by_sigma,
        "preferred_error_by_sigma",
        preferred,
        require_same_dtype=False,
    )
    if bool((preferred < 0.0).any().item()) or bool(
        (contrast < 0.0).any().item()
    ):
        raise DualConditionalRatioError("denoising errors must be nonnegative")
    weights = _require_floating_tensor("sigma_weights", sigma_weights)
    if weights.device != preferred.device:
        raise DualConditionalRatioError(
            "sigma_weights device differs from condition errors"
        )
    if weights.ndim != 1 or tuple(weights.shape) != (int(preferred.shape[0]),):
        raise DualConditionalRatioError(
            "sigma_weights must have exact shared shape [S]"
        )
    if bool((weights < 0.0).any().item()):
        raise DualConditionalRatioError(
            "shared sigma weights must be nonnegative with positive total mass"
        )

    error_dtype = _common_accumulation_dtype(preferred, contrast)
    common_dtype = _common_accumulation_dtype(
        preferred.to(dtype=error_dtype), weights
    )
    preferred_common = preferred.to(dtype=common_dtype)
    contrast_common = contrast.to(dtype=common_dtype)
    weights_common = weights.to(dtype=common_dtype)
    if not bool((weights_common.sum() > 0.0).item()):
        raise DualConditionalRatioError(
            "shared sigma weights must be nonnegative with positive total mass"
        )

    preferred_aggregated = aggregate_multi_sigma(
        preferred_common, weights_common
    )
    contrast_aggregated = aggregate_multi_sigma(contrast_common, weights_common)
    proxy = denoising_error_log_ratio_proxy(
        preferred_aggregated, contrast_aggregated, epsilon=epsilon
    )
    per_sigma_proxy = denoising_error_log_ratio_proxy(
        preferred_common, contrast_common, epsilon=epsilon
    )
    normalized_weights = weights_common
    normalized_weights = normalized_weights / normalized_weights.sum()
    return MultiSigmaRatioDiagnostics(
        proxy=proxy,
        preferred_error=preferred_aggregated,
        contrast_error=contrast_aggregated,
        preferred_error_by_sigma=preferred_common,
        contrast_error_by_sigma=contrast_common,
        per_sigma_proxy=per_sigma_proxy,
        normalized_sigma_weights=normalized_weights,
    )


def _candidate_threshold(
    name: str,
    value: float | torch.Tensor,
    *,
    prefix_shape: torch.Size,
    reference: torch.Tensor,
) -> torch.Tensor:
    threshold = torch.as_tensor(
        value, device=reference.device, dtype=reference.dtype
    )
    if not bool(torch.isfinite(threshold).all().item()):
        raise DualConditionalRatioError(f"{name} contains NaN or infinity")
    try:
        return torch.broadcast_to(threshold, prefix_shape)
    except RuntimeError as error:
        raise DualConditionalRatioError(
            f"{name} is not broadcastable to candidate batch dimensions"
        ) from error


def _axis_threshold(
    name: str,
    value: torch.Tensor,
    *,
    batch_shape: torch.Size,
    axis_count: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DualConditionalRatioError(f"{name} must be a torch.Tensor")
    if value.device != reference.device:
        raise DualConditionalRatioError(f"{name} device differs from margins")
    if not value.is_floating_point():
        raise DualConditionalRatioError(f"{name} must be floating point")
    threshold = value.to(dtype=reference.dtype)
    if not bool(torch.isfinite(threshold).all().item()):
        raise DualConditionalRatioError(f"{name} contains NaN or infinity")
    target_shape = (*batch_shape, axis_count)
    try:
        return torch.broadcast_to(threshold, target_shape)
    except RuntimeError as error:
        raise DualConditionalRatioError(
            f"{name} must contain one threshold per semantic axis and cannot "
            "vary along the candidate axis"
        ) from error


def multi_axis_lexicographic_candidate_selection(
    action_margins: torch.Tensor,
    preservation_margins: torch.Tensor,
    *,
    action_thresholds: torch.Tensor,
    preservation_thresholds: torch.Tensor,
    tie_breaker: torch.Tensor | None = None,
) -> MultiAxisLexicographicSelection:
    """Apply per-axis hard gates and select by worst calibrated margins.

    Inputs have layouts ``[..., K, J_A]`` and ``[..., K, J_P]``.  A candidate
    passes a family only when *every* semantic axis meets its corresponding
    threshold.  The threshold tensors contain one value per axis (optionally
    broadcast across batch dimensions) and never vary across candidates.

    If jointly feasible candidates exist, the selector maximizes their worst
    calibrated preservation margin first and their worst calibrated action
    margin second.  Here ``calibrated margin = observed margin - threshold``.
    The optional ``[..., K]`` tie breaker is third; exact ties choose the lowest
    candidate index.  If the hard-gate intersection is empty, index ``-1`` is
    returned instead of promoting a unilateral winner.
    """

    action_raw = _require_floating_tensor("action_margins", action_margins)
    preservation_raw = _require_floating_tensor(
        "preservation_margins", preservation_margins
    )
    if action_raw.ndim < 2 or int(action_raw.shape[-2]) < 1 or int(
        action_raw.shape[-1]
    ) < 1:
        raise DualConditionalRatioError(
            "action_margins must have nonempty [..., K, J_A] layout"
        )
    if preservation_raw.ndim < 2 or int(
        preservation_raw.shape[-1]
    ) < 1:
        raise DualConditionalRatioError(
            "preservation_margins must have nonempty [..., K, J_P] layout"
        )
    if action_raw.device != preservation_raw.device:
        raise DualConditionalRatioError(
            "action and preservation margins must share one device"
        )
    if action_raw.shape[:-1] != preservation_raw.shape[:-1]:
        raise DualConditionalRatioError(
            "action and preservation candidate dimensions must agree"
        )

    score_dtype = _common_accumulation_dtype(action_raw, preservation_raw)
    action = action_raw.to(dtype=score_dtype)
    preservation = preservation_raw.to(dtype=score_dtype)
    batch_shape = action.shape[:-2]
    candidate_shape = action.shape[:-1]
    action_axis_count = int(action.shape[-1])
    preservation_axis_count = int(preservation.shape[-1])
    action_threshold = _axis_threshold(
        "action_thresholds",
        action_thresholds,
        batch_shape=batch_shape,
        axis_count=action_axis_count,
        reference=action,
    ).unsqueeze(-2)
    preservation_threshold = _axis_threshold(
        "preservation_thresholds",
        preservation_thresholds,
        batch_shape=batch_shape,
        axis_count=preservation_axis_count,
        reference=preservation,
    ).unsqueeze(-2)

    action_calibrated = action - action_threshold
    preservation_calibrated = preservation - preservation_threshold
    action_axis_pass = action_calibrated >= 0.0
    preservation_axis_pass = preservation_calibrated >= 0.0
    action_pass = action_axis_pass.all(dim=-1)
    preservation_pass = preservation_axis_pass.all(dim=-1)
    joint_pass = action_pass & preservation_pass
    has_feasible = joint_pass.any(dim=-1)
    worst_action = action_calibrated.min(dim=-1).values
    worst_preservation = preservation_calibrated.min(dim=-1).values

    if tie_breaker is None:
        tie = torch.zeros(candidate_shape, device=action.device, dtype=score_dtype)
    else:
        tie_raw = _require_floating_tensor("tie_breaker", tie_breaker)
        if tie_raw.shape != candidate_shape:
            raise DualConditionalRatioError(
                "tie_breaker must have exact candidate layout [..., K]"
            )
        if tie_raw.device != action.device:
            raise DualConditionalRatioError(
                "tie_breaker device differs from margins"
            )
        tie = tie_raw.to(dtype=score_dtype)

    negative_infinity = action.new_tensor(float("-inf"))
    preservation_masked = torch.where(
        joint_pass, worst_preservation, negative_infinity
    )
    best_preservation = preservation_masked.max(dim=-1, keepdim=True).values
    preservation_winner = joint_pass & (
        worst_preservation == best_preservation
    )
    action_masked = torch.where(
        preservation_winner, worst_action, negative_infinity
    )
    best_action = action_masked.max(dim=-1, keepdim=True).values
    action_winner = preservation_winner & (worst_action == best_action)
    tie_masked = torch.where(action_winner, tie, negative_infinity)
    selected = tie_masked.argmax(dim=-1).to(dtype=torch.long)
    selected = torch.where(
        has_feasible,
        selected,
        torch.full_like(selected, -1),
    )
    return MultiAxisLexicographicSelection(
        index=selected,
        has_feasible=has_feasible,
        action_axis_pass=action_axis_pass,
        preservation_axis_pass=preservation_axis_pass,
        action_pass=action_pass,
        preservation_pass=preservation_pass,
        joint_pass=joint_pass,
        worst_action_calibrated_margin=worst_action,
        worst_preservation_calibrated_margin=worst_preservation,
    )


def lexicographic_candidate_selection(
    action_score: torch.Tensor,
    preservation_score: torch.Tensor,
    *,
    action_threshold: float | torch.Tensor,
    preservation_threshold: float | torch.Tensor,
    tie_breaker: torch.Tensor | None = None,
) -> LexicographicSelection:
    """One-axis compatibility wrapper for the multi-axis hard selector.

    Candidate scores have shape ``[..., K]``.  First, both hard gates are
    applied; if their intersection is empty, the returned index is ``-1``.
    Among jointly feasible candidates, preservation is maximized first, action
    second, and the optional tie breaker third.  Exact remaining ties choose the
    lowest candidate index through ``argmax``'s deterministic first occurrence.

    This ordering implements the constrained policy: action must happen, then
    source preservation decides among action-valid outputs.  A source copy and
    a source-agnostic T2V regeneration cannot be mislabeled as success merely by
    winning one side of a weighted sum.
    """

    action = _require_floating_tensor("action_score", action_score)
    if action.ndim < 1 or int(action.shape[-1]) < 1:
        raise DualConditionalRatioError(
            "action_score must have a nonempty candidate axis [..., K]"
        )
    preservation = _require_same_tensor_contract(
        "preservation_score",
        preservation_score,
        "action_score",
        action,
        require_same_dtype=False,
    )
    prefix_shape = action.shape[:-1]
    action_gate = _candidate_threshold(
        "action_threshold",
        action_threshold,
        prefix_shape=prefix_shape,
        reference=action,
    )
    preservation_gate = _candidate_threshold(
        "preservation_threshold",
        preservation_threshold,
        prefix_shape=prefix_shape,
        reference=preservation,
    )
    multi_axis = multi_axis_lexicographic_candidate_selection(
        action.unsqueeze(-1),
        preservation.unsqueeze(-1),
        action_thresholds=action_gate.unsqueeze(-1),
        preservation_thresholds=preservation_gate.unsqueeze(-1),
        tie_breaker=tie_breaker,
    )
    return LexicographicSelection(
        index=multi_axis.index,
        has_feasible=multi_axis.has_feasible,
        action_pass=multi_axis.action_pass,
        preservation_pass=multi_axis.preservation_pass,
        joint_pass=multi_axis.joint_pass,
    )


__all__ = [
    "DEFAULT_ENERGY_EPSILON",
    "SCHEMA_VERSION",
    "DualConditionalRatioError",
    "LexicographicSelection",
    "MultiAxisLexicographicSelection",
    "MultiSigmaRatioDiagnostics",
    "RectifiedFlowState",
    "action_t2v_cond_vs_noop_proxy",
    "aggregate_multi_sigma",
    "denoising_error_log_ratio_proxy",
    "lexicographic_candidate_selection",
    "masked_per_sample_mse",
    "multi_axis_lexicographic_candidate_selection",
    "multi_sigma_denoising_error_ratio_proxy",
    "rectified_flow_state",
    "source_correct_vs_wrong_proxy",
]
