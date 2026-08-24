"""Exact candidate-own action-energy cotangent for CAGE-Edit.

The frozen MACE scorer deliberately stops gradients and is kept unchanged.
This sibling module implements the training-side operation that MACE does not:

1. scan every registered prompt at every registered sigma under ``no_grad``;
2. detach a single global worst ``(sigma, hard-negative)`` cell;
3. replay only the action and selected hard-negative denoisers with respect to
   their noisy *input*; and
4. return the exact gradient of the selected flow-denoising margin with
   respect to the student's clean candidate.

For a clean candidate ``y`` and frozen T2V denoiser ``v_c``::

    q_s(y) = (1 - s) * y + s * epsilon
    E_c(y) = mean((v_c(q_s(y)) - (epsilon - y)) ** 2)

The returned gradient includes both terms

    2 / D * [I + (1 - s) * J_v(q_s)^T] * residual,

so it is not a velocity-norm score and not a transported T2V velocity.  Pure
T2V videos are absent from every public API.  They may calibrate prompts and
margins upstream, but never enter this operation as a target, condition,
noise, donor, or latent.

The implementation is intentionally batch-one.  A single detached worst cell
must be replayed for each candidate; callers obtain data parallelism by
placing independent candidates on separate DP replicas.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as functional

import mace_candidate_action_energy as mace


SCHEMA_VERSION = "bernini-cage-candidate-action-energy-input-vjp-v1"

SCAN_MODE = "no_grad_branch_scan"
REPLAY_MODE = "selected_input_vjp_replay"
FINITE_DIFFERENCE_PLUS_MODE = "finite_difference_plus_scan"
FINITE_DIFFERENCE_MINUS_MODE = "finite_difference_minus_scan"
_SCAN_MODES = frozenset(
    {SCAN_MODE, FINITE_DIFFERENCE_PLUS_MODE, FINITE_DIFFERENCE_MINUS_MODE}
)


class CAGECandidateActionEnergyVJPError(RuntimeError):
    """A candidate-own energy scan, replay, or gradient audit failed."""


@dataclass(frozen=True)
class EnergyCoordinate:
    """One common-noise T2V query coordinate."""

    coordinate_id: str
    sigma: float
    epsilon: torch.Tensor


@dataclass(frozen=True)
class DenoiseRequest:
    """Immutable request passed to a real or toy frozen T2V bridge."""

    mode: str
    coordinate_index: int
    coordinate_id: str
    sigma: float
    branch: str
    prompt: str
    x_sigma: torch.Tensor


@dataclass(frozen=True)
class EnergyVJPConfig:
    """Closed numerical contract for the selected worst-margin loss."""

    energy_epsilon: float = 1.0e-8
    target_margin: float = 0.0
    temperature: float = 1.0
    minimum_gradient_norm: float = 1.0e-12
    replay_rtol: float = 2.0e-5
    replay_atol: float = 2.0e-6
    finite_difference_rtol: float = 5.0e-2
    finite_difference_atol: float = 2.0e-4

    def validate(self) -> None:
        positive = {
            "energy_epsilon": self.energy_epsilon,
            "temperature": self.temperature,
            "minimum_gradient_norm": self.minimum_gradient_norm,
            "replay_rtol": self.replay_rtol,
            "replay_atol": self.replay_atol,
            "finite_difference_rtol": self.finite_difference_rtol,
            "finite_difference_atol": self.finite_difference_atol,
        }
        for name, value in positive.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise CAGECandidateActionEnergyVJPError(
                    f"{name} must be a positive finite scalar"
                )
        if (
            isinstance(self.target_margin, bool)
            or not isinstance(self.target_margin, (int, float))
            or not math.isfinite(float(self.target_margin))
        ):
            raise CAGECandidateActionEnergyVJPError(
                "target_margin must be a finite scalar"
            )


@dataclass(frozen=True)
class CandidateActionEnergyScan:
    """Detached multi-sigma branch scan and global worst-cell decision."""

    coordinate_ids: tuple[str, ...]
    sigmas: tuple[float, ...]
    branch_energies: torch.Tensor
    negative_log_energy_ratios: torch.Tensor
    selected_coordinate_index: int
    selected_coordinate_id: str
    selected_sigma: float
    selected_negative_index: int
    selected_negative_branch: str
    selected_action_energy: float
    selected_negative_energy: float
    selected_margin: float
    selected_loss: float
    call_order: tuple[tuple[str, int, str], ...]
    selection_detached: bool


@dataclass(frozen=True)
class CandidateActionEnergyVJPResult:
    """Exact selected-cell cotangent with serial replay diagnostics."""

    scan: CandidateActionEnergyScan
    direct_flow_target_gradient: torch.Tensor
    action_input_vjp: torch.Tensor
    negative_input_vjp: torch.Tensor
    gradient: torch.Tensor
    gradient_norm: float
    margin_derivative: float
    action_energy_derivative: float
    negative_energy_derivative: float
    replay_call_order: tuple[str, ...]
    replay_action_energy: float
    replay_negative_energy: float
    finite: bool
    nonzero: bool


@dataclass(frozen=True)
class FiniteDifferenceSignAudit:
    """Central-difference audit along normalized gradient descent."""

    step: float
    analytic_directional_derivative: float
    numerical_directional_derivative: float
    plus_loss: float
    minus_loss: float
    selected_coordinate_id: str
    selected_negative_branch: str
    selection_stable: bool
    descent_sign_passed: bool
    magnitude_agreement_passed: bool
    passed: bool


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise CAGECandidateActionEnergyVJPError(
            "contract value is not canonical finite ASCII JSON"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _detached_candidate(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.device.type == "meta"
        or value.ndim < 2
        or int(value.shape[0]) != 1
        or any(int(item) <= 0 for item in value.shape)
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise CAGECandidateActionEnergyVJPError(
            f"{label} must be detached finite FP32 batch-one [1,...]"
        )
    return value


def _validate_coordinates(
    value: Any, *, candidate: torch.Tensor
) -> tuple[EnergyCoordinate, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CAGECandidateActionEnergyVJPError(
            "coordinates must be a sequence"
        )
    result = tuple(value)
    if len(result) < 2:
        raise CAGECandidateActionEnergyVJPError(
            "CAGE action energy requires at least two sigma coordinates"
        )
    identifiers: list[str] = []
    for index, coordinate in enumerate(result):
        if not isinstance(coordinate, EnergyCoordinate):
            raise CAGECandidateActionEnergyVJPError(
                f"coordinate {index} is not an EnergyCoordinate"
            )
        if (
            not isinstance(coordinate.coordinate_id, str)
            or not coordinate.coordinate_id
            or coordinate.coordinate_id != coordinate.coordinate_id.strip()
            or "\x00" in coordinate.coordinate_id
        ):
            raise CAGECandidateActionEnergyVJPError(
                f"coordinate {index} has an invalid identifier"
            )
        if (
            isinstance(coordinate.sigma, bool)
            or not isinstance(coordinate.sigma, (int, float))
            or not math.isfinite(float(coordinate.sigma))
            or not 0.0 < float(coordinate.sigma) < 1.0
        ):
            raise CAGECandidateActionEnergyVJPError(
                f"coordinate {index} sigma must lie strictly in (0,1)"
            )
        epsilon = coordinate.epsilon
        if (
            not isinstance(epsilon, torch.Tensor)
            or epsilon.dtype != torch.float32
            or epsilon.shape != candidate.shape
            or epsilon.device != candidate.device
            or epsilon.requires_grad
            or epsilon.grad_fn is not None
            or not bool(torch.isfinite(epsilon).all().item())
        ):
            raise CAGECandidateActionEnergyVJPError(
                f"coordinate {index} epsilon must be detached finite FP32 with candidate geometry"
            )
        identifiers.append(coordinate.coordinate_id)
    if len(set(identifiers)) != len(identifiers):
        raise CAGECandidateActionEnergyVJPError(
            "energy coordinate identifiers must be unique"
        )
    return result


def _validate_prediction(
    value: Any,
    *,
    reference: torch.Tensor,
    label: str,
    replay: bool,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != reference.shape
        or value.device != reference.device
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise CAGECandidateActionEnergyVJPError(
            f"{label} must be a finite floating prediction with exact candidate geometry"
        )
    if replay:
        if not value.requires_grad or value.grad_fn is None:
            raise CAGECandidateActionEnergyVJPError(
                f"{label} replay must remain connected to x_sigma"
            )
    elif value.requires_grad or value.grad_fn is not None:
        raise CAGECandidateActionEnergyVJPError(
            f"{label} branch scan unexpectedly built a graph"
        )
    return value


def _loss_from_margin(margin: torch.Tensor, config: EnergyVJPConfig) -> torch.Tensor:
    return functional.softplus(
        (margin.new_tensor(float(config.target_margin)) - margin)
        / float(config.temperature)
    )


def _scan_candidate_action_energy(
    clean_candidate: torch.Tensor,
    coordinates: Sequence[EnergyCoordinate],
    prompt_by_branch: Mapping[str, str],
    denoise_callback: Callable[[DenoiseRequest], torch.Tensor],
    *,
    config: EnergyVJPConfig,
    scan_mode: str,
) -> CandidateActionEnergyScan:
    candidate = _detached_candidate(clean_candidate, label="clean_candidate")
    checked_coordinates = _validate_coordinates(coordinates, candidate=candidate)
    try:
        prompts = mace.validate_prompt_closure(prompt_by_branch)
    except mace.MACECandidateActionEnergyError as error:
        raise CAGECandidateActionEnergyVJPError(str(error)) from error
    if not callable(denoise_callback):
        raise CAGECandidateActionEnergyVJPError(
            "denoise_callback must be callable"
        )
    if scan_mode not in _SCAN_MODES:
        raise CAGECandidateActionEnergyVJPError("scan mode is not registered")

    candidate_before = candidate.clone()
    energy_rows: list[torch.Tensor] = []
    call_order: list[tuple[str, int, str]] = []
    for coordinate_index, coordinate in enumerate(checked_coordinates):
        epsilon_before = coordinate.epsilon.clone()
        sigma = float(coordinate.sigma)
        x_sigma = (
            (1.0 - sigma) * candidate + sigma * coordinate.epsilon
        ).detach()
        x_before = x_sigma.clone()
        target = coordinate.epsilon - candidate
        energies: list[torch.Tensor] = []
        with torch.no_grad():
            for branch in mace.BRANCH_ORDER:
                request = DenoiseRequest(
                    mode=scan_mode,
                    coordinate_index=coordinate_index,
                    coordinate_id=coordinate.coordinate_id,
                    sigma=sigma,
                    branch=branch,
                    prompt=prompts[branch],
                    x_sigma=x_sigma,
                )
                prediction = _validate_prediction(
                    denoise_callback(request),
                    reference=candidate,
                    label=f"scan {coordinate.coordinate_id}/{branch}",
                    replay=False,
                )
                residual = prediction.float() - target
                energy = residual.square().mean()
                if energy.dtype != torch.float32 or not bool(
                    torch.isfinite(energy).item()
                ):
                    raise CAGECandidateActionEnergyVJPError(
                        "branch scan produced an invalid energy"
                    )
                energies.append(energy.detach())
                call_order.append((scan_mode, coordinate_index, branch))
        if (
            not torch.equal(candidate, candidate_before)
            or not torch.equal(coordinate.epsilon, epsilon_before)
            or not torch.equal(x_sigma, x_before)
        ):
            raise CAGECandidateActionEnergyVJPError(
                "denoiser mutated a candidate-own scan tensor"
            )
        energy_rows.append(torch.stack(energies))

    branch_energies = torch.stack(energy_rows).detach().contiguous()
    action = branch_energies[:, :1]
    negatives = branch_energies[:, 1:]
    eps = float(config.energy_epsilon)
    margins = (
        torch.log(negatives + eps) - torch.log(action + eps)
    ).detach().contiguous()
    if margins.shape != (
        len(checked_coordinates),
        len(mace.HARD_NEGATIVE_BRANCHES),
    ) or not bool(torch.isfinite(margins).all().item()):
        raise CAGECandidateActionEnergyVJPError(
            "multi-sigma energy margins are invalid"
        )

    # torch.argmin is deterministic and returns the first row-major tie.  The
    # resulting Python integers make selection explicitly non-differentiable.
    flat_index = int(torch.argmin(margins.reshape(-1)).item())
    negative_count = len(mace.HARD_NEGATIVE_BRANCHES)
    coordinate_index = flat_index // negative_count
    negative_index = flat_index % negative_count
    selected_margin_tensor = margins[coordinate_index, negative_index]
    selected_loss_tensor = _loss_from_margin(selected_margin_tensor, config)
    if not bool(torch.isfinite(selected_loss_tensor).item()):
        raise CAGECandidateActionEnergyVJPError(
            "selected worst-margin loss is non-finite"
        )
    coordinate = checked_coordinates[coordinate_index]
    return CandidateActionEnergyScan(
        coordinate_ids=tuple(item.coordinate_id for item in checked_coordinates),
        sigmas=tuple(float(item.sigma) for item in checked_coordinates),
        branch_energies=branch_energies,
        negative_log_energy_ratios=margins,
        selected_coordinate_index=coordinate_index,
        selected_coordinate_id=coordinate.coordinate_id,
        selected_sigma=float(coordinate.sigma),
        selected_negative_index=negative_index,
        selected_negative_branch=mace.HARD_NEGATIVE_BRANCHES[negative_index],
        selected_action_energy=float(branch_energies[coordinate_index, 0].item()),
        selected_negative_energy=float(
            branch_energies[coordinate_index, 1 + negative_index].item()
        ),
        selected_margin=float(selected_margin_tensor.item()),
        selected_loss=float(selected_loss_tensor.item()),
        call_order=tuple(call_order),
        selection_detached=True,
    )


def scan_candidate_action_energy(
    clean_candidate: torch.Tensor,
    coordinates: Sequence[EnergyCoordinate],
    prompt_by_branch: Mapping[str, str],
    denoise_callback: Callable[[DenoiseRequest], torch.Tensor],
    *,
    config: Optional[EnergyVJPConfig] = None,
) -> CandidateActionEnergyScan:
    """Scan every branch/sigma under ``no_grad`` and detach one worst cell."""

    checked = EnergyVJPConfig() if config is None else config
    if not isinstance(checked, EnergyVJPConfig):
        raise CAGECandidateActionEnergyVJPError(
            "config must be EnergyVJPConfig"
        )
    checked.validate()
    return _scan_candidate_action_energy(
        clean_candidate,
        coordinates,
        prompt_by_branch,
        denoise_callback,
        config=checked,
        scan_mode=SCAN_MODE,
    )


def _selected_energy_derivatives(
    scan: CandidateActionEnergyScan, config: EnergyVJPConfig
) -> tuple[float, float, float]:
    z = (
        float(config.target_margin) - float(scan.selected_margin)
    ) / float(config.temperature)
    sigmoid = 1.0 / (1.0 + math.exp(-z)) if abs(z) < 40.0 else (
        1.0 if z > 0.0 else 0.0
    )
    margin_derivative = -sigmoid / float(config.temperature)
    action_derivative = -margin_derivative / (
        float(scan.selected_action_energy) + float(config.energy_epsilon)
    )
    negative_derivative = margin_derivative / (
        float(scan.selected_negative_energy) + float(config.energy_epsilon)
    )
    values = (margin_derivative, action_derivative, negative_derivative)
    if not all(math.isfinite(item) for item in values):
        raise CAGECandidateActionEnergyVJPError(
            "selected loss energy derivatives are non-finite"
        )
    return values


def compute_candidate_action_energy_vjp(
    clean_candidate: torch.Tensor,
    coordinates: Sequence[EnergyCoordinate],
    prompt_by_branch: Mapping[str, str],
    denoise_callback: Callable[[DenoiseRequest], torch.Tensor],
    *,
    config: Optional[EnergyVJPConfig] = None,
) -> CandidateActionEnergyVJPResult:
    """Return the exact selected worst-margin gradient with two replays only."""

    candidate = _detached_candidate(clean_candidate, label="clean_candidate")
    checked_coordinates = _validate_coordinates(coordinates, candidate=candidate)
    checked = EnergyVJPConfig() if config is None else config
    if not isinstance(checked, EnergyVJPConfig):
        raise CAGECandidateActionEnergyVJPError(
            "config must be EnergyVJPConfig"
        )
    checked.validate()
    try:
        prompts = mace.validate_prompt_closure(prompt_by_branch)
    except mace.MACECandidateActionEnergyError as error:
        raise CAGECandidateActionEnergyVJPError(str(error)) from error
    scan = _scan_candidate_action_energy(
        candidate,
        checked_coordinates,
        prompts,
        denoise_callback,
        config=checked,
        scan_mode=SCAN_MODE,
    )
    coordinate = checked_coordinates[scan.selected_coordinate_index]
    sigma = float(coordinate.sigma)
    target = coordinate.epsilon - candidate
    element_count = candidate.numel()
    if element_count <= 0:
        raise CAGECandidateActionEnergyVJPError(
            "candidate has no energy elements"
        )
    margin_d, action_d, negative_d = _selected_energy_derivatives(scan, checked)

    direct = torch.zeros_like(candidate)
    input_vjps: dict[str, torch.Tensor] = {}
    replay_energies: dict[str, float] = {}
    replay_order: list[str] = []
    selected = (
        (mace.ACTION_BRANCH, action_d, scan.selected_action_energy),
        (
            scan.selected_negative_branch,
            negative_d,
            scan.selected_negative_energy,
        ),
    )
    candidate_before = candidate.clone()
    epsilon_before = coordinate.epsilon.clone()
    for branch, energy_derivative, scanned_energy in selected:
        x_sigma = (
            (1.0 - sigma) * candidate + sigma * coordinate.epsilon
        ).detach().requires_grad_(True)
        request = DenoiseRequest(
            mode=REPLAY_MODE,
            coordinate_index=scan.selected_coordinate_index,
            coordinate_id=coordinate.coordinate_id,
            sigma=sigma,
            branch=branch,
            prompt=prompts[branch],
            x_sigma=x_sigma,
        )
        prediction = _validate_prediction(
            denoise_callback(request),
            reference=candidate,
            label=f"replay {coordinate.coordinate_id}/{branch}",
            replay=True,
        )
        residual = prediction.detach().float() - target
        replay_energy_tensor = residual.square().mean()
        replay_energy = float(replay_energy_tensor.item())
        if not math.isclose(
            replay_energy,
            float(scanned_energy),
            rel_tol=float(checked.replay_rtol),
            abs_tol=float(checked.replay_atol),
        ):
            raise CAGECandidateActionEnergyVJPError(
                f"selected {branch} replay changed scanned energy"
            )
        # dE/dprediction and the direct dE/dy term are identical because
        # residual = prediction - epsilon + y.  This direct accumulation is
        # the term omitted by a raw velocity-difference/SDS shortcut.
        prediction_cotangent = (
            residual * (2.0 * float(energy_derivative) / float(element_count))
        ).detach()
        direct = direct + prediction_cotangent
        input_gradient = torch.autograd.grad(
            prediction,
            x_sigma,
            grad_outputs=prediction_cotangent.to(prediction.dtype),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        if (
            input_gradient is None
            or input_gradient.shape != candidate.shape
            or not bool(torch.isfinite(input_gradient).all().item())
        ):
            raise CAGECandidateActionEnergyVJPError(
                f"selected {branch} input VJP is invalid"
            )
        input_vjps[branch] = input_gradient.detach().float().contiguous()
        replay_energies[branch] = replay_energy
        replay_order.append(branch)
        del prediction, x_sigma

    if (
        not torch.equal(candidate, candidate_before)
        or not torch.equal(coordinate.epsilon, epsilon_before)
    ):
        raise CAGECandidateActionEnergyVJPError(
            "selected input-VJP replay mutated candidate or epsilon"
        )
    gradient = (
        direct.float()
        + (1.0 - sigma)
        * (
            input_vjps[mace.ACTION_BRANCH]
            + input_vjps[scan.selected_negative_branch]
        )
    ).detach().contiguous()
    finite = bool(torch.isfinite(gradient).all().item())
    gradient_norm = float(torch.linalg.vector_norm(gradient).item())
    nonzero = math.isfinite(gradient_norm) and (
        gradient_norm > float(checked.minimum_gradient_norm)
    )
    if not finite:
        raise CAGECandidateActionEnergyVJPError(
            "candidate action-energy gradient is non-finite"
        )
    if not nonzero:
        raise CAGECandidateActionEnergyVJPError(
            "candidate action-energy gradient is zero or below the registered floor"
        )
    if tuple(replay_order) != (
        mace.ACTION_BRANCH,
        scan.selected_negative_branch,
    ):
        raise CAGECandidateActionEnergyVJPError(
            "input-VJP replay escaped the selected two-branch contract"
        )
    return CandidateActionEnergyVJPResult(
        scan=scan,
        direct_flow_target_gradient=direct.detach().float().contiguous(),
        action_input_vjp=input_vjps[mace.ACTION_BRANCH],
        negative_input_vjp=input_vjps[scan.selected_negative_branch],
        gradient=gradient,
        gradient_norm=gradient_norm,
        margin_derivative=margin_d,
        action_energy_derivative=action_d,
        negative_energy_derivative=negative_d,
        replay_call_order=tuple(replay_order),
        replay_action_energy=replay_energies[mace.ACTION_BRANCH],
        replay_negative_energy=replay_energies[scan.selected_negative_branch],
        finite=finite,
        nonzero=nonzero,
    )


def audit_candidate_action_energy_vjp_finite_difference(
    result: CandidateActionEnergyVJPResult,
    clean_candidate: torch.Tensor,
    coordinates: Sequence[EnergyCoordinate],
    prompt_by_branch: Mapping[str, str],
    denoise_callback: Callable[[DenoiseRequest], torch.Tensor],
    *,
    step: float,
    config: Optional[EnergyVJPConfig] = None,
) -> FiniteDifferenceSignAudit:
    """Fail closed unless a central difference verifies the descent sign."""

    if not isinstance(result, CandidateActionEnergyVJPResult):
        raise CAGECandidateActionEnergyVJPError(
            "finite-difference audit requires a CAGE VJP result"
        )
    candidate = _detached_candidate(clean_candidate, label="clean_candidate")
    checked_coordinates = _validate_coordinates(coordinates, candidate=candidate)
    checked = EnergyVJPConfig() if config is None else config
    if not isinstance(checked, EnergyVJPConfig):
        raise CAGECandidateActionEnergyVJPError(
            "config must be EnergyVJPConfig"
        )
    checked.validate()
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, float))
        or not math.isfinite(float(step))
        or float(step) <= 0.0
    ):
        raise CAGECandidateActionEnergyVJPError(
            "finite-difference step must be a positive finite scalar"
        )
    if result.gradient.shape != candidate.shape or not result.nonzero:
        raise CAGECandidateActionEnergyVJPError(
            "finite-difference audit gradient geometry differs"
        )
    direction = -result.gradient / result.gradient.new_tensor(result.gradient_norm)
    plus = (candidate + float(step) * direction).detach().contiguous()
    minus = (candidate - float(step) * direction).detach().contiguous()
    plus_scan = _scan_candidate_action_energy(
        plus,
        checked_coordinates,
        prompt_by_branch,
        denoise_callback,
        config=checked,
        scan_mode=FINITE_DIFFERENCE_PLUS_MODE,
    )
    minus_scan = _scan_candidate_action_energy(
        minus,
        checked_coordinates,
        prompt_by_branch,
        denoise_callback,
        config=checked,
        scan_mode=FINITE_DIFFERENCE_MINUS_MODE,
    )
    base_cell = (
        result.scan.selected_coordinate_index,
        result.scan.selected_negative_index,
    )
    plus_cell = (
        plus_scan.selected_coordinate_index,
        plus_scan.selected_negative_index,
    )
    minus_cell = (
        minus_scan.selected_coordinate_index,
        minus_scan.selected_negative_index,
    )
    stable = plus_cell == base_cell and minus_cell == base_cell
    if not stable:
        raise CAGECandidateActionEnergyVJPError(
            "finite-difference perturbation changed the detached worst cell"
        )
    analytic = float(torch.sum(result.gradient * direction).item())
    numerical = (
        float(plus_scan.selected_loss) - float(minus_scan.selected_loss)
    ) / (2.0 * float(step))
    sign_passed = (
        math.isfinite(analytic)
        and math.isfinite(numerical)
        and analytic < 0.0
        and numerical < 0.0
        and float(plus_scan.selected_loss) < float(minus_scan.selected_loss)
    )
    magnitude_passed = math.isclose(
        numerical,
        analytic,
        rel_tol=float(checked.finite_difference_rtol),
        abs_tol=float(checked.finite_difference_atol),
    )
    if not sign_passed:
        raise CAGECandidateActionEnergyVJPError(
            "finite difference does not verify the analytic descent sign"
        )
    if not magnitude_passed:
        raise CAGECandidateActionEnergyVJPError(
            "finite difference and analytic directional derivatives differ"
        )
    return FiniteDifferenceSignAudit(
        step=float(step),
        analytic_directional_derivative=analytic,
        numerical_directional_derivative=numerical,
        plus_loss=float(plus_scan.selected_loss),
        minus_loss=float(minus_scan.selected_loss),
        selected_coordinate_id=result.scan.selected_coordinate_id,
        selected_negative_branch=result.scan.selected_negative_branch,
        selection_stable=stable,
        descent_sign_passed=sign_passed,
        magnitude_agreement_passed=magnitude_passed,
        passed=True,
    )


def contract_receipt() -> Mapping[str, Any]:
    """Return a digest-bound statement of the mathematical/API boundary."""

    public = (
        scan_candidate_action_energy,
        compute_candidate_action_energy_vjp,
        audit_candidate_action_energy_vjp_finite_difference,
    )
    forbidden = {
        "source_video",
        "source_latent",
        "proposal_video",
        "proposal_latent",
        "target_video",
        "target_latent",
        "donor",
        "mask",
        "flow",
        "pose",
        "track",
        "trajectory",
    }
    offending = {
        function.__name__: sorted(
            set(inspect.signature(function).parameters) & forbidden
        )
        for function in public
        if set(inspect.signature(function).parameters) & forbidden
    }
    if offending:
        raise CAGECandidateActionEnergyVJPError(
            f"CAGE public API exposes forbidden media inputs: {offending}"
        )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "branch_order": list(mace.BRANCH_ORDER),
        "hard_negative_order": list(mace.HARD_NEGATIVE_BRANCHES),
        "coordinate_selection": "detached_global_argmin_over_sigma_and_negative",
        "scan": "all_branches_all_sigmas_under_no_grad",
        "replay": "selected_action_and_selected_hardest_negative_only",
        "energy": "mean_square(v_t2v((1-s)y+s*epsilon,c)-(epsilon-y))",
        "gradient": "direct_flow_target_term_plus_(1-sigma)_times_two_input_vjps",
        "velocity_norm_score": False,
        "pure_t2v_media_consumed": False,
        "proposal_target_noise_condition_donor_consumed": False,
        "finite_nonzero_gradient_required": True,
        "finite_difference_sign_audit_available": True,
        "batch_size": 1,
    }
    return {**unsigned, "digest": _object_sha256(unsigned)}


__all__ = [
    "CAGECandidateActionEnergyVJPError",
    "CandidateActionEnergyScan",
    "CandidateActionEnergyVJPResult",
    "DenoiseRequest",
    "EnergyCoordinate",
    "EnergyVJPConfig",
    "FINITE_DIFFERENCE_MINUS_MODE",
    "FINITE_DIFFERENCE_PLUS_MODE",
    "FiniteDifferenceSignAudit",
    "REPLAY_MODE",
    "SCAN_MODE",
    "SCHEMA_VERSION",
    "audit_candidate_action_energy_vjp_finite_difference",
    "compute_candidate_action_energy_vjp",
    "contract_receipt",
    "scan_candidate_action_energy",
]
