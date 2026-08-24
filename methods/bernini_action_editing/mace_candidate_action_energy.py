"""Frozen candidate-own-coordinate action-energy critic for Bernini MACE.

This module deliberately implements only a read-only tensor/core contract.  A
clean *student candidate* is noised once with one caller-supplied ``epsilon``
and ``sigma``.  A frozen T2V denoiser is then queried under the registered
action prompt and every registered hard-negative prompt on that exact same
state.  The generated action proposals used to design/calibrate those prompts
are never accepted as regression targets here.

The returned quantity is a denoising-energy ratio proxy, not a normalized
likelihood.  Positive reward means that the action condition has lower MSE
than every hard negative.  The minimum margin is used so an easy negative
cannot hide a failed reverse, incomplete, wrong-role, camera, or appearance
control.

There is intentionally no API slot for a source video, paired target, proposal
video, spatial mask, flow, pose, track, trajectory, or other privileged
condition.  The denoiser receives exactly ``(x_sigma, sigma, prompt)``.  This
core is also intentionally non-differentiable: it is a frozen scorer for
candidate selection/auditing, not a path for optimizing pixels through the
critic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch


SCHEMA_VERSION = "bernini-mace-candidate-own-coordinate-action-energy-v1"

ACTION_BRANCH = "action"
HARD_NEGATIVE_BRANCHES = (
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)
BRANCH_ORDER = (ACTION_BRANCH, *HARD_NEGATIVE_BRANCHES)

# These names are forbidden by construction: none is an argument of the only
# public evaluator.  Keeping the registry beside the executable contract makes
# accidental future API expansion straightforward to test.
FORBIDDEN_EXTERNAL_INPUT_NAMES = frozenset(
    {
        "source",
        "source_video",
        "source_latent",
        "target",
        "target_video",
        "target_latent",
        "proposal",
        "proposal_video",
        "proposal_latent",
        "mask",
        "motion_mask",
        "flow",
        "optical_flow",
        "pose",
        "track",
        "tracks",
        "trajectory",
        "trajectories",
        "reference_video",
        "edited_first_frame",
    }
)

DEFAULT_ENERGY_EPSILON = 1.0e-8
_ALLOWED_PREDICTION_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32}
)


class MACECandidateActionEnergyError(ValueError):
    """The frozen candidate-action energy contract is invalid."""


@dataclass(frozen=True)
class CandidateActionEnergyResult:
    """Per-candidate action-energy diagnostics.

    Shapes use ``B`` candidates, ten registered branches, and nine hard
    negatives:

    * ``branch_energies``: ``[10, B]`` in :data:`BRANCH_ORDER`;
    * ``negative_log_energy_ratios``: ``[9, B]`` in
      :data:`HARD_NEGATIVE_BRANCHES`;
    * ``reward``: ``[B]``, the minimum hard-negative margin; and
    * ``hardest_negative_index``: ``[B]``, indexing
      :data:`HARD_NEGATIVE_BRANCHES`.

    ``x_sigma`` and ``velocity_target`` are returned solely to make the
    candidate-own-coordinate construction auditable.  Both are detached FP32
    tensors constructed from the candidate and its shared noise, never from a
    proposal or paired target.
    """

    x_sigma: torch.Tensor
    velocity_target: torch.Tensor
    branch_energies: torch.Tensor
    negative_log_energy_ratios: torch.Tensor
    reward: torch.Tensor
    hardest_negative_index: torch.Tensor


def _require_detached_fp32_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise MACECandidateActionEnergyError(f"{name} must be a torch.Tensor")
    if value.device.type == "meta":
        raise MACECandidateActionEnergyError(f"{name} cannot be a meta tensor")
    if value.dtype != torch.float32:
        raise MACECandidateActionEnergyError(f"{name} must have dtype torch.float32")
    if value.requires_grad or value.grad_fn is not None:
        raise MACECandidateActionEnergyError(
            f"{name} must be detached and must not require gradients"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise MACECandidateActionEnergyError(f"{name} contains NaN or infinity")
    return value


def _validate_candidate_and_noise(
    clean_candidate: Any,
    epsilon: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean = _require_detached_fp32_tensor("clean_candidate", clean_candidate)
    noise = _require_detached_fp32_tensor("epsilon", epsilon)
    if clean.ndim < 2 or int(clean.shape[0]) < 1:
        raise MACECandidateActionEnergyError(
            "clean_candidate must have a non-empty batch-first [B, ...] layout"
        )
    if any(int(length) <= 0 for length in clean.shape):
        raise MACECandidateActionEnergyError(
            "clean_candidate cannot contain an empty dimension"
        )
    if noise.shape != clean.shape:
        raise MACECandidateActionEnergyError(
            "epsilon shape must exactly match clean_candidate"
        )
    if noise.device != clean.device:
        raise MACECandidateActionEnergyError(
            "epsilon device must exactly match clean_candidate"
        )
    return clean, noise


def _validate_sigma(
    sigma: Any,
    *,
    clean_candidate: torch.Tensor,
) -> torch.Tensor:
    value = _require_detached_fp32_tensor("sigma", sigma)
    if value.device != clean_candidate.device:
        raise MACECandidateActionEnergyError(
            "sigma device must exactly match clean_candidate"
        )
    batch_size = int(clean_candidate.shape[0])
    if value.ndim == 0:
        value = value.expand(batch_size)
    elif value.ndim == 1 and int(value.shape[0]) == batch_size:
        pass
    else:
        raise MACECandidateActionEnergyError(
            "sigma must be one FP32 scalar tensor or exact [B]"
        )
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise MACECandidateActionEnergyError("sigma must remain in [0, 1]")
    return value


def validate_prompt_closure(prompt_by_branch: Any) -> dict[str, str]:
    """Return the exact closed ten-branch prompt registry.

    Missing and additional branches both fail.  Prompt strings must be
    canonical non-empty UTF-8 text and must be distinct; otherwise a negative
    could silently alias the action or another control.
    """

    if not isinstance(prompt_by_branch, Mapping):
        raise MACECandidateActionEnergyError(
            "prompt_by_branch must be a mapping"
        )
    observed = set(prompt_by_branch.keys())
    expected = set(BRANCH_ORDER)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected, key=str)
    if missing or extra:
        raise MACECandidateActionEnergyError(
            f"prompt branch closure differs; missing={missing}, extra={extra}"
        )

    result: dict[str, str] = {}
    seen_prompts: dict[str, str] = {}
    for branch in BRANCH_ORDER:
        prompt = prompt_by_branch[branch]
        if not isinstance(prompt, str):
            raise MACECandidateActionEnergyError(
                f"prompt for {branch} must be a string"
            )
        if not prompt or prompt != prompt.strip() or "\x00" in prompt:
            raise MACECandidateActionEnergyError(
                f"prompt for {branch} must be canonical non-empty text"
            )
        try:
            prompt.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise MACECandidateActionEnergyError(
                f"prompt for {branch} is not valid UTF-8 text"
            ) from error
        prior = seen_prompts.get(prompt)
        if prior is not None:
            raise MACECandidateActionEnergyError(
                f"prompt for {branch} aliases branch {prior}"
            )
        seen_prompts[prompt] = branch
        result[branch] = prompt
    return result


def _tensor_mutation_token(tensor: torch.Tensor) -> tuple[str, int | None]:
    """Return a mutation token without touching an inference tensor version.

    PyTorch tensors materialized under ``torch.inference_mode()`` deliberately
    do not own a version counter; merely reading ``tensor._version`` raises.
    They also cannot be updated in-place by this evaluator's ``no_grad``
    forwards outside inference mode.  Ordinary tensors retain the stronger
    before/after version-counter audit used by the original implementation.
    """

    is_inference = getattr(torch, "is_inference", None)
    if callable(is_inference) and bool(is_inference(tensor)):
        return ("inference_immutable_outside_inference_mode", None)
    try:
        return ("version_counter", int(tensor._version))
    except RuntimeError as error:
        # Older PyTorch builds may expose inference tensors without
        # ``torch.is_inference``.  Only accept the one known no-counter error;
        # every other failure remains fail-closed.
        if "Inference tensors do not track version counter" in str(error):
            return ("inference_immutable_outside_inference_mode", None)
        raise MACECandidateActionEnergyError(
            "could not audit denoiser tensor mutation state"
        ) from error


def _module_state_contract(
    denoiser: Any,
    *,
    device: torch.device,
) -> tuple[
    tuple[
        str,
        int,
        tuple[str, int | None],
        torch.dtype,
        torch.device,
        tuple[int, ...],
    ],
    ...,
]:
    if not isinstance(denoiser, torch.nn.Module):
        raise MACECandidateActionEnergyError(
            "denoiser must be a torch.nn.Module so frozen state is auditable"
        )
    if denoiser.training:
        raise MACECandidateActionEnergyError("denoiser must be in eval mode")

    rows: list[tuple[str, int, int, torch.dtype, torch.device, tuple[int, ...]]] = []
    named_state = [
        *(('parameter.' + name, tensor) for name, tensor in denoiser.named_parameters()),
        *(('buffer.' + name, tensor) for name, tensor in denoiser.named_buffers()),
    ]
    names = [name for name, _ in named_state]
    if len(names) != len(set(names)):
        raise MACECandidateActionEnergyError("denoiser state names are not unique")
    for name, tensor in named_state:
        if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
            raise MACECandidateActionEnergyError(
                f"denoiser state {name} is not a materialized tensor"
            )
        if tensor.device != device:
            raise MACECandidateActionEnergyError(
                f"denoiser state {name} device differs from clean_candidate"
            )
        if name.startswith("parameter."):
            if tensor.requires_grad:
                raise MACECandidateActionEnergyError(
                    f"denoiser state {name} is trainable"
                )
            if tensor.grad is not None:
                raise MACECandidateActionEnergyError(
                    f"denoiser state {name} retains a gradient"
                )
        rows.append(
            (
                name,
                id(tensor),
                _tensor_mutation_token(tensor),
                tensor.dtype,
                tensor.device,
                tuple(int(x) for x in tensor.shape),
            )
        )
    return tuple(rows)


def _validate_prediction(
    prediction: Any,
    *,
    branch: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(prediction, torch.Tensor):
        raise MACECandidateActionEnergyError(
            f"denoiser prediction for {branch} must be a torch.Tensor"
        )
    if prediction.device.type == "meta":
        raise MACECandidateActionEnergyError(
            f"denoiser prediction for {branch} cannot be meta"
        )
    if prediction.shape != reference.shape:
        raise MACECandidateActionEnergyError(
            f"denoiser prediction for {branch} must have exact target-only shape"
        )
    if prediction.device != reference.device:
        raise MACECandidateActionEnergyError(
            f"denoiser prediction for {branch} device differs from candidate"
        )
    if prediction.dtype not in _ALLOWED_PREDICTION_DTYPES:
        raise MACECandidateActionEnergyError(
            f"denoiser prediction for {branch} has unsupported dtype"
        )
    if prediction.requires_grad or prediction.grad_fn is not None:
        raise MACECandidateActionEnergyError(
            f"denoiser prediction for {branch} unexpectedly carries gradients"
        )
    if not bool(torch.isfinite(prediction).all().item()):
        raise MACECandidateActionEnergyError(
            f"denoiser prediction for {branch} contains NaN or infinity"
        )
    return prediction


def _validate_energy_epsilon(value: Any) -> float:
    if isinstance(value, bool):
        raise MACECandidateActionEnergyError(
            "energy_epsilon must be a positive finite float"
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise MACECandidateActionEnergyError(
            "energy_epsilon must be a positive finite float"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise MACECandidateActionEnergyError(
            "energy_epsilon must be a positive finite float"
        )
    return result


def evaluate_candidate_action_energy(
    clean_candidate: torch.Tensor,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    prompt_by_branch: Mapping[str, str],
    denoiser: torch.nn.Module,
    *,
    energy_epsilon: float = DEFAULT_ENERGY_EPSILON,
) -> CandidateActionEnergyResult:
    """Evaluate one closed action-vs-hard-negatives energy packet.

    The caller provides one batch of clean candidate latents and one exact
    shared ``epsilon``/``sigma`` packet.  ``denoiser`` is called exactly ten
    times, once per branch in :data:`BRANCH_ORDER`, and receives the same
    ``x_sigma`` and normalized ``[B]`` sigma tensor objects every time.

    Direct critic gradients are forbidden.  Inputs must be detached, the
    denoiser must be frozen/eval, calls execute under ``torch.no_grad()``, and
    all returned diagnostics are detached FP32 tensors.
    """

    clean, noise = _validate_candidate_and_noise(clean_candidate, epsilon)
    sigma_by_candidate = _validate_sigma(sigma, clean_candidate=clean)
    prompts = validate_prompt_closure(prompt_by_branch)
    epsilon_value = _validate_energy_epsilon(energy_epsilon)
    pre_state = _module_state_contract(denoiser, device=clean.device)

    sigma_view = sigma_by_candidate.reshape(
        int(clean.shape[0]), *([1] * (clean.ndim - 1))
    )
    # FP32 is structural here, rather than merely an accumulation cast.
    x_sigma = (1.0 - sigma_view) * clean + sigma_view * noise
    velocity_target = noise - clean
    if x_sigma.dtype != torch.float32 or velocity_target.dtype != torch.float32:
        raise MACECandidateActionEnergyError(
            "candidate state and velocity target must remain FP32"
        )
    if x_sigma.requires_grad or velocity_target.requires_grad:
        raise MACECandidateActionEnergyError(
            "candidate state construction unexpectedly carries gradients"
        )

    x_version = int(x_sigma._version)
    sigma_version = int(sigma_by_candidate._version)
    energies: list[torch.Tensor] = []
    with torch.no_grad():
        for branch in BRANCH_ORDER:
            prediction = denoiser(x_sigma, sigma_by_candidate, prompts[branch])
            checked = _validate_prediction(
                prediction,
                branch=branch,
                reference=velocity_target,
            )
            squared_error = (checked.float() - velocity_target).square()
            energy = squared_error.flatten(start_dim=1).mean(dim=1)
            if energy.dtype != torch.float32 or energy.requires_grad:
                raise MACECandidateActionEnergyError(
                    f"energy for {branch} is not detached FP32"
                )
            if not bool(torch.isfinite(energy).all().item()):
                raise MACECandidateActionEnergyError(
                    f"energy for {branch} contains NaN or infinity"
                )
            energies.append(energy)
            if int(x_sigma._version) != x_version:
                raise MACECandidateActionEnergyError(
                    f"denoiser mutated the shared x_sigma while evaluating {branch}"
                )
            if int(sigma_by_candidate._version) != sigma_version:
                raise MACECandidateActionEnergyError(
                    f"denoiser mutated the shared sigma while evaluating {branch}"
                )

    post_state = _module_state_contract(denoiser, device=clean.device)
    if post_state != pre_state:
        raise MACECandidateActionEnergyError(
            "denoiser parameters or buffers changed during frozen evaluation"
        )
    if denoiser.training:
        raise MACECandidateActionEnergyError(
            "denoiser left eval mode during frozen evaluation"
        )

    branch_energies = torch.stack(energies, dim=0)
    action_energy = branch_energies[0]
    negative_energy = branch_energies[1:]
    # Evaluate the registered log ratio as one stable ``log1p`` expression.
    # The former ``log(E_neg + eps) - log(E_action + eps)`` subtracts two
    # nearly equal O(1) logarithms for the most scientifically interesting
    # candidates.  A one-ULP device difference in either logarithm can then
    # become thousands of ULPs in the small reward.  FP64 intermediates keep
    # ``epsilon`` effective even at the zero-energy boundary; the public
    # diagnostic remains detached FP32 as registered.
    action_energy_fp64 = action_energy.unsqueeze(0).to(torch.float64)
    negative_energy_fp64 = negative_energy.to(torch.float64)
    relative_regularized_energy_delta = (
        negative_energy_fp64 - action_energy_fp64
    ) / (action_energy_fp64 + epsilon_value)
    negative_log_energy_ratios = torch.log1p(
        relative_regularized_energy_delta
    ).to(torch.float32)
    reward, hardest_negative_index = negative_log_energy_ratios.min(dim=0)
    if (
        branch_energies.dtype != torch.float32
        or negative_log_energy_ratios.dtype != torch.float32
        or reward.dtype != torch.float32
        or branch_energies.requires_grad
        or negative_log_energy_ratios.requires_grad
        or reward.requires_grad
    ):
        raise MACECandidateActionEnergyError(
            "action-energy diagnostics must be detached FP32 tensors"
        )
    return CandidateActionEnergyResult(
        x_sigma=x_sigma.detach(),
        velocity_target=velocity_target.detach(),
        branch_energies=branch_energies.detach(),
        negative_log_energy_ratios=negative_log_energy_ratios.detach(),
        reward=reward.detach(),
        hardest_negative_index=hardest_negative_index.detach(),
    )


__all__ = [
    "ACTION_BRANCH",
    "BRANCH_ORDER",
    "CandidateActionEnergyResult",
    "DEFAULT_ENERGY_EPSILON",
    "FORBIDDEN_EXTERNAL_INPUT_NAMES",
    "HARD_NEGATIVE_BRANCHES",
    "MACECandidateActionEnergyError",
    "SCHEMA_VERSION",
    "evaluate_candidate_action_energy",
    "validate_prompt_closure",
]
