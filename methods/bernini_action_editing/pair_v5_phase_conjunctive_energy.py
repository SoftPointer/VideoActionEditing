"""Phase-conjunctive candidate-own action energy for PAIR-v5.

The original MACE score averages every spatial and temporal element before it
compares the action prompt with its hard negatives.  That reduction can assign
a good score to a candidate that is easy to explain for most of the clip but
fails the decisive contact, ordering, or terminal phase.  This module keeps
the 21 exact81 latent phases until after five pre-registered milestone tests:
actor binding, direction, contact, order, and terminal state.

For one clean native-RV2V candidate, callers provide one shared Gaussian,
one shared sigma, and the detached FP32 frozen-T2V spatial predictions for the
closed action/hard-negative prompt registry.  Every prediction must have exact
``[B,16,21,H,W]`` geometry.  Spatial dimensions and channels are averaged
first.  Registered temporal weights are then applied independently for each
milestone.  The final reward is the minimum action-vs-negative log-energy
margin over the full milestone x hard-negative product.  Consequently an easy
global negative cannot compensate for one failed terminal or ordering test.

Temporal weights are JSON-closed and SHA-256 committed independently of any
rollout.  Evaluation requires the separately pinned commitment digest; an
embedded digest alone is deliberately insufficient because a caller could
otherwise re-seal weights after observing candidates.  The external scheduler
must persist that digest before rollout.  The scorer also binds an opaque
frozen-T2V receipt digest and emits a content-addressed evaluation receipt.

There is intentionally no input slot for proposal media, source/paired target
media, donors, spatial masks, flow, pose, tracks, or trajectories.  Detached
predictions are the only critic output consumed.  This tensor core can enforce
FP32, detachment, branch closure, and one candidate-state API; the opaque model
receipt remains the caller's evidence that the upstream T2V itself was frozen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import re
from typing import Any

import torch


SCHEMA_VERSION = "bernini-pair-v5-phase-conjunctive-action-energy-v1"
PHASE_WEIGHT_SCHEMA = "bernini-pair-v5-phase-weight-commitment-v1"
EVALUATION_RECEIPT_SCHEMA = (
    "bernini-pair-v5-phase-conjunctive-evaluation-receipt-v1"
)

FRAME_COUNT = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21

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

MILESTONE_ORDER = (
    "actor",
    "direction",
    "contact",
    "order",
    "terminal",
)

REQUIRED_CAUSAL_NEGATIVES = frozenset(
    {
        "incomplete",
        "reverse",
        "shuffle",
        "wrong_actor",
        "wrong_object",
        "camera_only",
        "appearance_only",
    }
)

FORBIDDEN_EXTERNAL_INPUT_NAMES = frozenset(
    {
        "source",
        "source_video",
        "source_latent",
        "target",
        "target_video",
        "target_latent",
        "paired_target",
        "proposal",
        "proposal_video",
        "proposal_latent",
        "proposal_noise",
        "donor",
        "donor_video",
        "donor_latent",
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
NORMALIZATION_ATOL = 1.0e-6
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_PHASE_WEIGHT_FIELDS = frozenset(
    {
        "schema_version",
        "frame_count",
        "latent_phases",
        "milestone_order",
        "weights_by_milestone",
        "normalization_policy",
        "coverage_policy",
        "registration_digest",
    }
)
_EVALUATION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_digest",
        "phase_weight_registration_digest",
        "frozen_t2v_receipt_digest",
        "branch_order",
        "hard_negative_branches",
        "milestone_order",
        "candidate_shape",
        "energy_epsilon",
        "shared_candidate_state_sha256",
        "velocity_label_sha256",
        "sigma_sha256",
        "prediction_sha256_by_branch",
        "per_phase_branch_energy_sha256",
        "milestone_margin_sha256",
        "reward_sha256",
        "prediction_policy",
        "conjunction_policy",
        "proposal_visual_data_consumed",
        "privileged_visual_inputs_consumed",
        "receipt_digest",
    }
)


class PairV5PhaseEnergyError(ValueError):
    """The phase-conjunctive action-energy packet violates its contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one receipt object in the only accepted canonical form."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PairV5PhaseEnergyError("receipt value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed_mapping(
    value: Any,
    expected: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairV5PhaseEnergyError(f"{label} must be a mapping")
    keys = set(value.keys())
    if not all(isinstance(key, str) for key in keys):
        raise PairV5PhaseEnergyError(f"{label} keys must all be strings")
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise PairV5PhaseEnergyError(
            f"{label} closure differs; missing={missing}, extra={extra}"
        )
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5PhaseEnergyError(f"{label} must be lowercase SHA-256")
    return value


def _positive_finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise PairV5PhaseEnergyError(f"{label} must be a positive finite float")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PairV5PhaseEnergyError(
            f"{label} must be a positive finite float"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise PairV5PhaseEnergyError(f"{label} must be a positive finite float")
    return result


def _canonical_float32_weight(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise PairV5PhaseEnergyError(f"{label} must be a finite nonnegative number")
    try:
        raw = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PairV5PhaseEnergyError(
            f"{label} must be a finite nonnegative number"
        ) from error
    if not math.isfinite(raw) or raw < 0.0:
        raise PairV5PhaseEnergyError(
            f"{label} must be a finite nonnegative number"
        )
    canonical = float(torch.tensor(raw, dtype=torch.float32).item())
    if not math.isfinite(canonical) or canonical < 0.0:
        raise PairV5PhaseEnergyError(
            f"{label} is not representable as a finite nonnegative FP32 weight"
        )
    return canonical


def _normalize_weight_input(weights_by_milestone: Any) -> dict[str, list[float]]:
    if not isinstance(weights_by_milestone, Mapping):
        raise PairV5PhaseEnergyError("weights_by_milestone must be a mapping")
    keys = set(weights_by_milestone.keys())
    expected = set(MILESTONE_ORDER)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected, key=str)
    if missing or extra:
        raise PairV5PhaseEnergyError(
            "milestone weight closure differs; "
            f"missing={missing}, extra={extra}"
        )

    result: dict[str, list[float]] = {}
    coverage = [0.0] * LATENT_PHASES
    for milestone in MILESTONE_ORDER:
        raw_vector = weights_by_milestone[milestone]
        if (
            isinstance(raw_vector, (str, bytes, bytearray))
            or not isinstance(raw_vector, Sequence)
            or len(raw_vector) != LATENT_PHASES
        ):
            raise PairV5PhaseEnergyError(
                f"weights for {milestone} must contain exactly {LATENT_PHASES} phases"
            )
        vector = [
            _canonical_float32_weight(
                item,
                label=f"weights_by_milestone[{milestone}][{phase}]",
            )
            for phase, item in enumerate(raw_vector)
        ]
        if not math.isclose(
            math.fsum(vector),
            1.0,
            rel_tol=0.0,
            abs_tol=NORMALIZATION_ATOL,
        ):
            raise PairV5PhaseEnergyError(
                f"weights for {milestone} must sum to one"
            )
        for phase, weight in enumerate(vector):
            coverage[phase] += weight
        result[milestone] = vector

    uncovered = [phase for phase, total in enumerate(coverage) if total <= 0.0]
    if uncovered:
        raise PairV5PhaseEnergyError(
            f"phase weights do not cover all {LATENT_PHASES} phases; "
            f"uncovered={uncovered}"
        )
    return result


def make_phase_weight_commitment(
    weights_by_milestone: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Seal the closed 21-phase milestone weights before rollout.

    This function accepts no candidate or prediction.  The returned digest is
    intended to be persisted by the rollout scheduler and then passed back as
    ``registered_phase_weight_digest`` during evaluation.
    """

    unsigned = {
        "schema_version": PHASE_WEIGHT_SCHEMA,
        "frame_count": FRAME_COUNT,
        "latent_phases": LATENT_PHASES,
        "milestone_order": list(MILESTONE_ORDER),
        "weights_by_milestone": _normalize_weight_input(weights_by_milestone),
        "normalization_policy": "each_milestone_nonnegative_sum_one_fp32",
        "coverage_policy": "every_one_of_21_phases_positive_in_at_least_one_milestone",
    }
    result = {**unsigned, "registration_digest": object_sha256(unsigned)}
    return validate_phase_weight_commitment(result)


def validate_phase_weight_commitment(value: Any) -> dict[str, Any]:
    """Validate closure, phase semantics, and the embedded weight digest."""

    row = _closed_mapping(value, _PHASE_WEIGHT_FIELDS, label="phase commitment")
    if row["schema_version"] != PHASE_WEIGHT_SCHEMA:
        raise PairV5PhaseEnergyError("phase commitment schema_version differs")
    if (
        type(row["frame_count"]) is not int
        or type(row["latent_phases"]) is not int
        or row["frame_count"] != FRAME_COUNT
        or row["latent_phases"] != LATENT_PHASES
    ):
        raise PairV5PhaseEnergyError("phase commitment is not exact81/21-phase")
    if row["milestone_order"] != list(MILESTONE_ORDER):
        raise PairV5PhaseEnergyError("phase commitment milestone order differs")
    if row["normalization_policy"] != "each_milestone_nonnegative_sum_one_fp32":
        raise PairV5PhaseEnergyError("phase commitment normalization policy differs")
    if (
        row["coverage_policy"]
        != "every_one_of_21_phases_positive_in_at_least_one_milestone"
    ):
        raise PairV5PhaseEnergyError("phase commitment coverage policy differs")
    weights = _normalize_weight_input(row["weights_by_milestone"])
    if canonical_json_bytes(weights) != canonical_json_bytes(
        row["weights_by_milestone"]
    ):
        raise PairV5PhaseEnergyError("phase commitment weights are not canonical FP32")
    declared = _sha256(
        row["registration_digest"], label="phase registration_digest"
    )
    unsigned = dict(row)
    unsigned.pop("registration_digest")
    if object_sha256(unsigned) != declared:
        raise PairV5PhaseEnergyError("phase commitment embedded digest mismatch")
    return dict(row)


def contract_receipt() -> dict[str, Any]:
    """Return the closed algorithm and API contract with an embedded digest."""

    signature = set(inspect.signature(evaluate_phase_conjunctive_energy).parameters)
    if not signature.isdisjoint(FORBIDDEN_EXTERNAL_INPUT_NAMES):
        raise PairV5PhaseEnergyError("public scorer exposes a forbidden input slot")
    if not REQUIRED_CAUSAL_NEGATIVES.issubset(HARD_NEGATIVE_BRANCHES):
        raise PairV5PhaseEnergyError("required causal negatives are not closed")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "frame_count": FRAME_COUNT,
        "latent_channels": LATENT_CHANNELS,
        "latent_phases": LATENT_PHASES,
        "branch_order": list(BRANCH_ORDER),
        "hard_negative_branches": list(HARD_NEGATIVE_BRANCHES),
        "milestone_order": list(MILESTONE_ORDER),
        "candidate_coordinate": "one_clean_candidate_one_epsilon_one_sigma",
        "prediction_geometry": "B_16_21_H_W",
        "prediction_policy": "caller_supplied_frozen_t2v_receipt_detached_fp32",
        "spatial_reduction": "mean_over_channel_height_width_keep_21_phases",
        "temporal_reduction": "pre_registered_nonnegative_normalized_milestone_weights",
        "conjunction": "minimum_over_every_milestone_x_hard_negative_log_energy_margin",
        "phase_digest_policy": "external_digest_pinned_before_rollout_must_match_commitment",
        "proposal_role": "offline_prompt_and_critic_calibration_only",
        "proposal_visual_data_consumed": False,
        "privileged_visual_inputs_consumed": False,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def _detached_fp32(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PairV5PhaseEnergyError(f"{name} must be a torch.Tensor")
    if value.device.type == "meta":
        raise PairV5PhaseEnergyError(f"{name} cannot be a meta tensor")
    if value.dtype != torch.float32:
        raise PairV5PhaseEnergyError(f"{name} must be detached FP32")
    if value.requires_grad or value.grad_fn is not None:
        raise PairV5PhaseEnergyError(f"{name} must be detached FP32")
    if not bool(torch.isfinite(value).all().item()):
        raise PairV5PhaseEnergyError(f"{name} contains NaN or infinity")
    return value


def _validate_candidate_state(
    clean_candidate: Any,
    epsilon: Any,
    sigma: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean = _detached_fp32("clean_candidate", clean_candidate)
    noise = _detached_fp32("epsilon", epsilon)
    if (
        clean.ndim != 5
        or int(clean.shape[0]) < 1
        or int(clean.shape[1]) != LATENT_CHANNELS
        or int(clean.shape[2]) != LATENT_PHASES
        or int(clean.shape[3]) <= 0
        or int(clean.shape[4]) <= 0
    ):
        raise PairV5PhaseEnergyError(
            "clean_candidate must be exact81 [B,16,21,H,W]"
        )
    if noise.shape != clean.shape or noise.device != clean.device:
        raise PairV5PhaseEnergyError(
            "epsilon must exactly share candidate geometry and device"
        )

    sigma_tensor = _detached_fp32("sigma", sigma)
    batch = int(clean.shape[0])
    if sigma_tensor.ndim == 0:
        sigma_tensor = sigma_tensor.expand(batch)
    elif sigma_tensor.ndim != 1 or int(sigma_tensor.shape[0]) != batch:
        raise PairV5PhaseEnergyError("sigma must be one scalar or exact [B]")
    if sigma_tensor.device != clean.device:
        raise PairV5PhaseEnergyError("sigma must share the candidate device")
    if bool(((sigma_tensor < 0.0) | (sigma_tensor > 1.0)).any().item()):
        raise PairV5PhaseEnergyError("sigma must remain in [0,1]")
    return clean, noise, sigma_tensor


def _validate_predictions(
    prediction_by_branch: Any,
    *,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not isinstance(prediction_by_branch, Mapping):
        raise PairV5PhaseEnergyError("prediction_by_branch must be a mapping")
    keys = set(prediction_by_branch.keys())
    expected = set(BRANCH_ORDER)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected, key=str)
    if missing or extra:
        raise PairV5PhaseEnergyError(
            f"prediction branch closure differs; missing={missing}, extra={extra}"
        )
    result: dict[str, torch.Tensor] = {}
    for branch in BRANCH_ORDER:
        prediction = _detached_fp32(
            f"prediction_by_branch[{branch}]", prediction_by_branch[branch]
        )
        if prediction.shape != reference.shape or prediction.device != reference.device:
            raise PairV5PhaseEnergyError(
                f"prediction for {branch} must share exact [B,16,21,H,W] geometry/device"
            )
        result[branch] = prediction
    return result


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash exact tensor metadata and contiguous little/native-endian bytes."""

    tensor = _detached_fp32("receipt tensor", value).detach().cpu().contiguous()
    header = {
        "dtype": "float32",
        "shape": [int(length) for length in tensor.shape],
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(header))
    digest.update(b"\x00")
    digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PhaseConjunctiveEnergyResult:
    """Detached FP32 phase diagnostics plus a content-addressed receipt.

    Shapes use ``K=10`` branches, ``N=9`` negatives, and ``M=5`` milestones:

    * ``per_phase_branch_energies`` is ``[K,B,21]``;
    * ``milestone_branch_energies`` is ``[M,K,B]``;
    * ``milestone_negative_log_energy_ratios`` is ``[M,N,B]``;
    * ``milestone_rewards`` is ``[M,B]``;
    * ``reward`` is ``[B]`` and is the full conjunction;
    * ``global_*`` are audit-only values showing what the old temporal mean
      would have reported.
    """

    x_sigma: torch.Tensor
    velocity_label: torch.Tensor
    per_phase_branch_energies: torch.Tensor
    milestone_branch_energies: torch.Tensor
    milestone_negative_log_energy_ratios: torch.Tensor
    milestone_rewards: torch.Tensor
    reward: torch.Tensor
    hardest_milestone_index: torch.Tensor
    hardest_negative_index: torch.Tensor
    global_branch_energies: torch.Tensor
    global_negative_log_energy_ratios: torch.Tensor
    global_reward: torch.Tensor
    receipt: Mapping[str, Any]


def _make_evaluation_receipt(
    *,
    phase_weight_digest: str,
    frozen_t2v_receipt_digest: str,
    candidate_shape: Sequence[int],
    energy_epsilon: float,
    x_sigma: torch.Tensor,
    velocity_label: torch.Tensor,
    sigma: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    per_phase_branch_energies: torch.Tensor,
    milestone_margins: torch.Tensor,
    reward: torch.Tensor,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": EVALUATION_RECEIPT_SCHEMA,
        "contract_digest": contract_receipt()["digest"],
        "phase_weight_registration_digest": phase_weight_digest,
        "frozen_t2v_receipt_digest": frozen_t2v_receipt_digest,
        "branch_order": list(BRANCH_ORDER),
        "hard_negative_branches": list(HARD_NEGATIVE_BRANCHES),
        "milestone_order": list(MILESTONE_ORDER),
        "candidate_shape": [int(length) for length in candidate_shape],
        "energy_epsilon": float(energy_epsilon),
        "shared_candidate_state_sha256": tensor_sha256(x_sigma),
        "velocity_label_sha256": tensor_sha256(velocity_label),
        "sigma_sha256": tensor_sha256(sigma),
        "prediction_sha256_by_branch": {
            branch: tensor_sha256(predictions[branch]) for branch in BRANCH_ORDER
        },
        "per_phase_branch_energy_sha256": tensor_sha256(
            per_phase_branch_energies
        ),
        "milestone_margin_sha256": tensor_sha256(milestone_margins),
        "reward_sha256": tensor_sha256(reward),
        "prediction_policy": "frozen_t2v_receipt_bound_detached_fp32",
        "conjunction_policy": "minimum_over_all_milestone_negative_pairs",
        "proposal_visual_data_consumed": False,
        "privileged_visual_inputs_consumed": False,
    }
    value = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    return validate_evaluation_receipt(value)


def validate_evaluation_receipt(value: Any) -> dict[str, Any]:
    """Validate closure and integrity of an emitted evaluation receipt."""

    row = _closed_mapping(
        value, _EVALUATION_RECEIPT_FIELDS, label="evaluation receipt"
    )
    if row["schema_version"] != EVALUATION_RECEIPT_SCHEMA:
        raise PairV5PhaseEnergyError("evaluation receipt schema_version differs")
    if row["contract_digest"] != contract_receipt()["digest"]:
        raise PairV5PhaseEnergyError("evaluation receipt contract digest differs")
    for field in (
        "phase_weight_registration_digest",
        "frozen_t2v_receipt_digest",
        "shared_candidate_state_sha256",
        "velocity_label_sha256",
        "sigma_sha256",
        "per_phase_branch_energy_sha256",
        "milestone_margin_sha256",
        "reward_sha256",
    ):
        _sha256(row[field], label=f"evaluation receipt {field}")
    if row["branch_order"] != list(BRANCH_ORDER):
        raise PairV5PhaseEnergyError("evaluation receipt branch order differs")
    if row["hard_negative_branches"] != list(HARD_NEGATIVE_BRANCHES):
        raise PairV5PhaseEnergyError("evaluation receipt negative order differs")
    if row["milestone_order"] != list(MILESTONE_ORDER):
        raise PairV5PhaseEnergyError("evaluation receipt milestone order differs")
    shape = row["candidate_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 5
        or any(type(length) is not int or length <= 0 for length in shape)
        or shape[1] != LATENT_CHANNELS
        or shape[2] != LATENT_PHASES
    ):
        raise PairV5PhaseEnergyError("evaluation receipt candidate shape differs")
    _positive_finite_float(row["energy_epsilon"], label="receipt energy_epsilon")
    prediction_hashes = row["prediction_sha256_by_branch"]
    if not isinstance(prediction_hashes, Mapping):
        raise PairV5PhaseEnergyError("prediction digest registry must be a mapping")
    keys = set(prediction_hashes.keys())
    expected = set(BRANCH_ORDER)
    if keys != expected:
        raise PairV5PhaseEnergyError("prediction digest registry closure differs")
    for branch in BRANCH_ORDER:
        _sha256(prediction_hashes[branch], label=f"prediction digest {branch}")
    if row["prediction_policy"] != "frozen_t2v_receipt_bound_detached_fp32":
        raise PairV5PhaseEnergyError("evaluation receipt prediction policy differs")
    if row["conjunction_policy"] != "minimum_over_all_milestone_negative_pairs":
        raise PairV5PhaseEnergyError("evaluation receipt conjunction policy differs")
    if row["proposal_visual_data_consumed"] is not False:
        raise PairV5PhaseEnergyError("evaluation receipt consumed proposal visual data")
    if row["privileged_visual_inputs_consumed"] is not False:
        raise PairV5PhaseEnergyError("evaluation receipt consumed privileged visual data")
    declared = _sha256(row["receipt_digest"], label="receipt_digest")
    unsigned = dict(row)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        raise PairV5PhaseEnergyError("evaluation receipt embedded digest mismatch")
    return dict(row)


def evaluate_phase_conjunctive_energy(
    clean_candidate: torch.Tensor,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    prediction_by_branch: Mapping[str, torch.Tensor],
    phase_weight_commitment: Mapping[str, Any],
    *,
    registered_phase_weight_digest: str,
    frozen_t2v_receipt_digest: str,
    energy_epsilon: float = DEFAULT_ENERGY_EPSILON,
) -> PhaseConjunctiveEnergyResult:
    """Evaluate the closed milestone x hard-negative conjunction.

    ``registered_phase_weight_digest`` must be the digest persisted before
    rollout, not merely copied from a commitment supplied after rollout.
    Predictions are read under one branch mapping and cannot introduce
    branch-specific candidate states, noise, or sigma through this API.
    """

    clean, noise, sigma_by_candidate = _validate_candidate_state(
        clean_candidate, epsilon, sigma
    )
    predictions = _validate_predictions(prediction_by_branch, reference=clean)
    commitment = validate_phase_weight_commitment(phase_weight_commitment)
    pinned_digest = _sha256(
        registered_phase_weight_digest,
        label="registered_phase_weight_digest",
    )
    if commitment["registration_digest"] != pinned_digest:
        raise PairV5PhaseEnergyError(
            "phase commitment differs from the pre-registered digest"
        )
    frozen_digest = _sha256(
        frozen_t2v_receipt_digest, label="frozen_t2v_receipt_digest"
    )
    epsilon_value = _positive_finite_float(
        energy_epsilon, label="energy_epsilon"
    )

    sigma_view = sigma_by_candidate.reshape(
        int(clean.shape[0]), 1, 1, 1, 1
    )
    x_sigma = (1.0 - sigma_view) * clean + sigma_view * noise
    velocity_label = noise - clean
    if x_sigma.dtype != torch.float32 or velocity_label.dtype != torch.float32:
        raise PairV5PhaseEnergyError("candidate-own state must remain FP32")
    if x_sigma.requires_grad or velocity_label.requires_grad:
        raise PairV5PhaseEnergyError("candidate-own state must remain detached")

    prediction_versions = {
        branch: int(predictions[branch]._version) for branch in BRANCH_ORDER
    }
    phase_energies: list[torch.Tensor] = []
    with torch.no_grad():
        for branch in BRANCH_ORDER:
            squared = (predictions[branch] - velocity_label).square()
            # Preserve the exact 21 latent phases; reduce C/H/W only.
            phase_energy = squared.mean(dim=(1, 3, 4))
            if (
                phase_energy.dtype != torch.float32
                or phase_energy.requires_grad
                or tuple(phase_energy.shape)
                != (int(clean.shape[0]), LATENT_PHASES)
            ):
                raise PairV5PhaseEnergyError(
                    f"per-phase energy for {branch} violates FP32 geometry"
                )
            phase_energies.append(phase_energy)

    for branch in BRANCH_ORDER:
        if int(predictions[branch]._version) != prediction_versions[branch]:
            raise PairV5PhaseEnergyError(
                f"prediction for {branch} mutated during evaluation"
            )

    per_phase_branch_energies = torch.stack(phase_energies, dim=0)
    phase_weights = torch.tensor(
        [
            commitment["weights_by_milestone"][milestone]
            for milestone in MILESTONE_ORDER
        ],
        dtype=torch.float32,
        device=clean.device,
    )
    if not torch.allclose(
        phase_weights.sum(dim=1),
        torch.ones(len(MILESTONE_ORDER), device=clean.device),
        rtol=0.0,
        atol=NORMALIZATION_ATOL,
    ):
        raise PairV5PhaseEnergyError("materialized milestone weights do not sum to one")

    # [M,T] x [K,B,T] -> [M,K,B].
    milestone_branch_energies = torch.einsum(
        "mt,kbt->mkb", phase_weights, per_phase_branch_energies
    )
    action_energy = milestone_branch_energies[:, :1, :]
    negative_energy = milestone_branch_energies[:, 1:, :]
    milestone_margins = torch.log(negative_energy + epsilon_value) - torch.log(
        action_energy + epsilon_value
    )
    milestone_rewards = milestone_margins.min(dim=1).values

    flattened = milestone_margins.reshape(
        len(MILESTONE_ORDER) * len(HARD_NEGATIVE_BRANCHES), int(clean.shape[0])
    )
    reward, hardest_flat_index = flattened.min(dim=0)
    hardest_milestone_index = torch.div(
        hardest_flat_index,
        len(HARD_NEGATIVE_BRANCHES),
        rounding_mode="floor",
    )
    hardest_negative_index = hardest_flat_index.remainder(
        len(HARD_NEGATIVE_BRANCHES)
    )

    # Audit-only old reduction: average all 21 phases before the comparison.
    global_branch_energies = per_phase_branch_energies.mean(dim=2)
    global_negative_log_energy_ratios = torch.log(
        global_branch_energies[1:] + epsilon_value
    ) - torch.log(global_branch_energies[:1] + epsilon_value)
    global_reward = global_negative_log_energy_ratios.min(dim=0).values

    fp32_diagnostics = (
        per_phase_branch_energies,
        milestone_branch_energies,
        milestone_margins,
        milestone_rewards,
        reward,
        global_branch_energies,
        global_negative_log_energy_ratios,
        global_reward,
    )
    if any(
        value.dtype != torch.float32
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
        for value in fp32_diagnostics
    ):
        raise PairV5PhaseEnergyError(
            "phase-conjunctive diagnostics must remain finite detached FP32"
        )

    receipt = _make_evaluation_receipt(
        phase_weight_digest=pinned_digest,
        frozen_t2v_receipt_digest=frozen_digest,
        candidate_shape=clean.shape,
        energy_epsilon=epsilon_value,
        x_sigma=x_sigma.detach(),
        velocity_label=velocity_label.detach(),
        sigma=sigma_by_candidate.detach(),
        predictions=predictions,
        per_phase_branch_energies=per_phase_branch_energies.detach(),
        milestone_margins=milestone_margins.detach(),
        reward=reward.detach(),
    )
    return PhaseConjunctiveEnergyResult(
        x_sigma=x_sigma.detach(),
        velocity_label=velocity_label.detach(),
        per_phase_branch_energies=per_phase_branch_energies.detach(),
        milestone_branch_energies=milestone_branch_energies.detach(),
        milestone_negative_log_energy_ratios=milestone_margins.detach(),
        milestone_rewards=milestone_rewards.detach(),
        reward=reward.detach(),
        hardest_milestone_index=hardest_milestone_index.detach(),
        hardest_negative_index=hardest_negative_index.detach(),
        global_branch_energies=global_branch_energies.detach(),
        global_negative_log_energy_ratios=global_negative_log_energy_ratios.detach(),
        global_reward=global_reward.detach(),
        receipt=receipt,
    )


__all__ = [
    "ACTION_BRANCH",
    "BRANCH_ORDER",
    "DEFAULT_ENERGY_EPSILON",
    "EVALUATION_RECEIPT_SCHEMA",
    "FORBIDDEN_EXTERNAL_INPUT_NAMES",
    "FRAME_COUNT",
    "HARD_NEGATIVE_BRANCHES",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "MILESTONE_ORDER",
    "PHASE_WEIGHT_SCHEMA",
    "PairV5PhaseEnergyError",
    "PhaseConjunctiveEnergyResult",
    "REQUIRED_CAUSAL_NEGATIVES",
    "SCHEMA_VERSION",
    "canonical_json_bytes",
    "contract_receipt",
    "evaluate_phase_conjunctive_energy",
    "make_phase_weight_commitment",
    "object_sha256",
    "tensor_sha256",
    "validate_evaluation_receipt",
    "validate_phase_weight_commitment",
]
