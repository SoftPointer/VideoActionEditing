#!/usr/bin/env python3
"""Dependency-light CAPER source-conditional segmented flow-DPO core.

This file is deliberately only a mathematical core.  It consumes already
computed tensors, validates their provenance/geometry, and returns either a
graph-connected scalar loss or a sealed zero-update decision.  It does not
call a model, construct an optimizer, or execute an optimizer step.

Four rollout siblings are mandatory: ``target``, ``noop``, ``incomplete``
and ``phase_order_violation``.  Their receipts must bind the exact same source,
seed, frozen checkpoint, inference contract, official Gaussian, and pinned
exact40 coordinate.  In particular, a target from one seed can never be paired
with a counterfactual from another seed.

The exact81 RGB-frame intervals are mapped to Bernini's 21 latent phases as
follows (half-open except for the final RGB endpoint)::

    RGB [0,20)  -> latent [0:5]
    RGB [20,40) -> latent [5:10]
    RGB [40,80] -> latent [10:21]

A sealed selector commitment assigns ``noop``, ``incomplete`` and
``phase_order_violation`` to these three slices exactly once each.  The
module exposes a canonical default assignment, but the committed mapping is
explicit and digest-bound rather than inferred after loss construction.

Each segment has its own lower-confidence-bound preference gate.  Onset and
transition have independent motion/contrast floors; completion instead has a
temporal-drift ceiling and terminal-state-separation floor so a genuinely
stable hold passes while jitter does not.  Gates are conjoined; losses or
margins from other segments cannot compensate for one failed segment.  The
tensor objective is the usual reference-corrected rectified-flow DPO energy,
evaluated separately inside each fixed temporal slice.

The same target state also supplies a conditional anchor.  Correct-source
prediction energy is compared independently with source-dropped and
wrong-identity prediction energy, again relative to a detached frozen
reference.  If either current visual margin is below its frozen-reference
margin, the whole call is a zero-update decision and no loss is exposed.

There is intentionally no public input for a spatial mask, track, pose, flow,
trajectory, target video, proposal/donor, or T2V pixels/latents.  Pure-T2V
artifacts may be useful to an upstream evaluator, but cannot enter this core.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import hashlib
import inspect
import json
import math
import re
import struct
from typing import Any, Optional

import torch
import torch.nn.functional as functional


SCHEMA_VERSION = "bernini-caper-source-conditional-segment-dpo-v1"
SIBLING_RECEIPT_SCHEMA_VERSION = "bernini-caper-same-seed-sibling-receipt-v1"
SEGMENT_COMMITMENT_SCHEMA_VERSION = (
    "bernini-caper-source-conditional-segment-commitment-v1"
)
DECISION_RECEIPT_SCHEMA_VERSION = (
    "bernini-caper-source-conditional-segment-dpo-decision-v1"
)

FRAME_COUNT = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21
EXACT40_STEP_COUNT = 40
PINNED_EXACT40_SCHEDULE_SHA256 = (
    "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2"
)

SIBLING_ROLES = (
    "target",
    "noop",
    "incomplete",
    "phase_order_violation",
)
REJECTED_SIBLING_ROLES = SIBLING_ROLES[1:]
CONDITIONAL_NEGATIVE_ROLES = ("source_dropped", "wrong_identity")
CONDITION_PROVENANCE_ROLES = (
    "correct_source",
    "source_dropped",
    "wrong_identity",
)
PREDICTION_ROLES = (*SIBLING_ROLES, *CONDITIONAL_NEGATIVE_ROLES)
PHASE_ORDER = ("onset", "transition", "completion")

# RGB semantics and the corresponding non-overlapping exact81 latent slices.
# Latent phase i is anchored at RGB frame 4*i.  Thus the first two RGB ranges
# are half-open and the final range includes frame 80 / latent phase 20.
PHASE_FRAME_RANGES = {
    "onset": (0, 20),
    "transition": (20, 40),
    "completion": (40, 80),
}
PHASE_LATENT_SLICES = {
    "onset": (0, 5),
    "transition": (5, 10),
    "completion": (10, 21),
}
PHASE_REJECTED_ROLES = {
    "onset": "noop",
    "transition": "phase_order_violation",
    "completion": "incomplete",
}

# Extracted from the pinned Bernini-R UniPC shift-5 exact40 schedule.  Keeping
# the constants here avoids importing a renderer, scheduler, or repository
# training stack into this small tensor core.
NATIVE_EXACT40_TIMESTEPS = (
    999, 994, 989, 984, 978, 972, 965, 959, 952, 945,
    937, 929, 921, 912, 902, 893, 882, 871, 859, 847,
    833, 819, 803, 787, 769, 750, 729, 707, 682, 655,
    625, 593, 556, 516, 470, 418, 359, 291, 211, 117,
)
NATIVE_EXACT40_SIGMAS = (
    0.9999989867210388,
    0.9949031472206116,
    0.9895941615104675,
    0.9840595126152039,
    0.978284478187561,
    0.9722530841827393,
    0.9659478068351746,
    0.9593496322631836,
    0.9524376392364502,
    0.9451888799667358,
    0.9375780820846558,
    0.9295775294303894,
    0.9211564660072327,
    0.912280797958374,
    0.9029127359390259,
    0.893010139465332,
    0.8825258612632751,
    0.871407151222229,
    0.8595945835113525,
    0.8470211625099182,
    0.8336109519004822,
    0.8192774057388306,
    0.8039219379425049,
    0.7874310612678528,
    0.7696741223335266,
    0.7504994869232178,
    0.7297303080558777,
    0.7071589827537537,
    0.6825404167175293,
    0.6555827856063843,
    0.6259360909461975,
    0.5931769013404846,
    0.55678790807724,
    0.5161304473876953,
    0.4704066216945648,
    0.41860657930374146,
    0.3594328761100769,
    0.2911904454231262,
    0.21162153780460358,
    0.11765105277299881,
)

FORBIDDEN_PUBLIC_INPUT_NAMES = frozenset(
    {
        "spatial_mask",
        "mask",
        "motion_mask",
        "track",
        "tracks",
        "pose",
        "flow",
        "optical_flow",
        "trajectory",
        "trajectories",
        "t2v_pixels",
        "t2v_frames",
        "t2v_video",
        "t2v_latent",
        "proposal",
        "proposal_video",
        "proposal_latent",
        "donor",
        "donor_video",
        "target_video",
        "target_pixels",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class CAPERSourceConditionalSegmentDPOError(ValueError):
    """A receipt, tensor, selector, or numerical contract is invalid."""


# Short aliases are useful to callers while keeping one canonical exception.
CAPERConditionalDPOError = CAPERSourceConditionalSegmentDPOError
CAPERContractError = CAPERSourceConditionalSegmentDPOError


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a finite ASCII JSON receipt in canonical key order."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise CAPERSourceConditionalSegmentDPOError(
            "receipt value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _canonical_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be a canonical identifier"
        )
    return value


def _seed_id(value: Any) -> int | str:
    if isinstance(value, bool):
        raise CAPERSourceConditionalSegmentDPOError(
            "seed_id must be a nonnegative integer or canonical identifier"
        )
    if isinstance(value, int):
        if not 0 <= value < 2**63:
            raise CAPERSourceConditionalSegmentDPOError(
                "integer seed_id must lie in [0,2^63)"
            )
        return value
    return _canonical_id(value, label="seed_id")


def _finite_scalar(
    value: Any,
    *,
    label: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be a finite scalar"
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be a finite scalar"
        ) from error
    if not math.isfinite(result):
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be a finite scalar"
        )
    if positive and result <= 0.0:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be positive"
        )
    if nonnegative and result < 0.0:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be nonnegative"
        )
    return result


def _float32_be_hex(value: float) -> str:
    return struct.pack(">f", float(value)).hex()


NATIVE_EXACT40_SIGMA_FLOAT32_BE_HEX = tuple(
    _float32_be_hex(value) for value in NATIVE_EXACT40_SIGMAS
)


def exact40_coordinate(schedule_index: int) -> dict[str, Any]:
    """Return the pinned timestep/sigma coordinate for one exact40 index."""

    if (
        isinstance(schedule_index, bool)
        or not isinstance(schedule_index, int)
        or not 0 <= schedule_index < EXACT40_STEP_COUNT
    ):
        raise CAPERSourceConditionalSegmentDPOError(
            "exact40_index must be an integer in [0,40)"
        )
    return {
        "exact40_schedule_sha256": PINNED_EXACT40_SCHEDULE_SHA256,
        "exact40_step_count": EXACT40_STEP_COUNT,
        "exact40_index": schedule_index,
        "exact40_timestep": NATIVE_EXACT40_TIMESTEPS[schedule_index],
        "exact40_sigma_float32_be_hex": (
            NATIVE_EXACT40_SIGMA_FLOAT32_BE_HEX[schedule_index]
        ),
    }


def exact40_sigma_tensor(
    schedule_index: int,
    *,
    device: Optional[torch.device | str] = None,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Build the exact FP32 sigma expected by this core's tensor API."""

    coordinate = exact40_coordinate(schedule_index)
    numeric = struct.unpack(
        ">f", bytes.fromhex(coordinate["exact40_sigma_float32_be_hex"])
    )[0]
    if batch_size is None:
        return torch.tensor(numeric, dtype=torch.float32, device=device)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise CAPERSourceConditionalSegmentDPOError(
            "batch_size must be a positive integer"
        )
    return torch.full(
        (batch_size,), numeric, dtype=torch.float32, device=device
    )


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and logical bytes independent of strides."""

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise CAPERSourceConditionalSegmentDPOError(
            "tensor hash requires a real torch.Tensor"
        )
    cpu = value.detach().to(device="cpu").contiguous()
    if cpu.layout != torch.strided:
        raise CAPERSourceConditionalSegmentDPOError(
            "tensor hash supports only dense strided tensors"
        )
    metadata = {
        "dtype": str(cpu.dtype),
        "shape": [int(item) for item in cpu.shape],
        "layout": str(cpu.layout),
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\x00")
    raw = cpu.view(torch.uint8).reshape(-1)
    # Avoid materializing one Python integer object per byte for a full latent.
    for chunk in raw.split(1024 * 1024):
        digest.update(bytes(chunk.tolist()))
    return digest.hexdigest()


_SIBLING_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "sibling_role",
        "source_id",
        "source_media_sha256",
        "seed_id",
        "checkpoint_tree_sha256",
        "inference_contract_sha256",
        "official_gaussian_tensor_sha256",
        "candidate_clean_latent_sha256",
        "exact40_schedule_sha256",
        "exact40_step_count",
        "exact40_index",
        "exact40_timestep",
        "exact40_sigma_float32_be_hex",
        "receipt_digest",
    }
)


def make_sibling_receipt(
    *,
    sibling_role: str,
    source_id: str,
    source_media_sha256: str,
    seed_id: int | str,
    checkpoint_tree_sha256: str,
    inference_contract_sha256: str,
    official_gaussian_tensor_sha256: str,
    candidate_clean_latent_sha256: str,
    exact40_index: int,
) -> dict[str, Any]:
    """Create one sealed same-state sibling receipt.

    The exact40 schedule details are derived rather than caller-selected.
    """

    if sibling_role not in SIBLING_ROLES:
        raise CAPERSourceConditionalSegmentDPOError(
            f"unsupported sibling_role: {sibling_role!r}"
        )
    coordinate = exact40_coordinate(exact40_index)
    unsigned = {
        "schema_version": SIBLING_RECEIPT_SCHEMA_VERSION,
        "sibling_role": sibling_role,
        "source_id": _canonical_id(source_id, label="source_id"),
        "source_media_sha256": _sha256(
            source_media_sha256, label="source_media_sha256"
        ),
        "seed_id": _seed_id(seed_id),
        "checkpoint_tree_sha256": _sha256(
            checkpoint_tree_sha256, label="checkpoint_tree_sha256"
        ),
        "inference_contract_sha256": _sha256(
            inference_contract_sha256, label="inference_contract_sha256"
        ),
        "official_gaussian_tensor_sha256": _sha256(
            official_gaussian_tensor_sha256,
            label="official_gaussian_tensor_sha256",
        ),
        "candidate_clean_latent_sha256": _sha256(
            candidate_clean_latent_sha256,
            label="candidate_clean_latent_sha256",
        ),
        **coordinate,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def _closed_mapping(
    value: Any,
    expected_fields: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CAPERSourceConditionalSegmentDPOError(f"{label} must be a mapping")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} keys must be strings"
        )
    missing = sorted(expected_fields - keys)
    extra = sorted(keys - expected_fields)
    if missing or extra:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} closure differs; missing={missing}, extra={extra}"
        )
    return value


def validate_sibling_receipt(value: Any) -> dict[str, Any]:
    """Validate one sealed receipt including its pinned exact40 coordinate."""

    row = _closed_mapping(
        value, _SIBLING_RECEIPT_FIELDS, label="sibling receipt"
    )
    if row["schema_version"] != SIBLING_RECEIPT_SCHEMA_VERSION:
        raise CAPERSourceConditionalSegmentDPOError(
            "sibling receipt schema_version differs"
        )
    role = row["sibling_role"]
    if role not in SIBLING_ROLES:
        raise CAPERSourceConditionalSegmentDPOError(
            f"unsupported sibling_role: {role!r}"
        )
    _canonical_id(row["source_id"], label="source_id")
    _sha256(row["source_media_sha256"], label="source_media_sha256")
    _seed_id(row["seed_id"])
    _sha256(row["checkpoint_tree_sha256"], label="checkpoint_tree_sha256")
    _sha256(
        row["inference_contract_sha256"], label="inference_contract_sha256"
    )
    _sha256(
        row["official_gaussian_tensor_sha256"],
        label="official_gaussian_tensor_sha256",
    )
    _sha256(
        row["candidate_clean_latent_sha256"],
        label="candidate_clean_latent_sha256",
    )
    if (
        isinstance(row["exact40_index"], bool)
        or not isinstance(row["exact40_index"], int)
    ):
        raise CAPERSourceConditionalSegmentDPOError(
            "exact40_index must be an integer in [0,40)"
        )
    expected_coordinate = exact40_coordinate(row["exact40_index"])
    observed_coordinate = {
        key: row[key] for key in expected_coordinate
    }
    if observed_coordinate != expected_coordinate:
        raise CAPERSourceConditionalSegmentDPOError(
            "sibling receipt exact40 coordinate differs from the pinned schedule"
        )
    declared = _sha256(row["receipt_digest"], label="receipt_digest")
    unsigned = dict(row)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        raise CAPERSourceConditionalSegmentDPOError(
            "sibling receipt digest mismatch"
        )
    return dict(row)


@dataclass(frozen=True)
class ValidatedSiblingCoordinate:
    """The one coordinate shared by every admitted counterfactual sibling."""

    source_id: str
    source_media_sha256: str
    seed_id: int | str
    checkpoint_tree_sha256: str
    inference_contract_sha256: str
    official_gaussian_tensor_sha256: str
    exact40_schedule_sha256: str
    exact40_step_count: int
    exact40_index: int
    exact40_timestep: int
    exact40_sigma_float32_be_hex: str
    candidate_clean_latent_sha256_by_role: Mapping[str, str]
    sibling_receipts_sha256: str
    same_seed_sibling_admission_digest: str

    def receipt(self) -> dict[str, Any]:
        return asdict(self)


_COORDINATE_FIELDS = (
    "source_id",
    "source_media_sha256",
    "checkpoint_tree_sha256",
    "inference_contract_sha256",
    "official_gaussian_tensor_sha256",
    "exact40_schedule_sha256",
    "exact40_step_count",
    "exact40_index",
    "exact40_timestep",
    "exact40_sigma_float32_be_hex",
)


def validate_sibling_receipts(
    sibling_receipts: Mapping[str, Mapping[str, Any]],
    *,
    clean_latents: Optional[Mapping[str, torch.Tensor]] = None,
    official_epsilon: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
) -> ValidatedSiblingCoordinate:
    """Validate exact same-source/same-seed coordinates for all four roles."""

    if not isinstance(sibling_receipts, Mapping):
        raise CAPERSourceConditionalSegmentDPOError(
            "sibling_receipts must be a mapping"
        )
    expected_roles = set(SIBLING_ROLES)
    observed_roles = set(sibling_receipts)
    if any(not isinstance(role, str) for role in observed_roles):
        raise CAPERSourceConditionalSegmentDPOError(
            "sibling_receipts keys must be strings"
        )
    missing = sorted(expected_roles - observed_roles)
    extra = sorted(observed_roles - expected_roles)
    if missing or extra:
        raise CAPERSourceConditionalSegmentDPOError(
            "sibling receipt role closure differs; "
            f"missing={missing}, extra={extra}"
        )
    rows: dict[str, dict[str, Any]] = {}
    for role in SIBLING_ROLES:
        row = validate_sibling_receipt(sibling_receipts[role])
        if row["sibling_role"] != role:
            raise CAPERSourceConditionalSegmentDPOError(
                f"sibling receipt mapping key/role differs for {role}"
            )
        rows[role] = row

    first = rows["target"]
    for role in SIBLING_ROLES[1:]:
        row = rows[role]
        if type(row["seed_id"]) is not type(first["seed_id"]) or (
            row["seed_id"] != first["seed_id"]
        ):
            raise CAPERSourceConditionalSegmentDPOError(
                f"cross-seed pair forbidden: target and {role} seed_id differ"
            )
        for field_name in _COORDINATE_FIELDS:
            if row[field_name] != first[field_name]:
                raise CAPERSourceConditionalSegmentDPOError(
                    f"same-state sibling coordinate differs: {field_name} ({role})"
                )

    if official_epsilon is not None:
        epsilon = _detached_exact81(
            official_epsilon, label="official_epsilon"
        )
        observed_epsilon_sha = tensor_sha256(epsilon)
        if observed_epsilon_sha != first["official_gaussian_tensor_sha256"]:
            raise CAPERSourceConditionalSegmentDPOError(
                "official_epsilon tensor hash differs from sibling receipts"
            )
    if clean_latents is not None:
        clean_rows = _closed_role_mapping(
            clean_latents, SIBLING_ROLES, label="clean_latents"
        )
        validated_clean = {
            role: _detached_exact81(
                clean_rows[role], label=f"clean_latents[{role}]"
            )
            for role in SIBLING_ROLES
        }
        first_clean = validated_clean["target"]
        for role in SIBLING_ROLES:
            tensor = validated_clean[role]
            if tensor.shape != first_clean.shape or tensor.device != first_clean.device:
                raise CAPERSourceConditionalSegmentDPOError(
                    f"clean_latents[{role}] geometry/device differs from target"
                )
            if tensor_sha256(tensor) != rows[role]["candidate_clean_latent_sha256"]:
                raise CAPERSourceConditionalSegmentDPOError(
                    f"clean_latents[{role}] hash differs from its sibling receipt"
                )
    if sigma is not None:
        _validate_sigma(
            sigma,
            batch=(int(official_epsilon.shape[0]) if official_epsilon is not None else None),
            device=(official_epsilon.device if official_epsilon is not None else None),
            expected_hex=first["exact40_sigma_float32_be_hex"],
        )

    receipt_payload = {
        role: rows[role] for role in SIBLING_ROLES
    }
    return ValidatedSiblingCoordinate(
        **{field_name: first[field_name] for field_name in _COORDINATE_FIELDS},
        seed_id=first["seed_id"],
        candidate_clean_latent_sha256_by_role={
            role: rows[role]["candidate_clean_latent_sha256"]
            for role in SIBLING_ROLES
        },
        sibling_receipts_sha256=object_sha256(receipt_payload),
        same_seed_sibling_admission_digest=object_sha256(receipt_payload),
    )


@dataclass(frozen=True)
class SegmentSelector:
    """Pre-selection evidence for one fixed temporal sibling comparison.

    ``margin`` is target score minus rejected score.  ``uncertainty`` is the
    already-combined nonnegative pair uncertainty.  The selector passes only
    when ``margin - uncertainty >= minimum_margin``.  Dynamics are recomputed
    from tensors.  Onset/transition preregister independent positive floors
    for target motion and target-vs-rejected motion contrast.  Completion
    instead preregisters an upper bound on target temporal drift plus a lower
    bound on terminal target-vs-rejected state separation.  This distinction
    prevents a stable terminal hold from being rejected in favor of jitter.
    """

    phase: str
    rejected_role: str
    margin: float
    uncertainty: float
    minimum_margin: float
    minimum_target_motion_energy: Optional[float] = None
    minimum_motion_contrast_energy: Optional[float] = None
    maximum_target_temporal_drift: Optional[float] = None
    minimum_terminal_state_separation: Optional[float] = None

    def __post_init__(self) -> None:
        if self.phase not in PHASE_ORDER:
            raise CAPERSourceConditionalSegmentDPOError(
                f"unknown selector phase: {self.phase!r}"
            )
        if self.rejected_role not in REJECTED_SIBLING_ROLES:
            raise CAPERSourceConditionalSegmentDPOError(
                f"selector {self.phase} must reject one counterfactual sibling"
            )
        _finite_scalar(self.margin, label=f"{self.phase} margin")
        _finite_scalar(
            self.uncertainty,
            label=f"{self.phase} uncertainty",
            nonnegative=True,
        )
        _finite_scalar(
            self.minimum_margin,
            label=f"{self.phase} minimum_margin",
            positive=True,
        )
        if self.phase in ("onset", "transition"):
            _finite_scalar(
                self.minimum_target_motion_energy,
                label=f"{self.phase} minimum_target_motion_energy",
                positive=True,
            )
            _finite_scalar(
                self.minimum_motion_contrast_energy,
                label=f"{self.phase} minimum_motion_contrast_energy",
                positive=True,
            )
            if (
                self.maximum_target_temporal_drift is not None
                or self.minimum_terminal_state_separation is not None
            ):
                raise CAPERSourceConditionalSegmentDPOError(
                    f"{self.phase} cannot declare completion-only thresholds"
                )
        else:
            if (
                self.minimum_target_motion_energy is not None
                or self.minimum_motion_contrast_energy is not None
            ):
                raise CAPERSourceConditionalSegmentDPOError(
                    "completion cannot declare positive-motion thresholds"
                )
            _finite_scalar(
                self.maximum_target_temporal_drift,
                label="completion maximum_target_temporal_drift",
                nonnegative=True,
            )
            _finite_scalar(
                self.minimum_terminal_state_separation,
                label="completion minimum_terminal_state_separation",
                positive=True,
            )

    @property
    def lower_confidence_margin(self) -> float:
        return float(self.margin) - float(self.uncertainty)

    @property
    def preference_passed(self) -> bool:
        return self.lower_confidence_margin >= float(self.minimum_margin)


_SEGMENT_SELECTOR_FIELDS = frozenset(
    field.name for field in fields(SegmentSelector)
)


def _selector(value: Any, *, phase: str) -> SegmentSelector:
    if isinstance(value, SegmentSelector):
        result = value
    else:
        row = _closed_mapping(
            value,
            _SEGMENT_SELECTOR_FIELDS,
            label=f"segment selector {phase}",
        )
        result = SegmentSelector(**dict(row))
    if result.phase != phase:
        raise CAPERSourceConditionalSegmentDPOError(
            f"segment selector key/phase differs for {phase}"
        )
    return result


def validate_segment_selectors(
    segment_selectors: Mapping[str, SegmentSelector | Mapping[str, Any]],
) -> dict[str, SegmentSelector]:
    """Require one and only one preregistered selector for every phase."""

    if not isinstance(segment_selectors, Mapping):
        raise CAPERSourceConditionalSegmentDPOError(
            "segment_selectors must be a mapping"
        )
    observed = set(segment_selectors)
    expected = set(PHASE_ORDER)
    if any(not isinstance(key, str) for key in observed):
        raise CAPERSourceConditionalSegmentDPOError(
            "segment selector keys must be strings"
        )
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise CAPERSourceConditionalSegmentDPOError(
            "segment selector phase closure differs; "
            f"missing={missing}, extra={extra}"
        )
    result = {
        phase: _selector(segment_selectors[phase], phase=phase)
        for phase in PHASE_ORDER
    }
    selected_roles = tuple(result[phase].rejected_role for phase in PHASE_ORDER)
    if set(selected_roles) != set(REJECTED_SIBLING_ROLES) or len(
        set(selected_roles)
    ) != len(selected_roles):
        raise CAPERSourceConditionalSegmentDPOError(
            "segment selectors must use noop/incomplete/phase_order_violation "
            "exactly once"
        )
    return result


_SEGMENT_COMMITMENT_FIELDS = frozenset(
    {
        "schema_version",
        "frame_count",
        "latent_phases",
        "phase_order",
        "rgb_frame_ranges",
        "latent_slices",
        "rejected_role_by_phase",
        "selectors",
        "gate_composition",
        "registration_digest",
    }
)


def _selector_receipt(selector: SegmentSelector) -> dict[str, Any]:
    return {
        "phase": selector.phase,
        "rejected_role": selector.rejected_role,
        "margin": float(selector.margin),
        "uncertainty": float(selector.uncertainty),
        "minimum_margin": float(selector.minimum_margin),
        "minimum_target_motion_energy": (
            None
            if selector.minimum_target_motion_energy is None
            else float(selector.minimum_target_motion_energy)
        ),
        "minimum_motion_contrast_energy": (
            None
            if selector.minimum_motion_contrast_energy is None
            else float(selector.minimum_motion_contrast_energy)
        ),
        "maximum_target_temporal_drift": (
            None
            if selector.maximum_target_temporal_drift is None
            else float(selector.maximum_target_temporal_drift)
        ),
        "minimum_terminal_state_separation": (
            None
            if selector.minimum_terminal_state_separation is None
            else float(selector.minimum_terminal_state_separation)
        ),
    }


def make_segment_commitment(
    segment_selectors: Mapping[str, SegmentSelector | Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal all three selector-evidence rows before loss construction."""

    selectors = validate_segment_selectors(segment_selectors)
    unsigned = {
        "schema_version": SEGMENT_COMMITMENT_SCHEMA_VERSION,
        "frame_count": FRAME_COUNT,
        "latent_phases": LATENT_PHASES,
        "phase_order": list(PHASE_ORDER),
        "rgb_frame_ranges": {
            phase: list(PHASE_FRAME_RANGES[phase]) for phase in PHASE_ORDER
        },
        "latent_slices": {
            phase: list(PHASE_LATENT_SLICES[phase]) for phase in PHASE_ORDER
        },
        "rejected_role_by_phase": {
            phase: selectors[phase].rejected_role for phase in PHASE_ORDER
        },
        "selectors": {
            phase: _selector_receipt(selectors[phase]) for phase in PHASE_ORDER
        },
        "gate_composition": (
            "every_phase_lcb_and_phase_specific_dynamics_gate_conjoined"
        ),
    }
    return {**unsigned, "registration_digest": object_sha256(unsigned)}


def validate_segment_commitment(
    value: Any,
    *,
    registered_segment_commitment_digest: Optional[str] = None,
) -> tuple[dict[str, SegmentSelector], str]:
    """Validate a sealed commitment and its separately registered digest."""

    row = _closed_mapping(
        value,
        _SEGMENT_COMMITMENT_FIELDS,
        label="segment commitment",
    )
    if row["schema_version"] != SEGMENT_COMMITMENT_SCHEMA_VERSION:
        raise CAPERSourceConditionalSegmentDPOError(
            "segment commitment schema_version differs"
        )
    if row["frame_count"] != FRAME_COUNT or row["latent_phases"] != LATENT_PHASES:
        raise CAPERSourceConditionalSegmentDPOError(
            "segment commitment is not exact81/21-phase"
        )
    expected_static = {
        "phase_order": list(PHASE_ORDER),
        "rgb_frame_ranges": {
            phase: list(PHASE_FRAME_RANGES[phase]) for phase in PHASE_ORDER
        },
        "latent_slices": {
            phase: list(PHASE_LATENT_SLICES[phase]) for phase in PHASE_ORDER
        },
        "gate_composition": (
            "every_phase_lcb_and_phase_specific_dynamics_gate_conjoined"
        ),
    }
    for key, expected in expected_static.items():
        if row[key] != expected:
            raise CAPERSourceConditionalSegmentDPOError(
                f"segment commitment {key} differs"
            )
    selectors = validate_segment_selectors(row["selectors"])
    expected_rejected_roles = {
        phase: selectors[phase].rejected_role for phase in PHASE_ORDER
    }
    if row["rejected_role_by_phase"] != expected_rejected_roles:
        raise CAPERSourceConditionalSegmentDPOError(
            "segment commitment rejected_role_by_phase differs"
        )
    canonical_selectors = {
        phase: _selector_receipt(selectors[phase]) for phase in PHASE_ORDER
    }
    if canonical_json_bytes(row["selectors"]) != canonical_json_bytes(
        canonical_selectors
    ):
        raise CAPERSourceConditionalSegmentDPOError(
            "segment commitment selectors are not canonical"
        )
    declared = _sha256(
        row["registration_digest"], label="segment registration_digest"
    )
    unsigned = dict(row)
    unsigned.pop("registration_digest")
    if object_sha256(unsigned) != declared:
        raise CAPERSourceConditionalSegmentDPOError(
            "segment commitment embedded digest mismatch"
        )
    if registered_segment_commitment_digest is not None:
        registered = _sha256(
            registered_segment_commitment_digest,
            label="registered_segment_commitment_digest",
        )
        if declared != registered:
            raise CAPERSourceConditionalSegmentDPOError(
                "segment commitment differs from externally registered digest"
            )
    return selectors, declared


def validate_condition_provenance_digests(
    value: Any,
) -> dict[str, str]:
    """Validate correct/drop/wrong condition construction provenance."""

    rows = _closed_role_mapping(
        value,
        CONDITION_PROVENANCE_ROLES,
        label="condition_provenance_digests",
    )
    result = {
        role: _sha256(rows[role], label=f"condition provenance {role}")
        for role in CONDITION_PROVENANCE_ROLES
    }
    if len(set(result.values())) != len(CONDITION_PROVENANCE_ROLES):
        raise CAPERSourceConditionalSegmentDPOError(
            "correct/drop/wrong condition provenance digests must be distinct"
        )
    return result


def _finite_tensor(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be a torch.Tensor"
        )
    if value.device.type == "meta":
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} cannot be a meta tensor"
        )
    if value.layout != torch.strided:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be a dense strided tensor"
        )
    if not value.is_floating_point():
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be floating point"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} contains NaN or infinity"
        )
    return value


def _detached_exact81(value: Any, *, label: str) -> torch.Tensor:
    tensor = _finite_tensor(value, label=label)
    if tensor.dtype != torch.float32:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be detached FP32"
        )
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be detached FP32"
        )
    if (
        tensor.ndim != 5
        or int(tensor.shape[0]) < 1
        or int(tensor.shape[1]) != LATENT_CHANNELS
        or int(tensor.shape[2]) != LATENT_PHASES
        or int(tensor.shape[3]) <= 0
        or int(tensor.shape[4]) <= 0
    ):
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be exact81 [B,16,21,H,W]"
        )
    return tensor


def _closed_role_mapping(
    value: Any,
    roles: Sequence[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CAPERSourceConditionalSegmentDPOError(f"{label} must be a mapping")
    observed = set(value)
    expected = set(roles)
    if any(not isinstance(key, str) for key in observed):
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} keys must be strings"
        )
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} role closure differs; missing={missing}, extra={extra}"
        )
    return value


def _validate_clean_latents(
    clean_latents: Mapping[str, torch.Tensor],
    official_epsilon: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rows = _closed_role_mapping(
        clean_latents, SIBLING_ROLES, label="clean_latents"
    )
    clean = {
        role: _detached_exact81(rows[role], label=f"clean_latents[{role}]")
        for role in SIBLING_ROLES
    }
    epsilon = _detached_exact81(official_epsilon, label="official_epsilon")
    target = clean["target"]
    for role, tensor in (*clean.items(), ("official_epsilon", epsilon)):
        if tensor.shape != target.shape or tensor.device != target.device:
            raise CAPERSourceConditionalSegmentDPOError(
                f"{role} geometry/device differs from target"
            )
    return clean, epsilon


def _prediction(
    value: Any,
    *,
    label: str,
    reference: torch.Tensor,
    trainable: bool,
) -> torch.Tensor:
    tensor = _finite_tensor(value, label=label)
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} has unsupported dtype"
        )
    if tensor.shape != reference.shape or tensor.device != reference.device:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} geometry/device differs from target"
        )
    if trainable:
        if not tensor.requires_grad:
            raise CAPERSourceConditionalSegmentDPOError(
                f"{label} must remain connected to the student"
            )
    elif tensor.requires_grad or tensor.grad_fn is not None:
        raise CAPERSourceConditionalSegmentDPOError(
            f"{label} must be a detached frozen-reference prediction"
        )
    return tensor


def _validate_predictions(
    student_predictions: Mapping[str, torch.Tensor],
    reference_predictions: Mapping[str, torch.Tensor],
    *,
    target: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    student_rows = _closed_role_mapping(
        student_predictions, PREDICTION_ROLES, label="student_predictions"
    )
    reference_rows = _closed_role_mapping(
        reference_predictions, PREDICTION_ROLES, label="reference_predictions"
    )
    student = {
        role: _prediction(
            student_rows[role],
            label=f"student_predictions[{role}]",
            reference=target,
            trainable=True,
        )
        for role in PREDICTION_ROLES
    }
    reference = {
        role: _prediction(
            reference_rows[role],
            label=f"reference_predictions[{role}]",
            reference=target,
            trainable=False,
        )
        for role in PREDICTION_ROLES
    }
    return student, reference


def _validate_sigma(
    value: Any,
    *,
    batch: Optional[int],
    device: Optional[torch.device],
    expected_hex: str,
) -> torch.Tensor:
    tensor = _finite_tensor(value, label="sigma")
    if tensor.dtype != torch.float32 or tensor.requires_grad or tensor.grad_fn is not None:
        raise CAPERSourceConditionalSegmentDPOError(
            "sigma must be detached FP32"
        )
    if tensor.ndim == 0:
        if batch is None:
            result = tensor.reshape(1)
        else:
            result = tensor.expand(batch)
    elif tensor.ndim == 1 and (batch is None or int(tensor.shape[0]) == batch):
        result = tensor
    else:
        raise CAPERSourceConditionalSegmentDPOError(
            "sigma must be one scalar or exact [B]"
        )
    if device is not None and tensor.device != device:
        raise CAPERSourceConditionalSegmentDPOError(
            "sigma and candidate tensors must use one device"
        )
    expected = struct.unpack(">f", bytes.fromhex(expected_hex))[0]
    expected_tensor = torch.full_like(result, expected)
    if not bool(torch.equal(result, expected_tensor)):
        raise CAPERSourceConditionalSegmentDPOError(
            "sigma tensor differs from the sibling exact40 coordinate"
        )
    return result


def _per_sample_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    temporal_slice: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    if temporal_slice is not None:
        begin, stop = temporal_slice
        prediction = prediction[:, :, begin:stop]
        target = target[:, :, begin:stop]
    result = (
        (prediction.float() - target).square().flatten(start_dim=1).mean(dim=1)
    )
    if result.dtype != torch.float32 or result.ndim != 1 or not bool(
        torch.isfinite(result).all().item()
    ):
        raise CAPERSourceConditionalSegmentDPOError(
            "per-sample denoising energy is non-finite or malformed"
        )
    return result


def _reference_corrected_pair(
    student_winner: torch.Tensor,
    student_rejected: torch.Tensor,
    reference_winner: torch.Tensor,
    reference_rejected: torch.Tensor,
    winner_target: torch.Tensor,
    rejected_target: torch.Tensor,
    *,
    beta: float,
    temporal_slice: Optional[tuple[int, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    student_winner_error = _per_sample_mse(
        student_winner, winner_target, temporal_slice=temporal_slice
    )
    student_rejected_error = _per_sample_mse(
        student_rejected, rejected_target, temporal_slice=temporal_slice
    )
    reference_winner_error = _per_sample_mse(
        reference_winner, winner_target, temporal_slice=temporal_slice
    )
    reference_rejected_error = _per_sample_mse(
        reference_rejected, rejected_target, temporal_slice=temporal_slice
    )
    current_margin = student_rejected_error - student_winner_error
    reference_margin = reference_rejected_error - reference_winner_error
    advantage = current_margin - reference_margin
    per_sample_loss = functional.softplus(-beta * advantage)
    loss = per_sample_loss.mean()
    values = (
        current_margin,
        reference_margin,
        advantage,
        per_sample_loss,
        loss,
    )
    if any(value.dtype != torch.float32 for value in values) or any(
        not bool(torch.isfinite(value).all().item()) for value in values
    ):
        raise CAPERSourceConditionalSegmentDPOError(
            "reference-corrected flow-DPO result is non-finite or non-FP32"
        )
    if not loss.requires_grad:
        raise CAPERSourceConditionalSegmentDPOError(
            "reference-corrected flow-DPO loss lost the student graph"
        )
    if reference_margin.requires_grad or reference_margin.grad_fn is not None:
        raise CAPERSourceConditionalSegmentDPOError(
            "frozen-reference margin unexpectedly retained a graph"
        )
    return loss, current_margin, reference_margin, advantage


def _active_segment_motion_energies(
    target_clean: torch.Tensor,
    rejected_clean: torch.Tensor,
    temporal_slice: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    begin, stop = temporal_slice
    target_segment = target_clean[:, :, begin:stop].float()
    rejected_segment = rejected_clean[:, :, begin:stop].float()
    target_delta = target_segment[:, :, 1:] - target_segment[:, :, :-1]
    rejected_delta = rejected_segment[:, :, 1:] - rejected_segment[:, :, :-1]
    target_energy = target_delta.square().flatten(start_dim=1).mean(dim=1)
    contrast_energy = (
        (target_delta - rejected_delta).square().flatten(start_dim=1).mean(dim=1)
    )
    if any(
        value.dtype != torch.float32
        or value.ndim != 1
        or not bool(torch.isfinite(value).all().item())
        for value in (target_energy, contrast_energy)
    ):
        raise CAPERSourceConditionalSegmentDPOError(
            "segment motion diagnostic is non-finite or malformed"
        )
    return target_energy, contrast_energy


def _completion_hold_diagnostics(
    target_clean: torch.Tensor,
    rejected_clean: torch.Tensor,
    temporal_slice: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample max hold drift and final-state separation.

    Drift is the maximum, over consecutive completion phases, of spatial/
    channel mean squared target displacement.  Terminal separation is the
    spatial/channel mean squared target-vs-rejected distance at latent phase
    20 (RGB frame 80).  Neither metric is averaged across batch samples.
    """

    begin, stop = temporal_slice
    target_segment = target_clean[:, :, begin:stop].float()
    target_delta = target_segment[:, :, 1:] - target_segment[:, :, :-1]
    per_step_drift = (
        target_delta.permute(0, 2, 1, 3, 4)
        .square()
        .flatten(start_dim=2)
        .mean(dim=2)
    )
    maximum_drift = per_step_drift.max(dim=1).values
    terminal_delta = (
        target_clean[:, :, stop - 1].float()
        - rejected_clean[:, :, stop - 1].float()
    )
    terminal_separation = terminal_delta.square().flatten(start_dim=1).mean(dim=1)
    if any(
        value.dtype != torch.float32
        or value.ndim != 1
        or not bool(torch.isfinite(value).all().item())
        for value in (maximum_drift, terminal_separation)
    ):
        raise CAPERSourceConditionalSegmentDPOError(
            "completion hold diagnostic is non-finite or malformed"
        )
    return maximum_drift, terminal_separation


def _tensor_min(value: torch.Tensor) -> float:
    return float(value.detach().min().cpu().item())


def _tensor_max(value: torch.Tensor) -> float:
    return float(value.detach().max().cpu().item())


def _decision_receipt(
    *,
    authorized: bool,
    reasons: Sequence[str],
    coordinate: ValidatedSiblingCoordinate,
    beta: float,
    segment_commitment_digest: str,
    condition_provenance_digests: Mapping[str, str],
    segment_gates: Mapping[str, Mapping[str, Any]],
    conditional_gates: Mapping[str, Mapping[str, Any]],
    segment_advantages: Mapping[str, torch.Tensor],
    conditional_advantages: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if not isinstance(coordinate, ValidatedSiblingCoordinate):
        raise CAPERSourceConditionalSegmentDPOError(
            "decision coordinate must be ValidatedSiblingCoordinate"
        )
    provenance = validate_condition_provenance_digests(
        condition_provenance_digests
    )
    if any(type(reason) is not str or not reason for reason in reasons):
        raise CAPERSourceConditionalSegmentDPOError(
            "decision reasons must be nonempty strings"
        )
    if authorized and reasons:
        raise CAPERSourceConditionalSegmentDPOError(
            "authorized decision cannot contain failure reasons"
        )
    if not authorized and not reasons:
        raise CAPERSourceConditionalSegmentDPOError(
            "zero-update decision requires at least one reason"
        )
    unsigned = {
        "schema_version": DECISION_RECEIPT_SCHEMA_VERSION,
        "objective_contract_digest": contract_receipt()["digest"],
        "beta": _finite_scalar(beta, label="decision beta", positive=True),
        "status": "LOSS_AUTHORIZED" if authorized else "ZERO_UPDATE",
        "update_authorized": authorized,
        "optimizer_steps_authorized": 1 if authorized else 0,
        "loss_exposed": authorized,
        # This mathematical core never owns optimizer construction/execution.
        "optimizer_created": False,
        "optimizer_steps_executed": 0,
        "same_seed_sibling_admission_digest": (
            coordinate.same_seed_sibling_admission_digest
        ),
        "segment_commitment_digest": _sha256(
            segment_commitment_digest,
            label="decision segment_commitment_digest",
        ),
        "condition_provenance_digests": {
            role: provenance[role]
            for role in CONDITION_PROVENANCE_ROLES
        },
        "same_seed_sibling_coordinate": coordinate.receipt(),
        "segment_gates": {
            phase: dict(segment_gates[phase]) for phase in PHASE_ORDER
        },
        "conditional_anchor_gates": {
            role: dict(conditional_gates[role])
            for role in CONDITIONAL_NEGATIVE_ROLES
        },
        "segment_advantage_sha256": {
            phase: tensor_sha256(segment_advantages[phase])
            for phase in PHASE_ORDER
        },
        "conditional_advantage_sha256": {
            role: tensor_sha256(conditional_advantages[role])
            for role in CONDITIONAL_NEGATIVE_ROLES
        },
        "gate_composition": "conjunctive_no_segment_or_batch_compensation",
        "loss_composition": "worst_constraint_max_no_action_identity_scalar_compensation",
        "privileged_visual_inputs_consumed": False,
        "t2v_pixels_consumed": False,
        "reasons": list(reasons),
    }
    return {**unsigned, "decision_receipt_digest": object_sha256(unsigned)}


def zero_update_decision_receipt(
    *,
    reasons: Sequence[str],
    coordinate: ValidatedSiblingCoordinate,
    beta: float,
    segment_commitment_digest: str,
    condition_provenance_digests: Mapping[str, str],
    segment_gates: Mapping[str, Mapping[str, Any]],
    conditional_gates: Mapping[str, Mapping[str, Any]],
    segment_advantages: Mapping[str, torch.Tensor],
    conditional_advantages: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Seal the only decision returned when any scientific hard gate fails."""

    return _decision_receipt(
        authorized=False,
        reasons=reasons,
        coordinate=coordinate,
        beta=beta,
        segment_commitment_digest=segment_commitment_digest,
        condition_provenance_digests=condition_provenance_digests,
        segment_gates=segment_gates,
        conditional_gates=conditional_gates,
        segment_advantages=segment_advantages,
        conditional_advantages=conditional_advantages,
    )


@dataclass(frozen=True)
class CAPERSourceConditionalSegmentDPOResult:
    """A graph-connected authorized loss or a non-trainable zero decision."""

    authorized: bool
    loss: Optional[torch.Tensor]
    action_loss: Optional[torch.Tensor]
    conditional_anchor_loss: Optional[torch.Tensor]
    segment_losses: Mapping[str, torch.Tensor]
    conditional_anchor_losses: Mapping[str, torch.Tensor]
    segment_advantages: Mapping[str, torch.Tensor]
    conditional_advantages: Mapping[str, torch.Tensor]
    noisy_states: Mapping[str, torch.Tensor]
    velocity_targets: Mapping[str, torch.Tensor]
    decision_receipt: Mapping[str, Any]

    @property
    def zero_update(self) -> bool:
        return not self.authorized


def source_conditional_segment_dpo(
    sibling_receipts: Mapping[str, Mapping[str, Any]],
    clean_latents: Mapping[str, torch.Tensor],
    official_epsilon: torch.Tensor,
    sigma: torch.Tensor,
    student_predictions: Mapping[str, torch.Tensor],
    reference_predictions: Mapping[str, torch.Tensor],
    segment_commitment: Mapping[str, Any],
    registered_segment_commitment_digest: str,
    condition_provenance_digests: Mapping[str, str],
    *,
    beta: float,
    minimum_reference_visual_margin: float,
) -> CAPERSourceConditionalSegmentDPOResult:
    """Compute a gated, same-state, reference-corrected CAPER loss.

    ``student_predictions`` and ``reference_predictions`` have the exact
    closed keys in :data:`PREDICTION_ROLES`.  ``target`` is the correct-source
    prediction.  ``source_dropped`` and ``wrong_identity`` are predictions at
    the *same target noisy state*; no source or T2V pixels enter this function.
    The segment commitment's embedded digest must match the independently
    registered digest, and three distinct condition-construction provenance
    digests bind the correct/drop/wrong forward conditions in the decision.

    A malformed contract raises.  A valid packet that fails a scientific
    margin/motion/anchor gate returns ``authorized=False`` and ``loss=None``.
    Only a fully conjoined pass exposes a scalar suitable for an external
    training loop.  This function neither creates nor steps an optimizer.
    """

    beta_value = _finite_scalar(beta, label="beta", positive=True)
    reference_floor = _finite_scalar(
        minimum_reference_visual_margin,
        label="minimum_reference_visual_margin",
        positive=True,
    )
    # Any decline relative to the frozen reference is forbidden; this is not
    # a caller-tunable tolerance.
    retention_tolerance = 0.0
    selectors, segment_commitment_digest = validate_segment_commitment(
        segment_commitment,
        registered_segment_commitment_digest=(
            registered_segment_commitment_digest
        ),
    )
    condition_provenance = validate_condition_provenance_digests(
        condition_provenance_digests
    )
    clean, epsilon = _validate_clean_latents(clean_latents, official_epsilon)
    coordinate = validate_sibling_receipts(
        sibling_receipts,
        clean_latents=clean,
        official_epsilon=epsilon,
        sigma=sigma,
    )
    sigma_by_batch = _validate_sigma(
        sigma,
        batch=int(epsilon.shape[0]),
        device=epsilon.device,
        expected_hex=coordinate.exact40_sigma_float32_be_hex,
    )
    student, reference = _validate_predictions(
        student_predictions,
        reference_predictions,
        target=clean["target"],
    )

    # Rectified-flow states and targets.  All sibling branches share the one
    # official Gaussian and physical sigma, while each candidate retains its
    # own clean-latent velocity.
    sigma_view = sigma_by_batch.reshape(
        int(epsilon.shape[0]), *([1] * (epsilon.ndim - 1))
    )
    noisy_states = {
        role: ((1.0 - sigma_view) * clean[role] + sigma_view * epsilon)
        .detach()
        .contiguous()
        for role in SIBLING_ROLES
    }
    velocity_targets = {
        role: (epsilon - clean[role]).detach().contiguous()
        for role in SIBLING_ROLES
    }
    if any(
        value.dtype != torch.float32
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
        for value in (*noisy_states.values(), *velocity_targets.values())
    ):
        raise CAPERSourceConditionalSegmentDPOError(
            "rectified-flow noisy state or velocity target is invalid"
        )

    segment_losses: dict[str, torch.Tensor] = {}
    segment_advantages_graph: dict[str, torch.Tensor] = {}
    segment_gates: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    for phase in PHASE_ORDER:
        selector = selectors[phase]
        rejected_role = selector.rejected_role
        latent_slice = PHASE_LATENT_SLICES[phase]
        loss, current_margin, reference_margin, advantage = (
            _reference_corrected_pair(
                student["target"],
                student[rejected_role],
                reference["target"],
                reference[rejected_role],
                velocity_targets["target"],
                velocity_targets[rejected_role],
                beta=beta_value,
                temporal_slice=latent_slice,
            )
        )
        if phase in ("onset", "transition"):
            target_motion, contrast_motion = _active_segment_motion_energies(
                clean["target"], clean[rejected_role], latent_slice
            )
            minimum_target_motion = _tensor_min(target_motion)
            minimum_contrast_motion = _tensor_min(contrast_motion)
            target_motion_passed = minimum_target_motion >= float(
                selector.minimum_target_motion_energy
            )
            motion_contrast_passed = minimum_contrast_motion >= float(
                selector.minimum_motion_contrast_energy
            )
            if not target_motion_passed:
                reasons.append(f"minimum_target_motion_failed:{phase}")
            if not motion_contrast_passed:
                reasons.append(f"minimum_motion_contrast_failed:{phase}")
            dynamics_passed = target_motion_passed and motion_contrast_passed
            dynamics_gate = {
                "dynamics_gate_type": (
                    "minimum_target_motion_and_motion_contrast"
                ),
                "minimum_target_motion_energy": float(
                    selector.minimum_target_motion_energy
                ),
                "minimum_motion_contrast_energy": float(
                    selector.minimum_motion_contrast_energy
                ),
                "minimum_observed_target_motion_energy": (
                    minimum_target_motion
                ),
                "maximum_observed_target_motion_energy": _tensor_max(
                    target_motion
                ),
                "minimum_observed_target_vs_rejected_motion_energy": (
                    minimum_contrast_motion
                ),
                "target_motion_passed": target_motion_passed,
                "motion_contrast_passed": motion_contrast_passed,
                "maximum_target_temporal_drift": None,
                "minimum_terminal_state_separation": None,
                "maximum_observed_target_temporal_drift": None,
                "minimum_observed_terminal_state_separation": None,
            }
        else:
            target_drift, terminal_separation = _completion_hold_diagnostics(
                clean["target"], clean[rejected_role], latent_slice
            )
            maximum_target_drift = _tensor_max(target_drift)
            minimum_terminal_separation = _tensor_min(terminal_separation)
            temporal_stability_passed = maximum_target_drift <= float(
                selector.maximum_target_temporal_drift
            )
            terminal_separation_passed = minimum_terminal_separation >= float(
                selector.minimum_terminal_state_separation
            )
            if not temporal_stability_passed:
                reasons.append("completion_temporal_drift_exceeded")
            if not terminal_separation_passed:
                reasons.append("completion_terminal_separation_failed")
            dynamics_passed = (
                temporal_stability_passed and terminal_separation_passed
            )
            dynamics_gate = {
                "dynamics_gate_type": (
                    "maximum_temporal_drift_and_minimum_terminal_separation"
                ),
                "minimum_target_motion_energy": None,
                "minimum_motion_contrast_energy": None,
                "minimum_observed_target_motion_energy": None,
                "maximum_observed_target_motion_energy": None,
                "minimum_observed_target_vs_rejected_motion_energy": None,
                "target_motion_passed": None,
                "motion_contrast_passed": None,
                "maximum_target_temporal_drift": float(
                    selector.maximum_target_temporal_drift
                ),
                "minimum_terminal_state_separation": float(
                    selector.minimum_terminal_state_separation
                ),
                "maximum_observed_target_temporal_drift": maximum_target_drift,
                "minimum_observed_terminal_state_separation": (
                    minimum_terminal_separation
                ),
                "temporal_stability_passed": temporal_stability_passed,
                "terminal_state_separation_passed": (
                    terminal_separation_passed
                ),
            }
        preference_passed = selector.preference_passed
        if not preference_passed:
            reasons.append(f"segment_selector_margin_failed:{phase}")
        segment_losses[phase] = loss
        segment_advantages_graph[phase] = advantage
        segment_gates[phase] = {
            "rgb_frame_range": list(PHASE_FRAME_RANGES[phase]),
            "latent_slice": list(latent_slice),
            "rejected_role": rejected_role,
            "selector_margin": float(selector.margin),
            "selector_uncertainty": float(selector.uncertainty),
            "selector_lower_confidence_margin": (
                selector.lower_confidence_margin
            ),
            "minimum_selector_margin": float(selector.minimum_margin),
            "selector_passed": preference_passed,
            **dynamics_gate,
            "dynamics_passed": dynamics_passed,
            "all_samples_required": True,
            "current_margin_min": _tensor_min(current_margin),
            "reference_margin_min": _tensor_min(reference_margin),
        }

    # Correct-source is the target prediction itself.  Both intervention
    # predictions use its true velocity, so this is a conditional comparison,
    # not a comparison against source-dropped/wrong-identity pixels.
    correct_target = velocity_targets["target"]
    conditional_losses: dict[str, torch.Tensor] = {}
    conditional_advantages_graph: dict[str, torch.Tensor] = {}
    conditional_gates: dict[str, dict[str, Any]] = {}
    for role in CONDITIONAL_NEGATIVE_ROLES:
        loss, current_margin, reference_margin, advantage = (
            _reference_corrected_pair(
                student["target"],
                student[role],
                reference["target"],
                reference[role],
                correct_target,
                correct_target,
                beta=beta_value,
            )
        )
        reference_passed = bool(
            (reference_margin.detach() >= reference_floor).all().item()
        )
        retention = current_margin - reference_margin
        retention_passed = bool(
            (retention.detach() >= -retention_tolerance).all().item()
        )
        if not reference_passed:
            reasons.append(f"frozen_reference_visual_margin_failed:{role}")
        if not retention_passed:
            reasons.append(
                f"conditional_visual_margin_below_frozen_reference:{role}"
            )
        conditional_losses[role] = loss
        conditional_advantages_graph[role] = advantage
        conditional_gates[role] = {
            "correct_condition": "target_correct_source",
            "intervention_condition": role,
            "minimum_reference_visual_margin": reference_floor,
            "visual_margin_tolerance": retention_tolerance,
            "minimum_observed_current_margin": _tensor_min(current_margin),
            "minimum_observed_reference_margin": _tensor_min(reference_margin),
            "minimum_current_minus_reference_margin": _tensor_min(retention),
            "reference_margin_passed": reference_passed,
            "reference_margin_retained": retention_passed,
            "all_samples_required": True,
        }

    segment_advantages = {
        phase: value.detach().contiguous()
        for phase, value in segment_advantages_graph.items()
    }
    conditional_advantages = {
        role: value.detach().contiguous()
        for role, value in conditional_advantages_graph.items()
    }
    if reasons:
        receipt = zero_update_decision_receipt(
            reasons=tuple(reasons),
            coordinate=coordinate,
            beta=beta_value,
            segment_commitment_digest=segment_commitment_digest,
            condition_provenance_digests=condition_provenance,
            segment_gates=segment_gates,
            conditional_gates=conditional_gates,
            segment_advantages=segment_advantages,
            conditional_advantages=conditional_advantages,
        )
        return CAPERSourceConditionalSegmentDPOResult(
            authorized=False,
            loss=None,
            action_loss=None,
            conditional_anchor_loss=None,
            segment_losses={},
            conditional_anchor_losses={},
            segment_advantages=segment_advantages,
            conditional_advantages=conditional_advantages,
            noisy_states=noisy_states,
            velocity_targets=velocity_targets,
            decision_receipt=receipt,
        )

    # Worst-constraint composition is intentional.  It is not an arbitrary
    # weighted scalar sum: improving an already-easy action or identity term
    # cannot compensate for the currently worst constraint.
    action_loss = torch.stack(
        [segment_losses[phase] for phase in PHASE_ORDER]
    ).amax()
    conditional_anchor_loss = torch.stack(
        [conditional_losses[role] for role in CONDITIONAL_NEGATIVE_ROLES]
    ).amax()
    total_loss = torch.maximum(action_loss, conditional_anchor_loss)
    if (
        total_loss.dtype != torch.float32
        or total_loss.ndim != 0
        or not total_loss.requires_grad
        or not bool(torch.isfinite(total_loss).item())
    ):
        raise CAPERSourceConditionalSegmentDPOError(
            "authorized total loss is non-finite, detached, or malformed"
        )
    receipt = _decision_receipt(
        authorized=True,
        reasons=(),
        coordinate=coordinate,
        beta=beta_value,
        segment_commitment_digest=segment_commitment_digest,
        condition_provenance_digests=condition_provenance,
        segment_gates=segment_gates,
        conditional_gates=conditional_gates,
        segment_advantages=segment_advantages,
        conditional_advantages=conditional_advantages,
    )
    return CAPERSourceConditionalSegmentDPOResult(
        authorized=True,
        loss=total_loss,
        action_loss=action_loss,
        conditional_anchor_loss=conditional_anchor_loss,
        segment_losses=dict(segment_losses),
        conditional_anchor_losses=dict(conditional_losses),
        segment_advantages=segment_advantages,
        conditional_advantages=conditional_advantages,
        noisy_states=noisy_states,
        velocity_targets=velocity_targets,
        decision_receipt=receipt,
    )


# Descriptive aliases keep the public entry point easy to discover without
# adding alternate implementations or changing its closed tensor signature.
reference_corrected_source_conditional_segment_dpo = (
    source_conditional_segment_dpo
)
compute_caper_source_conditional_segment_dpo = source_conditional_segment_dpo


def contract_receipt() -> dict[str, Any]:
    """Return a digest-bound declaration of the closed mathematical API."""

    signature = set(inspect.signature(source_conditional_segment_dpo).parameters)
    if not signature.isdisjoint(FORBIDDEN_PUBLIC_INPUT_NAMES):
        raise CAPERSourceConditionalSegmentDPOError(
            "public loss signature exposes a forbidden visual input"
        )
    phase_vectors = {
        phase: [
            1 if PHASE_LATENT_SLICES[phase][0] <= index < PHASE_LATENT_SLICES[phase][1]
            else 0
            for index in range(LATENT_PHASES)
        ]
        for phase in PHASE_ORDER
    }
    if [
        sum(phase_vectors[phase][index] for phase in PHASE_ORDER)
        for index in range(LATENT_PHASES)
    ] != [1] * LATENT_PHASES:
        raise CAPERSourceConditionalSegmentDPOError(
            "latent phase selectors are not an exact partition"
        )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "frame_count": FRAME_COUNT,
        "latent_channels": LATENT_CHANNELS,
        "latent_phases": LATENT_PHASES,
        "required_sibling_roles": list(SIBLING_ROLES),
        "prediction_roles": list(PREDICTION_ROLES),
        "same_state_coordinates": [
            "source",
            "seed",
            "checkpoint",
            "inference_contract",
            "official_gaussian",
            "exact40",
        ],
        "sibling_tensor_binding": (
            "each_receipt_hashes_its_exact_clean_latent_and_the_shared_"
            "official_gaussian"
        ),
        "cross_seed_pair_allowed": False,
        "exact40_schedule_sha256": PINNED_EXACT40_SCHEDULE_SHA256,
        "phase_order": list(PHASE_ORDER),
        "rgb_frame_ranges": {
            phase: list(PHASE_FRAME_RANGES[phase]) for phase in PHASE_ORDER
        },
        "latent_slices": {
            phase: list(PHASE_LATENT_SLICES[phase]) for phase in PHASE_ORDER
        },
        "latent_selector_vectors": phase_vectors,
        "default_rejected_role_by_phase": {
            phase: PHASE_REJECTED_ROLES[phase] for phase in PHASE_ORDER
        },
        "selector_role_policy": (
            "commit_one_exact_permutation_of_noop_incomplete_"
            "phase_order_violation_over_three_phases"
        ),
        "flow_state": "x_sigma=(1-sigma)*clean+sigma*official_epsilon",
        "velocity_target": "official_epsilon-clean",
        "segment_advantage": (
            "(student_rejected_mse-student_target_mse)-"
            "(reference_rejected_mse-reference_target_mse)"
        ),
        "segment_loss": "softplus(-beta*segment_advantage)",
        "active_segment_gate": (
            "onset_transition_each_require_all_sample_minimum_target_motion_"
            "and_minimum_motion_contrast"
        ),
        "completion_segment_gate": (
            "all_samples_require_maximum_target_temporal_drift_at_or_below_"
            "ceiling_and_terminal_state_separation_at_or_above_floor"
        ),
        "phase_specific_selector_thresholds": {
            "onset_transition": [
                "minimum_target_motion_energy",
                "minimum_motion_contrast_energy",
            ],
            "completion": [
                "maximum_target_temporal_drift",
                "minimum_terminal_state_separation",
            ],
            "inactive_threshold_fields_must_be_null": True,
        },
        "segment_compensation_allowed": False,
        "segment_commitment": "embedded_digest_must_equal_external_registration_digest",
        "conditional_anchor_roles": ["target_correct_source", *CONDITIONAL_NEGATIVE_ROLES],
        "conditional_provenance": "distinct_correct_drop_wrong_condition_digests_required",
        "conditional_anchor": "reference_corrected_correct_source_vs_each_visual_intervention",
        "conditional_margin_regression_allowed": False,
        "action_loss": "maximum_of_three_segment_losses_after_all_segment_gates",
        "conditional_anchor_loss": (
            "maximum_of_source_dropped_and_wrong_identity_losses_after_"
            "both_anchor_gates"
        ),
        "total_loss": "maximum_of_action_loss_and_conditional_anchor_loss",
        "action_identity_scalar_compensation_allowed": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "privileged_visual_inputs_consumed": False,
        "t2v_pixels_consumed": False,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


__all__ = [
    "CAPERConditionalDPOError",
    "CAPERContractError",
    "CAPERSourceConditionalSegmentDPOError",
    "CAPERSourceConditionalSegmentDPOResult",
    "CONDITION_PROVENANCE_ROLES",
    "CONDITIONAL_NEGATIVE_ROLES",
    "DECISION_RECEIPT_SCHEMA_VERSION",
    "EXACT40_STEP_COUNT",
    "FORBIDDEN_PUBLIC_INPUT_NAMES",
    "FRAME_COUNT",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "NATIVE_EXACT40_SIGMAS",
    "NATIVE_EXACT40_SIGMA_FLOAT32_BE_HEX",
    "NATIVE_EXACT40_TIMESTEPS",
    "PHASE_FRAME_RANGES",
    "PHASE_LATENT_SLICES",
    "PHASE_ORDER",
    "PHASE_REJECTED_ROLES",
    "PINNED_EXACT40_SCHEDULE_SHA256",
    "PREDICTION_ROLES",
    "SCHEMA_VERSION",
    "SEGMENT_COMMITMENT_SCHEMA_VERSION",
    "SIBLING_RECEIPT_SCHEMA_VERSION",
    "SIBLING_ROLES",
    "SegmentSelector",
    "ValidatedSiblingCoordinate",
    "canonical_json_bytes",
    "compute_caper_source_conditional_segment_dpo",
    "contract_receipt",
    "exact40_coordinate",
    "exact40_sigma_tensor",
    "make_sibling_receipt",
    "make_segment_commitment",
    "object_sha256",
    "reference_corrected_source_conditional_segment_dpo",
    "source_conditional_segment_dpo",
    "tensor_sha256",
    "validate_segment_selectors",
    "validate_segment_commitment",
    "validate_condition_provenance_digests",
    "validate_sibling_receipt",
    "validate_sibling_receipts",
    "zero_update_decision_receipt",
]
