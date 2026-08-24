#!/usr/bin/env python3
"""BRAID Stage-B non-authorizing mathematical core.

This module deliberately cannot authenticate generated media, semantic action
scores, or a trainable checkpoint.  No such upstream implementation currently
exists in this repository.  A caller may provide a detached phase tensor only
to exercise the mathematical objective; the resulting packet is explicitly
untrusted and can never authorize checkpoint freeze or a semantic claim.

The deployed plan head reads a module-pinned canonical action registry only.
Its encoder retains explicit field and byte-position axes before a learned
reader, so it is not the permutation-invariant bag-of-bytes mean pool used by
the rejected draft.  The objective combines per-sample phase specificity,
robust family-centroid specificity, and within-family margin dispersion.  A
complete nuisance factorial, directional phase variation, and a substantive
head-gradient canary are mandatory for this math-only preflight.

Until an independent exact81 media materializer and frozen semantic scorer
reopen real media bytes and produce authenticated artifacts, freeze admission
is structurally unavailable: :func:`assert_stage_b_freeze_authorized` always
raises.  This file creates no optimizer, updates no parameter, writes no
checkpoint, and performs no media or filesystem I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as torch_f


METHOD = "bernini-braid-stage-b-math-only-v1"
SCHEMA_VERSION = "bernini-braid-stage-b-math-only-v1"
MATH_POPULATION_SCHEMA_VERSION = "bernini-braid-stage-b-untrusted-math-population-v1"
REGISTRY_BINDING_SCHEMA_VERSION = "bernini-braid-stage-b-pinned-registry-binding-v1"
OBJECTIVE_SCHEMA_VERSION = "bernini-braid-stage-b-nonauthorizing-objective-v1"

ROLE_ORDER = ("action", "noop", "reverse", "incomplete")
ROLE_TO_INDEX = MappingProxyType({role: index for index, role in enumerate(ROLE_ORDER)})
STAGE_NAMES = ("onset", "transition", "completion", "terminal_hold")
PLAN_STAGES = 4
PLAN_WIDTH = 32
PLAN_ELEMENTS = PLAN_STAGES * PLAN_WIDTH
CORE_EVIDENCE_SHAPE_SUFFIX = (len(ROLE_ORDER), PLAN_STAGES, PLAN_WIDTH)

SPLIT_ORDER = (
    "fit",
    "identity_holdout",
    "scene_holdout",
    "seed_holdout",
    "action_family_holdout",
)
NUISANCE_AXES = ("identity", "scene", "seed")
HOLDOUT_AXES = ("identity", "scene", "seed", "action_family")

CANONICAL_ACTION_FIELDS = (
    "verb_lemma",
    "initial_state",
    "transition_verb",
    "terminal_state",
    "temporal_modifier",
)
FIELD_COUNT = len(CANONICAL_ACTION_FIELDS)
MAX_CANONICAL_FIELD_BYTES = 96
FIELD_TOKEN_SLOTS = MAX_CANONICAL_FIELD_BYTES + 2
BYTE_PAD_TOKEN_ID = 0
BYTE_BOS_TOKEN_ID = 1
BYTE_EOS_TOKEN_ID = 2
BYTE_TOKEN_OFFSET = 3
BYTE_TOKEN_VOCAB_SIZE = 259

PINNED_PHASE_MARGIN_BY_STAGE = (0.20, 0.20, 0.20, 0.20)
PINNED_MIN_DIRECTIONAL_VARIANCE = 1.0e-6
PINNED_MIN_COMPONENT_VALUE = 1.0e-8
PINNED_MIN_HEAD_GRADIENT_MAX_ABS = 1.0e-8
PINNED_MIN_ORDER_CANARY_MAX_ABS = 1.0e-8

FORBIDDEN_CANONICAL_TERMS = frozenset(
    {
        "actor",
        "person",
        "man",
        "woman",
        "boy",
        "girl",
        "dog",
        "cat",
        "face",
        "identity",
        "species",
        "breed",
        "color",
        "clothes",
        "clothing",
        "shirt",
        "scene",
        "background",
        "camera",
        "zoom",
        "framing",
        "perspective",
        "left_side",
        "right_side",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    }
)
FORBIDDEN_OWNER_CHANNELS = (
    "rgb",
    "video",
    "image",
    "vae_latent",
    "clean_latent",
    "gaussian",
    "noise",
    "velocity",
    "full_hidden",
    "per_patch_feature",
    "patch_layout",
    "pixel",
    "box",
    "trajectory",
    "text_embedding",
    "source_caption",
)

# Immutable code-owned registry.  Adding an action requires a source revision;
# callers cannot submit per-sample canonical payloads to this module.
PINNED_CANONICAL_ACTION_REGISTRY = (
    (
        "jump",
        (
            ("action", ("jump", "grounded", "rise", "airborne", "hold")),
            ("noop", ("noop", "grounded", "remain", "grounded", "hold")),
            ("reverse", ("land", "airborne", "descend", "grounded", "hold")),
            (
                "incomplete",
                ("jump", "grounded", "begin_rise", "partial_rise", "stop_early"),
            ),
        ),
    ),
    (
        "sit",
        (
            ("action", ("sit", "standing", "lower", "seated", "hold")),
            ("noop", ("noop", "standing", "remain", "standing", "hold")),
            ("reverse", ("stand", "seated", "rise", "standing", "hold")),
            (
                "incomplete",
                ("sit", "standing", "begin_lower", "partial_seated", "stop_early"),
            ),
        ),
    ),
    (
        "turn",
        (
            (
                "action",
                ("turn_right", "facing_forward", "rotate_right", "facing_right", "hold"),
            ),
            (
                "noop",
                ("noop", "facing_forward", "remain", "facing_forward", "hold"),
            ),
            (
                "reverse",
                ("turn_left", "facing_right", "rotate_left", "facing_forward", "hold"),
            ),
            (
                "incomplete",
                (
                    "turn_right",
                    "facing_forward",
                    "begin_rotate",
                    "partial_turn",
                    "stop_early",
                ),
            ),
        ),
    ),
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_CANONICAL_VALUE_RE = re.compile(r"^[a-z0-9][a-z0-9_ -]{0,95}$")
_POPULATION_SEAL = object()
_REGISTRY_SEAL = object()


class BraidStageBError(RuntimeError):
    """A Stage-B mathematical invariant failed."""


class BraidStageBNotAuthorizingError(BraidStageBError):
    """Freeze is unavailable without an independent real-media pipeline."""


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
        raise BraidStageBError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _deep_json_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("ascii"))


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _SAFE_ID_RE.fullmatch(value) is None:
        raise BraidStageBError(f"{label} must be a path-safe identifier")
    return value


def _positive_real(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BraidStageBError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise BraidStageBError(f"{label} must be finite and strictly positive")
    return result


def _exact_sequence(value: Any, *, length: int, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BraidStageBError(f"{label} must be a sequence")
    result = tuple(value)
    if len(result) != length:
        raise BraidStageBError(f"{label} length differs")
    return result


def tensor_sha256(value: torch.Tensor, *, label: str) -> str:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() <= 0
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise BraidStageBError(f"{label} tensor closure differs")
    owned = value.detach().to(device="cpu").contiguous().clone()
    raw = bytes(owned.view(torch.uint8).reshape(-1).tolist())
    header = canonical_json_bytes(
        {
            "dtype": str(owned.dtype),
            "shape": list(map(int, owned.shape)),
            "layout": str(owned.layout),
        }
    )
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


def _module_state_receipt(module: nn.Module, *, component: str) -> dict[str, Any]:
    state = {
        name: {
            "shape": list(map(int, value.shape)),
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value.detach(), label=f"{component}/{name}"),
        }
        for name, value in module.state_dict().items()
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "component": component,
        "state": state,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


def _registry_as_json() -> list[dict[str, Any]]:
    return [
        {
            "action_family_id": family,
            "payload_by_role": {
                role: dict(zip(CANONICAL_ACTION_FIELDS, fields))
                for role, fields in roles
            },
        }
        for family, roles in PINNED_CANONICAL_ACTION_REGISTRY
    ]


PINNED_CANONICAL_REGISTRY_DIGEST = object_sha256(_registry_as_json())


def _registry_lookup() -> dict[str, dict[str, tuple[str, ...]]]:
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for family, roles in PINNED_CANONICAL_ACTION_REGISTRY:
        if family in result or tuple(role for role, _ in roles) != ROLE_ORDER:
            raise BraidStageBError("pinned canonical registry role closure differs")
        result[family] = {}
        for role, fields in roles:
            if len(fields) != FIELD_COUNT:
                raise BraidStageBError("pinned canonical registry field closure differs")
            checked: list[str] = []
            for field_name, value in zip(CANONICAL_ACTION_FIELDS, fields):
                if type(value) is not str or _CANONICAL_VALUE_RE.fullmatch(value) is None:
                    raise BraidStageBError(
                        f"pinned registry {family}/{role}/{field_name} is not canonical"
                    )
                terms = set(value.replace("-", "_").replace(" ", "_").split("_"))
                if terms & FORBIDDEN_CANONICAL_TERMS:
                    raise BraidStageBError(
                        f"pinned registry {family}/{role}/{field_name} leaks nuisance text"
                    )
                checked.append(value)
            result[family][role] = tuple(checked)
        digests = [object_sha256(result[family][role]) for role in ROLE_ORDER]
        if len(set(digests)) != len(digests):
            raise BraidStageBError("pinned registry roles must be distinct")
    action_digests = [object_sha256(result[family]["action"]) for family in result]
    if len(set(action_digests)) != len(action_digests):
        raise BraidStageBError("pinned registry action families must be distinct")
    return result


def _validate_phase_geometry(value: torch.Tensor) -> None:
    if value.ndim != 4 or tuple(map(int, value.shape[1:])) != CORE_EVIDENCE_SHAPE_SUFFIX:
        raise BraidStageBError("math phase tensor must be [B,4,4,32]")
    action = value[:, ROLE_TO_INDEX["action"]]
    noop = value[:, ROLE_TO_INDEX["noop"]]
    reverse = value[:, ROLE_TO_INDEX["reverse"]]
    incomplete = value[:, ROLE_TO_INDEX["incomplete"]]
    if int(torch.count_nonzero(noop).item()) != 0:
        raise BraidStageBError("noop phase evidence must be exact zero")
    if int(torch.count_nonzero(incomplete[:, 2:]).item()) != 0:
        raise BraidStageBError("incomplete phase evidence must omit late stages")
    active = {
        "action": (action, 4),
        "reverse": (reverse, 4),
        "incomplete": (incomplete, 2),
    }
    for role, (tensor, count) in active.items():
        norms = torch.linalg.vector_norm(tensor[:, :count], dim=-1)
        if not bool(torch.all(norms > 1.0e-8).item()):
            raise BraidStageBError(f"{role} phase evidence is degenerate")
        unit = torch_f.normalize(tensor[:, :count], dim=-1)
        for left in range(count):
            for right in range(left + 1, count):
                similarity = (unit[:, left] * unit[:, right]).sum(dim=-1)
                if bool(torch.any(similarity >= 0.9999).item()):
                    raise BraidStageBError(
                        f"{role} lacks same-role wrong-stage distinction"
                    )
    action_unit = torch_f.normalize(action, dim=-1)
    reverse_unit = torch_f.normalize(reverse, dim=-1)
    if bool(torch.any((action_unit * reverse_unit).sum(dim=-1) >= 0.9999).item()):
        raise BraidStageBError("action and reverse phase evidence are aliased")


def _axis_values(
    action_families: Sequence[str],
    identities: Sequence[str],
    scenes: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, tuple[Any, ...]]:
    return {
        "identity": tuple(identities),
        "scene": tuple(scenes),
        "seed": tuple(seeds),
        "action_family": tuple(action_families),
    }


def _validate_crossed_design(
    *,
    sample_ids: tuple[str, ...],
    action_families: tuple[str, ...],
    identities: tuple[str, ...],
    scenes: tuple[str, ...],
    seeds: tuple[int, ...],
    splits: tuple[str, ...],
) -> dict[str, Any]:
    batch = len(sample_ids)
    if batch < 20 or len(set(sample_ids)) != batch:
        raise BraidStageBError("math population needs at least 20 unique samples")
    if set(splits) != set(SPLIT_ORDER):
        raise BraidStageBError("fit and all four holdouts are mandatory")
    fit_indices = [index for index, split in enumerate(splits) if split == "fit"]
    if len(fit_indices) < 16:
        raise BraidStageBError("math fit needs at least sixteen rows")
    axes = _axis_values(action_families, identities, scenes, seeds)
    matched_counterparts: dict[str, list[dict[str, Any]]] = {}
    for axis in HOLDOUT_AXES:
        held = [
            index
            for index, split in enumerate(splits)
            if split == f"{axis}_holdout"
        ]
        fit_values = {axes[axis][index] for index in fit_indices}
        held_values = {axes[axis][index] for index in held}
        if not held or fit_values & held_values:
            raise BraidStageBError(f"{axis} holdout leaks into fit")
        other_axes = tuple(item for item in HOLDOUT_AXES if item != axis)
        rows: list[dict[str, Any]] = []
        for index in held:
            counterparts = [
                fit_index
                for fit_index in fit_indices
                if all(
                    axes[other][fit_index] == axes[other][index]
                    for other in other_axes
                )
            ]
            if not counterparts:
                raise BraidStageBError(
                    f"{axis} holdout is not a strict single-variable intervention"
                )
            rows.append(
                {
                    "held_sample_id": sample_ids[index],
                    "matched_fit_sample_ids": [sample_ids[item] for item in counterparts],
                }
            )
        matched_counterparts[axis] = rows

    fit_families = sorted({action_families[index] for index in fit_indices})
    if len(fit_families) < 2:
        raise BraidStageBError("math fit needs at least two action families")
    factorial: dict[str, dict[str, int]] = {}
    for family in fit_families:
        members = [
            index for index in fit_indices if action_families[index] == family
        ]
        identity_values = sorted({identities[index] for index in members})
        scene_values = sorted({scenes[index] for index in members})
        seed_values = sorted({seeds[index] for index in members})
        if min(len(identity_values), len(scene_values), len(seed_values)) < 2:
            raise BraidStageBError(
                f"fit family {family} lacks multi-valued nuisance axes"
            )
        expected = {
            (identity, scene, seed)
            for identity in identity_values
            for scene in scene_values
            for seed in seed_values
        }
        observed: dict[tuple[str, str, int], int] = {}
        for index in members:
            key = (identities[index], scenes[index], seeds[index])
            observed[key] = observed.get(key, 0) + 1
        if set(observed) != expected or len(set(observed.values())) != 1:
            raise BraidStageBError(
                f"fit family {family} is not a complete balanced nuisance factorial"
            )
        factorial[family] = {
            "identity_cardinality": len(identity_values),
            "scene_cardinality": len(scene_values),
            "seed_cardinality": len(seed_values),
            "cell_count": len(expected),
            "replicates_per_cell": next(iter(observed.values())),
        }
    return {
        "fit_sample_count": len(fit_indices),
        "fit_action_families": fit_families,
        "complete_nuisance_factorial_by_family": factorial,
        "strict_holdout_counterparts": matched_counterparts,
    }


def _validate_directional_variation(
    value: torch.Tensor,
    action_families: Sequence[str],
    splits: Sequence[str],
) -> dict[str, dict[str, list[float]]]:
    fit_indices = [index for index, split in enumerate(splits) if split == "fit"]
    result: dict[str, dict[str, list[float]]] = {}
    for family in sorted({action_families[index] for index in fit_indices}):
        members = torch.tensor(
            [
                index
                for index in fit_indices
                if action_families[index] == family
            ],
            dtype=torch.int64,
            device=value.device,
        )
        result[family] = {}
        for role, count in (("action", 4), ("reverse", 4), ("incomplete", 2)):
            phases = value.index_select(0, members)[:, ROLE_TO_INDEX[role], :count]
            unit = torch_f.normalize(phases, dim=-1)
            variance = (unit - unit.mean(dim=0, keepdim=True)).square().mean(dim=(0, 2))
            values = [float(item) for item in variance.cpu().tolist()]
            if any(item <= PINNED_MIN_DIRECTIONAL_VARIANCE for item in values):
                raise BraidStageBError(
                    f"fit family {family}/{role} has degenerate phase variation"
                )
            result[family][role] = values
    return result


class UntrustedStageBMathPopulation:
    """Detached caller math, explicitly carrying zero semantic authority."""

    __slots__ = (
        "_phase",
        "_sample_ids",
        "_action_families",
        "_identities",
        "_scenes",
        "_seeds",
        "_splits",
        "_receipt",
        "_seal",
        "_locked",
    )

    def __init__(self, *_: Any, **__: Any) -> None:
        raise BraidStageBError("math population requires its non-authorizing factory")

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("math population is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def _create(
        cls,
        seal: object,
        *,
        phase: torch.Tensor,
        sample_ids: tuple[str, ...],
        action_families: tuple[str, ...],
        identities: tuple[str, ...],
        scenes: tuple[str, ...],
        seeds: tuple[int, ...],
        splits: tuple[str, ...],
        receipt: Mapping[str, Any],
    ) -> "UntrustedStageBMathPopulation":
        if seal is not _POPULATION_SEAL:
            raise BraidStageBError("invalid math population factory seal")
        result = object.__new__(cls)
        object.__setattr__(result, "_phase", phase.detach().clone().contiguous())
        object.__setattr__(result, "_sample_ids", sample_ids)
        object.__setattr__(result, "_action_families", action_families)
        object.__setattr__(result, "_identities", identities)
        object.__setattr__(result, "_scenes", scenes)
        object.__setattr__(result, "_seeds", seeds)
        object.__setattr__(result, "_splits", splits)
        object.__setattr__(result, "_receipt", MappingProxyType(_deep_json_copy(receipt)))
        object.__setattr__(result, "_seal", seal)
        object.__setattr__(result, "_locked", True)
        result.assert_live()
        return result

    @property
    def batch_size(self) -> int:
        return int(self._phase.shape[0])

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    @property
    def action_family_ids(self) -> tuple[str, ...]:
        return self._action_families

    @property
    def identity_group_ids(self) -> tuple[str, ...]:
        return self._identities

    @property
    def scene_group_ids(self) -> tuple[str, ...]:
        return self._scenes

    @property
    def sealed_seeds(self) -> tuple[int, ...]:
        return self._seeds

    @property
    def split_by_sample(self) -> tuple[str, ...]:
        return self._splits

    def indices(self, split: str, *, device: torch.device) -> torch.Tensor:
        if split not in SPLIT_ORDER:
            raise BraidStageBError(f"unknown math split {split!r}")
        return torch.tensor(
            [index for index, value in enumerate(self._splits) if value == split],
            dtype=torch.int64,
            device=device,
        )

    def tensor(self, *, device: torch.device) -> torch.Tensor:
        self.assert_live()
        return self._phase.to(device=device).detach().clone().contiguous()

    def assert_live(self) -> None:
        if self._seal is not _POPULATION_SEAL:
            raise BraidStageBError("math population seal changed")
        _validate_phase_geometry(self._phase)
        design = _validate_crossed_design(
            sample_ids=self._sample_ids,
            action_families=self._action_families,
            identities=self._identities,
            scenes=self._scenes,
            seeds=self._seeds,
            splits=self._splits,
        )
        variation = _validate_directional_variation(
            self._phase, self._action_families, self._splits
        )
        receipt = dict(self._receipt)
        digest = receipt.pop("digest", None)
        if (
            receipt.get("tensor_sha256")
            != tensor_sha256(self._phase, label="sealed math population")
            or receipt.get("sample_ids") != list(self._sample_ids)
            or receipt.get("action_family_ids") != list(self._action_families)
            or receipt.get("identity_group_ids") != list(self._identities)
            or receipt.get("scene_group_ids") != list(self._scenes)
            or receipt.get("sealed_seeds") != list(self._seeds)
            or receipt.get("split_by_sample") != list(self._splits)
            or receipt.get("design") != design
            or receipt.get("directional_variance_by_family_role_stage") != variation
            or digest != object_sha256(receipt)
        ):
            raise BraidStageBError("math population live replay differs")

    def receipt(self) -> dict[str, Any]:
        self.assert_live()
        return _deep_json_copy(dict(self._receipt))


def build_untrusted_math_population(
    phase_evidence: torch.Tensor,
    *,
    sample_ids: Sequence[str],
    action_family_ids: Sequence[str],
    identity_group_ids: Sequence[str],
    scene_group_ids: Sequence[str],
    sealed_seeds: Sequence[int],
    split_by_sample: Sequence[str],
) -> UntrustedStageBMathPopulation:
    """Own caller math for objective preflight; confer no evidence authority."""

    if (
        type(phase_evidence) is not torch.Tensor
        or phase_evidence.requires_grad
        or phase_evidence.grad_fn is not None
        or phase_evidence.device.type == "meta"
        or not bool(torch.isfinite(phase_evidence).all().item())
    ):
        raise BraidStageBError("phase evidence must be a finite detached tensor")
    phase = phase_evidence.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()
    _validate_phase_geometry(phase)
    batch = int(phase.shape[0])
    samples = tuple(
        _safe_id(value, label="sample ID")
        for value in _exact_sequence(sample_ids, length=batch, label="sample IDs")
    )
    families = tuple(
        _safe_id(value, label="action-family ID")
        for value in _exact_sequence(
            action_family_ids, length=batch, label="action-family IDs"
        )
    )
    registry = _registry_lookup()
    if any(family not in registry for family in families):
        raise BraidStageBError("math population action family is absent from pinned registry")
    identities = tuple(
        _safe_id(value, label="identity-group ID")
        for value in _exact_sequence(
            identity_group_ids, length=batch, label="identity-group IDs"
        )
    )
    scenes = tuple(
        _safe_id(value, label="scene-group ID")
        for value in _exact_sequence(
            scene_group_ids, length=batch, label="scene-group IDs"
        )
    )
    seeds = _exact_sequence(sealed_seeds, length=batch, label="sealed seeds")
    if any(type(value) is not int or value < 0 for value in seeds):
        raise BraidStageBError("sealed seeds must be nonnegative integers")
    splits = _exact_sequence(split_by_sample, length=batch, label="split labels")
    if any(type(value) is not str or value not in SPLIT_ORDER for value in splits):
        raise BraidStageBError("math split label differs")
    design = _validate_crossed_design(
        sample_ids=samples,
        action_families=families,
        identities=identities,
        scenes=scenes,
        seeds=tuple(seeds),
        splits=tuple(splits),
    )
    variation = _validate_directional_variation(phase, families, splits)
    unsigned = {
        "schema_version": MATH_POPULATION_SCHEMA_VERSION,
        "method": METHOD,
        "shape": list(map(int, phase.shape)),
        "dtype": str(phase.dtype),
        "tensor_sha256": tensor_sha256(phase, label="untrusted math phase tensor"),
        "sample_ids": list(samples),
        "action_family_ids": list(families),
        "identity_group_ids": list(identities),
        "scene_group_ids": list(scenes),
        "sealed_seeds": list(seeds),
        "split_by_sample": list(splits),
        "design": design,
        "directional_variance_by_family_role_stage": variation,
        "source_trust": "caller_supplied_detached_math_only",
        "real_media_bytes_reopened": False,
        "independent_materializer_present": False,
        "independent_semantic_scorer_present": False,
        "semantic_evidence_authenticated": False,
        "freeze_authority": False,
    }
    receipt = {**unsigned, "digest": object_sha256(unsigned)}
    return UntrustedStageBMathPopulation._create(
        _POPULATION_SEAL,
        phase=phase,
        sample_ids=samples,
        action_families=families,
        identities=identities,
        scenes=scenes,
        seeds=tuple(seeds),
        splits=tuple(splits),
        receipt=receipt,
    )


def _encode_payload_fields(fields: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    values = _exact_sequence(fields, length=FIELD_COUNT, label="canonical fields")
    token_ids = torch.full(
        (FIELD_COUNT, FIELD_TOKEN_SLOTS),
        BYTE_PAD_TOKEN_ID,
        dtype=torch.int64,
        device="cpu",
    )
    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    for field_index, value in enumerate(values):
        if type(value) is not str:
            raise BraidStageBError("canonical registry field must be text")
        try:
            payload = value.encode("ascii")
        except UnicodeEncodeError as error:
            raise BraidStageBError("canonical registry field must be ASCII") from error
        if len(payload) > MAX_CANONICAL_FIELD_BYTES:
            raise BraidStageBError("canonical registry field exceeds pinned no-truncation limit")
        sequence = (
            BYTE_BOS_TOKEN_ID,
            *(int(byte) + BYTE_TOKEN_OFFSET for byte in payload),
            BYTE_EOS_TOKEN_ID,
        )
        token_ids[field_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.int64)
        mask[field_index, : len(sequence)] = True
    return token_ids, mask


def _replay_registry_tokens(
    action_families: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    registry = _registry_lookup()
    ids_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    for family in action_families:
        role_ids: list[torch.Tensor] = []
        role_masks: list[torch.Tensor] = []
        for role in ROLE_ORDER:
            ids, mask = _encode_payload_fields(registry[family][role])
            role_ids.append(ids)
            role_masks.append(mask)
        ids_rows.append(torch.stack(role_ids, dim=0))
        mask_rows.append(torch.stack(role_masks, dim=0))
    token_ids = torch.stack(ids_rows, dim=0).contiguous()
    attention_mask = torch.stack(mask_rows, dim=0).contiguous()
    return token_ids, attention_mask


class PinnedCanonicalRegistryBinding:
    """Factory-sealed token replay from the code-owned action registry."""

    __slots__ = (
        "_token_ids",
        "_attention_mask",
        "_population",
        "_receipt",
        "_seal",
        "_locked",
    )

    def __init__(self, *_: Any, **__: Any) -> None:
        raise BraidStageBError("registry binding requires pinned replay")

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("registry binding is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def _create(
        cls,
        seal: object,
        *,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        population: UntrustedStageBMathPopulation,
        receipt: Mapping[str, Any],
    ) -> "PinnedCanonicalRegistryBinding":
        if seal is not _REGISTRY_SEAL:
            raise BraidStageBError("invalid pinned registry seal")
        result = object.__new__(cls)
        object.__setattr__(result, "_token_ids", token_ids.detach().clone())
        object.__setattr__(result, "_attention_mask", attention_mask.detach().clone())
        object.__setattr__(result, "_population", population)
        object.__setattr__(result, "_receipt", MappingProxyType(_deep_json_copy(receipt)))
        object.__setattr__(result, "_seal", seal)
        object.__setattr__(result, "_locked", True)
        result.assert_live()
        return result

    @property
    def batch_size(self) -> int:
        return int(self._token_ids.shape[0])

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._population.sample_ids

    @property
    def action_family_ids(self) -> tuple[str, ...]:
        return self._population.action_family_ids

    def assert_live(self) -> None:
        if self._seal is not _REGISTRY_SEAL:
            raise BraidStageBError("registry binding seal changed")
        self._population.assert_live()
        replayed_ids, replayed_mask = _replay_registry_tokens(
            self._population.action_family_ids
        )
        receipt = dict(self._receipt)
        digest = receipt.pop("digest", None)
        if (
            not torch.equal(self._token_ids, replayed_ids)
            or not torch.equal(self._attention_mask, replayed_mask)
            or receipt.get("population_receipt_digest")
            != self._population.receipt()["digest"]
            or receipt.get("registry_digest") != PINNED_CANONICAL_REGISTRY_DIGEST
            or receipt.get("token_ids_sha256")
            != tensor_sha256(self._token_ids, label="sealed registry token IDs")
            or receipt.get("attention_mask_sha256")
            != tensor_sha256(self._attention_mask, label="sealed registry mask")
            or digest != object_sha256(receipt)
        ):
            raise BraidStageBError("pinned registry binding live replay differs")

    def tensors_for_role(
        self, role: str, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.assert_live()
        if role not in ROLE_TO_INDEX:
            raise BraidStageBError(f"unknown registry role {role!r}")
        index = ROLE_TO_INDEX[role]
        return (
            self._token_ids[:, index].to(device=device).detach().clone(),
            self._attention_mask[:, index].to(device=device).detach().clone(),
        )

    def receipt(self) -> dict[str, Any]:
        self.assert_live()
        return _deep_json_copy(dict(self._receipt))


def bind_pinned_canonical_registry(
    population: UntrustedStageBMathPopulation,
) -> PinnedCanonicalRegistryBinding:
    if type(population) is not UntrustedStageBMathPopulation:
        raise BraidStageBError("registry binding requires exact math population type")
    population.assert_live()
    token_ids, mask = _replay_registry_tokens(population.action_family_ids)
    unsigned = {
        "schema_version": REGISTRY_BINDING_SCHEMA_VERSION,
        "method": METHOD,
        "population_receipt_digest": population.receipt()["digest"],
        "registry_digest": PINNED_CANONICAL_REGISTRY_DIGEST,
        "registry_is_code_owned_and_caller_payloads_are_rejected": True,
        "shape": list(map(int, token_ids.shape)),
        "field_order": list(CANONICAL_ACTION_FIELDS),
        "field_axis_preserved": True,
        "byte_position_axis_preserved": True,
        "truncation_allowed": False,
        "token_ids_sha256": tensor_sha256(token_ids, label="registry token IDs"),
        "attention_mask_sha256": tensor_sha256(mask, label="registry attention mask"),
        "freeze_authority": False,
    }
    receipt = {**unsigned, "digest": object_sha256(unsigned)}
    return PinnedCanonicalRegistryBinding._create(
        _REGISTRY_SEAL,
        token_ids=token_ids,
        attention_mask=mask,
        population=population,
        receipt=receipt,
    )


@dataclass(frozen=True)
class BraidTextToPlanConfig:
    vocabulary_size: int = BYTE_TOKEN_VOCAB_SIZE
    token_width: int = 16
    hidden_width: int = 128
    pad_token_id: int = BYTE_PAD_TOKEN_ID
    field_count: int = FIELD_COUNT
    position_count: int = FIELD_TOKEN_SLOTS

    def validate(self) -> None:
        if (
            type(self.vocabulary_size) is not int
            or type(self.token_width) is not int
            or type(self.hidden_width) is not int
            or type(self.pad_token_id) is not int
            or type(self.field_count) is not int
            or type(self.position_count) is not int
            or self.vocabulary_size != BYTE_TOKEN_VOCAB_SIZE
            or self.pad_token_id != BYTE_PAD_TOKEN_ID
            or self.field_count != FIELD_COUNT
            or self.position_count != FIELD_TOKEN_SLOTS
            or self.token_width <= 0
            or self.hidden_width <= 0
        ):
            raise BraidStageBError("text-to-plan config differs from pinned registry encoder")


class BraidTextToPlanHead(nn.Module):
    """Exact-type, field- and byte-position-sensitive canonical plan head."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("BraidTextToPlanHead is final and cannot be subclassed")

    def __init__(self, config: BraidTextToPlanConfig) -> None:
        super().__init__()
        if type(config) is not BraidTextToPlanConfig:
            raise BraidStageBError("text-to-plan config type differs")
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocabulary_size,
            config.token_width,
            padding_idx=config.pad_token_id,
        )
        self.field_embedding = nn.Embedding(config.field_count, config.token_width)
        self.position_embedding = nn.Embedding(config.position_count, config.token_width)
        ordered_width = config.field_count * config.position_count * config.token_width
        self.reader = nn.Sequential(
            nn.Linear(ordered_width, config.hidden_width, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_width, PLAN_ELEMENTS, bias=False),
        )
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.field_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[config.pad_token_id].zero_()
        for module in self.reader:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def _ordered_features(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        expected = (FIELD_COUNT, FIELD_TOKEN_SLOTS)
        if (
            type(token_ids) is not torch.Tensor
            or type(attention_mask) is not torch.Tensor
            or token_ids.ndim != 3
            or tuple(map(int, token_ids.shape[1:])) != expected
            or tuple(attention_mask.shape) != tuple(token_ids.shape)
            or token_ids.dtype != torch.int64
            or attention_mask.dtype != torch.bool
        ):
            raise BraidStageBError("ordered canonical token geometry differs")
        device = self.token_embedding.weight.device
        token_ids = token_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)
        fields = torch.arange(FIELD_COUNT, device=device).view(1, FIELD_COUNT, 1)
        positions = torch.arange(FIELD_TOKEN_SLOTS, device=device).view(
            1, 1, FIELD_TOKEN_SLOTS
        )
        embedded = (
            self.token_embedding(token_ids)
            + self.field_embedding(fields)
            + self.position_embedding(positions)
        )
        embedded = embedded * attention_mask.to(dtype=embedded.dtype).unsqueeze(-1)
        return embedded.reshape(token_ids.shape[0], -1).contiguous()

    def forward(
        self,
        binding: PinnedCanonicalRegistryBinding,
        *,
        role: str = "action",
    ) -> torch.Tensor:
        if type(self) is not BraidTextToPlanHead:
            raise BraidStageBError("text-to-plan subclasses are forbidden")
        if type(binding) is not PinnedCanonicalRegistryBinding:
            raise BraidStageBError("plan head requires exact pinned registry binding")
        if role not in ROLE_TO_INDEX:
            raise BraidStageBError(f"unknown plan role {role!r}")
        token_ids, attention_mask = binding.tensors_for_role(
            role, device=self.token_embedding.weight.device
        )
        ordered = self._ordered_features(token_ids, attention_mask)
        learned = self.reader(ordered).reshape(-1, PLAN_STAGES, PLAN_WIDTH)
        if role == "noop":
            result = learned * 0.0
        elif role == "incomplete":
            result = torch.cat((learned[:, :2], learned[:, 2:] * 0.0), dim=1)
        else:
            result = learned
        if (
            tuple(result.shape) != (binding.batch_size, PLAN_STAGES, PLAN_WIDTH)
            or not result.requires_grad
            or result.grad_fn is None
            or not bool(torch.isfinite(result).all().item())
            or (role == "noop" and int(torch.count_nonzero(result).item()) != 0)
            or (
                role == "incomplete"
                and int(torch.count_nonzero(result[:, 2:]).item()) != 0
            )
        ):
            raise BraidStageBError("text-to-plan graph or geometry differs")
        return result

    def state_receipt(self) -> dict[str, Any]:
        unsigned = _module_state_receipt(
            self, component="field-position-sensitive-text-to-plan-head"
        )
        unsigned = dict(unsigned)
        unsigned.pop("digest")
        unsigned["config"] = {
            "vocabulary_size": self.config.vocabulary_size,
            "token_width": self.config.token_width,
            "hidden_width": self.config.hidden_width,
            "field_count": self.config.field_count,
            "position_count": self.config.position_count,
            "ordered_reader_input": True,
            "mean_pooling": False,
        }
        return {**unsigned, "digest": object_sha256(unsigned)}


def _anagram_canary_token_grids() -> tuple[torch.Tensor, torch.Tensor]:
    left, left_mask = _encode_payload_fields(
        ("rise", "grounded", "abc", "airborne", "hold")
    )
    right, right_mask = _encode_payload_fields(
        ("rise", "grounded", "cba", "airborne", "hold")
    )
    token_ids = torch.stack((left, right), dim=0)
    masks = torch.stack((left_mask, right_mask), dim=0)
    left_bag = sorted(token_ids[0][masks[0]].tolist())
    right_bag = sorted(token_ids[1][masks[1]].tolist())
    if left_bag != right_bag or torch.equal(token_ids[0], token_ids[1]):
        raise BraidStageBError("internal anagram order canary is malformed")
    return token_ids, masks


def _field_position_order_canary(head: BraidTextToPlanHead) -> torch.Tensor:
    token_ids, mask = _anagram_canary_token_grids()
    features = head._ordered_features(token_ids, mask)
    difference = (features[0] - features[1]).abs().max()
    if float(difference.detach().item()) <= PINNED_MIN_ORDER_CANARY_MAX_ABS:
        raise BraidStageBError("field/position encoder collapses anagram order")
    return difference


@dataclass(frozen=True)
class BraidStageBObjectiveConfig:
    phase_margin_by_stage: tuple[float, float, float, float] = PINNED_PHASE_MARGIN_BY_STAGE
    sample_phase_weight: float = 1.0
    robust_centroid_weight: float = 1.0
    margin_dispersion_weight: float = 0.1
    soft_margin_temperature: float = 5.0

    def validate(self) -> None:
        if self.phase_margin_by_stage != PINNED_PHASE_MARGIN_BY_STAGE:
            raise BraidStageBError("phase margins are pinned")
        for name in (
            "sample_phase_weight",
            "robust_centroid_weight",
            "margin_dispersion_weight",
            "soft_margin_temperature",
        ):
            _positive_real(getattr(self, name), label=name)


def _cosine(plan: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
    if plan.shape != evidence.shape:
        raise BraidStageBError("plan/evidence phase shape differs")
    return torch_f.cosine_similarity(plan.float(), evidence.float(), dim=-1, eps=1.0e-8)


def _phase_margin_tensors(
    plan_by_role: Mapping[str, torch.Tensor],
    evidence: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    active_slots = {"action": range(4), "reverse": range(4), "incomplete": range(2)}
    result: dict[str, torch.Tensor] = {}
    for target_role, slots in active_slots.items():
        target_index = ROLE_TO_INDEX[target_role]
        stage_margins: list[torch.Tensor] = []
        for stage in slots:
            plan_stage = plan_by_role[target_role][:, stage]
            positive = _cosine(plan_stage, evidence[:, target_index, stage])
            negatives = [
                _cosine(plan_stage, evidence[:, ROLE_TO_INDEX[role], stage])
                for role in ROLE_ORDER
                if role != target_role
            ]
            negatives.extend(
                _cosine(plan_stage, evidence[:, target_index, wrong_stage])
                for wrong_stage in slots
                if wrong_stage != stage
            )
            hardest = torch.stack(negatives, dim=1).max(dim=1).values
            stage_margins.append(positive - hardest)
        result[target_role] = torch.stack(stage_margins, dim=1)
    return MappingProxyType(result)


def _active_margin_vector(value: Mapping[str, torch.Tensor]) -> torch.Tensor:
    result = torch.cat(
        (value["action"], value["reverse"], value["incomplete"]), dim=1
    )
    if result.ndim != 2 or result.shape[1] != 10:
        raise BraidStageBError("active phase margin must contain ten slots")
    return result


def _within_family_robust_center(
    value: torch.Tensor, families: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.shape[0] != len(families):
        raise BraidStageBError("robust centering batch differs")
    medians = {
        family: value[
            torch.tensor(
                [index for index, item in enumerate(families) if item == family],
                dtype=torch.int64,
                device=value.device,
            )
        ].median(dim=0).values
        for family in sorted(set(families))
    }
    centered = torch.stack(
        [value[index] - medians[family] for index, family in enumerate(families)], dim=0
    )
    broadcast = torch.stack([medians[family] for family in families], dim=0)
    return centered, broadcast


def _soft_phase_loss(
    margins: Mapping[str, torch.Tensor], config: BraidStageBObjectiveConfig
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    temperature = config.soft_margin_temperature
    for values in margins.values():
        for stage in range(int(values.shape[1])):
            gap = config.phase_margin_by_stage[stage] - values[:, stage]
            terms.append(torch_f.softplus(temperature * gap).mean() / temperature)
    if len(terms) != 10:
        raise BraidStageBError("soft phase objective requires ten slots")
    return torch.stack(terms).mean()


def _objective_components(
    head: BraidTextToPlanHead,
    population: UntrustedStageBMathPopulation,
    binding: PinnedCanonicalRegistryBinding,
    config: BraidStageBObjectiveConfig,
) -> dict[str, Any]:
    plan_by_role = {role: head(binding, role=role) for role in ROLE_ORDER}
    for family in sorted(set(binding.action_family_ids)):
        members = [
            index
            for index, value in enumerate(binding.action_family_ids)
            if value == family
        ]
        for role in ROLE_ORDER:
            reference = plan_by_role[role][members[0]]
            if any(
                not torch.equal(reference, plan_by_role[role][index])
                for index in members[1:]
            ):
                raise BraidStageBError("same registry family produced sample-dependent plan")
    device = next(head.parameters()).device
    evidence = population.tensor(device=device)
    phase_margins = _phase_margin_tensors(plan_by_role, evidence)
    fit_indices = population.indices("fit", device=device)
    fit_families = tuple(
        family
        for family, split in zip(population.action_family_ids, population.split_by_sample)
        if split == "fit"
    )
    fit_margins = MappingProxyType(
        {
            role: value.index_select(0, fit_indices)
            for role, value in phase_margins.items()
        }
    )
    sample_phase_loss = _soft_phase_loss(fit_margins, config)
    fit_evidence = evidence.index_select(0, fit_indices)
    _, centroid_evidence = _within_family_robust_center(fit_evidence, fit_families)
    fit_plan = MappingProxyType(
        {
            role: value.index_select(0, fit_indices)
            for role, value in plan_by_role.items()
        }
    )
    centroid_margins = _phase_margin_tensors(fit_plan, centroid_evidence)
    robust_centroid_loss = _soft_phase_loss(centroid_margins, config)
    fit_margin_vector = _active_margin_vector(fit_margins)
    centered_margins, _ = _within_family_robust_center(
        fit_margin_vector, fit_families
    )
    margin_dispersion_loss = (
        torch.sqrt(centered_margins.square() + 1.0e-4) - 1.0e-2
    ).mean()
    if any(
        not bool(torch.isfinite(value).item())
        or float(value.detach().item()) <= PINNED_MIN_COMPONENT_VALUE
        for value in (sample_phase_loss, robust_centroid_loss, margin_dispersion_loss)
    ):
        raise BraidStageBError("necessary Stage-B math loss is degenerate")
    optimization_loss = (
        config.sample_phase_weight * sample_phase_loss
        + config.robust_centroid_weight * robust_centroid_loss
        + config.margin_dispersion_weight * margin_dispersion_loss
    )
    order_canary = _field_position_order_canary(head)
    return {
        "plan_by_role": MappingProxyType(plan_by_role),
        "phase_margins": phase_margins,
        "centroid_margins": centroid_margins,
        "fit_margin_vector": fit_margin_vector,
        "fit_families": fit_families,
        "sample_phase_loss": sample_phase_loss,
        "robust_centroid_loss": robust_centroid_loss,
        "margin_dispersion_loss": margin_dispersion_loss,
        "optimization_loss": optimization_loss,
        "order_canary": order_canary,
    }


def _loss_gradient_max_abs(loss: torch.Tensor, head: BraidTextToPlanHead) -> float:
    gradients = torch.autograd.grad(
        loss,
        tuple(head.parameters()),
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    values = [
        float(gradient.detach().abs().max().item())
        for gradient in gradients
        if gradient is not None and gradient.numel() > 0
    ]
    return max(values, default=0.0)


@dataclass(frozen=True)
class BraidStageBMathObjectiveBundle:
    action_plan: torch.Tensor
    plan_by_role: Mapping[str, torch.Tensor]
    optimization_loss: torch.Tensor
    sample_phase_loss: torch.Tensor
    robust_centroid_loss: torch.Tensor
    margin_dispersion_loss: torch.Tensor
    phase_margin_by_role_stage: Mapping[str, torch.Tensor]
    robust_centroid_margin_by_role_stage: Mapping[str, torch.Tensor]
    fit_margin_vector: torch.Tensor
    gradient_max_abs_by_loss: Mapping[str, float]
    field_position_anagram_canary_max_abs: float
    config: BraidStageBObjectiveConfig
    head: BraidTextToPlanHead
    population: UntrustedStageBMathPopulation
    binding: PinnedCanonicalRegistryBinding
    head_state_digest_at_build: str
    population_receipt_digest: str
    binding_receipt_digest: str

    def assert_live(self) -> None:
        if type(self.head) is not BraidTextToPlanHead:
            raise BraidStageBError("objective head exact type changed")
        if self.head.state_receipt()["digest"] != self.head_state_digest_at_build:
            raise BraidStageBError("objective head changed after graph build")
        self.population.assert_live()
        self.binding.assert_live()
        if (
            self.population.receipt()["digest"] != self.population_receipt_digest
            or self.binding.receipt()["digest"] != self.binding_receipt_digest
        ):
            raise BraidStageBError("objective upstream binding changed")
        replayed = _objective_components(
            self.head, self.population, self.binding, self.config
        )
        tensor_pairs = (
            (self.optimization_loss, replayed["optimization_loss"]),
            (self.sample_phase_loss, replayed["sample_phase_loss"]),
            (self.robust_centroid_loss, replayed["robust_centroid_loss"]),
            (self.margin_dispersion_loss, replayed["margin_dispersion_loss"]),
            (self.fit_margin_vector, replayed["fit_margin_vector"]),
        )
        if any(
            not torch.equal(left.detach(), right.detach())
            for left, right in tensor_pairs
        ):
            raise BraidStageBError("objective values differ from live math replay")
        for key, replay_key in (
            ("plan_by_role", "plan_by_role"),
            ("phase_margin_by_role_stage", "phase_margins"),
            ("robust_centroid_margin_by_role_stage", "centroid_margins"),
        ):
            stored = getattr(self, key)
            fresh = replayed[replay_key]
            if set(stored) != set(fresh) or any(
                not torch.equal(stored[role].detach(), fresh[role].detach())
                for role in fresh
            ):
                raise BraidStageBError(f"objective {key} differs from live replay")
        if (
            abs(
                self.field_position_anagram_canary_max_abs
                - float(replayed["order_canary"].detach().item())
            )
            > 0.0
            or any(
                value <= PINNED_MIN_HEAD_GRADIENT_MAX_ABS
                for value in self.gradient_max_abs_by_loss.values()
            )
        ):
            raise BraidStageBError("objective canary receipt differs")

    def receipt(self) -> dict[str, Any]:
        self.assert_live()
        unsigned = {
            "schema_version": OBJECTIVE_SCHEMA_VERSION,
            "method": METHOD,
            "output_action_plan_shape": list(map(int, self.action_plan.shape)),
            "losses": {
                "optimization": float(self.optimization_loss.detach().item()),
                "sample_phase": float(self.sample_phase_loss.detach().item()),
                "robust_centroid": float(self.robust_centroid_loss.detach().item()),
                "margin_dispersion": float(self.margin_dispersion_loss.detach().item()),
            },
            "gradient_max_abs_by_loss": dict(self.gradient_max_abs_by_loss),
            "minimum_required_head_gradient": PINNED_MIN_HEAD_GRADIENT_MAX_ABS,
            "field_position_anagram_canary_max_abs": (
                self.field_position_anagram_canary_max_abs
            ),
            "same_role_wrong_stage_negatives_used": True,
            "complete_nuisance_factorial_required": True,
            "population_receipt_digest": self.population_receipt_digest,
            "binding_receipt_digest": self.binding_receipt_digest,
            "head_state_digest": self.head_state_digest_at_build,
            "caller_math_is_semantic_evidence": False,
            "semantic_action_editing_success_claim": False,
            "hard_gate_available": False,
            "freeze_authority": False,
            "checkpoint_written": False,
            "parameter_update_executed": False,
        }
        return {**unsigned, "digest": object_sha256(unsigned)}


def compute_stage_b_objective(
    head: BraidTextToPlanHead,
    population: UntrustedStageBMathPopulation,
    binding: PinnedCanonicalRegistryBinding,
    *,
    config: BraidStageBObjectiveConfig = BraidStageBObjectiveConfig(),
) -> BraidStageBMathObjectiveBundle:
    """Build and gradient-audit math losses; perform no parameter update."""

    if type(head) is not BraidTextToPlanHead:
        raise BraidStageBError("Stage-B requires exact BraidTextToPlanHead type")
    if type(population) is not UntrustedStageBMathPopulation:
        raise BraidStageBError("Stage-B requires exact untrusted math population type")
    if type(binding) is not PinnedCanonicalRegistryBinding:
        raise BraidStageBError("Stage-B requires exact pinned registry binding type")
    if type(config) is not BraidStageBObjectiveConfig:
        raise BraidStageBError("Stage-B objective config type differs")
    config.validate()
    population.assert_live()
    binding.assert_live()
    if (
        binding.sample_ids != population.sample_ids
        or binding.action_family_ids != population.action_family_ids
    ):
        raise BraidStageBError("registry binding differs from math population")
    components = _objective_components(head, population, binding, config)
    losses = {
        "sample_phase": components["sample_phase_loss"],
        "robust_centroid": components["robust_centroid_loss"],
        "margin_dispersion": components["margin_dispersion_loss"],
    }
    gradient_max_abs = {
        name: _loss_gradient_max_abs(loss, head) for name, loss in losses.items()
    }
    if any(
        value <= PINNED_MIN_HEAD_GRADIENT_MAX_ABS
        for value in gradient_max_abs.values()
    ):
        raise BraidStageBError(
            "necessary Stage-B loss lacks a substantive plan-head gradient"
        )
    bundle = BraidStageBMathObjectiveBundle(
        action_plan=components["plan_by_role"]["action"],
        plan_by_role=components["plan_by_role"],
        optimization_loss=components["optimization_loss"],
        sample_phase_loss=components["sample_phase_loss"],
        robust_centroid_loss=components["robust_centroid_loss"],
        margin_dispersion_loss=components["margin_dispersion_loss"],
        phase_margin_by_role_stage=components["phase_margins"],
        robust_centroid_margin_by_role_stage=components["centroid_margins"],
        fit_margin_vector=components["fit_margin_vector"],
        gradient_max_abs_by_loss=MappingProxyType(gradient_max_abs),
        field_position_anagram_canary_max_abs=float(
            components["order_canary"].detach().item()
        ),
        config=config,
        head=head,
        population=population,
        binding=binding,
        head_state_digest_at_build=head.state_receipt()["digest"],
        population_receipt_digest=population.receipt()["digest"],
        binding_receipt_digest=binding.receipt()["digest"],
    )
    bundle.assert_live()
    return bundle


def assert_stage_b_freeze_authorized(*_: Any, **__: Any) -> None:
    """Always fail: this math-only module has no real-media admission authority."""

    raise BraidStageBNotAuthorizingError(
        "Stage-B freeze is structurally unavailable until an independent exact81 "
        "media materializer and frozen semantic scorer are implemented and audited"
    )


def public_input_contract() -> dict[str, Any]:
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "purpose": "non-authorizing mathematical and gradient preflight only",
        "allowed_inputs": [
            "detached_caller_phase_tensor_explicitly_marked_untrusted",
            "code_owned_pinned_canonical_registry_binding",
        ],
        "raw_mapping_artifact_input": False,
        "caller_media_score_input": False,
        "caller_video_digest_input": False,
        "caller_canonical_payload_input": False,
        "real_media_materializer_present": False,
        "independent_semantic_scorer_present": False,
        "semantic_evidence_authentication_available": False,
        "hard_gate_available": False,
        "freeze_authorization_available": False,
        "freeze_assertion_structurally_raises": True,
        "head_exact_type_required": True,
        "field_axis_preserved": True,
        "byte_position_axis_preserved": True,
        "mean_pooling_used": False,
        "complete_nuisance_factorial_required": True,
        "nondegenerate_directional_phase_variation_required": True,
        "per_loss_head_gradient_canary_required": True,
        "same_role_wrong_stage_negatives_used": True,
        "forbidden_owner_channels": list(FORBIDDEN_OWNER_CHANNELS),
        "optimizer_constructed": False,
        "backward_called": False,
        "autograd_gradient_canary_used": True,
        "parameter_update_executed": False,
        "checkpoint_io": False,
        "media_io": False,
        "required_missing_upstream": [
            "independent_exact81_media_byte_materializer",
            "frozen_action_and_control_scorer",
            "immutable_candidate_checkpoint_reopen_evaluator",
        ],
        "public_signatures": {
            "build_untrusted_math_population": str(
                inspect.signature(build_untrusted_math_population)
            ),
            "bind_pinned_canonical_registry": str(
                inspect.signature(bind_pinned_canonical_registry)
            ),
            "compute_stage_b_objective": str(
                inspect.signature(compute_stage_b_objective)
            ),
            "assert_stage_b_freeze_authorized": str(
                inspect.signature(assert_stage_b_freeze_authorized)
            ),
        },
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


__all__ = [
    "BraidStageBError",
    "BraidStageBMathObjectiveBundle",
    "BraidStageBNotAuthorizingError",
    "BraidStageBObjectiveConfig",
    "BraidTextToPlanConfig",
    "BraidTextToPlanHead",
    "CANONICAL_ACTION_FIELDS",
    "FIELD_COUNT",
    "FIELD_TOKEN_SLOTS",
    "FORBIDDEN_OWNER_CHANNELS",
    "HOLDOUT_AXES",
    "MATH_POPULATION_SCHEMA_VERSION",
    "METHOD",
    "NUISANCE_AXES",
    "OBJECTIVE_SCHEMA_VERSION",
    "PINNED_CANONICAL_ACTION_REGISTRY",
    "PINNED_CANONICAL_REGISTRY_DIGEST",
    "PINNED_MIN_HEAD_GRADIENT_MAX_ABS",
    "PLAN_STAGES",
    "PLAN_WIDTH",
    "PinnedCanonicalRegistryBinding",
    "REGISTRY_BINDING_SCHEMA_VERSION",
    "ROLE_ORDER",
    "SCHEMA_VERSION",
    "SPLIT_ORDER",
    "STAGE_NAMES",
    "UntrustedStageBMathPopulation",
    "assert_stage_b_freeze_authorized",
    "bind_pinned_canonical_registry",
    "build_untrusted_math_population",
    "canonical_json_bytes",
    "compute_stage_b_objective",
    "object_sha256",
    "public_input_contract",
]
