#!/usr/bin/env python3
"""FACT-Plan: sealed target-only temporal-plan research primitives.

This module provides three closed, CPU-testable contracts:

* an authenticated-provenance wrapper for a Bernini target pack
  ``[B,21,P,1536]``;
* an action-only provenance wrapper for detached temporal-plan slots
  ``[B,4,P,32]``; and
* a factory-only hard gate that owns and replays pure-T action-energy and
  correct/wrong-source ``V x I`` factorial score math.

The adapter accepts only those sealed objects.  Every call re-hashes their live
tensors, re-computes every provenance digest, replays the gate from its owned
scores and thresholds, and verifies cross-object query/checkpoint/source/action
bindings.  A bare same-shaped tensor or a publicly hand-built all-true gate is
therefore rejected.

The four temporal carriers have disjoint support and individually zero
temporal mean.  ``U`` is the only parameter, has no bias, is initialized to
exact zero, and is norm-capped independently for each segment and patch.  The
low-noise and evidence-off routes return the exact input tensor object without
evaluating ``U``.

This is still a non-authoritative research primitive.  It binds digests from
an upstream authentication receipt; it does not verify a signature, load or
run Bernini, construct an optimizer, call backward, or establish decoded
action/identity quality.  It consumes no mask, track, pose, flow, detector, or
visual localization annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping

import torch
from torch import nn


METHOD = "bernini-factorial-target-temporal-plan-adapter-v1"
SCHEMA_VERSION = "bernini-fact-plan-research-primitive-v2"
TARGET_PACK_SCHEMA_VERSION = "bernini-fact-plan-authenticated-target-pack-v1"
PLAN_PACK_SCHEMA_VERSION = "bernini-fact-plan-authenticated-action-plan-pack-v1"
GATE_EVIDENCE_SCHEMA_VERSION = "bernini-fact-plan-gate-evidence-v1"
GATE_SCHEMA_VERSION = "bernini-fact-plan-detached-hard-gate-v2"

LATENT_PHASES = 21
HIDDEN_SIZE = 1536
PLAN_SEGMENTS = 4
PLAN_WIDTH = 32

SEGMENT_NAMES = ("onset", "transition", "contact", "terminal_hold")
SEGMENT_RANGES = ((0, 5), (5, 10), (10, 15), (15, 21))

TARGET_SLICE_SEMANTICS = (
    "global-unpadded-bernini-target-suffix-after-authenticated-source-prefix-removal-v1"
)
PLAN_TENSOR_SEMANTICS = (
    "target-only-spatial-patch-temporal-action-plan-no-owner-layout-v1"
)

HIGH_SIGMA_MIN = 0.55
LOW_SIGMA_CUTOFF = 0.25
HIGH_SIGMA_WEIGHT = 1.0
MID_SIGMA_WEIGHT = 0.5
LOW_SIGMA_WEIGHT = 0.0

PURE_T_MARGIN_NAMES = (
    "noop_minus_action_energy",
    "reverse_minus_action_energy",
    "incomplete_minus_action_energy",
)
SOURCE_MARGIN_NAMES = (
    "correct_minus_wrong_joint_score",
    "correct_minus_wrong_video_main_effect",
    "correct_minus_wrong_image_main_effect",
    "correct_minus_wrong_factorial_interaction",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TARGET_PACK_TOKEN = object()
_PLAN_PACK_TOKEN = object()
_GATE_DECISION_TOKEN = object()


class FactPlanError(RuntimeError):
    """A closed FACT-Plan architecture, provenance, or gate was violated."""


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
        raise FactPlanError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FactPlanError(f"{label} must be lowercase SHA-256")
    return value


def _finite_nonnegative_real(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FactPlanError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise FactPlanError(f"{label} must be finite and nonnegative")
    return result


def _finite_positive_real(value: Any, *, label: str) -> float:
    result = _finite_nonnegative_real(value, label=label)
    if result <= 0.0:
        raise FactPlanError(f"{label} must be strictly positive")
    return result


def sigma_gate(sigma: Any) -> tuple[str, float]:
    """Return the fixed high/mid/low amplitude gate; low is direct exact-off."""

    if isinstance(sigma, bool) or not isinstance(sigma, Real):
        raise FactPlanError("sigma must be a real scalar in [0,1]")
    value = float(sigma)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise FactPlanError("sigma must be finite and lie in [0,1]")
    if value >= HIGH_SIGMA_MIN:
        return "high", HIGH_SIGMA_WEIGHT
    if value >= LOW_SIGMA_CUTOFF:
        return "mid", MID_SIGMA_WEIGHT
    return "low_exact_base", LOW_SIGMA_WEIGHT


def tensor_sha256(value: Any, *, label: str) -> str:
    """Hash exact tensor dtype, shape, and raw bytes, including BF16 safely."""

    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() <= 0
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise FactPlanError(f"{label} must be a finite dense non-meta tensor")
    owned = value.detach().to(device="cpu").contiguous().clone()
    raw = owned.view(torch.uint8).numpy().tobytes(order="C")
    header = canonical_json_bytes(
        {"dtype": str(owned.dtype), "shape": list(map(int, owned.shape))}
    )
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


def build_fixed_segment_basis() -> torch.Tensor:
    """Return the fixed disjoint ``[21,4]`` zero-DC temporal carriers."""

    basis = torch.zeros(LATENT_PHASES, PLAN_SEGMENTS, dtype=torch.float64)
    covered: list[int] = []
    for segment, (begin, stop) in enumerate(SEGMENT_RANGES):
        if not 0 <= begin < stop <= LATENT_PHASES:
            raise RuntimeError("FACT-Plan segment range is invalid")
        phases = torch.arange(stop - begin, dtype=torch.float64)
        ramp = phases - (float(stop - begin - 1) / 2.0)
        norm = torch.linalg.vector_norm(ramp)
        if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
            raise RuntimeError("FACT-Plan segment carrier is degenerate")
        basis[begin:stop, segment] = ramp / norm
        covered.extend(range(begin, stop))
    result = basis.to(dtype=torch.float32).contiguous()
    if covered != list(range(LATENT_PHASES)):
        raise RuntimeError("FACT-Plan ranges do not partition all phases")
    if not bool(torch.all(result.ne(0).sum(dim=1) <= 1).item()):
        raise RuntimeError("FACT-Plan segment carriers overlap")
    if not torch.allclose(
        result.sum(dim=0), torch.zeros(PLAN_SEGMENTS), rtol=0.0, atol=1.0e-7
    ):
        raise RuntimeError("FACT-Plan segment carrier has temporal DC")
    return result


@dataclass(frozen=True)
class TargetPackProvenance:
    schema_version: str
    shape: tuple[int, int, int, int]
    dtype: str
    tensor_sha256: str
    slice_semantics: str
    target_suffix_only: bool
    source_prefix_rows_included: bool
    padding_rows_included: bool
    authentication_receipt_sha256: str
    checkpoint_digest: str
    query_digest: str
    correct_source_digest: str
    digest: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "tensor_sha256": self.tensor_sha256,
            "slice_semantics": self.slice_semantics,
            "target_suffix_only": self.target_suffix_only,
            "source_prefix_rows_included": self.source_prefix_rows_included,
            "padding_rows_included": self.padding_rows_included,
            "authentication_receipt_sha256": self.authentication_receipt_sha256,
            "checkpoint_digest": self.checkpoint_digest,
            "query_digest": self.query_digest,
            "correct_source_digest": self.correct_source_digest,
        }

    def validate(self) -> None:
        if self.schema_version != TARGET_PACK_SCHEMA_VERSION:
            raise FactPlanError("target-pack provenance schema differs")
        if (
            type(self.shape) is not tuple
            or len(self.shape) != 4
            or any(type(value) is not int or value <= 0 for value in self.shape)
            or self.shape[1] != LATENT_PHASES
            or self.shape[3] != HIDDEN_SIZE
        ):
            raise FactPlanError("target-pack provenance shape differs")
        if self.dtype not in {"torch.float16", "torch.bfloat16", "torch.float32"}:
            raise FactPlanError("target-pack provenance dtype differs")
        _require_sha256(self.tensor_sha256, label="target tensor SHA")
        for name in (
            "authentication_receipt_sha256",
            "checkpoint_digest",
            "query_digest",
            "correct_source_digest",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            self.slice_semantics != TARGET_SLICE_SEMANTICS
            or self.target_suffix_only is not True
            or self.source_prefix_rows_included is not False
            or self.padding_rows_included is not False
        ):
            raise FactPlanError("target-pack slice semantics are not target-only")
        if self.digest != object_sha256(self.payload()):
            raise FactPlanError("target-pack provenance digest differs")


def make_target_pack_provenance(
    target_hidden: torch.Tensor,
    *,
    authentication_receipt_sha256: str,
    checkpoint_digest: str,
    query_digest: str,
    correct_source_digest: str,
) -> TargetPackProvenance:
    if not isinstance(target_hidden, torch.Tensor):
        raise FactPlanError("target_hidden must be a tensor")
    payload = {
        "schema_version": TARGET_PACK_SCHEMA_VERSION,
        "shape": list(map(int, target_hidden.shape)),
        "dtype": str(target_hidden.dtype),
        "tensor_sha256": tensor_sha256(target_hidden, label="target_hidden"),
        "slice_semantics": TARGET_SLICE_SEMANTICS,
        "target_suffix_only": True,
        "source_prefix_rows_included": False,
        "padding_rows_included": False,
        "authentication_receipt_sha256": _require_sha256(
            authentication_receipt_sha256, label="authentication receipt SHA"
        ),
        "checkpoint_digest": _require_sha256(
            checkpoint_digest, label="checkpoint digest"
        ),
        "query_digest": _require_sha256(query_digest, label="query digest"),
        "correct_source_digest": _require_sha256(
            correct_source_digest, label="correct-source digest"
        ),
    }
    provenance = TargetPackProvenance(
        schema_version=payload["schema_version"],
        shape=tuple(payload["shape"]),
        dtype=payload["dtype"],
        tensor_sha256=payload["tensor_sha256"],
        slice_semantics=payload["slice_semantics"],
        target_suffix_only=True,
        source_prefix_rows_included=False,
        padding_rows_included=False,
        authentication_receipt_sha256=payload["authentication_receipt_sha256"],
        checkpoint_digest=payload["checkpoint_digest"],
        query_digest=payload["query_digest"],
        correct_source_digest=payload["correct_source_digest"],
        digest=object_sha256(payload),
    )
    provenance.validate()
    return provenance


class AuthenticatedTargetPack:
    """Factory-sealed live target tensor plus re-computable provenance."""

    __slots__ = ("_tensor", "_provenance", "_seal")

    def __init__(self, *_: Any, **__: Any) -> None:
        raise FactPlanError("AuthenticatedTargetPack must be created by its factory")

    @classmethod
    def _create(
        cls,
        token: object,
        tensor: torch.Tensor,
        provenance: TargetPackProvenance,
    ) -> "AuthenticatedTargetPack":
        if token is not _TARGET_PACK_TOKEN:
            raise FactPlanError("target-pack factory seal differs")
        result = object.__new__(cls)
        object.__setattr__(result, "_tensor", tensor)
        object.__setattr__(result, "_provenance", provenance)
        object.__setattr__(result, "_seal", token)
        return result

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise FactPlanError("authenticated target pack is immutable")

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def provenance(self) -> TargetPackProvenance:
        return self._provenance

    def validate_live(self) -> None:
        if self._seal is not _TARGET_PACK_TOKEN:
            raise FactPlanError("target-pack factory seal differs")
        self._provenance.validate()
        value = self._tensor
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.dtype not in {torch.float16, torch.bfloat16, torch.float32}
            or tuple(map(int, value.shape)) != self._provenance.shape
            or not bool(torch.isfinite(value.detach()).all().item())
        ):
            raise FactPlanError("live target tensor differs from authenticated shape/dtype")
        if str(value.dtype) != self._provenance.dtype:
            raise FactPlanError("live target dtype differs from provenance")
        if tensor_sha256(value, label="live target tensor") != self._provenance.tensor_sha256:
            raise FactPlanError("live target tensor SHA differs from provenance")


def authenticate_target_pack(
    target_hidden: torch.Tensor, provenance: TargetPackProvenance
) -> AuthenticatedTargetPack:
    if not isinstance(provenance, TargetPackProvenance):
        raise FactPlanError("target provenance must be TargetPackProvenance")
    provenance.validate()
    pack = AuthenticatedTargetPack._create(
        _TARGET_PACK_TOKEN, target_hidden, provenance
    )
    pack.validate_live()
    return pack


@dataclass(frozen=True)
class ActionPlanPackProvenance:
    schema_version: str
    shape: tuple[int, int, int, int]
    dtype: str
    tensor_sha256: str
    tensor_semantics: str
    target_rows_only: bool
    action_only: bool
    owner_identity_or_layout_allowed: bool
    materializer_receipt_sha256: str
    checkpoint_digest: str
    query_digest: str
    pure_t_action_evidence_digest: str
    target_pack_provenance_digest: str
    digest: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "tensor_sha256": self.tensor_sha256,
            "tensor_semantics": self.tensor_semantics,
            "target_rows_only": self.target_rows_only,
            "action_only": self.action_only,
            "owner_identity_or_layout_allowed": self.owner_identity_or_layout_allowed,
            "materializer_receipt_sha256": self.materializer_receipt_sha256,
            "checkpoint_digest": self.checkpoint_digest,
            "query_digest": self.query_digest,
            "pure_t_action_evidence_digest": self.pure_t_action_evidence_digest,
            "target_pack_provenance_digest": self.target_pack_provenance_digest,
        }

    def validate(self) -> None:
        if self.schema_version != PLAN_PACK_SCHEMA_VERSION:
            raise FactPlanError("plan-pack provenance schema differs")
        if (
            type(self.shape) is not tuple
            or len(self.shape) != 4
            or any(type(value) is not int or value <= 0 for value in self.shape)
            or self.shape[1] != PLAN_SEGMENTS
            or self.shape[3] != PLAN_WIDTH
            or self.dtype != "torch.float32"
        ):
            raise FactPlanError("plan-pack shape/dtype differs")
        _require_sha256(self.tensor_sha256, label="plan tensor SHA")
        for name in (
            "materializer_receipt_sha256",
            "checkpoint_digest",
            "query_digest",
            "pure_t_action_evidence_digest",
            "target_pack_provenance_digest",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            self.tensor_semantics != PLAN_TENSOR_SEMANTICS
            or self.target_rows_only is not True
            or self.action_only is not True
            or self.owner_identity_or_layout_allowed is not False
        ):
            raise FactPlanError("plan-pack semantics are not target/action-only")
        if self.digest != object_sha256(self.payload()):
            raise FactPlanError("plan-pack provenance digest differs")


def make_action_plan_pack_provenance(
    plan_slots: torch.Tensor,
    *,
    materializer_receipt_sha256: str,
    checkpoint_digest: str,
    query_digest: str,
    pure_t_action_evidence_digest: str,
    target_pack_provenance_digest: str,
) -> ActionPlanPackProvenance:
    if not isinstance(plan_slots, torch.Tensor):
        raise FactPlanError("plan_slots must be a tensor")
    payload = {
        "schema_version": PLAN_PACK_SCHEMA_VERSION,
        "shape": list(map(int, plan_slots.shape)),
        "dtype": str(plan_slots.dtype),
        "tensor_sha256": tensor_sha256(plan_slots, label="plan_slots"),
        "tensor_semantics": PLAN_TENSOR_SEMANTICS,
        "target_rows_only": True,
        "action_only": True,
        "owner_identity_or_layout_allowed": False,
        "materializer_receipt_sha256": _require_sha256(
            materializer_receipt_sha256, label="plan materializer receipt SHA"
        ),
        "checkpoint_digest": _require_sha256(
            checkpoint_digest, label="checkpoint digest"
        ),
        "query_digest": _require_sha256(query_digest, label="query digest"),
        "pure_t_action_evidence_digest": _require_sha256(
            pure_t_action_evidence_digest, label="pure-T action evidence digest"
        ),
        "target_pack_provenance_digest": _require_sha256(
            target_pack_provenance_digest, label="target-pack provenance digest"
        ),
    }
    provenance = ActionPlanPackProvenance(
        schema_version=payload["schema_version"],
        shape=tuple(payload["shape"]),
        dtype=payload["dtype"],
        tensor_sha256=payload["tensor_sha256"],
        tensor_semantics=payload["tensor_semantics"],
        target_rows_only=True,
        action_only=True,
        owner_identity_or_layout_allowed=False,
        materializer_receipt_sha256=payload["materializer_receipt_sha256"],
        checkpoint_digest=payload["checkpoint_digest"],
        query_digest=payload["query_digest"],
        pure_t_action_evidence_digest=payload["pure_t_action_evidence_digest"],
        target_pack_provenance_digest=payload["target_pack_provenance_digest"],
        digest=object_sha256(payload),
    )
    provenance.validate()
    return provenance


class AuthenticatedActionPlanPack:
    """Factory-sealed detached plan tensor plus action-only provenance."""

    __slots__ = ("_tensor", "_provenance", "_seal")

    def __init__(self, *_: Any, **__: Any) -> None:
        raise FactPlanError("AuthenticatedActionPlanPack must be created by its factory")

    @classmethod
    def _create(
        cls,
        token: object,
        tensor: torch.Tensor,
        provenance: ActionPlanPackProvenance,
    ) -> "AuthenticatedActionPlanPack":
        if token is not _PLAN_PACK_TOKEN:
            raise FactPlanError("plan-pack factory seal differs")
        result = object.__new__(cls)
        object.__setattr__(result, "_tensor", tensor)
        object.__setattr__(result, "_provenance", provenance)
        object.__setattr__(result, "_seal", token)
        return result

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise FactPlanError("authenticated plan pack is immutable")

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def provenance(self) -> ActionPlanPackProvenance:
        return self._provenance

    def validate_live(self) -> None:
        if self._seal is not _PLAN_PACK_TOKEN:
            raise FactPlanError("plan-pack factory seal differs")
        self._provenance.validate()
        value = self._tensor
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.dtype != torch.float32
            or tuple(map(int, value.shape)) != self._provenance.shape
            or not value.is_contiguous()
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise FactPlanError("live plan tensor differs from detached FP32 provenance")
        if tensor_sha256(value, label="live plan tensor") != self._provenance.tensor_sha256:
            raise FactPlanError("live plan tensor SHA differs from provenance")


def authenticate_action_plan_pack(
    plan_slots: torch.Tensor, provenance: ActionPlanPackProvenance
) -> AuthenticatedActionPlanPack:
    if not isinstance(provenance, ActionPlanPackProvenance):
        raise FactPlanError("plan provenance must be ActionPlanPackProvenance")
    provenance.validate()
    pack = AuthenticatedActionPlanPack._create(
        _PLAN_PACK_TOKEN, plan_slots, provenance
    )
    pack.validate_live()
    return pack


@dataclass(frozen=True)
class PureTActionEnergies:
    """Lower-is-better pure-T energies for one matched query bank."""

    action: torch.Tensor
    noop: torch.Tensor
    reverse: torch.Tensor
    incomplete: torch.Tensor


@dataclass(frozen=True)
class VXIFactorialSourceScores:
    """Higher-is-better correct/wrong-source tables with shape ``[B,2,2]``."""

    correct_source: torch.Tensor
    wrong_source: torch.Tensor


@dataclass(frozen=True)
class FactPlanGateThresholds:
    pure_t_energy_margin: float = 0.05
    correct_joint_margin: float = 0.05
    video_main_effect_margin: float = 0.0
    image_main_effect_margin: float = 0.0
    factorial_interaction_margin: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "pure_t_energy_margin",
            "correct_joint_margin",
            "video_main_effect_margin",
            "image_main_effect_margin",
            "factorial_interaction_margin",
        ):
            _finite_nonnegative_real(getattr(self, name), label=name)

    def payload(self) -> Mapping[str, Any]:
        return {
            name: float(getattr(self, name)).hex()
            for name in (
                "pure_t_energy_margin",
                "correct_joint_margin",
                "video_main_effect_margin",
                "image_main_effect_margin",
                "factorial_interaction_margin",
            )
        }

    def digest(self) -> str:
        return object_sha256(self.payload())


@dataclass(frozen=True)
class FactPlanGateEvidence:
    schema_version: str
    scorer_receipt_sha256: str
    scorer_source_digest: str
    checkpoint_digest: str
    query_digest: str
    correct_source_digest: str
    pure_t_action_evidence_digest: str
    digest: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scorer_receipt_sha256": self.scorer_receipt_sha256,
            "scorer_source_digest": self.scorer_source_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "query_digest": self.query_digest,
            "correct_source_digest": self.correct_source_digest,
            "pure_t_action_evidence_digest": self.pure_t_action_evidence_digest,
        }

    def validate(self) -> None:
        if self.schema_version != GATE_EVIDENCE_SCHEMA_VERSION:
            raise FactPlanError("gate-evidence schema differs")
        for name in (
            "scorer_receipt_sha256",
            "scorer_source_digest",
            "checkpoint_digest",
            "query_digest",
            "correct_source_digest",
            "pure_t_action_evidence_digest",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.digest != object_sha256(self.payload()):
            raise FactPlanError("gate-evidence digest differs")


def make_gate_evidence(
    *,
    scorer_receipt_sha256: str,
    scorer_source_digest: str,
    checkpoint_digest: str,
    query_digest: str,
    correct_source_digest: str,
    pure_t_action_evidence_digest: str,
) -> FactPlanGateEvidence:
    payload = {
        "schema_version": GATE_EVIDENCE_SCHEMA_VERSION,
        "scorer_receipt_sha256": _require_sha256(
            scorer_receipt_sha256, label="scorer receipt SHA"
        ),
        "scorer_source_digest": _require_sha256(
            scorer_source_digest, label="scorer source digest"
        ),
        "checkpoint_digest": _require_sha256(
            checkpoint_digest, label="checkpoint digest"
        ),
        "query_digest": _require_sha256(query_digest, label="query digest"),
        "correct_source_digest": _require_sha256(
            correct_source_digest, label="correct-source digest"
        ),
        "pure_t_action_evidence_digest": _require_sha256(
            pure_t_action_evidence_digest, label="pure-T action evidence digest"
        ),
    }
    evidence = FactPlanGateEvidence(**payload, digest=object_sha256(payload))
    evidence.validate()
    return evidence


def _owned_score_vector(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not value.is_floating_point()
        or value.ndim != 1
        or value.numel() <= 0
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise FactPlanError(f"{label} must be a finite floating tensor [B]")
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()


def _owned_factorial_table(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not value.is_floating_point()
        or value.ndim != 3
        or tuple(map(int, value.shape[1:])) != (2, 2)
        or int(value.shape[0]) <= 0
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise FactPlanError(f"{label} must be a finite floating tensor [B,2,2]")
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()


def _factorial_terms(table: torch.Tensor) -> tuple[torch.Tensor, ...]:
    dropped = table[:, 0, 0]
    image = table[:, 0, 1]
    video = table[:, 1, 0]
    joint = table[:, 1, 1]
    video_main = 0.5 * ((video + joint) - (dropped + image))
    image_main = 0.5 * ((image + joint) - (dropped + video))
    interaction = joint - video - image + dropped
    return joint, video_main, image_main, interaction


def _replay_gate(
    pure_scores: torch.Tensor,
    correct_scores: torch.Tensor,
    wrong_scores: torch.Tensor,
    thresholds: FactPlanGateThresholds,
) -> tuple[torch.Tensor, ...]:
    action = pure_scores[:, 0]
    pure_margins = torch.stack(
        [pure_scores[:, index] - action for index in (1, 2, 3)], dim=1
    ).contiguous()
    correct_terms = _factorial_terms(correct_scores)
    wrong_terms = _factorial_terms(wrong_scores)
    source_margins = torch.stack(
        [left - right for left, right in zip(correct_terms, wrong_terms)], dim=1
    ).contiguous()
    pure_threshold = torch.full(
        (1, 3), thresholds.pure_t_energy_margin, dtype=torch.float64
    )
    source_threshold = torch.tensor(
        [[
            thresholds.correct_joint_margin,
            thresholds.video_main_effect_margin,
            thresholds.image_main_effect_margin,
            thresholds.factorial_interaction_margin,
        ]],
        dtype=torch.float64,
    )
    pure_pass = torch.all(pure_margins >= pure_threshold, dim=1).contiguous()
    source_pass = torch.all(source_margins >= source_threshold, dim=1).contiguous()
    per_example = (pure_pass & source_pass).contiguous()
    return pure_margins, source_margins, pure_pass, source_pass, per_example


def _score_digest(
    pure_scores: torch.Tensor,
    correct_scores: torch.Tensor,
    wrong_scores: torch.Tensor,
) -> str:
    return object_sha256(
        {
            "pure_t_scores_sha256": tensor_sha256(
                pure_scores, label="owned pure-T scores"
            ),
            "correct_source_scores_sha256": tensor_sha256(
                correct_scores, label="owned correct-source scores"
            ),
            "wrong_source_scores_sha256": tensor_sha256(
                wrong_scores, label="owned wrong-source scores"
            ),
        }
    )


class FactPlanHardGateDecision:
    """Factory-only gate that replays its owned scores during every validation."""

    __slots__ = (
        "_seal",
        "_pure_scores",
        "_correct_scores",
        "_wrong_scores",
        "_thresholds",
        "_evidence",
        "_pure_margins",
        "_source_margins",
        "_pure_pass",
        "_source_pass",
        "_per_example_pass",
        "_score_digest",
        "_threshold_digest",
        "_source_digest",
        "_digest",
    )

    def __init__(self, *_: Any, **__: Any) -> None:
        raise FactPlanError("gate decisions must be created by the scorer factory")

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise FactPlanError("gate decision is immutable")

    @classmethod
    def _create(
        cls,
        token: object,
        *,
        pure_scores: torch.Tensor,
        correct_scores: torch.Tensor,
        wrong_scores: torch.Tensor,
        thresholds: FactPlanGateThresholds,
        evidence: FactPlanGateEvidence,
    ) -> "FactPlanHardGateDecision":
        if token is not _GATE_DECISION_TOKEN:
            raise FactPlanError("gate-decision factory seal differs")
        replay = _replay_gate(pure_scores, correct_scores, wrong_scores, thresholds)
        pure_margins, source_margins, pure_pass, source_pass, per_example = replay
        score_digest = _score_digest(pure_scores, correct_scores, wrong_scores)
        threshold_digest = thresholds.digest()
        source_digest = evidence.digest
        payload = {
            "schema_version": GATE_SCHEMA_VERSION,
            "batch_size": int(pure_scores.shape[0]),
            "score_digest": score_digest,
            "threshold_digest": threshold_digest,
            "source_digest": source_digest,
            "pure_t_margins_sha256": tensor_sha256(
                pure_margins, label="pure-T margins"
            ),
            "source_margins_sha256": tensor_sha256(
                source_margins, label="source margins"
            ),
            "pure_t_pass": pure_pass.tolist(),
            "source_pass": source_pass.tolist(),
            "per_example_pass": per_example.tolist(),
        }
        result = object.__new__(cls)
        for name, value in (
            ("_seal", token),
            ("_pure_scores", pure_scores),
            ("_correct_scores", correct_scores),
            ("_wrong_scores", wrong_scores),
            ("_thresholds", thresholds),
            ("_evidence", evidence),
            ("_pure_margins", pure_margins),
            ("_source_margins", source_margins),
            ("_pure_pass", pure_pass),
            ("_source_pass", source_pass),
            ("_per_example_pass", per_example),
            ("_score_digest", score_digest),
            ("_threshold_digest", threshold_digest),
            ("_source_digest", source_digest),
            ("_digest", object_sha256(payload)),
        ):
            object.__setattr__(result, name, value)
        result.validate()
        return result

    @property
    def batch_size(self) -> int:
        return int(self._pure_scores.shape[0])

    @property
    def pure_t_margins(self) -> torch.Tensor:
        return self._pure_margins.clone()

    @property
    def source_margins(self) -> torch.Tensor:
        return self._source_margins.clone()

    @property
    def pure_t_pass(self) -> torch.Tensor:
        return self._pure_pass.clone()

    @property
    def source_pass(self) -> torch.Tensor:
        return self._source_pass.clone()

    @property
    def per_example_pass(self) -> torch.Tensor:
        return self._per_example_pass.clone()

    @property
    def all_examples_pass(self) -> bool:
        return bool(torch.all(self._per_example_pass).item())

    @property
    def evidence(self) -> FactPlanGateEvidence:
        return self._evidence

    @property
    def score_digest(self) -> str:
        return self._score_digest

    @property
    def threshold_digest(self) -> str:
        return self._threshold_digest

    @property
    def source_digest(self) -> str:
        return self._source_digest

    @property
    def digest(self) -> str:
        return self._digest

    def validate(self) -> None:
        if getattr(self, "_seal", None) is not _GATE_DECISION_TOKEN:
            raise FactPlanError("gate decision lacks its factory seal")
        self._evidence.validate()
        if not isinstance(self._thresholds, FactPlanGateThresholds):
            raise FactPlanError("gate thresholds differ")
        batch = self.batch_size
        score_specs = (
            ("pure scores", self._pure_scores, (batch, 4)),
            ("correct scores", self._correct_scores, (batch, 2, 2)),
            ("wrong scores", self._wrong_scores, (batch, 2, 2)),
        )
        for label, value, shape in score_specs:
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float64
                or tuple(value.shape) != shape
                or not value.is_contiguous()
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).all().item())
            ):
                raise FactPlanError(f"owned {label} changed")
        replay = _replay_gate(
            self._pure_scores,
            self._correct_scores,
            self._wrong_scores,
            self._thresholds,
        )
        stored = (
            self._pure_margins,
            self._source_margins,
            self._pure_pass,
            self._source_pass,
            self._per_example_pass,
        )
        if any(not torch.equal(left, right) for left, right in zip(replay, stored)):
            raise FactPlanError("gate decision differs from replayed score conjunction")
        score_digest = _score_digest(
            self._pure_scores, self._correct_scores, self._wrong_scores
        )
        threshold_digest = self._thresholds.digest()
        source_digest = self._evidence.digest
        if (
            score_digest != self._score_digest
            or threshold_digest != self._threshold_digest
            or source_digest != self._source_digest
        ):
            raise FactPlanError("gate score/threshold/source digest changed")
        payload = {
            "schema_version": GATE_SCHEMA_VERSION,
            "batch_size": batch,
            "score_digest": score_digest,
            "threshold_digest": threshold_digest,
            "source_digest": source_digest,
            "pure_t_margins_sha256": tensor_sha256(
                replay[0], label="replayed pure-T margins"
            ),
            "source_margins_sha256": tensor_sha256(
                replay[1], label="replayed source margins"
            ),
            "pure_t_pass": replay[2].tolist(),
            "source_pass": replay[3].tolist(),
            "per_example_pass": replay[4].tolist(),
        }
        if self._digest != object_sha256(payload):
            raise FactPlanError("gate decision digest differs")

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        return {
            "schema_version": GATE_SCHEMA_VERSION,
            "batch_size": self.batch_size,
            "score_digest": self.score_digest,
            "threshold_digest": self.threshold_digest,
            "source_digest": self.source_digest,
            "decision_digest": self.digest,
            "pure_t_margin_names": list(PURE_T_MARGIN_NAMES),
            "source_margin_names": list(SOURCE_MARGIN_NAMES),
            "pure_t_pass": self._pure_pass.tolist(),
            "source_pass": self._source_pass.tolist(),
            "per_example_pass": self._per_example_pass.tolist(),
            "input_scores_owned_detached_cpu_fp64": True,
            "axis_reduction": "boolean_conjunction_no_scalar_compensation",
            "scientific_authority": False,
            "update_authority": False,
        }


def evaluate_fact_plan_hard_gate(
    pure_t: PureTActionEnergies,
    source: VXIFactorialSourceScores,
    *,
    evidence: FactPlanGateEvidence,
    thresholds: FactPlanGateThresholds = FactPlanGateThresholds(),
) -> FactPlanHardGateDecision:
    """Factory one sealed decision from detached owned score copies."""

    if not isinstance(pure_t, PureTActionEnergies):
        raise FactPlanError("pure_t must be PureTActionEnergies")
    if not isinstance(source, VXIFactorialSourceScores):
        raise FactPlanError("source must be VXIFactorialSourceScores")
    if not isinstance(evidence, FactPlanGateEvidence):
        raise FactPlanError("evidence must be FactPlanGateEvidence")
    evidence.validate()
    if not isinstance(thresholds, FactPlanGateThresholds):
        raise FactPlanError("thresholds must be FactPlanGateThresholds")

    owned_pure = (
        _owned_score_vector(pure_t.action, label="pure-T action energy"),
        _owned_score_vector(pure_t.noop, label="pure-T noop energy"),
        _owned_score_vector(pure_t.reverse, label="pure-T reverse energy"),
        _owned_score_vector(pure_t.incomplete, label="pure-T incomplete energy"),
    )
    batch = int(owned_pure[0].numel())
    if any(int(value.numel()) != batch for value in owned_pure):
        raise FactPlanError("pure-T energy batch sizes differ")
    pure_scores = torch.stack(owned_pure, dim=1).contiguous()
    correct = _owned_factorial_table(
        source.correct_source, label="correct-source VxI scores"
    )
    wrong = _owned_factorial_table(
        source.wrong_source, label="wrong-source VxI scores"
    )
    if int(correct.shape[0]) != batch or tuple(wrong.shape) != tuple(correct.shape):
        raise FactPlanError("pure-T and VxI score batch sizes differ")
    return FactPlanHardGateDecision._create(
        _GATE_DECISION_TOKEN,
        pure_scores=pure_scores,
        correct_scores=correct,
        wrong_scores=wrong,
        thresholds=thresholds,
        evidence=evidence,
    )


class FactorialTargetTemporalPlanAdapter(nn.Module):
    """Zero-init bounded adapter accepting sealed target/plan packs only."""

    def __init__(self, *, max_segment_vector_norm: float = 2.0) -> None:
        super().__init__()
        self.max_segment_vector_norm = _finite_positive_real(
            max_segment_vector_norm, label="max_segment_vector_norm"
        )
        self.U = nn.Parameter(torch.zeros(PLAN_WIDTH, HIDDEN_SIZE, dtype=torch.float32))
        self.register_buffer(
            "temporal_basis", build_fixed_segment_basis(), persistent=True
        )

    def _validate_inputs(
        self,
        target_pack: Any,
        plan_pack: Any,
        decision: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(target_pack, AuthenticatedTargetPack):
            raise FactPlanError("adapter requires an AuthenticatedTargetPack")
        if not isinstance(plan_pack, AuthenticatedActionPlanPack):
            raise FactPlanError("adapter requires an AuthenticatedActionPlanPack")
        if not isinstance(decision, FactPlanHardGateDecision):
            raise FactPlanError("adapter requires a factory-sealed gate decision")
        target_pack.validate_live()
        plan_pack.validate_live()
        decision.validate()
        target = target_pack.tensor
        plan = plan_pack.tensor
        target_provenance = target_pack.provenance
        plan_provenance = plan_pack.provenance
        evidence = decision.evidence
        if (
            int(target.shape[0]) != decision.batch_size
            or int(plan.shape[0]) != decision.batch_size
            or int(target.shape[2]) != int(plan.shape[2])
            or target.device != plan.device
        ):
            raise FactPlanError("target/plan/gate batch, patch, or device binding differs")
        if (
            target_provenance.checkpoint_digest != plan_provenance.checkpoint_digest
            or target_provenance.checkpoint_digest != evidence.checkpoint_digest
            or target_provenance.query_digest != plan_provenance.query_digest
            or target_provenance.query_digest != evidence.query_digest
            or target_provenance.correct_source_digest != evidence.correct_source_digest
            or plan_provenance.pure_t_action_evidence_digest
            != evidence.pure_t_action_evidence_digest
            or plan_provenance.target_pack_provenance_digest
            != target_provenance.digest
        ):
            raise FactPlanError("target/plan/gate provenance cross-binding differs")
        expected_basis = build_fixed_segment_basis().to(device=self.temporal_basis.device)
        if (
            self.temporal_basis.dtype != torch.float32
            or self.temporal_basis.requires_grad
            or self.temporal_basis.device != self.U.device
            or not torch.equal(self.temporal_basis, expected_basis)
        ):
            raise FactPlanError("fixed temporal plan basis changed")
        if (
            self.U.dtype != torch.float32
            or self.U.device != target.device
            or tuple(self.U.shape) != (PLAN_WIDTH, HIDDEN_SIZE)
            or self.U.requires_grad is not True
            or not bool(torch.isfinite(self.U.detach()).all().item())
        ):
            raise FactPlanError("U must remain trainable finite FP32 [32,1536]")
        return target, plan, decision.per_example_pass

    def _delta_from_validated(
        self,
        target: torch.Tensor,
        plan: torch.Tensor,
        passed: torch.Tensor,
        gate_weight: float,
    ) -> torch.Tensor:
        if gate_weight == 0.0 or not bool(torch.any(passed).item()):
            return torch.zeros_like(target)
        coefficients = torch.matmul(plan, self.U)
        norms = torch.linalg.vector_norm(coefficients, dim=-1, keepdim=True)
        coefficients = coefficients * torch.clamp(
            self.max_segment_vector_norm / norms.clamp_min(1.0e-12), max=1.0
        )
        capped_norms = torch.linalg.vector_norm(coefficients.detach(), dim=-1)
        if bool(
            torch.any(capped_norms > self.max_segment_vector_norm + 2.0e-6).item()
        ):
            raise FactPlanError("per-segment vector norm cap failed")
        residual_fp32 = torch.einsum(
            "tk,bkpd->btpd", self.temporal_basis, coefficients
        )
        eligible = passed.to(device=target.device, dtype=torch.float32).reshape(
            -1, 1, 1, 1
        )
        residual_fp32 = residual_fp32 * eligible * float(gate_weight)
        if not torch.allclose(
            residual_fp32.detach().sum(dim=1),
            torch.zeros_like(residual_fp32[:, 0]),
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise FactPlanError("FP32 plan residual acquired temporal DC")
        result = residual_fp32.to(dtype=target.dtype)
        cast_dc = result.detach().float().sum(dim=1)
        cast_tolerance = 2.0e-6 if target.dtype == torch.float32 else 2.0e-3
        if not torch.allclose(
            cast_dc, torch.zeros_like(cast_dc), rtol=0.0, atol=cast_tolerance
        ):
            raise FactPlanError("dtype-cast plan residual acquired temporal DC")
        return result

    def adapter_delta(
        self,
        target_pack: AuthenticatedTargetPack,
        plan_pack: AuthenticatedActionPlanPack,
        *,
        sigma: Real,
        decision: FactPlanHardGateDecision,
    ) -> torch.Tensor:
        target, plan, passed = self._validate_inputs(
            target_pack, plan_pack, decision
        )
        _, gate_weight = sigma_gate(sigma)
        return self._delta_from_validated(target, plan, passed, gate_weight)

    def forward(
        self,
        target_pack: AuthenticatedTargetPack,
        plan_pack: AuthenticatedActionPlanPack,
        *,
        sigma: Real,
        decision: FactPlanHardGateDecision,
    ) -> torch.Tensor:
        target, plan, passed = self._validate_inputs(
            target_pack, plan_pack, decision
        )
        _, gate_weight = sigma_gate(sigma)
        if gate_weight == 0.0 or not bool(torch.any(passed).item()):
            return target
        delta = self._delta_from_validated(target, plan, passed, gate_weight)
        if bool(torch.all(passed).item()):
            return target + delta
        result = target.clone()
        selector = passed.to(device=target.device)
        result[selector] = target[selector] + delta[selector]
        return result

    def receipt(self) -> Mapping[str, Any]:
        named = tuple(self.named_parameters())
        only_u_trainable = (
            tuple(name for name, _ in named) == ("U",)
            and self.U.requires_grad is True
            and all(parameter.requires_grad for _, parameter in named)
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "target_pack_schema": TARGET_PACK_SCHEMA_VERSION,
            "plan_pack_schema": PLAN_PACK_SCHEMA_VERSION,
            "gate_schema": GATE_SCHEMA_VERSION,
            "target_shape": ["B", LATENT_PHASES, "P", HIDDEN_SIZE],
            "plan_shape": ["B", PLAN_SEGMENTS, "P", PLAN_WIDTH],
            "target_and_plan_live_tensor_sha_revalidated_per_call": True,
            "cross_bound_fields": [
                "checkpoint_digest",
                "query_digest",
                "correct_source_digest",
                "pure_t_action_evidence_digest",
                "target_pack_provenance_digest",
            ],
            "segment_names": list(SEGMENT_NAMES),
            "segment_ranges": [list(value) for value in SEGMENT_RANGES],
            "fixed_disjoint_plan_basis": True,
            "each_segment_basis_temporal_dc_zero": True,
            "global_posthoc_centering_used": False,
            "trainable_parameter_names": ["U"],
            "u_requires_grad": self.U.requires_grad is True,
            "only_u_trainable": only_u_trainable,
            "u_zero_at_receipt_time": bool(
                torch.count_nonzero(self.U.detach()).item() == 0
            ),
            "u_bias": False,
            "per_segment_vector_norm_cap": self.max_segment_vector_norm,
            "sigma_gate": {
                "high_min": HIGH_SIGMA_MIN,
                "low_cutoff": LOW_SIGMA_CUTOFF,
                "weights": [HIGH_SIGMA_WEIGHT, MID_SIGMA_WEIGHT, LOW_SIGMA_WEIGHT],
                "low_route": "direct_input_return_without_U_evaluation",
            },
            "score_math": "factory_sealed_replayed_per_axis_hard_conjunction",
            "scalar_score_compensation": False,
            "mask_track_pose_flow_used_by_primitive": False,
            "execution_attestation_provided": False,
            "semantic_action_claim": False,
            "scientific_authority": False,
            "update_authority": False,
        }


__all__ = [
    "ActionPlanPackProvenance",
    "AuthenticatedActionPlanPack",
    "AuthenticatedTargetPack",
    "FactPlanError",
    "FactPlanGateEvidence",
    "FactPlanGateThresholds",
    "FactPlanHardGateDecision",
    "FactorialTargetTemporalPlanAdapter",
    "GATE_EVIDENCE_SCHEMA_VERSION",
    "GATE_SCHEMA_VERSION",
    "HIDDEN_SIZE",
    "LATENT_PHASES",
    "METHOD",
    "PLAN_PACK_SCHEMA_VERSION",
    "PLAN_SEGMENTS",
    "PLAN_TENSOR_SEMANTICS",
    "PLAN_WIDTH",
    "PURE_T_MARGIN_NAMES",
    "PureTActionEnergies",
    "SCHEMA_VERSION",
    "SEGMENT_NAMES",
    "SEGMENT_RANGES",
    "SOURCE_MARGIN_NAMES",
    "TARGET_PACK_SCHEMA_VERSION",
    "TARGET_SLICE_SEMANTICS",
    "TargetPackProvenance",
    "VXIFactorialSourceScores",
    "authenticate_action_plan_pack",
    "authenticate_target_pack",
    "build_fixed_segment_basis",
    "canonical_json_bytes",
    "evaluate_fact_plan_hard_gate",
    "make_action_plan_pack_provenance",
    "make_gate_evidence",
    "make_target_pack_provenance",
    "object_sha256",
    "sigma_gate",
    "tensor_sha256",
]
