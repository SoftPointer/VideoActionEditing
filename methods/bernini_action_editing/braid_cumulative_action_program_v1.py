#!/usr/bin/env python3
"""BRAID cumulative action-program CPU research primitive.

BRAID represents an action as four spatial-orderless stage codes ``[B,4,32]``
and expands them with a fixed, start-anchored cumulative smooth-step basis.
Every stage rises once and then stays on a plateau.  Temporal centering and
zero-DC carriers are deliberately forbidden because they erase persistent end
states such as "sit and hold".

The plan has no patch axis.  Patch-specific residuals can arise only through
the current source-conditioned target hidden tensor: fixed projections encode
the current hidden and the four plan codes, their rank-32 coordinates interact
multiplicatively, and a zero-initialized stage decoder maps the interaction
back to 1536 channels.  The decoder is the only parameter.  This file does not
construct an optimizer or connect the primitive to Bernini.

Inputs are exact-type CPU tensors copied into locally owned snapshots.  Each
snapshot recomputes a local tensor SHA before use.  These checks prevent the
ordinary caller tensor from aliasing the object used by the primitive and
reject tensor/snapshot subclasses.  They are only local self-consistency, not
signature, checkpoint, source, or semantic authentication.

No mask, track, pose, flow, patch-layout plan, owner RGB/latent/noise/velocity,
or full owner hidden is accepted by the API.  All receipts remain explicitly
non-authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping

import torch
from torch import nn


METHOD = "bernini-braid-cumulative-action-program-v1"
SCHEMA_VERSION = "bernini-braid-cumulative-action-program-research-v1"
PROVENANCE_SCHEMA_VERSION = "bernini-braid-local-tensor-provenance-v1"
EVIDENCE_SCHEMA_VERSION = "bernini-braid-detached-segment-evidence-v1"

LATENT_PHASES = 21
HIDDEN_SIZE = 1536
PLAN_STAGES = 4
PLAN_WIDTH = 32
LOW_RANK = 32
CHANNELS_PER_RANK = HIDDEN_SIZE // LOW_RANK

STAGE_NAMES = ("onset", "transition", "completion", "terminal_hold")
STAGE_RISE_RANGES = ((0, 5), (5, 10), (10, 15), (15, 21))
STAGE_PLATEAU_STARTS = tuple(stop - 1 for _, stop in STAGE_RISE_RANGES)
PLAN_ROLES = ("action", "noop", "shuffled", "wrong")
FORBIDDEN_OWNER_CHANNELS = (
    "rgb",
    "clean_latent",
    "gaussian",
    "noise",
    "velocity",
    "full_hidden",
    "patch_layout",
    "text_embedding",
)

HIGH_SIGMA_MIN = 0.55
LOW_SIGMA_CUTOFF = 0.25
HIGH_SIGMA_WEIGHT = 1.0
MID_SIGMA_WEIGHT = 0.5
LOW_SIGMA_WEIGHT = 0.0

LOCAL_RESEARCH_AUTHORITY = "research-local-integrity-only-non-authoritative"


class BraidError(RuntimeError):
    """A BRAID CPU tensor, evidence, or geometry contract was violated."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise BraidError("value is not canonical finite ASCII JSON") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _local_tensor_sha256(value: torch.Tensor, *, label: str) -> str:
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.device.type != "cpu"
        or value.numel() <= 0
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise BraidError(f"{label} must be an exact-type finite CPU tensor")
    owned = value.detach().contiguous().clone()
    raw = owned.view(torch.uint8).numpy().tobytes(order="C")
    header = _canonical_json_bytes(
        {"dtype": str(owned.dtype), "shape": list(map(int, owned.shape))}
    )
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


def _closed_label(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 160
        or "\x00" in value
    ):
        raise BraidError(f"{label} must be a non-empty canonical research label")
    return value


def _positive_real(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BraidError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise BraidError(f"{label} must be finite and strictly positive")
    return result


def _nonnegative_real(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BraidError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise BraidError(f"{label} must be finite and nonnegative")
    return result


def sigma_gate(sigma: Any) -> tuple[str, float]:
    if isinstance(sigma, bool) or not isinstance(sigma, Real):
        raise BraidError("sigma must be a real scalar in [0,1]")
    value = float(sigma)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise BraidError("sigma must be finite and lie in [0,1]")
    if value >= HIGH_SIGMA_MIN:
        return "high", HIGH_SIGMA_WEIGHT
    if value >= LOW_SIGMA_CUTOFF:
        return "mid", MID_SIGMA_WEIGHT
    return "low_exact_base", LOW_SIGMA_WEIGHT


def build_cumulative_smoothstep_basis() -> torch.Tensor:
    """Build fixed start-zero, monotone, persistent ``[21,4]`` carriers."""

    basis = torch.zeros(LATENT_PHASES, PLAN_STAGES, dtype=torch.float64)
    for stage, (begin, stop) in enumerate(STAGE_RISE_RANGES):
        if not 0 <= begin < stop <= LATENT_PHASES or stop - begin < 2:
            raise RuntimeError("BRAID smooth-step range differs")
        coordinate = torch.linspace(0.0, 1.0, stop - begin, dtype=torch.float64)
        rise = coordinate.square() * (3.0 - 2.0 * coordinate)
        basis[begin:stop, stage] = rise
        basis[stop:, stage] = 1.0
    result = basis.to(dtype=torch.float32).contiguous()
    if not torch.equal(result[0], torch.zeros(PLAN_STAGES, dtype=torch.float32)):
        raise RuntimeError("BRAID basis is not start anchored")
    if bool(torch.any(result[1:] < result[:-1]).item()):
        raise RuntimeError("BRAID basis is not cumulative")
    if not torch.equal(result[-1], torch.ones(PLAN_STAGES, dtype=torch.float32)):
        raise RuntimeError("BRAID basis lacks terminal plateaus")
    for stage, plateau_start in enumerate(STAGE_PLATEAU_STARTS):
        if not torch.equal(
            result[plateau_start:, stage],
            torch.ones(LATENT_PHASES - plateau_start, dtype=torch.float32),
        ):
            raise RuntimeError("BRAID stage does not remain on its plateau")
    if bool(torch.any(result.sum(dim=0) <= 0.0).item()):
        raise RuntimeError("BRAID basis was temporally centered")
    return result


def build_fixed_hidden_encoder() -> torch.Tensor:
    """Return deterministic orthonormal ``A[32,1536]``."""

    if HIDDEN_SIZE % LOW_RANK != 0:
        raise RuntimeError("BRAID hidden size is not divisible by its rank")
    encoder = torch.zeros(LOW_RANK, HIDDEN_SIZE, dtype=torch.float32)
    scale = float(CHANNELS_PER_RANK) ** -0.5
    for rank in range(LOW_RANK):
        begin = rank * CHANNELS_PER_RANK
        stop = begin + CHANNELS_PER_RANK
        encoder[rank, begin:stop] = scale
    if not torch.allclose(
        encoder @ encoder.T, torch.eye(LOW_RANK), rtol=0.0, atol=2.0e-6
    ):
        raise RuntimeError("BRAID hidden encoder is not orthonormal")
    return encoder.contiguous()


def build_fixed_plan_projectors() -> torch.Tensor:
    """Return four deterministic orthogonal ``P_k[32,32]`` permutations."""

    identity = torch.eye(PLAN_WIDTH, dtype=torch.float32)
    projectors = torch.stack(
        [torch.roll(identity, shifts=stage * 5, dims=0) for stage in range(PLAN_STAGES)],
        dim=0,
    ).contiguous()
    for stage in range(PLAN_STAGES):
        if not torch.equal(projectors[stage] @ projectors[stage].T, identity):
            raise RuntimeError("BRAID plan projector is not orthogonal")
    return projectors


@dataclass(frozen=True)
class LocalTensorProvenance:
    """Locally computed integrity metadata; never upstream authority."""

    schema_version: str
    tensor_role: str
    shape: tuple[int, ...]
    dtype: str
    local_tensor_sha256: str
    semantics: str
    origin_label: str
    authority: str
    upstream_authentication_checked: bool
    digest: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tensor_role": self.tensor_role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "local_tensor_sha256": self.local_tensor_sha256,
            "semantics": self.semantics,
            "origin_label": self.origin_label,
            "authority": self.authority,
            "upstream_authentication_checked": self.upstream_authentication_checked,
        }

    def validate(self) -> None:
        if self.schema_version != PROVENANCE_SCHEMA_VERSION:
            raise BraidError("local provenance schema differs")
        if (
            type(self.tensor_role) is not str
            or not self.tensor_role
            or type(self.shape) is not tuple
            or not self.shape
            or any(type(value) is not int or value <= 0 for value in self.shape)
            or type(self.dtype) is not str
            or not self.dtype.startswith("torch.")
            or type(self.local_tensor_sha256) is not str
            or len(self.local_tensor_sha256) != 64
        ):
            raise BraidError("local provenance tensor metadata differs")
        _closed_label(self.origin_label, label="origin_label")
        if (
            self.authority != LOCAL_RESEARCH_AUTHORITY
            or self.upstream_authentication_checked is not False
        ):
            raise BraidError("local provenance overstated its authority")
        if self.digest != _object_sha256(self.payload()):
            raise BraidError("local provenance digest differs")


def _make_local_provenance(
    tensor: torch.Tensor,
    *,
    tensor_role: str,
    semantics: str,
    origin_label: str,
) -> LocalTensorProvenance:
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "tensor_role": tensor_role,
        "shape": list(map(int, tensor.shape)),
        "dtype": str(tensor.dtype),
        "local_tensor_sha256": _local_tensor_sha256(tensor, label=tensor_role),
        "semantics": semantics,
        "origin_label": _closed_label(origin_label, label="origin_label"),
        "authority": LOCAL_RESEARCH_AUTHORITY,
        "upstream_authentication_checked": False,
    }
    result = LocalTensorProvenance(
        schema_version=payload["schema_version"],
        tensor_role=payload["tensor_role"],
        shape=tuple(payload["shape"]),
        dtype=payload["dtype"],
        local_tensor_sha256=payload["local_tensor_sha256"],
        semantics=payload["semantics"],
        origin_label=payload["origin_label"],
        authority=payload["authority"],
        upstream_authentication_checked=False,
        digest=_object_sha256(payload),
    )
    result.validate()
    return result


class BraidTargetSnapshot:
    """Owned current target hidden; caller storage never aliases it."""

    def __init__(self, current_target_hidden: torch.Tensor, *, origin_label: str) -> None:
        if (
            type(current_target_hidden) is not torch.Tensor
            or current_target_hidden.device.type != "cpu"
            or current_target_hidden.dtype
            not in {torch.float16, torch.bfloat16, torch.float32}
            or current_target_hidden.layout != torch.strided
            or current_target_hidden.ndim != 4
            or int(current_target_hidden.shape[0]) <= 0
            or int(current_target_hidden.shape[1]) != LATENT_PHASES
            or int(current_target_hidden.shape[2]) <= 0
            or int(current_target_hidden.shape[3]) != HIDDEN_SIZE
            or not bool(torch.isfinite(current_target_hidden.detach()).all().item())
        ):
            raise BraidError(
                "current_target_hidden must be exact-type finite CPU [B,21,P,1536]"
            )
        self._tensor = current_target_hidden.detach().contiguous().clone()
        self._provenance = _make_local_provenance(
            self._tensor,
            tensor_role="current_source_conditioned_target_hidden",
            semantics=(
                "caller-labeled-current-source-conditioned-target-suffix;"
                "semantic-origin-not-verified-by-this-primitive"
            ),
            origin_label=origin_label,
        )
        if self._tensor.untyped_storage().data_ptr() == current_target_hidden.untyped_storage().data_ptr():
            raise BraidError("target snapshot aliases caller storage")

    @property
    def provenance(self) -> LocalTensorProvenance:
        return self._provenance

    def _consume_for_braid(self) -> torch.Tensor:
        if type(self) is not BraidTargetSnapshot:
            raise BraidError("target snapshot subclasses are forbidden")
        self._provenance.validate()
        if (
            type(self._tensor) is not torch.Tensor
            or tuple(map(int, self._tensor.shape)) != self._provenance.shape
            or str(self._tensor.dtype) != self._provenance.dtype
            or _local_tensor_sha256(self._tensor, label="live target snapshot")
            != self._provenance.local_tensor_sha256
        ):
            raise BraidError("owned target snapshot changed after construction")
        return self._tensor

    def tensor_copy(self) -> torch.Tensor:
        return self._consume_for_braid().clone()

    def receipt(self) -> Mapping[str, Any]:
        self._consume_for_braid()
        return {
            **self._provenance.payload(),
            "provenance_digest": self._provenance.digest,
            "caller_storage_owned_clone": True,
            "source_conditioning_semantics_verified": False,
            "scientific_authority": False,
        }


class BraidPlanSnapshot:
    """Owned spatial-orderless program ``[B,4,32]`` with an explicit role."""

    def __init__(
        self,
        plan: torch.Tensor,
        *,
        role: str,
        origin_label: str,
    ) -> None:
        if (
            type(plan) is not torch.Tensor
            or plan.device.type != "cpu"
            or plan.dtype != torch.float32
            or plan.layout != torch.strided
            or plan.ndim != 3
            or tuple(map(int, plan.shape[1:])) != (PLAN_STAGES, PLAN_WIDTH)
            or int(plan.shape[0]) <= 0
            or not bool(torch.isfinite(plan.detach()).all().item())
        ):
            raise BraidError("plan must be exact-type finite CPU FP32 [B,4,32]")
        if role not in PLAN_ROLES:
            raise BraidError("plan role is not registered")
        owned = plan.detach().contiguous().clone()
        per_example_zero = torch.all(owned == 0.0, dim=(1, 2))
        if role == "noop" and not bool(torch.all(per_example_zero).item()):
            raise BraidError("noop role requires the canonical all-zero plan")
        if role != "noop" and bool(torch.any(per_example_zero).item()):
            raise BraidError("non-noop role cannot claim a canonical noop plan")
        self._plan = owned
        self._role = role
        self._provenance = _make_local_provenance(
            self._plan,
            tensor_role=f"spatial_orderless_{role}_plan",
            semantics=(
                "four-stage-action-program-without-patch-axis;"
                "role-semantic-origin-not-verified-by-this-primitive"
            ),
            origin_label=origin_label,
        )
        self._canonical_noop_sha256 = _local_tensor_sha256(
            torch.zeros_like(self._plan), label="canonical noop plan"
        )
        if self._plan.untyped_storage().data_ptr() == plan.untyped_storage().data_ptr():
            raise BraidError("plan snapshot aliases caller storage")

    @property
    def role(self) -> str:
        return self._role

    def _consume_for_braid(self) -> torch.Tensor:
        if type(self) is not BraidPlanSnapshot:
            raise BraidError("plan snapshot subclasses are forbidden")
        self._provenance.validate()
        if (
            type(self._plan) is not torch.Tensor
            or tuple(map(int, self._plan.shape)) != self._provenance.shape
            or _local_tensor_sha256(self._plan, label="live plan snapshot")
            != self._provenance.local_tensor_sha256
        ):
            raise BraidError("owned plan snapshot changed after construction")
        zero = torch.all(self._plan == 0.0, dim=(1, 2))
        if self._role == "noop":
            if not bool(torch.all(zero).item()):
                raise BraidError("live noop plan is not canonical zero")
            if self._provenance.local_tensor_sha256 != self._canonical_noop_sha256:
                raise BraidError("noop plan digest differs from canonical zero")
        elif bool(torch.any(zero).item()):
            raise BraidError("live non-noop plan collapsed to noop")
        return self._plan

    def plan_copy(self) -> torch.Tensor:
        return self._consume_for_braid().clone()

    def receipt(self) -> Mapping[str, Any]:
        self._consume_for_braid()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "role": self._role,
            "shape": list(self._plan.shape),
            "spatial_orderless_by_shape": True,
            "patch_axis_present": False,
            "owner_layout_channel_present": False,
            "local_tensor_sha256": self._provenance.local_tensor_sha256,
            "canonical_noop_plan_sha256": self._canonical_noop_sha256,
            "provenance_digest": self._provenance.digest,
            "forbidden_owner_channels": list(FORBIDDEN_OWNER_CHANNELS),
            "forbidden_owner_channels_consumed_by_api": [],
            "authority": LOCAL_RESEARCH_AUTHORITY,
            "role_semantics_verified": False,
            "scientific_authority": False,
        }
        return {**payload, "digest": _object_sha256(payload)}


@dataclass(frozen=True)
class BraidEvidenceThresholds:
    correct_minus_shuffled: float = 0.05
    correct_minus_wrong: float = 0.05
    correct_minus_noop: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "correct_minus_shuffled",
            "correct_minus_wrong",
            "correct_minus_noop",
        ):
            _nonnegative_real(getattr(self, name), label=name)

    def payload(self) -> Mapping[str, Any]:
        return {
            name: float(getattr(self, name)).hex()
            for name in (
                "correct_minus_shuffled",
                "correct_minus_wrong",
                "correct_minus_noop",
            )
        }


class BraidEvidenceSnapshot:
    """Owned detached CPU FP64 segment scores and hard conjunction."""

    def __init__(
        self,
        correct: torch.Tensor,
        shuffled: torch.Tensor,
        wrong: torch.Tensor,
        noop: torch.Tensor,
        *,
        thresholds: BraidEvidenceThresholds = BraidEvidenceThresholds(),
        origin_label: str,
    ) -> None:
        if not isinstance(thresholds, BraidEvidenceThresholds):
            raise BraidError("thresholds must be BraidEvidenceThresholds")
        owned: list[torch.Tensor] = []
        for label, value in (
            ("correct", correct),
            ("shuffled", shuffled),
            ("wrong", wrong),
            ("noop", noop),
        ):
            if (
                type(value) is not torch.Tensor
                or value.device.type != "cpu"
                or not value.is_floating_point()
                or value.ndim != 2
                or int(value.shape[0]) <= 0
                or int(value.shape[1]) != PLAN_STAGES
                or not bool(torch.isfinite(value.detach()).all().item())
            ):
                raise BraidError(f"{label} scores must be exact finite CPU [B,4]")
            owned.append(
                value.detach().to(dtype=torch.float64).contiguous().clone()
            )
        if len({tuple(value.shape) for value in owned}) != 1:
            raise BraidError("evidence score shapes differ")
        self._correct, self._shuffled, self._wrong, self._noop = owned
        self._thresholds = thresholds
        self._origin_label = _closed_label(origin_label, label="origin_label")
        self._score_provenances = tuple(
            _make_local_provenance(
                value,
                tensor_role=f"{label}_segment_scores",
                semantics=(
                    "caller-origin-detached-segment-score;"
                    "semantic-validity-not-verified-by-this-primitive"
                ),
                origin_label=self._origin_label,
            )
            for label, value in zip(
                ("correct", "shuffled", "wrong", "noop"), owned
            )
        )
        self._margins, self._eligible = self._replay()
        payload = self._payload()
        self._digest = _object_sha256(payload)

    def _replay(self) -> tuple[torch.Tensor, torch.Tensor]:
        margins = torch.stack(
            (
                self._correct - self._shuffled,
                self._correct - self._wrong,
                self._correct - self._noop,
            ),
            dim=1,
        ).contiguous()
        threshold = torch.tensor(
            [
                self._thresholds.correct_minus_shuffled,
                self._thresholds.correct_minus_wrong,
                self._thresholds.correct_minus_noop,
            ],
            dtype=torch.float64,
        ).reshape(1, 3, 1)
        eligible = torch.all(margins >= threshold, dim=(1, 2)).contiguous()
        return margins, eligible

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "origin_label": self._origin_label,
            "score_provenance_digests": [
                value.digest for value in self._score_provenances
            ],
            "thresholds": self._thresholds.payload(),
            "margins_sha256": _local_tensor_sha256(
                self._margins, label="evidence margins"
            ),
            "eligible": self._eligible.tolist(),
            "reduction": "all-four-stages-all-three-controls-hard-conjunction",
            "authority": LOCAL_RESEARCH_AUTHORITY,
        }

    def _consume_for_braid(self) -> torch.Tensor:
        if type(self) is not BraidEvidenceSnapshot:
            raise BraidError("evidence snapshot subclasses are forbidden")
        values = (self._correct, self._shuffled, self._wrong, self._noop)
        for label, value, provenance in zip(
            ("correct", "shuffled", "wrong", "noop"),
            values,
            self._score_provenances,
        ):
            provenance.validate()
            if (
                type(value) is not torch.Tensor
                or value.dtype != torch.float64
                or value.requires_grad
                or value.grad_fn is not None
                or _local_tensor_sha256(value, label=f"live {label} scores")
                != provenance.local_tensor_sha256
            ):
                raise BraidError(f"owned {label} scores changed")
        margins, eligible = self._replay()
        if (
            not torch.equal(margins, self._margins)
            or not torch.equal(eligible, self._eligible)
            or self._digest != _object_sha256(self._payload())
        ):
            raise BraidError("evidence conjunction changed after construction")
        return self._eligible

    @property
    def margins(self) -> torch.Tensor:
        self._consume_for_braid()
        return self._margins.clone()

    def eligible_copy(self) -> torch.Tensor:
        return self._consume_for_braid().clone()

    def receipt(self) -> Mapping[str, Any]:
        self._consume_for_braid()
        return {
            **self._payload(),
            "digest": self._digest,
            "scores_owned_detached_cpu_fp64": True,
            "semantic_admission_authority": False,
            "update_authority": False,
        }


class BraidCumulativeActionProgram(nn.Module):
    """Rank-32 target-state-conditioned cumulative program adapter."""

    def __init__(
        self,
        *,
        max_segment_token_norm: float = 2.0,
        max_global_token_norm: float = 3.0,
    ) -> None:
        super().__init__()
        self.max_segment_token_norm = _positive_real(
            max_segment_token_norm, label="max_segment_token_norm"
        )
        self.max_global_token_norm = _positive_real(
            max_global_token_norm, label="max_global_token_norm"
        )
        self.register_buffer(
            "cumulative_basis", build_cumulative_smoothstep_basis(), persistent=True
        )
        self.register_buffer(
            "hidden_encoder", build_fixed_hidden_encoder(), persistent=True
        )
        self.register_buffer(
            "plan_projectors", build_fixed_plan_projectors(), persistent=True
        )
        self.stage_decoder = nn.Parameter(
            torch.zeros(PLAN_STAGES, HIDDEN_SIZE, LOW_RANK, dtype=torch.float32)
        )

    def _validate_geometry(self) -> None:
        expected_basis = build_cumulative_smoothstep_basis()
        expected_hidden = build_fixed_hidden_encoder()
        expected_plan = build_fixed_plan_projectors()
        named = tuple(self.named_parameters())
        if (
            self.cumulative_basis.device.type != "cpu"
            or self.hidden_encoder.device.type != "cpu"
            or self.plan_projectors.device.type != "cpu"
            or self.stage_decoder.device.type != "cpu"
            or self.cumulative_basis.dtype != torch.float32
            or self.hidden_encoder.dtype != torch.float32
            or self.plan_projectors.dtype != torch.float32
            or self.stage_decoder.dtype != torch.float32
            or not torch.equal(self.cumulative_basis, expected_basis)
            or not torch.equal(self.hidden_encoder, expected_hidden)
            or not torch.equal(self.plan_projectors, expected_plan)
            or tuple(self.stage_decoder.shape)
            != (PLAN_STAGES, HIDDEN_SIZE, LOW_RANK)
            or tuple(name for name, _ in named) != ("stage_decoder",)
            or self.stage_decoder.requires_grad is not True
            or not bool(torch.isfinite(self.stage_decoder.detach()).all().item())
        ):
            raise BraidError("BRAID projection/decoder geometry changed")

    def cumulative_program(self, plan: BraidPlanSnapshot) -> torch.Tensor:
        if type(plan) is not BraidPlanSnapshot:
            raise BraidError("plan must be an exact BraidPlanSnapshot")
        self._validate_geometry()
        code = plan._consume_for_braid()
        projected = torch.einsum("kro,bko->bkr", self.plan_projectors, code)
        program = torch.einsum("tk,bkr->btr", self.cumulative_basis, projected)
        if int(torch.count_nonzero(program[:, 0]).item()) != 0:
            raise BraidError("cumulative program is not start anchored")
        return program.contiguous()

    def _validated_inputs(
        self,
        target: Any,
        plan: Any,
        evidence: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if type(target) is not BraidTargetSnapshot:
            raise BraidError("target must be an exact BraidTargetSnapshot")
        if type(plan) is not BraidPlanSnapshot:
            raise BraidError("plan must be an exact BraidPlanSnapshot")
        if type(evidence) is not BraidEvidenceSnapshot:
            raise BraidError("evidence must be an exact BraidEvidenceSnapshot")
        self._validate_geometry()
        hidden = target._consume_for_braid()
        code = plan._consume_for_braid()
        eligible = evidence._consume_for_braid()
        if int(hidden.shape[0]) != int(code.shape[0]) or int(hidden.shape[0]) != int(
            eligible.numel()
        ):
            raise BraidError("target/plan/evidence batch sizes differ")
        if plan.role == "noop":
            active = torch.zeros_like(eligible)
        else:
            active = eligible.clone()
        return hidden, code, active

    @staticmethod
    def _cap_token_norm(value: torch.Tensor, cap: float) -> torch.Tensor:
        norms = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        return value * torch.clamp(cap / norms.clamp_min(1.0e-12), max=1.0)

    def _delta_from_validated(
        self,
        hidden: torch.Tensor,
        plan: torch.Tensor,
        active: torch.Tensor,
        gate_weight: float,
    ) -> torch.Tensor:
        if gate_weight == 0.0 or not bool(torch.any(active).item()):
            return torch.zeros_like(hidden)
        hidden_coordinate = torch.einsum(
            "rd,btpd->btpr", self.hidden_encoder, hidden.float()
        )
        plan_coordinate = torch.einsum(
            "kro,bko->bkr", self.plan_projectors, plan
        )
        result = torch.zeros_like(hidden, dtype=torch.float32)
        for stage in range(PLAN_STAGES):
            fused = hidden_coordinate * plan_coordinate[:, stage].reshape(
                int(hidden.shape[0]), 1, 1, LOW_RANK
            )
            decoded = torch.einsum(
                "btpr,dr->btpd", fused, self.stage_decoder[stage]
            )
            decoded = self._cap_token_norm(decoded, self.max_segment_token_norm)
            contribution = decoded * self.cumulative_basis[:, stage].reshape(
                1, LATENT_PHASES, 1, 1
            )
            if bool(
                torch.any(
                    torch.linalg.vector_norm(decoded.detach(), dim=-1)
                    > self.max_segment_token_norm + 2.0e-6
                ).item()
            ):
                raise BraidError("stage/patch token norm cap failed")
            result = result + contribution
        result = self._cap_token_norm(result, self.max_global_token_norm)
        if bool(
            torch.any(
                torch.linalg.vector_norm(result.detach(), dim=-1)
                > self.max_global_token_norm + 2.0e-6
            ).item()
        ):
            raise BraidError("global patch token norm cap failed")
        result = result * active.to(dtype=torch.float32).reshape(-1, 1, 1, 1)
        result = result * float(gate_weight)
        if int(torch.count_nonzero(result[:, 0]).item()) != 0:
            raise BraidError("BRAID residual changed the start phase")
        if not bool(torch.isfinite(result).all().item()):
            raise BraidError("BRAID residual is non-finite")
        return result.to(dtype=hidden.dtype)

    def adapter_delta(
        self,
        target: BraidTargetSnapshot,
        plan: BraidPlanSnapshot,
        evidence: BraidEvidenceSnapshot,
        *,
        sigma: Real,
    ) -> torch.Tensor:
        hidden, code, active = self._validated_inputs(target, plan, evidence)
        _, weight = sigma_gate(sigma)
        return self._delta_from_validated(hidden, code, active, weight)

    def forward(
        self,
        target: BraidTargetSnapshot,
        plan: BraidPlanSnapshot,
        evidence: BraidEvidenceSnapshot,
        *,
        sigma: Real,
    ) -> torch.Tensor:
        hidden, code, active = self._validated_inputs(target, plan, evidence)
        _, weight = sigma_gate(sigma)
        if weight == 0.0 or not bool(torch.any(active).item()):
            return hidden
        delta = self._delta_from_validated(hidden, code, active, weight)
        if bool(torch.all(active).item()):
            return hidden + delta
        output = hidden.clone()
        output[active] = hidden[active] + delta[active]
        return output

    def receipt(self) -> Mapping[str, Any]:
        self._validate_geometry()
        named = tuple(self.named_parameters())
        payload = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "scope": LOCAL_RESEARCH_AUTHORITY,
            "target_shape": ["B", LATENT_PHASES, "P", HIDDEN_SIZE],
            "plan_shape": ["B", PLAN_STAGES, PLAN_WIDTH],
            "plan_has_patch_axis": False,
            "plan_spatial_broadcast_before_hidden_interaction": True,
            "forbidden_owner_channels": list(FORBIDDEN_OWNER_CHANNELS),
            "stage_names": list(STAGE_NAMES),
            "stage_rise_ranges": [list(value) for value in STAGE_RISE_RANGES],
            "stage_plateau_starts": list(STAGE_PLATEAU_STARTS),
            "start_anchored": True,
            "cumulative_plateau": True,
            "temporal_centering_used": False,
            "zero_temporal_dc_required": False,
            "cumulative_basis_sha256": _local_tensor_sha256(
                self.cumulative_basis, label="cumulative smoothstep basis"
            ),
            "fixed_hidden_encoder_sha256": _local_tensor_sha256(
                self.hidden_encoder, label="fixed hidden encoder"
            ),
            "fixed_plan_projectors_sha256": _local_tensor_sha256(
                self.plan_projectors, label="fixed plan projectors"
            ),
            "trainable_parameter_names": [name for name, _ in named],
            "only_stage_decoder_trainable": (
                tuple(name for name, _ in named) == ("stage_decoder",)
                and self.stage_decoder.requires_grad is True
            ),
            "stage_decoder_zero_at_receipt_time": bool(
                torch.count_nonzero(self.stage_decoder.detach()).item() == 0
            ),
            "max_segment_token_norm": self.max_segment_token_norm,
            "max_global_token_norm": self.max_global_token_norm,
            "sigma_gate": {
                "high_min": HIGH_SIGMA_MIN,
                "low_cutoff": LOW_SIGMA_CUTOFF,
                "weights": [HIGH_SIGMA_WEIGHT, MID_SIGMA_WEIGHT, LOW_SIGMA_WEIGHT],
            },
            "source_conditioning_semantics_verified_by_primitive": False,
            "plan_role_semantics_verified_by_primitive": False,
            "scientific_authority": False,
            "update_authority": False,
        }
        return {**payload, "digest": _object_sha256(payload)}


__all__ = [
    "BraidCumulativeActionProgram",
    "BraidError",
    "BraidEvidenceSnapshot",
    "BraidEvidenceThresholds",
    "BraidPlanSnapshot",
    "BraidTargetSnapshot",
    "CHANNELS_PER_RANK",
    "EVIDENCE_SCHEMA_VERSION",
    "FORBIDDEN_OWNER_CHANNELS",
    "HIDDEN_SIZE",
    "LATENT_PHASES",
    "LOCAL_RESEARCH_AUTHORITY",
    "LOW_RANK",
    "LocalTensorProvenance",
    "METHOD",
    "PLAN_ROLES",
    "PLAN_STAGES",
    "PLAN_WIDTH",
    "PROVENANCE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "STAGE_NAMES",
    "STAGE_PLATEAU_STARTS",
    "STAGE_RISE_RANGES",
    "build_cumulative_smoothstep_basis",
    "build_fixed_hidden_encoder",
    "build_fixed_plan_projectors",
    "sigma_gate",
]
