"""Auditable PyTorch core for episodic privileged motion codes.

The paired target and a K-shot support set are *training-only* teachers.  They
are deliberately absent from :class:`AmortizedMotionCodePredictor.forward`,
whose complete inference boundary is ``(source_descriptor, text_descriptor)``.

This module does not render video and does not patch Bernini.  It defines the
small motion-code object, robust support aggregation, teacher-to-student losses,
and exact post-attention gating of Bernini's already separated projected heads.
It intentionally does *not* split a pre-projection 1536-vector into 12 chunks:
``to_k``/``to_v`` mix those channels, so such chunks are not attention heads.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

METHOD_NAME = "episodic-privileged-motion-code"
SCHEMA_VERSION = "bernini-epmc-core-v2"

LATENT_PHASES = 21
NONZERO_PHASES = LATENT_PHASES - 1
MOTION_BLOCKS = 16
ATTENTION_HEADS = 12
HEAD_DIM = 128
HIDDEN_SIZE = ATTENTION_HEADS * HEAD_DIM

PHASE_GATE_BOUND = 1.0
BLOCK_HEAD_GATE_BOUND = 1.0

REVERSE_PHASE_INDICES = (0, *tuple(range(LATENT_PHASES - 1, 0, -1)))
SHUFFLE_PHASE_INDICES = (
    0,
    17,
    18,
    1,
    6,
    16,
    4,
    12,
    11,
    7,
    13,
    19,
    2,
    15,
    8,
    3,
    9,
    20,
    5,
    10,
    14,
)

PROTOTYPE_HUBER_DELTA = 0.25
PROTOTYPE_MAX_ITERATIONS = 32
PROTOTYPE_TOLERANCE = 1.0e-6
EPSILON = 1.0e-8

INFERENCE_ARGUMENTS = ("source_descriptor", "text_descriptor")
FORBIDDEN_INFERENCE_ARGUMENTS = (
    "target",
    "support",
    "mask",
    "flow",
    "pose",
    "track",
    "trajectory",
    "reference",
)


class PrivilegedMotionCodeContractError(ValueError):
    """Raised when an EPMC tensor or receipt violates the frozen contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise PrivilegedMotionCodeContractError(
            "digest tensor must be a torch.Tensor"
        )
    if value.device.type == "meta":
        raise PrivilegedMotionCodeContractError("digest tensor cannot be meta")
    # ``repeat(1)`` forces stride 1 even for singleton slices that PyTorch may
    # call contiguous while retaining a larger source stride.
    detached = value.detach().reshape(-1).repeat(1).cpu()
    metadata = {
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(metadata))
    digest.update(b"\0")
    digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _require_tensor(name: str, value: Any) -> None:
    if not isinstance(value, torch.Tensor):
        raise PrivilegedMotionCodeContractError(f"{name} must be a torch.Tensor")
    if value.device.type == "meta":
        raise PrivilegedMotionCodeContractError(f"{name} cannot be a meta tensor")
    if not torch.is_floating_point(value):
        raise PrivilegedMotionCodeContractError(f"{name} must have floating dtype")


def _require_fp32_finite(name: str, value: Any) -> None:
    _require_tensor(name, value)
    if value.dtype != torch.float32:
        raise PrivilegedMotionCodeContractError(f"{name} must be float32")
    if not bool(torch.isfinite(value).all().item()):
        raise PrivilegedMotionCodeContractError(f"{name} contains NaN or infinity")


def _require_positive_zero(name: str, value: torch.Tensor) -> None:
    if int(torch.count_nonzero(value).item()) != 0:
        raise PrivilegedMotionCodeContractError(f"{name} must be exact zero")
    # Inspect payload bytes instead of ``torch.signbit``: older PyTorch builds
    # on macOS have returned False for IEEE-754 -0.0 despite retaining bit 31.
    bytes_view = value.detach().reshape(-1).repeat(1).view(torch.uint8)
    if int(torch.count_nonzero(bytes_view).item()):
        raise PrivilegedMotionCodeContractError(
            f"{name} must use byte-exact positive zero, not signed zero"
        )


@dataclass(frozen=True)
class MotionCode:
    """A compact bounded action code.

    ``phase_gates`` has shape ``[B,21]`` and its first phase is byte-exact
    positive zero.  ``block_head_gates`` has shape ``[B,16,12]``.  Both fields
    are bounded to [-1, 1], allowing a learned code to suppress or reverse a
    motion residual without unbounded amplification.
    """

    phase_gates: torch.Tensor
    block_head_gates: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return int(self.phase_gates.shape[0])

    def validate(self, *, require_noop: bool = False) -> None:
        _require_fp32_finite("phase_gates", self.phase_gates)
        _require_fp32_finite("block_head_gates", self.block_head_gates)
        if self.phase_gates.ndim != 2 or tuple(self.phase_gates.shape[1:]) != (
            LATENT_PHASES,
        ):
            raise PrivilegedMotionCodeContractError(
                "phase_gates must have exact shape [B,21]"
            )
        if self.block_head_gates.ndim != 3 or tuple(
            self.block_head_gates.shape[1:]
        ) != (MOTION_BLOCKS, ATTENTION_HEADS):
            raise PrivilegedMotionCodeContractError(
                "block_head_gates must have exact shape [B,16,12]"
            )
        if self.batch_size < 1 or int(self.block_head_gates.shape[0]) != self.batch_size:
            raise PrivilegedMotionCodeContractError(
                "motion-code batch dimensions must be positive and equal"
            )
        if self.phase_gates.device != self.block_head_gates.device:
            raise PrivilegedMotionCodeContractError(
                "motion-code tensors must share one device"
            )
        if bool((self.phase_gates.abs() > PHASE_GATE_BOUND).any().item()):
            raise PrivilegedMotionCodeContractError("phase gate escaped [-1,1]")
        if bool(
            (self.block_head_gates.abs() > BLOCK_HEAD_GATE_BOUND).any().item()
        ):
            raise PrivilegedMotionCodeContractError("block/head gate escaped [-1,1]")
        _require_positive_zero("phase_gates[:,0]", self.phase_gates[:, 0])
        if require_noop:
            _require_positive_zero("no-op phase gates", self.phase_gates)
            _require_positive_zero(
                "no-op block/head gates", self.block_head_gates
            )

    def flattened(self) -> torch.Tensor:
        """Return ``[B,212]`` without the structurally fixed phase-0 slot."""

        self.validate()
        return torch.cat(
            (
                self.phase_gates[:, 1:],
                self.block_head_gates.reshape(self.batch_size, -1),
            ),
            dim=1,
        )

    def audit_receipt(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": CONTRACT_RECEIPT_SHA256,
            "shape": {
                "phase_gates": [self.batch_size, LATENT_PHASES],
                "block_head_gates": [
                    self.batch_size,
                    MOTION_BLOCKS,
                    ATTENTION_HEADS,
                ],
            },
            "phase0_exact_positive_zero": True,
            "phase_gate_max_abs": float(self.phase_gates.detach().abs().max().item()),
            "block_head_gate_max_abs": float(
                self.block_head_gates.detach().abs().max().item()
            ),
            "phase_gates_sha256": _tensor_sha256(self.phase_gates),
            "block_head_gates_sha256": _tensor_sha256(self.block_head_gates),
        }


def decode_bounded_motion_code(
    phase_logits_nonzero: torch.Tensor,
    block_head_logits: torch.Tensor,
) -> MotionCode:
    """Map unconstrained logits to the frozen compact code with ``tanh``.

    Phase 0 is not accepted as a logit at all.  It is constructed from a fresh
    positive-zero tensor, so an optimizer cannot accidentally activate it.
    """

    _require_fp32_finite("phase_logits_nonzero", phase_logits_nonzero)
    _require_fp32_finite("block_head_logits", block_head_logits)
    if phase_logits_nonzero.ndim != 2 or tuple(
        phase_logits_nonzero.shape[1:]
    ) != (NONZERO_PHASES,):
        raise PrivilegedMotionCodeContractError(
            "phase_logits_nonzero must have exact shape [B,20]"
        )
    if block_head_logits.ndim != 3 or tuple(block_head_logits.shape[1:]) != (
        MOTION_BLOCKS,
        ATTENTION_HEADS,
    ):
        raise PrivilegedMotionCodeContractError(
            "block_head_logits must have exact shape [B,16,12]"
        )
    if tuple(phase_logits_nonzero.shape[:1]) != tuple(block_head_logits.shape[:1]):
        raise PrivilegedMotionCodeContractError("motion-code logit batches differ")
    if phase_logits_nonzero.device != block_head_logits.device:
        raise PrivilegedMotionCodeContractError(
            "motion-code logits must share one device"
        )
    if int(phase_logits_nonzero.shape[0]) < 1:
        raise PrivilegedMotionCodeContractError("motion-code batch must be positive")

    phase_nonzero = torch.tanh(phase_logits_nonzero) * PHASE_GATE_BOUND
    phase_zero = torch.zeros_like(phase_nonzero[:, :1])
    phase_gates = torch.cat((phase_zero, phase_nonzero), dim=1)
    block_head_gates = torch.tanh(block_head_logits) * BLOCK_HEAD_GATE_BOUND
    return MotionCode(phase_gates, block_head_gates)


def canonical_noop_motion_code(
    batch_size: int,
    *,
    device: torch.device | str | None = None,
) -> MotionCode:
    """Return the only canonical no-op code: positive zero in every field."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise PrivilegedMotionCodeContractError("batch_size must be a positive integer")
    return MotionCode(
        phase_gates=torch.zeros(
            batch_size, LATENT_PHASES, dtype=torch.float32, device=device
        ),
        block_head_gates=torch.zeros(
            batch_size,
            MOTION_BLOCKS,
            ATTENTION_HEADS,
            dtype=torch.float32,
            device=device,
        ),
    )


class LearnableEpisodicMotionCode(nn.Module):
    """The only trainable object required by the per-example code oracle."""

    def __init__(self, batch_size: int = 1) -> None:
        super().__init__()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise PrivilegedMotionCodeContractError(
                "batch_size must be a positive integer"
            )
        self.phase_logits_nonzero = nn.Parameter(
            torch.zeros(batch_size, NONZERO_PHASES, dtype=torch.float32)
        )
        self.block_head_logits = nn.Parameter(
            torch.zeros(
                batch_size,
                MOTION_BLOCKS,
                ATTENTION_HEADS,
                dtype=torch.float32,
            )
        )

    def forward(self) -> MotionCode:
        return decode_bounded_motion_code(
            self.phase_logits_nonzero, self.block_head_logits
        )


def _stacked_code(code: MotionCode) -> torch.Tensor:
    code.validate()
    return code.flattened()


def _code_from_flattened(flattened: torch.Tensor) -> MotionCode:
    _require_fp32_finite("flattened motion code", flattened)
    expected = NONZERO_PHASES + MOTION_BLOCKS * ATTENTION_HEADS
    if flattened.ndim != 2 or int(flattened.shape[1]) != expected:
        raise PrivilegedMotionCodeContractError(
            f"flattened motion code must have exact shape [B,{expected}]"
        )
    phase_nonzero = flattened[:, :NONZERO_PHASES]
    block_head = flattened[:, NONZERO_PHASES:].reshape(
        int(flattened.shape[0]), MOTION_BLOCKS, ATTENTION_HEADS
    )
    phase = torch.cat((torch.zeros_like(phase_nonzero[:, :1]), phase_nonzero), dim=1)
    return MotionCode(phase, block_head)


@dataclass(frozen=True)
class KShotPrototypeResult:
    """Detached training-only robust prototype and its audit evidence."""

    code: MotionCode
    support_count: int
    rule: str
    iterations: int
    converged: bool
    final_update_rms: float
    support_sha256: str

    def audit_receipt(self) -> dict[str, Any]:
        return {
            "method": METHOD_NAME,
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": CONTRACT_RECEIPT_SHA256,
            "training_only": True,
            "support_available_at_inference": False,
            "support_count": self.support_count,
            "aggregation_rule": self.rule,
            "k_equals_two_rule": "exact_arithmetic_midpoint",
            "huber_delta": PROTOTYPE_HUBER_DELTA,
            "max_iterations": PROTOTYPE_MAX_ITERATIONS,
            "tolerance": PROTOTYPE_TOLERANCE,
            "iterations": self.iterations,
            "converged": self.converged,
            "final_update_rms": self.final_update_rms,
            "support_sha256": self.support_sha256,
            "prototype": self.code.audit_receipt(),
        }


def build_training_support_prototype(
    support_codes: MotionCode,
) -> KShotPrototypeResult:
    """Build a detached robust prototype from K training support codes.

    Rules are intentionally closed and deterministic:

    * K=1 returns the single support code;
    * K=2 returns the exact arithmetic midpoint because a geometric median of
      two points is non-unique;
    * K>=3 uses spatial Huber IRLS, initialized by the coordinate-wise median.

    No corresponding support argument exists on the inference predictor.
    """

    support_codes.validate()
    points = _stacked_code(support_codes).detach()
    support_count = int(points.shape[0])
    support_digest = _tensor_sha256(points)

    with torch.no_grad():
        if support_count == 1:
            center = points.clone()
            rule = "single_support_identity"
            iterations = 0
            converged = True
            final_update_rms = 0.0
        elif support_count == 2:
            center = points.mean(dim=0, keepdim=True)
            rule = "exact_arithmetic_midpoint"
            iterations = 0
            converged = True
            final_update_rms = 0.0
        else:
            center = points.median(dim=0, keepdim=True).values
            rule = "spatial_huber_irls"
            converged = False
            final_update_rms = math.inf
            iterations = 0
            dimension_scale = math.sqrt(float(points.shape[1]))
            for iteration in range(1, PROTOTYPE_MAX_ITERATIONS + 1):
                residual = points - center
                distance = residual.square().sum(dim=1).sqrt() / dimension_scale
                safe_distance = distance.clamp_min(EPSILON)
                weights = torch.where(
                    distance <= PROTOTYPE_HUBER_DELTA,
                    torch.ones_like(distance),
                    PROTOTYPE_HUBER_DELTA / safe_distance,
                )
                updated = (weights[:, None] * points).sum(dim=0, keepdim=True)
                updated = updated / weights.sum().clamp_min(EPSILON)
                update_rms_tensor = (updated - center).square().mean().sqrt()
                final_update_rms = float(update_rms_tensor.item())
                center = updated
                iterations = iteration
                if final_update_rms <= PROTOTYPE_TOLERANCE:
                    converged = True
                    break

            if not converged:
                raise PrivilegedMotionCodeContractError(
                    "spatial Huber IRLS did not converge under the frozen "
                    f"{PROTOTYPE_MAX_ITERATIONS}-iteration contract"
                )

    if not bool(torch.isfinite(center).all().item()):
        raise PrivilegedMotionCodeContractError("prototype aggregation became non-finite")
    # A convex combination of valid gates stays within the frozen bounds.
    prototype = _code_from_flattened(center.contiguous())
    return KShotPrototypeResult(
        code=prototype,
        support_count=support_count,
        rule=rule,
        iterations=iterations,
        converged=converged,
        final_update_rms=final_update_rms,
        support_sha256=support_digest,
    )


@dataclass(frozen=True)
class ProjectedHeadGatingResult:
    """Post-attention projected heads and sufficient audit statistics."""

    projected_motion_heads_fp32: torch.Tensor
    effective_head_gates: torch.Tensor
    query_phase_ids: torch.Tensor
    block_index: int
    input_heads_sha256: str | None
    code_sha256: str | None
    audit_digests: bool

    def flattened_output(self) -> torch.Tensor:
        """Return the gated post-attention value as ``[B,Q,1536]``.

        Flattening is valid here because the input has already been projected
        and separated by Bernini attention into 12 actual 128-D heads.
        """

        batch_size, query_count = self.projected_motion_heads_fp32.shape[:2]
        return self.projected_motion_heads_fp32.reshape(
            int(batch_size), int(query_count), HIDDEN_SIZE
        )

    def audit_receipt(self) -> dict[str, Any]:
        if (
            not self.audit_digests
            or self.input_heads_sha256 is None
            or self.code_sha256 is None
        ):
            raise PrivilegedMotionCodeContractError(
                "runtime fast-path result has no audit digests"
            )
        return {
            "method": METHOD_NAME,
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": CONTRACT_RECEIPT_SHA256,
            "block_index": self.block_index,
            "shape": [
                int(item) for item in self.projected_motion_heads_fp32.shape
            ],
            "gating_point": "post_attention_projected_heads_before_output_merge",
            "preprojection_channel_chunk_gating": False,
            "source_query_output_exact_positive_zero": True,
            "target_phase0_output_exact_positive_zero": True,
            "input_heads_sha256": self.input_heads_sha256,
            "query_phase_ids_sha256": _tensor_sha256(self.query_phase_ids),
            "code_sha256": self.code_sha256,
            "effective_head_gates_sha256": _tensor_sha256(
                self.effective_head_gates
            ),
            "projected_motion_heads_fp32_sha256": _tensor_sha256(
                self.projected_motion_heads_fp32
            ),
        }


def gate_projected_motion_heads(
    projected_motion_heads: torch.Tensor,
    query_phase_ids: torch.Tensor,
    motion_code: MotionCode,
    *,
    block_index: int,
    audit_digests: bool = True,
) -> ProjectedHeadGatingResult:
    """Gate actual post-attention heads for one Bernini motion block.

    ``projected_motion_heads`` must be the attention result *after* value
    projection and head separation, with exact shape ``[B,Q,12,128]``.  It must
    not be a pre-``to_k``/``to_v`` 1536-D carrier reshaped into artificial
    chunks.  ``query_phase_ids`` maps each query to source (-1) or target phase
    0..20.  Source queries and target phase 0 are structurally disabled.

    For target phases 1..20, the effective signed gate is exactly
    ``0.5 * (phase_gate + block_head_gate)``.  A straight-through positive-zero
    correction preserves gradients for both zero-initialized gate families.
    """

    _require_fp32_finite("projected_motion_heads", projected_motion_heads)
    if projected_motion_heads.ndim != 4 or tuple(
        projected_motion_heads.shape[2:]
    ) != (ATTENTION_HEADS, HEAD_DIM):
        raise PrivilegedMotionCodeContractError(
            "projected_motion_heads must have exact shape [B,Q,12,128]"
        )
    batch_size = int(projected_motion_heads.shape[0])
    query_count = int(projected_motion_heads.shape[1])
    if batch_size < 1 or query_count < 1:
        raise PrivilegedMotionCodeContractError(
            "projected head batch and query dimensions must be positive"
        )
    if not isinstance(query_phase_ids, torch.Tensor):
        raise PrivilegedMotionCodeContractError(
            "query_phase_ids must be a torch.Tensor"
        )
    if query_phase_ids.device.type == "meta":
        raise PrivilegedMotionCodeContractError(
            "query_phase_ids cannot be a meta tensor"
        )
    if query_phase_ids.dtype != torch.int64:
        raise PrivilegedMotionCodeContractError(
            "query_phase_ids must have dtype torch.int64"
        )
    if query_phase_ids.ndim != 1 or int(query_phase_ids.shape[0]) != query_count:
        raise PrivilegedMotionCodeContractError(
            "query_phase_ids must have exact shape [Q]"
        )
    if query_phase_ids.device != projected_motion_heads.device:
        raise PrivilegedMotionCodeContractError(
            "projected heads and query_phase_ids must share one device"
        )
    if bool(
        ((query_phase_ids < -1) | (query_phase_ids >= LATENT_PHASES)).any().item()
    ):
        raise PrivilegedMotionCodeContractError(
            "query_phase_ids values must be source=-1 or target=0..20"
        )
    motion_code.validate()
    if projected_motion_heads.device != motion_code.phase_gates.device:
        raise PrivilegedMotionCodeContractError(
            "projected heads and motion code must share one device"
        )
    if batch_size != motion_code.batch_size:
        raise PrivilegedMotionCodeContractError(
            "projected-head and motion-code batch dimensions must match exactly"
        )
    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise PrivilegedMotionCodeContractError("block_index must be an integer")
    if not 0 <= block_index < MOTION_BLOCKS:
        raise PrivilegedMotionCodeContractError("block_index must lie in [0,15]")
    if type(audit_digests) is not bool:
        raise PrivilegedMotionCodeContractError("audit_digests must be a boolean")

    target_query = query_phase_ids >= 0
    safe_phase_ids = query_phase_ids.clamp_min(0)
    phase_gate = motion_code.phase_gates.index_select(1, safe_phase_ids)
    head_gate = motion_code.block_head_gates[:, block_index, None, :]
    effective_raw = 0.5 * (phase_gate[:, :, None] + head_gate)
    active_query = query_phase_ids > 0
    effective = torch.where(
        active_query[None, :, None],
        effective_raw,
        torch.zeros_like(effective_raw),
    )
    if bool((effective.abs() > 1.0).any().item()):
        raise PrivilegedMotionCodeContractError("effective gate escaped [-1,1]")
    disabled_query = ~active_query
    if bool(disabled_query.any().item()):
        _require_positive_zero(
            "source/phase-0 effective head gates", effective[:, disabled_query]
        )

    gate_field = effective[:, :, :, None]
    gated_raw = projected_motion_heads * gate_field
    # ``where`` canonicalizes disabled queries and zero gates to positive zero.
    # The straight-through value correction preserves the derivative of the
    # multiplication at gate==0 for target phases 1..20; a plain ``where``
    # would deadlock both code families at canonical zero initialization.
    active_gate = active_query[None, :, None, None] & (gate_field != 0)
    canonicalized = torch.where(
        active_gate,
        gated_raw,
        torch.zeros_like(gated_raw),
    )
    gated = gated_raw + (canonicalized - gated_raw).detach()
    gated = gated.contiguous()
    _require_fp32_finite("gated projected motion heads", gated)
    if bool((~target_query).any().item()):
        _require_positive_zero(
            "source projected-head output", gated[:, ~target_query]
        )
    phase_zero_query = query_phase_ids == 0
    if bool(phase_zero_query.any().item()):
        _require_positive_zero(
            "target phase-0 projected-head output", gated[:, phase_zero_query]
        )

    return ProjectedHeadGatingResult(
        projected_motion_heads_fp32=gated,
        effective_head_gates=effective,
        query_phase_ids=query_phase_ids.detach().clone(),
        block_index=block_index,
        input_heads_sha256=(
            _tensor_sha256(projected_motion_heads) if audit_digests else None
        ),
        code_sha256=(
            _tensor_sha256(motion_code.flattened()) if audit_digests else None
        ),
        audit_digests=audit_digests,
    )


def permute_motion_code_phases(
    code: MotionCode,
    phase_indices: tuple[int, ...],
) -> MotionCode:
    """Permute only temporal gates; block/head routing remains unchanged."""

    code.validate()
    if phase_indices not in (REVERSE_PHASE_INDICES, SHUFFLE_PHASE_INDICES):
        raise PrivilegedMotionCodeContractError(
            "only the frozen reverse and shuffle controls are permitted"
        )
    index = torch.tensor(phase_indices, dtype=torch.long, device=code.phase_gates.device)
    phase = code.phase_gates.index_select(1, index)
    # Recreate phase zero to canonicalize its bytes even for an adversarial view.
    phase = torch.cat((torch.zeros_like(phase[:, :1]), phase[:, 1:]), dim=1)
    return MotionCode(phase, code.block_head_gates.clone())


class AmortizedMotionCodePredictor(nn.Module):
    """Small phase-aware predictor with no privileged inference input.

    ``source_descriptor`` retains the 21-phase axis as ``[B,21,source_dim]``.
    The phase head therefore can respond to reverse/shuffle controls rather than
    trying to recover temporal order from a prematurely pooled descriptor.  A
    pooled source summary is used only for block/head routing.
    """

    def __init__(self, source_dim: int, text_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        for name, value in (
            ("source_dim", source_dim),
            ("text_dim", text_dim),
            ("hidden_dim", hidden_dim),
        ):
            minimum = 2 if name in ("source_dim", "text_dim") else 1
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise PrivilegedMotionCodeContractError(
                    f"{name} must be an integer >= {minimum}"
                )
        self.source_dim = source_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.source_norm = nn.LayerNorm(source_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.phase_network = nn.Sequential(
            nn.Linear(source_dim + text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.block_head_network = nn.Sequential(
            nn.Linear(source_dim + text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, MOTION_BLOCKS * ATTENTION_HEADS),
        )
        # Canonical zero is a safe initial inference behavior while both phase
        # and block/head outputs still receive gradients from code-space losses.
        for output_layer in (self.phase_network[-1], self.block_head_network[-1]):
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def _validate_descriptor(
        self, name: str, value: torch.Tensor, expected_dim: int
    ) -> None:
        _require_fp32_finite(name, value)
        if name == "source_descriptor":
            valid_shape = (
                value.ndim == 3
                and int(value.shape[1]) == LATENT_PHASES
                and int(value.shape[2]) == expected_dim
            )
            wanted = f"[B,{LATENT_PHASES},{expected_dim}]"
        else:
            valid_shape = value.ndim == 2 and int(value.shape[1]) == expected_dim
            wanted = f"[B,{expected_dim}]"
        if not valid_shape:
            raise PrivilegedMotionCodeContractError(
                f"{name} must have exact shape {wanted}"
            )
        if int(value.shape[0]) < 1:
            raise PrivilegedMotionCodeContractError(
                f"{name} batch dimension must be positive"
            )

    def forward(
        self,
        source_descriptor: torch.Tensor,
        text_descriptor: torch.Tensor,
    ) -> MotionCode:
        self._validate_descriptor(
            "source_descriptor", source_descriptor, self.source_dim
        )
        self._validate_descriptor("text_descriptor", text_descriptor, self.text_dim)
        if tuple(source_descriptor.shape[:1]) != tuple(text_descriptor.shape[:1]):
            raise PrivilegedMotionCodeContractError(
                "source and text descriptor batches must match"
            )
        if source_descriptor.device != text_descriptor.device:
            raise PrivilegedMotionCodeContractError(
                "source and text descriptors must share one device"
            )
        parameter = next(self.parameters())
        if source_descriptor.device != parameter.device:
            raise PrivilegedMotionCodeContractError(
                "descriptors and predictor parameters must share one device"
            )
        source_features = self.source_norm(source_descriptor)
        text_features = self.text_norm(text_descriptor)
        text_per_phase = text_features[:, None, :].expand(
            -1, NONZERO_PHASES, -1
        )
        phase_input = torch.cat(
            (source_features[:, 1:], text_per_phase), dim=2
        )
        phase_logits_nonzero = self.phase_network(phase_input).squeeze(-1)
        # The routing code chooses Bernini block/head capacity, while detailed
        # temporal order remains exclusively in the phase gates above.
        pooled_source = source_features[:, 1:].mean(dim=1)
        block_head_input = torch.cat((pooled_source, text_features), dim=1)
        block_head_logits = self.block_head_network(block_head_input)
        return decode_bounded_motion_code(
            phase_logits_nonzero,
            block_head_logits.reshape(
                int(block_head_logits.shape[0]), MOTION_BLOCKS, ATTENTION_HEADS
            ),
        )


@dataclass(frozen=True)
class MotionCodeLossConfig:
    huber_beta: float = 0.1
    wrong_action_margin: float = 0.2
    temporal_control_margin: float = 0.15
    cosine_weight: float = 1.0
    huber_weight: float = 1.0
    noop_weight: float = 1.0
    wrong_action_weight: float = 1.0
    reverse_weight: float = 0.5
    shuffle_weight: float = 0.5

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PrivilegedMotionCodeContractError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise PrivilegedMotionCodeContractError(
                    f"{name} must be finite and strictly positive"
                )


@dataclass(frozen=True)
class MotionCodeLossResult:
    total: torch.Tensor
    teacher_cosine: torch.Tensor
    teacher_huber: torch.Tensor
    noop_zero: torch.Tensor
    wrong_action_margin: torch.Tensor
    reverse_alignment: torch.Tensor
    reverse_sensitivity: torch.Tensor
    shuffle_alignment: torch.Tensor
    shuffle_sensitivity: torch.Tensor

    def detached_receipt(self) -> dict[str, float | str]:
        result: dict[str, float | str] = {
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": CONTRACT_RECEIPT_SHA256,
        }
        for name in (
            "total",
            "teacher_cosine",
            "teacher_huber",
            "noop_zero",
            "wrong_action_margin",
            "reverse_alignment",
            "reverse_sensitivity",
            "shuffle_alignment",
            "shuffle_sensitivity",
        ):
            value = getattr(self, name)
            if value.ndim != 0 or not bool(torch.isfinite(value).item()):
                raise PrivilegedMotionCodeContractError(
                    f"loss receipt field {name} is not a finite scalar"
                )
            result[name] = float(value.detach().item())
        return result


def _broadcast_teacher(teacher: MotionCode, batch_size: int) -> MotionCode:
    teacher.validate()
    if teacher.batch_size == batch_size:
        return teacher
    if teacher.batch_size != 1:
        raise PrivilegedMotionCodeContractError(
            "teacher batch must equal predictor batch or be a singleton prototype"
        )
    return MotionCode(
        teacher.phase_gates.expand(batch_size, -1),
        teacher.block_head_gates.expand(batch_size, -1, -1),
    )


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left, right, dim=1, eps=EPSILON)


def _alignment_loss(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    config: MotionCodeLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cosine = (1.0 - _cosine_similarity(predicted, teacher)).mean()
    huber = F.smooth_l1_loss(predicted, teacher, beta=config.huber_beta)
    return cosine, huber, config.cosine_weight * cosine + config.huber_weight * huber


def teacher_amortization_losses(
    predicted: MotionCode,
    teacher_prototype: MotionCode,
    noop_predicted: MotionCode,
    wrong_action_predicted: MotionCode,
    reverse_predicted: MotionCode,
    shuffle_predicted: MotionCode,
    *,
    config: MotionCodeLossConfig | None = None,
) -> MotionCodeLossResult:
    """Compute the complete privileged-teacher to amortized-code objective.

    Target/support tensors never enter this function; the only teacher object is
    an already-inverted compact code.  Reverse and shuffle targets are derived
    from that code using frozen phase permutations.  Margin terms ensure the
    predictor cannot minimize the objective by becoming instruction- or
    temporal-order invariant.
    """

    cfg = MotionCodeLossConfig() if config is None else config
    if not isinstance(cfg, MotionCodeLossConfig):
        raise PrivilegedMotionCodeContractError(
            "config must be a MotionCodeLossConfig"
        )
    predicted.validate()
    batch_size = predicted.batch_size
    variants = {
        "noop_predicted": noop_predicted,
        "wrong_action_predicted": wrong_action_predicted,
        "reverse_predicted": reverse_predicted,
        "shuffle_predicted": shuffle_predicted,
    }
    for name, code in variants.items():
        code.validate()
        if code.batch_size != batch_size:
            raise PrivilegedMotionCodeContractError(f"{name} batch differs")
        if code.phase_gates.device != predicted.phase_gates.device:
            raise PrivilegedMotionCodeContractError(f"{name} device differs")

    teacher = _broadcast_teacher(teacher_prototype, batch_size)
    if teacher.phase_gates.device != predicted.phase_gates.device:
        raise PrivilegedMotionCodeContractError("teacher and predictor devices differ")
    teacher_vector = teacher.flattened().detach()
    teacher_norm = teacher_vector.square().sum(dim=1).sqrt()
    if bool((teacher_norm <= EPSILON).any().item()):
        raise PrivilegedMotionCodeContractError(
            "an action teacher prototype cannot be the canonical no-op"
        )

    predicted_vector = predicted.flattened()
    teacher_cosine, teacher_huber, teacher_alignment = _alignment_loss(
        predicted_vector, teacher_vector, cfg
    )
    noop_zero = F.smooth_l1_loss(
        noop_predicted.flattened(),
        torch.zeros_like(noop_predicted.flattened()),
        beta=cfg.huber_beta,
    )

    correct_similarity = _cosine_similarity(predicted_vector, teacher_vector)
    wrong_similarity = _cosine_similarity(
        wrong_action_predicted.flattened(), teacher_vector
    )
    wrong_margin = F.relu(
        cfg.wrong_action_margin - correct_similarity + wrong_similarity
    ).mean()

    reverse_teacher = permute_motion_code_phases(teacher, REVERSE_PHASE_INDICES)
    shuffle_teacher = permute_motion_code_phases(teacher, SHUFFLE_PHASE_INDICES)
    reverse_vector = reverse_predicted.flattened()
    shuffle_vector = shuffle_predicted.flattened()
    reverse_target_vector = reverse_teacher.flattened().detach()
    shuffle_target_vector = shuffle_teacher.flattened().detach()

    reverse_cosine, reverse_huber, reverse_alignment = _alignment_loss(
        reverse_vector, reverse_target_vector, cfg
    )
    shuffle_cosine, shuffle_huber, shuffle_alignment = _alignment_loss(
        shuffle_vector, shuffle_target_vector, cfg
    )
    # The alignment scalars above include both cosine and Huber; keep names local
    # to make the returned receipt compact while retaining differentiability.
    del reverse_cosine, reverse_huber, shuffle_cosine, shuffle_huber

    reverse_sensitivity = F.relu(
        cfg.temporal_control_margin
        + _cosine_similarity(reverse_vector, teacher_vector)
        - _cosine_similarity(reverse_vector, reverse_target_vector)
    ).mean()
    shuffle_sensitivity = F.relu(
        cfg.temporal_control_margin
        + _cosine_similarity(shuffle_vector, teacher_vector)
        - _cosine_similarity(shuffle_vector, shuffle_target_vector)
    ).mean()

    total = (
        teacher_alignment
        + cfg.noop_weight * noop_zero
        + cfg.wrong_action_weight * wrong_margin
        + cfg.reverse_weight * (reverse_alignment + reverse_sensitivity)
        + cfg.shuffle_weight * (shuffle_alignment + shuffle_sensitivity)
    )
    if total.ndim != 0 or not bool(torch.isfinite(total).item()):
        raise PrivilegedMotionCodeContractError("motion-code total loss is invalid")
    return MotionCodeLossResult(
        total=total,
        teacher_cosine=teacher_cosine,
        teacher_huber=teacher_huber,
        noop_zero=noop_zero,
        wrong_action_margin=wrong_margin,
        reverse_alignment=reverse_alignment,
        reverse_sensitivity=reverse_sensitivity,
        shuffle_alignment=shuffle_alignment,
        shuffle_sensitivity=shuffle_sensitivity,
    )


def _contract_payload() -> dict[str, Any]:
    forward_parameters = tuple(
        name
        for name in inspect.signature(AmortizedMotionCodePredictor.forward).parameters
        if name != "self"
    )
    if forward_parameters != INFERENCE_ARGUMENTS:
        raise RuntimeError("predictor inference signature drifted")
    return {
        "method": METHOD_NAME,
        "schema_version": SCHEMA_VERSION,
        "tensor_core": "pure_pytorch_no_renderer",
        "latent_phases": LATENT_PHASES,
        "motion_blocks": MOTION_BLOCKS,
        "attention_heads": ATTENTION_HEADS,
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN_SIZE,
        "gating_input_shape": ["B", "Q", ATTENTION_HEADS, HEAD_DIM],
        "query_phase_ids_shape": ["Q"],
        "query_phase_semantics": {
            "source": -1,
            "target_min": 0,
            "target_max": LATENT_PHASES - 1,
        },
        "gating_point": "post_attention_projected_heads_before_output_merge",
        "preprojection_1536_channel_chunk_gating_forbidden": True,
        "code_shape": {
            "phase_gates": [LATENT_PHASES],
            "block_head_gates": [MOTION_BLOCKS, ATTENTION_HEADS],
        },
        "bounded_parameterization": {
            "function": "tanh",
            "phase_gate_range": [-PHASE_GATE_BOUND, PHASE_GATE_BOUND],
            "block_head_gate_range": [
                -BLOCK_HEAD_GATE_BOUND,
                BLOCK_HEAD_GATE_BOUND,
            ],
            "effective_gate": "0.5*(phase_gate+block_head_gate)",
        },
        "phase0": "byte_exact_positive_zero_and_not_parameterized",
        "source_query_output": "byte_exact_positive_zero",
        "target_phase0_output": "byte_exact_positive_zero",
        "canonical_noop": "all_code_fields_byte_exact_positive_zero",
        "coordinate_input_or_claim": False,
        "prototype": {
            "training_only": True,
            "k1": "single_support_identity",
            "k2": "exact_arithmetic_midpoint",
            "k_ge_3": "spatial_huber_irls",
            "huber_delta": PROTOTYPE_HUBER_DELTA,
            "max_iterations": PROTOTYPE_MAX_ITERATIONS,
            "tolerance": PROTOTYPE_TOLERANCE,
        },
        "inference_arguments": list(forward_parameters),
        "source_descriptor_shape": "[B,21,source_dim]",
        "text_descriptor_shape": "[B,text_dim]",
        "descriptor_minimum_width": 2,
        "temporal_order_preserved_until_phase_prediction": True,
        "target_available_at_inference": False,
        "support_available_at_inference": False,
        "forbidden_inference_conditions": list(FORBIDDEN_INFERENCE_ARGUMENTS),
    }


def build_contract_receipt() -> dict[str, Any]:
    """Return the canonical immutable EPMC core receipt."""

    payload = _contract_payload()
    return {**payload, "receipt_sha256": _canonical_json_sha256(payload)}


def validate_contract_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless ``receipt`` exactly matches this implementation."""

    if not isinstance(receipt, Mapping):
        raise PrivilegedMotionCodeContractError("receipt must be a mapping")
    candidate = dict(receipt)
    digest = candidate.pop("receipt_sha256", None)
    if not isinstance(digest, str) or len(digest) != 64:
        raise PrivilegedMotionCodeContractError("receipt digest is missing or malformed")
    if _canonical_json_sha256(candidate) != digest:
        raise PrivilegedMotionCodeContractError("receipt digest does not match payload")
    expected = build_contract_receipt()
    if dict(receipt) != expected:
        raise PrivilegedMotionCodeContractError(
            "receipt does not match the frozen EPMC inference/training contract"
        )
    return expected


CONTRACT_RECEIPT_SHA256 = (
    "c60906c535f35e785df0d107b7994c5dd24cb62918bfa5686e9db0f3105d4a5b"
)
CONTRACT_RECEIPT = build_contract_receipt()
if CONTRACT_RECEIPT["receipt_sha256"] != CONTRACT_RECEIPT_SHA256:
    raise RuntimeError("the frozen EPMC contract receipt digest is inconsistent")


__all__ = [
    "ATTENTION_HEADS",
    "AmortizedMotionCodePredictor",
    "BLOCK_HEAD_GATE_BOUND",
    "CONTRACT_RECEIPT",
    "CONTRACT_RECEIPT_SHA256",
    "FORBIDDEN_INFERENCE_ARGUMENTS",
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "INFERENCE_ARGUMENTS",
    "KShotPrototypeResult",
    "LATENT_PHASES",
    "LearnableEpisodicMotionCode",
    "METHOD_NAME",
    "MOTION_BLOCKS",
    "MotionCode",
    "MotionCodeLossConfig",
    "MotionCodeLossResult",
    "ProjectedHeadGatingResult",
    "PHASE_GATE_BOUND",
    "PrivilegedMotionCodeContractError",
    "REVERSE_PHASE_INDICES",
    "SCHEMA_VERSION",
    "SHUFFLE_PHASE_INDICES",
    "build_contract_receipt",
    "build_training_support_prototype",
    "canonical_noop_motion_code",
    "decode_bounded_motion_code",
    "gate_projected_motion_heads",
    "permute_motion_code_phases",
    "teacher_amortization_losses",
    "validate_contract_receipt",
]
