"""Structured action-state reconstruction for Bernini action plans.

This module closes a gap left by cosine-only action-plan distillation.  A plan
is useful only if a frozen decoder can recover the source-relative state
transition that the editor must execute.  The reconstruction target is
deliberately *not* RGB, a VAE latent, or a generator hidden state.  It contains
camera-compensated actor/object displacement, their relative displacement,
contact distance/speed, phase state, and global onset/duration/amplitude/
terminal coordinates.  Consequently a high reconstruction score cannot be
obtained merely by copying appearance from a self-generated anchor.

The intended lifecycle is:

1. build ``q_y`` and structured state targets from a clean source/target pair;
2. fit the decoder on train groups and freeze it;
3. qualify it once on content-disjoint held-out groups;
4. train ``q_pred`` from source+instruction against stop-gradient ``q_y`` and
   the frozen structured decoder;
5. use compatible ``q_anchor`` values only in the existing contrastive loss.

``q_anchor`` is therefore rejected as a point reconstruction teacher here.
The module provides a local/mechanical qualification report; it does not issue
dataset, teacher, optimizer, or scientific authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "bernini-structured-action-state-reconstruction-v1"
RECEIPT_SCHEMA_VERSION = "bernini-action-state-point-teacher-receipt-v1"
DECODER_RECEIPT_SCHEMA_VERSION = "bernini-frozen-action-state-decoder-receipt-v1"
AUDIT_RECEIPT_SCHEMA_VERSION = "bernini-local-action-state-audit-receipt-v1"
PHASE_COUNT = 21
ACTION_WIDTH = 256

PHASE_CONTINUOUS_AXES = (
    "actor_dx",
    "actor_dy",
    "object_dx",
    "object_dy",
    "relative_dx",
    "relative_dy",
    "contact_distance",
    "speed",
)
GLOBAL_CONTINUOUS_AXES = (
    "onset",
    "duration",
    "amplitude",
    "terminal_relative_dx",
    "terminal_relative_dy",
    "completion",
)
PHASE_STATE_CLASSES = (
    "separate",
    "approach",
    "contact",
    "held",
    "released",
)


class ActionStateReconstructionError(ValueError):
    """Raised when the structured action-state contract is violated."""


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ActionStateReconstructionError("%s must be a SHA-256 hex digest" % label)
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise ActionStateReconstructionError("%s must be a SHA-256 hex digest" % label)
    if lowered == "0" * 64:
        raise ActionStateReconstructionError("%s must not be the null digest" % label)
    return lowered


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: Any, *, label: str) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise ActionStateReconstructionError("%s must be a tensor" % label)
    detached = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(detached.dtype), "shape": list(detached.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    raw = bytes(detached.view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _state_payload_sha256(target: "StructuredActionStateV1") -> str:
    validate_structured_action_state_v1(target)
    return _canonical_sha256(
        {
            name: _tensor_sha256(getattr(target, name), label=name)
            for name in (
                "phase_continuous",
                "phase_continuous_valid",
                "phase_state",
                "phase_state_valid",
                "global_continuous",
                "global_continuous_valid",
            )
        }
    )


@dataclass(frozen=True)
class ActionStatePointTeacherReceiptV1:
    """Tamper-evident binding for a clean-pair ``q_y`` point target.

    This is deliberately not a bearer string.  The target bytes, ordered
    sample IDs, split manifest, and producing artifact are all committed by
    the receipt digest.  A self-generated-anchor origin is outside this ABI.
    The receipt remains a *local integrity receipt*, not external scientific
    qualification or training authority.
    """

    schema_version: str
    role: str
    origin: str
    sample_ids: Tuple[str, ...]
    target_payload_sha256: str
    split_manifest_sha256: str
    producer_artifact_sha256: str
    receipt_sha256: str


@dataclass(frozen=True)
class FrozenActionStateDecoderReceiptV1:
    schema_version: str
    state_dict_sha256: str
    checkpoint_artifact_sha256: str
    fit_split_manifest_sha256: str
    config_sha256: str
    receipt_sha256: str


@dataclass(frozen=True)
class LocalActionStateAuditReceiptV1:
    """Content binding for diagnostics; it can never sign qualification."""

    schema_version: str
    authority: str
    sample_ids: Tuple[str, ...]
    train_group_ids: Tuple[str, ...]
    heldout_group_ids: Tuple[str, ...]
    split_manifest_sha256: str
    decoder_receipt_sha256: str
    evaluator_artifact_sha256: str
    payload_sha256: str
    receipt_sha256: str


@dataclass(frozen=True)
class PredictedActionPlanReceiptV1:
    schema_version: str
    role: str
    origin: str
    sample_ids: Tuple[str, ...]
    plan_payload_sha256: str
    predictor_artifact_sha256: str
    receipt_sha256: str


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for action-state reconstruction") from error
    return torch


def _nn() -> Any:
    return _torch().nn


@dataclass(frozen=True)
class ActionStateDecoderConfigV1:
    action_width: int = ACTION_WIDTH
    hidden_width: int = 256
    phase_count: int = PHASE_COUNT
    phase_continuous_width: int = len(PHASE_CONTINUOUS_AXES)
    phase_state_classes: int = len(PHASE_STATE_CLASSES)
    global_continuous_width: int = len(GLOBAL_CONTINUOUS_AXES)

    def validate(self) -> None:
        for name in (
            "action_width",
            "hidden_width",
            "phase_count",
            "phase_continuous_width",
            "phase_state_classes",
            "global_continuous_width",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ActionStateReconstructionError(
                    "%s must be a positive integer" % name
                )
        if self.phase_count != PHASE_COUNT:
            raise ActionStateReconstructionError("action state requires exactly 21 phases")
        if self.phase_continuous_width != len(PHASE_CONTINUOUS_AXES):
            raise ActionStateReconstructionError("phase continuous axis ABI differs")
        if self.phase_state_classes != len(PHASE_STATE_CLASSES):
            raise ActionStateReconstructionError("phase state class ABI differs")
        if self.global_continuous_width != len(GLOBAL_CONTINUOUS_AXES):
            raise ActionStateReconstructionError("global continuous axis ABI differs")


@dataclass(frozen=True)
class StructuredActionStateV1:
    """Target-derived source-relative action state.

    Valid masks are explicit because object/contact annotations can abstain.
    Invalid values never silently become zero-valued supervision.
    """

    phase_continuous: Any
    phase_continuous_valid: Any
    phase_state: Any
    phase_state_valid: Any
    global_continuous: Any
    global_continuous_valid: Any


@dataclass(frozen=True)
class ActionStatePredictionV1:
    phase_continuous: Any
    phase_state_logits: Any
    global_continuous: Any


@dataclass(frozen=True)
class ActionStateLossConfigV1:
    phase_continuous_weight: float = 1.0
    phase_state_weight: float = 0.5
    global_continuous_weight: float = 0.5
    phase_velocity_weight: float = 0.25
    smooth_l1_beta: float = 0.1

    def validate(self) -> None:
        for name in (
            "phase_continuous_weight",
            "phase_state_weight",
            "global_continuous_weight",
            "phase_velocity_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ActionStateReconstructionError(
                    "%s must be finite and non-negative" % name
                )
        if not math.isfinite(float(self.smooth_l1_beta)) or self.smooth_l1_beta <= 0.0:
            raise ActionStateReconstructionError(
                "smooth_l1_beta must be finite and positive"
            )
        if (
            self.phase_continuous_weight
            + self.phase_state_weight
            + self.global_continuous_weight
            + self.phase_velocity_weight
            <= 0.0
        ):
            raise ActionStateReconstructionError("all action-state losses are disabled")


class ActionStateDecoderV1(_nn().Module):
    """Decode a 21-phase/global action plan into structured transitions."""

    def __init__(self, config: Optional[ActionStateDecoderConfigV1] = None) -> None:
        super().__init__()
        torch = _torch()
        nn = _nn()
        self.config = config or ActionStateDecoderConfigV1()
        self.config.validate()
        width = self.config.hidden_width
        self.phase_norm = nn.LayerNorm(self.config.action_width)
        self.phase_trunk = nn.Sequential(
            nn.Linear(self.config.action_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.phase_continuous_head = nn.Linear(
            width, self.config.phase_continuous_width
        )
        self.phase_state_head = nn.Linear(width, self.config.phase_state_classes)
        self.global_norm = nn.LayerNorm(2 * self.config.action_width)
        self.global_trunk = nn.Sequential(
            nn.Linear(2 * self.config.action_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.global_head = nn.Linear(width, self.config.global_continuous_width)
        self.register_buffer(
            "_abi",
            torch.tensor(
                (
                    1,
                    self.config.phase_count,
                    self.config.action_width,
                    self.config.phase_continuous_width,
                    self.config.phase_state_classes,
                    self.config.global_continuous_width,
                ),
                dtype=torch.int64,
            ),
            persistent=True,
        )
        # This module is an evaluator/teacher head, never an optimization
        # target in the action-code training graph.  Frozen parameters still
        # propagate gradients to q_pred inputs.
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "ActionStateDecoderV1":
        if type(mode) is not bool:
            raise ActionStateReconstructionError("decoder train mode must be boolean")
        if mode:
            raise ActionStateReconstructionError(
                "ActionStateDecoderV1 is frozen; fitting happens outside this module"
            )
        return super().train(False)

    def forward(self, plan: Any) -> ActionStatePredictionV1:
        torch = _torch()
        phase = getattr(plan, "phase_tokens", None)
        global_token = getattr(plan, "global_token", None)
        _validate_plan_tensors(
            phase,
            global_token,
            action_width=self.config.action_width,
            phase_count=self.config.phase_count,
        )
        output_dtype = phase.dtype
        phase_hidden = self.phase_trunk(self.phase_norm(phase.float()))
        phase_continuous = self.phase_continuous_head(phase_hidden)
        phase_state_logits = self.phase_state_head(phase_hidden)
        pooled_phase = phase.float().mean(dim=1)
        joined = torch.cat((global_token.float(), pooled_phase), dim=-1)
        global_hidden = self.global_trunk(self.global_norm(joined))
        global_continuous = self.global_head(global_hidden)
        prediction = ActionStatePredictionV1(
            phase_continuous=phase_continuous.to(dtype=output_dtype),
            phase_state_logits=phase_state_logits.to(dtype=output_dtype),
            global_continuous=global_continuous.to(dtype=output_dtype),
        )
        _validate_prediction(prediction, batch=int(phase.shape[0]), config=self.config)
        return prediction


def _validate_plan_tensors(
    phase: Any,
    global_token: Any,
    *,
    action_width: int,
    phase_count: int,
) -> None:
    torch = _torch()
    if not isinstance(phase, torch.Tensor) or not isinstance(global_token, torch.Tensor):
        raise ActionStateReconstructionError("action plan must contain tensor values")
    if phase.ndim != 3 or global_token.ndim != 2:
        raise ActionStateReconstructionError("action plan rank differs")
    if tuple(phase.shape[:2]) != (int(global_token.shape[0]), phase_count):
        raise ActionStateReconstructionError("action plan batch/phase geometry differs")
    if int(phase.shape[2]) != action_width or int(global_token.shape[1]) != action_width:
        raise ActionStateReconstructionError("action plan width differs")
    if phase.device != global_token.device or phase.dtype != global_token.dtype:
        raise ActionStateReconstructionError("action plan tensors must share dtype/device")
    if not phase.is_floating_point() or not global_token.is_floating_point():
        raise ActionStateReconstructionError("action plan tensors must be floating point")
    if not bool(torch.isfinite(phase).all()) or not bool(torch.isfinite(global_token).all()):
        raise ActionStateReconstructionError("action plan contains non-finite values")


def _validate_prediction(
    prediction: ActionStatePredictionV1,
    *,
    batch: int,
    config: ActionStateDecoderConfigV1,
) -> None:
    torch = _torch()
    expected = (
        (batch, config.phase_count, config.phase_continuous_width),
        (batch, config.phase_count, config.phase_state_classes),
        (batch, config.global_continuous_width),
    )
    observed = (
        tuple(prediction.phase_continuous.shape),
        tuple(prediction.phase_state_logits.shape),
        tuple(prediction.global_continuous.shape),
    )
    if observed != expected:
        raise ActionStateReconstructionError("action-state prediction geometry differs")
    values = (
        prediction.phase_continuous,
        prediction.phase_state_logits,
        prediction.global_continuous,
    )
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise ActionStateReconstructionError("action-state prediction is not tensor-valued")
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ActionStateReconstructionError("action-state prediction is non-finite")


def validate_structured_action_state_v1(
    target: StructuredActionStateV1,
    *,
    batch: Optional[int] = None,
) -> None:
    torch = _torch()
    if not isinstance(target, StructuredActionStateV1):
        raise ActionStateReconstructionError("target type differs")
    if not isinstance(target.phase_continuous, torch.Tensor):
        raise ActionStateReconstructionError("phase target must be a tensor")
    observed_batch = int(target.phase_continuous.shape[0])
    if batch is not None and observed_batch != batch:
        raise ActionStateReconstructionError("target batch differs")
    expected = {
        "phase_continuous": (observed_batch, PHASE_COUNT, len(PHASE_CONTINUOUS_AXES)),
        "phase_continuous_valid": (
            observed_batch,
            PHASE_COUNT,
            len(PHASE_CONTINUOUS_AXES),
        ),
        "phase_state": (observed_batch, PHASE_COUNT),
        "phase_state_valid": (observed_batch, PHASE_COUNT),
        "global_continuous": (observed_batch, len(GLOBAL_CONTINUOUS_AXES)),
        "global_continuous_valid": (
            observed_batch,
            len(GLOBAL_CONTINUOUS_AXES),
        ),
    }
    for name, shape in expected.items():
        value = getattr(target, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ActionStateReconstructionError("%s geometry differs" % name)
    for name in ("phase_continuous_valid", "phase_state_valid", "global_continuous_valid"):
        if getattr(target, name).dtype != torch.bool:
            raise ActionStateReconstructionError("%s must be bool" % name)
    if target.phase_state.dtype != torch.long:
        raise ActionStateReconstructionError("phase_state must be int64")
    valid_states = target.phase_state[target.phase_state_valid]
    if valid_states.numel() and (
        bool((valid_states < 0).any())
        or bool((valid_states >= len(PHASE_STATE_CLASSES)).any())
    ):
        raise ActionStateReconstructionError("phase_state index is outside the ABI")
    for name in ("phase_continuous", "global_continuous"):
        value = getattr(target, name)
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ActionStateReconstructionError("%s must be finite floating point" % name)
    devices = {getattr(target, name).device for name in expected}
    if len(devices) != 1:
        raise ActionStateReconstructionError("structured target tensors must share device")


def _validated_ids(values: Any, *, label: str, expected: Optional[int] = None) -> Tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise ActionStateReconstructionError("%s must be a nonempty tuple" % label)
    if any(type(value) is not str or not value for value in values):
        raise ActionStateReconstructionError("%s contains an invalid ID" % label)
    if len(set(values)) != len(values):
        raise ActionStateReconstructionError("%s must be unique" % label)
    if expected is not None and len(values) != expected:
        raise ActionStateReconstructionError("%s batch binding differs" % label)
    return values


def _plan_payload_sha256(plan: Any) -> str:
    phase = getattr(plan, "phase_tokens", None)
    global_token = getattr(plan, "global_token", None)
    if not isinstance(phase, _torch().Tensor) or not isinstance(global_token, _torch().Tensor):
        raise ActionStateReconstructionError("q_pred plan tensors are missing")
    return _canonical_sha256(
        {
            "phase_tokens": _tensor_sha256(phase, label="phase_tokens"),
            "global_token": _tensor_sha256(global_token, label="global_token"),
        }
    )


def build_action_state_point_teacher_receipt_v1(
    *,
    target: StructuredActionStateV1,
    sample_ids: Tuple[str, ...],
    split_manifest_sha256: str,
    producer_artifact_sha256: str,
) -> ActionStatePointTeacherReceiptV1:
    """Bind a clean source/target structured-state teacher to exact bytes.

    There is intentionally no caller-selected role or origin parameter.
    Anchor-derived data has no constructor in this point-loss ABI.
    """

    validate_structured_action_state_v1(target)
    samples = _validated_ids(
        sample_ids,
        label="point-teacher sample_ids",
        expected=int(target.phase_continuous.shape[0]),
    )
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "role": "q_y_structured_point_teacher",
        "origin": "clean_source_target_pair",
        "sample_ids": list(samples),
        "target_payload_sha256": _state_payload_sha256(target),
        "split_manifest_sha256": _require_sha256(
            split_manifest_sha256, label="split_manifest_sha256"
        ),
        "producer_artifact_sha256": _require_sha256(
            producer_artifact_sha256, label="producer_artifact_sha256"
        ),
    }
    return ActionStatePointTeacherReceiptV1(
        schema_version=payload["schema_version"],
        role=payload["role"],
        origin=payload["origin"],
        sample_ids=samples,
        target_payload_sha256=payload["target_payload_sha256"],
        split_manifest_sha256=payload["split_manifest_sha256"],
        producer_artifact_sha256=payload["producer_artifact_sha256"],
        receipt_sha256=_canonical_sha256(payload),
    )


def validate_action_state_point_teacher_receipt_v1(
    receipt: ActionStatePointTeacherReceiptV1,
    *,
    target: StructuredActionStateV1,
) -> None:
    if not isinstance(receipt, ActionStatePointTeacherReceiptV1):
        raise ActionStateReconstructionError(
            "point supervision requires a structured q_y receipt, not a role string"
        )
    samples = _validated_ids(
        receipt.sample_ids,
        label="point-teacher sample_ids",
        expected=int(target.phase_continuous.shape[0]),
    )
    payload = {
        "schema_version": receipt.schema_version,
        "role": receipt.role,
        "origin": receipt.origin,
        "sample_ids": list(samples),
        "target_payload_sha256": receipt.target_payload_sha256,
        "split_manifest_sha256": receipt.split_manifest_sha256,
        "producer_artifact_sha256": receipt.producer_artifact_sha256,
    }
    if receipt.schema_version != RECEIPT_SCHEMA_VERSION:
        raise ActionStateReconstructionError("point-teacher receipt schema differs")
    if receipt.role != "q_y_structured_point_teacher" or receipt.origin != "clean_source_target_pair":
        raise ActionStateReconstructionError(
            "anchor-derived or non-q_y data cannot enter structured point loss"
        )
    for name in (
        "target_payload_sha256",
        "split_manifest_sha256",
        "producer_artifact_sha256",
        "receipt_sha256",
    ):
        _require_sha256(getattr(receipt, name), label=name)
    if receipt.target_payload_sha256 != _state_payload_sha256(target):
        raise ActionStateReconstructionError("point-teacher target bytes differ from receipt")
    if receipt.receipt_sha256 != _canonical_sha256(payload):
        raise ActionStateReconstructionError("point-teacher receipt digest differs")


def build_predicted_action_plan_receipt_v1(
    *,
    plan: Any,
    sample_ids: Tuple[str, ...],
    predictor_artifact_sha256: str,
) -> PredictedActionPlanReceiptV1:
    phase = getattr(plan, "phase_tokens", None)
    global_token = getattr(plan, "global_token", None)
    if not isinstance(phase, _torch().Tensor) or not isinstance(global_token, _torch().Tensor):
        raise ActionStateReconstructionError("q_pred plan tensors are missing")
    samples = _validated_ids(sample_ids, label="q_pred sample_ids", expected=int(phase.shape[0]))
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "role": "q_pred_student",
        "origin": "source_instruction_only",
        "sample_ids": list(samples),
        "plan_payload_sha256": _plan_payload_sha256(plan),
        "predictor_artifact_sha256": _require_sha256(
            predictor_artifact_sha256, label="predictor_artifact_sha256"
        ),
    }
    return PredictedActionPlanReceiptV1(
        schema_version=payload["schema_version"],
        role=payload["role"],
        origin=payload["origin"],
        sample_ids=samples,
        plan_payload_sha256=payload["plan_payload_sha256"],
        predictor_artifact_sha256=payload["predictor_artifact_sha256"],
        receipt_sha256=_canonical_sha256(payload),
    )


def validate_predicted_action_plan_receipt_v1(
    receipt: PredictedActionPlanReceiptV1,
    *,
    plan: Any,
    expected_sample_ids: Tuple[str, ...],
) -> None:
    if not isinstance(receipt, PredictedActionPlanReceiptV1):
        raise ActionStateReconstructionError("q_pred requires a structured provenance receipt")
    samples = _validated_ids(
        receipt.sample_ids,
        label="q_pred sample_ids",
        expected=len(expected_sample_ids),
    )
    payload = {
        "schema_version": receipt.schema_version,
        "role": receipt.role,
        "origin": receipt.origin,
        "sample_ids": list(samples),
        "plan_payload_sha256": receipt.plan_payload_sha256,
        "predictor_artifact_sha256": receipt.predictor_artifact_sha256,
    }
    if receipt.schema_version != RECEIPT_SCHEMA_VERSION:
        raise ActionStateReconstructionError("q_pred receipt schema differs")
    if receipt.role != "q_pred_student" or receipt.origin != "source_instruction_only":
        raise ActionStateReconstructionError("anchor-derived plan cannot enter point loss")
    if samples != expected_sample_ids:
        raise ActionStateReconstructionError("q_pred/target sample order differs")
    for name in ("plan_payload_sha256", "predictor_artifact_sha256", "receipt_sha256"):
        _require_sha256(getattr(receipt, name), label=name)
    if receipt.plan_payload_sha256 != _plan_payload_sha256(plan):
        raise ActionStateReconstructionError("q_pred plan bytes differ from receipt")
    if receipt.receipt_sha256 != _canonical_sha256(payload):
        raise ActionStateReconstructionError("q_pred receipt digest differs")


def _decoder_state_dict_sha256(decoder: ActionStateDecoderV1) -> str:
    return _canonical_sha256(
        {
            name: _tensor_sha256(value, label="decoder.%s" % name)
            for name, value in sorted(decoder.state_dict().items())
        }
    )


def bind_frozen_action_state_decoder_v1(
    *,
    decoder: ActionStateDecoderV1,
    checkpoint_artifact_sha256: str,
    fit_split_manifest_sha256: str,
) -> FrozenActionStateDecoderReceiptV1:
    if not isinstance(decoder, ActionStateDecoderV1):
        raise ActionStateReconstructionError("decoder type differs")
    decoder.requires_grad_(False)
    decoder.eval()
    payload = {
        "schema_version": DECODER_RECEIPT_SCHEMA_VERSION,
        "state_dict_sha256": _decoder_state_dict_sha256(decoder),
        "checkpoint_artifact_sha256": _require_sha256(
            checkpoint_artifact_sha256, label="checkpoint_artifact_sha256"
        ),
        "fit_split_manifest_sha256": _require_sha256(
            fit_split_manifest_sha256, label="fit_split_manifest_sha256"
        ),
        "config_sha256": _canonical_sha256(asdict(decoder.config)),
    }
    return FrozenActionStateDecoderReceiptV1(
        **payload,
        receipt_sha256=_canonical_sha256(payload),
    )


def validate_frozen_action_state_decoder_v1(
    decoder: ActionStateDecoderV1,
    receipt: FrozenActionStateDecoderReceiptV1,
) -> None:
    if not isinstance(decoder, ActionStateDecoderV1) or not isinstance(
        receipt, FrozenActionStateDecoderReceiptV1
    ):
        raise ActionStateReconstructionError("frozen decoder/receipt type differs")
    if decoder.training:
        raise ActionStateReconstructionError("frozen decoder must remain in eval mode")
    if any(parameter.requires_grad for parameter in decoder.parameters()):
        raise ActionStateReconstructionError("frozen decoder parameters require gradients")
    payload = {
        "schema_version": receipt.schema_version,
        "state_dict_sha256": receipt.state_dict_sha256,
        "checkpoint_artifact_sha256": receipt.checkpoint_artifact_sha256,
        "fit_split_manifest_sha256": receipt.fit_split_manifest_sha256,
        "config_sha256": receipt.config_sha256,
    }
    if receipt.schema_version != DECODER_RECEIPT_SCHEMA_VERSION:
        raise ActionStateReconstructionError("decoder receipt schema differs")
    for name in (
        "state_dict_sha256",
        "checkpoint_artifact_sha256",
        "fit_split_manifest_sha256",
        "config_sha256",
        "receipt_sha256",
    ):
        _require_sha256(getattr(receipt, name), label=name)
    if receipt.state_dict_sha256 != _decoder_state_dict_sha256(decoder):
        raise ActionStateReconstructionError("decoder state differs from checkpoint receipt")
    if receipt.config_sha256 != _canonical_sha256(asdict(decoder.config)):
        raise ActionStateReconstructionError("decoder config differs from receipt")
    if receipt.receipt_sha256 != _canonical_sha256(payload):
        raise ActionStateReconstructionError("decoder receipt digest differs")


def _masked_smooth_l1(prediction: Any, target: Any, valid: Any, beta: float) -> Any:
    torch = _torch()
    if not bool(valid.any()):
        return prediction.float().sum() * 0.0
    loss = torch.nn.functional.smooth_l1_loss(
        prediction.float(), target.detach().float(), reduction="none", beta=float(beta)
    )
    weights = valid.float()
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def action_state_reconstruction_loss_v1(
    *,
    decoder: ActionStateDecoderV1,
    decoder_receipt: FrozenActionStateDecoderReceiptV1,
    plan: Any,
    plan_receipt: PredictedActionPlanReceiptV1,
    target: StructuredActionStateV1,
    target_receipt: ActionStatePointTeacherReceiptV1,
    config: Optional[ActionStateLossConfigV1] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Decode q_pred with a frozen head against receipt-bound clean-pair state."""

    torch = _torch()
    cfg = config or ActionStateLossConfigV1()
    cfg.validate()
    validate_structured_action_state_v1(target)
    validate_action_state_point_teacher_receipt_v1(target_receipt, target=target)
    validate_predicted_action_plan_receipt_v1(
        plan_receipt,
        plan=plan,
        expected_sample_ids=target_receipt.sample_ids,
    )
    validate_frozen_action_state_decoder_v1(decoder, decoder_receipt)
    prediction = decoder(plan)
    _validate_prediction(
        prediction,
        batch=int(target.phase_continuous.shape[0]),
        config=decoder.config,
    )
    phase = _masked_smooth_l1(
        prediction.phase_continuous,
        target.phase_continuous,
        target.phase_continuous_valid,
        cfg.smooth_l1_beta,
    )
    global_loss = _masked_smooth_l1(
        prediction.global_continuous,
        target.global_continuous,
        target.global_continuous_valid,
        cfg.smooth_l1_beta,
    )
    if bool(target.phase_state_valid.any()):
        logits = prediction.phase_state_logits[target.phase_state_valid].float()
        labels = target.phase_state[target.phase_state_valid].detach()
        state = torch.nn.functional.cross_entropy(logits, labels)
    else:
        state = prediction.phase_state_logits.float().sum() * 0.0
    velocity_valid = (
        target.phase_continuous_valid[:, 1:]
        & target.phase_continuous_valid[:, :-1]
    )
    predicted_velocity = (
        prediction.phase_continuous[:, 1:].float()
        - prediction.phase_continuous[:, :-1].float()
    )
    target_velocity = (
        target.phase_continuous[:, 1:].detach().float()
        - target.phase_continuous[:, :-1].detach().float()
    )
    velocity = _masked_smooth_l1(
        predicted_velocity,
        target_velocity,
        velocity_valid,
        cfg.smooth_l1_beta,
    )
    total = (
        cfg.phase_continuous_weight * phase
        + cfg.phase_state_weight * state
        + cfg.global_continuous_weight * global_loss
        + cfg.phase_velocity_weight * velocity
    )
    return total, {
        "phase_continuous": phase,
        "phase_state": state,
        "global_continuous": global_loss,
        "phase_velocity": velocity,
        "point_teacher_receipt_role": target_receipt.role,
        "point_teacher_receipt_sha256": target_receipt.receipt_sha256,
        "q_pred_receipt_sha256": plan_receipt.receipt_sha256,
        "frozen_decoder_receipt_sha256": decoder_receipt.receipt_sha256,
        "decoder_all_parameters_frozen": True,
        "q_anchor_point_loss_used": False,
        "local_mechanical_loss_only": True,
        "optimizer_authorized": False,
    }


def masked_axis_r2_v1(
    prediction: Any,
    target: Any,
    valid: Any,
    *,
    axis_names: Sequence[str],
    minimum_observations: int = 8,
    variance_epsilon: float = 1.0e-8,
) -> Dict[str, Any]:
    """Compute per-axis and variance-weighted held-out coefficient of determination."""

    torch = _torch()
    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or not isinstance(valid, torch.Tensor)
        or tuple(prediction.shape) != tuple(target.shape)
        or tuple(valid.shape) != tuple(target.shape)
        or valid.dtype != torch.bool
        or prediction.ndim < 2
        or int(prediction.shape[-1]) != len(axis_names)
    ):
        raise ActionStateReconstructionError("R2 tensors/axis names differ")
    if minimum_observations < 2:
        raise ActionStateReconstructionError("minimum_observations must be at least two")
    if not math.isfinite(variance_epsilon) or variance_epsilon <= 0.0:
        raise ActionStateReconstructionError("variance_epsilon must be positive")
    result: Dict[str, Any] = {}
    total_sse = 0.0
    total_sst = 0.0
    estimable = 0
    for index, name in enumerate(axis_names):
        mask = valid[..., index]
        count = int(mask.sum().item())
        if count < minimum_observations:
            result[str(name)] = {
                "status": "abstain_insufficient_observations",
                "observations": count,
                "r2": None,
                "sse": None,
                "sst": None,
            }
            continue
        observed = target[..., index][mask].detach().double()
        estimated = prediction[..., index][mask].detach().double()
        if not bool(torch.isfinite(observed).all()) or not bool(
            torch.isfinite(estimated).all()
        ):
            raise ActionStateReconstructionError("R2 input is non-finite")
        mean = observed.mean()
        sst = float(((observed - mean) ** 2).sum().item())
        sse = float(((estimated - observed) ** 2).sum().item())
        if sst <= variance_epsilon:
            result[str(name)] = {
                "status": "abstain_no_heldout_variance",
                "observations": count,
                "r2": None,
                "sse": sse,
                "sst": sst,
            }
            continue
        r2 = 1.0 - sse / sst
        result[str(name)] = {
            "status": "estimable",
            "observations": count,
            "r2": r2,
            "sse": sse,
            "sst": sst,
        }
        total_sse += sse
        total_sst += sst
        estimable += 1
    result["summary"] = {
        "estimable_axes": estimable,
        "total_axes": len(axis_names),
        "variance_weighted_r2": (
            1.0 - total_sse / total_sst if total_sst > variance_epsilon else None
        ),
        "total_sse": total_sse,
        "total_sst": total_sst,
    }
    return result


def phase_state_balanced_accuracy_v1(
    logits: Any,
    target: Any,
    valid: Any,
    *,
    minimum_observations_per_class: int = 8,
) -> Dict[str, Any]:
    torch = _torch()
    if type(minimum_observations_per_class) is not int or minimum_observations_per_class < 2:
        raise ActionStateReconstructionError(
            "minimum_observations_per_class must be an integer >= 2"
        )
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or int(logits.shape[-1]) != len(PHASE_STATE_CLASSES)
        or not isinstance(target, torch.Tensor)
        or tuple(target.shape) != tuple(logits.shape[:2])
        or not isinstance(valid, torch.Tensor)
        or tuple(valid.shape) != tuple(target.shape)
        or valid.dtype != torch.bool
    ):
        raise ActionStateReconstructionError("phase-state audit geometry differs")
    predicted = logits.detach().argmax(dim=-1)
    recalls: Dict[str, Optional[float]] = {}
    observations: Dict[str, int] = {}
    estimable = []
    for index, name in enumerate(PHASE_STATE_CLASSES):
        mask = valid & (target == index)
        count = int(mask.sum().item())
        observations[name] = count
        if count < minimum_observations_per_class:
            recalls[name] = None
            continue
        recall = float((predicted[mask] == index).float().mean().item())
        recalls[name] = recall
        estimable.append(recall)
    return {
        "balanced_accuracy": (
            float(sum(estimable) / len(estimable)) if estimable else None
        ),
        "estimable_classes": len(estimable),
        "total_classes": len(PHASE_STATE_CLASSES),
        "minimum_observations_per_class": minimum_observations_per_class,
        "class_observations": observations,
        "class_recalls": recalls,
    }


def effective_rank_v1(codes: Any, *, epsilon: float = 1.0e-12) -> float:
    torch = _torch()
    if not isinstance(codes, torch.Tensor) or codes.ndim != 2:
        raise ActionStateReconstructionError("action codes must be [N,D]")
    if int(codes.shape[0]) < 2 or int(codes.shape[1]) < 1:
        raise ActionStateReconstructionError("action code matrix is too small")
    values = codes.detach().double()
    if not bool(torch.isfinite(values).all()):
        raise ActionStateReconstructionError("action codes are non-finite")
    centered = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = float(energy.sum().item())
    if total <= epsilon:
        return 0.0
    probability = energy / energy.sum()
    probability = probability[probability > epsilon]
    entropy = -(probability * probability.log()).sum()
    return float(torch.exp(entropy).item())


@dataclass(frozen=True)
class ActionRepresentationGateConfigV1:
    minimum_phase_variance_weighted_r2: float = 0.60
    minimum_global_variance_weighted_r2: float = 0.60
    minimum_critical_axis_r2: float = 0.40
    minimum_state_balanced_accuracy: float = 0.60
    minimum_effective_rank: float = 8.0
    minimum_over_instruction_centroid_r2: float = 0.10
    minimum_over_source_only_r2: float = 0.10
    minimum_over_within_family_shuffle_r2: float = 0.10
    maximum_abs_appearance_correlation: float = 0.20

    def validate(self) -> None:
        for name, value in asdict(self).items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ActionStateReconstructionError("%s must be finite" % name)
        if self.minimum_effective_rank <= 0.0:
            raise ActionStateReconstructionError("minimum_effective_rank must be positive")
        if not 0.0 <= self.maximum_abs_appearance_correlation <= 1.0:
            raise ActionStateReconstructionError(
                "maximum_abs_appearance_correlation must lie in [0,1]"
            )


def _audit_tensor_payload_sha256(
    *,
    phase_prediction: Any,
    phase_target: Any,
    phase_valid: Any,
    global_prediction: Any,
    global_target: Any,
    global_valid: Any,
    phase_state_logits: Any,
    phase_state_target: Any,
    phase_state_valid: Any,
    action_codes: Any,
    instruction_centroid_r2: float,
    source_only_r2: float,
    within_family_shuffle_r2: float,
    appearance_correlation: float,
) -> str:
    tensors = {
        "phase_prediction": phase_prediction,
        "phase_target": phase_target,
        "phase_valid": phase_valid,
        "global_prediction": global_prediction,
        "global_target": global_target,
        "global_valid": global_valid,
        "phase_state_logits": phase_state_logits,
        "phase_state_target": phase_state_target,
        "phase_state_valid": phase_state_valid,
        "action_codes": action_codes,
    }
    payload: Dict[str, Any] = {
        name: _tensor_sha256(value, label="audit.%s" % name)
        for name, value in tensors.items()
    }
    for name, value in (
        ("instruction_centroid_r2", instruction_centroid_r2),
        ("source_only_r2", source_only_r2),
        ("within_family_shuffle_r2", within_family_shuffle_r2),
        ("appearance_correlation", appearance_correlation),
    ):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ActionStateReconstructionError("%s must be finite" % name)
        # Decimal repr is deterministic for a Python float and prevents a
        # caller from swapping unbound scalar baselines after receipt creation.
        payload[name] = repr(numeric)
    return _canonical_sha256(payload)


def build_local_action_state_audit_receipt_v1(
    *,
    sample_ids: Tuple[str, ...],
    train_group_ids: Tuple[str, ...],
    heldout_group_ids: Tuple[str, ...],
    split_manifest_sha256: str,
    decoder_receipt_sha256: str,
    evaluator_artifact_sha256: str,
    phase_prediction: Any,
    phase_target: Any,
    phase_valid: Any,
    global_prediction: Any,
    global_target: Any,
    global_valid: Any,
    phase_state_logits: Any,
    phase_state_target: Any,
    phase_state_valid: Any,
    action_codes: Any,
    instruction_centroid_r2: float,
    source_only_r2: float,
    within_family_shuffle_r2: float,
    appearance_correlation: float,
) -> LocalActionStateAuditReceiptV1:
    batch = int(phase_target.shape[0])
    samples = _validated_ids(sample_ids, label="audit sample_ids", expected=batch)
    train = _validated_ids(train_group_ids, label="audit train_group_ids")
    heldout = _validated_ids(heldout_group_ids, label="audit heldout_group_ids")
    if set(train).intersection(heldout):
        raise ActionStateReconstructionError("audit split is not content-disjoint")
    payload = {
        "schema_version": AUDIT_RECEIPT_SCHEMA_VERSION,
        "authority": "local_untrusted_diagnostic_only",
        "sample_ids": list(samples),
        "train_group_ids": list(train),
        "heldout_group_ids": list(heldout),
        "split_manifest_sha256": _require_sha256(
            split_manifest_sha256, label="split_manifest_sha256"
        ),
        "decoder_receipt_sha256": _require_sha256(
            decoder_receipt_sha256, label="decoder_receipt_sha256"
        ),
        "evaluator_artifact_sha256": _require_sha256(
            evaluator_artifact_sha256, label="evaluator_artifact_sha256"
        ),
        "payload_sha256": _audit_tensor_payload_sha256(
            phase_prediction=phase_prediction,
            phase_target=phase_target,
            phase_valid=phase_valid,
            global_prediction=global_prediction,
            global_target=global_target,
            global_valid=global_valid,
            phase_state_logits=phase_state_logits,
            phase_state_target=phase_state_target,
            phase_state_valid=phase_state_valid,
            action_codes=action_codes,
            instruction_centroid_r2=instruction_centroid_r2,
            source_only_r2=source_only_r2,
            within_family_shuffle_r2=within_family_shuffle_r2,
            appearance_correlation=appearance_correlation,
        ),
    }
    return LocalActionStateAuditReceiptV1(
        schema_version=payload["schema_version"],
        authority=payload["authority"],
        sample_ids=samples,
        train_group_ids=train,
        heldout_group_ids=heldout,
        split_manifest_sha256=payload["split_manifest_sha256"],
        decoder_receipt_sha256=payload["decoder_receipt_sha256"],
        evaluator_artifact_sha256=payload["evaluator_artifact_sha256"],
        payload_sha256=payload["payload_sha256"],
        receipt_sha256=_canonical_sha256(payload),
    )


def _validate_local_audit_receipt_v1(
    receipt: LocalActionStateAuditReceiptV1,
    *,
    expected_batch: int,
    observed_payload_sha256: str,
) -> None:
    if not isinstance(receipt, LocalActionStateAuditReceiptV1):
        raise ActionStateReconstructionError(
            "heldout diagnostics require a structured content-bound receipt"
        )
    samples = _validated_ids(receipt.sample_ids, label="audit sample_ids", expected=expected_batch)
    train = _validated_ids(receipt.train_group_ids, label="audit train_group_ids")
    heldout = _validated_ids(receipt.heldout_group_ids, label="audit heldout_group_ids")
    if set(train).intersection(heldout):
        raise ActionStateReconstructionError("audit split is not content-disjoint")
    payload = {
        "schema_version": receipt.schema_version,
        "authority": receipt.authority,
        "sample_ids": list(samples),
        "train_group_ids": list(train),
        "heldout_group_ids": list(heldout),
        "split_manifest_sha256": receipt.split_manifest_sha256,
        "decoder_receipt_sha256": receipt.decoder_receipt_sha256,
        "evaluator_artifact_sha256": receipt.evaluator_artifact_sha256,
        "payload_sha256": receipt.payload_sha256,
    }
    if receipt.schema_version != AUDIT_RECEIPT_SCHEMA_VERSION:
        raise ActionStateReconstructionError("audit receipt schema differs")
    if receipt.authority != "local_untrusted_diagnostic_only":
        raise ActionStateReconstructionError("this module cannot accept qualification authority")
    for name in (
        "split_manifest_sha256",
        "decoder_receipt_sha256",
        "evaluator_artifact_sha256",
        "payload_sha256",
        "receipt_sha256",
    ):
        _require_sha256(getattr(receipt, name), label=name)
    if receipt.payload_sha256 != observed_payload_sha256:
        raise ActionStateReconstructionError("audit tensors/scalars differ from receipt")
    if receipt.receipt_sha256 != _canonical_sha256(payload):
        raise ActionStateReconstructionError("audit receipt digest differs")


def build_action_representation_audit_v1(
    *,
    phase_prediction: Any,
    phase_target: Any,
    phase_valid: Any,
    global_prediction: Any,
    global_target: Any,
    global_valid: Any,
    phase_state_logits: Any,
    phase_state_target: Any,
    phase_state_valid: Any,
    action_codes: Any,
    instruction_centroid_r2: float,
    source_only_r2: float,
    within_family_shuffle_r2: float,
    appearance_correlation: float,
    audit_receipt: LocalActionStateAuditReceiptV1,
    config: Optional[ActionRepresentationGateConfigV1] = None,
) -> Dict[str, Any]:
    """Compute receipt-bound diagnostics without issuing qualification."""

    cfg = config or ActionRepresentationGateConfigV1()
    cfg.validate()
    batch = int(phase_target.shape[0])
    payload_sha256 = _audit_tensor_payload_sha256(
        phase_prediction=phase_prediction,
        phase_target=phase_target,
        phase_valid=phase_valid,
        global_prediction=global_prediction,
        global_target=global_target,
        global_valid=global_valid,
        phase_state_logits=phase_state_logits,
        phase_state_target=phase_state_target,
        phase_state_valid=phase_state_valid,
        action_codes=action_codes,
        instruction_centroid_r2=instruction_centroid_r2,
        source_only_r2=source_only_r2,
        within_family_shuffle_r2=within_family_shuffle_r2,
        appearance_correlation=appearance_correlation,
    )
    _validate_local_audit_receipt_v1(
        audit_receipt,
        expected_batch=batch,
        observed_payload_sha256=payload_sha256,
    )
    if int(action_codes.shape[0]) != batch:
        raise ActionStateReconstructionError("action-code/audit batch differs")
    phase = masked_axis_r2_v1(
        phase_prediction,
        phase_target,
        phase_valid,
        axis_names=PHASE_CONTINUOUS_AXES,
    )
    global_result = masked_axis_r2_v1(
        global_prediction,
        global_target,
        global_valid,
        axis_names=GLOBAL_CONTINUOUS_AXES,
    )
    state = phase_state_balanced_accuracy_v1(
        phase_state_logits, phase_state_target, phase_state_valid
    )
    rank = effective_rank_v1(action_codes)
    phase_r2 = phase["summary"]["variance_weighted_r2"]
    global_r2 = global_result["summary"]["variance_weighted_r2"]
    critical = ("actor_dx", "actor_dy", "object_dx", "object_dy", "relative_dx", "relative_dy")
    critical_values = [
        phase[name]["r2"]
        for name in critical
        if phase[name]["status"] == "estimable"
    ]
    critical_min = min(critical_values) if len(critical_values) == len(critical) else None
    checks = {
        "all_phase_axes_estimable": phase["summary"]["estimable_axes"]
        == len(PHASE_CONTINUOUS_AXES),
        "all_global_axes_estimable": global_result["summary"]["estimable_axes"]
        == len(GLOBAL_CONTINUOUS_AXES),
        "all_phase_classes_estimable": state["estimable_classes"]
        == len(PHASE_STATE_CLASSES),
        "phase_r2": phase_r2 is not None
        and phase_r2 >= cfg.minimum_phase_variance_weighted_r2,
        "global_r2": global_r2 is not None
        and global_r2 >= cfg.minimum_global_variance_weighted_r2,
        "critical_axis_r2": critical_min is not None
        and critical_min >= cfg.minimum_critical_axis_r2,
        "phase_state": state["balanced_accuracy"] is not None
        and state["balanced_accuracy"] >= cfg.minimum_state_balanced_accuracy,
        "effective_rank": rank >= cfg.minimum_effective_rank,
        "beats_instruction_centroid": phase_r2 is not None
        and phase_r2 - float(instruction_centroid_r2)
        >= cfg.minimum_over_instruction_centroid_r2,
        "beats_source_only": phase_r2 is not None
        and phase_r2 - float(source_only_r2) >= cfg.minimum_over_source_only_r2,
        "beats_within_family_shuffle": phase_r2 is not None
        and phase_r2 - float(within_family_shuffle_r2)
        >= cfg.minimum_over_within_family_shuffle_r2,
        "appearance_decoupled": abs(float(appearance_correlation))
        <= cfg.maximum_abs_appearance_correlation,
    }
    local_checks_passed = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "local_diagnostic_only_external_qualification_required",
        "phase": phase,
        "global": global_result,
        "phase_state": state,
        "effective_rank": rank,
        "valid_coverage": {
            "phase_continuous_valid": int(phase_valid.sum().item()),
            "phase_continuous_total": int(phase_valid.numel()),
            "global_continuous_valid": int(global_valid.sum().item()),
            "global_continuous_total": int(global_valid.numel()),
            "phase_state_valid": int(phase_state_valid.sum().item()),
            "phase_state_total": int(phase_state_valid.numel()),
        },
        "minimum_critical_axis_r2": critical_min,
        "shortcut_baselines": {
            "instruction_centroid_r2": float(instruction_centroid_r2),
            "source_only_r2": float(source_only_r2),
            "within_family_shuffle_r2": float(within_family_shuffle_r2),
            "appearance_correlation": float(appearance_correlation),
        },
        "checks": checks,
        "local_checks_passed": local_checks_passed,
        # Compatibility key is intentionally fail-closed.  No tensor values,
        # IDs, strings, or locally constructed receipt can flip it to True.
        "qualified": False,
        "formally_qualified": False,
        "qualification_authority_available": False,
        "audit_receipt_sha256": audit_receipt.receipt_sha256,
        "split_manifest_sha256": audit_receipt.split_manifest_sha256,
        "decoder_receipt_sha256": audit_receipt.decoder_receipt_sha256,
        "point_teacher_receipt_role": "q_y_structured_point_teacher",
        "q_anchor_role": "compatible_contrastive_only",
        "declared_group_ids_disjoint": True,
        "content_disjoint": False,
        "content_disjoint_formally_verified": False,
        "training_authorized": False,
        "optimizer_authorized": False,
        "selection_authorized": False,
    }


def contract_v1() -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_count": PHASE_COUNT,
        "action_width": ACTION_WIDTH,
        "phase_continuous_axes": list(PHASE_CONTINUOUS_AXES),
        "global_continuous_axes": list(GLOBAL_CONTINUOUS_AXES),
        "phase_state_classes": list(PHASE_STATE_CLASSES),
        "rgb_reconstruction_used": False,
        "vae_latent_reconstruction_used": False,
        "generator_hidden_reconstruction_used": False,
        "source_relative_state_reconstruction_used": True,
        "q_y_unique_point_teacher": True,
        "q_anchor_contrastive_only": True,
        "point_teacher_provenance": "structured_content_bound_receipt_only",
        "free_form_teacher_role_accepted": False,
        "decoder_requires_frozen_checkpoint_receipt": True,
        "external_receipt_authenticity_verifier_implemented_here": False,
        "external_receipt_authenticity_hard_blocker_for_training": True,
        "point_loss_is_authority": False,
        "heldout_r2_required": True,
        "local_audit_can_formally_qualify": False,
        "formal_qualification_issuer": "external_typed_evaluator_not_implemented",
        "shortcut_and_noncollapse_gates_required": True,
        "typed_q_y_encoder_implemented_here": False,
        "typed_q_y_encoder_hard_blocker": True,
        "old_r7_random_lift_accepted_as_typed_q_y": False,
        "training_authorized": False,
        "optimizer_authorized": False,
        "selection_authorized": False,
    }


__all__ = [
    "ACTION_WIDTH",
    "GLOBAL_CONTINUOUS_AXES",
    "PHASE_CONTINUOUS_AXES",
    "PHASE_COUNT",
    "PHASE_STATE_CLASSES",
    "SCHEMA_VERSION",
    "AUDIT_RECEIPT_SCHEMA_VERSION",
    "DECODER_RECEIPT_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "ActionRepresentationGateConfigV1",
    "ActionStateDecoderConfigV1",
    "ActionStateDecoderV1",
    "ActionStateLossConfigV1",
    "ActionStatePredictionV1",
    "ActionStatePointTeacherReceiptV1",
    "ActionStateReconstructionError",
    "FrozenActionStateDecoderReceiptV1",
    "LocalActionStateAuditReceiptV1",
    "PredictedActionPlanReceiptV1",
    "StructuredActionStateV1",
    "action_state_reconstruction_loss_v1",
    "bind_frozen_action_state_decoder_v1",
    "build_action_state_point_teacher_receipt_v1",
    "build_action_representation_audit_v1",
    "build_local_action_state_audit_receipt_v1",
    "build_predicted_action_plan_receipt_v1",
    "contract_v1",
    "effective_rank_v1",
    "masked_axis_r2_v1",
    "phase_state_balanced_accuracy_v1",
    "validate_action_state_point_teacher_receipt_v1",
    "validate_frozen_action_state_decoder_v1",
    "validate_predicted_action_plan_receipt_v1",
    "validate_structured_action_state_v1",
]
