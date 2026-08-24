#!/usr/bin/env python3
"""Fail-closed branch runtime for full-30 action learning.

This module is deliberately below a trainer and above the official model
helpers.  It owns no Bernini model implementation.  Instead it:

* starts every branch from one exact ``[source; noisy-target]`` patch object,
  rotary object, and timestep object;
* calls the existing direct-``shared_step`` target-tail helper;
* calls the existing graph-preserving Graft Wan-unpack helper; and
* makes frozen-base routing an inseparable combination of temporary
  ``eval`` mode, ``inference_mode``, PEFT ``disable_adapter()``, and
  ``official_frozen_native_only()``.

The output is the official post-final-norm/post-``proj_out`` target velocity,
never a source row or a pre-head hidden state.  Trainable outputs retain their
autograd graph; frozen outputs are required to be graph-free and are detached.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
import struct
from typing import Any, Callable, Optional

import torch

import packed_preservation_lora_v2 as packed_core


SCHEMA_VERSION = "bernini-full30-action-runtime-v1"
RECEIPT_SCHEMA_VERSION = "bernini-full30-action-runtime-record-receipt-v1"
PHASE_RECEIPT_SCHEMA_VERSION = "bernini-full30-action-runtime-phase-receipt-v1"
PLAN_SCHEMA_VERSION = "bernini-full30-action-runtime-evaluation-plan-v1"

ARMS = ("action-only", "action+retain")
ACTION_BRANCHES = ("action", "incomplete")
GLOBAL_BATCH = 8
LATENT_CHANNELS = 16
LATENT_PHASES = 21
PATCH_SIZE = (1, 2, 2)
PATCH_VECTOR_WIDTH = 64
HIDDEN_WIDTH = 1536
ROPE_WIDTH = 64
MODEL_ID = "transformer_1"
POST_HEAD_STAGE = "official-post-final-norm-proj-out-target-velocity"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


class Full30ActionRuntimeError(RuntimeError):
    """Raised before an ambiguous branch result can enter the objective."""


SharedStepTargetHelper = Callable[..., Any]
WanUnpackHelper = Callable[..., Any]


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise Full30ActionRuntimeError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    if "receipt_digest" in unsigned:
        raise Full30ActionRuntimeError("unsigned receipt already contains a digest")
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def canonical_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    if not isinstance(receipt, Mapping):
        raise Full30ActionRuntimeError("receipt must be a mapping")
    value = dict(receipt)
    digest = value.pop("receipt_digest", None)
    if type(digest) is not str or digest != object_sha256(value):
        raise Full30ActionRuntimeError("receipt digest differs")
    return canonical_json_bytes(dict(receipt))


def _safe_text(value: Any, *, label: str) -> str:
    if type(value) is not str or _SAFE_TEXT.fullmatch(value) is None:
        raise Full30ActionRuntimeError(f"{label} must be a safe nonempty identifier")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise Full30ActionRuntimeError(f"{label} must be lowercase SHA-256")
    return value


def tensor_sha256(value: torch.Tensor, *, label: str) -> str:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise Full30ActionRuntimeError(f"{label} must be a strided tensor")
    tensor = value.detach().contiguous().cpu()
    metadata = canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": [int(item) for item in tensor.shape],
        }
    )
    digest = hashlib.sha256(b"full30-action-runtime-tensor-v1\x00")
    digest.update(struct.pack(">Q", len(metadata)))
    digest.update(metadata)
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _shares_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    return (
        first.device == second.device
        and int(first.untyped_storage().data_ptr())
        == int(second.untyped_storage().data_ptr())
    )


@dataclass(frozen=True)
class ConditionBindingV1:
    """Bind an opaque existing helper condition to reviewed authority."""

    role: str
    authority_sha256: str
    condition: Any = field(repr=False, compare=False)

    def validate(self, *, expected_role: str) -> None:
        if self.role != expected_role:
            raise Full30ActionRuntimeError(
                f"condition role differs: {self.role!r} != {expected_role!r}"
            )
        _sha256(self.authority_sha256, label=f"{expected_role} condition authority")
        if self.condition is None:
            raise Full30ActionRuntimeError(f"{expected_role} condition is absent")


@dataclass(frozen=True)
class Full30ActionRecordV1:
    row_id: str
    source_iid: str
    branch: str
    source_patches: torch.Tensor = field(repr=False, compare=False)
    noisy_target_patches: torch.Tensor = field(repr=False, compare=False)
    rotary_embs: torch.Tensor = field(repr=False, compare=False)
    timestep: torch.Tensor = field(repr=False, compare=False)
    spatial_shape: tuple[int, int, int, int, int]
    branch_condition: ConditionBindingV1 = field(repr=False, compare=False)
    noop_condition: ConditionBindingV1 = field(repr=False, compare=False)


@dataclass(frozen=True)
class PhysicalEvaluationV1:
    physical_index: int
    record_index: int
    record_evaluation_index: int
    branch_slot: str
    route: str
    condition_role: str
    graph_policy: str

    def receipt(self) -> dict[str, Any]:
        return {
            "physical_index": self.physical_index,
            "record_index": self.record_index,
            "record_evaluation_index": self.record_evaluation_index,
            "branch_slot": self.branch_slot,
            "route": self.route,
            "condition_role": self.condition_role,
            "graph_policy": self.graph_policy,
        }


@dataclass(frozen=True)
class Full30ActionRuntimeOutputsV1:
    trainable_branch_velocity: torch.Tensor = field(repr=False, compare=False)
    frozen_noop_velocity: torch.Tensor = field(repr=False, compare=False)
    frozen_branch_velocity: torch.Tensor = field(repr=False, compare=False)
    trainable_noop_velocity: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False
    )
    receipt: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Full30ActionPhaseOutputsV1:
    trainable_branch_velocity: torch.Tensor = field(repr=False, compare=False)
    frozen_noop_velocity: torch.Tensor = field(repr=False, compare=False)
    frozen_branch_velocity: torch.Tensor = field(repr=False, compare=False)
    receipt: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Full30NoopPhaseOutputsV1:
    trainable_noop_velocity: torch.Tensor = field(repr=False, compare=False)
    receipt: Mapping[str, Any] = field(default_factory=dict)


def _per_record_slots(arm: str) -> tuple[tuple[str, str, str, str], ...]:
    if arm not in ARMS:
        raise Full30ActionRuntimeError(f"unknown formal arm: {arm!r}")
    slots = (
        ("trainable_branch", "trainable", "branch", "graph-preserved"),
        ("frozen_noop", "frozen-official", "noop", "detached"),
        ("frozen_branch", "frozen-official", "branch", "detached"),
    )
    if arm == "action+retain":
        slots += (("trainable_noop", "trainable", "noop", "graph-preserved"),)
    return slots


def build_physical_evaluation_plan_v1(
    arm: str, *, global_batch: int = GLOBAL_BATCH
) -> tuple[PhysicalEvaluationV1, ...]:
    if type(global_batch) is not int or global_batch <= 0:
        raise Full30ActionRuntimeError("global_batch must be a positive integer")
    slots = _per_record_slots(arm)
    result: list[PhysicalEvaluationV1] = []
    for record_index in range(global_batch):
        for local_index, (slot, route, condition, graph) in enumerate(slots):
            result.append(
                PhysicalEvaluationV1(
                    physical_index=len(result),
                    record_index=record_index,
                    record_evaluation_index=local_index,
                    branch_slot=slot,
                    route=route,
                    condition_role=condition,
                    graph_policy=graph,
                )
            )
    expected = global_batch * (3 if arm == "action-only" else 4)
    if len(result) != expected:
        raise Full30ActionRuntimeError("physical evaluation plan count differs")
    return tuple(result)


def physical_evaluation_plan_receipt_v1(arm: str) -> dict[str, Any]:
    plan = build_physical_evaluation_plan_v1(arm)
    rows = [item.receipt() for item in plan]
    value = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "arm": arm,
        "global_batch": GLOBAL_BATCH,
        "evaluations_per_record": 3 if arm == "action-only" else 4,
        "physical_evaluation_count": len(rows),
        "rows": rows,
    }
    return {**value, "plan_digest": object_sha256(value)}


def _official_shared_step_target_helper() -> SharedStepTargetHelper:
    # Lazy import keeps model-loading modules out of contract-only consumers.
    from infer_dclr_reward_runtime_smoke import shared_step_target_prediction

    return shared_step_target_prediction


def _official_wan_unpack_helper() -> WanUnpackHelper:
    # This helper is audited against the pinned Bernini einops order and
    # explicitly promises graph preservation.
    from graft_phase_a_native_training_closure_v1 import unpack_wan_target_velocity

    return unpack_wan_target_velocity


def unpack_post_head_target_velocity_v1(
    packed_target: torch.Tensor,
    *,
    spatial_shape: Sequence[int],
    unpack_helper: Optional[WanUnpackHelper] = None,
) -> torch.Tensor:
    """Use the existing Graft helper row-wise and preserve autograd for ``B``."""

    shape = tuple(int(item) for item in spatial_shape)
    if (
        len(shape) != 5
        or shape[0] <= 0
        or shape[1:3] != (LATENT_CHANNELS, LATENT_PHASES)
        or shape[3] <= 0
        or shape[4] <= 0
        or shape[3] % PATCH_SIZE[1]
        or shape[4] % PATCH_SIZE[2]
    ):
        raise Full30ActionRuntimeError(
            "post-head spatial shape must be [B,16,21,H,W] with positive even H/W"
        )
    batch, channels, phases, height, width = shape
    target_tokens = phases * (height // 2) * (width // 2)
    if (
        not isinstance(packed_target, torch.Tensor)
        or packed_target.layout != torch.strided
        or not packed_target.is_floating_point()
        or tuple(int(item) for item in packed_target.shape)
        != (batch, target_tokens, PATCH_VECTOR_WIDTH)
        or not bool(torch.isfinite(packed_target).all().item())
    ):
        raise Full30ActionRuntimeError(
            "official post-head target rows must be finite [B,target_tokens,64]"
        )
    helper = unpack_helper or _official_wan_unpack_helper()
    rows: list[torch.Tensor] = []
    for index in range(batch):
        try:
            row = helper(
                packed_target[index : index + 1],
                spatial_shape=(1, channels, phases, height, width),
            )
        except Exception as error:
            raise Full30ActionRuntimeError(
                "existing Graft post-head target-velocity unpack failed"
            ) from error
        if (
            not isinstance(row, torch.Tensor)
            or tuple(int(item) for item in row.shape)
            != (1, channels, phases, height, width)
            or row.device != packed_target.device
            or not row.is_floating_point()
            or not row.is_contiguous()
            or not bool(torch.isfinite(row).all().item())
        ):
            raise Full30ActionRuntimeError("Graft unpack result geometry differs")
        rows.append(row)
    result = rows[0] if batch == 1 else torch.cat(rows, dim=0).contiguous()
    if tuple(int(item) for item in result.shape) != shape:
        raise Full30ActionRuntimeError("post-head spatial velocity shape differs")
    return result


class Full30ActionBranchRuntimeV1:
    """Execute one logical full-30 record through its formal branch routes."""

    def __init__(
        self,
        *,
        renderer: Any,
        transformer: Any,
        adapter_controller: Any,
        shared_step_helper: Optional[SharedStepTargetHelper] = None,
        unpack_helper: Optional[WanUnpackHelper] = None,
        test_only_injected_helpers: bool = False,
    ) -> None:
        patch_embedding = getattr(transformer, "patch_embedding", None)
        disable_adapter = getattr(adapter_controller, "disable_adapter", None)
        if not callable(patch_embedding):
            raise Full30ActionRuntimeError("transformer.patch_embedding is unavailable")
        if not callable(disable_adapter):
            raise Full30ActionRuntimeError("PEFT controller.disable_adapter is unavailable")
        if (
            not callable(getattr(adapter_controller, "eval", None))
            or not callable(getattr(adapter_controller, "train", None))
            or type(getattr(adapter_controller, "training", None)) is not bool
        ):
            raise Full30ActionRuntimeError(
                "PEFT controller does not expose an exact train/eval state"
            )
        if shared_step_helper is not None and not test_only_injected_helpers:
            raise Full30ActionRuntimeError(
                "a non-official shared_step helper is allowed only in CPU contract tests"
            )
        if unpack_helper is not None and not test_only_injected_helpers:
            raise Full30ActionRuntimeError(
                "a non-official unpack helper is allowed only in CPU contract tests"
            )
        self.renderer = renderer
        self.transformer = transformer
        self.adapter_controller = adapter_controller
        self._shared_step_helper = (
            shared_step_helper or _official_shared_step_target_helper()
        )
        self._unpack_helper = unpack_helper
        self._test_only = bool(test_only_injected_helpers)

    def _validate_record(
        self, record: Full30ActionRecordV1
    ) -> tuple[int, int, tuple[int, int, int, int, int]]:
        if not isinstance(record, Full30ActionRecordV1):
            raise Full30ActionRuntimeError("record type differs")
        _safe_text(record.row_id, label="row_id")
        _safe_text(record.source_iid, label="source_iid")
        if record.branch not in ACTION_BRANCHES:
            raise Full30ActionRuntimeError("record branch must be action or incomplete")
        record.branch_condition.validate(expected_role="branch")
        record.noop_condition.validate(expected_role="noop")
        if record.branch_condition.condition is record.noop_condition.condition:
            raise Full30ActionRuntimeError("branch and noop conditions must be distinct objects")

        source = record.source_patches
        target = record.noisy_target_patches
        if (
            not isinstance(source, torch.Tensor)
            or not isinstance(target, torch.Tensor)
            or source.layout != torch.strided
            or target.layout != torch.strided
            or source.dtype != target.dtype
            or source.device != target.device
            or source.shape != target.shape
            or source.ndim != 5
            or tuple(int(item) for item in source.shape[1:])
            != (LATENT_CHANNELS, *PATCH_SIZE)
            or not source.is_floating_point()
            or not source.is_contiguous()
            or not target.is_contiguous()
            or source.requires_grad
            or target.requires_grad
            or source.grad_fn is not None
            or target.grad_fn is not None
            or not bool(torch.isfinite(source).all().item())
            or not bool(torch.isfinite(target).all().item())
            or _shares_storage(source, target)
        ):
            raise Full30ActionRuntimeError(
                "source/noisy-target patches must be distinct matching detached finite native patches"
            )
        target_tokens = int(target.shape[0])
        if target_tokens <= 0:
            raise Full30ActionRuntimeError("target patch span is empty")
        shape = tuple(int(item) for item in record.spatial_shape)
        if (
            len(shape) != 5
            or shape[:3] != (1, LATENT_CHANNELS, LATENT_PHASES)
            or shape[3] <= 0
            or shape[4] <= 0
            or shape[3] % 2
            or shape[4] % 2
            or target_tokens
            != LATENT_PHASES * (shape[3] // 2) * (shape[4] // 2)
        ):
            raise Full30ActionRuntimeError("record spatial/target patch geometry differs")

        rotary = record.rotary_embs
        timestep = record.timestep
        if (
            not isinstance(rotary, torch.Tensor)
            or rotary.layout != torch.strided
            or rotary.dtype != torch.complex128
            or tuple(int(item) for item in rotary.shape)
            != (1, 1, 2 * target_tokens, ROPE_WIDTH)
            or rotary.device != source.device
            or rotary.requires_grad
            or rotary.grad_fn is not None
            or not bool(torch.isfinite(rotary).all().item())
        ):
            raise Full30ActionRuntimeError("rotary must be the detached native packed object")
        if (
            not isinstance(timestep, torch.Tensor)
            or timestep.layout != torch.strided
            or tuple(int(item) for item in timestep.shape) != (1,)
            or timestep.dtype not in (torch.float32, torch.int64)
            or timestep.device != source.device
            or timestep.requires_grad
            or timestep.grad_fn is not None
            or (
                timestep.dtype == torch.float32
                and not bool(torch.isfinite(timestep).all().item())
            )
        ):
            raise Full30ActionRuntimeError("timestep must be one detached native FP32/INT64 object")
        return target_tokens, 2 * target_tokens, shape

    def _embed(self, packed_patches: torch.Tensor, *, total_tokens: int) -> torch.Tensor:
        embedded = self.transformer.patch_embedding(packed_patches)
        if (
            not isinstance(embedded, torch.Tensor)
            or embedded.layout != torch.strided
            or tuple(int(item) for item in embedded.shape)
            != (total_tokens, HIDDEN_WIDTH, 1, 1, 1)
            or embedded.device != packed_patches.device
            or not embedded.is_floating_point()
            or not bool(torch.isfinite(embedded).all().item())
        ):
            raise Full30ActionRuntimeError("official patch embedding geometry differs")
        tokens = embedded.flatten(1).unsqueeze(0)
        if tuple(int(item) for item in tokens.shape) != (1, total_tokens, HIDDEN_WIDTH):
            raise Full30ActionRuntimeError("embedded visual token geometry differs")
        return tokens

    def _call_shared_step(
        self,
        *,
        embedded: torch.Tensor,
        target_mask: torch.Tensor,
        target_tokens: int,
        record: Full30ActionRecordV1,
        condition: ConditionBindingV1,
    ) -> torch.Tensor:
        try:
            value = self._shared_step_helper(
                self.renderer,
                model_id=MODEL_ID,
                noisy_latents=embedded,
                rotary_embs=record.rotary_embs,
                target_tokens=target_tokens,
                target_mask=target_mask,
                timestep=record.timestep,
                condition=condition.condition,
            )
        except Exception as error:
            raise Full30ActionRuntimeError(
                "existing official shared_step target-tail helper failed"
            ) from error
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or tuple(int(item) for item in value.shape)
            != (1, target_tokens, PATCH_VECTOR_WIDTH)
            or value.device != embedded.device
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise Full30ActionRuntimeError(
                "shared_step result is not official post-head target rows [1,N,64]"
            )
        return value

    def _branch_forward(
        self,
        *,
        slot: str,
        frozen: bool,
        condition: ConditionBindingV1,
        record: Full30ActionRecordV1,
        packed_patches: torch.Tensor,
        target_mask: torch.Tensor,
        target_tokens: int,
        total_tokens: int,
        spatial_shape: tuple[int, int, int, int, int],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if frozen:
            disable_context = self.adapter_controller.disable_adapter()
            if not (
                callable(getattr(disable_context, "__enter__", None))
                and callable(getattr(disable_context, "__exit__", None))
            ):
                raise Full30ActionRuntimeError(
                    "PEFT disable_adapter did not return a context manager"
                )
            prior_training = self.adapter_controller.training
            try:
                self.adapter_controller.eval()
                if self.adapter_controller.training is not False:
                    raise Full30ActionRuntimeError(
                        "frozen official route did not enter eval mode"
                    )
                with torch.inference_mode(), disable_context, packed_core.official_frozen_native_only():
                    embedded = self._embed(packed_patches, total_tokens=total_tokens)
                    packed_target = self._call_shared_step(
                        embedded=embedded,
                        target_mask=target_mask,
                        target_tokens=target_tokens,
                        record=record,
                        condition=condition,
                    )
                    spatial = unpack_post_head_target_velocity_v1(
                        packed_target,
                        spatial_shape=spatial_shape,
                        unpack_helper=self._unpack_helper,
                    )
            finally:
                self.adapter_controller.train(prior_training)
            if self.adapter_controller.training is not prior_training:
                raise Full30ActionRuntimeError(
                    "frozen official route did not restore the prior model mode"
                )
            if (
                packed_target.requires_grad
                or packed_target.grad_fn is not None
                or spatial.requires_grad
                or spatial.grad_fn is not None
            ):
                raise Full30ActionRuntimeError("frozen official route retained a graph")
            spatial = spatial.detach().contiguous()
            graph_policy = "detached"
            compute_mode = "eval-inference-mode"
            model_training_during_forward = False
        else:
            if self.adapter_controller.training is not True:
                raise Full30ActionRuntimeError(
                    "trainable route requires model training mode"
                )
            with packed_core.packed_role_layout(target_tokens, target_tokens):
                embedded = self._embed(packed_patches, total_tokens=total_tokens)
            packed_target = self._call_shared_step(
                embedded=embedded,
                target_mask=target_mask,
                target_tokens=target_tokens,
                record=record,
                condition=condition,
            )
            spatial = unpack_post_head_target_velocity_v1(
                packed_target,
                spatial_shape=spatial_shape,
                unpack_helper=self._unpack_helper,
            )
            if (
                not packed_target.requires_grad
                or packed_target.grad_fn is None
                or not spatial.requires_grad
                or spatial.grad_fn is None
            ):
                raise Full30ActionRuntimeError(
                    "trainable official post-head target velocity lost its graph"
                )
            graph_policy = "graph-preserved"
            compute_mode = "train-autograd"
            model_training_during_forward = True
        row = {
            "branch_slot": slot,
            "route": "frozen-official" if frozen else "trainable",
            "condition_role": condition.role,
            "condition_authority_sha256": condition.authority_sha256,
            "output_stage": POST_HEAD_STAGE,
            "packed_target_shape": [int(item) for item in packed_target.shape],
            "spatial_velocity_shape": [int(item) for item in spatial.shape],
            "spatial_velocity_dtype": str(spatial.dtype),
            "graph_policy": graph_policy,
            "compute_mode": compute_mode,
            "model_training_during_forward": model_training_during_forward,
            "output_sha256": tensor_sha256(spatial, label=f"{slot} output"),
        }
        return spatial, row

    def _execute_phase_slots(
        self,
        *,
        phase: str,
        record: Full30ActionRecordV1,
        slots: tuple[tuple[str, str, str, str], ...],
    ) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
        """Execute exactly one optimizer phase without hidden extra branches."""

        expected_slots = {
            "action": (
                ("trainable_branch", "trainable", "branch", "graph-preserved"),
                ("frozen_noop", "frozen-official", "noop", "detached"),
                ("frozen_branch", "frozen-official", "branch", "detached"),
            ),
            "noop": (
                ("trainable_noop", "trainable", "noop", "graph-preserved"),
            ),
        }
        if phase not in expected_slots or slots != expected_slots[phase]:
            raise Full30ActionRuntimeError("runtime phase slot closure differs")
        target_tokens, total_tokens, spatial_shape = self._validate_record(record)
        packed_patches = torch.cat(
            (record.source_patches, record.noisy_target_patches), dim=0
        ).contiguous()
        if (
            tuple(int(item) for item in packed_patches.shape)
            != (total_tokens, LATENT_CHANNELS, *PATCH_SIZE)
            or _shares_storage(packed_patches, record.source_patches)
            or _shares_storage(packed_patches, record.noisy_target_patches)
        ):
            raise Full30ActionRuntimeError("phase shared source/noisy packed object differs")
        target_mask = torch.zeros(
            total_tokens, dtype=torch.bool, device=packed_patches.device
        )
        target_mask[target_tokens:] = True
        if (
            bool(target_mask[:target_tokens].any().item())
            or not bool(target_mask[target_tokens:].all().item())
        ):
            raise Full30ActionRuntimeError("phase target-tail mask selected source rows")

        tracked = {
            "source_patches": record.source_patches,
            "noisy_target_patches": record.noisy_target_patches,
            "packed_patches": packed_patches,
            "rotary_embs": record.rotary_embs,
            "timestep": record.timestep,
        }
        tracked_ids = {name: id(value) for name, value in tracked.items()}
        before_digests = {
            name: tensor_sha256(value, label=name) for name, value in tracked.items()
        }
        input_binding = {
            "row_id": record.row_id,
            "source_iid": record.source_iid,
            "branch": record.branch,
            "spatial_shape": list(spatial_shape),
            "branch_condition_authority_sha256": (
                record.branch_condition.authority_sha256
            ),
            "noop_condition_authority_sha256": record.noop_condition.authority_sha256,
            "input_sha256": before_digests,
        }
        input_binding_digest = object_sha256(input_binding)
        outputs: dict[str, torch.Tensor] = {}
        trace: list[dict[str, Any]] = []
        for slot, route, condition_role, _graph_policy in slots:
            condition = (
                record.branch_condition
                if condition_role == "branch"
                else record.noop_condition
            )
            value, row = self._branch_forward(
                slot=slot,
                frozen=route == "frozen-official",
                condition=condition,
                record=record,
                packed_patches=packed_patches,
                target_mask=target_mask,
                target_tokens=target_tokens,
                total_tokens=total_tokens,
                spatial_shape=spatial_shape,
            )
            if {name: id(item) for name, item in tracked.items()} != tracked_ids:
                raise Full30ActionRuntimeError("phase shared input object identity changed")
            after = {
                name: tensor_sha256(item, label=name) for name, item in tracked.items()
            }
            if after != before_digests:
                raise Full30ActionRuntimeError("phase shared input bytes changed")
            outputs[slot] = value
            trace.append(row)
        if set(outputs) != {slot for slot, _, _, _ in slots}:
            raise Full30ActionRuntimeError("runtime phase output closure differs")
        per_record = 3 if phase == "action" else 1
        receipt = _seal(
            {
                "schema_version": PHASE_RECEIPT_SCHEMA_VERSION,
                "runtime_schema_version": SCHEMA_VERSION,
                "phase": phase,
                "row_id": record.row_id,
                "source_iid": record.source_iid,
                "branch": record.branch,
                "input_binding": input_binding,
                "input_binding_digest": input_binding_digest,
                "input_contract": {
                    "same_record_object_within_phase": True,
                    "same_source_patch_object_within_phase": True,
                    "same_noisy_target_patch_object_within_phase": True,
                    "same_packed_patch_object_within_phase": True,
                    "same_rotary_object_within_phase": True,
                    "same_timestep_object_within_phase": True,
                    "input_bytes_immutable": True,
                    "target_tail_only": True,
                    "source_rows_selected": False,
                },
                "phase_evaluation_plan": {
                    "global_batch": GLOBAL_BATCH,
                    "evaluations_per_record": per_record,
                    "global_physical_evaluation_count": GLOBAL_BATCH * per_record,
                    "slots": [slot for slot, _, _, _ in slots],
                },
                "record_branch_trace": trace,
                "output_contract": {
                    "stage": POST_HEAD_STAGE,
                    "shape": list(spatial_shape),
                    "contiguous": True,
                },
            }
        )
        return outputs, receipt

    def execute_action_phase(
        self, *, record: Full30ActionRecordV1
    ) -> Full30ActionPhaseOutputsV1:
        """Execute only the three action-objective routes for one record."""

        slots = (
            ("trainable_branch", "trainable", "branch", "graph-preserved"),
            ("frozen_noop", "frozen-official", "noop", "detached"),
            ("frozen_branch", "frozen-official", "branch", "detached"),
        )
        outputs, receipt = self._execute_phase_slots(
            phase="action", record=record, slots=slots
        )
        return Full30ActionPhaseOutputsV1(
            trainable_branch_velocity=outputs["trainable_branch"],
            frozen_noop_velocity=outputs["frozen_noop"],
            frozen_branch_velocity=outputs["frozen_branch"],
            receipt=receipt,
        )

    def execute_noop_phase(
        self, *, record: Full30ActionRecordV1
    ) -> Full30NoopPhaseOutputsV1:
        """Replay only the trainable-noop target-row route for one record."""

        slots = (("trainable_noop", "trainable", "noop", "graph-preserved"),)
        outputs, receipt = self._execute_phase_slots(
            phase="noop", record=record, slots=slots
        )
        return Full30NoopPhaseOutputsV1(
            trainable_noop_velocity=outputs["trainable_noop"], receipt=receipt
        )

    def execute_record(
        self, *, arm: str, record: Full30ActionRecordV1
    ) -> Full30ActionRuntimeOutputsV1:
        _per_record_slots(arm)
        action = self.execute_action_phase(record=record)
        noop = self.execute_noop_phase(record=record) if arm == "action+retain" else None
        action_binding = action.receipt.get("input_binding_digest")
        noop_binding = None if noop is None else noop.receipt.get("input_binding_digest")
        if noop is not None and action_binding != noop_binding:
            raise Full30ActionRuntimeError("combined phase input authority differs")
        plan_receipt = physical_evaluation_plan_receipt_v1(arm)
        phase_receipts = {"action": dict(action.receipt)}
        if noop is not None:
            phase_receipts["noop"] = dict(noop.receipt)
        phase_count = sum(
            int(value["phase_evaluation_plan"]["global_physical_evaluation_count"])
            for value in phase_receipts.values()
        )
        if phase_count != plan_receipt["physical_evaluation_count"]:
            raise Full30ActionRuntimeError("combined phase physical evaluation count differs")
        input_binding = action.receipt["input_binding"]
        trace = list(action.receipt["record_branch_trace"])
        if noop is not None:
            trace.extend(noop.receipt["record_branch_trace"])
        receipt = _seal(
            {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "runtime_schema_version": SCHEMA_VERSION,
                "row_id": record.row_id,
                "source_iid": record.source_iid,
                "branch": record.branch,
                "arm": arm,
                "input_contract": {
                    "same_source_patch_object_all_branches": True,
                    "same_noisy_target_patch_object_all_branches": True,
                    "same_packed_patch_object_all_branches": arm == "action-only",
                    "same_packed_patch_authority_all_phases": True,
                    "same_rotary_object_all_branches": True,
                    "same_timestep_object_all_branches": True,
                    "input_bytes_immutable": True,
                    "target_tail_only": True,
                    "source_rows_selected": False,
                    "input_sha256": input_binding["input_sha256"],
                },
                "helper_binding": {
                    "shared_step_target_helper": (
                        f"{self._shared_step_helper.__module__}."
                        f"{getattr(self._shared_step_helper, '__qualname__', type(self._shared_step_helper).__qualname__)}"
                    ),
                    "wan_unpack_helper": (
                        "graft_phase_a_native_training_closure_v1.unpack_wan_target_velocity"
                        if self._unpack_helper is None
                        else (
                            f"{self._unpack_helper.__module__}."
                            f"{getattr(self._unpack_helper, '__qualname__', type(self._unpack_helper).__qualname__)}"
                        )
                    ),
                    "test_only_injected_helpers": self._test_only,
                    "model_implementation_copied": False,
                },
                "frozen_route_contract": {
                    "model_eval": True,
                    "torch_inference_mode": True,
                    "peft_disable_adapter": True,
                    "official_frozen_native_only": True,
                    "prior_model_mode_restored": True,
                    "detached": True,
                },
                "trainable_route_contract": {
                    "adapter_enabled": True,
                    "typed_patch_role_enabled": True,
                    "graph_preserved": True,
                },
                "output_contract": {
                    "stage": POST_HEAD_STAGE,
                    "shape": list(record.spatial_shape),
                    "contiguous": True,
                },
                "record_branch_trace": trace,
                "phase_execution": {
                    "strict_phase_public_api": True,
                    "action_before_noop": True,
                    "input_binding_digest": action_binding,
                    "phase_receipts": phase_receipts,
                },
                "formal_update_evaluation_plan": plan_receipt,
            }
        )
        return Full30ActionRuntimeOutputsV1(
            trainable_branch_velocity=action.trainable_branch_velocity,
            frozen_noop_velocity=action.frozen_noop_velocity,
            frozen_branch_velocity=action.frozen_branch_velocity,
            trainable_noop_velocity=(
                None if noop is None else noop.trainable_noop_velocity
            ),
            receipt=receipt,
        )


Full30ActionRuntimeV1 = Full30ActionBranchRuntimeV1


__all__ = [
    "ACTION_BRANCHES",
    "ARMS",
    "ConditionBindingV1",
    "Full30ActionBranchRuntimeV1",
    "Full30ActionPhaseOutputsV1",
    "Full30ActionRecordV1",
    "Full30ActionRuntimeError",
    "Full30ActionRuntimeOutputsV1",
    "Full30ActionRuntimeV1",
    "Full30NoopPhaseOutputsV1",
    "GLOBAL_BATCH",
    "POST_HEAD_STAGE",
    "PhysicalEvaluationV1",
    "build_physical_evaluation_plan_v1",
    "canonical_json_bytes",
    "canonical_receipt_bytes",
    "object_sha256",
    "physical_evaluation_plan_receipt_v1",
    "tensor_sha256",
    "unpack_post_head_target_velocity_v1",
]
