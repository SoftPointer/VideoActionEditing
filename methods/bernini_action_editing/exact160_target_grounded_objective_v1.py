#!/usr/bin/env python3
"""Pure target-grounded objective primitives for the exact160 contract.

This module intentionally owns no model, optimizer, data loader, checkpoint, or
remote-launch code.  In particular, it does not implement any of the legacy V4
cross-actor latent objectives.  Its narrow responsibilities are:

* standard rectified-flow noising and velocity targets for a clean edited
  target;
* separately normalized, non-empty target-coordinate event/context losses;
* target-side-validity alignment losses;
* a closed canonical action-anchor prototype containing only entity,
  relation, phase, and terminal views;
* the standard source-as-target no-op branch; and
* recursive rejection of legacy optimizer-connected fields.

The absence of a scalar ``total`` from the partition/prototype result objects
is deliberate.  A formal trainer must apply the separately preregistered
constraint/update rule required by ``md/action_editing/20260817_box`` rather
than silently inventing a weighted scalar soup here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Dict, Mapping, Sequence, Set, Tuple


SCHEMA = "bernini-exact160-target-grounded-objective-v1"

LATENT_CHANNELS = 16
LATENT_PHASES = 21
ENTITY_SLOTS = 3
DIRECTED_RELATION_SLOTS = 6
TERMINAL_SLOTS = 9
ENTITY_WIDTH = 256
RELATION_WIDTH = 128
PHASE_WIDTH = 128
TERMINAL_WIDTH = 256
DIRECTED_RELATION_PAIRS = (
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
)

ACTION_PROTOTYPE_FIELDS = (
    "q_entity",
    "q_relation",
    "q_phase",
    "q_terminal",
)
ACTION_VIEW_FIELDS = ("q_local",) + ACTION_PROTOTYPE_FIELDS
LOCAL_WIDTH = 64

# Exact normalized field names that are forbidden anywhere in an
# optimizer-connected objective payload.  This list is intentionally not a
# substring blacklist: formal payloads legitimately contain names such as
# ``edited_target`` and ``action_anchors``.  Callers may extend this closure in
# a new schema version, but may not weaken V1 at runtime.
FORBIDDEN_LEGACY_FIELDS = frozenset(
    {
        "teacher_unit",
        "frozen_source_action_velocity",
        "frozen_velocity_target",
        "frozen_relative_gain_band",
        "frozen_relative_band",
        "frozen_trust_radius",
        "psiout",
        "psi_out",
        "endpoint_action_target",
        "single_final_frame_action_score",
        "direct_anchor_target",
        "direct_anchor_sft",
        "anchor_clean",
        "anchor_latent",
        "action_anchor_latent",
        "action_anchor_latent_target",
        "anchor_action_trajectory",
        "dense_action_trajectory_supervision",
        "source_carrier",
        "source_carrier_target",
        "source_plus_anchor_trajectory_target",
        "fullfield_action_noop",
        "fullfield_action_noop_pcgrad_preserve",
        "current_adapter_noop_teacher",
    }
)


class Exact160ObjectiveError(RuntimeError):
    """Raised whenever an exact160 mathematical contract is violated."""


def _fail(message: str) -> None:
    raise Exact160ObjectiveError(message)


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("%s must be a lowercase SHA-256" % label)
    return value


def _torch() -> Any:
    import torch

    return torch


def _require_finite_floating_tensor(value: Any, *, label: str) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or not bool(value.is_floating_point()):
        _fail("%s must be a floating torch.Tensor" % label)
    if value.numel() <= 0 or not bool(torch.isfinite(value).all().item()):
        _fail("%s must be non-empty and finite" % label)
    return value


def _require_latent_field(value: Any, *, label: str) -> Any:
    tensor = _require_finite_floating_tensor(value, label=label)
    if (
        tensor.ndim != 5
        or int(tensor.shape[0]) <= 0
        or int(tensor.shape[1]) != LATENT_CHANNELS
        or int(tensor.shape[2]) != LATENT_PHASES
        or int(tensor.shape[3]) <= 0
        or int(tensor.shape[4]) <= 0
    ):
        _fail(
            "%s must have shape [B,%d,%d,H,W]"
            % (label, LATENT_CHANNELS, LATENT_PHASES)
        )
    return tensor


def _require_same_tensor_geometry(left: Any, right: Any, *, label: str) -> None:
    if (
        tuple(left.shape) != tuple(right.shape)
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        _fail("%s tensors must have identical shape, dtype, and device" % label)


def _broadcast_sigma(sigma: Any, reference: Any) -> Any:
    """Return a detached ``[B,1,1,1,1]`` sigma tensor.

    A scalar or one scalar per batch item is accepted.  Sigma is a frozen
    schedule coordinate, never a trainable value.
    """

    torch = _torch()
    batch = int(reference.shape[0])
    if isinstance(sigma, bool):
        _fail("sigma must be a finite scalar or one value per batch item")
    if isinstance(sigma, Real):
        numeric = float(sigma)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            _fail("sigma must be finite in [0,1]")
        result = torch.full(
            (batch, 1, 1, 1, 1),
            numeric,
            dtype=reference.dtype,
            device=reference.device,
        )
        return result
    if not isinstance(sigma, torch.Tensor) or not bool(sigma.is_floating_point()):
        _fail("sigma must be a finite scalar or floating torch.Tensor")
    if sigma.requires_grad:
        _fail("sigma must be a detached schedule coordinate")
    if sigma.numel() not in (1, batch):
        _fail("sigma tensor must contain one value or exactly B values")
    if not bool(torch.isfinite(sigma).all().item()):
        _fail("sigma tensor must be finite")
    flattened = sigma.reshape(-1)
    if bool(((flattened < 0.0) | (flattened > 1.0)).any().item()):
        _fail("sigma tensor must lie in [0,1]")
    if flattened.numel() == 1:
        flattened = flattened.expand(batch)
    return flattened.to(dtype=reference.dtype, device=reference.device).reshape(
        batch, 1, 1, 1, 1
    )


@dataclass(frozen=True)
class RectifiedFlowState:
    """One standard target-grounded rectified-flow state."""

    noisy_target: Any
    target_velocity: Any
    sigma: Any


def rectified_flow_state(clean_target: Any, epsilon: Any, sigma: Any) -> RectifiedFlowState:
    """Construct ``x_sigma=(1-sigma)z+sigma*epsilon`` and ``v*=epsilon-z``."""

    target = _require_latent_field(clean_target, label="clean edited target")
    noise = _require_latent_field(epsilon, label="rectified-flow epsilon")
    _require_same_tensor_geometry(target, noise, label="target/epsilon")
    if target.requires_grad or noise.requires_grad:
        _fail("clean target and epsilon must be detached flow evidence")
    coordinate = _broadcast_sigma(sigma, target)
    noisy = ((1.0 - coordinate) * target + coordinate * noise).contiguous()
    velocity = (noise - target).contiguous()
    torch = _torch()
    if not bool(torch.isfinite(noisy).all().item()) or not bool(
        torch.isfinite(velocity).all().item()
    ):
        _fail("rectified-flow state is non-finite")
    return RectifiedFlowState(
        noisy_target=noisy,
        target_velocity=velocity,
        sigma=coordinate,
    )


def predicted_clean(noisy_target: Any, predicted_velocity: Any, sigma: Any) -> Any:
    """Recover the model's clean prediction ``x0_hat=x_sigma-sigma*v_hat``."""

    noisy = _require_latent_field(noisy_target, label="noisy target")
    velocity = _require_latent_field(
        predicted_velocity, label="predicted target velocity"
    )
    _require_same_tensor_geometry(noisy, velocity, label="noisy/predicted velocity")
    coordinate = _broadcast_sigma(sigma, noisy)
    result = (noisy - coordinate * velocity).contiguous()
    if not bool(_torch().isfinite(result).all().item()):
        _fail("predicted clean target is non-finite")
    return result


def noop_source_as_target_state(source_clean: Any, epsilon: Any, sigma: Any) -> RectifiedFlowState:
    """Build the independent no-op branch with the source itself as target."""

    source = _require_latent_field(source_clean, label="no-op clean source target")
    return rectified_flow_state(source, epsilon, sigma)


def _canonical_cell_mask(mask: Any, reference: Any, *, label: str) -> Any:
    """Validate a target-coordinate mask and return ``[B,1,T,H,W]``."""

    torch = _torch()
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        _fail("%s must be a boolean torch.Tensor" % label)
    if mask.requires_grad:
        _fail("%s must be detached target-side evidence" % label)
    if mask.device != reference.device:
        _fail("%s must be on the prediction device" % label)
    expected_cell_shape = (
        int(reference.shape[0]),
        1,
        int(reference.shape[2]),
        int(reference.shape[3]),
        int(reference.shape[4]),
    )
    if tuple(mask.shape) == expected_cell_shape:
        return mask
    if tuple(mask.shape) == tuple(reference.shape):
        first = mask[:, :1]
        if not bool((mask == first.expand_as(mask)).all().item()):
            _fail("%s cannot select different latent channels" % label)
        return first
    _fail(
        "%s must have shape [B,1,T,H,W] or match the latent field" % label
    )


def _boolean_tensor_digest(mask: Any, *, label: str) -> str:
    """Content-address a detached boolean mask in a canonical C layout."""

    torch = _torch()
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        _fail("%s must be a boolean tensor" % label)
    if mask.requires_grad:
        _fail("%s must be detached" % label)
    canonical = mask.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"dtype": "bool", "shape": [int(item) for item in canonical.shape]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _snapshot_boolean_tensor(
    value: Any,
    *,
    label: str,
    expected_shape: Tuple[int, ...],
    expected_device: Any = None,
) -> Any:
    """Validate and clone one detached boolean evidence tensor."""

    torch = _torch()
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        _fail("%s must be a boolean torch.Tensor" % label)
    if value.requires_grad:
        _fail("%s must be detached evidence" % label)
    if tuple(value.shape) != expected_shape:
        _fail("%s has the wrong shape" % label)
    if expected_device is not None and value.device != expected_device:
        _fail("%s is on the wrong device" % label)
    return value.detach().clone().contiguous()


def _native_patch_event_mask(flow_event_mask: Any) -> Any:
    """Map ``[B,1,T,H,W]`` event cells to the native 2x2 patch grid."""

    batch, _, phases, height, width = (
        int(item) for item in flow_event_mask.shape
    )
    return (
        flow_event_mask[:, 0]
        .reshape(batch, phases, height // 2, 2, width // 2, 2)
        .any(dim=5)
        .any(dim=3)
        .contiguous()
    )


def _target_event_binding_metadata(
    *,
    target_latent_shape: Tuple[int, ...],
    min_event_cells: int,
    annotation_receipt_sha256: str,
    mapping_recipe_sha256: str,
    flow_event_mask_sha256: str,
    action_event_mask_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "bernini-exact160-target-event-mask-binding-v1",
        "target_latent_shape": [int(item) for item in target_latent_shape],
        "native_patch": [1, 2, 2],
        "mapping": "2x2_boolean_swept_union",
        "min_event_cells": min_event_cells,
        "annotation_receipt_sha256": annotation_receipt_sha256,
        "mapping_recipe_sha256": mapping_recipe_sha256,
        "flow_event_mask_sha256": flow_event_mask_sha256,
        "action_event_mask_sha256": action_event_mask_sha256,
    }


@dataclass(frozen=True)
class TargetEventMaskBinding:
    """One target-derived event mask bound across flow and action grids."""

    flow_event_mask: Any
    flow_context_mask: Any
    action_event_mask: Any
    target_latent_shape: Tuple[int, ...]
    min_event_cells: int
    annotation_receipt_sha256: str
    mapping_recipe_sha256: str
    flow_event_mask_sha256: str
    action_event_mask_sha256: str
    binding_digest: str


def bind_target_event_masks(
    clean_target: Any,
    target_event_mask: Any,
    *,
    annotation_receipt_sha256: str,
    mapping_recipe_sha256: str,
    min_event_cells: int = 1,
) -> TargetEventMaskBinding:
    """Freeze one clean-target mask for both flow and patch action losses.

    V1 pins the native Bernini patch mapping to a boolean swept-union over
    each 2x2 latent cell.  Dilation, visibility/confidence and phase-window
    policy are upstream evidence and are bound by ``mapping_recipe_sha256``;
    callers cannot provide a second, independently chosen action mask.
    """

    torch = _torch()
    target = _require_latent_field(clean_target, label="clean edited target")
    if target.requires_grad:
        _fail("clean target geometry evidence must be detached")
    if type(min_event_cells) is not int or min_event_cells <= 0:
        _fail("min_event_cells must be a positive built-in integer")
    annotation_sha = _require_sha256(
        annotation_receipt_sha256, label="annotation receipt SHA-256"
    )
    recipe_sha = _require_sha256(
        mapping_recipe_sha256, label="event-mask mapping recipe SHA-256"
    )
    event = _canonical_cell_mask(
        target_event_mask, target, label="clean-target event mask"
    ).detach().clone().contiguous()
    batch, _, phases, height, width = (int(item) for item in event.shape)
    if height % 2 or width % 2:
        _fail("target latent H/W must be divisible by the native 2x2 patch")
    context = (~event).contiguous()
    event_counts = event.reshape(batch, -1).sum(dim=1)
    context_counts = context.reshape(batch, -1).sum(dim=1)
    if bool((event_counts < min_event_cells).any().item()):
        _fail("every row must meet the target event-cell minimum")
    if bool((context_counts <= 0).any().item()):
        _fail("every row must retain target context cells")
    # [B,1,T,H,W] -> [B,T,H/2,W/2], preserving any participant event
    # evidence that intersects a native 2x2 patch.
    patch_event = _native_patch_event_mask(event)
    if bool((patch_event.reshape(batch, -1).sum(dim=1) <= 0).any().item()):
        _fail("target event mapping produced an empty action patch grid")
    flow_digest = _boolean_tensor_digest(event, label="flow event mask")
    action_digest = _boolean_tensor_digest(
        patch_event, label="action event mask"
    )
    target_latent_shape = tuple(int(item) for item in target.shape)
    metadata = _target_event_binding_metadata(
        target_latent_shape=target_latent_shape,
        min_event_cells=min_event_cells,
        annotation_receipt_sha256=annotation_sha,
        mapping_recipe_sha256=recipe_sha,
        flow_event_mask_sha256=flow_digest,
        action_event_mask_sha256=action_digest,
    )
    binding_digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    binding = TargetEventMaskBinding(
        flow_event_mask=event,
        flow_context_mask=context,
        action_event_mask=patch_event,
        target_latent_shape=target_latent_shape,
        min_event_cells=min_event_cells,
        annotation_receipt_sha256=annotation_sha,
        mapping_recipe_sha256=recipe_sha,
        flow_event_mask_sha256=flow_digest,
        action_event_mask_sha256=action_digest,
        binding_digest=binding_digest,
    )
    _revalidate_target_event_mask_binding(binding)
    return binding


@dataclass(frozen=True)
class _ValidatedTargetEventMaskBinding:
    flow_event_mask: Any
    flow_context_mask: Any
    action_event_mask: Any
    target_latent_shape: Tuple[int, ...]
    min_event_cells: int


def _revalidate_target_event_mask_binding(
    binding: TargetEventMaskBinding,
) -> _ValidatedTargetEventMaskBinding:
    """Recompute every event-binding invariant from immutable snapshots."""

    if type(binding) is not TargetEventMaskBinding:
        _fail("event objective requires a TargetEventMaskBinding")
    shape = binding.target_latent_shape
    if (
        type(shape) is not tuple
        or len(shape) != 5
        or any(type(item) is not int for item in shape)
    ):
        _fail("target event binding latent shape must be an exact integer tuple")
    batch, channels, phases, height, width = shape
    if (
        batch <= 0
        or channels != LATENT_CHANNELS
        or phases != LATENT_PHASES
        or height <= 0
        or width <= 0
        or height % 2
        or width % 2
    ):
        _fail("target event binding has invalid Bernini latent geometry")
    if type(binding.min_event_cells) is not int or binding.min_event_cells <= 0:
        _fail("target event binding min_event_cells must be a positive integer")
    annotation_sha = _require_sha256(
        binding.annotation_receipt_sha256,
        label="bound annotation receipt SHA-256",
    )
    recipe_sha = _require_sha256(
        binding.mapping_recipe_sha256,
        label="bound event-mask mapping recipe SHA-256",
    )
    recorded_flow_sha = _require_sha256(
        binding.flow_event_mask_sha256,
        label="recorded flow event-mask SHA-256",
    )
    recorded_action_sha = _require_sha256(
        binding.action_event_mask_sha256,
        label="recorded action event-mask SHA-256",
    )
    recorded_binding_sha = _require_sha256(
        binding.binding_digest,
        label="recorded target event binding digest",
    )
    flow_shape = (batch, 1, phases, height, width)
    action_shape = (batch, phases, height // 2, width // 2)
    flow_event = _snapshot_boolean_tensor(
        binding.flow_event_mask,
        label="bound flow event mask",
        expected_shape=flow_shape,
    )
    flow_context = _snapshot_boolean_tensor(
        binding.flow_context_mask,
        label="bound flow context mask",
        expected_shape=flow_shape,
        expected_device=flow_event.device,
    )
    action_event = _snapshot_boolean_tensor(
        binding.action_event_mask,
        label="bound action event mask",
        expected_shape=action_shape,
        expected_device=flow_event.device,
    )
    if not bool((flow_context == ~flow_event).all().item()):
        _fail("bound flow context mask is not the event-mask complement")
    event_counts = flow_event.reshape(batch, -1).sum(dim=1)
    context_counts = flow_context.reshape(batch, -1).sum(dim=1)
    if bool((event_counts < binding.min_event_cells).any().item()):
        _fail("bound flow event mask no longer meets min_event_cells")
    if bool((context_counts <= 0).any().item()):
        _fail("bound flow context mask is empty")
    expected_action_event = _native_patch_event_mask(flow_event)
    if not bool((action_event == expected_action_event).all().item()):
        _fail("bound action event mask differs from the native 2x2 flow union")
    flow_sha = _boolean_tensor_digest(flow_event, label="bound flow event mask")
    action_sha = _boolean_tensor_digest(
        action_event, label="bound action event mask"
    )
    if flow_sha != recorded_flow_sha or action_sha != recorded_action_sha:
        _fail("target event binding mask digest differs")
    metadata = _target_event_binding_metadata(
        target_latent_shape=shape,
        min_event_cells=binding.min_event_cells,
        annotation_receipt_sha256=annotation_sha,
        mapping_recipe_sha256=recipe_sha,
        flow_event_mask_sha256=flow_sha,
        action_event_mask_sha256=action_sha,
    )
    binding_sha = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if binding_sha != recorded_binding_sha:
        _fail("target event binding metadata digest differs")
    return _ValidatedTargetEventMaskBinding(
        flow_event_mask=flow_event,
        flow_context_mask=flow_context,
        action_event_mask=action_event,
        target_latent_shape=shape,
        min_event_cells=binding.min_event_cells,
    )


@dataclass(frozen=True)
class PartitionedFlowLoss:
    """Separately normalized event and context losses (intentionally no total)."""

    event: Any
    context: Any
    event_cells_per_sample: Tuple[int, ...]
    context_cells_per_sample: Tuple[int, ...]
    event_elements: int
    context_elements: int


def partitioned_flow_losses(
    predicted_velocity: Any,
    target_velocity: Any,
    event_mask: Any,
    context_mask: Any,
    *,
    min_event_cells: int = 1,
) -> PartitionedFlowLoss:
    """Compute disjoint target-event and target-context MSE means.

    The two partitions are returned separately.  Each batch item must have a
    non-empty event and context region; a tiny event cannot be diluted by a
    large context because each denominator is its own active-element count.
    """

    torch = _torch()
    prediction = _require_latent_field(
        predicted_velocity, label="predicted velocity"
    )
    target = _require_latent_field(target_velocity, label="target velocity")
    _require_same_tensor_geometry(prediction, target, label="prediction/target")
    if target.requires_grad:
        _fail("target velocity must be stop-gradient evidence")
    if type(min_event_cells) is not int or min_event_cells <= 0:
        _fail("min_event_cells must be a positive integer")
    event_cells = _canonical_cell_mask(
        event_mask, prediction, label="target event mask"
    )
    context_cells = _canonical_cell_mask(
        context_mask, prediction, label="target context mask"
    )
    if bool((event_cells & context_cells).any().item()):
        _fail("event/context masks must be disjoint")
    if not bool((event_cells | context_cells).all().item()):
        _fail("event/context masks must exhaust the target latent grid")
    event_counts_tensor = event_cells.reshape(int(prediction.shape[0]), -1).sum(dim=1)
    context_counts_tensor = context_cells.reshape(int(prediction.shape[0]), -1).sum(dim=1)
    event_counts = tuple(int(item) for item in event_counts_tensor.tolist())
    context_counts = tuple(int(item) for item in context_counts_tensor.tolist())
    if any(item < min_event_cells for item in event_counts):
        _fail("every row must meet the non-empty minimum target event-cell gate")
    if any(item <= 0 for item in context_counts):
        _fail("every row must have a non-empty target context partition")
    event_expanded = event_cells.expand_as(prediction)
    context_expanded = context_cells.expand_as(prediction)
    squared = (prediction.float() - target.float()).square()
    event_elements = int(event_expanded.sum().item())
    context_elements = int(context_expanded.sum().item())
    event_loss = squared[event_expanded].sum() / float(event_elements)
    context_loss = squared[context_expanded].sum() / float(context_elements)
    if not bool(torch.isfinite(torch.stack((event_loss, context_loss))).all().item()):
        _fail("partitioned flow losses are non-finite")
    return PartitionedFlowLoss(
        event=event_loss,
        context=context_loss,
        event_cells_per_sample=event_counts,
        context_cells_per_sample=context_counts,
        event_elements=event_elements,
        context_elements=context_elements,
    )


def partitioned_flow_losses_from_binding(
    predicted_velocity: Any,
    target_velocity: Any,
    binding: TargetEventMaskBinding,
) -> PartitionedFlowLoss:
    """Compute flow terms from the same bound target mask used by action loss."""

    validated = _revalidate_target_event_mask_binding(binding)
    if tuple(predicted_velocity.shape) != validated.target_latent_shape:
        _fail("predicted velocity geometry differs from target mask binding")
    if tuple(target_velocity.shape) != validated.target_latent_shape:
        _fail("target velocity geometry differs from target mask binding")
    return partitioned_flow_losses(
        predicted_velocity,
        target_velocity,
        validated.flow_event_mask,
        validated.flow_context_mask,
        min_event_cells=validated.min_event_cells,
    )


@dataclass(frozen=True)
class MaskedAlignmentTerm:
    """One alignment field with explicit structural-activity evidence."""

    value: Any
    active_elements: int
    active_rows: Tuple[bool, ...]


def _target_side_masked_term(
    student: Any,
    target: Any,
    target_validity: Any,
    *,
    label: str = "target-side alignment",
    allow_structurally_empty_rows: bool = False,
) -> MaskedAlignmentTerm:
    """MSE normalized only by detached, non-empty target-side validity.

    ``target_validity`` must be a prefix of the feature geometry; for example,
    ``[B,3,T]`` masks ``q_entity [B,3,T,D]``.  Student-predicted validity is
    deliberately not an argument to this function.
    """

    torch = _torch()
    prediction = _require_finite_floating_tensor(student, label=label + " student")
    truth = _require_finite_floating_tensor(target, label=label + " target")
    _require_same_tensor_geometry(prediction, truth, label=label)
    if truth.requires_grad:
        _fail("%s target must be stop-gradient evidence" % label)
    if not isinstance(target_validity, torch.Tensor) or target_validity.dtype != torch.bool:
        _fail("%s validity must be boolean" % label)
    if target_validity.requires_grad or target_validity.device != prediction.device:
        _fail("%s validity must be detached target-side evidence" % label)
    if target_validity.ndim <= 0 or target_validity.ndim > prediction.ndim:
        _fail("%s validity rank differs" % label)
    if tuple(target_validity.shape) != tuple(prediction.shape[: target_validity.ndim]):
        _fail("%s validity must match a leading feature prefix" % label)
    batch = int(prediction.shape[0])
    active_per_sample = target_validity.reshape(batch, -1).sum(dim=1)
    if not allow_structurally_empty_rows and bool((active_per_sample <= 0).any().item()):
        _fail("%s validity must be non-empty for every row" % label)
    expanded = target_validity
    for _ in range(prediction.ndim - target_validity.ndim):
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(prediction)
    elements = int(expanded.sum().item())
    if elements:
        loss = (prediction.float() - truth.float()).square()[expanded].sum() / float(elements)
    else:
        # A single-actor row has no directed i->j relation.  Preserve an
        # explicit graph-connected zero while reporting zero active elements;
        # consumers must not count this as a supervised relation term.
        loss = prediction.float().sum() * 0.0
    if not bool(torch.isfinite(loss).item()):
        _fail("%s loss is non-finite" % label)
    return MaskedAlignmentTerm(
        value=loss,
        active_elements=elements,
        active_rows=tuple(bool(item) for item in (active_per_sample > 0).tolist()),
    )


def target_side_masked_loss(
    student: Any,
    target: Any,
    target_validity: Any,
    *,
    label: str = "target-side alignment",
) -> Any:
    """Strict MSE normalized by non-empty detached target-side validity."""

    return _target_side_masked_term(
        student,
        target,
        target_validity,
        label=label,
        allow_structurally_empty_rows=False,
    ).value


def _canonical_participant_ids(
    value: Any, *, expected_batch: int
) -> Tuple[Tuple[str, ...], ...]:
    if type(expected_batch) is not int or expected_batch <= 0:
        _fail("causal participant binding batch must be positive")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail("causal participant IDs must contain one row sequence per batch item")
    if len(value) != expected_batch:
        _fail("causal participant ID rows must match the target batch")
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(
            row, (str, bytes, bytearray)
        ):
            _fail("causal participant IDs row %d must be a sequence" % row_index)
        if len(row) < 1 or len(row) > ENTITY_SLOTS:
            _fail("every row must have one to three causal participant IDs")
        canonical_row = []
        for participant_id in row:
            if (
                type(participant_id) is not str
                or not participant_id
                or participant_id != participant_id.strip()
            ):
                _fail("causal participant IDs must be non-empty canonical text")
            canonical_row.append(participant_id)
        if len(set(canonical_row)) != len(canonical_row):
            _fail("causal participant IDs must be unique within each row")
        rows.append(tuple(canonical_row))
    return tuple(rows)


def _relation_validity_from_entity(entity_validity: Any) -> Any:
    return _torch().stack(
        [
            entity_validity[:, left, :] & entity_validity[:, right, :]
            for left, right in DIRECTED_RELATION_PAIRS
        ],
        dim=1,
    ).contiguous()


def _target_participant_binding_metadata(
    *,
    causal_participant_ids_by_row: Tuple[Tuple[str, ...], ...],
    annotation_receipt_sha256: str,
    entity_validity_sha256: str,
    slot_presence_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "bernini-exact160-target-participant-binding-v1",
        "causal_participant_ids_by_row": [
            list(row) for row in causal_participant_ids_by_row
        ],
        "directed_relation_pairs": [
            [left, right] for left, right in DIRECTED_RELATION_PAIRS
        ],
        "annotation_receipt_sha256": annotation_receipt_sha256,
        "entity_validity_sha256": entity_validity_sha256,
        "slot_presence_sha256": slot_presence_sha256,
    }


@dataclass(frozen=True)
class TargetParticipantBinding:
    """Admission-bound causal participant slots and target entity validity."""

    causal_participant_ids_by_row: Tuple[Tuple[str, ...], ...]
    slot_presence: Any
    entity_validity: Any
    annotation_receipt_sha256: str
    slot_presence_sha256: str
    entity_validity_sha256: str
    binding_digest: str


@dataclass(frozen=True)
class _ValidatedTargetParticipantBinding:
    causal_participant_ids_by_row: Tuple[Tuple[str, ...], ...]
    slot_presence: Any
    entity_validity: Any
    relation_validity: Any


def _validate_participant_semantics(
    *,
    participant_ids: Tuple[Tuple[str, ...], ...],
    slot_presence: Any,
    entity_validity: Any,
) -> Any:
    """Validate slot exactness and derive the only legal relation validity."""

    torch = _torch()
    batch = len(participant_ids)
    expected_presence = torch.zeros(
        (batch, ENTITY_SLOTS), dtype=torch.bool, device=entity_validity.device
    )
    for row_index, row in enumerate(participant_ids):
        expected_presence[row_index, : len(row)] = True
    if not bool((slot_presence == expected_presence).all().item()):
        _fail("target participant slot presence differs from admission IDs")
    present_counts = entity_validity.sum(dim=2)
    if bool((present_counts[slot_presence] <= 0).any().item()):
        _fail("every admitted causal participant must have target entity validity")
    absent = (~slot_presence).unsqueeze(-1).expand_as(entity_validity)
    if bool(entity_validity[absent].any().item()):
        _fail("padding entity slots cannot carry target validity")
    relation_validity = _relation_validity_from_entity(entity_validity)
    relation_active = relation_validity.reshape(batch, -1).any(dim=1)
    for row_index, row in enumerate(participant_ids):
        if len(row) == 1 and bool(relation_active[row_index].item()):
            _fail("a single-actor row cannot have an active directed relation")
        if len(row) >= 2 and not bool(relation_active[row_index].item()):
            _fail("a multi-actor row must retain target relation evidence")
    return relation_validity


def bind_target_participants(
    target_entity_validity: Any,
    *,
    causal_participant_ids_by_row: Sequence[Sequence[str]],
    annotation_receipt_sha256: str,
) -> TargetParticipantBinding:
    """Bind admission participant IDs to exact target entity/relation validity."""

    torch = _torch()
    if (
        not isinstance(target_entity_validity, torch.Tensor)
        or target_entity_validity.dtype != torch.bool
        or target_entity_validity.ndim != 3
        or int(target_entity_validity.shape[0]) <= 0
        or tuple(target_entity_validity.shape[1:])
        != (ENTITY_SLOTS, LATENT_PHASES)
    ):
        _fail("target entity validity must have shape [B,3,21] and boolean dtype")
    if target_entity_validity.requires_grad:
        _fail("target entity validity must be detached admission evidence")
    batch = int(target_entity_validity.shape[0])
    participant_ids = _canonical_participant_ids(
        causal_participant_ids_by_row, expected_batch=batch
    )
    annotation_sha = _require_sha256(
        annotation_receipt_sha256,
        label="participant annotation receipt SHA-256",
    )
    entity_validity = target_entity_validity.detach().clone().contiguous()
    slot_presence = torch.zeros(
        (batch, ENTITY_SLOTS), dtype=torch.bool, device=entity_validity.device
    )
    for row_index, row in enumerate(participant_ids):
        slot_presence[row_index, : len(row)] = True
    _validate_participant_semantics(
        participant_ids=participant_ids,
        slot_presence=slot_presence,
        entity_validity=entity_validity,
    )
    slot_sha = _boolean_tensor_digest(
        slot_presence, label="target participant slot presence"
    )
    entity_sha = _boolean_tensor_digest(
        entity_validity, label="target entity validity"
    )
    metadata = _target_participant_binding_metadata(
        causal_participant_ids_by_row=participant_ids,
        annotation_receipt_sha256=annotation_sha,
        entity_validity_sha256=entity_sha,
        slot_presence_sha256=slot_sha,
    )
    binding_sha = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    binding = TargetParticipantBinding(
        causal_participant_ids_by_row=participant_ids,
        slot_presence=slot_presence,
        entity_validity=entity_validity,
        annotation_receipt_sha256=annotation_sha,
        slot_presence_sha256=slot_sha,
        entity_validity_sha256=entity_sha,
        binding_digest=binding_sha,
    )
    _revalidate_target_participant_binding(binding)
    return binding


def _revalidate_target_participant_binding(
    binding: TargetParticipantBinding,
) -> _ValidatedTargetParticipantBinding:
    if type(binding) is not TargetParticipantBinding:
        _fail("action alignment requires a TargetParticipantBinding")
    ids_value = binding.causal_participant_ids_by_row
    if type(ids_value) is not tuple:
        _fail("bound causal participant IDs must be an exact tuple")
    participant_ids = _canonical_participant_ids(
        ids_value, expected_batch=len(ids_value)
    )
    if any(type(row) is not tuple for row in ids_value):
        _fail("bound causal participant ID rows must be exact tuples")
    batch = len(participant_ids)
    slot_presence = _snapshot_boolean_tensor(
        binding.slot_presence,
        label="bound target participant slot presence",
        expected_shape=(batch, ENTITY_SLOTS),
    )
    entity_validity = _snapshot_boolean_tensor(
        binding.entity_validity,
        label="bound target entity validity",
        expected_shape=(batch, ENTITY_SLOTS, LATENT_PHASES),
        expected_device=slot_presence.device,
    )
    relation_validity = _validate_participant_semantics(
        participant_ids=participant_ids,
        slot_presence=slot_presence,
        entity_validity=entity_validity,
    )
    annotation_sha = _require_sha256(
        binding.annotation_receipt_sha256,
        label="bound participant annotation receipt SHA-256",
    )
    recorded_slot_sha = _require_sha256(
        binding.slot_presence_sha256,
        label="recorded target participant slot-presence SHA-256",
    )
    recorded_entity_sha = _require_sha256(
        binding.entity_validity_sha256,
        label="recorded target entity-validity SHA-256",
    )
    recorded_binding_sha = _require_sha256(
        binding.binding_digest,
        label="recorded target participant binding digest",
    )
    slot_sha = _boolean_tensor_digest(
        slot_presence, label="bound target participant slot presence"
    )
    entity_sha = _boolean_tensor_digest(
        entity_validity, label="bound target entity validity"
    )
    if slot_sha != recorded_slot_sha or entity_sha != recorded_entity_sha:
        _fail("target participant binding mask digest differs")
    metadata = _target_participant_binding_metadata(
        causal_participant_ids_by_row=participant_ids,
        annotation_receipt_sha256=annotation_sha,
        entity_validity_sha256=entity_sha,
        slot_presence_sha256=slot_sha,
    )
    binding_sha = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if binding_sha != recorded_binding_sha:
        _fail("target participant binding metadata digest differs")
    return _ValidatedTargetParticipantBinding(
        causal_participant_ids_by_row=participant_ids,
        slot_presence=slot_presence,
        entity_validity=entity_validity,
        relation_validity=relation_validity,
    )


@dataclass(frozen=True)
class CanonicalActionPrototype:
    """Closed canonical anchor/output action view; no raw local or camera view."""

    q_entity: Any
    q_relation: Any
    q_phase: Any
    q_terminal: Any

    def as_dict(self) -> Dict[str, Any]:
        return {
            "q_entity": self.q_entity,
            "q_relation": self.q_relation,
            "q_phase": self.q_phase,
            "q_terminal": self.q_terminal,
        }


def require_canonical_action_prototype(
    value: Mapping[str, Any], *, label: str = "canonical action prototype"
) -> CanonicalActionPrototype:
    """Accept only entity/relation/phase/terminal tensors with exact ABI shapes."""

    if not isinstance(value, Mapping) or set(value.keys()) != set(ACTION_PROTOTYPE_FIELDS):
        _fail(
            "%s must contain exactly q_entity/q_relation/q_phase/q_terminal; "
            "raw q_local and q_camera are forbidden" % label
        )
    shapes = {
        "q_entity": (ENTITY_SLOTS, LATENT_PHASES, ENTITY_WIDTH),
        "q_relation": (
            DIRECTED_RELATION_SLOTS,
            LATENT_PHASES,
            RELATION_WIDTH,
        ),
        "q_phase": (LATENT_PHASES, PHASE_WIDTH),
        "q_terminal": (TERMINAL_SLOTS, TERMINAL_WIDTH),
    }
    tensors: Dict[str, Any] = {}
    batch = None
    device = None
    for field in ACTION_PROTOTYPE_FIELDS:
        tensor = _require_finite_floating_tensor(
            value[field], label="%s.%s" % (label, field)
        )
        expected_tail = shapes[field]
        if tensor.ndim != len(expected_tail) + 1 or tuple(tensor.shape[1:]) != expected_tail:
            _fail("%s.%s has the wrong ELAL-3 ABI shape" % (label, field))
        if batch is None:
            batch = int(tensor.shape[0])
            device = tensor.device
        if int(tensor.shape[0]) != batch or tensor.device != device:
            _fail("%s fields must share batch and device" % label)
        tensors[field] = tensor
    return CanonicalActionPrototype(**tensors)


def prototype_alignment_losses(
    student: Mapping[str, Any],
    target: Mapping[str, Any],
    target_validity: Mapping[str, Any],
    target_participant_binding: TargetParticipantBinding,
) -> Dict[str, MaskedAlignmentTerm]:
    """Return separately normalized losses for the four canonical action views."""

    student_prototype = require_canonical_action_prototype(
        student, label="student action prototype"
    )
    target_prototype = require_canonical_action_prototype(
        target, label="target action prototype"
    )
    if not isinstance(target_validity, Mapping) or set(target_validity.keys()) != set(
        ACTION_PROTOTYPE_FIELDS
    ):
        _fail("prototype target validity must have the exact four action fields")
    student_fields = student_prototype.as_dict()
    target_fields = target_prototype.as_dict()
    if any(not bool(tensor.requires_grad) for tensor in student_fields.values()):
        _fail("student action prototype fields must retain optimizer gradient paths")
    if any(bool(tensor.requires_grad) for tensor in target_fields.values()):
        _fail("target action prototype fields must be stop-gradient evidence")
    batch = int(student_prototype.q_entity.shape[0])
    participant_binding = _revalidate_target_participant_binding(
        target_participant_binding
    )
    if len(participant_binding.causal_participant_ids_by_row) != batch:
        _fail("target participant binding batch differs from action prototypes")
    if participant_binding.entity_validity.device != student_prototype.q_entity.device:
        _fail("target participant binding is on the wrong action device")
    expected_validity_shapes = {
        "q_entity": (batch, ENTITY_SLOTS, LATENT_PHASES),
        "q_relation": (batch, DIRECTED_RELATION_SLOTS, LATENT_PHASES),
        "q_phase": (batch, LATENT_PHASES),
        "q_terminal": (batch, TERMINAL_SLOTS),
    }
    validity_fields: Dict[str, Any] = {}
    for field in ACTION_PROTOTYPE_FIELDS:
        validity_fields[field] = _snapshot_boolean_tensor(
            target_validity[field],
            label="prototype.%s target validity" % field,
            expected_shape=expected_validity_shapes[field],
            expected_device=student_fields[field].device,
        )
    # relation_valid is not an independently editable loss mask.  It is the
    # deterministic validity of the six directed i->j edges whose endpoints
    # are both target-valid at the same phase.  This both admits a legitimate
    # single-actor row (all six false) and prevents a multi-actor row from
    # erasing relation supervision with an all-false mask.
    entity_validity = validity_fields["q_entity"]
    relation_validity = validity_fields["q_relation"]
    if not bool(
        (entity_validity == participant_binding.entity_validity).all().item()
    ):
        _fail("prototype.q_entity validity differs from admission binding")
    expected_relation_validity = participant_binding.relation_validity
    if not bool((relation_validity == expected_relation_validity).all().item()):
        _fail("prototype.q_relation validity differs from target entity endpoints")
    return {
        field: _target_side_masked_term(
            student_fields[field],
            target_fields[field],
            validity_fields[field],
            label="prototype.%s" % field,
            allow_structurally_empty_rows=(field == "q_relation"),
        )
        for field in ACTION_PROTOTYPE_FIELDS
    }


@dataclass(frozen=True)
class ActionView:
    """Closed output/plan action projection, excluding camera nuisance.

    This is ``Q_action`` from the box contract.  It is deliberately distinct
    from :class:`CanonicalActionPrototype`: output and plan alignment retain
    the target-coordinate local action grid, whereas cross-appearance anchors
    are never allowed to contribute a raw local grid.
    """

    q_local: Any
    q_entity: Any
    q_relation: Any
    q_phase: Any
    q_terminal: Any

    def as_dict(self) -> Dict[str, Any]:
        return {
            "q_local": self.q_local,
            "q_entity": self.q_entity,
            "q_relation": self.q_relation,
            "q_phase": self.q_phase,
            "q_terminal": self.q_terminal,
        }


def require_action_view(
    value: Mapping[str, Any], *, label: str = "action view"
) -> ActionView:
    """Validate the exact ELAL-3 output/plan action projection.

    ``q_camera`` is a nuisance/preservation view and is therefore rejected,
    just like unknown fields.  Callers must explicitly project a full encoder
    result to these five action fields before connecting it to an objective.
    """

    if not isinstance(value, Mapping) or set(value.keys()) != set(ACTION_VIEW_FIELDS):
        _fail(
            "%s must contain exactly q_local/q_entity/q_relation/q_phase/"
            "q_terminal; q_camera and unknown fields are forbidden" % label
        )
    local = _require_finite_floating_tensor(value["q_local"], label=label + ".q_local")
    if (
        local.ndim != 5
        or int(local.shape[0]) <= 0
        or int(local.shape[1]) != LATENT_PHASES
        or int(local.shape[2]) <= 0
        or int(local.shape[3]) <= 0
        or int(local.shape[4]) != LOCAL_WIDTH
    ):
        _fail("%s.q_local must have shape [B,%d,h,w,%d]" % (label, LATENT_PHASES, LOCAL_WIDTH))
    prototype = require_canonical_action_prototype(
        {field: value[field] for field in ACTION_PROTOTYPE_FIELDS},
        label=label + ".prototype",
    )
    if any(
        int(tensor.shape[0]) != int(local.shape[0]) or tensor.device != local.device
        for tensor in prototype.as_dict().values()
    ):
        _fail("%s fields must share batch and device" % label)
    return ActionView(q_local=local, **prototype.as_dict())


def action_alignment_losses(
    student: Mapping[str, Any],
    target: Mapping[str, Any],
    target_event_binding: TargetEventMaskBinding,
    target_structured_validity: Mapping[str, Any],
    target_participant_binding: TargetParticipantBinding,
    *,
    label: str,
) -> Dict[str, MaskedAlignmentTerm]:
    """Return the five separately normalized ``Q_action`` losses.

    This primitive is used for both ``L_output_action`` and ``L_plan``.  The
    local field is supervised only at detached, non-empty clean-target event
    cells.  Entity/relation/phase/terminal denominators likewise come only
    from clean-target validity.  Student-predicted validity is intentionally
    not an argument, so an all-false prediction cannot erase its own loss.
    """

    if type(label) is not str or not label:
        _fail("action alignment label must be non-empty text")
    validated_event_binding = _revalidate_target_event_mask_binding(
        target_event_binding
    )
    student_view = require_action_view(student, label=label + " student")
    target_view = require_action_view(target, label=label + " target")
    student_fields = student_view.as_dict()
    target_fields = target_view.as_dict()
    if any(not bool(tensor.requires_grad) for tensor in student_fields.values()):
        _fail(label + " student fields must retain optimizer gradient paths")
    if any(bool(tensor.requires_grad) for tensor in target_fields.values()):
        _fail(label + " target fields must be stop-gradient evidence")
    local_shape = tuple(student_view.q_local.shape)
    if tuple(target_view.q_local.shape) != local_shape:
        _fail(label + " q_local geometry differs")
    bound_batch, _, bound_phases, bound_height, bound_width = (
        validated_event_binding.target_latent_shape
    )
    expected_local_shape = (
        bound_batch,
        bound_phases,
        bound_height // 2,
        bound_width // 2,
        LOCAL_WIDTH,
    )
    if local_shape != expected_local_shape:
        _fail(
            "%s q_local geometry is not the bound target latent native patch grid"
            % label
        )
    prototype_student = {
        field: student_fields[field] for field in ACTION_PROTOTYPE_FIELDS
    }
    prototype_target = {
        field: target_fields[field] for field in ACTION_PROTOTYPE_FIELDS
    }
    losses = {
        "q_local": _target_side_masked_term(
            student_view.q_local,
            target_view.q_local,
            validated_event_binding.action_event_mask,
            label=label + ".q_local",
            allow_structurally_empty_rows=False,
        )
    }
    losses.update(
        prototype_alignment_losses(
            prototype_student,
            prototype_target,
            target_structured_validity,
            target_participant_binding,
        )
    )
    return losses


def _normalized_field_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def reject_forbidden_legacy_fields(value: Any) -> None:
    """Recursively reject legacy objective field names in nested containers."""

    active: Set[int] = set()

    def visit(current: Any, path: str) -> None:
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in active:
                _fail("cyclic objective payload at %s" % path)
            active.add(identity)
            try:
                for key, nested in current.items():
                    if not isinstance(key, str):
                        _fail("objective payload key at %s must be text" % path)
                    normalized = _normalized_field_name(key)
                    if normalized in FORBIDDEN_LEGACY_FIELDS:
                        _fail("forbidden legacy objective field at %s.%s" % (path, key))
                    visit(nested, "%s.%s" % (path, key))
            finally:
                active.remove(identity)
            return
        if isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            identity = id(current)
            if identity in active:
                _fail("cyclic objective payload at %s" % path)
            active.add(identity)
            try:
                for index, nested in enumerate(current):
                    visit(nested, "%s[%d]" % (path, index))
            finally:
                active.remove(identity)

    visit(value, "$")


__all__ = (
    "ACTION_PROTOTYPE_FIELDS",
    "ACTION_VIEW_FIELDS",
    "ActionView",
    "CanonicalActionPrototype",
    "DIRECTED_RELATION_PAIRS",
    "Exact160ObjectiveError",
    "FORBIDDEN_LEGACY_FIELDS",
    "MaskedAlignmentTerm",
    "PartitionedFlowLoss",
    "RectifiedFlowState",
    "TargetEventMaskBinding",
    "TargetParticipantBinding",
    "action_alignment_losses",
    "bind_target_event_masks",
    "bind_target_participants",
    "noop_source_as_target_state",
    "partitioned_flow_losses",
    "partitioned_flow_losses_from_binding",
    "predicted_clean",
    "prototype_alignment_losses",
    "rectified_flow_state",
    "reject_forbidden_legacy_fields",
    "require_action_view",
    "require_canonical_action_prototype",
    "target_side_masked_loss",
)
