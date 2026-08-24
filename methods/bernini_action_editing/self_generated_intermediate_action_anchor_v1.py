#!/usr/bin/env python3
"""Frozen, zero-training intermediate action anchors for Bernini-R 1.3B.

The runtime represented here is deliberately *not* a generated-video donor.
For selected solver steps, a source-conditioned teacher forward is observed at
blocks 15 and 22.  On the exact same noisy state, timestep, rotary tensor and
source condition, one extra forward uses the canonical no-op instruction.  The
only teacher quantity retained is the detached action-minus-no-op intermediate
contrast.

Block 15 supplies spatial activity used to form deterministic object slots.
Block 22 supplies temporally high-pass semantic values.  Static temporal DC,
per-phase spatial common mode, and phase zero are removed before the values are
compressed to object masks, phase/slot vectors, and a dynamic interaction
graph.  Raw teacher hidden states and the teacher's terminal latent are not
part of the packet ABI.

The packet may be injected only into block-22 target rows of the positive
student branch.  Source/reference/padding rows remain bit-exact.  A smooth
mid-sigma gate and two RMS ceilings make the intervention small relative to
the frozen student's own hidden distribution.  Scale zero is a hard object
identity bypass and callers must not install the runtime patch for P0.

This file contains the tensor/layout core and a narrow hook controller.  It is
independent of checkpoint loading and video decode so it can be unit tested on
CPU before an AUH native canary is authorized.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn


METHOD = "bernini-self-generated-intermediate-action-anchor-v1"
SCHEMA_VERSION = "bernini-intermediate-action-anchor-packet-v1"
RECEIPT_SCHEMA = "bernini-intermediate-action-anchor-runtime-receipt-v1"

TOTAL_BLOCKS = 30
GEOMETRY_BLOCK = 15
SEMANTIC_BLOCK = 22
LATENT_PHASES = 21
HIDDEN_SIZE = 1536
DEFAULT_CAPTURE_STEPS = tuple(range(8, 28))
CANONICAL_NOOP_INSTRUCTION = (
    "Keep the source video exactly unchanged, including every subject, "
    "appearance, action, camera motion, background, timing, and composition."
)
CANONICAL_NOOP_SHA256 = (
    "fb5f23b5b9de175696cff019f035e81eb1ee6a1123db7e3b63afb604b88daf3a"
)
_ADMISSION_TOKEN = object()


class IntermediateActionAnchorError(RuntimeError):
    """Raised before an ambiguous capture or unsafe injection is accepted."""


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
        raise IntermediateActionAnchorError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise IntermediateActionAnchorError(f"{label} must be a finite float")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise IntermediateActionAnchorError(
            f"{label} must be a finite float"
        ) from error
    if not math.isfinite(result):
        raise IntermediateActionAnchorError(f"{label} must be finite")
    return result


def _exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise IntermediateActionAnchorError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise IntermediateActionAnchorError(f"{label} must be an integer") from error
    if result != value:
        raise IntermediateActionAnchorError(f"{label} must be exact")
    return result


def tensor_sha256(value: torch.Tensor) -> str:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise IntermediateActionAnchorError("hashed tensor must be finite and real")
    # ``expand`` can produce a size-one tensor with stride zero.  PyTorch may
    # still report that tensor as contiguous, so ``contiguous()`` alone does
    # not guarantee a byte-viewable storage layout.  Clone explicitly before
    # viewing as bytes; values, dtype and logical shape remain unchanged.
    owned = value.detach().to(device="cpu").clone(
        memory_format=torch.contiguous_format
    ).contiguous()
    payload = owned.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    header = canonical_json_bytes(
        {"dtype": str(owned.dtype), "shape": list(map(int, owned.shape))}
    )
    return hashlib.sha256(header + b"\0" + payload).hexdigest()


def bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    return bool(
        torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def validate_canonical_noop(instruction: Any, digest: Any) -> None:
    if (
        instruction != CANONICAL_NOOP_INSTRUCTION
        or digest != CANONICAL_NOOP_SHA256
        or hashlib.sha256(str(instruction).encode("utf-8")).hexdigest() != digest
    ):
        raise IntermediateActionAnchorError("canonical no-op authority differs")


_FORBIDDEN_TARGET_KEYS = frozenset(
    {
        "target",
        "target_video",
        "target_video_path",
        "target_rgb",
        "target_frames",
        "target_latent",
        "target_latents",
        "target_hidden",
        "target_embedding",
        "target_embeddings",
        "target_q",
        "target_k",
        "target_v",
        "target_flow",
        "target_mask",
        "target_track",
        "target_trajectory",
        "teacher_video",
        "teacher_rgb",
        "teacher_decode",
    }
)


def assert_target_isolation_payload(value: Any, *, path: str = "payload") -> None:
    """Reject any real-target or decoded-teacher input at the runtime boundary."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise IntermediateActionAnchorError(
                    f"{path} contains a non-text key"
                )
            if raw_key.casefold() in _FORBIDDEN_TARGET_KEYS:
                raise IntermediateActionAnchorError(
                    f"forbidden runtime input: {path}.{raw_key}"
                )
            assert_target_isolation_payload(child, path=f"{path}.{raw_key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            assert_target_isolation_payload(child, path=f"{path}[{index}]")


@dataclass(frozen=True)
class AnchorConfig:
    """Pinned representation and conservative injection controls."""

    patch_height: int
    patch_width: int
    hidden_size: int = HIDDEN_SIZE
    phases: int = LATENT_PHASES
    object_slots: int = 4
    geometry_block: int = GEOMETRY_BLOCK
    semantic_block: int = SEMANTIC_BLOCK
    capture_steps: tuple[int, ...] = DEFAULT_CAPTURE_STEPS
    spatial_sigma_patches: float = 4.0
    semantic_logit_weight: float = 0.35
    graph_distance_sigma_patches: float = 6.0
    graph_message_gain: float = 0.15
    activity_gate_multiplier: float = 2.5
    sigma_zero_below: float = 0.18
    sigma_full_from: float = 0.35
    sigma_full_to: float = 0.72
    sigma_zero_above: float = 0.90
    default_scale: float = 0.06
    max_injection_to_hidden_rms: float = 0.025
    max_injection_to_temporal_rms: float = 0.20
    min_teacher_delta_rms: float = 1.0e-6
    min_retained_fraction: float = 1.0e-4
    max_retained_fraction: float = 1.05

    def validate(self) -> None:
        integer_fields = (
            ("patch_height", self.patch_height),
            ("patch_width", self.patch_width),
            ("hidden_size", self.hidden_size),
            ("phases", self.phases),
            ("object_slots", self.object_slots),
            ("geometry_block", self.geometry_block),
            ("semantic_block", self.semantic_block),
        )
        for label, value in integer_fields:
            if _exact_int(value, label=label) <= 0 and label not in {
                "geometry_block",
                "semantic_block",
            }:
                raise IntermediateActionAnchorError(f"{label} must be positive")
        if (
            self.phases != LATENT_PHASES
            or self.geometry_block != GEOMETRY_BLOCK
            or self.semantic_block != SEMANTIC_BLOCK
            or self.geometry_block >= self.semantic_block
            or self.semantic_block >= TOTAL_BLOCKS
        ):
            raise IntermediateActionAnchorError(
                "hook layers must be Bernini block15 geometry and block22 semantics"
            )
        if not self.capture_steps or tuple(sorted(set(self.capture_steps))) != tuple(
            self.capture_steps
        ):
            raise IntermediateActionAnchorError(
                "capture_steps must be a non-empty increasing unique tuple"
            )
        if any(type(step) is not int or not 0 <= step < 40 for step in self.capture_steps):
            raise IntermediateActionAnchorError("capture step lies outside exact40")
        if self.object_slots > self.patch_height * self.patch_width:
            raise IntermediateActionAnchorError("object slot count exceeds patch grid")
        for label in (
            "spatial_sigma_patches",
            "graph_distance_sigma_patches",
            "activity_gate_multiplier",
            "max_injection_to_hidden_rms",
            "max_injection_to_temporal_rms",
            "min_teacher_delta_rms",
            "min_retained_fraction",
            "max_retained_fraction",
        ):
            if _finite_float(getattr(self, label), label=label) <= 0.0:
                raise IntermediateActionAnchorError(f"{label} must be positive")
        if not 0.0 <= self.semantic_logit_weight <= 1.0:
            raise IntermediateActionAnchorError(
                "semantic_logit_weight must lie in [0,1]"
            )
        if not 0.0 <= self.graph_message_gain <= 0.5:
            raise IntermediateActionAnchorError("graph_message_gain is unsafe")
        if not 0.0 <= self.default_scale <= 0.25:
            raise IntermediateActionAnchorError("default_scale is unsafe")
        if not (
            0.0
            <= self.sigma_zero_below
            < self.sigma_full_from
            <= self.sigma_full_to
            < self.sigma_zero_above
            <= 1.0
        ):
            raise IntermediateActionAnchorError("sigma band-pass gate differs")
        if self.min_retained_fraction > self.max_retained_fraction:
            raise IntermediateActionAnchorError("retained-fraction bounds differ")

    @property
    def patch_positions(self) -> int:
        return int(self.patch_height) * int(self.patch_width)

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        value = {
            "patch_grid": [self.patch_height, self.patch_width],
            "hidden_size": self.hidden_size,
            "phases": self.phases,
            "object_slots": self.object_slots,
            "geometry_block": self.geometry_block,
            "semantic_block": self.semantic_block,
            "capture_steps": list(self.capture_steps),
            "spatial_sigma_patches": self.spatial_sigma_patches,
            "semantic_logit_weight": self.semantic_logit_weight,
            "graph_distance_sigma_patches": self.graph_distance_sigma_patches,
            "graph_message_gain": self.graph_message_gain,
            "activity_gate_multiplier": self.activity_gate_multiplier,
            "sigma_gate": [
                self.sigma_zero_below,
                self.sigma_full_from,
                self.sigma_full_to,
                self.sigma_zero_above,
            ],
            "default_scale": self.default_scale,
            "max_injection_to_hidden_rms": self.max_injection_to_hidden_rms,
            "max_injection_to_temporal_rms": self.max_injection_to_temporal_rms,
            "teacher_delta_rms_bounds": [
                self.min_teacher_delta_rms,
                self.min_retained_fraction,
                self.max_retained_fraction,
            ],
        }
        return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class LocalTokenLayout:
    """Official append-pad/contiguous-SP layout for one condition+target branch."""

    condition_tokens: int
    patch_height: int
    patch_width: int
    phases: int
    sp_rank: int
    sp_size: int
    local_length: int
    shard_start: int
    total_tokens: int
    local_target_indices: torch.Tensor = field(repr=False)
    target_phase_indices: torch.Tensor = field(repr=False)
    target_patch_indices: torch.Tensor = field(repr=False)
    condition_rows: int
    padding_rows: int

    @classmethod
    def build(
        cls,
        *,
        condition_tokens: int,
        patch_height: int,
        patch_width: int,
        phases: int = LATENT_PHASES,
        sp_rank: int = 0,
        sp_size: int = 1,
        observed_local_length: Optional[int] = None,
    ) -> "LocalTokenLayout":
        values = {
            "condition_tokens": condition_tokens,
            "patch_height": patch_height,
            "patch_width": patch_width,
            "phases": phases,
            "sp_rank": sp_rank,
            "sp_size": sp_size,
        }
        for label, value in values.items():
            _exact_int(value, label=label)
        if (
            condition_tokens < 0
            or patch_height <= 0
            or patch_width <= 0
            or phases != LATENT_PHASES
            or sp_size not in (1, 4)
            or not 0 <= sp_rank < sp_size
        ):
            raise IntermediateActionAnchorError("native token layout differs")
        target_tokens = phases * patch_height * patch_width
        total = condition_tokens + target_tokens
        local_length = math.ceil(total / sp_size)
        if observed_local_length is not None and observed_local_length != local_length:
            raise IntermediateActionAnchorError("observed SP local length differs")
        start = sp_rank * local_length
        indices = torch.arange(local_length, dtype=torch.int64)
        global_indices = start + indices
        real = global_indices < total
        target = real & (global_indices >= condition_tokens)
        local_target = indices[target].contiguous()
        target_flat = (global_indices[target] - condition_tokens).contiguous()
        patch_positions = patch_height * patch_width
        phase = torch.div(target_flat, patch_positions, rounding_mode="floor")
        patch = torch.remainder(target_flat, patch_positions)
        condition_rows = int((real & (global_indices < condition_tokens)).sum().item())
        padding_rows = int((~real).sum().item())
        if (
            condition_rows + padding_rows + int(local_target.numel()) != local_length
            or (phase.numel() and int(phase.max().item()) >= phases)
            or (patch.numel() and int(patch.max().item()) >= patch_positions)
        ):
            raise IntermediateActionAnchorError("target-row partition differs")
        return cls(
            condition_tokens=condition_tokens,
            patch_height=patch_height,
            patch_width=patch_width,
            phases=phases,
            sp_rank=sp_rank,
            sp_size=sp_size,
            local_length=local_length,
            shard_start=start,
            total_tokens=total,
            local_target_indices=local_target,
            target_phase_indices=phase.contiguous(),
            target_patch_indices=patch.contiguous(),
            condition_rows=condition_rows,
            padding_rows=padding_rows,
        )

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "condition_tokens": self.condition_tokens,
            "target_tokens": self.phases * self.patch_height * self.patch_width,
            "total_tokens": self.total_tokens,
            "patch_grid": [self.patch_height, self.patch_width],
            "phases": self.phases,
            "sp_rank": self.sp_rank,
            "sp_size": self.sp_size,
            "local_length": self.local_length,
            "shard_start": self.shard_start,
            "selected_target_rows": int(self.local_target_indices.numel()),
            "condition_rows": self.condition_rows,
            "padding_rows": self.padding_rows,
            "global_order": "condition-prefix_then-phase-major-y-x-target",
            "sp_policy": "append-pad_then-contiguous-rank-chunk",
        }
        return {**value, "digest": object_sha256(value)}


def extract_local_target(hidden: torch.Tensor, layout: LocalTokenLayout) -> torch.Tensor:
    if (
        not isinstance(hidden, torch.Tensor)
        or hidden.ndim != 3
        or tuple(hidden.shape[:2]) != (1, layout.local_length)
        or not hidden.is_floating_point()
        or not bool(torch.isfinite(hidden.detach()).all().item())
    ):
        raise IntermediateActionAnchorError(
            "local hidden must be finite [1,local_length,D]"
        )
    selected = layout.local_target_indices.to(device=hidden.device)
    return hidden.index_select(1, selected).detach().contiguous()


def assemble_sp_target_grid(
    local_target_shards: Sequence[torch.Tensor],
    layouts: Sequence[LocalTokenLayout],
) -> torch.Tensor:
    """Assemble target-only SP shards; intended after one authenticated gather."""

    if (
        not isinstance(local_target_shards, Sequence)
        or not isinstance(layouts, Sequence)
        or len(local_target_shards) != len(layouts)
        or len(layouts) not in (1, 4)
    ):
        raise IntermediateActionAnchorError("SP target shard closure differs")
    ordered = sorted(zip(layouts, local_target_shards), key=lambda row: row[0].sp_rank)
    first_layout = ordered[0][0]
    if [row[0].sp_rank for row in ordered] != list(range(len(ordered))):
        raise IntermediateActionAnchorError("SP ranks are missing or repeated")
    width: Optional[int] = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    target_tokens = first_layout.phases * first_layout.patch_height * first_layout.patch_width
    result: Optional[torch.Tensor] = None
    seen = torch.zeros(target_tokens, dtype=torch.int64)
    for layout, shard in ordered:
        if (
            layout.sp_size != len(ordered)
            or layout.patch_height != first_layout.patch_height
            or layout.patch_width != first_layout.patch_width
            or layout.phases != first_layout.phases
            or layout.condition_tokens != first_layout.condition_tokens
            or not isinstance(shard, torch.Tensor)
            or shard.ndim != 3
            or int(shard.shape[0]) != 1
            or int(shard.shape[1]) != int(layout.local_target_indices.numel())
            or not bool(torch.isfinite(shard.detach()).all().item())
        ):
            raise IntermediateActionAnchorError("SP target shard geometry differs")
        if width is None:
            width = int(shard.shape[2])
            device = shard.device
            dtype = shard.dtype
            result = torch.empty((1, target_tokens, width), dtype=dtype, device=device)
        elif (
            int(shard.shape[2]) != width
            or shard.device != device
            or shard.dtype != dtype
        ):
            raise IntermediateActionAnchorError("SP target shard dtype/device differs")
        global_flat = (
            layout.target_phase_indices * first_layout.patch_height * first_layout.patch_width
            + layout.target_patch_indices
        )
        if global_flat.numel():
            result.index_copy_(1, global_flat.to(device=device), shard)
            seen.index_add_(0, global_flat.cpu(), torch.ones_like(global_flat.cpu()))
    if result is None or not torch.equal(seen, torch.ones_like(seen)):
        raise IntermediateActionAnchorError("SP target rows do not cover exactly once")
    return result.reshape(
        1,
        first_layout.phases,
        first_layout.patch_height,
        first_layout.patch_width,
        int(width),
    ).contiguous()


def smooth_bandpass_gate(sigma: float, config: AnchorConfig) -> float:
    config.validate()
    value = _finite_float(sigma, label="sigma")
    if not 0.0 <= value <= 1.0 + 1.0e-6:
        raise IntermediateActionAnchorError("sigma lies outside [0,1]")

    def smoothstep(x: float) -> float:
        clipped = min(1.0, max(0.0, x))
        return clipped * clipped * (3.0 - 2.0 * clipped)

    rise = smoothstep(
        (value - config.sigma_zero_below)
        / (config.sigma_full_from - config.sigma_zero_below)
    )
    fall = 1.0 - smoothstep(
        (value - config.sigma_full_to)
        / (config.sigma_zero_above - config.sigma_full_to)
    )
    return float(rise * fall)


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def _validate_teacher_grid(
    value: torch.Tensor, config: AnchorConfig, *, label: str
) -> torch.Tensor:
    expected = (
        1,
        config.phases,
        config.patch_height,
        config.patch_width,
        config.hidden_size,
    )
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != expected
        or not value.is_floating_point()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise IntermediateActionAnchorError(
            f"{label} must be detached finite {list(expected)}"
        )
    return value.detach().float()


def _appearance_and_camera_null(delta: torch.Tensor) -> torch.Tensor:
    # Per-location temporal DC is the static appearance coordinate.  The
    # remaining per-phase spatial mean is the global/camera coordinate.
    temporal = delta - delta.mean(dim=1, keepdim=True)
    camera = temporal.mean(dim=(2, 3), keepdim=True)
    return temporal - camera


def _deterministic_centers(activity: torch.Tensor, slots: int) -> torch.Tensor:
    """Top-activity centers with deterministic farthest-point suppression."""

    if activity.ndim != 2 or slots <= 0 or slots > activity.numel():
        raise IntermediateActionAnchorError("activity-center geometry differs")
    height, width = map(int, activity.shape)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=activity.device),
        torch.arange(width, dtype=torch.float32, device=activity.device),
        indexing="ij",
    )
    flat_activity = activity.reshape(-1)
    centers: list[torch.Tensor] = []
    for index in range(slots):
        if index == 0:
            selected = int(torch.argmax(flat_activity).item())
        else:
            stacked = torch.stack(centers)
            distance = (
                (yy.unsqueeze(-1) - stacked[:, 0]) ** 2
                + (xx.unsqueeze(-1) - stacked[:, 1]) ** 2
            ).amin(dim=-1)
            normalized_activity = flat_activity.reshape(height, width)
            denominator = normalized_activity.max().clamp_min(1.0e-12)
            score = distance.sqrt() * (0.25 + 0.75 * normalized_activity / denominator)
            for center in centers:
                score[int(center[0].item()), int(center[1].item())] = -1.0
            selected = int(torch.argmax(score.reshape(-1)).item())
        y, x = divmod(selected, width)
        centers.append(
            torch.tensor([float(y), float(x)], device=activity.device)
        )
    return torch.stack(centers).float()


@dataclass
class IntermediateActionAnchorPacket:
    """Compressed object/graph packet; it contains no raw teacher hidden."""

    config: AnchorConfig
    step_index: int
    sigma: float
    responsibilities: torch.Tensor = field(repr=False)  # [1,T,P,K]
    activity_gate: torch.Tensor = field(repr=False)  # [1,T,P,1]
    slot_values: torch.Tensor = field(repr=False)  # [1,T,K,D]
    centers_yx: torch.Tensor = field(repr=False)  # [1,T,K,2]
    interaction_graph: torch.Tensor = field(repr=False)  # [1,T,K,K]
    reconstruction_scale: float
    quality: Mapping[str, Any]

    def validate(self) -> None:
        self.config.validate()
        _exact_int(self.step_index, label="step_index")
        if self.step_index not in self.config.capture_steps:
            raise IntermediateActionAnchorError("packet step is not selected")
        sigma = _finite_float(self.sigma, label="packet sigma")
        if not 0.0 <= sigma <= 1.0 + 1.0e-6:
            raise IntermediateActionAnchorError("packet sigma differs")
        t = self.config.phases
        p = self.config.patch_positions
        k = self.config.object_slots
        d = self.config.hidden_size
        expected = {
            "responsibilities": ((1, t, p, k), self.responsibilities),
            "activity_gate": ((1, t, p, 1), self.activity_gate),
            "slot_values": ((1, t, k, d), self.slot_values),
            "centers_yx": ((1, t, k, 2), self.centers_yx),
            "interaction_graph": ((1, t, k, k), self.interaction_graph),
        }
        for label, (shape, tensor) in expected.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise IntermediateActionAnchorError(
                    f"packet {label} geometry/value differs"
                )
        if not bool(
            torch.allclose(
                self.responsibilities.sum(dim=-1),
                torch.ones_like(self.activity_gate[..., 0]),
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        ):
            raise IntermediateActionAnchorError("slot responsibilities do not sum to one")
        if bool((self.activity_gate < 0).any().item()) or bool(
            (self.activity_gate > 1).any().item()
        ):
            raise IntermediateActionAnchorError("activity gate lies outside [0,1]")
        graph_sum = self.interaction_graph.sum(dim=-1)
        if not bool(torch.allclose(graph_sum, torch.ones_like(graph_sum), atol=1e-5, rtol=1e-5)):
            raise IntermediateActionAnchorError("interaction graph is not row stochastic")
        if _finite_float(self.reconstruction_scale, label="reconstruction_scale") < 0.0:
            raise IntermediateActionAnchorError("reconstruction scale is negative")
        if self.quality.get("admitted") is not True:
            raise IntermediateActionAnchorError("packet did not pass its quality guard")

    def local_residual(
        self,
        phase_indices: torch.Tensor,
        patch_indices: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        self.validate()
        if (
            not isinstance(phase_indices, torch.Tensor)
            or not isinstance(patch_indices, torch.Tensor)
            or phase_indices.dtype != torch.int64
            or patch_indices.dtype != torch.int64
            or phase_indices.shape != patch_indices.shape
            or phase_indices.ndim != 1
        ):
            raise IntermediateActionAnchorError("local residual index geometry differs")
        phase = phase_indices.to(device=self.responsibilities.device)
        patch = patch_indices.to(device=self.responsibilities.device)
        weights = self.responsibilities[0, phase, patch, :]
        slots = self.slot_values[0, phase, :, :]
        residual = torch.einsum("nk,nkd->nd", weights, slots)
        residual = residual * self.activity_gate[0, phase, patch, :]
        residual = residual * float(self.reconstruction_scale)
        residual = residual.clone()
        residual[phase == 0] = 0.0
        return residual.to(device=device, dtype=torch.float32).unsqueeze(0).contiguous()

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        tensors = {
            "responsibilities": self.responsibilities,
            "activity_gate": self.activity_gate,
            "slot_values": self.slot_values,
            "centers_yx": self.centers_yx,
            "interaction_graph": self.interaction_graph,
        }
        tensor_rows = {
            name: {
                "shape": list(map(int, tensor.shape)),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor),
            }
            for name, tensor in tensors.items()
        }
        value = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "step_index": self.step_index,
            "sigma_float64_hex": float(self.sigma).hex(),
            "config_digest": self.config.receipt()["digest"],
            "representation": {
                "geometry_source": "block15_action_minus_same_state_noop",
                "semantic_source": "block22_action_minus_same_state_noop",
                "appearance_null": "per_patch_temporal_dc_removed",
                "camera_null": "per_phase_spatial_common_mode_removed",
                "phase0_hard_zero": True,
                "object_slots": self.config.object_slots,
                "dynamic_row_stochastic_interaction_graph": True,
                "raw_teacher_hidden_retained": False,
                "teacher_latent_or_rgb_retained": False,
            },
            "reconstruction_scale": self.reconstruction_scale,
            "quality": dict(self.quality),
            "tensors": tensor_rows,
        }
        return {**value, "digest": object_sha256(value)}


def _sha256_text(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IntermediateActionAnchorError(f"{label} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class SourceViewPacketEvidence:
    """One packet tied to a persistent source and one source-reference view."""

    view_id: str
    source_video_sha256: str
    reference_frame_indices: tuple[int, ...]
    packet: IntermediateActionAnchorPacket

    def validate(self) -> None:
        if not isinstance(self.view_id, str) or not self.view_id:
            raise IntermediateActionAnchorError("source view id must be nonempty")
        _sha256_text(self.source_video_sha256, label="source video")
        if (
            not isinstance(self.reference_frame_indices, tuple)
            or len(self.reference_frame_indices) != 4
            or tuple(sorted(set(self.reference_frame_indices)))
            != self.reference_frame_indices
            or any(
                type(index) is not int or not 0 <= index <= 80
                for index in self.reference_frame_indices
            )
        ):
            raise IntermediateActionAnchorError(
                "source view must bind four increasing exact81 frame indices"
            )
        self.packet.validate()

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        value = {
            "view_id": self.view_id,
            "source_video_sha256": self.source_video_sha256,
            "reference_frame_indices": list(self.reference_frame_indices),
            "packet_digest": self.packet.receipt()["digest"],
            "persistent_source_binding": True,
            "reference_view_is_source_derived": True,
        }
        return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class MultiViewControlEvidence:
    """Pre-injection controls; P0 replay remains a post-injection release gate."""

    noop_vs_noop_delta_rms: float
    action_delta_rms_reference: float
    teacher_observer_action_output_bit_exact: bool
    frozen_state_before_sha256: str
    frozen_state_after_teacher_sha256: str
    target_inputs_absent: bool

    def validate(self) -> None:
        noop = _finite_float(
            self.noop_vs_noop_delta_rms, label="noop-vs-noop delta RMS"
        )
        action_rms = _finite_float(
            self.action_delta_rms_reference, label="action delta RMS reference"
        )
        if noop < 0.0 or action_rms <= 0.0:
            raise IntermediateActionAnchorError("control RMS values differ")
        _sha256_text(self.frozen_state_before_sha256, label="freeze-before")
        _sha256_text(
            self.frozen_state_after_teacher_sha256, label="freeze-after-teacher"
        )
        if (
            self.teacher_observer_action_output_bit_exact is not True
            or self.target_inputs_absent is not True
            or self.frozen_state_before_sha256
            != self.frozen_state_after_teacher_sha256
        ):
            raise IntermediateActionAnchorError(
                "teacher observer/freeze/target-isolation control failed"
            )


@dataclass(frozen=True)
class MultiViewControlAdmission:
    """Unforgeable permission for one primary packet/source binding."""

    primary_packet_digest: str
    alternate_packet_digest: str
    source_video_sha256: str
    metrics: Mapping[str, Any]
    receipt_digest: str
    _token: Any = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self._token is not _ADMISSION_TOKEN:
            raise IntermediateActionAnchorError("multi-view admission is not authentic")
        for label, value in (
            ("primary packet", self.primary_packet_digest),
            ("alternate packet", self.alternate_packet_digest),
            ("source video", self.source_video_sha256),
            ("admission receipt", self.receipt_digest),
        ):
            _sha256_text(value, label=label)
        if self.metrics.get("admitted") is not True:
            raise IntermediateActionAnchorError("multi-view admission failed")

    def assert_packet(
        self,
        packet: IntermediateActionAnchorPacket,
        *,
        source_video_sha256: str,
    ) -> None:
        self.validate()
        if (
            packet.receipt()["digest"] != self.primary_packet_digest
            or _sha256_text(source_video_sha256, label="live source video")
            != self.source_video_sha256
        ):
            raise IntermediateActionAnchorError(
                "packet/live source differs from multi-view admission"
            )

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        return {
            "primary_packet_digest": self.primary_packet_digest,
            "alternate_packet_digest": self.alternate_packet_digest,
            "source_video_sha256": self.source_video_sha256,
            "metrics": dict(self.metrics),
            "receipt_digest": self.receipt_digest,
            "p0_exact_replay_required_post_injection": True,
        }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    if left_flat.shape != right_flat.shape:
        raise IntermediateActionAnchorError("admission feature shapes differ")
    denominator = left_flat.norm() * right_flat.norm()
    if float(denominator.item()) <= 1.0e-12:
        raise IntermediateActionAnchorError("admission feature is degenerate")
    value = float(((left_flat @ right_flat) / denominator).item())
    return min(1.0, max(-1.0, value))


def _permutation_invariant_packet_features(
    packet: IntermediateActionAnchorPacket,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase energy plus sorted slot/edge affinities, invariant to slot labels."""

    packet.validate()
    slots = packet.slot_values.float()
    phase_energy = slots.square().mean(dim=(2, 3)).sqrt().reshape(-1)
    slot_unit = torch.nn.functional.normalize(slots, dim=-1, eps=1.0e-6)
    slot_cosine = torch.einsum("btkd,btjd->btkj", slot_unit, slot_unit)
    k = packet.config.object_slots
    off_diagonal = ~torch.eye(k, dtype=torch.bool).reshape(1, 1, k, k)
    slot_signature = torch.sort(
        slot_cosine.masked_select(off_diagonal.expand_as(slot_cosine)).reshape(
            1, packet.config.phases, k * (k - 1)
        ),
        dim=-1,
    ).values
    graph_signature = torch.sort(
        packet.interaction_graph.masked_select(
            off_diagonal.expand_as(packet.interaction_graph)
        ).reshape(1, packet.config.phases, k * (k - 1)),
        dim=-1,
    ).values
    relational = torch.cat((slot_signature, graph_signature), dim=-1).reshape(-1)
    return phase_energy, relational


def admit_multiview_control(
    *,
    primary: SourceViewPacketEvidence,
    alternate: SourceViewPacketEvidence,
    controls: MultiViewControlEvidence,
    minimum_phase_energy_cosine: float = 0.70,
    minimum_relational_cosine: float = 0.75,
    maximum_noop_to_action_rms: float = 0.05,
) -> MultiViewControlAdmission:
    """Fail closed unless two source views and a no-op control agree."""

    primary.validate()
    alternate.validate()
    controls.validate()
    thresholds = (
        _finite_float(minimum_phase_energy_cosine, label="phase cosine threshold"),
        _finite_float(minimum_relational_cosine, label="relational cosine threshold"),
        _finite_float(maximum_noop_to_action_rms, label="no-op ratio threshold"),
    )
    if not (0.0 <= thresholds[0] <= 1.0 and 0.0 <= thresholds[1] <= 1.0):
        raise IntermediateActionAnchorError("admission cosine threshold differs")
    if not 0.0 <= thresholds[2] <= 0.25:
        raise IntermediateActionAnchorError("admission no-op threshold differs")
    if (
        primary.view_id == alternate.view_id
        or primary.reference_frame_indices == alternate.reference_frame_indices
        or primary.packet.receipt()["digest"]
        == alternate.packet.receipt()["digest"]
        or primary.source_video_sha256 != alternate.source_video_sha256
        or primary.packet.step_index != alternate.packet.step_index
        or primary.packet.config.receipt()["digest"]
        != alternate.packet.config.receipt()["digest"]
    ):
        raise IntermediateActionAnchorError(
            "multi-view evidence must use distinct source-reference views of one source"
        )
    primary_phase, primary_relational = _permutation_invariant_packet_features(
        primary.packet
    )
    alternate_phase, alternate_relational = _permutation_invariant_packet_features(
        alternate.packet
    )
    phase_cosine = _cosine(primary_phase, alternate_phase)
    relational_cosine = _cosine(primary_relational, alternate_relational)
    noop_ratio = float(
        controls.noop_vs_noop_delta_rms / controls.action_delta_rms_reference
    )
    admitted = (
        phase_cosine >= thresholds[0]
        and relational_cosine >= thresholds[1]
        and noop_ratio <= thresholds[2]
    )
    metrics = {
        "admitted": admitted,
        "source_view_count": 2,
        "distinct_reference_views": True,
        "phase_energy_cosine": phase_cosine,
        "minimum_phase_energy_cosine": thresholds[0],
        "permutation_invariant_relational_cosine": relational_cosine,
        "minimum_relational_cosine": thresholds[1],
        "noop_to_action_rms_ratio": noop_ratio,
        "maximum_noop_to_action_rms": thresholds[2],
        "teacher_observer_action_output_bit_exact": True,
        "frozen_state_unchanged_through_teacher": True,
        "target_inputs_absent": True,
        "fixed_slot_count_is_not_object_discovery": True,
        "typed_interaction_edges_absent": True,
        "persistent_entity_fsm_absent": True,
        "scientific_candidate": False,
    }
    if not admitted:
        raise IntermediateActionAnchorError(
            "packet failed multi-view/control admission"
        )
    value = {
        "primary": primary.receipt(),
        "alternate": alternate.receipt(),
        "controls": {
            "noop_vs_noop_delta_rms": controls.noop_vs_noop_delta_rms,
            "action_delta_rms_reference": controls.action_delta_rms_reference,
            "teacher_observer_action_output_bit_exact": (
                controls.teacher_observer_action_output_bit_exact
            ),
            "frozen_state_before_sha256": controls.frozen_state_before_sha256,
            "frozen_state_after_teacher_sha256": (
                controls.frozen_state_after_teacher_sha256
            ),
            "target_inputs_absent": controls.target_inputs_absent,
        },
        "metrics": metrics,
    }
    receipt_digest = object_sha256(value)
    result = MultiViewControlAdmission(
        primary_packet_digest=primary.packet.receipt()["digest"],
        alternate_packet_digest=alternate.packet.receipt()["digest"],
        source_video_sha256=primary.source_video_sha256,
        metrics=metrics,
        receipt_digest=receipt_digest,
    )
    object.__setattr__(result, "_token", _ADMISSION_TOKEN)
    result.validate()
    return result


def build_intermediate_action_anchor(
    *,
    geometry_action: torch.Tensor,
    geometry_noop: torch.Tensor,
    semantic_action: torch.Tensor,
    semantic_noop: torch.Tensor,
    config: AnchorConfig,
    step_index: int,
    sigma: float,
) -> IntermediateActionAnchorPacket:
    """Factor one same-state teacher contrast into slots and a dynamic graph."""

    config.validate()
    if _exact_int(step_index, label="step_index") not in config.capture_steps:
        raise IntermediateActionAnchorError("teacher step is not selected")
    sigma_value = _finite_float(sigma, label="sigma")
    geometry_action = _validate_teacher_grid(geometry_action, config, label="geometry action")
    geometry_noop = _validate_teacher_grid(geometry_noop, config, label="geometry no-op")
    semantic_action = _validate_teacher_grid(semantic_action, config, label="semantic action")
    semantic_noop = _validate_teacher_grid(semantic_noop, config, label="semantic no-op")

    geometry_delta = geometry_action - geometry_noop
    semantic_delta = semantic_action - semantic_noop
    teacher_delta_rms = float(_rms(semantic_delta).item())
    if teacher_delta_rms < config.min_teacher_delta_rms:
        raise IntermediateActionAnchorError("same-state teacher contrast is degenerate")
    geometry_signal = _appearance_and_camera_null(geometry_delta)
    semantic_signal = _appearance_and_camera_null(semantic_delta)

    # A small first temporal difference emphasizes motion boundaries without
    # replacing the semantically richer block-22 phase state.
    derivative = torch.zeros_like(semantic_signal)
    derivative[:, 1:] = semantic_signal[:, 1:] - semantic_signal[:, :-1]
    semantic_signal = semantic_signal + 0.25 * derivative
    semantic_signal = semantic_signal - semantic_signal.mean(dim=1, keepdim=True)
    semantic_signal[:, 0] = 0.0

    geometry_activity = geometry_signal.square().mean(dim=-1).sqrt()
    aggregate_activity = geometry_activity.mean(dim=(0, 1))
    centers = _deterministic_centers(aggregate_activity, config.object_slots)

    yy, xx = torch.meshgrid(
        torch.arange(config.patch_height, dtype=torch.float32, device=geometry_activity.device),
        torch.arange(config.patch_width, dtype=torch.float32, device=geometry_activity.device),
        indexing="ij",
    )
    distance2 = (
        (yy.unsqueeze(-1) - centers[:, 0]) ** 2
        + (xx.unsqueeze(-1) - centers[:, 1]) ** 2
    )
    spatial_logits = -distance2 / (2.0 * config.spatial_sigma_patches**2)

    # Fixed activity centers give deterministic slot prototypes; semantic
    # affinity then makes membership phase-dependent and object-centric.
    flat_signal = semantic_signal.reshape(
        1, config.phases, config.patch_positions, config.hidden_size
    )
    flat_activity = geometry_activity.reshape(1, config.phases, config.patch_positions)
    base_resp = torch.softmax(spatial_logits.reshape(1, 1, config.patch_positions, -1), dim=-1)
    proto_weight = base_resp * flat_activity.unsqueeze(-1)
    prototypes = torch.einsum("btpk,btpd->btkd", proto_weight, flat_signal)
    prototypes = prototypes / proto_weight.sum(dim=2).clamp_min(1.0e-6).unsqueeze(-1)
    token_unit = torch.nn.functional.normalize(flat_signal, dim=-1, eps=1.0e-6)
    proto_unit = torch.nn.functional.normalize(prototypes, dim=-1, eps=1.0e-6)
    semantic_affinity = torch.einsum("btpd,btkd->btpk", token_unit, proto_unit)
    logits = spatial_logits.reshape(1, 1, config.patch_positions, -1) + (
        config.semantic_logit_weight * semantic_affinity
    )
    responsibilities = torch.softmax(logits, dim=-1).float().contiguous()

    mean_activity = flat_activity.mean(dim=2, keepdim=True).clamp_min(1.0e-8)
    activity_gate = (
        flat_activity / (config.activity_gate_multiplier * mean_activity)
    ).clamp(0.0, 1.0).unsqueeze(-1).float().contiguous()
    slot_weight = responsibilities * flat_activity.unsqueeze(-1)
    slot_values = torch.einsum("btpk,btpd->btkd", slot_weight, flat_signal)
    slot_values = slot_values / slot_weight.sum(dim=2).clamp_min(1.0e-6).unsqueeze(-1)

    positions = torch.stack((yy, xx), dim=-1).reshape(config.patch_positions, 2)
    centers_by_phase = torch.einsum("btpk,pc->btkc", slot_weight, positions)
    centers_by_phase = centers_by_phase / slot_weight.sum(dim=2).clamp_min(1.0e-6).unsqueeze(-1)

    pair_delta = centers_by_phase.unsqueeze(3) - centers_by_phase.unsqueeze(2)
    pair_distance2 = pair_delta.square().sum(dim=-1)
    proximity = torch.exp(
        -pair_distance2 / (2.0 * config.graph_distance_sigma_patches**2)
    )
    slot_unit = torch.nn.functional.normalize(slot_values, dim=-1, eps=1.0e-6)
    semantic_pair = torch.einsum("btkd,btjd->btkj", slot_unit, slot_unit).clamp_min(0.0)
    interaction = proximity * (0.5 + 0.5 * semantic_pair)
    identity = torch.eye(config.object_slots, device=interaction.device).reshape(
        1, 1, config.object_slots, config.object_slots
    )
    interaction = interaction + identity
    interaction = interaction / interaction.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    messages = torch.einsum("btkj,btjd->btkd", interaction, slot_values)
    slot_values = (
        slot_values + config.graph_message_gain * (messages - slot_values)
    ).float().contiguous()

    provisional = torch.einsum("btpk,btkd->btpd", responsibilities, slot_values)
    provisional = provisional * activity_gate
    provisional[:, 0] = 0.0
    provisional_rms = float(_rms(provisional).item())
    semantic_signal_rms = float(_rms(semantic_signal).item())
    if provisional_rms <= 1.0e-12 or semantic_signal_rms <= 1.0e-12:
        raise IntermediateActionAnchorError("object action packet collapsed")
    reconstruction_scale = min(1.0, semantic_signal_rms / provisional_rms)
    reconstructed_rms = provisional_rms * reconstruction_scale
    retained_fraction = reconstructed_rms / teacher_delta_rms
    admitted = (
        config.min_retained_fraction
        <= retained_fraction
        <= config.max_retained_fraction
    )
    slot_mass = slot_weight.sum(dim=(0, 1, 2))
    quality = {
        "admitted": bool(admitted),
        "teacher_delta_rms": teacher_delta_rms,
        "geometry_action_minus_noop_rms": float(_rms(geometry_delta).item()),
        "appearance_camera_null_semantic_rms": semantic_signal_rms,
        "reconstructed_anchor_rms": reconstructed_rms,
        "retained_fraction": retained_fraction,
        "activity_gate_nonzero_fraction": float(
            (activity_gate > 0).float().mean().item()
        ),
        "activity_gate_mean": float(activity_gate.mean().item()),
        "slot_mass_min": float(slot_mass.min().item()),
        "slot_mass_max": float(slot_mass.max().item()),
        "static_appearance_transport": False,
        "spatial_common_mode_transport": False,
        "phase0_transport": False,
    }
    if not admitted:
        raise IntermediateActionAnchorError(
            "object action packet failed retained-energy quality guard"
        )
    packet = IntermediateActionAnchorPacket(
        config=config,
        step_index=step_index,
        sigma=sigma_value,
        responsibilities=responsibilities.detach().float().cpu().contiguous(),
        activity_gate=activity_gate.detach().float().cpu().contiguous(),
        slot_values=slot_values.detach().float().cpu().contiguous(),
        centers_yx=centers_by_phase.detach().float().cpu().contiguous(),
        interaction_graph=interaction.detach().float().cpu().contiguous(),
        reconstruction_scale=float(reconstruction_scale),
        quality=quality,
    )
    packet.validate()
    return packet


@dataclass(frozen=True)
class InjectionAudit:
    step_index: int
    sigma: float
    schedule_gate: float
    requested_scale: float
    clip_multiplier: float
    native_hidden_rms: float
    native_temporal_rms: float
    raw_anchor_rms: float
    applied_delta_rms: float
    selected_target_rows: int
    protected_rows: int
    protected_rows_bit_exact: bool
    phase0_rows_bit_exact: bool
    hard_bypass: bool

    def receipt(self) -> Mapping[str, Any]:
        value = dict(self.__dict__)
        return {**value, "digest": object_sha256(value)}


def inject_packet_into_local_hidden(
    native_hidden: torch.Tensor,
    *,
    packet: IntermediateActionAnchorPacket,
    admission: MultiViewControlAdmission,
    source_video_sha256: str,
    layout: LocalTokenLayout,
    sigma: float,
    scale: float,
) -> tuple[torch.Tensor, InjectionAudit]:
    """Inject one compressed packet into target rows with two quality ceilings."""

    packet.validate()
    admission.assert_packet(packet, source_video_sha256=source_video_sha256)
    requested_scale = _finite_float(scale, label="scale")
    if not 0.0 <= requested_scale <= 0.25:
        raise IntermediateActionAnchorError("injection scale lies outside [0,.25]")
    if (
        not isinstance(native_hidden, torch.Tensor)
        or native_hidden.ndim != 3
        or tuple(native_hidden.shape[:2]) != (1, layout.local_length)
        or int(native_hidden.shape[2]) != packet.config.hidden_size
        or not native_hidden.is_floating_point()
        or not bool(torch.isfinite(native_hidden.detach()).all().item())
        or layout.patch_height != packet.config.patch_height
        or layout.patch_width != packet.config.patch_width
        or layout.phases != packet.config.phases
    ):
        raise IntermediateActionAnchorError("student local hidden/layout differs")
    gate = smooth_bandpass_gate(sigma, packet.config)
    target_count = int(layout.local_target_indices.numel())
    protected_count = int(layout.local_length - target_count)
    if requested_scale == 0.0 or gate == 0.0 or target_count == 0:
        audit = InjectionAudit(
            step_index=packet.step_index,
            sigma=float(sigma),
            schedule_gate=gate,
            requested_scale=requested_scale,
            clip_multiplier=0.0,
            native_hidden_rms=float(_rms(native_hidden).item()),
            native_temporal_rms=0.0,
            raw_anchor_rms=0.0,
            applied_delta_rms=0.0,
            selected_target_rows=target_count,
            protected_rows=protected_count,
            protected_rows_bit_exact=True,
            phase0_rows_bit_exact=True,
            hard_bypass=True,
        )
        return native_hidden, audit

    target_index = layout.local_target_indices.to(device=native_hidden.device)
    phase = layout.target_phase_indices.to(device=native_hidden.device)
    patch = layout.target_patch_indices.to(device=native_hidden.device)
    anchor = packet.local_residual(phase.cpu(), patch.cpu(), device=native_hidden.device)
    target_native = native_hidden.index_select(1, target_index)
    raw_anchor_rms = float(_rms(anchor).item())
    hidden_rms = float(_rms(target_native).item())
    phase_centered = target_native.float().clone()
    for phase_index in range(packet.config.phases):
        mask = phase == phase_index
        if bool(mask.any().item()):
            phase_centered[:, mask, :] -= phase_centered[:, mask, :].mean(
                dim=1, keepdim=True
            )
    temporal_rms = float(_rms(phase_centered).item())
    proposed = anchor * (requested_scale * gate)
    proposed_rms = float(_rms(proposed).item())
    ceiling_hidden = packet.config.max_injection_to_hidden_rms * hidden_rms
    ceiling_temporal = packet.config.max_injection_to_temporal_rms * temporal_rms
    ceiling = min(ceiling_hidden, ceiling_temporal)
    if proposed_rms <= 1.0e-12 or ceiling <= 1.0e-12:
        multiplier = 0.0
    else:
        multiplier = min(1.0, ceiling / proposed_rms)
    applied = (proposed * multiplier).to(dtype=native_hidden.dtype)
    result = native_hidden.clone()
    result.index_copy_(1, target_index, target_native + applied)
    protected = torch.ones(layout.local_length, dtype=torch.bool, device=native_hidden.device)
    protected[target_index] = False
    protected_exact = bits_equal(result[:, protected, :], native_hidden[:, protected, :])
    phase0_local = target_index[phase == 0]
    phase0_exact = bits_equal(
        result.index_select(1, phase0_local), native_hidden.index_select(1, phase0_local)
    )
    if not protected_exact or not phase0_exact or not bool(torch.isfinite(result).all().item()):
        raise IntermediateActionAnchorError(
            "injection changed protected rows or produced non-finite hidden"
        )
    audit = InjectionAudit(
        step_index=packet.step_index,
        sigma=float(sigma),
        schedule_gate=gate,
        requested_scale=requested_scale,
        clip_multiplier=float(multiplier),
        native_hidden_rms=hidden_rms,
        native_temporal_rms=temporal_rms,
        raw_anchor_rms=raw_anchor_rms,
        applied_delta_rms=float(_rms(applied).item()),
        selected_target_rows=target_count,
        protected_rows=protected_count,
        protected_rows_bit_exact=protected_exact,
        phase0_rows_bit_exact=phase0_exact,
        hard_bypass=False,
    )
    return result, audit


def _output_tensor(output: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(output, torch.Tensor):
        return output, lambda value: value
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], lambda value: (value, *output[1:])
    raise IntermediateActionAnchorError(
        "Bernini block output must be Tensor or tensor-first tuple"
    )


@dataclass(frozen=True)
class HookInvocation:
    mode: str
    step_index: int
    sigma: float
    layout: LocalTokenLayout
    scale: float = 0.0
    packet: Optional[IntermediateActionAnchorPacket] = None
    admission: Optional[MultiViewControlAdmission] = None
    source_video_sha256: Optional[str] = None

    def validate(self) -> None:
        if self.mode not in {"capture_action", "capture_noop", "inject_student"}:
            raise IntermediateActionAnchorError("hook invocation mode differs")
        if type(self.step_index) is not int or not 0 <= self.step_index < 40:
            raise IntermediateActionAnchorError("hook step differs")
        _finite_float(self.sigma, label="hook sigma")
        if self.mode == "inject_student":
            if (
                self.packet is None
                or self.packet.step_index != self.step_index
                or self.admission is None
                or self.source_video_sha256 is None
            ):
                raise IntermediateActionAnchorError("student packet/step differs")
            self.admission.assert_packet(
                self.packet, source_video_sha256=self.source_video_sha256
            )
        elif (
            self.packet is not None
            or self.admission is not None
            or self.source_video_sha256 is not None
            or self.scale != 0.0
        ):
            raise IntermediateActionAnchorError("capture hook cannot inject")


class IntermediateAnchorHookController:
    """One-use block15/block22 capture and block22 student injection hooks.

    The native sampler adapter must activate exactly one invocation around one
    positive ``shared_step`` call.  Negative branches run with no invocation.
    SP4 callers gather the returned target-only shards and call
    :func:`assemble_sp_target_grid` before building a packet.
    """

    def __init__(self, transformer: nn.Module, config: AnchorConfig) -> None:
        config.validate()
        blocks = tuple(getattr(transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS:
            raise IntermediateActionAnchorError("transformer must expose 30 blocks")
        if any(parameter.requires_grad for parameter in transformer.parameters()):
            raise IntermediateActionAnchorError("freeze base before hook installation")
        self.transformer = transformer
        self.config = config
        self._handles: list[Any] = []
        self._active: Optional[HookInvocation] = None
        self._captures: dict[int, torch.Tensor] = {}
        self._audits: list[InjectionAudit] = []
        self._seen_blocks: set[int] = set()

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    def install(self) -> None:
        if self.installed or self._active is not None:
            raise IntermediateActionAnchorError("hook controller is already active")
        handles: list[Any] = []
        try:
            for index in (self.config.geometry_block, self.config.semantic_block):
                block = self.transformer.blocks[index]
                handles.append(block.register_forward_hook(self._make_hook(index)))
        except Exception:
            for handle in handles:
                handle.remove()
            raise
        self._handles = handles

    def remove(self) -> None:
        if not self.installed or self._active is not None:
            raise IntermediateActionAnchorError("cannot remove inactive/active hooks")
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def _make_hook(self, block_index: int) -> Any:
        def callback(_module: Any, _inputs: Any, output: Any) -> Any:
            invocation = self._active
            if invocation is None:
                return output
            if block_index in self._seen_blocks:
                raise IntermediateActionAnchorError("block hook fired repeatedly")
            native, rebuild = _output_tensor(output)
            if (
                native.ndim != 3
                or tuple(native.shape[:2]) != (1, invocation.layout.local_length)
                or int(native.shape[2]) != self.config.hidden_size
                or not bool(torch.isfinite(native.detach()).all().item())
            ):
                raise IntermediateActionAnchorError("hooked native hidden geometry differs")
            self._seen_blocks.add(block_index)
            if invocation.mode.startswith("capture_"):
                # Keep the shard on its native device so an SP4 adapter can
                # gather through NCCL.  The compressed packet moves to CPU
                # only after the same-state action/no-op delta is assembled.
                self._captures[block_index] = extract_local_target(
                    native, invocation.layout
                )
                return output
            if block_index != self.config.semantic_block:
                return output
            adapted, audit = inject_packet_into_local_hidden(
                native,
                packet=invocation.packet,
                admission=invocation.admission,
                source_video_sha256=invocation.source_video_sha256,
                layout=invocation.layout,
                sigma=invocation.sigma,
                scale=invocation.scale,
            )
            self._audits.append(audit)
            return rebuild(adapted)

        return callback

    @contextmanager
    def invoke(self, invocation: HookInvocation) -> Iterator[None]:
        invocation.validate()
        if not self.installed or self._active is not None:
            raise IntermediateActionAnchorError("hook invocation lifecycle differs")
        self._active = invocation
        self._captures.clear()
        self._seen_blocks.clear()
        audit_count_before = len(self._audits)
        try:
            yield
            expected = {self.config.geometry_block, self.config.semantic_block}
            if self._seen_blocks != expected:
                raise IntermediateActionAnchorError("selected hooks did not fire exactly once")
            if invocation.mode.startswith("capture_") and set(self._captures) != expected:
                raise IntermediateActionAnchorError("teacher hidden capture is incomplete")
            if (
                invocation.mode == "inject_student"
                and len(self._audits) != audit_count_before + 1
            ):
                raise IntermediateActionAnchorError("student injection audit is absent")
        finally:
            self._active = None
            self._seen_blocks.clear()

    def pop_captures(self) -> Mapping[int, torch.Tensor]:
        if self._active is not None:
            raise IntermediateActionAnchorError("cannot pop active captures")
        expected = {self.config.geometry_block, self.config.semantic_block}
        if set(self._captures) != expected:
            raise IntermediateActionAnchorError("capture closure differs")
        result = dict(self._captures)
        self._captures.clear()
        return result

    def pop_audits(self) -> tuple[InjectionAudit, ...]:
        if self._active is not None:
            raise IntermediateActionAnchorError("cannot pop active audits")
        result = tuple(self._audits)
        self._audits.clear()
        return result


class IntermediateAnchorTrajectoryBank:
    """Exact-step packet registry for one teacher/student seed and source."""

    def __init__(self, config: AnchorConfig) -> None:
        config.validate()
        self.config = config
        self._packets: dict[int, IntermediateActionAnchorPacket] = {}

    def add(self, packet: IntermediateActionAnchorPacket) -> None:
        packet.validate()
        if packet.config.receipt()["digest"] != self.config.receipt()["digest"]:
            raise IntermediateActionAnchorError("trajectory packet config differs")
        if packet.step_index in self._packets:
            raise IntermediateActionAnchorError("trajectory step already exists")
        self._packets[packet.step_index] = packet

    def get(self, step_index: int) -> Optional[IntermediateActionAnchorPacket]:
        return self._packets.get(step_index)

    def assert_complete(self) -> None:
        if tuple(sorted(self._packets)) != self.config.capture_steps:
            raise IntermediateActionAnchorError("teacher trajectory is incomplete")

    def receipt(self) -> Mapping[str, Any]:
        self.assert_complete()
        packet_rows = [self._packets[step].receipt() for step in self.config.capture_steps]
        value = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "config_digest": self.config.receipt()["digest"],
            "step_indices": list(self.config.capture_steps),
            "packet_digests": [row["digest"] for row in packet_rows],
            "teacher_terminal_latent_retained": False,
            "teacher_video_decoded": False,
            "raw_teacher_hidden_retained": False,
        }
        return {**value, "digest": object_sha256(value)}


def frozen_module_certificate(module: nn.Module) -> Mapping[str, Any]:
    """Small exact state certificate suitable before/after a canary."""

    if not isinstance(module, nn.Module) or module.training:
        raise IntermediateActionAnchorError("base model must be an eval nn.Module")
    rows = []
    for kind, iterator in (
        ("parameter", module.named_parameters()),
        ("buffer", module.named_buffers()),
    ):
        for name, tensor in iterator:
            if kind == "parameter" and (tensor.requires_grad or tensor.grad is not None):
                raise IntermediateActionAnchorError("base model is trainable")
            rows.append(
                {
                    "kind": kind,
                    "name": name,
                    "shape": list(map(int, tensor.shape)),
                    "dtype": str(tensor.dtype),
                    "sha256": tensor_sha256(tensor),
                }
            )
    value = {
        "base_frozen": True,
        "model_eval": True,
        "optimizer_absent": True,
        "trainable_parameter_count": 0,
        "state_rows": rows,
    }
    return {**value, "digest": object_sha256(value)}


def assert_p0_exact_replay(p0a: torch.Tensor, p0b: torch.Tensor) -> Mapping[str, Any]:
    if not bits_equal(p0a, p0b):
        raise IntermediateActionAnchorError("P0 frozen-base replay is not bit-exact")
    value = {
        "p0a_sha256": tensor_sha256(p0a),
        "p0b_sha256": tensor_sha256(p0b),
        "bit_exact": True,
        "scale_zero_hook_installation": False,
    }
    if value["p0a_sha256"] != value["p0b_sha256"]:
        raise IntermediateActionAnchorError("P0 replay digest differs")
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "AnchorConfig",
    "CANONICAL_NOOP_INSTRUCTION",
    "CANONICAL_NOOP_SHA256",
    "GEOMETRY_BLOCK",
    "HookInvocation",
    "InjectionAudit",
    "IntermediateActionAnchorError",
    "IntermediateActionAnchorPacket",
    "IntermediateAnchorHookController",
    "IntermediateAnchorTrajectoryBank",
    "LATENT_PHASES",
    "LocalTokenLayout",
    "METHOD",
    "MultiViewControlAdmission",
    "MultiViewControlEvidence",
    "SEMANTIC_BLOCK",
    "SourceViewPacketEvidence",
    "admit_multiview_control",
    "assemble_sp_target_grid",
    "assert_p0_exact_replay",
    "assert_target_isolation_payload",
    "bits_equal",
    "build_intermediate_action_anchor",
    "extract_local_target",
    "frozen_module_certificate",
    "inject_packet_into_local_hidden",
    "smooth_bandpass_gate",
    "validate_canonical_noop",
]
