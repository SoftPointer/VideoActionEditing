#!/usr/bin/env python3
"""Zero-init dense motion-token residual branch for Bernini-R 1.3B.

The self-generated anchor contributes only RAFT flow and validity.  One motion
feature vector is retained for every Bernini target patch token; no RGB, VAE
latent, actor appearance, or pooled 32-D statistic enters the branch.

The Bernini transformer stays frozen.  Selected blocks receive an independent
bottleneck residual after their native forward.  The residual output
projection is byte-zero at initialization, and an explicit activity mask makes
source tokens, phase zero, and a zero motion condition exact native no-ops.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
import math
from pathlib import Path
import types
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


SCHEMA_VERSION = "bernini-dense-flow-token-adapter-v1"
STATE_SCHEMA_VERSION = "bernini-dense-flow-token-adapter-state-v1"
BLOCK_INDICES = (0, 4, 8, 12, 16, 20, 24, 28)
EXPECTED_BLOCK_COUNT = 30
HIDDEN_WIDTH = 1536
BOTTLENECK_WIDTH = 256
FEATURE_WIDTH = 12
LATENT_PHASES = 21
PATCH_SIZE = (1, 2, 2)
MODES = ("local_mlp", "phase_attention_8x12", "phase_attention_12x20")
ATTENTION_MEMORY_SHAPES = {
    "phase_attention_8x12": (8, 12),
    "phase_attention_12x20": (12, 20),
}
EXPECTED_TRAINABLE_PARAMETERS = 6_318_080
ATTENTION_TRAINABLE_PARAMETERS = 6_348_800


class DenseFlowAdapterError(RuntimeError):
    pass


def expected_trainable_parameters(
    mode: str, *, block_count: Optional[int] = None
) -> int:
    if mode not in MODES:
        raise DenseFlowAdapterError("dense-flow parameter-count mode differs")
    count = len(BLOCK_INDICES) if block_count is None else int(block_count)
    if count <= 0 or count > EXPECTED_BLOCK_COUNT:
        raise DenseFlowAdapterError("dense-flow block count differs")
    full = (
        ATTENTION_TRAINABLE_PARAMETERS
        if mode in ATTENTION_MEMORY_SHAPES
        else EXPECTED_TRAINABLE_PARAMETERS
    )
    if full % len(BLOCK_INDICES):
        raise DenseFlowAdapterError("dense-flow per-block parameter count differs")
    return full // len(BLOCK_INDICES) * count


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _finite_tensor(value: Any, *, label: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise DenseFlowAdapterError(f"{label} must be a rank-{ndim} tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
        raise DenseFlowAdapterError(f"{label} must be finite floating point")
    return value


def _backward_warp(value: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp a [C,H,W] field from the previous phase into the current phase."""

    if value.ndim != 3 or flow.ndim != 3 or int(flow.shape[0]) != 2:
        raise DenseFlowAdapterError("cumulative-flow warp geometry differs")
    height, width = map(int, flow.shape[-2:])
    if _shape(value)[-2:] != (height, width):
        raise DenseFlowAdapterError("cumulative-flow spatial geometry differs")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=torch.float32),
        torch.arange(width, device=flow.device, dtype=torch.float32),
        indexing="ij",
    )
    sample_x = xx + flow[0].float()
    sample_y = yy + flow[1].float()
    grid = torch.stack(
        (
            2.0 * sample_x / max(width - 1, 1) - 1.0,
            2.0 * sample_y / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    ).unsqueeze(0)
    return F.grid_sample(
        value.float().unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]


def _cumulative_backward(local: torch.Tensor) -> torch.Tensor:
    if _shape(local)[:2] != (LATENT_PHASES - 1, 2):
        raise DenseFlowAdapterError("local backward flow must be [20,2,H,W]")
    values = [torch.zeros_like(local[0])]
    for phase in range(LATENT_PHASES - 1):
        step = local[phase].float()
        values.append(step + _backward_warp(values[-1], step))
    return torch.stack(values).contiguous()


def dense_flow_features_from_tensors(
    backward_raw: torch.Tensor,
    backward_camera_residual: torch.Tensor,
    validity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full source+target token features and exact activity mask.

    Output shapes are ``[1,2*T*(H/2)*(W/2),12]`` and ``[1,N,1]``.  Source
    tokens and target phase zero are byte-exact positive zero.
    """

    raw = _finite_tensor(backward_raw, label="raw backward flow", ndim=4).float()
    camera = _finite_tensor(
        backward_camera_residual, label="camera-residual backward flow", ndim=4
    ).float()
    confidence = _finite_tensor(validity, label="flow validity", ndim=4).float()
    if raw.shape != camera.shape or _shape(raw)[:2] != (20, 2):
        raise DenseFlowAdapterError("raw/camera flow geometry differs")
    height, width = map(int, raw.shape[-2:])
    if _shape(confidence) != (20, 1, height, width):
        raise DenseFlowAdapterError("flow validity geometry differs")
    if height % 2 or width % 2:
        raise DenseFlowAdapterError("latent flow grid must be divisible by patch 2x2")
    if bool((confidence < 0).any().item()) or bool((confidence > 1).any().item()):
        raise DenseFlowAdapterError("flow validity must lie in [0,1]")

    zero_flow = torch.zeros((1, 2, height, width), dtype=torch.float32, device=raw.device)
    zero_valid = torch.zeros((1, 1, height, width), dtype=torch.float32, device=raw.device)
    local_raw = torch.cat((zero_flow, raw), dim=0)
    local_camera = torch.cat((zero_flow, camera), dim=0)
    phase_valid = torch.cat((zero_valid, confidence), dim=0)
    cumulative_raw = _cumulative_backward(raw)
    cumulative_camera = _cumulative_backward(camera)
    local_magnitude = torch.linalg.vector_norm(local_raw, dim=1, keepdim=True)
    cumulative_magnitude = torch.linalg.vector_norm(
        cumulative_camera, dim=1, keepdim=True
    )
    phase = torch.linspace(
        0.0, 1.0, LATENT_PHASES, dtype=torch.float32, device=raw.device
    ).view(LATENT_PHASES, 1, 1, 1).expand(-1, 1, height, width)

    # Bounded physical features retain sign, direction, local/cumulative
    # magnitude, camera decomposition, confidence, and temporal order.
    field = torch.cat(
        (
            torch.tanh(local_raw / 4.0),
            torch.tanh(cumulative_raw / 8.0),
            torch.tanh(local_camera / 4.0),
            torch.tanh(cumulative_camera / 8.0),
            torch.log1p(local_magnitude) / 4.0,
            torch.log1p(cumulative_magnitude) / 4.0,
            phase_valid,
            phase,
        ),
        dim=1,
    )
    if _shape(field) != (LATENT_PHASES, FEATURE_WIDTH, height, width):
        raise DenseFlowAdapterError("dense flow feature width differs")
    pooled = F.avg_pool2d(field, kernel_size=2, stride=2)
    pooled_valid = F.avg_pool2d(phase_valid, kernel_size=2, stride=2)
    target = pooled.permute(0, 2, 3, 1).reshape(1, -1, FEATURE_WIDTH).contiguous()
    target_activity = pooled_valid.permute(0, 2, 3, 1).reshape(1, -1, 1).ge(0.25)
    target_activity[:, : (height // 2) * (width // 2)] = False
    source = torch.zeros_like(target)
    source_activity = torch.zeros_like(target_activity)
    features = torch.cat((source, target), dim=1).contiguous()
    activity = torch.cat((source_activity, target_activity), dim=1).contiguous()
    if bool(torch.count_nonzero(features[:, : target.shape[1]]).item()):
        raise DenseFlowAdapterError("source motion features are not exact zero")
    if bool(activity[:, : target.shape[1]].any().item()):
        raise DenseFlowAdapterError("source motion activity is not exact false")
    phase0_end = target.shape[1] + (height // 2) * (width // 2)
    if bool(activity[:, target.shape[1] : phase0_end].any().item()):
        raise DenseFlowAdapterError("target phase-zero activity is not exact false")
    return features, activity


def load_dense_flow_features(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors.torch import load_file

    resolved = Path(path).expanduser().resolve(strict=True)
    tensors = load_file(str(resolved), device="cpu")
    if set(tensors) != {"backward_raw", "backward_camera_residual", "validity"}:
        raise DenseFlowAdapterError("flow bundle tensor-key closure differs")
    return dense_flow_features_from_tensors(
        tensors["backward_raw"],
        tensors["backward_camera_residual"],
        tensors["validity"],
    )


@dataclass(frozen=True)
class DenseFlowInvocation:
    features: torch.Tensor = field(repr=False, compare=False)
    activity: torch.Tensor = field(repr=False, compare=False)
    mode: str = "local_mlp"
    spatial_shape: Optional[tuple[int, int]] = None

    def validate(self) -> None:
        features = _finite_tensor(self.features, label="motion features", ndim=3)
        if int(features.shape[0]) != 1 or int(features.shape[2]) != FEATURE_WIDTH:
            raise DenseFlowAdapterError("motion features must be [1,N,12]")
        if (
            not isinstance(self.activity, torch.Tensor)
            or self.activity.dtype != torch.bool
            or _shape(self.activity) != (1, int(features.shape[1]), 1)
        ):
            raise DenseFlowAdapterError("motion activity must be bool [1,N,1]")
        if self.activity.device != features.device:
            raise DenseFlowAdapterError("motion activity/features device differs")
        if self.mode not in MODES:
            raise DenseFlowAdapterError("dense-flow invocation mode differs")
        target_tokens = int(features.shape[1]) // 2
        if target_tokens % LATENT_PHASES:
            raise DenseFlowAdapterError("dense-flow target tokens lack 21 phases")
        if self.mode in ATTENTION_MEMORY_SHAPES:
            if (
                self.spatial_shape is None
                or len(self.spatial_shape) != 2
                or min(map(int, self.spatial_shape)) <= 0
                or math.prod(map(int, self.spatial_shape))
                != target_tokens // LATENT_PHASES
            ):
                raise DenseFlowAdapterError(
                    "phase-attention requires exact patch spatial shape"
                )


_CURRENT: contextvars.ContextVar[Optional[DenseFlowInvocation]] = contextvars.ContextVar(
    "bernini_dense_flow_invocation", default=None
)


@contextlib.contextmanager
def dense_flow_invocation(invocation: DenseFlowInvocation) -> Iterator[DenseFlowInvocation]:
    if not isinstance(invocation, DenseFlowInvocation):
        raise DenseFlowAdapterError("dense-flow context received the wrong type")
    invocation.validate()
    if _CURRENT.get() is not None:
        raise DenseFlowAdapterError("nested dense-flow invocations are forbidden")
    token = _CURRENT.set(invocation)
    try:
        yield invocation
    finally:
        _CURRENT.reset(token)


def current_dense_flow_invocation() -> Optional[DenseFlowInvocation]:
    return _CURRENT.get()


class DenseFlowResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_width: int = HIDDEN_WIDTH,
        feature_width: int = FEATURE_WIDTH,
        bottleneck_width: int = BOTTLENECK_WIDTH,
    ) -> None:
        super().__init__()
        self.hidden_width = int(hidden_width)
        self.feature_width = int(feature_width)
        self.bottleneck_width = int(bottleneck_width)
        if min(self.hidden_width, self.feature_width, self.bottleneck_width) <= 0:
            raise DenseFlowAdapterError("adapter widths must be positive")
        self.norm = nn.LayerNorm(self.hidden_width, elementwise_affine=False)
        self.hidden_down = nn.Linear(self.hidden_width, self.bottleneck_width, bias=False)
        self.flow_in = nn.Linear(self.feature_width, self.bottleneck_width, bias=True)
        self.output = nn.Linear(self.bottleneck_width, self.hidden_width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        features: torch.Tensor,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        hidden = _finite_tensor(hidden_states, label="adapter hidden states", ndim=3)
        motion = _finite_tensor(features, label="adapter motion features", ndim=3)
        if _shape(hidden)[:2] != _shape(motion)[:2] or int(hidden.shape[2]) != self.hidden_width:
            raise DenseFlowAdapterError("adapter hidden/motion geometry differs")
        if int(motion.shape[2]) != self.feature_width:
            raise DenseFlowAdapterError("adapter motion feature width differs")
        if activity.dtype != torch.bool or _shape(activity) != (*_shape(hidden)[:2], 1):
            raise DenseFlowAdapterError("adapter activity geometry differs")
        if hidden.device != motion.device or hidden.device != activity.device:
            raise DenseFlowAdapterError("adapter tensors changed device")
        # Keep the small trainable branch in FP32 even when the frozen Bernini
        # trunk runs BF16.  This avoids silently casting the zero-init output
        # gate or dense flow values to a lower-precision optimizer coordinate.
        motion = motion.float()
        z = F.silu(self.hidden_down(self.norm(hidden.float())) + self.flow_in(motion))
        delta = self.output(z)
        delta = torch.where(activity, delta, torch.zeros_like(delta))
        return (hidden.float() + delta.float()).to(hidden.dtype)

    def is_zero_effect(self) -> bool:
        return bool(torch.count_nonzero(self.output.weight.detach()).item() == 0)


class DenseFlowPhaseAttentionResidualBlock(nn.Module):
    """Query source-aware target tokens against full spatial action memory.

    Unlike the local MLP, this branch does not assume that an anchor flow at
    coordinate ``(x,y)`` belongs at the same coordinate in the edited source.
    Each target token retrieves from an ordered, per-phase flow memory.  The
    memory contains only physical flow/validity/phase features plus normalized
    anchor coordinates; it never reads anchor RGB or VAE appearance.
    """

    def __init__(
        self,
        *,
        mode: str,
        hidden_width: int = HIDDEN_WIDTH,
        feature_width: int = FEATURE_WIDTH,
        bottleneck_width: int = BOTTLENECK_WIDTH,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if mode not in ATTENTION_MEMORY_SHAPES:
            raise DenseFlowAdapterError("phase-attention mode differs")
        if bottleneck_width % heads:
            raise DenseFlowAdapterError("phase-attention head width differs")
        self.mode = mode
        self.memory_shape = ATTENTION_MEMORY_SHAPES[mode]
        self.hidden_width = int(hidden_width)
        self.feature_width = int(feature_width)
        self.bottleneck_width = int(bottleneck_width)
        self.heads = int(heads)
        self.norm_target = nn.LayerNorm(self.hidden_width, elementwise_affine=False)
        self.norm_memory = nn.LayerNorm(self.feature_width + 2, elementwise_affine=False)
        self.query = nn.Linear(self.hidden_width, self.bottleneck_width, bias=False)
        self.key = nn.Linear(self.feature_width + 2, self.bottleneck_width, bias=False)
        self.value = nn.Linear(self.feature_width + 2, self.bottleneck_width, bias=False)
        self.output = nn.Linear(self.bottleneck_width, self.hidden_width, bias=False)
        nn.init.zeros_(self.output.weight)

    def _memory(
        self, global_features: torch.Tensor, spatial_shape: tuple[int, int]
    ) -> torch.Tensor:
        height, width = map(int, spatial_shape)
        target_tokens = LATENT_PHASES * height * width
        if (
            global_features.ndim != 3
            or int(global_features.shape[1]) != 2 * target_tokens
            or int(global_features.shape[2]) != self.feature_width
        ):
            raise DenseFlowAdapterError("phase-attention memory geometry differs")
        batch = int(global_features.shape[0])
        target = global_features[:, target_tokens:].float().reshape(
            batch, LATENT_PHASES, height, width, self.feature_width
        ).permute(0, 1, 4, 2, 3)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=target.device),
            torch.linspace(-1.0, 1.0, width, device=target.device),
            indexing="ij",
        )
        position = torch.stack((xx, yy), dim=0).view(1, 1, 2, height, width)
        position = position.expand(batch, LATENT_PHASES, -1, -1, -1)
        field = torch.cat((target, position), dim=2).reshape(
            batch * LATENT_PHASES, self.feature_width + 2, height, width
        )
        pooled = F.adaptive_avg_pool2d(field, self.memory_shape)
        memory_tokens = math.prod(self.memory_shape)
        return pooled.reshape(
            batch, LATENT_PHASES, self.feature_width + 2, memory_tokens
        ).permute(0, 1, 3, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
        activity: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> torch.Tensor:
        if (
            hidden_states.ndim != 3
            or local_features.ndim != 3
            or tuple(local_features.shape[:2]) != tuple(hidden_states.shape[:2])
            or int(local_features.shape[2]) != self.feature_width
            or activity.dtype != torch.bool
            or tuple(activity.shape) != (*hidden_states.shape[:2], 1)
        ):
            raise DenseFlowAdapterError("phase-attention tensor geometry differs")
        target = self.norm_target(hidden_states.float())
        memory = self.norm_memory(
            self._memory(global_features.to(device=hidden_states.device), spatial_shape)
        )
        batch, tokens, _ = target.shape
        if int(memory.shape[0]) != batch:
            raise DenseFlowAdapterError("phase-attention batch geometry differs")
        head_width = self.bottleneck_width // self.heads
        query = self.query(target)
        phase = local_features[..., -1].float().mul(LATENT_PHASES - 1).round().long()
        if bool(((phase < 0) | (phase >= LATENT_PHASES))[activity[..., 0]].any().item()):
            raise DenseFlowAdapterError("active phase-attention token has invalid phase")
        retrieved = torch.zeros(
            batch,
            tokens,
            self.bottleneck_width,
            # Under the production BF16 autocast path the Q/K/V projections
            # return BF16 even though their normalized inputs are FP32.
            # index_copy_ requires an exact dtype match with value_phase.
            dtype=query.dtype,
            device=hidden_states.device,
        )
        for batch_index in range(batch):
            for phase_index in range(1, LATENT_PHASES):
                selected = torch.nonzero(
                    activity[batch_index, :, 0]
                    & phase[batch_index].eq(phase_index),
                    as_tuple=False,
                ).flatten()
                if not int(selected.numel()):
                    continue
                query_phase = query[batch_index : batch_index + 1, selected].reshape(
                    1, int(selected.numel()), self.heads, head_width
                ).transpose(1, 2)
                memory_phase = memory[
                    batch_index : batch_index + 1, phase_index
                ]
                memory_tokens = int(memory_phase.shape[1])
                key = self.key(memory_phase).reshape(
                    1, memory_tokens, self.heads, head_width
                ).transpose(1, 2)
                value = self.value(memory_phase).reshape(
                    1, memory_tokens, self.heads, head_width
                ).transpose(1, 2)
                value_phase = F.scaled_dot_product_attention(
                    query_phase, key, value, dropout_p=0.0, is_causal=False
                ).transpose(1, 2).reshape(
                    int(selected.numel()), self.bottleneck_width
                )
                retrieved[batch_index].index_copy_(0, selected, value_phase)
        delta = self.output(retrieved)
        delta = torch.where(activity, delta, torch.zeros_like(delta))
        return (hidden_states.float() + delta).to(hidden_states.dtype)

    def is_zero_effect(self) -> bool:
        return bool(torch.count_nonzero(self.output.weight.detach()).item() == 0)


def _resolve_transformer(model: Any) -> Any:
    queue = [model]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        blocks = getattr(candidate, "blocks", None)
        if blocks is not None:
            if len(blocks) != EXPECTED_BLOCK_COUNT:
                raise DenseFlowAdapterError("Bernini transformer block count differs")
            return candidate
        getter = getattr(candidate, "get_base_model", None)
        if callable(getter):
            try:
                queue.append(getter())
            except Exception:
                pass
        for name in ("diff_dec", "transformer", "base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
    raise DenseFlowAdapterError("could not resolve Bernini 30-block transformer")


def _local_motion(invocation: DenseFlowInvocation, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Preserve flow coordinates in FP32.  The frozen trunk may run BF16, but
    # the adapter immediately computes in FP32 and should not quantize the
    # physical conditioning field on every block invocation.
    features = invocation.features.to(device=hidden.device, dtype=torch.float32)
    activity = invocation.activity.to(device=hidden.device)
    if int(features.shape[1]) == int(hidden.shape[1]):
        return features, activity
    try:
        from bernini.parallel import padding_tensor_for_seqeunce_parallel, slice_input_tensor
    except ImportError as error:
        raise DenseFlowAdapterError("global motion features require Bernini SP helpers") from error
    features = slice_input_tensor(
        padding_tensor_for_seqeunce_parallel(features, dim=1), dim=1
    )
    activity = slice_input_tensor(
        padding_tensor_for_seqeunce_parallel(activity, dim=1), dim=1
    )
    if _shape(features)[:2] != _shape(hidden)[:2]:
        raise DenseFlowAdapterError("rank-local motion shard differs from hidden states")
    return features, activity.bool()


@dataclass
class DenseFlowPatchHandle:
    transformer: Any
    block_indices: tuple[int, ...]
    adapters: tuple[nn.Module, ...]
    original_forwards: tuple[Any, ...] = field(repr=False)
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise DenseFlowAdapterError("adapter patch is already restored")
        rows: list[tuple[str, nn.Parameter]] = []
        for index, adapter in zip(self.block_indices, self.adapters):
            for name, parameter in adapter.named_parameters():
                rows.append((f"blocks.{index}.dense_flow_adapter.{name}", parameter))
        if not rows or len({id(parameter) for _, parameter in rows}) != len(rows):
            raise DenseFlowAdapterError("adapter trainable parameter closure differs")
        return tuple(rows)

    def base_is_frozen(self) -> bool:
        trainable = {id(parameter) for _, parameter in self.trainable_named_parameters()}
        return all(
            id(parameter) in trainable or not parameter.requires_grad
            for parameter in self.transformer.parameters()
        )

    def zero_effect(self) -> bool:
        return all(adapter.is_zero_effect() for adapter in self.adapters)

    def state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().float().cpu().contiguous()
            for name, parameter in self.trainable_named_parameters()
        }

    def load_state_dict_strict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = dict(self.trainable_named_parameters())
        if set(state) != set(expected):
            raise DenseFlowAdapterError("adapter state-key closure differs")
        with torch.no_grad():
            for name, parameter in expected.items():
                value = state[name]
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                    raise DenseFlowAdapterError(f"adapter state geometry differs: {name}")
                if not bool(torch.isfinite(value).all().item()):
                    raise DenseFlowAdapterError(f"adapter state is non-finite: {name}")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def restore(self) -> None:
        if self.restored:
            return
        for index, adapter, original in zip(
            self.block_indices, self.adapters, self.original_forwards
        ):
            block = self.transformer.blocks[index]
            if getattr(block, "dense_flow_adapter", None) is not adapter:
                raise DenseFlowAdapterError("adapter module changed behind patch handle")
            block.forward = original
            delattr(block, "dense_flow_adapter")
        self.restored = True


def install_dense_flow_adapter(
    model: Any,
    *,
    mode: str = "local_mlp",
    block_indices: Sequence[int] = BLOCK_INDICES,
    hidden_width: int = HIDDEN_WIDTH,
    bottleneck_width: int = BOTTLENECK_WIDTH,
) -> DenseFlowPatchHandle:
    if mode not in MODES:
        raise DenseFlowAdapterError("dense-flow install mode differs")
    transformer = _resolve_transformer(model)
    transformer.requires_grad_(False)
    indices = tuple(int(item) for item in block_indices)
    if indices != tuple(sorted(set(indices))) or any(
        item < 0 or item >= EXPECTED_BLOCK_COUNT for item in indices
    ):
        raise DenseFlowAdapterError("adapter block indices must be sorted unique in [0,29]")
    adapters: list[nn.Module] = []
    originals: list[Any] = []
    installed: list[int] = []
    try:
        for index in indices:
            block = transformer.blocks[index]
            if hasattr(block, "dense_flow_adapter"):
                raise DenseFlowAdapterError(f"block {index} already has a dense-flow adapter")
            adapter = (
                DenseFlowPhaseAttentionResidualBlock(
                    mode=mode,
                    hidden_width=hidden_width,
                    bottleneck_width=bottleneck_width,
                )
                if mode in ATTENTION_MEMORY_SHAPES
                else DenseFlowResidualBlock(
                    hidden_width=hidden_width,
                    bottleneck_width=bottleneck_width,
                )
            )
            block.add_module("dense_flow_adapter", adapter)
            original = block.forward

            def wrapped_forward(
                self: Any,
                *args: Any,
                _original: Any = original,
                _adapter: nn.Module = adapter,
                **kwargs: Any,
            ) -> torch.Tensor:
                hidden = _original(*args, **kwargs)
                invocation = current_dense_flow_invocation()
                if invocation is None:
                    return hidden
                if invocation.mode != mode:
                    raise DenseFlowAdapterError(
                        "dense-flow invocation/install mode differs"
                    )
                features, activity = _local_motion(invocation, hidden)
                if mode in ATTENTION_MEMORY_SHAPES:
                    assert invocation.spatial_shape is not None
                    return _adapter(
                        hidden,
                        features,
                        invocation.features,
                        activity,
                        invocation.spatial_shape,
                    )
                return _adapter(hidden, features, activity)

            block.forward = types.MethodType(wrapped_forward, block)
            adapters.append(adapter)
            originals.append(original)
            installed.append(index)
    except Exception:
        for index, adapter, original in zip(
            reversed(installed), reversed(adapters), reversed(originals)
        ):
            block = transformer.blocks[index]
            block.forward = original
            if getattr(block, "dense_flow_adapter", None) is adapter:
                delattr(block, "dense_flow_adapter")
        raise
    handle = DenseFlowPatchHandle(
        transformer=transformer,
        block_indices=indices,
        adapters=tuple(adapters),
        original_forwards=tuple(originals),
    )
    if not handle.base_is_frozen() or not handle.zero_effect():
        handle.restore()
        raise DenseFlowAdapterError("adapter freeze/zero-init closure differs")
    return handle


__all__ = [
    "ATTENTION_MEMORY_SHAPES",
    "ATTENTION_TRAINABLE_PARAMETERS",
    "BLOCK_INDICES",
    "BOTTLENECK_WIDTH",
    "DenseFlowAdapterError",
    "DenseFlowInvocation",
    "DenseFlowPatchHandle",
    "DenseFlowPhaseAttentionResidualBlock",
    "DenseFlowResidualBlock",
    "FEATURE_WIDTH",
    "HIDDEN_WIDTH",
    "LATENT_PHASES",
    "MODES",
    "SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "current_dense_flow_invocation",
    "dense_flow_features_from_tensors",
    "dense_flow_invocation",
    "expected_trainable_parameters",
    "install_dense_flow_adapter",
    "load_dense_flow_features",
]
