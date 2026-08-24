#!/usr/bin/env python3
"""Strong packed-token preservation adapter for official Bernini-R 1.3B.

This module contains the model-free contract and the two trainable components
that are added around the pinned Bernini renderer:

* rank-256 LoRA on q/k/v/out in every one of the 30 Wan blocks; and
* explicit source/target patch deltas plus source/target role embeddings.

The main arm adapts both native self-attention (``attn1``) and text
cross-attention (``attn2``).  The registered capacity control adapts only
``attn1``.  In both arms LoRA is installed on the ordinary affine projections,
so it is evaluated for every local packed token on every SP rank.  There is no
target-row selector, routed early return, sparse block list, reward, or VLM.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-packed-preservation-lora-v2"
BERNINI_BLOCKS = 30
HIDDEN_SIZE = 1536
TEXT_WIDTH = 4096
PATCH_INPUT_CHANNELS = 16
PATCH_KERNEL = (1, 2, 2)
PATCH_VECTOR_WIDTH = 64
LORA_RANK = 256
LORA_ALPHA = 256
LORA_DROPOUT = 0.0
LORA_SCOPES = ("all-attention", "self-attention")
EXPECTED_MODULE_COUNTS = {
    "all-attention": BERNINI_BLOCKS * 2 * 4,
    "self-attention": BERNINI_BLOCKS * 4,
}
EXPECTED_LORA_PARAMETER_COUNTS = {
    "all-attention": 188_743_680,
    "self-attention": 94_371_840,
}
PATCH_ROLE_PARAMETER_COUNT = 202_752
EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNTS = {
    scope: count + PATCH_ROLE_PARAMETER_COUNT
    for scope, count in EXPECTED_LORA_PARAMETER_COUNTS.items()
}
PRETEXTS = ("noop", "cube", "speed", "tube")
PRETEXT_WEIGHTS = {"noop": 4, "cube": 2, "speed": 2, "tube": 2}
PRETEXT_CYCLE = tuple(
    name for name in PRETEXTS for _ in range(PRETEXT_WEIGHTS[name])
)
PRETEXT_INSTRUCTIONS = {
    "noop": (
        "Keep the source video exactly unchanged, including every subject, "
        "appearance, action, camera motion, background, timing, and composition."
    ),
    "cube": "Complete the missing regions in the video.",
    "speed": "Restore the video to normal playback speed.",
    "tube": "Restore the correct spatio-temporal order of the video segments.",
}
CHECKPOINT_STEPS_EXACT80 = (0, 20, 40, 60, 80)
CHECKPOINT_STEPS_CANARY2 = (0, 1, 2)
EXECUTION_SCOPES = ("optimizer-canary-2", "exact80")

_PROJECTION = re.compile(
    r"^(?P<prefix>(?:.+\.)?blocks\.(?P<block>\d+)\.attn(?P<attention>[12]))\."
    r"(?P<projection>to_q|to_k|to_v|to_out\.0)$"
)
_PROJECTIONS = ("to_q", "to_k", "to_v", "to_out.0")
_LAYOUT: ContextVar[Optional[tuple[int, int]]] = ContextVar(
    "bernini_packed_preservation_role_layout_v2", default=None
)
_OFFICIAL_FROZEN_NATIVE_ONLY_DEPTH: ContextVar[int] = ContextVar(
    "bernini_packed_preservation_official_frozen_native_only_depth_v2", default=0
)


class PackedPreservationV2Error(RuntimeError):
    """Raised before accepting an ambiguous model, sample, or update."""


def fail(message: str) -> NoReturn:
    raise PackedPreservationV2Error(message)


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
        raise PackedPreservationV2Error("non-canonical receipt value") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    block: int
    attention: int
    projection: str
    in_features: int
    out_features: int

    @property
    def lora_parameters(self) -> int:
        return LORA_RANK * (self.in_features + self.out_features)

    def receipt(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "block": self.block,
            "attention": self.attention,
            "projection": self.projection,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "rank": LORA_RANK,
            "trainable_parameters": self.lora_parameters,
        }


def _linear_shape(module: Any, *, name: str) -> tuple[int, int]:
    weight = getattr(module, "weight", None)
    shape = tuple(int(value) for value in getattr(weight, "shape", ()))
    if len(shape) != 2 or min(shape) <= 0:
        fail(f"attention projection is not one affine matrix: {name}")
    out_features, in_features = shape
    declared_in = getattr(module, "in_features", in_features)
    declared_out = getattr(module, "out_features", out_features)
    if int(declared_in) != in_features or int(declared_out) != out_features:
        fail(f"attention projection dimensions disagree: {name}")
    return in_features, out_features


def _qualified(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _validate_official_cross_attention_route(
    modules: Mapping[str, Any], specs: Sequence[ProjectionSpec]
) -> None:
    """Prove text is projected to 1536 before native attn2 q/k/v/out.

    In official Bernini-R 1.3B, the 4096-wide T5 feature is transformed by
    ``condition_embedder.text_embedder`` before entering every block.  The
    packed Wan processor then calls the ordinary 1536-wide ``to_k``/``to_v``;
    ``added_kv_proj_dim`` is null and no add-k/v affine is live.
    """

    cross = [item for item in specs if item.attention == 2]
    if len(cross) != BERNINI_BLOCKS * len(_PROJECTIONS):
        fail("official cross-attention affine inventory differs")
    expected_cross = {
        (block, projection)
        for block in range(BERNINI_BLOCKS)
        for projection in _PROJECTIONS
    }
    if {(item.block, item.projection) for item in cross} != expected_cross:
        fail("official cross-attention block/projection names differ")
    prefixes: set[str] = set()
    for item in cross:
        suffix = f"blocks.{item.block}.attn2.{item.projection}"
        if item.name == suffix:
            prefixes.add("")
        elif item.name.endswith(f".{suffix}"):
            prefixes.add(item.name[: -(len(suffix) + 1)])
        else:
            fail(f"cross-attention module name differs: {item.name}")
    if len(prefixes) != 1:
        fail("cross-attention modules do not share one transformer root")
    prefix = next(iter(prefixes))
    for block in range(BERNINI_BLOCKS):
        owner_name = _qualified(prefix, f"blocks.{block}.attn2")
        owner = modules.get(owner_name)
        if owner is None:
            fail(f"official attn2 owner is absent: {owner_name}")
        processor = getattr(owner, "processor", None)
        if processor is None or processor.__class__.__name__ != "WanAttnProcessor2_0":
            fail(f"official packed attn2 processor differs: {owner_name}")
        if any(
            getattr(owner, attribute, None) is not None
            for attribute in ("add_k_proj", "add_v_proj", "to_add_out")
        ):
            fail(f"attn2 added-k/v route must be absent: {owner_name}")
    text_shapes = {
        "condition_embedder.text_embedder.linear_1": (TEXT_WIDTH, HIDDEN_SIZE),
        "condition_embedder.text_embedder.linear_2": (HIDDEN_SIZE, HIDDEN_SIZE),
    }
    for suffix, expected in text_shapes.items():
        name = _qualified(prefix, suffix)
        module = modules.get(name)
        if module is None or _linear_shape(module, name=name) != expected:
            fail(f"official 4096-to-1536 text preprojection differs: {name}")


def select_projection_specs(model: Any, scope: str) -> tuple[ProjectionSpec, ...]:
    """Return an exact shape-audited all-block LoRA target set."""

    if scope not in LORA_SCOPES:
        fail(f"unsupported LoRA scope: {scope!r}")
    if not hasattr(model, "named_modules"):
        fail("model does not expose named_modules")
    named_modules = tuple(model.named_modules())
    if len({name for name, _ in named_modules}) != len(named_modules):
        fail("model exposes duplicate module names")
    modules = dict(named_modules)
    specs: list[ProjectionSpec] = []
    observed: dict[tuple[int, int], set[str]] = {}
    all_specs: list[ProjectionSpec] = []
    for name, module in named_modules:
        match = _PROJECTION.fullmatch(name)
        if match is None:
            continue
        attention = int(match.group("attention"))
        block = int(match.group("block"))
        projection = match.group("projection")
        in_features, out_features = _linear_shape(module, name=name)
        if (in_features, out_features) != (HIDDEN_SIZE, HIDDEN_SIZE):
            fail(
                f"official Bernini attention affine must be 1536x1536: {name} "
                f"has weight [{out_features},{in_features}]"
            )
        spec = ProjectionSpec(
            name=name,
            block=block,
            attention=attention,
            projection=projection,
            in_features=in_features,
            out_features=out_features,
        )
        all_specs.append(spec)
        if scope == "self-attention" and attention != 1:
            continue
        specs.append(spec)
        observed.setdefault((block, attention), set()).add(projection)
    _validate_official_cross_attention_route(modules, all_specs)
    expected_attentions = (1, 2) if scope == "all-attention" else (1,)
    expected_keys = {
        (block, attention)
        for block in range(BERNINI_BLOCKS)
        for attention in expected_attentions
    }
    if set(observed) != expected_keys:
        fail("LoRA target block/attention coverage differs from exact 30-block plan")
    if any(observed[key] != set(_PROJECTIONS) for key in expected_keys):
        fail("a selected attention does not contain exact q/k/v/out projections")
    specs.sort(key=lambda item: item.name)
    if len(specs) != EXPECTED_MODULE_COUNTS[scope]:
        fail("LoRA target module count differs")
    total = sum(item.lora_parameters for item in specs)
    expected_total = EXPECTED_LORA_PARAMETER_COUNTS[scope]
    if total != expected_total:
        fail(
            f"LoRA trainable parameter estimate differs: {total} != {expected_total}"
        )
    return tuple(specs)


def architecture_receipt(
    scope: str, specs: Sequence[ProjectionSpec]
) -> Mapping[str, Any]:
    selected = tuple(specs)
    if len(selected) != EXPECTED_MODULE_COUNTS.get(scope):
        fail("architecture receipt target count differs")
    lora_count = sum(item.lora_parameters for item in selected)
    if lora_count != EXPECTED_LORA_PARAMETER_COUNTS[scope]:
        fail("architecture receipt LoRA count differs")
    projection_rows = [item.receipt() for item in selected]
    value = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "blocks": BERNINI_BLOCKS,
        "target_module_count": len(selected),
        "target_modules_sha256": object_sha256(projection_rows),
        "lora_trainable_parameters": lora_count,
        "patch_role_trainable_parameters": PATCH_ROLE_PARAMETER_COUNT,
        "total_trainable_parameters": lora_count + PATCH_ROLE_PARAMETER_COUNT,
        "attention_affine_weight_shape": [HIDDEN_SIZE, HIDDEN_SIZE],
        "text_preprojection_weight_shapes": {
            "linear_1": [HIDDEN_SIZE, TEXT_WIDTH],
            "linear_2": [HIDDEN_SIZE, HIDDEN_SIZE],
        },
        "cross_attention_processor": "WanAttnProcessor2_0",
        "cross_attention_added_kv_projection": False,
        "base_projection_weights_frozen": True,
        "text_encoder_frozen": True,
        "all_local_packed_tokens_receive_lora": True,
        "target_row_gating": False,
        "targetless_sp_early_return": False,
        "sparse_block_routing": False,
    }
    return {**value, "digest": object_sha256(value)}


@contextmanager
def packed_role_layout(source_tokens: int, target_tokens: int) -> Iterator[None]:
    if (
        type(source_tokens) is not int
        or type(target_tokens) is not int
        or source_tokens <= 0
        or target_tokens <= 0
        or source_tokens != target_tokens
    ):
        fail("packed role layout requires equal positive source/target spans")
    if _LAYOUT.get() is not None:
        fail("nested packed role layout is forbidden")
    token = _LAYOUT.set((source_tokens, target_tokens))
    try:
        yield
    finally:
        _LAYOUT.reset(token)


@contextmanager
def official_frozen_native_only() -> Iterator[None]:
    """Temporarily bypass only the typed patch/role additions.

    The wrapped official patch embedding itself remains the executed module.
    A depth-valued ``ContextVar`` makes the route reentrant while keeping it
    isolated between threads and async contexts.  Callers that installed PEFT
    must independently enter ``disable_adapter()`` to bypass its LoRA layers.
    """

    depth = _OFFICIAL_FROZEN_NATIVE_ONLY_DEPTH.get()
    token = _OFFICIAL_FROZEN_NATIVE_ONLY_DEPTH.set(depth + 1)
    try:
        yield
    finally:
        _OFFICIAL_FROZEN_NATIVE_ONLY_DEPTH.reset(token)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH runtime dependency
        raise PackedPreservationV2Error("PyTorch is required") from error
    return torch


class TypedPackedPatchEmbeddingV2:  # replaced with nn.Module in __new__
    """Factory-backed type so this file remains importable without PyTorch."""

    def __new__(cls, native: Any) -> Any:
        torch = _torch()
        nn = torch.nn

        class _TypedPackedPatchEmbedding(nn.Module):
            def __init__(self, native_module: Any) -> None:
                super().__init__()
                weight = getattr(native_module, "weight", None)
                shape = tuple(int(value) for value in getattr(weight, "shape", ()))
                if shape != (
                    HIDDEN_SIZE,
                    PATCH_INPUT_CHANNELS,
                    *PATCH_KERNEL,
                ):
                    fail(f"native Bernini patch embedding shape differs: {shape}")
                self.native = native_module
                self.native.requires_grad_(False)
                self.source_delta = nn.Conv3d(
                    PATCH_INPUT_CHANNELS,
                    HIDDEN_SIZE,
                    kernel_size=PATCH_KERNEL,
                    stride=PATCH_KERNEL,
                    bias=True,
                )
                self.target_delta = nn.Conv3d(
                    PATCH_INPUT_CHANNELS,
                    HIDDEN_SIZE,
                    kernel_size=PATCH_KERNEL,
                    stride=PATCH_KERNEL,
                    bias=True,
                )
                self.role_embedding = nn.Parameter(
                    torch.zeros(2, HIDDEN_SIZE, dtype=torch.float32)
                )
                nn.init.zeros_(self.source_delta.weight)
                nn.init.zeros_(self.source_delta.bias)
                nn.init.zeros_(self.target_delta.weight)
                nn.init.zeros_(self.target_delta.bias)

            def forward(self, patches: Any) -> Any:
                if _OFFICIAL_FROZEN_NATIVE_ONLY_DEPTH.get() > 0:
                    return self.native(patches)
                layout = _LAYOUT.get()
                if layout is None:
                    fail("typed patch embedding called without an authenticated layout")
                source_tokens, target_tokens = layout
                if (
                    not isinstance(patches, torch.Tensor)
                    or patches.ndim != 5
                    or int(patches.shape[0]) != source_tokens + target_tokens
                    or tuple(int(value) for value in patches.shape[1:])
                    != (PATCH_INPUT_CHANNELS, *PATCH_KERNEL)
                ):
                    fail("packed patch tensor/layout differs")
                native = self.native(patches)
                source = self.source_delta(patches[:source_tokens])
                target = self.target_delta(patches[source_tokens:])
                delta = torch.cat((source, target), dim=0)
                roles = torch.cat(
                    (
                        self.role_embedding[0].expand(source_tokens, -1),
                        self.role_embedding[1].expand(target_tokens, -1),
                    ),
                    dim=0,
                ).to(device=native.device, dtype=native.dtype)
                roles = roles.reshape(source_tokens + target_tokens, HIDDEN_SIZE, 1, 1, 1)
                if native.shape != delta.shape or native.shape != roles.shape:
                    fail("typed patch output geometry differs")
                return native + delta.to(dtype=native.dtype) + roles

        return _TypedPackedPatchEmbedding(native)


def install_typed_patch_embedding(transformer: Any) -> Any:
    native = getattr(transformer, "patch_embedding", None)
    if native is None or native.__class__.__name__ == "_TypedPackedPatchEmbedding":
        fail("native patch embedding is missing or already wrapped")
    wrapped = TypedPackedPatchEmbeddingV2(native)
    transformer.patch_embedding = wrapped
    count = sum(
        int(parameter.numel())
        for name, parameter in wrapped.named_parameters()
        if name.startswith(("source_delta.", "target_delta.", "role_embedding"))
        and parameter.requires_grad
    )
    if count != PATCH_ROLE_PARAMETER_COUNT:
        fail(f"typed patch/role parameter count differs: {count}")
    return wrapped


def trainable_named_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    allowed_patch = (".source_delta.", ".target_delta.", ".role_embedding")
    result = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not result:
        fail("trainable parameter scope is empty")
    leaked = [
        name
        for name, _ in result
        if ".lora_A." not in name
        and ".lora_B." not in name
        and not any(marker in name for marker in allowed_patch)
    ]
    if leaked:
        fail(f"base/text parameter leaked into optimizer: {leaked[:8]}")
    return result


def verify_trainable_parameter_count(model: Any, scope: str) -> int:
    named = trainable_named_parameters(model)
    count = sum(int(parameter.numel()) for _, parameter in named)
    expected = EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNTS.get(scope)
    if count != expected:
        fail(f"actual trainable count differs: {count} != {expected}")
    return count


def validate_lora_installation(
    model: Any, specs: Sequence[ProjectionSpec]
) -> Mapping[str, Any]:
    """Prove each selected affine owns exactly one correctly shaped A/B pair."""

    named = dict(model.named_parameters())
    rows: list[Mapping[str, Any]] = []
    consumed: set[str] = set()
    for spec in specs:
        matches_a = [
            name
            for name in named
            if f"{spec.name}.lora_A." in name and name.endswith(".weight")
        ]
        matches_b = [
            name
            for name in named
            if f"{spec.name}.lora_B." in name and name.endswith(".weight")
        ]
        if len(matches_a) != 1 or len(matches_b) != 1:
            fail(f"LoRA A/B ownership differs: {spec.name}")
        name_a, name_b = matches_a[0], matches_b[0]
        parameter_a, parameter_b = named[name_a], named[name_b]
        if (
            tuple(int(value) for value in parameter_a.shape)
            != (LORA_RANK, spec.in_features)
            or tuple(int(value) for value in parameter_b.shape)
            != (spec.out_features, LORA_RANK)
            or not parameter_a.requires_grad
            or not parameter_b.requires_grad
        ):
            fail(f"LoRA A/B rank or affine dimensions differ: {spec.name}")
        consumed.update((name_a, name_b))
        rows.append(
            {
                "module": spec.name,
                "a": name_a,
                "b": name_b,
                "a_shape": list(parameter_a.shape),
                "b_shape": list(parameter_b.shape),
            }
        )
    actual_lora = {
        name
        for name, parameter in named.items()
        if parameter.requires_grad
        and (".lora_A." in name or ".lora_B." in name)
    }
    if consumed != actual_lora:
        fail("installed LoRA parameter-name set exceeds exact selected modules")
    value = {
        "selected_affines": len(specs),
        "lora_tensors": len(actual_lora),
        "rank": LORA_RANK,
        "rows_sha256": object_sha256(rows),
        "exact_one_a_and_b_per_affine": True,
    }
    return {**value, "digest": object_sha256(value)}


def trainable_inventory(model: Any) -> tuple[Mapping[str, Any], ...]:
    """Exact names/shapes/dtypes saved beside every inference adapter."""

    rows = tuple(
        {
            "name": name,
            "shape": [int(value) for value in parameter.shape],
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
        }
        for name, parameter in trainable_named_parameters(model)
    )
    if len({str(row["name"]) for row in rows}) != len(rows):
        fail("trainable inventory contains duplicate names")
    return rows


def export_trainable_state(model: Any) -> Mapping[str, Any]:
    """Return a CPU adapter state independent of optimizer internals."""

    torch = _torch()
    state = {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in trainable_named_parameters(model)
    }
    if not state or any(
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or not tensor.is_contiguous()
        or not bool(torch.isfinite(tensor.float()).all().item())
        for tensor in state.values()
    ):
        fail("exported inference adapter state differs")
    return state


def load_trainable_state_strict(model: Any, state: Mapping[str, Any]) -> None:
    """Install an inference adapter only when every exact name/shape matches."""

    torch = _torch()
    named = dict(trainable_named_parameters(model))
    if not isinstance(state, Mapping) or set(state) != set(named):
        fail("inference adapter parameter-name set differs")
    with torch.no_grad():
        for name, parameter in named.items():
            tensor = state[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != tuple(parameter.shape)
                or not bool(torch.isfinite(tensor.float()).all().item())
            ):
                fail(f"inference adapter tensor differs: {name}")
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def objective_for_logical_record(logical_record: int) -> str:
    if type(logical_record) is not int or logical_record < 0:
        fail("logical record must be a non-negative integer")
    return PRETEXT_CYCLE[logical_record % len(PRETEXT_CYCLE)]


def objective_histogram(logical_records: int) -> Mapping[str, int]:
    if type(logical_records) is not int or logical_records <= 0:
        fail("logical record count must be positive")
    result = {name: 0 for name in PRETEXTS}
    for index in range(logical_records):
        result[objective_for_logical_record(index)] += 1
    return result


def checkpoint_steps(execution_scope: str) -> tuple[int, ...]:
    if execution_scope == "optimizer-canary-2":
        return CHECKPOINT_STEPS_CANARY2
    if execution_scope == "exact80":
        return CHECKPOINT_STEPS_EXACT80
    fail(f"unsupported execution scope: {execution_scope!r}")


def optimizer_steps(execution_scope: str) -> int:
    if execution_scope == "optimizer-canary-2":
        return 2
    if execution_scope == "exact80":
        return 80
    fail(f"unsupported execution scope: {execution_scope!r}")


def _nonidentity_permutation(length: int, seed: int) -> list[int]:
    import random

    generator = random.Random(int(seed))
    order = list(range(length))
    generator.shuffle(order)
    if order == list(range(length)):
        order = order[1:] + order[:1]
    return order


def restoration_source(clean: Any, objective: str, seed: int) -> tuple[Any, Mapping[str, Any]]:
    """Deterministically corrupt only the real source; target remains ``clean``."""

    torch = _torch()
    if (
        objective not in PRETEXTS
        or not isinstance(clean, torch.Tensor)
        or clean.dtype != torch.float32
        or clean.device.type != "cpu"
        or clean.ndim != 5
        or tuple(int(value) for value in clean.shape[:3]) != (1, 16, 21)
        or clean.requires_grad
        or not clean.is_contiguous()
        or not bool(torch.isfinite(clean).all().item())
    ):
        fail("real-source restoration input differs")
    value = clean.clone()
    details: dict[str, Any] = {"objective": objective, "seed": int(seed)}
    if objective == "noop":
        pass
    elif objective == "cube":
        phases, height, width = (int(clean.shape[index]) for index in (2, 3, 4))
        temporal_span = max(1, math.ceil(phases * 0.60))
        height_span = max(1, math.ceil(height * 0.70))
        width_span = max(1, math.ceil(width * 0.70))
        temporal_start = int(seed) % (phases - temporal_span + 1)
        height_start = (int(seed) // 31) % (height - height_span + 1)
        width_start = (int(seed) // 997) % (width - width_span + 1)
        value[
            :,
            :,
            temporal_start : temporal_start + temporal_span,
            height_start : height_start + height_span,
            width_start : width_start + width_span,
        ] = 0.0
        masked = temporal_span * height_span * width_span
        details.update(
            corruption="contiguous_spatiotemporal_cuboid",
            cuboid_start=[temporal_start, height_start, width_start],
            cuboid_shape=[temporal_span, height_span, width_span],
            discrete_mask_ratio=masked / (phases * height * width),
        )
    elif objective == "speed":
        phases = int(clean.shape[2])
        indices = [min(2 * index, phases - 1) for index in range(phases)]
        value = clean.index_select(2, torch.tensor(indices, dtype=torch.long)).contiguous()
        details.update(
            corruption="two_x_temporal_resample_with_terminal_hold",
            temporal_resample_factor=2,
            phase_indices=indices,
        )
    else:
        phases, height, width = (int(value.shape[index]) for index in (2, 3, 4))
        if phases != 21 or height % 2 or width % 2:
            fail("2x2x2 tube shuffle requires exact21 and even spatial latent axes")
        # Preserve Wan's causal first phase; the remaining 20 phases form two
        # equal temporal bins, crossed with two height and two width bins.
        body = clean[:, :, 1:, :, :]
        tubes = []
        for temporal in range(2):
            for vertical in range(2):
                for horizontal in range(2):
                    tubes.append(
                        body[
                            :,
                            :,
                            temporal * 10 : (temporal + 1) * 10,
                            vertical * (height // 2) : (vertical + 1) * (height // 2),
                            horizontal * (width // 2) : (horizontal + 1) * (width // 2),
                        ]
                    )
        order = _nonidentity_permutation(8, seed)
        shuffled = torch.empty_like(body)
        output_index = 0
        for temporal in range(2):
            for vertical in range(2):
                for horizontal in range(2):
                    shuffled[
                        :,
                        :,
                        temporal * 10 : (temporal + 1) * 10,
                        vertical * (height // 2) : (vertical + 1) * (height // 2),
                        horizontal * (width // 2) : (horizontal + 1) * (width // 2),
                    ] = tubes[order[output_index]]
                    output_index += 1
        value[:, :, 1:, :, :] = shuffled
        details.update(tube_grid=[2, 2, 2], first_causal_phase_preserved=True, permutation=order)
    if value.data_ptr() == clean.data_ptr() or value.shape != clean.shape:
        fail("restoration corruption aliased or changed geometry")
    if objective != "noop" and torch.equal(value, clean):
        fail("registered restoration corruption was an identity")
    details.update(
        real_source_index0_only=True,
        synthetic_target_accessed=False,
        clean_target_is_original_source=True,
        reward_used=False,
        vlm_used=False,
    )
    return value.detach().contiguous(), details


def memory_estimate_receipt(scope: str) -> Mapping[str, Any]:
    if scope not in LORA_SCOPES:
        fail("memory estimate scope differs")
    value = {
        "scope": scope,
        "gpu": "AMD-MI210-64GiB",
        "world": 8,
        "topology": "DP2xSP4",
        "preregistered_memory_range_gib": None,
        "memory_pass_threshold_gib": None,
        "measurement_required": True,
        "authority": (
            "optimizer canary per-rank max_memory_allocated and "
            "max_memory_reserved receipt"
        ),
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "BERNINI_BLOCKS",
    "CHECKPOINT_STEPS_CANARY2",
    "CHECKPOINT_STEPS_EXACT80",
    "EXECUTION_SCOPES",
    "EXPECTED_LORA_PARAMETER_COUNTS",
    "EXPECTED_MODULE_COUNTS",
    "EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNTS",
    "LORA_ALPHA",
    "LORA_RANK",
    "LORA_SCOPES",
    "PATCH_ROLE_PARAMETER_COUNT",
    "PRETEXTS",
    "PRETEXT_INSTRUCTIONS",
    "PackedPreservationV2Error",
    "ProjectionSpec",
    "architecture_receipt",
    "checkpoint_steps",
    "install_typed_patch_embedding",
    "export_trainable_state",
    "load_trainable_state_strict",
    "memory_estimate_receipt",
    "object_sha256",
    "objective_for_logical_record",
    "objective_histogram",
    "optimizer_steps",
    "official_frozen_native_only",
    "packed_role_layout",
    "restoration_source",
    "select_projection_specs",
    "trainable_named_parameters",
    "trainable_inventory",
    "validate_lora_installation",
    "verify_trainable_parameter_count",
]
