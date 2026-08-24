#!/usr/bin/env python3
"""Read-only language-indexed relational action slots for Bernini-R 1.3B.

The existing Q-MOSAIC owner signature intentionally removes spatial order.
That is a useful guard against transporting an owner's layout, but it cannot
represent an object-grounded event such as::

    dog approaches bone -> mouth contacts bone -> grip -> lift -> hold

This module defines a stricter *diagnostic* coordinate for that case.  It
reconstructs soft visual localization weights only from Bernini's own frozen
cross-attention Q/K tensors and sealed text-token roles.  It consumes no
external mask, track, pose, flow, detector, segmentation, or trajectory.

Four same-noisy-state branches are required: ``action``, ``noop``,
``reverse`` and ``incomplete``.  Actor/object/anatomical-anchor/action soft
slots retain spatial order, and their five-stage relational path is contrasted
against the three controls.  A separate set of detached rows describes
source-native actor/object/background/camera preservation equalities.

The implementation is deliberately read-only:

* hooks never replace module inputs or outputs;
* all published tensors are detached owned CPU FP32 copies;
* no backward, optimizer, LoRA, checkpoint write, or parameter update exists;
* the receipt always says ``scientific_authority=false`` and
  ``real_auh_runtime_validated=false``;
* an internal diagnostic gate is not decoded action evidence and cannot
  authorize training.

The real hook surface is block-15 ``attn2.to_q``/``attn2.to_k`` after their
linear projections.  Q/K normalization, when present on the attention module,
is applied in the same head geometry before capture.  Target suffix rows are
restored from the official contiguous SP layout before slot construction.
The complete transformer/runtime seal remains the responsibility of the
authenticated Q-MOSAIC runner; this observer also seals the local attention
projection state around every capture.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
import math
import re
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn


METHOD = "bernini-language-indexed-relational-action-slots-v1"
SCHEMA_VERSION = "bernini-language-indexed-relational-action-slots-v1"
CAPTURE_SCHEMA_VERSION = "bernini-relational-qk-sp4-capture-v1"
RECEIPT_SCHEMA_VERSION = "bernini-relational-action-slot-probe-receipt-v1"
STATUS = "READ_ONLY_INTERNAL_RELATIONAL_DIAGNOSTIC_ZERO_UPDATES"

TOTAL_BLOCKS_1P3B = 30
HOOK_BLOCK_INDEX = 15
HIDDEN_SIZE_1P3B = 1536
SP_SIZE = 4
LATENT_PHASES = 21
FRAME_COUNT = 81
NATIVE_SCHEDULE_INDEX = 33
NATIVE_TIMESTEP = 516
NATIVE_SIGMA = 0.5161304473876953

BRANCH_ORDER = ("action", "noop", "reverse", "incomplete")
ROLE_ORDER = ("actor", "object", "anatomical_anchor", "action")
EVENT_STAGE_ORDER = ("approach", "contact", "grip", "lift", "hold")

RELATION_FEATURE_NAMES = (
    "actor_to_object_dx",
    "actor_to_object_dy",
    "actor_object_distance",
    "actor_object_bhattacharyya_overlap",
    "anchor_to_object_dx",
    "anchor_to_object_dy",
    "anchor_object_distance",
    "anchor_object_bhattacharyya_overlap",
    "action_actor_bhattacharyya_overlap",
    "action_object_bhattacharyya_overlap",
    "object_displacement_x_from_phase0",
    "object_displacement_y_from_phase0",
    "common_role_centroid_x",
    "common_role_centroid_y",
    "actor_object_relative_scale",
)

PRESERVATION_ROW_NAMES = (
    "actor_temporal_dc_appearance",
    "object_temporal_dc_appearance",
    "anatomical_anchor_temporal_dc_appearance",
    "background_temporal_dc_appearance",
    "camera_common_translation",
    "camera_relative_scale",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_EPS = 1.0e-12
_LAYOUT_TOKEN = object()
_LOCAL_CAPTURE_TOKEN = object()
_GLOBAL_CAPTURE_TOKEN = object()
_AUDIT_TOKEN = object()


class RelationalActionSlotError(RuntimeError):
    """A closed relational-slot observation contract was violated."""


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
        raise RelationalActionSlotError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RelationalActionSlotError(f"{label} must be lowercase SHA-256")
    return value


def _require_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise RelationalActionSlotError(f"{label} must be a closed identifier")
    return value


def _tensor_bytes(value: torch.Tensor) -> bytes:
    owned = value.detach().to(device="cpu").contiguous().clone()
    payload = io.BytesIO()
    modern = getattr(owned, "untyped_storage", None)
    if callable(modern):
        untyped = modern()
    else:  # pinned Torch 1.12 fallback
        storage = owned.storage()
        legacy = getattr(storage, "_untyped", None)
        if not callable(legacy):
            raise RelationalActionSlotError("untyped tensor storage is unavailable")
        untyped = legacy()
    untyped._write_file(payload, False, False, 1)
    raw = payload.getvalue()
    expected = int(owned.numel()) * int(owned.element_size())
    if len(raw) != expected:
        raise RelationalActionSlotError("tensor storage byte count differs")
    return raw


def tensor_sha256(value: torch.Tensor, *, label: str) -> str:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.numel() == 0
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise RelationalActionSlotError(f"{label} must be a finite tensor")
    owned = value.detach().to(device="cpu").contiguous().clone()
    header = canonical_json_bytes(
        {
            "dtype": str(owned.dtype),
            "shape": list(map(int, owned.shape)),
            "numel": int(owned.numel()),
        }
    )
    return hashlib.sha256(header + b"\x00" + _tensor_bytes(owned)).hexdigest()


def _finite_float_tensor(value: Any, *, label: str, ndim: int) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.ndim != ndim
        or value.numel() == 0
        or value.device.type == "meta"
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise RelationalActionSlotError(
            f"{label} must be a non-empty finite rank-{ndim} floating tensor"
        )
    return value


def _owned_fp32(value: torch.Tensor, *, label: str) -> torch.Tensor:
    _finite_float_tensor(value, label=label, ndim=value.ndim)
    result = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    if result.requires_grad or result.grad_fn is not None:
        raise RelationalActionSlotError(f"{label} retained an autograd graph")
    return result


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().double().reshape(-1)
    b = right.detach().double().reshape(-1)
    denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    if denominator <= _EPS:
        return -1.0
    return float(torch.dot(a, b).item() / denominator)


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().double()).item())


@dataclass(frozen=True)
class RelationalHookPoint:
    module_path: str
    hook_kind: str
    semantic_role: str


def bernini_relational_hook_plan() -> tuple[RelationalHookPoint, ...]:
    prefix = f"diff_dec.transformer.blocks.{HOOK_BLOCK_INDEX}.attn2"
    return (
        RelationalHookPoint(
            f"{prefix}.to_q",
            "post",
            "projected_visual_queries_before_optional_qk_normalization",
        ),
        RelationalHookPoint(
            f"{prefix}.to_k",
            "post",
            "projected_text_keys_before_optional_qk_normalization",
        ),
    )


@dataclass(frozen=True)
class LanguageRoleTokenBinding:
    """Prompt/token authority for the four internal language roles.

    Token indices refer to the actual text-key sequence observed at
    ``attn2.to_k``.  They are semantic text indices, never visual selections.
    """

    branch: str
    prompt_text: str
    token_ids: tuple[int, ...]
    valid_token_count: int
    actor_token_indices: tuple[int, ...]
    object_token_indices: tuple[int, ...]
    anatomical_anchor_token_indices: tuple[int, ...]
    action_token_indices: tuple[int, ...]
    tokenizer_receipt_digest: str
    text_encoder_receipt_digest: str

    def __post_init__(self) -> None:
        if self.branch not in BRANCH_ORDER:
            raise RelationalActionSlotError("language binding branch differs")
        if (
            not isinstance(self.prompt_text, str)
            or not self.prompt_text.strip()
            or self.prompt_text != self.prompt_text.strip()
            or "\x00" in self.prompt_text
        ):
            raise RelationalActionSlotError("prompt text must be canonical non-empty text")
        if (
            type(self.valid_token_count) is not int
            or self.valid_token_count <= 0
            or self.valid_token_count != len(self.token_ids)
            or any(type(value) is not int or value < 0 for value in self.token_ids)
        ):
            raise RelationalActionSlotError("token ids/valid length differ")
        role_rows = self.role_indices()
        observed: set[int] = set()
        for role in ROLE_ORDER:
            indices = role_rows[role]
            if (
                not isinstance(indices, tuple)
                or not indices
                or tuple(sorted(set(indices))) != indices
                or any(
                    type(index) is not int
                    or not 0 <= index < self.valid_token_count
                    for index in indices
                )
            ):
                raise RelationalActionSlotError(
                    f"{role} token indices must be sorted, unique, and in range"
                )
            overlap = observed.intersection(indices)
            if overlap:
                raise RelationalActionSlotError(
                    "language role token sets must be disjoint"
                )
            observed.update(indices)
        _require_sha256(
            self.tokenizer_receipt_digest, label="tokenizer receipt digest"
        )
        _require_sha256(
            self.text_encoder_receipt_digest, label="text encoder receipt digest"
        )

    def role_indices(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType(
            {
                "actor": self.actor_token_indices,
                "object": self.object_token_indices,
                "anatomical_anchor": self.anatomical_anchor_token_indices,
                "action": self.action_token_indices,
            }
        )

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "branch": self.branch,
            "prompt_text": self.prompt_text,
            "prompt_sha256": hashlib.sha256(
                self.prompt_text.encode("utf-8")
            ).hexdigest(),
            "token_ids": list(self.token_ids),
            "token_ids_sha256": object_sha256(list(self.token_ids)),
            "valid_token_count": self.valid_token_count,
            "role_token_indices": {
                key: list(value) for key, value in self.role_indices().items()
            },
            "tokenizer_receipt_digest": self.tokenizer_receipt_digest,
            "text_encoder_receipt_digest": self.text_encoder_receipt_digest,
            "visual_localization_annotation_used": False,
        }
        return MappingProxyType({**value, "digest": object_sha256(value)})


@dataclass(frozen=True)
class EventPhaseRanges:
    approach: tuple[int, int]
    contact: tuple[int, int]
    grip: tuple[int, int]
    lift: tuple[int, int]
    hold: tuple[int, int]

    def __post_init__(self) -> None:
        previous = 0
        for stage in EVENT_STAGE_ORDER:
            row = getattr(self, stage)
            if (
                not isinstance(row, tuple)
                or len(row) != 2
                or any(type(value) is not int for value in row)
                or row[0] != previous
                or row[1] <= row[0]
            ):
                raise RelationalActionSlotError(
                    "event ranges must be a contiguous non-empty five-stage partition"
                )
            previous = row[1]
        if previous != LATENT_PHASES:
            raise RelationalActionSlotError("event ranges must cover exact latent21")

    def receipt(self) -> Mapping[str, list[int]]:
        return MappingProxyType(
            {stage: list(getattr(self, stage)) for stage in EVENT_STAGE_ORDER}
        )

    def reduce(self, path: torch.Tensor) -> torch.Tensor:
        _finite_float_tensor(path, label="event path", ndim=2)
        if int(path.shape[0]) != LATENT_PHASES:
            raise RelationalActionSlotError("event path must contain latent21")
        return torch.stack(
            [
                path[slice(*getattr(self, stage))].mean(dim=0)
                for stage in EVENT_STAGE_ORDER
            ],
            dim=0,
        ).contiguous()


CDF_DOG_PREREGISTERED_EVENT_RANGES = EventPhaseRanges(
    approach=(0, 5),
    contact=(5, 8),
    grip=(8, 11),
    lift=(11, 16),
    hold=(16, 21),
)


@dataclass(frozen=True)
class TargetSuffixSPLayout:
    latent_phases: int
    patch_height: int
    patch_width: int
    patch_positions: int
    condition_tokens: int
    target_tokens: int
    total_tokens: int
    sp_rank: int
    sp_size: int
    local_length: int
    local_target_indices: torch.Tensor = field(repr=False)
    target_flat_indices: torch.Tensor = field(repr=False)
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def receipt(self) -> Mapping[str, Any]:
        if self._token is not _LAYOUT_TOKEN:
            raise RelationalActionSlotError("unsealed target-suffix layout")
        value = {
            "latent_phases": self.latent_phases,
            "patch_grid_height_width": [self.patch_height, self.patch_width],
            "patch_positions": self.patch_positions,
            "condition_tokens": self.condition_tokens,
            "target_tokens": self.target_tokens,
            "total_tokens": self.total_tokens,
            "sp_rank": self.sp_rank,
            "sp_size": self.sp_size,
            "local_length": self.local_length,
            "selected_target_rows": int(self.local_target_indices.numel()),
            "local_target_indices_sha256": tensor_sha256(
                self.local_target_indices, label="local target indices"
            ),
            "target_flat_indices_sha256": tensor_sha256(
                self.target_flat_indices, label="target flat indices"
            ),
            "global_order_formula": (
                "g=sp_rank*ceil(total_tokens/sp_size)+local_index;"
                "target iff condition_tokens<=g<total_tokens;"
                "target_flat=g-condition_tokens"
            ),
        }
        return MappingProxyType({**value, "digest": object_sha256(value)})


def build_target_suffix_sp_layout(
    *,
    patch_height: int,
    patch_width: int,
    condition_tokens: int,
    total_tokens: int,
    sp_rank: int,
    sp_size: int = SP_SIZE,
) -> TargetSuffixSPLayout:
    integers = (
        patch_height,
        patch_width,
        condition_tokens,
        total_tokens,
        sp_rank,
        sp_size,
    )
    if any(type(value) is not int for value in integers):
        raise RelationalActionSlotError("target-suffix layout values must be integers")
    if (
        patch_height <= 0
        or patch_width <= 0
        or condition_tokens < 0
        or total_tokens <= condition_tokens
        or sp_size <= 0
        or not 0 <= sp_rank < sp_size
    ):
        raise RelationalActionSlotError("target-suffix layout bounds differ")
    patch_positions = patch_height * patch_width
    target_tokens = LATENT_PHASES * patch_positions
    if total_tokens - condition_tokens != target_tokens:
        raise RelationalActionSlotError(
            "target suffix must be exact latent21 times the spatial patch grid"
        )
    local_length = (total_tokens + sp_size - 1) // sp_size
    local_index = torch.arange(local_length, dtype=torch.int64)
    global_index = sp_rank * local_length + local_index
    selected = (global_index >= condition_tokens) & (global_index < total_tokens)
    local_target = local_index[selected].contiguous()
    target_flat = (global_index[selected] - condition_tokens).contiguous()
    if (
        target_flat.numel()
        and (
            int(target_flat.min().item()) < 0
            or int(target_flat.max().item()) >= target_tokens
        )
    ):
        raise RelationalActionSlotError("target suffix index restoration differs")
    result = TargetSuffixSPLayout(
        latent_phases=LATENT_PHASES,
        patch_height=patch_height,
        patch_width=patch_width,
        patch_positions=patch_positions,
        condition_tokens=condition_tokens,
        target_tokens=target_tokens,
        total_tokens=total_tokens,
        sp_rank=sp_rank,
        sp_size=sp_size,
        local_length=local_length,
        local_target_indices=local_target,
        target_flat_indices=target_flat,
    )
    object.__setattr__(result, "_token", _LAYOUT_TOKEN)
    return result


@dataclass(frozen=True)
class LocalRelationalQKCapture:
    branch: str
    layout: TargetSuffixSPLayout
    binding: LanguageRoleTokenBinding
    target_queries: torch.Tensor = field(repr=False)
    text_keys: torch.Tensor = field(repr=False)
    head_count: int
    head_dim: int
    shared_noisy_state_sha256: str
    attention_projection_state_sha256: str
    q_normalization: str
    k_normalization: str
    runtime_origin: str
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def receipt(self) -> Mapping[str, Any]:
        if self._token is not _LOCAL_CAPTURE_TOKEN:
            raise RelationalActionSlotError("unsealed local Q/K capture")
        value = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "scope": "rank_local",
            "branch": self.branch,
            "layout_digest": self.layout.receipt()["digest"],
            "binding_digest": self.binding.receipt()["digest"],
            "target_query_shape": list(map(int, self.target_queries.shape)),
            "target_query_sha256": tensor_sha256(
                self.target_queries, label="local target queries"
            ),
            "text_key_shape": list(map(int, self.text_keys.shape)),
            "text_key_sha256": tensor_sha256(self.text_keys, label="local text keys"),
            "head_count": self.head_count,
            "head_dim": self.head_dim,
            "shared_noisy_state_sha256": self.shared_noisy_state_sha256,
            "attention_projection_state_sha256": (
                self.attention_projection_state_sha256
            ),
            "q_normalization": self.q_normalization,
            "k_normalization": self.k_normalization,
            "runtime_origin": self.runtime_origin,
            "detached_owned_cpu_fp32": True,
            "hook_replaced_input_or_output": False,
        }
        return MappingProxyType({**value, "digest": object_sha256(value)})


@dataclass(frozen=True)
class GlobalRelationalQKCapture:
    branch: str
    binding: LanguageRoleTokenBinding
    target_queries: torch.Tensor = field(repr=False)
    text_keys: torch.Tensor = field(repr=False)
    patch_height: int
    patch_width: int
    head_count: int
    head_dim: int
    shared_noisy_state_sha256: str
    runtime_origin: str
    rank_capture_digests: tuple[str, ...]
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    def receipt(self) -> Mapping[str, Any]:
        if self._token is not _GLOBAL_CAPTURE_TOKEN:
            raise RelationalActionSlotError("unsealed global Q/K capture")
        value = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "scope": "global_target_suffix",
            "branch": self.branch,
            "binding": dict(self.binding.receipt()),
            "target_query_shape": list(map(int, self.target_queries.shape)),
            "target_query_sha256": tensor_sha256(
                self.target_queries, label="global target queries"
            ),
            "text_key_shape": list(map(int, self.text_keys.shape)),
            "text_key_sha256": tensor_sha256(self.text_keys, label="global text keys"),
            "patch_grid_height_width": [self.patch_height, self.patch_width],
            "head_count": self.head_count,
            "head_dim": self.head_dim,
            "shared_noisy_state_sha256": self.shared_noisy_state_sha256,
            "runtime_origin": self.runtime_origin,
            "rank_capture_digests": list(self.rank_capture_digests),
            "external_visual_localizer_used": False,
            "detached_owned_cpu_fp32": True,
        }
        return MappingProxyType({**value, "digest": object_sha256(value)})


def _module_projection_state_sha256(attention: nn.Module) -> str:
    rows: list[Mapping[str, Any]] = []
    for prefix in ("to_q", "to_k", "norm_q", "norm_k"):
        module = getattr(attention, prefix, None)
        if module is None:
            rows.append({"module": prefix, "present": False})
            continue
        if not isinstance(module, nn.Module):
            raise RelationalActionSlotError(f"attn2.{prefix} is not a module")
        parameters = []
        for name, parameter in module.named_parameters(recurse=True):
            parameters.append(
                {
                    "name": name,
                    "requires_grad": bool(parameter.requires_grad),
                    "shape": list(map(int, parameter.shape)),
                    "dtype": str(parameter.dtype),
                    "sha256": tensor_sha256(
                        parameter, label=f"attn2.{prefix}.{name}"
                    ),
                }
            )
        buffers = []
        for name, buffer in module.named_buffers(recurse=True):
            buffers.append(
                {
                    "name": name,
                    "shape": list(map(int, buffer.shape)),
                    "dtype": str(buffer.dtype),
                    "sha256": tensor_sha256(buffer, label=f"attn2.{prefix}.{name}"),
                }
            )
        rows.append(
            {
                "module": prefix,
                "present": True,
                "type": f"{type(module).__module__}.{type(module).__qualname__}",
                "parameters": parameters,
                "buffers": buffers,
            }
        )
    return object_sha256(rows)


class BerniniRelationalCrossAttentionObserver:
    """One-forward, read-only Q/K observer for a Bernini cross-attention block.

    ``tiny_fixture=True`` exists only for unit tests.  Such captures retain
    ``runtime_origin=tiny_torch_fixture`` and can never gain scientific or AUH
    authority through this module.
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        block_index: int = HOOK_BLOCK_INDEX,
        expected_hidden_size: int = HIDDEN_SIZE_1P3B,
        tiny_fixture: bool = False,
    ) -> None:
        if not isinstance(transformer, nn.Module) or type(tiny_fixture) is not bool:
            raise RelationalActionSlotError("observer transformer/mode differs")
        blocks = tuple(getattr(transformer, "blocks", ()))
        if (
            type(block_index) is not int
            or type(expected_hidden_size) is not int
            or expected_hidden_size <= 0
            or not 0 <= block_index < len(blocks)
            or (not tiny_fixture and len(blocks) != TOTAL_BLOCKS_1P3B)
            or (not tiny_fixture and block_index != HOOK_BLOCK_INDEX)
            or (not tiny_fixture and expected_hidden_size != HIDDEN_SIZE_1P3B)
        ):
            raise RelationalActionSlotError("Bernini block geometry differs")
        attention = getattr(blocks[block_index], "attn2", None)
        q_module = getattr(attention, "to_q", None)
        k_module = getattr(attention, "to_k", None)
        heads = getattr(attention, "heads", None)
        if (
            not isinstance(attention, nn.Module)
            or not isinstance(q_module, nn.Module)
            or not isinstance(k_module, nn.Module)
            or type(heads) is not int
            or heads <= 0
            or expected_hidden_size % heads
        ):
            raise RelationalActionSlotError("Bernini attn2 Q/K/head structure differs")
        self.transformer = transformer
        self.attention = attention
        self.q_module = q_module
        self.k_module = k_module
        self.block_index = block_index
        self.expected_hidden_size = expected_hidden_size
        self.head_count = heads
        self.head_dim = expected_hidden_size // heads
        self.tiny_fixture = tiny_fixture
        self._q_handle: Any = None
        self._k_handle: Any = None
        self._pending: Optional[
            tuple[TargetSuffixSPLayout, LanguageRoleTokenBinding, str, str]
        ] = None
        self._q_capture: Optional[torch.Tensor] = None
        self._k_capture: Optional[torch.Tensor] = None
        self._q_calls = 0
        self._k_calls = 0
        self._state_before: Optional[str] = None

    def install(self) -> None:
        if self._q_handle is not None or self._k_handle is not None:
            raise RelationalActionSlotError("relational observer is already installed")
        self._q_handle = self.q_module.register_forward_hook(self._q_hook)
        try:
            self._k_handle = self.k_module.register_forward_hook(self._k_hook)
        except Exception:
            self._q_handle.remove()
            self._q_handle = None
            raise

    def remove(self) -> None:
        if self._pending is not None:
            raise RelationalActionSlotError("cannot remove an active relational capture")
        if self._q_handle is not None:
            self._q_handle.remove()
            self._q_handle = None
        if self._k_handle is not None:
            self._k_handle.remove()
            self._k_handle = None

    def _heads_and_normalize(
        self, value: Any, *, normalizer_name: str, label: str
    ) -> tuple[torch.Tensor, str]:
        tensor = _finite_float_tensor(value, label=label, ndim=3)
        if (
            int(tensor.shape[0]) != 1
            or int(tensor.shape[2]) != self.expected_hidden_size
        ):
            raise RelationalActionSlotError(f"{label} projected shape differs")
        # Bernini's pinned WanAttnProcessor2_0 applies ``norm_q/norm_k`` to
        # [B,S,inner_dim] and only then unflattens heads.  Preserve that exact
        # order and the live projection dtype.  Converting BF16 output to FP32
        # before a BF16 model-owned normalizer would be a different coordinate.
        projected = tensor.detach()
        normalizer = getattr(self.attention, normalizer_name, None)
        if normalizer is None:
            normalized = projected
            identity = "absent_identity"
        else:
            if not isinstance(normalizer, nn.Module):
                raise RelationalActionSlotError(
                    f"attn2.{normalizer_name} must be a module or None"
                )
            with torch.no_grad():
                normalized = normalizer(projected)
            identity = f"{type(normalizer).__module__}.{type(normalizer).__qualname__}"
        _finite_float_tensor(normalized, label=f"normalized {label}", ndim=3)
        if tuple(normalized.shape) != tuple(projected.shape):
            raise RelationalActionSlotError(f"normalized {label} shape differs")
        headed = normalized.reshape(
            1, int(normalized.shape[1]), self.head_count, self.head_dim
        )
        return _owned_fp32(headed[0], label=f"normalized {label}"), identity

    def _q_hook(self, module: Any, inputs: Any, output: Any) -> None:
        del module, inputs
        if self._pending is None or self._q_capture is not None:
            raise RelationalActionSlotError("unexpected or repeated Q hook call")
        layout, _, _, _ = self._pending
        headed, _ = self._heads_and_normalize(
            output, normalizer_name="norm_q", label="projected visual query"
        )
        if int(headed.shape[0]) != layout.local_length:
            raise RelationalActionSlotError("local Q length differs from SP layout")
        indices = layout.local_target_indices.to(dtype=torch.int64)
        self._q_capture = headed.index_select(0, indices).contiguous()
        self._q_calls += 1

    def _k_hook(self, module: Any, inputs: Any, output: Any) -> None:
        del module, inputs
        if self._pending is None or self._k_capture is not None:
            raise RelationalActionSlotError("unexpected or repeated K hook call")
        _, binding, _, _ = self._pending
        headed, _ = self._heads_and_normalize(
            output, normalizer_name="norm_k", label="projected text key"
        )
        if int(headed.shape[0]) < binding.valid_token_count:
            raise RelationalActionSlotError(
                "text key sequence is shorter than the sealed token binding"
            )
        self._k_capture = headed.contiguous()
        self._k_calls += 1

    @contextmanager
    def capture(
        self,
        *,
        layout: TargetSuffixSPLayout,
        binding: LanguageRoleTokenBinding,
        shared_noisy_state_sha256: str,
    ) -> Iterator[list[LocalRelationalQKCapture]]:
        if self._q_handle is None or self._k_handle is None:
            raise RelationalActionSlotError("install the relational observer first")
        if self._pending is not None or layout._token is not _LAYOUT_TOKEN:
            raise RelationalActionSlotError("nested or unsealed relational capture")
        _require_sha256(shared_noisy_state_sha256, label="shared noisy state SHA256")
        runtime_origin = (
            "tiny_torch_fixture"
            if self.tiny_fixture
            else "bernini_block15_attn2_qk_post_hooks"
        )
        self._pending = (layout, binding, shared_noisy_state_sha256, runtime_origin)
        self._q_capture = None
        self._k_capture = None
        q_before, k_before = self._q_calls, self._k_calls
        self._state_before = _module_projection_state_sha256(self.attention)
        holder: list[LocalRelationalQKCapture] = []
        failure: Optional[BaseException] = None
        try:
            yield holder
        except BaseException as error:
            failure = error
        finally:
            try:
                state_after = _module_projection_state_sha256(self.attention)
                if state_after != self._state_before:
                    raise RelationalActionSlotError(
                        "attention projection state changed during read-only capture"
                    )
                if failure is not None:
                    raise failure
                if (
                    self._q_calls != q_before + 1
                    or self._k_calls != k_before + 1
                    or self._q_capture is None
                    or self._k_capture is None
                ):
                    raise RelationalActionSlotError(
                        "Q and K hooks must each fire exactly once"
                    )
                normalizer_q = getattr(self.attention, "norm_q", None)
                normalizer_k = getattr(self.attention, "norm_k", None)
                capture = LocalRelationalQKCapture(
                    branch=binding.branch,
                    layout=layout,
                    binding=binding,
                    target_queries=self._q_capture,
                    text_keys=self._k_capture,
                    head_count=self.head_count,
                    head_dim=self.head_dim,
                    shared_noisy_state_sha256=shared_noisy_state_sha256,
                    attention_projection_state_sha256=state_after,
                    q_normalization=(
                        "absent_identity"
                        if normalizer_q is None
                        else f"{type(normalizer_q).__module__}.{type(normalizer_q).__qualname__}"
                    ),
                    k_normalization=(
                        "absent_identity"
                        if normalizer_k is None
                        else f"{type(normalizer_k).__module__}.{type(normalizer_k).__qualname__}"
                    ),
                    runtime_origin=runtime_origin,
                )
                object.__setattr__(capture, "_token", _LOCAL_CAPTURE_TOKEN)
                holder.append(capture)
            finally:
                self._pending = None
                self._q_capture = None
                self._k_capture = None
                self._state_before = None


def assemble_sp4_relational_capture(
    captures: Sequence[LocalRelationalQKCapture],
) -> GlobalRelationalQKCapture:
    """Restore one global latent21 target Q tensor from exact SP4 rows."""

    if len(captures) != SP_SIZE or any(
        type(row) is not LocalRelationalQKCapture
        or row._token is not _LOCAL_CAPTURE_TOKEN
        for row in captures
    ):
        raise RelationalActionSlotError("global relational capture requires four sealed rows")
    ordered = sorted(captures, key=lambda row: row.layout.sp_rank)
    first = ordered[0]
    if [row.layout.sp_rank for row in ordered] != list(range(SP_SIZE)):
        raise RelationalActionSlotError("SP4 relational rows have wrong rank order")
    binding_digest = first.binding.receipt()["digest"]
    key_digest = tensor_sha256(first.text_keys, label="SP4 reference text keys")
    invariant = (
        first.branch,
        first.layout.patch_height,
        first.layout.patch_width,
        first.layout.patch_positions,
        first.layout.target_tokens,
        first.head_count,
        first.head_dim,
        first.shared_noisy_state_sha256,
        first.attention_projection_state_sha256,
        first.q_normalization,
        first.k_normalization,
        first.runtime_origin,
    )
    for row in ordered:
        observed = (
            row.branch,
            row.layout.patch_height,
            row.layout.patch_width,
            row.layout.patch_positions,
            row.layout.target_tokens,
            row.head_count,
            row.head_dim,
            row.shared_noisy_state_sha256,
            row.attention_projection_state_sha256,
            row.q_normalization,
            row.k_normalization,
            row.runtime_origin,
        )
        if (
            row.layout.sp_size != SP_SIZE
            or observed != invariant
            or row.binding.receipt()["digest"] != binding_digest
            or tensor_sha256(row.text_keys, label="SP4 text keys") != key_digest
            or not torch.equal(row.text_keys, first.text_keys)
        ):
            raise RelationalActionSlotError("SP4 relational invariant differs")
    target_tokens = first.layout.target_tokens
    query = torch.zeros(
        target_tokens,
        first.head_count,
        first.head_dim,
        dtype=torch.float32,
    )
    coverage = torch.zeros(target_tokens, dtype=torch.int64)
    for row in ordered:
        flat = row.layout.target_flat_indices.to(dtype=torch.int64)
        if int(flat.numel()) != int(row.target_queries.shape[0]):
            raise RelationalActionSlotError("SP4 Q rows/index count differs")
        query.index_copy_(0, flat, row.target_queries)
        coverage.index_add_(0, flat, torch.ones_like(flat))
    if not torch.equal(coverage, torch.ones_like(coverage)):
        raise RelationalActionSlotError("SP4 rows do not cover each target token once")
    query = query.reshape(
        LATENT_PHASES,
        first.layout.patch_positions,
        first.head_count,
        first.head_dim,
    ).contiguous()
    runtime_origin = (
        "assembled_sp4_tiny_torch_fixture"
        if first.runtime_origin == "tiny_torch_fixture"
        else "assembled_exact_sp4_bernini_qk_hooks"
    )
    result = GlobalRelationalQKCapture(
        branch=first.branch,
        binding=first.binding,
        target_queries=query,
        text_keys=first.text_keys.clone().contiguous(),
        patch_height=first.layout.patch_height,
        patch_width=first.layout.patch_width,
        head_count=first.head_count,
        head_dim=first.head_dim,
        shared_noisy_state_sha256=first.shared_noisy_state_sha256,
        runtime_origin=runtime_origin,
        rank_capture_digests=tuple(row.receipt()["digest"] for row in ordered),
    )
    object.__setattr__(result, "_token", _GLOBAL_CAPTURE_TOKEN)
    return result


def _global_capture_unsafe_for_test(
    *,
    branch: str,
    binding: LanguageRoleTokenBinding,
    target_queries: torch.Tensor,
    text_keys: torch.Tensor,
    patch_height: int,
    patch_width: int,
    shared_noisy_state_sha256: str,
) -> GlobalRelationalQKCapture:
    """Tiny algebra fixture; never creates Bernini/AUH/scientific authority."""

    q = _owned_fp32(target_queries, label="fixture target queries")
    k = _owned_fp32(text_keys, label="fixture text keys")
    if (
        branch != binding.branch
        or q.ndim != 4
        or k.ndim != 3
        or tuple(map(int, q.shape[:2]))
        != (LATENT_PHASES, patch_height * patch_width)
        or int(q.shape[2]) != int(k.shape[1])
        or int(q.shape[3]) != int(k.shape[2])
        or int(k.shape[0]) < binding.valid_token_count
        or patch_height <= 0
        or patch_width <= 0
    ):
        raise RelationalActionSlotError("tiny global Q/K fixture geometry differs")
    _require_sha256(shared_noisy_state_sha256, label="fixture noisy-state SHA256")
    result = GlobalRelationalQKCapture(
        branch=branch,
        binding=binding,
        target_queries=q,
        text_keys=k,
        patch_height=patch_height,
        patch_width=patch_width,
        head_count=int(q.shape[2]),
        head_dim=int(q.shape[3]),
        shared_noisy_state_sha256=shared_noisy_state_sha256,
        runtime_origin="tiny_torch_fixture",
        rank_capture_digests=(),
    )
    object.__setattr__(result, "_token", _GLOBAL_CAPTURE_TOKEN)
    return result


@dataclass(frozen=True)
class LanguageIndexedSoftSlots:
    branch: str
    role_weights: Mapping[str, torch.Tensor] = field(repr=False)
    role_features: Mapping[str, torch.Tensor] = field(repr=False)
    role_centroids: Mapping[str, torch.Tensor] = field(repr=False)
    role_entropy: Mapping[str, torch.Tensor] = field(repr=False)
    background_weights: torch.Tensor = field(repr=False)
    background_features: torch.Tensor = field(repr=False)
    relation_path: torch.Tensor = field(repr=False)
    capture_digest: str
    digest: str


def _patch_centers(height: int, width: int) -> torch.Tensor:
    y = (torch.arange(height, dtype=torch.float32) + 0.5) / float(height)
    x = (torch.arange(width, dtype=torch.float32) + 0.5) / float(width)
    y = y * 2.0 - 1.0
    x = x * 2.0 - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1).contiguous()


def _bhattacharyya(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((left * right).clamp_min(0.0)).sum(dim=1)


def compute_language_indexed_soft_slots(
    capture: GlobalRelationalQKCapture,
) -> LanguageIndexedSoftSlots:
    """Build actor/object/anchor/action slots from frozen internal Q/K only."""

    if (
        type(capture) is not GlobalRelationalQKCapture
        or capture._token is not _GLOBAL_CAPTURE_TOKEN
    ):
        raise RelationalActionSlotError("soft slots require a sealed global capture")
    q = capture.target_queries.float()
    k = capture.text_keys.float()
    if (
        tuple(map(int, q.shape[:2]))
        != (LATENT_PHASES, capture.patch_height * capture.patch_width)
        or tuple(map(int, q.shape[2:])) != (capture.head_count, capture.head_dim)
        or tuple(map(int, k.shape[1:])) != (capture.head_count, capture.head_dim)
    ):
        raise RelationalActionSlotError("global Q/K changed after sealing")
    # This is the exact QK dot-product geometry after the observer has applied
    # optional q/k normalization.  It is reconstructed outside flash attention
    # so role-token subsets can be inspected without patching the processor.
    logits = torch.einsum("tphd,lhd->tphl", q, k) / math.sqrt(
        float(capture.head_dim)
    )
    weights: dict[str, torch.Tensor] = {}
    features: dict[str, torch.Tensor] = {}
    centroids: dict[str, torch.Tensor] = {}
    entropies: dict[str, torch.Tensor] = {}
    visual = q.reshape(
        LATENT_PHASES,
        capture.patch_height * capture.patch_width,
        capture.head_count * capture.head_dim,
    )
    centers = _patch_centers(capture.patch_height, capture.patch_width)
    for role, indices in capture.binding.role_indices().items():
        index = torch.tensor(indices, dtype=torch.int64)
        # log-mean-exp across all subword keys, then mean across heads.  No
        # threshold, top-k, connected component, or content-selected region is
        # introduced.
        role_logits = (
            torch.logsumexp(logits.index_select(3, index), dim=3)
            - math.log(float(len(indices)))
        ).mean(dim=2)
        role_weight = torch.softmax(role_logits, dim=1).contiguous()
        role_feature = torch.einsum("tp,tpd->td", role_weight, visual).contiguous()
        centroid = torch.einsum("tp,pd->td", role_weight, centers).contiguous()
        entropy = -(
            role_weight * role_weight.clamp_min(_EPS).log()
        ).sum(dim=1)
        weights[role] = role_weight
        features[role] = role_feature
        centroids[role] = centroid
        entropies[role] = entropy.contiguous()

    role_stack = torch.stack([weights[role] for role in ROLE_ORDER], dim=0)
    foreground = role_stack.max(dim=0).values
    background = (1.0 - foreground).clamp_min(_EPS)
    background = background / background.sum(dim=1, keepdim=True)
    background_feature = torch.einsum(
        "tp,tpd->td", background, visual
    ).contiguous()

    actor = centroids["actor"]
    obj = centroids["object"]
    anchor = centroids["anatomical_anchor"]
    action = centroids["action"]
    actor_object = obj - actor
    anchor_object = obj - anchor
    actor_object_distance = torch.linalg.vector_norm(actor_object, dim=1)
    anchor_object_distance = torch.linalg.vector_norm(anchor_object, dim=1)
    object_displacement = obj - obj[:1]
    common_centroid = torch.stack((actor, obj, anchor, action), dim=0).mean(dim=0)
    relative_scale = torch.sqrt(
        (
            torch.stack((actor, obj, anchor, action), dim=0)
            - common_centroid.unsqueeze(0)
        ).square().sum(dim=2).mean(dim=0).clamp_min(0.0)
    )
    relation = torch.stack(
        (
            actor_object[:, 0],
            actor_object[:, 1],
            actor_object_distance,
            _bhattacharyya(weights["actor"], weights["object"]),
            anchor_object[:, 0],
            anchor_object[:, 1],
            anchor_object_distance,
            _bhattacharyya(weights["anatomical_anchor"], weights["object"]),
            _bhattacharyya(weights["action"], weights["actor"]),
            _bhattacharyya(weights["action"], weights["object"]),
            object_displacement[:, 0],
            object_displacement[:, 1],
            common_centroid[:, 0],
            common_centroid[:, 1],
            relative_scale,
        ),
        dim=1,
    ).contiguous()
    if tuple(map(int, relation.shape)) != (
        LATENT_PHASES,
        len(RELATION_FEATURE_NAMES),
    ) or not bool(torch.isfinite(relation).all().item()):
        raise RelationalActionSlotError("relational path geometry/value differs")
    capture_digest = capture.receipt()["digest"]
    payload = {
        "branch": capture.branch,
        "capture_digest": capture_digest,
        "role_weight_sha256": {
            role: tensor_sha256(weights[role], label=f"{role} weights")
            for role in ROLE_ORDER
        },
        "role_feature_sha256": {
            role: tensor_sha256(features[role], label=f"{role} features")
            for role in ROLE_ORDER
        },
        "role_centroid_sha256": {
            role: tensor_sha256(centroids[role], label=f"{role} centroids")
            for role in ROLE_ORDER
        },
        "background_weight_sha256": tensor_sha256(
            background, label="background weights"
        ),
        "background_feature_sha256": tensor_sha256(
            background_feature, label="background features"
        ),
        "relation_feature_names": list(RELATION_FEATURE_NAMES),
        "relation_path_sha256": tensor_sha256(relation, label="relation path"),
        "softmax_temperature": 1.0,
        "threshold_or_topk_localization": False,
    }
    return LanguageIndexedSoftSlots(
        branch=capture.branch,
        role_weights=MappingProxyType(weights),
        role_features=MappingProxyType(features),
        role_centroids=MappingProxyType(centroids),
        role_entropy=MappingProxyType(entropies),
        background_weights=background.contiguous(),
        background_features=background_feature,
        relation_path=relation,
        capture_digest=capture_digest,
        digest=object_sha256(payload),
    )


@dataclass(frozen=True)
class SourceNativePreservationRows:
    rows: Mapping[str, torch.Tensor] = field(repr=False)
    normalized_magnitudes: Mapping[str, float]
    digest: str


def source_native_preservation_rows(
    action: LanguageIndexedSoftSlots,
    source_native_noop: LanguageIndexedSoftSlots,
) -> SourceNativePreservationRows:
    """Detached equality rows; these are proxies, not identity certification."""

    if action.branch != "action" or source_native_noop.branch != "noop":
        raise RelationalActionSlotError(
            "preservation rows require action and source-native noop slots"
        )
    rows = {
        "actor_temporal_dc_appearance": (
            action.role_features["actor"].mean(dim=0)
            - source_native_noop.role_features["actor"].mean(dim=0)
        ),
        "object_temporal_dc_appearance": (
            action.role_features["object"].mean(dim=0)
            - source_native_noop.role_features["object"].mean(dim=0)
        ),
        "anatomical_anchor_temporal_dc_appearance": (
            action.role_features["anatomical_anchor"].mean(dim=0)
            - source_native_noop.role_features["anatomical_anchor"].mean(dim=0)
        ),
        "background_temporal_dc_appearance": (
            action.background_features.mean(dim=0)
            - source_native_noop.background_features.mean(dim=0)
        ),
    }
    centroid_delta = torch.stack(
        [
            action.role_centroids[role]
            - source_native_noop.role_centroids[role]
            for role in ROLE_ORDER
        ],
        dim=0,
    )
    rows["camera_common_translation"] = centroid_delta.mean(dim=0).reshape(-1)
    scale_index = RELATION_FEATURE_NAMES.index("actor_object_relative_scale")
    rows["camera_relative_scale"] = (
        action.relation_path[:, scale_index]
        - source_native_noop.relation_path[:, scale_index]
    ).reshape(-1)
    if set(rows) != set(PRESERVATION_ROW_NAMES):
        raise RelationalActionSlotError("preservation row family differs")
    owned = {
        name: _owned_fp32(value, label=f"preservation row {name}")
        for name, value in rows.items()
    }
    references = {
        "actor_temporal_dc_appearance": source_native_noop.role_features[
            "actor"
        ].mean(dim=0),
        "object_temporal_dc_appearance": source_native_noop.role_features[
            "object"
        ].mean(dim=0),
        "anatomical_anchor_temporal_dc_appearance": (
            source_native_noop.role_features["anatomical_anchor"].mean(dim=0)
        ),
        "background_temporal_dc_appearance": (
            source_native_noop.background_features.mean(dim=0)
        ),
        "camera_common_translation": torch.ones_like(
            owned["camera_common_translation"]
        ),
        "camera_relative_scale": source_native_noop.relation_path[
            :, scale_index
        ],
    }
    magnitudes = {
        name: _norm(value) / max(_norm(references[name]), 1.0)
        for name, value in owned.items()
    }
    payload = {
        "row_sha256": {
            name: tensor_sha256(value, label=f"preservation {name}")
            for name, value in owned.items()
        },
        "normalized_magnitudes": magnitudes,
        "rows_are_detached_internal_proxy_equalities": True,
        "rows_are_parameter_vjps": False,
        "source_identity_or_camera_proven": False,
    }
    return SourceNativePreservationRows(
        rows=MappingProxyType(owned),
        normalized_magnitudes=MappingProxyType(magnitudes),
        digest=object_sha256(payload),
    )


@dataclass(frozen=True)
class FixedDiagnosticThresholds:
    minimum_action_noop_quotient_norm: float = 0.05
    minimum_reverse_retimed_cosine: float = 0.20
    maximum_action_reverse_same_order_cosine: float = 0.85
    minimum_incomplete_early_cosine: float = 0.20
    maximum_incomplete_terminal_energy_ratio: float = 0.85
    minimum_role_peak_over_uniform_ratio: float = 1.05
    maximum_actor_static_proxy_ratio: float = 0.75
    maximum_object_static_proxy_ratio: float = 0.75
    maximum_background_static_proxy_ratio: float = 0.75
    maximum_camera_common_translation_rms: float = 0.75


FIXED_DIAGNOSTIC_THRESHOLDS = FixedDiagnosticThresholds()


@dataclass(frozen=True)
class RelationalSlotAudit:
    slots: Mapping[str, LanguageIndexedSoftSlots] = field(repr=False)
    target_quotient: torch.Tensor = field(repr=False)
    reverse_quotient: torch.Tensor = field(repr=False)
    incomplete_quotient: torch.Tensor = field(repr=False)
    preservation: SourceNativePreservationRows
    metrics: Mapping[str, float]
    event_proxy_metrics: Mapping[str, float]
    reasons: tuple[str, ...]
    diagnostic_gate_passed: bool
    digest: str
    _token: Any = field(default=None, init=False, repr=False, compare=False)


def _start_anchor(path: torch.Tensor) -> torch.Tensor:
    return (path - path[:1]).contiguous()


def audit_relational_action_slots(
    captures: Mapping[str, GlobalRelationalQKCapture],
    *,
    event_ranges: EventPhaseRanges = CDF_DOG_PREREGISTERED_EVENT_RANGES,
) -> RelationalSlotAudit:
    """Compute a fixed representation/control diagnostic; authorize nothing."""

    if set(captures) != set(BRANCH_ORDER) or not isinstance(
        event_ranges, EventPhaseRanges
    ):
        raise RelationalActionSlotError(
            "relational audit requires exactly action/noop/reverse/incomplete"
        )
    ordered = [captures[branch] for branch in BRANCH_ORDER]
    if any(
        type(row) is not GlobalRelationalQKCapture
        or row._token is not _GLOBAL_CAPTURE_TOKEN
        or row.branch != branch
        for branch, row in zip(BRANCH_ORDER, ordered)
    ):
        raise RelationalActionSlotError("branch capture identity differs")
    first = ordered[0]
    geometry = (
        first.patch_height,
        first.patch_width,
        first.head_count,
        first.head_dim,
        first.shared_noisy_state_sha256,
        first.runtime_origin,
    )
    if any(
        (
            row.patch_height,
            row.patch_width,
            row.head_count,
            row.head_dim,
            row.shared_noisy_state_sha256,
            row.runtime_origin,
        )
        != geometry
        for row in ordered
    ):
        raise RelationalActionSlotError(
            "counterfactual branches must share exact noisy state and Q geometry"
        )
    tokenizer_digests = {
        row.binding.tokenizer_receipt_digest for row in ordered
    }
    encoder_digests = {
        row.binding.text_encoder_receipt_digest for row in ordered
    }
    if len(tokenizer_digests) != 1 or len(encoder_digests) != 1:
        raise RelationalActionSlotError(
            "counterfactual branches must share tokenizer/text-encoder authority"
        )

    slots = {
        branch: compute_language_indexed_soft_slots(captures[branch])
        for branch in BRANCH_ORDER
    }
    paths = {
        branch: _start_anchor(slots[branch].relation_path)
        for branch in BRANCH_ORDER
    }
    target = event_ranges.reduce(paths["action"] - paths["noop"])
    reverse = event_ranges.reduce(paths["reverse"] - paths["noop"])
    incomplete = event_ranges.reduce(paths["incomplete"] - paths["noop"])
    reverse_retimed = _start_anchor(reverse.flip(0))
    early_target = target[:3]
    early_incomplete = incomplete[:3]
    terminal_target = target[3:]
    terminal_incomplete = incomplete[3:]
    preservation = source_native_preservation_rows(slots["action"], slots["noop"])

    role_peak_ratios = []
    patch_positions = first.patch_height * first.patch_width
    for branch in BRANCH_ORDER:
        for role in ROLE_ORDER:
            peak = float(slots[branch].role_weights[role].max(dim=1).values.mean())
            role_peak_ratios.append(peak * float(patch_positions))

    metrics = {
        "action_noop_quotient_norm": _norm(target),
        "reverse_retimed_cosine": _cosine(target, reverse_retimed),
        "action_reverse_same_order_cosine": _cosine(target, reverse),
        "incomplete_early_cosine": _cosine(early_target, early_incomplete),
        "incomplete_terminal_energy_ratio": _norm(terminal_incomplete)
        / max(_norm(terminal_target), _EPS),
        "minimum_role_peak_over_uniform_ratio": min(role_peak_ratios),
        "actor_static_proxy_ratio": preservation.normalized_magnitudes[
            "actor_temporal_dc_appearance"
        ],
        "object_static_proxy_ratio": preservation.normalized_magnitudes[
            "object_temporal_dc_appearance"
        ],
        "background_static_proxy_ratio": preservation.normalized_magnitudes[
            "background_temporal_dc_appearance"
        ],
        "camera_common_translation_rms": _norm(
            preservation.rows["camera_common_translation"]
        )
        / math.sqrt(
            float(preservation.rows["camera_common_translation"].numel())
        ),
    }
    index = {name: RELATION_FEATURE_NAMES.index(name) for name in RELATION_FEATURE_NAMES}
    frame_quotient = paths["action"] - paths["noop"]
    stage_quotient = event_ranges.reduce(frame_quotient)
    event_metrics = {
        "approach_distance_decrease_proxy": -float(
            stage_quotient[0, index["actor_object_distance"]].item()
        ),
        "contact_overlap_gain_proxy": float(
            (
                stage_quotient[1, index["actor_object_bhattacharyya_overlap"]]
                - stage_quotient[0, index["actor_object_bhattacharyya_overlap"]]
            ).item()
        ),
        "grip_anchor_object_overlap_gain_proxy": float(
            (
                stage_quotient[2, index["anchor_object_bhattacharyya_overlap"]]
                - stage_quotient[1, index["anchor_object_bhattacharyya_overlap"]]
            ).item()
        ),
        "lift_screen_upward_proxy": -float(
            stage_quotient[3, index["object_displacement_y_from_phase0"]].item()
        ),
        "hold_anchor_object_overlap_proxy": float(
            stage_quotient[4, index["anchor_object_bhattacharyya_overlap"]].item()
        ),
        "hold_lift_retention_proxy": -abs(
            float(
                (
                    stage_quotient[4, index["object_displacement_y_from_phase0"]]
                    - stage_quotient[3, index["object_displacement_y_from_phase0"]]
                ).item()
            )
        ),
    }
    thresholds = FIXED_DIAGNOSTIC_THRESHOLDS
    checks = (
        (
            metrics["action_noop_quotient_norm"]
            >= thresholds.minimum_action_noop_quotient_norm,
            "action_noop_relational_quotient_too_small",
        ),
        (
            metrics["reverse_retimed_cosine"]
            >= thresholds.minimum_reverse_retimed_cosine,
            "reverse_control_does_not_retime_to_action_order",
        ),
        (
            metrics["action_reverse_same_order_cosine"]
            <= thresholds.maximum_action_reverse_same_order_cosine,
            "reverse_control_is_not_temporally_distinct",
        ),
        (
            metrics["incomplete_early_cosine"]
            >= thresholds.minimum_incomplete_early_cosine,
            "incomplete_control_lacks_shared_early_relation",
        ),
        (
            metrics["incomplete_terminal_energy_ratio"]
            <= thresholds.maximum_incomplete_terminal_energy_ratio,
            "incomplete_control_contains_terminal_relation",
        ),
        (
            metrics["minimum_role_peak_over_uniform_ratio"]
            >= thresholds.minimum_role_peak_over_uniform_ratio,
            "language_roles_are_spatially_uniform",
        ),
        (
            metrics["actor_static_proxy_ratio"]
            <= thresholds.maximum_actor_static_proxy_ratio,
            "actor_static_internal_proxy_drift",
        ),
        (
            metrics["object_static_proxy_ratio"]
            <= thresholds.maximum_object_static_proxy_ratio,
            "object_static_internal_proxy_drift",
        ),
        (
            metrics["background_static_proxy_ratio"]
            <= thresholds.maximum_background_static_proxy_ratio,
            "background_static_internal_proxy_drift",
        ),
        (
            metrics["camera_common_translation_rms"]
            <= thresholds.maximum_camera_common_translation_rms,
            "camera_common_translation_internal_proxy_drift",
        ),
    )
    reasons = tuple(reason for passed, reason in checks if not passed)
    payload = {
        "slot_digests": {branch: slots[branch].digest for branch in BRANCH_ORDER},
        "target_quotient_sha256": tensor_sha256(target, label="target quotient"),
        "reverse_quotient_sha256": tensor_sha256(reverse, label="reverse quotient"),
        "incomplete_quotient_sha256": tensor_sha256(
            incomplete, label="incomplete quotient"
        ),
        "preservation_digest": preservation.digest,
        "metrics": metrics,
        "event_proxy_metrics": event_metrics,
        "fixed_thresholds": asdict(thresholds),
        "reasons": list(reasons),
        "diagnostic_gate_passed": not reasons,
        "decoded_semantics_adjudicated": False,
    }
    result = RelationalSlotAudit(
        slots=MappingProxyType(slots),
        target_quotient=_owned_fp32(target, label="target quotient"),
        reverse_quotient=_owned_fp32(reverse, label="reverse quotient"),
        incomplete_quotient=_owned_fp32(incomplete, label="incomplete quotient"),
        preservation=preservation,
        metrics=MappingProxyType(metrics),
        event_proxy_metrics=MappingProxyType(event_metrics),
        reasons=reasons,
        diagnostic_gate_passed=not reasons,
        digest=object_sha256(payload),
    )
    object.__setattr__(result, "_token", _AUDIT_TOKEN)
    return result


@dataclass(frozen=True)
class RelationalProbeProvenance:
    probe_id: str
    checkpoint_content_sha256: str
    source_video_sha256: str
    source_registry_sha256: str
    method_revision_sha256: str
    shared_noisy_state_sha256: str
    query_seed: int
    schedule_index: int = NATIVE_SCHEDULE_INDEX
    timestep: int = NATIVE_TIMESTEP
    sigma: float = NATIVE_SIGMA

    def __post_init__(self) -> None:
        _require_identifier(self.probe_id, label="probe_id")
        for name in (
            "checkpoint_content_sha256",
            "source_video_sha256",
            "source_registry_sha256",
            "method_revision_sha256",
            "shared_noisy_state_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if type(self.query_seed) is not int or self.query_seed < 0:
            raise RelationalActionSlotError("query seed must be a nonnegative integer")
        if (
            self.schedule_index != NATIVE_SCHEDULE_INDEX
            or self.timestep != NATIVE_TIMESTEP
            or not math.isclose(float(self.sigma), NATIVE_SIGMA, rel_tol=0.0, abs_tol=0.0)
        ):
            raise RelationalActionSlotError("native schedule33/timestep516 coordinate differs")

    def receipt(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                **asdict(self),
                "frame_count": FRAME_COUNT,
                "latent_phases": LATENT_PHASES,
            }
        )


_RECEIPT_TOP_KEYS = {
    "schema_version",
    "method",
    "status",
    "provenance",
    "hook_plan",
    "event_stage_ranges",
    "branch_captures",
    "representation",
    "quotient",
    "source_native_preservation",
    "diagnostics",
    "forbidden_inputs_and_actions",
    "authority",
    "receipt_digest",
}


def build_relational_probe_receipt(
    *,
    provenance: RelationalProbeProvenance,
    captures: Mapping[str, GlobalRelationalQKCapture],
    audit: RelationalSlotAudit,
    event_ranges: EventPhaseRanges = CDF_DOG_PREREGISTERED_EVENT_RANGES,
) -> Mapping[str, Any]:
    if (
        not isinstance(provenance, RelationalProbeProvenance)
        or type(audit) is not RelationalSlotAudit
        or audit._token is not _AUDIT_TOKEN
        or set(captures) != set(BRANCH_ORDER)
        or provenance.shared_noisy_state_sha256
        != next(iter(captures.values())).shared_noisy_state_sha256
        or any(
            audit.slots[branch].capture_digest
            != captures[branch].receipt()["digest"]
            for branch in BRANCH_ORDER
        )
    ):
        raise RelationalActionSlotError("receipt provenance/capture/audit binding differs")
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "method": METHOD,
        "status": STATUS,
        "provenance": dict(provenance.receipt()),
        "hook_plan": [asdict(row) for row in bernini_relational_hook_plan()],
        "event_stage_ranges": dict(event_ranges.receipt()),
        "branch_captures": {
            branch: dict(captures[branch].receipt()) for branch in BRANCH_ORDER
        },
        "representation": {
            "role_order": list(ROLE_ORDER),
            "relation_feature_names": list(RELATION_FEATURE_NAMES),
            "slot_digests": {
                branch: audit.slots[branch].digest for branch in BRANCH_ORDER
            },
            "qk_softmax_temperature": 1.0,
            "threshold_or_topk_localization": False,
            "spatial_order_retained": True,
            "language_token_roles_are_only_localizer": True,
            "flash_attention_weights_claimed_exact": False,
            "qk_logits_reconstructed_after_model_owned_optional_norm_replay": True,
        },
        "quotient": {
            "formula": (
                "start_anchor(relation(branch))-start_anchor(relation(noop));"
                "stage_reduce=approach,contact,grip,lift,hold"
            ),
            "target_sha256": tensor_sha256(
                audit.target_quotient, label="receipt target quotient"
            ),
            "reverse_sha256": tensor_sha256(
                audit.reverse_quotient, label="receipt reverse quotient"
            ),
            "incomplete_sha256": tensor_sha256(
                audit.incomplete_quotient, label="receipt incomplete quotient"
            ),
            "action_vs_noop_reverse_incomplete_controls_required": True,
        },
        "source_native_preservation": {
            "row_names": list(PRESERVATION_ROW_NAMES),
            "row_sha256": {
                name: tensor_sha256(
                    audit.preservation.rows[name], label=f"receipt preservation {name}"
                )
                for name in PRESERVATION_ROW_NAMES
            },
            "normalized_magnitudes": dict(
                audit.preservation.normalized_magnitudes
            ),
            "digest": audit.preservation.digest,
            "detached_internal_proxy_equalities_only": True,
            "parameter_vjp_or_training_constraint_created": False,
        },
        "diagnostics": {
            "metrics": dict(audit.metrics),
            "event_proxy_metrics": dict(audit.event_proxy_metrics),
            "fixed_thresholds": asdict(FIXED_DIAGNOSTIC_THRESHOLDS),
            "reasons": list(audit.reasons),
            "diagnostic_gate_passed": audit.diagnostic_gate_passed,
            "ordered_event_semantics_adjudicated": False,
            "decoded_exact81_evaluation_performed": False,
            "source_correspondence_proven": False,
        },
        "forbidden_inputs_and_actions": {
            "external_mask": False,
            "external_track": False,
            "external_pose": False,
            "external_flow": False,
            "external_trajectory": False,
            "detector_or_segmenter": False,
            "content_selected_visual_topk_or_threshold": False,
            "generic_callback_evaluator": False,
            "self_reported_success_or_pass_input": False,
            "backward": False,
            "optimizer": False,
            "lora": False,
            "checkpoint_write": False,
            "parameter_mutation": False,
        },
        "authority": {
            "scientific_authority": False,
            "real_auh_runtime_validated": False,
            "decoded_action_success_claim_authorized": False,
            "identity_object_camera_preservation_claim_authorized": False,
            "training_updates_authorized": 0,
            "parameter_updates_executed": 0,
            "optimizer_created": False,
            "checkpoint_created": False,
            "single_example_conclusion_authorized": False,
        },
    }
    receipt = {**payload, "receipt_digest": object_sha256(payload)}
    validate_relational_probe_receipt(receipt)
    return MappingProxyType(receipt)


def validate_relational_probe_receipt(receipt: Mapping[str, Any]) -> None:
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_TOP_KEYS:
        raise RelationalActionSlotError("receipt does not match the closed schema")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if (
        receipt["schema_version"] != RECEIPT_SCHEMA_VERSION
        or receipt["method"] != METHOD
        or receipt["status"] != STATUS
        or _require_sha256(receipt["receipt_digest"], label="receipt digest")
        != object_sha256(unsigned)
    ):
        raise RelationalActionSlotError("receipt identity or seal differs")
    forbidden = receipt["forbidden_inputs_and_actions"]
    if not isinstance(forbidden, Mapping) or any(value is not False for value in forbidden.values()):
        raise RelationalActionSlotError("forbidden input/action receipt must remain false")
    authority = receipt["authority"]
    expected_authority = {
        "scientific_authority": False,
        "real_auh_runtime_validated": False,
        "decoded_action_success_claim_authorized": False,
        "identity_object_camera_preservation_claim_authorized": False,
        "training_updates_authorized": 0,
        "parameter_updates_executed": 0,
        "optimizer_created": False,
        "checkpoint_created": False,
        "single_example_conclusion_authorized": False,
    }
    if authority != expected_authority:
        raise RelationalActionSlotError("receipt authority must remain fail-closed")
    diagnostics = receipt["diagnostics"]
    if (
        not isinstance(diagnostics, Mapping)
        or diagnostics.get("ordered_event_semantics_adjudicated") is not False
        or diagnostics.get("decoded_exact81_evaluation_performed") is not False
        or diagnostics.get("source_correspondence_proven") is not False
        or bool(diagnostics.get("diagnostic_gate_passed"))
        != (not diagnostics.get("reasons"))
    ):
        raise RelationalActionSlotError("diagnostic receipt boundary differs")
    for group in (diagnostics.get("metrics"), diagnostics.get("event_proxy_metrics")):
        if not isinstance(group, Mapping) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in group.values()
        ):
            raise RelationalActionSlotError("receipt diagnostic scalar differs")


__all__ = [
    "BRANCH_ORDER",
    "BerniniRelationalCrossAttentionObserver",
    "CAPTURE_SCHEMA_VERSION",
    "CDF_DOG_PREREGISTERED_EVENT_RANGES",
    "EVENT_STAGE_ORDER",
    "EventPhaseRanges",
    "FIXED_DIAGNOSTIC_THRESHOLDS",
    "FRAME_COUNT",
    "GlobalRelationalQKCapture",
    "HOOK_BLOCK_INDEX",
    "HIDDEN_SIZE_1P3B",
    "LATENT_PHASES",
    "LanguageIndexedSoftSlots",
    "LanguageRoleTokenBinding",
    "LocalRelationalQKCapture",
    "METHOD",
    "PRESERVATION_ROW_NAMES",
    "RECEIPT_SCHEMA_VERSION",
    "RELATION_FEATURE_NAMES",
    "ROLE_ORDER",
    "RelationalActionSlotError",
    "RelationalHookPoint",
    "RelationalProbeProvenance",
    "RelationalSlotAudit",
    "SCHEMA_VERSION",
    "SP_SIZE",
    "STATUS",
    "SourceNativePreservationRows",
    "TargetSuffixSPLayout",
    "assemble_sp4_relational_capture",
    "audit_relational_action_slots",
    "bernini_relational_hook_plan",
    "build_relational_probe_receipt",
    "build_target_suffix_sp_layout",
    "canonical_json_bytes",
    "compute_language_indexed_soft_slots",
    "object_sha256",
    "source_native_preservation_rows",
    "tensor_sha256",
    "validate_relational_probe_receipt",
]
