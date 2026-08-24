#!/usr/bin/env python3
"""Late, source-conditioned appearance anchor for Bernini SAIC.

This module is the preservation branch of the Source-Anchored Inverse-Cycle
Action Operator (SAIC).  It wraps only ``attn1.to_q`` and
``attn1.to_out[0]`` in Bernini-R 1.3B blocks 23..29.  On the noisy-target
suffix of a native full-source ``V`` or ``VI`` branch, the residual is

    output_up(silu(state_down(hidden)))

with bias-free FP32 rank-8 projections and a zero-initialized output map.
Source-video rows, image-reference rows, and Ulysses append-padding rows are
therefore exact base-model bytes.  The branch is prompt-role agnostic: the
same preservation path can surround action and no-op forwards.  It is active
only at the five pinned low-noise UniPC coordinates 35..39.

Route metadata is not accepted from callers.  ``handle.route`` consumes the
native pack branch, audited live scheduler, and the exact timestep already
used by the real forward (official device-local INT64 or manual device-local
FP32).  It
derives the target selector from the native branch tensor, reads SP rank/size
from Bernini's live parallel state, and derives the sigma index from the
audited scheduler.  These objects remain bound and are revalidated while the
route is active.  Gradient checkpointing is forbidden because recomputation
outside the route would be ambiguous.

This is an adapter/routing primitive.  It consumes no object mask, pose,
track, optical flow, trajectory, proposal video, or target video.  A target
suffix is Bernini's native packed sequence structure, not a spatial object
mask.  The module does not create an optimizer, authorize training, or claim
semantic action-editing or appearance-preservation success.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Optional
import weakref

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from . import inference_sigma_strata as sigma_strata
    from . import source_self_native_ref_contrastive_v3 as native_pack
else:  # Direct execution/import from methods/bernini_action_editing.
    import inference_sigma_strata as sigma_strata
    import source_self_native_ref_contrastive_v3 as native_pack


SCHEMA_VERSION = "bernini-saic-source-anchor-adapter-v1"
CHECKPOINT_SCHEMA_VERSION = "bernini-saic-source-anchor-checkpoint-v1"
CLASSIFICATION = "adapter_primitive_only/no_training_authority"

TOTAL_BLOCKS_1P3B = 30
SOURCE_ANCHOR_BLOCK_INDICES = tuple(range(23, 30))
SOURCE_ANCHOR_RANK = 8
FULL_SOURCE_BRANCHES = ("V", "VI")
ACTIVE_SIGMA_INDICES = tuple(range(35, 40))
ALLOWED_SP_SIZES = frozenset({1, 4})
LATENT_PHASES_EXACT81 = 21

_ROUTE_MINT = object()


class SAICSourceAnchorError(RuntimeError):
    """Raised before an ambiguous source-anchor state can be used."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SAICSourceAnchorError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _storage_pointer(value: torch.Tensor) -> int:
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:  # pragma: no cover - older torch compatibility
        return int(value.storage().data_ptr())


def _tensor_binding(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        id(value),
        _storage_pointer(value),
        int(value.storage_offset()),
        tuple(map(int, value.shape)),
        tuple(map(int, value.stride())),
        value.dtype,
        value.device,
        value.layout,
        int(getattr(value, "_version", 0)),
    )


def _assert_tensor_binding(
    value: Any, expected: tuple[Any, ...], *, label: str
) -> torch.Tensor:
    if type(value) is not torch.Tensor or _tensor_binding(value) != expected:
        raise SAICSourceAnchorError(f"bound runtime {label} changed")
    return value


def _gradient_checkpointing_flags(transformer: nn.Module) -> tuple[str, ...]:
    owners: list[tuple[str, Any]] = [("transformer", transformer)]
    config = getattr(transformer, "config", None)
    if config is not None:
        owners.append(("transformer.config", config))
    for index, block in enumerate(tuple(getattr(transformer, "blocks", ()))):
        owners.append((f"blocks.{index}", block))
    enabled: list[str] = []
    for owner_name, owner in owners:
        for attribute in (
            "gradient_checkpointing",
            "is_gradient_checkpointing",
            "_gradient_checkpointing",
        ):
            value = (
                owner.get(attribute)
                if isinstance(owner, Mapping)
                else getattr(owner, attribute, None)
            )
            if callable(value) or value is None:
                continue
            try:
                active = bool(value)
            except Exception as error:
                raise SAICSourceAnchorError(
                    f"cannot read gradient-checkpointing flag {owner_name}.{attribute}"
                ) from error
            if active:
                enabled.append(f"{owner_name}.{attribute}")
    return tuple(enabled)


def _assert_gradient_checkpointing_disabled(transformer: nn.Module) -> None:
    enabled = _gradient_checkpointing_flags(transformer)
    if enabled:
        raise SAICSourceAnchorError(
            "gradient checkpointing is forbidden for the branch-local source "
            f"anchor route: {enabled[:3]}"
        )


def _get_live_parallel_state() -> Any:
    """Return Bernini's installed state; tests may patch this private seam."""

    try:
        from bernini.parallel import get_parallel_state

        state = get_parallel_state()
    except Exception as error:
        raise SAICSourceAnchorError(
            "Bernini live parallel state is unavailable"
        ) from error
    if state is None:
        raise SAICSourceAnchorError("Bernini live parallel state is unavailable")
    return state


def _parallel_coordinates(state: Any) -> tuple[int, int]:
    size = getattr(state, "ulysses_size", None)
    rank = getattr(state, "ulysses_rank", None)
    if type(size) is not int or size not in ALLOWED_SP_SIZES:
        raise SAICSourceAnchorError(
            "live Bernini Ulysses size must be production SP4 or test SP1"
        )
    if type(rank) is not int or not 0 <= rank < size:
        raise SAICSourceAnchorError("live Bernini Ulysses rank is invalid")
    return rank, size


def _timestep_schedule_index(timestep: Any) -> int:
    if (
        type(timestep) is not torch.Tensor
        # The manual Stage-A scorer builds device-local FP32 coordinates,
        # whereas the unmodified Bernini UniPC sampler copies its audited CPU
        # timeline to the forward device and forwards the device-local INT64
        # ``t.expand(1)`` view.  Both are
        # exact integer-valued representations of the same pinned coordinate;
        # accepting only FP32 made a trained anchor impossible to deploy in
        # the official sampler.
        or timestep.dtype not in (torch.float32, torch.int64)
        or timestep.numel() != 1
        or timestep.requires_grad
        or timestep.grad_fn is not None
        or not bool(torch.isfinite(timestep).all().item())
    ):
        raise SAICSourceAnchorError(
            "runtime timestep must be one detached finite exact INT64/FP32 tensor"
        )
    numeric = float(timestep.detach().item())
    if numeric != float(int(numeric)):
        raise SAICSourceAnchorError(
            "runtime timestep must be an exact integer-valued coordinate"
        )
    matches = [
        index
        for index, expected in enumerate(sigma_strata.PINNED_TIMESTEPS)
        if numeric == float(expected)
    ]
    if len(matches) != 1:
        raise SAICSourceAnchorError(
            "runtime timestep is not one unique pinned UniPC40 coordinate"
        )
    return matches[0]


def _validate_full_source_branch(branch: Any) -> Any:
    if type(branch) is not native_pack.NativeRV2VBranch:
        raise SAICSourceAnchorError(
            "source anchor requires an exact native NativeRV2VBranch"
        )
    if branch.name not in FULL_SOURCE_BRANCHES:
        raise SAICSourceAnchorError(
            "source anchor accepts only full-source native V/VI branches"
        )
    target_tokens = branch.total_tokens - branch.condition_tokens
    if target_tokens <= 0 or target_tokens % LATENT_PHASES_EXACT81:
        raise SAICSourceAnchorError(
            "native target suffix is not exact81 latent geometry"
        )
    patch_positions = target_tokens // LATENT_PHASES_EXACT81
    if branch.name == "V":
        expected_condition_tokens = target_tokens
        expected_source_ids = native_pack.VI_VIDEO_SOURCE_IDS + (0.0,)
    else:
        expected_condition_tokens = target_tokens + native_pack.REFERENCE_COUNT * patch_positions
        expected_source_ids = (
            native_pack.VI_VIDEO_SOURCE_IDS
            + native_pack.VI_IMAGE_SOURCE_IDS
            + (0.0,)
        )
    expected_order = native_pack.BRANCH_CONCAT_ORDER[branch.name]
    if (
        branch.condition_tokens != expected_condition_tokens
        or branch.source_ids != expected_source_ids
        or branch.concat_order != expected_order
        or type(branch.latents) is not torch.Tensor
        or branch.latents.ndim != 3
        or int(branch.latents.shape[0]) != 1
        or int(branch.latents.shape[1]) != branch.total_tokens
        or type(branch.target_mask) is not torch.Tensor
        or branch.target_mask.dtype != torch.bool
        or branch.target_mask.ndim != 1
        or int(branch.target_mask.numel()) != branch.total_tokens
        or branch.target_mask.device != branch.latents.device
    ):
        raise SAICSourceAnchorError("native full-source branch closure differs")
    expected_mask = torch.zeros_like(branch.target_mask)
    expected_mask[branch.condition_tokens :] = True
    if not torch.equal(branch.target_mask, expected_mask):
        raise SAICSourceAnchorError(
            "native branch target mask is not the exact noisy-target suffix"
        )
    return branch


def _derive_local_target_mask(
    global_mask: torch.Tensor, *, rank: int, size: int
) -> torch.Tensor:
    local_length = math.ceil(int(global_mask.numel()) / size)
    padded_length = local_length * size
    if padded_length > int(global_mask.numel()):
        global_mask = F.pad(
            global_mask,
            (0, padded_length - int(global_mask.numel())),
            value=False,
        )
    start = rank * local_length
    return global_mask[start : start + local_length].clone().contiguous()


@dataclass(frozen=True)
class _SourceAnchorRuntimeRoute:
    """Opaque route minted only from live native runtime objects."""

    branch: Any = field(repr=False, compare=False)
    scheduler: Any = field(repr=False, compare=False)
    timestep: torch.Tensor = field(repr=False, compare=False)
    parallel_state: Any = field(repr=False, compare=False)
    owner_token: Any = field(repr=False, compare=False)
    branch_name: str
    schedule_index: int
    timestep_dtype: str
    timestep_device: str
    sigma_float32_be_hex: str
    sequence_parallel_rank: int
    sequence_parallel_size: int
    local_target_mask: torch.Tensor = field(repr=False, compare=False)
    _branch_latents_binding: tuple[Any, ...] = field(repr=False, compare=False)
    _branch_mask_binding: tuple[Any, ...] = field(repr=False, compare=False)
    _scheduler_timesteps_binding: tuple[Any, ...] = field(repr=False, compare=False)
    _scheduler_sigmas_binding: tuple[Any, ...] = field(repr=False, compare=False)
    _timestep_binding: tuple[Any, ...] = field(repr=False, compare=False)
    _local_mask_binding: tuple[Any, ...] = field(repr=False, compare=False)
    _mint: Any = field(repr=False, compare=False)

    @property
    def adapter_active(self) -> bool:
        return self.schedule_index in ACTIVE_SIGMA_INDICES

    def assert_live(self, *, owner_token: Any, transformer: nn.Module) -> None:
        if self._mint is not _ROUTE_MINT or owner_token is not self.owner_token:
            raise SAICSourceAnchorError("source-anchor route owner differs")
        _assert_gradient_checkpointing_disabled(transformer)
        branch = _validate_full_source_branch(self.branch)
        if branch.name != self.branch_name:
            raise SAICSourceAnchorError("bound native branch role changed")
        _assert_tensor_binding(
            branch.latents, self._branch_latents_binding, label="branch latents"
        )
        _assert_tensor_binding(
            branch.target_mask, self._branch_mask_binding, label="target mask"
        )
        _assert_tensor_binding(
            self.timestep, self._timestep_binding, label="forward timestep"
        )
        if _timestep_schedule_index(self.timestep) != self.schedule_index:
            raise SAICSourceAnchorError("bound forward timestep value changed")
        timesteps = _assert_tensor_binding(
            getattr(self.scheduler, "timesteps", None),
            self._scheduler_timesteps_binding,
            label="scheduler timesteps",
        )
        sigmas = _assert_tensor_binding(
            getattr(self.scheduler, "sigmas", None),
            self._scheduler_sigmas_binding,
            label="scheduler sigmas",
        )
        if (
            int(timesteps[self.schedule_index].item())
            != sigma_strata.PINNED_TIMESTEPS[self.schedule_index]
            or sigma_strata._float32_hex(  # noqa: SLF001 - exact pinned-bit guard
                sigmas[self.schedule_index].item(), label="bound scheduler sigma"
            )
            != self.sigma_float32_be_hex
        ):
            raise SAICSourceAnchorError("bound scheduler coordinate changed")
        live_state = _get_live_parallel_state()
        rank, size = _parallel_coordinates(live_state)
        if (
            live_state is not self.parallel_state
            or rank != self.sequence_parallel_rank
            or size != self.sequence_parallel_size
        ):
            raise SAICSourceAnchorError("bound live Ulysses route changed")
        _assert_tensor_binding(
            self.local_target_mask,
            self._local_mask_binding,
            label="derived local target mask",
        )

    def selector(
        self, hidden_states: torch.Tensor, *, owner_token: Any, transformer: nn.Module
    ) -> torch.Tensor:
        self.assert_live(owner_token=owner_token, transformer=transformer)
        if (
            type(hidden_states) is not torch.Tensor
            or hidden_states.ndim != 3
            or int(hidden_states.shape[0]) != 1
            or hidden_states.device != self.branch.latents.device
            or int(hidden_states.shape[1]) != int(self.local_target_mask.numel())
        ):
            raise SAICSourceAnchorError(
                "hidden sequence differs from the runtime-derived native SP shard"
            )
        return self.local_target_mask

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "branch_name": self.branch_name,
            "full_source_conditioned": True,
            "schedule_index": self.schedule_index,
            "timestep_dtype": self.timestep_dtype,
            "timestep_device": self.timestep_device,
            "sigma_float32_be_hex": self.sigma_float32_be_hex,
            "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
            "sigma_active": self.adapter_active,
            "sequence_parallel_rank": self.sequence_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "local_rows": int(self.local_target_mask.numel()),
            "local_target_rows": int(self.local_target_mask.sum().item()),
            "mask_derivation": "native_global_suffix_append_false_then_live_sp_slice",
            "caller_supplied_rank_size_index_or_mask": False,
        }
        return {**value, "digest": _object_sha256(value)}


def _mint_runtime_route(
    *,
    branch: Any,
    scheduler: Any,
    timestep: torch.Tensor,
    transformer: nn.Module,
    owner_token: Any,
) -> _SourceAnchorRuntimeRoute:
    _assert_gradient_checkpointing_disabled(transformer)
    checked_branch = _validate_full_source_branch(branch)
    if timestep.device != checked_branch.latents.device:
        raise SAICSourceAnchorError(
            "runtime timestep and native branch must share the forward device"
        )
    try:
        schedule_receipt = sigma_strata.audit_runtime_unipc_schedule(
            scheduler, initialize=False
        )
    except Exception as error:
        raise SAICSourceAnchorError(
            f"runtime UniPC schedule is not pinned exact40: {error}"
        ) from error
    if schedule_receipt.get("schedule_sha256") != sigma_strata.SCHEDULE_SHA256:
        raise SAICSourceAnchorError("runtime UniPC schedule digest differs")
    schedule_index = _timestep_schedule_index(timestep)
    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if (
        type(timesteps) is not torch.Tensor
        or type(sigmas) is not torch.Tensor
        or int(timesteps[schedule_index].item())
        != sigma_strata.PINNED_TIMESTEPS[schedule_index]
    ):
        raise SAICSourceAnchorError("actual scheduler coordinate differs")
    sigma_hex = sigma_strata._float32_hex(  # noqa: SLF001 - exact pinned-bit guard
        sigmas[schedule_index].item(), label="actual scheduler sigma"
    )
    if sigma_hex != sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[schedule_index]:
        raise SAICSourceAnchorError("actual scheduler sigma differs from pinned bits")
    parallel_state = _get_live_parallel_state()
    rank, size = _parallel_coordinates(parallel_state)
    local_mask = _derive_local_target_mask(
        checked_branch.target_mask, rank=rank, size=size
    )
    route = _SourceAnchorRuntimeRoute(
        branch=checked_branch,
        scheduler=scheduler,
        timestep=timestep,
        parallel_state=parallel_state,
        owner_token=owner_token,
        branch_name=checked_branch.name,
        schedule_index=schedule_index,
        timestep_dtype=str(timestep.dtype),
        timestep_device=str(timestep.device),
        sigma_float32_be_hex=sigma_hex,
        sequence_parallel_rank=rank,
        sequence_parallel_size=size,
        local_target_mask=local_mask,
        _branch_latents_binding=_tensor_binding(checked_branch.latents),
        _branch_mask_binding=_tensor_binding(checked_branch.target_mask),
        _scheduler_timesteps_binding=_tensor_binding(timesteps),
        _scheduler_sigmas_binding=_tensor_binding(sigmas),
        _timestep_binding=_tensor_binding(timestep),
        _local_mask_binding=_tensor_binding(local_mask),
        _mint=_ROUTE_MINT,
    )
    route.assert_live(owner_token=owner_token, transformer=transformer)
    return route


_ACTIVE_ROUTE: ContextVar[Optional[_SourceAnchorRuntimeRoute]] = ContextVar(
    "bernini_saic_source_anchor_runtime_route", default=None
)


def active_route() -> Optional[_SourceAnchorRuntimeRoute]:
    """Return the current opaque route for diagnostics only."""

    return _ACTIVE_ROUTE.get()


@contextmanager
def _activate_route(route: _SourceAnchorRuntimeRoute) -> Iterator[None]:
    if type(route) is not _SourceAnchorRuntimeRoute or route._mint is not _ROUTE_MINT:
        raise SAICSourceAnchorError("route was not minted from the live runtime")
    if active_route() is not None:
        raise SAICSourceAnchorError("nested source-anchor routes are forbidden")
    token: Token[Optional[_SourceAnchorRuntimeRoute]] = _ACTIVE_ROUTE.set(route)
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


class SAICSourceAnchorResidual(nn.Module):
    """FP32 rank-8 target-row state residual around one frozen self-attn map."""

    def __init__(
        self,
        base: nn.Module,
        *,
        projection: str,
        transformer: nn.Module,
        owner_token: Any,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise SAICSourceAnchorError(f"{projection} base must be nn.Linear")
        if projection not in {"to_q", "to_out.0"}:
            raise SAICSourceAnchorError("only self-attention Q/O may be wrapped")
        if any(parameter.requires_grad for parameter in base.parameters()):
            raise SAICSourceAnchorError("wrapped base projection must be frozen")
        self.base = base
        self.projection = projection
        self.rank = SOURCE_ANCHOR_RANK
        self.state_down = nn.Linear(
            base.in_features, self.rank, bias=False, dtype=torch.float32
        )
        self.output_up = nn.Linear(
            self.rank, base.out_features, bias=False, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.state_down.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.output_up.weight)
        self._transformer_ref = weakref.ref(transformer)
        self._owner_token = owner_token

    @property
    def weight(self) -> Any:
        return self.base.weight

    @property
    def bias(self) -> Any:
        return self.base.bias

    def _transformer(self) -> nn.Module:
        transformer = self._transformer_ref()
        if transformer is None:
            raise SAICSourceAnchorError("source-anchor transformer no longer exists")
        return transformer

    def _selected_delta(
        self, hidden_states: torch.Tensor, selector: torch.Tensor
    ) -> torch.Tensor:
        selected = hidden_states[:, selector, :]
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            state = F.silu(self.state_down(selected.float()))
            delta = self.output_up(state)
        return delta.to(hidden_states.dtype)

    def adapter_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            (*hidden_states.shape[:-1], self.base.out_features),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        route = active_route()
        if route is None:
            return result
        selector = route.selector(
            hidden_states,
            owner_token=self._owner_token,
            transformer=self._transformer(),
        )
        if not route.adapter_active:
            return result
        result[:, selector, :] = self._selected_delta(hidden_states, selector)
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base(hidden_states)
        route = active_route()
        if route is None:
            return base
        selector = route.selector(
            hidden_states,
            owner_token=self._owner_token,
            transformer=self._transformer(),
        )
        if not route.adapter_active:
            return base
        result = base.clone()
        # Evaluating an empty selected tensor preserves identical distributed
        # autograd topology on source-only SP shards while changing no bytes.
        result[:, selector, :] = base[:, selector, :] + self._selected_delta(
            hidden_states, selector
        )
        return result


def trainable_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Digest an exact detached, contiguous, CPU-FP32 state mapping."""

    if not isinstance(state, Mapping) or not state:
        raise SAICSourceAnchorError("source-anchor state must be a nonempty mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if (
            type(name) is not str
            or name.encode("ascii", "strict").decode("ascii") != name
            or type(value) is not torch.Tensor
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or value.requires_grad
            or value.grad_fn is not None
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise SAICSourceAnchorError(
                f"state {name!r} must be detached finite contiguous CPU FP32"
            )
        digest.update(name.encode("ascii"))
        digest.update(_canonical_json(list(map(int, value.shape))))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class SAICSourceAnchorHandle:
    transformer: nn.Module
    q_wrappers: tuple[tuple[int, SAICSourceAnchorResidual], ...]
    o_wrappers: tuple[tuple[int, SAICSourceAnchorResidual], ...]
    original_q: tuple[tuple[int, nn.Module], ...]
    original_o: tuple[tuple[int, nn.Module], ...]
    original_patch_embedding_id: int
    protected_ids: tuple[tuple[int, ...], ...]
    owner_token: Any = field(repr=False)
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise SAICSourceAnchorError("source-anchor adapter was restored")
        result: list[tuple[str, nn.Parameter]] = []
        for index, wrapper in self.q_wrappers:
            result.extend(
                (
                    (
                        f"blocks.{index}.attn1.to_q.state_down.weight",
                        wrapper.state_down.weight,
                    ),
                    (
                        f"blocks.{index}.attn1.to_q.output_up.weight",
                        wrapper.output_up.weight,
                    ),
                )
            )
        for index, wrapper in self.o_wrappers:
            result.extend(
                (
                    (
                        f"blocks.{index}.attn1.to_out.0.state_down.weight",
                        wrapper.state_down.weight,
                    ),
                    (
                        f"blocks.{index}.attn1.to_out.0.output_up.weight",
                        wrapper.output_up.weight,
                    ),
                )
            )
        if len(result) != len(SOURCE_ANCHOR_BLOCK_INDICES) * 4:
            raise SAICSourceAnchorError("source-anchor trainable key count differs")
        if len({id(parameter) for _, parameter in result}) != len(result):
            raise SAICSourceAnchorError("source-anchor parameter alias detected")
        if any(
            parameter.dtype != torch.float32 or not parameter.requires_grad
            for _, parameter in result
        ):
            raise SAICSourceAnchorError("source-anchor parameter gauge differs")
        return tuple(result)

    def base_parameters_frozen(self) -> bool:
        trainable_ids = {
            id(parameter) for _, parameter in self.trainable_named_parameters()
        }
        observed = {
            id(parameter)
            for parameter in self.transformer.parameters()
            if parameter.requires_grad
        }
        return observed == trainable_ids

    def scope_untouched(self) -> bool:
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        return (
            len(blocks) == TOTAL_BLOCKS_1P3B
            and _capture_protected_ids(blocks) == self.protected_ids
            and id(getattr(self.transformer, "patch_embedding", None))
            == self.original_patch_embedding_id
        )

    @contextmanager
    def route(
        self,
        *,
        branch: Any,
        scheduler: Any,
        timestep: torch.Tensor,
    ) -> Iterator[_SourceAnchorRuntimeRoute]:
        """Bind one native forward without accepting mask/rank/size/index."""

        if self.restored:
            raise SAICSourceAnchorError("cannot route a restored source anchor")
        route = _mint_runtime_route(
            branch=branch,
            scheduler=scheduler,
            timestep=timestep,
            transformer=self.transformer,
            owner_token=self.owner_token,
        )
        route.assert_live(owner_token=self.owner_token, transformer=self.transformer)
        with _activate_route(route):
            try:
                yield route
            finally:
                route.assert_live(
                    owner_token=self.owner_token, transformer=self.transformer
                )

    def state_dict_for_save(self) -> Mapping[str, torch.Tensor]:
        state = {
            name: parameter.detach().float().cpu().contiguous().clone()
            for name, parameter in self.trainable_named_parameters()
        }
        trainable_state_digest(state)
        return state

    def load_trainable_state_dict(
        self, state: Mapping[str, torch.Tensor]
    ) -> Mapping[str, Any]:
        if self.restored:
            raise SAICSourceAnchorError("cannot load a restored source anchor")
        if not isinstance(state, Mapping):
            raise SAICSourceAnchorError("source-anchor state must be a mapping")
        expected = dict(self.trainable_named_parameters())
        if set(state) != set(expected):
            missing = sorted(set(expected) - set(state))
            unexpected = sorted(set(state) - set(expected))
            raise SAICSourceAnchorError(
                "source-anchor state key closure differs: "
                f"missing={missing[:2]} unexpected={unexpected[:2]}"
            )
        normalized: dict[str, torch.Tensor] = {}
        for name, parameter in expected.items():
            value = state[name]
            if (
                type(value) is not torch.Tensor
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or value.layout != torch.strided
                or value.requires_grad
                or value.grad_fn is not None
                or not value.is_contiguous()
                or tuple(value.shape) != tuple(parameter.shape)
                or not bool(torch.isfinite(value).all().item())
            ):
                raise SAICSourceAnchorError(
                    f"state {name} must be exact-shape finite contiguous CPU FP32"
                )
            normalized[name] = value
        state_digest = trainable_state_digest(normalized)
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(normalized[name].to(device=parameter.device))
        value = {
            "schema_version": SCHEMA_VERSION,
            "closed_exact_key_set": True,
            "state_key_count": len(normalized),
            "state_key_sha256": _object_sha256(sorted(normalized)),
            "state_tensor_sha256": state_digest,
        }
        return {**value, "digest": _object_sha256(value)}

    def save_checkpoint(self, path: os.PathLike[str] | str) -> Mapping[str, Any]:
        destination = Path(path)
        if not destination.parent.is_dir() or destination.is_dir():
            raise SAICSourceAnchorError(
                "checkpoint parent must exist and destination must not be a directory"
            )
        state = dict(self.state_dict_for_save())
        state_digest = trainable_state_digest(state)
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "adapter_schema_version": SCHEMA_VERSION,
            "state_tensor_sha256": state_digest,
            "state": state,
        }
        temporary_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                torch.save(payload, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
        value = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "path": str(destination),
            "state_key_count": len(state),
            "state_tensor_sha256": state_digest,
        }
        return {**value, "digest": _object_sha256(value)}

    def load_checkpoint(self, path: os.PathLike[str] | str) -> Mapping[str, Any]:
        source = Path(path)
        if not source.is_file():
            raise SAICSourceAnchorError("source-anchor checkpoint is not a file")
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except Exception as error:
            raise SAICSourceAnchorError(
                f"failed to read weights-only source-anchor checkpoint: {error}"
            ) from error
        expected_keys = {
            "checkpoint_schema_version",
            "adapter_schema_version",
            "state_tensor_sha256",
            "state",
        }
        if (
            not isinstance(payload, Mapping)
            or set(payload) != expected_keys
            or payload["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION
            or payload["adapter_schema_version"] != SCHEMA_VERSION
            or type(payload["state_tensor_sha256"]) is not str
            or not isinstance(payload["state"], Mapping)
        ):
            raise SAICSourceAnchorError("source-anchor checkpoint envelope differs")
        digest = trainable_state_digest(payload["state"])
        if digest != payload["state_tensor_sha256"]:
            raise SAICSourceAnchorError("source-anchor checkpoint digest differs")
        load_receipt = self.load_trainable_state_dict(payload["state"])
        value = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "path": str(source),
            "state_tensor_sha256": digest,
            "load_receipt_digest": load_receipt["digest"],
        }
        return {**value, "digest": _object_sha256(value)}

    def receipt(self) -> Mapping[str, Any]:
        patch = getattr(self.transformer, "patch_embedding", None)
        trainable = self.trainable_named_parameters()
        value = {
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "blocks": list(SOURCE_ANCHOR_BLOCK_INDICES),
            "projections": ["attn1.to_q", "attn1.to_out.0"],
            "operator": "output_up(silu(state_down(hidden)))",
            "rank": SOURCE_ANCHOR_RANK,
            "fp32_trainable": True,
            "bias": False,
            "output_up_zero_initialized_at_install": True,
            "full_source_native_branches": list(FULL_SOURCE_BRANCHES),
            "active_sigma_indices": list(ACTIVE_SIGMA_INDICES),
            "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
            "target_suffix_is_native_pack_structure_not_object_mask": True,
            "source_reference_padding_rows_exact_base": True,
            "prompt_role_agnostic_action_and_noop": True,
            "route_accepts_caller_rank_size_index_or_mask": False,
            "route_binds_live_parallel_native_mask_and_actual_scheduler_sigma": True,
            "accepted_timestep_representations": [
                "official_device_local_int64",
                "manual_device_local_float32",
            ],
            "gradient_checkpointing_supported": False,
            "patch_embedding_untouched": id(patch)
            == self.original_patch_embedding_id,
            "only_registered_self_attention_qo_replaced": self.scope_untouched(),
            "base_parameters_frozen": self.base_parameters_frozen(),
            "trainable_state_closed": True,
            "trainable_state_key_sha256": _object_sha256(
                sorted(name for name, _ in trainable)
            ),
            "trainable": [
                {
                    "name": name,
                    "shape": list(map(int, parameter.shape)),
                    "dtype": str(parameter.dtype),
                }
                for name, parameter in trainable
            ],
            "mask_pose_flow_track_trajectory_consumed": False,
            "proposal_or_target_video_consumed": False,
            "optimizer_created": False,
            "training_authorized": False,
            "semantic_action_success_claim": False,
            "appearance_preservation_success_claim": False,
        }
        return {**value, "digest": _object_sha256(value)}

    def restore(self) -> None:
        if self.restored or active_route() is not None:
            raise SAICSourceAnchorError("source anchor cannot be restored now")
        if not self.scope_untouched():
            raise SAICSourceAnchorError("source-anchor protected model scope changed")
        blocks = tuple(self.transformer.blocks)
        for index, original in self.original_q:
            blocks[index].attn1.to_q = original
        for index, original in self.original_o:
            blocks[index].attn1.to_out[0] = original
        self.restored = True


def _capture_protected_ids(
    blocks: tuple[nn.Module, ...]
) -> tuple[tuple[int, ...], ...]:
    """Capture every attention component except the registered late Q/O slots."""

    rows: list[tuple[int, ...]] = []
    for index, block in enumerate(blocks):
        attn1 = getattr(block, "attn1", None)
        attn2 = getattr(block, "attn2", None)
        self_out = getattr(attn1, "to_out", None)
        cross_out = getattr(attn2, "to_out", None)
        if (
            attn1 is None
            or attn2 is None
            or not isinstance(self_out, nn.ModuleList)
            or len(self_out) != 2
            or not isinstance(cross_out, nn.ModuleList)
            or len(cross_out) != 2
            or getattr(attn1, "to_q", None) is None
            or getattr(attn1, "to_k", None) is None
            or getattr(attn1, "to_v", None) is None
            or getattr(attn2, "to_q", None) is None
            or getattr(attn2, "to_k", None) is None
            or getattr(attn2, "to_v", None) is None
        ):
            raise SAICSourceAnchorError(
                f"block {index} native attention structure differs"
            )
        row = [
            id(attn1),
            id(attn1.to_k),
            id(attn1.to_v),
            id(self_out[1]),
            id(attn2),
            id(attn2.to_q),
            id(attn2.to_k),
            id(attn2.to_v),
            id(cross_out[0]),
            id(cross_out[1]),
        ]
        if index not in SOURCE_ANCHOR_BLOCK_INDICES:
            row.extend((id(attn1.to_q), id(self_out[0])))
        rows.append(tuple(row))
    return tuple(rows)


def install_saic_source_anchor_adapter(
    transformer: nn.Module,
) -> SAICSourceAnchorHandle:
    """Install the function-preserving late source-anchor wrappers."""

    if not isinstance(transformer, nn.Module):
        raise SAICSourceAnchorError("transformer must be nn.Module")
    _assert_gradient_checkpointing_disabled(transformer)
    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise SAICSourceAnchorError(
            "freeze the complete Bernini transformer before source-anchor installation"
        )
    blocks = tuple(getattr(transformer, "blocks", ()))
    patch = getattr(transformer, "patch_embedding", None)
    if (
        len(blocks) != TOTAL_BLOCKS_1P3B
        or not isinstance(patch, nn.Conv3d)
        or not callable(getattr(transformer, "patch_vae_latent", None))
    ):
        raise SAICSourceAnchorError(
            "Bernini-R 1.3B native transformer structure differs"
        )
    hidden = int(patch.out_channels)
    protected_ids = _capture_protected_ids(blocks)
    original_q: list[tuple[int, nn.Module]] = []
    original_o: list[tuple[int, nn.Module]] = []
    for index in SOURCE_ANCHOR_BLOCK_INDICES:
        attention = getattr(blocks[index], "attn1", None)
        query = getattr(attention, "to_q", None)
        output = getattr(attention, "to_out", None)
        if (
            not isinstance(query, nn.Linear)
            or not isinstance(output, nn.ModuleList)
            or len(output) != 2
            or not isinstance(output[0], nn.Linear)
            or query.in_features != hidden
            or query.out_features != hidden
            or output[0].in_features != hidden
            or output[0].out_features != hidden
        ):
            raise SAICSourceAnchorError(
                f"block {index} native self-attention Q/O differs"
            )
        original_q.append((index, query))
        original_o.append((index, output[0]))

    owner_token = object()
    device = patch.weight.device
    q_wrappers: list[tuple[int, SAICSourceAnchorResidual]] = []
    o_wrappers: list[tuple[int, SAICSourceAnchorResidual]] = []
    try:
        for (index, query), (_, output) in zip(original_q, original_o):
            q_wrapper = SAICSourceAnchorResidual(
                query,
                projection="to_q",
                transformer=transformer,
                owner_token=owner_token,
            ).to(device=device)
            o_wrapper = SAICSourceAnchorResidual(
                output,
                projection="to_out.0",
                transformer=transformer,
                owner_token=owner_token,
            ).to(device=device)
            blocks[index].attn1.to_q = q_wrapper
            blocks[index].attn1.to_out[0] = o_wrapper
            q_wrappers.append((index, q_wrapper))
            o_wrappers.append((index, o_wrapper))
    except Exception:
        for index, original in original_q:
            blocks[index].attn1.to_q = original
        for index, original in original_o:
            blocks[index].attn1.to_out[0] = original
        raise

    handle = SAICSourceAnchorHandle(
        transformer=transformer,
        q_wrappers=tuple(q_wrappers),
        o_wrappers=tuple(o_wrappers),
        original_q=tuple(original_q),
        original_o=tuple(original_o),
        original_patch_embedding_id=id(patch),
        protected_ids=protected_ids,
        owner_token=owner_token,
    )
    receipt = handle.receipt()
    if (
        receipt["only_registered_self_attention_qo_replaced"] is not True
        or receipt["base_parameters_frozen"] is not True
        or receipt["patch_embedding_untouched"] is not True
    ):
        handle.restore()
        raise SAICSourceAnchorError("source-anchor installation scope failed")
    return handle


__all__ = [
    "ACTIVE_SIGMA_INDICES",
    "CHECKPOINT_SCHEMA_VERSION",
    "CLASSIFICATION",
    "FULL_SOURCE_BRANCHES",
    "SAICSourceAnchorError",
    "SAICSourceAnchorHandle",
    "SAICSourceAnchorResidual",
    "SCHEMA_VERSION",
    "SOURCE_ANCHOR_BLOCK_INDICES",
    "SOURCE_ANCHOR_RANK",
    "TOTAL_BLOCKS_1P3B",
    "active_route",
    "install_saic_source_anchor_adapter",
    "trainable_state_digest",
]
