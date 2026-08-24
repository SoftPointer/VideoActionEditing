"""Pure-T2V-anchor Q/K/V transport for Bernini visual self-attention.

The anchor is evaluated online at the same solver step, candidate-noise chain,
and latent geometry as the edited state.  Its post-RoPE visual query/key tensors
are captured from a pure-T2V forward.  A following full-source V2V target
forward replaces only the target-token Q/K suffix with the captured tensors:

    source Q/K       := current full-source V2V Q/K
    target Q/K       := pure-T2V anchor Q/K
    source/target V  := current full-source V2V V

Consequently the anchor can alter the spatiotemporal attention routing without
copying its value/content stream.  This module does not define an endpoint
loss, a 32-D motion statistic, training, SGA, or ANC.  It is the block-level
transport seam consumed by an outer source-state SGA/ANC sampler.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Iterator, Optional, Sequence

import torch

import source_kv_replay as replay_runtime


METHOD = "bernini-online-pure-t2v-anchor-qk-transport-v3"
QK_ONLY_ZERO_RMS_BACKWARD_POLICY = (
    "exact_forward_zero_rms_zero_subgradient_v1"
)
BLOCK_COUNT = 30
DEFAULT_BLOCKS = tuple(range(16))
CAPTURE = "anchor_capture"
REPLAY = "target_replay"
MODES = (CAPTURE, REPLAY)
HARD_QK = "hard_qk"
HARD_K = "hard_k"
DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_ALL = (
    "dual_hard_q_early_source_kv_late_all"
)
DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_STATIC75 = (
    "dual_hard_q_early_source_kv_late_static75"
)
DUAL_SOURCE_KV_TRANSPORTS = (
    DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_ALL,
    DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_STATIC75,
)
DUAL_EARLY_ANCHOR_BLOCKS = tuple(range(4, 10))
DUAL_LATE_SOURCE_KV_BLOCKS = tuple(range(18, 30))
DUAL_DYNAMIC_TARGET_KEEP_FRACTION = 0.25
TEMPORAL_RESIDUAL_QK = "temporal_residual_qk"
TEMPORAL_RESIDUAL_K = "temporal_residual_k"
TEMPORAL_RESIDUAL_QKV = "temporal_residual_qkv"
TEMPORAL_RESIDUAL_V = "temporal_residual_v"
TEMPORAL_RESIDUAL_ATTN_OUTPUT = "temporal_residual_attn_output"
ACTION_NOOP_OBSERVER_ATTN_OUTPUT = "action_noop_observer_attn_output"
TEMPORAL_CORRESPONDENCE_ATTN_OUTPUT = "temporal_correspondence_attn_output"
TEMPORAL_CONTRAST_QK = "temporal_contrast_qk"
TEMPORAL_CONTRAST_ATTN_OUTPUT = "temporal_contrast_attn_output"
TEMPORAL_CORRESPONDENCE_CONTRAST_QK = "temporal_correspondence_contrast_qk"
TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK = (
    "temporal_correspondence_hard_contrast_qk"
)
TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT = (
    "temporal_correspondence_contrast_attn_output"
)
TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT = (
    "temporal_correspondence_hard_contrast_attn_output"
)
TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK = (
    "temporal_mutual_correspondence_contrast_qk"
)
TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT = (
    "temporal_mutual_correspondence_contrast_attn_output"
)
HARD_PHASE_MEAN_CONTRAST_QK = "hard_phase_mean_contrast_qk"
HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT = (
    "hard_phase_mean_contrast_attn_output"
)
HARD_PREROPE_PHASE_MEAN_CONTRAST_QK = (
    "hard_prerope_phase_mean_contrast_qk"
)
TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT = (
    "temporal_kernel_contrast_attn_output"
)
TARGET_OWNED_TEMPORAL_KERNEL_ATTN_OUTPUT_V14R2 = (
    "self_target_owned_temporal_kernel_attn_output_v14r2"
)
TARGET_OWNED_ACTIVITY_KERNEL_TOP10_ATTN_OUTPUT_V14R2 = (
    "self_target_owned_activity_kernel10_attn_output_v14r2"
)
TARGET_OWNED_ACTIVITY_KERNEL_TOP25_ATTN_OUTPUT_V14R2 = (
    "self_target_owned_activity_kernel25_attn_output_v14r2"
)
TARGET_GATED_HARD_KERNEL_TOP10_ATTN_OUTPUT = (
    "target_gated_hard_kernel_top10_attn_output"
)
TARGET_GATED_HARD_KERNEL_TOP25_ATTN_OUTPUT = (
    "target_gated_hard_kernel_top25_attn_output"
)
CORRESPONDENCE_GATED_HARD_KERNEL_TOP25_ATTN_OUTPUT = (
    "correspondence_gated_hard_kernel_top25_attn_output"
)
EVENT01_ROLE_GRAPH_HARD_ATTN_OUTPUT = (
    "event01_role_graph_hard_attn_output"
)
EVENT01_ROLE_GRAPH_LOGIT_BIAS_ATTN_OUTPUT = (
    "event01_role_graph_logit_bias_attn_output"
)
EVENT01_DYNAMIC_ROLE_GRAPH_LOGIT_BIAS_ATTN_OUTPUT = (
    "event01_dynamic_role_graph_logit_bias_attn_output"
)
EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_OBJECT_ATTN_OUTPUT = (
    "event01_dynamic_role_graph_source_object_attn_output"
)
EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT = (
    "event01_dynamic_source_object_value_attn_output"
)
EVENT01_DYNAMIC_SOURCE_OBJECT_OUTPUT_ATTN_OUTPUT = (
    "event01_dynamic_source_object_output_attn_output"
)
EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_MOVE_ATTN_OUTPUT = (
    "event01_dynamic_role_graph_source_patch_move_attn_output"
)
EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_ATTN_OUTPUT = (
    "event01_dynamic_role_graph_source_patch_value_attn_output"
)
EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_SIDE_ALIGNED_ATTN_OUTPUT = (
    "event01_dynamic_role_graph_source_patch_value_side_aligned_attn_output"
)
EVENT01_ROLE_GRAPH_TRANSPORTS = (
    EVENT01_ROLE_GRAPH_HARD_ATTN_OUTPUT,
    EVENT01_ROLE_GRAPH_LOGIT_BIAS_ATTN_OUTPUT,
    EVENT01_DYNAMIC_ROLE_GRAPH_LOGIT_BIAS_ATTN_OUTPUT,
    EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_OBJECT_ATTN_OUTPUT,
    EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT,
    EVENT01_DYNAMIC_SOURCE_OBJECT_OUTPUT_ATTN_OUTPUT,
    EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_MOVE_ATTN_OUTPUT,
    EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_ATTN_OUTPUT,
    EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_SIDE_ALIGNED_ATTN_OUTPUT,
)
TARGET_GATED_HARD_KERNEL_TRANSPORTS = (
    TARGET_GATED_HARD_KERNEL_TOP10_ATTN_OUTPUT,
    TARGET_GATED_HARD_KERNEL_TOP25_ATTN_OUTPUT,
    CORRESPONDENCE_GATED_HARD_KERNEL_TOP25_ATTN_OUTPUT,
)
TARGET_OWNED_QK_TRANSPORTS_V14R2 = (
    TARGET_OWNED_TEMPORAL_KERNEL_ATTN_OUTPUT_V14R2,
    TARGET_OWNED_ACTIVITY_KERNEL_TOP10_ATTN_OUTPUT_V14R2,
    TARGET_OWNED_ACTIVITY_KERNEL_TOP25_ATTN_OUTPUT_V14R2,
)
TRANSPORTS = (
    HARD_QK,
    HARD_K,
    *DUAL_SOURCE_KV_TRANSPORTS,
    TEMPORAL_RESIDUAL_QK,
    TEMPORAL_RESIDUAL_K,
    TEMPORAL_RESIDUAL_QKV,
    TEMPORAL_RESIDUAL_V,
    TEMPORAL_RESIDUAL_ATTN_OUTPUT,
    ACTION_NOOP_OBSERVER_ATTN_OUTPUT,
    TEMPORAL_CORRESPONDENCE_ATTN_OUTPUT,
    TEMPORAL_CONTRAST_QK,
    TEMPORAL_CONTRAST_ATTN_OUTPUT,
    TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
    TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
    TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
    TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
    TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
    TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
    HARD_PHASE_MEAN_CONTRAST_QK,
    HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
    HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
    TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
    *TARGET_OWNED_QK_TRANSPORTS_V14R2,
    *TARGET_GATED_HARD_KERNEL_TRANSPORTS,
    *EVENT01_ROLE_GRAPH_TRANSPORTS,
)
PAIRED_SUFFIX = "paired_source_target_suffix"
FULL_SEQUENCE = "full_target_sequence"
REPLAY_SCOPES = (PAIRED_SUFFIX, FULL_SEQUENCE)
LATENT_PHASES = 21
TEMPORAL_RESIDUAL_KEEP_FRACTION = 0.25
CORRESPONDENCE_ANCHOR_STRIDE = 4
ACTION_SLOT = "action"
NOOP_SLOT = "noop"
SLOTS = (ACTION_SLOT, NOOP_SLOT)
EVENT01_ROLE_PROPOSALS = 5
EVENT01_SPATIAL_HEIGHT = 36
EVENT01_SPATIAL_WIDTH = 26
EVENT01_ANCHOR_ACTOR_XY = (19.0, 15.5)
EVENT01_ANCHOR_OBJECT_XY = (7.0, 21.5)
EVENT01_SOURCE_ACTOR_XY = (12.0, 25.0)
EVENT01_SOURCE_OBJECT_PROPOSALS_XY = (
    (9.5, 29.0),
    (17.5, 30.5),
    (20.5, 28.5),
    (11.0, 32.0),
    (7.0, 34.5),
)
EVENT01_ANCHOR_ACTOR_TRAJECTORY_XY = (
    (18.5, 15.5), (18.0, 16.0), (17.5, 18.0), (17.0, 19.0),
    (16.5, 20.0), (16.5, 20.0), (16.5, 20.0), (16.5, 20.0),
    (16.5, 20.0), (16.5, 19.5), (16.0, 19.0), (16.0, 18.0),
    (16.0, 17.0), (16.0, 16.0), (16.0, 16.0), (16.0, 16.0),
    (16.0, 16.0), (16.0, 16.0), (16.0, 16.0), (16.0, 16.0),
    (16.0, 16.0),
)
EVENT01_ANCHOR_OBJECT_TRAJECTORY_XY = (
    (7.0, 21.5), (7.0, 21.5), (7.0, 21.5), (7.0, 21.5),
    (7.0, 21.5), (7.5, 21.5), (8.0, 21.0), (8.5, 20.5),
    (9.0, 20.0), (9.5, 19.5), (10.0, 19.0), (10.5, 18.5),
    (10.5, 18.0), (10.5, 18.0), (10.5, 18.0), (10.5, 18.0),
    (10.5, 18.0), (10.5, 18.0), (10.5, 18.0), (10.5, 18.0),
    (10.5, 18.0),
)
EVENT01_TARGET_ACTOR_TRAJECTORY_XY = (
    (12.0, 25.0), (12.2, 25.0), (12.5, 25.5), (12.5, 26.0),
    (12.5, 27.0), (12.5, 27.0), (12.5, 27.0), (12.5, 26.0),
    (12.5, 25.0), (12.5, 24.5), (12.5, 24.0), (12.5, 24.0),
    (12.5, 24.0), (12.5, 24.0), (12.5, 24.0), (12.5, 24.0),
    (12.5, 24.0), (12.5, 24.0), (12.5, 24.0), (12.5, 24.0),
    (12.5, 24.0),
)
EVENT01_TARGET_OBJECT_LIFT_PROGRESS = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.30, 0.50, 0.70,
    0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
)


class AnchorQKTransportError(RuntimeError):
    """Raised rather than applying an ambiguous cross-mode attention route."""


def _exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise AnchorQKTransportError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AnchorQKTransportError(f"{label} must be an integer") from error
    if result != value:
        raise AnchorQKTransportError(f"{label} must be exact")
    return result


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise AnchorQKTransportError("attention value must be a torch.Tensor")
    return tuple(int(item) for item in value.shape)


def _lengths(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AnchorQKTransportError(f"{label} must be a sequence")
    return tuple(_exact_int(item, label=f"{label} item") for item in value)


@dataclass(frozen=True)
class AnchorQKEntry:
    hidden_state: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_output: torch.Tensor


@dataclass(frozen=True)
class AnchorQKOnlyEntry:
    """The complete formal donor ABI: post-RoPE Q and K, and nothing else."""

    query: torch.Tensor
    key: torch.Tensor


class AnchorQKCacheBank:
    """One-use, step/candidate/rank-bound post-RoPE anchor Q/K cache."""

    def __init__(self, selected_block_indices: Sequence[int] = DEFAULT_BLOCKS) -> None:
        indices = tuple(
            _exact_int(item, label="selected block index")
            for item in selected_block_indices
        )
        if (
            not indices
            or indices != tuple(sorted(set(indices)))
            or any(item < 0 or item >= BLOCK_COUNT for item in indices)
        ):
            raise AnchorQKTransportError(
                "selected blocks must be a non-empty increasing subset of 0..29"
            )
        self.selected_block_indices = indices
        self._entries: dict[
            tuple[int, int, int, int, str],
            tuple[AnchorQKEntry | AnchorQKOnlyEntry, int, int],
        ] = {}
        self.capture_count = 0
        self.replay_count = 0
        self.qk_only_capture_count = 0
        self.qk_only_replay_count = 0
        self.source_kv_late_all_replay_count = 0
        self.source_kv_late_static75_replay_count = 0

    @staticmethod
    def _key(
        invocation: "AnchorQKInvocation", block_index: int
    ) -> tuple[int, int, int, int, str]:
        return (
            invocation.step_index,
            invocation.candidate_index,
            invocation.rank,
            block_index,
            invocation.slot,
        )

    def capture(
        self,
        *,
        invocation: "AnchorQKInvocation",
        block_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_output: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> None:
        if invocation.transport in TARGET_OWNED_QK_TRANSPORTS_V14R2:
            raise AnchorQKTransportError(
                "QK-only transports must use capture_qk_only; their cache ABI "
                "cannot accept donor V/hidden/output"
            )
        if invocation.mode != CAPTURE:
            raise AnchorQKTransportError("only an anchor-capture invocation may write Q/K")
        cache_key = self._key(invocation, block_index)
        if cache_key in self._entries:
            raise AnchorQKTransportError("anchor Q/K cache entry already exists")
        if (
            _shape(query) != _shape(key)
            or _shape(query) != _shape(value)
            or _shape(query) != _shape(attention_output)
            or query.ndim != 4
            or int(query.shape[0]) != 1
            or int(query.shape[1]) <= 0
            or hidden_state.ndim != 3
            or int(hidden_state.shape[0]) != 1
            or (
                invocation.transport == HARD_PREROPE_PHASE_MEAN_CONTRAST_QK
                and int(hidden_state.shape[1]) != int(query.shape[1])
            )
        ):
            raise AnchorQKTransportError("captured anchor Q/K geometry differs")
        if (
            query.dtype != key.dtype
            or query.dtype != value.dtype
            or query.dtype != attention_output.dtype
            or query.device != key.device
            or query.device != value.device
            or query.device != attention_output.device
            or query.dtype != hidden_state.dtype
            or query.device != hidden_state.device
        ):
            raise AnchorQKTransportError("captured anchor Q/K dtype/device differs")
        if any(
            not bool(torch.isfinite(item).all())
            for item in (hidden_state, query, key, value, attention_output)
        ):
            raise AnchorQKTransportError("captured anchor Q/K/V is non-finite")
        self._entries[cache_key] = (
            AnchorQKEntry(
                hidden_state=hidden_state.detach().clone(),
                query=query.detach().clone(),
                key=key.detach().clone(),
                value=value.detach().clone(),
                attention_output=attention_output.detach().clone(),
            ),
            invocation.replay_uses,
            invocation.replay_uses,
        )
        self.capture_count += 1

    def capture_qk_only(
        self,
        *,
        invocation: "AnchorQKInvocation",
        block_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> None:
        """Capture only donor Q/K for the v14r2 formal route.

        The signature intentionally has no V, hidden-state, attention-output,
        latent, RGB, or spatial-index argument.  This makes donor-content use
        impossible at the cache boundary instead of relying on convention.
        """

        if invocation.mode != CAPTURE:
            raise AnchorQKTransportError(
                "only an anchor-capture invocation may write QK-only entries"
            )
        if invocation.transport not in TARGET_OWNED_QK_TRANSPORTS_V14R2:
            raise AnchorQKTransportError(
                "capture_qk_only requires an explicit QK-only transport"
            )
        cache_key = self._key(invocation, block_index)
        if cache_key in self._entries:
            raise AnchorQKTransportError("anchor QK-only cache entry already exists")
        if (
            query.ndim != 4
            or _shape(query) != _shape(key)
            or int(query.shape[0]) != 1
            or int(query.shape[1]) <= 0
        ):
            raise AnchorQKTransportError("captured QK-only donor geometry differs")
        if query.dtype != key.dtype or query.device != key.device:
            raise AnchorQKTransportError("captured QK-only donor dtype/device differs")
        if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(key).all()):
            raise AnchorQKTransportError("captured QK-only donor is non-finite")
        self._entries[cache_key] = (
            AnchorQKOnlyEntry(
                query=query.detach().clone(),
                key=key.detach().clone(),
            ),
            invocation.replay_uses,
            invocation.replay_uses,
        )
        self.capture_count += 1
        self.qk_only_capture_count += 1

    def consume(
        self,
        *,
        invocation: "AnchorQKInvocation",
        block_index: int,
        current_query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        current_hidden_state: torch.Tensor,
    ) -> AnchorQKEntry:
        if invocation.transport in TARGET_OWNED_QK_TRANSPORTS_V14R2:
            raise AnchorQKTransportError(
                "QK-only transports must use consume_qk_only; legacy donor "
                "V/hidden/output access is forbidden"
            )
        if invocation.mode != REPLAY:
            raise AnchorQKTransportError("only a target-replay invocation may consume Q/K")
        cache_key = self._key(invocation, block_index)
        stored = self._entries.get(cache_key)
        if stored is None:
            raise AnchorQKTransportError("matching anchor Q/K cache entry is absent")
        entry, remaining, total = stored
        if invocation.replay_uses != total:
            raise AnchorQKTransportError("anchor Q/K replay-use contract differs")
        if (
            current_query.ndim != 4
            or _shape(current_query) != _shape(current_key)
            or _shape(current_query) != _shape(current_value)
            or current_hidden_state.ndim != 3
            or int(current_hidden_state.shape[0]) != 1
            or (
                invocation.transport == HARD_PREROPE_PHASE_MEAN_CONTRAST_QK
                and int(current_hidden_state.shape[1])
                != int(current_query.shape[1])
            )
        ):
            raise AnchorQKTransportError("current target replay Q/K/V geometry differs")
        current_tokens = int(current_query.shape[1])
        if invocation.replay_scope == PAIRED_SUFFIX:
            if current_tokens % 2:
                raise AnchorQKTransportError(
                    "paired target replay is not one equal source/target pair"
                )
            anchor_tokens = current_tokens // 2
        else:
            anchor_tokens = current_tokens
        expected = (
            int(current_query.shape[0]),
            anchor_tokens,
            int(current_query.shape[2]),
            int(current_query.shape[3]),
        )
        if any(
            _shape(item) != expected
            for item in (
                entry.query,
                entry.key,
                entry.value,
                entry.attention_output,
            )
        ):
            raise AnchorQKTransportError("anchor/current projected head geometry differs")
        if invocation.transport == HARD_PREROPE_PHASE_MEAN_CONTRAST_QK:
            expected_hidden = (
                int(current_hidden_state.shape[0]),
                anchor_tokens,
                int(current_hidden_state.shape[2]),
            )
            if _shape(entry.hidden_state) != expected_hidden:
                raise AnchorQKTransportError(
                    "anchor/current full hidden-state geometry differs"
                )
        if (
            entry.query.dtype != current_query.dtype
            or entry.key.dtype != current_key.dtype
            or entry.value.dtype != current_value.dtype
            or entry.attention_output.dtype != current_value.dtype
            or entry.query.device != current_query.device
            or entry.key.device != current_key.device
            or entry.value.device != current_value.device
            or entry.attention_output.device != current_value.device
            or entry.hidden_state.dtype != current_hidden_state.dtype
            or entry.hidden_state.device != current_hidden_state.device
        ):
            raise AnchorQKTransportError("anchor/current projected dtype/device differs")
        if remaining == 1:
            self._entries.pop(cache_key)
        else:
            self._entries[cache_key] = (entry, remaining - 1, total)
        self.replay_count += 1
        return entry

    def consume_qk_only(
        self,
        *,
        invocation: "AnchorQKInvocation",
        block_index: int,
        current_query: torch.Tensor,
        current_key: torch.Tensor,
    ) -> AnchorQKOnlyEntry:
        """Consume a Q/K-only donor entry against target-owned Q/K geometry."""

        if invocation.mode != REPLAY:
            raise AnchorQKTransportError(
                "only a target-replay invocation may consume QK-only entries"
            )
        if invocation.transport not in TARGET_OWNED_QK_TRANSPORTS_V14R2:
            raise AnchorQKTransportError(
                "consume_qk_only requires an explicit QK-only transport"
            )
        cache_key = self._key(invocation, block_index)
        stored = self._entries.get(cache_key)
        if stored is None:
            raise AnchorQKTransportError("matching QK-only donor entry is absent")
        entry, remaining, total = stored
        if not isinstance(entry, AnchorQKOnlyEntry):
            raise AnchorQKTransportError(
                "QK-only replay encountered a legacy content-bearing cache entry"
            )
        if invocation.replay_uses != total:
            raise AnchorQKTransportError("QK-only replay-use contract differs")
        if (
            current_query.ndim != 4
            or _shape(current_query) != _shape(current_key)
            or int(current_query.shape[0]) != 1
        ):
            raise AnchorQKTransportError("current target QK-only geometry differs")
        current_tokens = int(current_query.shape[1])
        if invocation.replay_scope == PAIRED_SUFFIX:
            if current_tokens % 2:
                raise AnchorQKTransportError(
                    "paired QK-only replay is not one equal source/target pair"
                )
            donor_tokens = current_tokens // 2
        else:
            donor_tokens = current_tokens
        expected = (
            int(current_query.shape[0]),
            donor_tokens,
            int(current_query.shape[2]),
            int(current_query.shape[3]),
        )
        if _shape(entry.query) != expected or _shape(entry.key) != expected:
            raise AnchorQKTransportError("donor/target QK-only head geometry differs")
        if (
            entry.query.dtype != current_query.dtype
            or entry.key.dtype != current_key.dtype
            or entry.query.device != current_query.device
            or entry.key.device != current_key.device
        ):
            raise AnchorQKTransportError("donor/target QK-only dtype/device differs")
        if remaining == 1:
            self._entries.pop(cache_key)
        else:
            self._entries[cache_key] = (entry, remaining - 1, total)
        self.replay_count += 1
        self.qk_only_replay_count += 1
        return entry

    def assert_empty(self) -> None:
        if self._entries:
            raise AnchorQKTransportError("unconsumed anchor Q/K entries remain")

    def receipt(self) -> dict[str, Any]:
        return {
            "method": METHOD,
            "selected_block_indices": list(self.selected_block_indices),
            "capture_count": self.capture_count,
            "replay_count": self.replay_count,
            "qk_only_capture_count": self.qk_only_capture_count,
            "qk_only_replay_count": self.qk_only_replay_count,
            "qk_only_cached_fields": ["query", "key"],
            "qk_only_forbidden_cached_fields": [
                "value",
                "hidden_state",
                "attention_output",
                "rgb",
                "latent",
                "absolute_spatial_coordinate",
            ],
            "qk_only_zero_rms_backward_policy": (
                QK_ONLY_ZERO_RMS_BACKWARD_POLICY
            ),
            "source_kv_late_all_replay_count": (
                self.source_kv_late_all_replay_count
            ),
            "source_kv_late_static75_replay_count": (
                self.source_kv_late_static75_replay_count
            ),
            "pending_entries": len(self._entries),
        }


@dataclass(frozen=True)
class AnchorQKInvocation:
    mode: str
    cache_bank: AnchorQKCacheBank
    step_index: int
    candidate_index: int
    rank: int
    ulysses_size: int
    transport: str = HARD_QK
    transport_strength: float = 1.0
    replay_uses: int = 1
    replay_scope: str = PAIRED_SUFFIX
    slot: str = ACTION_SLOT
    role_proposal_index: int = 0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise AnchorQKTransportError(f"mode must be one of {MODES}")
        if not isinstance(self.cache_bank, AnchorQKCacheBank):
            raise AnchorQKTransportError("cache_bank has the wrong type")
        for label in ("step_index", "candidate_index", "rank", "ulysses_size"):
            value = _exact_int(getattr(self, label), label=label)
            if value < 0 or (label == "ulysses_size" and value < 1):
                raise AnchorQKTransportError(f"{label} is outside its domain")
        if self.rank >= self.ulysses_size:
            raise AnchorQKTransportError("rank must be smaller than ulysses_size")
        if isinstance(self.replay_uses, bool) or self.replay_uses not in (1, 2):
            raise AnchorQKTransportError("replay_uses must be one or two")
        if self.replay_scope not in REPLAY_SCOPES:
            raise AnchorQKTransportError(
                f"replay_scope must be one of {REPLAY_SCOPES}"
            )
        if self.transport not in TRANSPORTS:
            raise AnchorQKTransportError(f"transport must be one of {TRANSPORTS}")
        if self.slot not in SLOTS:
            raise AnchorQKTransportError(f"slot must be one of {SLOTS}")
        role_proposal_index = _exact_int(
            self.role_proposal_index, label="role_proposal_index"
        )
        if not 0 <= role_proposal_index < EVENT01_ROLE_PROPOSALS:
            raise AnchorQKTransportError("role proposal index is outside 0..4")
        if (
            isinstance(self.transport_strength, bool)
            or not isinstance(self.transport_strength, (int, float))
            or not math.isfinite(float(self.transport_strength))
            or not 0.0 < float(self.transport_strength) <= 1.0
        ):
            raise AnchorQKTransportError("transport_strength must be in (0,1]")

    @property
    def branch_tag(self) -> str:
        return f"{self.mode}:{self.slot}:s{self.step_index}:c{self.candidate_index}"


_CURRENT: contextvars.ContextVar[Optional[AnchorQKInvocation]] = contextvars.ContextVar(
    "anchor_qk_transport_invocation", default=None
)


@contextlib.contextmanager
def anchor_qk_invocation(invocation: AnchorQKInvocation) -> Iterator[None]:
    if _CURRENT.get() is not None:
        raise AnchorQKTransportError("anchor Q/K invocations may not nest")
    token = _CURRENT.set(invocation)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_anchor_qk_invocation() -> Optional[AnchorQKInvocation]:
    return _CURRENT.get()


def _sparse_frame0_temporal_residual(
    current: torch.Tensor,
    anchor: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    """Move only sparse temporal changes while retaining current frame-0 basis.

    Both tensors are one target sequence in frame-major packed-token order.
    The transported quantity is the difference between the anchor and current
    frame-0-relative trajectories.  Constant anchor appearance therefore
    cancels, and the result is exactly current at latent phase zero.
    """

    if _shape(current) != _shape(anchor) or int(current.shape[1]) % LATENT_PHASES:
        raise AnchorQKTransportError("temporal residual requires matched 21-phase Q/K")
    batch, tokens, heads, width = _shape(current)
    spatial = tokens // LATENT_PHASES
    current_phase = current.reshape(batch, LATENT_PHASES, spatial, heads, width)
    anchor_phase = anchor.reshape(batch, LATENT_PHASES, spatial, heads, width)
    current_delta = current_phase - current_phase[:, :1]
    anchor_delta = anchor_phase - anchor_phase[:, :1]
    route = anchor_delta - current_delta

    # Keep only the spatial quarter with the largest anchor-vs-current temporal
    # discrepancy in each phase.  The score is computed in fp32, while the
    # transported Q/K remains in its native attention dtype.
    score = route.float().square().mean(dim=(-1, -2))
    keep = max(1, math.ceil(spatial * TEMPORAL_RESIDUAL_KEEP_FRACTION))
    top = torch.topk(score, k=keep, dim=2, largest=True, sorted=False).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(2, top, True)
    mask = mask[..., None, None]
    routed = current_phase + float(strength) * route * mask.to(route.dtype)
    # An exact first-phase identity is the initial-state preservation gate.
    routed[:, 0].copy_(current_phase[:, 0])
    return routed.reshape_as(current)


def _sparse_frame0_additive_contrast(
    current: torch.Tensor,
    action: torch.Tensor,
    noop: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    """Add sparse action-minus-noop attention change, never absolute anchor state."""

    if (
        _shape(current) != _shape(action)
        or _shape(current) != _shape(noop)
        or int(current.shape[1]) % LATENT_PHASES
    ):
        raise AnchorQKTransportError(
            "attention contrast requires three matched 21-phase tensors"
        )
    batch, tokens, heads, width = _shape(current)
    spatial = tokens // LATENT_PHASES
    current_phase = current.reshape(batch, LATENT_PHASES, spatial, heads, width)
    contrast = (action - noop).reshape(
        batch, LATENT_PHASES, spatial, heads, width
    )
    route = contrast - contrast[:, :1]
    score = route.float().square().mean(dim=(-1, -2))
    keep = max(1, math.ceil(spatial * TEMPORAL_RESIDUAL_KEEP_FRACTION))
    top = torch.topk(score, k=keep, dim=2, largest=True, sorted=False).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(2, top, True)
    routed = current_phase + float(strength) * route * mask[..., None, None].to(
        route.dtype
    )
    routed[:, 0].copy_(current_phase[:, 0])
    return routed.reshape_as(current)


def _hard_phase_mean_temporal_contrast(
    current: torch.Tensor,
    action: torch.Tensor,
    noop: torch.Tensor,
) -> torch.Tensor:
    """Hard-replace the coordinate-free temporal component with anchor action.

    The pure-T2V actor and source actor need not share spatial coordinates.
    Spatial averaging removes that false correspondence while retaining every
    head/channel.  The target keeps its own spatial residual and value/content
    stream; only its per-phase spatial mean is replaced by the anchor's
    dynamic-minus-static temporal trajectory.  Phase zero remains exact.
    """

    if (
        _shape(current) != _shape(action)
        or _shape(current) != _shape(noop)
        or int(current.shape[1]) % LATENT_PHASES
    ):
        raise AnchorQKTransportError(
            "phase-mean contrast requires three matched 21-phase tensors"
        )
    batch, tokens, heads, width = _shape(current)
    spatial = tokens // LATENT_PHASES
    current_phase = current.reshape(batch, LATENT_PHASES, spatial, heads, width)
    action_phase = action.reshape(batch, LATENT_PHASES, spatial, heads, width)
    noop_phase = noop.reshape(batch, LATENT_PHASES, spatial, heads, width)

    current_mean = current_phase.mean(dim=2, keepdim=True)
    spatial_residual = current_phase - current_mean
    anchor_mean = (action_phase - noop_phase).mean(dim=2, keepdim=True)
    anchor_temporal = anchor_mean - anchor_mean[:, :1]
    routed = spatial_residual + current_mean[:, :1] + anchor_temporal
    routed[:, 0].copy_(current_phase[:, 0])
    return routed.reshape_as(current)


def _temporal_attention_kernel_contrast_output(
    current_output: torch.Tensor,
    current_value: torch.Tensor,
    action_query: torch.Tensor,
    action_key: torch.Tensor,
    action_value: torch.Tensor,
    noop_query: torch.Tensor,
    noop_key: torch.Tensor,
    noop_value: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    """Transfer only dynamic-minus-static temporal attention topology.

    Dynamic and static pure-T2V anchor forwards share the same actor, noise,
    prompt and token coordinates.  Their per-head temporal attention matrices
    can therefore be contrasted without matching the anchor actor to the
    source actor.  Spatial positions are used only to estimate that global
    temporal operator: the quarter with the largest dynamic/static value
    response is selected and averaged.  The resulting ``T x T`` kernel is
    applied independently at every target spatial position to the target's own
    value stream.

    No anchor value, pixel, feature vector, or spatial coordinate enters the
    returned target output.  Each attention row in the dynamic/static
    difference sums to zero, so temporally constant target content cancels.
    Latent phase zero is kept bit-exact.
    """

    tensors = (
        current_output,
        current_value,
        action_query,
        action_key,
        action_value,
        noop_query,
        noop_key,
        noop_value,
    )
    if (
        any(item.ndim != 4 for item in tensors)
        or any(_shape(item) != _shape(current_output) for item in tensors[1:])
        or int(current_output.shape[1]) % LATENT_PHASES
    ):
        raise AnchorQKTransportError(
            "temporal-kernel contrast requires eight matched 21-phase tensors"
        )
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 < float(strength) <= 1.0
    ):
        raise AnchorQKTransportError("temporal-kernel strength must be in (0,1]")

    batch, tokens, heads, width = _shape(current_output)
    spatial = tokens // LATENT_PHASES

    def phase(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, LATENT_PHASES, spatial, heads, width)

    current_output_phase = phase(current_output)
    current_value_phase = phase(current_value)
    action_query_phase = phase(action_query)
    action_key_phase = phase(action_key)
    action_value_phase = phase(action_value)
    noop_query_phase = phase(noop_query)
    noop_key_phase = phase(noop_key)
    noop_value_phase = phase(noop_value)

    activity = (
        (action_value_phase - noop_value_phase)
        .float()
        .square()
        .mean(dim=(1, 3, 4))
    )
    keep = max(1, math.ceil(spatial * TEMPORAL_RESIDUAL_KEEP_FRACTION))
    active = torch.topk(
        activity, k=keep, dim=1, largest=True, sorted=False
    ).indices
    gather_index = active[:, None, :, None, None].expand(
        batch, LATENT_PHASES, keep, heads, width
    )

    def selected(tensor: torch.Tensor) -> torch.Tensor:
        return torch.gather(tensor, 2, gather_index).float()

    action_query_active = selected(action_query_phase)
    action_key_active = selected(action_key_phase)
    noop_query_active = selected(noop_query_phase)
    noop_key_active = selected(noop_key_phase)
    scale = math.sqrt(width)
    # [B,K,H,T_query,T_key].  The anchor coordinates agree only within its
    # own dynamic/static pair; K is averaged away before target application.
    action_logits = torch.einsum(
        "btkhd,bukhd->bkhtu", action_query_active, action_key_active
    ) / scale
    noop_logits = torch.einsum(
        "btkhd,bukhd->bkhtu", noop_query_active, noop_key_active
    ) / scale
    action_kernel = torch.softmax(action_logits, dim=-1).mean(dim=1)
    noop_kernel = torch.softmax(noop_logits, dim=-1).mean(dim=1)
    kernel_contrast = action_kernel - noop_kernel

    # [B,H,T,U] x [B,U,S,H,D] -> [B,T,S,H,D].  Only current target V is used.
    route = torch.einsum(
        "bhtu,bushd->btshd", kernel_contrast, current_value_phase.float()
    ).to(current_output.dtype)
    route[:, 0].zero_()
    routed = current_output_phase + float(strength) * route
    routed[:, 0].copy_(current_output_phase[:, 0])
    return routed.reshape_as(current_output)


def _qk_only_anchor_temporal_kernel_components(
    action_query: torch.Tensor,
    action_key: torch.Tensor,
    noop_query: torch.Tensor,
    noop_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a spatial-permutation-invariant action-minus-noop T-by-T kernel.

    The donor contributes post-RoPE Q/K only.  A donor site's weight is its
    Q/K action-minus-noop energy; both the site identity and absolute position
    are then integrated out.  Thus the returned ``[B,H,T,T]`` object carries
    temporal attention topology, never an anchor coordinate or content value.
    """

    donor = (action_query, action_key, noop_query, noop_key)
    if (
        any(item.ndim != 4 for item in donor)
        or any(_shape(item) != _shape(action_query) for item in donor[1:])
        or int(action_query.shape[1]) % LATENT_PHASES
    ):
        raise AnchorQKTransportError(
            "QK-only temporal kernel requires four matched 21-phase Q/K tensors"
        )
    batch, tokens, heads, width = _shape(action_query)
    spatial = tokens // LATENT_PHASES

    def phase(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(
            batch, LATENT_PHASES, spatial, heads, width
        ).float()

    action_q_raw = phase(action_query)
    action_k_raw = phase(action_key)
    noop_q_raw = phase(noop_query)
    noop_k_raw = phase(noop_key)

    def temporal_contrast(tensor: torch.Tensor) -> torch.Tensor:
        """Remove all time-constant and phase-0-only donor components.

        The explicit phase-0 subtraction is the audited Q/K analogue of the
        requested action/noop baseline.  Centering the remaining dynamic
        phases then makes a phase-0-only discrepancy exactly null instead of
        turning it into a constant offset on every later phase.  Phase 0 is
        written back as exact zero.  This is coordinate-free and linear.
        """

        relative = tensor - tensor[:, :1]
        dynamic = relative[:, 1:]
        dynamic = dynamic - dynamic.mean(dim=1, keepdim=True)
        return torch.cat((torch.zeros_like(relative[:, :1]), dynamic), dim=1)

    # These are Q/K only.  In particular, no donor value, hidden state,
    # attention output, RGB, latent, or absolute spatial coordinate can enter
    # either the spatial support or the temporal kernel.
    action_q = temporal_contrast(action_q_raw)
    action_k = temporal_contrast(action_k_raw)
    noop_q = temporal_contrast(noop_q_raw)
    noop_k = temporal_contrast(noop_k_raw)
    dq = temporal_contrast(action_q_raw - noop_q_raw)
    dk = temporal_contrast(action_k_raw - noop_k_raw)
    support = (dq.square() + dk.square()).mean(dim=(1, 3, 4))
    support_sum = support.sum(dim=1, keepdim=True)
    # A zero contrast must produce a zero route.  A uniform fallback would
    # re-enable phase-0/static caption differences through the kernel.
    support_weight = torch.where(
        support_sum > 1.0e-12,
        support / support_sum.clamp_min(1.0e-12),
        torch.zeros_like(support),
    )
    scale = math.sqrt(width)
    # [B,S,H,T,U]; S is donor-local and disappears in the weighted integral.
    action_logits = torch.einsum(
        "btshd,bushd->bshtu", action_q, action_k
    ) / scale
    noop_logits = torch.einsum(
        "btshd,bushd->bshtu", noop_q, noop_k
    ) / scale
    site_kernel_contrast = (
        torch.softmax(action_logits, dim=-1)
        - torch.softmax(noop_logits, dim=-1)
    )
    # Remove the donor caption/static phase's query-row component before the
    # donor spatial axis is integrated out.  This is a temporal-topology DC
    # removal, not a pixel/velocity or representation-space target.
    site_kernel_contrast = (
        site_kernel_contrast - site_kernel_contrast[..., :1, :]
    )
    kernel = torch.einsum("bs,bshtu->bhtu", support_weight, site_kernel_contrast)
    return kernel, support_weight


def _qk_only_anchor_temporal_kernel(
    action_query: torch.Tensor,
    action_key: torch.Tensor,
    noop_query: torch.Tensor,
    noop_key: torch.Tensor,
) -> torch.Tensor:
    """Return only the content-free temporal kernel used by the route."""

    kernel, _support_weight = _qk_only_anchor_temporal_kernel_components(
        action_query, action_key, noop_query, noop_key
    )
    return kernel


def _qk_only_temporal_kernel_contrast_output(
    current_output: torch.Tensor,
    current_value: torch.Tensor,
    action_query: torch.Tensor,
    action_key: torch.Tensor,
    noop_query: torch.Tensor,
    noop_key: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    """Apply a donor-QK temporal graph to target-owned V at every target site."""

    if (
        current_output.ndim != 4
        or _shape(current_output) != _shape(current_value)
        or int(current_output.shape[1]) % LATENT_PHASES
    ):
        raise AnchorQKTransportError(
            "QK-only temporal output requires matched target output/V geometry"
        )
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 < float(strength) <= 1.0
    ):
        raise AnchorQKTransportError("QK-only temporal strength must be in (0,1]")
    batch, tokens, heads, width = _shape(current_output)
    if any(
        int(item.shape[0]) != batch
        or int(item.shape[2]) != heads
        or int(item.shape[3]) != width
        for item in (action_query, action_key, noop_query, noop_key)
    ):
        raise AnchorQKTransportError(
            "donor Q/K and target output head geometry differs"
        )
    spatial = tokens // LATENT_PHASES
    current_output_phase = current_output.reshape(
        batch, LATENT_PHASES, spatial, heads, width
    )
    current_value_phase = current_value.reshape(
        batch, LATENT_PHASES, spatial, heads, width
    )
    kernel = _qk_only_anchor_temporal_kernel(
        action_query, action_key, noop_query, noop_key
    )
    route = torch.einsum(
        "bhtu,bushd->btshd", kernel, current_value_phase.float()
    ).to(current_output.dtype)
    route[:, 0].zero_()
    routed = current_output_phase + float(strength) * route
    routed[:, 0].copy_(current_output_phase[:, 0])
    return routed.reshape_as(current_output)


def _exact_forward_zero_subgradient_rms(
    tensor: torch.Tensor,
    *,
    dim: tuple[int, ...],
    keepdim: bool,
) -> torch.Tensor:
    """Return ordinary RMS values with a finite zero subgradient at RMS zero.

    ``sqrt(mean(x**2))`` has the correct forward value at ``x == 0`` but its
    raw autograd path evaluates a singular square-root derivative.  A later
    hard ``where`` cannot repair the resulting ``0 * inf -> NaN``.  Replacing
    the square-root input by one only on exact-zero rows keeps every forward
    value bit-exact while selecting the natural zero subgradient there.  NaN
    and nonzero inputs are deliberately not masked or clamped.
    """

    mean_square = tensor.square().mean(dim=dim, keepdim=keepdim)
    exact_zero = mean_square.eq(0)
    safe_mean_square = torch.where(
        exact_zero,
        torch.ones_like(mean_square),
        mean_square,
    )
    safe_root = safe_mean_square.sqrt()
    return torch.where(exact_zero, torch.zeros_like(safe_root), safe_root)


def _qk_only_target_gated_hard_temporal_kernel_contrast_output(
    current_output: torch.Tensor,
    current_query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    action_query: torch.Tensor,
    action_key: torch.Tensor,
    noop_query: torch.Tensor,
    noop_key: torch.Tensor,
    *,
    strength: float,
    target_keep_fraction: float,
) -> torch.Tensor:
    """Use target Q/K activity to place a donor-QK temporal graph.

    The anchor graph is spatially integrated before it meets the target.  The
    hard gate is computed solely from each target site's own frame-0-relative
    Q/K activity, and the replacement transports only that target site's V.
    """

    target = (current_output, current_query, current_key, current_value)
    if (
        any(item.ndim != 4 for item in target)
        or any(_shape(item) != _shape(current_output) for item in target[1:])
        or int(current_output.shape[1]) % LATENT_PHASES
    ):
        raise AnchorQKTransportError(
            "QK-only target gate requires matched target output/Q/K/V geometry"
        )
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 < float(strength) <= 1.0
        or not 0.0 < float(target_keep_fraction) <= 1.0
    ):
        raise AnchorQKTransportError("QK-only target-gate controls are invalid")
    batch, tokens, heads, width = _shape(current_output)
    if any(
        int(item.shape[0]) != batch
        or int(item.shape[2]) != heads
        or int(item.shape[3]) != width
        for item in (action_query, action_key, noop_query, noop_key)
    ):
        raise AnchorQKTransportError(
            "donor Q/K and target gate head geometry differs"
        )
    spatial = tokens // LATENT_PHASES

    def target_phase(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, LATENT_PHASES, spatial, heads, width)

    output_phase = target_phase(current_output)
    query_phase = target_phase(current_query).float()
    key_phase = target_phase(current_key).float()
    value_phase = target_phase(current_value)
    kernel = _qk_only_anchor_temporal_kernel(
        action_query, action_key, noop_query, noop_key
    )
    route = torch.einsum(
        "bhtu,bushd->btshd", kernel, value_phase.float()
    )
    route = route - route[:, :1]

    target_activity = (
        (query_phase - query_phase[:, :1]).square()
        + (key_phase - key_phase[:, :1]).square()
    ).mean(dim=(1, 3, 4))
    target_keep = max(1, math.ceil(spatial * float(target_keep_fraction)))
    target_active = torch.topk(
        target_activity, k=target_keep, dim=1, largest=True, sorted=False
    ).indices
    target_gate = torch.zeros(
        batch, spatial, device=current_output.device, dtype=torch.bool
    )
    target_gate.scatter_(1, target_active, True)
    target_gate = target_gate[:, None, :, None, None]

    current_temporal = output_phase.float() - output_phase[:, :1].float()
    current_rms = _exact_forward_zero_subgradient_rms(
        current_temporal,
        dim=(1, 4),
        keepdim=True,
    )
    route_rms = _exact_forward_zero_subgradient_rms(
        route,
        dim=(1, 4),
        keepdim=True,
    )
    route_scaled = route * current_rms / route_rms.clamp_min(1.0e-6)
    replacement = output_phase[:, :1].float() + route_scaled
    replacement = torch.where(
        route_rms > 1.0e-6, replacement, output_phase.float()
    )
    hard_routed = torch.where(target_gate, replacement, output_phase.float())
    routed = (
        output_phase.float()
        + float(strength) * (hard_routed - output_phase.float())
    ).to(current_output.dtype)
    routed[:, 0].copy_(output_phase[:, 0])
    return routed.reshape_as(current_output)


def _target_gated_hard_temporal_kernel_contrast_output(
    current_output: torch.Tensor,
    current_value: torch.Tensor,
    action_query: torch.Tensor,
    action_key: torch.Tensor,
    action_value: torch.Tensor,
    noop_query: torch.Tensor,
    noop_key: torch.Tensor,
    noop_value: torch.Tensor,
    *,
    strength: float,
    target_keep_fraction: float,
) -> torch.Tensor:
    """Hard-route anchor temporal topology only at target-active sites.

    The dynamic/static anchor pair supplies only a per-head ``T x T`` kernel.
    That kernel acts on the target's own value stream.  The resulting temporal
    trajectory is RMS-matched to the target block's current trajectory and
    hard-replaces only the target spatial sites with the largest existing
    temporal activity.  Non-selected sites and latent phase zero are exact.
    """

    tensors = (
        current_output,
        current_value,
        action_query,
        action_key,
        action_value,
        noop_query,
        noop_key,
        noop_value,
    )
    if (
        any(item.ndim != 4 for item in tensors)
        or any(_shape(item) != _shape(current_output) for item in tensors[1:])
        or int(current_output.shape[1]) % LATENT_PHASES
    ):
        raise AnchorQKTransportError(
            "target-gated hard kernel requires eight matched 21-phase tensors"
        )
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 < float(strength) <= 1.0
        or not 0.0 < float(target_keep_fraction) <= 1.0
    ):
        raise AnchorQKTransportError("target-gated hard kernel controls are invalid")

    batch, tokens, heads, width = _shape(current_output)
    spatial = tokens // LATENT_PHASES

    def phase(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, LATENT_PHASES, spatial, heads, width)

    current_output_phase = phase(current_output)
    current_value_phase = phase(current_value)
    action_query_phase = phase(action_query)
    action_key_phase = phase(action_key)
    action_value_phase = phase(action_value)
    noop_query_phase = phase(noop_query)
    noop_key_phase = phase(noop_key)
    noop_value_phase = phase(noop_value)

    anchor_activity = (
        (action_value_phase - noop_value_phase)
        .float()
        .square()
        .mean(dim=(1, 3, 4))
    )
    anchor_keep = max(
        1, math.ceil(spatial * TEMPORAL_RESIDUAL_KEEP_FRACTION)
    )
    anchor_active = torch.topk(
        anchor_activity, k=anchor_keep, dim=1, largest=True, sorted=False
    ).indices
    anchor_index = anchor_active[:, None, :, None, None].expand(
        batch, LATENT_PHASES, anchor_keep, heads, width
    )

    def anchor_selected(tensor: torch.Tensor) -> torch.Tensor:
        return torch.gather(tensor, 2, anchor_index).float()

    action_logits = torch.einsum(
        "btkhd,bukhd->bkhtu",
        anchor_selected(action_query_phase),
        anchor_selected(action_key_phase),
    ) / math.sqrt(width)
    noop_logits = torch.einsum(
        "btkhd,bukhd->bkhtu",
        anchor_selected(noop_query_phase),
        anchor_selected(noop_key_phase),
    ) / math.sqrt(width)
    kernel_contrast = (
        torch.softmax(action_logits, dim=-1).mean(dim=1)
        - torch.softmax(noop_logits, dim=-1).mean(dim=1)
    )
    route = torch.einsum(
        "bhtu,bushd->btshd", kernel_contrast, current_value_phase.float()
    )
    route = route - route[:, :1]

    current_temporal = (
        current_output_phase.float() - current_output_phase[:, :1].float()
    )
    target_activity = current_temporal.square().mean(dim=(1, 3, 4))
    target_keep = max(1, math.ceil(spatial * float(target_keep_fraction)))
    target_active = torch.topk(
        target_activity, k=target_keep, dim=1, largest=True, sorted=False
    ).indices
    target_gate = torch.zeros(
        batch, spatial, device=current_output.device, dtype=torch.bool
    )
    target_gate.scatter_(1, target_active, True)
    target_gate = target_gate[:, None, :, None, None]

    current_rms = current_temporal.square().mean(dim=(1, 4), keepdim=True).sqrt()
    route_rms = route.square().mean(dim=(1, 4), keepdim=True).sqrt()
    route_scaled = route * current_rms / route_rms.clamp_min(1.0e-6)
    replacement = current_output_phase[:, :1].float() + route_scaled
    replacement = torch.where(
        route_rms > 1.0e-6,
        replacement,
        current_output_phase.float(),
    )
    hard_routed = torch.where(
        target_gate,
        replacement,
        current_output_phase.float(),
    )
    routed = (
        current_output_phase.float()
        + float(strength)
        * (hard_routed - current_output_phase.float())
    ).to(current_output.dtype)
    routed[:, 0].copy_(current_output_phase[:, 0])
    return routed.reshape_as(current_output)


def _correspondence_gated_hard_temporal_kernel_contrast_output(
    current_output: torch.Tensor,
    current_value: torch.Tensor,
    action_query: torch.Tensor,
    action_key: torch.Tensor,
    action_value: torch.Tensor,
    noop_query: torch.Tensor,
    noop_key: torch.Tensor,
    noop_value: torch.Tensor,
    *,
    strength: float,
    target_keep_fraction: float,
    anchor_stride: int = CORRESPONDENCE_ANCHOR_STRIDE,
) -> torch.Tensor:
    """Apply matched anchor-local temporal graphs to target-owned values.

    A global ``T x T`` kernel erases which actor/object site performed each
    part of an interaction.  This operator first selects target sites with
    genuine target-side temporal activity, then matches each selected site's
    phase-zero value feature to a phase-zero anchor feature.  The matched
    anchor site supplies only its action-minus-noop temporal QK graph.  That
    graph acts on the selected target site's own value trajectory.

    The nearest-neighbour index is the only cross-appearance spatial signal:
    no anchor value vector, output feature, latent, RGB, or coordinate is
    written into the target.  Phase zero and all non-selected target sites are
    bit-exact.  RMS matching prevents arbitrary anchor-logit scale from
    changing the target feature magnitude.
    """

    tensors = (
        current_output,
        current_value,
        action_query,
        action_key,
        action_value,
        noop_query,
        noop_key,
        noop_value,
    )
    if (
        any(item.ndim != 4 for item in tensors)
        or any(_shape(item) != _shape(current_output) for item in tensors[1:])
        or int(current_output.shape[1]) % LATENT_PHASES
    ):
        raise AnchorQKTransportError(
            "correspondence-gated hard kernel requires eight matched "
            "21-phase tensors"
        )
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 < float(strength) <= 1.0
        or not 0.0 < float(target_keep_fraction) <= 1.0
        or isinstance(anchor_stride, bool)
        or not isinstance(anchor_stride, int)
        or anchor_stride < 1
    ):
        raise AnchorQKTransportError(
            "correspondence-gated hard kernel controls are invalid"
        )

    batch, tokens, heads, width = _shape(current_output)
    spatial = tokens // LATENT_PHASES

    def phase(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, LATENT_PHASES, spatial, heads, width)

    current_output_phase = phase(current_output)
    current_value_phase = phase(current_value)
    action_query_phase = phase(action_query)
    action_key_phase = phase(action_key)
    action_value_phase = phase(action_value)
    noop_query_phase = phase(noop_query)
    noop_key_phase = phase(noop_key)
    noop_value_phase = phase(noop_value)

    current_temporal = (
        current_output_phase.float() - current_output_phase[:, :1].float()
    )
    target_activity = current_temporal.square().mean(dim=(1, 3, 4))
    target_keep = max(1, math.ceil(spatial * float(target_keep_fraction)))
    target_index = torch.topk(
        target_activity, k=target_keep, dim=1, largest=True, sorted=False
    ).indices
    target_gather = target_index[:, None, :, None, None].expand(
        batch, LATENT_PHASES, target_keep, heads, width
    )

    current_feature = torch.nn.functional.normalize(
        current_value_phase[:, 0].float().mean(dim=2), dim=-1, eps=1.0e-6
    )
    anchor_feature = torch.nn.functional.normalize(
        (
            0.5
            * (action_value_phase[:, 0].float() + noop_value_phase[:, 0].float())
        ).mean(dim=2),
        dim=-1,
        eps=1.0e-6,
    )
    selected_target_feature = torch.gather(
        current_feature,
        1,
        target_index[..., None].expand(batch, target_keep, width),
    )
    anchor_candidates = torch.arange(
        0,
        spatial,
        anchor_stride,
        device=current_output.device,
        dtype=torch.long,
    )
    similarity = torch.matmul(
        selected_target_feature,
        anchor_feature.index_select(1, anchor_candidates).transpose(-1, -2),
    )
    matched_anchor_index = anchor_candidates[similarity.argmax(dim=-1)]
    anchor_gather = matched_anchor_index[:, None, :, None, None].expand(
        batch, LATENT_PHASES, target_keep, heads, width
    )

    def matched(tensor: torch.Tensor) -> torch.Tensor:
        return torch.gather(tensor, 2, anchor_gather).float()

    action_logits = torch.einsum(
        "btkhd,bukhd->bkhtu",
        matched(action_query_phase),
        matched(action_key_phase),
    ) / math.sqrt(width)
    noop_logits = torch.einsum(
        "btkhd,bukhd->bkhtu",
        matched(noop_query_phase),
        matched(noop_key_phase),
    ) / math.sqrt(width)
    local_kernel = (
        torch.softmax(action_logits, dim=-1)
        - torch.softmax(noop_logits, dim=-1)
    )
    selected_target_value = torch.gather(
        current_value_phase, 2, target_gather
    ).float()
    route = torch.einsum(
        "bkhtu,bukhd->btkhd", local_kernel, selected_target_value
    )
    route = route - route[:, :1]

    selected_current = torch.gather(
        current_output_phase, 2, target_gather
    ).float()
    selected_temporal = selected_current - selected_current[:, :1]
    current_rms = selected_temporal.square().mean(
        dim=(1, 4), keepdim=True
    ).sqrt()
    route_rms = route.square().mean(dim=(1, 4), keepdim=True).sqrt()
    route_scaled = route * current_rms / route_rms.clamp_min(1.0e-6)
    replacement = selected_current[:, :1] + route_scaled
    replacement = torch.where(
        route_rms > 1.0e-6, replacement, selected_current
    )
    blended = selected_current + float(strength) * (
        replacement - selected_current
    )

    routed = current_output_phase.float().clone()
    routed.scatter_(2, target_gather, blended)
    routed[:, 0].copy_(current_output_phase[:, 0])
    return routed.to(current_output.dtype).reshape_as(current_output)


def _event01_role_graph_hard_attention_output(
    current_output: torch.Tensor,
    current_value: torch.Tensor,
    action_query: torch.Tensor,
    action_key: torch.Tensor,
    noop_query: torch.Tensor,
    noop_key: torch.Tensor,
    *,
    strength: float,
    proposal_index: int,
) -> torch.Tensor:
    """Route an anchor actor/object attention graph onto source-owned values.

    The action and matched no-op anchor calls define a compact per-head graph
    over ``(phase, actor/object role)`` nodes.  The graph contrast is evaluated
    entirely inside the anchor coordinate system.  It is then applied to value
    features pooled from the source child and one explicit source-stone
    proposal.  Only those source-coordinate role regions are modified; no
    anchor value, RGB, clothing, background, or spatial token is copied.

    This is an intentionally Event01-specific, transparent geometry canary.
    It tests whether entity-role binding is the missing bridge before an
    automatic detector/correspondence module is introduced.
    """

    tensors = (
        current_output,
        current_value,
        action_query,
        action_key,
        noop_query,
        noop_key,
    )
    if (
        any(item.ndim != 4 for item in tensors)
        or any(_shape(item) != _shape(current_output) for item in tensors[1:])
        or isinstance(proposal_index, bool)
        or not isinstance(proposal_index, int)
        or not 0 <= proposal_index < EVENT01_ROLE_PROPOSALS
        or isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 < float(strength) <= 1.0
    ):
        raise AnchorQKTransportError("Event01 role-graph controls differ")
    batch, tokens, heads, width = _shape(current_output)
    spatial = EVENT01_SPATIAL_HEIGHT * EVENT01_SPATIAL_WIDTH
    if tokens != LATENT_PHASES * spatial:
        raise AnchorQKTransportError(
            "Event01 role graph requires the audited 21x36x26 attention-token geometry"
        )

    def phase(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, LATENT_PHASES, spatial, heads, width)

    device = current_output.device
    yy, xx = torch.meshgrid(
        torch.arange(EVENT01_SPATIAL_HEIGHT, device=device, dtype=torch.float32),
        torch.arange(EVENT01_SPATIAL_WIDTH, device=device, dtype=torch.float32),
        indexing="ij",
    )

    def gaussian_role_weights(
        centers: tuple[tuple[float, float], tuple[float, float]],
        *,
        actor_scale: tuple[float, float],
        object_scale: tuple[float, float],
    ) -> torch.Tensor:
        weights = []
        for (center_x, center_y), (scale_x, scale_y) in zip(
            centers, (actor_scale, object_scale)
        ):
            distance = (
                ((xx - float(center_x)) / scale_x).square()
                + ((yy - float(center_y)) / scale_y).square()
            )
            weight = torch.exp(-0.5 * distance).flatten()
            weights.append(weight / weight.sum().clamp_min(1.0e-12))
        return torch.stack(weights, dim=0)

    anchor_weights = gaussian_role_weights(
        (EVENT01_ANCHOR_ACTOR_XY, EVENT01_ANCHOR_OBJECT_XY),
        actor_scale=(4.5, 6.5),
        object_scale=(3.5, 3.5),
    )
    source_object = EVENT01_SOURCE_OBJECT_PROPOSALS_XY[proposal_index]
    target_weights = gaussian_role_weights(
        (EVENT01_SOURCE_ACTOR_XY, source_object),
        actor_scale=(4.5, 7.0),
        object_scale=(2.0, 1.75),
    )

    def pool_roles(tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("rs,btshd->btrhd", weights, phase(tensor).float())

    action_q = pool_roles(action_query, anchor_weights)
    action_k = pool_roles(action_key, anchor_weights)
    noop_q = pool_roles(noop_query, anchor_weights)
    noop_k = pool_roles(noop_key, anchor_weights)
    # [B,H,T,R,U,S]: query phase/role attends to key phase/role.
    action_logits = torch.einsum(
        "btrhd,bushd->bhtrus", action_q, action_k
    ) / math.sqrt(width)
    noop_logits = torch.einsum(
        "btrhd,bushd->bhtrus", noop_q, noop_k
    ) / math.sqrt(width)
    flat_shape = (batch, heads, LATENT_PHASES, 2, LATENT_PHASES * 2)
    action_kernel = torch.softmax(action_logits.reshape(flat_shape), dim=-1).reshape_as(
        action_logits
    )
    noop_kernel = torch.softmax(noop_logits.reshape(flat_shape), dim=-1).reshape_as(
        noop_logits
    )
    kernel_contrast = action_kernel - noop_kernel

    target_values = pool_roles(current_value, target_weights)
    role_route = torch.einsum(
        "bhtrus,bushd->btrhd", kernel_contrast, target_values
    )
    role_route = role_route - role_route[:, :1]

    current_phase = phase(current_output).float()
    current_roles = torch.einsum(
        "rs,btshd->btrhd", target_weights, current_phase
    )
    current_temporal = current_roles - current_roles[:, :1]
    route_rms = role_route.square().mean(dim=(1, 4), keepdim=True).sqrt()
    temporal_rms = current_temporal.square().mean(dim=(1, 4), keepdim=True).sqrt()
    absolute_rms = current_roles.square().mean(dim=(1, 4), keepdim=True).sqrt()
    desired_rms = torch.maximum(temporal_rms, 0.25 * absolute_rms)
    scaled_route = role_route * desired_rms / route_rms.clamp_min(1.0e-6)
    desired_roles = current_roles[:, :1] + scaled_route

    # Hard elliptical target supports.  Each site keeps its current spatial
    # residual, while the role-level temporal mean is replaced by the routed
    # actor/object graph.  Outside both roles and all of phase zero are exact.
    target_centers = (EVENT01_SOURCE_ACTOR_XY, source_object)
    target_scales = ((5.5, 8.5), (2.0, 1.75))
    raw_role_masks = []
    for (center_x, center_y), (scale_x, scale_y) in zip(
        target_centers, target_scales
    ):
        raw_role_masks.append(
            ((xx - float(center_x)) / scale_x).square()
            + ((yy - float(center_y)) / scale_y).square()
            <= 1.0
        )
    # Contact-object tokens take precedence in the overlap.  Otherwise the
    # broad actor ellipse can consume a nearby stone completely, silently
    # deleting the very recipient whose interaction graph we need to test.
    role_masks = (
        (raw_role_masks[0] & ~raw_role_masks[1]).flatten(),
        raw_role_masks[1].flatten(),
    )
    routed = current_phase.clone()
    for role_index, role_mask in enumerate(role_masks):
        if not bool(role_mask.any()):
            raise AnchorQKTransportError("Event01 target role support is empty")
        replacement = (
            current_phase[:, :, role_mask]
            - current_roles[:, :, role_index : role_index + 1]
            + desired_roles[:, :, role_index : role_index + 1]
        )
        routed[:, :, role_mask] = (
            current_phase[:, :, role_mask]
            + float(strength)
            * (replacement - current_phase[:, :, role_mask])
        )
    routed[:, 0].copy_(current_phase[:, 0])
    return routed.to(current_output.dtype).reshape_as(current_output)


def _event01_dynamic_target_centers(
    proposal_index: int,
    *,
    source_side_aligned: bool = False,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    if (
        isinstance(proposal_index, bool)
        or not isinstance(proposal_index, int)
        or not 0 <= proposal_index < EVENT01_ROLE_PROPOSALS
        or not isinstance(source_side_aligned, bool)
        or not (
            len(EVENT01_TARGET_ACTOR_TRAJECTORY_XY)
            == len(EVENT01_TARGET_OBJECT_LIFT_PROGRESS)
            == LATENT_PHASES
        )
    ):
        raise AnchorQKTransportError("Event01 dynamic target trajectory differs")
    source_object = EVENT01_SOURCE_OBJECT_PROPOSALS_XY[proposal_index]
    anchor_terminal_actor = EVENT01_ANCHOR_ACTOR_TRAJECTORY_XY[-1]
    anchor_terminal_object = EVENT01_ANCHOR_OBJECT_TRAJECTORY_XY[-1]
    # ANC transfers the action anchor's terminal object-to-actor relation.
    # The former source-relative heuristic incorrectly sent the object to the
    # source actor's right/up side, directly into Event01's foreground bush.
    terminal_dx = float(anchor_terminal_object[0]) - float(anchor_terminal_actor[0])
    terminal_dy = float(anchor_terminal_object[1]) - float(anchor_terminal_actor[1])
    if source_side_aligned:
        source_initial_dx = float(source_object[0]) - float(EVENT01_SOURCE_ACTOR_XY[0])
        if source_initial_dx != 0.0 and terminal_dx * source_initial_dx < 0.0:
            terminal_dx = -terminal_dx
    result = []
    for actor_xy, progress in zip(
        EVENT01_TARGET_ACTOR_TRAJECTORY_XY,
        EVENT01_TARGET_OBJECT_LIFT_PROGRESS,
    ):
        terminal = (
            float(actor_xy[0]) + terminal_dx,
            float(actor_xy[1]) + terminal_dy,
        )
        object_xy = (
            (1.0 - float(progress)) * float(source_object[0])
            + float(progress) * terminal[0],
            (1.0 - float(progress)) * float(source_object[1])
            + float(progress) * terminal[1],
        )
        result.append((actor_xy, object_xy))
    return tuple(result)


def _event01_dynamic_source_object_feature(
    current_feature: torch.Tensor,
    *,
    strength: float,
    proposal_index: int,
    feature_name: str,
) -> torch.Tensor:
    """Carry one source-stone phase-zero feature along its lift trajectory."""

    if (
        current_feature.ndim != 4
        or isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 < float(strength) <= 1.0
    ):
        raise AnchorQKTransportError(
            f"Event01 source-object {feature_name} controls differ"
        )
    batch, tokens, heads, width = _shape(current_feature)
    spatial = EVENT01_SPATIAL_HEIGHT * EVENT01_SPATIAL_WIDTH
    if tokens != LATENT_PHASES * spatial:
        raise AnchorQKTransportError(
            f"Event01 source-object {feature_name} route requires 21x36x26 tokens"
        )
    centers = _event01_dynamic_target_centers(proposal_index)
    device = current_feature.device
    yy, xx = torch.meshgrid(
        torch.arange(EVENT01_SPATIAL_HEIGHT, device=device, dtype=torch.float32),
        torch.arange(EVENT01_SPATIAL_WIDTH, device=device, dtype=torch.float32),
        indexing="ij",
    )

    def object_weight(center: tuple[float, float]) -> torch.Tensor:
        distance = (
            ((xx - float(center[0])) / 2.0).square()
            + ((yy - float(center[1])) / 1.75).square()
        )
        weight = torch.exp(-0.5 * distance).flatten()
        return weight / weight.sum().clamp_min(1.0e-12)

    current_phase = current_feature.reshape(
        batch, LATENT_PHASES, spatial, heads, width
    )
    phase0_weight = object_weight(EVENT01_SOURCE_OBJECT_PROPOSALS_XY[proposal_index])
    source_object_value = torch.einsum(
        "s,bshd->bhd", phase0_weight, current_phase[:, 0].float()
    )
    routed = current_phase.float().clone()
    for phase_index in range(1, LATENT_PHASES):
        object_xy = centers[phase_index][1]
        weight = object_weight(object_xy)
        current_mean = torch.einsum(
            "s,bshd->bhd", weight, current_phase[:, phase_index].float()
        )
        mask = (
            ((xx - float(object_xy[0])) / 2.0).square()
            + ((yy - float(object_xy[1])) / 1.75).square()
            <= 1.0
        ).flatten()
        replacement = (
            current_phase[:, phase_index, mask].float()
            - current_mean[:, None]
            + source_object_value[:, None]
        )
        routed[:, phase_index, mask] = (
            current_phase[:, phase_index, mask].float()
            + float(strength)
            * (replacement - current_phase[:, phase_index, mask].float())
        )
    routed[:, 0].copy_(current_phase[:, 0])
    return routed.to(current_feature.dtype).reshape_as(current_feature)


def _event01_dynamic_source_object_value(
    current_value: torch.Tensor,
    *,
    strength: float,
    proposal_index: int,
) -> torch.Tensor:
    """Carry the phase-zero source-stone V before attention."""

    return _event01_dynamic_source_object_feature(
        current_value,
        strength=strength,
        proposal_index=proposal_index,
        feature_name="value",
    )


def _event01_dynamic_source_object_output(
    current_output: torch.Tensor,
    *,
    strength: float,
    proposal_index: int,
) -> torch.Tensor:
    """Carry source-owned phase-zero attention output along the object path."""

    return _event01_dynamic_source_object_feature(
        current_output,
        strength=strength,
        proposal_index=proposal_index,
        feature_name="attention output",
    )


def _event01_dynamic_source_patch_move(
    current_output: torch.Tensor,
    *,
    strength: float,
    proposal_index: int,
    source_output: Optional[torch.Tensor] = None,
    source_side_aligned: bool = False,
) -> torch.Tensor:
    """Move an explicit source-branch patch instead of broadcasting a mean.

    The previous source-object carrier pooled a stone into one vector and then
    broadcast that vector over a comparatively large ellipse.  That operation
    discarded scale and within-object spatial structure, exactly matching the
    observed oversized-block failure.  This upper-bound route keeps the four
    phase-zero tokens nearest the selected source stone as an ordered 2x2
    patch, translates the patch along the action-derived object trajectory,
    and replaces the vacated source support with its local background ring.
    When ``source_output`` is supplied, the ordered carrier and vacancy fill
    come from the paired source branch, not the target suffix's caption-shaped
    phase zero.  The fallback exists only for direct unit tests and historical
    ABI compatibility.  No anchor value, RGB feature, or clean latent enters
    the carrier.
    """

    if (
        current_output.ndim != 4
        or isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 < float(strength) <= 1.0
        or not isinstance(source_side_aligned, bool)
    ):
        raise AnchorQKTransportError(
            "Event01 source-patch move controls differ"
        )
    batch, tokens, heads, width = _shape(current_output)
    if source_output is not None and (
        _shape(source_output) != _shape(current_output)
        or source_output.dtype != current_output.dtype
        or source_output.device != current_output.device
        or not bool(torch.isfinite(source_output).all())
    ):
        raise AnchorQKTransportError(
            "Event01 explicit source-patch carrier geometry differs"
        )
    spatial = EVENT01_SPATIAL_HEIGHT * EVENT01_SPATIAL_WIDTH
    if tokens != LATENT_PHASES * spatial:
        raise AnchorQKTransportError(
            "Event01 source-patch move requires 21x36x26 tokens"
        )
    centers = _event01_dynamic_target_centers(
        proposal_index,
        source_side_aligned=source_side_aligned,
    )
    source_x, source_y = EVENT01_SOURCE_OBJECT_PROPOSALS_XY[proposal_index]
    device = current_output.device
    yy, xx = torch.meshgrid(
        torch.arange(EVENT01_SPATIAL_HEIGHT, device=device, dtype=torch.float32),
        torch.arange(EVENT01_SPATIAL_WIDTH, device=device, dtype=torch.float32),
        indexing="ij",
    )
    patch_distance = (
        ((xx - float(source_x)) / 1.0).square()
        + ((yy - float(source_y)) / 0.75).square()
    )
    source_patch_flat = torch.topk(
        patch_distance.flatten(), k=4, largest=False, sorted=True
    ).indices
    source_patch_yx = torch.stack(
        (
            torch.div(
                source_patch_flat,
                EVENT01_SPATIAL_WIDTH,
                rounding_mode="floor",
            ),
            source_patch_flat.remainder(EVENT01_SPATIAL_WIDTH),
        ),
        dim=1,
    )
    source_distance = (
        ((xx - float(source_x)) / 2.5).square()
        + ((yy - float(source_y)) / 1.75).square()
    )
    source_ring_mask = (source_distance > 0.45) & (source_distance <= 1.0)
    if not bool(source_ring_mask.any()):
        raise AnchorQKTransportError("Event01 source-patch ring is empty")

    current_phase = current_output.reshape(
        batch, LATENT_PHASES, spatial, heads, width
    )
    source_phase = (
        source_output.reshape(batch, LATENT_PHASES, spatial, heads, width)
        if source_output is not None
        else current_phase
    )
    source_flat = source_patch_flat
    source_patch = source_phase[:, 0].index_select(1, source_flat).float()
    source_ring = source_phase[:, 0, source_ring_mask.flatten()].float().mean(
        dim=1, keepdim=True
    )
    source_signature = source_patch - source_ring
    routed = current_phase.float().clone()

    offsets = source_patch_yx.float()
    offsets[:, 0] -= float(source_y)
    offsets[:, 1] -= float(source_x)
    source_origin_flat = source_flat
    for phase_index in range(1, LATENT_PHASES):
        progress = float(EVENT01_TARGET_OBJECT_LIFT_PROGRESS[phase_index])
        if progress <= 0.0:
            continue
        object_x, object_y = centers[phase_index][1]
        target_y = torch.floor(offsets[:, 0] + float(object_y) + 0.5).long()
        target_x = torch.floor(offsets[:, 1] + float(object_x) + 0.5).long()
        if not bool(
            (
                (target_y >= 0)
                & (target_y < EVENT01_SPATIAL_HEIGHT)
                & (target_x >= 0)
                & (target_x < EVENT01_SPATIAL_WIDTH)
            ).all()
        ):
            raise AnchorQKTransportError(
                "Event01 translated source patch leaves the attention grid"
            )
        target_flat = target_y * EVENT01_SPATIAL_WIDTH + target_x
        if int(target_flat.unique().numel()) != int(source_flat.numel()):
            raise AnchorQKTransportError(
                "Event01 translated source patch contains duplicate tokens"
            )

        target_distance = (
            ((xx - float(object_x)) / 2.5).square()
            + ((yy - float(object_y)) / 1.75).square()
        )
        target_ring_mask = (target_distance > 0.45) & (target_distance <= 1.0)
        if not bool(target_ring_mask.any()):
            raise AnchorQKTransportError("Event01 target-patch ring is empty")
        target_ring = current_phase[
            :, phase_index, target_ring_mask.flatten()
        ].float().mean(dim=1, keepdim=True)

        moved = routed[:, phase_index].clone()
        # Once lift begins, remove the source-object residual at the original
        # site.  The target patch is written afterward so overlap at early
        # contact remains well-defined.
        moved[:, source_origin_flat] = source_ring
        moved[:, target_flat] = target_ring + source_signature
        routed[:, phase_index] = (
            current_phase[:, phase_index].float()
            + float(strength)
            * (moved - current_phase[:, phase_index].float())
        )
    routed[:, 0].copy_(current_phase[:, 0])
    return routed.to(current_output.dtype).reshape_as(current_output)


def _event01_role_graph_logit_bias_attention_output(
    current_output: torch.Tensor,
    current_query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    action_query: torch.Tensor,
    action_key: torch.Tensor,
    noop_query: torch.Tensor,
    noop_key: torch.Tensor,
    *,
    strength: float,
    proposal_index: int,
    dynamic_roles: bool = False,
    source_object_carry: bool = False,
    source_side_aligned: bool = False,
) -> torch.Tensor:
    """Add the anchor action/no-op role-logit contrast to current attention.

    Unlike the Round61 hard route, this operator never replaces the target
    actor/object temporal trajectory.  It computes the target model's own
    actor/object role logits, adds the matched pure-T2V action-minus-noop
    logit contrast, and applies only the resulting attention-output delta to
    source-coordinate role supports.  Current/source values remain the sole
    content stream.  If action and no-op Q/K are equal, the route is exactly
    the identity map (up to the softmax subtraction performed in float32).
    """

    tensors = (
        current_output,
        current_query,
        current_key,
        current_value,
        action_query,
        action_key,
        noop_query,
        noop_key,
    )
    if (
        any(item.ndim != 4 for item in tensors)
        or any(_shape(item) != _shape(current_output) for item in tensors[1:])
        or isinstance(proposal_index, bool)
        or not isinstance(proposal_index, int)
        or not 0 <= proposal_index < EVENT01_ROLE_PROPOSALS
        or isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 < float(strength) <= 1.0
        or not isinstance(dynamic_roles, bool)
        or not isinstance(source_object_carry, bool)
        or not isinstance(source_side_aligned, bool)
        or source_object_carry and not dynamic_roles
        or source_side_aligned and not dynamic_roles
    ):
        raise AnchorQKTransportError("Event01 role-logit controls differ")
    batch, tokens, heads, width = _shape(current_output)
    spatial = EVENT01_SPATIAL_HEIGHT * EVENT01_SPATIAL_WIDTH
    if tokens != LATENT_PHASES * spatial:
        raise AnchorQKTransportError(
            "Event01 role logit bias requires the audited 21x36x26 attention-token geometry"
        )

    def phase(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, LATENT_PHASES, spatial, heads, width)

    device = current_output.device
    yy, xx = torch.meshgrid(
        torch.arange(EVENT01_SPATIAL_HEIGHT, device=device, dtype=torch.float32),
        torch.arange(EVENT01_SPATIAL_WIDTH, device=device, dtype=torch.float32),
        indexing="ij",
    )

    def gaussian_role_weights(
        centers: tuple[tuple[float, float], tuple[float, float]],
        *,
        actor_scale: tuple[float, float],
        object_scale: tuple[float, float],
    ) -> torch.Tensor:
        weights = []
        for (center_x, center_y), (scale_x, scale_y) in zip(
            centers, (actor_scale, object_scale)
        ):
            distance = (
                ((xx - float(center_x)) / scale_x).square()
                + ((yy - float(center_y)) / scale_y).square()
            )
            weight = torch.exp(-0.5 * distance).flatten()
            weights.append(weight / weight.sum().clamp_min(1.0e-12))
        return torch.stack(weights, dim=0)

    source_object = EVENT01_SOURCE_OBJECT_PROPOSALS_XY[proposal_index]

    if dynamic_roles:
        if not (
            len(EVENT01_ANCHOR_ACTOR_TRAJECTORY_XY)
            == len(EVENT01_ANCHOR_OBJECT_TRAJECTORY_XY)
            == LATENT_PHASES
        ):
            raise AnchorQKTransportError("Event01 dynamic role trajectory differs")
        anchor_centers_by_phase = tuple(
            zip(
                EVENT01_ANCHOR_ACTOR_TRAJECTORY_XY,
                EVENT01_ANCHOR_OBJECT_TRAJECTORY_XY,
            )
        )
        target_centers_by_phase = _event01_dynamic_target_centers(
            proposal_index,
            source_side_aligned=source_side_aligned,
        )
    else:
        anchor_centers_by_phase = (
            (EVENT01_ANCHOR_ACTOR_XY, EVENT01_ANCHOR_OBJECT_XY),
        ) * LATENT_PHASES
        target_centers_by_phase = (
            (EVENT01_SOURCE_ACTOR_XY, source_object),
        ) * LATENT_PHASES

    anchor_weights = torch.stack(
        [
            gaussian_role_weights(
                centers,
                actor_scale=(4.5, 6.5),
                object_scale=(3.5, 3.5),
            )
            for centers in anchor_centers_by_phase
        ],
        dim=0,
    )
    target_weights = torch.stack(
        [
            gaussian_role_weights(
                centers,
                actor_scale=(4.5, 7.0),
                object_scale=(2.0, 1.75),
            )
            for centers in target_centers_by_phase
        ],
        dim=0,
    )

    def pool_roles(tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("trs,btshd->btrhd", weights, phase(tensor).float())

    action_q = pool_roles(action_query, anchor_weights)
    action_k = pool_roles(action_key, anchor_weights)
    noop_q = pool_roles(noop_query, anchor_weights)
    noop_k = pool_roles(noop_key, anchor_weights)
    current_q = pool_roles(current_query, target_weights)
    current_k = pool_roles(current_key, target_weights)
    target_values = pool_roles(current_value, target_weights)
    if source_object_carry:
        # Phase zero is clamped to the clean source video.  Repeating only the
        # selected source object's phase-zero V preserves its identity while
        # the anchor graph supplies motion/interaction relations.  Anchor V
        # is never used.
        target_values[:, :, 1].copy_(
            target_values[:, :1, 1].expand_as(target_values[:, :, 1])
        )

    def role_logits(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btrhd,bushd->bhtrus", query, key) / math.sqrt(width)

    action_logits = role_logits(action_q, action_k)
    noop_logits = role_logits(noop_q, noop_k)
    current_logits = role_logits(current_q, current_k)
    anchor_logit_contrast = action_logits - noop_logits
    anchor_logit_contrast = anchor_logit_contrast - anchor_logit_contrast.mean(
        dim=(-2, -1), keepdim=True
    )
    flat_shape = (batch, heads, LATENT_PHASES, 2, LATENT_PHASES * 2)
    current_kernel = torch.softmax(current_logits.reshape(flat_shape), dim=-1)
    biased_kernel = torch.softmax(
        (
            current_logits
            + float(strength) * anchor_logit_contrast
        ).reshape(flat_shape),
        dim=-1,
    )
    current_kernel = current_kernel.reshape_as(current_logits)
    biased_kernel = biased_kernel.reshape_as(current_logits)
    role_delta = torch.einsum(
        "bhtrus,bushd->btrhd", biased_kernel - current_kernel, target_values
    )
    role_delta[:, 0].zero_()

    target_scales = ((5.5, 8.5), (2.0, 1.75))
    raw_actor_masks = []
    raw_object_masks = []
    for target_centers in target_centers_by_phase:
        masks = []
        for (center_x, center_y), (scale_x, scale_y) in zip(
            target_centers, target_scales
        ):
            masks.append(
                ((xx - float(center_x)) / scale_x).square()
                + ((yy - float(center_y)) / scale_y).square()
                <= 1.0
            )
        raw_actor_masks.append(masks[0])
        raw_object_masks.append(masks[1])
    raw_actor_masks = torch.stack(raw_actor_masks, dim=0)
    raw_object_masks = torch.stack(raw_object_masks, dim=0)
    role_masks = (
        (raw_actor_masks & ~raw_object_masks).reshape(LATENT_PHASES, spatial),
        raw_object_masks.reshape(LATENT_PHASES, spatial),
    )
    routed = phase(current_output).float().clone()
    for role_index, role_mask in enumerate(role_masks):
        if not bool(role_mask.any(dim=1).all()):
            raise AnchorQKTransportError("Event01 target role support is empty")
        for phase_index in range(LATENT_PHASES):
            phase_mask = role_mask[phase_index]
            routed[:, phase_index, phase_mask] = (
                routed[:, phase_index, phase_mask]
                + role_delta[:, phase_index, role_index : role_index + 1]
            )
    routed[:, 0].copy_(phase(current_output)[:, 0])
    return routed.to(current_output.dtype).reshape_as(current_output)


def _sparse_correspondence_temporal_residual(
    current: torch.Tensor,
    anchor: torch.Tensor,
    *,
    current_reference: torch.Tensor,
    anchor_reference: torch.Tensor,
    strength: float,
    anchor_stride: int = CORRESPONDENCE_ANCHOR_STRIDE,
) -> torch.Tensor:
    """Transport temporal change after phase-0 semantic token alignment.

    Same-coordinate transport is invalid when the self-generated actor and the
    source actor occupy different image locations.  Phase-0 value features do
    not contain RoPE and therefore provide the least position-biased internal
    correspondence available at this seam.  Each current spatial token selects
    its nearest anchor token from a deterministic stride-4 bank; the resulting
    mapping is held fixed for all 21 phases before the static basis is removed.
    """

    tensors = (current, anchor, current_reference, anchor_reference)
    if (
        any(item.ndim != 4 for item in tensors)
        or any(_shape(item) != _shape(current) for item in tensors[1:])
        or int(current.shape[1]) % LATENT_PHASES
    ):
        raise AnchorQKTransportError(
            "correspondence residual requires four matched 21-phase tensors"
        )
    batch, tokens, heads, width = _shape(current)
    spatial = tokens // LATENT_PHASES
    current_phase = current.reshape(batch, LATENT_PHASES, spatial, heads, width)
    anchor_phase = anchor.reshape(batch, LATENT_PHASES, spatial, heads, width)
    current_ref = current_reference.reshape(
        batch, LATENT_PHASES, spatial, heads, width
    )
    anchor_ref = anchor_reference.reshape(
        batch, LATENT_PHASES, spatial, heads, width
    )

    # Mean over heads keeps semantic channel structure while bounding the
    # correspondence matrix cost.  Striding only the anchor candidate bank
    # preserves one decision for every current token.
    current_feature = torch.nn.functional.normalize(
        current_ref[:, 0].float().mean(dim=2), dim=-1, eps=1.0e-6
    )
    anchor_feature = torch.nn.functional.normalize(
        anchor_ref[:, 0].float().mean(dim=2), dim=-1, eps=1.0e-6
    )
    if (
        isinstance(anchor_stride, bool)
        or not isinstance(anchor_stride, int)
        or anchor_stride < 1
    ):
        raise AnchorQKTransportError("correspondence anchor stride must be positive")
    candidate_indices = torch.arange(
        0,
        spatial,
        anchor_stride,
        device=current.device,
        dtype=torch.long,
    )
    candidate_feature = anchor_feature.index_select(1, candidate_indices)
    similarity = torch.matmul(
        current_feature, candidate_feature.transpose(-1, -2)
    )
    best_candidate = similarity.argmax(dim=-1)
    anchor_index = candidate_indices[best_candidate]
    gather_index = anchor_index[:, None, :, None, None].expand(
        batch, LATENT_PHASES, spatial, heads, width
    )
    aligned_anchor = torch.gather(anchor_phase, 2, gather_index)

    current_delta = current_phase - current_phase[:, :1]
    anchor_delta = aligned_anchor - aligned_anchor[:, :1]
    route = anchor_delta - current_delta
    score = route.float().square().mean(dim=(-1, -2))
    keep = max(1, math.ceil(spatial * TEMPORAL_RESIDUAL_KEEP_FRACTION))
    top = torch.topk(score, k=keep, dim=2, largest=True, sorted=False).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(2, top, True)
    routed = current_phase + float(strength) * route * mask[..., None, None].to(
        route.dtype
    )
    routed[:, 0].copy_(current_phase[:, 0])
    return routed.reshape_as(current)


def _sparse_correspondence_temporal_contrast(
    current: torch.Tensor,
    action: torch.Tensor,
    noop: torch.Tensor,
    *,
    current_reference: torch.Tensor,
    anchor_reference: torch.Tensor,
    strength: float,
    hard_replace: bool,
    mutual_gate: bool = False,
    anchor_stride: int = CORRESPONDENCE_ANCHOR_STRIDE,
) -> torch.Tensor:
    """Route dynamic-minus-static anchor features after phase-0 alignment.

    The phase-0 value stream establishes a fixed current-token to anchor-token
    correspondence.  Dynamic and static anchor features are then gathered by
    that map at every latent phase.  Their difference removes the anchor's
    constant appearance basis; subtracting its phase-0 value removes global
    context leakage.  Soft mode adds the aligned action contrast.  Hard mode
    replaces the selected current temporal trajectory with the aligned action
    trajectory while retaining the current phase-0 basis exactly.
    """

    tensors = (current, action, noop, current_reference, anchor_reference)
    if (
        any(item.ndim != 4 for item in tensors)
        or any(_shape(item) != _shape(current) for item in tensors[1:])
        or int(current.shape[1]) % LATENT_PHASES
        or not isinstance(hard_replace, bool)
        or not isinstance(mutual_gate, bool)
    ):
        raise AnchorQKTransportError(
            "correspondence contrast requires five matched 21-phase tensors"
        )
    if (
        isinstance(anchor_stride, bool)
        or not isinstance(anchor_stride, int)
        or anchor_stride < 1
    ):
        raise AnchorQKTransportError("correspondence anchor stride must be positive")
    batch, tokens, heads, width = _shape(current)
    spatial = tokens // LATENT_PHASES
    current_phase = current.reshape(batch, LATENT_PHASES, spatial, heads, width)
    action_phase = action.reshape(batch, LATENT_PHASES, spatial, heads, width)
    noop_phase = noop.reshape(batch, LATENT_PHASES, spatial, heads, width)
    current_ref = current_reference.reshape(
        batch, LATENT_PHASES, spatial, heads, width
    )
    anchor_ref = anchor_reference.reshape(
        batch, LATENT_PHASES, spatial, heads, width
    )

    current_feature = torch.nn.functional.normalize(
        current_ref[:, 0].float().mean(dim=2), dim=-1, eps=1.0e-6
    )
    anchor_feature = torch.nn.functional.normalize(
        anchor_ref[:, 0].float().mean(dim=2), dim=-1, eps=1.0e-6
    )
    if mutual_gate:
        anchor_stride = 1
    candidate_indices = torch.arange(
        0,
        spatial,
        anchor_stride,
        device=current.device,
        dtype=torch.long,
    )
    similarity = torch.matmul(
        current_feature,
        anchor_feature.index_select(1, candidate_indices).transpose(-1, -2),
    )
    best_candidate = similarity.argmax(dim=-1)
    anchor_index = candidate_indices[best_candidate]
    mutual = None
    if mutual_gate:
        anchor_best_current = similarity.argmax(dim=1)
        matched_back = torch.gather(anchor_best_current, 1, best_candidate)
        current_indices = torch.arange(
            spatial, device=current.device, dtype=torch.long
        ).unsqueeze(0)
        mutual = matched_back == current_indices
        if not bool(mutual.any()):
            raise AnchorQKTransportError(
                "mutual correspondence produced no bidirectional match"
            )
    gather_index = anchor_index[:, None, :, None, None].expand(
        batch, LATENT_PHASES, spatial, heads, width
    )
    aligned_action = torch.gather(action_phase, 2, gather_index)
    aligned_noop = torch.gather(noop_phase, 2, gather_index)
    contrast = aligned_action - aligned_noop
    route = contrast - contrast[:, :1]
    score = route.float().square().mean(dim=(-1, -2))
    keep = max(1, math.ceil(spatial * TEMPORAL_RESIDUAL_KEEP_FRACTION))
    selection_score = score
    if mutual is not None:
        keep = min(keep, int(mutual.sum(dim=1).min().item()))
        selection_score = score.masked_fill(~mutual[:, None, :], -torch.inf)
    top = torch.topk(
        selection_score, k=keep, dim=2, largest=True, sorted=False
    ).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(2, top, True)
    mask = mask[..., None, None]
    if hard_replace:
        replacement = current_phase[:, :1] + route
        routed = torch.where(mask, replacement, current_phase)
    else:
        routed = current_phase + float(strength) * route * mask.to(route.dtype)
    routed[:, 0].copy_(current_phase[:, 0])
    return routed.reshape_as(current)


def _source_target_dynamic_mask_from_hidden(
    full_hidden_state: torch.Tensor,
    *,
    target_keep_fraction: float = DUAL_DYNAMIC_TARGET_KEEP_FRACTION,
) -> torch.Tensor:
    """Select target sites whose temporal evolution must remain editable.

    The mask is computed from pre-projection full hidden states, which are
    identical on every Ulysses rank.  This avoids a different top-k decision
    per head shard.  Phase zero is always source-authoritative.
    """

    if (
        full_hidden_state.ndim != 3
        or int(full_hidden_state.shape[0]) != 1
        or int(full_hidden_state.shape[1]) <= 0
        or int(full_hidden_state.shape[1]) % (2 * LATENT_PHASES)
        or not 0.0 < float(target_keep_fraction) <= 1.0
    ):
        raise AnchorQKTransportError(
            "source/target dynamic mask requires one equal 21-phase pair"
        )
    target = full_hidden_state[:, int(full_hidden_state.shape[1]) // 2 :]
    batch, tokens, width = _shape(target)
    spatial = tokens // LATENT_PHASES
    target_phase = target.reshape(batch, LATENT_PHASES, spatial, width).float()
    temporal = target_phase - target_phase[:, :1]
    activity = temporal.square().mean(dim=(1, 3))
    keep = max(1, math.ceil(spatial * float(target_keep_fraction)))
    active = torch.topk(
        activity, k=keep, dim=1, largest=True, sorted=False
    ).indices
    spatial_mask = torch.zeros(
        batch, spatial, device=target.device, dtype=torch.bool
    )
    spatial_mask.scatter_(1, active, True)
    mask = spatial_mask[:, None].expand(batch, LATENT_PHASES, spatial).clone()
    mask[:, 0].zero_()
    return mask.reshape(batch, tokens, 1, 1)


class AnchorQKSelfAttnProcessor:
    """Wrap Bernini ``attn1`` at the official post-RoPE Q/K/V boundary."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        cache_bank: AnchorQKCacheBank,
        varlen_attention_fn: Optional[Callable[..., Any]] = None,
        get_parallel_state_fn: Optional[Callable[[], Any]] = None,
        gather_heads_scatter_seq_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not callable(base_processor) or not callable(
            getattr(base_processor, "_project_qkv", None)
        ):
            raise AnchorQKTransportError("base attn1 processor lacks official Q/K/V path")
        index = _exact_int(block_index, label="block index")
        if index not in cache_bank.selected_block_indices:
            raise AnchorQKTransportError("block is outside the anchor Q/K cache scope")
        self.base_processor = base_processor
        self.block_index = index
        self.cache_bank = cache_bank
        self._varlen_attention_fn = varlen_attention_fn
        self._get_parallel_state_fn = get_parallel_state_fn
        self._gather_heads_scatter_seq_fn = gather_heads_scatter_seq_fn
        self.base_delegations = 0
        self.capture_calls = 0
        self.replay_calls = 0
        self.source_kv_late_replay_calls = 0
        self.last_projected: Optional[dict[str, torch.Tensor]] = None

    def _runtime_ops(self) -> tuple[Callable[..., Any], Callable[[], Any], Callable[..., Any]]:
        varlen_fn = self._varlen_attention_fn
        state_fn = self._get_parallel_state_fn
        inverse_fn = self._gather_heads_scatter_seq_fn
        if varlen_fn is None:
            from bernini.attention import varlen_attention as varlen_fn
        if state_fn is None or inverse_fn is None:
            from bernini.parallel import gather_heads_scatter_seq, get_parallel_state

            if state_fn is None:
                state_fn = get_parallel_state
            if inverse_fn is None:
                inverse_fn = gather_heads_scatter_seq
        return varlen_fn, state_fn, inverse_fn

    def _full_hidden_state_for_route(
        self,
        hidden_state: torch.Tensor,
        *,
        invocation: AnchorQKInvocation,
    ) -> torch.Tensor:
        if invocation.transport != HARD_PREROPE_PHASE_MEAN_CONTRAST_QK:
            return hidden_state
        _varlen_fn, state_fn, _inverse_fn = self._runtime_ops()
        state = state_fn()
        if not bool(getattr(state, "ulysses_enabled", False)):
            return hidden_state
        from bernini.parallel.ops import gather_outputs

        full = gather_outputs(hidden_state, gather_dim=1)
        if full.ndim != 3 or int(full.shape[0]) != 1:
            raise AnchorQKTransportError("gathered pre-RoPE hidden state differs")
        return full

    def _local_hidden_state_for_projection(
        self,
        full_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        _varlen_fn, state_fn, _inverse_fn = self._runtime_ops()
        state = state_fn()
        if not bool(getattr(state, "ulysses_enabled", False)):
            return full_hidden_state
        from bernini.parallel.ops import slice_input_tensor

        local = slice_input_tensor(full_hidden_state, dim=1)
        if local.ndim != 3 or int(local.shape[0]) != 1:
            raise AnchorQKTransportError("sliced pre-RoPE hidden state differs")
        return local

    def _full_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Gather the complete visual sequence identically on every rank."""

        _varlen_fn, state_fn, _inverse_fn = self._runtime_ops()
        state = state_fn()
        if not bool(getattr(state, "ulysses_enabled", False)):
            return hidden_state
        from bernini.parallel.ops import gather_outputs

        full = gather_outputs(hidden_state, gather_dim=1)
        if full.ndim != 3 or int(full.shape[0]) != 1:
            raise AnchorQKTransportError("gathered dual-route hidden state differs")
        return full

    def _source_reprojected_target_kv(
        self,
        *,
        attn: Any,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        origin_hidden_states_seq_len: Optional[int],
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        static_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project source hidden states again at target rotary coordinates."""

        full_hidden = self._full_hidden_state(hidden_states)
        full_tokens = int(full_hidden.shape[1])
        if full_tokens % 2 or full_tokens <= 0:
            raise AnchorQKTransportError(
                "late source K/V replay requires one equal source/target pair"
            )
        boundary = full_tokens // 2
        source_as_pair = torch.cat(
            (full_hidden[:, :boundary], full_hidden[:, :boundary]), dim=1
        )
        local_source_as_pair = self._local_hidden_state_for_projection(source_as_pair)
        _source_query, source_key, source_value = self.base_processor._project_qkv(
            attn,
            local_source_as_pair,
            None,
            rotary_emb,
            origin_hidden_states_seq_len,
            False,
        )
        if (
            _shape(source_key) != _shape(current_key)
            or _shape(source_value) != _shape(current_value)
            or int(source_key.shape[1]) != full_tokens
        ):
            raise AnchorQKTransportError(
                "source-at-target-position K/V geometry differs"
            )
        source_target_key = source_key[:, boundary:]
        source_target_value = source_value[:, boundary:]
        if not static_only:
            return source_target_key, source_target_value
        dynamic_mask = _source_target_dynamic_mask_from_hidden(full_hidden)
        current_target_key = current_key[:, boundary:]
        current_target_value = current_value[:, boundary:]
        return (
            torch.where(dynamic_mask, current_target_key, source_target_key),
            torch.where(dynamic_mask, current_target_value, source_target_value),
        )

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        batch_image_vae_seqlen=None,
        text_features_length=None,
        origin_hidden_states_seq_len: Optional[int] = None,
        split_hidden_states_seq_len: Optional[int] = None,
        cu_seqlens_q_cache=None,
        max_seqlen_q_cache=None,
        cu_seqlens_k_cross_cache=None,
        cu_seqlens_q_cross_cache=None,
        max_seqlen_k_cross_cache=None,
        max_seqlen_q_cross_cache=None,
    ) -> torch.Tensor:
        invocation = current_anchor_qk_invocation()
        if invocation is None:
            self.base_delegations += 1
            return self.base_processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                rotary_emb=rotary_emb,
                batch_image_vae_seqlen=batch_image_vae_seqlen,
                text_features_length=text_features_length,
                origin_hidden_states_seq_len=origin_hidden_states_seq_len,
                split_hidden_states_seq_len=split_hidden_states_seq_len,
                cu_seqlens_q_cache=cu_seqlens_q_cache,
                max_seqlen_q_cache=max_seqlen_q_cache,
                cu_seqlens_k_cross_cache=cu_seqlens_k_cross_cache,
                cu_seqlens_q_cross_cache=cu_seqlens_q_cross_cache,
                max_seqlen_k_cross_cache=max_seqlen_k_cross_cache,
                max_seqlen_q_cross_cache=max_seqlen_q_cross_cache,
            )
        if invocation.cache_bank is not self.cache_bank:
            raise AnchorQKTransportError("processor and invocation cache banks differ")
        if encoder_hidden_states is not None or attention_mask is not None:
            raise AnchorQKTransportError("anchor Q/K transport only wraps visual self-attention")
        if rotary_emb is None:
            raise AnchorQKTransportError("anchor Q/K transport requires post-RoPE projection")
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            raise AnchorQKTransportError("hidden_states must be [1,L,D]")

        query, key, value = self.base_processor._project_qkv(
            attn,
            hidden_states,
            None,
            rotary_emb,
            origin_hidden_states_seq_len,
            False,
        )
        if _shape(query) != _shape(key) or _shape(query) != _shape(value):
            raise AnchorQKTransportError("projected Q/K/V geometry differs")
        gathered_tokens = int(query.shape[1])
        if gathered_tokens <= 0 or (
            invocation.mode == REPLAY
            and invocation.replay_scope == PAIRED_SUFFIX
            and gathered_tokens % 2
        ):
            raise AnchorQKTransportError("projected visual token count cannot form the requested route")
        if sum(_lengths(batch_image_vae_seqlen, label="batch_image_vae_seqlen")) != gathered_tokens:
            raise AnchorQKTransportError("batch visual length differs from projected length")
        qk_only = invocation.transport in TARGET_OWNED_QK_TRANSPORTS_V14R2
        # QK-only capture/replay has no hidden-state cache or matching ABI.
        route_hidden_state = (
            hidden_states
            if qk_only
            else self._full_hidden_state_for_route(
                hidden_states,
                invocation=invocation,
            )
        )
        dual_source_kv = invocation.transport in DUAL_SOURCE_KV_TRANSPORTS
        dual_early_block = (
            dual_source_kv and self.block_index in DUAL_EARLY_ANCHOR_BLOCKS
        )
        dual_late_block = (
            dual_source_kv and self.block_index in DUAL_LATE_SOURCE_KV_BLOCKS
        )

        entry: Optional[AnchorQKEntry | AnchorQKOnlyEntry] = None
        noop_entry: Optional[AnchorQKEntry | AnchorQKOnlyEntry] = None
        if invocation.mode == REPLAY:
            if not dual_source_kv or dual_early_block:
                if qk_only:
                    entry = self.cache_bank.consume_qk_only(
                        invocation=invocation,
                        block_index=self.block_index,
                        current_query=query,
                        current_key=key,
                    )
                else:
                    entry = self.cache_bank.consume(
                        invocation=invocation,
                        block_index=self.block_index,
                        current_query=query,
                        current_key=key,
                        current_value=value,
                        current_hidden_state=route_hidden_state,
                    )
            if invocation.transport in (
                TEMPORAL_CONTRAST_QK,
                TEMPORAL_CONTRAST_ATTN_OUTPUT,
                TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
                TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
                TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
                TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
                TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                HARD_PHASE_MEAN_CONTRAST_QK,
                HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
                HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
                TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
                *TARGET_OWNED_QK_TRANSPORTS_V14R2,
                ACTION_NOOP_OBSERVER_ATTN_OUTPUT,
                *TARGET_GATED_HARD_KERNEL_TRANSPORTS,
                *EVENT01_ROLE_GRAPH_TRANSPORTS,
            ):
                noop_invocation = replace(invocation, slot=NOOP_SLOT)
                if qk_only:
                    noop_entry = self.cache_bank.consume_qk_only(
                        invocation=noop_invocation,
                        block_index=self.block_index,
                        current_query=query,
                        current_key=key,
                    )
                else:
                    noop_entry = self.cache_bank.consume(
                        invocation=noop_invocation,
                        block_index=self.block_index,
                        current_query=query,
                        current_key=key,
                        current_value=value,
                        current_hidden_state=route_hidden_state,
                    )
            source_tokens = (
                gathered_tokens // 2
                if invocation.replay_scope == PAIRED_SUFFIX
                else 0
            )
            if dual_source_kv:
                if invocation.replay_scope != PAIRED_SUFFIX:
                    raise AnchorQKTransportError(
                        "dual source K/V route requires a paired source/target sequence"
                    )
                if dual_early_block:
                    if entry is None:
                        raise AnchorQKTransportError(
                            "dual early anchor-Q route lacks its anchor entry"
                        )
                    query = torch.cat(
                        (query[:, :source_tokens], entry.query), dim=1
                    )
                elif dual_late_block:
                    routed_key, routed_value = self._source_reprojected_target_kv(
                        attn=attn,
                        hidden_states=hidden_states,
                        rotary_emb=rotary_emb,
                        origin_hidden_states_seq_len=origin_hidden_states_seq_len,
                        current_key=key,
                        current_value=value,
                        static_only=invocation.transport
                        == DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_STATIC75,
                    )
                    key = torch.cat((key[:, :source_tokens], routed_key), dim=1)
                    value = torch.cat(
                        (value[:, :source_tokens], routed_value), dim=1
                    )
                    self.source_kv_late_replay_calls += 1
                    if (
                        invocation.transport
                        == DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_STATIC75
                    ):
                        self.cache_bank.source_kv_late_static75_replay_count += 1
                    else:
                        self.cache_bank.source_kv_late_all_replay_count += 1
            elif invocation.transport == HARD_QK:
                query = torch.cat((query[:, :source_tokens], entry.query), dim=1)
                key = torch.cat((key[:, :source_tokens], entry.key), dim=1)
            elif invocation.transport == HARD_K:
                key = torch.cat((key[:, :source_tokens], entry.key), dim=1)
            elif invocation.transport in (
                TEMPORAL_RESIDUAL_QK,
                TEMPORAL_RESIDUAL_K,
                TEMPORAL_RESIDUAL_QKV,
                TEMPORAL_RESIDUAL_V,
                TEMPORAL_RESIDUAL_ATTN_OUTPUT,
                TEMPORAL_CORRESPONDENCE_ATTN_OUTPUT,
                TEMPORAL_CONTRAST_QK,
                TEMPORAL_CONTRAST_ATTN_OUTPUT,
                TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
                TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
                TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
                TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
                TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                HARD_PHASE_MEAN_CONTRAST_QK,
                HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
                HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
                TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
                *TARGET_OWNED_QK_TRANSPORTS_V14R2,
                *TARGET_GATED_HARD_KERNEL_TRANSPORTS,
                *EVENT01_ROLE_GRAPH_TRANSPORTS,
            ):
                current_query = query[:, source_tokens:]
                current_key = key[:, source_tokens:]
                current_value = value[:, source_tokens:]
                if invocation.transport in (
                    EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT,
                    EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_ATTN_OUTPUT,
                    EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_SIDE_ALIGNED_ATTN_OUTPUT,
                ):
                    current_value = (
                        _event01_dynamic_source_object_value(
                            current_value,
                            strength=invocation.transport_strength,
                            proposal_index=invocation.role_proposal_index,
                        )
                        if invocation.transport
                        == EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT
                        else _event01_dynamic_source_patch_move(
                            current_value,
                            strength=invocation.transport_strength,
                            proposal_index=invocation.role_proposal_index,
                            source_output=value[:, :source_tokens],
                            source_side_aligned=invocation.transport
                            == EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_SIDE_ALIGNED_ATTN_OUTPUT,
                        )
                    )
                    value = torch.cat((value[:, :source_tokens], current_value), dim=1)
                if invocation.transport == HARD_PREROPE_PHASE_MEAN_CONTRAST_QK:
                    if noop_entry is None:
                        raise AnchorQKTransportError(
                            "pre-RoPE phase-mean contrast lacks anchor static entry"
                        )
                    current_hidden = route_hidden_state[:, source_tokens:]
                    routed_hidden = _hard_phase_mean_temporal_contrast(
                        current_hidden.unsqueeze(2),
                        entry.hidden_state.unsqueeze(2),
                        noop_entry.hidden_state.unsqueeze(2),
                    ).squeeze(2)
                    routed_full_hidden = torch.cat(
                        (route_hidden_state[:, :source_tokens], routed_hidden), dim=1
                    )
                    routed_local_hidden = self._local_hidden_state_for_projection(
                        routed_full_hidden
                    )
                    routed_query, routed_key, _routed_value = (
                        self.base_processor._project_qkv(
                            attn,
                            routed_local_hidden,
                            None,
                            rotary_emb,
                            origin_hidden_states_seq_len,
                            False,
                        )
                    )
                    if (
                        _shape(routed_query) != _shape(query)
                        or _shape(routed_key) != _shape(key)
                    ):
                        raise AnchorQKTransportError(
                            "pre-RoPE routed Q/K geometry differs"
                        )
                    query = routed_query
                    key = routed_key
                if invocation.transport in (
                    TEMPORAL_RESIDUAL_QK,
                    TEMPORAL_RESIDUAL_QKV,
                ):
                    routed_query = _sparse_frame0_temporal_residual(
                        current_query,
                        entry.query,
                        strength=invocation.transport_strength,
                    )
                    query = torch.cat((query[:, :source_tokens], routed_query), dim=1)
                if invocation.transport == TEMPORAL_CONTRAST_QK:
                    if noop_entry is None:
                        raise AnchorQKTransportError(
                            "Q/K contrast route lacks anchor static entry"
                        )
                    routed_query = _sparse_frame0_additive_contrast(
                        current_query,
                        entry.query,
                        noop_entry.query,
                        strength=invocation.transport_strength,
                    )
                    query = torch.cat((query[:, :source_tokens], routed_query), dim=1)
                if invocation.transport == HARD_PHASE_MEAN_CONTRAST_QK:
                    if noop_entry is None:
                        raise AnchorQKTransportError(
                            "phase-mean Q/K contrast lacks anchor static entry"
                        )
                    routed_query = _hard_phase_mean_temporal_contrast(
                        current_query,
                        entry.query,
                        noop_entry.query,
                    )
                    query = torch.cat((query[:, :source_tokens], routed_query), dim=1)
                if invocation.transport in (
                    TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
                    TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
                    TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
                ):
                    if noop_entry is None:
                        raise AnchorQKTransportError(
                            "correspondence Q/K contrast lacks anchor static entry"
                        )
                    routed_query = _sparse_correspondence_temporal_contrast(
                        current_query,
                        entry.query,
                        noop_entry.query,
                        current_reference=current_value,
                        anchor_reference=entry.value,
                        strength=invocation.transport_strength,
                        hard_replace=invocation.transport
                        == TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
                        mutual_gate=invocation.transport
                        == TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
                    )
                    query = torch.cat((query[:, :source_tokens], routed_query), dim=1)
                if invocation.transport in (
                    TEMPORAL_RESIDUAL_QK,
                    TEMPORAL_RESIDUAL_K,
                    TEMPORAL_RESIDUAL_QKV,
                ):
                    routed_key = _sparse_frame0_temporal_residual(
                        current_key,
                        entry.key,
                        strength=invocation.transport_strength,
                    )
                    key = torch.cat((key[:, :source_tokens], routed_key), dim=1)
                if invocation.transport == TEMPORAL_CONTRAST_QK:
                    if noop_entry is None:
                        raise AnchorQKTransportError(
                            "Q/K contrast route lacks anchor static entry"
                        )
                    routed_key = _sparse_frame0_additive_contrast(
                        current_key,
                        entry.key,
                        noop_entry.key,
                        strength=invocation.transport_strength,
                    )
                    key = torch.cat((key[:, :source_tokens], routed_key), dim=1)
                if invocation.transport == HARD_PHASE_MEAN_CONTRAST_QK:
                    if noop_entry is None:
                        raise AnchorQKTransportError(
                            "phase-mean Q/K contrast lacks anchor static entry"
                        )
                    routed_key = _hard_phase_mean_temporal_contrast(
                        current_key,
                        entry.key,
                        noop_entry.key,
                    )
                    key = torch.cat((key[:, :source_tokens], routed_key), dim=1)
                if invocation.transport in (
                    TEMPORAL_CORRESPONDENCE_CONTRAST_QK,
                    TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
                    TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
                ):
                    if noop_entry is None:
                        raise AnchorQKTransportError(
                            "correspondence Q/K contrast lacks anchor static entry"
                        )
                    routed_key = _sparse_correspondence_temporal_contrast(
                        current_key,
                        entry.key,
                        noop_entry.key,
                        current_reference=current_value,
                        anchor_reference=entry.value,
                        strength=invocation.transport_strength,
                        hard_replace=invocation.transport
                        == TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK,
                        mutual_gate=invocation.transport
                        == TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK,
                    )
                    key = torch.cat((key[:, :source_tokens], routed_key), dim=1)
                if invocation.transport in (
                    TEMPORAL_RESIDUAL_QKV,
                    TEMPORAL_RESIDUAL_V,
                ):
                    routed_value = _sparse_frame0_temporal_residual(
                        current_value,
                        entry.value,
                        strength=invocation.transport_strength,
                    )
                    value = torch.cat((value[:, :source_tokens], routed_value), dim=1)
            self.replay_calls += 1

        self.last_projected = {
            "query": query.detach().clone(),
            "key": key.detach().clone(),
        }
        if not qk_only:
            self.last_projected["value"] = value.detach().clone()
        varlen_fn, state_fn, inverse_fn = self._runtime_ops()
        state = state_fn()
        enabled, rank, size = replay_runtime.parallel_identity(state)
        if (rank, size) != (invocation.rank, invocation.ulysses_size):
            raise AnchorQKTransportError("runtime and invocation Ulysses identity differs")

        query_dtype = query.dtype
        output = varlen_fn(
            query.squeeze(0).contiguous(),
            key.squeeze(0).contiguous(),
            value.squeeze(0).contiguous(),
            cu_seqlens_q=cu_seqlens_q_cache,
            cu_seqlens_k=cu_seqlens_q_cache,
            max_seqlen_q=max_seqlen_q_cache,
            max_seqlen_k=max_seqlen_q_cache,
            causal=False,
        )
        if _shape(output) != _shape(query)[1:]:
            raise AnchorQKTransportError("varlen attention output geometry differs")
        output = output.unsqueeze(0)
        if invocation.mode == CAPTURE and (
            not dual_source_kv or dual_early_block
        ):
            if qk_only:
                self.cache_bank.capture_qk_only(
                    invocation=invocation,
                    block_index=self.block_index,
                    query=query,
                    key=key,
                )
            else:
                self.cache_bank.capture(
                    invocation=invocation,
                    block_index=self.block_index,
                    query=query,
                    key=key,
                    value=value,
                    attention_output=output,
                    hidden_state=route_hidden_state,
                )
            self.capture_calls += 1
        elif invocation.transport in (
            TEMPORAL_RESIDUAL_ATTN_OUTPUT,
            TEMPORAL_CORRESPONDENCE_ATTN_OUTPUT,
            TEMPORAL_CONTRAST_ATTN_OUTPUT,
            TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
            TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
            TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
            HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT,
            TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
            *TARGET_OWNED_QK_TRANSPORTS_V14R2,
            ACTION_NOOP_OBSERVER_ATTN_OUTPUT,
            *TARGET_GATED_HARD_KERNEL_TRANSPORTS,
            *EVENT01_ROLE_GRAPH_TRANSPORTS,
        ):
            if entry is None:
                raise AnchorQKTransportError("attention-output route lacks anchor entry")
            source_tokens = (
                gathered_tokens // 2
                if invocation.replay_scope == PAIRED_SUFFIX
                else 0
            )
            if invocation.transport == ACTION_NOOP_OBSERVER_ATTN_OUTPUT:
                if noop_entry is None:
                    raise AnchorQKTransportError(
                        "action/no-op observer lacks its matched no-op entry"
                    )
                routed_output = output[:, source_tokens:]
            elif invocation.transport == TEMPORAL_RESIDUAL_ATTN_OUTPUT:
                routed_output = _sparse_frame0_temporal_residual(
                    output[:, source_tokens:],
                    entry.attention_output,
                    strength=invocation.transport_strength,
                )
            elif invocation.transport == TEMPORAL_CORRESPONDENCE_ATTN_OUTPUT:
                routed_output = _sparse_correspondence_temporal_residual(
                    output[:, source_tokens:],
                    entry.attention_output,
                    current_reference=value[:, source_tokens:],
                    anchor_reference=entry.value,
                    strength=invocation.transport_strength,
                )
            elif invocation.transport == TEMPORAL_CONTRAST_ATTN_OUTPUT:
                if noop_entry is None:
                    raise AnchorQKTransportError(
                        "attention contrast route lacks anchor no-op entry"
                    )
                routed_output = _sparse_frame0_additive_contrast(
                    output[:, source_tokens:],
                    entry.attention_output,
                    noop_entry.attention_output,
                    strength=invocation.transport_strength,
                )
            elif invocation.transport == HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT:
                if noop_entry is None:
                    raise AnchorQKTransportError(
                        "phase-mean output contrast lacks anchor static entry"
                    )
                routed_output = _hard_phase_mean_temporal_contrast(
                    output[:, source_tokens:],
                    entry.attention_output,
                    noop_entry.attention_output,
                )
            elif invocation.transport == TARGET_OWNED_TEMPORAL_KERNEL_ATTN_OUTPUT_V14R2:
                if not isinstance(entry, AnchorQKOnlyEntry) or not isinstance(
                    noop_entry, AnchorQKOnlyEntry
                ):
                    raise AnchorQKTransportError(
                        "QK-only temporal route lacks Q/K-only action/noop entries"
                    )
                routed_output = _qk_only_temporal_kernel_contrast_output(
                    output[:, source_tokens:],
                    value[:, source_tokens:],
                    entry.query,
                    entry.key,
                    noop_entry.query,
                    noop_entry.key,
                    strength=invocation.transport_strength,
                )
            elif invocation.transport in (
                TARGET_OWNED_ACTIVITY_KERNEL_TOP10_ATTN_OUTPUT_V14R2,
                TARGET_OWNED_ACTIVITY_KERNEL_TOP25_ATTN_OUTPUT_V14R2,
            ):
                if not isinstance(entry, AnchorQKOnlyEntry) or not isinstance(
                    noop_entry, AnchorQKOnlyEntry
                ):
                    raise AnchorQKTransportError(
                        "QK-only target gate lacks Q/K-only action/noop entries"
                    )
                routed_output = (
                    _qk_only_target_gated_hard_temporal_kernel_contrast_output(
                        output[:, source_tokens:],
                        query[:, source_tokens:],
                        key[:, source_tokens:],
                        value[:, source_tokens:],
                        entry.query,
                        entry.key,
                        noop_entry.query,
                        noop_entry.key,
                        strength=invocation.transport_strength,
                        target_keep_fraction=(
                            0.10
                            if invocation.transport
                            == TARGET_OWNED_ACTIVITY_KERNEL_TOP10_ATTN_OUTPUT_V14R2
                            else 0.25
                        ),
                    )
                )
            elif invocation.transport == TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT:
                if noop_entry is None:
                    raise AnchorQKTransportError(
                        "temporal-kernel output contrast lacks anchor static entry"
                    )
                routed_output = _temporal_attention_kernel_contrast_output(
                    output[:, source_tokens:],
                    value[:, source_tokens:],
                    entry.query,
                    entry.key,
                    entry.value,
                    noop_entry.query,
                    noop_entry.key,
                    noop_entry.value,
                    strength=invocation.transport_strength,
                )
            elif invocation.transport in TARGET_GATED_HARD_KERNEL_TRANSPORTS:
                if noop_entry is None:
                    raise AnchorQKTransportError(
                        "target-gated hard kernel lacks anchor static entry"
                    )
                if (
                    invocation.transport
                    == CORRESPONDENCE_GATED_HARD_KERNEL_TOP25_ATTN_OUTPUT
                ):
                    routed_output = (
                        _correspondence_gated_hard_temporal_kernel_contrast_output(
                            output[:, source_tokens:],
                            value[:, source_tokens:],
                            entry.query,
                            entry.key,
                            entry.value,
                            noop_entry.query,
                            noop_entry.key,
                            noop_entry.value,
                            strength=invocation.transport_strength,
                            target_keep_fraction=0.25,
                        )
                    )
                else:
                    routed_output = _target_gated_hard_temporal_kernel_contrast_output(
                        output[:, source_tokens:],
                        value[:, source_tokens:],
                        entry.query,
                        entry.key,
                        entry.value,
                        noop_entry.query,
                        noop_entry.key,
                        noop_entry.value,
                        strength=invocation.transport_strength,
                        target_keep_fraction=(
                            0.10
                            if invocation.transport
                            == TARGET_GATED_HARD_KERNEL_TOP10_ATTN_OUTPUT
                            else 0.25
                        ),
                    )
            elif invocation.transport == EVENT01_ROLE_GRAPH_HARD_ATTN_OUTPUT:
                if noop_entry is None:
                    raise AnchorQKTransportError(
                        "Event01 role graph lacks anchor no-op entry"
                    )
                routed_output = _event01_role_graph_hard_attention_output(
                    output[:, source_tokens:],
                    value[:, source_tokens:],
                    entry.query,
                    entry.key,
                    noop_entry.query,
                    noop_entry.key,
                    strength=invocation.transport_strength,
                    proposal_index=invocation.role_proposal_index,
                )
            elif invocation.transport == EVENT01_ROLE_GRAPH_LOGIT_BIAS_ATTN_OUTPUT:
                if noop_entry is None:
                    raise AnchorQKTransportError(
                        "Event01 role-logit bias lacks anchor no-op entry"
                    )
                routed_output = _event01_role_graph_logit_bias_attention_output(
                    output[:, source_tokens:],
                    query[:, source_tokens:],
                    key[:, source_tokens:],
                    value[:, source_tokens:],
                    entry.query,
                    entry.key,
                    noop_entry.query,
                    noop_entry.key,
                    strength=invocation.transport_strength,
                    proposal_index=invocation.role_proposal_index,
                )
            elif invocation.transport in (
                EVENT01_DYNAMIC_ROLE_GRAPH_LOGIT_BIAS_ATTN_OUTPUT,
                EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_OBJECT_ATTN_OUTPUT,
                EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT,
                EVENT01_DYNAMIC_SOURCE_OBJECT_OUTPUT_ATTN_OUTPUT,
                EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_MOVE_ATTN_OUTPUT,
                EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_ATTN_OUTPUT,
            ):
                if noop_entry is None:
                    raise AnchorQKTransportError(
                        "Event01 dynamic role graph lacks anchor no-op entry"
                    )
                current_output = output[:, source_tokens:]
                if (
                    invocation.transport
                    == EVENT01_DYNAMIC_SOURCE_OBJECT_OUTPUT_ATTN_OUTPUT
                ):
                    current_output = _event01_dynamic_source_object_output(
                        current_output,
                        strength=invocation.transport_strength,
                        proposal_index=invocation.role_proposal_index,
                    )
                routed_output = _event01_role_graph_logit_bias_attention_output(
                    current_output,
                    query[:, source_tokens:],
                    key[:, source_tokens:],
                    value[:, source_tokens:],
                    entry.query,
                    entry.key,
                    noop_entry.query,
                    noop_entry.key,
                    strength=invocation.transport_strength,
                    proposal_index=invocation.role_proposal_index,
                    dynamic_roles=True,
                    source_object_carry=invocation.transport
                    in (
                        EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_OBJECT_ATTN_OUTPUT,
                        EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT,
                        EVENT01_DYNAMIC_SOURCE_OBJECT_OUTPUT_ATTN_OUTPUT,
                    ),
                    source_side_aligned=invocation.transport
                    == EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_SIDE_ALIGNED_ATTN_OUTPUT,
                )
                if (
                    invocation.transport
                    == EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_MOVE_ATTN_OUTPUT
                ):
                    routed_output = _event01_dynamic_source_patch_move(
                        routed_output,
                        strength=invocation.transport_strength,
                        proposal_index=invocation.role_proposal_index,
                        source_output=output[:, :source_tokens],
                    )
            else:
                if noop_entry is None:
                    raise AnchorQKTransportError(
                        "correspondence output contrast lacks anchor static entry"
                    )
                routed_output = _sparse_correspondence_temporal_contrast(
                    output[:, source_tokens:],
                    entry.attention_output,
                    noop_entry.attention_output,
                    current_reference=value[:, source_tokens:],
                    anchor_reference=entry.value,
                    strength=invocation.transport_strength,
                    hard_replace=invocation.transport
                    == TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT,
                    mutual_gate=invocation.transport
                    == TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT,
                )
            output = torch.cat((output[:, :source_tokens], routed_output), dim=1)
        if enabled:
            output = inverse_fn(output, head_dim=2, seq_dim=1)
        output = output.flatten(2, 3).contiguous().to(query_dtype)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        return output


class AnchorQKPatchHandle:
    def __init__(self, transformer: Any, cache_bank: AnchorQKCacheBank) -> None:
        blocks = getattr(transformer, "blocks", None)
        if not isinstance(blocks, (list, torch.nn.ModuleList)) or len(blocks) != BLOCK_COUNT:
            raise AnchorQKTransportError("transformer must expose exactly 30 blocks")
        self.transformer = transformer
        self.cache_bank = cache_bank
        self.originals: dict[int, Any] = {}
        self.processors: dict[int, AnchorQKSelfAttnProcessor] = {}

    def install(self) -> None:
        if self.originals:
            raise AnchorQKTransportError("anchor Q/K patch is already installed")
        for index in self.cache_bank.selected_block_indices:
            attn = self.transformer.blocks[index].attn1
            original = getattr(attn, "processor", None)
            processor = AnchorQKSelfAttnProcessor(
                original, block_index=index, cache_bank=self.cache_bank
            )
            self.originals[index] = original
            self.processors[index] = processor
            setter = getattr(attn, "set_processor", None)
            if callable(setter):
                setter(processor)
            else:
                attn.processor = processor

    def restore(self) -> None:
        for index, original in self.originals.items():
            attn = self.transformer.blocks[index].attn1
            setter = getattr(attn, "set_processor", None)
            if callable(setter):
                setter(original)
            else:
                attn.processor = original
        self.originals.clear()


@contextlib.contextmanager
def install_anchor_qk_transport(
    transformer: Any,
    *,
    selected_block_indices: Sequence[int] = DEFAULT_BLOCKS,
) -> Iterator[AnchorQKPatchHandle]:
    bank = AnchorQKCacheBank(selected_block_indices)
    handle = AnchorQKPatchHandle(transformer, bank)
    handle.install()
    try:
        yield handle
        bank.assert_empty()
    finally:
        handle.restore()


__all__ = [
    "ACTION_SLOT",
    "AnchorQKCacheBank",
    "AnchorQKOnlyEntry",
    "AnchorQKInvocation",
    "AnchorQKSelfAttnProcessor",
    "AnchorQKTransportError",
    "ACTION_NOOP_OBSERVER_ATTN_OUTPUT",
    "CAPTURE",
    "DEFAULT_BLOCKS",
    "DUAL_DYNAMIC_TARGET_KEEP_FRACTION",
    "DUAL_EARLY_ANCHOR_BLOCKS",
    "DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_ALL",
    "DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_STATIC75",
    "DUAL_LATE_SOURCE_KV_BLOCKS",
    "DUAL_SOURCE_KV_TRANSPORTS",
    "EVENT01_ROLE_GRAPH_HARD_ATTN_OUTPUT",
    "EVENT01_ROLE_GRAPH_LOGIT_BIAS_ATTN_OUTPUT",
    "EVENT01_ROLE_GRAPH_TRANSPORTS",
    "EVENT01_DYNAMIC_ROLE_GRAPH_LOGIT_BIAS_ATTN_OUTPUT",
    "EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_OBJECT_ATTN_OUTPUT",
    "EVENT01_DYNAMIC_SOURCE_OBJECT_VALUE_ATTN_OUTPUT",
    "EVENT01_DYNAMIC_SOURCE_OBJECT_OUTPUT_ATTN_OUTPUT",
    "EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_MOVE_ATTN_OUTPUT",
    "EVENT01_DYNAMIC_ROLE_GRAPH_SOURCE_PATCH_VALUE_ATTN_OUTPUT",
    "EVENT01_ROLE_PROPOSALS",
    "EVENT01_SPATIAL_HEIGHT",
    "EVENT01_SPATIAL_WIDTH",
    "EVENT01_SOURCE_ACTOR_XY",
    "EVENT01_SOURCE_OBJECT_PROPOSALS_XY",
    "HARD_K",
    "HARD_QK",
    "FULL_SEQUENCE",
    "PAIRED_SUFFIX",
    "NOOP_SLOT",
    "TEMPORAL_RESIDUAL_K",
    "TEMPORAL_RESIDUAL_QK",
    "TEMPORAL_RESIDUAL_QKV",
    "TEMPORAL_RESIDUAL_V",
    "TEMPORAL_RESIDUAL_ATTN_OUTPUT",
    "TEMPORAL_CORRESPONDENCE_ATTN_OUTPUT",
    "TEMPORAL_CONTRAST_QK",
    "TEMPORAL_CONTRAST_ATTN_OUTPUT",
    "TEMPORAL_CORRESPONDENCE_CONTRAST_QK",
    "TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_QK",
    "TEMPORAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT",
    "TEMPORAL_CORRESPONDENCE_HARD_CONTRAST_ATTN_OUTPUT",
    "TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_QK",
    "TEMPORAL_MUTUAL_CORRESPONDENCE_CONTRAST_ATTN_OUTPUT",
    "HARD_PHASE_MEAN_CONTRAST_QK",
    "HARD_PHASE_MEAN_CONTRAST_ATTN_OUTPUT",
    "HARD_PREROPE_PHASE_MEAN_CONTRAST_QK",
    "TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT",
    "TARGET_OWNED_TEMPORAL_KERNEL_ATTN_OUTPUT_V14R2",
    "TARGET_OWNED_ACTIVITY_KERNEL_TOP10_ATTN_OUTPUT_V14R2",
    "TARGET_OWNED_ACTIVITY_KERNEL_TOP25_ATTN_OUTPUT_V14R2",
    "TARGET_OWNED_QK_TRANSPORTS_V14R2",
    "TARGET_GATED_HARD_KERNEL_TOP10_ATTN_OUTPUT",
    "TARGET_GATED_HARD_KERNEL_TOP25_ATTN_OUTPUT",
    "CORRESPONDENCE_GATED_HARD_KERNEL_TOP25_ATTN_OUTPUT",
    "TARGET_GATED_HARD_KERNEL_TRANSPORTS",
    "REPLAY",
    "anchor_qk_invocation",
    "install_anchor_qk_transport",
]
