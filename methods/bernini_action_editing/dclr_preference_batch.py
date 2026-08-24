"""Exact packed visual batch for a Bernini DCLR preference pair.

Bernini does not use a physical batch dimension for variable-length visual
samples.  The pinned ``WanTransformer3DModel.forward`` expects one physical
sequence ``[1, sum(L_i), D]`` and obtains logical sample boundaries from
``batch_image_vae_seqlen``.  Consequently a winner/loser MV2V preference pair
is packed as::

    [source, winner_target, source, loser_target]

with ``batch_vae_seqlen=[2N, 2N]``.  Its RoPE has the same physical packing,
``[1, 1, 4N, 64]`` (not ``[2, 1, 2N, 64]``), while model timesteps have the
logical shape ``[2]``.  Bernini's ``prepare_inputs_for_sp`` turns the two
lengths into the self-attention boundaries ``[0, 2N, 4N]``; that boundary is
recorded here so candidate isolation is testable before a model forward.

This module performs only tensor validation, patch embedding, packing, and
target construction.  It does not call ``shared_step``, run an optimizer, or
perform a distributed collective.  Text embeddings remain the caller's
responsibility and must independently describe two logical samples via a
two-entry ``batch_text_seqlen``.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

import torch

try:  # Package import.
    from . import dclr_preference_objective as preference_objective
    from . import dclr_runtime_contract as runtime_contract
except ImportError:  # Direct import with METHOD_ROOT on sys.path.
    import dclr_preference_objective as preference_objective
    import dclr_runtime_contract as runtime_contract


SCHEMA_VERSION = "bernini-dclr-preference-packed-batch-v1"
LATENT_CHANNELS = 16
LATENT_PHASES = 21
PATCH_SIZE = (1, 2, 2)
LOGICAL_CANDIDATES = 2
SOURCE_ID = 1
TARGET_SOURCE_ID = 0


class DCLRPreferenceBatchError(RuntimeError):
    """A visual preference pair violates the pinned Bernini packing."""


@dataclass(frozen=True)
class PackedPreferenceBatch:
    """One physical Bernini sequence containing two isolated MV2V samples."""

    normalized_source: torch.Tensor
    flow_state: preference_objective.SharedPairFlowState
    noisy_latents: torch.Tensor
    rotary_embs: torch.Tensor
    timesteps: torch.Tensor
    batch_vae_seqlen: tuple[int, int]
    target_selector: torch.Tensor
    candidate_target_selector: torch.Tensor
    target_true_velocity: torch.Tensor
    logical_self_attention_cu_seqlens: torch.Tensor
    source_token_count: int
    target_token_count: int
    source_id: int
    target_source_id: int
    sigma_float32_bits_hex: str
    timestep_float32_bits_hex: str

    @property
    def winner_epsilon(self) -> torch.Tensor:
        """The literal shared epsilon object used for the winner."""

        return self.flow_state.epsilon

    @property
    def loser_epsilon(self) -> torch.Tensor:
        """The same literal epsilon object used for the loser."""

        return self.flow_state.epsilon

    @property
    def logical_batch_size(self) -> int:
        return len(self.batch_vae_seqlen)

    @property
    def total_visual_tokens(self) -> int:
        return int(self.noisy_latents.shape[1])


def _require_tensor(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DCLRPreferenceBatchError(f"{label} must be a torch.Tensor")
    if value.device.type == "meta":
        raise DCLRPreferenceBatchError(f"{label} cannot be a meta tensor")
    return value


def _require_finite_detached_fp32(
    value: Any,
    *,
    label: str,
) -> torch.Tensor:
    tensor = _require_tensor(value, label=label)
    if tensor.dtype != torch.float32:
        raise DCLRPreferenceBatchError(f"{label} must be exact FP32")
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise DCLRPreferenceBatchError(
            f"{label} must be detached from every model graph"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise DCLRPreferenceBatchError(f"{label} contains NaN or infinity")
    return tensor


def _same_tensor(left: Any, right: Any) -> bool:
    return bool(
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and tuple(left.shape) == tuple(right.shape)
        and left.dtype == right.dtype
        and left.device == right.device
        and left.layout == right.layout
        and torch.equal(left, right)
    )


def _fp32_bits(value: torch.Tensor, *, label: str) -> str:
    tensor = _require_finite_detached_fp32(value, label=label)
    if tuple(tensor.shape) != (1,):
        raise DCLRPreferenceBatchError(f"{label} must have shape [1]")
    return struct.pack("!f", float(tensor.item())).hex()


def _require_normalized_spatial(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    tensor = _require_finite_detached_fp32(value, label=label)
    if tensor.ndim != 5:
        raise DCLRPreferenceBatchError(
            f"{label} must be normalized [1,16,21,H,W]"
        )
    shape = tuple(int(item) for item in tensor.shape)
    if (
        shape[0] != 1
        or shape[1] != LATENT_CHANNELS
        or shape[2] != LATENT_PHASES
        or shape[3] <= 0
        or shape[4] <= 0
        or shape[3] % PATCH_SIZE[1]
        or shape[4] % PATCH_SIZE[2]
    ):
        raise DCLRPreferenceBatchError(
            f"{label} must be normalized [1,16,21,H,W] with positive even H/W"
        )
    if expected_shape is not None and shape != expected_shape:
        raise DCLRPreferenceBatchError(
            f"{label} geometry differs from the shared preference state"
        )
    return tensor


def _validate_flow_state(
    value: Any,
) -> preference_objective.SharedPairFlowState:
    if not isinstance(value, preference_objective.SharedPairFlowState):
        raise DCLRPreferenceBatchError(
            "flow_state must be a SharedPairFlowState"
        )
    fields = {
        "flow_state.sigma": value.sigma,
        "flow_state.timestep": value.timestep,
        "flow_state.epsilon": value.epsilon,
        "flow_state.winner_clean": value.winner_clean,
        "flow_state.loser_clean": value.loser_clean,
        "flow_state.winner_x_sigma": value.winner_x_sigma,
        "flow_state.loser_x_sigma": value.loser_x_sigma,
        "flow_state.winner_true_velocity": value.winner_true_velocity,
        "flow_state.loser_true_velocity": value.loser_true_velocity,
    }
    checked = {
        label: _require_finite_detached_fp32(tensor, label=label)
        for label, tensor in fields.items()
    }
    sigma = checked["flow_state.sigma"]
    timestep = checked["flow_state.timestep"]
    if tuple(sigma.shape) != (1,) or tuple(timestep.shape) != (1,):
        raise DCLRPreferenceBatchError(
            "flow_state sigma and timestep must each be exact FP32 [1]"
        )
    try:
        expected_timestep = runtime_contract.fp32_sigma_to_timestep(sigma)
    except runtime_contract.DCLRRuntimeContractError as error:
        raise DCLRPreferenceBatchError(str(error)) from error
    if not _same_tensor(timestep, expected_timestep):
        raise DCLRPreferenceBatchError(
            "flow_state timestep is not the exact shared 1000*sigma value"
        )

    winner = _require_normalized_spatial(
        value.winner_clean, label="flow_state.winner_clean"
    )
    shape = tuple(int(item) for item in winner.shape)
    for label in (
        "flow_state.loser_clean",
        "flow_state.epsilon",
        "flow_state.winner_x_sigma",
        "flow_state.loser_x_sigma",
        "flow_state.winner_true_velocity",
        "flow_state.loser_true_velocity",
    ):
        _require_normalized_spatial(
            fields[label], label=label, expected_shape=shape
        )
    if len({tensor.device for tensor in fields.values()}) != 1:
        raise DCLRPreferenceBatchError(
            "all shared preference-state tensors must use one device"
        )
    if torch.equal(value.winner_clean, value.loser_clean):
        raise DCLRPreferenceBatchError(
            "winner and loser clean candidates must not be tensor-identical"
        )

    broadcast_sigma = sigma.reshape(1, 1, 1, 1, 1)
    one = torch.ones_like(broadcast_sigma)
    expected_winner_x = (
        (one - broadcast_sigma) * value.winner_clean
        + broadcast_sigma * value.epsilon
    )
    expected_loser_x = (
        (one - broadcast_sigma) * value.loser_clean
        + broadcast_sigma * value.epsilon
    )
    expected_winner_velocity = value.epsilon - value.winner_clean
    expected_loser_velocity = value.epsilon - value.loser_clean
    for label, observed, expected in (
        (
            "winner x_sigma",
            value.winner_x_sigma,
            expected_winner_x,
        ),
        ("loser x_sigma", value.loser_x_sigma, expected_loser_x),
        (
            "winner true velocity",
            value.winner_true_velocity,
            expected_winner_velocity,
        ),
        (
            "loser true velocity",
            value.loser_true_velocity,
            expected_loser_velocity,
        ),
    ):
        if not _same_tensor(observed, expected):
            raise DCLRPreferenceBatchError(
                f"flow_state {label} does not use its one shared epsilon/sigma"
            )
    return value


def pack_spatial_velocity(value: torch.Tensor) -> torch.Tensor:
    """Pack ``[1,16,21,H,W]`` in official ``(pt ph pw c)`` order."""

    tensor = _require_normalized_spatial(value, label="spatial velocity")
    batch, channels, phases, height, width = (
        int(item) for item in tensor.shape
    )
    pt, ph, pw = PATCH_SIZE
    packed = (
        tensor.reshape(
            batch,
            channels,
            phases // pt,
            pt,
            height // ph,
            ph,
            width // pw,
            pw,
        )
        .permute(0, 2, 4, 6, 3, 5, 7, 1)
        .reshape(
            batch,
            (phases // pt) * (height // ph) * (width // pw),
            pt * ph * pw * channels,
        )
    )
    if packed.dtype != torch.float32 or int(packed.shape[2]) != (
        runtime_contract.PINNED_PATCH_DIM
    ):
        raise DCLRPreferenceBatchError(
            "packed true velocity changed the pinned FP32 patch layout"
        )
    return packed


def _transformer_dtype(transformer: Any) -> torch.dtype:
    patch = getattr(transformer, "patch_vae_latent", None)
    if not callable(patch):
        raise DCLRPreferenceBatchError(
            "active transformer must expose callable patch_vae_latent"
        )
    dtype = getattr(transformer, "dtype", None)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise DCLRPreferenceBatchError(
            "active transformer exposes no supported floating dtype"
        )
    return dtype


def _patch_one(
    transformer: Any,
    spatial: torch.Tensor,
    *,
    source_id: int,
    expected_tokens: int,
    transformer_dtype: torch.dtype,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = transformer.patch_vae_latent(
        spatial.to(dtype=transformer_dtype), source_id=source_id
    )
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise DCLRPreferenceBatchError(
            f"{label} patch_vae_latent must return (tokens, rotary)"
        )
    tokens, rotary = result
    tokens = _require_tensor(tokens, label=f"{label} tokens")
    if (
        not tokens.is_floating_point()
        or tokens.dtype != transformer_dtype
        or tuple(tokens.shape)
        != (1, expected_tokens, runtime_contract.PINNED_INNER_DIM)
        or tokens.device != spatial.device
        or tokens.requires_grad
        or tokens.grad_fn is not None
        or not bool(torch.isfinite(tokens).all().item())
    ):
        raise DCLRPreferenceBatchError(
            f"{label} tokens must be detached transformer-dtype "
            f"[1,{expected_tokens},{runtime_contract.PINNED_INNER_DIM}]"
        )
    rotary = _require_tensor(rotary, label=f"{label} rotary")
    if (
        rotary.dtype != torch.complex128
        or tuple(rotary.shape)
        != (1, 1, expected_tokens, runtime_contract.PINNED_ROPE_DIM)
        or rotary.device != spatial.device
        or rotary.requires_grad
        or rotary.grad_fn is not None
        or not bool(torch.isfinite(rotary).all().item())
    ):
        raise DCLRPreferenceBatchError(
            f"{label} rotary must be detached complex128 "
            f"[1,1,{expected_tokens},{runtime_contract.PINNED_ROPE_DIM}]"
        )
    return tokens, rotary


def build_packed_preference_batch(
    transformer: Any,
    *,
    normalized_source: torch.Tensor,
    flow_state: preference_objective.SharedPairFlowState,
) -> PackedPreferenceBatch:
    """Patch and pack ``[S,y+,S,y-]`` for one Bernini ``shared_step``.

    Source and candidate latents are required as normalized detached FP32
    tensors.  They are cast only at the patch-embedding boundary; rectified
    flow states and velocity supervision remain FP32.  ``flow_state`` owns one
    epsilon object and one sigma/timestep, so no API exists for candidate-wise
    noise or time drift.
    """

    state = _validate_flow_state(flow_state)
    source = _require_normalized_spatial(
        normalized_source,
        label="normalized_source",
        expected_shape=tuple(int(item) for item in state.winner_clean.shape),
    )
    if source.device != state.winner_clean.device:
        raise DCLRPreferenceBatchError(
            "normalized source and preference state devices differ"
        )
    dtype = _transformer_dtype(transformer)
    _, _, phases, height, width = (int(item) for item in source.shape)
    expected_tokens = (
        (phases // PATCH_SIZE[0])
        * (height // PATCH_SIZE[1])
        * (width // PATCH_SIZE[2])
    )

    source_tokens, source_rotary = _patch_one(
        transformer,
        source,
        source_id=SOURCE_ID,
        expected_tokens=expected_tokens,
        transformer_dtype=dtype,
        label="source",
    )
    winner_tokens, winner_rotary = _patch_one(
        transformer,
        state.winner_x_sigma,
        source_id=TARGET_SOURCE_ID,
        expected_tokens=expected_tokens,
        transformer_dtype=dtype,
        label="winner target",
    )
    loser_tokens, loser_rotary = _patch_one(
        transformer,
        state.loser_x_sigma,
        source_id=TARGET_SOURCE_ID,
        expected_tokens=expected_tokens,
        transformer_dtype=dtype,
        label="loser target",
    )
    if not _same_tensor(winner_rotary, loser_rotary):
        raise DCLRPreferenceBatchError(
            "winner/loser target rotary must be exactly identical"
        )
    if _same_tensor(source_rotary, winner_rotary):
        raise DCLRPreferenceBatchError(
            "source_id=1 and target source_id=0 produced identical rotary; "
            "source-id rotary is not active"
        )

    noisy_latents = torch.cat(
        (source_tokens, winner_tokens, source_tokens, loser_tokens), dim=1
    )
    rotary_embs = torch.cat(
        (source_rotary, winner_rotary, source_rotary, loser_rotary), dim=2
    )
    candidate_length = 2 * expected_tokens
    timesteps = torch.cat((state.timestep, state.timestep), dim=0)
    candidate_selector = torch.cat(
        (
            torch.zeros(
                expected_tokens, dtype=torch.bool, device=source.device
            ),
            torch.ones(
                expected_tokens, dtype=torch.bool, device=source.device
            ),
        ),
        dim=0,
    )
    target_selector = torch.cat(
        (candidate_selector, candidate_selector), dim=0
    )
    target_true_velocity = torch.cat(
        (
            pack_spatial_velocity(state.winner_true_velocity),
            pack_spatial_velocity(state.loser_true_velocity),
        ),
        dim=1,
    )
    logical_cu = torch.tensor(
        [0, candidate_length, 2 * candidate_length],
        dtype=torch.int32,
        device=source.device,
    )
    batch = PackedPreferenceBatch(
        normalized_source=source,
        flow_state=state,
        noisy_latents=noisy_latents,
        rotary_embs=rotary_embs,
        timesteps=timesteps,
        batch_vae_seqlen=(candidate_length, candidate_length),
        target_selector=target_selector,
        candidate_target_selector=candidate_selector,
        target_true_velocity=target_true_velocity,
        logical_self_attention_cu_seqlens=logical_cu,
        source_token_count=expected_tokens,
        target_token_count=expected_tokens,
        source_id=SOURCE_ID,
        target_source_id=TARGET_SOURCE_ID,
        sigma_float32_bits_hex=_fp32_bits(state.sigma, label="flow_state.sigma"),
        timestep_float32_bits_hex=_fp32_bits(
            state.timestep, label="flow_state.timestep"
        ),
    )
    return validate_packed_preference_batch(batch)


def validate_packed_preference_batch(
    value: Any,
) -> PackedPreferenceBatch:
    """Revalidate a packed object before it reaches ``shared_step``."""

    if not isinstance(value, PackedPreferenceBatch):
        raise DCLRPreferenceBatchError(
            "value must be a PackedPreferenceBatch"
        )
    state = _validate_flow_state(value.flow_state)
    source = _require_normalized_spatial(
        value.normalized_source,
        label="normalized_source",
        expected_shape=tuple(int(item) for item in state.winner_clean.shape),
    )
    n = int(value.source_token_count)
    if (
        n <= 0
        or int(value.target_token_count) != n
        or value.source_id != SOURCE_ID
        or value.target_source_id != TARGET_SOURCE_ID
    ):
        raise DCLRPreferenceBatchError(
            "packed batch must use equal source/target N and source_id=1,target=0"
        )
    expected_n = (
        int(source.shape[2])
        * (int(source.shape[3]) // PATCH_SIZE[1])
        * (int(source.shape[4]) // PATCH_SIZE[2])
    )
    if n != expected_n:
        raise DCLRPreferenceBatchError(
            "packed token count differs from normalized spatial geometry"
        )
    candidate_length = 2 * n
    total = 2 * candidate_length
    visual = _require_tensor(value.noisy_latents, label="noisy_latents")
    if (
        not visual.is_floating_point()
        or tuple(visual.shape)
        != (1, total, runtime_contract.PINNED_INNER_DIM)
        or visual.device != source.device
        or visual.requires_grad
        or visual.grad_fn is not None
        or not bool(torch.isfinite(visual).all().item())
    ):
        raise DCLRPreferenceBatchError(
            f"noisy_latents must be detached [1,{total},"
            f"{runtime_contract.PINNED_INNER_DIM}] on source device"
        )
    rotary = _require_tensor(value.rotary_embs, label="rotary_embs")
    if (
        rotary.dtype != torch.complex128
        or tuple(rotary.shape)
        != (1, 1, total, runtime_contract.PINNED_ROPE_DIM)
        or rotary.device != source.device
        or rotary.requires_grad
        or rotary.grad_fn is not None
        or not bool(torch.isfinite(rotary).all().item())
    ):
        raise DCLRPreferenceBatchError(
            f"rotary_embs must be detached complex128 [1,1,{total},"
            f"{runtime_contract.PINNED_ROPE_DIM}]"
        )
    if tuple(value.batch_vae_seqlen) != (
        candidate_length,
        candidate_length,
    ):
        raise DCLRPreferenceBatchError(
            "batch_vae_seqlen must equal [2N,2N], never [N,N,N,N]"
        )

    timesteps = _require_finite_detached_fp32(
        value.timesteps, label="timesteps"
    )
    expected_timesteps = torch.cat(
        (state.timestep, state.timestep), dim=0
    )
    if not _same_tensor(timesteps, expected_timesteps):
        raise DCLRPreferenceBatchError(
            "logical candidates do not share the exact same timestep bits"
        )
    expected_sigma_bits = _fp32_bits(state.sigma, label="flow_state.sigma")
    expected_timestep_bits = _fp32_bits(
        state.timestep, label="flow_state.timestep"
    )
    if (
        value.sigma_float32_bits_hex != expected_sigma_bits
        or value.timestep_float32_bits_hex != expected_timestep_bits
    ):
        raise DCLRPreferenceBatchError(
            "stored sigma/timestep FP32 bit evidence differs from flow state"
        )

    expected_candidate_selector = torch.cat(
        (
            torch.zeros(n, dtype=torch.bool, device=source.device),
            torch.ones(n, dtype=torch.bool, device=source.device),
        )
    )
    expected_selector = torch.cat(
        (expected_candidate_selector, expected_candidate_selector)
    )
    if not _same_tensor(
        value.candidate_target_selector, expected_candidate_selector
    ):
        raise DCLRPreferenceBatchError(
            "candidate selector must be exactly 0N+1N"
        )
    if not _same_tensor(value.target_selector, expected_selector):
        raise DCLRPreferenceBatchError(
            "packed selector must be exactly 0N+1N+0N+1N"
        )
    expected_velocity = torch.cat(
        (
            pack_spatial_velocity(state.winner_true_velocity),
            pack_spatial_velocity(state.loser_true_velocity),
        ),
        dim=1,
    )
    if not _same_tensor(value.target_true_velocity, expected_velocity):
        raise DCLRPreferenceBatchError(
            "packed target velocity is not winner then loser target-tail FP32"
        )
    expected_cu = torch.tensor(
        [0, candidate_length, total],
        dtype=torch.int32,
        device=source.device,
    )
    if not _same_tensor(
        value.logical_self_attention_cu_seqlens, expected_cu
    ):
        raise DCLRPreferenceBatchError(
            "logical self-attention cu_seqlens must isolate [0,2N,4N]"
        )

    # The source prefix and its source-id RoPE are repeated exactly.  Target
    # RoPE is geometry-only within source_id=0 and must also match exactly.
    if not _same_tensor(visual[:, :n, :], visual[:, 2 * n : 3 * n, :]):
        raise DCLRPreferenceBatchError(
            "winner/loser do not contain the exact same source-token prefix"
        )
    if not _same_tensor(
        rotary[:, :, :n, :], rotary[:, :, 2 * n : 3 * n, :]
    ):
        raise DCLRPreferenceBatchError(
            "winner/loser source-prefix rotary differs"
        )
    if not _same_tensor(
        rotary[:, :, n : 2 * n, :], rotary[:, :, 3 * n : 4 * n, :]
    ):
        raise DCLRPreferenceBatchError(
            "winner/loser target rotary differs"
        )
    if _same_tensor(
        rotary[:, :, :n, :], rotary[:, :, n : 2 * n, :]
    ):
        raise DCLRPreferenceBatchError(
            "source and target rotary are identical; source-id routing vanished"
        )
    if value.winner_epsilon is not state.epsilon:
        raise DCLRPreferenceBatchError("winner lost the shared epsilon object")
    if value.loser_epsilon is not state.epsilon:
        raise DCLRPreferenceBatchError("loser lost the shared epsilon object")
    if value.winner_epsilon is not value.loser_epsilon:
        raise DCLRPreferenceBatchError(
            "winner and loser must share one literal epsilon object"
        )
    return value


def shared_step_visual_kwargs(
    batch: PackedPreferenceBatch,
) -> dict[str, Any]:
    """Return only the pinned visual/time subset of ``shared_step`` kwargs.

    The caller must add ``model_id``, packed positive ``cond_embeds``, and
    ``batch_text_seqlen`` with exactly two entries.  Cu-seqlens are deliberately
    not passed: pinned Bernini derives them internally from these two visual
    lengths, yielding the boundary declared on ``batch``.
    """

    checked = validate_packed_preference_batch(batch)
    return {
        "noisy_latents": checked.noisy_latents,
        "timesteps": checked.timesteps,
        "rotary_embs": checked.rotary_embs,
        "batch_vae_seqlen": list(checked.batch_vae_seqlen),
    }


__all__ = [
    "DCLRPreferenceBatchError",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "LOGICAL_CANDIDATES",
    "PATCH_SIZE",
    "PackedPreferenceBatch",
    "SCHEMA_VERSION",
    "SOURCE_ID",
    "TARGET_SOURCE_ID",
    "build_packed_preference_batch",
    "pack_spatial_velocity",
    "shared_step_visual_kwargs",
    "validate_packed_preference_batch",
]
