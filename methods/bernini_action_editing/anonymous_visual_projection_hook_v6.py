#!/usr/bin/env python3
"""Read-only anonymous visual projection hook for the Bernini V6 probe.

Only ``attn1`` at blocks 6/12/18/24 is wrapped.  The official processor is
delegated exactly once and its exact output object is returned.  During that
same call the wrapper observes the post-RoPE visual query and the official
attn1 hidden intermediate, projects observer-owned raw clones through fixed
signed-DCT column-orthogonal maps, and immediately zeroizes those clones.

No caption token, text K/V, role phrase, semantic slot, collective, decoder,
route, optimizer, or parameter update is available in this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch

import self_generated_anonymous_object_registry_v6 as registry


METHOD = "bernini-anonymous-visual-projection-hook-v6"
SCHEMA_VERSION = "bernini-anonymous-visual-projection-hook-v6"
OFFICIAL_TRANSFORMER_SOURCE_SHA256 = (
    "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223"
)
OFFICIAL_PROCESSOR_MODULE = "bernini.models.transformer_wan"
OFFICIAL_PROCESSOR_CLASS = "WanAttnProcessor2_0"
BLOCKS = registry.BLOCKS
WORLD_SIZE = 4
TOTAL_HEADS = 12
LOCAL_HEADS = TOTAL_HEADS // WORLD_SIZE
HEAD_DIM = 128
MODEL_WIDTH = TOTAL_HEADS * HEAD_DIM
QUERY_SKETCH_DIM = 16
HIDDEN_SKETCH_DIM = 16
PROJECTION_SEED = 2026082306
EXPECTED_BLOCK_COUNT = 30
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AnonymousVisualProjectionHookV6Error(RuntimeError):
    """A frozen-forward, projection, ownership, or WORLD4 gate failed."""


def validate_official_transformer_source_file_v6(source: Any) -> Path:
    """Resolve and byte-bind the actual official processor source file."""

    if not isinstance(source, (str, os.PathLike)):
        raise AnonymousVisualProjectionHookV6Error(
            "official processor source path is absent"
        )
    candidate = Path(source)
    try:
        canonical = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise AnonymousVisualProjectionHookV6Error(
            "official processor source cannot be resolved"
        ) from error
    if (
        not candidate.is_absolute()
        or candidate != canonical
        or candidate.is_symlink()
        or not candidate.is_file()
    ):
        raise AnonymousVisualProjectionHookV6Error(
            "official processor source is not a canonical plain file"
        )
    try:
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError as error:
        raise AnonymousVisualProjectionHookV6Error(
            "official processor source cannot be read"
        ) from error
    if digest != OFFICIAL_TRANSFORMER_SOURCE_SHA256:
        raise AnonymousVisualProjectionHookV6Error(
            "official processor source SHA-256 differs"
        )
    return canonical


def _canonical_digest(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise AnonymousVisualProjectionHookV6Error(
            "value is not canonical finite JSON"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _zeroize(values: Sequence[torch.Tensor]) -> None:
    with torch.inference_mode():
        for value in values:
            if isinstance(value, torch.Tensor) and value.device.type != "meta":
                value.zero_()


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise AnonymousVisualProjectionHookV6Error("tensor digest input differs")
    cpu = value.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"shape": list(cpu.shape), "dtype": str(cpu.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _frequency_order(input_dim: int, output_dim: int, seed: int) -> tuple[int, ...]:
    rows = []
    for frequency in range(input_dim):
        digest = hashlib.sha256(
            f"v6-frequency:{seed}:{input_dim}:{frequency}".encode("ascii")
        ).digest()
        rows.append((digest, frequency))
    rows.sort()
    return tuple(frequency for _digest, frequency in rows[:output_dim])


def fixed_signed_dct_projection(
    input_dim: int, output_dim: int, *, seed: int
) -> torch.Tensor:
    """Return a deterministic column-orthogonal float32 projection.

    A row-wise Rademacher sign preserves DCT column orthogonality.  Frequencies
    and signs are selected from SHA-256, avoiding mutable RNG state.
    """

    if (
        isinstance(input_dim, bool)
        or isinstance(output_dim, bool)
        or not isinstance(input_dim, int)
        or not isinstance(output_dim, int)
        or not 1 <= output_dim <= input_dim
    ):
        raise AnonymousVisualProjectionHookV6Error(
            "projection dimensions differ"
        )
    frequencies = _frequency_order(input_dim, output_dim, seed)
    index = torch.arange(input_dim, dtype=torch.float64).reshape(-1, 1)
    frequency = torch.tensor(frequencies, dtype=torch.float64).reshape(1, -1)
    matrix = torch.cos(math.pi * (index + 0.5) * frequency / float(input_dim))
    normalization = torch.full(
        (output_dim,), math.sqrt(2.0 / float(input_dim)), dtype=torch.float64
    )
    for column, value in enumerate(frequencies):
        if value == 0:
            normalization[column] = math.sqrt(1.0 / float(input_dim))
    matrix = matrix * normalization
    signs = []
    for row in range(input_dim):
        digest = hashlib.sha256(
            f"v6-sign:{seed}:{input_dim}:{row}".encode("ascii")
        ).digest()
        signs.append(1.0 if digest[0] & 1 else -1.0)
    matrix = matrix * torch.tensor(signs, dtype=torch.float64).reshape(-1, 1)
    gram = matrix.transpose(0, 1) @ matrix
    if not torch.allclose(
        gram,
        torch.eye(output_dim, dtype=torch.float64),
        atol=1.0e-10,
        rtol=1.0e-10,
    ):
        raise AnonymousVisualProjectionHookV6Error(
            "fixed projection is not column orthogonal"
        )
    return matrix.float().contiguous()


@dataclass(frozen=True)
class ProjectionAuthorityV6:
    query: torch.Tensor
    hidden: torch.Tensor
    digest: str

    @classmethod
    def create(cls) -> "ProjectionAuthorityV6":
        query = fixed_signed_dct_projection(
            MODEL_WIDTH, QUERY_SKETCH_DIM, seed=PROJECTION_SEED
        )
        hidden = fixed_signed_dct_projection(
            MODEL_WIDTH, HIDDEN_SKETCH_DIM, seed=PROJECTION_SEED + 1
        )
        value = {
            "kind": "seeded_signed_dct_column_orthogonal",
            "seed": PROJECTION_SEED,
            "query_shape": list(query.shape),
            "hidden_shape": list(hidden.shape),
            "query_sha256": _tensor_sha256(query),
            "hidden_sha256": _tensor_sha256(hidden),
        }
        return cls(query, hidden, _canonical_digest(value))

    def validate(self) -> None:
        if (
            tuple(self.query.shape) != (MODEL_WIDTH, QUERY_SKETCH_DIM)
            or tuple(self.hidden.shape) != (MODEL_WIDTH, HIDDEN_SKETCH_DIM)
            or self.query.dtype != torch.float32
            or self.hidden.dtype != torch.float32
            or self.query.device.type != "cpu"
            or self.hidden.device.type != "cpu"
            or _SHA256_RE.fullmatch(self.digest) is None
        ):
            raise AnonymousVisualProjectionHookV6Error(
                "projection authority differs"
            )
        for value, width in (
            (self.query, QUERY_SKETCH_DIM),
            (self.hidden, HIDDEN_SKETCH_DIM),
        ):
            gram = value.double().transpose(0, 1) @ value.double()
            if not torch.allclose(
                gram,
                torch.eye(width, dtype=torch.float64),
                atol=2.0e-6,
                rtol=2.0e-6,
            ):
                raise AnonymousVisualProjectionHookV6Error(
                    "float32 projection orthogonality differs"
                )

    def query_rank_slice(self, rank: int, device: torch.device) -> torch.Tensor:
        self.validate()
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < WORLD_SIZE:
            raise AnonymousVisualProjectionHookV6Error("projection rank differs")
        start = rank * LOCAL_HEADS * HEAD_DIM
        stop = start + LOCAL_HEADS * HEAD_DIM
        return self.query[start:stop].to(device=device).contiguous()

    def hidden_matrix(self, device: torch.device) -> torch.Tensor:
        self.validate()
        return self.hidden.to(device=device).contiguous()

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "kind": "seeded_signed_dct_column_orthogonal",
            "seed": PROJECTION_SEED,
            "query_input_dim": MODEL_WIDTH,
            "query_output_dim": QUERY_SKETCH_DIM,
            "hidden_input_dim": MODEL_WIDTH,
            "hidden_output_dim": HIDDEN_SKETCH_DIM,
            "query_sha256": _tensor_sha256(self.query),
            "hidden_sha256": _tensor_sha256(self.hidden),
            "digest": self.digest,
        }
        return value


@dataclass(frozen=True)
class AnonymousCaptureIdentityV6:
    appearance_id: str
    arm: str
    sigma_band: str
    step_index: int
    state_sha256: str
    timestep_sha256: str
    rotary_sha256: str
    patch_height: int
    patch_width: int

    def __post_init__(self) -> None:
        if self.appearance_id not in registry.APPEARANCE_IDS or self.arm not in registry.ARMS:
            raise AnonymousVisualProjectionHookV6Error("capture identity differs")
        if self.sigma_band not in registry.SIGMA_CELL_INDICES or (
            self.step_index != registry.SIGMA_CELL_INDICES[self.sigma_band]
        ):
            raise AnonymousVisualProjectionHookV6Error("capture sigma cell differs")
        if (self.patch_height, self.patch_width) != (
            registry.PATCH_HEIGHT,
            registry.PATCH_WIDTH,
        ):
            raise AnonymousVisualProjectionHookV6Error("capture patch grid differs")
        for value in (self.state_sha256, self.timestep_sha256, self.rotary_sha256):
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise AnonymousVisualProjectionHookV6Error("capture digest differs")

    @property
    def key(self) -> tuple[str, str, str, int, str, str, str]:
        return (
            self.appearance_id,
            self.arm,
            self.sigma_band,
            self.step_index,
            self.state_sha256,
            self.timestep_sha256,
            self.rotary_sha256,
        )


@dataclass(frozen=True)
class World4VisualLayoutV6:
    rank: int
    patch_height: int = registry.PATCH_HEIGHT
    patch_width: int = registry.PATCH_WIDTH

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or not 0 <= self.rank < WORLD_SIZE:
            raise AnonymousVisualProjectionHookV6Error("WORLD4 rank differs")
        if (self.patch_height, self.patch_width) != (
            registry.PATCH_HEIGHT,
            registry.PATCH_WIDTH,
        ):
            raise AnonymousVisualProjectionHookV6Error("WORLD4 grid differs")

    @property
    def global_tokens(self) -> int:
        return registry.PHASES * self.patch_height * self.patch_width

    @property
    def padded_local_tokens(self) -> int:
        return math.ceil(self.global_tokens / WORLD_SIZE)

    @property
    def global_start(self) -> int:
        return self.rank * self.padded_local_tokens

    @property
    def global_stop(self) -> int:
        return min(self.global_tokens, self.global_start + self.padded_local_tokens)

    @property
    def valid_local_tokens(self) -> int:
        return max(0, self.global_stop - self.global_start)


@dataclass(frozen=True)
class AnonymousRankInvocationV6:
    identity: AnonymousCaptureIdentityV6
    layout: World4VisualLayoutV6
    projection_digest: str

    def __post_init__(self) -> None:
        if (
            self.identity.patch_height != self.layout.patch_height
            or self.identity.patch_width != self.layout.patch_width
            or _SHA256_RE.fullmatch(self.projection_digest) is None
        ):
            raise AnonymousVisualProjectionHookV6Error(
                "rank invocation authority differs"
            )

    @property
    def key(self) -> tuple[Any, ...]:
        return (*self.identity.key, self.layout.rank)


def project_owned_raw_and_zeroize(
    raw: torch.Tensor, projection: torch.Tensor, *, label: str
) -> torch.Tensor:
    """Project one observer-owned raw clone and scrub it on every exit."""

    if (
        not isinstance(raw, torch.Tensor)
        or not isinstance(projection, torch.Tensor)
        or raw.device.type == "meta"
        or projection.device != raw.device
        or raw.shape[-1] != projection.shape[0]
        or projection.dtype != torch.float32
    ):
        _zeroize((raw,) if isinstance(raw, torch.Tensor) else ())
        raise AnonymousVisualProjectionHookV6Error(f"{label} projection ABI differs")
    result: Optional[torch.Tensor] = None
    try:
        if not bool(torch.isfinite(raw).all().item()):
            raise AnonymousVisualProjectionHookV6Error(f"{label} raw is non-finite")
        result = torch.matmul(raw.float(), projection).detach().contiguous()
        if not bool(torch.isfinite(result).all().item()):
            raise AnonymousVisualProjectionHookV6Error(
                f"{label} sketch is non-finite"
            )
        return result
    finally:
        _zeroize((raw,))


@dataclass
class ProjectedVisualRankShardV6:
    invocation: AnonymousRankInvocationV6
    block_index: int
    query_partial: torch.Tensor
    hidden_local: torch.Tensor
    consumed: bool = False

    def validate(self) -> None:
        layout = self.invocation.layout
        if self.block_index not in BLOCKS or self.consumed:
            raise AnonymousVisualProjectionHookV6Error("rank shard identity differs")
        expected_query = (1, layout.global_tokens, QUERY_SKETCH_DIM)
        expected_hidden = (1, layout.padded_local_tokens, HIDDEN_SKETCH_DIM)
        if (
            tuple(self.query_partial.shape) != expected_query
            or tuple(self.hidden_local.shape) != expected_hidden
            or self.query_partial.dtype != torch.float32
            or self.hidden_local.dtype != torch.float32
            or self.query_partial.device != self.hidden_local.device
            or not self.query_partial.is_contiguous()
            or not self.hidden_local.is_contiguous()
            or not bool(torch.isfinite(self.query_partial).all().item())
            or not bool(torch.isfinite(self.hidden_local).all().item())
        ):
            raise AnonymousVisualProjectionHookV6Error("rank sketch geometry differs")

    def zeroize(self) -> None:
        _zeroize((self.query_partial, self.hidden_local))
        self.consumed = True

    def collective_payload_and_zeroize(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
        self.validate()
        owned: list[torch.Tensor] = []
        succeeded = False
        try:
            query = self.query_partial.detach().clone(memory_format=torch.contiguous_format)
            owned.append(query)
            hidden = self.hidden_local.detach().clone(memory_format=torch.contiguous_format)
            owned.append(hidden)
            metadata = {
                "rank": self.invocation.layout.rank,
                "block_index": self.block_index,
                "identity_key": list(self.invocation.identity.key),
                "projection_digest": self.invocation.projection_digest,
                "valid_local_tokens": self.invocation.layout.valid_local_tokens,
            }
            self.zeroize()
            succeeded = True
            return query, hidden, metadata
        finally:
            if not succeeded:
                _zeroize(owned)
                self.zeroize()


@dataclass
class ProjectedVisualCaptureV6:
    identity: AnonymousCaptureIdentityV6
    block_index: int
    projection_digest: str
    query_sketch: torch.Tensor
    hidden_sketch: torch.Tensor
    consumed: bool = False

    def validate(self) -> None:
        expected_q = (
            1,
            registry.PHASES,
            registry.PATCHES,
            QUERY_SKETCH_DIM,
        )
        expected_h = (
            1,
            registry.PHASES,
            registry.PATCHES,
            HIDDEN_SKETCH_DIM,
        )
        if self.block_index not in BLOCKS or self.consumed:
            raise AnonymousVisualProjectionHookV6Error("global sketch identity differs")
        if (
            tuple(self.query_sketch.shape) != expected_q
            or tuple(self.hidden_sketch.shape) != expected_h
            or self.query_sketch.dtype != torch.float32
            or self.hidden_sketch.dtype != torch.float32
            or self.query_sketch.device != self.hidden_sketch.device
            or not self.query_sketch.is_contiguous()
            or not self.hidden_sketch.is_contiguous()
            or not bool(torch.isfinite(self.query_sketch).all().item())
            or not bool(torch.isfinite(self.hidden_sketch).all().item())
        ):
            raise AnonymousVisualProjectionHookV6Error("global sketch geometry differs")

    def zeroize(self) -> None:
        _zeroize((self.query_sketch, self.hidden_sketch))
        self.consumed = True


class InMemoryProjectedRankBankV6:
    def __init__(self) -> None:
        self._rows: dict[tuple[Any, ...], dict[int, ProjectedVisualRankShardV6]] = {}
        self.capture_count = 0
        self.taken_count = 0

    @contextmanager
    def observe(self, invocation: AnonymousRankInvocationV6) -> Iterator[None]:
        if not isinstance(invocation, AnonymousRankInvocationV6) or invocation.key in self._rows:
            raise AnonymousVisualProjectionHookV6Error("rank observation differs")
        token: Token = _ACTIVE_CAPTURE.set((self, invocation))
        self._rows[invocation.key] = {}
        succeeded = False
        try:
            yield
            rows = self._rows.get(invocation.key, {})
            if tuple(sorted(rows)) != BLOCKS:
                raise AnonymousVisualProjectionHookV6Error(
                    "rank forward did not capture all V6 blocks"
                )
            succeeded = True
        finally:
            _ACTIVE_CAPTURE.reset(token)
            if not succeeded:
                rows = self._rows.pop(invocation.key, {})
                for row in rows.values():
                    row.zeroize()

    def capture(self, shard: ProjectedVisualRankShardV6) -> None:
        active = _ACTIVE_CAPTURE.get()
        if active is None or active[0] is not self:
            shard.zeroize()
            raise AnonymousVisualProjectionHookV6Error("capture arrived outside context")
        invocation = active[1]
        rows = self._rows[invocation.key]
        if shard.invocation != invocation or shard.block_index in rows:
            shard.zeroize()
            raise AnonymousVisualProjectionHookV6Error("duplicate or foreign sketch")
        try:
            shard.validate()
        except BaseException:
            shard.zeroize()
            raise
        rows[shard.block_index] = shard
        self.capture_count += 1

    def take_rank(
        self, invocation: AnonymousRankInvocationV6
    ) -> tuple[ProjectedVisualRankShardV6, ...]:
        rows = self._rows.pop(invocation.key, None)
        if rows is None or tuple(sorted(rows)) != BLOCKS:
            if rows is not None:
                for row in rows.values():
                    row.zeroize()
            raise AnonymousVisualProjectionHookV6Error("rank sketch group is absent")
        result = tuple(rows[block] for block in BLOCKS)
        try:
            for row in result:
                row.validate()
        except BaseException:
            for row in result:
                row.zeroize()
            raise
        self.taken_count += 1
        return result

    def abort(self, invocation: AnonymousRankInvocationV6) -> None:
        rows = self._rows.pop(invocation.key, {})
        for row in rows.values():
            row.zeroize()

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "projected_rank_shard_count": self.capture_count,
            "taken_rank_invocation_count": self.taken_count,
            "resident_rank_invocation_count": len(self._rows),
            "raw_query_or_hidden_stored_in_bank": False,
            "caption_role_partition_stored_in_bank": False,
        }
        return {**value, "digest": _canonical_digest(value)}


_ACTIVE_CAPTURE: ContextVar[
    Optional[tuple[InMemoryProjectedRankBankV6, AnonymousRankInvocationV6]]
] = ContextVar("bernini_anonymous_visual_projection_v6", default=None)


def _processor_kwargs(values: Mapping[str, Any]) -> Mapping[str, Any]:
    names = (
        "encoder_hidden_states",
        "attention_mask",
        "rotary_emb",
        "batch_image_vae_seqlen",
        "text_features_length",
        "origin_hidden_states_seq_len",
        "split_hidden_states_seq_len",
        "cu_seqlens_q_cache",
        "max_seqlen_q_cache",
        "cu_seqlens_k_cross_cache",
        "cu_seqlens_q_cross_cache",
        "max_seqlen_k_cross_cache",
        "max_seqlen_q_cross_cache",
    )
    return {name: values[name] for name in names}


def _official_processor(value: Any) -> tuple[Any, Path]:
    value_type = type(value)
    if (
        not callable(value)
        or not callable(getattr(value, "_project_qkv", None))
        or value_type.__module__ != OFFICIAL_PROCESSOR_MODULE
        or value_type.__name__ != OFFICIAL_PROCESSOR_CLASS
        or "_project_qkv" in getattr(value, "__dict__", {})
    ):
        raise AnonymousVisualProjectionHookV6Error(
            "attention processor is not pristine official ABI"
        )
    try:
        source = inspect.getsourcefile(value_type)
    except (OSError, TypeError) as error:
        raise AnonymousVisualProjectionHookV6Error(
            "official processor source inspection failed"
        ) from error
    canonical = validate_official_transformer_source_file_v6(source)
    return value, canonical


def _validate_frozen_attention(attn: Any) -> None:
    if isinstance(attn, torch.nn.Module) and (
        attn.training or any(parameter.requires_grad for parameter in attn.parameters())
    ):
        raise AnonymousVisualProjectionHookV6Error(
            "V6 observer requires frozen eval attention"
        )


def _delegate_with_query_clone(
    base_processor: Any,
    attn: Any,
    hidden_states: torch.Tensor,
    kwargs: Mapping[str, Any],
) -> tuple[Any, torch.Tensor]:
    original = base_processor._project_qkv
    captured: list[torch.Tensor] = []

    def intercept(*args: Any, **inner_kwargs: Any):
        if captured:
            raise AnonymousVisualProjectionHookV6Error(
                "official QKV projection executed more than once"
            )
        query, key, value = original(*args, **inner_kwargs)
        raw_query = query.detach().clone(memory_format=torch.contiguous_format)
        captured.append(raw_query)
        return query, key, value

    base_processor._project_qkv = intercept
    succeeded = False
    try:
        output = base_processor(attn, hidden_states, **dict(kwargs))
        try:
            delattr(base_processor, "_project_qkv")
        except AttributeError as error:
            raise AnonymousVisualProjectionHookV6Error(
                "temporary projection interception was lost"
            ) from error
        if len(captured) != 1:
            raise AnonymousVisualProjectionHookV6Error(
                "official post-RoPE query was not captured exactly once"
            )
        succeeded = True
        return output, captured[0]
    finally:
        if "_project_qkv" in getattr(base_processor, "__dict__", {}):
            try:
                delattr(base_processor, "_project_qkv")
            except Exception:
                pass
        if not succeeded:
            _zeroize(captured)


class AnonymousAttn1ProjectionObserverV6:
    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        rank_bank: InMemoryProjectedRankBankV6,
        projection: ProjectionAuthorityV6,
    ) -> None:
        self.base_processor, official_source = _official_processor(base_processor)
        self.official_processor_source_path = str(official_source)
        if block_index not in BLOCKS or not isinstance(
            rank_bank, InMemoryProjectedRankBankV6
        ):
            raise AnonymousVisualProjectionHookV6Error("hook scope differs")
        projection.validate()
        self.block_index = block_index
        self.rank_bank = rank_bank
        self.projection = projection
        self.base_calls = 0
        self.observer_calls = 0

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
    ) -> Any:
        kwargs = _processor_kwargs(locals())
        active = _ACTIVE_CAPTURE.get()
        if active is None:
            output = self.base_processor(attn, hidden_states, **dict(kwargs))
            self.base_calls += 1
            return output
        if active[0] is not self.rank_bank:
            raise AnonymousVisualProjectionHookV6Error("hook/bank ownership differs")
        invocation = active[1]
        layout = invocation.layout
        _validate_frozen_attention(attn)
        if (
            encoder_hidden_states is not None
            or attention_mask is not None
            or not isinstance(rotary_emb, torch.Tensor)
            or tuple(hidden_states.shape) != (1, layout.padded_local_tokens, MODEL_WIDTH)
            or hidden_states.requires_grad
            or hidden_states.grad_fn is not None
            or origin_hidden_states_seq_len != layout.global_tokens
            or split_hidden_states_seq_len != layout.padded_local_tokens
        ):
            raise AnonymousVisualProjectionHookV6Error("attn1 call ABI differs")
        output, raw_query = _delegate_with_query_clone(
            self.base_processor, attn, hidden_states, kwargs
        )
        self.base_calls += 1
        raw_hidden: Optional[torch.Tensor] = None
        query_sketch: Optional[torch.Tensor] = None
        hidden_sketch: Optional[torch.Tensor] = None
        try:
            if (
                not isinstance(output, torch.Tensor)
                or tuple(output.shape) != (1, layout.padded_local_tokens, MODEL_WIDTH)
                or tuple(raw_query.shape)
                != (1, layout.global_tokens, LOCAL_HEADS, HEAD_DIM)
            ):
                raise AnonymousVisualProjectionHookV6Error(
                    "official query/hidden geometry differs"
                )
            raw_hidden = output.detach().clone(memory_format=torch.contiguous_format)
            query_sketch = project_owned_raw_and_zeroize(
                raw_query.reshape(1, layout.global_tokens, LOCAL_HEADS * HEAD_DIM),
                self.projection.query_rank_slice(layout.rank, raw_query.device),
                label="post-RoPE visual query",
            )
            hidden_sketch = project_owned_raw_and_zeroize(
                raw_hidden,
                self.projection.hidden_matrix(raw_hidden.device),
                label="attn1 hidden intermediate",
            )
            shard = ProjectedVisualRankShardV6(
                invocation,
                self.block_index,
                query_sketch,
                hidden_sketch,
            )
            self.rank_bank.capture(shard)
            query_sketch = None
            hidden_sketch = None
        finally:
            # ``raw_query`` may already have been scrubbed through its reshape.
            _zeroize(
                tuple(
                    value
                    for value in (raw_query, raw_hidden, query_sketch, hidden_sketch)
                    if isinstance(value, torch.Tensor)
                )
            )
        self.observer_calls += 1
        return output


@dataclass
class AnonymousVisualProjectionHookHandleV6:
    transformer: Any
    wrappers: tuple[AnonymousAttn1ProjectionObserverV6, ...]
    originals: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        for block_index, wrapper, original in zip(BLOCKS, self.wrappers, self.originals):
            block = self.transformer.blocks[block_index]
            if block.attn1.processor is not wrapper:
                raise AnonymousVisualProjectionHookV6Error(
                    "installed V6 hook changed behind handle"
                )
            _set_processor(block.attn1, original)
        self.restored = True

    def receipt(self) -> Mapping[str, Any]:
        source_paths = {
            wrapper.official_processor_source_path for wrapper in self.wrappers
        }
        if len(source_paths) != 1:
            raise AnonymousVisualProjectionHookV6Error(
                "installed official processor source paths differ"
            )
        value = {
            "schema_version": SCHEMA_VERSION,
            "blocks": list(BLOCKS),
            "official_transformer_source_sha256": OFFICIAL_TRANSFORMER_SOURCE_SHA256,
            "actual_official_processor_source_canonical_path": next(
                iter(source_paths)
            ),
            "actual_official_processor_source_sha256_verified": True,
            "official_processor_delegated_once": True,
            "official_output_forwarded_same_object": True,
            "post_rope_visual_query_observed": True,
            "attn1_hidden_intermediate_observed": True,
            "text_kv_observed": False,
            "caption_token_partition_used": False,
            "semantic_role_inventory_used": False,
            "collective_calls_inside_attention_added": 0,
            "candidate_output_modified": False,
            "raw_observer_clones_zeroized_after_projection": True,
            "projection": dict(self.wrappers[0].projection.receipt()),
            "restored": self.restored,
            "gpu_launch_authorized": False,
            "scientific_claim_authorized": False,
        }
        return {**value, "digest": _canonical_digest(value)}

    def __enter__(self) -> "AnonymousVisualProjectionHookHandleV6":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


def _set_processor(attn: Any, processor: Any) -> None:
    setter = getattr(attn, "set_processor", None)
    if callable(setter):
        setter(processor)
    else:
        attn.processor = processor


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
                raise AnonymousVisualProjectionHookV6Error(
                    "Bernini transformer must expose exactly 30 blocks"
                )
            if isinstance(candidate, torch.nn.Module) and (
                candidate.training
                or any(parameter.requires_grad for parameter in candidate.parameters())
            ):
                raise AnonymousVisualProjectionHookV6Error(
                    "V6 observer requires a completely frozen eval transformer"
                )
            return candidate
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                pass
        for name in ("diff_dec", "transformer", "base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
    raise AnonymousVisualProjectionHookV6Error(
        "cannot resolve frozen Bernini transformer"
    )


def install_anonymous_visual_projection_hook_v6(
    model: Any,
    *,
    rank_bank: InMemoryProjectedRankBankV6,
    projection: Optional[ProjectionAuthorityV6] = None,
) -> AnonymousVisualProjectionHookHandleV6:
    if not isinstance(rank_bank, InMemoryProjectedRankBankV6):
        raise AnonymousVisualProjectionHookV6Error("rank bank differs")
    authority = projection or ProjectionAuthorityV6.create()
    authority.validate()
    transformer = _resolve_transformer(model)
    originals: list[Any] = []
    wrappers: list[AnonymousAttn1ProjectionObserverV6] = []
    installed: list[int] = []
    try:
        for block_index in BLOCKS:
            block = transformer.blocks[block_index]
            original = getattr(block.attn1, "processor", None)
            wrapper = AnonymousAttn1ProjectionObserverV6(
                original,
                block_index=block_index,
                rank_bank=rank_bank,
                projection=authority,
            )
            originals.append(original)
            wrappers.append(wrapper)
            _set_processor(block.attn1, wrapper)
            if block.attn1.processor is not wrapper:
                raise AnonymousVisualProjectionHookV6Error(
                    "V6 hook installation did not stick"
                )
            installed.append(block_index)
    except BaseException:
        for block_index, original in zip(installed, originals):
            _set_processor(transformer.blocks[block_index].attn1, original)
        raise
    return AnonymousVisualProjectionHookHandleV6(
        transformer, tuple(wrappers), tuple(originals)
    )


def reconstruct_projected_world4_block_v6(
    *,
    identity: AnonymousCaptureIdentityV6,
    block_index: int,
    projection_digest: str,
    query_rank_major: torch.Tensor,
    hidden_rank_major: torch.Tensor,
    rank_metadata: Sequence[Mapping[str, Any]],
) -> ProjectedVisualCaptureV6:
    """Join four projected shards; consume and scrub rank-major inputs."""

    result: Optional[ProjectedVisualCaptureV6] = None
    assembled: list[torch.Tensor] = []
    succeeded = False
    try:
        layout0 = World4VisualLayoutV6(0, identity.patch_height, identity.patch_width)
        if (
            block_index not in BLOCKS
            or _SHA256_RE.fullmatch(projection_digest) is None
            or tuple(query_rank_major.shape)
            != (WORLD_SIZE, 1, layout0.global_tokens, QUERY_SKETCH_DIM)
            or tuple(hidden_rank_major.shape)
            != (WORLD_SIZE, 1, layout0.padded_local_tokens, HIDDEN_SKETCH_DIM)
            or len(rank_metadata) != WORLD_SIZE
        ):
            raise AnonymousVisualProjectionHookV6Error(
                "rank-major projected geometry differs"
            )
        for rank, row in enumerate(rank_metadata):
            layout = World4VisualLayoutV6(rank, identity.patch_height, identity.patch_width)
            if (
                not isinstance(row, Mapping)
                or row.get("rank") != rank
                or row.get("block_index") != block_index
                or tuple(row.get("identity_key", ())) != identity.key
                or row.get("projection_digest") != projection_digest
                or row.get("valid_local_tokens") != layout.valid_local_tokens
            ):
                raise AnonymousVisualProjectionHookV6Error(
                    "rank-major projected metadata differs"
                )
        query = query_rank_major.sum(dim=0)
        assembled.append(query)
        query_contiguous = query.contiguous()
        if query_contiguous is not query:
            assembled.append(query_contiguous)
        query = query_contiguous
        hidden_parts = []
        for rank in range(WORLD_SIZE):
            layout = World4VisualLayoutV6(rank, identity.patch_height, identity.patch_width)
            hidden_parts.append(
                hidden_rank_major[rank, :, : layout.valid_local_tokens]
            )
        hidden = torch.cat(hidden_parts, dim=1)
        assembled.append(hidden)
        hidden_contiguous = hidden.contiguous()
        if hidden_contiguous is not hidden:
            assembled.append(hidden_contiguous)
        hidden = hidden_contiguous
        if tuple(hidden.shape) != (1, layout0.global_tokens, HIDDEN_SKETCH_DIM):
            raise AnonymousVisualProjectionHookV6Error(
                "joined hidden sequence differs"
            )
        query_shaped = query.reshape(
            1, registry.PHASES, registry.PATCHES, QUERY_SKETCH_DIM
        ).contiguous()
        if query_shaped is not query:
            assembled.append(query_shaped)
        query = query_shaped
        hidden_shaped = hidden.reshape(
            1, registry.PHASES, registry.PATCHES, HIDDEN_SKETCH_DIM
        ).contiguous()
        if hidden_shaped is not hidden:
            assembled.append(hidden_shaped)
        hidden = hidden_shaped
        result = ProjectedVisualCaptureV6(
            identity,
            block_index,
            projection_digest,
            query,
            hidden,
        )
        result.validate()
        assembled.clear()
        succeeded = True
        return result
    finally:
        _zeroize((query_rank_major, hidden_rank_major))
        if not succeeded:
            _zeroize(assembled)
            if result is not None:
                result.zeroize()


__all__ = [
    "AnonymousCaptureIdentityV6",
    "AnonymousRankInvocationV6",
    "AnonymousVisualProjectionHookHandleV6",
    "AnonymousVisualProjectionHookV6Error",
    "BLOCKS",
    "HIDDEN_SKETCH_DIM",
    "InMemoryProjectedRankBankV6",
    "METHOD",
    "MODEL_WIDTH",
    "PROJECTION_SEED",
    "ProjectedVisualCaptureV6",
    "ProjectedVisualRankShardV6",
    "ProjectionAuthorityV6",
    "QUERY_SKETCH_DIM",
    "SCHEMA_VERSION",
    "WORLD_SIZE",
    "World4VisualLayoutV6",
    "fixed_signed_dct_projection",
    "install_anonymous_visual_projection_hook_v6",
    "project_owned_raw_and_zeroize",
    "reconstruct_projected_world4_block_v6",
    "validate_official_transformer_source_file_v6",
]
