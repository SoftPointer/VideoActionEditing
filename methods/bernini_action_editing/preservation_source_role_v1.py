#!/usr/bin/env python3
"""Role-explicit source adapter for preservation-only WORLD2/SP2 training.

The pretext task has three visual roles: an ordered, appearance-corrupted
donor video; three independently VAE-encoded clean source frames; and the
noisy target row.  Bernini's base projections remain frozen.  Only a learned
role embedding and target-row residuals in early/mid ``attn1.to_q`` and
``attn1.to_out.0`` are trainable.

This module deliberately contains no action labels, edited targets, masks,
tracks, flows, or pose.  Preserving the order of a source-derived latent donor
does not prove that the donor carries semantic motion.  The optional
source-rich base is non-Gaussian for ``rho > 0`` and therefore has to be used
under the identical contract at training and inference.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn


SCHEMA_VERSION = "bernini-preservation-source-role-adapter-v1"
CONDITIONAL_BASE_SCHEMA = "bernini-source-self-conditional-base-v1"
LATENT_CHANNELS = 16
LATENT_PHASES = 21
REFERENCE_COUNT = 3
REFERENCE_RGB_INDICES = (0, 40, 80)
PATCH_SHAPE = (1, 2, 2)
PATCH_VALUES = LATENT_CHANNELS * math.prod(PATCH_SHAPE)
TOTAL_BLOCKS_1P3B = 30
# Blocks 23--29 are deliberately untouched as a late synthesis/appearance
# guard.  This boundary is preregistered rather than selected after viewing.
TRAINABLE_BLOCK_INDICES = tuple(range(23))
FROZEN_LATE_BLOCK_INDICES = tuple(range(23, TOTAL_BLOCKS_1P3B))
_SHA256 = re.compile(r"[0-9a-f]{64}")

ROLE_PADDING = 0
ROLE_DONOR = 1
ROLE_REFERENCE = 2
ROLE_TARGET = 3
ROLE_NAMES = {
    ROLE_PADDING: "padding",
    ROLE_DONOR: "ordered_appearance_corrupted_donor",
    ROLE_REFERENCE: "independently_encoded_clean_source_reference",
    ROLE_TARGET: "noisy_reconstruction_target",
}


class SourceSelfRoleRepaintError(RuntimeError):
    """Raised before an ambiguous role route or adapter update."""


@dataclass(frozen=True)
class HeldoutFactorialCell:
    """One no-gradient donor-order x reference-identity intervention."""

    cell_id: str
    donor_order: str
    reference_identity: str
    optimizer_supervision: bool = False

    def __post_init__(self) -> None:
        expected = f"{self.donor_order}_{self.reference_identity}"
        if (
            self.donor_order not in {"ordered", "reverse"}
            or self.reference_identity not in {"correct_refs", "wrong_refs"}
            or self.cell_id != expected
            or self.optimizer_supervision is not False
        ):
            raise SourceSelfRoleRepaintError("heldout factorial cell contract differs")


def heldout_factorial_cells() -> tuple[HeldoutFactorialCell, ...]:
    """Return the preregistered 2x2 causal evaluation grid."""

    return tuple(
        HeldoutFactorialCell(
            f"{donor}_{refs}", donor_order=donor, reference_identity=refs
        )
        for donor in ("ordered", "reverse")
        for refs in ("correct_refs", "wrong_refs")
    )


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
        raise SourceSelfRoleRepaintError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceSelfRoleRepaintError(f"{label} must be an exact positive integer")
    return value


def _validated_rho(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
    ):
        raise SourceSelfRoleRepaintError("rho must be finite in [0,1)")
    return float(value)


def _dense_finite_fp32(value: Any, *, label: str, phases: int) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or value.ndim != 5
        or int(value.shape[0]) <= 0
        or tuple(int(item) for item in value.shape[1:3])
        != (LATENT_CHANNELS, phases)
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SourceSelfRoleRepaintError(
            f"{label} must be detached contiguous finite FP32 "
            f"[B,{LATENT_CHANNELS},{phases},H,W] with B>0"
        )
    if int(value.shape[3]) <= 0 or int(value.shape[4]) <= 0:
        raise SourceSelfRoleRepaintError(f"{label} has empty spatial geometry")
    return value


@dataclass(frozen=True)
class TokenRoleLayout:
    """Global visual-token role order before Ulysses append-padding."""

    roles: tuple[int, ...]
    donor_tokens: int
    reference_tokens: tuple[int, ...]
    target_tokens: int

    def __post_init__(self) -> None:
        donor = _positive_int(self.donor_tokens, label="donor_tokens")
        target = _positive_int(self.target_tokens, label="target_tokens")
        if not isinstance(self.reference_tokens, tuple) or len(self.reference_tokens) not in {
            0,
            REFERENCE_COUNT,
        }:
            raise SourceSelfRoleRepaintError(
                "reference token spans must be truly absent or exactly three"
            )
        refs = tuple(
            _positive_int(value, label=f"reference_tokens[{index}]")
            for index, value in enumerate(self.reference_tokens)
        )
        if len(self.roles) != donor + sum(refs) + target:
            raise SourceSelfRoleRepaintError("role length differs from declared spans")
        allowed = {ROLE_DONOR, ROLE_REFERENCE, ROLE_TARGET}
        if any(isinstance(role, bool) or role not in allowed for role in self.roles):
            raise SourceSelfRoleRepaintError("layout contains an unsupported role")
        expected = {
            ROLE_DONOR: donor,
            ROLE_REFERENCE: sum(refs),
            ROLE_TARGET: target,
        }
        if {role: self.roles.count(role) for role in allowed} != expected:
            raise SourceSelfRoleRepaintError("role counts differ from declared spans")

    @classmethod
    def contiguous(
        cls,
        *,
        donor_tokens: int,
        reference_tokens: Sequence[int],
        target_tokens: int,
    ) -> "TokenRoleLayout":
        refs = tuple(reference_tokens)
        if len(refs) not in {0, REFERENCE_COUNT}:
            raise SourceSelfRoleRepaintError(
                "reference token counts must be empty or exactly three"
            )
        refs_exact = tuple(
            _positive_int(value, label=f"reference_tokens[{index}]")
            for index, value in enumerate(refs)
        )
        donor = _positive_int(donor_tokens, label="donor_tokens")
        target = _positive_int(target_tokens, label="target_tokens")
        roles = (
            (ROLE_DONOR,) * donor
            + (ROLE_REFERENCE,) * sum(refs_exact)
            + (ROLE_TARGET,) * target
        )
        return cls(roles, donor, refs_exact, target)

    @property
    def reference_token_total(self) -> int:
        return sum(self.reference_tokens)

    @property
    def condition_tokens(self) -> int:
        return self.donor_tokens + self.reference_token_total

    @property
    def total_tokens(self) -> int:
        return len(self.roles)

    def receipt(self) -> Mapping[str, Any]:
        role_order = ["ordered_donor"]
        if self.reference_tokens:
            role_order.extend(("source_ref_0", "source_ref_40", "source_ref_80"))
        role_order.append("noisy_target")
        value = {
            "role_order": role_order,
            "donor_tokens": self.donor_tokens,
            "reference_tokens": list(self.reference_tokens),
            "target_tokens": self.target_tokens,
            "total_tokens": self.total_tokens,
            "reference_rgb_indices": (
                list(REFERENCE_RGB_INDICES) if self.reference_tokens else []
            ),
            "references_present": bool(self.reference_tokens),
        }
        return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class RouteInvocation:
    layout: TokenRoleLayout
    sequence_parallel_rank: int
    sequence_parallel_size: int
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.layout, TokenRoleLayout):
            raise SourceSelfRoleRepaintError("route requires a TokenRoleLayout")
        size = _positive_int(self.sequence_parallel_size, label="SP size")
        rank = self.sequence_parallel_rank
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
            raise SourceSelfRoleRepaintError("SP rank lies outside its group")
        if size not in {1, 2, 4}:
            raise SourceSelfRoleRepaintError(
                "preservation role supports only SP1 tests and SP2/SP4 AUH"
            )
        if not isinstance(self.enabled, bool):
            raise SourceSelfRoleRepaintError("enabled must be boolean")

    @property
    def local_length(self) -> int:
        return math.ceil(self.layout.total_tokens / self.sequence_parallel_size)

    def global_roles(self, *, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.layout.roles, dtype=torch.int64, device=device)

    def local_roles(self, *, device: torch.device) -> torch.Tensor:
        roles = self.global_roles(device=device)
        padded = self.local_length * self.sequence_parallel_size
        if padded > int(roles.numel()):
            roles = torch.cat(
                (
                    roles,
                    torch.full(
                        (padded - int(roles.numel()),),
                        ROLE_PADDING,
                        dtype=torch.int64,
                        device=device,
                    ),
                )
            )
        start = self.sequence_parallel_rank * self.local_length
        return roles[start : start + self.local_length].contiguous()


_ACTIVE_ROUTE: ContextVar[Optional[RouteInvocation]] = ContextVar(
    "bernini_source_self_role_repaint_route", default=None
)


def active_route() -> Optional[RouteInvocation]:
    return _ACTIVE_ROUTE.get()


@contextmanager
def activate_route(invocation: RouteInvocation) -> Iterator[None]:
    if not isinstance(invocation, RouteInvocation):
        raise SourceSelfRoleRepaintError("activate_route requires a RouteInvocation")
    if active_route() is not None:
        raise SourceSelfRoleRepaintError("nested role routes are forbidden")
    token: Token[Optional[RouteInvocation]] = _ACTIVE_ROUTE.set(invocation)
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


@contextmanager
def _replay_checkpoint_route(invocation: RouteInvocation) -> Iterator[None]:
    """Expose one exact route during checkpoint forward or recomputation."""

    if not isinstance(invocation, RouteInvocation):
        raise SourceSelfRoleRepaintError(
            "checkpoint route replay requires a RouteInvocation"
        )
    current = active_route()
    if current is invocation:
        yield
        return
    if current is not None:
        raise SourceSelfRoleRepaintError(
            "checkpoint recomputation entered a different role route"
        )
    with activate_route(invocation):
        yield


def checkpoint_route_context_fn() -> tuple[Any, Any]:
    """Capture the exact active route for non-reentrant checkpoint replay.

    PyTorch may execute checkpoint recomputation in a context that does not
    inherit this module's ``ContextVar``.  Returning two independent context
    manager instances makes both the original checkpointed forward and its
    backward-time recomputation observe the same invocation object.
    """

    invocation = active_route()
    if invocation is None:
        raise SourceSelfRoleRepaintError(
            "checkpoint was created without an active role route"
        )
    return (
        _replay_checkpoint_route(invocation),
        _replay_checkpoint_route(invocation),
    )


class RoleAwarePatchEmbedding(nn.Module):
    """Keep the frozen Conv3d embedding and add one explicit role vector."""

    def __init__(self, base: nn.Module, *, hidden_size: int):
        super().__init__()
        self.base = base
        self.hidden_size = _positive_int(hidden_size, label="hidden_size")
        self.role_embedding = nn.Embedding(len(ROLE_NAMES), self.hidden_size)
        nn.init.zeros_(self.role_embedding.weight)
        self.role_embedding.weight.requires_grad_(True)

    @property
    def weight(self) -> Any:
        return getattr(self.base, "weight")

    @property
    def bias(self) -> Any:
        return getattr(self.base, "bias", None)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        base_output = self.base(patches)
        invocation = active_route()
        if invocation is None or not invocation.enabled:
            return base_output
        if (
            patches.ndim != 5
            or tuple(int(item) for item in patches.shape[1:])
            != (LATENT_CHANNELS, *PATCH_SHAPE)
            or base_output.ndim != 5
            or tuple(int(item) for item in base_output.shape[2:]) != (1, 1, 1)
            or int(base_output.shape[0]) != invocation.layout.total_tokens
            or int(base_output.shape[1]) != self.hidden_size
        ):
            raise SourceSelfRoleRepaintError("role-aware patch geometry differs")
        roles = invocation.global_roles(device=patches.device)
        delta = self.role_embedding(roles).to(base_output.dtype)
        return base_output + delta[:, :, None, None, None]


class TargetRowLoRA(nn.Module):
    """Low-rank residual that is exactly zero outside local target rows."""

    def __init__(self, base: nn.Module, *, rank: int, alpha: float, projection: str):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise SourceSelfRoleRepaintError(f"{projection} base must be nn.Linear")
        self.base = base
        self.rank = _positive_int(rank, label="LoRA rank")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or float(alpha) <= 0.0
        ):
            raise SourceSelfRoleRepaintError("LoRA alpha must be finite and positive")
        if projection not in {"to_q", "to_out.0"}:
            raise SourceSelfRoleRepaintError("only self-attention Q/O may be wrapped")
        self.alpha = float(alpha)
        self.projection = projection
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False, dtype=torch.float32)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)

    @property
    def scale(self) -> float:
        return self.alpha / float(self.rank)

    @property
    def weight(self) -> Any:
        return self.base.weight

    @property
    def bias(self) -> Any:
        return self.base.bias

    def adapter_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        invocation = active_route()
        if invocation is None or not invocation.enabled:
            return torch.zeros_like(self.base(hidden_states))
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            raise SourceSelfRoleRepaintError("target-row LoRA expects [1,N,D]")
        roles = invocation.local_roles(device=hidden_states.device)
        if int(hidden_states.shape[1]) != int(roles.numel()):
            raise SourceSelfRoleRepaintError(
                "local sequence differs from append-pad/contiguous Ulysses selector"
            )
        selector = (roles == ROLE_TARGET).view(1, -1, 1)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            delta = self.lora_b(self.lora_a(hidden_states.float())) * self.scale
        return delta.to(hidden_states.dtype) * selector.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.base(hidden_states) + self.adapter_delta(hidden_states)


@dataclass
class SourceSelfAdapterHandle:
    transformer: nn.Module
    patch_wrapper: RoleAwarePatchEmbedding
    q_wrappers: tuple[tuple[int, TargetRowLoRA], ...]
    o_wrappers: tuple[tuple[int, TargetRowLoRA], ...]
    original_patch_embedding: nn.Module
    original_q: tuple[tuple[int, nn.Module], ...]
    original_o: tuple[tuple[int, nn.Module], ...]
    block_indices: tuple[int, ...]
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise SourceSelfRoleRepaintError("adapter is already restored")
        values: list[tuple[str, nn.Parameter]] = [
            ("role_embedding.weight", self.patch_wrapper.role_embedding.weight)
        ]
        for block_index, wrapper in self.q_wrappers:
            values.extend(
                (
                    (f"blocks.{block_index}.attn1.to_q.lora_a.weight", wrapper.lora_a.weight),
                    (f"blocks.{block_index}.attn1.to_q.lora_b.weight", wrapper.lora_b.weight),
                )
            )
        for block_index, wrapper in self.o_wrappers:
            values.extend(
                (
                    (f"blocks.{block_index}.attn1.to_out.0.lora_a.weight", wrapper.lora_a.weight),
                    (f"blocks.{block_index}.attn1.to_out.0.lora_b.weight", wrapper.lora_b.weight),
                )
            )
        if len({id(parameter) for _, parameter in values}) != len(values):
            raise SourceSelfRoleRepaintError("trainable parameter aliases another tensor")
        if any(not parameter.requires_grad for _, parameter in values):
            raise SourceSelfRoleRepaintError("adapter parameter is unexpectedly frozen")
        return tuple(values)

    def base_parameters_frozen(self) -> bool:
        trainable_ids = {id(value) for _, value in self.trainable_named_parameters()}
        return all(
            id(parameter) in trainable_ids or not parameter.requires_grad
            for parameter in self.transformer.parameters()
        )

    @contextmanager
    def route(self, invocation: RouteInvocation) -> Iterator[None]:
        if self.restored:
            raise SourceSelfRoleRepaintError("cannot route a restored adapter")
        with activate_route(invocation):
            yield

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "trainable_block_indices": list(self.block_indices),
            "frozen_block_indices": [
                index for index in range(TOTAL_BLOCKS_1P3B) if index not in self.block_indices
            ],
            "default_scope_is_early_mid_0_through_22": self.block_indices
            == TRAINABLE_BLOCK_INDICES,
            "registered_all30_ablation": self.block_indices == tuple(range(TOTAL_BLOCKS_1P3B)),
            "target_row_only": True,
            "self_attention_only": True,
            "projections": ["attn1.to_q", "attn1.to_out.0"],
            "key_value_trainable": False,
            "cross_attention_trainable": False,
            "late_blocks_trainable": any(
                index in self.block_indices for index in FROZEN_LATE_BLOCK_INDICES
            ),
            "role_embedding_after_frozen_patch_embedding": True,
            "source_id_is_not_role": True,
            "sp_selector": "append_pad_then_contiguous_rank_chunk",
            "context_covers_forward_and_backward": True,
            "checkpoint_context_fn_captures_exact_route_by_identity": True,
            "base_parameters_frozen": self.base_parameters_frozen(),
            "trainable": [
                {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
                for name, parameter in self.trainable_named_parameters()
            ],
            "semantic_motion_claim": False,
        }
        return {**value, "digest": object_sha256(value)}

    def safetensors_metadata(
        self, *, conditional_base_rho: float
    ) -> Mapping[str, str]:
        """Metadata required for strict renderer-side reconstruction."""

        rho = _validated_rho(conditional_base_rho)
        return {
            "schema_version": SCHEMA_VERSION,
            "block_indices_json": canonical_json_bytes(list(self.block_indices)).decode("ascii"),
            "projections_json": canonical_json_bytes(["attn1.to_q", "attn1.to_out.0"]).decode("ascii"),
            "target_row_only": "true",
            "role_embedding": "donor_reference_target",
            "lora_rank": str(self.q_wrappers[0][1].rank),
            "lora_alpha_hex": self.q_wrappers[0][1].alpha.hex(),
            "conditional_base_schema": CONDITIONAL_BASE_SCHEMA,
            "conditional_base_rho_hex": rho.hex(),
            "inference_requires_identical_conditional_base_rho": "true",
        }

    def restore(self) -> None:
        if self.restored or active_route() is not None:
            raise SourceSelfRoleRepaintError("adapter cannot be restored in its current state")
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        self.transformer.patch_embedding = self.original_patch_embedding
        for index, original in self.original_q:
            blocks[index].attn1.to_q = original
        for index, original in self.original_o:
            blocks[index].attn1.to_out[0] = original
        self.restored = True


def install_source_self_adapter(
    transformer: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    block_indices: Sequence[int] = TRAINABLE_BLOCK_INDICES,
) -> SourceSelfAdapterHandle:
    """Install the closed Q/O target-row adapter scope."""

    if not isinstance(transformer, nn.Module):
        raise SourceSelfRoleRepaintError("transformer must be an nn.Module")
    blocks = tuple(getattr(transformer, "blocks", ()))
    patch = getattr(transformer, "patch_embedding", None)
    indices = tuple(block_indices)
    if (
        len(blocks) != TOTAL_BLOCKS_1P3B
        or not isinstance(patch, nn.Conv3d)
        or tuple(int(item) for item in patch.kernel_size) != PATCH_SHAPE
        or indices not in {TRAINABLE_BLOCK_INDICES, tuple(range(TOTAL_BLOCKS_1P3B))}
    ):
        raise SourceSelfRoleRepaintError("Bernini 1.3B structure or closed block scope differs")
    hidden_size = int(patch.out_channels)
    originals_q: list[tuple[int, nn.Module]] = []
    originals_o: list[tuple[int, nn.Module]] = []
    for index in indices:
        attention = getattr(blocks[index], "attn1", None)
        query = getattr(attention, "to_q", None)
        output = getattr(attention, "to_out", None)
        if (
            not isinstance(query, nn.Linear)
            or not isinstance(output, nn.ModuleList)
            or len(output) != 2
            or not isinstance(output[0], nn.Linear)
            or query.in_features != hidden_size
            or query.out_features != hidden_size
            or output[0].in_features != hidden_size
            or output[0].out_features != hidden_size
        ):
            raise SourceSelfRoleRepaintError(f"block {index} self-attention Q/O differs")
        originals_q.append((index, query))
        originals_o.append((index, output[0]))

    wrapper = RoleAwarePatchEmbedding(patch, hidden_size=hidden_size)
    device = patch.weight.device
    wrapper.role_embedding.to(device=device, dtype=torch.float32)
    transformer.patch_embedding = wrapper
    q_wrappers: list[tuple[int, TargetRowLoRA]] = []
    o_wrappers: list[tuple[int, TargetRowLoRA]] = []
    try:
        for (index, query), (_, output) in zip(originals_q, originals_o):
            q_wrapper = TargetRowLoRA(
                query, rank=rank, alpha=alpha, projection="to_q"
            ).to(device=device)
            o_wrapper = TargetRowLoRA(
                output, rank=rank, alpha=alpha, projection="to_out.0"
            ).to(device=device)
            blocks[index].attn1.to_q = q_wrapper
            blocks[index].attn1.to_out[0] = o_wrapper
            q_wrappers.append((index, q_wrapper))
            o_wrappers.append((index, o_wrapper))
    except Exception:
        transformer.patch_embedding = patch
        for index, original in originals_q:
            blocks[index].attn1.to_q = original
        for index, original in originals_o:
            blocks[index].attn1.to_out[0] = original
        raise
    handle = SourceSelfAdapterHandle(
        transformer=transformer,
        patch_wrapper=wrapper,
        q_wrappers=tuple(q_wrappers),
        o_wrappers=tuple(o_wrappers),
        original_patch_embedding=patch,
        original_q=tuple(originals_q),
        original_o=tuple(originals_o),
        block_indices=indices,
    )
    if not handle.base_parameters_frozen():
        handle.restore()
        raise SourceSelfRoleRepaintError("complete Bernini base must be frozen first")
    return handle


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def strict_load_source_self_adapter(
    transformer: nn.Module,
    checkpoint: str | Path,
    *,
    expected_file_sha256: str,
    expected_rho: float,
    rank: int = 8,
    alpha: float = 8.0,
    block_indices: Sequence[int] = TRAINABLE_BLOCK_INDICES,
) -> tuple[SourceSelfAdapterHandle, Mapping[str, Any]]:
    """Install and strictly load a published adapter safetensors file.

    The tensor closure, metadata schema, registered block range, dtype, shape,
    finiteness and file SHA must all match before any parameter is copied.
    On failure the freshly installed wrappers are restored.
    """

    if type(expected_file_sha256) is not str or _SHA256.fullmatch(expected_file_sha256) is None:
        raise SourceSelfRoleRepaintError("adapter file SHA-256 is invalid")
    requested = Path(checkpoint).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise SourceSelfRoleRepaintError("adapter checkpoint must be absolute and non-symlink")
    try:
        path = requested.resolve(strict=True)
        mode = path.lstat().st_mode
    except OSError as error:
        raise SourceSelfRoleRepaintError(f"adapter checkpoint is unavailable: {error}") from error
    if path != requested or not stat.S_ISREG(mode) or path.is_symlink() or path.suffix != ".safetensors":
        raise SourceSelfRoleRepaintError("adapter checkpoint must be a canonical safetensors file")
    expected_rho_value = _validated_rho(expected_rho)
    before_identity = _stat_identity(path)
    actual_sha = _file_sha256(path)
    after_initial_hash = _stat_identity(path)
    if before_identity != after_initial_hash:
        raise SourceSelfRoleRepaintError("adapter checkpoint changed while hashing")
    if actual_sha != expected_file_sha256:
        raise SourceSelfRoleRepaintError("adapter checkpoint SHA-256 differs")
    handle = install_source_self_adapter(
        transformer, rank=rank, alpha=alpha, block_indices=block_indices
    )
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as opened:
            keys = tuple(sorted(opened.keys()))
            metadata = dict(opened.metadata() or {})
            tensors = {key: opened.get_tensor(key).contiguous() for key in keys}
        after_read_identity = _stat_identity(path)
        after_read_sha = _file_sha256(path)
        final_identity = _stat_identity(path)
        if (
            before_identity != after_read_identity
            or before_identity != final_identity
            or after_read_sha != expected_file_sha256
        ):
            raise SourceSelfRoleRepaintError("adapter checkpoint changed while loading")
        expected_metadata = dict(
            handle.safetensors_metadata(conditional_base_rho=expected_rho_value)
        )
        if metadata != expected_metadata:
            raise SourceSelfRoleRepaintError("adapter safetensors metadata differs")
        named = handle.trainable_named_parameters()
        expected_names = tuple(sorted(name for name, _ in named))
        if keys != expected_names:
            raise SourceSelfRoleRepaintError("adapter tensor key closure differs")
        parameter_map = dict(named)
        for name in expected_names:
            tensor = tensors[name]
            parameter = parameter_map[name]
            if (
                tensor.dtype != torch.float32
                or tuple(tensor.shape) != tuple(parameter.shape)
                or tensor.requires_grad
                or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise SourceSelfRoleRepaintError(f"adapter tensor contract differs: {name}")
        with torch.no_grad():
            for name, parameter in named:
                parameter.copy_(tensors[name].to(device=parameter.device))
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "path": str(path),
            "file_sha256": actual_sha,
            "conditional_base_schema": CONDITIONAL_BASE_SCHEMA,
            "conditional_base_rho_hex": expected_rho_value.hex(),
            "pre_post_stat_and_hash_stable": True,
            "metadata": metadata,
            "tensor_count": len(tensors),
            "tensor_names_sha256": object_sha256(list(expected_names)),
            "strict_tensor_closure": True,
            "base_parameters_frozen": handle.base_parameters_frozen(),
        }
        return handle, {**receipt, "digest": object_sha256(receipt)}
    except Exception:
        if not handle.restored:
            handle.restore()
        raise


def appearance_corrupted_donor(clean: Any, *, seed: int) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Apply one temporal-shared per-channel affine style corruption.

    Every phase receives the same channel transform.  Temporal order and
    spatial coordinates are left untouched, but this fact alone does not
    establish preservation of semantic motion.
    """

    value = _dense_finite_fp32(clean, label="clean source latent", phases=LATENT_PHASES)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise SourceSelfRoleRepaintError("donor corruption seed must lie in [0,2^63)")
    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    mean = value.mean(dim=(2, 3, 4), keepdim=True)
    std = value.std(dim=(2, 3, 4), unbiased=False, keepdim=True).clamp_min(1.0e-6)
    gain = torch.empty(
        (1, LATENT_CHANNELS, 1, 1, 1), dtype=torch.float32, device=value.device
    ).uniform_(0.55, 1.45, generator=generator)
    bias = torch.empty_like(gain).uniform_(-0.40, 0.40, generator=generator)
    donor = ((value - mean) * gain + mean + bias * std).detach().contiguous()
    if not bool(torch.isfinite(donor).all().item()) or torch.equal(donor, value):
        raise SourceSelfRoleRepaintError("appearance corruption is invalid or ineffective")
    receipt = {
        "kind": "temporal_shared_per_channel_affine_in_normalized_clean_latent",
        "seed": seed,
        "gain_interval": [0.55, 1.45],
        "bias_in_source_channel_std_interval": [-0.4, 0.4],
        "temporal_order_changed": False,
        "spatial_coordinates_changed": False,
        "semantic_motion_preservation_claimed": False,
        "donor_is_independently_vae_encoded": False,
        "donor_is_deterministically_derived_from_clean_source_posterior_mode": True,
        "online_latent_corruption_is_ablation_not_main_pretext": True,
    }
    return donor, {**receipt, "digest": object_sha256(receipt)}


def reverse_donor_phases(donor: Any) -> torch.Tensor:
    """Reverse phases 1..20 while retaining the causal first phase."""

    value = _dense_finite_fp32(donor, label="donor", phases=LATENT_PHASES)
    indices = torch.tensor(
        (0, *tuple(range(LATENT_PHASES - 1, 0, -1))),
        dtype=torch.int64,
        device=value.device,
    )
    result = value.index_select(2, indices).detach().contiguous()
    if tuple(result.shape) != tuple(value.shape):
        raise SourceSelfRoleRepaintError("reverse donor geometry differs")
    return result


def temporal_dc_identity_carrier(source: Any) -> torch.Tensor:
    """Return a centered temporal-constant (motion-light) source carrier."""

    value = _dense_finite_fp32(source, label="source carrier input", phases=LATENT_PHASES)
    carrier = value.mean(dim=2, keepdim=True).expand_as(value).clone()
    carrier.sub_(carrier.mean(dim=(1, 2, 3, 4), keepdim=True))
    norm = carrier.flatten(1).norm(dim=1)
    if bool((norm <= 1.0e-8).any().item()):
        raise SourceSelfRoleRepaintError("source identity carrier is degenerate")
    return carrier.detach().contiguous()


def source_rich_conditional_base(
    epsilon: Any,
    source: Any,
    *,
    rho: float = 0.0,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Construct ``sqrt(1-rho^2)e + rho*c`` with an orthogonal carrier.

    ``rho=0`` returns the exact input tensor.  For ``rho>0`` a centered
    temporal-DC source seed is Gram--Schmidt orthogonalized against epsilon
    independently per sample, then norm-matched.  This preserves the realized
    epsilon norm under the rotation; it still does not prove Gaussianity.
    """

    noise = _dense_finite_fp32(epsilon, label="epsilon", phases=LATENT_PHASES)
    src = _dense_finite_fp32(source, label="source", phases=LATENT_PHASES)
    if tuple(noise.shape) != tuple(src.shape):
        raise SourceSelfRoleRepaintError("epsilon/source geometry differs")
    rho_value = _validated_rho(rho)
    if rho_value == 0.0:
        result = noise
        carrier_kind = "not_constructed_for_exact_rho0"
        orthogonality = 0.0
        carrier_norm_error = 0.0
        energy_error = 0.0
        temporal_dc_error = 0.0
    else:
        carrier_seed = temporal_dc_identity_carrier(src)
        # Do Gram--Schmidt inside the temporal-DC subspace.  A constant-over-
        # time carrier is orthogonal to the full epsilon iff its single-frame
        # value is orthogonal to epsilon summed over time.  Float64 working
        # precision keeps the subsequent FP32 audit tight without introducing
        # temporal variation from epsilon into the identity carrier.
        dc_seed = carrier_seed[:, :, 0, :, :].double().flatten(1)
        epsilon_sum = noise.double().sum(dim=2).flatten(1)
        epsilon_sum_energy = epsilon_sum.square().sum(dim=1, keepdim=True)
        projection = (dc_seed * epsilon_sum).sum(dim=1, keepdim=True) / (
            epsilon_sum_energy.clamp_min(torch.finfo(torch.float64).tiny)
        )
        dc_carrier = dc_seed - projection * epsilon_sum
        dc_norm = dc_carrier.norm(dim=1, keepdim=True)
        flat_e = noise.flatten(1)
        e_norm = flat_e.norm(dim=1, keepdim=True)
        if bool((dc_norm <= 1.0e-8).any().item()) or bool((e_norm <= 1.0e-8).any().item()):
            raise SourceSelfRoleRepaintError(
                "conditional-base Gram-Schmidt normalization is degenerate"
            )
        target_dc_norm = e_norm.double() / math.sqrt(float(LATENT_PHASES))
        dc_carrier = dc_carrier * (target_dc_norm / dc_norm)
        carrier = (
            dc_carrier.reshape(
                int(noise.shape[0]),
                LATENT_CHANNELS,
                1,
                int(noise.shape[3]),
                int(noise.shape[4]),
            )
            .expand_as(noise)
            .to(dtype=torch.float32)
            .contiguous()
        )
        flat_c = carrier.flatten(1)
        mixed = math.sqrt(1.0 - rho_value * rho_value) * flat_e + rho_value * flat_c
        result = mixed.reshape_as(noise).detach().contiguous()
        orthogonality = float(
            ((flat_c * flat_e).sum(dim=1) / (e_norm[:, 0].square())).abs().max().item()
        )
        carrier_norm_error = float(
            ((flat_c.norm(dim=1) - e_norm[:, 0]).abs() / e_norm[:, 0]).max().item()
        )
        energy_error = float(
            ((mixed.norm(dim=1) - e_norm[:, 0]).abs() / e_norm[:, 0]).max().item()
        )
        temporal_dc_error = float(
            (carrier - carrier[:, :, :1]).abs().max().item()
        )
        if (
            not math.isfinite(orthogonality)
            or not math.isfinite(carrier_norm_error)
            or not math.isfinite(energy_error)
            or not math.isfinite(temporal_dc_error)
            or orthogonality > 2.0e-5
            or carrier_norm_error > 2.0e-5
            or energy_error > 2.0e-5
            or temporal_dc_error != 0.0
        ):
            raise SourceSelfRoleRepaintError("conditional-base numeric audit failed")
        carrier_kind = (
            "centered_temporal_dc_source_seed_then_per_sample_gram_schmidt_"
            "within_temporal_dc_subspace_against_epsilon_and_norm_matched"
        )
    receipt = {
        "schema_version": CONDITIONAL_BASE_SCHEMA,
        "rho_hex": rho_value.hex(),
        "equation": "eS=sqrt(1-rho^2)*epsilon+rho*normalized_motion_light_carrier",
        "carrier": carrier_kind,
        "rho0_is_byte_alias_of_standard_gaussian": rho_value == 0.0 and result.data_ptr() == noise.data_ptr(),
        "train_inference_contract_must_match": True,
        "gaussianity_claimed_for_rho_gt_zero": False,
        "energy_preservation_does_not_imply_gaussianity": True,
        "max_absolute_carrier_epsilon_cosine": orthogonality,
        "max_observed_relative_carrier_norm_error": carrier_norm_error,
        "max_observed_relative_energy_change": energy_error,
        "max_observed_temporal_dc_error": temporal_dc_error,
        "gram_schmidt_verified_for_rho_gt_zero": rho_value > 0.0,
        "temporal_dc_carrier_verified_for_rho_gt_zero": rho_value > 0.0,
        "carrier_norm_match_verified": rho_value == 0.0
        or carrier_norm_error <= 2.0e-5,
        "realized_energy_preservation_verified": rho_value == 0.0
        or energy_error <= 2.0e-5,
    }
    return result, {**receipt, "digest": object_sha256(receipt)}


__all__ = [
    "CONDITIONAL_BASE_SCHEMA",
    "FROZEN_LATE_BLOCK_INDICES",
    "HeldoutFactorialCell",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "PATCH_SHAPE",
    "PATCH_VALUES",
    "REFERENCE_COUNT",
    "REFERENCE_RGB_INDICES",
    "ROLE_DONOR",
    "ROLE_REFERENCE",
    "ROLE_TARGET",
    "RouteInvocation",
    "SCHEMA_VERSION",
    "SourceSelfAdapterHandle",
    "SourceSelfRoleRepaintError",
    "TRAINABLE_BLOCK_INDICES",
    "TargetRowLoRA",
    "TokenRoleLayout",
    "activate_route",
    "active_route",
    "checkpoint_route_context_fn",
    "appearance_corrupted_donor",
    "heldout_factorial_cells",
    "install_source_self_adapter",
    "strict_load_source_self_adapter",
    "object_sha256",
    "reverse_donor_phases",
    "source_rich_conditional_base",
    "temporal_dc_identity_carrier",
]
