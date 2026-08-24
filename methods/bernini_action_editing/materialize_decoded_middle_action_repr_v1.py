#!/usr/bin/env python3
"""Materialize firewall-safe decoded-video ``Delta H_middle`` caches.

This is the Stage-B representation *extractor*, not a generator trainer.  A
real target or a final self-generated exact81 MP4 is decoded, re-encoded by the
frozen Bernini VAE, and contrasted with an exact first-frame-repeat video.  The
two branches use the same caption, posterior/FM random seed, Gaussian,
timestep, rotary tensor, frozen renderer, and target-token geometry.
The VAE runs on rank0 only; WORLD4 receives one canonical raw-tensor posterior
payload and verifies its dtype/shape/value digest before constructing any FM
batch.  ``torch.save`` bytes are treated only as an ephemeral legacy transport,
never as posterior identity.

Read-only hooks observe block outputs at ``6, 12, 18, 24``.  Absolute action
and no-op hidden states live only long enough to form ``action - noop``.  Before
publication the residual has temporal DC, spatial camera common mode, and an
ephemeral no-op appearance direction removed; it is channel-whitened and sent
through a fixed low-rank Rademacher projection.  The safetensors cache contains
only those detached projected residuals.  In particular it contains no RGB,
VAE/clean latent, absolute hidden/value, Q/K, model endpoint, or video path.

Production invocation is WORLD4 / Ulysses-SP4.  CPU fakes belong in the unit
tests and are deliberately not exposed as evidence by this CLI.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import types
from typing import Any, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch
from torch import nn

import exact_local_video_materializer_v1 as exact_video
import infer_native_self_generated_intermediate_anchor_canary_v1 as native_bridge
import self_generated_intermediate_action_anchor_v1 as anchor_core


METHOD = "bernini-decoded-middle-action-representation-v1"
CACHE_SCHEMA = "bernini-decoded-middle-action-representation-cache-v1"
RECEIPT_SCHEMA = "bernini-decoded-middle-action-representation-receipt-v1"
POSTERIOR_PAYLOAD_SCHEMA = (
    "bernini-decoded-middle-canonical-posterior-payload-v1"
)
POSTERIOR_ENVELOPE_SCHEMA = (
    "bernini-decoded-middle-rank0-posterior-envelope-v1"
)
BLOCK_INDICES = (6, 12, 18, 24)
TARGET_CONTROL_ROLES = ("real_forward", "temporal_shuffle", "reverse")
SELF_GENERATED_CONTROL_ROLES = (
    "self_generated",
    "self_generated_temporal_shuffle",
    "self_generated_reverse",
)
CONTROL_ROLES = (*TARGET_CONTROL_ROLES, *SELF_GENERATED_CONTROL_ROLES)
PHASES = 21
HIDDEN_WIDTH = 1536
DEFAULT_SIGMAS = (0.85, 0.55, 0.20)
DEFAULT_PROJECTION_WIDTH = 256
DEFAULT_PROJECTION_SEED = 2026082401
EXPLICIT_GAUSSIAN_DOMAIN = (
    "bernini-decoded-middle-explicit-prepack-gaussian-v1"
)
PINNED_BERNINI_DATA_SHA256 = (
    "29aa4f89579c7771cb9f78706fde4f0dca0de954fdb2f5e2de1abacd8a0d6c65"
)
PINNED_PACK_VAE_LATENTS_SOURCE_SHA256 = (
    "445893fee2cca1f745265cea857740937f338a04b67e9f895fef943948c49c9f"
)
PINNED_PROCESS_RENDERER_SAMPLE_SOURCE_SHA256 = (
    "9e8532898267ea167f0776a71a30233cbfada4f94132e0b546f1740115ee372e"
)
# ``pack_vae_latents`` forms both the FM state and velocity in the tensor's
# floating dtype.  Reconstructing the common Gaussian after that computation
# therefore has a cancellation-dependent error, not a scale-independent
# tolerance.  Six dtype-rounding operations conservatively cover the two
# products plus sum used for x, the velocity subtraction, and substitution of
# the recovered clean/noise operands in the a-posteriori bound below.
MATCHED_GAUSSIAN_FORWARD_ERROR_OPERATIONS = 6
PHASE0_MATCH_ATOL = 2.0e-5
DETERMINISTIC_VAE_POLICY = (
    "rank0_two_branch_vae_encode_in_local_strict_deterministic_scope_"
    "with_exact_flag_restoration_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DecodedMiddleRepresentationError(RuntimeError):
    """Fail-closed representation extraction or publication error."""


def fail(message: str) -> None:
    raise DecodedMiddleRepresentationError(message)


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
        raise DecodedMiddleRepresentationError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
        named = path.stat()
    except OSError as error:
        raise DecodedMiddleRepresentationError(
            f"cannot hash file: {path}"
        ) from error
    final_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    named_identity = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
    )
    if identity != final_identity or identity != named_identity:
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        fail("tensor digest requires one finite materialized tensor")
    owned = value.detach().to(device="cpu").clone(
        memory_format=torch.contiguous_format
    ).contiguous()
    header = canonical_json_bytes(
        {"dtype": str(owned.dtype), "shape": list(map(int, owned.shape))}
    )
    payload = owned.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\0" + payload).hexdigest()


def explicit_gaussian_packed_shape(
    posterior_shape: Sequence[int],
) -> tuple[int, int, int, int, int]:
    """Return the one-target Wan patch shape before Bernini FM packing."""

    values = tuple(posterior_shape)
    if (
        len(values) != 5
        or any(type(value) is not int or value <= 0 for value in values)
        or tuple(values[:3]) != (1, 32, PHASES)
        or values[3] % 2
        or values[4] % 2
    ):
        fail("explicit Gaussian posterior geometry differs")
    return (
        PHASES * (values[3] // 2) * (values[4] // 2),
        16,
        1,
        2,
        2,
    )


def derive_explicit_gaussian_seed(
    *,
    base_seed: int,
    case_id: str,
    instruction_sha256: str,
) -> int:
    """Domain-separate the pre-pack Gaussian from all posterior/FM RNGs."""

    if (
        type(base_seed) is not int
        or base_seed < 0
        or _SAFE_CASE_ID.fullmatch(case_id) is None
    ):
        fail("explicit Gaussian seed binding differs")
    instruction_sha = require_sha256(
        instruction_sha256, label="Gaussian instruction"
    )
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": EXPLICIT_GAUSSIAN_DOMAIN,
                "base_seed": base_seed,
                "case_id": case_id,
                "instruction_sha256": instruction_sha,
            }
        )
    ).digest()
    # ``manual_seed`` accepts a signed 64-bit domain.  Avoid zero so a missing
    # seed cannot accidentally look identical to this derived authority.
    return int.from_bytes(digest[:8], "big") % (2**63 - 1) + 1


def generate_rank0_explicit_gaussian(
    shape: Sequence[int], *, derived_seed: int
) -> torch.Tensor:
    """Generate the sole canonical FP32 Gaussian on rank0 CPU."""

    dimensions = tuple(shape)
    if (
        len(dimensions) != 5
        or any(type(value) is not int or value <= 0 for value in dimensions)
        or tuple(dimensions[1:]) != (16, 1, 2, 2)
        or type(derived_seed) is not int
        or not 0 < derived_seed <= 2**63 - 1
    ):
        fail("rank0 explicit Gaussian geometry/seed differs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed)
    gaussian = torch.randn(
        dimensions,
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    ).contiguous()
    if (
        gaussian.requires_grad
        or gaussian.grad_fn is not None
        or not bool(torch.isfinite(gaussian).all().item())
    ):
        fail("rank0 explicit Gaussian is invalid")
    return gaussian


def broadcast_rank0_explicit_gaussian(
    gaussian: Optional[torch.Tensor],
    *,
    expected_shape: Sequence[int],
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Broadcast canonical Gaussian tensor bytes with WORLD4/NCCL.

    Object/pickle transport is deliberately not used.  Each rank receives one
    FP32 CUDA tensor through ``dist.broadcast`` and returns an owned canonical
    CPU tensor whose raw semantic digest can enter WORLD4 consensus.
    """

    import torch.distributed as dist

    dimensions = tuple(expected_shape)
    if (
        dist.is_available() is not True
        or dist.is_initialized() is not True
        or dist.get_world_size() != 4
        or type(rank) is not int
        or rank != dist.get_rank()
        or not isinstance(device, torch.device)
        or device.type != "cuda"
        or len(dimensions) != 5
        or tuple(dimensions[1:]) != (16, 1, 2, 2)
        or (rank == 0) is not (gaussian is not None)
    ):
        fail("rank0 explicit Gaussian WORLD4 broadcast contract differs")
    if rank == 0:
        assert gaussian is not None
        if (
            gaussian.device.type != "cpu"
            or gaussian.dtype != torch.float32
            or tuple(gaussian.shape) != dimensions
            or gaussian.requires_grad
            or gaussian.grad_fn is not None
            or not gaussian.is_contiguous()
            or not bool(torch.isfinite(gaussian).all().item())
        ):
            fail("rank0 explicit Gaussian tensor differs")
        transported = gaussian.to(device=device).contiguous()
    else:
        transported = torch.empty(dimensions, dtype=torch.float32, device=device)
    dist.broadcast(transported, src=0)
    owned = transported.detach().to(device="cpu").contiguous()
    if (
        owned.dtype != torch.float32
        or tuple(owned.shape) != dimensions
        or owned.requires_grad
        or owned.grad_fn is not None
        or not bool(torch.isfinite(owned).all().item())
    ):
        fail("broadcast explicit Gaussian tensor differs")
    return owned


def load_validated_materializer_posterior(
    blob: bytes,
    metadata: Mapping[str, Any],
    *,
    label: str,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Authenticate one materializer transport by its tensor semantics.

    ``BerniniVaeEncoder.encode`` returns a ``torch.save`` byte stream plus a
    metadata digest over the actual float32 posterior tensor.  A torch archive
    is a transport container, not a canonical identity: two archives can carry
    the same tensor while differing in pickle/zip serialization details.  This
    helper therefore reloads the trusted process-local blob with
    ``weights_only=True`` and verifies dtype, shape, finiteness, and the
    materializer's semantic digest before the tensor can enter the WORLD4
    broadcast envelope.
    """

    if (
        type(label) is not str
        or not label
        or type(blob) is not bytes
        or not blob
        or not isinstance(metadata, Mapping)
    ):
        fail("VAE posterior transport contract differs")
    shape_value = metadata.get("posterior_parameters_shape")
    dtype_value = metadata.get("posterior_parameters_dtype")
    digest_value = require_sha256(
        metadata.get("posterior_parameters_tensor_sha256"),
        label=f"{label} posterior tensor",
    )
    if (
        not isinstance(shape_value, (list, tuple))
        or len(shape_value) != 5
        or any(type(value) is not int or value <= 0 for value in shape_value)
        or tuple(shape_value[:3]) != (1, 32, PHASES)
        or dtype_value != str(torch.float32)
    ):
        fail(f"{label} VAE posterior metadata geometry differs")
    try:
        posterior = torch.load(
            io.BytesIO(blob), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise DecodedMiddleRepresentationError(
            f"cannot load {label} VAE posterior transport"
        ) from error
    if (
        not isinstance(posterior, torch.Tensor)
        or posterior.device.type != "cpu"
        or posterior.dtype != torch.float32
        or tuple(posterior.shape) != tuple(shape_value)
        or posterior.requires_grad
        or posterior.grad_fn is not None
        or not bool(torch.isfinite(posterior).all().item())
    ):
        fail(f"{label} VAE posterior tensor differs")
    posterior = posterior.detach().clone(
        memory_format=torch.contiguous_format
    ).contiguous()
    actual_digest = tensor_sha256(posterior)
    if actual_digest != digest_value:
        fail(f"{label} VAE posterior semantic digest differs")
    identity = {
        "tensor_sha256": actual_digest,
        "shape": list(map(int, posterior.shape)),
        "dtype": str(posterior.dtype),
        "identity_kind": "sha256_dtype_shape_raw_tensor_bytes",
    }
    return posterior, identity


def pack_canonical_posterior_payload(
    posterior: torch.Tensor,
) -> Mapping[str, Any]:
    """Pack one posterior as canonical raw tensor bytes for rank broadcast."""

    if (
        not isinstance(posterior, torch.Tensor)
        or posterior.device.type != "cpu"
        or posterior.dtype != torch.float32
        or posterior.ndim != 5
        or tuple(posterior.shape[:3]) != (1, 32, PHASES)
        or posterior.requires_grad
        or posterior.grad_fn is not None
        or not bool(torch.isfinite(posterior).all().item())
    ):
        fail("canonical posterior payload input differs")
    owned = posterior.detach().clone(
        memory_format=torch.contiguous_format
    ).contiguous()
    raw = owned.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return {
        "schema_version": POSTERIOR_PAYLOAD_SCHEMA,
        "dtype": str(owned.dtype),
        "shape": list(map(int, owned.shape)),
        "tensor_sha256": tensor_sha256(owned),
        "raw_byte_count": len(raw),
        "raw_tensor_bytes": raw,
    }


def unpack_canonical_posterior_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Reconstruct and authenticate a rank0 canonical posterior payload."""

    expected_keys = {
        "schema_version",
        "dtype",
        "shape",
        "tensor_sha256",
        "raw_byte_count",
        "raw_tensor_bytes",
    }
    if (
        type(label) is not str
        or not label
        or not isinstance(payload, Mapping)
        or set(payload) != expected_keys
        or payload.get("schema_version") != POSTERIOR_PAYLOAD_SCHEMA
        or payload.get("dtype") != str(torch.float32)
    ):
        fail(f"{label} canonical posterior payload closure differs")
    shape_value = payload.get("shape")
    raw = payload.get("raw_tensor_bytes")
    if (
        not isinstance(shape_value, list)
        or len(shape_value) != 5
        or any(type(value) is not int or value <= 0 for value in shape_value)
        or tuple(shape_value[:3]) != (1, 32, PHASES)
        or type(raw) is not bytes
        or type(payload.get("raw_byte_count")) is not int
        or payload.get("raw_byte_count") != len(raw)
        or len(raw)
        != math.prod(shape_value)
        * torch.empty((), dtype=torch.float32).element_size()
    ):
        fail(f"{label} canonical posterior payload geometry differs")
    expected_digest = require_sha256(
        payload.get("tensor_sha256"), label=f"{label} canonical posterior"
    )
    posterior = (
        torch.frombuffer(bytearray(raw), dtype=torch.float32)
        .clone()
        .reshape(tuple(shape_value))
        .contiguous()
    )
    actual_digest = tensor_sha256(posterior)
    if (
        actual_digest != expected_digest
        or not bool(torch.isfinite(posterior).all().item())
    ):
        fail(f"{label} canonical posterior payload digest differs")
    identity = {
        "tensor_sha256": actual_digest,
        "shape": list(map(int, posterior.shape)),
        "dtype": str(posterior.dtype),
        "identity_kind": "sha256_dtype_shape_raw_tensor_bytes",
    }
    return posterior, identity


def posterior_tensor_to_transport_blob(posterior: torch.Tensor) -> bytes:
    """Serialize a verified tensor solely for the legacy in-memory loader."""

    if (
        not isinstance(posterior, torch.Tensor)
        or posterior.device.type != "cpu"
        or posterior.dtype != torch.float32
        or posterior.ndim != 5
        or tuple(posterior.shape[:3]) != (1, 32, PHASES)
        or posterior.requires_grad
        or posterior.grad_fn is not None
        or not posterior.is_contiguous()
        or not bool(torch.isfinite(posterior).all().item())
    ):
        fail("posterior transport tensor differs")
    buffer = io.BytesIO()
    torch.save(posterior, buffer)
    return buffer.getvalue()


def parse_sigmas(value: str) -> tuple[float, ...]:
    try:
        sigmas = tuple(float(item.strip()) for item in value.split(","))
    except (TypeError, ValueError) as error:
        raise DecodedMiddleRepresentationError(
            "sigmas must be comma-separated finite floats"
        ) from error
    if (
        not sigmas
        or len(sigmas) > 8
        or len(set(sigmas)) != len(sigmas)
        or any(not math.isfinite(item) or not 0.0 < item < 1.0 for item in sigmas)
    ):
        fail("sigmas must be unique finite values strictly inside (0,1)")
    return sigmas


def canonical_input_role(value: str) -> str:
    if value == "shuffle":
        return "temporal_shuffle"
    if value not in CONTROL_ROLES:
        fail("input role must be one registered G1 representation control")
    return value


def is_self_generated_role(value: str) -> bool:
    """Return whether a registered role is sourced from a self-generated video."""

    if value not in CONTROL_ROLES:
        fail("input role must be one registered G1 representation control")
    return value in SELF_GENERATED_CONTROL_ROLES


def deterministic_projection(
    hidden_width: int,
    projection_width: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a case-independent fixed JL projection, never fitted on target."""

    if (
        type(hidden_width) is not int
        or type(projection_width) is not int
        or type(seed) is not int
        or hidden_width <= 0
        or projection_width <= 0
        or projection_width > hidden_width
        or seed < 0
    ):
        fail("low-rank projection configuration differs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(
        0,
        2,
        (hidden_width, projection_width),
        generator=generator,
        dtype=torch.int8,
        device="cpu",
    )
    projection = signs.float().mul_(2.0).sub_(1.0)
    projection.mul_(1.0 / math.sqrt(float(projection_width)))
    return projection.to(device=device, dtype=torch.float32).contiguous()


def _finite_detached_hidden(
    value: torch.Tensor, *, label: str, shape: Optional[Sequence[int]] = None
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 5
        or int(value.shape[0]) != 1
        or not value.is_floating_point()
        or value.requires_grad
        or value.grad_fn is not None
        or (shape is not None and tuple(value.shape) != tuple(shape))
        or not bool(torch.isfinite(value).all().item())
    ):
        fail(f"{label} must be one detached finite [1,T,H,W,D] tensor")
    return value


def preprocess_middle_delta(
    *,
    action_hidden: torch.Tensor,
    noop_hidden: torch.Tensor,
    appearance_direction: torch.Tensor,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Remove nuisance coordinates and project an ephemeral hidden contrast.

    The returned tensor is ``[T,P,R]``.  Neither absolute input hidden is
    referenced by the result or metrics after this function returns.
    """

    action = _finite_detached_hidden(action_hidden, label="action hidden")
    noop = _finite_detached_hidden(
        noop_hidden, label="no-op hidden", shape=action.shape
    )
    if int(action.shape[1]) != PHASES:
        fail("middle residual must have exact21 latent phases")
    hidden_width = int(action.shape[-1])
    if (
        not isinstance(appearance_direction, torch.Tensor)
        or tuple(appearance_direction.shape) != (hidden_width,)
        or appearance_direction.requires_grad
        or appearance_direction.grad_fn is not None
        or appearance_direction.device != action.device
        or not bool(torch.isfinite(appearance_direction).all().item())
        or not isinstance(projection, torch.Tensor)
        or projection.ndim != 2
        or tuple(projection.shape[:1]) != (hidden_width,)
        or projection.device != action.device
        or projection.dtype != torch.float32
        or projection.requires_grad
        or not bool(torch.isfinite(projection).all().item())
    ):
        fail("appearance/projector geometry differs")

    delta = action.float() - noop.float()
    raw_rms = float(delta.square().mean().sqrt().item())
    if not math.isfinite(raw_rms) or raw_rms <= 1.0e-10:
        fail("action-minus-first-frame-repeat middle residual is degenerate")

    # Per-patch temporal DC removes static appearance.  Per-phase spatial mean
    # removes global/camera common mode without estimating camera trajectories.
    signal = delta - delta.mean(dim=1, keepdim=True)
    signal = signal - signal.mean(dim=(2, 3), keepdim=True)

    direction_norm = appearance_direction.float().norm()
    if not bool(torch.isfinite(direction_norm).item()) or float(direction_norm.item()) <= 1.0e-8:
        fail("ephemeral no-op appearance direction is degenerate")
    unit = appearance_direction.float() / direction_norm
    appearance_component = torch.einsum("bthwd,d->bthw", signal, unit)
    appearance_rms = float(appearance_component.square().mean().sqrt().item())
    signal = signal - appearance_component.unsqueeze(-1) * unit

    # The first latent phase is source/initial-state authority.  VAE temporal
    # boundary effects are not allowed to become an action code.
    signal[:, 0].zero_()
    channel_rms = signal.square().mean(dim=(0, 1, 2, 3)).sqrt()
    positive = channel_rms[channel_rms > 0]
    if positive.numel() == 0:
        fail("nuisance removal collapsed every middle channel")
    floor = max(1.0e-6, float(positive.median().item()) * 1.0e-3)
    whitened = signal / channel_rms.clamp_min(floor).reshape(1, 1, 1, 1, -1)
    whitened[:, 0].zero_()
    projected = torch.matmul(whitened, projection)
    projected = projected.reshape(
        PHASES,
        int(action.shape[2]) * int(action.shape[3]),
        int(projection.shape[1]),
    ).detach().contiguous()
    if (
        projected.requires_grad
        or projected.grad_fn is not None
        or not bool(torch.isfinite(projected).all().item())
        or bool(projected[0].any().item())
    ):
        fail("projected middle residual violates detach/finite/phase0 closure")

    spatial_common = signal.mean(dim=(2, 3))
    appearance_after = torch.einsum("bthwd,d->bthw", signal, unit)
    metrics = {
        "raw_action_minus_noop_rms": raw_rms,
        "appearance_component_removed_rms": appearance_rms,
        "nuisance_removed_rms": float(signal.square().mean().sqrt().item()),
        "projected_rms": float(projected.float().square().mean().sqrt().item()),
        "spatial_common_mode_max_abs_after": float(spatial_common.abs().max().item()),
        "appearance_direction_max_abs_after": float(appearance_after.abs().max().item()),
        "channel_whitening_floor": floor,
        "phase0_hard_zero": True,
    }
    if not all(math.isfinite(float(value)) for value in metrics.values() if not isinstance(value, bool)):
        fail("middle preprocessing metrics are non-finite")
    return projected, metrics


def _first_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    fail("Bernini block output must be a Tensor or tensor-first tuple")


class MiddleBlockCaptureBank:
    """Read-only exact-block hook bank retaining target rows only, ephemerally."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        hidden_width: int,
        block_indices: tuple[int, ...] = BLOCK_INDICES,
    ) -> None:
        blocks = tuple(getattr(transformer, "blocks", ()))
        if (
            not isinstance(transformer, nn.Module)
            or len(blocks) != 30
            or block_indices != BLOCK_INDICES
            or hidden_width <= 0
            or any(parameter.requires_grad for parameter in transformer.parameters())
        ):
            fail("middle hook requires one frozen exact30 Bernini transformer")
        self.transformer = transformer
        self.hidden_width = int(hidden_width)
        self.block_indices = block_indices
        self._handles: list[Any] = []
        self._layout: Optional[anchor_core.LocalTokenLayout] = None
        self._captures: dict[int, torch.Tensor] = {}
        self._seen: set[int] = set()

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    def install(self) -> None:
        if self.installed or self._layout is not None:
            fail("middle hook bank lifecycle differs")
        handles = []
        try:
            for index in self.block_indices:
                handles.append(
                    self.transformer.blocks[index].register_forward_hook(
                        self._make_hook(index)
                    )
                )
        except Exception:
            for handle in handles:
                handle.remove()
            raise
        self._handles = handles

    def remove(self) -> None:
        if not self.installed or self._layout is not None:
            fail("cannot remove an inactive or active middle hook bank")
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def _make_hook(self, block_index: int) -> Any:
        def callback(_module: Any, _inputs: Any, output: Any) -> Any:
            layout = self._layout
            if layout is None:
                return output
            if block_index in self._seen:
                fail("selected middle block fired more than once")
            native = _first_output_tensor(output)
            if (
                native.ndim != 3
                or tuple(native.shape[:2]) != (1, layout.local_length)
                or int(native.shape[2]) != self.hidden_width
                or not native.is_floating_point()
                or not bool(torch.isfinite(native.detach()).all().item())
            ):
                fail("hooked Bernini middle hidden geometry differs")
            self._captures[block_index] = anchor_core.extract_local_target(
                native, layout
            )
            self._seen.add(block_index)
            # Returning the exact same object makes the observer structurally
            # read-only, including tuple side channels.
            return output

        return callback

    @contextmanager
    def capture(
        self, layout: anchor_core.LocalTokenLayout
    ) -> Iterator[None]:
        if not self.installed or self._layout is not None or self._captures:
            fail("middle capture lifecycle differs")
        self._layout = layout
        self._seen.clear()
        try:
            yield
            if self._seen != set(self.block_indices) or set(self._captures) != set(
                self.block_indices
            ):
                fail("middle capture did not close all four selected blocks")
        finally:
            self._layout = None
            self._seen.clear()

    def pop(self) -> dict[int, torch.Tensor]:
        if self._layout is not None or set(self._captures) != set(self.block_indices):
            fail("middle capture closure differs")
        result = dict(self._captures)
        self._captures.clear()
        return result


def distributed_appearance_direction(
    noop_local_hidden: torch.Tensor,
    *,
    process_group: Any = None,
) -> torch.Tensor:
    """Reduce an ephemeral no-op absolute hidden to one nuisance direction."""

    if (
        not isinstance(noop_local_hidden, torch.Tensor)
        or noop_local_hidden.ndim != 3
        or int(noop_local_hidden.shape[0]) != 1
        or int(noop_local_hidden.shape[1]) <= 0
        or noop_local_hidden.requires_grad
        or not bool(torch.isfinite(noop_local_hidden).all().item())
    ):
        fail("local no-op hidden for appearance projection differs")
    summed = noop_local_hidden.detach().float().sum(dim=(0, 1)).contiguous()
    count = torch.tensor(
        [int(noop_local_hidden.shape[1])],
        dtype=torch.float32,
        device=noop_local_hidden.device,
    )
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=process_group)
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=process_group)
    if float(count.item()) <= 0:
        fail("global no-op hidden count is empty")
    direction = (summed / count).detach().contiguous()
    if direction.requires_grad or not bool(torch.isfinite(direction).all().item()):
        fail("ephemeral appearance direction is invalid")
    return direction


@dataclass(frozen=True)
class MatchedPatchPair:
    action_clean: torch.Tensor
    noop_clean: torch.Tensor
    gaussian: torch.Tensor
    selector: torch.Tensor
    original_sigma: float
    noise_max_abs_error: float
    noise_max_abs_forward_error_bound: float
    noise_max_error_to_bound_ratio: float
    noise_original_dtype: str
    noise_dtype_epsilon: float
    canonical_gaussian_sha256: str
    phase0_clean_max_abs_error: float
    gaussian_authority: Mapping[str, Any]


@dataclass(frozen=True)
class ExplicitGaussianPackCapture:
    """Ephemeral pre-pack operands and bit-exact native outputs for one arm."""

    clean: torch.Tensor
    gaussian: torch.Tensor
    raw_noise_sigma: torch.Tensor
    packed_state: torch.Tensor
    target_velocity: torch.Tensor
    pack_call_count: int
    randn_like_injection_count: int
    packed_state_original_op_order_bit_exact: bool
    target_velocity_bit_exact: bool


@dataclass(frozen=True)
class CanonicalPosteriorPair:
    action: torch.Tensor
    noop: torch.Tensor
    action_identity: Mapping[str, Any]
    noop_identity: Mapping[str, Any]
    fps: float
    input_hw: tuple[int, int]
    bucket_hw: tuple[int, int]
    action_rgb_sha256: str
    noop_rgb_sha256: str
    deterministic_vae_authority: Optional[Mapping[str, Any]]


def _function_source_sha256(function: Any) -> str:
    try:
        source = inspect.getsource(function).encode("utf-8")
    except (OSError, TypeError, UnicodeError) as error:
        raise DecodedMiddleRepresentationError(
            "cannot inspect pinned Bernini function source"
        ) from error
    return hashlib.sha256(source).hexdigest()


def validate_pinned_renderer_data_functions(
    *, process_renderer_sample: Any, pack_vae_latents: Any
) -> Mapping[str, Any]:
    """Authenticate the exact vendor functions before cloning their code."""

    if (
        not isinstance(process_renderer_sample, types.FunctionType)
        or not isinstance(pack_vae_latents, types.FunctionType)
        or process_renderer_sample.__name__ != "process_renderer_sample"
        or pack_vae_latents.__name__ != "pack_vae_latents"
        or process_renderer_sample.__module__ != "bernini.training.data"
        or pack_vae_latents.__module__ != "bernini.training.data"
        or process_renderer_sample.__globals__.get("pack_vae_latents")
        is not pack_vae_latents
        or pack_vae_latents.__globals__.get("torch") is not torch
    ):
        fail("pinned Bernini renderer data function binding differs")
    try:
        process_path = Path(
            process_renderer_sample.__code__.co_filename
        ).resolve(strict=True)
        pack_path = Path(pack_vae_latents.__code__.co_filename).resolve(
            strict=True
        )
    except OSError as error:
        raise DecodedMiddleRepresentationError(
            "pinned Bernini renderer data source is unavailable"
        ) from error
    file_sha = file_sha256(process_path)
    pack_source_sha = _function_source_sha256(pack_vae_latents)
    process_source_sha = _function_source_sha256(process_renderer_sample)
    if (
        process_path != pack_path
        or file_sha != PINNED_BERNINI_DATA_SHA256
        or pack_source_sha != PINNED_PACK_VAE_LATENTS_SOURCE_SHA256
        or process_source_sha
        != PINNED_PROCESS_RENDERER_SAMPLE_SOURCE_SHA256
    ):
        fail("pinned Bernini renderer data source/function hash differs")
    return {
        "vendor_data_file_sha256": file_sha,
        "pack_vae_latents_source_sha256": pack_source_sha,
        "process_renderer_sample_source_sha256": process_source_sha,
        "vendor_module_mutated": False,
        "original_function_globals_mutated": False,
    }


def _clone_function_with_private_globals(
    function: types.FunctionType,
    *,
    replacements: Mapping[str, Any],
) -> types.FunctionType:
    """Clone one function code object without writing its module globals."""

    if not isinstance(function, types.FunctionType):
        fail("explicit Gaussian clone target is not a Python function")
    private_globals = dict(function.__globals__)
    private_globals.update(dict(replacements))
    clone = types.FunctionType(
        function.__code__,
        private_globals,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    clone.__kwdefaults__ = (
        None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    )
    clone.__annotations__ = dict(getattr(function, "__annotations__", {}))
    clone.__dict__.update(function.__dict__)
    clone.__module__ = function.__module__
    clone.__qualname__ = function.__qualname__
    clone.__doc__ = function.__doc__
    return clone


class _ExplicitGaussianCaptureCollector:
    def __init__(self, gaussian: torch.Tensor) -> None:
        if (
            not isinstance(gaussian, torch.Tensor)
            or gaussian.device.type != "cpu"
            or gaussian.dtype != torch.float32
            or gaussian.ndim != 5
            or tuple(gaussian.shape[1:]) != (16, 1, 2, 2)
            or gaussian.requires_grad
            or gaussian.grad_fn is not None
            or not gaussian.is_contiguous()
            or not bool(torch.isfinite(gaussian).all().item())
        ):
            fail("explicit pre-pack Gaussian authority differs")
        self.authority = gaussian.detach().clone().contiguous()
        self.pack_call_count = 0
        self.randn_like_injection_count = 0
        self.clean: Optional[torch.Tensor] = None
        self.raw_noise_sigma: Optional[torch.Tensor] = None
        self.capture: Optional[ExplicitGaussianPackCapture] = None

    def begin_pack(self, noise_sigma: Any) -> None:
        self.pack_call_count += 1
        if (
            self.pack_call_count != 1
            or not isinstance(noise_sigma, torch.Tensor)
            or noise_sigma.numel() != 1
            or not noise_sigma.dtype.is_floating_point
            or noise_sigma.device.type != "cpu"
            or noise_sigma.requires_grad
            or noise_sigma.grad_fn is not None
            or not bool(torch.isfinite(noise_sigma).all().item())
        ):
            fail("explicit Gaussian pack/noise-sigma authority differs")
        self.raw_noise_sigma = noise_sigma.detach().clone().contiguous()

    def inject(self, args: Sequence[Any], kwargs: Mapping[str, Any]) -> torch.Tensor:
        self.randn_like_injection_count += 1
        if (
            self.randn_like_injection_count != 1
            or len(tuple(args)) != 1
            or set(kwargs) != {"dtype"}
            or kwargs.get("dtype") != torch.float32
        ):
            fail("explicit Gaussian randn_like call contract differs")
        packed = tuple(args)[0]
        if (
            not isinstance(packed, torch.Tensor)
            or packed.device.type != "cpu"
            or not packed.dtype.is_floating_point
            or tuple(packed.shape) != tuple(self.authority.shape)
            or packed.requires_grad
            or packed.grad_fn is not None
            or not packed.is_contiguous()
            or not bool(torch.isfinite(packed).all().item())
        ):
            fail("explicit Gaussian packed clean geometry differs")
        self.clean = packed.detach().clone().contiguous()
        # Return an owned tensor with the exact supplied raw FP32 values.  No
        # RNG is consulted and no dtype/device conversion is permitted here.
        return self.authority.detach().clone().contiguous()

    def finish_pack(self, output: Any) -> None:
        if (
            self.pack_call_count != 1
            or self.randn_like_injection_count != 1
            or self.clean is None
            or self.raw_noise_sigma is None
            or self.capture is not None
            or not isinstance(output, Mapping)
        ):
            fail("explicit Gaussian pack capture did not close exactly once")
        state = output.get("input_vae_latents")
        velocity = output.get("target_velocity")
        selector = output.get("vae_latents_mask")
        if (
            not isinstance(state, torch.Tensor)
            or not isinstance(velocity, torch.Tensor)
            or not isinstance(selector, torch.Tensor)
            or selector.dtype != torch.bool
            or selector.ndim != 1
            or not bool(selector.all().item())
            or tuple(state.shape) != tuple(self.clean.shape)
            or tuple(velocity.shape) != tuple(self.clean.shape)
            or state.requires_grad
            or velocity.requires_grad
            or not bool(torch.isfinite(state).all().item())
            or not bool(torch.isfinite(velocity).all().item())
        ):
            fail("explicit Gaussian native pack output differs")
        # These expressions deliberately preserve the pinned vendor's exact
        # Python/PyTorch operation order and the raw sigma tensor dtype.
        expected_state = (
            (1 - self.raw_noise_sigma) * self.clean
            + self.raw_noise_sigma * self.authority
        )
        expected_velocity = self.authority - self.clean.float()
        if not torch.equal(state, expected_state):
            fail("native FM state is not bit-exact under original op order")
        if not torch.equal(velocity, expected_velocity):
            fail("native target velocity is not bit-exact from explicit operands")
        self.capture = ExplicitGaussianPackCapture(
            clean=self.clean.detach().clone().contiguous(),
            gaussian=self.authority.detach().clone().contiguous(),
            raw_noise_sigma=self.raw_noise_sigma.detach().clone().contiguous(),
            packed_state=state.detach().clone().contiguous(),
            target_velocity=velocity.detach().clone().contiguous(),
            pack_call_count=self.pack_call_count,
            randn_like_injection_count=self.randn_like_injection_count,
            packed_state_original_op_order_bit_exact=True,
            target_velocity_bit_exact=True,
        )

    def result(self) -> ExplicitGaussianPackCapture:
        if self.capture is None:
            fail("explicit Gaussian capture is incomplete")
        return self.capture


class _TorchExplicitGaussianProxy:
    """Delegate all torch operations except the sole pinned ``randn_like``."""

    def __init__(self, collector: _ExplicitGaussianCaptureCollector) -> None:
        self._collector = collector

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    def randn_like(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._collector.inject(args, kwargs)


def _run_cloned_renderer_process_with_explicit_gaussian(
    *,
    process_renderer_sample: types.FunctionType,
    pack_vae_latents: types.FunctionType,
    sample: Mapping[str, Any],
    gaussian: torch.Tensor,
    process_kwargs: Mapping[str, Any],
) -> tuple[Any, ExplicitGaussianPackCapture]:
    """Run private code/global clones with one externally owned Gaussian."""

    if (
        process_renderer_sample.__globals__.get("pack_vae_latents")
        is not pack_vae_latents
        or pack_vae_latents.__globals__.get("torch") is not torch
    ):
        fail("original Bernini function globals differ before private clone")
    original_process_pack = process_renderer_sample.__globals__["pack_vae_latents"]
    original_pack_torch = pack_vae_latents.__globals__["torch"]
    collector = _ExplicitGaussianCaptureCollector(gaussian)
    torch_proxy = _TorchExplicitGaussianProxy(collector)
    cloned_pack = _clone_function_with_private_globals(
        pack_vae_latents, replacements={"torch": torch_proxy}
    )

    def controlled_pack(
        vae_rope_func: Any,
        vae_type_list: torch.Tensor,
        image_inputs: Mapping[str, Any],
        video_inputs: Mapping[str, Any],
        noise_sigma: torch.Tensor,
        max_vae_frames: Optional[int] = None,
    ) -> Any:
        collector.begin_pack(noise_sigma)
        value = cloned_pack(
            vae_rope_func,
            vae_type_list,
            image_inputs,
            video_inputs,
            noise_sigma,
            max_vae_frames=max_vae_frames,
        )
        collector.finish_pack(value)
        return value

    cloned_process = _clone_function_with_private_globals(
        process_renderer_sample,
        replacements={"pack_vae_latents": controlled_pack},
    )
    transformed = cloned_process(dict(sample), **dict(process_kwargs))
    if (
        process_renderer_sample.__globals__.get("pack_vae_latents")
        is not original_process_pack
        or pack_vae_latents.__globals__.get("torch") is not original_pack_torch
        or original_process_pack is not pack_vae_latents
        or original_pack_torch is not torch
    ):
        fail("original Bernini module/function globals were mutated")
    return transformed, collector.result()


def build_explicit_gaussian_renderer_transform(
    *,
    process_renderer_sample: types.FunctionType,
    pack_vae_latents: types.FunctionType,
    tokenizer: Any,
    rope: Any,
    mean: Any,
    std: Any,
    scheduler: Any,
    device: torch.device,
    gaussian: torch.Tensor,
    collate: Any,
    seed_same_sample: Any,
    source_name: str,
) -> tuple[Any, Mapping[str, Any]]:
    """Build the production transform and authenticate its pinned code once."""

    vendor_identity = validate_pinned_renderer_data_functions(
        process_renderer_sample=process_renderer_sample,
        pack_vae_latents=pack_vae_latents,
    )
    if not isinstance(source_name, str) or not source_name.strip():
        fail("explicit Gaussian renderer source_name differs")
    authority = gaussian.detach().clone().contiguous()

    def transform(
        sample: Mapping[str, Any], seed: int
    ) -> tuple[dict[str, Any], ExplicitGaussianPackCapture]:
        if type(seed) is not int or seed < 0:
            fail("explicit Gaussian posterior seed differs")
        seed_same_sample(seed)
        transformed, capture = _run_cloned_renderer_process_with_explicit_gaussian(
            process_renderer_sample=process_renderer_sample,
            pack_vae_latents=pack_vae_latents,
            sample=sample,
            gaussian=authority,
            process_kwargs={
                "tokenizer": tokenizer,
                "vae_rope_func": rope,
                "vae_latent_mean": mean,
                "vae_latent_std": std,
                "noise_scheduler": scheduler,
                "text_dropout_rate": 0.0,
                "img_dropout_rate": 0.0,
                "video_dropout_rate": 0.0,
                "max_vae_frames": PHASES,
                "source_name": source_name,
            },
        )
        batch = collate(transformed, device)
        selector = batch.get("vae_latents_mask")
        if (
            not isinstance(selector, torch.Tensor)
            or selector.ndim != 2
            or tuple(selector.shape) != (1, int(capture.clean.shape[0]))
            or not bool(selector.all().item())
            or not torch.equal(
                batch.get("input_vae_latents", torch.empty(0)).detach().cpu(),
                capture.packed_state,
            )
            or not torch.equal(
                batch.get("target_velocity", torch.empty(0)).detach().cpu(),
                capture.target_velocity,
            )
        ):
            fail("collated explicit Gaussian batch differs from pre-pack capture")
        return batch, capture

    return transform, vendor_identity


def _deterministic_backend_flags() -> Mapping[str, bool]:
    return {
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


@contextmanager
def strict_deterministic_vae_encode_scope() -> Iterator[dict[str, Any]]:
    """Apply strict determinism locally and restore every process flag."""

    before = dict(_deterministic_backend_flags())
    state: dict[str, Any] = {
        "before_flags": before,
        "during_flags": None,
        "restored_flags": None,
        "flags_restored_exact": False,
    }
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        during = dict(_deterministic_backend_flags())
        expected_during = {
            "deterministic_algorithms_enabled": True,
            "deterministic_algorithms_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }
        if during != expected_during:
            fail("strict deterministic VAE encode scope activation differs")
        state["during_flags"] = during
        yield state
    finally:
        torch.use_deterministic_algorithms(
            before["deterministic_algorithms_enabled"],
            warn_only=before["deterministic_algorithms_warn_only"],
        )
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        restored = dict(_deterministic_backend_flags())
        state["restored_flags"] = restored
        state["flags_restored_exact"] = restored == before
        if restored != before:
            fail("strict deterministic VAE encode flags were not restored")


def validate_deterministic_vae_authority(
    authority: Mapping[str, Any],
    *,
    action: Optional[torch.Tensor] = None,
    noop: Optional[torch.Tensor] = None,
) -> Mapping[str, Any]:
    """Validate strict rank0 VAE determinism and exact raw phase0 parity."""

    if not isinstance(authority, Mapping):
        fail("deterministic VAE authority differs")
    expected_keys = {
        "authority_kind",
        "policy",
        "producer_rank",
        "encode_call_count",
        "scope",
        "before_flags",
        "during_flags",
        "restored_flags",
        "flags_restored_exact",
        "posterior_phase0_max_abs_error",
        "posterior_phase0_bit_exact",
        "action_phase0_posterior_sha256",
        "noop_phase0_posterior_sha256",
        "posterior_modified_after_encode",
        "posterior_copy_or_splice_used",
        "trainer_received_posterior",
    }
    flag_keys = {
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
    }
    before = authority.get("before_flags")
    during = authority.get("during_flags")
    restored = authority.get("restored_flags")
    phase0_error = authority.get("posterior_phase0_max_abs_error")
    action_sha = authority.get("action_phase0_posterior_sha256")
    noop_sha = authority.get("noop_phase0_posterior_sha256")
    if (
        set(authority) != expected_keys
        or authority.get("authority_kind")
        != "rank0_local_strict_deterministic_vae_encode"
        or authority.get("policy") != DETERMINISTIC_VAE_POLICY
        or authority.get("producer_rank") != 0
        or authority.get("encode_call_count") != 2
        or authority.get("scope") != "action_and_first_frame_repeat_encode_calls_only"
        or not isinstance(before, Mapping)
        or set(before) != flag_keys
        or any(type(value) is not bool for value in before.values())
        or dict(restored or {}) != dict(before)
        or not isinstance(during, Mapping)
        or dict(during)
        != {
            "deterministic_algorithms_enabled": True,
            "deterministic_algorithms_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }
        or authority.get("flags_restored_exact") is not True
        or isinstance(phase0_error, bool)
        or not isinstance(phase0_error, (int, float))
        or float(phase0_error) != 0.0
        or authority.get("posterior_phase0_bit_exact") is not True
        or type(action_sha) is not str
        or _SHA256.fullmatch(str(action_sha)) is None
        or noop_sha != action_sha
        or authority.get("posterior_modified_after_encode") is not False
        or authority.get("posterior_copy_or_splice_used") is not False
        or authority.get("trainer_received_posterior") is not False
    ):
        fail("deterministic VAE authority differs")
    if (action is None) is not (noop is None):
        fail("deterministic VAE posterior pair differs")
    if action is not None and noop is not None:
        if (
            not isinstance(action, torch.Tensor)
            or not isinstance(noop, torch.Tensor)
            or tuple(action.shape) != tuple(noop.shape)
            or action.ndim != 5
            or tuple(action.shape[:3]) != (1, 32, PHASES)
            or action.dtype != torch.float32
            or noop.dtype != torch.float32
            or action.device.type != "cpu"
            or noop.device.type != "cpu"
            or not action.is_contiguous()
            or not noop.is_contiguous()
            or action.requires_grad
            or noop.requires_grad
            or action.grad_fn is not None
            or noop.grad_fn is not None
            or not bool(torch.isfinite(action).all().item())
            or not bool(torch.isfinite(noop).all().item())
            or not torch.equal(action[:, :, 0], noop[:, :, 0])
            or tensor_sha256(action[:, :, 0]) != action_sha
            or tensor_sha256(noop[:, :, 0]) != noop_sha
        ):
            fail("deterministic VAE posterior phase0 differs")
    return dict(authority)


def build_deterministic_vae_authority(
    scope_state: Mapping[str, Any],
    *,
    action: torch.Tensor,
    noop: torch.Tensor,
) -> Mapping[str, Any]:
    """Seal the unmodified raw posterior pair after deterministic encoding."""

    if (
        not isinstance(scope_state, Mapping)
        or set(scope_state)
        != {"before_flags", "during_flags", "restored_flags", "flags_restored_exact"}
        or scope_state.get("flags_restored_exact") is not True
        or not isinstance(action, torch.Tensor)
        or not isinstance(noop, torch.Tensor)
        or tuple(action.shape) != tuple(noop.shape)
        or action.ndim != 5
        or tuple(action.shape[:3]) != (1, 32, PHASES)
    ):
        fail("deterministic VAE authority input differs")
    phase0_error = float(
        (action[:, :, 0] - noop[:, :, 0]).abs().max().item()
    )
    phase0_exact = torch.equal(action[:, :, 0], noop[:, :, 0])
    if phase0_error != 0.0 or not phase0_exact:
        fail(
            "deterministic action/first-frame-repeat posterior phase0 differs "
            f"(max_abs={phase0_error:.9g})"
        )
    authority = {
        "authority_kind": "rank0_local_strict_deterministic_vae_encode",
        "policy": DETERMINISTIC_VAE_POLICY,
        "producer_rank": 0,
        "encode_call_count": 2,
        "scope": "action_and_first_frame_repeat_encode_calls_only",
        "before_flags": dict(scope_state["before_flags"]),
        "during_flags": dict(scope_state["during_flags"]),
        "restored_flags": dict(scope_state["restored_flags"]),
        "flags_restored_exact": scope_state["flags_restored_exact"],
        "posterior_phase0_max_abs_error": phase0_error,
        "posterior_phase0_bit_exact": phase0_exact,
        "action_phase0_posterior_sha256": tensor_sha256(action[:, :, 0]),
        "noop_phase0_posterior_sha256": tensor_sha256(noop[:, :, 0]),
        "posterior_modified_after_encode": False,
        "posterior_copy_or_splice_used": False,
        "trainer_received_posterior": False,
    }
    return validate_deterministic_vae_authority(
        authority, action=action, noop=noop
    )


def build_rank0_posterior_envelope(
    *,
    action: torch.Tensor,
    noop: torch.Tensor,
    fps: float,
    input_hw: Sequence[int],
    bucket_hw: Sequence[int],
    action_rgb_sha256: str,
    noop_rgb_sha256: str,
    deterministic_vae_authority: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Build the sole rank0-owned, path-free posterior broadcast envelope."""

    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
        or len(tuple(input_hw)) != 2
        or len(tuple(bucket_hw)) != 2
        or any(type(value) is not int or value <= 0 for value in input_hw)
        or any(type(value) is not int or value <= 0 for value in bucket_hw)
    ):
        fail("rank0 posterior envelope video geometry differs")
    action_sha = require_sha256(action_rgb_sha256, label="action RGB tensor")
    noop_sha = require_sha256(noop_rgb_sha256, label="no-op RGB tensor")
    action_payload = pack_canonical_posterior_payload(action)
    noop_payload = pack_canonical_posterior_payload(noop)
    if action_payload["shape"] != noop_payload["shape"]:
        fail("rank0 action/no-op posterior geometry differs")
    deterministic_authority = None
    if deterministic_vae_authority is not None:
        deterministic_authority = validate_deterministic_vae_authority(
            deterministic_vae_authority, action=action, noop=noop
        )
    return {
        "schema_version": POSTERIOR_ENVELOPE_SCHEMA,
        "producer_rank": 0,
        "transport": "world4_nccl_object_broadcast_of_canonical_raw_tensor_bytes",
        "fps": float(fps),
        "input_hw": list(map(int, input_hw)),
        "bucket_hw": list(map(int, bucket_hw)),
        "action_rgb_tensor_sha256": action_sha,
        "noop_rgb_tensor_sha256": noop_sha,
        "action_posterior": action_payload,
        "noop_posterior": noop_payload,
        "deterministic_vae_authority": deterministic_authority,
    }


def unpack_rank0_posterior_envelope(
    envelope: Mapping[str, Any],
) -> CanonicalPosteriorPair:
    """Authenticate an ephemeral WORLD4 envelope and own its tensors."""

    expected_keys = {
        "schema_version",
        "producer_rank",
        "transport",
        "fps",
        "input_hw",
        "bucket_hw",
        "action_rgb_tensor_sha256",
        "noop_rgb_tensor_sha256",
        "action_posterior",
        "noop_posterior",
        "deterministic_vae_authority",
    }
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != expected_keys
        or envelope.get("schema_version") != POSTERIOR_ENVELOPE_SCHEMA
        or envelope.get("producer_rank") != 0
        or envelope.get("transport")
        != "world4_nccl_object_broadcast_of_canonical_raw_tensor_bytes"
    ):
        fail("rank0 posterior envelope closure differs")
    fps = envelope.get("fps")
    input_hw_value = envelope.get("input_hw")
    bucket_hw_value = envelope.get("bucket_hw")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
        or not isinstance(input_hw_value, list)
        or not isinstance(bucket_hw_value, list)
        or len(input_hw_value) != 2
        or len(bucket_hw_value) != 2
        or any(type(value) is not int or value <= 0 for value in input_hw_value)
        or any(type(value) is not int or value <= 0 for value in bucket_hw_value)
    ):
        fail("rank0 posterior envelope geometry differs")
    action, action_identity = unpack_canonical_posterior_payload(
        envelope.get("action_posterior"), label="action"
    )
    noop, noop_identity = unpack_canonical_posterior_payload(
        envelope.get("noop_posterior"), label="no-op"
    )
    expected_posterior_hw = (
        int(bucket_hw_value[0]) // 8,
        int(bucket_hw_value[1]) // 8,
    )
    if (
        action_identity["shape"] != noop_identity["shape"]
        or tuple(action_identity["shape"][3:]) != expected_posterior_hw
    ):
        fail("broadcast action/no-op posterior geometry differs")
    deterministic_authority_value = envelope.get("deterministic_vae_authority")
    deterministic_authority = None
    if deterministic_authority_value is not None:
        deterministic_authority = validate_deterministic_vae_authority(
            deterministic_authority_value,
            action=action,
            noop=noop,
        )
    return CanonicalPosteriorPair(
        action=action,
        noop=noop,
        action_identity=action_identity,
        noop_identity=noop_identity,
        fps=float(fps),
        input_hw=tuple(input_hw_value),
        bucket_hw=tuple(bucket_hw_value),
        action_rgb_sha256=require_sha256(
            envelope.get("action_rgb_tensor_sha256"),
            label="broadcast action RGB tensor",
        ),
        noop_rgb_sha256=require_sha256(
            envelope.get("noop_rgb_tensor_sha256"),
            label="broadcast no-op RGB tensor",
        ),
        deterministic_vae_authority=deterministic_authority,
    )


def _require_equal_tensor_field(
    action: Mapping[str, Any], noop: Mapping[str, Any], name: str
) -> None:
    left = action.get(name)
    right = noop.get(name)
    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or not torch.equal(left, right)
    ):
        fail(f"matched action/no-op field differs: {name}")


def _recover_gaussian_with_forward_error_bound(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Recover one Gaussian and its elementwise FM packing error bound.

    Bernini stores, in the original tensor dtype,::

        x = fl(fl((1-s) * clean) + fl(s * gaussian))
        v = fl(gaussian - clean)

    We evaluate ``x + (1-s) * v`` in float64 and bound the forward error from
    those dtype-rounding operations using Higham's ``gamma_k`` construction.
    The scale is elementwise and follows the actual interpolation and velocity
    operands, so large cancellation receives a proportionally larger bound
    while ordinary-scale genuinely different Gaussians still fail closed.
    """

    if (
        not isinstance(state, torch.Tensor)
        or not isinstance(velocity, torch.Tensor)
        or state.shape != velocity.shape
        or state.dtype != velocity.dtype
        or not state.dtype.is_floating_point
        or state.requires_grad
        or velocity.requires_grad
        or not math.isfinite(float(sigma))
        or not 0.0 <= float(sigma) <= 1.0
        or not bool(torch.isfinite(state).all().item())
        or not bool(torch.isfinite(velocity).all().item())
    ):
        fail("matched FM state dtype/geometry differs")
    dtype_epsilon = float(torch.finfo(state.dtype).eps)
    operations = MATCHED_GAUSSIAN_FORWARD_ERROR_OPERATIONS
    denominator = 1.0 - operations * dtype_epsilon
    if not math.isfinite(dtype_epsilon) or dtype_epsilon <= 0.0 or denominator <= 0.0:
        fail("matched FM dtype cannot support a finite forward-error bound")

    state64 = state.detach().to(dtype=torch.float64)
    velocity64 = velocity.detach().to(dtype=torch.float64)
    sigma64 = float(sigma)
    beta64 = 1.0 - sigma64
    clean64 = state64 - sigma64 * velocity64
    gaussian64 = state64 + beta64 * velocity64

    # The first two terms are the interpolation product/sum scale.  The final
    # term is the scaled velocity-subtraction scale.  gamma_6 also covers using
    # the reconstructed clean/Gaussian as a-posteriori operand estimates.
    operation_scale = (
        beta64 * clean64.abs()
        + sigma64 * gaussian64.abs()
        + beta64 * (gaussian64.abs() + clean64.abs())
    )
    gamma_dtype = (operations * dtype_epsilon) / denominator
    double_epsilon = float(torch.finfo(torch.float64).eps)
    gamma_double = (2.0 * double_epsilon) / (1.0 - 2.0 * double_epsilon)
    reconstruction_scale = state64.abs() + beta64 * velocity64.abs()
    bound64 = gamma_dtype * operation_scale + gamma_double * reconstruction_scale
    if (
        gaussian64.requires_grad
        or clean64.requires_grad
        or bound64.requires_grad
        or not bool(torch.isfinite(gaussian64).all().item())
        or not bool(torch.isfinite(clean64).all().item())
        or not bool(torch.isfinite(bound64).all().item())
        or bool((bound64 < 0.0).any().item())
    ):
        fail("matched Gaussian forward-error bound is invalid")
    return (
        clean64.detach().contiguous(),
        gaussian64.detach().contiguous(),
        bound64.detach().contiguous(),
        dtype_epsilon,
    )


def recover_matched_patch_pair(
    action: Mapping[str, Any],
    noop: Mapping[str, Any],
    *,
    spatial_shape: Sequence[int],
    patches_to_spatial: Any,
    noise_atol: Optional[float] = None,
    phase0_atol: float = PHASE0_MATCH_ATOL,
) -> MatchedPatchPair:
    """Legacy source-only recovery for one bit-identical batch self-pair.

    Production decoded action/no-op extraction is forbidden from using this
    inverse.  G2a still needs to retime one native source-only batch against
    itself, so this compatibility path accepts only bit-identical ``x`` and
    ``v`` branches and preserves the raw timestep dtype while forming sigma.
    """

    for field in (
        "input_ids",
        "attention_mask",
        "t5_input_lens",
        "input_vae_rope",
        "vae_latents_mask",
        "vae_seqlen",
    ):
        _require_equal_tensor_field(action, noop, field)
    action_t = action.get("timesteps")
    noop_t = noop.get("timesteps")
    if (
        not isinstance(action_t, torch.Tensor)
        or not isinstance(noop_t, torch.Tensor)
        or not torch.equal(action_t, noop_t)
        or action_t.numel() == 0
    ):
        fail("matched action/no-op timestep differs")
    raw_sigma = action_t.reshape(-1)[0] / action_t.new_tensor(1000)
    original_sigma = float(raw_sigma.item())
    if not math.isfinite(original_sigma) or not 0.0 <= original_sigma <= 1.0:
        fail("Bernini FM sigma lies outside [0,1]")

    selector = action["vae_latents_mask"].squeeze(0).bool()
    if selector.ndim != 1 or not bool(selector.all().item()):
        fail("decoded middle extractor requires one target-only T2V sequence")
    action_packed = action.get("input_vae_latents")
    noop_packed = noop.get("input_vae_latents")
    action_v = action.get("target_velocity")
    noop_v = noop.get("target_velocity")
    if (
        not isinstance(action_packed, torch.Tensor)
        or not isinstance(noop_packed, torch.Tensor)
        or not isinstance(action_v, torch.Tensor)
        or not isinstance(noop_v, torch.Tensor)
        or action_packed.dtype != noop_packed.dtype
        or action_v.dtype != noop_v.dtype
        or action_packed.dtype != action_v.dtype
        or not action_packed.dtype.is_floating_point
    ):
        fail("matched FM patch geometry differs")
    action_x = action_packed[selector]
    noop_x = noop_packed[selector]
    if (
        action_x.shape != action_v.shape
        or noop_x.shape != noop_v.shape
        or action_x.shape != noop_x.shape
        or action_x.numel() == 0
    ):
        fail("matched FM patch geometry differs")
    if not torch.equal(action_x, noop_x) or not torch.equal(action_v, noop_v):
        fail(
            "legacy FM inverse is restricted to one bit-identical source "
            "batch self-pair; decoded action/no-op requires pre-pack authority"
        )
    (
        action_clean64,
        action_noise64,
        action_noise_bound64,
        dtype_epsilon,
    ) = _recover_gaussian_with_forward_error_bound(
        action_x,
        action_v,
        sigma=original_sigma,
    )
    (
        noop_clean64,
        noop_noise64,
        noop_noise_bound64,
        noop_dtype_epsilon,
    ) = _recover_gaussian_with_forward_error_bound(
        noop_x,
        noop_v,
        sigma=original_sigma,
    )
    if dtype_epsilon != noop_dtype_epsilon:
        fail("matched action/no-op FM dtype epsilon differs")
    noise_abs_error64 = (action_noise64 - noop_noise64).abs()
    noise_forward_bound64 = action_noise_bound64 + noop_noise_bound64
    noise_error = float(noise_abs_error64.max().item())
    noise_bound = float(noise_forward_bound64.max().item())
    ratio64 = torch.where(
        noise_forward_bound64 > 0.0,
        noise_abs_error64
        / noise_forward_bound64.clamp_min(torch.finfo(torch.float64).tiny),
        torch.zeros_like(noise_abs_error64),
    )
    noise_ratio = float(ratio64.max().item())
    allowed_bound64 = noise_forward_bound64
    if noise_atol is not None:
        if (
            isinstance(noise_atol, bool)
            or not isinstance(noise_atol, (int, float))
            or not math.isfinite(float(noise_atol))
            or float(noise_atol) < 0.0
        ):
            fail("legacy noise tightening cap differs")
        allowed_bound64 = torch.minimum(
            allowed_bound64,
            torch.full_like(allowed_bound64, float(noise_atol)),
        )
    if not bool((noise_abs_error64 <= allowed_bound64).all().item()):
        fail(
            "decoded action/no-op branches do not share Gaussian noise "
            f"under the dtype-scaled forward-error bound "
            f"(max_abs={noise_error:.9g}, max_bound={noise_bound:.9g}, "
            f"max_ratio={noise_ratio:.9g})"
        )
    # Validation precedes authority creation.  Both renderer branches are
    # subsequently retimed from this one detached, byte-identifiable tensor.
    gaussian = (
        0.5 * (action_noise64 + noop_noise64)
    ).to(dtype=torch.float32).detach().contiguous()
    action_clean = action_clean64.to(dtype=torch.float32).detach().contiguous()
    noop_clean = noop_clean64.to(dtype=torch.float32).detach().contiguous()
    action_spatial = patches_to_spatial(
        action_clean, spatial_shape=spatial_shape
    ).float()
    noop_spatial = patches_to_spatial(noop_clean, spatial_shape=spatial_shape).float()
    if tuple(action_spatial.shape) != tuple(spatial_shape):
        fail("recovered clean latent spatial geometry differs")
    phase0_error = float(
        (action_spatial[:, :, 0] - noop_spatial[:, :, 0]).abs().max().item()
    )
    if not torch.allclose(
        action_spatial[:, :, 0],
        noop_spatial[:, :, 0],
        rtol=0.0,
        atol=phase0_atol,
    ):
        fail(
            "decoded action/first-frame-repeat latent phase0 differs "
            f"(max_abs={phase0_error:.9g})"
        )
    for label, value in (
        ("action clean", action_clean),
        ("no-op clean", noop_clean),
        ("canonical matched Gaussian", gaussian),
    ):
        if (
            value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            fail(f"{label} is non-finite")
    canonical_gaussian_sha256 = tensor_sha256(gaussian)
    return MatchedPatchPair(
        action_clean=action_clean,
        noop_clean=noop_clean,
        gaussian=gaussian,
        selector=selector.detach().contiguous(),
        original_sigma=original_sigma,
        noise_max_abs_error=noise_error,
        noise_max_abs_forward_error_bound=noise_bound,
        noise_max_error_to_bound_ratio=noise_ratio,
        noise_original_dtype=str(action_x.dtype),
        noise_dtype_epsilon=dtype_epsilon,
        canonical_gaussian_sha256=canonical_gaussian_sha256,
        phase0_clean_max_abs_error=phase0_error,
        gaussian_authority={
            "authority_kind": "legacy_bit_identical_source_self_pair_inverse",
            "recovered_from_x_or_velocity": True,
            "raw_sigma_dtype_preserved": True,
            "canonical_gaussian_sha256": canonical_gaussian_sha256,
        },
    )


def matched_patch_pair_from_explicit_captures(
    action: Mapping[str, Any],
    noop: Mapping[str, Any],
    *,
    action_capture: ExplicitGaussianPackCapture,
    noop_capture: ExplicitGaussianPackCapture,
    spatial_shape: Sequence[int],
    patches_to_spatial: Any,
    base_seed: int,
    derived_seed: int,
    vendor_identity: Mapping[str, Any],
    world_size: int = 4,
) -> MatchedPatchPair:
    """Create the paired FM authority directly from captured pre-pack operands."""

    for field in (
        "input_ids",
        "attention_mask",
        "t5_input_lens",
        "input_vae_rope",
        "vae_latents_mask",
        "vae_seqlen",
        "timesteps",
    ):
        _require_equal_tensor_field(action, noop, field)
    if (
        not isinstance(action_capture, ExplicitGaussianPackCapture)
        or not isinstance(noop_capture, ExplicitGaussianPackCapture)
        or type(base_seed) is not int
        or base_seed < 0
        or type(derived_seed) is not int
        or not 0 < derived_seed <= 2**63 - 1
        or world_size != 4
        or action_capture.pack_call_count != 1
        or noop_capture.pack_call_count != 1
        or action_capture.randn_like_injection_count != 1
        or noop_capture.randn_like_injection_count != 1
        or action_capture.packed_state_original_op_order_bit_exact is not True
        or noop_capture.packed_state_original_op_order_bit_exact is not True
        or action_capture.target_velocity_bit_exact is not True
        or noop_capture.target_velocity_bit_exact is not True
        or not torch.equal(action_capture.gaussian, noop_capture.gaussian)
        or not torch.equal(
            action_capture.raw_noise_sigma, noop_capture.raw_noise_sigma
        )
    ):
        fail("explicit action/no-op Gaussian authority differs")
    required_vendor = {
        "vendor_data_file_sha256": PINNED_BERNINI_DATA_SHA256,
        "pack_vae_latents_source_sha256": (
            PINNED_PACK_VAE_LATENTS_SOURCE_SHA256
        ),
        "process_renderer_sample_source_sha256": (
            PINNED_PROCESS_RENDERER_SAMPLE_SOURCE_SHA256
        ),
        "vendor_module_mutated": False,
        "original_function_globals_mutated": False,
    }
    if dict(vendor_identity) != required_vendor:
        fail("explicit Gaussian pinned vendor identity differs")

    selector = action["vae_latents_mask"].squeeze(0).bool()
    action_state = action.get("input_vae_latents")
    noop_state = noop.get("input_vae_latents")
    action_velocity = action.get("target_velocity")
    noop_velocity = noop.get("target_velocity")
    if (
        selector.ndim != 1
        or not bool(selector.all().item())
        or not isinstance(action_state, torch.Tensor)
        or not isinstance(noop_state, torch.Tensor)
        or not isinstance(action_velocity, torch.Tensor)
        or not isinstance(noop_velocity, torch.Tensor)
        or not torch.equal(
            action_state[selector].detach().cpu(), action_capture.packed_state
        )
        or not torch.equal(
            noop_state[selector].detach().cpu(), noop_capture.packed_state
        )
        or not torch.equal(
            action_velocity.detach().cpu(), action_capture.target_velocity
        )
        or not torch.equal(
            noop_velocity.detach().cpu(), noop_capture.target_velocity
        )
    ):
        fail("explicit Gaussian collated action/no-op state differs")

    device = action_state.device
    action_clean = action_capture.clean.detach().to(
        device=device, dtype=torch.float32
    ).contiguous()
    noop_clean = noop_capture.clean.detach().to(
        device=device, dtype=torch.float32
    ).contiguous()
    gaussian = action_capture.gaussian.detach().to(
        device=device, dtype=torch.float32
    ).contiguous()
    if (
        tuple(action_clean.shape) != tuple(gaussian.shape)
        or tuple(noop_clean.shape) != tuple(gaussian.shape)
        or tuple(selector.shape) != (int(gaussian.shape[0]),)
    ):
        fail("explicit Gaussian clean patch geometry differs")
    action_spatial = patches_to_spatial(
        action_clean, spatial_shape=spatial_shape
    ).float()
    noop_spatial = patches_to_spatial(
        noop_clean, spatial_shape=spatial_shape
    ).float()
    if (
        tuple(action_spatial.shape) != tuple(spatial_shape)
        or tuple(noop_spatial.shape) != tuple(spatial_shape)
    ):
        fail("explicit Gaussian clean latent spatial geometry differs")
    phase0_error = float(
        (action_spatial[:, :, 0] - noop_spatial[:, :, 0]).abs().max().item()
    )
    if phase0_error != 0.0 or not torch.equal(
        action_spatial[:, :, 0], noop_spatial[:, :, 0]
    ):
        fail(
            "deterministic decoded action/first-frame-repeat clean latent "
            "phase0 differs "
            f"(max_abs={phase0_error:.9g})"
        )
    gaussian_sha = tensor_sha256(gaussian)
    action_sigma_sha = tensor_sha256(action_capture.raw_noise_sigma)
    noop_sigma_sha = tensor_sha256(noop_capture.raw_noise_sigma)
    if action_sigma_sha != noop_sigma_sha:
        fail("explicit action/no-op raw noise sigma bytes differ")
    raw_sigma = action_capture.raw_noise_sigma
    original_sigma = float(raw_sigma.reshape(-1)[0].item())
    if not math.isfinite(original_sigma) or not 0.0 <= original_sigma <= 1.0:
        fail("explicit raw Bernini noise sigma lies outside [0,1]")
    authority = {
        "authority_kind": "rank0_domain_seeded_explicit_prepack_fp32_gaussian",
        "domain": EXPLICIT_GAUSSIAN_DOMAIN,
        "producer_rank": 0,
        "base_seed": base_seed,
        "derived_seed": derived_seed,
        "dtype": str(gaussian.dtype),
        "shape": list(map(int, gaussian.shape)),
        "canonical_gaussian_sha256": gaussian_sha,
        "broadcast_transport": "torch_distributed_nccl_fp32_tensor_broadcast",
        "world_size": world_size,
        "world4_raw_sha256_consensus": True,
        "action_injection_count": action_capture.randn_like_injection_count,
        "noop_injection_count": noop_capture.randn_like_injection_count,
        "action_gaussian_sha256": tensor_sha256(action_capture.gaussian),
        "noop_gaussian_sha256": tensor_sha256(noop_capture.gaussian),
        "raw_noise_sigma_dtype": str(raw_sigma.dtype),
        "raw_noise_sigma_shape": list(map(int, raw_sigma.shape)),
        "action_raw_noise_sigma_sha256": action_sigma_sha,
        "noop_raw_noise_sigma_sha256": noop_sigma_sha,
        "clean_capture_stage": "inside_cloned_pack_before_fm_interpolation",
        "packed_state_original_op_order_bit_exact": True,
        "target_velocity_bit_exact": True,
        "recovered_from_x_or_velocity": False,
        **dict(vendor_identity),
        "trainer_received_authority": False,
    }
    return MatchedPatchPair(
        action_clean=action_clean,
        noop_clean=noop_clean,
        gaussian=gaussian,
        selector=selector.detach().contiguous(),
        original_sigma=original_sigma,
        # There is no inverse/noise comparison in this authority path.
        noise_max_abs_error=0.0,
        noise_max_abs_forward_error_bound=0.0,
        noise_max_error_to_bound_ratio=0.0,
        noise_original_dtype=str(gaussian.dtype),
        noise_dtype_epsilon=float(torch.finfo(torch.float32).eps),
        canonical_gaussian_sha256=gaussian_sha,
        phase0_clean_max_abs_error=phase0_error,
        gaussian_authority=authority,
    )


def retime_fm_batch(
    batch: Mapping[str, Any],
    *,
    clean: torch.Tensor,
    gaussian: torch.Tensor,
    selector: torch.Tensor,
    sigma: float,
) -> dict[str, Any]:
    """Re-use one matched Gaussian at an exact registered physical sigma."""

    if (
        not math.isfinite(float(sigma))
        or not 0.0 < float(sigma) < 1.0
        or clean.shape != gaussian.shape
        or clean.requires_grad
        or gaussian.requires_grad
        or selector.dtype != torch.bool
        or int(selector.sum().item()) != int(clean.shape[0])
    ):
        fail("retimed FM state geometry differs")
    result = dict(batch)
    packed = batch.get("input_vae_latents")
    timesteps = batch.get("timesteps")
    if (
        not isinstance(packed, torch.Tensor)
        or not isinstance(timesteps, torch.Tensor)
        or int(packed.shape[0]) != int(selector.numel())
    ):
        fail("retimed Bernini batch fields are absent")
    state = packed.clone()
    state[selector] = (
        (1.0 - float(sigma)) * clean + float(sigma) * gaussian
    ).to(dtype=state.dtype)
    result["input_vae_latents"] = state
    result["target_velocity"] = (gaussian - clean).to(
        dtype=batch["target_velocity"].dtype
    )
    result["timesteps"] = torch.full_like(timesteps, float(sigma) * 1000.0)
    return result


def validate_cache_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    sigma_count: int,
    projection_width: int,
) -> Mapping[str, Mapping[str, Any]]:
    allowed = {f"middle_block_{index:02d}" for index in BLOCK_INDICES}
    if set(tensors) != allowed:
        fail("trainer cache tensor key closure differs")
    receipts: dict[str, Mapping[str, Any]] = {}
    positions: Optional[int] = None
    for key in sorted(tensors):
        tensor = tensors[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 4
            or tuple(tensor.shape[:2]) != (sigma_count, PHASES)
            or int(tensor.shape[-1]) != projection_width
            or int(tensor.shape[2]) <= 0
            or tensor.device.type != "cpu"
            or tensor.dtype not in (torch.float16, torch.float32)
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
            or bool(tensor[:, 0].any().item())
        ):
            fail(f"trainer cache tensor differs: {key}")
        if positions is None:
            positions = int(tensor.shape[2])
        elif int(tensor.shape[2]) != positions:
            fail("trainer cache spatial positions differ across blocks")
        receipts[key] = {
            "shape": list(map(int, tensor.shape)),
            "dtype": str(tensor.dtype),
            "sha256": tensor_sha256(tensor),
            "detached": True,
            "phase0_hard_zero": True,
        }
    return receipts


def validate_explicit_gaussian_authority_record(
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected_keys = {
        "authority_kind",
        "domain",
        "producer_rank",
        "base_seed",
        "derived_seed",
        "dtype",
        "shape",
        "canonical_gaussian_sha256",
        "broadcast_transport",
        "world_size",
        "world4_raw_sha256_consensus",
        "action_injection_count",
        "noop_injection_count",
        "action_gaussian_sha256",
        "noop_gaussian_sha256",
        "raw_noise_sigma_dtype",
        "raw_noise_sigma_shape",
        "action_raw_noise_sigma_sha256",
        "noop_raw_noise_sigma_sha256",
        "clean_capture_stage",
        "packed_state_original_op_order_bit_exact",
        "target_velocity_bit_exact",
        "recovered_from_x_or_velocity",
        "vendor_data_file_sha256",
        "pack_vae_latents_source_sha256",
        "process_renderer_sample_source_sha256",
        "vendor_module_mutated",
        "original_function_globals_mutated",
        "trainer_received_authority",
    }
    if not isinstance(authority, Mapping) or set(authority) != expected_keys:
        fail("explicit Gaussian receipt authority key closure differs")
    shape = authority.get("shape")
    sigma_shape = authority.get("raw_noise_sigma_shape")
    canonical_sha = authority.get("canonical_gaussian_sha256")
    if (
        authority.get("authority_kind")
        != "rank0_domain_seeded_explicit_prepack_fp32_gaussian"
        or authority.get("domain") != EXPLICIT_GAUSSIAN_DOMAIN
        or authority.get("producer_rank") != 0
        or type(authority.get("base_seed")) is not int
        or int(authority.get("base_seed", -1)) < 0
        or type(authority.get("derived_seed")) is not int
        or not 0 < int(authority.get("derived_seed", 0)) <= 2**63 - 1
        or authority.get("dtype") != str(torch.float32)
        or not isinstance(shape, list)
        or len(shape) != 5
        or any(type(value) is not int or value <= 0 for value in shape)
        or tuple(shape[1:]) != (16, 1, 2, 2)
        or type(canonical_sha) is not str
        or _SHA256.fullmatch(str(canonical_sha)) is None
        or authority.get("broadcast_transport")
        != "torch_distributed_nccl_fp32_tensor_broadcast"
        or authority.get("world_size") != 4
        or authority.get("world4_raw_sha256_consensus") is not True
        or authority.get("action_injection_count") != 1
        or authority.get("noop_injection_count") != 1
        or authority.get("action_gaussian_sha256") != canonical_sha
        or authority.get("noop_gaussian_sha256") != canonical_sha
        or authority.get("raw_noise_sigma_dtype")
        not in {"torch.bfloat16", "torch.float16", "torch.float32", "torch.float64"}
        or sigma_shape != [1]
        or type(authority.get("action_raw_noise_sigma_sha256")) is not str
        or _SHA256.fullmatch(
            str(authority.get("action_raw_noise_sigma_sha256", ""))
        )
        is None
        or authority.get("noop_raw_noise_sigma_sha256")
        != authority.get("action_raw_noise_sigma_sha256")
        or authority.get("clean_capture_stage")
        != "inside_cloned_pack_before_fm_interpolation"
        or authority.get("packed_state_original_op_order_bit_exact") is not True
        or authority.get("target_velocity_bit_exact") is not True
        or authority.get("recovered_from_x_or_velocity") is not False
        or authority.get("vendor_data_file_sha256")
        != PINNED_BERNINI_DATA_SHA256
        or authority.get("pack_vae_latents_source_sha256")
        != PINNED_PACK_VAE_LATENTS_SOURCE_SHA256
        or authority.get("process_renderer_sample_source_sha256")
        != PINNED_PROCESS_RENDERER_SAMPLE_SOURCE_SHA256
        or authority.get("vendor_module_mutated") is not False
        or authority.get("original_function_globals_mutated") is not False
        or authority.get("trainer_received_authority") is not False
    ):
        fail("explicit Gaussian receipt authority differs")
    return dict(authority)


def build_receipt(
    *,
    case_id: str,
    input_role: str,
    input_video_sha256: str,
    instruction_sha256: str,
    cache_path: Path,
    cache_sha256: str,
    cache_tensors: Mapping[str, Mapping[str, Any]],
    sigmas: Sequence[float],
    projection_width: int,
    projection_seed: int,
    projection_sha256: str,
    patch_grid: Sequence[int],
    noise_max_abs_error: float,
    noise_max_abs_forward_error_bound: float,
    noise_max_error_to_bound_ratio: float,
    noise_original_dtype: str,
    noise_dtype_epsilon: float,
    canonical_gaussian_sha256: str,
    gaussian_authority: Mapping[str, Any],
    deterministic_vae_authority: Mapping[str, Any],
    phase0_clean_max_abs_error: float,
    block_metrics: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    method_source_sha256: str,
) -> dict[str, Any]:
    require_sha256(input_video_sha256, label="input video")
    require_sha256(instruction_sha256, label="instruction")
    require_sha256(cache_sha256, label="cache")
    require_sha256(projection_sha256, label="projection")
    canonical_gaussian_sha = require_sha256(
        canonical_gaussian_sha256, label="canonical Gaussian"
    )
    explicit_authority = validate_explicit_gaussian_authority_record(
        gaussian_authority
    )
    deterministic_authority = validate_deterministic_vae_authority(
        deterministic_vae_authority
    )
    if (
        type(noise_original_dtype) is not str
        or not noise_original_dtype.startswith("torch.float")
        and noise_original_dtype != "torch.bfloat16"
        or isinstance(noise_dtype_epsilon, bool)
        or not isinstance(noise_dtype_epsilon, (int, float))
        or not math.isfinite(float(noise_dtype_epsilon))
        or float(noise_dtype_epsilon) <= 0.0
        or isinstance(noise_max_abs_error, bool)
        or not isinstance(noise_max_abs_error, (int, float))
        or not math.isfinite(float(noise_max_abs_error))
        or float(noise_max_abs_error) < 0.0
        or isinstance(noise_max_abs_forward_error_bound, bool)
        or not isinstance(noise_max_abs_forward_error_bound, (int, float))
        or not math.isfinite(float(noise_max_abs_forward_error_bound))
        or float(noise_max_abs_forward_error_bound) != 0.0
        or float(noise_max_abs_error) != 0.0
        or isinstance(noise_max_error_to_bound_ratio, bool)
        or not isinstance(noise_max_error_to_bound_ratio, (int, float))
        or not math.isfinite(float(noise_max_error_to_bound_ratio))
        or float(noise_max_error_to_bound_ratio) != 0.0
        or canonical_gaussian_sha
        != explicit_authority["canonical_gaussian_sha256"]
        or isinstance(phase0_clean_max_abs_error, bool)
        or not isinstance(phase0_clean_max_abs_error, (int, float))
        or float(phase0_clean_max_abs_error) != 0.0
    ):
        fail("explicit canonical Gaussian numerical receipt differs")
    if input_role not in CONTROL_ROLES:
        fail("input role differs")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD,
        "complete": True,
        "scientific_claim_authorized": False,
        "case_id": case_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "representation_origin": "decoded_video_reencode",
        "anchor_source_role": input_role,
        "input_video_sha256": input_video_sha256,
        "instruction_sha256": instruction_sha256,
        "cache": {
            "filename": cache_path.name,
            "sha256": cache_sha256,
            "schema_version": CACHE_SCHEMA,
            "tensor_key_allowlist": sorted(cache_tensors),
            "tensors": dict(cache_tensors),
        },
        "representation": {
            "blocks": list(BLOCK_INDICES),
            "capture": "post_transformer_block_output",
            "contrast": "decoded_action_minus_exact_first_frame_repeat",
            "decoded_video_reencode": True,
            "selfgen_native_trajectory": False,
            "noop_constructed_inside_extractor": True,
            "first_frame_repeat_rgb_exact": True,
            "same_caption": True,
            "same_gaussian": True,
            "same_timestep": True,
            "same_rotary": True,
            "sigmas": [float(value) for value in sigmas],
            "noise_max_abs_error": 0.0,
            "gaussian_match": {
                "comparison_stage": "before_fm_interpolation",
                "criterion": (
                    "same_canonical_raw_fp32_tensor_injected_exactly_once_"
                    "per_branch"
                ),
                "inverse_recovery_numerical_fields_applicable": False,
                "canonical_gaussian_sha256": canonical_gaussian_sha,
                "both_branches_retimed_from_canonical_gaussian": True,
                "fixed_absolute_tolerance_is_authority": False,
                "authority": explicit_authority,
            },
            "deterministic_vae_authority": deterministic_authority,
            "phase0_clean_max_abs_error": float(phase0_clean_max_abs_error),
            "phase0_match_atol": 0.0,
            "patch_grid": list(map(int, patch_grid)),
            "nuisance_removal": [
                "temporal_dc",
                "camera_spatial_common_mode",
                "ephemeral_noop_appearance_direction",
                "channel_whitening",
            ],
            "projection": {
                "kind": "case_independent_fixed_rademacher_jl",
                "width": int(projection_width),
                "seed": int(projection_seed),
                "sha256": projection_sha256,
                "fitted_on_input_video": False,
            },
            "block_metrics": dict(block_metrics),
        },
        "information_firewall": {
            "input_video_accessed_by_frozen_extractor": True,
            "target_video_accessed_by_extractor": not is_self_generated_role(
                input_role
            ),
            "target_rgb_or_vae_used_by_frozen_extractor": not is_self_generated_role(
                input_role
            ),
            "target_video_accessed_by_trainer": False,
            "target_rgb_or_vae_target_used_by_trainer": False,
            "anchor_role": "detached_action_representation_only",
            "trainer_receives_detached_representation_cache_only": True,
            "input_video_path_persisted": False,
            "input_rgb_frames_persisted": False,
            "input_vae_or_clean_latent_persisted": False,
            "absolute_action_hidden_persisted": False,
            "absolute_noop_hidden_persisted": False,
            "raw_q_or_k_or_value_persisted": False,
            "model_endpoint_or_velocity_persisted": False,
            "self_generated_rgb_or_latent_copied_to_output": False,
            "ephemeral_posterior_broadcast_inside_frozen_extractor_only": True,
            "broadcast_posterior_payload_persisted": False,
            "ephemeral_absolute_hidden_zero_reference_released_before_publication": True,
        },
        "training_authority": {
            "optimizer_created": False,
            "optimization_steps": 0,
            "generator_parameters_updated": False,
            "cache_is_not_a_flow_matching_target": True,
        },
        "model_identity": dict(model_identity),
        "runtime_identity": dict(runtime_identity),
        "method_source_sha256": require_sha256(
            method_source_sha256, label="method source"
        ),
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    firewall = receipt.get("information_firewall", {})
    training = receipt.get("training_authority", {})
    cache = receipt.get("cache", {})
    representation = receipt.get("representation", {})
    projection = representation.get("projection", {})
    gaussian_match = representation.get("gaussian_match", {})
    gaussian_authority = validate_explicit_gaussian_authority_record(
        gaussian_match.get("authority", {})
        if isinstance(gaussian_match, Mapping)
        else {}
    )
    deterministic_vae_authority = validate_deterministic_vae_authority(
        representation.get("deterministic_vae_authority", {})
        if isinstance(representation, Mapping)
        else {}
    )
    role = receipt.get("anchor_source_role")
    expected_keys = sorted(f"middle_block_{index:02d}" for index in BLOCK_INDICES)
    sigmas = representation.get("sigmas")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("complete") is not True
        or receipt.get("representation_origin") != "decoded_video_reencode"
        or role not in CONTROL_ROLES
        or object_sha256(candidate) != digest
        or cache.get("filename") != "middle_repr.safetensors"
        or cache.get("schema_version") != CACHE_SCHEMA
        or cache.get("tensor_key_allowlist") != expected_keys
        or sorted((cache.get("tensors") or {}).keys()) != expected_keys
        or representation.get("blocks") != list(BLOCK_INDICES)
        or representation.get("capture") != "post_transformer_block_output"
        or representation.get("contrast")
        != "decoded_action_minus_exact_first_frame_repeat"
        or representation.get("decoded_video_reencode") is not True
        or representation.get("selfgen_native_trajectory") is not False
        or representation.get("noop_constructed_inside_extractor") is not True
        or representation.get("first_frame_repeat_rgb_exact") is not True
        or representation.get("same_caption") is not True
        or representation.get("same_gaussian") is not True
        or representation.get("same_timestep") is not True
        or representation.get("same_rotary") is not True
        or representation.get("phase0_clean_max_abs_error") != 0.0
        or representation.get("phase0_match_atol") != 0.0
        or deterministic_vae_authority.get("posterior_phase0_bit_exact")
        is not True
        or deterministic_vae_authority.get("posterior_modified_after_encode")
        is not False
        or deterministic_vae_authority.get("posterior_copy_or_splice_used")
        is not False
        or "noise_match_atol" in representation
        or not isinstance(gaussian_match, Mapping)
        or set(gaussian_match)
        != {
            "comparison_stage",
            "criterion",
            "inverse_recovery_numerical_fields_applicable",
            "canonical_gaussian_sha256",
            "both_branches_retimed_from_canonical_gaussian",
            "fixed_absolute_tolerance_is_authority",
            "authority",
        }
        or representation.get("noise_max_abs_error") != 0.0
        or gaussian_match.get("comparison_stage") != "before_fm_interpolation"
        or gaussian_match.get("criterion")
        != (
            "same_canonical_raw_fp32_tensor_injected_exactly_once_"
            "per_branch"
        )
        or gaussian_match.get("inverse_recovery_numerical_fields_applicable")
        is not False
        or type(gaussian_match.get("canonical_gaussian_sha256")) is not str
        or _SHA256.fullmatch(
            str(gaussian_match.get("canonical_gaussian_sha256", ""))
        )
        is None
        or gaussian_match.get("canonical_gaussian_sha256")
        != gaussian_authority.get("canonical_gaussian_sha256")
        or gaussian_match.get("both_branches_retimed_from_canonical_gaussian")
        is not True
        or gaussian_match.get("fixed_absolute_tolerance_is_authority") is not False
        or not isinstance(sigmas, list)
        or not sigmas
        or len(sigmas) > 8
        or len(set(sigmas)) != len(sigmas)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
            for value in (sigmas or [])
        )
        or projection.get("kind")
        != "case_independent_fixed_rademacher_jl"
        or projection.get("fitted_on_input_video") is not False
        or not isinstance(projection.get("width"), int)
        or not 0 < projection.get("width", 0) <= HIDDEN_WIDTH
        or firewall.get("target_video_accessed_by_trainer") is not False
        or firewall.get("target_video_accessed_by_extractor")
        is not (role not in SELF_GENERATED_CONTROL_ROLES)
        or firewall.get("target_rgb_or_vae_used_by_frozen_extractor")
        is not (role not in SELF_GENERATED_CONTROL_ROLES)
        or firewall.get("target_rgb_or_vae_target_used_by_trainer") is not False
        or firewall.get("anchor_role")
        != "detached_action_representation_only"
        or firewall.get("trainer_receives_detached_representation_cache_only")
        is not True
        or firewall.get("input_video_path_persisted") is not False
        or firewall.get("input_rgb_frames_persisted") is not False
        or firewall.get("input_vae_or_clean_latent_persisted") is not False
        or firewall.get("absolute_action_hidden_persisted") is not False
        or firewall.get("absolute_noop_hidden_persisted") is not False
        or firewall.get("raw_q_or_k_or_value_persisted") is not False
        or firewall.get("model_endpoint_or_velocity_persisted") is not False
        or firewall.get(
            "ephemeral_posterior_broadcast_inside_frozen_extractor_only"
        )
        is not True
        or firewall.get("broadcast_posterior_payload_persisted") is not False
        or training.get("optimizer_created") is not False
        or training.get("optimization_steps") != 0
        or training.get("generator_parameters_updated") is not False
    ):
        fail("decoded middle representation receipt closure differs")
    serialized = canonical_json_bytes(receipt).decode("ascii").casefold()
    for forbidden in (
        '"input_video_path"',
        '"target_video_path"',
        '"rgb_frames"',
        '"vae_latent"',
        '"clean_latent"',
        '"absolute_hidden"',
        '"raw_query"',
        '"raw_key"',
        '"raw_value"',
    ):
        if forbidden in serialized:
            fail(f"forbidden trainer receipt field is present: {forbidden}")


def _load_unique_json(path: Path) -> dict[str, Any]:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate JSON key in representation receipt: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DecodedMiddleRepresentationError(
            "cannot read middle representation receipt"
        ) from error
    if not isinstance(value, dict):
        fail("middle representation receipt must contain one object")
    return value


def load_middle_representation_cache(
    cache_path: Path,
    receipt_path: Path,
    *,
    expected_role: Optional[str] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Fail-closed G1 consumer for one representation/control cache.

    This is the only supported scorer-facing loader.  It authenticates the
    receipt, cache bytes, safetensors metadata, key allowlist, tensor geometry,
    and per-tensor hashes before returning detached CPU tensors.
    """

    cache_requested = cache_path.expanduser().absolute()
    receipt_requested = receipt_path.expanduser().absolute()
    try:
        cache_lstat = cache_requested.lstat()
        receipt_lstat = receipt_requested.lstat()
        cache = cache_requested.resolve(strict=True)
        receipt_file = receipt_requested.resolve(strict=True)
    except OSError as error:
        raise DecodedMiddleRepresentationError(
            "middle representation cache/receipt is unavailable"
        ) from error
    if (
        cache != cache_requested
        or receipt_file != receipt_requested
        or not os.path.isfile(cache_requested)
        or not os.path.isfile(receipt_requested)
        or os.path.islink(cache_requested)
        or os.path.islink(receipt_requested)
        or not cache.is_file()
        or not receipt_file.is_file()
        or cache_lstat.st_nlink != 1
        or receipt_lstat.st_nlink != 1
    ):
        fail("middle representation cache/receipt topology differs")
    receipt = _load_unique_json(receipt_file)
    validate_receipt(receipt)
    role = canonical_input_role(str(receipt["anchor_source_role"]))
    if expected_role is not None and role != canonical_input_role(expected_role):
        fail("middle representation role differs from scorer request")
    cache_row = receipt["cache"]
    if (
        cache.name != cache_row.get("filename")
        or file_sha256(cache) != cache_row.get("sha256")
    ):
        fail("middle representation cache byte binding differs")
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as error:
        raise DecodedMiddleRepresentationError(
            "middle representation loading requires safetensors"
        ) from error
    with safe_open(str(cache), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        stored_keys = sorted(handle.keys())
    if (
        metadata.get("schema_version") != CACHE_SCHEMA
        or metadata.get("representation_origin") != "decoded_video_reencode"
        or metadata.get("anchor_source_role") != role
        or metadata.get("blocks") != ",".join(map(str, BLOCK_INDICES))
        or metadata.get("sigmas")
        != ",".join(
            f"{float(value):.9g}" for value in receipt["representation"]["sigmas"]
        )
        or metadata.get("projection_width")
        != str(receipt["representation"]["projection"]["width"])
        or metadata.get("contains_detached_projected_residuals_only") != "true"
        or metadata.get("contains_rgb_latent_absolute_hidden_qkv_or_endpoint")
        != "false"
        or stored_keys != cache_row.get("tensor_key_allowlist")
    ):
        fail("middle representation safetensors metadata closure differs")
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in load_file(str(cache), device="cpu").items()
    }
    tensor_rows = validate_cache_tensors(
        tensors,
        sigma_count=len(receipt["representation"]["sigmas"]),
        projection_width=int(receipt["representation"]["projection"]["width"]),
    )
    if tensor_rows != cache_row.get("tensors"):
        fail("middle representation per-tensor receipt differs")
    return tensors, receipt


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        fail(f"create-only JSON output already exists: {path}")
    raw = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_safetensors(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, str],
) -> None:
    if path.exists() or path.is_symlink():
        fail(f"create-only cache output already exists: {path}")
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise DecodedMiddleRepresentationError(
            "production cache publication requires safetensors"
        ) from error
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".safetensors", prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(dict(tensors), str(temporary), metadata=dict(metadata))
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _all_gather_equal(value: Any, *, label: str) -> list[Any]:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return [value]
    rows: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(rows, value)
    if len({canonical_json_bytes(row) for row in rows}) != 1:
        fail(f"WORLD4 ranks disagree on {label}")
    return rows


def _all_gather_rows(value: Any) -> list[Any]:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return [value]
    rows: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(rows, value)
    return rows


def broadcast_rank0_posterior_envelope(
    envelope: Optional[Mapping[str, Any]],
    *,
    rank: int,
    device: torch.device,
) -> Mapping[str, Any]:
    """Broadcast one rank0-only canonical envelope on the WORLD4 device.

    The explicit CUDA device is required for an NCCL/RCCL object collective.
    Nonzero ranks enter with ``None`` and therefore never own decoded RGB or a
    VAE instance.  The received object remains extractor-local and is released
    before any representation cache is published.
    """

    import torch.distributed as dist

    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() != 4
        or dist.get_rank() != rank
        or rank not in range(4)
        or not isinstance(device, torch.device)
        or device.type != "cuda"
        or (rank == 0 and not isinstance(envelope, Mapping))
        or (rank != 0 and envelope is not None)
    ):
        fail("rank0 posterior WORLD4 broadcast contract differs")
    values: list[Any] = [envelope if rank == 0 else None]
    dist.broadcast_object_list(values, src=0, device=device)
    received = values[0]
    if not isinstance(received, Mapping):
        fail("rank0 posterior WORLD4 broadcast returned no envelope")
    return received


def _model_load_guard(serialized_model_load: Any) -> Any:
    if os.environ.get("SLURM_JOB_ID", "").isdigit():
        return serialized_model_load()
    return nullcontext()


def trim_runtime_memory(*, device: Optional[torch.device] = None) -> Mapping[str, Any]:
    """Release Python/CUDA caches and audit glibc ``malloc_trim(0)``."""

    gc.collect()
    cuda_cache_released = False
    if torch.cuda.is_available():
        if device is not None:
            torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        cuda_cache_released = True
    malloc_trim_available = False
    malloc_trim_returned = False
    try:
        libc = ctypes.CDLL(None)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        malloc_trim_available = True
        malloc_trim_returned = bool(trim(0))
    except (AttributeError, OSError):
        pass
    return {
        "python_gc_collected": True,
        "cuda_cache_released": cuda_cache_released,
        "malloc_trim_available": malloc_trim_available,
        # A false return means glibc found no releasable top chunk; the call
        # still occurred and remains auditable.
        "malloc_trim_returned_nonzero": malloc_trim_returned,
    }


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--video-sha256", required=True)
    parser.add_argument(
        "--input-role",
        choices=(*CONTROL_ROLES, "shuffle"),
        required=True,
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sigmas",
        default=",".join(str(value) for value in DEFAULT_SIGMAS),
    )
    parser.add_argument("--projection-width", type=int, default=DEFAULT_PROJECTION_WIDTH)
    parser.add_argument("--projection-seed", type=int, default=DEFAULT_PROJECTION_SEED)
    parser.add_argument("--seed", type=int, default=2026082402)
    parser.add_argument("--max-pixels", type=int, default=245_760)
    parser.add_argument("--stride", type=int, default=16)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[float, ...]:
    if _SAFE_CASE_ID.fullmatch(args.case_id) is None:
        fail("case-id contains unsafe characters")
    require_sha256(args.video_sha256, label="video")
    if (
        not isinstance(args.instruction, str)
        or args.instruction != args.instruction.strip()
        or not args.instruction
        or "\x00" in args.instruction
    ):
        fail("instruction must be non-empty stripped text")
    if (
        args.projection_width <= 0
        or args.projection_width > HIDDEN_WIDTH
        or args.projection_seed < 0
        or args.seed < 0
        or args.max_pixels < args.stride * args.stride
        or args.stride <= 0
    ):
        fail("numeric extractor arguments differ")
    return parse_sigmas(args.sigmas)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    sigmas = _validate_args(args)
    input_role = canonical_input_role(args.input_role)
    instruction_sha = hashlib.sha256(args.instruction.encode("utf-8")).hexdigest()

    # Heavy Bernini imports are intentionally delayed until production entry.
    import torch.distributed as dist
    from transformers import AutoTokenizer

    import train_lora as legacy
    import train_self_generated_action_quotient_v1 as data

    bernini_root, veomni_root, bernini_revision, veomni_revision = (
        legacy.validate_source_trees(args.bernini_root, args.veomni_root)
    )
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    legacy.activate_source_trees(bernini_root, veomni_root)
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.training import data as bernini_data

    NoiseScheduler = bernini_data.NoiseScheduler

    contract = legacy.distributed_contract()
    if contract.world_size != 4 or contract.ulysses_size != 4:
        fail("production decoded middle extraction requires WORLD4/Ulysses-SP4")
    device, backend = legacy.initialise_distributed(contract)
    init_parallel_state(ulysses_size=4)

    video_requested = Path(args.video).expanduser().absolute()
    try:
        video_lstat = video_requested.lstat()
        video_path = video_requested.resolve(strict=True)
    except OSError as error:
        raise DecodedMiddleRepresentationError(
            "input video is unavailable"
        ) from error
    if (
        video_path != video_requested
        or not video_path.is_file()
        or os.path.islink(video_requested)
        or video_lstat.st_nlink != 1
    ):
        fail("input video must be one plain non-symlink file")
    if file_sha256(video_path) != args.video_sha256:
        fail("input video SHA-256 differs")
    output_root = Path(args.output).expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        fail("decoded middle output must be fresh")
    _all_gather_equal(
        {
            "video_sha256": args.video_sha256,
            "input_role": input_role,
            "case_id": args.case_id,
            "instruction_sha256": instruction_sha,
            "sigmas": sigmas,
            "projection_width": args.projection_width,
            "projection_seed": args.projection_seed,
            "seed": args.seed,
        },
        label="extractor invocation",
    )

    # The local materializer is source-hash pinned.  Only rank0 decodes RGB and
    # instantiates the VAE.  It authenticates the returned torch.save transport
    # against the materializer's tensor-semantic digest, then broadcasts one
    # path-free canonical raw-tensor envelope.  This avoids both non-canonical
    # archive identity and per-device VAE numerical drift.
    materializer = exact_video.install_exact_local_video_materializer()
    memory_trim_events: dict[str, Mapping[str, Any]] = {}
    rank0_envelope: Optional[Mapping[str, Any]] = None
    if contract.rank == 0:
        with _model_load_guard(data.serialized_model_load):
            frames, fps_rank0, input_hw_rank0 = materializer._decode_exact_video(
                video_path
            )
            bucket_hw_rank0 = materializer.source_aspect_bucket(
                *input_hw_rank0,
                max_pixels=args.max_pixels,
                stride=args.stride,
            )
            action_rgb = materializer._resize_video(
                frames, bucket_hw_rank0, None
            )
            noop_rgb = (
                action_rgb[:, :1]
                .expand_as(action_rgb)
                .clone()
                .contiguous()
            )
            if (
                tuple(action_rgb.shape[:2]) != (3, 81)
                or not torch.equal(action_rgb[:, 0], noop_rgb[:, 0])
                or not torch.equal(
                    noop_rgb, noop_rgb[:, :1].expand_as(noop_rgb)
                )
            ):
                fail("decoded first-frame-repeat RGB construction differs")
            action_rgb_sha_rank0 = tensor_sha256(action_rgb)
            noop_rgb_sha_rank0 = tensor_sha256(noop_rgb)
            encoder = materializer.BerniniVaeEncoder(
                checkpoint, device=str(device)
            )
            with strict_deterministic_vae_encode_scope() as vae_scope_state:
                encoded_action_blob, action_vae_meta = encoder.encode(action_rgb)
                encoded_noop_blob, noop_vae_meta = encoder.encode(noop_rgb)
            action_posterior_rank0, action_transport_identity = (
                load_validated_materializer_posterior(
                    encoded_action_blob,
                    action_vae_meta,
                    label="action",
                )
            )
            noop_posterior_rank0, noop_transport_identity = (
                load_validated_materializer_posterior(
                    encoded_noop_blob,
                    noop_vae_meta,
                    label="no-op",
                )
            )
            if (
                action_transport_identity["shape"]
                != noop_transport_identity["shape"]
            ):
                fail("action/no-op VAE posterior geometry differs")
            deterministic_vae_authority_rank0 = (
                build_deterministic_vae_authority(
                    vae_scope_state,
                    action=action_posterior_rank0,
                    noop=noop_posterior_rank0,
                )
            )
            rank0_envelope = build_rank0_posterior_envelope(
                action=action_posterior_rank0,
                noop=noop_posterior_rank0,
                fps=fps_rank0,
                input_hw=input_hw_rank0,
                bucket_hw=bucket_hw_rank0,
                action_rgb_sha256=action_rgb_sha_rank0,
                noop_rgb_sha256=noop_rgb_sha_rank0,
                deterministic_vae_authority=(
                    deterministic_vae_authority_rank0
                ),
            )
            del (
                encoder,
                frames,
                action_rgb,
                noop_rgb,
                encoded_action_blob,
                encoded_noop_blob,
                action_vae_meta,
                noop_vae_meta,
                action_posterior_rank0,
                noop_posterior_rank0,
                action_transport_identity,
                noop_transport_identity,
                deterministic_vae_authority_rank0,
                vae_scope_state,
            )
            memory_trim_events["after_rank0_vae_encode"] = trim_runtime_memory(
                device=device
            )
    received_envelope = broadcast_rank0_posterior_envelope(
        rank0_envelope,
        rank=contract.rank,
        device=device,
    )
    pair = unpack_rank0_posterior_envelope(received_envelope)
    fps = pair.fps
    input_hw = pair.input_hw
    bucket_hw = pair.bucket_hw
    action_rgb_sha = pair.action_rgb_sha256
    noop_rgb_sha = pair.noop_rgb_sha256
    posterior_shape = tuple(pair.action_identity["shape"])
    if pair.deterministic_vae_authority is None:
        fail("production decoded middle extraction lacks deterministic VAE authority")
    deterministic_vae_authority = dict(pair.deterministic_vae_authority)
    _all_gather_equal(
        deterministic_vae_authority,
        label="deterministic VAE encode authority",
    )
    posterior_identity = {
        "identity_kind": "sha256_dtype_shape_raw_tensor_bytes",
        "producer_rank": 0,
        "action_tensor_sha256": pair.action_identity["tensor_sha256"],
        "noop_tensor_sha256": pair.noop_identity["tensor_sha256"],
        "posterior_dtype": pair.action_identity["dtype"],
        "posterior_shape": list(posterior_shape),
    }
    _all_gather_equal(
        posterior_identity, label="canonical VAE posterior extraction"
    )
    gaussian_shape = explicit_gaussian_packed_shape(posterior_shape)
    gaussian_seed = derive_explicit_gaussian_seed(
        base_seed=args.seed,
        case_id=args.case_id,
        instruction_sha256=instruction_sha,
    )
    rank0_gaussian = (
        generate_rank0_explicit_gaussian(
            gaussian_shape, derived_seed=gaussian_seed
        )
        if contract.rank == 0
        else None
    )
    canonical_gaussian = broadcast_rank0_explicit_gaussian(
        rank0_gaussian,
        expected_shape=gaussian_shape,
        rank=contract.rank,
        device=device,
    )
    gaussian_broadcast_identity = {
        "domain": EXPLICIT_GAUSSIAN_DOMAIN,
        "producer_rank": 0,
        "base_seed": args.seed,
        "derived_seed": gaussian_seed,
        "dtype": str(canonical_gaussian.dtype),
        "shape": list(map(int, canonical_gaussian.shape)),
        "tensor_sha256": tensor_sha256(canonical_gaussian),
        "broadcast_transport": (
            "torch_distributed_nccl_fp32_tensor_broadcast"
        ),
    }
    _all_gather_equal(
        gaussian_broadcast_identity,
        label="canonical explicit pre-pack Gaussian raw SHA",
    )
    del rank0_gaussian
    action_blob = posterior_tensor_to_transport_blob(pair.action)
    noop_blob = posterior_tensor_to_transport_blob(pair.noop)
    # Verify the exact legacy transport consumed below on every rank.  Archive
    # bytes are deliberately absent from identity and from all receipts.
    action_roundtrip, action_roundtrip_identity = (
        load_validated_materializer_posterior(
            action_blob,
            {
                "posterior_parameters_shape": list(posterior_shape),
                "posterior_parameters_dtype": pair.action_identity["dtype"],
                "posterior_parameters_tensor_sha256": pair.action_identity[
                    "tensor_sha256"
                ],
            },
            label="broadcast action",
        )
    )
    noop_roundtrip, noop_roundtrip_identity = load_validated_materializer_posterior(
        noop_blob,
        {
            "posterior_parameters_shape": list(posterior_shape),
            "posterior_parameters_dtype": pair.noop_identity["dtype"],
            "posterior_parameters_tensor_sha256": pair.noop_identity[
                "tensor_sha256"
            ],
        },
        label="broadcast no-op",
    )
    if (
        action_roundtrip_identity != pair.action_identity
        or noop_roundtrip_identity != pair.noop_identity
    ):
        fail("broadcast posterior legacy transport semantic identity differs")
    del (
        action_roundtrip,
        noop_roundtrip,
        action_roundtrip_identity,
        noop_roundtrip_identity,
        pair,
        received_envelope,
        rank0_envelope,
    )
    memory_trim_events["after_canonical_posterior_broadcast"] = (
        trim_runtime_memory(device=device)
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    with _model_load_guard(data.serialized_model_load):
        renderer = BerniniRendererModel(config)
        renderer.eval().requires_grad_(False)
        renderer.t5_text_encoder.eval()
        renderer.to(device)
        memory_trim_events["after_serial_renderer_load"] = trim_runtime_memory(
            device=device
        )
    transformer = renderer.diff_dec.transformer
    if (
        transformer is None
        or renderer.diff_dec.transformer_2 is not None
        or len(tuple(getattr(transformer, "blocks", ()))) != 30
    ):
        fail("decoded middle extractor requires one exact30 Wan transformer")
    parameter_versions_before = {
        name: int(parameter._version) for name, parameter in renderer.named_parameters()
    }
    if any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in renderer.parameters()
    ):
        fail("frozen extractor unexpectedly exposes trainable parameters")

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    mean, std, _ = legacy._vae_statistics(checkpoint)
    scheduler = NoiseScheduler(**legacy.noise_scheduler_kwargs())
    transform, vendor_data_identity = build_explicit_gaussian_renderer_transform(
        process_renderer_sample=bernini_data.process_renderer_sample,
        pack_vae_latents=bernini_data.pack_vae_latents,
        tokenizer=tokenizer,
        rope=rope,
        mean=mean,
        std=std,
        scheduler=scheduler,
        device=device,
        gaussian=canonical_gaussian,
        collate=data.collate,
        seed_same_sample=legacy.seed_same_sample,
        source_name=legacy.TASK_SOURCE_NAME,
    )
    action_batch, action_pack_capture = transform(
        data.make_sample(
            instruction=args.instruction,
            source_blob=None,
            target_blob=action_blob,
        ),
        args.seed,
    )
    noop_batch, noop_pack_capture = transform(
        data.make_sample(
            instruction=args.instruction,
            source_blob=None,
            target_blob=noop_blob,
        ),
        args.seed,
    )
    del action_blob, noop_blob, canonical_gaussian
    del tokenizer, rope, scheduler, transform, mean, std, materializer
    memory_trim_events["after_batch_construction"] = trim_runtime_memory(
        device=device
    )

    spatial_shape = (
        1,
        16,
        PHASES,
        int(posterior_shape[3]),
        int(posterior_shape[4]),
    )
    matched = matched_patch_pair_from_explicit_captures(
        action_batch,
        noop_batch,
        action_capture=action_pack_capture,
        noop_capture=noop_pack_capture,
        spatial_shape=spatial_shape,
        patches_to_spatial=data.patches_to_spatial,
        base_seed=args.seed,
        derived_seed=gaussian_seed,
        vendor_identity=vendor_data_identity,
        world_size=contract.world_size,
    )
    if (
        matched.canonical_gaussian_sha256
        != gaussian_broadcast_identity["tensor_sha256"]
    ):
        fail("captured Gaussian differs from WORLD4 broadcast authority")
    matched_state_metrics = {
        "noise_max_abs_error": float(matched.noise_max_abs_error),
        "noise_max_abs_forward_error_bound": float(
            matched.noise_max_abs_forward_error_bound
        ),
        "noise_max_error_to_bound_ratio": float(
            matched.noise_max_error_to_bound_ratio
        ),
        "noise_original_dtype": matched.noise_original_dtype,
        "noise_dtype_epsilon": float(matched.noise_dtype_epsilon),
        "canonical_gaussian_sha256": matched.canonical_gaussian_sha256,
        "phase0_clean_max_abs_error": float(matched.phase0_clean_max_abs_error),
        "original_sampled_sigma": float(matched.original_sigma),
        "gaussian_authority": dict(matched.gaussian_authority),
    }
    _all_gather_equal(matched_state_metrics, label="matched FM state")
    del (
        action_pack_capture,
        noop_pack_capture,
        vendor_data_identity,
        gaussian_broadcast_identity,
    )
    memory_trim_events["after_explicit_prepack_authority_capture"] = (
        trim_runtime_memory(device=device)
    )
    patch_height = int(spatial_shape[-2]) // 2
    patch_width = int(spatial_shape[-1]) // 2
    layout = anchor_core.LocalTokenLayout.build(
        condition_tokens=0,
        patch_height=patch_height,
        patch_width=patch_width,
        phases=PHASES,
        sp_rank=contract.rank,
        sp_size=4,
    )
    if int(matched.selector.numel()) != PHASES * patch_height * patch_width:
        fail("target-only packed token count differs from decoded patch grid")

    projection = deterministic_projection(
        HIDDEN_WIDTH,
        args.projection_width,
        seed=args.projection_seed,
        device=device,
    )
    projection_sha = tensor_sha256(projection)
    _all_gather_equal(projection_sha, label="fixed projection")
    captures = MiddleBlockCaptureBank(
        transformer, hidden_width=HIDDEN_WIDTH
    )
    captures.install()
    per_block: dict[int, list[torch.Tensor]] = {index: [] for index in BLOCK_INDICES}
    metrics: dict[str, Any] = {}
    try:
        for sigma_index, sigma in enumerate(sigmas):
            action_sigma = retime_fm_batch(
                action_batch,
                clean=matched.action_clean,
                gaussian=matched.gaussian,
                selector=matched.selector,
                sigma=sigma,
            )
            noop_sigma = retime_fm_batch(
                noop_batch,
                clean=matched.noop_clean,
                gaussian=matched.gaussian,
                selector=matched.selector,
                sigma=sigma,
            )
            with captures.capture(layout), torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                data.predicted_target_velocity(
                    renderer, action_sigma, spatial_shape=spatial_shape
                )
            action_captures = captures.pop()
            with captures.capture(layout), torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                data.predicted_target_velocity(
                    renderer, noop_sigma, spatial_shape=spatial_shape
                )
            noop_captures = captures.pop()

            for block_index in BLOCK_INDICES:
                action_local = action_captures.pop(block_index)
                noop_local = noop_captures.pop(block_index)
                appearance_direction = distributed_appearance_direction(noop_local)
                local_delta = (
                    action_local.float() - noop_local.float()
                ).detach().contiguous()
                full_delta = native_bridge.distributed_assemble_local_delta(
                    local_delta, layout
                )
                full_delta = full_delta.reshape(
                    1,
                    PHASES,
                    patch_height,
                    patch_width,
                    HIDDEN_WIDTH,
                ).contiguous()
                zeros = torch.zeros_like(full_delta)
                projected, row_metrics = preprocess_middle_delta(
                    action_hidden=full_delta,
                    noop_hidden=zeros,
                    appearance_direction=appearance_direction,
                    projection=projection,
                )
                stored = projected.to(device="cpu", dtype=torch.float16).contiguous()
                per_block[block_index].append(stored)
                metrics[f"sigma_{sigma_index:02d}_block_{block_index:02d}"] = {
                    "sigma": float(sigma),
                    **dict(row_metrics),
                }
                # Absolute captures, appearance direction, and full-width
                # residual are all released before the next block.
                del (
                    action_local,
                    noop_local,
                    appearance_direction,
                    local_delta,
                    full_delta,
                    zeros,
                    projected,
                )
            if action_captures or noop_captures:
                fail("ephemeral four-block capture registry did not empty")
            del action_sigma, noop_sigma
    finally:
        captures.remove()

    cache_tensors = {
        f"middle_block_{block_index:02d}": torch.stack(
            per_block[block_index], dim=0
        ).contiguous()
        for block_index in BLOCK_INDICES
    }
    del per_block, matched, action_batch, noop_batch, projection
    gc.collect()
    torch.cuda.empty_cache()

    parameter_versions_after = {
        name: int(parameter._version) for name, parameter in renderer.named_parameters()
    }
    if (
        parameter_versions_before != parameter_versions_after
        or any(parameter.grad is not None for parameter in renderer.parameters())
    ):
        fail("frozen extractor parameters changed during representation capture")
    del captures, transformer, renderer, config
    memory_trim_events["after_renderer_release"] = trim_runtime_memory(
        device=device
    )
    memory_trim_world = _all_gather_rows(
        {"rank": contract.rank, "events": memory_trim_events}
    )
    cache_tensor_receipts = validate_cache_tensors(
        cache_tensors,
        sigma_count=len(sigmas),
        projection_width=args.projection_width,
    )
    _all_gather_equal(cache_tensor_receipts, label="projected representation tensors")

    if contract.rank == 0:
        output_root.mkdir(parents=True, exist_ok=False)
        cache_path = output_root / "middle_repr.safetensors"
        _atomic_safetensors(
            cache_path,
            cache_tensors,
            metadata={
                "schema_version": CACHE_SCHEMA,
                "method": METHOD,
                "representation_origin": "decoded_video_reencode",
                "anchor_source_role": input_role,
                "blocks": ",".join(map(str, BLOCK_INDICES)),
                "sigmas": ",".join(f"{value:.9g}" for value in sigmas),
                "projection_width": str(args.projection_width),
                "contains_detached_projected_residuals_only": "true",
                "contains_rgb_latent_absolute_hidden_qkv_or_endpoint": "false",
            },
        )
        cache_sha = file_sha256(cache_path)
        checkpoint_identity = {
            "checkpoint_tree_sha256": legacy.CHECKPOINT_TREE_SHA256,
            "transformer_config_sha256": file_sha256(
                checkpoint / "transformer/config.json"
            ),
            "vae_config_sha256": file_sha256(checkpoint / "vae/config.json"),
            "base_eval": True,
            "base_frozen": True,
            "optimizer_absent": True,
            "parameter_version_counters_unchanged": True,
        }
        runtime_identity = {
            "world_size": contract.world_size,
            "ulysses_size": contract.ulysses_size,
            "backend": backend,
            "bernini_revision": bernini_revision,
            "veomni_revision": veomni_revision,
            "video_frame_count": 81,
            "video_fps": fps,
            "input_hw": list(input_hw),
            "bucket_hw": list(bucket_hw),
            "action_rgb_tensor_sha256": action_rgb_sha,
            "first_frame_repeat_rgb_tensor_sha256": noop_rgb_sha,
            "vae_posterior_producer_rank": 0,
            "vae_posterior_rank_policy": (
                "rank0_only_encode_world4_nccl_broadcast_canonical_raw_tensor"
            ),
            "vae_posterior_identity_kind": posterior_identity["identity_kind"],
            "vae_action_posterior_tensor_sha256": posterior_identity[
                "action_tensor_sha256"
            ],
            "vae_noop_posterior_tensor_sha256": posterior_identity[
                "noop_tensor_sha256"
            ],
            "vae_posterior_shape": posterior_identity["posterior_shape"],
            "vae_posterior_dtype": posterior_identity["posterior_dtype"],
            "torch_save_blob_bytes_used_as_cross_rank_identity": False,
            "canonical_posterior_payload_persisted": False,
            "absolute_media_or_latent_bytes_persisted": False,
            "memory_trim_world": memory_trim_world,
        }
        receipt = build_receipt(
            case_id=args.case_id,
            input_role=input_role,
            input_video_sha256=args.video_sha256,
            instruction_sha256=instruction_sha,
            cache_path=cache_path,
            cache_sha256=cache_sha,
            cache_tensors=cache_tensor_receipts,
            sigmas=sigmas,
            projection_width=args.projection_width,
            projection_seed=args.projection_seed,
            projection_sha256=projection_sha,
            patch_grid=(PHASES, patch_height, patch_width),
            noise_max_abs_error=matched_state_metrics["noise_max_abs_error"],
            noise_max_abs_forward_error_bound=matched_state_metrics[
                "noise_max_abs_forward_error_bound"
            ],
            noise_max_error_to_bound_ratio=matched_state_metrics[
                "noise_max_error_to_bound_ratio"
            ],
            noise_original_dtype=matched_state_metrics["noise_original_dtype"],
            noise_dtype_epsilon=matched_state_metrics["noise_dtype_epsilon"],
            canonical_gaussian_sha256=matched_state_metrics[
                "canonical_gaussian_sha256"
            ],
            gaussian_authority=matched_state_metrics["gaussian_authority"],
            deterministic_vae_authority=deterministic_vae_authority,
            phase0_clean_max_abs_error=matched_state_metrics[
                "phase0_clean_max_abs_error"
            ],
            block_metrics=metrics,
            model_identity=checkpoint_identity,
            runtime_identity=runtime_identity,
            method_source_sha256=file_sha256(Path(__file__).resolve()),
        )
        _atomic_json(output_root / "receipt.json", receipt)
        print(
            json.dumps(
                {
                    "complete": True,
                    "case_id": args.case_id,
                    "input_role": input_role,
                    "cache": str(cache_path),
                    "cache_sha256": cache_sha,
                    "receipt_digest": receipt["receipt_digest"],
                    "optimization_steps": 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCK_INDICES",
    "CACHE_SCHEMA",
    "CONTROL_ROLES",
    "DETERMINISTIC_VAE_POLICY",
    "EXPLICIT_GAUSSIAN_DOMAIN",
    "PINNED_BERNINI_DATA_SHA256",
    "PINNED_PACK_VAE_LATENTS_SOURCE_SHA256",
    "PINNED_PROCESS_RENDERER_SAMPLE_SOURCE_SHA256",
    "SELF_GENERATED_CONTROL_ROLES",
    "TARGET_CONTROL_ROLES",
    "DecodedMiddleRepresentationError",
    "ExplicitGaussianPackCapture",
    "MatchedPatchPair",
    "MiddleBlockCaptureBank",
    "RECEIPT_SCHEMA",
    "build_receipt",
    "build_deterministic_vae_authority",
    "build_explicit_gaussian_renderer_transform",
    "broadcast_rank0_explicit_gaussian",
    "canonical_input_role",
    "derive_explicit_gaussian_seed",
    "deterministic_projection",
    "distributed_appearance_direction",
    "explicit_gaussian_packed_shape",
    "generate_rank0_explicit_gaussian",
    "parse_sigmas",
    "preprocess_middle_delta",
    "matched_patch_pair_from_explicit_captures",
    "recover_matched_patch_pair",
    "retime_fm_batch",
    "strict_deterministic_vae_encode_scope",
    "tensor_sha256",
    "validate_cache_tensors",
    "validate_explicit_gaussian_authority_record",
    "validate_deterministic_vae_authority",
    "validate_receipt",
    "load_middle_representation_cache",
    "is_self_generated_role",
]
