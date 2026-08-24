#!/usr/bin/env python3
"""Real Bernini-R 1.3B / ELAL-3 WORLD1 no-update integration probe.

This executable is deliberately narrower than a trainer.  It loads the pinned
real Bernini ``transformer_1`` weights, makes one native source-prefix plus
target-suffix forward, installs the ELAL-3 ``full-w64`` post-block hooks, and
makes one graph-preserving forward/backward on the exact same packed input.

The probe establishes only renderer integration facts:

* exactly 30 real Bernini blocks are hooked;
* every hook writes the target suffix only and leaves its immediate source
  input/output rows bit-exact;
* the final target prediction responds to ELAL-3;
* every one of the 30 ELAL-3 injections receives a finite non-zero gradient;
* frozen Bernini parameter version counters and all ELAL-3 parameter bytes are
  unchanged because no optimizer is constructed or stepped.

The deterministic finite tensors used here are engineering probe inputs.  They
are not a video dataset, action supervision, an ActionPredictor, or evidence of
source+instruction action-editing quality.  A passing receipt never authorizes
training or scientific promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

import elal3_c0_v1 as elal3
import train_lora as legacy


RECEIPT_SCHEMA_VERSION = "bernini-elal3-real-integration-probe-receipt-v1"
METHOD_NAME = "ELAL-3 real Bernini WORLD1 no-update integration probe v1"
DEFAULT_SEED = 20260817
PROBE_LATENT_SHAPE = (1, 16, 21, 2, 2)
PROBE_TEXT_TOKENS = 4
PROBE_TIMESTEP = 500.0
REGISTERED_ARM = "full-w64"
ROUTE_IDENTITY = "elal3-real-bernini-world1-full-w64-probe-v1"
_SAFE_BASENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ELAL3BerniniIntegrationProbeError(RuntimeError):
    """Raised instead of emitting ambiguous real-model evidence."""


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
        raise ELAL3BerniniIntegrationProbeError(
            "receipt is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    if "receipt_digest" in unsigned:
        raise ELAL3BerniniIntegrationProbeError(
            "unsigned receipt already contains receipt_digest"
        )
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_digest_tensor_bytes(
    digest: "hashlib._Hash", tensor: torch.Tensor
) -> None:
    """Stream CPU tensor bytes without requiring a NumPy runtime."""

    # Flatten first because older PyTorch releases reject dtype views of a
    # zero-dimensional scalar tensor.
    raw = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
    chunk_bytes = 1024 * 1024
    for start in range(0, int(raw.numel()), chunk_bytes):
        digest.update(bytes(raw[start : start + chunk_bytes].tolist()))


def tensor_sha256(value: torch.Tensor, *, label: str) -> str:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ELAL3BerniniIntegrationProbeError(
            f"{label} must be a strided tensor"
        )
    tensor = value.detach().contiguous().cpu()
    metadata = canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": [int(item) for item in tensor.shape],
        }
    )
    digest = hashlib.sha256(b"bernini-elal3-real-probe-tensor-v1\0")
    digest.update(struct.pack(">Q", len(metadata)))
    digest.update(metadata)
    _update_digest_tensor_bytes(digest, tensor)
    return digest.hexdigest()


def named_tensor_digest(
    values: Iterable[tuple[str, torch.Tensor]], *, label: str
) -> str:
    digest = hashlib.sha256(
        f"bernini-elal3-real-probe-named-tensors-v1|{label}\0".encode("ascii")
    )
    count = 0
    for name, value in values:
        if type(name) is not str or not name or not isinstance(value, torch.Tensor):
            raise ELAL3BerniniIntegrationProbeError(
                f"{label} named tensor inventory differs"
            )
        header = canonical_json_bytes(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": [int(item) for item in value.shape],
            }
        )
        tensor = value.detach().contiguous().cpu()
        digest.update(struct.pack(">Q", len(header)))
        digest.update(header)
        _update_digest_tensor_bytes(digest, tensor)
        count += 1
    if count <= 0:
        raise ELAL3BerniniIntegrationProbeError(f"{label} inventory is empty")
    return digest.hexdigest()


def _absolute_directory(value: str | Path, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute() or raw == Path("/"):
        raise ELAL3BerniniIntegrationProbeError(
            f"{label} must be an absolute non-root directory"
        )
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise ELAL3BerniniIntegrationProbeError(
            f"cannot resolve {label}: {error}"
        ) from error
    if not resolved.is_dir():
        raise ELAL3BerniniIntegrationProbeError(f"{label} is not a directory")
    return resolved


def resolve_create_only_output(value: str | Path) -> Path:
    output = Path(value).expanduser()
    if not output.is_absolute() or output == Path("/"):
        raise ELAL3BerniniIntegrationProbeError(
            "--output must be an absolute non-root JSON path"
        )
    if _SAFE_BASENAME.fullmatch(output.name) is None or output.suffix != ".json":
        raise ELAL3BerniniIntegrationProbeError(
            "--output basename must be safe and end in .json"
        )
    if output.exists() or output.is_symlink():
        raise ELAL3BerniniIntegrationProbeError("--output is create-only")
    parent = _absolute_directory(output.parent, label="output parent")
    canonical = parent / output.name
    if canonical != output:
        raise ELAL3BerniniIntegrationProbeError(
            "--output must already use its canonical absolute path"
        )
    return output


def write_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    output = resolve_create_only_output(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o640)
    try:
        payload = canonical_json_bytes(value) + b"\n"
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _namespace_seed(seed: int, namespace: str) -> int:
    if type(seed) is not int or seed < 0 or type(namespace) is not str or not namespace:
        raise ELAL3BerniniIntegrationProbeError("probe seed namespace differs")
    payload = f"elal3-real-bernini-probe-v1|{seed}|{namespace}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _cpu_randn(shape: Sequence[int], *, seed: int, namespace: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_namespace_seed(seed, namespace))
    return torch.randn(
        tuple(int(item) for item in shape),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).contiguous()


def deterministic_probe_latent(
    device: torch.device, *, seed: int
) -> elal3.ELAL3LatentV1:
    """Build one finite fixed-capacity action signal for integration only."""

    def value(shape: Sequence[int], namespace: str) -> torch.Tensor:
        return _cpu_randn(shape, seed=seed, namespace=namespace).to(device).contiguous()

    presence = torch.ones((1, 3), dtype=torch.bool, device=device)
    temporal = torch.ones((1, 3, 21), dtype=torch.bool, device=device)
    relation = torch.ones((1, 6, 21), dtype=torch.bool, device=device)
    phase = torch.ones((1, 21), dtype=torch.bool, device=device)
    latent = elal3.ELAL3LatentV1(
        q_local=value((1, 21, 1, 1, 64), "q_local"),
        q_entity=value((1, 3, 21, 256), "q_entity"),
        q_relation=value((1, 6, 21, 128), "q_relation"),
        q_phase=value((1, 21, 128), "q_phase"),
        q_terminal=value((1, 9, 256), "q_terminal"),
        q_camera=value((1, 21, 128), "q_camera"),
        entity_presence=presence.contiguous(),
        temporal_valid=temporal.contiguous(),
        relation_valid=relation.contiguous(),
        phase_valid=phase.contiguous(),
    )
    latent.validate()
    return latent


def _require_finite_tensor(
    value: Any, *, label: str, ndim: Optional[int] = None
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or not value.is_floating_point()
        or (ndim is not None and value.ndim != ndim)
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise ELAL3BerniniIntegrationProbeError(
            f"{label} must be one finite strided floating tensor"
        )
    return value


def _extract_output(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value
    elif isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
        result = value[0]
    elif isinstance(getattr(value, "sample", None), torch.Tensor):
        result = value.sample
    else:
        raise ELAL3BerniniIntegrationProbeError(
            "real Bernini transformer output ABI differs"
        )
    return _require_finite_tensor(result, label="transformer output", ndim=3)


def _forward_transformer(
    transformer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    text: torch.Tensor,
    rotary: torch.Tensor,
) -> torch.Tensor:
    try:
        value = transformer(
            hidden_states,
            timestep,
            encoder_hidden_states=text,
            rotary_emb=rotary,
            batch_image_vae_seqlen=[int(hidden_states.shape[1])],
            text_features_length=[int(text.shape[1])],
            return_dict=False,
        )
    except Exception as error:
        raise ELAL3BerniniIntegrationProbeError(
            "real Bernini transformer forward failed at the official packed ABI"
        ) from error
    return _extract_output(value)


def _delta_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape or left.device != right.device:
        raise ELAL3BerniniIntegrationProbeError("response tensors differ in geometry")
    delta = left.detach().float() - right.detach().float()
    if not bool(torch.isfinite(delta).all().item()):
        raise ELAL3BerniniIntegrationProbeError("response delta is non-finite")
    absolute = delta.abs()
    return {
        "element_count": int(delta.numel()),
        "nonzero_element_count": int(torch.count_nonzero(delta).item()),
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "rms": float(torch.sqrt(torch.mean(delta.square())).item()),
        "max_abs": float(absolute.max().item()) if delta.numel() else 0.0,
    }


def _gradient_group(
    named: Sequence[tuple[str, nn.Parameter]], *, label: str
) -> dict[str, Any]:
    if not named:
        raise ELAL3BerniniIntegrationProbeError(f"{label} parameter group is empty")
    missing: list[str] = []
    nonfinite: list[str] = []
    nonzero_tensors = 0
    squared_norm = 0.0
    for name, parameter in named:
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        finite = bool(torch.isfinite(gradient.detach()).all().item())
        if not finite:
            nonfinite.append(name)
            continue
        norm = float(torch.linalg.vector_norm(gradient.detach().float()).item())
        if norm > 0.0:
            nonzero_tensors += 1
        squared_norm += norm * norm
    result = {
        "label": label,
        "parameter_tensor_count": len(named),
        "missing_gradient_names": missing,
        "nonfinite_gradient_names": nonfinite,
        "nonzero_gradient_tensor_count": nonzero_tensors,
        "aggregate_l2": math.sqrt(squared_norm),
        "all_gradients_present_finite": not missing and not nonfinite,
        "has_nonzero_gradient": nonzero_tensors > 0 and squared_norm > 0.0,
    }
    return result


def _cuda_memory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
            "device_total_bytes": 0,
            "allocated_fraction_of_device": 0.0,
            "reserved_fraction_of_device": 0.0,
        }
    total = int(torch.cuda.get_device_properties(device).total_memory)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device_total_bytes": total,
        "allocated_fraction_of_device": allocated / total,
        "reserved_fraction_of_device": reserved / total,
    }


def _input_dtype_contract(
    transformer: nn.Module, *, device: torch.device
) -> tuple[torch.dtype, torch.dtype, dict[str, Any]]:
    """Bind native visual/text inputs to their actual ingress projections."""

    patch_embedding = getattr(transformer, "patch_embedding", None)
    patch_weight = getattr(patch_embedding, "weight", None)
    patch_bias = getattr(patch_embedding, "bias", None)
    condition_embedder = getattr(transformer, "condition_embedder", None)
    text_embedder = getattr(condition_embedder, "text_embedder", None)
    text_linear_1 = getattr(text_embedder, "linear_1", None)
    text_weight = getattr(text_linear_1, "weight", None)
    text_bias = getattr(text_linear_1, "bias", None)
    allowed = (torch.float16, torch.bfloat16, torch.float32)
    if (
        not isinstance(patch_weight, torch.Tensor)
        or patch_weight.layout != torch.strided
        or patch_weight.ndim != 5
        or int(patch_weight.shape[1]) != 16
        or patch_weight.device != device
        or patch_weight.dtype not in allowed
    ):
        raise ELAL3BerniniIntegrationProbeError(
            "patch_embedding.weight is not the supported native visual ingress"
        )
    if patch_bias is not None and (
        not isinstance(patch_bias, torch.Tensor)
        or patch_bias.device != device
        or patch_bias.dtype != patch_weight.dtype
    ):
        raise ELAL3BerniniIntegrationProbeError(
            "patch_embedding bias device/dtype differs from its weight"
        )
    if (
        not isinstance(text_weight, torch.Tensor)
        or text_weight.layout != torch.strided
        or text_weight.ndim != 2
        or int(text_weight.shape[1]) != 4096
        or text_weight.device != device
        or text_weight.dtype not in allowed
    ):
        raise ELAL3BerniniIntegrationProbeError(
            "condition_embedder.text_embedder.linear_1.weight is not the supported text ingress"
        )
    if text_bias is not None and (
        not isinstance(text_bias, torch.Tensor)
        or text_bias.device != device
        or text_bias.dtype != text_weight.dtype
    ):
        raise ELAL3BerniniIntegrationProbeError(
            "text ingress bias device/dtype differs from its weight"
        )
    first = next(transformer.parameters(), None)
    if first is None or first.device != device:
        raise ELAL3BerniniIntegrationProbeError(
            "transformer parameters are absent or on the wrong device"
        )
    receipt = {
        "binding": "actual-ingress-projection-weights-not-first-model-parameter",
        "first_model_parameter_dtype": str(first.dtype),
        "patch_embedding_weight_dtype": str(patch_weight.dtype),
        "patch_embedding_bias_dtype": (
            str(patch_bias.dtype) if patch_bias is not None else None
        ),
        "text_linear_1_weight_dtype": str(text_weight.dtype),
        "text_linear_1_bias_dtype": (
            str(text_bias.dtype) if text_bias is not None else None
        ),
        "visual_input_matches_patch_embedding": True,
        "text_input_matches_condition_embedding": True,
    }
    return patch_weight.dtype, text_weight.dtype, receipt


def _model_inputs(
    transformer: nn.Module,
    *,
    device: torch.device,
    seed: int,
    hidden_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    Mapping[str, Any],
]:
    visual_dtype, text_dtype, dtype_contract = _input_dtype_contract(
        transformer, device=device
    )
    source = _cpu_randn(
        PROBE_LATENT_SHAPE, seed=seed, namespace="source_vae_latent"
    ).to(device=device, dtype=visual_dtype)
    target = _cpu_randn(
        PROBE_LATENT_SHAPE, seed=seed, namespace="target_vae_latent"
    ).to(device=device, dtype=visual_dtype)
    if torch.equal(source, target):
        raise ELAL3BerniniIntegrationProbeError(
            "deterministic source and target probe tensors collided"
        )
    patch = getattr(transformer, "patch_vae_latent", None)
    if not callable(patch):
        raise ELAL3BerniniIntegrationProbeError(
            "real Bernini patch_vae_latent is unavailable"
        )
    with torch.no_grad():
        source_tokens, source_rotary = patch(source, source_id=1)
        target_tokens, target_rotary = patch(target, source_id=0)
    for label, value in (
        ("source tokens", source_tokens),
        ("target tokens", target_tokens),
    ):
        _require_finite_tensor(value, label=label, ndim=3)
    if (
        tuple(source_tokens.shape) != (1, 21, hidden_size)
        or tuple(target_tokens.shape) != tuple(source_tokens.shape)
        or source_tokens.dtype != target_tokens.dtype
        or source_tokens.dtype != visual_dtype
        or source_tokens.device != device
    ):
        raise ELAL3BerniniIntegrationProbeError(
            "native exact81 minimal source/target patch geometry differs"
        )
    if (
        not isinstance(source_rotary, torch.Tensor)
        or not isinstance(target_rotary, torch.Tensor)
        or source_rotary.shape != target_rotary.shape
        or tuple(source_rotary.shape[:3]) != (1, 1, 21)
        or source_rotary.device != device
        or target_rotary.device != device
    ):
        raise ELAL3BerniniIntegrationProbeError(
            "native source/target rotary geometry differs"
        )
    hidden = torch.cat((source_tokens, target_tokens), dim=1).detach().contiguous()
    rotary = torch.cat((source_rotary, target_rotary), dim=2).detach().contiguous()
    text = _cpu_randn(
        (1, PROBE_TEXT_TOKENS, 4096), seed=seed, namespace="text_condition"
    ).to(device=device, dtype=text_dtype).detach().contiguous()
    timestep = torch.tensor(
        [PROBE_TIMESTEP], dtype=torch.float32, device=device
    ).contiguous()
    return hidden, rotary, text, timestep, 21, dtype_contract


def run_loaded_transformer_probe(
    transformer: nn.Module,
    *,
    device: torch.device,
    seed: int,
    hidden_size: int = elal3.BERNINI_HIDDEN,
    test_only: bool = False,
) -> dict[str, Any]:
    """Run the no-update probe on an already-loaded transformer.

    ``test_only=True`` exists solely for the dedicated CPU contract tests.  The
    production CLI never exposes or selects it.
    """

    if not isinstance(transformer, nn.Module):
        raise ELAL3BerniniIntegrationProbeError("transformer must be an nn.Module")
    if type(seed) is not int or seed < 0:
        raise ELAL3BerniniIntegrationProbeError("seed must be a non-negative integer")
    blocks = tuple(getattr(transformer, "blocks", ()))
    if len(blocks) != elal3.BERNINI_BLOCKS or len({id(block) for block in blocks}) != 30:
        raise ELAL3BerniniIntegrationProbeError(
            "real integration probe requires 30 distinct Bernini blocks"
        )
    if bool(getattr(transformer, "gradient_checkpointing", False)):
        raise ELAL3BerniniIntegrationProbeError(
            "probe requires gradient checkpointing disabled for a single auditable route"
        )
    transformer.eval().requires_grad_(False)
    base_named = tuple(transformer.named_parameters())
    if not base_named or any(parameter.requires_grad for _, parameter in base_named):
        raise ELAL3BerniniIntegrationProbeError(
            "complete Bernini base must be frozen before ELAL-3 installation"
        )
    base_versions_before = {
        name: int(parameter._version) for name, parameter in base_named
    }
    hidden, rotary, text, timestep, condition_tokens, input_dtype_contract = _model_inputs(
        transformer,
        device=device,
        seed=seed,
        hidden_size=hidden_size,
    )
    total_tokens = int(hidden.shape[1])
    target_tokens = total_tokens - condition_tokens
    if (condition_tokens, target_tokens, total_tokens) != (21, 21, 42):
        raise ELAL3BerniniIntegrationProbeError(
            "probe route must be exact source21 + target21"
        )
    with torch.no_grad():
        baseline = _forward_transformer(
            transformer,
            hidden_states=hidden,
            timestep=timestep,
            text=text,
            rotary=rotary,
        ).detach().contiguous()
    if tuple(baseline.shape[:2]) != (1, total_tokens):
        raise ELAL3BerniniIntegrationProbeError(
            "baseline Bernini output token geometry differs"
        )

    handle: Optional[elal3.ELAL3C0HandleV1] = None
    try:
        handle = elal3.install_elal3_c0_v1(
            transformer,
            variant="full",
            attention_width=64,
            hidden_size=hidden_size,
            test_only=test_only,
        )
        if (
            len(handle.hooks) != 30
            or handle.native_block_ids != tuple(id(block) for block in blocks)
            or handle.variant != "full"
            or handle.attention_width != 64
        ):
            raise ELAL3BerniniIntegrationProbeError(
                "ELAL-3 full-w64 30/30 installation differs"
            )
        trainable = handle.trainable_named_parameters()
        if any(not name or not parameter.requires_grad for name, parameter in trainable):
            raise ELAL3BerniniIntegrationProbeError(
                "ELAL-3 trainable parameter inventory differs"
            )
        elal_state_before = named_tensor_digest(trainable, label="elal3-before")
        latent = deterministic_probe_latent(device, seed=seed)
        memory = handle.build_memory(latent)
        route = elal3.ELAL3RouteV1(
            total_tokens=total_tokens,
            condition_tokens=condition_tokens,
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
            memory=memory,
            route_identity=ROUTE_IDENTITY,
        )
        handle.clear_audit()
        with handle.route(route):
            adapted = _forward_transformer(
                transformer,
                hidden_states=hidden,
                timestep=timestep,
                text=text,
                rotary=rotary,
            ).contiguous()
        if adapted.shape != baseline.shape or not adapted.requires_grad:
            raise ELAL3BerniniIntegrationProbeError(
                "ELAL-3 real output lost its baseline geometry or autograd graph"
            )
        audit = tuple(dict(row) for row in handle.audit_records)
        indices = [row.get("block_index") for row in audit]
        if (
            len(audit) != 30
            or indices != list(range(30))
            or any(row.get("route_identity") != ROUTE_IDENTITY for row in audit)
            or any(row.get("source_rows") != condition_tokens for row in audit)
            or any(row.get("padding_rows") != 0 for row in audit)
            or not all(row.get("source_bit_exact") is True for row in audit)
            or not all(row.get("padding_bit_exact") is True for row in audit)
        ):
            raise ELAL3BerniniIntegrationProbeError(
                "30/30 immediate target-only hook audit failed"
            )

        target_response = _delta_metrics(
            adapted[:, condition_tokens:], baseline[:, condition_tokens:]
        )
        source_final_response = _delta_metrics(
            adapted[:, :condition_tokens], baseline[:, :condition_tokens]
        )
        if (
            target_response["nonzero_element_count"] <= 0
            or target_response["l2"] <= 0.0
            or target_response["max_abs"] <= 0.0
        ):
            raise ELAL3BerniniIntegrationProbeError(
                "real Bernini target suffix did not respond to ELAL-3"
            )
        objective = adapted[:, condition_tokens:].float().square().mean()
        if not bool(torch.isfinite(objective.detach()).item()) or not objective.requires_grad:
            raise ELAL3BerniniIntegrationProbeError(
                "probe scalar objective is non-finite or graph-free"
            )
        objective.backward()

        per_block_gradients = []
        for index, injection in enumerate(handle.components.injections):
            group = _gradient_group(
                tuple(injection.named_parameters()),
                label=f"injection-{index:02d}",
            )
            per_block_gradients.append({"block_index": index, **group})
        memory_gradients = _gradient_group(
            tuple(handle.components.memory_builder.named_parameters()),
            label="memory-builder",
        )
        if not all(
            row["all_gradients_present_finite"] and row["has_nonzero_gradient"]
            for row in per_block_gradients
        ):
            raise ELAL3BerniniIntegrationProbeError(
                "not all 30 real-block ELAL-3 injections received finite non-zero gradients"
            )
        if not (
            memory_gradients["all_gradients_present_finite"]
            and memory_gradients["has_nonzero_gradient"]
        ):
            raise ELAL3BerniniIntegrationProbeError(
                "ELAL-3 action-memory builder gradient audit failed"
            )
        if any(parameter.grad is not None for _, parameter in base_named):
            raise ELAL3BerniniIntegrationProbeError(
                "frozen Bernini base unexpectedly received gradients"
            )
        base_versions_after = {
            name: int(parameter._version) for name, parameter in base_named
        }
        if base_versions_after != base_versions_before:
            raise ELAL3BerniniIntegrationProbeError(
                "frozen Bernini parameter version counters changed"
            )
        elal_state_after = named_tensor_digest(trainable, label="elal3-before")
        if elal_state_after != elal_state_before:
            raise ELAL3BerniniIntegrationProbeError(
                "ELAL-3 parameter bytes changed despite no-update policy"
            )
        trainable_numel = sum(int(parameter.numel()) for _, parameter in trainable)
        base_numel = sum(int(parameter.numel()) for _, parameter in base_named)
        memory_at_completed_backward = _cuda_memory(device)
        result = {
            "registered_arm": REGISTERED_ARM,
            "route": {
                "identity": ROUTE_IDENTITY,
                "world_size": 1,
                "sequence_parallel_size": 1,
                "condition_tokens": condition_tokens,
                "target_tokens": target_tokens,
                "total_tokens": total_tokens,
                "packing": "contiguous-source-prefix-then-target-suffix",
            },
            "native_input": {
                "latent_shape_per_branch": list(PROBE_LATENT_SHAPE),
                "exact81_latent_phases": 21,
                "source_id": 1,
                "target_id": 0,
                "hidden_shape": [int(item) for item in hidden.shape],
                "hidden_dtype": str(hidden.dtype),
                "rotary_shape": [int(item) for item in rotary.shape],
                "rotary_dtype": str(rotary.dtype),
                "text_shape": [int(item) for item in text.shape],
                "text_dtype": str(text.dtype),
                "timestep": PROBE_TIMESTEP,
                "timestep_dtype": str(timestep.dtype),
                "input_dtype_contract": dict(input_dtype_contract),
                "hidden_sha256": tensor_sha256(hidden, label="packed hidden"),
                "rotary_sha256": tensor_sha256(rotary, label="packed rotary"),
                "text_sha256": tensor_sha256(text, label="probe text"),
                "deterministic_finite_engineering_probe_tensors": True,
                "video_dataset_consumed": False,
                "action_supervision_consumed": False,
            },
            "model_route_audit": {
                "hook_installation_count": len(handle.hooks),
                "native_distinct_block_count": len(set(handle.native_block_ids)),
                "forward_hook_record_count": len(audit),
                "ordered_block_indices": indices,
                "source_rows_per_hook": condition_tokens,
                "padding_rows_per_hook": 0,
                "source_prefix_immediate_pre_post_bit_exact_all30": True,
                "padding_immediate_pre_post_bit_exact_all30": True,
                "audit_records": list(audit),
            },
            "response": {
                "baseline_output_sha256": tensor_sha256(
                    baseline, label="baseline output"
                ),
                "adapted_output_sha256": tensor_sha256(
                    adapted, label="adapted output"
                ),
                "target_suffix": target_response,
                "final_source_prefix": source_final_response,
                "final_source_prefix_may_respond_after_later_noncausal_self_attention": True,
                "target_response_nonzero": True,
            },
            "backward": {
                "objective": "mean_square_of_real_bernini_adapted_target_prediction",
                "objective_value": float(objective.detach().item()),
                "per_block": per_block_gradients,
                "all30_injections_have_finite_nonzero_gradient": True,
                "memory_builder": memory_gradients,
                "frozen_base_gradient_tensor_count": 0,
            },
            "parameter_scope": {
                "frozen_base_parameter_tensor_count": len(base_named),
                "frozen_base_parameter_numel": base_numel,
                "elal3_parameter_tensor_count": len(trainable),
                "elal3_trainable_parameter_numel": trainable_numel,
                "only_elal3_parameters_require_grad": True,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "frozen_base_parameter_version_counters_unchanged": True,
                "elal3_parameter_sha256_before": elal_state_before,
                "elal3_parameter_sha256_after": elal_state_after,
                "elal3_parameter_bytes_unchanged": True,
            },
            "memory_at_completed_backward": memory_at_completed_backward,
            "engineering_gate_pass": True,
        }
    finally:
        if handle is not None and not handle.restored:
            handle.restore()
    return result


def checkpoint_inventory(
    checkpoint: Path, *, expected_tree_sha256: str
) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_tree_sha256) is None:
        raise ELAL3BerniniIntegrationProbeError(
            "expected checkpoint tree SHA-256 differs"
        )
    transformer_root = checkpoint / "transformer"
    config = transformer_root / "config.json"
    weights = tuple(sorted(transformer_root.glob("*.safetensors")))
    indices = tuple(sorted(transformer_root.glob("*.safetensors.index.json")))
    if not config.is_file() or (not weights and not indices):
        raise ELAL3BerniniIntegrationProbeError(
            "checkpoint transformer inventory is incomplete"
        )
    files = tuple(sorted({*weights, *indices}, key=lambda path: path.name))
    return {
        "canonical_path": str(checkpoint),
        "expected_pinned_tree_sha256": expected_tree_sha256,
        "expected_tree_identity_source": "train_lora.CHECKPOINT_TREE_SHA256",
        "content_tree_recomputed_by_this_probe": False,
        "transformer_config_sha256": file_sha256(config),
        "transformer_weight_or_index_file_count": len(files),
        "transformer_weight_or_index_total_bytes": sum(
            int(path.stat().st_size) for path in files
        ),
        "transformer_weight_or_index_inventory": [
            {"name": path.name, "bytes": int(path.stat().st_size)} for path in files
        ],
        "weights_successfully_deserialized_required_for_pass": True,
    }


def _validate_world1_environment() -> dict[str, Any]:
    values = {
        "WORLD_SIZE": os.environ.get("WORLD_SIZE", "1"),
        "RANK": os.environ.get("RANK", "0"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK", "0"),
    }
    try:
        normalized = {key: int(value) for key, value in values.items()}
    except ValueError as error:
        raise ELAL3BerniniIntegrationProbeError(
            "WORLD1 environment variables must be integers"
        ) from error
    if normalized != {"WORLD_SIZE": 1, "RANK": 0, "LOCAL_RANK": 0}:
        raise ELAL3BerniniIntegrationProbeError(
            f"probe requires WORLD1 rank0 environment, got {normalized}"
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise ELAL3BerniniIntegrationProbeError(
            "probe requires an uninitialized distributed process group"
        )
    return {
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "distributed_process_group_initialized": False,
    }


def run_real_probe(args: argparse.Namespace) -> dict[str, Any]:
    output = resolve_create_only_output(args.output)
    world = _validate_world1_environment()
    if (
        args.expected_bernini_commit != legacy.BERNINI_OFFICIAL_COMMIT
        or args.expected_veomni_commit != legacy.VEOMNI_TESTED_COMMIT
        or args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256
    ):
        raise ELAL3BerniniIntegrationProbeError(
            "runtime pins differ from the audited Bernini-R 1.3B release"
        )
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except Exception as error:
        raise ELAL3BerniniIntegrationProbeError(
            f"pinned Bernini source/checkpoint validation failed: {error}"
        ) from error
    legacy.activate_source_trees(bernini_root, veomni_root)
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise ELAL3BerniniIntegrationProbeError(
            "real probe requires an AUH ROCm CUDA-compatible device"
        )
    if not 0 <= args.device_index < torch.cuda.device_count():
        raise ELAL3BerniniIntegrationProbeError("--device-index is not visible")
    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    memory_before_load = _cuda_memory(device)
    try:
        from bernini.models.transformer_wan import WanTransformer3DModel
        from bernini.parallel import get_parallel_state
    except Exception as error:
        raise ELAL3BerniniIntegrationProbeError(
            f"cannot import pinned Bernini runtime: {error}"
        ) from error
    parallel = get_parallel_state()
    if (
        int(parallel.world_size) != 1
        or int(parallel.ulysses_size) != 1
        or parallel.ulysses_enabled
    ):
        raise ELAL3BerniniIntegrationProbeError(
            "live Bernini parallel state is not WORLD1/SP1"
        )
    try:
        transformer = WanTransformer3DModel.from_pretrained(
            str(checkpoint),
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        transformer.to(device)
    except Exception as error:
        raise ELAL3BerniniIntegrationProbeError(
            f"real Bernini-R 1.3B transformer deserialization failed: {error}"
        ) from error
    memory_after_load = _cuda_memory(device)
    if (
        transformer.__class__.__name__ != "WanTransformer3DModel"
        or int(getattr(transformer.config, "num_layers", -1)) != 30
        or int(getattr(transformer.config, "num_attention_heads", -1)) != 12
        or int(getattr(transformer.config, "attention_head_dim", -1)) != 128
        or len(tuple(getattr(transformer, "blocks", ()))) != 30
    ):
        raise ELAL3BerniniIntegrationProbeError(
            "deserialized model is not the pinned Bernini-R 1.3B transformer_1"
        )
    try:
        core = run_loaded_transformer_probe(
            transformer,
            device=device,
            seed=args.seed,
            hidden_size=elal3.BERNINI_HIDDEN,
            test_only=False,
        )
        memory_after_probe = _cuda_memory(device)
        inventory = checkpoint_inventory(
            checkpoint,
            expected_tree_sha256=args.expected_checkpoint_tree_sha256,
        )
        runner_path = Path(__file__).resolve(strict=True)
        unsigned = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "method": METHOD_NAME,
            "complete": True,
            "real_bernini_checkpoint_loaded": True,
            "synthetic_renderer_used": False,
            "registered_arm": REGISTERED_ARM,
            "seed": args.seed,
            "source_runtime": {
                "bernini_root": str(bernini_root),
                "bernini_commit": bernini_revision,
                "veomni_root": str(veomni_root),
                "veomni_commit": veomni_revision,
                "runner_path": str(runner_path),
                "runner_sha256": file_sha256(runner_path),
                "elal3_core_sha256": file_sha256(
                    Path(elal3.__file__).resolve(strict=True)
                ),
            },
            "checkpoint": {
                **inventory,
                "transformer_config": dict(transformer_config),
                "loaded_class": transformer.__class__.__name__,
                "first_model_parameter_dtype": core["native_input"][
                    "input_dtype_contract"
                ]["first_model_parameter_dtype"],
                "patch_embedding_weight_dtype": core["native_input"][
                    "input_dtype_contract"
                ]["patch_embedding_weight_dtype"],
                "text_linear_1_weight_dtype": core["native_input"][
                    "input_dtype_contract"
                ]["text_linear_1_weight_dtype"],
                "weights_successfully_deserialized": True,
            },
            "topology": {
                **world,
                "sequence_parallel_size": 1,
                "data_parallel_size": 1,
                "visible_accelerator_count": int(torch.cuda.device_count()),
                "selected_device_index": args.device_index,
                "selected_device_name": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "torch_hip": str(torch.version.hip),
            },
            "memory": {
                "before_model_load": memory_before_load,
                "after_model_load": memory_after_load,
                "after_forward_backward": memory_after_probe,
                "probe_memory_is_not_training_occupancy_evidence": True,
                "single_card_50_percent_training_occupancy_gate_evaluated": False,
            },
            "probe": core,
            "authority": {
                "training_authorized": False,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "parameters_updated": False,
                "formal_c1": False,
                "formal_exact160": False,
                "source_instruction_inference": False,
                "action_predictor_present": False,
                "video_dataset_consumed": False,
                "action_quality_claim_authorized": False,
                "scientific_promotion_authorized": False,
                "scope": "real-checkpoint renderer integration and VJP evidence only",
            },
            "engineering_gate_pass": True,
            "training_authorized": False,
        }
        receipt = seal(unsigned)
        write_create_only_json(output, receipt)
        return receipt
    finally:
        del transformer
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.CHECKPOINT_TREE_SHA256,
    )
    return parser


def exception_chain(error: BaseException) -> str:
    """Render the causal blocker chain without suppressing the native cause."""

    rows: list[str] = []
    seen: set[int] = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        rows.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " <- ".join(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run_real_probe(args)
    except Exception as error:
        print(
            "ELAL-3 real Bernini integration probe BLOCKED: "
            f"{exception_chain(error)}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        canonical_json_bytes(
            {
                "complete": receipt["complete"],
                "engineering_gate_pass": receipt["engineering_gate_pass"],
                "receipt_digest": receipt["receipt_digest"],
                "training_authorized": receipt["training_authorized"],
            }
        ).decode("ascii"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SEED",
    "ELAL3BerniniIntegrationProbeError",
    "METHOD_NAME",
    "RECEIPT_SCHEMA_VERSION",
    "REGISTERED_ARM",
    "build_parser",
    "canonical_json_bytes",
    "checkpoint_inventory",
    "deterministic_probe_latent",
    "exception_chain",
    "main",
    "named_tensor_digest",
    "object_sha256",
    "resolve_create_only_output",
    "run_loaded_transformer_probe",
    "run_real_probe",
    "seal",
    "tensor_sha256",
    "write_create_only_json",
]
