#!/usr/bin/env python3
"""Run the synthetic ELAL-3 all-30-block C0 gate and write one JSON receipt.

This CLI intentionally uses a small deterministic token-wise renderer harness.
It validates the representation, routing, intervention, and gradient mechanism
without loading Bernini weights.  A passing receipt is therefore engineering
evidence only and never authorizes R1/R2, training, or scientific promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

import elal3_c0_v1 as elal3


RECEIPT_SCHEMA_VERSION = "bernini-elal3-synthetic-c0-receipt-v1"
PAIRED_INITIALIZATION_SCHEMA_VERSION = "bernini-elal3-c0-paired-initialization-v1"
DEFAULT_SEED = 20260817
INTERVENTION_TOLERANCE = 1.0e-7
SP_EQUIVALENCE_ATOL = 5.0e-5


class ELAL3C0RunnerError(RuntimeError):
    pass


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
        raise ELAL3C0RunnerError("receipt is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    if "receipt_digest" in unsigned:
        raise ELAL3C0RunnerError("unsigned receipt already contains digest")
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def write_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise ELAL3C0RunnerError("--output must be an absolute file in an existing directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o640)
    try:
        payload = canonical_json_bytes(value) + b"\n"
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class SyntheticRendererBlock(nn.Module):
    """Cheap token-wise trainable block used only to test 30-layer VJPs."""

    def __init__(self, block_index: int) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.01 + block_index * 1.0e-4))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.gain.to(hidden.dtype) * torch.tanh(hidden)


class SyntheticThirtyBlockRenderer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList(
            SyntheticRendererBlock(index) for index in range(elal3.BERNINI_BLOCKS)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


def _namespace_seed(seed: int, namespace: str) -> int:
    if type(seed) is not int or type(namespace) is not str or not namespace:
        raise ELAL3C0RunnerError("paired generator seed/namespace differs")
    payload = (
        f"{PAIRED_INITIALIZATION_SCHEMA_VERSION}|seed={seed}|namespace={namespace}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _paired_cpu_randn(
    shape: Sequence[int], *, seed: int, namespace: str, std: float = 1.0
) -> torch.Tensor:
    normalized_shape = tuple(int(value) for value in shape)
    if (
        any(value < 0 for value in normalized_shape)
        or not math.isfinite(float(std))
        or float(std) <= 0.0
    ):
        raise ELAL3C0RunnerError("paired random tensor shape/std differs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_namespace_seed(seed, namespace))
    return (
        torch.randn(normalized_shape, generator=generator, device="cpu", dtype=torch.float32)
        .mul_(float(std))
        .contiguous()
    )


def _leaf_randn(
    shape: Sequence[int], *, device: torch.device, seed: Optional[int] = None, namespace: str = ""
) -> torch.Tensor:
    if seed is None:
        value = torch.randn(tuple(shape), device=device, dtype=torch.float32)
    else:
        value = _paired_cpu_randn(shape, seed=seed, namespace=namespace).to(device=device)
    return value.contiguous().requires_grad_()


def synthetic_latent(
    device: torch.device, *, seed: Optional[int] = None
) -> elal3.ELAL3LatentV1:
    batch = 1
    height = width = 1
    presence = torch.ones((batch, 3), dtype=torch.bool, device=device)
    temporal = torch.ones((batch, 3, 21), dtype=torch.bool, device=device)
    relation = torch.ones((batch, 6, 21), dtype=torch.bool, device=device)
    phase = torch.ones((batch, 21), dtype=torch.bool, device=device)
    value = elal3.ELAL3LatentV1(
        q_local=_leaf_randn(
            (batch, 21, height, width, 64),
            device=device,
            seed=seed,
            namespace="synthetic_input.q_local",
        ),
        q_entity=_leaf_randn(
            (batch, 3, 21, 256),
            device=device,
            seed=seed,
            namespace="synthetic_input.q_entity",
        ),
        q_relation=_leaf_randn(
            (batch, 6, 21, 128),
            device=device,
            seed=seed,
            namespace="synthetic_input.q_relation",
        ),
        q_phase=_leaf_randn(
            (batch, 21, 128),
            device=device,
            seed=seed,
            namespace="synthetic_input.q_phase",
        ),
        q_terminal=_leaf_randn(
            (batch, 9, 256),
            device=device,
            seed=seed,
            namespace="synthetic_input.q_terminal",
        ),
        q_camera=_leaf_randn(
            (batch, 21, 128),
            device=device,
            seed=seed,
            namespace="synthetic_input.q_camera",
        ),
        entity_presence=presence.contiguous(),
        temporal_valid=temporal.contiguous(),
        relation_valid=relation.contiguous(),
        phase_valid=phase.contiguous(),
    )
    value.validate()
    return value


def _target_tail(value: torch.Tensor, condition_tokens: int, target_tokens: int) -> torch.Tensor:
    return value[:, condition_tokens : condition_tokens + target_tokens]


def _run_sp(
    *,
    model: SyntheticThirtyBlockRenderer,
    handle: elal3.ELAL3C0HandleV1,
    latent: elal3.ELAL3LatentV1,
    global_hidden: torch.Tensor,
    condition_tokens: int,
    sp_size: int,
    route_label: str,
) -> tuple[torch.Tensor, tuple[Mapping[str, Any], ...]]:
    memory = handle.build_memory(latent)
    total_tokens = int(global_hidden.shape[1])
    local_length = math.ceil(total_tokens / sp_size)
    padded_tokens = local_length * sp_size
    if padded_tokens > total_tokens:
        hidden = torch.cat(
            (
                global_hidden,
                torch.zeros(
                    (int(global_hidden.shape[0]), padded_tokens - total_tokens, int(global_hidden.shape[2])),
                    dtype=global_hidden.dtype,
                    device=global_hidden.device,
                ),
            ),
            dim=1,
        )
    else:
        hidden = global_hidden
    handle.clear_audit()
    outputs = []
    for rank in range(sp_size):
        local = hidden[:, rank * local_length : (rank + 1) * local_length].contiguous()
        route = elal3.ELAL3RouteV1(
            total_tokens=total_tokens,
            condition_tokens=condition_tokens,
            sequence_parallel_rank=rank,
            sequence_parallel_size=sp_size,
            memory=memory,
            route_identity=f"{route_label}:sp{sp_size}:rank{rank}",
        )
        with handle.route(route):
            outputs.append(model(local))
    joined = torch.cat(outputs, dim=1)
    return joined[:, :total_tokens].contiguous(), tuple(handle.audit_records)


def _tensor_rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().item())


def _tensor_receipt(value: torch.Tensor) -> dict[str, Any]:
    raw_values = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    chunk_size = 1 << 20
    for start in range(0, int(raw_values.numel()), chunk_size):
        digest.update(bytes(raw_values[start : start + chunk_size].tolist()))
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "sha256": digest.hexdigest(),
    }


def _xavier_normal_std(fan_in: int, fan_out: int) -> float:
    return math.sqrt(2.0 / float(fan_in + fan_out))


def _paired_master_specs(hidden_size: int) -> tuple[dict[str, Any], ...]:
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ELAL3C0RunnerError("paired master hidden size differs")
    specs: list[dict[str, Any]] = []

    def normal(namespace: str, shape: Sequence[int], std: float) -> None:
        specs.append(
            {
                "namespace": namespace,
                "shape": [int(value) for value in shape],
                "initializer": {
                    "kind": "namespace_sha256_cpu_normal",
                    "mean": 0.0,
                    "std": float(std),
                },
            }
        )

    def constant(namespace: str, shape: Sequence[int], value: float) -> None:
        specs.append(
            {
                "namespace": namespace,
                "shape": [int(item) for item in shape],
                "initializer": {"kind": "constant", "value": float(value)},
            }
        )

    for block_index in range(elal3.BERNINI_BLOCKS):
        constant(
            f"renderer.blocks.{block_index}.gain",
            (),
            0.01 + block_index * 1.0e-4,
        )
    normal(
        "action.memory_builder.entity_slot",
        (elal3.ENTITY_SLOTS, elal3.MEMORY_WIDTH),
        elal3.MEMORY_WIDTH ** -0.5,
    )
    normal(
        "action.memory_builder.entity_time",
        (elal3.LATENT_PHASES, elal3.MEMORY_WIDTH),
        elal3.MEMORY_WIDTH ** -0.5,
    )
    normal(
        "action.memory_builder.relation_edge",
        (elal3.RELATION_SLOTS, elal3.MEMORY_WIDTH),
        elal3.MEMORY_WIDTH ** -0.5,
    )
    normal(
        "action.memory_builder.relation_time",
        (elal3.LATENT_PHASES, elal3.MEMORY_WIDTH),
        elal3.MEMORY_WIDTH ** -0.5,
    )
    normal(
        "action.memory_builder.phase_time",
        (elal3.LATENT_PHASES, elal3.MEMORY_WIDTH),
        elal3.MEMORY_WIDTH ** -0.5,
    )
    normal(
        "action.memory_builder.entity_projection.weight",
        (elal3.MEMORY_WIDTH, elal3.ENTITY_WIDTH),
        _xavier_normal_std(elal3.ENTITY_WIDTH, elal3.MEMORY_WIDTH),
    )
    normal(
        "action.memory_builder.relation_projection.weight",
        (elal3.MEMORY_WIDTH, elal3.RELATION_WIDTH),
        _xavier_normal_std(elal3.RELATION_WIDTH, elal3.MEMORY_WIDTH),
    )
    normal(
        "action.memory_builder.phase_projection.weight",
        (elal3.MEMORY_WIDTH, elal3.PHASE_WIDTH),
        _xavier_normal_std(elal3.PHASE_WIDTH, elal3.MEMORY_WIDTH),
    )
    max_attention_width = 128
    for block_index in range(elal3.BERNINI_BLOCKS):
        prefix = f"action.injections.{block_index}"
        constant(f"{prefix}.residual_gain", (), 1.0)
        normal(
            f"{prefix}.query.weight",
            (max_attention_width, hidden_size),
            _xavier_normal_std(hidden_size, max_attention_width),
        )
        for projection_name in ("key", "value"):
            normal(
                f"{prefix}.{projection_name}.weight",
                (max_attention_width, elal3.MEMORY_WIDTH),
                _xavier_normal_std(elal3.MEMORY_WIDTH, max_attention_width),
            )
        normal(
            f"{prefix}.output.weight",
            (hidden_size, max_attention_width),
            _xavier_normal_std(max_attention_width, hidden_size),
        )
        normal(
            f"{prefix}.local_projection.weight",
            (hidden_size, elal3.LOCAL_WIDTH),
            _xavier_normal_std(elal3.LOCAL_WIDTH, hidden_size),
        )
    normal(
        "frozen_output_encoder.weight",
        (16, hidden_size),
        _xavier_normal_std(hidden_size, 16),
    )
    namespaces = [str(spec["namespace"]) for spec in specs]
    if len(namespaces) != len(set(namespaces)):
        raise ELAL3C0RunnerError("paired master namespaces are not unique")
    return tuple(sorted(specs, key=lambda spec: str(spec["namespace"])))


def _materialize_paired_master_plan(
    *, seed: int, hidden_size: int
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    masters: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for spec in _paired_master_specs(hidden_size):
        namespace = str(spec["namespace"])
        shape = tuple(int(value) for value in spec["shape"])
        initializer = dict(spec["initializer"])
        if initializer["kind"] == "constant":
            tensor = torch.full(
                shape,
                float(initializer["value"]),
                dtype=torch.float32,
                device="cpu",
            ).contiguous()
        elif initializer["kind"] == "namespace_sha256_cpu_normal":
            initializer["namespace_seed"] = _namespace_seed(seed, namespace)
            tensor = _paired_cpu_randn(
                shape,
                seed=seed,
                namespace=namespace,
                std=float(initializer["std"]),
            )
        else:
            raise ELAL3C0RunnerError("paired master initializer differs")
        masters[namespace] = tensor
        tensor_row = _tensor_receipt(tensor)
        rows.append(
            {
                "namespace": namespace,
                "shape": tensor_row["shape"],
                "dtype": tensor_row["dtype"],
                "initializer": initializer,
                "sha256": tensor_row["sha256"],
            }
        )
    return masters, rows


def _active_master_slice(
    master: torch.Tensor, *, parameter_name: str, attention_width: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    if parameter_name.endswith((".query.weight", ".key.weight", ".value.weight")):
        return (
            master[:attention_width, :].contiguous(),
            {"kind": "prefix_rows", "start": 0, "stop": attention_width},
        )
    if parameter_name.endswith(".output.weight"):
        return (
            master[:, :attention_width].contiguous(),
            {"kind": "prefix_columns", "start": 0, "stop": attention_width},
        )
    return master, {"kind": "all"}


def _apply_paired_initialization(
    *,
    model: SyntheticThirtyBlockRenderer,
    handle: elal3.ELAL3C0HandleV1,
    frozen_output_encoder: nn.Linear,
    seed: int,
) -> dict[str, Any]:
    """Overwrite every compared parameter from one arm-independent master plan."""

    if frozen_output_encoder.bias is not None or frozen_output_encoder.out_features != 16:
        raise ELAL3C0RunnerError("frozen output encoder ABI differs")
    masters, master_rows = _materialize_paired_master_plan(
        seed=seed, hidden_size=handle.hidden_size
    )
    master_by_namespace = {str(row["namespace"]): row for row in master_rows}
    active_rows: list[dict[str, Any]] = []

    def load(
        *,
        component: str,
        parameter_name: str,
        parameter: nn.Parameter,
        namespace: str,
        active_slice: Optional[tuple[torch.Tensor, dict[str, Any]]] = None,
    ) -> None:
        if namespace not in masters:
            raise ELAL3C0RunnerError(f"paired master missing: {namespace}")
        master = masters[namespace]
        if active_slice is None:
            active, slice_receipt = master, {"kind": "all"}
        else:
            active, slice_receipt = active_slice
        if tuple(parameter.shape) != tuple(active.shape) or parameter.dtype != torch.float32:
            raise ELAL3C0RunnerError(f"paired active parameter ABI differs: {parameter_name}")
        with torch.no_grad():
            parameter.copy_(active.to(device=parameter.device, dtype=parameter.dtype))
        loaded = _tensor_receipt(parameter)
        master_row = master_by_namespace[namespace]
        active_rows.append(
            {
                "component": component,
                "parameter": parameter_name,
                "master_namespace": namespace,
                "master_sha256": master_row["sha256"],
                "master_shape": master_row["shape"],
                "active_slice": slice_receipt,
                "active_shape": loaded["shape"],
                "active_dtype": loaded["dtype"],
                "active_sha256": loaded["sha256"],
                "requires_grad": bool(parameter.requires_grad),
            }
        )

    for block_index, block in enumerate(model.blocks):
        block_parameters = dict(block.named_parameters())
        if set(block_parameters) != {"gain"}:
            raise ELAL3C0RunnerError("synthetic renderer parameter ABI differs")
        load(
            component="renderer",
            parameter_name=f"blocks.{block_index}.gain",
            parameter=block_parameters["gain"],
            namespace=f"renderer.blocks.{block_index}.gain",
        )

    action_parameters = dict(handle.trainable_named_parameters())
    expected_action_names = {
        "memory_builder.entity_slot",
        "memory_builder.entity_time",
        "memory_builder.phase_time",
        "memory_builder.entity_projection.weight",
        "memory_builder.phase_projection.weight",
    }
    if handle.variant == "full":
        expected_action_names.update(
            {
                "memory_builder.relation_edge",
                "memory_builder.relation_time",
                "memory_builder.relation_projection.weight",
            }
        )
    for block_index in range(elal3.BERNINI_BLOCKS):
        expected_action_names.update(
            {
                f"injections.{block_index}.residual_gain",
                f"injections.{block_index}.query.weight",
                f"injections.{block_index}.key.weight",
                f"injections.{block_index}.value.weight",
                f"injections.{block_index}.output.weight",
                f"injections.{block_index}.local_projection.weight",
            }
        )
    if set(action_parameters) != expected_action_names:
        raise ELAL3C0RunnerError("active action parameter set is not closed")
    for parameter_name in sorted(action_parameters):
        namespace = f"action.{parameter_name}"
        active_slice = _active_master_slice(
            masters[namespace],
            parameter_name=parameter_name,
            attention_width=handle.attention_width,
        )
        load(
            component="action",
            parameter_name=parameter_name,
            parameter=action_parameters[parameter_name],
            namespace=namespace,
            active_slice=active_slice,
        )

    frozen_output_encoder.requires_grad_(False)
    load(
        component="frozen_supervision",
        parameter_name="weight",
        parameter=frozen_output_encoder.weight,
        namespace="frozen_output_encoder.weight",
    )
    frozen_output_encoder.requires_grad_(False)
    active_rows.sort(key=lambda row: (str(row["component"]), str(row["parameter"])))
    frozen_tensor = _tensor_receipt(frozen_output_encoder.weight)
    frozen_receipt = {
        "namespace": "frozen_output_encoder.weight",
        "master_sha256": master_by_namespace["frozen_output_encoder.weight"]["sha256"],
        "shape": frozen_tensor["shape"],
        "dtype": frozen_tensor["dtype"],
        "sha256": frozen_tensor["sha256"],
        "requires_grad": bool(frozen_output_encoder.weight.requires_grad),
    }
    return {
        "paired_initialization_schema": PAIRED_INITIALIZATION_SCHEMA_VERSION,
        "paired_master_plan_rows": master_rows,
        "paired_master_plan_row_count": len(master_rows),
        "paired_master_plan_digest": object_sha256(master_rows),
        "paired_active_parameter_mapping": active_rows,
        "paired_active_parameter_row_count": len(active_rows),
        "paired_active_parameter_mapping_digest": object_sha256(active_rows),
        "frozen_output_encoder_receipt": frozen_receipt,
    }


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> tuple[float, bool, bool]:
    squared = 0.0
    finite = True
    present = False
    for parameter in parameters:
        if parameter.grad is None:
            finite = False
            continue
        present = True
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all().item()):
            finite = False
        squared += float(gradient.square().sum().item())
    norm = math.sqrt(squared)
    return norm, finite, present


def _action_parameter_audit(handle: elal3.ELAL3C0HandleV1) -> tuple[dict[str, float], bool]:
    rows: dict[str, float] = {}
    all_good = True
    for name, parameter in handle.trainable_named_parameters():
        if parameter.grad is None:
            rows[name] = 0.0
            all_good = False
            continue
        gradient = parameter.grad.detach().float()
        norm = float(gradient.norm().item())
        rows[name] = norm
        if not math.isfinite(norm) or norm <= 0.0:
            all_good = False
    return rows, all_good


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ELAL3C0RunnerError("--device cuda requested but torch.cuda is unavailable")
    if value not in ("cpu", "cuda"):
        raise ELAL3C0RunnerError("--device must be auto, cpu, or cuda")
    return torch.device(value)


def _checkpoint_route_gate(
    memory: elal3.ELAL3ActionMemoryV1,
) -> bool:
    route = elal3.ELAL3RouteV1(
        total_tokens=42,
        condition_tokens=21,
        sequence_parallel_rank=0,
        sequence_parallel_size=1,
        memory=memory,
        route_identity="checkpoint-replay-probe",
    )
    with elal3.activate_elal3_route_v1(route):
        forward_context, recompute_context = elal3.elal3_checkpoint_context_fn_v1()
    with forward_context:
        first = elal3.active_elal3_route_v1() is route
    with recompute_context:
        second = elal3.active_elal3_route_v1() is route
    return bool(first and second and elal3.active_elal3_route_v1() is None)


def run_gate(*, variant: str, attention_width: int, device: torch.device, seed: int) -> dict[str, Any]:
    if (variant, attention_width) not in {
        ("no_relation", 64), ("full", 64), ("full", 128)
    }:
        raise ELAL3C0RunnerError("allowed arms are no_relation-w64, full-w64, full-w128")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        starting_allocated = int(torch.cuda.memory_allocated(device))
    else:
        starting_allocated = 0
    start = time.perf_counter()
    hidden_size = elal3.BERNINI_HIDDEN if device.type == "cuda" else 32
    # Generate every compared input before any variant/width-dependent module
    # initialization consumes RNG.  Separate arm processes with the same seed
    # must therefore publish identical input digests.
    latent = synthetic_latent(device, seed=seed)
    target_tokens = latent.local_token_count
    condition_tokens = target_tokens
    total_tokens = condition_tokens + target_tokens
    hidden = _paired_cpu_randn(
        (1, total_tokens, hidden_size),
        seed=seed,
        namespace="synthetic_input.hidden",
    ).to(device=device).contiguous()
    input_rows = {
        name: _tensor_receipt(getattr(latent, name))
        for name in (
            "q_local",
            "q_entity",
            "q_relation",
            "q_phase",
            "q_terminal",
            "q_camera",
            "entity_presence",
            "temporal_valid",
            "relation_valid",
            "phase_valid",
        )
    }
    input_rows["hidden"] = _tensor_receipt(hidden)
    synthetic_input_digest = object_sha256(input_rows)
    model = SyntheticThirtyBlockRenderer(hidden_size).to(device)
    handle = elal3.install_elal3_c0_v1(
        model,
        variant=variant,
        attention_width=attention_width,
        hidden_size=hidden_size,
        test_only=True,
    )
    frozen_output_encoder = nn.Linear(hidden_size, 16, bias=False, device=device)
    paired_initialization = _apply_paired_initialization(
        model=model,
        handle=handle,
        frozen_output_encoder=frozen_output_encoder,
        seed=seed,
    )

    intervention_names = ["correct", "zero", "phase_reverse", "role_slot_swap", "relation_zero"]
    outputs: dict[str, torch.Tensor] = {}
    intervention_audits: dict[str, tuple[Mapping[str, Any], ...]] = {}
    with torch.no_grad():
        for name in intervention_names:
            candidate = elal3.intervene_elal3_v1(latent, name)
            output, audits = _run_sp(
                model=model,
                handle=handle,
                latent=candidate,
                global_hidden=hidden,
                condition_tokens=condition_tokens,
                sp_size=1,
                route_label=name,
            )
            outputs[name] = output
            intervention_audits[name] = audits
        sp4_output, sp4_audits = _run_sp(
            model=model,
            handle=handle,
            latent=latent,
            global_hidden=hidden,
            condition_tokens=condition_tokens,
            sp_size=4,
            route_label="correct",
        )

    correct_target = _target_tail(outputs["correct"], condition_tokens, target_tokens)
    intervention_deltas = {
        name: _tensor_rms(
            _target_tail(outputs[name], condition_tokens, target_tokens) - correct_target
        )
        for name in intervention_names
        if name != "correct"
    }
    sp_difference = float((sp4_output - outputs["correct"]).detach().float().abs().max().item())
    all_audits = [row for rows in intervention_audits.values() for row in rows] + list(sp4_audits)
    source_bit_exact = bool(all(row["source_bit_exact"] for row in all_audits))
    padding_bit_exact = bool(all(row["padding_bit_exact"] for row in all_audits))
    padding_rows_observed = sum(int(row["padding_rows"]) for row in sp4_audits)

    model.zero_grad(set_to_none=True)
    for tensor in (
        latent.q_local,
        latent.q_entity,
        latent.q_relation,
        latent.q_phase,
        latent.q_terminal,
        latent.q_camera,
    ):
        tensor.grad = None
    gradient_output, gradient_audits = _run_sp(
        model=model,
        handle=handle,
        latent=latent,
        global_hidden=hidden,
        condition_tokens=condition_tokens,
        sp_size=1,
        route_label="gradient",
    )
    target_hidden = _target_tail(gradient_output, condition_tokens, target_tokens)
    encoded = frozen_output_encoder(target_hidden)
    frozen_target = torch.linspace(-0.5, 0.5, target_tokens * 16, device=device).reshape(1, target_tokens, 16)
    output_action_loss = (encoded - frozen_target).float().square().mean()
    output_action_loss.backward()

    renderer_block_grad_norms = []
    renderer_block_grad_good = True
    action_block_grad_norms = []
    for index, block in enumerate(model.blocks):
        norm, finite, present = _gradient_norm(tuple(block.parameters()))
        renderer_block_grad_norms.append({"block_index": index, "grad_norm": norm})
        renderer_block_grad_good &= bool(finite and present and norm > 0.0)
        injection = handle.components.injections[index]
        action_norm, action_finite, action_present = _gradient_norm(tuple(injection.parameters()))
        action_block_grad_norms.append({"block_index": index, "grad_norm": action_norm})
        renderer_block_grad_good &= bool(action_finite and action_present and action_norm > 0.0)
    action_parameter_grad_norms, action_parameters_good = _action_parameter_audit(handle)
    latent_grad_norms: dict[str, Optional[float]] = {}
    for name in ("q_local", "q_entity", "q_relation", "q_phase", "q_terminal", "q_camera"):
        gradient = getattr(latent, name).grad
        latent_grad_norms[name] = None if gradient is None else float(gradient.detach().float().norm().item())
    required_latent_fields = ("q_local", "q_entity", "q_phase") + (("q_relation",) if variant == "full" else ())
    latent_gradient_good = all(
        latent_grad_norms[name] is not None
        and math.isfinite(float(latent_grad_norms[name]))
        and float(latent_grad_norms[name]) > 0.0
        for name in required_latent_fields
    )
    frozen_encoder_no_parameter_grad = all(
        parameter.grad is None for parameter in frozen_output_encoder.parameters()
    )
    checkpoint_route_replay = _checkpoint_route_gate(handle.build_memory(latent))

    gates = {
        "source_rows_bit_exact": source_bit_exact,
        "padding_rows_bit_exact": padding_bit_exact and padding_rows_observed > 0,
        "all_30_blocks_hooked_per_forward": all(
            len(rows) == elal3.BERNINI_BLOCKS for rows in intervention_audits.values()
        ) and len(gradient_audits) == elal3.BERNINI_BLOCKS,
        "sp4_matches_sp1": math.isfinite(sp_difference) and sp_difference <= SP_EQUIVALENCE_ATOL,
        "zero_intervention_non_equivalent": intervention_deltas["zero"] > INTERVENTION_TOLERANCE,
        "phase_reverse_non_equivalent": intervention_deltas["phase_reverse"] > INTERVENTION_TOLERANCE,
        "role_slot_swap_non_equivalent": intervention_deltas["role_slot_swap"] > INTERVENTION_TOLERANCE,
        "relation_zero_non_equivalent": (
            intervention_deltas["relation_zero"] > INTERVENTION_TOLERANCE
            if variant == "full"
            else intervention_deltas["relation_zero"] <= SP_EQUIVALENCE_ATOL
        ),
        "renderer_and_action_all_30_gradients_finite_nonzero": renderer_block_grad_good,
        "all_active_action_parameters_gradients_finite_nonzero": action_parameters_good,
        "required_action_latent_inputs_gradients_finite_nonzero": latent_gradient_good,
        "q_camera_nuisance_is_not_injected_into_action_loss": latent_grad_norms["q_camera"] is None,
        "frozen_output_encoder_parameters_have_no_gradient": frozen_encoder_no_parameter_grad,
        "checkpoint_route_identity_replays": checkpoint_route_replay,
        "output_action_loss_finite": bool(torch.isfinite(output_action_loss.detach()).item()),
    }
    engineering_gate_pass = bool(all(gates.values()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        ending_allocated = int(torch.cuda.memory_allocated(device))
    else:
        peak_allocated = None
        ending_allocated = None
    elapsed = time.perf_counter() - start

    action_parameters = sum(parameter.numel() for _, parameter in handle.trainable_named_parameters())
    renderer_parameters = sum(parameter.numel() for block in model.blocks for parameter in block.parameters())
    local_length_sp4 = math.ceil(total_tokens / 4)
    rough_saved_activation_elements = elal3.BERNINI_BLOCKS * (
        total_tokens * hidden_size
        + target_tokens * attention_width
        + 2 * elal3.MEMORY_TOKENS * attention_width
        + 8 * target_tokens * elal3.MEMORY_TOKENS
    )
    handle.restore()
    return seal(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "module_schema_version": elal3.SCHEMA_VERSION,
            "status": (
                "SYNTHETIC_C0_GO" if engineering_gate_pass and variant == "full"
                else "SYNTHETIC_ABLATION_GO" if engineering_gate_pass
                else "SYNTHETIC_C0_FAIL"
            ),
            "variant": variant,
            "attention_width": attention_width,
            "registered_arm": f"{variant}-w{attention_width}",
            "seed": seed,
            "device": str(device),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_hip_version": getattr(torch.version, "hip", None),
            "platform": platform.platform(),
            "synthetic_harness": True,
            "synthetic_inputs_generated_before_arm_initialization": True,
            "synthetic_input_rows": input_rows,
            "synthetic_input_digest": synthetic_input_digest,
            **paired_initialization,
            "real_bernini_checkpoint_loaded": False,
            "representation_tokenizer_loaded_or_qualified": False,
            "training_authorized": False,
            "scientific_promotion_authorized": False,
            "engineering_gate_pass": engineering_gate_pass,
            "synthetic_full_structure_gate_pass": bool(
                engineering_gate_pass and variant == "full"
            ),
            "complete_elal3_c0": False,
            "production_elal3_c0_authority": False,
            "gates": gates,
            "thresholds": {
                "intervention_rms_min_exclusive": INTERVENTION_TOLERANCE,
                "sp1_sp4_max_abs_atol": SP_EQUIVALENCE_ATOL,
            },
            "geometry": {
                "batch": 1,
                "latent_grid": [21, 1, 1],
                "source_tokens": condition_tokens,
                "target_tokens": target_tokens,
                "total_tokens": total_tokens,
                "sp4_local_length": local_length_sp4,
                "sp4_append_padding_tokens": local_length_sp4 * 4 - total_tokens,
                "action_memory_capacity_tokens": elal3.MEMORY_TOKENS,
                "effective_relation_tokens": 126 if variant == "full" else 0,
                "action_latent_fields": ["q_local", "q_entity", "q_relation", "q_phase", "q_terminal"],
                "nuisance_latent_fields": ["q_camera"],
                "hidden_size": hidden_size,
                "all_blocks": list(range(elal3.BERNINI_BLOCKS)),
            },
            "intervention_target_rms_deltas_from_correct": intervention_deltas,
            "sp1_sp4_max_abs_difference": sp_difference,
            "padding_rows_observed_across_block_hooks": padding_rows_observed,
            "output_action_loss": float(output_action_loss.detach().item()),
            "renderer_block_grad_norms": renderer_block_grad_norms,
            "action_injection_block_grad_norms": action_block_grad_norms,
            "action_parameter_grad_norms": action_parameter_grad_norms,
            "action_latent_input_grad_norms": latent_grad_norms,
            "parameter_estimate": {
                "action_parameter_elements": action_parameters,
                "action_parameter_bytes_fp32": action_parameters * 4,
                "synthetic_renderer_parameter_elements": renderer_parameters,
            },
            "activation_estimate": {
                "kind": "analytical_rough_saved-forward-elements_not_memory_authority",
                "elements": rough_saved_activation_elements,
                "bytes_if_fp32": rough_saved_activation_elements * 4,
            },
            "elapsed_seconds": elapsed,
            "cuda_memory": {
                "starting_allocated_bytes": starting_allocated if device.type == "cuda" else None,
                "ending_allocated_bytes": ending_allocated,
                "peak_allocated_bytes": peak_allocated,
            },
        }
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--variant", required=True, choices=("no_relation", "full"))
    result.add_argument("--attention-width", required=True, type=int, choices=(64, 128))
    result.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    result.add_argument("--output", required=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        print("ELAL-3 C0 output already exists", file=sys.stderr)
        return 3
    try:
        device = _device_from_arg(args.device)
        receipt = run_gate(
            variant=args.variant,
            attention_width=args.attention_width,
            device=device,
            seed=args.seed,
        )
    except Exception as error:
        failure = seal(
            {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "status": "SYNTHETIC_C0_ERROR",
                "variant": args.variant,
                "attention_width": args.attention_width,
                "registered_arm": f"{args.variant}-w{args.attention_width}",
                "seed": args.seed,
                "requested_device": args.device,
                "synthetic_harness": True,
                "engineering_gate_pass": False,
                "synthetic_full_structure_gate_pass": False,
                "complete_elal3_c0": False,
                "production_elal3_c0_authority": False,
                "training_authorized": False,
                "scientific_promotion_authorized": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        try:
            write_create_only_json(output, failure)
        except Exception as write_error:
            print(f"ELAL-3 C0 failed and receipt write failed: {write_error}", file=sys.stderr)
            return 4
        print(f"ELAL-3 C0 error receipt: {output}", file=sys.stderr)
        return 2
    write_create_only_json(output, receipt)
    print(str(output))
    return 0 if receipt["engineering_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
