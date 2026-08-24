#!/usr/bin/env python3
"""Strict checkpoint and native-RV2V bridge for packed preservation v2.

The training adapter wraps Bernini's patch embedding for an already-packed
``[source; target]`` tensor.  Native RV2V instead patch-embeds each source and
the noisy target separately before concatenation.  :class:`NativePatchRoute`
is the only translation between those two equivalent layouts: source IDs
greater than zero receive the learned source delta/role and source ID zero
receives the learned target delta/role.  It does not change the native
scheduler, source-ID rotary embedding, guidance branches, or attention path.

This module also validates the exact80 receipt and all five create-only
``adapter.pt`` checkpoints before a decode process is allowed to load them.
It contains no optimizer, backward pass, reward, feature evaluator, ranking,
selection, or quality verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MethodType
from typing import Any, Mapping, NoReturn, Optional, Sequence

import packed_preservation_lora_v2 as core


SCHEMA_VERSION = "bernini-packed-preservation-checkpoint-review-contract-v2"
TRAINING_RECEIPT_SCHEMA = "bernini-packed-preservation-training-v2"
TRAINING_METHOD = "bernini-packed-preservation-lora-v2"
CHECKPOINT_STEPS = (0, 20, 40, 60, 80)
CANARY_CHECKPOINT_STEPS = (0, 1, 2)
WORLD_SIZE = 4
SP_SIZE = 4
FRAME_COUNT = 81
FPS = 25
NUM_INFERENCE_STEPS = 40
SOURCE_ONLY_MANIFEST_SHA256 = (
    "128064fd335c4e48c567217c6e7bae43555a904875625c9d1e21178e6f7fcc3d"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PackedPreservationReviewError(RuntimeError):
    """Raised before an ambiguous checkpoint can enter native inference."""


def fail(message: str) -> NoReturn:
    raise PackedPreservationReviewError(message)


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
        raise PackedPreservationReviewError("non-canonical receipt value") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise PackedPreservationReviewError(f"{label} is unavailable") from error
    if resolved != requested or not requested.is_file() or requested.is_symlink():
        fail(f"{label} must be one canonical plain file")
    return requested


def _plain_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink directory")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise PackedPreservationReviewError(f"{label} is unavailable") from error
    if resolved != requested or not requested.is_dir() or requested.is_symlink():
        fail(f"{label} must be one canonical plain directory")
    return requested


def _strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackedPreservationReviewError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        fail(f"{label} root must be an object")
    return value


def _embedded_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = _sha256(unsigned.pop("receipt_digest", None), label=f"{label} digest")
    if object_sha256(unsigned) != declared:
        fail(f"{label} embedded digest differs")
    return declared


@dataclass(frozen=True)
class CheckpointAuthority:
    step: int
    directory: Path
    adapter: Path
    adapter_sha256: str
    metadata: Path
    metadata_sha256: str
    parameter_sha256: str
    inventory_sha256: str

    def receipt(self) -> Mapping[str, Any]:
        return {
            "step": self.step,
            "directory": str(self.directory),
            "adapter": str(self.adapter),
            "adapter_sha256": self.adapter_sha256,
            "metadata": str(self.metadata),
            "metadata_sha256": self.metadata_sha256,
            "parameter_sha256": self.parameter_sha256,
            "inventory_sha256": self.inventory_sha256,
        }


@dataclass(frozen=True)
class TrainingAuthority:
    receipt: Path
    receipt_file_sha256: str
    receipt_digest: str
    lora_scope: str
    source_only_manifest_sha256: str
    checkpoints: tuple[CheckpointAuthority, ...]

    def checkpoint(self, step: int) -> CheckpointAuthority:
        for value in self.checkpoints:
            if value.step == step:
                return value
        fail(f"checkpoint step is not authorized: {step}")

    def as_receipt(self) -> Mapping[str, Any]:
        return {
            "training_receipt": str(self.receipt),
            "training_receipt_file_sha256": self.receipt_file_sha256,
            "training_receipt_digest": self.receipt_digest,
            "lora_scope": self.lora_scope,
            "source_only_manifest_sha256": self.source_only_manifest_sha256,
            "checkpoints": [dict(item.receipt()) for item in self.checkpoints],
        }


def load_checkpoint_authority(
    directory_value: str | Path,
    *,
    expected_step: int,
    expected_lora_scope: str,
    expected_execution_scope: str = "exact80",
    verify_files: bool = True,
) -> CheckpointAuthority:
    """Validate one atomically published checkpoint before exact80 finishes.

    The training process renames the checkpoint directory only after writing
    and fsyncing ``adapter.pt``, ``optimizer.pt`` and ``metadata.json``.  This
    permits step-20 decoding to overlap steps 21..80 without reading mutable
    training state.  The final HTML builder later rebinds every shard to the
    terminal continuous-exact80 receipt.
    """

    valid_steps = (
        CHECKPOINT_STEPS
        if expected_execution_scope == "exact80"
        else CANARY_CHECKPOINT_STEPS
        if expected_execution_scope == "optimizer-canary-2"
        else ()
    )
    if expected_step not in valid_steps:
        fail("incremental checkpoint step differs")
    if expected_lora_scope not in core.LORA_SCOPES:
        fail("incremental checkpoint LoRA scope differs")
    directory = _plain_directory(
        directory_value, label=f"checkpoint {expected_step}"
    )
    if directory.name != f"checkpoint-{expected_step:08d}":
        fail("checkpoint directory name differs")
    adapter = _plain_file(directory / "adapter.pt", label=f"adapter step {expected_step}")
    optimizer = _plain_file(
        directory / "optimizer.pt", label=f"optimizer step {expected_step}"
    )
    metadata = _plain_file(
        directory / "metadata.json", label=f"metadata step {expected_step}"
    )
    metadata_value = _strict_json(metadata, label=f"checkpoint {expected_step} metadata")
    adapter_sha = _sha256(metadata_value.get("adapter_sha256"), label="adapter SHA")
    optimizer_sha = _sha256(
        metadata_value.get("optimizer_sha256"), label="optimizer SHA"
    )
    parameter_sha = _sha256(
        metadata_value.get("parameter_sha256"), label="checkpoint parameter SHA"
    )
    inventory_sha = _sha256(
        metadata_value.get("trainable_inventory_sha256"),
        label="checkpoint trainable inventory SHA",
    )
    architecture = metadata_value.get("architecture")
    if (
        metadata_value.get("schema_version") != TRAINING_RECEIPT_SCHEMA
        or metadata_value.get("method") != TRAINING_METHOD
        or metadata_value.get("execution_scope") != expected_execution_scope
        or metadata_value.get("step") != expected_step
        or metadata_value.get("lora_scope") != expected_lora_scope
        or metadata_value.get("rank") != core.LORA_RANK
        or metadata_value.get("source_only_manifest_sha256")
        != SOURCE_ONLY_MANIFEST_SHA256
        or metadata_value.get("adapter_file") != "adapter.pt"
        or metadata_value.get("optimizer_file") != "optimizer.pt"
        or metadata_value.get("roundtrip_parameter_sha256") != parameter_sha
        or metadata_value.get("strict_loader")
        != "packed_preservation_lora_v2.load_trainable_state_strict"
        or metadata_value.get("adapter_reload_verified") is not True
        or metadata_value.get("optimizer_reload_verified") is not True
        or metadata_value.get("same_architecture_strict_reload_verified") is not True
        or metadata_value.get("fresh_official_rv2v_inference_process_verified")
        is not False
        or not isinstance(architecture, Mapping)
        or architecture.get("scope") != expected_lora_scope
        or architecture.get("rank") != core.LORA_RANK
        or architecture.get("target_row_gating") is not False
        or architecture.get("all_local_packed_tokens_receive_lora") is not True
    ):
        fail(f"checkpoint {expected_step} metadata authority differs")
    inventory = metadata_value.get("trainable_inventory")
    if not isinstance(inventory, list) or core.object_sha256(inventory) != inventory_sha:
        fail(f"checkpoint {expected_step} inventory digest differs")
    metadata_sha = file_sha256(metadata)
    if verify_files and (
        file_sha256(adapter) != adapter_sha or file_sha256(optimizer) != optimizer_sha
    ):
        fail(f"checkpoint {expected_step} artifact bytes changed")
    return CheckpointAuthority(
        step=expected_step,
        directory=directory,
        adapter=adapter,
        adapter_sha256=adapter_sha,
        metadata=metadata,
        metadata_sha256=metadata_sha,
        parameter_sha256=parameter_sha,
        inventory_sha256=inventory_sha,
    )


def load_training_authority(
    path_value: str | Path,
    *,
    expected_file_sha256: str,
    expected_lora_scope: str,
    verify_files: bool = True,
) -> TrainingAuthority:
    """Validate one completed continuous exact80 run and all five adapters."""

    if expected_lora_scope not in core.LORA_SCOPES:
        fail("requested LoRA scope differs")
    path = _plain_file(path_value, label="packed preservation training receipt")
    observed_receipt_sha = file_sha256(path)
    if observed_receipt_sha != _sha256(
        expected_file_sha256, label="training receipt expected SHA"
    ):
        fail("training receipt file SHA differs")
    value = _strict_json(path, label="packed preservation training receipt")
    receipt_digest = _embedded_digest(value, label="packed preservation training receipt")
    architecture = value.get("architecture")
    dataset = value.get("dataset")
    objective = value.get("objective")
    distributed = value.get("distributed")
    if (
        value.get("schema_version") != TRAINING_RECEIPT_SCHEMA
        or value.get("method") != TRAINING_METHOD
        or value.get("complete") is not True
        or value.get("execution_scope") != "exact80"
        or value.get("optimizer_steps") != 80
        or value.get("continuous_single_process_trajectory") is not True
        or value.get("fresh_official_base") is not True
        or value.get("resume_consumed") is not False
        or value.get("checkpoint_steps") != list(CHECKPOINT_STEPS)
        or value.get("history_steps") != 80
        or not isinstance(architecture, Mapping)
        or architecture.get("scope") != expected_lora_scope
        or architecture.get("rank") != core.LORA_RANK
        or architecture.get("target_row_gating") is not False
        or architecture.get("all_local_packed_tokens_receive_lora") is not True
        or not isinstance(dataset, Mapping)
        or dataset.get("legacy_parquet_opened") is not False
        or dataset.get("synthetic_target_index1_bytes_read") is not False
        or not isinstance(objective, Mapping)
        or objective.get("target_always_original_real_source") is not True
        or objective.get("reward") is not False
        or objective.get("vlm") is not False
        or not isinstance(distributed, Mapping)
        or distributed.get("world_size") != 8
        or distributed.get("dp_size") != 2
        or distributed.get("sp_size") != 4
        or distributed.get("lora_applied_to_all_local_packed_tokens") is not True
        or distributed.get("targetless_early_return") is not False
    ):
        fail("training exact80/model/data/objective authority differs")
    source_manifest_sha = SOURCE_ONLY_MANIFEST_SHA256

    checkpoint_rows = value.get("checkpoints")
    parameter_digests = value.get("parameter_digests")
    if (
        not isinstance(checkpoint_rows, list)
        or len(checkpoint_rows) != len(CHECKPOINT_STEPS)
        or not isinstance(parameter_digests, Mapping)
        or set(parameter_digests) != {str(step) for step in CHECKPOINT_STEPS}
    ):
        fail("checkpoint cadence authority differs")
    authorities: list[CheckpointAuthority] = []
    for row, expected_step in zip(checkpoint_rows, CHECKPOINT_STEPS):
        if not isinstance(row, Mapping) or row.get("step") != expected_step:
            fail("checkpoint record order/step differs")
        directory = _plain_directory(row.get("path"), label=f"checkpoint {expected_step}")
        if directory.name != f"checkpoint-{expected_step:08d}":
            fail("checkpoint directory name differs")
        adapter = _plain_file(directory / "adapter.pt", label=f"adapter step {expected_step}")
        optimizer = _plain_file(
            directory / "optimizer.pt", label=f"optimizer step {expected_step}"
        )
        metadata = _plain_file(
            directory / "metadata.json", label=f"metadata step {expected_step}"
        )
        adapter_sha = _sha256(row.get("adapter_sha256"), label="adapter SHA")
        optimizer_sha = _sha256(row.get("optimizer_sha256"), label="optimizer SHA")
        metadata_sha = _sha256(row.get("metadata_sha256"), label="metadata SHA")
        metadata_value = _strict_json(metadata, label=f"checkpoint {expected_step} metadata")
        parameter_sha = _sha256(
            parameter_digests[str(expected_step)], label="checkpoint parameter SHA"
        )
        inventory_sha = _sha256(
            metadata_value.get("trainable_inventory_sha256"),
            label="checkpoint trainable inventory SHA",
        )
        if (
            metadata_value.get("schema_version") != TRAINING_RECEIPT_SCHEMA
            or metadata_value.get("execution_scope") != "exact80"
            or metadata_value.get("step") != expected_step
            or metadata_value.get("lora_scope") != expected_lora_scope
            or metadata_value.get("rank") != core.LORA_RANK
            or metadata_value.get("source_only_manifest_sha256")
            != source_manifest_sha
            or metadata_value.get("adapter_file") != "adapter.pt"
            or metadata_value.get("optimizer_file") != "optimizer.pt"
            or metadata_value.get("adapter_sha256") != adapter_sha
            or metadata_value.get("optimizer_sha256") != optimizer_sha
            or metadata_value.get("parameter_sha256") != parameter_sha
            or metadata_value.get("roundtrip_parameter_sha256") != parameter_sha
            or metadata_value.get("strict_loader")
            != "packed_preservation_lora_v2.load_trainable_state_strict"
            or metadata_value.get("adapter_reload_verified") is not True
            or metadata_value.get("optimizer_reload_verified") is not True
            or metadata_value.get("same_architecture_strict_reload_verified") is not True
            or metadata_value.get("fresh_official_rv2v_inference_process_verified")
            is not False
        ):
            fail(f"checkpoint {expected_step} metadata authority differs")
        inventory = metadata_value.get("trainable_inventory")
        if not isinstance(inventory, list) or core.object_sha256(inventory) != inventory_sha:
            fail(f"checkpoint {expected_step} inventory digest differs")
        if verify_files and (
            file_sha256(adapter) != adapter_sha
            or file_sha256(optimizer) != optimizer_sha
            or file_sha256(metadata) != metadata_sha
        ):
            fail(f"checkpoint {expected_step} artifact bytes changed")
        authorities.append(
            CheckpointAuthority(
                step=expected_step,
                directory=directory,
                adapter=adapter,
                adapter_sha256=adapter_sha,
                metadata=metadata,
                metadata_sha256=metadata_sha,
                parameter_sha256=parameter_sha,
                inventory_sha256=inventory_sha,
            )
        )
    if len({item.parameter_sha256 for item in authorities}) != len(authorities):
        fail("saved checkpoint parameter states are not all distinct")
    return TrainingAuthority(
        receipt=path,
        receipt_file_sha256=observed_receipt_sha,
        receipt_digest=receipt_digest,
        lora_scope=expected_lora_scope,
        source_only_manifest_sha256=source_manifest_sha,
        checkpoints=tuple(authorities),
    )


def trainable_parameter_digest(model: Any) -> str:
    """Reproduce the exact byte digest written by the training runner."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH supplies torch
        raise PackedPreservationReviewError("parameter digest requires PyTorch") from error
    digest = hashlib.sha256()
    for name, parameter in core.trainable_named_parameters(model):
        tensor = parameter.detach().contiguous()
        metadata = core.canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def strict_load_adapter(model: Any, checkpoint: CheckpointAuthority) -> Mapping[str, Any]:
    """Load one ``adapter.pt`` with exact name/shape/inventory/digest closure."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise PackedPreservationReviewError("strict adapter loading requires PyTorch") from error
    try:
        state = torch.load(checkpoint.adapter, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PackedPreservationReviewError(
            f"cannot load checkpoint adapter at step {checkpoint.step}"
        ) from error
    core.load_trainable_state_strict(model, state)
    del state
    inventory = list(core.trainable_inventory(model))
    inventory_sha = core.object_sha256(inventory)
    if inventory_sha != checkpoint.inventory_sha256:
        fail(f"checkpoint {checkpoint.step} live trainable inventory differs")
    parameter_sha = trainable_parameter_digest(model)
    if parameter_sha != checkpoint.parameter_sha256:
        fail(f"checkpoint {checkpoint.step} strict load changed parameter bytes")
    return {
        "step": checkpoint.step,
        "adapter_sha256": checkpoint.adapter_sha256,
        "parameter_sha256": parameter_sha,
        "inventory_sha256": inventory_sha,
        "strict_name_shape_load": True,
    }


def zero_effect_adapter(model: Any) -> bool:
    """Prove step-0 adds exact zero while allowing initialized LoRA-A."""

    required = {"lora_B": 0, "source_delta": 0, "target_delta": 0, "role": 0}
    for name, parameter in core.trainable_named_parameters(model):
        if ".lora_B." in name:
            group = "lora_B"
        elif ".source_delta." in name:
            group = "source_delta"
        elif ".target_delta." in name:
            group = "target_delta"
        elif ".role_embedding" in name:
            group = "role"
        else:
            continue
        required[group] += 1
        if bool(parameter.detach().count_nonzero().item()):
            return False
    return all(count > 0 for count in required.values())


def _source_id_value(source_id: Any) -> float:
    """Normalize Bernini's scalar source ID without changing its value."""

    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    value = source_id
    if torch is not None and isinstance(value, torch.Tensor):
        if int(value.numel()) != 1:
            fail("native source_id tensor must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail("native patch route requires one numeric source_id")
    numeric = float(value)
    if not numeric >= 0.0 or numeric != numeric:
        fail("native patch route source_id differs")
    return numeric


class NativePatchRoute:
    """Adapt typed packed patch parameters to official native ``source_id`` calls."""

    def __init__(self, transformer: Any) -> None:
        typed = getattr(transformer, "patch_embedding", None)
        if typed is None or typed.__class__.__name__ != "_TypedPackedPatchEmbedding":
            fail("native patch route requires the installed typed patch embedding")
        for name in ("native", "source_delta", "target_delta", "role_embedding"):
            if not hasattr(typed, name):
                fail("typed patch embedding component inventory differs")
        original = getattr(transformer, "patch_vae_latent", None)
        if not callable(original):
            fail("official native patch_vae_latent route differs")
        self.transformer = transformer
        self.typed = typed
        self.original = original
        self.installed = False
        self.calls: list[Mapping[str, Any]] = []

    def install(self) -> None:
        current = getattr(self.transformer, "patch_vae_latent", None)
        if (
            self.installed
            or not callable(current)
            or getattr(current, "__self__", None) is not getattr(self.original, "__self__", None)
            or getattr(current, "__func__", None) is not getattr(self.original, "__func__", None)
        ):
            fail("native patch route was already installed or changed")

        route = self

        def wrapped(instance: Any, hidden_states: Any, source_id: Any = None) -> Any:
            import torch

            if instance is not route.transformer:
                fail("native patch route owner differs")
            if (
                not isinstance(hidden_states, torch.Tensor)
                or hidden_states.ndim != 5
                or int(hidden_states.shape[0]) != 1
                or int(hidden_states.shape[1]) != core.PATCH_INPUT_CHANNELS
                or not bool(torch.isfinite(hidden_states.float()).all().item())
            ):
                fail("native patch route latent geometry differs")
            source_id_value = _source_id_value(source_id)
            is_source = source_id_value > 0.0
            rotary = instance.rope(hidden_states, source_id)
            delta_module = (
                route.typed.source_delta if is_source else route.typed.target_delta
            )
            # Training executes this exact FP32 trainable patch branch under
            # CUDA BF16 autocast.  Native sample() is no-grad but does not
            # establish autocast itself, so reproduce the training compute
            # contract explicitly rather than causing a BF16/FP32 Conv3d
            # mismatch or silently changing parameter dtype.
            with torch.autocast(
                device_type=hidden_states.device.type,
                dtype=torch.bfloat16,
                enabled=hidden_states.device.type == "cuda",
            ):
                native = route.typed.native(hidden_states)
                delta = delta_module(hidden_states)
            role_index = 0 if is_source else 1
            role = route.typed.role_embedding[role_index].to(
                device=native.device, dtype=native.dtype
            ).reshape(1, core.HIDDEN_SIZE, 1, 1, 1)
            if native.shape != delta.shape or int(native.shape[1]) != core.HIDDEN_SIZE:
                fail("native typed patch output geometry differs")
            embedded = (native + delta.to(dtype=native.dtype) + role).flatten(2).transpose(1, 2)
            if embedded.ndim != 3 or int(embedded.shape[0]) != 1:
                fail("native typed patch token geometry differs")
            route.calls.append(
                {
                    "role": "source" if is_source else "target",
                    "source_id": source_id_value,
                    "tokens": int(embedded.shape[1]),
                }
            )
            return embedded, rotary

        setattr(self.transformer, "patch_vae_latent", MethodType(wrapped, self.transformer))
        self.installed = True

    def restore(self) -> None:
        if not self.installed:
            fail("native patch route is not installed")
        setattr(self.transformer, "patch_vae_latent", self.original)
        self.installed = False

    def trace(self, *, clear: bool = False) -> Mapping[str, Any]:
        rows = [dict(item) for item in self.calls]
        source_calls = sum(item["role"] == "source" for item in rows)
        target_calls = sum(item["role"] == "target" for item in rows)
        if not rows or source_calls <= 0 or target_calls <= 0:
            fail("native patch route did not observe both source and target roles")
        value = {
            "calls": len(rows),
            "source_calls": source_calls,
            "target_calls": target_calls,
            "source_tokens": sum(
                int(item["tokens"]) for item in rows if item["role"] == "source"
            ),
            "target_tokens": sum(
                int(item["tokens"]) for item in rows if item["role"] == "target"
            ),
            "rows_sha256": object_sha256(rows),
            "source_id_zero_is_target": True,
            "source_id_positive_is_source": True,
            "native_rotary_unchanged": True,
        }
        if clear:
            self.calls.clear()
        return value


__all__ = [
    "CANARY_CHECKPOINT_STEPS",
    "CHECKPOINT_STEPS",
    "CheckpointAuthority",
    "FRAME_COUNT",
    "FPS",
    "NUM_INFERENCE_STEPS",
    "NativePatchRoute",
    "PackedPreservationReviewError",
    "SCHEMA_VERSION",
    "SP_SIZE",
    "TrainingAuthority",
    "WORLD_SIZE",
    "canonical_json_bytes",
    "file_sha256",
    "load_training_authority",
    "object_sha256",
    "strict_load_adapter",
    "trainable_parameter_digest",
    "zero_effect_adapter",
]
