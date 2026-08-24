#!/usr/bin/env python3
"""Strictly verify a one-step official-checkpoint PACT GPU smoke run.

The verifier first checks the immutable run artifacts and all recorded file
digests.  Only after those inexpensive checks pass does it reconstruct the
official OmniVideo2-1.3B base model and restore the final adapter with
``load_pact_adapter_bundle``.  It never modifies the run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import math
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pact.checkpoint import LoadedPactAdapters, load_pact_adapter_bundle  # noqa: E402
from pact.lora import expected_lora_module_count  # noqa: E402
from pact.training import (  # noqa: E402
    DiffSynthWanTrainingScheduler,
    validate_training_config,
)


SUMMARY_FORMAT = "pact-omnivideo2-gpu-smoke-verification-v2"
RUN_FORMAT = "pact-omnivideo2-run-v2"
DONE_FORMAT = "pact-omnivideo2-training-done-v2"
ADAPTER_FORMAT = "pact-omnivideo2-adapters-v2"
DIFFSYNTH_REFERENCE_REVISION = "ab12bf4119b7c9a23ff3359eefb41ba54a658ccb"

RUN_FIELDS = {
    "format",
    "config",
    "config_sha256",
    "manifest",
    "manifest_sha256",
    "payload_root",
    "checkpoint_dir",
    "checkpoint_sha256",
    "special_tokens_sha256",
    "encoder_contract_sha256",
    "diffsynth_reference_revision",
    "flow_master_dtype",
    "trainable_master_dtype",
    "base_model_dtype",
    "base_weights_saved",
    "single_gpu",
}
DONE_FIELDS = {
    "format",
    "optimizer_steps",
    "final_adapter_checkpoint",
    "final_adapter_sha256",
    "config_sha256",
    "manifest_sha256",
    "base_checkpoint_sha256",
    "special_tokens_sha256",
    "encoder_contract_sha256",
    "diffsynth_reference_revision",
    "flow_master_dtype",
    "trainable_master_dtype",
    "base_weights_saved",
    "lora_module_count",
    "trainable_model_adapter_parameters",
    "trainable_router_parameters",
    "elapsed_seconds",
    "torch_version",
    "torch_hip_version",
    "accelerator_name",
    "accelerator_peak_memory_allocated_bytes",
    "accelerator_peak_memory_reserved_bytes",
}
METRIC_FIELDS = {
    "step",
    "epoch",
    "batch",
    "atom_ids",
    "loss",
    "grad_norm",
    "gradient_groups",
    "timestep_id",
    "timestep_mean",
    "sigma_mean",
    "flow_training_weight",
    "learning_rates",
    "source_visual_tokens",
    "elapsed_seconds",
}
GRADIENT_GROUPS = {"lora", "visual_adapter", "vlm_projection", "router"}
GRADIENT_GROUP_FIELDS = {
    "parameter_tensors",
    "parameter_elements",
    "pre_clip_l2_norm",
}
LOSS_FIELDS = {
    "total",
    "velocity_edit",
    "velocity_preserve",
    "x0_boundary",
    "x0_temporal_outside",
    "router",
    "router_bce",
    "router_dice",
}


class SmokeVerificationError(ValueError):
    """Raised when a smoke artifact or restored adapter is inconsistent."""


@dataclass(frozen=True)
class VerifiedSmokeArtifacts:
    """Digest-verified one-step artifacts, before official model loading."""

    output_dir: Path
    checkpoint_dir: Path
    checkpoint_path: Path
    manifest_path: Path
    adapter_path: Path
    run: dict[str, Any]
    done: dict[str, Any]
    metric: dict[str, Any]
    config: dict[str, Any]
    config_sha256: str
    checkpoint_sha256: str
    manifest_sha256: str
    special_tokens_sha256: str | None
    adapter_sha256: str
    encoder_contract_sha256: str


ModelLoader = Callable[
    [Path, Path, Mapping[str, Any]], tuple[nn.Module, Any, Path]
]


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SmokeVerificationError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SmokeVerificationError(f"non-finite JSON constant is forbidden: {value}")


def _parse_json_text(text: str, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SmokeVerificationError(f"invalid JSON in {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise SmokeVerificationError(f"{name} must contain one JSON object")
    return value


def _regular_file(path: Path, *, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise SmokeVerificationError(
            f"{name} must be a regular non-symlink file: {path}"
        )
    return path


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    _regular_file(path, name=name)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SmokeVerificationError(f"{name} is not UTF-8: {path}") from exc
    return _parse_json_text(text, name=name)


def _expect_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise SmokeVerificationError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, *, name: str, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SmokeVerificationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_number(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise SmokeVerificationError(f"{name} must be a finite number")
    result = float(value)
    if nonnegative and result < 0.0:
        raise SmokeVerificationError(f"{name} must be non-negative")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SmokeVerificationError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SmokeVerificationError(f"{name} must be a non-negative integer")
    return value


def _read_one_metric(path: Path) -> dict[str, Any]:
    _regular_file(path, name="metrics.jsonl")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SmokeVerificationError("metrics.jsonl is not UTF-8") from exc
    if any(not line.strip() for line in lines):
        raise SmokeVerificationError("metrics.jsonl contains an empty record")
    if len(lines) != 1:
        raise SmokeVerificationError(
            f"metrics.jsonl must contain exactly one record, found {len(lines)}"
        )
    return _parse_json_text(lines[0], name="metrics.jsonl record 1")


def _validate_metric(metric: dict[str, Any], config: Mapping[str, Any]) -> None:
    _expect_fields(metric, METRIC_FIELDS, name="metrics record")
    if metric["step"] != 1 or isinstance(metric["step"], bool):
        raise SmokeVerificationError("metrics step must equal 1")
    _nonnegative_int(metric["epoch"], name="metrics.epoch")
    _nonnegative_int(metric["batch"], name="metrics.batch")
    atom_ids = metric["atom_ids"]
    if (
        not isinstance(atom_ids, list)
        or not atom_ids
        or any(not isinstance(item, str) or not item for item in atom_ids)
        or len(set(atom_ids)) != len(atom_ids)
    ):
        raise SmokeVerificationError(
            "metrics.atom_ids must be a non-empty unique string list"
        )
    if len(atom_ids) > int(config["training"]["batch_size"]):
        raise SmokeVerificationError("metrics.atom_ids exceeds configured batch size")

    loss = metric["loss"]
    if not isinstance(loss, dict):
        raise SmokeVerificationError("metrics.loss must be a JSON object")
    _expect_fields(loss, LOSS_FIELDS, name="metrics.loss")
    finite_loss = {
        key: _finite_number(value, name=f"metrics.loss.{key}", nonnegative=True)
        for key, value in loss.items()
    }
    if not math.isclose(
        finite_loss["router"],
        finite_loss["router_bce"] + finite_loss["router_dice"],
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise SmokeVerificationError(
            "metrics.loss.router differs from router_bce + router_dice"
        )
    timestep_id = _nonnegative_int(
        metric["timestep_id"], name="metrics.timestep_id"
    )
    scheduler = DiffSynthWanTrainingScheduler(shift=config["flow"]["shift"])
    if timestep_id >= scheduler.num_training_bins:
        raise SmokeVerificationError("metrics.timestep_id exceeds Wan training bins")
    scheduled = scheduler.at(timestep_id, len(atom_ids))
    timestep = _finite_number(metric["timestep_mean"], name="metrics.timestep_mean")
    sigma = _finite_number(metric["sigma_mean"], name="metrics.sigma_mean")
    flow_training_weight = _finite_number(
        metric["flow_training_weight"],
        name="metrics.flow_training_weight",
        nonnegative=True,
    )
    for name, actual, expected in (
        ("timestep_mean", timestep, float(scheduled.timestep.mean())),
        ("sigma_mean", sigma, float(scheduled.sigma.mean())),
        ("flow_training_weight", flow_training_weight, float(scheduled.flow_weight)),
    ):
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
            raise SmokeVerificationError(
                f"metrics.{name} differs from DiffSynth scheduler table"
            )

    weights = config["loss_weights"]
    raw_flow_total = sum(
        float(weights[key]) * finite_loss[key]
        for key in (
            "velocity_edit",
            "velocity_preserve",
            "x0_boundary",
            "x0_temporal_outside",
        )
    )
    expected_total = (
        flow_training_weight * raw_flow_total
        + float(weights["router"]) * finite_loss["router"]
    )
    if not math.isclose(
        finite_loss["total"], expected_total, rel_tol=1e-5, abs_tol=1e-6
    ):
        raise SmokeVerificationError(
            "metrics.loss.total differs from the configured weighted components"
        )
    _finite_number(metric["grad_norm"], name="metrics.grad_norm", nonnegative=True)
    gradient_groups = metric["gradient_groups"]
    if not isinstance(gradient_groups, dict):
        raise SmokeVerificationError("metrics.gradient_groups must be a JSON object")
    _expect_fields(
        gradient_groups, GRADIENT_GROUPS, name="metrics.gradient_groups"
    )
    for group_name, stats in gradient_groups.items():
        if not isinstance(stats, dict):
            raise SmokeVerificationError(
                f"metrics.gradient_groups.{group_name} must be a JSON object"
            )
        _expect_fields(
            stats,
            GRADIENT_GROUP_FIELDS,
            name=f"metrics.gradient_groups.{group_name}",
        )
        _positive_int(
            stats["parameter_tensors"],
            name=f"metrics.gradient_groups.{group_name}.parameter_tensors",
        )
        _positive_int(
            stats["parameter_elements"],
            name=f"metrics.gradient_groups.{group_name}.parameter_elements",
        )
        norm = _finite_number(
            stats["pre_clip_l2_norm"],
            name=f"metrics.gradient_groups.{group_name}.pre_clip_l2_norm",
            nonnegative=True,
        )
        if norm == 0.0:
            raise SmokeVerificationError(
                f"metrics gradient group {group_name!r} has a zero norm"
            )
    if not 0.0 <= sigma <= 1.0:
        raise SmokeVerificationError("metrics.sigma_mean must lie in [0, 1]")
    learning_rates = metric["learning_rates"]
    if not isinstance(learning_rates, dict):
        raise SmokeVerificationError("metrics.learning_rates must be a JSON object")
    expected_learning_rates = {
        "lora_router": float(config["optimizer"]["learning_rate"]),
        "pretrained_condition_adapters": float(
            config["optimizer"]["pretrained_adapter_learning_rate"]
        ),
    }
    if set(learning_rates) != set(expected_learning_rates):
        raise SmokeVerificationError("metrics.learning_rates fields differ")
    for name, expected in expected_learning_rates.items():
        actual = _finite_number(
            learning_rates[name],
            name=f"metrics.learning_rates.{name}",
            nonnegative=True,
        )
        if actual != expected:
            raise SmokeVerificationError(
                f"metrics.learning_rates.{name} differs from config"
            )
    _positive_int(
        metric["source_visual_tokens"], name="metrics.source_visual_tokens"
    )
    _finite_number(
        metric["elapsed_seconds"],
        name="metrics.elapsed_seconds",
        nonnegative=True,
    )


def _safe_adapter_metadata(path: Path) -> tuple[dict[str, Any], str]:
    """Read adapter metadata with the same safe-loading requirement as PACT."""

    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise SmokeVerificationError(
            "this PyTorch lacks safe weights_only loading; upgrade PyTorch"
        )
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SmokeVerificationError(f"cannot safely read final adapter: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SmokeVerificationError("final adapter must contain a mapping")
    metadata_fields = (
        "format",
        "step",
        "config_sha256",
        "validated_config",
        "base_checkpoint_sha256",
        "manifest_sha256",
        "special_tokens_sha256",
        "encoder_contract_sha256",
    )
    missing = set(metadata_fields) - set(payload)
    if missing:
        raise SmokeVerificationError(
            f"final adapter lacks provenance fields: {sorted(missing)}"
        )
    return {field: payload[field] for field in metadata_fields}, digest


def verify_smoke_artifacts(
    run_output_dir: os.PathLike[str] | str,
    checkpoint_dir: os.PathLike[str] | str,
) -> VerifiedSmokeArtifacts:
    """Verify one-step files and provenance without constructing OmniVideo."""

    output = Path(os.path.abspath(Path(run_output_dir).expanduser()))
    if output.is_symlink() or not output.is_dir():
        raise SmokeVerificationError(
            f"run output must be a non-symlink directory: {output}"
        )
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise SmokeVerificationError(
            f"checkpoint directory does not exist: {checkpoint_root}"
        )

    run = _read_json(output / "run.json", name="run.json")
    done = _read_json(output / "done.json", name="done.json")
    metric = _read_one_metric(output / "metrics.jsonl")
    _expect_fields(run, RUN_FIELDS, name="run.json")
    _expect_fields(done, DONE_FIELDS, name="done.json")
    if run["format"] != RUN_FORMAT:
        raise SmokeVerificationError(f"run.json format must be {RUN_FORMAT}")
    if done["format"] != DONE_FORMAT:
        raise SmokeVerificationError(f"done.json format must be {DONE_FORMAT}")
    if run["base_weights_saved"] is not False:
        raise SmokeVerificationError("run.json must attest base_weights_saved=false")
    _nonnegative_int(run["single_gpu"], name="run.single_gpu")
    if done["optimizer_steps"] != 1 or isinstance(done["optimizer_steps"], bool):
        raise SmokeVerificationError("done.json optimizer_steps must equal 1")
    if done["base_weights_saved"] is not False:
        raise SmokeVerificationError("done.json must attest base_weights_saved=false")
    _positive_int(done["lora_module_count"], name="done.lora_module_count")
    _positive_int(
        done["trainable_model_adapter_parameters"],
        name="done.trainable_model_adapter_parameters",
    )
    _positive_int(
        done["trainable_router_parameters"],
        name="done.trainable_router_parameters",
    )
    _finite_number(done["elapsed_seconds"], name="done.elapsed_seconds", nonnegative=True)
    for field in ("torch_version", "accelerator_name"):
        if not isinstance(done[field], str) or not done[field]:
            raise SmokeVerificationError(f"done.{field} must be a non-empty string")
    if done["torch_hip_version"] is not None and (
        not isinstance(done["torch_hip_version"], str) or not done["torch_hip_version"]
    ):
        raise SmokeVerificationError("done.torch_hip_version must be null or a non-empty string")
    _positive_int(
        done["accelerator_peak_memory_allocated_bytes"],
        name="done.accelerator_peak_memory_allocated_bytes",
    )
    _positive_int(
        done["accelerator_peak_memory_reserved_bytes"],
        name="done.accelerator_peak_memory_reserved_bytes",
    )

    try:
        config = validate_training_config(run["config"])
    except (TypeError, ValueError) as exc:
        raise SmokeVerificationError(f"run.json embeds an invalid config: {exc}") from exc
    if config["training"]["max_steps"] != 1:
        raise SmokeVerificationError("smoke config training.max_steps must equal 1")
    expected_lora_count = expected_lora_module_count(config["lora"]["scope"], 30)
    if done["lora_module_count"] != expected_lora_count:
        raise SmokeVerificationError(
            "done.json lora_module_count differs from configured closed scope"
        )
    for container_name, container in (("run", run), ("done", done)):
        if container["diffsynth_reference_revision"] != DIFFSYNTH_REFERENCE_REVISION:
            raise SmokeVerificationError(
                f"{container_name}.diffsynth_reference_revision differs"
            )
        if container["flow_master_dtype"] != "float32":
            raise SmokeVerificationError(f"{container_name}.flow_master_dtype must be float32")
        if container["trainable_master_dtype"] != "float32":
            raise SmokeVerificationError(
                f"{container_name}.trainable_master_dtype must be float32"
            )
    if run["base_model_dtype"] != "bfloat16":
        raise SmokeVerificationError("run.base_model_dtype must be bfloat16")
    _validate_metric(metric, config)

    recorded_checkpoint_dir = run["checkpoint_dir"]
    if not isinstance(recorded_checkpoint_dir, str) or not Path(
        recorded_checkpoint_dir
    ).is_absolute():
        raise SmokeVerificationError("run.checkpoint_dir must be an absolute path")
    if Path(recorded_checkpoint_dir).expanduser().resolve() != checkpoint_root:
        raise SmokeVerificationError(
            "provided checkpoint directory differs from run.json provenance"
        )

    config_digest = _digest(run["config_sha256"], name="run.config_sha256")
    manifest_digest = _digest(run["manifest_sha256"], name="run.manifest_sha256")
    checkpoint_digest = _digest(
        run["checkpoint_sha256"], name="run.checkpoint_sha256"
    )
    special_digest = _digest(
        run["special_tokens_sha256"],
        name="run.special_tokens_sha256",
        allow_none=True,
    )
    encoder_contract_digest = _digest(
        run["encoder_contract_sha256"],
        name="run.encoder_contract_sha256",
    )
    adapter_digest = _digest(
        done["final_adapter_sha256"], name="done.final_adapter_sha256"
    )
    for field, expected in (
        ("config_sha256", config_digest),
        ("manifest_sha256", manifest_digest),
        ("base_checkpoint_sha256", checkpoint_digest),
        ("special_tokens_sha256", special_digest),
        ("encoder_contract_sha256", encoder_contract_digest),
    ):
        if done[field] != expected:
            raise SmokeVerificationError(
                f"done.{field} differs from run.json provenance"
            )

    manifest_value = run["manifest"]
    if not isinstance(manifest_value, str) or not Path(manifest_value).is_absolute():
        raise SmokeVerificationError("run.manifest must be an absolute path")
    manifest_path = _regular_file(
        Path(manifest_value), name="recorded training manifest"
    )
    actual_manifest_digest = _sha256(manifest_path)
    if actual_manifest_digest != manifest_digest:
        raise SmokeVerificationError("training manifest SHA-256 differs from run.json")

    checkpoint_path = checkpoint_root / "transformer" / "pytorch_model.pt"
    if not checkpoint_path.is_file():
        raise SmokeVerificationError(
            f"official transformer checkpoint is missing: {checkpoint_path}"
        )
    actual_checkpoint_digest = _sha256(checkpoint_path)
    if actual_checkpoint_digest != checkpoint_digest:
        raise SmokeVerificationError(
            "official checkpoint SHA-256 differs from run.json"
        )

    special_path = checkpoint_root / "special_tokens.pkl"
    if special_path.is_file():
        actual_special_digest: str | None = _sha256(special_path)
    else:
        actual_special_digest = None
    if actual_special_digest != special_digest:
        raise SmokeVerificationError(
            "special_tokens.pkl presence or SHA-256 differs from run.json"
        )
    if config["model"]["require_special_tokens"] and actual_special_digest is None:
        raise SmokeVerificationError("required special_tokens.pkl is missing")

    adapter_name = done["final_adapter_checkpoint"]
    expected_adapter_name = "adapters_final_step_00000001.pt"
    if adapter_name != expected_adapter_name:
        raise SmokeVerificationError(
            f"final adapter filename must be {expected_adapter_name}"
        )
    adapter_path = _regular_file(output / adapter_name, name="final adapter")
    adapter_metadata, actual_adapter_digest = _safe_adapter_metadata(adapter_path)
    if actual_adapter_digest != adapter_digest:
        raise SmokeVerificationError("final adapter SHA-256 differs from done.json")
    expected_adapter_metadata = {
        "format": ADAPTER_FORMAT,
        "step": 1,
        "config_sha256": config_digest,
        "validated_config": config,
        "base_checkpoint_sha256": checkpoint_digest,
        "manifest_sha256": manifest_digest,
        "special_tokens_sha256": special_digest,
        "encoder_contract_sha256": encoder_contract_digest,
    }
    if adapter_metadata != expected_adapter_metadata:
        raise SmokeVerificationError(
            "final adapter metadata differs from run/done provenance"
        )

    return VerifiedSmokeArtifacts(
        output_dir=output,
        checkpoint_dir=checkpoint_root,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        adapter_path=adapter_path,
        run=run,
        done=done,
        metric=metric,
        config=config,
        config_sha256=config_digest,
        checkpoint_sha256=actual_checkpoint_digest,
        manifest_sha256=actual_manifest_digest,
        special_tokens_sha256=actual_special_digest,
        adapter_sha256=actual_adapter_digest,
        encoder_contract_sha256=encoder_contract_digest,
    )


def _official_model_loader(
    omnivideo_root: Path,
    checkpoint_dir: Path,
    config: Mapping[str, Any],
) -> tuple[nn.Module, Any, Path]:
    # Lazy import keeps artifact-only validation independent of the upstream
    # checkout and ensures the verifier uses the exact training constructor.
    from train_omnivideo2_pact import _load_official_model

    return _load_official_model(omnivideo_root, checkpoint_dir, config)


def _validate_restored_bundle(
    model: nn.Module,
    loaded: LoadedPactAdapters,
    artifacts: VerifiedSmokeArtifacts,
) -> None:
    if loaded.step != 1:
        raise SmokeVerificationError("strictly restored adapter step differs from 1")
    if loaded.checkpoint_sha256 != artifacts.adapter_sha256:
        raise SmokeVerificationError("restored adapter file digest differs")
    if loaded.manifest_sha256 != artifacts.manifest_sha256:
        raise SmokeVerificationError("restored adapter manifest digest differs")
    if loaded.special_tokens_sha256 != artifacts.special_tokens_sha256:
        raise SmokeVerificationError("restored adapter special-token digest differs")
    if loaded.encoder_contract_sha256 != artifacts.encoder_contract_sha256:
        raise SmokeVerificationError("restored adapter encoder-contract digest differs")
    if loaded.config != artifacts.config:
        raise SmokeVerificationError("restored adapter config differs from run.json")
    try:
        expected_lora_modules = expected_lora_module_count(
            artifacts.config["lora"]["scope"], int(model.wan_model.num_layers)
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise SmokeVerificationError(
            "restored base model lacks a valid Wan layer count"
        ) from exc
    if len(loaded.lora_modules) != expected_lora_modules:
        raise SmokeVerificationError(
            "restored LoRA count differs from configured closed scope"
        )


def verify_gpu_smoke(
    run_output_dir: os.PathLike[str] | str,
    omnivideo_root: os.PathLike[str] | str,
    checkpoint_dir: os.PathLike[str] | str,
    *,
    model_loader: ModelLoader | None = None,
) -> dict[str, Any]:
    """Verify artifacts, reconstruct the official base, and restore adapters."""

    artifacts = verify_smoke_artifacts(run_output_dir, checkpoint_dir)
    upstream_root = Path(omnivideo_root).expanduser().resolve()
    if not upstream_root.is_dir():
        raise SmokeVerificationError(
            f"OmniVideo root does not exist: {upstream_root}"
        )
    loader = _official_model_loader if model_loader is None else model_loader
    model, _official, loaded_checkpoint_path = loader(
        upstream_root, artifacts.checkpoint_dir, artifacts.config
    )
    if not isinstance(model, nn.Module):
        raise SmokeVerificationError("official model loader did not return nn.Module")
    if Path(loaded_checkpoint_path).expanduser().resolve() != artifacts.checkpoint_path.resolve():
        raise SmokeVerificationError(
            "official model loader used a different transformer checkpoint"
        )
    loaded = load_pact_adapter_bundle(
        model,
        artifacts.adapter_path,
        expected_base_checkpoint_sha256=artifacts.checkpoint_sha256,
        expected_manifest_sha256=artifacts.manifest_sha256,
        expected_special_tokens_sha256=artifacts.special_tokens_sha256,
        expected_encoder_contract_sha256=artifacts.encoder_contract_sha256,
    )
    _validate_restored_bundle(model, loaded, artifacts)

    loss = artifacts.metric["loss"]
    return {
        "format": SUMMARY_FORMAT,
        "status": "verified",
        "optimizer_steps": 1,
        "metrics_records": 1,
        "official_model_reconstructed": True,
        "adapter_strictly_reloaded": True,
        "lora_modules": len(loaded.lora_modules),
        "final_adapter_checkpoint": artifacts.adapter_path.name,
        "final_adapter_sha256": artifacts.adapter_sha256,
        "base_checkpoint_sha256": artifacts.checkpoint_sha256,
        "manifest_sha256": artifacts.manifest_sha256,
        "special_tokens_sha256": artifacts.special_tokens_sha256,
        "encoder_contract_sha256": artifacts.encoder_contract_sha256,
        "config_sha256": artifacts.config_sha256,
        "loss": {key: float(loss[key]) for key in sorted(loss)},
        "grad_norm": float(artifacts.metric["grad_norm"]),
        "gradient_groups": artifacts.metric["gradient_groups"],
        "sigma_mean": float(artifacts.metric["sigma_mean"]),
        "timestep_id": int(artifacts.metric["timestep_id"]),
        "timestep_mean": float(artifacts.metric["timestep_mean"]),
        "flow_training_weight": float(artifacts.metric["flow_training_weight"]),
        "accelerator_name": artifacts.done["accelerator_name"],
        "accelerator_peak_memory_allocated_bytes": artifacts.done[
            "accelerator_peak_memory_allocated_bytes"
        ],
        "accelerator_peak_memory_reserved_bytes": artifacts.done[
            "accelerator_peak_memory_reserved_bytes"
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--omnivideo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = verify_gpu_smoke(
            args.run_output_dir,
            args.omnivideo_root,
            args.checkpoint_dir,
        )
    except Exception as exc:
        failure = {
            "format": SUMMARY_FORMAT,
            "status": "rejected",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
