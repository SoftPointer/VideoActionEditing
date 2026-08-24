#!/usr/bin/env python3
"""Verify a four-rank one-step run over one real preview payload.

After digest-verifying the run artifacts, this verifier reconstructs a clean
official OmniVideo2-1.3B base, injects the checkpoint's exact validated LoRA
configuration, strictly reloads every adapter tensor, and strictly restores
the motion planner.  It proves only an engineering path: a one-row payload is
replicated by the DistributedSampler; no quality or convergence claim follows.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from action import (  # noqa: E402
    ActionConfig,
    TemporalMotionPlanPredictor,
    validate_action_config,
)
from action.omni import (  # noqa: E402
    enable_action_lora,
    load_official_omnivideo2_1_3b,
    load_special_tokens,
)
from action.checkpoint_contract import (  # noqa: E402
    ACTION_ADAPTER_CHECKPOINT_FIELDS,
    OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
    OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS,
    OMNIVIDEO2_1_3B_SPECIAL_TOKENS_SHA256,
    OMNIVIDEO2_1_3B_TRANSFORMER_SHA256,
    action_activation_contract_record,
    special_token_layout_record,
)
from pact.lora import (  # noqa: E402
    iter_lora_modules,
    load_lora_state_dict,
    lora_state_dict,
)


SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_RE = re.compile(r"[0-9a-f]{40}")


class RealSmokeAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class StrictAdapterReload:
    injected_modules: tuple[str, ...]
    lora_tensor_count: int
    planner_tensor_count: int
    base_checkpoint_sha256: str


ModelLoader = Callable[
    [Path, Path, ActionConfig], tuple[nn.Module, Any, Path]
]
SpecialTokenLoader = Callable[..., tuple[dict[str, Tensor] | None, int, str | None]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--omnivideo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-sample-id", required=True)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--expected-source-revision", required=True)
    return parser.parse_args(argv)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RealSmokeAuditError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RealSmokeAuditError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RealSmokeAuditError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_state(value: Any, *, name: str) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise RealSmokeAuditError(f"{name} must be a non-empty mapping")
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, Tensor):
            raise RealSmokeAuditError(f"{name} contains a non-tensor entry")
        if tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()):
            raise RealSmokeAuditError(f"{name}.{key} is not finite CPU state")
    return value


def _regular_file(path: Path, *, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RealSmokeAuditError(f"{name} must be a regular non-symlink file: {path}")
    return path


def _recorded_absolute_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise RealSmokeAuditError(f"{name} must be an absolute path")
    return Path(value).expanduser().resolve()


def _assert_exact_tensor_state(
    restored: Mapping[str, Any],
    saved: Mapping[str, Tensor],
    *,
    name: str,
) -> None:
    if set(restored) != set(saved):
        raise RealSmokeAuditError(
            f"{name} state keys differ after strict reload: "
            f"missing={sorted(set(saved) - set(restored))}, "
            f"unexpected={sorted(set(restored) - set(saved))}"
        )
    for key, expected in saved.items():
        actual = restored[key]
        if (
            not isinstance(actual, Tensor)
            or actual.device.type != "cpu"
            or actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not torch.equal(actual, expected)
        ):
            raise RealSmokeAuditError(
                f"{name}.{key} differs after strict reload"
            )


def _canonical_object_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optimizer_runtime_rows(
    value: Any,
    *,
    expected_world_size: int,
    name: str,
) -> list[dict[str, Any]]:
    expected_fields = {
        "rank",
        "optimizer_step",
        "microbatches",
        "isolated_optimizer_window_seconds",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
    }
    if not isinstance(value, list) or len(value) != expected_world_size:
        raise RealSmokeAuditError(
            f"{name} must contain exactly one row per rank"
        )
    rows: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise RealSmokeAuditError(f"{name} row fields differ")
        row = dict(raw)
        rank = row["rank"]
        if type(rank) is not int or not 0 <= rank < expected_world_size:
            raise RealSmokeAuditError(f"{name} rank differs")
        if row["optimizer_step"] != 1 or type(row["optimizer_step"]) is not int:
            raise RealSmokeAuditError(f"{name} optimizer step differs")
        if row["microbatches"] != 1 or type(row["microbatches"]) is not int:
            raise RealSmokeAuditError(f"{name} microbatch count differs")
        seconds = row["isolated_optimizer_window_seconds"]
        allocated = row["peak_memory_allocated_bytes"]
        reserved = row["peak_memory_reserved_bytes"]
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(float(seconds))
            or float(seconds) <= 0.0
        ):
            raise RealSmokeAuditError(f"{name} duration is not positive finite")
        if (
            type(allocated) is not int
            or type(reserved) is not int
            or allocated <= 0
            or reserved <= 0
            or reserved < allocated
        ):
            raise RealSmokeAuditError(f"{name} peak memory is invalid")
        rows.append(row)
    rows.sort(key=lambda row: row["rank"])
    if [row["rank"] for row in rows] != list(range(expected_world_size)):
        raise RealSmokeAuditError(f"{name} has duplicate or missing ranks")
    return rows


def _validate_runtime_evidence(
    metric: Mapping[str, Any],
    done: Mapping[str, Any],
    *,
    expected_world_size: int,
) -> list[dict[str, Any]]:
    runtime = _optimizer_runtime_rows(
        metric.get("runtime_all_ranks"),
        expected_world_size=expected_world_size,
        name="metric runtime_all_ranks",
    )
    first = _optimizer_runtime_rows(
        done.get("first_optimizer_step_runtime_all_ranks"),
        expected_world_size=expected_world_size,
        name="done first_optimizer_step_runtime_all_ranks",
    )
    if first != runtime:
        raise RealSmokeAuditError(
            "done first-step runtime differs from the metric row"
        )

    durations = done.get("one_step_duration_seconds_all_ranks")
    expected_durations = {
        str(row["rank"]): row["isolated_optimizer_window_seconds"]
        for row in runtime
    }
    if durations != expected_durations:
        raise RealSmokeAuditError("done one-step per-rank durations differ")

    maxima = done.get("runtime_maxima_all_ranks")
    if not isinstance(maxima, list) or len(maxima) != expected_world_size:
        raise RealSmokeAuditError(
            "done runtime_maxima_all_ranks must contain every rank"
        )
    expected_maxima = [
        {
            "rank": row["rank"],
            "optimizer_windows": 1,
            "max_isolated_optimizer_window_seconds": row[
                "isolated_optimizer_window_seconds"
            ],
            "max_peak_memory_allocated_bytes": row[
                "peak_memory_allocated_bytes"
            ],
            "max_peak_memory_reserved_bytes": row[
                "peak_memory_reserved_bytes"
            ],
        }
        for row in runtime
    ]
    if maxima != expected_maxima:
        raise RealSmokeAuditError("done per-rank runtime maxima differ")
    if (
        done.get("rank0_peak_memory_allocated_bytes")
        != runtime[0]["peak_memory_allocated_bytes"]
        or done.get("rank0_peak_memory_reserved_bytes")
        != runtime[0]["peak_memory_reserved_bytes"]
    ):
        raise RealSmokeAuditError("done rank-zero peak memory differs")
    return runtime


def _validate_source_budget(
    value: Any,
    *,
    config: ActionConfig,
    source_latent_shape: Sequence[int],
    expected_sample_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RealSmokeAuditError("source context budget must be an object")
    budget = dict(value)
    source_shape = [1, *source_latent_shape]
    latent_t, latent_h, latent_w = source_latent_shape[1:]
    patch_t, patch_h, patch_w = config.model.visual_patch_size
    visual_tokens = (
        (latent_t // patch_t) * (latent_h // patch_h) * (latent_w // patch_w)
    )
    nonvisual = budget.get("nonvisual_tokens")
    if type(nonvisual) is not int or nonvisual <= 0:
        raise RealSmokeAuditError("source context nonvisual token count is invalid")
    total = nonvisual + visual_tokens
    expected = {
        "source_shape": source_shape,
        "visual_tokens": visual_tokens,
        "total_tokens": total,
        "budget_tokens": config.model.max_context_len,
        "fixed_budget_padding_tokens": config.model.max_context_len - total,
        "fits": True,
        "source_truncated": False,
    }
    for field, expected_value in expected.items():
        if budget.get(field) != expected_value:
            raise RealSmokeAuditError(
                f"source context budget {field} differs"
            )
    if total > config.model.max_context_len:
        raise RealSmokeAuditError("source context exceeds the configured budget")
    for field, expected_value in (
        ("sample_id", expected_sample_id),
        ("task_type", "action_edit"),
        ("context_padding_mode", config.model.context_padding_mode),
        ("effective_wan_text_len", config.model.max_context_len),
        ("effective_padding_tokens", config.model.max_context_len - total),
        ("compressed", False),
        ("first_frame_exact", True),
        ("original_visual_tokens", visual_tokens),
        ("output_visual_tokens", visual_tokens),
        ("output_total_tokens", total),
    ):
        if field in budget and budget[field] != expected_value:
            raise RealSmokeAuditError(f"source context budget {field} differs")
    return budget


def _official_model_loader(
    omnivideo_root: Path,
    checkpoint_dir: Path,
    config: ActionConfig,
) -> tuple[nn.Module, Any, Path]:
    return load_official_omnivideo2_1_3b(
        omnivideo_root,
        checkpoint_dir,
        max_context_len=config.model.max_context_len,
        visual_patch_size=config.model.visual_patch_size,
        wan_patch_size=config.model.wan_patch_size,
    )


def reconstruct_and_strictly_reload_adapter(
    *,
    omnivideo_root: Path,
    checkpoint_dir: Path,
    config: ActionConfig,
    adapter_checkpoint: Mapping[str, Any],
    expected_base_checkpoint_sha256: str,
    model_loader: ModelLoader | None = None,
) -> StrictAdapterReload:
    """Rebuild a clean official base and exactly restore LoRA plus planner state."""

    upstream_root = omnivideo_root.expanduser().resolve()
    checkpoint_root = checkpoint_dir.expanduser().resolve()
    if upstream_root.is_symlink() or not upstream_root.is_dir():
        raise RealSmokeAuditError(
            f"OmniVideo root must be a non-symlink directory: {upstream_root}"
        )
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise RealSmokeAuditError(
            f"checkpoint root must be a non-symlink directory: {checkpoint_root}"
        )
    expected_base = _digest(
        expected_base_checkpoint_sha256, name="expected base checkpoint"
    )
    base_path = _regular_file(
        checkpoint_root / "transformer" / "pytorch_model.pt",
        name="official transformer checkpoint",
    )
    actual_base = _sha256(base_path)
    if actual_base != expected_base:
        raise RealSmokeAuditError(
            "official base checkpoint digest differs before model reconstruction"
        )

    loader = _official_model_loader if model_loader is None else model_loader
    try:
        model, _official, loaded_checkpoint = loader(
            upstream_root, checkpoint_root, config
        )
    except RealSmokeAuditError:
        raise
    except Exception as error:
        raise RealSmokeAuditError(
            f"official model reconstruction failed: {error}"
        ) from error
    if not isinstance(model, nn.Module):
        raise RealSmokeAuditError("official model loader did not return nn.Module")
    if Path(loaded_checkpoint).expanduser().resolve() != base_path:
        raise RealSmokeAuditError(
            "official model loader used a different transformer checkpoint"
        )
    preexisting = [name for name, _module in iter_lora_modules(model)]
    if preexisting:
        raise RealSmokeAuditError(
            f"reconstructed official base already contains LoRA: {preexisting[:20]}"
        )

    saved_modules = adapter_checkpoint.get("lora_modules")
    if (
        not isinstance(saved_modules, list)
        or not saved_modules
        or any(not isinstance(name, str) or not name for name in saved_modules)
        or len(set(saved_modules)) != len(saved_modules)
    ):
        raise RealSmokeAuditError(
            "adapter checkpoint lora_modules must be a non-empty unique string list"
        )
    try:
        injected, _trainable = enable_action_lora(
            model,
            scope=config.lora.scope,
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
        )
    except Exception as error:
        raise RealSmokeAuditError(
            f"exact-config LoRA injection failed: {error}"
        ) from error
    if injected != saved_modules:
        raise RealSmokeAuditError(
            "checkpoint injected module list differs from exact-config reconstruction"
        )

    saved_lora = _finite_state(
        adapter_checkpoint.get("lora_state_dict"), name="LoRA state"
    )
    try:
        loaded_modules = load_lora_state_dict(model, saved_lora)
    except Exception as error:
        raise RealSmokeAuditError(
            f"strict load_lora_state_dict failed: {error}"
        ) from error
    if loaded_modules != injected:
        raise RealSmokeAuditError(
            "strict LoRA reload module order differs from injection order"
        )
    _assert_exact_tensor_state(
        lora_state_dict(model), saved_lora, name="LoRA"
    )

    planner = TemporalMotionPlanPredictor(
        config.planner.num_tokens,
        input_dim=config.planner.input_dim,
        hidden_dim=config.planner.hidden_dim,
        depth=config.planner.depth,
    ).to(device="cpu", dtype=torch.float32)
    saved_planner = _finite_state(
        adapter_checkpoint.get("motion_planner_state_dict"), name="planner state"
    )
    try:
        incompatible = planner.load_state_dict(saved_planner, strict=True)
    except Exception as error:
        raise RealSmokeAuditError(f"strict planner load failed: {error}") from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RealSmokeAuditError(
            "strict planner load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    restored_planner = {
        key: tensor.detach().cpu().clone()
        for key, tensor in planner.state_dict().items()
    }
    _assert_exact_tensor_state(
        restored_planner, saved_planner, name="planner"
    )
    return StrictAdapterReload(
        injected_modules=tuple(injected),
        lora_tensor_count=len(saved_lora),
        planner_tensor_count=len(saved_planner),
        base_checkpoint_sha256=actual_base,
    )


def verify(
    materialized_root: Path,
    run_root: Path,
    omnivideo_root: Path,
    checkpoint_dir: Path,
    *,
    expected_sample_id: str,
    expected_world_size: int,
    expected_source_revision: str,
    model_loader: ModelLoader | None = None,
    special_token_loader: SpecialTokenLoader | None = None,
    _test_only_allow_unpinned_checkpoint: bool = False,
) -> dict[str, Any]:
    materialized_root = materialized_root.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    omnivideo_root = omnivideo_root.expanduser().resolve()
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if type(_test_only_allow_unpinned_checkpoint) is not bool:
        raise RealSmokeAuditError("test-only checkpoint override must be boolean")
    if _test_only_allow_unpinned_checkpoint and (
        model_loader is None or special_token_loader is None
    ):
        raise RealSmokeAuditError(
            "test-only unpinned override requires both injected loaders"
        )
    if not _test_only_allow_unpinned_checkpoint and (
        model_loader is not None or special_token_loader is not None
    ):
        raise RealSmokeAuditError(
            "injected loaders require the explicit test-only unpinned override"
        )
    if materialized_root.is_symlink() or not materialized_root.is_dir():
        raise RealSmokeAuditError("materialized_root must be a non-symlink directory")
    if run_root.is_symlink() or not run_root.is_dir():
        raise RealSmokeAuditError("run_root must be a non-symlink directory")
    if omnivideo_root.is_symlink() or not omnivideo_root.is_dir():
        raise RealSmokeAuditError("omnivideo_root must be a non-symlink directory")
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise RealSmokeAuditError("checkpoint_dir must be a non-symlink directory")
    if expected_world_size <= 1:
        raise RealSmokeAuditError("expected_world_size must be multi-rank")
    if GIT_RE.fullmatch(expected_source_revision) is None:
        raise RealSmokeAuditError("expected source revision must be a Git SHA")

    materialization_path = materialized_root / "materialization.json"
    materialization = _json(materialization_path)
    materialized_verification = _json(materialized_root / "verification.json")
    if (
        materialization.get("schema_version")
        != "omnivideo2-action-materialization-receipt-v2"
        or materialization.get("complete") is not True
        or materialization.get("sample_count") != 1
        or materialization.get("preview_only") is not True
        or materialization.get("training_authorized") is not False
        or materialization.get("scientific_claim_authorized") is not False
        or materialization.get("target_motion_tokens_usage") != "planner_loss_only"
    ):
        raise RealSmokeAuditError("materialization receipt boundary differs")
    receipt_digest = _digest(
        materialization.get("receipt_digest"), name="materialization receipt"
    )
    receipt_without_digest = dict(materialization)
    del receipt_without_digest["receipt_digest"]
    if _canonical_object_sha256(receipt_without_digest) != receipt_digest:
        raise RealSmokeAuditError("materialization receipt digest differs")
    if (
        materialized_verification.get("complete") is not True
        or materialized_verification.get("sample_id") != expected_sample_id
        or materialized_verification.get("preview_only") is not True
        or materialized_verification.get("scientific_quality_not_tested") is not True
        or materialized_verification.get("target_motion_tokens_usage")
        != "planner_loss_only"
        or materialized_verification.get("materialization_sha256")
        != _sha256(materialization_path)
    ):
        raise RealSmokeAuditError("materialization verification differs")
    manifest_path = materialized_root / "manifest.jsonl"
    if _sha256(manifest_path) != _digest(
        materialization.get("manifest_sha256"), name="materialization manifest"
    ):
        raise RealSmokeAuditError("materialization manifest digest differs")
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or rows[0].get("sample_id") != expected_sample_id:
        raise RealSmokeAuditError("materialized sample identity differs")

    run = _json(run_root / "run.json")
    done = _json(run_root / "done.json")
    if run.get("format") != "marp-omnivideo2-action-run-v2":
        raise RealSmokeAuditError("training run format differs")
    if done.get("format") != "marp-omnivideo2-action-training-done-v2":
        raise RealSmokeAuditError("training completion format differs")
    for value, name in ((run, "run"), (done, "done")):
        if value.get("world_size") != expected_world_size:
            raise RealSmokeAuditError(f"{name} world size differs")
        if value.get("preview_only") is not True:
            raise RealSmokeAuditError(f"{name} lost preview-only status")
        if value.get("production_claim_forbidden") is not True:
            raise RealSmokeAuditError(f"{name} permits a production claim")
        if value.get("target_motion_tokens_used_by_renderer") is not False:
            raise RealSmokeAuditError(f"{name} indicates target-token leakage")
        if (
            value.get("checkpoint_contract_id")
            != OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID
            or value.get("special_token_serialized_rows")
            != OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
            or value.get("special_token_layout") != special_token_layout_record()
        ):
            raise RealSmokeAuditError(f"{name} checkpoint contract differs")
    if run.get("source_revision") != expected_source_revision:
        raise RealSmokeAuditError("source revision differs")
    source_archive_digest = _digest(
        run.get("source_archive_sha256"), name="run source archive"
    )
    if (
        done.get("source_revision") != expected_source_revision
        or done.get("source_archive_sha256") != source_archive_digest
    ):
        raise RealSmokeAuditError("done source identity differs from run")
    if (
        materialized_verification.get("source_revision")
        != expected_source_revision
        or materialized_verification.get("source_archive_sha256")
        != source_archive_digest
    ):
        raise RealSmokeAuditError(
            "materialization verification source identity differs from run"
        )
    if _recorded_absolute_path(
        run.get("omnivideo_root"), name="run.omnivideo_root"
    ) != omnivideo_root:
        raise RealSmokeAuditError(
            "provided OmniVideo root differs from run.json provenance"
        )
    if _recorded_absolute_path(
        run.get("checkpoint_dir"), name="run.checkpoint_dir"
    ) != checkpoint_dir:
        raise RealSmokeAuditError(
            "provided checkpoint directory differs from run.json provenance"
        )

    raw_config = run.get("validated_config")
    if not isinstance(raw_config, Mapping):
        raise RealSmokeAuditError("run.validated_config must be an object")
    try:
        config = validate_action_config(raw_config)
    except (TypeError, ValueError) as error:
        raise RealSmokeAuditError(
            f"run embeds an invalid action config: {error}"
        ) from error
    validated_config = config.to_dict()
    if dict(raw_config) != validated_config:
        raise RealSmokeAuditError("run validated config is not canonical")
    if config.model.context_padding_mode != "fixed_budget":
        raise RealSmokeAuditError(
            "real DDP feasibility verification requires official fixed-budget padding"
        )

    temporal_indices = (
        list(range(81))
        if config.data.temporal_mode == "full_81_25fps"
        else list(range(0, 81, 2))
    )
    temporal_subsampled = config.data.temporal_mode != "full_81_25fps"
    expected_materialization = {
        "temporal_mode": config.data.temporal_mode,
        "temporal_indices": temporal_indices,
        "temporal_subsampled": temporal_subsampled,
        "source_frame_count": config.data.expected_raw_num_frames,
        "materialized_frame_count": config.data.video_num_frames,
        "source_fps": config.data.expected_raw_fps,
        "materialized_fps": config.data.video_fps,
        "spatial_profile": config.data.spatial_profile,
        "landscape_bucket_hw": [config.data.video_height, config.data.video_width],
        "portrait_bucket_hw": [config.data.video_width, config.data.video_height],
    }
    for field, expected_value in expected_materialization.items():
        if materialization.get(field) != expected_value:
            raise RealSmokeAuditError(
                f"materialization {field} differs from the training config"
            )
    for field, expected_value in (
        ("temporal_mode", config.data.temporal_mode),
        ("spatial_profile", config.data.spatial_profile),
        ("temporal_indices_verified", True),
        ("temporal_subsampled", temporal_subsampled),
    ):
        if materialized_verification.get(field) != expected_value:
            raise RealSmokeAuditError(
                f"materialization verification {field} differs"
            )
    allowed_latent_shapes = {
        tuple(shape) for shape in config.data.expected_latent_shapes
    }
    source_latent_shape = materialized_verification.get("source_latent_shape")
    target_latent_shape = materialized_verification.get("target_latent_shape")
    if (
        not isinstance(source_latent_shape, list)
        or tuple(source_latent_shape) not in allowed_latent_shapes
        or target_latent_shape != source_latent_shape
    ):
        raise RealSmokeAuditError(
            "materialized latent geometry differs from the training config"
        )
    expected_run_geometry = {
        "raw_video_num_frames": config.data.expected_raw_num_frames,
        "raw_video_fps": config.data.expected_raw_fps,
        "materialized_video_num_frames": config.data.video_num_frames,
        "materialized_video_fps": config.data.video_fps,
        "temporal_mode": config.data.temporal_mode,
        "spatial_profile": config.data.spatial_profile,
        "expected_latent_frames": config.data.expected_latent_shape[1],
        "allowed_latent_hw": [
            list(shape[2:]) for shape in config.data.expected_latent_shapes
        ],
        "context_padding_mode": config.model.context_padding_mode,
        "context_budget_tokens": config.model.max_context_len,
        "official_context_padding_is_unmasked": True,
        "temporal_smoke_only": config.data.smoke_only,
        "source_temporal_compression_allowed": False,
        "source_truncation_allowed": False,
        "mask_or_tube_inputs": False,
    }
    for field, expected_value in expected_run_geometry.items():
        if run.get(field) != expected_value:
            raise RealSmokeAuditError(f"run {field} differs from config/contract")
    config_digest = _digest(run.get("config_sha256"), name="run config")
    config_path = _regular_file(
        _recorded_absolute_path(run.get("config"), name="run.config"),
        name="recorded action config",
    )
    if _sha256(config_path) != config_digest:
        raise RealSmokeAuditError("recorded action config digest differs")
    if done.get("config_sha256") != config_digest:
        raise RealSmokeAuditError("done config digest differs from run")
    encoder_contract_digest = _digest(
        run.get("encoder_contract_sha256"), name="run encoder contract"
    )
    if materialization.get("encoder_contract_sha256") != encoder_contract_digest:
        raise RealSmokeAuditError(
            "materialization encoder contract differs from training run"
        )

    context_name = run.get("context_preflight")
    if (
        not isinstance(context_name, str)
        or not context_name
        or Path(context_name).name != context_name
    ):
        raise RealSmokeAuditError("unsafe context preflight file name")
    context_path = _regular_file(
        run_root / context_name, name="context preflight"
    )
    if _sha256(context_path) != _digest(
        run.get("context_preflight_sha256"), name="context preflight"
    ):
        raise RealSmokeAuditError("context preflight digest differs")
    try:
        context_rows = [
            json.loads(line)
            for line in context_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as error:
        raise RealSmokeAuditError(
            f"invalid context preflight JSONL: {error}"
        ) from error
    if run.get("context_rows_preflighted") != 1 or len(context_rows) != 1:
        raise RealSmokeAuditError("context preflight must contain the one real row")
    preflight_budget = _validate_source_budget(
        context_rows[0],
        config=config,
        source_latent_shape=source_latent_shape,
        expected_sample_id=expected_sample_id,
    )
    for field, expected_value in (
        ("row_index", 0),
        ("latent_shape", source_latent_shape),
        ("raw_video_frames", config.data.expected_raw_num_frames),
        ("materialized_video_frames", config.data.video_num_frames),
        ("materialized_video_fps", config.data.video_fps),
        ("latent_frames", config.data.expected_latent_shape[1]),
    ):
        if preflight_budget.get(field) != expected_value:
            raise RealSmokeAuditError(f"context preflight {field} differs")
    recorded_special_digest = run.get("special_tokens_sha256")
    if recorded_special_digest is not None:
        recorded_special_digest = _digest(
            recorded_special_digest, name="run special tokens"
        )
    special_tokens_path = checkpoint_dir / "special_tokens.pkl"
    if special_tokens_path.is_symlink():
        raise RealSmokeAuditError("special_tokens.pkl must not be a symlink")
    actual_special_digest = (
        _sha256(special_tokens_path) if special_tokens_path.is_file() else None
    )
    if actual_special_digest != recorded_special_digest:
        raise RealSmokeAuditError(
            "special_tokens.pkl presence/digest differs from run provenance"
        )
    if (
        not _test_only_allow_unpinned_checkpoint
        and actual_special_digest != OMNIVIDEO2_1_3B_SPECIAL_TOKENS_SHA256
    ):
        raise RealSmokeAuditError(
            "special_tokens.pkl digest differs from the pinned checkpoint contract"
        )
    if config.model.require_special_tokens and actual_special_digest is None:
        raise RealSmokeAuditError("configured required special_tokens.pkl is missing")
    try:
        loader = special_token_loader or load_special_tokens
        _special_tokens, actual_special_rows, loaded_special_digest = (
            loader(
                checkpoint_dir,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                required=config.model.require_special_tokens,
            )
        )
    except Exception as error:
        raise RealSmokeAuditError(
            f"cannot validate official special-token tensors: {error}"
        ) from error
    if loaded_special_digest != actual_special_digest:
        raise RealSmokeAuditError("special-token loader/file digest differs")
    if actual_special_rows != config.model.expected_special_token_rows:
        raise RealSmokeAuditError(
            "official special-token row count differs from validated config: "
            f"actual={actual_special_rows}, "
            f"expected={config.model.expected_special_token_rows}"
        )
    if run.get("special_token_rows") != actual_special_rows:
        raise RealSmokeAuditError("run special-token row count differs")
    if run.get("dataset_rows") != 1 or run.get("manifest_sha256") != _sha256(
        manifest_path
    ):
        raise RealSmokeAuditError("training did not bind the one-row manifest")
    if done.get("manifest_sha256") != run.get("manifest_sha256"):
        raise RealSmokeAuditError("done manifest digest differs from run")
    if done.get("optimizer_steps") != 1 or done.get("complete") is not True:
        raise RealSmokeAuditError("training did not complete one optimizer step")
    if done.get("temporal_smoke_only") is not config.data.smoke_only:
        raise RealSmokeAuditError("done temporal smoke boundary differs")
    if done.get("observed_task_types") != ["action_edit"]:
        raise RealSmokeAuditError("real smoke observed a non-action task")

    metric_lines = [
        line
        for line in (run_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(metric_lines) != 1:
        raise RealSmokeAuditError("real smoke must have exactly one metric row")
    metric = json.loads(metric_lines[0])
    if not isinstance(metric, Mapping):
        raise RealSmokeAuditError("real smoke metric row must be an object")
    if metric.get("step") != 1:
        raise RealSmokeAuditError("real smoke metric optimizer step differs")
    loss = metric.get("loss")
    if not isinstance(loss, Mapping) or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in loss.values()
    ):
        raise RealSmokeAuditError("loss is incomplete or non-finite")
    gradients = metric.get("gradient_groups_rank0")
    if not isinstance(gradients, Mapping) or any(
        not isinstance(value, Mapping)
        or not isinstance(value.get("l2_norm"), (int, float))
        or float(value["l2_norm"]) <= 0.0
        for value in gradients.values()
    ):
        raise RealSmokeAuditError("a trainable gradient group is zero or absent")
    budgets = metric.get("source_budgets_rank0")
    if not isinstance(budgets, list) or len(budgets) != 1:
        raise RealSmokeAuditError("source budget record differs")
    metric_budget = _validate_source_budget(
        budgets[0],
        config=config,
        source_latent_shape=source_latent_shape,
        expected_sample_id=expected_sample_id,
    )
    for field in (
        "source_shape",
        "nonvisual_tokens",
        "visual_tokens",
        "total_tokens",
        "budget_tokens",
        "fixed_budget_padding_tokens",
        "fits",
        "source_truncated",
        "sample_id",
        "task_type",
        "context_padding_mode",
        "effective_wan_text_len",
        "effective_padding_tokens",
    ):
        if metric_budget.get(field) != preflight_budget.get(field):
            raise RealSmokeAuditError(
                f"metric source budget {field} differs from preflight"
            )
    runtime_all_ranks = _validate_runtime_evidence(
        metric, done, expected_world_size=expected_world_size
    )
    task_records = metric.get("task_records_all_ranks")
    if not isinstance(task_records, list) or len(task_records) != expected_world_size:
        raise RealSmokeAuditError("distributed task records differ")
    if any(
        not isinstance(item, Mapping)
        or item.get("sample_id") != expected_sample_id
        or item.get("task_type") != "action_edit"
        or item.get("lora_gate") != 1.0
        or item.get("plan_gate") != 1.0
        for item in task_records
    ):
        raise RealSmokeAuditError("real sample task/gate routing differs")
    if sorted(item["rank"] for item in task_records) != list(
        range(expected_world_size)
    ):
        raise RealSmokeAuditError("distributed task records miss or duplicate ranks")

    expected_base_digest = _digest(
        run.get("base_checkpoint_sha256"), name="run base checkpoint"
    )
    if done.get("base_checkpoint_sha256") != expected_base_digest:
        raise RealSmokeAuditError("done base checkpoint digest differs from run")
    actual_base_path = _regular_file(
        checkpoint_dir / "transformer" / "pytorch_model.pt",
        name="official transformer checkpoint",
    )
    actual_base_digest = _sha256(actual_base_path)
    if actual_base_digest != expected_base_digest:
        raise RealSmokeAuditError(
            "official base checkpoint digest differs before model reconstruction"
        )
    if (
        not _test_only_allow_unpinned_checkpoint
        and actual_base_digest != OMNIVIDEO2_1_3B_TRANSFORMER_SHA256
    ):
        raise RealSmokeAuditError(
            "official transformer digest differs from the pinned checkpoint contract"
        )

    checkpoint_name = done.get("final_adapter_checkpoint")
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise RealSmokeAuditError("unsafe final checkpoint name")
    checkpoint_path = run_root / checkpoint_name
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise RealSmokeAuditError("final checkpoint is missing or a symlink")
    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha != _digest(
        done.get("final_adapter_sha256"), name="final adapter"
    ):
        raise RealSmokeAuditError("final adapter digest differs")
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise RealSmokeAuditError("PyTorch lacks safe checkpoint loading")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise RealSmokeAuditError(
            f"cannot safely load final adapter checkpoint: {error}"
        ) from error
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != ACTION_ADAPTER_CHECKPOINT_FIELDS
    ):
        raise RealSmokeAuditError("checkpoint closed schema differs")
    expected_checkpoint_metadata = {
        "format": "marp-omnivideo2-action-adapters-v2",
        "step": 1,
        "validated_config": validated_config,
        "config_sha256": config_digest,
        "manifest_sha256": run.get("manifest_sha256"),
        "base_checkpoint_sha256": expected_base_digest,
        "checkpoint_contract_id": OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
        "special_tokens_sha256": recorded_special_digest,
        "special_token_rows": actual_special_rows,
        "special_token_serialized_rows": (
            OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
        ),
        "special_token_layout": special_token_layout_record(),
        "encoder_contract_sha256": encoder_contract_digest,
        "world_size": expected_world_size,
        "preview_only": True,
        "temporal_smoke_only": config.data.smoke_only,
        "production_claim_forbidden": True,
        "source_revision": expected_source_revision,
        "source_archive_sha256": source_archive_digest,
        "target_motion_tokens_used_by_renderer": False,
        "base_weights_saved": False,
    }
    for field, expected_value in expected_checkpoint_metadata.items():
        if checkpoint.get(field) != expected_value:
            raise RealSmokeAuditError(
                f"final checkpoint {field} differs from run/done provenance"
            )
    if checkpoint.get("activation_contract") != action_activation_contract_record():
        raise RealSmokeAuditError("final checkpoint activation contract differs")
    for field in ("rank0_cpu_rng_state", "rank0_device_rng_state"):
        rng_state = checkpoint.get(field)
        if (
            not isinstance(rng_state, Tensor)
            or rng_state.device.type != "cpu"
            or rng_state.dtype != torch.uint8
            or rng_state.ndim != 1
            or rng_state.numel() == 0
        ):
            raise RealSmokeAuditError(f"final checkpoint {field} is invalid")
    for field, expected_value in (
        ("special_tokens_sha256", recorded_special_digest),
        ("special_token_rows", actual_special_rows),
        ("encoder_contract_sha256", encoder_contract_digest),
    ):
        if done.get(field) != expected_value:
            raise RealSmokeAuditError(f"done {field} differs from run")

    lora = _finite_state(checkpoint.get("lora_state_dict"), name="LoRA state")
    planner = _finite_state(
        checkpoint.get("motion_planner_state_dict"), name="planner state"
    )
    lora_b = [value for key, value in lora.items() if key.endswith(".lora_B.weight")]
    if not lora_b or any(not bool(torch.count_nonzero(value)) for value in lora_b):
        raise RealSmokeAuditError("not every LoRA-B tensor updated")
    checkpoint_modules = checkpoint.get("lora_modules")
    if (
        not isinstance(checkpoint_modules, list)
        or not checkpoint_modules
        or any(
            not isinstance(module_name, str) or not module_name
            for module_name in checkpoint_modules
        )
        or len(set(checkpoint_modules)) != len(checkpoint_modules)
        or done.get("lora_module_count") != len(checkpoint_modules)
    ):
        raise RealSmokeAuditError(
            "final checkpoint LoRA module list/count is invalid"
        )

    restored = reconstruct_and_strictly_reload_adapter(
        omnivideo_root=omnivideo_root,
        checkpoint_dir=checkpoint_dir,
        config=config,
        adapter_checkpoint=checkpoint,
        expected_base_checkpoint_sha256=expected_base_digest,
        model_loader=model_loader,
    )
    if restored.lora_tensor_count != len(lora):
        raise RealSmokeAuditError("strictly restored LoRA tensor count differs")
    if restored.planner_tensor_count != len(planner):
        raise RealSmokeAuditError("strictly restored planner tensor count differs")

    return {
        "status": "verified",
        "engineering_scope": "one real preview payload replicated over four ranks",
        "motion_editing_quality_tested": False,
        "sample_id": expected_sample_id,
        "world_size": expected_world_size,
        "distributed_sample_replication": True,
        "optimizer_steps": 1,
        "temporal_mode": config.data.temporal_mode,
        "spatial_profile": config.data.spatial_profile,
        "materialized_video_num_frames": config.data.video_num_frames,
        "materialized_video_fps": config.data.video_fps,
        "temporal_smoke_only": config.data.smoke_only,
        "context_budget_tokens": config.model.max_context_len,
        "source_visual_tokens": metric_budget["visual_tokens"],
        "source_nonvisual_tokens": metric_budget["nonvisual_tokens"],
        "source_total_context_tokens": metric_budget["total_tokens"],
        "fixed_budget_padding_tokens": metric_budget[
            "fixed_budget_padding_tokens"
        ],
        "source_latent_shape": materialized_verification.get(
            "source_latent_shape"
        ),
        "target_latent_shape": materialized_verification.get(
            "target_latent_shape"
        ),
        "source_uncompressed": True,
        "first_frame_exact": True,
        "mask_or_tube_inputs": False,
        "target_motion_tokens_used_by_renderer": False,
        "official_model_reconstructed": True,
        "adapter_strictly_reloaded": True,
        "runtime_all_ranks": runtime_all_ranks,
        "lora_injected_modules": list(restored.injected_modules),
        "lora_module_count": len(restored.injected_modules),
        "lora_tensor_count": len(lora),
        "lora_b_nonzero_tensor_count": len(lora_b),
        "planner_tensor_count": len(planner),
        "checkpoint_sha256": checkpoint_sha,
        "materialization_sha256": _sha256(materialization_path),
        "manifest_sha256": _sha256(manifest_path),
        "base_checkpoint_sha256": restored.base_checkpoint_sha256,
        "checkpoint_contract_id": OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
        "config_sha256": config_digest,
        "special_tokens_sha256": recorded_special_digest,
        "special_token_rows": actual_special_rows,
        "special_token_serialized_rows": (
            OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
        ),
        "special_token_layout": special_token_layout_record(),
        "encoder_contract_sha256": encoder_contract_digest,
        "source_revision": expected_source_revision,
        "source_archive_sha256": source_archive_digest,
    }


def main() -> None:
    args = parse_args()
    result = verify(
        args.materialized_root,
        args.run_output_dir,
        args.omnivideo_root,
        args.checkpoint_dir,
        expected_sample_id=args.expected_sample_id,
        expected_world_size=args.expected_world_size,
        expected_source_revision=args.expected_source_revision,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
