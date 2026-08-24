#!/usr/bin/env python3
"""DDP trainer for mask-free MARP-Omni action editing.

The renderer always receives the complete, actor-agnostic source condition and
is supervised against the complete target rectified-flow endpoint.  Offline
target motion tokens are planner labels only: the renderer sees the planner's
source-conditioned prediction in both training and inference.

Run with one process per GPU, for example::

    torchrun --standalone --nproc_per_node=4 train_omnivideo2_action.py ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.utils.checkpoint  # noqa: F401 - required by upstream Wan
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from action import (
    ActionConfig,
    ActionLatentDataset,
    DiffSynthWanTrainingScheduler,
    TemporalMotionPlanPredictor,
    collate_action_latents,
    full_target_flow_loss,
    load_action_config,
    motion_plan_loss,
    prepare_full_target_flow,
)
from action.omni import (
    enable_action_lora,
    load_official_omnivideo2_1_3b,
    load_special_tokens,
    nonvisual_token_counts,
    require_full_source_context,
    set_exact_omni_context_length,
    set_action_lora_gate,
    sha256_file,
    wan_sequence_length,
)
from action.checkpoint_contract import (
    ACTION_ADAPTER_CHECKPOINT_FIELDS,
    OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
    OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS,
    OMNIVIDEO2_1_3B_TRANSFORMER_SHA256,
    action_activation_contract_record,
    special_token_layout_record,
)
from pact.lora import lora_state_dict


LOGGER = logging.getLogger("marp.train")
CHECKPOINT_FORMAT = "marp-omnivideo2-action-adapters-v2"
RUN_FORMAT = "marp-omnivideo2-action-run-v2"
DONE_FORMAT = "marp-omnivideo2-action-training-done-v2"
PLAN_ACTIVE_TASKS = frozenset({"action_edit", "identity_reconstruction"})
LORA_ACTIVE_TASKS = frozenset(
    {"action_edit", "identity_reconstruction", "native_replay"}
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, default=None)
    parser.add_argument("--omnivideo-root", type=Path, default=Path("../Omni-Video"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument(
        "--allow-preview-exploration",
        action="store_true",
        help=(
            "Permit preview_only payloads only when the closed config also opts in. "
            "This never changes their non-production status."
        ),
    )
    parser.add_argument("--dry-run-contract", action="store_true")
    parser.add_argument("--dry-run-samples", type=int, default=1)
    parser.add_argument(
        "--source-revision",
        default=os.environ.get("MARP_SOURCE_REVISION", "uncommitted-workspace"),
    )
    parser.add_argument(
        "--source-archive-sha256",
        default=os.environ.get("MARP_SOURCE_ARCHIVE_SHA256"),
    )
    return parser.parse_args()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {path}")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")


def _write_exclusive_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _distributed_context(backend: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise RuntimeError("invalid torchrun rank environment")
    if not torch.cuda.is_available():
        raise RuntimeError("OmniVideo action training requires a CUDA/ROCm GPU")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds visible devices={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        dist.init_process_group(backend=backend, init_method="env://")
        if dist.get_world_size() != world_size or dist.get_rank() != rank:
            raise RuntimeError("initialized distributed rank differs from environment")
    return rank, local_rank, world_size, device


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _begin_optimizer_window(
    cuda_api: Any,
    device: torch.device | str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Synchronize, reset per-window peaks, and start an isolated wall clock."""

    cuda_api.synchronize(device)
    cuda_api.reset_peak_memory_stats(device)
    started_at = float(clock())
    if not math.isfinite(started_at):
        raise RuntimeError("optimizer-window start clock is non-finite")
    return started_at


def _finish_optimizer_window(
    cuda_api: Any,
    device: torch.device | str,
    *,
    started_at: float,
    rank: int,
    optimizer_step: int,
    microbatches: int,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, int | float]:
    """Synchronize and capture one rank's isolated time and peak memory."""

    if not isinstance(rank, int) or rank < 0:
        raise ValueError("runtime rank must be a non-negative integer")
    if not isinstance(optimizer_step, int) or optimizer_step <= 0:
        raise ValueError("runtime optimizer_step must be positive")
    if not isinstance(microbatches, int) or microbatches <= 0:
        raise ValueError("runtime microbatches must be positive")
    if not isinstance(started_at, (int, float)) or not math.isfinite(started_at):
        raise ValueError("runtime started_at must be finite")
    cuda_api.synchronize(device)
    elapsed = float(clock()) - float(started_at)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("optimizer-window elapsed time is invalid")
    allocated = int(cuda_api.max_memory_allocated(device))
    reserved = int(cuda_api.max_memory_reserved(device))
    if allocated < 0 or reserved < 0:
        raise RuntimeError("optimizer-window peak memory is negative")
    return {
        "rank": rank,
        "optimizer_step": optimizer_step,
        "microbatches": microbatches,
        "isolated_optimizer_window_seconds": elapsed,
        "peak_memory_allocated_bytes": allocated,
        "peak_memory_reserved_bytes": reserved,
    }


def _validate_runtime_all_ranks(
    records: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    optimizer_step: int,
) -> list[dict[str, int | float]]:
    """Close and order one gathered runtime record per distributed rank."""

    if len(records) != world_size:
        raise RuntimeError(
            f"runtime gather expected {world_size} ranks, got {len(records)}"
        )
    expected_fields = {
        "rank",
        "optimizer_step",
        "microbatches",
        "isolated_optimizer_window_seconds",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
    }
    result: list[dict[str, int | float]] = []
    for raw in records:
        record = dict(raw)
        if set(record) != expected_fields:
            raise RuntimeError("runtime rank record fields differ")
        rank = record["rank"]
        if type(rank) is not int or not 0 <= rank < world_size:
            raise RuntimeError("runtime rank is invalid")
        if record["optimizer_step"] != optimizer_step:
            raise RuntimeError("runtime optimizer step differs across ranks")
        if type(record["microbatches"]) is not int or record["microbatches"] <= 0:
            raise RuntimeError("runtime microbatch count is invalid")
        for field in (
            "isolated_optimizer_window_seconds",
            "peak_memory_allocated_bytes",
            "peak_memory_reserved_bytes",
        ):
            value = record[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RuntimeError(f"runtime {field} is not numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise RuntimeError(f"runtime {field} is invalid")
        result.append(record)  # type: ignore[arg-type]
    result.sort(key=lambda item: int(item["rank"]))
    if [item["rank"] for item in result] != list(range(world_size)):
        raise RuntimeError("runtime gather has duplicate or missing ranks")
    return result


def _update_runtime_maxima(
    maxima: dict[int, dict[str, int | float]],
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Accumulate run-level per-rank maxima from one optimizer window."""

    for record in records:
        rank = int(record["rank"])
        current = maxima.setdefault(
            rank,
            {
                "rank": rank,
                "optimizer_windows": 0,
                "max_isolated_optimizer_window_seconds": 0.0,
                "max_peak_memory_allocated_bytes": 0,
                "max_peak_memory_reserved_bytes": 0,
            },
        )
        current["optimizer_windows"] = int(current["optimizer_windows"]) + 1
        current["max_isolated_optimizer_window_seconds"] = max(
            float(current["max_isolated_optimizer_window_seconds"]),
            float(record["isolated_optimizer_window_seconds"]),
        )
        current["max_peak_memory_allocated_bytes"] = max(
            int(current["max_peak_memory_allocated_bytes"]),
            int(record["peak_memory_allocated_bytes"]),
        )
        current["max_peak_memory_reserved_bytes"] = max(
            int(current["max_peak_memory_reserved_bytes"]),
            int(record["peak_memory_reserved_bytes"]),
        )


def _dataset(args: argparse.Namespace, config: ActionConfig) -> ActionLatentDataset:
    if args.allow_preview_exploration and not config.training.allow_preview:
        raise ValueError(
            "--allow-preview-exploration requires training.allow_preview=true in config"
        )
    allow_preview = bool(
        args.allow_preview_exploration and config.training.allow_preview
    )
    return ActionLatentDataset(
        args.manifest.expanduser().resolve(),
        payload_root=(
            args.payload_root.expanduser().resolve()
            if args.payload_root is not None
            else None
        ),
        expected_motion_tokens=config.planner.num_tokens,
        expected_latent_shapes=config.data.expected_latent_shapes,
        expected_data_config=config.data,
        allowed_task_types=config.training.allowed_task_types,
        allow_preview=allow_preview,
        verify_payload_digest=True,
    )


def _task_gate(task_types: Sequence[str], *, device: torch.device) -> Tensor:
    if not task_types:
        raise ValueError("task_types cannot be empty")
    values = []
    for task_type in task_types:
        if task_type in LORA_ACTIVE_TASKS:
            values.append(1.0)
        elif task_type == "native_isolation_probe":
            values.append(0.0)
        else:
            raise ValueError(f"unsupported action task type: {task_type!r}")
    return torch.tensor(values, device=device, dtype=torch.float32)


class MARPOmniTrainingModel(nn.Module):
    """Own the only renderer path, making teacher-token leakage impossible."""

    def __init__(
        self,
        omni: nn.Module,
        planner: TemporalMotionPlanPredictor,
        *,
        special_tokens: dict[str, Tensor] | None,
        special_token_count: int,
        max_context_len: int,
        require_uncompressed_source: bool,
        context_padding_mode: str,
        visual_patch_size: Sequence[int],
        wan_patch_size: Sequence[int],
        model_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.omni = omni
        self.planner = planner
        self.special_tokens = special_tokens
        self.special_token_count = int(special_token_count)
        self.max_context_len = int(max_context_len)
        self.require_uncompressed_source = bool(require_uncompressed_source)
        self.context_padding_mode = context_padding_mode
        self.visual_patch_size = tuple(visual_patch_size)
        self.wan_patch_size = tuple(wan_patch_size)
        self.model_dtype = model_dtype

    def forward(
        self,
        x_t: Tensor,
        timestep: Tensor,
        text_context: Sequence[Tensor],
        source_vlm_context: Sequence[Tensor],
        source_latent: Tensor,
        sample_ids: Sequence[str],
        task_types: Sequence[str],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[dict[str, Any]]]:
        batch_size = x_t.shape[0]
        if (
            len(text_context) != batch_size
            or len(source_vlm_context) != batch_size
            or len(task_types) != batch_size
            or len(sample_ids) != batch_size
            or source_latent.shape[0] != batch_size
        ):
            raise ValueError("conditioning batch sizes differ")

        lora_gate = _task_gate(task_types, device=x_t.device)
        plan_gate = torch.tensor(
            [float(task_type in PLAN_ACTIVE_TASKS) for task_type in task_types],
            device=x_t.device,
            dtype=torch.float32,
        )
        set_action_lora_gate(self.omni, lora_gate)
        source_fp32 = [item.to(device=x_t.device, dtype=torch.float32) for item in source_vlm_context]
        # The planner has no target input. It receives Qwen features that were
        # encoded offline from source video + edit instruction.
        predicted_plan = self.planner(source_fp32)

        text = [item.to(device=x_t.device, dtype=self.model_dtype) for item in text_context]
        source_vlm = [item.to(device=x_t.device, dtype=self.model_dtype) for item in source_vlm_context]
        counts = nonvisual_token_counts(
            text,
            source_vlm,
            motion_plan_tokens=self.planner.num_tokens,
            special_token_count=self.special_token_count,
        )
        renderer_vlm: list[Tensor] = []
        source_conditions: list[Tensor] = []
        budget_records: list[dict[str, Any]] = []
        for index in range(batch_size):
            plan_active = task_types[index] in PLAN_ACTIVE_TASKS
            if plan_active:
                renderer_vlm.append(
                    torch.cat(
                        (source_vlm[index], predicted_plan[index].to(self.model_dtype)),
                        dim=0,
                    )
                )
                nonvisual = counts[index]
            else:
                # Native replay trains the LoRA against ordinary Omni edits but
                # receives no motion-plan condition. The explicit isolation
                # probe additionally sets the LoRA gate to zero.
                renderer_vlm.append(source_vlm[index])
                nonvisual = counts[index] - self.planner.num_tokens
            budget = require_full_source_context(
                source_latent[index : index + 1],
                max_context_len=self.max_context_len,
                nonvisual_tokens=nonvisual,
                visual_patch_size=self.visual_patch_size,
                sample_id=sample_ids[index],
                task_type=task_types[index],
            )
            condition = source_latent[index : index + 1]
            source_conditions.append(condition.to(dtype=self.model_dtype))
            record = budget.to_dict()
            record.update(
                {
                    "sample_index": index,
                    "sample_id": sample_ids[index],
                    "task_type": task_types[index],
                    "context_padding_mode": self.context_padding_mode,
                    "effective_wan_text_len": (
                        budget.budget_tokens
                        if self.context_padding_mode == "fixed_budget"
                        else budget.total_tokens
                    ),
                    "effective_padding_tokens": (
                        budget.fixed_budget_padding_tokens
                        if self.context_padding_mode == "fixed_budget"
                        else 0
                    ),
                    # Backward-compatible receipt aliases.
                    "original_visual_tokens": budget.visual_tokens,
                    "output_visual_tokens": budget.visual_tokens,
                    "output_total_tokens": budget.total_tokens,
                    "compressed": False,
                    "first_frame_exact": bool(
                        torch.equal(condition[:, :, 0], source_latent[index : index + 1, :, 0])
                    ),
                }
            )
            budget_records.append(record)
        if self.context_padding_mode == "batch_exact":
            if batch_size != 1:
                raise RuntimeError("batch_exact context padding requires batch size one")
            set_exact_omni_context_length(
                self.omni,
                exact_context_len=budget_records[0]["total_tokens"],
                max_context_len=self.max_context_len,
            )
        elif self.context_padding_mode == "fixed_budget":
            wan_model = getattr(self.omni, "wan_model", None)
            if (
                wan_model is None
                or int(getattr(wan_model, "text_len", -1)) != self.max_context_len
            ):
                raise RuntimeError(
                    "official-compatible fixed context length differs from config"
                )
        else:
            raise RuntimeError(
                f"unsupported context padding mode: {self.context_padding_mode!r}"
            )
        sequence_length = wan_sequence_length(x_t, self.wan_patch_size)
        with torch.amp.autocast("cuda", dtype=self.model_dtype):
            outputs = self.omni(
                list(x_t.to(dtype=self.model_dtype).unbind(0)),
                t=timestep,
                context=text,
                ar_vision_input=renderer_vlm,
                visual_emb=source_conditions,
                seq_len=sequence_length,
                special_token_dict=self.special_tokens,
                classifier_free_ratio=0.0,
                condition_mode="full",
            )
        if not isinstance(outputs, list) or len(outputs) != batch_size:
            raise RuntimeError("OmniVideo returned an unexpected batch structure")
        prediction = torch.stack(outputs, dim=0).float()
        if prediction.shape != x_t.shape:
            raise RuntimeError(
                f"Omni output {tuple(prediction.shape)} differs from x_t {tuple(x_t.shape)}"
            )
        return prediction, predicted_plan, lora_gate, plan_gate, budget_records


def _motion_loss(
    predicted: Tensor, target: Tensor, plan_gate: Tensor
) -> Tensor:
    selected = plan_gate.bool()
    if bool(selected.any()):
        loss = motion_plan_loss(predicted[selected], target[selected])
    else:
        loss = predicted.sum() * 0.0
    # Keep every planner parameter connected on a rank containing only native
    # replay. DDP then has the same gradient structure on every rank.
    return loss + predicted.sum() * 0.0


def _trainable_groups(model: MARPOmniTrainingModel) -> dict[str, list[nn.Parameter]]:
    groups: dict[str, list[nn.Parameter]] = {"action_lora": [], "motion_planner": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("planner."):
            groups["motion_planner"].append(parameter)
        elif ".lora_A." in name or ".lora_B." in name:
            groups["action_lora"].append(parameter)
        else:
            raise RuntimeError(f"forbidden trainable parameter: {name}")
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"trainable parameter is not FP32: {name}={parameter.dtype}")
    if any(not values for values in groups.values()):
        raise RuntimeError(f"empty trainable group: {groups.keys()}")
    return groups


def _gradient_stats(
    groups: Mapping[str, Sequence[nn.Parameter]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name, parameters in groups.items():
        missing = sum(parameter.grad is None for parameter in parameters)
        if missing:
            raise RuntimeError(f"gradient group {name} has {missing} disconnected tensors")
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if any(not bool(torch.isfinite(value).all()) for value in gradients):
            raise FloatingPointError(f"gradient group {name} contains NaN/Inf")
        squared = torch.stack(
            [value.detach().float().square().sum() for value in gradients]
        ).sum()
        result[name] = {
            "parameter_tensors": len(parameters),
            "parameter_elements": sum(parameter.numel() for parameter in parameters),
            "l2_norm": float(torch.sqrt(squared)),
        }
    return result


def _cpu_state(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _save_checkpoint(
    path: Path,
    *,
    model: MARPOmniTrainingModel,
    config: ActionConfig,
    injected: Sequence[str],
    step: int,
    config_sha256: str,
    manifest_sha256: str,
    base_checkpoint_sha256: str,
    special_tokens_sha256: str | None,
    special_token_rows: int,
    encoder_contract_sha256: str,
    world_size: int,
    preview_only: bool,
    source_revision: str,
    source_archive_sha256: str | None,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    if special_token_rows != config.model.expected_special_token_rows:
        raise ValueError(
            "checkpoint special-token row count differs from validated config"
        )
    production_claim_forbidden = bool(preview_only or config.data.smoke_only)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": step,
        "validated_config": config.to_dict(),
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "checkpoint_contract_id": OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
        "special_tokens_sha256": special_tokens_sha256,
        "special_token_rows": special_token_rows,
        "special_token_serialized_rows": (
            OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
        ),
        "special_token_layout": special_token_layout_record(),
        "encoder_contract_sha256": encoder_contract_sha256,
        "world_size": world_size,
        "preview_only": preview_only,
        "temporal_smoke_only": config.data.smoke_only,
        "production_claim_forbidden": production_claim_forbidden,
        "source_revision": source_revision,
        "source_archive_sha256": source_archive_sha256,
        "activation_contract": action_activation_contract_record(),
        "target_motion_tokens_used_by_renderer": False,
        "base_weights_saved": False,
        "lora_modules": list(injected),
        "lora_state_dict": lora_state_dict(model.omni),
        "motion_planner_state_dict": _cpu_state(model.planner),
        "rank0_cpu_rng_state": torch.get_rng_state(),
        "rank0_device_rng_state": torch.cuda.get_rng_state(),
    }
    if set(payload) != ACTION_ADAPTER_CHECKPOINT_FIELDS:
        raise RuntimeError("internal adapter checkpoint schema differs from contract")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale checkpoint temporary exists: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _preflight_context_rows(
    dataset: ActionLatentDataset,
    config: ActionConfig,
    *,
    special_token_count: int,
) -> tuple[list[dict[str, Any]], str]:
    """Validate every row's shape and full-source context before model forward."""

    if special_token_count != config.model.expected_special_token_rows:
        raise RuntimeError(
            "official special-token row count differs from config: "
            f"actual={special_token_count}, "
            f"expected={config.model.expected_special_token_rows}"
        )
    records: list[dict[str, Any]] = []
    encoder_contract_digest: str | None = None
    for row_index in range(len(dataset)):
        sample = dataset[row_index]
        current_digest = sample["encoder_contract_sha256"]
        if encoder_contract_digest is None:
            encoder_contract_digest = current_digest
        elif current_digest != encoder_contract_digest:
            raise ValueError("manifest mixes offline encoder contracts")
        counts = nonvisual_token_counts(
            [sample["text_context"]],
            [sample["source_vlm_context"]],
            motion_plan_tokens=config.planner.num_tokens,
            special_token_count=special_token_count,
        )
        nonvisual = counts[0]
        if sample["task_type"] not in PLAN_ACTIVE_TASKS:
            nonvisual -= config.planner.num_tokens
        source = sample["source_latent"].unsqueeze(0)
        budget = require_full_source_context(
            source,
            max_context_len=config.model.max_context_len,
            nonvisual_tokens=nonvisual,
            visual_patch_size=config.model.visual_patch_size,
            sample_id=sample["sample_id"],
            task_type=sample["task_type"],
        )
        record = budget.to_dict()
        record.update(
            {
                "row_index": row_index,
                "sample_id": sample["sample_id"],
                "task_type": sample["task_type"],
                "latent_shape": list(sample["source_latent"].shape),
                "raw_video_frames": config.data.expected_raw_num_frames,
                "materialized_video_frames": config.data.video_num_frames,
                "materialized_video_fps": config.data.video_fps,
                "latent_frames": int(sample["source_latent"].shape[1]),
                "allowed_latent_hw": [
                    list(shape[2:]) for shape in config.data.expected_latent_shapes
                ],
                "context_padding_mode": config.model.context_padding_mode,
                "effective_wan_text_len": (
                    budget.budget_tokens
                    if config.model.context_padding_mode == "fixed_budget"
                    else budget.total_tokens
                ),
                "effective_padding_tokens": (
                    budget.fixed_budget_padding_tokens
                    if config.model.context_padding_mode == "fixed_budget"
                    else 0
                ),
                "source_truncated": False,
            }
        )
        records.append(record)
    if encoder_contract_digest is None:
        raise RuntimeError("context preflight selected no rows")
    return records, encoder_contract_digest


def _dry_run(
    args: argparse.Namespace, config: ActionConfig, dataset: ActionLatentDataset
) -> None:
    if args.dry_run_samples < 0:
        raise ValueError("--dry-run-samples must be non-negative")
    requested = len(dataset) if args.dry_run_samples == 0 else args.dry_run_samples
    count = min(len(dataset), requested)
    if count <= 0:
        raise ValueError("dry-run selected no payloads")
    context_records, _ = _preflight_context_rows(
        dataset,
        config,
        special_token_count=config.model.expected_special_token_rows,
    )
    scheduler = DiffSynthWanTrainingScheduler(
        shift=config.flow.shift,
        num_train_timesteps=config.flow.num_train_timesteps,
    )
    planner = TemporalMotionPlanPredictor(
        config.planner.num_tokens,
        input_dim=config.planner.input_dim,
        hidden_dim=config.planner.hidden_dim,
        depth=config.planner.depth,
    )
    summaries = []
    for index in range(count):
        batch = collate_action_latents([dataset[index]])
        source = batch["source_latent"].float()
        target = batch["target_latent"].float()
        flow = prepare_full_target_flow(
            target, scheduler.at(500, 1).sigma, noise=torch.zeros_like(target)
        )
        predicted_plan = planner(batch["source_vlm_context"])
        plan_loss = motion_plan_loss(predicted_plan, batch["target_motion_tokens"])
        ideal_velocity = full_target_flow_loss(flow.target_velocity, flow.target_velocity)
        if not bool(torch.isfinite(plan_loss)) or float(ideal_velocity) != 0.0:
            raise RuntimeError("contract dry-run produced invalid objective")
        summaries.append(
            {
                "sample_id": batch["sample_id"][0],
                "task_type": batch["task_type"][0],
                "preview_only": bool(batch["preview_only"][0]),
                "latent_shape": list(source.shape),
                "source_equals_target": bool(torch.equal(source, target)),
                "target_motion_shape": list(batch["target_motion_tokens"].shape),
                "planner_output_shape": list(predicted_plan.shape),
                "full_target_oracle_loss": float(ideal_velocity),
                "contains_spatial_control": False,
            }
        )
    print(
        json.dumps(
            {
                "status": "mask-free-action-contract-ok",
                "manifest_rows_digest_verified": len(dataset),
                "payloads_fully_validated": count,
                "context_rows_preflighted": len(context_records),
                "context_preflight": context_records,
                "context_special_token_rows_source": "closed_config_expected",
                "checkpoint_contract_id": OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
                "special_token_rows": config.model.expected_special_token_rows,
                "special_token_serialized_rows": (
                    OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
                ),
                "special_token_layout": special_token_layout_record(),
                "target_motion_tokens_used_by_renderer": False,
                "samples": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _train(
    args: argparse.Namespace, config: ActionConfig, dataset: ActionLatentDataset
) -> None:
    if args.checkpoint_dir is None or args.output_dir is None:
        raise ValueError("training requires --checkpoint-dir and --output-dir")
    if config.training.mixed_precision != "bf16":
        raise ValueError("official OmniVideo2-1.3B action training requires bf16")
    special_tokens_cpu, special_count, special_digest = load_special_tokens(
        args.checkpoint_dir,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        required=config.model.require_special_tokens,
    )
    context_preflight, encoder_contract_digest = _preflight_context_rows(
        dataset,
        config,
        special_token_count=special_count,
    )
    rank, local_rank, world_size, device = _distributed_context(args.dist_backend)
    try:
        seed = config.seed
        random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        torch.set_float32_matmul_precision("high")

        if world_size > 1:
            gathered: list[str | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, encoder_contract_digest)
            if len(set(gathered)) != 1:
                raise ValueError("distributed ranks observed different encoder contracts")

        omni, official, checkpoint = load_official_omnivideo2_1_3b(
            args.omnivideo_root,
            args.checkpoint_dir,
            max_context_len=config.model.max_context_len,
            visual_patch_size=config.model.visual_patch_size,
            wan_patch_size=config.model.wan_patch_size,
        )
        injected, _ = enable_action_lora(
            omni,
            scope=config.lora.scope,
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
        )
        omni.to(device)
        special_tokens = (
            {
                key: value.to(device=device, dtype=official.param_dtype)
                for key, value in special_tokens_cpu.items()
            }
            if special_tokens_cpu is not None
            else None
        )
        planner = TemporalMotionPlanPredictor(
            config.planner.num_tokens,
            input_dim=config.planner.input_dim,
            hidden_dim=config.planner.hidden_dim,
            depth=config.planner.depth,
        ).to(device=device, dtype=torch.float32)
        if config.model.gradient_checkpointing:
            omni.enable_gradient_checkpointing()
        train_model = MARPOmniTrainingModel(
            omni,
            planner,
            special_tokens=special_tokens,
            special_token_count=special_count,
            max_context_len=config.model.max_context_len,
            require_uncompressed_source=config.model.require_uncompressed_source,
            context_padding_mode=config.model.context_padding_mode,
            visual_patch_size=config.model.visual_patch_size,
            wan_patch_size=config.model.wan_patch_size,
            model_dtype=official.param_dtype,
        ).to(device)
        groups = _trainable_groups(train_model)
        trainable = groups["action_lora"] + groups["motion_planner"]
        train_model.train()
        distributed_model: nn.Module
        if world_size > 1:
            distributed_model = DistributedDataParallel(
                train_model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            distributed_model = train_model

        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=config.training.num_workers,
            collate_fn=collate_action_latents,
            pin_memory=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed + rank),
        )
        if len(loader) == 0:
            raise RuntimeError("training DataLoader is empty")
        accumulation = config.training.gradient_accumulation_steps
        steps_per_pass = (len(loader) + accumulation - 1) // accumulation
        target_steps = config.training.max_steps or steps_per_pass
        if target_steps <= 0:
            raise RuntimeError("training target optimizer steps is zero")

        optimizer = torch.optim.AdamW(
            trainable,
            lr=config.optimizer.learning_rate,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        optimizer.zero_grad(set_to_none=True)
        scheduler = DiffSynthWanTrainingScheduler(
            shift=config.flow.shift,
            num_train_timesteps=config.flow.num_train_timesteps,
        )
        flow_generator = torch.Generator(device="cpu").manual_seed(seed + 1009 * rank)
        noise_generator = torch.Generator(device=device).manual_seed(seed + 9176 * rank)

        config_path = args.config.expanduser().resolve()
        manifest_path = args.manifest.expanduser().resolve()
        config_digest = sha256_file(config_path)
        manifest_digest = sha256_file(manifest_path)
        checkpoint_digest = sha256_file(checkpoint)
        if checkpoint_digest != OMNIVIDEO2_1_3B_TRANSFORMER_SHA256:
            raise RuntimeError(
                "loaded transformer digest changed before provenance was written"
            )
        preview_count = sum(bool(row["preview_only"]) for row in dataset.rows)
        preview_only = preview_count > 0
        production_claim_forbidden = bool(
            preview_only or config.data.smoke_only
        )
        output_dir = args.output_dir.expanduser().resolve()
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=False)
            config_snapshot_path = output_dir / "validated_config.json"
            _write_exclusive_bytes(config_snapshot_path, config_path.read_bytes())
            if sha256_file(config_snapshot_path) != config_digest:
                raise RuntimeError("persistent config snapshot digest differs")
            context_preflight_path = output_dir / "context_preflight.jsonl"
            for record in context_preflight:
                _append_jsonl(context_preflight_path, record)
            context_preflight_digest = sha256_file(context_preflight_path)
            _atomic_json(
                output_dir / "run.json",
                {
                    "format": RUN_FORMAT,
                    "validated_config": config.to_dict(),
                    "config": str(config_snapshot_path),
                    "source_config": str(config_path),
                    "config_sha256": config_digest,
                    "manifest": str(manifest_path),
                    "manifest_sha256": manifest_digest,
                    "payload_root": str(dataset.payload_root),
                    "dataset_rows": len(dataset),
                    "preview_rows": preview_count,
                    "preview_only": preview_only,
                    "temporal_smoke_only": config.data.smoke_only,
                    "preview_override_cli": bool(args.allow_preview_exploration),
                    "production_claim_forbidden": production_claim_forbidden,
                    "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
                    "base_checkpoint_sha256": checkpoint_digest,
                    "checkpoint_contract_id": (
                        OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID
                    ),
                    "special_tokens_sha256": special_digest,
                    "special_token_rows": special_count,
                    "special_token_serialized_rows": (
                        OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
                    ),
                    "special_token_layout": special_token_layout_record(),
                    "encoder_contract_sha256": encoder_contract_digest,
                    "context_preflight": context_preflight_path.name,
                    "context_preflight_sha256": context_preflight_digest,
                    "context_rows_preflighted": len(context_preflight),
                    "raw_video_num_frames": config.data.expected_raw_num_frames,
                    "raw_video_fps": config.data.expected_raw_fps,
                    "materialized_video_num_frames": config.data.video_num_frames,
                    "materialized_video_fps": config.data.video_fps,
                    "temporal_mode": config.data.temporal_mode,
                    "spatial_profile": config.data.spatial_profile,
                    "expected_latent_frames": config.data.expected_latent_shape[1],
                    "allowed_latent_hw": [
                        list(shape[2:])
                        for shape in config.data.expected_latent_shapes
                    ],
                    "context_padding_mode": config.model.context_padding_mode,
                    "context_budget_tokens": config.model.max_context_len,
                    "official_context_padding_is_unmasked": True,
                    "omnivideo_root": str(args.omnivideo_root.expanduser().resolve()),
                    "world_size": world_size,
                    "distributed_backend": args.dist_backend if world_size > 1 else None,
                    "source_revision": args.source_revision,
                    "source_archive_sha256": args.source_archive_sha256,
                    "flow_master_dtype": "float32",
                    "trainable_master_dtype": "float32",
                    "base_model_dtype": "bfloat16",
                    "base_weights_saved": False,
                    "mask_or_tube_inputs": False,
                    "source_temporal_compression_allowed": False,
                    "source_truncation_allowed": False,
                    "target_motion_tokens_used_by_renderer": False,
                    "native_replay_action_gate": 1.0,
                    "native_isolation_probe_action_gate": 0.0,
                },
            )
        _barrier(world_size)

        metrics_path = output_dir / "metrics.jsonl"
        global_step = 0
        epoch = 0
        start_time = time.monotonic()
        stop = False
        observed_task_types: set[str] = set()
        runtime_maxima_by_rank: dict[int, dict[str, int | float]] = {}
        first_optimizer_step_runtime: list[dict[str, int | float]] | None = None
        while not stop:
            if sampler is not None:
                sampler.set_epoch(epoch)
            window_loss_sum = torch.zeros(3, device=device, dtype=torch.float32)
            window_microbatches = 0
            window_task_records: list[dict[str, Any]] = []
            window_budgets: list[dict[str, Any]] = []
            window_started_at: float | None = None
            for batch_index, batch in enumerate(loader):
                if batch["encoder_contract_sha256"] != encoder_contract_digest:
                    raise ValueError("manifest mixes offline encoder contracts")
                if window_microbatches == 0:
                    if window_started_at is not None:
                        raise RuntimeError("optimizer-window timer was not cleared")
                    window_started_at = _begin_optimizer_window(
                        torch.cuda, device
                    )
                source = batch["source_latent"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                target = batch["target_latent"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                target_motion = batch["target_motion_tokens"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                flow_sample = scheduler.sample(
                    target.shape[0],
                    generator=flow_generator,
                    device=device,
                    dtype=torch.float32,
                )
                prepared = prepare_full_target_flow(
                    target, flow_sample.sigma, generator=noise_generator
                )
                window_start = (batch_index // accumulation) * accumulation
                window_size = min(accumulation, len(loader) - window_start)
                end_window = (
                    (batch_index + 1) % accumulation == 0
                    or batch_index + 1 == len(loader)
                )
                sync_context = (
                    contextlib.nullcontext()
                    if end_window or not isinstance(distributed_model, DistributedDataParallel)
                    else distributed_model.no_sync()
                )
                with sync_context:
                    prediction, predicted_plan, lora_gate, plan_gate, budgets = distributed_model(
                        prepared.x_t,
                        flow_sample.timestep,
                        batch["text_context"],
                        batch["source_vlm_context"],
                        source,
                        batch["sample_id"],
                        batch["task_type"],
                    )
                    velocity_loss = full_target_flow_loss(
                        prediction,
                        prepared.target_velocity,
                        sample_weight=flow_sample.flow_weight,
                    )
                    plan_loss = _motion_loss(predicted_plan, target_motion, plan_gate)
                    total_loss = velocity_loss + config.planner.weight * plan_loss
                    if not bool(torch.isfinite(total_loss).detach()):
                        raise FloatingPointError(
                            f"non-finite loss at epoch={epoch} batch={batch_index}"
                        )
                    (total_loss / window_size).backward()
                window_loss_sum += torch.stack(
                    [
                        total_loss.detach().float(),
                        velocity_loss.detach().float(),
                        plan_loss.detach().float(),
                    ]
                )
                window_microbatches += 1
                window_task_records.extend(
                    {
                        "rank": rank,
                        "sample_id": sample_id,
                        "task_type": task_type,
                        "lora_gate": float(lora_value),
                        "plan_gate": float(plan_value),
                    }
                    for sample_id, task_type, lora_value, plan_value in zip(
                        batch["sample_id"], batch["task_type"], lora_gate, plan_gate
                    )
                )
                window_budgets.extend(budgets)
                if not end_window:
                    continue

                gradient_stats = _gradient_stats(groups)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if window_microbatches <= 0:
                    raise RuntimeError("empty optimizer accumulation window")
                if window_started_at is None:
                    raise RuntimeError("optimizer-window timer was not started")
                local_runtime = _finish_optimizer_window(
                    torch.cuda,
                    device,
                    started_at=window_started_at,
                    rank=rank,
                    optimizer_step=global_step,
                    microbatches=window_microbatches,
                )
                if world_size > 1:
                    gathered_runtime: list[dict[str, int | float] | None] = [
                        None for _ in range(world_size)
                    ]
                    dist.all_gather_object(gathered_runtime, local_runtime)
                    if any(item is None for item in gathered_runtime):
                        raise RuntimeError("runtime gather returned an empty rank")
                    runtime_all_ranks = _validate_runtime_all_ranks(
                        [item for item in gathered_runtime if item is not None],
                        world_size=world_size,
                        optimizer_step=global_step,
                    )
                else:
                    runtime_all_ranks = _validate_runtime_all_ranks(
                        [local_runtime],
                        world_size=1,
                        optimizer_step=global_step,
                    )
                reduced = window_loss_sum / window_microbatches
                if world_size > 1:
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                    reduced /= world_size
                local_task_records = list(window_task_records)
                if world_size > 1:
                    rank_task_records: list[list[dict[str, Any]] | None] = [
                        None for _ in range(world_size)
                    ]
                    dist.all_gather_object(rank_task_records, local_task_records)
                else:
                    rank_task_records = [local_task_records]
                if rank == 0:
                    _update_runtime_maxima(
                        runtime_maxima_by_rank, runtime_all_ranks
                    )
                    if global_step == 1:
                        first_optimizer_step_runtime = [
                            dict(item) for item in runtime_all_ranks
                        ]
                    flat_task_records = [
                        item
                        for rank_items in rank_task_records
                        if rank_items is not None
                        for item in rank_items
                    ]
                    observed_task_types.update(
                        item["task_type"] for item in flat_task_records
                    )
                    record = {
                        "step": global_step,
                        "epoch": epoch,
                        "batch": batch_index,
                        "sample_ids_rank0": batch["sample_id"],
                        "task_types_rank0": batch["task_type"],
                        "preview_rank0": [bool(item) for item in batch["preview_only"]],
                        "loss": {
                            "total_world_mean": float(reduced[0]),
                            "velocity_world_mean": float(reduced[1]),
                            "motion_plan_world_mean": float(reduced[2]),
                        },
                        "flow": {
                            "timestep_id_rank0": flow_sample.timestep_id,
                            "timestep_mean_rank0": float(flow_sample.timestep.mean()),
                            "sigma_mean_rank0": float(flow_sample.sigma.mean()),
                            "bsmntw_weight_rank0": float(flow_sample.flow_weight),
                        },
                        "lora_gate_rank0": [float(item) for item in lora_gate],
                        "plan_gate_rank0": [float(item) for item in plan_gate],
                        "task_records_all_ranks": flat_task_records,
                        "source_budgets_rank0": list(window_budgets),
                        "runtime_all_ranks": runtime_all_ranks,
                        "gradient_groups_rank0": gradient_stats,
                        "elapsed_seconds": time.monotonic() - start_time,
                    }
                    _append_jsonl(metrics_path, record)
                    if global_step % config.training.log_every == 0:
                        LOGGER.info(
                            "step=%d total=%.6f velocity=%.6f plan=%.6f",
                            global_step,
                            record["loss"]["total_world_mean"],
                            record["loss"]["velocity_world_mean"],
                            record["loss"]["motion_plan_world_mean"],
                        )
                    if global_step % config.training.save_every == 0:
                        _save_checkpoint(
                            output_dir / f"action_adapters_step_{global_step:08d}.pt",
                            model=train_model,
                            config=config,
                            injected=injected,
                            step=global_step,
                            config_sha256=config_digest,
                            manifest_sha256=manifest_digest,
                            base_checkpoint_sha256=checkpoint_digest,
                            special_tokens_sha256=special_digest,
                            special_token_rows=special_count,
                            encoder_contract_sha256=encoder_contract_digest,
                            world_size=world_size,
                            preview_only=preview_only,
                            source_revision=args.source_revision,
                            source_archive_sha256=args.source_archive_sha256,
                        )
                _barrier(world_size)
                window_loss_sum.zero_()
                window_microbatches = 0
                window_task_records.clear()
                window_budgets.clear()
                window_started_at = None
                if global_step >= target_steps:
                    stop = True
                    break
            epoch += 1

        if global_step == 0:
            raise RuntimeError("training produced zero optimizer steps")
        _barrier(world_size)
        if rank == 0:
            if (
                len(runtime_maxima_by_rank) != world_size
                or first_optimizer_step_runtime is None
            ):
                raise RuntimeError("run-level runtime evidence is incomplete")
            runtime_maxima_all_ranks = [
                runtime_maxima_by_rank[index] for index in range(world_size)
            ]
            rank0_runtime = runtime_maxima_by_rank[0]
            final_checkpoint = output_dir / f"action_adapters_final_step_{global_step:08d}.pt"
            _save_checkpoint(
                final_checkpoint,
                model=train_model,
                config=config,
                injected=injected,
                step=global_step,
                config_sha256=config_digest,
                manifest_sha256=manifest_digest,
                base_checkpoint_sha256=checkpoint_digest,
                special_tokens_sha256=special_digest,
                special_token_rows=special_count,
                encoder_contract_sha256=encoder_contract_digest,
                world_size=world_size,
                preview_only=preview_only,
                source_revision=args.source_revision,
                source_archive_sha256=args.source_archive_sha256,
            )
            _atomic_json(
                output_dir / "done.json",
                {
                    "format": DONE_FORMAT,
                    "complete": True,
                    "optimizer_steps": global_step,
                    "world_size": world_size,
                    "source_revision": args.source_revision,
                    "source_archive_sha256": args.source_archive_sha256,
                    "final_adapter_checkpoint": final_checkpoint.name,
                    "final_adapter_sha256": sha256_file(final_checkpoint),
                    "config_sha256": config_digest,
                    "manifest_sha256": manifest_digest,
                    "base_checkpoint_sha256": checkpoint_digest,
                    "checkpoint_contract_id": (
                        OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID
                    ),
                    "special_tokens_sha256": special_digest,
                    "special_token_rows": special_count,
                    "special_token_serialized_rows": (
                        OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
                    ),
                    "special_token_layout": special_token_layout_record(),
                    "encoder_contract_sha256": encoder_contract_digest,
                    "preview_only": preview_only,
                    "temporal_smoke_only": config.data.smoke_only,
                    "production_claim_forbidden": production_claim_forbidden,
                    "mask_or_tube_inputs": False,
                    "target_motion_tokens_used_by_renderer": False,
                    "observed_task_types": sorted(observed_task_types),
                    "lora_module_count": len(injected),
                    "trainable_action_lora_parameters": sum(
                        parameter.numel() for parameter in groups["action_lora"]
                    ),
                    "trainable_motion_planner_parameters": sum(
                        parameter.numel() for parameter in groups["motion_planner"]
                    ),
                    "elapsed_seconds": time.monotonic() - start_time,
                    "runtime_maxima_all_ranks": runtime_maxima_all_ranks,
                    "first_optimizer_step_runtime_all_ranks": (
                        first_optimizer_step_runtime
                    ),
                    "one_step_duration_seconds_all_ranks": (
                        {
                            str(item["rank"]): item[
                                "isolated_optimizer_window_seconds"
                            ]
                            for item in first_optimizer_step_runtime
                        }
                        if global_step == 1
                        else None
                    ),
                    "torch_version": torch.__version__,
                    "torch_hip_version": torch.version.hip,
                    "rank0_accelerator": torch.cuda.get_device_name(device),
                    "rank0_peak_memory_allocated_bytes": rank0_runtime[
                        "max_peak_memory_allocated_bytes"
                    ],
                    "rank0_peak_memory_reserved_bytes": rank0_runtime[
                        "max_peak_memory_reserved_bytes"
                    ],
                },
            )
        _barrier(world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    config = load_action_config(args.config)
    dataset = _dataset(args, config)
    if args.dry_run_contract:
        _dry_run(args, config, dataset)
        return
    _train(args, config, dataset)


if __name__ == "__main__":
    main()
