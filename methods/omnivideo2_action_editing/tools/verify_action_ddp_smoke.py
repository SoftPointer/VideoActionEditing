#!/usr/bin/env python3
"""Fail-closed audit of a completed MARP four-rank synthetic smoke."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from action import validate_action_config  # noqa: E402
from action.checkpoint_contract import (  # noqa: E402
    ACTION_ADAPTER_CHECKPOINT_FIELDS,
    OMNIVIDEO2_1_3B_ACTIVE_SPECIAL_TOKEN_ROWS,
    OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
    OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS,
    OMNIVIDEO2_1_3B_SPECIAL_TOKENS_SHA256,
    OMNIVIDEO2_1_3B_TRANSFORMER_SHA256,
    action_activation_contract_record,
    special_token_layout_record,
)


SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_RE = re.compile(r"[0-9a-f]{40}")


class SmokeAuditError(RuntimeError):
    pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--expected-source-revision", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeAuditError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SmokeAuditError(f"JSON root is not an object: {path}")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SmokeAuditError(f"{name} is not a lowercase SHA-256")
    return value


def _finite_state(value: Any, name: str) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise SmokeAuditError(f"{name} must be a non-empty state mapping")
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, Tensor):
            raise SmokeAuditError(f"{name} contains a non-tensor entry")
        if tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()):
            raise SmokeAuditError(f"{name}.{key} is not finite CPU state")
    return value


def verify(
    root: Path, *, expected_world_size: int, expected_source_revision: str
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if expected_world_size <= 1:
        raise SmokeAuditError("expected_world_size must prove multi-rank execution")
    if GIT_RE.fullmatch(expected_source_revision) is None:
        raise SmokeAuditError("expected source revision must be a Git SHA")
    run = _json(root / "run.json")
    done = _json(root / "done.json")
    if run.get("format") != "marp-omnivideo2-action-run-v2":
        raise SmokeAuditError("run format differs")
    if done.get("format") != "marp-omnivideo2-action-training-done-v2":
        raise SmokeAuditError("done format differs")
    for value, name in ((run, "run"), (done, "done")):
        if value.get("world_size") != expected_world_size:
            raise SmokeAuditError(f"{name} does not prove expected world size")
        if value.get("preview_only") is not True:
            raise SmokeAuditError(f"{name} lost preview-only status")
        if value.get("temporal_smoke_only") is not True:
            raise SmokeAuditError(f"{name} lost synthetic-smoke status")
        if value.get("production_claim_forbidden") is not True:
            raise SmokeAuditError(f"{name} permits a production claim")
        if value.get("target_motion_tokens_used_by_renderer") is not False:
            raise SmokeAuditError(f"{name} indicates target-token leakage")
        if (
            value.get("checkpoint_contract_id")
            != OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID
            or value.get("special_token_serialized_rows")
            != OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
            or value.get("special_token_layout") != special_token_layout_record()
        ):
            raise SmokeAuditError(f"{name} checkpoint contract differs")
        if value.get("base_weights_saved") is not False and name == "run":
            raise SmokeAuditError("run claims base weights were saved")
    if run.get("mask_or_tube_inputs") is not False or done.get(
        "mask_or_tube_inputs"
    ) is not False:
        raise SmokeAuditError("run is not explicitly mask/tube-free")
    if run.get("source_revision") != expected_source_revision:
        raise SmokeAuditError("run source revision differs")
    if done.get("source_revision") != expected_source_revision:
        raise SmokeAuditError("done source revision differs")
    source_archive_digest = _digest(
        run.get("source_archive_sha256"), "run.source_archive"
    )
    if done.get("source_archive_sha256") != source_archive_digest:
        raise SmokeAuditError("run/done source archive differs")
    if done.get("complete") is not True or done.get("optimizer_steps") != 1:
        raise SmokeAuditError("one-step completion receipt differs")
    expected_tasks = {
        "action_edit",
        "identity_reconstruction",
        "native_replay",
        "native_isolation_probe",
    }
    if set(done.get("observed_task_types", [])) != expected_tasks:
        raise SmokeAuditError("completion receipt did not observe every task type")

    raw_config = run.get("validated_config")
    if not isinstance(raw_config, Mapping):
        raise SmokeAuditError("run.validated_config must be an object")
    try:
        config = validate_action_config(raw_config)
    except (TypeError, ValueError) as error:
        raise SmokeAuditError(f"run embeds an invalid action config: {error}") from error
    validated_config = config.to_dict()
    if dict(raw_config) != validated_config:
        raise SmokeAuditError("run validated config is not canonical")
    config_record = run.get("config")
    if not isinstance(config_record, str):
        raise SmokeAuditError("run config path is missing")
    recorded_config_path = Path(config_record).expanduser()
    config_path = recorded_config_path.resolve()
    if (
        config_path != (root / "validated_config.json").resolve()
        or not config_path.is_file()
        or recorded_config_path.is_symlink()
    ):
        raise SmokeAuditError("persistent config snapshot path differs")
    if _json(config_path) != validated_config:
        raise SmokeAuditError("persistent config snapshot differs from validated config")
    if _sha256(config_path) != _digest(run.get("config_sha256"), "run.config"):
        raise SmokeAuditError("persistent config snapshot digest differs")

    for field in (
        "config_sha256",
        "manifest_sha256",
        "base_checkpoint_sha256",
        "encoder_contract_sha256",
    ):
        if _digest(run.get(field), f"run.{field}") != _digest(
            done.get(field), f"done.{field}"
        ):
            raise SmokeAuditError(f"run/done {field} differs")
    if run.get("base_checkpoint_sha256") != OMNIVIDEO2_1_3B_TRANSFORMER_SHA256:
        raise SmokeAuditError("run base checkpoint differs from pinned contract")
    if _digest(run.get("special_tokens_sha256"), "run.special") != _digest(
        done.get("special_tokens_sha256"), "done.special"
    ):
        raise SmokeAuditError("run/done special-token identity differs")
    if run.get("special_tokens_sha256") != OMNIVIDEO2_1_3B_SPECIAL_TOKENS_SHA256:
        raise SmokeAuditError("run special tokens differ from pinned contract")
    special_token_rows = run.get("special_token_rows")
    if (
        special_token_rows != OMNIVIDEO2_1_3B_ACTIVE_SPECIAL_TOKEN_ROWS
        or done.get("special_token_rows") != special_token_rows
    ):
        raise SmokeAuditError("run/done special-token row count differs")

    checkpoint_name = done.get("final_adapter_checkpoint")
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise SmokeAuditError("final checkpoint name is unsafe")
    checkpoint_path = root / checkpoint_name
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise SmokeAuditError("final checkpoint is missing or a symlink")
    actual_checkpoint_digest = _sha256(checkpoint_path)
    if actual_checkpoint_digest != _digest(
        done.get("final_adapter_sha256"), "done.final_adapter_sha256"
    ):
        raise SmokeAuditError("final checkpoint digest differs")
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise SmokeAuditError("PyTorch lacks safe weights_only loading")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != ACTION_ADAPTER_CHECKPOINT_FIELDS
    ):
        raise SmokeAuditError("checkpoint closed schema differs")
    if checkpoint.get("format") != "marp-omnivideo2-action-adapters-v2":
        raise SmokeAuditError("adapter checkpoint format differs")
    if checkpoint.get("step") != 1 or checkpoint.get("world_size") != expected_world_size:
        raise SmokeAuditError("adapter step/world size differs")
    if checkpoint.get("source_revision") != expected_source_revision:
        raise SmokeAuditError("adapter source revision differs")
    if checkpoint.get("base_weights_saved") is not False:
        raise SmokeAuditError("adapter checkpoint claims base weights")
    if checkpoint.get("temporal_smoke_only") is not True:
        raise SmokeAuditError("adapter checkpoint lost synthetic-smoke status")
    if checkpoint.get("target_motion_tokens_used_by_renderer") is not False:
        raise SmokeAuditError("adapter checkpoint indicates target leakage")
    if checkpoint.get("special_token_rows") != special_token_rows:
        raise SmokeAuditError("adapter special-token row count differs")
    if (
        checkpoint.get("checkpoint_contract_id")
        != OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID
        or checkpoint.get("special_token_serialized_rows")
        != OMNIVIDEO2_1_3B_SERIALIZED_SPECIAL_TOKEN_ROWS
        or checkpoint.get("special_token_layout") != special_token_layout_record()
    ):
        raise SmokeAuditError("adapter checkpoint contract differs")
    if checkpoint.get("validated_config") != validated_config:
        raise SmokeAuditError("checkpoint validated config differs from run")
    for field in (
        "config_sha256",
        "manifest_sha256",
        "base_checkpoint_sha256",
        "special_tokens_sha256",
        "encoder_contract_sha256",
        "source_archive_sha256",
    ):
        if checkpoint.get(field) != run.get(field):
            raise SmokeAuditError(f"checkpoint {field} differs from run")
    lora = _finite_state(checkpoint.get("lora_state_dict"), "lora_state_dict")
    if any(
        not key.endswith((".lora_A.weight", ".lora_B.weight")) for key in lora
    ):
        raise SmokeAuditError("LoRA state contains a non-adapter key")
    planner = _finite_state(
        checkpoint.get("motion_planner_state_dict"), "motion_planner_state_dict"
    )
    if checkpoint.get("activation_contract") != action_activation_contract_record():
        raise SmokeAuditError("adapter activation contract differs")
    for field in ("rank0_cpu_rng_state", "rank0_device_rng_state"):
        rng_state = checkpoint.get(field)
        if (
            not isinstance(rng_state, Tensor)
            or rng_state.device.type != "cpu"
            or rng_state.dtype != torch.uint8
            or rng_state.ndim != 1
            or rng_state.numel() == 0
        ):
            raise SmokeAuditError(f"adapter {field} is invalid")

    try:
        metric_rows = [
            json.loads(line)
            for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeAuditError(f"cannot read metrics: {error}") from error
    if len(metric_rows) != 1 or metric_rows[0].get("step") != 1:
        raise SmokeAuditError("smoke must contain exactly one optimizer-step metric")
    loss = metric_rows[0].get("loss")
    if not isinstance(loss, Mapping) or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in loss.values()
    ):
        raise SmokeAuditError("metric loss is incomplete/non-finite")
    budgets = metric_rows[0].get("source_budgets_rank0")
    if not isinstance(budgets, list) or not budgets or any(
        not isinstance(item, Mapping) or item.get("first_frame_exact") is not True
        for item in budgets
    ):
        raise SmokeAuditError("source first-frame preservation was not recorded")
    task_records = metric_rows[0].get("task_records_all_ranks")
    if not isinstance(task_records, list) or len(task_records) != expected_world_size:
        raise SmokeAuditError("metrics do not contain one task record per rank")
    if {item.get("task_type") for item in task_records if isinstance(item, Mapping)} != expected_tasks:
        raise SmokeAuditError("not every fixture task type was exercised across ranks")
    for item in task_records:
        if not isinstance(item, Mapping):
            raise SmokeAuditError("invalid distributed task record")
        task_type = item.get("task_type")
        expected_lora_gate = 0.0 if task_type == "native_isolation_probe" else 1.0
        expected_plan_gate = 1.0 if task_type in {
            "action_edit",
            "identity_reconstruction",
        } else 0.0
        if item.get("lora_gate") != expected_lora_gate:
            raise SmokeAuditError("distributed task/LoRA-gate isolation differs")
        if item.get("plan_gate") != expected_plan_gate:
            raise SmokeAuditError("distributed task/plan-gate isolation differs")

    return {
        "status": "verified",
        "engineering_scope": "synthetic four-rank one-step only",
        "motion_editing_quality_tested": False,
        "world_size": expected_world_size,
        "optimizer_steps": 1,
        "checkpoint_contract_id": OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
        "special_token_rows": special_token_rows,
        "checkpoint_sha256": actual_checkpoint_digest,
        "lora_tensor_count": len(lora),
        "planner_tensor_count": len(planner),
        "preview_only": True,
        "mask_or_tube_inputs": False,
        "target_motion_tokens_used_by_renderer": False,
        "native_replay_lora_trained_without_plan": True,
        "native_isolation_probe_verified": True,
        "source_revision": expected_source_revision,
    }


def main() -> None:
    args = _args()
    result = verify(
        args.run_output_dir,
        expected_world_size=args.expected_world_size,
        expected_source_revision=args.expected_source_revision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
