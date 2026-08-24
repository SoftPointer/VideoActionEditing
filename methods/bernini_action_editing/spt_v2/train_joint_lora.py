#!/usr/bin/env python3
"""Joint Bernini LoRA training through the SPT clean-latent executor.

This is the conservative A-first diagnostic arm.  A frozen, strictly loaded
``phase_query_v2`` student sees only the clean source video and the raw edit
instruction.  Its plan is executed on the trainable Bernini prediction before
the loss is evaluated.  A paired-latent oracle is constructed under
``no_grad`` and is used only to build a proxy target and loss selectors; it is
never teacher-forced into the student execution path.

The trainer deliberately forbids an ordinary full-target loss.  Synthetic
target appearance is visible only at conservative oracle ``generate`` cells,
and a hard preflight budget blocks the run before model construction if any
selected row assigns too much of the video to generation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


SPT_ROOT = Path(__file__).resolve().parent
METHOD_ROOT = SPT_ROOT.parent
for _root in (SPT_ROOT, METHOD_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import motion_residual as motion  # noqa: E402
import p3t  # noqa: E402
import train_delta_lora as delta  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_p3t_lora as p3t_train  # noqa: E402
import phase_query_planner as phase_query  # noqa: E402
import phase_transport as spt  # noqa: E402
import train_student as planner_train  # noqa: E402


RECEIPT_SCHEMA = "bernini-spt-v2-joint-lora-receipt-v1"
OPTIMIZER_SCHEMA = "bernini-spt-v2-joint-lora-optimizer-v1"
METHOD_NAME = "spt-v2-executor-joint-cross-qout-lora-diagnostic-v1"
LORA_SCOPE = "cross_q_out"
INTEGRATION_STEPS = 40
INTEGRATION_FLOW_SHIFT = 5.0
DEFAULT_MAX_ORACLE_GENERATE_FRACTION = 0.12
HARD_MAX_ORACLE_GENERATE_FRACTION = 0.12
TEACHER_FEATURE_CHANNELS = 64


class JointTrainingError(RuntimeError):
    """Raised before an unsafe or contract-divergent optimizer step."""


@dataclass(frozen=True)
class PlannerBundle:
    root: Path
    config: Mapping[str, Any]
    receipt: Mapping[str, Any]
    identity: Mapping[str, Any]


@dataclass(frozen=True)
class JointGeometry:
    batch: int
    phases: int
    height: int
    width: int
    channels: int
    target_tokens: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train fresh Bernini cross-q/out LoRA through frozen SPT-v2 planner"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--planner-checkpoint", required=True)
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, choices=(81,), default=81)
    parser.add_argument("--train-prefix-rows", type=int, default=None)
    parser.add_argument("--selected-membership", default=None)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--lora-scope", choices=(LORA_SCOPE,), default=LORA_SCOPE)
    parser.add_argument("--executor-loss-weight", type=float, default=1.0)
    parser.add_argument("--generate-loss-weight", type=float, default=0.25)
    parser.add_argument("--complement-loss-weight", type=float, default=0.25)
    parser.add_argument("--base-noop-replay-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--max-oracle-generate-fraction",
        type=float,
        default=DEFAULT_MAX_ORACLE_GENERATE_FRACTION,
    )
    parser.add_argument("--teacher-temperature", type=float, default=0.08)
    parser.add_argument("--teacher-generate-threshold", type=float, default=0.35)
    parser.add_argument("--teacher-transport-margin", type=float, default=0.05)
    parser.add_argument("--integration-steps", type=int, default=INTEGRATION_STEPS)
    parser.add_argument(
        "--integration-flow-shift", type=float, default=INTEGRATION_FLOW_SHIFT
    )
    parser.add_argument("--noop-instruction", default=motion.DEFAULT_NOOP_INSTRUCTION)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.num_frames != legacy.NUM_FRAMES:
        raise JointTrainingError("joint SPT training requires exact 81-frame pairs")
    if args.max_steps <= 0 or args.save_every < 0:
        raise JointTrainingError("max-steps must be positive and save-every non-negative")
    if args.train_prefix_rows is not None and (
        type(args.train_prefix_rows) is not int or args.train_prefix_rows <= 0
    ):
        raise JointTrainingError("train-prefix-rows must be a positive integer")
    if args.selected_membership is not None and (
        not isinstance(args.selected_membership, str)
        or not args.selected_membership.strip()
    ):
        raise JointTrainingError("selected-membership must be a non-empty path")
    if args.selected_membership is not None and args.train_prefix_rows is not None:
        raise JointTrainingError(
            "selected-membership and train-prefix-rows are mutually exclusive"
        )
    if args.lora_scope != LORA_SCOPE:
        raise JointTrainingError("the first joint diagnostic requires fresh cross_q_out LoRA")
    for name in ("learning_rate", "max_grad_norm"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise JointTrainingError(f"{name} must be finite and positive")
    for name in (
        "weight_decay",
        "executor_loss_weight",
        "generate_loss_weight",
        "complement_loss_weight",
        "base_noop_replay_loss_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise JointTrainingError(f"{name} must be finite and non-negative")
    if not any(
        float(getattr(args, name)) > 0.0
        for name in (
            "executor_loss_weight",
            "generate_loss_weight",
            "complement_loss_weight",
            "base_noop_replay_loss_weight",
        )
    ):
        raise JointTrainingError("joint recipe has no active loss")
    budget = float(args.max_oracle_generate_fraction)
    if (
        not math.isfinite(budget)
        or budget <= 0.0
        or budget > HARD_MAX_ORACLE_GENERATE_FRACTION
    ):
        raise JointTrainingError(
            "max-oracle-generate-fraction must lie in (0,0.12]; "
            "the conservative ceiling is not relaxable"
        )
    for name in (
        "teacher_temperature",
        "teacher_generate_threshold",
        "teacher_transport_margin",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise JointTrainingError(f"{name} must be finite and positive")
    if (
        args.integration_steps != INTEGRATION_STEPS
        or float(args.integration_flow_shift) != INTEGRATION_FLOW_SHIFT
    ):
        raise JointTrainingError("joint integration weighting is fixed to 40-step shift-5")
    if not isinstance(args.noop_instruction, str) or not args.noop_instruction.strip():
        raise JointTrainingError("noop instruction must be non-empty")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        if re.fullmatch(r"[0-9a-fA-F]{40}", str(getattr(args, name))) is None:
            raise JointTrainingError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))) is None:
            raise JointTrainingError(f"{name} must be a lowercase SHA-256")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise JointTrainingError("checkpoint identity differs from the audited 1.3B tree")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise JointTrainingError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise JointTrainingError(f"{label} must contain one JSON object")
    return value


def _validate_receipt_digest(receipt: Mapping[str, Any], *, label: str) -> None:
    candidate = dict(receipt)
    declared = candidate.pop("receipt_digest", None)
    if not isinstance(declared, str) or legacy.object_sha256(candidate) != declared:
        raise JointTrainingError(f"{label} receipt digest differs")


def inspect_planner_bundle(path: str | Path) -> PlannerBundle:
    """Validate checkpoint metadata without importing torch/safetensors."""

    try:
        root = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        raise JointTrainingError(f"planner checkpoint is unavailable: {error}") from error
    if not root.is_dir() or root.is_symlink():
        raise JointTrainingError("planner checkpoint must be a plain directory")
    config_path = root / "planner_config.json"
    receipt_path = root / "receipt.json"
    weights_path = root / "planner.safetensors"
    for candidate in (config_path, receipt_path, weights_path):
        if not candidate.is_file() or candidate.is_symlink():
            raise JointTrainingError(f"planner checkpoint lacks plain file {candidate.name}")
    config = _read_json(config_path, label="planner config")
    receipt = _read_json(receipt_path, label="planner receipt")
    _validate_receipt_digest(receipt, label="planner")
    if receipt.get("schema_version") != planner_train.RECEIPT_SCHEMA:
        raise JointTrainingError("planner receipt is not the current phase_query_v2 schema")
    if (
        config.get("architecture") != phase_query.ARCHITECTURE_NAME
        or receipt.get("planner", {}).get("architecture")
        != phase_query.ARCHITECTURE_NAME
    ):
        raise JointTrainingError("planner checkpoint is not phase_query_v2")
    immutable = receipt.get("immutable_contract")
    immutable_value = immutable.get("value") if isinstance(immutable, dict) else None
    if not isinstance(immutable_value, dict) or immutable_value.get("planner_config") != config:
        raise JointTrainingError("planner config is not hash-bound by its receipt")
    supervision = receipt.get("supervision")
    if not isinstance(supervision, dict):
        raise JointTrainingError("planner receipt lacks supervision contract")
    if supervision.get("student_api") != ["source", "instruction_tokens"]:
        raise JointTrainingError("planner student API differs")
    if supervision.get("student_target_argument_exists") is not False:
        raise JointTrainingError("planner receipt does not forbid target input")
    step = receipt.get("global_step")
    if type(step) is not int or step <= 0:
        raise JointTrainingError("planner checkpoint must contain a positive trained step")
    identity = {
        "path": str(root),
        "architecture": phase_query.ARCHITECTURE_NAME,
        "schema_version": receipt["schema_version"],
        "global_step": step,
        "planner_config_sha256": legacy.file_sha256(config_path),
        "planner_weights_sha256": legacy.file_sha256(weights_path),
        "planner_receipt_sha256": legacy.file_sha256(receipt_path),
        "planner_receipt_digest": receipt["receipt_digest"],
    }
    identity["identity_digest"] = legacy.object_sha256(identity)
    return PlannerBundle(root=root, config=config, receipt=receipt, identity=identity)


def validate_planner_dataset(
    bundle: PlannerBundle,
    *,
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    training_membership: Optional[Mapping[str, Any]] = None,
) -> None:
    recorded = bundle.receipt.get("dataset")
    if not isinstance(recorded, dict):
        raise JointTrainingError("planner receipt lacks dataset identity")
    recorded_summary = recorded.get("summary")
    if not isinstance(recorded_summary, dict):
        raise JointTrainingError("planner receipt lacks dataset summary identity")
    if (
        recorded.get("signature") != dataset.signature
        or recorded_summary.get("index_sha256") != dataset_summary.get("index_sha256")
        or int(recorded.get("full_dataset_rows", -1)) != len(dataset)
    ):
        raise JointTrainingError("planner was not trained against this exact dataset")
    if training_membership is not None and (
        recorded.get("training_membership_sha256")
        != training_membership.get("membership_sha256")
        or recorded.get("training_selection")
        != training_membership.get("selection")
    ):
        raise JointTrainingError(
            "planner and joint LoRA training memberships are not identical"
        )


def student_plan(planner: Any, source: Any, raw_instruction_tokens: Any) -> spt.PhasePlan:
    """Only legal student call; paired target cannot be passed by signature."""

    return planner(source, raw_instruction_tokens)


def oracle_generate_fraction(plan: spt.PhasePlan) -> float:
    value = float(plan.gate_probs[:, spt.GATE_GENERATE].float().mean().item())
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise JointTrainingError("oracle generate fraction is invalid")
    return value


def enforce_oracle_generate_budget(
    plan: spt.PhasePlan, *, maximum: float, iid: str
) -> float:
    fraction = oracle_generate_fraction(plan)
    if fraction > float(maximum):
        raise JointTrainingError(
            f"oracle generate budget exceeded for {iid}: "
            f"{fraction:.6f} > {float(maximum):.6f}; "
            "ordinary synthetic-target generate supervision is blocked"
        )
    return fraction


def validate_joint_geometry(
    *,
    source: Any,
    target: Any,
    student: spt.PhasePlan,
    oracle: spt.PhasePlan,
    action_batch: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> JointGeometry:
    """Fail closed on every SPT/Bernini packed-layout boundary."""

    spt._validate_video(source, label="source")
    spt._validate_video(target, label="target")
    if tuple(source.shape) != tuple(target.shape):
        raise JointTrainingError("source/target SPT geometry differs")
    student.validate(source)
    oracle.validate(source)
    batch, phases, height, width, channels = map(int, source.shape)
    if batch != 1 or phases != legacy.LATENT_FRAMES or channels != 64:
        raise JointTrainingError("joint trainer requires [1,21,H,W,64] clean videos")
    target_tokens = phases * height * width
    selector = action_batch["vae_latents_mask"].squeeze(0).bool()
    if int(selector.numel()) != 2 * target_tokens:
        raise JointTrainingError("Bernini selector length differs from SPT pair geometry")
    if bool(selector[:target_tokens].any()) or not bool(selector[target_tokens:].all()):
        raise JointTrainingError("Bernini pair is not exact source-then-target order")
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != (batch, target_tokens, channels):
            raise JointTrainingError(
                f"{name} must be [1,{target_tokens},64], got {tuple(tensor.shape)}"
            )
        if not bool(tensor.isfinite().all().item()):
            raise JointTrainingError(f"{name} contains non-finite values")
    return JointGeometry(batch, phases, height, width, channels, target_tokens)


def _weighted_cell_mse(prediction: Any, target: Any, weight: Any) -> Any:
    import torch

    if tuple(prediction.shape) != tuple(target.shape) or prediction.ndim != 3:
        raise JointTrainingError("weighted MSE predictions must share [B,N,D]")
    if tuple(weight.shape) != tuple(prediction.shape[:2]):
        raise JointTrainingError("weighted MSE selector must be [B,N]")
    per_cell = (prediction.float() - target.float()).pow(2).mean(dim=-1)
    denominator = weight.float().sum()
    if float(denominator.detach().item()) <= 0.0:
        return per_cell.sum() * 0.0
    return (per_cell * weight.float()).sum() / denominator


def class_balanced_executor_loss(
    prediction: Any,
    target: Any,
    oracle_gates: Any,
) -> tuple[Any, dict[str, Any]]:
    """Normalize preserve/transport/generate errors independently."""

    import torch

    if prediction.ndim != 3 or tuple(prediction.shape) != tuple(target.shape):
        raise JointTrainingError("executor predictions must share [B,N,D]")
    if oracle_gates.ndim != 5 or int(oracle_gates.shape[1]) != 3:
        raise JointTrainingError("oracle gates must be [B,3,T,H,W]")
    flat = oracle_gates.float().reshape(int(oracle_gates.shape[0]), 3, -1)
    if tuple(flat.shape[::2]) != (int(prediction.shape[0]), int(prediction.shape[1])):
        raise JointTrainingError("oracle gate token count differs from executor")
    names = ("preserve", "transport", "generate")
    values: dict[str, Any] = {}
    active = []
    counts: dict[str, Any] = {}
    for index, name in enumerate(names):
        weight = flat[:, index]
        counts[f"executor_{name}_mass"] = weight.sum()
        value = _weighted_cell_mse(prediction, target, weight)
        values[f"executor_{name}"] = value
        if float(weight.sum().detach().item()) > 0.0:
            active.append(value)
    if not active:
        raise JointTrainingError("oracle contains no active P/T/G class")
    total = torch.stack(active).mean()
    return total, {**values, **counts}


def compute_joint_loss(
    *,
    action_prediction: Any,
    action_target: Any,
    copy_prediction: Any,
    base_action: Any,
    base_noop: Any,
    executed_prediction: Any,
    oracle_proxy: Any,
    oracle_plan: spt.PhasePlan,
    sigma: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    """Compute only executor/proxy, gated-generate, and frozen-prior losses."""

    import torch

    executor, executor_parts = class_balanced_executor_loss(
        executed_prediction, oracle_proxy, oracle_plan.gate_probs
    )
    gates = oracle_plan.gate_probs.float().reshape(
        int(oracle_plan.gate_probs.shape[0]), 3, -1
    )
    generate = gates[:, spt.GATE_GENERATE]
    non_generate = 1.0 - generate
    generate_loss = _weighted_cell_mse(action_prediction, action_target, generate)
    student_complement = p3t.temporal_complement(
        action_prediction, latent_frames=legacy.LATENT_FRAMES
    )
    base_complement = p3t.temporal_complement(
        base_action, latent_frames=legacy.LATENT_FRAMES
    )
    complement = _weighted_cell_mse(
        student_complement, base_complement, non_generate
    )
    replay = torch.mean((copy_prediction.float() - base_noop.float()) ** 2)
    interval = p3t.interval_weight(
        sigma,
        steps=args.integration_steps,
        flow_shift=args.integration_flow_shift,
    ).mean()
    total = (
        interval
        * (
            args.executor_loss_weight * executor
            + args.generate_loss_weight * generate_loss
        )
        + args.complement_loss_weight * complement
        + args.base_noop_replay_loss_weight * replay
    )
    return total, {
        "executor": executor,
        **executor_parts,
        "oracle_generate_raw_target": generate_loss,
        "frozen_base_non_generate_complement": complement,
        "base_noop_replay": replay,
        "integration_interval_weight": interval,
        "oracle_generate_fraction": generate.mean(),
        "ordinary_full_target_loss": torch.zeros_like(total),
    }


def _iid(row: Mapping[str, Any]) -> str:
    value = row.get("iid", row.get("id"))
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise JointTrainingError("every joint-training row requires a stable IID")
    return value


def _teacher_config(args: argparse.Namespace) -> spt.PhaseTransportConfig:
    config = spt.PhaseTransportConfig(
        latent_channels=64,
        text_channels=4096,
        hidden_channels=128,
        teacher_temperature=args.teacher_temperature,
        teacher_generate_threshold=args.teacher_generate_threshold,
        teacher_transport_margin=args.teacher_transport_margin,
        teacher_require_cycle=True,
    )
    config.validate()
    return config


def preflight_oracle_budget(
    *,
    dataset: Any,
    training_membership: Mapping[str, Any],
    vae_mean: Any,
    vae_std: Any,
    z_dim: int,
    device: Any,
    teacher_config: spt.PhaseTransportConfig,
    maximum: float,
) -> dict[str, Any]:
    """Audit every selected row before constructing a trainable Bernini model."""

    import torch

    rows = int(training_membership["training_rows"])
    checked: list[dict[str, Any]] = []
    members = training_membership.get("members")
    if not isinstance(members, list) or len(members) != rows:
        raise JointTrainingError("preflight membership row count differs")
    for member in members:
        row_index = int(member["row_index"])
        raw_row = dataset[row_index]
        iid = _iid(raw_row)
        legacy.assert_identical_row(legacy.dataset_identity(raw_row, row_index))
        source, target = planner_train._clean_pair(
            raw_row, vae_mean, vae_std, z_dim, device
        )
        with torch.no_grad():
            oracle = spt.build_oracle_plan(
                source,
                target,
                teacher_config,
                feature_channels=TEACHER_FEATURE_CHANNELS,
            )
            fraction = enforce_oracle_generate_budget(
                oracle, maximum=maximum, iid=iid
            )
        checked.append(
            {
                "row_index": row_index,
                "iid": iid,
                "generate_fraction": fraction,
                "oracle_diagnostics": dict(oracle.diagnostics or {}),
            }
        )
        del source, target, oracle
    values = [entry["generate_fraction"] for entry in checked]
    audit: dict[str, Any] = {
        "status": "passed",
        "policy": "fail-before-model-construction-on-any-row",
        "maximum_allowed": float(maximum),
        "hard_maximum": HARD_MAX_ORACLE_GENERATE_FRACTION,
        "teacher_feature_channels": TEACHER_FEATURE_CHANNELS,
        "rows_checked": len(checked),
        "max_observed": max(values),
        "mean_observed": sum(values) / len(values),
        "rows": checked,
    }
    audit["audit_digest"] = legacy.object_sha256(audit)
    return audit


def load_frozen_planner(bundle: PlannerBundle, *, device: Any) -> Any:
    from safetensors.torch import load_file

    try:
        config = phase_query.PhaseQueryPlannerConfig(**dict(bundle.config))
        config.validate()
    except (TypeError, spt.PhaseTransportError) as error:
        raise JointTrainingError(f"planner config cannot instantiate exactly: {error}") from error
    planner = phase_query.PhaseQueryPlanner(config).to(device)
    saved = load_file(str(bundle.root / "planner.safetensors"), device=str(device))
    expected = set(planner.state_dict())
    if set(saved) != expected:
        raise JointTrainingError(
            f"planner state-key scope differs: delta={len(set(saved) ^ expected)}"
        )
    planner.load_state_dict(saved, strict=True)
    planner.requires_grad_(False)
    planner.eval()
    if any(parameter.requires_grad for parameter in planner.parameters()):
        raise JointTrainingError("planner did not remain frozen")
    return planner


def _method_hashes() -> dict[str, str]:
    paths = (
        SPT_ROOT / "train_joint_lora.py",
        SPT_ROOT / "phase_transport.py",
        SPT_ROOT / "phase_query_planner.py",
        METHOD_ROOT / "motion_residual.py",
        METHOD_ROOT / "p3t.py",
    )
    return {str(path.relative_to(METHOD_ROOT)): legacy.file_sha256(path) for path in paths}


def _immutable_contract(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    training_membership: Mapping[str, Any],
    planner_bundle: PlannerBundle,
    oracle_budget_audit: Mapping[str, Any],
    target_modules: Sequence[str],
) -> dict[str, Any]:
    value = {
        "method": METHOD_NAME,
        "method_source_revision": args.method_source_revision.lower(),
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "method_files_sha256": _method_hashes(),
        "bernini_commit": args.expected_bernini_commit.lower(),
        "veomni_commit": args.expected_veomni_commit.lower(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "dataset_signature": dataset.signature,
        "dataset_summary_sha256": dataset_summary["sha256"],
        "dataset_index_sha256": dataset_summary["index_sha256"],
        "training_membership": dict(training_membership),
        "planner_identity": dict(planner_bundle.identity),
        "planner_frozen": True,
        "student_semantic_inputs": ["clean_source_video", "raw_edit_instruction"],
        "student_target_argument": False,
        "student_executor_plan": "phase_query_v2",
        "oracle_use": "no_grad_proxy_and_loss_only",
        "oracle_teacher_forcing": False,
        "oracle_generate_budget": {
            "maximum_allowed": float(args.max_oracle_generate_fraction),
            "audit_digest": oracle_budget_audit["audit_digest"],
            "rows_checked": oracle_budget_audit["rows_checked"],
            "max_observed": oracle_budget_audit["max_observed"],
        },
        "teacher_feature_channels": TEACHER_FEATURE_CHANNELS,
        "teacher_temperature": float(args.teacher_temperature),
        "teacher_generate_threshold": float(args.teacher_generate_threshold),
        "teacher_transport_margin": float(args.teacher_transport_margin),
        "lora_scope": LORA_SCOPE,
        "lora_rank": legacy.LORA_RANK,
        "lora_alpha": legacy.LORA_ALPHA,
        "target_modules": list(target_modules),
        "fresh_lora_unless_resume": True,
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "loss": {
            "executor_class_balanced_ptg": float(args.executor_loss_weight),
            "oracle_generate_gated_raw_target": float(args.generate_loss_weight),
            "frozen_base_non_generate_complement": float(args.complement_loss_weight),
            "base_noop_replay": float(args.base_noop_replay_loss_weight),
            "ordinary_full_target_repaint": False,
        },
        "integration_steps": INTEGRATION_STEPS,
        "integration_flow_shift": INTEGRATION_FLOW_SHIFT,
        "paired_source_sigma_noise": True,
        "external_mask_track_pose_flow": False,
    }
    return {"value": value, "digest": legacy.object_sha256(value)}


def _build_receipt(
    *,
    args: argparse.Namespace,
    global_step: int,
    metrics: Optional[Mapping[str, float]],
    immutable: Mapping[str, Any],
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    training_membership: Mapping[str, Any],
    planner_bundle: PlannerBundle,
    oracle_budget_audit: Mapping[str, Any],
    target_modules: Sequence[str],
    named_trainable: Sequence[tuple[str, Any]],
    initialization_digest: str,
    distributed: Any,
    backend: str,
    bernini_revision: str,
    veomni_revision: str,
    transformers_version: str,
    resumed_from: Optional[str],
) -> dict[str, Any]:
    names = [name for name, _ in named_trainable]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "global_step": int(global_step),
        "max_steps": int(args.max_steps),
        "last_metrics": dict(metrics) if metrics is not None else None,
        "immutable_contract": dict(immutable),
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "planner": {
            "identity": dict(planner_bundle.identity),
            "frozen": True,
            "student_api": ["source", "raw_instruction_tokens"],
            "target_argument_exists": False,
        },
        "oracle_generate_budget_audit": dict(oracle_budget_audit),
        "dataset": {
            "path": str(dataset.root),
            "signature": dataset.signature,
            "rows": len(dataset),
            "summary": dict(dataset_summary),
            "training_membership": dict(training_membership),
        },
        "adapter": {
            "scope": LORA_SCOPE,
            "rank": legacy.LORA_RANK,
            "alpha": legacy.LORA_ALPHA,
            "target_modules": list(target_modules),
            "target_module_count": len(target_modules),
            "target_modules_sha256": legacy.object_sha256(list(target_modules)),
            "parameter_names": names,
            "parameter_names_sha256": legacy.object_sha256(names),
            "trainable_parameter_count": sum(
                int(parameter.numel()) for _, parameter in named_trainable
            ),
            "initialization_digest": initialization_digest,
        },
        "supervision": {
            "student_plan_inputs": ["clean_source_video", "raw_edit_instruction"],
            "student_plan_executed_in_velocity_path": True,
            "paired_oracle_visible_to_student": False,
            "paired_oracle_teacher_forced_into_executor": False,
            "paired_oracle_use": ["proxy_velocity", "loss_selectors"],
            "executor_ptg_class_balanced": True,
            "raw_target_supervision_scope": "oracle_generate_cells_only",
            "ordinary_full_target_loss": False,
            "frozen_base_non_generate_complement": True,
            "base_noop_replay": True,
            "integration_weighting": "nearest-40step-shift5-interval-width",
            "external_mask_track_pose_flow": False,
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "max_gradient_norm": float(args.max_grad_norm),
            "parameter_names": names,
        },
        "distributed": {
            "world_size": distributed.world_size,
            "ulysses_size": distributed.ulysses_size,
            "backend": backend,
            "same_row_all_ranks": True,
            "same_seed_all_ranks": True,
            "explicit_lora_gradient_all_reduce": distributed.world_size > 1,
        },
        "transformers_version": transformers_version,
        "resumed_from": resumed_from,
        "experimental_training": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = legacy.canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    output: Path,
    global_step: int,
    receipt: Mapping[str, Any],
    immutable: Mapping[str, Any],
    parameter_names: Sequence[str],
    planner_identity: Mapping[str, Any],
    rank: int,
) -> Path:
    import torch
    import torch.distributed as dist

    final = output / f"checkpoint-{global_step:08d}"
    if rank == 0:
        if final.exists():
            raise JointTrainingError(f"refusing to overwrite checkpoint: {final}")
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / f".{final.name}.tmp-{os.getpid()}"
        if temporary.exists():
            raise JointTrainingError(f"stale temporary checkpoint exists: {temporary}")
        temporary.mkdir()
        model.save_pretrained(temporary / "adapter", safe_serialization=True)
        torch.save(
            {
                "schema_version": OPTIMIZER_SCHEMA,
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "immutable_contract": dict(immutable),
                "parameter_names": list(parameter_names),
                "planner_identity_digest": planner_identity["identity_digest"],
            },
            temporary / "optimizer.pt",
        )
        _atomic_json(temporary / "planner_identity.json", planner_identity)
        _atomic_json(temporary / "receipt.json", receipt)
        os.replace(temporary, final)
        _atomic_json(
            output / "latest.json",
            {
                "checkpoint": str(final),
                "global_step": global_step,
                "receipt_digest": receipt["receipt_digest"],
                "planner_identity_digest": planner_identity["identity_digest"],
            },
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return final


def _validate_resume(
    receipt: Mapping[str, Any],
    *,
    immutable: Mapping[str, Any],
    planner_identity: Mapping[str, Any],
) -> int:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise JointTrainingError("resume receipt schema differs")
    _validate_receipt_digest(receipt, label="resume")
    if receipt.get("immutable_contract") != immutable:
        raise JointTrainingError("resume immutable contract differs")
    saved_identity = receipt.get("planner", {}).get("identity")
    if saved_identity != planner_identity:
        raise JointTrainingError("resume planner identity differs")
    step = receipt.get("global_step")
    if type(step) is not int or step < 0:
        raise JointTrainingError("resume global step is invalid")
    return step


def _pack_target_from_batch(batch: Mapping[str, Any]) -> Any:
    selector = batch["vae_latents_mask"].squeeze(0).bool()
    selected = batch["input_vae_latents"][selector]
    return motion.flatten_velocity_patches(selected.unsqueeze(0))


def _pack_velocity(value: Any) -> Any:
    return motion.flatten_velocity_patches(value.unsqueeze(0))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    planner_bundle = inspect_planner_bundle(args.planner_checkpoint)
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
    except legacy.TrainingContractError as error:
        raise JointTrainingError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % 4:
        raise JointTrainingError("1.3B attention heads must be divisible by Ulysses=4")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.training.data import NoiseScheduler, process_renderer_sample

    distributed = legacy.distributed_contract()
    device, backend = legacy.initialise_distributed(distributed)
    from bernini.parallel import init_parallel_state

    init_parallel_state(ulysses_size=distributed.ulysses_size)
    if distributed.world_size == 4 and distributed.ulysses_size != 4:
        raise JointTrainingError("four-rank joint DiT training requires Ulysses=4")
    legacy.seed_same_sample(args.seed)
    output = Path(args.output).expanduser().resolve()
    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    dataset_summary = legacy.validate_preprocessed_dataset_summary(
        args.dataset_summary,
        dataset,
        allow_incomplete=args.allow_incomplete_dataset,
    )
    training_membership = planner_train._training_membership(
        dataset,
        args.train_prefix_rows,
        selected_membership=args.selected_membership,
        dataset_summary=dataset_summary,
    )
    validate_planner_dataset(
        planner_bundle,
        dataset=dataset,
        dataset_summary=dataset_summary,
        training_membership=training_membership,
    )

    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    teacher_config = _teacher_config(args)
    # This audit intentionally happens before Bernini/LoRA construction.  The
    # currently observed high-generate teacher is therefore unable to take even
    # one optimizer step or silently reintroduce full synthetic repainting.
    oracle_budget_audit = preflight_oracle_budget(
        dataset=dataset,
        training_membership=training_membership,
        vae_mean=vae_mean,
        vae_std=vae_std,
        z_dim=z_dim,
        device=device,
        teacher_config=teacher_config,
        maximum=args.max_oracle_generate_fraction,
    )

    config_dir = bernini_root / "configs/bernini_renderer_wan21_1p3b"
    config = BerniniRendererConfig.from_pretrained(
        str(config_dir),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    base_model = BerniniRendererModel(config)
    base_model.requires_grad_(False)
    base_model.t5_text_encoder.eval()
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    available_modules = legacy.select_attention_projection_names(base_model)
    try:
        target_modules = motion.select_lora_scope(available_modules, LORA_SCOPE)
    except motion.MotionContractError as error:
        raise JointTrainingError(str(error)) from error
    if len(target_modules) != 60:
        raise JointTrainingError("cross_q_out must resolve to exactly 60 projections")
    immutable = _immutable_contract(
        args=args,
        checkpoint=checkpoint,
        dataset=dataset,
        dataset_summary=dataset_summary,
        training_membership=training_membership,
        planner_bundle=planner_bundle,
        oracle_budget_audit=oracle_budget_audit,
        target_modules=target_modules,
    )

    resume_paths = None
    resumed_from: Optional[str] = None
    if args.resume:
        try:
            resume_paths = p3t_train._resolve_adapter(
                args.resume, require_training_state=True
            )
        except p3t_train.DeltaTrainingError as error:
            raise JointTrainingError(str(error)) from error
        assert resume_paths.receipt is not None
        prior_receipt = _read_json(resume_paths.receipt, label="resume receipt")
        global_step = _validate_resume(
            prior_receipt,
            immutable=immutable,
            planner_identity=planner_bundle.identity,
        )
        identity_path = resume_paths.root / "planner_identity.json"
        if (
            not identity_path.is_file()
            or _read_json(identity_path, label="resume planner identity")
            != planner_bundle.identity
        ):
            raise JointTrainingError("resume planner identity sidecar differs")
        try:
            model = p3t_train._load_peft_adapter(
                base_model=base_model,
                adapter=resume_paths.adapter,
                target_modules=target_modules,
                trainable=True,
            )
        except p3t_train.DeltaTrainingError as error:
            raise JointTrainingError(str(error)) from error
        resumed_from = str(resume_paths.root)
    else:
        lora_config = LoraConfig(
            r=legacy.LORA_RANK,
            lora_alpha=legacy.LORA_ALPHA,
            lora_dropout=0.0,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(base_model, lora_config)
        global_step = 0
    if global_step > args.max_steps:
        raise JointTrainingError("resume step exceeds requested max-steps")

    model.to(device)
    model.train()
    renderer = model.get_base_model()
    renderer.t5_text_encoder.eval()
    planner = load_frozen_planner(planner_bundle, device=device)
    named_trainable = legacy.trainable_lora_parameters(model)
    initialization_digest = legacy.synchronize_trainable_parameters(
        named_trainable, source_rank=0
    )
    parameter_names = [name for name, _ in named_trainable]
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if resume_paths is not None:
        assert resume_paths.optimizer is not None
        try:
            optimizer_state = torch.load(
                resume_paths.optimizer, map_location="cpu", weights_only=False
            )
        except TypeError:
            optimizer_state = torch.load(resume_paths.optimizer, map_location="cpu")
        if (
            optimizer_state.get("schema_version") != OPTIMIZER_SCHEMA
            or optimizer_state.get("immutable_contract") != immutable
            or optimizer_state.get("parameter_names") != parameter_names
            or optimizer_state.get("planner_identity_digest")
            != planner_bundle.identity["identity_digest"]
            or int(optimizer_state.get("global_step", -1)) != global_step
        ):
            raise JointTrainingError("resume optimizer contract differs")
        optimizer.load_state_dict(optimizer_state["optimizer"])
        p3t_train._optimizer_to(optimizer, device)
        if any(
            float(group["lr"]) != float(args.learning_rate)
            or float(group["weight_decay"]) != float(args.weight_decay)
            for group in optimizer.param_groups
        ):
            raise JointTrainingError("restored optimizer hyperparameters differ")

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(
        128, (1, 2, 2), 1024, use_src_id_rotary_emb=True
    )
    scheduler = NoiseScheduler(**legacy.noise_scheduler_kwargs())

    last_metrics: Optional[dict[str, float]] = None
    last_saved = global_step if resume_paths is not None else -1
    training_rows = int(training_membership["training_rows"])
    while global_step < args.max_steps:
        row_index = int(
            training_membership["members"][global_step % training_rows]["row_index"]
        )
        raw_row = dataset[row_index]
        iid = _iid(raw_row)
        identity = legacy.dataset_identity(raw_row, row_index)
        legacy.assert_identical_row(identity)
        current_seed = legacy.step_seed(args.seed, global_step, row_index)
        legacy.seed_same_sample(current_seed)
        source, target = planner_train._clean_pair(
            raw_row, vae_mean, vae_std, z_dim, device
        )
        try:
            action_batch, copy_batch, auxiliary = delta._prepare_paired_batches(
                raw_row=raw_row,
                tokenizer=tokenizer,
                rope=rope,
                vae_mean=vae_mean,
                vae_std=vae_std,
                z_dim=z_dim,
                scheduler=scheduler,
                noop_instruction=args.noop_instruction,
                process_renderer_sample=process_renderer_sample,
            )
        except (legacy.TrainingContractError, motion.MotionContractError) as error:
            raise JointTrainingError(str(error)) from error
        action_batch = legacy._move_batch(action_batch, device)
        copy_batch = legacy._move_batch(copy_batch, device)
        auxiliary = legacy._move_batch(auxiliary, device)
        sigma = auxiliary["sigma"].float().reshape(-1)
        if sigma.numel() != 1 or float(sigma.item()) < 1.0e-4:
            raise JointTrainingError("SPT executor requires one sigma >= 1e-4")

        raw_instruction_tokens = planner_train._embed_instruction(
            renderer, action_batch, device
        )
        if int(raw_instruction_tokens.shape[-1]) != int(planner.config.text_channels):
            raise JointTrainingError("planner/T5 text channel width differs")
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            # Neither call exposes target to the student.
            predicted_plan = student_plan(
                planner, source, raw_instruction_tokens
            )
            oracle_plan = spt.build_oracle_plan(
                source,
                target,
                teacher_config,
                feature_channels=TEACHER_FEATURE_CHANNELS,
            )
            enforce_oracle_generate_budget(
                oracle_plan,
                maximum=args.max_oracle_generate_fraction,
                iid=iid,
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action_prediction = motion.renderer_velocity_prediction(
                renderer, action_batch
            ).float()
            copy_prediction = motion.renderer_velocity_prediction(
                renderer, copy_batch
            ).float()
        with torch.no_grad(), model.disable_adapter(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            base_action = motion.renderer_velocity_prediction(
                renderer, action_batch
            ).float()
            base_noop = motion.renderer_velocity_prediction(
                renderer, copy_batch
            ).float()

        source_packed = spt.video_to_packed(source).float()
        noisy_packed = _pack_target_from_batch(action_batch).float()
        action_target = _pack_velocity(action_batch["target_velocity"]).float()
        executed_prediction = spt.execute_packed_velocity(
            source_packed=source_packed,
            noisy_packed=noisy_packed,
            base_velocity_packed=action_prediction,
            sigma=sigma,
            height=int(source.shape[2]),
            width=int(source.shape[3]),
            plan=predicted_plan,
        ).float()
        with torch.no_grad():
            # Passing the true velocity only to the oracle executor recovers T
            # in its generate branch.  It never enters predicted_plan above.
            oracle_proxy = spt.execute_packed_velocity(
                source_packed=source_packed,
                noisy_packed=noisy_packed,
                base_velocity_packed=action_target,
                sigma=sigma,
                height=int(source.shape[2]),
                width=int(source.shape[3]),
                plan=oracle_plan,
            ).float()
        validate_joint_geometry(
            source=source,
            target=target,
            student=predicted_plan,
            oracle=oracle_plan,
            action_batch=action_batch,
            tensors={
                "source_packed": source_packed,
                "noisy_packed": noisy_packed,
                "action_prediction": action_prediction,
                "action_target": action_target,
                "copy_prediction": copy_prediction,
                "base_action": base_action,
                "base_noop": base_noop,
                "executed_prediction": executed_prediction,
                "oracle_proxy": oracle_proxy,
            },
        )
        loss, components = compute_joint_loss(
            action_prediction=action_prediction,
            action_target=action_target,
            copy_prediction=copy_prediction,
            base_action=base_action,
            base_noop=base_noop,
            executed_prediction=executed_prediction,
            oracle_proxy=oracle_proxy,
            oracle_plan=oracle_plan,
            sigma=sigma,
            args=args,
        )
        finite = bool(torch.isfinite(loss.detach()).item()) and all(
            bool(torch.isfinite(value.detach()).item())
            for value in components.values()
        )
        if not legacy._distributed_boolean(finite, op="all"):
            raise JointTrainingError(
                f"non-finite joint loss at optimizer step {global_step + 1}"
            )
        loss.backward()
        try:
            gradient_norm = legacy.all_reduce_lora_gradients(named_trainable)
        except legacy.TrainingContractError as error:
            raise JointTrainingError(str(error)) from error
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named_trainable], args.max_grad_norm
        )
        optimizer.step()
        global_step += 1
        last_metrics = {
            "total": float(loss.detach().item()),
            **{
                name: float(value.detach().item())
                for name, value in components.items()
            },
            "preclip_gradient_norm": float(gradient_norm),
            "student_preserve_fraction": float(
                predicted_plan.gate_probs[:, spt.GATE_PRESERVE].float().mean().item()
            ),
            "student_transport_fraction": float(
                predicted_plan.gate_probs[:, spt.GATE_TRANSPORT].float().mean().item()
            ),
            "student_generate_fraction": float(
                predicted_plan.gate_probs[:, spt.GATE_GENERATE].float().mean().item()
            ),
        }
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "step": global_step,
                        "row": row_index,
                        "iid": iid,
                        "seed": current_seed,
                        **last_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.save_every > 0 and global_step % args.save_every == 0:
            receipt = _build_receipt(
                args=args,
                global_step=global_step,
                metrics=last_metrics,
                immutable=immutable,
                dataset=dataset,
                dataset_summary=dataset_summary,
                training_membership=training_membership,
                planner_bundle=planner_bundle,
                oracle_budget_audit=oracle_budget_audit,
                target_modules=target_modules,
                named_trainable=named_trainable,
                initialization_digest=initialization_digest,
                distributed=distributed,
                backend=backend,
                bernini_revision=bernini_revision,
                veomni_revision=veomni_revision,
                transformers_version=transformers_version,
                resumed_from=resumed_from,
            )
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                output=output,
                global_step=global_step,
                receipt=receipt,
                immutable=immutable,
                parameter_names=parameter_names,
                planner_identity=planner_bundle.identity,
                rank=distributed.rank,
            )
            last_saved = global_step

    if last_saved != global_step:
        receipt = _build_receipt(
            args=args,
            global_step=global_step,
            metrics=last_metrics,
            immutable=immutable,
            dataset=dataset,
            dataset_summary=dataset_summary,
            training_membership=training_membership,
            planner_bundle=planner_bundle,
            oracle_budget_audit=oracle_budget_audit,
            target_modules=target_modules,
            named_trainable=named_trainable,
            initialization_digest=initialization_digest,
            distributed=distributed,
            backend=backend,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            transformers_version=transformers_version,
            resumed_from=resumed_from,
        )
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            output=output,
            global_step=global_step,
            receipt=receipt,
            immutable=immutable,
            parameter_names=parameter_names,
            planner_identity=planner_bundle.identity,
            rank=distributed.rank,
        )
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
