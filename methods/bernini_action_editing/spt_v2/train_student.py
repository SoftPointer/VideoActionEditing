#!/usr/bin/env python3
"""Train an SPT-v2 phase-query or grounded SPT-v3 planner on exact 81f pairs.

The paired target is visible only inside ``build_oracle_plan``.  The student
call is centralized in :func:`student_plan`, whose API has exactly two semantic
inputs: clean source tokens and the full unpadded instruction-token sequence.
Bernini is frozen and used only for its pinned tokenizer/T5 text encoder; no
DiT parameter is optimized in this planner-only stage.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
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
for root in (SPT_ROOT, METHOD_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import motion_residual as motion  # noqa: E402
import train_delta_lora as delta  # noqa: E402
import train_lora as legacy  # noqa: E402
import phase_transport as spt  # noqa: E402
import phase_query_planner as phase_query  # noqa: E402
import grounded_phase_planner as grounded_phase  # noqa: E402


RECEIPT_SCHEMA = "bernini-spt-v2-student-receipt-v3"
OPTIMIZER_SCHEMA = "bernini-spt-v2-student-optimizer-v3"
METHOD_NAME = "self-predicted-phase-transport-phase-query-student-v3"
GROUNDED_RECEIPT_SCHEMA = "bernini-spt-v3p1-routing-student-receipt-v1"
GROUNDED_OPTIMIZER_SCHEMA = "bernini-spt-v3p1-routing-student-optimizer-v1"
GROUNDED_METHOD_NAME = "grounded-cell-phase-transport-routing-v3p1-student-v1"
PLANNER_ULYSSES_SIZE = 1
OFFSET_HUBER_BETA = 1.0
CHANGE_POS_WEIGHT_MIN = 1.0
CHANGE_POS_WEIGHT_MAX = 4.0
COUNTERFACTUAL_CHANGE_MARGIN = 1.0


class StudentTrainingError(RuntimeError):
    """Raised before a planner optimizer step when a contract differs."""


def _grounded_initialization_contract() -> dict[str, Any]:
    """Bind the nonzero semantic routes while preserving safe output heads."""

    return {
        "semantic_cross_attention_residual_initial_scale": float(
            grounded_phase.SEMANTIC_RESIDUAL_INITIAL_SCALE
        ),
        "slot_text_residual_initial_scale": float(
            grounded_phase.SEMANTIC_RESIDUAL_INITIAL_SCALE
        ),
        "slot_source_residual_initial_scale": float(
            grounded_phase.SEMANTIC_RESIDUAL_INITIAL_SCALE
        ),
        "cell_text_residual_initial_scale": float(
            grounded_phase.SEMANTIC_RESIDUAL_INITIAL_SCALE
        ),
        "cell_slot_residual_initial_scale": float(
            grounded_phase.SEMANTIC_RESIDUAL_INITIAL_SCALE
        ),
        "temporal_residual_initial_scale": float(
            grounded_phase.TEMPORAL_RESIDUAL_INITIAL_SCALE
        ),
        "slot_self_residual_initial_scale": float(
            grounded_phase.SLOT_SELF_INITIAL_SCALE
        ),
        "semantic_cross_attention_shared_coarse_and_fine": True,
        "zero_fusion_and_safe_routing_head_initialization_unchanged": True,
        "zero_output_heads_may_delay_semantic_backbone_nonzero_gradient": True,
    }


def configure_rank_local_runtime_cache() -> bool:
    """Isolate writable ROCm/compiler caches for each torchrun local rank.

    MIOpen's SQLite tuning database is not safe to discover previously unseen
    Conv3d kernels from four processes in one shared directory.  The launcher
    supplies a node-local root; each child resolves its own rank only after
    torchrun has populated ``LOCAL_RANK``.
    """

    root_value = os.environ.get("BERNINI_SPT_RANK_CACHE_ROOT")
    local_rank = os.environ.get("LOCAL_RANK")
    if root_value is None and local_rank is None:
        return False
    if not root_value or local_rank is None or re.fullmatch(r"[0-9]+", local_rank) is None:
        raise StudentTrainingError("rank-local cache root/LOCAL_RANK contract differs")
    root = Path(root_value).expanduser().resolve() / f"rank-{int(local_rank)}"
    locations = {
        "MIOPEN_USER_DB_PATH": root / "miopen-user",
        "MIOPEN_CUSTOM_CACHE_DIR": root / "miopen-custom",
        "TORCH_EXTENSIONS_DIR": root / "torch-extensions",
        "TRITON_CACHE_DIR": root / "triton",
    }
    for name, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    return True


def _method_name(architecture: str) -> str:
    if architecture == phase_query.ARCHITECTURE_NAME:
        return METHOD_NAME
    if architecture == grounded_phase.ARCHITECTURE_NAME:
        return GROUNDED_METHOD_NAME
    raise StudentTrainingError(f"unsupported planner architecture: {architecture}")


def _receipt_schema(architecture: str) -> str:
    if architecture == phase_query.ARCHITECTURE_NAME:
        return RECEIPT_SCHEMA
    if architecture == grounded_phase.ARCHITECTURE_NAME:
        return GROUNDED_RECEIPT_SCHEMA
    raise StudentTrainingError(f"unsupported planner architecture: {architecture}")


def _optimizer_schema(architecture: str) -> str:
    if architecture == phase_query.ARCHITECTURE_NAME:
        return OPTIMIZER_SCHEMA
    if architecture == grounded_phase.ARCHITECTURE_NAME:
        return GROUNDED_OPTIMIZER_SCHEMA
    raise StudentTrainingError(f"unsupported planner architecture: {architecture}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an exact-81f Bernini SPT planner")
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, choices=(81,), default=81)
    parser.add_argument("--max-steps", type=int, default=644)
    parser.add_argument("--save-every", type=int, default=64)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--train-prefix-rows",
        type=int,
        default=None,
        help="diagnostic only: deterministically train on the first N rows; default uses all rows",
    )
    parser.add_argument(
        "--selected-membership",
        default=None,
        help=(
            "strict hash-bound teacher-trust membership JSON; mutually exclusive "
            "with --train-prefix-rows"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--planner-architecture",
        choices=(phase_query.ARCHITECTURE_NAME, grounded_phase.ARCHITECTURE_NAME),
        default=phase_query.ARCHITECTURE_NAME,
    )
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--match-channels", type=int, default=32)
    parser.add_argument("--edit-slots", type=int, default=8)
    parser.add_argument("--dense-query-chunk-size", type=int, default=4096)
    parser.add_argument("--gate-loss-weight", type=float, default=1.0)
    parser.add_argument("--conditional-gate-loss-weight", type=float, default=1.0)
    parser.add_argument("--gate-mass-loss-weight", type=float, default=0.05)
    parser.add_argument("--offset-loss-weight", type=float, default=0.25)
    parser.add_argument("--smooth-loss-weight", type=float, default=0.01)
    parser.add_argument("--noop-loss-weight", type=float, default=0.25)
    parser.add_argument("--change-tversky-weight", type=float, default=0.5)
    parser.add_argument("--phase-change-mass-weight", type=float, default=0.1)
    parser.add_argument("--phase-generate-mass-weight", type=float, default=0.1)
    parser.add_argument("--mid-change-loss-weight", type=float, default=0.25)
    parser.add_argument("--coarse-change-loss-weight", type=float, default=0.125)
    parser.add_argument("--expected-offset-loss-weight", type=float, default=0.1)
    parser.add_argument("--noop-generate-weight", type=float, default=0.2)
    parser.add_argument("--noop-offset-weight", type=float, default=0.25)
    parser.add_argument("--counterfactual-change-loss-weight", type=float, default=0.0)
    parser.add_argument("--change-polarization-loss-weight", type=float, default=0.0)
    parser.add_argument("--teacher-temperature", type=float, default=0.08)
    parser.add_argument("--teacher-generate-threshold", type=float, default=0.35)
    parser.add_argument("--teacher-feature-channels", type=int, default=64)
    parser.add_argument("--max-generate-fraction-per-phase", type=float, default=0.12)
    parser.add_argument("--noop-instruction", default=motion.DEFAULT_NOOP_INSTRUCTION)
    parser.add_argument("--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.num_frames != legacy.NUM_FRAMES:
        raise StudentTrainingError("only exact 81-frame training is supported")
    if args.max_steps <= 0 or args.save_every < 0:
        raise StudentTrainingError("max-steps must be positive and save-every non-negative")
    if args.train_prefix_rows is not None and (
        type(args.train_prefix_rows) is not int or args.train_prefix_rows <= 0
    ):
        raise StudentTrainingError("train-prefix-rows must be a positive integer")
    if args.selected_membership is not None and (
        not isinstance(args.selected_membership, str)
        or not args.selected_membership.strip()
    ):
        raise StudentTrainingError("selected-membership must be a non-empty path")
    if args.selected_membership is not None and args.train_prefix_rows is not None:
        raise StudentTrainingError(
            "selected-membership and train-prefix-rows are mutually exclusive"
        )
    if type(args.hidden_channels) is not int or args.hidden_channels <= 0:
        raise StudentTrainingError("hidden-channels must be a positive integer")
    if args.planner_architecture not in (
        phase_query.ARCHITECTURE_NAME,
        grounded_phase.ARCHITECTURE_NAME,
    ):
        raise StudentTrainingError(
            "planner-architecture is unsupported"
        )
    for name in (
        "attention_heads",
        "match_channels",
        "edit_slots",
        "dense_query_chunk_size",
    ):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise StudentTrainingError(
                f"{name.replace('_', '-')} must be a positive integer"
            )
    if args.hidden_channels % args.attention_heads:
        raise StudentTrainingError("hidden-channels must be divisible by attention-heads")
    if args.teacher_feature_channels != 64:
        raise StudentTrainingError(
            "teacher-feature-channels must be 64; reduced-channel oracle is diagnostic only"
        )
    if args.max_generate_fraction_per_phase != 0.12:
        raise StudentTrainingError(
            "max-generate-fraction-per-phase must be 0.12 on the main training path"
        )
    for name in ("learning_rate", "max_grad_norm", "teacher_temperature", "teacher_generate_threshold"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise StudentTrainingError(f"{name} must be finite and positive")
    for name in (
        "weight_decay",
        "gate_loss_weight",
        "conditional_gate_loss_weight",
        "gate_mass_loss_weight",
        "offset_loss_weight",
        "smooth_loss_weight",
        "noop_loss_weight",
        "change_tversky_weight",
        "phase_change_mass_weight",
        "phase_generate_mass_weight",
        "mid_change_loss_weight",
        "coarse_change_loss_weight",
        "expected_offset_loss_weight",
        "noop_generate_weight",
        "noop_offset_weight",
        "counterfactual_change_loss_weight",
        "change_polarization_loss_weight",
    ):
        value = float(
            getattr(
                args,
                name,
                0.0
                if name
                in (
                    "counterfactual_change_loss_weight",
                    "change_polarization_loss_weight",
                )
                else None,
            )
        )
        if not math.isfinite(value) or value < 0.0:
            raise StudentTrainingError(f"{name} must be finite and non-negative")
    if args.planner_architecture == grounded_phase.ARCHITECTURE_NAME:
        action_supervision_active = any(
            float(value) > 0.0
            for value in (
                args.gate_loss_weight,
                getattr(args, "counterfactual_change_loss_weight", 0.0),
                args.offset_loss_weight,
                args.expected_offset_loss_weight,
            )
        )
        if not action_supervision_active:
            raise StudentTrainingError(
                "grounded planner recipe has no active action routing supervision"
            )
    elif args.gate_loss_weight == args.offset_loss_weight == args.noop_loss_weight == 0.0:
        raise StudentTrainingError("planner recipe has no active teacher/no-op supervision")
    if not isinstance(args.noop_instruction, str) or not args.noop_instruction.strip():
        raise StudentTrainingError("noop instruction must be non-empty")
    for name in ("expected_bernini_commit", "expected_veomni_commit"):
        if re.fullmatch(r"[0-9a-fA-F]{40}", getattr(args, name)) is None:
            raise StudentTrainingError(f"{name} must be a full SHA-1")
    if re.fullmatch(r"[0-9a-f]{64}", args.expected_checkpoint_tree_sha256) is None:
        raise StudentTrainingError("expected checkpoint hash must be lowercase SHA-256")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise StudentTrainingError("checkpoint identity differs from audited Bernini-R 1.3B")


def student_plan(planner: Any, source: Any, instruction_tokens: Any) -> spt.PhasePlan:
    """The only student entry point; target data cannot be passed by signature."""

    return planner(source, instruction_tokens)


def _unpadded_text_tokens(text_lens: Any, text_embs: Any) -> Any:
    """Return Bernini's complete single-sample T5 sequence as ``[1,L,D]``.

    The pinned encoder can report its padded downstream width in
    ``text_lens``.  Padding is an exact-zero suffix; this function removes only
    that suffix and rejects an internal zero row.  No token pooling is allowed.
    """

    import torch

    if isinstance(text_embs, (list, tuple)):
        if len(text_embs) != 1:
            raise StudentTrainingError("planner batches require exactly one text embedding")
        text_embs = text_embs[0]
    if not isinstance(text_embs, torch.Tensor):
        raise StudentTrainingError("Bernini text embedding is not a tensor")
    lengths = torch.as_tensor(text_lens).reshape(-1)
    if lengths.numel() != 1 or int(lengths[0]) <= 0:
        raise StudentTrainingError("planner batches require one positive text length")
    length = int(lengths[0])
    if text_embs.ndim == 3:
        if int(text_embs.shape[0]) != 1 or int(text_embs.shape[1]) < length:
            raise StudentTrainingError("batched T5 embedding shape differs")
        selected = text_embs[0, :length]
    elif text_embs.ndim == 2:
        if int(text_embs.shape[0]) < length:
            raise StudentTrainingError("concatenated T5 embedding is shorter than text length")
        selected = text_embs[:length]
    else:
        raise StudentTrainingError("T5 embedding must be [L,D] or [1,L,D]")
    if int(selected.shape[-1]) <= 0 or not bool(torch.isfinite(selected).all()):
        raise StudentTrainingError("T5 embedding is empty or non-finite")
    selected = selected.float()
    nonzero = selected.abs().sum(dim=-1) > 0
    if not bool(nonzero.any()):
        raise StudentTrainingError("T5 embedding contains no non-padding token")
    last = int(nonzero.nonzero(as_tuple=False)[-1, 0]) + 1
    if not bool(nonzero[:last].all()):
        raise StudentTrainingError("T5 embedding has an internal zero row")
    unpadded = selected[:last]
    if not bool(torch.isfinite(unpadded).all()):
        raise StudentTrainingError("unpadded T5 tokens are non-finite")
    return unpadded.unsqueeze(0)


def _embed_instruction(renderer: Any, batch: Mapping[str, Any], device: Any) -> Any:
    import torch

    keys = ("input_ids", "attention_mask", "t5_input_lens")
    moved = {
        key: batch[key].to(device, non_blocking=True)
        if isinstance(batch[key], torch.Tensor)
        else batch[key]
        for key in keys
    }
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        text_lens, text_embs = renderer.get_t5_text_embeddings(
            moved["input_ids"], moved["attention_mask"], moved["t5_input_lens"]
        )
    return _unpadded_text_tokens(text_lens, text_embs).to(device)


def _clean_pair(
    raw_row: Mapping[str, Any], vae_mean: Any, vae_std: Any, z_dim: int, device: Any
) -> tuple[Any, Any]:
    sample = legacy.sanitize_preprocessed_row(raw_row)
    source_shape, _ = legacy.validate_81_frame_latents(
        sample, expected_parameter_channels=2 * z_dim
    )
    source_mode = motion.unpack_clean_mode(
        sample["video_vae_latents"][0], vae_mean, vae_std, max_frames=legacy.LATENT_FRAMES
    )
    target_mode = motion.unpack_clean_mode(
        sample["video_vae_latents"][1], vae_mean, vae_std, max_frames=legacy.LATENT_FRAMES
    )
    source = spt.packed_to_video(
        motion.flatten_velocity_patches(source_mode.unsqueeze(0)),
        height=source_shape[3] // 2,
        width=source_shape[4] // 2,
    )
    target = spt.packed_to_video(
        motion.flatten_velocity_patches(target_mode.unsqueeze(0)),
        height=source_shape[3] // 2,
        width=source_shape[4] // 2,
    )
    return source.to(device, non_blocking=True), target.to(device, non_blocking=True)


def hierarchical_sparse_gate_loss(
    student_probs: Any,
    teacher_probs: Any,
) -> dict[str, Any]:
    """Factor the action gate into sparse change and conditional T/G losses.

    The first term is an ordinary cellwise binary cross entropy for
    ``change = 1 - preserve``.  It deliberately keeps the teacher's real
    preserve/change frequency instead of independently normalizing all three
    classes.  The second term asks transport versus generate only where the
    teacher actually changes a cell.  This removes the uniform three-class
    stationary shortcut while retaining a finite gradient from every rare
    generate-labelled change cell.
    """

    import torch
    import torch.nn.functional as functional

    if tuple(student_probs.shape) != tuple(teacher_probs.shape):
        raise StudentTrainingError("student/teacher gate shapes differ")
    if getattr(student_probs, "ndim", None) != 5 or int(student_probs.shape[1]) != 3:
        raise StudentTrainingError("gate probabilities must be [B,3,T,H,W]")
    student = student_probs.float()
    teacher = teacher_probs.float()
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise StudentTrainingError("gate probabilities are non-finite")
    for label, probabilities in (("student", student), ("teacher", teacher)):
        if bool((probabilities < 0.0).any() or (probabilities > 1.0).any()):
            raise StudentTrainingError(f"{label} gate probabilities leave [0,1]")
        sums = probabilities.sum(dim=1)
        if not bool(torch.allclose(sums, torch.ones_like(sums), atol=2e-5, rtol=0.0)):
            raise StudentTrainingError(f"{label} gate probabilities do not sum to one")

    epsilon = 1e-6
    student_change = 1.0 - student[:, spt.GATE_PRESERVE]
    teacher_change = 1.0 - teacher[:, spt.GATE_PRESERVE]
    # ``binary_cross_entropy`` on sigmoid probabilities is deliberately
    # rejected by CUDA/ROCm autocast, even when its inputs have already been
    # promoted with ``.float()``.  The planner API exposes normalized gate
    # probabilities (rather than logits), so keep the exact hierarchical
    # objective and establish an explicit FP32 island for this operation.
    # This is nested safely under the Bernini bf16 autocast used by training.
    with torch.autocast(device_type=student_change.device.type, enabled=False):
        change_bce = functional.binary_cross_entropy(
            student_change.float().clamp(epsilon, 1.0 - epsilon),
            teacher_change.float(),
            reduction="mean",
        )

    student_tg = student[:, spt.GATE_TRANSPORT : spt.GATE_GENERATE + 1]
    conditional_tg = student_tg / student_tg.sum(dim=1, keepdim=True).clamp_min(
        epsilon
    )
    teacher_tg = teacher[:, spt.GATE_TRANSPORT : spt.GATE_GENERATE + 1]
    teacher_change_mass = teacher_tg.sum()
    conditional_tg_ce = -(
        teacher_tg * conditional_tg.clamp_min(epsilon).log()
    ).sum() / teacher_change_mass.clamp_min(1.0)
    gate_mass_l1 = (student_change.mean() - teacher_change.mean()).abs()
    return {
        "change_bce": change_bce,
        "conditional_tg_ce": conditional_tg_ce,
        "gate_mass_l1": gate_mass_l1,
        "student_change_fraction": student_change.mean(),
        "teacher_change_fraction": teacher_change.mean(),
    }


def transport_offset_huber_loss(
    student_offsets: Any,
    teacher_offsets: Any,
    teacher_gate_probs: Any,
) -> Any:
    """Mean SmoothL1 over teacher-transport cells *and* all three axes."""

    import torch
    import torch.nn.functional as functional

    if tuple(student_offsets.shape) != tuple(teacher_offsets.shape):
        raise StudentTrainingError("student/teacher offset shapes differ")
    if getattr(student_offsets, "ndim", None) != 5 or int(student_offsets.shape[1]) != 3:
        raise StudentTrainingError("offsets must be [B,3,T,H,W]")
    expected_gate_shape = (
        int(student_offsets.shape[0]),
        3,
        *map(int, student_offsets.shape[2:]),
    )
    if tuple(teacher_gate_probs.shape) != expected_gate_shape:
        raise StudentTrainingError("teacher gate/offset shapes differ")
    student = student_offsets.float()
    teacher = teacher_offsets.float()
    weights = teacher_gate_probs[:, spt.GATE_TRANSPORT : spt.GATE_TRANSPORT + 1].float()
    if not (
        bool(torch.isfinite(student).all())
        and bool(torch.isfinite(teacher).all())
        and bool(torch.isfinite(weights).all())
    ):
        raise StudentTrainingError("offset supervision is non-finite")
    per_axis = functional.smooth_l1_loss(
        student,
        teacher,
        reduction="none",
        beta=OFFSET_HUBER_BETA,
    )
    denominator = weights.sum() * int(student.shape[1])
    return (per_axis * weights).sum() / denominator.clamp_min(1.0)


def transport_cell_offset_mae(
    student_offsets: Any,
    teacher_offsets: Any,
    teacher_gate_probs: Any,
) -> Any:
    """Diagnostic offset MAE with the same cell-and-axis normalization."""

    weights = teacher_gate_probs[
        :, spt.GATE_TRANSPORT : spt.GATE_TRANSPORT + 1
    ].float()
    denominator = weights.sum() * int(student_offsets.shape[1])
    return (
        (student_offsets.float() - teacher_offsets.float()).abs() * weights
    ).sum() / denominator.clamp_min(1.0)


def hard_gate_spatial_metrics(
    student_probs: Any,
    teacher_probs: Any,
) -> dict[str, Any]:
    """Hard gate and change-localization metrics with explicit empty-set rules.

    If neither plan changes any cell, precision/recall/IoU are one.  If only
    one side has change cells, the undefined precision or recall is zero.  A
    degenerate all-preserve prediction therefore cannot receive a flattering
    score when the teacher contains changes.
    """

    import torch

    if tuple(student_probs.shape) != tuple(teacher_probs.shape):
        raise StudentTrainingError("student/teacher gate shapes differ for hard metrics")
    student_hard = student_probs.float().argmax(dim=1)
    teacher_hard = teacher_probs.float().argmax(dim=1)
    predicted_change = student_hard != spt.GATE_PRESERVE
    teacher_change = teacher_hard != spt.GATE_PRESERVE
    true_positive = (predicted_change & teacher_change).float().sum()
    predicted_count = predicted_change.float().sum()
    teacher_count = teacher_change.float().sum()
    union_count = (predicted_change | teacher_change).float().sum()
    both_empty = (predicted_count == 0) & (teacher_count == 0)
    precision = torch.where(
        predicted_count > 0,
        true_positive / predicted_count.clamp_min(1.0),
        both_empty.float(),
    )
    recall = torch.where(
        teacher_count > 0,
        true_positive / teacher_count.clamp_min(1.0),
        both_empty.float(),
    )
    iou = torch.where(
        union_count > 0,
        true_positive / union_count.clamp_min(1.0),
        both_empty.float(),
    )
    return {
        "hard_gate_argmax_accuracy": (student_hard == teacher_hard).float().mean(),
        "change_iou": iou,
        "change_precision": precision,
        "change_recall": recall,
        "hard_predicted_change_fraction": predicted_change.float().mean(),
        "hard_teacher_change_fraction": teacher_change.float().mean(),
    }


def offset_total_variation_loss(offsets: Any) -> Any:
    """Mean temporal/spatial total variation over axes that have neighbours."""

    import torch

    terms = []
    for axis in (2, 3, 4):
        length = int(offsets.shape[axis])
        if length > 1:
            left = offsets.narrow(axis, 1, length - 1)
            right = offsets.narrow(axis, 0, length - 1)
            terms.append((left.float() - right.float()).abs().mean())
    if not terms:
        return offsets.float().sum() * 0.0
    return torch.stack(terms).mean()


def _binary_spatial_metrics(probability: Any, target: Any) -> dict[str, Any]:
    """Thresholded binary overlap metrics with explicit empty-set behavior."""

    import torch

    if tuple(probability.shape) != tuple(target.shape):
        raise StudentTrainingError("binary metric shapes differ")
    if int(probability.numel()) == 0:
        one = probability.new_tensor(1.0, dtype=torch.float32)
        zero = probability.new_tensor(0.0, dtype=torch.float32)
        return {
            "iou": one,
            "precision": one,
            "recall": one,
            "f1": one,
            "predicted_fraction": zero,
            "target_fraction": zero,
        }
    predicted = probability.float() >= 0.5
    truth = target.float() >= 0.5
    true_positive = (predicted & truth).float().sum()
    predicted_count = predicted.float().sum()
    truth_count = truth.float().sum()
    union = (predicted | truth).float().sum()
    both_empty = (predicted_count == 0) & (truth_count == 0)
    precision = torch.where(
        predicted_count > 0,
        true_positive / predicted_count.clamp_min(1.0),
        both_empty.float(),
    )
    recall = torch.where(
        truth_count > 0,
        true_positive / truth_count.clamp_min(1.0),
        both_empty.float(),
    )
    iou = torch.where(
        union > 0,
        true_positive / union.clamp_min(1.0),
        both_empty.float(),
    )
    f1 = torch.where(
        precision + recall > 0,
        2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12),
        both_empty.float(),
    )
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_fraction": predicted.float().mean(),
        "target_fraction": truth.float().mean(),
    }


def phase_balanced_bce_with_logits(logits: Any, target: Any) -> dict[str, Any]:
    """Per-(sample, phase) normalized BCE with bounded rare-positive weight.

    For each ``(B,T)`` plane, ``pos_weight = clamp(N/P, 1, 4)`` where ``P``
    and ``N`` are positive and negative target mass.  An empty-positive plane
    uses weight one.  The positive and negative logistic terms are divided by
    their effective weighted mass before averaging over ``B*T``; changing the
    foreground fraction therefore cannot silently change the loss scale.
    """

    import torch
    import torch.nn.functional as functional

    if (
        getattr(logits, "ndim", None) != 5
        or tuple(logits.shape) != tuple(target.shape)
        or int(logits.shape[1]) != 1
    ):
        raise StudentTrainingError(
            "phase-balanced BCE logits/target must share [B,1,T,H,W]"
        )
    value = logits.float()
    truth = target.float()
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(truth).all()):
        raise StudentTrainingError("phase-balanced BCE inputs are non-finite")
    if bool(((truth < 0.0) | (truth > 1.0)).any()):
        raise StudentTrainingError("phase-balanced BCE target leaves [0,1]")
    positive_mass = truth.sum(dim=(-2, -1)).squeeze(1)
    negative_mass = (1.0 - truth).sum(dim=(-2, -1)).squeeze(1)
    ratio = negative_mass / positive_mass.clamp_min(1.0e-12)
    pos_weight = ratio.clamp(CHANGE_POS_WEIGHT_MIN, CHANGE_POS_WEIGHT_MAX)
    pos_weight = torch.where(
        positive_mass > 0.0,
        pos_weight,
        torch.ones_like(pos_weight),
    )
    weight = pos_weight[:, None, :, None, None]
    positive_term = weight * truth * functional.softplus(-value)
    negative_term = (1.0 - truth) * functional.softplus(value)
    effective_mass = (weight * truth + (1.0 - truth)).sum(
        dim=(-2, -1)
    ).squeeze(1)
    per_phase = (positive_term + negative_term).sum(
        dim=(-2, -1)
    ).squeeze(1) / effective_mass.clamp_min(1.0)
    return {
        "loss": per_phase.mean(),
        "per_phase_loss": per_phase,
        "pos_weight": pos_weight,
        "mean_pos_weight": pos_weight.mean(),
        "max_pos_weight": pos_weight.max(),
        "min_pos_weight": pos_weight.min(),
    }


def counterfactual_change_loss(
    action_change_logits: Any,
    noop_change_logits: Any,
    teacher_change: Any,
) -> dict[str, Any]:
    """Localize instruction effect without inventing background differences.

    Changed cells regress ``action-noop`` to a bounded positive margin, while
    preserve cells regress it to exactly zero.  A bounded margin keeps the
    contrastive term from rewarding arbitrarily saturated routing logits.
    This is deliberately different from BCE(delta, teacher_change): assigning
    target zero to preserve cells would incorrectly drive ``action-noop`` to
    negative infinity instead of making the two instructions invariant there.
    The same bounded per-phase foreground reweighting as the direct routing
    loss prevents a tiny changed region from disappearing in the mean.
    """

    import torch
    import torch.nn.functional as functional

    if tuple(action_change_logits.shape) != tuple(noop_change_logits.shape):
        raise StudentTrainingError("action/no-op change-logit geometry differs")
    if (
        getattr(action_change_logits, "ndim", None) != 5
        or int(action_change_logits.shape[1]) != 1
        or tuple(action_change_logits.shape) != tuple(teacher_change.shape)
    ):
        raise StudentTrainingError(
            "counterfactual change logits/target must share [B,1,T,H,W]"
        )
    delta = action_change_logits.float() - noop_change_logits.float()
    target = teacher_change.float()
    if not bool(torch.isfinite(delta).all()) or not bool(torch.isfinite(target).all()):
        raise StudentTrainingError("counterfactual change inputs are non-finite")
    if bool(((target < 0.0) | (target > 1.0)).any()):
        raise StudentTrainingError("counterfactual teacher change leaves [0,1]")

    positive_mass = target.sum(dim=(-2, -1)).squeeze(1)
    negative_mass = (1.0 - target).sum(dim=(-2, -1)).squeeze(1)
    ratio = negative_mass / positive_mass.clamp_min(1.0e-12)
    pos_weight = ratio.clamp(CHANGE_POS_WEIGHT_MIN, CHANGE_POS_WEIGHT_MAX)
    pos_weight = torch.where(
        positive_mass > 0.0,
        pos_weight,
        torch.ones_like(pos_weight),
    )
    weight = pos_weight[:, None, :, None, None]
    target_delta = COUNTERFACTUAL_CHANGE_MARGIN * target
    per_cell = functional.smooth_l1_loss(
        delta,
        target_delta,
        reduction="none",
        beta=1.0,
    )
    effective_weight = weight * target + (1.0 - target)
    numerator = (effective_weight * per_cell).sum(dim=(-2, -1)).squeeze(1)
    effective_mass = effective_weight.sum(dim=(-2, -1)).squeeze(1)
    per_phase = numerator / effective_mass.clamp_min(1.0)
    return {
        "loss": per_phase.mean(),
        "per_phase_loss": per_phase,
        "pos_weight": pos_weight,
        "mean_pos_weight": pos_weight.mean(),
        "max_pos_weight": pos_weight.max(),
        "min_pos_weight": pos_weight.min(),
        "delta_logits": delta,
        "target_delta": target_delta,
        "changed_margin_smooth_l1": (
            target * per_cell
        ).sum() / target.sum().clamp_min(1.0),
        "preserve_zero_delta_smooth_l1": (
            (1.0 - target) * per_cell
        ).sum() / (1.0 - target).sum().clamp_min(1.0),
    }


def change_polarization_loss(change_logits: Any) -> Any:
    """Penalize uncertain routing while leaving the preferred pole supervised."""

    import torch

    if getattr(change_logits, "ndim", None) != 5 or int(change_logits.shape[1]) != 1:
        raise StudentTrainingError("change polarization logits must be [B,1,T,H,W]")
    logits = change_logits.float()
    if not bool(torch.isfinite(logits).all()):
        raise StudentTrainingError("change polarization logits are non-finite")
    probability = torch.sigmoid(logits)
    return (4.0 * probability * (1.0 - probability)).mean()


def grounded_change_loss(
    change_logits: Any,
    teacher_change: Any,
    *,
    tversky_weight: float,
    phase_mass_weight: float,
) -> dict[str, Any]:
    """BCE + recall-sensitive Tversky + per-phase mass calibration."""

    import torch

    if tuple(change_logits.shape) != tuple(teacher_change.shape):
        raise StudentTrainingError("grounded change-logit/teacher shapes differ")
    logits = change_logits.float()
    target = teacher_change.float()
    balanced = phase_balanced_bce_with_logits(logits, target)
    bce = balanced["loss"]
    probability = torch.sigmoid(logits)
    reduce_dims = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=reduce_dims)
    false_positive = (probability * (1.0 - target)).sum(dim=reduce_dims)
    false_negative = ((1.0 - probability) * target).sum(dim=reduce_dims)
    tversky = 1.0 - (
        (true_positive + 1.0e-6)
        / (
            true_positive
            + 0.3 * false_positive
            + 0.7 * false_negative
            + 1.0e-6
        )
    ).mean()
    phase_mass = (
        probability.mean(dim=(-2, -1)) - target.mean(dim=(-2, -1))
    ).abs().mean()
    total = bce + float(tversky_weight) * tversky + float(phase_mass_weight) * phase_mass
    metrics = _binary_spatial_metrics(probability, target)
    return {
        "total": total,
        "bce": bce,
        "mean_pos_weight": balanced["mean_pos_weight"],
        "max_pos_weight": balanced["max_pos_weight"],
        "min_pos_weight": balanced["min_pos_weight"],
        "tversky": tversky,
        "phase_mass_l1": phase_mass,
        "probability": probability,
        **{f"hard_{name}": value for name, value in metrics.items()},
    }


def _pooled_change_bce(logits: Any, full_teacher_change: Any) -> dict[str, Any]:
    import torch.nn.functional as functional

    if getattr(logits, "ndim", None) != 5 or int(logits.shape[1]) != 1:
        raise StudentTrainingError("deep change logits must be [B,1,T,H,W]")
    if int(logits.shape[2]) != int(full_teacher_change.shape[2]):
        raise StudentTrainingError("deep change supervision may not downsample time")
    target = functional.adaptive_avg_pool3d(
        full_teacher_change.float(), tuple(map(int, logits.shape[2:]))
    )
    return phase_balanced_bce_with_logits(logits.float(), target)


def categorical_offset_loss(
    logits: Any,
    candidates: Any,
    teacher_offsets: Any,
    teacher_gate_probs: Any,
) -> dict[str, Any]:
    """Categorical 125-candidate loss on teacher-transport cells only."""

    import torch
    import torch.nn.functional as functional

    if getattr(logits, "ndim", None) != 5:
        raise StudentTrainingError("offset candidate logits must be [B,K,T,H,W]")
    if getattr(candidates, "ndim", None) != 2 or int(candidates.shape[1]) != 3:
        raise StudentTrainingError("offset candidates must be [K,3]")
    if int(logits.shape[1]) != int(candidates.shape[0]):
        raise StudentTrainingError("offset candidate count differs")
    expected_candidates = torch.tensor(
        grounded_phase.candidate_lattice(),
        device=candidates.device,
        dtype=torch.float32,
    )
    if tuple(candidates.shape) != (125, 3) or not bool(
        torch.equal(candidates.float(), expected_candidates)
    ):
        raise StudentTrainingError("offset candidates are not the exact ordered 125 lattice")
    expected = (
        int(teacher_offsets.shape[0]),
        int(candidates.shape[0]),
        *map(int, teacher_offsets.shape[2:]),
    )
    if tuple(logits.shape) != expected:
        raise StudentTrainingError("offset candidate/teacher geometry differs")
    teacher_vectors = teacher_offsets.float().permute(0, 2, 3, 4, 1).unsqueeze(1)
    candidate_vectors = candidates.float().view(1, -1, 1, 1, 1, 3)
    matches = (teacher_vectors == candidate_vectors).all(dim=-1)
    transport = teacher_gate_probs[:, spt.GATE_TRANSPORT].float()
    match_count = matches.sum(dim=1)
    if bool(((transport > 0.5) & (match_count != 1)).any()):
        raise StudentTrainingError("teacher transport offset is outside the 125 candidates")
    target_index = matches.float().argmax(dim=1)
    per_cell = functional.cross_entropy(
        logits.float(), target_index, reduction="none"
    )
    denominator = transport.sum()
    loss = (per_cell * transport).sum() / denominator.clamp_min(1.0)
    predicted = logits.float().argmax(dim=1)
    accuracy = (
        ((predicted == target_index).float() * transport).sum()
        / denominator.clamp_min(1.0)
    )
    return {
        "loss": loss,
        "top1_accuracy": accuracy,
        "transport_cells": denominator,
    }


def _grounded_planner_loss(
    action: spt.PhasePlan,
    teacher: spt.PhasePlan,
    noop: spt.PhasePlan,
    source: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    """SPT-v3 loss aligned to factorized gates and categorical transport."""

    import torch
    import torch.nn.functional as functional

    action.validate(source)
    teacher.validate(source)
    noop.validate(source)
    required = {
        "change_logits",
        "novelty_logits",
        "prebudget_generate_probs",
        "generate_budget_scale",
        "offset_candidate_logits",
        "offset_candidates",
        "soft_offsets",
        "coarse_change_logits",
        "mid_change_logits",
    }
    for label, plan in (("action", action), ("noop", noop)):
        if not isinstance(plan.diagnostics, Mapping) or not required <= set(plan.diagnostics):
            raise StudentTrainingError(f"{label} grounded planner diagnostics are incomplete")
        if plan.diagnostics.get("architecture") != grounded_phase.ARCHITECTURE_NAME:
            raise StudentTrainingError(f"{label} plan is not grounded SPT-v3")
    action_diag = action.diagnostics
    noop_diag = noop.diagnostics
    teacher_change = 1.0 - teacher.gate_probs[
        :, spt.GATE_PRESERVE : spt.GATE_PRESERVE + 1
    ].float()
    change = grounded_change_loss(
        action_diag["change_logits"],
        teacher_change,
        tversky_weight=args.change_tversky_weight,
        phase_mass_weight=args.phase_change_mass_weight,
    )
    teacher_generate = teacher.gate_probs[
        :, spt.GATE_GENERATE : spt.GATE_GENERATE + 1
    ].float()
    # novelty is explicitly conditional on leaving Preserve.  Supervising it
    # with absolute G would multiply by change a second time at execution and
    # systematically suppress Generate on genuinely novel target cells.
    teacher_conditional_novelty = teacher_generate / teacher_change.clamp_min(1.0e-6)
    teacher_conditional_novelty = teacher_conditional_novelty.clamp(0.0, 1.0)
    novelty_per_cell = functional.binary_cross_entropy_with_logits(
        action_diag["novelty_logits"].float(),
        teacher_conditional_novelty,
        reduction="none",
    )
    novelty = (novelty_per_cell * teacher_change).sum() / teacher_change.sum().clamp_min(1.0)
    raw_generate = action_diag["prebudget_generate_probs"].float()
    phase_generate_mass = (
        raw_generate.mean(dim=(-2, -1)).squeeze(1)
        - teacher.gate_probs[:, spt.GATE_GENERATE].float().mean(dim=(-2, -1))
    ).abs().mean()
    mid_change = _pooled_change_bce(
        action_diag["mid_change_logits"], teacher_change
    )
    coarse_change = _pooled_change_bce(
        action_diag["coarse_change_logits"], teacher_change
    )
    counterfactual = counterfactual_change_loss(
        action_diag["change_logits"],
        noop_diag["change_logits"],
        teacher_change,
    )
    polarization = change_polarization_loss(action_diag["change_logits"])
    categorical = categorical_offset_loss(
        action_diag["offset_candidate_logits"],
        action_diag["offset_candidates"],
        teacher.offsets,
        teacher.gate_probs,
    )
    expected_offset = transport_offset_huber_loss(
        action_diag["soft_offsets"], teacher.offsets, teacher.gate_probs
    )
    offset_mae = transport_cell_offset_mae(
        action.offsets, teacher.offsets, teacher.gate_probs
    )
    noop_change_logits = noop_diag["change_logits"].float()
    noop_change = phase_balanced_bce_with_logits(
        noop_change_logits,
        torch.zeros_like(noop_change_logits),
    )
    noop_mid_change = phase_balanced_bce_with_logits(
        noop_diag["mid_change_logits"].float(),
        torch.zeros_like(noop_diag["mid_change_logits"], dtype=torch.float32),
    )
    noop_coarse_change = phase_balanced_bce_with_logits(
        noop_diag["coarse_change_logits"].float(),
        torch.zeros_like(noop_diag["coarse_change_logits"], dtype=torch.float32),
    )
    noop_generate = noop.gate_probs[:, spt.GATE_GENERATE].float().mean()
    noop_candidates = noop_diag["offset_candidates"]
    if not bool(torch.equal(noop_candidates.float(), action_diag["offset_candidates"].float())):
        raise StudentTrainingError("action/no-op offset candidate lattices differ")
    zero_matches = (noop_candidates.float() == 0).all(dim=1)
    if int(zero_matches.sum()) != 1:
        raise StudentTrainingError("offset candidate lattice has no unique zero")
    zero_index = int(zero_matches.nonzero(as_tuple=False)[0, 0])
    noop_offset_logits = noop_diag["offset_candidate_logits"].float()
    noop_offset_target = torch.full(
        (int(noop_offset_logits.shape[0]), *map(int, noop_offset_logits.shape[2:])),
        zero_index,
        dtype=torch.long,
        device=noop_offset_logits.device,
    )
    noop_offset_ce = functional.cross_entropy(
        noop_offset_logits, noop_offset_target, reduction="mean"
    )
    noop_offset_top1 = (
        noop_offset_logits.argmax(dim=1) == noop_offset_target
    ).float().mean()
    action_total = (
        args.gate_loss_weight
        * (
            change["total"]
            + args.conditional_gate_loss_weight * novelty
            + args.phase_generate_mass_weight * phase_generate_mass
            + args.mid_change_loss_weight * mid_change["loss"]
            + args.coarse_change_loss_weight * coarse_change["loss"]
        )
        + getattr(args, "counterfactual_change_loss_weight", 0.0)
        * counterfactual["loss"]
        + getattr(args, "change_polarization_loss_weight", 0.0) * polarization
        + args.offset_loss_weight * categorical["loss"]
        + args.expected_offset_loss_weight * expected_offset
    )
    noop_total = (
        noop_change["loss"]
        + args.mid_change_loss_weight * noop_mid_change["loss"]
        + args.coarse_change_loss_weight * noop_coarse_change["loss"]
        + args.noop_generate_weight * noop_generate
        + args.noop_offset_weight * noop_offset_ce
    )
    total = action_total + args.noop_loss_weight * noop_total

    novelty_probability = torch.sigmoid(action_diag["novelty_logits"].float())
    hard_teacher_change = teacher_change >= 0.5
    changed_count = int(hard_teacher_change.sum())
    novelty_metrics = _binary_spatial_metrics(
        novelty_probability[hard_teacher_change].reshape(1, 1, 1, 1, changed_count),
        teacher_conditional_novelty[hard_teacher_change].reshape(
            1, 1, 1, 1, changed_count
        ),
    )
    budget_scale = action_diag["generate_budget_scale"].float()
    hard_metrics = hard_gate_spatial_metrics(action.gate_probs, teacher.gate_probs)
    executed_change_probability = 1.0 - action.gate_probs[
        :, spt.GATE_PRESERVE : spt.GATE_PRESERVE + 1
    ].float()
    executed_change_metrics = _binary_spatial_metrics(
        executed_change_probability, teacher_change
    )
    return total, {
        "action_gate": change["total"] + args.conditional_gate_loss_weight * novelty,
        "action_change_bce": change["bce"],
        "action_change_mean_pos_weight": change["mean_pos_weight"],
        "action_change_max_pos_weight": change["max_pos_weight"],
        "action_change_min_pos_weight": change["min_pos_weight"],
        "action_change_tversky": change["tversky"],
        "action_phase_change_mass_l1": change["phase_mass_l1"],
        "action_conditional_tg_ce": novelty,
        "action_prebudget_phase_generate_mass_l1": phase_generate_mass,
        "action_mid_change_bce": mid_change["loss"],
        "action_mid_change_mean_pos_weight": mid_change["mean_pos_weight"],
        "action_mid_change_max_pos_weight": mid_change["max_pos_weight"],
        "action_coarse_change_bce": coarse_change["loss"],
        "action_coarse_change_mean_pos_weight": coarse_change["mean_pos_weight"],
        "action_coarse_change_max_pos_weight": coarse_change["max_pos_weight"],
        "action_counterfactual_change_smooth_l1": counterfactual["loss"],
        "action_counterfactual_mean_pos_weight": counterfactual["mean_pos_weight"],
        "action_counterfactual_max_pos_weight": counterfactual["max_pos_weight"],
        "action_counterfactual_changed_margin_smooth_l1": counterfactual[
            "changed_margin_smooth_l1"
        ],
        "action_counterfactual_preserve_zero_delta_smooth_l1": counterfactual[
            "preserve_zero_delta_smooth_l1"
        ],
        "action_change_polarization": polarization,
        "action_offset_candidate_ce": categorical["loss"],
        "offset_candidate_top1_accuracy": categorical["top1_accuracy"],
        "transport_cell_offset_mae": offset_mae,
        "action_expected_offset_huber": expected_offset,
        "change_head_iou": change["hard_iou"],
        "change_head_precision": change["hard_precision"],
        "change_head_recall": change["hard_recall"],
        "change_head_f1": change["hard_f1"],
        "executed_change_iou": executed_change_metrics["iou"],
        "executed_change_precision": executed_change_metrics["precision"],
        "executed_change_recall": executed_change_metrics["recall"],
        "executed_change_f1": executed_change_metrics["f1"],
        "executed_hard_change_fraction": executed_change_metrics[
            "predicted_fraction"
        ],
        "hard_predicted_change_fraction": change["hard_predicted_fraction"],
        "hard_teacher_change_fraction": change["hard_target_fraction"],
        "conditional_tg_f1": novelty_metrics["f1"],
        "student_change_fraction": change["probability"].mean(),
        "teacher_change_fraction": teacher_change.mean(),
        "student_preserve_fraction": action.gate_probs[:, spt.GATE_PRESERVE].float().mean(),
        "student_transport_fraction": action.gate_probs[:, spt.GATE_TRANSPORT].float().mean(),
        "student_generate_fraction": action.gate_probs[:, spt.GATE_GENERATE].float().mean(),
        "student_observed_max_generate_fraction_per_phase": action.gate_probs[
            :, spt.GATE_GENERATE
        ].float().mean(dim=(-2, -1)).max(),
        "student_prebudget_max_generate_fraction_per_phase": raw_generate.mean(
            dim=(-2, -1)
        ).max(),
        "student_min_generate_budget_scale": budget_scale.min(),
        "teacher_preserve_fraction": teacher.gate_probs[:, spt.GATE_PRESERVE].float().mean(),
        "teacher_transport_fraction": teacher.gate_probs[:, spt.GATE_TRANSPORT].float().mean(),
        "teacher_generate_fraction": teacher.gate_probs[:, spt.GATE_GENERATE].float().mean(),
        "noop_gate": noop_change["loss"],
        "noop_fine_change_bce": noop_change["loss"],
        "noop_mid_change_bce": noop_mid_change["loss"],
        "noop_coarse_change_bce": noop_coarse_change["loss"],
        "noop_generate_fraction": noop_generate,
        "noop_offset_zero_ce": noop_offset_ce,
        "noop_offset_zero_top1_accuracy": noop_offset_top1,
        "noop_hard_offset_abs_mean": noop.offsets.float().abs().mean(),
        "noop_hard_change_fraction": (
            torch.sigmoid(noop_change_logits) >= 0.5
        ).float().mean(),
        "noop_executed_hard_change_fraction": (
            1.0
            - noop.gate_probs[
                :, spt.GATE_PRESERVE : spt.GATE_PRESERVE + 1
            ].float()
            >= 0.5
        ).float().mean(),
        "hard_gate_argmax_accuracy": hard_metrics["hard_gate_argmax_accuracy"],
        "change_iou": hard_metrics["change_iou"],
        "change_precision": hard_metrics["change_precision"],
        "change_recall": hard_metrics["change_recall"],
    }


def _planner_loss(
    action: spt.PhasePlan,
    teacher: spt.PhasePlan,
    noop: spt.PhasePlan,
    source: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    import torch

    required_budget_diagnostics = {
        "prebudget_generate_fraction",
        "postbudget_generate_fraction",
        "budget_reject_fraction",
        "max_generate_fraction_per_phase",
        "observed_max_postbudget_generate_fraction_per_phase",
    }
    if not isinstance(teacher.diagnostics, Mapping) or not required_budget_diagnostics <= set(
        teacher.diagnostics
    ):
        raise StudentTrainingError("action teacher lacks conservative generate-budget diagnostics")
    declared_budget = teacher.diagnostics["max_generate_fraction_per_phase"]
    observed_max = float(
        teacher.diagnostics["observed_max_postbudget_generate_fraction_per_phase"]
    )
    if (
        declared_budget != args.max_generate_fraction_per_phase
        or observed_max > args.max_generate_fraction_per_phase + 1e-6
    ):
        raise StudentTrainingError("action teacher violates the 0.12 per-phase generate budget")
    if (
        isinstance(action.diagnostics, Mapping)
        and action.diagnostics.get("architecture")
        == grounded_phase.ARCHITECTURE_NAME
    ):
        return _grounded_planner_loss(action, teacher, noop, source, args)
    action.validate(source)
    teacher.validate(source)
    gate_parts = hierarchical_sparse_gate_loss(action.gate_probs, teacher.gate_probs)
    hierarchical_action_gate = (
        gate_parts["change_bce"]
        + args.conditional_gate_loss_weight * gate_parts["conditional_tg_ce"]
        + args.gate_mass_loss_weight * gate_parts["gate_mass_l1"]
    )
    action_offset = transport_offset_huber_loss(
        action.offsets,
        teacher.offsets,
        teacher.gate_probs,
    )
    action_offset_mae = transport_cell_offset_mae(
        action.offsets,
        teacher.offsets,
        teacher.gate_probs,
    )
    hard_metrics = hard_gate_spatial_metrics(action.gate_probs, teacher.gate_probs)
    action_smooth = offset_total_variation_loss(action.offsets)
    identity = spt.exact_identity_plan(source)
    noop_parts = spt.plan_distillation_loss(noop, identity, source)
    action_total = (
        args.gate_loss_weight * hierarchical_action_gate
        + args.offset_loss_weight * action_offset
        + args.smooth_loss_weight * action_smooth
    )
    noop_total = noop_parts["gate"] + args.offset_loss_weight * noop.offsets.float().abs().mean()
    total = action_total + args.noop_loss_weight * noop_total
    return total, {
        "action_gate": hierarchical_action_gate,
        "action_change_bce": gate_parts["change_bce"],
        "action_conditional_tg_ce": gate_parts["conditional_tg_ce"],
        "action_gate_mass_l1": gate_parts["gate_mass_l1"],
        "action_offset": action_offset,
        "transport_cell_offset_mae": action_offset_mae,
        "action_smooth": action_smooth,
        "noop_gate": noop_parts["gate"],
        "noop_offset": noop.offsets.float().abs().mean(),
        "student_change_fraction": gate_parts["student_change_fraction"],
        "teacher_change_fraction": gate_parts["teacher_change_fraction"],
        "student_preserve_fraction": action.gate_probs[:, spt.GATE_PRESERVE].float().mean(),
        "student_transport_fraction": action.gate_probs[:, spt.GATE_TRANSPORT].float().mean(),
        "student_generate_fraction": action.gate_probs[:, spt.GATE_GENERATE].float().mean(),
        "student_observed_max_generate_fraction_per_phase": action.gate_probs[
            :, spt.GATE_GENERATE
        ].float().mean(dim=(-2, -1)).max(),
        "teacher_preserve_fraction": teacher.gate_probs[:, spt.GATE_PRESERVE].float().mean(),
        "teacher_generate_fraction": teacher.gate_probs[:, spt.GATE_GENERATE].float().mean(),
        "teacher_transport_fraction": teacher.gate_probs[:, spt.GATE_TRANSPORT].float().mean(),
        "teacher_prebudget_generate_fraction": source.new_tensor(
            float(teacher.diagnostics["prebudget_generate_fraction"]), dtype=torch.float32
        ),
        "teacher_budget_reject_fraction": source.new_tensor(
            float(teacher.diagnostics["budget_reject_fraction"]), dtype=torch.float32
        ),
        "teacher_observed_max_generate_fraction_per_phase": source.new_tensor(
            observed_max, dtype=torch.float32
        ),
        **hard_metrics,
    }


def all_reduce_planner_gradients(named: Sequence[tuple[str, Any]]) -> float:
    """Explicitly average every replicated planner gradient across four ranks."""

    import torch
    import torch.distributed as dist

    missing = [name for name, parameter in named if parameter.grad is None]
    if missing:
        raise StudentTrainingError(f"planner parameters lack gradients: {missing[:8]}")
    finite = all(bool(torch.isfinite(parameter.grad).all()) for _, parameter in named)
    if not legacy._distributed_boolean(finite, op="all"):
        raise StudentTrainingError("planner gradient is non-finite")
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    for _, parameter in named:
        if world > 1:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world)
    squared = torch.zeros((), dtype=torch.float64, device=named[0][1].device)
    for _, parameter in named:
        squared += parameter.grad.detach().double().pow(2).sum()
    norm = math.sqrt(float(squared.item()))
    if not math.isfinite(norm):
        raise StudentTrainingError("planner gradient norm is non-finite")
    return norm


def data_parallel_row_index(
    global_step: int, world_size: int, rank: int, training_rows: int
) -> int:
    """Deterministically shard each optimizer-step cohort across DP ranks."""

    if global_step < 0 or world_size <= 0 or not 0 <= rank < world_size or training_rows <= 0:
        raise StudentTrainingError("invalid data-parallel row schedule")
    return (global_step * world_size + rank) % training_rows


def data_parallel_dataset_row_index(
    global_step: int,
    world_size: int,
    rank: int,
    training_membership: Mapping[str, Any],
) -> int:
    """Map the DP schedule ordinal to its exact bound dataset row."""

    training_rows = int(training_membership["training_rows"])
    members = training_membership.get("members")
    if not isinstance(members, list) or len(members) != training_rows:
        raise StudentTrainingError("training membership row count differs")
    ordinal = data_parallel_row_index(global_step, world_size, rank, training_rows)
    member = members[ordinal]
    if not isinstance(member, Mapping) or type(member.get("row_index")) is not int:
        raise StudentTrainingError("training membership has an invalid scheduled row")
    return int(member["row_index"])


def data_parallel_seed(
    base_seed: int, global_step: int, world_size: int, rank: int, row_index: int
) -> int:
    """Bind stochastic preprocessing to optimizer step, rank, and selected row."""

    if global_step < 0 or world_size <= 0 or not 0 <= rank < world_size or row_index < 0:
        raise StudentTrainingError("invalid data-parallel seed schedule")
    sample_ordinal = global_step * world_size + rank
    return legacy.step_seed(base_seed, sample_ordinal, row_index)


def validate_data_parallel_cohort(
    cohort: Sequence[Mapping[str, Any]],
    *,
    global_step: int,
    world_size: int,
    training_membership: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate rank-sharded row/IID/hash membership for one optimizer step."""

    training_rows = int(training_membership["training_rows"])
    if len(cohort) != world_size:
        raise StudentTrainingError("data-parallel cohort size differs from world size")
    ordered = sorted((dict(item) for item in cohort), key=lambda item: int(item["rank"]))
    if [int(item["rank"]) for item in ordered] != list(range(world_size)):
        raise StudentTrainingError("data-parallel cohort ranks are incomplete or duplicated")
    expected_rows = [
        data_parallel_dataset_row_index(
            global_step, world_size, rank, training_membership
        )
        for rank in range(world_size)
    ]
    actual_rows = [int(item["row_index"]) for item in ordered]
    if actual_rows != expected_rows:
        raise StudentTrainingError(
            f"data-parallel cohort row schedule differs: {actual_rows} vs {expected_rows}"
        )
    if training_rows >= world_size and len(set(actual_rows)) != world_size:
        raise StudentTrainingError("data-parallel cohort repeats a row within one step")
    members = {
        int(member["row_index"]): member for member in training_membership["members"]
    }
    for item in ordered:
        row_index = int(item["row_index"])
        member = members.get(row_index)
        if member is None:
            raise StudentTrainingError(f"row {row_index} is outside bound training membership")
        if str(item["iid"]) != str(member["iid"]):
            raise StudentTrainingError(f"row {row_index} iid differs from training membership")
        if str(item["identity_sha256"]) != str(member["identity_sha256"]):
            raise StudentTrainingError(
                f"row {row_index} identity hash differs from training membership"
            )
    return ordered


def assert_data_parallel_cohort(
    *,
    row_index: int,
    iid: str,
    identity_sha256: str,
    global_step: int,
    distributed: Any,
    training_membership: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Gather and prove that ranks consume distinct scheduled dataset rows."""

    import torch.distributed as dist

    local = {
        "rank": int(distributed.rank),
        "row_index": int(row_index),
        "iid": str(iid),
        "identity_sha256": str(identity_sha256),
    }
    if dist.is_available() and dist.is_initialized():
        cohort: list[Optional[dict[str, Any]]] = [None] * distributed.world_size
        dist.all_gather_object(cohort, local)
        if any(item is None for item in cohort):
            raise StudentTrainingError("data-parallel cohort gather returned an empty rank")
        materialized = [item for item in cohort if item is not None]
    else:
        materialized = [local]
    return validate_data_parallel_cohort(
        materialized,
        global_step=global_step,
        world_size=distributed.world_size,
        training_membership=training_membership,
    )


def _mean_metrics(values: Mapping[str, float], device: Any) -> dict[str, float]:
    """Report global-DP means rather than rank-0-only training metrics."""

    import torch
    import torch.distributed as dist

    names = sorted(values)
    tensor = torch.tensor([float(values[name]) for name in names], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    return {name: float(value) for name, value in zip(names, tensor.tolist())}


def _method_hashes(architecture: str) -> dict[str, str]:
    files = (
        "phase_transport.py",
        "phase_query_planner.py",
        "contracts.py",
        "train_student.py",
    )
    if architecture == grounded_phase.ARCHITECTURE_NAME:
        files = (*files, "grounded_phase_planner.py")
    return {name: legacy.file_sha256(SPT_ROOT / name) for name in files}


def _training_membership(
    dataset: Any,
    train_prefix_rows: Optional[int],
    *,
    selected_membership: Optional[str] = None,
    dataset_summary: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Hash-bind the exact ordered training cohort without reading latent blobs."""

    if selected_membership is not None:
        if train_prefix_rows is not None:
            raise StudentTrainingError(
                "selected membership cannot be combined with an ordered prefix"
            )
        if dataset_summary is None:
            raise StudentTrainingError(
                "dataset summary is required for selected membership validation"
            )
        # Delayed import avoids a module cycle: the read-only cohort auditor
        # reuses this trainer's latent unpacking helper, while this path only
        # needs its strict, hash-recomputing membership loader at runtime.
        try:
            import audit_teacher_cohort as cohort

            value, row_indices = cohort.load_selected_membership(
                selected_membership,
                dataset=dataset,
                dataset_summary=dataset_summary,
                require_sufficient=True,
            )
        except (ImportError, OSError, RuntimeError) as error:
            raise StudentTrainingError(
                f"selected teacher-trust membership is invalid: {error}"
            ) from error
        entries = [
            {
                "row_index": int(member["row_index"]),
                "iid": str(member["iid"]),
                "identity_sha256": str(member["identity_sha256"]),
            }
            for member in value["members"]
        ]
        if tuple(entry["row_index"] for entry in entries) != row_indices:
            raise StudentTrainingError("selected membership order differs after loading")
        return {
            "selection": "teacher_trust_membership",
            "full_dataset_rows": len(dataset),
            "training_rows": len(entries),
            "diagnostic_subset": len(entries) < len(dataset),
            "members": entries,
            "membership_sha256": legacy.object_sha256(entries),
            "selected_membership_schema": value["schema_version"],
            "selected_membership_digest": value["membership_digest"],
            "selected_membership_scan": dict(value["scan"]),
            "selected_membership_thresholds": dict(value["selector_thresholds"]),
            "implicit_dataset_fallback_forbidden": True,
        }

    training_rows = len(dataset) if train_prefix_rows is None else int(train_prefix_rows)
    if training_rows > len(dataset):
        raise StudentTrainingError(
            f"train-prefix-rows={training_rows} exceeds the full dataset size {len(dataset)}"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise StudentTrainingError("pyarrow is required to bind training membership") from error
    entries: list[dict[str, Any]] = []
    row_index = 0
    for path in dataset.files:
        parquet = pq.ParquetFile(path)
        schema_names = set(parquet.schema_arrow.names)
        identity_column = "iid" if "iid" in schema_names else "id" if "id" in schema_names else None
        columns = ["inputs"] + ([identity_column] if identity_column is not None else [])
        for row_group in range(parquet.metadata.num_row_groups):
            if len(entries) >= training_rows:
                break
            rows = parquet.read_row_group(row_group, columns=columns).to_pylist()
            for row in rows:
                if len(entries) >= training_rows:
                    break
                iid = str(row.get(identity_column, "")) if identity_column else ""
                if not iid.strip():
                    raise StudentTrainingError(
                        f"training row {row_index} has no stable iid/id for membership binding"
                    )
                entries.append(
                    {
                        "row_index": row_index,
                        "iid": iid,
                        "identity_sha256": legacy.dataset_identity(row, row_index),
                    }
                )
                row_index += 1
        if len(entries) >= training_rows:
            break
    if len(entries) != training_rows:
        raise StudentTrainingError(
            f"resolved {len(entries)} membership rows, expected {training_rows}"
        )
    return {
        "selection": "full_dataset" if training_rows == len(dataset) else "ordered_prefix",
        "full_dataset_rows": len(dataset),
        "training_rows": training_rows,
        "diagnostic_subset": training_rows < len(dataset),
        "members": entries,
        "membership_sha256": legacy.object_sha256(entries),
    }


def _immutable(
    *,
    args: argparse.Namespace,
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    planner_config: Any,
    teacher_config: spt.PhaseTransportConfig,
    training_membership: Mapping[str, Any],
    world_size: int,
) -> dict[str, Any]:
    architecture = str(planner_config.architecture)
    grounded = architecture == grounded_phase.ARCHITECTURE_NAME
    value = {
        "method": _method_name(architecture),
        "planner_architecture": architecture,
        "method_files_sha256": _method_hashes(architecture),
        "bernini_commit": args.expected_bernini_commit.lower(),
        "veomni_commit": args.expected_veomni_commit.lower(),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "dataset_signature": dataset.signature,
        "dataset_summary_sha256": dataset_summary["sha256"],
        "dataset_index_sha256": dataset_summary["index_sha256"],
        "training_membership": dict(training_membership),
        "data_parallel_world_size": int(world_size),
        "ulysses_size": PLANNER_ULYSSES_SIZE,
        "samples_per_optimizer_step": int(world_size),
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        # Store the JSON-domain representation (lists, not Python tuples) so a
        # receipt round-trip remains exactly comparable during strict resume.
        "planner_config": json.loads(
            legacy.canonical_json_bytes(asdict(planner_config)).decode("utf-8")
        ),
        "teacher_config": json.loads(
            legacy.canonical_json_bytes(asdict(teacher_config)).decode("utf-8")
        ),
        "gate_loss_weight": float(args.gate_loss_weight),
        "conditional_gate_loss_weight": float(args.conditional_gate_loss_weight),
        "gate_mass_loss_weight": float(args.gate_mass_loss_weight),
        "action_gate_loss": (
            "grounded_routing_v3p1_balanced_change_novelty_counterfactual"
            if grounded
            else "hierarchical_sparse_change_then_conditional_tg_v1"
        ),
        "action_change_loss": (
            "per_bt_normalized_pos_weight_1_to_4_bce_plus_tversky_plus_per_phase_mass"
            if grounded
            else "ordinary_cellwise_bce_on_one_minus_preserve"
        ),
        "change_class_balance": (
            {
                "scope": "per_sample_per_latent_phase",
                "positive_weight": "negative_mass_div_positive_mass",
                "minimum_positive_weight": CHANGE_POS_WEIGHT_MIN,
                "maximum_positive_weight": CHANGE_POS_WEIGHT_MAX,
                "empty_positive_phase_weight": 1.0,
                "normalization": "effective_weighted_spatial_mass_then_mean_BxT",
                "applies_to": [
                    "action_fine",
                    "action_mid_pooled",
                    "action_coarse_pooled",
                    "action_minus_noop_fine",
                ],
            }
            if grounded
            else None
        ),
        "counterfactual_change_loss_weight": float(
            getattr(args, "counterfactual_change_loss_weight", 0.0)
        ),
        "counterfactual_change_loss": (
            "balanced_smooth_l1_action_minus_noop_to_teacher_change_margin"
            if grounded
            else None
        ),
        "counterfactual_change_margin": (
            COUNTERFACTUAL_CHANGE_MARGIN if grounded else None
        ),
        "change_polarization_loss_weight": float(
            getattr(args, "change_polarization_loss_weight", 0.0)
        ),
        "change_polarization_loss": (
            "mean_4p_times_1_minus_p_on_action_change_probability"
            if grounded
            else None
        ),
        "noop_change_supervision": (
            "exact_zero_bce_at_fine_mid_coarse_with_shared_deep_weights"
            if grounded
            else "ordinary_preserve_cross_entropy"
        ),
        "action_conditional_loss": (
            "teacher_change_weighted_conditional_generate_ratio_bce_logits"
            if grounded
            else "teacher_change_cells_only_transport_vs_generate_ce"
        ),
        "action_gate_mass_calibration": (
            "per_phase_change_and_generate_l1"
            if grounded
            else "l1_student_vs_teacher_change_fraction"
        ),
        "offset_loss_weight": float(args.offset_loss_weight),
        "offset_loss": (
            "125_candidate_source_correlation_ce_plus_expected_huber"
            if grounded
            else "transport_cell_and_three_axis_mean_smooth_l1"
        ),
        "offset_huber_beta": OFFSET_HUBER_BETA,
        "smooth_loss_weight": float(args.smooth_loss_weight),
        "noop_loss_weight": float(args.noop_loss_weight),
        "teacher_feature_channels": int(args.teacher_feature_channels),
        "student_semantic_inputs": ["source_video", "edit_instruction"],
        "instruction_representation": "full_unpadded_t5_token_sequence",
        "instruction_pooling": None,
        "source_position_channels": ["normalized_t", "normalized_y", "normalized_x"],
        "phase_query_count": spt.LATENT_PHASES,
        "cross_attention_layers": (
            {
                "global_text": grounded_phase.GLOBAL_TEXT_LAYERS,
                "dense_text_shared_modules": grounded_phase.DENSE_TEXT_LAYERS,
                "dense_text_applications": 2,
                "temporal_axis": grounded_phase.TEMPORAL_ATTENTION_LAYERS,
            }
            if grounded
            else phase_query.CROSS_ATTENTION_LAYERS
        ),
        "structural_generate_budget": grounded,
        "grounded_initialization": (
            _grounded_initialization_contract() if grounded else None
        ),
        "inactive_loss_hyperparameters": (
            ["gate_mass_loss_weight", "smooth_loss_weight"]
            if grounded
            else [
                "change_tversky_weight",
                "phase_change_mass_weight",
                "phase_generate_mass_weight",
                "mid_change_loss_weight",
                "coarse_change_loss_weight",
                "expected_offset_loss_weight",
                "noop_generate_weight",
                "noop_offset_weight",
                "counterfactual_change_loss_weight",
                "change_polarization_loss_weight",
            ]
        ),
        "change_tversky_weight": float(args.change_tversky_weight),
        "phase_change_mass_weight": float(args.phase_change_mass_weight),
        "phase_generate_mass_weight": float(args.phase_generate_mass_weight),
        "mid_change_loss_weight": float(args.mid_change_loss_weight),
        "coarse_change_loss_weight": float(args.coarse_change_loss_weight),
        "expected_offset_loss_weight": float(args.expected_offset_loss_weight),
        "noop_generate_weight": float(args.noop_generate_weight),
        "noop_offset_weight": float(args.noop_offset_weight),
        "routing_only_offset_supervision_disabled": bool(
            grounded
            and args.offset_loss_weight == 0.0
            and args.expected_offset_loss_weight == 0.0
            and args.noop_offset_weight == 0.0
        ),
        "target_used_by_student": False,
        "target_used_by_training_teacher_only": True,
    }
    return {"value": value, "digest": legacy.object_sha256(value)}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = legacy.canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _receipt(
    *,
    args: argparse.Namespace,
    global_step: int,
    metrics: Optional[Mapping[str, float]],
    immutable: Mapping[str, Any],
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    training_membership: Mapping[str, Any],
    planner: Any,
    named: Sequence[tuple[str, Any]],
    initialization_digest: str,
    distributed: Any,
    backend: str,
    resumed_from: Optional[str],
    rank_local_runtime_cache: bool = False,
) -> dict[str, Any]:
    architecture = str(
        getattr(getattr(planner, "config", None), "architecture", args.planner_architecture)
    )
    grounded = architecture == grounded_phase.ARCHITECTURE_NAME
    if architecture not in (
        phase_query.ARCHITECTURE_NAME,
        grounded_phase.ARCHITECTURE_NAME,
    ):
        raise StudentTrainingError("receipt planner architecture is unsupported")
    value: dict[str, Any] = {
        "schema_version": _receipt_schema(architecture),
        "method": _method_name(architecture),
        "global_step": global_step,
        "global_samples_seen": global_step * distributed.world_size,
        "max_steps": args.max_steps,
        "last_metrics": dict(metrics) if metrics else None,
        "immutable_contract": dict(immutable),
        "dataset": {
            "path": str(dataset.root),
            "rows": len(dataset),
            "full_dataset_rows": len(dataset),
            "training_rows": training_membership["training_rows"],
            "diagnostic_subset": training_membership["diagnostic_subset"],
            "training_selection": training_membership["selection"],
            "training_membership_sha256": training_membership["membership_sha256"],
            "selected_membership_schema": training_membership.get(
                "selected_membership_schema"
            ),
            "selected_membership_digest": training_membership.get(
                "selected_membership_digest"
            ),
            "training_membership": list(training_membership["members"]),
            "signature": dataset.signature,
            "summary": dict(dataset_summary),
        },
        "planner": {
            "class": type(planner).__name__,
            "architecture": architecture,
            "parameter_count": sum(int(parameter.numel()) for _, parameter in named),
            "parameter_names": [name for name, _ in named],
            "parameter_names_sha256": legacy.object_sha256([name for name, _ in named]),
            "initialization_digest": initialization_digest,
        },
        "supervision": {
            "student_api": ["source", "instruction_tokens"],
            "instruction_representation": "full_unpadded_t5_token_sequence",
            "instruction_pooling": None,
            "source_position_channels": ["normalized_t", "normalized_y", "normalized_x"],
            "learned_phase_queries": 0 if grounded else spt.LATENT_PHASES,
            "grounded_edit_slots": args.edit_slots if grounded else 0,
            "explicit_sinusoidal_phase_encoding": True,
            "cross_attention_layers": (
                {
                    "global_text": grounded_phase.GLOBAL_TEXT_LAYERS,
                    "dense_text_shared_modules": grounded_phase.DENSE_TEXT_LAYERS,
                    "dense_text_applications": 2,
                    "temporal_axis": grounded_phase.TEMPORAL_ATTENTION_LAYERS,
                }
                if grounded
                else phase_query.CROSS_ATTENTION_LAYERS
            ),
            "student_target_argument_exists": False,
            "target_used_by_oracle_teacher_only": True,
            "external_mask_track_pose_flow": False,
            "noop_execution_is_exact_source_bypass": True,
            "noop_student_calibration": True,
            "action_gate_loss": (
                "grounded_routing_v3p1_balanced_change_novelty_counterfactual"
                if grounded
                else "hierarchical_sparse_change_then_conditional_tg_v1"
            ),
            "action_change_loss": (
                "per_bt_normalized_pos_weight_1_to_4_bce_plus_tversky_plus_per_phase_mass"
                if grounded
                else "ordinary_cellwise_bce_on_one_minus_preserve"
            ),
            "change_class_balance": (
                {
                    "scope": "per_sample_per_latent_phase",
                    "positive_weight": "negative_mass_div_positive_mass",
                    "minimum_positive_weight": CHANGE_POS_WEIGHT_MIN,
                    "maximum_positive_weight": CHANGE_POS_WEIGHT_MAX,
                    "empty_positive_phase_weight": 1.0,
                    "normalization": "effective_weighted_spatial_mass_then_mean_BxT",
                    "pooled_mid_and_coarse": True,
                }
                if grounded
                else None
            ),
            "counterfactual_change_loss": (
                "balanced_smooth_l1_action_minus_noop_to_teacher_change_margin"
                if grounded
                else None
            ),
            "counterfactual_change_margin": (
                COUNTERFACTUAL_CHANGE_MARGIN if grounded else None
            ),
            "counterfactual_change_loss_weight": (
                getattr(args, "counterfactual_change_loss_weight", 0.0)
                if grounded
                else 0.0
            ),
            "change_polarization_loss": (
                "mean_4p_times_1_minus_p_on_action_change_probability"
                if grounded
                else None
            ),
            "change_polarization_loss_weight": (
                getattr(args, "change_polarization_loss_weight", 0.0)
                if grounded
                else 0.0
            ),
            "noop_change_supervision": (
                "exact_zero_bce_at_fine_mid_coarse_with_shared_deep_weights"
                if grounded
                else "ordinary_preserve_cross_entropy"
            ),
            "action_conditional_loss": (
                "teacher_change_weighted_conditional_generate_ratio_bce_logits"
                if grounded
                else "teacher_change_cells_only_transport_vs_generate_ce"
            ),
            "conditional_gate_loss_weight": args.conditional_gate_loss_weight,
            "action_gate_mass_calibration": (
                "per_phase_change_and_generate_l1"
                if grounded
                else "l1_student_vs_teacher_change_fraction"
            ),
            "gate_mass_loss_weight": args.gate_mass_loss_weight,
            "offset_loss": (
                "125_candidate_source_correlation_ce_plus_expected_huber"
                if grounded
                else "transport_cell_and_three_axis_mean_smooth_l1"
            ),
            "offset_huber_beta": OFFSET_HUBER_BETA,
            "noop_gate_loss": (
                "fine_plus_weighted_mid_coarse_zero_change_bce_plus_weighted_generate_mass_plus_weighted_zero_offset_candidate_ce"
                if grounded
                else "ordinary_preserve_cross_entropy"
            ),
            "noop_gate_terms": (
                {
                    "fine_zero_change_bce_weight": 1.0,
                    "mid_zero_change_bce_weight": args.mid_change_loss_weight,
                    "coarse_zero_change_bce_weight": args.coarse_change_loss_weight,
                    "generate_mass_weight": args.noop_generate_weight,
                    "zero_offset_candidate_ce_weight": args.noop_offset_weight,
                    "outer_noop_weight": args.noop_loss_weight,
                }
                if grounded
                else None
            ),
            "noop_offset_weight": args.noop_offset_weight if grounded else 0.0,
            "max_generate_fraction_per_phase": args.max_generate_fraction_per_phase,
            "generate_budget_reject_fallback": "preserve",
            "generate_budget_is_structural": grounded,
            "grounded_initialization": (
                _grounded_initialization_contract() if grounded else None
            ),
            "routing_only_offset_supervision_disabled": bool(
                grounded
                and args.offset_loss_weight == 0.0
                and args.expected_offset_loss_weight == 0.0
                and args.noop_offset_weight == 0.0
            ),
            "transport_candidate_lattice": (
                [list(candidate) for candidate in grounded_phase.candidate_lattice()]
                if grounded
                else None
            ),
            "latent_phases": 21,
        },
        "distributed": {
            "world_size": distributed.world_size,
            "data_parallel_size": distributed.world_size,
            "ulysses_size": PLANNER_ULYSSES_SIZE,
            "backend": backend,
            "same_pair_all_ranks": False,
            "sample_schedule": "(global_step * world_size + rank) % training_rows",
            "seed_schedule": "step_seed(base_seed, global_step * world_size + rank, row_index)",
            "cohort_validation": "all_gather_rank_row_iid_identity_sha256",
            "samples_per_optimizer_step": distributed.world_size,
            "global_samples_seen": global_step * distributed.world_size,
            "explicit_planner_gradient_all_reduce": distributed.world_size > 1,
            "rank_local_runtime_cache": bool(rank_local_runtime_cache),
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_gradient_norm": args.max_grad_norm,
        },
        "resumed_from": resumed_from,
        "experimental_training": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    value["receipt_digest"] = legacy.object_sha256(value)
    return value


def _save(
    *,
    planner: Any,
    optimizer: Any,
    output: Path,
    global_step: int,
    receipt: Mapping[str, Any],
    immutable: Mapping[str, Any],
    named: Sequence[tuple[str, Any]],
    rank: int,
) -> Path:
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file

    final = output / f"checkpoint-{global_step:08d}"
    if rank == 0:
        if final.exists():
            raise StudentTrainingError(f"refusing to overwrite checkpoint: {final}")
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / f".{final.name}.tmp-{os.getpid()}"
        temporary.mkdir()
        state = {key: value.detach().cpu().contiguous() for key, value in planner.state_dict().items()}
        save_file(state, str(temporary / "planner.safetensors"))
        _atomic_json(temporary / "planner_config.json", asdict(planner.config))
        torch.save(
            {
                "schema_version": _optimizer_schema(str(planner.config.architecture)),
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "immutable_contract": dict(immutable),
                "parameter_names": [name for name, _ in named],
            },
            temporary / "optimizer.pt",
        )
        _atomic_json(temporary / "receipt.json", receipt)
        os.replace(temporary, final)
        _atomic_json(
            output / "latest.json",
            {
                "checkpoint": str(final),
                "global_step": global_step,
                "receipt_digest": receipt["receipt_digest"],
            },
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return final


def _load_resume(
    root: Path,
    expected_architecture: str = phase_query.ARCHITECTURE_NAME,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        root = root.expanduser().resolve(strict=True)
        receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
        config = json.loads((root / "planner_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudentTrainingError(f"cannot read resume checkpoint: {error}") from error
    for name in ("planner.safetensors", "optimizer.pt"):
        if not (root / name).is_file():
            raise StudentTrainingError(f"resume checkpoint lacks {name}")
    if (
        receipt.get("schema_version") != _receipt_schema(expected_architecture)
        or receipt.get("planner", {}).get("architecture") != expected_architecture
        or config.get("architecture") != expected_architecture
    ):
        raise StudentTrainingError(
            "resume checkpoint is not the requested current planner/loss schema; "
            "legacy, cross-architecture, global-pooled, and prior-loss-schema checkpoints "
            "cannot be resumed"
        )
    candidate = dict(receipt)
    declared = candidate.pop("receipt_digest", None)
    if legacy.object_sha256(candidate) != declared:
        raise StudentTrainingError("resume receipt digest differs")
    return receipt, config


def main(argv: Optional[Sequence[str]] = None) -> int:
    rank_local_runtime_cache = configure_rank_local_runtime_cache()
    args = build_parser().parse_args(argv)
    validate_cli(args)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise StudentTrainingError(str(error)) from error
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.training.data import NoiseScheduler, process_renderer_sample

    distributed = legacy.distributed_contract()
    device, backend = legacy.initialise_distributed(distributed)
    from bernini.parallel import init_parallel_state

    # The student planner is replicated data parallel.  It does not execute the
    # Wan DiT and therefore must not sequence-shard one sample with Ulysses.
    init_parallel_state(ulysses_size=PLANNER_ULYSSES_SIZE)
    legacy.seed_same_sample(args.seed)
    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    dataset_summary = legacy.validate_preprocessed_dataset_summary(
        args.dataset_summary, dataset, allow_incomplete=args.allow_incomplete_dataset
    )
    training_membership = _training_membership(
        dataset,
        args.train_prefix_rows,
        selected_membership=args.selected_membership,
        dataset_summary=dataset_summary,
    )
    if int(training_membership["training_rows"]) < distributed.world_size:
        raise StudentTrainingError(
            "training membership must contain at least one distinct row per data-parallel rank"
        )
    output = Path(args.output).expanduser().resolve()

    renderer_config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    renderer_config.dtype = torch.bfloat16
    renderer = BerniniRendererModel(renderer_config)
    renderer.requires_grad_(False)
    renderer.eval()
    renderer.t5_text_encoder.to(device)
    renderer.t5_text_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    scheduler = NoiseScheduler(**legacy.noise_scheduler_kwargs())

    # Probe the exact pinned T5 output width before constructing the planner.
    first_probe_row_index = int(training_membership["members"][0]["row_index"])
    first_row = dataset[first_probe_row_index]
    legacy.seed_same_sample(legacy.step_seed(args.seed, 0, 0))
    action_batch, copy_batch, _ = delta._prepare_paired_batches(
        raw_row=first_row,
        tokenizer=tokenizer,
        rope=rope,
        vae_mean=vae_mean,
        vae_std=vae_std,
        z_dim=z_dim,
        scheduler=scheduler,
        noop_instruction=args.noop_instruction,
        process_renderer_sample=process_renderer_sample,
    )
    first_tokens = _embed_instruction(renderer, action_batch, device)
    if args.planner_architecture == grounded_phase.ARCHITECTURE_NAME:
        planner_config = grounded_phase.GroundedPhasePlannerConfig(
            architecture=args.planner_architecture,
            latent_channels=64,
            text_channels=int(first_tokens.shape[-1]),
            hidden_channels=args.hidden_channels,
            attention_heads=args.attention_heads,
            match_channels=args.match_channels,
            edit_slots=args.edit_slots,
            dense_query_chunk_size=args.dense_query_chunk_size,
            max_generate_fraction_per_phase=args.max_generate_fraction_per_phase,
        )
        planner = grounded_phase.GroundedPhasePlanner(planner_config).to(device)
        teacher_max_temporal_offset = float(
            max(abs(value) for value in grounded_phase.TEMPORAL_CANDIDATES)
        )
        teacher_max_spatial_offset = float(
            max(abs(value) for value in grounded_phase.SPATIAL_CANDIDATES)
        )
    else:
        planner_config = phase_query.PhaseQueryPlannerConfig(
            architecture=args.planner_architecture,
            latent_channels=64,
            text_channels=int(first_tokens.shape[-1]),
            hidden_channels=args.hidden_channels,
            attention_heads=args.attention_heads,
        )
        planner = phase_query.PhaseQueryPlanner(planner_config).to(device)
        teacher_max_temporal_offset = planner_config.max_temporal_offset
        teacher_max_spatial_offset = planner_config.max_spatial_offset
    teacher_config = spt.PhaseTransportConfig(
        latent_channels=planner_config.latent_channels,
        text_channels=planner_config.text_channels,
        hidden_channels=planner_config.hidden_channels,
        max_temporal_offset=teacher_max_temporal_offset,
        max_spatial_offset=teacher_max_spatial_offset,
        teacher_temporal_offsets=tuple(grounded_phase.TEMPORAL_CANDIDATES),
        teacher_spatial_offsets=tuple(grounded_phase.SPATIAL_CANDIDATES),
        teacher_temperature=args.teacher_temperature,
        teacher_generate_threshold=args.teacher_generate_threshold,
        max_generate_fraction_per_phase=args.max_generate_fraction_per_phase,
    )
    immutable = _immutable(
        args=args,
        dataset=dataset,
        dataset_summary=dataset_summary,
        planner_config=planner_config,
        teacher_config=teacher_config,
        training_membership=training_membership,
        world_size=distributed.world_size,
    )

    global_step = 0
    resumed_from: Optional[str] = None
    resume_state: Optional[dict[str, Any]] = None
    if args.resume:
        resume_root = Path(args.resume).expanduser().resolve(strict=True)
        prior, saved_config = _load_resume(resume_root, args.planner_architecture)
        normalized_saved = dict(saved_config)
        for key in ("teacher_temporal_offsets", "teacher_spatial_offsets"):
            if isinstance(normalized_saved.get(key), list):
                normalized_saved[key] = tuple(normalized_saved[key])
        if prior.get("immutable_contract") != immutable or normalized_saved != asdict(planner_config):
            raise StudentTrainingError("resume immutable/config contract differs")
        global_step = int(prior.get("global_step", -1))
        if global_step < 0:
            raise StudentTrainingError("resume global step is invalid")
        saved = load_file(str(resume_root / "planner.safetensors"), device=str(device))
        if set(saved) != set(planner.state_dict()):
            raise StudentTrainingError("resume planner state-key scope differs")
        planner.load_state_dict(saved, strict=True)
        try:
            resume_state = torch.load(resume_root / "optimizer.pt", map_location="cpu", weights_only=False)
        except TypeError:
            resume_state = torch.load(resume_root / "optimizer.pt", map_location="cpu")
        resumed_from = str(resume_root)
    if global_step > args.max_steps:
        raise StudentTrainingError("resume step exceeds max-steps")

    planner.train()
    named = [(name, parameter) for name, parameter in planner.named_parameters() if parameter.requires_grad]
    initialization_digest = legacy.synchronize_trainable_parameters(named, source_rank=0)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if resume_state is not None:
        if (
            resume_state.get("schema_version")
            != _optimizer_schema(args.planner_architecture)
            or resume_state.get("immutable_contract") != immutable
            or resume_state.get("parameter_names") != [name for name, _ in named]
            or int(resume_state.get("global_step", -1)) != global_step
        ):
            raise StudentTrainingError("resume optimizer contract differs")
        optimizer.load_state_dict(resume_state["optimizer"])
        delta._optimizer_to(optimizer, device)

    last_metrics: Optional[dict[str, float]] = None
    last_saved = global_step if args.resume else -1
    while global_step < args.max_steps:
        row_index = data_parallel_dataset_row_index(
            global_step,
            distributed.world_size,
            distributed.rank,
            training_membership,
        )
        raw_row = dataset[row_index]
        identity = legacy.dataset_identity(raw_row, row_index)
        iid = str(raw_row.get("iid", raw_row.get("id", "")))
        cohort = assert_data_parallel_cohort(
            row_index=row_index,
            iid=iid,
            identity_sha256=identity,
            global_step=global_step,
            distributed=distributed,
            training_membership=training_membership,
        )
        seed = data_parallel_seed(
            args.seed,
            global_step,
            distributed.world_size,
            distributed.rank,
            row_index,
        )
        legacy.seed_same_sample(seed)
        source, target = _clean_pair(raw_row, vae_mean, vae_std, z_dim, device)
        action_batch, copy_batch, _ = delta._prepare_paired_batches(
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
        action_tokens = _embed_instruction(renderer, action_batch, device)
        noop_tokens = _embed_instruction(renderer, copy_batch, device)
        with torch.no_grad():
            teacher = spt.build_oracle_plan(
                source,
                target,
                teacher_config,
                feature_channels=args.teacher_feature_channels,
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Target is intentionally absent from both calls.
            action_plan = student_plan(planner, source, action_tokens)
            noop_plan = student_plan(planner, source, noop_tokens)
            loss, parts = _planner_loss(action_plan, teacher, noop_plan, source, args)
        finite = bool(torch.isfinite(loss.detach())) and all(
            bool(torch.isfinite(value.detach())) for value in parts.values()
        )
        if not legacy._distributed_boolean(finite, op="all"):
            raise StudentTrainingError(f"non-finite planner loss at step {global_step + 1}")
        loss.backward()
        gradient_norm = all_reduce_planner_gradients(named)
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named], args.max_grad_norm)
        optimizer.step()
        global_step += 1
        local_metrics = {
            "total": float(loss.detach()),
            **{name: float(value.detach()) for name, value in parts.items()},
        }
        last_metrics = _mean_metrics(local_metrics, device)
        last_metrics["preclip_gradient_norm"] = gradient_norm
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "step": global_step,
                        "cohort": cohort,
                        "global_samples_seen": global_step * distributed.world_size,
                        **last_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.save_every > 0 and global_step % args.save_every == 0:
            receipt = _receipt(
                args=args,
                global_step=global_step,
                metrics=last_metrics,
                immutable=immutable,
                dataset=dataset,
                dataset_summary=dataset_summary,
                training_membership=training_membership,
                planner=planner,
                named=named,
                initialization_digest=initialization_digest,
                distributed=distributed,
                backend=backend,
                resumed_from=resumed_from,
                rank_local_runtime_cache=rank_local_runtime_cache,
            )
            _save(
                planner=planner,
                optimizer=optimizer,
                output=output,
                global_step=global_step,
                receipt=receipt,
                immutable=immutable,
                named=named,
                rank=distributed.rank,
            )
            last_saved = global_step
    if last_saved != global_step:
        receipt = _receipt(
            args=args,
            global_step=global_step,
            metrics=last_metrics,
            immutable=immutable,
            dataset=dataset,
            dataset_summary=dataset_summary,
            training_membership=training_membership,
            planner=planner,
            named=named,
            initialization_digest=initialization_digest,
            distributed=distributed,
            backend=backend,
            resumed_from=resumed_from,
            rank_local_runtime_cache=rank_local_runtime_cache,
        )
        _save(
            planner=planner,
            optimizer=optimizer,
            output=output,
            global_step=global_step,
            receipt=receipt,
            immutable=immutable,
            named=named,
            rank=distributed.rank,
        )
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
