#!/usr/bin/env python3
"""Strict 81-frame Prior-Guided Tangent Trust-Region LoRA inference.

This is the inference half of the Bernini v5 action-editing experiment.  It
keeps the pinned Bernini ``v2v_apg`` sampler and its forty official UniPC
updates, but audits four transformer evaluations at every noisy state:

* frozen negative (LoRA disabled);
* frozen semantic no-op (LoRA disabled);
* frozen action (LoRA disabled); and
* adapted action (LoRA enabled, unmerged, unit scale).

All four branches are evaluated under ``torch.no_grad``.  With the official
``momentum=0`` normalized-guidance APG, their clean fields define the raw
motion difference ``B_raw=A0-N0``, executable causal prior
``B=Q0(B_raw)``, and one-sided adapter correction ``R=Q0(A_theta-A0)``.
Packed Wan tokens are reshaped explicitly from
``[B,N,D]`` to ``[B,21,S,D]`` before calling the exact operator shared with
training.  The clean field sent to the original UniPC step is
``source + B + gamma*C_B(R)``.  At every zero-gamma main-arm step the shared
operator returns the exact frozen-prior object and no adapter correction is
added.  Receipt-bound frozen-prior and causal-frozen-prior arms isolate
``source+B_raw`` from ``source+B`` at the same seed.

The only external conditions are an exact 81-frame source video and an edit
instruction.  No target, mask, track, flow, pose, trajectory, or first-frame
anchor is accepted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_delta_lora as adapter_loader  # noqa: E402
import inference_sigma_strata as sigma_strata  # noqa: E402
import motion_residual as motion  # noqa: E402
import prior_guided_tangent as tangent  # noqa: E402
import train_lora as trainer  # noqa: E402
import train_prior_tangent_lora as v5_train  # noqa: E402
import tri_branch_unipc as tri  # noqa: E402
from spt_v2 import infer_c2fr as frozen  # noqa: E402


INFERENCE_RECEIPT_SCHEMA = (
    "bernini-prior-guided-tangent-trust-region-lora-inference-receipt-v5"
)
METHOD_NAME = tangent.METHOD_NAME
REQUIRED_LORA_SCOPE = "cross_q"
REQUIRED_TARGET_MODULE_COUNT = 30
ADAPTER_SCALE = 1.0
LATENT_PHASES = 21
EXECUTION_ARMS = (
    "main",
    "frozen_prior",
    "causal_frozen_prior",
    "parallel_only",
)
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PriorTangentInferenceError(RuntimeError):
    """Raised before publication when a v5 inference invariant differs."""


def build_parser() -> argparse.ArgumentParser:
    parser = frozen.build_parser()
    parser.description = (
        "Run Bernini-R 1.3B prior-guided tangent LoRA on one exact 81-frame source"
    )
    parser.add_argument(
        "--adapter-checkpoint",
        default=None,
        help=(
            "required for main/parallel_only; optional for frozen-prior diagnostics"
        ),
    )
    parser.add_argument(
        "--execution-arm",
        choices=EXECUTION_ARMS,
        default="main",
        help="receipt-bound main method or same-seed frozen-prior diagnostic",
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    try:
        frozen.validate_cli(args)
    except frozen.C2FRInferenceError as error:
        raise PriorTangentInferenceError(str(error)) from error
    if args.execution_arm not in EXECUTION_ARMS:
        raise PriorTangentInferenceError("execution_arm differs from the audited arms")
    adapter_present = isinstance(args.adapter_checkpoint, str) and bool(
        args.adapter_checkpoint
    )
    if args.adapter_checkpoint is not None and not adapter_present:
        raise PriorTangentInferenceError(
            "adapter_checkpoint must be omitted or a non-empty path"
        )
    if args.execution_arm in ("main", "parallel_only") and not adapter_present:
        raise PriorTangentInferenceError(
            f"execution arm {args.execution_arm} requires --adapter-checkpoint"
        )
    if float(args.alpha) != ADAPTER_SCALE:
        raise PriorTangentInferenceError("formal v5 inference requires alpha exactly 1")
    if (
        float(args.max_generate_fraction) != frozen.DEFAULT_GENERATE_CAP
        or float(args.energy_coverage) != frozen.DEFAULT_ENERGY_COVERAGE
    ):
        raise PriorTangentInferenceError(
            "v5 does not expose the legacy binary sparse-router controls"
        )


def runtime_contract() -> dict[str, Any]:
    return {
        "method": METHOD_NAME,
        "external_inference_conditions": ["source_video", "action_instruction"],
        "internal_fixed_controls": ["semantic_noop_instruction", "negative_prompt"],
        "per_step_branches": [
            "frozen_negative",
            "frozen_noop",
            "frozen_action",
            "adapted_action",
        ],
        "frozen_branch_adapter_state": "disabled",
        "adapted_action_adapter_state": "enabled_unmerged_unit_scale",
        "all_branch_autograd": False,
        "apg": "official_normalized_guidance",
        "apg_momentum": 0.0,
        "packed_field_shape": "[B,N,D]",
        "operator_field_shape": "[B,21,S,D]",
        "latent_phases": LATENT_PHASES,
        "raw_prior": "B_raw=A0-N0",
        "prior": "B=Q0(B_raw)",
        "raw_adapter_correction": "R=Q0(A_theta-A0)",
        "execution": "source+B+gamma(step)*C_B(R)",
        "gamma_schedule": list(tangent.correction_gamma_schedule()),
        "gamma_zero_scheduler_boundary": "exact_causal_frozen_prior",
        "kappa_parallel": tangent.DEFAULT_KAPPA_PARALLEL,
        "kappa_perp": tangent.DEFAULT_KAPPA_PERP,
        "epsilon": tangent.DEFAULT_EPSILON,
        "binary_support_operator": False,
        "custom_integrator": False,
        "execution_arms": {
            "main": "source+Q0(B_raw)+gamma*C_B(R), kappa_perp=0.15",
            "parallel_only": "source+Q0(B_raw)+gamma*C_B(R), kappa_perp=0",
            "frozen_prior": "source+B_raw, adapter correction not routed",
            "causal_frozen_prior": "source+Q0(B_raw), adapter correction not routed",
        },
        "first_frame_anchor": False,
        "forbidden_conditions": [
            "target_video",
            "paired_target",
            "mask",
            "track",
            "swept_tube",
            "pose",
            "trajectory",
            "optical_flow",
            "first_frame_anchor",
        ],
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PriorTangentInferenceError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PriorTangentInferenceError(f"{label} must contain one JSON object")
    return value


def _audited_attention_projection_names() -> list[str]:
    return sorted(
        f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
        for block in range(30)
        for attention in (1, 2)
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    )


def _validate_serialized_target_coverage(
    serialized: Any, *, targets: Sequence[str]
) -> tuple[str, ...]:
    if (
        not isinstance(serialized, list)
        or not serialized
        or not all(isinstance(name, str) and name for name in serialized)
        or len(serialized) != len(set(serialized))
    ):
        raise PriorTangentInferenceError(
            "adapter serialized target_modules must be unique non-empty strings"
        )
    target_set = set(targets)
    matched: set[str] = set()
    for suffix in serialized:
        candidates = {
            target
            for target in target_set
            if target == suffix or target.endswith(f".{suffix}")
        }
        if not candidates:
            raise PriorTangentInferenceError(
                "adapter serialized target_modules exceed receipt scope"
            )
        matched.update(candidates)
    if matched != target_set:
        raise PriorTangentInferenceError(
            "adapter serialized target_modules do not cover receipt scope"
        )
    return tuple(sorted(serialized))


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PriorTangentInferenceError(f"training receipt lacks {label}")
    return value


def validate_training_adapter_contract(
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str = trainer.CHECKPOINT_TREE_SHA256,
) -> dict[str, Any]:
    """Fail closed unless the checkpoint is the exact v5 cross-q experiment."""

    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    if (
        not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or trainer.object_sha256(candidate) != digest
    ):
        raise PriorTangentInferenceError("training receipt digest differs")
    if receipt.get("schema_version") != v5_train.RECEIPT_SCHEMA:
        raise PriorTangentInferenceError("training receipt schema differs from v5")
    if receipt.get("method") != METHOD_NAME or v5_train.METHOD_NAME != METHOD_NAME:
        raise PriorTangentInferenceError("training method identity differs from v5")
    if receipt.get("bernini_commit") != trainer.BERNINI_OFFICIAL_COMMIT:
        raise PriorTangentInferenceError("training Bernini revision differs")
    if receipt.get("veomni_commit") != trainer.VEOMNI_TESTED_COMMIT:
        raise PriorTangentInferenceError("training VeOmni revision differs")
    checkpoint = _require_mapping(receipt.get("checkpoint"), label="checkpoint identity")
    if checkpoint.get("tree_sha256") != expected_checkpoint_tree_sha256:
        raise PriorTangentInferenceError("training checkpoint tree differs")

    global_step = receipt.get("global_step")
    if type(global_step) is not int or global_step < tangent.NUM_DENOISING_STEPS:
        raise PriorTangentInferenceError(
            "formal v5 inference requires at least one complete 40-sigma cycle"
        )
    if receipt.get("inference_sigma_strata") != sigma_strata.build_sigma_strata_receipt(
        completed_optimizer_steps=global_step
    ):
        raise PriorTangentInferenceError("training sigma-strata receipt differs")

    immutable = _require_mapping(
        receipt.get("immutable_contract"), label="immutable contract"
    )
    value = _require_mapping(immutable.get("value"), label="immutable value")
    if immutable.get("digest") != trainer.object_sha256(value):
        raise PriorTangentInferenceError("training immutable contract digest differs")
    adapter = _require_mapping(receipt.get("adapter"), label="adapter identity")
    supervision = _require_mapping(receipt.get("supervision"), label="supervision")

    targets = adapter.get("target_modules")
    expected_targets = motion.select_lora_scope(
        _audited_attention_projection_names(), REQUIRED_LORA_SCOPE
    )
    if (
        adapter.get("scope") != REQUIRED_LORA_SCOPE
        or not isinstance(targets, list)
        or targets != expected_targets
        or targets != sorted(set(targets))
        or len(targets) != REQUIRED_TARGET_MODULE_COUNT
        or adapter.get("target_module_count") != REQUIRED_TARGET_MODULE_COUNT
        or adapter.get("target_modules_sha256") != trainer.object_sha256(targets)
        or value.get("lora_scope") != REQUIRED_LORA_SCOPE
        or value.get("target_modules") != targets
    ):
        raise PriorTangentInferenceError(
            "training adapter must target exactly 30 cross-attention to_q modules"
        )
    if (
        adapter.get("rank") != trainer.LORA_RANK
        or float(adapter.get("alpha", -1.0)) != trainer.LORA_ALPHA
    ):
        raise PriorTangentInferenceError("training adapter rank/alpha differs")
    for name in ("initialization_digest", "checkpoint_parameter_digest"):
        item = adapter.get(name)
        if not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None:
            raise PriorTangentInferenceError(f"training adapter {name} is invalid")

    # These fields are the executable train/inference boundary.  Extra
    # diagnostics may evolve, but none of these values may be aliased away.
    expected_immutable = {
        "method": METHOD_NAME,
        "schema_version": v5_train.RECEIPT_SCHEMA,
        "checkpoint_tree_sha256": expected_checkpoint_tree_sha256,
        "frames": 81,
        "latent_phases": LATENT_PHASES,
        "learning_rate": v5_train.LEARNING_RATE,
        "lora_rank": trainer.LORA_RANK,
        "lora_alpha": trainer.LORA_ALPHA,
        "lora_scope": REQUIRED_LORA_SCOPE,
        "branches_per_endpoint": [
            "base_negative_adapter_off_no_grad",
            "base_noop_adapter_off_no_grad",
            "base_action_adapter_off_no_grad",
            "adapted_action_adapter_on_grad",
        ],
        "forwards_per_endpoint": 4,
        "forwards_per_optimizer_step": 8,
        "base_apg": {
            "guidance_scale": v5_train.APG_GUIDANCE_SCALE,
            "eta": v5_train.APG_ETA,
            "norm_threshold": v5_train.APG_NORM_THRESHOLD,
            "momentum": v5_train.APG_MOMENTUM,
            "negative_prompt_sha256": hashlib.sha256(
                v5_train.DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
            ).hexdigest(),
            "negative_tokenization": (
                "official_renderer_unconditional_verbatim"
            ),
            "clean_reconstruction": (
                "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
            ),
        },
        "raw_prior": "raw_frozen_prior=base_guided_action-base_guided_noop",
        "prior": "causal_frozen_prior=Q0(raw_frozen_prior)",
        "adapter_correction": "Q0(adapted_guided_action-base_guided_action)",
        "teacher_correction": "Q0((target-source)-causal_frozen_prior)",
        "phase_zero_contract": (
            "executed_motion_exact_zero_source_exactly_preserved"
        ),
        "trust_region": {
            "kappa_parallel": tangent.DEFAULT_KAPPA_PARALLEL,
            "kappa_perp": tangent.DEFAULT_KAPPA_PERP,
            "epsilon": tangent.DEFAULT_EPSILON,
            "phase_dim": 1,
        },
        "gamma_schedule": list(tangent.correction_gamma_schedule()),
        "gamma_schedule_sha256": trainer.object_sha256(
            list(tangent.correction_gamma_schedule())
        ),
        "gamma_contract": (
            "0-23 full; 24-34 inclusive cosine taper "
            "(gamma24=1,gamma34=0); 35-39 exact causal frozen prior"
        ),
        "sigma_schedule": "exact_40_step_flow_shift_5_cycle",
        "sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "sigma_selector": "absolute_global_step_mod_40",
        "inference_conditions": ["source_video", "action_instruction"],
        "training_only_conditions": ["target_video"],
        "forbidden_inference_conditions": [
            "target_video",
            "mask",
            "track",
            "swept_tube",
            "pose",
            "trajectory",
            "optical_flow",
            "first_frame_anchor",
        ],
    }
    for name, expected in expected_immutable.items():
        if value.get(name) != expected:
            raise PriorTangentInferenceError(
                f"training v5 immutable field differs: {name}"
            )
    expected_supervision = {
        "method": METHOD_NAME,
        "source_target_bridge": True,
        "four_branch_endpoint": True,
        "base_branches_adapter_disabled": True,
        "base_branches_no_grad": True,
        "adapted_action_only_trainable_forward": True,
        "causal_frozen_prior": "Q0(base_action-base_noop)",
        "executed_motion_phase_zero": "exact_zero",
        "official_apg_momentum": v5_train.APG_MOMENTUM,
        "official_apg_guidance_scale": v5_train.APG_GUIDANCE_SCALE,
        "official_apg_eta": v5_train.APG_ETA,
        "official_apg_norm_threshold": v5_train.APG_NORM_THRESHOLD,
        "negative_prompt_sha256": v5_train.NEGATIVE_PROMPT_SHA256,
        "negative_tokenization": "official_renderer_unconditional_verbatim",
        "field_loss_weight": v5_train.FIELD_LOSS_WEIGHT,
        "bridge_loss_weight": v5_train.BRIDGE_LOSS_WEIGHT,
        "late_replay_loss_weight": v5_train.LATE_REPLAY_LOSS_WEIGHT,
        "late_replay_gate": "1-gamma",
        "target_used_as_model_condition": False,
        "target_used_as_offline_teacher": True,
        "inference_conditions": ["source_video", "action_instruction"],
        "external_mask_track_flow_pose_trajectory": False,
        "post_video_acceptance": "pending",
        "production_claim_forbidden": True,
    }
    for name, expected in expected_supervision.items():
        if supervision.get(name) != expected:
            raise PriorTangentInferenceError(
                f"training v5 supervision field differs: {name}"
            )

    dataset = _require_mapping(receipt.get("dataset"), label="dataset identity")
    routing = _require_mapping(dataset.get("routing"), label="routing identity")
    if (
        dataset.get("rows") != 644
        or routing.get("default_tier") != "reject"
        or routing.get("explicit_route_counts")
        != {"full_pair": 0, "motion_only": 359, "reject": 285}
        or routing.get("file_sha256") != value.get("routing_file_sha256")
        or value.get("expected_routing_jsonl_sha256")
        != value.get("routing_file_sha256")
        or value.get("eligible_route_count") != 359
    ):
        raise PriorTangentInferenceError(
            "training did not use the hash-bound strict-359 cohort"
        )
    distributed = _require_mapping(receipt.get("distributed"), label="distributed contract")
    if (
        distributed.get("world_size") != 4
        or distributed.get("ulysses_size") != 4
        or distributed.get("same_pair_all_ranks") is not True
        or distributed.get("explicit_lora_gradient_all_reduce") is not True
    ):
        raise PriorTangentInferenceError("training distributed contract differs")
    if receipt.get("production_claim_forbidden") is not True:
        raise PriorTangentInferenceError("training receipt lost production restriction")
    if receipt.get("scientific_claim_authorized") is not False:
        raise PriorTangentInferenceError("training receipt carries unsupported claim")

    if adapter_config.get("peft_type") != "LORA":
        raise PriorTangentInferenceError("adapter is not LoRA")
    if adapter_config.get("r") != trainer.LORA_RANK:
        raise PriorTangentInferenceError("adapter rank differs")
    try:
        serialized_alpha = float(adapter_config.get("lora_alpha", -1))
        serialized_dropout = float(adapter_config.get("lora_dropout", -1))
    except (TypeError, ValueError) as error:
        raise PriorTangentInferenceError("adapter alpha/dropout are invalid") from error
    if serialized_alpha != trainer.LORA_ALPHA or serialized_dropout != 0.0:
        raise PriorTangentInferenceError("adapter alpha/dropout differ")
    if adapter_config.get("bias") != "none":
        raise PriorTangentInferenceError("adapter bias differs")
    if adapter_config.get("modules_to_save") not in (None, []):
        raise PriorTangentInferenceError("modules_to_save are forbidden")
    if adapter_config.get("use_dora") not in (None, False):
        raise PriorTangentInferenceError("DoRA is outside the v5 contract")
    if adapter_config.get("use_rslora") not in (None, False):
        raise PriorTangentInferenceError("RS-LoRA is outside the v5 contract")
    serialized = _validate_serialized_target_coverage(
        adapter_config.get("target_modules"), targets=targets
    )

    for name, pattern in (
        ("method_source_revision", _SHA1_RE),
        ("method_source_archive_sha256", _SHA256_RE),
    ):
        item = value.get(name)
        if not isinstance(item, str) or pattern.fullmatch(item) is None:
            raise PriorTangentInferenceError(f"training {name} is invalid")
    transformers_version = receipt.get("transformers_version")
    if not isinstance(transformers_version, str) or not transformers_version:
        raise PriorTangentInferenceError("training Transformers version is missing")
    expected_noop_sha = hashlib.sha256(
        motion.DEFAULT_NOOP_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    if value.get("noop_instruction_sha256") != expected_noop_sha:
        raise PriorTangentInferenceError("training semantic no-op differs")
    return {
        "receipt_digest": digest,
        "global_step": global_step,
        "scope": REQUIRED_LORA_SCOPE,
        "targets": list(targets),
        "serialized_target_modules": list(serialized),
        "target_modules_sha256": adapter["target_modules_sha256"],
        "initialization_digest": adapter["initialization_digest"],
        "checkpoint_parameter_digest": adapter["checkpoint_parameter_digest"],
        "transformers_version": transformers_version,
        "training_method_source_revision": value["method_source_revision"],
        "training_method_source_archive_sha256": value[
            "method_source_archive_sha256"
        ],
    }


def packed_to_phase_grid(packed: Any, *, layout: tri.PackedLatentLayout) -> Any:
    """Reshape exact Wan token order ``[B,N,D]`` to ``[B,21,S,D]``."""

    shape = tuple(int(value) for value in getattr(packed, "shape", ()))
    if shape != layout.packed_shape:
        raise PriorTangentInferenceError(
            f"packed field shape {shape} differs from {layout.packed_shape}"
        )
    if layout.frames != LATENT_PHASES or layout.tokens % LATENT_PHASES:
        raise PriorTangentInferenceError("v5 requires exactly 21 latent phases")
    cells = layout.tokens // LATENT_PHASES
    return packed.reshape(layout.batch, LATENT_PHASES, cells, layout.packed_channels)


def phase_grid_to_packed(grid: Any, *, layout: tri.PackedLatentLayout) -> Any:
    cells = layout.tokens // LATENT_PHASES
    expected = (layout.batch, LATENT_PHASES, cells, layout.packed_channels)
    shape = tuple(int(value) for value in getattr(grid, "shape", ()))
    if shape != expected:
        raise PriorTangentInferenceError(
            f"phase field shape {shape} differs from {expected}"
        )
    return grid.reshape(layout.packed_shape)


@dataclass(frozen=True)
class RawFourBranchStep:
    step_index: int
    timestep: Any
    timestep_float: float
    sigma: Any
    sigma_float: float
    model_id: str
    sample_packed: Any
    official_model_output: Any
    frozen_negative_velocity_packed: Any
    frozen_noop_velocity_packed: Any
    frozen_action_velocity_packed: Any
    adapted_action_velocity_packed: Optional[Any]
    source_phase: Any
    execution_arm: str
    adapter_loaded: bool
    apg: tri.APGParameters
    layout: tri.PackedLatentLayout


@dataclass(frozen=True)
class PriorTangentStepRecord:
    step_index: int
    timestep: float
    sigma: float
    gamma: float
    execution_arm: str
    model_id: str
    transformer_forwards: int
    frozen_negative_forwards: int
    frozen_noop_forwards: int
    frozen_action_forwards: int
    adapted_action_forwards: int
    original_scheduler_calls: int
    official_frozen_action_apg_exact: bool
    official_frozen_action_apg_rms_error: float
    official_frozen_action_apg_max_abs_error: float
    raw_prior_rms: float
    prior_phase0_rms: float
    q0_prior_rms: float
    raw_correction_rms: float
    trusted_correction_rms: float
    executed_correction_rms: float
    trusted_first_phase_max_abs: float
    phase_cells: int
    gamma_zero_exact_frozen_prior: bool
    adapter_correction_routed: bool
    adapter_loaded: bool


@dataclass
class PriorTangentTrace:
    execution_arm: str
    adapter_loaded: bool
    records: list[PriorTangentStepRecord] = field(default_factory=list)
    sample_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": runtime_contract(),
            "execution_arm": self.execution_arm,
            "adapter_loaded": self.adapter_loaded,
            "sample_calls": self.sample_calls,
            "step_count": len(self.records),
            "steps": [asdict(record) for record in self.records],
        }


def _tensor_stat(value: Any, *, label: str) -> float:
    try:
        result = float(value.detach().float().cpu().item())
    except Exception as error:
        raise PriorTangentInferenceError(f"cannot serialize {label}") from error
    if not math.isfinite(result) or result < 0.0:
        raise PriorTangentInferenceError(f"{label} must be finite and non-negative")
    return result


def project_prior_tangent_step(
    raw: RawFourBranchStep,
    *,
    base_action_momentum: Any,
    base_noop_momentum: Any,
    adapted_action_momentum: Optional[Any],
) -> tuple[tri.ProjectedVelocity, PriorTangentStepRecord]:
    """Reconstruct APG fields and apply the shared phasewise v5 operator."""

    import torch

    if raw.execution_arm not in EXECUTION_ARMS:
        raise PriorTangentInferenceError("unknown execution arm")
    if raw.apg.momentum != 0.0:
        raise PriorTangentInferenceError("v5 requires official APG momentum exactly zero")
    tensors = [
        raw.sample_packed,
        raw.official_model_output,
        raw.frozen_negative_velocity_packed,
        raw.frozen_noop_velocity_packed,
        raw.frozen_action_velocity_packed,
    ]
    if raw.adapter_loaded:
        if raw.adapted_action_velocity_packed is None:
            raise PriorTangentInferenceError("loaded adapter lacks adapted-action branch")
        tensors.append(raw.adapted_action_velocity_packed)
    elif raw.adapted_action_velocity_packed is not None:
        raise PriorTangentInferenceError("adapter-free diagnostic received adapted branch")
    if any(tuple(int(v) for v in tensor.shape) != raw.layout.packed_shape for tensor in tensors):
        raise PriorTangentInferenceError("four-branch packed shapes differ")
    sigma = raw.sigma
    if (
        not isinstance(sigma, torch.Tensor)
        or sigma.ndim != 0
        or sigma.device.type != "cpu"
        or sigma.dtype != torch.float32
        or not bool(torch.isfinite(sigma))
        or not bool(sigma > 0)
    ):
        raise PriorTangentInferenceError(
            "pinned UniPC sigma must be one positive CPU fp32 scalar"
        )

    sample = tri._packed_to_spatial(raw.sample_packed, raw.layout)
    official_v = tri._packed_to_spatial(raw.official_model_output, raw.layout)
    negative_v = tri._packed_to_spatial(
        raw.frozen_negative_velocity_packed, raw.layout
    )
    noop_v = tri._packed_to_spatial(raw.frozen_noop_velocity_packed, raw.layout)
    base_action_v = tri._packed_to_spatial(
        raw.frozen_action_velocity_packed, raw.layout
    )
    adapted_action_v = (
        tri._packed_to_spatial(raw.adapted_action_velocity_packed, raw.layout)
        if raw.adapter_loaded
        else None
    )
    negative_clean = tri.pinned_raw_condition_clean(sample, negative_v, sigma)
    noop_condition_clean = tri.pinned_raw_condition_clean(sample, noop_v, sigma)
    base_action_condition_clean = tri.pinned_raw_condition_clean(
        sample, base_action_v, sigma
    )
    guidance = raw.apg.guidance_scale_for(raw.model_id)
    base_action_clean = tri._normalized_guidance(
        base_action_condition_clean,
        negative_clean,
        guidance,
        base_action_momentum,
        raw.apg.eta,
        raw.apg.norm_threshold,
    )
    base_noop_clean = tri._normalized_guidance(
        noop_condition_clean,
        negative_clean,
        guidance,
        base_noop_momentum,
        raw.apg.eta,
        raw.apg.norm_threshold,
    )
    adapted_action_clean = None
    if raw.adapter_loaded:
        if adapted_action_momentum is None or adapted_action_v is None:
            raise PriorTangentInferenceError("adapted APG state is missing")
        adapted_action_condition_clean = tri.pinned_raw_condition_clean(
            sample, adapted_action_v, sigma
        )
        adapted_action_clean = tri._normalized_guidance(
            adapted_action_condition_clean,
            negative_clean,
            guidance,
            adapted_action_momentum,
            raw.apg.eta,
            raw.apg.norm_threshold,
        )

    rebuilt_base_velocity = tri._spatial_to_packed(
        (sample - base_action_clean) / sigma, raw.layout
    ).to(device=raw.official_model_output.device, dtype=raw.official_model_output.dtype)
    parity_error = rebuilt_base_velocity.float() - raw.official_model_output.float()
    parity_rms = _tensor_stat(parity_error.square().mean().sqrt(), label="APG parity RMS")
    parity_max = _tensor_stat(parity_error.abs().max(), label="APG parity maximum")
    parity_exact = bool(torch.equal(rebuilt_base_velocity, raw.official_model_output))
    if not parity_exact:
        raise PriorTangentInferenceError(
            "locally rebuilt frozen action APG differs from official model_output"
        )
    # The official field is authoritative after the exact certificate.
    base_action_clean = sample - sigma * official_v

    base_action_phase = packed_to_phase_grid(
        tri._spatial_to_packed(base_action_clean, raw.layout).float(), layout=raw.layout
    )
    base_noop_phase = packed_to_phase_grid(
        tri._spatial_to_packed(base_noop_clean, raw.layout).float(), layout=raw.layout
    )
    adapted_action_phase = (
        packed_to_phase_grid(
            tri._spatial_to_packed(adapted_action_clean, raw.layout).float(),
            layout=raw.layout,
        )
        if adapted_action_clean is not None
        else None
    )
    kappa_perp = (
        0.0
        if raw.execution_arm == "parallel_only"
        else tangent.DEFAULT_KAPPA_PERP
    )
    config = tangent.TangentTrustRegionConfig(
        kappa_parallel=tangent.DEFAULT_KAPPA_PARALLEL,
        kappa_perp=kappa_perp,
        epsilon=tangent.DEFAULT_EPSILON,
        phase_dim=1,
    )
    raw_prior = tangent.raw_frozen_prior(
        base_action_phase, base_noop_phase, phase_dim=1
    )
    prior = tangent.frozen_prior(
        base_action_phase, base_noop_phase, phase_dim=1
    )
    if not bool(torch.equal(prior, tangent.q0(raw_prior, phase_dim=1))):
        raise PriorTangentInferenceError("causal prior differs from Q0(raw prior)")
    source_phase = raw.source_phase
    if (
        tuple(int(value) for value in getattr(source_phase, "shape", ()))
        != tuple(int(value) for value in prior.shape)
        or source_phase.dtype != torch.float32
        or source_phase.device != prior.device
    ):
        raise PriorTangentInferenceError("source phase field differs from branch geometry")

    adapter_correction_routed = raw.execution_arm in ("main", "parallel_only")
    if adapter_correction_routed:
        if not raw.adapter_loaded or adapted_action_phase is None:
            raise PriorTangentInferenceError(
                "adapter-routed execution arm lacks an adapter branch"
            )
        raw_correction = tangent.adapter_correction(
            adapted_action_phase, base_action_phase, phase_dim=1
        )
        result = tangent.execute_prior_guided_field(
            prior, raw_correction, step_index=raw.step_index, config=config
        )
        trusted = result.trust_region.trusted_correction
        effective_gamma = float(result.gamma)
        executed_motion = result.executed_field
        gamma_zero_exact = (
            effective_gamma == 0.0 and executed_motion is result.prior
        )
    else:
        # Diagnostic arms deliberately do not construct or consume the LoRA
        # correction at the scheduler boundary.  The fourth forward is kept
        # only as a matched-compute certificate.
        raw_correction = torch.zeros_like(prior)
        trusted = torch.zeros_like(prior)
        effective_gamma = 0.0
        executed_motion = raw_prior if raw.execution_arm == "frozen_prior" else prior
        gamma_zero_exact = executed_motion is prior
    first_phase_max = _tensor_stat(
        trusted[:, :1].abs().max(), label="trusted first-phase maximum"
    )
    if first_phase_max != 0.0:
        raise PriorTangentInferenceError("trusted correction changed phase zero")

    if adapter_correction_routed and effective_gamma == 0.0 and not gamma_zero_exact:
        raise PriorTangentInferenceError("zero-gamma operator did not alias frozen prior")
    executed_phase = source_phase + executed_motion
    if raw.execution_arm == "frozen_prior":
        expected_phase0 = source_phase[:, :1] + raw_prior[:, :1]
    else:
        expected_phase0 = source_phase[:, :1]
    if not bool(torch.equal(executed_phase[:, :1], expected_phase0)):
        raise PriorTangentInferenceError("execution arm changed its phase-zero contract")
    executed_packed = phase_grid_to_packed(executed_phase, layout=raw.layout)
    executed_velocity = ((raw.sample_packed - executed_packed) / sigma).to(
        device=raw.official_model_output.device,
        dtype=raw.official_model_output.dtype,
    )
    if not bool(torch.isfinite(executed_velocity).all()):
        raise PriorTangentInferenceError("executed velocity is non-finite")

    record = PriorTangentStepRecord(
        step_index=raw.step_index,
        timestep=raw.timestep_float,
        sigma=raw.sigma_float,
        gamma=effective_gamma,
        execution_arm=raw.execution_arm,
        model_id=raw.model_id,
        transformer_forwards=4 if raw.adapter_loaded else 3,
        frozen_negative_forwards=1,
        frozen_noop_forwards=1,
        frozen_action_forwards=1,
        adapted_action_forwards=1 if raw.adapter_loaded else 0,
        original_scheduler_calls=1,
        official_frozen_action_apg_exact=parity_exact,
        official_frozen_action_apg_rms_error=parity_rms,
        official_frozen_action_apg_max_abs_error=parity_max,
        raw_prior_rms=_tensor_stat(
            raw_prior.square().mean().sqrt(), label="raw prior RMS"
        ),
        prior_phase0_rms=_tensor_stat(
            raw_prior[:, :1].square().mean().sqrt(), label="prior phase-zero RMS"
        ),
        q0_prior_rms=_tensor_stat(
            prior.square().mean().sqrt(), label="Q0 prior RMS"
        ),
        raw_correction_rms=_tensor_stat(
            raw_correction.square().mean().sqrt(), label="raw correction RMS"
        ),
        trusted_correction_rms=_tensor_stat(
            trusted.square().mean().sqrt(), label="trusted correction RMS"
        ),
        executed_correction_rms=_tensor_stat(
            (effective_gamma * trusted).square().mean().sqrt(),
            label="executed correction RMS",
        ),
        trusted_first_phase_max_abs=first_phase_max,
        phase_cells=raw.layout.tokens // LATENT_PHASES,
        gamma_zero_exact_frozen_prior=gamma_zero_exact,
        adapter_correction_routed=adapter_correction_routed,
        adapter_loaded=raw.adapter_loaded,
    )
    projected = tri.ProjectedVelocity(
        model_output=executed_velocity,
        correction_rms=_tensor_stat(
            executed_velocity.float().sub(raw.official_model_output.float()).square().mean().sqrt(),
            label="scheduler-boundary correction RMS",
        ),
        effective_guidance_scale=guidance,
        official_action_parity_rms_error=parity_rms,
        official_action_parity_max_abs_error=parity_max,
        official_action_exact_parity=parity_exact,
        sample_dtype=str(sample.dtype),
        branch_velocity_dtype=str(base_action_v.dtype),
        official_model_output_dtype=str(raw.official_model_output.dtype),
    )
    return projected, record


@dataclass
class _CapturedForward:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    bound: dict[str, Any]
    prediction: Any


@dataclass
class _ActiveSample:
    expected_steps: int
    action_t1: Any
    action_t2: Any
    uncond_t1: Any
    uncond_t2: Any
    noop_t1: Any
    noop_t2: Any
    apg: tri.APGParameters
    base_action_momentum: Any
    base_noop_momentum: Any
    adapted_action_momentum: Any
    pending_negative: Optional[_CapturedForward] = None
    pending_base_action: Optional[_CapturedForward] = None
    pending_noop: Optional[Any] = None
    pending_adapted_action: Optional[Any] = None
    integrated_steps: int = 0


def _branch_prompt(state: _ActiveSample, model_id: str, branch: str) -> Any:
    if model_id not in ("transformer_1", "transformer_2"):
        raise PriorTangentInferenceError(f"unexpected Bernini model_id {model_id!r}")
    suffix = "t1" if model_id == "transformer_1" else "t2"
    return getattr(state, f"{branch}_{suffix}")


class _InstalledFourBranch(tri._InstalledTriBranch):
    """Pinned tri-branch wrapper extended with one one-sided adapter branch."""

    def __init__(
        self,
        *args: Any,
        adapter_model: Any,
        source_clean: Any,
        execution_arm: str,
        **kwargs: Any,
    ) -> None:
        if adapter_model is not None and not callable(
            getattr(adapter_model, "disable_adapter", None)
        ):
            raise PriorTangentInferenceError(
                "PEFT model must expose reversible disable_adapter()"
            )
        self.adapter_model = adapter_model
        self.adapter_loaded = adapter_model is not None
        super().__init__(*args, **kwargs)
        if execution_arm not in EXECUTION_ARMS:
            raise PriorTangentInferenceError("unknown execution arm")
        source_packed = tri._spatial_to_packed(source_clean.float(), self.layout)
        self.source_phase = packed_to_phase_grid(
            source_packed.float(), layout=self.layout
        )
        self.execution_arm = execution_arm
        if execution_arm in ("main", "parallel_only") and not self.adapter_loaded:
            raise PriorTangentInferenceError(
                "adapter-routed execution arm requires an adapter model"
            )
        self.trace = PriorTangentTrace(
            execution_arm=execution_arm, adapter_loaded=self.adapter_loaded
        )
        self._active: Optional[_ActiveSample] = None

    def _call_frozen(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        with torch.no_grad():
            if self.adapter_model is None:
                return self._original_shared_step(*args, **kwargs)
            with self.adapter_model.disable_adapter():
                return self._original_shared_step(*args, **kwargs)

    def _call_adapted(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        if self.adapter_model is None:
            raise PriorTangentInferenceError("adapter-free diagnostic has no adapted call")
        with torch.no_grad():
            return self._original_shared_step(*args, **kwargs)

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise PriorTangentInferenceError("shared_step ran outside validated sample")
        bound = tri._bind_call(self._original_shared_step, args, kwargs)
        try:
            model_id = str(bound["model_id"])
            prompt = bound["cond_embeds"]
        except KeyError as error:
            raise PriorTangentInferenceError("shared_step lacks branch arguments") from error

        if state.pending_negative is None:
            if prompt is not _branch_prompt(state, model_id, "uncond"):
                raise PriorTangentInferenceError("negative prompt object differs")
            prediction = self._call_frozen(*args, **kwargs)
            state.pending_negative = _CapturedForward(
                tuple(args),
                dict(kwargs),
                bound,
                self._query_prediction(prediction, branch="frozen_negative"),
            )
            return prediction

        if any(
            value is not None
            for value in (
                state.pending_base_action,
                state.pending_noop,
                state.pending_adapted_action,
            )
        ):
            raise PriorTangentInferenceError(
                "more than two official shared_step calls occurred before UniPC"
            )
        negative = state.pending_negative
        if prompt is not _branch_prompt(state, model_id, "action"):
            raise PriorTangentInferenceError("action prompt object differs")
        if str(negative.bound.get("model_id")) != model_id:
            raise PriorTangentInferenceError("negative/action model_id differ")
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            tri._same_object(negative.bound.get(name), bound.get(name), label=name)
        tri._equal_metadata(
            negative.bound.get("batch_vae_seqlen"),
            bound.get("batch_vae_seqlen"),
            label="batch_vae_seqlen",
        )

        # Return the frozen action to the untouched vendor APG.  The adapted
        # action is an additional same-state query used only by our projector.
        base_action_prediction = self._call_frozen(*args, **kwargs)
        state.pending_base_action = _CapturedForward(
            tuple(args),
            dict(kwargs),
            bound,
            self._query_prediction(base_action_prediction, branch="frozen_action"),
        )
        if self.adapter_loaded:
            adapted_prediction = self._call_adapted(*args, **kwargs)
            state.pending_adapted_action = self._query_prediction(
                adapted_prediction, branch="adapted_action"
            )

        noop_prompt = _branch_prompt(state, model_id, "noop")
        noop_args, noop_kwargs = tri._replace_argument(
            self._original_shared_step,
            args,
            kwargs,
            name="cond_embeds",
            value=noop_prompt,
        )
        shape = getattr(noop_prompt, "shape", None)
        if shape is None or len(shape) < 2:
            raise PriorTangentInferenceError("noop embedding lacks [B,L,D]")
        noop_args, noop_kwargs = tri._replace_argument(
            self._original_shared_step,
            noop_args,
            noop_kwargs,
            name="batch_text_seqlen",
            value=[int(shape[1])],
        )
        noop_bound = tri._bind_call(self._original_shared_step, noop_args, noop_kwargs)
        for name in ("model_id", "noisy_latents", "timesteps", "rotary_embs"):
            if name == "model_id":
                tri._equal_metadata(
                    bound.get(name), noop_bound.get(name), label="noop model_id"
                )
            else:
                tri._same_object(
                    bound.get(name), noop_bound.get(name), label=f"action/noop {name}"
                )
        tri._equal_metadata(
            bound.get("batch_vae_seqlen"),
            noop_bound.get("batch_vae_seqlen"),
            label="action/noop batch_vae_seqlen",
        )
        noop_prediction = self._call_frozen(*noop_args, **noop_kwargs)
        state.pending_noop = self._query_prediction(
            noop_prediction, branch="frozen_noop"
        )
        return base_action_prediction

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise PriorTangentInferenceError("scheduler.step ran outside sample")
        required = [
            state.pending_negative,
            state.pending_base_action,
            state.pending_noop,
        ]
        if self.adapter_loaded:
            required.append(state.pending_adapted_action)
        if any(value is None for value in required):
            raise PriorTangentInferenceError("scheduler.step arrived before four branches")
        official = tri._extract_argument(args, kwargs, index=0, name="model_output")
        timestep = tri._extract_argument(args, kwargs, index=1, name="timestep")
        sample = tri._extract_argument(args, kwargs, index=2, name="sample")
        step_index, sigma, sigma_float = tri._resolve_sigma(self.scheduler, timestep)
        base_action = state.pending_base_action
        assert base_action is not None and state.pending_negative is not None
        raw = RawFourBranchStep(
            step_index=step_index,
            timestep=timestep,
            timestep_float=tri._coerce_scalar(timestep, label="timestep"),
            sigma=sigma,
            sigma_float=sigma_float,
            model_id=str(base_action.bound["model_id"]),
            sample_packed=sample,
            official_model_output=official,
            frozen_negative_velocity_packed=state.pending_negative.prediction,
            frozen_noop_velocity_packed=state.pending_noop,
            frozen_action_velocity_packed=base_action.prediction,
            adapted_action_velocity_packed=state.pending_adapted_action,
            source_phase=self.source_phase,
            execution_arm=self.execution_arm,
            adapter_loaded=self.adapter_loaded,
            apg=state.apg,
            layout=self.layout,
        )
        projected, record = project_prior_tangent_step(
            raw,
            base_action_momentum=state.base_action_momentum,
            base_noop_momentum=state.base_noop_momentum,
            adapted_action_momentum=state.adapted_action_momentum,
        )
        call_args, call_kwargs = tri._replace_argument(
            self._original_scheduler_step,
            args,
            kwargs,
            name="model_output",
            value=projected.model_output,
        )
        result = self._original_scheduler_step(*call_args, **call_kwargs)
        state.integrated_steps += 1
        self.trace.records.append(record)
        state.pending_negative = None
        state.pending_base_action = None
        state.pending_noop = None
        state.pending_adapted_action = None
        return result

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None:
            raise PriorTangentInferenceError("nested sample calls are forbidden")
        if self.diffusion.scheduler is not self.scheduler:
            raise PriorTangentInferenceError("diffusion scheduler changed")
        values = tri._bind_call(self._original_sample, args, kwargs)
        if values.get("guidance_mode") != "v2v_apg":
            raise PriorTangentInferenceError("v5 requires guidance_mode=v2v_apg")
        if int(values.get("num_inference_steps")) != tangent.NUM_DENOISING_STEPS:
            raise PriorTangentInferenceError("v5 requires exactly 40 UniPC steps")
        if not math.isclose(
            float(values.get("flow_shift")),
            self.expected_flow_shift,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise PriorTangentInferenceError("v5 requires flow shift 5")
        action_t1 = values.get("prompt_embeds")
        uncond_t1 = values.get("uncond_prompt_embeds")
        if action_t1 is None or uncond_t1 is None:
            raise PriorTangentInferenceError("action/negative embeddings are required")
        action_t2 = values.get("prompt_embeds_t2")
        if action_t2 is None:
            action_t2 = action_t1
        uncond_t2 = values.get("uncond_embeds_t2")
        if uncond_t2 is None:
            uncond_t2 = uncond_t1
        norm_threshold = values.get("norm_threshold")
        if isinstance(norm_threshold, (list, tuple)):
            norm_threshold = norm_threshold[0]
        apg = tri.APGParameters(
            guidance_scale=tri._coerce_scalar(values.get("omega_txt"), label="omega_txt"),
            omega_scale=tri._coerce_scalar(values.get("omega_scale"), label="omega_scale"),
            scale_transformer_2=getattr(self.diffusion, "transformer_2", None)
            is not None,
            eta=tri._coerce_scalar(values.get("eta"), label="eta"),
            norm_threshold=tri._coerce_scalar(norm_threshold, label="norm_threshold"),
            momentum=tri._coerce_scalar(values.get("momentum"), label="momentum"),
        )
        if apg.momentum != 0.0:
            raise PriorTangentInferenceError("official APG momentum must be exactly zero")
        state = _ActiveSample(
            expected_steps=self.expected_steps,
            action_t1=action_t1,
            action_t2=action_t2,
            uncond_t1=uncond_t1,
            uncond_t2=uncond_t2,
            noop_t1=self.noop_t1,
            noop_t2=self.noop_t2,
            apg=apg,
            base_action_momentum=tri._MomentumBuffer(0.0, branch="frozen_action"),
            base_noop_momentum=tri._MomentumBuffer(0.0, branch="frozen_noop"),
            adapted_action_momentum=(
                tri._MomentumBuffer(0.0, branch="adapted_action")
                if self.adapter_loaded
                else None
            ),
        )
        self._active = state
        records_before = len(self.trace.records)
        try:
            result = self._original_sample(*args, **kwargs)
            if any(
                value is not None
                for value in (
                    state.pending_negative,
                    state.pending_base_action,
                    state.pending_noop,
                    state.pending_adapted_action,
                )
            ):
                raise PriorTangentInferenceError("sample returned with incomplete branches")
            if state.integrated_steps != self.expected_steps:
                raise PriorTangentInferenceError("sample UniPC step count differs")
            if len(self.trace.records) - records_before != self.expected_steps:
                raise PriorTangentInferenceError("trace/UniPC step counts differ")
            buffers = [
                state.base_action_momentum,
                state.base_noop_momentum,
            ]
            if self.adapter_loaded:
                buffers.append(state.adapted_action_momentum)
            for buffer in buffers:
                if buffer.update_count != self.expected_steps:
                    raise PriorTangentInferenceError("APG branch count differs")
            self.trace.sample_calls += 1
            return result
        finally:
            self._active = None


@contextmanager
def four_branch_unipc_hook(
    renderer_or_diffusion: Any,
    *,
    adapter_model: Any,
    source_clean: Any,
    execution_arm: str,
    noop_prompt_embeds: Any,
    latent_shape: Sequence[int],
    bernini_commit: str,
    wan_diffusion_path: str | Path,
    expected_steps: int = tangent.NUM_DENOISING_STEPS,
    expected_flow_shift: float = 5.0,
) -> Iterator[PriorTangentTrace]:
    diffusion = tri.resolve_diffusion_core(renderer_or_diffusion)
    bridge = _InstalledFourBranch(
        diffusion,
        adapter_model=adapter_model,
        source_clean=source_clean,
        execution_arm=execution_arm,
        noop_prompt_embeds=noop_prompt_embeds,
        noop_prompt_embeds_t2=None,
        latent_shape=latent_shape,
        clean_field_callback=lambda fields: fields.action_guided_clean,
        expected_steps=expected_steps,
        expected_flow_shift=expected_flow_shift,
        projector=tri.project_clean_fields,
        bernini_commit=bernini_commit,
        wan_diffusion_path=wan_diffusion_path,
    )
    bridge.install()
    try:
        yield bridge.trace
    finally:
        bridge.restore()


def validate_runtime_schedule_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "timesteps": list(sigma_strata.PINNED_TIMESTEPS),
        "positive_sigmas": list(sigma_strata.PINNED_POSITIVE_SIGMAS),
        "positive_sigmas_float32_be_hex": list(
            sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
        ),
        "terminal_sigma": 0.0,
        "terminal_sigma_float32_be_hex": sigma_strata.TERMINAL_SIGMA_FLOAT32_HEX,
    }
    if not isinstance(audit, Mapping) or dict(audit) != expected:
        raise PriorTangentInferenceError("runtime UniPC schedule differs")
    return expected


def validate_execution_trace(
    trace: PriorTangentTrace,
    *,
    execution_arm: str,
    adapter_loaded: bool,
    runtime_schedule_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if execution_arm not in EXECUTION_ARMS:
        raise PriorTangentInferenceError("unknown execution arm")
    if not isinstance(trace, PriorTangentTrace) or trace.sample_calls != 1:
        raise PriorTangentInferenceError("v5 hook must observe exactly one sample")
    if trace.execution_arm != execution_arm:
        raise PriorTangentInferenceError("trace execution arm differs")
    if type(adapter_loaded) is not bool or trace.adapter_loaded is not adapter_loaded:
        raise PriorTangentInferenceError("trace adapter-loaded state differs")
    if execution_arm in ("main", "parallel_only") and not adapter_loaded:
        raise PriorTangentInferenceError("adapter-routed trace has no adapter")
    if len(trace.records) != tangent.NUM_DENOISING_STEPS:
        raise PriorTangentInferenceError("v5 must certify all 40 UniPC steps")
    audited = validate_runtime_schedule_audit(runtime_schedule_audit)
    sigmas: list[float] = []
    for expected_index, record in enumerate(trace.records):
        selected = sigma_strata.select_sigma_stratum(expected_index)
        try:
            sigma_strata.assert_selected_timestep_sigma(
                timestep=record.timestep, sigma=record.sigma, selected=selected
            )
        except sigma_strata.InferenceSigmaStrataError as error:
            raise PriorTangentInferenceError(
                f"runtime schedule differs at step {expected_index}"
            ) from error
        if record.step_index != expected_index or record.model_id != "transformer_1":
            raise PriorTangentInferenceError("v5 step order/expert differs")
        expected_forwards = 4 if adapter_loaded else 3
        if (
            record.transformer_forwards != expected_forwards
            or record.frozen_negative_forwards != 1
            or record.frozen_noop_forwards != 1
            or record.frozen_action_forwards != 1
            or record.adapted_action_forwards != (1 if adapter_loaded else 0)
            or record.original_scheduler_calls != 1
        ):
            raise PriorTangentInferenceError("v5 four-branch call count differs")
        if (
            record.official_frozen_action_apg_exact is not True
            or record.official_frozen_action_apg_rms_error != 0.0
            or record.official_frozen_action_apg_max_abs_error != 0.0
        ):
            raise PriorTangentInferenceError("frozen action APG certificate failed")
        expected_gamma = (
            tangent.correction_gamma(expected_index)
            if execution_arm in ("main", "parallel_only")
            else 0.0
        )
        if record.gamma != expected_gamma:
            raise PriorTangentInferenceError("v5 gamma schedule differs")
        if record.execution_arm != execution_arm:
            raise PriorTangentInferenceError("step execution arm differs")
        if record.adapter_loaded is not adapter_loaded:
            raise PriorTangentInferenceError("step adapter-loaded state differs")
        if record.phase_cells <= 0 or record.trusted_first_phase_max_abs != 0.0:
            raise PriorTangentInferenceError("phasewise correction contract differs")
        expected_routed = execution_arm in ("main", "parallel_only")
        if record.adapter_correction_routed is not expected_routed:
            raise PriorTangentInferenceError("adapter correction routing differs")
        if expected_routed:
            if (
                expected_gamma == 0.0
                and record.gamma_zero_exact_frozen_prior is not True
            ):
                raise PriorTangentInferenceError(
                    "zero-gamma step is not exact frozen-prior replay"
                )
            if (
                expected_gamma != 0.0
                and record.gamma_zero_exact_frozen_prior is not False
            ):
                raise PriorTangentInferenceError(
                    "nonzero-gamma step claims frozen-prior replay"
                )
        elif execution_arm == "frozen_prior":
            if record.gamma_zero_exact_frozen_prior is not False:
                raise PriorTangentInferenceError(
                    "raw-prior diagnostic incorrectly claimed causal-prior replay"
                )
        elif record.gamma_zero_exact_frozen_prior is not True:
            raise PriorTangentInferenceError(
                "causal-prior arm did not execute the formal frozen prior"
            )
        for item in (
            record.sigma,
            record.raw_prior_rms,
            record.prior_phase0_rms,
            record.q0_prior_rms,
            record.raw_correction_rms,
            record.trusted_correction_rms,
            record.executed_correction_rms,
        ):
            if not math.isfinite(float(item)) or float(item) < 0.0:
                raise PriorTangentInferenceError("v5 trace diagnostic is invalid")
        sigmas.append(float(record.sigma))
    if any(following >= current for current, following in zip(sigmas, sigmas[1:])):
        raise PriorTangentInferenceError("UniPC sigmas must strictly descend")
    payload = {
        "prior_guided_tangent": trace.as_dict(),
        "runtime_unipc_schedule_audit": audited,
        "certificate": {
            "execution_arm": execution_arm,
            "adapter_loaded": adapter_loaded,
            "step_count": tangent.NUM_DENOISING_STEPS,
            "original_unipc_calls": tangent.NUM_DENOISING_STEPS,
            "transformer_forwards": expected_forwards
            * tangent.NUM_DENOISING_STEPS,
            "frozen_adapter_disabled_forwards": (
                3 * tangent.NUM_DENOISING_STEPS if adapter_loaded else 0
            ),
            "frozen_base_forwards_without_adapter": (
                3 * tangent.NUM_DENOISING_STEPS if not adapter_loaded else 0
            ),
            "adapted_action_forwards": (
                tangent.NUM_DENOISING_STEPS if adapter_loaded else 0
            ),
            "gamma_zero_exact_frozen_steps": sum(
                record.gamma_zero_exact_frozen_prior for record in trace.records
            ),
            "adapter_correction_routed": execution_arm in ("main", "parallel_only"),
            "custom_integrator": False,
            "mask_flow_pose_inputs": False,
        },
    }
    payload["trace_digest"] = trainer.object_sha256(payload)
    return payload


def _strict_load_adapter(
    *,
    base_model: Any,
    bundle: Any,
    adapter_config: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> tuple[Any, int, int]:
    targets = list(identity["targets"])
    _validate_serialized_target_coverage(
        adapter_config.get("target_modules"), targets=targets
    )
    try:
        model, tensor_count = adapter_loader._strict_load_adapter(
            base_model=base_model,
            adapter_dir=bundle.adapter_dir,
            adapter_model_path=bundle.adapter_model_path,
            targets=targets,
        )
    except adapter_loader.DeltaInferenceError as error:
        raise PriorTangentInferenceError(str(error)) from error
    mapped: list[str] = []
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        scaling = getattr(module, "scaling", None)
        if lora_a is None and lora_b is None:
            continue
        try:
            complete = (
                "default" in lora_a and "default" in lora_b and "default" in scaling
            )
        except (TypeError, AttributeError):
            complete = False
        if not complete:
            raise PriorTangentInferenceError("runtime LoRA module is incomplete")
        matches = [
            target for target in targets if name == target or name.endswith(f".{target}")
        ]
        if len(matches) != 1:
            raise PriorTangentInferenceError("runtime LoRA module exceeds receipt scope")
        if float(scaling["default"]) != ADAPTER_SCALE:
            raise PriorTangentInferenceError("runtime LoRA scale differs from one")
        mapped.append(matches[0])
    if sorted(mapped) != targets or len(mapped) != len(set(mapped)):
        raise PriorTangentInferenceError("runtime active LoRA modules differ")
    if tensor_count != 2 * len(targets):
        raise PriorTangentInferenceError("runtime LoRA tensor count differs")
    named_lora = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if trainer.is_lora_parameter_name(name)
    ]
    try:
        digest = v5_train.v4._checkpoint_parameter_digest(named_lora)
    except Exception as error:
        raise PriorTangentInferenceError("cannot hash reloaded LoRA parameters") from error
    if digest != identity["checkpoint_parameter_digest"]:
        raise PriorTangentInferenceError("reloaded LoRA parameter digest differs")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise PriorTangentInferenceError("inference model contains trainable parameters")
    model.eval()
    return model, tensor_count, len(mapped)


def _method_hashes() -> dict[str, str]:
    paths = {
        "infer_prior_tangent_lora.py": METHOD_ROOT / "infer_prior_tangent_lora.py",
        "train_prior_tangent_lora.py": METHOD_ROOT / "train_prior_tangent_lora.py",
        "prior_guided_tangent.py": METHOD_ROOT / "prior_guided_tangent.py",
        "tri_branch_unipc.py": METHOD_ROOT / "tri_branch_unipc.py",
        "inference_sigma_strata.py": METHOD_ROOT / "inference_sigma_strata.py",
        "infer_delta_lora.py": METHOD_ROOT / "infer_delta_lora.py",
    }
    return {name: frozen.base.file_sha256(path) for name, path in paths.items()}


def build_inference_receipt(
    *,
    args: argparse.Namespace,
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    output_path: Path,
    output_sha256: str,
    noop_identity: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    inference_file_hashes: Mapping[str, str],
    wan_diffusion_path: Path,
    wan_diffusion_sha256: str,
    runtime_versions: Mapping[str, str],
    adapter_bundle: Optional[Any],
    adapter_identity: Optional[Mapping[str, Any]],
    adapter_config_sha256: Optional[str],
    adapter_model_sha256: Optional[str],
    training_receipt_file_sha256: Optional[str],
    adapter_tensor_count: int,
    active_lora_module_count: int,
) -> dict[str, Any]:
    audited_schedule = validate_runtime_schedule_audit(
        execution_trace.get("runtime_unipc_schedule_audit", {})
    )
    receipt = frozen.build_inference_receipt(
        args=args,
        source_path=source_path,
        source_sha256=source_sha256,
        source_metadata=source_metadata,
        output_path=output_path,
        output_sha256=output_sha256,
        noop_identity=noop_identity,
        execution_trace=execution_trace,
        bernini_revision=bernini_revision,
        veomni_revision=veomni_revision,
        inference_file_hashes=inference_file_hashes,
        wan_diffusion_path=wan_diffusion_path,
        wan_diffusion_sha256=wan_diffusion_sha256,
        runtime_versions=runtime_versions,
    )
    receipt.pop("receipt_digest", None)
    receipt["schema_version"] = INFERENCE_RECEIPT_SCHEMA
    receipt["method"] = METHOD_NAME
    receipt["method_files_sha256"] = _method_hashes()
    adapter_loaded = adapter_bundle is not None
    if adapter_loaded != (adapter_identity is not None):
        raise PriorTangentInferenceError("adapter bundle/identity presence differs")
    if args.execution_arm in ("main", "parallel_only") and not adapter_loaded:
        raise PriorTangentInferenceError("main/parallel receipt requires adapter")
    receipt["base_model"].update(
        {
            "frozen": True,
            "base_weights_frozen": True,
            "lora_or_peft_loaded": adapter_loaded,
            "adapter_loaded": adapter_loaded,
            "all_runtime_parameters_require_grad_false": True,
        }
    )
    if adapter_loaded:
        assert adapter_identity is not None and adapter_bundle is not None
        receipt["adapter"] = {
            "loaded": True,
            "checkpoint_root": str(adapter_bundle.checkpoint_root),
            "adapter_config_path": str(adapter_bundle.adapter_config_path),
            "adapter_config_sha256": adapter_config_sha256,
            "adapter_model_path": str(adapter_bundle.adapter_model_path),
            "adapter_model_sha256": adapter_model_sha256,
            "training_receipt_path": str(adapter_bundle.training_receipt_path),
            "training_receipt_file_sha256": training_receipt_file_sha256,
            "training_receipt_digest": adapter_identity["receipt_digest"],
            "training_global_step": adapter_identity["global_step"],
            "training_method_source_revision": adapter_identity[
                "training_method_source_revision"
            ],
            "training_method_source_archive_sha256": adapter_identity[
                "training_method_source_archive_sha256"
            ],
            "scope": adapter_identity["scope"],
            "target_module_count": len(adapter_identity["targets"]),
            "target_modules_sha256": adapter_identity["target_modules_sha256"],
            "serialized_target_modules": list(
                adapter_identity["serialized_target_modules"]
            ),
            "initialization_digest": adapter_identity["initialization_digest"],
            "checkpoint_parameter_digest": adapter_identity[
                "checkpoint_parameter_digest"
            ],
            "tensor_count": int(adapter_tensor_count),
            "active_lora_module_count": int(active_lora_module_count),
            "strict_tensor_reload_equal": True,
            "parameter_digest_verified_after_safetensors_reload": True,
            "target_modules_rebound_from_receipt": True,
            "merged": False,
            "scale": ADAPTER_SCALE,
        }
    else:
        receipt["adapter"] = {
            "loaded": False,
            "checkpoint_root": None,
            "tensor_count": 0,
            "active_lora_module_count": 0,
            "correction_available": False,
        }
    receipt["training_inference_alignment"] = {
        "training_receipt_schema": (
            v5_train.RECEIPT_SCHEMA if adapter_loaded else None
        ),
        "training_method": METHOD_NAME if adapter_loaded else None,
        "four_same_state_branches": adapter_loaded,
        "three_frozen_same_state_branches": True,
        "frozen_negative_noop_action_adapter_disabled": adapter_loaded,
        "frozen_negative_noop_action_base_only": not adapter_loaded,
        "adapted_action_adapter_enabled": adapter_loaded,
        "all_inference_branch_forwards_no_grad": True,
        "apg_momentum": 0.0,
        "packed_to_phase_shape": "[B,N,D]->[B,21,S,D]",
        "shared_operator_file": "prior_guided_tangent.py",
        "raw_prior_formula": "B_raw=A0-N0",
        "prior_formula": "B=Q0(B_raw)",
        "adapter_correction_formula": "R=Q0(A_theta-A0)",
        "executed_field_formula": "E=B+gamma(step)*C_B(R)",
        "gamma_schedule": list(tangent.correction_gamma_schedule()),
        "gamma_zero_exact_frozen_velocity": True,
        "runtime_sigma_schedule_sha256": audited_schedule["schedule_sha256"],
        "training_sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "mask_flow_pose_train_test_gap": False,
        "first_frame_anchor": False,
        "execution_arm": args.execution_arm,
        "main_method_claim": args.execution_arm == "main",
        "diagnostic_ablation": args.execution_arm != "main",
    }
    receipt["sampling"].pop("router_config", None)
    receipt["sampling"].pop("routing_contract", None)
    receipt["sampling"].update(
        {
            "adapter_loaded": adapter_loaded,
            "adapter_scale": ADAPTER_SCALE if adapter_loaded else None,
            "adapter_merged": False,
            "four_branch_contract": runtime_contract(),
            "execution_arm": args.execution_arm,
            "execution_arm_formula": runtime_contract()["execution_arms"][
                args.execution_arm
            ],
            "main_method_claim": args.execution_arm == "main",
            "diagnostic_ablation": args.execution_arm != "main",
            "transformer_forwards_per_step": 4 if adapter_loaded else 3,
            "legacy_binary_router": False,
            "runtime_unipc_schedule_audit": audited_schedule,
        }
    )
    receipt["receipt_digest"] = trainer.object_sha256(receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    frozen.configure_rank_local_caches()
    requested_source = Path(args.source_video).expanduser()
    if not requested_source.is_absolute():
        raise PriorTangentInferenceError("source video must be absolute")
    try:
        source_path = frozen.base._plain_file(
            requested_source.resolve(strict=True), label="source video"
        )
        output_path, receipt_path = frozen.base._resolve_output(args.output)
    except frozen.base.InferenceContractError as error:
        raise PriorTangentInferenceError(str(error)) from error
    adapter_loaded = args.adapter_checkpoint is not None
    bundle = None
    adapter_config = None
    identity = None
    adapter_config_sha256 = None
    adapter_model_sha256 = None
    training_receipt_file_sha256 = None
    if adapter_loaded:
        try:
            bundle = frozen.base.resolve_adapter_bundle(args.adapter_checkpoint)
        except frozen.base.InferenceContractError as error:
            raise PriorTangentInferenceError(str(error)) from error
        adapter_config = _read_json(bundle.adapter_config_path, label="adapter config")
        training_receipt = _read_json(
            bundle.training_receipt_path, label="training receipt"
        )
        identity = validate_training_adapter_contract(
            adapter_config,
            training_receipt,
            expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        )
        if (
            args.method_source_revision != identity["training_method_source_revision"]
            or args.method_source_archive_sha256
            != identity["training_method_source_archive_sha256"]
        ):
            raise PriorTangentInferenceError(
                "inference source archive must exactly match training archive"
            )
        adapter_config_sha256 = frozen.base.file_sha256(bundle.adapter_config_path)
        adapter_model_sha256 = frozen.base.file_sha256(bundle.adapter_model_path)
        training_receipt_file_sha256 = frozen.base.file_sha256(
            bundle.training_receipt_path
        )

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = trainer.validate_checkpoint(args.checkpoint)
        inference_file_hashes = frozen.base.validate_inference_source_files(bernini_root)
    except (frozen.base.InferenceContractError, trainer.TrainingContractError) as error:
        raise PriorTangentInferenceError(str(error)) from error
    if transformer_config["num_attention_heads"] % frozen.base.ULYSSES_SIZE:
        raise PriorTangentInferenceError("attention heads are not divisible by Ulysses=4")
    wan_diffusion_path = (
        bernini_root / "bernini/models/wan_diffusion.py"
    ).resolve(strict=True)
    try:
        wan_diffusion_sha256 = tri.validate_runtime_source_identity(
            bernini_commit=bernini_revision, wan_diffusion_path=wan_diffusion_path
        )
    except tri.TriBranchHookError as error:
        raise PriorTangentInferenceError(str(error)) from error
    trainer.activate_source_trees(bernini_root, veomni_root)

    import peft
    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if identity is not None and transformers_version != identity["transformers_version"]:
        raise PriorTangentInferenceError("Transformers version differs from training")
    if SYSTEM_PROMPTS.get("mv2v") != frozen.base.MV2V_SYSTEM_PROMPT:
        raise PriorTangentInferenceError("runtime mv2v system prompt differs")
    if DEFAULT_NEG_PROMPT != frozen.base.DEFAULT_NEGATIVE_PROMPT:
        raise PriorTangentInferenceError("runtime negative prompt differs")
    distributed = frozen.base.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise PriorTangentInferenceError("v5 requires four AUH ROCm-visible GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    try:
        source_tensor, source_metadata = frozen.base.prepare_exact_source(source_path)
    except frozen.base.InferenceContractError as error:
        raise PriorTangentInferenceError(str(error)) from error
    source_sha256 = frozen.base.file_sha256(source_path)
    action_prompt = frozen.base.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = frozen.base.build_training_prompt(
        motion.DEFAULT_NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **frozen.base.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except trainer.TrainingContractError as error:
        raise PriorTangentInferenceError(str(error)) from error
    if float(config.shift) != frozen.base.FLOW_SHIFT or config.use_unipc is not True:
        raise PriorTangentInferenceError("renderer must use official shift-5 UniPC")
    base_model = BerniniRendererModel(config)
    if any("lora_" in name.lower() for name, _ in base_model.named_modules()):
        raise PriorTangentInferenceError("base renderer unexpectedly contains LoRA")
    base_model.requires_grad_(False)
    base_model.eval()
    if adapter_loaded:
        assert bundle is not None and adapter_config is not None and identity is not None
        model, adapter_tensor_count, active_lora_module_count = _strict_load_adapter(
            base_model=base_model,
            bundle=bundle,
            adapter_config=adapter_config,
            identity=identity,
        )
        renderer = model.get_base_model()
    else:
        model = None
        adapter_tensor_count = 0
        active_lora_module_count = 0
        renderer = base_model
    renderer.requires_grad_(False)
    renderer.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **frozen.base.tokenizer_load_kwargs()
    )
    if (
        tokenizer.padding_side != "right"
        or tokenizer.init_kwargs.get("fix_mistral_regex") is not True
    ):
        raise PriorTangentInferenceError("tokenizer contract differs")
    action_ids, action_mask = frozen.base._tokenize_training_prompt(tokenizer, action_prompt)
    noop_ids, noop_mask = frozen.base._tokenize_training_prompt(tokenizer, noop_prompt)
    negative_ids, negative_mask = frozen.base._tokenize_renderer_negative(
        tokenizer, frozen.base.DEFAULT_NEGATIVE_PROMPT
    )

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    with torch.no_grad():
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        )
    bucket = source_metadata["source_derived_bucket_hw"]
    expected_latent_shape = (
        1,
        int(vae.config.z_dim),
        LATENT_PHASES,
        int(bucket[0]) // 8,
        int(bucket[1]) // 8,
    )
    if tuple(int(value) for value in source_latent.shape) != expected_latent_shape:
        raise PriorTangentInferenceError("source latent differs from exact 81f geometry")
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    noop_embeddings, noop_identity = frozen.encode_semantic_noop_prompt(
        renderer, noop_ids, noop_mask, device=device
    )
    sampling = frozen.exact_sampler_contract(seed=args.seed)
    if sampling.get("momentum") != 0.0 or sampling.get("num_frames") != 81:
        raise PriorTangentInferenceError("official sampler momentum/frame contract differs")
    try:
        diffusion = tri.resolve_diffusion_core(renderer)
        pre_schedule = sigma_strata.audit_runtime_unipc_schedule(
            diffusion.scheduler, initialize=True
        )
        with four_branch_unipc_hook(
            renderer,
            adapter_model=model,
            source_clean=source_latent,
            execution_arm=args.execution_arm,
            noop_prompt_embeds=noop_embeddings,
            latent_shape=expected_latent_shape,
            bernini_commit=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
            expected_steps=tangent.NUM_DENOISING_STEPS,
            expected_flow_shift=frozen.base.FLOW_SHIFT,
        ) as trace:
            with torch.no_grad():
                generated_latent = renderer.sample(
                    input_ids=action_ids.to(device),
                    attention_mask=action_mask.to(device),
                    uncond_input_ids=negative_ids.to(device),
                    uncond_attention_mask=negative_mask.to(device),
                    image_vae_latents=None,
                    multi_video_vae_latents=[source_latent],
                    multi_image_vae_latents=None,
                    width=int(bucket[1]),
                    height=int(bucket[0]),
                    device=device,
                    **sampling,
                )
        post_schedule = sigma_strata.audit_runtime_unipc_schedule(
            diffusion.scheduler, initialize=False
        )
    except (
        tri.TriBranchHookError,
        tangent.PriorGuidedTangentError,
        sigma_strata.InferenceSigmaStrataError,
    ) as error:
        raise PriorTangentInferenceError(str(error)) from error
    if pre_schedule != post_schedule:
        raise PriorTangentInferenceError("official sample changed pinned UniPC schedule")
    execution_trace = validate_execution_trace(
        trace,
        execution_arm=args.execution_arm,
        adapter_loaded=adapter_loaded,
        runtime_schedule_audit=post_schedule,
    )
    if tuple(int(value) for value in generated_latent.shape) != expected_latent_shape:
        raise PriorTangentInferenceError("generated latent differs from 81f geometry")
    if model is not None:
        model.to("cpu")
    else:
        renderer.to("cpu")
    del noop_embeddings, source_latent
    torch.cuda.empty_cache()

    if distributed.rank == 0:
        vae.to(device)
        with torch.no_grad():
            output = _vae_decode(vae, generated_latent)
        vae.to("cpu")
        expected_output_shape = (81, int(bucket[0]), int(bucket[1]), 3)
        if tuple(int(value) for value in output.shape) != expected_output_shape:
            raise PriorTangentInferenceError("decoded output differs from 81f geometry")
        temporary_output = output_path.with_name(
            f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
        )
        if temporary_output.exists() or temporary_output.is_symlink():
            raise PriorTangentInferenceError(f"stale temporary output: {temporary_output}")
        save_output(output, str(temporary_output), fps=int(frozen.base.FPS))
        os.replace(temporary_output, output_path)
        from tools import materialize_vae

        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(output_path)
        try:
            frozen.base.validate_exact_video_metadata(int(encoded.shape[0]), encoded_fps)
        except frozen.base.InferenceContractError as error:
            raise PriorTangentInferenceError(str(error)) from error
        if tuple(encoded_hw) != tuple(bucket):
            raise PriorTangentInferenceError("encoded output geometry differs")
        receipt = build_inference_receipt(
            args=args,
            source_path=source_path,
            source_sha256=source_sha256,
            source_metadata=source_metadata,
            output_path=output_path,
            output_sha256=frozen.base.file_sha256(output_path),
            noop_identity=noop_identity,
            execution_trace=execution_trace,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            inference_file_hashes=inference_file_hashes,
            wan_diffusion_path=wan_diffusion_path,
            wan_diffusion_sha256=wan_diffusion_sha256,
            runtime_versions={
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "transformers": transformers_version,
                "diffusers": diffusers_version,
                "peft": peft.__version__,
            },
            adapter_bundle=bundle,
            adapter_identity=identity,
            adapter_config_sha256=adapter_config_sha256,
            adapter_model_sha256=adapter_model_sha256,
            training_receipt_file_sha256=training_receipt_file_sha256,
            adapter_tensor_count=adapter_tensor_count,
            active_lora_module_count=active_lora_module_count,
        )
        frozen.base._atomic_write_json(receipt_path, receipt)
        print(frozen.base.canonical_json_bytes(receipt).decode("utf-8"), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
