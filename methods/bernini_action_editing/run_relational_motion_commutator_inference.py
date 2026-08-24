#!/usr/bin/env python3
"""End-to-end four-GPU AUH runner for Bernini Motion Commutator LoRA v7.

The deployed condition set is exactly one 81-frame source video and one action
instruction.  Paired target video and the T2V generator are training teachers
only and are never constructed here.  No mask, track, swept tube, optical flow,
pose, trajectory, or first-frame anchor argument exists.

The runner accepts only a completed receipt that the strict v7 loader declares
inference-ready.  A canary, a receipt with pending loader parity, or a checkpoint
without one complete 40-sigma cycle fails before model construction.

The training-matched trust-bound ratio is kappa=0.25.  Kappa 0.5 and 1.0 are
explicit inference-only ablations.  They are accepted for matched evaluation,
but the output receipt forbids treating them as the trained main arm.
For a trained V8 checkpoint, feasible-radius scale 1.0 remains the main arm;
2.5 and 4.0 scale both prior-derived terms (not the absolute floor) and are
separately hash-bound, inference-only diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tarfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_feasible_quotient_lora as v8_adapter  # noqa: E402
import infer_prior_tangent_lora as v5  # noqa: E402
import infer_relational_motion_commutator as rmc  # noqa: E402
import inference_sigma_strata as sigma_strata  # noqa: E402
import motion_commutator as commutator  # noqa: E402
import train_relational_motion_commutator_auh as v7_train  # noqa: E402


frozen = v5.frozen
trainer = v5.trainer
tri = v5.tri

METHOD_NAME = rmc.METHOD_NAME
INFERENCE_RECEIPT_SCHEMA = rmc.INFERENCE_RECEIPT_SCHEMA
V8_METHOD_NAME = rmc.V8_METHOD_NAME
V8_INFERENCE_RECEIPT_SCHEMA = rmc.V8_INFERENCE_RECEIPT_SCHEMA
NUM_FRAMES = rmc.NUM_FRAMES
LATENT_PHASES = rmc.LATENT_PHASES
NUM_INFERENCE_STEPS = rmc.NUM_DENOISING_STEPS
ULYSSES_SIZE = frozen.base.ULYSSES_SIZE
MAIN_KAPPA = 0.25
KAPPA_CHOICES = (0.25, 0.5, 1.0)
MAIN_V8_RADIUS_SCALE = rmc.MAIN_V8_RADIUS_SCALE
V8_RADIUS_SCALE_CHOICES = rmc.V8_RADIUS_SCALE_CHOICES
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class RelationalMotionCommutatorRunnerError(RuntimeError):
    """Raised before an invalid v7 run can publish an output or receipt."""


def _remove_inherited_option(
    parser: argparse.ArgumentParser, *, destination: str, default: Any
) -> None:
    matches = [action for action in parser._actions if action.dest == destination]
    if len(matches) != 1:
        raise RelationalMotionCommutatorRunnerError(
            f"inherited parser lacks exactly one {destination!r} option"
        )
    action = matches[0]
    parser._remove_action(action)
    for group in parser._action_groups:
        group_actions = getattr(group, "_group_actions", None)
        if group_actions is not None and action in group_actions:
            group_actions.remove(action)
    for option in action.option_strings:
        parser._option_string_actions.pop(option, None)
    parser.set_defaults(**{destination: default})
    if any(action.dest == destination for action in parser._actions):
        raise RelationalMotionCommutatorRunnerError(
            f"failed to fix inherited parser option {destination!r}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Expose only source-only v7 deployment controls."""

    parser = v5.build_parser()
    parser.description = (
        "Run Bernini-R 1.3B relational motion commutator v7 on one exact "
        "81-frame source"
    )
    for destination, default in (
        ("execution_arm", "main"),
        ("alpha", v5.ADAPTER_SCALE),
        ("max_generate_fraction", frozen.DEFAULT_GENERATE_CAP),
        ("energy_coverage", frozen.DEFAULT_ENERGY_COVERAGE),
    ):
        _remove_inherited_option(parser, destination=destination, default=default)
    adapter_actions = [
        action for action in parser._actions if action.dest == "adapter_checkpoint"
    ]
    if len(adapter_actions) != 1:
        raise RelationalMotionCommutatorRunnerError(
            "inherited parser lacks exactly one adapter checkpoint option"
        )
    adapter_actions[0].required = True
    adapter_actions[0].help = (
        "completed v7 checkpoint root (or its adapter/ directory); pending and "
        "canary receipts are rejected"
    )
    parser.add_argument(
        "--kappa",
        type=float,
        choices=KAPPA_CHOICES,
        default=MAIN_KAPPA,
        help=(
            "per-phase hard-bound ratio; 0.25 is the training-matched main arm, "
            "0.5/1.0 are inference-only ablations"
        ),
    )
    parser.add_argument(
        "--operator-mode",
        choices=rmc.OPERATOR_MODES,
        default=rmc.V7_RESIDUAL_ACTION_SECTION,
        help=(
            "v7 action-section residual or v8 reconstruction-section full "
            "quotient (V7-adapter diagnostic or V8-trained main arm)"
        ),
    )
    parser.add_argument(
        "--v8-radius-scale",
        type=float,
        choices=V8_RADIUS_SCALE_CHOICES,
        default=MAIN_V8_RADIUS_SCALE,
        help=(
            "scale both V8 learned-prior radius terms; 1.0 is the trained "
            "main arm and 2.5/4.0 are trained-V8 inference-only diagnostics"
        ),
    )
    parser.add_argument("--runtime-method-source-revision")
    parser.add_argument("--runtime-method-source-archive-sha256")
    parser.add_argument(
        "--runtime-method-source-archive",
        help="actual immutable runtime archive; its bytes are hashed in v8",
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    """Validate inherited source-only controls and immutable v7 geometry."""

    try:
        v5.validate_cli(args)
    except v5.PriorTangentInferenceError as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    if args.execution_arm != "main":
        raise RelationalMotionCommutatorRunnerError(
            "v7 runner supports only its commutator execution arm"
        )
    if not isinstance(args.adapter_checkpoint, str) or not args.adapter_checkpoint:
        raise RelationalMotionCommutatorRunnerError(
            "v7 inference requires --adapter-checkpoint"
        )
    if int(args.num_inference_steps) != NUM_INFERENCE_STEPS:
        raise RelationalMotionCommutatorRunnerError(
            "v7 requires exactly 40 official UniPC steps"
        )
    if (
        isinstance(args.kappa, bool)
        or not isinstance(args.kappa, (int, float))
        or not math.isfinite(float(args.kappa))
        or float(args.kappa) not in KAPPA_CHOICES
    ):
        raise RelationalMotionCommutatorRunnerError(
            "kappa must be one of 0.25, 0.5, 1.0"
        )
    if args.operator_mode not in rmc.OPERATOR_MODES:
        raise RelationalMotionCommutatorRunnerError("unknown operator mode")
    is_v8 = args.operator_mode == rmc.V8_RECONSTRUCTION_SECTION_FQT
    if is_v8 and float(args.kappa) != MAIN_KAPPA:
        raise RelationalMotionCommutatorRunnerError(
            "v8 reconstruction-section diagnostic fixes legacy kappa at 0.25"
        )
    if (
        isinstance(args.v8_radius_scale, bool)
        or not isinstance(args.v8_radius_scale, (int, float))
        or not math.isfinite(float(args.v8_radius_scale))
        or float(args.v8_radius_scale) not in V8_RADIUS_SCALE_CHOICES
    ):
        raise RelationalMotionCommutatorRunnerError(
            "V8 radius scale must be one of 1.0, 2.5, 4.0"
        )
    if not is_v8 and float(args.v8_radius_scale) != MAIN_V8_RADIUS_SCALE:
        raise RelationalMotionCommutatorRunnerError(
            "V7 requires --v8-radius-scale 1.0"
        )
    runtime_revision = args.runtime_method_source_revision
    runtime_archive = args.runtime_method_source_archive_sha256
    runtime_archive_path = args.runtime_method_source_archive
    if is_v8:
        if (
            not isinstance(runtime_revision, str)
            or _SHA1_RE.fullmatch(runtime_revision) is None
            or not isinstance(runtime_archive, str)
            or _SHA256_RE.fullmatch(runtime_archive) is None
            or not isinstance(runtime_archive_path, str)
            or not Path(runtime_archive_path).is_absolute()
        ):
            raise RelationalMotionCommutatorRunnerError(
                "v8 requires an absolute immutable runtime archive, revision, and hash"
            )
    elif any(
        value is not None
        for value in (runtime_revision, runtime_archive, runtime_archive_path)
    ):
        raise RelationalMotionCommutatorRunnerError(
            "v7 arm must not carry separate runtime source identity"
        )


def validate_runtime_method_source(
    args: argparse.Namespace,
    adapter_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a v8 receipt to the exact archive bytes executing this runner."""

    is_v8 = args.operator_mode == rmc.V8_RECONSTRUCTION_SECTION_FQT
    trained_v8 = bool(
        adapter_identity.get("training_receipt_schema")
        == v8_adapter.TRAINING_RECEIPT_SCHEMA
        and adapter_identity.get("training_method") == v8_adapter.METHOD_NAME
    )
    radius_scale = float(args.v8_radius_scale)
    if radius_scale != MAIN_V8_RADIUS_SCALE and not trained_v8:
        raise RelationalMotionCommutatorRunnerError(
            "non-unit V8 radius scales require a trained V8 checkpoint"
        )
    training_revision = adapter_identity["training_method_source_revision"]
    training_archive = adapter_identity[
        "training_method_source_archive_sha256"
    ]
    if not is_v8:
        return {
            "revision": training_revision,
            "archive_sha256": training_archive,
            "archive_path": None,
            "archive_hash_verified_by_runner": False,
            "differs_from_training_source": False,
            "matches_training_source": True,
        }
    try:
        archive_path = frozen.base._plain_file(
            Path(args.runtime_method_source_archive).resolve(strict=True),
            label="runtime method source archive",
        )
    except (OSError, frozen.base.InferenceContractError) as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    actual_hash = frozen.base.file_sha256(archive_path)
    if actual_hash != args.runtime_method_source_archive_sha256:
        raise RelationalMotionCommutatorRunnerError(
            "runtime source archive bytes differ from the declared SHA-256"
        )
    matches_training_source = bool(
        args.runtime_method_source_revision == training_revision
        and actual_hash == training_archive
    )
    if trained_v8:
        if (
            radius_scale == MAIN_V8_RADIUS_SCALE
            and not matches_training_source
        ):
            raise RelationalMotionCommutatorRunnerError(
                "unit-scale trained V8 runtime source must exactly match training"
            )
        if radius_scale != MAIN_V8_RADIUS_SCALE and (
            args.runtime_method_source_revision == training_revision
            or actual_hash == training_archive
        ):
            raise RelationalMotionCommutatorRunnerError(
                "scaled-radius trained V8 runtime revision and archive must "
                "both differ from training source"
            )
    elif (
        args.runtime_method_source_revision == training_revision
        or actual_hash == training_archive
    ):
        raise RelationalMotionCommutatorRunnerError(
            "v8 operator diagnostic source must differ from v7 training source"
        )
    current_method_hashes = _method_hashes()
    try:
        with tarfile.open(archive_path, mode="r:*") as handle:
            members = handle.getmembers()
            for short_name, expected_hash in current_method_hashes.items():
                archive_name = (
                    "methods/bernini_action_editing/" + short_name
                )
                matches = [
                    member for member in members if member.name == archive_name
                ]
                if len(matches) != 1 or not matches[0].isfile():
                    raise RelationalMotionCommutatorRunnerError(
                        f"runtime archive lacks one plain {archive_name}"
                    )
                stream = handle.extractfile(matches[0])
                if stream is None:
                    raise RelationalMotionCommutatorRunnerError(
                        f"cannot read runtime archive member {archive_name}"
                    )
                embedded_hash = hashlib.sha256(stream.read()).hexdigest()
                if embedded_hash != expected_hash:
                    raise RelationalMotionCommutatorRunnerError(
                        f"executing method file differs from archive member {archive_name}"
                    )
    except RelationalMotionCommutatorRunnerError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise RelationalMotionCommutatorRunnerError(
            f"cannot inspect runtime source archive: {error}"
        ) from error
    return {
        "revision": args.runtime_method_source_revision,
        "archive_sha256": actual_hash,
        "archive_path": str(archive_path),
        "archive_hash_verified_by_runner": True,
        "executing_method_files_verified_against_archive": True,
        "method_files_sha256": current_method_hashes,
        "differs_from_training_source": not matches_training_source,
        "matches_training_source": matches_training_source,
        "training_revision": training_revision,
        "training_archive_sha256": training_archive,
        "v8_radius_scale": radius_scale,
        "source_difference_allowed_for_radius_ablation": bool(
            trained_v8 and radius_scale != MAIN_V8_RADIUS_SCALE
        ),
    }


def is_inference_only_ablation(args: argparse.Namespace) -> bool:
    return (
        float(args.kappa) != MAIN_KAPPA
        or float(args.v8_radius_scale) != MAIN_V8_RADIUS_SCALE
        or args.operator_mode != rmc.V7_RESIDUAL_ACTION_SECTION
    )


def launcher_contract(
    operator_mode: str = rmc.V7_RESIDUAL_ACTION_SECTION,
    *,
    v8_training_matched: bool = False,
    v8_radius_scale: float = MAIN_V8_RADIUS_SCALE,
) -> dict[str, Any]:
    """Machine-readable four-rank AUH launch contract."""

    if type(v8_training_matched) is not bool:
        raise RelationalMotionCommutatorRunnerError(
            "launcher V8 training-match flag must be boolean"
        )
    try:
        feasible_config = rmc.feasible_quotient_config_for_radius_scale(
            v8_radius_scale
        )
        recovered_scale = rmc.validated_feasible_quotient_radius_scale(
            feasible_config,
            operator_mode=operator_mode,
        )
    except rmc.RelationalMotionCommutatorInferenceError as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    radius_ablation = bool(
        operator_mode == rmc.V8_RECONSTRUCTION_SECTION_FQT
        and recovered_scale != MAIN_V8_RADIUS_SCALE
    )
    if radius_ablation and not v8_training_matched:
        raise RelationalMotionCommutatorRunnerError(
            "radius-ablation launcher contract requires trained V8"
        )

    return {
        "launcher": "torchrun",
        "nproc_per_node": ULYSSES_SIZE,
        "world_size": ULYSSES_SIZE,
        "ulysses_size": ULYSSES_SIZE,
        "entrypoint": Path(__file__).name,
        "frames": NUM_FRAMES,
        "latent_phases": LATENT_PHASES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "required_external_conditions": ["source_video", "action_instruction"],
        "required_model_inputs": [
            "bernini_root",
            "veomni_root",
            "checkpoint",
            "adapter_checkpoint",
        ],
        "required_output": "output",
        "generator_loaded": False,
        "target_argument": False,
        "mask_flow_pose_track_anchor_arguments": False,
        "main_kappa": MAIN_KAPPA,
        "allowed_inference_kappa": list(KAPPA_CHOICES),
        "non_main_kappa_is_inference_only_ablation": True,
        "v8_radius_scale": recovered_scale,
        "allowed_v8_radius_scales": list(V8_RADIUS_SCALE_CHOICES),
        "non_unit_v8_radius_is_inference_only_ablation": True,
        "operator_mode": operator_mode,
        "allowed_operator_modes": list(rmc.OPERATOR_MODES),
        "v8_runtime_source_identity_separate_from_training_source": (
            operator_mode == rmc.V8_RECONSTRUCTION_SECTION_FQT
            and (not v8_training_matched or radius_ablation)
        ),
        "v8_runtime_source_identity_matches_training_source": (
            operator_mode == rmc.V8_RECONSTRUCTION_SECTION_FQT
            and v8_training_matched
            and not radius_ablation
        ),
        "v8_runtime_source_may_differ_for_radius_ablation": radius_ablation,
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelationalMotionCommutatorRunnerError(
            f"cannot read {label} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise RelationalMotionCommutatorRunnerError(
            f"{label} must contain one JSON object"
        )
    return value


def validate_release_training_receipt_status(receipt: Mapping[str, Any]) -> None:
    """Reject pending, canary, and pre-cycle receipts before heavy loading.

    The core strict loader performs the complete hash/schema/tensor validation.
    This cheap outer check makes the release-state policy visible in the E2E
    runner and prevents a pending checkpoint from reaching model construction.
    """

    if not isinstance(receipt, Mapping):
        raise RelationalMotionCommutatorRunnerError(
            "training receipt must be a mapping"
        )
    if receipt.get("inference_loader_parity_pending") is not False:
        raise RelationalMotionCommutatorRunnerError(
            "training receipt is pending inference-loader parity"
        )
    artifact = receipt.get("artifact_validation")
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("verified") is not True
        or artifact.get("status") != "post_save_strict_reload_complete"
    ):
        raise RelationalMotionCommutatorRunnerError(
            "training receipt lacks completed post-save strict reload"
        )
    if receipt.get("formal_40_sigma_cycle_complete") is not True:
        raise RelationalMotionCommutatorRunnerError(
            "training receipt is a canary or lacks a complete 40-sigma cycle"
        )
    global_step = receipt.get("global_step")
    if type(global_step) is not int or global_step < NUM_INFERENCE_STEPS:
        raise RelationalMotionCommutatorRunnerError(
            "training receipt has fewer than 40 accepted updates"
        )
    for flag in ("canary", "canary_only", "canary_checkpoint"):
        if receipt.get(flag) is True:
            raise RelationalMotionCommutatorRunnerError(
                f"training receipt carries forbidden {flag} status"
            )
    for key in ("checkpoint_status", "training_status", "release_status"):
        status = receipt.get(key)
        if isinstance(status, str) and status.strip().lower() in {
            "pending",
            "canary",
            "incomplete",
            "running",
        }:
            raise RelationalMotionCommutatorRunnerError(
                f"training receipt carries non-release {key}={status!r}"
            )


def is_trained_v8_receipt(receipt: Mapping[str, Any]) -> bool:
    """Classify only a self-consistent V8 schema/method pair."""

    if not isinstance(receipt, Mapping):
        raise RelationalMotionCommutatorRunnerError(
            "training receipt must be a mapping"
        )
    schema_match = (
        receipt.get("schema_version") == v8_adapter.TRAINING_RECEIPT_SCHEMA
    )
    method_match = receipt.get("method") == v8_adapter.METHOD_NAME
    if schema_match is not method_match:
        raise RelationalMotionCommutatorRunnerError(
            "training receipt mixes V7 and V8 schema/method identities"
        )
    return schema_match and method_match


def _method_hashes() -> dict[str, str]:
    paths = {
        "run_relational_motion_commutator_inference.py": Path(__file__).resolve(),
        "infer_relational_motion_commutator.py": METHOD_ROOT
        / "infer_relational_motion_commutator.py",
        "motion_commutator.py": METHOD_ROOT / "motion_commutator.py",
        "gauge_anchored_commutator.py": METHOD_ROOT
        / "gauge_anchored_commutator.py",
        "feasible_quotient_objective.py": METHOD_ROOT
        / "feasible_quotient_objective.py",
        "infer_feasible_quotient_lora.py": METHOD_ROOT
        / "infer_feasible_quotient_lora.py",
        "finalize_feasible_quotient_checkpoint.py": METHOD_ROOT
        / "finalize_feasible_quotient_checkpoint.py",
        "train_feasible_quotient_auh.py": METHOD_ROOT
        / "train_feasible_quotient_auh.py",
        "relational_commutator_objective.py": METHOD_ROOT
        / "relational_commutator_objective.py",
        "train_relational_motion_commutator_auh.py": METHOD_ROOT
        / "train_relational_motion_commutator_auh.py",
        "infer_prior_tangent_lora.py": METHOD_ROOT / "infer_prior_tangent_lora.py",
        "tri_branch_unipc.py": METHOD_ROOT / "tri_branch_unipc.py",
        "inference_sigma_strata.py": METHOD_ROOT / "inference_sigma_strata.py",
        "infer_delta_lora.py": METHOD_ROOT / "infer_delta_lora.py",
    }
    return {name: frozen.base.file_sha256(path) for name, path in paths.items()}


def _runtime_commutator_config(args: argparse.Namespace) -> Any:
    config = commutator.MotionCommutatorConfig(
        max_correction_increment_ratio=float(args.kappa),
        correction_increment_rms_floor=float(
            v7_train.MAIN_COMMUTATOR_CONFIG.correction_increment_rms_floor
        ),
        temporal_smoothing=True,
        epsilon=float(v7_train.MAIN_COMMUTATOR_CONFIG.epsilon),
    )
    config.validate()
    return config


def _runtime_feasible_quotient_config(args: argparse.Namespace) -> Any:
    try:
        config = rmc.feasible_quotient_config_for_radius_scale(
            float(args.v8_radius_scale)
        )
        rmc.validated_feasible_quotient_radius_scale(
            config,
            operator_mode=args.operator_mode,
        )
    except rmc.RelationalMotionCommutatorInferenceError as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    return config


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
    adapter_bundle: Any,
    adapter_identity: Mapping[str, Any],
    adapter_config_sha256: str,
    adapter_model_sha256: str,
    training_receipt_file_sha256: str,
    adapter_tensor_count: int,
    active_lora_module_count: int,
    runtime_source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a hash-bound v7 receipt and label kappa ablations."""

    audited_schedule = rmc.validate_runtime_schedule_audit(
        execution_trace.get("runtime_unipc_schedule_audit", {})
    )
    trained_v8 = bool(
        adapter_identity.get("training_receipt_schema")
        == v8_adapter.TRAINING_RECEIPT_SCHEMA
        and adapter_identity.get("training_method") == v8_adapter.METHOD_NAME
    )
    is_v8 = args.operator_mode == rmc.V8_RECONSTRUCTION_SECTION_FQT
    feasible_quotient_config = _runtime_feasible_quotient_config(args)
    radius_scale = rmc.validated_feasible_quotient_radius_scale(
        feasible_quotient_config,
        operator_mode=args.operator_mode,
    )
    radius_scale_ablation = bool(
        args.operator_mode == rmc.V8_RECONSTRUCTION_SECTION_FQT
        and radius_scale != MAIN_V8_RADIUS_SCALE
    )
    if radius_scale_ablation and not trained_v8:
        raise RelationalMotionCommutatorRunnerError(
            "radius-scale diagnostics require a trained V8 checkpoint"
        )
    if is_v8:
        runtime_matches_training = bool(
            runtime_source_identity.get("revision")
            == adapter_identity.get("training_method_source_revision")
            and runtime_source_identity.get("archive_sha256")
            == adapter_identity.get("training_method_source_archive_sha256")
        )
        if (
            runtime_source_identity.get("matches_training_source")
            is not runtime_matches_training
            or runtime_source_identity.get("differs_from_training_source")
            is runtime_matches_training
        ):
            raise RelationalMotionCommutatorRunnerError(
                "runtime/training source identity flags are inconsistent"
            )
        if trained_v8 and not radius_scale_ablation and not runtime_matches_training:
            raise RelationalMotionCommutatorRunnerError(
                "unit-scale trained V8 receipt requires matching runtime source"
            )
        if not trained_v8 and runtime_matches_training:
            raise RelationalMotionCommutatorRunnerError(
                "V7-adapter V8 falsification requires distinct runtime source"
            )
    expected_runtime_contract = rmc.runtime_contract(
        _runtime_commutator_config(args),
        operator_mode=args.operator_mode,
        feasible_quotient_config=feasible_quotient_config,
        v8_training_matched=trained_v8,
    )
    if execution_trace.get("contract") != expected_runtime_contract:
        raise RelationalMotionCommutatorRunnerError(
            "validated execution trace commutator config differs from CLI kappa"
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
    receipt["schema_version"] = (
        V8_INFERENCE_RECEIPT_SCHEMA if is_v8 else INFERENCE_RECEIPT_SCHEMA
    )
    receipt["method"] = V8_METHOD_NAME if is_v8 else METHOD_NAME
    if is_v8:
        receipt["method_source_revision"] = runtime_source_identity["revision"]
        receipt["method_source_archive_sha256"] = runtime_source_identity[
            "archive_sha256"
        ]
    receipt["method_files_sha256"] = _method_hashes()
    receipt["launcher_contract"] = launcher_contract(
        args.operator_mode,
        v8_training_matched=trained_v8,
        v8_radius_scale=radius_scale,
    )
    receipt["runtime_method_source"] = dict(runtime_source_identity)
    receipt["base_model"].update(
        {
            "frozen": True,
            "base_weights_frozen": True,
            "lora_or_peft_loaded": True,
            "adapter_loaded": True,
            "all_runtime_parameters_require_grad_false": True,
        }
    )
    receipt["input"].update(
        {
            "accepted_external_conditions": [
                "source_video",
                "action_instruction",
            ],
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "generator_prompt_argument": False,
            "external_mask_track_pose_flow_trajectory": False,
            "first_frame_anchor": False,
        }
    )
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
        "training_artifact_validation_digest": adapter_identity[
            "artifact_validation_digest"
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
        "scale": rmc.ADAPTER_SCALE,
    }
    release_schedule = list(commutator.release_rho_schedule())
    zero_release_steps = [
        index for index, rho in enumerate(release_schedule) if rho == 0.0
    ]
    receipt["training_inference_alignment"] = {
        "training_receipt_schema": (
            v8_adapter.TRAINING_RECEIPT_SCHEMA
            if trained_v8 else rmc.TRAINING_RECEIPT_SCHEMA
        ),
        "training_method": (
            v8_adapter.METHOD_NAME if trained_v8 else METHOD_NAME
        ),
        "training_teacher_loaded_at_inference": False,
        "inference_generator_forwards": 0,
        "five_same_state_editor_branches": True,
        "inference_forward_order": list(v7_train.INFERENCE_FORWARD_ORDER),
        "frozen_negative_noop_action_adapter_disabled": True,
        "adapted_noop_action_adapter_enabled_unmerged": True,
        "all_inference_branch_forwards_no_grad": True,
        "apg_momentum": 0.0,
        "packed_to_phase_shape": "[B,N,D]->[B,21,S,D]",
        "frozen_direction_formula": "B0=Q0(A0-N0)",
        "commutator_formula": (
            "qtheta=Q0(Atheta-Ntheta)"
            if trained_v8
            else "Ctheta=Q0((Atheta-Ntheta)-(A0-N0))"
        ),
        "operator_mode": args.operator_mode,
        "scheduler_clean_formula": (
            "frozen_noop_clean+rho*CenteredBound(FIR(Q0(Atheta-Ntheta)))"
            if is_v8
            else "frozen_action_clean+rho*BoundSmooth(Ctheta)"
        ),
        "appearance_carrier": (
            "frozen_noop_reconstruction_section"
            if is_v8
            else "frozen_action_clean"
        ),
        "hard_bound_formula": expected_runtime_contract["hard_bound_formula"],
        "training_kappa": None if trained_v8 else MAIN_KAPPA,
        "runtime_kappa": float(args.kappa),
        "runtime_kappa_training_matched": float(args.kappa) == MAIN_KAPPA,
        "runtime_kappa_inference_only_ablation": float(args.kappa) != MAIN_KAPPA,
        "v8_training_radius_scale": (
            MAIN_V8_RADIUS_SCALE if trained_v8 else None
        ),
        "v8_runtime_radius_scale": radius_scale if is_v8 else None,
        "v8_radius_scale_choices": (
            list(V8_RADIUS_SCALE_CHOICES) if is_v8 else None
        ),
        "v8_radius_scale_training_matched": not radius_scale_ablation,
        "v8_radius_scale_inference_only_ablation": radius_scale_ablation,
        "operator_training_matched": (
            (trained_v8 and not radius_scale_ablation) if is_v8 else True
        ),
        "overall_arm_training_matched": (
            (trained_v8 if is_v8 else True)
            and float(args.kappa) == MAIN_KAPPA
            and not radius_scale_ablation
        ),
        "release_schedule": release_schedule,
        "release_schedule_sha256": trainer.object_sha256(release_schedule),
        "zero_release_exact_official_model_output_steps": (
            [] if is_v8 else zero_release_steps
        ),
        "zero_release_noop_clean_section_steps": (
            zero_release_steps if is_v8 else []
        ),
        "v8_objective_is_projection_consistent": trained_v8,
        "v8_checkpoint_reuses_v7_adapter_for_falsification_only": (
            is_v8 and not trained_v8
        ),
        "v8_training_diffusion_query": (
            "target(beta=1)" if trained_v8 else None
        ),
        "runtime_sigma_schedule_sha256": audited_schedule["schedule_sha256"],
        "training_sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "mask_flow_pose_train_test_gap": False,
        "first_frame_anchor": False,
    }
    receipt["sampling"].pop("router_config", None)
    receipt["sampling"].pop("routing_contract", None)
    receipt["sampling"].pop("alpha", None)
    receipt["sampling"].update(
        {
            "adapter_loaded": True,
            "adapter_scale": rmc.ADAPTER_SCALE,
            "adapter_merged": False,
            "relational_motion_commutator_contract": expected_runtime_contract,
            "commutator_config": {
                "max_correction_increment_ratio": float(args.kappa),
                "correction_increment_rms_floor": float(
                    v7_train.MAIN_COMMUTATOR_CONFIG.correction_increment_rms_floor
                ),
                "temporal_smoothing": True,
                "epsilon": float(v7_train.MAIN_COMMUTATOR_CONFIG.epsilon),
            },
            "transformer_forwards_per_step": 5,
            "generator_forwards_per_step": 0,
            "legacy_binary_router": False,
            "runtime_unipc_schedule_audit": audited_schedule,
            "operator_mode": args.operator_mode,
            "feasible_quotient_config": (
                asdict(feasible_quotient_config)
                if is_v8
                else None
            ),
            "v8_radius_scale": radius_scale,
        }
    )
    overall_training_matched = bool(
        receipt["training_inference_alignment"][
            "overall_arm_training_matched"
        ]
    )
    receipt["evaluation_arm"] = (
        (
            (
                "v8_trained_feasible_radius_scale_inference_only_ablation"
                if radius_scale_ablation
                else "v8_projection_consistent_training_matched_main"
            )
            if trained_v8
            else "v8_reconstruction_section_fqt_falsification"
        )
        if is_v8
        else (
            "training_matched_main"
            if not is_inference_only_ablation(args)
            else "inference_only_kappa_ablation"
        )
    )
    receipt["inference_only_ablation"] = not overall_training_matched
    receipt["training_matched_main_arm"] = overall_training_matched
    receipt["training_matched"] = overall_training_matched
    receipt["v8_radius_scale_diagnostic"] = (
        {
            "enabled": radius_scale_ablation,
            "radius_scale": radius_scale,
            "audited_choices": list(V8_RADIUS_SCALE_CHOICES),
            "frozen_quotient_radius_ratio": float(
                feasible_quotient_config.frozen_quotient_radius_ratio
            ),
            "noop_dynamics_radius_ratio": float(
                feasible_quotient_config.noop_dynamics_radius_ratio
            ),
            "radius_floor": float(feasible_quotient_config.radius_floor),
            "radius_scale_training_matched": not radius_scale_ablation,
            "training_matched": overall_training_matched,
            "inference_only_ablation": radius_scale_ablation,
            "training_method_source_revision": adapter_identity[
                "training_method_source_revision"
            ],
            "training_method_source_archive_sha256": adapter_identity[
                "training_method_source_archive_sha256"
            ],
            "runtime_method_source_revision": runtime_source_identity[
                "revision"
            ],
            "runtime_method_source_archive_sha256": runtime_source_identity[
                "archive_sha256"
            ],
        }
        if is_v8
        else None
    )
    receipt["experimental_inference"] = True
    receipt["production_claim_forbidden"] = True
    receipt["scientific_claim_authorized"] = False
    receipt["receipt_digest"] = trainer.object_sha256(receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    runtime_commutator_config = _runtime_commutator_config(args)
    runtime_feasible_quotient_config = _runtime_feasible_quotient_config(args)
    frozen.configure_rank_local_caches()

    requested_source = Path(args.source_video).expanduser()
    if not requested_source.is_absolute():
        raise RelationalMotionCommutatorRunnerError("source video must be absolute")
    try:
        source_path = frozen.base._plain_file(
            requested_source.resolve(strict=True), label="source video"
        )
        output_path, receipt_path = frozen.base._resolve_output(args.output)
        bundle = frozen.base.resolve_adapter_bundle(args.adapter_checkpoint)
    except frozen.base.InferenceContractError as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error

    adapter_config = _read_json(bundle.adapter_config_path, label="adapter config")
    training_receipt = _read_json(
        bundle.training_receipt_path, label="training receipt"
    )
    validate_release_training_receipt_status(training_receipt)
    trained_v8 = is_trained_v8_receipt(training_receipt)
    if trained_v8 and args.operator_mode != rmc.V8_RECONSTRUCTION_SECTION_FQT:
        raise RelationalMotionCommutatorRunnerError(
            "an RS-FQT-trained adapter may only use the V8 operator"
        )
    if (
        float(args.v8_radius_scale) != MAIN_V8_RADIUS_SCALE
        and not trained_v8
    ):
        raise RelationalMotionCommutatorRunnerError(
            "non-unit V8 radius diagnostic requires a trained V8 adapter"
        )
    try:
        if trained_v8:
            identity = v8_adapter.validate_training_adapter_contract(
                adapter_config,
                training_receipt,
                expected_checkpoint_tree_sha256=(
                    args.expected_checkpoint_tree_sha256
                ),
            )
        else:
            identity = rmc.validate_training_adapter_contract(
                adapter_config,
                training_receipt,
                expected_checkpoint_tree_sha256=(
                    args.expected_checkpoint_tree_sha256
                ),
            )
            identity = {
                **identity,
                "training_receipt_schema": rmc.TRAINING_RECEIPT_SCHEMA,
                "training_method": METHOD_NAME,
                "projection_consistent_objective": False,
            }
    except (
        rmc.RelationalMotionCommutatorInferenceError,
        v8_adapter.FeasibleQuotientInferenceError,
    ) as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    runtime_source_identity = validate_runtime_method_source(args, identity)
    if (
        args.method_source_revision != identity["training_method_source_revision"]
        or args.method_source_archive_sha256
        != identity["training_method_source_archive_sha256"]
    ):
        raise RelationalMotionCommutatorRunnerError(
            "inference source archive must exactly match training archive"
        )
    adapter_config_sha256 = frozen.base.file_sha256(bundle.adapter_config_path)
    adapter_model_sha256 = frozen.base.file_sha256(bundle.adapter_model_path)
    training_receipt_file_sha256 = frozen.base.file_sha256(
        bundle.training_receipt_path
    )
    if (
        adapter_config_sha256 != identity["adapter_config_sha256"]
        or adapter_model_sha256 != identity["adapter_model_sha256"]
    ):
        raise RelationalMotionCommutatorRunnerError(
            "adapter artifact hashes differ from post-save validation"
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
        inference_file_hashes = frozen.base.validate_inference_source_files(
            bernini_root
        )
    except (frozen.base.InferenceContractError, trainer.TrainingContractError) as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    if transformer_config["num_attention_heads"] % ULYSSES_SIZE:
        raise RelationalMotionCommutatorRunnerError(
            "attention heads are not divisible by four-rank Ulysses"
        )
    wan_diffusion_path = (
        bernini_root / "bernini/models/wan_diffusion.py"
    ).resolve(strict=True)
    try:
        wan_diffusion_sha256 = tri.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
        )
    except tri.TriBranchHookError as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
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

    if transformers_version != identity["transformers_version"]:
        raise RelationalMotionCommutatorRunnerError(
            "Transformers version differs from training"
        )
    if SYSTEM_PROMPTS.get("mv2v") != frozen.base.MV2V_SYSTEM_PROMPT:
        raise RelationalMotionCommutatorRunnerError(
            "runtime MV2V system prompt differs"
        )
    if DEFAULT_NEG_PROMPT != frozen.base.DEFAULT_NEGATIVE_PROMPT:
        raise RelationalMotionCommutatorRunnerError(
            "runtime negative prompt differs"
        )
    try:
        distributed = frozen.base.inference_distributed_contract()
    except frozen.base.InferenceContractError as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    if (
        distributed.world_size != ULYSSES_SIZE
        or distributed.ulysses_size != ULYSSES_SIZE
    ):
        raise RelationalMotionCommutatorRunnerError(
            "v7 requires exactly four Ulysses ranks"
        )
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise RelationalMotionCommutatorRunnerError(
            "v7 requires four AUH ROCm-visible GPUs"
        )
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
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    source_sha256 = frozen.base.file_sha256(source_path)
    action_prompt = frozen.base.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = frozen.base.build_training_prompt(
        v5.motion.DEFAULT_NOOP_INSTRUCTION,
        prompt_cleaner=prompt_clean,
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
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    if float(config.shift) != frozen.base.FLOW_SHIFT or config.use_unipc is not True:
        raise RelationalMotionCommutatorRunnerError(
            "renderer must use official shift-5 UniPC"
        )
    base_model = BerniniRendererModel(config)
    if any("lora_" in name.lower() for name, _ in base_model.named_modules()):
        raise RelationalMotionCommutatorRunnerError(
            "base renderer unexpectedly contains LoRA"
        )
    base_model.requires_grad_(False)
    base_model.eval()
    try:
        model, adapter_tensor_count, active_lora_module_count, loaded_identity = (
            (
                v8_adapter.strict_load_adapter
                if trained_v8 else rmc.strict_load_adapter
            )(
                base_model=base_model,
                bundle=bundle,
                adapter_config=adapter_config,
                receipt=training_receipt,
                expected_checkpoint_tree_sha256=(
                    args.expected_checkpoint_tree_sha256
                ),
            )
        )
    except (
        rmc.RelationalMotionCommutatorInferenceError,
        v8_adapter.FeasibleQuotientInferenceError,
    ) as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    if not trained_v8:
        loaded_identity = {
            **loaded_identity,
            "training_receipt_schema": rmc.TRAINING_RECEIPT_SCHEMA,
            "training_method": METHOD_NAME,
            "projection_consistent_objective": False,
        }
    if loaded_identity != identity:
        raise RelationalMotionCommutatorRunnerError(
            "validated/reloaded adapter identities differ"
        )
    renderer = model.get_base_model()
    renderer.requires_grad_(False)
    renderer.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        **frozen.base.tokenizer_load_kwargs(),
    )
    if (
        tokenizer.padding_side != "right"
        or tokenizer.init_kwargs.get("fix_mistral_regex") is not True
    ):
        raise RelationalMotionCommutatorRunnerError("tokenizer contract differs")
    action_ids, action_mask = frozen.base._tokenize_training_prompt(
        tokenizer, action_prompt
    )
    noop_ids, noop_mask = frozen.base._tokenize_training_prompt(
        tokenizer, noop_prompt
    )
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
        raise RelationalMotionCommutatorRunnerError(
            "source latent differs from exact 81f geometry"
        )
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    noop_embeddings, noop_identity = frozen.encode_semantic_noop_prompt(
        renderer, noop_ids, noop_mask, device=device
    )
    sampling = frozen.exact_sampler_contract(seed=args.seed)
    if (
        sampling.get("momentum") != 0.0
        or sampling.get("num_frames") != NUM_FRAMES
        or sampling.get("num_inference_steps") != NUM_INFERENCE_STEPS
    ):
        raise RelationalMotionCommutatorRunnerError(
            "official sampler frame/step/APG contract differs"
        )
    try:
        diffusion = tri.resolve_diffusion_core(renderer)
        pre_schedule = sigma_strata.audit_runtime_unipc_schedule(
            diffusion.scheduler, initialize=True
        )
        with rmc.relational_motion_commutator_unipc_hook(
            renderer,
            adapter_model=model,
            source_clean=source_latent,
            noop_prompt_embeds=noop_embeddings,
            latent_shape=expected_latent_shape,
            bernini_commit=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
            commutator_config=runtime_commutator_config,
            operator_mode=args.operator_mode,
            feasible_quotient_config=runtime_feasible_quotient_config,
            v8_training_matched=trained_v8,
            expected_steps=NUM_INFERENCE_STEPS,
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
        rmc.RelationalMotionCommutatorInferenceError,
        v5.PriorTangentInferenceError,
        tri.TriBranchHookError,
        sigma_strata.InferenceSigmaStrataError,
        commutator.MotionCommutatorError,
        rmc.gauge.GaugeAnchoredCommutatorError,
        v7_train.RelationalCommutatorAUHError,
    ) as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    if pre_schedule != post_schedule:
        raise RelationalMotionCommutatorRunnerError(
            "official sample changed pinned UniPC schedule"
        )
    try:
        execution_trace = rmc.validate_execution_trace(
            trace, runtime_schedule_audit=post_schedule
        )
    except rmc.RelationalMotionCommutatorInferenceError as error:
        raise RelationalMotionCommutatorRunnerError(str(error)) from error
    if tuple(int(value) for value in generated_latent.shape) != expected_latent_shape:
        raise RelationalMotionCommutatorRunnerError(
            "generated latent differs from 81f geometry"
        )
    model.to("cpu")
    del noop_embeddings, source_latent
    torch.cuda.empty_cache()

    if distributed.rank == 0:
        vae.to(device)
        with torch.no_grad():
            output = _vae_decode(vae, generated_latent)
        vae.to("cpu")
        expected_output_shape = (NUM_FRAMES, int(bucket[0]), int(bucket[1]), 3)
        if tuple(int(value) for value in output.shape) != expected_output_shape:
            raise RelationalMotionCommutatorRunnerError(
                "decoded output differs from 81f geometry"
            )
        temporary_output = output_path.with_name(
            f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
        )
        if temporary_output.exists() or temporary_output.is_symlink():
            raise RelationalMotionCommutatorRunnerError(
                f"stale temporary output: {temporary_output}"
            )
        save_output(output, str(temporary_output), fps=int(frozen.base.FPS))
        os.replace(temporary_output, output_path)
        from tools import materialize_vae

        encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(
            output_path
        )
        try:
            frozen.base.validate_exact_video_metadata(
                int(encoded.shape[0]), encoded_fps
            )
        except frozen.base.InferenceContractError as error:
            raise RelationalMotionCommutatorRunnerError(str(error)) from error
        if tuple(encoded_hw) != tuple(bucket):
            raise RelationalMotionCommutatorRunnerError(
                "encoded output geometry differs"
            )
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
            runtime_source_identity=runtime_source_identity,
        )
        frozen.base._atomic_write_json(receipt_path, receipt)
        print(frozen.base.canonical_json_bytes(receipt).decode("utf-8"), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


__all__ = [
    "KAPPA_CHOICES",
    "MAIN_KAPPA",
    "MAIN_V8_RADIUS_SCALE",
    "RelationalMotionCommutatorRunnerError",
    "V8_RADIUS_SCALE_CHOICES",
    "build_inference_receipt",
    "build_parser",
    "is_inference_only_ablation",
    "is_trained_v8_receipt",
    "launcher_contract",
    "main",
    "validate_cli",
    "validate_runtime_method_source",
    "validate_release_training_receipt_status",
]


if __name__ == "__main__":
    raise SystemExit(main())
