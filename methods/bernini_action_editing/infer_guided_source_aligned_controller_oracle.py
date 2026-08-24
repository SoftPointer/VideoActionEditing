#!/usr/bin/env python3
"""Fail-closed guided source-aligned Bernini dog inference, V2.

This runner keeps the raw-conditional V1 oracle untouched as historical
diagnostic evidence.  Its formal registry contains a VAE control, the official
Bernini ``v2v_apg`` sampler, and four matched guided FlowEdit mechanism arms.
The only external model conditions are one canonical source video and one edit
instruction.  The no-op instruction and verbatim Bernini negative prompt are
fixed internal controls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import guided_source_aligned_controller as guided
import infer_source_aligned_controller_oracle as v1


RECEIPT_SCHEMA = "bernini-r-1p3b-guided-source-aligned-controller-oracle-v2"
EXPECTED_FRAMES = v1.EXPECTED_FRAMES
EXPECTED_FPS = v1.EXPECTED_FPS
EXPECTED_STEPS = v1.EXPECTED_STEPS
EXPECTED_SEED = 2027
EXPECTED_FLOW_SHIFT = v1.EXPECTED_FLOW_SHIFT
EXPECTED_ULYSSES_SIZE = v1.EXPECTED_ULYSSES_SIZE
EXPECTED_LATENT_PHASES = v1.EXPECTED_LATENT_PHASES
EXPECTED_BUCKET_HW = v1.EXPECTED_BUCKET_HW
EXPECTED_SOURCE_TOKENS = v1.EXPECTED_SOURCE_TOKENS
EXPECTED_SOURCE_SHA256 = v1.EXPECTED_SOURCE_SHA256
EXPECTED_SOURCE_LATENT_CONTENT_SHA256 = (
    "c697074e0b2b3dabe71fdcacb398128a4a72f87346dfa88f491d16509eeae9c7"
)
EXPECTED_SOURCE_LATENT_RAW_STORAGE_SHA256 = (
    "b762220eb8ca8e12c33b4bd4a8a476cc31446671bc931d8d54c9384146d08dc2"
)
SOURCE_LATENT_CONSENSUS_MIN_RANKS = 3
EXPECTED_ORIGINAL_SOURCE_PATH = v1.EXPECTED_ORIGINAL_SOURCE_PATH
EXPECTED_INSTRUCTION = v1.EXPECTED_INSTRUCTION
NOOP_INSTRUCTION = v1.NOOP_INSTRUCTION
NOOP_INSTRUCTION_SHA256 = v1.NOOP_INSTRUCTION_SHA256
BERNINI_COMMIT = v1.BERNINI_COMMIT
VEOMNI_COMMIT = v1.VEOMNI_COMMIT
CHECKPOINT_TREE_SHA256 = v1.CHECKPOINT_TREE_SHA256
CHECKPOINT_CONTENT_MANIFEST_SHA256 = v1.CHECKPOINT_CONTENT_MANIFEST_SHA256
CHECKPOINT_CONTENT_FILE_COUNT = v1.CHECKPOINT_CONTENT_FILE_COUNT

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

METHOD_RUNTIME_FILES = {
    "guided_runner": "infer_guided_source_aligned_controller_oracle.py",
    "guided_controller": "guided_source_aligned_controller.py",
    "raw_v1_controller": "source_aligned_controller.py",
    "raw_v1_runner_utils": "infer_source_aligned_controller_oracle.py",
    "differential_sampler": "differential_sampler.py",
    "tri_branch_apg": "tri_branch_unipc.py",
    "inference_runtime": "infer_lora.py",
    "checkpoint_runtime": "train_lora.py",
    "video_runtime": "tools/materialize_vae.py",
}
METHOD_ARCHIVE_MEMBERS = {
    label: f"methods/bernini_action_editing/{relative}"
    for label, relative in METHOD_RUNTIME_FILES.items()
}

legacy: Any = None
trainer: Any = None


class GuidedInferenceError(RuntimeError):
    """Raised instead of publishing an ambiguous guided V2 artifact."""


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    execution: str
    expected_shared_step_calls: int
    expected_fresh_noise_draws: int
    expected_candidate_evaluations: int
    decision_role: str


_ARM_SPECS = (
    ArmSpec("C0", "vae_identity", 0, 0, 0, "vae_encode_decode_control"),
    ArmSpec("O0", "official_v2v_apg_unipc", 80, 0, 0, "official_bernini_baseline"),
    ArmSpec("FIID1G", "guided_flowedit", 160, 40, 40, "iid_flowedit_control"),
    ArmSpec("FANC1G", "guided_flowedit", 160, 40, 40, "anc_mechanism_arm"),
    ArmSpec("FAVG5G", "guided_flowedit", 208, 52, 52, "uniform_bank_control"),
    ArmSpec("FSGA5G", "guided_flowedit", 208, 52, 52, "sga_mechanism_arm"),
)
ARM_SPECS = {item.arm: item for item in _ARM_SPECS}
ARM_NAMES = tuple(item.arm for item in _ARM_SPECS)
ARM_TABLE = [asdict(item) for item in _ARM_SPECS]
ARM_TABLE_SHA256 = v1.object_sha256(ARM_TABLE)


def arm_spec(name: str) -> ArmSpec:
    try:
        return ARM_SPECS[name]
    except (KeyError, TypeError) as error:
        raise GuidedInferenceError(
            f"arm must be one of {ARM_NAMES}, got {name!r}"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one frozen Bernini guided source-aligned V2 dog arm"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--original-source-path", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument("--expected-bernini-commit", default=BERNINI_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=VEOMNI_COMMIT)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive", required=True)
    parser.add_argument("--durable-method-source-archive", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--method-source-tree-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> ArmSpec:
    spec = arm_spec(args.arm)
    exact = {
        "instruction": EXPECTED_INSTRUCTION,
        "original_source_path": EXPECTED_ORIGINAL_SOURCE_PATH,
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "expected_bernini_commit": BERNINI_COMMIT,
        "expected_veomni_commit": VEOMNI_COMMIT,
        "expected_checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
    }
    for name, expected in exact.items():
        if getattr(args, name, None) != expected:
            raise GuidedInferenceError(f"canonical guided dog run differs at {name}")
    if _SHA1_RE.fullmatch(str(args.method_source_revision)) is None:
        raise GuidedInferenceError("method source revision must be a full Git SHA")
    for name in ("method_source_archive_sha256", "method_source_tree_sha256"):
        if _SHA256_RE.fullmatch(str(getattr(args, name))) is None:
            raise GuidedInferenceError(f"{name} must be a lowercase SHA-256")
    return spec


def _plain_absolute_file(value: str, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise GuidedInferenceError(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as error:
        raise GuidedInferenceError(f"cannot resolve {label}") from error
    if not stat.S_ISREG(mode):
        raise GuidedInferenceError(f"{label} is not a plain file")
    return resolved


def validate_method_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Bind the executed V2 files to both scratch and durable Git archives."""

    try:
        policy = v1._bytecode_policy()
        tree = v1.method_tree_manifest(METHOD_ROOT)
    except v1.SourceAlignedInferenceError as error:
        raise GuidedInferenceError(str(error)) from error
    if tree["tree_sha256"] != args.method_source_tree_sha256:
        raise GuidedInferenceError("guided runtime tree SHA-256 differs")
    archive = _plain_absolute_file(args.method_source_archive, label="scratch method archive")
    durable = _plain_absolute_file(
        args.durable_method_source_archive, label="durable method archive"
    )
    archive_sha = v1.file_sha256(archive)
    durable_sha = v1.file_sha256(durable)
    if archive_sha != args.method_source_archive_sha256 or durable_sha != archive_sha:
        raise GuidedInferenceError("scratch/durable method archive digest differs")
    member_hashes: dict[str, str] = {}
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            if handle.pax_headers.get("comment") != args.method_source_revision:
                raise GuidedInferenceError("method archive revision comment differs")
            members = handle.getmembers()
            for label, member_name in METHOD_ARCHIVE_MEMBERS.items():
                matches = [item for item in members if item.name == member_name]
                if len(matches) != 1 or not matches[0].isfile():
                    raise GuidedInferenceError(
                        f"method archive member differs: {member_name}"
                    )
                extracted = handle.extractfile(matches[0])
                if extracted is None:
                    raise GuidedInferenceError(f"cannot read archive member {member_name}")
                member_hashes[label] = hashlib.sha256(extracted.read()).hexdigest()
    except (OSError, tarfile.TarError) as error:
        raise GuidedInferenceError("cannot validate guided method archive") from error
    runtime_hashes = {
        label: v1.file_sha256(METHOD_ROOT / relative)
        for label, relative in METHOD_RUNTIME_FILES.items()
    }
    if runtime_hashes != member_hashes:
        raise GuidedInferenceError("executed guided files differ from Git archive")
    return {
        "revision": args.method_source_revision,
        "scratch_archive_path": str(archive),
        "durable_archive_path": str(durable),
        "archive_sha256": archive_sha,
        "archive_member_sha256": member_hashes,
        "runtime_source_sha256": runtime_hashes,
        "runtime_tree": tree,
        "bytecode_policy": policy,
    }


def schedule_identity(diffusion: Any) -> dict[str, Any]:
    """Hash actual pinned CPU schedule values without replacing sigma[0] by 1."""

    import torch

    scheduler = diffusion.scheduler
    sigmas_tensor = getattr(scheduler, "sigmas", None)
    timesteps_tensor = getattr(scheduler, "timesteps", None)
    if not isinstance(sigmas_tensor, torch.Tensor) or not isinstance(
        timesteps_tensor, torch.Tensor
    ):
        raise GuidedInferenceError("scheduler tensors are missing")
    if (
        sigmas_tensor.ndim != 1
        or sigmas_tensor.device.type != "cpu"
        or sigmas_tensor.dtype != torch.float32
    ):
        raise GuidedInferenceError("scheduler sigmas are not pinned CPU fp32")
    sigmas = tuple(
        float(item)
        for item in sigmas_tensor.detach().to(device="cpu", dtype=torch.float64).tolist()
    )
    try:
        intervals = guided.cdf.descending_sigma_intervals(
            sigmas, expected_steps=EXPECTED_STEPS
        )
        checked = guided.validate_pinned_sigma_intervals(intervals)
        sigma_scalars, sigma_fp32_digest = (
            guided.capture_pinned_scheduler_sigma_scalars(diffusion, checked)
        )
        if sigma_fp32_digest != guided.PINNED_UNIPC_SIGMA_FP32_DIGEST:
            raise GuidedInferenceError(
                "pinned UniPC CPU-fp32 sigma bit digest differs"
            )
    except Exception as error:
        raise GuidedInferenceError(str(error)) from error
    timesteps = tuple(
        float(item)
        for item in timesteps_tensor.detach().to(
            device="cpu", dtype=torch.float64
        ).tolist()
    )
    if len(timesteps) != EXPECTED_STEPS:
        raise GuidedInferenceError("pinned timestep count differs")
    sigma_path = tuple(checked[0][:1]) + tuple(pair[1] for pair in checked)
    payload = {
        "timesteps": list(timesteps),
        "sigmas": list(sigma_path),
        "flow_shift": EXPECTED_FLOW_SHIFT,
        "steps": EXPECTED_STEPS,
    }
    digest = v1.object_sha256(payload)
    if digest != guided.PINNED_UNIPC_SCHEDULE_DIGEST:
        raise GuidedInferenceError(
            "pinned UniPC full schedule digest differs before inference"
        )
    first_retention = (1.0 - sigma_path[0]) / (1.0 - guided.ANC_LOCK_SIGMA)
    if first_retention <= 0.0:
        raise GuidedInferenceError("pinned first ANC retention was rounded to zero")
    return {
        **payload,
        "digest": digest,
        "pinned_start_sigma": sigma_path[0],
        "first_anc_retained_variance": first_retention,
        "first_anc_predecessor_policy": "zero_initialized_per_dynaedit_pseudocode",
        "start_sigma_claimed_exact_one": False,
        "scheduler_sigma_fp32_digest": sigma_fp32_digest,
        "scheduler_sigma_dtype": str(sigmas_tensor.dtype),
        "scheduler_sigma_device": sigmas_tensor.device.type,
        "scheduler_sigma_direct_views": all(
            scalar.untyped_storage().data_ptr()
            == sigmas_tensor.untyped_storage().data_ptr()
            for scalar in sigma_scalars
        ),
    }


def preflight_schedule_identity(diffusion: Any, device: Any) -> dict[str, Any]:
    """Set and lock the real scheduler before any transformer forward."""

    config = guided.cdf.DifferentialFlowConfig(
        num_inference_steps=EXPECTED_STEPS,
        flow_shift=EXPECTED_FLOW_SHIFT,
        seed=EXPECTED_SEED,
        motion_scale=1.0,
    )
    try:
        guided.cdf._set_scheduler_timesteps(diffusion, config, device)
        return schedule_identity(diffusion)
    except GuidedInferenceError:
        raise
    except Exception as error:
        raise GuidedInferenceError(
            f"cannot preflight pinned UniPC schedule: {error}"
        ) from error


def validate_trace(
    trace: Mapping[str, Any],
    *,
    spec: ArmSpec,
    shared_step_calls: int,
    schedule: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    if shared_step_calls != spec.expected_shared_step_calls:
        raise GuidedInferenceError("guided shared_step call count differs")
    if spec.arm == "C0":
        if trace != {
            "mode": "vae_identity",
            "shared_step_calls": 0,
            "fresh_noise_draws": 0,
        } or schedule is not None:
            raise GuidedInferenceError("C0 trace differs")
        return {"validated": True, **dict(trace)}
    if schedule is None:
        raise GuidedInferenceError("active arm lacks a schedule identity")
    if schedule.get("digest") != guided.PINNED_UNIPC_SCHEDULE_DIGEST:
        raise GuidedInferenceError("active schedule is not the pinned full schedule")
    if not math.isclose(
        float(schedule.get("pinned_start_sigma", -1.0)),
        guided.PINNED_UNIPC_START_SIGMA,
        rel_tol=0.0,
        abs_tol=guided.PINNED_UNIPC_START_SIGMA_ATOL,
    ) or schedule.get("start_sigma_claimed_exact_one") is not False:
        raise GuidedInferenceError("active schedule start sigma claim differs")
    if float(schedule.get("first_anc_retained_variance", 0.0)) <= 0.0:
        raise GuidedInferenceError("active schedule lost first ANC retention")
    if (
        schedule.get("first_anc_predecessor_policy")
        != "zero_initialized_per_dynaedit_pseudocode"
        or schedule.get("scheduler_sigma_dtype") != "torch.float32"
        or schedule.get("scheduler_sigma_device") != "cpu"
        or schedule.get("scheduler_sigma_direct_views") is not True
        or schedule.get("scheduler_sigma_fp32_digest")
        != guided.PINNED_UNIPC_SIGMA_FP32_DIGEST
    ):
        raise GuidedInferenceError("active scheduler scalar provenance differs")
    if spec.arm == "O0":
        expected = {
            "mode": "official_v2v_apg_unipc",
            "branch_order": ["negative", "action"],
            "branch_counts": [40, 40],
            "shared_step_calls": 80,
            "fresh_noise_draws": 0,
            "schedule_digest": schedule["digest"],
            "scheduler_sigma_fp32_digest": schedule[
                "scheduler_sigma_fp32_digest"
            ],
            "scheduler_sigma_dtype": "torch.float32",
            "scheduler_sigma_device": "cpu",
            "scheduler_sigma_direct_views": True,
        }
        if dict(trace) != expected:
            raise GuidedInferenceError("O0 official trace differs")
        return {"validated": True, **expected}

    expected_counts = (
        [5, 5, 5] + [1] * 37
        if spec.arm in ("FAVG5G", "FSGA5G")
        else [1] * 40
    )
    if trace.get("arm") != spec.arm:
        raise GuidedInferenceError("guided trace arm differs")
    if list(trace.get("candidate_counts", ())) != expected_counts:
        raise GuidedInferenceError("guided candidate schedule differs")
    if trace.get("fresh_noise_draws") != spec.expected_fresh_noise_draws:
        raise GuidedInferenceError("guided fresh-noise draw count differs")
    if trace.get("total_shared_step_calls") != spec.expected_shared_step_calls:
        raise GuidedInferenceError("guided trace call total differs")
    if tuple(trace.get("branch_order", ())) != guided.BRANCH_ORDER:
        raise GuidedInferenceError("guided branch order differs")
    if tuple(trace.get("branch_counts", ())) != (
        spec.expected_candidate_evaluations,
    ) * 4:
        raise GuidedInferenceError("guided per-branch counts differ")
    if trace.get("schedule_digest") != schedule["digest"]:
        raise GuidedInferenceError("controller/runner schedule digest differs")
    if (
        trace.get("scheduler_sigma_fp32_digest")
        != schedule["scheduler_sigma_fp32_digest"]
        or trace.get("scheduler_sigma_dtype") != "torch.float32"
        or trace.get("scheduler_sigma_device") != "cpu"
        or trace.get("scheduler_sigma_direct_views") is not True
    ):
        raise GuidedInferenceError(
            "controller/runner scheduler scalar evidence differs"
        )
    sigmas = tuple(float(value) for value in trace.get("sigmas", ()))
    timesteps = tuple(float(value) for value in trace.get("timesteps", ()))
    retention = tuple(
        float(value) for value in trace.get("anc_retained_variance", ())
    )
    correlation = tuple(
        float(value) for value in trace.get("anc_nominal_correlation", ())
    )
    if (
        len(sigmas) != 41
        or len(timesteps) != 40
        or len(retention) != 40
        or len(correlation) != 40
        or sigmas != tuple(float(value) for value in schedule["sigmas"])
        or timesteps != tuple(float(value) for value in schedule["timesteps"])
    ):
        raise GuidedInferenceError("guided sigma/ANC trace length differs")
    if not math.isclose(
        sigmas[0],
        guided.PINNED_UNIPC_START_SIGMA,
        rel_tol=0.0,
        abs_tol=guided.PINNED_UNIPC_START_SIGMA_ATOL,
    ):
        raise GuidedInferenceError("guided trace rounded start sigma")
    expected_retention = tuple(
        0.0
        if spec.arm == "FIID1G"
        else guided.raw_controller.anc_retained_variance(
            sigma, lock_sigma=guided.ANC_LOCK_SIGMA
        )
        for sigma in sigmas[:-1]
    )
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-10)
        for actual, expected in zip(retention, expected_retention)
    ) or any(
        not math.isclose(actual, math.sqrt(expected), rel_tol=0.0, abs_tol=1.0e-10)
        for actual, expected in zip(correlation, expected_retention)
    ):
        raise GuidedInferenceError("guided ANC coefficient path differs")
    apg = dict(trace.get("apg_parameters", ()))
    if apg != {
        "guidance_mode": "v2v_apg",
        "guidance_scale": 4.0,
        "eta": 0.5,
        "norm_threshold": 50.0,
        "momentum": 0.0,
    }:
        raise GuidedInferenceError("guided APG parameters differ")
    if (
        trace.get("target_branch_query_parity") is not True
        or trace.get("source_branch_query_parity") is not True
        or trace.get("raw_velocity_dtype") != "torch.bfloat16"
        or trace.get("guided_velocity_dtype") != "torch.float32"
        or trace.get("apg_clean_dtype") != "torch.float32"
        or trace.get("delta_dtype") != "torch.float32"
        or trace.get("edit_state_dtype") != "torch.float32"
        or trace.get("candidate_continuation") != "candidate_0"
        or trace.get("weighted_noise_collapse_used") is not False
        or trace.get("anc_initial_predecessor_policy")
        != "zero_initialized_per_dynaedit_pseudocode"
    ):
        raise GuidedInferenceError("guided query/dtype/collapse evidence differs")
    bank = guided.noise_bank_pairing_contract(seed=EXPECTED_SEED)
    expected_used_noise_digest = guided.used_noise_key_digest(
        guided.GuidedSourceAlignedConfig(arm=spec.arm)
    )
    if (
        trace.get("full_noise_bank_digest") != bank["full_bank_digest"]
        or trace.get("candidate0_noise_bank_digest")
        != bank["candidate0_bank_digest"]
        or trace.get("used_noise_key_digest") != expected_used_noise_digest
    ):
        raise GuidedInferenceError("guided noise-bank pairing digest differs")
    for name in (
        "used_fresh_noise_content_digest",
        "candidate0_fresh_noise_content_digest",
    ):
        if _SHA256_RE.fullmatch(str(trace.get(name, ""))) is None:
            raise GuidedInferenceError("guided fresh-noise content digest differs")
    scores = trace.get("sga_scores")
    weights = trace.get("sga_weights")
    entropy = trace.get("sga_entropy")
    top1_margin = trace.get("sga_top1_margin")
    if not isinstance(scores, (list, tuple)) or len(scores) != 40:
        raise GuidedInferenceError("guided SGA scores differ")
    if not isinstance(weights, (list, tuple)) or len(weights) != 40:
        raise GuidedInferenceError("guided SGA weights differ")
    if not isinstance(entropy, (list, tuple)) or len(entropy) != 40:
        raise GuidedInferenceError("guided SGA entropy differs")
    if not isinstance(top1_margin, (list, tuple)) or len(top1_margin) != 40:
        raise GuidedInferenceError("guided SGA top-1 margin differs")
    for name in ("delta_rms", "update_rms", "noise_state_change_rms"):
        values = trace.get(name)
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 40
            or any(not math.isfinite(float(value)) for value in values)
        ):
            raise GuidedInferenceError(f"guided {name} trace differs")
    for index, count in enumerate(expected_counts):
        if len(scores[index]) != (count if count > 1 else 0):
            raise GuidedInferenceError("guided SGA score cardinality differs")
        score_values = tuple(float(value) for value in scores[index])
        if any(not math.isfinite(value) for value in score_values):
            raise GuidedInferenceError("guided SGA score is non-finite")
        if len(weights[index]) != count:
            raise GuidedInferenceError("guided SGA weight cardinality differs")
        values = tuple(float(value) for value in weights[index])
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise GuidedInferenceError("guided SGA weights do not sum to one")
        if spec.arm == "FAVG5G" and index < 3 and any(
            not math.isclose(value, 0.2, rel_tol=0.0, abs_tol=1.0e-7)
            for value in values
        ):
            raise GuidedInferenceError("FAVG5G weights are not uniform")
        if spec.arm == "FSGA5G" and index < 3:
            maximum = max(score_values)
            exponentials = tuple(
                math.exp((value - maximum) / guided.SGA_TEMPERATURE)
                for value in score_values
            )
            denominator = sum(exponentials)
            expected_weights = tuple(value / denominator for value in exponentials)
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=2.0e-6)
                for actual, expected in zip(values, expected_weights)
            ):
                raise GuidedInferenceError("FSGA5G weights differ from tau=1 softmax")
        expected_entropy = -sum(
            value * math.log(max(value, 1.0e-30)) for value in values
        )
        ordered = sorted(values, reverse=True)
        expected_margin = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
        if not math.isclose(
            float(entropy[index]),
            expected_entropy,
            rel_tol=0.0,
            abs_tol=2.0e-6,
        ) or not math.isclose(
            float(top1_margin[index]),
            expected_margin,
            rel_tol=0.0,
            abs_tol=2.0e-6,
        ):
            raise GuidedInferenceError(
                "guided SGA concentration statistics differ"
            )
    return {
        "validated": True,
        "arm": spec.arm,
        "shared_step_calls": shared_step_calls,
        "fresh_noise_draws": spec.expected_fresh_noise_draws,
        "effective_candidate_counts": expected_counts,
        "schedule_digest": schedule["digest"],
        "trace_digest": v1.object_sha256(dict(trace)),
    }


def select_pinned_source_latent_consensus(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, dict[str, Any]]:
    """Select a pinned three-of-four VAE encoding before controller execution.

    ROCm/MIOpen can occasionally choose a different convolution numerical path
    in one independently encoding rank.  The canonical dog oracle already has
    a source-latent byte identity established by the VAE control.  Require a
    strict majority to reproduce that exact identity, then use its lowest rank
    as the broadcast owner.  A 2/2 split or an unpinned majority fails closed.
    """

    if len(rows) != EXPECTED_ULYSSES_SIZE or sorted(
        row.get("rank") for row in rows
    ) != list(range(EXPECTED_ULYSSES_SIZE)):
        raise GuidedInferenceError(
            "source latent consensus requires four ordered rank encodings"
        )
    canonical_rows = [dict(row) for row in sorted(rows, key=lambda row: row["rank"])]
    matching_ranks: list[int] = []
    for row in canonical_rows:
        identity = row.get("identity")
        if not isinstance(identity, Mapping):
            raise GuidedInferenceError("source latent pre-broadcast identity is missing")
        if (
            identity.get("shape") != [1, 16, 21, 62, 60]
            or identity.get("dtype") != "torch.float32"
            or identity.get("numel") != 1_249_920
            or identity.get("byte_count") != 4_999_680
            or identity.get("finite") is not True
        ):
            raise GuidedInferenceError(
                "source latent pre-broadcast geometry or dtype differs"
            )
        if (
            identity.get("content_sha256")
            == EXPECTED_SOURCE_LATENT_CONTENT_SHA256
            and identity.get("raw_storage_sha256")
            == EXPECTED_SOURCE_LATENT_RAW_STORAGE_SHA256
        ):
            matching_ranks.append(int(row["rank"]))
    if len(matching_ranks) < SOURCE_LATENT_CONSENSUS_MIN_RANKS:
        raise GuidedInferenceError(
            "fewer than three ranks reproduced the pinned source VAE latent"
        )
    owner_rank = min(matching_ranks)
    return owner_rank, {
        "policy": "pinned_three_of_four_then_broadcast",
        "required_matching_ranks": SOURCE_LATENT_CONSENSUS_MIN_RANKS,
        "pinned_content_sha256": EXPECTED_SOURCE_LATENT_CONTENT_SHA256,
        "pinned_raw_storage_sha256": EXPECTED_SOURCE_LATENT_RAW_STORAGE_SHA256,
        "matching_ranks": matching_ranks,
        "owner_rank": owner_rank,
        "pre_broadcast_per_rank": canonical_rows,
        "pre_broadcast_digest": v1.object_sha256(canonical_rows),
    }


def synchronize_source_latent_consensus(
    source_latent: Any, *, rank: int, dist_module: Any
) -> tuple[Any, dict[str, Any]]:
    """Broadcast the pinned majority VAE latent and prove post-broadcast bytes."""

    source_latent = source_latent.contiguous()
    pre_row = {
        "rank": rank,
        "identity": v1.tensor_identity(
            source_latent, label="source latent before consensus"
        ),
    }
    pre_rows: list[Any] = [None] * EXPECTED_ULYSSES_SIZE
    dist_module.all_gather_object(pre_rows, pre_row)
    owner_rank, certificate = select_pinned_source_latent_consensus(pre_rows)
    dist_module.broadcast(source_latent, src=owner_rank)
    post_row = {
        "rank": rank,
        "identity": v1.tensor_identity(
            source_latent, label="source latent after consensus"
        ),
    }
    post_rows: list[Any] = [None] * EXPECTED_ULYSSES_SIZE
    dist_module.all_gather_object(post_rows, post_row)
    ordered_post = [
        dict(row) for row in sorted(post_rows, key=lambda row: row["rank"])
    ]
    reference = ordered_post[0]["identity"]
    if (
        any(row["identity"] != reference for row in ordered_post[1:])
        or reference.get("content_sha256")
        != EXPECTED_SOURCE_LATENT_CONTENT_SHA256
        or reference.get("raw_storage_sha256")
        != EXPECTED_SOURCE_LATENT_RAW_STORAGE_SHA256
    ):
        raise GuidedInferenceError(
            "source latent differs after pinned majority broadcast"
        )
    return source_latent, {
        **certificate,
        "post_broadcast_per_rank": ordered_post,
        "post_broadcast_digest": v1.object_sha256(ordered_post),
        "all_rank_post_broadcast_exact": True,
    }


def _rank_value_summary(value: Any) -> Any:
    """Keep scalar disagreements readable and hash larger values deterministically."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"sha256": v1.object_sha256(value), "type": type(value).__name__}


def _first_rank_difference(values: Sequence[Any], *, path: str) -> dict[str, Any]:
    """Describe the first structural leaf that differs across rank values."""

    reference = values[0]
    if all(value == reference for value in values[1:]):
        return {"path": path, "values": [_rank_value_summary(value) for value in values]}
    if all(isinstance(value, Mapping) for value in values):
        keys = sorted({str(key) for value in values for key in value})
        for key in keys:
            children = tuple(value.get(key) for value in values)
            if any(child != children[0] for child in children[1:]):
                return _first_rank_difference(children, path=f"{path}.{key}")
    sequence_types = (list, tuple)
    if all(isinstance(value, sequence_types) for value in values):
        lengths = tuple(len(value) for value in values)
        if any(length != lengths[0] for length in lengths[1:]):
            return {"path": f"{path}.length", "values": list(lengths)}
        for index in range(lengths[0]):
            children = tuple(value[index] for value in values)
            if any(child != children[0] for child in children[1:]):
                return _first_rank_difference(
                    children, path=f"{path}[{index}]"
                )
    return {
        "path": path,
        "values": [_rank_value_summary(value) for value in values],
    }


def validate_four_rank_runtime(
    rows: Sequence[Mapping[str, Any]], *, spec: ArmSpec
) -> dict[str, Any]:
    if len(rows) != EXPECTED_ULYSSES_SIZE or sorted(
        row.get("rank") for row in rows
    ) != list(range(EXPECTED_ULYSSES_SIZE)):
        raise GuidedInferenceError("exactly four ordered rank certificates are required")
    invariant = (
        "arm",
        "source_video_sha256",
        "source_latent",
        "source_latent_consensus",
        "action_prompt_embeddings",
        "noop_prompt_embeddings",
        "negative_prompt_embeddings",
        "generated_latent",
        "trace",
        "trace_validation",
        "schedule_identity",
        "freeze_before",
        "freeze_after",
        "shared_step_audit_restored",
        "method_manifest_digest",
    )
    reference = rows[0]
    mismatched = tuple(
        name
        for name in invariant
        if any(row.get(name) != reference.get(name) for row in rows[1:])
    )
    if mismatched:
        diagnostics = {
            name: _first_rank_difference(
                tuple(row.get(name) for row in rows), path=name
            )
            for name in mismatched
        }
        raise GuidedInferenceError(
            "rank-local guided runtime evidence differs: "
            + json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
        )
    if any(
        row.get("arm") != spec.arm
        or row.get("ulysses_size") != EXPECTED_ULYSSES_SIZE
        or row.get("trace_validation", {}).get("validated") is not True
        or row.get("shared_step_audit_restored") is not True
        or row.get("freeze_before") != row.get("freeze_after")
        or row.get("source_latent_consensus", {}).get(
            "all_rank_post_broadcast_exact"
        )
        is not True
        for row in rows
    ):
        raise GuidedInferenceError("one rank guided certificate is incomplete")
    source = reference["source_latent"]
    generated = reference["generated_latent"]
    identity_fields = (
        "shape",
        "dtype",
        "numel",
        "byte_count",
        "content_sha256",
        "raw_storage_sha256",
        "finite",
    )
    if spec.arm == "C0":
        if any(source.get(name) != generated.get(name) for name in identity_fields) or any(
            row.get("identity_object_reused") is not True for row in rows
        ):
            raise GuidedInferenceError("C0 source/generated latent is not byte exact")
    elif any(row.get("identity_object_reused") is not False for row in rows):
        raise GuidedInferenceError("active guided arm reused the source latent object")
    canonical_rows = [dict(row) for row in rows]
    return {
        "validated": True,
        "all_four_ranks_exact": True,
        "arm": spec.arm,
        "all_rank_source_latent_exact": True,
        "source_latent_consensus": dict(reference["source_latent_consensus"]),
        "all_rank_prompt_embeddings_exact": True,
        "all_rank_negative_embedding_exact": True,
        "all_rank_generated_latent_exact": True,
        "all_rank_controller_trace_exact": True,
        "c0_source_latent_byte_exact": spec.arm == "C0",
        "source_latent": dict(source),
        "generated_latent": dict(generated),
        "trace": dict(reference["trace"]),
        "trace_validation": dict(reference["trace_validation"]),
        "schedule_identity": reference["schedule_identity"],
        "per_rank": canonical_rows,
        "all_rank_certificate_digest": v1.object_sha256(canonical_rows),
    }


def artifact_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GuidedInferenceError("published artifact is not a plain file")
    info = path.stat()
    return {
        "path": str(path),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size": int(info.st_size),
        "sha256": v1.file_sha256(path),
    }


def unlink_owned_artifact(path: Path, identity: Optional[Mapping[str, Any]]) -> bool:
    """Remove only the exact inode/hash this process proved it published."""

    if identity is None or (not path.exists() and not path.is_symlink()):
        return False
    if path.is_symlink() or not path.is_file():
        return False
    current = artifact_identity(path)
    keys = ("path", "device", "inode", "size", "sha256")
    if any(current.get(key) != identity.get(key) for key in keys):
        return False
    path.unlink()
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _linked_identity(
    temporary_identity: Mapping[str, Any], final_path: Path
) -> dict[str, Any]:
    return {**dict(temporary_identity), "path": str(final_path)}


def publish_video_owned(
    decoded: Any,
    output_path: Path,
    *,
    save_output_fn: Callable[..., Any],
    transaction_token: str,
) -> dict[str, Any]:
    """Publish without overwrite and clean only this transaction's inode."""

    temporary = output_path.with_name(
        f".{output_path.stem}.guided-v2-tmp-{transaction_token}{output_path.suffix}"
    )
    if temporary.exists() or temporary.is_symlink():
        raise GuidedInferenceError("stale guided video temporary exists")
    temporary_identity: Optional[dict[str, Any]] = None
    linked_identity: Optional[dict[str, Any]] = None
    try:
        save_output_fn(decoded, str(temporary), fps=int(EXPECTED_FPS))
        temporary_identity = artifact_identity(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output_path)
        linked_identity = _linked_identity(temporary_identity, output_path)
        observed = artifact_identity(output_path)
        if observed != linked_identity:
            raise GuidedInferenceError("published video identity differs from owned inode")
        _fsync_directory(output_path.parent)
        return observed
    except BaseException:
        unlink_owned_artifact(output_path, linked_identity)
        raise
    finally:
        # A substituted temp is deliberately left untouched for diagnosis.
        unlink_owned_artifact(temporary, temporary_identity)


def publish_receipt_owned(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    transaction_token: str,
) -> dict[str, Any]:
    """Publish one canonical receipt with the same owned-identity protocol."""

    temporary = path.with_name(f".{path.name}.guided-v2-tmp-{transaction_token}")
    if temporary.exists() or temporary.is_symlink():
        raise GuidedInferenceError("stale guided receipt temporary exists")
    descriptor: Optional[int] = None
    temporary_identity: Optional[dict[str, Any]] = None
    linked_identity: Optional[dict[str, Any]] = None
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(v1.canonical_json_bytes(receipt) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_identity = artifact_identity(temporary)
        os.link(temporary, path)
        linked_identity = _linked_identity(temporary_identity, path)
        observed = artifact_identity(path)
        if observed != linked_identity:
            raise GuidedInferenceError("published receipt identity differs from owned inode")
        _fsync_directory(path.parent)
        return observed
    except BaseException:
        unlink_owned_artifact(path, linked_identity)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        unlink_owned_artifact(temporary, temporary_identity)


def build_receipt(
    *,
    args: argparse.Namespace,
    spec: ArmSpec,
    source_path: Path,
    source_sha256: str,
    source_metadata: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    method_pre: Mapping[str, Any],
    method_post: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    bernini_training_files: Mapping[str, str],
    bernini_inference_files: Mapping[str, str],
    prompt_hashes: Mapping[str, str],
    runtime: Mapping[str, Any],
    runtime_versions: Mapping[str, str],
    output_identity: Mapping[str, Any],
    transaction_token: str,
) -> dict[str, Any]:
    if dict(method_pre) != dict(method_post):
        raise GuidedInferenceError("guided method provenance changed pre/post")
    if any(_SHA256_RE.fullmatch(value) is None for value in prompt_hashes.values()):
        raise GuidedInferenceError("guided prompt hash is malformed")
    controller_config = (
        asdict(guided.GuidedSourceAlignedConfig(arm=spec.arm).validate())
        if spec.execution == "guided_flowedit"
        else None
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method_status": "guided_bernini_adaptation_not_official_dynaedit_reproduction",
        "official_dynaedit_reproduction_claimed": False,
        "raw_v1_status": "historical_diagnostic_not_formal_baseline",
        "method_provenance": {
            "pre": dict(method_pre),
            "post": dict(method_post),
            "pre_post_exact": True,
            "durable_archive_path": method_pre["durable_archive_path"],
        },
        "model_provenance": {
            "model": "Bernini-R-1.3B-Diffusers",
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
            "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
            "checkpoint_content": dict(checkpoint_identity),
            "bernini_training_files": dict(bernini_training_files),
            "bernini_inference_files": dict(bernini_inference_files),
        },
        "arm_registry": {
            "allowed_arms": list(ARM_NAMES),
            "arm_table": ARM_TABLE,
            "arm_table_sha256": ARM_TABLE_SHA256,
            "selected": asdict(spec),
            "method_cli_controls": ["arm"],
            "arbitrary_hyperparameter_cli_supported": False,
        },
        "weights": {
            "base_checkpoint_loaded": True,
            "base_frozen": True,
            "adapter_argument_supported": False,
            "adapter_loaded": False,
            "lora_module_count": 0,
        },
        "optimization": {
            "zero_training": True,
            "training_steps": 0,
            "optimizer_constructed": False,
            "backward_calls": 0,
        },
        "input": {
            "accepted_external_conditions": ["source_video", "edit_instruction"],
            "source_video_path": args.original_source_path,
            "staged_source_video_path": str(source_path),
            "source_video_sha256": source_sha256,
            "instruction_utf8_sha256": hashlib.sha256(
                args.instruction.encode("utf-8")
            ).hexdigest(),
            "target_video_argument": False,
            "target_video_accessed": False,
            "mask": False,
            "optical_flow": False,
            "pose": False,
            "track": False,
            "trajectory": False,
            "swept_tube": False,
            "first_frame_anchor": False,
        },
        "preprocessing": dict(source_metadata),
        "prompt_contract": {
            "action_prompt_uses_official_mv2v_training_path": True,
            "action_full_prompt_utf8_sha256": prompt_hashes["action"],
            "noop_is_internal_fixed_semantic_instruction": True,
            "noop_instruction_sha256": NOOP_INSTRUCTION_SHA256,
            "noop_full_prompt_utf8_sha256": prompt_hashes["noop"],
            "negative_prompt_is_verbatim_pinned_bernini_default": True,
            "negative_prompt_utf8_sha256": prompt_hashes["negative"],
            "negative_prompt_cleaner_applied": False,
            "action_prompt_embeddings": runtime["per_rank"][0][
                "action_prompt_embeddings"
            ],
            "noop_prompt_embeddings": runtime["per_rank"][0][
                "noop_prompt_embeddings"
            ],
            "negative_prompt_embeddings": runtime["per_rank"][0][
                "negative_prompt_embeddings"
            ],
            "tokenizer_fix_mistral_regex": True,
            "max_sequence_length": 512,
        },
        "sampling": {
            "num_frames": EXPECTED_FRAMES,
            "fps": EXPECTED_FPS,
            "num_inference_steps": EXPECTED_STEPS,
            "flow_shift": EXPECTED_FLOW_SHIFT,
            "seed": EXPECTED_SEED,
            "ulysses_size": EXPECTED_ULYSSES_SIZE,
            "guidance_mode": guided.APG_GUIDANCE_MODE,
            "apg": {
                "guidance_scale": guided.APG_GUIDANCE_SCALE,
                "eta": guided.APG_ETA,
                "norm_threshold": guided.APG_NORM_THRESHOLD,
                "momentum": guided.APG_MOMENTUM,
            },
            "controller_config": controller_config,
            "controller_contract": guided.guided_controller_contract(),
            "schedule_identity": runtime["schedule_identity"],
            "rank0_decode_and_save_only": True,
            "runtime_execution_certificate": dict(runtime),
        },
        "c0_control": (
            {
                "source_latent_object_reused": True,
                "source_latent_bytes_exact": runtime["c0_source_latent_byte_exact"],
                "operation": "source_vae_encode_then_decode_only",
                "source_mp4_byte_copy_claimed": False,
            }
            if spec.arm == "C0"
            else None
        ),
        "artifact_transaction": {
            "token": transaction_token,
            "output": dict(output_identity),
            "receipt_path": f"{output_identity['path']}.receipt.json",
            "publication": "exclusive_hardlink_no_overwrite_owned_identity_cleanup",
        },
        "output": {
            "path": output_identity["path"],
            "sha256": output_identity["sha256"],
            "device": output_identity["device"],
            "inode": output_identity["inode"],
            "size": output_identity["size"],
            "frame_count": EXPECTED_FRAMES,
            "fps": EXPECTED_FPS,
            "height": EXPECTED_BUCKET_HW[0],
            "width": EXPECTED_BUCKET_HW[1],
            "generated_latent_sha256": runtime["generated_latent"][
                "content_sha256"
            ],
            "all_rank_generated_latent_exact": True,
            "audio_preserved": False,
        },
        "runtime_versions": dict(runtime_versions),
        "experimental_inference": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = v1.object_sha256(receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    global legacy, trainer

    args = build_parser().parse_args(argv)
    spec = validate_cli(args)
    method_pre = validate_method_provenance(args)

    import infer_lora as legacy_module
    import train_lora as trainer_module

    legacy = legacy_module
    trainer = trainer_module
    # Reused V1 utilities deliberately resolve these modules through globals.
    v1.legacy = legacy
    v1.trainer = trainer
    v1.controller = guided

    source_path = _plain_absolute_file(args.source_video, label="source video")
    try:
        output_path, receipt_path = v1._resolve_output(args.output)
    except v1.SourceAlignedInferenceError as error:
        raise GuidedInferenceError(str(error)) from error
    manifest_requested = Path(args.checkpoint_content_manifest).expanduser()
    if not manifest_requested.is_absolute():
        raise GuidedInferenceError("checkpoint manifest must be absolute")
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
        bernini_inference_files = legacy.validate_inference_source_files(bernini_root)
    except (trainer.TrainingContractError, legacy.InferenceContractError) as error:
        raise GuidedInferenceError(str(error)) from error
    if transformer_config["num_attention_heads"] % EXPECTED_ULYSSES_SIZE:
        raise GuidedInferenceError("1.3B attention heads do not divide Ulysses=4")
    trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS
    from bernini.io_utils import save_output
    from bernini.cli import DEFAULT_NEG_PROMPT

    if SYSTEM_PROMPTS.get("mv2v") != legacy.MV2V_SYSTEM_PROMPT:
        raise GuidedInferenceError("runtime Bernini mv2v system prompt differs")
    if DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT:
        raise GuidedInferenceError("runtime Bernini negative prompt differs")
    if hashlib.sha256(NOOP_INSTRUCTION.encode("utf-8")).hexdigest() != NOOP_INSTRUCTION_SHA256:
        raise GuidedInferenceError("fixed semantic no-op digest differs")

    distributed = legacy.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise GuidedInferenceError("guided runner requires AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)
    torch.manual_seed(EXPECTED_SEED)
    torch.cuda.manual_seed_all(EXPECTED_SEED)

    checkpoint_messages: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_messages[0] = {
                "ok": True,
                "identity": v1.validate_checkpoint_content(
                    checkpoint, manifest_requested
                ),
            }
        except Exception as error:
            checkpoint_messages[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_messages, src=0)
    checkpoint_message = checkpoint_messages[0]
    if not isinstance(checkpoint_message, Mapping) or checkpoint_message.get("ok") is not True:
        raise GuidedInferenceError("rank-zero checkpoint content audit failed")
    checkpoint_identity = dict(checkpoint_message["identity"])

    try:
        source_tensor, source_metadata, source_sha = v1.prepare_hashed_source_snapshot(
            source_path
        )
    except v1.SourceAlignedInferenceError as error:
        raise GuidedInferenceError(str(error)) from error
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise GuidedInferenceError("canonical dog source SHA-256 differs")
    bucket = tuple(int(item) for item in source_metadata["source_derived_bucket_hw"])
    if bucket != EXPECTED_BUCKET_HW:
        raise GuidedInferenceError("canonical dog bucket differs")

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except trainer.TrainingContractError as error:
        raise GuidedInferenceError(str(error)) from error
    if float(config.shift) != EXPECTED_FLOW_SHIFT or config.use_unipc is not True:
        raise GuidedInferenceError("renderer is not pinned UniPC shift 5")
    model = BerniniRendererModel(config)
    model.requires_grad_(False)
    model.eval()
    freeze_before = v1.model_freeze_certificate(model)
    preflight_schedule = (
        None
        if spec.arm == "C0"
        else preflight_schedule_identity(
            guided.cdf.resolve_diffusion_core(model), device
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    action_prompt = legacy.build_training_prompt(
        args.instruction, prompt_cleaner=prompt_clean
    )
    noop_prompt = legacy.build_training_prompt(
        NOOP_INSTRUCTION, prompt_cleaner=prompt_clean
    )
    action_ids, action_mask = legacy._tokenize_training_prompt(tokenizer, action_prompt)
    noop_ids, noop_mask = legacy._tokenize_training_prompt(tokenizer, noop_prompt)
    negative_ids, negative_mask = legacy._tokenize_renderer_negative(
        tokenizer, DEFAULT_NEG_PROMPT
    )

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.requires_grad_(False)
    vae.eval().to(device)
    with torch.no_grad():
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        )
    expected_latent_shape = (
        1,
        int(vae.config.z_dim),
        EXPECTED_LATENT_PHASES,
        EXPECTED_BUCKET_HW[0] // 8,
        EXPECTED_BUCKET_HW[1] // 8,
    )
    if tuple(int(item) for item in source_latent.shape) != expected_latent_shape:
        raise GuidedInferenceError("source latent geometry differs")
    if source_latent.dtype != torch.float32:
        raise GuidedInferenceError("source VAE latent must be fp32")
    source_latent, source_latent_consensus = synchronize_source_latent_consensus(
        source_latent,
        rank=distributed.rank,
        dist_module=dist,
    )
    source_tokens = (
        expected_latent_shape[2]
        * (expected_latent_shape[3] // 2)
        * (expected_latent_shape[4] // 2)
    )
    if source_tokens != EXPECTED_SOURCE_TOKENS:
        raise GuidedInferenceError("source token geometry differs")
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    model.t5_text_encoder.to(device)
    with torch.no_grad():
        action_embeddings = model.encode_prompt(
            action_ids.to(device), action_mask.to(device)
        )
        noop_embeddings = model.encode_prompt(noop_ids.to(device), noop_mask.to(device))
        negative_embeddings = model.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        )
    # O0 calls the untouched official ``model.sample`` path below, which
    # performs its own prompt encoding and therefore still needs T5 resident.
    # Guided/C0 arms consume the audited embeddings directly and can offload.
    if spec.arm != "O0":
        model.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()
    for label, embeddings in (
        ("action", action_embeddings),
        ("noop", noop_embeddings),
        ("negative", negative_embeddings),
    ):
        if embeddings.dtype != torch.bfloat16:
            raise GuidedInferenceError(f"{label} embedding must be bfloat16")
    source_identity = v1.tensor_identity(source_latent, label="source latent")
    action_identity = v1.tensor_identity(action_embeddings, label="action prompt embeddings")
    noop_identity = v1.tensor_identity(noop_embeddings, label="noop prompt embeddings")
    negative_identity = v1.tensor_identity(
        negative_embeddings, label="negative prompt embeddings"
    )

    shared_step_audit = v1.SharedStepAudit(model)
    with shared_step_audit:
        with torch.no_grad():
            if spec.arm == "C0":
                generated_latent = source_latent
                trace_value: dict[str, Any] = {
                    "mode": "vae_identity",
                    "shared_step_calls": 0,
                    "fresh_noise_draws": 0,
                }
                schedule_value = None
            elif spec.arm == "O0":
                generated_latent = model.sample(
                    input_ids=action_ids.to(device),
                    attention_mask=action_mask.to(device),
                    uncond_input_ids=negative_ids.to(device),
                    uncond_attention_mask=negative_mask.to(device),
                    image_vae_latents=None,
                    multi_video_vae_latents=[source_latent],
                    multi_image_vae_latents=None,
                    width=EXPECTED_BUCKET_HW[1],
                    height=EXPECTED_BUCKET_HW[0],
                    device=device,
                    **legacy.sampler_contract(
                        steps=EXPECTED_STEPS, seed=EXPECTED_SEED
                    ),
                )
                schedule_value = schedule_identity(model.diff_dec)
                trace_value = {
                    "mode": "official_v2v_apg_unipc",
                    "branch_order": ["negative", "action"],
                    "branch_counts": [40, 40],
                    "shared_step_calls": 80,
                    "fresh_noise_draws": 0,
                    "schedule_digest": schedule_value["digest"],
                    "scheduler_sigma_fp32_digest": schedule_value[
                        "scheduler_sigma_fp32_digest"
                    ],
                    "scheduler_sigma_dtype": schedule_value[
                        "scheduler_sigma_dtype"
                    ],
                    "scheduler_sigma_device": schedule_value[
                        "scheduler_sigma_device"
                    ],
                    "scheduler_sigma_direct_views": schedule_value[
                        "scheduler_sigma_direct_views"
                    ],
                }
            else:
                model.diff_dec.transformer.to(device)
                generated_latent, trace = guided.sample_guided_source_aligned_controller(
                    model,
                    source_latent=source_latent,
                    source_rgb_frames=EXPECTED_FRAMES,
                    action_prompt_embeds=action_embeddings,
                    noop_prompt_embeds=noop_embeddings,
                    negative_prompt_embeds=negative_embeddings,
                    config=guided.GuidedSourceAlignedConfig(arm=spec.arm),
                    return_trace=True,
                )
                trace_value = asdict(trace)
                schedule_value = schedule_identity(model.diff_dec)
    if spec.arm != "C0" and schedule_value != preflight_schedule:
        raise GuidedInferenceError(
            "post-sampling schedule differs from the pre-forward preflight"
        )
    if spec.arm == "O0":
        model.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()
    if not shared_step_audit.restored:
        raise GuidedInferenceError("shared_step audit did not restore")
    if tuple(int(item) for item in generated_latent.shape) != expected_latent_shape:
        raise GuidedInferenceError("generated latent geometry differs")
    if generated_latent.dtype != torch.float32:
        raise GuidedInferenceError("generated latent must be fp32")
    identity_object_reused = generated_latent is source_latent
    if spec.arm == "C0":
        if not identity_object_reused or not bool(torch.equal(generated_latent, source_latent)):
            raise GuidedInferenceError("C0 did not reuse exact source latent")
    elif identity_object_reused:
        raise GuidedInferenceError("active arm reused source latent object")
    generated_identity = v1.tensor_identity(generated_latent, label="generated latent")
    trace_validation = validate_trace(
        trace_value,
        spec=spec,
        shared_step_calls=shared_step_audit.calls,
        schedule=schedule_value,
    )
    freeze_after = v1.model_freeze_certificate(model)
    if freeze_after != freeze_before:
        raise GuidedInferenceError("model freeze certificate changed")
    method_post = validate_method_provenance(args)
    if method_post != method_pre:
        raise GuidedInferenceError("guided method provenance changed during inference")

    local_row = {
        "rank": distributed.rank,
        "local_rank": distributed.local_rank,
        "ulysses_size": distributed.ulysses_size,
        "arm": spec.arm,
        "source_video_sha256": source_sha,
        "source_latent": source_identity,
        "source_latent_consensus": source_latent_consensus,
        "action_prompt_embeddings": action_identity,
        "noop_prompt_embeddings": noop_identity,
        "negative_prompt_embeddings": negative_identity,
        "generated_latent": generated_identity,
        "identity_object_reused": identity_object_reused,
        "trace": trace_value,
        "trace_validation": trace_validation,
        "schedule_identity": schedule_value,
        "freeze_before": freeze_before,
        "freeze_after": freeze_after,
        "shared_step_audit_restored": shared_step_audit.restored,
        "method_manifest_digest": v1.object_sha256(method_post),
    }
    rank_rows: list[Any] = [None] * EXPECTED_ULYSSES_SIZE
    dist.all_gather_object(rank_rows, local_row)
    runtime = validate_four_rank_runtime(rank_rows, spec=spec)

    model.to("cpu")
    del action_embeddings, noop_embeddings, negative_embeddings, source_latent
    torch.cuda.empty_cache()
    runtime_versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "transformers": transformers_version,
        "diffusers": diffusers_version,
    }
    prompt_hashes = {
        "action": hashlib.sha256(action_prompt.encode("utf-8")).hexdigest(),
        "noop": hashlib.sha256(noop_prompt.encode("utf-8")).hexdigest(),
        "negative": hashlib.sha256(DEFAULT_NEG_PROMPT.encode("utf-8")).hexdigest(),
    }
    transaction_token = v1.output_transaction_token()
    video_owned: Optional[dict[str, Any]] = None
    receipt_owned: Optional[dict[str, Any]] = None
    rank0_published = False
    if distributed.rank == 0:
        try:
            vae.to(device)
            with torch.no_grad():
                decoded = _vae_decode(vae, generated_latent)
            vae.to("cpu")
            if tuple(int(item) for item in decoded.shape) != (
                EXPECTED_FRAMES,
                EXPECTED_BUCKET_HW[0],
                EXPECTED_BUCKET_HW[1],
                3,
            ):
                raise GuidedInferenceError("decoded video geometry differs")
            video_owned = publish_video_owned(
                decoded,
                output_path,
                save_output_fn=save_output,
                transaction_token=transaction_token,
            )
            from tools import materialize_vae

            encoded, encoded_fps, encoded_hw = materialize_vae._decode_exact_video(
                output_path
            )
            legacy.validate_exact_video_metadata(int(encoded.shape[0]), encoded_fps)
            if tuple(encoded_hw) != EXPECTED_BUCKET_HW:
                raise GuidedInferenceError("encoded output geometry differs")
            method_publish = validate_method_provenance(args)
            if method_publish != method_pre:
                raise GuidedInferenceError("method provenance changed before publication")
            receipt = build_receipt(
                args=args,
                spec=spec,
                source_path=source_path,
                source_sha256=source_sha,
                source_metadata=source_metadata,
                checkpoint_identity=checkpoint_identity,
                method_pre=method_pre,
                method_post=method_publish,
                bernini_revision=bernini_revision,
                veomni_revision=veomni_revision,
                bernini_training_files=trainer.BERNINI_PINNED_FILE_HASHES,
                bernini_inference_files=bernini_inference_files,
                prompt_hashes=prompt_hashes,
                runtime=runtime,
                runtime_versions=runtime_versions,
                output_identity=video_owned,
                transaction_token=transaction_token,
            )
            receipt_owned = publish_receipt_owned(
                receipt_path,
                receipt,
                transaction_token=transaction_token,
            )
            print(v1.canonical_json_bytes(receipt).decode("utf-8"), flush=True)
            rank0_published = True
        except BaseException:
            unlink_owned_artifact(receipt_path, receipt_owned)
            unlink_owned_artifact(output_path, video_owned)
            raise

    try:
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:
        if distributed.rank == 0 and rank0_published:
            unlink_owned_artifact(receipt_path, receipt_owned)
            unlink_owned_artifact(output_path, video_owned)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_NAMES",
    "ARM_SPECS",
    "ARM_TABLE",
    "ARM_TABLE_SHA256",
    "ArmSpec",
    "GuidedInferenceError",
    "RECEIPT_SCHEMA",
    "arm_spec",
    "artifact_identity",
    "build_parser",
    "build_receipt",
    "main",
    "preflight_schedule_identity",
    "publish_receipt_owned",
    "publish_video_owned",
    "schedule_identity",
    "unlink_owned_artifact",
    "validate_cli",
    "validate_four_rank_runtime",
    "validate_method_provenance",
    "validate_trace",
]
