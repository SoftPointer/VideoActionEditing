#!/usr/bin/env python3
"""Versioned fixed-R2V SSFT visual-I0 diagnostic runner.

This is an experimental inference runner for one arm at a time.  It combines
the clean-state transport core in ``saic_source_state_flow_transport_v1`` with
the native Bernini field adapter in ``saic_native_source_state_field_v1``.
The registered minimal deterministic arms are:

``R00``
    The R2V system prompt with the existing no-reference two-forward APG field.
    Visual I0 is off for every exact40 cell.

``R11``
    The same R2V prompt with the existing three-forward source-I0 APG field.
    Visual I0 is on for every exact40 cell.

Both arms start from the byte-identical read-only source coordinate sealed by
Job 132387 and bind the same separately materialized, read-only source RGB
frame-0 coordinate.  The runner performs zero VAE encodes.  R00 does not pass
the admitted frame0 tensor to the native field; R11 passes that exact tensor
to both source-caption and target-caption field queries.  No full source-video
tokens enter either field.  No target video, mask, pose, flow, track,
trajectory, motion donor, or oracle edited frame is accepted.

The receipt is deliberately non-evaluative.  A successful receipt proves a
specific frozen exact40 execution and artifact publication; it does not prove
quality, action success, evaluator validity, checkpoint provenance beyond the
explicit manifest audit, Gaussianity of the registered noise bytes, training,
or an optimizer update.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import tarfile
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import differential_sampler as cdf  # noqa: E402
import guided_source_aligned_controller as guided  # noqa: E402
import infer_guided_source_aligned_controller_oracle as guided_runner  # noqa: E402
import infer_lora as legacy  # noqa: E402
import infer_native_identity_generation_canary as native_canary  # noqa: E402
import infer_source_aligned_controller_oracle as legacy_audit  # noqa: E402
import infer_source_kv_carrier_oracle as source_audit  # noqa: E402
import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import saic_pure_t2v_event_bank_v1 as event_bank  # noqa: E402
import saic_native_source_state_field_v1 as native_field  # noqa: E402
import saic_source_state_flow_transport_v1 as transport  # noqa: E402


SCHEMA_VERSION = "bernini-saic-fixed-r2v-visual-i0-diagnostic-inference-v2"
METHOD = "frozen-bernini-fixed-r2v-visual-i0-diagnostic"
SOURCE_CLEAN_ARTIFACT_SCHEMA = (
    "bernini-saic-source-clean-latent-artifact-v1"
)
SOURCE_CLEAN_RECEIPT_SCHEMA = (
    "bernini-saic-source-clean-latent-receipt-v1"
)
SOURCE_CLEAN_MATERIALIZER_METHOD = (
    "frozen-bernini-source-clean-latent-materializer"
)
SOURCE_CLEAN_TENSOR_KEY = "source_clean_latent"
FRAME0_ARTIFACT_SCHEMA = "bernini-saic-frame0-latent-artifact-v1"
FRAME0_RECEIPT_SCHEMA = "bernini-saic-frame0-latent-receipt-v1"
FRAME0_MATERIALIZER_METHOD = "frozen-bernini-frame0-latent-materializer"
FRAME0_TENSOR_KEY = "reference_frame0_latent"
FRAME0_ARTIFACT_METADATA = MappingProxyType({
    "schema_version": FRAME0_ARTIFACT_SCHEMA,
    "coordinate": "bernini_source_rgb_frame0_vae_latent",
    "frame_contract": "source_rgb_index0_latent1",
    "artifact_role": "saic_common_visual_i0_reference_coordinate",
    "source": "sealed_exact81_source_rgb_frame0_wan_vae_mode",
    "posterior": "mode",
    "sampling": "false",
    "authority": "false",
})
SOURCE_CLEAN_ARTIFACT_METADATA = MappingProxyType({
    "schema_version": SOURCE_CLEAN_ARTIFACT_SCHEMA,
    "coordinate": "bernini_normalized_clean_vae_latent",
    "frame_contract": "exact81_latent21",
    "posterior": "mode",
    "sampling": "false",
    "authority": "false",
    "artifact_role": "saic_common_source_outer_clean_state",
    "source": "sealed_source_video_private_snapshot_wan_vae_mode",
})
SOURCE_CLEAN_ACCEPTED_INPUT_ROLES = (
    "source_manifest",
    "selected_source_video",
    "checkpoint_and_source_code_provenance",
)
SOURCE_CLEAN_FORBIDDEN_INPUT_ROLES = tuple(source_set.EXPECTED_FORBIDDEN_INPUTS) + (
    "natural_language_instruction",
    "source_caption",
    "target_caption",
    "event_bank",
    "branch",
    "rollout_seed",
    "reference_frame",
    "shared_i0",
    "adapter",
    "lora",
)
SOURCE_CLEAN_MATERIALIZER_RUNTIME_FILES = (
    "materialize_saic_source_clean_latent_v1.py",
    "build_saic_reversible_source_set_v1.py",
    "infer_lora.py",
    "train_lora.py",
    "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py",
    "assets/saic_reversible_source_set_v1.json",
)
SOURCE_CLEAN_MATERIALIZER_ARCHIVE_MEMBERS = tuple(
    f"methods/bernini_action_editing/{relative}"
    for relative in SOURCE_CLEAN_MATERIALIZER_RUNTIME_FILES
)
MODEL_ID = "transformer_1"
FRAME_COUNT = 81
LATENT_FRAME_COUNT = 21
FPS = 25
NUM_INFERENCE_STEPS = 40
FLOW_SHIFT = 5.0
ULYSSES_SIZE = 4
DEFAULT_SEED = 2027
NOISE_GENERATOR_ID = "saic-sha256-keyed-cpu-torch-generator-fp32-v1"
# Preserve the v1 keyed-noise domain so same-row/seed candidate-zero bytes are
# matched with Job 132387 as well as between the two v2 arms.
NOISE_DOMAIN = b"bernini-saic-ssft-v1\0"
T2V_SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in text-to-video generation."
)
R2V_SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in subject-to-video generation."
)
_TASK_SYSTEM_PROMPTS = MappingProxyType({
    "t2v": T2V_SYSTEM_PROMPT,
    "r2v": R2V_SYSTEM_PROMPT,
})
TASK_SYSTEM_PROMPTS = _TASK_SYSTEM_PROMPTS
K1_SCHEDULE = (1,) * NUM_INFERENCE_STEPS
K5_EARLY_SCHEDULE = (5, 5, 5) + (1,) * (NUM_INFERENCE_STEPS - 3)
ZERO_SHA256 = "0" * 64
SOURCE_MANIFEST_RAW_SHA256 = (
    "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9"
)
EVENT_BANK_RAW_SHA256 = (
    "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"
)

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class SAICInferenceError(RuntimeError):
    """Raised before an ambiguous SSFT artifact can be published."""


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    field_regime: str
    task_name: str
    guidance_mode: str
    candidate_schedule: tuple[int, ...]
    anc_enabled: bool
    aggregation_mode: str
    temperature: Optional[float]
    anchor_latent_phase_zero: bool

    @property
    def expected_guided_queries(self) -> int:
        return 2 * sum(self.candidate_schedule)

    @property
    def expected_raw_forwards(self) -> int:
        return self.raw_forwards_per_guided_query * self.expected_guided_queries

    @property
    def raw_forwards_per_guided_query(self) -> int:
        if self.field_regime == "t2v_apg":
            return 2
        if self.field_regime == "r2v_apg_source_i0":
            # Official Bernini R2V APG: no-reference negative, I0-reference
            # negative, then I0-reference role condition.
            return 3
        raise SAICInferenceError("registered arm has an unknown field regime")

    @property
    def uses_reference_frame0(self) -> bool:
        return self.field_regime == "r2v_apg_source_i0"


_ARM_SPECS = (
    ArmSpec(
        "R00",
        "t2v_apg",
        "r2v",
        "t2v_apg",
        K1_SCHEDULE,
        False,
        "uniform",
        None,
        False,
    ),
    ArmSpec(
        "R11",
        "r2v_apg_source_i0",
        "r2v",
        "r2v_apg",
        K1_SCHEDULE,
        False,
        "uniform",
        None,
        False,
    ),
)
_REGISTERED_ARM_SPECS = MappingProxyType(
    {item.arm: item for item in _ARM_SPECS}
)
_REGISTERED_ARM_NAMES = tuple(item.arm for item in _ARM_SPECS)
ARM_SPECS = _REGISTERED_ARM_SPECS
ARM_NAMES = _REGISTERED_ARM_NAMES


@dataclass(frozen=True)
class NativeScheduleBundle:
    """Exact objects shared by the core, adapter, and execution receipt."""

    sigma_scalars: tuple[Any, ...]
    next_sigmas: tuple[float, ...]
    timestep_tensors: tuple[Any, ...]
    sigma_schedule: tuple[float, ...]
    core_sigma_schedule_sha256: str
    native_schedule_sha256: str
    pinned_schedule_sha256: str
    scheduler_sigma_fp32_sha256: str
    scalar_views_share_scheduler_storage: bool
    timestep_views_share_runtime_storage: bool


@dataclass(frozen=True)
class SealedSourceCoordinate:
    """Open, immutable source-clean bytes shared by all rollout arms.

    The descriptors remain open until terminal verification so a path swap,
    chmod, or in-place write cannot silently change the object whose bytes
    were admitted before rollout.
    """

    artifact_path: Path
    receipt_path: Path
    artifact_fd: int
    receipt_fd: int
    artifact_identity: Mapping[str, Any]
    receipt_identity: Mapping[str, Any]
    receipt: Mapping[str, Any]
    tensor: Any
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class SealedFrame0Coordinate:
    """Open immutable frame0 bytes shared by both diagnostic arms."""

    artifact_path: Path
    receipt_path: Path
    artifact_fd: int
    receipt_fd: int
    artifact_identity: Mapping[str, Any]
    receipt_identity: Mapping[str, Any]
    receipt: Mapping[str, Any]
    tensor: Any
    provenance: Mapping[str, Any]


RUNTIME_METHOD_FILES = (
    "infer_saic_source_state_flow_transport_v2.py",
    "infer_saic_source_state_flow_transport_v1.py",
    "materialize_saic_frame0_latent_v1.py",
    "saic_source_state_flow_transport_v1.py",
    "saic_native_source_state_field_v1.py",
    "dclr_runtime_contract.py",
    "differential_sampler.py",
    "guided_source_aligned_controller.py",
    "source_aligned_controller.py",
    "tri_branch_unipc.py",
    "infer_guided_source_aligned_controller_oracle.py",
    "infer_lora.py",
    "infer_native_identity_generation_canary.py",
    "infer_source_aligned_controller_oracle.py",
    "infer_source_kv_carrier_oracle.py",
    "infer_source_value_residual_oracle.py",
    "source_kv_replay.py",
    "source_kv_route_batches.py",
    "source_value_residual.py",
    "train_lora.py",
    "materialize_saic_source_clean_latent_v1.py",
    "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py",
    "build_saic_reversible_source_set_v1.py",
    "saic_pure_t2v_event_bank_v1.py",
    "assets/saic_reversible_source_set_v1.json",
    "assets/saic_pure_t2v_event_bank_v1.json",
)
RUNTIME_ARCHIVE_MEMBERS = tuple(
    f"methods/bernini_action_editing/{relative}" for relative in RUNTIME_METHOD_FILES
)


def arm_spec(
    name: str,
    _registry: Mapping[str, ArmSpec] = _REGISTERED_ARM_SPECS,
) -> ArmSpec:
    try:
        return _registry[name]
    except (KeyError, TypeError) as error:
        raise SAICInferenceError(
            f"arm must be one of {_REGISTERED_ARM_NAMES}, got {name!r}"
        ) from error


def sha256_utf8(value: str) -> str:
    if type(value) is not str:
        raise SAICInferenceError("UTF-8 hash input must be text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tensor_raw_sha256(value: Any) -> str:
    """Hash exact contiguous tensor bytes without attaching a quality claim."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - GPU runtime owns torch
        raise SAICInferenceError("PyTorch is required to hash tensors") from error
    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        raise SAICInferenceError("tensor hash input must be non-empty")
    if value.requires_grad or not value.is_floating_point():
        raise SAICInferenceError("tensor hash input must be detached floating data")
    contiguous = (
        value.detach().contiguous().view(torch.uint8).reshape(-1).cpu()
    )
    try:
        payload = contiguous.numpy().tobytes(order="C")
    except RuntimeError:
        payload = bytes(contiguous.tolist())
    return hashlib.sha256(payload).hexdigest()


def build_task_prompt(
    task_name: str,
    caption: str,
    *,
    prompt_cleaner: Callable[[str], str],
) -> str:
    """Build the exact Bernini training task prefix plus cleaned full caption."""

    if task_name not in _TASK_SYSTEM_PROMPTS:
        raise SAICInferenceError(f"unknown task prompt: {task_name!r}")
    if (
        type(caption) is not str
        or not caption.strip()
        or caption != caption.strip()
        or "\x00" in caption
    ):
        raise SAICInferenceError(
            "caption must be nonempty stripped source-content text without NUL"
        )
    cleaned = prompt_cleaner(caption)
    if type(cleaned) is not str or not cleaned.strip() or "\x00" in cleaned:
        raise SAICInferenceError("Wan prompt cleaner produced invalid text")
    return _TASK_SYSTEM_PROMPTS[task_name] + cleaned


def resolve_sealed_forward_cell(
    source_manifest: Mapping[str, Any],
    event_spec: Mapping[str, Any],
    *,
    row_id: str,
    branch: str,
    rollout_seed: int,
) -> dict[str, Any]:
    """Resolve one unique source/target caption pair from two validated assets.

    The target body is copied from the event bank's canonical
    ``full_t2v_caption``.  It is never synthesized by concatenating the source
    caption with an edit instruction, which could create contradictory state
    language such as "remains still" followed by a new motion.
    """

    if type(row_id) is not str or not row_id or row_id != row_id.strip():
        raise SAICInferenceError("row_id must be nonempty stripped text")
    if branch != "forward":
        raise SAICInferenceError("fixed-R2V diagnostic accepts only branch=forward")
    if type(rollout_seed) is not int or not 0 <= rollout_seed < 2**63:
        raise SAICInferenceError("rollout_seed must be in [0,2^63)")
    rows = source_manifest.get("rows")
    if not isinstance(rows, list):
        raise SAICInferenceError("source manifest rows are missing")
    selected_rows = [row for row in rows if row.get("row_id") == row_id]
    if len(selected_rows) != 1:
        raise SAICInferenceError("row_id does not select exactly one source row")
    row = selected_rows[0]
    candidates = [
        candidate
        for group in event_spec.get("groups", [])
        for candidate in group.get("candidates", [])
        if candidate.get("row_id") == row_id
        and candidate.get("branch") == branch
        and candidate.get("seed") == rollout_seed
    ]
    if len(candidates) != 1:
        raise SAICInferenceError(
            "row_id/branch/rollout_seed does not select exactly one event-bank cell"
        )
    candidate = candidates[0]
    if (
        candidate.get("iid") != row.get("iid")
        or candidate.get("source_media_sha256_for_nonuse_audit")
        != row.get("source_video_sha256")
        or candidate.get("source_caption_utf8_sha256")
        != sha256_utf8(row.get("source_caption"))
        or candidate.get("branch_instruction") != row.get("forward_instruction")
        or candidate.get("branch_instruction_utf8_sha256")
        != sha256_utf8(row.get("forward_instruction"))
        or candidate.get("full_t2v_caption_utf8_sha256")
        != sha256_utf8(candidate.get("full_t2v_caption"))
        or candidate.get("event_verified") is not False
        or candidate.get("optimizer_authorized") is not False
        or row.get("optimizer_eligible") is not False
    ):
        raise SAICInferenceError("sealed source/event cross-binding differs")
    return {
        "row_id": row_id,
        "iid": row["iid"],
        "analysis_split": row["analysis_split"],
        "actor_family": row["actor_family"],
        "candidate_id": candidate["candidate_id"],
        "branch": branch,
        "rollout_seed": rollout_seed,
        "source_video": row["source_video"],
        "source_video_sha256": row["source_video_sha256"],
        "source_caption_body": row["source_caption"],
        "source_caption_body_utf8_sha256": sha256_utf8(row["source_caption"]),
        "target_caption_body": candidate["full_t2v_caption"],
        "target_caption_body_utf8_sha256": candidate[
            "full_t2v_caption_utf8_sha256"
        ],
        "forward_instruction_utf8_sha256": candidate[
            "branch_instruction_utf8_sha256"
        ],
        "event_verified": False,
        "optimizer_authorized": False,
    }


def load_sealed_caption_cell(
    *,
    source_manifest_path: str | Path,
    source_manifest_raw_sha256: str,
    event_bank_path: str | Path,
    event_bank_raw_sha256: str,
    row_id: str,
    branch: str,
    rollout_seed: int,
    verify_bound_source_files: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load, validate, hash, and resolve one immutable forward cell."""

    source_path = _plain_absolute_file(
        source_manifest_path, label="sealed source manifest"
    )
    event_path = _plain_absolute_file(event_bank_path, label="sealed event bank")
    if source_set.file_sha256(source_path) != source_manifest_raw_sha256:
        raise SAICInferenceError("sealed source manifest raw SHA-256 differs")
    try:
        manifest = source_set.load_manifest(source_path)
        source_summary = source_set.validate_manifest(
            manifest, verify_bound_files=verify_bound_source_files
        )
        spec, actual_event_sha256 = event_bank.load_sealed_spec(
            event_path,
            expected_raw_sha256=event_bank_raw_sha256,
            source_manifest_path=source_path,
        )
    except (source_set.SAICReversibleSourceSetError, event_bank.SAICPureT2VEventBankError) as error:
        raise SAICInferenceError(str(error)) from error
    cell = resolve_sealed_forward_cell(
        manifest,
        spec,
        row_id=row_id,
        branch=branch,
        rollout_seed=rollout_seed,
    )
    assets = {
        "source_manifest_schema_version": manifest["schema_version"],
        "source_manifest_dataset_id": manifest["dataset_id"],
        "source_manifest_path": str(source_path),
        "source_manifest_raw_sha256": source_manifest_raw_sha256,
        "source_manifest_content_sha256": source_summary[
            "manifest_content_sha256"
        ],
        "source_manifest_bound_files_verified": bool(verify_bound_source_files),
        "event_bank_path": str(event_path),
        "event_bank_raw_sha256": actual_event_sha256,
        "event_bank_content_sha256": event_bank.object_sha256(spec),
        "source_manifest_terminal_events_verified": False,
        "event_bank_events_verified": False,
        "optimizer_authorized": False,
    }
    return cell, assets


def revalidate_terminal_sealed_input_bytes(
    *,
    sealed_assets: Mapping[str, Any],
    sealed_cell: Mapping[str, Any],
    selected_source_path: Path,
) -> dict[str, Any]:
    """Rehash the three semantic byte sources immediately before publish."""

    manifest = _plain_absolute_file(
        sealed_assets.get("source_manifest_path"),
        label="terminal sealed source manifest",
    )
    events = _plain_absolute_file(
        sealed_assets.get("event_bank_path"),
        label="terminal sealed event bank",
    )
    source = _plain_absolute_file(
        selected_source_path, label="terminal selected source video"
    )
    expected = (
        (
            manifest,
            sealed_assets.get("source_manifest_raw_sha256"),
            "source manifest",
        ),
        (
            events,
            sealed_assets.get("event_bank_raw_sha256"),
            "event bank",
        ),
        (
            source,
            sealed_cell.get("source_video_sha256"),
            "selected source video",
        ),
    )
    for path, digest, label in expected:
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise SAICInferenceError(f"terminal {label} expected digest differs")
        if source_set.file_sha256(path) != digest:
            raise SAICInferenceError(f"terminal {label} bytes differ")
    return {
        **dict(sealed_assets),
        "source_manifest_terminal_raw_sha256_verified": True,
        "event_bank_terminal_raw_sha256_verified": True,
        "selected_source_video_terminal_raw_sha256_verified": True,
    }


def guidance_contract(spec: ArmSpec) -> dict[str, Any]:
    """Return the fixed native field program whose digest enters every request."""

    if type(spec) is not ArmSpec or arm_spec(spec.arm) != spec:
        raise SAICInferenceError("guidance contract requires a registered arm")
    if spec.uses_reference_frame0:
        image_scale = 4.5
        chain_scales = (4.5, 4.0)
        norm_thresholds = (50.0, 50.0)
        momenta = (0.0, 0.0)
        branch_order = transport.EXPECTED_R2V_I0_BRANCH_ORDER
    else:
        image_scale = 0.0
        chain_scales = (4.0,)
        norm_thresholds = (50.0,)
        momenta = (0.0,)
        branch_order = transport.EXPECTED_T2V_V2V_BRANCH_ORDER
    return {
        "schema": f"{SCHEMA_VERSION}/native-guidance-contract-v1",
        "arm": spec.arm,
        "field_regime": spec.field_regime,
        "task_name": spec.task_name,
        "guidance_mode": spec.guidance_mode,
        "guidance_scale": 4.0,
        "image_guidance_scale": image_scale,
        "guidance_chain_scales": list(chain_scales),
        "apg_eta": 0.5,
        "apg_norm_threshold": 50.0,
        "apg_norm_thresholds": list(norm_thresholds),
        "apg_momentum": 0.0,
        "apg_momenta": list(momenta),
        "branch_order": list(branch_order),
        "guided_query_role_order_per_candidate": ["target", "source"],
        "per_guided_query_raw_forwards": spec.raw_forwards_per_guided_query,
        "per_candidate_guided_queries": 2,
        "per_candidate_raw_forwards": 2
        * spec.raw_forwards_per_guided_query,
        "candidate_schedule": list(spec.candidate_schedule),
        "candidate_continuation": "candidate_zero",
        "anc_enabled": spec.anc_enabled,
        "aggregation_mode": spec.aggregation_mode,
        "temperature": spec.temperature,
        "anchor_latent_phase_zero": spec.anchor_latent_phase_zero,
        "target_source_id": 0,
        "reference_source_id": 1 if spec.uses_reference_frame0 else None,
        "visual_condition": (
            "independently_vae_encoded_source_rgb_frame0"
            if spec.uses_reference_frame0
            else "none"
        ),
        "full_source_video_field_tokens": False,
        "spatial_arithmetic": "fp32",
    }


def keyed_noise_seed(master_seed: int, step: int, candidate: int) -> int:
    """Derive one deterministic arm-independent CPU generator seed."""

    for label, value in (
        ("master_seed", master_seed),
        ("step", step),
        ("candidate", candidate),
    ):
        if type(value) is not int or value < 0:
            raise SAICInferenceError(f"{label} must be a nonnegative integer")
    payload = (
        NOISE_DOMAIN
        + str(master_seed).encode("ascii")
        + b"\0"
        + str(step).encode("ascii")
        + b"\0"
        + str(candidate).encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        2**63 - 1
    )


def build_fresh_noise_bank(
    *,
    shape: Sequence[int],
    device: Any,
    master_seed: int,
    candidate_schedule: tuple[int, ...],
) -> tuple[tuple[Any, ...], ...]:
    """Materialize the actual ordered FP32 bank consumed by the core.

    The function establishes deterministic byte content and key provenance;
    neither property is evidence that the values are statistically Gaussian.
    """

    import torch

    dimensions = tuple(int(item) for item in shape)
    if (
        len(dimensions) != 5
        or dimensions[:3] != (1, 16, LATENT_FRAME_COUNT)
        or min(dimensions) <= 0
    ):
        raise SAICInferenceError(
            "noise bank shape must be Bernini [1,16,21,H,W]"
        )
    if type(candidate_schedule) is not tuple or candidate_schedule not in (
        K1_SCHEDULE,
        K5_EARLY_SCHEDULE,
    ):
        raise SAICInferenceError("noise bank candidate schedule is unregistered")
    cells: list[tuple[Any, ...]] = []
    for step, count in enumerate(candidate_schedule):
        candidates = []
        for candidate in range(count):
            generator = torch.Generator(device="cpu").manual_seed(
                keyed_noise_seed(master_seed, step, candidate)
            )
            cpu = torch.randn(
                dimensions,
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
            candidates.append(cpu.to(device=device).contiguous())
        cells.append(tuple(candidates))
    return tuple(cells)


def candidate_zero_noise_sha256(bank: Sequence[Sequence[Any]]) -> str:
    """Hash the exact candidate-zero sub-bank shared by all six arms."""

    import torch

    if len(bank) != NUM_INFERENCE_STEPS or any(len(cell) < 1 for cell in bank):
        raise SAICInferenceError("candidate-zero bank requires 40 nonempty cells")
    digest = hashlib.sha256(b"saic-candidate-zero-noise-bank-v1\0")
    for step, cell in enumerate(bank):
        value = cell[0]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise SAICInferenceError("candidate-zero bank must contain FP32 tensors")
        raw = bytes.fromhex(tensor_raw_sha256(value))
        digest.update(struct.pack(">I", step))
        digest.update(struct.pack(">I", value.ndim))
        for extent in value.shape:
            digest.update(struct.pack(">Q", int(extent)))
        digest.update(raw)
    return digest.hexdigest()


def bind_native_schedule_objects(
    *,
    scheduler_sigmas: Any,
    runtime_timesteps: Any,
    spec: ArmSpec,
    expected_pinned_schedule_sha256: Optional[str],
    expected_scheduler_sigma_fp32_sha256: Optional[str],
) -> NativeScheduleBundle:
    """Bind actual scheduler storage views to both core and native digests."""

    import torch

    if type(spec) is not ArmSpec or arm_spec(spec.arm) != spec:
        raise SAICInferenceError("schedule binding requires a registered arm")
    if (
        not isinstance(scheduler_sigmas, torch.Tensor)
        or scheduler_sigmas.dtype != torch.float32
        or scheduler_sigmas.device.type != "cpu"
        or tuple(scheduler_sigmas.shape) != (NUM_INFERENCE_STEPS + 1,)
        or scheduler_sigmas.requires_grad
        or not bool(torch.isfinite(scheduler_sigmas).all().item())
    ):
        raise SAICInferenceError(
            "scheduler.sigmas must be detached CPU FP32 with 41 values"
        )
    if (
        not isinstance(runtime_timesteps, torch.Tensor)
        or runtime_timesteps.dtype != torch.int64
        or tuple(runtime_timesteps.shape) != (NUM_INFERENCE_STEPS,)
        or runtime_timesteps.device.type == "meta"
        or runtime_timesteps.requires_grad
    ):
        raise SAICInferenceError(
            "runtime timesteps must be detached device-local INT64 with 40 values"
        )
    if struct.pack(">f", float(scheduler_sigmas[-1].item())) != b"\0\0\0\0":
        raise SAICInferenceError("scheduler terminal sigma must be bit-exact +0")
    sigma_scalars = tuple(scheduler_sigmas[index] for index in range(40))
    next_sigmas = tuple(
        float(scheduler_sigmas[index + 1].item()) for index in range(40)
    )
    timestep_tensors = tuple(
        runtime_timesteps[index : index + 1] for index in range(40)
    )
    sigma_schedule = tuple(float(value.item()) for value in scheduler_sigmas)
    if (
        sigma_schedule[0] <= 0.0
        or sigma_schedule[-1] != 0.0
        or any(right >= left for left, right in zip(sigma_schedule, sigma_schedule[1:]))
    ):
        raise SAICInferenceError("scheduler sigma values are not exact40 descending")
    previous_timestep: Optional[int] = None
    for index, (sigma, timestep) in enumerate(zip(sigma_scalars, timestep_tensors)):
        timestep_value = int(timestep.item())
        if (
            sigma.ndim != 0
            or tuple(timestep.shape) != (1,)
            or not 0 < timestep_value < 1000
            or (
                previous_timestep is not None
                and timestep_value >= previous_timestep
            )
        ):
            raise SAICInferenceError(
                f"native INT64 timestep is not strictly descending at cell {index}"
            )
        previous_timestep = timestep_value
    sigma_storage = scheduler_sigmas.untyped_storage().data_ptr()
    timestep_storage = runtime_timesteps.untyped_storage().data_ptr()
    sigma_views = all(
        item.untyped_storage().data_ptr() == sigma_storage for item in sigma_scalars
    )
    timestep_views = all(
        item.untyped_storage().data_ptr() == timestep_storage
        for item in timestep_tensors
    )
    if not sigma_views or not timestep_views:
        raise SAICInferenceError("native schedule scalar/slice objects are not direct views")
    sigma_fp32_payload = b"".join(
        struct.pack(">f", float(value.item())) for value in sigma_scalars
    )
    sigma_fp32_sha256 = hashlib.sha256(sigma_fp32_payload).hexdigest()
    schedule_payload = {
        "timesteps": [float(value.item()) for value in timestep_tensors],
        "sigmas": list(sigma_schedule),
        "flow_shift": FLOW_SHIFT,
        "steps": NUM_INFERENCE_STEPS,
    }
    pinned_sha256 = legacy.object_sha256(schedule_payload)
    for label, actual, expected in (
        (
            "pinned UniPC schedule",
            pinned_sha256,
            expected_pinned_schedule_sha256,
        ),
        (
            "scheduler sigma FP32",
            sigma_fp32_sha256,
            expected_scheduler_sigma_fp32_sha256,
        ),
    ):
        if expected is not None and actual != expected:
            raise SAICInferenceError(f"{label} digest differs")
    return NativeScheduleBundle(
        sigma_scalars=sigma_scalars,
        next_sigmas=next_sigmas,
        timestep_tensors=timestep_tensors,
        sigma_schedule=sigma_schedule,
        core_sigma_schedule_sha256=transport.sigma_schedule_sha256(
            sigma_schedule
        ),
        native_schedule_sha256=native_field.native_schedule_sha256(
            sigma_scalars,
            next_sigmas,
            timestep_tensors,
            spec.candidate_schedule,
            spec.aggregation_mode,
            spec.temperature,
        ),
        pinned_schedule_sha256=pinned_sha256,
        scheduler_sigma_fp32_sha256=sigma_fp32_sha256,
        scalar_views_share_scheduler_storage=True,
        timestep_views_share_runtime_storage=True,
    )


def prepare_native_schedule(
    diffusion: Any, device: Any, *, spec: ArmSpec
) -> NativeScheduleBundle:
    """Set the real UniPC shift-5 scheduler and capture its exact objects."""

    config = cdf.DifferentialFlowConfig(
        num_inference_steps=NUM_INFERENCE_STEPS,
        flow_shift=FLOW_SHIFT,
        seed=DEFAULT_SEED,
        motion_scale=1.0,
    )
    try:
        timesteps, intervals = cdf._set_scheduler_timesteps(diffusion, config, device)
        guided.validate_pinned_sigma_intervals(intervals)
    except Exception as error:
        raise SAICInferenceError(f"cannot establish pinned UniPC schedule: {error}") from error
    scheduler_sigmas = getattr(getattr(diffusion, "scheduler", None), "sigmas", None)
    return bind_native_schedule_objects(
        scheduler_sigmas=scheduler_sigmas,
        runtime_timesteps=timesteps,
        spec=spec,
        expected_pinned_schedule_sha256=guided.PINNED_UNIPC_SCHEDULE_DIGEST,
        expected_scheduler_sigma_fp32_sha256=(
            guided.PINNED_UNIPC_SIGMA_FP32_DIGEST
        ),
    )


def _plain_absolute_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise SAICInferenceError(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as error:
        raise SAICInferenceError(f"cannot resolve {label}") from error
    if not stat.S_ISREG(mode):
        raise SAICInferenceError(f"{label} is not a plain file")
    return resolved


def _plain_absolute_read_only_file(
    value: str | Path, *, label: str
) -> Path:
    """Resolve one immutable input without accepting a path alias."""

    requested = Path(value).expanduser()
    resolved = _plain_absolute_file(requested, label=label)
    if requested != resolved:
        raise SAICInferenceError(f"{label} path must already be canonical")
    try:
        observed = resolved.lstat()
    except OSError as error:
        raise SAICInferenceError(f"cannot stat {label}") from error
    if stat.S_IMODE(observed.st_mode) != 0o444:
        raise SAICInferenceError(f"{label} must be mode 0444")
    if observed.st_nlink != 1:
        raise SAICInferenceError(f"{label} must have exactly one filesystem link")
    return resolved


def _sealed_stat_identity(path: Path, observed: os.stat_result) -> dict[str, Any]:
    return {
        "path": str(path),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "size": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
        "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
        "link_count": int(observed.st_nlink),
    }


def _read_open_descriptor(descriptor: int, size: int, *, label: str) -> bytes:
    if type(descriptor) is not int or descriptor < 0 or type(size) is not int or size < 0:
        raise SAICInferenceError(f"{label} descriptor contract differs")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        try:
            chunk = os.pread(descriptor, min(8 * 1024 * 1024, size - offset), offset)
        except OSError as error:
            raise SAICInferenceError(f"cannot read {label}") from error
        if not chunk:
            raise SAICInferenceError(f"{label} ended before its sealed size")
        chunks.append(chunk)
        offset += len(chunk)
    try:
        trailing = os.pread(descriptor, 1, size)
    except OSError as error:
        raise SAICInferenceError(f"cannot finish reading {label}") from error
    if trailing:
        raise SAICInferenceError(f"{label} grew while it was read")
    return b"".join(chunks)


def _open_sealed_input(
    value: str | Path,
    *,
    label: str,
    expected_sha256: str,
    maximum_bytes: int,
) -> tuple[Path, int, bytes, dict[str, Any]]:
    """Open one 0444 file once and retain the descriptor for TOCTOU checks."""

    if type(expected_sha256) is not str or _SHA256.fullmatch(expected_sha256) is None:
        raise SAICInferenceError(f"{label} expected digest is malformed")
    path = _plain_absolute_read_only_file(value, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SAICInferenceError(f"cannot open {label}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise SAICInferenceError(f"{label} sealed file contract differs")
        payload = _read_open_descriptor(descriptor, int(before.st_size), label=label)
        after = os.fstat(descriptor)
        path_after = path.lstat()
        identity = _sealed_stat_identity(path, before)
        if (
            _sealed_stat_identity(path, after) != identity
            or _sealed_stat_identity(path, path_after) != identity
        ):
            raise SAICInferenceError(f"{label} identity changed while opening")
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            raise SAICInferenceError(f"{label} SHA-256 differs")
        identity["sha256"] = actual_sha256
        return path, descriptor, payload, identity
    except BaseException:
        os.close(descriptor)
        raise


def _reopen_sealed_descriptor(
    path: Path,
    descriptor: int,
    *,
    label: str,
    expected_identity: Mapping[str, Any],
) -> bytes:
    """Rehash both the retained inode and its still-bound canonical path."""

    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = path.lstat()
    except OSError as error:
        raise SAICInferenceError(f"cannot restat {label}") from error
    expected_stat = {
        key: expected_identity.get(key)
        for key in (
            "path",
            "device",
            "inode",
            "size",
            "mtime_ns",
            "ctime_ns",
            "mode",
            "link_count",
        )
    }
    if (
        path.is_symlink()
        or _sealed_stat_identity(path, descriptor_stat) != expected_stat
        or _sealed_stat_identity(path, path_stat) != expected_stat
    ):
        raise SAICInferenceError(f"{label} sealed identity changed")
    payload = _read_open_descriptor(
        descriptor, int(expected_identity["size"]), label=label
    )
    if hashlib.sha256(payload).hexdigest() != expected_identity.get("sha256"):
        raise SAICInferenceError(f"{label} sealed bytes changed")
    return payload


def runtime_source_hashes() -> dict[str, str]:
    """Hash every local method/runtime/asset byte consumed by this runner."""

    result: dict[str, str] = {}
    for relative in RUNTIME_METHOD_FILES:
        path = METHOD_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise SAICInferenceError(f"runtime source is missing/non-plain: {relative}")
        result[f"methods/bernini_action_editing/{relative}"] = legacy.file_sha256(path)
    return result


def validate_method_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Bind live runtime bytes to two identical, externally revision-labelled archives.

    The archive SHA-256 is supplied by the launcher.  This function proves
    byte equality, archive safety, the revision *label* in the Git pax header,
    and equality between every live Python/asset byte in the imported runtime
    closure and its archived copy.  It does not infer a Git revision from the
    label alone; the launcher separately uses ``git get-tar-commit-id``.
    """

    try:
        bytecode_policy = legacy_audit._bytecode_policy()
    except legacy_audit.SourceAlignedInferenceError as error:
        raise SAICInferenceError(str(error)) from error
    live = runtime_source_hashes()
    scratch = _plain_absolute_file(
        args.method_source_archive, label="scratch method archive"
    )
    durable = _plain_absolute_file(
        args.durable_method_source_archive, label="durable method archive"
    )
    scratch_sha256 = legacy.file_sha256(scratch)
    durable_sha256 = legacy.file_sha256(durable)
    if (
        scratch_sha256 != args.method_source_archive_sha256
        or durable_sha256 != scratch_sha256
    ):
        raise SAICInferenceError("scratch/durable method archive digest differs")
    member_hashes: dict[str, str] = {}
    try:
        with tarfile.open(scratch, mode="r:*") as handle:
            if handle.pax_headers.get("comment") != args.method_source_revision:
                raise SAICInferenceError("method archive revision comment differs")
            members = handle.getmembers()
            seen: set[str] = set()
            for member in members:
                pure = PurePosixPath(member.name)
                name = pure.as_posix().rstrip("/")
                scoped = name in {
                    "methods",
                    "methods/bernini_action_editing",
                } or name.startswith("methods/bernini_action_editing/")
                if (
                    not name
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or not scoped
                    or name in seen
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                    or (not member.isfile() and not member.isdir())
                ):
                    raise SAICInferenceError(
                        "method archive contains an unsafe, duplicate, or "
                        "out-of-scope member"
                    )
                seen.add(name)
            for relative in RUNTIME_ARCHIVE_MEMBERS:
                matches = [item for item in members if item.name == relative]
                if len(matches) != 1 or not matches[0].isfile():
                    raise SAICInferenceError(
                        f"method archive member differs: {relative}"
                    )
                extracted = handle.extractfile(matches[0])
                if extracted is None:
                    raise SAICInferenceError(
                        f"cannot read method archive member: {relative}"
                    )
                member_hashes[relative] = hashlib.sha256(extracted.read()).hexdigest()
    except (OSError, tarfile.TarError) as error:
        raise SAICInferenceError("cannot validate method source archive") from error
    if member_hashes != live:
        raise SAICInferenceError("live runtime bytes differ from method archive")
    return {
        "revision": args.method_source_revision,
        "scratch_archive_path": str(scratch),
        "durable_archive_path": str(durable),
        "archive_sha256": scratch_sha256,
        "archive_safe_scoped_duplicate_free_link_free": True,
        "revision_label_matches_archive_comment": True,
        "git_revision_verified_by_runner": False,
        "runtime_source_sha256": live,
        "runtime_source_index_sha256": legacy.object_sha256(live),
        "bytecode_policy": bytecode_policy,
    }


def _parse_source_clean_safetensors(
    payload: bytes,
    *,
    expected_shape: Sequence[int],
    expected_tensor_raw_sha256: str,
) -> tuple[Any, dict[str, str]]:
    """Parse one in-memory safetensors payload without reopening its path."""

    import torch
    from safetensors.torch import load as load_safetensors

    if type(payload) is not bytes or len(payload) < 9:
        raise SAICInferenceError("sealed source-clean safetensors bytes differ")
    header_size = struct.unpack("<Q", payload[:8])[0]
    if header_size <= 0 or header_size > len(payload) - 8:
        raise SAICInferenceError("sealed source-clean safetensors header differs")
    try:
        header = json.loads(payload[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SAICInferenceError(
            "cannot parse sealed source-clean safetensors header"
        ) from error
    if (
        type(header) is not dict
        or set(header) != {"__metadata__", SOURCE_CLEAN_TENSOR_KEY}
        or header.get("__metadata__") != dict(SOURCE_CLEAN_ARTIFACT_METADATA)
    ):
        raise SAICInferenceError("sealed source-clean metadata/keys differ")
    try:
        tensors = load_safetensors(payload)
    except Exception as error:
        raise SAICInferenceError("cannot load sealed source-clean tensor") from error
    if set(tensors) != {SOURCE_CLEAN_TENSOR_KEY}:
        raise SAICInferenceError("sealed source-clean tensor key differs")
    tensor = tensors[SOURCE_CLEAN_TENSOR_KEY]
    shape = tuple(int(item) for item in expected_shape)
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tuple(tensor.shape) != shape
        or tensor.requires_grad
        or tensor.grad_fn is not None
        or tensor.layout != torch.strided
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise SAICInferenceError("sealed source-clean tensor contract differs")
    stored = tensor.detach().clone().contiguous()
    actual_raw_sha256 = tensor_raw_sha256(stored)
    if actual_raw_sha256 != expected_tensor_raw_sha256:
        raise SAICInferenceError("sealed source-clean tensor raw SHA-256 differs")
    return stored, dict(SOURCE_CLEAN_ARTIFACT_METADATA)


def _parse_source_clean_receipt(payload: bytes) -> tuple[dict[str, Any], str]:
    try:
        receipt = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SAICInferenceError("cannot parse sealed source-clean receipt") from error
    expected_root_keys = {
        "schema_version",
        "method",
        "artifact",
        "sealed_inputs",
        "preprocessing",
        "model_closure",
        "encoding",
        "runtime",
        "authority",
        "receipt_digest",
    }
    if type(receipt) is not dict or set(receipt) != expected_root_keys:
        raise SAICInferenceError("sealed source-clean receipt root differs")
    if payload != legacy.canonical_json_bytes(receipt) + b"\n":
        raise SAICInferenceError("sealed source-clean receipt is not canonical JSON")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if (
        type(declared) is not str
        or _SHA256.fullmatch(declared) is None
        or legacy.object_sha256(unsigned) != declared
        or receipt.get("schema_version") != SOURCE_CLEAN_RECEIPT_SCHEMA
        or receipt.get("method") != SOURCE_CLEAN_MATERIALIZER_METHOD
    ):
        raise SAICInferenceError("sealed source-clean receipt digest/schema differs")
    return receipt, declared


def _stable_checkpoint_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "manifest_path",
        "manifest_sha256_computed",
        "manifest_sha256_expected",
        "verified_file_count",
        "every_file_sha256_verified",
        "verified_entries_digest",
    )
    return {key: value.get(key) for key in keys}


def runtime_vae_encoder_identity() -> dict[str, str]:
    """Return the active pinned encoder identity used for I0 frame 0."""

    try:
        import inspect
        from bernini.pipeline import _vae_encode as runtime_vae_encode
    except Exception as error:
        raise SAICInferenceError("cannot bind materializer VAE encoder") from error
    return {
        "encoder_symbol": "bernini.pipeline._vae_encode",
        "callable_module": runtime_vae_encode.__module__,
        "callable_name": runtime_vae_encode.__name__,
        "callable_qualname": runtime_vae_encode.__qualname__,
        "callable_signature": str(inspect.signature(runtime_vae_encode)),
    }


def _require_false_authority(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise SAICInferenceError(f"{label} authority map differs")
    for key, item in value.items():
        if type(key) is not str or not key:
            raise SAICInferenceError(f"{label} authority key differs")
        if item is not False:
            raise SAICInferenceError(f"{label} authority must remain false: {key}")


def _validate_source_clean_receipt_bindings_against_materializer(
    receipt: Mapping[str, Any],
    *,
    artifact_path: Path,
    artifact_identity: Mapping[str, Any],
    artifact_metadata: Mapping[str, str],
    receipt_path: Path,
    receipt_identity: Mapping[str, Any],
    receipt_digest: str,
    tensor: Any,
    tensor_raw_sha256: str,
    sealed_cell: Mapping[str, Any],
    sealed_assets: Mapping[str, Any],
    source_path: Path,
    source_metadata: Mapping[str, Any],
    source_pixels_raw_sha256: str,
    checkpoint_path: Path,
    checkpoint_tree_sha256: str,
    checkpoint_identity: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    bernini_inference_files: Mapping[str, str],
    method_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind materialized bytes to the source/checkpoint/method admitted here."""

    artifact = receipt.get("artifact")
    sealed = receipt.get("sealed_inputs")
    preprocessing = receipt.get("preprocessing")
    model = receipt.get("model_closure")
    encoding = receipt.get("encoding")
    runtime = receipt.get("runtime")
    authority = receipt.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (artifact, sealed, preprocessing, model, encoding, runtime)
    ):
        raise SAICInferenceError("sealed source-clean receipt sections differ")
    expected_authority_keys = {
        "quality_claim_authorized",
        "semantic_action_success_authorized",
        "ground_truth_authorized",
        "training_target_authorized",
        "selection_authorized",
        "optimizer_step_authorized",
        "checkpoint_or_lora_artifact",
        "production_claim_authorized",
    }
    if set(authority) != expected_authority_keys:
        raise SAICInferenceError("sealed source-clean authority keys differ")
    _require_false_authority(authority, label="sealed source-clean")

    forbidden_semantic_keys = {
        "event_bank",
        "target_video",
        "target_caption",
        "instruction",
        "mask",
        "pose",
        "flow",
        "track",
        "trajectory",
        "motion_donor",
        "oracle_frame",
        "generated_proposal",
    }
    for section_name, section in (
        ("sealed_inputs", sealed),
        ("preprocessing", preprocessing),
        ("model_closure", model),
        ("encoding", encoding),
        ("runtime", runtime),
    ):
        if forbidden_semantic_keys.intersection(section):
            raise SAICInferenceError(
                f"sealed source-clean {section_name} accepts semantic/oracle input"
            )
    expected_runtime_keys = {
        "device_requested",
        "world_size",
        "distributed_initialized",
        "python_version",
        "torch_version",
        "hip_version",
        "diffusers_version",
        "safetensors_version",
    }
    if (
        set(runtime) != expected_runtime_keys
        or type(runtime.get("device_requested")) is not str
        or not runtime["device_requested"].startswith("cuda:")
        or runtime.get("world_size") != 1
        or runtime.get("distributed_initialized") is not False
        or any(
            type(runtime.get(key)) is not str or not runtime[key]
            for key in (
                "python_version",
                "torch_version",
                "hip_version",
                "diffusers_version",
                "safetensors_version",
            )
        )
    ):
        raise SAICInferenceError("sealed source-clean runtime contract differs")

    expected_artifact = {
        "schema_version": SOURCE_CLEAN_ARTIFACT_SCHEMA,
        "path": str(artifact_path),
        "file_sha256": artifact_identity["sha256"],
        "size_bytes": artifact_identity["size"],
        "mode": "0444",
        "tensor_key": SOURCE_CLEAN_TENSOR_KEY,
        "tensor_raw_sha256": tensor_raw_sha256,
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
        "metadata": dict(artifact_metadata),
    }
    if set(artifact) != set(expected_artifact):
        raise SAICInferenceError("sealed source-clean artifact receipt keys differ")
    for key, expected in expected_artifact.items():
        if artifact.get(key) != expected:
            raise SAICInferenceError(
                f"sealed source-clean artifact receipt differs at {key}"
            )

    expected_sealed_keys = {
        "accepted_roles",
        "forbidden_roles",
        "source_manifest_path",
        "source_manifest_raw_sha256",
        "source_manifest_content_sha256",
        "source_manifest_schema_version",
        "source_manifest_dataset_id",
        "source_manifest_bound_files_verified",
        "row_id",
        "iid",
        "analysis_split",
        "actor_family",
        "source_video_path",
        "source_video_sha256",
        "source_video_rehashed_after_encode",
        "source_manifest_terminal_events_verified",
        "optimizer_authorized",
    }
    if set(sealed) != expected_sealed_keys:
        raise SAICInferenceError("sealed source-clean sealed-input keys differ")
    accepted_roles = sealed.get("accepted_roles")
    forbidden_roles = sealed.get("forbidden_roles")
    if (
        accepted_roles != list(SOURCE_CLEAN_ACCEPTED_INPUT_ROLES)
        or forbidden_roles != list(SOURCE_CLEAN_FORBIDDEN_INPUT_ROLES)
    ):
        raise SAICInferenceError("sealed source-clean role declaration differs")
    expected_sealed = {
        "source_manifest_path": sealed_assets.get("source_manifest_path"),
        "source_manifest_raw_sha256": sealed_assets.get(
            "source_manifest_raw_sha256"
        ),
        "source_manifest_content_sha256": sealed_assets.get(
            "source_manifest_content_sha256"
        ),
        "source_manifest_schema_version": sealed_assets.get(
            "source_manifest_schema_version"
        ),
        "source_manifest_dataset_id": sealed_assets.get(
            "source_manifest_dataset_id"
        ),
        "source_manifest_bound_files_verified": False,
        "row_id": sealed_cell.get("row_id"),
        "iid": sealed_cell.get("iid"),
        "analysis_split": sealed_cell.get("analysis_split"),
        "actor_family": sealed_cell.get("actor_family"),
        "source_video_path": str(source_path),
        "source_video_sha256": sealed_cell.get("source_video_sha256"),
        "source_video_rehashed_after_encode": True,
        "source_manifest_terminal_events_verified": False,
        "optimizer_authorized": False,
    }
    for key, expected in expected_sealed.items():
        if sealed.get(key) != expected:
            raise SAICInferenceError(
                f"sealed source-clean source binding differs at {key}"
            )

    expected_preprocessing = {
        "decoded_from_private_byte_snapshot": True,
        "frame_count": source_metadata.get("frame_count"),
        "fps": source_metadata.get("fps"),
        "reported_fps": source_metadata.get("reported_fps"),
        "source_input_hw": source_metadata.get("source_input_hw"),
        "source_derived_bucket_hw": source_metadata.get(
            "source_derived_bucket_hw"
        ),
        "max_pixels": source_metadata.get("max_pixels"),
        "stride": source_metadata.get("stride"),
        "temporal_policy": source_metadata.get("temporal_policy"),
        "spatial_policy": source_metadata.get("spatial_policy"),
        "resize": source_metadata.get("resize"),
        "external_shared_i0": source_metadata.get("external_shared_i0"),
        "source_pixels_raw_sha256": source_pixels_raw_sha256,
        "source_pixels_shape": [1, 3, FRAME_COUNT, *(
            int(item) for item in source_metadata.get("source_derived_bucket_hw", [])
        )],
        "source_pixels_dtype": "torch.float32",
    }
    if set(preprocessing) != set(expected_preprocessing):
        raise SAICInferenceError("sealed source-clean preprocessing keys differ")
    for key, expected in expected_preprocessing.items():
        if preprocessing.get(key) != expected:
            raise SAICInferenceError(
                f"sealed source-clean preprocessing differs at {key}"
            )

    expected_model = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "checkpoint_content_manifest_audit": _stable_checkpoint_identity(
            checkpoint_identity
        ),
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
        "bernini_inference_files": dict(bernini_inference_files),
        "bernini_inference_files_index_sha256": legacy.object_sha256(
            bernini_inference_files
        ),
        "method_source_revision": method_provenance.get("revision"),
        "method_source_archive_sha256": method_provenance.get("archive_sha256"),
    }
    if set(model) != set(expected_model) | {
        "runtime_source_index_sha256",
        "method_provenance",
    }:
        raise SAICInferenceError("sealed source-clean model-closure keys differ")
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise SAICInferenceError(
                f"sealed source-clean model binding differs at {key}"
            )
    materializer_method = model.get("method_provenance")
    runtime_source_index = model.get("runtime_source_index_sha256")
    expected_method_keys = {
        "revision",
        "scratch_archive_path",
        "durable_archive_path",
        "archive_sha256",
        "archive_safe_scoped_duplicate_free_link_free",
        "revision_label_matches_archive_comment",
        "git_revision_verified_by_runner",
        "runtime_source_sha256",
        "runtime_source_index_sha256",
        "bytecode_policy",
    }
    if (
        not isinstance(materializer_method, Mapping)
        or set(materializer_method) != expected_method_keys
        or _SHA256.fullmatch(str(runtime_source_index or "")) is None
        or materializer_method.get("revision") != method_provenance.get("revision")
        or materializer_method.get("archive_sha256")
        != method_provenance.get("archive_sha256")
        or materializer_method.get("scratch_archive_path")
        != method_provenance.get("scratch_archive_path")
        or materializer_method.get("durable_archive_path")
        != method_provenance.get("durable_archive_path")
        or materializer_method.get("runtime_source_index_sha256")
        != runtime_source_index
        or materializer_method.get(
            "archive_safe_scoped_duplicate_free_link_free"
        )
        is not True
        or materializer_method.get("revision_label_matches_archive_comment")
        is not True
        or materializer_method.get("git_revision_verified_by_runner") is not False
    ):
        raise SAICInferenceError("sealed source-clean method provenance differs")
    materializer_bytecode = materializer_method.get("bytecode_policy")
    if (
        not isinstance(materializer_bytecode, Mapping)
        or materializer_bytecode.get("pythondontwritebytecode_environment") != "1"
        or materializer_bytecode.get("dont_write_bytecode") is not True
        or materializer_bytecode.get("method_source_pycache_ignored") is not True
        or type(materializer_bytecode.get("resolved_private_empty_pycache_prefix"))
        is not str
    ):
        raise SAICInferenceError(
            "sealed source-clean materializer bytecode policy differs"
        )
    materializer_runtime_hashes = materializer_method.get("runtime_source_sha256")
    runner_runtime_hashes = method_provenance.get("runtime_source_sha256")
    if (
        not isinstance(materializer_runtime_hashes, Mapping)
        or not materializer_runtime_hashes
        or set(materializer_runtime_hashes)
        != set(SOURCE_CLEAN_MATERIALIZER_ARCHIVE_MEMBERS)
        or legacy.object_sha256(materializer_runtime_hashes) != runtime_source_index
        or not isinstance(runner_runtime_hashes, Mapping)
        or any(
            runner_runtime_hashes.get(key) != value
            for key, value in materializer_runtime_hashes.items()
        )
    ):
        raise SAICInferenceError(
            "sealed source-clean materializer runtime closure differs"
        )

    expected_encoding_keys = {
        "encoder_symbol",
        "callable_module",
        "callable_name",
        "callable_qualname",
        "callable_signature",
        "encoded_in_runner",
        "full_source_vae_encode_count",
        "total_vae_encode_count",
        "posterior_statistic",
        "sampling",
        "torch_inference_mode",
        "source_pixels_mutated",
        "source_pixels_before_sha256",
        "source_pixels_after_sha256",
        "vae_dtype",
        "vae_eval",
        "vae_requires_grad",
        "latent_frame_count",
        "finite",
    }
    if set(encoding) != expected_encoding_keys:
        raise SAICInferenceError("sealed source-clean encoding keys differ")
    encoder_identity = runtime_vae_encoder_identity()
    if (
        any(encoding.get(key) != value for key, value in encoder_identity.items())
        or encoding.get("full_source_vae_encode_count") != 1
        or encoding.get("total_vae_encode_count") != 1
        or encoding.get("encoded_in_runner") is not False
        or encoding.get("posterior_statistic") != "latent_dist.mode"
        or encoding.get("sampling") is not False
        or encoding.get("torch_inference_mode") is not True
        or encoding.get("source_pixels_mutated") is not False
        or encoding.get("source_pixels_before_sha256") != source_pixels_raw_sha256
        or encoding.get("source_pixels_after_sha256") != source_pixels_raw_sha256
        or encoding.get("vae_dtype") != "torch.float32"
        or encoding.get("vae_eval") is not True
        or encoding.get("vae_requires_grad") is not False
        or encoding.get("latent_frame_count") != LATENT_FRAME_COUNT
        or encoding.get("finite") is not True
    ):
        raise SAICInferenceError("sealed source-clean encoding contract differs")
    return {
        "schema_version": SOURCE_CLEAN_RECEIPT_SCHEMA,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_identity["sha256"],
        "artifact_size": artifact_identity["size"],
        "artifact_mode": "0444",
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": receipt_identity["sha256"],
        "receipt_digest": receipt_digest,
        "tensor_key": SOURCE_CLEAN_TENSOR_KEY,
        "tensor_raw_sha256": tensor_raw_sha256,
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
        "source_manifest_raw_sha256": sealed_assets[
            "source_manifest_raw_sha256"
        ],
        "source_manifest_content_sha256": sealed_assets[
            "source_manifest_content_sha256"
        ],
        "row_id": sealed_cell["row_id"],
        "source_video_sha256": sealed_cell["source_video_sha256"],
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "checkpoint_content_manifest_audit": _stable_checkpoint_identity(
            checkpoint_identity
        ),
        "source_derived_bucket_hw": list(
            source_metadata["source_derived_bucket_hw"]
        ),
        "materializer_method_source_revision": model[
            "method_source_revision"
        ],
        "materializer_method_source_archive_sha256": model[
            "method_source_archive_sha256"
        ],
        "materializer_runtime_source_index_sha256": runtime_source_index,
        "loaded_from_sealed_source_coordinate": True,
        "encoded_in_runner": False,
        "runner_reencoding_verified": False,
        "inference_available_source_video": True,
        "ground_truth": False,
        "quality_authority": False,
        "semantic_action_success": False,
        "identity_preservation_success": False,
        "training_authority": False,
        "optimizer_authority": False,
    }


def _validate_source_clean_receipt_bindings(
    receipt: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Admit the immutable Job132387 coordinate under its original closure.

    The receipt file itself is already opened read-only and checked against an
    exact launcher-supplied SHA-256.  Its embedded ec4bfb6 materializer
    provenance must therefore be validated as the upstream producer, never
    relabelled as this v2 runner's new archive.
    """

    model = receipt.get("model_closure")
    upstream = model.get("method_provenance") if isinstance(model, Mapping) else None
    current = kwargs.get("method_provenance")
    if not isinstance(upstream, Mapping) or not isinstance(current, Mapping):
        raise SAICInferenceError("inherited source-coordinate provenance differs")
    for key in (
        "revision",
        "archive_sha256",
        "scratch_archive_path",
        "durable_archive_path",
        "runtime_source_sha256",
        "runtime_source_index_sha256",
        "bytecode_policy",
    ):
        if key not in upstream:
            raise SAICInferenceError(
                "inherited source-coordinate producer closure is incomplete"
            )
    if (
        upstream.get("revision") != model.get("method_source_revision")
        or upstream.get("archive_sha256")
        != model.get("method_source_archive_sha256")
        or upstream.get("runtime_source_index_sha256")
        != model.get("runtime_source_index_sha256")
        or not isinstance(current.get("runtime_source_sha256"), Mapping)
        or any(
            current["runtime_source_sha256"].get(path) != digest
            for path, digest in upstream["runtime_source_sha256"].items()
        )
    ):
        raise SAICInferenceError(
            "inherited source-coordinate producer/runtime bytes differ"
        )
    delegated = dict(kwargs)
    delegated["method_provenance"] = dict(upstream)
    provenance = _validate_source_clean_receipt_bindings_against_materializer(
        receipt, **delegated
    )
    return {
        **provenance,
        "reused_from_slurm_job_id": "132387",
        "upstream_producer_revision": upstream["revision"],
        "upstream_producer_archive_sha256": upstream["archive_sha256"],
        "admitted_by_current_runtime_source_index_sha256": current[
            "runtime_source_index_sha256"
        ],
        "upstream_coordinate_relabelled": False,
    }


def load_sealed_source_coordinate(
    args: argparse.Namespace,
    *,
    sealed_cell: Mapping[str, Any],
    sealed_assets: Mapping[str, Any],
    source_path: Path,
    source_tensor: Any,
    source_metadata: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_identity: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    bernini_inference_files: Mapping[str, str],
    method_provenance: Mapping[str, Any],
    expected_shape: Sequence[int],
) -> SealedSourceCoordinate:
    """Load and fully bind one shared source-clean coordinate on CPU."""

    artifact_fd: Optional[int] = None
    receipt_fd: Optional[int] = None
    try:
        artifact_path, artifact_fd, artifact_payload, artifact_identity = (
            _open_sealed_input(
                args.source_clean_latent,
                label="sealed source-clean latent",
                expected_sha256=args.expected_source_clean_latent_sha256,
                maximum_bytes=2 * 1024 * 1024 * 1024,
            )
        )
        receipt_path, receipt_fd, receipt_payload, receipt_identity = (
            _open_sealed_input(
                args.source_clean_latent_receipt,
                label="sealed source-clean latent receipt",
                expected_sha256=(
                    args.expected_source_clean_latent_receipt_sha256
                ),
                maximum_bytes=4 * 1024 * 1024,
            )
        )
        if receipt_path != artifact_path.with_name(
            f"{artifact_path.name}.receipt.json"
        ):
            raise SAICInferenceError("sealed source-clean receipt path differs")
        tensor, artifact_metadata = _parse_source_clean_safetensors(
            artifact_payload,
            expected_shape=expected_shape,
            expected_tensor_raw_sha256=(
                args.expected_source_clean_tensor_raw_sha256
            ),
        )
        receipt, receipt_digest = _parse_source_clean_receipt(receipt_payload)
        source_pixels_raw_sha256 = tensor_raw_sha256(source_tensor)
        provenance = _validate_source_clean_receipt_bindings(
            receipt,
            artifact_path=artifact_path,
            artifact_identity=artifact_identity,
            artifact_metadata=artifact_metadata,
            receipt_path=receipt_path,
            receipt_identity=receipt_identity,
            receipt_digest=receipt_digest,
            tensor=tensor,
            tensor_raw_sha256=args.expected_source_clean_tensor_raw_sha256,
            sealed_cell=sealed_cell,
            sealed_assets=sealed_assets,
            source_path=source_path,
            source_metadata=source_metadata,
            source_pixels_raw_sha256=source_pixels_raw_sha256,
            checkpoint_path=checkpoint_path,
            checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
            checkpoint_identity=checkpoint_identity,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            bernini_inference_files=bernini_inference_files,
            method_provenance=method_provenance,
        )
        return SealedSourceCoordinate(
            artifact_path=artifact_path,
            receipt_path=receipt_path,
            artifact_fd=artifact_fd,
            receipt_fd=receipt_fd,
            artifact_identity=dict(artifact_identity),
            receipt_identity=dict(receipt_identity),
            receipt=dict(receipt),
            tensor=tensor,
            provenance=provenance,
        )
    except BaseException:
        if receipt_fd is not None:
            os.close(receipt_fd)
        if artifact_fd is not None:
            os.close(artifact_fd)
        raise


def revalidate_sealed_source_coordinate(
    coordinate: SealedSourceCoordinate, *, stage: str
) -> dict[str, Any]:
    """Reopen the retained source coordinate at a named execution boundary."""

    if stage not in {"pre_rollout", "pre_publish", "terminal"}:
        raise SAICInferenceError("sealed source-clean revalidation stage differs")
    artifact_payload = _reopen_sealed_descriptor(
        coordinate.artifact_path,
        coordinate.artifact_fd,
        label="sealed source-clean latent",
        expected_identity=coordinate.artifact_identity,
    )
    receipt_payload = _reopen_sealed_descriptor(
        coordinate.receipt_path,
        coordinate.receipt_fd,
        label="sealed source-clean latent receipt",
        expected_identity=coordinate.receipt_identity,
    )
    tensor, metadata = _parse_source_clean_safetensors(
        artifact_payload,
        expected_shape=coordinate.provenance["shape"],
        expected_tensor_raw_sha256=coordinate.provenance["tensor_raw_sha256"],
    )
    receipt, receipt_digest = _parse_source_clean_receipt(receipt_payload)
    if (
        metadata != dict(SOURCE_CLEAN_ARTIFACT_METADATA)
        or tensor_raw_sha256(tensor) != coordinate.provenance["tensor_raw_sha256"]
        or receipt != dict(coordinate.receipt)
        or receipt_digest != coordinate.provenance["receipt_digest"]
    ):
        raise SAICInferenceError("sealed source-clean terminal content differs")
    return {
        "stage": stage,
        "artifact_sha256": coordinate.provenance["artifact_sha256"],
        "receipt_file_sha256": coordinate.provenance["receipt_file_sha256"],
        "receipt_digest": coordinate.provenance["receipt_digest"],
        "tensor_raw_sha256": coordinate.provenance["tensor_raw_sha256"],
        "retained_descriptor_identity_verified": True,
        "canonical_path_identity_verified": True,
        "mode_0444_verified": True,
        "canonical_receipt_and_digest_verified": True,
        "tensor_reopened_byte_exact": True,
    }


def close_sealed_source_coordinate(coordinate: SealedSourceCoordinate) -> None:
    errors = []
    for descriptor in (coordinate.receipt_fd, coordinate.artifact_fd):
        try:
            os.close(descriptor)
        except OSError as error:
            errors.append(error)
    if errors:
        raise SAICInferenceError("cannot close sealed source-clean descriptors")


def _parse_frame0_safetensors(
    payload: bytes,
    *,
    expected_shape: Sequence[int],
    expected_tensor_raw_sha256: str,
) -> tuple[Any, dict[str, str]]:
    import torch
    from safetensors.torch import load as load_safetensors

    if type(payload) is not bytes or len(payload) < 9:
        raise SAICInferenceError("sealed frame0 safetensors bytes differ")
    header_size = struct.unpack("<Q", payload[:8])[0]
    if header_size <= 0 or header_size > len(payload) - 8:
        raise SAICInferenceError("sealed frame0 safetensors header differs")
    try:
        header = json.loads(payload[8 : 8 + header_size].decode("utf-8"))
        tensors = load_safetensors(payload)
    except Exception as error:
        raise SAICInferenceError("cannot parse sealed frame0 tensor") from error
    if (
        type(header) is not dict
        or set(header) != {"__metadata__", FRAME0_TENSOR_KEY}
        or header.get("__metadata__") != dict(FRAME0_ARTIFACT_METADATA)
        or set(tensors) != {FRAME0_TENSOR_KEY}
    ):
        raise SAICInferenceError("sealed frame0 metadata/keys differ")
    tensor = tensors[FRAME0_TENSOR_KEY]
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tuple(tensor.shape) != tuple(int(item) for item in expected_shape)
        or tensor.requires_grad
        or tensor.grad_fn is not None
        or tensor.layout != torch.strided
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise SAICInferenceError("sealed frame0 tensor contract differs")
    stored = tensor.detach().clone().contiguous()
    if tensor_raw_sha256(stored) != expected_tensor_raw_sha256:
        raise SAICInferenceError("sealed frame0 tensor raw SHA-256 differs")
    return stored, dict(FRAME0_ARTIFACT_METADATA)


def _parse_frame0_receipt(payload: bytes) -> tuple[dict[str, Any], str]:
    try:
        receipt = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SAICInferenceError("cannot parse sealed frame0 receipt") from error
    if (
        type(receipt) is not dict
        or set(receipt)
        != {
            "schema_version",
            "method",
            "artifact",
            "sealed_inputs",
            "preprocessing",
            "model_closure",
            "encoding",
            "runtime",
            "authority",
            "receipt_digest",
        }
        or payload != legacy.canonical_json_bytes(receipt) + b"\n"
    ):
        raise SAICInferenceError("sealed frame0 receipt shape/canonical bytes differ")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if (
        type(declared) is not str
        or _SHA256.fullmatch(declared) is None
        or legacy.object_sha256(unsigned) != declared
        or receipt.get("schema_version") != FRAME0_RECEIPT_SCHEMA
        or receipt.get("method") != FRAME0_MATERIALIZER_METHOD
    ):
        raise SAICInferenceError("sealed frame0 receipt digest/schema differs")
    return receipt, declared


def load_sealed_frame0_coordinate(
    args: argparse.Namespace,
    *,
    sealed_cell: Mapping[str, Any],
    sealed_assets: Mapping[str, Any],
    method_provenance: Mapping[str, Any],
    expected_shape: Sequence[int],
) -> SealedFrame0Coordinate:
    """Load the one fresh frame0 coordinate shared by R00 and R11."""

    artifact_fd: Optional[int] = None
    receipt_fd: Optional[int] = None
    try:
        artifact_path, artifact_fd, artifact_payload, artifact_identity = (
            _open_sealed_input(
                args.reference_frame0_latent,
                label="sealed reference-frame0 latent",
                expected_sha256=args.expected_reference_frame0_latent_sha256,
                maximum_bytes=256 * 1024 * 1024,
            )
        )
        receipt_path, receipt_fd, receipt_payload, receipt_identity = (
            _open_sealed_input(
                args.reference_frame0_latent_receipt,
                label="sealed reference-frame0 latent receipt",
                expected_sha256=(
                    args.expected_reference_frame0_latent_receipt_sha256
                ),
                maximum_bytes=4 * 1024 * 1024,
            )
        )
        if receipt_path != artifact_path.with_name(
            f"{artifact_path.name}.receipt.json"
        ):
            raise SAICInferenceError("sealed frame0 receipt path differs")
        tensor, metadata = _parse_frame0_safetensors(
            artifact_payload,
            expected_shape=expected_shape,
            expected_tensor_raw_sha256=(
                args.expected_reference_frame0_tensor_raw_sha256
            ),
        )
        receipt, receipt_digest = _parse_frame0_receipt(receipt_payload)
        artifact = receipt.get("artifact")
        sealed = receipt.get("sealed_inputs")
        model = receipt.get("model_closure")
        encoding = receipt.get("encoding")
        authority = receipt.get("authority")
        if not all(
            isinstance(value, Mapping)
            for value in (artifact, sealed, model, encoding, authority)
        ):
            raise SAICInferenceError("sealed frame0 receipt sections differ")
        expected_artifact = {
            "schema_version": FRAME0_ARTIFACT_SCHEMA,
            "path": str(artifact_path),
            "file_sha256": artifact_identity["sha256"],
            "size_bytes": artifact_identity["size"],
            "mode": "0444",
            "tensor_key": FRAME0_TENSOR_KEY,
            "tensor_raw_sha256": args.expected_reference_frame0_tensor_raw_sha256,
            "shape": [int(item) for item in tensor.shape],
            "dtype": str(tensor.dtype),
            "metadata": dict(metadata),
        }
        if dict(artifact) != expected_artifact:
            raise SAICInferenceError("sealed frame0 artifact receipt differs")
        if (
            sealed.get("source_manifest_path")
            != sealed_assets.get("source_manifest_path")
            or sealed.get("source_manifest_raw_sha256")
            != sealed_assets.get("source_manifest_raw_sha256")
            or sealed.get("row_id") != sealed_cell.get("row_id")
            or sealed.get("source_video_sha256")
            != sealed_cell.get("source_video_sha256")
            or sealed.get("optimizer_authorized") is not False
        ):
            raise SAICInferenceError("sealed frame0 source binding differs")
        producer = model.get("method_provenance")
        if (
            model.get("checkpoint_tree_sha256")
            != args.expected_checkpoint_tree_sha256
            or model.get("method_source_revision")
            != method_provenance.get("revision")
            or model.get("method_source_archive_sha256")
            != method_provenance.get("archive_sha256")
            or not isinstance(producer, Mapping)
            or producer.get("runtime_source_index_sha256")
            != model.get("runtime_source_index_sha256")
            or producer.get("archive_sha256")
            != method_provenance.get("archive_sha256")
            or producer.get("revision") != method_provenance.get("revision")
        ):
            raise SAICInferenceError("sealed frame0 producer closure differs")
        producer_hashes = producer.get("runtime_source_sha256")
        current_hashes = method_provenance.get("runtime_source_sha256")
        if (
            not isinstance(producer_hashes, Mapping)
            or not isinstance(current_hashes, Mapping)
            or any(current_hashes.get(key) != value for key, value in producer_hashes.items())
        ):
            raise SAICInferenceError("sealed frame0 producer bytes differ")
        if (
            encoding.get("full_source_vae_encode_count") != 0
            or encoding.get("source_frame0_vae_encode_count") != 1
            or encoding.get("total_vae_encode_count") != 1
            or encoding.get("source_rgb_indices") != [0]
            or encoding.get("temporal_video_latent_slice_used") is not False
            or _SHA256.fullmatch(
                str(encoding.get("expected_job132387_frame0_tensor_raw_sha256", ""))
            )
            is None
            or encoding.get("actual_reference_frame0_tensor_raw_sha256")
            != args.expected_reference_frame0_tensor_raw_sha256
            or type(encoding.get("job132387_frame0_tensor_raw_sha256_match"))
            is not bool
            or encoding.get("job132387_frame0_tensor_raw_sha256_match")
            is not (
                encoding.get("expected_job132387_frame0_tensor_raw_sha256")
                == args.expected_reference_frame0_tensor_raw_sha256
            )
            or encoding.get("encoded_in_runner") is not False
        ):
            raise SAICInferenceError("sealed frame0 encoding contract differs")
        _require_false_authority(authority, label="sealed frame0")
        provenance = {
            "schema_version": FRAME0_RECEIPT_SCHEMA,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_identity["sha256"],
            "receipt_path": str(receipt_path),
            "receipt_file_sha256": receipt_identity["sha256"],
            "receipt_digest": receipt_digest,
            "tensor_raw_sha256": args.expected_reference_frame0_tensor_raw_sha256,
            "shape": [int(item) for item in tensor.shape],
            "dtype": str(tensor.dtype),
            "row_id": sealed_cell["row_id"],
            "source_video_sha256": sealed_cell["source_video_sha256"],
            "fresh_materialization_for_this_release": True,
            "job132387_ephemeral_i0_tensor_raw_sha256": encoding[
                "expected_job132387_frame0_tensor_raw_sha256"
            ],
            "matches_job132387_ephemeral_i0_tensor": encoding[
                "job132387_frame0_tensor_raw_sha256_match"
            ],
            "encoded_in_runner": False,
            "training_authority": False,
            "selection_authority": False,
            "optimizer_authority": False,
        }
        return SealedFrame0Coordinate(
            artifact_path=artifact_path,
            receipt_path=receipt_path,
            artifact_fd=artifact_fd,
            receipt_fd=receipt_fd,
            artifact_identity=dict(artifact_identity),
            receipt_identity=dict(receipt_identity),
            receipt=dict(receipt),
            tensor=tensor,
            provenance=provenance,
        )
    except BaseException:
        if receipt_fd is not None:
            os.close(receipt_fd)
        if artifact_fd is not None:
            os.close(artifact_fd)
        raise


def revalidate_sealed_frame0_coordinate(
    coordinate: SealedFrame0Coordinate, *, stage: str
) -> dict[str, Any]:
    if stage not in {"pre_rollout", "pre_publish", "terminal"}:
        raise SAICInferenceError("sealed frame0 revalidation stage differs")
    artifact_payload = _reopen_sealed_descriptor(
        coordinate.artifact_path,
        coordinate.artifact_fd,
        label="sealed reference-frame0 latent",
        expected_identity=coordinate.artifact_identity,
    )
    receipt_payload = _reopen_sealed_descriptor(
        coordinate.receipt_path,
        coordinate.receipt_fd,
        label="sealed reference-frame0 receipt",
        expected_identity=coordinate.receipt_identity,
    )
    tensor, metadata = _parse_frame0_safetensors(
        artifact_payload,
        expected_shape=coordinate.provenance["shape"],
        expected_tensor_raw_sha256=coordinate.provenance["tensor_raw_sha256"],
    )
    receipt, receipt_digest = _parse_frame0_receipt(receipt_payload)
    if (
        metadata != dict(FRAME0_ARTIFACT_METADATA)
        or tensor_raw_sha256(tensor) != coordinate.provenance["tensor_raw_sha256"]
        or receipt != dict(coordinate.receipt)
        or receipt_digest != coordinate.provenance["receipt_digest"]
    ):
        raise SAICInferenceError("sealed frame0 terminal content differs")
    return {
        "stage": stage,
        "artifact_sha256": coordinate.provenance["artifact_sha256"],
        "receipt_file_sha256": coordinate.provenance["receipt_file_sha256"],
        "tensor_raw_sha256": coordinate.provenance["tensor_raw_sha256"],
        "retained_descriptor_identity_verified": True,
        "canonical_path_identity_verified": True,
        "mode_0444_verified": True,
    }


def close_sealed_frame0_coordinate(coordinate: SealedFrame0Coordinate) -> None:
    errors = []
    for descriptor in (coordinate.receipt_fd, coordinate.artifact_fd):
        try:
            os.close(descriptor)
        except OSError as error:
            errors.append(error)
    if errors:
        raise SAICInferenceError("cannot close sealed frame0 descriptors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--source-manifest",
        default=str(source_set.ASSET_PATH),
        help="absolute sealed saic_reversible_source_set_v1.json",
    )
    parser.add_argument(
        "--expected-source-manifest-sha256",
        default=SOURCE_MANIFEST_RAW_SHA256,
    )
    parser.add_argument(
        "--event-bank",
        default=str(event_bank.ASSET_PATH),
        help="absolute sealed saic_pure_t2v_event_bank_v1.json",
    )
    parser.add_argument(
        "--expected-event-bank-sha256", default=EVENT_BANK_RAW_SHA256
    )
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--branch", required=True, choices=("forward",))
    parser.add_argument("--rollout-seed", required=True, type=int)
    parser.add_argument("--arm", required=True, choices=_REGISTERED_ARM_NAMES)
    parser.add_argument("--source-clean-latent", required=True)
    parser.add_argument("--source-clean-latent-receipt", required=True)
    parser.add_argument(
        "--expected-source-clean-latent-sha256", required=True
    )
    parser.add_argument(
        "--expected-source-clean-latent-receipt-sha256", required=True
    )
    parser.add_argument(
        "--expected-source-clean-tensor-raw-sha256", required=True
    )
    parser.add_argument("--reference-frame0-latent", required=True)
    parser.add_argument("--reference-frame0-latent-receipt", required=True)
    parser.add_argument(
        "--expected-reference-frame0-latent-sha256", required=True
    )
    parser.add_argument(
        "--expected-reference-frame0-latent-receipt-sha256", required=True
    )
    parser.add_argument(
        "--expected-reference-frame0-tensor-raw-sha256", required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive", required=True)
    parser.add_argument("--durable-method-source-archive", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> ArmSpec:
    spec = arm_spec(args.arm)
    if args.branch != "forward":
        raise SAICInferenceError("fixed-R2V diagnostic is fixed to forward")
    if type(args.rollout_seed) is not int or not 0 <= args.rollout_seed < 2**63:
        raise SAICInferenceError("rollout_seed must be in [0,2^63)")
    if (
        type(args.row_id) is not str
        or not args.row_id
        or args.row_id != args.row_id.strip()
        or _SAFE_BASENAME.fullmatch(args.row_id) is None
    ):
        raise SAICInferenceError("row_id is not path-safe stripped text")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name)
        if type(value) is not str or _SHA1.fullmatch(value) is None:
            raise SAICInferenceError(f"{name} must be a full lowercase SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "expected_source_manifest_sha256",
        "expected_event_bank_sha256",
        "method_source_archive_sha256",
        "expected_source_clean_latent_sha256",
        "expected_source_clean_latent_receipt_sha256",
        "expected_source_clean_tensor_raw_sha256",
        "expected_reference_frame0_latent_sha256",
        "expected_reference_frame0_latent_receipt_sha256",
        "expected_reference_frame0_tensor_raw_sha256",
    ):
        value = getattr(args, name)
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise SAICInferenceError(f"{name} must be a lowercase SHA-256")
    exact = {
        "expected_bernini_commit": legacy.trainer.BERNINI_OFFICIAL_COMMIT,
        "expected_veomni_commit": legacy.trainer.VEOMNI_TESTED_COMMIT,
        "expected_checkpoint_tree_sha256": legacy.trainer.CHECKPOINT_TREE_SHA256,
        "expected_source_manifest_sha256": SOURCE_MANIFEST_RAW_SHA256,
        "expected_event_bank_sha256": EVENT_BANK_RAW_SHA256,
    }
    for name, expected in exact.items():
        if getattr(args, name) != expected:
            raise SAICInferenceError(f"unsupported pinned identity at {name}")
    try:
        output, receipt = legacy._resolve_output(args.output)
    except legacy.InferenceContractError as error:
        raise SAICInferenceError(str(error)) from error
    if output == Path("/") or receipt == Path("/"):
        raise SAICInferenceError("output may not resolve to filesystem root")
    resolve_normalized_clean_latent_output(output)
    source_clean = _plain_absolute_read_only_file(
        args.source_clean_latent, label="sealed source-clean latent"
    )
    source_clean_receipt = _plain_absolute_read_only_file(
        args.source_clean_latent_receipt,
        label="sealed source-clean latent receipt",
    )
    if source_clean.suffix != ".safetensors":
        raise SAICInferenceError("sealed source-clean latent must be safetensors")
    if source_clean_receipt != source_clean.with_name(
        f"{source_clean.name}.receipt.json"
    ):
        raise SAICInferenceError(
            "sealed source-clean receipt must use <artifact>.receipt.json"
        )
    frame0 = _plain_absolute_read_only_file(
        args.reference_frame0_latent, label="sealed reference-frame0 latent"
    )
    frame0_receipt = _plain_absolute_read_only_file(
        args.reference_frame0_latent_receipt,
        label="sealed reference-frame0 latent receipt",
    )
    if frame0.suffix != ".safetensors":
        raise SAICInferenceError("sealed reference-frame0 latent must be safetensors")
    if frame0_receipt != frame0.with_name(f"{frame0.name}.receipt.json"):
        raise SAICInferenceError(
            "sealed reference-frame0 receipt must use <artifact>.receipt.json"
        )
    if frame0.parent != output.parent:
        raise SAICInferenceError(
            "fresh frame0 coordinate must be a sibling of both arm outputs"
        )
    return spec


def resolve_output(value: str | Path) -> tuple[Path, Path]:
    try:
        return legacy._resolve_output(value)
    except legacy.InferenceContractError as error:
        raise SAICInferenceError(str(error)) from error


def resolve_normalized_clean_latent_output(output_path: str | Path) -> Path:
    """Resolve the create-only FP32 transport endpoint beside one MP4.

    ``<output>.normalized-clean-latent.safetensors`` is intentionally derived
    rather than accepted from the CLI, so it cannot escape the output
    transaction or be redirected to an unrelated downstream artifact.
    """

    output = Path(output_path)
    if (
        not output.is_absolute()
        or output.suffix.lower() != ".mp4"
        or output == Path("/")
    ):
        raise SAICInferenceError("clean-latent owner must be an absolute MP4 path")
    path = output.with_name(
        f"{output.name}.normalized-clean-latent.safetensors"
    )
    if path.exists() or path.is_symlink():
        raise SAICInferenceError(
            f"refusing to overwrite existing normalized clean latent: {path}"
        )
    if path.parent != output.parent:
        raise SAICInferenceError("clean-latent output escaped MP4 parent")
    return path


def validate_all_rank_runtime(
    rows: Sequence[Mapping[str, Any]], *, spec: ArmSpec
) -> dict[str, Any]:
    """Require one bit-identical execution certificate on all four ranks."""

    if (
        isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or len(rows) != ULYSSES_SIZE
    ):
        raise SAICInferenceError("runtime requires exactly four rank rows")
    if type(spec) is not ArmSpec or arm_spec(spec.arm) != spec:
        raise SAICInferenceError("rank validation requires a registered arm")
    canonical = []
    for item in rows:
        if not isinstance(item, Mapping):
            raise SAICInferenceError("rank runtime row is not a mapping")
        if set(item) != {
            "rank",
            "local_rank",
            "world_size",
            "ulysses_size",
            "certificate",
        }:
            raise SAICInferenceError("rank runtime row keys differ")
        if item["world_size"] != ULYSSES_SIZE or item["ulysses_size"] != ULYSSES_SIZE:
            raise SAICInferenceError("rank runtime world/Ulysses size differs")
        if not isinstance(item["certificate"], Mapping):
            raise SAICInferenceError("rank execution certificate is missing")
        canonical.append(dict(item))
    if {item["rank"] for item in canonical} != set(range(ULYSSES_SIZE)):
        raise SAICInferenceError("rank IDs are not exactly 0..3")
    if {item["local_rank"] for item in canonical} != set(range(ULYSSES_SIZE)):
        raise SAICInferenceError("local rank IDs are not exactly 0..3")
    ordered = sorted(canonical, key=lambda item: int(item["rank"]))
    reference = dict(ordered[0]["certificate"])
    if any(dict(item["certificate"]) != reference for item in ordered[1:]):
        raise SAICInferenceError("execution certificate differs across ranks")
    if (
        reference.get("arm") != spec.arm
        or reference.get("source_video_sha256") is None
        or reference.get("noise_bank_sha256") is None
        or reference.get("native_guided_query_attempt_count")
        != spec.expected_guided_queries
        or reference.get("native_guided_query_success_count")
        != spec.expected_guided_queries
        or reference.get("native_raw_transformer_forward_attempt_count")
        != spec.expected_raw_forwards
        or reference.get("native_raw_transformer_forward_success_count")
        != spec.expected_raw_forwards
        or reference.get("core_native_guided_count_reconciled") is not True
        or reference.get("core_native_raw_forward_count_reconciled") is not True
        or reference.get("model_freeze_unchanged") is not True
        or reference.get("loaded_from_sealed_source_coordinate") is not True
        or reference.get("source_clean_encoded_in_runner") is not False
    ):
        raise SAICInferenceError("rank execution certificate lacks fixed facts")
    for label in (
        "source_video_sha256",
        "source_latent_raw_sha256",
        "generated_latent_raw_sha256",
        "noise_bank_sha256",
        "candidate_zero_noise_sha256",
        "native_schedule_sha256",
        "core_sigma_schedule_sha256",
        "model_receipt_sha256",
        "runtime_source_index_sha256",
    ):
        if _SHA256.fullmatch(str(reference.get(label, ""))) is None:
            raise SAICInferenceError(f"rank certificate {label} is malformed")
    sealed_coordinate = reference.get("sealed_source_coordinate")
    if (
        not isinstance(sealed_coordinate, Mapping)
        or sealed_coordinate.get("loaded_from_sealed_source_coordinate") is not True
        or sealed_coordinate.get("encoded_in_runner") is not False
        or sealed_coordinate.get("runner_reencoding_verified") is not False
        or sealed_coordinate.get("inference_available_source_video") is not True
        or sealed_coordinate.get("ground_truth") is not False
        or sealed_coordinate.get("quality_authority") is not False
        or sealed_coordinate.get("semantic_action_success") is not False
        or sealed_coordinate.get("identity_preservation_success") is not False
        or sealed_coordinate.get("training_authority") is not False
        or sealed_coordinate.get("optimizer_authority") is not False
        or sealed_coordinate.get("tensor_raw_sha256")
        != reference.get("source_latent_raw_sha256")
        or sealed_coordinate.get("cpu_to_gpu_byte_exact") is not True
        or sealed_coordinate.get("rank0_broadcast_before_renderer") is not True
        or sealed_coordinate.get("all_rank_identity_after_broadcast") is not True
        or sealed_coordinate.get("terminal_rehash_recorded_in_this_receipt")
        is not False
        or sealed_coordinate.get("terminal_rehash_required_for_process_success")
        is not True
    ):
        raise SAICInferenceError("rank sealed source-coordinate facts differ")
    for label in (
        "artifact_sha256",
        "receipt_file_sha256",
        "receipt_digest",
        "tensor_raw_sha256",
        "source_manifest_raw_sha256",
        "source_manifest_content_sha256",
        "source_video_sha256",
        "checkpoint_tree_sha256",
        "materializer_method_source_archive_sha256",
        "materializer_runtime_source_index_sha256",
    ):
        if _SHA256.fullmatch(str(sealed_coordinate.get(label, ""))) is None:
            raise SAICInferenceError(
                f"rank sealed source-coordinate {label} is malformed"
            )
    for stage in ("pre_rollout", "pre_publish"):
        rehash = sealed_coordinate.get(f"{stage}_rehash")
        if (
            not isinstance(rehash, Mapping)
            or rehash.get("stage") != stage
            or rehash.get("tensor_raw_sha256")
            != sealed_coordinate.get("tensor_raw_sha256")
            or rehash.get("retained_descriptor_identity_verified") is not True
            or rehash.get("canonical_path_identity_verified") is not True
            or rehash.get("mode_0444_verified") is not True
            or rehash.get("canonical_receipt_and_digest_verified") is not True
            or rehash.get("tensor_reopened_byte_exact") is not True
        ):
            raise SAICInferenceError(
                f"rank sealed source-coordinate {stage} rehash differs"
            )
    sealed_frame0 = reference.get("sealed_frame0_coordinate")
    if (
        not isinstance(sealed_frame0, Mapping)
        or sealed_frame0.get("encoded_in_runner") is not False
        or sealed_frame0.get("fresh_materialization_for_this_release") is not True
        or type(sealed_frame0.get("matches_job132387_ephemeral_i0_tensor"))
        is not bool
        or _SHA256.fullmatch(
            str(sealed_frame0.get("job132387_ephemeral_i0_tensor_raw_sha256", ""))
        )
        is None
        or sealed_frame0.get("training_authority") is not False
        or sealed_frame0.get("selection_authority") is not False
        or sealed_frame0.get("optimizer_authority") is not False
        or sealed_frame0.get("tensor_raw_sha256")
        != reference.get("shared_frame0_raw_sha256")
        or sealed_frame0.get("cpu_to_gpu_byte_exact") is not True
        or sealed_frame0.get("all_rank_identity_after_broadcast") is not True
    ):
        raise SAICInferenceError("rank sealed frame0-coordinate facts differ")
    for stage in ("pre_rollout", "pre_publish"):
        rehash = sealed_frame0.get(f"{stage}_rehash")
        if (
            not isinstance(rehash, Mapping)
            or rehash.get("stage") != stage
            or rehash.get("tensor_raw_sha256")
            != sealed_frame0.get("tensor_raw_sha256")
            or rehash.get("retained_descriptor_identity_verified") is not True
            or rehash.get("canonical_path_identity_verified") is not True
            or rehash.get("mode_0444_verified") is not True
        ):
            raise SAICInferenceError(
                f"rank sealed frame0-coordinate {stage} rehash differs"
            )
    adapter = reference.get("native_adapter")
    initial_seals = (
        adapter.get("initial_model_content_seal_sha256_by_module")
        if isinstance(adapter, Mapping)
        else None
    )
    final_seals = (
        adapter.get("final_model_content_seal_sha256_by_module")
        if isinstance(adapter, Mapping)
        else None
    )
    expected_seal_labels = {"diffusion", "transformer"}
    try:
        initial_seal_map = dict(initial_seals)
        final_seal_map = dict(final_seals)
    except (TypeError, ValueError):
        initial_seal_map = {}
        final_seal_map = {}
    seals_valid = (
        isinstance(initial_seals, (list, tuple))
        and isinstance(final_seals, (list, tuple))
        and len(initial_seals) == len(expected_seal_labels)
        and len(final_seals) == len(expected_seal_labels)
        and initial_seal_map == final_seal_map
        and set(initial_seal_map) == expected_seal_labels
        and all(
            _SHA256.fullmatch(str(value)) is not None
            for value in initial_seal_map.values()
        )
    )
    if (
        not isinstance(adapter, Mapping)
        or adapter.get("field_regime") != spec.field_regime
        or adapter.get("rollout_complete") is not True
        or adapter.get("adapter_failed") is not False
        or adapter.get("failure_stage") is not None
        or adapter.get("initial_full_model_content_audit") is not True
        or adapter.get("final_full_model_content_audit") is not True
        or adapter.get("guided_query_count") != spec.expected_guided_queries
        or adapter.get("raw_transformer_forward_count")
        != spec.expected_raw_forwards
        or adapter.get("expected_guided_query_count")
        != spec.expected_guided_queries
        or adapter.get("expected_raw_transformer_forward_count")
        != spec.expected_raw_forwards
        or adapter.get("guided_query_attempt_count")
        != spec.expected_guided_queries
        or adapter.get("guided_query_success_count")
        != spec.expected_guided_queries
        or adapter.get("raw_transformer_forward_attempt_count")
        != spec.expected_raw_forwards
        or adapter.get("raw_transformer_forward_success_count")
        != spec.expected_raw_forwards
        or adapter.get("patch_query_attempt_count")
        != spec.expected_guided_queries
        or adapter.get("patch_query_success_count")
        != spec.expected_guided_queries
        or adapter.get("guided_query_count")
        != adapter.get("guided_query_success_count")
        or adapter.get("raw_transformer_forward_count")
        != adapter.get("raw_transformer_forward_attempt_count")
        or adapter.get("patch_query_count")
        != adapter.get("patch_query_attempt_count")
        or adapter.get("patch_reference_count")
        != adapter.get("patch_reference_attempt_count")
        or adapter.get("patch_reference_attempt_count")
        != adapter.get("patch_reference_success_count")
        or adapter.get("vendor_single_attempt_count")
        != adapter.get("vendor_single_success_count")
        or adapter.get("vendor_chain_attempt_count")
        != adapter.get("vendor_chain_success_count")
        or adapter.get("vendor_single_attempt_count")
        != (0 if spec.uses_reference_frame0 else spec.expected_guided_queries)
        or adapter.get("vendor_chain_attempt_count")
        != (spec.expected_guided_queries if spec.uses_reference_frame0 else 0)
        or (
            not spec.uses_reference_frame0
            and adapter.get("patch_reference_attempt_count") != 0
        )
        or (
            spec.uses_reference_frame0
            and adapter.get("patch_reference_attempt_count")
            != spec.expected_guided_queries
        )
        or adapter.get("model_checkpoint_use_verified") is not False
        or adapter.get("target_tail_direct_view") is not True
        or adapter.get("optimizer_step_allowed") is not False
        or adapter.get("training_update_allowed") is not False
        or adapter.get("semantic_action_success") is not False
        or not seals_valid
    ):
        raise SAICInferenceError("native adapter finalization facts differ")
    core = reference.get("transport_core")
    expected_guidance = guidance_contract(spec)
    if (
        not isinstance(core, Mapping)
        or core.get("guided_velocity_query_count")
        != spec.expected_guided_queries
        or core.get("raw_transformer_forward_count") != spec.expected_raw_forwards
        or core.get("noise_bank_digest_verified") is not True
        or core.get("raw_transformer_forward_count_verified") is not False
        or core.get("native_request_execution_verified") is not False
        or core.get("model_checkpoint_use_verified") is not False
        or core.get("optimizer_step_allowed") is not False
        or core.get("training_update_allowed") is not False
        or core.get("semantic_action_success") is not False
        or core.get("field_regime") != spec.field_regime
        or core.get("visual_condition_scope")
        != (
            "source_frame0_only_no_future_motion"
            if spec.uses_reference_frame0
            else "none"
        )
        or core.get("guidance_mode") != spec.guidance_mode
        or core.get("guidance_contract_sha256")
        != legacy.object_sha256(expected_guidance)
        or core.get("guidance_scale") != 4.0
        or core.get("image_guidance_scale")
        != expected_guidance["image_guidance_scale"]
        or core.get("guidance_chain_scales")
        != tuple(expected_guidance["guidance_chain_scales"])
        or core.get("apg_norm_thresholds")
        != tuple(expected_guidance["apg_norm_thresholds"])
        or core.get("apg_eta") != 0.5
        or core.get("apg_norm_threshold") != 50.0
        or core.get("apg_momentum") != 0.0
        or core.get("apg_momenta")
        != tuple(expected_guidance["apg_momenta"])
        or core.get("branch_order")
        != tuple(expected_guidance["branch_order"])
        or _SHA256.fullmatch(
            str(reference.get("transport_core_diagnostics_sha256", ""))
        )
        is None
    ):
        raise SAICInferenceError("transport core accounting facts differ")
    return {
        "all_rank_exact": True,
        "world_size": ULYSSES_SIZE,
        "ulysses_size": ULYSSES_SIZE,
        "certificate": reference,
        "per_rank": ordered,
        "per_rank_index_sha256": legacy.object_sha256(ordered),
    }


def build_model_receipt(
    *,
    checkpoint_identity: Mapping[str, Any],
    checkpoint_tree_sha256: str,
    bernini_revision: str,
    veomni_revision: str,
    bernini_inference_files: Mapping[str, str],
    runtime_source_index_sha256: str,
    renderer_config: Mapping[str, Any],
    freeze_certificate: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Build the external model/checkpoint receipt consumed by provenance."""

    payload = {
        "schema": f"{SCHEMA_VERSION}/frozen-model-receipt-v1",
        "model_id": MODEL_ID,
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "checkpoint_content_manifest_audit": dict(checkpoint_identity),
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
        "bernini_inference_files": dict(bernini_inference_files),
        "runtime_source_index_sha256": runtime_source_index_sha256,
        "renderer_config": dict(renderer_config),
        "freeze_certificate": dict(freeze_certificate),
        "adapter_or_lora_loaded": False,
        "optimizer_constructed": False,
        "training_mode_entered": False,
        "checkpoint_provenance_established_externally": True,
        "native_adapter_model_checkpoint_use_verified": False,
    }
    return payload, legacy.object_sha256(payload)


def build_reference_encoder_receipt(
    *,
    spec: ArmSpec,
    model_receipt_sha256: str,
    checkpoint_tree_sha256: str,
    vae_z_dim: int,
    bernini_pipeline_sha256: str,
) -> tuple[dict[str, Any], str]:
    if not spec.uses_reference_frame0:
        return {
            "used": False,
            "reference_frame0": False,
            "shared_sealed_frame0_coordinate_admitted": True,
            "passed_to_native_field": False,
            "encoder_sha256": ZERO_SHA256,
        }, ZERO_SHA256
    payload = {
        "schema": f"{SCHEMA_VERSION}/reference-encoder-v1",
        "operation": "read_presealed_independently_encoded_source_rgb_frame0",
        "source_rgb_indices": [0],
        "temporal_video_latent_slice_used": False,
        "encoded_in_runner": False,
        "passed_to_native_field": True,
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "frozen_model_receipt_sha256": model_receipt_sha256,
        "vae_class": "diffusers.models.AutoencoderKLWan",
        "vae_z_dim": int(vae_z_dim),
        "vae_runtime_dtype": "torch.float32",
        "bernini_pipeline_sha256": bernini_pipeline_sha256,
        "reference_source_id": 1,
    }
    return payload, legacy.object_sha256(payload)


def build_receipt(
    *,
    args: argparse.Namespace,
    spec: ArmSpec,
    sealed_cell: Mapping[str, Any],
    sealed_assets: Mapping[str, Any],
    source_path: Path,
    source_metadata: Mapping[str, Any],
    prompts: Mapping[str, str],
    prompt_identities: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    model_receipt: Mapping[str, Any],
    model_receipt_sha256: str,
    method_provenance: Mapping[str, Any],
    bernini_revision: str,
    veomni_revision: str,
    bernini_inference_files: Mapping[str, str],
    schedule: NativeScheduleBundle,
    guidance_contract_value: Mapping[str, Any],
    noise_bank_sha256: str,
    candidate_zero_sha256: str,
    sealed_source_coordinate: Mapping[str, Any],
    sealed_frame0_coordinate: Mapping[str, Any],
    source_latent_identity: Mapping[str, Any],
    reference_identity: Optional[Mapping[str, Any]],
    reference_encoder_receipt: Mapping[str, Any],
    reference_encoder_sha256: str,
    runtime: Mapping[str, Any],
    runtime_versions: Mapping[str, str],
    output_identity: Mapping[str, Any],
    normalized_clean_latent_identity: Mapping[str, Any],
    transaction_token: str,
) -> dict[str, Any]:
    """Create a non-authoritative, canonical execution receipt."""

    certificate = runtime.get("certificate")
    output_artifact_path = output_identity.get("path")
    expected_clean_latent_path = (
        f"{output_artifact_path}.normalized-clean-latent.safetensors"
        if type(output_artifact_path) is str
        else None
    )
    if (
        not isinstance(certificate, Mapping)
        or normalized_clean_latent_identity.get("path")
        != expected_clean_latent_path
        or _SHA256.fullmatch(
            str(normalized_clean_latent_identity.get("sha256", ""))
        )
        is None
        or normalized_clean_latent_identity.get("tensor_raw_sha256")
        != certificate.get("generated_latent_raw_sha256")
        or normalized_clean_latent_identity.get(
            "transport_endpoint_before_vae_decode"
        )
        is not True
        or normalized_clean_latent_identity.get("ground_truth") != "false"
        or normalized_clean_latent_identity.get("selected_for_training")
        != "false"
    ):
        raise SAICInferenceError(
            "normalized clean latent is not bound to the all-rank transport endpoint"
        )
    certificate_source_coordinate = certificate.get("sealed_source_coordinate")
    if (
        not isinstance(sealed_source_coordinate, Mapping)
        or not isinstance(certificate_source_coordinate, Mapping)
        or dict(sealed_source_coordinate) != dict(certificate_source_coordinate)
        or sealed_source_coordinate.get("loaded_from_sealed_source_coordinate")
        is not True
        or sealed_source_coordinate.get("encoded_in_runner") is not False
        or sealed_source_coordinate.get("tensor_raw_sha256")
        != certificate.get("source_latent_raw_sha256")
    ):
        raise SAICInferenceError(
            "sealed source coordinate is not bound to the all-rank source state"
        )
    certificate_frame0 = certificate.get("sealed_frame0_coordinate")
    if (
        not isinstance(sealed_frame0_coordinate, Mapping)
        or not isinstance(certificate_frame0, Mapping)
        or dict(sealed_frame0_coordinate) != dict(certificate_frame0)
        or sealed_frame0_coordinate.get("encoded_in_runner") is not False
        or sealed_frame0_coordinate.get("tensor_raw_sha256")
        != certificate.get("shared_frame0_raw_sha256")
    ):
        raise SAICInferenceError(
            "sealed frame0 coordinate is not bound to the all-rank input"
        )

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "arm": asdict(spec),
        "sealed_inputs": {
            **dict(sealed_assets),
            "selected_cell": dict(sealed_cell),
            "source_video_resolved_path": str(source_path),
            "source_video_sha256": sealed_cell["source_video_sha256"],
            "accepted_external_semantic_inputs": [
                "sealed_source_manifest",
                "sealed_event_bank",
                "row_id",
                "branch_forward",
                "rollout_seed",
            ],
            "accepted_external_derived_inputs": [
                "job132387_sealed_source_clean_coordinate",
                "fresh_sealed_source_rgb_frame0_coordinate",
            ],
            "free_form_source_caption_cli": False,
            "free_form_target_caption_cli": False,
            "target_video": False,
            "target_or_oracle_frame": False,
            "mask_or_swept_tube": False,
            "pose_flow_track_or_trajectory": False,
            "motion_donor": False,
            "external_reference": False,
        },
        "preprocessing": dict(source_metadata),
        "prompt_contract": {
            "task_name": spec.task_name,
            "task_system_prompt": _TASK_SYSTEM_PROMPTS[spec.task_name],
            "task_system_prompt_utf8_sha256": sha256_utf8(
                _TASK_SYSTEM_PROMPTS[spec.task_name]
            ),
            "source_body": sealed_cell["source_caption_body"],
            "source_body_utf8_sha256": sealed_cell[
                "source_caption_body_utf8_sha256"
            ],
            "target_body": sealed_cell["target_caption_body"],
            "target_body_utf8_sha256": sealed_cell[
                "target_caption_body_utf8_sha256"
            ],
            "source_full_prompt_utf8_sha256": sha256_utf8(prompts["source"]),
            "target_full_prompt_utf8_sha256": sha256_utf8(prompts["target"]),
            "negative_prompt_utf8_sha256": sha256_utf8(prompts["negative"]),
            "full_prompt_embeddings": dict(prompt_identities),
            "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "tokenizer_max_sequence_length": 512,
            "source_target_distinct": prompts["source"] != prompts["target"],
        },
        "model": {
            "model_id": MODEL_ID,
            "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
            "checkpoint_content": dict(checkpoint_identity),
            "frozen_model_receipt": dict(model_receipt),
            "frozen_model_receipt_sha256": model_receipt_sha256,
            "bernini_revision": bernini_revision,
            "veomni_revision": veomni_revision,
            "bernini_inference_files": dict(bernini_inference_files),
            "method_provenance": dict(method_provenance),
            "native_adapter_model_checkpoint_use_verified": False,
            "checkpoint_provenance_established_by_external_manifest_audit": True,
            "one_arm_per_process_group": True,
            "model_once_multi_arm_supported": False,
            "unsupported_model_once": True,
        },
        "transport": {
            "sealed_source_coordinate": dict(sealed_source_coordinate),
            "sealed_frame0_coordinate": dict(sealed_frame0_coordinate),
            "source_outer_clean_state": dict(source_latent_identity),
            "source_outer_clean_state_loaded_from_sealed_coordinate": True,
            "complete_source_video_vae_encoded_in_runner": False,
            "full_source_video_field_tokens": False,
            "reference_frame0": (
                None if reference_identity is None else dict(reference_identity)
            ),
            "reference_encoder": dict(reference_encoder_receipt),
            "reference_encoder_sha256": reference_encoder_sha256,
            "reference_is_source_rgb_frame0": True,
            "reference_is_available_at_inference": True,
            "reference_independently_vae_encoded": True,
            "reference_passed_to_native_field": spec.uses_reference_frame0,
            "reference_visual_i0_enabled_all_exact40_cells": (
                spec.uses_reference_frame0
            ),
            "reference_visual_i0_disabled_all_exact40_cells": (
                not spec.uses_reference_frame0
            ),
            "reference_from_temporal_latent_slice": False,
            "guidance_contract": dict(guidance_contract_value),
            "guidance_contract_sha256": legacy.object_sha256(
                guidance_contract_value
            ),
        },
        "schedule": {
            "num_frames": FRAME_COUNT,
            "latent_frames": LATENT_FRAME_COUNT,
            "fps": FPS,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "flow_shift": FLOW_SHIFT,
            "sigma_schedule": list(schedule.sigma_schedule),
            "core_sigma_schedule_sha256": schedule.core_sigma_schedule_sha256,
            "native_schedule_sha256": schedule.native_schedule_sha256,
            "pinned_unipc_schedule_sha256": schedule.pinned_schedule_sha256,
            "scheduler_sigma_fp32_sha256": schedule.scheduler_sigma_fp32_sha256,
            "sigma_scalar_direct_views": schedule.scalar_views_share_scheduler_storage,
            "timestep_direct_views": schedule.timestep_views_share_runtime_storage,
            "time_parameterization": "flow_time_equals_sigma",
        },
        "noise": {
            "generator_id": NOISE_GENERATOR_ID,
            "master_seed": sealed_cell["rollout_seed"],
            "actual_ordered_noise_bank_sha256": noise_bank_sha256,
            "actual_candidate_zero_subbank_sha256": candidate_zero_sha256,
            "candidate_schedule": list(spec.candidate_schedule),
            "candidate_continuation": "candidate_zero",
            "actual_tensor_bytes_digest_bound": True,
            "distribution_or_gaussianity_verified": False,
        },
        "distributed_execution": dict(runtime),
        "output": {
            **dict(output_identity),
            "normalized_clean_latent": dict(
                normalized_clean_latent_identity
            ),
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "height": source_metadata["source_derived_bucket_hw"][0],
            "width": source_metadata["source_derived_bucket_hw"][1],
            "rank0_decode_and_publish_only": True,
            "rank0_bundle_reopen_recorded_in_this_receipt": True,
            "all_rank_output_bundle_reopen_recorded_in_this_receipt": False,
            "all_rank_output_bundle_reopen_required_for_process_success": True,
            "transaction_token": transaction_token,
            "published_video_mode": "0444",
            "published_normalized_clean_latent_mode": "0444",
            "published_receipt_mode": "0444",
            "audio_preserved": False,
            "normalized_clean_latent_is_ground_truth": False,
            "normalized_clean_latent_selected_for_training": False,
        },
        "runtime_versions": dict(runtime_versions),
        "authority": {
            "frozen_inference_execution_receipt": True,
            "quality_authority": False,
            "evaluator_authority": False,
            "semantic_action_success": False,
            "identity_preservation_success": False,
            "training_authority": False,
            "optimizer_authority": False,
            "training_update_allowed": False,
            "optimizer_step_allowed": False,
            "checkpoint_or_lora_artifact": False,
            "production_claim_authorized": False,
        },
    }
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def publish_normalized_clean_latent_owned(
    latent: Any,
    path: Path,
    *,
    transaction_token: str,
) -> dict[str, Any]:
    """Create-only publish the exact FP32 pre-decode transport endpoint.

    The safetensors payload is written to a transaction-owned temporary inode,
    reopened and compared tensor-for-tensor, then hard-linked into the final
    name.  ``os.link`` supplies the no-overwrite commit point.  The artifact is
    a frozen inference endpoint only: it is neither ground truth nor a selected
    training target.
    """

    import torch
    from safetensors import safe_open
    from safetensors.torch import save as save_safetensors

    if (
        type(transaction_token) is not str
        or not transaction_token
        or _SAFE_BASENAME.fullmatch(transaction_token) is None
    ):
        raise SAICInferenceError("clean-latent transaction token is unsafe")
    if path.exists() or path.is_symlink() or path.suffix != ".safetensors":
        raise SAICInferenceError(
            "normalized clean latent path must be a fresh safetensors file"
        )
    if (
        type(latent) is not torch.Tensor
        or latent.requires_grad
        or latent.grad_fn is not None
        or latent.layout != torch.strided
        or not latent.is_floating_point()
        or latent.ndim != 5
        or tuple(int(item) for item in latent.shape[:3])
        != (1, 16, LATENT_FRAME_COUNT)
    ):
        raise SAICInferenceError(
            "transport endpoint must be detached [1,16,21,H,W] floating data"
        )
    stored = latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(stored).all().item()):
        raise SAICInferenceError("transport endpoint is non-finite")
    metadata = {
        "coordinate": "bernini_normalized_clean_vae_latent",
        "frame_contract": "exact81_latent21",
        "artifact_role": "source_state_flow_transport_endpoint",
        "source": "exact40_transport_edit_clean_before_vae_decode",
        "ground_truth": "false",
        "selected_for_training": "false",
    }
    payload = save_safetensors(
        {"normalized_clean_latent": stored}, metadata=metadata
    )
    if type(payload) is not bytes or not payload:
        raise SAICInferenceError("safetensors serializer returned invalid bytes")
    temporary = path.with_name(
        f".{path.name}.ssft-latent-tmp-{transaction_token}"
    )
    if temporary.exists() or temporary.is_symlink():
        raise SAICInferenceError("stale normalized clean latent temporary exists")
    descriptor: Optional[int] = None
    temporary_identity: Optional[dict[str, Any]] = None
    linked_identity: Optional[dict[str, Any]] = None
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_identity = guided_runner.artifact_identity(temporary)
        try:
            with safe_open(str(temporary), framework="pt", device="cpu") as opened:
                if list(opened.keys()) != ["normalized_clean_latent"]:
                    raise SAICInferenceError(
                        "transport endpoint safetensors key differs"
                    )
                restored = opened.get_tensor(
                    "normalized_clean_latent"
                ).contiguous()
                reopened_metadata = dict(opened.metadata() or {})
        except (OSError, RuntimeError) as error:
            raise SAICInferenceError(
                "cannot reopen normalized clean latent"
            ) from error
        if (
            reopened_metadata != metadata
            or restored.dtype != torch.float32
            or tuple(restored.shape) != tuple(stored.shape)
            or not torch.equal(restored, stored)
        ):
            raise SAICInferenceError(
                "normalized clean latent safetensors round trip differs"
            )
        tensor_sha256 = tensor_raw_sha256(restored)
        if tensor_sha256 != tensor_raw_sha256(stored):
            raise SAICInferenceError(
                "normalized clean latent tensor digest changed on reopen"
            )
        os.link(temporary, path)
        linked_identity = {**temporary_identity, "path": str(path)}
        observed = guided_runner.artifact_identity(path)
        if observed != linked_identity:
            raise SAICInferenceError(
                "published normalized clean latent identity differs"
            )
        guided_runner._fsync_directory(path.parent)
        return {
            **observed,
            "tensor_key": "normalized_clean_latent",
            "tensor_raw_sha256": tensor_sha256,
            "shape": [int(item) for item in stored.shape],
            "stored_dtype": str(stored.dtype),
            "transport_return_dtype": str(latent.dtype),
            **metadata,
            "transport_endpoint_before_vae_decode": True,
            "mp4_decode_reencode_used": False,
            "roundtrip_byte_exact_fp32": True,
            "quality_or_selection_authority": False,
        }
    except BaseException:
        guided_runner.unlink_owned_artifact(path, linked_identity)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        guided_runner.unlink_owned_artifact(temporary, temporary_identity)


def seal_published_bundle_read_only(
    output_path: Path,
    clean_latent_path: Optional[Path],
    receipt_path: Path,
    *,
    expected_video_identity: Mapping[str, Any],
    expected_clean_latent_identity: Optional[Mapping[str, Any]],
    expected_receipt_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the complete output bundle, reopen receipt, and verify digest."""

    artifacts: list[tuple[Path, Mapping[str, Any], str]] = [
        (output_path, expected_video_identity, "video")
    ]
    if clean_latent_path is not None:
        if expected_clean_latent_identity is None:
            raise SAICInferenceError("clean-latent seal identity is missing")
        artifacts.append(
            (
                clean_latent_path,
                expected_clean_latent_identity,
                "normalized clean latent",
            )
        )
    elif expected_clean_latent_identity is not None:
        raise SAICInferenceError("unexpected clean-latent seal identity")
    artifacts.append((receipt_path, expected_receipt_identity, "receipt"))
    for path, expected, label in artifacts:
        if path.is_symlink() or not path.is_file():
            raise SAICInferenceError(f"published {label} is not a plain file")
        observed = guided_runner.artifact_identity(path)
        identity_keys = ("path", "device", "inode", "size", "sha256")
        if any(observed.get(key) != expected.get(key) for key in identity_keys):
            raise SAICInferenceError(f"published {label} identity changed before seal")
        os.chmod(path, 0o444)
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise SAICInferenceError(f"published {label} is not mode 0444")
    try:
        reopened = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SAICInferenceError("cannot reopen published receipt") from error
    if type(reopened) is not dict:
        raise SAICInferenceError("reopened receipt root differs")
    stored = reopened.pop("receipt_digest", None)
    if _SHA256.fullmatch(str(stored)) is None or legacy.object_sha256(reopened) != stored:
        raise SAICInferenceError("reopened receipt digest differs")
    guided_runner._fsync_directory(output_path.parent)
    return {
        "video_mode": "0444",
        "clean_latent_mode": (
            "0444" if clean_latent_path is not None else None
        ),
        "receipt_mode": "0444",
        "receipt_reopened": True,
        "receipt_digest_verified": True,
    }


def seal_published_pair_read_only(
    output_path: Path,
    receipt_path: Path,
    *,
    expected_video_identity: Mapping[str, Any],
    expected_receipt_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper for tests/callers with no latent artifact."""

    return seal_published_bundle_read_only(
        output_path,
        None,
        receipt_path,
        expected_video_identity=expected_video_identity,
        expected_clean_latent_identity=None,
        expected_receipt_identity=expected_receipt_identity,
    )


def reopen_published_bundle_read_only(
    output_path: Path,
    clean_latent_path: Path,
    receipt_path: Path,
    *,
    expected_video_identity: Mapping[str, Any],
    expected_clean_latent_identity: Mapping[str, Any],
    expected_receipt_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Read-only terminal verification executed independently on every rank."""

    import torch
    from safetensors import safe_open

    observed_by_label: dict[str, dict[str, Any]] = {}
    for path, expected, label in (
        (output_path, expected_video_identity, "video"),
        (
            clean_latent_path,
            expected_clean_latent_identity,
            "normalized_clean_latent",
        ),
        (receipt_path, expected_receipt_identity, "receipt"),
    ):
        if path.is_symlink() or not path.is_file():
            raise SAICInferenceError(f"terminal published {label} is not plain")
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise SAICInferenceError(f"terminal published {label} mode differs")
        observed = guided_runner.artifact_identity(path)
        if any(
            observed.get(key) != expected.get(key)
            for key in ("path", "device", "inode", "size", "sha256")
        ):
            raise SAICInferenceError(
                f"terminal published {label} identity differs"
            )
        observed_by_label[label] = observed

    try:
        with safe_open(
            str(clean_latent_path), framework="pt", device="cpu"
        ) as opened:
            if list(opened.keys()) != ["normalized_clean_latent"]:
                raise SAICInferenceError(
                    "terminal normalized clean latent key differs"
                )
            clean_tensor = opened.get_tensor(
                "normalized_clean_latent"
            ).contiguous()
    except SAICInferenceError:
        raise
    except Exception as error:
        raise SAICInferenceError(
            "cannot terminal-reopen normalized clean latent"
        ) from error
    if (
        clean_tensor.dtype != torch.float32
        or tensor_raw_sha256(clean_tensor)
        != expected_clean_latent_identity.get("tensor_raw_sha256")
    ):
        raise SAICInferenceError(
            "terminal normalized clean latent tensor identity differs"
        )
    try:
        raw_receipt = receipt_path.read_bytes()
        reopened = json.loads(raw_receipt.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SAICInferenceError("cannot terminal-reopen receipt") from error
    if type(reopened) is not dict:
        raise SAICInferenceError("terminal receipt root differs")
    if raw_receipt != legacy.canonical_json_bytes(reopened) + b"\n":
        raise SAICInferenceError("terminal receipt bytes are not canonical JSON")
    unsigned = dict(reopened)
    declared = unsigned.pop("receipt_digest", None)
    if (
        type(declared) is not str
        or _SHA256.fullmatch(declared) is None
        or legacy.object_sha256(unsigned) != declared
        or reopened.get("output", {})
        .get("normalized_clean_latent", {})
        .get("tensor_raw_sha256")
        != expected_clean_latent_identity.get("tensor_raw_sha256")
    ):
        raise SAICInferenceError("terminal receipt content binding differs")
    return {
        "video_sha256": observed_by_label["video"]["sha256"],
        "normalized_clean_latent_file_sha256": observed_by_label[
            "normalized_clean_latent"
        ]["sha256"],
        "normalized_clean_latent_tensor_raw_sha256": (
            expected_clean_latent_identity["tensor_raw_sha256"]
        ),
        "receipt_file_sha256": observed_by_label["receipt"]["sha256"],
        "receipt_content_sha256": declared,
        "all_three_mode_0444": True,
        "receipt_canonical_and_digest_verified": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    spec = validate_cli(args)
    output_path, receipt_path = resolve_output(args.output)
    clean_latent_path = resolve_normalized_clean_latent_output(output_path)
    method_pre = validate_method_provenance(args)
    sealed_cell, sealed_assets = load_sealed_caption_cell(
        source_manifest_path=args.source_manifest,
        source_manifest_raw_sha256=args.expected_source_manifest_sha256,
        event_bank_path=args.event_bank,
        event_bank_raw_sha256=args.expected_event_bank_sha256,
        row_id=args.row_id,
        branch=args.branch,
        rollout_seed=args.rollout_seed,
        # The immutable manifest contains 8 rows, while this runner consumes
        # exactly one preregistered cell.  Re-probing every unrelated video on
        # every rank would widen the inference input closure and introduces an
        # unavailable compute-node ffprobe dependency.  The selected source is
        # instead byte-hashed below, decoded from a verified snapshot, checked
        # as exact81/25 fps, and rehashed again immediately before publication.
        verify_bound_source_files=False,
    )
    source_path = _plain_absolute_file(
        sealed_cell["source_video"], label="sealed selected source video"
    )
    manifest_path = Path(args.checkpoint_content_manifest).expanduser()
    if not manifest_path.is_absolute():
        raise SAICInferenceError("checkpoint-content-manifest must be absolute")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
        bernini_inference_files = legacy.validate_inference_source_files(
            bernini_root
        )
    except (legacy.trainer.TrainingContractError, legacy.InferenceContractError) as error:
        raise SAICInferenceError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise SAICInferenceError("Bernini-R 1.3B heads do not divide Ulysses=4")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from safetensors import __version__ as safetensors_version
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode
    from bernini.training.data import SYSTEM_PROMPTS

    for task_name, expected in _TASK_SYSTEM_PROMPTS.items():
        if SYSTEM_PROMPTS.get(task_name) != expected:
            raise SAICInferenceError(
                f"runtime Bernini {task_name} system prompt differs"
            )
    if DEFAULT_NEG_PROMPT != legacy.DEFAULT_NEGATIVE_PROMPT:
        raise SAICInferenceError("runtime Bernini negative prompt differs")

    try:
        distributed = legacy.inference_distributed_contract()
    except legacy.InferenceContractError as error:
        raise SAICInferenceError(str(error)) from error
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise SAICInferenceError("SSFT inference requires four AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=distributed.ulysses_size)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, manifest_path
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_rows, src=0)
    checkpoint_result = checkpoint_rows[0]
    if (
        not isinstance(checkpoint_result, Mapping)
        or checkpoint_result.get("ok") is not True
    ):
        raise SAICInferenceError(
            f"rank-zero checkpoint content audit failed: {checkpoint_result}"
        )
    checkpoint_identity = dict(checkpoint_result["identity"])

    try:
        source_tensor, source_metadata, source_sha256 = (
            source_audit.prepare_hashed_source_snapshot(source_path)
        )
    except source_audit.SourceKVCarrierOracleError as error:
        raise SAICInferenceError(str(error)) from error
    if source_sha256 != sealed_cell["source_video_sha256"]:
        raise SAICInferenceError("selected source bytes differ from sealed row")
    if (
        source_metadata.get("frame_count") != FRAME_COUNT
        or source_metadata.get("fps") != FPS
    ):
        raise SAICInferenceError("selected source is not exact81 at 25 fps")
    bucket_hw = tuple(
        int(item) for item in source_metadata["source_derived_bucket_hw"]
    )
    expected_source_shape = (
        1,
        16,
        LATENT_FRAME_COUNT,
        bucket_hw[0] // 8,
        bucket_hw[1] // 8,
    )
    sealed_source_coordinate = load_sealed_source_coordinate(
        args,
        sealed_cell=sealed_cell,
        sealed_assets=sealed_assets,
        source_path=source_path,
        source_tensor=source_tensor,
        source_metadata=source_metadata,
        checkpoint_path=checkpoint,
        checkpoint_identity=checkpoint_identity,
        bernini_revision=bernini_revision,
        veomni_revision=veomni_revision,
        bernini_inference_files=bernini_inference_files,
        method_provenance=method_pre,
        expected_shape=expected_source_shape,
    )
    expected_reference_shape = (
        1,
        expected_source_shape[1],
        1,
        bucket_hw[0] // 8,
        bucket_hw[1] // 8,
    )
    sealed_frame0_coordinate = load_sealed_frame0_coordinate(
        args,
        sealed_cell=sealed_cell,
        sealed_assets=sealed_assets,
        method_provenance=method_pre,
        expected_shape=expected_reference_shape,
    )

    source_prompt = build_task_prompt(
        spec.task_name,
        sealed_cell["source_caption_body"],
        prompt_cleaner=prompt_clean,
    )
    target_prompt = build_task_prompt(
        spec.task_name,
        sealed_cell["target_caption_body"],
        prompt_cleaner=prompt_clean,
    )
    if source_prompt == target_prompt:
        raise SAICInferenceError("active forward source/target prompts are identical")
    prompts = {
        "source": source_prompt,
        "target": target_prompt,
        "negative": DEFAULT_NEG_PROMPT,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    if (
        tokenizer.padding_side != "right"
        or tokenizer.init_kwargs.get("fix_mistral_regex") is not True
    ):
        raise SAICInferenceError("tokenizer contract differs")
    source_ids, source_mask = legacy._tokenize_training_prompt(
        tokenizer, source_prompt
    )
    target_ids, target_mask = legacy._tokenize_training_prompt(
        tokenizer, target_prompt
    )
    negative_ids, negative_mask = legacy._tokenize_renderer_negative(
        tokenizer, DEFAULT_NEG_PROMPT
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        legacy.trainer.validate_renderer_config_mapping(
            config.to_dict(), checkpoint
        )
    except legacy.trainer.TrainingContractError as error:
        raise SAICInferenceError(str(error)) from error
    if float(config.shift) != FLOW_SHIFT or config.use_unipc is not True:
        raise SAICInferenceError("renderer is not pinned UniPC shift 5")
    model = BerniniRendererModel(config)
    model.requires_grad_(False)
    model.eval()
    try:
        freeze_before = source_audit.model_freeze_certificate(model)
    except source_audit.SourceKVCarrierOracleError as error:
        raise SAICInferenceError(str(error)) from error

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    if int(vae.config.z_dim) != expected_source_shape[1]:
        raise SAICInferenceError("sealed source-clean VAE channel geometry differs")
    source_clean = sealed_source_coordinate.tensor.to(
        device=device, dtype=torch.float32
    ).contiguous()
    if (
        tensor_raw_sha256(source_clean)
        != sealed_source_coordinate.provenance["tensor_raw_sha256"]
    ):
        raise SAICInferenceError("source-clean CPU-to-GPU bytes differ")
    shared_frame0 = sealed_frame0_coordinate.tensor.to(
        device=device, dtype=torch.float32
    ).contiguous()
    if (
        tensor_raw_sha256(shared_frame0)
        != sealed_frame0_coordinate.provenance["tensor_raw_sha256"]
    ):
        raise SAICInferenceError("frame0 CPU-to-GPU bytes differ")
    reference_frame0 = shared_frame0 if spec.uses_reference_frame0 else None
    if (
        tuple(source_clean.shape) != expected_source_shape
        or source_clean.dtype != torch.float32
    ):
        raise SAICInferenceError("complete source clean latent geometry differs")
    if (
        tuple(shared_frame0.shape) != expected_reference_shape
        or shared_frame0.dtype != torch.float32
    ):
        raise SAICInferenceError("sealed source RGB frame-0 latent differs")

    source_broadcast = native_canary._broadcast_condition_from_rank_zero(
        source_clean,
        label="ssft_source_clean",
        world_size=distributed.world_size,
    )
    reference_broadcast = native_canary._broadcast_condition_from_rank_zero(
        shared_frame0,
        label="ssft_shared_source_rgb_frame0_coordinate",
        world_size=distributed.world_size,
    )
    source_latent_identity = native_canary._all_rank_tensor_identity(
        source_clean,
        label="ssft_source_clean",
        world_size=distributed.world_size,
    )
    if (
        source_latent_identity.get("all_rank_exact") is not True
        or source_latent_identity.get("identity", {}).get(
            "raw_storage_sha256"
        )
        != sealed_source_coordinate.provenance["tensor_raw_sha256"]
    ):
        raise SAICInferenceError(
            "sealed source-clean bytes differ after rank-zero broadcast"
        )
    reference_identity = native_canary._all_rank_tensor_identity(
        shared_frame0,
        label="ssft_shared_source_rgb_frame0_coordinate",
        world_size=distributed.world_size,
    )
    if (
        reference_identity.get("all_rank_exact") is not True
        or reference_identity.get("identity", {}).get("raw_storage_sha256")
        != sealed_frame0_coordinate.provenance["tensor_raw_sha256"]
    ):
        raise SAICInferenceError("sealed frame0 bytes differ after broadcast")
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    model.t5_text_encoder.to(device)
    with torch.inference_mode():
        source_condition = model.encode_prompt(
            source_ids.to(device), source_mask.to(device)
        ).contiguous()
        target_condition = model.encode_prompt(
            target_ids.to(device), target_mask.to(device)
        ).contiguous()
        negative_condition = model.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).contiguous()
    model.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()
    conditions = {
        "negative": negative_condition,
        "target": target_condition,
        "source": source_condition,
    }
    for label, condition in conditions.items():
        if (
            condition.dtype != torch.bfloat16
            or tuple(condition.shape) != (1, 512, 4096)
            or condition.requires_grad
        ):
            raise SAICInferenceError(f"{label} prompt condition closure differs")
        native_canary._broadcast_condition_from_rank_zero(
            condition,
            label=f"ssft_{label}_prompt_condition",
            world_size=distributed.world_size,
        )
    condition_raw_sha256 = {
        key: tensor_raw_sha256(value) for key, value in conditions.items()
    }
    if len(set(condition_raw_sha256.values())) != 3:
        raise SAICInferenceError("negative/source/target condition bytes are not distinct")
    prompt_identities = {
        key: native_canary._all_rank_tensor_identity(
            value,
            label=f"ssft_{key}_prompt_condition",
            world_size=distributed.world_size,
        )
        for key, value in conditions.items()
    }

    diffusion = cdf.resolve_diffusion_core(model)
    transformer = diffusion.transformer
    transformer.to(device)
    schedule = prepare_native_schedule(diffusion, device, spec=spec)
    renderer_receipt_config = {
        "config_class": "BerniniRendererConfig",
        "dtype": str(config.dtype),
        "shift": float(config.shift),
        "use_unipc": bool(config.use_unipc),
        "single_expert": MODEL_ID,
        "transformer_num_attention_heads": int(
            transformer_config["num_attention_heads"]
        ),
    }
    model_receipt, model_receipt_sha256 = build_model_receipt(
        checkpoint_identity=checkpoint_identity,
        checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
        bernini_revision=bernini_revision,
        veomni_revision=veomni_revision,
        bernini_inference_files=bernini_inference_files,
        runtime_source_index_sha256=method_pre["runtime_source_index_sha256"],
        renderer_config=renderer_receipt_config,
        freeze_certificate=freeze_before,
    )
    reference_encoder_receipt, reference_encoder_sha256 = (
        build_reference_encoder_receipt(
            spec=spec,
            model_receipt_sha256=model_receipt_sha256,
            checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
            vae_z_dim=int(vae.config.z_dim),
            bernini_pipeline_sha256=bernini_inference_files[
                "bernini/pipeline.py"
            ],
        )
    )

    noise_bank = build_fresh_noise_bank(
        shape=source_clean.shape,
        device=device,
        master_seed=args.rollout_seed,
        candidate_schedule=spec.candidate_schedule,
    )
    noise_bank_sha256 = transport.noise_bank_sha256(
        noise_bank, candidate_schedule=spec.candidate_schedule
    )
    candidate_zero_sha256 = candidate_zero_noise_sha256(noise_bank)
    guidance_value = guidance_contract(spec)
    guidance_sha256 = legacy.object_sha256(guidance_value)
    negative_prompt_sha256 = sha256_utf8(DEFAULT_NEG_PROMPT)
    native_program = {
        "image_guidance_scale": guidance_value["image_guidance_scale"],
        "guidance_chain_scales": tuple(
            guidance_value["guidance_chain_scales"]
        ),
        "apg_norm_thresholds": tuple(
            guidance_value["apg_norm_thresholds"]
        ),
        "apg_momenta": tuple(guidance_value["apg_momenta"]),
        "branch_order": tuple(guidance_value["branch_order"]),
        "raw_transformer_forwards_per_candidate": guidance_value[
            "per_candidate_raw_forwards"
        ],
    }
    native_binding = transport.NativeGuidanceBinding(
        model_id=MODEL_ID,
        checkpoint_sha256=args.expected_checkpoint_tree_sha256,
        negative_prompt_sha256=negative_prompt_sha256,
        field_regime=spec.field_regime,
        guidance_mode=spec.guidance_mode,
        guidance_contract_sha256=guidance_sha256,
        **native_program,
    ).validate()
    rollout_config = transport.FlowTransportRolloutConfig(
        native=native_binding,
        anc_enabled=spec.anc_enabled,
        noise_generator_id=NOISE_GENERATOR_ID,
        master_seed=args.rollout_seed,
        noise_bank_sha256=noise_bank_sha256,
        sigma_schedule=schedule.sigma_schedule,
        candidate_schedule=spec.candidate_schedule,
        candidate_continuation="candidate_zero",
        aggregation_mode=spec.aggregation_mode,
        temperature=spec.temperature,
        anchor_latent_phase_zero=spec.anchor_latent_phase_zero,
    ).validate()
    native_provenance = native_field.NativeFieldProvenance(
        model_id=MODEL_ID,
        checkpoint_sha256=args.expected_checkpoint_tree_sha256,
        model_receipt_sha256=model_receipt_sha256,
        guidance_contract_sha256=guidance_sha256,
        negative_prompt_sha256=negative_prompt_sha256,
        native_schedule_sha256=schedule.native_schedule_sha256,
        noise_generator_id=NOISE_GENERATOR_ID,
        master_seed=args.rollout_seed,
        noise_bank_sha256=noise_bank_sha256,
        reference_encoder_sha256=reference_encoder_sha256,
        reference_frame0_latent_sha256=(
            ZERO_SHA256
            if reference_frame0 is None
            else tensor_raw_sha256(reference_frame0)
        ),
        prompt_utf8_sha256_by_role={
            "target": sha256_utf8(target_prompt),
            "source": sha256_utf8(source_prompt),
        },
        prompt_condition_sha256_by_key=condition_raw_sha256,
    ).validate(regime=spec.field_regime)

    adapter_diagnostics: Any = None
    rollout: Any = None
    adapter = native_field.NativeSourceStateFieldAdapter(
        diffusion=diffusion,
        transformer=transformer,
        field_regime=spec.field_regime,
        conditions=conditions,
        captions={"target": target_prompt, "source": source_prompt},
        sigma_scalars=schedule.sigma_scalars,
        next_sigmas=schedule.next_sigmas,
        timestep_tensors=schedule.timestep_tensors,
        candidate_schedule=spec.candidate_schedule,
        aggregation_mode=spec.aggregation_mode,
        temperature=spec.temperature,
        provenance=native_provenance,
        reference_frame0_latent=reference_frame0,
    )
    source_coordinate_pre_rollout = revalidate_sealed_source_coordinate(
        sealed_source_coordinate, stage="pre_rollout"
    )
    frame0_coordinate_pre_rollout = revalidate_sealed_frame0_coordinate(
        sealed_frame0_coordinate, stage="pre_rollout"
    )
    if (
        tensor_raw_sha256(source_clean)
        != sealed_source_coordinate.provenance["tensor_raw_sha256"]
    ):
        raise SAICInferenceError(
            "broadcast source-clean bytes differ before rollout"
        )
    with torch.inference_mode():
        rollout = transport.run_exact40_source_state_flow_transport(
            config=rollout_config,
            source_clean=source_clean,
            source_caption=source_prompt,
            target_caption=target_prompt,
            sigma_schedule=schedule.sigma_schedule,
            fresh_noise_schedule=noise_bank,
            velocity_query=adapter,
        )
    adapter_diagnostics = adapter.finalize()
    if rollout is None or adapter_diagnostics is None:
        raise SAICInferenceError("SSFT rollout/finalization did not complete")
    generated_latent = rollout.edit_clean
    if (
        tuple(generated_latent.shape) != expected_source_shape
        or generated_latent.dtype != torch.float32
        or generated_latent is source_clean
        or not bool(torch.isfinite(generated_latent).all().item())
    ):
        raise SAICInferenceError("generated exact81 latent closure differs")
    core_diagnostics = asdict(rollout.diagnostics)
    core_summary = {
        key: core_diagnostics[key]
        for key in (
            "guided_velocity_query_count",
            "raw_transformer_forward_count",
            "field_regime",
            "visual_condition_scope",
            "guidance_mode",
            "guidance_contract_sha256",
            "guidance_scale",
            "image_guidance_scale",
            "guidance_chain_scales",
            "apg_eta",
            "apg_norm_threshold",
            "apg_norm_thresholds",
            "apg_momentum",
            "apg_momenta",
            "branch_order",
            "noise_bank_digest_verified",
            "raw_transformer_forward_count_verified",
            "native_request_execution_verified",
            "model_checkpoint_use_verified",
            "noise_distribution_verified",
            "optimizer_step_allowed",
            "training_update_allowed",
            "semantic_action_success",
        )
    }
    adapter_summary = asdict(adapter_diagnostics)
    if (
        core_summary["guided_velocity_query_count"]
        != adapter_summary["guided_query_success_count"]
        or core_summary["raw_transformer_forward_count"]
        != adapter_summary["raw_transformer_forward_success_count"]
    ):
        raise SAICInferenceError(
            "transport core and native adapter execution counts differ"
        )
    generated_identity = native_canary._all_rank_tensor_identity(
        generated_latent,
        label="ssft_generated_latent",
        world_size=distributed.world_size,
    )
    try:
        freeze_after = source_audit.model_freeze_certificate(model)
    except source_audit.SourceKVCarrierOracleError as error:
        raise SAICInferenceError(str(error)) from error
    if freeze_after != freeze_before:
        raise SAICInferenceError("frozen model certificate changed")
    method_post = validate_method_provenance(args)
    if method_post != method_pre:
        raise SAICInferenceError("method provenance changed during SSFT inference")
    source_coordinate_pre_publish = revalidate_sealed_source_coordinate(
        sealed_source_coordinate, stage="pre_publish"
    )
    frame0_coordinate_pre_publish = revalidate_sealed_frame0_coordinate(
        sealed_frame0_coordinate, stage="pre_publish"
    )
    if (
        tensor_raw_sha256(source_clean)
        != sealed_source_coordinate.provenance["tensor_raw_sha256"]
    ):
        raise SAICInferenceError(
            "source-clean bytes differ before artifact publication"
        )

    source_coordinate_certificate = {
        **dict(sealed_source_coordinate.provenance),
        "cpu_to_gpu_byte_exact": True,
        "rank0_broadcast_before_renderer": True,
        "all_rank_identity_after_broadcast": True,
        "pre_rollout_rehash": source_coordinate_pre_rollout,
        "pre_publish_rehash": source_coordinate_pre_publish,
        "terminal_rehash_recorded_in_this_receipt": False,
        "terminal_rehash_required_for_process_success": True,
    }
    frame0_coordinate_certificate = {
        **dict(sealed_frame0_coordinate.provenance),
        "cpu_to_gpu_byte_exact": True,
        "rank0_broadcast_before_renderer": True,
        "all_rank_identity_after_broadcast": True,
        "pre_rollout_rehash": frame0_coordinate_pre_rollout,
        "pre_publish_rehash": frame0_coordinate_pre_publish,
        "terminal_rehash_recorded_in_this_receipt": False,
        "terminal_rehash_required_for_process_success": True,
    }

    certificate = {
        "arm": spec.arm,
        "row_id": sealed_cell["row_id"],
        "candidate_id": sealed_cell["candidate_id"],
        "rollout_seed": args.rollout_seed,
        "source_video_sha256": source_sha256,
        "source_latent_raw_sha256": tensor_raw_sha256(source_clean),
        "sealed_source_coordinate": source_coordinate_certificate,
        "sealed_frame0_coordinate": frame0_coordinate_certificate,
        "loaded_from_sealed_source_coordinate": True,
        "source_clean_encoded_in_runner": False,
        "shared_frame0_raw_sha256": tensor_raw_sha256(shared_frame0),
        "native_field_reference_frame0_raw_sha256": (
            ZERO_SHA256
            if reference_frame0 is None
            else tensor_raw_sha256(reference_frame0)
        ),
        "visual_i0_enabled_all_exact40_cells": spec.uses_reference_frame0,
        "source_prompt_utf8_sha256": sha256_utf8(source_prompt),
        "target_prompt_utf8_sha256": sha256_utf8(target_prompt),
        "negative_prompt_utf8_sha256": negative_prompt_sha256,
        "condition_raw_sha256": condition_raw_sha256,
        "generated_latent_raw_sha256": tensor_raw_sha256(generated_latent),
        "noise_bank_sha256": noise_bank_sha256,
        "candidate_zero_noise_sha256": candidate_zero_sha256,
        "native_schedule_sha256": schedule.native_schedule_sha256,
        "core_sigma_schedule_sha256": schedule.core_sigma_schedule_sha256,
        "model_receipt_sha256": model_receipt_sha256,
        "runtime_source_index_sha256": method_pre[
            "runtime_source_index_sha256"
        ],
        "native_guided_query_attempt_count": adapter_summary[
            "guided_query_attempt_count"
        ],
        "native_guided_query_success_count": adapter_summary[
            "guided_query_success_count"
        ],
        "native_raw_transformer_forward_attempt_count": adapter_summary[
            "raw_transformer_forward_attempt_count"
        ],
        "native_raw_transformer_forward_success_count": adapter_summary[
            "raw_transformer_forward_success_count"
        ],
        "core_native_guided_count_reconciled": True,
        "core_native_raw_forward_count_reconciled": True,
        "model_freeze_unchanged": True,
        "source_rank0_broadcast": source_broadcast,
        "reference_rank0_broadcast": reference_broadcast,
        "native_adapter": adapter_summary,
        "transport_core": core_summary,
        "transport_core_diagnostics_sha256": legacy.object_sha256(
            core_diagnostics
        ),
    }
    local_row = {
        "rank": distributed.rank,
        "local_rank": distributed.local_rank,
        "world_size": distributed.world_size,
        "ulysses_size": distributed.ulysses_size,
        "certificate": certificate,
    }
    rank_rows: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(rank_rows, local_row)
    runtime = validate_all_rank_runtime(rank_rows, spec=spec)

    model.to("cpu")
    del adapter, noise_bank, rollout, source_clean, reference_frame0, shared_frame0
    del source_condition, target_condition, negative_condition, conditions
    torch.cuda.empty_cache()
    runtime_versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "transformers": transformers_version,
        "diffusers": diffusers_version,
        "safetensors": safetensors_version,
    }
    transaction_token = legacy_audit.output_transaction_token()
    video_owned: Optional[dict[str, Any]] = None
    clean_latent_owned: Optional[dict[str, Any]] = None
    receipt_owned: Optional[dict[str, Any]] = None
    rank0_published = False
    receipt_to_print: Optional[dict[str, Any]] = None
    publication_result: list[Any] = [None]
    if distributed.rank == 0:
        try:
            sealed_assets_publish = revalidate_terminal_sealed_input_bytes(
                sealed_assets=sealed_assets,
                sealed_cell=sealed_cell,
                selected_source_path=source_path,
            )
            clean_latent_owned = publish_normalized_clean_latent_owned(
                generated_latent,
                clean_latent_path,
                transaction_token=transaction_token,
            )
            vae.to(device)
            with torch.inference_mode():
                decoded = _vae_decode(vae, generated_latent)
            vae.to("cpu")
            if tuple(decoded.shape) != (
                FRAME_COUNT,
                bucket_hw[0],
                bucket_hw[1],
                3,
            ):
                raise SAICInferenceError("decoded exact81 video geometry differs")
            video_owned = guided_runner.publish_video_owned(
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
            if tuple(encoded_hw) != bucket_hw:
                raise SAICInferenceError("published output geometry differs")
            method_publish = validate_method_provenance(args)
            if method_publish != method_pre:
                raise SAICInferenceError(
                    "method provenance changed before artifact publication"
                )
            receipt = build_receipt(
                args=args,
                spec=spec,
                sealed_cell=sealed_cell,
                sealed_assets=sealed_assets_publish,
                source_path=source_path,
                source_metadata=source_metadata,
                prompts=prompts,
                prompt_identities=prompt_identities,
                checkpoint_identity=checkpoint_identity,
                model_receipt=model_receipt,
                model_receipt_sha256=model_receipt_sha256,
                method_provenance=method_publish,
                bernini_revision=bernini_revision,
                veomni_revision=veomni_revision,
                bernini_inference_files=bernini_inference_files,
                schedule=schedule,
                guidance_contract_value=guidance_value,
                noise_bank_sha256=noise_bank_sha256,
                candidate_zero_sha256=candidate_zero_sha256,
                sealed_source_coordinate=source_coordinate_certificate,
                sealed_frame0_coordinate=frame0_coordinate_certificate,
                source_latent_identity=source_latent_identity,
                reference_identity=reference_identity,
                reference_encoder_receipt=reference_encoder_receipt,
                reference_encoder_sha256=reference_encoder_sha256,
                runtime=runtime,
                runtime_versions=runtime_versions,
                output_identity=video_owned,
                normalized_clean_latent_identity=clean_latent_owned,
                transaction_token=transaction_token,
            )
            receipt_owned = guided_runner.publish_receipt_owned(
                receipt_path,
                receipt,
                transaction_token=transaction_token,
            )
            seal_published_bundle_read_only(
                output_path,
                clean_latent_path,
                receipt_path,
                expected_video_identity=video_owned,
                expected_clean_latent_identity=clean_latent_owned,
                expected_receipt_identity=receipt_owned,
            )
            receipt_to_print = receipt
            rank0_published = True
            publication_result[0] = {
                "ok": True,
                "video_identity": dict(video_owned),
                "clean_latent_identity": dict(clean_latent_owned),
                "receipt_identity": dict(receipt_owned),
            }
        except BaseException as error:
            guided_runner.unlink_owned_artifact(receipt_path, receipt_owned)
            guided_runner.unlink_owned_artifact(output_path, video_owned)
            guided_runner.unlink_owned_artifact(
                clean_latent_path, clean_latent_owned
            )
            publication_result[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }

    terminal_reopen_rows: list[Any] = [None] * ULYSSES_SIZE
    terminal_reopen_exact = False
    try:
        dist.broadcast_object_list(publication_result, src=0)
        try:
            published = publication_result[0]
            if not isinstance(published, Mapping) or published.get("ok") is not True:
                raise SAICInferenceError(
                    f"rank-zero publication did not complete: {published}"
                )
            video_identity = published.get("video_identity")
            clean_identity = published.get("clean_latent_identity")
            receipt_identity = published.get("receipt_identity")
            if not all(
                isinstance(value, Mapping)
                for value in (video_identity, clean_identity, receipt_identity)
            ):
                raise SAICInferenceError("published bundle identity maps differ")
            terminal_verification = reopen_published_bundle_read_only(
                output_path,
                clean_latent_path,
                receipt_path,
                expected_video_identity=video_identity,
                expected_clean_latent_identity=clean_identity,
                expected_receipt_identity=receipt_identity,
            )
            if (
                terminal_verification[
                    "normalized_clean_latent_tensor_raw_sha256"
                ]
                != certificate["generated_latent_raw_sha256"]
            ):
                raise SAICInferenceError(
                    "terminal clean-latent bytes differ from all-rank transport state"
                )
            terminal_source_coordinate = revalidate_sealed_source_coordinate(
                sealed_source_coordinate, stage="terminal"
            )
            terminal_frame0_coordinate = revalidate_sealed_frame0_coordinate(
                sealed_frame0_coordinate, stage="terminal"
            )
            if (
                terminal_source_coordinate["tensor_raw_sha256"]
                != certificate["source_latent_raw_sha256"]
            ):
                raise SAICInferenceError(
                    "terminal sealed source-coordinate bytes differ"
                )
            if (
                terminal_frame0_coordinate["tensor_raw_sha256"]
                != certificate["shared_frame0_raw_sha256"]
            ):
                raise SAICInferenceError(
                    "terminal sealed frame0-coordinate bytes differ"
                )
            terminal_local = {
                "ok": True,
                "verification": terminal_verification,
                "sealed_source_coordinate": terminal_source_coordinate,
                "sealed_frame0_coordinate": terminal_frame0_coordinate,
            }
        except BaseException as error:
            terminal_local = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        try:
            close_sealed_source_coordinate(sealed_source_coordinate)
            close_sealed_frame0_coordinate(sealed_frame0_coordinate)
        except BaseException as error:
            terminal_local = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        dist.all_gather_object(terminal_reopen_rows, terminal_local)
        terminal_reference = terminal_reopen_rows[0]
        terminal_reopen_exact = bool(
            isinstance(terminal_reference, Mapping)
            and terminal_reference.get("ok") is True
            and all(row == terminal_reference for row in terminal_reopen_rows[1:])
        )
        if distributed.rank == 0 and rank0_published and not terminal_reopen_exact:
            guided_runner.unlink_owned_artifact(receipt_path, receipt_owned)
            guided_runner.unlink_owned_artifact(output_path, video_owned)
            guided_runner.unlink_owned_artifact(
                clean_latent_path, clean_latent_owned
            )
            rank0_published = False
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:
        if distributed.rank == 0 and rank0_published:
            guided_runner.unlink_owned_artifact(receipt_path, receipt_owned)
            guided_runner.unlink_owned_artifact(output_path, video_owned)
            guided_runner.unlink_owned_artifact(
                clean_latent_path, clean_latent_owned
            )
        raise
    if (
        not isinstance(publication_result[0], Mapping)
        or publication_result[0].get("ok") is not True
    ):
        raise SAICInferenceError(
            f"rank-zero SSFT artifact publication failed: {publication_result[0]}"
        )
    if not terminal_reopen_exact:
        raise SAICInferenceError(
            "all-rank terminal bundle reopen differs: "
            f"{terminal_reopen_rows}"
        )
    if distributed.rank == 0:
        assert receipt_to_print is not None
        print(
            legacy.canonical_json_bytes(receipt_to_print).decode("utf-8"),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_NAMES",
    "ARM_SPECS",
    "ArmSpec",
    "EVENT_BANK_RAW_SHA256",
    "FRAME_COUNT",
    "LATENT_FRAME_COUNT",
    "METHOD",
    "NativeScheduleBundle",
    "SAICInferenceError",
    "SCHEMA_VERSION",
    "SOURCE_CLEAN_ACCEPTED_INPUT_ROLES",
    "SOURCE_CLEAN_ARTIFACT_METADATA",
    "SOURCE_CLEAN_ARTIFACT_SCHEMA",
    "SOURCE_CLEAN_FORBIDDEN_INPUT_ROLES",
    "SOURCE_CLEAN_MATERIALIZER_METHOD",
    "SOURCE_CLEAN_MATERIALIZER_ARCHIVE_MEMBERS",
    "SOURCE_CLEAN_MATERIALIZER_RUNTIME_FILES",
    "SOURCE_CLEAN_RECEIPT_SCHEMA",
    "SOURCE_CLEAN_TENSOR_KEY",
    "SOURCE_MANIFEST_RAW_SHA256",
    "FRAME0_ARTIFACT_METADATA",
    "FRAME0_ARTIFACT_SCHEMA",
    "FRAME0_RECEIPT_SCHEMA",
    "FRAME0_MATERIALIZER_METHOD",
    "FRAME0_TENSOR_KEY",
    "SealedFrame0Coordinate",
    "SealedSourceCoordinate",
    "arm_spec",
    "bind_native_schedule_objects",
    "build_fresh_noise_bank",
    "build_model_receipt",
    "build_parser",
    "build_receipt",
    "build_reference_encoder_receipt",
    "build_task_prompt",
    "candidate_zero_noise_sha256",
    "guidance_contract",
    "keyed_noise_seed",
    "load_sealed_source_coordinate",
    "load_sealed_frame0_coordinate",
    "load_sealed_caption_cell",
    "publish_normalized_clean_latent_owned",
    "revalidate_terminal_sealed_input_bytes",
    "revalidate_sealed_source_coordinate",
    "revalidate_sealed_frame0_coordinate",
    "reopen_published_bundle_read_only",
    "resolve_normalized_clean_latent_output",
    "resolve_sealed_forward_cell",
    "runtime_source_hashes",
    "runtime_vae_encoder_identity",
    "tensor_raw_sha256",
    "seal_published_bundle_read_only",
    "seal_published_pair_read_only",
    "validate_all_rank_runtime",
    "validate_cli",
    "close_sealed_source_coordinate",
    "close_sealed_frame0_coordinate",
]
