#!/usr/bin/env python3
"""Build and render the CDF-dog source-caption-specific pure-T2V Pass A bank.

This is an action-prior proposal bank, not an editor and not a training target.
It delegates every render to the pinned ``t2v`` arm of
``infer_native_identity_generation_canary.py``.  The CDF-dog MP4 is decoded by
that native runner only to verify its hash and choose the fixed 496x480 bucket;
no source pixel, source latent, reference, target, mask, flow, pose, track, or
trajectory is supplied to Bernini's sampler.

Two seeds are preregistered.  Each seed renders the complete four-way semantic
factorial with one byte-identical native Gaussian value: full action, no-op,
incomplete action, and reverse/negative action.  Completion of rendering does
not establish that the labelled events occurred.  Exact-40 output must be
qualified independently and the whole Pass A bank is rejected if either seed
fails the registered four-branch contract; selecting one seed or branch after
viewing is forbidden.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


MANIFEST_SCHEMA = "bernini-cdf-dog-source-caption-t2v-pass-a-manifest-v1"
BANK_RECEIPT_SCHEMA = "bernini-cdf-dog-source-caption-t2v-pass-a-receipt-v1"
NATIVE_RECEIPT_SCHEMA = "bernini-native-identity-generation-canary-v1"
METHOD = "cdf-dog-source-caption-specific-pure-t2v-pass-a"

FRAME_COUNT = 81
LATENT_FRAME_COUNT = 21
FPS = 25
VIDEO_HEIGHT = 496
VIDEO_WIDTH = 480
LATENT_SHAPE = (1, 16, LATENT_FRAME_COUNT, VIDEO_HEIGHT // 8, VIDEO_WIDTH // 8)
ALLOWED_STEPS = (1, 40)
EXACT_STEPS = 40
ULYSSES_SIZE = 4
BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)

# This experiment is deliberately tied to the exact CDF-dog source used in the
# Bernini comparison packet.  The bytes set the bucket and provenance only.
CDF_DOG_SOURCE_SHA256 = (
    "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
)

BRANCH_ORDER = ("full_action", "noop", "incomplete", "reverse")
SEED_ROWS = (
    # Reuse the two seeds registered before the earlier CDF-dog DMIQ render.
    # This makes the prompt redesign directly comparable and prevents a new
    # visual seed search after seeing the old proposals.
    {"seed_id": "seed-20260808", "seed": 20_260_808, "execution_group": "sp4-a"},
    {"seed_id": "seed-20260809", "seed": 20_260_809, "execution_group": "sp4-b"},
)
GROUPS = ("sp4-a", "sp4-b")

_COMMON_SCENE = (
    "An exactly 81-frame realistic video at 25 fps shows exactly one stocky "
    "tan-and-white pit bull wearing a black collar seated on plain gray concrete. "
    "A single long pale bone lies on the concrete beside the dog. "
    "The dog's mouth is empty at frame 0. There are no other dogs, people, "
    "animals, toys, or extra bones. The camera is a locked high overhead view: "
    "no pan, tilt, zoom, dolly, orbit, reframing, cut, or viewpoint change. "
)

BRANCH_PROMPTS = {
    "full_action": _COMMON_SCENE
    + (
        "The dog performs one clear ordered action. It lowers its head toward "
        "the pale bone, makes visible muzzle contact, closes its jaws to grip "
        "the bone, lifts the bone fully off the concrete, raises its head, and "
        "then holds the same bone securely in its mouth without dropping it "
        "through every frame from frame 65 through frame 80. The bone remains "
        "grounded until the grip and is visibly airborne after the lift. The "
        "dog's coat pattern, body shape, black collar, concrete, lighting, and "
        "locked camera remain stable throughout."
    ),
    "noop": _COMMON_SCENE
    + (
        "The dog remains completely still in the initial pose for all 81 "
        "frames. It does not lower its head, sniff, touch, push, grip, lift, or "
        "hold the bone. Its mouth stays empty and the pale bone stays completely "
        "motionless and ground-supported in its initial position. The coat "
        "pattern, body shape, black collar, concrete, lighting, and locked "
        "camera remain stable throughout."
    ),
    "incomplete": _COMMON_SCENE
    + (
        "The dog begins only an incomplete prefix: it lowers its head, sniffs "
        "the pale bone, and briefly touches the bone with its muzzle. It never "
        "closes its jaws around the bone, never grips it, and never lifts or "
        "holds it. The bone remains ground-supported and the dog ends with an "
        "empty mouth. The coat pattern, body shape, black collar, concrete, "
        "lighting, and locked camera remain stable throughout."
    ),
    "reverse": _COMMON_SCENE
    + (
        "Starting from the same empty-mouth dog and ground-supported bone, the "
        "dog performs only the registered negative action: it nudges and pushes "
        "the pale bone a short distance along the concrete. The bone stays in "
        "contact with the ground at all times. The dog never grips, lifts, "
        "carries, or holds the bone and ends with an empty mouth. The coat "
        "pattern, body shape, black collar, concrete, lighting, and locked "
        "camera remain stable throughout."
    ),
}

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}")


class SourceCaptionPassAError(RuntimeError):
    """Raised before ambiguous or privileged Pass A output is accepted."""


def canonical_json_bytes(value: Any) -> bytes:
    # Match infer_lora.object_sha256 exactly so native receipt digests can be
    # checked independently without importing the GPU runtime.
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SourceCaptionPassAError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise SourceCaptionPassAError(f"non-finite JSON constant: {value}")


def _inspect_initial_gaussian_safetensors(path: str | Path) -> dict[str, Any]:
    """Parse and hash the native Gaussian without trusting its receipt fields."""

    resolved = _plain_absolute_file(path, label="native Gaussian artifact")
    expected_numel = math.prod(LATENT_SHAPE)
    expected_nbytes = expected_numel * 4
    file_size = resolved.stat().st_size
    try:
        with resolved.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise SourceCaptionPassAError(
                    "native Gaussian safetensors header is truncated"
                )
            header_length = struct.unpack("<Q", prefix)[0]
            if (
                header_length <= 0
                or header_length > 1 << 20
                or header_length > file_size - 8
            ):
                raise SourceCaptionPassAError(
                    "native Gaussian safetensors header length differs"
                )
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise SourceCaptionPassAError(
                    "native Gaussian safetensors header is truncated"
                )
            try:
                header = json.loads(
                    header_bytes.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_pairs,
                    parse_constant=_reject_json_constant,
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                raise SourceCaptionPassAError(
                    "native Gaussian safetensors header is invalid"
                ) from error
            if not isinstance(header, dict) or set(header) != {
                "__metadata__",
                "official_initial_gaussian",
            }:
                raise SourceCaptionPassAError(
                    "native Gaussian safetensors tensor/key closure differs"
                )
            expected_metadata = {
                "coordinate": "bernini_native_target_latent_before_rearrange",
                "source": "observed_return_of_official_module_global_randn_tensor",
                "observer_only": "true",
                "external_initial_noise_injection": "false",
            }
            if header["__metadata__"] != expected_metadata:
                raise SourceCaptionPassAError(
                    "native Gaussian safetensors metadata differs"
                )
            tensor = header["official_initial_gaussian"]
            if (
                not isinstance(tensor, dict)
                or set(tensor) != {"dtype", "shape", "data_offsets"}
                or tensor.get("dtype") != "F32"
                or tensor.get("shape") != list(LATENT_SHAPE)
                or tensor.get("data_offsets") != [0, expected_nbytes]
            ):
                raise SourceCaptionPassAError(
                    "native Gaussian safetensors tensor contract differs"
                )
            if file_size != 8 + header_length + expected_nbytes:
                raise SourceCaptionPassAError(
                    "native Gaussian safetensors data extent differs"
                )
            artifact_digest = hashlib.sha256(prefix + header_bytes)
            value_digest = hashlib.sha256()
            remaining = expected_nbytes
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise SourceCaptionPassAError(
                        "native Gaussian safetensors data is truncated"
                    )
                artifact_digest.update(chunk)
                value_digest.update(chunk)
                remaining -= len(chunk)
            if handle.read(1):
                raise SourceCaptionPassAError(
                    "native Gaussian safetensors has trailing data"
                )
    except OSError as error:
        raise SourceCaptionPassAError(
            "native Gaussian safetensors could not be read"
        ) from error
    return {
        "tensor_key": "official_initial_gaussian",
        "shape": list(LATENT_SHAPE),
        "stored_dtype": "torch.float32",
        "numel": expected_numel,
        "byte_count": expected_nbytes,
        "artifact_file_sha256": artifact_digest.hexdigest(),
        "tensor_value_sha256": value_digest.hexdigest(),
        "independently_parsed_without_renderer_receipt": True,
    }


def _require_sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        raise SourceCaptionPassAError(f"{label} must be a full lowercase SHA-1")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SourceCaptionPassAError(f"{label} must be a lowercase SHA-256")
    return value


def _plain_absolute_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise SourceCaptionPassAError(f"{label} must be absolute")
    resolved = requested.resolve(strict=True)
    if requested != resolved or not resolved.is_file() or resolved.is_symlink():
        raise SourceCaptionPassAError(f"{label} must be a canonical plain file")
    return resolved


def _fresh_absolute_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested == Path("/"):
        raise SourceCaptionPassAError(f"{label} must be absolute and non-root")
    if _SLUG.fullmatch(requested.name) is None:
        raise SourceCaptionPassAError(f"{label} basename is unsafe")
    parent = requested.parent.resolve(strict=True)
    if requested.parent != parent or not parent.is_dir() or parent.is_symlink():
        raise SourceCaptionPassAError(f"{label} parent must be canonical and plain")
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise SourceCaptionPassAError(f"refusing to overwrite {label}")
    return output


def _plain_entries_root(output_root: Path) -> Path:
    requested = output_root / "entries"
    if requested.is_symlink():
        raise SourceCaptionPassAError("bank entries directory must be plain")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise SourceCaptionPassAError("bank entries directory is absent") from error
    if requested != resolved or not resolved.is_dir() or resolved.is_symlink():
        raise SourceCaptionPassAError("bank entries directory must be canonical and plain")
    return resolved


def write_json_create_only_atomically(path: str | Path, value: Any) -> Path:
    """Publish canonical JSON atomically without ever replacing an existing name."""

    output = _fresh_absolute_file(path, label="JSON output")
    payload = canonical_json_bytes(value)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    linked = False
    try:
        # A hard-link publication is atomic and create-only on one filesystem.
        os.link(temporary, output)
        linked = True
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory_fd)
            except OSError as error:
                if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                    raise
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise SourceCaptionPassAError("refusing concurrent JSON publication") from error
    except BaseException:
        if linked and output.is_file() and not output.is_symlink():
            output.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _load_json(path: str | Path, *, label: str) -> tuple[dict[str, Any], Path]:
    resolved = _plain_absolute_file(path, label=label)
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceCaptionPassAError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise SourceCaptionPassAError(f"{label} root must be an object")
    return value, resolved


def build_manifest(
    *,
    source_video: str | Path,
    expected_source_sha256: str,
    num_inference_steps: int,
    method_source_revision: str,
    method_source_archive_sha256: str,
) -> dict[str, Any]:
    source = _plain_absolute_file(source_video, label="CDF-dog source video")
    expected_source_sha256 = _require_sha256(
        expected_source_sha256, label="expected source SHA-256"
    )
    if expected_source_sha256 != CDF_DOG_SOURCE_SHA256:
        raise SourceCaptionPassAError("source digest is not the registered CDF-dog video")
    if file_sha256(source) != expected_source_sha256:
        raise SourceCaptionPassAError("CDF-dog source bytes differ")
    if type(num_inference_steps) is not int or num_inference_steps not in ALLOWED_STEPS:
        raise SourceCaptionPassAError("num inference steps must be exactly 1 or 40")
    revision = _require_sha1(method_source_revision, label="method source revision")
    archive_sha = _require_sha256(
        method_source_archive_sha256, label="method source archive SHA-256"
    )

    entries: list[dict[str, Any]] = []
    for seed_row in SEED_ROWS:
        for branch_index, branch in enumerate(BRANCH_ORDER):
            prompt = BRANCH_PROMPTS[branch]
            entry_id = f"{seed_row['seed_id']}-{branch.replace('_', '-')}"
            entries.append(
                {
                    "entry_id": entry_id,
                    "seed_id": seed_row["seed_id"],
                    "seed": seed_row["seed"],
                    "execution_group": seed_row["execution_group"],
                    "group_local_order": branch_index,
                    "semantic_branch": branch,
                    "prompt": prompt,
                    "prompt_utf8_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                    "output_subdir": f"entries/{entry_id}",
                }
            )

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "method": METHOD,
        "stage": "engineering-one-step"
        if num_inference_steps == 1
        else "exact40-qualification-candidate",
        "source_geometry_video": {
            "path": str(source),
            "sha256": expected_source_sha256,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "height": VIDEO_HEIGHT,
            "width": VIDEO_WIDTH,
            "renderer_use": "hash_verification_and_fixed_496x480_bucket_only",
            "pixels_forwarded_to_sampler": False,
            "source_latent_constructed_for_t2v": False,
            "reference_latent_constructed_for_t2v": False,
        },
        "renderer_contract": {
            "implementation": "infer_native_identity_generation_canary.py",
            "implementation_arm": "t2v",
            "guidance_mode": "t2v_apg",
            "method_source_revision": revision,
            "method_source_archive_sha256": archive_sha,
            "method_source_preregistered_before_render": True,
            "bernini_commit": BERNINI_COMMIT,
            "veomni_commit": VEOMNI_COMMIT,
            "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA256,
            "num_frames": FRAME_COUNT,
            "fps": FPS,
            "height": VIDEO_HEIGHT,
            "width": VIDEO_WIDTH,
            "latent_shape": list(LATENT_SHAPE),
            "num_inference_steps": num_inference_steps,
            "ulysses_size": ULYSSES_SIZE,
            "omega_vid": 1.25,
            "omega_img": 4.5,
            "omega_txt": 4.0,
            "omega_scale": 0.8,
            "flow_shift": 5.0,
            "eta": 0.5,
            "norm_threshold": [50.0, 50.0],
            "momentum": 0.0,
            "single_expert": "transformer_1",
            "target_initialization": "official_gen_wanx22_fresh_gaussian",
            "full_source_video_count": 0,
            "source_derived_reference_count": 0,
            "multi_video_vae_latents": None,
            "multi_image_vae_latents": None,
            "image_vae_latents": None,
            "target_mixed_with_source_latent": False,
            "training_performed": False,
        },
        "factorial_contract": {
            "branch_order": list(BRANCH_ORDER),
            "seed_rows": [dict(row) for row in SEED_ROWS],
            "seed_count": len(SEED_ROWS),
            "branch_count_per_seed": len(BRANCH_ORDER),
            "entry_count": len(entries),
            "same_seed_and_native_gaussian_within_seed_group_required": True,
            "all_branches_for_every_seed_required": True,
            "posthoc_single_seed_or_single_branch_selection_forbidden": True,
            "source_specific_content": {
                "subject": "exactly_one_stocky_tan_and_white_pit_bull",
                "collar": "black",
                "scene": "plain_gray_concrete",
                "camera": "locked_high_overhead",
                "object": "one_long_pale_bone",
                "other_dogs_or_toys": False,
            },
            "full_action_milestones": [
                "lower_head",
                "visible_muzzle_contact",
                "grip",
                "lift_fully_off_ground",
                "head_rise",
                "hold_frames_65_through_80",
            ],
        },
        "execution_topology": {
            "nodes": 1,
            "gpu_count": 8,
            "parallel_groups": [
                {
                    "group_id": row["execution_group"],
                    "visible_device_slots": [0, 1, 2, 3]
                    if row["execution_group"] == "sp4-a"
                    else [4, 5, 6, 7],
                    "world_size": ULYSSES_SIZE,
                    "seed_id": row["seed_id"],
                    "seed": row["seed"],
                    "branches": list(BRANCH_ORDER),
                }
                for row in SEED_ROWS
            ],
            "groups_run_concurrently": True,
            "entries_within_group_run_in_registered_order": True,
            "ulysses8_forbidden": True,
        },
        "entries": entries,
        "qualification_contract": {
            "renderer_labels_are_not_event_verification": True,
            "exact40_requires_independent_manual_qualification": True,
            "qualification_unit": "complete_two_seed_by_four_branch_bank",
            "reject_pass_a_if_either_seed_or_any_branch_fails": True,
            "best_seed_or_branch_selection_forbidden": True,
            "training_or_reward_use_before_qualification": False,
        },
        "interpretation": {
            "pure_t2v_action_proposal_bank": True,
            "video_editing_result": False,
            "quality_claim": False,
            "semantic_event_verified": False,
            "model_training_performed": False,
            "pass_a_status": "pending_render_and_independent_manual_qualification",
        },
    }
    manifest["manifest_digest"] = object_sha256(manifest)
    return validate_manifest(manifest)


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != MANIFEST_SCHEMA:
        raise SourceCaptionPassAError("manifest schema differs")
    declared = _require_sha256(value.get("manifest_digest"), label="manifest digest")
    unsigned = dict(value)
    unsigned.pop("manifest_digest", None)
    if object_sha256(unsigned) != declared:
        raise SourceCaptionPassAError("manifest embedded digest differs")
    source = value.get("source_geometry_video")
    renderer = value.get("renderer_contract")
    factorial = value.get("factorial_contract")
    topology = value.get("execution_topology")
    qualification = value.get("qualification_contract")
    if not all(
        isinstance(row, Mapping)
        for row in (source, renderer, factorial, topology, qualification)
    ):
        raise SourceCaptionPassAError("manifest contract section differs")
    if (
        source.get("sha256") != CDF_DOG_SOURCE_SHA256
        or source.get("frame_count") != FRAME_COUNT
        or source.get("fps") != FPS
        or source.get("height") != VIDEO_HEIGHT
        or source.get("width") != VIDEO_WIDTH
        or source.get("renderer_use")
        != "hash_verification_and_fixed_496x480_bucket_only"
        or source.get("pixels_forwarded_to_sampler") is not False
        or source.get("source_latent_constructed_for_t2v") is not False
        or source.get("reference_latent_constructed_for_t2v") is not False
    ):
        raise SourceCaptionPassAError("source geometry-only contract differs")
    steps = renderer.get("num_inference_steps")
    if (
        type(steps) is not int
        or steps not in ALLOWED_STEPS
        or renderer.get("implementation")
        != "infer_native_identity_generation_canary.py"
        or renderer.get("implementation_arm") != "t2v"
        or renderer.get("guidance_mode") != "t2v_apg"
        or renderer.get("bernini_commit") != BERNINI_COMMIT
        or renderer.get("veomni_commit") != VEOMNI_COMMIT
        or renderer.get("checkpoint_tree_sha256") != CHECKPOINT_TREE_SHA256
        or renderer.get("num_frames") != FRAME_COUNT
        or renderer.get("fps") != FPS
        or renderer.get("latent_shape") != list(LATENT_SHAPE)
        or renderer.get("ulysses_size") != ULYSSES_SIZE
        or renderer.get("omega_vid") != 1.25
        or renderer.get("omega_img") != 4.5
        or renderer.get("omega_txt") != 4.0
        or renderer.get("omega_scale") != 0.8
        or renderer.get("flow_shift") != 5.0
        or renderer.get("eta") != 0.5
        or renderer.get("norm_threshold") != [50.0, 50.0]
        or renderer.get("momentum") != 0.0
        or renderer.get("single_expert") != "transformer_1"
        or renderer.get("full_source_video_count") != 0
        or renderer.get("source_derived_reference_count") != 0
        or renderer.get("multi_video_vae_latents") is not None
        or renderer.get("multi_image_vae_latents") is not None
        or renderer.get("image_vae_latents") is not None
        or renderer.get("target_mixed_with_source_latent") is not False
        or renderer.get("training_performed") is not False
    ):
        raise SourceCaptionPassAError("pure native T2V renderer contract differs")
    _require_sha1(renderer.get("method_source_revision"), label="method revision")
    _require_sha256(
        renderer.get("method_source_archive_sha256"), label="method archive SHA-256"
    )
    if (
        factorial.get("branch_order") != list(BRANCH_ORDER)
        or factorial.get("seed_rows") != [dict(row) for row in SEED_ROWS]
        or factorial.get("entry_count") != 8
        or factorial.get("same_seed_and_native_gaussian_within_seed_group_required")
        is not True
        or factorial.get("all_branches_for_every_seed_required") is not True
        or factorial.get("posthoc_single_seed_or_single_branch_selection_forbidden")
        is not True
    ):
        raise SourceCaptionPassAError("factorial closure differs")
    if (
        topology.get("gpu_count") != 8
        or topology.get("groups_run_concurrently") is not True
        or topology.get("ulysses8_forbidden") is not True
    ):
        raise SourceCaptionPassAError("dual-WORLD4 topology differs")
    if (
        qualification.get("renderer_labels_are_not_event_verification") is not True
        or qualification.get("exact40_requires_independent_manual_qualification")
        is not True
        or qualification.get("reject_pass_a_if_either_seed_or_any_branch_fails")
        is not True
        or qualification.get("best_seed_or_branch_selection_forbidden") is not True
        or qualification.get("training_or_reward_use_before_qualification") is not False
    ):
        raise SourceCaptionPassAError("qualification fail-closed contract differs")

    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != 8:
        raise SourceCaptionPassAError("manifest must contain exactly eight entries")
    expected_pairs = [
        (row["seed_id"], branch, row["execution_group"], row["seed"])
        for row in SEED_ROWS
        for branch in BRANCH_ORDER
    ]
    observed_pairs: list[tuple[Any, Any, Any, Any]] = []
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SourceCaptionPassAError("entry must be an object")
        entry_id = entry.get("entry_id")
        branch = entry.get("semantic_branch")
        if (
            not isinstance(entry_id, str)
            or _SLUG.fullmatch(entry_id) is None
            or entry_id in seen_ids
            or branch not in BRANCH_ORDER
            or entry.get("prompt") != BRANCH_PROMPTS[branch]
            or entry.get("prompt_utf8_sha256")
            != hashlib.sha256(BRANCH_PROMPTS[branch].encode("utf-8")).hexdigest()
            or entry.get("output_subdir") != f"entries/{entry_id}"
        ):
            raise SourceCaptionPassAError("entry identity or prompt differs")
        seen_ids.add(entry_id)
        observed_pairs.append(
            (
                entry.get("seed_id"),
                branch,
                entry.get("execution_group"),
                entry.get("seed"),
            )
        )
    if observed_pairs != expected_pairs:
        raise SourceCaptionPassAError("entry order or seed/group binding differs")
    return dict(value)


def load_manifest(
    path: str | Path, *, expected_file_sha256: Optional[str] = None
) -> tuple[dict[str, Any], Path, str]:
    value, resolved = _load_json(path, label="Pass A manifest")
    actual_sha = file_sha256(resolved)
    if expected_file_sha256 is not None and actual_sha != _require_sha256(
        expected_file_sha256, label="manifest file SHA-256"
    ):
        raise SourceCaptionPassAError("manifest file SHA-256 differs")
    return validate_manifest(value), resolved, actual_sha


def entries_for_group(manifest: Mapping[str, Any], group: str) -> list[dict[str, Any]]:
    if group not in GROUPS:
        raise SourceCaptionPassAError("unknown execution group")
    rows = [
        dict(entry)
        for entry in manifest["entries"]
        if entry["execution_group"] == group
    ]
    if [row["semantic_branch"] for row in rows] != list(BRANCH_ORDER):
        raise SourceCaptionPassAError("execution group lacks its complete branch set")
    if len({row["seed"] for row in rows}) != 1:
        raise SourceCaptionPassAError("execution group spans multiple seeds")
    return rows


def entry_by_id(manifest: Mapping[str, Any], entry_id: str) -> dict[str, Any]:
    if not isinstance(entry_id, str) or _SLUG.fullmatch(entry_id) is None:
        raise SourceCaptionPassAError("entry id is unsafe")
    rows = [entry for entry in manifest["entries"] if entry["entry_id"] == entry_id]
    if len(rows) != 1:
        raise SourceCaptionPassAError("entry id is absent or non-unique")
    return dict(rows[0])


def render_entry(
    *,
    manifest_path: str,
    manifest_file_sha256: str,
    entry_id: str,
    output_root: str,
    bernini_root: str,
    veomni_root: str,
    checkpoint: str,
    checkpoint_content_manifest: str,
    source_video: str,
    method_source_revision: str,
    method_source_archive_sha256: str,
) -> int:
    manifest, _, _ = load_manifest(
        manifest_path, expected_file_sha256=manifest_file_sha256
    )
    entry = entry_by_id(manifest, entry_id)
    renderer = manifest["renderer_contract"]
    if (
        method_source_revision != renderer["method_source_revision"]
        or method_source_archive_sha256
        != renderer["method_source_archive_sha256"]
    ):
        raise SourceCaptionPassAError("runtime method archive differs from manifest")
    source = _plain_absolute_file(source_video, label="CDF-dog source video")
    if file_sha256(source) != CDF_DOG_SOURCE_SHA256:
        raise SourceCaptionPassAError("runtime CDF-dog source bytes differ")
    root = Path(output_root).expanduser()
    if (
        not root.is_absolute()
        or root != root.resolve(strict=True)
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise SourceCaptionPassAError("output root must be canonical and plain")
    entries_root = _plain_entries_root(root)
    output = root / entry["output_subdir"]
    if output.parent != entries_root or output.is_symlink():
        raise SourceCaptionPassAError("entry output escaped entries directory")

    # Reuse the pinned native T2V runner.  A fresh process renders every entry,
    # so this temporary constant only authorizes the one-step engineering rung;
    # exact40 follows the native default unchanged.  Sampling code is not copied
    # or replaced; finalization independently parses the native Gaussian artifact.
    import infer_native_identity_generation_canary as native

    steps = int(renderer["num_inference_steps"])
    original_steps = native.NUM_INFERENCE_STEPS
    native.NUM_INFERENCE_STEPS = steps
    try:
        return native.main(
            [
                "--bernini-root",
                bernini_root,
                "--veomni-root",
                veomni_root,
                "--checkpoint",
                checkpoint,
                "--checkpoint-content-manifest",
                checkpoint_content_manifest,
                "--source-video",
                str(source),
                "--expected-source-sha256",
                CDF_DOG_SOURCE_SHA256,
                "--action-prompt",
                entry["prompt"],
                "--expected-action-prompt-sha256",
                entry["prompt_utf8_sha256"],
                "--output-dir",
                str(output),
                "--arms",
                "t2v",
                "--num-inference-steps",
                str(steps),
                "--seed",
                str(entry["seed"]),
                "--expected-bernini-commit",
                renderer["bernini_commit"],
                "--expected-veomni-commit",
                renderer["veomni_commit"],
                "--expected-checkpoint-tree-sha256",
                renderer["checkpoint_tree_sha256"],
                "--method-source-revision",
                method_source_revision,
                "--method-source-archive-sha256",
                method_source_archive_sha256,
            ]
        )
    finally:
        native.NUM_INFERENCE_STEPS = original_steps


def _load_sealed_native_receipt(path: Path) -> dict[str, Any]:
    value, _ = _load_json(path, label="native T2V receipt")
    declared = _require_sha256(value.get("receipt_digest"), label="native receipt digest")
    unsigned = dict(value)
    unsigned.pop("receipt_digest", None)
    if object_sha256(unsigned) != declared:
        raise SourceCaptionPassAError("native receipt embedded digest differs")
    if value.get("schema_version") != NATIVE_RECEIPT_SCHEMA:
        raise SourceCaptionPassAError("native receipt schema differs")
    return value


def _audit_native_entry(
    manifest: Mapping[str, Any], entry: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    entries_root = _plain_entries_root(output_root)
    requested_entry_root = output_root / entry["output_subdir"]
    if requested_entry_root.parent != entries_root or requested_entry_root.is_symlink():
        raise SourceCaptionPassAError("native entry directory escaped entries root")
    entry_root = requested_entry_root.resolve(strict=True)
    if (
        requested_entry_root != entry_root
        or entry_root.parent != entries_root
        or not entry_root.is_dir()
        or entry_root.is_symlink()
    ):
        raise SourceCaptionPassAError("native entry directory is not plain")
    receipt_path = entry_root / "receipt.json"
    receipt = _load_sealed_native_receipt(receipt_path)
    if receipt.get("arms") != ["t2v"]:
        raise SourceCaptionPassAError("native entry is not exclusively T2V")
    renderer = manifest["renderer_contract"]
    inputs = receipt.get("input")
    conditioning = receipt.get("conditioning", {}).get("t2v")
    sampling = receipt.get("sampling", {}).get("t2v")
    source_ids = conditioning.get("source_ids") if isinstance(conditioning, Mapping) else None
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("source_video_sha256") != CDF_DOG_SOURCE_SHA256
        or inputs.get("action_prompt_utf8_sha256") != entry["prompt_utf8_sha256"]
        or inputs.get("accepted_external_conditions")
        != ["source_video", "action_prompt"]
        or inputs.get("target_video") is not False
        or inputs.get("external_reference_image_or_video") is not False
        or inputs.get("external_mask_flow_pose_track_trajectory") is not False
        or inputs.get("external_first_frame_anchor") is not False
        or not isinstance(conditioning, Mapping)
        or conditioning.get("full_source_video_count") != 0
        or conditioning.get("source_derived_reference_count") != 0
        or conditioning.get("source_frame_indices") != []
        or receipt.get("source_condition_artifact") is not None
        or not isinstance(source_ids, Mapping)
        or source_ids.get("conditioning_source_count") != 0
        or source_ids.get("video_source_ids") != []
        or source_ids.get("reference_source_ids") != []
    ):
        raise SourceCaptionPassAError("source or privileged condition entered T2V")
    if (
        not isinstance(sampling, Mapping)
        or sampling.get("num_frames") != FRAME_COUNT
        or sampling.get("num_inference_steps") != renderer["num_inference_steps"]
        or sampling.get("guidance_mode") != "t2v_apg"
        or sampling.get("omega_vid") != renderer["omega_vid"]
        or sampling.get("omega_img") != renderer["omega_img"]
        or sampling.get("omega_txt") != renderer["omega_txt"]
        or sampling.get("omega_scale") != renderer["omega_scale"]
        or sampling.get("flow_shift") != renderer["flow_shift"]
        or sampling.get("eta") != renderer["eta"]
        or sampling.get("norm_threshold") != renderer["norm_threshold"]
        or sampling.get("momentum") != renderer["momentum"]
        or sampling.get("single_expert") != renderer["single_expert"]
        or sampling.get("seed") != entry["seed"]
        or sampling.get("target_initialization")
        != "official_gen_wanx22_fresh_gaussian"
        or sampling.get("target_mixed_with_source_latent") is not False
        or sampling.get("custom_sampler_or_scheduler") is not False
        or sampling.get("ulysses_size") != ULYSSES_SIZE
    ):
        raise SourceCaptionPassAError("native exact sampling contract differs")
    geometry = receipt.get("latent_geometry")
    if not isinstance(geometry, Mapping) or geometry.get("video_latent_shape") != list(
        LATENT_SHAPE
    ):
        raise SourceCaptionPassAError("native exact81 latent geometry differs")
    if (
        receipt.get("method_source_revision") != renderer["method_source_revision"]
        or receipt.get("method_source_archive_sha256")
        != renderer["method_source_archive_sha256"]
        or receipt.get("bernini_commit") != renderer["bernini_commit"]
        or receipt.get("veomni_commit") != renderer["veomni_commit"]
        or receipt.get("checkpoint", {}).get("tree_sha256")
        != renderer["checkpoint_tree_sha256"]
        or receipt.get("freeze_certificate", {}).get("base_frozen") is not True
        or receipt.get("interpretation", {}).get("training_performed") is not False
    ):
        raise SourceCaptionPassAError("native frozen provenance differs")

    gaussian_map = receipt.get("initial_noise_artifacts")
    if not isinstance(gaussian_map, Mapping) or set(gaussian_map) != {"t2v"}:
        raise SourceCaptionPassAError("native Gaussian artifact is absent")
    gaussian = gaussian_map["t2v"]
    if not isinstance(gaussian, Mapping):
        raise SourceCaptionPassAError("native Gaussian receipt differs")
    gaussian_path = _plain_absolute_file(
        gaussian.get("path", ""), label="native Gaussian artifact"
    )
    independent_gaussian = _inspect_initial_gaussian_safetensors(gaussian_path)
    artifact_sha = _require_sha256(
        gaussian.get("sha256"), label="native Gaussian artifact SHA-256"
    )
    raw_sha = _require_sha256(
        gaussian.get("tensor_value_sha256"), label="native Gaussian value SHA-256"
    )
    if (
        gaussian_path.parent != entry_root
        or independent_gaussian["artifact_file_sha256"] != artifact_sha
        or independent_gaussian["tensor_value_sha256"] != raw_sha
        or gaussian.get("raw_value_sha256") != raw_sha
        or gaussian.get("tensor_key") != independent_gaussian["tensor_key"]
        or gaussian.get("shape") != list(LATENT_SHAPE)
        or gaussian.get("stored_dtype") != "torch.float32"
        or gaussian.get("numel") != independent_gaussian["numel"]
        or gaussian.get("byte_count") != independent_gaussian["byte_count"]
        or gaussian.get("official_randn_tensor_call_count") != 1
        or gaussian.get("captured_from_native_sampler") is not True
        or gaussian.get("observer_changed_return_value") is not False
        or gaussian.get("source_or_target_derived") is not False
        or gaussian.get("all_rank_identity", {}).get("all_rank_exact") is not True
    ):
        raise SourceCaptionPassAError("native Gaussian provenance differs")

    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"t2v"}:
        raise SourceCaptionPassAError("native T2V output set differs")
    output = outputs["t2v"]
    video = _plain_absolute_file(output.get("path", ""), label="native T2V video")
    clean = output.get("normalized_clean_latent")
    if not isinstance(clean, Mapping):
        raise SourceCaptionPassAError("native clean latent receipt is absent")
    clean_path = _plain_absolute_file(
        clean.get("path", ""), label="native clean latent"
    )
    if (
        video.parent != entry_root
        or clean_path.parent != entry_root
        or file_sha256(video) != output.get("sha256")
        or file_sha256(clean_path) != clean.get("sha256")
        or output.get("frame_count") != FRAME_COUNT
        or output.get("fps") != FPS
        or output.get("height") != VIDEO_HEIGHT
        or output.get("width") != VIDEO_WIDTH
        or clean.get("shape") != list(LATENT_SHAPE)
        or clean.get("native_sampler_before_vae_decode") is not True
        or clean.get("mp4_decode_reencode_used") is not False
    ):
        raise SourceCaptionPassAError("native T2V artifact contract differs")

    return {
        "entry_id": entry["entry_id"],
        "seed_id": entry["seed_id"],
        "seed": entry["seed"],
        "execution_group": entry["execution_group"],
        "semantic_branch": entry["semantic_branch"],
        "native_receipt_path": str(receipt_path),
        "native_receipt_sha256": file_sha256(receipt_path),
        "native_receipt_digest": receipt["receipt_digest"],
        "video_path": str(video),
        "video_sha256": output["sha256"],
        "clean_latent_path": str(clean_path),
        "clean_latent_sha256": clean["sha256"],
        "initial_gaussian_path": str(gaussian_path),
        "initial_gaussian_file_sha256": artifact_sha,
        "initial_gaussian_value_sha256": raw_sha,
        "initial_gaussian_independently_parsed": independent_gaussian[
            "independently_parsed_without_renderer_receipt"
        ],
        "pure_t2v_condition_audit_pass": True,
        "semantic_event_verified": False,
    }


def finalize_bank(
    *,
    manifest_path: str,
    manifest_file_sha256: str,
    output_root: str,
    output_receipt: str,
) -> dict[str, Any]:
    manifest, manifest_resolved, manifest_sha = load_manifest(
        manifest_path, expected_file_sha256=manifest_file_sha256
    )
    requested_root = Path(output_root).expanduser()
    if (
        not requested_root.is_absolute()
        or requested_root != requested_root.resolve(strict=True)
        or not requested_root.is_dir()
        or requested_root.is_symlink()
    ):
        raise SourceCaptionPassAError("bank output root must be canonical and plain")
    root = requested_root
    audited = [
        _audit_native_entry(manifest, entry, root) for entry in manifest["entries"]
    ]
    if any(
        row.get("initial_gaussian_independently_parsed") is not True
        for row in audited
    ):
        raise SourceCaptionPassAError(
            "every Gaussian must be independently parsed before bank finalization"
        )
    per_seed_gaussians: dict[str, str] = {}
    for seed_row in SEED_ROWS:
        rows = [row for row in audited if row["seed_id"] == seed_row["seed_id"]]
        if [row["semantic_branch"] for row in rows] != list(BRANCH_ORDER):
            raise SourceCaptionPassAError("a seed lacks its complete branch factorial")
        values = {row["initial_gaussian_value_sha256"] for row in rows}
        if len(values) != 1:
            raise SourceCaptionPassAError(
                "branches within one seed do not share an identical Gaussian"
            )
        per_seed_gaussians[seed_row["seed_id"]] = next(iter(values))
    if len(set(per_seed_gaussians.values())) != len(SEED_ROWS):
        raise SourceCaptionPassAError("the two preregistered seeds reused one Gaussian")

    receipt: dict[str, Any] = {
        "schema_version": BANK_RECEIPT_SCHEMA,
        "method": METHOD,
        "stage": manifest["stage"],
        "manifest_path": str(manifest_resolved),
        "manifest_file_sha256": manifest_sha,
        "manifest_digest": manifest["manifest_digest"],
        "method_source_revision": manifest["renderer_contract"][
            "method_source_revision"
        ],
        "method_source_archive_sha256": manifest["renderer_contract"][
            "method_source_archive_sha256"
        ],
        "entry_count": len(audited),
        "seed_count": len(SEED_ROWS),
        "branch_count_per_seed": len(BRANCH_ORDER),
        "entries": audited,
        "initial_gaussian_contract": {
            "per_seed_value_sha256": per_seed_gaussians,
            "same_value_across_all_four_branches_within_seed": True,
            "different_values_across_the_two_seeds": True,
            "tensor_values_recomputed_from_safetensors": True,
            "posthoc_seed_selection": False,
        },
        "condition_closure": {
            "renderer_arm": "t2v",
            "guidance_mode": "t2v_apg",
            "source_video_role": "hash_verification_and_fixed_496x480_bucket_only",
            "source_pixels_forwarded_to_sampler": False,
            "source_video_latent_consumed": False,
            "source_reference_latent_consumed": False,
            "target_video_consumed": False,
            "mask_flow_pose_track_trajectory_consumed": False,
            "all_native_entry_condition_audits_pass": True,
        },
        "qualification": {
            "manifest_semantic_labels_are_not_event_acceptance": True,
            "semantic_events_verified": False,
            "exact40_manual_qualification_required": manifest["renderer_contract"][
                "num_inference_steps"
            ]
            == EXACT_STEPS,
            "qualification_unit": "complete_two_seed_by_four_branch_bank",
            "reject_pass_a_if_either_seed_or_any_branch_fails": True,
            "single_seed_or_branch_selection_forbidden": True,
            "reward_or_training_use_authorized": False,
            "pass_a_status": "pending_independent_manual_qualification"
            if manifest["renderer_contract"]["num_inference_steps"] == EXACT_STEPS
            else "engineering_only_no_semantic_claim",
        },
        "interpretation": {
            "render_complete": True,
            "pure_t2v_action_proposal_bank": True,
            "editing_result": False,
            "quality_claim": False,
            "model_training_performed": False,
            "scientific_claim_authorized": False,
        },
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    requested_receipt = Path(output_receipt).expanduser()
    if not requested_receipt.is_absolute() or requested_receipt.parent != root:
        raise SourceCaptionPassAError("bank receipt must remain inside output root")
    write_json_create_only_atomically(output_receipt, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest")
    build.add_argument("--source-video", required=True)
    build.add_argument("--expected-source-sha256", required=True)
    build.add_argument("--num-inference-steps", type=int, choices=ALLOWED_STEPS, required=True)
    build.add_argument("--method-source-revision", required=True)
    build.add_argument("--method-source-archive-sha256", required=True)
    build.add_argument("--output", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--expected-file-sha256")

    listing = subparsers.add_parser("list-entry-ids")
    listing.add_argument("--manifest", required=True)
    listing.add_argument("--expected-file-sha256", required=True)
    listing.add_argument("--group", required=True, choices=GROUPS)

    render = subparsers.add_parser("render-entry")
    render.add_argument("--manifest", required=True)
    render.add_argument("--expected-file-sha256", required=True)
    render.add_argument("--entry-id", required=True)
    render.add_argument("--output-root", required=True)
    render.add_argument("--bernini-root", required=True)
    render.add_argument("--veomni-root", required=True)
    render.add_argument("--checkpoint", required=True)
    render.add_argument("--checkpoint-content-manifest", required=True)
    render.add_argument("--source-video", required=True)
    render.add_argument("--method-source-revision", required=True)
    render.add_argument("--method-source-archive-sha256", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--expected-file-sha256", required=True)
    finalize.add_argument("--output-root", required=True)
    finalize.add_argument("--output-receipt", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-manifest":
        manifest = build_manifest(
            source_video=args.source_video,
            expected_source_sha256=args.expected_source_sha256,
            num_inference_steps=args.num_inference_steps,
            method_source_revision=args.method_source_revision,
            method_source_archive_sha256=args.method_source_archive_sha256,
        )
        output = write_json_create_only_atomically(args.output, manifest)
        print(file_sha256(output), flush=True)
        return 0
    if args.command == "validate":
        manifest, _, _ = load_manifest(
            args.manifest, expected_file_sha256=args.expected_file_sha256
        )
        print(canonical_json_bytes(manifest).decode("utf-8"), end="", flush=True)
        return 0
    if args.command == "list-entry-ids":
        manifest, _, _ = load_manifest(
            args.manifest, expected_file_sha256=args.expected_file_sha256
        )
        for entry in entries_for_group(manifest, args.group):
            print(entry["entry_id"])
        return 0
    if args.command == "render-entry":
        return render_entry(
            manifest_path=args.manifest,
            manifest_file_sha256=args.expected_file_sha256,
            entry_id=args.entry_id,
            output_root=args.output_root,
            bernini_root=args.bernini_root,
            veomni_root=args.veomni_root,
            checkpoint=args.checkpoint,
            checkpoint_content_manifest=args.checkpoint_content_manifest,
            source_video=args.source_video,
            method_source_revision=args.method_source_revision,
            method_source_archive_sha256=args.method_source_archive_sha256,
        )
    if args.command == "finalize":
        receipt = finalize_bank(
            manifest_path=args.manifest,
            manifest_file_sha256=args.expected_file_sha256,
            output_root=args.output_root,
            output_receipt=args.output_receipt,
        )
        print(canonical_json_bytes(receipt).decode("utf-8"), end="", flush=True)
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_STEPS",
    "BANK_RECEIPT_SCHEMA",
    "BRANCH_ORDER",
    "BRANCH_PROMPTS",
    "CDF_DOG_SOURCE_SHA256",
    "FRAME_COUNT",
    "MANIFEST_SCHEMA",
    "SEED_ROWS",
    "SourceCaptionPassAError",
    "build_manifest",
    "entries_for_group",
    "finalize_bank",
    "main",
    "render_entry",
    "validate_manifest",
    "write_json_create_only_atomically",
]
