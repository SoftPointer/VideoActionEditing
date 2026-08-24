#!/usr/bin/env python3
"""Strict exact-two-row simulator C2 to ELAL-3 oracle-label bridge.

This module authenticates the two checked-in three-entity C2 simulator rows
and all sixteen media/annotation/receipt triples before exposing any label.
Labels are privileged simulator annotations with teacher-forced oracle q.
They are not an ActionPredictor, source+instruction inference, real-video
data, formal C2 evidence, or exact160 evidence.

Entity slots are semantic-role bound.  In particular, a role-swap variant
places the physical entity carrying the ``agent`` role in slot 0; it does not
silently preserve the annotation's physical entity order.  All six directed
K=3 relation edges are materialized and valid.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import torch

import elal3_c0_v1 as elal3


LABEL_SCHEMA_VERSION = "elal3-simulator-c2-oracle-q-label-v1"
PACKET_SCHEMA_VERSION = "elal3-simulator-gt-canary-v1"
ANNOTATION_SCHEMA_VERSION = "elal3-simulator-media-annotation-v1"
ANNOTATION_RECEIPT_SCHEMA_VERSION = "elal3-simulator-annotation-receipt-v1"
EXTERNAL_AUTHORITY_SCHEMA_VERSION = (
    "bernini-elal3-c2-simulator-oracle-q-derivative-authority-v1"
)
EXTERNAL_AUTHORITY_STATUS = (
    "AUTHORIZED_C2_SIMULATOR_ORACLE_Q_DIAGNOSTIC_ONLY"
)
UPSTREAM_STATUS = "ELAL3_SIM_DIAGNOSTIC"
EXPECTED_MANIFEST_SHA256 = (
    "2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"
)
EXPECTED_MANIFEST_DIGEST = (
    "1bc3b7cc155b25028eeab1e940cf6e6ead2c4c0ff189a4f8059f0a8928a383bd"
)
EXPECTED_EXTERNAL_AUTHORITY_RELATIVE_PATH = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c2_simulator_optimizer_diagnostic_authority_v1.json"
)
EXPECTED_EXTERNAL_AUTHORITY_PATH = (
    Path(__file__).resolve().parents[2]
    / EXPECTED_EXTERNAL_AUTHORITY_RELATIVE_PATH
)
EXPECTED_EXTERNAL_AUTHORITY_SHA256 = (
    "543aedd714c7a48c48b4dcc19d1dd6a8bba37d1edda9b1fa195083659380c64a"
)
EXPECTED_EXTERNAL_AUTHORITY_DIGEST = (
    "936e91cf3d1d39dd7f45d5f7a4d510dadcbcb4c2f89a8d22581638fccdefd599"
)
EXPECTED_EXPERIMENT_CONTRACT_RELATIVE_PATH = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c2_role_binding_experiment_contract_v1.json"
)
EXPECTED_EXPERIMENT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / EXPECTED_EXPERIMENT_CONTRACT_RELATIVE_PATH
)
EXPECTED_EXPERIMENT_CONTRACT_SHA256 = (
    "92d700bde0ff9c644f998344d3fecb48bc7c0361f6e948a93c42b924245b25f8"
)
EXPECTED_EXPERIMENT_CONTRACT_DIGEST = (
    "18462dcfbeb017e48a7ed6816559667fa8de1911081261cdc103bc6dd9a229d6"
)
EXPERIMENT_CONTRACT_SCHEMA_VERSION = (
    "bernini-elal3-c2-role-binding-experiment-contract-v1"
)
C2_ROW_IDS = (
    "c2-three-entity-blocking-response",
    "c2-three-entity-handover-occlusion",
)
MEDIA_ORDER = (
    "source",
    "target",
    "anchor",
    "wrong_agent",
    "wrong_object",
    "role_swap",
    "reverse",
    "phase_shuffle",
)
PHYSICAL_ENTITY_ORDER = ("agent", "patient", "object")
RELATION_EDGES = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
RGB_FRAMES = 81
FPS = 25
RGB_HEIGHT = 96
RGB_WIDTH = 128
LATENT_PHASES = 21
LATENT_TO_RGB = tuple(range(0, RGB_FRAMES, 4))
PHASE_CHANNELS = ("onset", "transition", "terminal", "hold")
ROLE_CODE_ORDER = (
    "agent",
    "wrong_agent",
    "patient",
    "co_agent",
    "receiver",
    "inactive",
    "instrument",
    "patient_object",
)
_ACTOR_ROLES = frozenset(("agent", "wrong_agent"))
_OBJECT_ROLES = frozenset(("instrument", "patient_object"))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ELAL3SimulatorC2LabelError(RuntimeError):
    """Raised before accepting an ambiguous C2 packet or oracle label."""


def fail(message: str) -> None:
    raise ELAL3SimulatorC2LabelError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ELAL3SimulatorC2LabelError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def stable_read_path(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_mode: int,
    allowed_root: Path,
    held_root_fd: Optional[int] = None,
) -> tuple[bytes, dict[str, Any]]:
    """Double-read a file through held no-follow openat directory FDs."""

    if _SHA256.fullmatch(expected_sha256) is None:
        fail(f"{label} expected SHA-256 differs")
    root = allowed_root.resolve(strict=True)
    requested = path
    if not requested.is_absolute():
        fail(f"{label} path must be absolute")
    try:
        relative = requested.relative_to(root)
    except ValueError as error:
        raise ELAL3SimulatorC2LabelError(f"{label} escapes allowed root") from error
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        fail(f"{label} relative component closure differs")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    owned_root = held_root_fd is None
    root_descriptor = (
        os.open(str(root), directory_flags)
        if held_root_fd is None
        else int(held_root_fd)
    )
    directory_descriptors: list[int] = []
    parent_descriptor = root_descriptor
    descriptor: Optional[int] = None
    try:
        root_fd_before = os.fstat(root_descriptor)
        root_named_before = root.lstat()
        if (
            _stat_identity(root_fd_before) != _stat_identity(root_named_before)
            or stat.S_ISLNK(root_named_before.st_mode)
            or not stat.S_ISDIR(root_named_before.st_mode)
        ):
            fail(f"{label} held root differs from named root")
        directory_identities = [_stat_identity(root_fd_before)]
        for component in relative.parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=parent_descriptor)
            directory_descriptors.append(child)
            parent_descriptor = child
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                fail(f"{label} parent component is not a directory")
            directory_identities.append(_stat_identity(info))
        final_name = relative.parts[-1]
        named_before = os.stat(
            final_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            stat.S_ISLNK(named_before.st_mode)
            or not stat.S_ISREG(named_before.st_mode)
            or named_before.st_nlink != 1
            or stat.S_IMODE(named_before.st_mode) != expected_mode
        ):
            fail(f"{label} named file type/mode/link closure differs")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(final_name, flags, dir_fd=parent_descriptor)
        before = os.fstat(descriptor)

        def read_all() -> bytes:
            blocks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block:
                    break
                blocks.append(block)
            return b"".join(blocks)

        first = read_all()
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = read_all()
        after = os.fstat(descriptor)
        named_after = os.stat(
            final_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        directory_identities_after = [
            _stat_identity(os.fstat(root_descriptor))
        ] + [
            _stat_identity(os.fstat(item)) for item in directory_descriptors
        ]
        root_named_after = root.lstat()
        root_named_identity_after = _stat_identity(root_named_after)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for item in reversed(directory_descriptors):
            os.close(item)
        if owned_root:
            os.close(root_descriptor)
    identity = _stat_identity(before)
    if (
        first != second
        or identity != _stat_identity(after)
        or identity != _stat_identity(named_before)
        or identity != _stat_identity(named_after)
        or directory_identities != directory_identities_after
        or directory_identities[0] != root_named_identity_after
        or hashlib.sha256(first).hexdigest() != expected_sha256
    ):
        fail(f"{label} held-FD identity/double-read/SHA replay differs")
    return first, {
        "path": str(requested),
        "sha256": expected_sha256,
        "size": len(first),
        "mode": expected_mode,
        "device": before.st_dev,
        "inode": before.st_ino,
        "nlink": before.st_nlink,
        "held_fd_double_read_verified": True,
        "held_openat_parent_chain_replayed": True,
    }


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    fail(f"non-finite JSON number: {value}")


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8", "strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except ELAL3SimulatorC2LabelError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ELAL3SimulatorC2LabelError(
            f"{label} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        fail(f"{label} must contain one JSON object")
    return value


def _plain_file(root: Path, relative: str, *, label: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or str(posix) != relative:
        fail(f"{label} relative path differs")
    path = root.joinpath(*posix.parts)
    try:
        info = path.lstat()
    except OSError as error:
        raise ELAL3SimulatorC2LabelError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        fail(f"{label} must be a non-symlink plain file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ELAL3SimulatorC2LabelError(
            f"{label} escapes packet root"
        ) from error
    return resolved


def _canonical_json_payload(payload: bytes, *, label: str) -> dict[str, Any]:
    value = _strict_json_bytes(payload, label=label)
    if payload != canonical_json_bytes(value) + b"\n":
        fail(f"{label} bytes are not canonical JSON+newline")
    return value


UPSTREAM_AUTHORITY = MappingProxyType(
    {
        "action_encoder_qualification_authorized": False,
        "exact160_claim_authorized": False,
        "exact160_eligible": False,
        "formal_c0_c1_c2_go_authorized": False,
        "model_output": False,
        "real_video_data": False,
        "scientific_claim_authorized": False,
        "simulator_only": True,
        "status": UPSTREAM_STATUS,
        "training_authorized": False,
        "training_use_forbidden": True,
    }
)

EXPECTED_ROW_METADATA: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        C2_ROW_IDS[0]: MappingProxyType(
            {
                "instruction": (
                    "The red agent moves the yellow barrier into the blue "
                    "patient's path; the patient decelerates, stops behind "
                    "it, and all entities hold."
                ),
                "semantic_roles": ("agent", "patient", "instrument"),
            }
        ),
        C2_ROW_IDS[1]: MappingProxyType(
            {
                "instruction": (
                    "The red agent carries the yellow object to the blue "
                    "co-agent, transfers it during partial occlusion, and "
                    "the co-agent holds it at the terminal state."
                ),
                "semantic_roles": ("agent", "co_agent", "patient_object"),
            }
        ),
    }
)

EXPECTED_ALLOWED_NODES = (
    {"holder_job_id": "141620", "node": "auh7-1b-gpu-226"},
    {"holder_job_id": "141618", "node": "auh7-1b-gpu-249"},
    {"holder_job_id": "141619", "node": "auh7-1b-gpu-257"},
)
EXPECTED_ALLOWED_OPERATIONS = (
    "frozen_bernini_vae_encode_exact16",
    "oracle_q_exact_two_row_optimizer_overfit_max10",
    "strict_checkpoint_reload_and_oracle_q_decode",
    "source_target_anchor_intervention_html_review",
)
EXPECTED_OBJECTIVE_RESTRICTIONS = MappingProxyType(
    {
        "frozen_base_velocity_reference_forbidden": True,
        "frozen_teacher_self_distillation_forbidden": True,
        "hand_tuned_reward_scalar_forbidden": True,
        "target_grounded_event_and_context_flow_only": True,
    }
)
EXPECTED_DISALLOWED_CLAIMS = MappingProxyType(
    {
        "exact160": True,
        "formal_c2": True,
        "production_model": True,
        "real_video_generalization": True,
        "scientific_promotion": True,
        "source_instruction_inference": True,
    }
)
EXPECTED_AUTHORIZATION_BASIS = MappingProxyType(
    {
        "date": "2026-08-17",
        "requester": "workspace_user",
        "requester_explicitly_directed_training_test_iteration_on_nodes_226_249_257": True,
        "requester_previously_accepted_elal3_design": True,
    }
)


def load_external_authority_v1(
    path: str | Path,
    *,
    expected_sha256: str = EXPECTED_EXTERNAL_AUTHORITY_SHA256,
) -> Mapping[str, Any]:
    """Authenticate the separately issued exact-C2 diagnostic authority."""

    if expected_sha256 != EXPECTED_EXTERNAL_AUTHORITY_SHA256:
        fail("external authority SHA literal differs")
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        fail("external authority path must be absolute")
    try:
        resolved = requested.resolve(strict=True)
        registered = EXPECTED_EXTERNAL_AUTHORITY_PATH.resolve(strict=True)
    except OSError as error:
        raise ELAL3SimulatorC2LabelError(
            "external authority is unavailable"
        ) from error
    if resolved != registered:
        fail("external authority is not the registered non-symlink file")
    payload, _ = stable_read_path(
        resolved,
        label="external C2 authority",
        expected_sha256=expected_sha256,
        expected_mode=0o644,
        allowed_root=resolved.parent,
    )
    value = _strict_json_bytes(payload, label="external C2 authority")
    expected_keys = {
        "allowed_nodes",
        "allowed_operations",
        "authority_digest",
        "authorization_basis",
        "authorized_row_ids",
        "disallowed_claims",
        "fresh_optimizer_run_required",
        "max_optimizer_updates_per_arm",
        "oracle_q_teacher_forced_required",
        "packet_manifest_sha256",
        "packet_status_preserved",
        "schema_version",
        "status",
        "supersedes_packet_training_use_forbidden_for_exact_scope_only",
        "training_objective_restrictions",
    }
    unsigned = dict(value)
    stored_digest = unsigned.pop("authority_digest", None)
    if (
        set(value) != expected_keys
        or value.get("schema_version") != EXTERNAL_AUTHORITY_SCHEMA_VERSION
        or value.get("status") != EXTERNAL_AUTHORITY_STATUS
        or value.get("packet_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or value.get("packet_status_preserved") != UPSTREAM_STATUS
        or value.get("authorized_row_ids") != list(C2_ROW_IDS)
        or value.get("allowed_nodes") != list(EXPECTED_ALLOWED_NODES)
        or value.get("allowed_operations") != list(EXPECTED_ALLOWED_OPERATIONS)
        or value.get("fresh_optimizer_run_required") is not True
        or value.get("max_optimizer_updates_per_arm") != 10
        or value.get("oracle_q_teacher_forced_required") is not True
        or value.get("supersedes_packet_training_use_forbidden_for_exact_scope_only")
        is not True
        or value.get("training_objective_restrictions")
        != dict(EXPECTED_OBJECTIVE_RESTRICTIONS)
        or value.get("disallowed_claims") != dict(EXPECTED_DISALLOWED_CLAIMS)
        or value.get("authorization_basis")
        != dict(EXPECTED_AUTHORIZATION_BASIS)
        or stored_digest != EXPECTED_EXTERNAL_AUTHORITY_DIGEST
        or object_sha256(unsigned) != stored_digest
    ):
        fail("external C2 authority closure/digest differs")
    return MappingProxyType(value)


def load_experiment_contract_v1(
    path: str | Path = EXPECTED_EXPERIMENT_CONTRACT_PATH,
    *,
    expected_sha256: str = EXPECTED_EXPERIMENT_CONTRACT_SHA256,
) -> Mapping[str, Any]:
    """Authenticate the final preregistered C2 role-binding contract."""

    if expected_sha256 != EXPECTED_EXPERIMENT_CONTRACT_SHA256:
        fail("experiment contract SHA literal differs")
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        fail("experiment contract path must be absolute")
    try:
        resolved = requested.resolve(strict=True)
        registered = EXPECTED_EXPERIMENT_CONTRACT_PATH.resolve(strict=True)
    except OSError as error:
        raise ELAL3SimulatorC2LabelError(
            "experiment contract is unavailable"
        ) from error
    if resolved != registered:
        fail("experiment contract is not the registered file")
    payload, _ = stable_read_path(
        resolved,
        label="C2 experiment contract",
        expected_sha256=expected_sha256,
        expected_mode=0o644,
        allowed_root=resolved.parent,
    )
    value = _strict_json_bytes(payload, label="C2 experiment contract")
    unsigned = dict(value)
    stored_digest = unsigned.pop("contract_digest", None)
    bindings = value.get("authority_bindings")
    boundaries = value.get("claim_boundaries")
    topology = value.get("topology")
    preregistered_gates = value.get("preregistered_gates")
    evaluation_energy_abi = (
        preregistered_gates.get("evaluation_energy_abi")
        if isinstance(preregistered_gates, Mapping)
        else None
    )
    if (
        value.get("schema_version") != EXPERIMENT_CONTRACT_SCHEMA_VERSION
        or value.get("status")
        != "PREREGISTERED_C2_SIMULATOR_ORACLE_Q_DIAGNOSTIC_ONLY"
        or value.get("packet_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or value.get("authorized_row_ids") != list(C2_ROW_IDS)
        or stored_digest != EXPECTED_EXPERIMENT_CONTRACT_DIGEST
        or object_sha256(unsigned) != stored_digest
        or not isinstance(bindings, Mapping)
        or bindings.get("derivative_authority_file_sha256")
        != EXPECTED_EXTERNAL_AUTHORITY_SHA256
        or bindings.get("derivative_authority_digest")
        != EXPECTED_EXTERNAL_AUTHORITY_DIGEST
        or not isinstance(boundaries, Mapping)
        or boundaries.get("teacher_forced_oracle_q_simulator_diagnostic_only")
        is not True
        or any(
            boundaries.get(key) is not False
            for key in (
                "exact160",
                "formal_c2",
                "production_model",
                "real_video_generalization",
                "scientific_promotion",
                "source_instruction_inference",
            )
        )
        or not isinstance(topology, Mapping)
        or topology.get("global_optimizer_updates") != 10
        or topology.get("world_size") != 8
        or topology.get("sequence_parallel_size") != 4
        or topology.get("data_parallel_size") != 2
        or not isinstance(evaluation_energy_abi, Mapping)
        or evaluation_energy_abi.get("renderer_timestep_dtype")
        != "torch.int64"
        or evaluation_energy_abi.get("renderer_timestep_value") != 999
        or evaluation_energy_abi.get("sigma_float32") != 1.0
        or evaluation_energy_abi.get("x_sigma") != "epsilon"
        or evaluation_energy_abi.get("target_velocity")
        != "epsilon-clean_latent_truth_variant"
        or evaluation_energy_abi.get("epsilon_shape")
        != [1, 16, 21, 52, 70]
    ):
        fail("C2 experiment contract closure/digest differs")
    return MappingProxyType(value)


def _validate_upstream_authority(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(UPSTREAM_AUTHORITY):
        fail(f"{label} upstream authority differs")


def _validate_rle(runs: Any, *, label: str) -> None:
    if not isinstance(runs, list):
        fail(f"{label} RLE is not a list")
    previous_stop = 0
    area = RGB_HEIGHT * RGB_WIDTH
    for run in runs:
        if (
            not isinstance(run, list)
            or len(run) != 2
            or any(type(item) is not int for item in run)
        ):
            fail(f"{label} RLE row differs")
        start, length = run
        if start < previous_stop or length <= 0 or start + length > area:
            fail(f"{label} RLE bounds/order differ")
        previous_stop = start + length


def _mask_from_runs(runs: Sequence[Sequence[int]]) -> torch.Tensor:
    flat = torch.zeros(RGB_HEIGHT * RGB_WIDTH, dtype=torch.bool)
    for start, length in runs:
        flat[int(start) : int(start) + int(length)] = True
    return flat.reshape(RGB_HEIGHT, RGB_WIDTH).contiguous()


def _validate_annotation(
    value: Mapping[str, Any], *, row_id: str, variant: str
) -> None:
    exact = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "row_id": row_id,
        "media_variant": variant,
        "entity_count": 3,
        "entity_order": list(PHYSICAL_ENTITY_ORDER),
        "fps": FPS,
        "frame_count": RGB_FRAMES,
        "coordinate_space": "this_media_native_128x96_rgb_grid",
        "simulator_gt": True,
        "status": UPSTREAM_STATUS,
        "tracker_or_estimator_used": False,
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            fail(f"{row_id}/{variant} annotation {key} differs")
    _validate_upstream_authority(
        value.get("authority"), label=f"{row_id}/{variant} annotation"
    )
    if set(value) != {
        "appearance",
        "appearance_disjoint_from_source",
        "authority",
        "camera_transform",
        "coordinate_space",
        "entity_count",
        "entity_order",
        "fps",
        "frame_count",
        "frames",
        "instance_masks",
        "media_variant",
        "phase_labels",
        "required_effect",
        "roles",
        "row_id",
        "schema_version",
        "signed_tracks",
        "simulator_gt",
        "status",
        "terminal_window_rgb_frames_inclusive",
        "tracker_or_estimator_used",
        "visibility_confidence",
    }:
        fail(f"{row_id}/{variant} annotation key closure differs")
    masks = value.get("instance_masks")
    tracks = value.get("signed_tracks")
    visibility = value.get("visibility_confidence")
    phases = value.get("phase_labels")
    camera = value.get("camera_transform")
    roles = value.get("roles")
    if (
        not isinstance(masks, Mapping)
        or masks.get("shape") != [3, 81, 96, 128]
        or not isinstance(tracks, Mapping)
        or tracks.get("dense_shape") != [3, 81, 96, 128, 2]
        or not isinstance(visibility, Mapping)
        or visibility.get("dense_shape") != [3, 81, 96, 128, 2]
        or not isinstance(phases, Mapping)
        or phases.get("shape") != [21, 4]
        or phases.get("channels") != list(PHASE_CHANNELS)
        or phases.get("latent_phase_to_rgb_frame") != list(LATENT_TO_RGB)
        or not isinstance(camera, Mapping)
        or camera.get("encoding") != "constant_identity_all_frames"
        or camera.get("matrix") != [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        or camera.get("shape") != [81, 3, 3]
        or not isinstance(roles, Mapping)
        or set(roles) != set(PHYSICAL_ENTITY_ORDER)
        or any(role not in ROLE_CODE_ORDER for role in roles.values())
        or value.get("terminal_window_rgb_frames_inclusive") != [65, 80]
    ):
        fail(f"{row_id}/{variant} annotation structural ABI differs")
    labels = phases.get("labels")
    if (
        not isinstance(labels, list)
        or len(labels) != LATENT_PHASES
        or any(
            not isinstance(row, list)
            or len(row) != 4
            or any(type(bit) is not int or bit not in (0, 1) for bit in row)
            or sum(row) > 1
            for row in labels
        )
    ):
        fail(f"{row_id}/{variant} phase labels differ")
    frames = value.get("frames")
    if not isinstance(frames, list) or len(frames) != RGB_FRAMES:
        fail(f"{row_id}/{variant} frame table differs")
    for frame_index, frame in enumerate(frames):
        entities = frame.get("entities") if isinstance(frame, Mapping) else None
        if (
            not isinstance(frame, Mapping)
            or frame.get("frame_index") != frame_index
            or not isinstance(entities, list)
            or [item.get("entity_id") for item in entities]
            != list(PHYSICAL_ENTITY_ORDER)
        ):
            fail(f"{row_id}/{variant} frame/entity order differs")
        for entity in entities:
            amodal_runs = entity.get("amodal_mask_runs")
            visible_runs = entity.get("visible_mask_runs")
            _validate_rle(amodal_runs, label="amodal")
            _validate_rle(visible_runs, label="visible")
            amodal = _mask_from_runs(amodal_runs)
            visible = _mask_from_runs(visible_runs)
            if bool((visible & ~amodal).any().item()) or not bool(amodal.any().item()):
                fail(f"{row_id}/{variant} visible/amodal support differs")
            center = entity.get("center_xy")
            motion = entity.get("signed_track_dxdy_from_previous_frame")
            if (
                not isinstance(center, list)
                or len(center) != 2
                or any(type(item) is not int for item in center)
                or not 0 <= center[0] < RGB_WIDTH
                or not 0 <= center[1] < RGB_HEIGHT
                or not isinstance(motion, list)
                or len(motion) != 2
                or any(type(item) is not int for item in motion)
            ):
                fail(f"{row_id}/{variant} entity track differs")
            for scalar_key in ("track_confidence", "visibility_fraction"):
                scalar = entity.get(scalar_key)
                if (
                    isinstance(scalar, bool)
                    or not isinstance(scalar, (int, float))
                    or not math.isfinite(float(scalar))
                    or not 0.0 <= float(scalar) <= 1.0
                ):
                    fail(f"{row_id}/{variant} {scalar_key} differs")


@dataclass(frozen=True)
class VerifiedC2RowV1:
    packet_root: Path
    manifest: Mapping[str, Any]
    row: Mapping[str, Any]
    annotations: Mapping[str, Mapping[str, Any]]
    annotation_receipts: Mapping[str, Mapping[str, Any]]
    media_paths: Mapping[str, Path]
    annotation_paths: Mapping[str, Path]
    annotation_receipt_paths: Mapping[str, Path]
    file_bindings: Mapping[str, Mapping[str, Mapping[str, Any]]]
    media_bytes: Mapping[str, bytes]

    @property
    def row_id(self) -> str:
        return str(self.row["row_id"])


@dataclass(frozen=True)
class VerifiedC2PacketV1:
    packet_root: Path
    manifest: Mapping[str, Any]
    rows: Mapping[str, VerifiedC2RowV1]


def _load_media_triple(
    root: Path,
    *,
    packet_root_fd: int,
    row_id: str,
    variant: str,
    media_entry: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    expected_base = f"{row_id}/{variant}"
    exact_paths = {
        "path": f"media/{expected_base}.mp4",
        "annotation_path": f"annotations/{expected_base}.annotations.json.gz",
        "annotation_receipt_path": (
            f"annotations/{expected_base}.annotation-receipt.json"
        ),
    }
    for key, expected in exact_paths.items():
        if media_entry.get(key) != expected:
            fail(f"{row_id}/{variant} manifest {key} differs")
    if (
        media_entry.get("variant") != variant
        or media_entry.get("simulator_gt") is not True
        or not isinstance(media_entry.get("probe"), Mapping)
        or media_entry["probe"].get("frame_count") != RGB_FRAMES
        or media_entry["probe"].get("fps_num") != FPS
        or media_entry["probe"].get("fps_den") != 1
        or media_entry["probe"].get("height") != RGB_HEIGHT
        or media_entry["probe"].get("width") != RGB_WIDTH
        or media_entry["probe"].get("pixel_format") != "yuv420p"
        or media_entry["probe"].get("all_frames_decoded_by_ffprobe") is not True
    ):
        fail(f"{row_id}/{variant} manifest media ABI differs")
    sha_fields = {
        "media": "sha256",
        "annotation": "annotation_sha256",
        "annotation_receipt": "annotation_receipt_sha256",
    }
    paths = {
        "media": _plain_file(root, exact_paths["path"], label="media"),
        "annotation": _plain_file(
            root, exact_paths["annotation_path"], label="annotation"
        ),
        "annotation_receipt": _plain_file(
            root,
            exact_paths["annotation_receipt_path"],
            label="annotation receipt",
        ),
    }
    payloads: dict[str, bytes] = {}
    stable_bindings: dict[str, Mapping[str, Any]] = {}
    for path_key, sha_key in sha_fields.items():
        expected_sha = media_entry.get(sha_key)
        if not isinstance(expected_sha, str) or _SHA256.fullmatch(expected_sha) is None:
            fail(f"{row_id}/{variant} {path_key} SHA-256 differs")
        payload, binding = stable_read_path(
            paths[path_key],
            label=f"{row_id}/{variant} {path_key}",
            expected_sha256=expected_sha,
            expected_mode=0o444,
            allowed_root=root,
            held_root_fd=packet_root_fd,
        )
        payloads[path_key] = payload
        stable_bindings[path_key] = MappingProxyType(binding)
    receipt = _canonical_json_payload(
        payloads["annotation_receipt"], label="annotation receipt"
    )
    unsigned_receipt = dict(receipt)
    stored_digest = unsigned_receipt.pop("annotation_receipt_digest", None)
    if (
        stored_digest != media_entry.get("annotation_receipt_digest")
        or object_sha256(unsigned_receipt) != stored_digest
        or receipt.get("schema_version") != ANNOTATION_RECEIPT_SCHEMA_VERSION
        or receipt.get("row_id") != row_id
        or receipt.get("media_variant") != variant
        or receipt.get("status") != UPSTREAM_STATUS
        or receipt.get("extractor")
        != "deterministic_analytic_simulator_gt_no_tracker"
        or receipt.get("extractor_version") != PACKET_SCHEMA_VERSION
    ):
        fail(f"{row_id}/{variant} annotation receipt differs")
    _validate_upstream_authority(
        receipt.get("authority"), label=f"{row_id}/{variant} receipt"
    )
    annotation_block = receipt.get("annotation")
    media_block = receipt.get("media")
    if (
        not isinstance(annotation_block, Mapping)
        or annotation_block.get("path") != exact_paths["annotation_path"]
        or annotation_block.get("sha256") != media_entry.get("annotation_sha256")
        or annotation_block.get("schema_version") != ANNOTATION_SCHEMA_VERSION
        or not isinstance(media_block, Mapping)
        or media_block.get("path") != exact_paths["path"]
        or media_block.get("sha256") != media_entry.get("sha256")
    ):
        fail(f"{row_id}/{variant} receipt media binding differs")
    try:
        decompressed = gzip.decompress(payloads["annotation"])
    except (OSError, EOFError) as error:
        raise ELAL3SimulatorC2LabelError(
            f"{row_id}/{variant} annotation gzip is invalid"
        ) from error
    annotation = _strict_json_bytes(decompressed, label="annotation")
    if (
        decompressed != canonical_json_bytes(annotation) + b"\n"
        or hashlib.sha256(decompressed).hexdigest()
        != annotation_block.get("uncompressed_canonical_json_sha256")
    ):
        fail(f"{row_id}/{variant} uncompressed annotation binding differs")
    _validate_annotation(annotation, row_id=row_id, variant=variant)
    returned_paths: dict[str, Any] = dict(paths)
    returned_paths["stable_bindings"] = MappingProxyType(stable_bindings)
    returned_paths["media_payload"] = payloads["media"]
    return (
        MappingProxyType(annotation),
        MappingProxyType(receipt),
        MappingProxyType(returned_paths),
    )


def load_verified_c2_packet(packet_root: str | Path) -> VerifiedC2PacketV1:
    """Authenticate exact2 C2 rows and all exact16 media triples."""

    requested = Path(packet_root).expanduser()
    if requested.is_symlink():
        fail("packet root cannot be a symlink")
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise ELAL3SimulatorC2LabelError("packet root is unavailable") from error
    if not root.is_dir():
        fail("packet root must be a directory")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    packet_root_fd = os.open(str(root), directory_flags)
    try:
        return _load_verified_c2_packet_from_fd(root, packet_root_fd)
    finally:
        os.close(packet_root_fd)


def _load_verified_c2_packet_from_fd(
    root: Path, packet_root_fd: int
) -> VerifiedC2PacketV1:
    manifest_path = _plain_file(root, "manifest.json", label="packet manifest")
    manifest_payload, _ = stable_read_path(
        manifest_path,
        label="packet manifest",
        expected_sha256=EXPECTED_MANIFEST_SHA256,
        expected_mode=0o444,
        allowed_root=root,
        held_root_fd=packet_root_fd,
    )
    manifest = _canonical_json_payload(
        manifest_payload, label="packet manifest"
    )
    unsigned_manifest = dict(manifest)
    stored_digest = unsigned_manifest.pop("manifest_digest", None)
    if (
        stored_digest != EXPECTED_MANIFEST_DIGEST
        or object_sha256(unsigned_manifest) != stored_digest
        or manifest.get("schema_version") != PACKET_SCHEMA_VERSION
        or manifest.get("status") != UPSTREAM_STATUS
        or manifest.get("row_count") != 3
        or manifest.get("c1_row_count") != 1
        or manifest.get("c2_row_count") != 2
        or manifest.get("media_count") != 24
        or manifest.get("frame_count") != RGB_FRAMES
        or manifest.get("fps") != FPS
        or manifest.get("latent_frame_count") != LATENT_PHASES
        or manifest.get("height") != RGB_HEIGHT
        or manifest.get("width") != RGB_WIDTH
        or manifest.get("media_order") != list(MEDIA_ORDER)
    ):
        fail("packet manifest structural ABI/digest differs")
    _validate_upstream_authority(manifest.get("authority"), label="packet manifest")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        fail("packet rows are absent")
    verified_rows: dict[str, VerifiedC2RowV1] = {}
    for row_id in C2_ROW_IDS:
        matches = [
            row
            for row in rows
            if isinstance(row, Mapping) and row.get("row_id") == row_id
        ]
        if len(matches) != 1:
            fail(f"registered C2 row is not exact-one: {row_id}")
        row = matches[0]
        metadata = EXPECTED_ROW_METADATA[row_id]
        participants = row.get("participants")
        if (
            row.get("gate") != "C2_THREE_ENTITY_ROLE_OCCLUSION"
            or row.get("formal_manifest_eligibility")
            != "diagnostic-only-not-exact160"
            or row.get("instruction") != metadata["instruction"]
            or row.get("entity_count") != 3
            or row.get("terminal_hold_rgb_frames_inclusive") != [65, 80]
            or row.get("negative_order") != list(MEDIA_ORDER[3:])
            or not isinstance(participants, list)
            or [item.get("entity_id") for item in participants]
            != list(PHYSICAL_ENTITY_ORDER)
            or [item.get("semantic_role") for item in participants]
            != list(metadata["semantic_roles"])
        ):
            fail(f"registered C2 row metadata differs: {row_id}")
        media = row.get("media")
        if not isinstance(media, Mapping) or set(media) != set(MEDIA_ORDER):
            fail(f"registered C2 media closure differs: {row_id}")
        annotations: dict[str, Mapping[str, Any]] = {}
        receipts: dict[str, Mapping[str, Any]] = {}
        media_paths: dict[str, Path] = {}
        annotation_paths: dict[str, Path] = {}
        receipt_paths: dict[str, Path] = {}
        file_bindings: dict[str, Mapping[str, Mapping[str, Any]]] = {}
        media_bytes: dict[str, bytes] = {}
        for variant in MEDIA_ORDER:
            annotation, receipt, paths = _load_media_triple(
                root,
                packet_root_fd=packet_root_fd,
                row_id=row_id,
                variant=variant,
                media_entry=media[variant],
            )
            annotations[variant] = annotation
            receipts[variant] = receipt
            media_paths[variant] = paths["media"]
            annotation_paths[variant] = paths["annotation"]
            receipt_paths[variant] = paths["annotation_receipt"]
            file_bindings[variant] = paths["stable_bindings"]
            media_bytes[variant] = paths["media_payload"]
        verified_rows[row_id] = VerifiedC2RowV1(
            packet_root=root,
            manifest=MappingProxyType(manifest),
            row=MappingProxyType(dict(row)),
            annotations=MappingProxyType(annotations),
            annotation_receipts=MappingProxyType(receipts),
            media_paths=MappingProxyType(media_paths),
            annotation_paths=MappingProxyType(annotation_paths),
            annotation_receipt_paths=MappingProxyType(receipt_paths),
            file_bindings=MappingProxyType(file_bindings),
            media_bytes=MappingProxyType(media_bytes),
        )
    return VerifiedC2PacketV1(
        packet_root=root,
        manifest=MappingProxyType(manifest),
        rows=MappingProxyType(verified_rows),
    )


def _slot_entity_ids(annotation: Mapping[str, Any]) -> tuple[str, str, str]:
    roles = annotation["roles"]
    actor = [entity for entity, role in roles.items() if role in _ACTOR_ROLES]
    objects = [entity for entity, role in roles.items() if role in _OBJECT_ROLES]
    if len(actor) != 1 or len(objects) != 1:
        fail("variant does not have exact-one semantic actor/object")
    remaining = [
        entity
        for entity in PHYSICAL_ENTITY_ORDER
        if entity not in {actor[0], objects[0]}
    ]
    if len(remaining) != 1:
        fail("variant semantic slot mapping is not bijective")
    result = (actor[0], remaining[0], objects[0])
    if len(set(result)) != 3:
        fail("variant semantic slot mapping contains duplicates")
    return result


def _entity_row(
    annotation: Mapping[str, Any], frame_index: int, entity_id: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in annotation["frames"][frame_index]["entities"]
        if row["entity_id"] == entity_id
    ]
    if len(matches) != 1:
        fail("annotation entity lookup is not exact-one")
    return matches[0]


def _mask_to_patch(mask: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    result = torch.zeros((patch_h, patch_w), dtype=torch.bool)
    coordinates = mask.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        return result
    yy = torch.div(
        coordinates[:, 0] * patch_h, RGB_HEIGHT, rounding_mode="floor"
    )
    xx = torch.div(
        coordinates[:, 1] * patch_w, RGB_WIDTH, rounding_mode="floor"
    )
    result[yy, xx] = True
    return result.contiguous()


def _phase_window(phase_index: int) -> range:
    if phase_index == 0:
        return range(0, 1)
    return range(4 * (phase_index - 1) + 1, 4 * phase_index + 1)


def _scatter_motion(
    counts: torch.Tensor,
    sums_x: torch.Tensor,
    sums_y: torch.Tensor,
    mask: torch.Tensor,
    motion: Sequence[int],
) -> None:
    patch_h, patch_w = counts.shape
    coordinates = mask.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        return
    yy = torch.div(
        coordinates[:, 0] * patch_h, RGB_HEIGHT, rounding_mode="floor"
    )
    xx = torch.div(
        coordinates[:, 1] * patch_w, RGB_WIDTH, rounding_mode="floor"
    )
    indices = yy * patch_w + xx
    weights = torch.ones(indices.numel(), dtype=torch.float32)
    minlength = patch_h * patch_w
    counts.reshape(-1).add_(
        torch.bincount(indices, weights=weights, minlength=minlength)
    )
    sums_x.reshape(-1).add_(
        torch.bincount(
            indices, weights=weights * float(motion[0]), minlength=minlength
        )
    )
    sums_y.reshape(-1).add_(
        torch.bincount(
            indices, weights=weights * float(motion[1]), minlength=minlength
        )
    )


def _tensor_receipt(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": [int(item) for item in tensor.shape],
        }
    )
    digest = hashlib.sha256(header + b"\0")
    byte_view = tensor.view(torch.uint8).reshape(-1)
    for offset in range(0, int(byte_view.numel()), 1 << 20):
        digest.update(bytes(byte_view[offset : offset + (1 << 20)].tolist()))
    return {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
        "sha256": digest.hexdigest(),
    }


def _build_cpu_oracle(
    verified: VerifiedC2RowV1,
    *,
    media_variant: str,
    patch_h: int,
    patch_w: int,
) -> tuple[
    elal3.ELAL3LatentV1,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    annotation = verified.annotations[media_variant]
    slot_ids = _slot_entity_ids(annotation)
    roles = annotation["roles"]
    q_local = torch.zeros((1, 21, patch_h, patch_w, 64), dtype=torch.float32)
    q_entity = torch.zeros((1, 3, 21, 256), dtype=torch.float32)
    q_relation = torch.zeros((1, 6, 21, 128), dtype=torch.float32)
    q_phase = torch.zeros((1, 21, 128), dtype=torch.float32)
    q_terminal = torch.zeros((1, 9, 256), dtype=torch.float32)
    q_camera = torch.zeros((1, 21, 128), dtype=torch.float32)
    role_amodal = torch.zeros(
        (1, 3, 21, patch_h, patch_w), dtype=torch.bool
    )
    role_visible = torch.zeros_like(role_amodal)
    role_event = torch.zeros_like(role_amodal)
    signed_motion = torch.zeros(
        (1, 2, 21, patch_h, patch_w), dtype=torch.float32
    )
    labels = annotation["phase_labels"]["labels"]
    cache: dict[tuple[int, str, str], torch.Tensor] = {}

    def mask(frame_index: int, entity_id: str, kind: str) -> torch.Tensor:
        key = (frame_index, entity_id, kind)
        if key not in cache:
            row = _entity_row(annotation, frame_index, entity_id)
            cache[key] = _mask_from_runs(row[f"{kind}_mask_runs"])
        return cache[key]

    grid_y = (
        (torch.arange(patch_h, dtype=torch.float32) + 0.5)
        / patch_h
        * 2.0
        - 1.0
    )[:, None].expand(patch_h, patch_w)
    grid_x = (
        (torch.arange(patch_w, dtype=torch.float32) + 0.5)
        / patch_w
        * 2.0
        - 1.0
    )[None, :].expand(patch_h, patch_w)

    for phase_index, rgb_index in enumerate(LATENT_TO_RGB):
        phase_bits = torch.tensor(labels[phase_index], dtype=torch.float32)
        q_phase[0, phase_index, :4] = phase_bits
        q_phase[0, phase_index, 4] = phase_index / 20.0
        q_phase[0, phase_index, 5] = float(not bool(phase_bits.any().item()))
        q_phase[0, phase_index, 6] = math.sin(
            2.0 * math.pi * phase_index / 20.0
        )
        q_phase[0, phase_index, 7] = math.cos(
            2.0 * math.pi * phase_index / 20.0
        )
        q_local[0, phase_index, :, :, 11:15] = phase_bits
        q_local[0, phase_index, :, :, 15] = grid_x
        q_local[0, phase_index, :, :, 16] = grid_y
        q_local[0, phase_index, :, :, 17] = phase_index / 20.0

        current_amodal: dict[str, torch.Tensor] = {}
        current_visible: dict[str, torch.Tensor] = {}
        for slot_index, entity_id in enumerate(slot_ids):
            current = _entity_row(annotation, rgb_index, entity_id)
            initial = _entity_row(annotation, 0, entity_id)
            amodal = mask(rgb_index, entity_id, "amodal")
            visible = mask(rgb_index, entity_id, "visible")
            current_amodal[entity_id] = amodal
            current_visible[entity_id] = visible
            role_amodal[0, slot_index, phase_index] = _mask_to_patch(
                amodal, patch_h, patch_w
            )
            role_visible[0, slot_index, phase_index] = _mask_to_patch(
                visible, patch_h, patch_w
            )
            q_local[0, phase_index, :, :, 3 + slot_index] = role_amodal[
                0, slot_index, phase_index
            ].float()
            q_local[0, phase_index, :, :, 6 + slot_index] = role_visible[
                0, slot_index, phase_index
            ].float()
            cx, cy = (float(item) for item in current["center_xy"])
            ix, iy = (float(item) for item in initial["center_xy"])
            dx, dy = (
                float(item)
                for item in current["signed_track_dxdy_from_previous_frame"]
            )
            features = q_entity[0, slot_index, phase_index]
            features[0] = cx / (RGB_WIDTH - 1) * 2.0 - 1.0
            features[1] = cy / (RGB_HEIGHT - 1) * 2.0 - 1.0
            features[2] = ix / (RGB_WIDTH - 1) * 2.0 - 1.0
            features[3] = iy / (RGB_HEIGHT - 1) * 2.0 - 1.0
            features[4] = (cx - ix) / RGB_WIDTH
            features[5] = (cy - iy) / RGB_HEIGHT
            features[6] = dx / RGB_WIDTH
            features[7] = dy / RGB_HEIGHT
            features[8] = float(current["visibility_fraction"])
            features[9] = float(current["track_confidence"])
            features[10] = 1.0
            features[11] = phase_index / 20.0
            features[12:16] = phase_bits
            features[16 + slot_index] = 1.0
            role_code = ROLE_CODE_ORDER.index(str(roles[entity_id]))
            features[19 + role_code] = 1.0
            features[27] = float(amodal.sum().item()) / (RGB_HEIGHT * RGB_WIDTH)
            features[28] = float(visible.sum().item()) / max(
                1.0, float(amodal.sum().item())
            )

        occluded_union = torch.zeros((RGB_HEIGHT, RGB_WIDTH), dtype=torch.bool)
        overlap_union = torch.zeros_like(occluded_union)
        for entity_id in slot_ids:
            occluded_union |= current_amodal[entity_id] & ~current_visible[entity_id]
        for left in range(3):
            for right in range(left + 1, 3):
                overlap_union |= (
                    current_amodal[slot_ids[left]]
                    & current_amodal[slot_ids[right]]
                )
        q_local[0, phase_index, :, :, 9] = _mask_to_patch(
            occluded_union, patch_h, patch_w
        ).float()
        q_local[0, phase_index, :, :, 10] = _mask_to_patch(
            overlap_union, patch_h, patch_w
        ).float()

        counts = torch.zeros((patch_h, patch_w), dtype=torch.float32)
        sums_x = torch.zeros_like(counts)
        sums_y = torch.zeros_like(counts)
        for frame_index in _phase_window(phase_index):
            for slot_index, entity_id in enumerate(slot_ids):
                entity = _entity_row(annotation, frame_index, entity_id)
                amodal = mask(frame_index, entity_id, "amodal")
                role_event[0, slot_index, phase_index] |= _mask_to_patch(
                    amodal, patch_h, patch_w
                )
                _scatter_motion(
                    counts,
                    sums_x,
                    sums_y,
                    amodal,
                    entity["signed_track_dxdy_from_previous_frame"],
                )
        denominator = counts.clamp_min(1.0)
        signed_motion[0, 0, phase_index] = sums_x / denominator
        signed_motion[0, 1, phase_index] = sums_y / denominator
        q_local[0, phase_index, :, :, 0] = (
            signed_motion[0, 0, phase_index] / RGB_WIDTH
        )
        q_local[0, phase_index, :, :, 1] = (
            signed_motion[0, 1, phase_index] / RGB_HEIGHT
        )
        q_local[0, phase_index, :, :, 2] = role_event[
            0, :, phase_index
        ].any(dim=0).float()

        for edge_index, (source_slot, target_slot) in enumerate(RELATION_EDGES):
            source_id = slot_ids[source_slot]
            target_id = slot_ids[target_slot]
            source_current = _entity_row(annotation, rgb_index, source_id)
            target_current = _entity_row(annotation, rgb_index, target_id)
            source_initial = _entity_row(annotation, 0, source_id)
            target_initial = _entity_row(annotation, 0, target_id)
            ax, ay = (float(item) for item in source_current["center_xy"])
            bx, by = (float(item) for item in target_current["center_xy"])
            iax, iay = (float(item) for item in source_initial["center_xy"])
            ibx, iby = (float(item) for item in target_initial["center_xy"])
            relation = q_relation[0, edge_index, phase_index]
            relation[0] = (bx - ax) / RGB_WIDTH
            relation[1] = (by - ay) / RGB_HEIGHT
            relation[2] = math.hypot(bx - ax, by - ay) / math.hypot(
                RGB_WIDTH, RGB_HEIGHT
            )
            relation[3] = (ibx - iax) / RGB_WIDTH
            relation[4] = (iby - iay) / RGB_HEIGHT
            relation[5] = relation[0] - relation[3]
            relation[6] = relation[1] - relation[4]
            relation[7] = float(
                bool(
                    (
                        current_amodal[source_id]
                        & current_amodal[target_id]
                    ).any().item()
                )
            )
            relation[8] = float(
                bool(
                    (
                        current_visible[source_id]
                        & current_visible[target_id]
                    ).any().item()
                )
            )
            relation[9] = source_slot / 2.0
            relation[10] = target_slot / 2.0
            relation[11:15] = phase_bits
            relation[15] = float(source_current["visibility_fraction"])
            relation[16] = float(target_current["visibility_fraction"])

    q_terminal[0, :3] = q_entity[0, :, 16:].mean(dim=1)
    q_terminal[0, 3:, :128] = q_relation[0, :, 16:].mean(dim=1)
    presence = torch.ones((1, 3), dtype=torch.bool)
    temporal_valid = torch.ones((1, 3, 21), dtype=torch.bool)
    relation_valid = torch.ones((1, 6, 21), dtype=torch.bool)
    phase_valid = torch.ones((1, 21), dtype=torch.bool)
    latent = elal3.ELAL3LatentV1(
        q_local=q_local.contiguous(),
        q_entity=q_entity.contiguous(),
        q_relation=q_relation.contiguous(),
        q_phase=q_phase.contiguous(),
        q_terminal=q_terminal.contiguous(),
        q_camera=q_camera.contiguous(),
        entity_presence=presence.contiguous(),
        temporal_valid=temporal_valid.contiguous(),
        relation_valid=relation_valid.contiguous(),
        phase_valid=phase_valid.contiguous(),
    )
    latent.validate()
    event = role_event.any(dim=1).contiguous()
    if (
        not bool(event.any().item())
        or not bool((~event).any().item())
        or not bool(torch.isfinite(signed_motion).all().item())
    ):
        fail("C2 event/context or signed motion evidence is degenerate")
    return (
        latent,
        event,
        signed_motion.contiguous(),
        role_amodal.contiguous(),
        role_visible.contiguous(),
        role_event.contiguous(),
    )


def _latent_to(
    latent: elal3.ELAL3LatentV1,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> elal3.ELAL3LatentV1:
    floats = {
        name: getattr(latent, name).to(device=device, dtype=dtype).contiguous()
        for name in (
            "q_local",
            "q_entity",
            "q_relation",
            "q_phase",
            "q_terminal",
            "q_camera",
        )
    }
    bools = {
        name: getattr(latent, name).to(device=device).contiguous()
        for name in (
            "entity_presence",
            "temporal_valid",
            "relation_valid",
            "phase_valid",
        )
    }
    result = elal3.ELAL3LatentV1(**floats, **bools)
    result.validate()
    return result


@dataclass(frozen=True)
class ELAL3SimulatorC2OracleLabelV1:
    latent: elal3.ELAL3LatentV1
    event_mask_patch: torch.Tensor
    context_mask_patch: torch.Tensor
    event_mask_vae: torch.Tensor
    context_mask_vae: torch.Tensor
    role_amodal_mask_patch: torch.Tensor
    role_visible_mask_patch: torch.Tensor
    role_event_mask_patch: torch.Tensor
    role_event_mask_vae: torch.Tensor
    signed_motion_patch: torch.Tensor
    receipt: Mapping[str, Any]
    verified_row: VerifiedC2RowV1

    @property
    def target_flow(self) -> torch.Tensor:
        """Analytic 2-D annotation motion, never diffusion velocity."""

        return self.signed_motion_patch


def load_oracle_q_label_v1(
    packet_root: str | Path,
    *,
    row_id: str,
    media_variant: str,
    patch_grid: Sequence[int],
    external_authority_path: str | Path = EXPECTED_EXTERNAL_AUTHORITY_PATH,
    external_authority_sha256: str = EXPECTED_EXTERNAL_AUTHORITY_SHA256,
    experiment_contract_path: str | Path = EXPECTED_EXPERIMENT_CONTRACT_PATH,
    experiment_contract_sha256: str = EXPECTED_EXPERIMENT_CONTRACT_SHA256,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> ELAL3SimulatorC2OracleLabelV1:
    """Return one authenticated variant-specific K3/E6 oracle label."""

    authority = load_external_authority_v1(
        external_authority_path, expected_sha256=external_authority_sha256
    )
    contract = load_experiment_contract_v1(
        experiment_contract_path,
        expected_sha256=experiment_contract_sha256,
    )
    if row_id not in C2_ROW_IDS:
        fail("row_id is not one of the registered exact2 C2 rows")
    if media_variant not in MEDIA_ORDER:
        fail("media_variant is not one of the registered exact8 variants")
    if isinstance(patch_grid, (str, bytes)):
        fail("patch_grid must be integer (21,H,W)")
    try:
        normalized = tuple(patch_grid)
    except TypeError as error:
        raise ELAL3SimulatorC2LabelError(
            "patch_grid must be integer (21,H,W)"
        ) from error
    if (
        len(normalized) != 3
        or any(type(item) is not int for item in normalized)
        or normalized[0] != LATENT_PHASES
        or normalized[1] <= 0
        or normalized[2] <= 0
    ):
        fail("patch_grid must be positive integer (21,H,W)")
    if (
        not isinstance(dtype, torch.dtype)
        or not torch.empty((), dtype=dtype).is_floating_point()
    ):
        fail("oracle dtype must be floating point")
    packet = load_verified_c2_packet(packet_root)
    verified = packet.rows[row_id]
    phases, patch_h, patch_w = normalized
    (
        cpu_latent,
        event_cpu,
        signed_cpu,
        role_amodal_cpu,
        role_visible_cpu,
        role_event_cpu,
    ) = _build_cpu_oracle(
        verified,
        media_variant=media_variant,
        patch_h=patch_h,
        patch_w=patch_w,
    )
    slot_ids = _slot_entity_ids(verified.annotations[media_variant])
    q_rows = {
        name: _tensor_receipt(getattr(cpu_latent, name))
        for name in (
            "q_local",
            "q_entity",
            "q_relation",
            "q_phase",
            "q_terminal",
            "q_camera",
            "entity_presence",
            "temporal_valid",
            "relation_valid",
            "phase_valid",
        )
    }
    mask_rows = {
        "event_mask_patch": _tensor_receipt(event_cpu),
        "role_amodal_mask_patch": _tensor_receipt(role_amodal_cpu),
        "role_visible_mask_patch": _tensor_receipt(role_visible_cpu),
        "role_event_mask_patch": _tensor_receipt(role_event_cpu),
        "signed_motion_patch": _tensor_receipt(signed_cpu),
    }
    media_entry = verified.row["media"][media_variant]
    unsigned_receipt = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "status": "ELAL3_SIMULATOR_C2_ORACLE_Q_LABEL_READY",
        "row_id": row_id,
        "media_variant": media_variant,
        "representation_variant": "full",
        "attention_width": 64,
        "teacher_forced_oracle_q": True,
        "external_optimizer_authority_verified": True,
        "action_encoder_qualified": False,
        "action_predictor_present": False,
        "source_instruction_inference_authorized": False,
        "real_video_data": False,
        "formal_c2_authorized": False,
        "scientific_claim_authorized": False,
        "exact160_claim_authorized": False,
        "source_packet": {
            "manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
            "manifest_digest": EXPECTED_MANIFEST_DIGEST,
            "row_id": row_id,
            "oracle_media_variant": media_variant,
            "media_sha256": media_entry["sha256"],
            "annotation_sha256": media_entry["annotation_sha256"],
        },
        "external_authority_binding": {
            "relative_path": EXPECTED_EXTERNAL_AUTHORITY_RELATIVE_PATH,
            "file_sha256": EXPECTED_EXTERNAL_AUTHORITY_SHA256,
            "authority_digest": authority["authority_digest"],
            "schema_version": authority["schema_version"],
            "authorized_row_ids": list(authority["authorized_row_ids"]),
            "max_optimizer_updates_per_arm": 10,
        },
        "experiment_contract_binding": {
            "relative_path": EXPECTED_EXPERIMENT_CONTRACT_RELATIVE_PATH,
            "file_sha256": EXPECTED_EXPERIMENT_CONTRACT_SHA256,
            "contract_digest": contract["contract_digest"],
            "schema_version": contract["schema_version"],
            "authorized_row_ids": list(contract["authorized_row_ids"]),
            "variant_order": list(MEDIA_ORDER),
            "renderer_timestep_dtype": "torch.int64",
            "renderer_timestep_value": 999,
            "sigma_float32": 1.0,
            "x_sigma": "epsilon",
            "target_velocity": "epsilon-clean_latent_truth_variant",
        },
        "patch_grid": [phases, patch_h, patch_w],
        "slot_entity_ids": list(slot_ids),
        "slot_roles": [
            verified.annotations[media_variant]["roles"][entity_id]
            for entity_id in slot_ids
        ],
        "physical_annotation_roles": dict(
            verified.annotations[media_variant]["roles"]
        ),
        "semantic_role_code_order_fixed_across_variants": True,
        "role_code_order": list(ROLE_CODE_ORDER),
        "relation_edges": [list(edge) for edge in RELATION_EDGES],
        "q_local_channel_layout": {
            "signed_motion_xy": [0, 2],
            "event_union": [2, 3],
            "role_amodal_slot0_2": [3, 6],
            "role_visible_slot0_2": [6, 9],
            "occluded_union": [9, 10],
            "pair_overlap_union": [10, 11],
            "phase_bits": [11, 15],
            "grid_x": [15, 16],
            "grid_y": [16, 17],
            "normalized_time": [17, 18],
            "reserved_zero": [18, 64],
        },
        "q_entity_channel_layout": {
            "current_center_xy": [0, 2],
            "initial_center_xy": [2, 4],
            "displacement_xy": [4, 6],
            "signed_track_xy": [6, 8],
            "visibility_confidence_presence_time": [8, 12],
            "phase_bits": [12, 16],
            "slot_onehot": [16, 19],
            "role_code_onehot": [19, 27],
            "amodal_area_visible_ratio": [27, 29],
            "reserved_zero": [29, 256],
        },
        "q_relation_channel_layout": {
            "current_directed_delta_xy_distance": [0, 3],
            "initial_directed_delta_xy": [3, 5],
            "directed_delta_change_xy": [5, 7],
            "amodal_visible_overlap": [7, 9],
            "endpoint_slot_codes": [9, 11],
            "phase_bits": [11, 15],
            "endpoint_visibility": [15, 17],
            "reserved_zero": [17, 128],
        },
        "coordinate_mapping": {
            "annotation_grid": [RGB_HEIGHT, RGB_WIDTH],
            "cross_media_coordinates_compared": False,
            "phase_to_rgb_frame": list(LATENT_TO_RGB),
            "swept_window": "t0={0};t>0={4(t-1)+1,...,4t}",
            "patch_to_vae_repeat": [2, 2],
        },
        "mask_tensor_rows": mask_rows,
        "role_event_union_equals_event": True,
        "q_tensor_rows": q_rows,
        "q_tensor_rows_digest": object_sha256(q_rows),
        "validity": {
            "entity_slots": 3,
            "directed_relation_slots": 6,
            "phases": 21,
            "present_entity_indices": [0, 1, 2],
            "valid_relation_indices": [0, 1, 2, 3, 4, 5],
        },
    }
    receipt = {
        **unsigned_receipt,
        "label_digest": object_sha256(unsigned_receipt),
    }
    target_device = torch.device("cpu") if device is None else torch.device(device)
    latent = _latent_to(cpu_latent, device=target_device, dtype=dtype)
    event = event_cpu.to(device=target_device).contiguous()
    context = (~event).contiguous()
    event_vae = (
        event[:, None]
        .repeat_interleave(2, dim=3)
        .repeat_interleave(2, dim=4)
        .contiguous()
    )
    context_vae = (~event_vae).contiguous()
    role_amodal = role_amodal_cpu.to(device=target_device).contiguous()
    role_visible = role_visible_cpu.to(device=target_device).contiguous()
    role_event = role_event_cpu.to(device=target_device).contiguous()
    role_event_vae = (
        role_event.repeat_interleave(2, dim=3)
        .repeat_interleave(2, dim=4)
        .contiguous()
    )
    signed = signed_cpu.to(device=target_device, dtype=dtype).contiguous()
    if not torch.equal(role_event_vae.any(dim=1), event_vae[:, 0]):
        fail("role-event VAE union differs from event mask")
    for value in (
        event,
        context,
        event_vae,
        context_vae,
        role_amodal,
        role_visible,
        role_event,
        role_event_vae,
        signed,
    ):
        if value.requires_grad or value.grad_fn is not None:
            fail("oracle evidence must be detached")
    return ELAL3SimulatorC2OracleLabelV1(
        latent=latent,
        event_mask_patch=event,
        context_mask_patch=context,
        event_mask_vae=event_vae,
        context_mask_vae=context_vae,
        role_amodal_mask_patch=role_amodal,
        role_visible_mask_patch=role_visible,
        role_event_mask_patch=role_event,
        role_event_mask_vae=role_event_vae,
        signed_motion_patch=signed,
        receipt=MappingProxyType(receipt),
        verified_row=verified,
    )


def _tensor_bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and bool(
            torch.equal(
                left.detach().contiguous().view(torch.uint8),
                right.detach().contiguous().view(torch.uint8),
            )
        )
    )


def role_only_slot_swap_v1(
    latent: elal3.ELAL3LatentV1,
) -> elal3.ELAL3LatentV1:
    """Swap only slots 0/1 in q_entity and directed q_relation.

    q_local, q_phase, q_terminal, q_camera and every validity tensor are kept
    byte-identical.  Role-code one-hots travel with their q_entity rows;
    q_relation endpoint-slot code channels are rewritten to match the new
    directed edge positions.  This is the preregistered role-only hybrid, not
    the broader C0 intervention helper.
    """

    if not isinstance(latent, elal3.ELAL3LatentV1):
        fail("role-only swap requires ELAL3LatentV1")
    latent.validate()
    permutation = torch.tensor(
        [1, 0, 2], dtype=torch.int64, device=latent.q_entity.device
    )
    entity = latent.q_entity.index_select(1, permutation).clone().contiguous()
    entity[:, :, :, 16:19] = 0.0
    for slot_index in range(3):
        entity[:, slot_index, :, 16 + slot_index] = 1.0
    edge_lookup = {edge: index for index, edge in enumerate(RELATION_EDGES)}
    relation_order = torch.tensor(
        [
            edge_lookup[(int(permutation[source]), int(permutation[target]))]
            for source, target in RELATION_EDGES
        ],
        dtype=torch.int64,
        device=latent.q_relation.device,
    )
    relation = latent.q_relation.index_select(1, relation_order).clone().contiguous()
    for edge_index, (source, target) in enumerate(RELATION_EDGES):
        relation[:, edge_index, :, 9] = source / 2.0
        relation[:, edge_index, :, 10] = target / 2.0
    result = elal3.ELAL3LatentV1(
        q_local=latent.q_local,
        q_entity=entity,
        q_relation=relation,
        q_phase=latent.q_phase,
        q_terminal=latent.q_terminal,
        q_camera=latent.q_camera,
        entity_presence=latent.entity_presence,
        temporal_valid=latent.temporal_valid,
        relation_valid=latent.relation_valid,
        phase_valid=latent.phase_valid,
    )
    result.validate()
    for name in (
        "q_local",
        "q_phase",
        "q_terminal",
        "q_camera",
        "entity_presence",
        "temporal_valid",
        "relation_valid",
        "phase_valid",
    ):
        if not _tensor_bits_equal(getattr(result, name), getattr(latent, name)):
            fail(f"role-only swap changed fixed field: {name}")
    return result


@dataclass(frozen=True)
class ELAL3RoleOnlySlotSwapV1:
    latent: elal3.ELAL3LatentV1
    receipt: Mapping[str, Any]


def build_role_only_slot_swap_v1(
    label: ELAL3SimulatorC2OracleLabelV1,
) -> ELAL3RoleOnlySlotSwapV1:
    """Build the role-only hybrid plus a byte-auditable receipt."""

    if not isinstance(label, ELAL3SimulatorC2OracleLabelV1):
        fail("role-only hybrid requires an authenticated C2 oracle label")
    swapped = role_only_slot_swap_v1(label.latent)
    fixed = (
        "q_local",
        "q_phase",
        "q_terminal",
        "q_camera",
        "entity_presence",
        "temporal_valid",
        "relation_valid",
        "phase_valid",
    )
    fixed_rows = {
        name: {
            "source": _tensor_receipt(getattr(label.latent, name)),
            "hybrid": _tensor_receipt(getattr(swapped, name)),
            "bit_identical": _tensor_bits_equal(
                getattr(label.latent, name), getattr(swapped, name)
            ),
        }
        for name in fixed
    }
    if not all(row["bit_identical"] for row in fixed_rows.values()):
        fail("role-only hybrid fixed-field receipt differs")
    unsigned = {
        "schema_version": "elal3-c2-role-only-slot-swap-v1",
        "status": "ELAL3_C2_ROLE_ONLY_HYBRID_READY",
        "source_label_digest": label.receipt["label_digest"],
        "row_id": label.receipt["row_id"],
        "media_variant": label.receipt["media_variant"],
        "new_slot_to_old_slot": [1, 0, 2],
        "relation_edges": [list(edge) for edge in RELATION_EDGES],
        "new_edge_to_old_edge_index": [2, 3, 0, 1, 5, 4],
        "q_entity_role_code_channels": [19, 27],
        "q_entity_endpoint_slot_channels": [16, 19],
        "q_relation_endpoint_slot_channels": [9, 11],
        "semantic_role_code_order_fixed": True,
        "fixed_tensor_rows": fixed_rows,
        "swapped_tensor_rows": {
            "q_entity_source": _tensor_receipt(label.latent.q_entity),
            "q_entity_hybrid": _tensor_receipt(swapped.q_entity),
            "q_relation_source": _tensor_receipt(label.latent.q_relation),
            "q_relation_hybrid": _tensor_receipt(swapped.q_relation),
        },
        "only_q_entity_and_q_relation_changed": True,
        "spatial_masks_external_and_fixed": True,
        "teacher_forced_oracle_q_simulator_diagnostic_only": True,
        "formal_c2_authorized": False,
        "source_instruction_inference_authorized": False,
    }
    receipt = {**unsigned, "hybrid_digest": object_sha256(unsigned)}
    return ELAL3RoleOnlySlotSwapV1(
        latent=swapped, receipt=MappingProxyType(receipt)
    )


@dataclass(frozen=True)
class ELAL3RoleOnlyHybridV1:
    latent: elal3.ELAL3LatentV1
    event_mask_patch: torch.Tensor
    context_mask_patch: torch.Tensor
    event_mask_vae: torch.Tensor
    context_mask_vae: torch.Tensor
    role_amodal_mask_patch: torch.Tensor
    role_visible_mask_patch: torch.Tensor
    role_event_mask_patch: torch.Tensor
    role_event_mask_vae: torch.Tensor
    receipt: Mapping[str, Any]


def build_role_only_hybrid_v1(
    matched: ELAL3SimulatorC2OracleLabelV1,
    opposite: ELAL3SimulatorC2OracleLabelV1,
) -> ELAL3RoleOnlyHybridV1:
    """Use opposite q_entity/q_relation while freezing every matched field.

    This exactly implements one preregistered role-only mismatch cell.  The
    caller supplies target/role_swap in either direction; spatial masks and
    the eight fixed latent fields always come from ``matched``.
    """

    if not isinstance(matched, ELAL3SimulatorC2OracleLabelV1) or not isinstance(
        opposite, ELAL3SimulatorC2OracleLabelV1
    ):
        fail("role-only hybrid requires two authenticated C2 labels")
    matched_row = matched.receipt.get("row_id")
    opposite_row = opposite.receipt.get("row_id")
    matched_variant = matched.receipt.get("media_variant")
    opposite_variant = opposite.receipt.get("media_variant")
    if (
        matched_row != opposite_row
        or {matched_variant, opposite_variant} != {"target", "role_swap"}
        or matched.latent.q_local.device != opposite.latent.q_local.device
        or matched.latent.q_local.dtype != opposite.latent.q_local.dtype
        or matched.receipt.get("patch_grid") != opposite.receipt.get("patch_grid")
    ):
        fail("role-only hybrid row/variant/device/dtype/grid pairing differs")
    result = elal3.ELAL3LatentV1(
        q_local=matched.latent.q_local,
        q_entity=opposite.latent.q_entity,
        q_relation=opposite.latent.q_relation,
        q_phase=matched.latent.q_phase,
        q_terminal=matched.latent.q_terminal,
        q_camera=matched.latent.q_camera,
        entity_presence=matched.latent.entity_presence,
        temporal_valid=matched.latent.temporal_valid,
        relation_valid=matched.latent.relation_valid,
        phase_valid=matched.latent.phase_valid,
    )
    result.validate()
    fixed_names = (
        "q_local",
        "q_phase",
        "q_terminal",
        "q_camera",
        "entity_presence",
        "temporal_valid",
        "relation_valid",
        "phase_valid",
    )
    fixed_proof = {
        name: {
            "matched": _tensor_receipt(getattr(matched.latent, name)),
            "result": _tensor_receipt(getattr(result, name)),
            "bit_identical": _tensor_bits_equal(
                getattr(matched.latent, name), getattr(result, name)
            ),
        }
        for name in fixed_names
    }
    swapped_proof = {
        name: {
            "opposite": _tensor_receipt(getattr(opposite.latent, name)),
            "result": _tensor_receipt(getattr(result, name)),
            "bit_identical": _tensor_bits_equal(
                getattr(opposite.latent, name), getattr(result, name)
            ),
        }
        for name in ("q_entity", "q_relation")
    }
    spatial_masks = {
        name: {
            "matched": _tensor_receipt(getattr(matched, name)),
            "result_source": "matched",
            "bit_identical": True,
        }
        for name in (
            "event_mask_patch",
            "context_mask_patch",
            "event_mask_vae",
            "context_mask_vae",
            "role_amodal_mask_patch",
            "role_visible_mask_patch",
            "role_event_mask_patch",
            "role_event_mask_vae",
        )
    }
    if not all(row["bit_identical"] for row in fixed_proof.values()) or not all(
        row["bit_identical"] for row in swapped_proof.values()
    ):
        fail("role-only hybrid tensor proof differs")
    unsigned = {
        "schema_version": "elal3-c2-role-only-opposite-hybrid-v1",
        "status": "ELAL3_C2_ROLE_ONLY_OPPOSITE_HYBRID_READY",
        "row_id": matched_row,
        "matched_variant": matched_variant,
        "opposite_variant": opposite_variant,
        "matched_label_digest": matched.receipt["label_digest"],
        "opposite_label_digest": opposite.receipt["label_digest"],
        "matched_slot_entity_ids": list(matched.receipt["slot_entity_ids"]),
        "matched_slot_roles": list(matched.receipt["slot_roles"]),
        "opposite_slot_entity_ids": list(opposite.receipt["slot_entity_ids"]),
        "opposite_slot_roles": list(opposite.receipt["slot_roles"]),
        "q_entity_role_code_channels": [19, 27],
        "q_relation_endpoint_slot_channels": [9, 11],
        "role_code_slice_rows": {
            "matched_q_entity": _tensor_receipt(
                matched.latent.q_entity[..., 19:27].contiguous()
            ),
            "opposite_q_entity": _tensor_receipt(
                opposite.latent.q_entity[..., 19:27].contiguous()
            ),
            "result_q_entity": _tensor_receipt(
                result.q_entity[..., 19:27].contiguous()
            ),
            "matched_q_relation_endpoint": _tensor_receipt(
                matched.latent.q_relation[..., 9:11].contiguous()
            ),
            "opposite_q_relation_endpoint": _tensor_receipt(
                opposite.latent.q_relation[..., 9:11].contiguous()
            ),
            "result_q_relation_endpoint": _tensor_receipt(
                result.q_relation[..., 9:11].contiguous()
            ),
        },
        "fixed_tensor_proof": fixed_proof,
        "opposite_tensor_proof": swapped_proof,
        "spatial_mask_proof": spatial_masks,
        "experiment_contract_binding": dict(
            matched.receipt["experiment_contract_binding"]
        ),
        "only_q_entity_and_q_relation_from_opposite": True,
        "all_eight_latent_fixed_fields_from_matched": True,
        "all_spatial_masks_from_matched": True,
        "semantic_role_code_order_fixed": True,
        "teacher_forced_oracle_q_simulator_diagnostic_only": True,
        "formal_c2_authorized": False,
        "source_instruction_inference_authorized": False,
    }
    receipt = {**unsigned, "hybrid_digest": object_sha256(unsigned)}
    return ELAL3RoleOnlyHybridV1(
        latent=result,
        event_mask_patch=matched.event_mask_patch,
        context_mask_patch=matched.context_mask_patch,
        event_mask_vae=matched.event_mask_vae,
        context_mask_vae=matched.context_mask_vae,
        role_amodal_mask_patch=matched.role_amodal_mask_patch,
        role_visible_mask_patch=matched.role_visible_mask_patch,
        role_event_mask_patch=matched.role_event_mask_patch,
        role_event_mask_vae=matched.role_event_mask_vae,
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "C2_ROW_IDS",
    "ELAL3SimulatorC2LabelError",
    "ELAL3SimulatorC2OracleLabelV1",
    "ELAL3RoleOnlySlotSwapV1",
    "ELAL3RoleOnlyHybridV1",
    "EXPECTED_EXTERNAL_AUTHORITY_PATH",
    "EXPECTED_EXTERNAL_AUTHORITY_SHA256",
    "EXPECTED_MANIFEST_SHA256",
    "LABEL_SCHEMA_VERSION",
    "MEDIA_ORDER",
    "RELATION_EDGES",
    "ROLE_CODE_ORDER",
    "VerifiedC2PacketV1",
    "VerifiedC2RowV1",
    "canonical_json_bytes",
    "load_external_authority_v1",
    "load_experiment_contract_v1",
    "load_oracle_q_label_v1",
    "load_verified_c2_packet",
    "object_sha256",
    "build_role_only_slot_swap_v1",
    "build_role_only_hybrid_v1",
    "role_only_slot_swap_v1",
]
