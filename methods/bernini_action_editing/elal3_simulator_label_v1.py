#!/usr/bin/env python3
"""Strict simulator-GT to ELAL-3 oracle-label bridge for the C1 canary.

This module is deliberately narrow.  It accepts exactly the checked-in
``c1-two-entity-push-to-goal`` simulator row, authenticates every one of its
eight media/annotation/receipt triples, and converts the *target* analytic
annotation into a deterministic ELAL-3 label.  The result is privileged,
teacher-forced simulator evidence.  It is not ``E_video``, an ActionPredictor,
source+instruction inference, real-video training data, or exact160 evidence.

The module cannot derive training authority from the simulator packet.  It
can publish a create-only, one-row, full-w64 optimizer envelope only after it
authenticates the separately issued, locally pinned diagnostic authority.
The original simulator packet remains immutable and keeps its original
``training_use_forbidden`` authority.
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


LABEL_SCHEMA_VERSION = "elal3-simulator-oracle-q-label-v1"
AUTHORITY_SCHEMA_VERSION = (
    "elal3-c1-simulator-oracle-q-optimizer-authority-v1"
)
AUTHORITY_STATUS = (
    "ELAL3_SIMULATOR_ORACLE_Q_OPTIMIZER_DIAGNOSTIC_AUTHORIZED"
)
EXPECTED_ROW_ID = "c1-two-entity-push-to-goal"
EXPECTED_MANIFEST_SHA256 = (
    "2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"
)
EXPECTED_MANIFEST_DIGEST = (
    "1bc3b7cc155b25028eeab1e940cf6e6ead2c4c0ff189a4f8059f0a8928a383bd"
)
EXPECTED_EXTERNAL_AUTHORITY_RELATIVE_PATH = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c1_simulator_optimizer_diagnostic_authority_v1.json"
)
EXPECTED_EXTERNAL_AUTHORITY_PATH = (
    Path(__file__).resolve().parents[2]
    / EXPECTED_EXTERNAL_AUTHORITY_RELATIVE_PATH
)
EXPECTED_EXTERNAL_AUTHORITY_SHA256 = (
    "298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b"
)
EXPECTED_EXTERNAL_AUTHORITY_SCHEMA_VERSION = (
    "bernini-elal3-simulator-optimizer-derivative-authority-v1"
)
EXPECTED_EXTERNAL_AUTHORITY_DIGEST = (
    "c1706ee5b3f8a3fa4c037dfa6dbdbc7d0b088d3682128e50e712e311dae35043"
)
EXPECTED_EXTERNAL_AUTHORITY_STATUS = (
    "AUTHORIZED_SIMULATOR_ORACLE_Q_DIAGNOSTIC_ONLY"
)
EXPECTED_INSTRUCTION = (
    "The designated red agent pushes the blue patient object into the green "
    "goal, then both hold the completed state."
)
EXPECTED_MEDIA_ORDER = (
    "source",
    "target",
    "anchor",
    "wrong_agent",
    "wrong_object",
    "role_swap",
    "reverse",
    "phase_shuffle",
)
ANNOTATION_SCHEMA_VERSION = "elal3-simulator-media-annotation-v1"
ANNOTATION_RECEIPT_SCHEMA_VERSION = (
    "elal3-simulator-annotation-receipt-v1"
)
UPSTREAM_STATUS = "ELAL3_SIM_DIAGNOSTIC"
RGB_FRAMES = 81
FPS = 25
RGB_HEIGHT = 96
RGB_WIDTH = 128
LATENT_PHASES = 21
LATENT_TO_RGB = tuple(range(0, RGB_FRAMES, 4))
ENTITY_ORDER = ("agent", "patient")
CAUSAL_PARTICIPANTS = frozenset(ENTITY_ORDER)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ELAL3SimulatorLabelError(RuntimeError):
    """Raised before accepting an ambiguous simulator label or authority."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ELAL3SimulatorLabelError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ELAL3SimulatorLabelError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ELAL3SimulatorLabelError(f"non-finite JSON number: {value}")


def _strict_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except ELAL3SimulatorLabelError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ELAL3SimulatorLabelError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ELAL3SimulatorLabelError(f"{label} must contain one JSON object")
    return value


def _canonical_json_file(path: Path, *, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    value = _strict_json_bytes(payload, label=label)
    if payload != canonical_json_bytes(value) + b"\n":
        raise ELAL3SimulatorLabelError(f"{label} bytes are not canonical JSON+newline")
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

EXPECTED_EXTERNAL_OBJECTIVE_RESTRICTIONS = MappingProxyType(
    {
        "frozen_base_velocity_reference_forbidden": True,
        "frozen_teacher_self_distillation_forbidden": True,
        "hand_tuned_reward_scalar_forbidden": True,
        "target_grounded_event_and_context_flow_only": True,
    }
)
EXPECTED_EXTERNAL_DISALLOWED_CLAIMS = MappingProxyType(
    {
        "exact160": True,
        "formal_c1": True,
        "production_model": True,
        "real_video_generalization": True,
        "scientific_promotion": True,
        "source_instruction_inference": True,
    }
)
EXPECTED_EXTERNAL_ALLOWED_OPERATIONS = (
    "frozen_bernini_vae_encode",
    "real_bernini_no_update_integration_probe",
    "oracle_q_exact_one_row_optimizer_overfit",
    "strict_checkpoint_reload_and_oracle_q_decode",
    "source_target_anchor_intervention_html_review",
)
EXPECTED_EXTERNAL_ALLOWED_NODES = (
    {"holder_job_id": "141620", "node": "auh7-1b-gpu-226"},
    {"holder_job_id": "141618", "node": "auh7-1b-gpu-249"},
    {"holder_job_id": "141619", "node": "auh7-1b-gpu-257"},
)
EXPECTED_EXTERNAL_AUTHORIZATION_BASIS = MappingProxyType(
    {
        "date": "2026-08-17",
        "requester": "workspace_user",
        "requester_explicitly_directed_training_on_nodes_226_249_257": True,
        "requester_previously_accepted_elal3_design": True,
    }
)


def _media_pin(
    role: str,
    media_sha256: str,
    annotation_sha256: str,
    receipt_sha256: str,
    receipt_digest: str,
) -> Mapping[str, str]:
    base = f"c1-two-entity-push-to-goal/{role}"
    return MappingProxyType(
        {
            "path": f"media/{base}.mp4",
            "sha256": media_sha256,
            "annotation_path": f"annotations/{base}.annotations.json.gz",
            "annotation_sha256": annotation_sha256,
            "annotation_receipt_path": f"annotations/{base}.annotation-receipt.json",
            "annotation_receipt_sha256": receipt_sha256,
            "annotation_receipt_digest": receipt_digest,
        }
    )


EXPECTED_MEDIA_PINS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "source": _media_pin(
            "source",
            "7a7b324931d46f12dec1796f4c15f75eb60921622d13c70187d8d7f1764d13c0",
            "a292fd96c9c5e6eee9743aa0ac369d56527aa88ec99a45edfa8c5b1d15a1226e",
            "bc9d1ca4f2f895ee0eacb1333b4c8d00c174405b43467f651169dbce0c0305f1",
            "016ed042f857ffcd3ac8eea728d56233ae436a513f62f25bf25470e36a88b105",
        ),
        "target": _media_pin(
            "target",
            "83d4439c7bdd803d04e38a6571ccc0aa633cdbb8df717ee475611afbce75c31a",
            "0861ea00ba7413aebe7b924ba26b706a94a420835138cc9d24481d627980de49",
            "f343838672a9fa6f8c9ec1d62aaf0e74b6951793a599677c776a2c530c270dd5",
            "51401f5a49b3e88c861a67f25b5d626fc349e5c0f8ffe6756d7bd15313d21429",
        ),
        "anchor": _media_pin(
            "anchor",
            "afc98b4e9bb4419854e5e01a598a3f75b5342346f7d9d1a98c0e9b6571e391ac",
            "5b9663590b2a38028462d61bf88c27b3c8b832136bffad6722a3e5d90a1cb094",
            "ded026bfa86ff8ffd25b7d95d83c60d5cc92261ddb5fc3a4376c50ee239f4f91",
            "5a459406d4824adcc9c39fe57b97e6050a95314efc3b88826b86ed750849c891",
        ),
        "wrong_agent": _media_pin(
            "wrong_agent",
            "e2c884117fb11a442e6ad646d65b5e5fe7622d1f76c52344188aec82cd02c616",
            "9304a2b1e5d28fdedbf6cb74d7473896307d86c8eb4f6d9917802ddae46f6af0",
            "7f8b66b2b94daa4ee12d8426faae11354a77f6062488f0e6eb7c577608a647c4",
            "6c5c7542f88b076f91a561d9285a1746133455109fed8f767ec388edc43a07a0",
        ),
        "wrong_object": _media_pin(
            "wrong_object",
            "a8af410610804efed0e052706f833caa5ab634c63bde0b79bc1fbb8f7ec44586",
            "92a68a3c7f74559d478930c039c1896f7a043b0bf7deebdb121aeb6a6c6bd811",
            "4d28e3620e4a3ce244dd29e664ae34b1e893894408406a3464fa0a7086b04d8d",
            "7eaa9d2d3d7f16c6f725479212699553024ca1f0ae44bfc343ec01c4eb15e99c",
        ),
        "role_swap": _media_pin(
            "role_swap",
            "2516a3596ec31e791958e8c9640ad27112a91d58b88f084b49e1de147e2d479d",
            "cd33c641649d98ba8044edd4fa740a66c43d739ef0cdad0a35c92a318b9ae96f",
            "98c55179fa5f2fcc106f720f1963bddfa594f6196ab1f79be902ea4ec23c5119",
            "8cddfe15e1f308e324c480ac76f6894ff7fda457405c9ffe8583106750d86386",
        ),
        "reverse": _media_pin(
            "reverse",
            "732def2a79c9e6eb4ff81924aef665ff89bc882b122cdd3c1460c1b8bd22803c",
            "4d5f61bf80226ceaec3c6c7cbce1e2edb6b67d79638c3ce97060b98b25416752",
            "004c0390bf0712640f38e7a36ed90122cbb7b51b3823a5c458034013779ba7d0",
            "6a6706feeb240cc155ae9d1238a52618f1940ecc18514a0e5480a47ba78acbb0",
        ),
        "phase_shuffle": _media_pin(
            "phase_shuffle",
            "61a7a6ec171e930e21819f5b102b8998f87b9d9886c8ff7d044481a773e1c6d0",
            "e6480b35506cf31af26507106f840c01a6eb90146f0384911ac0a8e591f4f93f",
            "821783417e008ec20e027fc03721b3811b980157826846fb31c370b8fe73082a",
            "a8e6bbc0ea6a32b8f6608687cb5d65e663940cfba401ff1edce5d0a9cf44a11f",
        ),
    }
)


def _pins_plain() -> dict[str, dict[str, str]]:
    return {
        role: dict(EXPECTED_MEDIA_PINS[role])
        for role in EXPECTED_MEDIA_ORDER
    }


EXPECTED_MEDIA_PINS_DIGEST = object_sha256(_pins_plain())


def _plain_file(root: Path, relative: str, *, label: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or str(posix) != relative:
        raise ELAL3SimulatorLabelError(f"{label} relative path differs")
    path = root.joinpath(*posix.parts)
    try:
        info = path.lstat()
    except OSError as error:
        raise ELAL3SimulatorLabelError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ELAL3SimulatorLabelError(f"{label} must be a non-symlink plain file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ELAL3SimulatorLabelError(f"{label} escapes packet root") from error
    return resolved


def _require_sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ELAL3SimulatorLabelError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_upstream_authority(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(UPSTREAM_AUTHORITY):
        raise ELAL3SimulatorLabelError(f"{label} authority differs")


def _load_external_optimizer_authority_v1(
    external_authority_path: str | Path,
    *,
    external_authority_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate the separately issued, exact-scope optimizer authority."""

    if (
        type(external_authority_sha256) is not str
        or external_authority_sha256 != EXPECTED_EXTERNAL_AUTHORITY_SHA256
    ):
        raise ELAL3SimulatorLabelError(
            "external authority SHA literal differs"
        )
    try:
        requested = Path(external_authority_path).expanduser()
    except TypeError as error:
        raise ELAL3SimulatorLabelError(
            "external authority path is required"
        ) from error
    if not requested.is_absolute():
        raise ELAL3SimulatorLabelError(
            "external authority path must be absolute"
        )
    try:
        info = requested.lstat()
    except OSError as error:
        raise ELAL3SimulatorLabelError(
            "external authority file is unavailable"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ELAL3SimulatorLabelError(
            "external authority must be a non-symlink plain file"
        )
    try:
        resolved = requested.resolve(strict=True)
        registered = EXPECTED_EXTERNAL_AUTHORITY_PATH.resolve(strict=True)
    except OSError as error:
        raise ELAL3SimulatorLabelError(
            "registered external authority is unavailable"
        ) from error
    if resolved != registered:
        raise ELAL3SimulatorLabelError(
            "external authority path is not the registered local file"
        )
    try:
        authority_bytes = resolved.read_bytes()
    except OSError as error:
        raise ELAL3SimulatorLabelError(
            "external authority file cannot be read"
        ) from error
    if (
        hashlib.sha256(authority_bytes).hexdigest()
        != EXPECTED_EXTERNAL_AUTHORITY_SHA256
    ):
        raise ELAL3SimulatorLabelError("external authority file SHA-256 differs")
    value = _strict_json_bytes(
        authority_bytes, label="external optimizer authority"
    )
    expected_top_keys = {
        "allowed_nodes",
        "allowed_operations",
        "authority_digest",
        "authorization_basis",
        "authorized_row_id",
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
    if set(value) != expected_top_keys:
        raise ELAL3SimulatorLabelError("external authority key closure differs")
    unsigned = dict(value)
    stored_digest = unsigned.pop("authority_digest", None)
    if (
        stored_digest != EXPECTED_EXTERNAL_AUTHORITY_DIGEST
        or object_sha256(unsigned) != stored_digest
    ):
        raise ELAL3SimulatorLabelError("external authority object digest differs")
    exact_scalars = {
        "schema_version": EXPECTED_EXTERNAL_AUTHORITY_SCHEMA_VERSION,
        "status": EXPECTED_EXTERNAL_AUTHORITY_STATUS,
        "authorized_row_id": EXPECTED_ROW_ID,
        "packet_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "packet_status_preserved": UPSTREAM_STATUS,
        "max_optimizer_updates_per_arm": 20,
        "fresh_optimizer_run_required": True,
        "oracle_q_teacher_forced_required": True,
        "supersedes_packet_training_use_forbidden_for_exact_scope_only": True,
    }
    for key, expected in exact_scalars.items():
        if value.get(key) != expected or type(value.get(key)) is not type(expected):
            raise ELAL3SimulatorLabelError(
                f"external authority {key} differs"
            )
    if (
        value.get("allowed_nodes") != list(EXPECTED_EXTERNAL_ALLOWED_NODES)
        or value.get("allowed_operations")
        != list(EXPECTED_EXTERNAL_ALLOWED_OPERATIONS)
        or value.get("authorization_basis")
        != dict(EXPECTED_EXTERNAL_AUTHORIZATION_BASIS)
        or value.get("disallowed_claims")
        != dict(EXPECTED_EXTERNAL_DISALLOWED_CLAIMS)
        or value.get("training_objective_restrictions")
        != dict(EXPECTED_EXTERNAL_OBJECTIVE_RESTRICTIONS)
    ):
        raise ELAL3SimulatorLabelError(
            "external authority exact scope/restrictions differ"
        )
    return MappingProxyType(value)


def _external_authority_binding(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "relative_path": EXPECTED_EXTERNAL_AUTHORITY_RELATIVE_PATH,
        "file_sha256": EXPECTED_EXTERNAL_AUTHORITY_SHA256,
        "schema_version": authority["schema_version"],
        "object_digest": authority["authority_digest"],
        "status": authority["status"],
        "authorized_row_id": authority["authorized_row_id"],
        "max_optimizer_updates_per_arm": authority[
            "max_optimizer_updates_per_arm"
        ],
        "training_objective_restrictions": dict(
            authority["training_objective_restrictions"]
        ),
    }


def _validate_rle(runs: Any, *, label: str) -> None:
    if not isinstance(runs, list):
        raise ELAL3SimulatorLabelError(f"{label} RLE is not a list")
    previous_stop = 0
    area = RGB_HEIGHT * RGB_WIDTH
    for index, run in enumerate(runs):
        if (
            not isinstance(run, list)
            or len(run) != 2
            or any(type(item) is not int for item in run)
        ):
            raise ELAL3SimulatorLabelError(f"{label} RLE row differs")
        start, length = run
        if start < previous_stop or length <= 0 or start + length > area:
            raise ELAL3SimulatorLabelError(f"{label} RLE bounds/order differ")
        previous_stop = start + length


def _validate_annotation(value: Mapping[str, Any], *, role: str) -> None:
    exact = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "row_id": EXPECTED_ROW_ID,
        "media_variant": role,
        "entity_count": 2,
        "entity_order": list(ENTITY_ORDER),
        "fps": FPS,
        "frame_count": RGB_FRAMES,
        "coordinate_space": "this_media_native_128x96_rgb_grid",
        "simulator_gt": True,
        "status": UPSTREAM_STATUS,
        "tracker_or_estimator_used": False,
    }
    for key, wanted in exact.items():
        if value.get(key) != wanted:
            raise ELAL3SimulatorLabelError(f"{role} annotation {key} differs")
    _validate_upstream_authority(value.get("authority"), label=f"{role} annotation")
    masks = value.get("instance_masks")
    tracks = value.get("signed_tracks")
    visibility = value.get("visibility_confidence")
    phases = value.get("phase_labels")
    if (
        not isinstance(masks, Mapping)
        or masks.get("shape") != [2, 81, 96, 128]
        or not isinstance(tracks, Mapping)
        or tracks.get("dense_shape") != [2, 81, 96, 128, 2]
        or not isinstance(visibility, Mapping)
        or visibility.get("dense_shape") != [2, 81, 96, 128, 2]
        or not isinstance(phases, Mapping)
        or phases.get("shape") != [21, 4]
        or phases.get("latent_phase_to_rgb_frame") != list(LATENT_TO_RGB)
        or phases.get("channels") != ["onset", "transition", "terminal", "hold"]
    ):
        raise ELAL3SimulatorLabelError(f"{role} annotation structural ABI differs")
    labels = phases.get("labels")
    if (
        not isinstance(labels, list)
        or len(labels) != LATENT_PHASES
        or any(
            not isinstance(row, list)
            or len(row) != 4
            or any(bit not in (0, 1) or type(bit) is not int for bit in row)
            for row in labels
        )
    ):
        raise ELAL3SimulatorLabelError(f"{role} phase labels differ")
    frames = value.get("frames")
    if not isinstance(frames, list) or len(frames) != RGB_FRAMES:
        raise ELAL3SimulatorLabelError(f"{role} frame table differs")
    for frame_index, frame in enumerate(frames):
        entities = frame.get("entities") if isinstance(frame, Mapping) else None
        if (
            frame.get("frame_index") != frame_index
            or not isinstance(entities, list)
            or [item.get("entity_id") for item in entities] != list(ENTITY_ORDER)
        ):
            raise ELAL3SimulatorLabelError(f"{role} frame/entity order differs")
        for entity in entities:
            _validate_rle(
                entity.get("amodal_mask_runs"),
                label=f"{role} frame {frame_index} amodal",
            )
            _validate_rle(
                entity.get("visible_mask_runs"),
                label=f"{role} frame {frame_index} visible",
            )
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
                raise ELAL3SimulatorLabelError(f"{role} entity track differs")
            for scalar_key in ("track_confidence", "visibility_fraction"):
                scalar = entity.get(scalar_key)
                if (
                    isinstance(scalar, bool)
                    or not isinstance(scalar, (int, float))
                    or not math.isfinite(float(scalar))
                    or not 0.0 <= float(scalar) <= 1.0
                ):
                    raise ELAL3SimulatorLabelError(
                        f"{role} entity {scalar_key} differs"
                    )


def _load_annotation(
    root: Path,
    *,
    role: str,
    manifest_media: Mapping[str, Any],
    pin: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    for key, wanted in pin.items():
        if manifest_media.get(key) != wanted:
            raise ELAL3SimulatorLabelError(f"manifest {role} {key} differs")
    paths = {
        "media": _plain_file(root, pin["path"], label=f"{role} media"),
        "annotation": _plain_file(
            root, pin["annotation_path"], label=f"{role} annotation"
        ),
        "annotation_receipt": _plain_file(
            root,
            pin["annotation_receipt_path"],
            label=f"{role} annotation receipt",
        ),
    }
    for path_key, sha_key in (
        ("media", "sha256"),
        ("annotation", "annotation_sha256"),
        ("annotation_receipt", "annotation_receipt_sha256"),
    ):
        if file_sha256(paths[path_key]) != pin[sha_key]:
            raise ELAL3SimulatorLabelError(f"{role} {path_key} SHA-256 differs")
    receipt = _canonical_json_file(
        paths["annotation_receipt"], label=f"{role} annotation receipt"
    )
    unsigned_receipt = dict(receipt)
    stored_digest = unsigned_receipt.pop("annotation_receipt_digest", None)
    if stored_digest != pin["annotation_receipt_digest"] or (
        object_sha256(unsigned_receipt) != stored_digest
    ):
        raise ELAL3SimulatorLabelError(f"{role} annotation receipt digest differs")
    if (
        receipt.get("schema_version") != ANNOTATION_RECEIPT_SCHEMA_VERSION
        or receipt.get("row_id") != EXPECTED_ROW_ID
        or receipt.get("media_variant") != role
        or receipt.get("status") != UPSTREAM_STATUS
        or receipt.get("extractor")
        != "deterministic_analytic_simulator_gt_no_tracker"
        or receipt.get("extractor_version") != "elal3-simulator-gt-canary-v1"
    ):
        raise ELAL3SimulatorLabelError(f"{role} annotation receipt ABI differs")
    _validate_upstream_authority(receipt.get("authority"), label=f"{role} receipt")
    annotation_block = receipt.get("annotation")
    media_block = receipt.get("media")
    if (
        not isinstance(annotation_block, Mapping)
        or annotation_block.get("path") != pin["annotation_path"]
        or annotation_block.get("sha256") != pin["annotation_sha256"]
        or annotation_block.get("schema_version") != ANNOTATION_SCHEMA_VERSION
        or not isinstance(media_block, Mapping)
        or media_block.get("path") != pin["path"]
        or media_block.get("sha256") != pin["sha256"]
    ):
        raise ELAL3SimulatorLabelError(f"{role} receipt media binding differs")
    try:
        decompressed = gzip.decompress(paths["annotation"].read_bytes())
    except (OSError, EOFError) as error:
        raise ELAL3SimulatorLabelError(f"{role} annotation gzip is invalid") from error
    annotation = _strict_json_bytes(decompressed, label=f"{role} annotation")
    if decompressed != canonical_json_bytes(annotation) + b"\n":
        raise ELAL3SimulatorLabelError(f"{role} annotation is not canonical JSON+newline")
    if hashlib.sha256(decompressed).hexdigest() != annotation_block.get(
        "uncompressed_canonical_json_sha256"
    ):
        raise ELAL3SimulatorLabelError(f"{role} uncompressed annotation SHA differs")
    _validate_annotation(annotation, role=role)
    return annotation, receipt, paths


@dataclass(frozen=True)
class VerifiedC1RowV1:
    packet_root: Path
    manifest: Mapping[str, Any]
    row: Mapping[str, Any]
    annotations: Mapping[str, Mapping[str, Any]]
    annotation_receipts: Mapping[str, Mapping[str, Any]]
    media_paths: Mapping[str, Path]
    annotation_paths: Mapping[str, Path]
    annotation_receipt_paths: Mapping[str, Path]

    @property
    def row_id(self) -> str:
        return EXPECTED_ROW_ID


def load_verified_c1_row(
    packet_root: str | Path, *, row_id: str = EXPECTED_ROW_ID
) -> VerifiedC1RowV1:
    """Authenticate the exact checked-in C1 row and all eight media triples."""

    if row_id != EXPECTED_ROW_ID:
        raise ELAL3SimulatorLabelError("only the registered C1 row is accepted")
    requested = Path(packet_root).expanduser()
    if requested.is_symlink():
        raise ELAL3SimulatorLabelError("packet root cannot be a symlink")
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise ELAL3SimulatorLabelError("packet root is unavailable") from error
    if not root.is_dir():
        raise ELAL3SimulatorLabelError("packet root must be a directory")
    manifest_path = _plain_file(root, "manifest.json", label="packet manifest")
    if file_sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ELAL3SimulatorLabelError("packet manifest SHA-256 differs")
    manifest = _canonical_json_file(manifest_path, label="packet manifest")
    unsigned_manifest = dict(manifest)
    stored_manifest_digest = unsigned_manifest.pop("manifest_digest", None)
    if stored_manifest_digest != EXPECTED_MANIFEST_DIGEST or (
        object_sha256(unsigned_manifest) != stored_manifest_digest
    ):
        raise ELAL3SimulatorLabelError("packet manifest digest differs")
    _validate_upstream_authority(manifest.get("authority"), label="packet manifest")
    if (
        manifest.get("schema_version") != "elal3-simulator-gt-canary-v1"
        or manifest.get("row_count") != 3
        or manifest.get("c1_row_count") != 1
        or manifest.get("c2_row_count") != 2
        or manifest.get("media_count") != 24
        or manifest.get("frame_count") != RGB_FRAMES
        or manifest.get("fps") != FPS
        or manifest.get("latent_frame_count") != LATENT_PHASES
        or manifest.get("height") != RGB_HEIGHT
        or manifest.get("width") != RGB_WIDTH
        or tuple(manifest.get("media_order", ())) != EXPECTED_MEDIA_ORDER
    ):
        raise ELAL3SimulatorLabelError("packet manifest structural ABI differs")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ELAL3SimulatorLabelError("packet rows are absent")
    matches = [row for row in rows if row.get("row_id") == EXPECTED_ROW_ID]
    if len(matches) != 1:
        raise ELAL3SimulatorLabelError("registered C1 row is not exact-one")
    row = matches[0]
    if (
        row.get("gate") != "C1_TWO_ENTITY_ONE_ROW_OVERFIT"
        or row.get("formal_manifest_eligibility")
        != "diagnostic-only-not-exact160"
        or row.get("instruction") != EXPECTED_INSTRUCTION
        or row.get("entity_count") != 2
        or row.get("terminal_hold_rgb_frames_inclusive") != [65, 80]
        or row.get("negative_order")
        != ["wrong_agent", "wrong_object", "role_swap", "reverse", "phase_shuffle"]
    ):
        raise ELAL3SimulatorLabelError("registered C1 row metadata differs")
    participants = row.get("participants")
    if (
        not isinstance(participants, list)
        or [item.get("entity_id") for item in participants] != list(ENTITY_ORDER)
        or [item.get("semantic_role") for item in participants]
        != ["agent", "patient_object"]
    ):
        raise ELAL3SimulatorLabelError("registered causal participants differ")
    media = row.get("media")
    if not isinstance(media, Mapping) or set(media) != set(EXPECTED_MEDIA_ORDER):
        raise ELAL3SimulatorLabelError("registered C1 media closure differs")
    annotations: dict[str, Mapping[str, Any]] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    media_paths: dict[str, Path] = {}
    annotation_paths: dict[str, Path] = {}
    receipt_paths: dict[str, Path] = {}
    for role in EXPECTED_MEDIA_ORDER:
        annotation, receipt, paths = _load_annotation(
            root,
            role=role,
            manifest_media=media[role],
            pin=EXPECTED_MEDIA_PINS[role],
        )
        annotations[role] = annotation
        receipts[role] = receipt
        media_paths[role] = paths["media"]
        annotation_paths[role] = paths["annotation"]
        receipt_paths[role] = paths["annotation_receipt"]
    return VerifiedC1RowV1(
        packet_root=root,
        manifest=MappingProxyType(manifest),
        row=MappingProxyType(row),
        annotations=MappingProxyType(annotations),
        annotation_receipts=MappingProxyType(receipts),
        media_paths=MappingProxyType(media_paths),
        annotation_paths=MappingProxyType(annotation_paths),
        annotation_receipt_paths=MappingProxyType(receipt_paths),
    )


def _mask_from_runs(runs: Sequence[Sequence[int]]) -> torch.Tensor:
    flat = torch.zeros(RGB_HEIGHT * RGB_WIDTH, dtype=torch.bool)
    for start, length in runs:
        flat[int(start) : int(start) + int(length)] = True
    return flat.reshape(RGB_HEIGHT, RGB_WIDTH).contiguous()


def _mask_to_patch(mask: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    result = torch.zeros((patch_h, patch_w), dtype=torch.bool)
    coordinates = mask.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        return result
    yy = torch.div(coordinates[:, 0] * patch_h, RGB_HEIGHT, rounding_mode="floor")
    xx = torch.div(coordinates[:, 1] * patch_w, RGB_WIDTH, rounding_mode="floor")
    result[yy, xx] = True
    return result.contiguous()


def _entity_row(annotation: Mapping[str, Any], frame_index: int, entity_id: str) -> Mapping[str, Any]:
    rows = annotation["frames"][frame_index]["entities"]
    matches = [row for row in rows if row["entity_id"] == entity_id]
    if len(matches) != 1:
        raise ELAL3SimulatorLabelError("annotation entity lookup is not exact-one")
    return matches[0]


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
    yy = torch.div(coordinates[:, 0] * patch_h, RGB_HEIGHT, rounding_mode="floor")
    xx = torch.div(coordinates[:, 1] * patch_w, RGB_WIDTH, rounding_mode="floor")
    flat_indices = yy * patch_w + xx
    weights = torch.ones(flat_indices.numel(), dtype=torch.float32)
    counts.reshape(-1).add_(torch.bincount(flat_indices, weights=weights, minlength=patch_h * patch_w))
    sums_x.reshape(-1).add_(
        torch.bincount(
            flat_indices,
            weights=weights * float(motion[0]),
            minlength=patch_h * patch_w,
        )
    )
    sums_y.reshape(-1).add_(
        torch.bincount(
            flat_indices,
            weights=weights * float(motion[1]),
            minlength=patch_h * patch_w,
        )
    )


def _tensor_receipt(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = canonical_json_bytes(
        {"dtype": str(tensor.dtype), "shape": [int(item) for item in tensor.shape]}
    )
    digest.update(metadata)
    digest.update(b"\0")
    # Do not depend on NumPy: cluster images intentionally vary in NumPy ABI,
    # while these receipts must stay identical.  Chunking also bounds the
    # temporary Python list used to expose raw bytes.
    byte_view = tensor.view(torch.uint8).reshape(-1)
    chunk_bytes = 1 << 20
    for offset in range(0, int(byte_view.numel()), chunk_bytes):
        digest.update(bytes(byte_view[offset : offset + chunk_bytes].tolist()))
    return {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
        "sha256": digest.hexdigest(),
    }


def _build_cpu_oracle(
    verified: VerifiedC1RowV1, *, patch_h: int, patch_w: int
) -> tuple[elal3.ELAL3LatentV1, torch.Tensor, torch.Tensor]:
    target = verified.annotations["target"]
    source = verified.annotations["source"]
    q_local = torch.zeros((1, 21, patch_h, patch_w, 64), dtype=torch.float32)
    q_entity = torch.zeros((1, 3, 21, 256), dtype=torch.float32)
    q_relation = torch.zeros((1, 6, 21, 128), dtype=torch.float32)
    q_phase = torch.zeros((1, 21, 128), dtype=torch.float32)
    q_terminal = torch.zeros((1, 9, 256), dtype=torch.float32)
    q_camera = torch.zeros((1, 21, 128), dtype=torch.float32)
    event = torch.zeros((1, 21, patch_h, patch_w), dtype=torch.bool)
    signed_motion = torch.zeros((1, 2, 21, patch_h, patch_w), dtype=torch.float32)
    target_labels = target["phase_labels"]["labels"]
    mask_cache: dict[tuple[int, str, str], torch.Tensor] = {}

    def mask(frame_index: int, entity_id: str, kind: str) -> torch.Tensor:
        key = (frame_index, entity_id, kind)
        if key not in mask_cache:
            row = _entity_row(target, frame_index, entity_id)
            mask_cache[key] = _mask_from_runs(row[f"{kind}_mask_runs"])
        return mask_cache[key]

    grid_y = (
        (torch.arange(patch_h, dtype=torch.float32) + 0.5) / patch_h * 2.0 - 1.0
    )[:, None].expand(patch_h, patch_w)
    grid_x = (
        (torch.arange(patch_w, dtype=torch.float32) + 0.5) / patch_w * 2.0 - 1.0
    )[None, :].expand(patch_h, patch_w)

    for phase_index, rgb_index in enumerate(LATENT_TO_RGB):
        phase_bits = torch.tensor(target_labels[phase_index], dtype=torch.float32)
        q_phase[0, phase_index, :4] = phase_bits
        q_phase[0, phase_index, 4] = phase_index / 20.0
        q_phase[0, phase_index, 5] = float(not bool(phase_bits.any().item()))
        q_phase[0, phase_index, 6] = math.sin(2.0 * math.pi * phase_index / 20.0)
        q_phase[0, phase_index, 7] = math.cos(2.0 * math.pi * phase_index / 20.0)
        q_local[0, phase_index, :, :, 8:12] = phase_bits
        q_local[0, phase_index, :, :, 12] = grid_x
        q_local[0, phase_index, :, :, 13] = grid_y
        q_local[0, phase_index, :, :, 14] = phase_index / 20.0

        current_masks: dict[str, torch.Tensor] = {}
        current_visible: dict[str, torch.Tensor] = {}
        for entity_index, entity_id in enumerate(ENTITY_ORDER):
            target_row = _entity_row(target, rgb_index, entity_id)
            source_row = _entity_row(source, rgb_index, entity_id)
            current_masks[entity_id] = mask(rgb_index, entity_id, "amodal")
            current_visible[entity_id] = mask(rgb_index, entity_id, "visible")
            amodal_patch = _mask_to_patch(
                current_masks[entity_id], patch_h, patch_w
            ).float()
            visible_patch = _mask_to_patch(
                current_visible[entity_id], patch_h, patch_w
            ).float()
            q_local[0, phase_index, :, :, 3 + entity_index] = amodal_patch
            q_local[0, phase_index, :, :, 5 + entity_index] = visible_patch
            tx, ty = (float(item) for item in target_row["center_xy"])
            sx, sy = (float(item) for item in source_row["center_xy"])
            dx, dy = (
                float(item)
                for item in target_row["signed_track_dxdy_from_previous_frame"]
            )
            features = q_entity[0, entity_index, phase_index]
            features[0] = tx / (RGB_WIDTH - 1) * 2.0 - 1.0
            features[1] = ty / (RGB_HEIGHT - 1) * 2.0 - 1.0
            features[2] = sx / (RGB_WIDTH - 1) * 2.0 - 1.0
            features[3] = sy / (RGB_HEIGHT - 1) * 2.0 - 1.0
            features[4] = (tx - sx) / RGB_WIDTH
            features[5] = (ty - sy) / RGB_HEIGHT
            features[6] = dx / RGB_WIDTH
            features[7] = dy / RGB_HEIGHT
            features[8] = float(target_row["visibility_fraction"])
            features[9] = float(target_row["track_confidence"])
            features[10] = 1.0
            features[11] = phase_index / 20.0
            features[12:16] = phase_bits

        overlap = current_masks["agent"] & current_masks["patient"]
        q_local[0, phase_index, :, :, 7] = _mask_to_patch(
            overlap, patch_h, patch_w
        ).float()
        counts = torch.zeros((patch_h, patch_w), dtype=torch.float32)
        sums_x = torch.zeros_like(counts)
        sums_y = torch.zeros_like(counts)
        swept = torch.zeros((patch_h, patch_w), dtype=torch.bool)
        for frame_index in _phase_window(phase_index):
            for entity_id in ENTITY_ORDER:
                entity_row = _entity_row(target, frame_index, entity_id)
                amodal = mask(frame_index, entity_id, "amodal")
                swept |= _mask_to_patch(amodal, patch_h, patch_w)
                _scatter_motion(
                    counts,
                    sums_x,
                    sums_y,
                    amodal,
                    entity_row["signed_track_dxdy_from_previous_frame"],
                )
        event[0, phase_index] = swept
        denominator = counts.clamp_min(1.0)
        signed_motion[0, 0, phase_index] = sums_x / denominator
        signed_motion[0, 1, phase_index] = sums_y / denominator
        q_local[0, phase_index, :, :, 0] = (
            signed_motion[0, 0, phase_index] / RGB_WIDTH
        )
        q_local[0, phase_index, :, :, 1] = (
            signed_motion[0, 1, phase_index] / RGB_HEIGHT
        )
        q_local[0, phase_index, :, :, 2] = swept.float()

        for edge_index, (source_index, target_index) in enumerate(
            elal3.RELATION_EDGES
        ):
            if source_index >= 2 or target_index >= 2:
                continue
            source_id = ENTITY_ORDER[source_index]
            target_id = ENTITY_ORDER[target_index]
            source_target = _entity_row(target, rgb_index, source_id)
            target_target = _entity_row(target, rgb_index, target_id)
            source_source = _entity_row(source, rgb_index, source_id)
            target_source = _entity_row(source, rgb_index, target_id)
            ax, ay = (float(item) for item in source_target["center_xy"])
            bx, by = (float(item) for item in target_target["center_xy"])
            sax, say = (float(item) for item in source_source["center_xy"])
            sbx, sby = (float(item) for item in target_source["center_xy"])
            relation = q_relation[0, edge_index, phase_index]
            relation[0] = (bx - ax) / RGB_WIDTH
            relation[1] = (by - ay) / RGB_HEIGHT
            relation[2] = math.hypot(bx - ax, by - ay) / math.hypot(
                RGB_WIDTH, RGB_HEIGHT
            )
            relation[3] = (sbx - sax) / RGB_WIDTH
            relation[4] = (sby - say) / RGB_HEIGHT
            relation[5] = relation[0] - relation[3]
            relation[6] = relation[1] - relation[4]
            relation[7] = float(
                bool((current_masks[source_id] & current_masks[target_id]).any())
            )
            relation[8] = float(source_index)
            relation[9] = float(target_index)
            relation[10:14] = phase_bits

    q_terminal[0, :3] = q_entity[0, :, 16:].mean(dim=1)
    q_terminal[0, 3:, :128] = q_relation[0, :, 16:].mean(dim=1)
    presence = torch.tensor([[True, True, False]], dtype=torch.bool)
    temporal = presence[:, :, None].expand(1, 3, 21).clone().contiguous()
    relation_valid = torch.zeros((1, 6, 21), dtype=torch.bool)
    relation_valid[:, 0, :] = True
    relation_valid[:, 2, :] = True
    phase_valid = torch.ones((1, 21), dtype=torch.bool)
    latent = elal3.ELAL3LatentV1(
        q_local=q_local.contiguous(),
        q_entity=q_entity.contiguous(),
        q_relation=q_relation.contiguous(),
        q_phase=q_phase.contiguous(),
        q_terminal=q_terminal.contiguous(),
        q_camera=q_camera.contiguous(),
        entity_presence=presence.contiguous(),
        temporal_valid=temporal,
        relation_valid=relation_valid.contiguous(),
        phase_valid=phase_valid.contiguous(),
    )
    latent.validate()
    if not bool(event.any()) or not bool((~event).any()):
        raise ELAL3SimulatorLabelError("event/context partition is degenerate")
    if not bool(torch.isfinite(signed_motion).all()):
        raise ELAL3SimulatorLabelError("signed simulator motion is non-finite")
    return latent, event.contiguous(), signed_motion.contiguous()


def _latent_to(
    latent: elal3.ELAL3LatentV1, *, device: torch.device, dtype: torch.dtype
) -> elal3.ELAL3LatentV1:
    kwargs: dict[str, torch.Tensor] = {}
    for name in (
        "q_local",
        "q_entity",
        "q_relation",
        "q_phase",
        "q_terminal",
        "q_camera",
    ):
        kwargs[name] = getattr(latent, name).to(device=device, dtype=dtype).contiguous()
    for name in (
        "entity_presence",
        "temporal_valid",
        "relation_valid",
        "phase_valid",
    ):
        kwargs[name] = getattr(latent, name).to(device=device).contiguous()
    result = elal3.ELAL3LatentV1(**kwargs)
    result.validate()
    return result


@dataclass(frozen=True)
class ELAL3SimulatorOracleLabelV1:
    latent: elal3.ELAL3LatentV1
    event_mask_patch: torch.Tensor
    context_mask_patch: torch.Tensor
    event_mask_vae: torch.Tensor
    context_mask_vae: torch.Tensor
    signed_motion_patch: torch.Tensor
    receipt: Mapping[str, Any]
    verified_row: VerifiedC1RowV1

    @property
    def target_flow(self) -> torch.Tensor:
        """Alias for analytic 2-D motion evidence, never diffusion velocity."""

        return self.signed_motion_patch


def _tensor_bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    return bool(
        torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def load_oracle_q_label_v1(
    packet_root: str | Path,
    *,
    row_id: str = EXPECTED_ROW_ID,
    patch_grid: Sequence[int],
    external_authority_path: str | Path,
    external_authority_sha256: str,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> ELAL3SimulatorOracleLabelV1:
    """Load the exact target-derived full-w64 oracle label.

    Label issuance requires the separately issued optimizer-diagnostic
    authority; the simulator packet's own ``training_use_forbidden`` flag is
    never treated as training permission.

    ``target_flow``/``signed_motion_patch`` is two-channel simulator annotation
    evidence.  It must never replace Bernini's VAE diffusion velocity target.
    """

    external_authority = _load_external_optimizer_authority_v1(
        external_authority_path,
        external_authority_sha256=external_authority_sha256,
    )
    if isinstance(patch_grid, (str, bytes)):
        raise ELAL3SimulatorLabelError("patch_grid must be integer (21,H,W)")
    try:
        normalized_grid = tuple(patch_grid)
    except TypeError as error:
        raise ELAL3SimulatorLabelError(
            "patch_grid must be integer (21,H,W)"
        ) from error
    if len(normalized_grid) != 3 or any(
        type(item) is not int for item in normalized_grid
    ):
        raise ELAL3SimulatorLabelError("patch_grid must be integer (21,H,W)")
    phases, patch_h, patch_w = normalized_grid
    if phases != LATENT_PHASES or patch_h <= 0 or patch_w <= 0:
        raise ELAL3SimulatorLabelError("patch_grid must be positive (21,H,W)")
    if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
        raise ELAL3SimulatorLabelError("oracle dtype must be floating point")
    target_device = torch.device("cpu") if device is None else torch.device(device)
    verified = load_verified_c1_row(packet_root, row_id=row_id)
    cpu_latent, event_cpu, signed_cpu = _build_cpu_oracle(
        verified, patch_h=patch_h, patch_w=patch_w
    )
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
    event_receipt = _tensor_receipt(event_cpu)
    signed_receipt = _tensor_receipt(signed_cpu)
    unsigned_receipt = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "status": "ELAL3_SIMULATOR_ORACLE_Q_LABEL_READY",
        "row_id": EXPECTED_ROW_ID,
        "representation_variant": "full",
        "attention_width": 64,
        "teacher_forced_oracle_q": True,
        "external_optimizer_authority_verified": True,
        "action_encoder_qualified": False,
        "action_predictor_present": False,
        "source_instruction_inference_authorized": False,
        "real_video_data": False,
        "scientific_claim_authorized": False,
        "exact160_claim_authorized": False,
        "source_packet": {
            "manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
            "manifest_digest": EXPECTED_MANIFEST_DIGEST,
            "media_pins_digest": EXPECTED_MEDIA_PINS_DIGEST,
            "row_id": EXPECTED_ROW_ID,
            "oracle_media_variant": "target",
        },
        "external_authority_binding": _external_authority_binding(
            external_authority
        ),
        "patch_grid": [phases, patch_h, patch_w],
        "coordinate_mapping": {
            "source_annotation_grid": [RGB_HEIGHT, RGB_WIDTH],
            "destination": "bernini_wan_patch_grid_after_vae8_and_patch2",
            "pixel_to_patch": "floor(y*Hp/96),floor(x*Wp/128)",
            "phase_to_rgb_frame": list(LATENT_TO_RGB),
            "swept_window": "t0={0};t>0={4(t-1)+1,...,4t}",
            "dilation": 0,
        },
        "event_mask": {
            **event_receipt,
            "nonzero_cells": int(event_cpu.sum().item()),
            "total_cells": int(event_cpu.numel()),
            "causal_participants": list(ENTITY_ORDER),
            "target_derived": True,
            "detached": True,
        },
        "signed_motion_patch": {
            **signed_receipt,
            "meaning": "simulator_2d_annotation_motion_not_bernini_diffusion_velocity",
            "optimizer_diffusion_target_authorized": False,
        },
        "q_tensor_rows": q_rows,
        "q_tensor_rows_digest": object_sha256(q_rows),
        "validity": {
            "entity_slots": 3,
            "directed_relation_slots": 6,
            "phases": 21,
            "present_entity_indices": [0, 1],
            "valid_relation_indices": [0, 2],
        },
    }
    receipt = {
        **unsigned_receipt,
        "label_digest": object_sha256(unsigned_receipt),
    }
    latent = _latent_to(cpu_latent, device=target_device, dtype=dtype)
    event = event_cpu.to(device=target_device).contiguous()
    context = (~event).contiguous()
    event_vae = event[:, None].repeat_interleave(2, dim=3).repeat_interleave(2, dim=4).contiguous()
    context_vae = (~event_vae).contiguous()
    signed = signed_cpu.to(device=target_device, dtype=dtype).contiguous()
    if (
        event.requires_grad
        or context.requires_grad
        or event_vae.requires_grad
        or context_vae.requires_grad
        or signed.requires_grad
    ):
        raise ELAL3SimulatorLabelError("oracle evidence must be detached")
    return ELAL3SimulatorOracleLabelV1(
        latent=latent,
        event_mask_patch=event,
        context_mask_patch=context,
        event_mask_vae=event_vae,
        context_mask_vae=context_vae,
        signed_motion_patch=signed,
        receipt=MappingProxyType(receipt),
        verified_row=verified,
    )


def build_derivative_authority_v1(
    label: ELAL3SimulatorOracleLabelV1,
    *,
    external_authority_path: str | Path,
    external_authority_sha256: str,
) -> dict[str, Any]:
    """Bind an authenticated label to separate exact-scope optimizer authority."""

    if not isinstance(label, ELAL3SimulatorOracleLabelV1):
        raise ELAL3SimulatorLabelError("authority requires an authenticated oracle label")
    external_authority = _load_external_optimizer_authority_v1(
        external_authority_path,
        external_authority_sha256=external_authority_sha256,
    )
    try:
        label.latent.validate()
    except Exception as error:
        raise ELAL3SimulatorLabelError(
            "authority oracle latent is invalid"
        ) from error
    label_digest = label.receipt.get("label_digest")
    _require_sha(label_digest, label="oracle label digest")
    unsigned_label_receipt = dict(label.receipt)
    unsigned_label_receipt.pop("label_digest", None)
    if object_sha256(unsigned_label_receipt) != label_digest:
        raise ELAL3SimulatorLabelError("oracle label receipt digest differs")
    patch_grid = label.receipt.get("patch_grid")
    if (
        not isinstance(patch_grid, list)
        or len(patch_grid) != 3
        or any(type(item) is not int for item in patch_grid)
    ):
        raise ELAL3SimulatorLabelError("oracle label patch grid differs")
    expected = load_oracle_q_label_v1(
        label.verified_row.packet_root,
        row_id=label.verified_row.row_id,
        patch_grid=patch_grid,
        external_authority_path=external_authority_path,
        external_authority_sha256=external_authority_sha256,
        device=label.latent.q_local.device,
        dtype=label.latent.q_local.dtype,
    )
    tensor_pairs = [
        (getattr(label.latent, name), getattr(expected.latent, name))
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
    ]
    tensor_pairs.extend(
        (
            (label.event_mask_patch, expected.event_mask_patch),
            (label.context_mask_patch, expected.context_mask_patch),
            (label.event_mask_vae, expected.event_mask_vae),
            (label.context_mask_vae, expected.context_mask_vae),
            (label.signed_motion_patch, expected.signed_motion_patch),
        )
    )
    if (
        dict(label.receipt) != dict(expected.receipt)
        or any(
            left.requires_grad
            or left.grad_fn is not None
            or not _tensor_bits_equal(left, right)
            for left, right in tensor_pairs
        )
    ):
        raise ELAL3SimulatorLabelError("authenticated oracle label differs")
    authority = {
        "status": AUTHORITY_STATUS,
        "simulator_optimizer_diagnostic_authorized": True,
        "training_authorized": True,
        "external_optimizer_authority_verified": True,
        "training_authority_source": (
            "separately-issued-pinned-local-authority"
        ),
        "training_authority_scope": (
            "one-row-oracle-q-simulator-overfit-diagnostic-only"
        ),
        "formal_training_authorized": False,
        "formal_c0_c1_c2_go_authorized": False,
        "exact160_eligible": False,
        "exact160_claim_authorized": False,
        "scientific_claim_authorized": False,
        "real_video_data": False,
        "source_instruction_inference_authorized": False,
        "model_output_claim_authorized": False,
        "oracle_q_teacher_forced": True,
        "action_encoder_qualified": False,
        "action_predictor_present": False,
        "upstream_training_use_forbidden_acknowledged": True,
        "upstream_packet_mutated": False,
    }
    scope = {
        "row_count": 1,
        "row_id": EXPECTED_ROW_ID,
        "media_variant_for_oracle_q": "target",
        "allowed_optimizer_updates_min": 1,
        "allowed_optimizer_updates_max": external_authority[
            "max_optimizer_updates_per_arm"
        ],
        "allowed_representation_variant": "full",
        "allowed_attention_width": 64,
        "decoded_review_required": True,
        "negative_interventions_required": [
            "zero",
            "phase_reverse",
            "role_slot_swap",
            "relation_zero",
        ],
    }
    unsigned = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "status": AUTHORITY_STATUS,
        "row_id": EXPECTED_ROW_ID,
        "source_packet": {
            "manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
            "manifest_digest": EXPECTED_MANIFEST_DIGEST,
            "media_pins_digest": EXPECTED_MEDIA_PINS_DIGEST,
            "read_only_upstream_packet": True,
        },
        "external_authority_binding": _external_authority_binding(
            external_authority
        ),
        "label_binding": {
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "label_digest": label_digest,
            "patch_grid": list(label.receipt["patch_grid"]),
            "event_mask_sha256": label.receipt["event_mask"]["sha256"],
            "q_tensor_rows_digest": label.receipt["q_tensor_rows_digest"],
            "oracle_media_variant": "target",
        },
        "scope": scope,
        "authority": authority,
    }
    return {**unsigned, "authority_digest": object_sha256(unsigned)}


def write_derivative_authority_create_only_v1(
    path: str | Path,
    label: ELAL3SimulatorOracleLabelV1,
    *,
    external_authority_path: str | Path,
    external_authority_sha256: str,
) -> dict[str, Any]:
    """Publish canonical authority bytes without overwriting any path."""

    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise ELAL3SimulatorLabelError("authority output must be absolute")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise ELAL3SimulatorLabelError("authority output parent is unavailable") from error
    if not parent.is_dir() or requested.name in ("", ".", ".."):
        raise ELAL3SimulatorLabelError("authority output path differs")
    output = parent / requested.name
    value = build_derivative_authority_v1(
        label,
        external_authority_path=external_authority_path,
        external_authority_sha256=external_authority_sha256,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output, flags, 0o440)
    except FileExistsError as error:
        raise ELAL3SimulatorLabelError("refusing to overwrite authority") from error
    try:
        payload = canonical_json_bytes(value) + b"\n"
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return value


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "AUTHORITY_STATUS",
    "ELAL3SimulatorLabelError",
    "ELAL3SimulatorOracleLabelV1",
    "EXPECTED_EXTERNAL_AUTHORITY_DIGEST",
    "EXPECTED_EXTERNAL_AUTHORITY_PATH",
    "EXPECTED_EXTERNAL_AUTHORITY_RELATIVE_PATH",
    "EXPECTED_EXTERNAL_AUTHORITY_SCHEMA_VERSION",
    "EXPECTED_EXTERNAL_AUTHORITY_SHA256",
    "EXPECTED_EXTERNAL_AUTHORITY_STATUS",
    "EXPECTED_MANIFEST_DIGEST",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_MEDIA_PINS",
    "EXPECTED_MEDIA_PINS_DIGEST",
    "EXPECTED_ROW_ID",
    "LABEL_SCHEMA_VERSION",
    "VerifiedC1RowV1",
    "build_derivative_authority_v1",
    "canonical_json_bytes",
    "file_sha256",
    "load_oracle_q_label_v1",
    "load_verified_c1_row",
    "object_sha256",
    "write_derivative_authority_create_only_v1",
]
