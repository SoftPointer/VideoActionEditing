#!/usr/bin/env python3
"""Decoded-video temporal counterfactual and exact81 event evidence.

This evaluator closes the evidence gap left by scalar-only endpoint packets.
It has three deliberately separate roles:

``prepare``
    Authenticate one native RV2V rollout, decode exactly 81 RGB frames, build
    chronological/reverse/fixed-phase-shuffle/freeze-first arms from *that same
    decoded video*, and emit lossless blind review media plus a private key.

external observation
    At least two independent observers score the blind media.  This module has
    no command that authors or fills observer labels.  Observer receipts must
    already exist as detached files and bind their own evidence/runtime bytes.

``seal``
    Replay every input byte, check observer independence/agreement, calculate
    the registered temporal energies and exact81 event traces, and emit a
    master receipt.  Thin endpoint projections bind and replay that master;
    four caller-supplied scalars or three caller-supplied probability arrays
    are never accepted as authority.

The module is CPU-only apart from invoking ffmpeg/ffprobe.  It performs no
training and never authorizes an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Optional, Sequence


EVENT_SPEC_SCHEMA = "bernini-decoded-action-event-spec-v1"
PUBLIC_CHALLENGE_SCHEMA = "bernini-decoded-temporal-event-public-challenge-v1"
PRIVATE_KEY_SCHEMA = "bernini-decoded-temporal-event-private-key-v1"
OBSERVER_RECEIPT_SCHEMA = "bernini-decoded-temporal-event-observer-receipt-v1"
OBSERVER_REGISTRATION_SCHEMA = (
    "bernini-decoded-temporal-event-observer-registration-v1"
)
MASTER_RECEIPT_SCHEMA = "bernini-decoded-temporal-event-master-receipt-v1"
TEMPORAL_PROJECTION_SCHEMA = (
    "bernini-pair-v5-same-video-temporal-counterfactual-packet-v2"
)
EVENT81_PROJECTION_SCHEMA = (
    "bernini-pair-v5-start-transition-terminal-hold-packet-v2"
)
ROLLOUT_SCHEMA = "pair-v5-native-rv2v4-rollout-receipt-v1"

FRAME_COUNT = 81
FPS = 25
LATENT_PHASES = 21
TRANSFORM_ORDER = ("target", "reverse", "shuffle", "freeze")
STATE_ORDER = ("start", "transition", "terminal")
EVIDENCE_ORDER = (*STATE_ORDER, "terminal_hold")
OBSERVER_KINDS = ("human_blind_annotation", "frozen_external_event_model")
MINIMUM_INDEPENDENT_OBSERVERS = 2
MAX_MEAN_ABSOLUTE_OBSERVER_DISAGREEMENT = 0.20
MINIMUM_FRAME_STATE_ARGMAX_AGREEMENT = 0.85
ENERGY_EPSILON = 1.0e-6

FRAME_WINDOWS = {
    "start": (0, 15),
    "transition": (16, 60),
    "terminal": (61, 80),
    "terminal_hold": (73, 80),
}

# Multiplication by eight is a fixed permutation modulo 21.  This is copied
# from the preregistered latent-phase counterfactual, but here it is applied to
# decoded 4n+1 frame blocks: phase 0 owns frame 0 and phase k>0 owns frames
# 4k-3..4k.  The transform therefore remains exact81 and requires no mask,
# tracking, flow, pose, or source-derived privileged inference input.
SHUFFLE_PHASE_ORDER = tuple((8 * index) % LATENT_PHASES for index in range(LATENT_PHASES))

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")


class DecodedTemporalEventError(ValueError):
    """A media, provenance, observer, transform, or receipt check failed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise DecodedTemporalEventError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _closed(value: Any, fields: Iterable[str], *, label: str) -> dict[str, Any]:
    expected = set(fields)
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise DecodedTemporalEventError(
            f"{label} field closure differs; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return dict(value)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DecodedTemporalEventError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise DecodedTemporalEventError(f"{label} must be a path-safe identifier")
    return value


def _text(value: Any, *, label: str, minimum: int = 8) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum or "\x00" in value:
        raise DecodedTemporalEventError(f"{label} must be nonempty text")
    return value


def _finite(value: Any, *, label: str, unit: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise DecodedTemporalEventError(f"{label} must be finite numeric")
    result = float(value)
    if unit and not 0.0 <= result <= 1.0:
        raise DecodedTemporalEventError(f"{label} must lie in [0,1]")
    return result


def _seal(unsigned: Mapping[str, Any], *, field: str = "receipt_digest") -> dict[str, Any]:
    value = dict(unsigned)
    return {**value, field: object_sha256(value)}


def _verify_seal(value: Mapping[str, Any], *, field: str, label: str) -> str:
    row = dict(value)
    digest = _sha256(row.pop(field, None), label=f"{label} {field}")
    if object_sha256(row) != digest:
        raise DecodedTemporalEventError(f"{label} embedded digest differs")
    return digest


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise DecodedTemporalEventError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _fresh_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise DecodedTemporalEventError(
            f"{label} must be a fresh absolute directory under a plain parent"
        )
    return path


def file_binding(path: str | Path) -> dict[str, str]:
    resolved = _plain_file(path, label="bound file")
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def validate_file_binding(
    value: Any, *, label: str, verify_bytes: bool = True
) -> dict[str, str]:
    row = _closed(value, {"path", "sha256"}, label=label)
    path = row["path"]
    digest = _sha256(row["sha256"], label=f"{label} SHA-256")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise DecodedTemporalEventError(f"{label} path must be absolute")
    if verify_bytes:
        resolved = _plain_file(path, label=label)
        if file_sha256(resolved) != digest:
            raise DecodedTemporalEventError(f"{label} file SHA-256 differs")
        path = str(resolved)
    return {"path": path, "sha256": digest}


def _load_bound_json(
    binding: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, str]]:
    checked = validate_file_binding(binding, label=label, verify_bytes=True)
    raw = Path(checked["path"]).read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DecodedTemporalEventError(f"{label} is invalid ASCII JSON") from error
    if not isinstance(value, dict):
        raise DecodedTemporalEventError(f"{label} root must be an object")
    return value, checked


def _write_create_only(path: Path, value: Mapping[str, Any], *, private: bool = False) -> None:
    if path.exists() or path.is_symlink():
        raise DecodedTemporalEventError(f"refusing to overwrite {path}")
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    os.chmod(path, 0o400 if private else 0o444)


_OBSERVER_REGISTRATION_FIELDS = {
    "schema_version",
    "observer_id",
    "observer_kind",
    "observer_authority_artifact",
    "observer_authority_digest",
    "observer_runtime_artifact",
    "model_or_protocol_artifact",
    "model_or_protocol_digest",
    "registered_before_candidate_review",
    "is_candidate_generator_or_challenge_preparer",
    "registration_digest",
}


def make_observer_registration(
    *,
    observer_id: str,
    observer_kind: str,
    observer_authority_artifact: Mapping[str, Any],
    observer_runtime_artifact: Mapping[str, Any],
    model_or_protocol_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    if observer_kind not in OBSERVER_KINDS:
        raise DecodedTemporalEventError("observer registration kind differs")
    authority = validate_file_binding(
        observer_authority_artifact,
        label="registered observer authority artifact",
        verify_bytes=True,
    )
    runtime = validate_file_binding(
        observer_runtime_artifact,
        label="registered observer runtime artifact",
        verify_bytes=True,
    )
    model = validate_file_binding(
        model_or_protocol_artifact,
        label="registered observer model/protocol artifact",
        verify_bytes=True,
    )
    unsigned = {
        "schema_version": OBSERVER_REGISTRATION_SCHEMA,
        "observer_id": _safe_id(observer_id, label="registered observer ID"),
        "observer_kind": observer_kind,
        "observer_authority_artifact": authority,
        "observer_authority_digest": authority["sha256"],
        "observer_runtime_artifact": runtime,
        "model_or_protocol_artifact": model,
        "model_or_protocol_digest": model["sha256"],
        "registered_before_candidate_review": True,
        "is_candidate_generator_or_challenge_preparer": False,
    }
    return {**unsigned, "registration_digest": object_sha256(unsigned)}


def validate_observer_registration(value: Any) -> dict[str, Any]:
    row = _closed(
        value, _OBSERVER_REGISTRATION_FIELDS, label="observer registration"
    )
    expected = make_observer_registration(
        observer_id=row["observer_id"],
        observer_kind=row["observer_kind"],
        observer_authority_artifact=row["observer_authority_artifact"],
        observer_runtime_artifact=row["observer_runtime_artifact"],
        model_or_protocol_artifact=row["model_or_protocol_artifact"],
    )
    if row != expected:
        raise DecodedTemporalEventError(
            "observer registration semantics or digest differs"
        )
    return row


def make_event_spec(
    *,
    action_family_id: str,
    source_video_sha256: str,
    complete_caption_sha256: str,
    actor_binding: str,
    start_state_question: str,
    transition_question: str,
    terminal_state_question: str,
    terminal_hold_question: str,
    registered_observers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    registrations = [validate_observer_registration(item) for item in registered_observers]
    if len(registrations) < MINIMUM_INDEPENDENT_OBSERVERS:
        raise DecodedTemporalEventError(
            "event spec must preregister at least two independent observers"
        )
    observer_ids = [item["observer_id"] for item in registrations]
    authority_digests = [item["observer_authority_digest"] for item in registrations]
    if (
        len(set(observer_ids)) != len(observer_ids)
        or len(set(authority_digests)) != len(authority_digests)
    ):
        raise DecodedTemporalEventError(
            "event-spec observer registrations are not independent"
        )
    unsigned = {
        "schema_version": EVENT_SPEC_SCHEMA,
        "action_family_id": _safe_id(action_family_id, label="action family ID"),
        "source_video_sha256": _sha256(
            source_video_sha256, label="event-spec source SHA-256"
        ),
        "complete_caption_sha256": _sha256(
            complete_caption_sha256, label="event-spec caption SHA-256"
        ),
        "actor_binding": _text(actor_binding, label="actor binding"),
        "state_questions": {
            "start": _text(start_state_question, label="start-state question"),
            "transition": _text(transition_question, label="transition question"),
            "terminal": _text(terminal_state_question, label="terminal-state question"),
            "terminal_hold": _text(
                terminal_hold_question, label="terminal-hold question"
            ),
        },
        "frame_windows": {name: list(window) for name, window in FRAME_WINDOWS.items()},
        "counterfactual_transform_order": list(TRANSFORM_ORDER),
        "registered_observers": registrations,
        "minimum_independent_observers": MINIMUM_INDEPENDENT_OBSERVERS,
        "candidate_seed_or_filename_used_to_define_event": False,
        "event_spec_authored_before_candidate_review": True,
        "mask_flow_pose_track_or_trajectory_required": False,
    }
    return {**unsigned, "event_spec_digest": object_sha256(unsigned)}


_EVENT_SPEC_FIELDS = {
    "schema_version",
    "action_family_id",
    "source_video_sha256",
    "complete_caption_sha256",
    "actor_binding",
    "state_questions",
    "frame_windows",
    "counterfactual_transform_order",
    "registered_observers",
    "minimum_independent_observers",
    "candidate_seed_or_filename_used_to_define_event",
    "event_spec_authored_before_candidate_review",
    "mask_flow_pose_track_or_trajectory_required",
    "event_spec_digest",
}


def validate_event_spec(value: Any) -> dict[str, Any]:
    row = _closed(value, _EVENT_SPEC_FIELDS, label="event spec")
    rebuilt = make_event_spec(
        action_family_id=row["action_family_id"],
        source_video_sha256=row["source_video_sha256"],
        complete_caption_sha256=row["complete_caption_sha256"],
        actor_binding=row["actor_binding"],
        start_state_question=_closed(
            row["state_questions"], STATE_ORDER + ("terminal_hold",), label="state questions"
        )["start"],
        transition_question=row["state_questions"]["transition"],
        terminal_state_question=row["state_questions"]["terminal"],
        terminal_hold_question=row["state_questions"]["terminal_hold"],
        registered_observers=row["registered_observers"],
    )
    if row != rebuilt:
        raise DecodedTemporalEventError("event spec semantics or digest differs")
    return row


def _phase_frame_indices(phase: int) -> tuple[int, ...]:
    if phase == 0:
        return (0,)
    if not 1 <= phase < LATENT_PHASES:
        raise DecodedTemporalEventError("phase index is outside exact21")
    return tuple(range(4 * phase - 3, 4 * phase + 1))


def temporal_index_map(transform_name: str) -> tuple[int, ...]:
    if transform_name == "target":
        result = tuple(range(FRAME_COUNT))
    elif transform_name == "reverse":
        result = tuple(range(FRAME_COUNT - 1, -1, -1))
    elif transform_name == "freeze":
        result = (0,) * FRAME_COUNT
    elif transform_name == "shuffle":
        result = tuple(
            frame
            for phase in SHUFFLE_PHASE_ORDER
            for frame in _phase_frame_indices(phase)
        )
    else:
        raise DecodedTemporalEventError("unknown decoded temporal transform")
    if len(result) != FRAME_COUNT or any(not 0 <= index < FRAME_COUNT for index in result):
        raise DecodedTemporalEventError("decoded temporal map is not exact81")
    if transform_name in ("target", "reverse", "shuffle") and sorted(result) != list(
        range(FRAME_COUNT)
    ):
        raise DecodedTemporalEventError(
            f"{transform_name} must preserve the exact decoded-frame multiset"
        )
    return result


def apply_frame_map(frames: Sequence[bytes], transform_name: str) -> tuple[bytes, ...]:
    if len(frames) != FRAME_COUNT or any(not isinstance(frame, bytes) for frame in frames):
        raise DecodedTemporalEventError("decoded frame sequence must contain 81 byte frames")
    return tuple(frames[index] for index in temporal_index_map(transform_name))


def _run(command: Sequence[str], *, input_bytes: Optional[bytes] = None) -> bytes:
    try:
        completed = subprocess.run(
            list(command),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise DecodedTemporalEventError(f"failed to execute {command[0]}") from error
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise DecodedTemporalEventError(
            f"command failed ({completed.returncode}): {command[0]}: {stderr}"
        )
    return completed.stdout


def probe_exact81_video(path: str | Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    video = _plain_file(path, label="candidate/review video")
    raw = _run(
        (
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(video),
        )
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
        streams = payload["streams"]
        stream = streams[0]
    except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise DecodedTemporalEventError("ffprobe output closure differs") from error
    width = stream.get("width")
    height = stream.get("height")
    frames = stream.get("nb_read_frames")
    rate = stream.get("avg_frame_rate")
    if (
        len(streams) != 1
        or type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or frames != str(FRAME_COUNT)
        or rate != f"{FPS}/1"
    ):
        raise DecodedTemporalEventError("video is not one exact81/25fps stream")
    return {
        "width": width,
        "height": height,
        "fps": FPS,
        "frame_count": FRAME_COUNT,
        "pixel_format": "rgb24",
    }


def decode_exact81_rgb24(
    path: str | Path, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe"
) -> tuple[tuple[bytes, ...], dict[str, Any]]:
    video = _plain_file(path, label="decoded video")
    geometry = probe_exact81_video(video, ffprobe=ffprobe)
    raw = _run(
        (
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        )
    )
    frame_bytes = geometry["width"] * geometry["height"] * 3
    if len(raw) != FRAME_COUNT * frame_bytes:
        raise DecodedTemporalEventError("decoded RGB byte count is not exact81")
    frames = tuple(
        raw[index * frame_bytes : (index + 1) * frame_bytes]
        for index in range(FRAME_COUNT)
    )
    metadata = {
        **geometry,
        "decoded_rgb24_sha256": hashlib.sha256(raw).hexdigest(),
        "per_frame_sha256": [hashlib.sha256(frame).hexdigest() for frame in frames],
        "decoder_contract": "ffmpeg_vsync0_rgb24_exact81",
    }
    return frames, metadata


def _encode_lossless_review(
    frames: Sequence[bytes],
    *,
    width: int,
    height: int,
    output: Path,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise DecodedTemporalEventError(f"refusing to overwrite {output}")
    raw = b"".join(frames)
    _run(
        (
            ffmpeg,
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(FPS),
            "-i",
            "pipe:0",
            "-frames:v",
            str(FRAME_COUNT),
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-g",
            "1",
            str(output),
        ),
        input_bytes=raw,
    )
    decoded, metadata = decode_exact81_rgb24(output, ffmpeg=ffmpeg, ffprobe=ffprobe)
    decoded_raw = b"".join(decoded)
    if decoded_raw != raw:
        raise DecodedTemporalEventError("lossless review media failed RGB byte replay")
    return {
        "file": file_binding(output),
        "container": "matroska",
        "codec": "ffv1",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "decoded_rgb24_sha256": metadata["decoded_rgb24_sha256"],
        "per_frame_sha256_digest": object_sha256(metadata["per_frame_sha256"]),
        "lossless_rgb24_replay_verified": True,
    }


def _validate_embedded_receipt(value: Any, *, label: str) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise DecodedTemporalEventError(f"{label} must be an object")
    row = dict(value)
    digest = _verify_seal(row, field="receipt_digest", label=label)
    return row, digest


def _load_rollout_receipt(path: str | Path) -> dict[str, Any]:
    binding = file_binding(path)
    value, _ = _load_bound_json(binding, label="native rollout receipt")
    row, digest = _validate_embedded_receipt(value, label="native rollout receipt")
    candidate = row.get("candidate")
    sampling = row.get("sampling_contract")
    artifacts = row.get("artifacts")
    if not all(isinstance(item, Mapping) for item in (candidate, sampling, artifacts)):
        raise DecodedTemporalEventError("native rollout receipt structure differs")
    if (
        row.get("schema_version") != ROLLOUT_SCHEMA
        or sampling.get("condition_mode") != "rv2v4"
        or sampling.get("num_frames") != FRAME_COUNT
        or sampling.get("latent_frames") != LATENT_PHASES
        or sampling.get("fps") != FPS
        or sampling.get("num_inference_steps") != 40
        or sampling.get("source_reference_indices") != [0, 27, 53, 80]
    ):
        raise DecodedTemporalEventError("native rollout is not pinned RV2V4 exact81/40")
    candidate_id = _safe_id(candidate.get("candidate_id"), label="candidate ID")
    source_path = _plain_file(candidate.get("source_video"), label="source video")
    source_sha = _sha256(candidate.get("source_video_sha256"), label="source SHA-256")
    if file_sha256(source_path) != source_sha:
        raise DecodedTemporalEventError("source video SHA-256 differs")
    caption = _text(candidate.get("complete_caption"), label="complete caption")
    caption_sha = _sha256(
        candidate.get("complete_caption_sha256"), label="caption SHA-256"
    )
    if hashlib.sha256(caption.encode("utf-8")).hexdigest() != caption_sha:
        raise DecodedTemporalEventError("complete caption SHA-256 differs")
    seed = candidate.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise DecodedTemporalEventError("candidate seed differs")
    mp4 = artifacts.get("mp4")
    if not isinstance(mp4, Mapping):
        raise DecodedTemporalEventError("candidate MP4 artifact is absent")
    # The native receipt carries additional exact81 geometry and latent
    # provenance inside the MP4 artifact.  Project only its path/hash pair;
    # never weaken ``validate_file_binding`` to accept arbitrary extra fields.
    mp4_binding = validate_file_binding(
        {"path": mp4.get("path"), "sha256": mp4.get("sha256")},
        label="candidate MP4",
        verify_bytes=True,
    )
    if (
        mp4.get("frame_count") != FRAME_COUNT
        or mp4.get("fps") != FPS
        or type(mp4.get("width")) is not int
        or type(mp4.get("height")) is not int
        or mp4.get("width") <= 0
        or mp4.get("height") <= 0
    ):
        raise DecodedTemporalEventError("native MP4 artifact geometry differs")
    native_path = _plain_file(row.get("native_receipt_path"), label="native receipt")
    native_file = file_binding(native_path)
    if native_file["sha256"] != _sha256(
        row.get("native_receipt_sha256"), label="native receipt file SHA-256"
    ):
        raise DecodedTemporalEventError("native receipt file binding differs")
    native_value, _ = _load_bound_json(native_file, label="native receipt")
    _, native_digest = _validate_embedded_receipt(native_value, label="native receipt")
    if native_digest != _sha256(
        row.get("native_receipt_digest"), label="native receipt digest"
    ):
        raise DecodedTemporalEventError("native receipt digest binding differs")
    return {
        "candidate_id": candidate_id,
        "source_video": {"path": str(source_path), "sha256": source_sha},
        "complete_caption": caption,
        "complete_caption_sha256": caption_sha,
        "seed": seed,
        "candidate_mp4": mp4_binding,
        "rollout_receipt": binding,
        "rollout_receipt_digest": digest,
        "native_receipt": native_file,
        "native_receipt_digest": native_digest,
    }


def _blind_id(salt: bytes, candidate_mp4_sha256: str, transform_name: str) -> str:
    digest = hashlib.sha256(
        b"bernini-decoded-blind-arm-v1\x00"
        + salt
        + bytes.fromhex(candidate_mp4_sha256)
        + transform_name.encode("ascii")
    ).hexdigest()
    return f"arm-{digest[:24]}"


_PUBLIC_FIELDS = {
    "schema_version",
    "challenge_id",
    "action_family_id",
    "actor_binding",
    "state_questions",
    "frame_windows",
    "frame_count",
    "fps",
    "blind_arm_order",
    "blind_arms",
    "candidate_identity_exposed_to_observer",
    "transform_identity_exposed_to_observer",
    "review_media_are_lossless_exact81_transforms",
    "challenge_digest",
}

_PRIVATE_FIELDS = {
    "schema_version",
    "challenge_digest",
    "public_challenge_file",
    "event_spec_file",
    "event_spec_digest",
    "preparer_id",
    "blind_salt_sha256",
    "candidate_id",
    "source_video",
    "complete_caption_sha256",
    "seed",
    "candidate_mp4",
    "rollout_receipt",
    "rollout_receipt_digest",
    "native_receipt",
    "native_receipt_digest",
    "decoded_candidate",
    "transform_by_blind_id",
    "transform_order",
    "same_candidate_video_only",
    "external_observer_labels_present",
    "training_performed",
    "optimizer_authorized",
    "receipt_digest",
}


def _validate_public_challenge(value: Any, *, replay_media: bool) -> dict[str, Any]:
    row = _closed(value, _PUBLIC_FIELDS, label="public challenge")
    digest = _verify_seal(row, field="challenge_digest", label="public challenge")
    _safe_id(row["challenge_id"], label="challenge ID")
    _safe_id(row["action_family_id"], label="challenge action family")
    _text(row["actor_binding"], label="challenge actor binding")
    questions = _closed(
        row["state_questions"], STATE_ORDER + ("terminal_hold",), label="challenge questions"
    )
    for name, question in questions.items():
        _text(question, label=f"challenge {name} question")
    if row["frame_windows"] != {
        name: list(window) for name, window in FRAME_WINDOWS.items()
    }:
        raise DecodedTemporalEventError("challenge frame windows differ")
    order = row["blind_arm_order"]
    arms = row["blind_arms"]
    if (
        row["frame_count"] != FRAME_COUNT
        or row["fps"] != FPS
        or not isinstance(order, list)
        or len(order) != len(TRANSFORM_ORDER)
        or len(set(order)) != len(TRANSFORM_ORDER)
        or not isinstance(arms, Mapping)
        or set(arms) != set(order)
        or row["candidate_identity_exposed_to_observer"] is not False
        or row["transform_identity_exposed_to_observer"] is not False
        or row["review_media_are_lossless_exact81_transforms"] is not True
    ):
        raise DecodedTemporalEventError("public challenge authority differs")
    checked_arms: dict[str, Any] = {}
    for blind_id in order:
        _safe_id(blind_id, label="blind arm ID")
        arm = _closed(
            arms[blind_id],
            {
                "blind_arm_id",
                "review_media",
                "frame_count",
                "fps",
                "decoded_rgb24_sha256",
                "per_frame_sha256_digest",
            },
            label="public blind arm",
        )
        media = validate_file_binding(
            arm["review_media"], label=f"{blind_id} review media", verify_bytes=replay_media
        )
        if (
            arm["blind_arm_id"] != blind_id
            or arm["frame_count"] != FRAME_COUNT
            or arm["fps"] != FPS
        ):
            raise DecodedTemporalEventError("public blind-arm geometry differs")
        decoded_sha = _sha256(
            arm["decoded_rgb24_sha256"], label="blind-arm decoded RGB SHA-256"
        )
        per_frame_digest = _sha256(
            arm["per_frame_sha256_digest"], label="blind-arm frame-set digest"
        )
        if replay_media:
            _, metadata = decode_exact81_rgb24(media["path"])
            if (
                metadata["decoded_rgb24_sha256"] != decoded_sha
                or object_sha256(metadata["per_frame_sha256"]) != per_frame_digest
            ):
                raise DecodedTemporalEventError("blind review media RGB replay differs")
        checked_arms[blind_id] = {
            **arm,
            "review_media": media,
            "decoded_rgb24_sha256": decoded_sha,
            "per_frame_sha256_digest": per_frame_digest,
        }
    row["blind_arms"] = checked_arms
    row["challenge_digest"] = digest
    return row


def prepare_challenge(
    *,
    rollout_receipt_path: str | Path,
    event_spec_path: str | Path,
    blind_salt_path: str | Path,
    preparer_id: str,
    output_dir: str | Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = _fresh_directory(output_dir, label="challenge output")
    rollout = _load_rollout_receipt(rollout_receipt_path)
    event_binding = file_binding(event_spec_path)
    event_raw, _ = _load_bound_json(event_binding, label="event spec")
    event = validate_event_spec(event_raw)
    if (
        event["source_video_sha256"] != rollout["source_video"]["sha256"]
        or event["complete_caption_sha256"] != rollout["complete_caption_sha256"]
    ):
        raise DecodedTemporalEventError("event spec does not bind this source/caption")
    salt_path = _plain_file(blind_salt_path, label="blind salt")
    salt = salt_path.read_bytes()
    if len(salt) < 32:
        raise DecodedTemporalEventError("blind salt must contain at least 32 bytes")
    preparer = _safe_id(preparer_id, label="challenge preparer ID")
    frames, decoded = decode_exact81_rgb24(
        rollout["candidate_mp4"]["path"], ffmpeg=ffmpeg, ffprobe=ffprobe
    )
    output.mkdir(mode=0o700)

    transform_private: dict[str, Any] = {}
    public_arms: dict[str, Any] = {}
    for transform_name in TRANSFORM_ORDER:
        transformed = apply_frame_map(frames, transform_name)
        blind_id = _blind_id(salt, rollout["candidate_mp4"]["sha256"], transform_name)
        if blind_id in transform_private:
            raise DecodedTemporalEventError("blind arm ID collision")
        review = output / f"{blind_id}.mkv"
        media = _encode_lossless_review(
            transformed,
            width=decoded["width"],
            height=decoded["height"],
            output=review,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
        mapping = temporal_index_map(transform_name)
        transform_private[blind_id] = {
            "blind_arm_id": blind_id,
            "transform_name": transform_name,
            "frame_index_map": list(mapping),
            "frame_index_map_digest": object_sha256(list(mapping)),
            "review_media": media["file"],
            "decoded_rgb24_sha256": media["decoded_rgb24_sha256"],
            "per_frame_sha256_digest": media["per_frame_sha256_digest"],
        }
        public_arms[blind_id] = {
            "blind_arm_id": blind_id,
            "review_media": media["file"],
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "decoded_rgb24_sha256": media["decoded_rgb24_sha256"],
            "per_frame_sha256_digest": media["per_frame_sha256_digest"],
        }

    blind_order = sorted(public_arms)
    challenge_id = "challenge-" + hashlib.sha256(
        b"bernini-decoded-challenge-v1\x00"
        + bytes.fromhex(rollout["candidate_mp4"]["sha256"])
        + bytes.fromhex(event["event_spec_digest"])
        + hashlib.sha256(salt).digest()
    ).hexdigest()[:24]
    public_unsigned = {
        "schema_version": PUBLIC_CHALLENGE_SCHEMA,
        "challenge_id": challenge_id,
        "action_family_id": event["action_family_id"],
        "actor_binding": event["actor_binding"],
        "state_questions": event["state_questions"],
        "frame_windows": event["frame_windows"],
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "blind_arm_order": blind_order,
        "blind_arms": {blind_id: public_arms[blind_id] for blind_id in blind_order},
        "candidate_identity_exposed_to_observer": False,
        "transform_identity_exposed_to_observer": False,
        "review_media_are_lossless_exact81_transforms": True,
    }
    public = {**public_unsigned, "challenge_digest": object_sha256(public_unsigned)}
    public_path = output / "public-challenge.json"
    _write_create_only(public_path, public)

    private_unsigned = {
        "schema_version": PRIVATE_KEY_SCHEMA,
        "challenge_digest": public["challenge_digest"],
        "public_challenge_file": file_binding(public_path),
        "event_spec_file": event_binding,
        "event_spec_digest": event["event_spec_digest"],
        "preparer_id": preparer,
        "blind_salt_sha256": hashlib.sha256(salt).hexdigest(),
        "candidate_id": rollout["candidate_id"],
        "source_video": rollout["source_video"],
        "complete_caption_sha256": rollout["complete_caption_sha256"],
        "seed": rollout["seed"],
        "candidate_mp4": rollout["candidate_mp4"],
        "rollout_receipt": rollout["rollout_receipt"],
        "rollout_receipt_digest": rollout["rollout_receipt_digest"],
        "native_receipt": rollout["native_receipt"],
        "native_receipt_digest": rollout["native_receipt_digest"],
        "decoded_candidate": decoded,
        "transform_by_blind_id": {
            blind_id: transform_private[blind_id] for blind_id in blind_order
        },
        "transform_order": list(TRANSFORM_ORDER),
        "same_candidate_video_only": True,
        "external_observer_labels_present": False,
        "training_performed": False,
        "optimizer_authorized": False,
    }
    private = _seal(private_unsigned)
    _write_create_only(output / "private-transform-key.json", private, private=True)
    return public, private


def validate_prepared_challenge(
    public_value: Any,
    private_value: Any,
    *,
    replay_media: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    public = _validate_public_challenge(public_value, replay_media=replay_media)
    private = _closed(private_value, _PRIVATE_FIELDS, label="private transform key")
    digest = _verify_seal(private, field="receipt_digest", label="private transform key")
    if private["schema_version"] != PRIVATE_KEY_SCHEMA:
        raise DecodedTemporalEventError("private transform-key schema differs")
    public_binding = validate_file_binding(
        private["public_challenge_file"], label="bound public challenge", verify_bytes=True
    )
    loaded_public, _ = _load_bound_json(public_binding, label="bound public challenge")
    if loaded_public != public_value or private["challenge_digest"] != public["challenge_digest"]:
        raise DecodedTemporalEventError("private/public challenge binding differs")
    event_binding = validate_file_binding(
        private["event_spec_file"], label="bound event spec", verify_bytes=True
    )
    event_raw, _ = _load_bound_json(event_binding, label="bound event spec")
    event = validate_event_spec(event_raw)
    if event["event_spec_digest"] != private["event_spec_digest"]:
        raise DecodedTemporalEventError("private event-spec binding differs")
    rollout_binding = validate_file_binding(
        private["rollout_receipt"], label="bound rollout receipt", verify_bytes=True
    )
    rollout = _load_rollout_receipt(rollout_binding["path"])
    native_binding = validate_file_binding(
        private["native_receipt"], label="bound native receipt", verify_bytes=True
    )
    if (
        rollout["candidate_id"] != private["candidate_id"]
        or rollout["source_video"] != private["source_video"]
        or rollout["complete_caption_sha256"] != private["complete_caption_sha256"]
        or rollout["seed"] != private["seed"]
        or rollout["candidate_mp4"] != private["candidate_mp4"]
        or rollout["rollout_receipt"] != rollout_binding
        or rollout["rollout_receipt_digest"] != private["rollout_receipt_digest"]
        or rollout["native_receipt"] != native_binding
        or rollout["native_receipt_digest"] != private["native_receipt_digest"]
        or event["source_video_sha256"] != private["source_video"]["sha256"]
        or event["complete_caption_sha256"] != private["complete_caption_sha256"]
        or public["action_family_id"] != event["action_family_id"]
        or public["actor_binding"] != event["actor_binding"]
        or public["state_questions"] != event["state_questions"]
    ):
        raise DecodedTemporalEventError("private source/prompt/rollout join differs")
    transforms = private["transform_by_blind_id"]
    if (
        not isinstance(transforms, Mapping)
        or set(transforms) != set(public["blind_arm_order"])
        or private["transform_order"] != list(TRANSFORM_ORDER)
        or private["same_candidate_video_only"] is not True
        or private["external_observer_labels_present"] is not False
        or private["training_performed"] is not False
        or private["optimizer_authorized"] is not False
    ):
        raise DecodedTemporalEventError("private transform authority differs")
    transform_names: list[str] = []
    decoded_frames: Optional[tuple[bytes, ...]] = None
    decoded_metadata: Optional[dict[str, Any]] = None
    if replay_media:
        decoded_frames, decoded_metadata = decode_exact81_rgb24(
            private["candidate_mp4"]["path"]
        )
        if decoded_metadata != private["decoded_candidate"]:
            raise DecodedTemporalEventError("candidate decoded-frame replay differs")
    for blind_id in public["blind_arm_order"]:
        item = _closed(
            transforms[blind_id],
            {
                "blind_arm_id",
                "transform_name",
                "frame_index_map",
                "frame_index_map_digest",
                "review_media",
                "decoded_rgb24_sha256",
                "per_frame_sha256_digest",
            },
            label="private transform arm",
        )
        name = item["transform_name"]
        transform_names.append(name)
        mapping = temporal_index_map(name)
        if (
            item["blind_arm_id"] != blind_id
            or item["frame_index_map"] != list(mapping)
            or item["frame_index_map_digest"] != object_sha256(list(mapping))
            or item["review_media"] != public["blind_arms"][blind_id]["review_media"]
            or item["decoded_rgb24_sha256"]
            != public["blind_arms"][blind_id]["decoded_rgb24_sha256"]
            or item["per_frame_sha256_digest"]
            != public["blind_arms"][blind_id]["per_frame_sha256_digest"]
        ):
            raise DecodedTemporalEventError("private/public temporal arm differs")
        if replay_media:
            assert decoded_frames is not None
            transformed = apply_frame_map(decoded_frames, name)
            raw = b"".join(transformed)
            if (
                hashlib.sha256(raw).hexdigest() != item["decoded_rgb24_sha256"]
                or object_sha256(
                    [hashlib.sha256(frame).hexdigest() for frame in transformed]
                )
                != item["per_frame_sha256_digest"]
            ):
                raise DecodedTemporalEventError(
                    "review arm is not the registered same-candidate transform"
                )
    if sorted(transform_names) != sorted(TRANSFORM_ORDER):
        raise DecodedTemporalEventError("private transform set differs")
    private["receipt_digest"] = digest
    return public, private


_OBSERVER_FIELDS = {
    "schema_version",
    "observer_id",
    "observer_kind",
    "observer_authority_digest",
    "challenge_digest",
    "event_spec_digest",
    "blind_arm_order",
    "arm_observations_by_blind_id",
    "detached_evidence_artifact",
    "observer_runtime_artifact",
    "model_or_protocol_digest",
    "independent_from_candidate_generator",
    "independent_from_challenge_preparer",
    "transform_identity_was_hidden",
    "candidate_identity_was_hidden",
    "labels_not_inferred_from_filename_branch_or_seed",
    "annotation_complete",
    "receipt_self_signature_authorizes_optimizer",
    "receipt_digest",
}


def _probability_array(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != FRAME_COUNT:
        raise DecodedTemporalEventError(f"{label} must contain exact81 probabilities")
    return [_finite(item, label=f"{label} probability", unit=True) for item in value]


def validate_observer_receipt(
    value: Any,
    *,
    public_challenge: Mapping[str, Any],
    event_spec: Mapping[str, Any],
    replay_files: bool = True,
) -> dict[str, Any]:
    public = _validate_public_challenge(public_challenge, replay_media=False)
    event = validate_event_spec(event_spec)
    row = _closed(value, _OBSERVER_FIELDS, label="observer receipt")
    digest = _verify_seal(row, field="receipt_digest", label="observer receipt")
    observer_id = _safe_id(row["observer_id"], label="observer ID")
    if row["observer_kind"] not in OBSERVER_KINDS:
        raise DecodedTemporalEventError("observer kind differs")
    authority = _sha256(
        row["observer_authority_digest"], label="observer authority digest"
    )
    event_digest = _sha256(row["event_spec_digest"], label="observer event-spec digest")
    evidence = validate_file_binding(
        row["detached_evidence_artifact"],
        label="detached observer evidence",
        verify_bytes=replay_files,
    )
    runtime = validate_file_binding(
        row["observer_runtime_artifact"],
        label="observer runtime artifact",
        verify_bytes=replay_files,
    )
    _sha256(row["model_or_protocol_digest"], label="observer model/protocol digest")
    if (
        row["schema_version"] != OBSERVER_RECEIPT_SCHEMA
        or row["challenge_digest"] != public["challenge_digest"]
        or event_digest != event["event_spec_digest"]
        or row["blind_arm_order"] != public["blind_arm_order"]
        or row["independent_from_candidate_generator"] is not True
        or row["independent_from_challenge_preparer"] is not True
        or row["transform_identity_was_hidden"] is not True
        or row["candidate_identity_was_hidden"] is not True
        or row["labels_not_inferred_from_filename_branch_or_seed"] is not True
        or row["annotation_complete"] is not True
        or row["receipt_self_signature_authorizes_optimizer"] is not False
    ):
        raise DecodedTemporalEventError("observer authority/independence differs")
    registrations = {
        item["observer_id"]: item for item in event["registered_observers"]
    }
    registration = registrations.get(observer_id)
    if (
        registration is None
        or row["observer_kind"] != registration["observer_kind"]
        or authority != registration["observer_authority_digest"]
        or runtime != registration["observer_runtime_artifact"]
        or row["model_or_protocol_digest"]
        != registration["model_or_protocol_digest"]
    ):
        raise DecodedTemporalEventError(
            "observer receipt is not one of the preregistered external authorities"
        )
    observations = row["arm_observations_by_blind_id"]
    if not isinstance(observations, Mapping) or set(observations) != set(
        public["blind_arm_order"]
    ):
        raise DecodedTemporalEventError("observer blind-arm coverage differs")
    checked: dict[str, Any] = {}
    for blind_id in public["blind_arm_order"]:
        arm = _closed(
            observations[blind_id],
            {
                "blind_arm_id",
                "review_media_sha256",
                "frame_indices",
                "start_probability_by_frame",
                "transition_probability_by_frame",
                "terminal_probability_by_frame",
                "terminal_hold_probability_by_frame",
                "ambiguous_or_unreviewable",
            },
            label="observer arm",
        )
        if (
            arm["blind_arm_id"] != blind_id
            or arm["review_media_sha256"]
            != public["blind_arms"][blind_id]["review_media"]["sha256"]
            or arm["frame_indices"] != list(range(FRAME_COUNT))
            or arm["ambiguous_or_unreviewable"] is not False
        ):
            raise DecodedTemporalEventError("observer arm media/frame authority differs")
        checked[blind_id] = {
            **arm,
            "start_probability_by_frame": _probability_array(
                arm["start_probability_by_frame"], label=f"{blind_id} start"
            ),
            "transition_probability_by_frame": _probability_array(
                arm["transition_probability_by_frame"], label=f"{blind_id} transition"
            ),
            "terminal_probability_by_frame": _probability_array(
                arm["terminal_probability_by_frame"], label=f"{blind_id} terminal"
            ),
            "terminal_hold_probability_by_frame": _probability_array(
                arm["terminal_hold_probability_by_frame"],
                label=f"{blind_id} terminal hold",
            ),
        }
    row.update(
        {
            "observer_id": observer_id,
            "observer_authority_digest": authority,
            "event_spec_digest": event_digest,
            "detached_evidence_artifact": evidence,
            "observer_runtime_artifact": runtime,
            "arm_observations_by_blind_id": checked,
            "receipt_digest": digest,
        }
    )
    return row


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise DecodedTemporalEventError("mean input is empty")
    return float(sum(values) / len(values))


def _event_energy(probabilities: Mapping[str, Sequence[float]]) -> float:
    start = probabilities["start"]
    transition = probabilities["transition"]
    terminal = probabilities["terminal"]
    terminal_hold_probability = probabilities["terminal_hold"]

    def nlog(probability: float) -> float:
        return -math.log(max(ENERGY_EPSILON, min(1.0, probability)))

    def nlog_not(probability: float) -> float:
        return -math.log(max(ENERGY_EPSILON, min(1.0, 1.0 - probability)))

    start_positive = _mean([nlog(value) for value in start[0:16]])
    transition_top5 = sorted(transition[16:61], reverse=True)[:5]
    transition_positive = _mean([nlog(value) for value in transition_top5])
    terminal_positive = _mean([nlog(value) for value in terminal[61:81]])
    terminal_hold = _mean(
        [nlog(value) for value in terminal_hold_probability[73:81]]
    )
    no_early_terminal = _mean([nlog_not(value) for value in terminal[0:16]])
    no_late_start = _mean([nlog_not(value) for value in start[73:81]])
    result = _mean(
        (
            start_positive,
            transition_positive,
            terminal_positive,
            terminal_hold,
            no_early_terminal,
            no_late_start,
        )
    )
    if not math.isfinite(result) or result < 0.0:
        raise DecodedTemporalEventError("registered event energy is invalid")
    return result


def _pairwise_agreement(
    observers: Sequence[Mapping[str, Any]], blind_id: str
) -> tuple[dict[str, float], float]:
    by_state: dict[str, float] = {}
    for state in EVIDENCE_ORDER:
        field = f"{state}_probability_by_frame"
        differences = []
        for left_index in range(len(observers)):
            left = observers[left_index]["arm_observations_by_blind_id"][blind_id][field]
            for right_index in range(left_index + 1, len(observers)):
                right = observers[right_index]["arm_observations_by_blind_id"][blind_id][field]
                differences.extend(abs(a - b) for a, b in zip(left, right))
        by_state[state] = _mean(differences)
    categorical = []
    for left_index in range(len(observers)):
        left_arm = observers[left_index]["arm_observations_by_blind_id"][blind_id]
        for right_index in range(left_index + 1, len(observers)):
            right_arm = observers[right_index]["arm_observations_by_blind_id"][blind_id]
            for frame in range(FRAME_COUNT):
                left_state = max(
                    STATE_ORDER,
                    key=lambda name: left_arm[f"{name}_probability_by_frame"][frame],
                )
                right_state = max(
                    STATE_ORDER,
                    key=lambda name: right_arm[f"{name}_probability_by_frame"][frame],
                )
                categorical.append(left_state == right_state)
    return by_state, _mean([1.0 if item else 0.0 for item in categorical])


_MASTER_FIELDS = {
    "schema_version",
    "evaluator_implementation",
    "public_challenge_file",
    "private_transform_key_file",
    "challenge_digest",
    "private_transform_key_digest",
    "event_spec_digest",
    "action_family_id",
    "candidate_id",
    "source_video_sha256",
    "complete_caption_sha256",
    "seed",
    "candidate_mp4_sha256",
    "rollout_receipt_digest",
    "native_receipt_digest",
    "observer_receipt_files",
    "observer_receipt_digests",
    "observer_id_order",
    "observer_authority_digest_order",
    "observer_count",
    "minimum_independent_observers",
    "consensus_probability_by_transform",
    "agreement_by_transform",
    "branch_energy_by_name",
    "chronological_event_probability_by_frame",
    "independent_observer_gate_passed",
    "observer_agreement_gate_passed",
    "same_candidate_transform_replay_passed",
    "evidence_valid",
    "failure_reasons",
    "input_closure",
    "training_performed",
    "optimizer_authorized",
    "receipt_digest",
}


def _module_binding() -> dict[str, str]:
    return file_binding(Path(__file__).resolve())


def _build_master_from_bindings(
    *,
    public_challenge_file: Mapping[str, Any],
    private_transform_key_file: Mapping[str, Any],
    observer_receipt_files: Sequence[Mapping[str, Any]],
    replay_media: bool,
) -> dict[str, Any]:
    public_raw, public_binding = _load_bound_json(
        public_challenge_file, label="master public challenge"
    )
    private_raw, private_binding = _load_bound_json(
        private_transform_key_file, label="master private transform key"
    )
    public, private = validate_prepared_challenge(
        public_raw, private_raw, replay_media=replay_media
    )
    event_raw, _ = _load_bound_json(private["event_spec_file"], label="master event spec")
    event = validate_event_spec(event_raw)
    if not isinstance(observer_receipt_files, Sequence) or isinstance(
        observer_receipt_files, (str, bytes)
    ):
        raise DecodedTemporalEventError("observer receipt bindings must be a sequence")
    observers = []
    observer_bindings = []
    for index, binding in enumerate(observer_receipt_files):
        raw, checked_binding = _load_bound_json(
            binding, label=f"observer receipt {index}"
        )
        observers.append(
            validate_observer_receipt(
                raw,
                public_challenge=public,
                event_spec=event,
                replay_files=True,
            )
        )
        observer_bindings.append(checked_binding)
    if len(observers) < MINIMUM_INDEPENDENT_OBSERVERS:
        raise DecodedTemporalEventError("at least two detached observers are required")
    observer_ids = [row["observer_id"] for row in observers]
    authorities = [row["observer_authority_digest"] for row in observers]
    evidence_hashes = [row["detached_evidence_artifact"]["sha256"] for row in observers]
    independence = bool(
        len(set(observer_ids)) == len(observer_ids)
        and len(set(authorities)) == len(authorities)
        and len(set(evidence_hashes)) == len(evidence_hashes)
        and private["preparer_id"] not in set(observer_ids)
    )

    consensus_by_transform: dict[str, Any] = {}
    agreement_by_transform: dict[str, Any] = {}
    energy_by_name: dict[str, float] = {}
    agreement_pass = True
    blind_id_by_transform = {
        item["transform_name"]: blind_id
        for blind_id, item in private["transform_by_blind_id"].items()
    }
    for transform_name in TRANSFORM_ORDER:
        blind_id = blind_id_by_transform[transform_name]
        consensus: dict[str, list[float]] = {}
        for state in EVIDENCE_ORDER:
            field = f"{state}_probability_by_frame"
            consensus[state] = [
                _mean(
                    [
                        observer["arm_observations_by_blind_id"][blind_id][field][frame]
                        for observer in observers
                    ]
                )
                for frame in range(FRAME_COUNT)
            ]
        mean_abs, argmax_agreement = _pairwise_agreement(observers, blind_id)
        arm_pass = bool(
            all(
                value <= MAX_MEAN_ABSOLUTE_OBSERVER_DISAGREEMENT
                for value in mean_abs.values()
            )
            and argmax_agreement >= MINIMUM_FRAME_STATE_ARGMAX_AGREEMENT
        )
        agreement_pass = agreement_pass and arm_pass
        consensus_by_transform[transform_name] = {
            "blind_arm_id": blind_id,
            "start_probability_by_frame": consensus["start"],
            "transition_probability_by_frame": consensus["transition"],
            "terminal_probability_by_frame": consensus["terminal"],
            "terminal_hold_probability_by_frame": consensus["terminal_hold"],
        }
        agreement_by_transform[transform_name] = {
            "mean_absolute_probability_disagreement_by_state": mean_abs,
            "frame_state_argmax_agreement": argmax_agreement,
            "maximum_allowed_mean_absolute_probability_disagreement": (
                MAX_MEAN_ABSOLUTE_OBSERVER_DISAGREEMENT
            ),
            "minimum_required_frame_state_argmax_agreement": (
                MINIMUM_FRAME_STATE_ARGMAX_AGREEMENT
            ),
            "passed": arm_pass,
        }
        energy_by_name[transform_name] = _event_energy(consensus)

    failures = []
    if not independence:
        failures.append("independent_observer_authority")
    if not agreement_pass:
        failures.append("observer_agreement")
    evidence_valid = not failures
    chronological = consensus_by_transform["target"]
    unsigned = {
        "schema_version": MASTER_RECEIPT_SCHEMA,
        "evaluator_implementation": _module_binding(),
        "public_challenge_file": public_binding,
        "private_transform_key_file": private_binding,
        "challenge_digest": public["challenge_digest"],
        "private_transform_key_digest": private["receipt_digest"],
        "event_spec_digest": event["event_spec_digest"],
        "action_family_id": event["action_family_id"],
        "candidate_id": private["candidate_id"],
        "source_video_sha256": private["source_video"]["sha256"],
        "complete_caption_sha256": private["complete_caption_sha256"],
        "seed": private["seed"],
        "candidate_mp4_sha256": private["candidate_mp4"]["sha256"],
        "rollout_receipt_digest": private["rollout_receipt_digest"],
        "native_receipt_digest": private["native_receipt_digest"],
        "observer_receipt_files": observer_bindings,
        "observer_receipt_digests": [row["receipt_digest"] for row in observers],
        "observer_id_order": observer_ids,
        "observer_authority_digest_order": authorities,
        "observer_count": len(observers),
        "minimum_independent_observers": MINIMUM_INDEPENDENT_OBSERVERS,
        "consensus_probability_by_transform": consensus_by_transform,
        "agreement_by_transform": agreement_by_transform,
        "branch_energy_by_name": energy_by_name,
        "chronological_event_probability_by_frame": {
            "start": chronological["start_probability_by_frame"],
            "transition": chronological["transition_probability_by_frame"],
            "terminal": chronological["terminal_probability_by_frame"],
            "terminal_hold": chronological[
                "terminal_hold_probability_by_frame"
            ],
        },
        "independent_observer_gate_passed": independence,
        "observer_agreement_gate_passed": agreement_pass,
        "same_candidate_transform_replay_passed": bool(replay_media),
        "evidence_valid": evidence_valid and bool(replay_media),
        "failure_reasons": failures if replay_media else [*failures, "media_not_replayed"],
        "input_closure": {
            "same_candidate_decoded_video_only": True,
            "chronological_reverse_shuffle_freeze_from_same_rgb_tensor": True,
            "t2v_media_latent_noise_or_donor_consumed": False,
            "source_video_used_as_event_target": False,
            "mask_flow_pose_track_or_trajectory_consumed": False,
            "observer_labels_consumed_by_model": False,
            "observer_labels_are_detached_post_generation_evidence": True,
        },
        "training_performed": False,
        "optimizer_authorized": False,
    }
    return _seal(unsigned)


def build_master_receipt(
    *,
    public_challenge_file: Mapping[str, Any],
    private_transform_key_file: Mapping[str, Any],
    observer_receipt_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _build_master_from_bindings(
        public_challenge_file=public_challenge_file,
        private_transform_key_file=private_transform_key_file,
        observer_receipt_files=observer_receipt_files,
        replay_media=True,
    )


def validate_master_receipt(value: Any, *, replay_files: bool = True) -> dict[str, Any]:
    row = _closed(value, _MASTER_FIELDS, label="decoded temporal/event master receipt")
    _verify_seal(row, field="receipt_digest", label="decoded temporal/event master receipt")
    implementation = validate_file_binding(
        row["evaluator_implementation"],
        label="master evaluator implementation",
        verify_bytes=True,
    )
    if implementation != _module_binding():
        raise DecodedTemporalEventError("master evaluator implementation differs")
    rebuilt = _build_master_from_bindings(
        public_challenge_file=row["public_challenge_file"],
        private_transform_key_file=row["private_transform_key_file"],
        observer_receipt_files=row["observer_receipt_files"],
        replay_media=replay_files,
    )
    if row != rebuilt:
        raise DecodedTemporalEventError("master receipt does not replay exactly")
    return row


def _projection_unsigned(
    master: Mapping[str, Any], master_file: Mapping[str, Any], *, kind: str
) -> dict[str, Any]:
    checked = validate_master_receipt(master, replay_files=True)
    binding = validate_file_binding(
        master_file, label="endpoint master receipt", verify_bytes=True
    )
    loaded, _ = _load_bound_json(binding, label="endpoint master receipt")
    if loaded != checked:
        raise DecodedTemporalEventError("endpoint master receipt bytes differ")
    common = {
        "candidate_id": checked["candidate_id"],
        "analysis_split": "fit",
        "action_family_id": checked["action_family_id"],
        "source_video_sha256": checked["source_video_sha256"],
        "complete_caption_sha256": checked["complete_caption_sha256"],
        "candidate_mp4_sha256": checked["candidate_mp4_sha256"],
        "rollout_receipt_digest": checked["rollout_receipt_digest"],
        "evaluator_implementation": checked["evaluator_implementation"],
        "master_receipt_file": binding,
        "master_receipt_digest": checked["receipt_digest"],
        "frame_count": FRAME_COUNT,
        "evidence_valid": checked["evidence_valid"],
    }
    if kind == "temporal":
        return {
            "schema_version": TEMPORAL_PROJECTION_SCHEMA,
            **common,
            "probe_bank_digest": object_sha256(
                {
                    "event_spec_digest": checked["event_spec_digest"],
                    "observer_authority_digest_order": checked[
                        "observer_authority_digest_order"
                    ],
                }
            ),
            "branch_energy_by_name": checked["branch_energy_by_name"],
        }
    if kind == "event81":
        traces = checked["chronological_event_probability_by_frame"]
        return {
            "schema_version": EVENT81_PROJECTION_SCHEMA,
            **common,
            "frame_indices": list(range(FRAME_COUNT)),
            "start_probability_by_frame": traces["start"],
            "transition_probability_by_frame": traces["transition"],
            "terminal_probability_by_frame": traces["terminal"],
            "terminal_hold_probability_by_frame": traces["terminal_hold"],
        }
    raise DecodedTemporalEventError("unknown endpoint projection kind")


def make_endpoint_projections(
    master: Mapping[str, Any], *, master_file: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    temporal = _seal(_projection_unsigned(master, master_file, kind="temporal"))
    event81 = _seal(_projection_unsigned(master, master_file, kind="event81"))
    return temporal, event81


def validate_temporal_projection(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "source_video_sha256",
        "complete_caption_sha256",
        "candidate_mp4_sha256",
        "rollout_receipt_digest",
        "evaluator_implementation",
        "master_receipt_file",
        "master_receipt_digest",
        "frame_count",
        "evidence_valid",
        "probe_bank_digest",
        "branch_energy_by_name",
        "receipt_digest",
    }
    row = _closed(value, fields, label="temporal endpoint projection")
    _verify_seal(row, field="receipt_digest", label="temporal endpoint projection")
    master_raw, _ = _load_bound_json(
        row["master_receipt_file"], label="temporal projection master"
    )
    master = validate_master_receipt(master_raw, replay_files=True)
    expected = _seal(
        _projection_unsigned(master, row["master_receipt_file"], kind="temporal")
    )
    if row != expected:
        raise DecodedTemporalEventError("temporal endpoint projection differs")
    return row


def validate_event81_projection(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "source_video_sha256",
        "complete_caption_sha256",
        "candidate_mp4_sha256",
        "rollout_receipt_digest",
        "evaluator_implementation",
        "master_receipt_file",
        "master_receipt_digest",
        "frame_count",
        "evidence_valid",
        "frame_indices",
        "start_probability_by_frame",
        "transition_probability_by_frame",
        "terminal_probability_by_frame",
        "terminal_hold_probability_by_frame",
        "receipt_digest",
    }
    row = _closed(value, fields, label="event81 endpoint projection")
    _verify_seal(row, field="receipt_digest", label="event81 endpoint projection")
    master_raw, _ = _load_bound_json(
        row["master_receipt_file"], label="event81 projection master"
    )
    master = validate_master_receipt(master_raw, replay_files=True)
    expected = _seal(
        _projection_unsigned(master, row["master_receipt_file"], kind="event81")
    )
    if row != expected:
        raise DecodedTemporalEventError("event81 endpoint projection differs")
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--rollout-receipt", required=True)
    prepare.add_argument("--event-spec", required=True)
    prepare.add_argument("--blind-salt", required=True)
    prepare.add_argument("--preparer-id", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--ffmpeg", default="ffmpeg")
    prepare.add_argument("--ffprobe", default="ffprobe")

    seal = subparsers.add_parser("seal")
    seal.add_argument("--public-challenge", required=True)
    seal.add_argument("--private-transform-key", required=True)
    seal.add_argument("--observer-receipt", action="append", required=True)
    seal.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare_challenge(
            rollout_receipt_path=args.rollout_receipt,
            event_spec_path=args.event_spec,
            blind_salt_path=args.blind_salt,
            preparer_id=args.preparer_id,
            output_dir=args.output_dir,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
        )
        return 0
    if args.command == "seal":
        output = _fresh_directory(args.output_dir, label="seal output")
        output.mkdir(mode=0o700)
        master = build_master_receipt(
            public_challenge_file=file_binding(args.public_challenge),
            private_transform_key_file=file_binding(args.private_transform_key),
            observer_receipt_files=[file_binding(path) for path in args.observer_receipt],
        )
        master_path = output / "decoded-temporal-event-master-v1.json"
        _write_create_only(master_path, master, private=True)
        temporal, event81 = make_endpoint_projections(
            master, master_file=file_binding(master_path)
        )
        _write_create_only(output / "temporal-counterfactual-v2.json", temporal)
        _write_create_only(output / "event81-v2.json", event81)
        return 0
    raise DecodedTemporalEventError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DecodedTemporalEventError",
    "ENERGY_EPSILON",
    "EVIDENCE_ORDER",
    "EVENT81_PROJECTION_SCHEMA",
    "EVENT_SPEC_SCHEMA",
    "FPS",
    "FRAME_COUNT",
    "FRAME_WINDOWS",
    "MASTER_RECEIPT_SCHEMA",
    "MAX_MEAN_ABSOLUTE_OBSERVER_DISAGREEMENT",
    "MINIMUM_FRAME_STATE_ARGMAX_AGREEMENT",
    "MINIMUM_INDEPENDENT_OBSERVERS",
    "OBSERVER_RECEIPT_SCHEMA",
    "OBSERVER_REGISTRATION_SCHEMA",
    "PRIVATE_KEY_SCHEMA",
    "PUBLIC_CHALLENGE_SCHEMA",
    "SHUFFLE_PHASE_ORDER",
    "STATE_ORDER",
    "TEMPORAL_PROJECTION_SCHEMA",
    "TRANSFORM_ORDER",
    "apply_frame_map",
    "build_master_receipt",
    "canonical_json_bytes",
    "decode_exact81_rgb24",
    "file_binding",
    "file_sha256",
    "main",
    "make_endpoint_projections",
    "make_event_spec",
    "make_observer_registration",
    "object_sha256",
    "prepare_challenge",
    "probe_exact81_video",
    "temporal_index_map",
    "validate_event81_projection",
    "validate_event_spec",
    "validate_file_binding",
    "validate_master_receipt",
    "validate_observer_receipt",
    "validate_observer_registration",
    "validate_prepared_challenge",
    "validate_temporal_projection",
]
