#!/usr/bin/env python3
"""Fail-closed registry and receipt contract for Stage-B checkpoint review.

The contract deliberately separates two authorities:

* the physical source-only v3 manifest decides whether an IID is genuinely in
  the Stage-B held-out split; and
* a fixed authoring registry binds that IID to a raw exact81 source video,
  full instructions and one seed before any checkpoint decode is inspected.

Historical ``fit`` examples cannot be relabelled as held out.  The four fixed
sentinels are real members of the source-only-v3 held-out split and cover an
animal, a person, a hand/object interaction and a physical emitter.  There is
no invented old/new cohort.  Every checkpoint emits ten *logical* review rows.
``correct`` and ``forward`` are an intentional alias pair: the first labels the
carrier-control axis and the second labels the typed-instruction axis; both
must reference the same physical decode.  This keeps the user's requested
ten-column review without spending a second identical model sample.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence

import clean_source_visual_context_training_v1 as source_data


AUTHORING_SCHEMA = "bernini-clean-source-visual-context-review-authoring-v2"
MANIFEST_SCHEMA = "bernini-clean-source-visual-context-review-manifest-v2"
SHARD_SCHEMA = "bernini-clean-source-visual-context-checkpoint-decode-shard-v2"
AGGREGATE_SCHEMA = "bernini-clean-source-visual-context-checkpoint-review-v2"
CHECKPOINT_STEPS = (0, 20, 40, 60, 80)
SENTINEL_ORDER = (
    "animal-dog-pick",
    "human-runner-jump",
    "hand-object-blueprint-roll",
    "emitter-fireworks-explode",
)
# These identities were frozen from the *actual* source-only-v3 held-out rows,
# before checkpoint decoding.  The two wrong-owner pairs deliberately have
# identical normalized latent geometry.  They do not claim to isolate owner
# from scene/entity: that confound is displayed in the review page.
SENTINEL_IDENTITIES: Mapping[str, Mapping[str, Any]] = {
    "animal-dog-pick": {
        "iid": "50b62816e1c2452a",
        "action_family": "pick",
        "diversity_role": "animal",
        "source_entity_type": "animal",
        "source_video_sha256": "611077ae50513f6d8651e3cbf5ce253983fee35a92d2e2bd114cd2d4f15ba491",
        "forward_instruction_sha256": "4712b63c6b18c1880a0d02b4923ebd461d838c56d4e3a4d5a980aeec60eb6116",
        "seed": 52005001,
        "wrong_owner_iid": "05df75552e354c57",
        "latent_shape": [1, 16, 21, 60, 62],
    },
    "human-runner-jump": {
        "iid": "1b0e34725c7648c4",
        "action_family": "jump",
        "diversity_role": "human",
        "source_entity_type": "person",
        "source_video_sha256": "45d8882e2b55da7d0a9de3dd8c11b8bd1bcf5379ab3db636e09d55f53e4c9be0",
        "forward_instruction_sha256": "6d66441e1894e50a073fb2701688214994ec5a864725d658296ab917e9566455",
        "seed": 52005002,
        "wrong_owner_iid": "0eb9a074ff834237",
        "latent_shape": [1, 16, 21, 70, 52],
    },
    "hand-object-blueprint-roll": {
        "iid": "05df75552e354c57",
        "action_family": "roll",
        "diversity_role": "hand-object-interaction",
        "source_entity_type": "person-hand",
        "source_video_sha256": "0a1d172b4833be0d3c60b53c39a1e546d6180a48a063dd219e5fab951899dfae",
        "forward_instruction_sha256": "188207bfdb4a662bfcfb9b3a7700af7eea5499605d1f176ab0bd4bbe11ade650",
        "seed": 52005003,
        "wrong_owner_iid": "50b62816e1c2452a",
        "latent_shape": [1, 16, 21, 60, 62],
    },
    "emitter-fireworks-explode": {
        "iid": "0eb9a074ff834237",
        "action_family": "explode",
        "diversity_role": "physical-emitter",
        "source_entity_type": "fluid_or_emitter",
        "source_video_sha256": "f7d48644d023c78daa9a0540ba8db06eba1a70a6af06e5f8462a85c3e900d24a",
        "forward_instruction_sha256": "16a109b8f968e7daf3d96d51bcd7b53fe4f6d5e130fe814f4dfbcf77c507ec57",
        "seed": 52005004,
        "wrong_owner_iid": "1b0e34725c7648c4",
        "latent_shape": [1, 16, 21, 70, 52],
    },
}
SOURCE_CONTROLS = ("correct", "carrier-off", "wrong-owner", "order-permutation")
TEXT_BRANCHES = (
    "noop",
    "forward",
    "reverse",
    "incomplete",
    "camera-only",
    "appearance-only",
)
LOGICAL_ARM_ORDER = (*SOURCE_CONTROLS, *TEXT_BRANCHES)
PHYSICAL_DECODE_ARMS = (
    "correct",
    "carrier-off",
    "wrong-owner",
    "order-permutation",
    "noop",
    "reverse",
    "incomplete",
    "camera-only",
    "appearance-only",
)
FRAME_COUNT = 81
FPS = 25
NUM_INFERENCE_STEPS = 40
WORLD_SIZE = 4
SP_SIZE = 4
MEMORY_INPUT_KINDS = ("clean_source", "same_noise_forward_noised_source")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
_FORBIDDEN_EVALUATION_KEYS = {
    "score",
    "scores",
    "scalar",
    "scalars",
    "reward",
    "rewards",
    "ranking",
    "rankings",
    "verdict",
    "verdicts",
    "selected",
    "selection",
}


class CheckpointReviewContractError(RuntimeError):
    """Raised before ambiguous evidence can enter the review page."""


def fail(message: str) -> NoReturn:
    raise CheckpointReviewContractError(message)


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
        raise CheckpointReviewContractError(
            "value is not finite canonical ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        fail(f"{label} must be non-empty text")
    return value.strip()


def _plain_file(value: Any, *, label: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CheckpointReviewContractError(f"{label} is unavailable") from error
    if resolved != path or not path.is_file() or path.is_symlink():
        fail(f"{label} must be one canonical plain file")
    return path


def _strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointReviewContractError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        fail(f"{label} root must be an object")
    return value


def _embedded_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    unsigned = dict(value)
    digest = _sha(unsigned.pop(field, None), label=f"{label} {field}")
    if object_sha256(unsigned) != digest:
        fail(f"{label} embedded digest differs")
    return digest


def _walk_forbidden_keys(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("_", "-")
            if normalized.replace("-", "") in {
                item.replace("-", "") for item in _FORBIDDEN_EVALUATION_KEYS
            }:
                fail(f"forbidden evaluator field at {path}.{key}")
            _walk_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden_keys(child, path=f"{path}[{index}]")


def _ffprobe_exact81(path: Path) -> Mapping[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate,r_frame_rate,codec_name,width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        root = json.loads(completed.stdout)
        streams = root.get("streams")
        stream = streams[0] if isinstance(streams, list) and len(streams) == 1 else None
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise CheckpointReviewContractError(f"cannot ffprobe {path}") from error
    if not isinstance(stream, Mapping):
        fail(f"{path} must contain exactly one video stream")
    try:
        frames = int(stream.get("nb_read_frames"))
        numerator, denominator = str(stream.get("avg_frame_rate")).split("/", 1)
        fps = float(numerator) / float(denominator)
        width, height = int(stream.get("width")), int(stream.get("height"))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise CheckpointReviewContractError(f"invalid ffprobe metadata for {path}") from error
    if frames != FRAME_COUNT or abs(fps - FPS) > 1.0e-9 or width <= 0 or height <= 0:
        fail(f"{path} must be full 81-frame 25-fps media")
    return {
        "frame_count": frames,
        "fps": FPS,
        "codec": _text(stream.get("codec_name"), label="video codec"),
        "width": width,
        "height": height,
    }


@dataclass(frozen=True)
class Sentinel:
    sentinel_id: str
    diversity_role: str
    source_entity_type: str
    iid: str
    action_family: str
    source_video: str
    source_video_sha256: str
    source_caption: str
    seed: int
    wrong_owner_iid: str
    wrong_owner_source_video_sha256: str
    instructions: Mapping[str, str]
    instruction_sha256: Mapping[str, str]
    latent_shape: tuple[int, int, int, int, int]
    source_posterior_path: str
    source_posterior_file_sha256: str

    def receipt(self) -> Mapping[str, Any]:
        return {
            "sentinel_id": self.sentinel_id,
            "diversity_role": self.diversity_role,
            "source_entity_type": self.source_entity_type,
            "iid": self.iid,
            "action_family": self.action_family,
            "source_video": self.source_video,
            "source_video_sha256": self.source_video_sha256,
            "source_caption": self.source_caption,
            "seed": self.seed,
            "wrong_owner_iid": self.wrong_owner_iid,
            "wrong_owner_source_video_sha256": self.wrong_owner_source_video_sha256,
            "instructions": dict(self.instructions),
            "instruction_sha256": dict(self.instruction_sha256),
            "latent_shape": list(self.latent_shape),
            "source_posterior_path": self.source_posterior_path,
            "source_posterior_file_sha256": self.source_posterior_file_sha256,
        }


def _validate_authoring(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    expected_root = {
        "schema_version",
        "authoring_id",
        "source_only_manifest",
        "raw_full644",
        "sentinels",
        "authority",
        "authoring_digest",
    }
    if set(value) != expected_root or value.get("schema_version") != AUTHORING_SCHEMA:
        fail("review authoring schema/fields differ")
    _embedded_digest(value, field="authoring_digest", label="review authoring")
    _text(value.get("authoring_id"), label="authoring_id")
    source_only = value.get("source_only_manifest")
    if (
        not isinstance(source_only, Mapping)
        or set(source_only)
        != {"path", "file_sha256", "manifest_digest", "selected_split"}
        or source_only.get("selected_split") != "heldout"
    ):
        fail("review authoring source-only authority differs")
    _sha(source_only.get("file_sha256"), label="source-only manifest file SHA")
    _sha(source_only.get("manifest_digest"), label="source-only manifest digest")
    raw_full644 = value.get("raw_full644")
    if (
        not isinstance(raw_full644, Mapping)
        or raw_full644
        != {
            "path": str(source_data.PINNED_RAW_PARQUET),
            "file_sha256": source_data.PINNED_RAW_PARQUET_SHA256,
            "safe_columns_read": [
                "iid",
                "group_id",
                "family",
                "inputs",
                "source_video_path",
                "source_video_declared_path",
                "source_video_sha256",
                "edit_instruction_sha256",
                "selection_gates_json",
                "strict_selection_gates_all_true",
            ],
            "videos_column_read": False,
            "target_video_path_read": False,
            "target_video_bytes_read": False,
            "target_video_copied": False,
            "synthetic_target_semantics_used": False,
        }
    ):
        fail("review authoring pinned raw/full644 authority differs")
    if value.get("authority") != {
        "fixed_before_checkpoint_decode": True,
        "quality_based_selection": False,
        "optimizer_access": False,
        "sentinel_rule": "fixed-actual-v3-heldout-diversity-four-v1",
        "forward_instruction_authority": "pinned-raw-full644-inputs-text",
        "typed_controls_manually_preregistered": True,
        "target_video_available_to_review": False,
    }:
        fail("review authoring authority differs")
    rows = value.get("sentinels")
    expected_row = {
        "sentinel_id",
        "diversity_role",
        "source_entity_type",
        "iid",
        "action_family",
        "source_video",
        "source_video_sha256",
        "source_caption",
        "seed",
        "wrong_owner_iid",
        "latent_shape",
        "instructions",
    }
    if (
        not isinstance(rows, list)
        or len(rows) != len(SENTINEL_ORDER)
        or tuple(row.get("sentinel_id") for row in rows if isinstance(row, Mapping))
        != SENTINEL_ORDER
        or any(not isinstance(row, Mapping) or set(row) != expected_row for row in rows)
    ):
        fail("review authoring must contain the exact ordered real held-out rows")
    for row in rows:
        sentinel_id = str(row["sentinel_id"])
        identity = SENTINEL_IDENTITIES.get(sentinel_id)
        if not isinstance(identity, Mapping) or any(
            row.get(field) != identity[field]
            for field in (
                "iid",
                "action_family",
                "diversity_role",
                "source_entity_type",
                "source_video_sha256",
                "seed",
                "wrong_owner_iid",
                "latent_shape",
            )
        ):
            fail(f"{sentinel_id} fixed held-out identity differs")
        _text(row.get("iid"), label=f"{sentinel_id} IID")
        _text(row.get("action_family"), label=f"{sentinel_id} action family")
        _sha(row.get("source_video_sha256"), label=f"{sentinel_id} source SHA")
        _text(row.get("source_caption"), label=f"{sentinel_id} source caption")
        if type(row.get("seed")) is not int or not 0 <= row["seed"] < 2**63:
            fail(f"{sentinel_id} seed differs")
        instructions = row.get("instructions")
        latent_shape = row.get("latent_shape")
        if (
            not isinstance(latent_shape, list)
            or len(latent_shape) != 5
            or tuple(latent_shape[:3]) != (1, 16, 21)
            or any(type(value) is not int or value <= 0 for value in latent_shape)
        ):
            fail(f"{sentinel_id} normalized latent shape differs")
        if (
            not isinstance(instructions, Mapping)
            or set(instructions) != set(TEXT_BRANCHES)
            or any(not _text(instructions.get(branch), label=f"{sentinel_id} {branch}") for branch in TEXT_BRANCHES)
            or len(set(instructions.values())) != len(TEXT_BRANCHES)
        ):
            fail(f"{sentinel_id} full typed instruction closure differs")
        if (
            hashlib.sha256(instructions["forward"].encode("utf-8")).hexdigest()
            != identity["forward_instruction_sha256"]
        ):
            fail(f"{sentinel_id} forward instruction is not the pinned raw text")
    if len({row["iid"] for row in rows}) != 4 or len({row["seed"] for row in rows}) != 4:
        fail("sentinel IID/seed values must be unique")
    by_iid = {row["iid"]: row for row in rows}
    for row in rows:
        wrong = by_iid.get(row["wrong_owner_iid"])
        if (
            not isinstance(wrong, Mapping)
            or wrong["iid"] == row["iid"]
            or wrong["latent_shape"] != row["latent_shape"]
        ):
            fail(
                f"{row['sentinel_id']} wrong owner must be a different registered "
                "held-out source at identical latent geometry"
            )
    return tuple(rows)


def materialize_manifest_value(
    *,
    source_only_manifest_path: str | Path,
    authoring_path: str | Path,
    verify_files: bool = True,
    verify_source_media: bool = True,
) -> Mapping[str, Any]:
    source_path = _plain_file(source_only_manifest_path, label="source-only manifest")
    author_path = _plain_file(authoring_path, label="review authoring")
    try:
        source_manifest = source_data.load_source_only_split_manifest(
            source_path, verify_files=verify_files
        )
    except Exception as error:
        raise CheckpointReviewContractError(str(error)) from error
    authoring = _strict_json(author_path, label="review authoring")
    authored_rows = _validate_authoring(authoring)
    if (
        authoring["source_only_manifest"]["path"] != str(source_path)
        or authoring["source_only_manifest"]["file_sha256"] != file_sha256(source_path)
        or authoring["source_only_manifest"]["manifest_digest"]
        != source_manifest.manifest_digest
    ):
        fail("review authoring is not bound to this source-only manifest")
    heldout = {row.iid: row for row in source_manifest.rows_for_split("heldout")}
    authored_by_iid = {row["iid"]: row for row in authored_rows}
    sentinels: list[Mapping[str, Any]] = []
    for authored in authored_rows:
        sentinel_id = str(authored["sentinel_id"])
        source_row = heldout.get(str(authored["iid"]))
        if source_row is None:
            fail(f"{sentinel_id} is not in the sealed Stage-B heldout split")
        if (
            source_row.action_family != authored["action_family"]
            or source_row.source_video_sha256 != authored["source_video_sha256"]
        ):
            fail(f"{sentinel_id} authoring/source-only identity differs")
        if verify_files:
            posterior_path = _plain_file(
                source_row.source_posterior_path,
                label=f"{sentinel_id} source-only posterior",
            )
            if file_sha256(posterior_path) != source_row.source_posterior_file_sha256:
                fail(f"{sentinel_id} source-only posterior bytes differ")
            try:
                parameters = source_data._decode_source_posterior_parameters(
                    posterior_path.read_bytes(), iid=source_row.iid
                )
            except Exception as error:
                raise CheckpointReviewContractError(str(error)) from error
            observed_latent_shape = [
                1,
                16,
                21,
                int(parameters.shape[3]),
                int(parameters.shape[4]),
            ]
            if observed_latent_shape != authored["latent_shape"]:
                fail(f"{sentinel_id} authored/posterior latent geometry differs")
        raw_video = _plain_file(authored["source_video"], label=f"{sentinel_id} source video")
        if verify_source_media:
            if file_sha256(raw_video) != source_row.source_video_sha256:
                fail(f"{sentinel_id} raw source bytes differ")
            media = _ffprobe_exact81(raw_video)
        else:
            media = {"frame_count": FRAME_COUNT, "fps": FPS, "not_verified": True}
        wrong = authored_by_iid[str(authored["wrong_owner_iid"])]
        instructions = {branch: str(authored["instructions"][branch]) for branch in TEXT_BRANCHES}
        sentinel = Sentinel(
            sentinel_id=sentinel_id,
            diversity_role=str(authored["diversity_role"]),
            source_entity_type=str(authored["source_entity_type"]),
            iid=source_row.iid,
            action_family=source_row.action_family,
            source_video=str(raw_video),
            source_video_sha256=source_row.source_video_sha256,
            source_caption=str(authored["source_caption"]),
            seed=int(authored["seed"]),
            wrong_owner_iid=str(wrong["iid"]),
            wrong_owner_source_video_sha256=str(wrong["source_video_sha256"]),
            instructions=instructions,
            instruction_sha256={
                branch: hashlib.sha256(instructions[branch].encode("utf-8")).hexdigest()
                for branch in TEXT_BRANCHES
            },
            latent_shape=tuple(int(value) for value in authored["latent_shape"]),
            source_posterior_path=source_row.source_posterior_path,
            source_posterior_file_sha256=source_row.source_posterior_file_sha256,
        )
        sentinels.append({**sentinel.receipt(), "source_media": media})
    unsigned = {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_id": str(authoring["authoring_id"]),
        "source_only_manifest": {
            "path": str(source_path),
            "file_sha256": file_sha256(source_path),
            "manifest_digest": source_manifest.manifest_digest,
            "selected_split": "heldout",
            "train_overlap_count": 0,
        },
        "authoring": {
            "path": str(author_path),
            "file_sha256": file_sha256(author_path),
            "authoring_digest": authoring["authoring_digest"],
            "fixed_before_checkpoint_decode": True,
            "raw_full644_file_sha256": source_data.PINNED_RAW_PARQUET_SHA256,
            "target_video_bytes_read": False,
        },
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "sentinel_order": list(SENTINEL_ORDER),
        "source_controls": list(SOURCE_CONTROLS),
        "text_branches": list(TEXT_BRANCHES),
        "logical_arm_order": list(LOGICAL_ARM_ORDER),
        "physical_decode_arms": list(PHYSICAL_DECODE_ARMS),
        "correct_forward_alias": {
            "logical_alias": True,
            "same_instruction": "forward",
            "same_source_control": "correct",
            "same_seed": True,
            "same_physical_mp4_required": True,
        },
        "sentinels": sentinels,
        "sampling": {
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "world_size": WORLD_SIZE,
            "sequence_parallel_size": SP_SIZE,
            "same_seed_within_sentinel_all_checkpoints_and_arms": True,
        },
        "authority": {
            "training_performed_by_review": False,
            "optimizer_present": False,
            "target_video_available": False,
            "feature_evaluator_present": False,
            "vlm_evaluator_present": False,
            "manual_video_review_required": True,
            "decoded_quality_claimed": False,
        },
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}


def load_manifest(
    path_value: str | Path,
    *,
    expected_file_sha256: Optional[str] = None,
    verify_files: bool = True,
) -> Mapping[str, Any]:
    path = _plain_file(path_value, label="checkpoint review manifest")
    observed_sha = file_sha256(path)
    if expected_file_sha256 is not None and observed_sha != _sha(
        expected_file_sha256, label="expected review manifest SHA"
    ):
        fail("checkpoint review manifest file SHA differs")
    value = _strict_json(path, label="checkpoint review manifest")
    expected_root = {
        "schema_version",
        "manifest_id",
        "source_only_manifest",
        "authoring",
        "checkpoint_steps",
        "sentinel_order",
        "source_controls",
        "text_branches",
        "logical_arm_order",
        "physical_decode_arms",
        "correct_forward_alias",
        "sentinels",
        "sampling",
        "authority",
        "manifest_digest",
    }
    if set(value) != expected_root or value.get("schema_version") != MANIFEST_SCHEMA:
        fail("checkpoint review manifest schema/fields differ")
    _embedded_digest(value, field="manifest_digest", label="checkpoint review manifest")
    if (
        value.get("checkpoint_steps") != list(CHECKPOINT_STEPS)
        or value.get("sentinel_order") != list(SENTINEL_ORDER)
        or value.get("source_controls") != list(SOURCE_CONTROLS)
        or value.get("text_branches") != list(TEXT_BRANCHES)
        or value.get("logical_arm_order") != list(LOGICAL_ARM_ORDER)
        or value.get("physical_decode_arms") != list(PHYSICAL_DECODE_ARMS)
    ):
        fail("checkpoint review fixed grid differs")
    authoring = value.get("authoring")
    if (
        not isinstance(authoring, Mapping)
        or set(authoring)
        != {
            "path",
            "file_sha256",
            "authoring_digest",
            "fixed_before_checkpoint_decode",
            "raw_full644_file_sha256",
            "target_video_bytes_read",
        }
        or authoring.get("fixed_before_checkpoint_decode") is not True
        or authoring.get("raw_full644_file_sha256")
        != source_data.PINNED_RAW_PARQUET_SHA256
        or authoring.get("target_video_bytes_read") is not False
    ):
        fail("checkpoint review authoring authority differs")
    _sha(authoring.get("file_sha256"), label="review authoring file SHA")
    _sha(authoring.get("authoring_digest"), label="review authoring digest")
    sentinels = value.get("sentinels")
    if (
        not isinstance(sentinels, list)
        or len(sentinels) != 4
        or tuple(row.get("sentinel_id") for row in sentinels if isinstance(row, Mapping))
        != SENTINEL_ORDER
    ):
        fail("checkpoint review sentinel closure differs")
    if len({row.get("iid") for row in sentinels}) != 4 or any(
        row.get("seed") is None for row in sentinels
    ):
        fail("checkpoint review sentinel identities differ")
    by_iid = {row.get("iid"): row for row in sentinels}
    for row in sentinels:
        identity = SENTINEL_IDENTITIES[row["sentinel_id"]]
        expected_fields = {
            "sentinel_id",
            "diversity_role",
            "source_entity_type",
            "iid",
            "action_family",
            "source_video",
            "source_video_sha256",
            "source_caption",
            "seed",
            "wrong_owner_iid",
            "wrong_owner_source_video_sha256",
            "instructions",
            "instruction_sha256",
            "latent_shape",
            "source_posterior_path",
            "source_posterior_file_sha256",
            "source_media",
        }
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_fields
            or any(
                row.get(field) != identity[field]
                for field in (
                    "iid",
                    "action_family",
                    "diversity_role",
                    "source_entity_type",
                    "source_video_sha256",
                    "seed",
                    "wrong_owner_iid",
                    "latent_shape",
                )
            )
            or not isinstance(row.get("instructions"), Mapping)
            or set(row["instructions"]) != set(TEXT_BRANCHES)
            or not isinstance(row.get("instruction_sha256"), Mapping)
            or set(row["instruction_sha256"]) != set(TEXT_BRANCHES)
        ):
            fail("checkpoint review instruction closure differs")
        latent_shape = row.get("latent_shape")
        if (
            not isinstance(latent_shape, list)
            or len(latent_shape) != 5
            or tuple(latent_shape[:3]) != (1, 16, 21)
            or any(type(value) is not int or value <= 0 for value in latent_shape)
        ):
            fail("checkpoint review latent geometry differs")
        for branch in TEXT_BRANCHES:
            text = _text(row["instructions"][branch], label=f"{row['sentinel_id']} {branch}")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["instruction_sha256"][branch]:
                fail("checkpoint review instruction SHA differs")
        if row["instruction_sha256"]["forward"] != identity["forward_instruction_sha256"]:
            fail("checkpoint review forward instruction is not pinned raw text")
        _sha(row.get("source_video_sha256"), label="sentinel source SHA")
        _sha(row.get("wrong_owner_source_video_sha256"), label="wrong-owner source SHA")
        if row["source_video_sha256"] == row["wrong_owner_source_video_sha256"]:
            fail("wrong owner aliases the correct source")
        wrong = by_iid.get(row["wrong_owner_iid"])
        if (
            not isinstance(wrong, Mapping)
            or wrong.get("source_video_sha256")
            != row["wrong_owner_source_video_sha256"]
            or wrong.get("latent_shape") != row["latent_shape"]
        ):
            fail("wrong owner is not the fixed equal-geometry source")
        if verify_files:
            source = _plain_file(row["source_video"], label=f"{row['sentinel_id']} source")
            posterior = _plain_file(
                row["source_posterior_path"], label=f"{row['sentinel_id']} source posterior"
            )
            if (
                file_sha256(source) != row["source_video_sha256"]
                or file_sha256(posterior) != row["source_posterior_file_sha256"]
            ):
                fail("checkpoint review sentinel file bytes differ")
    _walk_forbidden_keys(value)
    return value


def logical_record_key(step: int, sentinel_id: str, arm: str) -> str:
    if step not in CHECKPOINT_STEPS or sentinel_id not in SENTINEL_ORDER or arm not in LOGICAL_ARM_ORDER:
        fail("logical record coordinate differs")
    value = f"step-{step:08d}__{sentinel_id}__{arm}"
    if _SAFE.fullmatch(value) is None:
        fail("logical record key is unsafe")
    return value


def expected_logical_coordinates(step: int) -> tuple[tuple[str, str], ...]:
    if step not in CHECKPOINT_STEPS:
        fail("checkpoint step is outside exact cadence")
    return tuple((sentinel, arm) for sentinel in SENTINEL_ORDER for arm in LOGICAL_ARM_ORDER)


def validate_shard_receipt(
    value: Mapping[str, Any],
    *,
    expected_step: int,
    expected_manifest_digest: str,
    manifest_value: Optional[Mapping[str, Any]] = None,
    media_root: Optional[Path] = None,
    verify_media: bool = True,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail("checkpoint decode shard receipt root differs")
    _walk_forbidden_keys(value)
    expected_root = {
        "schema_version",
        "complete",
        "checkpoint",
        "review_manifest_digest",
        "memory_input_kind",
        "source_records",
        "native_records",
        "logical_records",
        "execution",
        "authority",
        "receipt_digest",
    }
    if (
        set(value) != expected_root
        or value.get("schema_version") != SHARD_SCHEMA
        or value.get("complete") is not True
        or expected_step not in CHECKPOINT_STEPS
        or value.get("review_manifest_digest") != _sha(
            expected_manifest_digest, label="expected review manifest digest"
        )
        or value.get("memory_input_kind") not in MEMORY_INPUT_KINDS
    ):
        fail("checkpoint decode shard root differs")
    _embedded_digest(value, field="receipt_digest", label="checkpoint decode shard")
    checkpoint = value.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint)
        != {
            "step",
            "logical_records_seen",
            "path",
            "file_sha256",
            "adapter_parameter_digest",
            "strict_load_succeeded",
        }
        or checkpoint.get("step") != expected_step
        or checkpoint.get("logical_records_seen") != expected_step * 8
        or checkpoint.get("strict_load_succeeded") is not True
    ):
        fail("checkpoint decode strict-load record differs")
    _sha(checkpoint.get("file_sha256"), label="checkpoint file SHA")
    _sha(checkpoint.get("adapter_parameter_digest"), label="adapter parameter digest")
    source_records = value.get("source_records")
    native_records = value.get("native_records")
    logical = value.get("logical_records")
    if (
        not isinstance(source_records, list)
        or tuple(row.get("sentinel_id") for row in source_records if isinstance(row, Mapping)) != SENTINEL_ORDER
        or not isinstance(native_records, list)
        or (expected_step == 0 and tuple(row.get("sentinel_id") for row in native_records if isinstance(row, Mapping)) != SENTINEL_ORDER)
        or (expected_step != 0 and native_records != [])
        or not isinstance(logical, list)
        or len(logical) != len(SENTINEL_ORDER) * len(LOGICAL_ARM_ORDER)
    ):
        fail("checkpoint decode source/native/logical record count differs")
    expected_coordinates = expected_logical_coordinates(expected_step)
    observed_coordinates = tuple((row.get("sentinel_id"), row.get("arm")) for row in logical if isinstance(row, Mapping))
    if observed_coordinates != expected_coordinates:
        fail("checkpoint decode logical record order differs")
    media_cache: dict[Path, Mapping[str, Any]] = {}

    def validate_media_record(record: Mapping[str, Any], *, label: str) -> None:
        required = {"relative_mp4", "mp4_sha256", "frame_count", "fps"}
        if not required.issubset(record) or record.get("frame_count") != FRAME_COUNT or record.get("fps") != FPS:
            fail(f"{label} exact81 media fields differ")
        _sha(record.get("mp4_sha256"), label=f"{label} MP4 SHA")
        relative = Path(str(record.get("relative_mp4")))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".mp4":
            fail(f"{label} MP4 path is unsafe")
        if verify_media:
            if media_root is None:
                fail("media_root is required for media verification")
            path = (media_root / relative).resolve(strict=True)
            root = media_root.resolve(strict=True)
            if path == root or root not in path.parents or path.is_symlink() or not path.is_file():
                fail(f"{label} MP4 escapes the shard root")
            if file_sha256(path) != record["mp4_sha256"]:
                fail(f"{label} MP4 bytes differ")
            metadata = media_cache.setdefault(path, _ffprobe_exact81(path))
            if metadata["frame_count"] != FRAME_COUNT or metadata["fps"] != FPS:
                fail(f"{label} MP4 metadata differs")

    source_by_id: dict[str, Mapping[str, Any]] = {}
    for row in source_records:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "sentinel_id",
                "iid",
                "diversity_role",
                "source_entity_type",
                "source_caption",
                "source_video_sha256",
                "wrong_owner_source_video_sha256",
                "seed",
                "relative_mp4",
                "mp4_sha256",
                "frame_count",
                "fps",
            }
            or row.get("source_video_sha256") != row.get("mp4_sha256")
        ):
            fail("source snapshot logical record differs")
        validate_media_record(row, label=f"source {row.get('sentinel_id')}")
        source_by_id[str(row["sentinel_id"])] = row
    for row in native_records:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "sentinel_id",
                "iid",
                "source_video_sha256",
                "seed",
                "instruction",
                "instruction_utf8_sha256",
                "route_trace_digest",
                "initial_gaussian_sha256",
                "relative_mp4",
                "mp4_sha256",
                "frame_count",
                "fps",
            }
            or row.get("sentinel_id") not in source_by_id
            or row.get("iid") != source_by_id[row["sentinel_id"]]["iid"]
            or row.get("source_video_sha256")
            != source_by_id[row["sentinel_id"]]["source_video_sha256"]
            or row.get("seed") != source_by_id[row["sentinel_id"]]["seed"]
        ):
            fail("native record differs")
        _sha(row.get("instruction_utf8_sha256"), label="native instruction SHA")
        _sha(row.get("route_trace_digest"), label="native route trace digest")
        _sha(row.get("initial_gaussian_sha256"), label="native initial Gaussian SHA")
        if hashlib.sha256(str(row.get("instruction")).encode("utf-8")).hexdigest() != row["instruction_utf8_sha256"]:
            fail("native instruction bytes differ")
        validate_media_record(row, label=f"native {row.get('sentinel_id')}")
    for row in logical:
        expected_fields = {
            "record_id",
            "checkpoint_step",
            "checkpoint_file_sha256",
            "sentinel_id",
            "iid",
            "diversity_role",
            "source_entity_type",
            "source_video_sha256",
            "seed",
            "arm",
            "axis",
            "source_control",
            "text_branch",
            "instruction",
            "instruction_utf8_sha256",
            "memory_source_video_sha256",
            "memory_transform",
            "route_trace_digest",
            "initial_gaussian_sha256",
            "relative_mp4",
            "mp4_sha256",
            "frame_count",
            "fps",
            "physical_decode_id",
        }
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            fail("logical checkpoint record fields differ")
        sentinel_id, arm = str(row["sentinel_id"]), str(row["arm"])
        source = source_by_id.get(sentinel_id)
        if (
            source is None
            or row.get("record_id") != logical_record_key(expected_step, sentinel_id, arm)
            or row.get("checkpoint_step") != expected_step
            or row.get("checkpoint_file_sha256") != checkpoint["file_sha256"]
            or row.get("iid") != source["iid"]
            or row.get("diversity_role") != source["diversity_role"]
            or row.get("source_entity_type") != source["source_entity_type"]
            or row.get("source_video_sha256") != source["source_video_sha256"]
            or row.get("seed") != source["seed"]
        ):
            fail("logical checkpoint/source/seed binding differs")
        _sha(row.get("instruction_utf8_sha256"), label="instruction SHA")
        if hashlib.sha256(str(row.get("instruction")).encode("utf-8")).hexdigest() != row["instruction_utf8_sha256"]:
            fail("logical checkpoint instruction bytes differ")
        _sha(row.get("route_trace_digest"), label="route trace digest")
        _sha(row.get("initial_gaussian_sha256"), label="initial Gaussian SHA")
        if arm in SOURCE_CONTROLS:
            expected_axis = "source-control"
            expected_source_control = arm
            expected_text_branch = "forward"
        else:
            expected_axis = "typed-instruction"
            expected_source_control = "correct"
            expected_text_branch = arm
        if (
            row.get("axis") != expected_axis
            or row.get("source_control") != expected_source_control
            or row.get("text_branch") != expected_text_branch
        ):
            fail("logical source-control/text-branch semantics differ")
        if expected_source_control == "carrier-off":
            expected_memory_sha = None
            expected_transform = None
        elif expected_source_control == "wrong-owner":
            expected_memory_sha = source["wrong_owner_source_video_sha256"]
            expected_transform = "identity"
        elif expected_source_control == "order-permutation":
            expected_memory_sha = source["source_video_sha256"]
            expected_transform = "reverse-phase-order-20-to-0"
        else:
            expected_memory_sha = source["source_video_sha256"]
            expected_transform = "identity"
        if (
            row.get("memory_source_video_sha256") != expected_memory_sha
            or row.get("memory_transform") != expected_transform
        ):
            fail("logical memory owner/transform binding differs")
        validate_media_record(row, label=f"logical {row['record_id']}")
    by_coordinate = {(row["sentinel_id"], row["arm"]): row for row in logical}
    for sentinel in SENTINEL_ORDER:
        correct = by_coordinate[(sentinel, "correct")]
        forward = by_coordinate[(sentinel, "forward")]
        if any(
            correct[field] != forward[field]
            for field in (
                "instruction",
                "instruction_utf8_sha256",
                "relative_mp4",
                "mp4_sha256",
                "physical_decode_id",
                "route_trace_digest",
                "initial_gaussian_sha256",
            )
        ):
            fail("correct/forward logical alias lost physical identity")
    execution = value.get("execution")
    authority = value.get("authority")
    if (
        execution
        != {
            "world_size": WORLD_SIZE,
            "sequence_parallel_size": SP_SIZE,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "same_seed_all_arms_within_sentinel": True,
            "same_source_all_checkpoints": True,
            "parent_allocation_released": False,
        }
        or authority
        != {
            "decoded_checkpoint_inference_executed": True,
            "optimizer_present": False,
            "backward_performed": False,
            "parameter_update": False,
            "feature_evaluator_present": False,
            "vlm_evaluator_present": False,
            "manual_review_pending": True,
            "quality_claimed": False,
        }
    ):
        fail("checkpoint decode execution/authority differs")
    if manifest_value is not None:
        if manifest_value.get("manifest_digest") != expected_manifest_digest:
            fail("provided review manifest digest differs")
        manifest_by_id = {
            row["sentinel_id"]: row for row in manifest_value.get("sentinels", [])
        }
        for sentinel_id, source in source_by_id.items():
            manifest_row = manifest_by_id.get(sentinel_id)
            if not isinstance(manifest_row, Mapping) or any(
                source[field] != manifest_row[field]
                for field in (
                    "iid",
                    "diversity_role",
                    "source_entity_type",
                    "source_caption",
                    "source_video_sha256",
                    "wrong_owner_source_video_sha256",
                    "seed",
                )
            ):
                fail("shard source record differs from fixed review manifest")
        for row in logical:
            manifest_row = manifest_by_id[row["sentinel_id"]]
            text_branch = row["text_branch"]
            if (
                row["instruction"] != manifest_row["instructions"][text_branch]
                or row["instruction_utf8_sha256"]
                != manifest_row["instruction_sha256"][text_branch]
            ):
                fail("shard instruction differs from fixed review manifest")
        for row in native_records:
            manifest_row = manifest_by_id[row["sentinel_id"]]
            if (
                row["instruction"] != manifest_row["instructions"]["forward"]
                or row["instruction_utf8_sha256"]
                != manifest_row["instruction_sha256"]["forward"]
            ):
                fail("native instruction differs from fixed forward instruction")
    return value


def write_create_only_json(path_value: str | Path, value: Mapping[str, Any]) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute() or path.exists() or path.is_symlink() or not path.parent.is_dir():
        fail("output JSON must be a fresh absolute file in an existing directory")
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


__all__ = [
    "AGGREGATE_SCHEMA",
    "AUTHORING_SCHEMA",
    "CHECKPOINT_STEPS",
    "CheckpointReviewContractError",
    "FPS",
    "FRAME_COUNT",
    "LOGICAL_ARM_ORDER",
    "MEMORY_INPUT_KINDS",
    "MANIFEST_SCHEMA",
    "NUM_INFERENCE_STEPS",
    "PHYSICAL_DECODE_ARMS",
    "SENTINEL_IDENTITIES",
    "SENTINEL_ORDER",
    "SHARD_SCHEMA",
    "SOURCE_CONTROLS",
    "SP_SIZE",
    "TEXT_BRANCHES",
    "WORLD_SIZE",
    "canonical_json_bytes",
    "expected_logical_coordinates",
    "file_sha256",
    "load_manifest",
    "logical_record_key",
    "materialize_manifest_value",
    "object_sha256",
    "validate_shard_receipt",
    "write_create_only_json",
]
