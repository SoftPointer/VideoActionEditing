#!/usr/bin/env python3
"""Seal SAIC's text-only forward/reverse/no-op Bernini event bank.

The immutable SAIC source manifest contains real source-video paths because it
will eventually drive an editor.  This pure-T2V bank deliberately does not.
It projects only text-bound identity/scene descriptions, typed event text,
rollout seeds, and non-semantic geometry metadata into sixty preregistered
Bernini-R 1.3B proposals.  A launch-local constant-black exact81 clip supplies
the legacy native runner's spatial-bucket argument; no real source path or
source RGB is present in either the sealed bank or a candidate envelope.

Rendering is proposal collection only.  Neither this module nor its receipts
can qualify an event, select a seed, create a training target, or authorize an
optimizer update.
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
from typing import Any, Iterable, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import build_saic_reversible_source_set_v1 as source_set  # noqa: E402


SCHEMA_VERSION = "bernini-saic-pure-t2v-event-bank-spec-v1"
CANDIDATE_SCHEMA_VERSION = "bernini-saic-pure-t2v-event-candidate-v1"
PLAN_SCHEMA_VERSION = "bernini-saic-pure-t2v-event-plan-v1"
PROXY_RECEIPT_SCHEMA_VERSION = "bernini-saic-black-geometry-proxy-receipt-v1"
ASSET_PATH = METHOD_ROOT / "assets" / "saic_pure_t2v_event_bank_v1.json"
SOURCE_MANIFEST_CONTENT_SHA256 = (
    "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f"
)

BRANCH_ORDER = ("forward", "reverse", "noop")
GROUP_LAYOUT = (
    ("sp4-a", "dog", [0, 1, 2, 3]),
    ("sp4-b", "human", [4, 5, 6, 7]),
)
FRAME_COUNT = 81
FPS = 25
INFERENCE_STEPS = 40
ULYSSES_SIZE = 4

SAMPLING_CONTRACT = {
    "model": "Bernini-R-1.3B-Diffusers",
    "native_arm": "t2v",
    "guidance_mode": "t2v_apg",
    "num_frames": FRAME_COUNT,
    "latent_frames": 21,
    "fps": FPS,
    "num_inference_steps": INFERENCE_STEPS,
    "ulysses_size": ULYSSES_SIZE,
    "target_initialization": "official_gen_wanx22_fresh_gaussian",
    "same_row_seed_gaussian_across_branches": True,
}
SEMANTIC_INPUT_CLOSURE = {
    "accepted_semantic_inputs": [
        "identity_scene_caption",
        "branch_start_state_caption",
        "branch_instruction",
    ],
    "real_source_video_path_present": False,
    "real_source_rgb_read": False,
    "real_source_latent_read_or_created": False,
    "real_source_noise_read_or_created": False,
    "target_video": False,
    "reference_image_or_video": False,
    "mask_flow_pose_track_trajectory": False,
    "motion_donor": False,
    "generated_proposal_as_condition_target_donor_or_noise": False,
}
GEOMETRY_PROXY_CONTRACT = {
    "content": "constant_black_frames_created_without_source_media",
    "num_frames": FRAME_COUNT,
    "fps": FPS,
    "audio": False,
    "role": "legacy_native_runner_bucket_shape_only",
    "pixels_enter_transformer": False,
    "vae_latent_created": False,
    "source_path_or_bytes_used_to_create_proxy": False,
    "proxy_sha_bound_before_gpu_render": True,
}
ARTIFACT_AUTHORITY = {
    "proposal_media_requires_detached_full81_audit": True,
    "event_verified": False,
    "identity_preservation_verified": False,
    "seed_selection_authorized": False,
    "training_target_authorized": False,
    "optimizer_update_authorized": False,
}

# These descriptions are a sealed, text-only projection of the source caption,
# Qwen i0 string, and the already-authored forward/inverse/no-op instructions.
# They intentionally omit the q0 pose so the reverse branch does not receive a
# contradictory starting state.  No source media is opened by this projection.
TEXT_PROJECTIONS: Mapping[str, Mapping[str, str]] = {
    "7b88a1ca1f804f41": {
        "identity_scene_caption": (
            "A single gray French Bulldog with a black harness and attached leash, "
            "upright ears, and visible tongue is in yellow autumn leaves, facing "
            "slightly left of the camera. Its coat, body proportions, harness, "
            "leash, leaves, lighting, scale, and locked-off framing stay stable."
        ),
        "q0_caption": (
            "At frame 0 the French Bulldog is standing stably on all four legs at "
            "one fixed location."
        ),
        "q1_caption": (
            "At frame 0 the French Bulldog is already in a stable seated pose at "
            "that fixed location and facing direction."
        ),
    },
    "841b5e0080a1441d": {
        "identity_scene_caption": (
            "A single black-and-tan shepherd with perked ears, a collar, and an "
            "attached leash is on a grassy field looking toward the camera. Its "
            "coat pattern, body proportions, collar, leash, field, scale, lighting, "
            "and locked-off framing stay stable."
        ),
        "q0_caption": (
            "At frame 0 the shepherd is standing stably on all four legs, with only "
            "minor natural head and ear motion."
        ),
        "q1_caption": (
            "At frame 0 the shepherd is already seated stably on the grass facing "
            "the camera."
        ),
    },
    "a35b590961d24694": {
        "identity_scene_caption": (
            "A brown-haired woman wearing a black jacket, pink top, cream trousers, "
            "and black platform boots is centered against a warm yellow studio "
            "background. Her face, hair, body proportions, clothing, background, "
            "scale, lighting, and locked-off framing stay stable."
        ),
        "q0_caption": (
            "At frame 0 the woman holds a one-knee kneeling pose, with the right "
            "knee on the floor and the left leg forward."
        ),
        "q1_caption": (
            "At frame 0 the woman is already in a stable upright standing pose at "
            "the same studio location."
        ),
    },
    "31c34509415745ca": {
        "identity_scene_caption": (
            "A smiling red-braided subject wearing a black outfit, shoes, and bright "
            "socks is on a shopping-mall floor. The face, red braids, body "
            "proportions, clothing, shoes, mall background, scale, lighting, and "
            "locked-off framing stay stable."
        ),
        "q0_caption": (
            "At frame 0 the subject kneels on the right knee with the left foot "
            "planted and begins a natural outward arm gesture."
        ),
        "q1_caption": (
            "At frame 0 the subject is already standing upright at the same mall "
            "location while smiling with a natural outward arm position."
        ),
    },
    "99cde432839f4240": {
        "identity_scene_caption": (
            "A single pointed-ear dog with its tongue visible is in a monochrome "
            "forest, facing forward with its tail curved upward. Its face, coat, "
            "body proportions, forest background, monochrome appearance, scale, "
            "lighting, and locked-off framing stay stable."
        ),
        "q0_caption": (
            "At frame 0 the dog is standing stably on all four legs, allowing only "
            "minor facial and tail motion."
        ),
        "q1_caption": (
            "At frame 0 the dog is already in a stable seated pose at the same "
            "forest location, with its tail relaxed toward the ground."
        ),
    },
    "6ea45d35943742bb": {
        "identity_scene_caption": (
            "A single white dog with a collar, upright tail, perked ears, open mouth, "
            "and visible tongue is on grass under tree shadows, initially facing "
            "forward. Its face, white coat, body proportions, collar, grass, tree "
            "shadows, scale, lighting, and locked-off framing stay stable."
        ),
        "q0_caption": (
            "At frame 0 the white dog is standing stably on all four legs and then "
            "begins a slow natural head turn to the right."
        ),
        "q1_caption": (
            "At frame 0 the white dog is already seated stably at the same location "
            "and then begins the same slow natural head turn to the right."
        ),
    },
    "311c82f83eca4a7f": {
        "identity_scene_caption": (
            "A blond tattooed subject wearing the same gym clothing holds a "
            "smartphone in the left hand in a gym-mirror selfie scene. The face, "
            "hair, tattoos, body proportions, clothing, phone, mirror background, "
            "scale, lighting, and locked-off framing stay stable."
        ),
        "q0_caption": (
            "At frame 0 the subject kneels on the right knee with the left foot "
            "planted while the right arm begins a bicep flex."
        ),
        "q1_caption": (
            "At frame 0 the subject is already standing upright at the same gym "
            "location, holding the phone while beginning the same right-arm flex."
        ),
    },
    "6d346c38cf504493": {
        "identity_scene_caption": (
            "A blue-wig cosplay subject in the same costume safely holds two white "
            "cosplay prop pistols on a convention photo set with background people "
            "and equipment. The face, blue wig, body proportions, costume, props, "
            "background, scale, lighting, and locked-off framing stay stable."
        ),
        "q0_caption": (
            "At frame 0 the subject kneels on the right knee with the left foot "
            "planted while keeping a safe grip on both white props."
        ),
        "q1_caption": (
            "At frame 0 the subject is already standing upright at the same photo-set "
            "location while keeping the same safe grip on both white props."
        ),
    },
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,191}$")


class SAICPureT2VEventBankError(RuntimeError):
    """Raised before an ambiguous or privileged event-bank artifact is used."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise SAICPureT2VEventBankError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_create_only(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Creation modes are filtered through the process umask.  Formal AUH
        # launchers deliberately use ``umask 077``, so creating a purported
        # 0444 receipt directly leaves it at 0400.  Rendezvous readers use the
        # 0600 -> 0444 transition as the same-inode publication signal: keep
        # the file private while it is incomplete, then set the requested
        # terminal mode explicitly after the bytes are durable.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise SAICPureT2VEventBankError(f"refusing to overwrite {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SAICPureT2VEventBankError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: Any, *, label: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SAICPureT2VEventBankError(f"{label} contains non-finite float")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, label=f"{label}[{index}]")


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise SAICPureT2VEventBankError(f"{label} must be an absolute plain file")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SAICPureT2VEventBankError(f"cannot read {label}") from error
    if type(value) is not dict:
        raise SAICPureT2VEventBankError(f"{label} must contain one object")
    _reject_nonfinite(value, label=label)
    return value


def _closed(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise SAICPureT2VEventBankError(
            f"{label} keys differ: observed={observed!r}, expected={sorted(expected)!r}"
        )
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise SAICPureT2VEventBankError(f"{label} must be path-safe")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SAICPureT2VEventBankError(f"{label} must be lowercase SHA-256")
    return value


def _ascii_text(value: Any, *, label: str, minimum: int = 20) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum or "\x00" in value:
        raise SAICPureT2VEventBankError(f"{label} is absent or too short")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise SAICPureT2VEventBankError(f"{label} must be ASCII") from error
    return value


def _branch_instruction(row: Mapping[str, Any], branch: str) -> str:
    field = {"forward": "forward_instruction", "reverse": "inverse_instruction", "noop": "noop_instruction"}[branch]
    return str(row[field])


def _full_prompt(row: Mapping[str, Any], branch: str) -> str:
    projection = TEXT_PROJECTIONS[row["iid"]]
    start = projection["q1_caption"] if branch == "reverse" else projection["q0_caption"]
    return " ".join(
        (
            "An exactly 81-frame realistic video at 25 fps is one continuous shot with no cut.",
            projection["identity_scene_caption"],
            start,
            _branch_instruction(row, branch),
        )
    )


def author_spec(
    source_manifest_path: str | Path = source_set.ASSET_PATH,
) -> dict[str, Any]:
    source_path = Path(source_manifest_path).resolve(strict=True)
    manifest = source_set.load_manifest(source_path)
    summary = source_set.validate_manifest(manifest)
    if summary["manifest_content_sha256"] != SOURCE_MANIFEST_CONTENT_SHA256:
        raise SAICPureT2VEventBankError("immutable source manifest content changed")
    if set(TEXT_PROJECTIONS) != {row["iid"] for row in manifest["rows"]}:
        raise SAICPureT2VEventBankError("text projection coverage differs")

    groups = []
    for group_id, actor_family, visible_gpus in GROUP_LAYOUT:
        candidates = []
        ordinal = 0
        for row in manifest["rows"]:
            if row["actor_family"] != actor_family:
                continue
            projection = TEXT_PROJECTIONS[row["iid"]]
            for seed in row["rollout_seeds"]:
                for branch in BRANCH_ORDER:
                    instruction = _branch_instruction(row, branch)
                    start_state = (
                        projection["q1_caption"]
                        if branch == "reverse"
                        else projection["q0_caption"]
                    )
                    full_prompt = _full_prompt(row, branch)
                    candidates.append(
                        {
                            "candidate_id": f"saic-{row['iid']}-{branch}-s{seed}",
                            "ordinal": ordinal,
                            "row_id": row["row_id"],
                            "iid": row["iid"],
                            "analysis_split": row["analysis_split"],
                            "actor_family": actor_family,
                            "action_family_id": row["action_family_id"],
                            "initial_state_type": row["initial_state_type"],
                            "terminal_state_type": row["terminal_state_type"],
                            "source_media_sha256_for_nonuse_audit": row[
                                "source_video_sha256"
                            ],
                            "source_geometry_hw": [
                                row["media_probe"]["height"],
                                row["media_probe"]["width"],
                            ],
                            "source_caption_utf8_sha256": text_sha256(
                                row["source_caption"]
                            ),
                            "identity_scene_caption": projection[
                                "identity_scene_caption"
                            ],
                            "identity_scene_caption_utf8_sha256": text_sha256(
                                projection["identity_scene_caption"]
                            ),
                            "branch": branch,
                            "branch_start_state_caption": start_state,
                            "branch_start_state_caption_utf8_sha256": text_sha256(
                                start_state
                            ),
                            "branch_instruction": instruction,
                            "branch_instruction_utf8_sha256": text_sha256(instruction),
                            "full_t2v_caption": full_prompt,
                            "full_t2v_caption_utf8_sha256": text_sha256(full_prompt),
                            "seed": seed,
                            "event_audit_status": "pending_detached_full81_review",
                            "event_verified": False,
                            "identity_preservation_verified": False,
                            "optimizer_authorized": False,
                        }
                    )
                    ordinal += 1
        groups.append(
            {
                "group_id": group_id,
                "actor_family": actor_family,
                "visible_gpus": visible_gpus,
                "candidates": candidates,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "bank_id": "saic-text-only-forward-reverse-noop-exact81-v1",
        "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
        "source_manifest_file_sha256": file_sha256(source_path),
        "sampling_contract": SAMPLING_CONTRACT,
        "semantic_input_closure": SEMANTIC_INPUT_CLOSURE,
        "geometry_proxy_contract": GEOMETRY_PROXY_CONTRACT,
        "artifact_authority": ARTIFACT_AUTHORITY,
        "branch_order": list(BRANCH_ORDER),
        "groups": groups,
    }


_ROOT_FIELDS = {
    "schema_version",
    "bank_id",
    "source_manifest_content_sha256",
    "source_manifest_file_sha256",
    "sampling_contract",
    "semantic_input_closure",
    "geometry_proxy_contract",
    "artifact_authority",
    "branch_order",
    "groups",
}
_GROUP_FIELDS = {"group_id", "actor_family", "visible_gpus", "candidates"}
_CANDIDATE_FIELDS = {
    "candidate_id",
    "ordinal",
    "row_id",
    "iid",
    "analysis_split",
    "actor_family",
    "action_family_id",
    "initial_state_type",
    "terminal_state_type",
    "source_media_sha256_for_nonuse_audit",
    "source_geometry_hw",
    "source_caption_utf8_sha256",
    "identity_scene_caption",
    "identity_scene_caption_utf8_sha256",
    "branch",
    "branch_start_state_caption",
    "branch_start_state_caption_utf8_sha256",
    "branch_instruction",
    "branch_instruction_utf8_sha256",
    "full_t2v_caption",
    "full_t2v_caption_utf8_sha256",
    "seed",
    "event_audit_status",
    "event_verified",
    "identity_preservation_verified",
    "optimizer_authorized",
}
_ENVELOPE_FIELDS = {
    "schema_version",
    "root_spec_raw_sha256",
    "source_manifest_content_sha256",
    "group_id",
    "actor_family",
    "visible_gpus",
    "sampling_contract",
    "semantic_input_closure",
    "geometry_proxy_contract",
    "artifact_authority",
    "candidate",
    "geometry_proxy",
}


def validate_spec(
    value: Mapping[str, Any],
    *,
    source_manifest_path: str | Path = source_set.ASSET_PATH,
) -> dict[str, Any]:
    root = _closed(value, _ROOT_FIELDS, label="event spec")
    if (
        root["schema_version"] != SCHEMA_VERSION
        or root["bank_id"] != "saic-text-only-forward-reverse-noop-exact81-v1"
        or root["source_manifest_content_sha256"]
        != SOURCE_MANIFEST_CONTENT_SHA256
        or root["sampling_contract"] != SAMPLING_CONTRACT
        or root["semantic_input_closure"] != SEMANTIC_INPUT_CLOSURE
        or root["geometry_proxy_contract"] != GEOMETRY_PROXY_CONTRACT
        or root["artifact_authority"] != ARTIFACT_AUTHORITY
        or root["branch_order"] != list(BRANCH_ORDER)
    ):
        raise SAICPureT2VEventBankError("event spec root contract differs")

    source_path = Path(source_manifest_path).resolve(strict=True)
    manifest = source_set.load_manifest(source_path)
    source_summary = source_set.validate_manifest(manifest)
    if (
        source_summary["manifest_content_sha256"] != SOURCE_MANIFEST_CONTENT_SHA256
        or root["source_manifest_file_sha256"] != file_sha256(source_path)
    ):
        raise SAICPureT2VEventBankError("event spec source binding differs")
    expected = author_spec(source_path)
    if root != expected:
        raise SAICPureT2VEventBankError("event spec differs from sealed text projection")

    groups = root["groups"]
    if not isinstance(groups, list) or len(groups) != 2:
        raise SAICPureT2VEventBankError("two SP4 groups are required")
    ids: set[str] = set()
    count = 0
    for group, (group_id, actor, gpus) in zip(groups, GROUP_LAYOUT):
        _closed(group, _GROUP_FIELDS, label=f"group {group_id}")
        if (
            group["group_id"] != group_id
            or group["actor_family"] != actor
            or group["visible_gpus"] != gpus
            or len(group["candidates"]) != 30
        ):
            raise SAICPureT2VEventBankError(f"{group_id} layout differs")
        for ordinal, candidate in enumerate(group["candidates"]):
            _closed(candidate, _CANDIDATE_FIELDS, label=f"candidate {ordinal}")
            candidate_id = _safe_id(candidate["candidate_id"], label="candidate_id")
            if candidate_id in ids or candidate["ordinal"] != ordinal:
                raise SAICPureT2VEventBankError("candidate identity/order differs")
            ids.add(candidate_id)
            if candidate["actor_family"] != actor or candidate["branch"] not in BRANCH_ORDER:
                raise SAICPureT2VEventBankError("candidate group/branch differs")
            geometry = candidate["source_geometry_hw"]
            if (
                not isinstance(geometry, list)
                or len(geometry) != 2
                or any(type(item) is not int or item <= 0 for item in geometry)
            ):
                raise SAICPureT2VEventBankError("candidate source geometry differs")
            for text_field, sha_field in (
                ("identity_scene_caption", "identity_scene_caption_utf8_sha256"),
                ("branch_start_state_caption", "branch_start_state_caption_utf8_sha256"),
                ("branch_instruction", "branch_instruction_utf8_sha256"),
                ("full_t2v_caption", "full_t2v_caption_utf8_sha256"),
            ):
                text = _ascii_text(candidate[text_field], label=text_field)
                if text_sha256(text) != _sha(candidate[sha_field], label=sha_field):
                    raise SAICPureT2VEventBankError(f"{text_field} hash differs")
            if (
                type(candidate["seed"]) is not int
                or not 0 <= candidate["seed"] < 2**63
                or candidate["event_audit_status"]
                != "pending_detached_full81_review"
                or candidate["event_verified"] is not False
                or candidate["identity_preservation_verified"] is not False
                or candidate["optimizer_authorized"] is not False
            ):
                raise SAICPureT2VEventBankError("candidate authority differs")
            count += 1
    if count != 60:
        raise SAICPureT2VEventBankError("event bank must contain all 60 attempts")
    serialized = canonical_json_bytes(root)
    for row in manifest["rows"]:
        if row["source_video"].encode("ascii") in serialized:
            raise SAICPureT2VEventBankError("real source path leaked into event spec")
    return {
        "schema_version": SCHEMA_VERSION,
        "spec_content_sha256": object_sha256(root),
        "candidate_count": 60,
        "row_count": 8,
        "seed_cell_count": 20,
        "event_verified": False,
        "optimizer_authorized": False,
    }


def load_sealed_spec(
    path: str | Path,
    *,
    expected_raw_sha256: str,
    source_manifest_path: str | Path,
) -> tuple[dict[str, Any], str]:
    expected = _sha(expected_raw_sha256, label="expected spec raw SHA-256")
    source = Path(path)
    value = _load_json(source, label="sealed event spec")
    actual = file_sha256(source)
    if actual != expected:
        raise SAICPureT2VEventBankError("sealed event spec raw SHA-256 differs")
    validate_spec(value, source_manifest_path=source_manifest_path)
    return value, actual


def build_asset(
    *, source_manifest_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    value = author_spec(source_manifest_path)
    validate_spec(value, source_manifest_path=source_manifest_path)
    output = Path(output_path)
    _write_create_only(output, canonical_json_bytes(value) + b"\n")
    return {
        "output_path": str(output),
        "output_raw_sha256": file_sha256(output),
        **validate_spec(value, source_manifest_path=source_manifest_path),
    }


def _plain_executable(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if (
        not value.is_absolute()
        or not value.is_file()
        or value.is_symlink()
        or not os.access(value, os.X_OK)
    ):
        raise SAICPureT2VEventBankError(f"{label} must be an absolute executable")
    return value.resolve(strict=True)


def _ffprobe_exact81(ffprobe: Path, path: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        value = json.loads(subprocess.check_output(command, text=True))
        streams = value["streams"]
        if len(streams) != 1:
            raise ValueError("expected one video stream")
        stream = streams[0]
        result = {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "r_frame_rate": str(stream["r_frame_rate"]),
            "avg_frame_rate": str(stream["avg_frame_rate"]),
            "nb_frames": int(stream["nb_frames"]),
            "nb_read_frames": int(stream["nb_read_frames"]),
        }
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise SAICPureT2VEventBankError(f"ffprobe failed for {path}") from error
    if (
        result["r_frame_rate"] != "25/1"
        or result["avg_frame_rate"] != "25/1"
        or result["nb_frames"] != FRAME_COUNT
        or result["nb_read_frames"] != FRAME_COUNT
    ):
        raise SAICPureT2VEventBankError("geometry proxy is not exact81/25fps")
    return result


def materialize_geometry_proxies(
    *,
    spec: Mapping[str, Any],
    output_dir: str | Path,
    ffmpeg_path: str | Path,
    ffprobe_path: str | Path,
) -> dict[str, Any]:
    """Create launch-local black videos without reading any source file."""

    ffmpeg = _plain_executable(ffmpeg_path, label="ffmpeg")
    ffprobe = _plain_executable(ffprobe_path, label="ffprobe")
    output = Path(output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise SAICPureT2VEventBankError("proxy output must be a fresh absolute directory")
    output.mkdir(parents=False, exist_ok=False)
    geometries = sorted(
        {
            tuple(candidate["source_geometry_hw"])
            for group in spec["groups"]
            for candidate in group["candidates"]
        }
    )
    records = []
    for height, width in geometries:
        name = f"black-exact81-h{height}-w{width}.mp4"
        final = output / name
        temporary = output / f".{name}.partial.mp4"
        if final.exists() or temporary.exists():
            raise SAICPureT2VEventBankError("proxy path is not create-only")
        command = [
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r={FPS}",
            "-frames:v",
            str(FRAME_COUNT),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        try:
            subprocess.run(command, check=True)
            probe = _ffprobe_exact81(ffprobe, temporary)
            if probe["height"] != height or probe["width"] != width:
                raise SAICPureT2VEventBankError("geometry proxy dimensions differ")
            try:
                os.link(temporary, final)
            except FileExistsError as error:
                raise SAICPureT2VEventBankError(
                    "refusing concurrent geometry-proxy overwrite"
                ) from error
            temporary.unlink()
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
        records.append(
            {
                "height": height,
                "width": width,
                "path": str(final),
                "sha256": file_sha256(final),
                "probe": probe,
                "source_media_read": False,
            }
        )
    version = subprocess.check_output([str(ffmpeg), "-version"], text=True).splitlines()[0]
    unsigned = {
        "schema_version": PROXY_RECEIPT_SCHEMA_VERSION,
        "geometry_proxy_contract": GEOMETRY_PROXY_CONTRACT,
        "ffmpeg_path": str(ffmpeg),
        "ffmpeg_version_line": version,
        "ffprobe_path": str(ffprobe),
        "records": records,
        "source_media_paths_opened": [],
        "source_media_bytes_read": 0,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    receipt_path = output / "geometry-proxy-receipt.json"
    _write_create_only(receipt_path, canonical_json_bytes(receipt) + b"\n")
    return receipt


def load_proxy_receipt(path: str | Path) -> dict[str, Any]:
    receipt = _load_json(path, label="geometry proxy receipt")
    _closed(
        receipt,
        {
            "schema_version",
            "geometry_proxy_contract",
            "ffmpeg_path",
            "ffmpeg_version_line",
            "ffprobe_path",
            "records",
            "source_media_paths_opened",
            "source_media_bytes_read",
            "receipt_digest",
        },
        label="geometry proxy receipt",
    )
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if (
        receipt.get("schema_version") != PROXY_RECEIPT_SCHEMA_VERSION
        or receipt.get("geometry_proxy_contract") != GEOMETRY_PROXY_CONTRACT
        or receipt.get("source_media_paths_opened") != []
        or receipt.get("source_media_bytes_read") != 0
        or declared != object_sha256(unsigned)
    ):
        raise SAICPureT2VEventBankError("geometry proxy receipt contract differs")
    records = receipt.get("records")
    if not isinstance(records, list) or len(records) != 5:
        raise SAICPureT2VEventBankError("geometry proxy receipt requires five records")
    seen_geometry: set[tuple[int, int]] = set()
    for record in records:
        _closed(
            record,
            {"height", "width", "path", "sha256", "probe", "source_media_read"},
            label="geometry proxy record",
        )
        path_value = Path(record.get("path", ""))
        geometry = (record.get("height"), record.get("width"))
        if (
            any(type(item) is not int or item <= 0 for item in geometry)
            or geometry in seen_geometry
            or not path_value.is_absolute()
            or not path_value.is_file()
            or path_value.is_symlink()
            or file_sha256(path_value) != record.get("sha256")
            or record.get("source_media_read") is not False
        ):
            raise SAICPureT2VEventBankError("geometry proxy artifact differs")
        seen_geometry.add(geometry)
    return receipt


def materialize_plan(
    *,
    spec_path: str | Path,
    expected_spec_raw_sha256: str,
    source_manifest_path: str | Path,
    proxy_receipt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    spec, raw_sha = load_sealed_spec(
        spec_path,
        expected_raw_sha256=expected_spec_raw_sha256,
        source_manifest_path=source_manifest_path,
    )
    proxy_receipt = load_proxy_receipt(proxy_receipt_path)
    proxies = {
        (record["height"], record["width"]): record
        for record in proxy_receipt["records"]
    }
    required = {
        tuple(candidate["source_geometry_hw"])
        for group in spec["groups"]
        for candidate in group["candidates"]
    }
    if set(proxies) != required:
        raise SAICPureT2VEventBankError("proxy geometry coverage differs")
    output = Path(output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise SAICPureT2VEventBankError("plan output must be a fresh absolute directory")
    output.mkdir(parents=False, exist_ok=False)
    records = []
    for group in spec["groups"]:
        group_dir = output / group["group_id"]
        group_dir.mkdir()
        for candidate in group["candidates"]:
            proxy = proxies[tuple(candidate["source_geometry_hw"])]
            if proxy["sha256"] == candidate["source_media_sha256_for_nonuse_audit"]:
                raise SAICPureT2VEventBankError("geometry proxy aliases real source bytes")
            envelope = {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "root_spec_raw_sha256": raw_sha,
                "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
                "group_id": group["group_id"],
                "actor_family": group["actor_family"],
                "visible_gpus": group["visible_gpus"],
                "sampling_contract": SAMPLING_CONTRACT,
                "semantic_input_closure": SEMANTIC_INPUT_CLOSURE,
                "geometry_proxy_contract": GEOMETRY_PROXY_CONTRACT,
                "artifact_authority": ARTIFACT_AUTHORITY,
                "candidate": candidate,
                "geometry_proxy": {
                    "path": proxy["path"],
                    "sha256": proxy["sha256"],
                    "height": proxy["height"],
                    "width": proxy["width"],
                    "source_media_read": False,
                },
            }
            filename = f"{candidate['ordinal']:04d}-{candidate['candidate_id']}.json"
            path = group_dir / filename
            _write_create_only(path, canonical_json_bytes(envelope) + b"\n", mode=0o400)
            records.append(
                {
                    "group_id": group["group_id"],
                    "candidate_id": candidate["candidate_id"],
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )
    unsigned = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "root_spec_raw_sha256": raw_sha,
        "proxy_receipt_path": str(Path(proxy_receipt_path).resolve(strict=True)),
        "proxy_receipt_sha256": file_sha256(proxy_receipt_path),
        "candidate_count": len(records),
        "records": records,
        "event_verified": False,
        "optimizer_authorized": False,
    }
    manifest = {**unsigned, "manifest_digest": object_sha256(unsigned)}
    _write_create_only(
        output / "manifest.json", canonical_json_bytes(manifest) + b"\n", mode=0o400
    )
    return manifest


def load_candidate_envelope(
    path: str | Path, *, expected_root_spec_sha256: str
) -> dict[str, Any]:
    envelope = _closed(
        _load_json(path, label="candidate envelope"),
        _ENVELOPE_FIELDS,
        label="candidate envelope",
    )
    if (
        envelope["schema_version"] != CANDIDATE_SCHEMA_VERSION
        or envelope["root_spec_raw_sha256"]
        != _sha(expected_root_spec_sha256, label="root spec SHA-256")
        or envelope["source_manifest_content_sha256"]
        != SOURCE_MANIFEST_CONTENT_SHA256
        or envelope["sampling_contract"] != SAMPLING_CONTRACT
        or envelope["semantic_input_closure"] != SEMANTIC_INPUT_CLOSURE
        or envelope["geometry_proxy_contract"] != GEOMETRY_PROXY_CONTRACT
        or envelope["artifact_authority"] != ARTIFACT_AUTHORITY
    ):
        raise SAICPureT2VEventBankError("candidate envelope root contract differs")
    group_layout = {
        group_id: (actor, gpus) for group_id, actor, gpus in GROUP_LAYOUT
    }
    if group_layout.get(envelope["group_id"]) != (
        envelope["actor_family"],
        envelope["visible_gpus"],
    ):
        raise SAICPureT2VEventBankError("candidate envelope group differs")
    _closed(envelope["candidate"], _CANDIDATE_FIELDS, label="candidate")
    proxy = _closed(
        envelope["geometry_proxy"],
        {"path", "sha256", "height", "width", "source_media_read"},
        label="geometry proxy",
    )
    proxy_path = Path(proxy["path"])
    if (
        not proxy_path.is_absolute()
        or not proxy_path.is_file()
        or proxy_path.is_symlink()
        or file_sha256(proxy_path) != _sha(proxy["sha256"], label="proxy SHA-256")
        or proxy["source_media_read"] is not False
        or [proxy["height"], proxy["width"]]
        != envelope["candidate"]["source_geometry_hw"]
        or proxy["sha256"]
        == envelope["candidate"]["source_media_sha256_for_nonuse_audit"]
    ):
        raise SAICPureT2VEventBankError("candidate geometry proxy differs")
    return dict(envelope)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-asset")
    build.add_argument("--source-manifest", required=True)
    build.add_argument("--output", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--source-manifest", required=True)
    validate.add_argument("--spec", required=True)
    validate.add_argument("--expected-spec-raw-sha256", required=True)
    proxy = sub.add_parser("materialize-proxies")
    proxy.add_argument("--source-manifest", required=True)
    proxy.add_argument("--spec", required=True)
    proxy.add_argument("--expected-spec-raw-sha256", required=True)
    proxy.add_argument("--output-dir", required=True)
    proxy.add_argument("--ffmpeg", required=True)
    proxy.add_argument("--ffprobe", required=True)
    plan = sub.add_parser("materialize-plan")
    plan.add_argument("--source-manifest", required=True)
    plan.add_argument("--spec", required=True)
    plan.add_argument("--expected-spec-raw-sha256", required=True)
    plan.add_argument("--proxy-receipt", required=True)
    plan.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-asset":
        result = build_asset(
            source_manifest_path=args.source_manifest, output_path=args.output
        )
    elif args.command == "validate":
        spec, raw_sha = load_sealed_spec(
            args.spec,
            expected_raw_sha256=args.expected_spec_raw_sha256,
            source_manifest_path=args.source_manifest,
        )
        result = {**validate_spec(spec, source_manifest_path=args.source_manifest), "raw_sha256": raw_sha}
    elif args.command == "materialize-proxies":
        spec, _ = load_sealed_spec(
            args.spec,
            expected_raw_sha256=args.expected_spec_raw_sha256,
            source_manifest_path=args.source_manifest,
        )
        result = materialize_geometry_proxies(
            spec=spec,
            output_dir=args.output_dir,
            ffmpeg_path=args.ffmpeg,
            ffprobe_path=args.ffprobe,
        )
    else:
        result = materialize_plan(
            spec_path=args.spec,
            expected_spec_raw_sha256=args.expected_spec_raw_sha256,
            source_manifest_path=args.source_manifest,
            proxy_receipt_path=args.proxy_receipt,
            output_dir=args.output_dir,
        )
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
