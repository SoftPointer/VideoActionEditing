#!/usr/bin/env python3
"""Closed BOX-EXP-013 arms incomplete-only repair exact2 plan.

The executable population is exactly two new ``incomplete`` calls, one for
each already-passing arms-action seed from BOX-EXP-011.  The action side of
each pair is external immutable authority and is physically checked when this
plan is built.  No action call, diagnostic call, training step, or optimizer
entrypoint is present here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence

import pair_v5_t2v_calibration_bank_spec as bank_contract


PLAN_SCHEMA = "bernini-full30-action-arms-incomplete-repair-exact2-plan-v1"
PLAN_ID = "BOX-EXP-013-arms-incomplete-only-repair-exact2-v1"
PLAN_FILENAME = "full30-action-arms-incomplete-repair-exact2-plan-v1.json"
DATASET = "fit_arms_incomplete_only_repair_exact2"
SEED1_SPEC_SHA256 = "2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab"
SEED2_SPEC_SHA256 = "0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e"
FROZEN_FIT_REPAIR_ARCHIVE_SHA256 = (
    "24456d098551f07e7bb634259dad95209258c4db0153aee305ae14ec28b05f43"
)
FROZEN_FIT_REPAIR_MANIFEST_SHA256 = (
    "fe7e6130c77de9b627428452b0bc3db3772689fc26efd79ee38cd0abee3c70d3"
)
FROZEN_FIT_REPAIR_REVISION = "603865eb19543b14222f85c6f6bceb19872717aa"
EXTERNAL_KEY_SHA256 = (
    "4c0864c7018b28b284a49d7134bce574e8d8fe47d5d795a71497b78fff446f8c"
)
EXTERNAL_REVIEW_RECEIPT_SHA256 = (
    "1b40da8dde07f348c2501adf3fd62fb528062053cde6e99c62f6d02e3ad8a4bc"
)
EXTERNAL_KEY_DIGEST = (
    "e12961e37a193c8c893c0b61e12cd7b5598cee6a1c32eb78bafc324777b21b1c"
)
EXTERNAL_PACKET_ID = "packet_473d776896869b6bcfea29684099f827"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_PROMPT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])(?:hands-on-hips|hips|hip)(?![A-Za-z])", re.IGNORECASE
)

INCOMPLETE_PROMPT = (
    "A continuous medium portrait shows a single adult woman with both arms clearly "
    "raised in the original simple one-subject scene. In one clear positive motion, "
    "the main woman lowers both raised arms into a symmetric lower-chest to mid-torso "
    "pose, with both elbows bent at about ninety degrees, both forearms held horizontal, "
    "and both hands visibly separated above the waist. She settles into this before-terminal "
    "pose early and holds it steadily through the final frame while the camera remains "
    "locked. The shot stays continuous, the illumination remains stable, and the final "
    "frame is temporally coherent."
)
INCOMPLETE_PROMPT_SHA256 = (
    "225d66cf0ad29fa7b7b51bf6177843629f2f8710d60b3278008495cbb049cde4"
)
PROMPTS: Mapping[str, Mapping[str, str]] = {
    "arms_incomplete": {
        "text": INCOMPLETE_PROMPT,
        "utf8_sha256": INCOMPLETE_PROMPT_SHA256,
    }
}
PROMPT_BUNDLE_SHA256 = (
    "6abd07b6f952171879790f34d1e908e79472405dbc6c4ac87290529d8426c102"
)

CELL_SPECS: tuple[Mapping[str, Any], ...] = (
    {
        "seed_slot": "seed1",
        "group_id": "sp4-a",
        "visible_gpus": [0, 1, 2, 3],
        "iid": "00435ad621c44fac",
        "seed": 2026080821,
        "source_candidate_id": (
            "pair5-t2v-reserve4-v1-00435ad621c44fac-incomplete"
        ),
        "candidate_id": (
            "pair5-t2v-arms-incomplete-repair-v1-seed1-"
            "00435ad621c44fac-incomplete"
        ),
        "external_action_candidate_id": (
            "pair5-t2v-fit-repair-v1-seed1-00435ad621c44fac-action"
        ),
    },
    {
        "seed_slot": "seed2",
        "group_id": "sp4-a",
        "visible_gpus": [0, 1, 2, 3],
        "iid": "00435ad621c44fac",
        "seed": 2026080921,
        "source_candidate_id": (
            "pair5-t2v-reserve4-seed2-00435ad621c44fac-incomplete"
        ),
        "candidate_id": (
            "pair5-t2v-arms-incomplete-repair-v1-seed2-"
            "00435ad621c44fac-incomplete"
        ),
        "external_action_candidate_id": (
            "pair5-t2v-fit-repair-v1-seed2-00435ad621c44fac-action"
        ),
    },
)

EXTERNAL_ACTION_PINS: Mapping[int, Mapping[str, str]] = {
    2026080821: {
        "candidate_id": CELL_SPECS[0]["external_action_candidate_id"],
        "sample_id": "4566e8433708577e1aca1131",
        "label": "formal_00",
        "mp4_sha256": (
            "6f07c69ba2a8ff613ce2c74accfc7578d63602a7e0bf86b486a0b1af9554330b"
        ),
        "native_receipt_sha256": (
            "9f37dbd1a1791ccb8704fde083b80f352b359d95e962c9b2bd2c8ef34db9ff00"
        ),
        "native_receipt_digest": (
            "5ea39f92365cfd0c2e9ec6025f42500b99214fd72a30054baf1ba3186837465f"
        ),
        "calibration_receipt_sha256": (
            "b1a5799d6087e0dbe7b329f9834152999f00d6f5a1886b80adfa1e7dde81d973"
        ),
        "calibration_receipt_digest": (
            "e8599208c4097c8a5386975fad0bdd30fe291f4395026723621ff019e1bf7ace"
        ),
        "remote_media_path": (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/"
            "bernini_generic_source_anchored_action_v1_20260814/data_prep/"
            "full30-action-fit-repair-exact8-r1-603865eb-j136140-r1/generation/"
            "sp4-a/pair5-t2v-fit-repair-v1-seed1-00435ad621c44fac-action/t2v.mp4"
        ),
        "gaussian_sha256": (
            "5daec46acf39a64afaf158e3dfd28fd744693b1137a7e2b4911df371bcdca1b3"
        ),
    },
    2026080921: {
        "candidate_id": CELL_SPECS[1]["external_action_candidate_id"],
        "sample_id": "2c0e0569d885f26d6196184a",
        "label": "formal_02",
        "mp4_sha256": (
            "6c1451ca0c85d151346a2efbacd740e88a113cbfafc5e2373ff97fba9dea6fbc"
        ),
        "native_receipt_sha256": (
            "c41622b36ec9e75522af5e038282d2399ab3cb0ecc58a8d04fd56a76aae09f3d"
        ),
        "native_receipt_digest": (
            "63e776f476ba9d075c76da5c41135bdbaa3d082bdcbc4dda5f4faf8679de83db"
        ),
        "calibration_receipt_sha256": (
            "2e0541aea47808949334d689a17d826f74bb0aa6061899aa5c9482fed9da06f0"
        ),
        "calibration_receipt_digest": (
            "fd9f7ecd561fbe06e84393c42b5bf0dfc8885c722be541951098dd3e7d84b7e2"
        ),
        "remote_media_path": (
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/"
            "bernini_generic_source_anchored_action_v1_20260814/data_prep/"
            "full30-action-fit-repair-exact8-r1-603865eb-j136140-r1/generation/"
            "sp4-a/pair5-t2v-fit-repair-v1-seed2-00435ad621c44fac-action/t2v.mp4"
        ),
        "gaussian_sha256": (
            "6c29c9d3e0cb830406e6dd62f6b7be2592aa10c1a8ac532110e68126023cdf7a"
        ),
    },
}

GAUSSIAN_IDENTITY_FIELDS = (
    "raw_value_sha256",
    "content_sha256",
    "shape",
    "dtype",
    "stored_dtype",
    "generator_initial_seed",
)


class ArmsIncompleteExact2PlanError(RuntimeError):
    """Raised before widened or unbound BOX-EXP-013 authority can pass."""


def fail(message: str) -> NoReturn:
    raise ArmsIncompleteExact2PlanError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ArmsIncompleteExact2PlanError(
            "value is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def plain_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata, resolved = path.lstat(), path.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2PlanError(f"{label} is unavailable") from error
    require(
        resolved == path
        and stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be one canonical plain file",
    )
    return resolved


def load_json(
    value: str | Path, label: str, expected_sha256: Optional[str] = None
) -> tuple[dict[str, Any], Path, str]:
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArmsIncompleteExact2PlanError(
            f"{label} cannot be opened without following links"
        ) from error

    def stable_fields(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_blocks,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    try:
        opened = os.fstat(descriptor)
        first_chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            first_chunks.append(chunk)
        middle = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second_chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            second_chunks.append(chunk)
        closed = os.fstat(descriptor)
    except OSError as error:
        raise ArmsIncompleteExact2PlanError(f"{label} stable read failed") from error
    finally:
        os.close(descriptor)
    try:
        named = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ArmsIncompleteExact2PlanError(
            f"{label} named identity is unavailable"
        ) from error
    raw = b"".join(first_chunks)
    second = b"".join(second_chunks)
    observed = hashlib.sha256(raw).hexdigest()
    require(
        resolved == path
        and stat.S_ISREG(opened.st_mode)
        and not stat.S_ISLNK(named.st_mode)
        and opened.st_nlink == 1
        and stable_fields(opened)
        == stable_fields(middle)
        == stable_fields(closed)
        == stable_fields(named)
        and raw == second
        and len(raw) == opened.st_size,
        f"{label} changed during its single-fd stable read",
    )
    if expected_sha256 is not None:
        require(
            SHA256_RE.fullmatch(expected_sha256) is not None
            and observed == expected_sha256,
            f"{label} SHA-256 differs",
        )
    try:
        result = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ArmsIncompleteExact2PlanError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArmsIncompleteExact2PlanError(f"{label} is not valid JSON") from error
    require(type(result) is dict, f"{label} must be an object")
    require(raw == canonical_json_bytes(result) + b"\n", f"{label} is not canonical JSON")
    return result, path, observed


def write_create_only(path: Path, raw: bytes, label: str) -> str:
    require(
        path.is_absolute()
        and path.parent.is_dir()
        and not path.parent.is_symlink()
        and not path.exists()
        and not path.is_symlink(),
        f"{label} must be a fresh absolute path",
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Retain any partial create-only artifact as poisoned evidence.  This
        # release never authorizes a public output path to be deleted.
        raise
    observed = hashlib.sha256(raw).hexdigest()
    require(file_sha256(path) == observed, f"{label} write replay differs")
    return observed


def _validate_signed_receipt(
    value: Mapping[str, Any], expected_digest: str, label: str
) -> None:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    require(
        declared == expected_digest
        and SHA256_RE.fullmatch(expected_digest) is not None
        and object_sha256(unsigned) == expected_digest,
        f"{label} digest differs",
    )


def _assert_prompt_freeze() -> None:
    observed_tokens = FORBIDDEN_PROMPT_TOKEN_RE.findall(INCOMPLETE_PROMPT)
    require(not observed_tokens, "incomplete prompt contains a forbidden terminal token")
    require(
        hashlib.sha256(INCOMPLETE_PROMPT.encode("utf-8")).hexdigest()
        == INCOMPLETE_PROMPT_SHA256,
        "incomplete prompt SHA differs",
    )
    require(object_sha256(PROMPTS) == PROMPT_BUNDLE_SHA256, "prompt bundle SHA differs")
    required_positive_phrases = (
        "symmetric lower-chest to mid-torso pose",
        "elbows bent at about ninety degrees",
        "forearms held horizontal",
        "hands visibly separated above the waist",
        "settles into this before-terminal pose early",
        "holds it steadily through the final frame",
    )
    require(
        all(phrase in INCOMPLETE_PROMPT for phrase in required_positive_phrases),
        "positive before-terminal pose contract differs",
    )


def _gaussian_identity(artifact: Mapping[str, Any]) -> dict[str, Any]:
    result = {field: artifact.get(field) for field in GAUSSIAN_IDENTITY_FIELDS}
    require(
        SHA256_RE.fullmatch(str(result["raw_value_sha256"])) is not None
        and SHA256_RE.fullmatch(str(result["content_sha256"])) is not None
        and result["shape"] == [1, 16, 21, 74, 50]
        and result["dtype"] == "torch.float32"
        and result["stored_dtype"] == "torch.float32"
        and type(result["generator_initial_seed"]) is int,
        "official Gaussian identity differs",
    )
    return result


def _validate_external_action_authority(
    external_key: str | Path, external_review_receipt: str | Path,
    external_evidence_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    key, key_path, key_sha = load_json(
        external_key, "BOX-EXP-011 sealed key", EXTERNAL_KEY_SHA256
    )
    review, review_path, review_sha = load_json(
        external_review_receipt,
        "BOX-EXP-011 blind reviewer receipt",
        EXTERNAL_REVIEW_RECEIPT_SHA256,
    )
    unsigned_key = dict(key)
    declared_key_digest = unsigned_key.pop("key_digest", None)
    require(
        key.get("schema_version") == "box-exp-011-arms4-opaque-key-v1"
        and key.get("packet_id") == EXTERNAL_PACKET_ID
        and declared_key_digest == EXTERNAL_KEY_DIGEST
        and object_sha256(unsigned_key) == EXTERNAL_KEY_DIGEST,
        "BOX-EXP-011 sealed key closure differs",
    )
    require(
        review.get("packet_id") == EXTERNAL_PACKET_ID
        and review.get("receipt_schema_version")
        == "opaque-unified-event-full81-reviewer-receipt-v1"
        and review.get("blind_declaration", {}).get("mode") == "blind"
        and review.get("blind_declaration", {}).get("sealed_key_read") is False
        and review.get("coverage", {}).get("required_coverage_satisfied") is True
        and review.get("coverage", {}).get("decoded_and_visually_inspected_frames_per_sample")
        == 81,
        "BOX-EXP-011 blind full81 review authority differs",
    )
    mappings = key.get("mapping")
    samples = review.get("samples")
    require(isinstance(mappings, list) and isinstance(samples, list), "external action evidence differs")
    sample_by_id = {sample.get("sample_id"): sample for sample in samples}
    require(len(sample_by_id) == len(samples), "review sample ids are not unique")
    evidence_root: Optional[Path] = None
    if external_evidence_root is not None:
        root = Path(external_evidence_root)
        require(root.is_absolute(), "external evidence root must be absolute")
        try:
            metadata, resolved_root = root.lstat(), root.resolve(strict=True)
        except OSError as error:
            raise ArmsIncompleteExact2PlanError(
                "external evidence root is unavailable"
            ) from error
        require(
            resolved_root == root
            and stat.S_ISDIR(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode),
            "external evidence root must be one canonical directory",
        )
        evidence_root = resolved_root
    selected: list[dict[str, Any]] = []
    for cell in CELL_SPECS:
        seed = cell["seed"]
        pin = EXTERNAL_ACTION_PINS[seed]
        rows = [
            row
            for row in mappings
            if row.get("seed") == seed and row.get("semantic_branch") == "action"
        ]
        require(len(rows) == 1, f"external action seed {seed} does not resolve once")
        row = rows[0]
        require(
            all(row.get(name) == value for name, value in pin.items() if name in row)
            and row.get("candidate_id") == pin["candidate_id"]
            and row.get("sample_id") == pin["sample_id"]
            and row.get("label") == pin["label"]
            and row.get("mp4_sha256") == pin["mp4_sha256"]
            and row.get("native_receipt_sha256") == pin["native_receipt_sha256"]
            and row.get("native_receipt_digest") == pin["native_receipt_digest"]
            and row.get("calibration_receipt_sha256")
            == pin["calibration_receipt_sha256"]
            and row.get("calibration_receipt_digest")
            == pin["calibration_receipt_digest"]
            and row.get("remote_media_path") == pin["remote_media_path"],
            f"external action pin differs for seed {seed}",
        )
        media_value = (
            evidence_root / "source_media" / f"{pin['label']}.mp4"
            if evidence_root is not None
            else Path(row["source_media_path"])
        )
        native_value = (
            evidence_root / "source_receipts" / pin["label"] / "receipt.json"
            if evidence_root is not None
            else Path(row["native_receipt_path"])
        )
        calibration_value = (
            evidence_root
            / "source_receipts"
            / pin["label"]
            / "pair-v5-t2v-calibration-receipt.json"
            if evidence_root is not None
            else Path(row["calibration_receipt_path"])
        )
        media = plain_file(media_value, f"external action MP4 seed {seed}")
        native, native_path, native_sha = load_json(
            native_value,
            f"external native receipt seed {seed}",
            pin["native_receipt_sha256"],
        )
        calibration, calibration_path, calibration_sha = load_json(
            calibration_value,
            f"external calibration receipt seed {seed}",
            pin["calibration_receipt_sha256"],
        )
        _validate_signed_receipt(
            native, pin["native_receipt_digest"], f"external native receipt seed {seed}"
        )
        _validate_signed_receipt(
            calibration,
            pin["calibration_receipt_digest"],
            f"external calibration receipt seed {seed}",
        )
        candidate = calibration.get("candidate", {})
        mp4 = calibration.get("artifacts", {}).get("mp4", {})
        gaussian = calibration.get("artifacts", {}).get("official_initial_gaussian", {})
        native_output = native.get("outputs", {}).get("t2v", {})
        native_gaussian = native.get("initial_noise_artifacts", {}).get("t2v", {})
        identity = _gaussian_identity(gaussian)
        remote_parent = Path(pin["remote_media_path"]).parent
        expected_gaussian_path = remote_parent / "t2v.official-initial-gaussian.safetensors"
        require(
            file_sha256(media) == pin["mp4_sha256"]
            and candidate.get("candidate_id") == pin["candidate_id"]
            and candidate.get("semantic_branch") == "action"
            and candidate.get("seed") == seed
            and candidate.get("calibration_group_id")
            == f"cell-00435ad621c44fac-s{seed}"
            and mp4.get("path") == pin["remote_media_path"]
            and mp4.get("sha256") == pin["mp4_sha256"]
            and mp4.get("frame_count") == 81
            and native_output.get("path") == pin["remote_media_path"]
            and native_output.get("sha256") == pin["mp4_sha256"]
            and native_output.get("frame_count") == 81
            and gaussian.get("sha256") == pin["gaussian_sha256"]
            and gaussian.get("path") == str(expected_gaussian_path)
            and _gaussian_identity(native_gaussian) == identity,
            f"external action physical receipt binding differs for seed {seed}",
        )
        review_sample = sample_by_id.get(pin["sample_id"], {})
        require(
            review_sample.get("video_sha256") == pin["mp4_sha256"]
            and review_sample.get("frame_count") == 81
            and review_sample.get("classification") == "complete_and_hold"
            and review_sample.get("continuity", {}).get("continuous_subject_and_shot")
            is True
            and review_sample.get("continuity", {}).get("event_breaking_discontinuity")
            is False,
            f"external action did not retain blind complete-and-hold authority for seed {seed}",
        )
        selected.append(
            {
                "seed": seed,
                "calibration_group_id": candidate["calibration_group_id"],
                "candidate_id": pin["candidate_id"],
                "semantic_branch": "action",
                "blind_review": {
                    "packet_id": EXTERNAL_PACKET_ID,
                    "sample_id": pin["sample_id"],
                    "review_receipt_file_sha256": review_sha,
                    "all_81_frames_reviewed": True,
                    "classification": "complete_and_hold",
                },
                "mp4": {
                    "runtime_path": pin["remote_media_path"],
                    "file_sha256": pin["mp4_sha256"],
                    "frame_count": 81,
                },
                "native_receipt": {
                    "runtime_path": str(remote_parent / "receipt.json"),
                    "build_evidence_path": str(native_path),
                    "file_sha256": native_sha,
                    "receipt_digest": pin["native_receipt_digest"],
                },
                "calibration_receipt": {
                    "runtime_path": str(
                        remote_parent / "pair-v5-t2v-calibration-receipt.json"
                    ),
                    "build_evidence_path": str(calibration_path),
                    "file_sha256": calibration_sha,
                    "receipt_digest": pin["calibration_receipt_digest"],
                },
                "official_initial_gaussian": {
                    "runtime_path": str(expected_gaussian_path),
                    "file_sha256": pin["gaussian_sha256"],
                    "identity": identity,
                    "physically_reopen_at_completion": True,
                },
            }
        )
    provenance = {
        "sealed_key": {
            "build_evidence_path": str(key_path),
            "file_sha256": key_sha,
            "key_digest": EXTERNAL_KEY_DIGEST,
        },
        "blind_reviewer_receipt": {
            "build_evidence_path": str(review_path),
            "file_sha256": review_sha,
            "full81": True,
            "blind": True,
        },
    }
    return provenance, selected


def _repair_spec_value(source: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    _assert_prompt_freeze()
    repaired = copy.deepcopy(source)
    found = 0
    for group in repaired.get("groups", []):
        for candidate in group.get("candidates", []):
            if candidate.get("candidate_id") != cell["source_candidate_id"]:
                continue
            require(
                group.get("group_id") == "sp4-a"
                and group.get("visible_gpus") == [0, 1, 2, 3]
                and candidate.get("analysis_split") == "fit"
                and candidate.get("semantic_branch") == "incomplete"
                and candidate.get("seed") == cell["seed"],
                f"source arms incomplete cell differs for {cell['seed_slot']}",
            )
            candidate["candidate_id"] = cell["candidate_id"]
            candidate["full_t2v_caption"] = INCOMPLETE_PROMPT
            candidate["full_t2v_caption_utf8_sha256"] = INCOMPLETE_PROMPT_SHA256
            found += 1
    require(found == 1, f"{cell['seed_slot']} must repair exactly one candidate")
    try:
        return bank_contract.validate_root_spec(repaired)
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise ArmsIncompleteExact2PlanError(str(error)) from error


def _materialize_slot(
    *, cell: Mapping[str, Any], source_path: Path, source_sha: str,
    source: Mapping[str, Any], output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repaired = _repair_spec_value(source, cell)
    spec_dir = output / "sealed-repair-specs"
    spec_dir.mkdir(mode=0o700, exist_ok=True)
    repaired_path = spec_dir / f"{cell['seed_slot']}-arms-incomplete-repair-root-spec-v1.json"
    repaired_raw = bank_contract.canonical_json_bytes(repaired) + b"\n"
    repaired_sha = write_create_only(repaired_path, repaired_raw, "repaired root spec")
    candidate_dir = output / f"{cell['seed_slot']}-candidate-plan"
    try:
        manifest = bank_contract.materialize_plan(
            spec_path=repaired_path,
            expected_sha256=repaired_sha,
            output_dir=candidate_dir,
        )
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise ArmsIncompleteExact2PlanError(str(error)) from error
    records = [
        row for row in manifest["candidate_records"]
        if row["candidate_id"] == cell["candidate_id"]
    ]
    require(len(records) == 1, f"{cell['seed_slot']} executable candidate closure differs")
    record = records[0]
    envelope_path = plain_file(record["path"], "arms incomplete candidate envelope")
    require(file_sha256(envelope_path) == record["sha256"], "candidate envelope SHA differs")
    try:
        envelope = bank_contract.load_candidate_envelope(envelope_path, repaired_sha)
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise ArmsIncompleteExact2PlanError(str(error)) from error
    candidate = envelope["candidate"]
    require(
        envelope["group_id"] == "sp4-a"
        and envelope["visible_gpus"] == [0, 1, 2, 3]
        and candidate["candidate_id"] == cell["candidate_id"]
        and candidate["seed"] == cell["seed"]
        and candidate["semantic_branch"] == "incomplete"
        and candidate["full_t2v_caption_utf8_sha256"]
        == INCOMPLETE_PROMPT_SHA256,
        "materialized exact2 task differs",
    )
    task = {
        "seed_slot": cell["seed_slot"],
        "root_spec_path": str(repaired_path),
        "root_spec_sha256": repaired_sha,
        "candidate_spec_path": str(envelope_path),
        "candidate_spec_sha256": record["sha256"],
        "group_id": envelope["group_id"],
        "visible_gpus": list(envelope["visible_gpus"]),
        "ordinal": envelope["ordinal"],
        "candidate_id": candidate["candidate_id"],
        "analysis_split": candidate["analysis_split"],
        "calibration_group_id": candidate["calibration_group_id"],
        "semantic_branch": candidate["semantic_branch"],
        "seed": candidate["seed"],
        "prompt_utf8_sha256": candidate["full_t2v_caption_utf8_sha256"],
        "source_geometry_video": candidate["geometry_source_video"],
        "source_geometry_video_sha256": candidate["geometry_source_video_sha256"],
    }
    ref = {
        "seed_slot": cell["seed_slot"],
        "source_root_spec_path": str(source_path),
        "source_root_spec_sha256": source_sha,
        "repaired_root_spec_path": str(repaired_path),
        "repaired_root_spec_sha256": repaired_sha,
        "candidate_plan_manifest_path": str(candidate_dir / "manifest.json"),
        "candidate_plan_manifest_sha256": file_sha256(candidate_dir / "manifest.json"),
        "candidate_plan_manifest_digest": manifest["manifest_digest"],
        "executable_candidate_count": 1,
    }
    return ref, task


def _fixed_header() -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA,
        "plan_id": PLAN_ID,
        "experiment_id": "BOX-EXP-013",
        "analysis_split": "fit",
        "repair_kind": "arms_incomplete_only_positive_before_terminal_repair",
        "dataset": DATASET,
        "purpose": "complete the two R4 arms seed pairs without regenerating passing actions",
        "scientific_target": "two complete same-seed action/before-terminal arms cells",
        "learning_target": "N/A; frozen data authoring only",
        "numeric_target": {
            "new_incomplete_full81_pass": [2, 2],
            "external_action_full81_pass": [2, 2],
            "same_gaussian_cross_run_pair_pass": [2, 2],
            "optimizer_updates": 0,
        },
        "baseline": "BOX-EXP-011 arms action 2/2 PASS while arms incomplete 0/2",
        "core_validation": (
            "two new exact40/exact81 incomplete clips independently pass full81; "
            "completion physically reopens both sides and recomputes same-seed "
            "cross-run official-Gaussian equality"
        ),
        "frozen_fit_repair_resource_stack": {
            "archive_sha256": FROZEN_FIT_REPAIR_ARCHIVE_SHA256,
            "manifest_sha256": FROZEN_FIT_REPAIR_MANIFEST_SHA256,
            "method_revision": FROZEN_FIT_REPAIR_REVISION,
            "modified": False,
        },
        "external_action_authority_pins": {
            "sealed_key_file_sha256": EXTERNAL_KEY_SHA256,
            "sealed_key_digest": EXTERNAL_KEY_DIGEST,
            "blind_reviewer_receipt_file_sha256": EXTERNAL_REVIEW_RECEIPT_SHA256,
            "packet_id": EXTERNAL_PACKET_ID,
            "action_count": 2,
            "blind_full81_complete_and_hold_required": True,
        },
        "prompt_freeze": {
            "prompts": {name: dict(value) for name, value in PROMPTS.items()},
            "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
            "forbidden_token_pattern": FORBIDDEN_PROMPT_TOKEN_RE.pattern,
            "forbidden_token_count": 0,
            "frozen_before_any_new_media": True,
        },
        "formal_candidate_count": 2,
        "comparator_cell_count": 2,
        "new_branch_order": ["incomplete", "incomplete"],
        "execution_contract": {
            "formal_dataset": DATASET,
            "formal_generation_invocation_count": 2,
            "num_frames": 81,
            "num_inference_steps": 40,
            "diagnostic_task_count": 0,
            "diagnostic_generation_allowed": False,
            "action_generation_allowed": False,
            "only_incomplete_generation_allowed": True,
            "topology": "one_model_replica_world4_dp1_sp4",
            "candidates_execute_strictly_serial": True,
            "external_action_and_new_incomplete_same_seed_required": True,
            "cross_run_same_official_gaussian_required_per_cell": True,
            "generated_media_is_editor_input_or_target": False,
            "optimizer_created": False,
            "optimizer_authorized": False,
            "training_authorized": False,
        },
    }


def _validate_external_rows(rows: Any) -> None:
    require(isinstance(rows, list) and len(rows) == 2, "external action count differs")
    for row, cell in zip(rows, CELL_SPECS):
        pin = EXTERNAL_ACTION_PINS[cell["seed"]]
        remote_parent = Path(pin["remote_media_path"]).parent
        require(
            row.get("seed") == cell["seed"]
            and row.get("calibration_group_id") == f"cell-00435ad621c44fac-s{cell['seed']}"
            and row.get("candidate_id") == pin["candidate_id"]
            and row.get("semantic_branch") == "action"
            and row.get("blind_review", {}).get("sample_id") == pin["sample_id"]
            and row.get("blind_review", {}).get("review_receipt_file_sha256")
            == EXTERNAL_REVIEW_RECEIPT_SHA256
            and row.get("blind_review", {}).get("all_81_frames_reviewed") is True
            and row.get("blind_review", {}).get("classification") == "complete_and_hold"
            and row.get("mp4", {}).get("runtime_path") == pin["remote_media_path"]
            and row.get("mp4", {}).get("file_sha256") == pin["mp4_sha256"]
            and row.get("native_receipt", {}).get("file_sha256")
            == pin["native_receipt_sha256"]
            and row.get("native_receipt", {}).get("runtime_path")
            == str(remote_parent / "receipt.json")
            and row.get("native_receipt", {}).get("receipt_digest")
            == pin["native_receipt_digest"]
            and row.get("calibration_receipt", {}).get("file_sha256")
            == pin["calibration_receipt_sha256"]
            and row.get("calibration_receipt", {}).get("runtime_path")
            == str(remote_parent / "pair-v5-t2v-calibration-receipt.json")
            and row.get("calibration_receipt", {}).get("receipt_digest")
            == pin["calibration_receipt_digest"]
            and row.get("official_initial_gaussian", {}).get("file_sha256")
            == pin["gaussian_sha256"]
            and row.get("official_initial_gaussian", {}).get("runtime_path")
            == str(remote_parent / "t2v.official-initial-gaussian.safetensors")
            and row.get("official_initial_gaussian", {}).get(
                "physically_reopen_at_completion"
            ) is True
            and _gaussian_identity(
                row.get("official_initial_gaussian", {}).get("identity", {})
            )["generator_initial_seed"] == cell["seed"],
            f"external action row differs for seed {cell['seed']}",
        )


def build_plan(
    *, seed1_spec: str | Path, expected_seed1_spec_sha256: str,
    seed2_spec: str | Path, expected_seed2_spec_sha256: str,
    external_key: str | Path, external_review_receipt: str | Path,
    output_dir: str | Path,
    external_evidence_root: str | Path | None = None,
) -> dict[str, Any]:
    _assert_prompt_freeze()
    require(
        expected_seed1_spec_sha256 == SEED1_SPEC_SHA256
        and expected_seed2_spec_sha256 == SEED2_SPEC_SHA256,
        "source reserve4 spec authority differs",
    )
    output = Path(output_dir)
    require(
        output.is_absolute()
        and output != Path("/")
        and not output.exists()
        and not output.is_symlink()
        and output.parent.is_dir()
        and not output.parent.is_symlink(),
        "plan output directory must be fresh and absolute",
    )
    provenance, external_rows = _validate_external_action_authority(
        external_key, external_review_receipt, external_evidence_root
    )
    _validate_external_rows(external_rows)
    paths = {
        "seed1": plain_file(seed1_spec, "seed1 root spec"),
        "seed2": plain_file(seed2_spec, "seed2 root spec"),
    }
    expected = {"seed1": SEED1_SPEC_SHA256, "seed2": SEED2_SPEC_SHA256}
    sources: dict[str, Mapping[str, Any]] = {}
    source_refs: list[dict[str, Any]] = []
    for slot in ("seed1", "seed2"):
        try:
            source, observed = bank_contract.load_sealed_spec(paths[slot], expected[slot])
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise ArmsIncompleteExact2PlanError(str(error)) from error
        sources[slot] = source
        source_refs.append(
            {"seed_slot": slot, "path": str(paths[slot]), "file_sha256": observed}
        )
    output.mkdir(mode=0o700)
    repaired_refs: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for cell in CELL_SPECS:
        ref, task = _materialize_slot(
            cell=cell,
            source_path=paths[cell["seed_slot"]],
            source_sha=expected[cell["seed_slot"]],
            source=sources[cell["seed_slot"]],
            output=output,
        )
        repaired_refs.append(ref)
        tasks.append(task)
    cells = []
    for cell, task, action in zip(CELL_SPECS, tasks, external_rows):
        require(
            task["seed"] == action["seed"] == cell["seed"]
            and task["calibration_group_id"] == action["calibration_group_id"],
            f"same-seed external pair binding differs for {cell['seed']}",
        )
        cells.append(
            {
                "seed_slot": cell["seed_slot"],
                "group_id": "sp4-a",
                "visible_gpus": [0, 1, 2, 3],
                "iid": cell["iid"],
                "seed": cell["seed"],
                "calibration_group_id": task["calibration_group_id"],
                "external_action_candidate_id": action["candidate_id"],
                "new_incomplete_candidate_id": task["candidate_id"],
                "pair_branch_order": ["action", "incomplete"],
                "same_source_geometry": True,
                "same_seed_cross_run_official_gaussian_required": True,
            }
        )
    unsigned = {
        **_fixed_header(),
        "source_specs": source_refs,
        "repaired_specs": repaired_refs,
        "external_authority_provenance": provenance,
        "external_action_cells": external_rows,
        "admission_tasks": tasks,
        "seed_cells": cells,
        "shards": [
            {
                "shard_id": "arms-incomplete-repair-sp4-a-exact2-v1",
                "group_id": "sp4-a",
                "visible_gpus": [0, 1, 2, 3],
                "candidate_ids": [task["candidate_id"] for task in tasks],
                "candidate_count": 2,
            }
        ],
    }
    value = {**unsigned, "plan_digest": object_sha256(unsigned)}
    plan_path = output / PLAN_FILENAME
    plan_sha = write_create_only(
        plan_path, canonical_json_bytes(value) + b"\n", "exact2 plan"
    )
    return {**value, "_path": str(plan_path), "_file_sha256": plan_sha}


def load_plan(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], Path, str]:
    value, resolved, observed = load_json(path, "arms incomplete exact2 plan", expected_sha256)
    unsigned = dict(value)
    declared = unsigned.pop("plan_digest", None)
    require(
        isinstance(declared, str)
        and SHA256_RE.fullmatch(declared) is not None
        and object_sha256(unsigned) == declared,
        "arms incomplete exact2 plan digest differs",
    )
    fixed = _fixed_header()
    require(
        all(value.get(key) == expected for key, expected in fixed.items()),
        "arms incomplete exact2 fixed authority differs",
    )
    _validate_external_rows(value.get("external_action_cells"))
    sources = value.get("source_specs")
    repaired = value.get("repaired_specs")
    tasks = value.get("admission_tasks")
    cells = value.get("seed_cells")
    require(
        isinstance(sources, list)
        and isinstance(repaired, list)
        and isinstance(tasks, list)
        and isinstance(cells, list)
        and len(sources) == len(repaired) == len(tasks) == len(cells) == 2,
        "exact2 plan reference closure differs",
    )
    expected_source = {"seed1": SEED1_SPEC_SHA256, "seed2": SEED2_SPEC_SHA256}
    for source_ref, repaired_ref, cell, task in zip(
        sources, repaired, CELL_SPECS, tasks
    ):
        slot = cell["seed_slot"]
        require(
            source_ref.get("seed_slot") == slot
            and source_ref.get("file_sha256") == expected_source[slot]
            and repaired_ref.get("seed_slot") == slot
            and repaired_ref.get("source_root_spec_sha256") == expected_source[slot]
            and repaired_ref.get("executable_candidate_count") == 1,
            f"{slot} source/repaired reference differs",
        )
        source_path = plain_file(source_ref["path"], f"{slot} source spec")
        try:
            source, source_sha = bank_contract.load_sealed_spec(
                source_path, expected_source[slot]
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise ArmsIncompleteExact2PlanError(str(error)) from error
        expected_raw = (
            bank_contract.canonical_json_bytes(_repair_spec_value(source, cell)) + b"\n"
        )
        repaired_path = plain_file(
            repaired_ref["repaired_root_spec_path"], f"{slot} repaired spec"
        )
        repaired_sha = hashlib.sha256(expected_raw).hexdigest()
        require(
            source_sha == expected_source[slot]
            and repaired_path.read_bytes() == expected_raw
            and repaired_ref.get("repaired_root_spec_sha256") == repaired_sha,
            f"{slot} repaired spec replay differs",
        )
        envelope_path = plain_file(task["candidate_spec_path"], "exact2 candidate envelope")
        require(
            task.get("seed_slot") == slot
            and task.get("candidate_id") == cell["candidate_id"]
            and task.get("semantic_branch") == "incomplete"
            and task.get("analysis_split") == "fit"
            and task.get("seed") == cell["seed"]
            and task.get("group_id") == "sp4-a"
            and task.get("visible_gpus") == [0, 1, 2, 3]
            and task.get("prompt_utf8_sha256") == INCOMPLETE_PROMPT_SHA256
            and file_sha256(envelope_path) == task.get("candidate_spec_sha256"),
            f"{slot} executable task differs",
        )
        try:
            envelope = bank_contract.load_candidate_envelope(
                envelope_path, task["root_spec_sha256"]
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise ArmsIncompleteExact2PlanError(str(error)) from error
        require(
            envelope["candidate"]["candidate_id"] == cell["candidate_id"]
            and envelope["candidate"]["full_t2v_caption"] == INCOMPLETE_PROMPT,
            f"{slot} candidate envelope replay differs",
        )
    expected_cells = []
    for cell, task, action in zip(
        CELL_SPECS, tasks, value["external_action_cells"]
    ):
        expected_cells.append(
            {
                "seed_slot": cell["seed_slot"],
                "group_id": "sp4-a",
                "visible_gpus": [0, 1, 2, 3],
                "iid": cell["iid"],
                "seed": cell["seed"],
                "calibration_group_id": task["calibration_group_id"],
                "external_action_candidate_id": action["candidate_id"],
                "new_incomplete_candidate_id": task["candidate_id"],
                "pair_branch_order": ["action", "incomplete"],
                "same_source_geometry": True,
                "same_seed_cross_run_official_gaussian_required": True,
            }
        )
    require(
        cells == expected_cells
        and value.get("shards")
        == [
            {
                "shard_id": "arms-incomplete-repair-sp4-a-exact2-v1",
                "group_id": "sp4-a",
                "visible_gpus": [0, 1, 2, 3],
                "candidate_ids": [task["candidate_id"] for task in tasks],
                "candidate_count": 2,
            }
        ]
        and "diagnostic_tasks" not in value,
        "exact2 cell/shard closure differs",
    )
    return value, resolved, observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan")
    build.add_argument("--seed1-spec", required=True)
    build.add_argument("--expected-seed1-spec-sha256", required=True)
    build.add_argument("--seed2-spec", required=True)
    build.add_argument("--expected-seed2-spec-sha256", required=True)
    build.add_argument("--external-key", required=True)
    build.add_argument("--external-review-receipt", required=True)
    build.add_argument("--external-evidence-root")
    build.add_argument("--output-dir", required=True)
    validate = commands.add_parser("validate-plan")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-plan":
        value = build_plan(
            seed1_spec=args.seed1_spec,
            expected_seed1_spec_sha256=args.expected_seed1_spec_sha256,
            seed2_spec=args.seed2_spec,
            expected_seed2_spec_sha256=args.expected_seed2_spec_sha256,
            external_key=args.external_key,
            external_review_receipt=args.external_review_receipt,
            external_evidence_root=args.external_evidence_root,
            output_dir=args.output_dir,
        )
        result = {
            "plan_path": value["_path"],
            "plan_file_sha256": value["_file_sha256"],
            "formal_candidate_count": 2,
            "diagnostic_task_count": 0,
            "prompt_utf8_sha256": INCOMPLETE_PROMPT_SHA256,
        }
    else:
        value, resolved, observed = load_plan(args.plan, args.expected_plan_sha256)
        result = {
            "plan_path": str(resolved),
            "plan_file_sha256": observed,
            "plan_digest": value["plan_digest"],
            "formal_candidate_count": 2,
            "diagnostic_task_count": 0,
            "prompt_utf8_sha256": INCOMPLETE_PROMPT_SHA256,
        }
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArmsIncompleteExact2PlanError",
    "CELL_SPECS",
    "DATASET",
    "EXTERNAL_ACTION_PINS",
    "EXTERNAL_KEY_SHA256",
    "EXTERNAL_REVIEW_RECEIPT_SHA256",
    "FORBIDDEN_PROMPT_TOKEN_RE",
    "GAUSSIAN_IDENTITY_FIELDS",
    "INCOMPLETE_PROMPT",
    "INCOMPLETE_PROMPT_SHA256",
    "PLAN_SCHEMA",
    "PROMPTS",
    "PROMPT_BUNDLE_SHA256",
    "_assert_prompt_freeze",
    "_gaussian_identity",
    "build_plan",
    "canonical_json_bytes",
    "file_sha256",
    "load_json",
    "load_plan",
    "object_sha256",
]
