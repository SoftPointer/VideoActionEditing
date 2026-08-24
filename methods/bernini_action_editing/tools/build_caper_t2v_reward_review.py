#!/usr/bin/env python3
"""Build a fail-visible review page for the sealed PAIR-v5 pure-T2V bank.

The page is an inspection surface for calibration/reward evidence only.  It
does not select a rollout, and it never grants any generated MP4, latent, or
Gaussian authority as an editor target, donor, condition, policy candidate,
or initial noise.  The four registered cells and all forty registered MACE
branches remain in the machine-readable audit even when an artifact is
missing or invalid; only action/noop/incomplete/reverse are rendered as video
cards, while the other six branches remain visible in each cell's audit
table.

The bank/spec/receipt structure is fail-closed.  Candidate media are accepted
only after the root-spec hash, embedded bank and candidate receipt seals,
native receipt binding, artifact hashes, exact81/25fps ffprobe result, and the
same-cell official-Gaussian proof all agree.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_calibration_bank_spec as contract  # noqa: E402


FRAME_COUNT = 81
FPS = 25
DURATION_SECONDS = FRAME_COUNT / FPS
EXPECTED_CELL_COUNT = 4
EXPECTED_CANDIDATE_COUNT = 40
PAIR_RECEIPT_FILENAME = "pair-v5-t2v-calibration-receipt.json"
NATIVE_RECEIPT_FILENAME = "receipt.json"
DISPLAY_BRANCHES = ("action", "noop", "incomplete", "reverse")
AUDIT_ONLY_BRANCHES = tuple(
    branch for branch in contract.MACE_BRANCH_ORDER if branch not in DISPLAY_BRANCHES
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_BANK_FIELDS = {
    "schema_version",
    "root_spec_raw_sha256",
    "candidate_count",
    "cell_count",
    "mace_branch_order",
    "sampling_contract",
    "semantic_input_closure",
    "artifact_use_contract",
    "split_contract",
    "split_group_membership",
    "fit_confirmation_all_registered_axes_disjoint",
    "same_cell_gaussian_proofs",
    "candidate_receipts",
    "interpretation",
    "receipt_digest",
}
_BANK_CANDIDATE_FIELDS = {
    "candidate_id",
    "analysis_split",
    "action_family_id",
    "calibration_group_id",
    "semantic_branch",
    "receipt_path",
    "receipt_sha256",
    "receipt_digest",
    "mp4_sha256",
    "predecode_clean_latent_sha256",
    "official_initial_gaussian_sha256",
}
_CELL_PROOF_FIELDS = {
    "analysis_split",
    "action_family_id",
    "calibration_group_id",
    "semantic_branch_count",
    "semantic_branch_order",
    "all_ten_official_gaussian_tensor_values_byte_equal",
    "all_container_files_individually_sha256_verified",
    "official_gaussian_file_sha256_by_branch",
    "official_gaussian_raw_value_sha256",
    "official_gaussian_content_sha256",
    "seed",
}
_PAIR_RECEIPT_FIELDS = {
    "schema_version",
    "root_spec_raw_sha256",
    "candidate_envelope_sha256",
    "group_id",
    "visible_gpus",
    "runtime_topology",
    "ordinal",
    "candidate",
    "sampling_contract",
    "semantic_input_closure",
    "artifact_use_contract",
    "split_contract",
    "geometry_use_certificate",
    "native_receipt_path",
    "native_receipt_sha256",
    "native_receipt_digest",
    "artifacts",
    "interpretation",
    "receipt_digest",
}
_PAIR_INTERPRETATION = {
    "calibration_evidence_only": True,
    "event_qualified_from_generation_receipt": False,
    "action_success_not_implied": True,
    "training_performed": False,
    "parameter_update_performed": False,
    "optimizer_authorized": False,
    "t2v_media_as_rv2v_policy_candidate_forbidden": True,
    "donor_or_pseudo_target_use_forbidden": True,
}
_BANK_INTERPRETATION = {
    "calibration_evidence_only": True,
    "event_qualification_performed": False,
    "action_success_not_implied": True,
    "training_performed": False,
    "parameter_update_performed": False,
    "optimizer_authorized": False,
    "t2v_negative_media_are_rv2v_policy_candidates": False,
    "t2v_media_as_condition_target_donor_or_noise_forbidden": True,
}
_NATIVE_SCHEMA = "bernini-native-identity-generation-canary-v1"
_NATIVE_METHOD = "frozen-bernini-native-identity-generation-canary"
_ARTIFACT_BASENAMES = {
    "mp4": "t2v.mp4",
    "predecode_clean_latent": "t2v.normalized-clean-latent.safetensors",
    "official_initial_gaussian": "t2v.official-initial-gaussian.safetensors",
}


class CaperT2VReviewError(RuntimeError):
    """Raised when a sealed authority boundary cannot be interpreted safely."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CaperT2VReviewError("value is not canonical finite JSON") from error


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CaperT2VReviewError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise CaperT2VReviewError(f"non-finite JSON constant: {token}")


def _plain_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise CaperT2VReviewError(f"{label} is absent or not an absolute plain file: {path}")
    return path.resolve(strict=True)


def _plain_dir(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path == Path("/") or path.is_symlink() or not path.is_dir():
        raise CaperT2VReviewError(f"{label} is absent or not an absolute plain directory")
    return path.resolve(strict=True)


def _absolute_user_path(path: Path) -> Path:
    """Make a CLI path absolute without resolving away a terminal symlink."""

    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _plain_file(path, label=label)
    try:
        value = json.loads(
            resolved.read_bytes(),
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaperT2VReviewError(f"{label} is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CaperT2VReviewError(f"{label} root is not an object")
    return value


def _load_sealed(
    path: Path, *, label: str, fields: set[str] | None = None
) -> tuple[dict[str, Any], str]:
    value = _load_object(path, label=label)
    if fields is not None and set(value) != fields:
        raise CaperT2VReviewError(f"{label} field closure differs")
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(declared, str)
        or _SHA256_RE.fullmatch(declared) is None
        or object_sha256(unsigned) != declared
    ):
        raise CaperT2VReviewError(f"{label} embedded receipt seal differs")
    return value, declared


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CaperT2VReviewError(f"{label} is not lowercase SHA-256")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaperT2VReviewError(f"{label} is not an object")
    return value


def _pointer_has_suffix(value: object, suffix: Sequence[str], *, label: str) -> None:
    if not isinstance(value, str) or "\x00" in value:
        raise CaperT2VReviewError(f"{label} pointer is not path text")
    pointer = Path(value)
    if not pointer.is_absolute() or tuple(pointer.parts[-len(suffix) :]) != tuple(suffix):
        raise CaperT2VReviewError(f"{label} pointer suffix differs")


def _status(
    state: str,
    message: str,
    *,
    path: Path | None = None,
    receipt: Path | None = None,
    probe: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"status": state, "message": message}
    if path is not None:
        row["path"] = str(path)
    if receipt is not None:
        row["receipt"] = str(receipt)
    if probe is not None:
        row["probe"] = dict(probe)
    if details:
        row["details"] = dict(details)
    return row


def probe_exact81_video(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    """Count the first video stream and require exactly 81 frames at 25 fps."""

    video = _plain_file(path, label="T2V MP4")
    command = (
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
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise CaperT2VReviewError(f"cannot execute ffprobe for {video}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise CaperT2VReviewError(f"ffprobe failed for {video}: {detail}")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        streams = payload["streams"]
        stream = streams[0]
    except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise CaperT2VReviewError(f"ffprobe output differs for {video}") from error
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
        raise CaperT2VReviewError(
            f"not exact81/25fps: {video} "
            f"(frames={frames!r}, fps={rate!r}, size={width!r}x{height!r})"
        )
    return {
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "width": width,
        "height": height,
        "ffprobe_count_frames": True,
    }


class _ArtifactValidator:
    def __init__(self, *, ffprobe: str) -> None:
        self.ffprobe = ffprobe
        self._hashes: dict[Path, str] = {}
        self._probes: dict[Path, dict[str, Any]] = {}

    def hash(self, path: Path, expected: object, *, label: str) -> str:
        resolved = _plain_file(path, label=label)
        digest = _sha(expected, label=f"{label} declared SHA-256")
        if resolved not in self._hashes:
            self._hashes[resolved] = file_sha256(resolved)
        if self._hashes[resolved] != digest:
            raise CaperT2VReviewError(f"{label} SHA-256 differs")
        return digest

    def video(self, path: Path, expected: object, *, label: str) -> dict[str, Any]:
        digest = self.hash(path, expected, label=label)
        resolved = path.resolve(strict=True)
        if resolved not in self._probes:
            self._probes[resolved] = probe_exact81_video(resolved, ffprobe=self.ffprobe)
        return {"sha256": digest, "probe": self._probes[resolved]}


def _validate_bank_authority(
    *, bank_root: Path, bank_receipt_path: Path, root_spec_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[tuple[tuple[str, str, str], list[dict[str, Any]]]],
]:
    spec_path = _plain_file(root_spec_path, label="sealed PAIR-v5 root spec")
    bank_path = _plain_file(bank_receipt_path, label="sealed PAIR-v5 bank receipt")
    bank, _ = _load_sealed(bank_path, label="bank receipt", fields=_BANK_FIELDS)
    if bank.get("schema_version") != contract.BANK_RECEIPT_SCHEMA_VERSION:
        raise CaperT2VReviewError("bank receipt schema differs")
    spec_raw = spec_path.read_bytes()
    spec_sha = hashlib.sha256(spec_raw).hexdigest()
    if bank.get("root_spec_raw_sha256") != spec_sha:
        raise CaperT2VReviewError("bank receipt/root-spec raw SHA-256 binding differs")
    try:
        raw_spec = json.loads(
            spec_raw,
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
        )
        spec = contract.validate_root_spec(raw_spec)
    except (UnicodeDecodeError, json.JSONDecodeError, contract.PairT2VCalibrationSpecError) as error:
        raise CaperT2VReviewError(f"sealed root spec is invalid: {error}") from error

    if (
        bank.get("candidate_count") != EXPECTED_CANDIDATE_COUNT
        or bank.get("cell_count") != EXPECTED_CELL_COUNT
        or bank.get("mace_branch_order") != list(contract.MACE_BRANCH_ORDER)
        or bank.get("sampling_contract") != contract.SAMPLING_CONTRACT
        or bank.get("semantic_input_closure") != contract.SEMANTIC_INPUT_CLOSURE
        or bank.get("artifact_use_contract") != contract.ARTIFACT_USE_CONTRACT
        or bank.get("split_contract") != contract.SPLIT_CONTRACT
        or bank.get("fit_confirmation_all_registered_axes_disjoint") is not True
        or bank.get("interpretation") != _BANK_INTERPRETATION
    ):
        raise CaperT2VReviewError("bank closure or calibration-only authority differs")

    candidates: list[dict[str, Any]] = []
    candidate_coordinates: dict[str, tuple[str, list[int], int]] = {}
    cells_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    cell_order: list[tuple[str, str, str]] = []
    for group in spec["groups"]:
        for ordinal, candidate in enumerate(group["candidates"]):
            row = dict(candidate)
            candidates.append(row)
            candidate_coordinates[row["candidate_id"]] = (
                group["group_id"],
                list(group["visible_gpus"]),
                ordinal,
            )
            key = (
                row["analysis_split"],
                row["action_family_id"],
                row["calibration_group_id"],
            )
            if key not in cells_by_key:
                cell_order.append(key)
                cells_by_key[key] = []
            cells_by_key[key].append(row)
    if len(candidates) != EXPECTED_CANDIDATE_COUNT or len(cells_by_key) != EXPECTED_CELL_COUNT:
        raise CaperT2VReviewError("root spec is not the formal four-cell/forty-candidate bank")
    if len({row["candidate_id"] for row in candidates}) != EXPECTED_CANDIDATE_COUNT:
        raise CaperT2VReviewError("root spec candidate IDs are not unique")
    for key in cell_order:
        if [row["semantic_branch"] for row in cells_by_key[key]] != list(
            contract.MACE_BRANCH_ORDER
        ):
            raise CaperT2VReviewError(f"cell {key!r} lost exact ten-branch closure")

    split_membership = {
        split: {
            axis: sorted(
                {
                    row[axis]
                    for row in candidates
                    if row["analysis_split"] == split
                }
            )
            for axis in contract.SPLIT_GROUP_AXES
        }
        for split in contract.ANALYSIS_SPLITS
    }
    if bank.get("split_group_membership") != split_membership:
        raise CaperT2VReviewError("bank fit/confirmation membership differs from spec")

    receipt_rows = bank.get("candidate_receipts")
    if not isinstance(receipt_rows, list) or len(receipt_rows) != EXPECTED_CANDIDATE_COUNT:
        raise CaperT2VReviewError("bank candidate receipt closure is not exactly forty")
    for index, (candidate, row) in enumerate(zip(candidates, receipt_rows)):
        if not isinstance(row, dict) or set(row) != _BANK_CANDIDATE_FIELDS:
            raise CaperT2VReviewError(f"bank candidate row {index} field closure differs")
        expected_identity = {
            "candidate_id": candidate["candidate_id"],
            "analysis_split": candidate["analysis_split"],
            "action_family_id": candidate["action_family_id"],
            "calibration_group_id": candidate["calibration_group_id"],
            "semantic_branch": candidate["semantic_branch"],
        }
        if any(row.get(key) != value for key, value in expected_identity.items()):
            raise CaperT2VReviewError(f"bank candidate row {index} differs from spec order")
        _pointer_has_suffix(
            row.get("receipt_path"),
            (candidate["candidate_id"], PAIR_RECEIPT_FILENAME),
            label=f"bank candidate {index} receipt",
        )
        for field in (
            "receipt_sha256",
            "receipt_digest",
            "mp4_sha256",
            "predecode_clean_latent_sha256",
            "official_initial_gaussian_sha256",
        ):
            _sha(row.get(field), label=f"bank candidate {index} {field}")

    proofs = bank.get("same_cell_gaussian_proofs")
    if not isinstance(proofs, list) or len(proofs) != EXPECTED_CELL_COUNT:
        raise CaperT2VReviewError("bank same-cell Gaussian proof closure is not exactly four")
    for index, (key, proof) in enumerate(zip(cell_order, proofs)):
        if not isinstance(proof, dict) or set(proof) != _CELL_PROOF_FIELDS:
            raise CaperT2VReviewError(f"same-cell Gaussian proof {index} field closure differs")
        if tuple(proof.get(name) for name in (
            "analysis_split", "action_family_id", "calibration_group_id"
        )) != key:
            raise CaperT2VReviewError(f"same-cell Gaussian proof {index} order differs")

    # The coordinate map is private evidence used by candidate inspection.
    for candidate in candidates:
        candidate["_execution_coordinate"] = candidate_coordinates[candidate["candidate_id"]]
    ordered_cells = [(key, cells_by_key[key]) for key in cell_order]
    return spec, bank, candidates, ordered_cells


def _validate_native_receipt(
    *,
    native: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_dir: Path,
    pair_artifacts: Mapping[str, Any],
    validator: _ArtifactValidator,
) -> dict[str, Any]:
    if (
        native.get("schema_version") != _NATIVE_SCHEMA
        or native.get("method") != _NATIVE_METHOD
        or native.get("arms") != ["t2v"]
    ):
        raise CaperT2VReviewError("native receipt did not execute frozen T2V-only")
    native_input = _mapping(native.get("input"), label="native input")
    if (
        native_input.get("source_video_sha256")
        != candidate["geometry_source_video_sha256"]
        or native_input.get("action_prompt_utf8_sha256")
        != candidate["full_t2v_caption_utf8_sha256"]
        or native_input.get("target_video") is not False
        or native_input.get("external_reference_image_or_video") is not False
        or native_input.get("external_mask_flow_pose_track_trajectory") is not False
        or native_input.get("external_first_frame_anchor") is not False
    ):
        raise CaperT2VReviewError("native prompt/geometry or no-target boundary differs")
    preprocessing = _mapping(native.get("preprocessing"), label="native preprocessing")
    bucket = preprocessing.get("source_derived_bucket_hw")
    if (
        preprocessing.get("frame_count") != FRAME_COUNT
        or preprocessing.get("fps") != FPS
        or not isinstance(bucket, list)
        or len(bucket) != 2
        or any(type(item) is not int or item <= 0 or item % 16 for item in bucket)
    ):
        raise CaperT2VReviewError("native exact81 geometry bucket differs")
    conditioning = _mapping(
        _mapping(native.get("conditioning"), label="native conditioning").get("t2v"),
        label="native T2V conditioning",
    )
    expected_source_ids = {
        "target_source_id": 0,
        "video_source_ids": [],
        "reference_source_ids": [],
        "conditioning_source_count": 0,
        "max_conditioning_source_id": 0,
        "within_pretrained_source_ids_1_through_5": True,
        "source_id_interpolation_required": False,
    }
    if (
        conditioning.get("full_source_video_count") != 0
        or conditioning.get("source_derived_reference_count") != 0
        or conditioning.get("source_frame_indices") != []
        or conditioning.get("reference_encoding") != "none"
        or conditioning.get("source_ids") != expected_source_ids
    ):
        raise CaperT2VReviewError("source content entered native T2V conditioning")
    identities = _mapping(native.get("condition_identities"), label="native identities")
    if (
        identities.get("references") != {}
        or identities.get("full_source_video") is not None
        or identities.get("rank_zero_broadcasts")
        != {"references": {}, "full_source_video": None}
        or native.get("source_condition_artifact") is not None
    ):
        raise CaperT2VReviewError("native T2V created a source latent/reference")
    sampling = _mapping(
        _mapping(native.get("sampling"), label="native sampling").get("t2v"),
        label="native T2V sampling",
    )
    guidance = contract.SAMPLING_CONTRACT["guidance"]
    if (
        sampling.get("num_frames") != FRAME_COUNT
        or sampling.get("num_inference_steps") != 40
        or sampling.get("guidance_mode") != "t2v_apg"
        or sampling.get("seed") != candidate["seed"]
        or sampling.get("omega_txt") != guidance["omega_txt"]
        or sampling.get("omega_vid") != guidance["omega_vid"]
        or sampling.get("omega_img") != guidance["omega_img"]
        or sampling.get("target_initialization") != contract.TARGET_INITIALIZATION
        or sampling.get("target_mixed_with_source_latent") is not False
        or sampling.get("custom_sampler_or_scheduler") is not False
        or sampling.get("ulysses_size") != 4
    ):
        raise CaperT2VReviewError("native T2V sampling contract differs")
    expected_shape = [1, 16, 21, int(bucket[0]) // 8, int(bucket[1]) // 8]
    geometry = _mapping(native.get("latent_geometry"), label="native latent geometry")
    if geometry.get("video_latent_shape") != expected_shape:
        raise CaperT2VReviewError("native latent/bucket geometry differs")

    output = _mapping(
        _mapping(native.get("outputs"), label="native outputs").get("t2v"),
        label="native T2V output",
    )
    clean = _mapping(output.get("normalized_clean_latent"), label="native clean latent")
    gaussian = _mapping(
        _mapping(native.get("initial_noise_artifacts"), label="native noise").get("t2v"),
        label="native official Gaussian",
    )
    if (
        output.get("frame_count") != FRAME_COUNT
        or output.get("fps") != FPS
        or output.get("height") != bucket[0]
        or output.get("width") != bucket[1]
        or clean.get("shape") != expected_shape
        or clean.get("native_sampler_before_vae_decode") is not True
        or clean.get("mp4_decode_reencode_used") is not False
        or gaussian.get("shape") != expected_shape
        or gaussian.get("generator_initial_seed") != candidate["seed"]
        or gaussian.get("captured_from_native_sampler") is not True
        or gaussian.get("external_initial_noise_injection") is not False
        or gaussian.get("source_or_target_derived") is not False
        or gaussian.get("observer_changed_return_value") is not False
        or gaussian.get("official_randn_tensor_call_count") != 1
    ):
        raise CaperT2VReviewError("native exact81 clean/Gaussian provenance differs")
    for field in ("raw_value_sha256", "content_sha256"):
        _sha(gaussian.get(field), label=f"official Gaussian {field}")
    if _mapping(native.get("interpretation"), label="native interpretation").get(
        "training_performed"
    ) is not False:
        raise CaperT2VReviewError("native generation receipt performed training")

    expected_artifacts = {
        "mp4": output,
        "predecode_clean_latent": clean,
        "official_initial_gaussian": gaussian,
    }
    if pair_artifacts != expected_artifacts:
        raise CaperT2VReviewError("PAIR artifact declarations differ from native receipt")
    validated: dict[str, Any] = {}
    for role, artifact in expected_artifacts.items():
        basename = _ARTIFACT_BASENAMES[role]
        _pointer_has_suffix(
            artifact.get("path"), (candidate["candidate_id"], basename), label=role
        )
        actual = candidate_dir / basename
        if role == "mp4":
            validated[role] = validator.video(
                actual, artifact.get("sha256"), label=f"{candidate['candidate_id']} MP4"
            )
        else:
            validated[role] = {
                "sha256": validator.hash(
                    actual, artifact.get("sha256"), label=f"{candidate['candidate_id']} {role}"
                )
            }
    validated["gaussian_identity"] = {
        field: gaussian.get(field)
        for field in (
            "raw_value_sha256",
            "content_sha256",
            "shape",
            "dtype",
            "stored_dtype",
            "generator_initial_seed",
        )
    }
    return validated


def _inspect_candidate(
    *,
    bank_root: Path,
    candidate: Mapping[str, Any],
    bank_row: Mapping[str, Any],
    root_spec_sha256: str,
    validator: _ArtifactValidator,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    candidate_id = str(candidate["candidate_id"])
    candidate_dir = bank_root / candidate_id
    receipt_path = candidate_dir / PAIR_RECEIPT_FILENAME
    mp4_path = candidate_dir / _ARTIFACT_BASENAMES["mp4"]
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return (
            _status(
                "missing",
                "registered PAIR candidate receipt is absent",
                path=mp4_path,
                receipt=receipt_path,
            ),
            None,
        )
    try:
        pair, pair_digest = _load_sealed(
            receipt_path,
            label=f"{candidate_id} PAIR receipt",
            fields=_PAIR_RECEIPT_FIELDS,
        )
        coordinate = candidate["_execution_coordinate"]
        expected_runtime = {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": ",".join(str(item) for item in coordinate[1]),
        }
        if (
            pair.get("schema_version") != contract.RECEIPT_SCHEMA_VERSION
            or pair.get("root_spec_raw_sha256") != root_spec_sha256
            or pair.get("candidate")
            != {key: value for key, value in candidate.items() if not key.startswith("_")}
            or pair.get("group_id") != coordinate[0]
            or pair.get("visible_gpus") != coordinate[1]
            or pair.get("ordinal") != coordinate[2]
            or pair.get("runtime_topology") != expected_runtime
            or pair.get("sampling_contract") != contract.SAMPLING_CONTRACT
            or pair.get("semantic_input_closure") != contract.SEMANTIC_INPUT_CLOSURE
            or pair.get("artifact_use_contract") != contract.ARTIFACT_USE_CONTRACT
            or pair.get("split_contract") != contract.SPLIT_CONTRACT
            or pair.get("interpretation") != _PAIR_INTERPRETATION
        ):
            raise CaperT2VReviewError("PAIR candidate/spec or calibration-only binding differs")
        _sha(pair.get("candidate_envelope_sha256"), label="candidate envelope SHA-256")
        if (
            bank_row.get("receipt_sha256") != file_sha256(receipt_path)
            or bank_row.get("receipt_digest") != pair_digest
        ):
            raise CaperT2VReviewError("bank/candidate receipt seal binding differs")
        geometry = _mapping(pair.get("geometry_use_certificate"), label="geometry certificate")
        if (
            set(geometry)
            != {
                "video_sha256",
                "bucket_hw",
                "latent_shape",
                "used_to_derive_bucket_shape",
                "vae_latent_created",
                "pixels_entered_transformer",
                "content_conditioning_count",
            }
            or geometry.get("video_sha256") != candidate["geometry_source_video_sha256"]
            or geometry.get("used_to_derive_bucket_shape") is not True
            or geometry.get("vae_latent_created") is not False
            or geometry.get("pixels_entered_transformer") is not False
            or geometry.get("content_conditioning_count") != 0
        ):
            raise CaperT2VReviewError("geometry-only certificate differs")
        native_path = candidate_dir / NATIVE_RECEIPT_FILENAME
        _pointer_has_suffix(
            pair.get("native_receipt_path"),
            (candidate_id, NATIVE_RECEIPT_FILENAME),
            label="native receipt",
        )
        native, native_digest = _load_sealed(
            native_path, label=f"{candidate_id} native receipt"
        )
        if (
            pair.get("native_receipt_sha256") != file_sha256(native_path)
            or pair.get("native_receipt_digest") != native_digest
        ):
            raise CaperT2VReviewError("PAIR/native receipt seal binding differs")
        pair_artifacts = _mapping(pair.get("artifacts"), label="PAIR artifacts")
        if set(pair_artifacts) != set(_ARTIFACT_BASENAMES):
            raise CaperT2VReviewError("PAIR artifact role closure differs")
        evidence = _validate_native_receipt(
            native=native,
            candidate=candidate,
            candidate_dir=candidate_dir,
            pair_artifacts=pair_artifacts,
            validator=validator,
        )
        if (
            bank_row.get("mp4_sha256") != evidence["mp4"]["sha256"]
            or bank_row.get("predecode_clean_latent_sha256")
            != evidence["predecode_clean_latent"]["sha256"]
            or bank_row.get("official_initial_gaussian_sha256")
            != evidence["official_initial_gaussian"]["sha256"]
        ):
            raise CaperT2VReviewError("bank/candidate artifact SHA-256 binding differs")
        probe = evidence["mp4"]["probe"]
        if (
            probe["width"] != geometry.get("bucket_hw", [None, None])[1]
            or probe["height"] != geometry.get("bucket_hw", [None, None])[0]
        ):
            raise CaperT2VReviewError("ffprobe MP4 dimensions differ from sealed bucket")
        return (
            _status(
                "valid",
                "sealed pure-T2V calibration evidence verified",
                path=mp4_path,
                receipt=receipt_path,
                probe=probe,
                details={
                    "mp4_sha256": evidence["mp4"]["sha256"],
                    "receipt_digest": pair_digest,
                    "calibration_evidence_only": True,
                    "donor_or_target_authority": False,
                },
            ),
            evidence,
        )
    except (CaperT2VReviewError, OSError) as error:
        return (
            _status(
                "invalid",
                str(error),
                path=mp4_path,
                receipt=receipt_path,
            ),
            None,
        )


def _check_cell_proof(
    *,
    key: tuple[str, str, str],
    candidates: Sequence[Mapping[str, Any]],
    proof: Mapping[str, Any],
    evidence_by_branch: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    try:
        branches = list(contract.MACE_BRANCH_ORDER)
        if (
            tuple(proof.get(name) for name in (
                "analysis_split", "action_family_id", "calibration_group_id"
            )) != key
            or proof.get("semantic_branch_count") != len(branches)
            or proof.get("semantic_branch_order") != branches
            or proof.get("all_ten_official_gaussian_tensor_values_byte_equal") is not True
            or proof.get("all_container_files_individually_sha256_verified") is not True
        ):
            raise CaperT2VReviewError("sealed same-cell Gaussian proof header differs")
        by_branch = proof.get("official_gaussian_file_sha256_by_branch")
        # Canonical JSON sorts object keys, so dictionary iteration order is
        # not the semantic MACE order.  Exact order is carried separately by
        # ``semantic_branch_order``; this map must have exactly the same keys.
        if (
            not isinstance(by_branch, dict)
            or len(by_branch) != len(branches)
            or set(by_branch) != set(branches)
        ):
            raise CaperT2VReviewError("same-cell Gaussian file proof branch closure differs")
        _sha(proof.get("official_gaussian_raw_value_sha256"), label="proof raw Gaussian")
        _sha(proof.get("official_gaussian_content_sha256"), label="proof content Gaussian")
        if proof.get("seed") != candidates[0]["seed"]:
            raise CaperT2VReviewError("same-cell Gaussian proof seed differs")
        identities: list[dict[str, Any]] = []
        for candidate in candidates:
            branch = candidate["semantic_branch"]
            evidence = evidence_by_branch.get(branch)
            if evidence is None:
                raise CaperT2VReviewError(f"{branch} has no validated Gaussian evidence")
            identity = dict(evidence["gaussian_identity"])
            identities.append(identity)
            if by_branch[branch] != evidence["official_initial_gaussian"]["sha256"]:
                raise CaperT2VReviewError(f"{branch} Gaussian container proof differs")
        canonical_identities = {object_sha256(identity) for identity in identities}
        if len(canonical_identities) != 1:
            raise CaperT2VReviewError("ten official Gaussian tensor identities differ")
        first = identities[0]
        if (
            first.get("raw_value_sha256")
            != proof.get("official_gaussian_raw_value_sha256")
            or first.get("content_sha256")
            != proof.get("official_gaussian_content_sha256")
            or first.get("generator_initial_seed") != proof.get("seed")
        ):
            raise CaperT2VReviewError("observed Gaussian identity differs from sealed proof")
        return _status(
            "valid",
            "all ten branches share one sealed official Gaussian tensor value",
            details={
                "raw_value_sha256": first["raw_value_sha256"],
                "content_sha256": first["content_sha256"],
                "seed": first["generator_initial_seed"],
            },
        )
    except CaperT2VReviewError as error:
        return _status("invalid", str(error))


def _unexpected_receipts(bank_root: Path, expected_ids: Sequence[str]) -> list[str]:
    expected = {bank_root / candidate_id / PAIR_RECEIPT_FILENAME for candidate_id in expected_ids}
    observed = {
        path
        for path in bank_root.glob(f"*/{PAIR_RECEIPT_FILENAME}")
        if path.is_file() and not path.is_symlink()
    }
    return sorted(str(path.relative_to(bank_root)) for path in observed - expected)


def build_review(
    *,
    bank_root: Path,
    bank_receipt_path: Path,
    root_spec_path: Path,
    output_html: Path,
    audit_json: Path | None = None,
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    """Verify the formal four-cell bank and write HTML plus JSON audit."""

    bank_root = _plain_dir(_absolute_user_path(bank_root), label="T2V bank root")
    bank_receipt_path = _absolute_user_path(bank_receipt_path)
    root_spec_path = _absolute_user_path(root_spec_path)
    spec, bank, candidates, ordered_cells = _validate_bank_authority(
        bank_root=bank_root,
        bank_receipt_path=bank_receipt_path,
        root_spec_path=root_spec_path,
    )
    root_spec_sha256 = bank["root_spec_raw_sha256"]
    validator = _ArtifactValidator(ffprobe=ffprobe)
    bank_rows = {row["candidate_id"]: row for row in bank["candidate_receipts"]}
    proof_rows = {
        (
            proof["analysis_split"],
            proof["action_family_id"],
            proof["calibration_group_id"],
        ): proof
        for proof in bank["same_cell_gaussian_proofs"]
    }

    cells: list[dict[str, Any]] = []
    for key, cell_candidates in ordered_cells:
        branch_rows: dict[str, dict[str, Any]] = {}
        evidence: dict[str, dict[str, Any] | None] = {}
        for candidate in cell_candidates:
            branch = candidate["semantic_branch"]
            status, branch_evidence = _inspect_candidate(
                bank_root=bank_root,
                candidate=candidate,
                bank_row=bank_rows[candidate["candidate_id"]],
                root_spec_sha256=root_spec_sha256,
                validator=validator,
            )
            branch_rows[branch] = {
                "candidate_id": candidate["candidate_id"],
                "semantic_branch": branch,
                "caption": candidate["full_t2v_caption"],
                "caption_utf8_sha256": candidate["full_t2v_caption_utf8_sha256"],
                "media": status,
            }
            evidence[branch] = branch_evidence
        proof_status = _check_cell_proof(
            key=key,
            candidates=cell_candidates,
            proof=proof_rows[key],
            evidence_by_branch=evidence,
        )
        if proof_status["status"] != "valid":
            for branch in contract.MACE_BRANCH_ORDER:
                media = branch_rows[branch]["media"]
                if media["status"] == "valid":
                    branch_rows[branch]["media"] = _status(
                        "invalid",
                        f"cell Gaussian proof invalid: {proof_status['message']}",
                        path=Path(media["path"]),
                        receipt=Path(media["receipt"]),
                        details={"prior_media_validation": "valid"},
                    )
        cells.append(
            {
                "cell_id": key[2],
                "analysis_split": key[0],
                "action_family_id": key[1],
                "calibration_group_id": key[2],
                "seed": cell_candidates[0]["seed"],
                "prompt_group_id": cell_candidates[0]["prompt_group_id"],
                "actor_group_id": cell_candidates[0]["actor_group_id"],
                "scene_group_id": cell_candidates[0]["scene_group_id"],
                "action_group_id": cell_candidates[0]["action_group_id"],
                "same_cell_gaussian": proof_status,
                "branches": branch_rows,
                "display_branch_order": list(DISPLAY_BRANCHES),
                "audit_only_branch_order": list(AUDIT_ONLY_BRANCHES),
            }
        )

    unexpected = _unexpected_receipts(
        bank_root, [candidate["candidate_id"] for candidate in candidates]
    )
    statuses = [
        cell["branches"][branch]["media"]["status"]
        for cell in cells
        for branch in contract.MACE_BRANCH_ORDER
    ]
    counts = {state: statuses.count(state) for state in sorted(set(statuses))}
    review_complete = (
        len(cells) == EXPECTED_CELL_COUNT
        and len(statuses) == EXPECTED_CANDIDATE_COUNT
        and all(state == "valid" for state in statuses)
        and all(cell["same_cell_gaussian"]["status"] == "valid" for cell in cells)
        and not unexpected
    )
    audit: dict[str, Any] = {
        "schema_version": "bernini-caper-pure-t2v-reward-html-review-audit-v1",
        "review_complete": review_complete,
        "authority": {
            "purpose": "pure_t2v_reward_calibration_review_only",
            "event_qualification_performed": False,
            "action_success_implied": False,
            "training_or_parameter_update_performed": False,
            "donor_selection_performed": False,
            "target_selection_performed": False,
            "media_as_condition_target_donor_policy_candidate_or_noise_authorized": False,
        },
        "selection_policy": {
            "root_spec_order_only": True,
            "fixed_four_cells": True,
            "fixed_ten_branches_per_cell": True,
            "displayed_video_branches": list(DISPLAY_BRANCHES),
            "audit_only_branches": list(AUDIT_ONLY_BRANCHES),
            "missing_or_invalid_candidates_hidden": False,
            "seed_filtering_or_best_of_k": False,
        },
        "bank_root": str(bank_root),
        "root_spec": {
            "path": str(root_spec_path),
            "sha256": root_spec_sha256,
            "schema_version": spec["schema_version"],
        },
        "bank_receipt": {
            "path": str(bank_receipt_path),
            "file_sha256": file_sha256(bank_receipt_path),
            "receipt_digest": bank["receipt_digest"],
            "schema_version": bank["schema_version"],
        },
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "expected_candidate_count": EXPECTED_CANDIDATE_COUNT,
        "candidate_count": len(statuses),
        "unexpected_candidate_receipts": unexpected,
        "status_counts": counts,
        "cells": cells,
    }
    audit["audit_digest"] = object_sha256(audit)

    output_html = output_html.expanduser().resolve()
    audit_json = (
        audit_json.expanduser().resolve()
        if audit_json is not None
        else output_html.with_name(f"{output_html.stem}.audit.json")
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    if audit_json.parent != output_html.parent:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(_render_html(audit, output_html, audit_json), encoding="utf-8")
    audit_json.write_bytes(canonical_json_bytes(audit) + b"\n")
    return audit


def _url(output_html: Path, value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = Path(value)
    try:
        relative = os.path.relpath(path, start=output_html.parent)
    except ValueError:
        return path.as_uri() if path.is_absolute() else quote(path.as_posix())
    return quote(Path(relative).as_posix(), safe="/:._-")


def _badge(state: str) -> str:
    return f'<span class="badge {html.escape(state)}">{html.escape(state.upper())}</span>'


def _media_card(
    *, title: str, branch: Mapping[str, Any], group: str, output_html: Path
) -> str:
    media = _mapping(branch.get("media"), label="HTML branch media")
    state = str(media.get("status", "invalid"))
    message = html.escape(str(media.get("message", "no diagnostic")))
    receipt_url = _url(output_html, media.get("receipt"))
    receipt_link = (
        f'<a href="{html.escape(receipt_url)}">receipt</a>' if receipt_url else ""
    )
    caption = html.escape(str(branch.get("caption", "")))
    head = (
        f'<div class="head"><h3>{html.escape(title)}</h3><div>{_badge(state)} '
        f'{receipt_link}</div><p>{message}</p><details><summary>caption</summary>'
        f'<p>{caption}</p></details></div>'
    )
    if state == "valid" and (video_url := _url(output_html, media.get("path"))):
        body = (
            f'<video data-group="{html.escape(group)}" controls muted playsinline '
            f'preload="metadata" src="{html.escape(video_url)}"></video>'
        )
    else:
        details = media.get("details")
        detail_text = html.escape(
            json.dumps(details, sort_keys=True, ensure_ascii=False, default=str)
            if isinstance(details, dict)
            else ""
        )
        body = (
            f'<div class="placeholder {html.escape(state)}"><strong>'
            f'{html.escape(state.upper())}</strong><span>{message}</span>'
            f'<code>{detail_text}</code></div>'
        )
    return f'<article class="card {html.escape(state)}">{head}{body}</article>'


def _branch_audit_rows(cell: Mapping[str, Any], output_html: Path) -> str:
    rows = []
    for branch_name in contract.MACE_BRANCH_ORDER:
        branch = cell["branches"][branch_name]
        media = branch["media"]
        receipt_url = _url(output_html, media.get("receipt"))
        link = f'<a href="{html.escape(receipt_url)}">receipt</a>' if receipt_url else ""
        placement = "video card" if branch_name in DISPLAY_BRANCHES else "audit only"
        rows.append(
            "<tr><td>"
            + html.escape(branch_name)
            + "</td><td>"
            + html.escape(placement)
            + "</td><td>"
            + _badge(str(media["status"]))
            + "</td><td>"
            + html.escape(str(branch["candidate_id"]))
            + "</td><td>"
            + html.escape(str(media["message"]))
            + " "
            + link
            + "</td></tr>"
        )
    return "".join(rows)


def _scope_controls(group: str, *, global_scope: bool) -> str:
    label = "全部" if global_scope else "本 cell"
    return (
        f'<div class="controls" data-controls="{html.escape(group)}">'
        f'<button class="primary" data-command="play" data-scope="{html.escape(group)}">播放{label}</button>'
        f'<button data-command="pause" data-scope="{html.escape(group)}">暂停{label}</button>'
        f'<button data-command="reset" data-scope="{html.escape(group)}">回到 0</button>'
        f'<label>速度 <select data-rate="{html.escape(group)}"><option>.5</option>'
        '<option selected>1</option><option>1.5</option><option>2</option></select></label>'
        f'<input data-seek="{html.escape(group)}" type="range" min="0" '
        f'max="{DURATION_SECONDS:.2f}" step="0.01" value="0">'
        f'<span data-time="{html.escape(group)}">0.00 / {DURATION_SECONDS:.2f} s</span></div>'
    )


def _render_html(audit: Mapping[str, Any], output_html: Path, audit_json: Path) -> str:
    sections: list[str] = []
    titles = {
        "action": "Action",
        "noop": "No-op",
        "incomplete": "Incomplete",
        "reverse": "Reverse",
    }
    for cell in audit["cells"]:
        group = str(cell["cell_id"])
        cards = "".join(
            _media_card(
                title=titles[branch],
                branch=cell["branches"][branch],
                group=group,
                output_html=output_html,
            )
            for branch in DISPLAY_BRANCHES
        )
        proof = cell["same_cell_gaussian"]
        sections.append(
            f'<section class="sample" id="{html.escape(group)}"><div class="sample-title">'
            f'<div><h2>{html.escape(group)}</h2><p>{html.escape(cell["analysis_split"])} · '
            f'{html.escape(cell["action_family_id"])} · seed={html.escape(str(cell["seed"]))}</p>'
            f'<p>same-cell Gaussian {_badge(str(proof["status"]))}: '
            f'{html.escape(str(proof["message"]))}</p></div></div>'
            + _scope_controls(group, global_scope=False)
            + f'<div class="grid">{cards}</div><details class="audit"><summary>'
            '10-branch audit（其余 6 个 branch 不作为视频卡，但不隐藏）</summary>'
            '<table><thead><tr><th>branch</th><th>placement</th><th>status</th>'
            '<th>candidate</th><th>evidence</th></tr></thead><tbody>'
            + _branch_audit_rows(cell, output_html)
            + "</tbody></table></details></section>"
        )
    complete = bool(audit["review_complete"])
    verdict = "COMPLETE" if complete else "INCOMPLETE / FAIL-VISIBLE"
    verdict_class = "valid" if complete else "invalid"
    counts = " ".join(
        f"{html.escape(key)}={value}"
        for key, value in sorted(audit["status_counts"].items())
    )
    audit_url = _url(output_html, str(audit_json)) or f"{output_html.stem}.audit.json"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAPER pure-T2V reward-bank review</title><style>
:root{{--bg:#070b12;--panel:#101824;--card:#090e16;--line:#293851;--text:#eef5ff;--muted:#a8b4c8;--ok:#55d69e;--bad:#ff7883;--miss:#ffd06f}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 0,#17243a 0,transparent 36rem),var(--bg);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC",sans-serif}}main{{max-width:1900px;margin:auto;padding:20px}}.hero,.sample{{border:1px solid var(--line);border-radius:16px;background:rgba(16,24,36,.96);padding:17px;margin-bottom:17px}}h1,h2,h3,p{{margin-top:0}}h1{{margin-bottom:7px;font-size:clamp(25px,3vw,38px)}}h2{{font-size:18px;margin-bottom:4px}}h3{{font-size:15px;margin-bottom:7px}}a{{color:#9ac4ff}}.muted,.head p,.sample-title p{{color:var(--muted)}}.controls{{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:12px 0;padding:10px;border:1px solid var(--line);border-radius:12px;background:rgba(7,11,18,.92)}}.hero>.controls{{position:sticky;top:0;z-index:5;backdrop-filter:blur(10px)}}button,select{{border:1px solid #3a4e6d;border-radius:8px;background:#17263b;color:var(--text);padding:7px 10px;cursor:pointer}}button.primary{{background:#215da8}}input[type=range]{{width:min(420px,46vw)}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(230px,1fr));gap:10px}}.card{{overflow:hidden;border:1px solid var(--line);border-radius:11px;background:var(--card)}}.card.invalid,.card.failed{{border-color:#793d48}}.card.missing{{border-color:#786333}}.head{{padding:10px;min-height:157px}}.head p{{font-size:12px;margin:7px 0}}video{{display:block;width:100%;aspect-ratio:31/30;object-fit:contain;background:#000}}.placeholder{{aspect-ratio:31/30;display:flex;flex-direction:column;justify-content:center;gap:8px;padding:15px;background:#0b111b;text-align:center}}.placeholder span{{color:var(--muted)}}.placeholder code{{max-height:120px;overflow:auto;white-space:pre-wrap;text-align:left;font-size:10px}}.badge{{display:inline-block;padding:2px 7px;border:1px solid var(--line);border-radius:999px;font-size:10px}}.badge.valid{{color:var(--ok);border-color:#286d53}}.badge.failed,.badge.invalid{{color:var(--bad);border-color:#7d3b47}}.badge.missing{{color:var(--miss);border-color:#796431}}.audit{{margin-top:12px}}table{{width:100%;border-collapse:collapse;margin-top:9px}}th,td{{border:1px solid var(--line);padding:7px;text-align:left;font-size:12px}}@media(max-width:1200px){{.grid{{grid-template-columns:repeat(2,minmax(220px,1fr))}}}}@media(max-width:600px){{main{{padding:9px}}.grid{{grid-template-columns:1fr}}.head{{min-height:0}}}}
</style></head><body><main><section class="hero"><h1>CAPER pure-T2V reward bank</h1><p class="muted">sealed PAIR-v5 core4 · 4 cells × 10 branches · exact81 / 25 fps · fixed order · no seed selection</p><p>{_badge(verdict_class)} <strong>{verdict}</strong> · {html.escape(counts)} · <a href="{html.escape(audit_url)}">machine-readable audit</a></p><p><strong>Authority boundary:</strong> these source-free videos are reward/calibration evidence only. They are not editor targets, donors, conditions, policy candidates, latents, velocities, or noise.</p>{_scope_controls("all", global_scope=True)}</section>
{"".join(sections)}</main><script>
const videos=[...document.querySelectorAll('video')],limit={DURATION_SECONDS:.2f};let active=[],leader=null;const clamp=x=>Math.max(0,Math.min(limit,Number(x)||0));const scoped=g=>g==='all'?videos:videos.filter(v=>v.dataset.group===g);const controls=g=>({{seek:document.querySelector(`[data-seek="${{CSS.escape(g)}}"]`),time:document.querySelector(`[data-time="${{CSS.escape(g)}}"]`),rate:document.querySelector(`[data-rate="${{CSS.escape(g)}}"]`)}});const ready=v=>v.readyState>=1?Promise.resolve():new Promise(resolve=>{{const done=()=>resolve();v.addEventListener('loadedmetadata',done,{{once:true}});v.addEventListener('error',done,{{once:true}})}});async function seekOne(v,t){{await ready(v);if(Number.isFinite(v.duration))t=Math.min(t,Math.max(0,v.duration-.001));try{{v.currentTime=t}}catch(_e){{}}}}function pauseScope(g){{scoped(g).forEach(v=>v.pause());if(g==='all'||active.some(v=>v.dataset.group===g)){{active=[];leader=null}}}}async function playScope(g){{pauseScope('all');const c=controls(g);active=scoped(g);const t=clamp(c.seek.value);await Promise.all(active.map(v=>seekOne(v,t)));active.forEach(v=>v.playbackRate=Number(c.rate.value));leader=active[0]||null;await Promise.allSettled(active.map(v=>v.play()))}}async function resetScope(g){{pauseScope(g);const c=controls(g);await Promise.all(scoped(g).map(v=>seekOne(v,0)));c.seek.value=0;c.time.textContent=`0.00 / ${{limit.toFixed(2)}} s`}}document.querySelectorAll('[data-command]').forEach(b=>b.onclick=()=>{{const g=b.dataset.scope,a=b.dataset.command;if(a==='play')playScope(g);else if(a==='pause')pauseScope(g);else resetScope(g)}});document.querySelectorAll('[data-rate]').forEach(s=>s.onchange=()=>scoped(s.dataset.rate).forEach(v=>v.playbackRate=Number(s.value)));document.querySelectorAll('[data-seek]').forEach(s=>s.oninput=async()=>{{const g=s.dataset.seek;pauseScope(g);const t=clamp(s.value);await Promise.all(scoped(g).map(v=>seekOne(v,t)));controls(g).time.textContent=`${{t.toFixed(2)}} / ${{limit.toFixed(2)}} s`}});videos.forEach(v=>v.addEventListener('timeupdate',()=>{{if(v!==leader||v.paused)return;const t=clamp(v.currentTime);const groups=['all',v.dataset.group];groups.forEach(g=>{{const c=controls(g);c.seek.value=t;c.time.textContent=`${{t.toFixed(2)}} / ${{limit.toFixed(2)}} s`}});active.forEach(other=>{{if(other!==v&&!other.seeking&&Math.abs(other.currentTime-t)>.08)other.currentTime=t}})}}));
</script></body></html>"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--root-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="return zero after writing a fail-visible incomplete page",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit = build_review(
        bank_root=args.bank_root,
        bank_receipt_path=args.bank_receipt,
        root_spec_path=args.root_spec,
        output_html=args.output,
        audit_json=args.audit_json,
        ffprobe=args.ffprobe,
    )
    print(
        canonical_json_bytes(
            {
                "review_complete": audit["review_complete"],
                "status_counts": audit["status_counts"],
                "cell_count": len(audit["cells"]),
                "candidate_count": audit["candidate_count"],
                "audit_digest": audit["audit_digest"],
            }
        ).decode("utf-8")
    )
    return 0 if audit["review_complete"] or args.allow_incomplete else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_ONLY_BRANCHES",
    "CaperT2VReviewError",
    "DISPLAY_BRANCHES",
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_CELL_COUNT",
    "FRAME_COUNT",
    "FPS",
    "build_review",
    "canonical_json_bytes",
    "file_sha256",
    "main",
    "object_sha256",
    "probe_exact81_video",
]
