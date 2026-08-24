#!/usr/bin/env python3
"""Build the two-row SEER event-erasure training dataset.

The target of each row is an independently reviewed, self-generated action
video.  Its source is constructed from target frames 0..31 followed by exact
copies of frame 31.  Consequently source and target share one generated
identity/background/camera coordinate, while no transition or terminal frame
is available in the source.

``build`` creates the videos, shared-I0 arrays, Bernini raw parquet and its
legacy-compatible receipts, plus an explicit full-pair routing file.
``finalize`` runs only after VAE materialization and create-only publishes the
manifest that binds the materialized shards, summary and index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import build_renderer_dataset as raw_builder  # noqa: E402
import train_seer_event_erasure_smoke as owner_contract  # noqa: E402


RAW_MANIFEST_SCHEMA = "bernini-seer-event-erasure-raw-dataset-v1"
FINAL_MANIFEST_SCHEMA = "bernini-seer-event-erasure-dataset-v1"
ROUTING_SCHEMA = "bernini-cdf-routing-v1"
FRAME_COUNT = 81
FPS = 25
CUTOFF = 32
AUTHORITY = {
    "experimental_parameter_update_authorized": True,
    "self_generated_video_target_authorized_for_this_fresh_experiment": True,
    "training_completion_is_method_success": False,
    "heldout_decoded_review_required": True,
    "production_claim_authorized": False,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SeerDatasetError(RuntimeError):
    """Raised before publishing an ambiguous training row."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SeerDatasetError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SeerDatasetError(f"{label} must be a lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SeerDatasetError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SeerDatasetError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise SeerDatasetError(f"{label} must be a plain file")
    return path.resolve(strict=True)


def _directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SeerDatasetError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SeerDatasetError(f"{label} is unavailable: {error}") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise SeerDatasetError(f"{label} must be a plain directory")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SeerDatasetError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise SeerDatasetError(f"{label} must contain one object")
    return value


def _check_object_digest(value: Mapping[str, Any], *, label: str) -> str:
    candidate = dict(value)
    declared = candidate.pop("receipt_digest", None)
    if not isinstance(declared, str) or object_sha256(candidate) != declared:
        raise SeerDatasetError(f"{label} canonical digest differs")
    return declared


def _atomic_create(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    if path.exists() or path.is_symlink():
        raise SeerDatasetError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, mode)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_create(path, canonical_json_bytes(value) + b"\n")


def event_erasure_index_map() -> list[int]:
    return list(range(CUTOFF)) + [CUTOFF - 1] * (FRAME_COUNT - CUTOFF)


def event_erased_rgb_frames(target_frames: Sequence[bytes]) -> list[bytes]:
    """Return the exact pre-transition prefix followed by a frame-31 hold."""

    if len(target_frames) != FRAME_COUNT:
        raise SeerDatasetError(f"target must contain exactly {FRAME_COUNT} RGB frames")
    return [target_frames[index] for index in event_erasure_index_map()]


def _run(
    command: Sequence[str],
    *,
    capture: bool = False,
    input_payload: Optional[bytes] = None,
) -> bytes:
    try:
        result = subprocess.run(
            list(command),
            check=True,
            input=input_payload,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = error.stderr.decode("utf-8", errors="replace")[-2000:]
        raise SeerDatasetError(f"command failed: {command[0]}: {detail}") from error
    return result.stdout if capture else b""


def _video_metadata(ffprobe: Optional[Path], video: Path) -> tuple[int, int, int, float]:
    if ffprobe is None:
        try:
            import av
            with av.open(str(video), mode="r") as container:
                if not container.streams.video:
                    raise SeerDatasetError(f"video stream is missing: {video}")
                stream = container.streams.video[0]
                width, height = int(stream.width), int(stream.height)
                rate = stream.average_rate
                if rate is None:
                    raise SeerDatasetError(f"video average rate is missing: {video}")
                fps = float(rate)
                count = sum(1 for _ in container.decode(stream))
        except (ImportError, OSError, ValueError, av.error.FFmpegError if "av" in locals() else OSError) as error:
            if isinstance(error, SeerDatasetError):
                raise
            raise SeerDatasetError(f"cannot read PyAV metadata for {video}: {error}") from error
        if width <= 0 or height <= 0 or count != FRAME_COUNT or abs(fps - FPS) > 1e-6:
            raise SeerDatasetError(
                f"video must be exact {FRAME_COUNT}f@{FPS}: {video}: "
                f"{width}x{height} {count} {fps}"
            )
        return width, height, count, fps
    payload = _run(
        [
            str(ffprobe), "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate",
            "-of", "json", str(video),
        ],
        capture=True,
    )
    try:
        value = json.loads(payload)
        stream = value["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        count = int(stream["nb_read_frames"])
        numerator, denominator = stream["avg_frame_rate"].split("/", 1)
        fps = int(numerator) / int(denominator)
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as error:
        raise SeerDatasetError(f"cannot parse video metadata for {video}") from error
    if width <= 0 or height <= 0 or count != FRAME_COUNT or abs(fps - FPS) > 1e-6:
        raise SeerDatasetError(
            f"video must be exact {FRAME_COUNT}f@{FPS}: {video}: {width}x{height} {count} {fps}"
        )
    return width, height, count, fps


def _decode_rgb(ffmpeg: Path, video: Path, width: int, height: int) -> list[bytes]:
    payload = _run(
        [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(video), "-map", "0:v:0", "-vsync", "0", "-pix_fmt", "rgb24",
            "-f", "rawvideo", "pipe:1",
        ],
        capture=True,
    )
    frame_bytes = width * height * 3
    if len(payload) != frame_bytes * FRAME_COUNT:
        raise SeerDatasetError(f"decoded RGB byte count differs for {video}")
    return [
        payload[offset : offset + frame_bytes]
        for offset in range(0, len(payload), frame_bytes)
    ]


def _write_shared_i0(path: Path, rgb: bytes, width: int, height: int) -> None:
    try:
        import numpy as np
    except ImportError as error:
        raise SeerDatasetError("NumPy is required to write shared I0") from error
    array = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)
    chw = (array.astype(np.float32) / 255.0 * 2.0 - 1.0).transpose(2, 0, 1)
    chw = np.ascontiguousarray(chw.astype("<f4", copy=False))
    if path.exists() or path.is_symlink():
        raise SeerDatasetError(f"refusing to overwrite {path}")
    with path.open("xb") as handle:
        np.save(handle, chw, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)


def _build_erased_video(
    *, ffmpeg: Path, ffprobe: Optional[Path], target: Path, source: Path, shared_i0: Path
) -> dict[str, Any]:
    width, height, _, _ = _video_metadata(ffprobe, target)
    if source.exists() or source.is_symlink():
        raise SeerDatasetError(f"refusing to overwrite {source}")
    target_frames = _decode_rgb(ffmpeg, target, width, height)
    erased_frames = event_erased_rgb_frames(target_frames)
    _run(
        [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", str(FPS),
            "-i", "pipe:0",
            "-frames:v", str(FRAME_COUNT), "-an", "-c:v", "libx264rgb", "-crf", "0",
            "-preset", "medium", "-pix_fmt", "rgb24", str(source),
        ],
        input_payload=b"".join(erased_frames),
    )
    os.chmod(source, 0o444)
    source_meta = _video_metadata(ffprobe, source)
    if source_meta[:2] != (width, height):
        raise SeerDatasetError("event-erased video geometry differs")
    source_frames = _decode_rgb(ffmpeg, source, width, height)
    index_map = event_erasure_index_map()
    mismatches = [
        index for index, target_index in enumerate(index_map)
        if source_frames[index] != target_frames[target_index]
    ]
    if mismatches:
        raise SeerDatasetError(f"event-erasure RGB mapping differs at {mismatches[:8]}")
    _write_shared_i0(shared_i0, target_frames[0], width, height)
    return {
        "width": width,
        "height": height,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "index_map": index_map,
        "index_map_sha256": object_sha256(index_map),
        "prefix_rgb_exact": True,
        "transition_indices_absent": max(index_map) < CUTOFF,
    }


def _verify_owner_evidence(owner: Mapping[str, Any], owner_path: Path) -> None:
    for stem, label in (
        ("owner_registry", "owner registry"),
        ("owner_master_receipt", "owner master receipt"),
    ):
        binding = owner[stem]
        path = _plain_file(binding["path"], label=label)
        if file_sha256(path) != binding["sha256"]:
            raise SeerDatasetError(f"{label} hash differs")
        if stem == "owner_master_receipt":
            value = _read_json(path, label=label)
            if _check_object_digest(value, label=label) != binding["receipt_digest"]:
                raise SeerDatasetError("owner master embedded digest differs")
    for row in owner["rows"]:
        target = _plain_file(row["target_video"], label=f"{row['iid']} target")
        if file_sha256(target) != row["target_video_sha256"]:
            raise SeerDatasetError(f"{row['iid']} target hash differs")
        for path_field, sha_field, label in (
            ("owner_generation_receipt", "owner_generation_receipt_sha256", "generation receipt"),
            ("native_receipt", "native_receipt_sha256", "native receipt"),
        ):
            path = _plain_file(row[path_field], label=f"{row['iid']} {label}")
            if file_sha256(path) != row[sha_field]:
                raise SeerDatasetError(f"{row['iid']} {label} hash differs")
            if path_field == "owner_generation_receipt":
                value = _read_json(path, label=label)
                if _check_object_digest(value, label=label) != row["owner_generation_receipt_digest"]:
                    raise SeerDatasetError(f"{row['iid']} generation digest differs")
    if file_sha256(owner_path) != owner_contract._sha256(owner_path):
        raise SeerDatasetError("owner spec changed during validation")


def _renderer_row(
    owner_row: Mapping[str, Any], *, owner_path: Path, owner_sha: str,
    source: Path, source_sha: str, shared_i0: Path, shared_sha: str,
) -> dict[str, Any]:
    instruction = str(owner_row["instruction"])
    messages = [
        {"type": "video", "has_loss": 0},
        {"type": "text", "text": instruction, "has_loss": 0},
        {"type": "video_gen", "has_loss": 1},
    ]
    target = Path(str(owner_row["target_video"])).resolve(strict=True)
    gates = {
        "detached_full81_action_review_pass": True,
        "same_generated_video_coordinate": True,
        "event_transition_erased_from_source": True,
        "decoded_rgb_index_map_exact": True,
        "fresh_user_directed_experiment": True,
    }
    upstream = {
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "production_eligible": False,
        "post_video_acceptance": "fresh_user_directed_event_qualified",
    }
    row: dict[str, Any] = {
        "schema_version": raw_builder.ROW_FORMAT,
        "inputs": canonical_json_bytes(messages).decode("utf-8"),
        "videos": [{"video_path": str(source)}, {"video_path": str(target)}],
        "iid": owner_row["iid"],
        "group_id": owner_row["source_iid"],
        "family": owner_row["actor_family"],
        "edit_instruction_sha256": owner_row["instruction_utf8_sha256"],
        "source_video_path": str(source),
        "source_video_declared_path": str(source),
        "source_video_sha256": source_sha,
        "target_video_path": str(target),
        "target_video_declared_path": str(target),
        "target_video_sha256": owner_row["target_video_sha256"],
        "shared_i0_path": str(shared_i0),
        "shared_i0_sha256": shared_sha,
        "preview_manifest_path": str(owner_path),
        "preview_manifest_sha256": owner_sha,
        "preview_row_digest": object_sha256(dict(owner_row)),
        "preview_row_file_sha256": owner_sha,
        "experimental_inclusion_policy": raw_builder.STRICT_INCLUSION_POLICY,
        "selection_gates_json": canonical_json_bytes(gates).decode("utf-8"),
        "strict_selection_gates_all_true": True,
        "upstream_authorization_json": canonical_json_bytes(upstream).decode("utf-8"),
        **upstream,
        "experimental_training_acknowledged": True,
        "production_claim_forbidden": True,
    }
    row["renderer_row_digest"] = object_sha256(row)
    return row


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise SeerDatasetError(f"symlink in SEER output: {path}")
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def build(
    *, owner_spec: Path, expected_owner_sha256: str, output_root: Path,
    ffmpeg: Path, ffprobe: Optional[Path],
) -> dict[str, Any]:
    expected_owner_sha256 = _sha(expected_owner_sha256, label="owner spec hash")
    owner_spec = _plain_file(owner_spec, label="owner spec")
    if file_sha256(owner_spec) != expected_owner_sha256:
        raise SeerDatasetError("owner spec raw hash differs")
    owner = owner_contract._load_owner_spec(owner_spec, expected_owner_sha256)
    _verify_owner_evidence(owner, owner_spec)
    if output_root.exists() or output_root.is_symlink():
        raise SeerDatasetError("output root must be fresh")
    output_root.mkdir(parents=True, mode=0o700)
    media_root = output_root / "media"
    i0_root = output_root / "shared_i0"
    media_root.mkdir(mode=0o700)
    i0_root.mkdir(mode=0o700)
    manifest_rows: list[dict[str, Any]] = []
    parquet_rows: list[dict[str, Any]] = []
    for owner_row in owner["rows"]:
        iid = str(owner_row["iid"])
        target = _plain_file(owner_row["target_video"], label=f"{iid} target")
        source = media_root / f"{iid}.mp4"
        shared = i0_root / f"{iid}.npy"
        evidence = _build_erased_video(
            ffmpeg=ffmpeg, ffprobe=ffprobe, target=target, source=source, shared_i0=shared
        )
        source_sha, shared_sha = file_sha256(source), file_sha256(shared)
        manifest_rows.append(
            {
                "iid": iid,
                "source_iid": owner_row["source_iid"],
                "source_video": str(source.resolve()),
                "source_video_sha256": source_sha,
                "target_video": str(target),
                "target_video_sha256": owner_row["target_video_sha256"],
                "shared_i0_path": str(shared.resolve()),
                "shared_i0_sha256": shared_sha,
                "index_map_sha256": evidence["index_map_sha256"],
                "prefix_rgb_exact": True,
                "transition_indices_absent": True,
            }
        )
        parquet_rows.append(
            _renderer_row(
                owner_row, owner_path=owner_spec, owner_sha=expected_owner_sha256,
                source=source.resolve(), source_sha=source_sha,
                shared_i0=shared.resolve(), shared_sha=shared_sha,
            )
        )

    raw_parquet = output_root / "raw.parquet"
    raw_builder.write_parquet(parquet_rows, raw_parquet)
    os.chmod(raw_parquet, 0o444)
    receipt: dict[str, Any] = {
        "schema_version": raw_builder.RECEIPT_FORMAT,
        "complete": True,
        "experimental_training_acknowledged": True,
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "production_eligible": False,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "source_preview_manifest": str(owner_spec),
        "source_preview_manifest_sha256": expected_owner_sha256,
        "experimental_inclusion_policy": raw_builder.STRICT_INCLUSION_POLICY,
        "broader_natural_release_inclusion_acknowledged": False,
        "sample_count": len(parquet_rows),
        "strict_selection_rows": len(parquet_rows),
        "non_strict_selection_rows": 0,
        "sample_ids": [row["iid"] for row in parquet_rows],
        "renderer_row_digests_sha256": object_sha256(
            [row["renderer_row_digest"] for row in parquet_rows]
        ),
        "bernini_messages": list(raw_builder.BERNINI_MESSAGE_TYPES),
        "parquet_path": str(raw_parquet.resolve()),
        "parquet_sha256": file_sha256(raw_parquet),
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    receipt_path = output_root / "raw.receipt.json"
    _write_json(receipt_path, receipt)
    done: dict[str, Any] = {
        "schema_version": raw_builder.JOB_DONE_FORMAT,
        "complete": True,
        "sample_count": len(parquet_rows),
        "strict_selection_rows": len(parquet_rows),
        "non_strict_selection_rows": 0,
        "experimental_inclusion_policy": raw_builder.STRICT_INCLUSION_POLICY,
        "raw_parquet_sha256": file_sha256(raw_parquet),
        "raw_receipt_sha256": file_sha256(receipt_path),
        "preview_manifest_sha256": expected_owner_sha256,
    }
    done["job_done_digest"] = object_sha256(done)
    done_path = output_root / "raw.job_done.json"
    _write_json(done_path, done)

    routing_path = output_root / "full_pair.routing.jsonl"
    routing_payload = b"".join(
        canonical_json_bytes(
            {
                "schema_version": ROUTING_SCHEMA,
                "iid": row["iid"],
                "tier": "full_pair",
                "full_target_weight": 1.0,
                "review": "event-qualified same-generated-coordinate SEER pair",
            }
        ) + b"\n"
        for row in sorted(manifest_rows, key=lambda value: value["iid"])
    )
    _atomic_create(routing_path, routing_payload)
    raw_manifest: dict[str, Any] = {
        "schema_version": RAW_MANIFEST_SCHEMA,
        "owner_spec": {"path": str(owner_spec), "sha256": expected_owner_sha256},
        "event_erasure": {
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "cutoff_exclusive": CUTOFF,
            "index_map": event_erasure_index_map(),
            "index_map_sha256": object_sha256(event_erasure_index_map()),
        },
        "rows": manifest_rows,
        "raw": {
            "parquet_path": str(raw_parquet.resolve()),
            "parquet_sha256": file_sha256(raw_parquet),
            "receipt_path": str(receipt_path.resolve()),
            "receipt_sha256": file_sha256(receipt_path),
            "job_done_path": str(done_path.resolve()),
            "job_done_sha256": file_sha256(done_path),
        },
        "routing": {
            "path": str(routing_path.resolve()),
            "sha256": file_sha256(routing_path),
            "row_count": len(manifest_rows),
        },
        "authority": dict(AUTHORITY),
    }
    raw_manifest["manifest_digest"] = object_sha256(raw_manifest)
    raw_manifest_path = output_root / "seer_raw_manifest.json"
    _write_json(raw_manifest_path, raw_manifest)
    _seal_tree(output_root)
    return {
        "status": "raw_event_erasure_dataset_built_not_method_success",
        "output_root": str(output_root.resolve()),
        "raw_manifest": str(raw_manifest_path.resolve()),
        "raw_manifest_sha256": file_sha256(raw_manifest_path),
        "raw_parquet": str(raw_parquet.resolve()),
        "raw_receipt": str(receipt_path.resolve()),
        "raw_job_done": str(done_path.resolve()),
        "routing_jsonl": str(routing_path.resolve()),
        "routing_sha256": file_sha256(routing_path),
        "row_count": len(manifest_rows),
    }


def finalize(
    *, raw_manifest_path: Path, expected_raw_manifest_sha256: str,
    parquet_directory: Path, dataset_summary: Path, index_path: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    raw_manifest_path = _plain_file(raw_manifest_path, label="raw SEER manifest")
    if file_sha256(raw_manifest_path) != _sha(
        expected_raw_manifest_sha256, label="raw manifest hash"
    ):
        raise SeerDatasetError("raw SEER manifest hash differs")
    raw = _read_json(raw_manifest_path, label="raw SEER manifest")
    candidate = dict(raw)
    declared = candidate.pop("manifest_digest", None)
    if raw.get("schema_version") != RAW_MANIFEST_SCHEMA or object_sha256(candidate) != declared:
        raise SeerDatasetError("raw SEER manifest schema/digest differs")
    parquet_directory = _directory(parquet_directory, label="VAE parquet directory")
    dataset_summary = _plain_file(dataset_summary, label="VAE dataset summary")
    index_path = _plain_file(index_path, label="VAE dataset index")
    summary = _read_json(dataset_summary, label="VAE dataset summary")
    row_count = len(raw.get("rows", []))
    if (
        summary.get("complete") is not True
        or summary.get("expected_sample_count") != row_count
        or summary.get("materialized_sample_count") != row_count
        or summary.get("missing_sample_count") != 0
        or summary.get("experimental_inclusion_policy") != raw_builder.STRICT_INCLUSION_POLICY
        or Path(str(summary.get("shards_directory"))).resolve() != parquet_directory
        or Path(str(summary.get("index_path"))).resolve() != index_path
        or summary.get("index_sha256") != file_sha256(index_path)
    ):
        raise SeerDatasetError("VAE summary/index closure differs")
    final = {key: value for key, value in raw.items() if key != "manifest_digest"}
    final["schema_version"] = FINAL_MANIFEST_SCHEMA
    final["vae"] = {
        "parquet_directory": str(parquet_directory),
        "dataset_summary_path": str(dataset_summary),
        "dataset_summary_sha256": file_sha256(dataset_summary),
        "index_path": str(index_path),
        "index_sha256": file_sha256(index_path),
        "row_count": row_count,
    }
    final["manifest_digest"] = object_sha256(final)
    if not output_manifest.is_absolute():
        raise SeerDatasetError("final manifest output must be absolute")
    _write_json(output_manifest, final)
    return {
        "status": "seer_dataset_finalized_not_method_success",
        "manifest": str(output_manifest.resolve()),
        "manifest_sha256": file_sha256(output_manifest),
        "manifest_digest": final["manifest_digest"],
        "row_count": row_count,
        "routing_jsonl": raw["routing"]["path"],
        "routing_sha256": raw["routing"]["sha256"],
    }


def _executable(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SeerDatasetError(f"{label} must be absolute")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise SeerDatasetError(f"{label} must resolve to an executable file")
    return resolved


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--owner-spec", type=Path, required=True)
    build_parser.add_argument("--expected-owner-spec-sha256", required=True)
    build_parser.add_argument("--output-root", type=Path, required=True)
    build_parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    build_parser.add_argument(
        "--ffprobe",
        default=None,
        help="optional executable ffprobe; when omitted, use the frozen Python PyAV",
    )
    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--raw-manifest", type=Path, required=True)
    final_parser.add_argument("--expected-raw-manifest-sha256", required=True)
    final_parser.add_argument("--parquet-directory", type=Path, required=True)
    final_parser.add_argument("--dataset-summary", type=Path, required=True)
    final_parser.add_argument("--index", type=Path, required=True)
    final_parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        result = build(
            owner_spec=args.owner_spec,
            expected_owner_sha256=args.expected_owner_spec_sha256,
            output_root=args.output_root.expanduser(),
            ffmpeg=_executable(args.ffmpeg, label="ffmpeg"),
            ffprobe=(
                _executable(args.ffprobe, label="ffprobe")
                if args.ffprobe is not None
                else None
            ),
        )
    else:
        result = finalize(
            raw_manifest_path=args.raw_manifest,
            expected_raw_manifest_sha256=args.expected_raw_manifest_sha256,
            parquet_directory=args.parquet_directory,
            dataset_summary=args.dataset_summary,
            index_path=args.index,
            output_manifest=args.output_manifest.expanduser(),
        )
    print(canonical_json_bytes(result).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SeerDatasetError, owner_contract.SeerSmokeError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
