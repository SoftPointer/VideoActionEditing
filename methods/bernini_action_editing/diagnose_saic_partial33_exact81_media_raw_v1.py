#!/usr/bin/env python3
"""CPU-only exact81 camera/technical/temporal diagnostics for sealed SAIC r5.

The only accepted bank index is the already sealed exact33 source-bound input
manifest.  Every candidate and its correct registered source are re-hashed
before ``saic_exact81_media_diagnostics_v1.build_diagnostic`` snapshots and
measures them.  The resulting measurements are descriptive raw diagnostics;
they cannot rank or select a candidate, verify an event or preservation, make
a scientific claim, authorize training, or authorize an optimizer update.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import stat
from typing import Any, Iterable, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
PINNED_LOCAL_SOURCES = {
    "saic_exact81_media_diagnostics_v1.py":
        "3658056640b0adc3411c04c029ce99efd5a4d9388be638f659bd8eb472399e0a",
    "diagnose_saic_partial33_source_bound_dinov2_raw_v1.py":
        "3faad816ee0a4f320b83afb94377da7a437c535f60105d3fd8578b7068851600",
}
for _name, _expected_sha in PINNED_LOCAL_SOURCES.items():
    _path = METHOD_ROOT / _name
    if (
        not _path.is_file()
        or _path.is_symlink()
        or hashlib.sha256(_path.read_bytes()).hexdigest() != _expected_sha
    ):
        raise RuntimeError(f"pinned local source differs: {_name}")

import diagnose_saic_partial33_source_bound_dinov2_raw_v1 as frozen_specialization  # noqa: E402
import saic_exact81_media_diagnostics_v1 as diagnostics  # noqa: E402

# The exact33 specialization configures its pinned exact47 implementation and
# its nested partial28 helper to the exact33 executable identity.  Use that
# configured implementation directly so manifest self/source validation stays
# bound to the frozen exact33 validator, while the base algorithm remains
# transitively pinned by the specialization itself.
frozen_input = frozen_specialization.core


SCHEMA_VERSION = "bernini-saic-partial33-exact81-media-raw-v1"
AGGREGATE_SCHEMA = f"{SCHEMA_VERSION}-aggregate"
PREFLIGHT_SCHEMA = f"{SCHEMA_VERSION}-preflight"
EXPECTED_CANDIDATE_COUNT = 33
EXPECTED_REGISTERED_SOURCE_COUNT = 8
EXPECTED_CANDIDATE_BOUND_SOURCE_COUNT = 5
EXPECTED_INPUT_MANIFEST_SHA256 = (
    "35d757481bc62f64dfea12d4924b555cc989d207d8b7a502a9d7f214ef815834"
)
EXPECTED_FROZEN_INPUT_SOURCE_SHA256 = PINNED_LOCAL_SOURCES[
    "diagnose_saic_partial33_source_bound_dinov2_raw_v1.py"
]
EXPECTED_EXACT81_SOURCE_SHA256 = PINNED_LOCAL_SOURCES[
    "saic_exact81_media_diagnostics_v1.py"
]
EXPECTED_ROOT_SPEC_SHA256 = (
    "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9"
)
EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f"
)
EXPECTED_SOURCE_VALIDATOR_SUMMARY_SHA256 = (
    "257d3aafaaee126ff2c1a061413d01bd0457676eb5d1ee027671221a5a794218"
)
EXPECTED_DECODED_EVALUATOR_SHA256 = (
    "f96758ae8e975db27dd3b58fd06c70185912de6a548ecb69302a48c0ea3ecef4"
)
EXPECTED_GEOMETRY_SHA256 = (
    "7371b8292ec8b100961d49f70ef6095a2133331137120d5fd94e6c78e6fbfd02"
)
EXPECTED_SELECTED_FRAME_INDICES = list(range(0, 81, 5))
EXPECTED_WRONG_SOURCE_POLICY = "same_actor_family_iid_lexical_cyclic_next_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")

AUTHORITY = {
    "measurement_runtime_qualified": False,
    "identity_authority": False,
    "identity_preservation_verified": False,
    "event_authority": False,
    "event_verified": False,
    "source_binding_authority": False,
    "absolute_camera_or_technical_success_claimed": False,
    "scientific_claim_authorized": False,
    "ranking_authorized": False,
    "candidate_selection_allowed": False,
    "training_allowed": False,
    "training_target_authorized": False,
    "optimizer_step_allowed": False,
    "optimizer_or_parameter_update_authorized": False,
}

_BASE_ATTEMPT_FIELDS = {
    "candidate_id", "ordinal", "iid", "row_id", "actor_family",
    "analysis_split", "branch", "seed", "receipt_path", "receipt_sha256",
    "receipt_digest", "native_receipt_path", "native_receipt_sha256",
    "native_receipt_digest", "candidate_envelope_path",
    "candidate_envelope_sha256", "mp4_path", "mp4_sha256",
    "declared_frame_count", "declared_fps", "upstream_event_verified",
    "upstream_identity_preservation_verified",
    "upstream_selection_authorized", "upstream_training_target_authorized",
}
_ATTEMPT_FIELDS = _BASE_ATTEMPT_FIELDS | {"correct_source", "wrong_source"}
_SOURCE_FIELDS = {
    "iid", "row_id", "analysis_split", "actor_family", "actor_group_id",
    "scene_group_id", "source_video", "source_video_sha256",
}
_SUMMARY_METRICS = {
    "camera_global_mean_xy_l2_difference_mean": (
        "camera_trajectory", "global_mean_xy_l2_difference_mean"
    ),
    "camera_global_mean_xy_l2_difference_p90": (
        "camera_trajectory", "global_mean_xy_l2_difference_p90"
    ),
    "camera_global_mean_xy_l2_difference_max": (
        "camera_trajectory", "global_mean_xy_l2_difference_max"
    ),
    "camera_global_speed_mean_absolute_difference": (
        "camera_trajectory", "global_speed_mean_absolute_difference"
    ),
    "camera_cumulative_global_endpoint_l2_difference": (
        "camera_trajectory", "cumulative_global_endpoint_l2_difference"
    ),
    "scene_cut_ratio_absolute_difference": (
        "scene_cut_ratio_absolute_difference",
    ),
    "temporal_energy_cv_absolute_difference": (
        "temporal_energy_cv_absolute_difference",
    ),
    "technical_sharpness_retention_diagnostic": (
        "technical", "sharpness_retention_diagnostic"
    ),
    "technical_candidate_exposure_diagnostic": (
        "technical", "candidate_exposure_diagnostic"
    ),
    "technical_nonfreeze_retention_diagnostic": (
        "technical", "nonfreeze_retention_diagnostic"
    ),
    "technical_global_flicker_agreement_diagnostic": (
        "technical", "global_flicker_agreement_diagnostic"
    ),
    "technical_geometric_mean_diagnostic": (
        "technical", "geometric_mean_technical_diagnostic"
    ),
}


class Exact47MediaRawError(RuntimeError):
    """An input, runtime, media, coverage, or authority invariant failed."""


def _configure_frozen_input() -> None:
    """Restore both validator layers after any same-interpreter import mutation."""
    frozen_specialization._configure_core()
    frozen_specialization.core.partition_indices = frozen_specialization.partition_indices
    expected_path = (METHOD_ROOT / "diagnose_saic_partial33_source_bound_dinov2_raw_v1.py").resolve(
        strict=True
    )
    if (
        frozen_input is not frozen_specialization.core
        or Path(frozen_input.__file__).resolve(strict=True) != expected_path
        or Path(frozen_input.core.__file__).resolve(strict=True) != expected_path
        or frozen_input.EXPECTED_ATTEMPT_COUNT != EXPECTED_CANDIDATE_COUNT
        or frozen_input.EXPECTED_WORLD_SIZE != 8
        or frozen_input.core.EXPECTED_ATTEMPT_COUNT != EXPECTED_CANDIDATE_COUNT
        or frozen_input.core.EXPECTED_WORLD_SIZE != 8
    ):
        raise Exact47MediaRawError("frozen exact33 input-validator identity differs")


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
        raise Exact47MediaRawError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Exact47MediaRawError(message)


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise Exact47MediaRawError(f"{label} must be lowercase SHA-256")
    return value


def _closed(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise Exact47MediaRawError(f"{label} field closure differs")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise Exact47MediaRawError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Exact47MediaRawError(f"{label} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Exact47MediaRawError(f"{label} must be a plain non-symlink file")
    return path.resolve(strict=True)


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise Exact47MediaRawError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Exact47MediaRawError(f"{label} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise Exact47MediaRawError(f"{label} must be a plain non-symlink directory")
    return path.resolve(strict=True)


def _stable_hash(path_value: Any, hash_value: Any, *, label: str) -> Path:
    path = _plain_file(path_value, label=label)
    expected = _sha(hash_value, label=f"{label} SHA-256")
    before = path.stat()
    observed = file_sha256(path)
    after = path.stat()
    if observed != expected or (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise Exact47MediaRawError(f"{label} hash or identity differs")
    return path


def _inside(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Exact47MediaRawError(f"{label} escaped attempts root") from error


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if not path.is_absolute() or path.parent.is_symlink() or not path.parent.is_dir():
        raise Exact47MediaRawError("receipt parent must be an existing plain directory")
    raw = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o400)
    except OSError as error:
        raise Exact47MediaRawError(f"refusing to overwrite receipt: {path}") from error
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _verify_self(expected_sha256: str) -> str:
    expected = _sha(expected_sha256, label="batch diagnostic source SHA-256")
    actual = file_sha256(Path(__file__).resolve(strict=True))
    if actual != expected:
        raise Exact47MediaRawError("batch diagnostic source SHA-256 differs")
    return actual


def _validate_source_binding(value: Any, *, label: str) -> Mapping[str, Any]:
    row = _closed(value, _SOURCE_FIELDS, label=label)
    _sha(row.get("source_video_sha256"), label=f"{label} video SHA-256")
    _require(type(row.get("iid")) is str, f"{label} IID differs")
    return row


def load_and_validate_input_manifest(
    path: str | Path, *, expected_sha256: str
) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    _configure_frozen_input()
    expected = _sha(expected_sha256, label="input manifest SHA-256")
    if expected != EXPECTED_INPUT_MANIFEST_SHA256:
        raise Exact47MediaRawError("only the sealed r5 exact33 input manifest is accepted")
    try:
        manifest, raw_sha = frozen_input.load_input_manifest(
            path,
            expected_sha256=expected,
            expected_source_sha256=EXPECTED_FROZEN_INPUT_SOURCE_SHA256,
        )
    except Exception as error:
        raise Exact47MediaRawError(
            f"sealed source-bound input manifest validation failed: {error}"
        ) from error
    if raw_sha != EXPECTED_INPUT_MANIFEST_SHA256:
        raise Exact47MediaRawError("sealed r5 exact33 input manifest SHA-256 differs")
    if (
        manifest.get("root_spec_raw_sha256") != EXPECTED_ROOT_SPEC_SHA256
        or manifest.get("attempt_count") != EXPECTED_CANDIDATE_COUNT
        or manifest.get("world_size") != 8
        or manifest.get("partition_rule")
        != "candidate_order_index_modulo_world_size"
        or manifest.get("selected_frame_indices") != EXPECTED_SELECTED_FRAME_INDICES
    ):
        raise Exact47MediaRawError("sealed r5 exact33 bank contract differs")

    evidence = manifest.get("source_manifest")
    _closed(
        evidence,
        {
            "path", "raw_sha256", "content_sha256", "validator_summary_sha256",
            "bound_files_verified", "wrong_source_policy",
        },
        label="source manifest evidence",
    )
    if (
        evidence.get("raw_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or evidence.get("content_sha256")
        != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
        or evidence.get("validator_summary_sha256")
        != EXPECTED_SOURCE_VALIDATOR_SUMMARY_SHA256
        or evidence.get("bound_files_verified") is not True
        or evidence.get("wrong_source_policy") != EXPECTED_WRONG_SOURCE_POLICY
    ):
        raise Exact47MediaRawError("sealed eight-source manifest evidence differs")
    try:
        sources, policy = frozen_input._source_closure(
            evidence["path"], evidence["raw_sha256"]
        )
    except Exception as error:
        raise Exact47MediaRawError(f"eight-source closure replay failed: {error}") from error
    if (
        len(sources) != EXPECTED_REGISTERED_SOURCE_COUNT
        or canonical_json_bytes(policy.get("evidence")) != canonical_json_bytes(evidence)
    ):
        raise Exact47MediaRawError("registered eight-source closure differs")

    attempts_root = _plain_directory(manifest.get("attempts_root"), label="attempts root")
    rows = manifest.get("attempts")
    if type(rows) is not list or len(rows) != EXPECTED_CANDIDATE_COUNT:
        raise Exact47MediaRawError("sealed bank must contain exactly 33 attempts")
    receipt_paths = sorted(
        attempts_root.rglob(frozen_input.core.ATTEMPT_BASENAME),
        key=lambda item: item.as_posix(),
    )
    sealed_receipt_paths = sorted(
        _plain_file(row.get("receipt_path"), label="sealed generation receipt")
        for row in rows if isinstance(row, Mapping)
    )
    if len(receipt_paths) != EXPECTED_CANDIDATE_COUNT or receipt_paths != sealed_receipt_paths:
        raise Exact47MediaRawError("attempt root is not the sealed exact33 receipt set")

    candidate_ids: list[str] = []
    mp4_paths: set[str] = set()
    mp4_hashes: set[str] = set()
    candidate_mp4_bindings: set[tuple[str, str]] = set()
    used_sources: set[str] = set()
    validated_by_id: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(rows):
        row = _closed(value, _ATTEMPT_FIELDS, label=f"attempt[{index}]")
        candidate_id = row.get("candidate_id")
        if type(candidate_id) is not str or _SAFE_ID.fullmatch(candidate_id) is None:
            raise Exact47MediaRawError("candidate ID is not path-safe")
        try:
            replayed = frozen_input.core.validate_attempt_receipt(
                row["receipt_path"],
                expected_root_spec_sha256=EXPECTED_ROOT_SPEC_SHA256,
            )
        except Exception as error:
            raise Exact47MediaRawError(
                f"generation receipt replay failed for {candidate_id}: {error}"
            ) from error
        sealed_base = {key: row[key] for key in _BASE_ATTEMPT_FIELDS}
        if canonical_json_bytes(sealed_base) != canonical_json_bytes(replayed):
            raise Exact47MediaRawError("generation receipt replay differs from sealed row")

        correct = _validate_source_binding(
            row.get("correct_source"), label=f"{candidate_id} correct source"
        )
        wrong = _validate_source_binding(
            row.get("wrong_source"), label=f"{candidate_id} wrong source"
        )
        iid = row.get("iid")
        expected_correct = sources.get(iid)
        expected_wrong_iid = policy.get("wrong_by_iid", {}).get(iid)
        expected_wrong = sources.get(expected_wrong_iid)
        if (
            expected_correct is None
            or expected_wrong is None
            or canonical_json_bytes(correct) != canonical_json_bytes(expected_correct)
            or canonical_json_bytes(wrong) != canonical_json_bytes(expected_wrong)
            or correct.get("iid") == wrong.get("iid")
            or correct.get("source_video_sha256") == wrong.get("source_video_sha256")
        ):
            raise Exact47MediaRawError("correct/wrong source closure differs")

        receipt_path = _plain_file(row["receipt_path"], label="generation receipt")
        mp4_path = _stable_hash(
            row["mp4_path"], row["mp4_sha256"], label="candidate MP4"
        )
        _inside(receipt_path, attempts_root, label="generation receipt")
        _inside(mp4_path, attempts_root, label="candidate MP4")
        candidate_ids.append(candidate_id)
        mp4_paths.add(str(mp4_path))
        mp4_hashes.add(row["mp4_sha256"])
        candidate_mp4_bindings.add((candidate_id, row["mp4_sha256"]))
        used_sources.add(correct["iid"])
        validated_by_id[candidate_id] = dict(row)

    if candidate_ids != sorted(candidate_ids):
        raise Exact47MediaRawError("candidate order is not the sealed lexical order")
    if (
        len(set(candidate_ids)) != EXPECTED_CANDIDATE_COUNT
        or len(mp4_paths) != EXPECTED_CANDIDATE_COUNT
        or len(mp4_hashes) != EXPECTED_CANDIDATE_COUNT
        or len(candidate_mp4_bindings) != EXPECTED_CANDIDATE_COUNT
        or len(used_sources) != EXPECTED_CANDIDATE_BOUND_SOURCE_COUNT
        or len(validated_by_id) != EXPECTED_CANDIDATE_COUNT
    ):
        raise Exact47MediaRawError("exact33 candidate/MP4/source coverage differs")
    return manifest, raw_sha, validated_by_id


def _fresh_output_root(value: str | Path) -> tuple[Path, Path]:
    output = Path(value)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise Exact47MediaRawError("output root must be fresh, absolute, and non-root")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir() or output != parent / output.name:
        raise Exact47MediaRawError("output root parent or canonical spelling differs")
    output.mkdir(mode=0o700)
    diagnostic_root = output / "diagnostics"
    diagnostic_root.mkdir(mode=0o700)
    return output, diagnostic_root


def _diagnostic_task(row: Mapping[str, Any], *, output: Path) -> dict[str, str]:
    return {
        "candidate_id": row["candidate_id"],
        "source_video": row["correct_source"]["source_video"],
        "source_sha256": row["correct_source"]["source_video_sha256"],
        "candidate_video": row["mp4_path"],
        "candidate_sha256": row["mp4_sha256"],
        "output": str(output / f"{row['candidate_id']}.json"),
    }


def _worker(task: Mapping[str, str]) -> dict[str, str]:
    # ProcessPool workers may otherwise each construct a large OpenCV pool on
    # top of the outer 16-way process pool.  One OpenCV thread per worker keeps
    # this CPU-only 32-CPU step bounded and avoids hidden oversubscription.
    if hasattr(diagnostics.cv2, "setNumThreads"):
        diagnostics.cv2.setNumThreads(1)
    if hasattr(diagnostics.cv2, "ocl"):
        diagnostics.cv2.ocl.setUseOpenCL(False)
    value = diagnostics.build_diagnostic(
        source_video=task["source_video"],
        expected_source_sha256=task["source_sha256"],
        candidate_video=task["candidate_video"],
        expected_candidate_sha256=task["candidate_sha256"],
    )
    file_hash = diagnostics.write_diagnostic_create_only(task["output"], value)
    return {
        "candidate_id": task["candidate_id"],
        "diagnostic_digest": value["diagnostic_digest"],
        "diagnostic_file_sha256": file_hash,
    }


def _run_tasks(tasks: Sequence[Mapping[str, str]], *, workers: int) -> dict[str, dict[str, str]]:
    if type(workers) is not int or not 1 <= workers <= 32:
        raise Exact47MediaRawError("workers must be an integer in [1,32]")
    results: list[dict[str, str]] = []
    if workers == 1:
        results = [_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, task): task for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())
    indexed = {row["candidate_id"]: row for row in results}
    if len(results) != len(tasks) or len(indexed) != len(tasks):
        raise Exact47MediaRawError("diagnostic worker coverage differs")
    return indexed


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Exact47MediaRawError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Exact47MediaRawError(f"{label} is non-finite")
    return result


def _metric_projection(value: Mapping[str, Any]) -> dict[str, float]:
    comparisons = value.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise Exact47MediaRawError("diagnostic comparisons are absent")
    result: dict[str, float] = {}
    for label, keys in _SUMMARY_METRICS.items():
        current: Any = comparisons
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                raise Exact47MediaRawError(f"diagnostic metric is absent: {label}")
            current = current[key]
        result[label] = _number(current, label=label)
    return result


def descriptive_statistics(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    if not rows:
        raise Exact47MediaRawError("descriptive statistics require rows")
    if any(set(row) != set(_SUMMARY_METRICS) for row in rows):
        raise Exact47MediaRawError("descriptive metric field closure differs")
    result: dict[str, Any] = {}
    for key in _SUMMARY_METRICS:
        values = [_number(row[key], label=key) for row in rows]
        result[key] = {
            "count": len(values),
            "mean": _number(statistics.fmean(values), label=f"{key} mean"),
            "median": _number(statistics.median(values), label=f"{key} median"),
            "minimum": min(values),
            "maximum": max(values),
        }
    return result


def _validate_runtime(
    runtime: Any, *, expected_ffmpeg_sha256: str, expected_ffprobe_sha256: str
) -> dict[str, Any]:
    if not isinstance(runtime, Mapping):
        raise Exact47MediaRawError("diagnostic runtime identity is absent")
    try:
        if (
            runtime["implementation_sha256"] != EXPECTED_EXACT81_SOURCE_SHA256
            or runtime["decoded_evaluator_sha256"] != EXPECTED_DECODED_EVALUATOR_SHA256
            or runtime["geometry_sha256"] != EXPECTED_GEOMETRY_SHA256
            or runtime["ffmpeg"]["sha256"] != _sha(
                expected_ffmpeg_sha256, label="ffmpeg SHA-256"
            )
            or runtime["ffprobe"]["sha256"] != _sha(
                expected_ffprobe_sha256, label="ffprobe SHA-256"
            )
        ):
            raise Exact47MediaRawError("diagnostic runtime source/tool pins differ")
    except (KeyError, TypeError) as error:
        raise Exact47MediaRawError("diagnostic runtime identity is malformed") from error
    return json.loads(canonical_json_bytes(runtime).decode("ascii"))


def _validated_result(
    *,
    row: Mapping[str, Any],
    diagnostic_path: Path,
    worker_result: Mapping[str, str],
    expected_ffmpeg_sha256: str,
    expected_ffprobe_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = diagnostics.load_canonical_diagnostic(diagnostic_path)
    try:
        checked = diagnostics._validate_diagnostic_structure(value)
    except Exception as error:
        raise Exact47MediaRawError(f"diagnostic structure differs: {error}") from error
    raw_sha = file_sha256(diagnostic_path)
    media = checked.get("media")
    if not isinstance(media, Mapping):
        raise Exact47MediaRawError("diagnostic media closure is absent")
    expected_media = (
        (
            "source", row["correct_source"]["source_video"],
            row["correct_source"]["source_video_sha256"],
        ),
        ("candidate", row["mp4_path"], row["mp4_sha256"]),
    )
    for label, expected_path, expected_sha in expected_media:
        binding = media.get(label)
        decode = binding.get("decode") if isinstance(binding, Mapping) else None
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != expected_path
            or binding.get("sha256") != expected_sha
            or not isinstance(decode, Mapping)
            or decode.get("frame_count") != 81
            or decode.get("fps_numerator") != 25
            or decode.get("fps_denominator") != 1
        ):
            raise Exact47MediaRawError(f"diagnostic {label} exact81 binding differs")
    if (
        checked.get("authority") != dict(diagnostics.AUTHORITY)
        or any(checked["authority"].values())
        or worker_result.get("candidate_id") != row["candidate_id"]
        or worker_result.get("diagnostic_digest") != checked.get("diagnostic_digest")
        or worker_result.get("diagnostic_file_sha256") != raw_sha
        or checked.get("source", {}).get("motion_summary", {}).get("transition_count") != 80
        or checked.get("candidate", {}).get("motion_summary", {}).get("transition_count") != 80
    ):
        raise Exact47MediaRawError("diagnostic authority, seal, or full80 closure differs")
    runtime = _validate_runtime(
        checked.get("runtime"),
        expected_ffmpeg_sha256=expected_ffmpeg_sha256,
        expected_ffprobe_sha256=expected_ffprobe_sha256,
    )
    result = {
        "candidate_id": row["candidate_id"],
        "ordinal": row["ordinal"],
        "iid": row["iid"],
        "row_id": row["row_id"],
        "actor_family": row["actor_family"],
        "analysis_split": row["analysis_split"],
        "branch": row["branch"],
        "seed": row["seed"],
        "source_video_sha256": row["correct_source"]["source_video_sha256"],
        "candidate_mp4_sha256": row["mp4_sha256"],
        "diagnostic_path": str(diagnostic_path.resolve(strict=True)),
        "diagnostic_file_sha256": raw_sha,
        "diagnostic_digest": checked["diagnostic_digest"],
        "raw_descriptive_metrics": _metric_projection(checked),
        "thresholds": None,
        "authority": dict(AUTHORITY),
    }
    return result, runtime


def _common(args: Any) -> tuple[str, dict[str, Any], str, dict[str, dict[str, Any]]]:
    source_sha = _verify_self(args.expected_source_sha256)
    exact_source = _stable_hash(
        args.exact81_source,
        args.expected_exact81_source_sha256,
        label="exact81 diagnostic source",
    )
    if (
        args.expected_exact81_source_sha256 != EXPECTED_EXACT81_SOURCE_SHA256
        or exact_source != Path(diagnostics.__file__).resolve(strict=True)
    ):
        raise Exact47MediaRawError("imported exact81 diagnostic source differs")
    manifest, manifest_sha, by_id = load_and_validate_input_manifest(
        args.input_manifest,
        expected_sha256=args.expected_input_manifest_sha256,
    )
    return source_sha, manifest, manifest_sha, by_id


def preflight(args: Any) -> int:
    source_sha, manifest, manifest_sha, by_id = _common(args)
    output_root, diagnostic_root = _fresh_output_root(args.output_root)
    candidate_id = next(iter(by_id))
    row = by_id[candidate_id]
    task = _diagnostic_task(row, output=diagnostic_root)
    worker = _worker(task)
    result, runtime = _validated_result(
        row=row,
        diagnostic_path=Path(task["output"]),
        worker_result=worker,
        expected_ffmpeg_sha256=args.expected_ffmpeg_sha256,
        expected_ffprobe_sha256=args.expected_ffprobe_sha256,
    )
    unsigned = {
        "schema_version": PREFLIGHT_SCHEMA,
        "batch_source_sha256": source_sha,
        "input_manifest_path": str(_plain_file(args.input_manifest, label="input manifest")),
        "input_manifest_sha256": manifest_sha,
        "root_spec_raw_sha256": manifest["root_spec_raw_sha256"],
        "source_manifest": dict(manifest["source_manifest"]),
        "registered_source_count": EXPECTED_REGISTERED_SOURCE_COUNT,
        "candidate_count": 1,
        "coverage": "one_sealed_lexical_first_candidate_only_not_batch",
        "candidate_result": result,
        "runtime": runtime,
        "interpretation": {
            "measurement": "camera_technical_temporal_consistency_raw_diagnostic_only",
            "preflight_only": True,
            "thresholds": None,
            "ranking_or_selection_performed": False,
        },
        "authority": dict(AUTHORITY),
    }
    _write_create_only(
        output_root / "preflight-receipt.json",
        {**unsigned, "receipt_digest": object_sha256(unsigned)},
    )
    return 0


def run_full(args: Any) -> int:
    source_sha, manifest, manifest_sha, by_id = _common(args)
    output_root, diagnostic_root = _fresh_output_root(args.output_root)
    ordered_rows = [by_id[row["candidate_id"]] for row in manifest["attempts"]]
    tasks = [_diagnostic_task(row, output=diagnostic_root) for row in ordered_rows]
    workers = _run_tasks(tasks, workers=args.workers)
    results: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    for row, task in zip(ordered_rows, tasks):
        result, runtime = _validated_result(
            row=row,
            diagnostic_path=Path(task["output"]),
            worker_result=workers[row["candidate_id"]],
            expected_ffmpeg_sha256=args.expected_ffmpeg_sha256,
            expected_ffprobe_sha256=args.expected_ffprobe_sha256,
        )
        results.append(result)
        runtimes.append(runtime)
    if (
        len(results) != EXPECTED_CANDIDATE_COUNT
        or len({row["candidate_id"] for row in results}) != EXPECTED_CANDIDATE_COUNT
        or len({row["diagnostic_path"] for row in results}) != EXPECTED_CANDIDATE_COUNT
        or len({row["diagnostic_file_sha256"] for row in results})
        != EXPECTED_CANDIDATE_COUNT
        or any(canonical_json_bytes(runtime) != canonical_json_bytes(runtimes[0]) for runtime in runtimes)
    ):
        raise Exact47MediaRawError("full aggregate exact33/runtime coverage differs")
    used_source_iids = sorted({row["iid"] for row in results})
    unsigned = {
        "schema_version": AGGREGATE_SCHEMA,
        "batch_source": {
            "path": str(Path(__file__).resolve(strict=True)),
            "sha256": source_sha,
        },
        "input_validator_source": {
            "path": str(Path(frozen_input.__file__).resolve(strict=True)),
            "sha256": EXPECTED_FROZEN_INPUT_SOURCE_SHA256,
        },
        "input_manifest": {
            "path": str(_plain_file(args.input_manifest, label="input manifest")),
            "raw_sha256": manifest_sha,
            "receipt_digest": manifest["receipt_digest"],
        },
        "root_spec_raw_sha256": manifest["root_spec_raw_sha256"],
        "source_manifest": dict(manifest["source_manifest"]),
        "registered_source_count": EXPECTED_REGISTERED_SOURCE_COUNT,
        "candidate_bound_source_count": len(used_source_iids),
        "candidate_bound_source_iids": used_source_iids,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "coverage": "exactly_once_complete_sealed_r5_partial33_exact81_media_raw",
        "candidate_order": [row["candidate_id"] for row in results],
        "candidate_results": results,
        "descriptive_statistics_in_sealed_candidate_order": descriptive_statistics(
            [row["raw_descriptive_metrics"] for row in results]
        ),
        "runtime_source_and_tool_closure": runtimes[0],
        "interpretation": {
            "measurement": "camera_technical_temporal_consistency_raw_diagnostic_only",
            "exact81_frames": 81,
            "full_transitions": 80,
            "identity_appearance_background_non_target_event_source_bind_inverse": "unavailable",
            "thresholds": None,
            "descriptive_statistics_only": True,
            "ranking_or_selection_performed": False,
            "no_absolute_success_or_preservation_claim": True,
        },
        "authority": dict(AUTHORITY),
    }
    _write_create_only(
        output_root / "aggregate-receipt.json",
        {**unsigned, "receipt_digest": object_sha256(unsigned)},
    )
    for path in diagnostic_root.iterdir():
        if path.is_file() and not path.is_symlink():
            path.chmod(0o400)
    diagnostic_root.chmod(0o500)
    output_root.chmod(0o500)
    return 0


def _common_arguments(parser: Any) -> None:
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--exact81-source", required=True)
    parser.add_argument("--expected-exact81-source-sha256", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--expected-ffmpeg-sha256", required=True)
    parser.add_argument("--expected-ffprobe-sha256", required=True)
    parser.add_argument("--output-root", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    _common_arguments(check)
    full = commands.add_parser("full")
    _common_arguments(full)
    full.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    args = build_parser().parse_args(argv)
    return {"preflight": preflight, "full": run_full}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
