#!/usr/bin/env python3
"""Author the narrow case01 source-bone intervention renderer plan.

This module does not launch a renderer, training, SSH, or Slurm.  It records
one rejected four-source candidate so its provenance is inspectable while the
replacement interventions are audited.  A later production plan can use the
same matched design to distinguish source-object reuse from a text-prompt
shortcut while keeping the successful full644 R64 inference coordinates
identical:

``original``
    The exact held-out source bytes.
``removed``
    The source bone is covered by deterministic ffmpeg ``removelogo`` spatial
    interpolation (candidate r4; rejected because its rectangular tube is
    conspicuous).
``translated``
    The same source bone is moved 150 pixels upward; the original location
    uses the same removal treatment as ``removed``.
``sham``
    The source bone remains while an unrelated region receives a comparable
    deterministic ffmpeg spatial-interpolation perturbation.

The existing r5g/r5f top-level plan validator is intentionally not reused:
it is closed over the exact historical shared8 source SHA-256 values and 16
task IDs.  Its lower-level retained-source path is compatible with alternate
absolute source paths, so a new narrow controller can consume the tasks
emitted here without changing ``infer_lora.py`` or the frozen runner files.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "bernini-case01-source-bone-intervention-plan-v1"
EXPERIMENT_ID = "case01-288545b9c031491a-source-bone-intervention-r1"
REFERENCE_PLAN_SHA256 = (
    "097b601d180ee7122230fa7d98dcac9c7102489195c065a6d03eb7e38131dfbe"
)
REFERENCE_PLAN_DIGEST = (
    "2136926b734796333788ab9f296e6cca076989bfc25b7a589eae773d42b61a00"
)
IID = "288545b9c031491a"
INSTRUCTION = "Make the dog pick up the bone and hold it in its mouth."
INSTRUCTION_SHA256 = (
    "84df12ede824d239a4c7c3d21dccdf22663535d1e504e7b280544c8a9be0fd5d"
)
SEED = 2027
NUM_INFERENCE_STEPS = 40
SOURCE_ONSET_POLICY = "none"
VARIANT_ORDER = ("original", "removed", "translated", "sham")
CANDIDATE_VARIANT_SHA256 = {
    "original": (
        "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
    ),
    "removed": (
        "218d9420f2a20bf2d50ad300218b5d1ea14aae4c28f0fad498b6e2de1d5fad49"
    ),
    "translated": (
        "6c9d2a12c884d111cb8d4a1cdbb627132b2ce3e4b760b46b52ddf4ea37a8eea3"
    ),
    "sham": (
        "007694ce136eb8353299b22ad3c904c106ea43b94ece3b5302cb87619b72d80c"
    ),
}
ASSET_AUTHORITY_STATUS = "REJECTED_FIXED_RECTANGULAR_BLUR_TUBE_DO_NOT_LAUNCH"
EXPECTED_VIDEO = {
    "frame_count": 81,
    "fps_num": 25,
    "fps_den": 1,
    "width": 704,
    "height": 736,
}
FULL644_PROFILE = "full644-r64-reference-dpo-preservation-one-pass-v1"
EXPECTED_CHECKPOINT = {
    "sha256": "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2",
    "manifest_digest": "7bae23da51a3c5a67adb41ee85dd026c374d2581bd3409e868e18b2f6f4dffc4",
    "global_step": 644,
    "receipt_digest": "aaf348a7daa6c5ca2fe721771857287125ee02eb2c9a499f45b11a2e113d15d7",
    "file_count": 5,
    "adapter_config_sha256": "94bfaf73d714d7e77095ff68ce57e24932e0c05bde324263f5fe321660b95f62",
    "adapter_model_sha256": "44efdc5a0501238250b1d32ae2859abe248ffc37b152cd8db86ff84b378d6b22",
    "training_receipt_sha256": "3402c8c93c092bfc4490bf86790ab6429b4cbaad38358956cb0beeb5df7d4c4c",
    "optimizer_sha256": "77b7b22db4da92f28f23b4ae91c7271f55ab6a92353bfc8b0bbeb30529a7af63",
}
EXPECTED_PRODUCER = {
    "inference_receipt_schema": "bernini-r-1p3b-action-lora-inference-receipt-v5",
    "infer_lora_sha256": "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
    "method_source_revision": "ce4cffc1e8a144448c92252d9fb63087f03bbd8c",
    "method_source_archive_sha256": "12a28ddec99704963af42f1a82b09dff31828e3af8e53e5d0bbd0d43db272828",
    "ffprobe_sha256": "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5",
}
STOP_TUPLE = (
    "ASSET_AUTHORITY_REJECTED_DO_NOT_LAUNCH",
    "INPUT_INTERVENTION_VISUAL_REVIEW_NOT_APPROVED",
    "ANY_SOURCE_SHA_OR_81F_25FPS_704X736_GEOMETRY_DIFFERS",
    "ANY_NATIVE_RECEIPT_DIFFERS_ON_INSTRUCTION_SEED_SAMPLER_OR_CHECKPOINT",
    "ANY_OF_FOUR_R64_TASKS_FAILS_OR_IS_MISSING",
    "BLIND_OBJECT_REUSE_REVIEW_INCOMPLETE",
    "LARGE_SCALE_OBJECT_ADAPTER_TRAINING_FORBIDDEN_AT_STAGE0",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class InterventionPlanError(RuntimeError):
    """A matched case01 intervention-plan invariant differs."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise InterventionPlanError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_file(
    path_value: str | Path,
    *,
    expected_sha256: str | None = None,
    return_bytes: bool = False,
) -> tuple[bytes | None, str, int]:
    path = Path(path_value).expanduser()
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise InterventionPlanError(f"file is not one canonical absolute path: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
            if return_bytes:
                chunks.append(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    identity = lambda row: (  # noqa: E731 - compact immutable stat projection
        row.st_dev,
        row.st_ino,
        row.st_uid,
        row.st_gid,
        row.st_mode,
        row.st_nlink,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    observed = digest.hexdigest()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or identity(before) != identity(after)
        or identity(before) != identity(named)
        or size != before.st_size
        or (expected_sha256 is not None and observed != expected_sha256)
    ):
        raise InterventionPlanError(f"stable file identity/SHA differs: {path}")
    return (b"".join(chunks) if return_bytes else None), observed, size


def _strict_json_file(
    path_value: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, _, _ = _stable_file(
        path_value, expected_sha256=expected_sha256, return_bytes=True
    )
    if raw is None:
        raise InterventionPlanError("stable JSON reader returned no bytes")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InterventionPlanError("JSON authority cannot be decoded") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise InterventionPlanError("JSON authority is not canonical JSON plus LF")
    return value


def validate_reference_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    tasks = value.get("tasks")
    checkpoint = value.get("checkpoint_manifest")
    producer = value.get("producer")
    if (
        value.get("schema_version")
        != "bernini-full644-exploratory-matched-eval-plan-v1"
        or value.get("plan_digest") != REFERENCE_PLAN_DIGEST
        or object_sha256({key: item for key, item in value.items() if key != "plan_digest"})
        != REFERENCE_PLAN_DIGEST
        or not isinstance(tasks, list)
        or len(tasks) != 16
        or not isinstance(checkpoint, dict)
        or any(checkpoint.get(key) != expected for key, expected in EXPECTED_CHECKPOINT.items())
        or not isinstance(checkpoint.get("path"), str)
        or not Path(checkpoint["path"]).is_absolute()
        or not isinstance(producer, dict)
        or any(producer.get(key) != expected for key, expected in EXPECTED_PRODUCER.items())
        or not isinstance(producer.get("infer_lora_path"), str)
        or not Path(producer["infer_lora_path"]).is_absolute()
        or not isinstance(producer.get("ffprobe_path"), str)
        or not Path(producer["ffprobe_path"]).is_absolute()
    ):
        raise InterventionPlanError("reference full644 matched plan identity differs")
    case01 = [task for task in tasks if task.get("case_index") == 1]
    if (
        len(case01) != 2
        or {task.get("arm") for task in case01} != {"base", "full644"}
        or any(task.get("iid") != IID for task in case01)
        or any(task.get("instruction") != INSTRUCTION for task in case01)
        or any(task.get("instruction_sha256") != INSTRUCTION_SHA256 for task in case01)
        or any(task.get("seed") != SEED for task in case01)
        or any(task.get("num_inference_steps") != NUM_INFERENCE_STEPS for task in case01)
        or any(task.get("source_onset_policy") != SOURCE_ONSET_POLICY for task in case01)
    ):
        raise InterventionPlanError("reference case01 matched coordinates differ")
    return dict(value)


def probe_video(path: Path, ffprobe: Path) -> dict[str, int]:
    if not ffprobe.is_absolute() or ffprobe.resolve(strict=True) != ffprobe:
        raise InterventionPlanError("ffprobe path is not canonical")
    try:
        result = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise InterventionPlanError(f"ffprobe failed for {path}") from error
    try:
        payload = json.loads(result.stdout.decode("utf-8"))
        streams = payload["streams"]
        stream = streams[0]
        rate = Fraction(stream["r_frame_rate"])
        average = Fraction(stream["avg_frame_rate"])
        observed = {
            "frame_count": int(stream["nb_frames"]),
            "fps_num": rate.numerator,
            "fps_den": rate.denominator,
            "width": int(stream["width"]),
            "height": int(stream["height"]),
        }
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise InterventionPlanError("ffprobe response schema differs") from error
    if len(streams) != 1 or average != rate or observed != EXPECTED_VIDEO:
        raise InterventionPlanError(f"source video geometry differs: {path}")
    return observed


def _variant_semantics() -> dict[str, dict[str, Any]]:
    return {
        "original": {
            "bone_present": True,
            "bone_position": "source_original",
            "original_bone_region_treatment": "none",
            "sham_region_treatment": "none",
        },
        "removed": {
            "bone_present": False,
            "bone_position": "absent",
            "original_bone_region_treatment": (
                "deterministic_ffmpeg_removelogo_spatial_interpolation_r4"
            ),
            "sham_region_treatment": "none",
        },
        "translated": {
            "bone_present": True,
            "bone_position": "same_source_pixels_shift_y_minus_150",
            "original_bone_region_treatment": (
                "same_deterministic_ffmpeg_removelogo_spatial_interpolation_r4_as_removed"
            ),
            "sham_region_treatment": "none",
        },
        "sham": {
            "bone_present": True,
            "bone_position": "source_original",
            "original_bone_region_treatment": "none",
            "sham_region_treatment": (
                "unrelated_region_deterministic_ffmpeg_spatial_interpolation_r2"
            ),
        },
    }


def build_plan(
    *,
    reference_plan: Mapping[str, Any],
    reference_plan_path: Path,
    sources: Mapping[str, Path],
    output_root: Path,
    arms: tuple[str, ...] = ("full644",),
    probe: Callable[[Path], Mapping[str, int]],
) -> dict[str, Any]:
    reference = validate_reference_plan(reference_plan)
    if arms not in (("full644",), ("base", "full644")):
        raise InterventionPlanError("arms must be R64-only or matched Base+R64")
    if set(sources) != set(VARIANT_ORDER):
        raise InterventionPlanError("all four exact intervention sources are required")
    if (
        not output_root.is_absolute()
        or os.path.normpath(str(output_root)) != str(output_root)
        or not output_root.is_dir()
        or output_root.is_symlink()
        or output_root.resolve(strict=True) != output_root
    ):
        raise InterventionPlanError("output root must be an existing canonical directory")
    source_rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        path = sources[variant]
        _, sha256, size = _stable_file(
            path,
            expected_sha256=CANDIDATE_VARIANT_SHA256[variant],
            return_bytes=False,
        )
        geometry = dict(probe(path))
        if geometry != EXPECTED_VIDEO:
            raise InterventionPlanError(f"{variant} geometry differs")
        source_rows.append(
            {
                "variant": variant,
                "path": str(path),
                "sha256": sha256,
                "size": size,
                "geometry": geometry,
                "semantics": _variant_semantics()[variant],
                "human_visual_reviewed": False,
            }
        )
    tasks: list[dict[str, Any]] = []
    checkpoint = dict(reference["checkpoint_manifest"])
    for row in source_rows:
        for arm in arms:
            task_id = f"case01-bone-{row['variant']}-{arm}"
            video = output_root / f"{task_id}.mp4"
            receipt = video.with_name(video.name + ".receipt.json")
            if any(path.exists() or path.is_symlink() for path in (video, receipt)):
                raise InterventionPlanError(f"planned output is not fresh: {video}")
            tasks.append(
                {
                    "task_id": task_id,
                    "case_index": 1,
                    "iid": IID,
                    "intervention_variant": row["variant"],
                    "source_video": row["path"],
                    "source_video_sha256": row["sha256"],
                    "instruction": INSTRUCTION,
                    "instruction_sha256": INSTRUCTION_SHA256,
                    "seed": SEED,
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "source_onset_policy": SOURCE_ONSET_POLICY,
                    "arm": arm,
                    "adapter": (
                        None
                        if arm == "base"
                        else {
                            "checkpoint_root": str(Path(checkpoint["path"]).parent),
                            "checkpoint_manifest": checkpoint,
                            "adapter_model_sha256": checkpoint["adapter_model_sha256"],
                            "profile": FULL644_PROFILE,
                        }
                    ),
                    "output": {
                        "video_path": str(video),
                        "receipt_path": str(receipt),
                        "create_only": True,
                    },
                }
            )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "production_ready": False,
        "asset_authority": {
            "status": ASSET_AUTHORITY_STATUS,
            "candidate_hashes_only": True,
            "launch_allowed": False,
            "rejection_reason": "conspicuous_fixed_rectangular_blur_tube",
            "replacement_required": (
                "per_frame_SAM2_mask_dilate3_deterministic_ffmpeg_removelogo_"
                "spatial_interpolation_plus_spatially_matched_sham_v2"
            ),
            "codec_only_transcoded_present_control_required": True,
        },
        "reference_plan": {
            "path": str(reference_plan_path),
            "sha256": REFERENCE_PLAN_SHA256,
            "plan_digest": REFERENCE_PLAN_DIGEST,
        },
        "checkpoint_manifest": checkpoint,
        "producer": dict(reference["producer"]),
        "source_interventions": source_rows,
        "condition_contract": {
            "iid": IID,
            "instruction": INSTRUCTION,
            "instruction_sha256": INSTRUCTION_SHA256,
            "seed": SEED,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "source_onset_policy": SOURCE_ONSET_POLICY,
            "sampler_receipt_must_equal_reference_case01": True,
            "nominal_only_intended_treatment": "source_object_presence_or_position",
            "known_codec_container_confound": True,
            "codec_only_transcoded_present_control_required": True,
        },
        "arms": list(arms),
        "task_count": len(tasks),
        "tasks": tasks,
        "execution": {
            "launch_performed": False,
            "training_performed": False,
            "frozen_infer_lora_modified": False,
            "frozen_r5g_r5f_files_modified": False,
            "existing_top_level_runner_accepts_plan": False,
            "lower_level_retained_source_path_supports_alternate_absolute_paths": True,
            "new_narrow_controller_and_release_manifest_required": True,
            "candidate_plan_must_not_launch": True,
            "recommended_first_pass_after_asset_reaudit": (
                "five_R64_tasks_exact_original_codec_only_present_removed_translated_"
                "spatially_matched_sham_serial_same_model_capture"
            ),
            "base_followup_policy": "only_if_R64_factorial_is_ambiguous",
        },
        "decision_table": {
            "prompt_shortcut": (
                "removed still produces the same proxy prop and translated does not "
                "move source-object use"
            ),
            "source_object_dependence": (
                "removed suppresses same-instance reuse and translated changes the "
                "picked-up object location/identity while sham matches original"
            ),
            "intervention_artifact_confound": (
                "removed and translated both fail similarly while original and sham "
                "match; require better mask-localized removal/spatial interpolation "
                "before any model conclusion"
            ),
            "codec_container_confound": (
                "exact original differs in codec/container bytes from edited variants; "
                "require a codec-only transcoded-present control for a causal claim"
            ),
            "identity_path_still_missing": (
                "source-bone dependence is present but dog or bone fine identity still drifts"
            ),
        },
        "stop_tuple": list(STOP_TUPLE),
    }
    plan["plan_digest"] = object_sha256(plan)
    validate_plan(plan, reopen_sources=False)
    return plan


def validate_plan(
    plan: Mapping[str, Any], *, reopen_sources: bool = True,
    probe: Callable[[Path], Mapping[str, int]] | None = None,
) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "production_ready",
        "asset_authority",
        "reference_plan",
        "checkpoint_manifest",
        "producer",
        "source_interventions",
        "condition_contract",
        "arms",
        "task_count",
        "tasks",
        "execution",
        "decision_table",
        "stop_tuple",
        "plan_digest",
    }
    if not isinstance(plan, Mapping) or set(plan) != required:
        raise InterventionPlanError("plan root schema differs")
    unsigned = {key: value for key, value in plan.items() if key != "plan_digest"}
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("experiment_id") != EXPERIMENT_ID
        or plan.get("production_ready") is not False
        or plan.get("plan_digest") != object_sha256(unsigned)
        or plan.get("stop_tuple") != list(STOP_TUPLE)
        or plan.get("arms") not in (["full644"], ["base", "full644"])
    ):
        raise InterventionPlanError("plan identity/gate closure differs")
    asset_authority = plan.get("asset_authority")
    if (
        not isinstance(asset_authority, Mapping)
        or asset_authority.get("status") != ASSET_AUTHORITY_STATUS
        or asset_authority.get("candidate_hashes_only") is not True
        or asset_authority.get("launch_allowed") is not False
        or asset_authority.get("rejection_reason")
        != "conspicuous_fixed_rectangular_blur_tube"
        or asset_authority.get("codec_only_transcoded_present_control_required")
        is not True
    ):
        raise InterventionPlanError("rejected candidate asset boundary differs")
    reference = plan.get("reference_plan")
    checkpoint = plan.get("checkpoint_manifest")
    producer = plan.get("producer")
    if (
        not isinstance(reference, Mapping)
        or reference.get("sha256") != REFERENCE_PLAN_SHA256
        or reference.get("plan_digest") != REFERENCE_PLAN_DIGEST
        or not isinstance(reference.get("path"), str)
        or not Path(reference["path"]).is_absolute()
        or not isinstance(checkpoint, Mapping)
        or any(checkpoint.get(key) != value for key, value in EXPECTED_CHECKPOINT.items())
        or not isinstance(producer, Mapping)
        or any(producer.get(key) != value for key, value in EXPECTED_PRODUCER.items())
    ):
        raise InterventionPlanError("reference/checkpoint/producer closure differs")
    sources = plan.get("source_interventions")
    tasks = plan.get("tasks")
    arms = tuple(plan["arms"])
    expected_task_count = len(VARIANT_ORDER) * len(arms)
    if (
        not isinstance(sources, list)
        or [row.get("variant") for row in sources] != list(VARIANT_ORDER)
        or not isinstance(tasks, list)
        or len(tasks) != expected_task_count
        or plan.get("task_count") != expected_task_count
    ):
        raise InterventionPlanError("source/task factorial closure differs")
    source_by_variant: dict[str, Mapping[str, Any]] = {}
    for row in sources:
        variant = row.get("variant")
        path = Path(row.get("path", ""))
        if (
            set(row)
            != {
                "variant",
                "path",
                "sha256",
                "size",
                "geometry",
                "semantics",
                "human_visual_reviewed",
            }
            or variant not in VARIANT_ORDER
            or row.get("sha256") != CANDIDATE_VARIANT_SHA256[variant]
            or row.get("geometry") != EXPECTED_VIDEO
            or row.get("semantics") != _variant_semantics()[variant]
            or row.get("human_visual_reviewed") is not False
            or not path.is_absolute()
            or type(row.get("size")) is not int
            or row["size"] <= 0
        ):
            raise InterventionPlanError(f"source intervention differs: {variant}")
        if reopen_sources:
            _, observed, size = _stable_file(
                path,
                expected_sha256=CANDIDATE_VARIANT_SHA256[variant],
                return_bytes=False,
            )
            if observed != row["sha256"] or size != row["size"]:
                raise InterventionPlanError(f"source changed: {variant}")
            if probe is not None and dict(probe(path)) != EXPECTED_VIDEO:
                raise InterventionPlanError(f"source geometry changed: {variant}")
        source_by_variant[variant] = row
    expected_ids: list[str] = []
    condition_values: set[tuple[Any, ...]] = set()
    for variant in VARIANT_ORDER:
        for arm in arms:
            expected_id = f"case01-bone-{variant}-{arm}"
            expected_ids.append(expected_id)
            matching = [task for task in tasks if task.get("task_id") == expected_id]
            if len(matching) != 1:
                raise InterventionPlanError("task ID factorial differs")
            task = matching[0]
            source = source_by_variant[variant]
            condition_values.add(
                (
                    task.get("iid"),
                    task.get("instruction"),
                    task.get("instruction_sha256"),
                    task.get("seed"),
                    task.get("num_inference_steps"),
                    task.get("source_onset_policy"),
                )
            )
            adapter = task.get("adapter")
            output = task.get("output")
            if (
                task.get("case_index") != 1
                or task.get("iid") != IID
                or task.get("intervention_variant") != variant
                or task.get("source_video") != source["path"]
                or task.get("source_video_sha256") != source["sha256"]
                or task.get("instruction") != INSTRUCTION
                or task.get("instruction_sha256") != INSTRUCTION_SHA256
                or task.get("seed") != SEED
                or task.get("num_inference_steps") != NUM_INFERENCE_STEPS
                or task.get("source_onset_policy") != SOURCE_ONSET_POLICY
                or task.get("arm") != arm
                or (arm == "base" and adapter is not None)
                or (arm == "full644" and not isinstance(adapter, Mapping))
                or not isinstance(output, Mapping)
                or set(output) != {"video_path", "receipt_path", "create_only"}
                or output.get("create_only") is not True
                or not Path(output.get("video_path", "")).is_absolute()
                or Path(output.get("receipt_path", ""))
                != Path(output["video_path"]).with_name(
                    Path(output["video_path"]).name + ".receipt.json"
                )
            ):
                raise InterventionPlanError(f"task treatment differs: {expected_id}")
            if arm == "full644" and (
                adapter.get("profile") != FULL644_PROFILE
                or adapter.get("adapter_model_sha256")
                != EXPECTED_CHECKPOINT["adapter_model_sha256"]
                or adapter.get("checkpoint_manifest") != checkpoint
                or adapter.get("checkpoint_root")
                != str(Path(checkpoint["path"]).parent)
            ):
                raise InterventionPlanError("R64 adapter binding differs")
    if [task.get("task_id") for task in tasks] != expected_ids or len(condition_values) != 1:
        raise InterventionPlanError("task order or matched condition differs")
    execution = plan.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("launch_performed") is not False
        or execution.get("training_performed") is not False
        or execution.get("frozen_infer_lora_modified") is not False
        or execution.get("frozen_r5g_r5f_files_modified") is not False
        or execution.get("existing_top_level_runner_accepts_plan") is not False
        or execution.get(
            "lower_level_retained_source_path_supports_alternate_absolute_paths"
        )
        is not True
        or execution.get("new_narrow_controller_and_release_manifest_required")
        is not True
        or execution.get("candidate_plan_must_not_launch") is not True
    ):
        raise InterventionPlanError("execution boundary differs")


def write_create_only(path_value: str | Path, value: Mapping[str, Any]) -> str:
    path = Path(path_value).expanduser()
    if (
        not path.is_absolute()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.exists()
        or path.is_symlink()
    ):
        raise InterventionPlanError("output plan path must be fresh and canonical")
    payload = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise InterventionPlanError("create-only write made no progress")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _parse_sources(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise InterventionPlanError("--source must be VARIANT=/absolute/path")
        variant, path_raw = raw.split("=", 1)
        if variant in result or variant not in VARIANT_ORDER:
            raise InterventionPlanError(f"source variant differs: {variant}")
        result[variant] = Path(path_raw)
    if set(result) != set(VARIANT_ORDER):
        raise InterventionPlanError("exactly four --source arguments are required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-plan", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Repeat exactly four times as VARIANT=/absolute/source.mp4",
    )
    parser.add_argument("--probe-ffprobe", required=True)
    parser.add_argument("--probe-ffprobe-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Author eight Base+R64 tasks instead of the recommended four R64 tasks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference_path = Path(args.reference_plan)
    reference = _strict_json_file(
        reference_path, expected_sha256=REFERENCE_PLAN_SHA256
    )
    ffprobe = Path(args.probe_ffprobe)
    _stable_file(
        ffprobe,
        expected_sha256=args.probe_ffprobe_sha256,
        return_bytes=False,
    )
    sources = _parse_sources(args.source)
    plan = build_plan(
        reference_plan=reference,
        reference_plan_path=reference_path,
        sources=sources,
        output_root=Path(args.output_root),
        arms=("base", "full644") if args.include_base else ("full644",),
        probe=lambda path: probe_video(path, ffprobe),
    )
    sha256 = write_create_only(args.output_plan, plan)
    print(
        canonical_json_bytes(
            {
                "output_plan": str(Path(args.output_plan)),
                "sha256": sha256,
                "plan_digest": plan["plan_digest"],
                "task_count": plan["task_count"],
                "arms": plan["arms"],
                "launch_performed": False,
                "stop_tuple": list(STOP_TUPLE),
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
