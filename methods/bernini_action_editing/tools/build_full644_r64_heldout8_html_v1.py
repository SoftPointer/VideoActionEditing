#!/usr/bin/env python3
"""Build a portable review-only HTML site from one real r5g full16 run.

This builder is intentionally downstream of inference.  It never fabricates
media, fills missing cases, or assigns semantic scores.  The input bundle must
contain the exact current-R64 Shared8 full16 result closure::

    bundle/
      plan.json
      eval-report.json
      runner-attestation.json
      media/
        case00-source.mp4
        case00-base.mp4
        case00-base.mp4.receipt.json
        case00-full644.mp4
        case00-full644.mp4.receipt.json
        ... case07 ...

The three JSON authorities and every generated video/receipt must come from a
successful ``full16-production`` run.  The eight source files are copied from
the plan-bound Goku Heldout8 paths when assembling the local bundle.  Output is
created atomically and uses only relative paths, so ``index.html`` can be
opened directly or through a tiny local HTTP server.

This is a technical review page, not the stricter human-review publisher in
``md/action_editing/20260819_full644_exploratory_matched_eval/build_report.py``.
It therefore says exactly what the underlying plan authorizes: IID-disjoint
Heldout8 engineering diagnostics, not a formal training evaluation or a
scientific generalization claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "bernini-full644-exploratory-matched-eval-plan-v1"
REPORT_SCHEMA = "bernini-full644-exploratory-matched-eval-report-v2"
ATTESTATION_SCHEMA = "full644-exploratory-matched-runner-attestation-auh-r5"
RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-inference-receipt-v5"
SITE_SCHEMA = "bernini-full644-r64-heldout8-offline-review-site-v1"
FULL16_CAMPAIGN = "full16-production"
FULL644_PROFILE = "full644-r64-reference-dpo-preservation-one-pass-v1"
FULL644_ADAPTER_SHA256 = (
    "44efdc5a0501238250b1d32ae2859abe248ffc37b152cd8db86ff84b378d6b22"
)
FULL644_CHECKPOINT_MANIFEST_SHA256 = (
    "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2"
)
FULL644_TRAINING_RECEIPT_SHA256 = (
    "3402c8c93c092bfc4490bf86790ab6429b4cbaad38358956cb0beeb5df7d4c4c"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
CHECKPOINT_FRAMES = (0, 20, 40, 60, 80)
EXPECTED_MEDIA_FRAME_COUNT = 81
EXPECTED_MEDIA_FPS = Fraction(25, 1)

HELDOUT8 = (
    {
        "iid": "1852ada01d7c43a4",
        "split": "test",
        "instruction": (
            "Show the car driving dynamically through the snowy landscape, "
            "kicking up snow."
        ),
    },
    {
        "iid": "288545b9c031491a",
        "split": "test",
        "instruction": "Make the dog pick up the bone and hold it in its mouth.",
    },
    {
        "iid": "5ae88e1170c544b8",
        "split": "test",
        "instruction": (
            "Make the large, pink bubblegum bubble burst, leaving deflated "
            "remnants clinging around her mouth."
        ),
    },
    {
        "iid": "81473c034c1b4839",
        "split": "test",
        "instruction": (
            "Make the cat stand on its hind legs, with its front paws reaching "
            "up towards the window."
        ),
    },
    {
        "iid": "2766a3662fbf43d1",
        "split": "test",
        "instruction": (
            "Make the seagull on the railing spread its wings and begin to fly "
            "upwards and slightly to the right of its current position."
        ),
    },
    {
        "iid": "219c4c5f56e74b86",
        "split": "validation",
        "instruction": (
            "Extend the person's right arm forward to make contact with the "
            "punching bag."
        ),
    },
    {
        "iid": "2206cde2643e470a",
        "split": "validation",
        "instruction": "Make the man stand upright and release the fish into the water.",
    },
    {
        "iid": "7a2f54be92024a19",
        "split": "validation",
        "instruction": (
            "Have the person on the ledge jump into the water with arms outstretched."
        ),
    },
)
TASK_IDS = tuple(
    f"shared8-{case_index:02d}-{arm}"
    for case_index in range(8)
    for arm in ("base", "full644")
)


class SiteBuildError(RuntimeError):
    """One fail-closed publication check did not pass."""


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
        raise SiteBuildError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise SiteBuildError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def stable_file(path: Path, *, label: str) -> tuple[bytes, str, int]:
    """Read one unchanged, non-symlink regular file."""

    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise SiteBuildError(f"{label} is not a plain regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            fd_before = os.fstat(descriptor)
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            size = 0
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                chunks.append(block)
                size += len(block)
            fd_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = path.lstat()
    except SiteBuildError:
        raise
    except OSError as error:
        raise SiteBuildError(f"cannot stably read {label}: {path}: {error}") from error
    if not (
        _identity(before)
        == _identity(fd_before)
        == _identity(fd_after)
        == _identity(after)
    ) or size != before.st_size:
        raise SiteBuildError(f"{label} changed while being read: {path}")
    return b"".join(chunks), digest.hexdigest(), size


def load_json(path: Path, *, label: str) -> tuple[dict[str, Any], str, int]:
    raw, sha256, size = stable_file(path, label=label)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SiteBuildError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise SiteBuildError(f"{label} root is not an object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise SiteBuildError(f"{label} is not canonical JSON plus LF")
    return value, sha256, size


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SiteBuildError(f"{label} is not one lowercase SHA-256")
    return value


def require_digest(value: Mapping[str, Any], field: str, *, label: str) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    if not isinstance(claimed, str) or claimed != object_sha256(unsigned):
        raise SiteBuildError(f"{label} canonical {field} differs")
    return claimed


def require_true(value: Any, *, label: str) -> None:
    if type(value) is not bool or value is not True:
        raise SiteBuildError(f"{label} must be true")


def require_false(value: Any, *, label: str) -> None:
    if type(value) is not bool or value is not False:
        raise SiteBuildError(f"{label} must be false")


def validate_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise SiteBuildError("plan schema differs")
    require_digest(plan, "plan_digest", label="plan")
    require_true(plan.get("production_ready"), label="plan production_ready")
    if plan.get("pair_count") != 8 or plan.get("task_count") != 16:
        raise SiteBuildError("plan is not exact8/exact16")

    claims = plan.get("claim_limits")
    if not isinstance(claims, Mapping):
        raise SiteBuildError("plan claim limits are absent")
    expected_claims = {
        "content_disjoint_split": False,
        "evaluation_role": "engineering_diagnostic_only",
        "formal_claim_authorized": False,
        "historical_shared8_exposed": True,
        "human_reviewed_labels": False,
        "iid_heldout_diagnostic": True,
        "iid_overlap_with_full644": 0,
        "scientific_generalization_claim_authorized": False,
    }
    if dict(claims) != expected_claims:
        raise SiteBuildError("plan claim limits do not authorize this Heldout8 page")

    checkpoint = plan.get("checkpoint_manifest")
    if not isinstance(checkpoint, Mapping):
        raise SiteBuildError("plan checkpoint manifest is absent")
    if (
        checkpoint.get("global_step") != 644
        or checkpoint.get("adapter_model_sha256") != FULL644_ADAPTER_SHA256
        or checkpoint.get("sha256") != FULL644_CHECKPOINT_MANIFEST_SHA256
        or checkpoint.get("training_receipt_sha256")
        != FULL644_TRAINING_RECEIPT_SHA256
    ):
        raise SiteBuildError("plan is not bound to the current R64 checkpoint-644")

    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        raise SiteBuildError("plan execution contract is absent")
    for field in (
        "all_16_tasks_required_no_cherry_pick",
        "external_frozen_runner_attestation_required",
        "receipt_contract_alone_cannot_prove_process_execution",
    ):
        require_true(execution.get(field), label=f"plan execution.{field}")
    require_false(
        execution.get("training_or_inference_launched"),
        label="plan execution.training_or_inference_launched",
    )

    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 16:
        raise SiteBuildError("plan task closure is not exact16")
    if tuple(task.get("task_id") for task in tasks if isinstance(task, Mapping)) != TASK_IDS:
        raise SiteBuildError("plan task order is not Base then R64 for all Heldout8 cases")

    cases: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    for case_index, expected in enumerate(HELDOUT8):
        base = tasks[case_index * 2]
        adapted = tasks[case_index * 2 + 1]
        if not isinstance(base, Mapping) or not isinstance(adapted, Mapping):
            raise SiteBuildError(f"case {case_index:02d} plan row is not an object")
        for arm, task in (("base", base), ("full644", adapted)):
            if (
                task.get("case_index") != case_index
                or task.get("arm") != arm
                or task.get("iid") != expected["iid"]
                or task.get("instruction") != expected["instruction"]
                or task.get("seed") != 2026 + case_index
                or task.get("num_inference_steps") != 40
                or task.get("source_onset_policy") != "none"
            ):
                raise SiteBuildError(f"case {case_index:02d}/{arm} plan binding differs")
            instruction_sha = hashlib.sha256(
                expected["instruction"].encode("utf-8")
            ).hexdigest()
            if task.get("instruction_sha256") != instruction_sha:
                raise SiteBuildError(f"case {case_index:02d}/{arm} instruction SHA differs")
            source_sha = require_sha256(
                task.get("source_video_sha256"),
                label=f"case {case_index:02d}/{arm} source SHA",
            )
            source_path = task.get("source_video")
            if (
                not isinstance(source_path, str)
                or not source_path.endswith(f"/{expected['iid']}/source.mp4")
            ):
                raise SiteBuildError(f"case {case_index:02d}/{arm} source path differs")
            output = task.get("output")
            if not isinstance(output, Mapping) or output.get("create_only") is not True:
                raise SiteBuildError(f"case {case_index:02d}/{arm} output contract differs")
            basename = f"case{case_index:02d}-{arm}.mp4"
            if (
                Path(str(output.get("video_path", ""))).name != basename
                or Path(str(output.get("receipt_path", ""))).name
                != basename + ".receipt.json"
            ):
                raise SiteBuildError(f"case {case_index:02d}/{arm} output basename differs")
        pair_fields = (
            "iid",
            "instruction",
            "instruction_sha256",
            "seed",
            "source_onset_policy",
            "source_video",
            "source_video_sha256",
            "num_inference_steps",
        )
        if any(base.get(field) != adapted.get(field) for field in pair_fields):
            raise SiteBuildError(f"case {case_index:02d} matched pair differs")
        if base.get("adapter") is not None:
            raise SiteBuildError(f"case {case_index:02d} Base arm unexpectedly has an adapter")
        adapter = adapted.get("adapter")
        if not isinstance(adapter, Mapping):
            raise SiteBuildError(f"case {case_index:02d} R64 adapter binding is absent")
        nested_manifest = adapter.get("checkpoint_manifest")
        if (
            adapter.get("profile") != FULL644_PROFILE
            or adapter.get("adapter_model_sha256") != FULL644_ADAPTER_SHA256
            or not isinstance(nested_manifest, Mapping)
            or dict(nested_manifest) != dict(checkpoint)
        ):
            raise SiteBuildError(f"case {case_index:02d} is not current Full644/R64")
        source_hash = str(base["source_video_sha256"])
        source_hashes.add(source_hash)
        cases.append(
            {
                "case_index": case_index,
                "iid": expected["iid"],
                "split": expected["split"],
                "instruction": expected["instruction"],
                "instruction_sha256": base["instruction_sha256"],
                "seed": base["seed"],
                "source_sha256": source_hash,
            }
        )
    if len(source_hashes) != 8:
        raise SiteBuildError("Heldout8 source hashes are not unique")
    return cases


def validate_report(
    report: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise SiteBuildError("evaluation report schema differs")
    require_digest(report, "report_digest", label="evaluation report")
    if (
        report.get("plan_schema_version") != PLAN_SCHEMA
        or report.get("plan_digest") != plan.get("plan_digest")
        or report.get("pair_count") != 8
        or report.get("verified_task_count") != 16
    ):
        raise SiteBuildError("evaluation report is not bound to this exact16 plan")
    for field in (
        "all_16_tasks_verified_no_cherry_pick",
        "retained_publication_root_fd_replayed",
        "retained_ffprobe_executable_fd_replayed",
        "retained_publication_leaf_fds_replayed",
        "external_frozen_runner_attestation_still_required",
    ):
        require_true(report.get(field), label=f"report {field}")
    require_false(
        report.get("producer_execution_proven_by_receipt_contract"),
        label="report receipt-only execution claim",
    )
    if report.get("claim_limits") != plan.get("claim_limits"):
        raise SiteBuildError("report claim limits differ from plan")
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != 16:
        raise SiteBuildError("evaluation report result closure is not exact16")
    if tuple(row.get("task_id") for row in rows if isinstance(row, Mapping)) != TASK_IDS:
        raise SiteBuildError("evaluation report task order differs")
    results: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise SiteBuildError(f"report row {index} is not an object")
        task_id = TASK_IDS[index]
        case_index = index // 2
        arm = "base" if index % 2 == 0 else "full644"
        basename = f"case{case_index:02d}-{arm}.mp4"
        probe = raw_row.get("media_probe")
        if (
            raw_row.get("task_id") != task_id
            or raw_row.get("arm") != arm
            or Path(str(raw_row.get("output_path", ""))).name != basename
            or Path(str(raw_row.get("receipt_path", ""))).name
            != basename + ".receipt.json"
            or type(raw_row.get("output_size")) is not int
            or raw_row["output_size"] <= 0
            or not isinstance(probe, Mapping)
            or probe.get("frame_count") != EXPECTED_MEDIA_FRAME_COUNT
            or probe.get("fps_num") != EXPECTED_MEDIA_FPS.numerator
            or probe.get("fps_den") != EXPECTED_MEDIA_FPS.denominator
            or probe.get("stream_count") != 1
            or type(probe.get("width")) is not int
            or type(probe.get("height")) is not int
            or probe["width"] <= 0
            or probe["height"] <= 0
        ):
            raise SiteBuildError(f"report row {task_id} media contract differs")
        require_sha256(raw_row.get("output_sha256"), label=f"{task_id} output SHA")
        require_sha256(
            raw_row.get("receipt_file_sha256"), label=f"{task_id} receipt file SHA"
        )
        require_sha256(raw_row.get("receipt_digest"), label=f"{task_id} receipt digest")
        results[task_id] = dict(raw_row)
    return results


def validate_attestation(
    attestation: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    report: Mapping[str, Any],
    report_sha256: str,
) -> None:
    if attestation.get("schema_version") != ATTESTATION_SCHEMA:
        raise SiteBuildError("runner attestation schema differs")
    require_digest(attestation, "attestation_digest", label="runner attestation")
    if (
        attestation.get("status") != "COMPLETE"
        or attestation.get("campaign_mode") != FULL16_CAMPAIGN
        or attestation.get("task_count") != 16
        or attestation.get("task_ids") != list(TASK_IDS)
        or attestation.get("retry_count") != 0
        or attestation.get("unselected_task_count") != 0
    ):
        raise SiteBuildError("runner attestation is not one successful exact16 campaign")
    for field in (
        "formal_full16_report",
        "external_runner_attestation_present",
        "all_selected_tasks_attempted_exactly_once",
        "all_selected_tasks_succeeded",
        "same_model_capture_all_selected_tasks",
        "runner_task_json_replayed_for_all_tasks",
        "campaign_report_exactly_cross_linked_to_all_task_chains",
        "all_rank0_encoders_used_retained_ffmpeg_executable",
        "native_publication_before_parent_post_use_replay",
        "native_receipts_replayed_0400_single_link",
        "no_false_post_use_before_publication_claim",
        "receipt_contract_alone_did_not_prove_execution",
    ):
        require_true(attestation.get(field), label=f"attestation {field}")
    for field in (
        "manual_visual_review_required_before_full16",
        "scientific_claim_authorized",
        "formal_claim_authorized",
        "unselected_tasks_executed",
        "unselected_outputs_and_internal_artifacts_absent",
    ):
        require_false(attestation.get(field), label=f"attestation {field}")
    require_true(attestation.get("exploratory_only"), label="attestation exploratory_only")
    if (
        attestation.get("unselected_task_ids") != []
        or attestation.get("unselected_absence_row_count") != 0
    ):
        raise SiteBuildError("full16 attestation unselected-task closure differs")

    retained_root = attestation.get("retained_publication_root")
    retained_ffprobe = attestation.get("retained_ffprobe_executable")
    retained_tasks = attestation.get("retained_task_publications")
    if (
        not isinstance(retained_root, Mapping)
        or retained_root.get("held_through_attestation_publication") is not True
        or not isinstance(retained_ffprobe, Mapping)
        or retained_ffprobe.get("held_through_result_verification") is not True
        or not isinstance(retained_tasks, Mapping)
        or set(retained_tasks) != set(TASK_IDS)
        or any(
            not isinstance(row, Mapping)
            or row.get("held_through_result_verification") is not True
            for row in retained_tasks.values()
        )
    ):
        raise SiteBuildError("runner retained-publication authority closure differs")

    plan_binding = attestation.get("plan")
    report_binding = attestation.get("verified_report")
    if (
        not isinstance(plan_binding, Mapping)
        or plan_binding.get("sha256") != plan_sha256
        or plan_binding.get("plan_digest") != plan.get("plan_digest")
    ):
        raise SiteBuildError("attestation/plan binding differs")
    if (
        not isinstance(report_binding, Mapping)
        or report_binding.get("sha256") != report_sha256
        or report_binding.get("report_digest") != report.get("report_digest")
        or report_binding.get("verified_task_count") != 16
    ):
        raise SiteBuildError("attestation/report binding differs")


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_sha256: str,
    result: Mapping[str, Any],
    case: Mapping[str, Any],
    arm: str,
) -> None:
    task_id = f"shared8-{case['case_index']:02d}-{arm}"
    basename = f"case{case['case_index']:02d}-{arm}.mp4"
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise SiteBuildError(f"{task_id} receipt schema differs")
    receipt_digest = require_digest(receipt, "receipt_digest", label=f"{task_id} receipt")
    if (
        receipt_sha256 != result.get("receipt_file_sha256")
        or receipt_digest != result.get("receipt_digest")
    ):
        raise SiteBuildError(f"{task_id} receipt/report binding differs")
    require_true(receipt.get("experimental_inference"), label=f"{task_id} experimental")
    require_true(
        receipt.get("production_claim_forbidden"), label=f"{task_id} production forbidden"
    )
    require_false(
        receipt.get("scientific_claim_authorized"), label=f"{task_id} scientific claim"
    )

    input_row = receipt.get("input")
    output = receipt.get("output")
    sampling = receipt.get("sampling")
    adapter = receipt.get("adapter")
    if not all(isinstance(row, Mapping) for row in (input_row, output, sampling, adapter)):
        raise SiteBuildError(f"{task_id} receipt contract is incomplete")
    if (
        input_row.get("source_video_sha256") != case["source_sha256"]
        or input_row.get("instruction_utf8_sha256") != case["instruction_sha256"]
        or input_row.get("accepted_model_conditions")
        != ["source_video", "edit_instruction"]
        or input_row.get("target_accessed_by_inference") is not False
        or input_row.get("target_video_argument") is not False
        or input_row.get("reference_image_or_video") is not False
        or input_row.get("external_mask_or_swept_tube") is not False
        or input_row.get("external_tracking_pose_or_trajectory") is not False
        or input_row.get("external_shared_i0") is not False
    ):
        raise SiteBuildError(f"{task_id} inference input contract differs")
    if (
        Path(str(output.get("path", ""))).name != basename
        or output.get("sha256") != result.get("output_sha256")
        or output.get("size") != result.get("output_size")
        or output.get("frame_count") != EXPECTED_MEDIA_FRAME_COUNT
        or Fraction(str(output.get("fps"))) != EXPECTED_MEDIA_FPS
        or output.get("width") != result["media_probe"]["width"]
        or output.get("height") != result["media_probe"]["height"]
    ):
        raise SiteBuildError(f"{task_id} receipt output contract differs")
    if (
        sampling.get("num_frames") != EXPECTED_MEDIA_FRAME_COUNT
        or sampling.get("num_inference_steps") != 40
        or sampling.get("seed") != case["seed"]
        or sampling.get("source_onset_policy") != "none"
    ):
        raise SiteBuildError(f"{task_id} sampling contract differs")
    if arm == "base":
        if (
            adapter.get("enabled") is not False
            or adapter.get("mode") != "frozen_base_no_adapter"
        ):
            raise SiteBuildError(f"{task_id} is not the frozen Base arm")
    elif (
        adapter.get("enabled") is not True
        or adapter.get("profile") != FULL644_PROFILE
        or adapter.get("adapter_model_sha256") != FULL644_ADAPTER_SHA256
        or adapter.get("lora_rank") != 64
        or adapter.get("training_global_step") != 644
        or adapter.get("safe_merged_for_inference") is not True
        or adapter.get("strictly_reloaded") is not True
    ):
        raise SiteBuildError(f"{task_id} is not the current R64 arm")


def resolve_tool(value: str, *, label: str) -> Path:
    candidate = Path(value)
    if candidate.parent != Path(".") or candidate.is_absolute():
        resolved = candidate.resolve(strict=True)
    else:
        located = shutil.which(value)
        if located is None:
            raise SiteBuildError(f"cannot find {label}: {value}")
        resolved = Path(located).resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        raise SiteBuildError(f"{label} is not an executable regular file: {resolved}")
    return resolved


def probe_video(path: Path, ffprobe: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update({"LC_ALL": "C", "LANG": "C"})
    process = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v",
            "-show_entries",
            (
                "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,"
                "nb_frames,nb_read_frames"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        env=environment,
    )
    if process.returncode != 0:
        raise SiteBuildError(
            f"ffprobe failed for {path.name}: "
            + process.stderr.decode("utf-8", "replace")[:500]
        )
    try:
        payload = json.loads(process.stdout.decode("utf-8", "strict"))
        streams = payload["streams"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SiteBuildError(f"ffprobe output differs for {path.name}") from error
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise SiteBuildError(f"{path.name} does not contain exactly one video stream")
    stream = streams[0]
    count_raw = stream.get("nb_read_frames")
    if count_raw in (None, "N/A"):
        count_raw = stream.get("nb_frames")
    fps_raw = stream.get("avg_frame_rate")
    if fps_raw in (None, "0/0"):
        fps_raw = stream.get("r_frame_rate")
    try:
        frame_count = int(count_raw)
        fps = Fraction(str(fps_raw))
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise SiteBuildError(f"ffprobe media fields differ for {path.name}") from error
    if (
        stream.get("codec_name") != "h264"
        or frame_count != EXPECTED_MEDIA_FRAME_COUNT
        or fps != EXPECTED_MEDIA_FPS
        or width <= 0
        or height <= 0
    ):
        raise SiteBuildError(
            f"{path.name} is not H.264 81f@25fps with positive dimensions"
        )
    return {
        "codec": "h264",
        "frame_count": frame_count,
        "fps_num": fps.numerator,
        "fps_den": fps.denominator,
        "width": width,
        "height": height,
    }


def make_contact_sheet(video: Path, output: Path, ffmpeg: Path) -> None:
    expression = "+".join(f"eq(n\\,{frame})" for frame in CHECKPOINT_FRAMES)
    environment = dict(os.environ)
    environment.update({"LC_ALL": "C", "LANG": "C"})
    process = subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select={expression},scale=320:-2,tile=5x1",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        env=environment,
    )
    if process.returncode != 0:
        raise SiteBuildError(
            f"ffmpeg contact sheet failed for {video.name}: "
            + process.stderr.decode("utf-8", "replace")[:500]
        )
    raw, _, size = stable_file(output, label=f"contact sheet {output.name}")
    if size < 4 or not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        raise SiteBuildError(f"contact sheet is not a complete JPEG: {output.name}")


def _write_new(path: Path, raw: bytes, *, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, mode)
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise SiteBuildError(f"cannot create {path}: {error}") from error


def _copy_verified(
    source: Path, destination: Path, *, expected_sha256: str, label: str
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise SiteBuildError(f"staged destination is not fresh: {destination}")
    shutil.copyfile(source, destination, follow_symlinks=False)
    os.chmod(destination, 0o444)
    _, observed_sha256, size = stable_file(destination, label=label)
    if observed_sha256 != expected_sha256:
        raise SiteBuildError(f"published copy SHA differs: {destination.name}")
    return {"sha256": observed_sha256, "size": size}


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_html(
    cases: Sequence[Mapping[str, Any]],
    *,
    report_sha256: str,
    attestation_sha256: str,
    build_time: str,
) -> str:
    sections: list[str] = []
    for case in cases:
        case_index = int(case["case_index"])
        video_cells: list[str] = []
        sheet_cells: list[str] = []
        for role, title, subtitle in (
            ("source", "Source", "Heldout8 原始输入"),
            ("base", "Base", "当前同配置 frozen base"),
            ("full644", "Full644 / R64", "当前 checkpoint-644"),
        ):
            video = case["videos"][role]
            basename = video["basename"]
            probe = video["probe"]
            receipt_link = ""
            if role != "source":
                receipt_link = (
                    f' · <a href="assets/media/{_h(basename)}.receipt.json">receipt</a>'
                )
            video_cells.append(
                f'''<article class="video-card" data-role="{_h(role)}">
  <div class="video-head"><div><h3>{_h(title)}</h3><p>{_h(subtitle)}</p></div><button class="play-one" type="button">播放 / 暂停</button></div>
  <video controls muted playsinline preload="metadata" poster="assets/sheets/{_h(basename)}.jpg" src="assets/media/{_h(basename)}"></video>
  <div class="media-meta"><span>{probe['frame_count']}f · {probe['fps_num'] // probe['fps_den']}fps</span><span>{probe['width']}×{probe['height']}</span><span>H.264</span></div>
  <code title="{_h(video['sha256'])}">sha256 {_h(video['sha256'])}</code>
  <div class="asset-links"><a href="assets/media/{_h(basename)}">视频文件</a> · <a href="assets/sheets/{_h(basename)}.jpg">5 帧图</a>{receipt_link}</div>
</article>'''
            )
            sheet_cells.append(
                f'''<button class="sheet" type="button" data-role="{_h(role)}" data-image="assets/sheets/{_h(basename)}.jpg" data-title="Case {case_index:02d} · {_h(title)} · frames 0/20/40/60/80">
  <span>{_h(title)}</span><img loading="lazy" src="assets/sheets/{_h(basename)}.jpg" alt="Case {case_index:02d} {_h(title)} 五帧 contact sheet">
</button>'''
            )
        search = " ".join(
            (
                f"{case_index:02d}",
                str(case["iid"]),
                str(case["split"]),
                str(case["instruction"]),
            )
        ).lower()
        sections.append(
            f'''<article class="case" id="case-{case_index:02d}" data-split="{_h(case['split'])}" data-search="{_h(search)}">
  <header class="case-head">
    <div class="case-number">{case_index:02d}</div>
    <div><div class="case-title"><h2>Case {case_index:02d}</h2><span class="pill">{_h(case['split'])}</span><span class="pill current">current R64</span></div><code class="iid">IID {_h(case['iid'])}</code></div>
    <button class="sync-case" type="button">同步从头播放三列</button>
  </header>
  <p class="instruction"><span>Instruction</span>{_h(case['instruction'])}</p>
  <div class="case-meta"><span>seed {_h(case['seed'])}</span><span>instruction sha256 {_h(case['instruction_sha256'])}</span><span>Full644 membership IID overlap: 0</span></div>
  <div class="video-grid">{''.join(video_cells)}</div>
  <details class="sheets" open><summary>5 帧 contact sheets · frames 0 / 20 / 40 / 60 / 80</summary><div class="sheet-grid">{''.join(sheet_cells)}</div></details>
</article>'''
        )
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>当前 Full644 R64 · Heldout8 八例推理结果</title>
<style>
:root{{--bg:#080c11;--panel:#111821;--panel2:#172230;--line:#2a394a;--text:#eef4fa;--muted:#9dadbd;--blue:#70b6ff;--green:#63d69d;--amber:#ffc85c;--red:#ff7d72;--max:1580px}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;color:var(--text);background:radial-gradient(circle at 12% -10%,rgba(112,182,255,.14),transparent 35rem),radial-gradient(circle at 90% 0,rgba(99,214,157,.09),transparent 32rem),var(--bg);font:15px/1.58 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}
a{{color:#9acbff}} button,input{{font:inherit}} code{{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}} .wrap{{width:min(calc(100% - 32px),var(--max));margin:auto}}
.topbar{{position:sticky;top:0;z-index:30;border-bottom:1px solid rgba(255,255,255,.08);background:rgba(8,12,17,.9);backdrop-filter:blur(16px)}} .topbar .wrap{{min-height:62px;display:flex;align-items:center;justify-content:space-between;gap:18px}} .brand{{font-weight:800}} .brand::before{{content:"";display:inline-block;width:9px;height:9px;margin-right:10px;border-radius:50%;background:var(--green);box-shadow:0 0 0 6px rgba(99,214,157,.12)}} .topmeta{{color:var(--muted);font-size:12px;text-align:right}}
header.hero{{padding:66px 0 34px}} .eyebrow{{color:var(--green);font-size:12px;font-weight:850;letter-spacing:.15em;text-transform:uppercase}} h1{{margin:12px 0 18px;font-size:clamp(38px,6vw,78px);line-height:1;letter-spacing:-.05em}} .lede{{max-width:1020px;margin:0;color:#c2ccd6;font-size:clamp(17px,2vw,22px)}}
.scope{{display:grid;grid-template-columns:auto 1fr;gap:15px;margin-top:28px;padding:20px;border:1px solid rgba(255,200,92,.48);border-radius:16px;background:linear-gradient(135deg,rgba(255,200,92,.14),rgba(255,200,92,.04))}} .scope .icon{{width:36px;height:36px;display:grid;place-items:center;border-radius:50%;color:#1c1404;background:var(--amber);font-weight:900}} .scope strong{{color:#ffe09a;font-size:17px}} .scope p{{margin:4px 0 0;color:#ddd2ba}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}} .stat{{padding:17px;border:1px solid var(--line);border-radius:14px;background:rgba(17,24,33,.82)}} .stat b{{display:block;color:var(--green);font-size:27px}} .stat span{{color:var(--muted);font-size:12px}}
.filters{{position:sticky;top:62px;z-index:20;padding:12px 0;border-block:1px solid rgba(255,255,255,.07);background:rgba(8,12,17,.88);backdrop-filter:blur(14px)}} .filter-row{{display:flex;flex-wrap:wrap;align-items:center;gap:8px}} .filter,.column-filter{{padding:8px 12px;border:1px solid var(--line);border-radius:999px;color:var(--muted);background:var(--panel);cursor:pointer}} .filter.active,.column-filter.active{{color:#08120d;border-color:var(--green);background:var(--green);font-weight:750}} #search{{min-width:280px;flex:1;padding:9px 13px;border:1px solid var(--line);border-radius:10px;color:var(--text);background:var(--panel)}} #visible-count{{color:var(--muted);font-size:12px;margin-left:auto}}
.case-list{{display:grid;gap:28px;padding:30px 0}} .case{{scroll-margin-top:128px;overflow:hidden;border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,rgba(23,34,48,.94),rgba(13,19,27,.96));box-shadow:0 24px 72px rgba(0,0,0,.24)}} .case[hidden]{{display:none}} .case-head{{display:grid;grid-template-columns:58px 1fr auto;gap:16px;align-items:center;padding:21px 22px 15px}} .case-number{{width:52px;height:52px;display:grid;place-items:center;border:1px solid var(--line);border-radius:14px;color:var(--green);background:#0a1017;font-weight:850;font-size:18px}} .case-title{{display:flex;flex-wrap:wrap;align-items:center;gap:8px}} .case-title h2{{margin:0;font-size:26px}} .pill{{padding:3px 8px;border:1px solid var(--line);border-radius:999px;color:var(--muted);font-size:11px}} .pill.current{{color:#9de5bd;border-color:rgba(99,214,157,.42);background:rgba(99,214,157,.07)}} .iid{{display:block;margin-top:3px;color:var(--muted)}}
.sync-case,.play-one{{padding:7px 10px;border:1px solid #47647f;border-radius:8px;color:var(--text);background:#1a2a3b;cursor:pointer}} .sync-case:hover,.play-one:hover{{background:#243a51}} .instruction{{margin:0;padding:0 22px 17px;color:#d9e3ec;font-size:17px}} .instruction span{{display:block;color:var(--blue);font-size:11px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}} .case-meta{{display:flex;flex-wrap:wrap;gap:7px;padding:0 22px 18px}} .case-meta span{{padding:4px 8px;border:1px solid var(--line);border-radius:7px;color:var(--muted);background:#0c131b;font:11px ui-monospace,SFMono-Regular,monospace;word-break:break-all}}
.video-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;padding:0 12px 12px}} .video-card{{overflow:hidden;border:1px solid var(--line);border-radius:13px;background:var(--panel)}} .video-card.column-hidden{{display:none}} .video-head{{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:12px}} .video-head h3{{margin:0;font-size:17px}} .video-head p{{margin:2px 0 0;color:var(--muted);font-size:11px}} .play-one{{padding:5px 8px;font-size:11px}} video{{display:block;width:100%;aspect-ratio:576/416;background:#000;object-fit:contain}} .media-meta{{display:flex;flex-wrap:wrap;gap:6px;padding:9px 11px 4px}} .media-meta span{{padding:3px 6px;border-radius:6px;color:#b8c6d3;background:#0b1219;font-size:10px}} .video-card code{{display:block;padding:3px 11px;color:var(--muted)}} .asset-links{{padding:5px 11px 12px;color:var(--muted);font-size:11px}}
.sheets{{border-top:1px solid var(--line);background:#090e14}} .sheets summary{{padding:11px 15px;color:var(--muted);cursor:pointer}} .sheet-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;padding:0 12px 12px}} .sheet{{padding:0;overflow:hidden;border:1px solid var(--line);border-radius:10px;color:var(--text);background:#0c1219;cursor:zoom-in;text-align:left}} .sheet span{{display:block;padding:7px 9px;font-size:11px}} .sheet img{{display:block;width:100%;height:auto}} .sheet.column-hidden{{display:none}}
.empty{{display:none;margin:30px 0;padding:30px;border:1px dashed var(--line);border-radius:14px;color:var(--muted);text-align:center}} footer{{padding:18px 0 42px;color:var(--muted);font-size:12px}} footer .evidence{{display:flex;flex-wrap:wrap;gap:8px 16px;margin-bottom:8px}}
dialog{{width:min(96vw,1800px);max-width:none;padding:0;border:1px solid #47596b;border-radius:14px;color:var(--text);background:#070b10;box-shadow:0 35px 120px rgba(0,0,0,.8)}} dialog::backdrop{{background:rgba(0,0,0,.86);backdrop-filter:blur(5px)}} dialog img{{display:block;width:100%;height:auto}} .dialog-bar{{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:10px 12px;border-bottom:1px solid var(--line)}} .dialog-bar span{{color:var(--muted)}} .dialog-bar button{{padding:6px 9px;border:1px solid var(--line);border-radius:7px;color:var(--text);background:var(--panel);cursor:pointer}}
@media(max-width:980px){{.stats{{grid-template-columns:repeat(2,1fr)}}.video-grid,.sheet-grid{{grid-template-columns:1fr}}.case-head{{grid-template-columns:50px 1fr}}.sync-case{{grid-column:2;justify-self:start}}.video-card.column-hidden,.sheet.column-hidden{{display:none}}}}
@media(max-width:560px){{.wrap{{width:min(calc(100% - 20px),var(--max))}}.topmeta{{display:none}}header.hero{{padding-top:42px}}.stats{{grid-template-columns:1fr 1fr}}.scope{{grid-template-columns:1fr}}#search{{min-width:100%}}.case-head{{padding-inline:14px}}.instruction,.case-meta{{padding-inline:14px}}}}
</style>
</head>
<body>
<nav class="topbar"><div class="wrap"><div class="brand">Full644 / R64 · Heldout8</div><div class="topmeta">当前 checkpoint-644 · 8 例 · Source / Base / R64</div></div></nav>
<header class="hero wrap">
  <div class="eyebrow">Current checkpoint · offline visual review</div>
  <h1>当前 R64 checkpoint<br>八例推理结果</h1>
  <p class="lede">当前 Full644/R64 checkpoint-644 在固定 Goku Heldout8 上的 Source、同配置 Base 与 R64 对照；逐例展示完整视频和帧 0/20/40/60/80。</p>
  <div class="scope"><div class="icon">!</div><div><strong>这是 Heldout8 推理集，不是 Full644 的训练子集，也不是 formal training evaluation。</strong><p>冻结计划记录它与命名 Full644 training membership 的 IID overlap 为 0/8；这只证明 named membership 的 IID-disjoint，不证明 content-disjoint、未知预训练 exposure 或科学泛化。页面只呈现技术通过后的真实媒体，不自动给出视觉成功结论。</p></div></div>
  <div class="stats"><div class="stat"><b>8</b><span>Heldout8 cases</span></div><div class="stat"><b>24</b><span>真实视频；三列 × 八例</span></div><div class="stat"><b>81f · 25fps</b><span>每个媒体均重新 ffprobe</span></div><div class="stat"><b>R64 / 644</b><span>LoRA rank / checkpoint step</span></div></div>
</header>
<section class="filters"><div class="wrap filter-row" aria-label="结果筛选">
  <button class="filter active" type="button" data-split="all">全部 8 例</button><button class="filter" type="button" data-split="test">test · 5</button><button class="filter" type="button" data-split="validation">validation · 3</button>
  <button class="column-filter active" type="button" data-role="source">Source</button><button class="column-filter active" type="button" data-role="base">Base</button><button class="column-filter active" type="button" data-role="full644">R64</button>
  <input id="search" type="search" placeholder="筛选 IID / 指令 / case" aria-label="筛选 IID、指令或 case"><span id="visible-count">8 / 8</span>
</div></section>
<main class="wrap"><div class="case-list">{''.join(sections)}</div><div class="empty" id="empty">没有匹配的案例。</div></main>
<footer class="wrap"><div class="evidence"><a href="evidence/plan.json">plan</a><a href="evidence/eval-report.json">full16 eval report</a><a href="evidence/runner-attestation.json">runner attestation</a><a href="site-manifest.json">site manifest</a></div><div>Portable relative-path HTML · report {_h(report_sha256)} · attestation {_h(attestation_sha256)} · built {_h(build_time)}</div></footer>
<dialog id="lightbox"><div class="dialog-bar"><span id="dialog-title"></span><button type="button" id="dialog-close">关闭</button></div><img id="dialog-image" alt="放大的五帧 contact sheet"></dialog>
<script>
const cases=[...document.querySelectorAll('.case')];const splitButtons=[...document.querySelectorAll('.filter')];const columnButtons=[...document.querySelectorAll('.column-filter')];const search=document.querySelector('#search');let split='all';
function applyFilter(){{const query=search.value.trim().toLowerCase();let visible=0;for(const card of cases){{const show=(split==='all'||card.dataset.split===split)&&(!query||card.dataset.search.includes(query));card.hidden=!show;if(show)visible++;else card.querySelectorAll('video').forEach(video=>video.pause());}}document.querySelector('#visible-count').textContent=`${{visible}} / 8`;document.querySelector('#empty').style.display=visible?'none':'block';}}
splitButtons.forEach(button=>button.addEventListener('click',()=>{{split=button.dataset.split;splitButtons.forEach(item=>item.classList.toggle('active',item===button));applyFilter();}}));search.addEventListener('input',applyFilter);
columnButtons.forEach(button=>button.addEventListener('click',()=>{{button.classList.toggle('active');const role=button.dataset.role;const hidden=!button.classList.contains('active');document.querySelectorAll(`.video-card[data-role="${{role}}"],.sheet[data-role="${{role}}"]`).forEach(item=>item.classList.toggle('column-hidden',hidden));if(hidden)document.querySelectorAll(`.video-card[data-role="${{role}}"] video`).forEach(video=>video.pause());}}));
document.querySelectorAll('.play-one').forEach(button=>button.addEventListener('click',async()=>{{const video=button.closest('.video-card').querySelector('video');if(video.paused)await video.play().catch(()=>{{}});else video.pause();}}));
document.querySelectorAll('.sync-case').forEach(button=>button.addEventListener('click',async()=>{{const videos=[...button.closest('.case').querySelectorAll('.video-card:not(.column-hidden) video')];videos.forEach(video=>{{video.pause();video.currentTime=0;}});await Promise.allSettled(videos.map(video=>video.play()));}}));
const dialog=document.querySelector('#lightbox');document.querySelectorAll('.sheet').forEach(button=>button.addEventListener('click',()=>{{document.querySelector('#dialog-image').src=button.dataset.image;document.querySelector('#dialog-title').textContent=button.dataset.title;dialog.showModal();}}));document.querySelector('#dialog-close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{{if(event.target===dialog)dialog.close();}});document.addEventListener('keydown',event=>{{if(event.key==='Escape'&&dialog.open)dialog.close();}});
</script>
</body>
</html>
'''


def _bundle_paths(bundle: Path) -> tuple[Path, Path, Path, Path]:
    try:
        info = bundle.lstat()
    except OSError as error:
        raise SiteBuildError(f"cannot inspect bundle: {bundle}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SiteBuildError("bundle is not a plain directory")
    expected = {"plan.json", "eval-report.json", "runner-attestation.json", "media"}
    observed = {item.name for item in bundle.iterdir()}
    if observed != expected:
        raise SiteBuildError(
            "bundle root closure differs; "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )
    media = bundle / "media"
    media_info = media.lstat()
    if stat.S_ISLNK(media_info.st_mode) or not stat.S_ISDIR(media_info.st_mode):
        raise SiteBuildError("bundle media is not a plain directory")
    expected_media: set[str] = set()
    for case_index in range(8):
        expected_media.add(f"case{case_index:02d}-source.mp4")
        for arm in ("base", "full644"):
            basename = f"case{case_index:02d}-{arm}.mp4"
            expected_media.update((basename, basename + ".receipt.json"))
    observed_media = {item.name for item in media.iterdir()}
    if observed_media != expected_media:
        raise SiteBuildError(
            "bundle media closure differs; "
            f"missing={sorted(expected_media - observed_media)} "
            f"extra={sorted(observed_media - expected_media)}"
        )
    return (
        bundle / "plan.json",
        bundle / "eval-report.json",
        bundle / "runner-attestation.json",
        media,
    )


def build_site(
    *, bundle: Path, output: Path, ffmpeg: Path, ffprobe: Path
) -> dict[str, Any]:
    plan_path, report_path, attestation_path, media_root = _bundle_paths(bundle)
    plan, plan_sha256, _ = load_json(plan_path, label="plan")
    cases = validate_plan(plan)
    report, report_sha256, _ = load_json(report_path, label="evaluation report")
    results = validate_report(report, plan)
    attestation, attestation_sha256, _ = load_json(
        attestation_path, label="runner attestation"
    )
    validate_attestation(
        attestation,
        plan=plan,
        plan_sha256=plan_sha256,
        report=report,
        report_sha256=report_sha256,
    )

    validated_media: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_index = int(case["case_index"])
        case["videos"] = {}
        for arm in ("source", "base", "full644"):
            basename = f"case{case_index:02d}-{arm}.mp4"
            path = media_root / basename
            _, media_sha256, media_size = stable_file(path, label=basename)
            probe = probe_video(path, ffprobe)
            _, replay_sha256, replay_size = stable_file(path, label=f"{basename} replay")
            if replay_sha256 != media_sha256 or replay_size != media_size:
                raise SiteBuildError(f"{basename} changed during ffprobe")
            if arm == "source":
                if media_sha256 != case["source_sha256"]:
                    raise SiteBuildError(f"case {case_index:02d} source SHA differs")
            else:
                task_id = f"shared8-{case_index:02d}-{arm}"
                result = results[task_id]
                if (
                    media_sha256 != result["output_sha256"]
                    or media_size != result["output_size"]
                    or probe["frame_count"] != result["media_probe"]["frame_count"]
                    or probe["fps_num"] != result["media_probe"]["fps_num"]
                    or probe["fps_den"] != result["media_probe"]["fps_den"]
                    or probe["width"] != result["media_probe"]["width"]
                    or probe["height"] != result["media_probe"]["height"]
                ):
                    raise SiteBuildError(f"{task_id} local media/report binding differs")
                receipt_path = media_root / (basename + ".receipt.json")
                receipt, receipt_sha256, _ = load_json(
                    receipt_path, label=f"{task_id} native receipt"
                )
                validate_receipt(
                    receipt,
                    receipt_sha256=receipt_sha256,
                    result=result,
                    case=case,
                    arm=arm,
                )
                validated_media[basename + ".receipt.json"] = {
                    "source": receipt_path,
                    "sha256": receipt_sha256,
                }
            case["videos"][arm] = {
                "source": path,
                "basename": basename,
                "sha256": media_sha256,
                "size": media_size,
                "probe": probe,
            }
            validated_media[basename] = {"source": path, "sha256": media_sha256}

    output = output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise SiteBuildError(f"output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-stage-", dir=output.parent))
    try:
        media_out = stage / "assets" / "media"
        sheets_out = stage / "assets" / "sheets"
        evidence_out = stage / "evidence"
        for directory in (stage / "assets", media_out, sheets_out, evidence_out):
            directory.mkdir(mode=0o755)

        copied_files: list[dict[str, Any]] = []
        for basename, row in sorted(validated_media.items()):
            destination = media_out / basename
            copied = _copy_verified(
                row["source"],
                destination,
                expected_sha256=row["sha256"],
                label=f"published {basename}",
            )
            copied_files.append(
                {"path": f"assets/media/{basename}", **copied}
            )
        evidence_sources = (
            ("plan.json", plan_path, plan_sha256),
            ("eval-report.json", report_path, report_sha256),
            ("runner-attestation.json", attestation_path, attestation_sha256),
        )
        for basename, source, sha256 in evidence_sources:
            copied = _copy_verified(
                source,
                evidence_out / basename,
                expected_sha256=sha256,
                label=f"published evidence {basename}",
            )
            copied_files.append({"path": f"evidence/{basename}", **copied})

        for case in cases:
            for arm in ("source", "base", "full644"):
                basename = case["videos"][arm]["basename"]
                sheet_path = sheets_out / (basename + ".jpg")
                make_contact_sheet(media_out / basename, sheet_path, ffmpeg)
                _, sheet_sha256, sheet_size = stable_file(
                    sheet_path, label=f"published sheet {basename}"
                )
                os.chmod(sheet_path, 0o444)
                copied_files.append(
                    {
                        "path": f"assets/sheets/{basename}.jpg",
                        "sha256": sheet_sha256,
                        "size": sheet_size,
                    }
                )

        build_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        index_raw = render_html(
            cases,
            report_sha256=report_sha256,
            attestation_sha256=attestation_sha256,
            build_time=build_time,
        ).encode("utf-8")
        _write_new(stage / "index.html", index_raw)
        copied_files.append(
            {
                "path": "index.html",
                "sha256": hashlib.sha256(index_raw).hexdigest(),
                "size": len(index_raw),
            }
        )
        manifest: dict[str, Any] = {
            "schema_version": SITE_SCHEMA,
            "status": "COMPLETE_TECHNICAL_REVIEW_SITE",
            "built_at_utc": build_time,
            "case_count": 8,
            "video_count": 24,
            "generated_output_count": 16,
            "contact_sheet_count": 24,
            "contact_sheet_frames": list(CHECKPOINT_FRAMES),
            "media_contract": {
                "frame_count": EXPECTED_MEDIA_FRAME_COUNT,
                "fps_num": EXPECTED_MEDIA_FPS.numerator,
                "fps_den": EXPECTED_MEDIA_FPS.denominator,
                "codec": "h264",
            },
            "dataset_scope": {
                "name": "Goku legacy Heldout8 inference set",
                "full644_training_subset": False,
                "iid_overlap_with_named_full644_membership": 0,
                "content_disjoint_proven": False,
                "formal_training_evaluation": False,
                "scientific_generalization_claim_authorized": False,
            },
            "checkpoint": {
                "profile": FULL644_PROFILE,
                "global_step": 644,
                "lora_rank": 64,
                "adapter_model_sha256": FULL644_ADAPTER_SHA256,
                "checkpoint_manifest_sha256": FULL644_CHECKPOINT_MANIFEST_SHA256,
                "training_receipt_sha256": FULL644_TRAINING_RECEIPT_SHA256,
            },
            "authorities": {
                "plan_sha256": plan_sha256,
                "plan_digest": plan["plan_digest"],
                "report_sha256": report_sha256,
                "report_digest": report["report_digest"],
                "attestation_sha256": attestation_sha256,
                "attestation_digest": attestation["attestation_digest"],
            },
            "human_visual_judgment_added_by_builder": False,
            "files_excluding_this_manifest": sorted(
                copied_files, key=lambda row: row["path"]
            ),
        }
        manifest["manifest_digest"] = object_sha256(manifest)
        manifest_raw = canonical_json_bytes(manifest) + b"\n"
        _write_new(stage / "site-manifest.json", manifest_raw)

        os.replace(stage, output)
        stage = None
        return {
            "output": str(output),
            "index": str(output / "index.html"),
            "manifest": str(output / "site-manifest.json"),
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "case_count": 8,
            "video_count": 24,
            "contact_sheet_count": 24,
        }
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a portable current-R64 Heldout8 HTML page from a real, "
            "successful r5g full16 bundle."
        )
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help=(
            "fresh input directory containing plan.json, eval-report.json, "
            "runner-attestation.json, and exact media/ closure"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="fresh output directory; it is never overwritten",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_site(
            bundle=Path(args.bundle).resolve(strict=True),
            output=Path(args.output),
            ffmpeg=resolve_tool(args.ffmpeg, label="ffmpeg"),
            ffprobe=resolve_tool(args.ffprobe, label="ffprobe"),
        )
    except (OSError, SiteBuildError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
