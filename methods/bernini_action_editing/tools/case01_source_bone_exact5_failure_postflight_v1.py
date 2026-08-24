#!/usr/bin/env python3
"""Produce and revalidate a strict exact5 FAILED_NO_RETRY postflight manifest.

This tool never creates a success report or repairs runner evidence.  It reads
the five completed task publications and their persistent runner artifacts
after the preregistered exact-original parity gate failed, verifies their
current named-path bytes, and writes one fresh, claim-limited postmortem
manifest.  The manifest explicitly does not attest retained FDs, model-final
state, or a successful/formal evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

try:
    from methods.bernini_action_editing.tools import (
        build_case01_source_bone_exact5_r64_html_v1 as contract,
    )
    from methods.bernini_action_editing import (
        action_preservation_decoded_eval_model_authority_v2 as model_authority,
    )
except ModuleNotFoundError:  # Direct execution from this tools directory.
    import build_case01_source_bone_exact5_r64_html_v1 as contract
    method_root = Path(__file__).resolve().parents[1]
    if str(method_root) not in sys.path:
        sys.path.insert(0, str(method_root))
    import action_preservation_decoded_eval_model_authority_v2 as model_authority


POSTFLIGHT_SCHEMA = "case01-source-bone-exact5-r64-failure-postflight-v1"
POSTFLIGHT_STATUS = "POSTFLIGHT_VERIFIED_FAILED_NO_RETRY_PARITY_MISMATCH"
POSTFLIGHT_REL = Path("postflight/case01_source_bone_exact5_failure_postflight_v1.json")
REFERENCE_REL = Path("reference/historical-case01-full644-r64.mp4")
REFERENCE_RECEIPT_REL = Path(
    "reference/historical-case01-full644-r64.mp4.receipt.json"
)
REFERENCE_RECEIPT_SHA256 = (
    "2fe6f9017d7d7e43e082761c3c3d7099457a1c2b7ca96b6af6987b4aa55087b2"
)
REFERENCE_RECEIPT_DIGEST = (
    "64ed2c6b1ba5992eeb2fb39ca9bc8e1cce0692e7f25548cc004846d725456626"
)
FAILURE_ERROR_TYPE = "MatchedRunnerV2Error"
FAILURE_ERROR = (
    "exact_original deterministic parity failed against frozen case01 R64"
)
RUNNER_TASK_SCHEMA = "full644-exploratory-matched-runner-task-auh-r5"
CHAIN_SCHEMA = "full644-exploratory-matched-consumption-chain-v2"

ARTIFACT_SUFFIXES = {
    "model_capture": "-model-capture.json",
    "model_pre_use": "-model-pre-use.json",
    "consumption_input": "-consumption-input.json",
    "model_post_use": "-model-post-use.json",
    "eval_consumption_chain": "-eval-consumption-chain.json",
    "adapter_capture": "-adapter-capture.json",
    "adapter_pre_use": "-adapter-pre-use.json",
    "adapter_post_use": "-adapter-post-use.json",
    "adapter_final": "-adapter-final.json",
}
ARTIFACT_DIGEST_FIELDS = {
    "model_capture": "capture_digest",
    "model_pre_use": "use_digest",
    "consumption_input": "consumption_input_digest",
    "model_post_use": "use_digest",
    "adapter_capture": "capture_digest",
    "adapter_pre_use": "use_digest",
    "adapter_post_use": "use_digest",
    "adapter_final": "adapter_final_digest",
    "eval_consumption_chain": "consumption_digest",
}


class PostflightError(contract.SiteBuildError):
    """The persisted failure package differs from its frozen contract."""


def _exact_names(path: Path, expected: set[str], *, label: str) -> None:
    try:
        contract._exact_names(path, expected, label=label)
    except contract.SiteBuildError as error:
        raise PostflightError(str(error)) from error


def _mode_row(
    path: Path, *, permissions: int | set[int] | frozenset[int], label: str,
) -> dict[str, int]:
    try:
        info = path.lstat()
    except OSError as error:
        raise PostflightError(f"missing {label}: {path}") from error
    allowed = {permissions} if isinstance(permissions, int) else set(permissions)
    observed_permissions = stat.S_IMODE(info.st_mode)
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or observed_permissions not in allowed
        or info.st_nlink != 1
    ):
        raise PostflightError(
            f"{label} must be regular mode in "
            f"{[format(value, '04o') for value in sorted(allowed)]}/single-link: "
            f"{path}"
        )
    return {"mode": observed_permissions, "nlink": 1}


def _expected_media_names() -> set[str]:
    names: set[str] = set()
    for index, task_id in enumerate(contract.TASK_IDS):
        prefix = f".matched-v2-{index:02d}-{task_id}"
        names.update(prefix + suffix for suffix in ARTIFACT_SUFFIXES.values())
        names.add(prefix + ".log")
        names.add(prefix + "-runner-task.json")
        names.add(f"{task_id}.mp4")
        names.add(f"{task_id}.mp4.receipt.json")
    return names


def bundle_paths(
    bundle: Path, *, require_manifest: bool,
) -> dict[str, Path]:
    root_names = {"plan", "final", "sources", "outputs", "reference"}
    if require_manifest:
        root_names.add("postflight")
    try:
        contract._plain_directory(bundle, label="failure bundle root")
        for name in root_names:
            contract._plain_directory(
                bundle / name, label=f"failure bundle {name}",
            )
    except contract.SiteBuildError as error:
        raise PostflightError(str(error)) from error
    _exact_names(
        bundle / "plan", {contract.PLAN_REL.name}, label="failure bundle plan",
    )
    _exact_names(
        bundle / "final", {contract.ATTESTATION_REL.name},
        label="failure bundle final (success report must be absent)",
    )
    _exact_names(
        bundle / "sources",
        {f"{variant}.mp4" for variant in contract.VARIANT_ORDER},
        label="failure bundle sources",
    )
    _exact_names(bundle / "outputs", {"media"}, label="failure outputs")
    _exact_names(
        bundle / "outputs/media", _expected_media_names(),
        label="failure persistent task publications",
    )
    _exact_names(
        bundle / "reference", {REFERENCE_REL.name, REFERENCE_RECEIPT_REL.name},
        label="frozen historical reference",
    )
    if require_manifest:
        _exact_names(
            bundle / "postflight", {POSTFLIGHT_REL.name},
            label="failure postflight",
        )
    return {
        "plan": bundle / contract.PLAN_REL,
        "failure": bundle / contract.ATTESTATION_REL,
        "sources": bundle / "sources",
        "media": bundle / "outputs/media",
        "reference": bundle / REFERENCE_REL,
        "reference_receipt": bundle / REFERENCE_RECEIPT_REL,
        "manifest": bundle / POSTFLIGHT_REL,
        "auxiliary_root_entries": sorted(
            item.name for item in bundle.iterdir() if item.name not in root_names
        ),
    }


def validate_failure_attestation(
    value: Mapping[str, Any], *, plan_sha256: str,
) -> None:
    fields = {
        "schema_version", "status", "error_type", "error", "plan_path",
        "plan_sha256", "runner_path", "retry_allowed",
        "partial_outputs_are_not_results", "scientific_claim_authorized",
        "failure_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PostflightError("FAILED_NO_RETRY attestation schema differs")
    contract.require_digest(value, "failure_digest", label="failure attestation")
    plan_path = contract.require_absolute_path(
        value.get("plan_path"), label="failure plan path",
    )
    runner_path = contract.require_absolute_path(
        value.get("runner_path"), label="failure runner path",
    )
    if (
        value.get("schema_version") != contract.FAILURE_SCHEMA
        or value.get("status") != "FAILED_NO_RETRY"
        or value.get("error_type") != FAILURE_ERROR_TYPE
        or value.get("error") != FAILURE_ERROR
        or value.get("plan_sha256") != plan_sha256
        or plan_path.name != contract.PLAN_REL.name
        or runner_path.name != "case01_source_bone_exact5_runner_v1.py"
        or value.get("retry_allowed") is not False
        or value.get("partial_outputs_are_not_results") is not True
        or value.get("scientific_claim_authorized") is not False
    ):
        raise PostflightError("failure is not the exact preregistered parity gate")


def _load_0400_json(path: Path, *, label: str) -> tuple[dict[str, Any], str, int]:
    # A directly mounted runner package is 0400.  The local assembled mirror
    # intentionally copies these private leaves as owner-only 0600.  Both are
    # explicit, fail-closed staging profiles; hashes and embedded native
    # publication identities remain the content authority.
    _mode_row(path, permissions={0o400, 0o600}, label=label)
    return contract.load_json(path, label=label)


def _chain_expected(chain: Mapping[str, Any]) -> dict[str, Any]:
    digest_fields = (
        "consumption_input_digest", "model_capture_digest",
        "model_pre_use_digest", "model_post_use_digest",
        "adapter_capture_digest", "adapter_pre_use_digest",
        "adapter_post_use_digest", "adapter_final_digest",
        "native_inference_receipt_digest", "native_receipt_file_sha256",
        "native_output_sha256",
    )
    for field in digest_fields:
        contract.require_sha256(chain.get(field), label=f"chain {field}")
    row: dict[str, Any] = {
        "schema_version": CHAIN_SCHEMA,
        "task_id": chain.get("task_id"),
        "consumption_input_digest": chain.get("consumption_input_digest"),
        "model_capture_digest": chain.get("model_capture_digest"),
        "model_pre_use_digest": chain.get("model_pre_use_digest"),
        "model_post_use_digest": chain.get("model_post_use_digest"),
        "adapter_capture_digest": chain.get("adapter_capture_digest"),
        "adapter_pre_use_digest": chain.get("adapter_pre_use_digest"),
        "adapter_post_use_digest": chain.get("adapter_post_use_digest"),
        "adapter_final_digest": chain.get("adapter_final_digest"),
        "native_inference_receipt_digest": chain.get(
            "native_inference_receipt_digest"
        ),
        "native_receipt_file_sha256": chain.get("native_receipt_file_sha256"),
        "native_output_sha256": chain.get("native_output_sha256"),
        "event_order": [
            "native_output_and_receipt_published",
            "adapter_post_use_replayed_or_base_control",
            "adapter_final_closed_or_base_control",
            "model_post_use_replayed",
            "eval_consumption_chain_sealed",
        ],
        "native_publication_completed_before_parent_post_use_replay": True,
        "parent_post_use_closed_before_native_publication": False,
        "all_post_use_replays_completed_before_runner_result": True,
        "training_loss_read_or_used": False,
    }
    row["consumption_digest"] = contract.object_sha256(row)
    return row


def _validate_runner_task(
    row: Mapping[str, Any], *, index: int, case: Mapping[str, Any],
    output_sha256: str, output_size: int, receipt: Mapping[str, Any],
    receipt_sha256: str, receipt_size: int,
    artifact_values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    task_id = contract.TASK_IDS[index]
    if not isinstance(row, Mapping) or set(row) != contract._TASK_RESULT_FIELDS:
        raise PostflightError(f"{task_id} runner-task schema differs")
    result_digest = contract.require_digest(
        row, "task_result_digest", label=f"{task_id} runner-task",
    )
    expected_task_input = contract.object_sha256({
        "schema_version": "full644-exploratory-matched-task-input-v2",
        "plan_digest": case["plan_digest"], "task": case["task"],
    })
    for field in (
        "task_input_digest", "argv_digest", "environment_digest",
        "ffmpeg_exec_authority_digest", "publication_handoff_authority_digest",
        "publication_handoff_payload_digest", "model_capture_digest",
        "adapter_capture_digest", "consumption_input_digest",
        "consumption_digest", "native_receipt_digest",
        "native_receipt_file_sha256", "native_output_sha256",
    ):
        contract.require_sha256(row.get(field), label=f"{task_id} {field}")
    receipt_identity = contract.require_stat_identity(
        row.get("native_receipt_identity"),
        label=f"{task_id} retained receipt identity",
        permissions=0o400, nlink=1, size=receipt_size,
    )
    output_identity = contract.require_stat_identity(
        row.get("native_output_identity"),
        label=f"{task_id} retained output identity",
        permissions=0o444, nlink=1, size=output_size,
    )
    references = row.get("authority_artifacts")
    prefix = f".matched-v2-{index:02d}-{task_id}"
    if (
        row.get("schema_version") != RUNNER_TASK_SCHEMA
        or row.get("task_index") != index
        or row.get("task_id") != task_id
        or row.get("arm") != "full644"
        or row.get("plan_digest") != case["plan_digest"]
        or row.get("task_input_digest") != expected_task_input
        or row.get("return_code") != 0
        or row.get("attempt_count") != 1
        or row.get("retry_allowed") is not False
        or row.get("output_path") != case["task"]["output"]["video_path"]
        or row.get("receipt_path") != case["task"]["output"]["receipt_path"]
        or row.get("native_output_sha256") != output_sha256
        or row.get("native_output_size") != output_size
        or row.get("native_receipt_file_sha256") != receipt_sha256
        or row.get("native_receipt_digest") != receipt.get("receipt_digest")
        or row.get("log_basename") != prefix + ".log"
        or row.get("native_publication_completed_before_parent_post_use_replay")
        is not True
        or row.get("parent_post_use_closed_before_native_publication") is not False
        or row.get("post_use_replay_complete") is not True
        or not isinstance(references, Mapping)
        or set(references) != set(ARTIFACT_SUFFIXES)
        or any(
            not isinstance(reference, Mapping)
            or set(reference) != {"basename", "sha256"}
            or reference.get("basename") != prefix + ARTIFACT_SUFFIXES[role]
            or reference.get("sha256")
            != artifact_values[role]["_file_sha256"]
            for role, reference in references.items()
        )
    ):
        raise PostflightError(f"{task_id} persisted runner-task differs")

    output_receipt = receipt["output"]
    model_consumption = receipt["model_consumption"]
    if (
        output_receipt.get("publication_identity") != output_identity
        or model_consumption.get("model_capture_digest")
        != row["model_capture_digest"]
        or model_consumption.get("adapter_capture_digest")
        != row["adapter_capture_digest"]
        or receipt.get("consumption_input_digest")
        != row["consumption_input_digest"]
        or receipt.get("task_input_digest") != row["task_input_digest"]
    ):
        raise PostflightError(f"{task_id} receipt/runner-task cross-link differs")

    values = artifact_values
    clean_values = {
        role: {key: item for key, item in value.items() if key != "_file_sha256"}
        for role, value in values.items()
    }
    digests = {
        role: contract.require_digest(
            clean_values[role], ARTIFACT_DIGEST_FIELDS[role],
            label=f"{task_id} {role}",
        )
        for role in clean_values
    }
    consumption_input = dict(values["consumption_input"])
    consumption_input.pop("_file_sha256", None)
    try:
        validated_consumption = model_authority.validate_consumption_input(
            consumption_input
        )
    except Exception as error:
        raise PostflightError(
            f"{task_id} consumption-input validation failed: {error}"
        ) from error
    chain = dict(values["eval_consumption_chain"])
    chain.pop("_file_sha256", None)
    if chain != _chain_expected(chain):
        raise PostflightError(f"{task_id} eval consumption chain differs")
    if (
        validated_consumption.get("task_id") != task_id
        or validated_consumption.get("consumption_input_digest")
        != row["consumption_input_digest"]
        or validated_consumption.get("model", {}).get("capture_digest")
        != digests["model_capture"]
        or validated_consumption.get("model", {}).get("pre_use_digest")
        != digests["model_pre_use"]
        or validated_consumption.get("adapter", {}).get("capture_digest")
        != digests["adapter_capture"]
        or validated_consumption.get("adapter", {}).get("pre_use_digest")
        != digests["adapter_pre_use"]
        or chain["task_id"] != task_id
        or chain["consumption_input_digest"] != row["consumption_input_digest"]
        or chain["consumption_digest"] != row["consumption_digest"]
        or chain["model_capture_digest"] != digests["model_capture"]
        or chain["model_pre_use_digest"] != digests["model_pre_use"]
        or chain["model_post_use_digest"] != digests["model_post_use"]
        or chain["adapter_capture_digest"] != digests["adapter_capture"]
        or chain["adapter_pre_use_digest"] != digests["adapter_pre_use"]
        or chain["adapter_post_use_digest"] != digests["adapter_post_use"]
        or chain["adapter_final_digest"] != digests["adapter_final"]
        or chain["native_inference_receipt_digest"] != receipt["receipt_digest"]
        or chain["native_receipt_file_sha256"] != receipt_sha256
        or chain["native_output_sha256"] != output_sha256
    ):
        raise PostflightError(f"{task_id} persistent consumption chain differs")

    handoff_payload = {
        "schema_version":
        "full644-exploratory-matched-publication-handoff-payload-v1",
        "task_id": task_id, "output_path": row["output_path"],
        "output_identity": output_identity, "output_sha256": output_sha256,
        "output_size": output_size, "receipt_path": row["receipt_path"],
        "receipt_identity": receipt_identity, "receipt_sha256": receipt_sha256,
        "receipt_size": receipt_size, "receipt_digest": receipt["receipt_digest"],
    }
    if (
        row["publication_handoff_payload_digest"]
        != contract.object_sha256(handoff_payload)
    ):
        raise PostflightError(f"{task_id} sealed handoff payload digest differs")
    artifact_rows = [
        {
            "role": role, "basename": references[role]["basename"],
            "sha256": references[role]["sha256"],
        }
        for role in sorted(references)
    ]
    return {
        "task_result_digest": result_digest,
        "task_input_digest": expected_task_input,
        "artifact_rows": artifact_rows,
        "artifact_rows_digest": contract.object_sha256(artifact_rows),
        "consumption_digest": row["consumption_digest"],
        "model_capture_digest": row["model_capture_digest"],
        "adapter_capture_digest": row["adapter_capture_digest"],
        "consumption_input_digest": row["consumption_input_digest"],
    }


def decode_rgb24(path: Path, ffmpeg: Path) -> dict[str, Any]:
    process = subprocess.run(
        [
            str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-an", "-sn", "-dn", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
    )
    if process.returncode != 0 or not process.stdout:
        raise PostflightError(
            f"ffmpeg RGB24 decode failed for {path.name}: "
            + process.stderr.decode("utf-8", "replace")[:500]
        )
    return {
        "pixel_format": "rgb24", "byte_count": len(process.stdout),
        "sha256": hashlib.sha256(process.stdout).hexdigest(),
    }


def _tool_authority(path: Path, *, label: str) -> dict[str, Any]:
    _, sha256, size = contract.stable_file(path, label=label)
    return {"path": str(path), "sha256": sha256, "size": size}


def collect_bundle(
    *, bundle: Path, ffprobe: Path, ffmpeg: Path, require_manifest: bool,
) -> dict[str, Any]:
    paths = bundle_paths(bundle, require_manifest=require_manifest)
    plan_mode = _mode_row(
        paths["plan"], permissions={0o444, 0o644}, label="exact5 plan",
    )
    plan, plan_sha256, plan_size = contract.load_json(paths["plan"], label="exact5 plan")
    cases = contract.validate_plan(plan)
    failure_mode = _mode_row(
        paths["failure"], permissions={0o444, 0o644},
        label="failure attestation",
    )
    failure, failure_sha256, failure_size = contract.load_json(
        paths["failure"], label="failure attestation",
    )
    validate_failure_attestation(failure, plan_sha256=plan_sha256)

    reference_mode = _mode_row(
        paths["reference"], permissions=0o444, label="historical reference",
    )
    _, reference_sha256, reference_size = contract.stable_file(
        paths["reference"], label="historical reference",
    )
    reference_receipt_mode = _mode_row(
        paths["reference_receipt"], permissions=0o444,
        label="historical reference receipt",
    )
    reference_receipt, reference_receipt_sha256, reference_receipt_size = (
        contract.load_json(
            paths["reference_receipt"], label="historical reference receipt",
        )
    )
    contract.require_digest(
        reference_receipt, "receipt_digest", label="historical reference receipt",
    )
    reference_receipt_output = reference_receipt.get("output")
    reference_probe = contract.probe_video(paths["reference"], ffprobe)
    if (
        reference_sha256 != contract.REFERENCE_OUTPUT_SHA256
        or reference_receipt_sha256 != REFERENCE_RECEIPT_SHA256
        or reference_receipt.get("receipt_digest") != REFERENCE_RECEIPT_DIGEST
        or not isinstance(reference_receipt_output, Mapping)
        or reference_receipt_output.get("sha256") != reference_sha256
        or reference_receipt_output.get("size") != reference_size
        or reference_receipt_output.get("frame_count") != 81
        or reference_receipt_output.get("fps") != 25.0
    ):
        raise PostflightError("historical reference bytes differ from frozen R64")

    collected_cases: list[dict[str, Any]] = []
    coordinates: list[dict[str, bytes | str]] = []
    for index, case in enumerate(cases):
        variant = str(case["id"])
        task_id = str(case["task_id"])
        prefix = f".matched-v2-{index:02d}-{task_id}"
        source_path = paths["sources"] / f"{variant}.mp4"
        output_path = paths["media"] / f"{task_id}.mp4"
        receipt_path = paths["media"] / f"{task_id}.mp4.receipt.json"
        runner_task_path = paths["media"] / (prefix + "-runner-task.json")
        log_path = paths["media"] / (prefix + ".log")

        source_mode = _mode_row(
            source_path, permissions={0o444, 0o644},
            label=f"{variant} source",
        )
        _, source_sha256, source_size = contract.stable_file(
            source_path, label=f"{variant} source",
        )
        source_probe = contract.probe_video(source_path, ffprobe)
        if (
            source_sha256 != case["source_sha256"]
            or source_size != case["source_size"]
            or source_probe != {"codec": "h264", **contract.EXPECTED_SOURCE_VIDEO}
        ):
            raise PostflightError(f"{variant} source authority differs")

        output_mode = _mode_row(
            output_path, permissions={0o444, 0o644},
            label=f"{task_id} output",
        )
        _, output_sha256, output_size = contract.stable_file(
            output_path, label=f"{task_id} output",
        )
        output_probe = contract.probe_video(output_path, ffprobe)
        receipt_mode = _mode_row(
            receipt_path, permissions={0o400, 0o600},
            label=f"{task_id} receipt",
        )
        receipt, receipt_sha256, receipt_size = contract.load_json(
            receipt_path, label=f"{task_id} receipt",
        )
        result = {
            "receipt_file_sha256": receipt_sha256,
            "receipt_digest": receipt.get("receipt_digest"),
            "output_sha256": output_sha256, "output_size": output_size,
            "media_probe": {
                "width": output_probe["width"], "height": output_probe["height"],
            },
        }
        coordinates.append(contract.validate_receipt(
            receipt, receipt_sha256=receipt_sha256, receipt_size=receipt_size,
            result=result, case=case, checkpoint=plan["checkpoint_manifest"],
        ))

        artifact_values: dict[str, dict[str, Any]] = {}
        artifact_rows: list[dict[str, Any]] = []
        for role, suffix in ARTIFACT_SUFFIXES.items():
            artifact_path = paths["media"] / (prefix + suffix)
            artifact_mode = _mode_row(
                artifact_path, permissions={0o400, 0o600},
                label=f"{task_id} {role}",
            )
            value, file_sha256, file_size = contract.load_json(
                artifact_path, label=f"{task_id} {role}",
            )
            value["_file_sha256"] = file_sha256
            artifact_values[role] = value
            artifact_rows.append({
                "role": role, "path": f"outputs/media/{artifact_path.name}",
                "sha256": file_sha256, "size": file_size,
                "current_mode": artifact_mode["mode"],
                "runner_publication_mode": 0o400,
                "embedded_digest": value[ARTIFACT_DIGEST_FIELDS[role]],
            })
        runner_task_mode = _mode_row(
            runner_task_path, permissions={0o400, 0o600},
            label=f"{task_id} runner-task",
        )
        runner_task, runner_task_sha256, runner_task_size = contract.load_json(
            runner_task_path, label=f"{task_id} runner-task",
        )
        log_mode = _mode_row(
            log_path, permissions={0o400, 0o600}, label=f"{task_id} log",
        )
        _, log_sha256, log_size = contract.stable_file(
            log_path, label=f"{task_id} log",
        )
        replay = _validate_runner_task(
            runner_task, index=index, case=case,
            output_sha256=output_sha256, output_size=output_size,
            receipt=receipt, receipt_sha256=receipt_sha256,
            receipt_size=receipt_size, artifact_values=artifact_values,
        )
        collected_cases.append({
            **case, "source_path": source_path, "output_path": output_path,
            "receipt_path": receipt_path, "runner_task_path": runner_task_path,
            "source_sha256": source_sha256, "source_size": source_size,
            "source_mode": source_mode,
            "source_probe": source_probe, "output_sha256": output_sha256,
            "output_size": output_size, "output_mode": output_mode,
            "output_probe": output_probe,
            "receipt_sha256": receipt_sha256, "receipt_size": receipt_size,
            "receipt_mode": receipt_mode,
            "receipt_digest": receipt["receipt_digest"],
            "runner_task_sha256": runner_task_sha256,
            "runner_task_size": runner_task_size,
            "runner_task_mode": runner_task_mode, "task_replay": replay,
            "artifact_rows": sorted(artifact_rows, key=lambda row: row["role"]),
            "log_path": log_path, "log_sha256": log_sha256,
            "log_size": log_size, "log_mode": log_mode, "receipt": receipt,
        })
    if (
        len({row["sampling"] for row in coordinates}) != 1
        or len({row["prompt"] for row in coordinates}) != 1
        or len({row["model_capture"] for row in coordinates}) != 1
    ):
        raise PostflightError("five partial tasks do not share sampler/prompt/model")
    exact_output_sha = collected_cases[0]["output_sha256"]
    if exact_output_sha == contract.REFERENCE_OUTPUT_SHA256:
        raise PostflightError("FAILED parity package unexpectedly matches reference bytes")
    exact_decode = decode_rgb24(collected_cases[0]["output_path"], ffmpeg)
    reference_decode = decode_rgb24(paths["reference"], ffmpeg)
    if exact_decode["sha256"] == reference_decode["sha256"]:
        raise PostflightError("decoded exact-original unexpectedly matches reference")
    return {
        "paths": paths, "plan": plan, "plan_sha256": plan_sha256,
        "plan_size": plan_size, "failure": failure,
        "plan_mode": plan_mode, "failure_mode": failure_mode,
        "failure_sha256": failure_sha256, "failure_size": failure_size,
        "cases": collected_cases, "reference_path": paths["reference"],
        "reference_sha256": reference_sha256, "reference_size": reference_size,
        "reference_mode": reference_mode,
        "reference_receipt": reference_receipt,
        "reference_receipt_sha256": reference_receipt_sha256,
        "reference_receipt_size": reference_receipt_size,
        "reference_receipt_mode": reference_receipt_mode,
        "reference_probe": reference_probe, "exact_decode": exact_decode,
        "reference_decode": reference_decode,
        "ffprobe_authority": _tool_authority(ffprobe, label="local ffprobe"),
        "ffmpeg_authority": _tool_authority(ffmpeg, label="local ffmpeg"),
        "auxiliary_root_entries": paths["auxiliary_root_entries"],
    }


def build_manifest(observed: Mapping[str, Any], *, observed_at_utc: str) -> dict[str, Any]:
    task_rows: list[dict[str, Any]] = []
    for index, case in enumerate(observed["cases"]):
        replay = case["task_replay"]
        receipt = case["receipt"]
        task_rows.append({
            "task_index": index, "task_id": case["task_id"],
            "variant": case["id"], "task_input_digest": replay["task_input_digest"],
            "source": {
                "path": f"sources/{case['id']}.mp4",
                "authority_path": case["task"]["source_video"],
                "sha256": case["source_sha256"], "size": case["source_size"],
                "current_mode": case["source_mode"]["mode"],
                "native_source_authority_mode": receipt["input"][
                    "source_video_physical_authority"
                ]["mode"],
                "nlink": 1, "media_probe": case["source_probe"],
            },
            "runner_task": {
                "path": f"outputs/media/{case['runner_task_path'].name}",
                "sha256": case["runner_task_sha256"],
                "size": case["runner_task_size"],
                "current_mode": case["runner_task_mode"]["mode"],
                "runner_publication_mode": 0o400, "nlink": 1,
                "task_result_digest": replay["task_result_digest"],
            },
            "output": {
                "path": f"outputs/media/{case['task_id']}.mp4",
                "authority_path": case["task"]["output"]["video_path"],
                "sha256": case["output_sha256"], "size": case["output_size"],
                "current_mode": case["output_mode"]["mode"],
                "native_publication_mode": 0o444, "nlink": 1,
            },
            "receipt": {
                "path": f"outputs/media/{case['task_id']}.mp4.receipt.json",
                "authority_path": case["task"]["output"]["receipt_path"],
                "sha256": case["receipt_sha256"], "size": case["receipt_size"],
                "current_mode": case["receipt_mode"]["mode"],
                "native_publication_mode": 0o400, "nlink": 1,
                "receipt_digest": case["receipt_digest"],
                "model_capture_digest": replay["model_capture_digest"],
                "adapter_capture_digest": replay["adapter_capture_digest"],
                "consumption_input_digest": replay["consumption_input_digest"],
                "sampling_digest": contract.object_sha256(receipt["sampling"]),
                "prompt_contract_digest": contract.object_sha256(
                    receipt["prompt_contract"]
                ),
            },
            "authority_artifacts": case["artifact_rows"],
            "artifact_rows_digest": replay["artifact_rows_digest"],
            "persistent_log": {
                "path": f"outputs/media/{case['log_path'].name}",
                "sha256": case["log_sha256"], "size": case["log_size"],
                "current_mode": case["log_mode"]["mode"],
                "runner_publication_mode": 0o400, "nlink": 1,
            },
            "consumption_chain": {
                "consumption_digest": replay["consumption_digest"],
                "persistent_chain_replayed": True,
            },
            "media_probe": case["output_probe"],
            "current_state_verified": {
                "named_path_bytes_rehashed": True,
                "persistent_json_digests_replayed": True,
                "receipt_output_runner_task_cross_linked": True,
                "retained_fd_state_replayed": False,
            },
        })
    manifest: dict[str, Any] = {
        "schema_version": POSTFLIGHT_SCHEMA, "status": POSTFLIGHT_STATUS,
        "observed_at_utc": observed_at_utc, "campaign_mode": contract.CAMPAIGN,
        "plan": {
            "path": str(contract.PLAN_REL), "sha256": observed["plan_sha256"],
            "size": observed["plan_size"],
            "current_mode": observed["plan_mode"]["mode"],
            "runner_package_mode": 0o444,
            "plan_digest": observed["plan"]["plan_digest"],
        },
        "code_authority": {
            "expected_runner_sha256": contract.EXACT5_RUNNER_SHA256,
            "expected_eval_sha256": contract.EXACT5_EVAL_SHA256,
            "runner_or_eval_bytes_in_postflight_bundle": False,
            "control_flow_interpretation_uses_frozen_expected_pins": True,
        },
        "package_authority": {
            "asset_authority_digest": observed["plan"]["asset_authority"][
                "authority_digest"
            ],
            "current_named_paths_rehashed_after_failure": True,
            "source_snapshot_authority_replayed_from_plan": True,
            "critical_subdirectory_closure_exact": True,
            "auxiliary_root_entries_not_used_as_authority": observed[
                "auxiliary_root_entries"
            ],
            "assembled_mirror_permissions_are_recorded_separately": True,
        },
        "gpu_attempt": {
            "included": False, "terminal_scheduler_evidence": False,
            "pre_srun_claim_not_treated_as_terminal": True,
        },
        "failure_attestation": {
            "path": str(contract.ATTESTATION_REL),
            "sha256": observed["failure_sha256"], "size": observed["failure_size"],
            "current_mode": observed["failure_mode"]["mode"],
            "runner_publication_mode": 0o444, "nlink": 1,
            "failure_digest": observed["failure"]["failure_digest"],
            "error_type": FAILURE_ERROR_TYPE, "error": FAILURE_ERROR,
        },
        "terminal_evidence": {
            "named_failure_attestation_observed": True,
            "scheduler_terminal_record_included": False,
            "runner_success": False, "retry_allowed": False,
        },
        "reference_parity": {
            "variant": "exact_original", "policy": "HARD_FAIL",
            "status": "FAIL",
            "reference_path": str(REFERENCE_REL),
            "reference_output_sha256": contract.REFERENCE_OUTPUT_SHA256,
            "reference_output_size": observed["reference_size"],
            "reference_media_probe": observed["reference_probe"],
            "reference_current_mode": observed["reference_mode"]["mode"],
            "reference_receipt": {
                "path": str(REFERENCE_RECEIPT_REL),
                "sha256": observed["reference_receipt_sha256"],
                "size": observed["reference_receipt_size"],
                "current_mode": observed["reference_receipt_mode"]["mode"],
                "receipt_digest": observed["reference_receipt"][
                    "receipt_digest"
                ],
            },
            "observed_output_sha256": observed["cases"][0]["output_sha256"],
            "encoded_bytes_equal": False,
            "decoded_rgb24_equal": False,
            "observed_decoded_rgb24": observed["exact_decode"],
            "reference_decoded_rgb24": observed["reference_decode"],
            "historical_reference_is_not_a_current_task_arm": True,
        },
        "task_count": 5, "task_ids": list(contract.TASK_IDS),
        "task_rows": task_rows,
        "task_rows_digest": contract.object_sha256(task_rows),
        "artifact_inventory": {
            "source_video_count": 5, "named_output_count": 5,
            "native_receipt_count": 5, "persistent_authority_json_count": 45,
            "persistent_runner_task_json_count": 5,
            "persistent_log_count": 5, "persistent_internal_count": 55,
            "historical_reference_video_count": 1,
            "success_report_present": False,
        },
        "absent_success_closure": {
            "success_report_absent": True,
            "success_attestation_absent": True,
            "model_final_not_persisted": True,
            "physical_bindings_not_persisted": True,
            "artifact_replays_not_persisted": True,
            "verified_report_rows_not_persisted": True,
        },
        "evidence_scope": {
            "pinned_runner_control_flow_inference": True,
            "after_the_fact_current_path_observation": True,
            "observer_precommitted": False,
            "retained_fd_state_replay": False,
            "publication_handoff_fd_replay": False,
            "model_final_replay": False,
        },
        "probe_authority": {
            "ffprobe": observed["ffprobe_authority"],
            "ffmpeg_rgb24_decoder": observed["ffmpeg_authority"],
        },
        "claim_limits": {
            "failure_postmortem_only": True,
            "partial_outputs_are_not_results": True,
            "postmortem_visualization_only": True,
            "manual_blind_review_required": True,
            "manual_review_json_included": False,
            "formal_training_evaluation": False,
            "scientific_claim_authorized": False,
            "formal_claim_authorized": False,
        },
    }
    manifest["manifest_digest"] = contract.object_sha256(manifest)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any], observed: Mapping[str, Any],
) -> None:
    if not isinstance(manifest, Mapping):
        raise PostflightError("postflight manifest root is absent")
    contract.require_digest(manifest, "manifest_digest", label="postflight manifest")
    observed_at = manifest.get("observed_at_utc")
    if not isinstance(observed_at, str):
        raise PostflightError("postflight observation time is absent")
    try:
        parsed = datetime.fromisoformat(observed_at)
    except ValueError as error:
        raise PostflightError("postflight observation time differs") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PostflightError("postflight observation time lacks timezone")
    expected = build_manifest(observed, observed_at_utc=observed_at)
    if dict(manifest) != expected:
        raise PostflightError("postflight manifest/current bundle cross-link differs")


def load_verified_bundle(
    bundle: Path, ffprobe: Path, ffmpeg: Path,
) -> dict[str, Any]:
    observed = collect_bundle(
        bundle=bundle, ffprobe=ffprobe, ffmpeg=ffmpeg, require_manifest=True,
    )
    manifest, manifest_sha256, manifest_size = contract.load_json(
        observed["paths"]["manifest"], label="failure postflight manifest",
    )
    _mode_row(
        observed["paths"]["manifest"], permissions=0o444,
        label="failure postflight manifest",
    )
    validate_manifest(manifest, observed)
    return {
        **observed, "manifest": manifest, "manifest_sha256": manifest_sha256,
        "manifest_size": manifest_size,
    }


def produce_manifest(
    *, bundle: Path, ffprobe: Path, ffmpeg: Path,
) -> dict[str, Any]:
    observed = collect_bundle(
        bundle=bundle, ffprobe=ffprobe, ffmpeg=ffmpeg, require_manifest=False,
    )
    postflight_dir = bundle / "postflight"
    if postflight_dir.exists() or postflight_dir.is_symlink():
        raise PostflightError(f"postflight destination must be fresh: {postflight_dir}")
    manifest = build_manifest(
        observed,
        observed_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    raw = contract.canonical_json_bytes(manifest) + b"\n"
    try:
        postflight_dir.mkdir(mode=0o755)
        contract._write_new(postflight_dir / POSTFLIGHT_REL.name, raw, mode=0o444)
    except BaseException:
        try:
            if postflight_dir.is_dir() and not any(postflight_dir.iterdir()):
                postflight_dir.rmdir()
        except OSError:
            pass
        raise
    verified = load_verified_bundle(bundle, ffprobe, ffmpeg)
    return {
        "manifest": str(postflight_dir / POSTFLIGHT_REL.name),
        "manifest_sha256": verified["manifest_sha256"],
        "manifest_digest": verified["manifest"]["manifest_digest"],
        "status": POSTFLIGHT_STATUS, "task_count": 5,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce one strict, claim-limited exact5 failure postflight manifest."
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = produce_manifest(
            bundle=Path(args.bundle).expanduser().absolute(),
            ffprobe=contract.resolve_tool(args.ffprobe, label="ffprobe"),
            ffmpeg=contract.resolve_tool(args.ffmpeg, label="ffmpeg"),
        )
    except (OSError, PostflightError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
