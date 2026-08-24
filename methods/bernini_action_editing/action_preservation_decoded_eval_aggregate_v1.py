#!/usr/bin/env python3
"""Verify exact-264 decoded evidence and materialize an opaque blind-review packet.

This command is local and create-only.  It accepts no loss input and performs
no model ranking, scheduler action, retry, upload, or scientific promotion.
All four production shard summaries and every native inference receipt must
close before a packet is published.  Missing calibration is recorded as a
machine ABSTAIN while the blind packet is still produced.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import action_preservation_decoded_eval_bridge_v1 as bridge
import action_preservation_decoded_eval_decoder_adapter_v1 as decoder_adapter
import action_preservation_decoded_eval_executor_v1 as executor
import action_preservation_decoded_eval_plan_v1 as plan


AGGREGATE_SCHEMA = "bernini-action-preservation-decoded-eval-aggregate-v2"
PRIVATE_PACKET_SCHEMA = "bernini-action-preservation-blind-private-map-v1"
PUBLIC_PACKET_SCHEMA = "blind-full-video-review-packet-v1"

AGGREGATE_FILENAME = "evaluation_complete.json"
PRIVATE_FILENAME = "private_blind_mapping.json"
PUBLIC_FILENAME = "blind_review_packet.json"
MEDIA_DIRECTORY = "media"


class DecodedEvaluationAggregateError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return plan.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _plain(path: Path, *, directory: bool, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise DecodedEvaluationAggregateError(f"{label} does not exist") from error
    wanted = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not wanted or stat.S_ISLNK(info.st_mode):
        raise DecodedEvaluationAggregateError(f"{label} is not a plain artifact")
    return path


def _bytes(path: Path, *, label: str) -> bytes:
    try:
        raw, _ = bridge._stable_file(path, label=label)
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    return raw


def _json(path: Path, *, label: str) -> dict[str, Any]:
    raw = _bytes(path, label=label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecodedEvaluationAggregateError(f"cannot decode {label}: {error}") from error
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value) + b"\n":
        raise DecodedEvaluationAggregateError(f"{label} is not canonical JSON")
    return dict(value)


def _validate_summary(
    value: Any, *, bundle: Mapping[str, Any], shard: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version", "evaluation_id", "evaluation_manifest_digest",
        "publication_digest", "shard_digest", "holder",
        "holder_execution_authority", "executor_verified_release_capture",
        "planned_task_count",
        "attempted_task_count", "success_count", "failure_count", "results",
        "all_tasks_attempted_exactly_once", "automatic_retry_count",
        "retry_allowed", "failure_artifacts_retained", "execution_backend",
        "tool_files_verified", "training_loss_read_or_used", "network_used",
        "remote_launch_performed", "scientific_promotion_authorized",
        "summary_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DecodedEvaluationAggregateError("shard summary field closure differs")
    row = dict(value)
    executor_capture = executor._capture_evidence(
        row["executor_verified_release_capture"],
        label="shard executor verified release capture",
        allow_none=not bool(row.get("tool_files_verified")),
    )
    holder_execution = row["holder_execution_authority"]
    holder_execution_fields = {
        "expected_job_id", "expected_node", "observed_slurm_job_id",
        "observed_hostname", "exact_holder_match", "holder_execution_digest",
    }
    if not isinstance(holder_execution, Mapping) or set(holder_execution) != holder_execution_fields:
        raise DecodedEvaluationAggregateError(
            "shard holder execution authority closure differs"
        )
    unsigned_holder = dict(holder_execution)
    claimed_holder = unsigned_holder.pop("holder_execution_digest")
    if (
        holder_execution["expected_job_id"] != shard["holder"]["job_id"]
        or holder_execution["expected_node"] != shard["holder"]["node"]
        or holder_execution["observed_slurm_job_id"] != shard["holder"]["job_id"]
        or holder_execution["observed_hostname"] != shard["holder"]["node"]
        or holder_execution["exact_holder_match"] is not True
        or claimed_holder != object_sha256(unsigned_holder)
    ):
        raise DecodedEvaluationAggregateError(
            "shard holder execution authority differs"
        )
    expected_ids = [executor._task_id(task) for task in shard["tasks"]]
    expected_relpaths = [task["record"]["output_relpath"] for task in shard["tasks"]]
    results = row["results"]
    if (
        row["schema_version"] != executor.SHARD_SUMMARY_SCHEMA
        or row["evaluation_id"] != bundle["manifest"]["evaluation_id"]
        or row["evaluation_manifest_digest"] != bundle["manifest"]["manifest_digest"]
        or row["publication_digest"]
        != bundle["publication_receipt"]["publication_digest"]
        or row["shard_digest"] != shard["shard_digest"]
        or row["holder"] != shard["holder"]
        or row["planned_task_count"] != 66
        or row["attempted_task_count"] != 66
        or row["success_count"] != 66
        or row["failure_count"] != 0
        or not isinstance(results, list)
        or [item.get("task_id") for item in results] != expected_ids
        or [item.get("output_relpath") for item in results] != expected_relpaths
        or any(item.get("status") != "success" for item in results)
        or row["all_tasks_attempted_exactly_once"] is not True
        or row["automatic_retry_count"] != 0
        or row["retry_allowed"] is not False
        or row["failure_artifacts_retained"] is not True
        or row["execution_backend"] != "pinned_local_subprocess"
        or row["tool_files_verified"] is not True
        or executor_capture is None
        or executor_capture["target"]
        != "action_preservation_decoded_eval_executor_v1.py"
        or row["training_loss_read_or_used"] is not False
        or row["network_used"] is not False
        or row["remote_launch_performed"] is not False
        or row["scientific_promotion_authorized"] is not False
    ):
        raise DecodedEvaluationAggregateError(
            "shard is not an exact successful production execution"
        )
    unsigned = dict(row)
    claimed = unsigned.pop("summary_digest", None)
    if not isinstance(claimed, str) or object_sha256(unsigned) != claimed:
        raise DecodedEvaluationAggregateError("shard summary digest differs")
    return row


def collect_verified_outputs(
    *, evaluation_root: str | Path, physical_bindings_path: str | Path,
    physical_bindings_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    root = Path(evaluation_root)
    try:
        bundle = executor.load_published_bundle(root)
        bindings = bridge.load_physical_bindings(
            physical_bindings_path,
            expected_sha256=physical_bindings_sha256,
            verify_files=True,
        )
    except (executor.DecodedEvaluationExecutorError, bridge.DecodedEvaluationBridgeError) as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    try:
        for relative_path, module_path in (
            ("action_preservation_decoded_eval_aggregate_v1.py", __file__),
            (
                "action_preservation_decoded_eval_executor_v1.py",
                executor.__file__,
            ),
            (
                "action_preservation_decoded_eval_decoder_adapter_v1.py",
                decoder_adapter.__file__,
            ),
            (
                "action_preservation_decoded_eval_bridge_v1.py",
                bridge.__file__,
            ),
            (
                "action_preservation_decoded_eval_plan_v1.py",
                plan.__file__,
            ),
            ("action_preservation_gate_v1.py", plan.gate.__file__),
        ):
            bridge.require_running_eval_release_member(
                bindings["eval_release"],
                relative_path=relative_path,
                running_path=module_path,
            )
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    if (
        bindings["evaluation_root"] != str(root)
        or bindings["evaluation_id"] != bundle["manifest"]["evaluation_id"]
        or bindings["input_digest"] != bundle["input_spec"]["input_digest"]
        or bindings["manifest_digest"] != bundle["manifest"]["manifest_digest"]
    ):
        raise DecodedEvaluationAggregateError("aggregate physical binding differs")
    pin_file_map = {
        "source_manifest_sha256": "source_manifest",
        "adapter_release_manifest_sha256": "adapter_release_manifest",
        "model_release_manifest_sha256": "model_release_manifest",
        "inference_release_manifest_sha256": "inference_release_manifest",
        "inference_config_sha256": "inference_config",
        "source_preprocessing_sha256": "source_preprocessing",
    }
    if any(
        bindings["pin_files"][file_key]["sha256"]
        != bundle["input_spec"]["pins"][pin_key]
        for pin_key, file_key in pin_file_map.items()
    ) or (
        bindings["runtime"]["infer_lora"]["sha256"]
        != bundle["input_spec"]["pins"]["inference_source_sha256"]
    ) or bindings["calibration_digest"] != bundle["input_spec"]["pins"][
        "calibration_digest"
    ]:
        raise DecodedEvaluationAggregateError(
            "aggregate physical pin files differ from evaluation input"
        )
    physical_identity = {
        "path": str(Path(physical_bindings_path).resolve(strict=True)),
        "sha256": physical_bindings_sha256,
    }
    outputs: list[dict[str, Any]] = []
    summaries: list[dict[str, str]] = []
    expected_decoder_sha = plan.file_sha256(decoder_adapter.__file__)
    release_members = {
        item["relative_path"]: item for item in bindings["eval_release"]["members"]
    }
    expected_executor_sha = release_members[
        "action_preservation_decoded_eval_executor_v1.py"
    ]["sha256"]
    reprobe_video = executor.ffprobe_video_prober(
        bindings["runtime"]["ffprobe"]["path"]
    )
    for holder in plan.HOLDER_ROWS:
        job_id = holder["job_id"]
        shard = bundle["shards"][job_id]
        shard_root = root / executor.EXECUTION_DIRECTORY / job_id
        summary_path = shard_root / executor.SUMMARY_FILENAME
        summary = _validate_summary(
            _json(summary_path, label=f"holder {job_id} summary"),
            bundle=bundle,
            shard=shard,
        )
        expected_executor_arguments = [
            "--evaluation-root", str(root),
            "--holder-job-id", job_id,
            "--decoder-adapter", bindings["runtime"]["decoder_adapter"]["path"],
            "--decoder-adapter-sha256",
            bindings["runtime"]["decoder_adapter"]["sha256"],
            "--ffprobe", bindings["runtime"]["ffprobe"]["path"],
            "--ffprobe-sha256", bindings["runtime"]["ffprobe"]["sha256"],
            "--physical-bindings", physical_identity["path"],
            "--physical-bindings-sha256", physical_identity["sha256"],
            "--confirmation", f"execute-local-decoded-eval-shard-v1-{job_id}",
        ]
        try:
            replayed_executor_capture = bridge.validate_verified_capture_receipt(
                bindings,
                receipt_path=summary["executor_verified_release_capture"][
                    "receipt_path"
                ],
                target="action_preservation_decoded_eval_executor_v1.py",
                expected_arguments=expected_executor_arguments,
                expected_capture_digest=summary[
                    "executor_verified_release_capture"
                ]["capture_digest"],
                verify_file=True,
            )
        except bridge.DecodedEvaluationBridgeError as error:
            raise DecodedEvaluationAggregateError(str(error)) from error
        if replayed_executor_capture != summary[
            "executor_verified_release_capture"
        ]:
            raise DecodedEvaluationAggregateError(
                "holder executor verified capture replay differs"
            )
        summaries.append(
            {
                "job_id": job_id,
                "node": holder["node"],
                "summary_path": str(summary_path),
                "summary_sha256": plan.file_sha256(summary_path),
                "summary_digest": summary["summary_digest"],
                "holder_execution_digest": summary[
                    "holder_execution_authority"
                ]["holder_execution_digest"],
                "executor_verified_release_capture": replayed_executor_capture,
            }
        )
        for task, result in zip(shard["tasks"], summary["results"]):
            task_id = executor._task_id(task)
            task_root = shard_root / "tasks" / task_id
            request_path = task_root / executor.INPUT_RECEIPT_FILENAME
            process_path = task_root / executor.PROCESS_RECEIPT_FILENAME
            output_receipt_path = task_root / executor.OUTPUT_RECEIPT_FILENAME
            staging_path = task_root / executor.STAGING_VIDEO_FILENAME
            final_path = root / task["record"]["output_relpath"]
            try:
                request = executor.validate_task_input_receipt(
                    _json(request_path, label=f"{task_id} input"),
                    task=task, bundle=bundle, shard=shard,
                )
                process = _json(process_path, label=f"{task_id} process")
                stdout = _bytes(task_root / executor.STDOUT_FILENAME, label=f"{task_id} stdout")
                stderr = _bytes(task_root / executor.STDERR_FILENAME, label=f"{task_id} stderr")
                expected_process = executor.build_process_receipt(
                    input_receipt=request,
                    observation={
                        "return_code": process.get("return_code"),
                        "stdout": stdout,
                        "stderr": stderr,
                    },
                    request_path=request_path,
                    staging_path=staging_path,
                )
                if process != expected_process or process["return_code"] != 0:
                    raise DecodedEvaluationAggregateError(
                        f"{task_id} process receipt differs"
                    )
                output_receipt = executor.validate_output_receipt(
                    _json(output_receipt_path, label=f"{task_id} output receipt"),
                    input_receipt=request,
                    process_receipt=process,
                    output_path=final_path,
                )
                replayed_probe = executor.validate_probe_result(
                    reprobe_video(final_path)
                )
                if output_receipt["probe"] != replayed_probe:
                    raise DecodedEvaluationAggregateError(
                        f"{task_id} current media probe differs from sealed receipt"
                    )
                normalized, task_bindings, source, checkpoint = (
                    decoder_adapter.resolve_request(request, verify_files=False)
                )
                native_receipt_path = staging_path.with_name(
                    staging_path.name + ".receipt.json"
                )
                native_receipt = decoder_adapter.validate_inference_receipt(
                    _json(native_receipt_path, label=f"{task_id} native receipt"),
                    request=normalized,
                    bindings=task_bindings,
                    source=source,
                    checkpoint=checkpoint,
                    output_path=staging_path,
                )
                decoder_capture = bridge.validate_verified_capture_receipt(
                    task_bindings,
                    receipt_path=(
                        task_root / executor.DECODER_RUNTIME_CAPTURE_FILENAME
                    ),
                    target=(
                        "action_preservation_decoded_eval_decoder_adapter_v1.py"
                    ),
                    expected_arguments=[
                        "--request", str(request_path),
                        "--output", str(staging_path),
                    ],
                    verify_file=True,
                )
                inference_arguments = decoder_adapter.inference_target_arguments(
                    request=normalized, bindings=task_bindings, source=source,
                    checkpoint=checkpoint, output_path=staging_path,
                )
                inference_capture = bridge.validate_verified_capture_receipt(
                    task_bindings,
                    receipt_path=decoder_adapter.inference_runtime_capture_path(
                        staging_path
                    ),
                    target="infer_lora.py",
                    expected_arguments=inference_arguments,
                    verify_file=True,
                )
                native = {
                    "receipt_path": str(native_receipt_path),
                    "receipt_sha256": plan.file_sha256(native_receipt_path),
                    "receipt_digest": native_receipt["receipt_digest"],
                    "decoder_verified_release_capture": decoder_capture,
                    "inference_verified_release_capture": inference_capture,
                }
            except (
                executor.DecodedEvaluationExecutorError,
                decoder_adapter.DecodedEvaluationDecoderError,
                bridge.DecodedEvaluationBridgeError,
            ) as error:
                raise DecodedEvaluationAggregateError(str(error)) from error
            if (
                request["physical_bindings"] != physical_identity
                or request["executor_verified_release_capture"]
                != replayed_executor_capture
                or request["decoder_adapter"]["sha256"] != expected_decoder_sha
                or request["executor_source_sha256"] != expected_executor_sha
                or request["tool_files_verified"] is not True
                or output_receipt["native_inference_receipt"] != native
                or output_receipt["exact_input_binding_satisfied"] is not True
                or output_receipt["media_contract_satisfied"] is not True
                or result["terminal_receipt_digest"] != output_receipt["output_digest"]
            ):
                raise DecodedEvaluationAggregateError(
                    f"{task_id} production evidence closure differs"
                )
            staging_info = _plain(staging_path, directory=False, label="sealed staging").lstat()
            final_info = _plain(final_path, directory=False, label="sealed output").lstat()
            if (
                (staging_info.st_dev, staging_info.st_ino)
                != (final_info.st_dev, final_info.st_ino)
                or staging_info.st_nlink != 2
                or final_info.st_nlink != 2
                or stat.S_IMODE(final_info.st_mode) != 0o444
            ):
                raise DecodedEvaluationAggregateError(
                    f"{task_id} output is not sealed to its staging evidence"
                )
            outputs.append(
                {
                    "task_kind": task["task_kind"],
                    "task_id": task_id,
                    "record": task["record"],
                    "output_path": str(final_path),
                    "output_video_sha256": output_receipt["output_video_sha256"],
                    "output_receipt_path": str(output_receipt_path),
                    "output_receipt_sha256": plan.file_sha256(output_receipt_path),
                    "output_digest": output_receipt["output_digest"],
                }
            )
    if (
        len(outputs) != 264
        or sum(item["task_kind"] == "adapter_candidate" for item in outputs) != 256
        or sum(item["task_kind"] == "frozen_base_control" for item in outputs) != 8
    ):
        raise DecodedEvaluationAggregateError("exact 264 output closure differs")
    try:
        replayed_bindings = bridge.load_physical_bindings(
            physical_identity["path"], expected_sha256=physical_identity["sha256"],
            verify_files=True,
        )
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    if replayed_bindings != bindings:
        raise DecodedEvaluationAggregateError(
            "physical bindings changed during aggregate verification"
        )
    return bundle, bindings, outputs, summaries


def build_blind_packet(
    *, bundle: Mapping[str, Any], bindings: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]],
    blinding_key: bytes,
    aggregate_verified_release_capture: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if len(blinding_key) < 32:
        raise DecodedEvaluationAggregateError("blinding key must contain at least 32 bytes")
    candidates = [dict(item) for item in outputs if item["task_kind"] == "adapter_candidate"]
    controls = {
        item["task_id"]: dict(item)
        for item in outputs
        if item["task_kind"] == "frozen_base_control"
    }
    if len(candidates) != 256 or len(controls) != 8 or len(outputs) != 264:
        raise DecodedEvaluationAggregateError("blind packet exact-264 closure differs")
    if len(summaries) != 4:
        raise DecodedEvaluationAggregateError("blind packet four-holder closure differs")
    aggregate_capture = executor._capture_evidence(
        aggregate_verified_release_capture,
        label="aggregate verified release capture",
    )
    if (
        aggregate_capture is None
        or aggregate_capture["target"]
        != "action_preservation_decoded_eval_aggregate_v1.py"
    ):
        raise DecodedEvaluationAggregateError(
            "aggregate verified release capture differs"
        )
    sources = {item["iid"]: item for item in bindings["sources"]}
    private_rows = []
    public_rows = []
    for candidate in candidates:
        record = candidate["record"]
        control = controls.get(record["matched_frozen_base_control_id"])
        if control is None:
            raise DecodedEvaluationAggregateError("matched frozen-base output is missing")
        base = control["record"]
        if any(
            record[key] != base[key]
            for key in (
                "iid", "seed", "onset_policy", "source_video_sha256",
                "source_receipt_sha256", "instruction", "instruction_sha256",
                "action_review_contract",
                "model_release_manifest_sha256", "inference_source_sha256",
                "inference_release_manifest_sha256", "inference_config_sha256",
                "source_preprocessing_sha256", "calibration_digest",
            )
        ):
            raise DecodedEvaluationAggregateError("matched frozen-base pairing differs")
        blind_digest = hmac.new(
            blinding_key,
            ("id\0" + record["candidate_id"]).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        order_digest = hmac.new(
            blinding_key,
            ("order\0" + record["candidate_id"]).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        blind_id = "blind-" + blind_digest[:32]
        source = sources[record["iid"]]["source_video"]
        public_row = {
            "blind_candidate_id": blind_id,
            "source_media_sha256": source["sha256"],
            "source_receipt_sha256": record["source_receipt_sha256"],
            "source_media_relpath": f"{MEDIA_DIRECTORY}/{source['sha256']}.mp4",
            "review_media_sha256": candidate["output_video_sha256"],
            "review_media_relpath": (
                f"{MEDIA_DIRECTORY}/{candidate['output_video_sha256']}.mp4"
            ),
            "review_output_digest": candidate["output_digest"],
            "full_video_receipt_sha256": candidate["output_receipt_sha256"],
            "matched_base_media_sha256": control["output_video_sha256"],
            "matched_base_media_relpath": (
                f"{MEDIA_DIRECTORY}/{control['output_video_sha256']}.mp4"
            ),
            "matched_base_output_digest": control["output_digest"],
            "matched_base_full_video_receipt_sha256": control[
                "output_receipt_sha256"
            ],
            "instruction": record["instruction"],
            "instruction_sha256": record["instruction_sha256"],
            "action_review_contract": record["action_review_contract"],
            "action_review_contract_digest": record["action_review_contract"][
                "contract_digest"
            ],
            "required_axes": list(plan.REVIEW_AXES),
            "minimum_independent_reviewer_count": 2,
            "full_81_frame_video_required": True,
        }
        public_row["blind_row_digest"] = object_sha256(public_row)
        public_rows.append(public_row)
        private_row = {
            "blind_candidate_id": blind_id,
            "blind_row_digest": public_row["blind_row_digest"],
            "order_digest": order_digest,
            "candidate_id": record["candidate_id"],
            "arm": record["arm"],
            "checkpoint_step": record["checkpoint_step"],
            "iid": record["iid"],
            "onset_policy": record["onset_policy"]["name"],
            "matched_control_id": control["task_id"],
            "candidate_output_path": candidate["output_path"],
            "candidate_output_receipt_path": candidate["output_receipt_path"],
            "candidate_output_receipt_sha256": candidate["output_receipt_sha256"],
            "candidate_output_digest": candidate["output_digest"],
            "matched_base_output_receipt_path": control["output_receipt_path"],
            "matched_base_output_receipt_sha256": control["output_receipt_sha256"],
            "matched_base_output_digest": control["output_digest"],
            "instruction_sha256": record["instruction_sha256"],
            "action_review_contract_digest": record["action_review_contract"][
                "contract_digest"
            ],
        }
        private_row["private_row_digest"] = object_sha256(private_row)
        private_rows.append(private_row)
    ordering = {
        row["blind_candidate_id"]: row["order_digest"] for row in private_rows
    }
    private_rows.sort(key=lambda item: item["order_digest"])
    public_rows.sort(key=lambda item: ordering[item["blind_candidate_id"]])
    if len({item["blind_candidate_id"] for item in public_rows}) != 256:
        raise DecodedEvaluationAggregateError("blind candidate identifiers collide")
    key_sha = hashlib.sha256(blinding_key).hexdigest()
    private = {
        "schema_version": PRIVATE_PACKET_SCHEMA,
        "evaluation_id": bundle["manifest"]["evaluation_id"],
        "evaluation_manifest_digest": bundle["manifest"]["manifest_digest"],
        "blinding_key_sha256": key_sha,
        "rows": private_rows,
        "row_count": 256,
        "method_arm_checkpoint_policy_private": True,
    }
    private["private_mapping_digest"] = object_sha256(private)
    public = {
        "schema_version": PUBLIC_PACKET_SCHEMA,
        "packet_id": "packet-" + hmac.new(
            blinding_key, b"packet", hashlib.sha256
        ).hexdigest()[:32],
        "review_contract_digest": bundle["review_contract"]["contract_digest"],
        "private_mapping_digest": private["private_mapping_digest"],
        "rows": public_rows,
        "row_count": 256,
        "method_hidden": True,
        "arm_hidden": True,
        "checkpoint_hidden": True,
        "onset_policy_hidden": True,
        "private_key_in_public_packet": False,
        "training_loss_present": False,
    }
    public["public_packet_digest"] = object_sha256(public)
    calibration = bindings["calibration_digest"]
    aggregate = {
        "schema_version": AGGREGATE_SCHEMA,
        "evaluation_id": bundle["manifest"]["evaluation_id"],
        "evaluation_manifest_digest": bundle["manifest"]["manifest_digest"],
        "physical_bindings_digest": bindings["physical_bindings_digest"],
        "holder_summaries": [dict(item) for item in summaries],
        "holder_count": 4,
        "candidate_output_count": 256,
        "matched_base_output_count": 8,
        "total_output_count": 264,
        "exact_full81_at_25fps_pts_verified": True,
        "all_native_inference_receipts_verified": True,
        "all_outputs_create_only_and_sealed": True,
        "aggregate_verified_release_capture": aggregate_capture,
        "automatic_retry_count": 0,
        "training_loss_read_or_used": False,
        "checkpoint_loss_ranking": False,
        "private_mapping_digest": private["private_mapping_digest"],
        "public_packet_digest": public["public_packet_digest"],
        "blinding_key_sha256": key_sha,
        "machine_calibration_digest": calibration,
        "machine_status": (
            "WAIT_FOR_MACHINE_MEASUREMENT"
            if calibration is not None
            else "ABSTAIN_CALIBRATION_MISSING"
        ),
        "blind_review_status": "WAIT_FOR_BLIND_REVIEW",
        "next_action": "WAIT_FOR_BLIND_REVIEW",
        "scientific_promotion_authorized": False,
    }
    aggregate["aggregate_digest"] = object_sha256(aggregate)
    return aggregate, private, public


def _write(path: Path, payload: bytes, *, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError as error:
        raise DecodedEvaluationAggregateError(f"refusing to overwrite: {path}") from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise DecodedEvaluationAggregateError("create-only write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_media(source: Path, destination: Path, expected_sha256: str) -> None:
    source = _plain(source, directory=False, label="blind packet source media")
    source_flags = os.O_RDONLY
    output_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
        output_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, source_flags)
    try:
        before = os.fstat(source_descriptor)
        try:
            output_descriptor = os.open(destination, output_flags, 0o444)
        except FileExistsError as error:
            raise DecodedEvaluationAggregateError(
                f"refusing to overwrite: {destination}"
            ) from error
        first_digest = hashlib.sha256()
        try:
            while True:
                block = os.read(source_descriptor, 1024 * 1024)
                if not block:
                    break
                first_digest.update(block)
                offset = 0
                while offset < len(block):
                    written = os.write(output_descriptor, block[offset:])
                    if written <= 0:
                        raise DecodedEvaluationAggregateError(
                            "blind media copy made no progress"
                        )
                    offset += written
            os.fchmod(output_descriptor, 0o444)
            os.fsync(output_descriptor)
            middle = os.fstat(source_descriptor)
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            second_digest = hashlib.sha256()
            while True:
                block = os.read(source_descriptor, 1024 * 1024)
                if not block:
                    break
                second_digest.update(block)
            destination_before = os.fstat(output_descriptor)
            os.lseek(output_descriptor, 0, os.SEEK_SET)
            destination_digest = hashlib.sha256()
            while True:
                block = os.read(output_descriptor, 1024 * 1024)
                if not block:
                    break
                destination_digest.update(block)
            destination_after = os.fstat(output_descriptor)
            destination_named = destination.lstat()
        finally:
            os.close(output_descriptor)
        after = os.fstat(source_descriptor)
        named = source.lstat()
    finally:
        os.close(source_descriptor)
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_uid, item.st_gid, item.st_mode,
        item.st_nlink, item.st_rdev, item.st_size,
        getattr(item, "st_blocks", 0), item.st_mtime_ns, item.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink not in {1, 2}
        or identity(before) != identity(middle)
        or identity(before) != identity(after)
        or identity(before) != identity(named)
        or first_digest.hexdigest() != expected_sha256
        or second_digest.hexdigest() != expected_sha256
        or destination_digest.hexdigest() != expected_sha256
        or identity(destination_before) != identity(destination_after)
        or identity(destination_before) != identity(destination_named)
        or destination_before.st_nlink != 1
        or stat.S_IMODE(destination_before.st_mode) != 0o444
    ):
        raise DecodedEvaluationAggregateError("blind packet source media changed or differs")


def publish(
    *, aggregate_root: str | Path, aggregate: Mapping[str, Any],
    private: Mapping[str, Any], public: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]], bindings: Mapping[str, Any],
) -> Path:
    root = Path(aggregate_root)
    if not root.is_absolute() or str(root) == os.path.sep or os.path.normpath(str(root)) != str(root):
        raise DecodedEvaluationAggregateError("aggregate root must be normalized and absolute")
    if os.path.lexists(root):
        raise DecodedEvaluationAggregateError("aggregate root is not fresh")
    _plain(root.parent, directory=True, label="aggregate parent")
    os.mkdir(root, 0o700)
    media_root = root / MEDIA_DIRECTORY
    os.mkdir(media_root, 0o700)
    media: dict[str, Path] = {}
    for source in bindings["sources"]:
        media[source["source_video"]["sha256"]] = Path(source["source_video"]["path"])
    for item in outputs:
        media[item["output_video_sha256"]] = Path(item["output_path"])
    for digest, source_path in sorted(media.items()):
        _copy_media(source_path, media_root / f"{digest}.mp4", digest)
    os.chmod(media_root, 0o555)
    _write(
        root / PRIVATE_FILENAME, canonical_json_bytes(private) + b"\n", mode=0o400
    )
    _write(root / PUBLIC_FILENAME, canonical_json_bytes(public) + b"\n", mode=0o444)
    output = root / AGGREGATE_FILENAME
    _write(output, canonical_json_bytes(aggregate) + b"\n", mode=0o444)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--physical-bindings", required=True)
    parser.add_argument("--physical-bindings-sha256", required=True)
    parser.add_argument("--blinding-key-file", required=True)
    parser.add_argument("--aggregate-root", required=True)
    args = parser.parse_args(argv)
    key_path = Path(args.blinding_key_file)
    if (
        not key_path.is_absolute()
        or key_path.resolve(strict=True) != key_path
        or stat.S_IMODE(key_path.lstat().st_mode) != 0o400
        or key_path.lstat().st_nlink != 1
    ):
        raise DecodedEvaluationAggregateError(
            "blinding key must be one canonical absolute mode-0400 single-link file"
        )
    blinding_key = _bytes(key_path, label="blinding key")
    bundle, bindings, outputs, summaries = collect_verified_outputs(
        evaluation_root=args.evaluation_root,
        physical_bindings_path=args.physical_bindings,
        physical_bindings_sha256=args.physical_bindings_sha256,
    )
    expected_arguments = list(sys.argv[1:] if argv is None else argv)
    expected_capture_path = Path(args.aggregate_root).with_name(
        Path(args.aggregate_root).name + ".runtime-capture.json"
    )
    try:
        aggregate_capture = bridge.validate_running_verified_capture(
            bindings,
            target="action_preservation_decoded_eval_aggregate_v1.py",
            expected_arguments=expected_arguments,
            verify_file=True,
        )
    except bridge.DecodedEvaluationBridgeError as error:
        raise DecodedEvaluationAggregateError(str(error)) from error
    if aggregate_capture["receipt_path"] != str(expected_capture_path):
        raise DecodedEvaluationAggregateError(
            "aggregate runtime capture path differs"
        )
    aggregate, private, public = build_blind_packet(
        bundle=bundle, bindings=bindings, outputs=outputs,
        summaries=summaries, blinding_key=blinding_key,
        aggregate_verified_release_capture=aggregate_capture,
    )
    output = publish(
        aggregate_root=args.aggregate_root, aggregate=aggregate,
        private=private, public=public, outputs=outputs, bindings=bindings,
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
