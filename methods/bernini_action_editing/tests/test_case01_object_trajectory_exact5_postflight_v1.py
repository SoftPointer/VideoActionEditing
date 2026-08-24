#!/usr/bin/env python3
"""Hostile tests for the trajectory exact5 postflight and offline site."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from methods.bernini_action_editing import case01_source_object_strict_eval_v1 as strict_eval
from methods.bernini_action_editing.tools import (
    build_case01_object_trajectory_exact5_html_v1 as html_builder,
)
from methods.bernini_action_editing.tools import (
    case01_object_trajectory_exact5_postflight_v1 as postflight,
)
from methods.bernini_action_editing.tests import (
    test_case01_object_trajectory_exact5_core_v1 as v3_fixture_helpers,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
POSTFLIGHT_SCRIPT = (
    REPO_ROOT
    / "methods/bernini_action_editing/tools/"
    "case01_object_trajectory_exact5_postflight_v1.py"
)
HTML_SCRIPT = (
    REPO_ROOT
    / "methods/bernini_action_editing/tools/"
    "build_case01_object_trajectory_exact5_html_v1.py"
)


def _seal(value: dict, field: str) -> dict:
    value.pop(field, None)
    value[field] = postflight.object_sha256(value)
    return value


def _write_json(path: Path, value: dict, mode: int) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = postflight.canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    path.chmod(mode)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _rewrite_json(path: Path, value: dict, mode: int) -> tuple[str, int]:
    if path.exists():
        path.chmod(0o600)
    return _write_json(path, value, mode)


def _rewrite_bytes(path: Path, payload: bytes, mode: int) -> tuple[str, int]:
    if path.exists():
        path.chmod(0o600)
    return _write_bytes(path, payload, mode)


def _write_bytes(path: Path, payload: bytes, mode: int) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _minimal_jpeg(*, width: int, height: int) -> bytes:
    """Enough JPEG structure for the production SOF geometry parser."""

    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + bytes([8])
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + bytes([3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0])
        + b"\xff\xd9"
    )


def _synthetic_scanned_jpeg(
    *, width: int, height: int, entropy: bytes = b"synthetic"
) -> bytes:
    """Parser-level JPEG fixture with SOS/EOI closure, not real media."""

    sof_only = _minimal_jpeg(width=width, height=height)[:-2]
    sos = (
        b"\xff\xda\x00\x0c\x03"
        b"\x01\x00\x02\x00\x03\x00\x00\x3f\x00"
    )
    return sof_only + sos + entropy + b"\xff\xd9"


def _passing_observation(arm: str) -> dict:
    return {
        "variant": arm,
        "review_coverage": {
            "all_81_decoded_frames_reviewed": True,
            "source_and_output_pair_reviewed": True,
            "frame_range": [0, 80],
            "frame_count": 81,
        },
        "dog_identity": {
            "subject_track_id": "dog#1",
            "identity_switch_observed": False,
            "first_mismatch_frame": None,
            "cues": [
                {
                    "name": cue,
                    "source": f"source-{cue}",
                    "output": f"preserved-{cue}",
                    "preserved": True,
                }
                for cue in strict_eval.IDENTITY_CUES
            ],
        },
        "source_bone": {
            "patient_track_id": "bone#1",
            "input_patient_available": True,
            "same_instance_continuity": "PROVEN",
            "left_initial_support": True,
            "entered_effector_region": True,
            "terminal_hold": True,
            "source_instance_remains_in_background": False,
            "duplicate_or_substitute_prop": {
                "observed": False,
                "frame_interval": None,
                "description": "No duplicate or substitute prop in the test observation.",
            },
            "observed_state": "bone#1 moves continuously from source support to dog#1.mouth",
        },
        "action_trace": {
            "patient_track_id": "bone#1",
            "effector_region_id": "dog#1.mouth",
            "minimum_hold_frames": 10,
            "stages": [
                {
                    "name": stage,
                    "observed": True,
                    "frame_interval": interval,
                    "evidence": f"Test-only all-frame evidence for {stage}.",
                }
                for stage, interval in zip(
                    strict_eval.ACTION_STAGES,
                    ([0, 15], [20, 22], [25, 30], [35, 42], [50, 80]),
                )
            ],
        },
    }


def _stat_identity(path: Path) -> dict[str, int]:
    info = path.lstat()
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": info.st_mode,
        "nlink": info.st_nlink,
        "rdev": info.st_rdev,
        "size": info.st_size,
        "blocks": getattr(info, "st_blocks", 0),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _directory_identity(path: Path) -> dict[str, int]:
    info = path.lstat()
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": info.st_mode,
        "rdev": info.st_rdev,
    }


def _authority_digest(tag: str) -> str:
    return postflight.object_sha256({"test-authority": tag})


class CompletedRunFixture:
    def __init__(
        self,
        root: Path,
        *,
        real_ffmpeg: Path,
        real_ffprobe: Path,
        output_template: Path,
    ) -> None:
        self.root = root
        self.output_root = root / "native-media"
        self.output_root.mkdir(parents=True)
        evaluator = postflight.trajectory_eval
        self.source = root / "exact_original.mp4"
        shutil.copyfile(v3_fixture_helpers.SOURCE, self.source)
        self.source.chmod(0o444)
        self.ffprobe = root / "ffprobe"
        shutil.copyfile(real_ffprobe, self.ffprobe)
        self.ffprobe.chmod(0o555)
        ffprobe_payload = self.ffprobe.read_bytes()
        ffprobe_sha = hashlib.sha256(ffprobe_payload).hexdigest()
        ffprobe_size = len(ffprobe_payload)
        self.ffmpeg = real_ffmpeg
        ffmpeg_payload = self.ffmpeg.read_bytes()
        ffmpeg_sha = hashlib.sha256(ffmpeg_payload).hexdigest()
        ffmpeg_size = len(ffmpeg_payload)
        self.ffmpeg_sha = ffmpeg_sha
        self.ffmpeg_size = ffmpeg_size
        self.cached_sheets: dict[str, dict] = {}
        with mock.patch.object(
            postflight, "EXPECTED_FFMPEG_SHA256", ffmpeg_sha
        ), mock.patch.object(
            postflight, "EXPECTED_FFMPEG_SIZE", ffmpeg_size
        ):
            for kind, video, probe in (
                ("source", self.source, postflight.EXPECTED_SOURCE_PROBE),
                ("output", output_template, postflight.EXPECTED_OUTPUT_PROBE),
            ):
                cached_path = root / f"cached-{kind}-all81.jpg"
                cached = postflight.make_all81_sheet(
                    video,
                    cached_path,
                    ffmpeg=self.ffmpeg,
                    input_probe=probe,
                )
                cached["payload"] = cached_path.read_bytes()
                self.cached_sheets[kind] = cached
        source_authority = evaluator.build_file_authority(
            self.source, role="exact_original_source"
        )
        condition_authorities = {
            "stage0_masks": evaluator.build_file_authority(
                v3_fixture_helpers.STAGE0,
                role="stage0_masks",
                payload_digest=evaluator.EXPECTED_STAGE0_RECEIPT_DIGEST,
            ),
            "g0_mouth_track": evaluator.build_file_authority(
                v3_fixture_helpers.G0, role="g0_mouth_track"
            ),
            "trajectory_scaffold": evaluator.build_file_authority(
                v3_fixture_helpers.SCAFFOLD,
                role="trajectory_scaffold",
                payload_digest=(
                    evaluator.EXPECTED_TRAJECTORY_SCAFFOLD_ARTIFACT_DIGEST
                ),
            ),
            "aux_bone_removed_source": evaluator.build_file_authority(
                v3_fixture_helpers.REMOVED,
                role="aux_bone_removed_source",
            ),
        }
        admission_authorities = {
            "scaffold_independent_audit": evaluator.build_file_authority(
                v3_fixture_helpers.SCAFFOLD_AUDIT,
                role="scaffold_independent_audit",
                payload_digest=evaluator.EXPECTED_SCAFFOLD_AUDIT_DIGEST,
            )
        }
        release = root / "sealed-release"
        release.mkdir()
        sealed_fixture = Path(
            "/tmp/case01_object_trajectory_v1_sealed_methods_fixture"
        )
        legacy_source = sealed_fixture / "infer_lora_full644_r5_frozen_acc46.py"
        if not legacy_source.is_file():
            raise AssertionError("sealed v5 legacy inference fixture is required")
        producer_sources = {
            "infer_lora_path": (
                legacy_source,
                "infer_lora_full644_r5_frozen_acc46.py",
            ),
            "inference_wrapper_path": (
                REPO_ROOT / "methods/bernini_action_editing/"
                "infer_case01_object_trajectory_oracle_auh_r5f_v3.py",
                "infer_case01_object_trajectory_oracle_auh_r5f_v3.py",
            ),
            "object_wrapper_inner_path": (
                REPO_ROOT / "methods/bernini_action_editing/"
                "infer_case01_object_trajectory_oracle_v1.py",
                "infer_case01_object_trajectory_oracle_v1.py",
            ),
            "trajectory_projection_module_path": (
                REPO_ROOT / "methods/bernini_action_editing/"
                "object_trajectory_projection_v1.py",
                "object_trajectory_projection_v1.py",
            ),
            "trajectory_scaffold_module_path": (
                REPO_ROOT / "methods/bernini_action_editing/"
                "case01_oracle_object_trajectory_v1.py",
                "case01_oracle_object_trajectory_v1.py",
            ),
        }
        copied_sources: dict[str, Path] = {}
        for field, (source, basename) in producer_sources.items():
            target = release / basename
            shutil.copyfile(source, target)
            target.chmod(0o444)
            copied_sources[field] = target
        producer = evaluator.incomplete_producer()
        producer.update(
            {
                **{field: str(path) for field, path in copied_sources.items()},
                "ffprobe_path": str(self.ffprobe),
                "ffprobe_sha256": ffprobe_sha,
                "ffprobe_size": ffprobe_size,
                "method_source_revision": "postflight-real-v3-fixture",
                "method_source_archive_sha256": "f" * 64,
                "pins_complete": True,
            }
        )
        self.producer = producer
        checkpoint_manifest = {
            "path": str(root / "checkpoint/terminal-manifest.json"),
            "pin_complete": True,
            **evaluator.EXPECTED_CHECKPOINT,
        }
        self.plan = evaluator.build_plan(
            source_authority=source_authority,
            condition_authorities=condition_authorities,
            admission_authorities=admission_authorities,
            checkpoint_manifest=checkpoint_manifest,
            producer=producer,
            output_root=self.output_root,
            launch_allowed=True,
        )
        tasks = self.plan["tasks"]
        self.tasks = tasks
        output_rows: list[tuple[Path, str, int, Path]] = []
        output_payload = output_template.read_bytes()
        for arm, task in zip(postflight.ARM_ORDER, tasks):
            output = Path(task["output"]["video_path"])
            output_sha, output_size = _write_bytes(
                output, output_payload, 0o444
            )
            output_rows.append(
                (output, output_sha, output_size, Path(task["output"]["receipt_path"]))
            )
        self.plan_path = root / "object_trajectory_exact5_plan_v3.json"
        plan_sha, _ = _write_json(self.plan_path, self.plan, 0o444)

        model_capture = "1" * 64
        receipt_rows: list[dict] = []
        receipt_files: list[tuple[str, int]] = []
        result_rows: list[dict] = []
        helper_evaluator = v3_fixture_helpers.evaluator
        helper_receipt_schema = v3_fixture_helpers.wrapper.WRAPPER_RECEIPT_SCHEMA
        helper_runtime_schema = v3_fixture_helpers.wrapper.RUNTIME_TRACE_SCHEMA
        helper_hashes = v3_fixture_helpers._FakeAssets.producer_hashes

        def v3_producer_hashes(assets) -> dict[str, str]:
            fixture_producer = assets._producer
            return evaluator._expected_oracle_producer_hashes(fixture_producer)

        v3_fixture_helpers.evaluator = evaluator
        v3_fixture_helpers.wrapper.WRAPPER_RECEIPT_SCHEMA = (
            evaluator.INFERENCE_RECEIPT_SCHEMA
        )
        v3_fixture_helpers.wrapper.RUNTIME_TRACE_SCHEMA = (
            evaluator.OBJECT_ORACLE_RUNTIME_SCHEMA
        )
        v3_fixture_helpers._FakeAssets.producer_hashes = v3_producer_hashes
        try:
            for index, (arm, task, output_row) in enumerate(
                zip(postflight.ARM_ORDER, tasks, output_rows)
            ):
                output, output_sha, output_size, receipt_path = output_row
                task_input = postflight.object_sha256(
                    {
                        "schema_version": "full644-exploratory-matched-task-input-v2",
                        "plan_digest": self.plan["plan_digest"],
                        "task": task,
                    }
                )
                receipt = (
                    v3_fixture_helpers._legacy_receipt(
                        task, producer, output_sha=output_sha
                    )
                    if arm in {"null_before", "null_after"}
                    else v3_fixture_helpers._custom_receipt(task, producer)
                )
                receipt["task_input_digest"] = task_input
                receipt["model_consumption"]["task_input_digest"] = task_input
                receipt["output"]["path"] = str(output)
                receipt["output"]["sha256"] = output_sha
                receipt["output"]["size"] = output_size
                receipt["output"]["sealed_source_sha256"] = output_sha
                receipt["output"]["sealed_source_size"] = output_size
                receipt["output"]["publication_identity"] = _stat_identity(output)
                receipt["output"]["prepublication_identity"]["size"] = (
                    output_size
                )
                _seal(receipt, "receipt_digest")
                if arm in {"null_before", "null_after"}:
                    evaluator.validate_off_inference_receipt(receipt, task, producer)
                else:
                    evaluator.validate_custom_inference_receipt(receipt, task, producer)
                receipt_sha, receipt_size = _write_json(receipt_path, receipt, 0o400)
                receipt_rows.append(receipt)
                receipt_files.append((receipt_sha, receipt_size))
                result_rows.append(
                    {
                        "task_id": task["task_id"],
                        "arm": "full644",
                        "oracle_arm": arm,
                        "receipt_path": str(receipt_path),
                        "output_path": str(output),
                        "receipt_file_sha256": receipt_sha,
                        "receipt_digest": receipt["receipt_digest"],
                        "output_sha256": output_sha,
                        "output_size": output_size,
                        "media_probe": {
                            "ffprobe_path": str(self.ffprobe),
                            "ffprobe_sha256": ffprobe_sha,
                            "ffprobe_size": ffprobe_size,
                            **postflight.EXPECTED_OUTPUT_PROBE,
                        },
                    }
                )
        finally:
            v3_fixture_helpers._FakeAssets.producer_hashes = helper_hashes
            v3_fixture_helpers.wrapper.RUNTIME_TRACE_SCHEMA = helper_runtime_schema
            v3_fixture_helpers.wrapper.WRAPPER_RECEIPT_SCHEMA = helper_receipt_schema
            v3_fixture_helpers.evaluator = helper_evaluator
        self.receipts = receipt_rows
        self.report = {
            "schema_version": postflight.trajectory_eval.REPORT_SCHEMA,
            "status": "ENGINEERING_ORACLE_COMPLETE_AWAITING_MANUAL_REVIEW",
            "campaign_mode": postflight.trajectory_eval.CAMPAIGN,
            "plan_schema_version": postflight.trajectory_eval.SCHEMA_VERSION,
            "plan_digest": self.plan["plan_digest"],
            "task_count": 5,
            "task_ids": list(postflight.TASK_IDS),
            "variant_order": list(postflight.ARM_ORDER),
            "all_exact5_tasks_verified_no_cherry_pick": True,
            "same_model_capture_all_tasks": True,
            "null_envelope": evaluator.validate_null_envelope_receipts(
                receipt_rows[0], receipt_rows[4]
            ),
            "retained_publication_root_fd_replayed": True,
            "retained_ffprobe_executable_fd_replayed": True,
            "retained_publication_leaf_fds_replayed": True,
            "manual_blind_review_required": True,
            "formal_full16_report": False,
            "results": result_rows,
            "claim_limits": self.plan["claim_limits"],
        }
        _seal(self.report, "report_digest")
        self.report_path = root / "object_trajectory_exact5_report_v3.json"
        report_sha, _ = _write_json(self.report_path, self.report, 0o444)

        pinned_identities: dict[str, dict] = {}
        producer_role_paths = {
            "adapter": producer["inference_wrapper_path"],
            "object_wrapper_inner": producer["object_wrapper_inner_path"],
            "legacy_infer_lora": producer["infer_lora_path"],
            "trajectory_projection": producer["trajectory_projection_module_path"],
            "trajectory_scaffold": producer["trajectory_scaffold_module_path"],
            "ffprobe": producer["ffprobe_path"],
        }
        producer_role_hashes = {
            "adapter": producer["inference_wrapper_sha256"],
            "object_wrapper_inner": producer["object_wrapper_inner_sha256"],
            "legacy_infer_lora": producer["infer_lora_sha256"],
            "trajectory_projection": producer["trajectory_projection_module_sha256"],
            "trajectory_scaffold": producer["trajectory_scaffold_module_sha256"],
            "ffprobe": producer["ffprobe_sha256"],
        }
        producer_role_sizes = {
            "adapter": producer["inference_wrapper_size"],
            "object_wrapper_inner": producer["object_wrapper_inner_size"],
            "legacy_infer_lora": producer["infer_lora_size"],
            "trajectory_projection": producer["trajectory_projection_module_size"],
            "trajectory_scaffold": producer["trajectory_scaffold_module_size"],
            "ffprobe": producer["ffprobe_size"],
            "ffmpeg": self.ffmpeg_size,
        }
        for index, role in enumerate(sorted(postflight.PHYSICAL_IDENTITY_ROLES)):
            sha = producer_role_hashes.get(
                role,
                (
                    self.ffmpeg_sha
                    if role == "ffmpeg"
                    else postflight.PINNED_PHYSICAL_SHA256.get(
                        role, _authority_digest(role)
                    )
                ),
            )
            path = producer_role_paths.get(role, f"/frozen/{role}")
            mode = 0o555 if role in {"python", "ffmpeg", "ffprobe"} else 0o444
            pinned_identities[role] = {
                "path": path,
                "sha256": sha,
                "size": (
                    producer_role_sizes.get(
                        role,
                        postflight.PINNED_PHYSICAL_SIZE.get(role, 100 + index),
                    )
                ),
                "mode": mode,
                "device": 90,
                "inode": 1000 + index,
                "uid": os.getuid(),
                "gid": os.getgid(),
                "nlink": 1,
            }

        def full_identity(role: str) -> dict[str, int]:
            row = pinned_identities[role]
            return {
                "device": row["device"], "inode": row["inode"],
                "uid": row["uid"], "gid": row["gid"],
                "mode": stat.S_IFREG | row["mode"], "nlink": row["nlink"],
                "rdev": 0, "size": row["size"], "blocks": 8,
                "mtime_ns": 11, "ctime_ns": 12,
            }

        captured_entry = {
            "schema_version": "full644-exploratory-matched-captured-runner-entry-authority-v1",
            "runner_fd": 20,
            "runner_path": pinned_identities["runner"]["path"],
            "runner_sha256": pinned_identities["runner"]["sha256"],
            "runner_identity": full_identity("runner"),
            "python_fd": 21,
            "python_path": pinned_identities["python"]["path"],
            "python_sha256": pinned_identities["python"]["sha256"],
            "python_identity": full_identity("python"),
            "release_digest": _authority_digest("release"),
            "bootstrap_sha256": _authority_digest("bootstrap"),
            "entry_method": "slurm-spooled-or-trusted-stdin-held-python-fd-v1",
            "slurm_export_none_required": True,
            "bash_privileged_startup_required": True,
            "captured_source_entry": True,
        }
        _seal(captured_entry, "authority_digest")
        exec_rows = []
        for fd, (exec_role, physical_role) in enumerate(
            zip(postflight.EXEC_AUTHORITY_ROLES, ("python", "bridge", "adapter", "ffmpeg")),
            start=30,
        ):
            pinned = pinned_identities[physical_role]
            exec_rows.append(
                {
                    "role": exec_role,
                    "fd": fd,
                    "source_path": pinned["path"],
                    "sha256": pinned["sha256"],
                    "identity": full_identity(physical_role),
                }
            )
        exec_authority = {
            "schema_version": "full644-exploratory-matched-exec-authority-v2",
            "rows": exec_rows,
            "rows_digest": postflight.object_sha256(exec_rows),
        }
        _seal(exec_authority, "binding_digest")
        ffprobe_authority = {
            "schema_version": "bernini-full644-exploratory-matched-ffprobe-exec-authority-v1",
            "fd": 40,
            "source_path": pinned_identities["ffprobe"]["path"],
            "sha256": pinned_identities["ffprobe"]["sha256"],
            "identity": full_identity("ffprobe"),
        }
        _seal(ffprobe_authority, "authority_digest")
        slurm_sources = {
            "job_id": "SLURM_JOB_ID", "step_id": "SLURM_STEP_ID",
            "gpu_count": "SLURM_GPUS_ON_NODE", "gpus_per_node": "SLURM_GPUS_PER_NODE",
            "step_gpu_indices": "SLURM_STEP_GPUS", "job_node_count": "SLURM_NNODES",
            "step_node_count": "SLURM_STEP_NUM_NODES", "job_nodelist": "SLURM_JOB_NODELIST",
            "step_nodelist": "SLURM_STEP_NODELIST",
        }
        slurm_raw = {
            "SLURM_JOB_ID": "123", "SLURM_STEP_ID": "1",
            "SLURM_GPUS_ON_NODE": "8", "SLURM_GPUS_PER_NODE": "8",
            "SLURM_STEP_GPUS": "0,1,2,3,4,5,6,7", "SLURM_NNODES": "1",
            "SLURM_STEP_NUM_NODES": "1", "SLURM_JOB_NODELIST": "node-test",
            "SLURM_STEP_NODELIST": "node-test",
        }
        self.attestation_path = root / "object_trajectory_exact5_runner_attestation_v3.json"
        authority_binding = {
            "source_authority": source_authority,
            "condition_authorities": condition_authorities,
            "admission_authorities": admission_authorities,
        }
        physical_bindings = {
            "schema_version": "case01-object-trajectory-exact5-physical-bindings-v3",
            "plan_path": str(self.plan_path),
            "plan_sha256": plan_sha,
            "plan_digest": self.plan["plan_digest"],
            "authority_binding_digest": postflight.object_sha256(authority_binding),
            "source_authority_digest": source_authority["authority_digest"],
            "condition_authority_digests": {
                key: row["authority_digest"] for key, row in sorted(condition_authorities.items())
            },
            "admission_authority_digests": {
                key: row["authority_digest"] for key, row in sorted(admission_authorities.items())
            },
            "producer_roles_distinct": {
                "invoked_adapter_source": "r5f_composite_inference_wrapper",
                "object_wrapper_inner_source": "source_loaded_support_only",
                "single_frozen_legacy_module": "base_adapter_infer_lora",
                "composite_inner_and_legacy_hashes_distinct": True,
            },
            "allocation": {
                "holder_job_id": "123", "node": "node-test", "slurm_step_id": "1",
                "slurm_environment_source_names": slurm_sources,
                "slurm_environment_raw_values": slurm_raw,
                "slurm_observed_absent_fields": ["SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"],
                "normalized_slurm_authority": {
                    "job_node_count": 1, "step_node_count": 1,
                    "gpu_count_on_node": 8, "gpus_per_node": 8,
                    "step_gpu_indices": list(range(8)),
                    "job_node": "node-test", "step_node": "node-test",
                },
                "world_size": 4, "ulysses_size": 4, "reserved_gpu_count": 8,
                "visible_gpu_indices": [0, 1, 2, 3],
            },
            "identities": pinned_identities,
            "captured_runner_entry": captured_entry,
            "captured_runner_entry_required": True,
            "exec_authority": exec_authority,
            "exec_authority_retained_source_and_python_fds": True,
            "ffprobe_authority": ffprobe_authority,
            "ffprobe_retained_executable_fd": True,
            "isolated_child_interpreters": "-I -S -B",
            "child_environment_exact_allowlist": True,
            "model_root": "/frozen/model",
            "bernini_root": "/frozen/bernini",
            "veomni_root": "/frozen/veomni",
            "campaign_mode": postflight.trajectory_eval.CAMPAIGN,
            "formal_full16_report": False,
            "task_count": 5,
            "task_ids": list(postflight.TASK_IDS),
            "retry_allowed": False,
            "final_artifacts": {
                "output_report": str(self.report_path),
                "runner_attestation": str(self.attestation_path),
            },
        }
        _seal(physical_bindings, "physical_bindings_digest")

        retained_tasks: dict[str, dict] = {}
        retained_handoffs: dict[str, dict] = {}
        task_results: list[dict] = []
        artifact_replays: list[dict] = []
        consumptions: list[str] = []
        for index, (task, result, receipt, receipt_file) in enumerate(
            zip(tasks, result_rows, receipt_rows, receipt_files)
        ):
            task_id = task["task_id"]
            receipt_sha, receipt_size = receipt_file
            receipt_identity = _stat_identity(Path(task["output"]["receipt_path"]))
            output_identity = _stat_identity(Path(task["output"]["video_path"]))
            receipt_fd = 100 + 2 * index
            output_fd = receipt_fd + 1
            retained_tasks[task_id] = {
                "authority_digest": "",
                "receipt_fd": receipt_fd,
                "output_fd": output_fd,
                "held_through_result_verification": True,
            }
            handoff_authority = _authority_digest(f"handoff-authority-{index}")
            handoff_payload = {
                "schema_version": "full644-exploratory-matched-publication-handoff-payload-v1",
                "task_id": task_id,
                "output_path": result["output_path"],
                "output_identity": output_identity,
                "output_sha256": result["output_sha256"],
                "output_size": result["output_size"],
                "receipt_path": result["receipt_path"],
                "receipt_identity": receipt_identity,
                "receipt_sha256": receipt_sha,
                "receipt_size": receipt_size,
                "receipt_digest": receipt["receipt_digest"],
            }
            retained_handoffs[task_id] = {
                "authority_digest": handoff_authority,
                "fd": 200 + index,
                "payload_digest": postflight.object_sha256(handoff_payload),
                "held_sealed_through_attestation": True,
            }
            publication_authority = {
                "schema_version": "bernini-full644-exploratory-matched-publication-authority-v1",
                "task_id": task_id,
                "output_path": result["output_path"],
                "output_fd": output_fd,
                "output_identity": output_identity,
                "output_sha256": result["output_sha256"],
                "output_size": result["output_size"],
                "receipt_path": result["receipt_path"],
                "receipt_fd": receipt_fd,
                "receipt_identity": receipt_identity,
                "receipt_sha256": receipt_sha,
                "receipt_size": receipt_size,
            }
            retained_tasks[task_id]["authority_digest"] = postflight.object_sha256(
                publication_authority
            )
            authority_artifacts = {
                role: {
                    "basename": f".matched-v2-{index:02d}-{task_id}{suffix}",
                    "sha256": _authority_digest(
                        "artifact-model-capture"
                        if role == "model_capture"
                        else f"artifact-{index}-{role}"
                    ),
                }
                for role, suffix in postflight.AUTHORITY_ARTIFACT_SUFFIXES.items()
            }
            consumption_digest = _authority_digest(f"consumption-{index}")
            consumptions.append(consumption_digest)
            row = {
                "schema_version": "full644-exploratory-matched-runner-task-auh-r5",
                "task_index": index,
                "task_id": task_id,
                "arm": "full644",
                "plan_digest": self.plan["plan_digest"],
                "task_input_digest": receipt["task_input_digest"],
                "argv_digest": _authority_digest(f"argv-{index}"),
                "environment_digest": _authority_digest(f"environment-{index}"),
                "ffmpeg_exec_authority_digest": postflight.object_sha256(exec_rows[3]),
                "publication_handoff_authority_digest": handoff_authority,
                "publication_handoff_payload_digest": retained_handoffs[task_id]["payload_digest"],
                "return_code": 0,
                "attempt_count": 1,
                "retry_allowed": False,
                "model_capture_digest": model_capture,
                "adapter_capture_digest": receipt["model_consumption"]["adapter_capture_digest"],
                "consumption_input_digest": receipt["consumption_input_digest"],
                "consumption_digest": consumption_digest,
                "native_receipt_digest": receipt["receipt_digest"],
                "native_receipt_file_sha256": receipt_sha,
                "native_output_sha256": result["output_sha256"],
                "native_output_size": result["output_size"],
                "native_receipt_identity": receipt_identity,
                "native_output_identity": output_identity,
                "output_path": result["output_path"],
                "receipt_path": result["receipt_path"],
                "log_basename": f".matched-v2-{index:02d}-{task_id}.log",
                "authority_artifacts": authority_artifacts,
                "native_publication_completed_before_parent_post_use_replay": True,
                "parent_post_use_closed_before_native_publication": False,
                "post_use_replay_complete": True,
            }
            _seal(row, "task_result_digest")
            replay_rows = [
                {"role": role, **authority_artifacts[role]}
                for role in sorted(authority_artifacts)
            ]
            replay = {
                "task_id": task_id,
                "artifact_count": 9,
                "artifact_rows_digest": postflight.object_sha256(replay_rows),
                "consumption_digest": consumption_digest,
                "task_result_digest": row["task_result_digest"],
                "runner_task_file_sha256": hashlib.sha256(
                    postflight.canonical_json_bytes(row) + b"\n"
                ).hexdigest(),
                "native_receipt_file_sha256": receipt_sha,
                "native_receipt_mode": 0o400,
                "native_receipt_nlink": 1,
                "native_output_sha256": result["output_sha256"],
                "publication_authority_digest": retained_tasks[task_id]["authority_digest"],
                "publication_handoff_authority_digest": handoff_authority,
                "publication_handoff_payload_digest": retained_handoffs[task_id]["payload_digest"],
                "retained_receipt_and_output_fds_replayed": True,
                "v2_verified_result_cross_linked": True,
                "all_post_use_artifacts_replayed": True,
            }
            task_results.append(row)
            artifact_replays.append(replay)
        model_final = {
            "schema_version": "bernini-action-preservation-model-held-fd-final-v3",
            "model_capture_digest": model_capture,
            "task_count": 5,
            "task_consumption_digests": consumptions,
            "task_consumption_set_digest": postflight.object_sha256(consumptions),
            "final_rehash_digest": _authority_digest("final-rehash"),
            "private_parent_current_identity": _stat_identity(self.output_root),
            "all_model_bytes_rehashed_after_last_task": True,
            "all_model_file_and_directory_fds_retained_through_final_rehash": True,
        }
        _seal(model_final, "model_final_digest")
        self.attestation = {
            "schema_version": "case01-object-trajectory-exact5-runner-attestation-v3",
            "status": "EXACT5_COMPLETE_AWAITING_BLIND_REVIEW",
            "campaign_mode": postflight.trajectory_eval.CAMPAIGN,
            "formal_full16_report": False,
            "manual_blind_review_required": True,
            "plan": {
                "path": str(self.plan_path),
                "sha256": plan_sha,
                "plan_digest": self.plan["plan_digest"],
            },
            "physical_bindings": physical_bindings,
            "captured_runner_entry": {
                "authority_digest": captured_entry["authority_digest"],
                "release_digest": captured_entry["release_digest"],
                "bootstrap_sha256": captured_entry["bootstrap_sha256"],
                "captured_source_entry": True,
                "held_through_attestation_publication": True,
            },
            "retained_publication_root": {
                "path": str(self.output_root),
                "fd": 50,
                "immutable_identity": _directory_identity(self.output_root),
                "held_through_attestation_publication": True,
            },
            "retained_ffprobe_executable": {
                "authority_digest": ffprobe_authority["authority_digest"],
                "fd": ffprobe_authority["fd"],
                "source_path": ffprobe_authority["source_path"],
                "sha256": ffprobe_authority["sha256"],
                "held_through_result_verification": True,
            },
            "retained_task_publications": retained_tasks,
            "retained_child_publication_handoffs": retained_handoffs,
            "retained_final_parents": {
                "output_report": {
                    "path": str(root), "fd": 60,
                    "immutable_identity": _directory_identity(root),
                },
                "runner_attestation": {
                    "path": str(root), "fd": 61,
                    "immutable_identity": _directory_identity(root),
                },
            },
            "task_count": 5,
            "task_ids": list(postflight.TASK_IDS),
            "unselected_task_ids": [],
            "unselected_task_count": 0,
            "all_exact5_tasks_attempted_exactly_once": True,
            "all_exact5_tasks_succeeded": True,
            "retry_count": 0,
            "task_result_digests": [row["task_result_digest"] for row in task_results],
            "task_environment_digests": [row["environment_digest"] for row in task_results],
            "ffmpeg_exec_authority_digest": postflight.object_sha256(exec_rows[3]),
            "all_rank0_encoders_used_retained_ffmpeg_executable": True,
            "task_results": task_results,
            "task_artifact_replays": artifact_replays,
            "runner_task_json_replayed_for_all_tasks": True,
            "native_publication_before_parent_post_use_replay": True,
            "all_model_adapter_post_use_replays_complete": True,
            "native_receipts_replayed_0400_single_link": True,
            "model_capture_digest": model_capture,
            "same_model_capture_all_exact5_tasks": True,
            "model_final": model_final,
            "verified_report": {
                "path": str(self.report_path),
                "sha256": report_sha,
                "report_digest": self.report["report_digest"],
                "verified_task_count": 5,
            },
            "reused_frozen_execution_contract": {
                "frozen_runner_sha256": postflight.PINNED_PHYSICAL_SHA256["frozen_runner"],
                "retained_model_adapter_fd_closure": True,
                "sealed_publication_handoff": True,
                "four_rank_torchrun": True,
                "post_use_replay": True,
            },
            "exploratory_only": True,
            "scientific_claim_authorized": False,
            "formal_claim_authorized": False,
        }
        _seal(self.attestation, "attestation_digest")
        _write_json(self.attestation_path, self.attestation, 0o444)

    def probe(self, path: Path, _: Path) -> dict:
        return postflight._probe_video(path, self.ffprobe)

    def sheet_builder(
        self,
        video: Path,
        output: Path,
        *,
        ffmpeg: Path,
        input_probe: dict,
    ) -> dict:
        del video
        del ffmpeg
        kind = (
            "source"
            if input_probe == postflight.EXPECTED_SOURCE_PROBE
            else "output"
        )
        cached = self.cached_sheets[kind]
        sha256, size = _write_bytes(output, cached["payload"], 0o444)
        if sha256 != cached["sha256"] or size != cached["size"]:
            raise AssertionError("cached real all81 fixture bytes differ")
        return {
            "path": output,
            "sha256": sha256,
            "size": size,
            "mode": 0o444,
            "sheet_contract": copy.deepcopy(cached["sheet_contract"]),
            "decode_replay": copy.deepcopy(cached["decode_replay"]),
        }


class ObjectTrajectoryExact5PostflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.media_temporary = tempfile.TemporaryDirectory(dir="/tmp")
        media_root = Path(cls.media_temporary.name).resolve()
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            raise AssertionError("real ffmpeg/ffprobe fixtures are required")
        cls.real_ffmpeg = Path(ffmpeg).resolve(strict=True)
        cls.real_ffprobe = Path(ffprobe).resolve(strict=True)
        cls.real_output_template = media_root / "real-output-81.mp4"
        completed = subprocess.run(
            [
                str(cls.real_ffmpeg),
                "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(v3_fixture_helpers.SOURCE),
                "-vf", "scale=480:496:flags=lanczos",
                "-frames:v", "81", "-an", "-c:v", "mpeg4",
                "-q:v", "5", "-movflags", "+faststart",
                str(cls.real_output_template),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            env={"LC_ALL": "C", "LANG": "C"},
        )
        if completed.returncode != 0:
            raise AssertionError(
                "real output fixture render failed: "
                + completed.stderr.decode("utf-8", "replace")[:300]
            )
        cls.real_output_template.chmod(0o444)
        observed = postflight._probe_video(
            cls.real_output_template, cls.real_ffprobe
        )
        if observed != postflight.EXPECTED_OUTPUT_PROBE:
            raise AssertionError(f"real output fixture probe differs: {observed}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.real_output_template.chmod(0o600)
        cls.media_temporary.cleanup()
        super().tearDownClass()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/tmp")
        self.root = Path(self.temporary.name).resolve()
        self.fixture = CompletedRunFixture(
            self.root / "run",
            real_ffmpeg=self.real_ffmpeg,
            real_ffprobe=self.real_ffprobe,
            output_template=self.real_output_template,
        )
        self.ffmpeg_sha = mock.patch.object(
            postflight, "EXPECTED_FFMPEG_SHA256", self.fixture.ffmpeg_sha
        )
        self.ffmpeg_size = mock.patch.object(
            postflight, "EXPECTED_FFMPEG_SIZE", self.fixture.ffmpeg_size
        )
        self.physical_ffmpeg_sha = mock.patch.dict(
            postflight.PINNED_PHYSICAL_SHA256,
            {"ffmpeg": self.fixture.ffmpeg_sha},
        )
        self.physical_ffmpeg_size = mock.patch.dict(
            postflight.PINNED_PHYSICAL_SIZE,
            {"ffmpeg": self.fixture.ffmpeg_size},
        )
        self.ffmpeg_sha.start()
        self.ffmpeg_size.start()
        self.physical_ffmpeg_sha.start()
        self.physical_ffmpeg_size.start()

    def tearDown(self) -> None:
        self.physical_ffmpeg_size.stop()
        self.physical_ffmpeg_sha.stop()
        self.ffmpeg_size.stop()
        self.ffmpeg_sha.stop()
        self.temporary.cleanup()

    def _produce(
        self,
        destination: Path | None = None,
        *,
        ffmpeg: Path | None = None,
    ) -> dict:
        bundle = self.root / "bundle" if destination is None else destination
        return postflight.produce_bundle(
            plan_path=self.fixture.plan_path,
            report_path=self.fixture.report_path,
            attestation_path=self.fixture.attestation_path,
            bundle_root=bundle,
            ffmpeg=self.fixture.ffmpeg if ffmpeg is None else ffmpeg,
            probe=self.fixture.probe,
            sheet_builder=self.fixture.sheet_builder,
        )

    def _validate_bundle(self, root: Path, **kwargs) -> dict:
        return postflight.validate_bundle(
            root, ffmpeg=self.fixture.ffmpeg, **kwargs
        )

    def _build_strict_report(self, **kwargs) -> dict:
        return postflight.build_strict_report(
            ffmpeg=self.fixture.ffmpeg, **kwargs
        )

    def _build_site(self, **kwargs) -> dict:
        return html_builder.build_site(
            ffmpeg=self.fixture.ffmpeg, **kwargs
        )

    def _validate_site(self, root: Path, **kwargs) -> dict:
        return html_builder.validate_site(
            root, ffmpeg=self.fixture.ffmpeg, **kwargs
        )

    @staticmethod
    def _store_bundle_manifest(bundle: dict, manifest: dict) -> None:
        _seal(manifest, "manifest_digest")
        _rewrite_json(bundle["manifest_path"], manifest, 0o400)

    @staticmethod
    def _store_site_manifest(site: dict, manifest: dict) -> None:
        _seal(manifest, "manifest_digest")
        _rewrite_json(site["manifest_path"], manifest, 0o400)

    @staticmethod
    def _refresh_authority(authority: dict, path: Path) -> None:
        payload = path.read_bytes()
        authority["sha256"] = hashlib.sha256(payload).hexdigest()
        authority["size"] = len(payload)

    def _rewrite_bundle_attestation(self, bundle: dict, attestation: dict) -> None:
        _seal(attestation, "attestation_digest")
        path = bundle["runner_attestation_authority"]["absolute_path"]
        sha256, size = _rewrite_json(path, attestation, 0o444)
        manifest = copy.deepcopy(bundle["manifest"])
        manifest["runner_documents"]["attestation"].update(
            {"sha256": sha256, "size": size}
        )
        manifest["runner_documents"]["attestation_digest"] = attestation[
            "attestation_digest"
        ]
        self._store_bundle_manifest(bundle, manifest)

    def _rewrite_bundle_report(self, bundle: dict, report: dict) -> None:
        _seal(report, "report_digest")
        report_path = bundle["runner_report_authority"]["absolute_path"]
        report_sha, report_size = _rewrite_json(report_path, report, 0o444)
        attestation = copy.deepcopy(bundle["runner_attestation"])
        attestation["verified_report"]["sha256"] = report_sha
        attestation["verified_report"]["report_digest"] = report["report_digest"]
        _seal(attestation, "attestation_digest")
        attestation_path = bundle["runner_attestation_authority"]["absolute_path"]
        attestation_sha, attestation_size = _rewrite_json(
            attestation_path, attestation, 0o444
        )
        manifest = copy.deepcopy(bundle["manifest"])
        manifest["runner_documents"]["report"].update(
            {"sha256": report_sha, "size": report_size}
        )
        manifest["runner_documents"]["report_digest"] = report["report_digest"]
        manifest["runner_documents"]["attestation"].update(
            {"sha256": attestation_sha, "size": attestation_size}
        )
        manifest["runner_documents"]["attestation_digest"] = attestation[
            "attestation_digest"
        ]
        self._store_bundle_manifest(bundle, manifest)

    @staticmethod
    def _reseal_task_result(attestation: dict, index: int) -> None:
        row = attestation["task_results"][index]
        _seal(row, "task_result_digest")
        attestation["task_result_digests"][index] = row["task_result_digest"]
        replay = attestation["task_artifact_replays"][index]
        replay["task_result_digest"] = row["task_result_digest"]
        replay["runner_task_file_sha256"] = hashlib.sha256(
            postflight.canonical_json_bytes(row) + b"\n"
        ).hexdigest()

    def _observations(self, bundle: dict, *, primary_identity_failure: bool = False) -> dict:
        arms = [_passing_observation(arm) for arm in postflight.ARM_ORDER]
        if primary_identity_failure:
            row = next(item for item in arms if item["variant"] == postflight.PRIMARY_ARM)
            row["dog_identity"]["identity_switch_observed"] = True
            row["dog_identity"]["first_mismatch_frame"] = 0
        value = {
            "schema_version": postflight.OBSERVATION_SCHEMA,
            "status": postflight.OBSERVATION_STATUS,
            "case_id": postflight.CASE_ID,
            "iid": postflight.IID,
            "instruction": postflight.INSTRUCTION,
            "arm_order": list(postflight.ARM_ORDER),
            "review_method": {
                "reviewer_role": "independent-all81-visual-auditor",
                "review_design": postflight.REVIEW_DESIGN,
                "randomized_arm_aliases_used": False,
                "sealed_alias_key_used": False,
                "all81_sheet_layout": "9x9-row-major",
                "decoded_videos_reviewed": True,
                "structured_after_review": True,
                "automatic_output_tracking": False,
            },
            "evidence_bindings": {
                "postflight_manifest_sha256": bundle["manifest_sha256"],
                "postflight_manifest_digest": bundle["manifest"]["manifest_digest"],
                "runner_report_digest": bundle["runner_report"]["report_digest"],
                "runner_attestation_digest": bundle["runner_attestation"]["attestation_digest"],
                "source_video_sha256": bundle["source_video_authority"]["sha256"],
                "source_all81_sheet_sha256": bundle["source_sheet_authority"]["sha256"],
                "arms": {
                    row["arm"]: {
                        "output_sha256": row["output_authority"]["sha256"],
                        "receipt_sha256": row["receipt_authority"]["sha256"],
                        "all81_sheet_sha256": row["sheet_authority"]["sha256"],
                    }
                    for row in bundle["arms"]
                },
            },
            "claim_limits": {
                "automatic_identity_metric_claimed": False,
                "automatic_source_object_correspondence_claimed": False,
                "learned_object_centric_method_claimed": False,
                "formal_causal_claim_authorized": False,
                "scientific_claim_authorized": False,
            },
            "arms": arms,
        }
        return _seal(value, "observations_digest")

    def test_missing_one_real_output_fails_before_bundle_or_staging_creation(self) -> None:
        missing = Path(self.fixture.plan["tasks"][2]["output"]["video_path"])
        missing.chmod(0o600)
        missing.unlink()
        destination = self.root / "must-not-exist"
        with self.assertRaisesRegex(postflight.PostflightError, "missing bone_only|missing trajectory"):
            self._produce(destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".must-not-exist.staging-*")), [])

    def test_complete_five_arm_run_produces_and_revalidates_exact_bundle(self) -> None:
        bundle = self._produce()
        self.assertEqual(bundle["manifest"]["completion_gates"]["real_output_count"], 5)
        self.assertEqual([row["arm"] for row in bundle["arms"]], list(postflight.ARM_ORDER))
        self.assertEqual(len(list((bundle["root"] / "media").glob("*.mp4"))), 5)
        self.assertEqual(len(list((bundle["root"] / "sheets").glob("*.jpg"))), 6)
        self.assertEqual(
            bundle["manifest"]["render_authority"],
            {
                "ffmpeg": {
                    "sha256": self.fixture.ffmpeg_sha,
                    "size": self.fixture.ffmpeg_size,
                },
                "ffprobe": {
                    "role": "runner-media-probe",
                    "sha256": self.fixture.plan["producer"]["ffprobe_sha256"],
                    "size": self.fixture.plan["producer"]["ffprobe_size"],
                },
                "all81_filtergraph": postflight.ALL81_FILTERGRAPH,
            },
        )
        source_contract = bundle["manifest"]["source"]["all81_sheet"]["sheet_contract"]
        self.assertEqual(source_contract["frame_indices"], list(range(81)))
        self.assertEqual(source_contract["tile_count"], 81)
        self.assertEqual(source_contract["image"], {"width": 1470, "height": 1542})
        for row in bundle["manifest"]["arms"]:
            contract = row["all81_sheet"]["sheet_contract"]
            self.assertEqual(contract["frame_indices"], list(range(81)))
            self.assertEqual(contract["tile_count"], 81)
            self.assertEqual(contract["image"], {"width": 1470, "height": 1524})
        self.assertEqual(
            bundle["manifest"]["evaluation_authority"],
            {
                "trajectory_eval": {
                    "role": "trajectory-plan-report-receipt-evaluator",
                    "sha256": postflight.EXPECTED_TRAJECTORY_EVAL_SHA256,
                    "size": postflight.EXPECTED_TRAJECTORY_EVAL_SIZE,
                },
                "strict_eval": {
                    "role": "strict-source-object-visual-gate-evaluator",
                    "sha256": postflight.EXPECTED_STRICT_EVAL_SHA256,
                    "size": postflight.EXPECTED_STRICT_EVAL_SIZE,
                },
            },
        )
        physical = bundle["runner_attestation"]["physical_bindings"]
        self.assertEqual(
            bundle["plan"]["schema_version"],
            "case01-object-trajectory-exact5-plan-v3",
        )
        self.assertEqual(
            bundle["runner_report"]["schema_version"],
            "case01-object-trajectory-exact5-report-v3",
        )
        self.assertEqual(
            bundle["runner_attestation"]["schema_version"],
            "case01-object-trajectory-exact5-runner-attestation-v3",
        )
        self.assertEqual(
            physical["schema_version"],
            "case01-object-trajectory-exact5-physical-bindings-v3",
        )
        self.assertEqual(
            physical["producer_roles_distinct"],
            {
                "invoked_adapter_source": "r5f_composite_inference_wrapper",
                "object_wrapper_inner_source": "source_loaded_support_only",
                "single_frozen_legacy_module": "base_adapter_infer_lora",
                "composite_inner_and_legacy_hashes_distinct": True,
            },
        )
        self.assertEqual(
            (
                physical["identities"]["runner"]["sha256"],
                physical["identities"]["runner"]["size"],
                physical["identities"]["exact5_eval"]["sha256"],
                physical["identities"]["exact5_eval"]["size"],
            ),
            (
                "02207e64a129444b26adf8bd92307102c4a91e85d2a029fa60030a7e9e6f45c8",
                21_716,
                "cfdfc5fec04243265b6c122649fed9144d89510d17184a77782c0ec0ddc5ed8a",
                116_374,
            ),
        )
        producer = bundle["plan"]["producer"]
        self.assertEqual(
            (
                producer["inference_wrapper_sha256"],
                producer["inference_wrapper_size"],
                producer["object_wrapper_inner_sha256"],
                producer["object_wrapper_inner_size"],
                producer["infer_lora_sha256"],
                producer["infer_lora_size"],
            ),
            (
                "b30bba5c9cd233d412ffd88d8413311e9ffbb79d3ddf69aaf6eb2ee96183b489",
                42_184,
                "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9",
                74_281,
                "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
                177_300,
            ),
        )
        self.assertEqual(
            [row["receipt"]["schema_version"] for row in bundle["arms"]],
            [
                "bernini-r-1p3b-action-lora-inference-receipt-v5",
                "bernini-r-1p3b-case01-object-trajectory-oracle-inference-receipt-v4",
                "bernini-r-1p3b-case01-object-trajectory-oracle-inference-receipt-v4",
                "bernini-r-1p3b-case01-object-trajectory-oracle-inference-receipt-v4",
                "bernini-r-1p3b-action-lora-inference-receipt-v5",
            ],
        )
        for row in bundle["arms"][1:4]:
            self.assertEqual(
                row["receipt"]["object_oracle"]["schema_version"],
                "bernini-case01-object-trajectory-oracle-runtime-v4",
            )
        self._validate_bundle(bundle["root"])

    def test_manifest_fixed_authority_and_exact_type_contracts_are_hostile(self) -> None:
        bundle = self._produce()
        original = copy.deepcopy(bundle["manifest"])
        cases = (
            (
                "retired-postflight-v2",
                lambda value: value.__setitem__(
                    "schema_version",
                    "case01-object-trajectory-exact5-postflight-bundle-v2",
                ),
            ),
            ("plan-role", lambda value: value["runner_documents"]["plan"].__setitem__("role", "runner-report")),
            ("source-cross-type-path", lambda value: value["source"]["video"].__setitem__("path", "media/null_before.mp4")),
            ("receipt-role", lambda value: value["arms"][0]["receipt"].__setitem__("role", "null_before-output")),
            ("completion-count-float", lambda value: value["completion_gates"].__setitem__("real_output_count", 5.0)),
            ("completion-bool-int", lambda value: value["completion_gates"].__setitem__("runner_report_replayed", 1)),
            ("claim-bool-int", lambda value: value["claim_limits"].__setitem__("scientific_claim_authorized", 0)),
            ("render-size-float", lambda value: value["render_authority"]["ffmpeg"].__setitem__("size", float(postflight.EXPECTED_FFMPEG_SIZE))),
            ("eval-size-float", lambda value: value["evaluation_authority"]["trajectory_eval"].__setitem__("size", float(postflight.EXPECTED_TRAJECTORY_EVAL_SIZE))),
            ("strict-eval-sha", lambda value: value["evaluation_authority"]["strict_eval"].__setitem__("sha256", "0" * 64)),
            ("media-width-float", lambda value: value["arms"][0]["media_probe"].__setitem__("width", 480.0)),
            ("sheet-tile-count-float", lambda value: value["arms"][0]["all81_sheet"]["sheet_contract"].__setitem__("tile_count", 81.0)),
            ("decode-count-bool", lambda value: value["arms"][0]["all81_sheet"]["decode_replay"].__setitem__("decoded_frame_count", True)),
            ("decode-render-count-float", lambda value: value["source"]["all81_sheet"]["decode_replay"].__setitem__("deterministic_render_count", 2.0)),
            ("decode-byte-equal-int", lambda value: value["arms"][0]["all81_sheet"]["decode_replay"].__setitem__("deterministic_rerender_byte_equal", 1)),
            ("decode-render-sha", lambda value: value["arms"][0]["all81_sheet"]["decode_replay"].__setitem__("render_sha256", "0" * 64)),
            ("strict-rule", lambda value: value.__setitem__("strict_success_rule", "weaker rule")),
            ("render-extra-key", lambda value: value["render_authority"].__setitem__("unbound", False)),
            ("media-extra-key", lambda value: value["source"]["media_probe"].__setitem__("codec", "h264")),
            ("sheet-extra-key", lambda value: value["source"]["all81_sheet"]["sheet_contract"].__setitem__("extra", 1)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                hostile = copy.deepcopy(original)
                mutate(hostile)
                self._store_bundle_manifest(bundle, hostile)
                with self.assertRaises(postflight.PostflightError):
                    self._validate_bundle(bundle["root"])
        self._store_bundle_manifest(bundle, copy.deepcopy(original))
        self._validate_bundle(bundle["root"])

    def test_copied_json_is_canonical_self_digest_closed(self) -> None:
        bundle = self._produce()
        original_manifest = copy.deepcopy(bundle["manifest"])
        tracked_paths = {
            "plan": bundle["plan_authority"]["absolute_path"],
            "report": bundle["runner_report_authority"]["absolute_path"],
            "attestation": bundle["runner_attestation_authority"]["absolute_path"],
            "receipt": bundle["arms"][0]["receipt_authority"]["absolute_path"],
        }
        original_bytes = {name: path.read_bytes() for name, path in tracked_paths.items()}

        def restore() -> None:
            for name, path in tracked_paths.items():
                _rewrite_bytes(path, original_bytes[name], 0o444)
            self._store_bundle_manifest(bundle, copy.deepcopy(original_manifest))

        for name in ("plan", "attestation", "receipt"):
            with self.subTest(name=f"self-digest-{name}"):
                restore()
                value = json.loads(original_bytes[name])
                digest_field = {
                    "plan": "plan_digest",
                    "attestation": "attestation_digest",
                    "receipt": "receipt_digest",
                }[name]
                value[digest_field] = "0" * 64
                _rewrite_json(tracked_paths[name], value, 0o444)
                manifest = copy.deepcopy(original_manifest)
                authority = {
                    "plan": manifest["runner_documents"]["plan"],
                    "attestation": manifest["runner_documents"]["attestation"],
                    "receipt": manifest["arms"][0]["receipt"],
                }[name]
                self._refresh_authority(authority, tracked_paths[name])
                self._store_bundle_manifest(bundle, manifest)
                with self.assertRaisesRegex(postflight.PostflightError, "digest differs"):
                    self._validate_bundle(bundle["root"])

        restore()
        noncanonical = original_bytes["report"] + b" "
        _rewrite_bytes(tracked_paths["report"], noncanonical, 0o444)
        manifest = copy.deepcopy(original_manifest)
        self._refresh_authority(manifest["runner_documents"]["report"], tracked_paths["report"])
        self._store_bundle_manifest(bundle, manifest)
        with self.assertRaisesRegex(postflight.PostflightError, "not canonical JSON plus LF"):
            self._validate_bundle(bundle["root"])
        restore()
        self._validate_bundle(bundle["root"])

    def test_portable_plan_replays_complete_semantics_without_native_reopen(self) -> None:
        bundle = self._produce()
        real_validator = postflight.trajectory_eval.validate_plan
        captured: dict = {}

        def semantic_replay(value: dict, **kwargs) -> dict:
            captured["value"] = copy.deepcopy(value)
            captured["kwargs"] = dict(kwargs)
            return real_validator(value, **kwargs)

        with mock.patch.object(
            postflight.trajectory_eval,
            "validate_plan",
            side_effect=semantic_replay,
        ):
            self._validate_bundle(bundle["root"])
        self.assertEqual(
            captured["kwargs"],
            {
                "reopen_sources": False,
                "require_fresh_outputs": False,
                "require_launchable": True,
            },
        )
        replayed = captured["value"]
        self.assertEqual(set(replayed), set(bundle["plan"]))
        self.assertEqual(replayed["producer"], bundle["plan"]["producer"])
        self.assertEqual(
            replayed["condition_authorities"],
            bundle["plan"]["condition_authorities"],
        )
        self.assertEqual(
            replayed["admission_authorities"],
            bundle["plan"]["admission_authorities"],
        )
        for task, original in zip(replayed["tasks"], bundle["plan"]["tasks"]):
            stripped = copy.deepcopy(task)
            stripped.pop("output")
            original_stripped = copy.deepcopy(original)
            original_stripped.pop("output")
            self.assertEqual(stripped, original_stripped)
            self.assertEqual(Path(task["output"]["video_path"]).parent, bundle["root"])

        plan = copy.deepcopy(bundle["plan"])
        plan["condition_contract"] = {"hostile": True}
        _seal(plan, "plan_digest")
        plan_path = bundle["plan_authority"]["absolute_path"]
        plan_sha, plan_size = _rewrite_json(plan_path, plan, 0o444)
        manifest = copy.deepcopy(bundle["manifest"])
        manifest["runner_documents"]["plan"].update(
            {"sha256": plan_sha, "size": plan_size}
        )
        self._store_bundle_manifest(bundle, manifest)
        with self.assertRaisesRegex(
            postflight.PostflightError, "portable trajectory plan replay failed"
        ):
            self._validate_bundle(bundle["root"])

    def test_retired_upstream_abis_are_rejected_by_real_v3_replay(self) -> None:
        evaluator = postflight.trajectory_eval
        for old_schema in (
            "case01-object-trajectory-exact5-plan-v1",
            "case01-object-trajectory-exact5-plan-v2",
        ):
            with self.subTest(kind="plan", schema=old_schema):
                hostile = copy.deepcopy(self.fixture.plan)
                hostile["schema_version"] = old_schema
                _seal(hostile, "plan_digest")
                with self.assertRaises(evaluator.ObjectTrajectoryEvalError):
                    evaluator.validate_plan(
                        hostile,
                        reopen_sources=False,
                        require_fresh_outputs=False,
                        require_launchable=True,
                    )

        report_cases = (
            ("schema-v1", "schema_version", "case01-object-trajectory-exact5-report-v1"),
            ("schema-v2", "schema_version", "case01-object-trajectory-exact5-report-v2"),
            ("campaign-v1", "campaign_mode", "case01-object-trajectory-exact5-r64-engineering-oracle"),
            ("campaign-v2", "campaign_mode", "case01-object-trajectory-exact5-r64-engineering-oracle-v2"),
        )
        for name, field, old_value in report_cases:
            with self.subTest(kind="report", name=name):
                hostile = copy.deepcopy(self.fixture.report)
                hostile[field] = old_value
                _seal(hostile, "report_digest")
                with self.assertRaises(postflight.PostflightError):
                    postflight._validate_runner_report(
                        hostile, plan=self.fixture.plan
                    )

        off_receipt = copy.deepcopy(self.fixture.receipts[0])
        off_receipt["schema_version"] = (
            "bernini-r-1p3b-action-lora-inference-receipt-v4"
        )
        _seal(off_receipt, "receipt_digest")
        with self.assertRaises(evaluator.ObjectTrajectoryEvalError):
            evaluator.validate_off_inference_receipt(
                off_receipt, self.fixture.tasks[0], self.fixture.producer
            )

        custom_receipt = copy.deepcopy(self.fixture.receipts[1])
        custom_receipt["schema_version"] = (
            "bernini-r-1p3b-case01-object-trajectory-oracle-inference-receipt-v3"
        )
        _seal(custom_receipt, "receipt_digest")
        with self.assertRaises(evaluator.ObjectTrajectoryEvalError):
            evaluator.validate_custom_inference_receipt(
                custom_receipt, self.fixture.tasks[1], self.fixture.producer
            )

        old_runtime = copy.deepcopy(self.fixture.receipts[1])
        old_runtime["object_oracle"]["schema_version"] = (
            "bernini-case01-object-trajectory-oracle-runtime-v3"
        )
        _seal(old_runtime, "receipt_digest")
        with self.assertRaises(evaluator.ObjectTrajectoryEvalError):
            evaluator.validate_custom_inference_receipt(
                old_runtime, self.fixture.tasks[1], self.fixture.producer
            )

        bundle = self._produce(self.root / "retired-attestation-bundle")
        original = copy.deepcopy(bundle["runner_attestation"])
        attestation_cases = (
            ("attestation-v1", "case01-object-trajectory-exact5-runner-attestation-v1", None),
            ("attestation-v2", "case01-object-trajectory-exact5-runner-attestation-v2", None),
            ("physical-v1", None, "case01-object-trajectory-exact5-physical-bindings-v1"),
            ("physical-v2", None, "case01-object-trajectory-exact5-physical-bindings-v2"),
        )
        for name, top_schema, physical_schema in attestation_cases:
            with self.subTest(kind="attestation", name=name):
                hostile = copy.deepcopy(original)
                if top_schema is not None:
                    hostile["schema_version"] = top_schema
                if physical_schema is not None:
                    hostile["physical_bindings"]["schema_version"] = physical_schema
                    _seal(
                        hostile["physical_bindings"],
                        "physical_bindings_digest",
                    )
                self._rewrite_bundle_attestation(bundle, hostile)
                with self.assertRaises(postflight.PostflightError):
                    self._validate_bundle(bundle["root"])
        self._rewrite_bundle_attestation(bundle, original)
        self._validate_bundle(bundle["root"])

    def test_report_exact9_ffprobe_and_null_envelope_replay_are_hostile(self) -> None:
        cases = (
            (
                "media-int-alias",
                lambda report: report["results"][0]["media_probe"].__setitem__(
                    "width", True
                ),
                "media probe.*integer",
            ),
            (
                "media-extra-key",
                lambda report: report["results"][0]["media_probe"].__setitem__(
                    "codec", "h264"
                ),
                "result closure differs",
            ),
            (
                "ffprobe-plan-mismatch",
                lambda report: report["results"][0]["media_probe"].__setitem__(
                    "ffprobe_sha256", "f" * 64
                ),
                "ffprobe authority differs",
            ),
            (
                "null-envelope-not-from-receipts",
                lambda report: report["null_envelope"].__setitem__(
                    "observed_output_sha256_equal",
                    not report["null_envelope"]["observed_output_sha256_equal"],
                ),
                "null envelope is not recomputed",
            ),
        )
        for index, (name, mutate, error) in enumerate(cases):
            with self.subTest(name=name):
                bundle = self._produce(self.root / f"report-hostile-{index}")
                report = copy.deepcopy(bundle["runner_report"])
                mutate(report)
                self._rewrite_bundle_report(bundle, report)
                with self.assertRaisesRegex(postflight.PostflightError, error):
                    self._validate_bundle(bundle["root"])

    def test_attestation_nested_physical_task_artifact_model_and_frozen_replay_is_hostile(self) -> None:
        def producer_bool_alias(attestation: dict) -> None:
            physical = attestation["physical_bindings"]
            physical["producer_roles_distinct"][
                "wrapper_and_legacy_hashes_distinct"
            ] = 1
            _seal(physical, "physical_bindings_digest")

        def retained_exec_fd_alias(attestation: dict) -> None:
            physical = attestation["physical_bindings"]
            authority = physical["exec_authority"]
            authority["rows"][0]["fd"] = True
            authority["rows_digest"] = postflight.object_sha256(authority["rows"])
            _seal(authority, "binding_digest")
            _seal(physical, "physical_bindings_digest")

        def runner_ffmpeg_size_break(attestation: dict) -> None:
            physical = attestation["physical_bindings"]
            physical["identities"]["ffmpeg"]["size"] += 1
            authority = physical["exec_authority"]
            authority["rows"][3]["identity"]["size"] += 1
            authority["rows_digest"] = postflight.object_sha256(authority["rows"])
            _seal(authority, "binding_digest")
            attestation["ffmpeg_exec_authority_digest"] = postflight.object_sha256(
                authority["rows"][3]
            )
            for index, row in enumerate(attestation["task_results"]):
                row["ffmpeg_exec_authority_digest"] = attestation[
                    "ffmpeg_exec_authority_digest"
                ]
                self._reseal_task_result(attestation, index)
            _seal(physical, "physical_bindings_digest")

        def task_index_alias(attestation: dict) -> None:
            attestation["task_results"][0]["task_index"] = 0.0
            self._reseal_task_result(attestation, 0)

        def artifact_count_alias(attestation: dict) -> None:
            attestation["task_artifact_replays"][0]["artifact_count"] = 9.0

        def model_count_alias(attestation: dict) -> None:
            model = attestation["model_final"]
            model["task_count"] = 5.0
            _seal(model, "model_final_digest")

        def frozen_bool_alias(attestation: dict) -> None:
            attestation["reused_frozen_execution_contract"]["four_rank_torchrun"] = 1

        def adapter_receipt_break(attestation: dict) -> None:
            attestation["task_results"][0]["adapter_capture_digest"] = "e" * 64
            self._reseal_task_result(attestation, 0)

        cases = (
            ("producer-bool-alias", producer_bool_alias, "role separation differs"),
            ("retained-exec-fd-bool", retained_exec_fd_alias, "exec cross-link differs"),
            (
                "runner-ffmpeg-size",
                runner_ffmpeg_size_break,
                "physical source size ffmpeg integer differs",
            ),
            ("task-index-float", task_index_alias, "task result/report/plan closure differs"),
            ("artifact-count-float", artifact_count_alias, "artifact/publication replay differs"),
            ("model-count-float", model_count_alias, "model final task-consumption closure differs"),
            ("frozen-bool-alias", frozen_bool_alias, "frozen execution contract differs"),
            ("adapter-receipt-break", adapter_receipt_break, "receipt/task-result/consumption closure differs"),
        )
        for index, (name, mutate, error) in enumerate(cases):
            with self.subTest(name=name):
                bundle = self._produce(self.root / f"attestation-hostile-{index}")
                attestation = copy.deepcopy(bundle["runner_attestation"])
                mutate(attestation)
                self._rewrite_bundle_attestation(bundle, attestation)
                with self.assertRaisesRegex(postflight.PostflightError, error):
                    self._validate_bundle(bundle["root"])

    def test_task_report_receipt_output_cross_links_and_native_path_reuse_are_hostile(self) -> None:
        bundle = self._produce()
        manifest = copy.deepcopy(bundle["manifest"])
        first = bundle["arms"][0]
        receipt_path = first["receipt_authority"]["absolute_path"]
        receipt = copy.deepcopy(first["receipt"])
        receipt["output"]["path"] = self.fixture.plan["tasks"][1]["output"]["video_path"]
        _seal(receipt, "receipt_digest")
        receipt_sha, receipt_size = _rewrite_json(receipt_path, receipt, 0o444)
        manifest["arms"][0]["receipt"].update(
            {"sha256": receipt_sha, "size": receipt_size}
        )
        manifest["arms"][0]["receipt_digest"] = receipt["receipt_digest"]

        report_path = bundle["runner_report_authority"]["absolute_path"]
        report = copy.deepcopy(bundle["runner_report"])
        report["results"][0]["receipt_file_sha256"] = receipt_sha
        report["results"][0]["receipt_digest"] = receipt["receipt_digest"]
        _seal(report, "report_digest")
        report_sha, report_size = _rewrite_json(report_path, report, 0o444)
        manifest["runner_documents"]["report"].update({"sha256": report_sha, "size": report_size})
        manifest["runner_documents"]["report_digest"] = report["report_digest"]

        attestation_path = bundle["runner_attestation_authority"]["absolute_path"]
        attestation = copy.deepcopy(bundle["runner_attestation"])
        attestation["verified_report"]["sha256"] = report_sha
        attestation["verified_report"]["report_digest"] = report["report_digest"]
        _seal(attestation, "attestation_digest")
        attestation_sha, attestation_size = _rewrite_json(attestation_path, attestation, 0o444)
        manifest["runner_documents"]["attestation"].update(
            {"sha256": attestation_sha, "size": attestation_size}
        )
        manifest["runner_documents"]["attestation_digest"] = attestation["attestation_digest"]
        self._store_bundle_manifest(bundle, manifest)
        with self.assertRaisesRegex(
            postflight.PostflightError, "cross-link differs|closure differs"
        ):
            self._validate_bundle(bundle["root"])

        second_bundle = self._produce(self.root / "native-path-reuse-bundle")
        plan_path = second_bundle["plan_authority"]["absolute_path"]
        plan = copy.deepcopy(second_bundle["plan"])
        plan["tasks"][1]["output"] = copy.deepcopy(plan["tasks"][0]["output"])
        _seal(plan, "plan_digest")
        plan_sha, plan_size = _rewrite_json(plan_path, plan, 0o444)
        second_manifest = copy.deepcopy(second_bundle["manifest"])
        second_manifest["runner_documents"]["plan"].update(
            {"sha256": plan_sha, "size": plan_size}
        )
        self._store_bundle_manifest(second_bundle, second_manifest)
        with self.assertRaises(postflight.PostflightError):
            self._validate_bundle(second_bundle["root"])

    def test_ffmpeg_pin_rejects_changed_bytes_before_staging(self) -> None:
        wrong_ffmpeg = self.root / "wrong-ffmpeg"
        _write_bytes(wrong_ffmpeg, b"wrong-ffmpeg", 0o555)
        destination = self.root / "wrong-ffmpeg-bundle"
        with self.assertRaisesRegex(postflight.PostflightError, "ffmpeg executable SHA-256 differs"):
            self._produce(destination, ffmpeg=wrong_ffmpeg)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".wrong-ffmpeg-bundle.staging-*")), [])

    def test_all81_builder_wrong_contract_or_jpeg_dimensions_is_hostile(self) -> None:
        def wrong_contract(video: Path, output: Path, *, ffmpeg: Path, input_probe: dict) -> dict:
            value = self.fixture.sheet_builder(
                video, output, ffmpeg=ffmpeg, input_probe=input_probe
            )
            value["sheet_contract"] = copy.deepcopy(value["sheet_contract"])
            value["sheet_contract"]["tile_count"] = 80
            return value

        first = self.root / "wrong-contract"
        with self.assertRaisesRegex(postflight.PostflightError, "builder contract differs"):
            postflight.produce_bundle(
                plan_path=self.fixture.plan_path,
                report_path=self.fixture.report_path,
                attestation_path=self.fixture.attestation_path,
                bundle_root=first,
                ffmpeg=self.fixture.ffmpeg,
                probe=self.fixture.probe,
                sheet_builder=wrong_contract,
            )
        self.assertFalse(first.exists())

        def wrong_dimensions(video: Path, output: Path, *, ffmpeg: Path, input_probe: dict) -> dict:
            value = self.fixture.sheet_builder(
                video, output, ffmpeg=ffmpeg, input_probe=input_probe
            )
            contract = value["sheet_contract"]
            payload = _synthetic_scanned_jpeg(
                width=contract["image"]["width"] + 2,
                height=contract["image"]["height"],
            )
            sha256, size = _rewrite_bytes(output, payload, 0o444)
            value.update({"sha256": sha256, "size": size})
            value["decode_replay"]["render_sha256"] = sha256
            return value

        second = self.root / "wrong-dimensions"
        with self.assertRaisesRegex(postflight.PostflightError, "sheet dimensions differ"):
            postflight.produce_bundle(
                plan_path=self.fixture.plan_path,
                report_path=self.fixture.report_path,
                attestation_path=self.fixture.attestation_path,
                bundle_root=second,
                ffmpeg=self.fixture.ffmpeg,
                probe=self.fixture.probe,
                sheet_builder=wrong_dimensions,
            )
        self.assertFalse(second.exists())

    def test_bundle_validator_rejects_resealed_sparse_tile_declaration(self) -> None:
        bundle = self._produce()
        manifest_path = bundle["manifest_path"]
        manifest = copy.deepcopy(bundle["manifest"])
        manifest["source"]["all81_sheet"]["sheet_contract"]["frame_indices"] = [0, 80]
        _seal(manifest, "manifest_digest")
        manifest_path.chmod(0o600)
        _write_json(manifest_path, manifest, 0o400)
        with self.assertRaisesRegex(postflight.PostflightError, "sheet.*coverage differs"):
            self._validate_bundle(bundle["root"])

    def test_bundle_sheet_bytes_must_equal_real_pinned_mp4_rerender(self) -> None:
        bundle = self._produce(self.root / "sheet-byte-hostile-bundle")
        sheet_path = bundle["arms"][0]["sheet_authority"]["absolute_path"]
        original_sheet = sheet_path.read_bytes()
        original_manifest = copy.deepcopy(bundle["manifest"])
        expected_image = original_manifest["arms"][0]["all81_sheet"][
            "sheet_contract"
        ]["image"]

        fake_path = self.root / "same-dimension-fake.jpg"
        completed = subprocess.run(
            [
                str(self.fixture.ffmpeg),
                "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i",
                (
                    "color=c=red:s="
                    f"{expected_image['width']}x{expected_image['height']}:r=1"
                ),
                "-frames:v", "1", "-q:v", "2", str(fake_path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            env={"LC_ALL": "C", "LANG": "C"},
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", "replace"),
        )
        fake_payload = fake_path.read_bytes()
        self.assertEqual(
            postflight._jpeg_dimensions(fake_payload, label="real fake sheet"),
            (expected_image["width"], expected_image["height"]),
        )

        cases = (
            (
                "same-dimension-valid-jpeg",
                fake_payload,
                "deterministic replay SHA-256 differs|bytes differ from pinned-decoder",
            ),
            (
                "no-sos-minimal-sof",
                _minimal_jpeg(**expected_image),
                "lacks a start-of-scan marker",
            ),
            (
                "trailing-junk",
                original_sheet + b"hostile-trailing-junk",
                "EOI/trailing-byte closure differs",
            ),
        )
        for name, payload, error in cases:
            with self.subTest(name=name):
                sha256, size = _rewrite_bytes(sheet_path, payload, 0o444)
                manifest = copy.deepcopy(original_manifest)
                sheet = manifest["arms"][0]["all81_sheet"]
                sheet.update({"sha256": sha256, "size": size})
                sheet["decode_replay"]["render_sha256"] = sha256
                self._store_bundle_manifest(bundle, manifest)
                with self.assertRaisesRegex(postflight.PostflightError, error):
                    self._validate_bundle(bundle["root"])
        _rewrite_bytes(sheet_path, original_sheet, 0o444)
        self._store_bundle_manifest(bundle, original_manifest)
        self._validate_bundle(bundle["root"])

    def test_publication_reservation_never_replaces_existing_empty_directory(self) -> None:
        staging = self.root / "publication-staging"
        staging.mkdir()
        (staging / "member").mkdir()
        destination = self.root / "preexisting-empty"
        destination.mkdir()
        before = destination.stat()
        marker = postflight.build_publication_marker(
            kind="test-directory",
            authority_role="test-authority",
            authority_path="authority.json",
            authority_sha256="a" * 64,
            authority_digest="b" * 64,
        )
        with mock.patch.object(postflight.os.path, "lexists", return_value=False):
            with self.assertRaisesRegex(postflight.PostflightError, "concurrently occupied"):
                postflight._publish_directory_create_only(
                    staging, destination, marker=marker
                )
        after = destination.stat()
        self.assertEqual((before.st_dev, before.st_ino), (after.st_dev, after.st_ino))
        self.assertEqual(list(destination.iterdir()), [])
        self.assertTrue((staging / "member").is_dir())

    def test_publication_marker_and_owned_directory_cleanup_are_fail_closed(self) -> None:
        bundle = self._produce()
        marker_path = bundle["root"] / postflight.PUBLICATION_MARKER_REL
        marker_bytes = marker_path.read_bytes()
        marker_path.chmod(0o600)
        marker_path.unlink()
        with self.assertRaisesRegex(postflight.PostflightError, "name closure differs"):
            self._validate_bundle(bundle["root"])
        _write_bytes(marker_path, marker_bytes, 0o400)

        marker = json.loads(marker_bytes)
        marker["authority"]["role"] = "wrong-authority"
        _seal(marker, "marker_digest")
        _rewrite_json(marker_path, marker, 0o400)
        with self.assertRaisesRegex(
            postflight.PostflightError, "publication marker authority differs"
        ):
            self._validate_bundle(bundle["root"])

        staging = self.root / "failure-staging"
        staging.mkdir()
        _write_bytes(staging / "a", b"a", 0o444)
        _write_bytes(staging / "b", b"b", 0o444)
        destination = self.root / "failed-publication"
        publication_marker = postflight.build_publication_marker(
            kind="test-directory",
            authority_role="test-authority",
            authority_path="authority.json",
            authority_sha256="a" * 64,
            authority_digest="b" * 64,
        )
        real_rename = os.rename
        calls = 0

        def fail_second_rename(source: Path, target: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected publication transfer failure")
            real_rename(source, target)

        with mock.patch.object(postflight.os, "rename", side_effect=fail_second_rename):
            with self.assertRaisesRegex(OSError, "injected publication"):
                postflight._publish_directory_create_only(
                    staging, destination, marker=publication_marker
                )
        self.assertFalse(destination.exists())
        self.assertTrue((staging / "b").is_file())

        owned_directory = self.root / "owned-directory"
        owned_directory.mkdir()
        owned = postflight._owned_identity(owned_directory.lstat())
        displaced = self.root / "displaced-owned-directory"
        os.rename(owned_directory, displaced)
        owned_directory.mkdir()
        sentinel = owned_directory / "sentinel"
        _write_bytes(sentinel, b"replacement", 0o444)
        postflight._cleanup_owned_directory(owned_directory, owned)
        self.assertEqual(sentinel.read_bytes(), b"replacement")

    def test_final_revalidation_failure_removes_owned_bundle_and_site(self) -> None:
        bundle_destination = self.root / "final-revalidation-bundle"
        real_validate_bundle = postflight.validate_bundle

        def fail_only_published_bundle(root, **kwargs):
            result = real_validate_bundle(root, **kwargs)
            if Path(root) == bundle_destination:
                raise postflight.PostflightError(
                    "injected final bundle revalidation failure"
                )
            return result

        with mock.patch.object(
            postflight, "validate_bundle", side_effect=fail_only_published_bundle
        ):
            with self.assertRaisesRegex(
                postflight.PostflightError, "injected final bundle"
            ):
                self._produce(bundle_destination)
        self.assertFalse(os.path.lexists(bundle_destination))
        self.assertEqual(
            list(self.root.glob(f".{bundle_destination.name}.staging-*")), []
        )
        self.assertEqual(
            list(self.root.glob(f".{bundle_destination.name}.cleanup-*")), []
        )

        bundle = self._produce(self.root / "site-input-bundle")
        observations_path = self.root / "final-revalidation-observations.json"
        _write_json(observations_path, self._observations(bundle), 0o444)
        report = self._build_strict_report(
            bundle_root=bundle["root"], observations_path=observations_path
        )
        report_path = self.root / "final-revalidation-report.json"
        _write_json(report_path, report, 0o444)
        site_destination = self.root / "final-revalidation-site"
        real_validate_site = html_builder.validate_site

        def fail_only_published_site(root, **kwargs):
            result = real_validate_site(root, **kwargs)
            if Path(root) == site_destination:
                raise html_builder.SiteBuildError(
                    "injected final site revalidation failure"
                )
            return result

        with mock.patch.object(
            html_builder, "validate_site", side_effect=fail_only_published_site
        ):
            with self.assertRaisesRegex(
                html_builder.SiteBuildError, "injected final site"
            ):
                self._build_site(
                    bundle_root=bundle["root"],
                    observations_path=observations_path,
                    strict_report_path=report_path,
                    site_root=site_destination,
                )
        self.assertFalse(os.path.lexists(site_destination))
        self.assertEqual(
            list(self.root.glob(f".{site_destination.name}.staging-*")), []
        )
        self.assertEqual(
            list(self.root.glob(f".{site_destination.name}.cleanup-*")), []
        )

    def test_cleanup_quarantine_never_deletes_racing_replacement(self) -> None:
        real_rename = os.rename
        owned_file = self.root / "toctou-owned-file"
        _write_bytes(owned_file, b"owned", 0o400)
        file_identity = postflight._owned_identity(owned_file.lstat())
        displaced_file = self.root / "toctou-displaced-file"

        def replace_file_during_rename(source, target):
            if Path(source) == owned_file:
                real_rename(owned_file, displaced_file)
                _write_bytes(owned_file, b"replacement", 0o444)
            real_rename(source, target)

        with mock.patch.object(
            postflight.os, "rename", side_effect=replace_file_during_rename
        ):
            postflight._cleanup_owned_file(owned_file, file_identity)
        self.assertEqual(owned_file.read_bytes(), b"replacement")
        self.assertEqual(displaced_file.read_bytes(), b"owned")
        self.assertEqual(list(self.root.glob(".toctou-owned-file.cleanup-*")), [])

        owned_directory = self.root / "toctou-owned-directory"
        owned_directory.mkdir()
        _write_bytes(owned_directory / "owned", b"owned", 0o444)
        directory_identity = postflight._owned_identity(owned_directory.lstat())
        displaced_directory = self.root / "toctou-displaced-directory"

        def replace_directory_during_rename(source, target):
            if Path(source) == owned_directory:
                real_rename(owned_directory, displaced_directory)
                owned_directory.mkdir()
                _write_bytes(
                    owned_directory / "replacement", b"replacement", 0o444
                )
            real_rename(source, target)

        with mock.patch.object(
            postflight.os, "rename", side_effect=replace_directory_during_rename
        ):
            postflight._cleanup_owned_directory(
                owned_directory, directory_identity
            )
        self.assertEqual(
            (owned_directory / "replacement").read_bytes(), b"replacement"
        )
        self.assertEqual(
            (displaced_directory / "owned").read_bytes(), b"owned"
        )
        self.assertEqual(
            list(self.root.glob(".toctou-owned-directory.cleanup-*")), []
        )

    def test_create_only_json_handles_short_writes_replays_and_rejects_symlink_parent(self) -> None:
        value = {"schema_version": "short-write-probe-v1", "payload": "x" * 257}
        _seal(value, "probe_digest")
        output = self.root / "plain-parent" / "receipt.json"
        real_write = os.write

        def short_write(descriptor: int, payload: bytes) -> int:
            return real_write(descriptor, payload[: min(7, len(payload))])

        with mock.patch.object(postflight.os, "write", side_effect=short_write):
            postflight._write_create_only_json(
                output, value, digest_field="probe_digest"
            )
        self.assertEqual(output.stat().st_mode & 0o777, 0o400)
        self.assertEqual(output.read_bytes(), postflight.canonical_json_bytes(value) + b"\n")

        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        refused = linked_parent / "must-not-exist.json"
        with self.assertRaisesRegex(postflight.PostflightError, "plain directory|resolves elsewhere"):
            postflight._write_create_only_json(
                refused, value, digest_field="probe_digest"
            )
        self.assertFalse((real_parent / refused.name).exists())

        deep_refused = linked_parent / "new" / "nested" / "must-not-exist.json"
        with self.assertRaisesRegex(
            postflight.PostflightError, "plain directory|resolves elsewhere"
        ):
            postflight._write_create_only_json(
                deep_refused, value, digest_field="probe_digest"
            )
        self.assertFalse((real_parent / "new").exists())

        zero_progress = self.root / "plain-parent" / "zero-progress.json"
        with mock.patch.object(postflight.os, "write", return_value=0):
            with self.assertRaisesRegex(postflight.PostflightError, "no progress"):
                postflight._write_create_only_json(
                    zero_progress, value, digest_field="probe_digest"
                )
        self.assertFalse(zero_progress.exists())

        replacement_target = self.root / "plain-parent" / "replacement.json"
        displaced_target = self.root / "plain-parent" / "displaced.json"
        real_stable_file = postflight._stable_file

        def replace_before_replay(path: Path, **kwargs):
            if Path(path) == replacement_target:
                os.rename(replacement_target, displaced_target)
                _write_bytes(replacement_target, b"replacement-owned-elsewhere", 0o600)
                raise postflight.PostflightError("injected replay failure")
            return real_stable_file(path, **kwargs)

        with mock.patch.object(
            postflight, "_stable_file", side_effect=replace_before_replay
        ):
            with self.assertRaisesRegex(postflight.PostflightError, "injected replay"):
                postflight._write_create_only_json(
                    replacement_target, value, digest_field="probe_digest"
                )
        self.assertEqual(replacement_target.read_bytes(), b"replacement-owned-elsewhere")
        self.assertTrue(displaced_target.is_file())

    def test_all81_external_writer_failure_cleans_only_owned_output(self) -> None:
        output = self.root / "wrong-sheet.jpg"
        expected = postflight._sheet_contract(postflight.EXPECTED_OUTPUT_PROBE)

        def wrong_sheet_process(arguments, **kwargs):
            del kwargs
            if arguments[-1] == "-":
                rows = "\n".join(
                    f"0, {index}, {index}, 1, 1, " + "0" * 64
                    for index in range(81)
                )
                return subprocess.CompletedProcess(
                    arguments, 0, ("# framehash\n" + rows + "\n").encode(), b""
                )
            _write_bytes(
                Path(arguments[-1]),
                _synthetic_scanned_jpeg(
                    width=expected["image"]["width"] + 2,
                    height=expected["image"]["height"],
                ),
                0o600,
            )
            return subprocess.CompletedProcess(arguments, 0, b"", b"")

        with mock.patch.object(
            postflight.subprocess, "run", side_effect=wrong_sheet_process
        ):
            with self.assertRaisesRegex(postflight.PostflightError, "dimensions differ"):
                postflight.make_all81_sheet(
                    self.fixture.source,
                    output,
                    ffmpeg=self.fixture.ffmpeg,
                    input_probe=postflight.EXPECTED_OUTPUT_PROBE,
                )
        self.assertFalse(output.exists())

    def test_all81_decode_and_rerender_hostiles_leave_zero_residue(self) -> None:
        expected = postflight._sheet_contract(postflight.EXPECTED_OUTPUT_PROBE)

        def framehash(count: int, *, malformed: bool = False) -> bytes:
            rows = []
            for index in range(count):
                digest = "not-a-sha" if malformed and index == 0 else "0" * 64
                rows.append(f"0, {index}, {index}, 1, 1, {digest}")
            return ("# framehash\n" + "\n".join(rows) + "\n").encode()

        for name, count, malformed, expected_error in (
            ("short", 80, False, "frame count differs"),
            ("malformed", 81, True, "framehash sequence differs"),
        ):
            with self.subTest(name=name):
                output = self.root / f"{name}-decode.jpg"

                def decode_only(arguments, **kwargs):
                    del kwargs
                    self.assertEqual(arguments[-1], "-")
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        framehash(count, malformed=malformed),
                        b"",
                    )

                with mock.patch.object(
                    postflight.subprocess, "run", side_effect=decode_only
                ):
                    with self.assertRaisesRegex(
                        postflight.PostflightError, expected_error
                    ):
                        postflight.make_all81_sheet(
                            self.fixture.source,
                            output,
                            ffmpeg=self.fixture.ffmpeg,
                            input_probe=postflight.EXPECTED_OUTPUT_PROBE,
                        )
                self.assertFalse(os.path.lexists(output))
                self.assertEqual(
                    list(self.root.glob(f".{output.name}.render-*")), []
                )

        output = self.root / "nondeterministic-render.jpg"

        def differing_renders(arguments, **kwargs):
            del kwargs
            if arguments[-1] == "-":
                return subprocess.CompletedProcess(
                    arguments, 0, framehash(81), b""
                )
            render_path = Path(arguments[-1])
            suffix = b"first" if render_path.name == "render-1.jpg" else b"second"
            _write_bytes(
                render_path,
                _synthetic_scanned_jpeg(
                    **expected["image"], entropy=suffix
                ),
                0o600,
            )
            return subprocess.CompletedProcess(arguments, 0, b"", b"")

        with mock.patch.object(
            postflight.subprocess, "run", side_effect=differing_renders
        ):
            with self.assertRaisesRegex(
                postflight.PostflightError, "rerender bytes differ"
            ):
                postflight.make_all81_sheet(
                    self.fixture.source,
                    output,
                    ffmpeg=self.fixture.ffmpeg,
                    input_probe=postflight.EXPECTED_OUTPUT_PROBE,
                )
        self.assertFalse(os.path.lexists(output))
        self.assertEqual(
            list(self.root.glob(f".{output.name}.render-*")), []
        )

    def test_observation_timing_text_and_mismatch_contracts_are_hostile(self) -> None:
        bundle = self._produce()
        cases = []

        def case(name: str, mutate, error: str) -> None:
            cases.append((name, mutate, error))

        case(
            "minimum-nine",
            lambda arm: arm["action_trace"].__setitem__("minimum_hold_frames", 9),
            "must equal 10",
        )
        case(
            "minimum-eleven",
            lambda arm: arm["action_trace"].__setitem__("minimum_hold_frames", 11),
            "must equal 10",
        )
        case(
            "hold-not-terminal",
            lambda arm: arm["action_trace"]["stages"][-1].__setitem__("frame_interval", [50, 79]),
            "end at frame 80",
        )
        case(
            "hold-nine-frames",
            lambda arm: arm["action_trace"]["stages"][-1].__setitem__("frame_interval", [72, 80]),
            "at least 10",
        )
        case(
            "switch-without-frame",
            lambda arm: arm["dog_identity"].update(
                {"identity_switch_observed": True, "first_mismatch_frame": None}
            ),
            "mismatch frame/range",
        )
        case(
            "switch-frame-out-of-range",
            lambda arm: arm["dog_identity"].update(
                {"identity_switch_observed": True, "first_mismatch_frame": 81}
            ),
            "mismatch frame/range",
        )
        case(
            "no-switch-with-frame",
            lambda arm: arm["dog_identity"].__setitem__("first_mismatch_frame", 0),
            "frame/flag coupling",
        )
        case(
            "empty-cue-source",
            lambda arm: arm["dog_identity"]["cues"][0].__setitem__("source", "  "),
            "cue source",
        )
        case(
            "empty-cue-output",
            lambda arm: arm["dog_identity"]["cues"][0].__setitem__("output", ""),
            "cue output",
        )
        case(
            "empty-duplicate-description",
            lambda arm: arm["source_bone"]["duplicate_or_substitute_prop"].__setitem__(
                "description", ""
            ),
            "duplicate description",
        )
        case(
            "empty-observed-state",
            lambda arm: arm["source_bone"].__setitem__("observed_state", "\t"),
            "observed state",
        )
        case(
            "empty-stage-evidence",
            lambda arm: arm["action_trace"]["stages"][2].__setitem__("evidence", ""),
            "stage evidence",
        )
        case(
            "coverage-bool-int",
            lambda arm: arm["review_coverage"].__setitem__("all_81_decoded_frames_reviewed", 1),
            "coverage boolean type",
        )
        case(
            "coverage-count-float",
            lambda arm: arm["review_coverage"].__setitem__("frame_count", 81.0),
            "frame count.*integer",
        )
        case(
            "stage-observed-int",
            lambda arm: arm["action_trace"]["stages"][0].__setitem__("observed", 1),
            "stage schema/type",
        )
        case(
            "stage-interval-bool",
            lambda arm: arm["action_trace"]["stages"][0].__setitem__("frame_interval", [False, 15]),
            "interval type/range",
        )
        case(
            "duplicate-observed-int",
            lambda arm: arm["source_bone"]["duplicate_or_substitute_prop"].__setitem__("observed", 0),
            "duplicate observation schema/type",
        )

        for index, (name, mutate, error) in enumerate(cases):
            with self.subTest(name=name):
                observations = self._observations(bundle)
                mutate(observations["arms"][0])
                _seal(observations, "observations_digest")
                path = self.root / f"hostile-observations-{index}.json"
                _write_json(path, observations, 0o444)
                with self.assertRaisesRegex(postflight.PostflightError, error):
                    postflight.validate_observations(bundle, path)

        observations = self._observations(bundle)
        observations["review_method"]["decoded_videos_reviewed"] = 1
        _seal(observations, "observations_digest")
        path = self.root / "hostile-review-method-type.json"
        _write_json(path, observations, 0o444)
        with self.assertRaisesRegex(postflight.PostflightError, "review method.*boolean"):
            postflight.validate_observations(bundle, path)

    def test_unfilled_binding_skeleton_is_honest_and_cannot_be_evaluated(self) -> None:
        bundle = self._produce()
        skeleton = postflight.build_observation_skeleton(
            bundle_root=bundle["root"], ffmpeg=self.fixture.ffmpeg
        )
        self.assertEqual(skeleton["status"], "UNFILLED_REQUIRES_INDEPENDENT_NONBLIND_ALL81_REVIEW")
        self.assertEqual(skeleton["review_method"]["review_design"], "independent_nonblind")
        self.assertIsNone(skeleton["review_method"]["decoded_videos_reviewed"])
        self.assertIsNone(skeleton["arms"][0]["dog_identity"]["identity_switch_observed"])
        self.assertIsNone(skeleton["arms"][0]["action_trace"]["stages"][0]["evidence"])
        self.assertEqual(skeleton["evidence_bindings"], postflight._observation_bindings(bundle))
        path = self.root / "observation-skeleton.json"
        _write_json(path, skeleton, 0o444)
        with self.assertRaisesRegex(postflight.PostflightError, "observations digest differs"):
            postflight.validate_observations(bundle, path)

    def test_strict_primary_gate_cannot_be_compensated_by_other_four_arms(self) -> None:
        bundle = self._produce()
        observations = self._observations(bundle, primary_identity_failure=True)
        observations_path = self.root / "observations.json"
        _write_json(observations_path, observations, 0o444)
        report = self._build_strict_report(
            bundle_root=bundle["root"], observations_path=observations_path
        )
        self.assertEqual(report["review_design"], "independent_nonblind")
        self.assertEqual(report["counts"], {"arm_count": 5, "pass_count": 4, "fail_count": 1})
        self.assertEqual(report["primary_canary_status"], "FAIL")
        primary = next(row for row in report["arms"] if row["variant"] == postflight.PRIMARY_ARM)
        self.assertEqual(primary["gate_statuses"]["dog_identity_retention"], "FAIL")
        self.assertEqual(primary["gate_statuses"]["same_source_bone_reuse"], "PASS")
        self.assertEqual(primary["gate_statuses"]["ordered_source_bone_action"], "PASS")

    def test_offline_html_requires_sealed_observations_and_recomputed_report(self) -> None:
        bundle = self._produce()
        observations = self._observations(bundle)
        observations_path = self.root / "observations.json"
        _write_json(observations_path, observations, 0o444)
        report = self._build_strict_report(
            bundle_root=bundle["root"], observations_path=observations_path
        )
        report_path = self.root / "strict-report.json"
        _write_json(report_path, report, 0o444)
        site = self._build_site(
            bundle_root=bundle["root"],
            observations_path=observations_path,
            strict_report_path=report_path,
            site_root=self.root / "site",
        )
        self.assertTrue(site["index_path"].is_file())
        index = site["index_path"].read_text(encoding="utf-8")
        self.assertIn("independent non-blind", index)
        self.assertIn("not described as blind", index)
        self.assertEqual(site["manifest"]["review_design"], "independent_nonblind")
        for arm in postflight.ARM_ORDER:
            self.assertIn(f"bundle/media/{arm}.mp4", index)
            self.assertIn(f"bundle/sheets/{arm}-all81.jpg", index)
        self.assertEqual(site["manifest"]["placeholder_media_count"], 0)
        self.assertEqual(
            report["observation_authority"]["role"], postflight.OBSERVATION_ROLE
        )
        self.assertNotIn("path", report["observation_authority"])
        self.assertEqual(
            report["postflight_authority"],
            {
                "role": "postflight-manifest",
                "path": postflight.MANIFEST_REL.as_posix(),
                "sha256": bundle["manifest_sha256"],
                "manifest_digest": bundle["manifest"]["manifest_digest"],
            },
        )

        altered = copy.deepcopy(report)
        altered["primary_canary_status"] = "FAIL"
        _seal(altered, "report_digest")
        altered_path = self.root / "altered-report.json"
        _write_json(altered_path, altered, 0o444)
        refused = self.root / "refused-site"
        with self.assertRaisesRegex(html_builder.SiteBuildError, "recomputed"):
            self._build_site(
                bundle_root=bundle["root"],
                observations_path=observations_path,
                strict_report_path=altered_path,
                site_root=refused,
            )
        self.assertFalse(refused.exists())

        numeric_alias = copy.deepcopy(report)
        numeric_alias["counts"]["arm_count"] = 5.0
        _seal(numeric_alias, "report_digest")
        numeric_alias_path = self.root / "numeric-alias-report.json"
        _write_json(numeric_alias_path, numeric_alias, 0o444)
        with self.assertRaisesRegex(html_builder.SiteBuildError, "recomputed"):
            self._build_site(
                bundle_root=bundle["root"],
                observations_path=observations_path,
                strict_report_path=numeric_alias_path,
                site_root=self.root / "numeric-alias-refused-site",
            )

        retired_v2 = copy.deepcopy(report)
        retired_v2["schema_version"] = (
            "case01-object-trajectory-exact5-strict-manual-report-v2"
        )
        _seal(retired_v2, "report_digest")
        retired_v2_path = self.root / "retired-v2-report.json"
        _write_json(retired_v2_path, retired_v2, 0o444)
        with self.assertRaisesRegex(html_builder.SiteBuildError, "recomputed"):
            self._build_site(
                bundle_root=bundle["root"],
                observations_path=observations_path,
                strict_report_path=retired_v2_path,
                site_root=self.root / "retired-v2-report-refused-site",
            )

    def test_site_replays_canonical_evidence_assets_and_exact_html_bytes(self) -> None:
        bundle = self._produce()
        observations_path = self.root / "site-hostile-observations.json"
        observations = self._observations(bundle)
        _write_json(observations_path, observations, 0o444)
        report = self._build_strict_report(
            bundle_root=bundle["root"], observations_path=observations_path
        )
        report_path = self.root / "site-hostile-report.json"
        _write_json(report_path, report, 0o444)
        site = self._build_site(
            bundle_root=bundle["root"],
            observations_path=observations_path,
            strict_report_path=report_path,
            site_root=self.root / "hostile-site",
        )
        root = site["root"]
        original_manifest = copy.deepcopy(site["manifest"])
        tracked = {
            "index.html": (root / "index.html", 0o400),
            ".publication-complete.json": (
                root / ".publication-complete.json", 0o400
            ),
            "bundle/media/null_before.mp4": (root / "bundle/media/null_before.mp4", 0o444),
            "bundle/evidence/postflight-manifest.json": (root / "bundle/evidence/postflight-manifest.json", 0o400),
            "evidence/strict-observations.json": (root / "evidence/strict-observations.json", 0o444),
            "evidence/strict-report.json": (root / "evidence/strict-report.json", 0o444),
        }
        original_bytes = {name: path.read_bytes() for name, (path, _) in tracked.items()}

        def restore() -> dict:
            for name, (path, mode) in tracked.items():
                _rewrite_bytes(path, original_bytes[name], mode)
            restored = copy.deepcopy(original_manifest)
            self._store_site_manifest(site, restored)
            return restored

        manifest = restore()
        index_path, _ = tracked["index.html"]
        _rewrite_bytes(index_path, original_bytes["index.html"] + b"\n<!-- resealed tamper -->", 0o400)
        artifact = next(row for row in manifest["artifacts"] if row["path"] == "index.html")
        self._refresh_authority(artifact, index_path)
        self._store_site_manifest(site, manifest)
        with self.assertRaisesRegex(html_builder.SiteBuildError, "index bytes differ"):
            self._validate_site(root)

        manifest = restore()
        asset_path, _ = tracked["bundle/media/null_before.mp4"]
        _rewrite_bytes(asset_path, original_bytes["bundle/media/null_before.mp4"] + b"tamper", 0o444)
        artifact = next(
            row for row in manifest["artifacts"]
            if row["path"] == "bundle/media/null_before.mp4"
        )
        self._refresh_authority(artifact, asset_path)
        self._store_site_manifest(site, manifest)
        with self.assertRaisesRegex(postflight.PostflightError, "SHA-256 differs"):
            self._validate_site(root)

        manifest = restore()
        obs_path, _ = tracked["evidence/strict-observations.json"]
        _rewrite_bytes(obs_path, original_bytes["evidence/strict-observations.json"] + b" ", 0o444)
        artifact = next(
            row for row in manifest["artifacts"]
            if row["path"] == "evidence/strict-observations.json"
        )
        self._refresh_authority(artifact, obs_path)
        manifest["observations_sha256"] = artifact["sha256"]
        self._store_site_manifest(site, manifest)
        with self.assertRaisesRegex(postflight.PostflightError, "not canonical JSON plus LF"):
            self._validate_site(root)

        manifest = restore()
        copied_postflight_path, _ = tracked["bundle/evidence/postflight-manifest.json"]
        copied_postflight = json.loads(original_bytes["bundle/evidence/postflight-manifest.json"])
        copied_postflight["strict_success_rule"] = "weaker"
        _seal(copied_postflight, "manifest_digest")
        _rewrite_json(copied_postflight_path, copied_postflight, 0o400)
        artifact = next(
            row for row in manifest["artifacts"]
            if row["path"] == "bundle/evidence/postflight-manifest.json"
        )
        self._refresh_authority(artifact, copied_postflight_path)
        manifest["postflight_manifest_sha256"] = artifact["sha256"]
        manifest["postflight_manifest_digest"] = copied_postflight["manifest_digest"]
        self._store_site_manifest(site, manifest)
        with self.assertRaisesRegex(postflight.PostflightError, "identity/rule differs"):
            self._validate_site(root)

        manifest = restore()
        copied_report_path, _ = tracked["evidence/strict-report.json"]
        copied_report = json.loads(original_bytes["evidence/strict-report.json"])
        copied_report["status"] = "RESEALED_BUT_FALSE"
        _seal(copied_report, "report_digest")
        report_sha, _ = _rewrite_json(copied_report_path, copied_report, 0o444)
        artifact = next(
            row for row in manifest["artifacts"]
            if row["path"] == "evidence/strict-report.json"
        )
        self._refresh_authority(artifact, copied_report_path)
        manifest["strict_report_sha256"] = report_sha
        manifest["strict_report_digest"] = copied_report["report_digest"]
        manifest["strict_report_status"] = copied_report["status"]
        self._store_site_manifest(site, manifest)
        with self.assertRaisesRegex(html_builder.SiteBuildError, "not deterministically recomputed"):
            self._validate_site(root)

        manifest = restore()
        manifest["schema_version"] = (
            "case01-object-trajectory-exact5-offline-review-site-v2"
        )
        self._store_site_manifest(site, manifest)
        with self.assertRaisesRegex(html_builder.SiteBuildError, "closure differs"):
            self._validate_site(root)

        manifest = restore()
        manifest["observations_digest"] = "0" * 64
        self._store_site_manifest(site, manifest)
        with self.assertRaisesRegex(html_builder.SiteBuildError, "top-level binding differs"):
            self._validate_site(root)

        manifest = restore()
        manifest["postflight_manifest_status"] = "RESEALED_FALSE_STATUS"
        self._store_site_manifest(site, manifest)
        with self.assertRaisesRegex(html_builder.SiteBuildError, "top-level binding differs"):
            self._validate_site(root)

        manifest = restore()
        manifest["embedded_bundle_file_count"] = True
        self._store_site_manifest(site, manifest)
        with self.assertRaisesRegex(postflight.PostflightError, "integer differs"):
            self._validate_site(root)

        manifest = restore()
        next(row for row in manifest["artifacts"] if row["path"] == "index.html")["role"] = "wrong-role"
        self._store_site_manifest(site, manifest)
        with self.assertRaisesRegex(postflight.PostflightError, "role differs"):
            self._validate_site(root)
        restore()
        self._validate_site(root)

        site_marker_path, _ = tracked[".publication-complete.json"]
        site_marker_path.chmod(0o600)
        site_marker_path.unlink()
        with self.assertRaisesRegex(postflight.PostflightError, "name closure differs"):
            self._validate_site(root)
        _write_bytes(
            site_marker_path,
            original_bytes[".publication-complete.json"],
            0o400,
        )
        self._validate_site(root)

        missing_receipt = root / "bundle/receipts/trajectory_bone_only.receipt.json"
        missing_receipt.chmod(0o600)
        missing_receipt.unlink()
        with self.assertRaisesRegex(postflight.PostflightError, "receipt.*name closure differs"):
            self._validate_site(root)

    def test_real_cli_missing_inputs_fails_before_create_normal_and_optimized(self) -> None:
        for optimize in (False, True):
            prefix = [sys.executable, *(["-O"] if optimize else [])]
            suffix = "optimized" if optimize else "normal"
            bundle_destination = self.root / f"{suffix}-no-bundle"
            skeleton_destination = self.root / f"{suffix}-no-skeleton.json"
            report_destination = self.root / f"{suffix}-no-report.json"
            site_destination = self.root / f"{suffix}-no-site"
            commands = (
                (
                    [
                        *prefix,
                        str(POSTFLIGHT_SCRIPT),
                        "produce",
                        "--plan", str(self.root / "absent-plan.json"),
                        "--report", str(self.root / "absent-report.json"),
                        "--attestation", str(self.root / "absent-attestation.json"),
                        "--bundle", str(bundle_destination),
                        "--ffmpeg", str(self.fixture.ffmpeg),
                    ],
                    bundle_destination,
                ),
                (
                    [*prefix, str(POSTFLIGHT_SCRIPT), "prepare-observations",
                     "--bundle", str(self.root / "absent-bundle"),
                     "--ffmpeg", str(self.fixture.ffmpeg),
                     "--output", str(skeleton_destination)],
                    skeleton_destination,
                ),
                (
                    [*prefix, str(POSTFLIGHT_SCRIPT), "evaluate",
                     "--bundle", str(self.root / "absent-bundle"),
                     "--ffmpeg", str(self.fixture.ffmpeg),
                     "--observations", str(self.root / "absent-observations.json"),
                     "--output", str(report_destination)],
                    report_destination,
                ),
                (
                    [*prefix, str(HTML_SCRIPT), "build",
                     "--bundle", str(self.root / "absent-bundle"),
                     "--ffmpeg", str(self.fixture.ffmpeg),
                     "--observations", str(self.root / "absent-observations.json"),
                     "--strict-report", str(self.root / "absent-strict-report.json"),
                     "--site", str(site_destination)],
                    site_destination,
                ),
            )
            for command, destination in commands:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertIn("EVIDENCE_INVALID", completed.stdout)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
