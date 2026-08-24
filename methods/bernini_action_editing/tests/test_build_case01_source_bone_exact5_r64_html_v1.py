#!/usr/bin/env python3
"""Local, renderer-free tests for the case01 exact5 offline HTML builder."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest


METHOD = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    METHOD / "tools/build_case01_source_bone_exact5_r64_html_v1.py"
)
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures/case01_source_bone_exact5_html_contract_v1.json"
)
SPEC = importlib.util.spec_from_file_location("case01_exact5_html_builder", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def sealed(value: dict, field: str) -> dict:
    result = copy.deepcopy(value)
    result[field] = builder.object_sha256(result)
    return result


def stat_identity(
    *, inode: int, size: int, permissions: int, directory: bool = False,
    nlink: int = 1,
) -> dict[str, int]:
    file_type = stat.S_IFDIR if directory else stat.S_IFREG
    return {
        "device": 48, "inode": inode, "uid": 2012, "gid": 2000,
        "mode": file_type | permissions, "nlink": nlink, "rdev": 0,
        "size": size, "blocks": max(1, (size + 511) // 512),
        "mtime_ns": 1000 + inode, "ctime_ns": 2000 + inode,
    }


def directory_identity(*, inode: int) -> dict[str, int]:
    full = stat_identity(
        inode=inode, size=4096, permissions=0o755, directory=True,
        nlink=2,
    )
    return {
        field: full[field] for field in builder._DIRECTORY_IDENTITY_FIELDS
    }


def pin_identity(
    role: str, *, digest: str, index: int, permissions: int = 0o444,
    size: int | None = None,
) -> dict:
    return {
        "path": f"/release/{role}", "sha256": digest,
        "size": 1000 + index if size is None else size,
        "mode": permissions, "device": 48, "inode": 10_000 + index,
        "uid": 2012, "gid": 2000, "nlink": 1,
    }


def full_from_pin(row: dict, *, permissions: int | None = None) -> dict[str, int]:
    return stat_identity(
        inode=row["inode"], size=row["size"],
        permissions=row["mode"] if permissions is None else permissions,
    )


def make_plan() -> dict:
    sources = []
    for variant in builder.VARIANTS:
        sources.append(
            {
                "variant": variant["id"],
                "path": f"/authority/videos/{variant['id']}.mp4",
                "sha256": variant["source_sha256"],
                "size": variant["source_size"],
                "geometry": dict(builder.EXPECTED_SOURCE_VIDEO),
                "treatment": variant["treatment"],
                "bone_present": variant["bone_present"],
                "bone_position": variant["bone_position"],
                "visual_audit_status": "PASS",
            }
        )
    authority = {
        "schema_version": "case01-source-bone-exact5-asset-authority-v1",
        "status": "APPROVED_FOR_EXACT5_R64_RENDERER_CANARY",
        "launch_allowed": True,
        "independent_visual_audit_status": "PASS_P0_0_P1_0",
        "manifest_path": "/authority/manifest.json",
        "manifest_sha256": "0a62b74056f4be1ab17ed632d31068964aed27c607212f58c2a7d17b74becf5e",
        "manifest_size": 249_082,
        "manifest_digest": "879318860b7d96824ec2da4b10b657b320945285a1607faf8c89bb577a1cc538",
        "independent_audit_receipt_path": "/authority/audit.json",
        "independent_audit_receipt_sha256": "040c53a3647ae957212a1d2d6da3ffa75b4207ace07e1c7ba6ce128033dce969",
        "independent_audit_receipt_size": 8_285,
        "independent_audit_receipt_digest": "13ea77d95e8529585f1bcda1ff5fc9b1f71a42062adfa2994c6dfbe51d22d7d1",
        "iid": builder.IID,
        "sources": sources,
        "source_rows_digest": builder.object_sha256(sources),
    }
    authority = sealed(authority, "authority_digest")
    checkpoint = {
        **builder.EXPECTED_CHECKPOINT,
        "path": "/checkpoint/checkpoint_manifest.json",
    }
    tasks = []
    for index, (variant, task_id) in enumerate(
        zip(builder.VARIANTS, builder.TASK_IDS)
    ):
        output = f"/run/outputs/media/{task_id}.mp4"
        tasks.append(
            {
                "task_id": task_id,
                "case_index": 1,
                "iid": builder.IID,
                "intervention_variant": variant["id"],
                "source_video": sources[index]["path"],
                "source_video_sha256": variant["source_sha256"],
                "instruction": builder.INSTRUCTION,
                "instruction_sha256": builder.INSTRUCTION_SHA256,
                "seed": builder.SEED,
                "num_inference_steps": 40,
                "source_onset_policy": "none",
                "arm": "full644",
                "adapter": {
                    "checkpoint_root": "/checkpoint",
                    "checkpoint_manifest": checkpoint,
                    "adapter_model_sha256": builder.EXPECTED_CHECKPOINT[
                        "adapter_model_sha256"
                    ],
                    "profile": builder.PROFILE,
                },
                "output": {
                    "video_path": output,
                    "receipt_path": output + ".receipt.json",
                    "create_only": True,
                },
            }
        )
    plan = {
        "schema_version": builder.PLAN_SCHEMA,
        "experiment_id": builder.EXPERIMENT_ID,
        "production_ready": True,
        "launch_allowed": True,
        "asset_authority": authority,
        "checkpoint_manifest": checkpoint,
        "producer": {
            **builder.EXPECTED_PRODUCER,
            "infer_lora_path": "/code/infer_lora.py",
            "ffprobe_path": "/runtime/ffprobe",
        },
        "condition_contract": {
            "iid": builder.IID,
            "instruction": builder.INSTRUCTION,
            "instruction_sha256": builder.INSTRUCTION_SHA256,
            "seed": builder.SEED,
            "num_inference_steps": 40,
            "source_onset_policy": "none",
            "same_sampler_all_tasks": True,
            "same_model_capture_all_tasks_required": True,
            "codec_only_present_control_required": True,
        },
        "arms": ["full644"],
        "task_count": 5,
        "tasks": tasks,
        "claim_limits": dict(builder.CLAIM_LIMITS),
    }
    return sealed(plan, "plan_digest")


def make_report(plan: dict) -> dict:
    rows = []
    for index, task_id in enumerate(builder.TASK_IDS):
        output_sha = (
            builder.REFERENCE_OUTPUT_SHA256 if index == 0 else sha(f"output-{index}")
        )
        rows.append(
            {
                "task_id": task_id,
                "arm": "full644",
                "receipt_path": f"/run/outputs/media/{task_id}.mp4.receipt.json",
                "receipt_file_sha256": sha(f"receipt-file-{index}"),
                "receipt_digest": sha(f"receipt-digest-{index}"),
                "output_path": f"/run/outputs/media/{task_id}.mp4",
                "output_sha256": output_sha,
                "output_size": 1000 + index,
                "media_probe": {
                    "ffprobe_path": "/runtime/ffprobe",
                    "ffprobe_sha256": builder.FFPROBE_SHA256,
                    "ffprobe_size": 216_841,
                    "stream_count": 1,
                    "frame_count": 81,
                    "fps_num": 25,
                    "fps_den": 1,
                    "width": 480,
                    "height": 496,
                },
            }
        )
    report = {
        "schema_version": builder.REPORT_SCHEMA,
        "status": "EXACT5_COMPLETE_AWAITING_BLIND_REVIEW",
        "campaign_mode": builder.CAMPAIGN,
        "plan_schema_version": builder.PLAN_SCHEMA,
        "plan_digest": plan["plan_digest"],
        "task_count": 5,
        "task_ids": list(builder.TASK_IDS),
        "variant_order": list(builder.VARIANT_ORDER),
        "all_exact5_tasks_verified_no_cherry_pick": True,
        "same_sampler_all_tasks": True,
        "same_prompt_contract_all_tasks": True,
        "same_model_capture_all_tasks": True,
        "deterministic_reference_parity": {
            "policy": "HARD_FAIL",
            "variant": "exact_original",
            "reference_output_sha256": builder.REFERENCE_OUTPUT_SHA256,
            "observed_output_sha256": builder.REFERENCE_OUTPUT_SHA256,
            "status": "PASS",
            "kept_separate_from_intervention_effect_interpretation": True,
        },
        "codec_only_control_interpretation": (
            "isolates matched transcode/container effects from source-object treatment"
        ),
        "retained_publication_root_fd_replayed": True,
        "retained_ffprobe_executable_fd_replayed": True,
        "retained_publication_leaf_fds_replayed": True,
        "manual_blind_review_required": True,
        "formal_full16_report": False,
        "results": rows,
        "claim_limits": dict(builder.CLAIM_LIMITS),
    }
    return sealed(report, "report_digest")


def make_attestation(
    plan: dict, report: dict, *, plan_sha256: str, report_sha256: str,
    receipt_sizes: list[int],
) -> dict:
    model_capture = sha("shared-model-capture")
    task_results = []
    artifacts = []
    task_digests = []
    environment_digests = []
    retained_tasks = {}
    handoffs = {}
    physical_identities = {}
    for index, role in enumerate(sorted(builder._PHYSICAL_IDENTITY_ROLES)):
        digest = builder._PINNED_PHYSICAL_SHA256.get(role, sha(role))
        permissions = 0o755 if role in {"python", "ffmpeg", "ffprobe"} else 0o444
        size = 216_841 if role == "ffprobe" else None
        physical_identities[role] = pin_identity(
            role, digest=digest, index=index, permissions=permissions, size=size,
        )
    physical_identities["ffprobe"]["path"] = plan["producer"]["ffprobe_path"]
    physical_identities["infer_lora"]["path"] = plan["producer"]["infer_lora_path"]
    entry = {
        "schema_version":
        "full644-exploratory-matched-captured-runner-entry-authority-v1",
        "runner_fd": 30,
        "runner_path": physical_identities["runner"]["path"],
        "runner_sha256": builder.EXACT5_RUNNER_SHA256,
        "runner_identity": full_from_pin(physical_identities["runner"]),
        "python_fd": 31,
        "python_path": physical_identities["python"]["path"],
        "python_sha256": physical_identities["python"]["sha256"],
        "python_identity": full_from_pin(physical_identities["python"]),
        "release_digest": sha("release"),
        "bootstrap_sha256": sha("bootstrap"),
        "entry_method": "slurm-spooled-or-trusted-stdin-held-python-fd-v1",
        "slurm_export_none_required": True,
        "bash_privileged_startup_required": True,
        "captured_source_entry": True,
    }
    entry = sealed(entry, "authority_digest")
    exec_rows = []
    for fd, role, physical_role in zip(
        range(40, 44), builder._EXEC_AUTHORITY_ROLES,
        ("python", "bridge", "adapter", "ffmpeg"),
    ):
        physical = physical_identities[physical_role]
        exec_rows.append({
            "role": role, "fd": fd, "source_path": physical["path"],
            "sha256": physical["sha256"], "identity": full_from_pin(physical),
        })
    exec_authority = {
        "schema_version": "full644-exploratory-matched-exec-authority-v2",
        "rows": exec_rows,
        "rows_digest": builder.object_sha256(exec_rows),
    }
    exec_authority = sealed(exec_authority, "binding_digest")
    ffprobe_physical = physical_identities["ffprobe"]
    ffprobe_authority = {
        "schema_version":
        "bernini-full644-exploratory-matched-ffprobe-exec-authority-v1",
        "fd": 44, "source_path": ffprobe_physical["path"],
        "sha256": ffprobe_physical["sha256"],
        "identity": full_from_pin(ffprobe_physical),
    }
    ffprobe_authority = sealed(ffprobe_authority, "authority_digest")
    raw_values = {
        "SLURM_JOB_ID": "143999", "SLURM_STEP_ID": "1",
        "SLURM_GPUS_ON_NODE": "8", "SLURM_GPUS_PER_NODE": "8",
        "SLURM_STEP_GPUS": "0,1,2,3,4,5,6,7", "SLURM_NNODES": "1",
        "SLURM_STEP_NUM_NODES": "1", "SLURM_JOB_NODELIST": "node293",
        "SLURM_STEP_NODELIST": "node293",
    }
    allocation = {
        "holder_job_id": "143999", "node": "node293", "slurm_step_id": "1",
        "slurm_environment_source_names": {
            "job_id": "SLURM_JOB_ID", "step_id": "SLURM_STEP_ID",
            "gpu_count": "SLURM_GPUS_ON_NODE",
            "gpus_per_node": "SLURM_GPUS_PER_NODE",
            "step_gpu_indices": "SLURM_STEP_GPUS",
            "job_node_count": "SLURM_NNODES",
            "step_node_count": "SLURM_STEP_NUM_NODES",
            "job_nodelist": "SLURM_JOB_NODELIST",
            "step_nodelist": "SLURM_STEP_NODELIST",
        },
        "slurm_environment_raw_values": raw_values,
        "slurm_observed_absent_fields": ["SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"],
        "normalized_slurm_authority": {
            "job_node_count": 1, "step_node_count": 1,
            "gpu_count_on_node": 8, "gpus_per_node": 8,
            "step_gpu_indices": list(range(8)), "job_node": "node293",
            "step_node": "node293",
        },
        "world_size": 4, "ulysses_size": 4, "reserved_gpu_count": 8,
        "visible_gpu_indices": [0, 1, 2, 3],
    }
    final_artifacts = {
        "output_report": f"/run/final/{builder.REPORT_REL.name}",
        "runner_attestation": f"/run/final/{builder.ATTESTATION_REL.name}",
    }
    physical = {
        "schema_version": "case01-source-bone-exact5-physical-bindings-v1",
        "plan_path": f"/run/plan/{builder.PLAN_REL.name}",
        "plan_sha256": plan_sha256, "plan_digest": plan["plan_digest"],
        "asset_authority_digest": plan["asset_authority"]["authority_digest"],
        "allocation": allocation, "identities": physical_identities,
        "captured_runner_entry": entry, "captured_runner_entry_required": True,
        "exec_authority": exec_authority,
        "exec_authority_retained_source_and_python_fds": True,
        "ffprobe_authority": ffprobe_authority,
        "ffprobe_retained_executable_fd": True,
        "isolated_child_interpreters": "-I -S -B",
        "child_environment_exact_allowlist": True,
        "model_root": "/model", "bernini_root": "/release/bernini",
        "veomni_root": "/release/veomni", "campaign_mode": builder.CAMPAIGN,
        "formal_full16_report": False, "task_count": 5,
        "task_ids": list(builder.TASK_IDS), "retry_allowed": False,
        "final_artifacts": final_artifacts,
    }
    physical = sealed(physical, "physical_bindings_digest")
    for index, (task_id, result) in enumerate(
        zip(builder.TASK_IDS, report["results"])
    ):
        environment = sha(f"environment-{index}")
        task_input = builder.object_sha256({
            "schema_version": "full644-exploratory-matched-task-input-v2",
            "plan_digest": plan["plan_digest"], "task": plan["tasks"][index],
        })
        consumption_input = sha(f"consumption-input-{index}")
        consumption = sha(f"consumption-{index}")
        adapter_capture = sha(f"adapter-capture-{index}")
        receipt_identity = stat_identity(
            inode=20_000 + index * 2, size=receipt_sizes[index],
            permissions=0o400,
        )
        output_identity = stat_identity(
            inode=20_001 + index * 2, size=result["output_size"],
            permissions=0o444,
        )
        retained_task = {
            "receipt_fd": 100 + index * 3, "output_fd": 101 + index * 3,
            "held_through_result_verification": True,
        }
        handoff_authority = sha(f"handoff-authority-{index}")
        payload = {
            "schema_version":
            "full644-exploratory-matched-publication-handoff-payload-v1",
            "task_id": task_id, "output_path": result["output_path"],
            "output_identity": output_identity,
            "output_sha256": result["output_sha256"],
            "output_size": result["output_size"],
            "receipt_path": result["receipt_path"],
            "receipt_identity": receipt_identity,
            "receipt_sha256": result["receipt_file_sha256"],
            "receipt_size": receipt_sizes[index],
            "receipt_digest": result["receipt_digest"],
        }
        payload_digest = builder.object_sha256(payload)
        handoffs[task_id] = {
            "authority_digest": handoff_authority, "fd": 102 + index * 3,
            "payload_digest": payload_digest,
            "held_sealed_through_attestation": True,
        }
        prefix = f".matched-v2-{index:02d}-{task_id}"
        suffixes = {
            "model_capture": "-model-capture.json",
            "model_pre_use": "-model-pre-use.json",
            "consumption_input": "-consumption-input.json",
            "adapter_capture": "-adapter-capture.json",
            "adapter_pre_use": "-adapter-pre-use.json",
            "adapter_post_use": "-adapter-post-use.json",
            "adapter_final": "-adapter-final.json",
            "model_post_use": "-model-post-use.json",
            "eval_consumption_chain": "-eval-consumption-chain.json",
        }
        authority_artifacts = {
            role: {
                "basename": prefix + suffix,
                "sha256": (
                    sha("model-capture-file")
                    if role == "model_capture" else sha(f"{role}-{index}")
                ),
            }
            for role, suffix in suffixes.items()
        }
        row = {
            "schema_version": "full644-exploratory-matched-runner-task-auh-r5",
            "task_id": task_id, "task_index": index, "arm": "full644",
            "attempt_count": 1, "return_code": 0, "retry_allowed": False,
            "plan_digest": plan["plan_digest"], "task_input_digest": task_input,
            "argv_digest": sha(f"argv-{index}"),
            "environment_digest": environment,
            "ffmpeg_exec_authority_digest": builder.object_sha256(exec_rows[3]),
            "publication_handoff_authority_digest": handoff_authority,
            "publication_handoff_payload_digest": payload_digest,
            "native_output_sha256": result["output_sha256"],
            "native_output_size": result["output_size"],
            "native_receipt_file_sha256": result["receipt_file_sha256"],
            "native_receipt_digest": result["receipt_digest"],
            "native_receipt_identity": receipt_identity,
            "native_output_identity": output_identity,
            "output_path": result["output_path"],
            "receipt_path": result["receipt_path"],
            "log_basename": prefix + ".log",
            "authority_artifacts": authority_artifacts,
            "native_publication_completed_before_parent_post_use_replay": True,
            "parent_post_use_closed_before_native_publication": False,
            "post_use_replay_complete": True,
            "model_capture_digest": model_capture,
            "adapter_capture_digest": adapter_capture,
            "consumption_input_digest": consumption_input,
            "consumption_digest": consumption,
        }
        row = sealed(row, "task_result_digest")
        publication_authority = {
            "schema_version":
            "bernini-full644-exploratory-matched-publication-authority-v1",
            "task_id": task_id, "output_path": result["output_path"],
            "output_fd": retained_task["output_fd"],
            "output_identity": output_identity,
            "output_sha256": result["output_sha256"],
            "output_size": result["output_size"],
            "receipt_path": result["receipt_path"],
            "receipt_fd": retained_task["receipt_fd"],
            "receipt_identity": receipt_identity,
            "receipt_sha256": result["receipt_file_sha256"],
            "receipt_size": receipt_sizes[index],
        }
        retained_task["authority_digest"] = builder.object_sha256(
            publication_authority
        )
        retained_tasks[task_id] = retained_task
        task_results.append(row)
        task_digests.append(row["task_result_digest"])
        environment_digests.append(environment)
        replay_rows = [
            {
                "role": role, "basename": authority_artifacts[role]["basename"],
                "sha256": authority_artifacts[role]["sha256"],
            }
            for role in sorted(authority_artifacts)
        ]
        artifacts.append(
            {
                "task_id": task_id,
                "task_result_digest": row["task_result_digest"],
                "artifact_count": 9,
                "artifact_rows_digest": builder.object_sha256(replay_rows),
                "consumption_digest": consumption,
                "runner_task_file_sha256": hashlib.sha256(
                    builder.canonical_json_bytes(row) + b"\n"
                ).hexdigest(),
                "native_output_sha256": result["output_sha256"],
                "native_receipt_file_sha256": result["receipt_file_sha256"],
                "native_receipt_mode": 0o400,
                "native_receipt_nlink": 1,
                "publication_authority_digest": retained_task["authority_digest"],
                "publication_handoff_authority_digest": handoff_authority,
                "publication_handoff_payload_digest": payload_digest,
                "retained_receipt_and_output_fds_replayed": True,
                "v2_verified_result_cross_linked": True,
                "all_post_use_artifacts_replayed": True,
            }
        )
    consumptions = [row["consumption_digest"] for row in task_results]
    model_final = {
        "schema_version": "bernini-action-preservation-model-held-fd-final-v3",
        "model_capture_digest": model_capture, "task_count": 5,
        "task_consumption_digests": consumptions,
        "task_consumption_set_digest": builder.object_sha256(consumptions),
        "final_rehash_digest": sha("model-final-rehash"),
        "private_parent_current_identity": stat_identity(
            inode=30_000, size=4096, permissions=0o700, directory=True,
            nlink=2,
        ),
        "all_model_bytes_rehashed_after_last_task": True,
        "all_model_file_and_directory_fds_retained_through_final_rehash": True,
    }
    model_final = sealed(model_final, "model_final_digest")
    attestation = {
        "schema_version": builder.ATTESTATION_SCHEMA,
        "status": "EXACT5_COMPLETE_AWAITING_BLIND_REVIEW",
        "campaign_mode": builder.CAMPAIGN,
        "formal_full16_report": False,
        "manual_blind_review_required": True,
        "plan": {
            "path": physical["plan_path"],
            "sha256": plan_sha256,
            "plan_digest": plan["plan_digest"],
        },
        "physical_bindings": physical,
        "captured_runner_entry": {
            "authority_digest": entry["authority_digest"],
            "release_digest": entry["release_digest"],
            "bootstrap_sha256": entry["bootstrap_sha256"],
            "captured_source_entry": True,
            "held_through_attestation_publication": True,
        },
        "retained_publication_root": {
            "path": "/run/outputs/media", "fd": 50,
            "immutable_identity": directory_identity(inode=40_000),
            "held_through_attestation_publication": True,
        },
        "retained_ffprobe_executable": {
            "authority_digest": ffprobe_authority["authority_digest"],
            "fd": ffprobe_authority["fd"],
            "source_path": ffprobe_authority["source_path"],
            "sha256": builder.FFPROBE_SHA256,
            "held_through_result_verification": True,
        },
        "retained_task_publications": retained_tasks,
        "retained_child_publication_handoffs": handoffs,
        "retained_final_parents": {
            "output_report": {
                "path": "/run/final", "fd": 51,
                "immutable_identity": directory_identity(inode=40_001),
            },
            "runner_attestation": {
                "path": "/run/final", "fd": 52,
                "immutable_identity": directory_identity(inode=40_002),
            },
        },
        "task_count": 5,
        "task_ids": list(builder.TASK_IDS),
        "unselected_task_ids": [],
        "unselected_task_count": 0,
        "all_exact5_tasks_attempted_exactly_once": True,
        "all_exact5_tasks_succeeded": True,
        "retry_count": 0,
        "task_result_digests": task_digests,
        "task_environment_digests": environment_digests,
        "ffmpeg_exec_authority_digest": builder.object_sha256(exec_rows[3]),
        "all_rank0_encoders_used_retained_ffmpeg_executable": True,
        "task_results": task_results,
        "task_artifact_replays": artifacts,
        "runner_task_json_replayed_for_all_tasks": True,
        "native_publication_before_parent_post_use_replay": True,
        "all_model_adapter_post_use_replays_complete": True,
        "native_receipts_replayed_0400_single_link": True,
        "model_capture_digest": model_capture,
        "same_model_capture_all_exact5_tasks": True,
        "model_final": model_final,
        "verified_report": {
            "path": final_artifacts["output_report"],
            "sha256": report_sha256,
            "report_digest": report["report_digest"],
            "verified_task_count": 5,
        },
        "reused_frozen_execution_contract": {
            "frozen_runner_sha256": builder.FROZEN_RUNNER_SHA256,
            "retained_model_adapter_fd_closure": True,
            "sealed_publication_handoff": True,
            "four_rank_torchrun": True,
            "post_use_replay": True,
        },
        "exploratory_only": True,
        "scientific_claim_authorized": False,
        "formal_claim_authorized": False,
    }
    return sealed(attestation, "attestation_digest")


def make_receipt(plan: dict, report: dict, case: dict, index: int) -> dict:
    task = case["task"]
    result = report["results"][index]
    source_authority = {
        "path": task["source_video"],
        "sha256": case["source_sha256"],
        "size": case["source_size"],
        "mode": 420,
        "device": 1,
        "inode": 2 + index,
        "uid": 2012,
        "gid": 2000,
        "nlink": 1,
        "rdev": 0,
        "blocks": 8,
        "mtime_ns": 1,
        "ctime_ns": 1,
    }
    source_digest = builder.object_sha256(source_authority)
    rank_digest = sha("rank-evidence")
    consumption_input = sha(f"consumption-input-{index}")
    task_input = builder.object_sha256({
        "schema_version": "full644-exploratory-matched-task-input-v2",
        "plan_digest": plan["plan_digest"], "task": plan["tasks"][index],
    })
    receipt = {
        "schema_version": builder.RECEIPT_SCHEMA,
        "infer_lora_source_sha256": builder.EXPECTED_PRODUCER["infer_lora_sha256"],
        "method_source_revision": builder.EXPECTED_PRODUCER["method_source_revision"],
        "method_source_archive_sha256": builder.EXPECTED_PRODUCER[
            "method_source_archive_sha256"
        ],
        "bernini_commit": builder.EXPECTED_BERNINI_COMMIT,
        "veomni_commit": builder.EXPECTED_VEOMNI_COMMIT,
        "bernini_inference_files": dict(builder.EXPECTED_BERNINI_INFERENCE_FILES),
        "checkpoint_tree_sha256": builder.EXPECTED_CHECKPOINT_TREE_SHA256,
        "adapter": {
            "enabled": True,
            "mode": "lora_safe_merge",
            "strictly_reloaded": True,
            "safe_merged_for_inference": True,
            "training_global_step": 644,
            "profile": builder.PROFILE,
            "lora_rank": 64,
            "lora_alpha": 64,
            "tensor_count": 480,
            "target_module_count": 240,
            "target_modules_sha256": "d253ba3f11ec5ac26710a829d543a18b939c6f111c64be785264fcd852f3f35a",
            "checkpoint_root": "/proc/self/fd/69",
            "checkpoint_manifest": plan["checkpoint_manifest"],
            "adapter_model_path": "/proc/self/fd/69/adapter/adapter_model.safetensors",
            "adapter_model_sha256": builder.EXPECTED_CHECKPOINT[
                "adapter_model_sha256"
            ],
            "training_receipt_path": "/proc/self/fd/69/receipt.json",
            "training_receipt_digest": builder.EXPECTED_CHECKPOINT["receipt_digest"],
        },
        "input": {
            "source_video_path": task["source_video"],
            "source_video_sha256": case["source_sha256"],
            "instruction_utf8_sha256": builder.INSTRUCTION_SHA256,
            "instruction_utf8_bytes": len(builder.INSTRUCTION.encode("utf-8")),
            "accepted_model_conditions": ["source_video", "edit_instruction"],
            "target_video_argument": False,
            "target_accessed_by_inference": False,
            "external_mask_or_swept_tube": False,
            "external_tracking_pose_or_trajectory": False,
            "reference_image_or_video": False,
            "external_shared_i0": False,
            "source_video_physical_authority": source_authority,
            "source_video_physical_authority_digest": source_digest,
            "retained_source_fd_consumed": True,
            "source_video_pre_and_post_decode_rehashed": True,
        },
        "preprocessing": {
            "frame_count": 81,
            "fps": 25.0,
            "reported_fps": 25.0,
            "source_input_hw": [736, 704],
            "source_derived_bucket_hw": [496, 480],
            "max_pixels": 245_760,
            "stride": 16,
            "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
            "spatial_policy": "sqrt_max_pixels_then_floor_each_dimension_to_stride",
            "resize": "torchvision_bicubic_antialias_true",
            "external_shared_i0": False,
        },
        "prompt_contract": {
            "task": "mv2v",
            "system_prompt_sha256": "12ce75b4360bf5f6d2fdb1e22619438fad6363fd5356634fa698fcb28a83e0ba",
            "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "tokenizer_fix_mistral_regex": True,
            "tokenizer_padding_side": "right",
            "max_sequence_length": 512,
            "prompt_enhancer": False,
        },
        "sampling": {
            "num_frames": 81,
            "num_inference_steps": 40,
            "guidance_mode": "v2v_apg",
            "omega_vid": 1.25,
            "omega_img": 0.0,
            "omega_txt": 4.0,
            "omega_scale": 0.8,
            "flow_shift": 5.0,
            "seed": builder.SEED,
            "eta": 0.5,
            "norm_threshold": [50.0, 50.0],
            "momentum": 0.0,
            "single_expert": "transformer_1",
            "ulysses_size": 4,
            "rank0_decode_and_save_only": True,
            "source_onset_policy": "none",
        },
        "output": {
            "path": task["output"]["video_path"],
            "sha256": result["output_sha256"],
            "frame_count": 81,
            "fps": 25.0,
            "height": 496,
            "width": 480,
            "audio_preserved": False,
            "size": result["output_size"],
            "publication_identity": stat_identity(
                inode=20_001 + index * 2, size=result["output_size"],
                permissions=0o444,
            ),
            "prepublication_identity": stat_identity(
                inode=25_000 + index, size=result["output_size"],
                permissions=0o600, nlink=0,
            ),
            "anonymous_creation_method": "linux-sealed-memfd-v1",
            "anonymous_seal_mask": 15,
            "sealed_source_sha256": result["output_sha256"],
            "sealed_source_size": result["output_size"],
            "anonymous_inode_encoded_and_decoded_before_publication": True,
            "create_only_copy_publication_after_decode": True,
            "sealed_source_and_publication_bytes_equal": True,
            "retained_inode_encoded_and_replayed": True,
            "named_output_never_replaced": True,
        },
        "runtime_versions": {
            "torch": "2.7.1+rocm6.3",
            "torch_hip": "6.3",
            "transformers": "5.5.4",
            "diffusers": "0.38.0",
            "peft": "0.19.1",
        },
        "experimental_inference": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "consumption_input_digest": consumption_input,
        "task_input_digest": task_input,
        "model_consumption": {
            "consumption_input_digest": consumption_input,
            "task_input_digest": task_input,
            "model_capture_digest": sha("shared-model-capture"),
            "model_view_root": "/proc/self/fd/46",
            "adapter_capture_digest": sha(f"adapter-capture-{index}"),
            "adapter_view_root": "/proc/self/fd/69",
            "fd_view_files_authorized": 27,
            "inherited_fd_binding_digest": sha(f"fd-binding-{index}"),
            "inherited_fd_count": 30,
            "ptrace_authorization_used": False,
            "source_video_sha256": case["source_sha256"],
            "source_video_physical_authority_digest": source_digest,
            "all_ranks_use_retained_source_fd": True,
            "four_rank_attestation": {
                "world_size": 4,
                "all_ranks_replayed_exact_fd_views": True,
                "rank_evidence_digest": rank_digest,
                "ordered_rank_evidence_digests": [rank_digest] * 4,
            },
        },
    }
    return sealed(receipt, "receipt_digest")


class Exact5HtmlBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = make_plan()
        report = make_report(self.plan)
        cases = builder.validate_plan(self.plan)
        self.receipts = [
            make_receipt(self.plan, report, case, index)
            for index, case in enumerate(cases)
        ]
        self.receipt_raw = [
            builder.canonical_json_bytes(receipt) + b"\n"
            for receipt in self.receipts
        ]
        for row, receipt, raw in zip(
            report["results"], self.receipts, self.receipt_raw
        ):
            row["receipt_file_sha256"] = hashlib.sha256(raw).hexdigest()
            row["receipt_digest"] = receipt["receipt_digest"]
        self.report = sealed(
            {key: value for key, value in report.items() if key != "report_digest"},
            "report_digest",
        )

    def test_contract_fixture_matches_frozen_constants(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text("utf-8"))
        self.assertEqual(fixture["campaign"], builder.CAMPAIGN)
        self.assertEqual(fixture["instruction"], builder.INSTRUCTION)
        self.assertEqual(fixture["task_ids"], list(builder.TASK_IDS))
        self.assertEqual(fixture["variant_order"], list(builder.VARIANT_ORDER))
        self.assertEqual(fixture["claim_limits"], builder.CLAIM_LIMITS)
        self.assertEqual(
            fixture["reference_exact_original_output_sha256"],
            builder.REFERENCE_OUTPUT_SHA256,
        )
        self.assertEqual(
            fixture["source_sha256"],
            {row["id"]: row["source_sha256"] for row in builder.VARIANTS},
        )
        self.assertEqual(
            fixture["source_size"],
            {row["id"]: row["source_size"] for row in builder.VARIANTS},
        )

    def test_plan_report_attestation_and_receipt_cross_bind(self) -> None:
        cases = builder.validate_plan(self.plan)
        results = builder.validate_report(self.report, self.plan)
        plan_sha = sha("plan-file")
        report_sha = sha("report-file")
        attestation = make_attestation(
            self.plan, self.report, plan_sha256=plan_sha,
            report_sha256=report_sha,
            receipt_sizes=[len(raw) for raw in self.receipt_raw],
        )
        builder.validate_attestation(
            attestation, plan=self.plan, plan_sha256=plan_sha,
            report=self.report, report_sha256=report_sha,
        )
        receipt = self.receipts[0]
        receipt_raw = self.receipt_raw[0]
        receipt_result = copy.deepcopy(results[builder.TASK_IDS[0]])
        coordinates = builder.validate_receipt(
            receipt,
            receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
            result=receipt_result,
            case=cases[0], checkpoint=self.plan["checkpoint_manifest"],
            task_evidence=attestation["task_results"][0],
            receipt_size=len(receipt_raw),
        )
        self.assertEqual(coordinates["model_capture"], sha("shared-model-capture"))

        bad_parity = copy.deepcopy(self.report)
        bad_parity["deterministic_reference_parity"]["status"] = "FAIL"
        bad_parity = sealed(
            {key: value for key, value in bad_parity.items() if key != "report_digest"},
            "report_digest",
        )
        with self.assertRaisesRegex(builder.SiteBuildError, "parity"):
            builder.validate_report(bad_parity, self.plan)

        bad_receipt = copy.deepcopy(receipt)
        bad_receipt["adapter"]["lora_rank"] = 32
        bad_receipt = sealed(
            {key: value for key, value in bad_receipt.items() if key != "receipt_digest"},
            "receipt_digest",
        )
        bad_raw = builder.canonical_json_bytes(bad_receipt) + b"\n"
        bad_result = copy.deepcopy(receipt_result)
        bad_result["receipt_file_sha256"] = hashlib.sha256(bad_raw).hexdigest()
        bad_result["receipt_digest"] = bad_receipt["receipt_digest"]
        with self.assertRaises(builder.SiteBuildError):
            builder.validate_receipt(
                bad_receipt,
                receipt_sha256=hashlib.sha256(bad_raw).hexdigest(),
                result=bad_result, case=cases[0],
                checkpoint=self.plan["checkpoint_manifest"],
                task_evidence=attestation["task_results"][0],
                receipt_size=len(bad_raw),
            )

    def test_missing_output_hard_fails_without_creating_site(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            root = Path(value) / "bundle"
            for directory in (
                root / "plan", root / "final", root / "sources",
                root / "outputs/media",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (root / builder.PLAN_REL).touch()
            (root / builder.REPORT_REL).touch()
            (root / builder.ATTESTATION_REL).touch()
            for variant in builder.VARIANT_ORDER:
                (root / "sources" / f"{variant}.mp4").touch()
            for task_id in builder.TASK_IDS:
                (root / "outputs/media" / f"{task_id}.mp4").touch()
                (root / "outputs/media" / f"{task_id}.mp4.receipt.json").touch()
            missing = root / "outputs/media" / f"{builder.TASK_IDS[-1]}.mp4"
            missing.unlink()
            site = Path(value) / "site"
            with self.assertRaisesRegex(builder.SiteBuildError, "missing"):
                builder.build_site(
                    bundle=root, output=site,
                    ffmpeg=Path("/not-used/ffmpeg"),
                    ffprobe=Path("/not-used/ffprobe"),
                )
            self.assertFalse(site.exists())

    def test_strict_json_rejects_duplicates_nan_and_missing_lf(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            root = Path(value)
            good = root / "good.json"
            good.write_bytes(b'{"a":1}\n')
            loaded, _, _ = builder.load_json(good, label="good")
            self.assertEqual(loaded, {"a": 1})
            for name, raw in (
                ("duplicate", b'{"a":1,"a":1}\n'),
                ("nan", b'{"a":NaN}\n'),
                ("missing-lf", b'{"a":1}'),
            ):
                path = root / f"{name}.json"
                path.write_bytes(raw)
                with self.assertRaises(builder.SiteBuildError, msg=name):
                    builder.load_json(path, label=name)

    def test_render_is_all_relative_accessible_and_claim_limited(self) -> None:
        cases = builder.validate_plan(self.plan)
        for case in cases:
            variant = case["id"]
            task_id = case["task_id"]
            case["media"] = {
                "source": {
                    "basename": f"{variant}-source.mp4",
                    "sha256": case["source_sha256"],
                    "probe": {
                        "frame_count": 81, "fps_num": 25, "fps_den": 1,
                        "width": 704, "height": 736,
                    },
                },
                "result": {
                    "basename": f"{variant}-r64.mp4",
                    "receipt_basename": f"{task_id}.mp4.receipt.json",
                    "sha256": self.report["results"][
                        list(builder.TASK_IDS).index(task_id)
                    ]["output_sha256"],
                    "probe": {
                        "frame_count": 81, "fps_num": 25, "fps_den": 1,
                        "width": 480, "height": 496,
                    },
                },
            }
        rendered = builder.render_html(
            cases, report_sha256=sha("report"),
            attestation_sha256=sha("attestation"),
            build_time="2026-08-21T00:00:00+00:00",
        )
        self.assertEqual(rendered.count('<article class="case"'), 5)
        self.assertIn("非 Full644 训练子集", rendered)
        self.assertIn("非 formal training evaluation", rendered)
        self.assertIn("五组不是五个独立数据集样本", rendered)
        self.assertIn("manual blind review", rendered)
        self.assertIn("assets/sheets/exact_original-all81.jpg", rendered)
        self.assertIn("可见视频同步从头播放", rendered)
        self.assertIn('aria-label="结果筛选与播放控制"', rendered)
        self.assertNotIn("file://", rendered)
        self.assertNotIn("/vast/", rendered)
        self.assertNotIn("/run/", rendered)


if __name__ == "__main__":
    unittest.main()
